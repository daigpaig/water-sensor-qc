"""Evaluation: metrics, fixed-pipeline baseline, and ablation.

Detection precision/recall/F1 per anomaly type + macro-F1, imputation RMSE/MAE
vs a linear-interpolation baseline, decision-quality checks, a no-reasoning
fixed-pipeline baseline, and tool-subset ablation. See CLAUDE.md §10.

**Only the detection metrics are implemented so far.** Imputation error,
decision quality, the fixed-pipeline baseline and the ablation are still to come
(Phase 4, CLAUDE.md §12).

**How a detector's output becomes a *typed* prediction.** The label file names an
anomaly type per row; a detector only says "this row fired". The bridge is
:data:`TOOL_TO_TYPE`, which maps each §7 detector onto the §6 failure type it
exists to catch. Scoring is then four independent binary problems, one per type:
a row is predicted ``spike`` if any spike detector flagged it, and so on. A row
may be predicted as more than one type — these are separate binary problems, not
one multi-class problem, because the detectors genuinely overlap.

``interpolateByRolling`` is deliberately absent from the mapping: it is an
*action*, and the rows it touches were imputed, not detected. Counting them would
score gap detection twice.

**A flag is a candidate; the VERDICT is the answer.** Scoring raw flags is wrong
for an agent: §6 requires it to flag a suspicious excursion, inspect it, and keep
it if it turns out to be real water. Counting that kept row as a false positive
punishes exactly the behaviour the project wants. Since 2026-08-13 the agent says
so outright — every decision carries ``verdict`` (``anomaly``/``normal``) and,
when anomalous, the ``anomaly_type`` it is claiming — and that is what is scored:

    labelled anomaly + called anomaly -> TP   labelled anomaly + called normal -> FN
    real water + called normal        -> TN   real water + called anomaly      -> FP

That last cell is the expensive one (§1: removing real data is worse than leaving
a flagged point in place), and only verdict-aware scoring can see it at all.

The **type** also comes from the agent rather than from :data:`TOOL_TO_TYPE`. A
row ``flagJumps`` found but the agent classified as a spike counts as a spike
claim, because classifying it is the agent's job; grading it on the detector's
guess measures the detector.

Two fallbacks exist, and the table header always says which path ran:

* a flag log written before ``verdict`` existed is scored on the old inference —
  a row counts as claimed only where the agent ``delete``/``correct``/``impute``d
  it (:data:`POSITIVE_ACTIONS`), with the type from ``TOOL_TO_TYPE``. That reads
  an untreatable-but-correctly-identified anomaly as a rejection, which is the
  asymmetry the verdict field was added to remove.
* with no flag log at all, raw flags are scored — detector reach, not agent
  quality.

Two of the four numbers are also not comparable to the others, and the printed
table says so:

* **gap** is nearly free — ``flagNAN`` finds exactly the NaN rows the labels were
  built from, so it scores ~1.0 and pulls macro-F1 up with it.
* **level_shift** is scored per row against a §9 injected *window*, while
  ``flagJumps`` marks a step's *edge*. Row recall therefore cannot be high; that
  is the §9.1 wall seen from the other side, not a tuning failure.

CLI
---
    # score a run's log against the dataset it was run on
    python -m src.evaluate data/injected/02054550/l1/02054550_l1.csv \
        --log logs/run_20260801_131008.jsonl

    # score only the held-out last 20% (§10 splitting)
    python -m src.evaluate <csv> --log <jsonl> --split test

Scoring covers the whole series by default; §10 wants held-out scoring for
reported results, which is what ``--split test`` is for.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, mean_squared_error, mean_absolute_error
import saqc

from src.inspect_data import DATETIME_COL

# Which §6 failure type each §7 detector is trying to catch. A tool absent from
# this mapping contributes to no type's score.
TOOL_TO_TYPE: dict[str, str] = {
    "flagRange": "spike",
    "flagUniLOF": "spike",
    "flagZScore": "spike",
    "flagConstants": "plateau",
    "flagPlateau": "plateau",
    "flagJumps": "level_shift",
    "flagNAN": "gap",
}

# Scored types, in report order. Excludes the empty-string non-anomaly label.
SCORED_TYPES: tuple[str, ...] = ("spike", "plateau", "level_shift", "gap")

# §5 flag-log actions that mean "the agent judged this value wrong". `keep` is
# the agent rejecting its own detector's candidate, and `impute` is what it does
# to a gap it already agreed was a gap — so `impute` counts as acting on it too.
POSITIVE_ACTIONS: frozenset[str] = frozenset({"delete", "correct", "impute"})

TEST_FRACTION = 0.2  # §10: last 20% held out, never shuffled

# Wrapper name (as it appears in a §5 result dict) -> SaQC method, so a run log
# can be scored with the same TOOL_TO_TYPE table a live SaQC object uses.
_WRAPPER_TO_METHOD: dict[str, str] = {
    "flag_range": "flagRange",
    "flag_spike_unilof": "flagUniLOF",
    "flag_zscore": "flagZScore",
    "flag_constants": "flagConstants",
    "flag_plateau": "flagPlateau",
    "flag_jumps": "flagJumps",
    "flag_nan": "flagNAN",
}
_WRAPPER_TO_TYPE: dict[str, str] = {
    wrapper: TOOL_TO_TYPE[method]
    for wrapper, method in _WRAPPER_TO_METHOD.items()
    if method in TOOL_TO_TYPE
}


@dataclass(frozen=True)
class TypeScore:
    """Binary detection scores for one anomaly type."""

    anomaly_type: str
    n_true: int   # labelled rows of this type (EPISODES when by_episode)
    n_pred: int   # rows predicted as this type (EPISODES when by_episode)
    precision: float
    recall: float
    f1: float
    # Scored by episode overlap rather than per row — see EPISODE_TYPES. The table says
    # so per row, because a 1/1 episode recall and a 1/117 row recall are very different
    # claims and must not be read as the same number.
    by_episode: bool = False


@dataclass(frozen=True)
class ImputationScore:
    """Error metrics for imputed values."""
    rmse: float
    mae: float
    baseline_rmse: float
    baseline_mae: float
    n_imputed: int


# --------------------------------------------------------------------------- inputs
def load_labels(series_path: Path) -> pd.DataFrame:
    """Load the §5 labels file that sits beside a dataset CSV."""
    path = series_path.with_name(f"{series_path.stem}_labels.csv")
    if not path.exists():
        raise FileNotFoundError(f"no labels beside {series_path.name} — expected {path}")
    return pd.read_csv(path, parse_dates=[DATETIME_COL])


def predictions_from_qc(qc, field: str = "value") -> dict[str, set]:
    """Typed predictions from a live SaQC object's flag history.

    Reads ``qc._flags.history[field]`` rather than ``qc.flags``: a later test does
    not overwrite an existing flag, so diffing successive flag frames
    under-attributes any row that two tools both caught (§7.1).
    """
    history = qc._flags.history[field]
    out: dict[str, set] = {t: set() for t in SCORED_TYPES}
    for col in history.hist.columns:
        tool = history.meta[col].get("func", col)
        anomaly_type = TOOL_TO_TYPE.get(tool)
        if anomaly_type is None:
            continue  # an action (e.g. interpolateByRolling), not a detector
        out[anomaly_type].update(history.hist.index[history.hist[col] > 0])
    return out


def predictions_from_log(log_path: Path) -> dict[str, set]:
    """Typed predictions from an agent run's JSONL log.

    Every tool result carries its own ``flagged_datetimes`` (§5), so a finished
    run can be scored without re-running it. This re-reads the log rather than
    importing ``src.workbench.visualize_log``, because dependencies point
    ``workbench -> top level`` and never back (§4).
    """
    out: dict[str, set] = {t: set() for t in SCORED_TYPES}
    seen: set[str] = set()

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue  # a run killed mid-write leaves a partial last line
        if event.get("event") != "api_call":
            continue
        for message in event.get("messages") or []:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                # agent.py re-logs the entire history every step, so the same
                # result appears many times; count each tool call once.
                if block["tool_use_id"] in seen:
                    continue
                seen.add(block["tool_use_id"])
                try:
                    result = json.loads(block["content"])
                except (json.JSONDecodeError, TypeError):
                    continue  # an is_error result is plain text, not JSON
                if not isinstance(result, dict):
                    continue
                anomaly_type = _WRAPPER_TO_TYPE.get(str(result.get("tool")))
                stamps = result.get("flagged_datetimes") or []
                if anomaly_type is None or not stamps:
                    continue
                parsed = pd.to_datetime(pd.Series(stamps), errors="coerce").dropna()
                out[anomaly_type].update(parsed)
    return out


def predictions_from_flag_log(path: Path) -> dict[str, set]:
    """Typed predictions from a §5 flag log (``*_flags.json``).

    **Prefer this over :func:`predictions_from_log`.** Since 2026-08-10 a detector
    returns at most ``MAX_FLAGGED_DATETIMES`` timestamps — an evenly-spaced *sample*,
    because the full lists were 34% of the agent's context and its cost is dominated
    by input tokens (§8). The run log therefore no longer contains every flagged row,
    and scoring it undercounts every detector's reach. The flag log does: it carries
    one entry per flagged row, with the ``+``-joined names of the tests that flagged it.

    A row is predicted as every type its flagging tests map to, since one row can be
    flagged by several detectors (§5).
    """
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a §5 flag log (expected a JSON list).")

    out: dict[str, set] = {t: set() for t in SCORED_TYPES}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stamp = pd.to_datetime(entry.get("datetime"), errors="coerce")
        if pd.isna(stamp):
            continue
        for func in str(entry.get("flagged_by", "")).split("+"):
            anomaly_type = TOOL_TO_TYPE.get(func.strip())
            if anomaly_type is not None:
                out[anomaly_type].add(stamp)
    return out


def log_flag_lists_are_truncated(log_path: Path) -> bool:
    """True if any tool result in *log_path* reports a sampled timestamp list.

    The wrappers say so in the result dict (``n_flagged_datetimes_shown`` below
    ``n_flagged``), which is what makes the undercount detectable rather than silent.
    """
    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "api_call":
            continue
        for message in event.get("messages") or []:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not isinstance(block, dict) or block.get("type") != "tool_result":
                    continue
                try:
                    result = json.loads(block["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(result, dict):
                    continue
                shown = result.get("n_flagged_datetimes_shown")
                if shown is not None and shown < (result.get("n_flagged") or 0):
                    return True
    return False


def load_decisions(path: Path) -> dict[pd.Timestamp, str]:
    """Load a §5 flag log (``*_flags.json``) as ``timestamp -> action``.

    The contract is a list of ``{datetime, flagged_by, verdict, anomaly_type,
    action, reason}``. A row appearing twice keeps the stronger action: acting on
    a value beats keeping it, so one detector's ``delete`` is not undone by
    another's ``keep``.

    This reads the *treatment*, which is what :func:`report_decisions` grades.
    Detection scoring reads :func:`load_verdicts` instead.
    """
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a §5 flag log (expected a JSON list).")

    decisions: dict[pd.Timestamp, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stamp = pd.to_datetime(entry.get("datetime"), errors="coerce")
        action = str(entry.get("action", "")).strip().lower()
        if pd.isna(stamp) or not action:
            continue
        if decisions.get(stamp) in POSITIVE_ACTIONS:
            continue  # already acted on; a later `keep` must not override it
        decisions[stamp] = action
    return decisions


def load_verdicts(path: Path) -> dict[pd.Timestamp, str] | None:
    """Load a §5 flag log as ``timestamp -> anomaly_type the agent claimed``.

    **This is the run's answer** (§5, 2026-08-13). The agent states a ``verdict``
    of ``anomaly`` or ``normal`` per segment, and an ``anomaly_type`` with it when
    the verdict is ``anomaly``; the returned mapping holds only the anomalous
    rows, so a timestamp's absence means "not claimed", whether the agent judged
    it normal or never adjudicated it.

    Returns ``None`` for a log written before verdicts existed, so the caller can
    fall back to inferring the claim from the action. Old logs stay scoreable, but
    the two paths are not the same measurement and the printed table says which
    one ran.
    """
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a §5 flag log (expected a JSON list).")

    if not any(isinstance(e, dict) and "verdict" in e for e in entries):
        return None

    verdicts: dict[pd.Timestamp, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        stamp = pd.to_datetime(entry.get("datetime"), errors="coerce")
        if pd.isna(stamp):
            continue
        if str(entry.get("verdict", "")).strip().lower() != "anomaly":
            continue
        anomaly_type = str(entry.get("anomaly_type", "")).strip().lower()
        if anomaly_type in SCORED_TYPES:
            verdicts[stamp] = anomaly_type
    return verdicts


def predictions_from_verdicts(verdicts: dict[pd.Timestamp, str]) -> dict[str, set]:
    """Typed predictions straight from the agent's stated verdicts.

    The type comes from the agent, not from :data:`TOOL_TO_TYPE`. That is the
    point: a row ``flagJumps`` found but the agent classified as a spike is scored
    as a spike claim, because the classification is the agent's job and grading it
    on the detector's guess measures the detector instead.

    A row therefore predicts exactly ONE type here, where flag-derived predictions
    can predict several — the agent commits to one answer per segment.
    """
    out: dict[str, set] = {t: set() for t in SCORED_TYPES}
    for stamp, anomaly_type in verdicts.items():
        out[anomaly_type].add(stamp)
    return out


def apply_decisions(
    predictions: dict[str, set],
    decisions: dict[pd.Timestamp, str],
) -> dict[str, set]:
    """Narrow flagged rows to the ones the agent actually acted on.

    The **fallback** path, for flag logs written before the ``verdict`` field
    existed: there, whether the agent claimed a row was anomalous can only be
    inferred from whether it acted on the value. A flagged row it decided to
    ``keep`` is dropped from the predictions — a candidate considered and
    rejected, scoring as a true negative rather than a false positive — and a row
    with no decision is dropped too, an undecided flag being no claim at all.

    The inference is imperfect in one direction, which is why the verdict field
    replaced it: an anomaly the agent identified correctly but could not treat
    reads here as a rejection. Prefer :func:`predictions_from_verdicts`.
    """
    return {
        anomaly_type: {t for t in stamps if decisions.get(t) in POSITIVE_ACTIONS}
        for anomaly_type, stamps in predictions.items()
    }


# --------------------------------------------------------------------------- scoring
def score(
    predictions: dict[str, set],
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
    decisions: dict[pd.Timestamp, str] | None = None,
) -> tuple[list[TypeScore], float]:
    """Per-type precision / recall / F1 over ``index``, plus macro-F1.

    ``index`` is the set of rows being scored, so restricting it to the held-out
    tail is all a train/test split needs to be.

    Pass ``decisions`` to score what the agent *concluded* rather than what its
    detectors *flagged* — see the module docstring. Without it, every flag counts
    as a positive claim, which understates any agent that inspects and keeps.
    """
    if decisions is not None:
        predictions = apply_decisions(predictions, decisions)

    labelled = labels.set_index(DATETIME_COL)["anomaly_type"].reindex(index).fillna("")

    scores: list[TypeScore] = []
    for anomaly_type in SCORED_TYPES:
        y_true = (labelled == anomaly_type).to_numpy()
        y_pred = index.isin(predictions.get(anomaly_type, set()))

        if anomaly_type in EPISODE_TYPES:
            precision, recall, f1, n_true, n_pred = score_episodes(
                index[y_true], index[y_pred]
            )
            by_episode = True
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average="binary", zero_division=0
            )
            n_true, n_pred = int(y_true.sum()), int(y_pred.sum())
            by_episode = False

        scores.append(
            TypeScore(
                anomaly_type=anomaly_type,
                n_true=n_true,
                n_pred=n_pred,
                precision=float(precision),
                recall=float(recall),
                f1=float(f1),
                by_episode=by_episode,
            )
        )

    macro_f1 = sum(s.f1 for s in scores) / len(scores) if scores else 0.0
    return scores, macro_f1


# Types scored by EPISODE overlap rather than per row. A level shift is a span sitting
# at the wrong level, and per-row scoring measures how much of that span was claimed
# rather than whether the event was found — which for an edge detector is close to a
# category error (§9.1). One correctly identified 10-hour shift and one that missed
# half its window both mean "found it".
EPISODE_TYPES: frozenset[str] = frozenset({"level_shift"})

# Runs of the same type separated by less than this are one event: §9 lets a segment
# anomaly span dropouts, so a single injected shift arrives as several labelled runs
# split by NaN gaps (measured: one shift, five fragments, on 01467200_l2).
EPISODE_BRIDGE = pd.Timedelta("6h")


def _episodes(stamps: pd.DatetimeIndex, bridge: pd.Timedelta = EPISODE_BRIDGE) -> list[tuple]:
    """Contiguous runs of *stamps*, merged across gaps shorter than *bridge*."""
    if len(stamps) == 0:
        return []
    ordered = pd.DatetimeIndex(sorted(stamps))
    spans = [[ordered[0], ordered[0]]]
    for ts in ordered[1:]:
        if ts - spans[-1][1] <= bridge:
            spans[-1][1] = ts
        else:
            spans.append([ts, ts])
    return [(a, b) for a, b in spans]


def score_episodes(truth: pd.DatetimeIndex, predicted: pd.DatetimeIndex) -> tuple:
    """(precision, recall, f1, n_true_episodes, n_pred_episodes) by overlap.

    An episode counts as found if any predicted episode overlaps it at all. That is a
    deliberately generous criterion: the question this answers is "did the run notice
    the event", and the span it wrote is graded separately by the row scores.
    """
    t_eps, p_eps = _episodes(truth), _episodes(predicted)
    overlaps = lambda a, b: a[0] <= b[1] and b[0] <= a[1]
    hit = sum(1 for t in t_eps if any(overlaps(t, p) for p in p_eps))
    useful = sum(1 for p in p_eps if any(overlaps(p, t) for t in t_eps))
    precision = useful / len(p_eps) if p_eps else 0.0
    recall = hit / len(t_eps) if t_eps else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1, len(t_eps), len(p_eps)


def score_imputation(
    clean_series: pd.Series,
    raw_series: pd.Series,
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> ImputationScore | None:
    """Compute RMSE and MAE for imputed gaps vs the known true values,
    compared to a simple linear-interpolation baseline.
    """
    clean = clean_series.reindex(index)
    raw = raw_series.reindex(index)
    lbl = labels.set_index(DATETIME_COL).reindex(index)

    # We only score rows that were synthetically injected (so we have a true_value)
    mask = (lbl["source"] == "injected") & lbl["true_value"].notna()
    if not mask.any():
        return None

    y_true = lbl.loc[mask, "true_value"].astype(float)
    y_pred = clean.loc[mask].astype(float)

    # Compute a simple linear interpolation baseline on the same masked gaps
    baseline_input = raw.copy()
    baseline_input.loc[mask] = np.nan
    baseline_pred = baseline_input.interpolate(method="linear").loc[mask].astype(float)

    # Only score rows where the agent actually imputed a value
    # (if they left it NaN, the error is technically infinite)
    valid = y_pred.notna() & baseline_pred.notna()
    if not valid.any():
        return None

    y_true_v = y_true[valid]
    y_pred_v = y_pred[valid]
    base_pred_v = baseline_pred[valid]

    return ImputationScore(
        rmse=float(mean_squared_error(y_true_v, y_pred_v)) ** 0.5,
        mae=float(mean_absolute_error(y_true_v, y_pred_v)),
        baseline_rmse=float(mean_squared_error(y_true_v, base_pred_v)) ** 0.5,
        baseline_mae=float(mean_absolute_error(y_true_v, base_pred_v)),
        n_imputed=int(valid.sum()),
    )


def baseline_predictions(series: pd.Series) -> dict[str, set]:
    """Run a fixed sequence of SaQC methods as a dumb baseline.
    
    The agent should beat this ruleset. If it doesn't, we need to explain why.
    """
    df = series.to_frame(name="value")
    qc = saqc.SaQC(df)
    
    # Same default parameters as a standard pipeline
    qc = qc.flagRange("value", min=0, max=2000)
    qc = qc.flagUniLOF("value", n=20, thresh=1.5)
    qc = qc.flagConstants("value", window="6h", thresh=0.001)
    qc = qc.flagJumps("value", thresh=5, window="1h")
    qc = qc.flagNAN("value")
    
    return predictions_from_qc(qc)


def format_imputation(score: ImputationScore | None) -> str:
    """Render imputation metrics as a plain-text comparison."""
    if score is None:
        return "Imputation: No labelled synthetic gaps found to score."
    
    lines = [
        "Imputation Error (vs True Value)",
        "-" * 45,
        f"Rows imputed   : {score.n_imputed:>6,}",
        "",
        f"               {'Agent':>10} {'Linear Base':>14}",
        f"RMSE           : {score.rmse:>10.3f} {score.baseline_rmse:>14.3f}",
        f"MAE            : {score.mae:>10.3f} {score.baseline_mae:>14.3f}",
    ]
    return "\n".join(lines)


def report_decisions(
    decisions: dict[pd.Timestamp, str],
    labels: pd.DataFrame,
    index: pd.DatetimeIndex,
) -> str:
    """Provide a detailed breakdown of agent decisions vs known true labels."""
    lbl = labels.set_index(DATETIME_COL).reindex(index)
    
    lines = [
        "Decision Quality Breakdown",
        "-" * 60,
    ]
    
    # We only care about rows the agent acted on (or kept)
    decision_rows = lbl.loc[lbl.index.isin(decisions.keys())].copy()
    if decision_rows.empty:
        return "\n".join(lines) + "\nNo decisions recorded in this slice."
        
    correct_actions = 0
    total_actions = len(decision_rows)
    
    # Tally up
    for ts, row in decision_rows.iterrows():
        action = decisions[ts]
        is_anomaly = bool(row["is_anomaly"])
        anomaly_type = str(row["anomaly_type"]) if is_anomaly else "real water"
        
        # Simple heuristic for correct action:
        # If it's real water, it MUST be kept.
        # If it's an anomaly, it should be deleted/corrected/imputed.
        if not is_anomaly and action == "keep":
            correct_actions += 1
        elif is_anomaly and action in POSITIVE_ACTIONS:
            correct_actions += 1
            
    lines.append(f"Total flags decided: {total_actions:,}")
    lines.append(f"Correct decisions  : {correct_actions:,} ({correct_actions/total_actions*100:.1f}%)")
    lines.append("")
    
    # Add a confusion matrix of (Is Anomaly) x (Action Taken)
    lines.append(f"{'Action':<10} | {'On Anomaly (TP/FN)':<20} | {'On Real Water (FP/TN)':<20}")
    lines.append("-" * 60)
    
    for action in ["delete", "correct", "impute", "keep", "other"]:
        if action == "other":
            mask = ~decision_rows.index.map(lambda t: decisions[t] in ["delete", "correct", "impute", "keep"])
        else:
            mask = decision_rows.index.map(lambda t: decisions[t] == action)
            
        if not mask.any():
            continue
            
        on_anomaly = (mask & (decision_rows["is_anomaly"] == True)).sum()
        on_real = (mask & (decision_rows["is_anomaly"] == False)).sum()
        
        label_a = "TP" if action in POSITIVE_ACTIONS else "FN"
        label_r = "FP" if action in POSITIVE_ACTIONS else "TN"
        
        lines.append(f"{action:<10} | {on_anomaly:>6,} {label_a:<13} | {on_real:>6,} {label_r:<13}")
        
    return "\n".join(lines)


def format_table(
    scores: list[TypeScore],
    macro_f1: float,
    scored_decisions: bool = False,
    scored_verdicts: bool = False,
) -> str:
    """Render the scores as a plain-text table, with the §10 caveats attached."""
    if scored_verdicts:
        mode = (
            "scoring VERDICTS — the agent's own answer: which segments it called\n"
            "         anomalous, and which type it called them. This is the run."
        )
    elif scored_decisions:
        mode = (
            "scoring ACTIONS — no verdicts in this log, so the claim is inferred from\n"
            "         whether the agent acted on the value (pre-2026-08-13 log)"
        )
    else:
        mode = (
            "scoring FLAGS — no decision log, so rows the agent inspected and KEPT\n"
            "         still count against precision (see below)"
        )
    lines = [
        f"mode   : {mode}",
        "",
        f"{'type':<13} {'n_true':>8} {'n_pred':>8} {'prec':>7} {'recall':>7} {'F1':>7}",
        "-" * 55,
    ]
    for s in scores:
        # An episode-scored row is marked, because "1 of 1" and "1 of 117" are not the
        # same claim and the column headings alone cannot tell them apart.
        mark = " *" if s.by_episode else ""
        lines.append(
            f"{s.anomaly_type:<13} {s.n_true:>8,} {s.n_pred:>8,} "
            f"{s.precision:>7.3f} {s.recall:>7.3f} {s.f1:>7.3f}{mark}"
        )
    lines += ["-" * 55, f"{'macro-F1':<13} {macro_f1:>45.3f}"]
    if any(s.by_episode for s in scores):
        marked = ", ".join(s.anomaly_type for s in scores if s.by_episode)
        lines += [
            "",
            f"  * {marked} is scored by EPISODE OVERLAP, so n_true/n_pred are counts of",
            "    events, not rows. A level shift is a span sitting at the wrong level and",
            "    flagJumps flags its EDGES, so per-row recall measures how much of the span",
            "    was claimed rather than whether the event was found (§9.1).",
        ]

    lines += [
        "",
        "Read with care:",
    ]
    if not scored_decisions:
        lines += [
            "  NOT THE      these are detector flags, not the agent's conclusions. A",
            "  agent's      storm peak it flagged, inspected and correctly KEPT still",
            "  answer       counts as a false positive here — so precision measures how",
            "               much the detectors reached for, not how often the agent was",
            "               wrong. Pass --decisions <flags.json> to score properly.",
        ]
    elif not scored_verdicts:
        lines += [
            "  INFERRED     this log predates the `verdict` field, so a row counts as a",
            "  claims       claim only where the agent DELETED / CORRECTED / IMPUTED it.",
            "               An anomaly it identified correctly but chose not to treat",
            "               reads here as a rejection, and the type comes from whichever",
            "               detector fired rather than from the agent. Re-run the agent",
            "               to get verdict-scored numbers.",
        ]
    lines += [
        "  gap          flagNAN finds exactly the NaN rows the labels were built",
        "               from, so this is near-free and inflates macro-F1.",
    ]
    if scored_verdicts:
        lines += [
            "  level_shift  scored by EPISODE, so recall asks \"was the event found\",",
            "               not \"how much of its window was claimed\". Judge span quality",
            "               from the audit page, not from this number.",
        ]
    else:
        lines += [
            "  level_shift  scored by EPISODE overlap — see the note above the table.",
        ]
    absent = [s.anomaly_type for s in scores if s.n_true == 0]
    if absent:
        lines.append(
            f"  UNMEASURABLE {', '.join(absent)} has NO labelled rows in this slice, so its"
            "\n               F1 is 0 by convention, not by performance — and that zero is"
            "\n               still averaged into macro-F1 above. Score these types on a"
            "\n               slice that contains them, or pool levels (§9)."
        )
    thin = [s.anomaly_type for s in scores if 0 < s.n_true < 20]
    if thin:
        lines.append(
            f"  thin         {', '.join(thin)} rest on very few labelled rows; §9 warns"
            "\n               level 1 carries only 1-2 plateau events."
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Score detection against the §5 injected labels (per type + macro-F1)."
    )
    parser.add_argument("series", type=Path, help="Dataset CSV; labels are read from beside it.")
    parser.add_argument(
        "--log", type=Path,
        help="Agent run log (logs/run_*.jsonl) whose flags should be scored.",
    )
    parser.add_argument(
        "--decisions", type=Path,
        help="§5 flag log (*_flags.json). Scores what the agent DECIDED rather than "
             "what its detectors flagged — a flagged row it kept stops counting as a "
             "false positive. Strongly recommended; without it precision is not a "
             "measure of the agent.",
    )
    parser.add_argument(
        "--clean", type=Path,
        help="The agent's output CSV (*_clean.csv). Triggers imputation scoring.",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Compute and compare against a fixed-pipeline SaQC baseline.",
    )
    parser.add_argument(
        "--split", choices=["all", "test"], default="all",
        help="'test' scores only the held-out last 20%% (§10). Default: the whole series.",
    )
    args = parser.parse_args(argv)

    if not args.log and not args.baseline:
        parser.error("Must provide --log to score an agent run, or --baseline to score the baseline.")

    series = pd.read_csv(args.series, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
    index = pd.DatetimeIndex(series[DATETIME_COL])
    if args.split == "test":
        index = index[int(len(index) * (1 - TEST_FRACTION)):]

    labels = load_labels(args.series)
    
    if args.log:
        decisions = load_decisions(args.decisions) if args.decisions else None
        verdicts = load_verdicts(args.decisions) if args.decisions else None
        truncated = log_flag_lists_are_truncated(args.log)

        # The flag log holds every flagged row; the run log holds only the sample the
        # agent was shown (§5). Score the complete source whenever it is available.
        if verdicts is not None:
            # The agent said what each segment IS. Score that, and nothing else.
            predictions = predictions_from_verdicts(verdicts)
            source = f"agent verdicts in {args.decisions.name}"
            scores, macro_f1 = score(predictions, labels, index, decisions=None)
        elif args.decisions:
            predictions = predictions_from_flag_log(args.decisions)
            source = f"flag log ({args.decisions.name}), typed by detector"
            scores, macro_f1 = score(predictions, labels, index, decisions=decisions)
        else:
            predictions = predictions_from_log(args.log)
            source = f"run log ({args.log.name})"
            scores, macro_f1 = score(predictions, labels, index, decisions=None)

        print(f"series : {args.series.name}  ({len(index):,} rows scored, split={args.split})")
        print(f"log    : {args.log.name}")
        print(f"flagged rows read from: {source}")
        if truncated and not args.decisions:
            print(
                "\n  *** WARNING: this run's log carries SAMPLED timestamp lists, so the\n"
                "      numbers below undercount every detector — they are not a measure of\n"
                "      the run. Pass --decisions <flags.json> to score the complete set. ***"
            )
        if decisions is not None:
            print(f"flags  : {args.decisions.name}  ({len(decisions):,} decided rows)")
        if verdicts is not None:
            print(
                f"verdicts: {len(verdicts):,} row(s) the agent CALLED anomalous, of "
                f"{len(decisions or {}):,} flagged"
            )
        print()
        print(format_table(
            scores, macro_f1,
            scored_decisions=decisions is not None,
            scored_verdicts=verdicts is not None,
        ))
        
        if decisions is not None:
            print("\n" + report_decisions(decisions, labels, index))
            
        if args.clean:
            clean_df = pd.read_csv(args.clean, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
            clean_series = clean_df.set_index(DATETIME_COL)["value"]
            raw_series = series.set_index(DATETIME_COL)["value"]
            imp_score = score_imputation(clean_series, raw_series, labels, index)
            print("\n" + format_imputation(imp_score))
            
    if args.baseline:
        print("\n" + "=" * 60)
        print("FIXED-PIPELINE BASELINE")
        print("=" * 60)
        base_preds = baseline_predictions(series.set_index(DATETIME_COL)["value"])
        base_scores, base_f1 = score(base_preds, labels, index, decisions=None)
        print(format_table(base_scores, base_f1, scored_decisions=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
