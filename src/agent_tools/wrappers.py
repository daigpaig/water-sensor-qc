"""SaQC-wrapping tool functions.

Each function wraps a SaQC 2.8 method and returns the tool-result dict defined in
CLAUDE.md §5. Utility (inspect_dataset, get_flag_summary, export_clean_data),
detection (flag_range, flag_constants, flag_plateau, flag_spike_unilof, flag_zscore,
flag_jumps, flag_nan), action (impute_rolling), and context (describe_point,
describe_points -- thin pass-throughs to context.py, see §7.3) tools.

NOTE: verify every SaQC method name/signature against the SaQC 2.8 API before use.

Implemented in Phase 2
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import saqc

from src.agent_tools import context
from src.inspect_data import summarise_series, DATETIME_COL, ANOMALY_TYPES


def _find_nan_runs(series: pd.Series) -> list[dict]:
    """Return one dict per contiguous NaN run in *series*.

    Each dict contains:
      start        -- first NaN timestamp
      end          -- last NaN timestamp
      duration_td  -- pd.Timedelta (end - start; zero for single-row gaps)
      n_rows       -- number of NaN rows in the run
    """
    runs: list[dict] = []
    if not series.isna().any():
        return runs

    is_nan = series.isna()
    # Label each contiguous block of the same boolean
    group_ids = (is_nan != is_nan.shift()).cumsum()
    for _, grp in is_nan.groupby(group_ids):
        if not grp.iloc[0]:          # skip non-NaN blocks
            continue
        start = grp.index[0]
        end   = grp.index[-1]
        runs.append({
            "start":       start,
            "end":         end,
            "duration_td": end - start,
            "n_rows":      len(grp),
        })
    return runs


# How many flagged timestamps a result may carry back to the agent. Measured on
# 03447687_l1 (2026-08-10): the timestamp lists were 34% of a 149k-token request
# — 8,900 stamps across five detectors, re-sent in full on every one of 15 turns,
# and the run's input tokens were 88% of its $4.91 bill. The agent needs these to
# pick points for describe_points, not to read exhaustively, so a sample serves the
# same purpose. The sample is EVENLY SPACED, not the first N: a head would put
# every inspected point in the first weeks of a two-year record.
#
# Raised 250 -> 1000 (2026-08-11) for two measured reasons. The cap cost recall:
# flagUniLOF returned 290 candidates, so 40 were withheld, and 2 of the run's 9
# missed spikes were rows the agent was never shown and therefore could not ask
# about. And the 34% figure above predates prompt caching (§8) — a longer list is
# now written once at 1.25x and read at ~0.1x per turn, so the marginal cost of the
# extra 750 is a few cents across a run. 1000 clears every spike detector's output
# whole (which is what matters: each of those timestamps is an individual decision)
# while still capping flagNAN and flagPlateau, whose thousands of rows are read as
# ranges rather than one at a time.
MAX_FLAGGED_DATETIMES = 1000

# Same reasoning for the per-gap list; a 1,168-gap record does not need to send
# every gap to convey the distribution.
MAX_GAPS_SUMMARY = 40


def _sample_evenly(items: list, limit: int) -> list:
    """Return at most *limit* items, evenly spaced across *items* (endpoints kept)."""
    if len(items) <= limit:
        return items
    idx = np.linspace(0, len(items) - 1, limit).round().astype(int)
    return [items[i] for i in dict.fromkeys(idx.tolist())]


def _build_result(
    tool_name: str,
    params: dict,
    qc_input: saqc.SaQC,
    qc_output: saqc.SaQC,
    field: str,
    custom_msg: str = None
) -> dict:
    """Helper to construct the standardized JSON-serializable tool result."""
    # Note: qc_output.flags is a DictOfSeries, we need to find what was newly flagged
    # by looking at the history.
    history = qc_output._flags.history[field]

    if len(history.hist.columns) > 0:
        # The last column in the history corresponds to the most recently applied test
        last_test = history.hist.columns[-1]

        # Flags in history are floats (UNFLAGGED=-inf, GOOD=0, DOUBTFUL=25, BAD=255)
        # We consider anything > 0 as flagged for detection/actions.
        flagged_mask = history.hist[last_test] > 0

        flagged_datetimes = history.hist.index[flagged_mask].strftime('%Y-%m-%dT%H:%M:%S').tolist()
        n_flagged = len(flagged_datetimes)
        n_total = len(history.hist)
        pct_flagged = round(n_flagged / n_total, 4) if n_total > 0 else 0.0
    else:
        n_flagged = 0
        pct_flagged = 0.0
        flagged_datetimes = []

    # Rows flagged on this field by ANY call so far, not just this one. SaQC never
    # re-flags a row a previous test already flagged, so `n_flagged` above counts only
    # what THIS call added; without the running total, an agent that re-runs a detector
    # with looser parameters cannot tell how much is flagged altogether (§7.1).
    n_flagged_total = int((qc_output.flags[field] > 0).sum())

    msg = custom_msg or f"Flagged {n_flagged} values ({pct_flagged*100:.1f}%) using {tool_name}."
    if custom_msg is None and n_flagged_total != n_flagged:
        msg += (
            f" These are rows not already flagged by an earlier call; "
            f"{n_flagged_total} row(s) are now flagged on '{field}' in total."
        )

    # Truncate loudly, never silently: an agent that thinks it received every flagged
    # timestamp will write decision spans that miss thousands of rows (§5).
    shown = _sample_evenly(flagged_datetimes, MAX_FLAGGED_DATETIMES)
    if len(shown) < n_flagged:
        msg += (
            f" NOTE: showing {len(shown)} of {n_flagged} flagged timestamps, sampled"
            " evenly across the record to keep the context affordable. The counts above"
            " are exact and complete; the LIST is a sample. Use it to choose points to"
            " inspect, and write decision spans that cover whole segments by time range"
            " rather than enumerating the timestamps you were shown."
        )

    return {
        "tool": tool_name,
        "params": params,
        "n_flagged": n_flagged,
        "pct_flagged": pct_flagged,
        "n_flagged_total": n_flagged_total,
        "flagged_datetimes": shown,
        "n_flagged_datetimes_shown": len(shown),
        "message": msg,
        "qc": qc_output  # Keep the qc object for the next steps
    }


def inspect_dataset(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Summarizes the dataset. It looks at the data and counts the rows, missing values,
    and checks the start/end times.
    """
    df = qc.data.to_pandas()
    df = df.reset_index()
    df.rename(columns={"index": DATETIME_COL}, inplace=True)
    summary = summarise_series(df, value_col=field)
    # Where the sensor is noisy, as part of the FIRST thing every run sees (§8).
    # A flagged point is judged against the record-wide scale unless the run knows
    # its neighbourhood is busy — which is how 104 ordinary readings were deleted
    # on 01467200_l1. Making this a returned measurement rather than a prompt
    # instruction is deliberate: the agent had per-point noise_ratio in every
    # describe_points row already and did not act on it.
    series = df.set_index(DATETIME_COL)[field]
    noise = context.noise_profile(series)
    # What counts as a big step IN THIS RECORD, for flag_jumps.thresh (§7.6).
    # thresh is in data units and nothing else the summary reports can size it:
    # the p99 of the 3h jump statistic is 2.9 x the value MAD on 01467200 and
    # 1.4 x on 040851385, but 33 x on 02054550. A run that sized it
    # from `std` instead set thresh at roughly the 60th percentile of ordinary
    # window-to-window movement, got 2,865 flags, and blanket-kept all of them.
    jumps = context.jump_scale(series)
    return {
        "tool": "inspect_dataset",
        "params": {"field": field},
        "n_rows": summary.n_rows,
        "message": (
            "Dataset inspected. " + noise.get("message", "") + " " + jumps.get("message", "")
        ),
        "qc": qc,
        "summary": summary.to_dict(),
        "noise_profile": {k: v for k, v in noise.items() if k not in ("tool", "params")},
        "jump_scale": {k: v for k, v in jumps.items() if k not in ("tool", "params")},
    }


def get_flag_summary(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Looks at the history of the data and counts how many bad data points were found
    by each tool that the robot used so far.
    """
    history = qc._flags.history[field]
    summary = {}
    for col in history.hist.columns:
        test_name = history.meta[col].get("func", col)
        flagged = (history.hist[col] > 0).sum()
        summary[test_name] = summary.get(test_name, 0) + int(flagged)

    return {
        "tool": "get_flag_summary",
        "params": {"field": field},
        "message": f"Flag summary retrieved: {summary}",
        "qc": qc,
        "summary": summary
    }


# The §5 flag-log vocabulary. `undecided` is not one of the four actions the agent
# may choose: it is what a flagged row gets when the agent recorded no decision for
# it, written explicitly so the log distinguishes "the agent looked at this and kept
# it" from "nobody ever adjudicated this". `evaluate.py` scores both as non-positive,
# but only one of them is a defensible answer.
DECISION_ACTIONS = ("delete", "correct", "keep", "impute")
UNDECIDED = "undecided"

# THE VERDICT IS THE RUN'S ANSWER (2026-08-13). A flag is a candidate; the verdict is
# the agent saying "this row IS / IS NOT anomalous", and §10 scores precision/recall
# against it. It used to be inferred from `action` — delete/correct/impute read as a
# positive claim, keep as a rejection — which conflated the detection call with the
# treatment call and left the agent no way to say "this is a real artifact I chose not
# to touch". Stating it costs one field and makes the claim auditable.
VERDICTS = ("anomaly", "normal")

# The §6 type the agent assigns when its verdict is `anomaly`. Taken from the §5 label
# vocabulary (minus the empty non-anomaly member) so the two cannot drift apart —
# scoring compares this string against the label file's `anomaly_type` directly.
VERDICT_ANOMALY_TYPES = tuple(sorted(ANOMALY_TYPES - {""}))

# How far verdict and action may diverge (settled 2026-08-13).
#
#   normal  -> the action MUST be `keep`. Judging a value genuine and then deleting it
#              is not a defensible pair, and allowing it would let a run score as a
#              rejection while the value it "kept" is gone from the file.
#   anomaly -> normally delete / correct / impute, but `keep` IS allowed when the agent
#              says why the row cannot be treated. The case this exists for is a gap
#              longer than any defensible imputation window: `gap` is the right verdict
#              and leaving it NaN is the right action, and forcing consistency here
#              would make the agent either lie about the verdict or impute a gap it
#              had just judged unfillable.
#
# The asymmetry is the whole point: it is one-directional, so `normal` still means
# exactly one thing, and detection recall is not quietly bought with untreated rows.
UNTREATED_ANOMALY_ACTION = "keep"

# `flagged_by` for a row no detector flagged, brought into the log by an `anomaly`
# decision span covering it (§5). Distinguishes "a decision claimed this" from "a
# detector found this" — the two are different provenance and the audit pages say so.
SPAN_CLAIMED = "decision-span"
MIN_UNTREATED_REASON_CHARS = 20

# How the agent rates its own call. `judgement-call` obliges it to write out the
# reasoning; `clear` says the evidence was one-sided and a one-line reason suffices.
DIFFICULTIES = ("clear", "judgement-call")

# Where a row's explanation came from. The distinction matters because "the agent
# reasoned about this point" and "a span covering half the record happened to include
# it" produce identically-shaped log entries otherwise, and every run so far has had a
# blanket absorb points the agent had measured.
RATIONALE_SOURCES = (
    "agent-deliberation",   # a judgement call, with the agent's reasoning attached
    "agent-reason",         # a specific decision the agent called clear-cut
    "blanket",              # swept up by a span covering a large share of all flags
    "deterministic",        # code decided: imputed by the filler, or never adjudicated
)

# A span covering more than this share of all flagged rows is a blanket, not a verdict
# about any particular row.
BLANKET_SHARE = 0.20


def _rationale(source: str, action: str, flagged_by: str, reason: str) -> str:
    """One sentence saying why this row ended up as it did.

    Deterministic for the cases where code knows the answer; the agent's own words
    where it actually made a call. The point is that a reader never has to guess which
    of the two they are looking at.
    """
    if source == "deterministic":
        if action == "impute":
            return (
                f"Filled by impute_rolling. No judgement was recorded for this row — "
                f"{_IMPUTE_FUNC} wrote a value here and that is what the log reflects."
            )
        return (
            f"Flagged by {flagged_by}, but no decision span covered it, so the agent "
            "never adjudicated this row. It counts as no claim either way."
        )
    if source == "blanket":
        return (
            f"Covered by a blanket '{action}' span rather than judged individually. "
            f"The agent's stated grounds for the blanket: {reason or '(none given)'}"
        )
    return reason or f"Decided '{action}' with no reason recorded."

def _validate_verdict(
    decision: dict, action: str, reason: str, i: int
) -> tuple[str, str]:
    """Validate one decision's ``verdict`` / ``anomaly_type`` pair against *action*.

    Returns the cleaned ``(verdict, anomaly_type)``; ``anomaly_type`` is ``""`` for a
    ``normal`` verdict. Raises :class:`ValueError` with a message written for the agent
    to act on, since a rejected export comes back to it as a tool error it can retry.
    """
    verdict = str(decision.get("verdict", "")).strip().lower()
    if verdict not in VERDICTS:
        raise ValueError(
            f"decisions[{i}] has verdict={decision.get('verdict')!r}; must be one of "
            f"{', '.join(VERDICTS)}. The verdict is this run's actual answer — whether "
            "the segment IS anomalous — and it is what precision and recall are "
            "measured against, so it cannot be left off."
        )

    anomaly_type = str(decision.get("anomaly_type", "")).strip().lower()
    if verdict == "anomaly":
        if anomaly_type not in VERDICT_ANOMALY_TYPES:
            raise ValueError(
                f"decisions[{i}] is verdict='anomaly' but anomaly_type="
                f"{decision.get('anomaly_type')!r}; must be one of "
                f"{', '.join(VERDICT_ANOMALY_TYPES)}. Name the failure you are claiming: "
                "the type is scored against the labels as YOUR classification, not "
                "inferred from whichever detector happened to fire."
            )
    else:
        if anomaly_type:
            raise ValueError(
                f"decisions[{i}] is verdict='normal' but also carries anomaly_type="
                f"{anomaly_type!r}. A segment judged genuine has no failure type — drop "
                "the field, or change the verdict to 'anomaly' if you meant that."
            )
        # See UNTREATED_ANOMALY_ACTION: the one-directional half of the rule.
        if action != UNTREATED_ANOMALY_ACTION:
            raise ValueError(
                f"decisions[{i}] is verdict='normal' but action={action!r}. A segment "
                f"you judge genuine must be '{UNTREATED_ANOMALY_ACTION}' — deleting or "
                "correcting a value you just called real water contradicts your own "
                "verdict, and the cleaned file would not match what you claimed. If the "
                "value IS wrong, set verdict='anomaly' with an anomaly_type."
            )

    if verdict == "anomaly" and action == UNTREATED_ANOMALY_ACTION:
        # Allowed, but only as a stated choice. Without this the pair becomes the
        # cost-free way to claim a detection while doing nothing about it.
        if len(reason) < MIN_UNTREATED_REASON_CHARS:
            raise ValueError(
                f"decisions[{i}] claims an anomaly ({anomaly_type}) but keeps the value, "
                f"with a `reason` of only {len(reason)} characters. That pair is allowed "
                "— a gap longer than any defensible window is exactly it — but you must "
                "say why the segment cannot be treated, not leave it blank."
            )

    return verdict, anomaly_type


# A row flagged only by the imputer was, factually, imputed — there is nothing for the
# agent to adjudicate, so it is not counted as undecided.
_IMPUTE_FUNC = "interpolateByRolling"


def _flagged_by_row(qc: saqc.SaQC, field: str) -> pd.Series:
    """Map each flagged timestamp to the '+'-joined names of the tests that flagged it.

    Read from the history rather than the flag frame: a row two tests both flagged
    keeps only the first test's attribution in ``qc.flags`` (§7.1).
    """
    history = qc._flags.history[field]
    names: dict[pd.Timestamp, list[str]] = {}
    for col in history.hist.columns:
        test_name = history.meta[col].get("func", col)
        for stamp in history.hist.index[history.hist[col] > 0]:
            names.setdefault(stamp, [])
            if test_name not in names[stamp]:
                names[stamp].append(test_name)
    if not names:
        return pd.Series(dtype=object)
    return pd.Series(
        {stamp: "+".join(tests) for stamp, tests in names.items()}
    ).sort_index()


def _build_flag_log(
    qc: saqc.SaQC,
    field: str,
    decisions: list[dict] | None,
) -> tuple[list[dict], dict]:
    """Build the §5 flag log: one entry per flagged row, with the agent's decision.

    *decisions* is the agent's list of ``{start, end, verdict, anomaly_type, action,
    reason}`` spans (``end`` inclusive, defaulting to ``start``). A row no span covers
    is written as ``undecided`` so the omission is visible rather than silently scoring
    as a keep.

    **``verdict`` is the answer; ``action`` is the treatment.** The two were one field
    until 2026-08-13, with detection inferred from whether the action was destructive,
    which meant the agent could not say "this is a real artifact I am not touching" and
    every such row scored as a claim that the value was fine. They are now separate and
    separately validated (:func:`_validate_verdict`).

    **The NARROWEST span covering a row wins**, with author order breaking ties. It
    used to be the first span listed, which quietly inverted the agent's own
    reasoning: a run that judged one point a spike on measured evidence and then
    swept up the remainder with a whole-series ``keep`` lost the specific verdict
    whenever the catch-all happened to be listed first. Measured on
    03447687_l1 (2026-08-11): a row read as ``reads_like=spike, robust_z=7.4,
    width=3`` was recorded as ``keep``. A broad span is a statement about what is
    left over, so it must lose to anything more specific no matter where it appears.

    Returns the entries plus a stats dict for the tool result / message.
    """
    flagged_by = _flagged_by_row(qc, field)

    # A DECISION SPAN CLAIMING AN ANOMALY MARKS EVERY ROW IT COVERS, flagged or not.
    #
    # Until 2026-08-24 the log held one entry per FLAGGED row, and that silently capped
    # what a windowed anomaly could ever claim. A level shift is a span sitting at the
    # wrong level; flagJumps flags its two EDGES, so the interior was never in the log and
    # no span could reach it. Measured on 01467200_l1: the agent identified the shift
    # exactly (2023-09-15 11:40 -> 21:30, against a true 11:40 -> 21:25) and wrote a span
    # over the whole window — and claimed 2 of 117 rows, because only 3 rows inside it had
    # been flagged. Every run scored ~0 on level_shift for this reason, whatever the
    # detector or prompt did.
    #
    # Only `anomaly` spans materialise rows. A `normal` span is a statement about
    # candidates the detectors raised, so it annotates flagged rows and nothing else —
    # otherwise a whole-record catch-all keep would write an entry for all 210,816 rows.
    index = pd.DatetimeIndex(qc.data.to_pandas().index)
    claimed: list[pd.Timestamp] = []
    for decision in decisions or []:
        if not isinstance(decision, dict):
            continue
        if str(decision.get("verdict", "")).strip().lower() != "anomaly":
            continue
        try:
            lo = pd.Timestamp(decision["start"])
            hi = pd.Timestamp(decision.get("end") or decision["start"])
        except (KeyError, ValueError, TypeError):
            continue                      # validated properly below; skip here
        if hi < lo:
            lo, hi = hi, lo
        claimed.append(index[(index >= lo) & (index <= hi)])
    extra = index[[]] if not claimed else pd.DatetimeIndex(np.concatenate(
        [c.to_numpy() for c in claimed])).unique()
    n_materialised = len(extra.difference(pd.DatetimeIndex(flagged_by.index)))

    stamps = pd.DatetimeIndex(flagged_by.index).union(extra).sort_values()
    # Rows nothing flagged carry an explicit marker rather than a blank, so a reader can
    # tell "a decision claimed this" from "a detector found this".
    flagged_by = flagged_by.reindex(stamps).fillna(SPAN_CLAIMED)

    actions = pd.Series(index=stamps, dtype=object)
    reasons = pd.Series("", index=stamps, dtype=object)
    deliberations = pd.Series("", index=stamps, dtype=object)
    sources = pd.Series("", index=stamps, dtype=object)
    verdicts = pd.Series("", index=stamps, dtype=object)
    anomaly_types = pd.Series("", index=stamps, dtype=object)
    # Which span claimed each row (its author order), or -1. `rationale_source` says
    # a row was swept up by a blanket; this says by WHICH one and how wide it was,
    # which is what an auditor needs to see to judge whether the row was really judged.
    claimed_by = pd.Series(-1, index=stamps, dtype=int)

    # Validate everything before applying anything, so a bad span late in the list
    # cannot leave a half-written log behind.
    spans: list[dict] = []
    for i, decision in enumerate(decisions or []):
        if not isinstance(decision, dict):
            raise ValueError(f"decisions[{i}] is not an object with start/action/reason.")
        action = str(decision.get("action", "")).strip().lower()
        if action not in DECISION_ACTIONS:
            raise ValueError(
                f"decisions[{i}] has action={decision.get('action')!r}; "
                f"must be one of {', '.join(DECISION_ACTIONS)}."
            )
        try:
            start = pd.Timestamp(decision["start"])
            end = pd.Timestamp(decision.get("end") or decision["start"])
        except (KeyError, ValueError, TypeError) as exc:
            raise ValueError(f"decisions[{i}] has an unparseable start/end: {exc}") from exc
        if end < start:
            raise ValueError(f"decisions[{i}] ends ({end}) before it starts ({start}).")

        # REQUIRED, with no default. It used to default to "clear", and the result
        # was that the field carried no information at all: measured on the
        # 2026-08-20 run, 145 of 147 spans left it unset, so 98.6% of the run was
        # implicitly "clear-cut" — and those deletions were wrong 37.8% of the time.
        # A default that manufactures a confidence claim on the agent's behalf is
        # worse than no field, because it reads as a judgement nobody made.
        if "difficulty" not in decision or decision.get("difficulty") in (None, ""):
            raise ValueError(
                f"decisions[{i}] is missing `difficulty`. Every decision must say "
                f"whether it was clear-cut or a judgement call — there is no default. "
                f"Use one of: {', '.join(DIFFICULTIES)}. Mark it 'judgement-call' "
                "whenever a reasonable reviewer could disagree; those are the spans a "
                "human will be asked to check."
            )
        difficulty = str(decision["difficulty"]).strip().lower()
        if difficulty not in DIFFICULTIES:
            raise ValueError(
                f"decisions[{i}] has difficulty={decision.get('difficulty')!r}; "
                f"must be one of {', '.join(DIFFICULTIES)}."
            )
        deliberation = str(decision.get("deliberation", "")).strip()
        # A judgement call without its reasoning is the thing this field exists to
        # prevent, so it fails loudly (§13) rather than recording an empty rationale.
        if difficulty == "judgement-call" and len(deliberation) < 40:
            raise ValueError(
                f"decisions[{i}] is marked difficulty='judgement-call' but its "
                f"`deliberation` is missing or too short ({len(deliberation)} chars). "
                "A judgement call must carry the reasoning behind it: what the numbers "
                "said, what you weighed against what, and what would have changed your "
                "mind. Mark it 'clear' only if the evidence was genuinely one-sided."
            )

        reason = str(decision.get("reason", "")).strip()
        verdict, anomaly_type = _validate_verdict(decision, action, reason, i)

        spans.append({
            "start": start, "end": end, "action": action, "order": i,
            "reason": reason, "verdict": verdict, "anomaly_type": anomaly_type,
            "difficulty": difficulty, "deliberation": deliberation,
        })

    unmatched: list[str] = []
    # Narrowest first, ties broken by the order the agent wrote them (the index keeps
    # the sort stable and explicit). A row is then claimed by the most specific span
    # that covers it, whatever position the catch-all occupies.
    for span in sorted(spans, key=lambda s: (s["end"] - s["start"], s["order"])):
        within = (stamps >= span["start"]) & (stamps <= span["end"])
        if not within.any():
            # Reported only when the span matches NO flagged row at all — a mistyped
            # timestamp. A broad span whose rows were all claimed by narrower ones is
            # doing exactly its job and must not be flagged as an error.
            unmatched.append(
                f"{span['start'].isoformat()}–{span['end'].isoformat()} ({span['action']})"
            )
            continue
        covered = within & actions.isna().to_numpy()
        if not covered.any():
            continue
        span["reach"] = int(within.sum())
        span["claimed"] = int(covered.sum())
        claimed_by[covered] = span["order"]
        actions[covered] = span["action"]
        reasons[covered] = span["reason"]
        deliberations[covered] = span["deliberation"]
        verdicts[covered] = span["verdict"]
        anomaly_types[covered] = span["anomaly_type"]
        # A span sweeping up a large share of everything flagged is a statement about
        # the remainder, not a judgement about any particular row — and telling the two
        # apart per row is the whole point of recording a source. Every run so far has
        # had one of these absorb points the agent had actually measured.
        blanket = int(within.sum()) > BLANKET_SHARE * max(len(stamps), 1)
        sources[covered] = (
            "blanket" if blanket
            else "agent-deliberation" if span["deliberation"]
            else "agent-reason"
        )

    # A row the imputer FILLED is `impute`, full stop — that is a fact about what
    # happened to the data, not a verdict the agent can override. In particular a
    # broad catch-all `keep` span (which is what an agent naturally writes to sweep
    # up its remaining spike/jump flags) otherwise swallows every filled gap in its
    # time range and records "left untouched" on rows whose value was replaced.
    # Measured on 03447687_l1, 2026-08-10: one such span mislabelled 3,429 filled
    # rows as `keep` and took gap F1 from 1.0 to 0.
    #
    # An explicit delete/correct still wins: choosing to act further on a filled
    # value is a real decision, where "keep" on it is simply false.
    filled = flagged_by.str.contains(_IMPUTE_FUNC, regex=False).to_numpy()
    overridden = filled & actions.isin(["keep"]).to_numpy()
    imputed_rows = filled & (actions.isna().to_numpy() | overridden)
    actions[imputed_rows] = "impute"
    undecided_rows = actions.isna().to_numpy()
    actions = actions.fillna(UNDECIDED)
    n_keep_overridden = int(overridden.sum())
    sources[imputed_rows] = "deterministic"
    sources[undecided_rows] = "deterministic"
    # Code decided these, so attributing them to a span the agent wrote would be a
    # lie — including for a `keep` span the imputer overrode (counted separately as
    # n_keep_rewritten_to_impute).
    claimed_by[imputed_rows] = -1
    claimed_by[undecided_rows] = -1

    # A row the filler wrote a value into was missing, and §5 is explicit that every
    # missing run is a gap — so the verdict here is a fact about the data, not a call
    # the agent has to make. It follows the action for the same reason the action is
    # forced above: leaving these blank would let a run impute 3,403 rows and record
    # no claim about any of them.
    verdicts[imputed_rows] = "anomaly"
    anomaly_types[imputed_rows] = "gap"
    # Nobody adjudicated these, which is neither "anomaly" nor "normal" (§10 scores it
    # as no claim in either direction). Naming it keeps a forgotten segment visible
    # instead of letting it read as a considered "normal".
    verdicts[undecided_rows] = UNDECIDED
    anomaly_types[undecided_rows] = ""

    by_order = {span["order"]: span for span in spans}
    entries = []
    for i, stamp in enumerate(stamps):
        source = sources.iloc[i] or "deterministic"
        span = by_order.get(int(claimed_by.iloc[i]))
        entries.append({
            "datetime": stamp.strftime("%Y-%m-%dT%H:%M:%S"),
            "flagged_by": flagged_by.loc[stamp],
            # The run's answer for this row, and what §10 scores. `action` says what
            # was done to the value; `verdict` says what the agent concluded it IS.
            "verdict": verdicts.iloc[i],
            "anomaly_type": anomaly_types.iloc[i],
            "action": actions.loc[stamp],
            "reason": reasons.loc[stamp],
            # Where the explanation comes from, so a reader can tell a considered call
            # from a blanket or from something code decided. See RATIONALE_SOURCES.
            "rationale_source": source,
            "rationale": _rationale(
                source, actions.loc[stamp], flagged_by.loc[stamp], reasons.loc[stamp]
            ),
            "deliberation": deliberations.iloc[i],
            # The span that claimed this row. `n_flagged_rows_in_span` is its whole
            # reach and `n_rows_claimed` what survived narrower spans, so a reader can
            # see at a glance whether this row was judged or absorbed.
            "decided_by": None if span is None else {
                "start": span["start"].strftime("%Y-%m-%dT%H:%M:%S"),
                "end": span["end"].strftime("%Y-%m-%dT%H:%M:%S"),
                "n_flagged_rows_in_span": span["reach"],
                "n_rows_claimed": span["claimed"],
                "difficulty": span["difficulty"],
            },
        })
    stats = {
        "n_entries": len(entries),
        "n_undecided": int((actions == UNDECIDED).sum()),
        "n_keep_rewritten_to_impute": n_keep_overridden,
        "n_by_verdict": {
            verdict: int((verdicts == verdict).sum())
            for verdict in (*VERDICTS, UNDECIDED)
            if (verdicts == verdict).any()
        },
        "n_by_anomaly_type": {
            anomaly_type: int((anomaly_types == anomaly_type).sum())
            for anomaly_type in VERDICT_ANOMALY_TYPES
            if (anomaly_types == anomaly_type).any()
        },
        # verdict='anomaly' + action='keep': allowed, but it is the pair that claims a
        # detection without touching the value, so the count is surfaced rather than
        # buried in the per-row entries.
        "n_anomalies_left_untreated": int(
            ((verdicts == "anomaly") & (actions == UNTREATED_ANOMALY_ACTION)).sum()
        ),
        "n_by_action": {
            action: int((actions == action).sum())
            for action in (*DECISION_ACTIONS, UNDECIDED)
            if (actions == action).any()
        },
        "n_by_rationale_source": {
            source: sum(1 for e in entries if e["rationale_source"] == source)
            for source in RATIONALE_SOURCES
            if any(e["rationale_source"] == source for e in entries)
        },
        "decisions_matching_no_flagged_row": unmatched,
        "n_rows_claimed_by_span_not_flagged": n_materialised,
    }
    return entries, stats


def export_clean_data(
    qc: saqc.SaQC,
    field: str = "value",
    decisions: list[dict] | None = None,
    output_dir: str | Path | None = None,
    stem: str | None = None,
) -> dict:
    """
    Takes the final, cleaned data and gives it back as a simple spreadsheet-like format,
    marking which points were flagged by the tools, and writes the §5 flag log.

    *decisions* is the agent's per-segment verdict list — ``{start, end, verdict,
    anomaly_type, action, reason}`` — and is what turns a pile of flags into an answer:
    `evaluate.py` scores a flagged row as a positive claim only where the agent said
    ``verdict='anomaly'``, so a run that exports without decisions scores as if it
    claimed nothing.

    *output_dir* and *stem* are supplied by the runner, not by the agent (they are
    absent from the tool schema): given both, the flag log is written to
    ``<output_dir>/<stem>_flags.json``. Without them the entries are returned only.
    """
    df = qc.data.to_pandas()

    # Contract: 'flag' column naming the action, if any
    # Since saqc just returns float flags, we can map > 0 to 'flagged'
    # and we can deduce actions from history if needed, but for now we'll
    # just create a generic flag column if it's flagged.
    history = qc._flags.history[field]
    df['flag'] = None

    for col in history.hist.columns:
        test_name = history.meta[col].get("func", col)
        mask = history.hist[col] > 0
        df.loc[mask, 'flag'] = test_name

    entries, stats = _build_flag_log(qc, field, decisions)

    flags_path = None
    if output_dir is not None and stem is not None:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        flags_path = out_dir / f"{stem}_flags.json"
        flags_path.write_text(json.dumps(entries, indent=2))
        flags_path = str(flags_path)

    msg = (
        f"Data exported. Flag log holds {stats['n_entries']} flagged row(s). "
        f"VERDICTS (this is what the run is scored on): {stats['n_by_verdict']}"
        + (f", by type {stats['n_by_anomaly_type']}" if stats["n_by_anomaly_type"] else "")
        + f". Actions taken: {stats['n_by_action']}."
    )
    if flags_path:
        msg += f" Written to {flags_path}."
    if stats["n_anomalies_left_untreated"]:
        msg += (
            f" NOTE: {stats['n_anomalies_left_untreated']} row(s) are called anomalous but"
            " kept as recorded. That is a legitimate pair for something you cannot treat"
            " (a gap longer than any defensible window); check it is what you meant on"
            " every one of them, since it claims the detection without changing the value."
        )
    if stats["n_undecided"]:
        msg += (
            f" WARNING: {stats['n_undecided']} flagged row(s) carry NO decision and are"
            " recorded as verdict 'undecided' — they count as no claim in either"
            " direction when this run is scored, so they are neither a detection nor a"
            " rejection. Call export_clean_data again with a `decisions` entry covering"
            " every flagged segment (each with a verdict, an action and a reason)."
        )
    blanket = stats["n_by_rationale_source"].get("blanket", 0)
    if blanket:
        msg += (
            f" NOTE: {blanket} flagged row(s) were swept up by a blanket span rather than"
            " judged individually — including any you measured. If you inspected a point"
            " and formed a view about it, give it its own span so the record shows that,"
            " rather than letting a catch-all speak for it."
        )
    if stats["n_keep_rewritten_to_impute"]:
        msg += (
            f" NOTE: {stats['n_keep_rewritten_to_impute']} row(s) you marked 'keep' were"
            " filled by impute_rolling and are recorded as 'impute' — their value was"
            " replaced, so 'keep' would misdescribe the file. If you meant to sweep up"
            " only your spike/jump flags, narrow that span so it does not span the gaps."
        )
    if stats["decisions_matching_no_flagged_row"]:
        msg += (
            " WARNING: these decision spans matched no flagged row (wrong timestamp, or"
            " already covered by an earlier span): "
            + "; ".join(stats["decisions_matching_no_flagged_row"])
            + "."
        )

    return {
        "tool": "export_clean_data",
        "params": {"field": field, "n_decisions": len(decisions or [])},
        "message": msg,
        "flags_path": flags_path,
        **stats,
        "qc": qc,
        "df": df,
        "flags": entries,
    }


def flag_range(qc: saqc.SaQC, field: str = "value", min=None, max=None) -> dict:
    """
    Flags any data points that are too high or too low based on a set minimum and maximum limit.
    """
    params = {"min": min, "max": max}
    qc_out = qc.flagRange(field, min=min, max=max)
    return _build_result("flag_range", params, qc, qc_out, field)


def flag_constants(qc: saqc.SaQC, field: str = "value", thresh=0.0, window=None, min_periods=2) -> dict:
    """
    Flags data points that get "stuck" (like a broken thermometer showing the exact same
    number for hours). It checks if values stay completely flat for a certain time window.
    """
    params = {"thresh": thresh, "window": window, "min_periods": min_periods}
    qc_out = qc.flagConstants(field, thresh=thresh, window=window, min_periods=min_periods)
    return _build_result("flag_constants", params, qc, qc_out, field)


def flag_plateau(qc: saqc.SaQC, field: str = "value", min_length="1h", max_length=None, min_jump=None, granularity=None) -> dict:
    """
    Flags a "plateau" - when the data suddenly jumps up, stays flat for a while, and then
    drops back down. This happens when debris gets stuck on the sensor temporarily.
    """
    params = {"min_length": min_length, "max_length": max_length, "min_jump": min_jump, "granularity": granularity}
    # min_jump is not a valid argument for flagPlateau in saqc 2.8 maybe? Wait.
    # CLAUDE.md §7: `flagPlateau` | `min_length`, `max_length`, `min_jump`, `granularity`
    # We will pass kwargs dynamically to avoid None defaults if they aren't accepted.
    kwargs = {}
    if min_length is not None: kwargs["min_length"] = min_length
    if max_length is not None: kwargs["max_length"] = max_length
    if min_jump is not None: kwargs["min_jump"] = min_jump
    if granularity is not None: kwargs["granularity"] = granularity

    # flagPlateau is the one §7 method that raises on perfectly ordinary input, and it
    # is data-dependent rather than length-monotonic (§7.1): 'attempt to get argmin of
    # an empty sequence' from _getAnomalyCenter, or a numpy window-shape error when the
    # window outruns the array. One crashy detector must not sink the whole run, so the
    # failure comes back as a normal result saying it found nothing and why.
    try:
        qc_out = qc.flagPlateau(field, **kwargs)
    except ValueError as exc:
        return {
            "tool": "flag_plateau",
            "params": params,
            "n_flagged": 0,
            "pct_flagged": 0.0,
            "n_flagged_total": int((qc.flags[field] > 0).sum()),
            "flagged_datetimes": [],
            "message": (
                f"flag_plateau could not run on this series and flagged nothing: {exc}. "
                "This is a known SaQC 2.8 defect, not a statement about the data — it says "
                "nothing about whether plateaus are present. Do not retry it with the same "
                "parameters; rely on flag_constants for stuck-sensor detection instead."
            ),
            "failed": True,
            "qc": qc,  # unchanged: nothing was flagged
        }
    return _build_result("flag_plateau", params, qc, qc_out, field)


def flag_spike_unilof(qc: saqc.SaQC, field: str = "value", n=20, thresh=None, density='auto', slope_correct=True) -> dict:
    """
    Flags sudden, sharp "spikes" in the data (outliers) using a smart math trick called
    Local Outlier Factor. It looks for points that are very different from their neighbors.
    """
    params = {"n": n, "thresh": thresh, "density": density, "slope_correct": slope_correct}
    qc_out = qc.flagUniLOF(field, n=n, thresh=thresh, density=density, slope_correct=slope_correct)
    return _build_result("flag_spike_unilof", params, qc, qc_out, field)


def flag_zscore(qc: saqc.SaQC, field: str = "value", method='standard', window=None, thresh=3.0) -> dict:
    """
    Another way to find spikes. It calculates an average over a rolling window of time,
    and flags any data points that stray too far away from that local average.
    """
    params = {"method": method, "window": window, "thresh": thresh}
    qc_out = qc.flagZScore(field, method=method, window=window, thresh=thresh)
    return _build_result("flag_zscore", params, qc, qc_out, field)


def flag_jumps(qc: saqc.SaQC, field: str = "value", thresh=None, window=None) -> dict:
    """
    Flags permanent jumps in the data. For example, if the sensor is bumped into a different
    position and the readings suddenly jump up and stay there forever.

    `thresh` is the minimum difference between the mean of the preceding `window` and the
    mean of the following one, in data units. It has NO portable default -- the workable
    value is 6.2 / 13.8 / 75.7 FNU on the three project gauges -- so it is required, and
    `inspect_dataset`'s `jump_scale` block measures it for the series at hand (§7.6).
    `thresh=0` flags every change point in the record and is rejected rather than run.
    """
    if thresh is None or window is None:
        raise ValueError(
            "flag_jumps needs both thresh and window; take them from inspect_dataset's "
            "jump_scale block (recommended_thresh / recommended_window)."
        )
    if thresh <= 0:
        raise ValueError(
            f"flag_jumps thresh must be > 0, got {thresh}: a non-positive threshold flags "
            "every change point in the record."
        )
    params = {"thresh": thresh, "window": window}
    qc_out = qc.flagJumps(field, thresh=thresh, window=window)
    return _build_result("flag_jumps", params, qc, qc_out, field)


def flag_nan(qc: saqc.SaQC, field: str = "value") -> dict:
    """
    Flags places where the data is completely missing (NaN - Not a Number).
    """
    params = {}
    qc_out = qc.flagNAN(field)
    return _build_result("flag_nan", params, qc, qc_out, field)


def impute_rolling(
    qc: saqc.SaQC,
    field: str = "value",
    window=None,
    func: str = "median",
    min_periods: int = 0,
    max_gap: str | None = None,
) -> dict:
    """Fill NaN gaps using a rolling window median (or other aggregation).

    Only fills gaps whose duration is <= max_gap. Longer gaps are left as NaN
    and reported in the result so the agent knows they were skipped.

    max_gap: pandas offset string, e.g. '3h'. If None, all gaps are imputed
    up to what the window can reach. Always set max_gap to the longest gap you
    are willing to accept; do not impute multi-day outages.
    """
    params = {"window": window, "func": func, "min_periods": min_periods, "max_gap": max_gap}

    # --- 1. Analyse gaps before touching the data ---
    pre_series = qc.data.to_pandas()[field]
    nan_runs   = _find_nan_runs(pre_series)

    max_gap_td = pd.Timedelta(max_gap) if max_gap is not None else None

    fillable_runs  = []
    too_large_runs = []
    for run in nan_runs:
        if max_gap_td is not None and run["duration_td"] > max_gap_td:
            too_large_runs.append(run)
        else:
            fillable_runs.append(run)

    # --- 2. Call SaQC's interpolateByRolling ---
    # flag=25 (DOUBTFUL) so imputed rows appear in the flag history (§7.1).
    #
    # dfilter=np.inf is load-bearing. SaQC masks every row whose flag is >= `dfilter`
    # before a function runs, and the default masks BAD — so a row an earlier detector
    # flagged looks MISSING to the imputer, which fills it, silently replacing a real
    # reading that was never absent. Measured on 03447687_l1 (2026-08-10): 1,879 rows
    # that were never NaN were rewritten with a rolling median, 1,866 of them ordinary
    # water and 13 of them injected anomalies the agent consequently never judged.
    # It also corrupted three separate measurements — evaluate.py's spike precision,
    # and the `missed` counts on both audit pages — because an overwritten row carries
    # `action=impute` and reads as a successful gap fill.
    #
    # np.inf means nothing is masked, so the imputer sees the real data and fills only
    # genuine NaN. Verified in scratchpad/probe_impute_dfilter.py: a flagged spike
    # survives untouched while a real gap is still filled.
    pre_nans = int(pre_series.isna().sum())
    qc_out   = qc.interpolateByRolling(
        field, window=window, func=func, min_periods=min_periods, flag=25,
        dfilter=np.inf,
    )
    post_series = qc_out.data.to_pandas()[field]
    post_nans   = int(post_series.isna().sum())
    n_imputed   = pre_nans - post_nans
    n_total     = len(pre_series)

    # --- 3. Warn if any too-large gap was partially filled ---
    # SaQC's rolling window naturally can't bridge a gap wider than `window`,
    # but it will fill the edges.  Report every such case explicitly.
    partial_fill_warnings: list[str] = []
    for run in too_large_runs:
        run_slice = post_series.loc[run["start"]:run["end"]]
        n_partial  = int(run_slice.notna().sum())
        if n_partial > 0:
            partial_fill_warnings.append(
                f"Gap {run['start'].isoformat()}–{run['end'].isoformat()} "
                f"({run['duration_td']}) exceeds max_gap='{max_gap}': "
                f"{n_partial} edge row(s) were partially filled."
            )

    # --- 4. Build gap summary (JSON-serialisable, for agent context) ---
    gaps_summary = [
        {
            "start":           run["start"].isoformat(),
            "end":             run["end"].isoformat(),
            "duration":        str(run["duration_td"]),
            "n_rows":          run["n_rows"],
            "skipped_too_large": (max_gap_td is not None and run["duration_td"] > max_gap_td),
        }
        for run in nan_runs
    ]

    # --- 5. Compose message ---
    pct_imputed = round(n_imputed / n_total, 4) if n_total > 0 else 0.0
    msg_parts   = [
        f"Imputed {n_imputed} values ({pct_imputed * 100:.1f}%) across "
        f"{len(fillable_runs)} of {len(nan_runs)} gap(s). "
        f"{post_nans} NaN(s) remain."
    ]
    if too_large_runs:
        msg_parts.append(
            f"{len(too_large_runs)} gap(s) exceeded max_gap='{max_gap}' and were not imputed."
        )
    if partial_fill_warnings:
        msg_parts.append("PARTIAL FILL WARNING: " + " | ".join(partial_fill_warnings))

    msg = " ".join(msg_parts)

    result = _build_result("impute_rolling", params, qc, qc_out, field, custom_msg=msg)
    result["n_imputed"]            = n_imputed
    result["n_gaps_total"]         = len(nan_runs)
    result["n_gaps_filled"]        = len(fillable_runs)
    result["n_gaps_skipped_large"] = len(too_large_runs)
    # Longest gaps first, then capped: which gaps were too long to fill is the decision
    # this list informs, and those are exactly the ones at the top.
    by_length = sorted(gaps_summary, key=lambda g: g["n_rows"], reverse=True)
    result["gaps_summary"]         = by_length[:MAX_GAPS_SUMMARY]
    if len(by_length) > MAX_GAPS_SUMMARY:
        result["message"] += (
            f" NOTE: gaps_summary lists the {MAX_GAPS_SUMMARY} longest of"
            f" {len(by_length)} gaps; the counts above cover all of them."
        )
    return result


# ---------------------------------------------------------------------------
# Context tools (CLAUDE.md §7.3)
#
# Thin pass-throughs to src/agent_tools/context.py. The measurement lives there;
# these exist so every tool the agent can call is reachable from one module with
# one calling convention -- `qc` first, like every wrapper above. They OBSERVE:
# nothing here flags or mutates, so the results carry no `qc` key and the caller's
# SaQC object is unchanged (`inspect_dataset` sets that precedent in §5).
#
# Only the two aggregators are wrapped. The eight primitives behind them stay
# library functions: nine near-identical tools would eat the 25-call cap, and
# describe_point already returns all of them at once.
# ---------------------------------------------------------------------------

def describe_point(
    qc: saqc.SaQC,
    at: str,
    field: str = "value",
    # 45 min, matching context.slope_context — the scale at which the fall/rise ratio
    # separates a flush event from an artifact (it inverts past ~90 min).
    n_before: int = 3,
    n_after: int = 3,
    window: str = "6h",
    shift_window: str = "24h",
) -> dict:
    """Measure the shape of the series around ONE timestamp.

    Answers the question a detector cannot: is this excursion real water or a
    sensor artifact? Returns the eight nested measurement blocks from
    :mod:`src.agent_tools.context` (slope, excursion, recovery, level_shift,
    flatness, neighbourhood, gap, history) plus a ``reads_like`` hint.

    Raises ValueError if *at* is not a timestamp in the series -- deliberately,
    rather than rounding silently to a neighbour (§13).
    """
    return context.describe_point(
        qc,
        at,
        field=field,
        n_before=n_before,
        n_after=n_after,
        window=window,
        shift_window=shift_window,
    )


def describe_points(
    qc: saqc.SaQC,
    ats: list[str],
    field: str = "value",
    max_points: int = 100,
    window: str = "6h",
) -> dict:
    """Compact shape measurements for a LIST of timestamps -- e.g. a detector's output.

    One row per timestamp (value, reads_like, robust_z, width_samples,
    peak_sharpness, fall_rise_ratio, samples_to_recover, recovered) plus a tally
    by label, so a whole detector result can be judged in a single call.

    Timestamps beyond *max_points* are reported in ``n_truncated`` rather than
    dropped silently; unresolvable ones land in ``errors`` instead of raising, so
    one bad timestamp cannot sink the batch.

    ``max_points`` was 20 until 2026-08-11, and that default — not the call budget,
    not any property of the data — was what limited a run to inspecting 80 of 290
    spike candidates: the agent filled the advertised batch size exactly, four times,
    and 5 of its 9 missed spikes were candidates it was shown but could never ask
    about. A row costs ~340 characters (~85 tokens), so 100 points is ~8.5k tokens
    in one result, written once and thereafter read from cache (§8). Raise it further
    for a detector that returns more; the cap exists to keep one call bounded, not to
    ration inspection.
    """
    return context.describe_points(
        qc,
        ats,
        field=field,
        max_points=max_points,
        window=window,
    )
