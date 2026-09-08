"""Propose candidate anomalies in a "clean" series, for human review.

A base that was *not* vetted by USGS record processing — a `data/turbidity/provisional/`
series, or any segment picked by eye — may still contain real anomalies, which
would score as false positives against the injected labels. This module is how
that assumption gets checked; it is dormant for the default
`data/turbidity/approved/` bases (CLAUDE.md §9.1). It runs the §7 SaQC 2.8 detectors
over such a series at
deliberately *sensitive* settings, groups the flagged rows into contiguous
**segments**, and ranks them. Nothing here decides anything: every segment is a
*candidate* that a human confirms or rejects in the review page built by
``src.workbench.review``.

Because the aim is recall (miss nothing) rather than precision, the output is
expected to contain false positives. That is what the review step is for.

Per-type notes
--------------
- **spike** — ``flagUniLOF`` + ``flagZScore``; a segment found by both is ranked
  higher via its score, but either alone is enough to propose it. Both run at
  settings tuned for >=95% recall against the injected labels
  (``scratchpad/tune_spike_recall.py``, measured by
  ``tests/test_candidates.py::test_candidate_type_recall_on_injected_datasets``),
  which means thousands of segments per two-year series — recall first,
  precision never. ``max_per_type`` is how you trade that back.
- **plateau** — ``flagConstants`` (stuck at any level) + ``flagPlateau``
  (offset segment). These find different failures, so both run (CLAUDE.md §7.1).
- **level_shift** — ``flagJumps``, then a **sharpness filter**. ``flagJumps``
  flags any change of ``thresh`` within its window, which on storm-driven
  turbidity means every rising and falling limb — gradual slopes that are normal
  water behaviour, not sensor faults. A genuine level shift (recalibration,
  sensor swap) moves most of its magnitude in a single sample; a storm spreads it
  over hours. So a candidate is kept only if its sharpest one-sample move is a
  large fraction of the net step (:attr:`DetectConfig.shift_min_sharpness`).
  **Known limit:** this cannot reject a flash-flood onset, which is also sharp and
  sustained. No univariate test separates the two — a real sensor step and a
  sudden storm look identical in one series. Across the three project gauges,
  every level_shift candidate that survived review was rejected by the human, so
  treat this detector's output as "look here", not "this is an artifact".
Gaps are **not** proposed for review — :func:`merge_decisions` labels them
directly from the data. Nothing about a NaN run is a judgement call: the value is
either present or it is not, and §5 is explicit that *every* missing run is
``anomaly_type=gap``. Queueing them would only invite a reviewer to press
"normal" on one and produce labels that contradict the contract.

Drift is **not** proposed: the drift track is shelved (CLAUDE.md §9.2). SaQC 2.8
ships no univariate drift detector, so it had to be a local heuristic, it was the
weakest proposer here, and seasonal turbidity swells are indistinguishable from
fouling anyway.

Thresholds in data units (``flagJumps.thresh``, ``flagConstants.thresh``) are
derived from the series' own robust scale, so the defaults transfer across
gauges on very different turbidity ranges. Every one can be overridden.

Usage
-----
    from src.workbench.candidates import find_candidates

    result = find_candidates("data/turbidity/provisional/06818000_turbidity_63680_provisional.csv")
    print(result.summary())

See ``src.workbench.review`` for the CLI and the labelling page.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import saqc

from src.inspect_data import (
    DATETIME_COL,
    VALUE_COL,
    load_series,
    reindex_to_grid,
)

FIELD = VALUE_COL

AnomalyType = Literal["spike", "plateau", "level_shift", "gap"]

#: Types proposed for human review, in presentation order. Point-like first —
#: they are the quickest to judge, so the reviewer builds a feel for the series
#: before reaching the slower, more ambiguous segment types. ``gap`` is
#: deliberately absent; see the module docstring.
REVIEW_TYPES: tuple[str, ...] = ("spike", "plateau", "level_shift")

#: Every type that can appear in a merged labels file. ``gap`` comes last so it
#: wins overlap resolution in :func:`merge_decisions`: a missing value is missing
#: whatever else a detector thought was happening there.
LABEL_TYPES: tuple[str, ...] = REVIEW_TYPES + ("gap",)

#: Colour per type, shared with the review page.
TYPE_COLORS: dict[str, str] = {
    "spike": "#dc2626",
    "plateau": "#7c3aed",
    "level_shift": "#d97706",
    "gap": "#0891b2",
}


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class DetectConfig:
    """Tuning knobs for candidate proposal. Defaults aim at recall, not precision.

    Values expressed as multiples of the series' robust scale (``*_sigmas``)
    are resolved against the loaded data in :func:`find_candidates`, so the same
    config works on a 0-40 NTU gauge and a 0-1000 NTU one.
    """

    # spike
    #
    # Tuned for >=95% recall of the injected spikes on all nine datasets
    # (scratchpad/tune_spike_recall.py); the earlier n=20/thresh=1.5,
    # zscore thresh=10 defaults recalled only 59-81% of them, which is a miss the
    # review step cannot repair. These settings are deliberately extreme: they
    # flag 4-10% of the series across ~2,500 spike segments per dataset, so the
    # proposal is now far past what a human can review one-by-one. That is the
    # trade this module's docstring already names — recall first, precision
    # never — but see max_per_type below.
    #
    # n is the LOF neighbourhood. 10 beats 20 at every threshold: an injected
    # spike is 1-3 rows (inject.SPIKE_LEN_ROWS), and a 20-sample neighbourhood
    # blurs a 3-row burst into its own local density. At thresh=1.1 on
    # 02198840_l3, n=10 recalls 95.7% vs 87.6% for n=20.
    unilof_n: int = 10
    unilof_thresh: float = 1.1
    zscore_window: str = "12h"
    zscore_thresh: float = 3.0
    # A residual floor in data units. Without it, a 12h window of quantised
    # readings has a near-zero MAD and every wiggle scores an enormous modified
    # z: on 11501000 (0.1 NTU quantisation) flagZScore stuck at ~195 segments
    # even at thresh=30. At 3 step-sigmas that falls to 7. Probed, see
    # scratchpad/tune_zscore.py.
    # Lowered 3.0 -> 2.0: the floor still does its job (it is what keeps a
    # quantised window from flagging everything) but 3.0 cost ~5 points of spike
    # recall on 03447687 for no reduction in segment count worth having.
    zscore_min_residual_sigmas: float = 2.0  # x step scale
    # plateau
    constants_window: str = "3h"
    constants_thresh_sigmas: float = 0.1  # x step scale; must stay << noise sd (§7.1)
    plateau_min_length: str = "1h"
    # level shift
    jumps_window: str = "12h"
    jumps_thresh_sigmas: float = 24.0  # x step scale
    # flagJumps flags any change of `thresh` within `window`, so on storm-driven
    # turbidity it fires on every rising/falling limb — the "gradual slopes" a
    # reviewer rejects. A real level shift (recalibration, sensor swap) moves
    # most of its magnitude in one sample; a storm spreads it over hours. Keep a
    # candidate only if its sharpest single step is at least this fraction of the
    # net level change, and the net change clears the jump threshold. On the three
    # gauges this cut level_shift candidates from 85/23/5 to 8/1/0. It does NOT
    # remove flash-flood onsets, which are sharp and sustained too (probed in
    # scratchpad/tune_level_shift.py; see the module docstring).
    shift_min_sharpness: float = 0.5
    shift_persist_window: str = "12h"  # horizon for the before/after medians
    # NB: no gap settings. Gaps are labelled from the NaN mask at merge with no
    # length threshold (§5) — there is nothing here to tune.
    # grouping / presentation
    #: Merge flagged runs separated by less than this — but for **segment** types
    #: only (plateau, level_shift), where a detector legitimately marks one long
    #: event in patches. Spikes are never bridged: §6 defines a spike as one/few
    #: values far from neighbours, so bridging glues distinct spikes together and
    #: drags the normal rows between them into the label. At 2h on a 15-min grid
    #: that put 12-27% never-flagged filler inside spike candidates.
    segment_bridge: str = "2h"
    #: Keep only the worst N candidates per type, reporting the rest in
    #: ``CandidateSet.truncated``. **Off by default.** It used to default to 50,
    #: which silently made the cap — not the detectors — the binding constraint on
    #: recall: at the settings above a dataset proposes ~2,500 spike segments, so
    #: a cap of 50 discarded ~98% of them and dropped measured spike recall from
    #: ~97% to ~4%. A truncated candidate is not "found" in any usable sense; it
    #: never reaches the reviewer. This is a *presentation* limit for the review
    #: page, so it is now opt-in (``--max-per-type``) and the caller chooses how
    #: much of the queue to face.
    max_per_type: int | None = None


# --------------------------------------------------------------------------- types
@dataclass(frozen=True)
class Candidate:
    """One proposed anomaly segment awaiting a human decision."""

    candidate_id: str
    anomaly_type: str
    detectors: tuple[str, ...]
    start_pos: int  # inclusive row position in the gridded series
    end_pos: int  # exclusive
    start: pd.Timestamp  # timestamp of start_pos
    end: pd.Timestamp  # timestamp of the LAST included row (inclusive), so the
    #: exported start/end read the way a human expects and can be narrowed by hand
    n_rows: int
    duration_hours: float
    score: float  # severity, comparable within a type only
    v_min: float | None
    v_max: float | None
    note: str = ""

    def to_row(self) -> dict[str, Any]:
        """Flat dict for the candidates CSV."""
        return {
            "candidate_id": self.candidate_id,
            "anomaly_type": self.anomaly_type,
            "detectors": "|".join(self.detectors),
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "n_rows": self.n_rows,
            "duration_hours": round(self.duration_hours, 3),
            "score": round(self.score, 4),
            "v_min": self.v_min,
            "v_max": self.v_max,
            "note": self.note,
            "decision": "",  # filled in by the reviewer
        }


@dataclass
class CandidateSet:
    """Candidates for one series, plus the data and context needed to review them."""

    path: Path
    series: pd.Series  # gridded, NaN where missing
    candidates: list[Candidate]
    config: DetectConfig
    level_scale: float
    step_scale: float
    step_minutes: float
    truncated: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        """Candidates as a DataFrame in the candidates-CSV column order."""
        if not self.candidates:
            return pd.DataFrame(
                columns=[
                    "candidate_id", "anomaly_type", "detectors", "start", "end",
                    "n_rows", "duration_hours", "score", "v_min", "v_max",
                    "note", "decision",
                ]
            )
        return pd.DataFrame([c.to_row() for c in self.candidates])

    def counts(self) -> dict[str, int]:
        """Number of candidates per anomaly type."""
        out: dict[str, int] = {}
        for c in self.candidates:
            out[c.anomaly_type] = out.get(c.anomaly_type, 0) + 1
        return out

    def gap_totals(self) -> tuple[int, int]:
        """``(number of missing runs, number of missing rows)`` — all of them.

        Reported rather than reviewed: these are labelled wholesale by
        :func:`merge_decisions`, with no length threshold, per §5.
        """
        mask = self.series.isna().to_numpy()
        return len(_runs(mask)), int(mask.sum())

    def summary(self) -> str:
        """Human-readable proposal summary for the terminal."""
        lines = [
            f"{self.path.name}: {len(self.series):,} rows on a "
            f"{self.step_minutes:.0f}min grid "
            f"({self.series.isna().sum():,} missing)",
            f"  robust scale: level={self.level_scale:.4g}  step={self.step_scale:.4g}",
            f"  {len(self.candidates)} candidate(s) for review:",
        ]
        counts = self.counts()
        for t in REVIEW_TYPES:
            extra = self.truncated.get(t, 0)
            tail = f"  (+{extra} lower-scoring, dropped)" if extra else ""
            lines.append(f"    {t:<12} {counts.get(t, 0):>4}{tail}")

        n_runs, n_rows = self.gap_totals()
        lines.append(
            f"  gap: {n_runs:,} missing run(s), {n_rows:,} row(s) — labelled "
            "automatically at merge, not reviewed (§5)"
        )
        for tool, msg in self.errors.items():
            lines.append(f"  ! {tool} failed: {msg}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- helpers
def robust_scales(series: pd.Series) -> tuple[float, float]:
    """Return ``(level_scale, step_scale)`` — MAD-based sigmas of the series.

    ``level_scale`` describes how far values spread around their median;
    ``step_scale`` how far consecutive samples move. Both use the MAD
    (x1.4826) so a series that is *already* full of anomalies does not inflate
    its own thresholds the way a plain std would.
    """
    valid = series.dropna()
    if valid.empty:
        return 0.0, 0.0

    def _mad(x: pd.Series) -> float:
        if x.empty:
            return 0.0
        return float(1.4826 * (x - x.median()).abs().median())

    level = _mad(valid)
    step = _mad(valid.diff().dropna())
    # A heavily quantised series can have a zero MAD; fall back to a small
    # positive scale so thresholds stay finite and nothing flags everything.
    if level <= 0:
        level = float(valid.std(ddof=1) or 1.0)
    if step <= 0:
        step = max(level * 0.01, 1e-9)
    return level, step


def _median_step_minutes(index: pd.DatetimeIndex) -> float:
    if len(index) < 2:
        return 15.0
    diffs = index.to_series().diff().dropna()
    if diffs.empty:
        return 15.0
    return float(diffs.median().total_seconds() / 60.0)


def _rows_for(offset: str, step_minutes: float) -> int:
    """Convert an offset string to a whole number of rows on this grid."""
    minutes = pd.Timedelta(offset).total_seconds() / 60.0
    return max(1, int(round(minutes / max(step_minutes, 1e-9))))


def _runs(mask: np.ndarray, bridge: int = 0) -> list[tuple[int, int]]:
    """Contiguous True runs of ``mask`` as ``[start, end)`` positions.

    Runs separated by fewer than ``bridge`` False rows are merged, so one
    physical event that a detector flags patchily stays one candidate rather
    than becoming a dozen near-identical review prompts.
    """
    flat = np.asarray(mask, dtype=bool)
    if not flat.any():
        return []
    d = np.diff(np.concatenate([[0], flat.view(np.int8), [0]]))
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)

    merged: list[list[int]] = [[int(starts[0]), int(ends[0])]]
    for s, e in zip(starts[1:], ends[1:]):
        if s - merged[-1][1] <= bridge:
            merged[-1][1] = int(e)
        else:
            merged.append([int(s), int(e)])
    return [(a, b) for a, b in merged]


def _run_saqc(series: pd.Series, method: str, params: dict[str, Any]) -> pd.Series:
    """Apply one SaQC 2.8 method to a fresh object; return the flagged mask.

    One method per :class:`saqc.SaQC` instance, deliberately: with a single test
    applied there is no flag-attribution ambiguity, so we can read ``.flags``
    directly instead of walking the history (CLAUDE.md §7.1).
    """
    qc = saqc.SaQC(pd.DataFrame({FIELD: series}))
    out = getattr(qc, method)(FIELD, **params)
    flagged = out.flags[FIELD] > saqc.UNFLAGGED
    return flagged.reindex(series.index, fill_value=False).fillna(False).astype(bool)


# --------------------------------------------------------------------------- scoring
def _score_spike(values: pd.Series, seg: slice, step_scale: float, win: int) -> float:
    """How far the segment departs from its local baseline, in step sigmas."""
    lo = max(0, seg.start - win)
    hi = min(len(values), seg.stop + win)
    context = values.iloc[lo:hi]
    baseline = context.median()
    body = values.iloc[seg]
    if body.dropna().empty or not np.isfinite(baseline):
        return 0.0
    return float((body - baseline).abs().max() / max(step_scale, 1e-9))


def _shift_metrics(
    values: pd.Series, seg: slice, win: int
) -> tuple[float, float]:
    """``(net level change, sharpness)`` for a candidate level shift.

    ``net`` is the absolute difference between the median of the ``win`` rows
    before the edge and the ``win`` rows after it — the size of the step itself,
    in data units. ``sharpness`` is the single largest one-sample move inside the
    edge divided by ``net``: ~1.0 when the whole step happens in one sample (a
    recalibration / sensor swap), and small when the change is spread over many
    samples (a storm rising or falling limb — a gradual slope, not a step).

    This is the discriminator behind :attr:`DetectConfig.shift_min_sharpness`.
    It cannot separate a genuine step from a *flash-flood onset*, which is also
    sharp and sustained — no univariate test can (see the module docstring).
    """
    before = values.iloc[max(0, seg.start - win) : seg.start].median()
    after = values.iloc[seg.stop : seg.stop + win].median()
    if not (np.isfinite(before) and np.isfinite(after)):
        return 0.0, 0.0
    net = abs(after - before)
    # include one sample either side so the edge's own jump is inside the window
    edge = values.iloc[max(0, seg.start - 1) : seg.stop + 1]
    max_move = edge.diff().abs().max()
    sharpness = float(max_move / net) if net > 0 else 0.0
    return float(net), sharpness


def _score_shift(values: pd.Series, seg: slice, step_scale: float, win: int) -> float:
    """Size of the level change across the segment, in step sigmas."""
    net, _ = _shift_metrics(values, seg, win)
    return net / max(step_scale, 1e-9)


# --------------------------------------------------------------------------- detect
def _detect_spikes(
    series: pd.Series, cfg: DetectConfig, step_scale: float, errors: dict[str, str]
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, method, params in (
        (
            "flagUniLOF",
            "flagUniLOF",
            {"n": cfg.unilof_n, "thresh": cfg.unilof_thresh,
             "density": "auto", "slope_correct": True},
        ),
        (
            "flagZScore",
            "flagZScore",
            {"method": "modified", "window": cfg.zscore_window,
             "thresh": cfg.zscore_thresh,
             "min_residuals": cfg.zscore_min_residual_sigmas * step_scale},
        ),
    ):
        try:
            masks[name] = _run_saqc(series, method, params).to_numpy()
        except Exception as exc:  # a detector failing must not sink the run
            errors[name] = f"{type(exc).__name__}: {exc}"
    return masks


def _detect_plateaus(
    series: pd.Series, cfg: DetectConfig, step_scale: float, errors: dict[str, str]
) -> dict[str, np.ndarray]:
    masks: dict[str, np.ndarray] = {}
    for name, method, params in (
        (
            "flagConstants",
            "flagConstants",
            {"thresh": cfg.constants_thresh_sigmas * step_scale,
             "window": cfg.constants_window, "min_periods": 2},
        ),
        (
            "flagPlateau",
            "flagPlateau",
            {"min_length": cfg.plateau_min_length},
        ),
    ):
        try:
            masks[name] = _run_saqc(series, method, params).to_numpy()
        except Exception as exc:
            errors[name] = f"{type(exc).__name__}: {exc}"
    return masks


def _detect_jumps(
    series: pd.Series, cfg: DetectConfig, step_scale: float, errors: dict[str, str]
) -> dict[str, np.ndarray]:
    try:
        mask = _run_saqc(
            series,
            "flagJumps",
            {"thresh": cfg.jumps_thresh_sigmas * step_scale,
             "window": cfg.jumps_window},
        ).to_numpy()
        return {"flagJumps": mask}
    except Exception as exc:
        errors["flagJumps"] = f"{type(exc).__name__}: {exc}"
        return {}


def find_candidates(
    path: str | Path,
    *,
    config: DetectConfig | None = None,
    value_col: str = VALUE_COL,
) -> CandidateSet:
    """Load a series and propose reviewable candidates (:data:`REVIEW_TYPES`).

    The series is re-gridded onto its modal timestep first, so missing samples
    become explicit NaN rows. Those rows are excluded from every candidate mask
    (a NaN carries no evidence of a spike or a plateau) and are labelled as gaps
    wholesale by :func:`merge_decisions`.
    """
    path = Path(path)
    cfg = config or DetectConfig()

    df = load_series(path, value_col=value_col)
    df = reindex_to_grid(df, value_col=value_col)
    series = df.set_index(DATETIME_COL)[value_col].astype(float).sort_index()
    series.name = FIELD

    level_scale, step_scale = robust_scales(series)
    step_minutes = _median_step_minutes(pd.DatetimeIndex(series.index))
    segment_bridge = _rows_for(cfg.segment_bridge, step_minutes)
    context = _rows_for("24h", step_minutes)
    persist = _rows_for(cfg.shift_persist_window, step_minutes)
    jump_floor = cfg.jumps_thresh_sigmas * step_scale  # min net step, data units

    errors: dict[str, str] = {}
    per_type: dict[str, dict[str, np.ndarray]] = {
        "spike": _detect_spikes(series, cfg, step_scale, errors),
        "plateau": _detect_plateaus(series, cfg, step_scale, errors),
        "level_shift": _detect_jumps(series, cfg, step_scale, errors),
    }

    is_nan = series.isna().to_numpy()
    values = series
    n = len(series)
    proposals: dict[str, list[Candidate]] = {t: [] for t in REVIEW_TYPES}

    for atype, masks in per_type.items():
        if not masks:
            continue
        union = np.zeros(n, dtype=bool)
        for m in masks.values():
            union |= m
        # A NaN row carries no evidence of a spike or a plateau, and letting one
        # in would merge unrelated events across a dropout.
        union &= ~is_nan

        # Point events are never bridged; segment events are (see segment_bridge).
        bridge = 0 if atype == "spike" else segment_bridge
        for a, b in _runs(union, bridge=bridge):
            seg = slice(a, b)
            detectors = tuple(
                name for name, m in masks.items() if m[a:b].any()
            )
            if atype == "spike":
                score = _score_spike(values, seg, step_scale, context)
            elif atype == "level_shift":
                # Drop gradual slopes: keep only a sharp, threshold-sized step.
                net, sharpness = _shift_metrics(values, seg, persist)
                if net < jump_floor or sharpness < cfg.shift_min_sharpness:
                    continue
                score = net / max(step_scale, 1e-9)
            else:  # plateau — duration is the severity
                score = (b - a) * step_minutes / 60.0

            body = values.iloc[seg].dropna()
            proposals[atype].append(
                Candidate(
                    candidate_id="",  # assigned after ranking
                    anomaly_type=atype,
                    detectors=detectors,
                    start_pos=a,
                    end_pos=b,
                    start=series.index[a],
                    end=series.index[b - 1],
                    n_rows=b - a,
                    duration_hours=(b - a) * step_minutes / 60.0,
                    score=score,
                    v_min=float(body.min()) if not body.empty else None,
                    v_max=float(body.max()) if not body.empty else None,
                )
            )

    # Rank within type, cap, then id in presentation order.
    ordered: list[Candidate] = []
    truncated: dict[str, int] = {}
    for atype in REVIEW_TYPES:
        items = sorted(proposals[atype], key=lambda c: c.score, reverse=True)
        if cfg.max_per_type is not None and len(items) > cfg.max_per_type:
            truncated[atype] = len(items) - cfg.max_per_type
            items = items[: cfg.max_per_type]
        items.sort(key=lambda c: c.start_pos)
        for i, c in enumerate(items, start=1):
            ordered.append(
                Candidate(**{**c.__dict__, "candidate_id": f"{atype}_{i:03d}"})
            )

    return CandidateSet(
        path=path,
        series=series,
        candidates=ordered,
        config=cfg,
        level_scale=level_scale,
        step_scale=step_scale,
        step_minutes=step_minutes,
        truncated=truncated,
        errors=errors,
    )


# --------------------------------------------------------------------------- merge
def _decided_spans(
    result: CandidateSet, decisions: pd.DataFrame, chosen: set[str]
) -> dict[str, tuple[int, int]]:
    """Row span ``[start, end)`` to label for each chosen candidate.

    Defaults to the candidate's full extent, but honours a narrowed
    ``start``/``end`` from the decisions frame (``end`` inclusive). A narrowed
    span must lie inside the original: the reviewer is trimming a proposal, not
    inventing a new one, and a stray timestamp would otherwise label rows nobody
    ever looked at.
    """
    by_id = {c.candidate_id: c for c in result.candidates}
    index = result.series.index
    spans = {cid: (by_id[cid].start_pos, by_id[cid].end_pos)
             for cid in chosen if cid in by_id}

    if not {"start", "end"} <= set(decisions.columns):
        return spans

    for row in decisions.itertuples():
        cid = str(row.candidate_id)
        c = by_id.get(cid)
        if cid not in chosen or c is None:
            continue
        start, end = pd.to_datetime(row.start), pd.to_datetime(row.end)
        if pd.isna(start) or pd.isna(end):
            continue
        a = int(index.searchsorted(start, side="left"))
        b = int(index.searchsorted(end, side="right"))  # end is inclusive
        if a >= b:
            raise ValueError(f"{cid}: start {start} is after end {end}")
        if a < c.start_pos or b > c.end_pos:
            raise ValueError(
                f"{cid}: reviewed span {start}..{end} falls outside the candidate "
                f"({c.start}..{c.end}). A decision may narrow a candidate, never "
                "extend it — re-run 'detect' if you need a different span."
            )
        spans[cid] = (a, b)
    return spans


def merge_decisions(
    result: CandidateSet,
    decisions: pd.DataFrame,
    *,
    include_unsure: bool = False,
) -> pd.DataFrame:
    """Fold reviewed decisions back into a §5-shaped labels frame.

    ``decisions`` is the CSV exported by the review page: at minimum
    ``candidate_id`` and ``decision`` (``anomaly`` / ``normal`` / ``unsure`` /
    empty). Only ``anomaly`` rows become labels — plus ``unsure`` when
    ``include_unsure`` is set, which is useful for a second pass.

    **Gaps are added here, not reviewed.** Every missing run in the series is
    labelled ``anomaly_type=gap`` regardless of length, which is what §5
    requires — "**Every** missing run is labelled ``is_anomaly=True,
    anomaly_type=gap``". No length threshold applies: on 06818000, 93% of
    missing rows sit in runs shorter than an hour, and dropping those would
    under-label the series by an order of magnitude.

    ``source`` is always ``natural`` and ``true_value`` always NaN: these
    anomalies were already in the record, so no uncontaminated value is known
    for them. That is exactly the §5 case where imputation cannot be scored.

    **Narrowed spans are honoured.** A detector marks a window; only part of it
    may actually be the anomaly. If ``decisions`` carries ``start``/``end``
    columns (the export always does, and they can be edited by hand), the
    labelled rows are that span rather than the candidate's full extent —
    validated to lie inside the original, so a typo cannot silently label
    unrelated rows. ``end`` is **inclusive**.

    Overlap resolution: among reviewed types the *earlier* entry in
    :data:`REVIEW_TYPES` wins, so a confirmed spike inside a confirmed
    level_shift stays labelled ``spike``. ``gap`` is applied last and beats
    everything — a missing value is missing whatever else a detector thought was
    happening there.
    """
    if "candidate_id" not in decisions.columns or "decision" not in decisions.columns:
        raise ValueError(
            "decisions frame needs 'candidate_id' and 'decision' columns; "
            f"got {list(decisions.columns)}"
        )

    keep = {"anomaly"} | ({"unsure"} if include_unsure else set())
    chosen = {
        str(r.candidate_id)
        for r in decisions.itertuples()
        if str(r.decision).strip().lower() in keep
    }

    index = result.series.index
    n = len(index)
    is_anomaly = np.zeros(n, dtype=bool)
    anomaly_type = np.full(n, "", dtype=object)

    by_id = {c.candidate_id: c for c in result.candidates}
    spans = _decided_spans(result, decisions, chosen)

    # Apply lowest-priority type first so higher-priority types overwrite it.
    # Positional slices throughout: `.loc` on the default RangeIndex would treat
    # end_pos as inclusive and label one row too many.
    for atype in reversed(REVIEW_TYPES):
        for cid in chosen:
            c = by_id.get(cid)
            if c is None or c.anomaly_type != atype:
                continue
            a, b = spans[cid]
            is_anomaly[a:b] = True
            anomaly_type[a:b] = atype

    # Gaps last, and unconditionally: no review, no length threshold (§5).
    missing = result.series.isna().to_numpy()
    is_anomaly[missing] = True
    anomaly_type[missing] = "gap"

    return pd.DataFrame(
        {
            DATETIME_COL: index,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "true_value": np.nan,
            "source": np.where(is_anomaly, "natural", ""),
        }
    ).reset_index(drop=True)
