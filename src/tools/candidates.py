"""Propose candidate anomalies in a "clean" series, for human review.

The `data/clean/` segments were picked by eye, so they may still contain real
anomalies. This module runs the §7 SaQC 2.8 detectors over such a series at
deliberately *sensitive* settings, groups the flagged rows into contiguous
**segments**, and ranks them. Nothing here decides anything: every segment is a
*candidate* that a human confirms or rejects in the review page built by
``src.tools.review``.

Because the aim is recall (miss nothing) rather than precision, the output is
expected to contain false positives. That is what the review step is for.

Per-type notes
--------------
- **spike** — ``flagUniLOF`` + ``flagZScore``; a segment found by both is ranked
  higher via its score, but either alone is enough to propose it.
- **plateau** — ``flagConstants`` (stuck at any level) + ``flagPlateau``
  (offset segment). These find different failures, so both run (CLAUDE.md §7.1).
- **level_shift** — ``flagJumps``. It marks the *transition*, not the shifted
  window, so a candidate's span is the transition ± context.
- **gap** — ``flagNAN`` on the re-gridded series, then runs shorter than
  ``min_gap`` are dropped: isolated dropouts are not gaps (§9).

Drift is **not** proposed: the drift track is shelved (CLAUDE.md §9.2). SaQC 2.8
ships no univariate drift detector, so it had to be a local heuristic, it was the
weakest proposer here, and seasonal turbidity swells are indistinguishable from
fouling anyway.

Thresholds in data units (``flagJumps.thresh``, ``flagConstants.thresh``) are
derived from the series' own robust scale, so the defaults transfer across
gauges on very different turbidity ranges. Every one can be overridden.

Usage
-----
    from src.tools.candidates import find_candidates

    result = find_candidates("data/clean/06818000_clean_20240515_20250123.csv")
    print(result.summary())

See ``src.tools.review`` for the CLI and the labelling page.
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

#: Order candidates are presented in. Point-like types first — they are the
#: quickest to judge, so the reviewer builds a feel for the series before
#: reaching the slow, ambiguous segment types.
TYPE_ORDER: tuple[str, ...] = ("spike", "plateau", "level_shift", "gap")

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
    unilof_n: int = 20
    unilof_thresh: float = 1.5
    zscore_window: str = "12h"
    zscore_thresh: float = 10.0
    # A residual floor in data units. Without it, a 12h window of quantised
    # readings has a near-zero MAD and every wiggle scores an enormous modified
    # z: on 11501000 (0.1 NTU quantisation) flagZScore stuck at ~195 segments
    # even at thresh=30. At 3 step-sigmas that falls to 7. Probed, see
    # scratchpad/tune_zscore.py.
    zscore_min_residual_sigmas: float = 3.0  # x step scale
    # plateau
    constants_window: str = "3h"
    constants_thresh_sigmas: float = 0.1  # x step scale; must stay << noise sd (§7.1)
    plateau_min_length: str = "1h"
    # level shift
    jumps_window: str = "12h"
    jumps_thresh_sigmas: float = 24.0  # x step scale
    # gap
    min_gap: str = "1h"  # shorter NaN runs are dropouts, not gaps (§9)
    # grouping / presentation
    bridge: str = "2h"  # merge flagged runs separated by less than this
    max_per_type: int = 50  # keep the worst N per type; the rest are reported


# --------------------------------------------------------------------------- types
@dataclass(frozen=True)
class Candidate:
    """One proposed anomaly segment awaiting a human decision."""

    candidate_id: str
    anomaly_type: str
    detectors: tuple[str, ...]
    start_pos: int  # inclusive row position in the gridded series
    end_pos: int  # exclusive
    start: pd.Timestamp
    end: pd.Timestamp
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
        for t in TYPE_ORDER:
            if t in counts:
                extra = self.truncated.get(t, 0)
                tail = f"  (+{extra} lower-scoring, dropped)" if extra else ""
                lines.append(f"    {t:<12} {counts[t]:>4}{tail}")
        for t in TYPE_ORDER:
            if t not in counts:
                lines.append(f"    {t:<12} {0:>4}")
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


def _score_shift(values: pd.Series, seg: slice, step_scale: float, win: int) -> float:
    """Size of the level change across the segment, in step sigmas."""
    before = values.iloc[max(0, seg.start - win) : seg.start].median()
    after = values.iloc[seg.stop : seg.stop + win].median()
    if not (np.isfinite(before) and np.isfinite(after)):
        return 0.0
    return float(abs(after - before) / max(step_scale, 1e-9))


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


def _detect_gaps(
    series: pd.Series, cfg: DetectConfig, errors: dict[str, str]
) -> dict[str, np.ndarray]:
    try:
        return {"flagNAN": _run_saqc(series, "flagNAN", {}).to_numpy()}
    except Exception as exc:
        errors["flagNAN"] = f"{type(exc).__name__}: {exc}"
        return {}


def find_candidates(
    path: str | Path,
    *,
    config: DetectConfig | None = None,
    value_col: str = VALUE_COL,
) -> CandidateSet:
    """Load a series and propose candidate anomalies of every type.

    The series is re-gridded onto its modal timestep first, so missing samples
    become explicit NaN rows and gap detection has something to find (§9).
    """
    path = Path(path)
    cfg = config or DetectConfig()

    df = load_series(path, value_col=value_col)
    df = reindex_to_grid(df, value_col=value_col)
    series = df.set_index(DATETIME_COL)[value_col].astype(float).sort_index()
    series.name = FIELD

    level_scale, step_scale = robust_scales(series)
    step_minutes = _median_step_minutes(pd.DatetimeIndex(series.index))
    bridge = _rows_for(cfg.bridge, step_minutes)
    context = _rows_for("24h", step_minutes)
    min_gap_rows = _rows_for(cfg.min_gap, step_minutes)

    errors: dict[str, str] = {}
    per_type: dict[str, dict[str, np.ndarray]] = {
        "spike": _detect_spikes(series, cfg, step_scale, errors),
        "plateau": _detect_plateaus(series, cfg, step_scale, errors),
        "level_shift": _detect_jumps(series, cfg, step_scale, errors),
        "gap": _detect_gaps(series, cfg, errors),
    }

    is_nan = series.isna().to_numpy()
    values = series
    n = len(series)
    proposals: dict[str, list[Candidate]] = {t: [] for t in TYPE_ORDER}

    for atype, masks in per_type.items():
        if not masks:
            continue
        union = np.zeros(n, dtype=bool)
        for m in masks.values():
            union |= m
        # Only gap candidates may sit on missing rows; for every other type a
        # NaN row carries no evidence, and letting them in merges unrelated
        # events across dropouts.
        if atype != "gap":
            union &= ~is_nan

        for a, b in _runs(union, bridge=0 if atype == "gap" else bridge):
            if atype == "gap" and (b - a) < min_gap_rows:
                continue  # isolated dropout, not a gap (§9)

            seg = slice(a, b)
            detectors = tuple(
                name for name, m in masks.items() if m[a:b].any()
            )
            if atype == "spike":
                score = _score_spike(values, seg, step_scale, context)
            elif atype == "level_shift":
                score = _score_shift(values, seg, step_scale, context)
            else:  # plateau, gap — duration is the severity
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
                    end=series.index[min(b, n - 1)],
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
    for atype in TYPE_ORDER:
        items = sorted(proposals[atype], key=lambda c: c.score, reverse=True)
        if len(items) > cfg.max_per_type:
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

    ``source`` is always ``natural`` and ``true_value`` always NaN: these
    anomalies were already in the record, so no uncontaminated value is known
    for them. That is exactly the §5 case where imputation cannot be scored.

    Overlapping confirmed candidates of different types are resolved by
    :data:`TYPE_ORDER` — the earlier type wins, so a spike sitting inside a
    confirmed level_shift stays labelled ``spike``.
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
    # Apply lowest-priority type first so higher-priority types overwrite it.
    # Positional slices throughout: `.loc` on the default RangeIndex would treat
    # end_pos as inclusive and label one row too many.
    for atype in reversed(TYPE_ORDER):
        for cid in chosen:
            c = by_id.get(cid)
            if c is None or c.anomaly_type != atype:
                continue
            is_anomaly[c.start_pos : c.end_pos] = True
            anomaly_type[c.start_pos : c.end_pos] = atype

    return pd.DataFrame(
        {
            DATETIME_COL: index,
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "true_value": np.nan,
            "source": np.where(is_anomaly, "natural", ""),
        }
    ).reset_index(drop=True)
