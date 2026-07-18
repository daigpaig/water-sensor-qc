"""Synthetic anomaly injection into clean segments.

Injects the five failure types (spike, plateau, level_shift, gap, drift) into a
clean base series, records ground-truth labels, and writes the maintenance
schedule that drift correction needs. Everything is driven by a fixed seed, so a
given (base, level, seed) always reproduces byte-identical output.

Design decisions worth knowing before reading the code
------------------------------------------------------
**Drift recurs; it is not a once-per-series event.** Drift is fouling and
calibration wander, which accumulates until someone cleans or recalibrates the
sensor and then starts over. So a series carries one drift episode per
maintenance cycle, each ramping from 0 to its full offset and resetting to 0 at
the maintenance event that ends it. SaQC 2.8's ``correctDrift`` assumes exactly
this: its ``maintenance_field`` is a series whose index is the start of each
maintenance event and whose values are that event's end, and it corrects drift
*between* consecutive events. We emit that schedule (§5) so ``correct_drift``
has the support points it requires.

**The contamination level scales the point-like types only.** ``level`` sets the
share of rows occupied by spike / plateau / gap, where "percent of rows" is a
natural unit. Drift is driven by *maintenance interval* instead — a shorter
interval means a less-maintained sensor, which is the story the three levels
tell — and episodes keep physically realistic durations (weeks, per §6). Their
row-share is therefore a consequence, reported in the manifest rather than
targeted, and it is large: level 3 lands near 50% total anomalous. The headline
``point_pct`` understates the total by design; ``evaluate.py`` must read per-type
counts from the labels, never infer them from the headline number.

**Magnitudes are sized by a robust local scale, capped.** See ``_local_scale``: a
rolling standard deviation is inflated by the storm events we must preserve, and
sizing anomalies with it produced a 2,973 FNU "spike" on a river whose median is
7.2 FNU — absurd, and trivially detectable in a way that would flatter the
metrics.

**Level shift is injected as a bounded window, not a permanent step.** A literal
permanent step would either label every subsequent row anomalous (one mid-series
shift = ~50% contamination) or leave post-step rows with ``true_value != value``
while marked not-anomalous, which the §5 label contract cannot express. So the
offset applies over a finite window and the series returns to its true level,
consistent with how drift is handled.

**Natural gaps are labelled too.** The clean bases were selected for long
unbroken stretches but still contain small natural gaps (the pull filter allows
up to 3h). Every missing run is labelled ``is_anomaly=True, anomaly_type=gap``,
including pre-existing ones, so the ground truth is honest rather than
pretending the base is pristine. The ``source`` column separates them: only
``injected`` gaps have a known ``true_value``, so imputation RMSE/MAE (§10) is
scored on those alone, while both count for detection precision/recall.

**Segment anomalies may span isolated dropouts; point anomalies may not.** Real
records carry many single-sample dropouts, and a stuck sensor or a drifting one
spans them without difficulty — so plateau, level_shift and drift place across
runs of up to ``SPANNABLE_DROPOUT_ROWS`` missing samples, leaving those rows
missing (and labelled ``gap``/``natural``). Spikes and injected gaps need real
readings in every row they touch: a spike displaces a value, and an injected gap
must delete a value we know, or its ``true_value`` would be unknown and its
``injected`` source a lie.

CLI
---
    # Inject all three levels into every clean base (the default):
    python -m src.inject

    # One base, one level, custom seed:
    python -m src.inject --input data/clean/11501000_clean_20231226_20240803.csv \\
        --levels 2 --seed 7

    # See what would be written, without writing it:
    python -m src.inject --dry-run

Usage from Python
-----------------
    from src.inject import inject_series, load_base

    base = load_base("data/clean/11501000_clean_20231226_20240803.csv")
    result = inject_series(base, level=2, seed=42, name="11501000_l2")
    result.data      # datetime/value frame with anomalies
    result.labels    # datetime/is_anomaly/anomaly_type/true_value/source
    result.maintenance  # start/end frame for correctDrift
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.inspect_data import (
    DATETIME_COL,
    VALUE_COL,
    ContractError,
    load_series,
    reindex_to_grid,
)

# ---------------------------------------------------------------------------
# Paths + contract vocabulary (CLAUDE.md §5)
# ---------------------------------------------------------------------------
DEFAULT_CLEAN_DIR = Path("data/clean")
DEFAULT_OUTDIR = Path("data/injected")
DEFAULT_SEED = 42

SOURCE_INJECTED = "injected"
SOURCE_NATURAL = "natural"

SPIKE = "spike"
PLATEAU = "plateau"
LEVEL_SHIFT = "level_shift"
GAP = "gap"
DRIFT = "drift"

# Turbidity is a physical non-negative quantity; injected offsets clamp here.
VALUE_FLOOR = 0.0

# ---------------------------------------------------------------------------
# Injection parameters
#
# Durations are expressed in real time and converted to row counts using the
# base's own timestep, so these stay meaningful if we ever inject into an hourly
# series rather than a 15-min one.
# ---------------------------------------------------------------------------
# Spike: one to a few samples thrown far off by debris or an air bubble.
SPIKE_LEN_ROWS = (1, 3)
# Sampled *log-uniformly* over this range (see ``_loguniform``), not uniformly:
# real spikes span an order of magnitude — most are modest, a few are enormous —
# so a flat U(4, 10) made every spike a similar, uniformly-large size and gave the
# detector an unrealistically easy, narrow target. Log-uniform over a wider band
# puts many spikes just above the local noise (2-3x) and a long tail of big ones
# (up to 15x), which is both more realistic and a harder test of recall.
SPIKE_MAGNITUDE_SCALE = (2.0, 15.0)
SPIKE_UPWARD_PROB = 0.8  # debris/fouling pushes turbidity readings up far more often

# Plateau: the sensor sticks and repeats one reading.
PLATEAU_DURATION_HOURS = (2.0, 12.0)

# Gap: a dropout. §6 imputes short gaps only, so we keep injected gaps short.
GAP_DURATION_HOURS = (0.5, 6.0)

# Level shift: a bounded offset window (see the module docstring).
LEVEL_SHIFT_DURATION_HOURS = (6.0, 72.0)
LEVEL_SHIFT_MAGNITUDE_SCALE = (2.0, 5.0)

# Drift: slow creep over weeks, ending at a maintenance reset (§6).
DRIFT_DURATION_DAYS = (14.0, 28.0)
DRIFT_MAGNITUDE_SCALE = (1.0, 3.0)
DRIFT_UPWARD_PROB = 0.85  # fouling makes the reading creep up, not down
MAINTENANCE_DURATION_HOURS = (1.0, 4.0)

# How the point budget splits across the point-like types, by row count.
POINT_BUDGET_WEIGHTS: dict[str, float] = {SPIKE: 0.2, PLATEAU: 0.4, GAP: 0.4}

# How far under target the point budget may land before the dataset is flagged as
# not carrying its nominal level (see ``point_budget_met`` in the manifest).
# Relative, not absolute: a flat 0.5-point tolerance is simultaneously too tight
# at level 3 (12% target) and too loose at level 1, where it would wave through a
# dataset carrying a sixth less contamination than it claims.
POINT_BUDGET_TOLERANCE_FRAC = 0.10

# Keep injected segments this many rows clear of each other, so anomalies stay
# individually attributable.
PLACEMENT_MARGIN_ROWS = 4

# A natural dropout this short does not block a *segment* anomaly from spanning
# it. Real records are full of isolated single-sample dropouts — 06818000 has
# ~2,700 of them, 2,216 exactly one sample long, with the longest run in 253 days
# being 10 samples (2.5h) — and a sensor stuck for 12 hours with one absent
# sample mid-window is entirely plausible. Refusing to span them is what an
# earlier version did, and it left only 16.8% of that series usable (vs 93.3%
# here), starving level 3 of plateaus and level shifts entirely. Runs longer than
# this are real outages and do block placement.
SPANNABLE_DROPOUT_ROWS = 2

# Rolling window (in rows) for the local scale used to size magnitudes.
LOCAL_SCALE_WINDOW_ROWS = 192  # 2 days at 15-min

# The local scale is a rolling MAD, clamped to a band around the series' global
# MAD. Both bounds matter: without the floor, a quiet stretch of a clear river
# gets anomalies too small to see; without the cap, a storm's variability sizes
# the anomaly and 03447687 receives a 3,000 FNU "spike" on a river that peaks at
# 1,000. See the module docstring on why MAD rather than standard deviation.
LOCAL_SCALE_FLOOR_K = 0.1
LOCAL_SCALE_CAP_K = 3.0


@dataclass(frozen=True)
class ContaminationLevel:
    """One of the three contamination levels (CLAUDE.md §9).

    ``point_pct`` is the share of rows given to spike/plateau/gap. Drift is set by
    ``maintenance_interval_days`` — how often the sensor gets serviced — and
    level_shift by count. See the module docstring.

    Drift is deliberately parameterised by *interval* rather than by a count of
    episodes. Maintenance is periodic, so a longer record simply contains more
    cycles; fixing the count instead would make "level 3" mean 37% drift on a
    220-day base but 16% on a 513-day one, and the levels would not be comparable
    across datasets. With an interval, drift's row-share is roughly
    ``DRIFT_DURATION_DAYS / maintenance_interval_days`` on any base.
    """

    level: int
    name: str
    point_pct: float
    maintenance_interval_days: float
    n_level_shifts: int


LEVELS: dict[int, ContaminationLevel] = {
    1: ContaminationLevel(
        1, "low", point_pct=3.0, maintenance_interval_days=200.0, n_level_shifts=1
    ),
    2: ContaminationLevel(
        2, "medium", point_pct=7.0, maintenance_interval_days=100.0, n_level_shifts=2
    ),
    3: ContaminationLevel(
        3, "high", point_pct=12.0, maintenance_interval_days=55.0, n_level_shifts=3
    ),
}


@dataclass(frozen=True)
class Segment:
    """One injected anomaly, as a half-open row range ``[start_idx, end_idx)``."""

    anomaly_type: str
    start_idx: int
    end_idx: int
    params: dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> int:
        return self.end_idx - self.start_idx


@dataclass(frozen=True)
class MaintenanceEvent:
    """A maintenance visit, as a half-open row range. Ends a drift episode."""

    start_idx: int
    end_idx: int


@dataclass
class InjectionResult:
    """Everything one (base, level, seed) injection produces."""

    name: str
    level: ContaminationLevel
    seed: int
    data: pd.DataFrame
    labels: pd.DataFrame
    maintenance: pd.DataFrame
    segments: list[Segment]
    manifest: dict[str, Any]


# ---------------------------------------------------------------------------
# The five injectors
#
# Each takes a values array and an explicit geometry, returns a new array plus
# the Segment describing what it did, and never consults a random number
# generator — placement and magnitude are the orchestrator's job. That keeps
# each one deterministic and testable in isolation.
# ---------------------------------------------------------------------------
def _validate_window(values: np.ndarray, start: int, length: int, kind: str) -> None:
    if length < 1:
        raise ValueError(f"{kind}: length must be >= 1 (got {length}).")
    if start < 0 or start + length > len(values):
        raise ValueError(
            f"{kind}: window [{start}, {start + length}) does not fit a series of "
            f"length {len(values)}."
        )


def inject_spike(
    values: np.ndarray,
    start: int,
    *,
    magnitude: float,
    length: int = 1,
) -> tuple[np.ndarray, Segment]:
    """Throw ``length`` samples off by ``magnitude`` (signed, absolute units).

    A spike is one or a few values far from their neighbours — debris passing the
    optic, an air bubble, a power glitch. The result is clamped at
    ``VALUE_FLOOR`` so a large downward spike cannot make turbidity negative.
    """
    _validate_window(values, start, length, "inject_spike")
    if magnitude == 0:
        raise ValueError("inject_spike: magnitude must be non-zero.")

    out = values.astype(float).copy()
    stop = start + length
    out[start:stop] = np.maximum(out[start:stop] + magnitude, VALUE_FLOOR)
    return out, Segment(SPIKE, start, stop, {"magnitude": float(magnitude)})


def inject_plateau(
    values: np.ndarray,
    start: int,
    *,
    length: int,
    level: float | None = None,
) -> tuple[np.ndarray, Segment]:
    """Freeze the series at one value for ``length`` rows (a stuck sensor).

    ``level`` defaults to the reading at ``start``: the sensor sticks at whatever
    it last saw. Passing an explicit level covers the case where it sticks at a
    rail instead.

    Rows already missing stay missing. A stuck sensor and a dropped sample are two
    independent failures, so freezing the reading must not invent data where the
    logger recorded none — otherwise those rows would hold a plateau value while
    the labels still (correctly) call them a gap.
    """
    _validate_window(values, start, length, "inject_plateau")

    out = values.astype(float).copy()
    stop = start + length
    if level is None:
        level = float(out[start])
    if not np.isfinite(level):
        raise ValueError(
            f"inject_plateau: level at row {start} is not finite ({level}); "
            "cannot stick a sensor at NaN."
        )
    window = out[start:stop]
    out[start:stop] = np.where(np.isnan(window), np.nan, level)
    return out, Segment(PLATEAU, start, stop, {"level": float(level)})


def inject_level_shift(
    values: np.ndarray,
    start: int,
    *,
    length: int,
    magnitude: float,
) -> tuple[np.ndarray, Segment]:
    """Offset a bounded window by ``magnitude``, then return to the true level.

    Bounded rather than permanent so the whole shifted window can be labelled
    without one event consuming the series — see the module docstring.
    """
    _validate_window(values, start, length, "inject_level_shift")
    if magnitude == 0:
        raise ValueError("inject_level_shift: magnitude must be non-zero.")

    out = values.astype(float).copy()
    stop = start + length
    out[start:stop] = np.maximum(out[start:stop] + magnitude, VALUE_FLOOR)
    return out, Segment(LEVEL_SHIFT, start, stop, {"magnitude": float(magnitude)})


def inject_gap(
    values: np.ndarray,
    start: int,
    *,
    length: int,
) -> tuple[np.ndarray, Segment]:
    """Blank ``length`` rows to NaN (a dropout).

    The caller keeps the original values in the label file's ``true_value``, which
    is what makes injected gaps scoreable for imputation error while natural gaps
    are not.
    """
    _validate_window(values, start, length, "inject_gap")

    out = values.astype(float).copy()
    stop = start + length
    out[start:stop] = np.nan
    return out, Segment(GAP, start, stop, {})


def inject_drift(
    values: np.ndarray,
    start: int,
    *,
    length: int,
    magnitude: float,
    model: str = "linear",
) -> tuple[np.ndarray, Segment]:
    """Ramp a growing offset across a window, reaching ``magnitude`` at the end.

    The offset starts at 0 and grows to ``magnitude`` on the final row, which is
    where the maintenance reset lands — so the series is unaffected before the
    episode and back to truth after it. ``model`` selects a ``linear`` ramp or an
    ``exponential`` one (fouling that accelerates); both are shapes SaQC 2.8's
    ``correctDrift`` can model.
    """
    _validate_window(values, start, length, "inject_drift")
    if magnitude == 0:
        raise ValueError("inject_drift: magnitude must be non-zero.")
    if model not in {"linear", "exponential"}:
        raise ValueError(f"inject_drift: model must be linear|exponential (got {model!r}).")

    # A 1-row drift has no ramp to speak of; treat it as reaching full offset.
    if length == 1:
        ramp = np.array([1.0])
    else:
        t = np.linspace(0.0, 1.0, length)
        if model == "linear":
            ramp = t
        else:
            k = 3.0  # curvature; fouling creeps slowly then accelerates
            ramp = (np.expm1(k * t)) / np.expm1(k)

    out = values.astype(float).copy()
    stop = start + length
    out[start:stop] = np.maximum(out[start:stop] + ramp * magnitude, VALUE_FLOOR)
    return out, Segment(
        DRIFT, start, stop, {"magnitude": float(magnitude), "model": model}
    )


# ---------------------------------------------------------------------------
# Placement helpers
# ---------------------------------------------------------------------------
def _rows_per(td: pd.Timedelta, step: pd.Timedelta) -> int:
    """Convert a duration to a row count on this series' grid (at least 1)."""
    return max(1, int(round(td / step)))


def _robust_scale(x: np.ndarray) -> float:
    """Global MAD, rescaled to be comparable with a standard deviation."""
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return 1.0
    mad = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)
    if np.isfinite(mad) and mad > 0:
        return mad
    # A base with no spread (or spread only in its tails) still needs a non-zero
    # scale, or every injected magnitude would be zero and rejected downstream.
    sd = float(np.std(finite))
    return sd if np.isfinite(sd) and sd > 0 else 1.0


def _local_scale(values: np.ndarray, window: int) -> np.ndarray:
    """Per-row local variability, used to size injected magnitudes.

    A rolling *median absolute deviation*, not a rolling standard deviation.
    Standard deviation is inflated by exactly the events we must preserve: on a
    flashy river the sd inside a storm is enormous, so a "4-10x local sd" spike
    injected there becomes physically absurd (2,973 FNU on a series whose median
    is 7.2 and whose max is 1,000) and trivially detectable, which would flatter
    the detection metrics rather than test them.

    Clamped to ``[FLOOR_K, CAP_K] * global MAD``: the floor keeps anomalies
    visible in quiet stretches, the cap keeps storms from sizing them.
    """
    s = pd.Series(values)
    min_periods = max(2, window // 4)
    med = s.rolling(window, center=True, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(
        window, center=True, min_periods=min_periods
    ).median() * 1.4826

    global_mad = _robust_scale(values)
    filled = mad.bfill().ffill().fillna(global_mad).to_numpy(copy=True)
    filled[~np.isfinite(filled)] = global_mad
    return np.clip(filled, LOCAL_SCALE_FLOOR_K * global_mad, LOCAL_SCALE_CAP_K * global_mad)


def _free_runs(occupied: np.ndarray, need: int) -> tuple[np.ndarray, np.ndarray]:
    """Half-open ``[start, stop)`` runs of unoccupied rows at least ``need`` long."""
    free = (~occupied).astype(np.int8)
    edges = np.diff(np.concatenate(([0], free, [0])))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    keep = (stops - starts) >= need
    return starts[keep], stops[keep]


def _find_free_start(
    rng: np.random.Generator,
    occupied: np.ndarray,
    length: int,
    *,
    margin: int = PLACEMENT_MARGIN_ROWS,
) -> int | None:
    """Pick a start row uniformly among those where the window (plus margin) is free.

    Enumerates the free runs and samples over valid start positions directly,
    rather than guessing at random and rejecting. Rejection sampling looks simpler
    but silently fails on a fragmented series: 06818000 carries 14.3% natural gaps
    scattered through it, and a guess-and-check placer gives up long before the
    budget is spent even though >50% of the series is free.

    Returns None only when no run is genuinely long enough.
    """
    need = length + 2 * margin
    starts, stops = _free_runs(occupied, need)
    if len(starts) == 0:
        return None

    # Valid starts per run, so the choice is uniform over positions rather than
    # over runs — otherwise a 20-row gap would attract as many anomalies as a
    # 20,000-row clean stretch.
    widths = (stops - starts) - need + 1
    offset = int(rng.integers(0, int(widths.sum())))
    run = int(np.searchsorted(np.cumsum(widths), offset, side="right"))
    within = offset - int(np.cumsum(widths)[run] - widths[run])
    return int(starts[run]) + margin + within


def _mark(occupied: np.ndarray, start: int, length: int) -> None:
    occupied[start : start + length] = True


def _uniform_int(rng: np.random.Generator, bounds: tuple[int, int]) -> int:
    lo, hi = bounds
    return int(rng.integers(lo, hi + 1))


def _uniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    lo, hi = bounds
    return float(rng.uniform(lo, hi))


def _loguniform(rng: np.random.Generator, bounds: tuple[float, float]) -> float:
    """Sample uniformly in log-space, so the result spreads across orders of magnitude.

    Used for spike magnitudes: a plain uniform draw clusters every spike near the
    middle of its range, whereas real spikes are mostly small with a heavy tail of
    large ones. Log-uniform reproduces that spread — the median sits well below the
    arithmetic mean of the bounds, and large spikes stay rare rather than typical.
    """
    lo, hi = bounds
    if lo <= 0 or hi <= 0:
        raise ValueError(f"_loguniform: bounds must be positive (got {bounds}).")
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _signed(rng: np.random.Generator, magnitude: float, upward_prob: float) -> float:
    return magnitude if rng.random() < upward_prob else -magnitude


def _derive_seed(master_seed: int, level: int, name: str) -> np.random.Generator:
    """Per-(name, level) generator derived from the master seed.

    Derived rather than shared so that adding a level or a base does not shift
    the stream for the others. ``zlib.crc32`` because Python's ``hash`` on
    strings is salted per process and would break reproducibility.
    """
    return np.random.default_rng([master_seed, level, zlib.crc32(name.encode())])


# ---------------------------------------------------------------------------
# Base loading
# ---------------------------------------------------------------------------
def load_base(path: str | Path, *, value_col: str = VALUE_COL) -> pd.DataFrame:
    """Load a clean base CSV onto a regular grid, so natural gaps become NaN rows.

    The raw USGS feed omits gap rows entirely rather than NaN-filling them (see
    ``pull_usgs``), so re-gridding is what makes natural gaps visible — and they
    have to be visible before we can label them or avoid injecting on top of them.
    """
    df = load_series(path, value_col=value_col)
    return reindex_to_grid(df, value_col=value_col)


def _median_step(df: pd.DataFrame) -> pd.Timedelta:
    diffs = pd.Series(pd.DatetimeIndex(df[DATETIME_COL])).diff().dropna()
    if diffs.empty:
        raise ContractError("base series has no usable timestep (fewer than 2 rows).")
    step = diffs.median()
    if step <= pd.Timedelta(0):
        raise ContractError(f"base series has a non-positive timestep ({step}).")
    return step


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def _n_drift_episodes(n_rows: int, step: pd.Timedelta, interval_days: float) -> int:
    """How many maintenance cycles fit in this base, at this level's interval.

    At least one: a base too short for even a single cycle still gets one episode,
    so every dataset has drift positives to score (§10 needs per-type recall, which
    is undefined for a type with no positives). ``_place_drift_episodes`` raises if
    the base cannot fit even that.
    """
    span_days = n_rows * step / pd.Timedelta(days=1)
    return max(1, int(round(span_days / interval_days)))


def _place_drift_episodes(
    rng: np.random.Generator,
    n_rows: int,
    step: pd.Timedelta,
    n_episodes: int,
) -> list[tuple[int, int, int]]:
    """Lay out drift episodes as (start, length, maintenance_length) row triples.

    Episodes are spread one per equal block of the series rather than placed at
    random, because maintenance is roughly periodic in practice — a sensor gets
    visited on a schedule, and drift is what accrues between visits.
    """
    if n_episodes < 1:
        return []

    min_len = _rows_per(pd.Timedelta(days=DRIFT_DURATION_DAYS[0]), step)
    max_maint = _rows_per(pd.Timedelta(hours=MAINTENANCE_DURATION_HOURS[1]), step)
    block = n_rows // n_episodes

    # Refuse rather than silently shrink drift into something unrealistic: a base
    # too short for this level's drift budget is nonsensical input (§13).
    if block < min_len + max_maint + 2 * PLACEMENT_MARGIN_ROWS:
        span_days = n_rows * step / pd.Timedelta(days=1)
        raise ValueError(
            f"base is too short for {n_episodes} drift episode(s): {span_days:.0f} days "
            f"leaves {block * step / pd.Timedelta(days=1):.0f} days per episode, but a "
            f"realistic episode needs >= {DRIFT_DURATION_DAYS[0]:.0f} days plus a "
            f"maintenance window. Use a longer base or a lower level."
        )

    episodes: list[tuple[int, int, int]] = []
    for i in range(n_episodes):
        block_start = i * block
        block_end = block_start + block
        maint_len = _rows_per(
            pd.Timedelta(hours=_uniform(rng, MAINTENANCE_DURATION_HOURS)), step
        )
        # Longest episode that still leaves room for its maintenance window.
        room = block_end - block_start - maint_len - PLACEMENT_MARGIN_ROWS
        want = _rows_per(pd.Timedelta(days=_uniform(rng, DRIFT_DURATION_DAYS)), step)
        length = max(min_len, min(want, room))
        slack = room - length
        start = block_start + (int(rng.integers(0, slack)) if slack > 0 else 0)
        episodes.append((start, length, maint_len))
    return episodes


def _inject_drift_and_maintenance(
    rng: np.random.Generator,
    values: np.ndarray,
    occupied: np.ndarray,
    scale: np.ndarray,
    step: pd.Timedelta,
    interval_days: float,
) -> tuple[np.ndarray, list[Segment], list[MaintenanceEvent]]:
    segments: list[Segment] = []
    events: list[MaintenanceEvent] = []

    n_episodes = _n_drift_episodes(len(values), step, interval_days)
    for start, length, maint_len in _place_drift_episodes(rng, len(values), step, n_episodes):
        local = float(np.median(scale[start : start + length]))
        magnitude = _signed(
            rng, _uniform(rng, DRIFT_MAGNITUDE_SCALE) * local, DRIFT_UPWARD_PROB
        )
        model = "linear" if rng.random() < 0.5 else "exponential"
        values, seg = inject_drift(
            values, start, length=length, magnitude=magnitude, model=model
        )
        segments.append(seg)
        _mark(occupied, start, length)

        # The maintenance visit sits immediately after the episode: the drift is
        # reset here, which is what correctDrift calibrates against. We leave the
        # data itself untouched across the visit so cal_range has clean readings
        # on both sides.
        maint_start = seg.end_idx
        maint_stop = min(len(values), maint_start + maint_len)
        if maint_stop > maint_start:
            events.append(MaintenanceEvent(maint_start, maint_stop))
            _mark(occupied, maint_start, maint_stop - maint_start)

    return values, segments, events


def _inject_level_shifts(
    rng: np.random.Generator,
    values: np.ndarray,
    occupied: np.ndarray,
    scale: np.ndarray,
    step: pd.Timedelta,
    n_shifts: int,
) -> tuple[np.ndarray, list[Segment]]:
    segments: list[Segment] = []
    for _ in range(n_shifts):
        length = _rows_per(
            pd.Timedelta(hours=_uniform(rng, LEVEL_SHIFT_DURATION_HOURS)), step
        )
        start = _find_free_start(rng, occupied, length)
        if start is None:
            break
        local = float(np.median(scale[start : start + length]))
        magnitude = _signed(rng, _uniform(rng, LEVEL_SHIFT_MAGNITUDE_SCALE) * local, 0.5)
        values, seg = inject_level_shift(values, start, length=length, magnitude=magnitude)
        segments.append(seg)
        _mark(occupied, start, length)
    return values, segments


def _inject_point_anomalies(
    rng: np.random.Generator,
    values: np.ndarray,
    base: np.ndarray,
    occupied: np.ndarray,
    scale: np.ndarray,
    step: pd.Timedelta,
    budget_rows: int,
) -> tuple[np.ndarray, list[Segment]]:
    """Spend the point budget across spike / plateau / gap by row weight."""
    segments: list[Segment] = []

    for kind, weight in POINT_BUDGET_WEIGHTS.items():
        target = int(round(budget_rows * weight))
        spent = 0
        # Generous attempt cap: placement gets harder as the series fills up, and
        # we stop on budget, not on attempts.
        attempts = 0
        max_attempts = 10_000

        while spent < target and attempts < max_attempts:
            attempts += 1

            if kind == SPIKE:
                length = _uniform_int(rng, SPIKE_LEN_ROWS)
            elif kind == PLATEAU:
                length = _rows_per(
                    pd.Timedelta(hours=_uniform(rng, PLATEAU_DURATION_HOURS)), step
                )
            else:
                length = _rows_per(
                    pd.Timedelta(hours=_uniform(rng, GAP_DURATION_HOURS)), step
                )
            length = min(length, max(1, target - spent))

            start = _find_free_start(rng, occupied, length)
            if start is None:
                break

            window = base[start : start + length]
            local = float(np.median(scale[start : start + length]))

            if kind == SPIKE:
                # A spike displaces a reading, so every row it touches needs one.
                if not np.isfinite(window).all():
                    continue
                magnitude = _signed(
                    rng, _loguniform(rng, SPIKE_MAGNITUDE_SCALE) * local, SPIKE_UPWARD_PROB
                )
                values, seg = inject_spike(values, start, magnitude=magnitude, length=length)
            elif kind == PLATEAU:
                # A stuck sensor may span an isolated dropout, but it has to stick
                # at a real reading, so only the first row must be present.
                if not np.isfinite(base[start]):
                    continue
                # A plateau over already-flat data is indistinguishable from the
                # base, so only stick the sensor where the reading actually moves.
                if float(np.nanstd(window)) <= 0.0:
                    continue
                values, seg = inject_plateau(values, start, length=length)
            else:
                # An injected gap must delete values we know, or its true_value
                # would be unknown and `source=injected` would be a lie.
                if not np.isfinite(window).all():
                    continue
                values, seg = inject_gap(values, start, length=length)

            segments.append(seg)
            _mark(occupied, start, length)
            spent += length

    return values, segments


def _natural_gap_mask(base: np.ndarray) -> np.ndarray:
    """Rows missing in the clean base itself, before we inject anything."""
    return ~np.isfinite(base)


def _runs(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-open ``[start, stop)`` runs of True in a boolean mask."""
    edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
    return np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)


def _blocking_gap_mask(natural_gaps: np.ndarray, max_span: int) -> np.ndarray:
    """Natural gaps that genuinely block placement: the runs longer than ``max_span``.

    Isolated short dropouts are left unblocked so segment anomalies (plateau,
    level_shift, drift) can span them, exactly as they span them in reality. Only
    real outages stop a segment from being placed.
    """
    blocking = natural_gaps.copy()
    starts, stops = _runs(natural_gaps)
    for start, stop in zip(starts, stops):
        if (stop - start) <= max_span:
            blocking[start:stop] = False
    return blocking


def _build_labels(
    df: pd.DataFrame,
    base: np.ndarray,
    segments: list[Segment],
    natural_gaps: np.ndarray,
) -> pd.DataFrame:
    """Assemble the §5 labels frame, row-aligned with the injected series."""
    n = len(base)
    is_anomaly = np.zeros(n, dtype=bool)
    anomaly_type = np.array([""] * n, dtype=object)
    source = np.array([""] * n, dtype=object)

    for seg in segments:
        is_anomaly[seg.start_idx : seg.end_idx] = True
        anomaly_type[seg.start_idx : seg.end_idx] = seg.anomaly_type
        source[seg.start_idx : seg.end_idx] = SOURCE_INJECTED

    # Natural gaps are anomalies too — the base is not pristine and the labels
    # should not pretend otherwise. They carry no true_value (it was never
    # recorded), which is what excludes them from imputation scoring.
    is_anomaly[natural_gaps] = True
    anomaly_type[natural_gaps] = GAP
    source[natural_gaps] = SOURCE_NATURAL

    return pd.DataFrame(
        {
            DATETIME_COL: df[DATETIME_COL].to_numpy(),
            "is_anomaly": is_anomaly,
            "anomaly_type": anomaly_type,
            "true_value": base,
            "source": source,
        }
    )


def _build_manifest(
    name: str,
    level: ContaminationLevel,
    seed: int,
    labels: pd.DataFrame,
    segments: list[Segment],
    n_rows: int,
) -> dict[str, Any]:
    """Per-type accounting, so evaluation never has to infer counts from the level."""
    by_type: dict[str, Any] = {}
    for kind in (SPIKE, PLATEAU, LEVEL_SHIFT, GAP, DRIFT):
        rows = int((labels["anomaly_type"] == kind).sum())
        events = sum(1 for s in segments if s.anomaly_type == kind)
        by_type[kind] = {
            "n_events": events,
            "n_rows": rows,
            "pct_rows": round(100.0 * rows / n_rows, 3) if n_rows else 0.0,
        }

    natural_gap_rows = int((labels["source"] == SOURCE_NATURAL).sum())
    total_anom = int(labels["is_anomaly"].sum())
    point_rows = sum(by_type[k]["n_rows"] for k in POINT_BUDGET_WEIGHTS)
    # Natural gaps land in the gap row count but were not part of the budget.
    point_rows_injected = point_rows - natural_gap_rows

    # Per-type recall (§10) is undefined for a type with no positives, so a
    # dataset missing a type cannot be scored on it and must say so.
    types_missing = [k for k, v in by_type.items() if v["n_events"] == 0]

    actual_point_pct = round(100.0 * point_rows_injected / n_rows, 3) if n_rows else 0.0
    # A base with too little usable room leaves nowhere to put the longer point
    # anomalies, so the budget silently under-fills. Record that rather than let
    # a dataset claim a contamination level it does not carry.
    budget_met = actual_point_pct >= level.point_pct * (1.0 - POINT_BUDGET_TOLERANCE_FRAC)

    return {
        "name": name,
        "seed": seed,
        "level": level.level,
        "level_name": level.name,
        "n_rows": n_rows,
        "maintenance_interval_days": level.maintenance_interval_days,
        "target_point_pct": level.point_pct,
        "actual_point_pct": actual_point_pct,
        "point_budget_met": budget_met,
        "types_missing": types_missing,
        "scoreable": budget_met and not types_missing,
        "total_anomalous_rows": total_anom,
        "total_anomalous_pct": round(100.0 * total_anom / n_rows, 3) if n_rows else 0.0,
        "natural_gap_rows": natural_gap_rows,
        "natural_gap_pct": round(100.0 * natural_gap_rows / n_rows, 3) if n_rows else 0.0,
        "by_type": by_type,
    }


def inject_series(
    df: pd.DataFrame,
    *,
    level: int,
    seed: int = DEFAULT_SEED,
    name: str = "series",
    value_col: str = VALUE_COL,
) -> InjectionResult:
    """Inject all five anomaly types into one clean base at one contamination level.

    ``df`` must already sit on a regular grid (use :func:`load_base`), so that
    natural gaps are explicit NaN rows and labels stay row-aligned by datetime.

    Injection order runs largest-footprint first — drift, then level shifts, then
    the point types — because each claims its rows before the next one places, and
    the small types have far more freedom to find a home in what is left.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {sorted(LEVELS)} (got {level}).")
    if value_col not in df.columns:
        raise ContractError(f"base frame missing value column {value_col!r}.")
    if len(df) < 2:
        raise ContractError(f"base frame has {len(df)} row(s); need at least 2.")

    cfg = LEVELS[level]
    rng = _derive_seed(seed, level, name)
    step = _median_step(df)

    base = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    values = base.copy()
    natural_gaps = _natural_gap_mask(base)
    scale = _local_scale(base, LOCAL_SCALE_WINDOW_ROWS)

    # Only real outages block placement. Isolated dropouts stay unblocked so
    # segment anomalies can span them the way they do in reality; the per-type
    # rules in `_inject_point_anomalies` handle what each type actually needs.
    occupied = _blocking_gap_mask(natural_gaps, SPANNABLE_DROPOUT_ROWS)

    values, drift_segs, maint_events = _inject_drift_and_maintenance(
        rng, values, occupied, scale, step, cfg.maintenance_interval_days
    )
    values, shift_segs = _inject_level_shifts(
        rng, values, occupied, scale, step, cfg.n_level_shifts
    )
    budget_rows = int(round(cfg.point_pct / 100.0 * len(base)))
    values, point_segs = _inject_point_anomalies(
        rng, values, base, occupied, scale, step, budget_rows
    )

    segments = sorted(
        drift_segs + shift_segs + point_segs, key=lambda s: s.start_idx
    )

    data = pd.DataFrame(
        {DATETIME_COL: df[DATETIME_COL].to_numpy(), VALUE_COL: values}
    )
    labels = _build_labels(df, base, segments, natural_gaps)

    times = pd.DatetimeIndex(df[DATETIME_COL])
    maintenance = pd.DataFrame(
        {
            "start": [times[e.start_idx] for e in maint_events],
            "end": [times[min(e.end_idx, len(times) - 1)] for e in maint_events],
        }
    )

    manifest = _build_manifest(name, cfg, seed, labels, segments, len(base))
    return InjectionResult(
        name=name,
        level=cfg,
        seed=seed,
        data=data,
        labels=labels,
        maintenance=maintenance,
        segments=segments,
        manifest=manifest,
    )


# ---------------------------------------------------------------------------
# Writing + CLI
# ---------------------------------------------------------------------------
def write_result(result: InjectionResult, outdir: Path) -> dict[str, Path]:
    """Write the §5 quartet: series, labels, maintenance schedule, manifest."""
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {
        "data": outdir / f"{result.name}.csv",
        "labels": outdir / f"{result.name}_labels.csv",
        "maintenance": outdir / f"{result.name}_maintenance.csv",
        "manifest": outdir / f"{result.name}_manifest.json",
    }
    result.data.to_csv(paths["data"], index=False)
    result.labels.to_csv(paths["labels"], index=False)
    result.maintenance.to_csv(paths["maintenance"], index=False)
    paths["manifest"].write_text(json.dumps(result.manifest, indent=2))
    return paths


def _base_name(path: Path, level: int) -> str:
    """``03447687_clean_20230807_20250101.csv`` + level 2 -> ``03447687_l2``."""
    stem = path.stem
    site = stem.split("_")[0]
    return f"{site}_l{level}"


def format_manifest(manifest: dict[str, Any]) -> str:
    """Human-readable per-type breakdown for the terminal."""
    lines = [
        f"  rows              : {manifest['n_rows']:,}",
        f"  point budget      : {manifest['target_point_pct']:.1f}% target -> "
        f"{manifest['actual_point_pct']:.2f}% actual (spike/plateau/gap, injected only)",
        f"  total anomalous   : {manifest['total_anomalous_rows']:,} rows "
        f"({manifest['total_anomalous_pct']:.2f}%, incl. drift/level_shift/natural gaps)",
        f"  natural gap rows  : {manifest['natural_gap_rows']:,} "
        f"({manifest['natural_gap_pct']:.2f}%)",
        "  by type:",
    ]
    for kind, stats in manifest["by_type"].items():
        lines.append(
            f"    {kind:<12} {stats['n_events']:>4} events  "
            f"{stats['n_rows']:>7,} rows  ({stats['pct_rows']:.2f}%)"
        )
    if not manifest["point_budget_met"]:
        lines.append(
            f"  ⚠ point budget UNMET ({manifest['actual_point_pct']:.2f}% of "
            f"{manifest['target_point_pct']:.1f}%): the base has too little usable room "
            f"to place the longer plateaus/gaps. This dataset does not carry its "
            f"nominal level."
        )
    if manifest["types_missing"]:
        lines.append(
            f"  ⚠ NO POSITIVES for {manifest['types_missing']}: per-type recall is "
            f"undefined, so this dataset cannot be scored on those types. The base is "
            f"too fragmented to host them — use a denser base."
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Inject the five synthetic anomaly types into clean base segments, "
            "with ground-truth labels and a maintenance schedule."
        ),
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="*",
        help="Clean base CSV(s). Default: every *.csv in data/clean.",
    )
    parser.add_argument(
        "--levels",
        type=int,
        nargs="*",
        default=sorted(LEVELS),
        choices=sorted(LEVELS),
        help="Contamination level(s) to build (default: all three).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Master seed; per-dataset streams derive from it (default: {DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help=f"Directory for injected datasets (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--value-col",
        default=VALUE_COL,
        help=f"Numeric value column in the base CSV (default: {VALUE_COL}).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written and exit without writing.",
    )
    args = parser.parse_args(argv)

    inputs = list(args.input) if args.input else sorted(DEFAULT_CLEAN_DIR.glob("*.csv"))
    if not inputs:
        print(
            f"ERROR: no base CSVs found (looked in {DEFAULT_CLEAN_DIR}). "
            "Pass --input explicitly.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print(f"Would inject seed={args.seed} levels={args.levels} -> {args.outdir}")
        for path in inputs:
            for level in args.levels:
                print(f"  {path.name} -> {_base_name(path, level)}.csv (+ labels, maintenance)")
        return 0

    for path in inputs:
        print(f"\n{path}")
        base = load_base(path, value_col=args.value_col)
        for level in args.levels:
            name = _base_name(path, level)
            result = inject_series(
                base, level=level, seed=args.seed, name=name, value_col=args.value_col
            )
            paths = write_result(result, args.outdir)
            print(f"  level {level} ({result.level.name}) -> {paths['data']}")
            print(format_manifest(result.manifest))

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
