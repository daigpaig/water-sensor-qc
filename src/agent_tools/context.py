"""Point-context tools: describe the neighbourhood of one timestamp.

The detectors in ``wrappers.py`` answer *where* something looks unusual. They do
not answer the question the agent actually has to decide (CLAUDE.md §6, §8):
**is this excursion real water or a sensor artifact?** The agent cannot see the
series, so the only way it can tell a storm from a spike is to be handed numbers
that describe the *shape* around a flagged point.

That is what this module is. Every public function takes a series plus one
timestamp and returns a small JSON-serialisable dict of measurements about the
points surrounding it. Nothing here flags, mutates or writes anything.

The discriminating measurements, and what they mean physically:

  ``slope_context``      rise vs fall gradient either side of the point, and the
                         single-sample steps into and out of it. At the default
                         45-minute window a *gentle* fall against a sharp rise
                         (ratio ≲ 0.8) means a real flush event; ≈ 1 means an
                         artifact. The window matters more than anything else
                         here — the signal reverses past ~90 min. Read its
                         docstring before changing it.
  ``excursion_context``  how *wide* the excursion is at half its height, and how
                         much of it happens in a single sample. One sample wide
                         and sharp ⇒ artifact; hours wide ⇒ event.
  ``recovery_context``   how long until the series returns to its pre-event
                         baseline. Spike: 1–2 samples. Storm: hours to days.
  ``level_shift_context`` median before vs after over a long window, and whether
                         the new level *persists* well beyond the transition.
  ``flatness_context``   repeated-value evidence for a stuck sensor (plateau).
  ``neighbourhood_stats`` local median / robust sigma / robust z / percentile.
  ``noise_context``      how noisy this stretch is compared with the rest of the
                         record, and how far the point stands out *within* that
                         stretch. The measurement that stops a noisy hour being
                         read as fifty separate sensor failures.
  ``gap_context``        distance to the nearest NaN run; values on a gap edge
                         are the usual suspects for telemetry artifacts.
  ``historical_context`` has the series *ever* reached this level elsewhere, and
                         in how many separate episodes. A level reached 40 times
                         before is part of the regime, not an outlier.
  ``describe_point``     runs all of the above and adds a ``reads_like`` hint.
  ``describe_points``    the same, compacted, for a whole list of flagged
                         timestamps — one tool call for a detector's output.

Contract notes
--------------
* Results follow the spirit of CLAUDE.md §5 (``tool``, ``params``, ``message``
  plus tool-specific fields) but carry no ``n_flagged`` / ``flagged_datetimes``:
  these tools observe, they never flag. ``inspect_dataset`` sets the precedent.
* All thresholds are expressed in **robust sigmas of this series**, never in
  data units, so they transfer between an 8 FNU mountain river and a 28 FNU
  canal (CLAUDE.md §7.1, §7.2).
* Quantised data makes a windowed MAD collapse to 0 and any robust z explode
  (§7.1). Every sigma here is therefore floored by the series-wide robust
  first-difference scale, computed once in ``_step_sigma``.
* Every ``reads_like`` string is a **hint, not a verdict**. It states which
  shape the numbers resemble; the agent still decides.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import saqc

from src.inspect_data import DATETIME_COL, VALUE_COL

# Scale factor that makes the MAD a consistent estimator of sigma for Gaussian data.
_MAD_TO_SIGMA = 1.4826

# The §7.4 noisy-stretch guard, re-tuned for the 5-min bases (2026-08-20). A point
# is "unremarkable here" — reads_like returns `noisy-stretch` rather than `spike` —
# when its surroundings are ELEVATED_NOISE_RATIO x noisier than a typical window of
# this record AND its own largest single-sample move is under
# LOCAL_STEP_UNREMARKABLE x the LOCAL sample-to-sample scale.
#
# Was 3.0 / 5, fitted on the retired 15-min gauges. Those values fire on 0 of the
# 104 surviving false positives at 5-min. Refitted on 2,892 FPs / 392 true spikes
# pooled from 02054550 and 040851385 — never the gauge we report on — by
# scratchpad/tune_noise_spare_rule.py, maximising FPs spared subject to losing at
# most 5% of true spikes. See the noise_context docstring for the full table.
# ramp_context: how long a climb has to last, and how steadily it has to move, before
# the point on top of it reads as an event rather than an artifact. Fitted on 02054550
# and 040851385, never on the gauge we report on (scratchpad/tune_ramp_rule.py), by the
# same criterion as the §7.4 noise rule: most FPs spared at <=5% of true spikes lost.
#
# BE HONEST ABOUT HOW WEAK THIS IS. The two populations overlap heavily — true spike
# rise median 180 min vs false positive 220 min, monotonic 0.39 vs 0.45 — because §9
# injects spikes ON TOP OF whatever the series was doing, including rising limbs, so a
# genuine injected spike sits on a ramp about as often as real water does. The fitted
# rule spares 23.1% of false positives at 4.8% of true spikes, against 60.2%/4.1%
# for the noise rule. Treat a ramp as ONE piece of evidence the agent weighs, never as
# a decider — and expect it to matter more on real records than on injected ones.
RAMP_MINUTES = 30.0
RAMP_DIRECTNESS = 0.3

# shift_window_context / find_shift_windows: what makes a span between two jumps a
# level SHIFT rather than a storm. Interior elevation alone does not separate them —
# measured on 01467200_l1, 1 of 24 elevated windows was the injected shift, and the
# biggest storm scored z=+8.6 against the shift's +4.5. Edge sharpness is the lever
# (§9.1). Fitted on gauges we do not score.
# How long the new level must hold before it is a SHIFT rather than an excursion.
#
# This was hard-coded as `6 * 60` with the comment "§9: injected shifts run 6-72 h".
# That range was correct when it was written and stopped being correct on 2026-07-31:
# §9 now injects 4-24 h shifts, so the low end of the range was unreachable — a 4 h
# shift was "transient, not a shift" by definition. tests/test_context.py asserts this
# against inject.LEVEL_SHIFT_DURATION_HOURS so it cannot go stale a third time.
SHIFT_MIN_HOLD_MINUTES = 4 * 60

# A departure from the new level must last this long to end the hold. See the walk in
# level_shift_context for what the old two-sample rule did to real shifts.
HOLD_BREAK_AFTER = "1h"

SHIFT_INTERIOR_SIGMAS = 3.0
SHIFT_EDGE_SHARPNESS = 0.5

ELEVATED_NOISE_RATIO = 2.0
LOCAL_STEP_UNREMARKABLE = 8.0

# A sigma floor of last resort, used only when a series has no variation at all.
_SIGMA_EPS = 1e-9

SeriesLike = "saqc.SaQC | pd.DataFrame | pd.Series"


# ---------------------------------------------------------------------------
# Input plumbing
# ---------------------------------------------------------------------------
def as_series(source, field: str = VALUE_COL) -> pd.Series:
    """Coerce a SaQC object / DataFrame / Series to a datetime-indexed float Series.

    Accepts what the agent has (``saqc.SaQC``) and what a script has (a frame
    loaded by :func:`src.inspect_data.load_series`, or a bare Series). Sorts the
    index, since every measurement below assumes monotonic time (§7.1).
    """
    if isinstance(source, saqc.SaQC):
        series = source.data.to_pandas()[field]
    elif isinstance(source, pd.DataFrame):
        frame = source
        if not isinstance(frame.index, pd.DatetimeIndex):
            if DATETIME_COL not in frame.columns:
                raise ValueError(
                    f"DataFrame needs a DatetimeIndex or a {DATETIME_COL!r} column."
                )
            frame = frame.set_index(DATETIME_COL)
        if field not in frame.columns:
            raise ValueError(f"column {field!r} not in frame; got {list(frame.columns)}.")
        series = frame[field]
    elif isinstance(source, pd.Series):
        series = source
    else:
        raise TypeError(f"unsupported source type {type(source).__name__}.")

    if not isinstance(series.index, pd.DatetimeIndex):
        series = series.copy()
        series.index = pd.DatetimeIndex(series.index)
    series = pd.to_numeric(series, errors="coerce").astype(float)
    if not series.index.is_monotonic_increasing:
        series = series.sort_index()
    return series


def resolve_timestamp(series: pd.Series, at) -> pd.Timestamp:
    """Return the index timestamp addressed by *at*, failing loudly if there is none.

    An exact hit is used as-is. Otherwise the nearest sample is used, but only
    within half a sampling step — a timestamp that falls outside the series or
    between grids is a caller error, not something to silently round away (§13).
    """
    if len(series) == 0:
        raise ValueError("series is empty.")
    ts = pd.Timestamp(at)
    if ts.tz is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    if ts in series.index:
        return ts

    pos = int(series.index.get_indexer([ts], method="nearest")[0])
    nearest = series.index[pos]
    tolerance = _median_step(series) / 2
    if abs(nearest - ts) > tolerance:
        raise ValueError(
            f"timestamp {ts.isoformat()} is not in the series "
            f"(nearest sample {nearest.isoformat()}, {abs(nearest - ts)} away)."
        )
    return nearest


def _median_step(series: pd.Series) -> pd.Timedelta:
    """Median positive sampling step; falls back to 15 min on a degenerate index."""
    if len(series) < 2:
        return pd.Timedelta("15min")
    diffs = series.index.to_series().diff().dropna()
    positive = diffs[diffs > pd.Timedelta(0)]
    if positive.empty:
        return pd.Timedelta("15min")
    return positive.median()


def _pos(series: pd.Series, ts: pd.Timestamp) -> int:
    return int(series.index.get_indexer([ts])[0])


def _round(x, digits: int = 1) -> float | None:
    """`_f` with a caller-chosen precision — for numbers a reader will scan, not compute with."""
    value = _f(x)
    return None if value is None else round(value, digits)


def _f(x) -> float | None:
    """Round to a JSON-friendly float; NaN / inf / non-numeric become None."""
    if x is None:
        return None
    try:
        value = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    return round(value, 6)


def _iso(ts) -> str | None:
    return None if ts is None or pd.isna(ts) else pd.Timestamp(ts).isoformat()


# ---------------------------------------------------------------------------
# Robust scale helpers
# ---------------------------------------------------------------------------
def _mad_sigma(values: np.ndarray) -> float:
    """MAD-based sigma estimate of *values* (no floor applied)."""
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return 0.0
    return float(_MAD_TO_SIGMA * np.median(np.abs(clean - np.median(clean))))


def _step_sigma(series: pd.Series) -> float:
    """Series-wide robust scale of the first differences.

    This is the sigma floor used everywhere in this module. A rolling MAD goes to
    zero on quantised turbidity (§7.1), which would make every local robust z
    meaningless; the sample-to-sample scale of the whole record does not.
    """
    diffs = series.diff().to_numpy()
    sigma = _mad_sigma(diffs)
    if sigma <= 0:
        spread = _mad_sigma(series.to_numpy())
        sigma = spread if spread > 0 else _SIGMA_EPS
    return max(sigma, _SIGMA_EPS)


def _local_sigma(values: np.ndarray, floor: float) -> float:
    return max(_mad_sigma(values), floor)


def _slope_per_hour(segment: pd.Series) -> tuple[float | None, int]:
    """Least-squares slope of *segment* in data units per hour, plus n valid points."""
    clean = segment.dropna()
    if len(clean) < 2:
        return None, len(clean)
    hours = (clean.index - clean.index[0]).total_seconds().to_numpy() / 3600.0
    if np.ptp(hours) == 0:
        return None, len(clean)
    slope = float(np.polyfit(hours, clean.to_numpy(), 1)[0])
    return slope, len(clean)


def _bool_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs in *mask* as inclusive (start, end) positions."""
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for i, flag in enumerate(mask):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _result(tool: str, params: dict, ts: pd.Timestamp, message: str, **fields) -> dict:
    out = {"tool": tool, "params": params, "at": _iso(ts), "message": message}
    out.update(fields)
    return out


# ---------------------------------------------------------------------------
# 1. Slope — the storm-vs-spike discriminator
# ---------------------------------------------------------------------------
SLOPE_WINDOW = "45min"


def slope_context(
    source,
    at,
    field: str = VALUE_COL,
    n_before: int | None = None,
    n_after: int | None = None,
    window: str = SLOPE_WINDOW,
) -> dict:
    """Gradient leading into *at* vs the gradient leading out of it.

    THE WINDOW IS A DURATION, AND THAT IS THE WHOLE POINT. §7.3 measured the
    optimum at **45 minutes** and expressed it as "3 samples" because the data was
    15-minute. When the project moved to 5-minute bases that default silently
    became 15 minutes — a 3x narrower window — and §7.3 warns in terms that the
    signal REVERSES at the wrong width. Measured on 01467200_l1 at
    2023-07-09 22:50, a gradual 2.5h rise the run deleted as a spike:

        default 3 samples (15 min at 5-min):  ratio 0.54  "symmetric"     -> spike
        45 minutes (9 samples at 5-min):      ratio 3.09  "gradual_before" -> storm

    So *window* is derived from the series' own sampling step, and `n_before` /
    `n_after` remain as an explicit sample-count override for callers that really
    want one. Each window *includes* the point itself, so the rise into the peak
    and the fall out of it are both measured.

    THE WINDOW IS THE WHOLE BALLGAME, AND THE SIGNAL INVERTS WITH IT (measured
    2026-08-11 on 31 flush events vs 14 injected spikes, 03447687_l1). Separation,
    as P(artifact ratio > flush ratio):

        window            flush median   artifact median   separation
        2 samples (30m)       0.58            0.99            0.56
        3 samples (45m)       0.75            0.99            0.84   <- default
        4 samples (60m)       0.83            1.00            0.71
        6 samples (90m)       1.11            0.98            0.42
        8 samples (120m)      1.24            0.99            0.34
        12 samples (180m)     1.43            0.93            0.27

    A small first-flush event rises in one sample and decays over 3-5, so at a
    45-minute window its fall gradient is visibly gentler than its rise and the
    ratio sits near 0.75; a debris strike falls as fast as it rose and sits at
    0.99. Past ~90 minutes the decay is over and the window fills with the flat
    surroundings, the ratio drifts above 1, and the ordering REVERSES — which is
    why the earlier reading of this function (2 h and wider, on storm peaks rather
    than flush events) concluded the ratio was useless and recorded that in §7.3.
    It is not useless; it was being measured past the scale of the thing it
    measures. Default therefore moved 8 -> 3 samples.

    This is still one signal among several: at a ``< 0.8`` threshold the rule
    "gentle fall means real water, keep it" catches 77% of flush events and
    wrongly keeps 14% of genuine artifacts. Read it with width and recovery time
    (``excursion_context``, ``recovery_context``), not instead of them.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    if n_before is None or n_after is None:
        span = pd.Timedelta(window)
        step = pd.Series(series.index).diff().median()
        derived = max(int(round(span / step)), 2) if step and step > pd.Timedelta(0) else 3
        n_before = derived if n_before is None else n_before
        n_after = derived if n_after is None else n_after
    params = {"field": field, "n_before": n_before, "n_after": n_after, "window": window}

    before = series.iloc[max(0, pos - n_before + 1) : pos + 1]
    after = series.iloc[pos : pos + n_after]
    slope_before, n_valid_before = _slope_per_hour(before)
    slope_after, n_valid_after = _slope_per_hour(after)

    ratio = None
    if slope_before is not None and slope_after is not None and abs(slope_before) > 0:
        ratio = abs(slope_after) / abs(slope_before)

    # Single-sample steps either side: the sharpest local evidence there is.
    prev_value = series.iloc[pos - 1] if pos > 0 else np.nan
    next_value = series.iloc[pos + 1] if pos + 1 < len(series) else np.nan
    value = series.iloc[pos]
    delta_before = value - prev_value
    delta_after = next_value - value

    if slope_before is None or slope_after is None:
        shape = "unknown"
    elif slope_before > 0 and slope_after < 0:
        shape = "peak"
    elif slope_before < 0 and slope_after > 0:
        shape = "trough"
    elif slope_before >= 0 and slope_after >= 0:
        shape = "rising"
    else:
        shape = "falling"

    if ratio is None:
        symmetry = "unknown"
    elif ratio < 0.5:
        symmetry = "gradual_after"      # storm-like recession
    elif ratio > 2.0:
        symmetry = "gradual_before"     # gradual build, abrupt drop
    else:
        symmetry = "symmetric"          # spike-like

    step_sigma = _step_sigma(series)
    msg = (
        f"{shape}: slope {slope_before if slope_before is None else round(slope_before, 3)} "
        f"-> {slope_after if slope_after is None else round(slope_after, 3)} units/h "
        f"(ratio {'n/a' if ratio is None else round(ratio, 2)}, {symmetry}). "
        "A symmetric peak is spike-like; gradual_after is storm-like."
    )

    return _result(
        "slope_context", params, ts, msg,
        value=_f(value),
        slope_before_per_hour=_f(slope_before),
        slope_after_per_hour=_f(slope_after),
        fall_rise_ratio=_f(ratio),
        shape=shape,
        symmetry=symmetry,
        delta_before=_f(delta_before),
        delta_after=_f(delta_after),
        delta_before_sigmas=_f(delta_before / step_sigma),
        delta_after_sigmas=_f(delta_after / step_sigma),
        n_valid_before=n_valid_before,
        n_valid_after=n_valid_after,
    )


# ---------------------------------------------------------------------------
# 2. Excursion extent — how wide, how sharp
# ---------------------------------------------------------------------------
def excursion_context(
    source,
    at,
    field: str = VALUE_COL,
    baseline_window: str = "12h",
    max_samples: int = 400,
) -> dict:
    """Width and sharpness of the excursion containing *at*.

    The baseline is the median of ``baseline_window`` either side (a median is
    unmoved by a short excursion). Width is measured at *half height*: walk
    outward while values stay past the halfway mark between baseline and peak.

    ``peak_sharpness`` is the largest single-sample move inside the excursion as
    a fraction of the whole excursion — the same lever that separates a
    recalibration step from a storm limb in §9.1. Near 1.0 means the excursion
    happened in one sample (artifact-like); 0.1 means it built over many.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    params = {
        "field": field,
        "baseline_window": baseline_window,
        "max_samples": max_samples,
    }

    value = series.iloc[pos]
    window = pd.Timedelta(baseline_window)
    local = series.loc[ts - window : ts + window]
    baseline = float(np.nanmedian(local.to_numpy())) if local.notna().any() else np.nan

    if not np.isfinite(value) or not np.isfinite(baseline):
        # Keep the key set identical on every path so callers can read a field
        # without first checking which branch produced the dict.
        return _result(
            "excursion_context", params, ts,
            "Value or local baseline is NaN; no excursion could be measured.",
            value=_f(value), baseline=_f(baseline), excursion=None,
            excursion_sigmas=None, direction=None, n_samples=0,
            duration_minutes=None, start=None, end=None,
            max_single_step=None, peak_sharpness=None, isolated=None,
        )

    excursion = value - baseline
    half = baseline + 0.5 * excursion
    positive = excursion >= 0

    def _past_half(v: float) -> bool:
        if not np.isfinite(v):
            return False
        return v >= half if positive else v <= half

    lo = hi = pos
    while lo - 1 >= 0 and pos - (lo - 1) <= max_samples and _past_half(series.iloc[lo - 1]):
        lo -= 1
    while hi + 1 < len(series) and (hi + 1) - pos <= max_samples and _past_half(series.iloc[hi + 1]):
        hi += 1

    span = series.iloc[lo : hi + 1]
    n_samples = len(span)
    duration = span.index[-1] - span.index[0]

    # Sharpness over the run-up and run-down, including the samples just outside
    # the half-height span so a one-sample jump into the excursion is counted.
    edge_lo = max(0, lo - 1)
    edge_hi = min(len(series) - 1, hi + 1)
    edge = series.iloc[edge_lo : edge_hi + 1]
    max_step = float(np.nanmax(np.abs(edge.diff().to_numpy()))) if len(edge) > 1 else np.nan
    sharpness = max_step / abs(excursion) if abs(excursion) > 0 else None

    step_sigma = _step_sigma(series)
    isolated = n_samples <= 2

    msg = (
        f"Excursion of {round(excursion, 3)} units above baseline {round(baseline, 3)} "
        f"spans {n_samples} sample(s) ({duration}) at half height; "
        f"peak sharpness {'n/a' if sharpness is None else round(sharpness, 2)}. "
        + ("One or two samples wide => spike-like."
           if isolated else "Multi-sample width => event-like, not a point artifact.")
    )

    return _result(
        "excursion_context", params, ts, msg,
        value=_f(value),
        baseline=_f(baseline),
        excursion=_f(excursion),
        excursion_sigmas=_f(excursion / step_sigma),
        direction="up" if positive else "down",
        n_samples=n_samples,
        duration_minutes=_f(duration.total_seconds() / 60.0),
        start=_iso(span.index[0]),
        end=_iso(span.index[-1]),
        max_single_step=_f(max_step),
        peak_sharpness=_f(sharpness),
        isolated=bool(isolated),
    )


# ---------------------------------------------------------------------------
# 3. Recovery — how long until the series comes back
# ---------------------------------------------------------------------------
def recovery_context(
    source,
    at,
    field: str = VALUE_COL,
    baseline_window: str = "6h",
    tolerance_sigmas: float = 2.0,
    max_search: str = "7D",
    n_confirm: int = 2,
    anchor: str = "excursion",
) -> dict:
    """How long after *at* the series returns to its pre-event baseline.

    Recovery requires ``n_confirm`` consecutive samples back inside
    ``tolerance_sigmas`` of the baseline, so one sample dipping through the band
    does not count as a return.

    ``anchor`` decides where the baseline is measured from:

      ``'excursion'`` (default) — ``baseline_window`` ending where the current
        excursion *began* (via :func:`excursion_context`). This matters: for a
        point in the middle of a 10-hour storm, the six hours before the *point*
        are already storm, so a point-anchored baseline would report an instant
        recovery for an event that is nowhere near over.
      ``'point'`` — the plain ``baseline_window`` before *at*.

    Typical readings: an artifact spike recovers in 1–2 samples; a storm takes
    hours; a level shift never recovers (``recovered=False``).
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    params = {
        "field": field,
        "baseline_window": baseline_window,
        "tolerance_sigmas": tolerance_sigmas,
        "max_search": max_search,
        "n_confirm": n_confirm,
        "anchor": anchor,
    }

    onset = ts
    if anchor == "excursion":
        span = excursion_context(series, ts, field)
        if span.get("start"):
            onset = min(ts, pd.Timestamp(span["start"]))
    pre = series.loc[onset - pd.Timedelta(baseline_window) : onset].iloc[:-1]
    pre_valid = pre.dropna()
    if pre_valid.empty:
        return _result(
            "recovery_context", params, ts,
            "No valid samples before this point; recovery is undefined "
            "(the point sits at the very start of the record or after a long gap).",
            recovered=None, recovered_at=None, n_samples_to_recover=None,
            minutes_to_recover=None, baseline=None, baseline_sigma=None,
            tolerance=None, n_samples_searched=0,
        )

    baseline = float(np.median(pre_valid.to_numpy()))
    sigma = _local_sigma(pre_valid.to_numpy(), _step_sigma(series))
    tolerance = tolerance_sigmas * sigma
    params["baseline_anchored_at"] = _iso(onset)

    forward = series.iloc[pos + 1 :].loc[: ts + pd.Timedelta(max_search)]
    streak = 0
    recovered_at = None
    n_checked = 0
    for stamp, v in forward.items():
        n_checked += 1
        if np.isfinite(v) and abs(v - baseline) <= tolerance:
            streak += 1
            if streak >= n_confirm:
                # The recovery began at the first sample of the confirming streak.
                recovered_at = forward.index[n_checked - streak]
                break
        else:
            streak = 0

    if recovered_at is None:
        msg = (
            f"Did not return to baseline {round(baseline, 3)} ±{round(tolerance, 3)} "
            f"within {max_search} ({n_checked} samples checked). Consistent with a level "
            "shift or a long event, not a point artifact."
        )
        return _result(
            "recovery_context", params, ts, msg,
            recovered=False,
            recovered_at=None,
            n_samples_to_recover=None,
            minutes_to_recover=None,
            baseline=_f(baseline),
            baseline_sigma=_f(sigma),
            tolerance=_f(tolerance),
            n_samples_searched=n_checked,
        )

    n_samples = int(series.index.get_indexer([recovered_at])[0] - pos)
    minutes = (recovered_at - ts).total_seconds() / 60.0
    msg = (
        f"Returned to baseline {round(baseline, 3)} ±{round(tolerance, 3)} after "
        f"{n_samples} sample(s) ({round(minutes, 1)} min). "
        + ("Immediate recovery => spike-like."
           if n_samples <= 2 else "Slow recovery => event-like.")
    )
    return _result(
        "recovery_context", params, ts, msg,
        recovered=True,
        recovered_at=_iso(recovered_at),
        n_samples_to_recover=n_samples,
        minutes_to_recover=_f(minutes),
        baseline=_f(baseline),
        baseline_sigma=_f(sigma),
        tolerance=_f(tolerance),
        n_samples_searched=n_checked,
    )


# ---------------------------------------------------------------------------
# 4. Level shift — before vs after, and does it stick
# ---------------------------------------------------------------------------
def level_shift_context(
    source,
    at,
    field: str = VALUE_COL,
    window: str = "12h",
    skip_samples: int = 2,
    hold_sigmas: float = 3.0,
    max_hold: str = "30D",
) -> dict:
    """Compare the level before *at* with the level after, and see how long it holds.

    ``skip_samples`` excludes the transition itself from both medians. The step is
    reported in robust sigmas of the *pre* window, so it transfers across gauges.

    ``hold_minutes`` then walks forward while the series stays within
    ``hold_sigmas`` of the new median: that is how long the new level actually
    lasted. This is deliberately a duration rather than a ratio, because this
    project's injected level shifts are *bounded windows* of 6–72 h (§9), not
    literal permanent steps — a persistence test that assumed "forever" would
    score every injected shift as a transient event. Read it as: a few samples =
    a spike, hours-to-days = a shifted segment, running to the end of the record
    = a permanent recalibration step.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    params = {
        "field": field,
        "window": window,
        "skip_samples": skip_samples,
        "hold_sigmas": hold_sigmas,
        "max_hold": max_hold,
    }

    span = pd.Timedelta(window)
    before = series.iloc[: max(0, pos - skip_samples)].loc[ts - span :].dropna()
    after = series.iloc[pos + skip_samples + 1 :].loc[: ts + span].dropna()

    if before.empty or after.empty:
        return _result(
            "level_shift_context", params, ts,
            f"Not enough valid data either side ({len(before)} before, {len(after)} after).",
            median_before=None, median_after=None, step=None, step_sigmas=None,
            sigma_before=None, sigma_after=None, n_before=len(before), n_after=len(after),
            max_edge_step=None, step_sharpness=None,
            hold_samples=None, hold_minutes=None, holds_to_end_of_record=None,
        )

    median_before = float(np.median(before.to_numpy()))
    median_after = float(np.median(after.to_numpy()))
    step = median_after - median_before
    floor = _step_sigma(series)
    sigma_before = _local_sigma(before.to_numpy(), floor)
    sigma_after = _local_sigma(after.to_numpy(), floor)
    step_sigmas = step / sigma_before

    # How much of the step happens in a single sample? This is the §9.1 lever: a
    # recalibration or sensor swap moves most of its magnitude in one sample, a
    # storm limb spreads it over hours. It is what stops every storm rise from
    # reading as a level shift.
    edge = series.iloc[max(0, pos - skip_samples - 1) : pos + skip_samples + 2]
    edge_diffs = np.abs(edge.diff().to_numpy()) if len(edge) > 1 else np.array([])
    finite = edge_diffs[np.isfinite(edge_diffs)]
    max_edge_step = float(finite.max()) if finite.size else np.nan
    step_sharpness = max_edge_step / abs(step) if abs(step) > 0 else None

    # How long does the post-step level last? Two consecutive samples outside the
    # band end the hold, so one stray reading does not truncate it. The band uses
    # the SMALLER of the two sigmas: during a storm the post-window sigma is huge,
    # and scaling the band by it would let any storm "hold" indefinitely.
    tolerance = hold_sigmas * min(sigma_before, sigma_after)
    forward = series.iloc[pos + 1 :].loc[: ts + pd.Timedelta(max_hold)]
    hold_end = ts
    # END THE HOLD ON A SUSTAINED DEPARTURE, NOT ON TWO SAMPLES. Two consecutive
    # out-of-band readings is a threshold a shifted segment trips on its own noise, and
    # a NaN run inside a shift — which §9 explicitly allows — has the same effect.
    # Measured on 01467200: a 23.5 h shift reported a 0.0 h hold and a 22.0 h shift
    # reported 4.8 h, so both read as "transient excursion, not a shift" and the run
    # rejected the very events it had just found.
    first_miss = None
    for stamp, v in forward.items():
        if np.isfinite(v) and abs(v - median_after) > tolerance:
            if first_miss is None:
                first_miss = stamp
            elif stamp - first_miss >= pd.Timedelta(HOLD_BREAK_AFTER):
                break
        else:
            first_miss = None
            hold_end = stamp
    hold_minutes = (hold_end - ts).total_seconds() / 60.0
    hold_samples = int(series.index.get_indexer([hold_end])[0] - pos)
    holds_to_end = bool(hold_end >= forward.index[-1]) if len(forward) else False

    if abs(step_sigmas) < 1.0:
        verdict = "no meaningful step at this point"
    elif hold_minutes >= SHIFT_MIN_HOLD_MINUTES:
        verdict = (
            f"the new level held for {round(hold_minutes / 60, 1)} h "
            + ("to the end of the searched span => shift-like"
               if holds_to_end else "=> shift-like (a bounded shifted segment)")
        )
    else:
        verdict = (
            f"the new level held only {round(hold_minutes, 0)} min => "
            "transient excursion, not a shift"
        )

    msg = (
        f"Median {round(median_before, 3)} -> {round(median_after, 3)} "
        f"(step {round(step, 3)}, {round(step_sigmas, 1)} robust sigmas); {verdict}."
    )
    return _result(
        "level_shift_context", params, ts, msg,
        median_before=_f(median_before),
        median_after=_f(median_after),
        step=_f(step),
        step_sigmas=_f(step_sigmas),
        sigma_before=_f(sigma_before),
        sigma_after=_f(sigma_after),
        n_before=len(before),
        n_after=len(after),
        max_edge_step=_f(max_edge_step),
        step_sharpness=_f(step_sharpness),
        hold_samples=hold_samples,
        hold_minutes=_f(hold_minutes),
        holds_to_end_of_record=holds_to_end,
    )


# ---------------------------------------------------------------------------
# 5. Flatness — stuck-sensor evidence
# ---------------------------------------------------------------------------
def flatness_context(
    source,
    at,
    field: str = VALUE_COL,
    window: str = "6h",
    tol: float = 0.0,
) -> dict:
    """Repeated-value evidence around *at* (the plateau / stuck-sensor signature).

    ``tol`` is the largest sample-to-sample move still counted as "unchanged", in
    data units; 0.0 means bit-identical readings. Keep it far below the noise sd
    — §7.1 measured that a loose threshold swallows the whole series.

    Reports the run of unchanged values *containing* the point, the longest such
    run anywhere in the window, and how much of the window is unchanged at all.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    params = {"field": field, "window": window, "tol": tol}

    span = pd.Timedelta(window)
    local = series.loc[ts - span : ts + span].dropna()
    if len(local) < 2:
        return _result(
            "flatness_context", params, ts,
            f"Only {len(local)} valid sample(s) in ±{window}; flatness undefined.",
            run_length_at_point=None, run_minutes_at_point=None, run_start=None,
            run_end=None, longest_run=None, frac_unchanged=None,
            n_unique=int(local.nunique()), window_range=None, n_valid=len(local),
        )

    unchanged = local.diff().abs().to_numpy() <= tol
    unchanged[0] = False
    frac_unchanged = float(unchanged.mean())

    # Group into runs of unchanged values; a run of k "unchanged" diffs is k+1 samples.
    runs = _bool_runs(unchanged)
    longest = max((end - start + 2 for start, end in runs), default=1)

    run_at_point = 1
    run_bounds = (ts, ts)
    if ts in local.index:
        idx = int(local.index.get_indexer([ts])[0])
        for start, end in runs:
            if start - 1 <= idx <= end:
                run_at_point = end - start + 2
                run_bounds = (local.index[start - 1], local.index[end])
                break

    step = _median_step(series)
    minutes_at_point = run_at_point * step.total_seconds() / 60.0
    value_range = float(np.ptp(local.to_numpy()))

    msg = (
        f"{run_at_point} consecutive unchanged sample(s) at this point "
        f"(~{round(minutes_at_point, 0)} min); longest run in ±{window} is {longest}; "
        f"{round(frac_unchanged * 100, 1)}% of the window is unchanged, "
        f"range {round(value_range, 3)}. "
        + ("Long unchanged run => stuck-sensor-like."
           if run_at_point >= 8 else "No sustained flat run at this point.")
    )
    return _result(
        "flatness_context", params, ts, msg,
        run_length_at_point=run_at_point,
        run_minutes_at_point=_f(minutes_at_point),
        run_start=_iso(run_bounds[0]),
        run_end=_iso(run_bounds[1]),
        longest_run=longest,
        frac_unchanged=_f(frac_unchanged),
        n_unique=int(local.nunique()),
        window_range=_f(value_range),
        n_valid=len(local),
    )


# ---------------------------------------------------------------------------
# 6. Neighbourhood statistics
# ---------------------------------------------------------------------------
def neighbourhood_stats(
    source,
    at,
    field: str = VALUE_COL,
    window: str = "6h",
) -> dict:
    """Local median / robust sigma / robust z / percentile rank around *at*.

    The point itself is excluded from the statistics it is compared against, so a
    large excursion cannot inflate its own reference. ``robust_z`` uses a sigma
    floored by the series-wide step scale, which is what keeps it finite on
    quantised data (§7.1).
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    params = {"field": field, "window": window}

    span = pd.Timedelta(window)
    local = series.loc[ts - span : ts + span]
    others = local.drop(index=ts, errors="ignore").dropna()
    value = series.loc[ts]

    if others.empty:
        return _result(
            "neighbourhood_stats", params, ts,
            f"No other valid samples within ±{window}.",
            value=_f(value), local_median=None, local_robust_sigma=None,
            robust_z=None, percentile_in_window=None, local_min=None,
            local_max=None, n_valid=0, pct_nan_in_window=_f(local.isna().mean() * 100),
        )

    values = others.to_numpy()
    median = float(np.median(values))
    sigma = _local_sigma(values, _step_sigma(series))
    robust_z = (value - median) / sigma if np.isfinite(value) else None
    pct_rank = float((values < value).mean() * 100) if np.isfinite(value) else None
    pct_nan = float(local.isna().mean() * 100)

    msg = (
        f"Value {_f(value)} vs local median {round(median, 3)} "
        f"(robust sigma {round(sigma, 3)}): "
        f"{'n/a' if robust_z is None else round(robust_z, 1)} sigmas, "
        f"percentile {'n/a' if pct_rank is None else round(pct_rank, 1)} "
        f"of the ±{window} neighbourhood."
    )
    return _result(
        "neighbourhood_stats", params, ts, msg,
        value=_f(value),
        local_median=_f(median),
        local_robust_sigma=_f(sigma),
        robust_z=_f(robust_z),
        percentile_in_window=_f(pct_rank),
        local_min=_f(np.min(values)),
        local_max=_f(np.max(values)),
        n_valid=int(others.notna().sum()),
        pct_nan_in_window=_f(pct_nan),
    )


# ---------------------------------------------------------------------------
# 7. Local noise — is this stretch noisy, or is this point an outlier?
# ---------------------------------------------------------------------------
def _block_noise(diffs: np.ndarray, block: int) -> np.ndarray:
    """Robust first-difference scale of every non-overlapping *block* of ``diffs``.

    Vectorised deliberately: this runs on the whole record on every call, and the
    obvious ``rolling().apply(mad)`` is O(n·w) in Python and takes seconds on a
    two-year 15-minute series. A reshape plus two ``nanmedian``s along an axis is
    milliseconds, and non-overlapping blocks are the right unit anyway — the
    question is "how noisy is a window like this one", not a per-row curve.

    All-NaN blocks (a long gap) come back NaN and are dropped by the caller.
    """
    n = (len(diffs) // block) * block
    if n < block:
        return np.array([])
    grid = diffs[:n].reshape(-1, block)
    with np.errstate(invalid="ignore"):
        med = np.nanmedian(grid, axis=1, keepdims=True)
        sigma = _MAD_TO_SIGMA * np.nanmedian(np.abs(grid - med), axis=1)
    return sigma


def noise_context(
    source,
    at,
    field: str = VALUE_COL,
    window: str = "90min",
    elevated_ratio: float = ELEVATED_NOISE_RATIO,
) -> dict:
    """How noisy is the data around *at*, and does the point stand out within it?

    THE PROBLEM THIS EXISTS TO SOLVE. Every other measurement in this module
    describes one point against a *record-wide* scale, so a point that moves 18
    robust sigmas is 18 sigmas whether its neighbours are flat or thrashing. In a
    stretch where the sensor is noisy — or where the water is genuinely moving
    fast — the spike detectors fire on dozens of points, each one looks extreme by
    that record-wide standard, and the run deletes them as dozens of separate
    sensor failures. They are not separate failures. They are one noisy stretch,
    and the right output is a single decision about the stretch.

    Two numbers carry this, and they answer different halves of the question:

    ``noise_ratio``
        The window's robust sample-to-sample scale divided by the record-typical
        one. 1.0 is an ordinary stretch of this record; 10 means the data here is
        moving ten times as much as it usually does.
    ``point_step_sigmas_local``
        The point's own largest single-sample move, measured against **its own
        neighbourhood** rather than the record. This is the discriminator. A real
        artifact jumps far further than the samples around it are jumping; a false
        positive in a busy stretch is doing what everything near it is doing.

    MEASURED (``scratchpad/tune_noise_context.py``; 4 datasets across all three
    gauges, injected labels as truth). Three populations: 168 injected spikes that
    a ``flagUniLOF(thresh=1.5)`` run found, 662 **false positives** from the same
    run — points it flagged that the labels call normal water, i.e. exactly the
    population this docstring is about — and 1600 ordinary normal rows.

        metric                    true spike    false positive    normal
        noise_ratio                   1.2            11.3           1.0
        point_step_sigmas_local      17.1             1.6           1.1
        point_step_sigmas_global     18.7            16.7           1.0
        turning_fraction             0.55            0.27          0.55

    Read the third row first: the measurement the agent already had —
    ``slope_context``'s ``delta_before_sigmas``, which is scaled by the *record*
    step sigma — reads 18.7 on a real spike and 16.7 on a false positive. It
    cannot tell them apart at all (separation 0.53, a coin flip). Rescaling the
    same move against the local neighbourhood moves it to 17.1 vs 1.6, separation
    **0.87**. The information was always there; the denominator was wrong.

    Rules, scored as what they buy and what they cost:

        spare the point when...                    FPs spared   spikes lost
        point_step_sigmas_local < 5                   83.1%        16.7%
        noise_ratio > 5                               70.7%         7.1%
        noise_ratio > 3 and step_sigmas_local < 5     76.7%         3.0%
                                        (all three measured on the 15-min bases)

    RE-TUNED FOR THE 5-MIN BASES (2026-08-20). The rule above spares **0 of 104**
    surviving false positives on 01467200_l1: it was fitted on a FP population of
    storm limbs at median noise_ratio 11.3, and the 5-min FP population sits at
    median 2.2. The separator still holds — FP median noise_ratio 2.2 vs true-spike
    0.9, FP median step_sigmas_local 8.4 vs 14.9 — so only the thresholds moved.
    Refitted over 2,892 FPs and 392 true spikes pooled from 02054550 l1/l2 and
    040851385 l1/l2 (never the gauge we report on; scratchpad/tune_noise_spare_rule.py):

        noise_ratio > 2.0 and step_sigmas_local < 8   60.2%        4.1%   <- used

    Held out on 01467200_l1 the same rule spares 34 of 104 FPs and loses 0 of 57
    true spikes. Note the transfer gap — 60% sparing where fitted, 33% held out —
    the direction carries, the magnitude does not fully.

    THE WINDOW MATTERS AND ±90 MIN IS THE PEAK. Swept at ±1 / 1.5 / 2 / 3 h, the
    combined rule spares 83.7 / 76.7 / 72.4 / 61.0% of false positives at 6.5 /
    3.0 / 1.2 / 2.4% of true spikes. Widen it and the window fills with calm
    surroundings, the local scale falls back toward the record scale, and the
    measurement degrades into the global one it exists to replace.

    ``turning_fraction`` splits the two ways a stretch can be busy, which need
    different decisions: it is the share of samples where the series reverses
    direction. White noise reverses about half the time (a thrashing sensor, and
    also a calm baseline, both ≈0.55); a storm limb climbing steadily almost never
    does (≈0.27). So high ``noise_ratio`` with a high turning fraction is a noisy
    *sensor*, while high ``noise_ratio`` with a low one is water genuinely moving
    fast. Both mean "do not delete these points one at a time"; only the first is
    a data-quality problem at all.

    ``episode_start`` / ``episode_end`` bound the contiguous elevated-noise
    stretch containing the point, so a run can write **one** decision span over it
    instead of one per flagged row (§5).
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    params = {"field": field, "window": window, "elevated_ratio": elevated_ratio}

    step = _median_step(series)
    half = max(2, int(round(pd.Timedelta(window) / step)))
    block = 2 * half + 1

    diffs = series.diff().to_numpy()
    global_sigma = _step_sigma(series)

    lo, hi = max(0, pos - half), min(len(series), pos + half + 1)
    local_sigma = max(_mad_sigma(diffs[lo:hi]), _SIGMA_EPS)

    # Record-typical noise: the MEDIAN window, not the record-wide first-difference
    # scale. They are usually close, but the median block is the honest reference
    # for "is this window unusual", and it is what noise_percentile ranks against.
    raw_blocks = _block_noise(diffs, block)
    blocks = raw_blocks[np.isfinite(raw_blocks)] if raw_blocks.size else raw_blocks
    if blocks.size:
        baseline = max(float(np.median(blocks)), _SIGMA_EPS)
        percentile = float((blocks < local_sigma).mean() * 100)
    else:
        baseline = global_sigma
        percentile = None
    noise_ratio = local_sigma / baseline

    # The point's own move, against the local scale and the record scale. The gap
    # between the two IS the finding whenever this tool changes a decision.
    prev_value = series.iloc[pos - 1] if pos > 0 else np.nan
    next_value = series.iloc[pos + 1] if pos + 1 < len(series) else np.nan
    value = series.iloc[pos]
    adjacent = np.array([abs(value - prev_value), abs(next_value - value)])
    point_step = float(np.nanmax(adjacent)) if np.isfinite(adjacent).any() else np.nan

    window_steps = np.abs(diffs[lo:hi])
    window_steps = window_steps[np.isfinite(window_steps)]
    if window_steps.size and np.isfinite(point_step):
        pct_as_much = float((window_steps >= point_step).mean() * 100)
    else:
        pct_as_much = None

    # Direction reversals: separates a thrashing sensor from water moving fast.
    finite = diffs[lo:hi]
    finite = finite[np.isfinite(finite)]
    if finite.size >= 3:
        signs = np.sign(finite)
        turning = float((signs[1:] * signs[:-1] < 0).mean())
    else:
        turning = None

    # Extent of the elevated-noise stretch, so the agent can make ONE decision
    # about it rather than one per flagged row.
    episode_start = episode_end = None
    n_blocks = len(blocks) if blocks.size else 0
    if raw_blocks.size:
        b = min(pos // block, len(raw_blocks) - 1)
        elevated = np.where(np.isfinite(raw_blocks), raw_blocks, 0.0) >= elevated_ratio * baseline
        if elevated[b]:
            start = b
            while start - 1 >= 0 and elevated[start - 1]:
                start -= 1
            end = b
            while end + 1 < len(elevated) and elevated[end + 1]:
                end += 1
            episode_start = series.index[start * block]
            episode_end = series.index[min((end + 1) * block - 1, len(series) - 1)]

    if noise_ratio >= 5:
        regime = "severe"
    elif noise_ratio >= elevated_ratio:
        regime = "elevated"
    elif noise_ratio >= 0.6:
        regime = "typical"
    else:
        regime = "quiet"

    if turning is None:
        variation = "unknown"
    elif turning >= 0.45:
        variation = "noise-like"          # reverses constantly: the sensor is thrashing
    elif turning < 0.35:
        variation = "directional"         # moving steadily: real water, e.g. a storm limb
    else:
        variation = "mixed"

    stands_out = (
        point_step / local_sigma if np.isfinite(point_step) and local_sigma > 0 else None
    )
    # The measured rule. Deliberately stated as "unremarkable here", not "keep":
    # the verdict is the agent's (§7.3), this only says the point is not separable
    # from its surroundings.
    unremarkable = bool(
        noise_ratio > elevated_ratio
        and stands_out is not None
        and stands_out < LOCAL_STEP_UNREMARKABLE
    )

    if unremarkable:
        verdict = (
            f"This point does NOT stand out from its surroundings: it moves "
            f"{round(stands_out, 1)}x the local sample-to-sample scale in a stretch that is "
            f"already {round(noise_ratio, 1)}x noisier than this record's typical window. "
            + ("The variation here reverses direction constantly, so the SENSOR is noisy here. "
               if variation == "noise-like" else
               "The variation here is directional, so this is water genuinely moving fast, "
               "not sensor noise. " if variation == "directional" else "")
            + "Treat the stretch as one segment rather than deleting its points individually; "
              "measured, this pattern is a false positive ~4 times in 5."
        )
    elif stands_out is not None and stands_out >= 5:
        verdict = (
            f"This point DOES stand out: it moves {round(stands_out, 1)}x the local "
            f"sample-to-sample scale, so it is not explained by the variability around it."
        )
    else:
        verdict = "The point's own step could not be measured (it sits at an edge or a gap)."

    msg = (
        f"Local noise {round(local_sigma, 4)} vs record-typical {round(baseline, 4)} "
        f"= {round(noise_ratio, 1)}x ({regime}"
        + (f", {round(percentile, 1)}th percentile of windows" if percentile is not None else "")
        + f"; variation is {variation}). {verdict}"
    )

    return _result(
        "noise_context", params, ts, msg,
        value=_f(value),
        local_noise=_f(local_sigma),
        baseline_noise=_f(baseline),
        record_noise=_f(global_sigma),
        noise_ratio=_f(noise_ratio),
        noise_percentile=_f(percentile),
        noise_regime=regime,
        point_step=_f(point_step),
        point_step_sigmas_local=_f(stands_out),
        point_step_sigmas_global=_f(point_step / global_sigma) if np.isfinite(point_step) else None,
        pct_window_moving_as_much=_f(pct_as_much),
        turning_fraction=_f(turning),
        variation_kind=variation,
        point_unremarkable_here=unremarkable,
        episode_start=_iso(episode_start),
        episode_end=_iso(episode_end),
        episode_hours=_f(
            (episode_end - episode_start).total_seconds() / 3600.0
            if episode_start is not None else None
        ),
        window_samples=int(hi - lo),
        n_reference_windows=n_blocks,
    )


# ---------------------------------------------------------------------------
# 8. Gap proximity
# ---------------------------------------------------------------------------
def gap_context(
    source,
    at,
    field: str = VALUE_COL,
    window: str = "24h",
) -> dict:
    """Where the nearest missing-data runs sit relative to *at*.

    A reading on the edge of a dropout is a different kind of suspect from one in
    the middle of a dense stretch: telemetry artifacts cluster at gap boundaries.
    If the point itself is NaN, the enclosing run is reported instead.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    pos = _pos(series, ts)
    params = {"field": field, "window": window}

    is_nan = series.isna().to_numpy()
    runs = _bool_runs(is_nan)
    step = _median_step(series)

    enclosing = None
    prev_run = None
    next_run = None
    for start, end in runs:
        if start <= pos <= end:
            enclosing = (start, end)
        elif end < pos:
            prev_run = (start, end)
        elif start > pos and next_run is None:
            next_run = (start, end)

    def _describe(run: tuple[int, int] | None) -> dict | None:
        if run is None:
            return None
        start, end = run
        n_rows = end - start + 1
        return {
            "start": _iso(series.index[start]),
            "end": _iso(series.index[end]),
            "n_rows": n_rows,
            "minutes": _f(n_rows * step.total_seconds() / 60.0),
        }

    samples_to_prev = pos - prev_run[1] if prev_run else None
    samples_to_next = next_run[0] - pos if next_run else None
    adjacent = bool(
        (samples_to_prev is not None and samples_to_prev <= 1)
        or (samples_to_next is not None and samples_to_next <= 1)
    )

    span = pd.Timedelta(window)
    local = series.loc[ts - span : ts + span]
    local_runs = _bool_runs(local.isna().to_numpy())
    longest_local = max((end - start + 1 for start, end in local_runs), default=0)

    if enclosing is not None:
        msg = (
            f"This point is itself missing; it sits inside a gap of "
            f"{enclosing[1] - enclosing[0] + 1} row(s)."
        )
    elif adjacent:
        msg = "This point is immediately adjacent to a gap edge — a common artifact location."
    else:
        parts = []
        if samples_to_prev is not None:
            parts.append(f"{samples_to_prev} sample(s) after the previous gap")
        if samples_to_next is not None:
            parts.append(f"{samples_to_next} sample(s) before the next gap")
        msg = "Not adjacent to a gap" + (" (" + ", ".join(parts) + ")." if parts else ".")

    return _result(
        "gap_context", params, ts, msg,
        is_missing=bool(is_nan[pos]),
        enclosing_gap=_describe(enclosing),
        previous_gap=_describe(prev_run),
        next_gap=_describe(next_run),
        samples_to_previous_gap=samples_to_prev,
        samples_to_next_gap=samples_to_next,
        adjacent_to_gap=adjacent,
        pct_nan_in_window=_f(local.isna().mean() * 100),
        longest_gap_in_window_rows=longest_local,
    )


# ---------------------------------------------------------------------------
# 9. Historical precedent
# ---------------------------------------------------------------------------
def historical_context(
    source,
    at,
    field: str = VALUE_COL,
    exclude_window: str = "7D",
) -> dict:
    """Has this series reached this level before, and in how many separate episodes?

    Everything within ``exclude_window`` of *at* is removed first, so the event
    itself does not count as its own precedent. ``n_episodes_at_or_beyond``
    counts contiguous episodes rather than rows: 40 rows in one storm is one
    precedent, 40 rows across 12 storms is a regime.

    On a short record the requested exclusion could swallow everything, leaving
    no precedent to report. It is therefore capped at 10% of the record span, and
    the window actually used is echoed back in ``params``.
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)

    record_span = series.index[-1] - series.index[0]
    span = min(pd.Timedelta(exclude_window), record_span / 10)
    params = {
        "field": field,
        "exclude_window": exclude_window,
        "exclude_window_used": str(span),
    }

    value = series.loc[ts]
    rest = series.drop(series.loc[ts - span : ts + span].index).dropna()

    if not np.isfinite(value) or rest.empty:
        return _result(
            "historical_context", params, ts,
            "Value is missing or the rest of the record is empty; no precedent available.",
            value=_f(value), series_median=None, series_robust_sigma=None,
            percentile_in_record=None, n_rows_at_or_beyond=None,
            n_episodes_at_or_beyond=None, last_seen_at=None,
            days_since_last_seen=None, direction=None,
        )

    values = rest.to_numpy()
    median = float(np.median(values))
    above = value >= median
    mask = values >= value if above else values <= value
    episodes = _bool_runs(mask)

    prior = rest.loc[: ts - span]
    prior_hits = prior[prior >= value] if above else prior[prior <= value]
    last_seen = prior_hits.index[-1] if len(prior_hits) else None
    days_since = (ts - last_seen).total_seconds() / 86400.0 if last_seen is not None else None

    pct_rank = float((values < value).mean() * 100)
    msg = (
        f"Value {_f(value)} sits at percentile {round(pct_rank, 2)} of the record "
        f"(excluding ±{exclude_window}); the series reached this level in "
        f"{len(episodes)} separate episode(s) elsewhere"
        + (f", most recently {round(days_since, 1)} days earlier. " if days_since is not None
           else ". ")
        + ("Frequent precedent => this level is part of the regime."
           if len(episodes) >= 5 else "Little or no precedent => unusual for this sensor.")
    )
    return _result(
        "historical_context", params, ts, msg,
        value=_f(value),
        series_median=_f(median),
        series_robust_sigma=_f(_local_sigma(values, _step_sigma(series))),
        percentile_in_record=_f(pct_rank),
        n_rows_at_or_beyond=int(mask.sum()),
        n_episodes_at_or_beyond=len(episodes),
        last_seen_at=_iso(last_seen),
        days_since_last_seen=_f(days_since),
        direction="above_median" if above else "below_median",
    )


# ---------------------------------------------------------------------------
# 10. Aggregators
# ---------------------------------------------------------------------------
def _reads_like(parts: dict) -> tuple[str, str]:
    """Heuristic label + one-line justification from the individual contexts.

    Cut points were measured, not guessed: ``scratchpad/demo_point_context.py``
    prints the per-class quartiles they come from (level-2 datasets, all three
    gauges, injected labels as truth). What separates the classes:

      metric              spike    storm peak   normal   plateau
      |robust_z| (±6h)     4.3        1.3        0.1      0.0
      excursion width      2-3       7.5-25      2-14    22-43 samples
      samples to recover    2         12+         1        1
      step_sharpness      10.2        1.0        1.2      0.1

    The resulting rules and their measured hit rates on that sample:
      spike     |z|>=3 and width<=4      -> 87/120 spikes, 11 storm peaks, 0 normal
      plateau   flat run >=8 samples     -> 134/134 plateau rows, 9 storm peaks
      shift     see below                -> 6/9 shift onsets, 6 storm peaks (weak)
      gap       the point is NaN         -> exact

    The level_shift rule is weak and known to be: §7.2 and §9.1 already record
    that level shifts cannot be separated from storm limbs reliably, and this
    confirms it on different measurements. Treat that label as "look here".

    Deliberately conservative: anything that is not a clear match falls through
    to "inconclusive" rather than inventing a verdict the numbers do not support.
    """
    gap = parts.get("gap", {})
    if gap.get("is_missing"):
        return "gap", "the point is missing data."

    flat = parts.get("flatness", {})
    if (flat.get("run_length_at_point") or 0) >= 8:
        return (
            "plateau/stuck",
            f"{flat['run_length_at_point']} consecutive unchanged samples "
            f"(~{flat.get('run_minutes_at_point')} min).",
        )

    shift = parts.get("level_shift", {})
    recovery = parts.get("recovery", {})
    exc = parts.get("excursion", {})
    neighbourhood = parts.get("neighbourhood", {})

    width = exc.get("n_samples") or 0
    robust_z = neighbourhood.get("robust_z")
    abs_z = abs(robust_z) if robust_z is not None else None
    n_recover = recovery.get("n_samples_to_recover")
    step_sigmas = shift.get("step_sigmas")
    step_sharpness = shift.get("step_sharpness")
    hold_minutes = shift.get("hold_minutes")
    exc_sigmas = exc.get("excursion_sigmas")

    # Spike: far from its own neighbourhood AND narrow. Sign-agnostic — injected
    # spikes go both ways, and |z| is what recovers the downward ones.
    if abs_z is not None and abs_z >= 3 and 0 < width <= 4:
        # ...unless the whole stretch is doing this. A narrow high-z excursion in a
        # stretch already 3x noisier than the record, whose own step is under 5x the
        # LOCAL sample-to-sample scale, is a false positive ~4 times in 5 (measured;
        # see noise_context). Falling through to "spike" here is what produces a run
        # that deletes fifty points out of one noisy hour.
        noise = parts.get("noise", {})
        if noise.get("point_unremarkable_here"):
            return (
                "noisy-stretch",
                f"narrow and {round(robust_z, 1)} sigmas out by the RECORD's scale, but it "
                f"moves only {noise.get('point_step_sigmas_local')}x the LOCAL scale in a "
                f"stretch {noise.get('noise_ratio')}x noisier than typical "
                f"({noise.get('variation_kind')}) — judge the stretch, not the point.",
            )
        return (
            "spike",
            f"{width}-sample excursion at {round(robust_z, 1)} robust sigmas from the "
            f"local median, recovery in {n_recover} sample(s).",
        )

    # Level shift: the whole level moved and stayed moved, the move was not a
    # single-sample jump (that is a spike), and the point is not itself an
    # outlier — it is the level around it that changed.
    if (
        step_sigmas is not None
        and abs(step_sigmas) >= 3
        and hold_minutes is not None
        and hold_minutes >= SHIFT_MIN_HOLD_MINUTES
        and step_sharpness is not None
        and step_sharpness <= 2
        and (abs_z is None or abs_z < 2)
    ):
        return (
            "level_shift",
            f"level moved {round(step_sigmas, 1)} sigmas and held for "
            f"{round(hold_minutes / 60, 1)} h (weak signal — see §9.1).",
        )

    # Event: wide, substantial, and slow to come back, without being a local outlier.
    if (
        width >= 6
        and exc_sigmas is not None
        and abs(exc_sigmas) >= 2
        and (n_recover is None or n_recover >= 6)
        and (abs_z is None or abs_z < 3)
    ):
        return (
            "event/storm-like",
            f"{width} samples wide at {round(exc_sigmas, 1)} sigmas, "
            f"recovery in {n_recover} sample(s) — too broad and too slow for a point artifact.",
        )

    return (
        "inconclusive",
        "shape does not match any single signature cleanly; read the individual contexts.",
    )


def describe_point(
    source,
    at,
    field: str = VALUE_COL,
    # None -> derived from SLOPE_WINDOW (45 min) against the series' own sampling step,
    # matching slope_context. Hard-coding 3 samples here meant 45 min on 15-min data and
    # 15 min on 5-min data, and the fall/rise ratio inverts at the wrong width — see
    # slope_context's docstring for the measured table.
    n_before: int | None = None,
    n_after: int | None = None,
    window: str = "6h",
    shift_window: str = "24h",
) -> dict:
    """Run every context function on one timestamp and add a ``reads_like`` hint.

    This is the one-call version for the agent: it returns the full nested
    measurements plus a short heuristic label. The label is a hint drawn from the
    numbers below it, **not** a verdict — the agent decides the action (§8).
    """
    series = as_series(source, field)
    ts = resolve_timestamp(series, at)
    params = {
        "field": field,
        "n_before": n_before,
        "n_after": n_after,
        "window": window,
        "shift_window": shift_window,
    }

    parts = {
        "slope": slope_context(series, ts, field, n_before=n_before, n_after=n_after),
        "excursion": excursion_context(series, ts, field),
        "recovery": recovery_context(series, ts, field, baseline_window=window),
        "level_shift": level_shift_context(series, ts, field, window=shift_window),
        "flatness": flatness_context(series, ts, field, window=window),
        "neighbourhood": neighbourhood_stats(series, ts, field, window=window),
        # Deliberately NOT passed `window`: noise_context peaks at its own ±90 min
        # and degrades toward the global measurement as the window widens, so it
        # must not inherit the 6 h neighbourhood window (see its docstring).
        "noise": noise_context(series, ts, field),
        "gap": gap_context(series, ts, field),
        "history": historical_context(series, ts, field),
    }

    label, why = _reads_like(parts)
    msg = f"{_iso(ts)} = {_f(series.loc[ts])}: reads like a {label} — {why} (hint, not a verdict)"

    return _result(
        "describe_point", params, ts, msg,
        value=_f(series.loc[ts]),
        reads_like=label,
        reads_like_reason=why,
        **parts,
    )


# Points per describe_points call. Raised 20 -> 100 (2026-08-20): 20 was the
# binding constraint on how much of a detector's output the agent ever measured.
# On 01467200_l1 it measured 45 of 10,463 decided rows (0.43%) and deleted 367
# spikes having individually inspected 33 of them — and 194 of those deletions
# were normal water. Measured cost is ~90 tokens/point and flat from 20 to 263
# points, so 100 covers a typical full detector output in ~3 calls.
DEFAULT_MAX_POINTS = 100

# Hard ceiling, because the parameter is agent-settable and the cost is linear:
# `MAX_FLAGGED_DATETIMES` is 1000, so an uncapped call could return ~90k tokens
# in one result. 300 still covers a full detector output in a single call.
MAX_POINTS_CEILING = 300


def ramp_context(
    source,
    at,
    field: str = VALUE_COL,
    max_window: str = "6h",
    smooth: str = "20min",
) -> dict:
    """How long did the series take to CLIMB to this point, and to come back down?

    ``slope_context`` measures the gradient in the 45 minutes either side — the scale
    that separates a debris strike from a small flush. This measures the scale ABOVE
    that: the whole ramp. A storm peak is the top of a rise lasting hours; an artifact
    is a value that was not climbing to anything.

    The gap this closes, measured on 01467200_l1 at 2023-07-09 22:50 — turbidity 4.4
    -> 7.7 over two and a half hours, peaking at 13.5, back to 3.5 within the hour:
    every existing tool judged it on a window of 90 minutes or less, saw a narrow
    excursion, and the run deleted it as a spike. Nothing measured the ramp it sat on
    top of, because nothing looked further back than 90 minutes.

    Walks out from the point over a smoothed copy (a rolling median, so single-sample
    noise does not end the walk) to the lowest point within *max_window* on each side,
    and reports how long each leg took and how steadily it moved. ``monotonic_fraction``
    is the share of steps going the expected way — 1.0 is a clean climb, 0.5 is noise.

    Reports numbers and a conservative label; it does not decide. A long, steady rise
    is evidence the point is the top of something real, and evidence is what the agent
    weighs (§7.3).
    """
    series = as_series(source, field).dropna()
    ts = resolve_timestamp(series, at)
    params = {"field": field, "max_window": max_window, "smooth": smooth}
    if len(series) < 5 or ts not in series.index:
        return {"tool": "ramp_context", "params": params, "at": str(ts),
                "message": "Not enough data around this point to measure a ramp."}

    span = pd.Timedelta(max_window)
    step = pd.Series(series.index).diff().median()
    win = max(int(round(pd.Timedelta(smooth) / step)), 1) if step and step > pd.Timedelta(0) else 3
    # Smooth the NEIGHBOURHOOD, not the record. Rolling over a 210k-row series costs
    # ~50 ms per call and nothing outside +/-max_window can affect the answer; a small
    # pad keeps the centred window honest at the edges.
    local = series.loc[ts - span - pd.Timedelta(smooth): ts + span + pd.Timedelta(smooth)]
    smoothed = local.rolling(win, center=True, min_periods=1).median()

    peak = float(series.loc[ts])

    def leg(direction: int) -> dict:
        """Walk out to the foot of the ramp and measure how DIRECTLY it got there.

        The foot is the lowest smoothed value within *max_window*, but the walk stops
        early if the series was ever as high as the point itself — past that we are
        looking at a different, larger event, not the run-up to this one.

        Steadiness is a DIRECTNESS ratio, |net change| / sum|step changes|, not a share
        of steps going the right way. Two failures made that choice: a fraction-of-steps
        measure reads 0.5 on a clean climb that happens to be approached across flat
        baseline (the flat steps count against it), and it collapses on a real ramp with
        one local dip in it. Directness is immune to both — flat stretches add nothing to
        either side of the ratio, and a small dip costs only twice its own size. 1.0 is a
        straight climb; 0.3 is a series wandering to the same place.
        """
        lo, hi = (ts - span, ts) if direction < 0 else (ts, ts + span)
        seg = smoothed.loc[lo:hi]
        if len(seg) < 3:
            return {"minutes": None, "magnitude": None, "directness": None}

        ordered = seg.iloc[::-1] if direction < 0 else seg
        taller = np.flatnonzero(ordered.to_numpy() > peak)
        if len(taller):
            ordered = ordered.iloc[:taller[0]]        # stop at the larger event
        if len(ordered) < 2:
            return {"minutes": 0.0, "magnitude": 0.0, "directness": None}

        foot_at = ordered.idxmin()
        walk = smoothed.loc[foot_at:ts] if direction < 0 else smoothed.loc[ts:foot_at]
        if len(walk) < 2:
            return {"minutes": 0.0, "magnitude": 0.0, "directness": None}
        diffs = walk.diff().dropna().to_numpy()
        travelled = float(np.sum(np.abs(diffs)))
        net = abs(float(walk.iloc[-1] - walk.iloc[0]))
        return {
            "minutes": _round(abs((ts - foot_at).total_seconds()) / 60.0, 0),
            "magnitude": _round(peak - float(smoothed.loc[foot_at]), 2),
            "directness": _round(net / travelled if travelled > 0 else None, 2),
        }

    rise, fall = leg(-1), leg(+1)

    def wandering(part: dict) -> bool:
        """A long approach that did NOT go straight there.

        Note the direction, which is the opposite of the obvious guess and was measured
        rather than assumed: a true injected spike reads directness 0.75 — a clean jump
        straight up from its foot — while real water flagged in error reads 0.28, having
        meandered its way up. So a *wandering* climb is the water-like shape.
        """
        return bool(part["minutes"] and part["minutes"] >= RAMP_MINUTES
                    and part["directness"] is not None
                    and part["directness"] <= RAMP_DIRECTNESS)

    rising, falling = wandering(rise), wandering(fall)
    if rising and falling:
        label, why = "ramped", "wandered up and back down over hours — the shape of water moving"
    elif rising:
        label, why = "ramped-up", "wandered its way up over hours before this point"
    elif falling:
        label, why = "ramped-down", "wandered back down over hours after this point"
    else:
        label, why = "no-ramp", "no long, meandering approach — it went straight there, or nowhere"

    msg = (
        f"Rose {rise['magnitude']} over {rise['minutes']} min "
        f"(directness {rise['directness']}), fell {fall['magnitude']} over "
        f"{fall['minutes']} min (directness {fall['directness']}): {label} — {why}. "
    )
    msg += ("A long, meandering climb is what a catchment does; a displaced reading goes "
            "straight up from wherever the series was. Evidence for REAL WATER."
            if rising else
            "Nothing long and meandering leads into this point, which is consistent with "
            "an artifact — but check width, recovery and rainfall before concluding that.")
    return {"tool": "ramp_context", "params": params, "at": str(ts),
            "value": _f(peak), "rise": rise, "fall": fall,
            "reads_like": label, "message": msg}


def _edge_sharpness(series, at, k: int = 6) -> float | None:
    """Largest single-sample move at a transition, as a share of the net step across it.

    §9.1's lever, and the only one that works: a recalibration or sensor swap moves most
    of its magnitude in ONE sample, while a storm spreads the same net change over hours.
    1.0 is a cliff; 0.1 is a ramp.
    """
    pos = _pos(series, resolve_timestamp(series, at))
    lo, hi = max(0, pos - k), min(len(series), pos + k + 1)
    seg = series.iloc[lo:hi].to_numpy(dtype=float)
    if len(seg) < 4:
        return None
    diffs = np.abs(np.diff(seg))
    diffs = diffs[np.isfinite(diffs)]
    if not len(diffs):
        return None
    before = np.nanmedian(seg[: max(1, len(seg) // 3)])
    after = np.nanmedian(seg[-max(1, len(seg) // 3):])
    net = abs(after - before)
    if not np.isfinite(net) or net <= 0:
        return None
    return float(np.max(diffs) / net)


def shift_window_context(source, start, end, field: str = VALUE_COL, pad: str = "6h") -> dict:
    """Is the span between two jumps a LEVEL SHIFT, or a storm?

    The definition this implements: a level shift is the window between two jump points
    whose interior sits significantly above (or below) its surroundings — plus one clause
    the data forced, because elevation alone does not separate the two.

    WHY THE EXTRA CLAUSE. Pairing jumps and testing interior elevation recovers the whole
    shifted window, which is what row-scoring needs: measured on 01467200, row recall goes
    from ~0.5% (an edge detector against a window label) to 100%. But of 24 such windows on
    l1, ONE was the injected shift and 23 were storms — and the biggest storm scored z=+8.6
    against the real shift's z=+4.5, so elevation strength ranks them the wrong way round.
    §9.1 hit the same wall from the other side and found the lever that does work: EDGE
    SHARPNESS. A recalibration moves most of its magnitude in one sample; a storm takes
    hours to rise and hours to recede.
    """
    series = as_series(source, field)
    a = resolve_timestamp(series, start)
    b = resolve_timestamp(series, end)
    if b < a:
        a, b = b, a
    params = {"field": field, "pad": pad}

    inside = series.loc[a:b].to_numpy(dtype=float)
    padding = pd.Timedelta(pad)
    before = series.loc[a - padding: a].to_numpy(dtype=float)
    after = series.loc[b: b + padding].to_numpy(dtype=float)
    surround = np.concatenate([before, after])
    hours = (b - a).total_seconds() / 3600.0

    if np.sum(np.isfinite(inside)) < 4 or np.sum(np.isfinite(surround)) < 4:
        return {"tool": "shift_window_context", "params": params,
                "start": str(a), "end": str(b), "hours": _round(hours, 1),
                "message": "Too little data inside or around this window to judge it."}

    sigma = _MAD_TO_SIGMA * float(np.nanmedian(np.abs(surround - np.nanmedian(surround))))
    # Floor the scale on the record's own first-difference spread, the same guard §7.1
    # needs for quantised turbidity. Without it, surroundings that happen to be flat give
    # sigma == 0 and the elevation test divides by nothing — so the most obvious possible
    # shift, a clean step out of a quiet baseline, reads as "not shifted".
    floor = _MAD_TO_SIGMA * float(np.nanmedian(np.abs(np.diff(
        series.to_numpy(dtype=float)[np.isfinite(series.to_numpy(dtype=float))]))))
    if not np.isfinite(floor) or floor <= 0:
        floor = 1e-9
    sigma = max(sigma, floor)
    step = float(np.nanmean(inside) - np.nanmean(surround))
    interior_sigmas = step / sigma if sigma > 0 else None

    onset = _edge_sharpness(series, a)
    end_sharp = _edge_sharpness(series, b)
    sharp_edges = [x for x in (onset, end_sharp) if x is not None]
    abrupt = bool(sharp_edges) and min(sharp_edges) >= SHIFT_EDGE_SHARPNESS
    elevated = interior_sigmas is not None and abs(interior_sigmas) >= SHIFT_INTERIOR_SIGMAS

    if elevated and abrupt:
        label = "level-shift-like"
        why = ("the interior sits well away from its surroundings AND both edges are "
               "abrupt — the signature of a recalibration or sensor swap")
    elif elevated:
        label = "event-like"
        why = ("the interior is elevated, but at least one edge RAMPS rather than steps. "
               "A storm rises and recedes over hours; a shift does not. This is the "
               "distinction elevation alone cannot make")
    else:
        label = "not-shifted"
        why = "the interior is not meaningfully different from its surroundings"

    return {
        "tool": "shift_window_context", "params": params,
        "start": str(a), "end": str(b), "hours": _round(hours, 1),
        "interior_mean": _round(float(np.nanmean(inside)), 3),
        "surrounding_mean": _round(float(np.nanmean(surround)), 3),
        "interior_sigmas": _round(interior_sigmas, 1),
        "onset_sharpness": _round(onset, 2),
        "end_sharpness": _round(end_sharp, 2),
        "reads_like": label,
        "message": (
            f"{hours:.1f}h window: interior mean {np.nanmean(inside):.2f} vs surroundings "
            f"{np.nanmean(surround):.2f} ({_round(interior_sigmas, 1)} robust sigmas), "
            f"edges {_round(onset, 2)} / {_round(end_sharp, 2)} sharpness "
            f"(1.0 = one-sample cliff). Reads as {label} — {why}."
        ),
    }


def _jump_stamps_from_history(source, field: str) -> list:
    """Every timestamp flagJumps flagged on *field*, read from the SaQC history.

    §7.1: `qc.flags` attributes a doubly-flagged row to the FIRST test only, so the
    history is the only honest source for "which rows did this particular test flag".
    """
    try:
        history = source._flags.history[field]
    except (AttributeError, KeyError):
        return []
    stamps = []
    for col in history.hist.columns:
        if "jump" in str(history.meta[col].get("func", "")).lower():
            hit = history.hist[col] > 0
            stamps.extend(history.hist.index[hit.fillna(False)])
    return sorted(set(stamps))


def find_shift_windows(source, ats=None, field: str = VALUE_COL, min_hours: float = 1.0,
                       max_hours: float = 48.0, max_windows: int = 20) -> dict:
    """Pair up jump timestamps into candidate level-shift windows and rank them.

    **`ats` is optional, and omitting it is the right default** (2026-08-25). Given
    nothing, every timestamp flagJumps flagged is read straight from the SaQC history,
    so the agent cannot hand over a subset that leaves the real shift out. Run M did
    exactly that: it chose 76 of 145 jump timestamps, its subset excluded the injected
    shift's two edges, the tool returned one unrelated window, and the run concluded
    there were no level shifts. Runs K and L passed 134 and 53 and both happened to
    include it — so the failure is silent and depends on which subset gets picked.

    *ats* is a detector's flagged_datetimes — normally flag_jumps'. Every nearby pair
    inside the duration bounds becomes a candidate window, each is measured by
    :func:`shift_window_context`, and overlapping candidates are merged so one event is
    reported once rather than as thirty windows with sliding start times.
    """
    series = as_series(source, field)
    from_history = False
    if not ats:
        ats = _jump_stamps_from_history(source, field)
        from_history = True
        if not ats:
            return {
                "tool": "find_shift_windows",
                "params": {"field": field, "min_hours": min_hours,
                           "max_hours": max_hours},
                "n_jumps": 0, "n_windows": 0, "n_level_shift_like": 0, "windows": [],
                "message": (
                    "No jump flags found on this field, so there are no edges to pair "
                    "into windows. Run flag_jumps first (take `thresh` from "
                    "inspect_dataset's jump_scale.recommended_thresh)."
                ),
            }
    stamps = sorted({resolve_timestamp(series, a) for a in ats})
    params = {"field": field, "min_hours": min_hours, "max_hours": max_hours,
              "jumps_from": "flag history" if from_history else "caller"}

    raw = []
    for i, a in enumerate(stamps):
        for b in stamps[i + 1: i + 12]:
            hours = (b - a).total_seconds() / 3600.0
            if hours < min_hours:
                continue
            if hours > max_hours:
                break
            measured = shift_window_context(series, a, b, field=field)
            if measured.get("reads_like") in ("level-shift-like", "event-like"):
                raw.append(measured)
            break

    raw.sort(key=lambda r: (r["start"], r["end"]))
    merged = []
    for cand in raw:
        if merged and cand["start"] <= merged[-1]["end"]:
            keep = dict(max(merged[-1], cand, key=lambda r: abs(r.get("interior_sigmas") or 0)))
            keep["start"] = min(merged[-1]["start"], cand["start"])
            keep["end"] = max(merged[-1]["end"], cand["end"])
            merged[-1] = keep
        else:
            merged.append(cand)

    order = {"level-shift-like": 0, "event-like": 1}
    merged.sort(key=lambda r: (order.get(r["reads_like"], 2), -abs(r.get("interior_sigmas") or 0)))
    shown = merged[:max_windows]
    n_shift = sum(1 for r in merged if r["reads_like"] == "level-shift-like")
    return {
        "tool": "find_shift_windows", "params": params,
        "n_jumps": len(stamps), "n_windows": len(merged), "n_level_shift_like": n_shift,
        "windows": [{k: r[k] for k in ("start", "end", "hours", "interior_sigmas",
                                       "onset_sharpness", "end_sharpness", "reads_like")}
                    for r in shown],
        "message": (
            f"Paired {len(stamps)} jump(s) into {len(merged)} candidate window(s): "
            f"{n_shift} read as level-shift-like (abrupt edges), "
            f"{len(merged) - n_shift} as event-like (elevated but ramped — storms). "
            "A level-shift-like window is a claim about the WHOLE span: if you accept it, "
            "write one decision span covering it, not just its edges."
        ),
    }


def noise_profile(
    source,
    field: str = VALUE_COL,
    window: str = "3h",
    elevated_ratio: float = ELEVATED_NOISE_RATIO,
    max_stretches: int = 20,
) -> dict:
    """Where in the record is the sensor noisy? — a rolling scan, not fixed blocks.

    ``noise_context`` answers "is THIS point in a noisy stretch". This answers the
    record-level question a run needs *before* it starts judging points: which
    calendar stretches are noisy at all, and how much of the record they cover.
    Without it the agent judges every flagged point against the record-wide scale
    and deletes ordinary water that happens to sit in a busy week — the failure
    that produced 104 false positives on 01467200_l1.

    Rolling, deliberately. Fixed non-overlapping blocks (what ``_block_noise`` uses
    to build a *reference distribution*) dilute a stretch that straddles a boundary
    and can push it under the threshold in both halves; a rolling measure finds the
    real edges. The §7.4 warning that ``rolling().apply(mad)`` costs seconds does
    not bite here: this runs once per run, and it uses pandas' native rolling
    median rather than ``.apply``, so it is C-speed.

    Returns the elevated stretches (merged contiguous runs, widest first, capped at
    *max_stretches*) plus the counts, so a long noisy record cannot bloat the
    inspect_dataset payload.
    """
    series = as_series(source, field).dropna()
    if len(series) < 3:
        return {"tool": "noise_profile", "n_stretches": 0, "stretches": [],
                "share_of_record_elevated": 0.0,
                "message": "Series too short to profile noise."}

    params = {"field": field, "window": window, "elevated_ratio": elevated_ratio}
    step = pd.Series(series.index).diff().median()
    n = max(int(pd.Timedelta(window) / step), 3) if step and step > pd.Timedelta(0) else 36

    # Robust local scale from first differences, exactly the quantity noise_context
    # compares a point against — so the two agree about what "noisy" means.
    moves = series.diff().abs()
    local = moves.rolling(n, center=True, min_periods=max(3, n // 3)).median() * _MAD_TO_SIGMA
    typical = float(np.nanmedian(local.to_numpy()))
    if not np.isfinite(typical) or typical <= 0:
        return {"tool": "noise_profile", "params": params, "n_stretches": 0,
                "stretches": [], "share_of_record_elevated": 0.0,
                "message": "Noise scale is degenerate (quantised or constant series)."}

    ratio = local / typical
    hot = (ratio > elevated_ratio).to_numpy()

    stretches, start = [], None
    for i, is_hot in enumerate(hot):
        if is_hot and start is None:
            start = i
        elif not is_hot and start is not None:
            stretches.append((start, i - 1))
            start = None
    if start is not None:
        stretches.append((start, len(hot) - 1))

    # Bridge runs separated by less than one window. A rolling threshold flickers
    # across a single noisy episode and splits it into fragments — unbridged, one
    # storm reported as 796 "stretches", which is true and useless. Bridging at the
    # window width gives back the episodes a reader would point at on a plot.
    bridged: list[tuple[int, int]] = []
    for a, b in stretches:
        if bridged and a - bridged[-1][1] <= n:
            bridged[-1] = (bridged[-1][0], b)
        else:
            bridged.append((a, b))
    stretches = bridged

    rows = []
    for a, b in stretches:
        seg = ratio.iloc[a:b + 1]
        rows.append({
            "start": series.index[a].strftime("%Y-%m-%dT%H:%M:%S"),
            "end": series.index[b].strftime("%Y-%m-%dT%H:%M:%S"),
            "n_samples": int(b - a + 1),
            "peak_noise_ratio": _round(float(np.nanmax(seg.to_numpy())), 1),
            "median_noise_ratio": _round(float(np.nanmedian(seg.to_numpy())), 1),
        })
    covered = sum(r["n_samples"] for r in rows)
    rows.sort(key=lambda r: r["n_samples"], reverse=True)
    shown = rows[:max_stretches]

    share = covered / max(len(series), 1)
    msg = (
        f"{len(rows)} stretch(es) are more than {elevated_ratio}x noisier than this "
        f"record's typical {window} window, covering {100 * share:.1f}% of it."
    )
    if len(rows) > len(shown):
        msg += f" The {len(shown)} widest are listed; {len(rows) - len(shown)} more are not."
    if rows:
        msg += (" A flagged point inside one of these is far more likely to be ordinary "
                "water than a sensor artifact — judge it against its local neighbourhood, "
                "not against the record.")
    return {
        "tool": "noise_profile",
        "params": params,
        "record_typical_step_sigma": _round(typical, 4),
        "n_stretches": len(rows),
        "share_of_record_elevated": _round(share, 4),
        "stretches": shown,
        "message": msg,
    }


# --- flag_jumps' own scale (§7.6) -------------------------------------------
# The windows offered to the agent. All four are cheap (three rolling means over
# the record), so the block reports every one and lets the agent choose.
JUMP_WINDOWS = ("1h", "3h", "6h", "12h")

# The tuned defaults. `thresh` at the p99 of the series' OWN jump statistic, at a
# 6h window, is the measured knee: 13/14 injected level_shift events on 01467200
# for 134 candidates, and 131-162 candidates on every one of the nine datasets --
# a stable budget, which no multiple of any static summary stat gives.
JUMP_THRESH_QUANTILE = 0.99
JUMP_RECOMMENDED_WINDOW = "6h"


def _jump_statistic(series: pd.Series, window: str) -> np.ndarray:
    """``|mean(previous window) - mean(next window)|`` at every timestamp.

    This is exactly what ``flagJumps`` thresholds (SaQC 2.8
    ``breaks.flagJumps`` -> ``_getChangePoints`` with
    ``stat_func=|mean(x) - mean(y)|``), reproduced here so a threshold can be
    quoted as a quantile of the series' own distribution rather than guessed in
    data units. Backward window closed on the right, forward window open on the
    left, matching SaQC's split.
    """
    s = series.dropna()
    if len(s) < 2:
        return np.array([], dtype=float)
    bwd = s.rolling(window, min_periods=1).mean()
    fwd = s[::-1].rolling(window, min_periods=1, closed="left").mean()[::-1]
    return np.abs(bwd.to_numpy() - fwd.to_numpy())


def jump_scale(
    source,
    field: str = VALUE_COL,
    windows: tuple[str, ...] = JUMP_WINDOWS,
    quantile: float = JUMP_THRESH_QUANTILE,
) -> dict:
    """How big is a *big* step in THIS record? -- flag_jumps' threshold, measured.

    ``flag_jumps.thresh`` is in data units, and the run has no way to guess a
    workable one. The old guidance ("1-5 FNU, scale it to the series") failed in
    its unit, not merely its value: measured across the three 5-min gauges, the
    p99 of the 3h jump statistic is 2.9 x the value MAD on 01467200 and 1.4 x on
    040851385, but 33 x on 02054550, whose median is 1.9 FNU and whose storms
    swing hundreds of FNU. A run that
    reasoned "3.0 FNU is only 0.57 std" therefore set thresh at roughly the 60th
    percentile of ordinary window-to-window movement, got 2,865 flags across two
    years, correctly concluded they were storm limbs, and blanket-kept the lot --
    which is how a dataset with three injected level shifts scores zero recall.

    The fix is to measure the statistic flagJumps actually thresholds and quote a
    quantile of it, because that is the only normaliser on the same axis as the
    decision. At p99 the candidate count is 131-162 on every one of the nine
    injected datasets; at a fixed multiple of the value MAD the same setting
    ranges 58 to 4,054. ``scratchpad/tune_jumps_knee.py`` reproduces the sweep.

    Observes only -- no flags, no mutation.
    """
    series = as_series(source, field).dropna()
    params = {"field": field, "windows": list(windows), "quantile": quantile}
    if len(series) < 3:
        return {"tool": "jump_scale", "params": params, "by_window": {},
                "message": "Series too short to measure a jump scale."}

    by_window: dict[str, dict] = {}
    for window in windows:
        stat = _jump_statistic(series, window)
        if stat.size == 0 or not np.isfinite(stat).any():
            continue
        by_window[window] = {
            "p50": _round(float(np.nanquantile(stat, 0.50)), 3),
            "p90": _round(float(np.nanquantile(stat, 0.90)), 3),
            "p99": _round(float(np.nanquantile(stat, 0.99)), 3),
            "p99_9": _round(float(np.nanquantile(stat, 0.999)), 3),
            "suggested_thresh": _round(float(np.nanquantile(stat, quantile)), 3),
        }

    rec_window = JUMP_RECOMMENDED_WINDOW if JUMP_RECOMMENDED_WINDOW in by_window else (
        next(iter(by_window), None)
    )
    rec_thresh = by_window[rec_window]["suggested_thresh"] if rec_window else None

    msg = (
        f"flag_jumps scale measured on this record. Start at thresh={rec_thresh}, "
        f"window='{rec_window}' -- the p{quantile * 100:g} of this series' own "
        "window-to-window mean difference. Do NOT set thresh from the mean or std: "
        "a threshold below this leaves thousands of ordinary storm limbs flagged and "
        "tells you nothing. If the result is still more candidates than you can "
        "triage, step up to the p99_9 value for that window rather than guessing."
    ) if rec_thresh is not None else "Jump scale is degenerate (constant series)."

    return {
        "tool": "jump_scale",
        "params": params,
        "statistic": "abs(mean(previous window) - mean(next window)) -- what flagJumps thresholds",
        "by_window": by_window,
        "recommended_window": rec_window,
        "recommended_thresh": rec_thresh,
        "message": msg,
    }


def describe_points(
    source,
    ats,
    field: str = VALUE_COL,
    max_points: int = DEFAULT_MAX_POINTS,
    window: str = "6h",
) -> dict:
    """Compact context for a list of timestamps — e.g. a detector's flagged rows.

    Returns one small row per point (value, ``reads_like``, and the handful of
    numbers that drove it) plus a tally by label, so one call covers a whole
    detector output inside the 25-call budget (§2).

    Truncation is explicit: if more than ``max_points`` timestamps are supplied,
    the extras are reported in ``n_truncated`` and named in the message rather
    than silently dropped. ``max_points`` is additionally clamped to
    :data:`MAX_POINTS_CEILING`, and a clamp is reported the same way — measured
    cost is ~90 tokens per point, so an uncapped call on a 1000-timestamp
    detector output would return ~90k tokens in a single result.
    """
    series = as_series(source, field)
    requested = list(ats)
    capped = min(int(max_points), MAX_POINTS_CEILING)
    selected = requested[:capped]
    params = {"field": field, "max_points": capped, "window": window,
              "n_requested": len(requested)}
    if capped != int(max_points):
        params["max_points_requested"] = int(max_points)

    rows: list[dict] = []
    errors: list[dict] = []
    for at in selected:
        try:
            full = describe_point(series, at, field, window=window)
        except ValueError as exc:
            errors.append({"at": str(at), "error": str(exc)})
            continue
        exc_part = full["excursion"]
        slope_part = full["slope"]
        rec_part = full["recovery"]
        noise_part = full["noise"]
        rows.append({
            "at": full["at"],
            "value": full["value"],
            "reads_like": full["reads_like"],
            "reason": full["reads_like_reason"],
            "robust_z": full["neighbourhood"].get("robust_z"),
            "width_samples": exc_part.get("n_samples"),
            "peak_sharpness": exc_part.get("peak_sharpness"),
            "fall_rise_ratio": slope_part.get("fall_rise_ratio"),
            "samples_to_recover": rec_part.get("n_samples_to_recover"),
            "recovered": rec_part.get("recovered"),
            # Two noise columns, because triage is where they pay: scanning 100 rows
            # and seeing noise_ratio ~10 on all of them is how a run notices it is
            # looking at one noisy stretch and not 100 sensor failures.
            "noise_ratio": noise_part.get("noise_ratio"),
            "step_sigmas_local": noise_part.get("point_step_sigmas_local"),
        })

    tally: dict[str, int] = {}
    for row in rows:
        tally[row["reads_like"]] = tally.get(row["reads_like"], 0) + 1

    n_truncated = len(requested) - len(selected)
    msg = f"Described {len(rows)} of {len(requested)} point(s): {tally}."
    if n_truncated:
        msg += (
            f" {n_truncated} timestamp(s) were NOT described (max_points={capped}"
            + (f", clamped from the {int(max_points)} you asked for; the ceiling is "
               f"{MAX_POINTS_CEILING}" if capped != int(max_points) else "")
            + "); "
            "call again with the remaining timestamps if you need them."
        )
    if errors:
        msg += f" {len(errors)} timestamp(s) could not be resolved."

    return {
        "tool": "describe_points",
        "params": params,
        "message": msg,
        "n_described": len(rows),
        "n_truncated": n_truncated,
        "tally": tally,
        "points": rows,
        "errors": errors,
    }
