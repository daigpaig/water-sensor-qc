"""Does `recovery_context` report a fast recovery against an INVALID reference?

`markdowns/spike_error_profiles.md` §3 claims a metric bug: when the baseline after
an excursion sits below the baseline before it, the series crosses the pre-event
level within a sample or two "not because it recovered, but because it overshot
downward into a new regime" — and `recovery_context` then reports 1-2 samples,
which the prompt reads as the canonical artifact signature.

The doc puts that first and calls it "a small fix with a disproportionate effect".
Reading `context.py:566-598` the mechanism cannot be quite as stated: the recovery
band is TWO-SIDED (`abs(v - baseline) <= tolerance`), it needs `n_confirm=2`
consecutive samples inside, and the baseline is anchored at the excursion ONSET
rather than at the point. So a series that drops well below the pre-event level
falls THROUGH the band without lodging and reports `recovered=False`; the bug can
only bite in the narrower band where the post-event level is lower but still
within tolerance.

This script measures which of those is actually happening, on the wrongly-deleted
points of a scored run, with the correctly-deleted points as a control. Without
the control the audit is unfalsifiable: "every wrongly-deleted point sits at a
regime boundary" is only interesting if the correctly-deleted ones do not.

Run:

    .venv/bin/python -m scratchpad.audit_recovery_reference \
        data/turbidity/injected/01467200/l1/01467200_l1.csv \
        data/agent_runs/01467200_l1_flags.json

Nothing here writes to the project; it measures and prints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences

from src.agent_tools import context as ctx
from src.inspect_data import DATETIME_COL, VALUE_COL

# A decision only matters here if it destroyed or rewrote a value: §7.7 makes the
# same scoping choice for the precipitation audit.
IRREVERSIBLE = frozenset({"delete", "correct"})

# §3's own proposed cutoff for `recovery_reference_valid`.
SHIFT_SIGMAS_INVALID = 2.0
# The prompt reads <= 2 samples as the artifact signature (context.py:626).
FAST_RECOVERY_SAMPLES = 2

BASELINE_WINDOW = pd.Timedelta("6h")
SUPPORT_WINDOW = pd.Timedelta("6h")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_series(path: Path) -> pd.Series:
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    return df.set_index(DATETIME_COL)[VALUE_COL].sort_index()


def load_labels(path: Path) -> pd.DataFrame:
    labels = path.with_name(f"{path.stem}_labels.csv")
    return pd.read_csv(labels, parse_dates=[DATETIME_COL]).set_index(DATETIME_COL)


def load_spike_decisions(path: Path) -> pd.DataFrame:
    """Rows the run claimed as a SPIKE and then deleted or corrected."""
    entries = json.loads(path.read_text())
    rows = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        if str(e.get("verdict", "")).lower() != "anomaly":
            continue
        if str(e.get("anomaly_type", "")).lower() != "spike":
            continue
        if str(e.get("action", "")).lower() not in IRREVERSIBLE:
            continue
        stamp = pd.to_datetime(e.get("datetime"), errors="coerce")
        if pd.isna(stamp):
            continue
        rows.append(
            {
                DATETIME_COL: stamp,
                "action": e.get("action"),
                "difficulty": (e.get("decided_by") or {}).get("difficulty"),
                "rationale_source": e.get("rationale_source"),
            }
        )
    return pd.DataFrame(rows).set_index(DATETIME_COL).sort_index()


def group_events(stamps: pd.DatetimeIndex, step: pd.Timedelta) -> list[list[pd.Timestamp]]:
    """Contiguous runs of decided rows are ONE event, not three.

    §6 defines a spike as one/few values, and §9 injects 1-3 samples, so scoring
    per row would triple-count a single wrong call.
    """
    events: list[list[pd.Timestamp]] = []
    for stamp in stamps:
        if events and (stamp - events[-1][-1]) <= step * 1.5:
            events[-1].append(stamp)
        else:
            events.append([stamp])
    return events


# ---------------------------------------------------------------------------
# The measurement §3 says is missing
# ---------------------------------------------------------------------------
def baseline_shift(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp,
                   step_sigma: float) -> dict:
    """Robust level before the excursion vs after it, and the step between."""
    pre = series.loc[start - BASELINE_WINDOW: start].iloc[:-1].dropna()
    post = series.loc[end: end + BASELINE_WINDOW].iloc[1:].dropna()
    if pre.empty or post.empty:
        return {"baseline_before": None, "baseline_after": None,
                "shift": None, "shift_sigmas": None, "pre_sigma": None}

    before = float(np.median(pre.to_numpy()))
    after = float(np.median(post.to_numpy()))
    pre_sigma = ctx._local_sigma(pre.to_numpy(), step_sigma)
    return {
        "baseline_before": before,
        "baseline_after": after,
        "shift": after - before,
        # Scaled by the SAME sigma recovery_context builds its tolerance from, so
        # the two numbers are directly comparable.
        "shift_sigmas": (after - before) / pre_sigma if pre_sigma else None,
        "pre_sigma": pre_sigma,
    }


def support_quality(series: pd.Series, ts: pd.Timestamp, step: pd.Timedelta) -> dict:
    """§4: how much of the measurement window actually exists."""
    window = series.loc[ts - SUPPORT_WINDOW: ts + SUPPORT_WINDOW]
    expected = int(2 * SUPPORT_WINDOW / step) + 1
    valid = int(window.notna().sum())
    runs = ctx._bool_runs(window.isna().to_numpy())
    longest = max((b - a for a, b in runs), default=0)
    return {
        "support_fraction": valid / expected if expected else None,
        "longest_gap_min": longest * step.total_seconds() / 60.0,
    }


def monotone_runs(series: pd.Series, ts: pd.Timestamp, tol: float, k: int = 24) -> dict:
    """§5: consecutive rising samples before the peak, falling samples after.

    A step must EXCEED `tol` to extend the run. Counting a flat step as "still
    rising" is not a neutral choice here: §9 injects a spike as a RECTANGULAR
    displacement of 1-3 samples (`inject.py:316`), so its flat top would extend
    the run and hand the injected spikes the very shape the measure is meant to
    find on real water.
    """
    pos = series.index.get_indexer([ts])[0]
    before = series.iloc[max(0, pos - k): pos + 1].to_numpy()
    after = series.iloc[pos: pos + k + 1].to_numpy()

    n_up = 0
    for i in range(len(before) - 1, 0, -1):
        d = before[i] - before[i - 1]
        if not np.isfinite(d) or d <= tol:
            break
        n_up += 1
    n_down = 0
    for i in range(len(after) - 1):
        d = after[i + 1] - after[i]
        if not np.isfinite(d) or d >= -tol:
            break
        n_down += 1
    return {"n_monotone_before": n_up, "n_monotone_after": n_down}


def prominence_table(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Local maxima and their topographic prominence (§2).

    Peaks below one robust step-sigma are excluded: on quantised 5-min turbidity
    every wiggle is a local maximum, and counting those makes
    `n_peaks_half_prominence` a measure of the record's noise rather than of how
    often it repeats a feature.
    """
    values = series.to_numpy(dtype=float)
    filled = pd.Series(values).ffill().bfill().to_numpy()
    floor = ctx._step_sigma(series)
    peaks, _ = find_peaks(filled, prominence=floor)
    prom = peak_prominences(filled, peaks)[0]
    return peaks, prom


# ---------------------------------------------------------------------------
# Per-point profile
# ---------------------------------------------------------------------------
def profile(series: pd.Series, ts: pd.Timestamp, step: pd.Timedelta,
            step_sigma: float, peaks: np.ndarray, prom: np.ndarray) -> dict:
    exc = ctx.excursion_context(series, ts)
    rec = ctx.recovery_context(series, ts)
    desc = ctx.describe_point(series, ts)

    start = pd.Timestamp(exc["start"]) if exc.get("start") else ts
    end = pd.Timestamp(exc["end"]) if exc.get("end") else ts
    shift = baseline_shift(series, start, end, step_sigma)

    row: dict = {
        "datetime": ts,
        "value": float(series.loc[ts]),
        "reads_like": desc.get("reads_like"),
        "n_samples_to_recover": rec.get("n_samples_to_recover"),
        "recovered": rec.get("recovered"),
        "tolerance": rec.get("tolerance"),
        "width_samples": exc.get("n_samples"),
        "peak_sharpness": exc.get("peak_sharpness"),
        "excursion_sigmas": exc.get("excursion_sigmas"),
    }
    row.update(shift)
    row.update(support_quality(series, ts, step))
    row.update(monotone_runs(series, ts, tol=0.5 * step_sigma))

    # §3's two derived conditions.
    ss = row.get("shift_sigmas")
    row["recovery_reference_invalid"] = (
        ss is not None and abs(ss) > SHIFT_SIGMAS_INVALID
    )
    # The precise mechanism claimed: post level lower, yet inside the band the
    # recovery test uses — so "returned to baseline" is a regime change, not a
    # recovery.
    tol = row.get("tolerance")
    row["post_below_but_inside_band"] = (
        row.get("shift") is not None and tol is not None
        and row["shift"] < 0 and abs(row["shift"]) <= tol
    )
    row["fast_recovery"] = (
        row["n_samples_to_recover"] is not None
        and row["n_samples_to_recover"] <= FAST_RECOVERY_SAMPLES
    )

    # §2 prominence percentile.
    pos = series.index.get_indexer([ts])[0]
    if len(peaks):
        hit = np.abs(peaks - pos)
        nearest = int(np.argmin(hit))
        if hit[nearest] <= 2:
            p = float(prom[nearest])
            row["prominence"] = p
            row["prominence_pct"] = float((prom <= p).mean() * 100)
            row["n_peaks_half_prom"] = int((prom >= 0.5 * p).sum())
        else:
            row["prominence"] = row["prominence_pct"] = row["n_peaks_half_prom"] = None
    return row


def build(series: pd.Series, labels: pd.DataFrame, decided: pd.DataFrame) -> pd.DataFrame:
    step = ctx._median_step(series)
    step_sigma = ctx._step_sigma(series)
    peaks, prom = prominence_table(series)

    is_anom = labels["is_anomaly"].reindex(series.index).fillna(False).astype(bool)
    rows = []
    for event in group_events(decided.index, step):
        # The event's representative is its most extreme sample.
        vals = series.reindex(event)
        ts = vals.idxmax() if vals.notna().any() else event[0]
        rec = profile(series, ts, step, step_sigma, peaks, prom)
        rec["n_rows"] = len(event)
        # Truth for the EVENT: injected anywhere in it => a correct deletion.
        rec["truth"] = "true_spike" if bool(is_anom.reindex(event).any()) else "real_water"
        rec["difficulty"] = decided.loc[event[0], "difficulty"]
        rec["rationale_source"] = decided.loc[event[0], "rationale_source"]
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _rate(frame: pd.DataFrame, col: str) -> str:
    if frame.empty:
        return "     n/a"
    vals = frame[col].fillna(False).astype(bool)
    return f"{vals.sum():3d}/{len(frame):3d} ({vals.mean() * 100:4.1f}%)"


def report(df: pd.DataFrame) -> None:
    wrong = df[df["truth"] == "real_water"]
    right = df[df["truth"] == "true_spike"]

    print(f"\nSpike events deleted/corrected: {len(df)} "
          f"({len(wrong)} on real water, {len(right)} on injected spikes)\n")

    print("§3  RECOVERY REFERENCE")
    print("-" * 72)
    print(f"{'condition':<44}{'wrongly deleted':>17}{'correctly':>11}")
    for col, name in [
        ("fast_recovery", "recovery <= 2 samples (artifact signature)"),
        ("recovery_reference_invalid", "|baseline shift| > 2 sigma"),
        ("post_below_but_inside_band", "post level lower AND inside recovery band"),
    ]:
        print(f"{name:<44}{_rate(wrong, col):>17}{_rate(right, col):>11}")

    both = wrong[wrong["fast_recovery"] & wrong["recovery_reference_invalid"]]
    both_r = right[right["fast_recovery"] & right["recovery_reference_invalid"]]
    print(f"{'BUG FIRES: fast recovery on invalid ref':<44}"
          f"{_rate(wrong, 'fast_recovery') if both.empty else f'{len(both):3d}/{len(wrong):3d} ({len(both)/max(len(wrong),1)*100:4.1f}%)':>17}"
          f"{f'{len(both_r):3d}/{max(len(right),1):3d}':>11}")

    print("\nOTHER PROFILES (same events)")
    print("-" * 72)
    print(f"{'condition':<44}{'wrongly deleted':>17}{'correctly':>11}")
    df2 = df.assign(
        unsupported=(df["support_fraction"] < 0.7) | (df["longest_gap_min"] > 30),
        broad_base=(df["n_monotone_before"] >= 3) | (df["n_monotone_after"] >= 3),
        pattern_typical=df["n_peaks_half_prom"].fillna(0) >= 50,
    )
    w2, r2 = df2[df2["truth"] == "real_water"], df2[df2["truth"] == "true_spike"]
    for col, name in [
        ("unsupported", "§4 support < 0.7 or gap > 30 min in window"),
        ("broad_base", "§5 monotone run >= 3 either side"),
        ("pattern_typical", "§2 >= 50 record peaks at half prominence"),
    ]:
        print(f"{name:<44}{_rate(w2, col):>17}{_rate(r2, col):>11}")

    covered = w2["unsupported"] | w2["broad_base"] | w2["pattern_typical"] | \
        w2["recovery_reference_invalid"].fillna(False)
    covered_r = r2["unsupported"] | r2["broad_base"] | r2["pattern_typical"] | \
        r2["recovery_reference_invalid"].fillna(False)
    print(f"\n{'ANY of the four flags (§6 OR-rule)':<44}"
          f"{f'{covered.sum():3d}/{len(w2):3d} ({covered.mean()*100:4.1f}%)':>17}"
          f"{f'{covered_r.sum():3d}/{max(len(r2),1):3d} ({covered_r.mean()*100:4.1f}%)':>11}")
    print("  (left column is the goal: every wrong deletion carries a flag.")
    print("   right column is the cost: true spikes forced to judgement-call.)")

    print("\nMEDIANS")
    print("-" * 72)
    cols = ["shift_sigmas", "n_samples_to_recover", "width_samples", "peak_sharpness",
            "n_monotone_before", "n_monotone_after", "n_peaks_half_prom", "prominence_pct"]
    med = pd.DataFrame({
        "wrongly deleted": wrong[cols].median(numeric_only=True),
        "correctly deleted": right[cols].median(numeric_only=True),
    })
    print(med.round(2).to_string())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("series", type=Path)
    ap.add_argument("flags", type=Path)
    ap.add_argument("--out", type=Path, help="Write the per-event table here.")
    args = ap.parse_args(argv)

    series = load_series(args.series)
    labels = load_labels(args.series)
    decided = load_spike_decisions(args.flags)
    if decided.empty:
        print("No spike delete/correct decisions in this flag log.")
        return 1

    df = build(series, labels, decided)
    report(df)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
