"""Demo: what src/agent_tools/context.py tells you about a point.

Run from the repo root (the module imports `src.*`):

    python -m scratchpad.demo_point_context
    python -m scratchpad.demo_point_context --dataset data/turbidity/injected/02198840/l3/02198840_l3.csv

Part 1 prints the full context for one point of each labelled kind — an injected
spike, a *genuine* high-turbidity peak the labels say is NOT an anomaly, a
plateau, a level shift, and a gap — so you can see which numbers separate them.

Part 2 tabulates each measurement per true class, which is where the cut points
in `context._reads_like` came from. It also tests the rise-vs-fall gradient idea
directly, across window sizes: that one does NOT work, and the table says why.

Part 3 runs `describe_points` over injected spikes vs genuine storm peaks and
cross-tabs the `reads_like` hint against the ground truth. That is the question
the agent actually faces: handed a flagged timestamp, can these numbers tell a
storm from an artifact?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_tools import context as ctx
from src.inspect_data import DATETIME_COL, load_series, reindex_to_grid, validate_labels_csv

DEFAULT_DATASET = Path("data/turbidity/injected/03447687/l2/03447687_l2.csv")


def load(dataset: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Load the series (re-gridded so gaps are explicit NaN) and its §5 labels."""
    df = reindex_to_grid(load_series(dataset))
    series = df.set_index(DATETIME_COL)["value"].astype(float)
    labels = validate_labels_csv(dataset.with_name(f"{dataset.stem}_labels.csv"))
    labels = labels.set_index(DATETIME_COL).reindex(series.index)
    labels["is_anomaly"] = labels["is_anomaly"].fillna(False).astype(bool)
    labels["anomaly_type"] = labels["anomaly_type"].fillna("")
    return series, labels


def pick_examples(series: pd.Series, labels: pd.DataFrame) -> dict[str, pd.Timestamp]:
    """One representative timestamp per case, chosen from the ground truth."""
    picks: dict[str, pd.Timestamp] = {}

    spikes = labels.index[labels["anomaly_type"] == "spike"]
    if len(spikes):
        # The most extreme injected spike, so the shape is unambiguous.
        deviation = (series[spikes] - labels.loc[spikes, "true_value"]).abs()
        picks["injected spike"] = deviation.idxmax()

    clean = series[~labels["is_anomaly"] & series.notna()]
    # The tallest reading the labels say is real water: a genuine storm peak.
    # Exclude readings sitting in a flat run — this sensor pegs at 1000 FNU during
    # big storms, and a clipped peak is its own (real) QC finding, not a storm shape.
    moving = clean[(clean.diff().abs() > 0) & (clean.diff(-1).abs() > 0)]
    if len(moving):
        picks["genuine peak (NOT an anomaly)"] = moving.idxmax()
    if len(clean):
        picks["clipped peak (also NOT an anomaly)"] = clean.idxmax()

    for kind, where in (("plateau", "mid"), ("level_shift", "onset"), ("gap", "mid")):
        rows = labels.index[labels["anomaly_type"] == kind]
        if not len(rows):
            continue
        longest = max(_runs(rows), key=len)
        # A level shift is judged at its transition; a plateau/gap in its middle.
        picks[f"injected {kind}"] = longest[0] if where == "onset" else longest[len(longest) // 2]
    return picks


def _runs(stamps: pd.DatetimeIndex) -> list[pd.DatetimeIndex]:
    """Split a sorted DatetimeIndex into contiguous-in-time runs."""
    if not len(stamps):
        return []
    gaps = stamps.to_series().diff()
    step = gaps.dropna().min()
    breaks = np.flatnonzero((gaps > step).to_numpy())
    return [pd.DatetimeIndex(part) for part in np.split(stamps, breaks) if len(part)]


# ---------------------------------------------------------------------------
# Part 1 — one point of each kind, in full
# ---------------------------------------------------------------------------
def show_point(series: pd.Series, at: pd.Timestamp, title: str) -> None:
    print("\n" + "=" * 78)
    print(f"{title}   @ {at}   value = {series.loc[at]}")
    print("=" * 78)

    full = ctx.describe_point(series, at)
    print(f"  reads_like : {full['reads_like']}  ({full['reads_like_reason']})")

    slope = full["slope"]
    print(
        f"\n  slope_context      rise {slope['slope_before_per_hour']} u/h -> "
        f"fall {slope['slope_after_per_hour']} u/h | shape={slope['shape']} "
        f"ratio={slope['fall_rise_ratio']} ({slope['symmetry']})"
    )
    exc = full["excursion"]
    print(
        f"  excursion_context  {exc['excursion']} above baseline {exc['baseline']} | "
        f"width={exc['n_samples']} samples ({exc['duration_minutes']} min) "
        f"sharpness={exc['peak_sharpness']} isolated={exc['isolated']}"
    )
    rec = full["recovery"]
    print(
        f"  recovery_context   recovered={rec['recovered']} after "
        f"{rec.get('n_samples_to_recover')} samples "
        f"({rec.get('minutes_to_recover')} min) to baseline {rec.get('baseline')}"
    )
    shift = full["level_shift"]
    print(
        f"  level_shift_ctx    {shift['median_before']} -> {shift['median_after']} "
        f"(step {shift['step']} = {shift['step_sigmas']} sigmas) "
        f"held for {shift['hold_minutes']} min"
    )
    flat = full["flatness"]
    print(
        f"  flatness_context   unchanged run at point={flat['run_length_at_point']} "
        f"(longest nearby {flat['longest_run']}), {flat['frac_unchanged']} of window flat, "
        f"n_unique={flat['n_unique']}"
    )
    nb = full["neighbourhood"]
    print(
        f"  neighbourhood      median={nb['local_median']} sigma={nb['local_robust_sigma']} "
        f"robust_z={nb['robust_z']} pctile={nb['percentile_in_window']}"
    )
    gap = full["gap"]
    print(
        f"  gap_context        missing={gap['is_missing']} adjacent_to_gap={gap['adjacent_to_gap']} "
        f"to_prev={gap['samples_to_previous_gap']} to_next={gap['samples_to_next_gap']}"
    )
    hist = full["history"]
    print(
        f"  historical_context pctile in record={hist['percentile_in_record']} | "
        f"{hist['n_episodes_at_or_beyond']} separate episode(s) reached this level, "
        f"last seen {hist['days_since_last_seen']} days earlier"
    )
    print(f"\n  message: {full['message']}")


# ---------------------------------------------------------------------------
# Part 2 — which measurement actually separates the classes?
# ---------------------------------------------------------------------------
def sample_by_class(
    series: pd.Series, labels: pd.DataFrame, n: int, rng: np.random.Generator
) -> dict[str, list[pd.Timestamp]]:
    """Sample timestamps per ground-truth class, including two 'not an anomaly' classes."""
    groups: dict[str, list[pd.Timestamp]] = {}
    clean_mask = ~labels["is_anomaly"] & series.notna()
    clean = series[clean_mask]

    spikes = labels.index[(labels["anomaly_type"] == "spike") & series.notna()]
    groups["spike"] = list(rng.choice(spikes, min(n, len(spikes)), replace=False))

    for kind in ("plateau", "level_shift"):
        runs = _runs(labels.index[labels["anomaly_type"] == kind])
        groups[f"{kind}_onset"] = [r[0] for r in runs][:n]

    # Genuine storm peaks: local maxima (max within ±2 h) among the highest
    # readings the labels call real water. These are what a spike detector trips on.
    local_max = series >= series.rolling("4h", center=True).max()
    peaks = series[local_max & clean_mask & (series >= clean.quantile(0.98))].index
    groups["storm_peak"] = list(rng.choice(peaks, min(n, len(peaks)), replace=False))

    mid = clean[(clean > clean.quantile(0.4)) & (clean < clean.quantile(0.6))].index
    groups["normal"] = list(rng.choice(mid, min(n, len(mid)), replace=False))
    return groups


def evidence_table(series: pd.Series, labels: pd.DataFrame, n: int, seed: int) -> None:
    """Per-class quartiles for every measurement + the slope-ratio window sweep."""
    rng = np.random.default_rng(seed)
    rows = []
    for truth, stamps in sample_by_class(series, labels, n, rng).items():
        for at in stamps:
            d = ctx.describe_point(series, at)
            row = {
                "truth": truth,
                "|robust_z|": abs(d["neighbourhood"]["robust_z"] or 0),
                "width": d["excursion"]["n_samples"],
                "|exc_sigmas|": abs(d["excursion"]["excursion_sigmas"] or 0),
                "recover": d["recovery"]["n_samples_to_recover"],
                "|step_sig|": abs(d["level_shift"]["step_sigmas"] or 0),
                "step_sharp": d["level_shift"]["step_sharpness"],
                "flat_run": d["flatness"]["run_length_at_point"],
                "hint": d["reads_like"],
            }
            # The user's hypothesis, swept over window sizes (samples on a 15-min grid).
            for k in (4, 8, 24, 96):
                row[f"fall/rise n{k}"] = ctx.slope_context(
                    series, at, n_before=k, n_after=k
                )["fall_rise_ratio"]
            rows.append(row)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 78)
    print("Median of each measurement, by ground-truth class")
    print("=" * 78)
    metrics = ["|robust_z|", "width", "|exc_sigmas|", "recover", "|step_sig|",
               "step_sharp", "flat_run"]
    print(df.groupby("truth")[metrics].median().round(2).to_string())
    print("\n  => |robust_z|, width and recover separate spike from storm_peak cleanly.")

    print("\n" + "-" * 78)
    print("The rise-vs-fall gradient idea, swept over window size (median ratio)")
    print("-" * 78)
    ratios = [c for c in df.columns if c.startswith("fall/rise")]
    print(df.groupby("truth")[ratios].median().round(2).to_string())
    print(
        "\n  => spike and storm_peak sit on top of each other at every window.\n"
        "     A storm's asymmetry is an EVENT-scale property (rise in hours,\n"
        "     recession over days), not a local gradient either side of one sample.\n"
        "     width + recovery time capture it; the gradient ratio does not."
    )


# ---------------------------------------------------------------------------
# Part 3 — does the hint separate artifacts from real water?
# ---------------------------------------------------------------------------
def crosstab(series: pd.Series, labels: pd.DataFrame, n_each: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    print("\n" + "=" * 78)
    print(f"describe_points on {n_each} injected spikes vs {n_each} genuine high peaks")
    print("=" * 78)

    groups = sample_by_class(series, labels, n_each, rng)
    spikes = labels.index[(labels["anomaly_type"] == "spike") & series.notna()]

    for truth, stamps in (("injected spike", groups["spike"]),
                          ("genuine peak", groups["storm_peak"])):
        result = ctx.describe_points(series, stamps, max_points=len(stamps))
        print(f"\n  truth = {truth:<15} n={result['n_described']}")
        for label, count in sorted(result["tally"].items(), key=lambda kv: -kv[1]):
            print(f"      reads_like={label:<18} {count:>3}")
        rows = pd.DataFrame(result["points"])
        if not rows.empty:
            print(
                "      medians: "
                f"|robust_z|={rows['robust_z'].abs().median():.2f}, "
                f"width={rows['width_samples'].median()} samples, "
                f"fall/rise={rows['fall_rise_ratio'].median():.2f}, "
                f"recovery={rows['samples_to_recover'].median()} samples"
            )

    # One truncation example: max_points is reported, never silent.
    truncated = ctx.describe_points(series, list(spikes[:10]), max_points=3)
    print(f"\n  truncation is explicit -> {truncated['message']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--n-each", type=int, default=40)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--json", action="store_true", help="Dump one full result as JSON.")
    args = parser.parse_args()

    series, labels = load(args.dataset)
    print(f"dataset: {args.dataset}  rows={len(series):,}  "
          f"{series.index[0]} -> {series.index[-1]}")

    picks = pick_examples(series, labels)
    for title, at in picks.items():
        show_point(series, at, title)

    evidence_table(series, labels, args.n_each, args.seed)
    crosstab(series, labels, args.n_each, args.seed)

    if args.json and picks:
        first = next(iter(picks.values()))
        print("\nfull JSON for one point (this is what the agent would receive):")
        print(json.dumps(ctx.describe_point(series, first), indent=2)[:4000])


if __name__ == "__main__":
    main()
