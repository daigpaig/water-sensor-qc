"""Tune flag_jumps.thresh against the injected level_shift labels (§7.2).

The shipped suggestion (1-5 FNU, default 2) was set on the retired 15-min gauges and
is far too low for the 5-min bases: on 01467200 (mean 7.72 FNU) thresh=3.0 flags
2,865 rows -- storm limbs, ~3-4 per day -- and the agent blanket-keeps the lot.

This sweeps thresh expressed in units of the series' own robust scale, and scores
EVENT-level recall (does any flag land on the injected shift?) and how many
candidate episodes the agent would have to adjudicate.

Run from a file, never a heredoc (§7.1 multiprocessing trap does not apply to
flagJumps, but keep the habit).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import saqc

ROOT = Path(__file__).resolve().parents[1]
INJECTED = ROOT / "data" / "injected"

WINDOWS = ["1h", "3h", "6h", "12h"]
# thresh as a multiple of the series' robust (MAD) scale
KS = [0.5, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20]


def robust_scale(x: np.ndarray) -> float:
    """Global MAD * 1.4826 -- identical to inject._robust_scale."""
    f = x[np.isfinite(x)]
    mad = float(np.median(np.abs(f - np.median(f))) * 1.4826)
    return mad if mad > 0 else float(np.std(f))


def step_scale(x: np.ndarray) -> float:
    """Robust scale of the FIRST DIFFERENCES -- the sample-to-sample noise."""
    d = np.diff(x)
    d = d[np.isfinite(d)]
    return float(np.median(np.abs(d - np.median(d))) * 1.4826)


def label_spans(labels: pd.DataFrame, kind: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    m = (labels["anomaly_type"] == kind).to_numpy()
    idx = pd.DatetimeIndex(labels["datetime"])
    spans, start = [], None
    for i, v in enumerate(m):
        if v and start is None:
            start = i
        elif not v and start is not None:
            spans.append((idx[start], idx[i - 1]))
            start = None
    if start is not None:
        spans.append((idx[start], idx[-1]))
    return spans


def cluster(times: pd.DatetimeIndex, tol: pd.Timedelta) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(times) == 0:
        return []
    out, s, prev = [], times[0], times[0]
    for t in times[1:]:
        if t - prev > tol:
            out.append((s, prev))
            s = t
        prev = t
    out.append((s, prev))
    return out


def main() -> None:
    rows = []
    for series_csv in sorted(INJECTED.glob("*/l*/*.csv")):
        if series_csv.name.endswith("_labels.csv"):
            continue
        stem = series_csv.stem
        labels = pd.read_csv(series_csv.with_name(f"{stem}_labels.csv"), parse_dates=["datetime"])
        df = pd.read_csv(series_csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
        vals = df["value"].to_numpy(dtype=float)

        mad = robust_scale(vals)
        smad = step_scale(vals)
        med = float(np.nanmedian(vals))
        std = float(np.nanstd(vals, ddof=1))
        truth = label_spans(labels, "level_shift")

        for window in WINDOWS:
            tol = pd.Timedelta(window)
            for k in KS:
                thresh = k * mad
                qc = saqc.SaQC(df[["value"]].copy())
                try:
                    out = qc.flagJumps("value", thresh=thresh, window=window)
                except Exception as exc:  # noqa: BLE001
                    print(f"  {stem} w={window} k={k}: FAILED {exc}")
                    continue
                flags = out.flags["value"]
                hit = pd.DatetimeIndex(flags.index[flags.to_numpy() > 0])

                clusters = cluster(hit, tol)
                detected = sum(
                    any(c[1] >= s - tol and c[0] <= e + tol for c in clusters) for s, e in truth
                )
                tp_clusters = sum(
                    any(c[1] >= s - tol and c[0] <= e + tol for s, e in truth) for c in clusters
                )
                rows.append(
                    dict(
                        dataset=stem, gauge=stem.split("_")[0], level=stem.split("_")[1],
                        median=med, std=std, mad=mad, step_mad=smad,
                        window=window, k=k, thresh=thresh,
                        n_events=len(truth), detected=detected,
                        n_flagged=len(hit), pct_flagged=100.0 * len(hit) / len(df),
                        n_clusters=len(clusters), fp_clusters=len(clusters) - tp_clusters,
                    )
                )
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "scratchpad" / "tune_jumps_results.csv", index=False)

    print("\n=== series scales ===")
    print(res.groupby("dataset")[["median", "std", "mad", "step_mad"]].first().round(3).to_string())

    for window in WINDOWS:
        w = res[res.window == window]
        print(f"\n=== window={window}: recall (events found / total) and FP candidate episodes ===")
        rec = w.groupby("k").apply(
            lambda g: pd.Series(
                {
                    "recall": g.detected.sum() / g.n_events.sum(),
                    "events": f"{g.detected.sum()}/{g.n_events.sum()}",
                    "fp_clusters_median": g.fp_clusters.median(),
                    "fp_clusters_max": g.fp_clusters.max(),
                    "pct_flagged_median": g.pct_flagged.median(),
                }
            ),
            include_groups=False,
        )
        print(rec.round(3).to_string())


if __name__ == "__main__":
    main()
