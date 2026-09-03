"""Sweep the spike detector params for >=95% recall on the injected spikes.

Recall is what matters here, not precision: candidates.py proposes, a human
disposes (CLAUDE.md §9.1). Missed spikes never reach the reviewer at all.

Spike candidates are never bridged and exclude NaN rows, so a spike candidate's
row coverage IS the detector union mask — recall can be read straight off the
mask without assembling Candidates, and without the max_per_type cap.

    .venv/bin/python -m scratchpad.tune_spike_recall [diagnose|unilof|zscore|combo]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.inspect_data import DATETIME_COL, VALUE_COL, load_series, reindex_to_grid
from src.workbench.candidates import DetectConfig, _run_saqc, _runs, robust_scales

ROOT = Path("data/injected")


def datasets(levels: tuple[str, ...] = ("l1", "l2", "l3")) -> list[Path]:
    return sorted(
        p for p in ROOT.glob("*/l[1-3]/*_l[1-3].csv")
        if p.parent.name in levels
    )


def load(path: Path):
    """``(series, injected-spike mask, step_scale)`` for one dataset."""
    df = reindex_to_grid(load_series(path, value_col=VALUE_COL), value_col=VALUE_COL)
    series = df.set_index(DATETIME_COL)[VALUE_COL].astype(float).sort_index()
    series.name = VALUE_COL
    labels = pd.read_csv(
        path.with_name(f"{path.stem}_labels.csv"), parse_dates=["datetime"]
    ).set_index("datetime").reindex(series.index)
    spikes = (
        labels["is_anomaly"].fillna(False).to_numpy(dtype=bool)
        & (labels["source"].fillna("").to_numpy() == "injected")
        & (labels["anomaly_type"].fillna("").to_numpy() == "spike")
    )
    return series, spikes, robust_scales(series)[1]


def score(mask: np.ndarray, spikes: np.ndarray, series: pd.Series) -> str:
    mask = mask & ~series.isna().to_numpy()
    hits = int((mask & spikes).sum())
    events = _runs(spikes)
    found = sum(1 for a, b in events if mask[a:b].any())
    return (
        f"recall {100 * hits / spikes.sum():5.1f}% ({hits}/{int(spikes.sum())})  "
        f"events {found}/{len(events)}  "
        f"flagged {int(mask.sum()):>5} rows / {len(_runs(mask)):>4} segments  "
        f"FP {int((mask & ~spikes).sum()):>5}"
    )


# ------------------------------------------------------------------- diagnose
def diagnose() -> None:
    """How big are the injected spikes, and are the missed ones the small ones?"""
    cfg = DetectConfig()
    for path in datasets(("l3",)):
        series, spikes, step = load(path)
        values = series.to_numpy()
        base = series.rolling(97, center=True, min_periods=5).median().to_numpy()
        mask = _run_saqc(
            series, "flagUniLOF",
            {"n": cfg.unilof_n, "thresh": cfg.unilof_thresh,
             "density": "auto", "slope_correct": True},
        ).to_numpy()
        print(f"\n{path.stem}  step_scale={step:.4g}")
        mags = []
        for a, b in _runs(spikes):
            m = np.nanmax(np.abs(values[a:b] - base[a:b])) / step
            mags.append((m, bool(mask[a:b].any()), b - a))
        arr = np.array([m for m, _, _ in mags])
        hit = np.array([h for _, h, _ in mags])
        print(f"  spike magnitude in step-sigmas: min={arr.min():.1f} "
              f"p10={np.percentile(arr, 10):.1f} median={np.median(arr):.1f} "
              f"max={arr.max():.1f}")
        for lo, hi in ((0, 5), (5, 10), (10, 25), (25, 100), (100, 1e9)):
            sel = (arr >= lo) & (arr < hi)
            if sel.any():
                print(f"    {lo:>3}-{hi:<5.0f} sigma: {sel.sum():>3} events, "
                      f"UniLOF@1.5 found {hit[sel].sum():>3} "
                      f"({100 * hit[sel].mean():.0f}%)")
        lens = np.array([n for _, _, n in mags])
        print(f"  event length rows: {np.bincount(lens)[1:]} (index = 1,2,3...)")


# --------------------------------------------------------------------- sweeps
def unilof() -> None:
    for path in datasets(("l3",)):
        series, spikes, step = load(path)
        print(f"\n{path.stem}")
        for n in (10, 20, 40):
            for thresh in (1.0, 1.05, 1.1, 1.2, 1.3, 1.5):
                mask = _run_saqc(
                    series, "flagUniLOF",
                    {"n": n, "thresh": thresh, "density": "auto",
                     "slope_correct": True},
                ).to_numpy()
                print(f"  n={n:<3} thresh={thresh:<5} {score(mask, spikes, series)}")


def zscore() -> None:
    for path in datasets(("l3",)):
        series, spikes, step = load(path)
        print(f"\n{path.stem}  step_scale={step:.4g}")
        for sig in (0.5, 1.0, 2.0, 3.0):
            for thresh in (3.0, 4.0, 6.0, 8.0, 10.0):
                mask = _run_saqc(
                    series, "flagZScore",
                    {"method": "modified", "window": "12h", "thresh": thresh,
                     "min_residuals": sig * step},
                ).to_numpy()
                print(f"  min_res={sig:<4} thresh={thresh:<5} "
                      f"{score(mask, spikes, series)}")


def combo() -> None:
    """Best candidate settings, union of both detectors, on all nine datasets."""
    configs = {
        "current": dict(un=1.5, un_n=20, zt=10.0, zs=3.0),
        "A": dict(un=1.2, un_n=10, zt=4.0, zs=2.0),
        "B": dict(un=1.1, un_n=10, zt=3.0, zs=2.0),
        "C": dict(un=1.05, un_n=10, zt=3.0, zs=1.0),
        "D": dict(un=1.1, un_n=10, zt=4.0, zs=2.0),
    }
    for path in datasets():
        series, spikes, step = load(path)
        print(f"\n{path.stem}  step_scale={step:.4g}  "
              f"{int(spikes.sum())} injected spike rows")
        for name, c in configs.items():
            u = _run_saqc(
                series, "flagUniLOF",
                {"n": c["un_n"], "thresh": c["un"], "density": "auto",
                 "slope_correct": True},
            ).to_numpy()
            z = _run_saqc(
                series, "flagZScore",
                {"method": "modified", "window": "12h", "thresh": c["zt"],
                 "min_residuals": c["zs"] * step},
            ).to_numpy()
            print(f"  {name:<8} {score(u | z, spikes, series)}")


if __name__ == "__main__":
    {"diagnose": diagnose, "unilof": unilof, "zscore": zscore, "combo": combo}[
        sys.argv[1] if len(sys.argv) > 1 else "diagnose"
    ]()
