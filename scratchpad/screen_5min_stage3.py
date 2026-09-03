"""Stage 3: full §9 gauge-quality evaluation of the 5-min turbidity candidates.

Stage 2 established which gauges sample at 5 minutes. This one downloads the
actual two-year window for each and applies the CLAUDE.md §9 base requirements:

  * **100% USGS-approved** over the window (approved records are drift/fouling
    corrected per TM 1-D3, which is what makes them injectable clean bases),
  * **consistent 5-min step** and **>=95% complete**,
  * a **CALM baseline** — approval keeps *real* storm spikes, and an unlabelled
    base spike scores as a false positive against the injected labels (§9.1).

"Calm" is measured the same way for every candidate *and* for the three incumbent
15-min gauges, so old and new are directly comparable: ``flagUniLOF(n=20,
thresh=1.5)`` — the §7.2 default spike setting — over the re-gridded approved
series, reported as a share of rows.

Each site's tidy pull is cached under ``scratchpad/out/probe5/`` so re-running is
free and the final pull costs nothing.

Run:  .venv/bin/python -m scratchpad.screen_5min_stage3
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.datasets.pull_usgs import (  # noqa: E402
    TURBIDITY_PARAM,
    filter_approved,
    longest_unbroken_run_days,
    summarise_series,
    tidy_frame,
)

CACHE = Path("scratchpad/out/probe5")
INCUMBENTS = {
    "03447687": "French Broad R nr Fletcher, NC [INCUMBENT 15-min]",
    "02198840": "Savannah R at I-95 nr Port Wentworth, GA [INCUMBENT 15-min]",
    "08041770": "LNVA Canal at Beaumont, TX [INCUMBENT 15-min]",
}


def cached_pull(site: str, start: str, end: str) -> pd.DataFrame | None:
    """Tidy datetime/value/qualifier frame for one site, cached on disk."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{site}.csv"
    if path.exists():
        return pd.read_csv(path, parse_dates=["datetime"])
    from dataretrieval import nwis

    try:
        raw, _ = nwis.get_iv(sites=site, parameterCd=TURBIDITY_PARAM,
                             start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        print(f"  [{site}] {type(exc).__name__}: {exc}")
        return None
    if raw is None or raw.empty:
        print(f"  [{site}] no data in window")
        return None
    tidy = tidy_frame(raw, TURBIDITY_PARAM)
    tidy.to_csv(path, index=False)
    return tidy


def spike_share(series: pd.Series, step_min: float) -> tuple[int, float]:
    """Rows flagged by the §7.2 default spike detector, as count and % of rows.

    Run on the re-gridded series so the neighbourhood LOF sees a regular index.
    Comparable across gauges only because the settings are held fixed.
    """
    import saqc

    if series.notna().sum() < 100:
        return (0, float("nan"))
    qc = saqc.SaQC({"value": series})
    try:
        qc = qc.flagUniLOF("value", n=20, thresh=1.5)
    except Exception as exc:  # noqa: BLE001
        print(f"    flagUniLOF failed: {type(exc).__name__}: {exc}")
        return (-1, float("nan"))
    flags = qc.flags["value"]
    n = int((flags > 0).sum())
    return (n, 100.0 * n / int(series.notna().sum()))


def evaluate(site: str, name: str, start: str, end: str, *,
             skip_spikes: bool = False) -> dict | None:
    tidy = cached_pull(site, start, end)
    if tidy is None or tidy.empty:
        return None
    n_raw = len(tidy)
    appr = filter_approved(tidy)
    approved_pct = 100.0 * len(appr) / n_raw if n_raw else float("nan")
    if appr.empty:
        return {"site_no": site, "station_nm": name, "n_raw": n_raw,
                "approved_pct": 0.0, "note": "no approved rows in window"}

    idx = pd.DatetimeIndex(appr["datetime"])
    median_dt, span_days, completeness, nan_pct, longest_gap = summarise_series(
        idx, appr["value"]
    )
    ser = pd.Series(appr["value"].to_numpy(), index=idx)
    ser = ser[~ser.index.duplicated(keep="first")].sort_index()
    unbroken = longest_unbroken_run_days(ser, pd.Timedelta("3h"))

    # Re-grid onto the modal step so gaps are explicit (as inspect_data does).
    step = pd.Timedelta(minutes=median_dt if np.isfinite(median_dt) else 5)
    grid = pd.date_range(ser.index.min(), ser.index.max(), freq=step)
    gridded = ser.reindex(grid)

    n_spike, pct_spike = (0, float("nan")) if skip_spikes else spike_share(
        gridded, median_dt
    )
    v = appr["value"].dropna()
    return {
        "site_no": site, "station_nm": name,
        "approved_pct": round(approved_pct, 1),
        "n_approved": len(appr),
        "dt_min": round(median_dt, 1),
        "span_days": round(span_days, 0),
        "complete_pct": round(completeness, 1),
        "grid_complete_pct": round(100.0 * gridded.notna().mean(), 1),
        "nan_pct": round(nan_pct, 2),
        "longest_gap_hr": round(longest_gap, 1),
        "unbroken_days": round(unbroken, 0),
        "median_fnu": round(float(v.median()), 1),
        "p05_fnu": round(float(v.quantile(0.05)), 1),
        "p95_fnu": round(float(v.quantile(0.95)), 1),
        "max_fnu": round(float(v.max()), 1),
        "n_base_spikes": n_spike,
        "base_spike_pct": round(pct_spike, 3) if np.isfinite(pct_spike) else np.nan,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2025-07-01")
    ap.add_argument("--cadence", type=Path,
                    default=Path("scratchpad/out/screen_5min_cadence.csv"))
    ap.add_argument("--out", type=Path,
                    default=Path("scratchpad/out/screen_5min_quality.csv"))
    ap.add_argument("--max-dt", type=float, default=5.0)
    ap.add_argument("--sites", nargs="*", default=None)
    ap.add_argument("--skip-spikes", action="store_true")
    args = ap.parse_args()

    cad = pd.read_csv(args.cadence, dtype={"site_no": str})
    five = cad[cad.dt_min <= args.max_dt]
    targets = {r.site_no: r.station_nm for r in five.itertuples()}
    targets |= INCUMBENTS
    if args.sites:
        targets = {s: targets.get(s, "") for s in args.sites}

    rows = []
    for i, (site, name) in enumerate(targets.items(), 1):
        print(f"[{i}/{len(targets)}] {site} {name}")
        res = evaluate(site, name, args.start, args.end,
                       skip_spikes=args.skip_spikes)
        if res:
            rows.append(res)
            print("   " + " | ".join(
                f"{k}={v}" for k, v in res.items()
                if k in ("approved_pct", "dt_min", "grid_complete_pct",
                         "median_fnu", "base_spike_pct")))
    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
