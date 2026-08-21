"""Stage 2: measure the ACTUAL sampling cadence of every candidate turbidity gauge.

The site-service catalog reports ``count_nu`` as *days* of record for unit-value
series, not sample count, so cadence cannot be inferred from it (stage 1 found
every series reporting a nonsense 1440-min step). The only way to know a gauge's
sampling step is to look at its timestamps.

Pulling 700 gauges one at a time is slow; the NWIS iv service accepts a *list*
of sites per request, so this probes a short shared window in batches and takes
the modal timestamp step per site. Cheap enough to cover the whole national
shortlist.

Run:  .venv/bin/python -m scratchpad.screen_5min_stage2
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BATCH = 30


def probe_batch(sites: list[str], start: str, end: str) -> dict[str, dict]:
    """Modal step (min), n obs, and approval mix for each site in one request."""
    from dataretrieval import nwis

    out: dict[str, dict] = {}
    try:
        df, _ = nwis.get_iv(sites=sites, parameterCd="63680", start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        print(f"  batch failed ({type(exc).__name__}: {exc}); falling back per-site")
        for s in sites:
            try:
                d, _ = nwis.get_iv(sites=s, parameterCd="63680", start=start, end=end)
            except Exception:  # noqa: BLE001
                continue
            if d is not None and not d.empty:
                out |= _summarise(d, s)
        return out
    if df is None or df.empty:
        return out
    if "site_no" in (df.index.names or []):
        for site, sub in df.groupby(level="site_no"):
            out |= _summarise(sub.droplevel("site_no"), str(site))
    else:
        out |= _summarise(df, sites[0])
    return out


def _summarise(d: pd.DataFrame, site: str) -> dict[str, dict]:
    """Modal timestamp step and approval share for one site's short window."""
    val_cols = [c for c in d.columns if str(c).startswith("63680")
                and not str(c).endswith("_cd")]
    if not val_cols:
        return {}
    col = "63680" if "63680" in val_cols else val_cols[0]
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.sort_values()
    if len(idx) < 10:
        return {site: {"dt_min": np.nan, "n": len(idx), "approved_pct": np.nan}}
    steps = pd.Series(np.diff(idx.to_numpy()) / np.timedelta64(1, "m"))
    q = d.get(f"{col}_cd", pd.Series("", index=d.index)).astype(str)
    approved = 100.0 * q.str.startswith("A").mean()
    return {site: {"dt_min": float(steps.mode().iloc[0]), "n": len(idx),
                   "approved_pct": float(approved)}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--catalog", type=Path,
                    default=Path("scratchpad/out/screen_5min_catalog.csv"))
    ap.add_argument("--start", default="2025-05-01")
    ap.add_argument("--end", default="2025-05-04")
    ap.add_argument("--min-record-days", type=float, default=730.0)
    ap.add_argument("--min-end", default="2025-06-01")
    ap.add_argument("--out", type=Path,
                    default=Path("scratchpad/out/screen_5min_cadence.csv"))
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog, dtype={"site_no": str, "parm_cd": str})
    turb = cat[(cat.data_type_cd == "uv") & (cat.parm_cd == "63680")].copy()
    turb["begin"] = pd.to_datetime(turb.begin_date, errors="coerce")
    turb["end"] = pd.to_datetime(turb.end_date, errors="coerce")
    turb["record_days"] = (turb.end - turb.begin).dt.total_seconds() / 86400.0
    cand = turb[(turb.record_days >= args.min_record_days)
                & (turb.end >= pd.Timestamp(args.min_end))]
    # One row per site (a site may expose several turbidity time series).
    meta = (cand.sort_values("record_days", ascending=False)
                .drop_duplicates("site_no")
                .set_index("site_no"))
    sites = list(meta.index)
    print(f"probing {len(sites)} sites for cadence over {args.start}..{args.end}")

    results: dict[str, dict] = {}
    for i in range(0, len(sites), BATCH):
        chunk = sites[i:i + BATCH]
        results |= probe_batch(chunk, args.start, args.end)
        print(f"  {min(i + BATCH, len(sites))}/{len(sites)} probed "
              f"({len(results)} with data)")

    rows = []
    for site, r in results.items():
        m = meta.loc[site]
        rows.append({
            "site_no": site, "station_nm": m.station_nm, "state": m.state,
            "lat": m.dec_lat_va, "lon": m.dec_long_va, "huc_cd": m.huc_cd,
            "site_tp_cd": m.site_tp_cd,
            "begin_date": m.begin_date, "end_date": m.end_date,
            "record_days": round(float(m.record_days), 0),
            **r,
        })
    out = pd.DataFrame(rows).sort_values("dt_min")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\n-> {args.out}")
    print("cadence (modal step, min) across probed sites:")
    print(out.dt_min.value_counts().sort_index().head(20).to_string())
    five = out[out.dt_min <= 5.0]
    print(f"\n{len(five)} sites sampling at <= 5 min:")
    with pd.option_context("display.width", 240, "display.max_rows", 300):
        print(five[["site_no", "station_nm", "state", "dt_min", "n",
                    "approved_pct", "record_days"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
