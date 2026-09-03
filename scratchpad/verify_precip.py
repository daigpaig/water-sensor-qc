"""Verify that a candidate precipitation series really covers the project window.

The site-service catalog's ``begin_date``/``end_date`` describe the *site's*
series, and several nearby gauges advertise a long record but only started
reporting instantaneous values recently. So before claiming a turbidity gauge has
usable rain data, actually pull the window and measure it.

Run:  .venv/bin/python -m scratchpad.verify_precip
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

START, END = "2023-07-01", "2025-07-01"

# turbidity gauge -> nearby instantaneous-precip candidates (site, name, km)
PAIRS: dict[str, list[tuple[str, str, float]]] = {
    "02054550": [
        ("371520080015100", "MET STN Hidden Valley, Roanoke VA", 7.0),
        ("371824080002600", "MET STN Rt 117, Roanoke VA", 9.2),
        ("371518079591700", "MET STN Shrine Hill Park, Roanoke VA", 10.7),
    ],
    "01467200": [
        ("01473169", "Valley Creek at Valley Forge PA", 31.1),
        ("400145074555401", "Bridgeboro USGS weather station NJ", 19.9),
        ("01475548", "Cobbs Creek at US 13 Darby PA", 12.0),
    ],
    "040851385": [
        ("04085078", "Dutchman Creek at Ashwaubenon WI", 7.9),
        ("04072150", "Duck Creek near Howard WI", 9.5),
        ("04085108", "East River nr Greenleaf WI", 18.6),
    ],
}


def check(site: str) -> dict:
    from dataretrieval import nwis

    try:
        df, _ = nwis.get_iv(sites=site, parameterCd="00045", start=START, end=END)
    except Exception as exc:  # noqa: BLE001
        return {"error": f"{type(exc).__name__}: {exc}"}
    if df is None or df.empty:
        return {"error": "no data in window"}
    col = next((c for c in df.columns
                if str(c).startswith("00045") and not str(c).endswith("_cd")), None)
    if col is None:
        return {"error": f"no 00045 column ({list(df.columns)})"}
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.sort_values()
    vals = pd.to_numeric(df[col], errors="coerce")
    steps = pd.Series(np.diff(idx.to_numpy()) / np.timedelta64(1, "m"))
    span_days = (idx.max() - idx.min()).total_seconds() / 86400.0
    dt = float(steps.mode().iloc[0]) if len(steps) else np.nan
    expected = span_days * 1440.0 / dt if dt else np.nan
    q = df.get(f"{col}_cd", pd.Series("", index=df.index)).astype(str)
    return {
        "n": len(idx),
        "first": str(idx.min())[:10],
        "last": str(idx.max())[:10],
        "span_days": round(span_days, 0),
        "dt_min": dt,
        "complete_pct": round(100.0 * len(idx) / expected, 1) if expected else np.nan,
        "total_in": round(float(vals.sum()), 1),
        "approved_pct": round(100.0 * float(q.str.startswith("A").mean()), 0),
    }


def main() -> int:
    for turb, cands in PAIRS.items():
        print(f"\n=== precipitation near turbidity gauge {turb} ===")
        for site, name, km in cands:
            r = check(site)
            if "error" in r:
                print(f"  {km:5.1f} km  {site:<16} {name:<38} -> {r['error']}")
                continue
            print(f"  {km:5.1f} km  {site:<16} {name:<38} -> "
                  f"{r['n']:,} obs @ {r['dt_min']:.0f}min, {r['first']}..{r['last']} "
                  f"({r['span_days']:.0f}d, {r['complete_pct']}% complete), "
                  f"{r['total_in']:.0f} in total, {r['approved_pct']:.0f}% approved")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
