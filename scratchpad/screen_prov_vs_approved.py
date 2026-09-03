"""Screen USGS turbidity gauges that expose BOTH an approved and a provisional span.

NWIS serves the *current* state of each record: once a period is approved, the
provisional values it used to carry are overwritten and are not retrievable. So
there is no way to get the same timestamps in two states. What a live record does
give you is a moving approval boundary: everything older than the boundary is
``A``, everything newer is ``P``, from the same sensor at the same site.

This script screens candidate sites for that shape:
  - a long approved span (>= --min-approved-days)
  - a usable provisional tail (>= --min-provisional-days)
  - consistent sub-hourly sampling, decent completeness

Run:  .venv/bin/python -m scratchpad.screen_prov_vs_approved
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from src.datasets.pull_usgs import TURBIDITY_PARAM, select_turbidity_column  # noqa: E402

# Candidate pool, chosen for geographic + turbidity-regime spread.
CANDIDATES: dict[str, str] = {
    "03447687": "French Broad R nr Fletcher, NC (S. Appalachia)",
    "02198840": "Savannah R at I-95 nr Port Wentworth, GA (tidal)",
    "08041770": "LNVA Canal at Beaumont, TX (managed canal)",
    "06818000": "Missouri R at St Joseph, MO (large plains river)",
    "01646500": "Potomac R nr Washington DC (mid-Atlantic)",
    "02336000": "Chattahoochee R nr Atlanta, GA (piedmont)",
    "12340500": "Blackfoot R nr Bonner, MT (clear mountain)",
    "11501000": "Sprague R nr Chiloquin, OR (high desert)",
    "05288705": "Mississippi R at Brooklyn Park, MN (upper Miss.)",
    "14211720": "Willamette R at Portland, OR (Pacific NW)",
    "09380000": "Colorado R at Lees Ferry, AZ (arid canyon)",
    "01463500": "Delaware R at Trenton, NJ (NE tidal-fresh)",
    "06934500": "Missouri R at Hermann, MO (lower Missouri)",
    "15515500": "Tanana R at Fairbanks, AK (glacial)",
    "05587450": "Mississippi R at Grafton, IL (mid Miss.)",
    "02035000": "James R at Cartersville, VA (piedmont)",
    "01578310": "Susquehanna R at Conowingo, MD (large NE)",
    "07022000": "Mississippi R at Thebes, IL (lower Miss.)",
    "13317000": "Salmon R at Whitebird, ID (mountain)",
    "10336610": "Upper Truckee R at South Lake Tahoe, CA (Sierra)",
    "02489500": "Pearl R nr Bogalusa, LA (Gulf coastal plain)",
    "04157005": "Saginaw R at Saginaw, MI (Great Lakes)",
    "08313000": "Rio Grande at Otowi Bridge, NM (arid SW)",
    "01594440": "Patuxent R nr Bowie, MD (Chesapeake)",
}


def _span_stats(idx: pd.DatetimeIndex, vals: pd.Series) -> dict[str, float]:
    """Median step (min), span (days), completeness (%), and value quantiles."""
    if len(idx) < 2:
        return {"days": 0.0, "dt_min": np.nan, "complete": np.nan,
                "median": np.nan, "p95": np.nan}
    steps = np.diff(idx.to_numpy()) / np.timedelta64(1, "m")
    dt = float(np.median(steps))
    span_min = (idx.max() - idx.min()).total_seconds() / 60.0
    return {
        "days": span_min / 1440.0,
        "dt_min": dt,
        "complete": 100.0 * len(idx) / (span_min / dt) if dt and span_min else np.nan,
        "median": float(vals.median()),
        "p95": float(vals.quantile(0.95)),
    }


def screen_site(site: str, start: str, end: str) -> dict | None:
    """Return the approved/provisional split for one site, or None if no data."""
    from dataretrieval import nwis

    try:
        df, _ = nwis.get_iv(sites=site, parameterCd=TURBIDITY_PARAM, start=start, end=end)
    except Exception as exc:  # noqa: BLE001 - screening, keep going
        return {"site": site, "error": f"{type(exc).__name__}: {exc}"}
    if df is None or df.empty:
        return {"site": site, "error": "no data"}

    val_col = select_turbidity_column(df, TURBIDITY_PARAM)
    idx = pd.DatetimeIndex(df.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    vals = pd.to_numeric(df[val_col], errors="coerce")
    q = df.get(f"{val_col}_cd", pd.Series("", index=df.index)).astype(str)

    is_appr = q.str.startswith("A").to_numpy()
    ok = ~vals.isna().to_numpy()
    a_idx, p_idx = idx[is_appr & ok], idx[~is_appr & ok]
    a_val = pd.Series(vals.to_numpy()[is_appr & ok])
    p_val = pd.Series(vals.to_numpy()[~is_appr & ok])

    return {
        "site": site,
        "error": None,
        "col": str(val_col),
        "codes": dict(q.value_counts().head(5)),
        "boundary": str(p_idx.min()) if len(p_idx) else "-",
        "approved": _span_stats(a_idx, a_val) | {"n": len(a_idx)},
        "provisional": _span_stats(p_idx, p_val) | {"n": len(p_idx)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2026-07-29")
    ap.add_argument("--min-approved-days", type=float, default=365.0)
    ap.add_argument("--min-provisional-days", type=float, default=60.0)
    ap.add_argument("--sites", nargs="*", default=None)
    args = ap.parse_args()

    sites = args.sites or list(CANDIDATES)
    rows = []
    for site in sites:
        name = CANDIDATES.get(site, "")
        r = screen_site(site, args.start, args.end)
        if r.get("error"):
            print(f"{site}  SKIP  {r['error']:<40}  {name}")
            continue
        a, p = r["approved"], r["provisional"]
        verdict = (
            "OK  " if a["days"] >= args.min_approved_days
            and p["days"] >= args.min_provisional_days else "no  "
        )
        print(
            f"{site} {verdict} A: {a['n']:>6,} rows {a['days']:>6.0f}d "
            f"{a['dt_min']:>4.0f}min {a['complete']:>5.1f}% med{a['median']:>7.1f} | "
            f"P: {p['n']:>6,} rows {p['days']:>6.0f}d {p['dt_min']:>4.0f}min "
            f"{p['complete']:>5.1f}% med{p['median']:>7.1f} | bnd {r['boundary'][:10]} | {name}"
        )
        rows.append((site, name, r, verdict.strip() == "OK"))

    print("\nPASS:", [s for s, _, _, ok in rows if ok])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
