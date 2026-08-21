"""Find precipitation data near a turbidity gauge (CLAUDE.md §9).

Rainfall is decision-relevant context: the hardest call the agent makes is
"storm peak or sensor artifact?" (§7.3), and a rain record either side of a
turbidity excursion is the one piece of evidence that settles it from outside
the series.

**No USGS 5-min turbidity gauge records precipitation at the same site** —
checked against the national series catalog, 0 of 57. So precipitation has to
come from a *nearby* station. This script searches the NWIS site service by
bounding box for instantaneous ``00045`` (precipitation, total, inches) series,
computes great-circle distance to the turbidity gauge, and reports cadence and
record overlap for the closest ones.

Run:  .venv/bin/python -m scratchpad.find_precip --sites 01648010 03197950
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PRECIP_PARAM = "00045"
EARTH_R_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2, lon2) -> np.ndarray:
    """Great-circle distance in km (vectorised over the second point)."""
    p1, p2 = np.radians(lat1), np.radians(np.asarray(lat2, dtype=float))
    dphi = p2 - p1
    dlam = np.radians(np.asarray(lon2, dtype=float) - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def precip_sites_near(lat: float, lon: float, half_deg: float = 0.5) -> pd.DataFrame:
    """Instantaneous-precip series within a lat/lon box around a point."""
    from dataretrieval import nwis

    bbox = f"{lon - half_deg:.4f},{lat - half_deg:.4f},{lon + half_deg:.4f},{lat + half_deg:.4f}"
    # The site service intermittently fails DNS resolution mid-sweep; one
    # transient error should not read as "this gauge has no rain data".
    df = None
    for attempt in range(1, 4):
        try:
            df, _ = nwis.get_info(bBox=bbox, parameterCd=PRECIP_PARAM,
                                  hasDataTypeCd="iv", seriesCatalogOutput=True)
            break
        except Exception as exc:  # noqa: BLE001
            print(f"  bbox query attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == 3:
                return pd.DataFrame()
            time.sleep(5.0)
    if df is None or df.empty:
        return pd.DataFrame()
    p = df[(df.parm_cd == PRECIP_PARAM) & (df.data_type_cd == "uv")].copy()
    if p.empty:
        return p
    p["dist_km"] = haversine_km(lat, lon, p.dec_lat_va, p.dec_long_va)
    p["begin"] = pd.to_datetime(p.begin_date, errors="coerce")
    p["end"] = pd.to_datetime(p.end_date, errors="coerce")
    return p.sort_values("dist_km")


def probe_cadence(site: str, start: str, end: str) -> tuple[float, int]:
    """Modal step (min) and observation count for a precip series."""
    from dataretrieval import nwis

    try:
        df, _ = nwis.get_iv(sites=site, parameterCd=PRECIP_PARAM,
                            start=start, end=end)
    except Exception:  # noqa: BLE001
        return (np.nan, 0)
    if df is None or df.empty:
        return (np.nan, 0)
    # A multi-site frame indexes on (site_no, datetime); drop the site level so
    # the remaining index is genuinely datetimes and not an object tuple index.
    idx = df.index
    if isinstance(idx, pd.MultiIndex):
        idx = idx.get_level_values("datetime")
    idx = pd.DatetimeIndex(pd.to_datetime(idx, utc=True, errors="coerce"))
    idx = idx.tz_convert("UTC").tz_localize(None).sort_values()
    if len(idx) < 10:
        return (np.nan, len(idx))
    steps = pd.Series(np.diff(idx.to_numpy()) / np.timedelta64(1, "m"))
    return (float(steps.mode().iloc[0]), len(idx))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", required=True,
                    help="USGS turbidity site numbers.")
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2025-07-01")
    ap.add_argument("--half-deg", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=5, help="Nearest N to probe.")
    ap.add_argument("--catalog", default="scratchpad/out/screen_5min_catalog.csv")
    args = ap.parse_args()

    cat = pd.read_csv(args.catalog, dtype={"site_no": str, "parm_cd": str})
    loc = cat.drop_duplicates("site_no").set_index("site_no")

    rows = []
    for site in args.sites:
        if site in loc.index:
            lat = float(loc.loc[site, "dec_lat_va"])
            lon = float(loc.loc[site, "dec_long_va"])
            name = loc.loc[site, "station_nm"]
        else:
            from dataretrieval import nwis
            info, _ = nwis.get_info(sites=site)
            lat = float(info.dec_lat_va.iloc[0])
            lon = float(info.dec_long_va.iloc[0])
            name = str(info.station_nm.iloc[0])
        print(f"\n=== {site}  {name}  ({lat:.4f}, {lon:.4f}) ===")
        near = precip_sites_near(lat, lon, args.half_deg)
        if near.empty:
            print("  no instantaneous precipitation series within the box")
            continue
        near = near.drop_duplicates("site_no").head(args.top)
        for r in near.itertuples():
            dt_min, n = probe_cadence(r.site_no, args.start,
                                      pd.Timestamp(args.start) + pd.Timedelta("3D"))
            covers = (pd.notna(r.begin) and pd.notna(r.end)
                      and r.begin <= pd.Timestamp(args.start)
                      and r.end >= pd.Timestamp(args.end))
            print(f"  {r.dist_km:6.1f} km  {r.site_no:<16} {str(r.station_nm)[:52]:<52} "
                  f"{str(r.begin_date)[:10]}..{str(r.end_date)[:10]}  "
                  f"dt={dt_min if np.isfinite(dt_min) else '?'}min  "
                  f"{'COVERS window' if covers else 'partial'}")
            rows.append({"turbidity_site": site, "precip_site": r.site_no,
                         "precip_name": r.station_nm, "dist_km": round(r.dist_km, 1),
                         "begin": r.begin_date, "end": r.end_date,
                         "dt_min": dt_min, "covers_window": covers})
    out = pd.DataFrame(rows)
    out.to_csv("scratchpad/out/precip_near.csv", index=False)
    print(f"\n-> scratchpad/out/precip_near.csv ({len(out)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
