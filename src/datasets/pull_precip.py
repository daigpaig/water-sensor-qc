"""Pull instantaneous precipitation near a turbidity gauge -> data/precip/.

WHY THIS EXISTS. The hardest call the agent makes is "storm peak or sensor
artifact?" (§7.3), and for a class of points it cannot be settled from the
turbidity series at all — they look like errors to the eye too. Rainfall either
side of the excursion is the one piece of evidence that comes from outside the
series.

NO 5-MIN TURBIDITY GAUGE RECORDS ITS OWN RAIN. Checked against the national
series catalog: 0 of 57 (`scratchpad/find_precip.py`). So precipitation always
comes from a *nearby* station, and how near is a property of the evidence that
has to travel with it — see PRECIP_STATIONS below, and note that 31 km is far
enough that a summer convective cell can rain on one and not the other.

Output, per turbidity gauge:
    data/precip/<gauge>/<station>.csv       datetime,precip_in
    data/precip/<gauge>/manifest.json       stations, distances, coverage
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PRECIP_PARAM = "00045"          # precipitation, total, inches
DEFAULT_START, DEFAULT_END = "2023-07-01", "2025-07-01"
DEFAULT_ROOT = Path("data/precip")

# Verified against the real window by `scratchpad/verify_precip.py` — the site
# catalog's begin/end describes the SITE, and several nearby gauges advertise a
# long record while only reporting instantaneous values recently. Only stations
# that actually returned data for 2023-07..2025-07 are listed.
#
# 01467200 is the gauge this project reports on. Its nearest station with FULL
# coverage is 31.1 km away; the closer one starts 2024-10, so both are pulled and
# the query tool prefers the nearest station that has data at the timestamp.
PRECIP_STATIONS: dict[str, list[dict]] = {
    "01467200": [
        {"site": "400145074555401", "name": "Bridgeboro USGS weather station NJ",
         "km": 19.9, "note": "closest, but only covers 2024-10 onward"},
        {"site": "01473169", "name": "Valley Creek at Valley Forge PA",
         "km": 31.1, "note": "full window, 15-min, 100% approved"},
    ],
}


def fetch(site: str, start: str, end: str) -> pd.DataFrame | None:
    """One station's instantaneous precipitation as datetime,precip_in."""
    from dataretrieval import nwis

    frame, _ = nwis.get_iv(sites=site, parameterCd=PRECIP_PARAM, start=start, end=end)
    if frame is None or frame.empty:
        return None
    col = next((c for c in frame.columns
                if str(c).startswith(PRECIP_PARAM) and not str(c).endswith("_cd")), None)
    if col is None:
        return None
    index = pd.DatetimeIndex(frame.index)
    if index.tz is not None:                       # NWIS returns tz-aware; §7.1 wants naive
        index = index.tz_convert("UTC").tz_localize(None)
    # .to_numpy(), not the Series: NWIS names the index "datetime" too, and building a
    # frame from an index-aligned Series makes 'datetime' both an index level and a
    # column, which pandas refuses to sort on.
    out = pd.DataFrame({
        "datetime": index.to_numpy(),
        "precip_in": pd.to_numeric(frame[col], errors="coerce").to_numpy(),
    })
    return out.dropna().sort_values("datetime").reset_index(drop=True)


def pull(gauge: str, root: Path = DEFAULT_ROOT,
         start: str = DEFAULT_START, end: str = DEFAULT_END) -> dict:
    stations = PRECIP_STATIONS.get(gauge)
    if not stations:
        raise SystemExit(f"No verified precipitation stations for {gauge}. "
                         f"Known: {', '.join(PRECIP_STATIONS) or '(none)'}. "
                         "Run scratchpad/find_precip.py + verify_precip.py first.")
    out_dir = root / gauge
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    for station in stations:
        frame = fetch(station["site"], start, end)
        if frame is None or frame.empty:
            print(f"  {station['site']}: no data in window — skipped")
            continue
        path = out_dir / f"{station['site']}.csv"
        frame.to_csv(path, index=False)
        span = (frame["datetime"].max() - frame["datetime"].min()).total_seconds() / 86400
        steps = frame["datetime"].diff().dt.total_seconds().div(60)
        entries.append({
            **station,
            "file": path.name,
            "n_obs": int(len(frame)),
            "first": str(frame["datetime"].min()),
            "last": str(frame["datetime"].max()),
            "span_days": round(span, 1),
            "dt_min": float(steps.mode().iloc[0]) if len(steps.dropna()) else None,
            "total_in": round(float(frame["precip_in"].sum()), 1),
        })
        print(f"  {station['site']}: {len(frame):,} obs, {entries[-1]['first'][:10]}"
              f"..{entries[-1]['last'][:10]}, {entries[-1]['total_in']} in -> {path}")

    manifest = {"gauge": gauge, "window": [start, end], "param": PRECIP_PARAM,
                "stations": entries,
                "note": ("Nearest station with full coverage is 31.1 km away. A summer "
                         "convective cell can rain on one and not the other, so ABSENCE "
                         "of rain here is weak evidence; presence is strong.")}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("gauge", nargs="?", default="01467200")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    args = parser.parse_args(argv)
    print(f"precipitation near {args.gauge} ({args.start}..{args.end}):")
    pull(args.gauge, args.root, args.start, args.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
