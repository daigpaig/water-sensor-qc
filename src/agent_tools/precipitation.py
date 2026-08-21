"""Was it raining? — outside evidence for the storm-vs-artifact call.

WHY THIS IS A SEPARATE MODULE. Every other tool in `agent_tools` reasons about the
turbidity series itself. For a class of points that is not enough: they look like
errors to the eye as well as to the detectors, and only something from OUTSIDE the
series settles them. Rain is that something — turbidity rises because rain washes
sediment in, so a real excursion usually has rain behind it and an artifact does not.

TWO LIMITS THAT TRAVEL WITH EVERY ANSWER, because they change how much the answer is
worth:

1. NO 5-MIN TURBIDITY GAUGE RECORDS ITS OWN RAIN (0 of 57 in the national catalog),
   so the nearest station with full coverage of 01467200's window is **31 km away**.
   A summer convective cell is often 5-15 km across and can rain on one and not the
   other. So PRESENCE of rain is strong evidence; ABSENCE is weak. Every result says
   which station answered and how far away it was.
2. RAIN LEADS TURBIDITY. Sediment has to be washed in and travel downstream, and the
   lag depends on the catchment. Rather than assume a number, this reports several
   windows before the point (0-1h, 1-3h, 3-12h) and lets the agent judge which lag
   the record supports.

Data comes from `src.datasets.pull_precip` (data/precip/<gauge>/). If it has not been
pulled, every call says so plainly rather than silently reporting "no rain" — which
would be the most dangerous possible failure here, since it reads as evidence FOR
deleting the point.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_ROOT = Path("data/precip")

# Lag windows before the point, as (label, from, to) in hours back.
LAG_WINDOWS: tuple[tuple[str, float, float], ...] = (
    ("0-1h", 0.0, 1.0),
    ("1-3h", 1.0, 3.0),
    ("3-12h", 3.0, 12.0),
)
# Inches over a window that count as "it rained". A tipping-bucket gauge reports in
# 0.01 in increments, so anything at or below one tip is indistinguishable from noise.
WET_INCHES = 0.02


@lru_cache(maxsize=8)
def _load(gauge: str, root_str: str) -> tuple:
    """(list of (site, km, series), manifest) for one gauge, cached across calls."""
    root = Path(root_str) / gauge
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return (), None
    manifest = json.loads(manifest_path.read_text())
    loaded = []
    for entry in manifest.get("stations", []):
        path = root / entry["file"]
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")
        loaded.append((entry["site"], float(entry["km"]), entry.get("name", ""),
                       frame.set_index("datetime")["precip_in"]))
    loaded.sort(key=lambda item: item[1])          # nearest first
    return tuple(loaded), manifest


def _missing(gauge: str, root: Path) -> dict:
    return {
        "tool": "precip_context", "gauge": gauge, "available": False,
        "message": (
            f"NO PRECIPITATION DATA for gauge {gauge} (looked in {root / gauge}). "
            "This is NOT the same as 'it did not rain' — nothing is known either way, "
            "so this evidence is simply unavailable and the decision must rest on the "
            "series alone. Pull it with: python -m src.datasets.pull_precip " + gauge
        ),
    }


def precip_context(source, at, gauge: str = "01467200",
                   root: Path | str = DEFAULT_ROOT) -> dict:
    """Rain before and after one timestamp, from the nearest station that has data.

    *source* is unused — the signature matches the other context tools so the agent
    dispatch can pass the series uniformly.
    """
    root = Path(root)
    stations, manifest = _load(gauge, str(root))
    if not stations:
        return _missing(gauge, root)

    ts = pd.Timestamp(at)
    for site, km, name, series in stations:          # nearest first
        window = series.loc[ts - pd.Timedelta("12h"): ts + pd.Timedelta("3h")]
        if window.empty:
            continue                                  # this station has no data here

        lags = {}
        for label, lo, hi in LAG_WINDOWS:
            seg = series.loc[ts - pd.Timedelta(hours=hi): ts - pd.Timedelta(hours=lo)]
            lags[label] = round(float(seg.sum()), 3) if len(seg) else None
        after = series.loc[ts: ts + pd.Timedelta("3h")]
        total_before = sum(v for v in lags.values() if v)

        wet = total_before >= WET_INCHES
        if wet:
            verdict = (
                f"IT RAINED before this point ({round(total_before, 2)} in over the 12h "
                f"leading up to it). Rain washes sediment in, so an excursion here has a "
                f"physical cause and is more likely REAL WATER than an artifact."
            )
        else:
            verdict = (
                f"No rain recorded before this point (<{WET_INCHES} in over 12h). Treat "
                f"this as WEAK evidence: the nearest station is {km:.0f} km away and a "
                f"summer storm cell can miss it entirely. It does not license a deletion "
                f"on its own — it only fails to support the excursion."
            )
        return {
            "tool": "precip_context", "available": True, "gauge": gauge,
            "at": str(ts), "station": site, "station_name": name, "distance_km": km,
            "rain_before_in": lags,
            "rain_after_3h_in": round(float(after.sum()), 3) if len(after) else None,
            "total_before_12h_in": round(total_before, 3),
            "rained": bool(wet),
            "message": (f"{site} ({name}, {km:.0f} km): before = {lags}, "
                        f"after 3h = {round(float(after.sum()), 3) if len(after) else None} in. "
                        + verdict),
        }

    return {
        "tool": "precip_context", "available": False, "gauge": gauge, "at": str(ts),
        "message": (f"No precipitation station covers {ts}. The closest station's record "
                    "starts later or ends earlier. Nothing is known either way — this is "
                    "not evidence of dry weather."),
    }


def precip_context_points(source, ats, gauge: str = "01467200",
                          root: Path | str = DEFAULT_ROOT,
                          max_points: int = 300) -> dict:
    """The batch form — every point about to be called a spike, in one call.

    Mirrors describe_points: one compact row per timestamp plus a tally, so auditing a
    whole detector's output against rainfall costs one call rather than hundreds.
    """
    requested = [str(a) for a in ats]
    selected = requested[:max_points]
    rows, tally = [], {}
    for a in selected:
        r = precip_context(source, a, gauge=gauge, root=root)
        if not r.get("available"):
            key = "unknown"
            rows.append({"at": a, "rained": None, "total_before_12h_in": None})
        else:
            key = "rained" if r["rained"] else "dry"
            rows.append({"at": a, "rained": r["rained"],
                         "total_before_12h_in": r["total_before_12h_in"],
                         "rain_before_in": r["rain_before_in"]})
        tally[key] = tally.get(key, 0) + 1

    n_trunc = len(requested) - len(selected)
    msg = f"Checked {len(rows)} point(s) against rainfall: {tally}."
    if tally.get("rained"):
        msg += (f" {tally['rained']} of them had rain in the 12h before — those excursions "
                "have a physical cause and deleting them needs a specific reason.")
    if tally.get("unknown"):
        msg += f" {tally['unknown']} could not be checked (no station covers them)."
    if n_trunc:
        msg += f" {n_trunc} timestamp(s) were NOT checked (max_points={max_points})."
    return {"tool": "precip_context_points", "gauge": gauge, "n_checked": len(rows),
            "n_truncated": n_trunc, "tally": tally, "points": rows, "message": msg}
