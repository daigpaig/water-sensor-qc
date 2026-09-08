"""Screen USGS gauges nationwide for **5-minute** specific conductance (CLAUDE.md §9.6).

This is the §9 turbidity screen (``screen_5min_turbidity.py`` + ``_stage2`` +
``_stage3``) repeated for parameter ``00095``, collapsed into one script because
the three stages differ only in what they filter on. It picks the gauges recorded
in ``pull_usgs.VARIABLES``: **3 approved bases** and **5 provisional** ones.

The three stages, and why each exists
-------------------------------------
1. **Catalog sweep** (cheap). The NWIS site service with
   ``seriesCatalogOutput=true`` lists, per site and parameter, the period of
   record. One request per state covers the country.

   **Cadence CANNOT be read from this catalog.** ``count_nu`` is *days* of
   record for a unit-value series, not a sample count, so every series computes
   to a nonsense 1440-min step (the turbidity screen found exactly this). The
   catalog can only narrow the field by record length and recency.

   Unlike the turbidity sweep this does **not** cache the raw national catalog:
   that file was 218 MB and is pure intermediate. Only the filtered shortlist is
   written.

2. **Cadence probe** (measured). Pull a short shared window for each candidate
   and take the modal timestamp step. The iv service accepts a list of sites per
   request, so this batches rather than pulling hundreds of series one at a time.
   Only sites whose *measured* modal step is 5 min survive.

3. **Quality measurement** (per finalist). Pull both windows and measure what
   actually decides the pick: approval share, completeness against the modal
   step, and the value scale. Two windows because the two halves of
   ``data/specific_conductance/`` answer different questions — an approved base
   needs a clean two-year span, a provisional series needs a recent tail that is
   genuinely *un*approved.

Diversity is chosen from the stage-3 table, not measured by it. Specific
conductance spans three orders of magnitude in the field — a granite headwater
runs ~20 uS/cm, Midwestern ag drainage ~800, arid-basin and road-salt-affected
rivers a few thousand, and a tidal reach tens of thousands — so "scale" here is a
far wider axis than turbidity's, and the point of the pick is to span it rather
than to find the calmest gauges. Region is taken from the HUC and the state so
the set is not three gauges in one basin.

Run
---
    .venv/bin/python -m scratchpad.screen_5min_conductance              # all stages
    .venv/bin/python -m scratchpad.screen_5min_conductance --stage 3    # just re-measure

Outputs (all small; the big intermediates are deliberately not cached)
    scratchpad/out/screen_sc_shortlist.csv   stage 1: record-length survivors
    scratchpad/out/screen_sc_cadence.csv     stage 2: MEASURED modal step per site
    scratchpad/out/screen_sc_quality.csv     stage 3: approval/completeness/scale
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.pull_usgs import (
    CONDUCTANCE_PARAM,
    CONDUCTANCE_PROVISIONAL_END,
    CONDUCTANCE_PROVISIONAL_START,
    DEFAULT_END,
    DEFAULT_START,
    filter_by_approval,
    longest_unbroken_run_days,
    summarise_series,
    tidy_frame,
)

warnings.filterwarnings("ignore")

UV = "uv"  # NWIS `data_type_cd` for instantaneous ("unit") values
BATCH = 30  # sites per iv request in the cadence probe

STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID",
    "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS",
    "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK",
    "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV",
    "WI", "WY", "DC", "PR",
]

OUT = Path("scratchpad/out")
SHORTLIST = OUT / "screen_sc_shortlist.csv"
CADENCE = OUT / "screen_sc_cadence.csv"
QUALITY = OUT / "screen_sc_quality.csv"

#: A measured modal step at or below this counts as "5-minute". Some gauges
#: report on a 5-min clock but drop the occasional sample, which nudges the
#: modal step only if the dropouts are dense; the tolerance is for float noise
#: in the diff, not for admitting 15-min series.
FIVE_MIN_MAX = 5.5


# ---------------------------------------------------------------------------
# Stage 1 — catalog sweep
# ---------------------------------------------------------------------------
def fetch_state(state: str) -> pd.DataFrame | None:
    """Series catalog for every site in ``state`` that records conductance."""
    from dataretrieval import nwis

    try:
        df, _ = nwis.get_info(
            stateCd=state, parameterCd=CONDUCTANCE_PARAM, seriesCatalogOutput=True
        )
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        print(f"  [{state}] {type(exc).__name__}: {exc}")
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df["state"] = state
    return df


def stage1(min_record_days: float, min_end: str) -> pd.DataFrame:
    """Sweep every state's catalog; keep long, still-active conductance series."""
    frames = []
    for st in STATES:
        df = fetch_state(st)
        n = 0 if df is None else len(df)
        print(f"{st}: {n} catalog rows")
        if df is not None:
            # Trim to the columns we use before concatenating — the raw national
            # catalog is hundreds of MB and none of the rest is ever read.
            keep = [c for c in ("site_no", "station_nm", "state", "dec_lat_va",
                                "dec_long_va", "huc_cd", "site_tp_cd", "parm_cd",
                                "data_type_cd", "begin_date", "end_date", "count_nu")
                    if c in df.columns]
            frames.append(df[keep])
    if not frames:
        raise SystemExit("No catalog rows returned — is the network up?")

    cat = pd.concat(frames, ignore_index=True)
    sc = cat[(cat["data_type_cd"] == UV)
             & (cat["parm_cd"].astype(str) == CONDUCTANCE_PARAM)].copy()
    sc["begin"] = pd.to_datetime(sc["begin_date"], errors="coerce")
    sc["end"] = pd.to_datetime(sc["end_date"], errors="coerce")
    sc["record_days"] = (sc["end"] - sc["begin"]).dt.total_seconds() / 86400.0

    print(f"\n{len(sc):,} specific-conductance ({CONDUCTANCE_PARAM}) "
          f"unit-value series nationwide.")

    short = sc[(sc["record_days"] >= min_record_days)
               & (sc["end"] >= pd.Timestamp(min_end))]
    # A site can expose several conductance series (multiple sondes); one row each.
    short = (short.sort_values("record_days", ascending=False)
                  .drop_duplicates("site_no"))
    OUT.mkdir(parents=True, exist_ok=True)
    short.to_csv(SHORTLIST, index=False)
    print(f"{len(short)} sites with >={min_record_days:.0f} d of record still "
          f"active through {min_end} -> {SHORTLIST}")
    return short


# ---------------------------------------------------------------------------
# Stage 2 — measured cadence
# ---------------------------------------------------------------------------
def _summarise_probe(d: pd.DataFrame, site: str) -> dict[str, dict]:
    """Modal timestamp step and approval share for one site's short window."""
    val_cols = [c for c in d.columns
                if str(c).startswith(CONDUCTANCE_PARAM) and not str(c).endswith("_cd")]
    if not val_cols:
        return {}
    col = CONDUCTANCE_PARAM if CONDUCTANCE_PARAM in val_cols else val_cols[0]
    idx = pd.DatetimeIndex(d.index)
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    idx = idx.sort_values()
    if len(idx) < 10:
        return {site: {"dt_min": np.nan, "n_probe": len(idx), "approved_pct": np.nan}}
    steps = pd.Series(np.diff(idx.to_numpy()) / np.timedelta64(1, "m"))
    q = d.get(f"{col}_cd", pd.Series("", index=d.index)).astype(str)
    return {site: {
        "dt_min": float(steps.mode().iloc[0]),
        "n_probe": len(idx),
        "approved_pct": float(100.0 * q.str.startswith("A").mean()),
    }}


def probe_batch(sites: list[str], start: str, end: str) -> dict[str, dict]:
    """Modal step + approval mix for each site, one request for the whole batch."""
    from dataretrieval import nwis

    out: dict[str, dict] = {}
    try:
        df, _ = nwis.get_iv(
            sites=sites, parameterCd=CONDUCTANCE_PARAM, start=start, end=end
        )
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        print(f"  batch failed ({type(exc).__name__}); falling back per-site")
        for s in sites:
            try:
                d, _ = nwis.get_iv(
                    sites=s, parameterCd=CONDUCTANCE_PARAM, start=start, end=end
                )
            except Exception:  # noqa: BLE001
                continue
            if d is not None and not d.empty:
                out |= _summarise_probe(d, s)
        return out
    if df is None or df.empty:
        return out
    if "site_no" in (df.index.names or []):
        for site, sub in df.groupby(level="site_no"):
            out |= _summarise_probe(sub.droplevel("site_no"), str(site))
    else:
        out |= _summarise_probe(df, sites[0])
    return out


def stage2(short: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Measure each candidate's ACTUAL modal step; keep the 5-minute ones."""
    meta = short.set_index(short["site_no"].astype(str))
    sites = list(meta.index)
    print(f"\nprobing {len(sites)} sites for cadence over {start}..{end}")

    results: dict[str, dict] = {}
    for i in range(0, len(sites), BATCH):
        chunk = sites[i:i + BATCH]
        results |= probe_batch(chunk, start, end)
        print(f"  {min(i + BATCH, len(sites))}/{len(sites)} probed "
              f"({len(results)} with data)")

    rows = []
    for site, r in results.items():
        m = meta.loc[site]
        rows.append({
            "site_no": site,
            "station_nm": m.get("station_nm", ""),
            "state": m.get("state", ""),
            "lat": m.get("dec_lat_va", np.nan),
            "lon": m.get("dec_long_va", np.nan),
            "huc_cd": m.get("huc_cd", ""),
            "site_tp_cd": m.get("site_tp_cd", ""),
            "begin_date": m.get("begin_date", ""),
            "end_date": m.get("end_date", ""),
            "record_days": round(float(m.get("record_days", np.nan)), 0),
            **r,
        })
    out = pd.DataFrame(rows).sort_values("dt_min")
    out.to_csv(CADENCE, index=False)

    print("\nMeasured cadence distribution (modal step, minutes):")
    print(out["dt_min"].round(0).value_counts().sort_index().head(12).to_string())
    five = out[out["dt_min"] <= FIVE_MIN_MAX]
    print(f"\n{len(five)} of {len(out)} probed sites sample at <=5 min -> {CADENCE}")
    return five


# ---------------------------------------------------------------------------
# Stage 3 — quality over the two real windows
# ---------------------------------------------------------------------------
def measure_window(site: str, start: str, end: str) -> dict:
    """Approval share, completeness and value scale for one site and window."""
    from dataretrieval import nwis

    try:
        raw, _ = nwis.get_iv(
            sites=site, parameterCd=CONDUCTANCE_PARAM, start=start, end=end
        )
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        return {"error": f"{type(exc).__name__}"}
    if raw is None or raw.empty:
        return {"error": "empty"}

    tidy = tidy_frame(raw, CONDUCTANCE_PARAM)
    n_raw = len(tidy)
    approved = filter_by_approval(tidy, "approved")
    provisional = filter_by_approval(tidy, "provisional")

    idx = pd.DatetimeIndex(tidy["datetime"])
    median_dt, span_days, completeness, nan_pct, longest_gap = summarise_series(
        idx, tidy["value"]
    )
    ser = pd.Series(tidy["value"].to_numpy(), index=idx)
    return {
        "n_raw": n_raw,
        "approved_pct": round(100.0 * len(approved) / n_raw, 1) if n_raw else np.nan,
        "provisional_pct": round(100.0 * len(provisional) / n_raw, 1) if n_raw else np.nan,
        "median_dt_min": round(median_dt, 1),
        "span_days": round(span_days, 0),
        "completeness_pct": round(completeness, 1),
        "nan_pct": round(nan_pct, 2),
        "longest_gap_hr": round(longest_gap, 1),
        "unbroken_days": round(longest_unbroken_run_days(ser, pd.Timedelta("3h")), 0),
        "value_median": round(float(tidy["value"].median()), 1),
        "value_p05": round(float(tidy["value"].quantile(0.05)), 1),
        "value_p95": round(float(tidy["value"].quantile(0.95)), 1),
    }


def stage3(five: pd.DataFrame) -> pd.DataFrame:
    """Measure every 5-min finalist over BOTH the approved and provisional windows."""
    rows = []
    sites = list(five["site_no"].astype(str))
    for i, site in enumerate(sites, 1):
        meta = five[five["site_no"].astype(str) == site].iloc[0]
        print(f"[{i}/{len(sites)}] {site} {meta.get('station_nm', '')}")
        appr = measure_window(site, DEFAULT_START, DEFAULT_END)
        prov = measure_window(
            site, CONDUCTANCE_PROVISIONAL_START, CONDUCTANCE_PROVISIONAL_END
        )
        row = {
            "site_no": site,
            "station_nm": meta.get("station_nm", ""),
            "state": meta.get("state", ""),
            "lat": meta.get("lat", np.nan),
            "lon": meta.get("lon", np.nan),
            # HUC-2 is the top-level water-resource region; it is the coarse
            # "different part of the country" axis the diversity pick needs.
            "huc2": str(meta.get("huc_cd", ""))[:2],
            "dt_min": meta.get("dt_min", np.nan),
        }
        row |= {f"appr_{k}": v for k, v in appr.items()}
        row |= {f"prov_{k}": v for k, v in prov.items()}
        rows.append(row)
        print(f"      approved window: {appr}")
        print(f"   provisional window: {prov}")

    out = pd.DataFrame(rows)
    out.to_csv(QUALITY, index=False)
    print(f"\n{len(out)} finalists measured -> {QUALITY}")
    return out


def report(q: pd.DataFrame) -> None:
    """Print the two candidate tables a human picks the final gauges from."""
    if q.empty:
        print("\nNo finalists to report.")
        return

    # An approved BASE wants a clean, near-fully-approved two-year span.
    base = q[(q.get("appr_approved_pct", 0) >= 95)
             & (q.get("appr_completeness_pct", 0) >= 90)]
    print(f"\n=== approved-base candidates ({len(base)}) "
          f"— >=95% approved, >=90% complete over {DEFAULT_START}..{DEFAULT_END} ===")
    cols = ["site_no", "station_nm", "state", "huc2", "appr_approved_pct",
            "appr_completeness_pct", "appr_value_median", "appr_value_p05",
            "appr_value_p95"]
    print(base[[c for c in cols if c in base.columns]]
          .sort_values("appr_value_median").to_string(index=False))

    # A PROVISIONAL series wants the opposite: a recent window that USGS has
    # genuinely not signed off yet.
    prov = q[q.get("prov_provisional_pct", 0) >= 50]
    print(f"\n=== provisional candidates ({len(prov)}) — >=50% unapproved over "
          f"{CONDUCTANCE_PROVISIONAL_START}..{CONDUCTANCE_PROVISIONAL_END} ===")
    cols = ["site_no", "station_nm", "state", "huc2", "prov_provisional_pct",
            "prov_completeness_pct", "prov_span_days", "prov_value_median",
            "prov_value_p05", "prov_value_p95"]
    print(prov[[c for c in cols if c in prov.columns]]
          .sort_values("prov_value_median").to_string(index=False))

    print("\nPick for SPREAD, not for rank: different huc2 (region) and a value "
          "median spanning as much of the uS/cm range as the pool allows, then "
          "record the choice in pull_usgs.VARIABLES.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", type=int, default=0,
                    help="Run only this stage (1/2/3); 0 (default) runs all, "
                         "reusing any stage output already on disk.")
    ap.add_argument("--min-record-days", type=float, default=730.0,
                    help="Required conductance period of record, in days.")
    ap.add_argument("--min-end", default="2026-06-01",
                    help="Record must extend at least to this date.")
    ap.add_argument("--probe-start", default="2026-05-01",
                    help="Short window used to MEASURE cadence.")
    ap.add_argument("--probe-end", default="2026-05-04")
    args = ap.parse_args()

    short = None
    if args.stage in (0, 1) or not SHORTLIST.exists():
        short = stage1(args.min_record_days, args.min_end)
    if args.stage == 1:
        return 0
    if short is None:
        short = pd.read_csv(SHORTLIST, dtype={"site_no": str})

    five = None
    if args.stage in (0, 2) or not CADENCE.exists():
        five = stage2(short, args.probe_start, args.probe_end)
    if args.stage == 2:
        return 0
    if five is None:
        cad = pd.read_csv(CADENCE, dtype={"site_no": str})
        five = cad[cad["dt_min"] <= FIVE_MIN_MAX]

    q = stage3(five) if args.stage in (0, 3) or not QUALITY.exists() \
        else pd.read_csv(QUALITY, dtype={"site_no": str})
    report(q)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
