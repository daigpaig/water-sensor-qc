"""Screen USGS gauges nationwide for **5-minute** continuous turbidity (CLAUDE.md §9).

Stage 1 (this script, cheap): the NWIS *site service* series catalog
(``seriesCatalogOutput=true``) lists, per site and parameter, the period of
record and the number of unit values recorded (``count_nu``). Dividing that
count by the span in days gives **samples per day** — 288 for a 5-min series,
96 for 15-min — so we can infer sampling frequency for every turbidity gauge in
the country without downloading a single observation.

The same catalog also carries every *other* parameter each site records, so the
precipitation check (param ``00045``, "Precipitation, total, inches") comes free
in the same request rather than needing a second national sweep.

Stage 2 (``screen_5min_stage2.py``) pulls actual data for the shortlist and
verifies approval status, completeness, and a calm baseline.

Run:  .venv/bin/python -m scratchpad.screen_5min_turbidity
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

TURBIDITY_PARAMS = ("63680", "63676", "63675", "00076")  # 63680 is the standard optical FNU code
PRECIP_PARAM = "00045"
# NWIS `data_type_cd` for instantaneous ("unit") values.
UV = "uv"

STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY","DC","PR",
]


def fetch_state(state: str) -> pd.DataFrame | None:
    """Series catalog for every site in ``state`` that records turbidity."""
    from dataretrieval import nwis

    try:
        df, _ = nwis.get_info(
            stateCd=state, parameterCd="63680", seriesCatalogOutput=True
        )
    except Exception as exc:  # noqa: BLE001 - screening; keep going
        print(f"  [{state}] {type(exc).__name__}: {exc}")
        return None
    if df is None or df.empty:
        return None
    df = df.copy()
    df["state"] = state
    return df


def samples_per_day(row: pd.Series) -> float:
    """Infer sampling cadence from the catalog's record count and span."""
    try:
        begin = pd.to_datetime(row["begin_date"])
        end = pd.to_datetime(row["end_date"])
    except Exception:  # noqa: BLE001
        return np.nan
    if pd.isna(begin) or pd.isna(end):
        return np.nan
    days = (end - begin).total_seconds() / 86400.0
    n = pd.to_numeric(row.get("count_nu"), errors="coerce")
    if not days or pd.isna(n) or n <= 0:
        return np.nan
    return float(n) / days


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=Path("scratchpad/out/screen_5min_catalog.csv"))
    ap.add_argument("--min-record-days", type=float, default=730.0,
                    help="Required turbidity period of record, in days.")
    ap.add_argument("--min-end", default="2025-06-01",
                    help="Turbidity record must extend at least to this date.")
    args = ap.parse_args()

    frames = []
    for st in STATES:
        df = fetch_state(st)
        n = 0 if df is None else len(df)
        print(f"{st}: {n} catalog rows")
        if df is not None:
            frames.append(df)
    if not frames:
        print("No catalog rows returned.")
        return 1
    cat = pd.concat(frames, ignore_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    cat.to_csv(args.out, index=False)
    print(f"\nFull catalog -> {args.out}  ({len(cat):,} rows)")

    # --- turbidity unit-value series -------------------------------------
    turb = cat[(cat["data_type_cd"] == UV) & (cat["parm_cd"] == "63680")].copy()
    turb["samples_per_day"] = turb.apply(samples_per_day, axis=1)
    turb["record_days"] = (
        pd.to_datetime(turb["end_date"], errors="coerce")
        - pd.to_datetime(turb["begin_date"], errors="coerce")
    ).dt.total_seconds() / 86400.0
    turb["inferred_dt_min"] = 1440.0 / turb["samples_per_day"]

    print(f"\n{len(turb):,} turbidity (63680) unit-value series nationwide.")
    print("Inferred cadence distribution (min):")
    print(turb["inferred_dt_min"].round(0).value_counts().head(15).to_string())

    # 5-min series: >=200 samples/day (allows for gaps in a true 288/day series)
    five = turb[
        (turb["samples_per_day"] >= 200)
        & (turb["record_days"] >= args.min_record_days)
        & (pd.to_datetime(turb["end_date"], errors="coerce")
           >= pd.Timestamp(args.min_end))
    ].copy()

    # --- precipitation at the same sites ---------------------------------
    precip_sites = set(
        cat.loc[(cat["parm_cd"] == PRECIP_PARAM) & (cat["data_type_cd"] == UV),
                "site_no"].astype(str)
    )
    five["has_precip_iv"] = five["site_no"].astype(str).isin(precip_sites)

    five = five.sort_values("samples_per_day", ascending=False)
    cols = ["site_no", "station_nm", "state", "dec_lat_va", "dec_long_va",
            "begin_date", "end_date", "count_nu", "samples_per_day",
            "inferred_dt_min", "record_days", "has_precip_iv"]
    short = five[cols]
    short_path = args.out.with_name("screen_5min_shortlist.csv")
    short.to_csv(short_path, index=False)
    print(f"\n{len(short)} candidate 5-min turbidity gauges "
          f"(>=200 samples/day, >={args.min_record_days:.0f} d record, "
          f"active through {args.min_end}) -> {short_path}")
    with pd.option_context("display.width", 220, "display.max_rows", 200):
        print(short.to_string(index=False))
    print(f"\nwith 5-min precip (00045 iv) at the SAME site: "
          f"{int(short['has_precip_iv'].sum())}/{len(short)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
