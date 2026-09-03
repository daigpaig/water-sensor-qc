"""Stage 4: is a candidate's 5-min sampling step STABLE across the whole window?

CLAUDE.md §9.1 records the failure this guards against: "a series whose sampling
rate changes mid-record, where re-gridding to the modal step manufactures phantom
NaN". A gauge that logged 15-min in 2023 and 5-min in 2025 has a 5-min *modal*
step and would look fine to stage 3, but re-gridding it to 5 min invents two
missing rows out of every three for the early period — and §5 labels every
missing run as a gap, so those phantoms would become ground-truth anomalies.

This reports the monthly modal step and the share of steps that are exactly the
modal one, so a mid-record change is visible rather than averaged away.

Run:  .venv/bin/python -m scratchpad.check_cadence_stability --sites 01648010 ...
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

CACHE = Path("scratchpad/out/probe5")


def stability(site: str) -> dict | None:
    path = CACHE / f"{site}.csv"
    if not path.exists():
        print(f"  [{site}] not cached — run screen_5min_stage3 first")
        return None
    df = pd.read_csv(path, parse_dates=["datetime"])
    df = df[df["qualifier"].astype(str).str.startswith("A")]
    if df.empty:
        return None
    idx = pd.DatetimeIndex(df["datetime"]).sort_values()
    idx = idx[~idx.duplicated(keep="first")]
    steps = pd.Series(np.diff(idx.to_numpy()) / np.timedelta64(1, "m"),
                      index=idx[1:])
    modal = float(steps.mode().iloc[0])
    monthly = steps.groupby(pd.Grouper(freq="MS")).agg(
        modal_step=lambda s: float(s.mode().iloc[0]) if len(s) else np.nan,
        n=lambda s: len(s),
    )
    months_off = monthly[(monthly.modal_step != modal) & (monthly.n > 100)]
    return {
        "site_no": site,
        "modal_step_min": modal,
        "pct_steps_at_modal": round(100.0 * float((steps == modal).mean()), 1),
        "n_months": len(monthly),
        "n_months_off_modal": len(months_off),
        "months_off": ", ".join(
            f"{i:%Y-%m}={r.modal_step:.0f}m" for i, r in months_off.iterrows()
        ) or "-",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", nargs="+", required=True)
    args = ap.parse_args()
    rows = [r for s in args.sites if (r := stability(s))]
    out = pd.DataFrame(rows)
    with pd.option_context("display.width", 220):
        print(out.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
