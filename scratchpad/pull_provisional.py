"""Pull a few PROVISIONAL turbidity slices to eyeball realistic sensor anomalies.

Approved USGS data has already had fouling/calibration-drift corrections applied
(TM 1-D3), so it is a poor place to see raw sensor faults. *Provisional* data
(qualifier ``P``, not yet through record processing) still carries the
uncorrected anomalies — fouling spikes, drift, dropouts. This grabs a recent
window (mostly provisional, since approval lags ~1 year) for a few varied gauges,
keeps the non-approved rows, and writes them to ``data/raw/provisional/`` (which
is gitignored via ``data/raw/*`` and is a sibling of ``data/raw/approved/``, the
only directory the injection pipeline globs — so provisional data can never be
mistaken for a clean base).

    python scratchpad/pull_provisional.py
    python -m src.tools.visualize data/raw/provisional/*.csv \\
        --out figures/provisional_overview.html --mark-spikes --no-open
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd

from dataretrieval import nwis

from src.pull_usgs import TURBIDITY_PARAM, tidy_frame
from src.tools.visualize import detect_spikes

# Varied geography + turbidity regime, each with visible provisional anomalies.
SITES: dict[str, str] = {
    "06818000": "Missouri R at St. Joseph, MO (Great Plains; big spike)",
    "01646500": "Potomac R nr Washington, DC (Mid-Atlantic; frequent spikes)",
    "02336000": "Chattahoochee R at Atlanta, GA (urban river)",
}
START, END = "2025-07-01", "2026-07-01"
OUTDIR = Path("data/raw/provisional")


def keep_provisional(tidy: pd.DataFrame) -> pd.DataFrame:
    """Keep only non-approved rows (qualifier not starting 'A'): P / blank / NaN."""
    q = tidy["qualifier"].astype("string")
    keep = ~q.str.startswith("A").fillna(False)
    return tidy.loc[keep.to_numpy()].reset_index(drop=True)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for site, desc in SITES.items():
        df, _ = nwis.get_iv(sites=site, parameterCd=TURBIDITY_PARAM, start=START, end=END)
        tidy = tidy_frame(df)
        prov = keep_provisional(tidy)
        if prov.empty:
            print(f"{site}: no provisional rows in {START}..{END} — skipped.")
            continue
        out = OUTDIR / f"{site}_turbidity_{TURBIDITY_PARAM}_provisional.csv"
        prov.to_csv(out, index=False)
        s = pd.Series(prov["value"].to_numpy(), index=pd.DatetimeIndex(prov["datetime"]))
        n_spikes = int(detect_spikes(s).sum())
        print(
            f"{site}  {desc}\n"
            f"   {len(prov):,} provisional rows | "
            f"{prov['datetime'].min():%Y-%m-%d} -> {prov['datetime'].max():%Y-%m-%d} | "
            f"median {prov['value'].median():.1f} | max {prov['value'].max():.1f} FNU | "
            f"~{n_spikes} spikes -> {out}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
