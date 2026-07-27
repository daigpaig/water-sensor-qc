"""Pull the one real 'expert judged this bad' marker USGS turbidity carries: `e`.

The hunt for typed condition codes (Ice/Eqp/Mnt/Fld) came up empty for turbidity
(§9.2): across IV and daily-values on many gauges, turbidity qualifiers are only
approval status (A/P), estimated (`e`), censored (`>`), and grade codes like `[4]`
= partial-day aggregate. USGS handles bad turbidity by *deleting* it (→ gaps) or
*estimating* it before approval, not by tagging a fault type.

The nearest thing to an expert label is therefore `e` (estimated): a run of `e`
values is a stretch the hydrographer judged the raw sensor untrustworthy and
replaced with an estimate. 06934500 (Verdigris R, KS) carries one clean ~11-day
`e` run. This pulls a window around it and plots it, marking the estimated points,
so you can see where an expert intervened. Note: the approved value *at* the `e`
points is the smooth estimate, not the original raw fault (that was overwritten).

    PYTHONPATH=. .venv/bin/python scratchpad/pull_expert_flagged.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dataretrieval import nwis

from src.pull_usgs import TURBIDITY_PARAM, tidy_frame

SITE = "06934500"
OUTDIR = Path("data/raw/expert_flagged")
FIG = Path("figures/expert_estimated_06934500.html")


def main() -> int:
    df, _ = nwis.get_iv(sites=SITE, parameterCd=TURBIDITY_PARAM,
                        start="2022-01-01", end="2025-01-01")
    tidy = tidy_frame(df)
    q = tidy["qualifier"].astype(str)
    est = (q.str.endswith("e") | q.str.contains(r"\be\b", regex=True)).to_numpy()

    # Locate the estimated run and take a context window of +-25 days around it.
    est_idx = np.flatnonzero(est)
    lo, hi = est_idx.min(), est_idx.max()
    t0 = tidy["datetime"].iloc[lo] - pd.Timedelta(days=25)
    t1 = tidy["datetime"].iloc[hi] + pd.Timedelta(days=25)
    win = tidy[(tidy["datetime"] >= t0) & (tidy["datetime"] <= t1)].copy()
    win_est = est[(tidy["datetime"] >= t0) & (tidy["datetime"] <= t1)]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUTDIR / f"{SITE}_turbidity_{TURBIDITY_PARAM}.csv"
    tidy.to_csv(out_csv, index=False)
    print(f"{SITE}: {len(tidy):,} rows saved -> {out_csv}")
    print(f"   estimated run: {tidy['datetime'].iloc[lo]:%Y-%m-%d} -> "
          f"{tidy['datetime'].iloc[hi]:%Y-%m-%d}  ({int(est.sum())} points)")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=win["datetime"], y=win["value"].to_numpy().tolist(),
        mode="lines", line=dict(color="#2563eb", width=1), name="turbidity (approved)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU<extra></extra>",
    ))
    sub = win[win_est]
    fig.add_trace(go.Scattergl(
        x=sub["datetime"], y=sub["value"].to_numpy().tolist(),
        mode="markers", marker=dict(color="#dc2626", size=4), name="estimated (e) — expert-flagged",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (estimated)<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        title=f"{SITE} turbidity — the ~11-day stretch a hydrographer flagged 'estimated' (e)",
        xaxis_title="datetime", yaxis_title="FNU",
        height=500, hovermode="x", legend=dict(orientation="h", y=1.02, x=1, xanchor="right"),
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG, include_plotlyjs=True)
    print(f"   wrote {FIG}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
