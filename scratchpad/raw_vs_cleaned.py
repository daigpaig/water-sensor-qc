"""See expert cleaning: the same sensor's RAW (provisional) vs CLEANED (approved) data.

A literal same-timestamp before->after is NOT retrievable from USGS: they overwrite
provisional with approved, so once experts delete a spike the raw value is gone.
The honest next-best is to pull a window that straddles the approval boundary and
colour each row by status. On one sensor you then see:
  - the APPROVED (older) segment: cleaned — spikes removed/corrected, smooth;
  - the PROVISIONAL (recent) segment: raw — the spikes are still there.
USGS removes/corrects those provisional spikes during record processing before
approval. (This is raw-vs-processed on one gauge, not a same-point diff.)

06818000 (Missouri R, MO) is a good example: its approved history is clean, and its
2026 provisional data carries a 2000 FNU sensor spike — exactly the kind an expert
deletes.

    PYTHONPATH=. .venv/bin/python scratchpad/raw_vs_cleaned.py
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
from src.tools.visualize import detect_spikes, insert_gap_breaks

SITE = "06818000"
FIG = Path("figures/raw_vs_cleaned_06818000.html")


def main() -> int:
    df, _ = nwis.get_iv(sites=SITE, parameterCd=TURBIDITY_PARAM,
                        start="2024-07-01", end="2026-07-02")
    tidy = tidy_frame(df).set_index("datetime")
    q = tidy["qualifier"].astype(str)
    approved = q.str.startswith("A").to_numpy()

    val = tidy["value"]
    appr = val.where(approved)                 # approved rows only (NaN elsewhere)
    prov = val.where(~approved)                # provisional rows only
    n_a, n_p = int(approved.sum()), int((~approved).sum())
    print(f"{SITE}: {len(tidy):,} rows | approved={n_a:,} | provisional={n_p:,}")
    print(f"   approved range {appr.min():.0f}-{appr.max():.0f} FNU | "
          f"provisional range {prov.min():.0f}-{prov.max():.0f} FNU")

    fig = go.Figure()
    fig.add_trace(go.Scattergl(
        x=insert_gap_breaks(appr).index, y=insert_gap_breaks(appr).to_numpy().tolist(),
        mode="lines", line=dict(color="#2563eb", width=1), connectgaps=False,
        name="APPROVED (cleaned by experts)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (approved)<extra></extra>",
    ))
    fig.add_trace(go.Scattergl(
        x=insert_gap_breaks(prov).index, y=insert_gap_breaks(prov).to_numpy().tolist(),
        mode="lines", line=dict(color="#d97706", width=1), connectgaps=False,
        name="PROVISIONAL (raw, not yet cleaned)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (provisional)<extra></extra>",
    ))
    # Mark raw spikes in the provisional segment — the ones an expert would remove.
    prov_series = pd.Series(prov.to_numpy(), index=tidy.index)
    spike = detect_spikes(prov_series).to_numpy() & (~approved)
    sp = tidy[spike]
    fig.add_trace(go.Scattergl(
        x=sp.index, y=sp["value"].to_numpy().tolist(), mode="markers",
        marker=dict(color="#dc2626", size=7, symbol="circle-open", line=dict(width=1.5)),
        name="raw spikes (expert will delete/correct)",
        hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (raw spike)<extra></extra>",
    ))
    fig.update_layout(
        template="plotly_white",
        title=(f"{SITE} — same sensor, cleaned vs raw. Blue = APPROVED (experts "
               f"removed/corrected spikes). Amber = PROVISIONAL (raw spikes still present)."),
        xaxis_title="datetime", yaxis_title="FNU", height=560, hovermode="x",
        legend=dict(orientation="h", y=1.03, x=1, xanchor="right"),
    )
    FIG.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(FIG, include_plotlyjs=True)
    print(f"   wrote {FIG}  ({int(spike.sum())} raw spikes marked in the provisional segment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
