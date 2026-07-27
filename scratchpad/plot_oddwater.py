"""Plot the oddwater EXPERT-LABELLED river turbidity data (a real before/after).

Source: the `oddwater` R package (Talagala, Hyndman, Leigh, Mengersen &
Smith-Miles, Water Resources Research 55(11), 2019; github.com/pridiltal/oddwater).
Turbidity from in-situ sensors on the Pioneer River and Sandy Creek, Queensland
(data supplied by the Queensland Dept. of Environment and Science), with a
per-point expert anomaly label (`label_Tur`) and anomaly TYPE (`type_Tur`).

This is the closest public thing to "experts labelled this a spike, so it would be
cleaned": each point carries the raw value AND the expert's verdict/type, so the
highlighted points are exactly what a hydrographer would remove or correct.

    PYTHONPATH=. .venv/bin/python scratchpad/plot_oddwater.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

REF = Path("data/reference/oddwater")
FIGDIR = Path("figures")

# Official type legend (man/data_*.Rd), grouped to this project's vocabulary.
CODE_NAME = {
    "A": "spike — sudden large",
    "B": "plateau — persistent/low-variability",
    "C": "level_shift — constant offset",
    "D": "level_shift — sudden shift",
    "E": "high variability",
    "F": "impossible value",
    "G": "out-of-sensor-range",
    "H": "drift",
    "I": "spike — cluster",
    "J": "spike — sudden small",
    "K": "gap — missing",
    "L": "other untrustworthy",
}
CODE_COLOR = {
    "A": "#dc2626", "J": "#f97316", "I": "#b91c1c",     # spikes: reds/orange
    "B": "#d97706",                                       # plateau: amber
    "C": "#7c3aed", "D": "#6d28d9",                       # level_shift: purples
    "K": "#0891b2",                                       # gap: cyan
    "H": "#059669",                                       # drift: green
    "E": "#64748b", "F": "#334155", "G": "#94a3b8", "L": "#e11d48",  # other
}

SITES = {"data_pioneer_anom": "Pioneer River", "data_sandy_anom": "Sandy Creek"}


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    for stem, nice in SITES.items():
        df = pd.read_csv(REF / f"{stem}.csv")
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], dayfirst=True)
        df["type_Tur"] = df["type_Tur"].astype(str)
        n_anom = int((df["label_Tur"] == 1).sum())
        present = [c for c in CODE_NAME if (df["type_Tur"] == c).any()]
        print(f"{nice}: {len(df):,} pts | {n_anom} expert-labelled anomalies | "
              f"types: {', '.join(present)}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["Timestamp"], y=df["Tur"], mode="lines",
            line=dict(color="#2563eb", width=1), name="turbidity (raw)",
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} NTU<extra></extra>",
        ))
        for code in present:
            sub = df[df["type_Tur"] == code]
            fig.add_trace(go.Scatter(
                x=sub["Timestamp"], y=sub["Tur"], mode="markers",
                marker=dict(color=CODE_COLOR.get(code, "#000"), size=6),
                name=f"{code}: {CODE_NAME[code]}  (n={len(sub)})",
                hovertemplate=(f"%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.1f}} NTU<br>"
                               f"{CODE_NAME[code]}<extra></extra>"),
            ))
        fig.update_layout(
            template="plotly_white",
            title=(f"{nice} — real turbidity with EXPERT anomaly labels (oddwater / "
                   f"Talagala et al. 2019). Highlighted = flagged by a hydrographer."),
            xaxis_title="datetime", yaxis_title="turbidity (NTU)",
            height=560, hovermode="closest",
            legend=dict(orientation="v", y=1, x=1.01, xanchor="left"),
            margin=dict(r=340),
        )
        out = FIGDIR / f"oddwater_labeled_{stem.replace('data_','').replace('_anom','')}.html"
        fig.write_html(out, include_plotlyjs=True)
        print(f"   wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
