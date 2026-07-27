"""Plot injected datasets with their ground-truth anomalies coloured by type.

Companion to ``visualize.py`` (which reviews raw/approved series). For each
injected dataset — ``data/injected/<gauge>_l<level>.csv`` plus its row-aligned
``*_labels.csv`` (§5) — this stacks the uninjected base on top and one panel per
level below it: the base panel shows the clean (uninjected) series, and each level
panel shows that level's injected series with every labelled anomaly as a coloured
marker by type. So you can compare each level against the same base directly above.
One HTML per gauge, four stacked panels (base / l1 / l2 / l3), shared zoomable
x-axis.

Each anomaly is marked at its (contaminated) ``value``. **Injected gaps** have a
NaN ``value``, so they are marked at the removed ``true_value`` instead — which
means only *injected* gaps show as markers; *natural* gaps carry no ``true_value``
and simply render as breaks in the line (same gap-break logic as ``visualize``).

CLI
---
    # Every gauge found in data/injected (default) -> figures/injected_<gauge>.html:
    python -m src.tools.visualize_injected

    # One gauge, don't open a browser:
    python -m src.tools.visualize_injected --gauge 12340500 --no-open
"""
from __future__ import annotations

import argparse
import re
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pull_usgs import DEFAULT_MAX_GAP
from src.tools.visualize import _AUTOSCALE_Y_JS, insert_gap_breaks

# One stable colour per anomaly type (spike/plateau/level_shift/gap), matching
# the four failure types in CLAUDE.md §6.
ANOMALY_COLORS: dict[str, str] = {
    "spike": "#dc2626",        # red
    "plateau": "#d97706",      # amber
    "level_shift": "#7c3aed",  # purple
    "gap": "#0891b2",          # cyan (injected gaps, marked at true_value)
}
BASE_COLOR = "#2563eb"
CLEAN_COLOR = "#94a3b8"  # faint grey: the uninjected base drawn behind the injected line
DEFAULT_INJECTED_DIR = Path("data/injected")
DEFAULT_OUTDIR = Path("figures")
LEVELS: tuple[int, ...] = (1, 2, 3)


def find_gauges(injected_dir: Path = DEFAULT_INJECTED_DIR) -> list[str]:
    """Return the sorted gauge ids that have at least one ``<gauge>_l<level>.csv``."""
    gauges = set()
    for p in injected_dir.glob("*_l[1-3].csv"):
        m = re.match(r"(.+)_l[1-3]$", p.stem)
        if m:
            gauges.add(m.group(1))
    return sorted(gauges)


def load_injected(gauge: str, level: int, injected_dir: Path = DEFAULT_INJECTED_DIR):
    """Load one injected dataset joined to its labels, indexed by datetime.

    Returns a frame with:
    - ``value``: the injected (contaminated) series;
    - ``clean``: the original uninjected base, reconstructed as ``true_value``
      where a row is anomalous and ``value`` where it is not — this is exactly the
      pre-injection series (NaN only at *natural* gaps, which have no true_value);
    - ``anomaly_type`` and ``mark_y`` (the y at which to draw the anomaly marker:
      ``value``, or ``true_value`` where ``value`` is NaN, i.e. an injected gap).
    """
    data = pd.read_csv(
        injected_dir / f"{gauge}_l{level}.csv", parse_dates=["datetime"]
    ).set_index("datetime")
    labels = pd.read_csv(
        injected_dir / f"{gauge}_l{level}_labels.csv", parse_dates=["datetime"]
    ).set_index("datetime")
    out = pd.DataFrame(index=data.index)
    out["value"] = data["value"]
    lab = labels.reindex(out.index)
    is_anom = lab["is_anomaly"].fillna(False).astype(bool)
    out["anomaly_type"] = lab["anomaly_type"].fillna("")
    out["clean"] = out["value"].where(~is_anom, lab["true_value"])
    out["mark_y"] = out["value"].where(out["value"].notna(), lab["true_value"])
    return out


def build_figure(gauge: str, levels: dict[int, pd.DataFrame], max_gap: str = DEFAULT_MAX_GAP) -> go.Figure:
    """Stacked figure: the uninjected base on top, then one panel per level.

    Top panel is the clean (uninjected) base; each level panel below shows that
    level's injected series with anomalies as coloured markers by type. The base is
    identical for every level, so it is drawn once from the lowest level present.
    """
    lv = sorted(levels)
    n_rows = len(lv) + 1
    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        subplot_titles=[f"{gauge} — clean base (uninjected)"]
        + [f"{gauge} — level {n} (injected)" for n in lv],
        vertical_spacing=0.05,
    )

    # Row 1: the clean base (same across levels).
    clean = insert_gap_breaks(levels[lv[0]]["clean"], max_gap)
    fig.add_trace(
        go.Scattergl(
            x=clean.index,
            y=clean.to_numpy().tolist(),
            mode="lines",
            line=dict(color=CLEAN_COLOR, width=1),
            name="clean base (uninjected)",
            legendgroup="clean",
            connectgaps=False,
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (clean)<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="FNU", row=1, col=1)

    # Rows 2..N: one injected level per panel.
    for i, level in enumerate(lv):
        row = i + 2
        df = levels[level]
        line = insert_gap_breaks(df["value"], max_gap)
        fig.add_trace(
            go.Scattergl(
                x=line.index,
                # Plain list so plotly emits a JSON number array (not base64),
                # which the in-browser y-autoscale handler indexes directly.
                y=line.to_numpy().tolist(),
                mode="lines",
                line=dict(color=BASE_COLOR, width=1),
                name="injected value",
                legendgroup="value",
                showlegend=(i == 0),
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU<extra></extra>",
            ),
            row=row,
            col=1,
        )
        for atype, color in ANOMALY_COLORS.items():
            mask = (df["anomaly_type"] == atype) & df["mark_y"].notna()
            if not mask.any():
                continue
            sub = df.loc[mask]
            fig.add_trace(
                go.Scattergl(
                    x=sub.index,
                    y=sub["mark_y"].to_numpy().tolist(),
                    mode="markers",
                    marker=dict(color=color, size=5, symbol="circle"),
                    name=atype,
                    legendgroup=atype,
                    showlegend=(i == 0),
                    hovertemplate=(
                        "%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU<br>"
                        f"({atype})<extra></extra>"
                    ),
                ),
                row=row,
                col=1,
            )
        fig.update_yaxes(title_text="FNU", row=row, col=1)
    fig.update_xaxes(title_text="datetime", row=n_rows, col=1)
    fig.update_layout(
        template="plotly_white",
        title=f"Injected turbidity — {gauge} (uninjected base on top; anomalies by type)",
        height=max(340, 300 * n_rows),
        hovermode="closest",
        margin=dict(t=80, r=30, l=60, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def visualize_gauge(
    gauge: str,
    injected_dir: Path = DEFAULT_INJECTED_DIR,
    outdir: Path = DEFAULT_OUTDIR,
    open_browser: bool = True,
    max_gap: str = DEFAULT_MAX_GAP,
) -> Path:
    """Build and write ``figures/injected_<gauge>.html``; return its path."""
    levels = {}
    for level in LEVELS:
        if (injected_dir / f"{gauge}_l{level}.csv").is_file():
            levels[level] = load_injected(gauge, level, injected_dir)
    if not levels:
        raise FileNotFoundError(f"No injected datasets found for gauge {gauge!r}.")

    print(f"{gauge}: {len(levels)} level(s)")
    for level, df in sorted(levels.items()):
        counts = df["anomaly_type"].value_counts()
        summary = ", ".join(f"{t} {int(counts.get(t, 0))}" for t in ANOMALY_COLORS)
        print(f"  l{level}: {len(df):,} rows | {summary}")

    fig = build_figure(gauge, levels, max_gap)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"injected_{gauge}.html"
    fig.write_html(out, include_plotlyjs=True, post_script=_AUTOSCALE_Y_JS)
    print(f"  wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot injected datasets with anomalies coloured by type."
    )
    parser.add_argument(
        "--gauge", help="Gauge id to plot (default: every gauge in data/injected)."
    )
    parser.add_argument(
        "--injected-dir", type=Path, default=DEFAULT_INJECTED_DIR,
        help=f"Directory of injected datasets (default: {DEFAULT_INJECTED_DIR}).",
    )
    parser.add_argument(
        "--outdir", type=Path, default=DEFAULT_OUTDIR,
        help=f"Directory for output HTML (default: {DEFAULT_OUTDIR}).",
    )
    parser.add_argument(
        "--max-gap", default=DEFAULT_MAX_GAP,
        help=f"Break the line where a gap exceeds this (default: {DEFAULT_MAX_GAP}).",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Write the HTML but do not open a browser."
    )
    args = parser.parse_args(argv)

    gauges = [args.gauge] if args.gauge else find_gauges(args.injected_dir)
    if not gauges:
        raise FileNotFoundError(f"No injected datasets found in {args.injected_dir}/.")
    for gauge in gauges:
        visualize_gauge(
            gauge, injected_dir=args.injected_dir, outdir=args.outdir,
            open_browser=not args.no_open, max_gap=args.max_gap,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
