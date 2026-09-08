"""Stack every PROVISIONAL series on one zoomable page, one panel above another.

Provisional data is what USGS has *not* signed off: no record processing, no
fouling or calibration-drift correction, nothing clearly-erroneous removed
(CLAUDE.md §9, §9.3). It is therefore the closest thing in the project to the
raw problem the QC tool exists to solve, and the first thing worth doing with a
fresh provisional pull is looking at all of it at once — which artifacts are
common, which gauges misbehave, whether a series is worth keeping at all.

This is a *review* aid, not part of the QC pipeline. It discovers every
``data/<variable>/provisional/*.csv`` and gives each its own panel on a shared,
zoomable x-axis.

Why one panel per series rather than one overlaid axis
-----------------------------------------------------
Specific conductance spans three orders of magnitude between gauges — a granite
headwater runs ~20 uS/cm and a tidal reach tens of thousands — and turbidity is
in different units entirely. Overlaying them on one axis would flatten every
low-scale series into a straight line at the bottom. Each panel keeps its own
y-axis, its own unit, and **rescales that y-axis to whatever is in view** as you
zoom the shared x-axis (Plotly does not do this itself; the handler comes from
``visualize.py``).

Real gaps in the raw USGS feed are *missing rows*, not NaN, so a NaN break is
inserted wherever a gap exceeds ``--max-gap`` (default 3h) — matching the
project's "unbroken stretch" definition. A span continuous apart from a few
scattered dropouts renders as one line; a real outage lifts the pen.

CLI
---
    # Every provisional series the repo has, all variables:
    python -m src.workbench.visualize_provisional

    # Just specific conductance, don't open a browser:
    python -m src.workbench.visualize_provisional --variable specific_conductance --no-open

    # Specific files, in the order given:
    python -m src.workbench.visualize_provisional data/specific_conductance/provisional/*.csv
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.datasets.pull_usgs import DEFAULT_MAX_GAP, VARIABLES, longest_unbroken_run_days
from src.workbench.visualize import (
    PANEL_COLORS,
    _AUTOSCALE_Y_JS,
    insert_gap_breaks,
    load_series,
)

DEFAULT_DATA_ROOT = Path("data")
DEFAULT_OUT = Path("figures/provisional_overview.html")

#: Fallback when a file's variable cannot be identified from its path.
UNKNOWN_UNIT = "value"

#: Panel height in pixels. Enough to read an excursion's shape without making a
#: five-panel page unscrollable.
PANEL_HEIGHT = 300


def discover(root: Path = DEFAULT_DATA_ROOT, variable: str | None = None) -> list[Path]:
    """Every ``data/<variable>/provisional/*.csv``, sorted by variable then site.

    Restricted to the ``provisional/`` directories on purpose: the approval split
    is load-bearing (CLAUDE.md §9), so this page can never silently include an
    approved base and describe it as unvetted.
    """
    names = [variable] if variable else sorted(VARIABLES)
    paths: list[Path] = []
    for name in names:
        paths.extend(sorted((root / name / "provisional").glob("*.csv")))
    return paths


def variable_for(path: Path) -> tuple[str, str]:
    """``(variable_name, unit)`` for a series file, from its ``data/`` location.

    Falls back to the unit-less ``UNKNOWN_UNIT`` rather than guessing, because a
    wrong unit on an axis is a quiet way to mislead a reader about scale.
    """
    for parent in path.parents:
        if parent.name in VARIABLES:
            v = VARIABLES[parent.name]
            return v.name, v.unit
    return path.parent.name or "unknown", UNKNOWN_UNIT


def panel_title(path: Path, s: pd.Series, unit: str, max_gap: str) -> str:
    """One-line panel header: what the series is and how much of it there is."""
    span_days = (s.index.max() - s.index.min()).total_seconds() / 86400.0
    step = pd.Series(s.index).diff().median()
    step_min = step.total_seconds() / 60.0 if pd.notna(step) else float("nan")
    return (
        f"{path.stem}  —  {len(s):,} pts | {span_days:.0f} d | "
        f"{step_min:.0f}-min | {s.min():.1f}–{s.max():.1f} {unit}"
    )


def describe(path: Path, s: pd.Series, unit: str, max_gap: str) -> str:
    """Terminal summary for one series, mirroring ``visualize._describe``."""
    steps = pd.Series(s.index).diff()
    n_gaps = int((steps > pd.Timedelta(max_gap)).sum())
    span_days = (s.index.max() - s.index.min()).total_seconds() / 86400.0
    longest = longest_unbroken_run_days(s, pd.Timedelta(max_gap))
    return (
        f"  {path.stem}: {len(s):,} pts | {span_days:.0f} d | "
        f"~{steps.median()} step | {n_gaps} gaps>{max_gap} | "
        f"longest unbroken {longest:.0f} d | "
        f"range {s.min():.1f}-{s.max():.1f} {unit}"
    )


def build_figure(
    paths: list[Path],
    series: dict[Path, pd.Series],
    max_gap: str = DEFAULT_MAX_GAP,
) -> go.Figure:
    """Stacked, shared-x figure: one panel per provisional series."""
    fig = make_subplots(
        rows=len(paths),
        cols=1,
        shared_xaxes=True,
        subplot_titles=[
            panel_title(p, series[p], variable_for(p)[1], max_gap) for p in paths
        ],
        vertical_spacing=min(0.06, 1.5 / max(len(paths), 1)),
    )
    for i, path in enumerate(paths, start=1):
        unit = variable_for(path)[1]
        s = insert_gap_breaks(series[path], max_gap)
        fig.add_trace(
            go.Scattergl(
                x=s.index,
                # Plain Python list (not a numpy array) so plotly serialises y as
                # a JSON number array rather than a base64 typed array; the
                # in-browser y-autoscale handler indexes y[j] directly.
                y=s.to_numpy().tolist(),
                mode="lines",
                line=dict(color=PANEL_COLORS[(i - 1) % len(PANEL_COLORS)], width=1),
                name=path.stem,
                connectgaps=False,
                hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.1f}} {unit}<extra></extra>",
            ),
            row=i,
            col=1,
        )
        fig.update_yaxes(title_text=unit, row=i, col=1)

    fig.update_xaxes(title_text="datetime", row=len(paths), col=1)
    fig.update_layout(
        template="plotly_white",
        title="Provisional series — unapproved USGS data, all gauges",
        height=max(PANEL_HEIGHT + 120, PANEL_HEIGHT * len(paths)),
        showlegend=False,
        hovermode="x",
        margin=dict(t=80, r=30, l=70, b=50),
    )
    return fig


def visualize_provisional(
    paths: list[Path],
    out: Path = DEFAULT_OUT,
    open_browser: bool = True,
    max_gap: str = DEFAULT_MAX_GAP,
) -> Path:
    """Load ``paths``, write the stacked HTML to ``out``, and return its path."""
    if not paths:
        raise ValueError("No provisional CSV files to visualize.")

    series = {p: load_series(p) for p in paths}
    print(f"Loaded {len(series)} provisional series:")
    for path in paths:
        print(describe(path, series[path], variable_for(path)[1], max_gap))

    fig = build_figure(paths, series, max_gap)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, post_script=_AUTOSCALE_Y_JS)
    print(f"Wrote {out}")

    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def _resolve_paths(args_paths: list[str], variable: str | None) -> list[Path]:
    """The CSVs to plot: the ones given, else every provisional series found."""
    if args_paths:
        paths = [Path(p) for p in args_paths]
        missing = [p for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(f"CSV(s) not found: {', '.join(map(str, missing))}")
        return paths

    paths = discover(variable=variable)
    if not paths:
        where = f"data/{variable}/provisional/" if variable \
            else "data/*/provisional/"
        raise FileNotFoundError(
            f"No provisional CSVs found in {where}. Pull some first, e.g. "
            f"`python -m src.datasets.pull_usgs --variable specific_conductance "
            f"--approval provisional`."
        )
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stack every provisional series on one zoomable page."
    )
    parser.add_argument(
        "paths", nargs="*",
        help="CSV file(s) to plot (default: every data/<variable>/provisional/*.csv).",
    )
    parser.add_argument(
        "--variable", choices=sorted(VARIABLES), default=None,
        help="Only plot this variable's provisional series (default: all of them).",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT,
        help=f"Output HTML path (default: {DEFAULT_OUT}).",
    )
    parser.add_argument(
        "--max-gap", default=DEFAULT_MAX_GAP,
        help=f"Break the line only where a gap exceeds this (pandas offset, e.g. "
             f"'3h', '90min'; default: {DEFAULT_MAX_GAP}). Smaller gaps are bridged.",
    )
    parser.add_argument(
        "--no-open", action="store_true",
        help="Write the HTML but do not open it in a browser.",
    )
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.paths, args.variable)
    visualize_provisional(
        paths, out=args.out, open_browser=not args.no_open, max_gap=args.max_gap
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
