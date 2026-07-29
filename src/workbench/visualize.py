"""Quick CLI tool to eyeball turbidity series and spot clean sections.

This is a *review* aid, not part of the QC pipeline: it plots each CSV
(``datetime``/``value``) as its own stacked line panel with a shared x-axis and
writes an interactive HTML you can zoom and pan. Each series gets its own panel
(and y-axis) because the gauges live on very different scales, so overlaying
them on one axis would flatten the low-turbidity sites.

Real gaps in the raw USGS feed show up as *missing rows*, not NaN (see
``pull_usgs.py``). To make meaningful dropouts visible, we insert a NaN break
wherever a gap exceeds ``--max-gap`` (default 3h) — matching the project's
"unbroken stretch" definition — so a span that is continuous apart from a few
scattered samples renders as one line, while real gaps lift the line.

CLI
---
    # View every approved turbidity CSV in data/raw/approved (default):
    python -m src.workbench.visualize

    # Specific files, custom output, and don't auto-open a browser:
    python -m src.workbench.visualize data/raw/provisional/06818000_turbidity_63680_provisional.csv \\
        --out figures/one.html --no-open

    # Only break the line at gaps longer than 6 hours:
    python -m src.workbench.visualize --max-gap 6h
"""
from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.datasets.pull_usgs import DEFAULT_MAX_GAP, longest_unbroken_run_days

# Calm, distinguishable panel colors (blue / green / amber, then cycled).
PANEL_COLORS: tuple[str, ...] = ("#2563eb", "#059669", "#d97706", "#7c3aed")
DEFAULT_RAWDIR = Path("data/raw/approved")
DEFAULT_OUT = Path("figures/turbidity_overview.html")

# Injected into the HTML after the plot is drawn. Plotly does not rescale the
# y-axis when you zoom the x-axis, so on every x-range change we recompute each
# panel's y-range from only the points currently in view (and pad it 5%). The
# x timestamps are parsed to numbers once and cached. We ignore relayout events
# that don't touch an x-axis, which also stops our own y-updates from recursing.
_AUTOSCALE_Y_JS = """
var gd = document.getElementById('{plot_id}');

function _axKey(prefix, id) { return prefix + id.slice(1); }  // 'x2' -> 'xaxis2'

function _visibleYRanges() {
    var data = gd.data, full = gd._fullLayout;
    if (!gd._xnum) {
        gd._xnum = data.map(function (tr) {
            return (tr.x || []).map(function (v) { return +new Date(v); });
        });
    }
    var byY = {};
    data.forEach(function (tr, i) {
        var yid = tr.yaxis || 'y';
        (byY[yid] = byY[yid] || []).push(i);
    });
    var updates = {};
    Object.keys(byY).forEach(function (yid) {
        var idxs = byY[yid];
        var xax = full[_axKey('xaxis', data[idxs[0]].xaxis || 'x')];
        if (!xax || !xax.range) return;
        var x0 = +new Date(xax.range[0]), x1 = +new Date(xax.range[1]);
        if (x1 < x0) { var t = x0; x0 = x1; x1 = t; }
        var ymin = Infinity, ymax = -Infinity;
        idxs.forEach(function (i) {
            var xs = gd._xnum[i], ys = data[i].y;
            for (var j = 0; j < xs.length; j++) {
                var yv = ys[j];
                if (yv === null || yv === undefined || isNaN(yv)) continue;
                if (xs[j] < x0 || xs[j] > x1) continue;
                if (yv < ymin) ymin = yv;
                if (yv > ymax) ymax = yv;
            }
        });
        if (ymin === Infinity) return;  // nothing visible in this panel
        var pad = (ymax > ymin) ? (ymax - ymin) * 0.05 : (Math.abs(ymax) * 0.05 || 1);
        updates[_axKey('yaxis', yid) + '.range'] = [ymin - pad, ymax + pad];
    });
    return updates;
}

gd.on('plotly_relayout', function (ed) {
    var keys = Object.keys(ed || {});
    if (!keys.some(function (k) { return k.indexOf('xaxis') === 0; })) return;
    var upd = _visibleYRanges();
    if (Object.keys(upd).length) Plotly.relayout(gd, upd);
});
"""


def load_series(path: Path) -> pd.Series:
    """Load a ``datetime``/``value`` CSV into a time-indexed float Series.

    Rows are sorted by time and duplicate timestamps are dropped (keeping the
    first). The returned Series is named after the file stem so it can label a
    panel.
    """
    df = pd.read_csv(path, usecols=["datetime", "value"], parse_dates=["datetime"])
    s = (
        df.set_index("datetime")["value"]
        .sort_index()
    )
    s = s[~s.index.duplicated(keep="first")]
    s.name = path.stem
    return s


def insert_gap_breaks(s: pd.Series, max_gap: str = DEFAULT_MAX_GAP) -> pd.Series:
    """Insert a NaN sample inside every real gap so the plotted line breaks there.

    A "gap" is any step between consecutive timestamps larger than ``max_gap``
    (a pandas offset, e.g. ``"3h"``). Small scattered dropouts below ``max_gap``
    are bridged, so a stretch that is continuous apart from a few missing
    samples renders as one unbroken line — the whole point of hunting clean
    sections. All original points are kept; we only add NaN sentinels, so no
    data is lost. Series shorter than 3 points are returned unchanged.
    """
    if len(s) < 3:
        return s
    steps = pd.Series(s.index).diff()
    median_step = steps.median()
    if pd.isna(median_step) or median_step == pd.Timedelta(0):
        return s
    gap_mask = (steps > pd.Timedelta(max_gap)).to_numpy()
    if not gap_mask.any():
        return s
    # Place each break one median-step into the gap (safely inside it).
    break_index = pd.DatetimeIndex(s.index[gap_mask]) - median_step
    breaks = pd.Series(np.nan, index=break_index, name=s.name)
    return pd.concat([s, breaks]).sort_index()


def detect_spikes(
    s: pd.Series,
    thresh: float = 16.0,
    window: int = 48,
    max_width: int = 3,
) -> pd.Series:
    """Boolean mask of true isolated spikes (and physically impossible negatives).

    The discriminator between a *sensor spike* and a *genuine event* (storm rise,
    level shift, plateau) is **width**: a spike is a narrow pop of one to a few
    samples that the surrounding baseline ignores, whereas a storm stays elevated
    for many samples and a level shift never comes back. So a point is a spike
    only when both hold:

    - its residual from a **centred** rolling-median baseline (``window`` points
      wide) exceeds ``thresh`` robust scale units — it sticks out from the local
      trend; the centred median is barely moved by a narrow pop but *tracks* a
      real multi-hour storm, so storm samples have small residuals, and
    - it belongs to a contiguous run of such points no longer than ``max_width``
      — this rejects storms and the transition band of a level shift, which
      produce long runs of elevated residuals.

    Scale is the rolling MAD about that baseline, floored so a dead-calm stretch
    can't make the ratio explode. Negative turbidity is impossible, so any
    ``value < 0`` is flagged regardless of width.

    This is a *review* heuristic for eyeballing where genuine spikes sit, not the
    QC pipeline's detector. ``max_width`` is in samples, so at a 15-min step
    ``3`` means "up to 45 min"; lower ``thresh`` or higher ``max_width`` = more
    marks.
    """
    v = s.astype(float)
    minp = max(3, window // 3)
    # Centred baseline: a narrow spike is ~1/window of the window, so the median
    # ignores it; a sustained storm pulls the median up with it.
    med = v.rolling(window, center=True, min_periods=minp).median()
    resid = v - med
    scale = resid.abs().rolling(window, center=True, min_periods=minp).median()
    floor = (0.02 * med.abs()).clip(lower=0.5)
    scale = scale.clip(lower=floor).fillna(floor)

    elevated = (resid.abs() / scale > thresh).fillna(False).to_numpy()

    # Run-length filter: keep only runs of elevated points no wider than
    # ``max_width`` — wider excursions are events, not spikes.
    if elevated.any():
        starts = np.r_[True, elevated[1:] != elevated[:-1]]
        run_id = np.cumsum(starts)
        run_len = pd.Series(elevated).groupby(run_id).transform("size").to_numpy()
        spike = elevated & (run_len <= max_width)
    else:
        spike = elevated

    spike = spike | (v < 0).to_numpy()
    return pd.Series(spike, index=s.index)


def _describe(name: str, s: pd.Series, max_gap: str) -> str:
    """One-line terminal summary of a series (points, span, gaps, longest run)."""
    steps = pd.Series(s.index).diff()
    median_step = steps.median()
    n_gaps = int((steps > pd.Timedelta(max_gap)).sum())
    span_days = (s.index.max() - s.index.min()).total_seconds() / 86400.0
    longest = longest_unbroken_run_days(s, pd.Timedelta(max_gap))
    return (
        f"  {name}: {len(s):,} pts | {span_days:.0f} d | "
        f"~{median_step} step | {n_gaps} gaps>{max_gap} | "
        f"longest unbroken {longest:.0f} d | "
        f"range {s.min():.1f}-{s.max():.1f} FNU"
    )


def build_figure(
    series: dict[str, pd.Series],
    max_gap: str = DEFAULT_MAX_GAP,
    mark_spikes: bool = False,
    spike_thresh: float = 16.0,
) -> go.Figure:
    """Build a stacked, shared-x figure with one line panel per series.

    When ``mark_spikes`` is set, each panel also overlays a red open-circle
    marker at every point :func:`detect_spikes` flags, so excursions are easy to
    locate against the baseline.
    """
    names = list(series)
    fig = make_subplots(
        rows=len(names),
        cols=1,
        shared_xaxes=True,
        subplot_titles=names,
        vertical_spacing=0.06,
    )
    for i, name in enumerate(names, start=1):
        raw = series[name]
        s = insert_gap_breaks(raw, max_gap)
        color = PANEL_COLORS[(i - 1) % len(PANEL_COLORS)]
        fig.add_trace(
            go.Scattergl(
                x=s.index,
                # Plain Python list (not a numpy array) so plotly serialises y
                # as a JSON number array rather than a base64 typed array; the
                # in-browser y-autoscale handler indexes y[j] directly.
                y=s.to_numpy().tolist(),
                mode="lines",
                line=dict(color=color, width=1),
                name=name,
                connectgaps=False,
                hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU<extra></extra>",
            ),
            row=i,
            col=1,
        )
        if mark_spikes:
            # Detect on the original series (no NaN gap-break sentinels) so the
            # rolling median/MAD isn't polluted by inserted NaNs.
            mask = detect_spikes(raw, thresh=spike_thresh).to_numpy()
            sp = raw[mask]
            fig.add_trace(
                go.Scattergl(
                    x=sp.index,
                    y=sp.to_numpy().tolist(),  # list: same reason as above
                    mode="markers",
                    marker=dict(
                        color="#dc2626", size=6, symbol="circle-open",
                        line=dict(width=1.2),
                    ),
                    name=f"{name} spikes",
                    hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.1f} FNU (spike)<extra></extra>",
                ),
                row=i,
                col=1,
            )
        fig.update_yaxes(title_text="FNU", row=i, col=1)
    fig.update_xaxes(title_text="datetime", row=len(names), col=1)
    fig.update_layout(
        template="plotly_white",
        title="Turbidity — clean-section review",
        height=max(320, 300 * len(names)),
        showlegend=False,
        hovermode="x",
        margin=dict(t=70, r=30, l=60, b=50),
    )
    return fig


def visualize(
    paths: list[Path],
    out: Path = DEFAULT_OUT,
    open_browser: bool = True,
    max_gap: str = DEFAULT_MAX_GAP,
    mark_spikes: bool = False,
    spike_thresh: float = 16.0,
) -> Path:
    """Load ``paths``, write an interactive HTML to ``out``, and return its path."""
    if not paths:
        raise ValueError("No CSV files to visualize.")
    series = {p.stem: load_series(p) for p in paths}

    print(f"Loaded {len(series)} series:")
    for name, s in series.items():
        print(_describe(name, s, max_gap))
        if mark_spikes:
            print(f"    {int(detect_spikes(s, thresh=spike_thresh).sum())} spikes marked")

    fig = build_figure(series, max_gap, mark_spikes=mark_spikes, spike_thresh=spike_thresh)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs=True, post_script=_AUTOSCALE_Y_JS)
    print(f"Wrote {out}")

    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def _resolve_paths(args_paths: list[str]) -> list[Path]:
    """Return the CSVs to plot: the given files, or all turbidity CSVs in raw."""
    if args_paths:
        paths = [Path(p) for p in args_paths]
    else:
        paths = sorted(DEFAULT_RAWDIR.glob("*_turbidity_*.csv"))
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"CSV(s) not found: {', '.join(map(str, missing))}")
    if not paths:
        raise FileNotFoundError(f"No turbidity CSVs found in {DEFAULT_RAWDIR}/.")
    return paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Plot turbidity series as stacked line charts to spot clean sections."
    )
    parser.add_argument(
        "paths", nargs="*",
        help=f"CSV file(s) to plot (default: all *_turbidity_*.csv in {DEFAULT_RAWDIR}).",
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
    parser.add_argument(
        "--mark-spikes", action="store_true",
        help="Overlay red markers on isolated spikes (and negative readings).",
    )
    parser.add_argument(
        "--spike-thresh", type=float, default=16.0,
        help="Spike sensitivity: MAD-scaled deviation above which a point is "
             "marked (lower = more marks; default: 15).",
    )
    args = parser.parse_args(argv)

    paths = _resolve_paths(args.paths)
    visualize(
        paths, out=args.out, open_browser=not args.no_open, max_gap=args.max_gap,
        mark_spikes=args.mark_spikes, spike_thresh=args.spike_thresh,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
