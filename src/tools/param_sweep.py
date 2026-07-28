"""Parameter-range finder: sweep one SaQC parameter and see what it detects.

Phase-2 precursor. The agent (§8) has to pick parameters from a bounded range for
each tool, so we first need to know what range is defensible. This runs one SaQC
method across a grid of values for a single parameter, scores each against the
injected labels (§5), and writes an interactive HTML with two linked views:

  top    — precision / recall / F1 (or RMSE) vs the parameter, to pick the range
  bottom — the series itself with TP / FP / FN marked, on a slider over the grid,
           so you can see *what* each value catches and misses

Scoring is per-row against `anomaly_type`, restricted to the type the tool exists
to find. Rows whose value is NaN are excluded for every tool except `flag_nan` —
nothing can legitimately detect a spike in a missing reading.

Usage
-----
    # list what can be swept
    python -m src.tools.param_sweep --list

    # sweep one parameter (uses the built-in default grid)
    python -m src.tools.param_sweep --dataset data/injected/03447687/l2/03447687_l2.csv \\
        --tool flag_spike_unilof --param thresh

    # your own grid, zoomed to a window
    python -m src.tools.param_sweep --dataset data/injected/03447687/l2/03447687_l2.csv \\
        --tool flag_constants --param thresh --values 0,0.001,0.01,0.1 \\
        --start 2024-07-01 --end 2024-08-01
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import saqc

from src.inspect_data import (
    DATETIME_COL,
    VALUE_COL,
    validate_labels_csv,
    validate_series_csv,
)

FIELD = VALUE_COL

Kind = Literal["detect", "impute"]

COLORS = {
    "series": "#94a3b8",
    "tp": "#059669",
    "fp": "#dc2626",
    "fn": "#d97706",
    "precision": "#2563eb",
    "recall": "#d97706",
    "f1": "#059669",
    "n_flagged": "#94a3b8",
}


@dataclass(frozen=True)
class ToolSpec:
    """One sweepable QC tool: its SaQC method, what it targets, and its grids."""

    method: str
    target: str  # anomaly_type it exists to find ("" for action tools)
    kind: Kind
    grids: dict[str, list[Any]]  # param -> default sweep grid
    fixed: dict[str, Any] = field(default_factory=dict)
    note: str = ""
    # Rows of slack when matching flags to episodes. A transition detector marks
    # the edge of a segment, not its body, so it needs slack; a plateau detector
    # should land inside the segment and needs none.
    tolerance: int = 0


# Default grids are deliberately WIDE — the point of the sweep is to find where
# the useful band sits, then narrow it. Verified against saqc 2.8.0 signatures.
TOOLS: dict[str, ToolSpec] = {
    "flag_spike_unilof": ToolSpec(
        method="flagUniLOF",
        target="spike",
        kind="detect",
        grids={
            # 1.0 is the LOF floor (a perfect inlier); >10 flags nothing on turbidity.
            # 'auto' last: it is a reference point, not part of the numeric ramp.
            "thresh": [1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0,
                       6.0, 10.0, "auto"],
            "n": [3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 350, 500],
            "p": [1, 2],
            "density": ["auto", 0.2, 0.5, 1.0, 2.0, 5.0],
            "slope_correct": [True, False],
            "min_offset": [None, 0.5, 1.0, 2.0, 5.0, 10.0],
        },
        fixed={"n": 20, "thresh": 1.5},
        note="thresh is the LOF cutoff (lower = more sensitive); n is the neighbourhood.",
    ),
    "flag_zscore": ToolSpec(
        method="flagZScore",
        target="spike",
        kind="detect",
        grids={
            # extends past the textbook 3.0: on turbidity, 'modified' F1 was still
            # climbing at 6.0, so the useful band sits far higher than expected
            "thresh": [2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 25.0, 40.0],
            "window": ["1h", "3h", "6h", "12h", "24h", "3d", "7d"],
            "method": ["standard", "modified"],
        },
        fixed={"window": "12h", "thresh": 3.0, "method": "modified"},
        note="'modified' uses MAD and is far more robust when spikes are dense.",
    ),
    "flag_range": ToolSpec(
        method="flagRange",
        target="spike",
        kind="detect",
        grids={
            "max": [50, 100, 200, 500, 1000, 2000, 5000],
            "min": [-1, 0, 0.1, 1.0],
        },
        fixed={"min": 0},
        note="Physical plausibility limits, not a sensitivity knob — set from domain "
        "range, then confirm here that it does not clip real extreme events.",
    ),
    "flag_constants": ToolSpec(
        method="flagConstants",
        target="plateau",
        kind="detect",
        grids={
            "thresh": [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
            "window": ["1h", "2h", "3h", "6h", "12h", "24h", "48h"],
            "min_periods": [2, 3, 5, 10, 20],
        },
        fixed={"thresh": 0.01, "window": "6h"},
        note="thresh must be much smaller than the signal's noise sd or it swallows "
        "the whole series (§7.1).",
    ),
    "flag_plateau": ToolSpec(
        method="flagPlateau",
        target="plateau",
        kind="detect",
        grids={
            "min_length": ["30min", "1h", "2h", "3h", "6h", "12h"],
            "min_jump": [None, 0.1, 0.5, 1.0, 2.0, 5.0],
            "granularity": [None, 3, 5, 10, 20],
        },
        fixed={"min_length": "1h"},
        note="min_length must sit WELL BELOW the true plateau length (§7.1); it finds "
        "offset segments, not level-matched stuck runs.",
    ),
    "flag_jumps": ToolSpec(
        method="flagJumps",
        target="level_shift",
        kind="detect",
        grids={
            "thresh": [0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            "window": ["1h", "3h", "6h", "12h", "24h", "3d"],
            "min_periods": [0, 2, 5, 10],
        },
        fixed={"thresh": 5.0, "window": "12h"},
        note="thresh is in data units — scale it to the series' own spread. Tune on "
        "EVENT recall: it marks the transition, not the whole shifted window.",
        tolerance=8,
    ),
    "impute_rolling": ToolSpec(
        method="interpolateByRolling",
        target="gap",
        kind="impute",
        grids={
            "window": ["1h", "3h", "6h", "12h", "24h", "48h", "7d"],
            "func": ["median", "mean"],
            "min_periods": [0, 1, 2, 5],
        },
        fixed={"window": "6h", "func": "median"},
        note="Scored as RMSE vs true_value on INJECTED gaps only (§5). window must "
        "exceed the gap or it half-fills it (§7.1).",
    ),
}


# --------------------------------------------------------------------------- data
def load_dataset(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    """Load an injected series with its labels."""
    series_df = validate_series_csv(path)
    series = series_df.set_index(DATETIME_COL)[VALUE_COL].astype(float).sort_index()

    labels_path = path.with_name(f"{path.stem}_labels.csv")
    if not labels_path.exists():
        raise SystemExit(f"labels not found: {labels_path}")
    labels = validate_labels_csv(labels_path).set_index(DATETIME_COL).sort_index()
    labels = labels.reindex(series.index)

    return series, labels


# --------------------------------------------------------------------------- run
def apply_tool(
    series: pd.Series,
    spec: ToolSpec,
    params: dict[str, Any],
) -> tuple[pd.Series, pd.Series]:
    """Run one SaQC method. Returns (flagged mask, resulting values)."""
    qc = saqc.SaQC(pd.DataFrame({FIELD: series}))

    out = getattr(qc, spec.method)(FIELD, **params)
    values = out.data[FIELD]
    flagged = out.flags[FIELD] > saqc.UNFLAGGED

    return flagged.reindex(series.index, fill_value=False), values.reindex(series.index)


def find_events(
    labels: pd.DataFrame, target: str, bridge: int = 24
) -> list[tuple[int, int]]:
    """Contiguous episodes of `target`, as [start, end) positions.

    Label runs get shredded by interleaved gap rows — one injected level_shift on
    06818000_l2 appears as 20+ separate runs. Runs separated only by gap/NaN rows
    (up to `bridge` of them) are therefore merged back into one episode, matching
    the manifest's `n_events`.
    """
    is_target = (labels["anomaly_type"] == target).to_numpy()
    is_gap = (labels["anomaly_type"] == "gap").to_numpy()

    d = np.diff(np.concatenate([[0], is_target.view(np.int8), [0]]))
    starts = list(np.flatnonzero(d == 1))
    ends = list(np.flatnonzero(d == -1))
    if not starts:
        return []

    merged = [[starts[0], ends[0]]]
    for s, e in zip(starts[1:], ends[1:]):
        prev_end = merged[-1][1]
        between = slice(prev_end, s)
        # bridge only across gaps, never across genuinely clean data
        if s - prev_end <= bridge and is_gap[between].all():
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(int(a), int(b)) for a, b in merged]


def score_events(
    flagged: pd.Series, events: list[tuple[int, int]], tolerance: int
) -> dict[str, Any]:
    """Episode-level recall / precision.

    A transition detector (flagJumps) marks the *edge* of a level shift while the
    labels mark the whole window, so point-wise F1 is structurally capped near
    zero. Here an episode counts as detected if any flag lands inside it (or
    within `tolerance` rows of it), and a flag cluster counts as a true positive
    if it reaches any episode.
    """
    if not events:
        return {"event_recall": float("nan"), "event_precision": float("nan"),
                "event_f1": float("nan"), "n_events": 0, "n_events_hit": 0}

    pred = flagged.to_numpy()
    hit_idx = np.flatnonzero(pred)

    n_hit = sum(
        bool(((hit_idx >= a - tolerance) & (hit_idx < b + tolerance)).any())
        for a, b in events
    )

    # cluster contiguous flags, then ask whether each cluster reaches an episode
    clusters: list[tuple[int, int]] = []
    for i in hit_idx:
        if clusters and i - clusters[-1][1] <= max(tolerance, 1):
            clusters[-1] = (clusters[-1][0], int(i))
        else:
            clusters.append((int(i), int(i)))
    good = sum(
        any(c0 <= b + tolerance and c1 >= a - tolerance for a, b in events)
        for c0, c1 in clusters
    )

    recall = n_hit / len(events)
    precision = good / len(clusters) if clusters else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1,
        "n_events": len(events),
        "n_events_hit": n_hit,
        "n_clusters": len(clusters),
    }


def score_detect(
    flagged: pd.Series, labels: pd.DataFrame, series: pd.Series, target: str
) -> dict[str, Any]:
    """Per-row precision / recall / F1 for one anomaly type."""
    eligible = series.notna() if target != "gap" else pd.Series(True, index=series.index)
    truth = (labels["anomaly_type"] == target) & eligible
    pred = flagged & eligible

    tp = int((truth & pred).sum())
    fp = int((~truth & pred).sum())
    fn = int((truth & ~pred).sum())

    # Split FPs: hitting a *different* anomaly is not the same error as hitting clean data.
    other = labels["anomaly_type"].isin([""]).eq(False) & ~truth
    fp_other_anomaly = int((pred & other).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "fp_other_anomaly": fp_other_anomaly,
        "fp_clean": fp - fp_other_anomaly,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_flagged": int(pred.sum()),
    }


def baseline_series(series: pd.Series, kind: Kind) -> pd.Series:
    """What the action is measured against (§10).

    Imputation competes with linear interpolation.
    """
    if kind == "impute":
        return series.interpolate(method="time", limit_direction="both")
    return series


def score_values(
    values: pd.Series,
    labels: pd.DataFrame,
    baseline: pd.Series,
    target: str,
) -> dict[str, Any]:
    """RMSE / MAE vs true_value on the rows this action is meant to fix.

    `n_scored` matters as much as the error: a narrow rolling window fills only
    part of each gap, so its RMSE is computed over an easier subset. We therefore
    also report the error on the rows filled by *this* setting AND the baseline,
    plus coverage, so the two are comparable.
    """
    mask = (labels["anomaly_type"] == target) & labels["true_value"].notna()
    if target == "gap":  # only injected gaps carry a true_value (§5)
        mask &= labels["source"] == "injected"

    truth = labels.loc[mask, "true_value"].astype(float)
    got = values.loc[mask]
    base = baseline.loc[mask]
    n_target = int(mask.sum())

    both = truth.notna() & got.notna()
    n = int(both.sum())
    if n == 0:
        return {"rmse": float("nan"), "mae": float("nan"), "n_scored": 0,
                "n_target": n_target, "coverage": 0.0,
                "baseline_rmse": float("nan")}

    err = (got[both] - truth[both]).astype(float)
    base_err = (base[both] - truth[both]).astype(float)
    return {
        "rmse": float(np.sqrt((err**2).mean())),
        "mae": float(err.abs().mean()),
        "n_scored": n,
        "n_target": n_target,
        "coverage": n / n_target if n_target else 0.0,
        # scored on the SAME rows, so the comparison is like-for-like
        "baseline_rmse": float(np.sqrt((base_err**2).mean())),
    }


def sweep(
    series: pd.Series,
    labels: pd.DataFrame,
    tool: str,
    param: str,
    values: list[Any],
    tolerance: int = 0,
) -> tuple[pd.DataFrame, list[pd.Series], list[pd.Series]]:
    """Run the tool once per value. Returns (metrics, flag masks, value series)."""
    spec = TOOLS[tool]
    rows, masks, outs = [], [], []
    baseline = baseline_series(series, spec.kind)
    events = find_events(labels, spec.target)

    for v in values:
        params = {**spec.fixed, param: v}
        params = {k: p for k, p in params.items() if p is not None or k == param}
        if v is None:
            params.pop(param, None)

        try:
            flagged, out_values = apply_tool(series, spec, params)
        except Exception as exc:  # noqa: BLE001 - a bad param is a real result
            print(f"  {param}={v!r}: FAILED ({type(exc).__name__}: {exc})")
            rows.append({param: v, "error": type(exc).__name__})
            masks.append(pd.Series(False, index=series.index))
            outs.append(series)
            continue

        if spec.kind == "detect":
            metrics = score_detect(flagged, labels, series, spec.target)
            metrics |= score_events(flagged, events, tolerance)
            print(
                f"  {param}={v!r:>12}  point P={metrics['precision']:.3f} "
                f"R={metrics['recall']:.3f} F1={metrics['f1']:.3f}  |  "
                f"event R={metrics['event_recall']:.2f} "
                f"({metrics['n_events_hit']}/{metrics['n_events']}) "
                f"P={metrics['event_precision']:.2f}  flagged={metrics['n_flagged']}"
            )
        else:
            metrics = score_values(out_values, labels, baseline, spec.target)
            verdict = "beats" if metrics["rmse"] < metrics["baseline_rmse"] else "LOSES to"
            print(
                f"  {param}={v!r:>12}  RMSE={metrics['rmse']:.4f} "
                f"{verdict} baseline {metrics['baseline_rmse']:.4f}  "
                f"coverage={metrics['coverage']:.0%} "
                f"({metrics['n_scored']}/{metrics['n_target']})"
            )

        rows.append({param: v, **metrics})
        masks.append(flagged)
        outs.append(out_values)

    return pd.DataFrame(rows), masks, outs


def sweep_2d(
    series: pd.Series,
    labels: pd.DataFrame,
    tool: str,
    param_a: str,
    values_a: list[Any],
    param_b: str,
    values_b: list[Any],
    tolerance: int = 0,
) -> pd.DataFrame:
    """Cross-product sweep of two parameters.

    A 1D sweep holds the other knobs at a guess, which is misleading when they
    interact — for flagUniLOF the best `n` depends on `thresh` and vice versa.
    Returns long-form rows: one per (a, b) cell.
    """
    spec = TOOLS[tool]
    events = find_events(labels, spec.target)
    baseline = baseline_series(series, spec.kind)
    rows = []

    for va in values_a:
        for vb in values_b:
            params = {**spec.fixed, param_a: va, param_b: vb}
            params = {k: p for k, p in params.items()
                      if p is not None or k in (param_a, param_b)}
            for key, val in ((param_a, va), (param_b, vb)):
                if val is None:
                    params.pop(key, None)
            try:
                flagged, out_values = apply_tool(series, spec, params)
            except Exception as exc:  # noqa: BLE001
                rows.append({param_a: va, param_b: vb, "error": type(exc).__name__})
                continue

            if spec.kind == "detect":
                m = score_detect(flagged, labels, series, spec.target)
                m |= score_events(flagged, events, tolerance)
            else:
                m = score_values(out_values, labels, baseline, spec.target)
            rows.append({param_a: va, param_b: vb, **m})
        print(f"  {param_a}={va!r} done")

    return pd.DataFrame(rows)


def build_heatmap(
    grid: pd.DataFrame, tool: str, param_a: str, values_a: list[Any],
    param_b: str, values_b: list[Any],
) -> go.Figure:
    spec = TOOLS[tool]
    detect = spec.kind == "detect"
    panels = (["f1", "event_f1", "recall", "precision"] if detect
              else ["rmse", "coverage"])
    titles = {"f1": "point F1", "event_f1": "event F1", "recall": "point recall",
              "precision": "point precision", "rmse": "RMSE", "coverage": "coverage"}

    fig = make_subplots(
        rows=2, cols=2, subplot_titles=[titles[p] for p in panels],
        horizontal_spacing=0.13, vertical_spacing=0.14,
    )
    ticks_a = [repr(v) for v in values_a]
    ticks_b = [repr(v) for v in values_b]

    for i, key in enumerate(panels):
        if key not in grid:
            continue
        z = (grid.pivot(index=param_a, columns=param_b, values=key)
             .reindex(index=values_a, columns=values_b).to_numpy(dtype=float))
        r, c = i // 2 + 1, i % 2 + 1
        fig.add_trace(
            go.Heatmap(
                z=z, x=list(range(len(values_b))), y=list(range(len(values_a))),
                colorscale="Viridis" if key != "rmse" else "Viridis_r",
                zmin=0, zmax=1 if key not in ("rmse",) else None,
                colorbar=dict(len=0.38, y=0.81 if r == 1 else 0.19,
                              x=0.43 if c == 1 else 1.005, thickness=12),
                hovertemplate=(f"{param_b}=%{{x}}<br>{param_a}=%{{y}}<br>"
                               f"{titles[key]}=%{{z:.3f}}<extra></extra>"),
            ),
            row=r, col=c,
        )
        fig.update_xaxes(tickmode="array", tickvals=list(range(len(values_b))),
                         ticktext=ticks_b, title_text=param_b, row=r, col=c)
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(values_a))),
                         ticktext=ticks_a, title_text=param_a, row=r, col=c)

        # mark the best cell so the useful region is unmistakable
        if np.isfinite(z).any():
            best = (np.nanargmin(z) if key == "rmse" else np.nanargmax(z))
            by, bx = np.unravel_index(best, z.shape)
            fig.add_trace(
                go.Scatter(x=[bx], y=[by], mode="markers", showlegend=False,
                           marker=dict(symbol="star", size=15, color="#f8fafc",
                                       line=dict(color="#111827", width=1.2)),
                           hovertemplate=f"best {titles[key]} "
                                         f"{z[by, bx]:.3f}<extra></extra>"),
                row=r, col=c,
            )

    fig.update_layout(
        title=dict(text=f"{tool}: {param_a} x {param_b}", x=0.01, font=dict(size=17)),
        height=820, template="plotly_white", showlegend=False,
        margin=dict(l=80, r=90, t=90, b=70),
    )
    return fig


def build_explorer(
    series: pd.Series,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    masks: list[pd.Series],
    tool: str,
    param: str,
    values: list[Any],
    events: list[tuple[int, int]],
    pad: int = 60,
) -> go.Figure:
    """Full-height time series + a threshold slider: what got caught, what didn't.

    Two sliders. The lower one steps the parameter and recolours the markers.
    The upper one zooms to each labelled episode in turn, because at 24k rows a
    two-row spike is invisible at full extent — without it you cannot actually
    see what changed.
    """
    spec = TOOLS[tool]
    eligible = (series.notna() if spec.target != "gap"
                else pd.Series(True, index=series.index))
    truth = (labels["anomaly_type"] == spec.target) & eligible

    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=series.index, y=series.to_numpy(), name="series", mode="lines",
            line=dict(color="#cbd5e1", width=1.2),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>",
        )
    )

    # Ground truth, always visible underneath: every spike we are meant to find.
    fig.add_trace(
        go.Scattergl(
            x=series.index[truth], y=series[truth].to_numpy(),
            name=f"labelled {spec.target} ({int(truth.sum())})", mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=15,
                        line=dict(color="#64748b", width=1.4)),
            hovertemplate=f"labelled {spec.target}<br>%{{x|%Y-%m-%d %H:%M}}<extra></extra>",
        )
    )
    n_static = 2

    for i, mask in enumerate(masks):
        pred = mask & eligible
        vis = i == 0
        for name, sel, color, symbol, size in (
            ("caught", truth & pred, COLORS["tp"], "circle", 9),
            ("MISSED", truth & ~pred, COLORS["fn"], "diamond", 10),
            ("false alarm", ~truth & pred, COLORS["fp"], "x", 7),
        ):
            fig.add_trace(
                go.Scattergl(
                    x=series.index[sel], y=series[sel].to_numpy(),
                    name=f"{name} ({int(sel.sum())})", mode="markers",
                    marker=dict(color=color, size=size, symbol=symbol,
                                line=dict(width=0)),
                    visible=vis, legendgroup=name,
                    hovertemplate=f"{name}<br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}}<extra></extra>",
                )
            )

    per_step = 3
    steps = []
    for i, v in enumerate(values):
        visible = [True] * n_static + [False] * (len(fig.data) - n_static)
        for j in range(per_step):
            visible[n_static + i * per_step + j] = True
        m = metrics.iloc[i]
        title = (f"<b>{tool}</b>   {param} = {v!r}   —   "
                 f"caught {int(m.get('tp', 0))}/{int(m.get('tp', 0) + m.get('fn', 0))} "
                 f"rows &nbsp;|&nbsp; false alarms {int(m.get('fp', 0))} "
                 f"&nbsp;|&nbsp; P {m.get('precision', 0):.2f} "
                 f"R {m.get('recall', 0):.2f} F1 {m.get('f1', 0):.2f}")
        steps.append(dict(method="update", label=repr(v),
                          args=[{"visible": visible}, {"title.text": title}]))

    # episode navigator: zoom to each labelled episode, padded for context
    idx = series.index
    ep_steps = [dict(method="relayout", label="all",
                     args=[{"xaxis.range": [idx[0], idx[-1]]}])]
    for k, (a, b) in enumerate(events):
        lo = idx[max(0, a - pad)]
        hi = idx[min(len(idx) - 1, b + pad)]
        ep_steps.append(dict(method="relayout", label=str(k + 1),
                             args=[{"xaxis.range": [lo, hi]}]))

    fig.update_layout(
        title=dict(text=steps[0]["args"][1]["title.text"], x=0.01, font=dict(size=15)),
        sliders=[
            dict(active=0, y=-0.02, pad=dict(t=40, b=10),
                 currentvalue=dict(prefix=f"{param} = ", font=dict(size=15)),
                 steps=steps),
            dict(active=0, y=-0.28, pad=dict(t=40, b=10),
                 currentvalue=dict(prefix=f"zoom to {spec.target} episode: ",
                                   font=dict(size=13)),
                 steps=ep_steps, len=0.9),
        ],
        height=760, template="plotly_white", hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=70, r=40, t=90, b=210),
        xaxis=dict(rangeslider=dict(visible=True, thickness=0.05)),
        yaxis=dict(title="value"),
    )
    return fig


# --------------------------------------------------------------------------- plot
def build_figure(
    series: pd.Series,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    masks: list[pd.Series],
    outs: list[pd.Series],
    tool: str,
    param: str,
    values: list[Any],
) -> go.Figure:
    spec = TOOLS[tool]
    detect = spec.kind == "detect"
    x = list(range(len(values)))
    ticks = [repr(v) for v in values]

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.34, 0.66],
        vertical_spacing=0.11,
        subplot_titles=(
            f"{'Detection quality' if detect else 'Correction error'} vs {param}",
            f"Drag the slider to see what each {param} value does",
        ),
        specs=[[{"secondary_y": True}], [{}]],
    )

    # ---- row 1: metric curves (always visible)
    if detect:
        for key in ("precision", "recall", "f1"):
            fig.add_trace(
                go.Scatter(
                    x=x, y=metrics.get(key, pd.Series(dtype=float)), name=key.upper(),
                    mode="lines+markers", line=dict(color=COLORS[key], width=2.5),
                ),
                row=1, col=1, secondary_y=False,
            )
        # Episode-level curves: the ones that matter for transition detectors.
        for key, dash in (("event_recall", "dash"), ("event_precision", "dot")):
            if key in metrics:
                fig.add_trace(
                    go.Scatter(
                        x=x, y=metrics[key], name=key.replace("_", " "),
                        mode="lines+markers",
                        line=dict(color=COLORS["recall" if "recall" in key
                                  else "precision"], width=1.6, dash=dash),
                        marker=dict(symbol="square", size=6),
                    ),
                    row=1, col=1, secondary_y=False,
                )
        fig.add_trace(
            go.Scatter(
                x=x, y=metrics.get("n_flagged", pd.Series(dtype=float)),
                name="n flagged", mode="lines", line=dict(color=COLORS["n_flagged"],
                width=1.5, dash="dot"),
            ),
            row=1, col=1, secondary_y=True,
        )
        fig.update_yaxes(title_text="score", range=[0, 1.02], row=1, col=1,
                         secondary_y=False)
        fig.update_yaxes(title_text="n flagged", row=1, col=1, secondary_y=True,
                         showgrid=False)
    else:
        fig.add_trace(
            go.Scatter(x=x, y=metrics.get("rmse", pd.Series(dtype=float)), name="RMSE",
                       mode="lines+markers", line=dict(color=COLORS["f1"], width=2.5)),
            row=1, col=1, secondary_y=False,
        )
        base = metrics.get("baseline_rmse", pd.Series(dtype=float))
        base_name = "linear-interp baseline"
        if len(base):
            fig.add_trace(
                go.Scatter(x=x, y=base, name=base_name, mode="lines",
                           line=dict(color=COLORS["fp"], width=1.5, dash="dash")),
                row=1, col=1, secondary_y=False,
            )
        cov = metrics.get("coverage", pd.Series(dtype=float))
        if len(cov):
            # Without this, a narrow window looks "accurate" purely by filling less.
            fig.add_trace(
                go.Scatter(x=x, y=cov, name="coverage", mode="lines+markers",
                           line=dict(color=COLORS["n_flagged"], width=1.5, dash="dot")),
                row=1, col=1, secondary_y=True,
            )
            fig.update_yaxes(title_text="coverage", range=[0, 1.02], row=1, col=1,
                             secondary_y=True, showgrid=False, tickformat=".0%")
        fig.update_yaxes(title_text="RMSE", row=1, col=1, secondary_y=False)

    n_static = len(fig.data)

    # ---- row 2: the series itself (always visible)
    fig.add_trace(
        go.Scattergl(
            x=series.index, y=series.to_numpy(), name="series", mode="lines",
            line=dict(color=COLORS["series"], width=1),
            hovertemplate="%{x|%Y-%m-%d %H:%M}<br>%{y:.2f}<extra></extra>",
        ),
        row=2, col=1,
    )
    n_static += 1

    # ---- per-step traces
    target_rows = labels["anomaly_type"] == spec.target
    for i, mask in enumerate(masks):
        vis = i == 0
        # highlight the current point on the curve
        fig.add_trace(
            go.Scatter(
                x=[i], y=[metrics.iloc[i].get("f1" if detect else "rmse", 0)],
                mode="markers", marker=dict(size=15, color="rgba(0,0,0,0)",
                line=dict(color="#111827", width=2.5)),
                showlegend=False, visible=vis, hoverinfo="skip",
            ),
            row=1, col=1, secondary_y=False,
        )

        if detect:
            eligible = series.notna() if spec.target != "gap" else pd.Series(True, index=series.index)
            truth = target_rows & eligible
            pred = mask & eligible
            groups = [
                ("TP", truth & pred, COLORS["tp"], 6),
                ("FP", ~truth & pred, COLORS["fp"], 5),
                ("FN (missed)", truth & ~pred, COLORS["fn"], 5),
            ]
            for name, sel, color, size in groups:
                fig.add_trace(
                    go.Scattergl(
                        x=series.index[sel], y=series[sel].to_numpy(),
                        name=f"{name} ({int(sel.sum())})", mode="markers",
                        marker=dict(color=color, size=size,
                                    symbol="x" if name.startswith("FN") else "circle"),
                        visible=vis,
                        hovertemplate=f"{name}<br>%{{x|%Y-%m-%d %H:%M}}<br>%{{y:.2f}}<extra></extra>",
                    ),
                    row=2, col=1,
                )
        else:
            fig.add_trace(
                go.Scattergl(
                    x=outs[i].index, y=outs[i].to_numpy(), name="result",
                    mode="lines", line=dict(color=COLORS["tp"], width=1.5), visible=vis,
                ),
                row=2, col=1,
            )
            tv = labels["true_value"].astype(float)
            sel = target_rows & tv.notna()
            fig.add_trace(
                go.Scattergl(
                    x=series.index[sel], y=tv[sel].to_numpy(), name="true value",
                    mode="markers", marker=dict(color=COLORS["fn"], size=4),
                    visible=vis,
                ),
                row=2, col=1,
            )

    per_step = (len(fig.data) - n_static) // len(masks)

    steps = []
    for i, v in enumerate(values):
        visible = [True] * n_static + [False] * (len(fig.data) - n_static)
        for j in range(per_step):
            visible[n_static + i * per_step + j] = True
        label = f"{metrics.iloc[i]['f1']:.3f}" if detect and "f1" in metrics else ""
        steps.append(
            dict(
                method="update",
                args=[{"visible": visible},
                      {"title.text": f"{tool}  —  {param} = {v!r}"
                                     + (f"   (F1 {label})" if label else "")}],
                label=repr(v),
            )
        )

    fig.update_layout(
        title=dict(text=f"{tool}  —  {param} = {values[0]!r}", x=0.01, font=dict(size=17)),
        sliders=[dict(active=0, currentvalue=dict(prefix=f"{param} = ",
                 font=dict(size=14)), pad=dict(t=60), steps=steps)],
        height=880,
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, x=0),
        margin=dict(l=70, r=60, t=100, b=140),
    )
    fig.update_xaxes(tickmode="array", tickvals=x, ticktext=ticks,
                     title_text=param, row=1, col=1)
    fig.update_yaxes(title_text="value", row=2, col=1)

    if spec.note:
        fig.add_annotation(
            text=f"<i>{spec.note}</i>", xref="paper", yref="paper", x=0, y=-0.16,
            showarrow=False, font=dict(size=11, color="#475569"), align="left",
        )
    return fig


# --------------------------------------------------------------------------- cli
def parse_value(raw: str) -> Any:
    """'3' -> 3, '1.5' -> 1.5, 'none' -> None, '6h' -> '6h'."""
    if raw.lower() in {"none", "null"}:
        return None
    for cast in (int, float):
        try:
            return cast(raw)
        except ValueError:
            continue
    return raw


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="list tools and sweepable params")
    p.add_argument("--dataset", type=Path, help="data/injected/<gauge>/l<level>/<name>.csv")
    p.add_argument("--tool", choices=sorted(TOOLS))
    p.add_argument("--param")
    p.add_argument("--values", help="comma-separated grid (default: built-in)")
    p.add_argument("--param2", help="second param -> 2D heatmap instead of a slider")
    p.add_argument("--values2", help="grid for --param2")
    p.add_argument("--explore", action="store_true",
                   help="full-height series + threshold slider + episode navigator")
    p.add_argument("--tolerance", type=int,
                   help="rows of slack when matching flags to episodes "
                        "(default: per-tool)")
    p.add_argument("--start", help="restrict to this window, e.g. 2024-07-01")
    p.add_argument("--end")
    p.add_argument("--out", type=Path, help="output HTML (default: figures/sweep_*.html)")
    args = p.parse_args(argv)

    if args.list:
        for name, spec in sorted(TOOLS.items()):
            print(f"\n{name}  ->  {spec.method}   [{spec.kind}, target={spec.target!r}]")
            if spec.note:
                print(f"    {spec.note}")
            for prm, grid in spec.grids.items():
                mark = "*" if prm in spec.fixed else " "
                print(f"   {mark} {prm:<14} {grid}")
            print(f"     fixed while sweeping: {spec.fixed}")
        print("\n  * = also has a default value used when another param is swept")
        return 0

    if not (args.dataset and args.tool and args.param):
        p.error("--dataset, --tool and --param are required (or use --list)")

    spec = TOOLS[args.tool]
    if args.param not in spec.grids:
        p.error(f"{args.tool} has no sweepable param {args.param!r}; "
                f"choose from {sorted(spec.grids)}")

    values = ([parse_value(v) for v in args.values.split(",")]
              if args.values else spec.grids[args.param])

    series, labels = load_dataset(args.dataset)
    if args.start or args.end:
        series = series.loc[args.start : args.end]
        labels = labels.loc[args.start : args.end]

    tolerance = args.tolerance if args.tolerance is not None else spec.tolerance
    n_events = len(find_events(labels, spec.target))
    print(f"{args.dataset.name}: {len(series)} rows, "
          f"{int(labels['anomaly_type'].eq(spec.target).sum())} {spec.target!r} rows "
          f"in {n_events} episodes")
    print(f"sweeping {args.tool}.{args.param} over {values}")
    if spec.fixed:
        print(f"holding {({k: v for k, v in spec.fixed.items() if k != args.param})}")
    if spec.kind == "detect":
        print(f"episode match tolerance: {tolerance} rows")

    if args.param2:
        if args.param2 not in spec.grids:
            p.error(f"{args.tool} has no sweepable param {args.param2!r}")
        values2 = ([parse_value(v) for v in args.values2.split(",")]
                   if args.values2 else spec.grids[args.param2])
        print(f"  x {args.param2} over {values2}  "
              f"({len(values) * len(values2)} runs)")
        grid = sweep_2d(series, labels, args.tool, args.param, values,
                        args.param2, values2, tolerance)

        key = "event_f1" if spec.tolerance else ("f1" if spec.kind == "detect" else "rmse")
        if key in grid and grid[key].notna().any():
            i = grid[key].idxmin() if key == "rmse" else grid[key].idxmax()
            best = grid.loc[i]
            ba, bb = best[args.param], best[args.param2]
            print(f"\nbest {key} {best[key]:.3f} at "
                  f"{args.param}={ba.item() if hasattr(ba, 'item') else ba!r}, "
                  f"{args.param2}={bb.item() if hasattr(bb, 'item') else bb!r}")

        out = args.out or (Path("figures") /
                           f"sweep_{args.tool}_{args.param}_x_{args.param2}.html")
        out.parent.mkdir(parents=True, exist_ok=True)
        build_heatmap(grid, args.tool, args.param, values,
                      args.param2, values2).write_html(out, include_plotlyjs="cdn")
        grid.to_csv(out.with_suffix(".csv"), index=False)
        print(f"wrote {out}\nwrote {out.with_suffix('.csv')}")
        return 0

    metrics, masks, outs = sweep(series, labels, args.tool, args.param,
                                 values, tolerance)

    if spec.kind == "detect" and "f1" in metrics:
        i = int(metrics["f1"].idxmax())
        print(f"\nbest point F1 {metrics['f1'][i]:.3f} at {args.param}={values[i]!r}")
        if metrics.get("event_f1", pd.Series(dtype=float)).notna().any():
            j = int(metrics["event_f1"].idxmax())
            print(f"best event F1 {metrics['event_f1'][j]:.3f} at "
                  f"{args.param}={values[j]!r}  "
                  f"(episodes hit {int(metrics['n_events_hit'][j])}/"
                  f"{int(metrics['n_events'][j])})")
        if spec.tolerance:
            print("  ^ this tool marks transitions, so trust the EVENT row")
    elif "rmse" in metrics and metrics["rmse"].notna().any():
        i = int(metrics["rmse"].idxmin())
        print(f"\nlowest RMSE {metrics['rmse'][i]:.4f} at {args.param}={values[i]!r} "
              f"(baseline {metrics['baseline_rmse'][i]:.4f}, "
              f"coverage {metrics['coverage'][i]:.0%})")

    if args.explore:
        out = args.out or Path("figures") / f"explore_{args.tool}_{args.param}.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        build_explorer(series, labels, metrics, masks, args.tool, args.param,
                       values, find_events(labels, spec.target)).write_html(
            out, include_plotlyjs="cdn")
        print(f"wrote {out}")
        return 0

    out = args.out or Path("figures") / f"sweep_{args.tool}_{args.param}.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_figure(series, labels, metrics, masks, outs,
                       args.tool, args.param, values)
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
