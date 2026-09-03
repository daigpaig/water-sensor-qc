"""Turbidity against rainfall, on one time axis, for a human to interrogate.

WHY THIS EXISTS. §7.7 makes rain the only evidence in this project that comes from
OUTSIDE the turbidity series, and it is what settles the hardest call the agent
makes: is this excursion real water or a sensor artifact? The agent gets that
evidence as three numbers per point (`precip_context`'s 0-1h / 1-3h / 3-12h lag
windows). A person cannot judge whether those numbers are being read sensibly from
three numbers at a time — you have to SEE the storm arrive and the turbidity
respond, then click into the point and read the same breakdown the agent read.

This page is that view. Two panels on one shared time axis:

  * turbidity, full resolution, with the §5 ground-truth anomalies as coloured
    markers by type (the same palette `visualize_injected.py` uses, so the two
    pages read the same way);
  * rainfall from ONE selected nearby station, as bars at the station's native
    cadence plus a 12h running total, which is the window `precip_context`
    actually sums over.

Click any turbidity point (or type a timestamp) and the panel reports the lag
windows for EVERY station, not just the plotted one. That cross-station column is
the point of the layout rather than a decoration: §7.7's whole caveat is that a
convective cell 5-15 km across can rain on one bucket and not its neighbour, so
where the stations disagree, that disagreement IS the evidence, and averaging it
away or showing one bucket would hide it.

THREE THINGS THIS PAGE IS CAREFUL ABOUT, each a trap §7.7 or §9.1 already paid for:

1. **"No data" never renders as "no rain."** That is the dangerous failure here --
   it reads as evidence FOR deleting a point. Each station carries explicit
   coverage intervals, and a timestamp outside them says "no coverage" with no
   inches figure at all.
2. **Absence of rain is reported as weak evidence, presence as strong.** The panel
   says so in words on every dry verdict, scaled by the station's distance.
3. **Timestamps are naive ISO strings, never Date objects, and window arithmetic
   runs on UTC-parsed milliseconds.** Plotly renders a Date in the VIEWER's
   timezone (§9.1), which silently shifts the axis away from the CSV; and parsing
   a naive string with `new Date()` uses local time, so a 12h window straddling a
   DST boundary would sum 11h or 13h of rain. Both are avoided explicitly.

CLI
---
    # Level-1 dataset for a gauge (the default level is 1):
    python -m src.workbench.precip_overlay 01467200
    python -m src.workbench.precip_overlay 02054550 --level 3

    # Or point straight at any datetime/value CSV (labels optional, found beside it):
    python -m src.workbench.precip_overlay data/raw/approved/01467200_turbidity_63680.csv \\
        --gauge 01467200 --out figures/precip/one.html --no-open
"""
from __future__ import annotations

import argparse
import json
import re
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from src.agent_tools.precipitation import LAG_WINDOWS, WET_INCHES
from src.datasets.inject import dataset_dir
from src.datasets.pull_usgs import DEFAULT_MAX_GAP
from src.workbench.visualize import insert_gap_breaks
from src.workbench.visualize_injected import ANOMALY_COLORS

DEFAULT_INJECTED_DIR = Path("data/injected")
DEFAULT_PRECIP_ROOT = Path("data/precip")
DEFAULT_OUTDIR = Path("figures/precip")

TURBIDITY_COLOR = "#2563eb"
RAIN_COLOR = "#0ea5e9"
RAIN_TOTAL_COLOR = "#1e3a8a"

# How far to look either side of a looked-up / clicked point when the page
# recentres on it. Expressed as a DURATION: §7.3 records that a tuned constant
# written in SAMPLES silently changes meaning when the cadence does.
DEFAULT_FOCUS = "12h"

# The 12h running total drawn over the rain bars. This is the same span
# `precip_context` sums for its "did it rain before this point" verdict
# (LAG_WINDOWS reaches back 12h), so the line shows at a glance what the agent's
# evidence would have said anywhere on the record.
RAIN_TOTAL_WINDOW = "12h"
# The running total is resampled before plotting -- it is a smooth envelope, so
# full resolution buys nothing but bytes. The BARS stay at native cadence.
RAIN_TOTAL_STEP = "1h"

# A station's record is treated as broken when consecutive observations are
# further apart than this multiple of its own modal step. Coverage is what
# separates "it did not rain" from "nothing is known here" (§7.7), so it is
# measured from the timestamps rather than assumed from the manifest's window.
COVERAGE_GAP_STEPS = 6.0


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_series(path: Path) -> pd.Series:
    """A ``datetime``/``value`` CSV as a sorted, de-duplicated float Series."""
    frame = pd.read_csv(path, usecols=["datetime", "value"], parse_dates=["datetime"])
    series = frame.set_index("datetime")["value"].sort_index()
    return series[~series.index.duplicated(keep="first")].astype(float)


def load_labels(path: Path) -> pd.DataFrame | None:
    """The §5 label file beside *path*, or None when the series is unlabelled.

    Labels live in the same directory as the series by contract (§5), found as
    ``<stem>_labels.csv`` -- never rebuilt from the directory layout by hand.
    """
    label_path = path.with_name(f"{path.stem}_labels.csv")
    if not label_path.exists():
        return None
    frame = pd.read_csv(label_path, parse_dates=["datetime"])
    return frame.set_index("datetime").sort_index()


def _modal_step_minutes(index: pd.DatetimeIndex) -> float:
    steps = pd.Series(index).diff().dropna()
    if steps.empty:
        return float("nan")
    return float(steps.mode().iloc[0].total_seconds() / 60.0)


def coverage_intervals(index: pd.DatetimeIndex,
                       gap_steps: float = COVERAGE_GAP_STEPS) -> list[list[str]]:
    """Contiguous stretches the station actually reported over, as ISO pairs.

    Anything outside these is "no coverage", which the page must never render as
    "no rain" (§7.7). A break is any step wider than *gap_steps* times the
    station's own modal step, so a 5-min met station and a 15-min tipping bucket
    are each judged against their own cadence rather than a shared constant.
    """
    if len(index) == 0:
        return []
    if len(index) == 1:
        return [[_iso(index[0]), _iso(index[0])]]
    step = _modal_step_minutes(index)
    tol = pd.Timedelta(minutes=step * gap_steps) if step == step else pd.Timedelta("1h")
    diffs = pd.Series(index).diff()
    breaks = np.flatnonzero((diffs > tol).to_numpy())
    starts = np.r_[0, breaks]
    ends = np.r_[breaks - 1, len(index) - 1]
    return [[_iso(index[s]), _iso(index[e])] for s, e in zip(starts, ends)]


def _iso(ts) -> str:
    """Naive ISO string -- the only timestamp form that reaches the page (§9.1)."""
    return pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def load_stations(gauge: str, root: Path = DEFAULT_PRECIP_ROOT) -> tuple[list[dict], dict | None]:
    """Every pulled rain station for *gauge*, nearest first, with its series.

    Returns ``([], None)`` when nothing has been pulled -- the caller renders that
    as a stated absence rather than an empty rain panel.
    """
    manifest_path = root / gauge / "manifest.json"
    if not manifest_path.exists():
        return [], None
    manifest = json.loads(manifest_path.read_text())
    stations: list[dict] = []
    for entry in manifest.get("stations", []):
        path = root / gauge / entry["file"]
        if not path.exists():
            continue
        frame = pd.read_csv(path, parse_dates=["datetime"]).sort_values("datetime")
        series = frame.set_index("datetime")["precip_in"].astype(float)
        series = series[~series.index.duplicated(keep="first")]
        stations.append({**entry, "series": series})
    stations.sort(key=lambda s: float(s["km"]))
    return stations, manifest


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------
def _station_payload(station: dict) -> dict:
    """One station as the page needs it: bars, running total, coverage, identity.

    Only NONZERO observations become bars. That is lossless for a bar chart and
    for the lag-window sums the panel computes (a zero adds nothing to either),
    and it is what makes full-resolution rain affordable: a tipping bucket
    reporting 76 inches in 0.01 in tips has ~7,600 nonzero rows out of ~210,000.
    Coverage is carried separately and comes from the FULL index, so dropping the
    zeros cannot be mistaken for the station going quiet.
    """
    series = station["series"]
    wet = series[series > 0]
    total = series.rolling(RAIN_TOTAL_WINDOW).sum().resample(RAIN_TOTAL_STEP).max()
    total = total.dropna()
    return {
        "site": str(station["site"]),
        "name": str(station.get("name", "")),
        "km": float(station["km"]),
        "note": str(station.get("note", "")),
        "dt_min": _modal_step_minutes(pd.DatetimeIndex(series.index)),
        "total_in": round(float(series.sum()), 2),
        "bar_x": [_iso(t) for t in wet.index],
        "bar_y": [round(float(v), 3) for v in wet.to_numpy()],
        "tot_x": [_iso(t) for t in total.index],
        "tot_y": [round(float(v), 3) for v in total.to_numpy()],
        "coverage": coverage_intervals(pd.DatetimeIndex(series.index)),
    }


def default_station(series: pd.Series, stations: list[dict]) -> int:
    """Which station the page should plot first: the best-covered, not the nearest.

    Nearest-first is right for `precip_context`, which falls through to the next
    station whenever the closest has no data at that timestamp. A plotted panel
    cannot fall through -- it shows one station -- so opening on the nearest is
    actively misleading where that station covers only part of the window. On
    01467200 the nearest bucket (19.9 km) starts 2024-10, so the page would open
    on eighteen months of blank rain panel beside a turbidity record full of
    storms, which reads as "it never rained" (§7.7's dangerous failure) rather
    than "this station was not yet reporting".

    Ties within a percentage point go to the nearer station, since equal coverage
    makes distance the only thing left to prefer on.
    """
    if not stations:
        return 0
    if len(series.index) == 0:
        return 0
    lo, hi = series.index.min(), series.index.max()
    span = (hi - lo).total_seconds()
    best, best_score = 0, -1.0
    for i, station in enumerate(stations):          # already sorted nearest-first
        covered = 0.0
        for start, end in coverage_intervals(pd.DatetimeIndex(station["series"].index)):
            a, b = max(pd.Timestamp(start), lo), min(pd.Timestamp(end), hi)
            if b > a:
                covered += (b - a).total_seconds()
        score = covered / span if span else 0.0
        if score > best_score + 0.01:               # strict: ties keep the nearer
            best, best_score = i, score
    return best


def build_payload(series: pd.Series, labels: pd.DataFrame | None, gauge: str,
                  title: str, stations: list[dict], manifest: dict | None,
                  max_gap: str = DEFAULT_MAX_GAP, focus: str = DEFAULT_FOCUS) -> dict:
    """Everything the page renders, as one JSON-serialisable dict.

    Assembled in Python rather than in JS so it is testable -- the same reason
    `provenance.py` generates its narrative here (§10.1).
    """
    line = insert_gap_breaks(series, max_gap)
    payload: dict = {
        "gauge": gauge,
        "title": title,
        "focus_minutes": pd.Timedelta(focus).total_seconds() / 60.0,
        "wet_inches": WET_INCHES,
        "lag_windows": [[label, lo, hi] for label, lo, hi in LAG_WINDOWS],
        "series_x": [_iso(t) for t in line.index],
        "series_y": [None if not np.isfinite(v) else round(float(v), 3)
                     for v in line.to_numpy()],
        "stations": [_station_payload(s) for s in stations],
        "default_station": default_station(series, stations),
        "manifest_note": (manifest or {}).get("note", ""),
        "anomalies": {},
        "anomaly_colors": dict(ANOMALY_COLORS),
    }
    if labels is not None:
        # Mark a gap at its true_value: the value itself is NaN and cannot be
        # plotted. A NATURAL gap has no true_value (§5), so it has nowhere to
        # draw and is simply absent from the markers rather than drawn at zero.
        aligned = labels.reindex(series.index)
        mark_y = series.where(series.notna(), aligned.get("true_value"))
        for atype in ANOMALY_COLORS:
            mask = (aligned.get("anomaly_type") == atype) & mark_y.notna()
            if not bool(mask.any()):
                continue
            sub = mark_y[mask]
            payload["anomalies"][atype] = {
                "x": [_iso(t) for t in sub.index],
                "y": [round(float(v), 3) for v in sub.to_numpy()],
            }
    return payload


# --------------------------------------------------------------------------
# html
# --------------------------------------------------------------------------
def _plotly_js() -> str:
    """Inline plotly.js so the page works offline, with no server and no CDN."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def build_html(payload: dict) -> str:
    return (_TEMPLATE
            .replace("/*PLOTLY_JS*/", _plotly_js())
            .replace("/*PAYLOAD*/", json.dumps(payload)))


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
_GAUGE_RE = re.compile(r"^\d{8,15}$")


def resolve_target(target: str, level: int, injected_dir: Path) -> tuple[Path, str]:
    """``01467200`` -> its level-*level* dataset; a path -> itself. Returns (path, gauge)."""
    if _GAUGE_RE.match(target):
        name = f"{target}_l{level}"
        path = dataset_dir(injected_dir, name) / f"{name}.csv"
        if not path.exists():
            raise SystemExit(f"No such dataset: {path}")
        return path, target
    path = Path(target)
    if not path.exists():
        raise SystemExit(f"No such file: {path}")
    gauge_match = re.match(r"(\d{8,15})", path.stem)
    return path, (gauge_match.group(1) if gauge_match else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target", help="A gauge id (e.g. 01467200) or a datetime/value CSV.")
    parser.add_argument("--level", type=int, default=1,
                        help="Contamination level when TARGET is a gauge id (default 1).")
    parser.add_argument("--gauge", default=None,
                        help="Override which gauge's rain to load (inferred from the stem).")
    parser.add_argument("--injected-dir", type=Path, default=DEFAULT_INJECTED_DIR)
    parser.add_argument("--precip-root", type=Path, default=DEFAULT_PRECIP_ROOT)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--max-gap", default=DEFAULT_MAX_GAP,
                        help=f"Break the turbidity line at gaps wider than this (default {DEFAULT_MAX_GAP}).")
    parser.add_argument("--focus", default=DEFAULT_FOCUS,
                        help=f"Half-width of the window a lookup recentres on (default {DEFAULT_FOCUS}).")
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    path, inferred = resolve_target(args.target, args.level, args.injected_dir)
    gauge = args.gauge or inferred
    series = load_series(path)
    labels = load_labels(path)
    stations, manifest = load_stations(gauge, args.precip_root) if gauge else ([], None)
    if not stations:
        print(f"  ! no precipitation pulled for gauge {gauge or '(unknown)'} — the rain "
              f"panel will say so rather than render empty.\n"
              f"    pull it with: python -m src.datasets.pull_precip {gauge}")

    payload = build_payload(series, labels, gauge, path.stem, stations, manifest,
                            max_gap=args.max_gap, focus=args.focus)
    out = args.out or (DEFAULT_OUTDIR / f"{path.stem}_precip.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(payload))

    size_mb = out.stat().st_size / 1e6
    print(f"{path.stem}: {len(series):,} turbidity points, "
          f"{len(stations)} rain station(s) -> {out} ({size_mb:.1f} MB)")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


_TEMPLATE = r"""<!doctype html>
<meta charset="utf-8">
<title>Turbidity vs rainfall</title>
<script>/*PLOTLY_JS*/</script>
<style>
  :root { --ink:#0f172a; --muted:#64748b; --line:#e2e8f0; --bg:#f8fafc; }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;
         color:var(--ink); background:#fff; }
  header { padding:10px 16px; border-bottom:1px solid var(--line); display:flex;
           gap:14px; align-items:center; flex-wrap:wrap; background:var(--bg); }
  h1 { font-size:15px; margin:0; font-weight:600; }
  .grow { flex:1; }
  label { font-size:12px; color:var(--muted); }
  input, select, button { font:inherit; padding:5px 8px; border:1px solid #cbd5e1;
                          border-radius:6px; background:#fff; color:var(--ink); }
  input { width:172px; font-variant-numeric:tabular-nums; }
  select { max-width:210px; }
  button { cursor:pointer; }
  button:hover { background:var(--bg); }
  button.step { width:32px; padding:5px 0; }
  #wrap { display:flex; align-items:stretch; height:calc(100vh - 53px); min-height:520px; }
  #plot { flex:1; min-width:0; height:100%; }
  #side { width:370px; flex:none; border-left:1px solid var(--line); overflow-y:auto;
          padding:14px 16px; }
  @media (max-width:1100px) { #side { width:300px; } }
  #side h2 { font-size:13px; margin:0 0 2px; font-weight:600; }
  .stamp { font-variant-numeric:tabular-nums; font-size:17px; font-weight:600; }
  .sub { color:var(--muted); font-size:12px; }
  table { border-collapse:collapse; width:100%; margin-top:10px; font-size:12px; }
  th, td { padding:4px 6px; border-bottom:1px solid var(--line); text-align:right;
           font-variant-numeric:tabular-nums; }
  th:first-child, td:first-child { text-align:left; font-variant-numeric:normal; }
  thead th { color:var(--muted); font-weight:600; }
  tbody tr:last-child td { border-bottom:none; font-weight:600; }
  td.wet { color:#0369a1; font-weight:600; }
  td.none { color:#cbd5e1; }
  .verdict { margin-top:12px; padding:9px 11px; border-radius:7px; font-size:12.5px; }
  .verdict.wet { background:#ecfeff; border:1px solid #a5f3fc; }
  .verdict.dry { background:#fffbeb; border:1px solid #fde68a; }
  .verdict.unknown { background:#f1f5f9; border:1px solid #cbd5e1; }
  .caveat { margin-top:10px; font-size:11.5px; color:var(--muted); border-top:1px solid var(--line);
            padding-top:8px; }
  .chip { display:inline-block; padding:1px 7px; border-radius:99px; font-size:11px;
          color:#fff; font-weight:600; }
  .empty { color:var(--muted); font-size:13px; }
  kbd { font:11px ui-monospace,Menlo,monospace; background:var(--bg); border:1px solid #cbd5e1;
        border-bottom-width:2px; border-radius:4px; padding:0 4px; }
</style>

<header>
  <h1 id="title"></h1>
  <span class="grow"></span>
  <label for="stn">rain station</label>
  <select id="stn"></select>
  <label for="ts">timestamp</label>
  <input id="ts" placeholder="2023-09-15 11:40" autocomplete="off">
  <button id="go">Go</button>
  <button class="step" id="prev" title="previous sample (←)">&larr;</button>
  <button class="step" id="next" title="next sample (→)">&rarr;</button>
  <button id="reset">Reset zoom</button>
</header>

<div id="wrap">
  <div id="plot"></div>
  <div id="side"><p class="empty">Click a turbidity point, or type a timestamp above,
    to see the rainfall that led up to it.</p></div>
</div>

<script>
const P = /*PAYLOAD*/;
const gd = document.getElementById('plot');

/* Naive "YYYY-MM-DD HH:MM:SS" -> epoch ms, parsed as UTC.
   `new Date(s)` on a naive string uses the VIEWER's local time, so a 12h window
   straddling a DST boundary would sum 11h or 13h of rain. Date.UTC has no such
   discontinuity, and since every timestamp on this page goes through here, the
   offset is uniform and differences are exact. */
function ms(s) {
  return Date.UTC(+s.slice(0,4), +s.slice(5,7)-1, +s.slice(8,10),
                  +s.slice(11,13)||0, +s.slice(14,16)||0, +s.slice(17,19)||0);
}
function pad(n) { return String(n).padStart(2, '0'); }
function iso(m) {
  const d = new Date(m);
  return d.getUTCFullYear() + '-' + pad(d.getUTCMonth()+1) + '-' + pad(d.getUTCDate()) +
         ' ' + pad(d.getUTCHours()) + ':' + pad(d.getUTCMinutes()) + ':' + pad(d.getUTCSeconds());
}
const HOUR = 3600e3;

/* ---- precompute numeric axes once ---- */
const SX = P.series_x.map(ms);
P.stations.forEach(st => {
  st._bar = st.bar_x.map(ms);
  st._cov = st.coverage.map(iv => [ms(iv[0]), ms(iv[1])]);
});

function covered(st, m) { return st._cov.some(iv => m >= iv[0] && m <= iv[1]); }

/* Inches over [m - hi h, m - lo h]. Only nonzero observations are carried, which
   is exact for a sum. Returns null when the station has no coverage there --
   "unknown" and "zero" must never render the same way (CLAUDE.md §7.7). */
function rainOver(st, m, loH, hiH) {
  const a = m - hiH * HOUR, b = m - loH * HOUR;
  if (!covered(st, a) && !covered(st, b)) return null;
  let sum = 0;
  for (let i = 0; i < st._bar.length; i++) {
    const t = st._bar[i];
    if (t < a) continue;
    if (t > b) break;
    sum += st.bar_y[i];
  }
  return sum;
}
function rainAfter(st, m, hours) {
  if (!covered(st, m)) return null;
  let sum = 0;
  for (let i = 0; i < st._bar.length; i++) {
    const t = st._bar[i];
    if (t < m) continue;
    if (t > m + hours * HOUR) break;
    sum += st.bar_y[i];
  }
  return sum;
}

/* ---- figure ---- */
const traces = [{
  x: P.series_x, y: P.series_y, type: 'scattergl', mode: 'lines',
  line: {color: '#2563eb', width: 1}, name: 'turbidity', connectgaps: false,
  hovertemplate: '%{x}<br>%{y:.2f} FNU<extra></extra>',
}];
const ANOM_START = traces.length;
const anomKeys = Object.keys(P.anomalies);
anomKeys.forEach(k => {
  traces.push({
    /* `scatter`, not `scattergl`: these are a few hundred points, and every
       scattergl trace shares one regl context, so restyling ANY of them pays to
       rebuild the 210k-point turbidity buffers. Only the main line earns WebGL. */
    x: P.anomalies[k].x, y: P.anomalies[k].y, type: 'scatter', mode: 'markers',
    marker: {color: P.anomaly_colors[k], size: 5}, name: k,
    hovertemplate: '%{x}<br>%{y:.2f} FNU<br>' + k + '<extra></extra>',
  });
});
/* The selection marker rides on the turbidity panel and is restyled in place
   rather than added and removed, so trace indices stay stable. */
const DEF_STN = P.default_station || 0;
const SEL = traces.length;
traces.push({
  /* SVG for the same reason, and it matters most here: this trace is restyled on
     every lookup and every arrow-key step. Measured on the WebGL path that
     restyle cost 1.9 s -- for one point. */
  x: [], y: [], type: 'scatter', mode: 'markers', name: 'selected',
  marker: {color: '#0f172a', size: 13, symbol: 'circle-open', line: {width: 2.5}},
  hoverinfo: 'skip', showlegend: false,
});
/* ONE rain trace and ONE running-total trace, whose data the selector swaps.
   The obvious alternative -- a trace per station toggled with `visible` -- was
   measured at 42 s for the first switch and ~2 s after, because toggling
   visibility makes Plotly recalculate the whole figure INCLUDING the 210k-point
   turbidity scattergl, and the first switch additionally has to cold-build the
   traces that had never been drawn. Swapping the arrays on two existing traces
   keeps the trace count fixed, so nothing is ever cold, and it lightens the
   first paint too. */
const BAR = traces.length;
traces.push({
  x: [], y: [], type: 'bar', name: 'rain', xaxis: 'x2', yaxis: 'y2',
  marker: {color: '#0ea5e9'},
  hovertemplate: '%{x}<br>%{y:.2f} in<extra></extra>',
});
const TOT = traces.length;
traces.push({
  x: [], y: [], type: 'scattergl', mode: 'lines', xaxis: 'x2', yaxis: 'y3',
  name: 'running 12h total', line: {color: '#1e3a8a', width: 1.2}, connectgaps: true,
  hovertemplate: '%{x}<br>%{y:.2f} in / 12h<extra></extra>',
});
if (P.stations.length) {
  const d = P.stations[DEF_STN];
  traces[BAR].x = d.bar_x; traces[BAR].y = d.bar_y;
  traces[TOT].x = d.tot_x; traces[TOT].y = d.tot_y;
}

const layout = {
  template: 'plotly_white', autosize: true, margin: {l: 58, r: 58, t: 26, b: 38}, hovermode: 'closest',
  bargap: 0, dragmode: 'zoom', showlegend: true,
  legend: {orientation: 'h', y: 1.06, x: 0},
  xaxis:  {domain: [0, 1], anchor: 'y',  matches: 'x2', showticklabels: false},
  yaxis:  {domain: [0.42, 1], title: {text: 'turbidity (FNU)'}},
  xaxis2: {domain: [0, 1], anchor: 'y2'},
  yaxis2: {domain: [0, 0.34], title: {text: 'rain (in)'}, rangemode: 'tozero'},
  yaxis3: {domain: [0, 0.34], overlaying: 'y2', side: 'right', rangemode: 'tozero',
           title: {text: 'running 12h (in)'}, showgrid: false},
};
document.getElementById('title').textContent =
  P.title + '  ·  turbidity vs rainfall';

/* Plotly measures the div at newPlot time. `#plot` is a flex child, so on a cold
   load that measurement can land before the flex layout has settled and the
   figure is drawn at a fraction of the width it should have. Resize once the
   plot exists, and again whenever the container changes. */
let _fitW = 0, _fitH = 0;
function fit() {
  const box = gd.getBoundingClientRect();
  const w = Math.round(box.width), h = Math.round(box.height);
  if (w < 50 || h < 50) return;
  if (w === _fitW && h === _fitH) return;   /* the observer must not feed itself */
  _fitW = w; _fitH = h;
  Plotly.relayout(gd, {width: w, height: h});
}
new ResizeObserver(fit).observe(document.getElementById('wrap'));

Plotly.newPlot(gd, traces, layout, {responsive: true, scrollZoom: true}).then(() => {
  fit();
  /* `gd.on(...)` does not exist until Plotly has plotted into the div -- wiring
     it before the first draw throws and takes the rest of init down with it,
     keyboard handlers included (CLAUDE.md §9.1). */
  gd.on('plotly_click', ev => {
    const pt = ev.points && ev.points[0];
    if (!pt || pt.data.xaxis === 'x2') return;   /* rain panel clicks are not points */
    select(ms(pt.x));
  });
  gd.on('plotly_relayout', autoscaleY);
  autoscaleY({'xaxis.range': true});
});

/* Plotly does not rescale y when you zoom x, and a two-year turbidity record has
   storm peaks two orders above baseline -- without this every zoomed view is a
   flat line at the bottom of the panel. Recompute each panel's y from only what
   is in view. */
let _guard = false;

function autoscaleY(ed) {
  if (_guard) return;
  const keys = Object.keys(ed || {});
  if (!keys.some(k => k.indexOf('xaxis') === 0)) return;
  const upd = rangeUpdates();
  if (!Object.keys(upd).length) return;
  _guard = true;
  Plotly.relayout(gd, upd).then(() => { _guard = false; });
}

/* y-ranges for a given x window. Callers that are ABOUT to move the x axis pass
   the window they are moving to, so the data change, the x range and both y
   ranges go out as one render instead of three -- each render pays the full cost
   of recalculating the 210k-point turbidity trace. */
function rangeUpdates(x0, x1) {
  if (x0 === undefined) {
    const r = gd._fullLayout.xaxis2.range;
    if (!r) return {};
    x0 = ms(String(r[0]).replace('T', ' '));
    x1 = ms(String(r[1]).replace('T', ' '));
  }
  const upd = {};
  if (x1 < x0) { const t = x0; x0 = x1; x1 = t; }
  /* turbidity */
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < SX.length; i++) {
    if (SX[i] < x0 || SX[i] > x1) continue;
    const v = P.series_y[i];
    if (v === null) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (lo !== Infinity) {
    const pad = (hi > lo) ? (hi - lo) * 0.06 : (Math.abs(hi) * 0.06 || 1);
    upd['yaxis.range'] = [lo - pad, hi + pad];
  }
  /* rain: bars and total both start at zero, so only the top moves */
  const st = P.stations[stnIdx()];
  if (st) {
    let bmax = 0, tmax = 0;
    for (let i = 0; i < st._bar.length; i++) {
      const t = st._bar[i];
      if (t >= x0 && t <= x1 && st.bar_y[i] > bmax) bmax = st.bar_y[i];
    }
    for (let i = 0; i < st.tot_x.length; i++) {
      const t = ms(st.tot_x[i]);
      if (t >= x0 && t <= x1 && st.tot_y[i] > tmax) tmax = st.tot_y[i];
    }
    upd['yaxis2.range'] = [0, (bmax || 0.05) * 1.15];
    upd['yaxis3.range'] = [0, (tmax || 0.1) * 1.15];
  }
  return upd;
}

/* ---- station selector ---- */
const stnSel = document.getElementById('stn');
P.stations.forEach((st, i) => {
  const o = document.createElement('option');
  o.value = i;
  o.textContent = st.km.toFixed(1) + ' km · ' + st.site;
  o.title = st.name + (st.note ? ' — ' + st.note : '');
  stnSel.appendChild(o);
});
if (P.stations.length) stnSel.value = String(DEF_STN);
if (!P.stations.length) {
  const o = document.createElement('option');
  o.textContent = 'none pulled';
  stnSel.appendChild(o);
  stnSel.disabled = true;
}
function stnIdx() { return P.stations.length ? +stnSel.value : -1; }
stnSel.addEventListener('change', () => {
  const st = P.stations[stnIdx()];
  if (!st) return;
  /* Data and y-ranges go in ONE call: three separate renders each pay the full
     cost of recalculating the 210k-point turbidity trace. */
  _guard = true;
  Plotly.update(gd, {x: [st.bar_x, st.tot_x], y: [st.bar_y, st.tot_y]},
                rangeUpdates(), [BAR, TOT]).then(() => {
    _guard = false;
    if (selMs !== null) render(selMs);
  });
});

/* ---- selection + lookup ---- */
let selMs = null;

function nearestIdx(m) {
  let lo = 0, hi = SX.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (SX[mid] < m) lo = mid + 1; else hi = mid;
  }
  /* SX carries NaN sentinels inserted to break the line at real gaps; step off
     them so a lookup lands on an actual reading. */
  let best = lo, bestD = Infinity;
  for (let i = Math.max(0, lo - 3); i <= Math.min(SX.length - 1, lo + 3); i++) {
    if (P.series_y[i] === null) continue;
    const d = Math.abs(SX[i] - m);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

function select(m, recentre) {
  const i = nearestIdx(m);
  selMs = SX[i];
  let lay = {};
  if (recentre) {
    const half = P.focus_minutes * 60e3, a = selMs - half, b = selMs + half;
    lay = rangeUpdates(a, b);
    lay['xaxis2.range'] = [iso(a), iso(b)];
  }
  _guard = true;
  Plotly.update(gd, {x: [[P.series_x[i]]], y: [[P.series_y[i]]]}, lay, [SEL])
    .then(() => { _guard = false; });
  render(selMs);
}

function stepBy(k) {
  if (selMs === null) return;
  let i = nearestIdx(selMs) + k;
  while (i >= 0 && i < SX.length && P.series_y[i] === null) i += k;
  if (i < 0 || i >= SX.length) return;
  select(SX[i], true);
}

/* Accepts a partial stamp: "2023-09-15", "2023-09-15 11", "2023-09-15 11:40".
   Missing fields default to the start of the coarser unit. */
function parseStamp(text) {
  const t = text.trim().replace('T', ' ');
  const m = t.match(/^(\d{4})-(\d{2})-(\d{2})(?:[ ](\d{1,2})(?::(\d{2}))?(?::(\d{2}))?)?$/);
  if (!m) return null;
  return Date.UTC(+m[1], +m[2]-1, +m[3], +(m[4]||0), +(m[5]||0), +(m[6]||0));
}

const tsInput = document.getElementById('ts');
function doLookup() {
  const m = parseStamp(tsInput.value);
  if (m === null) {
    tsInput.style.borderColor = '#dc2626';
    setTimeout(() => { tsInput.style.borderColor = ''; }, 900);
    return;
  }
  select(m, true);
}
document.getElementById('go').addEventListener('click', doLookup);
tsInput.addEventListener('keydown', e => { if (e.key === 'Enter') doLookup(); });
document.getElementById('prev').addEventListener('click', () => stepBy(-1));
document.getElementById('next').addEventListener('click', () => stepBy(1));
document.getElementById('reset').addEventListener('click', () => {
  _guard = true;
  Plotly.relayout(gd, rangeUpdates(SX[0], SX[SX.length - 1]))
    .then(() => { _guard = false; return Plotly.relayout(gd, {'xaxis2.autorange': true}); });
});
document.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'ArrowLeft') { stepBy(-1); e.preventDefault(); }
  if (e.key === 'ArrowRight') { stepBy(1); e.preventDefault(); }
  if (e.key === '/') { tsInput.focus(); tsInput.select(); e.preventDefault(); }
});

/* ---- detail panel ---- */
function fmtIn(v) {
  if (v === null) return ['—', 'none', 'no coverage'];
  if (v >= P.wet_inches) return [v.toFixed(2), 'wet', ''];
  return [v.toFixed(2), '', ''];
}

function render(m) {
  const side = document.getElementById('side');
  const i = nearestIdx(m);
  const val = P.series_y[i];
  let html = '<div class="stamp">' + P.series_x[i] + '</div>' +
             '<div class="sub">turbidity ' +
             (val === null ? '<b>missing</b>' : '<b>' + val.toFixed(2) + '</b> FNU') + '</div>';

  const label = anomAt(P.series_x[i]);
  if (label) {
    html += '<div style="margin-top:6px"><span class="chip" style="background:' +
            P.anomaly_colors[label] + '">' + label + '</span> ' +
            '<span class="sub">ground truth</span></div>';
  }

  if (!P.stations.length) {
    html += '<div class="verdict unknown"><b>No precipitation data for gauge ' +
            (P.gauge || '(unknown)') + '.</b> This is NOT the same as “it did not ' +
            'rain” — nothing is known either way. Pull it with ' +
            '<code>python -m src.datasets.pull_precip ' + (P.gauge || '&lt;gauge&gt;') +
            '</code>.</div>';
    side.innerHTML = html;
    return;
  }

  /* Every station gets a column, not just the plotted one. Where they disagree,
     that IS the evidence -- a convective cell 5-15 km across can rain on one
     bucket and not its neighbour (CLAUDE.md §7.7). */
  html += '<table><thead><tr><th>before</th>';
  P.stations.forEach((st, j) => {
    html += '<th' + (j === stnIdx() ? ' style="color:#0f172a"' : '') + '>' +
            st.km.toFixed(1) + ' km</th>';
  });
  html += '</tr></thead><tbody>';
  P.lag_windows.forEach(w => {
    html += '<tr><td>' + w[0] + '</td>';
    P.stations.forEach(st => {
      const f = fmtIn(rainOver(st, m, w[1], w[2]));
      html += '<td class="' + f[1] + '" title="' + f[2] + '">' + f[0] + '</td>';
    });
    html += '</tr>';
  });
  html += '<tr><td>after 0–3h</td>';
  P.stations.forEach(st => {
    const f = fmtIn(rainAfter(st, m, 3));
    html += '<td class="' + f[1] + '" title="' + f[2] + '">' + f[0] + '</td>';
  });
  html += '</tr><tr><td>total 12h before</td>';
  P.stations.forEach(st => {
    const f = fmtIn(rainOver(st, m, 0, 12));
    html += '<td class="' + f[1] + '">' + f[0] + '</td>';
  });
  html += '</tr></tbody></table>';

  const st = P.stations[stnIdx()];
  const tot = rainOver(st, m, 0, 12);
  const anyWet = P.stations.some(s => { const v = rainOver(s, m, 0, 12);
                                        return v !== null && v >= P.wet_inches; });
  if (tot === null) {
    html += '<div class="verdict unknown"><b>' + st.site + ' has no data here.</b> ' +
            'Nothing is known about rain at this timestamp from this station — that ' +
            'is not the same as no rain, and licenses nothing either way.</div>';
  } else if (tot >= P.wet_inches) {
    html += '<div class="verdict wet"><b>It rained</b> — ' + tot.toFixed(2) +
            ' in over the 12h before this point, ' + st.km.toFixed(1) + ' km away. Rain ' +
            'washes sediment in, so an excursion here has a physical cause and is more ' +
            'likely <b>real water</b> than an artifact. Presence of rain is the strong ' +
            'direction of this evidence.</div>';
  } else {
    html += '<div class="verdict dry"><b>No rain recorded</b> — ' + tot.toFixed(2) +
            ' in over the 12h before, at a station <b>' + st.km.toFixed(1) + ' km away</b>. ' +
            'This is <b>weak</b> evidence: a summer convective cell is often 5–15 km ' +
            'across and can rain on the catchment without reaching this bucket. Absence ' +
            'licenses nothing on its own.' +
            (anyWet ? ' <b>Another station on this page did record rain here</b> — ' +
                      'check the columns above.' : '') + '</div>';
  }

  html += '<div class="caveat"><b>' + st.site + '</b> — ' + st.name + '<br>' +
          st.km.toFixed(1) + ' km away' +
          (st.dt_min ? ', ' + st.dt_min.toFixed(0) + '-min cadence' : '') +
          ', ' + st.total_in.toFixed(0) + ' in over the record.' +
          (st.note ? '<br>' + st.note : '') +
          (P.manifest_note ? '<br><br>' + P.manifest_note : '') +
          '<br><br>Rain leads turbidity by an amount that depends on the catchment, so ' +
          'the windows above are reported rather than reduced to one lag. ' +
          '&ldquo;Wet&rdquo; is ≥ ' + P.wet_inches + ' in — one tip of a bucket ' +
          'is indistinguishable from noise.' +
          '<br><br><kbd>←</kbd> <kbd>→</kbd> step samples · <kbd>/</kbd> ' +
          'focus the timestamp box.</div>';
  side.innerHTML = html;
}

/* ground-truth type at an exact stamp, if any */
const ANOM_AT = {};
anomKeys.forEach(k => P.anomalies[k].x.forEach(x => { ANOM_AT[x] = k; }));
function anomAt(x) { return ANOM_AT[x] || null; }
</script>
"""

if __name__ == "__main__":
    raise SystemExit(main())
