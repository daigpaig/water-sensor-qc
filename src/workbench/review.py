"""Build a keyboard-driven page for reviewing proposed anomalies, and merge the result.

``src.workbench.candidates`` proposes segments; this module walks a human through
them one at a time and writes the verdicts back out. The page is a single
self-contained HTML file — no server, no Streamlit — so it opens straight from
disk and keeps working offline. Progress is mirrored into ``localStorage`` after
every keystroke, so closing the tab mid-review loses nothing.

Keys (also shown in the page's own help panel):

    A / 1   anomaly        N / 2   normal        U / 3   unsure
    ->      next           <-      previous      Z   undo last decision
    F       cycle zoom     R       whole span    E   export decisions CSV

Deciding auto-advances, so a straight run through is one keypress per candidate.

A detector marks a *window*; sometimes only one sample in it is the anomaly.
**Clicking a point narrows the label to that sample**, and the export carries the
narrowed ``start``/``end`` so :func:`~src.workbench.candidates.merge_decisions`
labels only what was endorsed. A decision may narrow a proposal, never extend it.

CLI
---
    # propose candidates and open the review page
    python -m src.workbench.review detect data/raw/provisional/06818000_turbidity_63680_provisional.csv

    # be pickier / more paranoid
    python -m src.workbench.review detect <csv> --jumps-sigmas 32 --max-per-type 25

    # fold the page's exported CSV back into a §5 labels file
    python -m src.workbench.review merge <csv> ~/Downloads/<name>_decisions.csv

Artefacts land in ``data/review/`` (candidates and merged labels) and
``figures/`` (the page). ``merge`` writes ``<name>_review_labels.csv`` in the §5
labels contract, with ``source=natural`` and an empty ``true_value`` (these
anomalies were already in the record, so no clean value is known — see §5 on why
that means they count for detection but not for imputation scoring).

Only spike, plateau and level_shift are reviewed. **Gaps never appear in the
queue**: whether a value is missing is not a judgement call, and ``merge`` labels
every missing run from the data itself (§5). See ``src.workbench.candidates``.
"""
from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from src.inspect_data import ContractError
from src.workbench.candidates import (
    TYPE_COLORS,
    LABEL_TYPES,
    REVIEW_TYPES,
    CandidateSet,
    DetectConfig,
    find_candidates,
    merge_decisions,
)

DEFAULT_PAGE_DIR = Path("figures")
#: Review artefacts live here, not beside the series: the raw subdirectories are
#: globbed by name elsewhere in the project (`data/raw/approved/*.csv` feeds
#: `src.datasets.inject`), and a stray `*_candidates.csv` there breaks those lookups.
DEFAULT_DATA_DIR = Path("data/review")


# --------------------------------------------------------------------------- payload
def build_payload(result: CandidateSet) -> dict:
    """JSON payload embedded in the page.

    The series sits on a regular grid, so timestamps are sent as a start epoch
    plus a step rather than 50k ISO strings — the page reconstructs them. Values
    carry ``None`` for missing samples so Plotly breaks the line at real gaps.
    """
    series = result.series
    values = [None if not np.isfinite(v) else round(float(v), 6) for v in series.to_numpy()]
    t0 = int(pd.Timestamp(series.index[0]).value // 1_000_000)  # ns -> ms
    step_ms = int(round(result.step_minutes * 60_000))

    candidates = [
        {
            "id": c.candidate_id,
            "type": c.anomaly_type,
            "detectors": list(c.detectors),
            "i0": c.start_pos,
            "i1": c.end_pos,
            "start": c.start.isoformat(sep=" "),
            "end": c.end.isoformat(sep=" "),
            "n_rows": c.n_rows,
            "hours": round(c.duration_hours, 2),
            "score": round(c.score, 3),
            "v_min": c.v_min,
            "v_max": c.v_max,
            "note": c.note,
        }
        for c in result.candidates
    ]

    return {
        "name": result.path.stem,
        "file": result.path.name,
        "t0": t0,
        "step_ms": step_ms,
        "n": len(series),
        "values": values,
        "candidates": candidates,
        "type_order": list(REVIEW_TYPES),
        "type_colors": TYPE_COLORS,
        "level_scale": round(result.level_scale, 6),
        "step_scale": round(result.step_scale, 6),
        "truncated": result.truncated,
        "errors": result.errors,
    }


def _plotly_js() -> str:
    """Inline plotly.js so the page works with no network and no CDN."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def build_review_html(result: CandidateSet) -> str:
    """Render the self-contained review page for a :class:`CandidateSet`."""
    payload = json.dumps(build_payload(result), allow_nan=False)
    # Only sequence that could end the host <script> element early.
    payload = payload.replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("/*PLOTLY_JS*/", _plotly_js())
        .replace("/*PAYLOAD*/", payload)
        .replace("__TITLE__", f"Review — {result.path.stem}")
    )


# --------------------------------------------------------------------------- template
# Kept as one string with sentinel comments rather than an f-string: the CSS and
# JS below are full of braces, and escaping every one of them would make this
# unreadable and unmaintainable.
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f8fafc; --panel: #ffffff; --ink: #0f172a; --muted: #64748b;
    --line: #e2e8f0; --anomaly: #dc2626; --normal: #059669; --unsure: #d97706;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; background: var(--bg); color: var(--ink);
    font: 14px/1.5 -apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif;
  }
  header {
    display: flex; align-items: baseline; gap: 16px; flex-wrap: wrap;
    padding: 12px 20px; background: var(--panel); border-bottom: 1px solid var(--line);
    position: sticky; top: 0; z-index: 10;
  }
  header h1 { font-size: 15px; margin: 0; font-weight: 600; }
  header .spacer { flex: 1; }
  .tally { display: flex; gap: 14px; font-variant-numeric: tabular-nums; }
  .tally b { font-weight: 600; }
  .tally .a { color: var(--anomaly); } .tally .n { color: var(--normal); }
  .tally .u { color: var(--unsure); } .tally .p { color: var(--muted); }
  #bar { height: 4px; background: var(--line); width: 100%; }
  #bar > div { height: 100%; width: 0; background: var(--ink); transition: width .15s; }

  main { display: grid; grid-template-columns: 1fr 300px; gap: 16px; padding: 16px 20px; }
  @media (max-width: 900px) { main { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px;
  }
  #main-plot { height: 380px; }
  #overview-plot { height: 110px; }

  dl.meta { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 4px 12px; }
  dl.meta dt { color: var(--muted); }
  dl.meta dd { margin: 0; font-variant-numeric: tabular-nums; word-break: break-word; }
  .pill {
    display: inline-block; padding: 1px 8px; border-radius: 999px;
    color: #fff; font-size: 12px; font-weight: 600;
  }

  .actions { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 20px 20px; }
  button {
    font: inherit; font-weight: 600; padding: 10px 16px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    cursor: pointer;
  }
  button:hover { border-color: var(--muted); }
  button kbd {
    font: inherit; font-size: 11px; opacity: .65; margin-left: 6px;
    border: 1px solid currentColor; border-radius: 4px; padding: 0 4px;
  }
  button.anomaly { background: var(--anomaly); border-color: var(--anomaly); color: #fff; }
  button.normal { background: var(--normal); border-color: var(--normal); color: #fff; }
  button.unsure { background: var(--unsure); border-color: var(--unsure); color: #fff; }
  .actions .spacer { flex: 1; }

  .verdict { font-weight: 600; }
  .verdict.anomaly { color: var(--anomaly); } .verdict.normal { color: var(--normal); }
  .verdict.unsure { color: var(--unsure); } .verdict.pending { color: var(--muted); }

  select, label.ctl { font: inherit; }
  .ctl { color: var(--muted); display: flex; align-items: center; gap: 6px; }
  .hint { color: var(--muted); font-size: 12px; padding: 0 20px 24px; }
  .hint code {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 4px; padding: 1px 5px;
  }
  .warn { color: var(--anomaly); }
</style>
</head>
<body>

<header>
  <h1 id="title"></h1>
  <label class="ctl">type
    <select id="filter"><option value="">all</option></select>
  </label>
  <span class="spacer"></span>
  <span class="tally">
    <span class="a">anomaly <b id="t-anomaly">0</b></span>
    <span class="n">normal <b id="t-normal">0</b></span>
    <span class="u">unsure <b id="t-unsure">0</b></span>
    <span class="p">left <b id="t-pending">0</b></span>
  </span>
</header>
<div id="bar"><div></div></div>

<main>
  <div class="card"><div id="main-plot"></div></div>
  <div class="card">
    <dl class="meta">
      <dt>candidate</dt><dd id="m-pos"></dd>
      <dt>type</dt><dd><span class="pill" id="m-type"></span></dd>
      <dt>verdict</dt><dd class="verdict" id="m-verdict"></dd>
      <dt>start</dt><dd id="m-start"></dd>
      <dt>end</dt><dd id="m-end"></dd>
      <dt>rows</dt><dd id="m-rows"></dd>
      <dt>duration</dt><dd id="m-hours"></dd>
      <dt>values</dt><dd id="m-values"></dd>
      <dt>severity</dt><dd id="m-score"></dd>
      <dt>found by</dt><dd id="m-detectors"></dd>
    </dl>
    <p id="m-note" class="hint" style="padding:8px 0 0"></p>
  </div>
</main>

<div class="actions">
  <button class="anomaly" data-decide="anomaly">Anomaly<kbd>A</kbd></button>
  <button class="normal" data-decide="normal">Normal<kbd>N</kbd></button>
  <button class="unsure" data-decide="unsure">Unsure<kbd>U</kbd></button>
  <button data-nav="-1">Prev<kbd>&larr;</kbd></button>
  <button data-nav="1">Next<kbd>&rarr;</kbd></button>
  <button id="undo">Undo<kbd>Z</kbd></button>
  <button id="reset">Whole span<kbd>R</kbd></button>
  <span class="spacer"></span>
  <button id="context">Zoom: x2<kbd>F</kbd></button>
  <button id="export">Export CSV<kbd>E</kbd></button>
</div>

<div class="card" style="margin: 0 20px 16px"><div id="overview-plot"></div></div>

<p class="hint">
  <b>Click any point</b> to label just that sample instead of the whole highlighted span —
  useful when a detector's window is wider than the actual anomaly. <code>R</code> restores
  the full span. Decisions save to this browser automatically
  (<code>localStorage</code>) — you can close the tab and come back. Press <code>E</code> to
  download the CSV, then run
  <code>python -m src.workbench.review merge&lt;series.csv&gt; &lt;decisions.csv&gt;</code>.
  <span id="warnings"></span>
</p>

<script>/*PLOTLY_JS*/</script>
<script id="payload" type="application/json">/*PAYLOAD*/</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("payload").textContent);
  const KEY = "wsqc-review:" + D.name + ":" + D.candidates.length;
  // Padding around a candidate is a MULTIPLE of its own length, not a fixed
  // window: a 15-minute spike and a 3-day level shift both have to be judged here,
  // and any single absolute span makes one of them unreadable. The floor keeps
  // one-sample candidates from being framed too tightly to judge.
  const ZOOMS = [
    { label: "x0.5", mult: 0.5 },
    { label: "x2", mult: 2 },
    { label: "x6", mult: 6 },
    { label: "x20", mult: 20 },
    { label: "all", mult: Infinity },
  ];
  const MIN_PAD_MS = 2 * 3600e3;

  // ---- state -------------------------------------------------------------
  let decisions = {};   // id -> "anomaly" | "normal" | "unsure"
  let spans = {};       // id -> [i0, i1)  — only when narrowed from the proposal
  try {
    const raw = JSON.parse(localStorage.getItem(KEY)) || {};
    decisions = raw.decisions || {};
    spans = raw.spans || {};
  } catch (e) { decisions = {}; spans = {}; }
  let idx = 0;
  let ctxI = 1;
  let filter = "";
  const history = [];

  const at = (i) => D.t0 + i * D.step_ms;
  const save = () => {
    try { localStorage.setItem(KEY, JSON.stringify({ decisions, spans })); } catch (e) {}
  };
  const view = () => D.candidates.filter((c) => !filter || c.type === filter);
  const cur = () => view()[idx];
  // What will actually be labelled: the proposal, unless narrowed by clicking.
  const spanOf = (c) => spans[c.id] || [c.i0, c.i1];
  // "YYYY-MM-DD HH:MM:SS" in the series' own (naive) clock, matching the CSV.
  const fmt = (ms) => new Date(ms).toISOString().slice(0, 19).replace("T", " ");

  // ---- plots -------------------------------------------------------------
  // Slice on our side rather than handing Plotly the whole series each time:
  // a redraw happens on every keypress, and a 50k-point relayout is visibly slow.
  // x values are naive ISO strings, never Date objects: Plotly renders a Date in
  // the VIEWER's timezone, which would print an axis hours away from the
  // timestamps in the side panel and the exported CSV. The series has no
  // timezone, so it must be shown verbatim.
  function slice(a, b) {
    a = Math.max(0, a); b = Math.min(D.n, b);
    const x = new Array(b - a), y = new Array(b - a);
    for (let i = a; i < b; i++) { x[i - a] = fmt(at(i)); y[i - a] = D.values[i]; }
    return { x, y };
  }

  const LAYOUT = {
    margin: { l: 52, r: 12, t: 8, b: 32 },
    showlegend: false, hovermode: "x unified",
    xaxis: { gridcolor: "#eef2f7" },
    yaxis: { gridcolor: "#eef2f7", title: { text: "value" } },
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "system-ui, sans-serif", size: 11, color: "#334155" },
  };
  const CONFIG = { displayModeBar: false, responsive: true };

  function drawMain() {
    const c = cur();
    if (!c) { Plotly.purge("main-plot"); return; }
    const color = D.type_colors[c.type] || "#0f172a";
    const mult = ZOOMS[ctxI].mult;
    const padMs = Math.max((c.i1 - c.i0) * D.step_ms * mult, MIN_PAD_MS);
    const pad = Math.ceil(padMs / D.step_ms);
    const a = mult === Infinity ? 0 : c.i0 - pad;
    const b = mult === Infinity ? D.n : c.i1 + pad;

    const [s0, s1] = spanOf(c);
    const ctx = slice(a, b);
    const prop = slice(c.i0, c.i1);
    const sel = slice(s0, s1);
    const traces = [
      // scatter (not scattergl) so plotly_click reports the point reliably
      { x: ctx.x, y: ctx.y, mode: "lines+markers", type: "scatter", name: "series",
        line: { color: "#94a3b8", width: 1.4 },
        marker: { color: "#94a3b8", size: 4 }, connectgaps: false },
      // the proposal, greyed where the reviewer has excluded it
      { x: prop.x, y: prop.y, mode: "lines", type: "scatter", name: c.type,
        line: { color: color, width: 1.6, dash: "dot" }, opacity: 0.55,
        connectgaps: false, hoverinfo: "skip" },
      { x: sel.x, y: sel.y, mode: "lines+markers", type: "scatter", name: "selected",
        line: { color: color, width: 2.4 }, marker: { color: color, size: 7 },
        connectgaps: false },
    ];
    // A gap-like all-NaN stretch draws nothing above, so band the span too.
    const layout = Object.assign({}, LAYOUT, {
      // i1 is exclusive, so band to at(i1): a one-row candidate then gets a
      // full sample-width band instead of a zero-width invisible rect.
      shapes: [{
        type: "rect", xref: "x", yref: "paper",
        x0: fmt(at(s0)), x1: fmt(at(s1)),
        y0: 0, y1: 1, fillcolor: color, opacity: 0.15, line: { width: 0 },
      }],
    });
    Plotly.react("main-plot", traces, layout, CONFIG);
    wireClick();
  }

  // Plotly only adds `.on` to the div once it has plotted, so this cannot be
  // registered with the other listeners at start-up — doing so throws and takes
  // the whole init down with it.
  let clickWired = false;
  function wireClick() {
    const gd = document.getElementById("main-plot");
    if (clickWired || typeof gd.on !== "function") return;
    clickWired = true;
    // Click a sample to label only that point instead of the whole proposal.
    gd.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0];
      if (!p) return;
      narrowTo(Math.round((Date.parse(p.x + "Z") - D.t0) / D.step_ms));
    });
  }

  function drawOverview() {
    // Stride-decimated: this strip is for orientation, not for reading values.
    const stride = Math.max(1, Math.ceil(D.n / 2500));
    const x = [], y = [];
    for (let i = 0; i < D.n; i += stride) { x.push(fmt(at(i))); y.push(D.values[i]); }

    const marks = { anomaly: [], normal: [], unsure: [], pending: [] };
    for (const c of D.candidates) {
      const mid = fmt(at(Math.floor((c.i0 + c.i1) / 2)));
      (marks[decisions[c.id] || "pending"]).push(mid);
    }
    const tick = (times, color, name) => ({
      x: times, y: times.map(() => -0.85), mode: "markers", type: "scattergl", name: name,
      marker: { color: color, size: 7, symbol: "triangle-up" }, yaxis: "y2",
      hovertemplate: name + " %{x}<extra></extra>",
    });

    const c = cur();
    const traces = [
      { x: x, y: y, mode: "lines", type: "scattergl", line: { color: "#cbd5e1", width: 1 },
        connectgaps: false, hoverinfo: "skip" },
      tick(marks.pending, "#94a3b8", "undecided"),
      tick(marks.normal, "#059669", "normal"),
      tick(marks.unsure, "#d97706", "unsure"),
      tick(marks.anomaly, "#dc2626", "anomaly"),
    ];
    const layout = Object.assign({}, LAYOUT, {
      margin: { l: 52, r: 12, t: 6, b: 24 },
      yaxis: { gridcolor: "#eef2f7", title: null, fixedrange: true },
      yaxis2: { overlaying: "y", range: [-1, 1], visible: false, fixedrange: true },
      showlegend: false,
      shapes: c ? [{
        type: "line", xref: "x", yref: "paper",
        x0: fmt(at(c.i0)), x1: fmt(at(c.i0)),
        y0: 0, y1: 1, line: { color: "#0f172a", width: 1.5, dash: "dot" },
      }] : [],
    });
    Plotly.react("overview-plot", traces, layout, CONFIG);
  }

  // ---- panel -------------------------------------------------------------
  const $ = (id) => document.getElementById(id);

  function render() {
    const v = view();
    if (idx >= v.length) idx = Math.max(0, v.length - 1);
    const c = cur();

    let a = 0, n = 0, u = 0;
    for (const cd of D.candidates) {
      const d = decisions[cd.id];
      if (d === "anomaly") a++; else if (d === "normal") n++; else if (d === "unsure") u++;
    }
    const done = a + n + u;
    $("t-anomaly").textContent = a; $("t-normal").textContent = n;
    $("t-unsure").textContent = u;
    $("t-pending").textContent = D.candidates.length - done;
    $("bar").firstElementChild.style.width =
      (D.candidates.length ? (100 * done) / D.candidates.length : 0) + "%";

    if (!c) {
      $("m-pos").textContent = "no candidates" + (filter ? " of this type" : "");
      Plotly.purge("main-plot");
      drawOverview();
      return;
    }

    $("m-pos").textContent = c.id + "  (" + (idx + 1) + " of " + v.length + ")";
    const pill = $("m-type");
    pill.textContent = c.type;
    pill.style.background = D.type_colors[c.type] || "#334155";
    const d = decisions[c.id] || "pending";
    const verdict = $("m-verdict");
    verdict.textContent = d === "pending" ? "— not yet judged" : d;
    verdict.className = "verdict " + d;
    const [s0, s1] = spanOf(c);
    const narrowed = s0 !== c.i0 || s1 !== c.i1;
    $("m-start").textContent = fmt(at(s0));
    $("m-end").textContent = fmt(at(s1 - 1));
    $("m-rows").textContent =
      (s1 - s0) + (narrowed ? " of " + c.n_rows + " (narrowed — R resets)" : "");
    $("m-hours").textContent = ((s1 - s0) * D.step_ms / 3600e3).toFixed(2) + " h";
    $("m-values").textContent =
      c.v_min === null ? "all missing" : c.v_min + " – " + c.v_max;
    // Severity is only comparable within a type, and its unit differs by type —
    // spell it out rather than showing a bare number that invites cross-type
    // comparison.
    const UNITS = {
      spike: " sigma from local median",
      level_shift: " sigma step",
      plateau: " h stuck",
      gap: " h missing",
    };
    $("m-score").textContent = c.score + (UNITS[c.type] || "");
    $("m-detectors").textContent = c.detectors.join(", ");
    $("m-note").textContent = c.note || "";

    drawMain();
    drawOverview();
  }

  // ---- actions -----------------------------------------------------------
  function decide(verdict) {
    const c = cur();
    if (!c) return;
    history.push({ id: c.id, prev: decisions[c.id], prevSpan: spans[c.id], idx: idx });
    decisions[c.id] = verdict;
    save();
    if (idx < view().length - 1) idx++;
    render();
  }

  // Narrow what gets labelled to a single sample. A detector marks a window;
  // often only one point in it is the anomaly, and labelling the rest as one too
  // would poison the ground truth. Can only narrow, never extend past the
  // proposal — merge_decisions enforces the same rule on the Python side.
  function narrowTo(i) {
    const c = cur();
    if (!c || i < c.i0 || i >= c.i1) return;
    history.push({ id: c.id, prev: decisions[c.id], prevSpan: spans[c.id], idx: idx });
    spans[c.id] = [i, i + 1];
    save();
    render();
  }

  function resetSpan() {
    const c = cur();
    if (!c || !spans[c.id]) return;
    history.push({ id: c.id, prev: decisions[c.id], prevSpan: spans[c.id], idx: idx });
    delete spans[c.id];
    save();
    render();
  }

  function undo() {
    const h = history.pop();
    if (!h) return;
    if (h.prev === undefined) delete decisions[h.id]; else decisions[h.id] = h.prev;
    if (h.prevSpan === undefined) delete spans[h.id]; else spans[h.id] = h.prevSpan;
    idx = h.idx;
    save();
    render();
  }

  function nav(step) {
    const v = view();
    if (!v.length) return;
    idx = Math.min(v.length - 1, Math.max(0, idx + step));
    render();
  }

  function exportCsv() {
    const head = ["candidate_id", "anomaly_type", "start", "end", "n_rows",
                  "duration_hours", "score", "detectors", "decision"];
    const rows = [head.join(",")];
    for (const c of D.candidates) {
      // start/end are the span the reviewer actually endorsed (end inclusive),
      // which merge_decisions labels — not necessarily the whole proposal.
      const [s0, s1] = spanOf(c);
      rows.push([c.id, c.type, fmt(at(s0)), fmt(at(s1 - 1)), s1 - s0,
                 ((s1 - s0) * D.step_ms / 3600e3).toFixed(2), c.score,
                 '"' + c.detectors.join("|") + '"', decisions[c.id] || ""].join(","));
    }
    const blob = new Blob([rows.join("\n") + "\n"], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = D.name + "_decisions.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  // ---- wiring ------------------------------------------------------------
  $("title").textContent = D.file + " — " + D.candidates.length + " candidates";

  const sel = $("filter");
  for (const t of D.type_order) {
    const n = D.candidates.filter((c) => c.type === t).length;
    const o = document.createElement("option");
    o.value = t; o.textContent = t + " (" + n + ")";
    if (!n) o.disabled = true;
    sel.appendChild(o);
  }
  sel.addEventListener("change", () => { filter = sel.value; idx = 0; render(); });

  document.querySelectorAll("[data-decide]").forEach((b) =>
    b.addEventListener("click", () => decide(b.dataset.decide)));
  document.querySelectorAll("[data-nav]").forEach((b) =>
    b.addEventListener("click", () => nav(+b.dataset.nav)));
  $("undo").addEventListener("click", undo);
  $("export").addEventListener("click", exportCsv);
  $("context").addEventListener("click", cycleContext);
  $("reset").addEventListener("click", resetSpan);

  function cycleContext() {
    ctxI = (ctxI + 1) % ZOOMS.length;
    $("context").firstChild.textContent = "Zoom: " + ZOOMS[ctxI].label;
    drawMain();
  }

  document.addEventListener("keydown", (e) => {
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (/^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    const map = { a: "anomaly", "1": "anomaly", n: "normal", "2": "normal",
                  u: "unsure", "3": "unsure" };
    if (map[k]) { decide(map[k]); }
    else if (e.key === "ArrowRight" || k === " ") { nav(1); }
    else if (e.key === "ArrowLeft") { nav(-1); }
    else if (k === "z") { undo(); }
    else if (k === "e") { exportCsv(); }
    else if (k === "f") { cycleContext(); }
    else if (k === "r") { resetSpan(); }
    else return;
    e.preventDefault();
  });

  const warn = [];
  for (const [t, n] of Object.entries(D.truncated || {})) {
    warn.push(n + " lower-scoring " + t + " candidate(s) were dropped by --max-per-type");
  }
  for (const [tool, msg] of Object.entries(D.errors || {})) {
    warn.push(tool + " failed and proposed nothing: " + msg);
  }
  if (warn.length) $("warnings").innerHTML = '<br><span class="warn">' + warn.join("<br>") + "</span>";

  render();
})();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- CLI
def _config_from_args(args: argparse.Namespace) -> DetectConfig:
    return DetectConfig(
        unilof_thresh=args.unilof_thresh,
        zscore_thresh=args.zscore_thresh,
        jumps_thresh_sigmas=args.jumps_sigmas,
        max_per_type=args.max_per_type,
    )


def _cmd_detect(args: argparse.Namespace) -> int:
    result = find_candidates(args.path, config=_config_from_args(args))
    print(result.summary())

    DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = DEFAULT_DATA_DIR / f"{args.path.stem}_candidates.csv"
    result.frame().to_csv(out_csv, index=False)

    out_html = args.out or DEFAULT_PAGE_DIR / f"{args.path.stem}_review.html"
    out_html.parent.mkdir(parents=True, exist_ok=True)
    out_html.write_text(build_review_html(result), encoding="utf-8")

    print(f"\n  candidates -> {out_csv}")
    print(f"  review page -> {out_html}")
    if not args.no_open:
        webbrowser.open(out_html.resolve().as_uri())
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    result = find_candidates(args.path, config=_config_from_args(args))
    decisions = pd.read_csv(args.decisions)
    missing = [c for c in ("candidate_id", "decision") if c not in decisions.columns]
    if missing:
        raise ValueError(
            f"{args.decisions.name} is missing column(s) {missing}; expected the CSV "
            f"exported by the review page, got columns={list(decisions.columns)}"
        )

    known = {c.candidate_id for c in result.candidates}
    unknown = sorted(set(decisions["candidate_id"].astype(str)) - known)
    if unknown:
        # The page's ids come from the detector settings, so a mismatch means the
        # decisions were made against a different proposal run. Merging anyway
        # would silently drop those verdicts.
        raise SystemExit(
            f"{len(unknown)} candidate_id(s) in {args.decisions.name} do not exist in "
            f"this proposal run (e.g. {unknown[:3]}). Re-run 'detect' with the same "
            "options you used to build the review page, or re-review."
        )

    labels = merge_decisions(result, decisions, include_unsure=args.include_unsure)
    out = args.out or DEFAULT_DATA_DIR / f"{args.path.stem}_review_labels.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out, index=False)

    n = int(labels["is_anomaly"].sum())
    counts = labels.loc[labels["is_anomaly"], "anomaly_type"].value_counts().to_dict()
    print(f"wrote {out}")
    print(f"  {n:,} of {len(labels):,} rows marked anomalous ({100 * n / len(labels):.2f}%)")
    for t in LABEL_TYPES:
        if t in counts:
            note = "  (auto-labelled, not reviewed)" if t == "gap" else ""
            print(f"    {t:<12} {counts[t]:>6} rows{note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Propose anomalies in a 'clean' series and review them by keyboard.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("path", type=Path, help="Series CSV (datetime,value).")
        p.add_argument("--unilof-thresh", type=float, default=DetectConfig.unilof_thresh,
                       help="flagUniLOF cutoff; lower = more spike candidates.")
        p.add_argument("--zscore-thresh", type=float, default=DetectConfig.zscore_thresh,
                       help="flagZScore (modified) threshold.")
        p.add_argument("--jumps-sigmas", type=float,
                       default=DetectConfig.jumps_thresh_sigmas,
                       help="flagJumps threshold, in robust step-sigmas of this series.")
        p.add_argument("--max-per-type", type=int, default=DetectConfig.max_per_type,
                       help="Keep at most this many candidates per type (default: "
                            "no limit). Trims the queue to something reviewable, "
                            "at the cost of recall — dropped candidates never "
                            "reach you.")

    p_det = sub.add_parser("detect", help="Propose candidates and build the review page.")
    add_common(p_det)
    p_det.add_argument("--out", type=Path, default=None, help="Review page path (HTML).")
    p_det.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    p_det.set_defaults(func=_cmd_detect)

    p_mrg = sub.add_parser("merge", help="Fold an exported decisions CSV into §5 labels.")
    add_common(p_mrg)
    p_mrg.add_argument("decisions", type=Path, help="CSV exported from the review page.")
    p_mrg.add_argument("--out", type=Path, default=None, help="Labels CSV path.")
    p_mrg.add_argument("--include-unsure", action="store_true",
                       help="Also label 'unsure' candidates as anomalies.")
    p_mrg.set_defaults(func=_cmd_merge)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (ContractError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
