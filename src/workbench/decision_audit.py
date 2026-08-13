"""Click any point in a run's output and find out what happened to it, and why.

`spike_audit` answers one question about one anomaly type. This answers the general
one, for every point in the series and all four §6 types: was this row flagged, by
what, what did the agent decide, what reason did it give, what does the ground truth
say — and if nothing happened to it, say that explicitly rather than leaving a blank.

That last part is the point. A run's flag log only contains rows that were flagged,
so "no entry" silently covers two very different situations: a row no detector
looked twice at, and a labelled anomaly every detector walked past. The page
separates them.

Categories (pick with 1-6, or the counts in the sidebar):

  wrongly-deleted  the agent removed a value the labels call normal water. THE ONE
                   TO REVIEW: §1 says removing real data is the worst outcome, and
                   on this project many of these turn out to be real events the
                   label file never recorded — pass --raw and the panel tells you
                   whether the same excursion exists in the approved base.
  deleted          removed, and the labels agree it was an anomaly.
  missed           a labelled anomaly that WAS flagged, and the agent kept anyway.
  undetected       a labelled anomaly NO detector flagged. Upstream of the agent.
  imputed          a gap the agent filled.
  overwritten-anomaly  a labelled anomaly impute_rolling overwrote before the agent
                   judged it — the §7.1 dfilter bug. A MISS in all but name, and it is
                   split out because folding it in with the rest let the page report
                   missed=0 on 03447687_l1 while 10 spikes and 3 level shifts sat in a
                   bucket named after the mechanism instead of the outcome.
  overwritten-water  impute_rolling replaced a real reading that was NEVER MISSING.
  left-missing     a gap left as NaN. Correct behaviour for a long outage (§6 says
                   impute short gaps only), so it is deliberately NOT counted as a
                   miss — without this split the "missed" bucket read 1,763 on
                   03447687_l1 and was almost entirely correct decisions.
  kept             flagged, inspected or not, and left in place.

Every point carries its provenance in the side panel: flagged_by, the decision and
the agent's own words for it, the label, and the describe_point(s) measurement the
agent had when it decided — or "never measured", which is a different failure from
deciding wrongly.

Self-contained HTML, plotly.js inlined, no server — like the other workbench pages.

CLI
---
    python -m src.workbench.decision_audit \\
        data/injected/03447687/l1/03447687_l1.csv \\
        --flags data/agent_runs/03447687_l1_flags.json \\
        --log logs/run_20260811_114419.jsonl \\
        --raw data/raw/approved/03447687_turbidity_63680.csv

Pages land in `figures/decision_audit/`.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd

from src.inspect_data import DATETIME_COL
from src.workbench.spike_audit import (
    load_flag_log,
    load_run,
    robust_step_sigma,
    _local_sigmas,
)

DEFAULT_OUTDIR = Path("figures/decision_audit")

# §5 actions meaning the agent judged the value wrong and removed/replaced it.
REMOVING_ACTIONS = frozenset({"delete", "correct"})

CATEGORIES = (
    "wrongly-deleted", "missed", "overwritten-anomaly", "undetected",
    "overwritten-water", "deleted", "imputed", "left-missing", "kept",
)

# Categories in which a labelled anomaly went unhandled. Summed in the CLI output so
# the headline cannot read "missed=0" while thirteen anomalies sit in another bucket.
UNHANDLED_ANOMALY = ("missed", "overwritten-anomaly", "undetected")

# Samples drawn either side of the selected point in the detail plot.
CONTEXT_SAMPLES = 96          # 24 h on a 15-min grid


def categorise(action: str, label: str, flagged: bool) -> str:
    """Which of the six buckets one row falls into.

    `label` is the §5 anomaly_type, or "" for normal water.
    """
    is_anomaly = bool(label)
    if not flagged:
        return "undetected" if is_anomaly else ""       # unflagged normal water is not a case
    if action == "impute":
        # `impute` on a row that was never missing means interpolateByRolling replaced a
        # real reading — SaQC filters already-flagged rows out of the imputer's input, so
        # it "fills" them (§7.1, the dfilter bug). Filing those under `imputed` reports a
        # silent overwrite as a success: on 03447687_l1 it hid all 10 injected spikes the
        # run failed to judge, and the page showed missed=0.
        if label == "gap":
            return "imputed"
        # Splitting these is not cosmetic. An anomaly the imputer overwrote was never
        # judged by the agent at all, which is a MISS by any reading — but it does not
        # show up as one, because the row's action is `impute`. On 03447687_l1 that let
        # the page report missed=0 while 10 spikes and 3 level shifts sat in a bucket
        # named after the mechanism rather than the outcome.
        return "overwritten-anomaly" if is_anomaly else "overwritten-water"
    if action in REMOVING_ACTIONS:
        return "deleted" if is_anomaly else "wrongly-deleted"
    # A gap left as NaN is not a miss — §6 says impute SHORT gaps only, and filling a
    # multi-day outage with a rolling median invents data. Without this branch the
    # "missed" bucket is swamped by correct behaviour: on 03447687_l1 it read 1,763,
    # of which all but a handful were long outages the agent deliberately left alone.
    if label == "gap":
        return "left-missing"
    return "missed" if is_anomaly else "kept"


def build_cases(
    series: pd.Series,
    labels: pd.DataFrame,
    flag_log: pd.DataFrame,
    measured: dict[pd.Timestamp, dict],
    raw: pd.Series | None = None,
) -> list[dict]:
    """One record per point worth clicking: every flagged row + every labelled anomaly.

    Points that are neither are deliberately absent — the page reports those from the
    series itself ("never flagged, not a labelled anomaly") rather than embedding
    70,000 empty records.
    """
    sigma = robust_step_sigma(series)
    raw_sigma = robust_step_sigma(raw) if raw is not None else None

    label_type = labels["anomaly_type"].fillna("")
    anomalies = pd.DatetimeIndex(labels.index[label_type != ""])
    interesting = pd.DatetimeIndex(
        sorted(set(flag_log.index) | set(anomalies))
    )

    position = {ts: i for i, ts in enumerate(series.index)}
    cases: list[dict] = []
    for at in interesting:
        if at not in position:
            continue
        flagged = at in flag_log.index
        action = str(flag_log["action"].get(at, "")) if flagged else ""
        label = str(label_type.get(at, ""))
        category = categorise(action, label, flagged)
        if not category:
            continue

        value = float(series.get(at, np.nan))
        true_value = labels["true_value"].get(at)
        true_value = float(true_value) if pd.notna(true_value) else None
        record = {
            "at": at.strftime("%Y-%m-%dT%H:%M:%S"),
            "i": position[at],
            "category": category,
            "value": None if not np.isfinite(value) else round(value, 3),
            "label": label or "normal water",
            # pd.isna, not `or ""` — a float NaN is TRUTHY, so `or` let it through and the
            # panel rendered "normal water (nan)".
            "source": ("" if pd.isna(labels["source"].get(at))
                       else str(labels["source"].get(at))),
            "flagged": bool(flagged),
            "flagged_by": str(flag_log["flagged_by"].get(at, "")) if flagged else "",
            "action": action or "(never flagged)",
            "reason": str(flag_log["reason"].get(at, "")) if flagged else "",
            "rationale": str(flag_log["rationale"].get(at, "")) if flagged else "",
            "rationale_source": str(flag_log["rationale_source"].get(at, "")) if flagged else "",
            "deliberation": str(flag_log["deliberation"].get(at, "")) if flagged else "",
            "measured": at in measured,
            "measurement": measured.get(at),
            "true_value": None if true_value is None else round(true_value, 3),
            "local_sigmas": _round(_local_sigmas(series, at, sigma)),
        }
        if raw is not None and at in raw.index:
            record["raw_value"] = round(float(raw[at]), 3)
            record["raw_sigmas"] = _round(_local_sigmas(raw, at, raw_sigma))
        cases.append(record)
    return cases


def _round(x, n=1):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def build_payload(series: pd.Series, cases: list[dict], title: str,
                  has_raw: bool) -> dict:
    index = pd.DatetimeIndex(series.index)
    # as_unit("ms"), NOT view("int64") // 1e6 — these CSVs are datetime64[us] and that
    # idiom silently yields seconds, plotting the whole record inside 1970 (§9.1).
    epoch_ms = index.as_unit("ms").astype("int64").tolist()
    values = [None if not np.isfinite(v) else round(float(v), 3)
              for v in series.to_numpy()]

    counts = {c: sum(1 for k in cases if k["category"] == c) for c in CATEGORIES}
    return {
        "title": title,
        "epoch_ms": epoch_ms,
        "values": values,
        "cases": cases,
        "counts": counts,
        "has_raw": has_raw,
        "context": CONTEXT_SAMPLES,
        "n_rows": len(series),
    }


def build_html(payload: dict) -> str:
    from plotly.offline import get_plotlyjs

    return _TEMPLATE.replace("/*PLOTLY_JS*/", get_plotlyjs()).replace(
        "/*PAYLOAD*/", json.dumps(payload)
    )


_TEMPLATE = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Decision audit</title>
<script>/*PLOTLY_JS*/</script>
<style>
  :root { --bg:#0f1115; --panel:#171a21; --line:#242835; --text:#e6e9ef; --dim:#9aa3b2; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); height:100vh;
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
         display:grid; grid-template-columns:320px 1fr; }
  #side { border-right:1px solid var(--line); background:var(--panel); overflow-y:auto; }
  /* min-width:0 is load-bearing: a grid item defaults to min-width:auto, so the wide
     <pre> and the plotly SVG stop this column from ever shrinking and the whole page
     scrolls sideways with the panel clipped off-screen. */
  #main { display:grid; grid-template-rows:auto auto 1fr auto; overflow:hidden;
          min-width:0; }
  h1 { font-size:15px; margin:0; padding:13px 15px; border-bottom:1px solid var(--line); }
  h1 small { display:block; color:var(--dim); font-weight:400; font-size:12px; margin-top:3px; }
  .cat { padding:9px 15px; cursor:pointer; display:flex; justify-content:space-between;
         border-left:3px solid transparent; }
  .cat:hover { background:#1e222c; }
  .cat.on { background:#252b39; border-left-color:currentColor; }
  .cat .n { color:var(--dim); font-variant-numeric:tabular-nums; }
  .hint { padding:4px 15px 12px; color:var(--dim); font-size:11.5px; }
  .item { padding:5px 15px; cursor:pointer; font-size:12.5px;
          display:flex; justify-content:space-between; font-variant-numeric:tabular-nums; }
  .item:hover { background:#1e222c; }
  .item.sel { background:#2b3242; }
  .item .m { color:var(--dim); font-size:11px; }
  #head { padding:11px 18px; border-bottom:1px solid var(--line); display:flex;
          gap:24px; flex-wrap:wrap; align-items:baseline; }
  #head .k { color:var(--dim); font-size:11px; letter-spacing:.08em; text-transform:uppercase; }
  #head .v { font-size:18px; font-variant-numeric:tabular-nums; }
  #overview { height:112px; min-width:0; }
  #detail { min-height:0; min-width:0; }
  #panel { border-top:1px solid var(--line); background:var(--panel); padding:12px 18px;
           overflow-y:auto; max-height:34vh; min-width:0; }
  .row { display:grid; grid-template-columns:160px minmax(0,1fr); gap:10px; padding:3px 0;
         border-bottom:1px solid #1d212b; overflow-wrap:anywhere; }
  .row .k { color:var(--dim); }
  .note { padding:9px 12px; border-radius:6px; margin-bottom:10px; line-height:1.45; }
  .note.bad { background:#3a1a1a; border:1px solid #7f1d1d; }
  .note.warn { background:#3b2a10; border:1px solid #7c5310; }
  .note.ok { background:#12261a; border:1px solid #14532d; }
  .note.flat { background:#1b1f28; border:1px solid var(--line); }
  pre { margin:8px 0 0; padding:9px 11px; background:#0c0e13; border:1px solid var(--line);
        border-radius:6px; overflow-x:auto; font-size:11.5px; color:#cbd5e1; }
  kbd { background:#0c0e13; border:1px solid var(--line); border-radius:4px; padding:1px 5px;
        font-size:11px; }
</style>
<body>
<div id="side"></div>
<div id="main">
  <div id="head"></div>
  <div id="overview"></div>
  <div id="detail"></div>
  <div id="panel"></div>
</div>
<script>
const D = /*PAYLOAD*/;
const COLOR = {
  "wrongly-deleted":"#ef4444", "missed":"#f59e0b", "overwritten-anomaly":"#fb923c",
  "undetected":"#a855f7", "overwritten-water":"#fb7185", "deleted":"#22c55e",
  "imputed":"#38bdf8", "left-missing":"#7dd3fc", "kept":"#64748b",
};
const BLURB = {
  "wrongly-deleted":"Removed a value the labels call normal water — review these first.",
  "deleted":"Removed, and the labels agree it was an anomaly.",
  "missed":"A labelled anomaly that was flagged, then kept.",
  "undetected":"A labelled anomaly no detector flagged.",
  "imputed":"A gap the agent filled.",
  "overwritten-anomaly":"A labelled anomaly the imputer overwrote before the agent judged it — a miss in all but name.",
  "overwritten-water":"impute_rolling replaced a real reading that was never missing.",
  "left-missing":"A gap left as NaN. Correct for a long outage — §6 says impute short gaps only.",
  "kept":"Flagged and left in place.",
};
const $ = (id) => document.getElementById(id);
const iso = (ms) => new Date(ms).toISOString().slice(0, 19);   // naive round-trip
const f2 = (v, n=2) => (v === null || v === undefined) ? "&mdash;" : (+v).toFixed(n);
const byAt = new Map(D.cases.map(c => [c.at, c]));

let cat = "wrongly-deleted";
let sel = 0;

const inCat = () => D.cases.filter(c => c.category === cat);

function renderSide() {
  let h = `<h1>Decision audit<small>${D.title}</small></h1>`;
  for (const c of Object.keys(COLOR)) {
    h += `<div class="cat ${c===cat?"on":""}" style="color:${COLOR[c]}" onclick="pick('${c}')">
            <span>${c}</span><span class="n">${D.counts[c]}</span></div>`;
  }
  h += `<div class="hint">Click any point on either plot — including one that was never
        flagged. <kbd>1</kbd>-<kbd>9</kbd> switch category, <kbd>J</kbd>/<kbd>K</kbd> step.</div>`;
  const rows = inCat();
  for (let i = 0; i < rows.length; i++) {
    const c = rows[i];
    const extra = c.category === "wrongly-deleted" && c.raw_sigmas != null
      ? `${f2(c.raw_sigmas,1)}σ raw`
      : (c.category === "missed" && !c.measured ? "never measured" : f2(c.value));
    h += `<div class="item ${i===sel?"sel":""}" onclick="sel=${i};draw()">
            <span>${c.at.replace("T"," ")}</span><span class="m">${extra}</span></div>`;
  }
  $("side").innerHTML = h;
}

function pick(c) { cat = c; sel = 0; draw(); }

function overview() {
  const stride = Math.max(1, Math.ceil(D.n_rows / 3000));
  const x = [], y = [];
  for (let i = 0; i < D.n_rows; i += stride) { x.push(iso(D.epoch_ms[i])); y.push(D.values[i]); }
  const traces = [{ x, y, mode:"lines", type:"scattergl", name:"series",
                    line:{color:"#3f4757", width:1}, hoverinfo:"skip" }];
  for (const c of Object.keys(COLOR)) {
    const pts = D.cases.filter(k => k.category === c);
    if (!pts.length) continue;
    traces.push({
      x: pts.map(p => p.at), y: pts.map(p => p.value),
      mode:"markers", type:"scattergl", name:c,
      marker:{ color:COLOR[c], size:c==="kept"?4:7, opacity:c==="kept"?0.5:0.95 },
    });
  }
  Plotly.react("overview", traces, {
    margin:{l:52,r:16,t:6,b:22}, paper_bgcolor:"#0f1115", plot_bgcolor:"#0f1115",
    font:{color:"#9aa3b2",size:10}, showlegend:false,
    xaxis:{gridcolor:"#1d212b"}, yaxis:{gridcolor:"#1d212b"},
  }, {displayModeBar:false, responsive:true});
}

function draw() {
  const rows = inCat();
  if (rows.length) sel = Math.max(0, Math.min(sel, rows.length - 1));
  const c = rows[sel];
  renderSide();
  if (!c) { $("head").innerHTML = `<div class="k">no cases in this category</div>`; return; }
  showPoint(c.at);
}

function showPoint(at) {
  const c = byAt.get(at);
  const i = c ? c.i : nearestIndex(at);
  const lo = Math.max(0, i - D.context), hi = Math.min(D.n_rows, i + D.context);
  const x = [], y = [];
  for (let k = lo; k < hi; k++) { x.push(iso(D.epoch_ms[k])); y.push(D.values[k]); }

  const traces = [{ x, y, mode:"lines+markers", type:"scatter", name:"series",
                    line:{color:"#8b93a5",width:1.4}, marker:{size:3.5,color:"#8b93a5"},
                    connectgaps:false }];
  // Every case inside the window, so neighbouring decisions are visible in context.
  for (const k of Object.keys(COLOR)) {
    const pts = D.cases.filter(p => p.i >= lo && p.i < hi && p.category === k);
    if (!pts.length) continue;
    traces.push({ x:pts.map(p=>p.at), y:pts.map(p=>p.value), mode:"markers", type:"scatter",
                  name:k, marker:{color:COLOR[k], size:9, symbol:"circle-open",
                  line:{width:2.5,color:COLOR[k]}} });
  }
  if (c && c.true_value != null && c.label !== "normal water") {
    traces.push({ x:[c.at], y:[c.true_value], mode:"markers", type:"scatter",
                  name:"true value", marker:{color:"#facc15", size:10, symbol:"x"} });
  }
  Plotly.react("detail", traces, {
    margin:{l:52,r:16,t:8,b:34}, paper_bgcolor:"#0f1115", plot_bgcolor:"#0f1115",
    font:{color:"#9aa3b2",size:11}, xaxis:{gridcolor:"#1d212b"},
    yaxis:{gridcolor:"#1d212b",title:"FNU"},
    legend:{orientation:"h", y:1.14, font:{size:10}},
    shapes:[{type:"line", x0:at, x1:at, yref:"paper", y0:0, y1:1,
             line:{color:"#4b5563", width:1, dash:"dot"}}],
  }, {displayModeBar:false, responsive:true});
  wireClick();
  renderHead(at, c);
  renderPanel(at, c);
}

function nearestIndex(at) {
  const t = Date.parse(at + "Z");
  let lo = 0, hi = D.n_rows - 1;
  while (lo < hi) { const mid = (lo + hi) >> 1;
    if (D.epoch_ms[mid] < t) lo = mid + 1; else hi = mid; }
  return lo;
}

function renderHead(at, c) {
  const v = c ? c.value : D.values[nearestIndex(at)];
  $("head").innerHTML = `
    <div><div class="k">timestamp</div><div class="v">${at.replace("T"," ")}</div></div>
    <div><div class="k">value</div><div class="v">${f2(v)}</div></div>
    <div><div class="k">outcome</div><div class="v" style="color:${c?COLOR[c.category]:"#9aa3b2"}">
         ${c ? c.category : "nothing happened"}</div></div>
    <div><div class="k">decision</div><div class="v">${c ? c.action : "never flagged"}</div></div>`;
}

const SRC_LABEL = {
  "agent-deliberation": ["#22c55e", "the agent reasoned this one through"],
  "agent-reason":       ["#38bdf8", "the agent called this clear-cut"],
  "blanket":            ["#f59e0b", "swept up by a blanket span, not judged individually"],
  "deterministic":      ["#64748b", "decided by code, not by the agent"],
};

function rationaleBlock(c) {
  const [colour, gloss] = SRC_LABEL[c.rationale_source] || ["#64748b", ""];
  let h = "";
  if (c.rationale_source) {
    h += `<div style="font-size:11px;color:${colour};text-transform:uppercase;
          letter-spacing:.07em;margin-bottom:3px">${c.rationale_source} &mdash; ${gloss}</div>`;
  }
  h += (c.rationale || c.reason || "&mdash;");
  if (c.deliberation) {
    h += `<div style="margin-top:8px;padding:9px 11px;border-left:3px solid #22c55e;
          background:#12261a;border-radius:0 6px 6px 0">
          <div style="font-size:11px;color:#86efac;text-transform:uppercase;
               letter-spacing:.07em;margin-bottom:4px">the agent's working</div>
          ${c.deliberation}</div>`;
  }
  return h;
}

function renderPanel(at, c) {
  const row = (k, v) => `<div class="row"><div class="k">${k}</div><div>${v}</div></div>`;
  if (!c) {
    $("panel").innerHTML = `<div class="note flat"><b>Nothing happened to this point.</b>
      No detector flagged it, so the agent never saw it, and the label file does not call it
      an anomaly. It is ordinary water that was correctly left alone — the overwhelming
      majority of the ${D.n_rows.toLocaleString()} rows are in this state.</div>`;
    return;
  }
  let note;
  if (c.category === "wrongly-deleted") {
    note = D.has_raw && c.raw_sigmas >= 5
      ? [`bad`, `The agent deleted this, and the labels call it normal water — but the same
          excursion is <b>${f2(c.raw_sigmas,1)}σ in the raw approved base</b> and injection
          never touched it. Judge it yourself: this is either a real event the labels omit,
          or a genuine false positive.`]
      : [`bad`, `The agent deleted a value the labels call normal water. ${D.has_raw
          ? `Only ${f2(c.raw_sigmas,1)}σ in the raw base, so this looks like a genuine
             false positive.` : `Pass <code>--raw</code> to check it against the approved base.`}`];
  } else if (c.category === "missed") {
    note = c.measured
      ? [`warn`, `A labelled anomaly that was flagged and measured, and kept anyway. The
          measurement below is what the agent had — read it against its stated reason.`]
      : [`bad`, `A labelled anomaly that was flagged but <b>never measured</b>. No
          describe_point(s) call covered it, so a decision span swept it up unexamined.`];
  } else if (c.category === "overwritten-anomaly") {
    note = [`bad`, `<b>A labelled ${c.label} that the agent never judged.</b>
      impute_rolling overwrote it before any decision was made — SaQC filters
      already-flagged rows out of the imputer's input, so it treats them as gaps and
      fills them (the dfilter bug, §7.1). Count this as a miss: the value is gone,
      replaced by a rolling median, and no reasoning was recorded.`];
  } else if (c.category === "overwritten-water") {
    note = [`bad`, `<b>impute_rolling replaced this value, and it was never missing.</b>
      SaQC filters already-flagged rows out of the imputer's input, so it treats them as
      gaps and fills them (the dfilter bug, §7.1). The agent never made a real decision
      here — whatever the label says, this row was silently rewritten with a rolling
      median rather than judged.`];
  } else if (c.category === "undetected") {
    note = [`bad`, `A labelled anomaly that <b>no detector flagged</b>. The agent never had
      the chance to judge it — this is detector tuning, not decision quality.`];
  } else {
    note = [(c.category === "kept" || c.category === "left-missing") ? "flat" : "ok",
            BLURB[c.category]];
  }
  let h = `<div class="note ${note[0]}">${note[1]}</div>`;
  h += row("ground truth", c.label + (c.source ? ` (${c.source})` : ""));
  if (c.true_value != null) h += row("true value", `${f2(c.true_value)} &rarr; ${f2(c.value)}`);
  h += row("displacement", `${f2(c.local_sigmas,1)}σ from the local median`);
  if (D.has_raw) h += row("raw approved base", c.raw_value == null ? "not present"
        : `${f2(c.raw_value)} (${f2(c.raw_sigmas,1)}σ there)`);
  h += row("flagged by", c.flagged_by || "nothing — no detector fired here");
  h += row("agent decided", c.action);
  h += row("why", rationaleBlock(c));
  h += row("did it measure it?", c.measured ? "yes" : "<b>no — never inspected</b>");
  if (c.measurement) h += `<pre>${JSON.stringify(c.measurement, null, 2)}</pre>`;
  $("panel").innerHTML = h;
}

let wired = new Set();
function wireClick() {
  // Plotly attaches `.on` only after it has plotted into the div; registering at
  // start-up throws and takes the rest of init down with it (§9.1).
  for (const id of ["detail", "overview"]) {
    const gd = $(id);
    if (wired.has(id) || typeof gd.on !== "function") continue;
    wired.add(id);
    gd.on("plotly_click", (ev) => {
      const p = ev.points && ev.points[0];
      if (!p) return;
      const c = byAt.get(p.x);
      if (c) { cat = c.category; sel = inCat().indexOf(c); renderSide(); }
      showPoint(p.x);
    });
  }
}

addEventListener("keydown", (e) => {
  if (/^[1-9]$/.test(e.key)) { pick(Object.keys(COLOR)[+e.key - 1]); return; }
  if (e.key === "j" || e.key === "ArrowDown") { sel++; draw(); }
  else if (e.key === "k" || e.key === "ArrowUp") { sel--; draw(); }
});

overview();
draw();
wireClick();
</script>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Click any point in a run's output and see what happened to it, and why."
    )
    parser.add_argument("series", type=Path, help="Injected dataset CSV; labels read beside it.")
    parser.add_argument("--flags", type=Path, required=True, help="§5 flag log (*_flags.json).")
    parser.add_argument("--log", type=Path, required=True, help="Agent run log (logs/run_*.jsonl).")
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="Raw APPROVED base CSV. Strongly recommended: it is what distinguishes a real "
             "false positive from an event the label file never recorded.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.series, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
    series = frame.set_index(DATETIME_COL)["value"]

    labels_path = args.series.with_name(f"{args.series.stem}_labels.csv")
    if not labels_path.exists():
        parser.error(f"labels not found beside the series: {labels_path}")
    labels = pd.read_csv(labels_path, parse_dates=[DATETIME_COL]).set_index(DATETIME_COL)

    raw = None
    if args.raw is not None:
        raw_frame = pd.read_csv(args.raw, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
        raw = raw_frame.set_index(DATETIME_COL)["value"]

    _, measured = load_run(args.log)
    cases = build_cases(series, labels, load_flag_log(args.flags), measured, raw)
    payload = build_payload(
        series, cases, title=f"{args.series.stem} · {args.log.name}", has_raw=raw is not None
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{args.series.stem}_{args.log.stem}_decisions.html"
    out.write_text(build_html(payload))

    print("  " + ", ".join(f"{k}={v}" for k, v in payload["counts"].items()))
    unhandled = sum(payload["counts"][c] for c in UNHANDLED_ANOMALY)
    if unhandled:
        parts = ", ".join(f"{payload['counts'][c]} {c}" for c in UNHANDLED_ANOMALY
                          if payload["counts"][c])
        print(f"  {unhandled} labelled anomalies went unhandled ({parts}).")
    print(f"  {len(cases):,} clickable points out of {len(series):,} rows; "
          "the rest were never flagged and are not labelled anomalies.")
    if not payload["has_raw"]:
        print("  NOTE: no --raw, so wrongly-deleted points cannot be checked against the base.")
    print(f"  wrote {out}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
