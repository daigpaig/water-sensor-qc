"""Audit one run's SPIKE decisions, case by case, with the evidence behind each.

`evaluate.py` says a run scored 0.06 precision on spikes. This page says *why*, one
decision at a time: what the detectors flagged, what the agent measured, what it
decided, what the label file says, and — the part that turned out to matter — what
the RAW APPROVED BASE says at the same timestamp.

Six outcomes, which is the whole point of the page (the middle four are the ones
worth your time):

  caught      an injected spike the agent acted on. Working as intended.
  deleted-natural  a row the labels call normal water that the agent deleted. Scored
              as a false positive — but §9 is explicit that an approved record still
              contains real sensor spikes, and the labels record none of them. If the
              same excursion is present in the raw base, the "error" is the ground
              truth's, not the agent's. Pass --raw and the page tells you which.
  missed      an injected spike the agent measured or saw, and KEPT. Check `measured`
              first: on the run this was written for, 6 of 9 misses had never been
              measured at all, so the failure was coverage, not judgement.
  overwritten an injected spike impute_rolling overwrote before the agent judged it
              (the §7.1 dfilter bug). Also a failure, but a different one — no
              decision was ever made, so it is not filed under `missed`.
  undetected  an injected spike NO spike detector flagged. Upstream of the agent
              entirely — a flagUniLOF recall gap, not a decision error.
  kept-natural  normal water the agent correctly kept. Listed for completeness and
              collapsed by default; this is the bulk.

The page is one self-contained HTML file with plotly.js inlined — no server, works
offline — like `src.workbench.review` and `src.workbench.visualize_log`.

Keys: J/K or arrows to step, 1-6 to jump to an outcome group, A to toggle the
normal-water cases, R to reset the zoom.

CLI
---
    python -m src.workbench.spike_audit \\
        data/turbidity/injected/02054550/l1/02054550_l1.csv \\
        --flags data/agent_runs/03447687_l1_flags.json \\
        --log logs/run_20260810_174153.jsonl \\
        --raw data/turbidity/approved/02054550_turbidity_63680.csv

`--raw` is optional but is the single most useful flag on this page: without it a
deleted-natural case cannot be told from a mislabelled base spike.

Pages land in `figures/spike_audit/`.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.inspect_data import DATETIME_COL

DEFAULT_OUTDIR = Path("figures/spike_audit")

# Which SaQC methods are trying to catch a spike. Matches evaluate.TOOL_TO_TYPE;
# kept local so `workbench -> top level` stays the only dependency direction (§4).
SPIKE_FUNCS = ("flagUniLOF", "flagZScore", "flagRange")

# §5 actions that mean the agent judged the value wrong.
POSITIVE_ACTIONS = frozenset({"delete", "correct"})

# Context drawn either side of the selected point in the detail plot, as a DURATION.
# It used to be a fixed 96 samples, which silently meant 24 h on the 15-min bases and
# 8 h once the project moved to 5-min ones — the plot quietly stopped showing the
# surroundings a storm-vs-spike call depends on.
CONTEXT_WINDOW = pd.Timedelta("24h")
CONTEXT_SAMPLES = 96          # fallback when the step cannot be inferred


def context_samples(index: pd.DatetimeIndex) -> int:
    """How many samples span :data:`CONTEXT_WINDOW` on *this* series' grid."""
    if len(index) < 2:
        return CONTEXT_SAMPLES
    step = pd.Series(index).diff().median()
    if pd.isna(step) or step <= pd.Timedelta(0):
        return CONTEXT_SAMPLES
    return max(8, int(round(CONTEXT_WINDOW / step)))

# Shared vocabulary with decision_audit: `missed` means the agent judged it and kept
# it; `overwritten` means the imputer took it before any judgement was made. Keep the
# two pages in step — they were briefly inconsistent and reported different counts for
# the same rows.
OUTCOMES = ("caught", "deleted-natural", "missed", "overwritten", "undetected",
            "kept-natural")


# --------------------------------------------------------------------------- model
@dataclass
class Case:
    """One spike-relevant timestamp, with everything known about it."""

    at: pd.Timestamp
    outcome: str
    value: float
    label: str                      # 'spike' | 'real water' | another anomaly type
    action: str                     # the agent's decision, or '(undecided)'
    reason: str = ""
    flagged_by: str = ""
    true_value: float | None = None
    injected_sigmas: float | None = None      # |injected - true| in robust sigmas
    local_sigmas: float | None = None         # |value - local median| in robust sigmas
    raw_value: float | None = None
    raw_sigmas: float | None = None
    measurement: dict | None = None           # what describe_point(s) returned, if anything

    @property
    def measured(self) -> bool:
        return self.measurement is not None


# --------------------------------------------------------------------------- inputs
def load_run(log_path: Path) -> tuple[list[dict], dict[pd.Timestamp, dict]]:
    """Return the agent's `decisions` list and every point it actually measured.

    The measurements are what separate "judged wrong" from "never looked", which is
    the distinction the page exists to draw.
    """
    decisions: list[dict] = []
    measured: dict[pd.Timestamp, dict] = {}

    for line in log_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue                      # a killed run leaves a partial last line

        if event.get("event") == "api_response":
            for block in event["response"].get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "export_clean_data":
                    decisions = block["input"].get("decisions", []) or decisions
            continue

        if event.get("event") != "api_call":
            continue
        for message in event.get("messages") or []:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                try:
                    result = json.loads(block["content"])
                except (json.JSONDecodeError, TypeError):
                    continue              # an is_error result is plain text
                if not isinstance(result, dict):
                    continue
                if result.get("tool") == "describe_points":
                    for row in result.get("points") or []:
                        if "at" in row:
                            measured.setdefault(pd.Timestamp(row["at"]), row)
                elif result.get("tool") == "describe_point" and "at" in result:
                    measured.setdefault(pd.Timestamp(result["at"]), result)

    return decisions, measured


# §5 flag-log fields the audit pages read, in panel order. `verdict` leads because it
# is the run's answer; `action` is only what was done about it (§5.1).
_FLAG_LOG_COLUMNS: tuple[str, ...] = (
    "flagged_by", "verdict", "anomaly_type", "action", "reason",
    "rationale", "rationale_source", "deliberation", "decided_by",
)


def load_flag_log(path: Path) -> pd.DataFrame:
    """The §5 flag log as a frame indexed by timestamp."""
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a §5 flag log (expected a JSON list).")
    frame = pd.DataFrame(entries)
    if frame.empty:
        return pd.DataFrame(columns=list(_FLAG_LOG_COLUMNS))
    frame[DATETIME_COL] = pd.to_datetime(frame["datetime"])
    # These arrived in stages — rationale/rationale_source/deliberation 2026-08-11,
    # verdict/anomaly_type 2026-08-13, decided_by 2026-08-18 — so a log written before
    # any of them is read with the field blank rather than rejected.
    for column in _FLAG_LOG_COLUMNS:
        if column not in frame.columns:
            frame[column] = None if column == "decided_by" else ""
    return frame.set_index(DATETIME_COL)[list(_FLAG_LOG_COLUMNS)]


def robust_step_sigma(series: pd.Series) -> float:
    """Series-wide robust first-difference scale — the §7.3 floor.

    Used instead of a windowed MAD because a windowed MAD collapses to 0 on
    quantised turbidity and makes every local z meaningless (§7.1).
    """
    diffs = np.abs(np.diff(series.dropna().to_numpy()))
    sigma = 1.4826 * float(np.nanmedian(diffs)) if len(diffs) else 0.0
    return sigma or 1.0


def _local_sigmas(series: pd.Series, at: pd.Timestamp, sigma: float,
                  window: str = "6h") -> float | None:
    local = series.loc[at - pd.Timedelta(window): at + pd.Timedelta(window)].dropna()
    value = series.get(at)
    if local.empty or value is None or not np.isfinite(value):
        return None
    return float(abs(value - np.median(local.to_numpy())) / sigma)


# --------------------------------------------------------------------------- cases
def build_cases(
    series: pd.Series,
    labels: pd.DataFrame,
    flag_log: pd.DataFrame,
    decisions: list[dict],
    measured: dict[pd.Timestamp, dict],
    raw: pd.Series | None = None,
) -> list[Case]:
    """One Case per spike-flagged row, plus every injected spike nothing flagged.

    ``decisions`` is accepted for provenance only — the verdict comes from the flag
    log, which is what the run actually wrote.
    """
    sigma = robust_step_sigma(series)
    raw_sigma = robust_step_sigma(raw) if raw is not None else None

    is_spike_flag = flag_log["flagged_by"].fillna("").apply(
        lambda s: any(f in s for f in SPIKE_FUNCS)
    )
    flagged = pd.DatetimeIndex(flag_log.index[is_spike_flag.to_numpy()])

    # Read the verdict straight from the flag log. `export_clean_data` already
    # resolved overlapping spans (narrowest wins, §5) and this page must show what
    # the run actually recorded — re-deriving it here from `decisions` reproduced the
    # resolution rule in a second place, they drifted, and the page reported 0 deletes
    # for a run whose log contained 80 (2026-08-11).
    action = flag_log["action"].reindex(flagged).fillna("(undecided)")
    reason = flag_log["reason"].reindex(flagged).fillna("")

    label_type = labels["anomaly_type"].reindex(flagged).fillna("real water")
    cases: list[Case] = []

    for at in flagged:
        is_injected = label_type[at] == "spike"
        acted = action[at] in POSITIVE_ACTIONS
        if is_injected:
            # `missed` means the agent looked and KEPT it. A row the imputer overwrote
            # was never judged at all, and calling that a miss made this page disagree
            # with decision_audit about the same 10 rows of the same run — same word,
            # two meanings (2026-08-11). Both now split it out.
            outcome = ("caught" if acted
                       else "overwritten" if action[at] == "impute"
                       else "missed")
        else:
            outcome = "deleted-natural" if acted else "kept-natural"

        true_value = labels["true_value"].get(at)
        true_value = float(true_value) if pd.notna(true_value) else None
        value = float(series.get(at, np.nan))

        case = Case(
            at=at,
            outcome=outcome,
            value=value,
            label=str(label_type[at]),
            action=str(action[at]),
            reason=str(reason[at]),
            flagged_by=str(flag_log["flagged_by"].get(at, "")),
            true_value=true_value,
            injected_sigmas=(
                abs(value - true_value) / sigma
                if true_value is not None and np.isfinite(value) else None
            ),
            local_sigmas=_local_sigmas(series, at, sigma),
            measurement=measured.get(at),
        )
        if raw is not None and at in raw.index:
            case.raw_value = float(raw[at])
            case.raw_sigmas = _local_sigmas(raw, at, raw_sigma)
        cases.append(case)

    # Injected spikes no spike detector reached — a detector recall gap, not a
    # decision error, and invisible on any page built only from what was flagged.
    for at in labels.index[labels["anomaly_type"] == "spike"]:
        if at in set(flagged):
            continue
        true_value = labels["true_value"].get(at)
        true_value = float(true_value) if pd.notna(true_value) else None
        value = float(series.get(at, np.nan))
        cases.append(Case(
            at=at,
            outcome="undetected",
            value=value,
            label="spike",
            action="(never flagged)",
            reason="No spike detector flagged this row, so the agent never saw it.",
            true_value=true_value,
            injected_sigmas=(
                abs(value - true_value) / sigma
                if true_value is not None and np.isfinite(value) else None
            ),
            local_sigmas=_local_sigmas(series, at, sigma),
        ))

    cases.sort(key=lambda c: (OUTCOMES.index(c.outcome), c.at))
    return cases


# --------------------------------------------------------------------------- payload
def _round(x, n=3):
    return None if x is None or not np.isfinite(x) else round(float(x), n)


def build_payload(series: pd.Series, labels: pd.DataFrame, cases: list[Case],
                  title: str, has_raw: bool) -> dict:
    index = pd.DatetimeIndex(series.index)
    # Epoch milliseconds with the naive timestamps read AS UTC, so that JS's
    # toISOString() renders the same wall-clock string back. Handing Plotly a Date
    # built any other way prints hours away from the timestamps in the side panel —
    # verified in a browser, and silent (§9.1).
    #
    # `.as_unit("ms")` and NOT `.view("int64") // 1_000_000`: these CSVs parse to
    # `datetime64[us]`, not the nanoseconds that idiom assumes, so the division
    # yielded SECONDS and the whole series plotted at 1970 — caught in a browser,
    # because the axis is the only place it shows.
    epoch_ms = index.as_unit("ms").astype("int64").tolist()
    position = {ts: i for i, ts in enumerate(index)}

    payload_cases = []
    for case in cases:
        payload_cases.append({
            "at": case.at.strftime("%Y-%m-%dT%H:%M:%S"),
            "i": position.get(case.at, -1),
            "outcome": case.outcome,
            "value": _round(case.value),
            "label": case.label,
            "action": case.action,
            "reason": case.reason,
            "flagged_by": case.flagged_by,
            "measured": case.measured,
            "measurement": case.measurement,
            "true_value": _round(case.true_value),
            "injected_sigmas": _round(case.injected_sigmas, 1),
            "local_sigmas": _round(case.local_sigmas, 1),
            "raw_value": _round(case.raw_value),
            "raw_sigmas": _round(case.raw_sigmas, 1),
        })

    counts = {o: sum(1 for c in cases if c.outcome == o) for o in OUTCOMES}
    unmeasured_misses = sum(
        1 for c in cases if c.outcome == "missed" and not c.measured
    )
    natural_in_raw = sum(
        1 for c in cases
        if c.outcome == "deleted-natural" and (c.raw_sigmas or 0) >= 5
    )
    return {
        "title": title,
        "epoch_ms": epoch_ms,
        "values": [None if not np.isfinite(v) else round(float(v), 3)
                   for v in series.to_numpy()],
        "cases": payload_cases,
        "counts": counts,
        "has_raw": has_raw,
        "unmeasured_misses": unmeasured_misses,
        "natural_in_raw": natural_in_raw,
        "context": context_samples(index),
    }


def _plotly_js() -> str:
    """Inline plotly.js so the page works with no network and no CDN."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def build_html(payload: dict) -> str:
    return _TEMPLATE.replace("/*PLOTLY_JS*/", _plotly_js()).replace(
        "/*PAYLOAD*/", json.dumps(payload)
    )


# --------------------------------------------------------------------------- page
_TEMPLATE = r"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Spike decision audit</title>
<script>/*PLOTLY_JS*/</script>
<style>
  :root {
    --bg:#0f1115; --panel:#171a21; --line:#242835; --text:#e6e9ef; --dim:#9aa3b2;
    --caught:#22c55e; --deleted-natural:#ef4444; --missed:#f59e0b;
    --undetected:#a855f7; --kept-natural:#64748b;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text);
         font:14px/1.5 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif;
         display:grid; grid-template-columns:340px 1fr; height:100vh; }
  #side { border-right:1px solid var(--line); overflow-y:auto; background:var(--panel); }
  #main { display:grid; grid-template-rows:auto 1fr auto; overflow:hidden; }
  h1 { font-size:15px; margin:0; padding:14px 16px; border-bottom:1px solid var(--line); }
  h1 small { color:var(--dim); font-weight:400; display:block; margin-top:3px; font-size:12px; }
  .group { padding:10px 16px 4px; color:var(--dim); font-size:11px; letter-spacing:.09em;
           text-transform:uppercase; display:flex; justify-content:space-between; }
  .case { padding:7px 16px; cursor:pointer; border-left:3px solid transparent;
          display:flex; justify-content:space-between; gap:8px; font-variant-numeric:tabular-nums; }
  .case:hover { background:#1e222c; }
  .case.sel { background:#252b39; border-left-color:currentColor; }
  .case .t { color:var(--text); font-size:12.5px; }
  .case .m { font-size:11px; color:var(--dim); }
  .flag { color:#f59e0b; font-weight:600; }
  #head { padding:12px 18px; border-bottom:1px solid var(--line); display:flex;
          gap:22px; align-items:baseline; flex-wrap:wrap; }
  #head .k { color:var(--dim); font-size:11px; text-transform:uppercase; letter-spacing:.08em; }
  #head .v { font-size:19px; font-variant-numeric:tabular-nums; }
  #plot { min-height:0; }
  #evidence { border-top:1px solid var(--line); padding:12px 18px; overflow-y:auto;
              max-height:38vh; background:var(--panel); }
  .row { display:grid; grid-template-columns:150px 1fr; gap:10px; padding:3px 0;
         border-bottom:1px solid #1d212b; }
  .row .k { color:var(--dim); }
  .verdict { padding:9px 12px; border-radius:6px; margin-bottom:10px; line-height:1.45; }
  .verdict.warn { background:#3b2a10; border:1px solid #7c5310; }
  .verdict.bad  { background:#3a1a1a; border:1px solid #7f1d1d; }
  .verdict.ok   { background:#12261a; border:1px solid #14532d; }
  pre { margin:8px 0 0; padding:9px 11px; background:#0c0e13; border:1px solid var(--line);
        border-radius:6px; overflow-x:auto; font-size:11.5px; color:#cbd5e1; }
  kbd { background:#0c0e13; border:1px solid var(--line); border-radius:4px;
        padding:1px 5px; font-size:11px; }
</style>
<body>
<div id="side"></div>
<div id="main">
  <div id="head"></div>
  <div id="plot"></div>
  <div id="evidence"></div>
</div>
<script>
const D = /*PAYLOAD*/;
const COLOR = {
  "caught":"#22c55e", "deleted-natural":"#ef4444", "missed":"#f59e0b",
  "overwritten":"#fb923c", "undetected":"#a855f7", "kept-natural":"#64748b",
};
const BLURB = {
  "caught": "Injected spike, acted on. Working as intended.",
  "deleted-natural": "Deleted a row the labels call normal water. Scored as a false positive — but the labels record none of the base's own spikes.",
  "missed": "Injected spike the agent kept. Check whether it was ever measured.",
  "overwritten": "impute_rolling overwrote this injected spike before the agent judged it.",
  "undetected": "No spike detector flagged this row. Upstream of the agent — a detector recall gap.",
  "kept-natural": "Normal water, correctly kept.",
};
const $ = (id) => document.getElementById(id);
const iso = (ms) => new Date(ms).toISOString().slice(0, 19);   // naive round-trip
const fmt = (v, n=2) => (v === null || v === undefined) ? "&mdash;" : (+v).toFixed(n);

let showNatural = false;
let sel = 0;

function visible() {
  return D.cases.filter(c => showNatural || c.outcome !== "kept-natural");
}

function renderSide() {
  const parts = [`<h1>Spike decision audit<small>${D.title}</small></h1>`];
  for (const outcome of Object.keys(COLOR)) {
    const rows = visible().filter(c => c.outcome === outcome);
    if (!rows.length && outcome !== "kept-natural") continue;
    parts.push(`<div class="group"><span style="color:${COLOR[outcome]}">${outcome}</span>
                <span>${D.counts[outcome]}</span></div>`);
    if (outcome === "kept-natural" && !showNatural) {
      parts.push(`<div class="case"><span class="m">hidden &mdash; press <kbd>A</kbd></span></div>`);
      continue;
    }
    for (const c of rows) {
      const i = visible().indexOf(c);
      const note = c.outcome === "missed" && !c.measured ? `<span class="flag">never measured</span>`
                 : c.outcome === "deleted-natural" && c.raw_sigmas >= 5 ? `<span class="flag">${fmt(c.raw_sigmas,1)}σ in raw</span>`
                 : `${fmt(c.value)}`;
      parts.push(
        `<div class="case ${i === sel ? "sel" : ""}" style="color:${COLOR[outcome]}"
              onclick="select(${i})">
           <span class="t">${c.at.replace("T", " ")}</span><span class="m">${note}</span>
         </div>`);
    }
  }
  $("side").innerHTML = parts.join("");
}

function verdict(c) {
  if (c.outcome === "deleted-natural") {
    if (D.has_raw && c.raw_sigmas >= 5)
      return [`bad`, `This excursion is <b>${fmt(c.raw_sigmas,1)}σ in the raw approved base</b>
        and injection did not touch it. It is a real sensor spike the label file never
        recorded, so the agent's delete is defensible and the "false positive" is the
        ground truth's gap, not the agent's error.`];
    if (D.has_raw)
      return [`warn`, `Only ${fmt(c.raw_sigmas,1)}σ in the raw base — this one does look
        like a genuine deletion of real water.`];
    return [`warn`, `Pass <code>--raw</code> to see whether this excursion exists in the
      approved base. Without it, a mislabelled base spike and a real error look identical.`];
  }
  if (c.outcome === "missed")
    return c.measured
      ? [`warn`, `Measured, and still kept. The evidence below is what the agent had —
          read it against the decision reason to see whether the measurement was wrong or
          the inference was.`]
      : [`bad`, `<b>Never measured.</b> No describe_point(s) call covered this timestamp,
          so a broad decision span swept it up unexamined. This is a coverage failure, not
          a judgement one.`];
  if (c.outcome === "overwritten")
    return [`bad`, `<b>impute_rolling overwrote this injected spike</b> before the agent
      judged it — SaQC filters already-flagged rows out of the imputer's input, so it
      treats them as gaps and fills them (§7.1). No decision was ever recorded; the
      value is simply gone.`];
  if (c.outcome === "undetected")
    return [`bad`, `No spike detector flagged this row at all (${fmt(c.injected_sigmas,1)}σ
      injected). Nothing the agent does can recover it — this is detector tuning.`];
  if (c.outcome === "caught") return [`ok`, BLURB[c.outcome]];
  return [`ok`, BLURB[c.outcome]];
}

function renderCase() {
  const rows = visible();
  if (!rows.length) return;
  sel = Math.max(0, Math.min(sel, rows.length - 1));
  const c = rows[sel];

  $("head").innerHTML = `
    <div><div class="k">case</div><div class="v">${sel + 1} / ${rows.length}</div></div>
    <div><div class="k">outcome</div>
         <div class="v" style="color:${COLOR[c.outcome]}">${c.outcome}</div></div>
    <div><div class="k">timestamp</div><div class="v">${c.at.replace("T", " ")}</div></div>
    <div><div class="k">value</div><div class="v">${fmt(c.value)}</div></div>
    <div><div class="k">decision</div><div class="v">${c.action}</div></div>`;

  // --- plot: context window with the case marked -------------------------------
  const lo = Math.max(0, c.i - D.context), hi = Math.min(D.epoch_ms.length, c.i + D.context);
  const x = [], y = [];
  for (let i = lo; i < hi; i++) { x.push(iso(D.epoch_ms[i])); y.push(D.values[i]); }
  const traces = [
    { x, y, mode: "lines+markers", type: "scatter", name: "series",
      line: { color: "#8b93a5", width: 1.4 }, marker: { size: 3.5, color: "#8b93a5" },
      connectgaps: false },
    { x: [c.at], y: [c.value], mode: "markers", type: "scatter", name: c.outcome,
      marker: { color: COLOR[c.outcome], size: 13, symbol: "circle-open",
                line: { width: 3, color: COLOR[c.outcome] } } },
  ];
  if (c.true_value !== null && c.true_value !== undefined && c.label === "spike") {
    traces.push({ x: [c.at], y: [c.true_value], mode: "markers", type: "scatter",
      name: "true value", marker: { color: "#38bdf8", size: 9, symbol: "x" } });
  }
  Plotly.react("plot", traces, {
    margin: { l: 52, r: 16, t: 10, b: 34 }, paper_bgcolor: "#0f1115",
    plot_bgcolor: "#0f1115", font: { color: "#9aa3b2", size: 11 },
    xaxis: { gridcolor: "#1d212b" }, yaxis: { gridcolor: "#1d212b", title: "FNU" },
    showlegend: true, legend: { orientation: "h", y: 1.12, font: { size: 10 } },
  }, { displayModeBar: false, responsive: true });
  wireClick();

  // --- evidence ------------------------------------------------------------------
  const [cls, text] = verdict(c);
  const row = (k, v) => `<div class="row"><div class="k">${k}</div><div>${v}</div></div>`;
  let html = `<div class="verdict ${cls}">${text}</div>`;
  html += row("label", c.label);
  if (c.label === "spike")
    html += row("injected size", `${fmt(c.injected_sigmas,1)}σ &nbsp;
                 (true value ${fmt(c.true_value)} &rarr; ${fmt(c.value)})`);
  html += row("displacement", `${fmt(c.local_sigmas,1)}σ from the local median`);
  if (D.has_raw)
    html += row("raw approved base", c.raw_value === null || c.raw_value === undefined
      ? "not present in the raw file"
      : `${fmt(c.raw_value)} &nbsp; (${fmt(c.raw_sigmas,1)}σ excursion there)`);
  html += row("flagged by", c.flagged_by || "&mdash;");
  html += row("agent measured it", c.measured
    ? "yes" : `<span class="flag">no — never inspected</span>`);
  html += row("decision reason", c.reason || "&mdash;");
  if (c.measurement)
    html += `<pre>${JSON.stringify(c.measurement, null, 2)}</pre>`;
  $("evidence").innerHTML = html;

  renderSide();
  const el = document.querySelector(".case.sel");
  if (el) el.scrollIntoView({ block: "nearest" });
}

function select(i) { sel = i; renderCase(); }

// Plotly only adds `.on` to the div once it has plotted, so this cannot be wired
// at start-up — doing so throws and takes the rest of init down with it (§9.1).
let clickWired = false;
function wireClick() {
  const gd = $("plot");
  if (clickWired || typeof gd.on !== "function") return;
  clickWired = true;
  gd.on("plotly_click", (ev) => {
    const p = ev.points && ev.points[0];
    if (!p) return;
    const i = visible().findIndex(c => c.at === p.x);
    if (i >= 0) select(i);
  });
}

addEventListener("keydown", (e) => {
  const rows = visible();
  if (e.key === "j" || e.key === "ArrowDown" || e.key === "ArrowRight") { sel++; renderCase(); }
  else if (e.key === "k" || e.key === "ArrowUp" || e.key === "ArrowLeft") { sel--; renderCase(); }
  else if (e.key === "a" || e.key === "A") { showNatural = !showNatural; sel = 0; renderCase(); }
  else if (e.key === "r" || e.key === "R") { renderCase(); }
  else if (/^[1-6]$/.test(e.key)) {
    const want = Object.keys(COLOR)[+e.key - 1];
    const i = rows.findIndex(c => c.outcome === want);
    if (i >= 0) { sel = i; renderCase(); }
  }
});

renderCase();
</script>
"""


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit one run's spike decisions case by case, with the evidence."
    )
    parser.add_argument("series", type=Path, help="Injected dataset CSV; labels are read beside it.")
    parser.add_argument("--flags", type=Path, required=True, help="§5 flag log (*_flags.json).")
    parser.add_argument("--log", type=Path, required=True, help="Agent run log (logs/run_*.jsonl).")
    parser.add_argument(
        "--raw", type=Path, default=None,
        help="Raw APPROVED base CSV. Optional but strongly recommended: without it a "
             "deleted base spike cannot be told from a genuine false positive.",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    args = parser.parse_args(argv)

    series_df = pd.read_csv(args.series, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
    series = series_df.set_index(DATETIME_COL)["value"]

    labels_path = args.series.with_name(f"{args.series.stem}_labels.csv")
    if not labels_path.exists():
        parser.error(f"labels not found beside the series: {labels_path}")
    labels = pd.read_csv(labels_path, parse_dates=[DATETIME_COL]).set_index(DATETIME_COL)

    raw = None
    if args.raw is not None:
        raw_df = pd.read_csv(args.raw, parse_dates=[DATETIME_COL]).sort_values(DATETIME_COL)
        raw = raw_df.set_index(DATETIME_COL)["value"]

    decisions, measured = load_run(args.log)
    cases = build_cases(series, labels, load_flag_log(args.flags), decisions, measured, raw)

    payload = build_payload(
        series, labels, cases,
        title=f"{args.series.stem} · {args.log.name}",
        has_raw=raw is not None,
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / f"{args.series.stem}_{args.log.stem}_spikes.html"
    out.write_text(build_html(payload))

    counts = payload["counts"]
    print(f"  {len(cases)} spike cases: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if payload["unmeasured_misses"]:
        print(f"  {payload['unmeasured_misses']} of {counts['missed']} missed spikes were "
              f"NEVER MEASURED — a coverage failure, not a judgement one.")
    if payload["has_raw"] and payload["natural_in_raw"]:
        print(f"  {payload['natural_in_raw']} of {counts['deleted-natural']} 'false positives' "
              f"are >=5σ excursions in the RAW APPROVED BASE — real spikes the labels omit.")
    elif not payload["has_raw"]:
        print("  NOTE: no --raw given, so deleted-natural cases cannot be checked against "
              "the approved base — the most useful column on the page is missing.")
    print(f"  wrote {out}")

    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
