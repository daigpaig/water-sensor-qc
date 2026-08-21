"""Build a click-through page for one agent run in ``logs/*.jsonl``.

An agent run is a sequence of ReAct iterations (CLAUDE.md §8), and the JSONL log
records every one of them — but as raw API payloads, which is unreadable. This
module turns a log into a single self-contained HTML page: a list of iterations
on the left, the series on the right, and **the points that iteration flagged
drawn on the series**. Clicking (or arrowing) through the steps replays the run.

Like ``src.workbench.review``, the page is one HTML file with plotly.js inlined —
no server, no Streamlit, works offline.

Two different things get called "the agent's reasoning", and the page separates
them:

* **Prose** — the text the model writes for the reader between tool calls. Always
  logged (it is a ``text`` block in the response), always shown.
* **Thinking** — the model's extended-thinking trace. Only exists if the run
  *asked* for it: ``agent.py`` must pass ``thinking={"type": "adaptive"}`` to
  ``messages.create``. Without that parameter no thinking blocks are produced,
  so there is nothing in the log to display and the panel stays hidden.

What each iteration shows:

* the agent's own reasoning text for that turn, and its thinking trace if present,
* the tool it called and the exact parameters it passed,
* the ``message`` the tool returned (which carries caveats nothing else reports),
* its flagged timestamps, drawn as rings over the series,
* every flag from earlier iterations, in grey rings, so you can see what is new,
* optionally the §5 ground-truth labels, so you can see what it *should* have hit,
* optionally the run's FINAL VERDICT per row, read from the §5 flag log.

A flag and a verdict are different claims and the page keeps them apart. A flag says
a detector fired; the verdict is what the run concluded about that row after looking
at it, and §10 scores the verdict alone. They are drawn as separate layers for that
reason — the interesting rows are the ones where they disagree, and a page that drew
only flags could not show a run that flagged 2,595 rows and concluded nothing.

Truth and action are kept in separate visual channels: **truth is a solid dot**,
coloured by anomaly type, and **a flag is a hollow ring** in a colour no type
uses. Caught, missed and false-positive are then readable without a legend — dot
in a ring, bare dot, bare ring. Gaps carry no value to plot at, so gap labels and
``flag_nan`` hits are drawn as ticks pinned along the bottom of the plot rather
than silently vanishing.

Keys: ``<-``/``->`` (or ``J``/``K``) step, ``C`` cumulative flags, ``G`` ground
truth, ``V`` final verdicts, ``Z`` zoom to this step's flags, ``R`` whole series.

CLI
---
    # newest log, series auto-detected from the inspect_dataset summary
    python -m src.workbench.visualize_log

    # a specific log and series
    python -m src.workbench.visualize_log logs/run_20260731_133616.jsonl \
        --series data/injected/02054550/l1/02054550_l1.csv

The log does not record which dataset was run, so the series is matched by row
count and time range against ``data/injected`` when ``--series`` is omitted; pass
it explicitly for anything outside that tree. Ground-truth labels are picked up
automatically from ``<stem>_labels.csv`` beside the series (§5) when present.

Pages land in ``figures/logs/``.
"""
from __future__ import annotations

import argparse
import json
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.workbench.visualize_injected import ANOMALY_COLORS

DEFAULT_LOG_DIR = Path("logs")
DEFAULT_INJECTED_DIR = Path("data/injected")
DEFAULT_OUTDIR = Path("figures/logs")

DATETIME_COL = "datetime"
VALUE_COL = "value"

# Result keys that are bulk data, rendered on the plot rather than in the panel.
_BULK_KEYS = frozenset({"tool", "params", "message", "flagged_datetimes", "points", "gaps"})

# Which part of the loop a tool belongs to (§7), used only for the page's colour
# coding of the step list.
TOOL_KINDS: dict[str, str] = {
    "inspect_dataset": "utility",
    "get_flag_summary": "utility",
    "export_clean_data": "utility",
    "flag_range": "detect",
    "flag_constants": "detect",
    "flag_plateau": "detect",
    "flag_spike_unilof": "detect",
    "flag_zscore": "detect",
    "flag_jumps": "detect",
    "flag_nan": "detect",
    "impute_rolling": "action",
    "describe_point": "context",
    "describe_points": "context",
}


# --------------------------------------------------------------------------- log model
@dataclass
class ToolCall:
    """One ``tool_use`` block and the ``tool_result`` we sent back for it."""

    id: str
    name: str
    params: dict
    message: str = ""
    is_error: bool = False
    error: str = ""
    flagged: list[str] = field(default_factory=list)
    context_points: list[dict] = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def kind(self) -> str:
        return TOOL_KINDS.get(self.name, "other")


@dataclass
class Step:
    """One iteration of the ReAct loop: one API response and its tool results.

    ``text`` is the prose the model wrote for the user; ``thinking`` is its
    extended-thinking trace, which only exists if the run asked for it (see the
    module docstring).
    """

    index: int
    text: str = ""
    thinking: str = ""
    stop_reason: str | None = None
    usage: dict = field(default_factory=dict)
    calls: list[ToolCall] = field(default_factory=list)


@dataclass
class RunLog:
    path: Path
    prompt_version: str | None
    model: str | None
    steps: list[Step]
    max_steps_reached: bool
    summary: dict | None  # the inspect_dataset summary, if the run called it
    # Where export_clean_data wrote the §5 flag log, read from the run_summary
    # event. The verdicts are NOT in this JSONL: the log records the `decisions`
    # SPANS the agent passed, but §5 resolves those per row (narrowest span wins,
    # uncovered rows become `undecided`), and that resolution happens inside
    # export_clean_data. Re-deriving it here would be a second implementation of
    # the rule that could disagree with the artefact, so the page reads the
    # artefact instead. None if the run never exported.
    flags_path: str | None = None


def load_events(path: Path) -> list[dict]:
    """Read a JSONL log, skipping blank and unparseable lines.

    A run that dies mid-write can leave a truncated final line; that is worth
    visualising, not worth crashing on.
    """
    events: list[dict] = []
    for i, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            print(f"  warning: {path.name} line {i} is not valid JSON, skipped")
    return events


def _result_blocks(events: list[dict]) -> tuple[dict[str, dict], dict[int, list[str]]]:
    """Map ``tool_use_id -> tool_result block``, and step index -> its call ids.

    ``agent.py`` appends the whole running ``messages`` list to every ``api_call``
    event, so the same tool_result appears in many events; the last message of
    the ``api_call`` at step *k* holds exactly the results produced by step
    *k-1*'s tool calls. That structural link is what pairs a call to its step,
    and it holds for any log — including ones whose ``api_response`` payload is
    not a full Messages API dump.
    """
    by_id: dict[str, dict] = {}
    ids_by_step: dict[int, list[str]] = {}

    for event in events:
        if event.get("event") != "api_call":
            continue
        messages = event.get("messages") or []
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    by_id[block["tool_use_id"]] = block

        last = messages[-1] if messages else None
        step = event.get("step")
        if (
            isinstance(last, dict)
            and last.get("role") == "user"
            and isinstance(last.get("content"), list)
            and isinstance(step, int)
            and step > 0
        ):
            ids = [
                b["tool_use_id"]
                for b in last["content"]
                if isinstance(b, dict) and b.get("type") == "tool_result"
            ]
            if ids:
                ids_by_step[step - 1] = ids

    return by_id, ids_by_step


def _fill_from_result(call: ToolCall, block: dict) -> None:
    """Populate a :class:`ToolCall` from the tool_result we returned to the model."""
    content = block.get("content")
    if block.get("is_error"):
        call.is_error = True
        call.error = str(content)
        if call.name == "?" and isinstance(content, str) and "Error executing " in content:
            call.name = content.split("Error executing ", 1)[1].split(":", 1)[0].strip()
        return

    try:
        result = json.loads(content) if isinstance(content, str) else content
    except json.JSONDecodeError:
        call.message = str(content)[:500]
        return
    if not isinstance(result, dict):
        call.message = str(content)[:500]
        return

    if call.name == "?":
        call.name = str(result.get("tool", "?"))
    if not call.params:
        call.params = result.get("params") or {}
    call.message = str(result.get("message", ""))
    call.flagged = [str(t) for t in (result.get("flagged_datetimes") or [])]
    call.context_points = [p for p in (result.get("points") or []) if isinstance(p, dict)]
    call.extra = {k: v for k, v in result.items() if k not in _BULK_KEYS}


def parse_run(events: list[dict], path: Path) -> RunLog:
    """Turn raw log events into the per-iteration structure the page renders."""
    prompt_version = None
    for event in events:
        if event.get("event") == "system_prompt":
            prompt_version = event.get("version")
            break

    by_id, ids_by_step = _result_blocks(events)
    model = None
    steps: list[Step] = []

    for event in events:
        if event.get("event") != "api_response":
            continue
        response = event.get("response")
        response = response if isinstance(response, dict) else {}
        index = event.get("step", len(steps))
        model = response.get("model") or model

        step = Step(
            index=index,
            stop_reason=response.get("stop_reason"),
            usage=response.get("usage") or {},
        )

        # Preferred source for names/params: the model's own tool_use blocks.
        texts: list[str] = []
        thoughts: list[str] = []
        use_blocks: dict[str, dict] = {}
        content = response.get("content")
        if isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue  # a non-dict here means the log stringified the block
                kind = block.get("type")
                if kind == "text":
                    texts.append(str(block.get("text", "")))
                elif kind == "thinking":
                    # Present only when the run requested thinking. The field is
                    # an empty string when display was "omitted", so an empty
                    # block is not the same as no block.
                    thoughts.append(str(block.get("thinking", "")))
                elif kind == "redacted_thinking":
                    thoughts.append("[redacted thinking block]")
                elif kind == "tool_use":
                    use_blocks[str(block.get("id"))] = block
        step.text = "\n".join(t for t in texts if t).strip()
        step.thinking = "\n\n".join(t for t in thoughts if t).strip()

        # Ids come from the structural pairing so the step list is right even
        # when the response payload carries no content blocks.
        ids = ids_by_step.get(index) or list(use_blocks)
        for call_id in ids:
            use = use_blocks.get(call_id, {})
            call = ToolCall(
                id=call_id,
                name=str(use.get("name", "?")),
                params=dict(use.get("input") or {}),
            )
            block = by_id.get(call_id)
            if block is not None:
                _fill_from_result(call, block)
            step.calls.append(call)

        steps.append(step)

    summary = None
    for step in steps:
        for call in step.calls:
            if call.name == "inspect_dataset" and isinstance(call.extra.get("summary"), dict):
                summary = call.extra["summary"]
                break

    flags_path = None
    for event in events:
        if event.get("event") == "run_summary" and event.get("flags_path"):
            flags_path = str(event["flags_path"])

    return RunLog(
        path=path,
        prompt_version=prompt_version,
        model=model,
        steps=steps,
        max_steps_reached=any(e.get("event") == "max_steps_reached" for e in events),
        summary=summary,
        flags_path=flags_path,
    )


# --------------------------------------------------------------------------- series
def load_series(path: Path) -> pd.Series:
    """Load a §5 dataset CSV as a sorted, datetime-indexed value series."""
    df = pd.read_csv(path, parse_dates=[DATETIME_COL])
    series = df.set_index(DATETIME_COL)[VALUE_COL].sort_index()
    return series


def load_flag_log(path: Path) -> list[dict]:
    """Load a §5 flag log (``*_flags.json``): one entry per flagged row.

    Returns the entries as written. Raises if the file is not a list, because a
    silently-empty verdict layer is worse than a loud failure — the whole point
    of the layer is to show what the run concluded, and an empty one reads as
    "the agent decided nothing", which is a real and different outcome (§5).
    """
    entries = json.loads(path.read_text())
    if not isinstance(entries, list):
        raise ValueError(f"{path} is not a §5 flag log (expected a list of entries).")
    return entries


def load_labels(series_path: Path) -> pd.DataFrame | None:
    """Load the §5 labels file beside the series, if it exists.

    Labels live *next to* the series by contract (§5), so this is a
    ``with_name`` lookup and never a search.
    """
    path = series_path.with_name(f"{series_path.stem}_labels.csv")
    if not path.exists():
        return None
    labels = pd.read_csv(path, parse_dates=[DATETIME_COL])
    return labels[labels["is_anomaly"].astype(bool)]


def _fingerprint(series: pd.Series) -> tuple:
    """The identifying numbers ``inspect_dataset`` reports, for series matching."""
    values = series.to_numpy(dtype=float)
    return (
        len(series),
        pd.Timestamp(series.index[0]).isoformat(),
        pd.Timestamp(series.index[-1]).isoformat(),
        int(np.isnan(values).sum()),
        round(float(np.nanmean(values)), 6),
    )


def autodetect_series(run: RunLog, injected_dir: Path = DEFAULT_INJECTED_DIR) -> list[Path]:
    """Datasets whose fingerprint matches the run's own ``inspect_dataset`` summary.

    ``agent.py`` does not log which file it was given, so the series has to be
    recovered from what the run itself reported. Row count and time range alone
    are **not** enough — every injected dataset shares them, since all nine are
    the same 2-year 15-min window — so the NaN count and the mean are matched
    too. Returns every match: more than one means the run cannot be attributed,
    and the caller should ask for ``--series`` rather than guess.
    """
    if not run.summary:
        return []
    column = (run.summary.get("columns") or {}).get(run.summary.get("value_column", VALUE_COL), {})
    if run.summary.get("n_rows") is None or "mean" not in column:
        return []
    want = (
        run.summary["n_rows"],
        run.summary.get("time_start"),
        run.summary.get("time_end"),
        run.summary.get("n_nan"),
        round(float(column["mean"]), 6),
    )

    matches = []
    for path in sorted(injected_dir.glob("*/l[1-3]/*_l[1-3].csv")):
        try:
            got = _fingerprint(load_series(path))
        except (KeyError, ValueError, IndexError):
            continue
        if all(w is None or g == w for g, w in zip(got, want)):
            matches.append(path)
    return matches


# --------------------------------------------------------------------------- payload
def _positions(index: pd.DatetimeIndex, stamps: list[str]) -> list[int]:
    """Row positions for a list of ISO timestamps; unmatched stamps are dropped.

    A detector reports timestamps, the page draws row positions — integers keep
    the payload small (a gap detector can return thousands of stamps).
    """
    if not stamps:
        return []
    wanted = pd.to_datetime(pd.Series(stamps), errors="coerce")
    found = index.get_indexer(pd.DatetimeIndex(wanted.dropna()))
    return sorted(int(i) for i in found if i >= 0)


def _grid(index: pd.DatetimeIndex) -> tuple[int, int, bool]:
    """Return ``(t0_ms, step_ms, is_regular)`` for a datetime index.

    A regular grid lets the page rebuild timestamps from ``t0 + i * step``
    instead of shipping tens of thousands of ISO strings.

    Deltas are taken as Timedeltas rather than raw int64: pandas 3 stores these
    indexes as ``datetime64[us]``, so a ``.view("int64")`` is microseconds, not
    the nanoseconds the old pandas idiom assumed — a silent factor-of-1000.
    ``Timestamp.value`` is always nanoseconds, so ``t0`` is safe as written.
    """
    t0 = int(pd.Timestamp(index[0]).value // 1_000_000)
    if len(index) < 2:
        return t0, 0, True
    deltas = np.asarray(np.diff(index).astype("timedelta64[ms]"), dtype="int64")
    step = int(np.median(deltas))
    return t0, step, bool(step > 0 and np.all(deltas == step))


def _verdict_layer(index: pd.DatetimeIndex, entries: list[dict]) -> dict:
    """Group §5 flag-log entries into the page's verdict layer.

    Keyed by the category the page draws: ``anomaly:<type>`` for a row the agent
    called faulty (so the marker can take the same colour the truth dot uses for
    that type, and a mis-classification shows up as two colours on one point),
    plus ``normal`` and ``undecided``.

    `undecided` is kept as its own category rather than folded into `normal`,
    because §5 is explicit that they are not the same thing: one is a judgement,
    the other is a row nobody looked at, and a page that drew them alike would
    hide exactly the failure the field was added to expose.
    """
    by_stamp: dict[str, dict] = {}
    for entry in entries:
        stamp = entry.get("datetime")
        if stamp:
            by_stamp[str(stamp)] = entry

    positions = index.get_indexer(pd.DatetimeIndex(pd.to_datetime(
        pd.Series(list(by_stamp)), errors="coerce"
    ).dropna()))

    layer: dict[str, list[dict]] = {}
    for stamp, pos in zip(by_stamp, positions):
        if pos < 0:
            continue                       # a log written against a different series
        entry = by_stamp[stamp]
        verdict = str(entry.get("verdict") or "undecided")
        atype = str(entry.get("anomaly_type") or "")
        key = f"anomaly:{atype or 'untyped'}" if verdict == "anomaly" else verdict
        layer.setdefault(key, []).append({
            "i": int(pos),
            "a": str(entry.get("action") or ""),
            "t": atype,
            "by": str(entry.get("flagged_by") or ""),
            "src": str(entry.get("rationale_source") or ""),
            # The reason is the only free text here and logs run to thousands of
            # rows, so it is clipped: the panel is for orientation, and
            # decision_audit.py is the tool for reading one decision in full.
            "r": str(entry.get("reason") or "")[:180],
        })
    return layer


def build_payload(
    run: RunLog,
    series: pd.Series,
    series_path: Path,
    labels: pd.DataFrame | None = None,
    flag_entries: list[dict] | None = None,
) -> dict:
    """JSON payload embedded in the page."""
    index = pd.DatetimeIndex(series.index)
    t0, step_ms, regular = _grid(index)
    values = [None if not np.isfinite(v) else round(float(v), 6) for v in series.to_numpy()]

    steps = []
    for step in run.steps:
        calls = []
        for call in step.calls:
            idx = _positions(index, call.flagged)
            ctx = []
            for point in call.context_points:
                for pos in _positions(index, [str(point.get("at"))]):
                    ctx.append({"i": pos, "reads_like": str(point.get("reads_like", ""))})
            calls.append(
                {
                    "name": call.name,
                    "kind": call.kind,
                    "params": call.params,
                    "message": call.message,
                    "is_error": call.is_error,
                    "error": call.error,
                    "idx": idx,
                    "n_flagged": len(idx),
                    "n_reported": len(call.flagged),
                    "ctx": ctx,
                    "extra": call.extra,
                }
            )
        steps.append(
            {
                "i": step.index,
                "text": step.text,
                "thinking": step.thinking,
                "stop_reason": step.stop_reason,
                "usage": step.usage,
                "calls": calls,
            }
        )

    truth: dict[str, list[int]] = {}
    if labels is not None and len(labels):
        for anomaly_type, group in labels.groupby("anomaly_type"):
            stamps = [t.isoformat() for t in pd.DatetimeIndex(group[DATETIME_COL])]
            positions = _positions(index, stamps)
            if positions:
                truth[str(anomaly_type)] = positions

    return {
        "log": run.path.name,
        "series": series_path.name,
        "model": run.model,
        "prompt_version": run.prompt_version,
        "max_steps_reached": run.max_steps_reached,
        "t0": t0,
        "step_ms": step_ms,
        "n": len(series),
        # Only shipped for an irregular index; on a grid the page rebuilds them.
        "times": None if regular else [t.isoformat(sep=" ") for t in index],
        "values": values,
        "steps": steps,
        "truth": truth,
        "truth_colors": {k: v for k, v in ANOMALY_COLORS.items()},
        "verdicts": _verdict_layer(index, flag_entries) if flag_entries else {},
        "flags_file": Path(run.flags_path).name if run.flags_path else None,
    }


def _plotly_js() -> str:
    """Inline plotly.js so the page works with no network and no CDN."""
    from plotly.offline import get_plotlyjs

    return get_plotlyjs()


def build_html(payload: dict) -> str:
    """Render the self-contained replay page."""
    body = json.dumps(payload, allow_nan=False, default=str)
    # Only sequence that could end the host <script> element early.
    body = body.replace("</", "<\\/")
    return (
        _TEMPLATE
        .replace("/*PLOTLY_JS*/", _plotly_js())
        .replace("/*PAYLOAD*/", body)
        .replace("__TITLE__", f"Run — {payload['log']}")
    )


def resolve_flags_path(run: RunLog, explicit: Path | None) -> Path | None:
    """Find the §5 flag log for this run: ``--flags`` first, then the log's own record.

    ``export_clean_data`` reports where it wrote, and ``agent.py`` puts that in the
    ``run_summary`` event, so a run that exported can usually find its own verdicts
    with no argument. The path is relative to the repo root as the run recorded it;
    if the artefact has since moved, ``--flags`` is the override.
    """
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"--flags {explicit} does not exist.")
        return explicit
    if not run.flags_path:
        return None
    candidate = Path(run.flags_path)
    return candidate if candidate.exists() else None


def visualize_log(
    log_path: Path,
    series_path: Path | None = None,
    outdir: Path = DEFAULT_OUTDIR,
    show_truth: bool = True,
    open_browser: bool = True,
    flags_path: Path | None = None,
    show_verdicts: bool = True,
) -> Path:
    """Build the replay page for one log file and return where it was written."""
    run = parse_run(load_events(log_path), log_path)
    if not run.steps:
        raise ValueError(f"{log_path} holds no api_response events — nothing to replay.")

    if series_path is None:
        matches = autodetect_series(run)
        if len(matches) != 1:
            detail = (
                "matched " + ", ".join(str(m) for m in matches)
                if matches
                else f"no match in {DEFAULT_INJECTED_DIR}"
            )
            raise ValueError(
                f"Could not attribute {log_path.name} to one dataset ({detail}). "
                "The log does not record its input — pass --series <csv>."
            )
        series_path = matches[0]
        print(f"  matched series {series_path} (by rows, time range, NaN count and mean)")

    series = load_series(series_path)
    labels = load_labels(series_path) if show_truth else None
    if show_truth and labels is None:
        print(f"  no labels file beside {series_path.name} — ground truth layer omitted")

    entries = None
    if show_verdicts:
        resolved = resolve_flags_path(run, flags_path)
        if resolved is None:
            print(
                "  no flag log found — verdict layer omitted. "
                + ("This run never called export_clean_data, so it recorded no verdicts."
                   if not run.flags_path
                   else f"The run wrote {run.flags_path}, which is no longer there; pass --flags.")
            )
        else:
            entries = load_flag_log(resolved)
            print(f"  verdicts from {resolved} ({len(entries)} decided row(s))")

    payload = build_payload(run, series, series_path, labels, entries)
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{log_path.stem}.html"
    out.write_text(build_html(payload))

    n_calls = sum(len(s["calls"]) for s in payload["steps"])
    print(f"  {len(payload['steps'])} iteration(s), {n_calls} tool call(s)")
    print(f"  wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out


def latest_log(log_dir: Path = DEFAULT_LOG_DIR) -> Path | None:
    """Newest ``run_*.jsonl`` in ``log_dir``; the filename carries the timestamp."""
    logs = sorted(log_dir.glob("run_*.jsonl"))
    return logs[-1] if logs else None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay one agent run from logs/*.jsonl as a click-through page."
    )
    parser.add_argument(
        "log", nargs="?", type=Path,
        help="Log file to replay (default: the newest logs/run_*.jsonl).",
    )
    parser.add_argument(
        "--series", type=Path,
        help="Dataset CSV the run was performed on (default: auto-detected from "
             "the run's own inspect_dataset summary).",
    )
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument(
        "--no-truth", action="store_true",
        help="Skip the ground-truth label layer even if a labels file exists.",
    )
    parser.add_argument(
        "--flags", type=Path,
        help="§5 flag log (*_flags.json) holding the run's per-row verdicts "
             "(default: the path the run itself recorded in its run_summary).",
    )
    parser.add_argument(
        "--no-verdicts", action="store_true",
        help="Skip the final-verdict layer even if a flag log is available.",
    )
    parser.add_argument("--no-open", action="store_true", help="Do not open a browser.")
    args = parser.parse_args(argv)

    log_path = args.log or latest_log()
    if log_path is None:
        parser.error(f"no run_*.jsonl found in {DEFAULT_LOG_DIR}; pass a log path")
    if not log_path.exists():
        parser.error(f"log not found: {log_path}")

    visualize_log(
        log_path,
        series_path=args.series,
        outdir=args.outdir,
        show_truth=not args.no_truth,
        open_browser=not args.no_open,
        flags_path=args.flags,
        show_verdicts=not args.no_verdicts,
    )
    return 0


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
    --line: #e2e8f0; --now: #dc2626; --past: #94a3b8; --ctx: #7c3aed;
    --detect: #2563eb; --action: #059669; --utility: #64748b; --context: #7c3aed;
    --error: #dc2626;
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
  header .meta { color: var(--muted); font-variant-numeric: tabular-nums; }
  header .warn { color: var(--error); font-weight: 600; }

  main { display: grid; grid-template-columns: 270px 1fr; gap: 16px; padding: 16px 20px; }
  @media (max-width: 1000px) { main { grid-template-columns: 1fr; } }
  .card {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 8px; padding: 12px;
  }
  #plot { height: 330px; }
  #timeline { height: 110px; }

  /* step list */
  #steps { padding: 6px; max-height: 490px; overflow-y: auto; }
  .step {
    display: grid; grid-template-columns: 26px 1fr auto; gap: 8px; align-items: baseline;
    padding: 7px 8px; border-radius: 6px; cursor: pointer; border: 1px solid transparent;
  }
  .step:hover { background: var(--bg); }
  .step.sel { background: var(--bg); border-color: var(--muted); }
  .step .n { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }
  .step .name { font-weight: 600; word-break: break-word; }
  .step .name.detect { color: var(--detect); }
  .step .name.action { color: var(--action); }
  .step .name.context { color: var(--context); }
  .step .name.utility { color: var(--utility); }
  .step .name.err { color: var(--error); }
  .step .cnt { color: var(--muted); font-variant-numeric: tabular-nums; font-size: 12px; }

  dl.kv { margin: 0; display: grid; grid-template-columns: auto 1fr; gap: 3px 12px; }
  dl.kv dt { color: var(--muted); }
  dl.kv dd { margin: 0; font-variant-numeric: tabular-nums; word-break: break-word; }
  h2 { font-size: 13px; text-transform: uppercase; letter-spacing: .04em;
       color: var(--muted); margin: 0 0 8px; font-weight: 600; }
  .thought { white-space: pre-wrap; margin: 0; }
  .thought:empty::before { content: "(no reasoning text in this turn)"; color: var(--muted); }
  /* Thinking is the model reasoning to itself; prose is what it says to us.
     Kept visually distinct so the two are never read as the same channel. */
  .thought.thinking {
    color: var(--muted); font-style: italic;
    border-left: 3px solid var(--line); padding-left: 10px;
    max-height: 220px; overflow-y: auto;
  }
  .tag {
    margin: 0 0 4px; font-size: 11px; text-transform: uppercase;
    letter-spacing: .06em; color: var(--muted); font-weight: 600;
  }
  #thinking-wrap { margin-bottom: 12px; }
  pre.json {
    margin: 6px 0 0; padding: 8px; background: var(--bg); border: 1px solid var(--line);
    border-radius: 6px; overflow-x: auto; font-size: 12px; max-height: 220px;
  }
  .msg { margin: 6px 0 0; }
  .msg.err { color: var(--error); }
  .call + .call { margin-top: 14px; padding-top: 14px; border-top: 1px solid var(--line); }
  .call h3 { font-size: 13px; margin: 0 0 6px; font-weight: 600; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 1200px) { .cols { grid-template-columns: 1fr; } }

  .actions { display: flex; gap: 8px; flex-wrap: wrap; padding: 0 20px 16px; }
  button {
    font: inherit; font-weight: 600; padding: 8px 14px; border-radius: 8px;
    border: 1px solid var(--line); background: var(--panel); color: var(--ink);
    cursor: pointer;
  }
  button:hover { border-color: var(--muted); }
  button.on { background: var(--ink); border-color: var(--ink); color: #fff; }
  button kbd {
    font: inherit; font-size: 11px; opacity: .65; margin-left: 6px;
    border: 1px solid currentColor; border-radius: 4px; padding: 0 4px;
  }
  .hint { color: var(--muted); font-size: 12px; padding: 0 20px 24px; }
  .hint code {
    background: var(--panel); border: 1px solid var(--line);
    border-radius: 4px; padding: 1px 5px;
  }
</style>
</head>
<body>

<header>
  <h1 id="title"></h1>
  <span class="meta" id="sub"></span>
  <span class="spacer"></span>
  <span class="meta" id="tokens"></span>
  <span class="warn" id="capwarn"></span>
</header>

<p class="meta" id="verdictbar" style="padding: 0 20px; margin: 6px 0 0"></p>

<main>
  <div class="card" id="steps"></div>
  <div>
    <div class="card"><div id="plot"></div></div>
    <div class="card" style="margin-top:16px"><div id="timeline"></div></div>
  </div>
</main>

<div class="actions">
  <button data-nav="-1">Prev<kbd>&larr;</kbd></button>
  <button data-nav="1">Next<kbd>&rarr;</kbd></button>
  <button id="cumulative" class="on">Earlier flags<kbd>C</kbd></button>
  <button id="truth">Ground truth<kbd>G</kbd></button>
  <button id="verdicts">Final verdict<kbd>V</kbd></button>
  <button id="zoom">Zoom to flags<kbd>Z</kbd></button>
  <button id="reset">Whole series<kbd>R</kbd></button>
</div>

<div style="padding: 0 20px 20px" class="cols">
  <div class="card">
    <h2>Reasoning</h2>
    <div id="thinking-wrap" hidden>
      <p class="tag">thinking</p>
      <p class="thought thinking" id="thinking"></p>
    </div>
    <p class="thought" id="thought"></p>
  </div>
  <div class="card">
    <h2 id="call-head">Tool call</h2>
    <div id="calls"></div>
  </div>
</div>

<p class="hint">
  <b>Solid dots are the truth</b> (<code>G</code>) — the §5 injected labels, one colour per
  anomaly type. <b>Rings are what the agent flagged</b>: dark for this iteration, grey for
  earlier ones. So a coloured dot inside a dark ring was <b>caught</b>, a bare coloured dot
  was <b>missed</b>, and a bare ring is a <b>false positive</b>.
  <b>Squares are the run's final verdict</b> (<code>V</code>) from the §5 flag log — what it
  concluded, not what any one iteration flagged, so they do not change as you step. A square
  is coloured by the type <em>the agent</em> assigned, from the same palette as the truth
  dots: <b>same colour as the dot underneath = detected and classified correctly</b>, a
  different colour = it found something real and called it the wrong thing (§5.1 scores the
  agent's own <code>anomaly_type</code>). Sky-blue squares are rows it inspected and called
  <b>real water</b>; amber squares are <b>undecided</b> — flagged and never judged, which §10
  counts as no claim at all. Hover any square for its action and reason; use
  <code>python -m src.workbench.decision_audit</code> to read one decision in full.
  Green diamonds are point-context probes. Points with no value of their own — gaps, and anything
  <code>flag_nan</code> hit — are drawn as ticks along the bottom, since a NaN cannot be
  plotted at its own height. Built by
  <code>python -m src.workbench.visualize_log</code>.
</p>

<script>/*PLOTLY_JS*/</script>
<script id="payload" type="application/json">/*PAYLOAD*/</script>
<script>
(function () {
  "use strict";
  const D = JSON.parse(document.getElementById("payload").textContent);
  const $ = (id) => document.getElementById(id);

  // "YYYY-MM-DD HH:MM:SS" in the series' own (naive) clock, matching the CSV.
  // x values are naive ISO strings, never Date objects: Plotly renders a Date in
  // the viewer's timezone, which silently shifts the axis off the data.
  const fmtMs = (ms) => new Date(ms).toISOString().slice(0, 19).replace("T", " ");
  const at = D.times ? (i) => D.times[i] : (i) => fmtMs(D.t0 + i * D.step_ms);

  // Plotly hands axis ranges back as naive strings; read them in the same naive
  // clock they were written in, never as local time.
  const parseMs = (v) =>
    typeof v === "number" ? v : Date.parse(String(v).replace(" ", "T") + "Z");
  const timesMs = D.times ? D.times.map(parseMs) : null;
  function posAt(ms) {
    if (!timesMs) return Math.round((ms - D.t0) / D.step_ms);
    let lo = 0, hi = timesMs.length - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (timesMs[mid] < ms) lo = mid + 1; else hi = mid;
    }
    return lo;
  }
  function extent(a, b) {
    let lo = Infinity, hi = -Infinity;
    for (let i = Math.max(0, a); i <= Math.min(D.n - 1, b); i++) {
      const v = D.values[i];
      if (v === null) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    return isFinite(lo) ? [lo, hi] : null;
  }

  let sel = 0;
  let showPast = true;
  // On by default when labels exist: the point of the page is comparing what the
  // agent flagged against what is actually there, and a layer you have to know
  // to switch on reads as "there is no ground truth".
  let showTruth = Object.keys(D.truth).length > 0;
  // Same reasoning as showTruth: the run's own conclusion is the thing you came
  // to look at, so it is on wherever a flag log was found.
  let showVerdicts = Object.keys(D.verdicts).length > 0;
  let plotted = false;

  // ---------------------------------------------------------------- flag sets
  // A step can hold more than one tool call (the loop allows it), so a step's
  // flags are the union over its calls.
  const stepIdx = D.steps.map((s) => {
    const out = [];
    s.calls.forEach((c) => { for (const i of c.idx) out.push(i); });
    return out;
  });
  const stepCtx = D.steps.map((s) => {
    const out = [];
    s.calls.forEach((c) => { for (const p of c.ctx) out.push(p); });
    return out;
  });
  // Union of every earlier step, so "new this iteration" is visible at a glance.
  const priorIdx = [];
  {
    const seen = new Set();
    D.steps.forEach((_, k) => {
      priorIdx.push(Array.from(seen));
      for (const i of stepIdx[k]) seen.add(i);
    });
  }

  // Two visual channels, deliberately kept apart:
  //   WHAT IS TRUE  -> solid fill, one colour per §6 anomaly type (D.truth_colors)
  //   WHAT THE AGENT DID -> hollow ring drawn on top, in colours no truth type uses
  // So a red dot inside a dark ring is a spike the agent caught; a bare red dot is
  // one it missed; a bare ring is a false positive.
  const FLAG_NOW = "#111827";   // ring: flagged by this iteration
  const FLAG_PAST = "#94a3b8";  // ring: flagged by an earlier iteration
  const CTX = "#15803d";        // point-context probe (no truth type is green)

  // THIRD CHANNEL: the run's FINAL VERDICT per row, from the §5 flag log — what
  // the agent concluded, as opposed to what any one iteration flagged. Drawn as
  // an open SQUARE so it cannot be confused with a flag ring or a context
  // diamond, and it is step-independent: a verdict is a property of the run, not
  // of the iteration you happen to be looking at.
  //
  // An `anomaly` verdict takes the colour of the type the AGENT assigned, from
  // the same palette the truth dots use. That is the point of the layer: where
  // the square and the dot underneath it are the same colour the agent both
  // detected and classified the row correctly, and where they differ it found
  // something real and called it the wrong thing (§5.1 scores the agent's own
  // anomaly_type, so this distinction is the metric, not a detail).
  const VERDICT_NORMAL = "#0ea5e9";     // called real water — no truth type is sky
  const VERDICT_UNDECIDED = "#f59e0b";  // flagged, never adjudicated (§5)

  // A gap's value is NaN by definition, so plotting it at its own y draws
  // nothing at all — every gap label and every flag_nan hit would be invisible.
  // Those points go to a rug pinned just above the x axis instead.
  // Rug rows, on the fixed 0-1 overlay axis. Truth sits lowest and the two flag
  // layers stack above it, so a labelled gap and a flag on the same NaN row read
  // as two ticks rather than one tick hiding another. Kept clear of the axis
  // line — at 0.02 a size-9 tick is half-clipped by it and looks absent.
  const RUG = { truth: 0.05, past: 0.10, now: 0.15, verdict: 0.22 };

  function split(idx, rugY) {
    const here = { x: [], y: [] }, missing = { x: [], y: [] };
    for (const i of idx) {
      const v = D.values[i];
      if (v === null) { missing.x.push(at(i)); missing.y.push(rugY); }
      else { here.x.push(at(i)); here.y.push(v); }
    }
    return [here, missing];
  }

  function markerTraces(idx, name, marker, label, rugY) {
    const [here, missing] = split(idx, rugY);
    const out = [];
    if (here.x.length) {
      out.push({
        type: "scattergl", mode: "markers", name: name, x: here.x, y: here.y,
        marker: marker,
        hovertemplate: "%{x}<br>%{y}<br>" + label + "<extra></extra>",
      });
    }
    if (missing.x.length) {
      out.push({
        type: "scattergl", mode: "markers",
        // A gap category is rug-ONLY, so the rug has to carry the legend entry
        // or "truth: gap" disappears from the key entirely.
        name: here.x.length ? name + " (no value)" : name,
        showlegend: here.x.length === 0,
        x: missing.x, y: missing.y, yaxis: "y2",
        // line-ns-open IS a line, so it must carry a stroke width — inheriting a
        // solid dot's line.width of 0 renders an invisible tick.
        marker: {
          symbol: "line-ns-open", size: 9, color: marker.color,
          line: { width: 1.8, color: marker.color },
        },
        hovertemplate: "%{x}<br>" + label + " — value missing<extra></extra>",
      });
    }
    return out;
  }

  // The verdict layer needs PER-POINT hover (each row carries its own action and
  // reason), which markerTraces cannot express — its hovertemplate is one fixed
  // label for the whole trace. Hence a parallel builder rather than a parameter.
  function verdictTraces(key, rows) {
    const isAnomaly = key.indexOf("anomaly:") === 0;
    const type = isAnomaly ? key.slice(8) : "";
    const color = isAnomaly
      ? (D.truth_colors[type] || "#0f172a")
      : (key === "normal" ? VERDICT_NORMAL : VERDICT_UNDECIDED);
    const name = isAnomaly ? "verdict: " + type : "verdict: " + key;

    const here = { x: [], y: [], t: [] }, missing = { x: [], y: [], t: [] };
    for (const r of rows) {
      const v = D.values[r.i];
      const detail = [
        isAnomaly ? "verdict: anomaly (" + (r.t || "untyped") + ")" : "verdict: " + key,
        r.a ? "action: " + r.a : "",
        r.by ? "flagged by: " + r.by : "",
        r.src ? "rationale: " + r.src : "",
        r.r ? "&mdash; " + r.r : "",
      ].filter(Boolean).join("<br>");
      const bucket = v === null ? missing : here;
      bucket.x.push(at(r.i));
      bucket.y.push(v === null ? RUG.verdict : v);
      bucket.t.push(detail);
    }

    const out = [];
    if (here.x.length) {
      out.push({
        type: "scattergl", mode: "markers", name: name,
        x: here.x, y: here.y, text: here.t,
        marker: { size: 13, symbol: "square-open", color: color, line: { width: 2 } },
        hovertemplate: "%{x}<br>%{y}<br>%{text}<extra></extra>",
      });
    }
    if (missing.x.length) {
      out.push({
        type: "scattergl", mode: "markers",
        name: here.x.length ? name + " (no value)" : name,
        showlegend: here.x.length === 0,
        x: missing.x, y: missing.y, text: missing.t, yaxis: "y2",
        marker: {
          symbol: "line-ns-open", size: 9, color: color,
          line: { width: 1.8, color: color },
        },
        hovertemplate: "%{x}<br>%{text}<br>value missing<extra></extra>",
      });
    }
    return out;
  }

  function tracesFor(k) {
    const traces = [{
      type: "scattergl", mode: "lines", name: "series",
      x: Array.from({ length: D.n }, (_, i) => at(i)),
      y: D.values,
      line: { color: "#cbd5e1", width: 1 },
      hovertemplate: "%{x}<br>%{y}<extra></extra>",
    }];

    // Truth first, so the agent's rings land on top of the dots they explain.
    if (showTruth) {
      for (const [type, idx] of Object.entries(D.truth)) {
        const color = D.truth_colors[type] || "#0f172a";
        traces.push(...markerTraces(idx, "truth: " + type, {
          size: 8, color: color, line: { width: 0 },
        }, "labelled " + type, RUG.truth));
      }
    }

    if (showPast && priorIdx[k].length) {
      traces.push(...markerTraces(priorIdx[k], "flagged earlier", {
        size: 12, symbol: "circle-open", color: FLAG_PAST, line: { width: 1.4 },
      }, "flagged earlier", RUG.past));
    }

    if (stepIdx[k].length) {
      traces.push(...markerTraces(stepIdx[k], "flagged this step", {
        size: 16, symbol: "circle-open", color: FLAG_NOW, line: { width: 2.2 },
      }, "flagged this step", RUG.now));
    }

    // Verdicts last: they are the run's conclusion, so they read on top of both
    // the truth they are judged against and the flags they were drawn from.
    if (showVerdicts) {
      for (const [key, rows] of Object.entries(D.verdicts)) {
        traces.push(...verdictTraces(key, rows));
      }
    }

    const ctx = stepCtx[k];
    if (ctx.length) {
      traces.push({
        type: "scattergl", mode: "markers", name: "context probe",
        x: ctx.map((p) => at(p.i)), y: ctx.map((p) => D.values[p.i]),
        text: ctx.map((p) => p.reads_like),
        marker: { size: 18, symbol: "diamond-open", color: CTX, line: { width: 2.2 } },
        hovertemplate: "%{x}<br>%{y}<br>reads like: %{text}<extra></extra>",
      });
    }
    return traces;
  }

  const LAYOUT = {
    margin: { l: 52, r: 12, t: 8, b: 34 },
    xaxis: { type: "date", showgrid: true, gridcolor: "#f1f5f9" },
    yaxis: { title: { text: "value" }, gridcolor: "#f1f5f9" },
    // Fixed 0-1 overlay carrying the missing-value rug, so it stays pinned just
    // above the x axis no matter how the data axis is zoomed.
    yaxis2: { overlaying: "y", range: [0, 1], visible: false, fixedrange: true },
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0 },
    hovermode: "closest",
    plot_bgcolor: "#fff", paper_bgcolor: "#fff",
  };

  function drawPlot() {
    const gd = $("plot");
    const layout = Object.assign({}, LAYOUT);
    if (plotted) {
      // Keep the user's zoom while stepping through iterations.
      const cur = gd.layout;
      if (cur && cur.xaxis && cur.xaxis.range) {
        layout.xaxis = Object.assign({}, LAYOUT.xaxis, { range: cur.xaxis.range.slice() });
      }
      if (cur && cur.yaxis && cur.yaxis.range) {
        layout.yaxis = Object.assign({}, LAYOUT.yaxis, { range: cur.yaxis.range.slice() });
      }
    }
    Plotly.react(gd, tracesFor(sel), layout, { responsive: true, displaylogo: false });
    if (!plotted) {
      // gd.on(...) only exists once Plotly has plotted into the div, so this is
      // wired after the first draw, never at start-up.
      gd.on("plotly_relayout", autoscaleY);
    }
    plotted = true;
  }

  // Plotly autoranges y over ALL the data, so zooming into a week of a two-year
  // series leaves the y axis scaled to the largest storm in the record and the
  // window flat against the axis. Rescale y to what is actually in view.
  let rescaling = false;
  function autoscaleY() {
    if (rescaling) return;
    const gd = $("plot");
    const ax = gd.layout && gd.layout.xaxis;
    if (!ax || ax.autorange || !ax.range) return;  // whole series: let Plotly do it
    const span = extent(posAt(parseMs(ax.range[0])), posAt(parseMs(ax.range[1])));
    if (!span) return;
    const pad = (span[1] - span[0]) * 0.08 || 1;
    rescaling = true;
    Plotly.relayout(gd, { "yaxis.range": [span[0] - pad, span[1] + pad] })
      .then(() => { rescaling = false; }, () => { rescaling = false; });
  }

  // ---------------------------------------------------------------- timeline
  // One row per iteration, its flags laid out in time. Clicking a row selects
  // that iteration, so the whole run is navigable from the plot as well.
  function drawTimeline() {
    // Height scales with the number of iterations, or the row labels collide.
    $("timeline").style.height =
      Math.max(110, 17 * D.steps.length + 46) + "px";
    const traces = [];
    D.steps.forEach((s, k) => {
      const idx = stepIdx[k];
      if (!idx.length) return;
      traces.push({
        type: "scattergl", mode: "markers", showlegend: false,
        x: idx.map((i) => at(i)), y: idx.map(() => k),
        marker: {
          size: 6, symbol: "line-ns-open", line: { width: 1.5 },
          color: k === sel ? FLAG_NOW : "#cbd5e1",
        },
        name: "step " + s.i,
        hovertemplate: "step " + s.i + "<br>%{x}<extra></extra>",
      });
    });
    Plotly.react($("timeline"), traces, {
      margin: { l: 52, r: 12, t: 6, b: 34 },
      xaxis: { type: "date", gridcolor: "#f1f5f9" },
      yaxis: {
        title: { text: "iteration" }, autorange: "reversed",
        dtick: 1, gridcolor: "#f1f5f9",
      },
      plot_bgcolor: "#fff", paper_bgcolor: "#fff", hovermode: "closest",
    }, { responsive: true, displaylogo: false });
  }

  // ---------------------------------------------------------------- panels
  function labelFor(s) {
    if (!s.calls.length) return s.stop_reason === "tool_use" ? "(no tool call)" : "final answer";
    return s.calls.map((c) => c.name).join(", ");
  }

  function buildList() {
    const box = $("steps");
    box.innerHTML = "";
    D.steps.forEach((s, k) => {
      const row = document.createElement("div");
      row.className = "step";
      row.dataset.k = String(k);
      const err = s.calls.some((c) => c.is_error);
      const kind = s.calls.length ? s.calls[0].kind : "utility";
      const n = stepIdx[k].length;
      row.innerHTML =
        '<span class="n">' + s.i + '</span>' +
        '<span class="name ' + (err ? "err" : kind) + '">' + labelFor(s) + '</span>' +
        '<span class="cnt">' + (n ? n.toLocaleString() : "") + '</span>';
      row.addEventListener("click", () => select(k));
      box.appendChild(row);
    });
  }

  function renderCall(c, before) {
    const box = document.createElement("div");
    box.className = "call";

    const kv = document.createElement("dl");
    kv.className = "kv";
    const add = (k, v) => {
      const dt = document.createElement("dt"); dt.textContent = k;
      const dd = document.createElement("dd"); dd.textContent = v;
      kv.appendChild(dt); kv.appendChild(dd);
    };
    add("tool", c.name);
    add("kind", c.kind);
    for (const [k, v] of Object.entries(c.params || {})) {
      add(k, v === null ? "null" : String(v));
    }
    add("flagged", c.n_flagged.toLocaleString() +
        (c.n_reported !== c.n_flagged
          ? " (" + c.n_reported.toLocaleString() + " reported, rest off-grid)" : ""));
    // "new" is per call, against everything flagged before this step: it answers
    // "did this detector find anything the earlier ones had not?"
    add("new vs earlier", c.idx.filter((i) => !before.has(i)).length.toLocaleString());
    box.appendChild(kv);

    const msg = document.createElement("p");
    msg.className = "msg" + (c.is_error ? " err" : "");
    msg.textContent = c.is_error ? c.error : c.message;
    box.appendChild(msg);

    if (Object.keys(c.extra || {}).length) {
      const pre = document.createElement("pre");
      pre.className = "json";
      pre.textContent = JSON.stringify(c.extra, null, 1);
      box.appendChild(pre);
    }
    return box;
  }

  function renderPanels() {
    const s = D.steps[sel];
    $("thought").textContent = s.text || "";
    // Hidden entirely when the run didn't request thinking, so an absent trace
    // never looks like an empty one.
    $("thinking-wrap").hidden = !s.thinking;
    $("thinking").textContent = s.thinking || "";

    const box = $("calls");
    box.innerHTML = "";

    if (!s.calls.length) {
      $("call-head").textContent = "No tool call";
      const p = document.createElement("p");
      p.className = "msg";
      p.textContent = "stop reason: " + (s.stop_reason || "—");
      box.appendChild(p);
      return;
    }

    // A turn may emit several tool_use blocks, so every call is rendered.
    $("call-head").textContent =
      s.calls.length > 1 ? s.calls.length + " tool calls" : "Tool call";
    const before = new Set(priorIdx[sel]);
    s.calls.forEach((c) => box.appendChild(renderCall(c, before)));
  }

  function select(k) {
    sel = Math.max(0, Math.min(D.steps.length - 1, k));
    document.querySelectorAll(".step").forEach((el) => {
      el.classList.toggle("sel", Number(el.dataset.k) === sel);
    });
    const cur = document.querySelector(".step.sel");
    if (cur) cur.scrollIntoView({ block: "nearest" });
    drawPlot();
    drawTimeline();
    renderPanels();
  }

  // ---------------------------------------------------------------- controls
  function zoomToFlags() {
    const idx = stepIdx[sel];
    if (!idx.length) return;
    const lo = Math.min.apply(null, idx), hi = Math.max.apply(null, idx);
    const pad = Math.max(Math.round((hi - lo) * 0.25), 24);
    const a = Math.max(0, lo - pad), b = Math.min(D.n - 1, hi + pad);
    // y follows from the relayout handler.
    Plotly.relayout($("plot"), { "xaxis.range": [at(a), at(b)] });
  }

  function resetZoom() {
    Plotly.relayout($("plot"), { "xaxis.autorange": true, "yaxis.autorange": true });
  }

  document.querySelectorAll("[data-nav]").forEach((b) => {
    b.addEventListener("click", () => select(sel + Number(b.dataset.nav)));
  });
  $("cumulative").addEventListener("click", () => {
    showPast = !showPast;
    $("cumulative").classList.toggle("on", showPast);
    drawPlot();
  });
  $("truth").addEventListener("click", () => {
    showTruth = !showTruth;
    $("truth").classList.toggle("on", showTruth);
    drawPlot();
  });
  $("verdicts").addEventListener("click", () => {
    showVerdicts = !showVerdicts;
    $("verdicts").classList.toggle("on", showVerdicts);
    drawPlot();
  });
  $("zoom").addEventListener("click", zoomToFlags);
  $("reset").addEventListener("click", resetZoom);

  document.addEventListener("keydown", (ev) => {
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const k = ev.key.toLowerCase();
    if (k === "arrowright" || k === "k") { select(sel + 1); ev.preventDefault(); }
    else if (k === "arrowleft" || k === "j") { select(sel - 1); ev.preventDefault(); }
    else if (k === "c") $("cumulative").click();
    else if (k === "g") $("truth").click();
    else if (k === "v") $("verdicts").click();
    else if (k === "z") zoomToFlags();
    else if (k === "r") resetZoom();
  });

  // ---------------------------------------------------------------- init
  $("title").textContent = D.log;
  const bits = [D.series, D.model || "model n/a", D.prompt_version || "prompt n/a",
                D.steps.length + " iterations"];
  $("sub").textContent = bits.join(" · ");
  const tok = D.steps.reduce((a, s) => {
    const u = s.usage || {};
    return a + (u.input_tokens || 0) + (u.output_tokens || 0);
  }, 0);
  $("tokens").textContent = tok ? tok.toLocaleString() + " tokens" : "";
  if (D.max_steps_reached) $("capwarn").textContent = "hit the call cap";
  $("truth").disabled = !showTruth;
  $("truth").classList.toggle("on", showTruth);
  $("verdicts").disabled = !showVerdicts;
  $("verdicts").classList.toggle("on", showVerdicts);
  if (showVerdicts) {
    // Tally by category so the header states the run's answer in words. A run
    // with `undecided` rows is the case worth surfacing loudest: those are rows
    // it flagged and never adjudicated, which §10 scores as no claim at all.
    const counts = {};
    let total = 0;
    for (const [key, rows] of Object.entries(D.verdicts)) {
      counts[key] = rows.length;
      total += rows.length;
    }
    const parts = Object.entries(counts)
      .sort((a, b) => b[1] - a[1])
      .map(([k, n]) => n + " " + k.replace("anomaly:", ""));
    $("verdictbar").textContent =
      total + " decided row(s) from " + (D.flags_file || "flag log") + ": " + parts.join(", ")
      + (counts.undecided ? "  \u2014 undecided rows were flagged but never judged" : "");
  } else {
    $("verdictbar").textContent = "";
  }

  buildList();
  select(0);  // draws both plots, so the handlers below have something to bind to
  $("timeline").on("plotly_click", (ev) => {
    if (ev.points && ev.points.length) select(Math.round(ev.points[0].y));
  });
})();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
