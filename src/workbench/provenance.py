"""Why did this point get the verdict it got? — reconstructed from the run's own record.

The §5 flag log says *what* the run concluded about a row. It does not say how the
run got there, and for a row nothing happened to it says nothing at all. This module
rebuilds the path, per timestamp, out of the two artefacts a run leaves behind — the
JSONL API log (`logs/run_*.jsonl`) and the flag log (`*_flags.json`) — and renders it
as sentences a reader can follow without knowing the codebase.

Four situations, and the point is that they are *different* and must read differently:

* **flagged, called an anomaly** — which detector fired and with what parameters, which
  `describe_point(s)` call measured it and what the numbers were, which decision span
  claimed it, and what the agent said.
* **flagged, called normal** — the same trail, ending in the agent rejecting its own
  detector's candidate. §6 asks for exactly this, so it needs to read as a decision
  rather than as an omission.
* **flagged, undecided** — no decision span covered it. Nobody concluded anything.
* **never flagged** — the interesting one. A flag log has no entry, so the honest answer
  is not "nothing happened" but a **roll-call**: these detectors ran, with these
  parameters, and not one of them fired here, so the agent was never shown this row.
  Without the roll-call an unflagged labelled anomaly and ordinary water look identical.

Nothing here judges. It reports what the run did, quotes what it said, and — where a
label file is available — states what the ground truth says alongside, marked as
something the agent could not see (§0).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field as _field
from pathlib import Path

import pandas as pd

# The canonical wrapper-name -> SaQC-method table. The join it enables is what lets a
# row's `flagged_by` ("flagUniLOF", a SaQC name) be traced back to the call the agent
# actually made ("flag_spike_unilof", with its parameters). Duplicating it here would
# be one more thing to keep in step.
from src.evaluate import _WRAPPER_TO_METHOD

METHOD_TO_WRAPPER: dict[str, str] = {m: w for w, m in _WRAPPER_TO_METHOD.items()}

# Measurement keys worth putting in a sentence, in reading order, with their units.
# Anything else in a describe_point row still reaches the reader as raw JSON.
_MEASURED_KEYS: tuple[tuple[str, str], ...] = (
    ("robust_z", "{:.1f} robust sigmas from the local median"),
    ("width_samples", "{:.0f} sample(s) wide"),
    ("samples_to_recover", "recovered in {:.0f} sample(s)"),
    ("fall_rise_ratio", "fall/rise ratio {:.2f}"),
    ("noise_ratio", "surrounding noise {:.1f}x the record-typical stretch"),
    ("step_sigmas_local", "its own step is {:.1f} sigmas by LOCAL scale"),
)


def _is_context_tool(name: str) -> bool:
    """`describe_point(s)` and every §7.3 primitive, without a list to keep in step."""
    return name.startswith("describe_point") or name.endswith("_context")


@dataclass(frozen=True)
class ToolCall:
    """One tool call the agent made, with what it returned."""

    step: int
    name: str
    params: dict
    n_flagged: int | None = None
    n_flagged_total: int | None = None
    message: str = ""
    failed: bool = False

    @property
    def kind(self) -> str:
        if self.name in _WRAPPER_TO_METHOD:
            return "detector"
        if _is_context_tool(self.name):
            return "context"
        if self.name in ("impute_rolling", "export_clean_data"):
            return "action"
        return "utility"

    @property
    def method(self) -> str | None:
        """The SaQC method this call ran, for joining against ``flagged_by``."""
        return _WRAPPER_TO_METHOD.get(self.name)

    def signature(self) -> str:
        """``flag_spike_unilof(thresh=1.5, n=20)`` — what the agent actually asked for."""
        inner = ", ".join(f"{k}={v!r}" for k, v in self.params.items()
                          if k != "ats" and not isinstance(v, (list, dict)))
        return f"{self.name}({inner})"


@dataclass
class Trace:
    """Everything one run log says about how it reached its conclusions."""

    calls: list[ToolCall] = _field(default_factory=list)
    # timestamp -> (step, the describe_point row the agent was handed)
    measured: dict[pd.Timestamp, tuple[int, dict]] = _field(default_factory=dict)
    # timestamp -> [(step, tool name), ...] for single-point §7.3 primitives
    probed: dict[pd.Timestamp, list[tuple[int, str]]] = _field(default_factory=dict)
    thinking: dict[int, str] = _field(default_factory=dict)
    decisions: list[dict] = _field(default_factory=list)

    @property
    def detectors(self) -> list[ToolCall]:
        return [c for c in self.calls if c.kind == "detector"]

    def calls_of(self, method: str) -> list[ToolCall]:
        """Every call that ran *method*. Usually one; a re-tuned detector gives more."""
        return [c for c in self.detectors if c.method == method]


def load_trace(log_path: Path) -> Trace:
    """Parse a run's JSONL log into a :class:`Trace`.

    Tool calls live in ``api_response`` events and their results come back inside the
    *next* ``api_call``'s message history — which repeats the whole conversation every
    step, so results are taken first-seen-wins.
    """
    trace = Trace()
    uses: dict[str, ToolCall] = {}
    results: dict[str, dict] = {}
    order: list[str] = []

    for line in Path(log_path).read_text().splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue                       # a killed run leaves a partial last line

        if event.get("event") == "api_response":
            step = int(event.get("step", len(order)))
            for block in event["response"].get("content", []):
                if block.get("type") == "thinking" and block.get("thinking"):
                    trace.thinking.setdefault(step, block["thinking"])
                if block.get("type") != "tool_use":
                    continue
                uses[block["id"]] = ToolCall(
                    step=step, name=block.get("name", "?"), params=block.get("input") or {}
                )
                order.append(block["id"])
                if block.get("name") == "export_clean_data":
                    trace.decisions = (block.get("input") or {}).get("decisions") or trace.decisions
            continue

        if event.get("event") != "api_call":
            continue
        for message in event.get("messages") or []:
            if not isinstance(message, dict) or not isinstance(message.get("content"), list):
                continue
            for block in message["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    results.setdefault(block.get("tool_use_id", ""), block)

    for use_id in order:
        call = uses[use_id]
        block = results.get(use_id)
        payload: dict | None = None
        if block is not None:
            try:
                payload = json.loads(block["content"])
            except (json.JSONDecodeError, TypeError, KeyError):
                payload = None              # an is_error result is plain text
        if isinstance(payload, dict):
            call = ToolCall(
                step=call.step, name=call.name, params=call.params,
                n_flagged=payload.get("n_flagged"),
                n_flagged_total=payload.get("n_flagged_total"),
                message=str(payload.get("message", "")),
                failed=bool(payload.get("failed")),
            )
            _harvest_measurements(trace, call, payload)
        trace.calls.append(call)

    return trace


def _harvest_measurements(trace: Trace, call: ToolCall, payload: dict) -> None:
    """Record which timestamps this call actually measured, and what it said about them."""
    tool = payload.get("tool") or call.name
    if tool == "describe_points":
        for row in payload.get("points") or []:
            if "at" in row:
                trace.measured.setdefault(pd.Timestamp(row["at"]), (call.step, row))
    elif tool == "describe_point" and "at" in payload:
        trace.measured.setdefault(pd.Timestamp(payload["at"]), (call.step, payload))
    elif _is_context_tool(call.name) and call.params.get("at"):
        # A single-question primitive (noise_context, level_shift_context, ...). It is
        # not a full describe_point row, so it is recorded as "the agent asked about
        # this exact point" rather than folded into `measured`.
        at = pd.Timestamp(call.params["at"])
        trace.probed.setdefault(at, []).append((call.step, call.name))


def _fmt_measurement(row: dict) -> str:
    parts = []
    for key, template in _MEASURED_KEYS:
        value = row.get(key)
        if value is None or (isinstance(value, float) and value != value):
            continue
        try:
            parts.append(template.format(float(value)))
        except (TypeError, ValueError):
            continue
    return "; ".join(parts)


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.0f}%" if whole else "0%"


def explain_point(
    at: pd.Timestamp,
    entry: dict | None,
    trace: Trace,
    *,
    n_flagged_rows: int = 0,
    label: str = "",
    label_source: str = "",
) -> dict:
    """Assemble the natural-language answer to "why this verdict, for this point".

    *entry* is the row's §5 flag-log entry, or ``None`` if the run never flagged it —
    which is a substantive answer, not a missing one. *label* is the ground truth, used
    only for the closing section and always marked as something the agent could not see.
    """
    verdict = (entry or {}).get("verdict", "") or ("undecided" if entry else "none")
    anomaly_type = (entry or {}).get("anomaly_type", "") or ""
    action = (entry or {}).get("action", "") or ""

    sections: list[dict] = []
    sections.append(_section_detection(at, entry, trace))
    sections.append(_section_inspection(at, trace))
    if entry is not None:
        sections.append(_section_decision(entry, n_flagged_rows))
        sections.append(_section_treatment(entry))
    if label or entry is not None:
        sections.append(_section_truth(entry, label, label_source))

    return {
        "headline": _headline(entry, verdict, anomaly_type),
        "verdict": verdict,
        "anomaly_type": anomaly_type,
        "action": action,
        "sections": [s for s in sections if s],
        # Aligned with `call_table(trace)`, not self-describing: the call list is the
        # same for every point in the run, so repeating it per case would multiply the
        # page size by the number of clickable points for no added information.
        "roles": roles_for(at, entry, trace),
    }


def _headline(entry: dict | None, verdict: str, anomaly_type: str) -> str:
    if entry is None:
        return "No detector flagged this point, so the run made no claim about it."
    if verdict == "anomaly":
        return f"The run's answer: this is an anomaly — type {anomaly_type or 'unspecified'}."
    if verdict == "normal":
        return "The run's answer: this is real water, not an artifact."
    return "Flagged, but the run never decided — it made no claim either way."


def _section_detection(at: pd.Timestamp, entry: dict | None, trace: Trace) -> dict:
    detectors = trace.detectors
    fired_methods = [m for m in str((entry or {}).get("flagged_by", "")).split("+") if m]
    ran = {c.method for c in detectors}

    if entry is None or not fired_methods:
        roll = [f"{c.signature()} — flagged {c.n_flagged or 0} row(s) elsewhere"
                for c in detectors]
        text = (
            f"No detector fired on this row. The run ran {len(detectors)} detector(s), and "
            "every one of them covered this timestamp:"
        )
        return {"kind": "detection", "title": "What flagged it", "text": text,
                "bullets": roll or ["The run ran no detectors at all."],
                "tone": "warn",
                "footer": ("None of them flagged it, so this point was never put in front of "
                           "the agent and no reasoning about it exists.")}

    bullets = []
    for method in fired_methods:
        matches = trace.calls_of(method)
        if not matches:
            bullets.append(f"{method} — flagged this row (the call is not in this log)")
        for call in matches:
            total = call.n_flagged_total if call.n_flagged_total is not None else call.n_flagged
            bullets.append(
                f"{method} flagged it — called at step {call.step} as {call.signature()}, "
                f"which flagged {call.n_flagged or 0} new row(s)"
                + (f" ({total} flagged in total by then)" if total != call.n_flagged else "")
            )
    silent = [c for c in detectors if c.method not in fired_methods]
    footer = ""
    if silent:
        footer = ("The other detector(s) ran and did not fire here: "
                  + ", ".join(c.signature() for c in silent) + ".")
    plural = "" if len(fired_methods) == 1 else "s"
    return {"kind": "detection", "title": "What flagged it",
            "text": f"{len(fired_methods)} detector{plural} put this row forward as a candidate. "
                    "A flag is a candidate, not the answer:",
            "bullets": bullets, "footer": footer, "tone": "note"}


def _section_inspection(at: pd.Timestamp, trace: Trace) -> dict:
    measured = trace.measured.get(at)
    probes = trace.probed.get(at, [])
    bullets = []
    if measured is not None:
        step, row = measured
        reads = row.get("reads_like") or "no read"
        detail = _fmt_measurement(row)
        bullets.append(f"describe_point(s) at step {step} read it as **{reads}**"
                       + (f" — {detail}" if detail else ""))
        if row.get("reason"):
            bullets.append(f"In the tool's own words: “{row['reason']}”")
    for step, name in probes:
        bullets.append(f"{name} called on this exact timestamp at step {step}")

    if not bullets:
        return {"kind": "inspection", "title": "Did the agent look at it",
                "text": "**No.** No describe_point, describe_points or context call covered "
                        "this timestamp, so any decision that reached it was made without "
                        "measuring it.",
                "bullets": [], "footer": ""}
    return {"kind": "inspection", "title": "Did the agent look at it",
            "text": "Yes — these are the measurements it had when it decided:",
            "bullets": bullets, "footer": ""}


def _section_decision(entry: dict, n_flagged_rows: int) -> dict:
    source = entry.get("rationale_source", "")
    span = entry.get("decided_by")
    bullets = []
    footer = ""
    tone = "note"

    if source == "deterministic":
        text = ("No judgement was made about this row by the agent. Code decided it: "
                + (entry.get("rationale") or "see the action below") + ".")
    elif span:
        reach = int(span.get("n_flagged_rows_in_span", 0))
        text = (f"A decision span covering {span['start'].replace('T', ' ')} → "
                f"{span['end'].replace('T', ' ')} claimed this row. That span reached "
                f"{reach:,} of the run's {n_flagged_rows:,} flagged rows "
                f"({_pct(reach, n_flagged_rows)}).")
        if source == "blanket":
            footer = ("This is a **blanket**: a span so wide it is a statement about the "
                      "leftovers rather than a judgement about this point. Whatever it "
                      "says, this row was swept up, not individually assessed.")
            tone = "warn"
        elif span.get("difficulty") == "judgement-call":
            footer = "The agent marked this decision a judgement call, so it had to show its working."
    else:
        text = "The run recorded a decision for this row, but not which span produced it."

    if entry.get("reason"):
        bullets.append(f"Its stated reason: “{entry['reason']}”")
    if entry.get("deliberation"):
        bullets.append(f"Its working: “{entry['deliberation']}”")
    if entry.get("verdict") == "undecided":
        text = ("**No decision span covered this row.** It was flagged and then left "
                "unadjudicated, which is why the verdict is `undecided` — the run makes "
                "no claim about it in either direction, and it scores as neither a "
                "detection nor a rejection.")
        bullets, footer = [], ""

    return {"kind": "decision", "title": "How the verdict was reached",
            "text": text, "bullets": bullets, "footer": footer, "tone": tone}


def _section_treatment(entry: dict) -> dict:
    verdict, action = entry.get("verdict", ""), entry.get("action", "")
    text = f"The value was **{action}**."
    footer = ""
    if verdict == "anomaly" and action == "keep":
        footer = ("Called an anomaly and left in place. That pair is allowed only when the "
                  "segment cannot be treated — a gap longer than any defensible "
                  "imputation window — and the reason above has to say so.")
    elif verdict == "normal":
        footer = ("A `normal` verdict must keep the value: calling something real water and "
                  "then deleting it is not a defensible pair, so the export refuses it.")
    elif verdict == "undecided":
        footer = "Nothing was concluded, so nothing was done on purpose."
    return {"kind": "treatment", "title": "What was done to the value",
            "text": text, "bullets": [], "footer": footer, "tone": "note"}


def _section_truth(entry: dict | None, label: str, label_source: str) -> dict:
    verdict = (entry or {}).get("verdict", "none" if entry is None else "")
    claimed = (entry or {}).get("anomaly_type", "")
    truth = label or "normal water"
    if label_source:
        truth += f" ({label_source})"

    if entry is None:
        text = (f"Ground truth: **{truth}**. "
                + ("No detector reached it, so this is a detector miss upstream of the agent "
                   "— it never had the chance to judge it." if label
                   else "Correctly left alone."))
    elif verdict == "anomaly" and label == claimed:
        text = f"Ground truth: **{truth}** — the run's claim matches, in type as well as verdict."
    elif verdict == "anomaly" and label:
        text = (f"Ground truth: **{truth}**, but the run claimed **{claimed}**. Counted as a "
                f"detection of {claimed} that the labels do not support, and as a missed {label}.")
    elif verdict == "anomaly":
        text = (f"Ground truth: **{truth}**. The run called this an anomaly, so it scores as a "
                "false positive — the expensive kind of error (§1: removing real "
                "data is worse than leaving a flagged point alone).")
    elif verdict == "normal" and label:
        text = f"Ground truth: **{truth}**. The run rejected it, so this is a missed detection."
    elif verdict == "normal":
        text = f"Ground truth: **{truth}** — the run was right to reject its own detector here."
    else:
        text = (f"Ground truth: **{truth}**. The run made no claim, so this counts as neither a "
                "detection nor a rejection.")
    return {"kind": "truth", "title": "Against the labels (the agent could not see these)",
            "text": text, "bullets": [], "footer": ""}


def call_table(trace: Trace) -> list[dict]:
    """The run's call sequence, once. Per-point bearing comes from :func:`roles_for`."""
    return [{"step": c.step, "tool": c.name, "signature": c.signature(),
             "kind": c.kind, "failed": c.failed} for c in trace.calls]


def roles_for(at: pd.Timestamp, entry: dict | None, trace: Trace) -> list[str]:
    """What each call in :func:`call_table` did *to this point*, in the same order.

    ``""`` means the call had no bearing on it. The distinction that matters is
    "ran and did not flag it" vs nothing at all: the first is evidence about the
    point, the second is a call that could never have said anything about it.
    """
    fired = {m for m in str((entry or {}).get("flagged_by", "")).split("+") if m}
    measured_step = trace.measured.get(at, (None, None))[0]
    probe_steps = {step for step, _ in trace.probed.get(at, [])}

    roles = []
    for call in trace.calls:
        if call.kind == "detector":
            role = "flagged it" if call.method in fired else "ran, did not flag it"
        elif call.step == measured_step:
            role = "measured it"
        elif call.step in probe_steps:
            role = "asked about it"
        elif call.name == "impute_rolling" and "interpolateByRolling" in fired:
            role = "filled it"
        elif call.name == "export_clean_data":
            role = "wrote the verdict"
        else:
            role = ""
        roles.append(role)
    return roles
