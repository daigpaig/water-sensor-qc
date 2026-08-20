"""The per-point story a run leaves behind (src/workbench/provenance.py).

The thing under test is not a number, it is whether a reader can tell four
situations apart: flagged-and-called-anomalous, flagged-and-rejected, flagged-and-
never-decided, and never-flagged-at-all. The last one is the one that used to read
as a blank, so most of these tests are about it saying something.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.workbench.provenance import (
    call_table,
    explain_point,
    load_trace,
    roles_for,
)


def _log(tmp_path, calls, decisions=None):
    """Write a minimal JSONL run log: [(name, params, result_dict), ...]."""
    lines = []
    pending = []
    for step, (name, params, result) in enumerate(calls):
        use_id = f"tu{step}"
        lines.append(json.dumps({
            "event": "api_call",
            "messages": [{"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": pid, "content": json.dumps(res)}
                for pid, res in pending
            ]}],
        }))
        pending = [(use_id, result)]
        lines.append(json.dumps({
            "event": "api_response", "step": step,
            "response": {"content": [
                {"type": "thinking", "thinking": f"thought at {step}", "signature": "x"},
                {"type": "tool_use", "id": use_id, "name": name, "input": params},
            ]},
        }))
    # the trailing results only reach the log on the next call
    lines.append(json.dumps({
        "event": "api_call",
        "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": pid, "content": json.dumps(res)}
            for pid, res in pending
        ]}],
    }))
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(lines))
    return path


DETECTORS = [
    ("flag_range", {"min": 0, "max": 1000},
     {"tool": "flag_range", "n_flagged": 0, "message": "none"}),
    ("flag_spike_unilof", {"thresh": 1.5, "n": 20},
     {"tool": "flag_spike_unilof", "n_flagged": 55, "message": "55"}),
    ("flag_nan", {}, {"tool": "flag_nan", "n_flagged": 478, "message": "478"}),
]

MEASURE = ("describe_points", {"ats": ["2024-01-01T00:00:00"]}, {
    "tool": "describe_points",
    "points": [{"at": "2024-01-01T00:00:00", "value": 42.0, "reads_like": "spike",
                "reason": "1-sample excursion at 9.4 robust sigmas.",
                "robust_z": 9.4, "width_samples": 1, "samples_to_recover": 1}],
})

AT = pd.Timestamp("2024-01-01T00:00:00")


def _entry(**over):
    entry = {
        "datetime": "2024-01-01T00:00:00", "flagged_by": "flagUniLOF",
        "verdict": "anomaly", "anomaly_type": "spike", "action": "delete",
        "reason": "1-sample excursion, unambiguous artifact.",
        "rationale_source": "agent-reason", "rationale": "", "deliberation": "",
        "decided_by": {"start": "2024-01-01T00:00:00", "end": "2024-01-01T00:00:00",
                       "n_flagged_rows_in_span": 1, "n_rows_claimed": 1,
                       "difficulty": "clear"},
    }
    entry.update(over)
    return entry


def _text(explanation) -> str:
    parts = []
    for section in explanation["sections"]:
        parts += [section["title"], section["text"], section.get("footer", "")]
        parts += section.get("bullets", [])
    return " ".join(parts)


def test_the_trace_recovers_every_call_with_its_parameters(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS + [MEASURE]))
    assert [c.name for c in trace.calls] == [
        "flag_range", "flag_spike_unilof", "flag_nan", "describe_points"
    ]
    unilof = trace.calls_of("flagUniLOF")
    assert len(unilof) == 1
    assert unilof[0].n_flagged == 55
    assert unilof[0].signature() == "flag_spike_unilof(thresh=1.5, n=20)"
    assert AT in trace.measured


def test_an_unflagged_point_gets_a_roll_call_not_a_blank(tmp_path):
    """The ask this module exists for: which tools looked and stayed silent."""
    trace = load_trace(_log(tmp_path, DETECTORS))
    explanation = explain_point(pd.Timestamp("2024-06-06"), None, trace)

    text = _text(explanation)
    assert "No detector fired on this row" in text
    # every detector that ran is named, with the parameters it ran at
    for signature in ("flag_range(min=0, max=1000)",
                      "flag_spike_unilof(thresh=1.5, n=20)", "flag_nan()"):
        assert signature in text
    assert "never put in front of the agent" in text
    assert explanation["verdict"] == "none"


def test_an_unflagged_labelled_anomaly_is_named_a_detector_miss(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS))
    explanation = explain_point(pd.Timestamp("2024-06-06"), None, trace, label="spike")
    text = _text(explanation)
    assert "detector miss upstream of the agent" in text
    assert "never had the chance to judge it" in text


def test_a_spike_verdict_shows_detector_measurement_and_reasoning(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS + [MEASURE]))
    explanation = explain_point(AT, _entry(), trace, n_flagged_rows=100, label="spike")

    assert explanation["headline"] == "The run's answer: this is an anomaly — type spike."
    text = _text(explanation)
    assert "flagUniLOF flagged it" in text and "step 1" in text     # which, and when
    assert "flag_spike_unilof(thresh=1.5, n=20)" in text            # at what settings
    assert "did not fire here" in text                              # and who stayed quiet
    assert "9.4 robust sigmas" in text                              # what it measured
    assert "unambiguous artifact" in text                           # what it said
    assert "matches, in type as well as verdict" in text            # vs ground truth


def test_a_blanket_says_so_instead_of_reading_as_a_judgement(tmp_path):
    """A row swept up by a catch-all must not read like a considered decision."""
    trace = load_trace(_log(tmp_path, DETECTORS))
    entry = _entry(
        verdict="normal", anomaly_type="", action="keep", rationale_source="blanket",
        reason="Broad catch-all for storm limbs.",
        decided_by={"start": "2023-07-01T00:00:00", "end": "2025-07-02T00:00:00",
                    "n_flagged_rows_in_span": 555, "n_rows_claimed": 400,
                    "difficulty": "clear"},
    )
    text = _text(explain_point(AT, entry, trace, n_flagged_rows=703))
    assert "blanket" in text
    assert "swept up, not individually assessed" in text
    assert "555" in text and "703" in text and "79%" in text


def test_never_measured_is_reported_as_a_different_failure_from_deciding_wrongly(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS))       # no describe_points at all
    text = _text(explain_point(AT, _entry(), trace, n_flagged_rows=1))
    assert "No describe_point" in text
    assert "without measuring it" in text


def test_an_undecided_row_is_not_reported_as_a_rejection(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS))
    entry = _entry(verdict="undecided", anomaly_type="", action="undecided",
                   reason="", rationale_source="deterministic", decided_by=None)
    explanation = explain_point(AT, entry, trace, n_flagged_rows=10)
    text = _text(explanation)
    assert "No decision span covered this row" in text
    assert "neither a detection nor a rejection" in text
    assert explanation["headline"].startswith("Flagged, but the run never decided")


def test_an_anomaly_left_untreated_explains_the_pair(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS))
    entry = _entry(anomaly_type="gap", action="keep", flagged_by="flagNAN",
                   reason="11-day outage, far longer than any defensible window.")
    text = _text(explain_point(AT, entry, trace, n_flagged_rows=10, label="gap"))
    assert "Called an anomaly and left in place" in text
    assert "cannot be treated" in text


def test_rejecting_a_real_detection_reads_as_a_miss_not_a_success(tmp_path):
    trace = load_trace(_log(tmp_path, DETECTORS + [MEASURE]))
    entry = _entry(verdict="normal", anomaly_type="", action="keep",
                   reason="Storm limb, 14 samples wide.")
    text = _text(explain_point(AT, entry, trace, n_flagged_rows=10, label="spike"))
    assert "missed detection" in text
    # and the reverse: no label means the rejection was right
    ok = _text(explain_point(AT, entry, trace, n_flagged_rows=10, label=""))
    assert "right to reject its own detector" in ok


def test_roles_align_with_the_call_table_one_for_one(tmp_path):
    """The page indexes roles positionally, so a drift here mislabels every step."""
    trace = load_trace(_log(tmp_path, DETECTORS + [MEASURE]))
    table = call_table(trace)
    roles = roles_for(AT, _entry(), trace)
    assert len(roles) == len(table) == 4
    assert dict(zip((c["tool"] for c in table), roles)) == {
        "flag_range": "ran, did not flag it",
        "flag_spike_unilof": "flagged it",
        "flag_nan": "ran, did not flag it",
        "describe_points": "measured it",
    }


def test_a_partial_log_from_a_killed_run_still_parses(tmp_path):
    path = _log(tmp_path, DETECTORS)
    path.write_text(path.read_text() + '\n{"event": "api_resp')   # truncated last line
    assert len(load_trace(path).calls) == 3


@pytest.mark.parametrize("missing", ["verdict", "decided_by", "rationale_source"])
def test_a_log_written_before_a_field_existed_still_explains(tmp_path, missing):
    """Older runs must degrade to a weaker story, never to a traceback."""
    trace = load_trace(_log(tmp_path, DETECTORS + [MEASURE]))
    entry = _entry()
    entry.pop(missing)
    text = _text(explain_point(AT, entry, trace, n_flagged_rows=10))
    assert "flagUniLOF flagged it" in text
