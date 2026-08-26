"""The click-any-point audit page (src/workbench/decision_audit.py).

Only the payload is tested — the page's JS is verified in a browser (§9.1 lists three
silent rendering traps that no data-level test can catch). What matters here is that
every clickable point carries its verdict and its explanation, and that a point
nothing happened to still has an answer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.workbench.decision_audit import (
    CATEGORIES, UNHANDLED_ANOMALY,
    build_cases, build_html, build_payload, categorise, context_samples,
)
from src.workbench.provenance import load_trace


def _fixture(tmp_path):
    index = pd.date_range("2024-01-01", periods=200, freq="15min")
    values = pd.Series(10.0 + np.sin(np.arange(200) / 8), index=index)
    values.iloc[50] = 90.0                       # a spike the run catches
    values.iloc[120] = 88.0                      # a spike no detector reaches

    labels = pd.DataFrame(
        {"is_anomaly": False, "anomaly_type": "", "true_value": np.nan, "source": ""},
        index=index,
    )
    for i in (50, 120):
        labels.iloc[i, labels.columns.get_loc("anomaly_type")] = "spike"
        labels.iloc[i, labels.columns.get_loc("source")] = "injected"

    flag_log = pd.DataFrame(
        [{"flagged_by": "flagUniLOF", "verdict": "anomaly", "anomaly_type": "spike",
          "action": "delete", "reason": "1 sample, robust_z 40",
          "rationale": "", "rationale_source": "agent-reason", "deliberation": "",
          "decided_by": {"start": str(index[50]), "end": str(index[50]),
                         "n_flagged_rows_in_span": 1, "n_rows_claimed": 1,
                         "difficulty": "clear"}}],
        index=pd.DatetimeIndex([index[50]]),
    )

    log = tmp_path / "run.jsonl"
    log.write_text("\n".join([
        json.dumps({"event": "api_response", "step": 0, "response": {"content": [
            {"type": "tool_use", "id": "a", "name": "flag_spike_unilof",
             "input": {"thresh": 1.5}}]}}),
        json.dumps({"event": "api_call", "messages": [{"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "a",
             "content": json.dumps({"tool": "flag_spike_unilof", "n_flagged": 1})}]}]}),
    ]))
    return values, labels, flag_log, load_trace(log)


def test_a_clickable_point_carries_the_verdict_not_just_the_action(tmp_path):
    values, labels, flag_log, trace = _fixture(tmp_path)
    cases = build_cases(values, labels, flag_log, trace)

    caught = next(c for c in cases if c["at"].startswith("2024-01-01T12:30"))
    assert caught["verdict"] == "anomaly"
    assert caught["anomaly_type"] == "spike"
    assert caught["action"] == "delete"
    assert "this is an anomaly" in caught["explain"]["headline"]


def test_an_undetected_anomaly_says_which_detector_stayed_silent(tmp_path):
    values, labels, flag_log, trace = _fixture(tmp_path)
    cases = build_cases(values, labels, flag_log, trace)

    missed = next(c for c in cases if c["category"] == "undetected")
    text = " ".join(s["text"] + " ".join(s["bullets"])
                    for s in missed["explain"]["sections"])
    assert "flag_spike_unilof(thresh=1.5)" in text
    assert "No detector fired on this row" in text
    assert missed["verdict"] == ""          # no entry, so no claim — not a blank "normal"


def test_a_point_nothing_happened_to_still_gets_an_answer(tmp_path):
    values, labels, flag_log, trace = _fixture(tmp_path)
    cases = build_cases(values, labels, flag_log, trace)
    payload = build_payload(values, cases, title="t", has_raw=False, trace=trace)

    quiet = payload["quiet"]
    assert "No detector flagged this point" in quiet["headline"]
    # The roll-call is what makes it an answer rather than a shrug, and it is carried
    # once for the whole page: `cases` covers every row that has its own story.
    assert len(quiet["roles"]) == len(payload["calls"]) == 1
    assert quiet["roles"] == ["ran, did not flag it"]


def test_the_page_is_one_self_contained_file(tmp_path):
    values, labels, flag_log, trace = _fixture(tmp_path)
    cases = build_cases(values, labels, flag_log, trace)
    html = build_html(build_payload(values, cases, title="t", has_raw=False, trace=trace))

    # Booleans first: asserting on the 8 MB string itself makes a failure take
    # minutes while pytest builds a diff of it.
    external = "<script src=" in html or "<link " in html
    inlined = "Plotly" in html                   # plotly.js in the file, not from a CDN
    unfilled = "/*PAYLOAD*/" in html or "/*PLOTLY_JS*/" in html
    assert not external and inlined and not unfilled


def test_the_detail_window_is_a_duration_not_a_sample_count():
    """A fixed 96 samples meant 24 h at 15-min and 8 h at 5-min — silently."""
    assert context_samples(pd.date_range("2024-01-01", periods=100, freq="15min")) == 96
    assert context_samples(pd.date_range("2024-01-01", periods=100, freq="5min")) == 288
    assert context_samples(pd.date_range("2024-01-01", periods=100, freq="1h")) == 24
    # Degenerate input falls back rather than dividing by zero.
    assert context_samples(pd.DatetimeIndex(["2024-01-01"])) == 96


def test_an_inspected_but_unflagged_point_is_not_filed_as_nothing():
    """The agent may describe_point any timestamp, flagged or not."""
    assert categorise("", "", flagged=False, inspected=True) == "inspected-not-flagged"
    assert categorise("", "", flagged=False, inspected=False) == ""
    assert categorise("", "spike", flagged=False, inspected=True) == "undetected"


# --- the verdict, not the action, decides found-vs-missed (§5.1) ---------------

def test_an_anomaly_the_agent_identified_and_kept_is_not_a_miss():
    """§6 makes `keep` the DEFAULT action for a level shift, so scoring the action
    filed every correctly-found shift under `missed`. Measured on run I: the page
    said missed=117 — every row of the one level shift — while evaluate.py scored
    that same flag log 1.000. Two readers of one log must not disagree about
    whether the event was found."""
    assert categorise("keep", "level_shift", flagged=True, verdict="anomaly") == "identified-kept"
    assert "identified-kept" not in UNHANDLED_ANOMALY


def test_a_miss_is_an_anomaly_the_agent_called_normal():
    for verdict in ("normal", "undecided"):
        assert categorise("keep", "spike", flagged=True, verdict=verdict) == "missed"


def test_an_anomaly_verdict_on_real_water_is_not_filed_as_a_quiet_keep():
    """The mirror case. §10 scores the verdict, so this costs precision even though
    the value survived — filing it under `kept` hid it among correct decisions."""
    assert categorise("keep", "", flagged=True, verdict="anomaly") == "false-anomaly"
    assert categorise("keep", "", flagged=True, verdict="normal") == "kept"


def test_a_gap_called_normal_is_a_miss_however_right_the_action_looks():
    assert categorise("keep", "gap", flagged=True, verdict="anomaly") == "left-missing"
    assert categorise("keep", "gap", flagged=True, verdict="normal") == "missed"


def test_a_log_written_before_verdicts_existed_renders_as_it_always_did():
    """Back-compat is load-bearing: silently recategorising an old run would make
    two archived pages of the same log disagree."""
    assert categorise("keep", "spike", flagged=True) == "missed"
    assert categorise("keep", "gap", flagged=True) == "left-missing"
    assert categorise("keep", "", flagged=True) == "kept"


def test_every_category_the_classifier_can_emit_is_declared():
    """CATEGORIES drives the sidebar, the colour table and the CLI counts; a bucket
    missing from it renders as an uncoloured, uncountable ghost."""
    emitted = {
        categorise(a, l, flagged=f, inspected=i, verdict=v)
        for a in ("keep", "delete", "correct", "impute", "")
        for l in ("", "spike", "plateau", "level_shift", "gap")
        for f in (True, False)
        for i in (True, False)
        for v in ("", "anomaly", "normal", "undecided")
    } - {""}
    assert emitted <= set(CATEGORIES), emitted - set(CATEGORIES)
