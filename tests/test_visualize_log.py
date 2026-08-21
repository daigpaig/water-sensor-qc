"""Tests for the agent-run replay page (``src.workbench.visualize_log``)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.workbench import visualize_log as vl


# --------------------------------------------------------------------------- fixtures
def _series_csv(tmp_path: Path, n: int = 12) -> Path:
    """A §5 dataset CSV on a regular 15-minute grid, with one NaN."""
    index = pd.date_range("2024-01-01", periods=n, freq="15min")
    values = [float(i) for i in range(n)]
    values[5] = float("nan")
    path = tmp_path / "gauge_l1.csv"
    pd.DataFrame({"datetime": index, "value": values}).to_csv(path, index=False)
    return path


def _tool_result(call_id: str, payload: dict) -> dict:
    return {"type": "tool_result", "tool_use_id": call_id, "content": json.dumps(payload)}


def _log(events: list[dict], tmp_path: Path, name: str = "run_x.jsonl") -> Path:
    path = tmp_path / name
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    return path


def _two_step_events(with_content: bool) -> list[dict]:
    """A two-iteration run: one flagging call, then a final text answer.

    ``with_content`` toggles whether the logged API response carries real
    Messages API content blocks. Both shapes occur in ``logs/`` and both have to
    parse, so every test runs against each.
    """
    inspect = {
        "tool": "inspect_dataset",
        "params": {"field": "value"},
        "n_rows": 12,
        "message": "Dataset inspected.",
        "summary": {"n_rows": 12},
    }
    spikes = {
        "tool": "flag_spike_unilof",
        "params": {"n": 20, "thresh": 1.5},
        "n_flagged": 2,
        "pct_flagged": 0.1667,
        "flagged_datetimes": ["2024-01-01T00:30:00", "2024-01-01T01:00:00"],
        "message": "Flagged 2 values.",
    }

    def response(blocks, stop_reason):
        dump = {"model": "claude-sonnet-4-6", "stop_reason": stop_reason,
                "usage": {"input_tokens": 10, "output_tokens": 3}}
        if with_content:
            dump["content"] = blocks
        return dump

    step0_blocks = [
        {"type": "text", "text": "Inspecting first."},
        {"type": "tool_use", "id": "c0", "name": "inspect_dataset", "input": {"field": "value"}},
        {"type": "tool_use", "id": "c1", "name": "flag_spike_unilof",
         "input": {"n": 20, "thresh": 1.5}},
    ]
    return [
        {"event": "system_prompt", "content": "...", "version": "v0.2-draft"},
        {"event": "api_call", "step": 0, "messages": [{"role": "user", "content": "go"}]},
        {"event": "api_response", "step": 0, "response": response(step0_blocks, "tool_use")},
        {"event": "api_call", "step": 1, "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": ["<opaque>"]},
            {"role": "user", "content": [_tool_result("c0", inspect), _tool_result("c1", spikes)]},
        ]},
        {"event": "api_response", "step": 1,
         "response": response([{"type": "text", "text": "Done."}], "end_turn")},
    ]


# --------------------------------------------------------------------------- parsing
@pytest.mark.parametrize("with_content", [True, False], ids=["api-dump", "no-content"])
def test_parse_run_recovers_every_call(with_content, tmp_path):
    """Both log shapes yield the same steps, calls, params and flags."""
    run = vl.parse_run(_two_step_events(with_content), tmp_path / "run_x.jsonl")

    assert run.prompt_version == "v0.2-draft"
    assert [s.index for s in run.steps] == [0, 1]

    # Two tool_use blocks in one turn: the loop allows it, so both must survive.
    names = [c.name for c in run.steps[0].calls]
    assert names == ["inspect_dataset", "flag_spike_unilof"]
    assert run.steps[0].usage["output_tokens"] == 3
    assert run.summary == {"n_rows": 12}

    spike_call = run.steps[0].calls[1]
    assert spike_call.params["thresh"] == 1.5
    assert spike_call.flagged == ["2024-01-01T00:30:00", "2024-01-01T01:00:00"]
    assert spike_call.kind == "detect"
    # flagged_datetimes is bulk data for the plot, never the detail panel.
    assert "flagged_datetimes" not in spike_call.extra

    assert run.steps[1].calls == []
    assert run.steps[1].stop_reason == "end_turn"
    # Reasoning text lives ONLY in the response dump. A log whose api_response
    # carries no content blocks keeps every call and count but loses the words —
    # which is why agent.py must keep logging response.model_dump().
    assert run.steps[1].text == ("Done." if with_content else "")
    assert run.steps[0].text == ("Inspecting first." if with_content else "")


def test_parse_run_captures_thinking_blocks(tmp_path):
    """Extended-thinking blocks are a separate channel from the prose text."""
    events = [
        {"event": "api_call", "step": 0, "messages": [{"role": "user", "content": "go"}]},
        {"event": "api_response", "step": 0, "response": {"stop_reason": "end_turn", "content": [
            {"type": "thinking", "thinking": "The NaN share is 4.7%, so gaps dominate."},
            {"type": "redacted_thinking", "data": "<encrypted>"},
            {"type": "text", "text": "I will start by inspecting the dataset."},
        ]}},
    ]
    step = vl.parse_run(events, tmp_path / "run_x.jsonl").steps[0]

    assert "gaps dominate" in step.thinking
    assert "[redacted thinking block]" in step.thinking
    assert step.text == "I will start by inspecting the dataset."  # kept separate


def test_parse_run_has_no_thinking_when_none_was_requested(tmp_path):
    """A run that never asked for thinking yields an empty trace, not a fake one."""
    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")
    assert all(s.thinking == "" for s in run.steps)


def test_parse_run_keeps_errored_calls(tmp_path):
    """A tool that raised is still an iteration, and says so."""
    events = [
        {"event": "api_call", "step": 0, "messages": [{"role": "user", "content": "go"}]},
        {"event": "api_response", "step": 0, "response": {"stop_reason": "tool_use"}},
        {"event": "api_call", "step": 1, "messages": [
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": "c0", "is_error": True,
                "content": "Error executing flag_plateau: attempt to get argmin of an empty sequence",
            }]},
        ]},
        {"event": "max_steps_reached", "step": 25},
    ]
    run = vl.parse_run(events, tmp_path / "run_x.jsonl")
    call = run.steps[0].calls[0]
    assert call.is_error
    assert call.name == "flag_plateau"  # recovered from the error text
    assert "argmin" in call.error
    assert run.max_steps_reached


def test_load_events_survives_a_truncated_line(tmp_path):
    """A run killed mid-write leaves a partial last line; that is not fatal."""
    path = tmp_path / "run_x.jsonl"
    path.write_text('{"event": "system_prompt", "version": "v1"}\n{"event": "api_ca\n')
    assert [e["event"] for e in vl.load_events(path)] == ["system_prompt"]


# --------------------------------------------------------------------------- payload
def test_build_payload_maps_flags_to_row_positions(tmp_path):
    series_path = _series_csv(tmp_path)
    series = vl.load_series(series_path)
    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")

    payload = vl.build_payload(run, series, series_path)

    assert payload["n"] == 12
    assert payload["step_ms"] == 15 * 60 * 1000  # not 1000x off: pandas 3 is us-based
    assert payload["times"] is None  # regular grid, so timestamps are rebuilt in-page
    assert pd.Timestamp(payload["t0"], unit="ms") == series.index[0]
    # 00:30 and 01:00 are rows 2 and 4.
    assert payload["steps"][0]["calls"][1]["idx"] == [2, 4]
    assert payload["values"][5] is None  # NaN survives as null so the line breaks


def test_build_payload_ships_timestamps_for_an_irregular_index(tmp_path):
    path = tmp_path / "irregular_l1.csv"
    stamps = ["2024-01-01T00:00", "2024-01-01T00:15", "2024-01-01T02:00"]
    pd.DataFrame({"datetime": stamps, "value": [1.0, 2.0, 3.0]}).to_csv(path, index=False)
    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")

    payload = vl.build_payload(run, vl.load_series(path), path)

    assert payload["times"] is not None
    assert len(payload["times"]) == 3


def test_build_payload_reads_ground_truth_labels(tmp_path):
    series_path = _series_csv(tmp_path)
    labels = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=12, freq="15min"),
        "is_anomaly": [False] * 12,
        "anomaly_type": [""] * 12,
    })
    labels.loc[[2, 3], ["is_anomaly", "anomaly_type"]] = [True, "spike"]
    labels.loc[[5], ["is_anomaly", "anomaly_type"]] = [True, "gap"]
    labels.to_csv(series_path.with_name(f"{series_path.stem}_labels.csv"), index=False)

    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")
    payload = vl.build_payload(
        run, vl.load_series(series_path), series_path, vl.load_labels(series_path)
    )

    assert payload["truth"] == {"spike": [2, 3], "gap": [5]}


# --------------------------------------------------------------------------- verdicts
def _flag_log(tmp_path: Path, entries: list[dict]) -> Path:
    path = tmp_path / "gauge_l1_flags.json"
    path.write_text(json.dumps(entries))
    return path


def test_parse_run_finds_the_flag_log_the_run_recorded(tmp_path):
    """The verdicts live in a separate artefact; the log says where."""
    events = _two_step_events(True) + [
        {"event": "run_summary", "steps": 2, "flags_path": "data/agent_runs/gauge_l1_flags.json"}
    ]
    run = vl.parse_run(events, tmp_path / "run_x.jsonl")
    assert run.flags_path == "data/agent_runs/gauge_l1_flags.json"

    # A run that never exported has no verdicts to show, and must say so rather
    # than silently drawing an empty layer.
    assert vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl").flags_path is None


def test_verdict_layer_keys_anomalies_by_the_agents_own_type(tmp_path):
    """§5.1: the type scored is the agent's classification, so it drives the colour."""
    series_path = _series_csv(tmp_path)
    series = vl.load_series(series_path)
    index = pd.DatetimeIndex(series.index)

    layer = vl._verdict_layer(index, [
        {"datetime": "2024-01-01T00:30:00", "verdict": "anomaly", "anomaly_type": "spike",
         "action": "delete", "reason": "narrow, high z", "flagged_by": "flagUniLOF"},
        {"datetime": "2024-01-01T01:00:00", "verdict": "normal", "action": "keep",
         "reason": "storm limb"},
        {"datetime": "2024-01-01T01:15:00", "verdict": "undecided", "action": "undecided",
         "reason": ""},
        {"datetime": "2024-01-01T01:30:00", "verdict": "anomaly", "anomaly_type": "gap",
         "action": "impute", "reason": "short gap"},
    ])

    assert set(layer) == {"anomaly:spike", "normal", "undecided", "anomaly:gap"}
    assert layer["anomaly:spike"][0]["i"] == 2
    assert layer["anomaly:spike"][0]["a"] == "delete"
    assert layer["anomaly:spike"][0]["by"] == "flagUniLOF"
    # undecided must NOT be folded into normal: one is a judgement, the other is
    # a row nobody looked at, and §10 needs them distinguishable.
    assert layer["undecided"][0]["i"] == 5


def test_verdict_layer_drops_timestamps_outside_the_series(tmp_path):
    """A flag log paired with the wrong series must not invent row positions."""
    series = vl.load_series(_series_csv(tmp_path))
    layer = vl._verdict_layer(pd.DatetimeIndex(series.index), [
        {"datetime": "1999-01-01T00:00:00", "verdict": "anomaly", "anomaly_type": "spike"},
        {"datetime": "2024-01-01T00:30:00", "verdict": "anomaly", "anomaly_type": "spike"},
    ])
    assert layer == {"anomaly:spike": [{"i": 2, "a": "", "t": "spike", "by": "", "src": "", "r": ""}]}


def test_build_payload_carries_the_verdict_layer(tmp_path):
    series_path = _series_csv(tmp_path)
    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")
    entries = [{"datetime": "2024-01-01T00:30:00", "verdict": "anomaly",
                "anomaly_type": "spike", "action": "delete", "reason": "x"}]

    payload = vl.build_payload(
        run, vl.load_series(series_path), series_path, None, entries
    )
    assert payload["verdicts"]["anomaly:spike"][0]["i"] == 2

    # No flag log at all -> an empty layer, and the page turns the toggle off.
    bare = vl.build_payload(run, vl.load_series(series_path), series_path)
    assert bare["verdicts"] == {}


def test_resolve_flags_path_prefers_the_explicit_override(tmp_path):
    run = vl.parse_run(_two_step_events(True), tmp_path / "run_x.jsonl")
    explicit = _flag_log(tmp_path, [{"datetime": "2024-01-01T00:30:00", "verdict": "normal"}])

    assert vl.resolve_flags_path(run, explicit) == explicit
    with pytest.raises(FileNotFoundError):
        vl.resolve_flags_path(run, tmp_path / "nope.json")
    # Recorded but since moved: not an error, just no layer.
    run.flags_path = str(tmp_path / "gone.json")
    assert vl.resolve_flags_path(run, None) is None


def test_load_flag_log_rejects_a_non_list(tmp_path):
    """An empty verdict layer reads as 'the agent concluded nothing' — fail loudly."""
    bad = tmp_path / "bad_flags.json"
    bad.write_text(json.dumps({"datetime": "2024-01-01T00:30:00"}))
    with pytest.raises(ValueError, match="not a §5 flag log"):
        vl.load_flag_log(bad)


def test_autodetect_needs_more_than_row_count(tmp_path):
    """Every injected dataset shares a row count and time range, so the mean and
    NaN count are what actually identify one. A wrong match is worse than none."""
    injected = tmp_path / "injected"
    for gauge, offset in (("aaa", 0.0), ("bbb", 100.0)):
        d = injected / gauge / "l1"
        d.mkdir(parents=True)
        pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=8, freq="15min"),
            "value": [offset + i for i in range(8)],
        }).to_csv(d / f"{gauge}_l1.csv", index=False)

    run = vl.parse_run([], tmp_path / "run_x.jsonl")
    run.summary = {
        "n_rows": 8, "n_nan": 0, "value_column": "value",
        "time_start": "2024-01-01T00:00:00", "time_end": "2024-01-01T01:45:00",
        "columns": {"value": {"mean": 103.5}},
    }
    assert [p.stem for p in vl.autodetect_series(run, injected)] == ["bbb_l1"]

    run.summary["columns"]["value"]["mean"] = 999.0
    assert vl.autodetect_series(run, injected) == []


# --------------------------------------------------------------------------- page
def test_visualize_log_writes_a_self_contained_page(tmp_path):
    series_path = _series_csv(tmp_path)
    log_path = _log(_two_step_events(True), tmp_path)

    out = vl.visualize_log(
        log_path, series_path=series_path, outdir=tmp_path / "figures", open_browser=False
    )

    html = out.read_text()
    assert out.name == "run_x.html"
    assert "Plotly" in html  # plotly.js inlined: the page must work offline
    assert "<script src=" not in html  # nothing is fetched from a CDN
    # The payload must not be able to close its host <script> element early.
    assert "</script>" not in html.split('id="payload"', 1)[1].split("</script>", 1)[0]


def test_visualize_log_refuses_an_ambiguous_series(tmp_path, monkeypatch):
    log_path = _log(_two_step_events(True), tmp_path)
    monkeypatch.setattr(vl, "autodetect_series", lambda run, *a, **k: [Path("a"), Path("b")])
    with pytest.raises(ValueError, match="--series"):
        vl.visualize_log(log_path, outdir=tmp_path / "figures", open_browser=False)
