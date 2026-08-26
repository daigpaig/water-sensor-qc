"""Tests for the agent ReAct loop and tool dispatch logic (B1, B2, B3)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import anthropic
import numpy as np
import pandas as pd
import pytest
import saqc

# NOTE: do NOT put `sys.modules["saqc"] = MagicMock()` here. It was here to let these
# tests run without saqc installed, but sys.modules is process-global and pytest imports
# every test module into one process: the mock leaked into test_tools / test_context /
# test_inject and turned 91 passing tests red, while each file still passed on its own.
# saqc 2.8 is a pinned hard dependency of this project (CLAUDE.md §2), so use the real one.

from src.agent import RunSummary, run_agent


def _toy_qc():
    """Create a toy SaQC object for testing."""
    idx = pd.date_range("2024-01-01", periods=10, freq="15min")
    data = pd.DataFrame({"value": range(10)}, index=idx)
    return saqc.SaQC(data)


def _wire_stream(mock_client, responses):
    """Wire `client.messages.stream(...)` to hand back each response in turn.

    The loop streams and calls `get_final_message()` (the SDK refuses non-streaming
    requests at this max_tokens), so mocking `messages.create` no longer intercepts
    anything. An entry that is an exception is raised from the stream() call itself,
    which is how an API failure reaches the loop's guard.
    """
    entries = []
    for r in responses:
        if isinstance(r, BaseException):
            entries.append(r)
            continue
        cm = MagicMock()
        cm.__enter__.return_value.get_final_message.return_value = r
        cm.__exit__.return_value = False
        entries.append(cm)
    mock_client.messages.stream.side_effect = entries
    return mock_client.messages.stream


def _mock_usage(
    input_tokens: int = 100,
    output_tokens: int = 50,
    cache_write_tokens: int = 0,
    cache_read_tokens: int = 0,
):
    """Create a mock usage object matching the Anthropic response.usage shape.

    The cache fields must be set explicitly: a bare MagicMock returns a truthy
    child mock for any attribute, so leaving them out makes the cost arithmetic
    add a MagicMock to an int and the failure looks nothing like its cause.
    """
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_write_tokens
    usage.cache_read_input_tokens = cache_read_tokens
    return usage


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_agent_terminates_on_text(mock_anthropic, tmp_path):
    """Test that the agent loops correctly and breaks when producing a final text report."""
    qc = _toy_qc()

    # Mock the client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Create a mock response
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    mock_response.usage = _mock_usage(500, 200)

    # Mock the content blocks
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "This is the final report."
    mock_response.content = [text_block]
    mock_response.model_dump.return_value = {"mock": "dump"}

    _wire_stream(mock_client, [mock_response])

    final_qc, clean_df, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    # Ensure it only ran 1 step since the first response was "end_turn"
    assert mock_client.messages.stream.call_count == 1
    assert "This is the final report." in report
    assert final_qc is qc
    assert clean_df is None
    assert isinstance(summary, RunSummary)

    # Check that a log file was created
    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1

    with open(log_files[0], "r") as f:
        logs = [json.loads(line) for line in f]

    assert logs[0]["event"] == "system_prompt"
    assert logs[1]["event"] == "api_call"
    assert logs[2]["event"] == "api_response"


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_agent_tool_dispatch(mock_anthropic, tmp_path):
    """Test that the agent dispatches tools to wrappers correctly and updates state."""
    qc = _toy_qc()

    # Mock the client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # 1. First response: Call inspect_dataset (wrapper) and describe_point (context)
    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.usage = _mock_usage(1000, 300)

    tool1 = MagicMock()
    tool1.type = "tool_use"
    tool1.name = "inspect_dataset"
    tool1.input = {"field": "value"}
    tool1.id = "call_1"

    tool2 = MagicMock()
    tool2.type = "tool_use"
    tool2.name = "describe_point"
    tool2.input = {"at": "2024-01-01T00:15:00", "field": "value"}
    tool2.id = "call_2"

    resp1.content = [tool1, tool2]
    resp1.model_dump.return_value = {"mock": "dump1"}

    # 2. Second response: Call export_clean_data and end
    resp2 = MagicMock()
    resp2.stop_reason = "tool_use"
    resp2.usage = _mock_usage(2000, 400)

    tool3 = MagicMock()
    tool3.type = "tool_use"
    tool3.name = "export_clean_data"
    tool3.input = {"field": "value"}
    tool3.id = "call_3"

    resp2.content = [tool3]
    resp2.model_dump.return_value = {"mock": "dump2"}

    # 3. Third response: End turn with text
    resp3 = MagicMock()
    resp3.stop_reason = "end_turn"
    resp3.usage = _mock_usage(3000, 500)

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "All done exporting."
    resp3.content = [text_block]
    resp3.model_dump.return_value = {"mock": "dump3"}

    # Wire the mock responses sequentially
    _wire_stream(mock_client, [resp1, resp2, resp3])

    final_qc, clean_df, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    # Ensure it ran 3 steps
    assert mock_client.messages.stream.call_count == 3
    assert "All done exporting." in report

    # Clean df should be returned from export_clean_data
    assert clean_df is not None

    # Read the log file and check history
    log_files = list(tmp_path.glob("*.jsonl"))
    with open(log_files[0], "r") as f:
        logs = [json.loads(line) for line in f]

    # Look for the user message that contains the tool results
    user_msgs = [e for e in logs if e["event"] == "api_call"]
    assert len(user_msgs) == 3

    # The second API call (step 1) should have the tool results for call_1 and call_2
    messages_payload = user_msgs[1]["messages"]
    last_msg = messages_payload[-1]
    assert last_msg["role"] == "user"
    assert len(last_msg["content"]) == 2 # 2 tool results

    # call_1 is inspect_dataset, dispatched to wrappers with qc=. It should SUCCEED:
    # the previous version of this test asserted is_error here, but that was the mocked
    # saqc failing, not the dispatch working.
    t1_res = last_msg["content"][0]
    assert t1_res["type"] == "tool_result"
    assert t1_res["tool_use_id"] == "call_1"
    assert "is_error" not in t1_res
    assert json.loads(t1_res["content"])["tool"] == "inspect_dataset"

    # call_2 is describe_point, which lives in context.py and is re-exported by wrappers.
    # It observes only, so it returns a result without a qc object.
    t2_res = last_msg["content"][1]
    assert t2_res["tool_use_id"] == "call_2"
    assert "is_error" not in t2_res
    payload = json.loads(t2_res["content"])
    assert payload["tool"] == "describe_point"
    assert "reads_like" in payload


# --------------------------------------------------------------------------- B3 tests

@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_token_tracking(mock_anthropic, tmp_path):
    """B3: run_agent accumulates token usage and logs a run_summary event."""
    qc = _toy_qc()

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Two-step run: one tool call, then end_turn
    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.usage = _mock_usage(input_tokens=1200, output_tokens=350)

    tool1 = MagicMock()
    tool1.type = "tool_use"
    tool1.name = "inspect_dataset"
    tool1.input = {"field": "value"}
    tool1.id = "tok_call_1"
    resp1.content = [tool1]
    resp1.model_dump.return_value = {"mock": "dump1"}

    resp2 = MagicMock()
    resp2.stop_reason = "end_turn"
    resp2.usage = _mock_usage(input_tokens=5000, output_tokens=800)

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Done."
    resp2.content = [text_block]
    resp2.model_dump.return_value = {"mock": "dump2"}

    _wire_stream(mock_client, [resp1, resp2])

    final_qc, clean_df, report, summary = run_agent(qc, max_steps=10, log_dir=str(tmp_path))

    # Check the RunSummary
    assert summary.steps == 2
    assert summary.input_tokens == 1200 + 5000
    assert summary.output_tokens == 350 + 800
    assert summary.est_cost_usd > 0

    # Verify the run_summary event was written to the JSONL log
    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    with open(log_files[0], "r") as f:
        logs = [json.loads(line) for line in f]

    summary_events = [e for e in logs if e["event"] == "run_summary"]
    assert len(summary_events) == 1
    se = summary_events[0]
    assert se["steps"] == 2
    assert se["input_tokens"] == 6200
    assert se["output_tokens"] == 1150
    assert "est_cost_usd" in se
    assert "log_path" in se


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_cli_writes_outputs(mock_anthropic, tmp_path):
    """B3: main() loads a CSV, runs the agent, and writes *_clean, *_flags, *_report."""
    from src.agent import main

    # Create a minimal injected-style CSV in tmp_path
    series_dir = tmp_path / "data"
    series_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    df = pd.DataFrame({"datetime": idx, "value": range(20)})
    csv_path = series_dir / "test_gauge_l1.csv"
    df.to_csv(csv_path, index=False)

    # Mock the Anthropic client to call export_clean_data then end
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    # Step 1: tool_use calling export_clean_data
    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    resp1.usage = _mock_usage(800, 200)

    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.name = "export_clean_data"
    tool_block.input = {"field": "value"}
    tool_block.id = "cli_call_1"
    resp1.content = [tool_block]
    resp1.model_dump.return_value = {"mock": "resp1"}

    # Step 2: end_turn with a text report
    resp2 = MagicMock()
    resp2.stop_reason = "end_turn"
    resp2.usage = _mock_usage(1500, 400)

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "QC complete. No anomalies found."
    resp2.content = [text_block]
    resp2.model_dump.return_value = {"mock": "resp2"}

    _wire_stream(mock_client, [resp1, resp2])

    log_dir = str(tmp_path / "logs")
    result = main([str(csv_path), "--output-dir", str(series_dir), "--log-dir", log_dir])

    assert result == 0

    # Check that all three output files were created
    clean_path = series_dir / "test_gauge_l1_clean.csv"
    flags_path = series_dir / "test_gauge_l1_flags.json"
    report_path = series_dir / "test_gauge_l1_report.txt"

    assert clean_path.exists(), f"Missing {clean_path}"
    assert flags_path.exists(), f"Missing {flags_path}"
    assert report_path.exists(), f"Missing {report_path}"

    # Validate the clean CSV has the expected columns
    clean_df = pd.read_csv(clean_path)
    assert "datetime" in clean_df.columns
    assert "value" in clean_df.columns
    assert "flag" in clean_df.columns

    # Validate the flags JSON is a list
    flags = json.loads(flags_path.read_text())
    assert isinstance(flags, list)

    # Validate the report is non-empty
    report_text = report_path.read_text()
    assert "QC complete" in report_text

    # Check that a JSONL log was written
    log_files = list(Path(log_dir).glob("*.jsonl"))
    assert len(log_files) == 1


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_cli_flag_log_carries_the_agents_decisions(mock_anthropic, tmp_path):
    """The §5 flag log is written by export_clean_data, with action + reason intact.

    Without this the log records only that a row was flagged, which `evaluate.py`
    scores as no claim at all — the run's reasoning never reaches the metrics.
    """
    from src.agent import main

    series_dir = tmp_path / "data"
    series_dir.mkdir()
    idx = pd.date_range("2024-01-01", periods=40, freq="15min")
    values = [10.0] * 40
    values[20] = 500.0                     # a spike flag_range will catch
    df = pd.DataFrame({"datetime": idx, "value": values})
    csv_path = series_dir / "test_gauge_l1.csv"
    df.to_csv(csv_path, index=False)

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    def _tool_step(name, tool_input, call_id):
        resp = MagicMock()
        resp.stop_reason = "tool_use"
        resp.usage = _mock_usage(800, 200)
        block = MagicMock()
        block.type = "tool_use"
        block.name = name
        block.input = tool_input
        block.id = call_id
        resp.content = [block]
        resp.model_dump.return_value = {"mock": name}
        return resp

    spike_at = idx[20].strftime("%Y-%m-%dT%H:%M:%S")
    resp1 = _tool_step("flag_range", {"min": 0, "max": 100}, "c1")
    # §7.7: deleting a spike that was never checked against rainfall is refused, so a
    # run that means to delete one has to make this call. This gauge has no rain data
    # at all, which is exactly the case that must NOT deadlock: the tool answers "no
    # station covers this", the timestamp counts as looked-at, and the export proceeds.
    resp_precip = _tool_step("precip_context_points", {"ats": [spike_at]}, "c1b")
    resp2 = _tool_step(
        "export_clean_data",
        {"decisions": [
            {"start": spike_at, "difficulty": "clear", "verdict": "anomaly", "anomaly_type": "spike",
             "action": "delete", "reason": "500 NTU, 1 sample, robust_z 30"},
        ]},
        "c2",
    )
    resp3 = MagicMock()
    resp3.stop_reason = "end_turn"
    resp3.usage = _mock_usage(1500, 400)
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "One spike deleted."
    resp3.content = [text_block]
    resp3.model_dump.return_value = {"mock": "resp3"}

    _wire_stream(mock_client, [resp1, resp_precip, resp2, resp3])

    assert main([str(csv_path), "--output-dir", str(series_dir),
                 "--log-dir", str(tmp_path / "logs")]) == 0

    flags = json.loads((series_dir / "test_gauge_l1_flags.json").read_text())
    assert len(flags) == 1
    entry = flags[0]
    assert entry["datetime"] == spike_at
    assert entry["action"] == "delete"
    assert entry["reason"] == "500 NTU, 1 sample, robust_z 30"
    assert "flagRange" in entry["flagged_by"]

    # The entries themselves must not be echoed back to the model — a real run has
    # thousands of them and they would swamp the context.
    log_file = next(Path(tmp_path / "logs").glob("*.jsonl"))
    calls = [json.loads(l) for l in log_file.read_text().splitlines()]
    tool_results = [
        block
        for call in calls if call["event"] == "api_call"
        for msg in call["messages"] if msg["role"] == "user" and isinstance(msg["content"], list)
        for block in msg["content"] if isinstance(block, dict) and block.get("type") == "tool_result"
    ]
    exported = [b for b in tool_results if "Flag log holds" in str(b["content"])]
    assert exported, "export_clean_data result never reached the model"
    assert '"flags"' not in str(exported[0]["content"])


def _truncated_turn():
    """A turn that spent its whole budget thinking: `content` is a lone thinking block."""
    truncated = MagicMock()
    truncated.stop_reason = "max_tokens"
    truncated.usage = _mock_usage(500, 16000)
    thinking_block = MagicMock()
    thinking_block.type = "thinking"
    truncated.content = [thinking_block]
    truncated.model_dump.return_value = {"mock": "truncated"}
    return truncated


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_a_truncated_thinking_turn_is_retried_and_never_appended(mock_anthropic, tmp_path):
    """A turn that hits max_tokens mid-thought is asked for again, not appended.

    Its content is a lone `thinking` block, and the API rejects an assistant message
    whose final block is `thinking` — so appending it makes the NEXT call 400 with an
    error that says nothing about the real cause (08041770_l1, 2026-08-13).

    Not appending it has a useful consequence: the history is unchanged, so the turn
    can simply be requested again. Ending the run instead discarded 15 completed steps
    on 01467200_l1 (2026-08-19) when the export turn ran out of budget while planning.
    """
    from src.agent import run_agent

    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    qc = saqc.SaQC(pd.DataFrame({"value": range(20)}, index=idx))

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    _wire_stream(mock_client, [_truncated_turn(), _truncated_turn()])

    _, _, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    # Retried once, then gave up — not five attempts, and not one.
    assert mock_client.messages.stream.call_count == 2
    assert "twice" in report

    # The nudge is a USER message: the truncated assistant turn itself must never
    # reach the history, or the retry 400s on the block the API refuses to accept.
    sent = mock_client.messages.stream.call_args_list[-1].kwargs["messages"]
    assert all(m["role"] != "assistant" for m in sent)
    assert "export_clean_data" in sent[-1]["content"]


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_a_run_continues_normally_after_recovering_from_a_truncated_turn(
    mock_anthropic, tmp_path
):
    """The retry is a real second chance — the run goes on to finish."""
    from src.agent import run_agent

    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    qc = saqc.SaQC(pd.DataFrame({"value": range(20)}, index=idx))

    finished = MagicMock()
    finished.stop_reason = "end_turn"
    finished.usage = _mock_usage(500, 200)
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "Report: nothing was flagged."
    finished.content = [text_block]
    finished.model_dump.return_value = {"mock": "finished"}

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    _wire_stream(mock_client, [_truncated_turn(), finished])

    _, _, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    assert "Report: nothing was flagged." in report
    assert "twice" not in report


@patch("src.agent.anthropic.Anthropic")
@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test_key"})
def test_an_api_error_mid_loop_does_not_discard_the_run(mock_anthropic, tmp_path):
    """A 400 at step N must not throw away the N-1 tool calls already made."""
    from src.agent import run_agent

    idx = pd.date_range("2024-01-01", periods=20, freq="15min")
    qc = saqc.SaQC(pd.DataFrame({"value": range(20)}, index=idx))

    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client

    ok = MagicMock()
    ok.stop_reason = "tool_use"
    ok.usage = _mock_usage(400, 100)
    block = MagicMock()
    block.type = "tool_use"; block.name = "inspect_dataset"; block.input = {}; block.id = "c1"
    ok.content = [block]
    ok.model_dump.return_value = {"mock": "ok"}

    _wire_stream(mock_client, [
        ok,
        anthropic.BadRequestError(
            "credit balance is too low",
            response=MagicMock(status_code=400, headers={}),
            body=None,
        ),
    ])

    _, _, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    assert "stopped at step 1" in report
    assert summary.steps == 1          # the completed step is still accounted for

def test_a_mid_stream_failure_is_retried_rather_than_ending_the_run():
    """The SDK cannot retry once bytes are flowing, so this layer must.

    An httpx.ReadTimeout while the final export response streamed killed a 14-step run
    on 01467200_l1 (2026-08-18) — it is not an anthropic.APIError, so it escaped the
    handler that exists to break gracefully.
    """
    import httpx
    from src import agent as agent_module

    class _Stream:
        def __init__(self, message): self.message = message
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def get_final_message(self): return self.message

    calls = {"n": 0}

    class _Messages:
        def stream(self, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise httpx.ReadTimeout("the read operation timed out")
            return _Stream("final")

    class _Client:
        messages = _Messages()

    monkey = agent_module._STREAM_BACKOFF_SECONDS
    agent_module._STREAM_BACKOFF_SECONDS = 0.0
    try:
        assert agent_module._stream_message(_Client(), model="m") == "final"
        assert calls["n"] == 2                      # failed once, succeeded on the retry
    finally:
        agent_module._STREAM_BACKOFF_SECONDS = monkey


def test_a_stream_that_keeps_failing_raises_so_the_run_can_export_what_it_has():
    """Exhausted retries must surface as an exception the caller catches, not a hang."""
    import httpx
    import pytest
    from src import agent as agent_module

    class _Messages:
        def stream(self, **kwargs):
            raise httpx.ReadTimeout("still timing out")

    class _Client:
        messages = _Messages()

    monkey = agent_module._STREAM_BACKOFF_SECONDS
    agent_module._STREAM_BACKOFF_SECONDS = 0.0
    try:
        with pytest.raises(httpx.HTTPError):
            agent_module._stream_message(_Client(), model="m")
    finally:
        agent_module._STREAM_BACKOFF_SECONDS = monkey


# --- every published schema must actually be reachable (2026-08-25) ------------

def test_every_published_tool_has_a_dispatch_path():
    """The precipitation tools were published to the model, resolved fine by
    `_get_tool_function`, and MANDATED by the prompt since v0.13 — and every call came
    back `Unknown tool: precip_context_points`, because the dispatch if/elif chain
    checked only `wrappers` and `context` before raising. Nothing failed loudly: the
    model saw a tool error, worked around it, and the run looked normal. §7.7's claim
    that rainfall "cannot move the synthetic benchmark" was measured on a tool that
    had never once executed.

    A schema the model can call and the runner cannot dispatch is the exact shape of
    that bug, so assert the two sets agree rather than testing one tool.
    """
    from src.agent import _get_tool_function, context, precipitation
    from src.agent_tools import wrappers
    from src.agent_tools.schemas import TOOL_SCHEMAS

    for schema in TOOL_SCHEMAS:
        name = schema["name"]
        _get_tool_function(name)          # resolvable...
        assert (hasattr(wrappers, name) or hasattr(context, name)
                or hasattr(precipitation, name)), (
            f"{name} is published to the model but the dispatch chain in run_agent "
            "has no branch that can reach it"
        )


def test_the_runner_asks_the_right_river_for_its_rain():
    """`gauge` is absent from the schema and defaults to 01467200 inside the module, so
    without injection every other gauge would be answered with the wrong river's rain —
    and the result would look perfectly well-formed."""
    import inspect

    from src.agent import run_agent
    from src.agent_tools import precipitation

    assert "gauge" not in inspect.signature(precipitation.precip_context).bind_partial().arguments
    src = inspect.getsource(run_agent)
    assert 'stem.split("_")[0]' in src, "the gauge must come from the dataset stem (§5)"
    assert '"gauge": precip_gauge' in src, "the injected gauge must reach the tool call"


# --- mid-stream API errors are retried too (2026-08-25) ------------------------

def test_a_500_inside_an_open_stream_is_retried_not_fatal():
    """The SDK's max_retries stops applying once the first byte arrives, so a 5xx
    delivered as an error event INSIDE an open stream had no retry behind it: run K
    died at step 8 on `APIStatusError: Internal server error` and discarded eight
    completed steps. Same shape as the ReadTimeout that killed a run on 2026-08-18 —
    the handler existed and named the wrong exception type."""
    import anthropic
    from unittest.mock import MagicMock, patch

    from src.agent import _stream_message

    boom = anthropic.APIStatusError(
        "Internal server error",
        response=MagicMock(status_code=500, headers={}),
        body=None,
    )
    good = MagicMock(name="final")
    client, calls = MagicMock(), []

    def stream(**kwargs):
        calls.append(1)
        ctx = MagicMock()
        if len(calls) == 1:
            ctx.__enter__ = MagicMock(side_effect=boom)
        else:
            ctx.__enter__ = MagicMock(return_value=MagicMock(
                get_final_message=MagicMock(return_value=good)))
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    client.messages.stream = stream
    with patch("src.agent.time.sleep"):
        assert _stream_message(client) is good
    assert len(calls) == 2, "the 500 should have been retried exactly once"


def test_a_request_that_will_never_succeed_is_not_retried_four_times():
    """Each retry re-sends the whole conversation, so burning _STREAM_ATTEMPTS on a
    400 costs real money to fail identically four times."""
    import anthropic
    from unittest.mock import MagicMock, patch

    from src.agent import _stream_message

    bad = anthropic.APIStatusError(
        "invalid request",
        response=MagicMock(status_code=400, headers={}),
        body=None,
    )
    client, calls = MagicMock(), []

    def stream(**kwargs):
        calls.append(1)
        ctx = MagicMock()
        ctx.__enter__ = MagicMock(side_effect=bad)
        ctx.__exit__ = MagicMock(return_value=False)
        return ctx

    client.messages.stream = stream
    with patch("src.agent.time.sleep"), pytest.raises(anthropic.APIStatusError):
        _stream_message(client)
    assert len(calls) == 1


def test_transient_is_judged_by_the_same_predicate_that_drives_the_retry():
    """Run K's 500 arrived as the GENERIC APIStatusError, so an isinstance check
    against InternalServerError reported a plainly transient failure as permanent —
    telling the reader re-running would not help, when it was the only thing that would."""
    import anthropic
    from unittest.mock import MagicMock

    from src.agent import _is_retryable

    def status_error(code):
        return anthropic.APIStatusError(
            "boom", response=MagicMock(status_code=code, headers={}), body=None)

    assert _is_retryable(status_error(500))
    assert _is_retryable(status_error(529))
    assert _is_retryable(status_error(429))
    assert not _is_retryable(status_error(400))
    assert not _is_retryable(status_error(401))
