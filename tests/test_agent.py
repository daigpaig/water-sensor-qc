"""Tests for the agent ReAct loop and tool dispatch logic (B1, B2, B3)."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

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


def _mock_usage(input_tokens: int = 100, output_tokens: int = 50):
    """Create a mock usage object matching the Anthropic response.usage shape."""
    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
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

    mock_client.messages.create.return_value = mock_response

    final_qc, clean_df, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    # Ensure it only ran 1 step since the first response was "end_turn"
    assert mock_client.messages.create.call_count == 1
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
    mock_client.messages.create.side_effect = [resp1, resp2, resp3]

    final_qc, clean_df, report, summary = run_agent(qc, max_steps=5, log_dir=str(tmp_path))

    # Ensure it ran 3 steps
    assert mock_client.messages.create.call_count == 3
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

    mock_client.messages.create.side_effect = [resp1, resp2]

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

    mock_client.messages.create.side_effect = [resp1, resp2]

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
