"""Tests for the agent ReAct loop and tool dispatch logic (B1 and B2)."""

import json
import sys
from unittest.mock import MagicMock, patch

# Mock saqc so we can test agent logic in any Python environment (e.g. without saqc 2.8)
sys.modules["saqc"] = MagicMock()

import pandas as pd
import pytest
import saqc

from src.agent import run_agent


def _toy_qc():
    """Create a toy SaQC object for testing."""
    idx = pd.date_range("2024-01-01", periods=10, freq="15min")
    data = pd.DataFrame({"value": range(10)}, index=idx)
    return saqc.SaQC(data)


@patch("src.agent.anthropic.Anthropic")
def test_agent_terminates_on_text(mock_anthropic, tmp_path):
    """Test that the agent loops correctly and breaks when producing a final text report."""
    qc = _toy_qc()
    
    # Mock the client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # Create a mock response
    mock_response = MagicMock()
    mock_response.stop_reason = "end_turn"
    
    # Mock the content blocks
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "This is the final report."
    mock_response.content = [text_block]
    mock_response.model_dump.return_value = {"mock": "dump"}
    
    mock_client.messages.create.return_value = mock_response
    
    final_qc, clean_df, report = run_agent(qc, max_steps=5, log_dir=str(tmp_path))
    
    # Ensure it only ran 1 step since the first response was "end_turn"
    assert mock_client.messages.create.call_count == 1
    assert "This is the final report." in report
    assert final_qc is qc
    assert clean_df is None
    
    # Check that a log file was created
    log_files = list(tmp_path.glob("*.jsonl"))
    assert len(log_files) == 1
    
    with open(log_files[0], "r") as f:
        logs = [json.loads(line) for line in f]
    
    assert logs[0]["event"] == "system_prompt"
    assert logs[1]["event"] == "api_call"
    assert logs[2]["event"] == "api_response"


@patch("src.agent.anthropic.Anthropic")
def test_agent_tool_dispatch(mock_anthropic, tmp_path):
    """Test that the agent dispatches tools to wrappers correctly and updates state."""
    qc = _toy_qc()
    
    # Mock the client
    mock_client = MagicMock()
    mock_anthropic.return_value = mock_client
    
    # 1. First response: Call inspect_dataset (wrapper) and describe_point (context)
    resp1 = MagicMock()
    resp1.stop_reason = "tool_use"
    
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
    
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "All done exporting."
    resp3.content = [text_block]
    resp3.model_dump.return_value = {"mock": "dump3"}
    
    # Wire the mock responses sequentially
    mock_client.messages.create.side_effect = [resp1, resp2, resp3]
    
    final_qc, clean_df, report = run_agent(qc, max_steps=5, log_dir=str(tmp_path))
    
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
    
    t1_res = last_msg["content"][0]
    assert t1_res["type"] == "tool_result"
    assert t1_res["tool_use_id"] == "call_1"
    assert t1_res["is_error"] is True
    assert "Error executing inspect_dataset" in t1_res["content"]
