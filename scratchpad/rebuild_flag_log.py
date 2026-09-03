"""Rebuild a run's §5 flag log from its JSONL, applying the current attribution rules.

A run's log already holds everything the flag log needs: each tool result carries its
`flagged_datetimes`, and the export call carries the agent's `decisions`. So a log
written before a fix to `wrappers._build_flag_log` can be regenerated exactly, without
re-running the agent or even SaQC.

    python -m scratchpad.rebuild_flag_log logs/run_*.jsonl out_flags.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.agent_tools.wrappers import DECISION_ACTIONS, UNDECIDED, _IMPUTE_FUNC

# The §7 wrapper name -> the SaQC func name the history records under.
TOOL_FUNCS = {
    "flag_range": "flagRange", "flag_constants": "flagConstants",
    "flag_plateau": "flagPlateau", "flag_spike_unilof": "flagUniLOF",
    "flag_zscore": "flagZScore", "flag_jumps": "flagJumps", "flag_nan": "flagNAN",
    "impute_rolling": _IMPUTE_FUNC,
}


def rebuild(log_path: Path) -> list[dict]:
    flagged_by: dict[str, list[str]] = {}
    decisions: list[dict] = []

    for line in log_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "api_response":
            continue
        for block in event["response"]["content"]:
            if block.get("type") == "tool_use" and block["name"] == "export_clean_data":
                decisions = block["input"].get("decisions", []) or decisions

    # Tool results come back on the NEXT api_call's message list; read them there.
    seen: set[str] = set()
    for line in log_path.read_text().splitlines():
        event = json.loads(line)
        if event.get("event") != "api_call":
            continue
        for msg in event["messages"]:
            if msg["role"] != "user" or not isinstance(msg["content"], list):
                continue
            for block in msg["content"]:
                if not (isinstance(block, dict) and block.get("type") == "tool_result"):
                    continue
                try:
                    payload = json.loads(block["content"])
                except (json.JSONDecodeError, TypeError):
                    continue
                tool = payload.get("tool")
                key = f"{tool}:{json.dumps(payload.get('params'), sort_keys=True)}"
                if tool not in TOOL_FUNCS or key in seen:
                    continue
                seen.add(key)
                func = TOOL_FUNCS[tool]
                for stamp in payload.get("flagged_datetimes", []):
                    names = flagged_by.setdefault(stamp, [])
                    if func not in names:
                        names.append(func)

    stamps = pd.DatetimeIndex(sorted(pd.Timestamp(s) for s in flagged_by))
    actions = pd.Series(index=stamps, dtype=object)
    reasons = pd.Series("", index=stamps, dtype=object)

    for decision in decisions:
        action = str(decision.get("action", "")).strip().lower()
        if action not in DECISION_ACTIONS:
            continue
        start = pd.Timestamp(decision["start"])
        end = pd.Timestamp(decision.get("end") or decision["start"])
        covered = (stamps >= start) & (stamps <= end) & actions.isna().to_numpy()
        actions[covered] = action
        reasons[covered] = str(decision.get("reason", "")).strip()

    joined = pd.Series({pd.Timestamp(s): "+".join(v) for s, v in flagged_by.items()}).sort_index()
    filled = joined.str.contains(_IMPUTE_FUNC, regex=False).to_numpy()
    actions[filled & (actions.isna().to_numpy() | actions.isin(["keep"]).to_numpy())] = "impute"
    actions = actions.fillna(UNDECIDED)

    return [
        {
            "datetime": s.strftime("%Y-%m-%dT%H:%M:%S"),
            "flagged_by": joined.loc[s],
            "action": actions.loc[s],
            "reason": reasons.loc[s],
        }
        for s in stamps
    ]


if __name__ == "__main__":
    entries = rebuild(Path(sys.argv[1]))
    Path(sys.argv[2]).write_text(json.dumps(entries, indent=2))
    counts = pd.Series([e["action"] for e in entries]).value_counts().to_dict()
    print(f"{len(entries):,} entries -> {sys.argv[2]}  {counts}")
