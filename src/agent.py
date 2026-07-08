"""The QC agent: a single ReAct reasoning loop + Anthropic API logger.

Uses the Anthropic Messages API multi-turn tool-use pattern with model
claude-sonnet-4-6. Enforces the 25-tool-call cap, dispatches tool calls to
src/tools/wrappers.py, and logs every API call to logs/*.jsonl. The versioned
system prompt lives here. See CLAUDE.md §8.

Implemented in Phase 3 (see CLAUDE.md §12).
"""
