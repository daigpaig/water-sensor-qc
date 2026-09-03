"""Does cache_control actually produce cache reads on this run's prompt + tools?

Two identical cheap calls: the first writes the cache, the second must read it.
`cache_read_input_tokens` staying at 0 on the second call means something in the
prefix is not byte-stable (§8) — run this after any edit to SYSTEM_PROMPT or
TOOL_SCHEMAS. Costs a few cents.
"""
import os

import anthropic
from dotenv import load_dotenv

from src.agent import MODEL, SYSTEM_PROMPT
from src.agent_tools.schemas import TOOL_SCHEMAS

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
messages = [{"role": "user", "content": "Reply with the single word OK."}]

for i in (1, 2):
    usage = client.messages.create(
        model=MODEL,
        max_tokens=16,
        thinking={"type": "disabled"},
        cache_control={"type": "ephemeral"},
        system=SYSTEM_PROMPT,
        messages=messages,
        tools=TOOL_SCHEMAS,
    ).usage
    print(
        f"call {i}: input={usage.input_tokens:,}  "
        f"write={usage.cache_creation_input_tokens:,}  "
        f"read={usage.cache_read_input_tokens:,}"
    )
