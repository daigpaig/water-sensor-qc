"""One streaming call at the real MAX_TOKENS, to prove the loop's request shape works.

agent.py streams because the SDK refuses non-streaming requests whose max_tokens
implies a possible >10-minute response. Run this after touching MAX_TOKENS or the
request parameters — it costs a fraction of a cent and fails in seconds rather than
14 steps into a run.
"""
import os

import anthropic
from dotenv import load_dotenv

from src.agent import MAX_TOKENS, MODEL, SYSTEM_PROMPT
from src.agent_tools.schemas import TOOL_SCHEMAS

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

with client.messages.stream(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    thinking={"type": "adaptive"},
    cache_control={"type": "ephemeral"},
    system=SYSTEM_PROMPT,
    messages=[{"role": "user", "content": "Reply with the single word OK."}],
    tools=TOOL_SCHEMAS,
) as stream:
    response = stream.get_final_message()

print(f"  max_tokens={MAX_TOKENS:,}  stop_reason={response.stop_reason}")
print(f"  content blocks: {[b.type for b in response.content]}")
print(f"  cache write={response.usage.cache_creation_input_tokens:,} "
      f"read={response.usage.cache_read_input_tokens:,}")
