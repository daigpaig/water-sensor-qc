"""Is `ttl` accepted on TOP-LEVEL cache_control, and does the entry read back?

The docs show `ttl` on a content-block cache_control; agent.py uses the top-level
auto-placing form. If `ttl` were rejected there, every agent run would 400 on the
first call — so this is checked before a run, not during one.
"""
import os
import anthropic
from dotenv import load_dotenv

load_dotenv()
client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
system = "You are a QC assistant. " + ("Reference material. " * 700)
common = dict(model="claude-sonnet-4-6", max_tokens=16, system=system,
              messages=[{"role": "user", "content": "ok"}])

for label, cc in (("1h ", {"type": "ephemeral", "ttl": "1h"}),):
    try:
        a = client.messages.create(cache_control=cc, **common)
        b = client.messages.create(cache_control=cc, **common)
        print(f"{label} ACCEPTED  call1 write={a.usage.cache_creation_input_tokens:,} "
              f"read={a.usage.cache_read_input_tokens:,} | "
              f"call2 write={b.usage.cache_creation_input_tokens:,} "
              f"read={b.usage.cache_read_input_tokens:,}")
    except Exception as exc:
        print(f"{label} REJECTED: {type(exc).__name__}: {str(exc)[:280]}")
