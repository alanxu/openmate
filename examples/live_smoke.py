"""Live smoke test — confirm the configured REAL model drives the agent end to end.

Unlike the offline examples, this calls the real provider (MiniMax via the
Anthropic-compatible API by default; needs a key in ``.env``). It runs one
tool-using turn, prints the trace, and exits non-zero on failure — handy as a
quick "is my key + wiring working?" check.

    python examples/live_smoke.py
"""

import asyncio
import os
import sys

from openmate import Agent, default_model, default_services
from openmate.adapters.tools.builtin import read_only_tools
from openmate.config import load_env


async def main() -> int:
    load_env()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")):
        print("no API key — set ANTHROPIC_API_KEY (or MINIMAX_API_KEY) in .env", file=sys.stderr)
        return 2

    agent = Agent(
        name="smoke",
        model=default_model(),
        instructions="You are concise. Use tools when they help.",
        services=default_services(verbose=True),  # prints the live trace
        tools=read_only_tools(),
        max_steps=5,
        max_tokens=512,
    )
    result = await agent.run(
        "Use the calculator to compute 1234 * 5678, then state the result."
    )

    print("\nfinal:", result.text)
    ok = result.ok and str(1234 * 5678) in result.text.replace(",", "")
    print("smoke:", "PASS" if ok else "FAIL", f"· {result.steps} steps · {result.usage.total_tokens} tokens")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
