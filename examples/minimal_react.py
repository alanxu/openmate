"""The 'hello world' — a minimal ReAct agent (designs/architecture.md §20.1).

Requires an API key (see .env). Run:  python examples/minimal_react.py
"""

import asyncio

from openmate import Agent, Runtime, default_model, default_services
from openmate.adapters.tools.builtin import read_only_tools


async def main() -> None:
    agent = Agent(
        name="assistant",
        model=default_model(),  # swap providers by changing this one line
        instructions="You are a helpful research assistant. Be concise.",
        tools=read_only_tools(),
    )
    result = await Runtime(default_services()).run(
        agent, "What is 1234 * 5678, and what time is it now?"
    )
    print("\n--- final answer ---")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
