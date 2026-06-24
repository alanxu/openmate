"""Defining your own tool with the @tool decorator, then letting the agent use it.

Requires an API key (see .env). Run:  python examples/custom_tool.py
"""

import asyncio

from openmate import Agent, Runtime, default_model, default_services
from openmate.adapters.tools.builtin import calculator
from openmate.adapters.tools.native import tool


@tool(side_effecting=False, idempotent=True)
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in a piece of text."""
    return len(text.split())


async def main() -> None:
    agent = Agent(
        name="assistant",
        model=default_model(),
        instructions="You are a helpful assistant. Use tools when they help.",
        tools=[calculator, word_count],
    )
    result = await Runtime(default_services()).run(
        agent,
        "How many words are in 'the quick brown fox jumps', and what is that count squared?",
    )
    print("\n--- final answer ---")
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
