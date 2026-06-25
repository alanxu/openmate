"""Run the full agent loop with NO API key, using the deterministic FakeModel.

This proves the kernel works end to end and shows the determinism property the
design is built around. Run:  python examples/offline_fake.py
"""

import asyncio

from openmate import Agent, Services
from openmate.adapters.models.fake import text_response, tool_call_response, FakeModel
from openmate.adapters.stores.memory import InMemoryStore
from openmate.adapters.tools.builtin import calculator
from openmate.kernel.events import EventBus
from openmate.ports.tracer import NullTracer


async def main() -> None:
    # A scripted "conversation": first the model calls the calculator, then it
    # reads the tool result and produces a final answer.
    model = FakeModel(
        [
            tool_call_response("c1", "calculator", {"expression": "6 * 7"}),
            text_response("The answer is 42."),
        ]
    )
    services = Services(store=InMemoryStore(), tracer=NullTracer(), bus=EventBus())
    agent = Agent(
        name="assistant",
        model=model,
        instructions="You are a helpful assistant.",
        services=services,
        tools=[calculator],
    )
    result = await agent.run("What is 6 * 7?")

    print("final:", result.text)
    print("reason:", result.reason, "| steps:", result.steps)
    assert result.text == "The answer is 42."


if __name__ == "__main__":
    asyncio.run(main())
