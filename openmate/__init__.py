"""OpenMate — a from-scratch, provider-agnostic AI agent (MVP).

The public surface is small. A useful agent is a few lines::

    from openmate import Agent, Runtime, default_model, default_services
    from openmate.adapters.tools.builtin import read_only_tools

    agent = Agent(name="assistant", model=default_model(),
                  instructions="You are a helpful assistant.",
                  tools=read_only_tools())
    result = await Runtime(default_services()).run(agent, "What is 12 * 9?")
    print(result.text)
"""

from __future__ import annotations

from .config import default_model, default_services
from .kernel.runtime import Runtime
from .kernel.types import (
    Agent,
    Message,
    RunContext,
    RunResult,
    RunState,
    Services,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Message",
    "RunContext",
    "RunResult",
    "RunState",
    "Runtime",
    "Services",
    "TextPart",
    "ToolCallPart",
    "ToolResultPart",
    "Usage",
    "default_model",
    "default_services",
]
