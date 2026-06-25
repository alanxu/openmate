"""OpenMate — a from-scratch, provider-agnostic AI agent (MVP).

The public surface is small. A useful agent is a few lines::

    from openmate import Agent, default_model, default_services
    from openmate.adapters.tools.builtin import read_only_tools

    agent = Agent(name="assistant", model=default_model(),
                  instructions="You are a helpful assistant.",
                  services=default_services(), tools=read_only_tools())
    result = await agent.run("What is 12 * 9?")
    print(result.text)

For anything with external tools or skills, ``assemble()`` wires a list of tool
providers (Shell, MCP, Skills) into a ready-to-run ``Agent``::

    async with assemble(name="email", system="…", model=…, services=…,
                        providers=[MCPProvider([...]), SkillProvider(["./skills"])]) as agent:
        await agent.run("Triage my inbox")
"""

from __future__ import annotations

from .agent.assemble import assemble
from .config import default_model, default_services
from .kernel.agent import Agent, Harness, HumanDecision
from .kernel.types import (
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
from .skills.skill import SkillProvider
from .tools.provider import MCPProvider, NativeProvider, ShellProvider, ToolProvider

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "Harness",
    "HumanDecision",
    "MCPProvider",
    "Message",
    "NativeProvider",
    "RunContext",
    "RunResult",
    "RunState",
    "Services",
    "ShellProvider",
    "SkillProvider",
    "TextPart",
    "ToolCallPart",
    "ToolProvider",
    "ToolResultPart",
    "Usage",
    "assemble",
    "default_model",
    "default_services",
]
