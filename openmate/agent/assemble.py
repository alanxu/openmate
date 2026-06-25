"""``assemble()`` — resolve tool providers into a ready-to-run ``Agent``.

You rarely construct an ``Agent`` field by field. ``assemble()`` resolves a list
of **tool providers** (Shell, MCP, Skills — docs/04) into the agent's
``Harness``, wraps it in the ``Agent`` facade, and owns provider lifecycle
(connecting on entry, closing MCP connections on exit). The yielded ``Agent`` is
the only object you hold — call ``run()``.

A complete assistant is one ``assemble()`` call::

    async with assemble(
            name="email", model=model, services=services,
            system="You are Alan's email assistant. Triage, summarize, draft.",
            providers=[MCPProvider([gmail_server], scope_allowlist=[...]),
                       SkillProvider(["./skills/email"])],
    ) as agent:
        await agent.run("Triage my inbox")
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any, AsyncIterator

from ..kernel.agent import Agent

if TYPE_CHECKING:
    from ..kernel.types import Services
    from ..ports.model import Model
    from ..ports.tool import Tool
    from ..tools.provider import ToolProvider


@asynccontextmanager
async def assemble(
    *,
    name: str,
    system: str,
    model: "Model",
    services: "Services",
    providers: list["ToolProvider"],
    **policy: Any,
) -> AsyncIterator[Agent]:
    """Build an ``Agent`` from providers; tear providers down on exit.

    ``**policy`` is forwarded to the ``Agent`` constructor — the harness policy
    slots (``planner``/``memory``/``context_policy``/``guardrails``/``stop``) plus
    the generation knobs (``max_steps``/``temperature``/``max_tokens``).
    """
    tools: list[Tool] = []
    fragments: list[str] = []
    started: list[ToolProvider] = []
    try:
        for p in providers:
            await p.setup()
            started.append(p)
            tools += await p.tools()
            fragment = p.system_fragment()
            if fragment:
                fragments.append(fragment)
        instructions = "\n\n".join([system, *fragments])
        yield Agent(
            name=name,
            model=model,
            services=services,
            instructions=instructions,
            tools=tools,
            **policy,
        )
    finally:
        for p in reversed(started):  # tear down in reverse order of setup
            await p.teardown()
