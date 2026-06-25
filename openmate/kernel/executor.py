"""The tool executor — dispatches the tool calls a model requests.

PoC form: sequential dispatch with per-tool timeouts. Tool *results* are always
model-legible: a failure becomes a ``ToolResultPart(is_error=True)`` the model
can recover from, never an exception that crashes the run.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..ports.tool import ToolResult
from .events import ToolCallRequested, ToolReturned
from .types import RunContext, TextPart, ToolCallPart, ToolResultPart

if TYPE_CHECKING:
    from .agent import Agent
    from .types import RunState, Services


class ToolExecutor:
    def __init__(self, svc: "Services") -> None:
        self.svc = svc

    async def dispatch(
        self, calls: list[ToolCallPart], agent: "Agent", state: "RunState"
    ) -> list[ToolResultPart]:
        registry = {t.spec.name: t for t in agent.tools}
        out: list[ToolResultPart] = []
        for c in calls:
            self.svc.bus.emit(
                ToolCallRequested(state.thread_id, state.step, self.svc.clock(), c)
            )
            t0 = self.svc.clock()
            tool = registry.get(c.name)
            if tool is None:
                names = ", ".join(registry) or "none"
                res = ToolResult(
                    [TextPart(f"unknown tool '{c.name}'. Available tools: {names}")],
                    is_error=True,
                )
            else:
                ctx = RunContext(state=state, services=self.svc, agent=agent)
                try:
                    res = await asyncio.wait_for(
                        tool.invoke(c.args or {}, ctx), timeout=tool.spec.timeout_s
                    )
                except asyncio.TimeoutError:
                    res = ToolResult(
                        [TextPart(f"tool '{c.name}' timed out after {tool.spec.timeout_s}s")],
                        is_error=True,
                    )
                except Exception as e:  # noqa: BLE001 — surface as a recoverable result
                    res = ToolResult(
                        [TextPart(f"tool '{c.name}' raised {type(e).__name__}: {e}")],
                        is_error=True,
                    )
            ms = (self.svc.clock() - t0) * 1000.0
            part = ToolResultPart(call_id=c.id, content=res.content, is_error=res.is_error)
            self.svc.bus.emit(
                ToolReturned(state.thread_id, state.step, self.svc.clock(), part, ms)
            )
            out.append(part)
        return out
