"""The agent loop — OpenMate's heart.

A correct, readable async ReAct loop with a hard step cap (PoC / Phase 0 of
designs/02-agent-loop-and-runtime.md): assemble context -> ask the model what to
do -> execute the tools it requested (then loop) or accept its final answer
(then stop) -> checkpoint after every step.

Everything else in the design (interceptors, StopPolicy, pause/resume,
pluggable reasoning strategies) layers additively on top of this spine.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

from ..ports.model import ModelRequest
from .events import (
    CheckpointSaved,
    MessageAdded,
    ModelRequested,
    RunFinished,
    RunStarted,
)
from .executor import ToolExecutor
from .types import Message, RunState, Services, TextPart

if TYPE_CHECKING:
    from .types import Agent, RunResult


def _system_message(agent: "Agent") -> Message:
    return Message("system", [TextPart(agent.instructions)])


def _user_message(text: str) -> Message:
    return Message("user", [TextPart(text)])


class Runtime:
    """Turns a declarative ``Agent`` + input into a ``RunResult`` by driving the loop.

    Owns *mechanism* only; all *policy* is injected via the ``Agent`` and ``Services``.
    """

    def __init__(self, svc: Services) -> None:
        self.svc = svc
        self.executor = ToolExecutor(svc)

    async def run(
        self, agent: "Agent", user_input: str, *, thread_id: str | None = None
    ) -> "RunResult":
        thread_id = thread_id or self.svc.new_id()
        state = await self._init(agent, user_input, thread_id)
        self.svc.bus.emit(RunStarted(thread_id, state.step, self.svc.clock()))

        while state.status == "running" and state.step < agent.max_steps:
            specs = [t.spec for t in agent.tools]
            self.svc.bus.emit(
                ModelRequested(thread_id, state.step, self.svc.clock(), len(state.messages), len(specs))
            )
            req = ModelRequest(
                messages=state.messages,
                tools=specs or None,
                temperature=agent.temperature,
                max_tokens=agent.max_tokens,
            )
            resp = await agent.model.generate(req)

            state = state.with_messages(resp.message).with_usage(resp.usage)
            self.svc.bus.emit(MessageAdded(thread_id, state.step, self.svc.clock(), resp.message))

            calls = resp.message.tool_calls
            if not calls:
                state = state.stop("done", "natural")
                break

            results = await self.executor.dispatch(calls, agent, state)
            state = state.with_messages(Message("tool", list(results))).advance()
            rev = await self.svc.store.save(thread_id, state)
            self.svc.bus.emit(CheckpointSaved(thread_id, state.step, self.svc.clock(), rev))

        if state.status == "running":  # exited via the step cap
            state = state.stop("done", "max_steps")

        result = state.to_result()
        await self.svc.store.save(thread_id, state)
        self.svc.bus.emit(RunFinished(thread_id, state.step, self.svc.clock(), result))
        return result

    async def _init(self, agent: "Agent", user_input: str, thread_id: str) -> RunState:
        """Start a fresh thread, or continue an existing one (short-term memory).

        Each call to ``run`` is one conversational turn with its own step budget,
        so ``step`` resets while the transcript accumulates across turns.
        """
        existing = await self.svc.store.load(thread_id)
        if existing is not None and existing.messages:
            return replace(
                existing,
                messages=[*existing.messages, _user_message(user_input)],
                status="running",
                step=0,
                scratch={},
            )
        return RunState(
            thread_id=thread_id,
            messages=[_system_message(agent), _user_message(user_input)],
        )
