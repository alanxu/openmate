"""The agent loop engine — the spine behind ``Agent.run()``.

``Agent`` ([01]) is the facade callers hold; this module is the *internal* engine
its ``run``/``stream``/``resume``/``cancel`` delegate to. Callers never import it
directly. It owns *mechanism* only — step sequencing, checkpointing, streaming,
termination — while all *policy* (model, tools, instructions, step budget) is
carried on the ``Agent`` it's handed.

PoC / Phase 0 of docs/02-agent-loop-and-runtime.md: a correct, readable async
ReAct loop with a hard step cap. The interceptor chain, composable ``StopPolicy``,
and full pause/resume are additive layers above this spine.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import TYPE_CHECKING, AsyncIterator

from ..ports.model import ModelRequest
from .errors import OpenMateError
from .events import (
    CheckpointSaved,
    Event,
    MessageAdded,
    ModelRequested,
    RunFinished,
    RunStarted,
)
from .executor import ToolExecutor
from .types import Message, RunState, TextPart

if TYPE_CHECKING:
    from .agent import Agent, HumanDecision
    from .types import RunResult

# Threads asked to cancel, observed cooperatively at each step. A minimal stand-in
# for the Phase-3 ``Cancelled`` stop policy that reads a flag from the store.
_cancelled: set[str] = set()

# A private sentinel that marks the end of a stream, distinct from any Event.
_DONE = object()


def _system_message(agent: "Agent") -> Message:
    return Message("system", [TextPart(agent.instructions)])


def _user_message(text: str) -> Message:
    return Message("user", [TextPart(text)])


async def drive(
    agent: "Agent",
    user_input: str,
    *,
    thread_id: str | None = None,
    chain: list | None = None,  # interceptor chain — reserved for Phase 1
) -> "RunResult":
    """Start (or continue) a thread with ``user_input`` and run it to a result."""
    svc = agent.services
    thread_id = thread_id or svc.new_id()
    state = await _init(agent, user_input, thread_id)
    svc.bus.emit(RunStarted(thread_id, state.step, svc.clock()))
    state = await _loop(agent, state)
    return await _finish(agent, state)


async def resume(
    agent: "Agent", thread_id: str, decision: "HumanDecision | None" = None
) -> "RunResult":
    """Re-enter the loop on a checkpointed thread.

    The PoC has no ``Pause`` outcome yet, so a ``decision`` is recorded on the
    state and the loop simply continues from the last checkpoint — the durable
    substrate HITL (Phase 3) builds on.
    """
    svc = agent.services
    state = await svc.store.load(thread_id)
    if state is None:
        raise OpenMateError(f"cannot resume: no checkpoint for thread '{thread_id}'")
    state = replace(state, status="running")
    if decision is not None:
        state = replace(
            state, scratch={**state.scratch, "resume_decision": decision.action}
        )
    svc.bus.emit(RunStarted(thread_id, state.step, svc.clock()))
    state = await _loop(agent, state)
    return await _finish(agent, state)


async def cancel(agent: "Agent", thread_id: str) -> None:
    """Flag a thread for cooperative cancellation at its next step boundary."""
    _cancelled.add(thread_id)


async def stream(
    agent: "Agent", user_input: str, *, thread_id: str | None = None
) -> AsyncIterator[Event]:
    """Drive a run and yield each ``Event`` as it is emitted.

    The synchronous bus is bridged to the async caller through a queue: ``drive``
    runs as a task, every emitted event lands in the queue, and the generator
    yields them in order until the run finishes (then re-raises any failure).
    """
    svc = agent.services
    queue: asyncio.Queue = asyncio.Queue()
    handler = svc.bus.subscribe(queue.put_nowait)

    async def _run() -> None:
        try:
            await drive(agent, user_input, thread_id=thread_id)
        finally:
            queue.put_nowait(_DONE)

    task = asyncio.create_task(_run())
    try:
        while True:
            ev = await queue.get()
            if ev is _DONE:
                break
            yield ev
    finally:
        svc.bus.unsubscribe(handler)
        await task  # join and surface any exception raised inside the run


async def _loop(agent: "Agent", state: RunState) -> RunState:
    """The ReAct spine: decide → act (then loop) or finish → checkpoint → stop."""
    svc = agent.services
    executor = ToolExecutor(svc)
    tid = state.thread_id

    while state.status == "running" and state.step < agent.max_steps:
        if tid in _cancelled:
            _cancelled.discard(tid)
            return state.stop("done", "cancelled")

        specs = [t.spec for t in agent.tools]
        svc.bus.emit(
            ModelRequested(tid, state.step, svc.clock(), len(state.messages), len(specs))
        )
        req = ModelRequest(
            messages=state.messages,
            tools=specs or None,
            temperature=agent.temperature,
            max_tokens=agent.max_tokens,
        )
        resp = await agent.model.generate(req)

        state = state.with_messages(resp.message).with_usage(resp.usage)
        svc.bus.emit(MessageAdded(tid, state.step, svc.clock(), resp.message))

        calls = resp.message.tool_calls
        if not calls:
            return state.stop("done", "natural")

        results = await executor.dispatch(calls, agent, state)
        state = state.with_messages(Message("tool", list(results))).advance()
        rev = await svc.store.save(tid, state)
        svc.bus.emit(CheckpointSaved(tid, state.step, svc.clock(), rev))

    if state.status == "running":  # exited via the step cap
        state = state.stop("done", "max_steps")
    return state


async def _finish(agent: "Agent", state: RunState) -> "RunResult":
    svc = agent.services
    result = state.to_result()
    await svc.store.save(state.thread_id, state)
    svc.bus.emit(RunFinished(state.thread_id, state.step, svc.clock(), result))
    return result


async def _init(agent: "Agent", user_input: str, thread_id: str) -> RunState:
    """Start a fresh thread, or continue an existing one (short-term memory).

    Each ``drive`` call is one conversational turn with its own step budget, so
    ``step`` resets while the transcript accumulates across turns.
    """
    existing = await agent.services.store.load(thread_id)
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
