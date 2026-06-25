"""Shared test helpers."""

from __future__ import annotations

from typing import Callable

from openmate.adapters.stores.memory import InMemoryStore
from openmate.kernel.events import Event, EventBus
from openmate.kernel.types import (
    Message,
    Services,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from openmate.ports.model import ModelCapabilities, ModelRequest, ModelResponse
from openmate.ports.tracer import NullTracer


def make_services() -> tuple[Services, list[Event]]:
    """A deterministic ``Services`` (counter clock + ids) plus a captured event list."""
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(events.append)

    tick = {"n": 0}

    def clock() -> float:
        tick["n"] += 1
        return float(tick["n"])

    seq = {"n": 0}

    def new_id() -> str:
        seq["n"] += 1
        return f"id{seq['n']}"

    svc = Services(store=InMemoryStore(), tracer=NullTracer(), bus=bus, clock=clock, new_id=new_id)
    return svc, events


# --- a deterministic, observation-aware stand-in for a real model ------------
class ReactiveModel:
    """A fake model driven by a *policy* — deterministic, but genuinely reactive.

    Unlike ``FakeModel`` (a flat script), the policy is called as
    ``policy(messages, turn) -> ModelResponse`` each loop step, so it can branch on
    the tool *observations* in ``messages`` (and on the 0-based ``turn`` index).
    That lets a cassette faithfully test observation→action dependency, error
    recovery, and plan-driven execution without a network call.
    """

    name = "reactive"
    capabilities = ModelCapabilities(tool_calling=True, parallel_tools=True, streaming=False)

    def __init__(self, policy: Callable[[list[Message], int], ModelResponse]) -> None:
        self.policy = policy
        self.requests: list[ModelRequest] = []

    async def generate(self, req: ModelRequest) -> ModelResponse:
        turn = len(self.requests)
        self.requests.append(req)
        return self.policy(req.messages, turn)


def tool_observations(messages: list[Message]) -> list[tuple[str, str, bool]]:
    """Every tool result so far as ``(call_id, text, is_error)``, in order."""
    out: list[tuple[str, str, bool]] = []
    for m in messages:
        if m.role == "tool":
            for p in m.content:
                if isinstance(p, ToolResultPart):
                    text = "".join(c.text for c in p.content if isinstance(c, TextPart))
                    out.append((p.call_id, text, p.is_error))
    return out


def tools_called(messages: list[Message]) -> list[str]:
    """Names of every tool the assistant has called so far, in order."""
    return [
        p.name
        for m in messages
        if m.role == "assistant"
        for p in m.content
        if isinstance(p, ToolCallPart)
    ]


def text_response(text: str) -> ModelResponse:
    return ModelResponse(
        Message("assistant", [TextPart(text)]), Usage(1, 1), finish_reason="stop"
    )


def call_response(call_id: str, name: str, args: dict) -> ModelResponse:
    return ModelResponse(
        Message("assistant", [ToolCallPart(call_id, name, args)]),
        Usage(1, 1),
        finish_reason="tool_calls",
    )


def calls_response(calls: list[tuple[str, str, dict]]) -> ModelResponse:
    """A single assistant message that requests several tools at once (batch step)."""
    return ModelResponse(
        Message("assistant", [ToolCallPart(i, n, a) for i, n, a in calls]),
        Usage(1, 1),
        finish_reason="tool_calls",
    )
