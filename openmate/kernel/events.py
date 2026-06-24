"""The event stream — the system's public narration.

Every state transition emits a typed ``Event``. The tracer, UI, evaluator, and
persistence layer are all just consumers of this stream, which is what makes the
system observable and replayable by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .types import Message, RunResult, ToolCallPart, ToolResultPart


@dataclass
class Event:
    thread_id: str
    step: int
    ts: float  # from services.clock


@dataclass
class RunStarted(Event):
    pass


@dataclass
class MessageAdded(Event):
    message: "Message"


@dataclass
class ModelRequested(Event):
    n_messages: int
    n_tools: int


@dataclass
class ToolCallRequested(Event):
    call: "ToolCallPart"


@dataclass
class ToolReturned(Event):
    result: "ToolResultPart"
    ms: float


@dataclass
class CheckpointSaved(Event):
    rev: int


@dataclass
class RunFinished(Event):
    result: "RunResult"


class EventBus:
    """A synchronous, in-process fan-out bus (sufficient for the PoC)."""

    def __init__(self) -> None:
        self._subs: list[Callable[[Event], None]] = []

    def subscribe(self, fn: Callable[[Event], None]) -> Callable[[Event], None]:
        self._subs.append(fn)
        return fn

    def emit(self, ev: Event) -> None:
        for fn in list(self._subs):
            fn(ev)
