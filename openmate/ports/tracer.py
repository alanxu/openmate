"""The Tracer port — projects the canonical event stream onto an observability backend.

Because traces are derived from events, switching backends never touches
application code.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, ContextManager, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..kernel.events import Event


@runtime_checkable
class Tracer(Protocol):
    def record(self, event: "Event") -> None: ...

    def span(self, name: str, **attrs) -> ContextManager: ...


class NullTracer:
    """A tracer that does nothing — handy in tests."""

    def record(self, event: "Event") -> None:
        return None

    def span(self, name: str, **attrs) -> ContextManager:
        return nullcontext()
