"""The Store port — the durability substrate for short-term memory and checkpoints
(they are the same ``RunState``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..kernel.types import RunState


@runtime_checkable
class Store(Protocol):
    async def load(self, thread_id: str) -> "RunState | None": ...

    async def save(self, thread_id: str, state: "RunState") -> int:  # returns revision
        ...

    async def history(self, thread_id: str) -> list["RunState"]:  # time-travel
        ...
