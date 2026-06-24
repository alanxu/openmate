"""In-memory store — the default for tests and single-process runs."""

from __future__ import annotations

from ...kernel.types import RunState


class InMemoryStore:
    def __init__(self) -> None:
        self._latest: dict[str, RunState] = {}
        self._history: dict[str, list[RunState]] = {}

    async def load(self, thread_id: str) -> RunState | None:
        return self._latest.get(thread_id)

    async def save(self, thread_id: str, state: RunState) -> int:
        self._latest[thread_id] = state
        self._history.setdefault(thread_id, []).append(state)
        return state.rev

    async def history(self, thread_id: str) -> list[RunState]:
        return list(self._history.get(thread_id, []))
