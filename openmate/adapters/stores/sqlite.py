"""SQLite checkpoint store — the local/PoC durability default.

Every checkpoint is a JSON-serialized ``RunState`` row, so a conversation
survives across processes: re-run with the same ``thread_id`` and the transcript
is restored. Keeping every revision also enables time-travel / forking.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from ...kernel.codec import state_from_jsonable, state_to_jsonable
from ...kernel.types import RunState


class SQLiteStore:
    def __init__(self, path: str | Path = "openmate.sqlite") -> None:
        self.path = str(path)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS checkpoints (
                thread_id TEXT NOT NULL,
                rev       INTEGER NOT NULL,
                state     TEXT NOT NULL,
                PRIMARY KEY (thread_id, rev)
            )
            """
        )
        self._conn.commit()

    async def load(self, thread_id: str) -> RunState | None:
        row = self._conn.execute(
            "SELECT state FROM checkpoints WHERE thread_id=? ORDER BY rev DESC LIMIT 1",
            (thread_id,),
        ).fetchone()
        if row is None:
            return None
        return state_from_jsonable(json.loads(row[0]))

    async def save(self, thread_id: str, state: RunState) -> int:
        self._conn.execute(
            "INSERT OR REPLACE INTO checkpoints (thread_id, rev, state) VALUES (?, ?, ?)",
            (thread_id, state.rev, json.dumps(state_to_jsonable(state))),
        )
        self._conn.commit()
        return state.rev

    async def history(self, thread_id: str) -> list[RunState]:
        rows = self._conn.execute(
            "SELECT state FROM checkpoints WHERE thread_id=? ORDER BY rev ASC",
            (thread_id,),
        ).fetchall()
        return [state_from_jsonable(json.loads(r[0])) for r in rows]

    def close(self) -> None:
        self._conn.close()
