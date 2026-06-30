"""SQLite checkpoint store — the local/PoC durability default.

Every checkpoint is a JSON-serialized ``RunState`` row, so a conversation
survives across processes: re-run with the same ``thread_id`` and the transcript
is restored. Keeping every revision also enables time-travel / forking.
"""

from __future__ import annotations

import json
import sqlite3
import time
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
        # Thread index — additive, adapter-local bookkeeping for UI history lists.
        # Not part of the ``Store`` protocol: ``RunState`` carries no timestamp, so
        # this is sourced here at the edge rather than threaded through the kernel.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS threads (
                thread_id  TEXT PRIMARY KEY,
                title      TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        # Per-thread knowledge sources (UI: '+' → Add knowledge) — which ingested
        # sources belong to which thread, and the chunk ids they produced, so a
        # later "remove this doc" can delete the matching vectors precisely.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_knowledge (
                thread_id  TEXT NOT NULL,
                source     TEXT NOT NULL,
                chunk_ids  TEXT NOT NULL,
                added_at   REAL NOT NULL,
                PRIMARY KEY (thread_id, source)
            )
            """
        )
        # Per-thread editable folders (UI: '+' → Add folder for editing) — the
        # extra filesystem roots a thread's agent is allowed to read/write beyond
        # the server's cwd. See ``openmate/adapters/tools/builtin.py:make_file_tools``.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_folders (
                thread_id TEXT NOT NULL,
                path      TEXT NOT NULL,
                added_at  REAL NOT NULL,
                PRIMARY KEY (thread_id, path)
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
        self._touch_thread(thread_id, state)
        self._conn.commit()
        return state.rev

    def _touch_thread(self, thread_id: str, state: RunState) -> None:
        now = time.time()
        title = self._derive_title(state)
        self._conn.execute(
            """
            INSERT INTO threads (thread_id, title, created_at, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET
                updated_at = excluded.updated_at,
                title = COALESCE(threads.title, excluded.title)
            """,
            (thread_id, title, now, now),
        )

    @staticmethod
    def _derive_title(state: RunState, max_len: int = 60) -> str | None:
        """First user message, truncated — a reasonable default chat title."""
        for m in state.messages:
            if m.role == "user":
                text = m.text.strip()
                if text:
                    return text[:max_len]
        return None

    def list_threads(self, limit: int = 100) -> list[dict]:
        """Most-recently-updated threads first — backs the UI's history sidebar."""
        rows = self._conn.execute(
            "SELECT thread_id, title, created_at, updated_at FROM threads "
            "ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {"thread_id": r[0], "title": r[1] or "Untitled", "created_at": r[2], "updated_at": r[3]}
            for r in rows
        ]

    async def history(self, thread_id: str) -> list[RunState]:
        rows = self._conn.execute(
            "SELECT state FROM checkpoints WHERE thread_id=? ORDER BY rev ASC",
            (thread_id,),
        ).fetchall()
        return [state_from_jsonable(json.loads(r[0])) for r in rows]

    # --- per-thread knowledge (UI: '+' → Add knowledge) ------------------------

    def add_knowledge(self, thread_id: str, source: str, chunk_ids: list[str]) -> None:
        self._conn.execute(
            """
            INSERT INTO thread_knowledge (thread_id, source, chunk_ids, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(thread_id, source) DO UPDATE SET
                chunk_ids = excluded.chunk_ids,
                added_at = excluded.added_at
            """,
            (thread_id, source, json.dumps(chunk_ids), time.time()),
        )
        self._conn.commit()

    def list_knowledge(self, thread_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT source, chunk_ids, added_at FROM thread_knowledge "
            "WHERE thread_id=? ORDER BY added_at DESC",
            (thread_id,),
        ).fetchall()
        return [
            {"source": r[0], "chunk_ids": json.loads(r[1]), "added_at": r[2]}
            for r in rows
        ]

    def remove_knowledge(self, thread_id: str, source: str) -> list[str]:
        """Delete the tracking row and return the chunk ids it had, so the caller
        can also remove those vectors from the ``VectorStore``."""
        row = self._conn.execute(
            "SELECT chunk_ids FROM thread_knowledge WHERE thread_id=? AND source=?",
            (thread_id, source),
        ).fetchone()
        chunk_ids = json.loads(row[0]) if row else []
        self._conn.execute(
            "DELETE FROM thread_knowledge WHERE thread_id=? AND source=?",
            (thread_id, source),
        )
        self._conn.commit()
        return chunk_ids

    # --- per-thread editable folders (UI: '+' → Add folder for editing) --------

    def add_folder(self, thread_id: str, path: str) -> None:
        self._conn.execute(
            """
            INSERT INTO thread_folders (thread_id, path, added_at)
            VALUES (?, ?, ?)
            ON CONFLICT(thread_id, path) DO UPDATE SET added_at = excluded.added_at
            """,
            (thread_id, path, time.time()),
        )
        self._conn.commit()

    def list_folders(self, thread_id: str) -> list[str]:
        rows = self._conn.execute(
            "SELECT path FROM thread_folders WHERE thread_id=? ORDER BY added_at ASC",
            (thread_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def remove_folder(self, thread_id: str, path: str) -> None:
        self._conn.execute(
            "DELETE FROM thread_folders WHERE thread_id=? AND path=?",
            (thread_id, path),
        )
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()
