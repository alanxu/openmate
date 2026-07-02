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
        # Libraries: named knowledge bases reusable across threads. A thread's
        # private KB is a library whose id == thread_id; "shared" libraries are
        # created explicitly and attached to many threads (thread_libraries).
        # Chunks are tagged library_id in the vector store; library_knowledge is
        # the catalog + the id-set for precise deletion. See docs/07 §Scoping and
        # shared libraries.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS libraries (
                library_id TEXT PRIMARY KEY,
                name       TEXT,
                kind       TEXT NOT NULL,   -- 'private' | 'shared'
                embedder   TEXT,
                dim        INTEGER,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS library_knowledge (
                library_id TEXT NOT NULL,
                source     TEXT NOT NULL,
                chunk_ids  TEXT NOT NULL,
                added_at   REAL NOT NULL,
                PRIMARY KEY (library_id, source)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS thread_libraries (
                thread_id   TEXT NOT NULL,
                library_id  TEXT NOT NULL,
                attached_at REAL NOT NULL,
                PRIMARY KEY (thread_id, library_id)
            )
            """
        )
        # Projects: a bundle of work directories (file/shell roots) + goals that a
        # task attaches to. Directories widen the agent's file/shell scope; goals
        # steer it. See docs/19 — Workspace.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                name       TEXT,
                goals      TEXT,
                created_at REAL NOT NULL
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS project_directories (
                project_id TEXT NOT NULL,
                path       TEXT NOT NULL,
                added_at   REAL NOT NULL,
                PRIMARY KEY (project_id, path)
            )
            """
        )
        self._ensure_column("threads", "project_id", "TEXT")  # task → attached project
        self._conn.commit()
        self._migrate_thread_knowledge()

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

    # --- libraries: named knowledge bases, reusable across threads -------------

    def _ensure_private_library_row(self, thread_id: str, embedder=None, dim=None) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO libraries (library_id, name, kind, embedder, dim, created_at) "
            "VALUES (?, NULL, 'private', ?, ?, ?)",
            (thread_id, embedder, dim, time.time()),
        )
        self._conn.execute(
            "INSERT OR IGNORE INTO thread_libraries (thread_id, library_id, attached_at) "
            "VALUES (?, ?, ?)",
            (thread_id, thread_id, time.time()),
        )

    def ensure_private_library(self, thread_id: str, *, embedder=None, dim=None) -> None:
        self._ensure_private_library_row(thread_id, embedder, dim)
        self._conn.commit()

    def create_library(self, library_id, name, *, kind="shared", embedder=None, dim=None) -> dict:
        self._conn.execute(
            "INSERT OR REPLACE INTO libraries (library_id, name, kind, embedder, dim, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (library_id, name, kind, embedder, dim, time.time()),
        )
        self._conn.commit()
        return self.get_library(library_id)

    def get_library(self, library_id) -> dict | None:
        row = self._conn.execute(
            "SELECT library_id, name, kind, embedder, dim, created_at FROM libraries WHERE library_id=?",
            (library_id,),
        ).fetchone()
        if not row:
            return None
        return {"library_id": row[0], "name": row[1], "kind": row[2],
                "embedder": row[3], "dim": row[4], "created_at": row[5]}

    def _library_chunk_count(self, library_id) -> int:
        rows = self._conn.execute(
            "SELECT chunk_ids FROM library_knowledge WHERE library_id=?", (library_id,)
        ).fetchall()
        return sum(len(json.loads(r[0])) for r in rows)

    def list_libraries(self, *, kind=None) -> list[dict]:
        q = "SELECT library_id, name, kind, embedder, dim, created_at FROM libraries"
        params: tuple = ()
        if kind:
            q += " WHERE kind=?"
            params = (kind,)
        q += " ORDER BY created_at ASC"
        return [
            {"library_id": r[0], "name": r[1], "kind": r[2], "embedder": r[3], "dim": r[4],
             "created_at": r[5], "n_chunks": self._library_chunk_count(r[0])}
            for r in self._conn.execute(q, params).fetchall()
        ]

    def rename_library(self, library_id, name) -> None:
        self._conn.execute("UPDATE libraries SET name=? WHERE library_id=?", (name, library_id))
        self._conn.commit()

    def promote_library(self, library_id, name) -> dict | None:
        """Promote in place: a private library becomes a named shared one — its id
        (and therefore its already-attached threads and tagged chunks) unchanged."""
        self._conn.execute(
            "UPDATE libraries SET kind='shared', name=? WHERE library_id=?", (name, library_id)
        )
        self._conn.commit()
        return self.get_library(library_id)

    def delete_library(self, library_id) -> list[str]:
        """Drop the library, its sources, and every attachment; return all chunk ids
        so the caller can delete the matching vectors from the vector store."""
        rows = self._conn.execute(
            "SELECT chunk_ids FROM library_knowledge WHERE library_id=?", (library_id,)
        ).fetchall()
        chunk_ids = [cid for r in rows for cid in json.loads(r[0])]
        self._conn.execute("DELETE FROM library_knowledge WHERE library_id=?", (library_id,))
        self._conn.execute("DELETE FROM thread_libraries WHERE library_id=?", (library_id,))
        self._conn.execute("DELETE FROM libraries WHERE library_id=?", (library_id,))
        self._conn.commit()
        return chunk_ids

    # --- thread ↔ library attachments (M:N) ------------------------------------

    def attach_library(self, thread_id, library_id) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO thread_libraries (thread_id, library_id, attached_at) "
            "VALUES (?, ?, ?)",
            (thread_id, library_id, time.time()),
        )
        self._conn.commit()

    def detach_library(self, thread_id, library_id) -> None:
        self._conn.execute(
            "DELETE FROM thread_libraries WHERE thread_id=? AND library_id=?",
            (thread_id, library_id),
        )
        self._conn.commit()

    def list_thread_libraries(self, thread_id) -> list[dict]:
        """Libraries attached to a thread — private first, then shared by attach
        order. This set compiles into the retrieval filter (library_id ∈ …)."""
        rows = self._conn.execute(
            """
            SELECT l.library_id, l.name, l.kind, l.embedder, l.dim
            FROM thread_libraries tl JOIN libraries l ON tl.library_id = l.library_id
            WHERE tl.thread_id=?
            ORDER BY (l.kind='private') DESC, tl.attached_at ASC
            """,
            (thread_id,),
        ).fetchall()
        return [
            {"library_id": r[0], "name": r[1], "kind": r[2], "embedder": r[3], "dim": r[4],
             "n_chunks": self._library_chunk_count(r[0])}
            for r in rows
        ]

    # --- sources within a library ----------------------------------------------

    def add_library_source(self, library_id, source, chunk_ids) -> None:
        self._conn.execute(
            """
            INSERT INTO library_knowledge (library_id, source, chunk_ids, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(library_id, source) DO UPDATE SET
                chunk_ids = excluded.chunk_ids,
                added_at = excluded.added_at
            """,
            (library_id, source, json.dumps(chunk_ids), time.time()),
        )
        self._conn.commit()

    def list_library_sources(self, library_id) -> list[dict]:
        rows = self._conn.execute(
            "SELECT source, chunk_ids, added_at FROM library_knowledge "
            "WHERE library_id=? ORDER BY added_at DESC",
            (library_id,),
        ).fetchall()
        return [{"source": r[0], "chunk_ids": json.loads(r[1]), "added_at": r[2]} for r in rows]

    def remove_library_source(self, library_id, source) -> list[str]:
        row = self._conn.execute(
            "SELECT chunk_ids FROM library_knowledge WHERE library_id=? AND source=?",
            (library_id, source),
        ).fetchone()
        chunk_ids = json.loads(row[0]) if row else []
        self._conn.execute(
            "DELETE FROM library_knowledge WHERE library_id=? AND source=?",
            (library_id, source),
        )
        self._conn.commit()
        return chunk_ids

    # --- per-thread knowledge (back-compat: the thread's private library) -------

    def add_knowledge(self, thread_id: str, source: str, chunk_ids: list[str]) -> None:
        self._ensure_private_library_row(thread_id)
        self.add_library_source(thread_id, source, chunk_ids)

    def list_knowledge(self, thread_id: str) -> list[dict]:
        return self.list_library_sources(thread_id)

    def remove_knowledge(self, thread_id: str, source: str) -> list[str]:
        return self.remove_library_source(thread_id, source)

    def _migrate_thread_knowledge(self) -> None:
        """One-shot, idempotent: fold legacy ``thread_knowledge`` rows into the
        library model (private library == thread_id) so pre-libraries DBs keep
        their knowledge. Skips sources already present in ``library_knowledge``."""
        try:
            rows = self._conn.execute(
                "SELECT thread_id, source, chunk_ids, added_at FROM thread_knowledge"
            ).fetchall()
        except sqlite3.OperationalError:
            return
        for thread_id, source, chunk_ids, added_at in rows:
            if self._conn.execute(
                "SELECT 1 FROM library_knowledge WHERE library_id=? AND source=?",
                (thread_id, source),
            ).fetchone():
                continue
            self._ensure_private_library_row(thread_id)
            self._conn.execute(
                "INSERT OR REPLACE INTO library_knowledge (library_id, source, chunk_ids, added_at) "
                "VALUES (?, ?, ?, ?)",
                (thread_id, source, chunk_ids, added_at),
            )
        self._conn.commit()

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

    # --- projects: work directories + goals, attached to tasks -----------------

    def _ensure_column(self, table: str, col: str, decl: str) -> None:
        cols = [r[1] for r in self._conn.execute(f"PRAGMA table_info({table})").fetchall()]
        if col not in cols:
            self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def create_project(self, project_id, name, *, goals="") -> dict:
        self._conn.execute(
            "INSERT OR REPLACE INTO projects (project_id, name, goals, created_at) VALUES (?, ?, ?, ?)",
            (project_id, name, goals, time.time()),
        )
        self._conn.commit()
        return self.get_project(project_id)

    def get_project(self, project_id) -> dict | None:
        row = self._conn.execute(
            "SELECT project_id, name, goals, created_at FROM projects WHERE project_id=?",
            (project_id,),
        ).fetchone()
        if not row:
            return None
        return {"project_id": row[0], "name": row[1], "goals": row[2] or "", "created_at": row[3],
                "directories": self.list_project_directories(project_id)}

    def list_projects(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT project_id, name, goals, created_at FROM projects ORDER BY created_at ASC"
        ).fetchall()
        return [
            {"project_id": r[0], "name": r[1], "goals": r[2] or "", "created_at": r[3],
             "n_dirs": len(self.list_project_directories(r[0])), "has_goals": bool((r[2] or "").strip())}
            for r in rows
        ]

    def update_project(self, project_id, *, name=None, goals=None) -> dict | None:
        if name is not None:
            self._conn.execute("UPDATE projects SET name=? WHERE project_id=?", (name, project_id))
        if goals is not None:
            self._conn.execute("UPDATE projects SET goals=? WHERE project_id=?", (goals, project_id))
        self._conn.commit()
        return self.get_project(project_id)

    def delete_project(self, project_id) -> None:
        """Delete the project and detach its tasks (their project_id → NULL)."""
        self._conn.execute("DELETE FROM project_directories WHERE project_id=?", (project_id,))
        self._conn.execute("UPDATE threads SET project_id=NULL WHERE project_id=?", (project_id,))
        self._conn.execute("DELETE FROM projects WHERE project_id=?", (project_id,))
        self._conn.commit()

    def add_project_directory(self, project_id, path) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO project_directories (project_id, path, added_at) VALUES (?, ?, ?)",
            (project_id, path, time.time()),
        )
        self._conn.commit()

    def list_project_directories(self, project_id) -> list[str]:
        rows = self._conn.execute(
            "SELECT path FROM project_directories WHERE project_id=? ORDER BY added_at ASC",
            (project_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def remove_project_directory(self, project_id, path) -> None:
        self._conn.execute(
            "DELETE FROM project_directories WHERE project_id=? AND path=?", (project_id, path)
        )
        self._conn.commit()

    # --- task ↔ project ---------------------------------------------------------

    def set_thread_project(self, thread_id, project_id) -> None:
        """Attach (or detach, with project_id=None) a task to a project. Creates the
        thread row if the task has no checkpoint yet."""
        self._conn.execute(
            "INSERT OR IGNORE INTO threads (thread_id, title, created_at, updated_at) VALUES (?, NULL, ?, ?)",
            (thread_id, time.time(), time.time()),
        )
        self._conn.execute("UPDATE threads SET project_id=? WHERE thread_id=?", (project_id, thread_id))
        self._conn.commit()

    def get_thread_project(self, thread_id) -> dict | None:
        row = self._conn.execute(
            "SELECT project_id FROM threads WHERE thread_id=?", (thread_id,)
        ).fetchone()
        if not row or not row[0]:
            return None
        return self.get_project(row[0])

    def list_project_threads(self, project_id) -> list[dict]:
        rows = self._conn.execute(
            "SELECT thread_id, title FROM threads WHERE project_id=? ORDER BY updated_at DESC",
            (project_id,),
        ).fetchall()
        return [{"thread_id": r[0], "title": r[1] or "Untitled"} for r in rows]

    def close(self) -> None:
        self._conn.close()
