"""Projects: work directories + goals, and task↔project attachment (docs/19)."""

from __future__ import annotations

import sqlite3

from openmate.adapters.stores.sqlite import SQLiteStore


def test_project_crud_and_directories(tmp_path):
    s = SQLiteStore(str(tmp_path / "om.sqlite"))
    try:
        p = s.create_project("proj_1", "Website revamp", goals="Ship the new pricing page")
        assert p["name"] == "Website revamp" and p["goals"].startswith("Ship")

        s.add_project_directory("proj_1", str(tmp_path))
        s.add_project_directory("proj_1", str(tmp_path))  # idempotent
        assert s.list_project_directories("proj_1") == [str(tmp_path)]
        assert s.get_project("proj_1")["directories"] == [str(tmp_path)]

        row = s.list_projects()[0]
        assert row["n_dirs"] == 1 and row["has_goals"] is True

        s.update_project("proj_1", name="Website v2", goals="")
        p2 = s.get_project("proj_1")
        assert p2["name"] == "Website v2" and p2["goals"] == ""
        assert s.list_projects()[0]["has_goals"] is False

        s.remove_project_directory("proj_1", str(tmp_path))
        assert s.list_project_directories("proj_1") == []
    finally:
        s.close()


def test_task_project_attach_and_delete_detaches(tmp_path):
    s = SQLiteStore(str(tmp_path / "om.sqlite"))
    try:
        s.create_project("proj_x", "X")
        s.set_thread_project("t1", "proj_x")
        s.set_thread_project("t2", "proj_x")
        assert s.get_thread_project("t1")["project_id"] == "proj_x"
        assert {t["thread_id"] for t in s.list_project_threads("proj_x")} == {"t1", "t2"}

        s.set_thread_project("t1", None)  # detach one
        assert s.get_thread_project("t1") is None

        s.delete_project("proj_x")  # deleting detaches the rest
        assert s.get_project("proj_x") is None
        assert s.get_thread_project("t2") is None
        assert s.list_project_threads("proj_x") == []
    finally:
        s.close()


def test_project_id_column_added_to_legacy_db(tmp_path):
    db = str(tmp_path / "old.sqlite")
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE threads (thread_id TEXT PRIMARY KEY, title TEXT, created_at REAL, updated_at REAL)")
    conn.execute("INSERT INTO threads VALUES ('t9', 'T', 0, 0)")
    conn.commit()
    conn.close()

    s = SQLiteStore(db)  # __init__ runs _ensure_column("threads", "project_id", ...)
    try:
        cols = [r[1] for r in s._conn.execute("PRAGMA table_info(threads)").fetchall()]
        assert "project_id" in cols
        s.create_project("p", "P")
        s.set_thread_project("t9", "p")
        assert s.get_thread_project("t9")["project_id"] == "p"
    finally:
        s.close()
