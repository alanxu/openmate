"""Shared libraries: vector-store ``$in``/``get``, library-qualified ingest,
SQLite library CRUD + attach, cross-thread reuse, and legacy migration
(docs/07 §Scoping and shared libraries)."""

from __future__ import annotations

import json
import sqlite3
import time

from openmate.adapters.stores.sqlite import SQLiteStore
from openmate.ports.retriever import VectorRecord
from rag.chunking import FixedWindowChunker
from rag.embedders import HashingEmbedder
from rag.loaders import FileLoader
from rag.pipeline import NaivePipeline
from rag.retrievers import DenseRetriever
from rag.stores import InMemoryVectorStore


# --- vector store: $in filter + get() ---------------------------------------
async def test_in_memory_store_in_filter_and_get():
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], "a", {"library_id": "L1"}),
            VectorRecord("b", [0.9, 0.1], "b", {"library_id": "L2"}),
            VectorRecord("c", [0.8, 0.2], "c", {"library_id": "L3"}),
        ]
    )
    hits = await store.query([1.0, 0.0], k=5, filters={"library_id": {"$in": ["L1", "L2"]}})
    assert {d.id for d in hits} == {"a", "b"}  # L3 excluded by the $in filter

    got = await store.get(["a", "c", "missing"])
    assert {r.id for r in got} == {"a", "c"}  # missing ids are skipped
    assert got and got[0].vector  # embeddings come back (for move/copy without re-embed)


# --- pipeline: library-qualified ids ----------------------------------------
async def test_ingest_id_prefix_namespaces_and_is_idempotent(tmp_path):
    (tmp_path / "note.md").write_text("The refund window is thirty days.", encoding="utf-8")
    src = str(tmp_path / "note.md")
    emb = HashingEmbedder(dim=128)
    store = InMemoryVectorStore()
    pipe = NaivePipeline(FileLoader(), FixedWindowChunker(120, 20), emb, store)

    a = await pipe.ingest(src, extra_metadata={"library_id": "LA"}, id_prefix="LA:")
    b = await pipe.ingest(src, extra_metadata={"library_id": "LB"}, id_prefix="LB:")
    assert all(cid.startswith("LA:") for cid in a.chunk_ids)
    assert all(cid.startswith("LB:") for cid in b.chunk_ids)
    # same source in two libraries → two physical copies, no id collision
    assert await store.count() == len(a.chunk_ids) + len(b.chunk_ids)

    before = await store.count()
    await pipe.ingest(src, extra_metadata={"library_id": "LA"}, id_prefix="LA:")
    assert await store.count() == before  # re-ingest into LA overwrote in place


async def test_cross_thread_reuse_via_library_filter(tmp_path):
    (tmp_path / "shared.md").write_text(
        "Enterprise refunds are prorated within thirty days of the invoice.", encoding="utf-8"
    )
    emb = HashingEmbedder(dim=256)
    store = InMemoryVectorStore()
    pipe = NaivePipeline(FileLoader(), FixedWindowChunker(120, 20), emb, store)
    await pipe.ingest(
        str(tmp_path / "shared.md"),
        extra_metadata={"library_id": "lib_shared"},
        id_prefix="lib_shared:",
    )
    retriever = DenseRetriever(emb, store)

    # a thread that attaches lib_shared sees it (private id + shared id in the $in set)
    hits = await retriever.retrieve(
        "refund window", k=3, filters={"library_id": {"$in": ["t_other", "lib_shared"]}}
    )
    assert hits and hits[0].metadata["library_id"] == "lib_shared"
    # a thread that hasn't attached it sees nothing
    none = await retriever.retrieve("refund window", k=3, filters={"library_id": {"$in": ["t_other"]}})
    assert none == []


# --- SQLite: library CRUD + attach ------------------------------------------
def test_sqlite_library_crud_and_attach(tmp_path):
    s = SQLiteStore(str(tmp_path / "om.sqlite"))
    try:
        lib = s.create_library("lib_x", "Docs", embedder="hashing", dim=256)
        assert lib["kind"] == "shared" and lib["name"] == "Docs"
        s.add_library_source("lib_x", "a.md", ["lib_x:a.md#0", "lib_x:a.md#1"])
        assert s.list_libraries()[0]["n_chunks"] == 2

        s.ensure_private_library("t1")
        s.ensure_private_library("t2")
        s.attach_library("t1", "lib_x")
        s.attach_library("t2", "lib_x")
        assert {l["library_id"] for l in s.list_thread_libraries("t1")} == {"t1", "lib_x"}
        # private library sorts first
        assert s.list_thread_libraries("t1")[0]["kind"] == "private"

        s.detach_library("t1", "lib_x")
        assert {l["library_id"] for l in s.list_thread_libraries("t1")} == {"t1"}
        assert "lib_x" in {l["library_id"] for l in s.list_thread_libraries("t2")}

        removed = s.remove_library_source("lib_x", "a.md")
        assert removed == ["lib_x:a.md#0", "lib_x:a.md#1"]

        s.add_library_source("t2", "n.md", ["t2:n.md#0"])
        promoted = s.promote_library("t2", "T2 shared")
        assert promoted["kind"] == "shared" and promoted["name"] == "T2 shared"
    finally:
        s.close()


def test_sqlite_backcompat_and_delete(tmp_path):
    s = SQLiteStore(str(tmp_path / "om.sqlite"))
    try:
        # legacy per-thread API routes to the thread's private library
        s.add_knowledge("t9", "d.md", ["t9:d.md#0", "t9:d.md#1"])
        got = s.list_knowledge("t9")
        assert got and got[0]["source"] == "d.md"
        assert {l["library_id"] for l in s.list_thread_libraries("t9")} == {"t9"}

        s.create_library("lib_z", "Z", embedder="hashing", dim=256)
        s.add_library_source("lib_z", "z.md", ["lib_z:z.md#0"])
        s.attach_library("t9", "lib_z")
        chunk_ids = s.delete_library("lib_z")
        assert chunk_ids == ["lib_z:z.md#0"]
        assert s.get_library("lib_z") is None
        assert "lib_z" not in {l["library_id"] for l in s.list_thread_libraries("t9")}
    finally:
        s.close()


def test_sqlite_migrates_legacy_thread_knowledge(tmp_path):
    db = str(tmp_path / "legacy.sqlite")
    SQLiteStore(db).close()  # create schema (incl. legacy thread_knowledge)
    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO thread_knowledge (thread_id, source, chunk_ids, added_at) VALUES (?, ?, ?, ?)",
        ("t_old", "old.md", json.dumps(["old.md#0"]), time.time()),
    )
    conn.commit()
    conn.close()

    s = SQLiteStore(db)  # reopening runs the one-shot migration
    try:
        srcs = s.list_library_sources("t_old")
        assert srcs and srcs[0]["source"] == "old.md"
        assert {l["library_id"] for l in s.list_thread_libraries("t_old")} == {"t_old"}
    finally:
        s.close()
