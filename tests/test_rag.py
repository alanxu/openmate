"""RAG: chunking, embedding, stores, retrieval, naive + agentic generation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from openmate.adapters.models.fake import FakeModel, text_response, tool_call_response
from openmate.ports.retriever import RawDoc, VectorRecord
from rag.chunking import FixedWindowChunker
from rag.embedders import HashingEmbedder
from rag.generate import agentic_answer, answer
from rag.loaders import FileLoader
from rag.pipeline import NaivePipeline
from rag.retrievers import DenseRetriever
from rag.stores import InMemoryVectorStore
from rag.tools import RetrieveTool

_has_chroma = importlib.util.find_spec("chromadb") is not None


# --- chunking ----------------------------------------------------------------
def test_chunker_overlap_and_deterministic_ids():
    doc = RawDoc(id="d", text="word " * 500, metadata={"source": "d"})
    chunks = FixedWindowChunker(size=200, overlap=40).split(doc)
    assert len(chunks) > 1
    assert [c.id for c in chunks] == [f"d#{i}" for i in range(len(chunks))]
    assert chunks[0].metadata["chunk_index"] == 0


def test_chunker_is_idempotent():
    doc = RawDoc(id="d", text="alpha beta gamma " * 100)
    a = FixedWindowChunker(size=120, overlap=20).split(doc)
    b = FixedWindowChunker(size=120, overlap=20).split(doc)
    assert [(c.id, c.text) for c in a] == [(c.id, c.text) for c in b]


# --- embeddings --------------------------------------------------------------
async def test_hashing_embedder_is_deterministic_and_normalized():
    emb = HashingEmbedder(dim=64)
    v1 = (await emb.embed(["the agent loop"]))[0]
    v2 = (await emb.embed(["the agent loop"]))[0]
    assert v1 == v2  # deterministic (hashlib, not salted hash())
    assert abs(sum(x * x for x in v1) ** 0.5 - 1.0) < 1e-9  # L2-normalized
    assert emb.dim == 64


async def test_hashing_embedder_similar_text_scores_higher():
    emb = HashingEmbedder(dim=512)
    q, near, far = await emb.embed(
        ["agent loop react cycle", "the agent loop runs a react cycle", "banana smoothie recipe"]
    )

    def cos(a, b):
        return sum(x * y for x, y in zip(a, b))

    assert cos(q, near) > cos(q, far)


# --- in-memory store ---------------------------------------------------------
async def test_in_memory_store_ranks_by_cosine_and_filters():
    store = InMemoryVectorStore()
    await store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], "a", {"src": "x"}),
            VectorRecord("b", [0.0, 1.0], "b", {"src": "y"}),
            VectorRecord("c", [0.9, 0.1], "c", {"src": "x"}),
        ]
    )
    assert await store.count() == 3
    top = await store.query([1.0, 0.0], k=2)
    assert [d.id for d in top] == ["a", "c"]  # closest first
    filtered = await store.query([1.0, 0.0], k=5, filters={"src": "y"})
    assert [d.id for d in filtered] == ["b"]


async def test_in_memory_store_persists_to_json(tmp_path):
    path = str(tmp_path / "vs.json")
    s1 = InMemoryVectorStore(path=path)
    await s1.upsert([VectorRecord("a", [1.0, 0.0], "hello", {"src": "x"})])
    s2 = InMemoryVectorStore(path=path)  # reload from disk
    assert await s2.count() == 1
    assert (await s2.query([1.0, 0.0], k=1))[0].text == "hello"
    await s2.delete(None)
    assert await s2.count() == 0


# --- pipeline + retrieval ----------------------------------------------------
def _corpus(tmp_path):
    (tmp_path / "loop.md").write_text(
        "# The Agent Loop\nThe loop decides with the model, acts with tools, then "
        "checkpoints and checks the stop policy.",
        encoding="utf-8",
    )
    (tmp_path / "rag.md").write_text(
        "# Retrieval\nRAG retrieves chunks from a vector store and grounds the answer.",
        encoding="utf-8",
    )
    return tmp_path


async def _ingested_retriever(tmp_path):
    emb = HashingEmbedder(dim=512)
    store = InMemoryVectorStore()
    pipeline = NaivePipeline(FileLoader(), FixedWindowChunker(120, 20), emb, store)
    report = await pipeline.ingest(str(_corpus(tmp_path)))
    return DenseRetriever(emb, store), store, report


async def test_pipeline_ingests_and_retrieves(tmp_path):
    retriever, store, report = await _ingested_retriever(tmp_path)
    assert report.documents == 2 and report.chunks >= 2
    hits = await retriever.retrieve("how does the agent loop work", k=1)
    assert hits and hits[0].source == "loop.md"


async def test_ingestion_is_idempotent(tmp_path):
    retriever, store, _ = await _ingested_retriever(tmp_path)
    before = await store.count()
    # re-ingest the same corpus through a fresh pipeline over the same store
    emb = HashingEmbedder(dim=512)
    await NaivePipeline(FileLoader(), FixedWindowChunker(120, 20), emb, store).ingest(
        str(tmp_path)
    )
    assert await store.count() == before  # overwrote, didn't duplicate


def test_loader_skips_hidden_dirs(tmp_path):
    (tmp_path / "keep.md").write_text("real content", encoding="utf-8")
    hidden = tmp_path / ".rag"
    hidden.mkdir()
    (hidden / "memory.json").write_text('[{"id":"x"}]', encoding="utf-8")
    ids = [d.id for d in FileLoader().load(str(tmp_path))]
    assert ids == ["keep.md"]  # the .rag store dir is skipped


# --- retrieve tool -----------------------------------------------------------
async def test_retrieve_tool_formats_and_records_sources(tmp_path):
    retriever, _, _ = await _ingested_retriever(tmp_path)
    tool = RetrieveTool(retriever, k=2)
    assert tool.spec.name == "rag_search" and tool.spec.side_effecting is False

    from types import SimpleNamespace

    ctx = SimpleNamespace(state=SimpleNamespace(scratch={}))
    res = await tool.invoke({"query": "agent loop"}, ctx)
    assert not res.is_error
    assert "source=loop.md" in res.content[0].text
    assert ctx.state.scratch["rag_sources"]  # recorded for agentic reporting


async def test_retrieve_tool_missing_query_is_error():
    res = await RetrieveTool(InMemoryVectorStore() and DenseRetriever(HashingEmbedder(8), InMemoryVectorStore())).invoke({}, None)
    assert res.is_error


# --- generation: naive + agentic --------------------------------------------
async def test_naive_answer_grounds_and_returns_sources(tmp_path):
    retriever, _, _ = await _ingested_retriever(tmp_path)
    model = FakeModel([text_response("The loop decides, acts, then checkpoints [1].")])
    result = await answer("what is the agent loop?", retriever, model, k=2)
    assert "[1]" in result["answer"]
    assert result["sources"] and result["sources"][0]["source"] in {"loop.md", "rag.md"}
    # the model actually received the retrieved sources in its prompt
    assert "Sources:" in model.requests[0].messages[-1].text


async def test_naive_answer_handles_empty_index():
    retriever = DenseRetriever(HashingEmbedder(8), InMemoryVectorStore())
    result = await answer("anything?", retriever, FakeModel([text_response("unused")]))
    assert result["sources"] == []
    assert "couldn't find" in result["answer"].lower()


async def test_agentic_answer_loops_search_then_answers(tmp_path):
    retriever, _, _ = await _ingested_retriever(tmp_path)
    script = [
        tool_call_response("c1", "rag_search", {"query": "agent loop", "k": 2}),
        tool_call_response("c2", "rag_search", {"query": "stop policy", "k": 2}),
        text_response("The loop runs decide→act→checkpoint→stop [1][2]."),
    ]
    result = await agentic_answer("explain the loop and stopping", retriever, FakeModel(script))
    assert result["searches"] == 2  # it retrieved twice before answering
    assert result["steps"] == 2
    assert "[1]" in result["answer"]
    assert result["sources"]  # populated from scratch by RetrieveTool


# --- chroma (the default store) ---------------------------------------------
@pytest.mark.skipif(not _has_chroma, reason="chromadb not installed")
async def test_chroma_store_roundtrip(tmp_path):
    from rag.stores import ChromaVectorStore

    store = ChromaVectorStore(collection="rag_test", path=str(tmp_path / "chroma"))
    await store.delete(None)
    await store.upsert(
        [
            VectorRecord("a", [1.0, 0.0], "alpha", {"src": "x"}),
            VectorRecord("b", [0.0, 1.0], "beta", {"src": "y"}),
        ]
    )
    assert await store.count() == 2
    top = await store.query([1.0, 0.0], k=1)
    assert top[0].id == "a" and top[0].score > 0.9  # cosine similarity, near 1
    filtered = await store.query([1.0, 0.0], k=5, filters={"src": "y"})
    assert [d.id for d in filtered] == ["b"]


# --- the MCP server (end-to-end over stdio) ---------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
_has_mcp = importlib.util.find_spec("mcp") is not None


@pytest.mark.skipif(not _has_mcp, reason="mcp SDK not installed")
async def test_mcp_server_rag_search_over_stdio(tmp_path):
    # seed a memory store (JSON) in-process, then point the server at the same file
    db = str(tmp_path / "kb.json")
    emb = HashingEmbedder(dim=256)  # must match the server's default RAG_EMBED_DIM
    store = InMemoryVectorStore(path=db)
    await NaivePipeline(FileLoader(), FixedWindowChunker(120, 20), emb, store).ingest(
        str(_corpus(tmp_path))
    )

    from openmate.adapters.tools.mcp_client import MCPClient, MCPServerSpec

    client = MCPClient()
    await client.connect(
        MCPServerSpec(
            name="rag",
            command=[sys.executable, "-m", "rag.mcp_server"],
            cwd=str(_REPO_ROOT),
            env={"RAG_STORE": "memory", "RAG_DB": db, "RAG_EMBEDDER": "hashing"},
        )
    )
    try:
        tools = {t.spec.name: t for t in await client.list_tools()}
        assert {"rag_search", "rag_answer", "rag_agentic_answer", "rag_stats"} <= set(tools)
        assert tools["rag_search"].spec.side_effecting is False

        res = await tools["rag_search"].invoke({"query": "agent loop", "k": 2}, ctx=None)
        assert not res.is_error
        payload = json.loads(res.content[0].text)
        assert payload["results"] and payload["results"][0]["source"] == "loop.md"

        stats = json.loads((await tools["rag_stats"].invoke({}, ctx=None)).content[0].text)
        assert stats["chunks"] >= 2
    finally:
        await client.close()
