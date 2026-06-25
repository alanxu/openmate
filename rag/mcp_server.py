"""An MCP server exposing RAG — both naive and agentic (docs/07 §Phase 0 & 2).

Run it over stdio (this is what an OpenMate ``MCPServerSpec`` launches):

    python -m rag.mcp_server                 # from the repo root
    # configure via env: RAG_STORE, RAG_DB, RAG_COLLECTION, RAG_EMBEDDER, RAG_EMBED_DIM

Tools:

    rag_search(query, k)            read   retrieval only — no LLM, works offline
    rag_answer(question, k)         read   NAIVE RAG: retrieve top-k → answer once
    rag_agentic_answer(question)    read   AGENTIC RAG: agent loops retrieve→judge→re-query
    rag_stats()                     read   indexed chunk count

``rag_search``/``rag_stats`` need only the embedder + vector store (no API key).
``rag_answer``/``rag_agentic_answer`` also need a chat model (``default_model()``);
if no key is configured they return an ``error`` instead of raising.
"""

from __future__ import annotations

import os
import sys

# Allow `python rag/mcp_server.py` as well as `python -m rag.mcp_server`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from rag.factory import build_embedder, build_retriever, build_store
from rag.generate import agentic_answer, answer

app = FastMCP("openmate-rag")

# Built once from env config; shared by every tool call.
_embedder = build_embedder()
_store = build_store()
_retriever = build_retriever(_embedder, _store)
_model = None  # lazily built — only the answer tools need a chat model


def _get_model():
    global _model
    if _model is None:
        from openmate.config import default_model

        _model = default_model()
    return _model


_READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)


@app.tool(
    name="rag_search",
    description=(
        "Search the knowledge base and return the top-k passages with sources and "
        "scores. Read-only; the retrieval primitive an agent calls (repeatedly, with "
        "refined queries) to do agentic RAG itself."
    ),
    annotations=_READ,
)
async def rag_search(query: str, k: int = 5) -> dict:
    docs = await _retriever.retrieve(query, k=k)
    return {
        "results": [
            {"id": d.id, "source": d.source, "score": round(d.score, 4), "text": d.text}
            for d in docs
        ]
    }


@app.tool(
    name="rag_answer",
    description=(
        "NAIVE RAG: retrieve the top-k passages and answer the question from them in "
        "one shot, with citations. Best for direct, single-hop questions."
    ),
    annotations=_READ,
)
async def rag_answer(question: str, k: int = 5) -> dict:
    try:
        return await answer(question, _retriever, _get_model(), k=k)
    except Exception as e:  # noqa: BLE001 — surface config/model errors as data
        return {"error": f"{type(e).__name__}: {e}"}


@app.tool(
    name="rag_agentic_answer",
    description=(
        "AGENTIC RAG: an agent searches the knowledge base over multiple steps — "
        "judging results and re-querying with refined or decomposed queries — before "
        "answering with citations. Best for multi-hop or comparison questions."
    ),
    annotations=_READ,
)
async def rag_agentic_answer(question: str, k: int = 5, max_steps: int = 6) -> dict:
    try:
        return await agentic_answer(
            question, _retriever, _get_model(), k=k, max_steps=max_steps
        )
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


@app.tool(
    name="rag_stats",
    description="Report how many chunks are indexed in the knowledge base.",
    annotations=_READ,
)
async def rag_stats() -> dict:
    return {"chunks": await _store.count(), "embedder": _embedder.name}


if __name__ == "__main__":
    app.run()
