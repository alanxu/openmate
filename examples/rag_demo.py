"""RAG over OpenMate's own design docs — naive AND agentic, fully OFFLINE.

Ingests ``docs/`` into a vector store, then shows the three modes from
docs/07:

  1. retrieval only  — real, deterministic, no model (HashingEmbedder + cosine);
  2. naive RAG       — retrieve top-k → answer once;
  3. agentic RAG     — an agent loops rag_search → judge → re-query → answer.

Generation uses a scripted ``FakeModel`` so the demo is deterministic and needs no
API key. To use a real model, swap ``FakeModel(...)`` for ``default_model()`` and
run ``python -m rag.cli ingest docs/`` first to persist to Chroma.

Run:  python examples/rag_demo.py
"""

import asyncio

from openmate.adapters.models.fake import FakeModel, text_response, tool_call_response
from rag import (
    DenseRetriever,
    FileLoader,
    FixedWindowChunker,
    HashingEmbedder,
    InMemoryVectorStore,
    NaivePipeline,
    agentic_answer,
    answer,
    format_documents,
)


async def main() -> None:
    # --- ingest: load → chunk → embed → index (the design docs as our corpus) ---
    embedder = HashingEmbedder(dim=512)
    store = InMemoryVectorStore()
    pipeline = NaivePipeline(FileLoader(), FixedWindowChunker(900, 150), embedder, store)
    report = await pipeline.ingest("docs/")
    print(f"ingested {report.documents} docs → {report.chunks} chunks → {await store.count()} indexed\n")

    retriever = DenseRetriever(embedder, store)
    question = "How does the agent loop decide and act each step?"

    # --- 1. retrieval only (no model — real and deterministic) ---
    print("=== retrieval only ===")
    hits = await retriever.retrieve(question, k=3)
    for d in hits:
        print(f"  {d.score:.3f}  {d.source}")

    # --- 2. naive RAG (retrieve + answer once) ---
    print("\n=== naive RAG ===")
    naive_model = FakeModel(
        [text_response("Each step the loop calls the model to decide, runs any "
                       "requested tools, then checkpoints and checks the stop policy [1].")]
    )
    res = await answer(question, retriever, naive_model, k=3)
    print(" ", res["answer"])
    print("  sources:", [s["source"] for s in res["sources"]])

    # --- 3. agentic RAG (multi-step retrieve → judge → re-query → answer) ---
    print("\n=== agentic RAG ===")
    agentic_model = FakeModel(
        [
            tool_call_response("c1", "rag_search", {"query": "agent loop decide act step", "k": 3}),
            tool_call_response("c2", "rag_search", {"query": "stop policy termination", "k": 3}),
            text_response("The loop is decide→act→checkpoint→check-stop: the model picks "
                          "tool calls or a final answer, tools run, state is checkpointed, "
                          "and a stop policy ends the run [1][2]."),
        ]
    )
    ag = await agentic_answer(question, retriever, agentic_model, k=3, max_steps=6)
    print(" ", ag["answer"])
    print(f"  [{ag['searches']} searches · {ag['steps']} steps · {len(ag['sources'])} sources retrieved]")


if __name__ == "__main__":
    asyncio.run(main())
