"""Retrievers — the read side of RAG (docs/07 §Phase 0: dense retrieval).

``DenseRetriever`` is the naive baseline: embed the query, ask the vector store for
nearest neighbours. Hybrid (dense + BM25) and reranking are docs/07 §Phase 1 —
additive layers behind the same :class:`Retriever` port.
"""

from __future__ import annotations

from openmate.ports.retriever import Document, Embedder, Retriever, VectorStore


class DenseRetriever(Retriever):
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    async def retrieve(
        self, query: str, *, k: int = 5, filters: dict | None = None
    ) -> list[Document]:
        vector = (await self.embedder.embed([query]))[0]
        return await self.store.query(vector, k=k, filters=filters)
