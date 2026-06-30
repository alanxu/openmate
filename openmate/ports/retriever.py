"""Retrieval ports — the swappable interfaces RAG is built from (docs/07).

Layer 1, alongside Model/Tool/Store/Tracer. The concrete implementations
(pipeline, embedders, vector stores, retrievers, the retrieve tool, the CLI, and
the MCP server) live in the top-level ``rag/`` package; everything there is coded
against these protocols so backends swap without touching callers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

Vector = list[float]


@dataclass
class RawDoc:
    """A source document before chunking."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A slice of a ``RawDoc`` — the unit that gets embedded and indexed."""

    id: str
    doc_id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class VectorRecord:
    """A chunk plus its embedding, ready to upsert into a ``VectorStore``."""

    id: str
    vector: Vector
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Document:
    """A retrieved result: the chunk text, where it came from, and its score."""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)
    score: float = 0.0
    embedding: Vector | None = None

    @property
    def source(self) -> str:
        return str(self.metadata.get("source", self.metadata.get("doc_id", self.id)))


@dataclass
class IngestReport:
    """What an ingestion run did — used by the CLI and idempotency tests."""

    documents: int = 0
    chunks: int = 0
    sources: list[str] = field(default_factory=list)
    chunk_ids: list[str] = field(default_factory=list)  # for later targeted deletion


@runtime_checkable
class Embedder(Protocol):
    """Turns text into vectors. Shares the model boundary spirit of docs/03."""

    name: str
    dim: int

    async def embed(self, texts: list[str]) -> list[Vector]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persists vectors and answers nearest-neighbour queries."""

    async def upsert(self, records: list[VectorRecord]) -> None: ...

    async def query(
        self, vector: Vector, *, k: int, filters: dict | None = None
    ) -> list[Document]: ...

    async def count(self) -> int: ...

    async def delete(self, ids: list[str] | None = None) -> None:
        """Delete by id, or clear the whole collection when ``ids is None``."""
        ...


@runtime_checkable
class Retriever(Protocol):
    """The read side of RAG: a query string in, ranked ``Document``s out."""

    async def retrieve(
        self, query: str, *, k: int, filters: dict | None = None
    ) -> list[Document]: ...


class Loader(Protocol):
    """Reads raw documents from a source (a file, a directory, a URL…)."""

    def load(self, src: str) -> Iterable[RawDoc]: ...


class Chunker(Protocol):
    """Splits a ``RawDoc`` into indexable ``Chunk``s."""

    def split(self, doc: RawDoc) -> list[Chunk]: ...


@runtime_checkable
class Indexer(Protocol):
    """The write side of RAG: source in, an ``IngestReport`` out."""

    async def ingest(self, src: str) -> IngestReport: ...
