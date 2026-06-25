"""Config → components. Shared wiring for the CLI and the MCP server.

Every knob has an env-var default so the CLI, the MCP server (launched as a
subprocess), and tests all build the *same* embedder + store + retriever from the
same configuration.
"""

from __future__ import annotations

import os

from openmate.kernel.errors import ConfigError
from openmate.ports.retriever import Embedder, Retriever, VectorStore

from .chunking import FixedWindowChunker
from .embedders import HashingEmbedder, OpenAICompatibleEmbedder
from .loaders import FileLoader
from .pipeline import NaivePipeline
from .retrievers import DenseRetriever
from .stores import ChromaVectorStore, InMemoryVectorStore

DEFAULT_STORE = os.environ.get("RAG_STORE", "chroma")
DEFAULT_DB = os.environ.get("RAG_DB", "./.rag")
DEFAULT_COLLECTION = os.environ.get("RAG_COLLECTION", "openmate")
DEFAULT_EMBEDDER = os.environ.get("RAG_EMBEDDER", "hashing")
DEFAULT_DIM = int(os.environ.get("RAG_EMBED_DIM", "256"))


def build_embedder(kind: str | None = None, *, dim: int | None = None) -> Embedder:
    kind = (kind or DEFAULT_EMBEDDER).lower()
    if kind == "hashing":
        return HashingEmbedder(dim=dim or DEFAULT_DIM)
    if kind in ("openai", "openai-compatible"):
        return OpenAICompatibleEmbedder(
            model=os.environ.get("RAG_EMBED_MODEL", "text-embedding-3-small"),
            api_key=os.environ.get("RAG_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("RAG_EMBED_BASE_URL", "https://api.openai.com/v1"),
            dim=dim or int(os.environ.get("RAG_EMBED_DIM", "1536")),
        )
    raise ConfigError(f"unknown embedder '{kind}' (use: hashing, openai)")


def build_store(
    kind: str | None = None, *, db: str | None = None, collection: str | None = None
) -> VectorStore:
    kind = (kind or DEFAULT_STORE).lower()
    if kind == "chroma":
        return ChromaVectorStore(
            collection=collection or DEFAULT_COLLECTION, path=db or DEFAULT_DB
        )
    if kind == "memory":
        # in-memory store persists to a JSON file so ingest/query across processes works.
        # Accept either a .json file path directly, or a directory (→ <dir>/memory.json).
        base = db or DEFAULT_DB
        path = base if base.endswith(".json") else os.path.join(base, "memory.json")
        return InMemoryVectorStore(path=path)
    raise ConfigError(f"unknown store '{kind}' (use: chroma, memory)")


def build_retriever(embedder: Embedder, store: VectorStore) -> Retriever:
    return DenseRetriever(embedder, store)


def build_pipeline(
    embedder: Embedder,
    store: VectorStore,
    *,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> NaivePipeline:
    return NaivePipeline(
        loader=FileLoader(),
        chunker=FixedWindowChunker(size=chunk_size, overlap=overlap),
        embedder=embedder,
        store=store,
    )
