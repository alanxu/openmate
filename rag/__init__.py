"""OpenMate RAG — retrieval-augmented generation over a pluggable vector store.

Implements docs/07 against the ``openmate.ports.retriever`` ports. The default
vector DB is **Chroma** (popular, embedded, open-source); an in-memory store is the
zero-dependency fallback. Provides both **naive RAG** (:func:`answer`) and
**agentic RAG** (:func:`agentic_answer`), an ingestion/query **CLI**
(``python -m rag.cli``), and an **MCP server** (``python -m rag.mcp_server``).

    from rag import build_embedder, build_store, build_retriever, build_pipeline, answer

    emb, store = build_embedder("hashing"), build_store("memory")
    await build_pipeline(emb, store).ingest("docs/")
    print((await answer("what is the agent loop?", build_retriever(emb, store), model))["answer"])
"""

from .chunking import FixedWindowChunker
from .embedders import HashingEmbedder, OpenAICompatibleEmbedder
from .factory import (
    build_embedder,
    build_pipeline,
    build_retriever,
    build_store,
)
from .generate import agentic_answer, answer
from .loaders import FileLoader
from .pipeline import NaivePipeline
from .retrievers import DenseRetriever
from .stores import ChromaVectorStore, InMemoryVectorStore
from .tools import RagProvider, RetrieveTool, format_documents

__all__ = [
    "ChromaVectorStore",
    "DenseRetriever",
    "FileLoader",
    "FixedWindowChunker",
    "HashingEmbedder",
    "InMemoryVectorStore",
    "NaivePipeline",
    "OpenAICompatibleEmbedder",
    "RagProvider",
    "RetrieveTool",
    "agentic_answer",
    "answer",
    "build_embedder",
    "build_pipeline",
    "build_retriever",
    "build_store",
    "format_documents",
]
