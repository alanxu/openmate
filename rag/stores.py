"""Vector stores (docs/07 §Phase 0: "ship one VectorStore adapter").

The chosen vector DB is **Chroma** — the most widely-used open-source, AI-native
vector database, embedded (no server) and pip-installable. :class:`ChromaVectorStore`
is the default for the CLI and MCP server (persistent on disk).

:class:`InMemoryVectorStore` is a zero-dependency fallback (brute-force cosine,
optional JSON persistence) so the pipeline, the offline demo, and the test suite
run with nothing installed. Both honor the same :class:`VectorStore` port, so
switching is one flag.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from openmate.kernel.errors import ConfigError
from openmate.ports.retriever import Document, Vector, VectorRecord


def _cosine(a: Vector, b: Vector) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def _matches(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    return all(metadata.get(k) == v for k, v in filters.items())


class InMemoryVectorStore:
    """Brute-force cosine search over an in-process dict; optional JSON persistence."""

    def __init__(self, path: str | None = None) -> None:
        self.path = Path(path).expanduser() if path else None
        self._records: dict[str, VectorRecord] = {}
        if self.path and self.path.exists():
            self._load()

    def _load(self) -> None:
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self._records = {
            r["id"]: VectorRecord(r["id"], r["vector"], r["text"], r.get("metadata", {}))
            for r in raw
        }

    def _save(self) -> None:
        if not self.path:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(
                [
                    {"id": r.id, "vector": r.vector, "text": r.text, "metadata": r.metadata}
                    for r in self._records.values()
                ]
            ),
            encoding="utf-8",
        )

    async def upsert(self, records: list[VectorRecord]) -> None:
        for r in records:
            self._records[r.id] = r
        self._save()

    async def query(
        self, vector: Vector, *, k: int, filters: dict | None = None
    ) -> list[Document]:
        scored = [
            Document(
                id=r.id,
                text=r.text,
                metadata=r.metadata,
                score=_cosine(vector, r.vector),
            )
            for r in self._records.values()
            if _matches(r.metadata, filters)
        ]
        scored.sort(key=lambda d: d.score, reverse=True)
        return scored[:k]

    async def count(self) -> int:
        return len(self._records)

    async def delete(self, ids: list[str] | None = None) -> None:
        if ids is None:
            self._records.clear()
        else:
            for i in ids:
                self._records.pop(i, None)
        self._save()


class ChromaVectorStore:
    """Adapter over an embedded Chroma collection (cosine space).

    We supply our own embeddings (Chroma's ``embedding_function`` is unused), so
    the embedder stays swappable behind the :class:`Embedder` port. ``path=None``
    uses an ephemeral client (handy for tests); a path gives a persistent client.
    """

    def __init__(self, *, collection: str = "openmate", path: str | None = None) -> None:
        try:
            import chromadb
        except ImportError as e:
            raise ConfigError(
                'Chroma is the default vector store — install it with '
                '`pip install "openmate[rag]"`, or use --store memory.'
            ) from e
        client = (
            chromadb.PersistentClient(path=str(Path(path).expanduser()))
            if path
            else chromadb.EphemeralClient()
        )
        self._collection = client.get_or_create_collection(
            collection, configuration={"hnsw": {"space": "cosine"}}
        )

    async def upsert(self, records: list[VectorRecord]) -> None:
        if not records:
            return
        self._collection.upsert(
            ids=[r.id for r in records],
            embeddings=[r.vector for r in records],
            documents=[r.text for r in records],
            metadatas=[r.metadata or {"_": ""} for r in records],
        )

    async def query(
        self, vector: Vector, *, k: int, filters: dict | None = None
    ) -> list[Document]:
        res = self._collection.query(
            query_embeddings=[vector],
            n_results=k,
            where=filters or None,
            include=["documents", "metadatas", "distances"],
        )
        docs: list[Document] = []
        for id_, text, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            docs.append(
                # cosine distance → similarity
                Document(id=id_, text=text, metadata=meta or {}, score=1.0 - float(dist))
            )
        return docs

    async def count(self) -> int:
        return self._collection.count()

    async def delete(self, ids: list[str] | None = None) -> None:
        if ids is None:
            existing = self._collection.get(include=[])["ids"]
            if existing:
                self._collection.delete(ids=existing)
        else:
            self._collection.delete(ids=ids)
