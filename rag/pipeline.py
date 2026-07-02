"""The ingestion pipeline — load → chunk → embed → upsert (docs/07 §Phase 0).

``NaivePipeline.ingest`` is idempotent: chunk ids are derived from the source path
and chunk index, so re-ingesting the same corpus overwrites in place instead of
duplicating (verified by the test suite).
"""

from __future__ import annotations

from openmate.ports.retriever import (
    Chunker,
    Embedder,
    IngestReport,
    Loader,
    VectorRecord,
    VectorStore,
)


class NaivePipeline:
    def __init__(
        self,
        loader: Loader,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        *,
        batch_size: int = 128,
    ) -> None:
        self.loader = loader
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.batch_size = batch_size

    async def ingest(
        self, src: str, *, extra_metadata: dict | None = None, id_prefix: str = ""
    ) -> IngestReport:
        """Load → chunk → embed → upsert.

        ``extra_metadata`` is merged into every loaded document's metadata
        *before* chunking, so it propagates into every chunk and (via the chunker's
        ``{**doc.metadata, ...}``) every upserted record. This is how callers scope
        a corpus — e.g. ``{"library_id": lib}`` so a later ``retrieve(..., filters=...)``
        only sees that library's knowledge — without any change to the Loader/Chunker/
        VectorStore ports themselves.

        ``id_prefix`` is prepended to every chunk id (and to ``report.chunk_ids``).
        Chunk ids are otherwise path-derived (``{doc_id}#{i}``), so the same source
        ingested into two scopes would collide; prefixing with e.g. ``f"{library_id}:"``
        keeps each scope's copies distinct while preserving per-scope idempotency.
        """
        report = IngestReport()
        batch: list = []  # list[Chunk]
        chunk_ids: list[str] = []

        async def flush() -> None:
            if not batch:
                return
            vectors = await self.embedder.embed([c.text for c in batch])
            await self.store.upsert(
                [
                    VectorRecord(id=id_prefix + c.id, vector=v, text=c.text, metadata=c.metadata)
                    for c, v in zip(batch, vectors)
                ]
            )
            batch.clear()

        for doc in self.loader.load(src):
            if extra_metadata:
                doc.metadata = {**doc.metadata, **extra_metadata}
            report.documents += 1
            report.sources.append(doc.id)
            for chunk in self.chunker.split(doc):
                report.chunks += 1
                chunk_ids.append(id_prefix + chunk.id)
                batch.append(chunk)
                if len(batch) >= self.batch_size:
                    await flush()
        await flush()
        report.chunk_ids = chunk_ids
        return report
