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

    async def ingest(self, src: str) -> IngestReport:
        report = IngestReport()
        batch: list = []  # list[Chunk]

        async def flush() -> None:
            if not batch:
                return
            vectors = await self.embedder.embed([c.text for c in batch])
            await self.store.upsert(
                [
                    VectorRecord(id=c.id, vector=v, text=c.text, metadata=c.metadata)
                    for c, v in zip(batch, vectors)
                ]
            )
            batch.clear()

        for doc in self.loader.load(src):
            report.documents += 1
            report.sources.append(doc.id)
            for chunk in self.chunker.split(doc):
                report.chunks += 1
                batch.append(chunk)
                if len(batch) >= self.batch_size:
                    await flush()
        await flush()
        return report
