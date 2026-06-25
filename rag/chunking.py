"""Chunkers — split a document into indexable pieces (docs/07 §Phase 0).

Chunk size/overlap dominates retrieval quality (docs/07 trade-offs); tune
empirically. Chunk **ids are deterministic** (``{doc_id}#{index}``) so re-ingesting
the same source overwrites rather than duplicates — the ingestion-idempotency
guarantee the tests check.
"""

from __future__ import annotations

import re

from openmate.ports.retriever import Chunk, RawDoc

_WS = re.compile(r"\s+")


class FixedWindowChunker:
    """Fixed-size character windows with overlap, snapped to word boundaries.

    The PoC default. ``size`` and ``overlap`` are in characters; windows try not
    to cut a word in half so chunks stay readable.
    """

    def __init__(self, size: int = 1000, overlap: int = 200) -> None:
        if overlap >= size:
            raise ValueError("overlap must be smaller than size")
        self.size = size
        self.overlap = overlap

    def split(self, doc: RawDoc) -> list[Chunk]:
        text = _WS.sub(" ", doc.text).strip()
        if not text:
            return []
        step = self.size - self.overlap
        chunks: list[Chunk] = []
        start = 0
        index = 0
        n = len(text)
        while start < n:
            end = min(start + self.size, n)
            # snap the end back to the last space so we don't split a word (unless
            # this is the final window or no space is found in range)
            if end < n:
                space = text.rfind(" ", start + step, end)
                if space != -1:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        id=f"{doc.id}#{index}",
                        doc_id=doc.id,
                        text=piece,
                        metadata={**doc.metadata, "doc_id": doc.id, "chunk_index": index},
                    )
                )
                index += 1
            if end >= n:
                break
            start = max(end - self.overlap, start + 1)
        return chunks
