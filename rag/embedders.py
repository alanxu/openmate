"""Embedders — turn text into vectors (docs/07; embeddings port from docs/03).

Two implementations:

* :class:`HashingEmbedder` — the zero-dependency, fully **deterministic** default.
  Feature-hashing over word unigrams+bigrams, so it needs no model, no API key,
  and no network. It captures lexical overlap (not deep semantics), which is plenty
  for the PoC, the offline demo, and reproducible tests. It hashes with
  ``hashlib`` (not the builtin ``hash()``, which is salted per process) so the same
  text embeds identically across the ingest and query processes.

* :class:`OpenAICompatibleEmbedder` — the real, semantic option. Hits any
  ``/v1/embeddings`` endpoint (OpenAI, MiniMax, most gateways), so it stays
  provider-agnostic like the chat model port.
"""

from __future__ import annotations

import hashlib
import math
import re

from openmate.ports.retriever import Vector

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    words = _TOKEN.findall(text.lower())
    bigrams = [f"{a}_{b}" for a, b in zip(words, words[1:])]
    return words + bigrams


def _l2_normalize(vec: list[float]) -> Vector:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


class HashingEmbedder:
    """Deterministic feature-hashing embedder — no deps, no network."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hashing-{dim}"
        self.dim = dim

    def _embed_one(self, text: str) -> Vector:
        vec = [0.0] * self.dim
        for tok in _tokens(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "big") % self.dim
            sign = 1.0 if h[4] & 1 else -1.0  # signed hashing reduces collisions
            vec[bucket] += sign
        return _l2_normalize(vec)

    async def embed(self, texts: list[str]) -> list[Vector]:
        return [self._embed_one(t) for t in texts]


class OpenAICompatibleEmbedder:
    """Embeddings via any OpenAI-compatible ``/v1/embeddings`` endpoint."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
        batch_size: int = 64,
    ) -> None:
        self.name = model
        self.dim = dim
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    async def embed(self, texts: list[str]) -> list[Vector]:
        import httpx

        if not self.api_key:
            raise RuntimeError("OpenAICompatibleEmbedder needs an api_key")
        out: list[Vector] = []
        async with httpx.AsyncClient(timeout=60) as client:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                resp = await client.post(
                    f"{self.base_url}/embeddings",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "input": batch},
                )
                resp.raise_for_status()
                data = sorted(resp.json()["data"], key=lambda d: d["index"])
                out.extend(d["embedding"] for d in data)
        if out:
            self.dim = len(out[0])
        return out
