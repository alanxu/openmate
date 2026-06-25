"""Loaders — read raw documents from disk (docs/07 §Phase 0).

The PoC loader handles a single text/markdown file or a directory tree of them.
Each file becomes one ``RawDoc`` whose id is its path relative to the ingest root,
so ids are stable across runs (ingestion idempotency).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from openmate.ports.retriever import RawDoc

# Extensions treated as plain text. PDFs/HTML/etc. are docs/07 §Phase 3 (a
# richer Loader per type) — out of PoC scope.
TEXT_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".py", ".json", ".yaml", ".yml", ".toml"}


class FileLoader:
    """Load a file or a directory of text files into ``RawDoc``s."""

    def __init__(self, extensions: set[str] | None = None) -> None:
        self.extensions = extensions or TEXT_EXTENSIONS

    def load(self, src: str) -> Iterable[RawDoc]:
        root = Path(src).expanduser()
        if not root.exists():
            raise FileNotFoundError(f"no such path: {src}")
        if root.is_file():
            files = [root]
            base = root.parent
        else:
            files = sorted(p for p in root.rglob("*") if p.is_file())
            base = root
        for path in files:
            if path.suffix.lower() not in self.extensions:
                continue
            # skip hidden files and anything under a dot-dir (.git, .venv, the ./.rag
            # store itself) relative to the ingest root — never index your own DB
            rel_parts = path.relative_to(base).parts if base in path.parents or base == path.parent else ()
            if any(part.startswith(".") for part in rel_parts):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if not text.strip():
                continue
            rel = path.relative_to(base) if base in path.parents or base == path.parent else path
            yield RawDoc(
                id=str(rel),
                text=text,
                metadata={"source": str(rel), "path": str(path), "ext": path.suffix.lower()},
            )
