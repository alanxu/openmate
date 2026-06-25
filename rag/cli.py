"""The RAG CLI — ingest a corpus, then query / answer over it.

    python -m rag.cli ingest docs/                 # load → chunk → embed → index
    python -m rag.cli query "what is the agent loop?" # retrieval only (no LLM)
    python -m rag.cli answer "what is the agent loop?" # naive RAG (one-shot, needs a model)
    python -m rag.cli agentic "compare the loop and the kernel"  # agentic RAG (multi-step)
    python -m rag.cli stats                            # how many chunks are indexed
    python -m rag.cli reset                            # clear the collection

Defaults to the Chroma vector store persisted under ./.rag (override with --db,
or use --store memory for a zero-dependency JSON store). Ingest and query must use
the *same* embedder (--embedder), since their vectors have to live in one space.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from openmate.kernel.errors import OpenMateError

from .factory import (
    DEFAULT_COLLECTION,
    DEFAULT_DB,
    DEFAULT_EMBEDDER,
    DEFAULT_STORE,
    build_embedder,
    build_pipeline,
    build_retriever,
    build_store,
)
from .generate import agentic_answer, answer
from .tools import format_documents


def _components(args):
    embedder = build_embedder(args.embedder, dim=args.dim)
    store = build_store(args.store, db=args.db, collection=args.collection)
    return embedder, store


def _parse_filters(pairs: list[str] | None) -> dict | None:
    if not pairs:
        return None
    out: dict = {}
    for p in pairs:
        if "=" not in p:
            raise OpenMateError(f"--filter must be key=value, got: {p}")
        k, _, v = p.partition("=")
        out[k.strip()] = v.strip()
    return out


async def _ingest(args) -> int:
    embedder, store = _components(args)
    pipeline = build_pipeline(embedder, store, chunk_size=args.chunk_size, overlap=args.overlap)
    report = await pipeline.ingest(args.path)
    total = await store.count()
    print(
        f"ingested {report.documents} document(s) → {report.chunks} chunk(s) "
        f"[{embedder.name}] · collection now holds {total} chunk(s)"
    )
    return 0


async def _query(args) -> int:
    embedder, store = _components(args)
    retriever = build_retriever(embedder, store)
    docs = await retriever.retrieve(args.text, k=args.k, filters=_parse_filters(args.filter))
    print(format_documents(docs))
    return 0 if docs else 1


async def _answer(args) -> int:
    from openmate.config import default_model

    embedder, store = _components(args)
    retriever = build_retriever(embedder, store)
    result = await answer(args.text, retriever, default_model(args.model), k=args.k)
    print(result["answer"])
    if result["sources"]:
        print("\nsources:")
        for i, s in enumerate(result["sources"], 1):
            print(f"  [{i}] {s['source']} (score={s['score']:.3f})")
    return 0


async def _agentic(args) -> int:
    from openmate.config import default_model, default_services

    embedder, store = _components(args)
    retriever = build_retriever(embedder, store)
    services = default_services(verbose=args.verbose) if args.verbose else None
    result = await agentic_answer(
        args.text, retriever, default_model(args.model),
        services=services, k=args.k, max_steps=args.max_steps,
    )
    print(result["answer"])
    print(f"\n[{result['searches']} search(es) · {result['steps']} step(s) · {result['reason']}]")
    return 0


async def _stats(args) -> int:
    _, store = _components(args)
    print(f"collection '{args.collection}' [{args.store}@{args.db}] holds {await store.count()} chunk(s)")
    return 0


async def _reset(args) -> int:
    _, store = _components(args)
    if not args.yes:
        print("refusing to clear without --yes", file=sys.stderr)
        return 2
    await store.delete(None)
    print(f"cleared collection '{args.collection}'")
    return 0


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--store", choices=["chroma", "memory"], default=DEFAULT_STORE,
                        help=f"vector store (default: {DEFAULT_STORE})")
    common.add_argument("--db", default=DEFAULT_DB, help=f"persist path (default: {DEFAULT_DB})")
    common.add_argument("--collection", default=DEFAULT_COLLECTION, help="collection name")
    common.add_argument("--embedder", choices=["hashing", "openai"], default=DEFAULT_EMBEDDER,
                        help=f"embedder (default: {DEFAULT_EMBEDDER}); must match between ingest and query")
    common.add_argument("--dim", type=int, default=None, help="embedding dimension (hashing)")

    p = argparse.ArgumentParser(prog="rag", description="OpenMate RAG: ingest and retrieve.")
    sub = p.add_subparsers(dest="command", required=True)

    ing = sub.add_parser("ingest", parents=[common], help="load → chunk → embed → index a path")
    ing.add_argument("path", help="file or directory to ingest")
    ing.add_argument("--chunk-size", type=int, default=1000)
    ing.add_argument("--overlap", type=int, default=200)
    ing.set_defaults(fn=_ingest)

    q = sub.add_parser("query", parents=[common], help="retrieve passages (no LLM)")
    q.add_argument("text", help="the query")
    q.add_argument("-k", type=int, default=5, help="number of passages")
    q.add_argument("--filter", action="append", help="metadata filter key=value (repeatable)")
    q.set_defaults(fn=_query)

    a = sub.add_parser("answer", parents=[common], help="naive RAG: retrieve + answer once")
    a.add_argument("text", help="the question")
    a.add_argument("-k", type=int, default=5)
    a.add_argument("--model", default=None, help="model name override")
    a.set_defaults(fn=_answer)

    ag = sub.add_parser("agentic", parents=[common], help="agentic RAG: multi-step retrieve + answer")
    ag.add_argument("text", help="the question")
    ag.add_argument("-k", type=int, default=5)
    ag.add_argument("--max-steps", type=int, default=6)
    ag.add_argument("--model", default=None, help="model name override")
    ag.add_argument("--verbose", action="store_true", help="trace the agent's searches")
    ag.set_defaults(fn=_agentic)

    st = sub.add_parser("stats", parents=[common], help="show indexed chunk count")
    st.set_defaults(fn=_stats)

    rs = sub.add_parser("reset", parents=[common], help="clear the collection")
    rs.add_argument("--yes", action="store_true", help="confirm the wipe")
    rs.set_defaults(fn=_reset)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return asyncio.run(args.fn(args))
    except OpenMateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
