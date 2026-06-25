# OpenMate RAG

Retrieval-augmented generation for OpenMate — the implementation of
[`docs/07-retrieval-rag.md`](../docs/07-retrieval-rag.md). It does **both**
naive RAG (one-shot retrieve→answer) and **agentic RAG** (an agent that loops
retrieve→judge→re-query), ships an **ingestion/retrieval CLI**, and an **MCP
server** exposing it all.

## Vector DB

The default store is **[Chroma](https://www.trychroma.com/)** — the most popular
open-source, AI-native vector database: embedded (no server to run), pip-installable,
cosine similarity. A zero-dependency `InMemoryVectorStore` (brute-force cosine,
JSON persistence) is the fallback for tests and offline use. Both implement the
same `VectorStore` port, so `--store memory` ↔ `--store chroma` is one flag.

```bash
pip install "openmate[rag]"     # adds chromadb
```

## Layout

```
openmate/ports/retriever.py   # the ports: Document, Embedder, VectorStore, Retriever, …
rag/
├── loaders.py      # FileLoader (text/markdown/code; skips hidden dirs)
├── chunking.py     # FixedWindowChunker (deterministic ids → idempotent ingest)
├── embedders.py    # HashingEmbedder (default, deterministic) · OpenAICompatibleEmbedder
├── stores.py       # ChromaVectorStore (default) · InMemoryVectorStore
├── retrievers.py   # DenseRetriever (embed query → ANN search)
├── pipeline.py     # NaivePipeline: load → chunk → embed → upsert
├── generate.py     # answer() = naive RAG · agentic_answer() = agentic RAG
├── tools.py        # RetrieveTool (rag_search) · RagProvider
├── factory.py      # config → components (shared by CLI + server)
├── cli.py          # python -m rag.cli  (ingest/query/answer/agentic/stats/reset)
└── mcp_server.py   # python -m rag.mcp_server  (rag_search/rag_answer/rag_agentic_answer)
```

## Embeddings

| Embedder | When |
|---|---|
| `HashingEmbedder` (default) | zero deps, deterministic, no network — feature-hashing over word uni/bi-grams (lexical, not deep-semantic). Great for the PoC, the demo, and reproducible tests. |
| `OpenAICompatibleEmbedder` | real semantic embeddings via any `/v1/embeddings` endpoint (OpenAI, MiniMax, gateways). Set `RAG_EMBEDDER=openai` + `RAG_EMBED_API_KEY`/`OPENAI_API_KEY`. |

> Ingest and query must use the **same** embedder — their vectors share one space.

## CLI

```bash
# 1. ingest a corpus (defaults to Chroma persisted under ./.rag)
python -m rag.cli ingest docs/

# 2. retrieval only — no LLM, fully offline
python -m rag.cli query "how does the agent loop stop?" -k 5

# 3. naive RAG — retrieve + answer once (needs a chat model via .env)
python -m rag.cli answer "how does the agent loop stop?"

# 4. agentic RAG — multi-step retrieve→judge→re-query (add --verbose to watch)
python -m rag.cli agentic "compare the kernel and the agent loop" --verbose

python -m rag.cli stats          # indexed chunk count
python -m rag.cli reset --yes    # clear the collection
```

Useful flags: `--store {chroma,memory}`, `--db PATH`, `--collection NAME`,
`--embedder {hashing,openai}`, `--dim`, `-k`, `--filter key=value` (query),
`--chunk-size`/`--overlap` (ingest), `--max-steps` (agentic).

## MCP server (both RAG and agentic RAG)

Exposes retrieval to any MCP client (an OpenMate agent, Claude, etc.):

| Tool | Mode | Needs a model |
|---|---|---|
| `rag_search(query, k)` | retrieval primitive (an external agent loops it → agentic RAG) | no |
| `rag_answer(question, k)` | **naive RAG** — retrieve + answer once | yes |
| `rag_agentic_answer(question, k, max_steps)` | **agentic RAG** — internal retrieve→judge→re-query loop | yes |
| `rag_stats()` | indexed chunk count | no |

```bash
# configure via env, then run over stdio
RAG_STORE=chroma RAG_DB=./.rag RAG_COLLECTION=openmate python -m rag.mcp_server
```

Mount it in an agent the same way as any MCP server:

```python
from openmate import assemble, MCPProvider
from openmate.adapters.tools.mcp_client import MCPServerSpec

rag = MCPServerSpec(name="rag", command=["python", "-m", "rag.mcp_server"], cwd=".",
                    env={"RAG_DB": "./.rag", "RAG_COLLECTION": "openmate"})
async with assemble(name="kb", system="Answer from the knowledge base.",
                    model=model, services=services,
                    providers=[MCPProvider([rag])]) as agent:
    await agent.run("What does the loop guard do?")
```

`rag_search`/`rag_stats` work with no API key (embedder + store only);
`rag_answer`/`rag_agentic_answer` build a chat model via `default_model()` and
return an `error` field instead of raising if no key is configured.

## In-process API

```python
from rag import build_embedder, build_store, build_retriever, build_pipeline, answer, agentic_answer

emb, store = build_embedder("hashing"), build_store("memory")
await build_pipeline(emb, store).ingest("docs/")
retriever = build_retriever(emb, store)

await answer("what is the agent loop?", retriever, model)          # naive
await agentic_answer("compare the loop and kernel", retriever, model)  # agentic
```

See [`examples/rag_demo.py`](../examples/rag_demo.py) for a runnable, offline
walk-through of all three modes.

## Configuration (env)

| Var | Default | Meaning |
|---|---|---|
| `RAG_STORE` | `chroma` | `chroma` or `memory` |
| `RAG_DB` | `./.rag` | Chroma persist dir, or memory JSON file/dir |
| `RAG_COLLECTION` | `openmate` | collection name |
| `RAG_EMBEDDER` | `hashing` | `hashing` or `openai` |
| `RAG_EMBED_DIM` | `256` | hashing dimension |
| `RAG_EMBED_MODEL` / `RAG_EMBED_BASE_URL` / `RAG_EMBED_API_KEY` | — | for the `openai` embedder |

## What's here vs. the design

Phase 0 (naive RAG end-to-end) is fully implemented; agentic RAG (Phase 2) falls
out of running an OpenMate agent over `RetrieveTool`. Hybrid + reranking (Phase 1),
GraphRAG/RAPTOR (Phase 3), and citation/grounding verification (Phase 4) are
additive layers behind the same ports — not in this slice.
