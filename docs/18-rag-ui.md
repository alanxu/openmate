# 18 — RAG in the Web UI

> How a user **uses** and **manages** retrieval from the browser. This is the product surface over the RAG backend in [07](07-retrieval-rag.md); it renders retrieval-as-a-tool ([04](04-tools-and-mcp.md)) through the event trajectory ([11](11-observability-and-evaluation.md)) and leans on grounding/upload safety from [10](10-safety-and-guardrails.md). Backend algorithms live in 07 — this doc owns only what the user sees and touches. The workspace shell that hosts this (the left-panel Tasks / Libraries / Projects navigation) is designed in [19](19-workspace.md); library *management* relocates there, while this doc keeps the in-chat RAG experience — grounded answers, citations, retrieval modes.

## Scope & responsibilities

This doc owns the browser-facing RAG experience served by `ui/server.py` + `ui/static/`:

- **Using RAG** — grounded answers with visible citations, retrieval shown in the trajectory, and a control for *whether/how* the knowledge base is consulted (naive vs agentic).
- **Managing RAG** — ingesting sources, browsing/inspecting the knowledge base, removing sources, and (advanced) chunking/embedder/scope config.

It does **not** own retrieval quality (hybrid, rerank, graph), the `Retriever`/`VectorStore` ports, or ingestion internals — those are [07](07-retrieval-rag.md). The UI is a thin edge that maps that spectrum onto endpoints and DOM.

---

## The model today (what already exists)

RAG is largely plumbed; it is just not *legible* in the product. Key facts, grounded in code:

- **Per-thread knowledge base.** `RetrieveTool` (`rag/tools.py`) exposes `rag_search(query, k)`. With `scope_to_thread=True` (default) it merges `{"thread_id": ctx.state.thread_id}` into every query's filters, so one shared vector store isolates each thread's knowledge — no per-thread retriever.
- **Sources are already recorded.** Each `rag_search` call appends its hits to `ctx.state.scratch["rag_sources"]` as `{id, source, score}`. `format_documents` returns numbered blocks `[n] score=… · source=…` in the tool result.
- **Agentic by default.** `build_agent` (`ui/server.py`) wires `tools = [*all_tools(extra_roots), RetrieveTool(self.retriever)]` under `DEFAULT_INSTRUCTIONS` with `max_steps=12`. The model decides when to retrieve — that *is* agentic RAG. There is no naive one-shot path exposed.
- **Knowledge management endpoints exist:** `POST/GET/DELETE /api/threads/{id}/knowledge` — add via file upload or pasted text (ingested with `extra_metadata={"thread_id": id}`), list as `{source, added_at, n_chunks}`, remove by deleting the tracked chunk ids from the vector store. Tracking rows live in `thread_knowledge` (`adapters/stores/sqlite.py`).
- **Retrieval is already in the trajectory** — but only as a generic collapsed `⚒ rag_search` trace card (`ui/static/app.js`), showing raw args + the `[n] score · source` result text.

**Gaps.** No citations in answers; no "was the KB used?" signal; no way to force/skip retrieval; the `[n]`-citation instruction exists in `RagProvider.system_fragment()` but the UI uses `DEFAULT_INSTRUCTIONS`, so it is inactive; management is add/list/remove only (no stats, preview, corpus search, re-index, or config); `folders` are for file-editing scope (`make_file_tools` extra_roots), not RAG ingestion.

---

## Core abstractions (the UI ↔ backend contract)

**Existing knowledge endpoints** (`ui/server.py`):

```
GET    /api/threads/{id}/knowledge        -> [{source, added_at, n_chunks}]
POST   /api/threads/{id}/knowledge        (multipart file | JSON {text, name})
                                          -> {source, n_chunks}
DELETE /api/threads/{id}/knowledge        {source} -> {removed: n_chunks}
```

**Existing chat stream** (`GET /api/chat/stream`, SSE) emits one JSON per kernel event
(`event_to_json`): `RunStarted`, `MessageAdded`, `ModelRequested{n_messages,n_tools}`,
`ModelResponded{ms,finish_reason,usage}`, `ModelStreamed{delta}`,
`ToolCallRequested{call:{id,name,args}}`, `ToolReturned{ms,result}`,
`CheckpointSaved{rev}`, `RunFinished{status,reason,text}`.
> Note: `RunFinished` does **not** currently carry `scratch["rag_sources"]`; Phase 0 adds it.

**Proposed new endpoints** (thin wrappers over `rag/` verbs already in `rag/cli.py`):

```
GET  /api/threads/{id}/knowledge/stats           -> {n_chunks, n_sources}          # wraps `stats`
GET  /api/threads/{id}/knowledge/search?q=&k=     -> [{source, score, snippet}]     # wraps `query` (retrieve-only, no LLM)
GET  /api/threads/{id}/knowledge/{source}/chunks  -> [{id, text}]                   # preview a source
POST /api/threads/{id}/knowledge/{source}/reindex -> {source, n_chunks}             # re-ingest
```

**Two scopes**, both already expressible on the backend:

| Scope | Mechanism | UI |
|---|---|---|
| This chat | `scope_to_thread=True` → `thread_id` filter | default |
| Shared library | `scope_to_thread=False` (or a `"shared"` tag) | toggle |

> Full backend data model, CRUD, and migration for libraries: [07 §Scoping and shared libraries](07-retrieval-rag.md#scoping-and-shared-libraries).

---

## Phase 0 — Make retrieval visible (grounded answers + citations)

The highest-value, lowest-cost slice: retrieval already happens, so this is mostly frontend plus one small backend change.

1. **Surface sources.** Add `sources: state.scratch.get("rag_sources", [])` to the `RunFinished` JSON in `event_to_json`.
2. **Activate citations.** Append the `RagProvider` fragment ("prefer retrieved evidence; cite sources as `[n]`") to the UI agent's instructions so the model emits `[n]` markers.
3. **Grounded bar.** Under an answer whose turn produced `rag_sources`, render `Grounded · N of K passages used`, expandable to `source · score · snippet` rows; `[n]` markers link to the rows.
4. **Restyle the retrieval trace.** Promote the `rag_search` card from a generic `⚒` tool card to a labeled "Knowledge search" card (query + ranked hits).
5. **Retrieval-mode selector** in the composer — makes the naive↔agentic spectrum a user choice via a `?mode=` param on `/api/chat/stream`:
   - **Auto** (default) — today's behavior: agent decides when to call `rag_search` (agentic RAG).
   - **Always search** — force one retrieval before answering (naive one-shot; `rag/generate.py:answer`).
   - **Off** — omit `RetrieveTool` for the turn.
6. **KB stats** in the panel header via `GET …/knowledge/stats`.

**Acceptance:** ask a grounded question with a source attached → the answer cites `[n]`, the grounded bar lists the exact `source · score` used, and the mode selector visibly changes whether/how retrieval runs.

---

## Phase 1 — Inspect & steer

- **Corpus search preview** — a search box in the panel that hits `…/knowledge/search` (retrieve-only, no LLM) so a user can see *what the retriever returns* for a query. Best debugging tool for retrieval quality.
- **Source preview** — expand a source to its chunks via `…/{source}/chunks`.
- **Scope toggle** — "This chat" vs "Shared library" (`scope_to_thread` per query), for reusing a corpus across threads. See [Creating and attaching a library](#creating-and-attaching-a-library).

---

## Phase 2 — Power management

- **Re-index** a source after it changes (`…/{source}/reindex`).
- **Folder-into-RAG ingest** — recursively ingest a directory's documents (distinct from today's "add folder for editing", which only widens file-tool roots).
- **Chunking / embedder / k config** — expose the ingest+retrieve knobs already in `rag/cli.py` (`chunk_size` 1000, `overlap` 200, `embedder` hashing/openai, `k` 5); collapsed "advanced" by default.
- **Ingest progress** — stream progress for large files/folders (ingestion is async).

---

## Phase 3 — Quality & scale (optional, tie to 07/16)

Surface hybrid/rerank choice, de-dup on ingest, cross-thread library management, and a "retrieval eval" affordance that runs RAGAS recall over a source ([16](16-eval-plan.md), [17](17-eval-report.md)).

---

## Creating and attaching a library

Libraries are created **directly** and then *selected* (attached) by a task — there is no per-thread "private" knowledge that you accumulate and later share. Create a library, add sources, attach it to any number of tasks. Backend model + CRUD: [07 §Scoping and shared libraries](07-retrieval-rag.md#scoping-and-shared-libraries); the left-panel navigation is [19](19-workspace.md).

**Flow:**

1. **New library** — the `＋ New` button in the left panel's **Libraries** section names a library and opens its manager.
2. **Add sources** — in the library manager, paste text or upload files; each becomes a source (chunked + embedded into that library).
3. **Attach to a task** — a task's attachment bar has `+ attach`, which lists libraries to select; the task then retrieves across every attached library.

**Library manager** (center pane): the library name (rename / delete) and its sources (add text / upload file / remove). **Embedder** is fixed at first ingest — a library is single-embedder/-dim — so all its sources, and any task that co-queries it, share one embedding space.

**Endpoints:**

| UI action | Endpoint |
|---|---|
| Create library | `POST /api/libraries {name, embedder?}` |
| Ingest into a library | `POST /api/libraries/{lib}/knowledge` (file/text) |
| List sources / remove source | `GET` / `DELETE /api/libraries/{lib}/knowledge` |
| Rename / delete library | `PATCH` / `DELETE /api/libraries/{lib}` |
| List libraries | `GET /api/libraries` |
| Attach / detach on a task | `POST` / `DELETE /api/threads/{tid}/libraries {library_id}` |

**Edge cases:**

- **Embedder lock** — read-only after the first source; attaching an embedder-mismatched library is rejected, not silently merged.
- **Name** — required; duplicates allowed (id is the key).
- **Empty library** — a valid state; the manager shows an empty-state prompt to add the first source.
- **Delete** — removes the library and its chunks; attached tasks lose access (confirm first).

---

## UI layout

Three panes (already present in `ui/static/index.html`); RAG touches two of them:

```
┌────────────┬─────────────────────────────┬──────────────────────┐
│  sidebar   │  messages / trajectory      │  context sidebar     │
│  (threads) │                             │                      │
│            │  user bubble                │  KNOWLEDGE  142 chunks│
│            │  assistant answer  [1][2]   │   scope: [chat|shared]│
│            │  └ Grounded · 2 of 5   ▾    │   search…            │
│            │     [1] refund.pdf 0.82 …   │   • refund.pdf  34 ✎ ⟳ ×│
│            │  ⚒ Knowledge search  ▾      │   • /docs/handbook 61 │
│            │                             │   • pasted note   3   │
│            │  ┌ composer ─────────────┐  │   [ drop / paste /    │
│            │  │ + ask…  [Auto|Always|Off] ▸ │  add folder ]       │
│            │  └───────────────────────┘  │   ⚙ chunk 1000/200 …  │
└────────────┴─────────────────────────────┴──────────────────────┘
```

- **Composer** — `+` add-knowledge menu (exists) + retrieval-mode selector (Phase 0).
- **Under each answer** — grounded bar + citations (Phase 0).
- **Context sidebar** — the knowledge panel: stats, scope, corpus search, per-source rows (preview/re-index/remove), add-zone, advanced config.

---

## Testing & verification

- **Offline (`tests/`, FakeModel):** knowledge endpoints (add/list/remove round-trip, stats count, search shape, `{source}/chunks`); `event_to_json` includes `sources` on `RunFinished`; citation/grounded-bar rendering from a fixed `rag_sources`; mode selector maps to tool-present / forced / absent; library create/promote/attach round-trip, and a second thread retrieves from an attached shared library (embedder-mismatch attach rejected).
- **Live (`evals/`, opt-in):** a seeded source → the answer cites the correct `source`; "Always search" retrieves even when the model wouldn't; corpus search returns relevant chunks. Ties into RAGAS recall from [16](16-eval-plan.md).

---

## Trade-offs & open questions

- **Default scope** — per-thread is the safe default (isolation, no cross-talk); a shared library is convenient but risks leaking one thread's docs into another. Start per-thread; make shared explicit.
- **Citation trust** — the model can cite `[n]` incorrectly or claim grounding it doesn't have. Citations are shown from `rag_sources` (what was *retrieved*), not proof the claim is supported; a real grounding/attribution check belongs in output guardrails ([10](10-safety-and-guardrails.md)).
- **Showing scores** — raw cosine scores are useful for power users but noisy for others; consider hiding behind the expanded view.
- **Naive vs agentic** — "Always search" is cheaper/predictable but weaker on multi-hop; "Auto" is stronger but can skip retrieval entirely. Exposing both as a toggle documents the spectrum instead of hiding it.
- **Re-index cost** — large sources re-embed on every edit; needs progress + possibly content-hash skip.
- **Upload safety** — uploads land under `STATE.uploads_dir/{thread_id}/`; treat filenames/paths as untrusted and keep ingestion confined (same trust boundary caveat as `make_file_tools`/`make_shell_tool` in [04](04-tools-and-mcp.md)).
