# 19 — Workspace: Tasks, Libraries, Projects

> The product's information architecture. Three primitives — **Task**, **Library**, **Project** — surfaced from a single left panel. Tasks are chat workspaces ([02](02-agent-loop-and-runtime.md)); Libraries are reusable knowledge ([07](07-retrieval-rag.md) §Scoping and shared libraries); Projects bundle work directories + goals. This supersedes the scattered left-sidebar / right-sidebar layout; [18](18-rag-ui.md) now owns only the *in-chat* RAG experience (grounded answers, citations, retrieval modes).

## Scope & responsibilities

The browser workspace **shell**: the three primitives, their relationships, the left-panel navigation, the data model + endpoints that back them, and how a task's attachments (project + libraries) configure its agent at `build_agent` time. Retrieval internals are [07]; the grounded-answer UX is [18]; file/shell confinement is [04](04-tools-and-mcp.md); goal/context injection is [09](09-context-engineering.md).

---

## The three primitives

- **Task** — an interactive chat workspace (a `thread`): runs the agent loop, transcript persists via checkpoints. Attaches to **≤1 Project** and **0..N Libraries**. This is the only place the agent runs.
- **Library** — a named, reusable knowledge base (sources → chunks), attachable to many tasks. Owns knowledge; see [07].
- **Project** — a bundle of **work directories** (the file/shell roots the agent may read/write) plus **goals** (free text that steers the agent). Groups related tasks and gives them a shared working context.

```
   Project 1───N Task N───N Library
     │ owns          (attached)   │ owns
     ├─ directories  (file/shell roots)   sources → chunks
     └─ goals        (agent steering text)

   Task = chat + agent loop; inherits dirs+goals from its Project,
          retrieves across its attached Libraries.
```

Two primitives are "load-bearing now" (Task, Library — both already have backends); **Project is new** and lands second (see Phasing).

---

## What exists vs. the gap

| Layer | Today | Change |
|---|---|---|
| Task store | `threads(thread_id, title, created_at, updated_at)` + `checkpoints` | add `project_id` column (nullable) |
| Task ↔ Library | `thread_libraries` (M:N) + endpoints | — (reuse) |
| Library store | `libraries` / `library_knowledge` + endpoints | add rename/delete endpoints; a top-level browser UI |
| Project | — | **new** `projects` + `project_directories` tables + endpoints + editor UI |
| Work dirs | `thread_folders` (per-thread) | become **project** directories; thread-local kept as a fallback (see migration) |
| Left panel | Tasks (thread list) only | three sections: Tasks / Libraries / Projects |
| Right sidebar | Knowledge selector + list + Folders | **removed**; management moves left, attachments show in-task |
| `build_agent` | `extra_roots = list_folders(thread_id)`; libs from `list_thread_libraries`; `instructions = DEFAULT` | roots = project dirs ∪ thread folders; `instructions = DEFAULT + project goals`; libs unchanged |

---

## Data model (additions)

```sql
projects(
  project_id TEXT PRIMARY KEY, name TEXT,
  goals      TEXT,                      -- free-text steering, injected into the agent
  created_at REAL NOT NULL)

project_directories(                    -- the file/shell roots a project grants
  project_id TEXT NOT NULL, path TEXT NOT NULL,
  added_at   REAL NOT NULL, PRIMARY KEY (project_id, path))

-- Task ← Project:
ALTER TABLE threads ADD COLUMN project_id TEXT;   -- nullable; NULL = no project

-- Task ↔ Library already exists: thread_libraries (M:N).
```

**Work-directory migration / back-compat.** A task's **effective file/shell roots = `project_directories(task.project)` ∪ `thread_folders(task)`**. Existing `thread_folders` keep working as task-local roots (no migration needed); Projects add a shared, reusable grouping on top. (Alternative considered: migrate each thread's folders into an auto-created project — heavier, deferred; see Trade-offs.)

---

## How a task's attachments configure its agent

`build_agent(thread_id)` ([ui/server.py](../ui/server.py)) resolves attachments at build time — attaching/detaching is just data the next turn re-reads:

```python
project = store.project_of(thread_id)                       # None or a row
roots   = store.project_directories(project) + store.list_folders(thread_id)
goals   = project["goals"] if project else ""
instructions = DEFAULT_INSTRUCTIONS + (f"\n\nProject goals:\n{goals}" if goals else "")
lib_ids = [l["library_id"] for l in store.list_thread_libraries(thread_id)]  # unchanged
tools   = [*all_tools(roots), RetrieveTool(retriever, base_filters={"library_id": {"$in": lib_ids}}, scope_to_thread=False)]
```

So: **Project re-scopes file/shell access + injects goals; Libraries re-scope retrieval.** Goals are steering context ([09]); the file confinement caveat from [04] still applies (roots widen cwd, they aren't a sandbox).

---

## Left-panel navigation (the restructure)

One left panel, a section switcher (**Tasks · Libraries · Projects**); the center pane shows the selected entity. The right context-sidebar is removed.

```
┌───────────────┬───────────────────────────────────────────┐
│ Tasks Libs Proj│  ── task workspace (chat) ──              │
│ [+ New task]  │  Refund policy Q&A                         │
│ ▸ Refund pol… ●│  ┌ attachments ─────────────────────────┐ │
│ ▸ Onboarding  │  │ Project: Website revamp ▾   Libraries:│ │
│ ▸ Q3 report   │  │  Product docs ×  Legal ×   + attach   │ │
│               │  └───────────────────────────────────────┘ │
│               │  user: what's our refund window?          │
│               │  assistant: 30 days … [rag_search]         │
└───────────────┴───────────────────────────────────────────┘
  Libraries mode → list libraries → open one → source manager
  Projects  mode → list projects  → open one → goals + dirs editor
```

- **Tasks** — history list + New task. Opening a task → the chat pane with an **attachment bar** (Project chip + Library chips, editable). Replaces the right sidebar.
- **Libraries** — list all libraries (name · #sources · #chunks); New; open one → the source manager (ingest / list / remove / preview; rename; delete; promote). This is the library manager promoted out of [18].
- **Projects** — list all projects; New; open one → editor: name, **goals** textarea, **work-directory** list (add/remove), and the tasks using it.

---

## Endpoints (exists vs. new)

| Area | Endpoint | Status |
|---|---|---|
| Tasks | `GET /api/threads`, `GET /api/threads/{id}` | ✓ |
| Task → Project | `PATCH /api/threads/{id}` `{project_id}` | new (attach/detach) |
| Task ↔ Libraries | `GET/POST/DELETE /api/threads/{id}/libraries` | ✓ |
| Libraries | `GET/POST /api/libraries`, `…/{id}/knowledge`, `/promote` | ✓ |
| Libraries | `PATCH/DELETE /api/libraries/{id}` (rename/delete) | new |
| Projects | `GET/POST /api/projects`, `GET/PATCH/DELETE /api/projects/{id}` | new |
| Projects | `GET/POST/DELETE /api/projects/{id}/directories` | new |

---

## Phasing

- **P1 — Left-panel IA (no new primitive).** Fold the existing Tasks list + Library browser/manager into a sectioned left panel; remove the right sidebar; show a per-task attachment bar for libraries. Pure re-org over backends that already exist.
- **P2 — Projects.** `projects` + `project_directories` tables + endpoints + the project editor; `threads.project_id`; `build_agent` uses project dirs + goals; keep `thread_folders` as task-local fallback.
- **P3 — Polish.** Cross-links (open a library/project from a task chip), goals surfaced in the trajectory/context, "tasks in this project" list, drag-to-attach.

---

## Testing & verification

- **Offline (`tests/`):** projects CRUD + directories; `threads.project_id` attach/detach; `build_agent` resolves `roots == project_dirs ∪ thread_folders` and injects goals into instructions; library rename/delete; library-browser endpoints.
- **Live (`evals/`):** a task attached to a project can read/write only within project dirs; goals visibly steer behavior; attached libraries ground answers (ties to [16](16-eval-plan.md)).

---

## Trade-offs & open questions

- **Work-dir migration** — union (recommended, zero-migration) vs. migrate each thread's folders into an auto project vs. drop task-local folders entirely.
- **Goals injection** — system-prompt append (simple, shown here) vs. a managed context block / memory ([06](06-memory-and-state.md), [09]); needs a length budget so goals don't crowd the window.
- **Project as boundary** — it's a *convenience* grouping of roots + goals, not a security boundary; file confinement is still cwd + roots, not a sandbox ([04] caveat).
- **Center-pane modality** — one pane that switches between chat / library manager / project editor, vs. keeping chat central and opening Libraries/Projects as full-pane routes. Recommend one switching pane for a single coherent surface.
- **Deleting a shared library / project** in use by other tasks — confirm + cascade rules (library delete already returns chunk ids to purge; project delete should null out `threads.project_id`).
