"""The OpenMate UI server — a thin Starlette app over the kernel's event stream.

Mirrors the architecture's own claim (see ``openmate/kernel/events.py``): the
tracer, persistence, and UI are all just consumers of the ``Event`` stream
``agent.stream()`` yields. This module adds no new agent behavior — it
translates that stream to Server-Sent Events for a browser, and reads thread
history back out of ``SQLiteStore`` for the sidebar.

Lives at the project root (sibling to ``openmate/``), not inside the package,
because it is an edge consumer of the library, not part of it — same reason
``examples/`` and ``servers/`` sit alongside ``openmate/`` rather than inside it.
Started via ``openmate ui`` (see ``openmate/cli.py``), never imported directly
by the kernel/adapters.
"""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path

from sse_starlette.sse import EventSourceResponse
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent  # project root: parent of ui/ and openmate/
sys.path.insert(0, str(ROOT))

from openmate.adapters.stores.sqlite import SQLiteStore  # noqa: E402
from openmate.adapters.tools.builtin import all_tools  # noqa: E402
from openmate.adapters.tracers.jsonl import (  # noqa: E402
    DEFAULT_LOG_DIR,
    default_log_path,
    force_attach,
    list_log_files,
)
from openmate.config import default_model, default_services  # noqa: E402
from openmate.kernel.agent import Agent  # noqa: E402
from openmate.kernel.events import (  # noqa: E402
    CheckpointSaved,
    MessageAdded,
    ModelRequested,
    ModelResponded,
    ModelStreamed,
    RunFinished,
    RunStarted,
    ToolCallRequested,
    ToolReturned,
)
from openmate.kernel.types import (  # noqa: E402
    Message,
    Part,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
)
from rag.factory import build_embedder, build_pipeline, build_retriever, build_store  # noqa: E402
from rag.tools import RetrieveTool  # noqa: E402

STATIC_DIR = Path(__file__).resolve().parent / "static"

DEFAULT_INSTRUCTIONS = (
    "You are OpenMate, a concise and helpful AI assistant. "
    "You can call tools to do arithmetic, read/write/list files, run shell commands, "
    "fetch URLs, and tell the time. File and shell access is confined to the server's "
    "working directory plus any folders the user has explicitly added for this thread — "
    "if an action needs a path outside those, say so instead of guessing. "
    "If the thread has knowledge attached, use `rag_search` to ground answers in it and "
    "prefer retrieved evidence over prior knowledge. "
    "Think step by step and use tools when they help. "
    "When you have the answer, reply directly without calling a tool."
)


# --- app state, set once at startup by run() ------------------------------------
class AppState:
    def __init__(self, db_path: str, allow_write: bool, model_name: str | None) -> None:
        self.db_path = db_path
        self.allow_write = allow_write
        self.model_name = model_name

        # One shared embedder/vector-store/retriever for the whole server. All
        # knowledge lives in the same collection, isolated by the `library_id`
        # metadata tag — a thread's private KB is the library whose id == thread_id,
        # plus any shared libraries it attaches. See rag/tools.py:RetrieveTool and
        # rag/pipeline.py:NaivePipeline.ingest's `extra_metadata`/`id_prefix`.
        self.embedder = build_embedder()
        self.vector_store = build_store()
        self.retriever = build_retriever(self.embedder, self.vector_store)
        self.pipeline = build_pipeline(self.embedder, self.vector_store)

        # Where pasted-text "knowledge" gets materialized as files before
        # ingestion, since the existing pipeline only loads from disk.
        self.uploads_dir = Path(db_path).resolve().parent / ".openmate_uploads"
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

    def new_store(self) -> SQLiteStore:
        return SQLiteStore(self.db_path)

    def build_agent(self, store: SQLiteStore, thread_id: str) -> Agent:
        services = default_services(store=store)
        # The UI's Logs tab needs data, so always attach a per-thread JSONL
        # logger to the bus (unless the user set OPENMATE_LOG=0 to opt out).
        # The logger writes one file per thread to ~/.openmate/logs/.
        force_attach(services.bus)
        # A task's file/shell roots = its attached project's directories plus any
        # task-local folders; the project's goals (if any) steer the agent. See docs/19.
        project = store.get_thread_project(thread_id)
        proj_dirs = project["directories"] if project else []
        extra_roots = list(dict.fromkeys([*proj_dirs, *store.list_folders(thread_id)]))
        # A task retrieves across the libraries it has selected (attached). There is
        # no implicit per-thread library — knowledge lives only in libraries you
        # create and then attach. Empty selection → the sentinel matches nothing.
        lib_ids = [lib["library_id"] for lib in store.list_thread_libraries(thread_id)]
        retrieve = RetrieveTool(
            self.retriever,
            base_filters={"library_id": {"$in": lib_ids or ["__none__"]}},
            scope_to_thread=False,
        )
        instructions = DEFAULT_INSTRUCTIONS
        if project and project["goals"].strip():
            instructions += "\n\nProject goals for this task (keep them in mind):\n" + project["goals"].strip()
        # The chat UI always gets write + shell by default (this is a local
        # single-user assistant, not a multi-tenant service), scoped to the server's
        # cwd plus the task's project directories and any task-local folders — see
        # openmate/adapters/tools/builtin.py:make_shell_tool's docstring for the
        # (cwd-only) confinement caveat.
        tools = [*all_tools(extra_roots), retrieve]
        return Agent(
            name="openmate",
            model=default_model(self.model_name),
            instructions=instructions,
            services=services,
            tools=tools,
            max_steps=12,
        )


STATE: AppState | None = None


# --- JSON serialization: dataclasses -> plain dicts for the browser ------------
def part_to_json(p: Part) -> dict:
    if isinstance(p, TextPart):
        return {"kind": "text", "text": p.text}
    if isinstance(p, ToolCallPart):
        return {"kind": "tool_call", "id": p.id, "name": p.name, "args": p.args}
    if isinstance(p, ToolResultPart):
        return {
            "kind": "tool_result",
            "call_id": p.call_id,
            "content": [part_to_json(c) for c in p.content],
            "is_error": p.is_error,
        }
    if isinstance(p, ThinkingPart):
        return {"kind": "thinking", "text": p.text}
    return {"kind": "unknown"}


def message_to_json(m: Message) -> dict:
    return {
        "role": m.role,
        "name": m.name,
        "content": [part_to_json(p) for p in m.content],
        "text": m.text,
        "tool_calls": [part_to_json(c) for c in m.tool_calls],
    }


def event_to_json(ev) -> dict:
    """Map every ``Event`` subtype to a JSON-able dict the frontend renders.

    One case per type in ``openmate/kernel/events.py`` — keep this in sync with
    that file; it is the UI's only contract with the kernel's event taxonomy.
    """
    base = {"type": type(ev).__name__, "thread_id": ev.thread_id, "step": ev.step, "ts": ev.ts}

    if isinstance(ev, RunStarted):
        return base

    if isinstance(ev, MessageAdded):
        base["message"] = message_to_json(ev.message)
        return base

    if isinstance(ev, ModelRequested):
        base["n_messages"] = ev.n_messages
        base["n_tools"] = ev.n_tools
        return base

    if isinstance(ev, ModelResponded):
        base["ms"] = ev.ms
        base["finish_reason"] = ev.response.finish_reason
        u = ev.response.usage
        base["usage"] = {
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "total_tokens": u.total_tokens,
        }
        return base

    if isinstance(ev, ModelStreamed):
        d = ev.delta
        base["delta"] = {
            "kind": d.kind,
            "text": d.data if d.kind in ("text", "thinking") else None,
        }
        return base

    if isinstance(ev, ToolCallRequested):
        c = ev.call
        base["call"] = {"id": c.id, "name": c.name, "args": c.args}
        return base

    if isinstance(ev, ToolReturned):
        base["ms"] = ev.ms
        base["result"] = part_to_json(ev.result)
        return base

    if isinstance(ev, CheckpointSaved):
        base["rev"] = ev.rev
        return base

    if isinstance(ev, RunFinished):
        r = ev.result
        base["status"] = r.status
        base["reason"] = r.reason
        base["text"] = r.text
        base["steps"] = r.steps
        base["usage"] = {
            "prompt_tokens": r.usage.prompt_tokens,
            "completion_tokens": r.usage.completion_tokens,
            "total_tokens": r.usage.total_tokens,
        }
        return base

    return base


# --- routes ----------------------------------------------------------------------
async def index(request: Request) -> FileResponse:
    response = FileResponse(STATIC_DIR / "index.html")
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


async def debug_tools(request: Request) -> JSONResponse:
    """GET /api/debug/tools — what tools the *currently running* server process
    actually hands the agent. Exists because code edits to this file don't take
    effect until the ``openmate ui`` process is restarted — hit this endpoint
    after a restart to confirm the change actually loaded, instead of guessing."""
    thread_id = request.query_params.get("thread_id", "__debug__")
    store = STATE.new_store()
    try:
        agent = STATE.build_agent(store, thread_id)
        extra_roots = store.list_folders(thread_id)
    finally:
        store.close()
    return JSONResponse(
        {
            "tools": [t.spec.name for t in agent.tools],
            "extra_roots_for_thread": extra_roots,
        }
    )


async def list_threads(request: Request) -> JSONResponse:
    store = STATE.new_store()
    try:
        return JSONResponse(store.list_threads())
    finally:
        store.close()


async def get_thread(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        state = await store.load(thread_id)
    finally:
        store.close()
    if state is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(
        {
            "thread_id": state.thread_id,
            "status": state.status,
            "messages": [message_to_json(m) for m in state.messages if m.role != "system"],
        }
    )


async def chat_stream(request: Request) -> EventSourceResponse:
    thread_id = request.query_params.get("thread_id")
    message = request.query_params.get("message", "")
    if not thread_id or not message:
        return JSONResponse({"error": "thread_id and message are required"}, status_code=400)

    store = STATE.new_store()
    agent = STATE.build_agent(store, thread_id)

    async def gen():
        try:
            async for ev in agent.stream(message, thread_id=thread_id):
                yield {"event": type(ev).__name__, "data": json.dumps(event_to_json(ev))}
        except Exception as e:  # noqa: BLE001 — surface to the client, don't crash the worker
            yield {"event": "Error", "data": json.dumps({"error": str(e)})}
        finally:
            store.close()

    return EventSourceResponse(gen())


# --- knowledge & libraries: ingest into the shared RAG store, tagged library_id -
async def _materialize(request: Request, dest_dir: Path) -> "tuple[str, Path]":
    """Write an uploaded file or pasted text into ``dest_dir``; return (label, path).
    Raises ValueError on bad input (handlers turn it into a 400)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if "multipart/form-data" in request.headers.get("content-type", ""):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            raise ValueError("missing 'file' in form data")
        label = upload.filename or f"upload-{uuid.uuid4().hex[:8]}.txt"
        path = dest_dir / label
        path.write_bytes(await upload.read())
        return label, path
    body = await request.json()
    text = (body or {}).get("text")
    if not text or not text.strip():
        raise ValueError("missing 'text' (or upload a 'file')")
    label = (body or {}).get("name") or f"pasted-{uuid.uuid4().hex[:8]}.txt"
    if not label.endswith((".txt", ".md")):
        label += ".txt"
    path = dest_dir / label
    path.write_text(text, encoding="utf-8")
    return label, path


async def _ingest_into_library(store: SQLiteStore, library_id: str, label: str, path: Path) -> int:
    """Ingest one source into a library: tag chunks with library_id, namespace their
    ids by library (so the same source can live in several libraries), and record the
    id-set for precise removal. Returns the chunk count."""
    report = await STATE.pipeline.ingest(
        str(path),
        extra_metadata={"library_id": library_id},
        id_prefix=f"{library_id}:",
    )
    store.add_library_source(library_id, label, report.chunk_ids)
    return len(report.chunk_ids)


def _sources_json(items: list) -> list:
    return [{"source": i["source"], "added_at": i["added_at"], "n_chunks": len(i["chunk_ids"])} for i in items]


async def list_knowledge(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        items = store.list_knowledge(thread_id)
    finally:
        store.close()
    return JSONResponse(_sources_json(items))


async def add_knowledge(request: Request) -> JSONResponse:
    """POST /api/threads/{id}/knowledge — add to this thread's private library."""
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        store.ensure_private_library(thread_id, embedder=STATE.embedder.name, dim=STATE.embedder.dim)
        try:
            label, path = await _materialize(request, STATE.uploads_dir / thread_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        n = await _ingest_into_library(store, thread_id, label, path)
    finally:
        store.close()
    return JSONResponse({"source": label, "n_chunks": n})


async def remove_knowledge(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    source = request.query_params.get("source")
    if not source:
        return JSONResponse({"error": "source query param is required"}, status_code=400)
    store = STATE.new_store()
    try:
        chunk_ids = store.remove_knowledge(thread_id, source)
    finally:
        store.close()
    if chunk_ids:
        await STATE.vector_store.delete(ids=chunk_ids)
    return JSONResponse({"removed": source, "n_chunks": len(chunk_ids)})


# --- libraries: create / list, ingest into, attach to threads ------------------
async def list_libraries(request: Request) -> JSONResponse:
    store = STATE.new_store()
    try:
        return JSONResponse(store.list_libraries())
    finally:
        store.close()


async def create_library(request: Request) -> JSONResponse:
    body = await request.json()
    name = ((body or {}).get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "missing 'name'"}, status_code=400)
    library_id = "lib_" + uuid.uuid4().hex[:12]
    store = STATE.new_store()
    try:
        lib = store.create_library(
            library_id, name, kind="shared",
            embedder=STATE.embedder.name, dim=STATE.embedder.dim,
        )
    finally:
        store.close()
    return JSONResponse(lib)


async def list_library_knowledge(request: Request) -> JSONResponse:
    library_id = request.path_params["library_id"]
    store = STATE.new_store()
    try:
        items = store.list_library_sources(library_id)
    finally:
        store.close()
    return JSONResponse(_sources_json(items))


async def add_library_knowledge(request: Request) -> JSONResponse:
    """Add knowledge to a library from a server-side folder/file path (JSON {path} —
    a directory ingests every text file under it), pasted text (JSON {text}), or an
    uploaded file (multipart)."""
    library_id = request.path_params["library_id"]
    store = STATE.new_store()
    try:
        if store.get_library(library_id) is None:
            return JSONResponse({"error": f"no such library: {library_id}"}, status_code=404)
        if "multipart/form-data" not in request.headers.get("content-type", ""):
            body = await request.json()
            raw_path = (body or {}).get("path")
            if raw_path:
                p = Path(raw_path).expanduser().resolve()
                if not p.exists():
                    return JSONResponse({"error": f"no such path: {p}"}, status_code=400)
                label = (body or {}).get("name") or p.name
                n = await _ingest_into_library(store, library_id, label, p)
                return JSONResponse({"source": label, "n_chunks": n})
            text = (body or {}).get("text")
            if not text or not text.strip():
                return JSONResponse({"error": "provide 'path', 'text', or upload a file"}, status_code=400)
            dest = STATE.uploads_dir / library_id
            dest.mkdir(parents=True, exist_ok=True)
            label = (body or {}).get("name") or f"pasted-{uuid.uuid4().hex[:8]}.txt"
            if not label.endswith((".txt", ".md")):
                label += ".txt"
            fp = dest / label
            fp.write_text(text, encoding="utf-8")
            n = await _ingest_into_library(store, library_id, label, fp)
            return JSONResponse({"source": label, "n_chunks": n})
        try:
            label, path = await _materialize(request, STATE.uploads_dir / library_id)
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)
        n = await _ingest_into_library(store, library_id, label, path)
        return JSONResponse({"source": label, "n_chunks": n})
    finally:
        store.close()


async def remove_library_knowledge(request: Request) -> JSONResponse:
    library_id = request.path_params["library_id"]
    source = request.query_params.get("source")
    if not source:
        return JSONResponse({"error": "source query param is required"}, status_code=400)
    store = STATE.new_store()
    try:
        chunk_ids = store.remove_library_source(library_id, source)
    finally:
        store.close()
    if chunk_ids:
        await STATE.vector_store.delete(ids=chunk_ids)
    return JSONResponse({"removed": source, "n_chunks": len(chunk_ids)})


async def list_thread_libraries(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        libs = store.list_thread_libraries(thread_id)
    finally:
        store.close()
    return JSONResponse(libs)


async def attach_thread_library(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    body = await request.json()
    library_id = (body or {}).get("library_id")
    if not library_id:
        return JSONResponse({"error": "missing 'library_id'"}, status_code=400)
    store = STATE.new_store()
    try:
        lib = store.get_library(library_id)
        if lib is None:
            return JSONResponse({"error": f"no such library: {library_id}"}, status_code=404)
        if lib["embedder"] and lib["embedder"] != STATE.embedder.name:
            return JSONResponse(
                {"error": f"embedder mismatch: library uses {lib['embedder']}, "
                          f"server uses {STATE.embedder.name}"},
                status_code=409,
            )
        store.attach_library(thread_id, library_id)
    finally:
        store.close()
    return JSONResponse({"attached": library_id})


async def detach_thread_library(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    library_id = request.query_params.get("library_id")
    if not library_id:
        return JSONResponse({"error": "library_id query param is required"}, status_code=400)
    if library_id == thread_id:
        return JSONResponse({"error": "cannot detach the thread's own (private) library"}, status_code=400)
    store = STATE.new_store()
    try:
        store.detach_library(thread_id, library_id)
    finally:
        store.close()
    return JSONResponse({"detached": library_id})


# --- projects: work directories + goals, attached to a task --------------------
async def list_projects(request: Request) -> JSONResponse:
    store = STATE.new_store()
    try:
        return JSONResponse(store.list_projects())
    finally:
        store.close()


async def create_project(request: Request) -> JSONResponse:
    body = await request.json()
    name = ((body or {}).get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "missing 'name'"}, status_code=400)
    project_id = "proj_" + uuid.uuid4().hex[:12]
    store = STATE.new_store()
    try:
        proj = store.create_project(project_id, name, goals=(body or {}).get("goals", ""))
    finally:
        store.close()
    return JSONResponse(proj)


async def get_project(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    store = STATE.new_store()
    try:
        proj = store.get_project(project_id)
        if proj is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        proj["tasks"] = store.list_project_threads(project_id)
    finally:
        store.close()
    return JSONResponse(proj)


async def update_project(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    body = await request.json()
    store = STATE.new_store()
    try:
        if store.get_project(project_id) is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        proj = store.update_project(project_id, name=(body or {}).get("name"), goals=(body or {}).get("goals"))
    finally:
        store.close()
    return JSONResponse(proj)


async def delete_project(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    store = STATE.new_store()
    try:
        store.delete_project(project_id)
    finally:
        store.close()
    return JSONResponse({"deleted": project_id})


async def list_project_directories(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    store = STATE.new_store()
    try:
        return JSONResponse(store.list_project_directories(project_id))
    finally:
        store.close()


async def add_project_directory(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    body = await request.json()
    raw = (body or {}).get("path")
    if not raw:
        return JSONResponse({"error": "missing 'path'"}, status_code=400)
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_dir():
        return JSONResponse({"error": f"not a directory: {resolved}"}, status_code=400)
    store = STATE.new_store()
    try:
        store.add_project_directory(project_id, str(resolved))
    finally:
        store.close()
    return JSONResponse({"path": str(resolved)})


async def remove_project_directory(request: Request) -> JSONResponse:
    project_id = request.path_params["project_id"]
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path query param is required"}, status_code=400)
    store = STATE.new_store()
    try:
        store.remove_project_directory(project_id, path)
    finally:
        store.close()
    return JSONResponse({"removed": path})


# --- task ↔ project attachment + library rename/delete -------------------------
async def update_thread(request: Request) -> JSONResponse:
    """PATCH /api/threads/{id} — attach the task to a project (project_id=null detaches)."""
    thread_id = request.path_params["thread_id"]
    body = await request.json()
    if not body or "project_id" not in body:
        return JSONResponse({"error": "missing 'project_id' (use null to detach)"}, status_code=400)
    project_id = body.get("project_id")
    store = STATE.new_store()
    try:
        if project_id and store.get_project(project_id) is None:
            return JSONResponse({"error": f"no such project: {project_id}"}, status_code=404)
        store.set_thread_project(thread_id, project_id)
    finally:
        store.close()
    return JSONResponse({"thread_id": thread_id, "project_id": project_id})


async def get_thread_project(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        return JSONResponse(store.get_thread_project(thread_id))
    finally:
        store.close()


async def update_library(request: Request) -> JSONResponse:
    library_id = request.path_params["library_id"]
    body = await request.json()
    name = ((body or {}).get("name") or "").strip()
    if not name:
        return JSONResponse({"error": "missing 'name'"}, status_code=400)
    store = STATE.new_store()
    try:
        if store.get_library(library_id) is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        store.rename_library(library_id, name)
        lib = store.get_library(library_id)
    finally:
        store.close()
    return JSONResponse(lib)


async def delete_library(request: Request) -> JSONResponse:
    library_id = request.path_params["library_id"]
    store = STATE.new_store()
    try:
        chunk_ids = store.delete_library(library_id)
    finally:
        store.close()
    if chunk_ids:
        await STATE.vector_store.delete(ids=chunk_ids)
    return JSONResponse({"deleted": library_id, "n_chunks": len(chunk_ids)})


# --- logs: per-thread JSONL run log, one file per thread ------------------------
def _read_log_entries(thread_id: str, *, limit: int | None = None) -> list[dict]:
    """Parse one thread's JSONL log file into a list of event records.

    Lines that fail to parse are returned as ``{"_parse_error": ..., "_raw": ...}``
    so the viewer can show partial logs instead of nothing when a line is
    malformed (e.g. a write was interrupted).
    """
    path = default_log_path(thread_id)
    if not path.is_file():
        return []
    out: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                out.append({"_parse_error": str(e), "_line": i + 1, "_raw": line})
            if limit is not None and len(out) >= limit:
                break
    return out


async def list_logs(request: Request) -> JSONResponse:
    """GET /api/logs — all thread log files on disk + a thread_id→has_log map.

    The Threads list comes from SQLite (the canonical task index); the log
    directory is the source of truth for "which threads have a log file". A
    thread can exist without ever having produced a log (older sessions, or
    runs that errored before the first event), and a log file can exist for
    a thread whose state has been wiped. We report both so the UI shows the
    truth on each side.
    """
    files = list_log_files()
    log_threads = {f["thread_id"] for f in files}
    store = STATE.new_store()
    try:
        threads = store.list_threads()
    finally:
        store.close()
    thread_titles = {t["thread_id"]: t.get("title") or "Untitled" for t in threads}

    # Threads in the log dir but not in sqlite (orphaned logs).
    orphan_logs = [
        {
            "thread_id": f["thread_id"],
            "size": f["size"],
            "mtime": f["mtime"],
        }
        for f in files
        if f["thread_id"] not in thread_titles
    ]
    # Threads in sqlite, with a `has_log` flag.
    thread_rows = []
    for t in threads:
        thread_rows.append(
            {
                "thread_id": t["thread_id"],
                "title": t.get("title") or "Untitled",
                "updated_at": t.get("updated_at"),
                "has_log": t["thread_id"] in log_threads,
            }
        )
    # Sort: threads with logs first (newest updated_at), then without logs.
    thread_rows.sort(key=lambda r: (not r["has_log"], -(r.get("updated_at") or 0)))
    return JSONResponse(
        {
            "log_dir": str(DEFAULT_LOG_DIR),
            "threads": thread_rows,
            "orphan_logs": orphan_logs,
        }
    )


async def get_thread_log(request: Request) -> JSONResponse:
    """GET /api/logs/{thread_id} — parsed log records for one thread.

    Returns 404 if no log file exists for that thread. The full event stream
    — ModelRequested with merged wire kwargs, ModelResponded with raw payload,
    tool calls, results, checkpoints — comes back as a JSON array.
    """
    thread_id = request.path_params["thread_id"]
    path = default_log_path(thread_id)
    if not path.is_file():
        return JSONResponse({"error": "no log file for this thread"}, status_code=404)
    try:
        stat = path.stat()
    except OSError as e:
        return JSONResponse({"error": f"cannot stat log file: {e}"}, status_code=500)
    entries = _read_log_entries(thread_id)
    return JSONResponse(
        {
            "thread_id": thread_id,
            "path": str(path),
            "size": stat.st_size,
            "mtime": stat.st_mtime,
            "entries": entries,
            "n_entries": len(entries),
        }
    )


async def tail_thread_log(request: Request) -> JSONResponse:
    """GET /api/logs/{thread_id}/tail?since=<n> — entries past offset ``n``.

    Used by the UI to poll for new entries while a run is in flight, so the
    log viewer updates live without re-fetching the whole file.
    """
    thread_id = request.path_params["thread_id"]
    since = int(request.query_params.get("since", "0") or "0")
    entries = _read_log_entries(thread_id)
    return JSONResponse(
        {"thread_id": thread_id, "entries": entries[since:], "next_offset": len(entries)}
    )


# --- folders: grant a thread's agent read/write access to an extra directory ---
async def list_folders(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        paths = store.list_folders(thread_id)
    finally:
        store.close()
    return JSONResponse(paths)


async def add_folder(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    body = await request.json()
    raw = (body or {}).get("path")
    if not raw:
        return JSONResponse({"error": "missing 'path'"}, status_code=400)
    resolved = Path(raw).expanduser().resolve()
    if not resolved.is_dir():
        return JSONResponse({"error": f"not a directory: {resolved}"}, status_code=400)
    store = STATE.new_store()
    try:
        store.add_folder(thread_id, str(resolved))
    finally:
        store.close()
    return JSONResponse({"path": str(resolved)})


async def remove_folder(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    path = request.query_params.get("path")
    if not path:
        return JSONResponse({"error": "path query param is required"}, status_code=400)
    store = STATE.new_store()
    try:
        store.remove_folder(thread_id, path)
    finally:
        store.close()
    return JSONResponse({"removed": path})


class NoCacheStaticFiles(StaticFiles):
    """Dev-mode static files — browsers (and embedded webviews) cache CSS/JS
    aggressively by default and a plain hard-refresh doesn't reliably bust
    that. We edit ``static/`` constantly while iterating on the UI, so make
    every response explicitly non-cacheable rather than chasing stale-cache
    ghosts again."""

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        for h in ("etag", "last-modified"):
            if h in response.headers:
                del response.headers[h]
        return response


def make_app(db_path: str, allow_write: bool, model_name: str | None) -> Starlette:
    global STATE
    STATE = AppState(db_path, allow_write, model_name)
    routes = [
        Route("/", index),
        Route("/api/debug/tools", debug_tools),
        Route("/api/threads", list_threads),
        Route("/api/threads/{thread_id}", get_thread),
        Route("/api/chat/stream", chat_stream),
        Route("/api/threads/{thread_id}/knowledge", list_knowledge, methods=["GET"]),
        Route("/api/threads/{thread_id}/knowledge", add_knowledge, methods=["POST"]),
        Route("/api/threads/{thread_id}/knowledge", remove_knowledge, methods=["DELETE"]),
        Route("/api/libraries", list_libraries, methods=["GET"]),
        Route("/api/libraries", create_library, methods=["POST"]),
        Route("/api/libraries/{library_id}/knowledge", list_library_knowledge, methods=["GET"]),
        Route("/api/libraries/{library_id}/knowledge", add_library_knowledge, methods=["POST"]),
        Route("/api/libraries/{library_id}/knowledge", remove_library_knowledge, methods=["DELETE"]),
        Route("/api/threads/{thread_id}/libraries", list_thread_libraries, methods=["GET"]),
        Route("/api/threads/{thread_id}/libraries", attach_thread_library, methods=["POST"]),
        Route("/api/threads/{thread_id}/libraries", detach_thread_library, methods=["DELETE"]),
        Route("/api/threads/{thread_id}", update_thread, methods=["PATCH"]),
        Route("/api/threads/{thread_id}/project", get_thread_project, methods=["GET"]),
        Route("/api/libraries/{library_id}", update_library, methods=["PATCH"]),
        Route("/api/libraries/{library_id}", delete_library, methods=["DELETE"]),
        Route("/api/projects", list_projects, methods=["GET"]),
        Route("/api/projects", create_project, methods=["POST"]),
        Route("/api/projects/{project_id}", get_project, methods=["GET"]),
        Route("/api/projects/{project_id}", update_project, methods=["PATCH"]),
        Route("/api/projects/{project_id}", delete_project, methods=["DELETE"]),
        Route("/api/projects/{project_id}/directories", list_project_directories, methods=["GET"]),
        Route("/api/projects/{project_id}/directories", add_project_directory, methods=["POST"]),
        Route("/api/projects/{project_id}/directories", remove_project_directory, methods=["DELETE"]),
        Route("/api/threads/{thread_id}/folders", list_folders, methods=["GET"]),
        Route("/api/threads/{thread_id}/folders", add_folder, methods=["POST"]),
        Route("/api/threads/{thread_id}/folders", remove_folder, methods=["DELETE"]),
        Route("/api/logs", list_logs),
        Route("/api/logs/{thread_id}/tail", tail_thread_log),
        Route("/api/logs/{thread_id}", get_thread_log),
        Mount("/static", NoCacheStaticFiles(directory=STATIC_DIR), name="static"),
    ]
    return Starlette(routes=routes)


def run(host: str = "127.0.0.1", port: int = 8765, db: str = "openmate.sqlite",
        allow_write: bool = False, model: str | None = None) -> None:
    import uvicorn

    app = make_app(db, allow_write, model)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()
