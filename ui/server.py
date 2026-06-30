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

        # One shared embedder/vector-store/retriever for the whole server. Every
        # thread's knowledge lives in the same collection, isolated only by the
        # `thread_id` metadata tag — see rag/tools.py:RetrieveTool.scope_to_thread
        # and rag/pipeline.py:NaivePipeline.ingest's `extra_metadata`.
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
        extra_roots = store.list_folders(thread_id)
        # The chat UI always gets write + shell by default (this is a local
        # single-user assistant, not a multi-tenant service), scoped to the
        # server's cwd plus whatever folders this thread has added via the '+'
        # menu — see openmate/adapters/tools/builtin.py:make_shell_tool's
        # docstring for the (cwd-only) confinement caveat. ``self.allow_write``
        # is no longer load-bearing here but is kept on AppState/CLI for
        # backward compatibility with anything else constructing it.
        tools = [*all_tools(extra_roots), RetrieveTool(self.retriever)]
        return Agent(
            name="openmate",
            model=default_model(self.model_name),
            instructions=DEFAULT_INSTRUCTIONS,
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


# --- knowledge: ingest into the shared RAG store, tagged with thread_id --------
async def list_knowledge(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    store = STATE.new_store()
    try:
        items = store.list_knowledge(thread_id)
    finally:
        store.close()
    return JSONResponse(
        [{"source": i["source"], "added_at": i["added_at"], "n_chunks": len(i["chunk_ids"])} for i in items]
    )


async def add_knowledge(request: Request) -> JSONResponse:
    thread_id = request.path_params["thread_id"]
    content_type = request.headers.get("content-type", "")

    label: str
    path: Path
    if "multipart/form-data" in content_type:
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "read"):
            return JSONResponse({"error": "missing 'file' in form data"}, status_code=400)
        label = upload.filename or f"upload-{uuid.uuid4().hex[:8]}.txt"
        dest_dir = STATE.uploads_dir / thread_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / label
        data = await upload.read()
        path.write_bytes(data)
    else:
        body = await request.json()
        text = (body or {}).get("text")
        if not text or not text.strip():
            return JSONResponse({"error": "missing 'text' (or upload a 'file')"}, status_code=400)
        label = (body or {}).get("name") or f"pasted-{uuid.uuid4().hex[:8]}.txt"
        if not label.endswith((".txt", ".md")):
            label += ".txt"
        dest_dir = STATE.uploads_dir / thread_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = dest_dir / label
        path.write_text(text, encoding="utf-8")

    report = await STATE.pipeline.ingest(str(path), extra_metadata={"thread_id": thread_id})
    store = STATE.new_store()
    try:
        store.add_knowledge(thread_id, label, report.chunk_ids)
    finally:
        store.close()
    return JSONResponse({"source": label, "n_chunks": len(report.chunk_ids)})


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
        Route("/api/threads/{thread_id}/folders", list_folders, methods=["GET"]),
        Route("/api/threads/{thread_id}/folders", add_folder, methods=["POST"]),
        Route("/api/threads/{thread_id}/folders", remove_folder, methods=["DELETE"]),
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
