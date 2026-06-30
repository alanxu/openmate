# OpenMate UI

A browser UI for OpenMate — new task, chat history sidebar, and an in-chat
trajectory view modeled on the Claude Code desktop app: tool calls render as
collapsed cards (name + status), click to expand args/result/latency.

```
openmate ui                       # http://127.0.0.1:8765
openmate ui --port 9000 --db my.sqlite --allow-write
```

## Architecture

- `server.py` — a Starlette app. No new agent behavior: every route is a thin
  consumer of the same primitives the rest of OpenMate already exposes.
  - `GET /` — serves `static/index.html`.
  - `GET /api/threads` — `SQLiteStore.list_threads()`, for the sidebar.
  - `GET /api/threads/{id}` — full message history for reopening a past chat.
  - `GET /api/chat/stream?thread_id=&message=` — Server-Sent Events over
    `agent.stream()`. One SSE event per `openmate.kernel.events.Event`
    subtype (`RunStarted`, `MessageAdded`, `ModelStreamed`, `ToolCallRequested`,
    `ToolReturned`, `RunFinished`, …) — see `event_to_json()`.
- `static/` — vanilla HTML/CSS/JS, no build step (no React/bundler dependency).
  `app.js` opens a browser `EventSource` against the stream endpoint and
  renders bubbles + collapsible trace cards as events arrive.

## Why a separate folder, not inside `openmate/`

This is an edge consumer of the library (like `examples/` or `servers/`), not
part of it — `openmate/` stays free of any web-framework dependency. The CLI
(`openmate/cli.py`'s `ui` subcommand) lazily imports `server.py` by file path,
so `openmate run`/`chat` never need starlette/uvicorn installed.

## Persistence

Always backed by `SQLiteStore` (`--db`, default `openmate.sqlite`) — the
sidebar's history list requires persistence across restarts, so this
subcommand doesn't offer an in-memory mode the way `run`/`chat` do.

## Known gaps (Phase 0)

- No thread deletion/rename from the UI yet (would need new endpoints + a
  `SQLiteStore` method).
- No live token-level streaming of *thinking* output beyond what
  `ModelStreamed` already carries (off by default — `stream_model=False`).
- Single in-flight run per browser tab (no multi-tab/concurrent-run guard).
