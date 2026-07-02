"""Per-thread JSONL run loggers — writes the run's event stream to one file per thread.

Like every tracer, this is just a consumer of the event bus (architecture P2):
the loop never knows it's there. Each ``ModelRequested`` carries the full
``ModelRequest`` plus the provider's wire kwargs (the literal HTTP body sent),
and each ``ModelResponded`` carries the provider's ``raw`` response object —
so the JSONL file is a complete, replayable record of the agent↔model dialogue.

Default location: ``~/.openmate/logs/<thread_id>.jsonl`` (one file per thread).
"""

from __future__ import annotations

import atexit
import json
import os
import threading
from pathlib import Path
from typing import Any

from ...kernel.codec import message_to_jsonable, part_to_jsonable
from ...kernel.events import (
    CheckpointSaved,
    Event,
    MessageAdded,
    ModelRequested,
    ModelResponded,
    ModelStreamed,
    RunFinished,
    RunStarted,
    ToolCallRequested,
    ToolReturned,
)

DEFAULT_LOG_DIR = Path.home() / ".openmate" / "logs"


def default_log_path(thread_id: str) -> Path:
    """Path to the log file for one thread — ``~/.openmate/logs/<thread_id>.jsonl``."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in (thread_id or "unknown"))
    return DEFAULT_LOG_DIR / f"{safe}.jsonl"


def list_log_files() -> list[dict[str, Any]]:
    """All thread log files under ``DEFAULT_LOG_DIR``, newest first.

    Each entry: ``{thread_id, path, size, mtime}``. ``thread_id`` is the file stem.
    """
    if not DEFAULT_LOG_DIR.is_dir():
        return []
    out: list[dict[str, Any]] = []
    for p in DEFAULT_LOG_DIR.glob("*.jsonl"):
        try:
            stat = p.stat()
        except OSError:
            continue
        out.append(
            {"thread_id": p.stem, "path": str(p), "size": stat.st_size, "mtime": stat.st_mtime}
        )
    out.sort(key=lambda d: d["mtime"], reverse=True)
    return out


def _raw_to_jsonable(raw: Any) -> Any:
    """Best-effort JSON form of a provider's raw payload (pydantic / dataclass / str)."""
    if raw is None:
        return None
    dump = getattr(raw, "model_dump", None)  # anthropic SDK objects are pydantic
    if callable(dump):
        try:
            return dump(mode="json")
        except Exception:  # noqa: BLE001
            pass
    try:
        return json.loads(json.dumps(raw, default=str))
    except Exception:  # noqa: BLE001
        return str(raw)


def _serialize_model_request(ev: ModelRequested) -> dict[str, Any]:
    req = ev.request
    out: dict[str, Any] = {
        "messages": [message_to_jsonable(m) for m in req.messages],
        "tools": [
            {"name": t.name, "description": t.description, "parameters": t.parameters}
            for t in (req.tools or [])
        ],
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if ev.wire is not None:
        out["wire"] = ev.wire
    return out


def _event_record(ev: Event) -> dict[str, Any]:
    """Serialize a non-ModelRequested event."""
    rec: dict[str, Any] = {
        "event": type(ev).__name__,
        "thread_id": ev.thread_id,
        "step": ev.step,
        "ts": ev.ts,
    }
    if isinstance(ev, ModelResponded):
        resp = ev.response
        rec["ms"] = ev.ms
        rec["finish_reason"] = resp.finish_reason
        rec["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        rec["message"] = message_to_jsonable(resp.message)
        rec["raw"] = _raw_to_jsonable(resp.raw)
    elif isinstance(ev, MessageAdded):
        rec["message"] = message_to_jsonable(ev.message)
    elif isinstance(ev, ToolCallRequested):
        rec["call"] = part_to_jsonable(ev.call)
    elif isinstance(ev, ToolReturned):
        rec["ms"] = ev.ms
        rec["result"] = part_to_jsonable(ev.result)
    elif isinstance(ev, CheckpointSaved):
        rec["rev"] = ev.rev
    elif isinstance(ev, RunFinished):
        r = ev.result
        rec["result"] = {
            "status": r.status,
            "reason": r.reason,
            "steps": r.steps,
            "text": r.text,
            "usage": {
                "prompt_tokens": r.usage.prompt_tokens,
                "completion_tokens": r.usage.completion_tokens,
            },
        }
    elif isinstance(ev, RunStarted):
        pass
    return rec


class PerThreadJsonlLogger:
    """One bus subscriber that writes a separate ``<thread_id>.jsonl`` per thread.

    The logger lazy-opens each thread's file on its first event. The
    ``ModelRequested`` event is buffered until the matching ``ModelResponded``
    arrives — the model adapter stashes the provider's literal wire kwargs
    (the HTTP body sent) on ``raw._openmate_wire`` during ``generate()``, so
    by the time the response reaches us we can merge the wire into the
    buffered request and write one combined record. This is what makes the
    JSONL a faithful 100% record of the agent↔model dialogue.
    """

    def __init__(self, log_dir: Path | None = None) -> None:
        self.log_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._handles: dict[str, Any] = {}
        # (thread_id, step) -> the in-flight ModelRequested event
        self._pending_requests: dict[tuple[str, int], ModelRequested] = {}

    def attach(self, bus) -> "PerThreadJsonlLogger":
        bus.subscribe(self.record)
        return self

    def _handle(self, thread_id: str) -> Any:
        h = self._handles.get(thread_id)
        if h is not None:
            return h
        path = self.log_dir / default_log_path(thread_id).name
        h = path.open("a", encoding="utf-8")
        self._handles[thread_id] = h
        return h

    def _flush_pending(self, thread_id: str, step: int) -> None:
        """If a request is still buffered for this step, write it out as-is
        (without wire kwargs — happens when the response never arrived)."""
        key = (thread_id, step)
        ev = self._pending_requests.pop(key, None)
        if ev is None:
            return
        rec = {
            "event": "ModelRequested",
            "thread_id": ev.thread_id,
            "step": ev.step,
            "ts": ev.ts,
            "request": _serialize_model_request(ev),
            "note": "no response observed — wire kwargs unavailable",
        }
        h = self._handle(thread_id)
        h.write(json.dumps(rec, ensure_ascii=False) + "\n")
        h.flush()

    def record(self, event: Event) -> None:
        # Per-token stream deltas would flood the log; the complete response is
        # captured in the ModelResponded record. Skip them.
        if isinstance(event, ModelStreamed):
            return
        thread_id = getattr(event, "thread_id", None) or "unknown"

        with self._lock:
            if isinstance(event, ModelRequested):
                # Buffer; we'll merge the wire in when ModelResponded arrives.
                # If a previous request for this step was never responded to,
                # flush it as-is so it isn't lost.
                self._flush_pending(thread_id, event.step)
                self._pending_requests[(thread_id, event.step)] = event
                return

            if isinstance(event, ModelResponded):
                # Pull the wire kwargs the adapter stashed on raw during generate().
                wire = getattr(getattr(event.response, "raw", None), "_openmate_wire", None)
                req = self._pending_requests.pop((thread_id, event.step), None)
                if req is not None:
                    if wire is not None and req.wire is None:
                        req.wire = wire
                    rec = {
                        "event": "ModelRequested",
                        "thread_id": req.thread_id,
                        "step": req.step,
                        "ts": req.ts,
                        "request": _serialize_model_request(req),
                    }
                    h = self._handle(thread_id)
                    h.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    h.flush()

            rec = _event_record(event)
            h = self._handle(thread_id)
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
            h.flush()

    def close(self) -> None:
        with self._lock:
            # Flush any unresponded requests so no event is silently lost.
            for (thread_id, step) in list(self._pending_requests.keys()):
                self._flush_pending(thread_id, step)
            for h in self._handles.values():
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass
            self._handles.clear()

    def close_thread(self, thread_id: str) -> None:
        with self._lock:
            # Flush any unresponded requests for this thread.
            for key in list(self._pending_requests.keys()):
                if key[0] == thread_id:
                    self._flush_pending(*key)
            h = self._handles.pop(thread_id, None)
            if h is not None:
                try:
                    h.close()
                except Exception:  # noqa: BLE001
                    pass


# Backwards-compat: keep the original JsonlLogger name working for tests
# (tests/test_logging.py imports it) — it writes everything to a single
# caller-specified path regardless of the event's thread_id. Internally it
# still uses the buffering design so the request/response pairing still works.
class JsonlLogger(PerThreadJsonlLogger):
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        super().__init__(log_dir=path.parent)
        self._fixed_path = path
        self._stem = path.stem
        # Pre-open the requested file.
        self._handles[self._stem] = path.open("a", encoding="utf-8")

    def _handle(self, thread_id: str) -> Any:
        # Always write to the caller-specified file, regardless of thread_id.
        return self._handles[self._stem]


# --- attach helpers -------------------------------------------------------------
# One process-level logger instance, subscribed to every bus that asks for it.
# File handles are shared and protected by ``PerThreadJsonlLogger._lock`` — no
# duplication, no handle leaks even under the UI server's per-request bus churn.
_attached_buses: set[int] = set()
_singleton: "PerThreadJsonlLogger | None" = None


def _explicit_optout() -> bool:
    """``OPENMATE_LOG=0`` is the escape hatch — blocks even forced attaches."""
    val = os.environ.get("OPENMATE_LOG")
    return bool(val and val in ("0", "false", "no", "off"))


def _get_singleton(log_dir: Path | None = None) -> "PerThreadJsonlLogger":
    global _singleton
    if _singleton is None:
        _singleton = PerThreadJsonlLogger(log_dir=log_dir)
        atexit.register(_singleton.close)
    return _singleton


def attach_if_enabled(bus, *, log_dir: Path | None = None) -> "PerThreadJsonlLogger | None":
    """Attach a per-thread JSONL logger to ``bus`` iff ``OPENMATE_LOG`` is set.

    The CLI sets it via ``--log``; the eval harness sets it via ``--log``.
    """
    if _explicit_optout() or id(bus) in _attached_buses:
        return _singleton
    val = os.environ.get("OPENMATE_LOG")
    if not val:
        return None
    logger = _get_singleton(log_dir=log_dir)
    logger.attach(bus)
    _attached_buses.add(id(bus))
    return logger


def force_attach(bus, *, log_dir: Path | None = None) -> "PerThreadJsonlLogger | None":
    """Attach a per-thread JSONL logger unconditionally — used by the UI server.

    The UI's log viewer needs data, so the server always turns logging on,
    regardless of ``OPENMATE_LOG``. Setting ``OPENMATE_LOG=0`` still blocks
    even this path (escape hatch for users who don't want logs on disk).
    """
    if _explicit_optout():
        return None
    if id(bus) in _attached_buses:
        return _singleton
    logger = _get_singleton(log_dir=log_dir)
    logger.attach(bus)
    _attached_buses.add(id(bus))
    return logger


def get_logger() -> "PerThreadJsonlLogger | None":
    """The process-wide singleton — created lazily by attach/force_attach, or None
    if logging is disabled or has never been attached."""
    return _singleton