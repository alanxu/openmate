"""A JSONL event logger — writes the run's event stream to a file.

Like ``ConsoleTracer``, it's just a consumer of the event bus (architecture P2),
so it captures model I/O without any change to the loop or the model: each
``ModelRequested`` carries the full request and each ``ModelResponded`` carries
the full response (including ``raw`` — the provider's literal payload). One JSON
object per line; the file is flushed per event so a crashed run still has a
complete log up to the failure.

Default location: ``~/.openmate/logs/<timestamp>.jsonl``.
"""

from __future__ import annotations

import atexit
import json
import os
from datetime import datetime
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


def default_log_path(thread_id: str | None = None) -> Path:
    """A timestamped log path under ``~/.openmate/logs``."""
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    stem = f"{ts}-{thread_id}" if thread_id else ts
    return DEFAULT_LOG_DIR / f"{stem}.jsonl"


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


def event_to_jsonable(ev: Event) -> dict[str, Any]:
    """Serialize one event, with full request/response payloads for model I/O."""
    rec: dict[str, Any] = {
        "event": type(ev).__name__,
        "thread_id": ev.thread_id,
        "step": ev.step,
        "ts": ev.ts,
    }
    if isinstance(ev, ModelRequested):
        req = ev.request
        rec["request"] = {
            "messages": [message_to_jsonable(m) for m in req.messages],
            "tools": [
                {"name": t.name, "description": t.description, "parameters": t.parameters}
                for t in (req.tools or [])
            ],
            "temperature": req.temperature,
            "max_tokens": req.max_tokens,
        }
    elif isinstance(ev, ModelResponded):
        resp = ev.response
        rec["ms"] = ev.ms
        rec["finish_reason"] = resp.finish_reason
        rec["usage"] = {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }
        rec["message"] = message_to_jsonable(resp.message)
        rec["raw"] = _raw_to_jsonable(resp.raw)  # the provider's literal response payload
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


class JsonlLogger:
    """Subscribes to the event bus and appends each event as a JSON line."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

    def attach(self, bus) -> "JsonlLogger":
        bus.subscribe(self.record)
        return self

    def record(self, event: Event) -> None:
        # Per-token stream deltas would flood the log; the complete response is
        # already captured in the ModelResponded record. Skip them.
        if isinstance(event, ModelStreamed):
            return
        self._fh.write(json.dumps(event_to_jsonable(event), ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


_session_logger: JsonlLogger | None = None


def attach_if_enabled(bus) -> None:
    """The single logging switch: attach a shared session logger to ``bus`` iff the
    ``OPENMATE_LOG`` env var is set. Any component that builds an event bus calls
    this; the CLI and the eval harness just set ``OPENMATE_LOG`` (their ``--log``
    flag does). Set it to ``1`` for the default path, or to a file path to choose."""
    global _session_logger
    val = os.environ.get("OPENMATE_LOG")
    if not val or val == "0":
        return
    if _session_logger is None:
        path = default_log_path() if val in ("1", "true", "yes", "on") else Path(val)
        _session_logger = JsonlLogger(path)
        atexit.register(_session_logger.close)
        print(f"openmate: logging to {_session_logger.path}")
    _session_logger.attach(bus)
