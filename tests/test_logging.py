"""JSONL run logger — captures the raw model request/response (offline)."""

from __future__ import annotations

import json

from openmate.adapters.tracers.jsonl import (
    JsonlLogger,
    _raw_to_jsonable,
    default_log_path,
)
from openmate.kernel.events import EventBus, ModelRequested, ModelResponded
from openmate.kernel.types import Message, TextPart, Usage
from openmate.ports.model import ModelRequest, ModelResponse


def test_jsonl_logger_captures_raw_model_io(tmp_path):
    bus = EventBus()
    path = tmp_path / "run.jsonl"
    logger = JsonlLogger(path).attach(bus)

    req = ModelRequest(
        messages=[
            Message("system", [TextPart("be helpful")]),
            Message("user", [TextPart("what is 2+2?")]),
        ],
        max_tokens=64,
    )
    bus.emit(ModelRequested("t1", 0, 0.0, req))
    resp = ModelResponse(
        Message("assistant", [TextPart("4")]),
        Usage(prompt_tokens=10, completion_tokens=1),
        finish_reason="stop",
        raw={"id": "msg_abc", "stop_reason": "end_turn"},
    )
    bus.emit(ModelResponded("t1", 0, 0.0, resp, 12.5))
    logger.close()

    recs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    # the full request (the prompt the model received) is captured
    assert recs[0]["event"] == "ModelRequested"
    msgs = recs[0]["request"]["messages"]
    assert msgs[0]["content"][0]["text"] == "be helpful"
    assert msgs[1]["content"][0]["text"] == "what is 2+2?"
    assert recs[0]["request"]["max_tokens"] == 64

    # the full response, including the provider's raw payload, is captured
    assert recs[1]["event"] == "ModelResponded"
    assert recs[1]["message"]["content"][0]["text"] == "4"
    assert recs[1]["finish_reason"] == "stop"
    assert recs[1]["usage"]["prompt_tokens"] == 10
    assert recs[1]["raw"]["id"] == "msg_abc"
    assert recs[1]["ms"] == 12.5


def test_raw_payload_serialized_via_model_dump():
    class FakeRaw:  # mimics an anthropic SDK (pydantic) response object
        def model_dump(self, mode="python"):
            return {"id": "x", "mode": mode}

    assert _raw_to_jsonable(FakeRaw()) == {"id": "x", "mode": "json"}


def test_default_log_path_under_openmate_logs():
    p = default_log_path("thread9")
    assert p.parent.name == "logs" and p.parent.parent.name == ".openmate"
    assert p.suffix == ".jsonl" and "thread9" in p.name


def test_attach_if_enabled_is_gated_by_env(tmp_path, monkeypatch):
    """The single switch: OPENMATE_LOG off -> nothing; set to a path -> logs there."""
    import openmate.adapters.tracers.jsonl as j

    j._session_logger = None  # reset module singleton for the test
    bus = EventBus()

    monkeypatch.delenv("OPENMATE_LOG", raising=False)
    j.attach_if_enabled(bus)
    assert j._session_logger is None  # off by default

    p = tmp_path / "log.jsonl"
    monkeypatch.setenv("OPENMATE_LOG", str(p))  # a path value logs there
    j.attach_if_enabled(bus)
    bus.emit(
        ModelResponded(
            "t", 0, 0.0,
            ModelResponse(Message("assistant", [TextPart("hi")]), Usage(1, 1), finish_reason="stop"),
            1.0,
        )
    )
    assert p.exists() and "ModelResponded" in p.read_text(encoding="utf-8")

    j._session_logger.close()
    j._session_logger = None  # don't leak into other tests
