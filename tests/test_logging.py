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
    """The single switch: OPENMATE_LOG off -> nothing; set -> logger attached."""
    import openmate.adapters.tracers.jsonl as j

    j._singleton = None  # reset module singleton for the test
    j._attached_buses.clear()
    bus = EventBus()

    monkeypatch.delenv("OPENMATE_LOG", raising=False)
    assert j.attach_if_enabled(bus) is None  # off by default

    monkeypatch.setenv("OPENMATE_LOG", "1")  # truthy -> logger attached
    logger = j.attach_if_enabled(bus)
    assert logger is not None
    bus.emit(
        ModelResponded(
            "t", 0, 0.0,
            ModelResponse(Message("assistant", [TextPart("hi")]), Usage(1, 1), finish_reason="stop"),
            1.0,
        )
    )
    logger.close()
    j._singleton = None  # don't leak into other tests
    # The event landed in t.jsonl under the default log dir.
    written = (j.DEFAULT_LOG_DIR / "t.jsonl")
    assert written.is_file() and "ModelResponded" in written.read_text(encoding="utf-8")
    written.unlink()


def test_force_attach_writes_per_thread_files(tmp_path, monkeypatch):
    """force_attach() (used by the UI server) always attaches a per-thread logger,
    even without OPENMATE_LOG. Setting OPENMATE_LOG=0 is the explicit opt-out."""
    import openmate.adapters.tracers.jsonl as j

    j._singleton = None
    j._attached_buses.clear()
    monkeypatch.delenv("OPENMATE_LOG", raising=False)
    bus = EventBus()
    logger = j.force_attach(bus, log_dir=tmp_path)
    assert logger is not None
    bus.emit(
        ModelRequested(
            "thr-A", 0, 0.0,
            ModelRequest(messages=[Message("user", [TextPart("hi")])], max_tokens=8),
        )
    )
    bus.emit(
        ModelResponded(
            "thr-A", 0, 0.0,
            ModelResponse(Message("assistant", [TextPart("hello")]), Usage(1, 1), finish_reason="stop"),
            1.0,
        )
    )
    bus.emit(
        ModelResponded(
            "thr-B", 0, 0.0,
            ModelResponse(Message("assistant", [TextPart("other")]), Usage(1, 1), finish_reason="stop"),
            1.0,
        )
    )
    logger.close()
    a = (tmp_path / "thr-A.jsonl").read_text(encoding="utf-8")
    b = (tmp_path / "thr-B.jsonl").read_text(encoding="utf-8")
    assert "ModelRequested" in a and "ModelResponded" in a
    assert "ModelResponded" in b and "ModelRequested" not in b

    j._singleton = None
    j._attached_buses.clear()


def test_list_log_files_returns_per_thread(tmp_path):
    import openmate.adapters.tracers.jsonl as j

    j.DEFAULT_LOG_DIR = tmp_path
    (tmp_path / "alpha.jsonl").write_text("{}\n", encoding="utf-8")
    (tmp_path / "beta.jsonl").write_text("{}\n", encoding="utf-8")
    files = j.list_log_files()
    assert {f["thread_id"] for f in files} == {"alpha", "beta"}
    assert all(f["path"].endswith(".jsonl") for f in files)


def test_explicit_optout_blocks_force_attach(monkeypatch):
    """Setting OPENMATE_LOG=0 is the escape hatch — even force_attach refuses."""
    import openmate.adapters.tracers.jsonl as j

    j._singleton = None
    j._attached_buses.clear()
    monkeypatch.setenv("OPENMATE_LOG", "0")
    bus = EventBus()
    assert j.force_attach(bus) is None
    assert j.attach_if_enabled(bus) is None
    j._singleton = None
    j._attached_buses.clear()
