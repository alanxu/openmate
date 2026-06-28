"""Kernel: serialization round-trip, Usage math, and state helpers."""

from __future__ import annotations

from openmate.kernel.codec import state_from_jsonable, state_to_jsonable
from openmate.kernel.types import (
    Message,
    RunState,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)


def _sample_state() -> RunState:
    return RunState(
        thread_id="t1",
        messages=[
            Message("system", [TextPart("be helpful")]),
            Message("user", [TextPart("what is 2+2?")]),
            Message(
                "assistant",
                [
                    ThinkingPart("the user wants arithmetic", signature="sig123"),
                    TextPart("let me compute"),
                    ToolCallPart("c1", "calc", {"e": "2+2"}),
                ],
            ),
            Message("tool", [ToolResultPart("c1", [TextPart("4")], is_error=False)]),
            Message("assistant", [TextPart("It is 4.")]),
        ],
        step=2,
        status="done",
        usage=Usage(prompt_tokens=10, completion_tokens=5),
        rev=4,
    )


def test_state_roundtrip_is_lossless():
    s = _sample_state()
    assert state_from_jsonable(state_to_jsonable(s)) == s


def test_usage_accumulates():
    total = Usage(1, 2) + Usage(3, 4)
    assert (total.prompt_tokens, total.completion_tokens) == (4, 6)
    assert total.total_tokens == 10


def test_message_text_accessor():
    m = Message(
        "assistant",
        [ThinkingPart("pondering"), TextPart("a"), ToolCallPart("c", "n", {}), TextPart("b")],
    )
    assert m.text == "ab"  # thinking is not part of the user-visible text
    assert m.tool_calls[0].name == "n"


def test_state_helpers():
    s = RunState("t", [Message("user", [TextPart("hi")])])
    s2 = s.with_messages(Message("assistant", [TextPart("yo")]))
    assert len(s2.messages) == 2 and s2.rev == 1
    s3 = s2.advance()
    assert s3.step == 1 and s3.rev == 2
    result = s3.stop("done", "natural").to_result()
    assert result.ok and result.text == "yo" and result.reason == "natural"
