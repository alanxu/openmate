"""The Anthropic/MiniMax adapter translation layer (no network).

These tests pin the OpenMate <-> Anthropic Messages API mapping that makes the
MiniMax endpoint work.
"""

from __future__ import annotations

import types

from openmate.adapters.models.anthropic import AnthropicModel
from openmate.kernel.types import Message, TextPart, ToolCallPart, ToolResultPart
from openmate.ports.model import ModelRequest


def _block(**kw):
    return types.SimpleNamespace(**kw)


class _DummyMessages:
    def __init__(self, raw):
        self._raw = raw
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._raw


class _DummyClient:
    def __init__(self, raw):
        self.messages = _DummyMessages(raw)


def _model(client=None) -> AnthropicModel:
    return AnthropicModel("MiniMax-M2", client=client or object())


def test_system_is_extracted_and_tools_become_user_messages():
    m = _model()
    msgs = [
        Message("system", [TextPart("be nice")]),
        Message("user", [TextPart("hi")]),
        Message("assistant", [TextPart("checking"), ToolCallPart("c1", "calc", {"x": 1})]),
        Message("tool", [ToolResultPart("c1", [TextPart("42")])]),
    ]
    system, wire = m._messages_to_wire(msgs)

    assert system == "be nice"
    assert wire[0] == {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    assert wire[1]["role"] == "assistant"
    assert any(b["type"] == "tool_use" and b["id"] == "c1" for b in wire[1]["content"])
    # tool results are carried as tool_result blocks on a *user* message
    assert wire[2]["role"] == "user"
    assert wire[2]["content"][0]["tool_use_id"] == "c1"
    assert wire[2]["content"][0]["content"] == "42"


def test_response_parsing_maps_blocks_and_usage():
    m = _model()
    raw = types.SimpleNamespace(
        content=[
            _block(type="text", text="hello"),
            _block(type="tool_use", id="c1", name="calc", input={"x": 1}),
        ],
        usage=types.SimpleNamespace(input_tokens=10, output_tokens=3),
        stop_reason="tool_use",
    )
    resp = m._response_from_wire(raw)

    assert resp.finish_reason == "tool_calls"
    assert resp.message.text == "hello"
    assert resp.message.tool_calls[0].name == "calc"
    assert resp.message.tool_calls[0].args == {"x": 1}
    assert resp.usage.prompt_tokens == 10 and resp.usage.completion_tokens == 3


async def test_generate_calls_client_with_right_kwargs():
    raw = types.SimpleNamespace(
        content=[_block(type="text", text="hi there")],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="end_turn",
    )
    client = _DummyClient(raw)
    m = _model(client)
    resp = await m.generate(ModelRequest(messages=[Message("user", [TextPart("hi")])], max_tokens=64))

    assert resp.message.text == "hi there"
    sent = client.messages.calls[0]
    assert sent["model"] == "MiniMax-M2"
    assert sent["max_tokens"] == 64
    assert sent["messages"][0]["role"] == "user"
