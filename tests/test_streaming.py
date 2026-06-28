"""Token streaming — the opt-in ``model.stream`` path (offline, FakeModel).

The default path (``stream_model=False``, ``generate``) is exercised everywhere
else; these pin the additive streaming branch: deltas are emitted and reassemble
into the same ``ModelResponse`` the non-streaming path would produce.
"""

from __future__ import annotations

from helpers import make_services

from openmate.adapters.models.fake import FakeModel, text_response, tool_call_response
from openmate.adapters.tools.builtin import calculator
from openmate.kernel.agent import Agent
from openmate.kernel.events import ModelResponded, ModelStreamed
from openmate.ports.model import ModelRequest


async def test_fake_model_stream_yields_text_then_done():
    model = FakeModel([text_response("hello world")])
    deltas = [d async for d in model.stream(ModelRequest(messages=[]))]
    assert deltas[0].kind == "text" and deltas[0].data == "hello world"
    assert deltas[-1].kind == "done"
    assert deltas[-1].data.message.text == "hello world"  # terminal delta = full response


async def test_stream_model_path_emits_deltas_and_reassembles():
    model = FakeModel(
        [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("It is 4.")]
    )
    svc, events = make_services()
    agent = Agent(
        name="a", model=model, instructions="be helpful",
        services=svc, tools=[calculator], stream_model=True,
    )
    result = await agent.run("what is 2+2?")

    assert result.ok and result.text == "It is 4."
    streamed = [e for e in events if isinstance(e, ModelStreamed)]
    assert any(e.delta.kind == "text" and e.delta.data == "It is 4." for e in streamed)
    assert any(e.delta.kind == "done" for e in streamed)
    # the loop still emits ModelResponded carrying the reassembled response
    responded = [e for e in events if isinstance(e, ModelResponded)]
    assert responded and responded[-1].response.message.text == "It is 4."


async def test_default_path_does_not_stream():
    model = FakeModel([text_response("hi")])
    svc, events = make_services()
    agent = Agent(name="a", model=model, instructions="i", services=svc)  # stream_model defaults False
    await agent.run("hello")
    assert not any(isinstance(e, ModelStreamed) for e in events)
