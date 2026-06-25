"""The loop: ReAct, the step cap, short-term memory, streaming, and determinism."""

from __future__ import annotations

from helpers import make_services

from openmate.adapters.models.fake import FakeModel, text_response, tool_call_response
from openmate.adapters.tools.builtin import calculator
from openmate.kernel.agent import Agent
from openmate.kernel.events import RunFinished, RunStarted, ToolReturned


def _agent(model, svc, **kw) -> Agent:
    return Agent(
        name="a", model=model, instructions="be helpful", services=svc, tools=[calculator], **kw
    )


async def test_tool_call_then_final_answer():
    model = FakeModel(
        [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("It is 4.")]
    )
    svc, events = make_services()
    result = await _agent(model, svc).run("what is 2+2?")

    assert result.ok
    assert result.text == "It is 4."
    assert result.reason == "natural"
    # the calculator actually ran and returned a non-error result
    returns = [e for e in events if isinstance(e, ToolReturned)]
    assert len(returns) == 1 and not returns[0].result.is_error
    assert returns[0].result.content[0].text == "4"
    assert any(isinstance(e, RunStarted) for e in events)
    assert any(isinstance(e, RunFinished) for e in events)


async def test_step_cap_stops_runaway_loop():
    # a model that never stops calling tools must be halted by the step cap
    script = [tool_call_response(f"c{i}", "calculator", {"expression": "1+1"}) for i in range(50)]
    svc, _ = make_services()
    result = await _agent(FakeModel(script), svc, max_steps=3).run("loop forever")

    assert result.reason == "max_steps"
    assert result.steps == 3


async def test_bad_tool_args_are_recoverable():
    # model calls calculator with a missing arg, sees the error, then answers
    model = FakeModel(
        [tool_call_response("c1", "calculator", {}), text_response("Sorry, fixed it.")]
    )
    svc, events = make_services()
    result = await _agent(model, svc).run("do math")

    assert result.ok  # the run did not crash
    err = [e for e in events if isinstance(e, ToolReturned)][0]
    assert err.result.is_error


async def test_short_term_memory_across_turns():
    model = FakeModel([text_response("Hi Alan!"), text_response("Your name is Alan.")])
    svc, _ = make_services()
    agent = _agent(model, svc)

    await agent.run("My name is Alan.", thread_id="t1")
    await agent.run("What is my name?", thread_id="t1")

    # the second turn's request carries the first turn's transcript (loaded from the store)
    first_req, last_req = model.requests[0], model.requests[-1]
    assert len(last_req.messages) > len(first_req.messages)
    assert any("Alan" in m.text for m in last_req.messages)


async def test_convenience_and_harness_constructors_agree():
    # the convenience form bundles tools into a Harness equivalent to passing one
    from openmate.kernel.agent import Harness

    svc, _ = make_services()
    a = Agent(name="a", model=FakeModel([]), instructions="x", services=svc, tools=[calculator])
    b = Agent(
        name="a",
        model=FakeModel([]),
        instructions="x",
        services=svc,
        harness=Harness(tools=[calculator]),
    )
    assert [t.spec.name for t in a.tools] == [t.spec.name for t in b.tools] == ["calculator"]


async def test_cannot_pass_both_harness_and_tools():
    from openmate.kernel.agent import Harness

    svc, _ = make_services()
    try:
        Agent(
            name="a",
            model=FakeModel([]),
            instructions="x",
            services=svc,
            harness=Harness(tools=[]),
            tools=[calculator],
        )
    except TypeError:
        return
    raise AssertionError("expected TypeError when passing both harness= and tools=")


async def test_stream_yields_events_then_finishes():
    model = FakeModel(
        [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("4")]
    )
    svc, _ = make_services()
    seen = [type(e).__name__ async for e in _agent(model, svc).stream("2+2?")]
    assert seen[0] == "RunStarted"
    assert seen[-1] == "RunFinished"
    assert "ToolReturned" in seen


async def test_determinism_same_script_same_events():
    async def execute():
        model = FakeModel(
            [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("4")]
        )
        svc, events = make_services()
        await _agent(model, svc).run("2+2?", thread_id="fixed")
        return [(type(e).__name__, e.step) for e in events]

    assert await execute() == await execute()
