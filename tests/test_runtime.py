"""Runtime: the ReAct loop, the step cap, short-term memory, and determinism."""

from __future__ import annotations

from helpers import make_services

from openmate.adapters.models.fake import FakeModel, text_response, tool_call_response
from openmate.adapters.tools.builtin import calculator
from openmate.kernel.events import RunFinished, RunStarted, ToolReturned
from openmate.kernel.runtime import Runtime
from openmate.kernel.types import Agent


def _agent(model, **kw) -> Agent:
    return Agent(name="a", model=model, instructions="be helpful", tools=[calculator], **kw)


async def test_tool_call_then_final_answer():
    model = FakeModel(
        [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("It is 4.")]
    )
    svc, events = make_services()
    result = await Runtime(svc).run(_agent(model), "what is 2+2?")

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
    result = await Runtime(svc).run(_agent(FakeModel(script), max_steps=3), "loop forever")

    assert result.reason == "max_steps"
    assert result.steps == 3


async def test_bad_tool_args_are_recoverable():
    # model calls calculator with a missing arg, sees the error, then answers
    model = FakeModel(
        [tool_call_response("c1", "calculator", {}), text_response("Sorry, fixed it.")]
    )
    svc, events = make_services()
    result = await Runtime(svc).run(_agent(model), "do math")

    assert result.ok  # the run did not crash
    err = [e for e in events if isinstance(e, ToolReturned)][0]
    assert err.result.is_error


async def test_short_term_memory_across_turns():
    model = FakeModel([text_response("Hi Alan!"), text_response("Your name is Alan.")])
    svc, _ = make_services()
    runtime = Runtime(svc)

    await runtime.run(_agent(model), "My name is Alan.", thread_id="t1")
    await runtime.run(_agent(model), "What is my name?", thread_id="t1")

    # the second turn's request carries the first turn's transcript (loaded from the store)
    first_req, last_req = model.requests[0], model.requests[-1]
    assert len(last_req.messages) > len(first_req.messages)
    assert any("Alan" in m.text for m in last_req.messages)


async def test_determinism_same_script_same_events():
    def run_once():
        model = FakeModel(
            [tool_call_response("c1", "calculator", {"expression": "2+2"}), text_response("4")]
        )
        svc, events = make_services()
        return events, model, svc

    async def execute(triple):
        events, model, svc = triple
        await Runtime(svc).run(_agent(model), "2+2?", thread_id="fixed")
        return [(type(e).__name__, e.step) for e in events]

    a = await execute(run_once())
    b = await execute(run_once())
    assert a == b
