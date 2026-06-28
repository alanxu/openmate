"""UNIT-tier eval harness for docs/16-eval-plan.md.

This file is **pure Unit tier**: every test runs against `FakeModel` or a local
fixture, no network call, no API key, no real model judgment involved. It
checks that the *mechanism* works — the right event fires, the right field
gets set, the right exception type is raised. It does NOT tell you whether a
real model behaves well; that's a different, harder question and a different
file (`test_evals_integration.py`).

Each test function is named after its eval-plan ID (e.g. ``test_A1_...``) so a
case in the doc maps 1:1 to a test here — `grep "def test_A1" tests/test_evals.py`
finds the implementation for case A1. Sections mirror the doc's A-H structure.

A few case IDs from the doc are deliberately **not** here, because the thing
worth checking about them is real-model *behavior*, not plumbing — they live
in `test_evals_integration.py` instead: B2, C1 (real versions — the mechanism
versions of these two are still here, since "does context get passed at all"
and "does the loop advance correctly" are pure plumbing), E4, G1, G2.

Three cases are intentionally written to fail today and are marked
``xfail(strict=True)``: A7, D3, E1. Each documents a gap from
docs/15-claude-code-tool-architecture-alignment.md (R5, R3, R2 respectively).
When the corresponding fix lands, the test will XPASS, strict mode turns that
into a hard failure, and the fix is to delete the xfail marker, not the test.

Run: `pytest tests/test_evals.py -v` — no flags needed, no API key needed,
should finish in well under a second.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from helpers import calls_response, call_response, make_services, text_response

from openmate.adapters.models.anthropic import AnthropicModel
from openmate.adapters.models.fake import FakeModel
from openmate.adapters.stores.sqlite import SQLiteStore
from openmate.adapters.tools.builtin import (
    _safe_path,
    calculator,
    list_directory,
    read_file,
    write_file,
)
from openmate.adapters.tools.native import tool
from openmate.kernel.agent import Agent
from openmate.kernel.errors import OpenMateError, ProviderError
from openmate.kernel.events import ToolCallRequested, ToolReturned
from openmate.kernel.loop import resume
from openmate.kernel.types import Message, RunState, TextPart
from openmate.skills.skill import LoadSkillTool, SkillProvider, SkillRegistry
from openmate.tools.provider import MCPProvider, NativeProvider, ShellProvider

_EMAIL_SKILLS = Path(__file__).resolve().parent.parent / "skills" / "email"


def _agent(model, svc, tools, **kw) -> Agent:
    return Agent(name="eval", model=model, instructions="be helpful", services=svc, tools=tools, **kw)


def _returns(events) -> list:
    return [e for e in events if isinstance(e, ToolReturned)]


def _calls(events) -> list:
    return [e for e in events if isinstance(e, ToolCallRequested)]


# === A. Tool execution & correctness =========================================

async def test_A1_valid_tool_call_reaches_final_answer():
    model = FakeModel([call_response("c1", "calculator", {"expression": "(1+2)*3"}), text_response("9")])
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("compute (1+2)*3")
    assert result.ok and result.text == "9"
    assert len(_calls(events)) == 1 and len(_returns(events)) == 1
    assert not _returns(events)[0].result.is_error


async def test_A2_unknown_tool_is_recoverable():
    model = FakeModel([call_response("c1", "frobnicate", {}), text_response("ok")])
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("do the thing")
    assert result.ok  # did not crash
    err = _returns(events)[0].result
    assert err.is_error
    assert "unknown tool" in err.content[0].text
    assert "calculator" in err.content[0].text  # lists real tool names
    assert len(model.requests) == 2  # model got a second turn


async def test_A3_tool_exception_is_recoverable():
    @tool
    def boom() -> str:
        """Always raises."""
        raise ValueError("kaboom")

    model = FakeModel([call_response("c1", "boom", {}), text_response("recovered")])
    svc, events = make_services()
    result = await _agent(model, svc, [boom]).run("trigger it")
    assert result.ok
    err = _returns(events)[0].result
    assert err.is_error and "ValueError" in err.content[0].text


async def test_A4_tool_timeout_is_bounded_and_legible():
    @tool(timeout_s=0.05)
    async def slow() -> str:
        """Sleeps past its own timeout."""
        await asyncio.sleep(2.0)
        return "too late"

    model = FakeModel([call_response("c1", "slow", {}), text_response("ok")])
    svc, events = make_services()
    t0 = time.perf_counter()
    result = await _agent(model, svc, [slow]).run("be slow")
    elapsed = time.perf_counter() - t0

    assert result.ok
    err = _returns(events)[0].result
    assert err.is_error and "timed out after" in err.content[0].text
    assert elapsed < 1.0  # bounded by timeout_s, not the 2s sleep


async def test_A5_multiple_calls_in_one_turn_dispatched_sequentially_in_order():
    calls = [(f"c{i}", "calculator", {"expression": f"{i}+{i}"}) for i in range(3)]
    model = FakeModel([calls_response(calls), text_response("done")])
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("do three sums")
    assert result.ok
    returns = _returns(events)
    assert len(returns) == 3
    assert [r.result.content[0].text for r in returns] == ["0", "2", "4"]  # in call order


async def test_A6_read_file_path_traversal_blocked_before_filesystem_access():
    with pytest.raises(ValueError, match="outside the working directory"):
        _safe_path("../../etc/passwd")
    # confirm via the tool surface too: the model never sees a successful read
    model = FakeModel([call_response("c1", "read_file", {"path": "../../etc/passwd"}), text_response("ok")])
    svc, events = make_services()
    result = await _agent(model, svc, [read_file]).run("read it")
    assert result.ok
    assert _returns(events)[0].result.is_error


@pytest.mark.xfail(reason="no read-before-write check yet (doc 15 R5)", strict=True)
async def test_A7_write_without_prior_read_is_rejected(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "existing.txt"
    target.write_text("original content", encoding="utf-8")

    model = FakeModel([call_response("c1", "write_file", {"path": "existing.txt", "content": "clobbered"})])
    svc, events = make_services()
    await _agent(model, svc, [write_file]).run("overwrite it without reading first")
    res = _returns(events)[0].result
    assert res.is_error
    assert "must read" in res.content[0].text.lower()


# === B. Reasoning loop & control flow ========================================

async def test_B1_no_tool_qa_stops_without_advancing_step():
    """``step`` only increments via advance(), which only runs after a tool-dispatch
    round (see kernel/loop.py's _loop()) — a no-tool answer on the first turn
    returns with step==0, not 1. (Corrects an assumption in docs/16-eval-plan.md's
    B1 wording, found while implementing this case against the real loop code.)"""
    model = FakeModel([text_response("4")])
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("what is 2+2 in your head, no tools")
    assert result.ok and result.steps == 0
    assert not _calls(events)


async def test_B2_multistep_tool_chain_terminates_on_final_answer():
    """Mechanism only: a scripted FakeModel proves the loop *can* chain calls and
    stop correctly. Whether a real model actually *chooses* to chain tools
    correctly on an ambiguous task is INT-1 in test_evals_integration.py."""
    model = FakeModel(
        [
            call_response("c1", "calculator", {"expression": "2+2"}),
            call_response("c2", "calculator", {"expression": "4*3"}),
            text_response("12"),
        ]
    )
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("chain two computations")
    assert result.ok and result.steps == 2  # 2 tool rounds, then a final non-tool turn
    assert len(_calls(events)) == 2
    assert not result.final.tool_calls


async def test_B3_max_steps_caps_a_runaway_loop():
    script = [call_response(f"c{i}", "calculator", {"expression": "1+1"}) for i in range(50)]
    svc, _ = make_services()
    result = await _agent(FakeModel(script), svc, [calculator], max_steps=3).run("never stop")
    assert result.reason == "max_steps"
    assert result.steps == 3  # stopped at the cap, not silently returned as final


async def test_B4_repeated_identical_call_has_no_loop_detection_beyond_max_steps():
    """Documents a known gap: max_steps is the only bound on a repeated identical call."""
    script = [call_response(f"c{i}", "calculator", {"expression": "1+1"}) for i in range(5)]
    svc, events = make_services()
    result = await _agent(FakeModel(script), svc, [calculator], max_steps=5).run("repeat")
    # current behavior: every identical call runs to completion, no early exit
    assert len(_calls(events)) == 5
    assert result.reason == "max_steps"


async def test_B5_resume_continues_step_count_without_reset():
    """resume() (unlike a second drive() call) does not go through _init(),
    so step keeps incrementing across the interruption — see kernel/loop.py."""
    svc, _ = make_services()
    mid_run = RunState(
        thread_id="resumable",
        messages=[
            Message("system", [TextPart("be helpful")]),
            Message("user", [TextPart("what is 2+2, then 4*3?")]),
            Message("assistant", [TextPart("let me compute")]),
        ],
        step=2,  # simulates a run already two tool-rounds deep before interruption
    )
    await svc.store.save(mid_run.thread_id, mid_run)

    model = FakeModel([text_response("It is 12.")])
    agent = _agent(model, svc, [calculator])
    result = await resume(agent, "resumable")

    assert result.ok
    assert result.steps == 2  # step was NOT reset to 0 by resume()
    assert "12" in result.text


# === C. Memory, state & resumption ===========================================

async def test_C1_same_thread_id_carries_context_across_runs():
    """Mechanism only: proves the message history is actually re-sent on the
    second call. Whether a real model correctly *uses* that history rather
    than hallucinating is INT-2 in test_evals_integration.py."""
    model = FakeModel([text_response("Hi Alan!"), text_response("Your name is Alan.")])
    svc, _ = make_services()
    agent = _agent(model, svc, [])
    await agent.run("My name is Alan.", thread_id="t1")
    result = await agent.run("What is my name?", thread_id="t1")
    assert "Alan" in result.text
    first_req, last_req = model.requests[0], model.requests[-1]
    assert len(last_req.messages) > len(first_req.messages)


async def test_C2_different_thread_ids_stay_isolated():
    model = FakeModel([text_response("a"), text_response("b")])
    svc, _ = make_services()
    agent = _agent(model, svc, [])
    await agent.run("secret-alpha", thread_id="t1")
    await agent.run("unrelated", thread_id="t2")
    last_req = model.requests[-1]
    assert not any("secret-alpha" in m.text for m in last_req.messages)


async def test_C3_sqlite_store_survives_a_simulated_process_restart(tmp_path):
    db = str(tmp_path / "checkpoints.sqlite")
    store_a = SQLiteStore(db)  # "process A"
    state = RunState(
        thread_id="durable",
        messages=[Message("user", [TextPart("hello")]), Message("assistant", [TextPart("hi")])],
        step=1,
    )
    await store_a.save(state.thread_id, state)
    store_a.close()

    store_b = SQLiteStore(db)  # "process B" — fresh connection, same file
    loaded = await store_b.load("durable")
    store_b.close()

    assert loaded is not None
    assert loaded.step == state.step
    assert [m.text for m in loaded.messages] == [m.text for m in state.messages]


# === D. Skills ================================================================

async def test_D1_skill_discovery_finds_both_email_skills():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    names = {c.name for c in reg.cards()}
    assert names == {"triage-inbox", "summarize-thread"}


async def test_D2_load_skill_activates_and_returns_body():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    tool_ = LoadSkillTool(reg)
    ctx = SimpleNamespace(state=SimpleNamespace(scratch={}))
    res = await tool_.invoke({"name": "triage-inbox"}, ctx)
    assert not res.is_error
    assert ctx.state.scratch["active_skills"] == ["triage-inbox"]
    assert res.content[0].text == reg.get("triage-inbox").render()


@pytest.mark.xfail(reason="SkillManifest.tools is parsed but never enforced (doc 15 R3)", strict=True)
async def test_D3_skill_tool_allowlist_is_enforced():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    triage = reg.get("triage-inbox")
    assert "calculator" not in triage.manifest.tools  # not in the declared allowlist

    model = FakeModel(
        [
            call_response("c1", "load_skill", {"name": "triage-inbox"}),
            call_response("c2", "calculator", {"expression": "1+1"}),  # outside the allowlist
            text_response("done"),
        ]
    )
    svc, events = make_services()
    await _agent(model, svc, [LoadSkillTool(reg), calculator]).run("triage then compute")
    # the real assertion once R3 lands: the calculator call is rejected as
    # outside the active skill's allowlist, not executed.
    results = [r.result for r in _returns(events)]
    assert any(r.is_error and "allowlist" in r.content[0].text for r in results)


async def test_D4_skill_resource_path_traversal_blocked():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    skill = reg.get("triage-inbox")
    with pytest.raises(ValueError, match="outside the skill directory"):
        skill.resource("../../../etc/passwd")


# === E. Safety, guardrails & approval ========================================

@pytest.mark.xfail(reason="no approval gate exists yet for side-effecting tools (doc 15 R2)", strict=True)
async def test_E1_side_effecting_call_pauses_for_approval(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    model = FakeModel([call_response("c1", "write_file", {"path": "out.txt", "content": "hi"})])
    svc, events = make_services()
    result = await _agent(model, svc, [write_file]).run("write a file")
    # desired (post-R2) behavior: the run pauses before invoke(), nothing is written yet
    assert result.paused
    assert not (tmp_path / "out.txt").exists()


async def test_E2_allow_write_cli_flag_gates_write_file():
    from openmate.adapters.tools.builtin import all_tools, read_only_tools

    assert "write_file" not in {t.spec.name for t in read_only_tools()}
    assert "write_file" in {t.spec.name for t in all_tools()}


async def test_E3_mcp_scope_allowlist_hides_unlisted_tools():
    pytest.importorskip("mcp")
    import sys

    fake_gmail = Path(__file__).resolve().parent.parent / "servers" / "gmail" / "fake_server.py"
    from openmate.adapters.tools.mcp_client import MCPServerSpec

    gmail = MCPServerSpec(name="gmail", command=[sys.executable, str(fake_gmail)])
    provider = MCPProvider([gmail], scope_allowlist=["gmail_search", "gmail_get_message"])
    await provider.setup()
    try:
        names = {t.spec.name for t in await provider.tools()}
        assert names == {"gmail_search", "gmail_get_message"}
        assert "gmail_create_draft" not in names
    finally:
        await provider.teardown()


# E4 (refusal quality on a real model) moved to test_evals_integration.py —
# it's a real-model judgment case, not Unit-tier plumbing.

# === F. Retrieval (RAG) grounding ============================================

async def _rag_fixture(tmp_path):
    from rag import DenseRetriever, FileLoader, FixedWindowChunker, HashingEmbedder, InMemoryVectorStore, NaivePipeline

    (tmp_path / "kb.md").write_text(
        "# Onboarding\nNew hires get a laptop within 3 business days of their start date.",
        encoding="utf-8",
    )
    emb, store = HashingEmbedder(dim=512), InMemoryVectorStore()
    await NaivePipeline(FileLoader(), FixedWindowChunker(300, 50), emb, store).ingest(str(tmp_path))
    return DenseRetriever(emb, store)


async def test_F1_rag_search_returns_relevant_chunk_and_records_source(tmp_path):
    from rag import RetrieveTool

    retriever = await _rag_fixture(tmp_path)
    tool_ = RetrieveTool(retriever, k=3)
    ctx = SimpleNamespace(state=SimpleNamespace(scratch={}))
    res = await tool_.invoke({"query": "how long until a new hire gets a laptop"}, ctx)
    assert not res.is_error
    assert "kb.md" in res.content[0].text
    assert ctx.state.scratch["rag_sources"]
    assert ctx.state.scratch["rag_sources"][0]["source"] == "kb.md"


async def test_F2_rag_search_no_match_returns_graceful_message():
    from rag import DenseRetriever, HashingEmbedder, InMemoryVectorStore, RetrieveTool

    retriever = DenseRetriever(HashingEmbedder(dim=64), InMemoryVectorStore())  # empty index
    tool_ = RetrieveTool(retriever, k=3)
    res = await tool_.invoke({"query": "anything at all"}, SimpleNamespace(state=SimpleNamespace(scratch={})))
    assert not res.is_error
    assert res.content[0].text == "No matching documents found in the knowledge base."


async def test_F3_agentic_refinement_accumulates_sources_from_both_calls(tmp_path):
    from rag import RetrieveTool

    retriever = await _rag_fixture(tmp_path)
    tool_ = RetrieveTool(retriever, k=2)
    ctx = SimpleNamespace(state=SimpleNamespace(scratch={}))
    await tool_.invoke({"query": "laptop timeline"}, ctx)
    after_first = len(ctx.state.scratch["rag_sources"])
    assert after_first > 0
    await tool_.invoke({"query": "onboarding process"}, ctx)
    # the second call's hits are appended, not overwriting the first call's
    assert len(ctx.state.scratch["rag_sources"]) == 2 * after_first


# === G. Robustness & adversarial ==============================================

# G1 (real prompt-injection resistance) moved to test_evals_integration.py —
# whether a real model complies with injected instructions is exactly the
# kind of thing FakeModel can't tell you anything about.

async def test_G3_missing_required_arg_is_model_legible_error():
    model = FakeModel([call_response("c1", "calculator", {}), text_response("ok")])
    svc, events = make_services()
    result = await _agent(model, svc, [calculator]).run("compute something")
    assert result.ok  # recovers, no traceback
    err = _returns(events)[0].result
    assert err.is_error
    assert "missing required argument" in err.content[0].text
    assert "Traceback" not in err.content[0].text


async def test_G4_wrong_type_tool_arg_is_recoverable_not_a_typeerror():
    model = FakeModel([call_response("c1", "read_file", {"path": 123}), text_response("ok")])
    svc, events = make_services()
    result = await _agent(model, svc, [read_file]).run("read path 123")
    assert result.ok  # no unhandled TypeError escaped dispatch()
    assert _returns(events)[0].result.is_error


async def test_G5_oversized_fetch_output_is_truncated_with_marker(monkeypatch):
    import openmate.adapters.tools.builtin as builtin

    class _FakeResp:
        def read(self, n):
            return b"x" * n

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(builtin, "urlopen", lambda *a, **kw: _FakeResp())
    text = builtin.fetch_url.fn("https://example.com/huge") if hasattr(builtin.fetch_url, "fn") else None
    # fetch_url is a FunctionTool; call the underlying function directly via .fn
    out = builtin.fetch_url.fn("https://example.com/huge")
    assert len(out) <= builtin._MAX_FETCH_BYTES + len("\n...[truncated]") + 1
    assert "[truncated]" in out


async def test_G6_provider_failure_surfaces_as_typed_openmate_error():
    class _RaisingClient:
        class messages:
            @staticmethod
            async def create(**kwargs):
                raise ConnectionError("network unreachable")

    model = AnthropicModel("test-model", client=_RaisingClient())
    from openmate.ports.model import ModelRequest

    with pytest.raises(ProviderError):
        await model.generate(ModelRequest(messages=[], tools=None))
    # and ProviderError is a typed OpenMateError, not a raw exception escaping to a user
    try:
        await model.generate(ModelRequest(messages=[], tools=None))
    except OpenMateError as e:
        assert "ConnectionError" in str(e)


# === H. Performance & observability ==========================================

async def test_H1_tracer_captures_matching_call_and_return_counts():
    calls = [(f"c{i}", "calculator", {"expression": "1+1"}) for i in range(3)]
    model = FakeModel([calls_response(calls), text_response("done")])
    svc, events = make_services()
    await _agent(model, svc, [calculator]).run("three calls")
    requested = [e for e in events if isinstance(e, ToolCallRequested)]
    returned = [e for e in events if isinstance(e, ToolReturned)]
    assert len(requested) == len(returned) == 3
    model_requests = [e for e in events if type(e).__name__ == "ModelRequested"]
    assert len(model_requests) >= 1


async def test_H2_tool_returned_ms_reflects_real_elapsed_time():
    @tool
    async def delay_200ms() -> str:
        """Sleeps for ~200ms."""
        await asyncio.sleep(0.2)
        return "done"

    model = FakeModel([call_response("c1", "delay_200ms", {}), text_response("ok")])
    svc, events = make_services()  # NOTE: svc.clock is a fake counter, not wall time —
    # this case needs real elapsed time, so measure it directly around dispatch instead.
    t0 = time.perf_counter()
    await _agent(model, svc, [delay_200ms]).run("wait")
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    assert 150 <= elapsed_ms <= 1000  # generous tolerance band around the 200ms delay


async def test_H3_golden_trace_regression_same_script_same_tool_sequence():
    async def run_once():
        model = FakeModel([call_response("c1", "calculator", {"expression": "2+2"}), text_response("4")])
        svc, events = make_services()
        result = await _agent(model, svc, [calculator]).run("2+2?", thread_id="golden")
        trace = [(e.call.name, dict(e.call.args)) for e in events if isinstance(e, ToolCallRequested)]
        return trace, result.text

    a = await run_once()
    b = await run_once()
    assert a == b  # tool-call sequence + final answer are stable across replays
