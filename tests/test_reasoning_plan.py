"""Plan-and-Execute reasoning — 5 cases (P1–P5), run against the REAL model.

The model is given the soft planning tools (``create_plan``/``update_plan``) and a
directive system prompt; it decides the steps, executes them, and updates status.
Live + tolerant: we assert the model planned (a plan landed in state) and produced
the right outcome, not exact steps.
"""

from __future__ import annotations

import pytest
from helpers import make_services

from openmate.adapters.tools.builtin import calculator, list_directory, read_file
from openmate.adapters.tools.native import tool
from openmate.kernel.agent import Agent
from openmate.kernel.events import ToolCallRequested, ToolReturned
from openmate.strategies.planning_tools import get_plan, planning_tools

pytestmark = pytest.mark.live

_SYS = (
    "You are a planning agent. Your FIRST action for EVERY task MUST be to call "
    "create_plan with the list of steps (each step has a goal; add deps when a step "
    "needs an earlier step's result). Do NOT call any other tool before create_plan. "
    "After planning, execute the steps with the other tools, calling update_plan to "
    "mark each step done as you finish it. When all steps are done, give a short answer."
)


def _agent(model, svc, tools):
    return Agent(name="plan", model=model, instructions=_SYS, services=svc,
                 tools=[*planning_tools(), *tools], max_steps=20, max_tokens=600)


def _called(events):
    return [e.call.name for e in events if isinstance(e, ToolCallRequested)]


def _norm(text):
    return text.lower().replace(",", "").replace(" ", "")


# --- P1: decompose, then execute in order -----------------------------------
async def test_p1_decompose_then_execute(live_model, tmp_path):
    out = tmp_path / "index.md"

    @tool
    def write_index(content: str) -> str:
        """Write the combined index file."""
        out.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes"

    svc, events = make_services()
    result = await _agent(live_model, svc, [list_directory, read_file, write_index]).run(
        "Start by calling create_plan to lay out the steps. Then create an index of the "
        "skills in the 'skills/email' directory: list them, and write a short index file "
        "using the write_index tool."
    )
    assert result.ok
    assert "create_plan" in _called(events)  # it planned before acting
    plan = get_plan(result.state)
    assert plan is not None and len(plan.steps) >= 2
    assert out.exists()  # the plan was actually executed to completion


# --- P2: dependency DAG — divide depends on two sums ------------------------
async def test_p2_dependency_dag(live_model):
    data = {"a": [1, 2, 3, 4], "b": [2, 3]}  # sum_a=10, sum_b=5 → 10/5 = 2

    @tool
    def file_sum(name: str) -> int:
        """Sum the numbers in the named file ('a' or 'b')."""
        return sum(data[name])

    svc, events = make_services()
    result = await _agent(live_model, svc, [file_sum, calculator]).run(
        "Compute the sum of file 'a' and the sum of file 'b' (use file_sum), then divide "
        "the sum of a by the sum of b (use calculator). Report the ratio."
    )
    assert result.ok
    assert "create_plan" in _called(events)
    assert _norm(result.text).find("2") != -1  # 10 / 5 = 2
    plan = get_plan(result.state)
    assert plan is not None and len(plan.steps) >= 3


# --- P3: replan / recover when a step diverges ------------------------------
async def test_p3_recovers_from_failed_step(live_model):
    @tool
    def fetch_record(name: str) -> str:
        """Fetch a record by name. The 'primary' record is currently missing."""
        if name == "primary":
            raise FileNotFoundError("primary record is unavailable")
        return f"record:{name}"

    svc, events = make_services()
    result = await _agent(live_model, svc, [fetch_record]).run(
        "Fetch the 'primary' record and report it. If it is unavailable, fetch the "
        "'fallback' record instead and report that."
    )
    assert result.ok
    returns = [e for e in events if isinstance(e, ToolReturned)]
    assert any(r.result.is_error for r in returns)  # the primary fetch failed
    assert "fallback" in result.text.lower()  # and the agent recovered


# --- P4: gather independent facts, then synthesize --------------------------
async def test_p4_gather_then_synthesize(live_model):
    from rag import DenseRetriever, HashingEmbedder, InMemoryVectorStore, RetrieveTool
    from openmate.ports.retriever import VectorRecord

    emb = HashingEmbedder(dim=512)
    store = InMemoryVectorStore()
    for cid, text in [
        ("f1", "alpha apple: the first fact is about apples."),
        ("f2", "beta banana: the second fact is about bananas."),
        ("f3", "gamma grape: the third fact is about grapes."),
    ]:
        vec = (await emb.embed([text]))[0]
        await store.upsert([VectorRecord(cid, vec, text, {"source": cid})])

    svc, events = make_services()
    result = await _agent(live_model, svc, [RetrieveTool(DenseRetriever(emb, store), k=1)]).run(
        "Use rag_search to look up three facts — one about 'alpha', one about 'beta', one "
        "about 'gamma' — then synthesize them into one sentence."
    )
    assert result.ok
    assert _called(events).count("rag_search") >= 3  # gathered all three
    assert "create_plan" in _called(events)


# --- P5: plan once up front, then execute (plan-once-execute-many) -----------
async def test_p5_plans_once_then_executes(live_model):
    svc, events = make_services()
    result = await _agent(live_model, svc, [calculator]).run(
        "Do two separate calculations and report both: (a) 12 * 9, and (b) 144 / 12. "
        "Plan the two steps first, then execute them."
    )
    assert result.ok
    called = _called(events)
    assert called.count("create_plan") == 1  # planned exactly once, up front
    assert called.count("calculator") >= 2  # then executed the steps
    assert "108" in _norm(result.text) and "12" in _norm(result.text)
