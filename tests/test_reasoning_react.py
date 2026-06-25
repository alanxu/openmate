"""ReAct reasoning — 5 cases (R1–R5), run against the REAL model.

Live tests: marked ``live`` (skipped unless ``--run-live`` / ``OPENMATE_LIVE_TESTS``),
self-skip without an API key, and assert TOLERANTLY — on tool usage (captured from
the event bus) and answer correctness — never on exact wording. The model itself
does the ReAct reasoning; we only check the observable outcome.
"""

from __future__ import annotations

import pytest
from helpers import make_services

from openmate.adapters.tools.builtin import calculator, list_directory, read_file
from openmate.adapters.tools.native import tool
from openmate.kernel.agent import Agent
from openmate.kernel.events import ToolCallRequested, ToolReturned

pytestmark = pytest.mark.live

_SYS = "You are a precise assistant. Use the provided tools when the task calls for them. Keep answers short."


def _agent(model, svc, tools):
    return Agent(name="react", model=model, instructions=_SYS, services=svc,
                 tools=tools, max_steps=8, max_tokens=512)


def _called(events):
    return [e.call.name for e in events if isinstance(e, ToolCallRequested)]


def _norm(text):
    return text.lower().replace(",", "").replace(" ", "")


# --- R1: chain two tools — one tool's output feeds the next ------------------
async def test_r1_tool_output_feeds_next_tool(live_model):
    @tool(side_effecting=False, idempotent=True)
    def word_count(text: str) -> int:
        """Count whitespace-separated words in a piece of text."""
        return len(text.split())

    svc, events = make_services()
    result = await _agent(live_model, svc, [word_count, calculator]).run(
        "Use word_count on the sentence 'the quick brown fox jumps over', then use the "
        "calculator to square that count. State only the final number."
    )
    assert result.ok
    called = _called(events)
    assert "word_count" in called and "calculator" in called
    assert "36" in _norm(result.text)  # 6 words -> 36


# --- R2: observe a tool error and correct within the loop -------------------
async def test_r2_recovers_from_tool_error(live_model):
    @tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        return a / b  # b == 0 raises -> executor returns an is_error result

    svc, events = make_services()
    result = await _agent(live_model, svc, [divide]).run(
        "Use the divide tool for 10 divided by 0, and also for 10 divided by 2. "
        "Report what happened for each."
    )
    assert result.ok
    returns = [e for e in events if isinstance(e, ToolReturned)]
    assert any(r.result.is_error for r in returns)  # the 10/0 call errored
    assert "5" in _norm(result.text)  # ...and it still reported 10/2 = 5


# --- R3: observation-driven search across files -----------------------------
async def test_r3_searches_files_until_match(live_model):
    svc, events = make_services()
    result = await _agent(live_model, svc, [list_directory, read_file]).run(
        "Look in the 'docs' directory and find which markdown file mentions "
        "'NoProgress'. Read files as needed, then name the file."
    )
    assert result.ok
    assert "read_file" in _called(events)  # it actually read files to find out
    assert "02" in result.text or "agent-loop" in result.text.lower()


# --- R4: agentic retrieval — retrieve, then ground the answer ----------------
async def test_r4_agentic_retrieval_grounds_answer(live_model):
    from rag import DenseRetriever, HashingEmbedder, InMemoryVectorStore, RetrieveTool
    from openmate.ports.retriever import VectorRecord

    emb = HashingEmbedder(dim=512)
    store = InMemoryVectorStore()
    fact = "The loop guard is the NoProgress stop policy: it halts when tool calls repeat or oscillate."
    vec = (await emb.embed([fact]))[0]
    await store.upsert([VectorRecord("kb1", vec, fact, {"source": "loop.md"})])

    svc, events = make_services()
    result = await _agent(live_model, svc, [RetrieveTool(DenseRetriever(emb, store), k=3)]).run(
        "Using rag_search on the knowledge base, what is the loop guard? Answer from the sources."
    )
    assert "rag_search" in _called(events)
    assert "noprogress" in result.text.lower()  # grounded in the retrieved fact


# --- R5: action conditioned on a runtime observation ------------------------
@pytest.mark.parametrize("hour, expect_listing", [(14, True), (22, False)])
async def test_r5_conditional_action_on_observation(live_model, hour, expect_listing):
    @tool
    def clock() -> int:
        """Return the current hour of day (0-23, UTC)."""
        return hour

    svc, events = make_services()
    result = await _agent(live_model, svc, [clock, list_directory]).run(
        "First check the current hour with the clock tool. If it is between 9 and 17 "
        "inclusive, list the 'skills' directory. Otherwise just reply 'after hours'."
    )
    assert result.ok
    assert "clock" in _called(events)  # it consulted the observation first
    listed = "list_directory" in _called(events)
    assert listed is expect_listing  # the branch taken depends on the observed hour
