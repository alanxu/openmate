"""Live tests — exercise the OpenMate agent against a REAL model.

Unlike the FakeModel cassette suite, these call the configured provider (MiniMax
via the Anthropic-compatible API by default — see ``.env`` / ``openmate/config.py``).
They cost money, need the network, and are non-deterministic, so they are:

  * marked ``live`` and SKIPPED by default (see conftest.py). Run them with
        pytest --run-live tests/test_live.py -v
    or  OPENMATE_LIVE_TESTS=1 pytest tests/test_live.py -v
  * self-skipped if no API key is configured;
  * asserted TOLERANTLY — on *tool usage* and *answer correctness* (a number, a
    name, a recovered error), never on exact wording, because real output varies.

Each agent is capped (``max_steps=5``, ``max_tokens=512``) to keep cost small.
"""

from __future__ import annotations

import os

import pytest
from helpers import make_services

from openmate.adapters.tools.builtin import read_only_tools
from openmate.adapters.tools.native import tool
from openmate.agent.assemble import assemble
from openmate.kernel.agent import Agent
from openmate.kernel.events import ToolCallRequested, ToolReturned
from openmate.tools.provider import NativeProvider

pytestmark = pytest.mark.live  # the whole module is opt-in / live


@pytest.fixture(scope="session")
def model():
    """The configured real model, or skip the whole module if no key is set."""
    from openmate.config import default_model, load_env

    load_env()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")):
        pytest.skip("no API key — set ANTHROPIC_API_KEY (or MINIMAX_API_KEY) in .env")
    return default_model()


# --- helpers -----------------------------------------------------------------
_INSTRUCTIONS = "You are a precise assistant. Use tools when asked. Keep answers short."


def _agent(model, svc, tools, **kw) -> Agent:
    return Agent(
        name="live",
        model=model,
        instructions=_INSTRUCTIONS,
        services=svc,
        tools=tools,
        max_steps=5,
        max_tokens=512,
        **kw,
    )


def _called(events) -> list[str]:
    return [e.call.name for e in events if isinstance(e, ToolCallRequested)]


def _norm(text: str) -> str:
    """Normalize for number/word matching (drop commas and spaces, lowercase)."""
    return text.lower().replace(",", "").replace(" ", "")


# --- 1. plain generation: right answer, no inappropriate tool use ------------
async def test_answers_without_misusing_tools(model):
    svc, events = make_services()
    result = await _agent(model, svc, read_only_tools()).run(
        "What is the capital of France? Answer in one word."
    )
    assert result.ok
    assert "paris" in result.text.lower()
    assert "calculator" not in _called(events)  # a geography question isn't math


# --- 2. single tool call: the calculator does the arithmetic -----------------
async def test_calculator_tool_used_and_correct(model):
    svc, events = make_services()
    result = await _agent(model, svc, read_only_tools()).run(
        "You must use the calculator tool to compute 9876 * 5432, then state the result."
    )
    assert result.ok
    assert "calculator" in _called(events)
    assert str(9876 * 5432) in _norm(result.text)


# --- 3. ReAct multi-tool chain: one tool's output feeds the next -------------
async def test_react_chains_two_tools(model):
    @tool(side_effecting=False, idempotent=True)
    def word_count(text: str) -> int:
        """Count the whitespace-separated words in a piece of text."""
        return len(text.split())

    svc, events = make_services()
    result = await _agent(model, svc, [word_count, *read_only_tools()]).run(
        "Use word_count on the sentence 'the quick brown fox jumps over', then use the "
        "calculator to square that count. State only the final number."
    )
    assert result.ok
    called = _called(events)
    assert "word_count" in called and "calculator" in called
    assert "36" in _norm(result.text)  # 6 words -> 36


# --- 4. recoverable tool error: the run survives a failing call --------------
async def test_tool_error_is_recoverable(model):
    @tool
    def divide(a: float, b: float) -> float:
        """Divide a by b."""
        return a / b  # b == 0 raises -> executor returns an is_error result

    svc, events = make_services()
    result = await _agent(model, svc, [divide]).run(
        "Use the divide tool for 10 divided by 0, and also for 10 divided by 2. "
        "Report what happened for each."
    )
    assert result.ok  # the failing call did not crash the run
    returns = [e for e in events if isinstance(e, ToolReturned)]
    assert any(r.result.is_error for r in returns)  # one call errored...
    assert "5" in _norm(result.text)  # ...and it still reported 10 / 2 = 5


# --- 5. short-term memory across turns (same thread_id) ----------------------
async def test_multi_turn_memory(model):
    svc, _ = make_services()
    agent = _agent(model, svc, [])
    await agent.run("Remember: my favorite number is 7. Reply 'ok'.", thread_id="mem")
    result = await agent.run(
        "What is my favorite number? Answer with just the number.", thread_id="mem"
    )
    assert "7" in _norm(result.text)


# --- 6. streaming events from a live run -------------------------------------
async def test_streaming_emits_events(model):
    svc, _ = make_services()
    seen = [
        type(e).__name__ async for e in _agent(model, svc, []).stream("Say hello in one word.")
    ]
    assert seen[0] == "RunStarted"
    assert seen[-1] == "RunFinished"
    assert "MessageAdded" in seen


# --- 7. the assemble() path with a real model -------------------------------
async def test_assemble_with_real_model(model):
    svc, events = make_services()
    async with assemble(
        name="live",
        system="Use tools when asked; be terse.",
        model=model,
        services=svc,
        providers=[NativeProvider(read_only_tools())],
        max_steps=5,
        max_tokens=512,
    ) as agent:
        result = await agent.run("Use the calculator to compute 2 + 2 * 10.")
    assert result.ok
    assert "calculator" in _called(events)
    assert "22" in _norm(result.text)  # precedence: 2 + 20


# --- 8. agentic RAG: ground an answer in a planted fact ----------------------
async def test_rag_grounded_answer(model, tmp_path):
    from rag import (
        DenseRetriever,
        FileLoader,
        FixedWindowChunker,
        HashingEmbedder,
        InMemoryVectorStore,
        NaivePipeline,
        RetrieveTool,
    )

    (tmp_path / "kb.md").write_text(
        "# Zorblax protocol\nThe Zorblax protocol requires exactly 7 handshakes "
        "before a session may open.",
        encoding="utf-8",
    )
    emb, store = HashingEmbedder(dim=512), InMemoryVectorStore()
    await NaivePipeline(FileLoader(), FixedWindowChunker(300, 50), emb, store).ingest(
        str(tmp_path)
    )
    retriever = DenseRetriever(emb, store)

    svc, events = make_services()
    result = await _agent(model, svc, [RetrieveTool(retriever, k=3)]).run(
        "Using the knowledge base via rag_search, how many handshakes does the Zorblax "
        "protocol require? Answer with the number."
    )
    assert "rag_search" in _called(events)  # it actually retrieved
    assert "7" in _norm(result.text)  # grounded in the planted fact
