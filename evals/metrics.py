"""Live eval harness — the reusable machinery the integration suite runs on.

Separated from the offline unit tests under ``tests/`` so the real-model eval
tier is self-contained: metric functions (Wilson CI, tool-use P/R/F1,
LLM-as-judge), the per-trial reporter, the independent script-runner verifier,
the per-case agent builder, and a deterministic ``Services`` factory.

The ``--run-live`` flag, the ``live`` marker, and the ``live_model`` fixture
live in the repo-root ``conftest.py`` (shared with ``tests/``); everything that
is plain importable code lives here.

``make_services`` mirrors ``tests/helpers.py``'s version on purpose — the two
tiers are independent suites and neither should import the other.
"""

from __future__ import annotations

import os
import re
import subprocess

from openmate.adapters.stores.memory import InMemoryStore
from openmate.kernel.agent import Agent
from openmate.kernel.events import Event, EventBus, ToolCallRequested
from openmate.kernel.types import Services
from openmate.ports.tracer import NullTracer


def make_services(verbose: bool | None = None) -> tuple[Services, list[Event]]:
    """A deterministic ``Services`` (counter clock + ids) plus a captured event list.

    By default the tracer is silent (evals measure outcomes, they don't narrate).
    Set ``verbose=True`` — or the ``OPENMATE_EVAL_VERBOSE`` env var, or pytest's
    ``--eval-verbose`` flag — to attach a ConsoleTracer that prints every event
    live: each model turn, the agent's text, tool calls + args, tool results, and
    the final usage. Requires pytest ``-s`` so the output isn't captured.
    """
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(events.append)

    tick = {"n": 0}

    def clock() -> float:
        tick["n"] += 1
        return float(tick["n"])

    seq = {"n": 0}

    def new_id() -> str:
        seq["n"] += 1
        return f"id{seq['n']}"

    if verbose is None:
        verbose = bool(os.environ.get("OPENMATE_EVAL_VERBOSE"))
    if verbose:
        from openmate.adapters.tracers.console import ConsoleTracer

        # verbose renders the full ModelRequested/ModelResponded payloads.
        tracer = ConsoleTracer(verbose=True).attach(bus)
    else:
        tracer = NullTracer()

    svc = Services(store=InMemoryStore(), tracer=tracer, bus=bus, clock=clock, new_id=new_id)
    return svc, events


def _agent(model, svc, tools, **kw) -> Agent:
    return Agent(
        name="integration-eval",
        model=model,
        instructions=kw.pop("instructions", "You are a precise, careful general-purpose assistant. Use tools when they help."),
        services=svc,
        tools=tools,
        max_steps=kw.pop("max_steps", 6),
        max_tokens=kw.pop("max_tokens", 700),
        **kw,
    )


def _tool_calls(events) -> list:
    return [e.call for e in events if isinstance(e, ToolCallRequested)]


def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion estimated from k/n trials.

    The plain normal-approximation interval breaks down exactly where these
    evals live — small N (3-5), and rates that often sit at or near 0% or
    100% (safety cases). Wilson stays inside [0, 1] in those cases and is
    the standard correction for small-sample binomial proportions; reporting
    it (not just the point estimate) is what makes "4/5 = 80%" legible as
    "could plausibly be anywhere from ~38% to ~96%" rather than a precise
    number it isn't."""
    if n == 0:
        return (0.0, 0.0)
    phat = k / n
    denom = 1 + z**2 / n
    center = phat + z**2 / (2 * n)
    adj = z * ((phat * (1 - phat) / n + z**2 / (4 * n**2)) ** 0.5)
    return (max(0.0, (center - adj) / denom), min(1.0, (center + adj) / denom))


def _report(case_id: str, label: str, trials: list[dict], *, metric_name: str = "success rate") -> float:
    n_pass = sum(1 for t in trials if t["pass"])
    n = len(trials)
    print(f"\n--- {case_id}: {label} ---")
    for i, t in enumerate(trials):
        mark = "PASS" if t["pass"] else "FAIL"
        print(f"  trial {i + 1}/{n} [{mark}] {t.get('note', '')}")
    rate = n_pass / n
    lo, hi = _wilson_ci(n_pass, n)
    print(f"  {metric_name}: {n_pass}/{n} = {rate:.0%}  (95% Wilson CI: [{lo:.0%}, {hi:.0%}])")
    return rate


def _prf1(calls: list, required_tool_names: set[str]) -> tuple[float, float, float]:
    """Tool-use precision/recall/F1 against a minimal required tool set —
    the set-overlap scoring used by agentic tool-use benchmarks (e.g.
    ToolBench/AgentBench), in place of a boolean "did it call a tool."
    Precision penalizes redundant/wasted calls; recall penalizes skipped
    required steps. Simplification, stated plainly: this scores tool
    *names* used, not whether each call's arguments were also correct — a
    full action-trace match would need a per-case reference trajectory,
    which is future work, not something to fake with a thin name check."""
    names = [c.name for c in calls]
    if not names:
        return (0.0, 0.0, 0.0)
    correct = sum(1 for n in names if n in required_tool_names)
    precision = correct / len(names)
    used_required_types = {n for n in names if n in required_tool_names}
    recall = len(used_required_types) / len(required_tool_names) if required_tool_names else 1.0
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return (precision, recall, f1)


async def _llm_judge(model, rubric_prompt: str) -> tuple[int, str]:
    """LLM-as-judge: the real model scores a transcript against a written
    1-5 rubric, for the cases where no programmatic verifier can capture
    quality (a clarifying question that's actually useful vs. just "?", a
    refactor that's actually clearer, a conflict-surfacing answer that
    reads as appropriately hedged vs. confidently wrong). Supplements, never
    replaces, the structural checks already in each test — the judge only
    grades quality on top of a state that's already been verified safe/
    correct. The rubric prompt is plain text, included at the call site in
    each test, so what's being asked of the judge is auditable, not a
    black box."""
    from openmate.kernel.types import Message, TextPart
    from openmate.ports.model import ModelRequest

    resp = await model.generate(ModelRequest(messages=[Message("user", [TextPart(rubric_prompt)])], max_tokens=200))
    text = "".join(p.text for p in resp.message.content if hasattr(p, "text"))
    m = re.search(r"\b([1-5])\b", text)
    score = int(m.group(1)) if m else 0
    return score, text


def _run_script(args: list[str], cwd) -> tuple[int, str, str]:
    p = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=15)
    return p.returncode, p.stdout, p.stderr
