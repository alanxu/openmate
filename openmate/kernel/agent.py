"""The ``Agent`` facade and its ``Harness`` — *what* the agent is and can do.

Kept out of :mod:`openmate.kernel.types` on purpose: ``types`` is the kernel's
pure vocabulary (data only, no behavior, no dependency on higher layers), whereas
``Agent`` is a *behavioral* facade — ``run``/``stream``/``resume``/``cancel``
delegate to the loop engine in :mod:`openmate.kernel.loop`. Putting it here lets
those delegations be plain top-level imports instead of cycle-dodging lazy ones.

``Agent`` composes a ``model`` + ``instructions`` + a ``Harness`` (its environment)
+ ``Services`` (shared infra); ``loop`` only TYPE_CHECKING-imports ``Agent``, so
``agent → loop → types`` stays an acyclic dependency chain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from .loop import cancel as _cancel
from .loop import drive as _drive
from .loop import resume as _resume
from .loop import stream as _stream

if TYPE_CHECKING:
    from ..ports.model import Model
    from ..ports.tool import Tool
    from .types import RunResult, Services


@dataclass
class Harness:
    """The agent's *environment* — what it can do and how it behaves.

    Its tools plus the pluggable policies (planner, memory, context, guardrails,
    stop) that higher layers inject. The PoC populates ``tools`` only; the policy
    slots are forward-looking seams that fall back to library defaults when unset
    (e.g. ``planner=None`` means ReAct, ``stop=None`` means the ``max_steps`` cap).
    """

    tools: list["Tool"] = field(default_factory=list)
    planner: Any | None = None  # ReasoningStrategy (05) — default ReAct
    memory: Any | None = None  # Memory (06)
    context_policy: Any | None = None  # ContextPolicy (09)
    guardrails: Any | None = None  # GuardrailSet (10)
    stop: Any | None = None  # StopPolicy (02) — PoC default = max_steps


# the convenience-constructor fields that get bundled into a Harness
_HARNESS_POLICY = ("planner", "memory", "context_policy", "guardrails", "stop")


@dataclass
class HumanDecision:
    """A human's answer to a paused run (HITL) — injected via ``Agent.resume``."""

    action: Literal["approve", "reject", "edit"]
    edited_args: dict | None = None


class Agent:
    """The facade callers hold — the only object you ``run()``.

    It composes a ``model``, ``instructions``, a ``Harness`` (the environment) and
    ``Services`` (shared infra), and drives the loop. The loop engine lives in
    :mod:`openmate.kernel.loop`; ``run``/``stream``/``resume``/``cancel`` are thin
    delegations to it, so callers never touch the engine directly.

    Construct it with an explicit ``Harness``::

        Agent(name="a", model=m, instructions="…", services=svc,
              harness=Harness(tools=[...]))

    or with the convenience form, which bundles the environment fields into a
    ``Harness`` for you::

        Agent(name="a", model=m, instructions="…", services=svc, tools=[...])
    """

    def __init__(
        self,
        *,
        name: str,
        model: "Model",
        instructions: str,
        services: "Services",
        harness: Harness | None = None,
        tools: list["Tool"] | None = None,
        max_steps: int = 12,  # PoC stop rule; replaced by a composable StopPolicy later
        temperature: float | None = None,
        max_tokens: int = 2048,
        **policy: Any,
    ) -> None:
        unknown = set(policy) - set(_HARNESS_POLICY)
        if unknown:
            raise TypeError(f"unexpected Agent argument(s): {', '.join(sorted(unknown))}")
        if harness is not None:
            if tools is not None or policy:
                raise TypeError(
                    "pass either harness= or the convenience fields "
                    "(tools=/planner=/memory=/…), not both"
                )
        else:
            harness = Harness(tools=list(tools or []), **policy)
        self.name = name
        self.model = model
        self.instructions = instructions
        self.services = services
        self.harness = harness
        self.max_steps = max_steps
        self.temperature = temperature
        self.max_tokens = max_tokens

    @property
    def tools(self) -> list["Tool"]:
        """The agent's tools — sugar for ``harness.tools`` (the executor reads this)."""
        return self.harness.tools

    async def run(self, user_input: str, *, thread_id: str | None = None) -> "RunResult":
        """Drive the loop to completion and return the terminal ``RunResult``."""
        return await _drive(self, user_input, thread_id=thread_id)

    def stream(self, user_input: str, *, thread_id: str | None = None):
        """Yield ``Event``s as the run executes (an async iterator)."""
        return _stream(self, user_input, thread_id=thread_id)

    async def resume(
        self, thread_id: str, decision: "HumanDecision | None" = None
    ) -> "RunResult":
        """Continue a checkpointed thread, optionally injecting a human decision."""
        return await _resume(self, thread_id, decision)

    async def cancel(self, thread_id: str) -> None:
        """Request cooperative cancellation of an in-flight run."""
        await _cancel(self, thread_id)

    def __repr__(self) -> str:
        model = getattr(self.model, "name", "?")
        return f"Agent(name={self.name!r}, model={model!r}, tools={len(self.tools)})"
