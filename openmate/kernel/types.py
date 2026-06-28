"""Domain model — the irreducible vocabulary every other module is built from.

Layer 0 of the architecture (see docs/01-domain-model-and-kernel.md). Zero
third-party dependencies. Types are immutable where practical; all
nondeterministic inputs (clock, rng, id generation) arrive through ``Services``
so runs are reproducible.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Callable, Literal
from uuid import uuid4

if TYPE_CHECKING:  # imported only for type hints — keeps the kernel dependency-free
    from ..ports.store import Store
    from ..ports.tracer import Tracer
    from .agent import Agent
    from .events import EventBus

Role = Literal["system", "user", "assistant", "tool"]


# --- content parts: a closed union; adapters translate to/from provider formats ---
@dataclass(frozen=True)
class TextPart:
    text: str


@dataclass(frozen=True)
class ToolCallPart:
    id: str
    name: str
    args: dict


@dataclass(frozen=True)
class ToolResultPart:
    call_id: str
    content: list["Part"]
    is_error: bool = False


@dataclass(frozen=True)
class ThinkingPart:
    text: str
    signature: str | None = None  # provider reasoning trace; captured for display/audit, not re-sent


Part = TextPart | ToolCallPart | ToolResultPart | ThinkingPart  # widened in later phases


@dataclass(frozen=True)
class Message:
    role: Role
    content: list[Part]
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:
        """Concatenated text of all ``TextPart``s — a convenience accessor."""
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

    @property
    def tool_calls(self) -> list[ToolCallPart]:
        return [p for p in self.content if isinstance(p, ToolCallPart)]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, o: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + o.prompt_tokens,
            self.completion_tokens + o.completion_tokens,
            self.cost_usd + o.cost_usd,
            self.wall_ms + o.wall_ms,
        )


@dataclass
class RunState:
    """The complete, serializable state of one execution — and the checkpoint.

    Persist it and you can resume or replay. All mutations go through the helper
    methods below so there is a single update path.
    """

    thread_id: str
    messages: list[Message]
    scratch: dict = field(default_factory=dict)  # strategy-private working memory
    step: int = 0
    status: Literal["running", "paused", "done", "error"] = "running"
    usage: Usage = field(default_factory=Usage)
    cursor: dict = field(default_factory=dict)  # resume point (HITL / crash)
    rev: int = 0  # monotonic checkpoint revision

    def with_messages(self, *msgs: Message) -> "RunState":
        return replace(self, messages=[*self.messages, *msgs], rev=self.rev + 1)

    def advance(self) -> "RunState":
        return replace(self, step=self.step + 1, rev=self.rev + 1)

    def with_usage(self, u: Usage) -> "RunState":
        return replace(self, usage=self.usage + u)

    def stop(self, status: Literal["done", "paused", "error"], reason: str) -> "RunState":
        return replace(self, status=status, scratch={**self.scratch, "stop_reason": reason})

    def to_result(self) -> "RunResult":
        """Terminal, caller-facing projection. The single construction path for RunResult."""
        final = next((m for m in reversed(self.messages) if m.role == "assistant"), None)
        status: Literal["done", "paused", "error"] = (
            self.status if self.status in ("done", "paused", "error") else "done"
        )
        return RunResult(
            thread_id=self.thread_id,
            status=status,
            final=final,
            reason=self.scratch.get("stop_reason", "natural"),
            state=self,
            usage=self.usage,
            steps=self.step,
        )


@dataclass
class RunResult:
    """What ``Agent.run()`` returns and ``RunFinished`` carries."""

    thread_id: str
    status: Literal["done", "paused", "error"]
    final: Message | None = None
    reason: str | None = None  # natural | max_steps | error | paused | ...
    state: RunState | None = None
    usage: Usage = field(default_factory=Usage)
    steps: int = 0
    error: str | None = None

    @property
    def text(self) -> str:
        return self.final.text if self.final else ""

    @property
    def ok(self) -> bool:
        return self.status == "done"

    @property
    def paused(self) -> bool:
        return self.status == "paused"


def _default_new_id() -> str:
    return uuid4().hex


@dataclass
class Services:
    """The resolved bundle of infrastructure ports handed to the loop and tools.

    Threading this explicitly — rather than reaching for globals — is what keeps
    runs deterministic and testable.
    """

    store: "Store"
    tracer: "Tracer"
    bus: "EventBus"
    clock: Callable[[], float] = time.time  # injected; never time.time() directly downstream
    rng: random.Random = field(default_factory=random.Random)  # seeded; never global random
    new_id: Callable[[], str] = _default_new_id


@dataclass
class RunContext:
    """Per-run handle passed to tools and strategies.

    ``agent`` is typed as the :class:`~openmate.kernel.agent.Agent` facade but only
    under ``TYPE_CHECKING`` — ``types`` (pure vocabulary) never imports the
    behavioral facade at runtime, keeping the kernel's dependency direction clean.
    """

    state: RunState
    services: Services
    agent: "Agent"
    deadline: float | None = None  # epoch seconds; tools self-limit
