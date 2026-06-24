# 01 — Domain Model & Kernel

> The irreducible types every other module is built from. Part of OpenMate; see [architecture.md §5](architecture.md#5-the-core-domain-model-the-kernel). This is **Layer 0** — zero third-party deps beyond a validation lib.

## Scope & responsibilities

The kernel owns the *vocabulary* of the system: the data types for conversations (`Message`/`Part`), execution state (`RunState`), narration (`Event`), the declarative agent (`Agent`), and the runtime service bundle (`Services`/`RunContext`). It also owns the **event bus** and the rules for evolving state. It deliberately contains **no behavior** beyond constructing, validating, updating, and serializing these values — all behavior lives in higher layers, which keeps the kernel small and stable.

Design rules: types are immutable where practical (frozen dataclasses), every nondeterministic input arrives through `Services` (clock, rng, id-gen) so runs replay (architecture P8), and every type is serializable to/from JSON so state is a portable checkpoint.

---

## Core abstractions (class level)

```python
# openmate/kernel/types.py
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import Literal, Protocol, Any, Callable, Iterable

Role = Literal["system", "user", "assistant", "tool"]

# --- content parts (a closed union; adapters translate to/from provider formats) ---
@dataclass(frozen=True)
class TextPart:       text: str
@dataclass(frozen=True)
class ToolCallPart:   id: str; name: str; args: dict
@dataclass(frozen=True)
class ToolResultPart: call_id: str; content: list["Part"]; is_error: bool = False
Part = TextPart | ToolCallPart | ToolResultPart        # widened in later phases

@dataclass(frozen=True)
class Message:
    role: Role
    content: list[Part]
    name: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def text(self) -> str:                              # convenience accessor
        return "".join(p.text for p in self.content if isinstance(p, TextPart))

@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    wall_ms: float = 0.0
    def __add__(self, o: "Usage") -> "Usage": ...       # accumulate across steps
```

`RunState` is the single source of truth and the checkpoint unit:

```python
@dataclass
class RunState:
    thread_id: str
    messages: list[Message]
    plan: "Plan | None" = None
    scratch: dict = field(default_factory=dict)         # strategy-private working memory
    step: int = 0
    status: Literal["running", "paused", "done", "error"] = "running"
    usage: Usage = field(default_factory=Usage)
    cursor: dict = field(default_factory=dict)          # resume point (HITL / crash)
    rev: int = 0                                        # monotonic checkpoint revision

    # all mutations go through these helpers so we keep one update path
    def with_messages(self, *msgs: Message) -> "RunState": ...
    def advance(self) -> "RunState": ...                # step += 1, rev += 1
    def finish(self, answer: Message) -> "RunState": ...
    def to_result(self) -> "RunResult": ...             # terminal projection → RunResult
```

`RunResult` is the terminal, caller-facing projection of a run — what `Runtime.run()`/`Runtime.resume()` ([02](02-agent-loop-and-runtime.md)) and `Orchestrator.run()` ([08](08-multi-agent-orchestration.md)) return, what the `RunFinished` event carries, and what the eval harness ([11](11-observability-and-evaluation.md)) scores. `RunState.to_result()` is its single construction path.

```python
@dataclass
class RunResult:
    thread_id: str
    status: Literal["done", "paused", "error"]    # terminal projection of RunState.status (never "running")
    final: Message | None = None                  # the answer; None when paused/errored
    reason: str | None = None                     # natural | budget | no_progress | cancelled | paused | error
    state: RunState | None = None                 # full final state, for inspection / resume
    usage: Usage = field(default_factory=Usage)    # tokens, cost, wall-time for budgets (12) / evals (11)
    steps: int = 0
    error: str | None = None                      # message when status == "error"

    @property
    def text(self) -> str: return self.final.text if self.final else ""
    @property
    def ok(self) -> bool: return self.status == "done"
    @property
    def paused(self) -> bool: return self.status == "paused"   # caller should await a decision + Runtime.resume(...)
```

`RunContext` is the per-run handle passed to tools and strategies; `Services` is the resolved port bundle:

```python
@dataclass
class Services:
    model: "Model"
    store: "Store"
    tracer: "Tracer"
    bus: "EventBus"
    clock: Callable[[], float]          # injected; never time.time() directly
    rng: "random.Random"                # seeded; never global random
    new_id: Callable[[], str]           # injected id generator

@dataclass
class RunContext:
    state: RunState
    services: Services
    agent: "Agent"
    deadline: float | None = None       # epoch seconds; tools self-limit
```

---

## Phase 0 — PoC (foundational)

**Goal:** the smallest type set that lets a ReAct loop run end to end against a fake model.

Ship exactly: `Role`, the three `Part`s above, `Message`, `Usage`, `RunState`, `Agent` (minimal — see below), `Services`, `RunContext`, a synchronous in-process `EventBus`, and a handful of `Event`s. Nothing else.

```python
# openmate/kernel/events.py
@dataclass
class Event:
    thread_id: str; step: int; ts: float            # base; ts from services.clock
class RunStarted(Event):    ...
class MessageAdded(Event):  message: Message
class ToolCallRequested(Event): call: ToolCallPart
class ToolReturned(Event):  result: ToolResultPart; ms: float
class RunFinished(Event):   result: "RunResult"

class EventBus:
    def __init__(self): self._subs: list[Callable[[Event], None]] = []
    def subscribe(self, fn): self._subs.append(fn)
    def emit(self, ev: Event):                       # synchronous fan-out in PoC
        for fn in self._subs: fn(ev)
```

```python
# openmate/kernel/agent.py  (PoC form — grows in 02/03)
@dataclass
class Agent:
    name: str
    model: "Model"
    instructions: str
    tools: list["Tool"] = field(default_factory=list)
    max_steps: int = 20                              # PoC stop rule; replaced by StopPolicy in 02
```

**PoC acceptance:** construct an `Agent`, run a loop that appends `Message`s to `RunState`, emit events, and serialize the final `RunState` to JSON and back losslessly.

---

## Phase 1 — Rich content model

Widen `Part` to cover modern model I/O. Each new part is additive; adapters that don't support a part degrade it (e.g., drop `ThinkingPart`, base64-inline an `ImagePart`).

```python
@dataclass(frozen=True)
class ImagePart:     data: bytes | str; mime: str            # bytes or URL
@dataclass(frozen=True)
class AudioPart:     data: bytes | str; mime: str
@dataclass(frozen=True)
class FilePart:      uri: str; mime: str; name: str | None = None
@dataclass(frozen=True)
class ThinkingPart:  text: str; signature: str | None = None # provider reasoning trace
@dataclass(frozen=True)
class DataPart:      schema_ref: str; value: Any             # validated structured output
@dataclass(frozen=True)
class Citation:      doc_id: str; span: tuple[int, int]; quote: str
@dataclass(frozen=True)
class CitedTextPart: text: str; citations: list[Citation]    # grounded generation (links to 07/10)
Part = (TextPart | ToolCallPart | ToolResultPart | ImagePart | AudioPart
        | FilePart | ThinkingPart | DataPart | CitedTextPart)
```

Techniques in this phase: **content normalization** (a `normalize(parts)` that merges adjacent text, strips empty parts), **part redaction** (replace sensitive parts with `RedactedPart` for logs), and **token-aware truncation hooks** (parts expose an estimated size for the context assembler in [09](09-context-engineering.md)).

---

## Phase 2 — State management & event sourcing

Make state evolution principled so checkpointing, time-travel, and multi-agent shared state are safe.

- **Immutable updates with reducers.** All mutations are pure functions `RunState -> RunState`. A `Reducer` registry lets orchestration merge concurrent sub-agent updates deterministically (last-writer-wins, append, or custom per field) — the foundation for shared-state orchestration in [08](08-multi-agent-orchestration.md).

```python
class Reducer(Protocol):
    def merge(self, base: Any, incoming: Any) -> Any: ...
@dataclass
class StateSchema:                       # declares how each RunState field merges
    reducers: dict[str, Reducer]         # e.g. {"messages": Append(), "scratch": DictMerge()}
```

- **Event-sourced history.** `RunState` can be reconstructed by folding the event log: `state = reduce(apply, events, initial)`. This gives free **time-travel** (replay to event N) and **forking** (branch a run from any point) used by debugging in [11](11-observability-and-evaluation.md).
- **Structural sharing.** Large `messages` lists use copy-on-write slices so per-step checkpoints don't deep-copy the whole transcript.
- **Deterministic identity.** All ids (`tool_call.id`, message ids) come from `services.new_id`; in record/replay mode it's a counter, so two runs produce identical ids.

---

## Phase 3 — Serialization, versioning & safety

- **Stable wire format.** A `codec` module (`to_jsonable`/`from_jsonable`) with a tagged union encoding for `Part`/`Event` (`{"_t": "ToolCallPart", ...}`). Bytes parts are content-addressed and offloaded to a blob store; the state holds a reference (ties to memory offloading in [09](09-context-engineering.md)).
- **Schema versioning & migration.** Every serialized `RunState` carries `schema_version`. A `Migrator` chain upgrades old checkpoints on load so long-lived threads survive code changes.
- **Redaction & PII policy at the type layer.** `Message.metadata` can carry sensitivity tags; a `Redactor` produces a log-safe projection so traces/checkpoints never leak secrets (enforced with [10](10-safety-and-guardrails.md)).
- **Validation.** Pydantic (or `msgspec`) models validate inbound tool args and structured outputs; invalid data becomes a typed `ToolResultPart(is_error=True)` rather than an exception (recoverable per architecture P6).

---

## Event taxonomy (grows across phases)

| Event | Emitted when | First phase |
|---|---|---|
| `RunStarted` / `RunFinished` | run boundaries | 0 |
| `MessageAdded` | any message appended | 0 |
| `ToolCallRequested` / `ToolReturned` | tool dispatch | 0 |
| `ModelRequested` / `ModelStreamed` | model call + token deltas | 1 (with 03) |
| `PlanUpdated` | planner revises plan | 1 (with 05) |
| `CheckpointSaved` | state persisted | 2 |
| `GuardrailTriggered` | guardrail acts | (with 10) |
| `HandoffOccurred` | active agent swapped | (with 08) |

---

## Testing & verification

- **Round-trip property test:** `from_jsonable(to_jsonable(x)) == x` for every type (Hypothesis-generated).
- **Determinism test:** same seed + recorded model ⇒ byte-identical event log (the replay guarantee).
- **Reducer laws:** merges are associative where claimed; concurrent merges converge.
- **Migration test:** a corpus of old-version checkpoints loads cleanly after migration.

## Trade-offs & open questions

Frozen dataclasses vs. Pydantic models for hot-path types (speed vs. validation ergonomics — current lean: dataclasses in kernel, Pydantic at boundaries). Whether `scratch` should be strongly typed per strategy or stay an open dict (PoC: open dict; revisit if it becomes a bug source). How aggressively to offload bytes (latency vs. window size).
