# 05 — Planning & Reasoning Strategies

> The swappable "what do I do next?" policy. Part of OpenMate; see [architecture.md §7](architecture.md#7-planning--reasoning-strategies). Every pattern here implements one port — `ReasoningStrategy` — so switching is a one-line change to `Agent.planner`.

## Scope & responsibilities

A `ReasoningStrategy` answers a single question each step: *given the current state, what is the next `StepOutcome`?* (tool calls, a final answer, a subgoal, or a handoff). The runtime ([02](02-agent-loop-and-runtime.md)) calls it inside the `ReasoningInterceptor`; the strategy may itself make several model calls. This module is where ReAct, plan-and-execute, reflection, and search live — and crucially, where they **compose** (reflection wraps any strategy).

---

## Core abstractions (class level)

```python
# openmate/ports/planner.py
class ReasoningStrategy(Protocol):
    async def step(self, state: RunState, svc: Services) -> StepOutcome: ...

@dataclass
class PlanStep:
    id: str; goal: str; tool_hint: str | None = None
    status: Literal["pending","active","done","failed","skipped"] = "pending"
    result: Any = None                                  # filled from PlanStepResult.output
    deps: list[str] = field(default_factory=list)       # step ids that must finish first (DAG edges)
    attempts: int = 0

@dataclass
class PlanStepResult:                                   # ── the OUTCOME OF STEP EXECUTION (one PlanStep) ──
    step_id: str                                        # NOT the loop-level StepOutcome (02); this is internal
    status: Literal["done","failed","needs_replan"]
    output: Any = None                                  # the step's product (text / DataPart / artifact ref)
    observations: list[Part] = field(default_factory=list)        # tool results gathered while executing
    tool_calls: list[ToolCallPart] = field(default_factory=list)  # what the step actually invoked
    error: str | None = None
    diverged: bool = False                              # an observation broke a downstream assumption → replan
    usage: Usage = field(default_factory=Usage)

@dataclass
class Plan:
    steps: list[PlanStep]; rationale: str; revision: int = 0
    def next_ready(self) -> PlanStep | None: ...        # first step whose deps are all done
    def ready_batch(self, max_parallel: int = 1) -> list[PlanStep]: ...  # all dep-satisfied steps (parallel)
    def record(self, r: PlanStepResult) -> None: ...    # apply a result: set the step's status + result
    def patch(self, ops: list["PlanOp"]) -> "Plan": ...  # plan repair; bumps revision
    @property
    def past_steps(self) -> list[tuple[PlanStep, Any]]: ...  # completed (step, output) — fed to the replanner
    @property
    def done(self) -> bool: ...

class Critic(Protocol):                                  # reflection / goal-stop (Phase 2)
    async def assess(self, state: RunState, svc: Services) -> "Verdict": ...  # accept | revise(feedback)

class Joiner(Protocol):                                  # plan-execute: decide finish-vs-replan when steps run out
    async def decide(self, state: RunState, svc: Services) -> "JoinVerdict": ...
@dataclass
class JoinVerdict: finish: bool; feedback: str | None = None
```

---

## Phase 0 — PoC (foundational): ReAct

**Goal:** the default strategy — one model call per step, returning tool calls or a final answer. This *is* the loop's natural mode; it's the foundation everything else extends.

```python
# openmate/strategies/react.py
class ReAct(ReasoningStrategy):
    def __init__(self, model: Model): self.model = model
    async def step(self, state, svc):
        resp = await self.model.generate(ModelRequest(state.messages, tools=specs(state)))
        calls = [p for p in resp.message.content if isinstance(p, ToolCallPart)]
        return ToolCalls(calls) if calls else Final(resp.message)
```

Include a **prompted ReAct fallback** (`PromptedReAct`) for models without native tool calling: instruct a `Thought/Action/Action Input/Observation` format and parse it, emitting the same `StepOutcome`. This keeps the runtime provider-agnostic ([03](03-model-port-and-providers.md) Phase 2).

**PoC acceptance:** ReAct solves a multi-tool task; the prompted variant produces identical outcomes against a no-tool-calling `FakeModel`.

---

## Phase 1 — Plan-and-execute

Separate planning from execution: a planner writes the whole plan up front; an executor runs the steps; a replanner/joiner revises only when reality diverges. Best for long, structured tasks where mid-stream drift is costly, and it enables a **cheap-executor / strong-planner** split ([03](03-model-port-and-providers.md) routing) — the strong model plans once, steps run on a cheap model.

**Loop-cooperative, not self-contained.** `PlanExecute.step()` runs once per *outer loop iteration* and returns a loop-level `StepOutcome` ([02](02-agent-loop-and-runtime.md)); it does **not** dispatch tools itself. It emits `ToolCalls` for the active plan step and lets the runtime execute them — so guardrails ([10](10-safety-and-guardrails.md)), sandboxing ([04](04-tools-and-mcp.md)), HITL, and tracing all go through the one tool path — then reads the results on the next iteration. Active-step bookkeeping lives in `state.scratch`, so it's checkpointed and resumable for free ([06](06-memory-and-state.md)). Keep the two "outcomes" distinct:

> **`PlanStepResult`** = the outcome of executing *one plan step* (internal; defined above). **`StepOutcome`** = what the strategy returns to the runtime *each loop step* (`ToolCalls | Final | …`). `step()` translates between them.

```python
class PlanExecute(ReasoningStrategy):
    def __init__(self, planner_model, executor_model, *,
                 joiner: Joiner, repair=True, max_parallel=1, max_revisions=3): ...

    async def step(self, state, svc) -> StepOutcome:
        # 0. plan once
        if state.plan is None:
            state.plan = await self._plan(state, svc); svc.bus.emit(PlanUpdated(plan=state.plan))
        plan = state.plan
        # 1. fold the tool results dispatched last iteration into the active step
        if self._awaiting_results(state):
            result: PlanStepResult = self._collect(state, svc)      # ← the step-execution outcome
            plan.record(result); svc.bus.emit(PlanUpdated(plan=plan))
            if result.diverged and self.repair and plan.revision < self.max_revisions:
                state.plan = plan = await self._replan(plan, result, state, svc)
        # 2. pick the next ready step(s); if none, ask the joiner finish-or-replan
        ready = plan.ready_batch(self.max_parallel)
        if not ready:
            verdict = await self.joiner.decide(state, svc)
            if verdict.finish or plan.revision >= self.max_revisions:
                return Final(await self._synthesize(state, svc))
            state.plan = plan = await self._replan(plan, None, state, svc)
            ready = plan.ready_batch(self.max_parallel)
        # 3. turn ready steps into concrete tool calls for the runtime to execute
        return ToolCalls(await self._next_actions(ready, state, svc))
```

**Method by method:**

- `_plan(state, svc) -> Plan` — the **planner**: one call to `planner_model` decomposes the objective into a `PlanStep` DAG (goal, `tool_hint`, `deps`). Cap decomposition depth/breadth to prevent over-planning; emit as structured data, never prose.
- `_next_actions(ready, state, svc) -> list[ToolCallPart]` — the **executor**: `executor_model` (cheap) maps each ready step's `goal` into concrete tool calls. With `max_parallel>1` it emits calls for several independent steps at once for the runtime to dispatch concurrently ([02](02-agent-loop-and-runtime.md) Phase 4).
- `_collect(state, svc) -> PlanStepResult` — folds the returned tool results into a `PlanStepResult`: sets `status` (`done`/`failed`/`needs_replan`), captures `observations`/`output`, and sets `diverged=True` when an observation invalidates a downstream step. **This is the step-execution outcome** (replacing the old, type-confused `_execute`/`_diverged` pair).
- `_replan(plan, result, state, svc) -> Plan` — the **replanner**: patches the *remaining* steps (`plan.patch`) using `plan.past_steps` plus the failure, bumping `revision`. Patch, don't restart.
- `_synthesize(state, svc) -> Message` — the **solver / joiner-finish**: combines `past_steps` into the final answer with citations.
- `joiner.decide(...)` — when no steps remain, decides **finish vs. replan** so the agent never stops with the goal unmet (the explicit verifier the bare PoC lacked).

### Production references & named variants

This is what ships in production; OpenMate models each as a configuration of the same machinery:

- **LangGraph "plan-and-execute"** — planner → step executor → **replanner** that, from accumulated `past_steps`, decides refine-or-finish. Directly mirrored by the class above. ([ref](https://www.langchain.com/blog/planning-agents))
- **ReWOO** ("Reasoning WithOut Observation") — plan *all* steps up front with later steps referencing earlier outputs via variables (`#E1`), execute without feeding observations back to the planner, then a **solver** combines. Fewer planner calls/tokens. → `class ReWOO(PlanExecute)` with `repair=False` + variable substitution in `_next_actions`.
- **LLMCompiler** — planner streams a **task DAG**; a fetching unit schedules tasks as deps resolve for **parallel** execution; a joiner decides finish-or-replan. → `class LLMCompiler(PlanExecute)` with `max_parallel>1`, leaning on `Plan.ready_batch`. ([ref](https://www.langchain.com/blog/planning-agents))
- **Manus (Flow mode / `PlanningFlow`)** — separate planning, execution, and **verification** agents in sandboxed VMs; adjusts the plan or picks alternate paths on failure. Maps to `PlanExecute` + a `Joiner` verifier + worker isolation ([08](08-multi-agent-orchestration.md)); the OSS **OpenManus** is the planner-executor-with-sandbox port. ([ref](https://dev.to/jamesli/openmanus-architecture-deep-dive-enterprise-ai-agent-development-with-real-world-case-studies-5hi4))
- **Devin** — a single planning loop (plan → execute → replan inline): the monolithic end of the spectrum.
- **Claude Code** — an explicit **PLAN → EXECUTE → VERIFY** loop: decompose the request into a structured task list with dependencies, work through it marking `in_progress → completed`, then verify (tests/review). The `TaskCreate`/`TaskUpdate`/`TaskList` tools (which replaced the older single `TodoWrite` in early 2026) add dependency tracking + persistence — effectively `Plan`/`PlanStep` surfaced to the user — with an opt-in **"plan mode"** that gates execution on human approval (HITL, [10](10-safety-and-guardrails.md)). Note Claude's selection is *soft*: the model itself decides whether to lay out a plan (see [§ choosing a strategy](#choosing-a-strategy)), rather than switching between two engines. ([ref](https://code.claude.com/docs/en/agent-sdk/todo-tracking))
- **"Todo-list" coding agents & deep-research** — a visible structured checklist executed step-by-step; same shape, surfaced in the UI. Lineage: BabyAGI, HuggingGPT.

**Production best practices, baked in:** store the plan as **structured data** with per-step `status` (inspectable, debuggable, resumable) — that's `Plan`/`PlanStep`; always keep a **replanner/joiner** for finish-vs-continue; reserve the strong model for planning and run steps cheaply; cap `revisions` to bound cost.

Techniques recap: explicit **plan as data** (observable via `PlanUpdated`, editable during HITL); **plan repair** (patch, not restart); **dependency DAG** + `ready_batch` for parallel steps; **finish-or-replan joiner**; **decomposition limits**.

### Planning as a tool (soft, Claude-style selection)

A third design sits between Phase 0 ReAct and the Phase 1 `PlanExecute` strategy: run a **plain ReAct loop** but give the agent **planning tools**. The model decides *from the tool's own description* whether a task warrants a plan, so strategy selection is **emergent, not routed** — this is how Claude Code works (`TaskCreate`/`TaskUpdate`).

```python
# openmate/strategies/planning_tools.py
class CreatePlanTool(Tool):
    spec = ToolSpec(
        name="create_plan",
        # the soft-selection heuristic lives HERE — the model reads it and self-selects
        description="Lay out a step-by-step plan BEFORE acting. Use for non-trivial work "
                    "(3+ dependent steps); skip it for simple tasks.",
        parameters=schema_of(rationale=str, steps=list[PlanStepInput]),  # PlanStepInput: goal, tool_hint, deps
        side_effecting=False,
    )
    async def invoke(self, args, ctx: RunContext) -> ToolResult:
        ctx.state.plan = Plan(steps=[PlanStep(id=new_id(), **s) for s in args["steps"]],
                              rationale=args.get("rationale", ""))
        ctx.services.bus.emit(PlanUpdated(plan=ctx.state.plan))
        return ToolResult([TextPart(render_plan(ctx.state.plan))])     # echo so the model sees the plan

class UpdatePlanTool(Tool):                                             # mark steps done/failed; add/edit/drop
    spec = ToolSpec(name="update_plan",
                    description="Mark step status or revise the plan as you learn more.",
                    parameters=schema_of(ops=list[PlanOpInput]), side_effecting=False)
    async def invoke(self, args, ctx):
        ctx.state.plan = ctx.state.plan.patch(parse_ops(args["ops"]))
        ctx.services.bus.emit(PlanUpdated(plan=ctx.state.plan))
        return ToolResult([TextPart(render_plan(ctx.state.plan))])
```

How it wires into the loop (no new strategy needed):

- **State writes:** planning tools mutate `RunState.plan` via `RunContext` — sanctioned because `plan` is run-scoped state. (Stricter alternative: return a `PlanUpdate` *effect* the runtime applies, keeping all tools side-effect-free — pick one convention and hold it.)
- **Selection heuristic in the description:** a capable model reads "use for 3+ dependent steps" and self-selects whether to plan — no `StrategyRouter`, no separate engine.
- **Keep the plan in view:** a `ContextInterceptor` ([09](09-context-engineering.md)) **pins the current plan** into the window each turn so the model stays anchored and keeps it updated; without this it forgets the plan it wrote.
- **Assembly:** `Agent(planner=ReAct(...), tools=[CreatePlanTool(), UpdatePlanTool(), *task_tools])`. The plan stays observable (`PlanUpdated`), checkpointed, and HITL-editable ([10](10-safety-and-guardrails.md)) — same guarantees as `PlanExecute`.

**Soft (planning-as-tool) vs. hard (`PlanExecute` strategy) vs. router:**

| Dimension | Planning-as-tool (soft) | `PlanExecute` strategy (hard) | `StrategyRouter` |
|---|---|---|---|
| Who decides to plan | the model, via tool description | always (per strategy) | a classifier, up front |
| Determinism | low (model judgment) | high | high |
| Strong-planner / cheap-executor split | no — one model, one loop | yes | possible |
| Plan adherence / drift control | soft (re-inject + prompt) | structural (executor constrained) | depends |
| Parallel / DAG guarantees | model's free choice | scheduler-enforced (`ready_batch`) | depends |
| Adaptivity (replan mid-task) | high (replan anytime) | medium (explicit replan step) | low |
| Build cost | tiny (two tools) | medium | medium–high (classifier) |
| Best for | capable models; interactive/coding agents | long structured pipelines; cost-sensitive | mixed workloads needing forced routing |

The soft path trades **determinism and the model-cost split** for **simplicity, adaptivity, and a single execution path** — a strong default with capable models, which is exactly why Claude Code uses it. Reach for `PlanExecute` when you need the planner/executor split, enforced parallelism, or hard drift control.

---

## Phase 2 — Reflection / self-correction

A critic evaluates a candidate against the goal; the actor revises until accepted or budget hit. Highest accuracy gain; composes over *any* strategy.

```python
class Reflexion(ReasoningStrategy):
    def __init__(self, actor: ReasoningStrategy, critic: Critic, max_revisions=2): ...
    async def step(self, state, svc):
        outcome = await self.actor.step(state, svc)
        if isinstance(outcome, Final):
            verdict = await self.critic.assess(state.with_messages(outcome.message), svc)
            if verdict.revise and state.scratch.get("revisions",0) < self.max_revisions:
                state.scratch["revisions"] = state.scratch.get("revisions",0)+1
                return self.actor.step(state.with_messages(critique_msg(verdict)), svc)  # retry w/ feedback
        return outcome
```

Variants to include: **self-refine** (model critiques itself), **critic model** (a separate, possibly stronger model), **verifier tools** (run tests/checks as the critic — e.g., execute generated code), and **Reflexion memory** (persist lessons across attempts into long-term memory, [06](06-memory-and-state.md)). `GoalReached` stop policy ([02](02-agent-loop-and-runtime.md)) reuses the same `Critic`.

---

## Phase 3 — Search & advanced reasoning

For tasks with verifiable intermediate states, generalize "a line of reasoning" to "a search over reasonings."

- **Tree-of-Thought (ToT):** expand multiple candidate branches, score with a value function, keep the best (beam search).

```python
class TreeOfThought(ReasoningStrategy):
    def __init__(self, model, value_fn: "ValueFn", branch=3, beam=2, depth=3): ...
    # expand → evaluate → prune → repeat; return best leaf as Final
```

- **Graph-of-Thought / MCTS:** rollouts with backpropagation for harder search.
- **Least-to-most / decomposition prompting:** solve easy sub-questions first to scaffold hard ones.
- **Self-consistency:** sample N reasoning paths, majority-vote the answer (cheap accuracy boost for single-answer tasks).
- **Plan caching:** memoize plans for recurring task shapes (procedural memory, [06](06-memory-and-state.md)).

These are opt-in and expensive; the strategy port makes them swappable without touching the loop.

---

## Phase 4 — Adaptive & meta-strategies

- **Strategy router:** a meta-strategy that picks ReAct vs. plan-execute vs. ToT per task based on difficulty/length signals (a small classifier or heuristic), so cheap tasks stay cheap.
- **Budget-aware reasoning:** scale reflection depth / sample count to the remaining budget ([12](12-production-and-reliability.md)).
- **Learned planning:** mine successful trajectories ([11](11-observability-and-evaluation.md)) into reusable skills/plans (procedural memory).

---

## Choosing a strategy

| Strategy | Best for | Cost | Drift control |
|---|---|---|---|
| `ReAct` | open-ended/exploratory | low | low |
| `PlanExecute` | long, structured pipelines | medium | high |
| `Reflexion` (wrap) | quality-critical output | high | medium |
| `TreeOfThought` | verifiable search problems | very high | high |
| `StrategyRouter` | mixed workloads | adaptive | adaptive |

**Default posture (after Anthropic's *Building Effective Agents*):** start with the **simplest** pattern that works — a plain ReAct loop — and escalate to `PlanExecute` / orchestrator-workers ([08](08-multi-agent-orchestration.md)) only when subtasks are numerous, interdependent, or unpredictable up front. Anthropic frames this as *workflows* (predefined paths) vs *agents* (the model directs itself dynamically) and stresses keeping the plan **transparent** (the visible task list). OpenMate encodes this by defaulting `Agent.planner` to ReAct and making `PlanExecute` opt-in.

**Automatic selection — two designs.** (1) *Hard routing:* `StrategyRouter` (Phase 4) classifies the task by difficulty/length signals and picks ReAct vs. `PlanExecute` vs. `TreeOfThought` before the loop runs. (2) *Soft, Claude-style:* keep ReAct as the strategy and expose planning as a **tool** (a `plan`/`TaskCreate` tool the model calls when it judges the task complex), so the model self-selects *within* one loop — `Plan` already lives on `RunState` ([01](01-domain-model-and-kernel.md)), so this needs only a planning tool ([04](04-tools-and-mcp.md)). OpenMate supports both; the router is the current default.

## Testing & verification

- **Outcome parity:** native vs. prompted ReAct produce equivalent outcomes on a fixed cassette.
- **Plan repair:** inject a failing step; assert the plan patches rather than restarts and still completes.
- **Reflection lift:** measure accuracy delta with/without `Reflexion` on a held-out task suite ([11](11-observability-and-evaluation.md)) — the claim must be measurable.
- **Search correctness:** ToT on a toy puzzle with a known optimum finds it within beam/depth bounds.
- **Soft selection:** with planning tools available, the agent creates a plan for a 5-step task and skips it for a 1-step task (the description-driven heuristic works); the pinned plan persists in-context across turns.

## Trade-offs & open questions

Reflection cost vs. quality (cap revisions; gate on task value). When to re-plan vs. patch (divergence threshold tuning). Whether to expose `scratch` typing per strategy. ToT/MCTS rarely pay off outside verifiable domains — keep them clearly optional. Soft planning-as-a-tool vs. hard `PlanExecute` strategy — model judgment + simplicity vs. enforced planner/executor split and structural drift control (see Phase 1); and whether planning tools should mutate `RunState` directly or return effects.
