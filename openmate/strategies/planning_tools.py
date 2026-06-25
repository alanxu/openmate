"""Plan-and-execute, the *soft* way — planning as tools (docs/05 §"Planning as a tool").

Rather than a dedicated ``PlanExecute`` strategy/engine, the agent runs the plain
ReAct loop and is given two tools: ``create_plan`` lays out a step DAG up front,
``update_plan`` marks step status or revises the plan as reality diverges. The
model self-selects whether to plan (the heuristic lives in the tool description),
so this needs **no kernel/loop change** — only these tools.

The plan is stored as JSON-able data in ``RunState.scratch["plan"]`` (so it's
checkpointed and resumable for free); ``get_plan`` reads it back as a :class:`Plan`.
"""

from __future__ import annotations

from openmate.kernel.types import RunContext, TextPart
from openmate.ports.planner import Plan, PlanStep, render_plan
from openmate.ports.tool import ToolResult, ToolSpec

_PLAN_KEY = "plan"


def get_plan(state) -> Plan | None:
    data = state.scratch.get(_PLAN_KEY)
    return Plan.from_dict(data) if data else None


def _save(state, plan: Plan) -> None:
    state.scratch[_PLAN_KEY] = plan.to_dict()


class CreatePlanTool:
    """``create_plan(rationale, steps)`` — write a step-by-step plan before acting."""

    spec = ToolSpec(
        name="create_plan",
        description=(
            "Lay out a step-by-step plan BEFORE acting. Use for non-trivial work "
            "(3+ dependent steps); skip it for simple tasks. Each step has a goal and "
            "may declare deps (ids of steps that must finish first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "rationale": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "goal": {"type": "string"},
                            "tool_hint": {"type": "string"},
                            "deps": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["goal"],
                    },
                },
            },
            "required": ["steps"],
        },
        side_effecting=False,
        idempotent=True,
    )

    async def invoke(self, args: dict, ctx: RunContext) -> ToolResult:
        raw = (args or {}).get("steps") or []
        if not raw:
            return ToolResult([TextPart("create_plan needs at least one step")], is_error=True)
        steps = [
            PlanStep(
                id=s.get("id") or f"s{i + 1}",
                goal=s["goal"],
                tool_hint=s.get("tool_hint"),
                deps=list(s.get("deps") or []),
            )
            for i, s in enumerate(raw)
        ]
        plan = Plan(steps=steps, rationale=(args or {}).get("rationale", ""))
        _save(ctx.state, plan)
        return ToolResult([TextPart(render_plan(plan))])


class UpdatePlanTool:
    """``update_plan(ops)`` — mark step status or revise the plan (replan)."""

    spec = ToolSpec(
        name="update_plan",
        description=(
            "Update the plan as you learn more: mark a step done/failed/active, or "
            "revise it by adding/editing a step. Ops: "
            "{step, status?, result?} to set status; "
            "{op:'add', id?, goal, deps?} to add a step; "
            "{op:'edit', step, goal} to change a step's goal. Adding or editing "
            "(a replan) bumps the plan revision."
        ),
        parameters={
            "type": "object",
            "properties": {"ops": {"type": "array", "items": {"type": "object"}}},
            "required": ["ops"],
        },
        side_effecting=False,
        idempotent=False,
    )

    async def invoke(self, args: dict, ctx: RunContext) -> ToolResult:
        plan = get_plan(ctx.state)
        if plan is None:
            return ToolResult(
                [TextPart("no plan yet — call create_plan first")], is_error=True
            )
        structural = False
        for op in (args or {}).get("ops") or []:
            kind = op.get("op", "set")
            if kind == "add":
                plan.steps.append(
                    PlanStep(
                        id=op.get("id") or f"s{len(plan.steps) + 1}",
                        goal=op["goal"],
                        tool_hint=op.get("tool_hint"),
                        deps=list(op.get("deps") or []),
                    )
                )
                structural = True
            else:  # set / edit an existing step
                try:
                    step = plan.get(op["step"])
                except KeyError:
                    return ToolResult(
                        [TextPart(f"no such step: {op.get('step')!r}")], is_error=True
                    )
                if op.get("op") == "edit" and "goal" in op:
                    step.goal = op["goal"]
                    structural = True
                if "status" in op:
                    step.status = op["status"]
                if "result" in op:
                    step.result = op["result"]
        if structural:
            plan.revision += 1
        _save(ctx.state, plan)
        return ToolResult([TextPart(render_plan(plan))])


def planning_tools() -> list:
    """The two tools that turn the ReAct loop into soft plan-and-execute."""
    return [CreatePlanTool(), UpdatePlanTool()]
