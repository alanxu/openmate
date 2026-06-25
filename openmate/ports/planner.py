"""Planning data types — the plan-as-data the planning tools manipulate (docs/05).

Phase 1 of docs/05 stores the plan as **structured, inspectable data** with
per-step status and a dependency DAG. These types are the minimal slice that the
soft "planning-as-a-tool" path needs; the full ``ReasoningStrategy`` engine
(hard ``PlanExecute``) would also live behind this port.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

Status = Literal["pending", "active", "done", "failed", "skipped"]
_DONE = ("done", "skipped")


@dataclass
class PlanStep:
    id: str
    goal: str
    tool_hint: str | None = None
    deps: list[str] = field(default_factory=list)  # step ids that must finish first
    status: Status = "pending"
    result: Any = None
    attempts: int = 0


@dataclass
class Plan:
    steps: list[PlanStep]
    rationale: str = ""
    revision: int = 0

    def get(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def _is_done(self, step_id: str) -> bool:
        try:
            return self.get(step_id).status in _DONE
        except KeyError:
            return False

    def _ready(self) -> list[PlanStep]:
        return [
            s
            for s in self.steps
            if s.status == "pending" and all(self._is_done(d) for d in s.deps)
        ]

    def next_ready(self) -> PlanStep | None:
        """The first pending step whose dependencies are all satisfied."""
        ready = self._ready()
        return ready[0] if ready else None

    def ready_batch(self, max_parallel: int | None = None) -> list[PlanStep]:
        """All dependency-satisfied pending steps (capped to ``max_parallel``)."""
        ready = self._ready()
        return ready if max_parallel is None else ready[:max_parallel]

    def record(self, step_id: str, status: Status, result: Any = None) -> None:
        step = self.get(step_id)
        step.status = status
        step.attempts += 1
        if result is not None:
            step.result = result

    @property
    def done(self) -> bool:
        return all(s.status in _DONE for s in self.steps)

    @property
    def past_steps(self) -> list[tuple[PlanStep, Any]]:
        return [(s, s.result) for s in self.steps if s.status == "done"]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(
            steps=[PlanStep(**s) for s in d.get("steps", [])],
            rationale=d.get("rationale", ""),
            revision=d.get("revision", 0),
        )


def render_plan(plan: Plan) -> str:
    """A compact, model-legible rendering (what the planning tools echo back)."""
    marks = {"pending": "[ ]", "active": "[~]", "done": "[x]", "failed": "[!]", "skipped": "[-]"}
    lines = [f"Plan (rev {plan.revision}): {plan.rationale}".rstrip()]
    for s in plan.steps:
        dep = f" (after {', '.join(s.deps)})" if s.deps else ""
        lines.append(f"  {marks.get(s.status, '[ ]')} {s.id}: {s.goal}{dep}")
    return "\n".join(lines)
