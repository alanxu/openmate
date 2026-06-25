"""Reasoning strategies beyond the default ReAct loop (docs/05).

* ``planning_tools`` — the soft, Claude-style **plan-and-execute** path: two tools
  the model calls within the normal ReAct loop to lay out and update a plan.
* ``tot`` — a self-contained **Tree-of-Thoughts** search engine (branch → evaluate
  → prune → best leaf), usable directly or wrapped as a tool.
"""

from .planning_tools import CreatePlanTool, UpdatePlanTool, get_plan, planning_tools
from .tot import SearchResult, TreeOfThought

__all__ = [
    "CreatePlanTool",
    "SearchResult",
    "TreeOfThought",
    "UpdatePlanTool",
    "get_plan",
    "planning_tools",
]
