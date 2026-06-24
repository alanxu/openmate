"""The Tool port — how an agent senses and affects the world.

Deliberately tiny: MCP servers, other frameworks' tools, and sub-agents all
become ``Tool``s via adapters. A tool's ``description`` and ``parameters`` are
model-facing prompt surface — designed, not dashed off.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..kernel.types import Part, RunContext


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str  # the model reads this — it is prompt surface
    parameters: dict  # JSON Schema for args
    side_effecting: bool = True  # read-only tools may skip approval
    timeout_s: float = 30.0
    idempotent: bool = False  # safe to retry/replay


@dataclass
class ToolResult:
    content: list["Part"]
    is_error: bool = False  # model-legible failure, recoverable within the loop
    retriable: bool = False


@runtime_checkable
class Tool(Protocol):
    spec: ToolSpec

    async def invoke(self, args: dict, ctx: "RunContext") -> ToolResult: ...
