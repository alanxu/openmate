"""The Model port — the swappable LLM boundary.

The single rule: no code outside ``adapters/models/*`` knows which provider is in
use. The port normalizes generation, tool calling, and usage accounting, and
advertises capabilities so the loop can adapt instead of assuming a floor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, AsyncIterator, Literal, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..kernel.types import Message, Usage
    from .tool import ToolSpec


@dataclass(frozen=True)
class ModelCapabilities:
    tool_calling: bool = True
    parallel_tools: bool = True
    structured_output: bool = False
    vision: bool = False
    thinking: bool = False
    prompt_caching: bool = False
    streaming: bool = True
    max_context: int = 200_000
    max_output: int = 8192


@dataclass
class ModelRequest:
    messages: list["Message"]
    tools: list["ToolSpec"] | None = None
    temperature: float | None = None
    max_tokens: int = 2048
    stop: list[str] | None = None
    extra: dict = field(default_factory=dict)  # namespaced provider escape hatch


@dataclass
class ModelResponse:
    message: "Message"  # text and/or tool calls
    usage: "Usage"
    finish_reason: Literal["stop", "tool_calls", "length", "filter"]
    raw: Any = None  # provider payload, for debugging only


@dataclass
class StreamDelta:
    """One incremental chunk yielded by ``Model.stream``.

    ``text``/``thinking`` carry a string fragment; the terminal ``done`` delta
    carries the fully-assembled ``ModelResponse`` in ``data`` — so a streaming
    caller reassembles to exactly what ``generate`` would have returned.
    """

    kind: Literal["text", "thinking", "tool_call_partial", "usage", "done"]
    data: Any


@runtime_checkable
class Model(Protocol):
    name: str
    capabilities: ModelCapabilities

    async def generate(self, req: ModelRequest) -> ModelResponse: ...

    # Optional: token-level streaming. Yields StreamDelta chunks, ending with a
    # ``done`` delta carrying the final ModelResponse. Only used when the agent
    # opts in (``stream_model=True``); ``generate`` remains the default path.
    def stream(self, req: ModelRequest) -> AsyncIterator[StreamDelta]: ...
