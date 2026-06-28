"""A human-readable tracer that prints the event stream to the console.

It subscribes to the ``EventBus`` and renders each event, so a local run is
observable. One knob, ``verbose``:

- ``verbose=False`` (default) — a concise narration: the assistant's text, tool
  calls, results, and the final usage line.
- ``verbose=True`` — adds the **full model request and response payloads** (the
  prompt sent and the message received) plus checkpoints. This is display logic
  over the ``ModelRequested`` / ``ModelResponded`` events — the model I/O is
  captured as data at the loop boundary; this consumer just renders it.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

from ...kernel.events import (
    CheckpointSaved,
    Event,
    MessageAdded,
    ModelRequested,
    ModelResponded,
    RunFinished,
    RunStarted,
    ToolCallRequested,
    ToolReturned,
)
from ...kernel.types import TextPart, ThinkingPart, ToolCallPart, ToolResultPart


def _fmt_args(args: dict) -> str:
    items = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return items if len(items) <= 120 else items[:117] + "..."


def _render_parts(content, width: int = 1500) -> str:
    """Render a message's content parts as readable text (for the model-I/O dump)."""
    out: list[str] = []
    for p in content:
        if isinstance(p, TextPart):
            t = p.text
            out.append(t if len(t) <= width else t[:width] + f"...[{len(t)} chars]")
        elif isinstance(p, ThinkingPart):
            t = p.text
            out.append(f"💭 thinking: {t if len(t) <= width else t[:width] + f'...[{len(t)} chars]'}")
        elif isinstance(p, ToolCallPart):
            out.append(f"⮑ tool_call {p.name}({p.args})")
        elif isinstance(p, ToolResultPart):
            txt = "".join(c.text for c in p.content if isinstance(c, TextPart))
            if len(txt) > width:
                txt = txt[:width] + f"...[{len(txt)} chars]"
            out.append(f"⮐ tool_result[{p.call_id}{' ERROR' if p.is_error else ''}] {txt}")
    return "\n      ".join(s for s in out if s)


class ConsoleTracer:
    def __init__(self, verbose: bool = False) -> None:
        self.verbose = verbose

    def attach(self, bus) -> "ConsoleTracer":
        bus.subscribe(self.record)
        return self

    def span(self, name: str, **attrs) -> ContextManager:
        return nullcontext()

    def record(self, event: Event) -> None:
        if isinstance(event, RunStarted):
            print("\n\033[2m▶ run started\033[0m")
        elif isinstance(event, ModelRequested):
            if self.verbose:
                tools = (
                    " · tools: " + ", ".join(t.name for t in event.request.tools)
                    if event.request.tools
                    else ""
                )
                print(f"\033[35m╭── model request · {event.n_messages} msgs{tools}\033[0m")
                for m in event.request.messages:
                    print(f"\033[35m│\033[0m \033[1m{m.role}:\033[0m {_render_parts(m.content)}")
        elif isinstance(event, ModelResponded):
            if self.verbose:
                u = event.response.usage
                print(
                    f"\033[34m╰── response · finish={event.response.finish_reason} · "
                    f"tokens in={u.prompt_tokens} out={u.completion_tokens} · {event.ms:.0f}ms\033[0m\n"
                    f"      {_render_parts(event.response.message.content)}"
                )
        elif isinstance(event, MessageAdded):
            # In verbose mode the response block already shows the assistant
            # message verbatim, so don't print it twice.
            if not self.verbose:
                self._print_assistant(event.message)
        elif isinstance(event, ToolCallRequested):
            c = event.call
            print(f"\033[36m  🔧 {c.name}(\033[0m{_fmt_args(c.args)}\033[36m)\033[0m")
        elif isinstance(event, ToolReturned):
            status = "\033[31merror\033[0m" if event.result.is_error else "\033[32mok\033[0m"
            preview = "".join(
                p.text for p in event.result.content if isinstance(p, TextPart)
            ).replace("\n", " ")
            if len(preview) > 100:
                preview = preview[:97] + "..."
            print(f"\033[2m     ↳ {status} ({event.ms:.0f}ms) {preview}\033[0m")
        elif isinstance(event, CheckpointSaved):
            if self.verbose:
                print(f"\033[2m  💾 checkpoint rev={event.rev}\033[0m")
        elif isinstance(event, RunFinished):
            r = event.result
            u = r.usage
            print(
                f"\033[2m■ {r.status} ({r.reason}) · {r.steps} steps · "
                f"{u.total_tokens} tokens\033[0m"
            )

    @staticmethod
    def _print_assistant(message) -> None:
        text = message.text.strip()
        if text:
            print(f"\033[1m🤖 {text}\033[0m")
