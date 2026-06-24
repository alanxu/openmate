"""A human-readable tracer that prints the event stream to the console.

It subscribes to the ``EventBus`` and renders each event, so a local run is
fully observable: you see every model turn, tool call, result, and the final
usage summary. ``verbose=True`` adds low-level model-request events.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import ContextManager

from ...kernel.events import (
    CheckpointSaved,
    Event,
    MessageAdded,
    ModelRequested,
    RunFinished,
    RunStarted,
    ToolCallRequested,
    ToolReturned,
)
from ...kernel.types import TextPart, ToolCallPart


def _fmt_args(args: dict) -> str:
    items = ", ".join(f"{k}={v!r}" for k, v in (args or {}).items())
    return items if len(items) <= 120 else items[:117] + "..."


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
                print(f"\033[2m  → model · {event.n_messages} msgs · {event.n_tools} tools\033[0m")
        elif isinstance(event, MessageAdded):
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
