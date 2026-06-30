"""The CLI edge — a thin shell over the ``Agent`` facade.

    openmate run "What is 12 * 9?"        # one-shot
    openmate chat                          # interactive REPL (remembers the conversation)
    openmate chat --store sqlite --thread mychat   # persistent, resumable thread
    openmate ui                            # browser UI: new task + history + trajectory view

Run ``openmate --help`` for all options.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from .adapters.stores.memory import InMemoryStore
from .adapters.stores.sqlite import SQLiteStore
from .adapters.tools.builtin import all_tools, read_only_tools
from .config import DEFAULT_MODEL, default_model, default_services
from .kernel.agent import Agent
from .kernel.errors import OpenMateError
from .kernel.types import Services

DEFAULT_INSTRUCTIONS = (
    "You are OpenMate, a concise and helpful AI assistant. "
    "You can call tools to do arithmetic, read and list files, fetch URLs, and tell the time. "
    "Think step by step and use tools when they help. "
    "When you have the answer, reply directly without calling a tool."
)


def _build_agent(args, services: Services) -> Agent:
    tools = [] if args.no_tools else (all_tools() if args.allow_write else read_only_tools())
    return Agent(
        name="openmate",
        model=default_model(args.model),
        instructions=DEFAULT_INSTRUCTIONS,
        services=services,
        tools=tools,
        max_steps=args.max_steps,
    )


def _build_store(args):
    if args.store == "sqlite":
        return SQLiteStore(args.db)
    return InMemoryStore()


async def _run_once(args) -> int:
    services = default_services(store=_build_store(args), verbose=args.verbose)
    agent = _build_agent(args, services)
    result = await agent.run(args.prompt, thread_id=args.thread)
    if not args.verbose:  # the tracer already printed the answer in verbose mode
        print(result.text)
    return 0 if result.ok else 1


async def _chat(args) -> int:
    services = default_services(store=_build_store(args), verbose=args.verbose)
    agent = _build_agent(args, services)
    thread_id = args.thread or services.new_id()
    print(f"OpenMate chat · model={agent.model.name} · thread={thread_id}")
    print("Type your message; 'exit' or Ctrl-D to quit.\n")
    while True:
        try:
            text = input("\033[34myou ›\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text.lower() in ("exit", "quit"):
            break
        await agent.run(text, thread_id=thread_id)
        print()
    return 0


def _ui(args) -> int:
    """Launch the browser UI — see ``ui/server.py`` at the project root.

    Imported lazily (not at module top) so ``openmate run``/``chat`` never pay
    for or require starlette/uvicorn, which are only needed by this subcommand.
    """
    import importlib.util
    import sys
    from pathlib import Path

    ui_root = Path(__file__).resolve().parent.parent  # project root: parent of openmate/
    server_path = ui_root / "ui" / "server.py"
    if not server_path.exists():
        print(
            f"error: UI server not found at {server_path}. "
            "The ui/ folder must sit alongside the openmate/ package.",
            file=sys.stderr,
        )
        return 2
    spec = importlib.util.spec_from_file_location("openmate_ui_server", server_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.run(
        host=args.host,
        port=args.port,
        db=args.db,
        allow_write=args.allow_write,
        model=args.model,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default=None, help=f"model name (default: $OPENMATE_MODEL or {DEFAULT_MODEL})")
    common.add_argument("--store", choices=["memory", "sqlite"], default="memory", help="checkpoint store")
    common.add_argument("--db", default="openmate.sqlite", help="sqlite db path (with --store sqlite)")
    common.add_argument("--thread", default=None, help="thread id (resume a conversation)")
    common.add_argument("--max-steps", type=int, default=12, help="max tool-use rounds per turn")
    common.add_argument("--no-tools", action="store_true", help="run without any tools")
    common.add_argument("--allow-write", action="store_true", help="enable the side-effecting write_file tool")
    common.add_argument("--verbose", action="store_true", help="show model-request and checkpoint events")
    common.add_argument("--log", action="store_true", help="log the full event stream (raw model I/O) to ~/.openmate/logs")

    p = argparse.ArgumentParser(prog="openmate", description="Run the OpenMate agent locally.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="answer a single prompt and exit")
    run.add_argument("prompt", help="the prompt to answer")
    sub.add_parser("chat", parents=[common], help="start an interactive chat session")

    ui = sub.add_parser(
        "ui", help="launch the browser UI (new task, history, collapsible trajectory view)"
    )
    ui.add_argument("--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)")
    ui.add_argument("--port", type=int, default=8765, help="bind port (default: 8765)")
    ui.add_argument("--db", default="openmate.sqlite", help="sqlite db path (threads always persist)")
    ui.add_argument("--model", default=None, help=f"model name (default: $OPENMATE_MODEL or {DEFAULT_MODEL})")
    ui.add_argument("--allow-write", action="store_true", help="enable the side-effecting write_file tool")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if getattr(args, "log", False):
        os.environ["OPENMATE_LOG"] = "1"  # one switch; default_services attaches the logger
    try:
        if args.command == "run":
            return asyncio.run(_run_once(args))
        if args.command == "ui":
            return _ui(args)
        return asyncio.run(_chat(args))
    except OpenMateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
