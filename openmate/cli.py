"""The CLI edge — a thin shell over the ``Runtime``.

    openmate run "What is 12 * 9?"        # one-shot
    openmate chat                          # interactive REPL (remembers the conversation)
    openmate chat --store sqlite --thread mychat   # persistent, resumable thread

Run ``openmate --help`` for all options.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from .adapters.stores.memory import InMemoryStore
from .adapters.stores.sqlite import SQLiteStore
from .adapters.tools.builtin import all_tools, read_only_tools
from .config import DEFAULT_MODEL, default_model, default_services
from .kernel.errors import OpenMateError
from .kernel.runtime import Runtime
from .kernel.types import Agent

DEFAULT_INSTRUCTIONS = (
    "You are OpenMate, a concise and helpful AI assistant. "
    "You can call tools to do arithmetic, read and list files, fetch URLs, and tell the time. "
    "Think step by step and use tools when they help. "
    "When you have the answer, reply directly without calling a tool."
)


def _build_agent(args) -> Agent:
    tools = [] if args.no_tools else (all_tools() if args.allow_write else read_only_tools())
    return Agent(
        name="openmate",
        model=default_model(args.model),
        instructions=DEFAULT_INSTRUCTIONS,
        tools=tools,
        max_steps=args.max_steps,
    )


def _build_store(args):
    if args.store == "sqlite":
        return SQLiteStore(args.db)
    return InMemoryStore()


async def _run_once(args) -> int:
    services = default_services(store=_build_store(args), verbose=args.verbose)
    agent = _build_agent(args)
    runtime = Runtime(services)
    result = await runtime.run(agent, args.prompt, thread_id=args.thread)
    if not args.verbose:  # the tracer already printed the answer in verbose mode
        print(result.text)
    return 0 if result.ok else 1


async def _chat(args) -> int:
    services = default_services(store=_build_store(args), verbose=args.verbose)
    agent = _build_agent(args)
    runtime = Runtime(services)
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
        await runtime.run(agent, text, thread_id=thread_id)
        print()
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

    p = argparse.ArgumentParser(prog="openmate", description="Run the OpenMate agent locally.")
    sub = p.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", parents=[common], help="answer a single prompt and exit")
    run.add_argument("prompt", help="the prompt to answer")
    sub.add_parser("chat", parents=[common], help="start an interactive chat session")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "run":
            return asyncio.run(_run_once(args))
        return asyncio.run(_chat(args))
    except OpenMateError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
