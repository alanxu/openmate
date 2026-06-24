# OpenMate

A from-scratch, provider-agnostic AI agent — built on the ports-and-adapters
architecture described in [`designs/`](designs/). This repository contains the
**MVP** (Phase 0 of the design): a small, typed kernel and a clean async ReAct
loop that you can actually run.

Out of the box it runs as a **Claude-style agent on a MiniMax model**: OpenMate's
Claude/Anthropic model adapter talks to MiniMax's Anthropic-compatible API
endpoint, so the same code that would drive Claude drives MiniMax with nothing
but a base-URL and key change.

```
you › What is 1234 * 5678? Use the calculator, then state the result.
  🔧 calculator(expression='1234 * 5678')
     ↳ ok (0ms) 7006652
🤖 1234 × 5678 = 7,006,652.
■ done (natural) · 1 steps · 1107 tokens
```

---

## What the MVP implements

The foundational (Phase 0) slice across the design, end to end:

- **Kernel** ([`openmate/kernel`](openmate/kernel)) — the domain types
  (`Message`/`Part`, `RunState`, `Usage`, `Agent`), a synchronous event bus, the
  async **ReAct loop** with a hard step cap, sequential tool dispatch, and
  lossless JSON serialization of run state (the checkpoint).
- **Ports** ([`openmate/ports`](openmate/ports)) — the `Model`, `Tool`, `Store`,
  and `Tracer` interfaces the kernel depends on.
- **Adapters** ([`openmate/adapters`](openmate/adapters)) — a Claude/Anthropic
  model adapter (→ MiniMax) and a deterministic `FakeModel`; in-memory and
  **SQLite** checkpoint stores; a console tracer; native Python tools.
- **Tools** — `calculator`, `current_time`, `read_file`, `list_directory`,
  `fetch_url`, and (opt-in) `write_file`, plus a `@tool` decorator that derives a
  JSON schema from a function's type hints.
- **Edges** — a CLI (`openmate run` / `openmate chat`) and runnable
  [`examples/`](examples).

Everything above Phase 0 in the design (interceptor chain, pluggable reasoning
strategies, RAG, multi-agent, full guardrail stack, OTel tracing) is intentionally
*not* in the MVP — the kernel is structured so those are additive layers. See
[`designs/README.md`](designs/README.md) for the full roadmap.

---

## Run it locally

### 1. Requirements

- Python **3.10+** (tested on 3.14)

### 2. Install

```bash
git clone <this-repo> && cd openmate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 3. Configure the key

A ready-to-use [`.env`](.env) is already present (and git-ignored) with the
MiniMax key wired in. It looks like this:

```bash
ANTHROPIC_API_KEY=sk-...            # your MiniMax key, used as the Anthropic key
ANTHROPIC_BASE_URL=https://api.minimax.io/anthropic
OPENMATE_MODEL=MiniMax-M2
```

To use your own key, copy [`.env.example`](.env.example) to `.env` and edit it.
OpenMate's `.env` is **authoritative** for local runs — it overrides any
`ANTHROPIC_*` variables already set in your shell, so a global Anthropic key
won't shadow it.

### 4. Run

```bash
# one-shot
openmate run "What is the 12th Fibonacci number? Use the calculator."

# interactive chat (remembers the conversation within the session)
openmate chat

# persistent, resumable thread (checkpointed to SQLite across processes)
openmate chat --store sqlite --db mychat.sqlite --thread work

# see every model request and checkpoint
openmate run "Summarize the README" --verbose
```

`openmate` is the installed console script; `python -m openmate ...` works too.

Useful flags: `--model`, `--store {memory,sqlite}`, `--db`, `--thread`,
`--max-steps`, `--no-tools`, `--allow-write`, `--verbose`. Run `openmate run --help`.

### 5. Run without any API key (offline)

The full loop runs against a deterministic `FakeModel`, no network required:

```bash
python examples/offline_fake.py
```

---

## Use it from Python

```python
import asyncio
from openmate import Agent, Runtime, default_model, default_services
from openmate.adapters.tools.builtin import read_only_tools

async def main():
    agent = Agent(
        name="assistant",
        model=default_model(),          # Claude adapter → MiniMax (one line to swap)
        instructions="You are a helpful assistant.",
        tools=read_only_tools(),
    )
    result = await Runtime(default_services()).run(agent, "What is 12 * 9?")
    print(result.text)

asyncio.run(main())
```

Defining your own tool is just a decorated function (see
[`examples/custom_tool.py`](examples/custom_tool.py)):

```python
from openmate.adapters.tools.native import tool

@tool(side_effecting=False)
def word_count(text: str) -> int:
    """Count the whitespace-separated words in a piece of text."""
    return len(text.split())
```

---

## Swapping the provider

The point of the Model port: the kernel never knows which provider is in use.

- **Use real Claude** — set `ANTHROPIC_BASE_URL=https://api.anthropic.com`, put a
  real Anthropic key in `ANTHROPIC_API_KEY`, and `OPENMATE_MODEL=claude-...`.
- **A different MiniMax model** — set `OPENMATE_MODEL=MiniMax-M2.5` (or `M3`).
- **Another Anthropic-compatible gateway** — point `ANTHROPIC_BASE_URL` at it.

No application code changes — only configuration.

---

## Project layout

```
openmate/
├── kernel/        # Layer 0 — types, events, the loop, executor, codec
├── ports/         # Layer 1 — Model · Tool · Store · Tracer interfaces
├── adapters/      # Layer 2 — implementations of the ports
│   ├── models/    #   anthropic.py (→ MiniMax), fake.py
│   ├── stores/    #   memory.py, sqlite.py
│   ├── tracers/   #   console.py
│   └── tools/     #   native.py (@tool), builtin.py
├── config.py      # env loading + default model/services wiring
└── cli.py         # the CLI edge
examples/          # minimal_react · custom_tool · offline_fake
tests/             # kernel, runtime, tools, adapter translation
designs/           # the full design docs this MVP is derived from
```

---

## Testing

```bash
pytest                 # 17 tests, all offline (no API key needed)
```

The suite covers the serialization round-trip, the ReAct loop (tool call →
result → final answer), the step-cap loop guard, recoverable tool errors,
cross-turn memory, run determinism, and the Anthropic↔MiniMax translation layer
(with a stubbed client, so no network).

---

## How the MiniMax wiring works

MiniMax exposes an **Anthropic-compatible** Messages API at
`https://api.minimax.io/anthropic`. OpenMate's
[`AnthropicModel`](openmate/adapters/models/anthropic.py) uses the official
`anthropic` SDK with `base_url` pointed there and the MiniMax key passed as the
API key. The adapter owns the translation between OpenMate's `Message`/`Part`
values and the Anthropic wire format (system prompt extraction, `tool_use` /
`tool_result` blocks, usage accounting). Because all of that is confined to the
adapter, the kernel, tools, and loop are completely provider-neutral.

> Security note: `.env` holds a real key and is git-ignored. Don't commit keys.
