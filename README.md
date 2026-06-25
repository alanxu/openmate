# OpenMate

A from-scratch, provider-agnostic AI agent — built on the ports-and-adapters
architecture described in [`docs/`](docs/). This repository contains the
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
  (`Message`/`Part`, `RunState`, `Usage`, the `Agent` facade + `Harness`), a
  synchronous event bus, the async **ReAct loop** ([`loop.py`](openmate/kernel/loop.py),
  the engine behind `Agent.run()`) with a hard step cap, sequential tool dispatch,
  and lossless JSON serialization of run state (the checkpoint).
- **Ports** ([`openmate/ports`](openmate/ports)) — the `Model`, `Tool`, `Store`,
  and `Tracer` interfaces the kernel depends on.
- **Adapters** ([`openmate/adapters`](openmate/adapters)) — a Claude/Anthropic
  model adapter (→ MiniMax) and a deterministic `FakeModel`; in-memory and
  **SQLite** checkpoint stores; a console tracer; native Python tools; an **MCP
  client** that adapts any MCP server's tools into OpenMate `Tool`s.
- **Tool providers & assembly** ([`openmate/tools`](openmate/tools),
  [`assemble()`](openmate/agent/assemble.py)) — a `ToolProvider` seam (Native,
  Shell, MCP) that `assemble()` resolves into a ready-to-run agent, owning
  provider lifecycle (e.g. closing MCP connections).
- **Skills** ([`openmate/skills`](openmate/skills)) — the **SKILL.md** system with
  three-level progressive disclosure: cards in the system prompt, full body on
  `load_skill`, bundled resources on demand.
- **Tools** — `calculator`, `current_time`, `read_file`, `list_directory`,
  `fetch_url`, a sandboxed `shell`, and (opt-in) `write_file`, plus a `@tool`
  decorator that derives a JSON schema from a function's type hints.
- **Gmail MCP server** ([`servers/gmail`](servers/gmail)) — a read-focused Gmail
  integration (search / get message / get thread / list labels / create draft)
  with OAuth + result shaping, plus a credential-free **fake** for offline runs.
- **RAG** ([`rag/`](rag), ports in [`openmate/ports/retriever.py`](openmate/ports/retriever.py))
  — ingestion (load → chunk → embed → index), dense retrieval over a **Chroma**
  vector store, **naive** and **agentic** RAG, an ingestion/retrieval **CLI**, and
  an **MCP server** exposing both.
- **Edges** — a CLI (`openmate run` / `openmate chat`) and runnable
  [`examples/`](examples), including a fully-offline **email assistant** and
  **RAG demo**.

Everything above Phase 0 in the design (interceptor chain, pluggable reasoning
strategies, RAG, multi-agent, full guardrail stack, OTel tracing) is intentionally
*not* in the MVP — the kernel is structured so those are additive layers. See
[`docs/README.md`](docs/README.md) for the full roadmap.

---

## Email assistant — MCP + skills, end to end (offline)

The design's headline example, runnable with **no credentials**: an agent
`assemble()`d over the Gmail MCP server and the email skills, driven by a scripted
model so the whole flow is deterministic.

```bash
pip install "mcp>=1.2"                 # the only extra the fake server needs
python examples/email_assistant.py
```

It launches [`servers/gmail/fake_server.py`](servers/gmail/fake_server.py) as a
subprocess, speaks MCP over stdio, loads the `triage-inbox` skill on demand,
searches the (seeded) unread mail, reads a message, and prints a ranked digest.

```python
from openmate import assemble, MCPProvider, SkillProvider
from openmate.adapters.tools.mcp_client import MCPServerSpec

gmail = MCPServerSpec(name="gmail", command=["python", "servers/gmail/server.py"],
                      namespace_prefix="gmail_")
async with assemble(
        name="email", system="You are Alan's email assistant.",
        model=default_model(), services=default_services(),
        providers=[
            MCPProvider([gmail], scope_allowlist=[          # least privilege
                "gmail_search", "gmail_get_message", "gmail_get_thread",
                "gmail_list_labels", "gmail_create_draft"]),
            SkillProvider(["./skills/email"]),
        ]) as agent:
    await agent.run("Triage my inbox")                       # never sends; drafts only
```

To go live, swap the fake for [`servers/gmail/server.py`](servers/gmail/server.py)
and supply Google OAuth credentials — see [`servers/gmail/README.md`](servers/gmail/README.md).
`gmail_search` and friends carry `readOnlyHint`, which the MCP client maps to
`side_effecting=False` (no approval gate); a send tool is intentionally absent.

---

## Retrieval (RAG) — naive and agentic

Ground the agent in your own documents. The [`rag/`](rag) package implements
[`docs/07`](docs/07-retrieval-rag.md): ingest a corpus, retrieve over a
**[Chroma](https://www.trychroma.com/)** vector store (the default open-source DB;
an in-memory store is the zero-dep fallback), and answer in two modes — **naive**
(one-shot retrieve→answer) and **agentic** (an agent loops retrieve→judge→re-query).

```bash
pip install "openmate[rag]"                      # adds chromadb

python -m rag.cli ingest docs/                # load → chunk → embed → index
python -m rag.cli query "how does the loop stop?"   # retrieval only (no LLM, offline)
python -m rag.cli answer "how does the loop stop?"  # naive RAG
python -m rag.cli agentic "compare kernel vs loop"  # agentic RAG (multi-step)
```

A fully-offline walk-through of all three modes (retrieval is real; generation uses
a scripted model) — no API key:

```bash
python examples/rag_demo.py
```

There's also an **MCP server** ([`rag/mcp_server.py`](rag/mcp_server.py)) exposing
`rag_search` (retrieval), `rag_answer` (naive), and `rag_agentic_answer` (agentic),
so any MCP client can use the knowledge base the same way it uses Gmail. Agentic
RAG is "just an OpenMate agent whose tools are retrievers" — `RetrieveTool` plus the
existing loop, no special engine. See [`rag/README.md`](rag/README.md) for the full
reference.

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
from openmate import Agent, default_model, default_services
from openmate.adapters.tools.builtin import read_only_tools

async def main():
    agent = Agent(
        name="assistant",
        model=default_model(),          # Claude adapter → MiniMax (one line to swap)
        instructions="You are a helpful assistant.",
        services=default_services(),
        tools=read_only_tools(),
    )
    result = await agent.run("What is 12 * 9?")   # the Agent facade drives the loop
    print(result.text)

asyncio.run(main())
```

`Agent` is the **facade** you hold: it composes a `model`, `instructions`, a
`Harness` (its environment — tools + policies) and `Services` (shared infra), and
`run()`/`stream()`/`resume()` delegate to the loop engine in
[`openmate/kernel/loop.py`](openmate/kernel/loop.py).

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

## Use it from Jupyter

A thin IPython extension (`openmate/jupyter_magic`) wraps the RAG and MCP APIs
as line magics so you can explore interactively. See
[**`docs/jupyter-magics.md`**](docs/jupyter-magics.md) for the full reference;
the 30-second version:

```bash
pip install -e ".[rag-ui]"      # adds jupyterlab + ipykernel
python -m ipykernel install --user --name=openmate --display-name="OpenMate (.venv)"
.venv/bin/jupyter lab
```

```python
%load_ext openmate.jupyter_magic

%rag_ingest docs/
docs = %rag_query explain the agent loop --k 3

%mcp_add fake_gmail python3 servers/gmail/server.py
%mcp_list fake_gmail
%mcp_call fake_gmail list_messages {"max_results": 5}
```

A worked walkthrough lives at
[`examples/notebooks/openmate_demo.ipynb`](examples/notebooks/openmate_demo.ipynb).

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
├── kernel/        # Layer 0 — types (data), agent (facade + Harness), events, loop, executor, codec
├── ports/         # Layer 1 — Model · Tool · Store · Tracer · Retriever interfaces
├── adapters/      # Layer 2 — implementations of the ports
│   ├── models/    #   anthropic.py (→ MiniMax), fake.py
│   ├── stores/    #   memory.py, sqlite.py
│   ├── tracers/   #   console.py
│   └── tools/     #   native.py (@tool), builtin.py, mcp_client.py
├── tools/         # ToolProvider seam — Native · Shell · MCP providers
├── skills/        # the SKILL.md system — registry, loader, SkillProvider
├── agent/         # assemble() — providers → a runnable Agent
├── config.py      # env loading + default model/services wiring
└── cli.py         # the CLI edge
rag/               # RAG — loaders, chunking, embedders, Chroma store, retrievers,
                   #   pipeline, naive+agentic generate, CLI, MCP server
servers/gmail/     # the Gmail MCP server (real) + fake_server.py (offline)
skills/email/      # triage-inbox · summarize-thread (SKILL.md procedures)
examples/          # minimal_react · custom_tool · offline_fake · email_assistant · rag_demo
tests/             # kernel, loop, tools, mcp, skills, providers, rag, adapter translation
docs/           # the full design docs this MVP is derived from
```

---

## Testing

```bash
pytest                 # 59 tests, all offline (no API key needed)
```

The suite covers the serialization round-trip, the ReAct loop (tool call →
result → final answer), the step-cap loop guard, recoverable tool errors,
cross-turn memory, run determinism, streaming, the Anthropic↔MiniMax translation
layer (stubbed client, no network), the MCP client (annotation→spec mapping,
result shaping, a real stdio round-trip against the fake Gmail server), the
skills system (frontmatter, discovery, progressive disclosure), `assemble()`
provider composition + least-privilege scoping, and RAG (chunking idempotency,
deterministic embeddings, Chroma + in-memory stores, dense retrieval, naive +
agentic generation, and the RAG MCP server over stdio).

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
