# OpenMate Jupyter magics

A thin IPython extension (`openmate/jupyter_magic`) that lets you drive OpenMate's
RAG and MCP integrations from inside a Jupyter notebook — `%rag_ingest`,
`%rag_query`, `%mcp_add`, `%mcp_list`, `%mcp_call`.

It exists because neither Chroma nor MCP ship official IPython magics. Rather
than depend on a fragile third-party extension that targets one version of
one of them, we wrote a small wrapper over OpenMate's *own* `rag/` and
`openmate.adapters.tools.mcp_client` APIs. The magic surface is always in
sync with the rest of the project, and it's about 250 lines.

---

## Install

The magics live in the core `openmate` package; the only thing missing on a
fresh checkout is Jupyter itself.

```bash
# from a fresh checkout
git clone <this-repo> && cd openmate
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[rag-ui]"        # rag + chroma + jupyter + jupyterlab + ipykernel

# register the venv as a Jupyter kernel so it shows up in the launcher
python -m ipykernel install --user --name=openmate --display-name="OpenMate (.venv)"

# go
.venv/bin/jupyter lab
```

If you already have a kernel from earlier runs, you can skip the
`ipykernel install` step — it'll just re-register.

---

## Quickstart

In the first cell:

```python
%load_ext openmate.jupyter_magic
```

You'll see a short banner reminding you of the available magics. After that,
treat every cell as a place where you can mix normal Python with `%rag_*` and
`%mcp_*` magics.

```python
# 1. Ingest anything you can point at — a file, a directory, a glob.
%rag_ingest docs/

# 2. Query it. Use --k N to change the number of returned docs.
docs = %rag_query explain the agent loop --k 3
docs[:2]                            # show the top two as plain dicts

# 3. Talk to an MCP server. The subprocess keeps running between calls.
%mcp_add fake_gmail python3 servers/gmail/server.py
%mcp_list fake_gmail
%mcp_call fake_gmail list_messages {"max_results": 5}
```

That's the whole loop.

---

## Magic reference

### `%rag_ingest <path>`

Walks `path` (file or directory), chunks each file with `FixedWindowChunker`,
embeds with the configured embedder, writes into the configured vector store.
Prints how many chunks landed where.

```python
%rag_ingest docs/                # a directory
%rag_ingest README.md              # a single file
%rag_ingest ~/Documents/notes/     # an absolute path
```

The default embedder, store, and DB location come from env vars — see
[Configuration](#configuration) below.

### `%rag_query <question> [--k N]`

Embeds the question, asks the configured retriever for the top-k nearest
neighbours, prints a formatted view via `rag.tools.format_documents`, and
**returns a list of dicts** so you can use the result from Python:

```python
docs = %rag_query what is the tools port --k 5
for d in docs:
    print(d["id"], d["score"], d["text"][:80])
```

Each dict has `id`, `score`, `text` (truncated to 400 chars), `metadata`.

> **Gotcha:** IPython treats `?` at the end of a line as a help-introspection
> request (`object?` triggers `pinfo`). Avoid ending queries with `?`. Reword
> to a statement form, or escape it.

### `%mcp_add <name> <command> [args...]`

Connects to an MCP server by spawning it as a stdio subprocess. `<name>` is
the handle you'll use for subsequent `%mcp_list` / `%mcp_call` invocations.

```python
# in-repo fake server, no OAuth, no internet
%mcp_add fake_gmail python3 servers/gmail/server.py

# the real OSS Gmail server (needs the OAuth dance — see .env.example)
%mcp_add gmail npx -y @gongrzhe/server-gmail-autoauth-mcp

# anything that speaks MCP over stdio
%mcp_add postgres npx -y @modelcontextprotocol/server-postgres
```

The subprocess is launched exactly once and stays alive for the lifetime of
the kernel. Restarting the kernel closes it.

> **macOS gotcha:** don't end your command with `python` — Homebrew puts only
> `python3` on PATH, and the subprocess inherits the kernel's environment. Use
> `python3` (which on Linux usually exists and points at your venv), or the
> absolute path to your venv's `bin/python`.

### `%mcp_list [server_name]`

Lists the tools a connected MCP server exposes, with their descriptions.
Returns a list of `{name, remote_name, description, schema}` dicts.

```python
%mcp_list fake_gmail
# - fake_gmail__list_messages: List recent messages in the user's mailbox.
# - fake_gmail__get_message:   Fetch a single message by ID.
# - fake_gmail__create_draft:  Create a draft email.
```

If you've set `namespace_prefix` on the `MCPServerSpec` (rare — only for
collision avoidance), the listed `name` is the namespaced form; `remote_name`
is what gets sent on the wire.

### `%mcp_call <server> <tool> '<json_args>'`

Calls a tool. The trailing JSON is parsed into a dict and forwarded as the
tool's arguments; omit it for tools that take no input.

```python
%mcp_call fake_gmail list_messages {"max_results": 5}
%mcp_call fake_gmail get_message   {"id": "msg_001"}
%mcp_call fake_gmail list_labels                          # no args
```

The MCP result's content blocks are flattened to plain text in the cell
output, and the full `CallToolResult` object is returned for programmatic
use.

### `%mcp_servers`

Lists every MCP server currently connected in this kernel, with its command
and transport. No args.

```python
%mcp_servers
# - fake_gmail: ['python3', 'servers/gmail/server.py'] (stdio)
# - npx_gmail:  ['npx', '-y', '@gongrzhe/server-gmail-autoauth-mcp'] (stdio)
```

### `%mcp_close <name>` / `%mcp_close --all`

Disconnects and shuts down a server's subprocess so you can free resources
or replace a wedged connection.

```python
%mcp_close fake_gmail       # disconnect one
%mcp_close --all            # disconnect everything
```

**Idempotency note:** `%mcp_add <existing_name>` **refuses** to overwrite —
it prints a message naming the existing command and tells you to close first.
This avoids the silent-overwrite footgun where an in-flight tool call would
suddenly start failing because its client got replaced under it.

---

## Configuration

The magics build their components from the same env vars the CLI uses
(`rag.factory.build_*`). The full list:

| Var | Default | Meaning |
|---|---|---|
| `RAG_STORE` | `chroma` | `chroma` or `memory` |
| `RAG_DB` | `./.rag` | For `chroma`: directory holding `chroma.sqlite3`. For `memory`: path to the JSON file (or directory containing `memory.json`). |
| `RAG_COLLECTION` | `openmate` | Chroma collection name (3–512 chars, `[a-zA-Z0-9._-]`). |
| `RAG_EMBEDDER` | `hashing` | `hashing` or `openai` |
| `RAG_EMBED_DIM` | `256` | Hashing embedder dim. OpenAI defaults to 1536. |
| `RAG_EMBED_MODEL` | `text-embedding-3-small` | OpenAI model name. |
| `RAG_EMBED_API_KEY` | (falls back to `OPENAI_API_KEY`) | OpenAI key. |
| `RAG_EMBED_BASE_URL` | `https://api.openai.com/v1` | OpenAI-compatible base URL. |

Set them before launching Jupyter, or in a `%env` cell:

```python
%env RAG_STORE=chroma RAG_DB=/Users/me/.rag RAG_COLLECTION=openmate
%env RAG_EMBEDDER=openai RAG_EMBED_API_KEY=sk-... 
```

Reload the extension (`%reload_ext openmate.jupyter_magic`) after changing
env vars — the magics don't watch for changes.

---

## How it works (architecture)

The magics are **synchronous** functions — that's how IPython's `line_magic`
API works. But the underlying code (`rag.retrievers.DenseRetriever`,
`MCPClient`) is async. Bridging sync → async inside a Jupyter kernel is
tricky for two reasons:

1. **The kernel already has a running event loop.** Naive `asyncio.run(coro)`
   creates a *new* loop per call. Stateless coroutines (a one-shot query) are
   fine; stateful clients (MCPClient, whose stdio streams are bound to the
   loop they were created in) break on the second call.
2. **Some libraries rely on `sniffio`** (anyio uses it) to detect the active
   async backend. `nest_asyncio` makes the kernel loop nestable but doesn't
   correctly propagate `sniffio`'s `_running_loop` thread-local, so
   anyio-backed code throws `NoEventLoopError`.

The solution: a **dedicated background event loop in a worker thread**, owned
by the extension module. Every magic call schedules its coroutine on this
loop via `asyncio.run_coroutine_threadsafe` and blocks on the future.

```
kernel (main thread, kernel loop)        background worker thread
─────────────────────────────            ─────────────────────────
McpMagics.mcp_add (sync)         ─────►  loop.run_forever()
  asyncio.run_coroutine_threadsafe       await client.connect(spec)
  future.result()  ◄─────────────────    (streams bound to this loop)
                                        await client.list_tools()
McpMagics.mcp_list (sync)        ─────►  await client.list_tools()
  asyncio.run_coroutine_threadsafe       (same client, same loop ✓)
```

Properties of this design:

- **State persists across magic calls** — `MCPClient` connects once, stays
  connected, and every subsequent `%mcp_*` reuses the same streams.
- **No contact with the kernel loop** — we never `.result()` from the loop's
  own thread, so there's no deadlock.
- **Daemon thread, dies with the kernel** — no orphan subprocesses between
  sessions.
- **No `nest_asyncio` dependency** — the loop doesn't need to be nestable
  because we don't run on it.

The trade-off: if the kernel dies abruptly (Ctrl-C × 2, `kill -9`), the
background thread and its MCP subprocess may not get a chance to clean up.
You can verify and kill leftover processes with `pgrep -f mcp_server` or by
restarting the kernel.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'openmate.jupyter_magic'`**
You launched Jupyter outside the venv. `source .venv/bin/activate` first, or
use `.venv/bin/jupyter lab` directly.

**`ModuleNotFoundError: No module named 'mcp'` when calling `%mcp_add`**
The MCP server subprocess can't find `mcp`. Either activate the venv before
launching Jupyter, or use the venv's absolute path: `%mcp_add fake_gmail
/Users/you/projects/openmate/.venv/bin/python3 servers/gmail/server.py`.

**`FileNotFoundError: [Errno 2] No such file or directory: 'python'`**
macOS only — use `python3` or an absolute path. `python` isn't on PATH.

**`Object 'X' not found` / `NameError: name 'docs' is not defined`**
Your query ended in `?` and IPython ate it for introspection. Reword.

**`NoEventLoopError: Not currently running on any asynchronous event loop`**
You overrode `_run` or patched `nest_asyncio` in a conflicting way. The
default implementation uses the background-loop architecture — call
`%reload_ext openmate.jupyter_magic` to reset.

**MCP subprocess outlives the kernel**
On a forced kernel shutdown, the background thread may not get a chance to
close the connection. `pgrep -f servers/gmail/server.py | xargs kill` to
clean up, or just restart the kernel.

---

## Going further

The magics are pure wrappers — anything in `rag/` or
`openmate.adapters.tools.mcp_client` is reachable from a normal cell. Use the
magics for the 80% case, drop to Python for the rest:

```python
# Real LLM-synthesized answer (not just retrieved docs)
from rag import build_embedder, build_retriever, answer
from openmate.ports.model import default_model

emb, store = build_embedder(), build_store("memory")
# ... ingest ...
retriever = build_retriever(emb, store)
print(await answer("what is the agent loop?", retriever, default_model()))
```

```python
# Talk to MCP directly when the magic is too high-level
from openmate.adapters.tools.mcp_client import MCPClient, MCPServerSpec
import asyncio

async def main():
    c = MCPClient()
    await c.connect(MCPServerSpec(name="gmail", command=["npx", "-y", "@gongrzhe/server-gmail-autoauth-mcp"]))
    tools = await c.list_tools()
    print(tools)

# Run this with nest_asyncio.apply() first, or use the same background loop
# the magics already manage:
from openmate.jupyter_magic import _ensure_bg_loop, asyncio
loop = _ensure_bg_loop()
asyncio.run_coroutine_threadsafe(main(), loop).result()
```

See `examples/notebooks/openmate_demo.ipynb` for a worked example, and
`docs/04-tools-and-mcp.md` for why MCP is part of OpenMate in the first
place.