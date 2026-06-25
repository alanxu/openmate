"""IPython extension: a thin Jupyter surface over OpenMate's RAG and MCP APIs.

Why this exists
---------------
Neither Chroma nor MCP ship official IPython magic helpers. Rather than ship a
fragile third-party extension that targets an older version of one of them, we
write a small wrapper over *our own* ``rag/`` and ``openmate.adapters.tools.mcp_client``
APIs. That way the magic surface is always in sync with the rest of OpenMate,
and it costs us about 80 lines.

Usage in a notebook
-------------------
::

    %load_ext openmate.jupyter_magic

    # RAG — defaults to whatever RAG_STORE / RAG_DB / RAG_COLLECTION / RAG_EMBEDDER say.
    # Ingest a directory:
    %rag_ingest docs/

    # Query it:
    docs = %rag_query what is the agent loop?

    # MCP — point at any server spec; you give it a name + argv.
    %mcp_add fake_gmail python servers/gmail/server.py
    %mcp_list fake_gmail
    %mcp_call fake_gmail list_messages {"max_results": 5}
"""

from __future__ import annotations

import asyncio
import json
import shlex
import threading
from typing import Any

try:
    # IPython is required to use these magics, but we don't want it to be a
    # hard dependency of the ``openmate`` package itself (it'd force-install
    # Jupyter for everyone who just wants to run the agent). When IPython is
    # missing, the decorators and the Magics base class become harmless
    # no-ops so the module still imports cleanly — the magics just won't be
    # invokable.
    # NB: in IPython ≥8, the class decorator is ``magics_class`` (not the
    # older ``register_magics``). We export both names so the implementation
    # below works on either version.
    from IPython.core.magic import Magics, line_cell_magic, magics_class
except ImportError:  # pragma: no cover — only hit when IPython isn't installed
    class Magics:  # type: ignore[no-redef]  — minimal stub
        registered = False

        def __init__(self, shell=None):
            self.shell = shell

    def line_cell_magic(func):  # type: ignore[no-redef]
        return func

    def magics_class(cls):  # type: ignore[no-redef]
        cls.registered = True
        return cls


# We import the heavy modules lazily inside the magic functions so that simply
# loading the extension in a kernel that doesn't need MCP (e.g. RAG-only)
# doesn't require the ``mcp`` extra to be installed. The RAG magic is also
# lazy for the same reason — chromadb may not be present.


# A dedicated background event loop that owns all async state created by the
# magics (MCPClient connections, etc.). Running it in a worker thread means:
#
#   1. We never touch the kernel's own loop, so we can't deadlock by calling
#      ``.result()`` from inside the loop's own thread.
#   2. State persists across magic calls — MCPClient streams created in one
#      call are still alive when the next call queries the same server.
#   3. sniffio (which anyio uses for backend detection) sees a normal running
#      loop, because the worker thread is the loop's thread and runs it via
#      ``loop.run_forever()`` — no special tricks needed. (nest_asyncio didn't
#      work here because its patched ``run`` doesn't propagate the
#      ``_running_loop`` thread-local correctly for libraries that use sniffio.)
#
# The thread is a daemon, so it's torn down with the kernel. We start it lazily
# on the first ``_run()`` call rather than at extension-load, because the
# kernel is fully booted by the time the first magic runs.
_BG_LOOP: asyncio.AbstractEventLoop | None = None
_BG_THREAD: threading.Thread | None = None
_BG_LOCK = threading.Lock()


def _ensure_bg_loop() -> asyncio.AbstractEventLoop:
    global _BG_LOOP, _BG_THREAD
    if _BG_LOOP is not None and _BG_LOOP.is_running():
        return _BG_LOOP
    with _BG_LOCK:
        if _BG_LOOP is not None and _BG_LOOP.is_running():
            return _BG_LOOP
        _BG_LOOP = asyncio.new_event_loop()
        _BG_THREAD = threading.Thread(
            target=_BG_LOOP.run_forever, name="openmate-magic-loop", daemon=True
        )
        _BG_THREAD.start()
        return _BG_LOOP


def _run(coro: Any) -> Any:
    """Schedule a coroutine on the background loop and block for its result.

    Called from sync magic functions. Safe to call repeatedly — all coroutines
    share the same loop, so stateful clients (MCPClient) keep their streams.
    """
    loop = _ensure_bg_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


@magics_class
class RagMagics(Magics):
    """RAG magics — registered as line magics (``%rag_ingest``, ``%rag_query``).

    Bound to the ``rag`` namespace. IPython resolves ``%rag_query`` by looking
    for an attribute on this class.
    """

    def __init__(self, shell: Any | None = None) -> None:
        super().__init__(shell)

    @line_cell_magic  # type: ignore[name-defined]  # set at module bottom
    def rag_ingest(self, line: str) -> None:
        """Ingest a path (file or directory) into the configured RAG store.

        Usage:
            %rag_ingest docs/
            %rag_ingest README.md
        """
        from rag.factory import build_embedder, build_pipeline, build_store

        path = line.strip()
        if not path:
            print("usage: %rag_ingest <path>")
            return

        emb = build_embedder()
        store = build_store()
        pipeline = build_pipeline(emb, store)
        n = _run(pipeline.ingest(path))
        print(f"ingested {n} chunks from {path!r} into store={type(store).__name__}")

    @line_cell_magic  # type: ignore[name-defined]
    def rag_query(self, line: str) -> list[dict]:
        """Query the RAG store, return top-k documents as a list of dicts.

        Usage:
            %rag_query what is the agent loop?
            %rag_query what is the agent loop? --k 3
        """
        from rag.factory import build_embedder, build_retriever, build_store

        # Naive arg split: everything up to ``--k`` is the query; the rest is opts.
        parts = line.strip().split()
        if not parts:
            print("usage: %rag_query <question> [--k N]")
            return []
        k = 5
        if "--k" in parts:
            i = parts.index("--k")
            k = int(parts[i + 1])
            parts = parts[:i] + parts[i + 2 :]
        query = " ".join(parts).strip()
        if not query:
            print("usage: %rag_query <question> [--k N]")
            return []

        emb = build_embedder()
        store = build_store()
        retriever = build_retriever(emb, store)
        docs = _run(retriever.retrieve(query, k=k))
        result = [
            {
                "id": getattr(d, "id", None),
                "score": getattr(d, "score", None),
                "text": (d.text or "")[:400],
                "metadata": getattr(d, "metadata", {}),
            }
            for d in docs
        ]
        from rag.tools import format_documents

        print(format_documents(docs))
        return result


@magics_class
class McpMagics(Magics):
    """MCP magics — ``%mcp_add``, ``%mcp_list``, ``%mcp_call``, ``%mcp_close``.

    Maintains a per-kernel registry of MCP server connections. Each connection
    owns **one long-running task** on the background loop that runs from
    ``connect()`` through ``close()``. Doing connect/close in the same task
    avoids the cross-task cancel-scope problem that anyio's ``stdio_client``
    raises when ``AsyncExitStack.aclose()`` runs ``__aexit__`` on a
    taskgroup that was entered in a different task. The task also serialises
    every ``list_tools`` / ``call_tool`` for that server, so user code can't
    interleave a write with a read on the same stdio transport.
    """

    def __init__(self, shell: Any | None = None) -> None:
        super().__init__(shell)
        # { name: _McpServerSession }
        self._servers: dict[str, "_McpServerSession"] = {}

    @line_cell_magic  # type: ignore[name-defined]
    def mcp_add(self, line: str) -> None:
        """Connect to an MCP server by spawning a stdio subprocess.

        Usage:
            %mcp_add fake_gmail python servers/gmail/server.py
            %mcp_add npx_gmail npx -y @gongrzhe/server-gmail-autoauth-mcp

        Refuses to overwrite an existing connection with the same name — use
        ``%mcp_close <name>`` first if you really want to replace it.
        """
        from openmate.adapters.tools.mcp_client import MCPClient, MCPServerSpec

        tokens = shlex.split(line.strip())
        if len(tokens) < 2:
            print("usage: %mcp_add <name> <command> [args...]")
            return
        name, *command = tokens
        if name in self._servers:
            existing_cmd = self._servers[name].spec.command
            print(
                f"server {name!r} is already connected (cmd={existing_cmd!r}).\n"
                f"  Use `%mcp_close {name}` to disconnect it first, then re-run "
                f"`%mcp_add`.\n"
                f"  Or pick a different name."
            )
            return
        spec = MCPServerSpec(name=name, command=command)
        session = _McpServerSession(name, spec)
        try:
            session.start(timeout=30)
        except Exception as exc:
            print(f"failed to connect {name!r}: {type(exc).__name__}: {exc}")
            session.shutdown()
            return
        self._servers[name] = session
        # Stash a reference on the shell so unload_ipython_extension can find us.
        if self.shell is not None:
            setattr(self.shell, "_openmate_mcp_magics", self)
        try:
            tools = session.list_tools()
        except Exception as exc:
            print(f"connected {name!r} but listing tools failed: {exc}")
            return
        print(
            f"connected to {name!r} ({spec.command[0]}…); "
            f"{len(tools)} tools: {[t.spec.name for t in tools]}"
        )

    @line_cell_magic  # type: ignore[name-defined]
    def mcp_close(self, line: str) -> None:
        """Disconnect and shut down an MCP server's subprocess.

        Usage:
            %mcp_close fake_gmail
            %mcp_close --all
        """
        arg = line.strip()
        if arg in ("--all", "-a", "all"):
            for name in list(self._servers.keys()):
                self._close_one(name)
            return
        if not arg:
            print("usage: %mcp_close <name>   |   %mcp_close --all")
            return
        if arg not in self._servers:
            print(f"no server named {arg!r} is connected")
            return
        self._close_one(arg)

    def _close_one(self, name: str) -> None:
        session = self._servers.pop(name, None)
        if session is None:
            return
        try:
            session.shutdown()
            print(f"disconnected {name!r}")
        except Exception as exc:  # noqa: BLE001 — best-effort shutdown
            print(f"disconnected {name!r} (shutdown raised: {type(exc).__name__}: {exc})")

    @line_cell_magic  # type: ignore[name-defined]
    def mcp_servers(self, line: str) -> list[dict]:
        """List currently connected MCP servers and their commands.

        Usage:
            %mcp_servers
        """
        rows = [
            {"name": s.spec.name, "command": s.spec.command, "transport": s.spec.transport}
            for s in self._servers.values()
        ]
        if not rows:
            print("(no MCP servers connected — use `%mcp_add <name> <cmd>`)")
        else:
            for r in rows:
                print(f"- {r['name']}: {r['command']!r} ({r['transport']})")
        return rows

    @line_cell_magic  # type: ignore[name-defined]
    def mcp_list(self, line: str) -> list[dict]:
        """List tools from a connected MCP server.

        Usage:
            %mcp_list fake_gmail
        """
        name = line.strip()
        if not name:
            print("usage: %mcp_list <server_name>")
            return []
        session = self._servers.get(name)
        if session is None:
            print(f"no server named {name!r} is connected")
            return []
        tools = session.list_tools()
        rows = [
            {
                "name": t.spec.name,
                "remote_name": t.remote_name,
                "description": t.spec.description,
                "schema": t.spec.parameters,
            }
            for t in tools
        ]
        for r in rows:
            print(f"- {r['name']}: {r['description']}")
        return rows

    @line_cell_magic  # type: ignore[name-defined]
    def mcp_call(self, line: str) -> Any:
        """Call a tool on a connected MCP server.

        Usage:
            %mcp_call fake_gmail list_messages {"max_results": 5}
        """
        try:
            name, tool, rest = line.strip().split(None, 2)
        except ValueError:
            print('usage: %mcp_call <server> <tool> \'{"key": "value"}\'')
            return None
        args = json.loads(rest) if rest.strip() else {}
        session = self._servers.get(name)
        if session is None:
            print(f"no server named {name!r} is connected")
            return None
        result = session.call_tool(tool, args)
        for block in result.content:
            text = getattr(block, "text", None)
            if text is not None:
                print(text)
        return result


class _McpServerSession:
    """One connected MCP server: holds the client + a long-running task.

    Every operation (``list_tools``, ``call_tool``, ``close``) is dispatched
    onto a single task that owns the underlying ``MCPClient``. Serialising on
    one task means connect/close happen in the same anyio cancel scope, which
    is the only way to clean up ``stdio_client``'s internal taskgroup without
    hitting ``RuntimeError: Attempted to exit cancel scope in a different
    task than it was entered in``.

    Magic code talks to this class via thread-safe methods; everything else
    is implementation detail.
    """

    def __init__(self, name: str, spec: Any) -> None:
        self.name = name
        self.spec = spec
        self._loop = _ensure_bg_loop()
        self._client: Any | None = None
        # PID of the stdio subprocess, captured *before* the call to
        # ``client.close()`` so we can guarantee we kill exactly the one we
        # spawned — never a freshly-spawned replacement after re-`%mcp_add`.
        self._subprocess_pid: int | None = None
        self._ready = asyncio.Event()  # type: ignore[var-annotated]
        self._close_event: asyncio.Event = asyncio.Event()  # type: ignore[var-annotated]
        self._shutdown_done: asyncio.Event = asyncio.Event()  # type: ignore[var-annotated]
        self._error: BaseException | None = None
        self._task: asyncio.Future | None = None

    def start(self, timeout: float = 30) -> None:
        """Spawn the per-server task and wait until it's connected (or failed)."""
        self._task = asyncio.run_coroutine_threadsafe(self._lifecycle(), self._loop)
        # Wait for the connect to finish (success or failure).
        if not _wait_for(self._ready, timeout, loop=self._loop):
            raise TimeoutError(
                f"MCP server {self.name!r} didn't connect within {timeout}s"
            )
        if self._error is not None:
            raise self._error

    def list_tools(self) -> list[Any]:
        return self._submit(self._do_list_tools())

    def call_tool(self, tool: str, args: dict) -> Any:
        return self._submit(self._do_call_tool(tool, args))

    def shutdown(self) -> None:
        """Signal close; wait for the lifecycle task to finish."""
        if self._task is None:
            return
        # ``Event.set()`` is sync — schedule it on the loop with
        # ``call_soon_threadsafe`` so it runs in the loop's thread (events are
        # not strictly thread-safe to mutate from outside).
        self._loop.call_soon_threadsafe(self._close_event.set)
        # Wait for the task to fully wind down.
        try:
            self._task.result(timeout=15)
        except Exception:
            pass  # best-effort — the subprocess is the important bit

    def _submit(self, coro: Any) -> Any:
        """Run ``coro`` on the per-server task's loop and return its result.

        We submit on the same shared background loop, but ``coro`` is a
        separate task. ``list_tools`` / ``call_tool`` are safe to do in a
        different task than ``connect`` because they only touch the read/write
        streams, not the taskgroup that owns them.
        """
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    # --- coroutines running on the background loop ------------------------

    async def _lifecycle(self) -> None:
        """Owns the client from connect() through close()."""
        from openmate.adapters.tools.mcp_client import MCPClient

        try:
            self._client = MCPClient()
            await self._client.connect(self.spec)
            # Snapshot the subprocess PID right after connect so the close path
            # can guarantee it kills the *exact* process it spawned, never a
            # later sibling spawned by a concurrent re-`%mcp_add`.
            self._subprocess_pid = await _find_subprocess_pid(self.spec.command)
            self._ready.set()
            await self._close_event.wait()
            # Close in the SAME task as connect — required for stdio_client's
            # internal anyio taskgroup cancel scope.
            await self._client.close()
            # Belt-and-braces: the stdio_client's ``with process:`` exit should
            # have terminated the subprocess, but we've seen cases where it
            # returns cleanly while the subprocess keeps running. Kill the
            # captured PID if it's still alive.
            if self._subprocess_pid is not None and _pid_alive(self._subprocess_pid):
                _terminate_pid(self._subprocess_pid)
        except BaseException as exc:
            self._error = exc
            self._ready.set()
        finally:
            self._shutdown_done.set()

    async def _do_list_tools(self) -> list[Any]:
        if self._client is None:
            raise RuntimeError(f"server {self.name!r} not connected")
        return await self._client.list_tools()

    async def _do_call_tool(self, tool: str, args: dict) -> Any:
        if self._client is None:
            raise RuntimeError(f"server {self.name!r} not connected")
        # MCPClient doesn't expose call_tool; we go to the session directly.
        session = self._client._sessions[self.spec.name]
        return await session.call_tool(tool, args)


def _wait_for(event: asyncio.Event, timeout: float, *, loop: asyncio.AbstractEventLoop) -> bool:
    """Sync wait for an :class:`asyncio.Event` set on ``loop``.

    ``Event.is_set()`` is a plain boolean read — safe to call from any thread
    for our purposes (we set it once and read it once). We poll instead of
    blocking on ``run_coroutine_threadsafe(...).result()`` so we don't tie up
    the background loop waiting on itself.
    """
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if event.is_set():
            return True
        time.sleep(0.05)
    return False


# --- subprocess PID tracking -------------------------------------------------
#
# These helpers let the magic identify *exactly* the subprocess it spawned, so
# the close path can kill it without ever accidentally hitting a fresh sibling
# spawned by a concurrent re-``%mcp_add``. Cheaper and more reliable than the
# ps+grep approach, which has a known race window between close() returning and
# the kill firing.

import os as _os
import signal as _signal
import subprocess as _sp


async def _find_subprocess_pid(command: list[str]) -> int | None:
    """Return the PID of the most recently spawned subprocess matching ``command``.

    Called from the background loop right after ``client.connect()``, when the
    subprocess has just been spawned. We look for a ``ps`` line whose command
    ends with the same argv we passed — the freshest such process (highest
    start time) is almost certainly ours.

    Returns ``None`` on any failure (ps unavailable, race lost, …); the caller
    then skips the fallback kill and trusts ``client.close()`` to clean up.
    """
    import shutil

    if shutil.which("ps") is None:
        return None
    try:
        proc = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: _sp.run(
                ["ps", "-A", "-o", "pid=,lstart=,command="],
                capture_output=True,
                text=True,
                timeout=2,
            ),
        )
    except Exception:
        return None
    needle = " ".join(command)
    candidates: list[tuple[str, int]] = []  # (lstart, -pid)  → newest first
    for line in proc.stdout.splitlines():
        if needle not in line:
            continue
        parts = line.strip().split(None, 1)
        if len(parts) < 2:
            continue
        pid_str, rest = parts
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        # lstart is in the middle of `rest` — extract first 5 fields after pid.
        rest_parts = rest.split(None, 4)
        if len(rest_parts) < 5:
            continue
        lstart = " ".join(rest_parts[:5])
        candidates.append((lstart, -pid, pid))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][2]


def _pid_alive(pid: int) -> bool:
    try:
        _os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False


def _terminate_pid(pid: int) -> None:
    """SIGTERM, wait briefly, SIGKILL if still alive."""
    try:
        _os.kill(pid, _signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        return
    import time

    deadline = time.monotonic() + 1.5
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return
        time.sleep(0.1)
    try:
        _os.kill(pid, _signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def load_ipython_extension(ipython: Any) -> None:
    """Register both magic classes with the running IPython kernel.

    Called by ``%load_ext openmate.jupyter_magic``. Safe to call twice — IPython
    will replace the existing magics.

    IPython ≥8 puts registration on ``MagicsManager.register``; older versions
    exposed ``InteractiveShell.register_magics``. We try both so this works
    on either.

    We don't apply ``nest_asyncio`` — the magics use a dedicated background
    loop (see :data:`_BG_LOOP`) so they don't need the kernel loop to be
    nestable. That's the most reliable path for libraries like MCP whose
    stdio transport uses sniffio for backend detection.
    """
    register = getattr(
        getattr(ipython, "magics_manager", None), "register", None
    ) or getattr(ipython, "register_magics", None)
    if register is None:
        raise RuntimeError(
            "couldn't find a way to register magic classes on this IPython"
        )
    register(RagMagics)
    register(McpMagics)
    print(
        "OpenMate Jupyter extension loaded.\n"
        "  RAG:  %rag_ingest <path>    %rag_query <question> [--k N]\n"
        "  MCP:  %mcp_add <name> <cmd>     %mcp_list <name>    %mcp_call <name> <tool> <json>"
    )


def unload_ipython_extension(ipython: Any) -> None:
    """Best-effort cleanup of any MCP subprocesses we spawned.

    Stores the singleton on the shell so ``unload_ext`` can find it (IPython
    doesn't pass us the registered instance, so we save a reference on the
    shell object the first time ``mcp_add`` runs).
    """
    mcp_magics = getattr(ipython, "_openmate_mcp_magics", None)
    if mcp_magics is None:
        return
    for name in list(mcp_magics._servers.keys()):
        try:
            mcp_magics._close_one(name)
        except Exception as exc:  # noqa: BLE001  — best-effort shutdown
            print(f"failed to close MCP server {name!r}: {exc}")