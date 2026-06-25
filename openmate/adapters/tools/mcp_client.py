"""MCP client — acquire third-party tools over the Model Context Protocol.

The primary way OpenMate reaches external systems (Gmail, Calendar, a DB): an MCP
server's ``tools`` become OpenMate ``Tool``s through :class:`MCPToolAdapter`, so
they flow through the *same* registry, scoping, approval, and tracing as native
tools — an MCP tool is not privileged for being external (docs/04 §Phase 4).

The ``mcp`` SDK is an *optional* dependency, imported lazily inside
:meth:`MCPClient.connect`. Everything that doesn't open a real transport (spec
translation, result shaping, the adapter's ``invoke`` against any session object)
works without it and is unit-testable with a fake session.
"""

from __future__ import annotations

import json
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ...kernel.errors import ConfigError
from ...kernel.types import TextPart
from ...ports.tool import ToolResult, ToolSpec

if TYPE_CHECKING:
    from ...kernel.types import RunContext


@dataclass
class MCPServerSpec:
    """How to reach one MCP server.

    PoC supports the ``stdio`` transport (a local subprocess). ``command`` is the
    argv to launch it; ``namespace_prefix`` guards against tool-name collisions
    across sources (e.g. ``gmail_`` so ``search`` can't clash with a native one).
    """

    name: str
    command: list[str]
    transport: str = "stdio"
    env: dict[str, str] | None = None
    cwd: str | None = None
    namespace_prefix: str = ""
    ttl_s: float = 300.0  # how long a tools/list result stays cached


def _namespaced(remote_name: str, prefix: str) -> str:
    if not prefix or remote_name.startswith(prefix):
        return remote_name
    return f"{prefix}{remote_name}"


def spec_from_mcp_tool(mcp_tool: Any, *, namespace_prefix: str = "") -> ToolSpec:
    """Translate an MCP tool definition into a :class:`ToolSpec`.

    The MCP ``annotations`` (``readOnlyHint``/``destructiveHint``/``idempotentHint``)
    are mapped *once* here into the flags every downstream concern reuses:
    ``readOnlyHint → side_effecting=False`` (read-only tools skip approval),
    ``idempotentHint → idempotent`` (safe to retry/replay).
    """
    ann = getattr(mcp_tool, "annotations", None)
    read_only = bool(getattr(ann, "readOnlyHint", False)) if ann else False
    idempotent = bool(getattr(ann, "idempotentHint", False)) if ann else False
    schema = getattr(mcp_tool, "inputSchema", None) or {
        "type": "object",
        "properties": {},
    }
    return ToolSpec(
        name=_namespaced(mcp_tool.name, namespace_prefix),
        description=getattr(mcp_tool, "description", None) or mcp_tool.name,
        parameters=schema,
        side_effecting=not read_only,
        idempotent=idempotent,
    )


def result_from_mcp(call_result: Any) -> ToolResult:
    """Shape an MCP ``CallToolResult`` into a model-legible :class:`ToolResult`."""
    parts: list[TextPart] = []
    for c in getattr(call_result, "content", None) or []:
        text = getattr(c, "text", None)
        if text is not None:
            parts.append(TextPart(text))
        else:  # non-text block (image/resource ref) — represent compactly
            parts.append(TextPart(str(getattr(c, "data", c))))
    structured = getattr(call_result, "structuredContent", None)
    if not parts and structured is not None:
        parts.append(TextPart(json.dumps(structured, indent=2, default=str)))
    if not parts:
        parts.append(TextPart(""))
    return ToolResult(parts, is_error=bool(getattr(call_result, "isError", False)))


class MCPToolAdapter:
    """Wraps one remote MCP tool as an OpenMate :class:`Tool`.

    The model sees ``spec.name`` (possibly namespaced); calls are dispatched to the
    server under the original ``remote_name``, so namespacing is transparent.
    """

    def __init__(self, session: Any, mcp_tool: Any, *, namespace_prefix: str = "") -> None:
        self._session = session
        self.remote_name: str = mcp_tool.name
        self.spec = spec_from_mcp_tool(mcp_tool, namespace_prefix=namespace_prefix)

    async def invoke(self, args: dict, ctx: "RunContext") -> ToolResult:
        try:
            res = await self._session.call_tool(self.remote_name, args or {})
        except Exception as e:  # noqa: BLE001 — surface as a recoverable result
            return ToolResult(
                [TextPart(f"MCP tool '{self.remote_name}' failed: {type(e).__name__}: {e}")],
                is_error=True,
            )
        return result_from_mcp(res)


class MCPClient:
    """Connects to one or more MCP servers and exposes their tools as ``Tool``s.

    Holds the transport/session lifecycle on an ``AsyncExitStack`` so a single
    :meth:`close` tears every connection down cleanly (the ``MCPProvider`` calls
    it on ``teardown``).
    """

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._sessions: dict[str, Any] = {}
        self._specs: dict[str, MCPServerSpec] = {}

    async def connect(self, spec: MCPServerSpec) -> None:
        if spec.transport != "stdio":
            raise ConfigError(
                f"MCP transport '{spec.transport}' not supported yet (PoC: stdio only)"
            )
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:  # optional dependency
            raise ConfigError(
                "the MCP SDK is required for MCP servers — install with "
                "`pip install \"openmate[mcp]\"`"
            ) from e

        if not spec.command:
            raise ConfigError(f"MCP server '{spec.name}' has an empty command")

        if self._stack is None:
            self._stack = AsyncExitStack()
            await self._stack.__aenter__()

        params = StdioServerParameters(
            command=spec.command[0],
            args=list(spec.command[1:]),
            env=spec.env,
            cwd=spec.cwd,
        )
        read, write = await self._stack.enter_async_context(stdio_client(params))
        session = await self._stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        self._sessions[spec.name] = session
        self._specs[spec.name] = spec

    async def list_tools(self) -> list[MCPToolAdapter]:
        """Discover and adapt every connected server's tools (namespaced)."""
        tools: list[MCPToolAdapter] = []
        for name, session in self._sessions.items():
            spec = self._specs[name]
            resp = await session.list_tools()
            for t in resp.tools:
                tools.append(
                    MCPToolAdapter(session, t, namespace_prefix=spec.namespace_prefix)
                )
        return tools

    async def read_resource(self, uri: str) -> str:
        """Read an MCP resource (e.g. ``gmail://thread/{id}``) as text.

        Tries each connected server and returns the first that resolves it.
        """
        last_err: Exception | None = None
        for session in self._sessions.values():
            try:
                res = await session.read_resource(uri)
            except Exception as e:  # noqa: BLE001 — try the next server
                last_err = e
                continue
            return "\n".join(
                getattr(c, "text", "") for c in getattr(res, "contents", []) or []
            )
        raise ConfigError(f"no connected MCP server could read resource '{uri}'") from last_err

    async def close(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
            self._stack = None
        self._sessions.clear()
        self._specs.clear()
