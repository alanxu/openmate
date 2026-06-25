"""Tool providers — the assembly seam (docs/04 §Phase 5).

A ``ToolProvider`` is a build-time *factory* that contributes tools (and maybe a
system-prompt fragment). It's how shell, MCP, and skills are sourced uniformly:
:func:`openmate.agent.assemble.assemble` resolves a list of providers into the
agent's ``Harness.tools`` and owns their lifecycle (e.g. closing MCP connections).

Roles, no overlap: an ``Agent`` is the facade you ``run()``; its ``Harness`` is the
environment; ``Services`` is shared infra. A provider is none of these — once it
has produced its tools, the running loop never touches it again.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from ..kernel.types import TextPart
from ..ports.tool import ToolResult, ToolSpec

if TYPE_CHECKING:
    from ..adapters.tools.mcp_client import MCPServerSpec
    from ..kernel.types import RunContext
    from ..ports.tool import Tool


@runtime_checkable
class ToolProvider(Protocol):
    name: str

    async def setup(self) -> None:
        """Connect / discover. Idempotent — ``assemble`` calls it once."""
        ...

    async def tools(self) -> list["Tool"]:
        """The tools this provider contributes."""
        ...

    def system_fragment(self) -> str | None:
        """Optional prompt text (e.g. skill cards) merged into instructions."""
        ...

    async def teardown(self) -> None:
        """Release anything ``setup`` acquired."""
        ...


class NativeProvider:
    """Contributes a fixed list of in-process ``Tool``s (e.g. the built-ins).

    The simplest provider — it just hands back tools it was given, so native
    Python tools sit in ``assemble()`` alongside MCP and skills uniformly.
    """

    def __init__(
        self,
        tools: list["Tool"],
        *,
        name: str = "native",
        system_fragment: str | None = None,
    ) -> None:
        self.name = name
        self._tools = list(tools)
        self._fragment = system_fragment

    async def setup(self) -> None:
        return None

    async def tools(self) -> list["Tool"]:
        return list(self._tools)

    def system_fragment(self) -> str | None:
        return self._fragment

    async def teardown(self) -> None:
        return None


class LocalSandbox:
    """A minimal PoC sandbox: run a command in a subprocess scoped to ``cwd``.

    NOT isolated — it has no container, no network egress controls, no dropped
    credentials. Those are docs/04 §Phase 3 (``SubprocessSandbox`` rlimits,
    ``ContainerSandbox`` gVisor/Firecracker). This exists so ``ShellProvider`` is
    real for local automation; treat the shell tool as side-effecting and gate it
    behind approval in any non-local deployment.
    """

    def __init__(self, cwd: str | None = None) -> None:
        self.cwd = cwd

    async def run(self, command: str, timeout: float) -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=self.cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return 124, "", f"timed out after {timeout}s"
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated, {len(text)} chars total]"


class ShellTool:
    """Run a shell command in a sandbox — file ops, scripts, computation."""

    spec = ToolSpec(
        name="shell",
        description=(
            "Run a shell command in an isolated working directory and return its "
            "stdout, stderr, and exit code. Use for file operations, running "
            "scripts, and quick computation."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "the command line to run"},
                "timeout_s": {"type": "number", "description": "max seconds to wait"},
            },
            "required": ["command"],
        },
        side_effecting=True,  # → policy / approval (10)
        idempotent=False,
    )

    def __init__(self, sandbox: LocalSandbox) -> None:
        self.sandbox = sandbox

    async def invoke(self, args: dict, ctx: "RunContext") -> ToolResult:
        command = (args or {}).get("command")
        if not command:
            return ToolResult([TextPart("missing required argument: command")], is_error=True)
        timeout = float((args or {}).get("timeout_s") or 30.0)
        code, out, err = await self.sandbox.run(command, timeout)
        body = f"$ {command}\n{_truncate(out)}"
        if err:
            body += f"\n[stderr]\n{_truncate(err)}"
        body += f"\n[exit {code}]"
        return ToolResult([TextPart(body)], is_error=code != 0)


class ShellProvider:
    """Contributes a single sandboxed :class:`ShellTool`."""

    name = "shell"

    def __init__(self, sandbox: LocalSandbox | None = None) -> None:
        self.sandbox = sandbox or LocalSandbox()

    async def setup(self) -> None:
        return None

    async def tools(self) -> list["Tool"]:
        return [ShellTool(self.sandbox)]

    def system_fragment(self) -> str | None:
        return None

    async def teardown(self) -> None:
        return None


class MCPProvider:
    """Contributes the tools of one or more MCP servers (Gmail, Calendar, a DB).

    ``scope_allowlist`` is the capability boundary (least privilege): only the
    named tools are exposed to the agent, so a triage profile need not even *see*
    a send tool. Connections are opened on ``setup`` and closed on ``teardown``.
    """

    name = "mcp"

    def __init__(
        self,
        servers: list["MCPServerSpec"],
        *,
        scope_allowlist: list[str] | None = None,
    ) -> None:
        from ..adapters.tools.mcp_client import MCPClient

        self.servers = list(servers)
        self.client = MCPClient()
        self._allow = set(scope_allowlist) if scope_allowlist is not None else None

    async def setup(self) -> None:
        for s in self.servers:
            await self.client.connect(s)

    async def tools(self) -> list["Tool"]:
        tools = await self.client.list_tools()
        if self._allow is not None:
            tools = [
                t
                for t in tools
                if t.spec.name in self._allow or t.remote_name in self._allow
            ]
        return tools

    def system_fragment(self) -> str | None:
        return None

    async def teardown(self) -> None:
        await self.client.close()
