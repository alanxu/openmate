# 04 — Tools & Capabilities (incl. MCP)

> How an agent senses and affects the world. Part of OpenMate; see [architecture.md §8](architecture.md#8-tools--capabilities-incl-mcp). The `Tool` port is tiny on purpose — MCP servers, other frameworks' tools, and sub-agents all become `Tool`s via adapters.

## Scope & responsibilities

This module owns the tool contract, the **executor** (dispatch, concurrency, timeouts, retries, sandboxing), the **registry** (scoping, namespacing, discovery), and the **MCP** client/server integration. Tool *authorization* (allowlists, approval) is defined here as a hook but enforced by safety ([10](10-safety-and-guardrails.md)); tool *results* flow back as `ToolResultPart`s ([01](01-domain-model-and-kernel.md)).

---

## Core abstractions (class level)

```python
# openmate/ports/tool.py
@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str                 # model-facing prompt surface — designed, not dashed off
    parameters: dict                 # JSON Schema for args
    side_effecting: bool = True      # read-only tools may skip approval (10)
    timeout_s: float = 30.0
    idempotent: bool = False         # safe to retry/replay (12)
    cost_hint: float | None = None   # for budgeting/routing

@dataclass
class ToolResult:
    content: list[Part]
    is_error: bool = False
    retriable: bool = False
    artifacts: list["Artifact"] = field(default_factory=list)  # files/blobs offloaded (01)

class Tool(Protocol):
    spec: ToolSpec
    async def invoke(self, args: dict, ctx: RunContext) -> ToolResult: ...

# decorator sugar for native python tools
def tool(*, name=None, description=None, **meta) -> Callable[[Callable], Tool]: ...
```

---

## Phase 0 — PoC (foundational)

**Goal:** define a tool from a Python function, validate args, dispatch sequentially, return model-legible results/errors.

```python
# openmate/adapters/tools/native.py
class FunctionTool(Tool):
    def __init__(self, fn, spec): self.fn, self.spec = fn, spec
    async def invoke(self, args, ctx):
        try:
            validate(args, self.spec.parameters)            # JSON-schema validate
            out = await maybe_await(self.fn(**args))
            return ToolResult([TextPart(str(out))])
        except ValidationError as e:
            return ToolResult([TextPart(f"invalid args: {e}")], is_error=True)  # recoverable (P6)

# openmate/kernel/executor.py  (PoC: sequential)
class ToolExecutor:
    async def dispatch(self, calls, agent, svc) -> list[ToolResultPart]:
        out = []
        for c in calls:
            tool = agent.registry.resolve(c.name)
            svc.bus.emit(ToolCallRequested(call=c))
            t0 = svc.clock()
            res = await tool.invoke(c.args, RunContext(...))
            svc.bus.emit(ToolReturned(result=..., ms=(svc.clock()-t0)*1000))
            out.append(ToolResultPart(c.id, res.content, res.is_error))
        return out
```

Generate `ToolSpec.parameters` automatically from type hints/docstring (à la function-calling helpers). Ship two or three real tools to exercise it: `web_search`, `python_eval` (sandbox stub), `read_file`.

**PoC acceptance:** the model calls a tool, gets a result, and continues; bad args produce a corrective error the model recovers from, not a crash.

---

## Phase 1 — Robust dispatch

- **Parallel execution:** independent calls run via `asyncio.gather` with a semaphore (max concurrency); results re-ordered by `call.id` for determinism.
- **Timeouts & cancellation:** per-tool `timeout_s` enforced with `asyncio.wait_for`; `RunContext.deadline` propagated so long tools self-abort.
- **Retries:** transient failures (`retriable=True`) retried with backoff; non-idempotent tools never auto-retried without an idempotency key ([12](12-production-and-reliability.md)).
- **Result shaping:** large outputs truncated with a "N more bytes — use `fetch` to page" hint; big blobs become `Artifact`s referenced by id (keeps the window small, [09](09-context-engineering.md)).
- **Error taxonomy:** distinguish *user-fixable* (bad args), *transient* (network), and *fatal* (auth) so the model and runtime react appropriately.

---

## Phase 2 — Registry, scoping & discovery

Agents receive a **scoped registry**, not the global tool list — capability *and* safety boundary (least privilege).

```python
class ToolRegistry:
    def register(self, tool: Tool, *, namespace: str = "") -> None: ...
    def scope(self, *, allow: list[str] | None = None, tags: list[str] | None = None) -> "ToolRegistry": ...
    def resolve(self, name: str) -> Tool: ...      # namespaced, collision-safe
    def specs(self) -> list[ToolSpec]: ...         # what the model sees this turn
```

Techniques: **namespacing** to avoid collisions across sources (`mcp:github/create_issue` vs `native/create_issue`); **dynamic tool selection** — when the toolset is large, retrieve the top-k relevant `ToolSpec`s per turn (RAG over tools) instead of dumping all of them into context; **per-run / per-user scopes**; **tool versioning** so prompts pin a spec version ([12](12-production-and-reliability.md)).

---

## Phase 3 — Sandboxing & untrusted execution

Side-effecting and code-exec tools run in isolation so a buggy or hijacked tool can't reach the host (defense-in-depth with [10](10-safety-and-guardrails.md)).

```python
class Sandbox(Protocol):
    async def run(self, spec: "ExecSpec", ctx: RunContext) -> ExecResult: ...
class SubprocessSandbox(Sandbox): ...     # rlimits, temp cwd, dropped env
class ContainerSandbox(Sandbox): ...      # gVisor/Firecracker, no-net or egress allowlist
class RemoteSandbox(Sandbox): ...         # ephemeral worker / microVM
```

Controls: filesystem jail (scoped temp dir), **network egress allowlist** (default deny), CPU/mem/time rlimits, no ambient credentials (secrets injected per-call, scoped). The Python/code tool, the shell tool, and any "computer use" tool route through a `Sandbox`.

---

## Phase 4 — MCP & interoperable tool sources

MCP is the primary way to acquire third-party tools/data (architecture §8.4). Map its primitives onto OpenMate ports:

| MCP primitive | OpenMate mapping |
|---|---|
| Tools (side effects) | `Tool` via `MCPToolAdapter` |
| Resources (read-only) | `Retriever`/resource source ([07](07-retrieval-rag.md)) |
| Prompts (templates) | context fragments / canned instructions ([09](09-context-engineering.md)) |

```python
# openmate/adapters/tools/mcp_client.py
class MCPClient:
    async def connect(self, server: "MCPServerSpec") -> None: ...   # stdio or Streamable HTTP
    async def list_tools(self) -> list[Tool]: ...                   # cached per ttlMs/cacheScope
    async def list_resources(self) -> list["Resource"]: ...
    # negotiates capabilities on connect; refreshes tools/list on TTL expiry
```

Techniques: **capability negotiation** handshake on connect; **transport choice** (stdio for local subprocess servers, Streamable HTTP for remote — the 2026 spec trends stateless with routing headers + cache hints); **cache `tools/list`** per server hints; apply the **same scoping/approval/tracing** as native tools (an MCP tool is not privileged for being external). Other interop sources via the same adapter pattern: a LangChain/LlamaIndex tool, an OpenAI function tool, or an entire **agent-as-tool** ([08](08-multi-agent-orchestration.md)).

**Expose-as-server:** `MCPServer` publishes selected OpenMate tools/agents so other systems (or agents over A2A) can consume them — the symmetric half of interoperability ([13](13-framework-interoperability.md)).

---

## Testing & verification

- **Schema fidelity:** generated `parameters` validate the function's real signature; bad args rejected.
- **Concurrency:** parallel dispatch returns deterministic order; a slow tool hits its timeout without hanging the run.
- **Sandbox escape tests:** code tool cannot read outside its jail, cannot reach disallowed hosts, cannot see secrets.
- **MCP contract:** a mock MCP server exercises connect/list/call, TTL cache expiry, and reconnection.

## Trade-offs & open questions

Sandbox tier vs. latency (subprocess for PoC, container/remote for prod). How aggressively to truncate tool output vs. paging (start conservative). Static tool lists vs. retrieved tool selection threshold (switch to retrieval past ~30–50 tools). Whether resources should be modeled as retrievers or first-class read-only tools (lean: retrievers).
