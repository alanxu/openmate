# 15 — Tool Architecture: Aligning OpenMate with Claude Code

> Owns: how OpenMate's tool/skill layer should evolve to behave like a mature
> general-purpose agent. This is a gap-analysis-and-recommendation doc, not a
> spec — it compares Claude Code's shipped tool architecture against
> OpenMate's actual code (read directly, not the aspirational design docs) and
> proposes concrete, scoped changes. Related: [04](04-tools-and-mcp.md),
> [10](10-safety-and-guardrails.md), [14](14-skills.md).

## 1. Why Claude Code, why now

Claude Code is a production agent that has converged on a tool architecture
through real usage at scale. OpenMate is a PoC with the right shape (ports,
providers, a `Tool` protocol) but several of its safety and consolidation
decisions are still open. Rather than re-derive these from first principles,
this doc treats Claude Code's tool design as a reference implementation and
asks: where does OpenMate already match it, where does it diverge, and which
divergences are bugs versus legitimate PoC scope-cuts.

## 2. Claude Code's tool architecture, summarized

Source: Claude Code's tools reference (`code.claude.com/docs/en/tools-reference`,
checked 2026-06-26).

**A small set of well-scoped primitives, not one per use case.** Claude Code
does not have a dedicated "list directory" tool — directory listing happens
via `ls` through `Bash`. It doesn't have a dedicated "search file contents"
tool that duplicates `Bash grep` either — instead `Grep` exists as its own
tool because it wraps ripgrep with real value-add (regex modes, `glob`/`type`
scoping, gitignore-awareness) that a raw shell call wouldn't give you for
free. The rule implied by the actual tool list: a capability gets its own
tool only when it adds real value over composing existing primitives (typed
params, caching, scoping, safety), not merely because the capability is
common.

**Permission rules are a first-class, uniform addressing scheme.** Every tool
name is a string usable in `permissions.allow`/`deny`, in `--allowedTools`,
in subagent `tools`/`disallowedTools` frontmatter, and in hook matchers. Several
tools share a richer specifier format on top of the bare name —
`Bash(npm run *)` (command pattern), `Read(~/secrets/**)` (path pattern,
shared across `Read`/`Grep`/`Glob`/`LSP`), `WebFetch(domain:example.com)`
(domain match), `Skill(deploy *)` (skill-name match). This is what lets one
tool (`Bash`) stay generically powerful while still being governable at a
fine grain — you don't need fifteen narrow tools to get fifteen narrow
permission boundaries; you need one tool plus a pattern language.

**Read-only vs side-effecting is enforced at the permission layer, not
hand-rolled per tool.** The reference table marks each tool's
"Permission Required" column directly (`Read`: No, `Write`: Yes, `Bash`: Yes,
`Grep`/`Glob`: No). This is the same signal as OpenMate's `ToolSpec.side_effecting`
— but in Claude Code it actually gates execution (prompts the user, or is
checked against `allow`/`deny`/`ask` rules) rather than being a label nothing
reads.

**Sandboxing is a separate, additive layer under the tool, not a different
tool.** `Bash` itself didn't fork into "SafeBash" and "Bash" — the same tool
gained OS-level filesystem/network isolation (`/en/sandboxing`, bubblewrap on
Linux, Seatbelt on macOS) as an opt-in hardening layer. Sandbox network rules
are configured independently of permission rules (a domain allowed for
`WebFetch` still needs its own sandbox network rule). This separation —
*capability* (what the tool can address) vs *permission* (what the user
allows) vs *isolation* (what the OS lets the process touch even if allowed)
— is the architectural point most worth importing.

**Subagents scope tools by composition, not by a separate tool-design
exercise.** A subagent either inherits every parent tool, gets an explicit
allowlist (`tools`), or gets the parent set minus a list (`disallowedTools`).
Skills similarly declare `allowed-tools` in frontmatter and the `Skill` tool
enforces it. Scoping is declarative and enforced at the dispatch boundary,
the same mechanism reused for two different extensibility surfaces
(subagents and skills).

**Read/Edit/Write have hard, mechanical safety invariants, not judgment
calls.** Edit requires read-before-edit plus exact, unique string match.
Write requires the file to have been read first if it already exists. These
aren't guardrail-engine policy — they're load-bearing preconditions baked
into the tool's `invoke` logic itself, cheap to check and impossible to
bypass via prompting.

**Lossy-by-design tools say so.** `WebFetch` explicitly degrades the page
through a small extraction model and documents that "a result that says a
page does not mention something may only mean the prompt did not ask about
it" — i.e., the tool's docstring manages the calling model's trust in the
tool's own output. `WebSearch` returns titles/URLs only, deferring full
content to a follow-up `WebFetch`, keeping the two concerns (find vs. read)
separate.

## 3. Side-by-side: Claude Code vs. OpenMate today

| Concern | Claude Code | OpenMate (as implemented, not as designed) |
|---|---|---|
| Directory listing | No dedicated tool; `ls` via `Bash` | Dedicated `list_directory` tool, path-confined |
| File read | `Read`, handles text/image/PDF/notebook, paginates large files | `read_file`, plain text only, hard truncation at 100KB with no pagination continuation |
| File write | `Write` requires read-before-overwrite; `Edit` requires exact-match + uniqueness | `write_file` overwrites unconditionally, no read-before-write check |
| Shell | `Bash`, sandboxable (OS-level), permission-gated, command-pattern rules | `ShellTool` over `LocalSandbox`, explicitly documented as **not** isolated, no path confinement, no command-pattern rules |
| Side-effecting flag | Enforced — gates a permission prompt | `ToolSpec.side_effecting` exists, set correctly everywhere, **read by nothing** (confirmed: `executor.py`'s dispatch loop never checks it) |
| Tool-name addressing for permissions | Every tool name is a permission key, with per-tool specifier grammar | No permission/allow-deny layer exists at all; the only gate is a static `--allow-write` CLi flag that removes one tool from the list |
| Subagent tool scoping | Declarative `tools`/`disallowedTools`, enforced at dispatch | N/A — OpenMate has no subagent/orchestration concept yet (consistent with PoC scope; not a gap to fix now) |
| Skill tool scoping | `allowed-tools` frontmatter, enforced by the `Skill` tool | `SkillManifest.tools` is parsed but **never read again** anywhere (confirmed via grep across `skill.py` and `assemble.py`) — a skill can call any tool in the agent's full toolset |
| MCP tool scoping | Tools addressable like any other tool under the same permission grammar | `MCPProvider` actually enforces `scope_allowlist` by filtering `tools()` — this is the one scoping mechanism in OpenMate that genuinely works |
| Sandboxing | OS-level (bubblewrap/Seatbelt), additive layer under `Bash`, independent of permission rules | None; `LocalSandbox` is bare `asyncio.create_subprocess_shell` with a timeout |
| Search | `WebSearch` (find) + `WebFetch` (read), separated concerns | `fetch_url` only — no way to discover a URL from a query |
| Default entrypoint wiring | All tools reachable through the one running agent | `cli.py` builds `Agent` directly from a static built-in tool list; never calls `assemble()`, so `ShellProvider`/`SkillProvider`/`MCPProvider` are unreachable from the only user-facing entrypoint |

## 4. Recommendations, in priority order

Each maps to a specific file and is scoped to be implementable independently.

### R1 — Wire `cli.py` through `assemble()`
**File:** `openmate/cli.py`, `_build_agent`.
Currently builds `Agent` from a hardcoded `all_tools()`/`read_only_tools()` list.
Change it to construct an `assemble()` context with `NativeProvider` (builtin
tools), `ShellProvider`, `SkillProvider` (discovering `skills/`), and
`MCPProvider` when servers are configured. Without this, every fix below is
inert for an actual `openmate run`/`chat` user — this is the prerequisite, not
optional polish.

### R2 — Build the permission-rule layer Claude Code has and OpenMate doesn't
**Files:** new `openmate/kernel/permissions.py` (or similar), consumed by
`kernel/executor.py`.
Add an allow/deny/ask rule set keyed by tool name, with a command-pattern
specifier for `shell` (mirroring `Bash(npm run *)`) and a path-pattern
specifier for `read_file`/`write_file` (mirroring `Read(~/secrets/**)`).
`executor.py`'s `dispatch()` checks this before `tool.invoke()`. This single
mechanism replaces both the dead `ToolSpec.side_effecting` check and the
unenforced `SkillManifest.tools` allowlist — both become instances of "check
the rule set for this tool name" rather than two different ad hoc
mechanisms.

### R3 — Enforce `SkillManifest.tools` at dispatch
**Files:** `openmate/skills/skill.py`, `openmate/kernel/executor.py`.
Once R2's rule layer exists, threading `active_skills` (already recorded in
`ctx.state.scratch["active_skills"]` by `LoadSkillTool`) through to a
per-call check against each active skill's declared `tools` list is a small
addition. Until R2 lands, even a standalone version of this check (skill
allowlist only, no broader permission system) is worth doing — it's the gap
most directly exploitable by a skill author's mistake, independent of the
shell-safety story.

### R4 — Harden `shell` instead of multiplying read-only tools
**Files:** `openmate/tools/provider.py` (`LocalSandbox`, `ShellTool`).
Per the earlier conversation in this session: don't keep `list_directory`
and `read_file` as parallel bespoke tools indefinitely. Path-confine
`LocalSandbox` the same way `_safe_path()` confines `read_file`/`list_directory`/
`write_file` today, then let R2's command-pattern rules classify read-style
shell commands (`ls`, `cat`, `grep`, `find`) as low-risk and writes/network
as gated. This converges OpenMate toward Claude Code's "one powerful tool,
governed by rules" shape instead of "many narrow tools, each separately
safe." (`list_directory`/`read_file` can stay as thin, zero-risk
conveniences in the meantime — removing them is not urgent, consolidating
shell's safety story is.)

### R5 — Read-before-write invariant on `write_file`
**File:** `openmate/adapters/tools/builtin.py`, `write_file`.
Cheap, mechanical, and directly borrowed from Claude's `Write` tool: track
which paths have been read in this run's scratch state and refuse to
overwrite an existing, unread file. This catches a whole class of "the model
clobbered a file it never looked at" failures for the cost of one dict
lookup.

### R6 — Add a `web_search` tool, separate from `fetch_url`
**File:** new tool in `openmate/adapters/tools/builtin.py` or a dedicated
provider.
Keep the Claude Code separation: search returns titles/URLs only (cheap,
fast, many results), fetch reads one page in full (expensive, lossy,
targeted). Don't merge them into one "look this up" tool — the two-step
shape is what lets the model triage before paying the cost of a full fetch.

### R7 — File-size pagination for `read_file`, not just truncation
**File:** `openmate/adapters/tools/builtin.py`, `read_file`.
Currently hard-truncates at 100KB with no way to continue. Add an
`offset`/`limit` style continuation (mirroring `Read`'s `PARTIAL view`
pattern) so a large file is recoverable across multiple calls instead of
permanently losing the tail.

### What NOT to copy yet

Two things from Claude Code are explicitly out of scope for OpenMate's
current stage, and pulling them in early would be over-engineering:

- **OS-level sandboxing (bubblewrap/Seatbelt).** As established earlier in
  this project's review: proportionate for a single-user local PoC is the
  permission/approval layer (R2), not full process isolation. Revisit only if
  OpenMate moves to multi-user or exposed-service deployment.
- **The full hook system** (`PreToolUse`/`PostToolUse` arbitrary command
  hooks). OpenMate's interceptor chain (designed in `docs/02`, not yet built)
  is the right eventual home for this, but it's a bigger lift than R1–R7 and
  shouldn't block them.

## 5. Net effect if R1–R3 land

`cli.py` actually loads skills and shell through `assemble()` (R1), every
tool dispatch is checked against a named rule set instead of an unread flag
(R2), and a loaded skill is mechanically confined to the tools it declared
(R3). That closes the two gaps repeatedly flagged in this project's review —
the inert `side_effecting` flag and the unenforced skill allowlist — using
one shared mechanism modeled directly on how Claude Code already solved the
same problem, rather than two bespoke patches.
