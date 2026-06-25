"""The SKILL.md system — portable procedures, loaded on demand (docs/14).

A **skill** is a directory with a ``SKILL.md`` (YAML frontmatter + Markdown body)
plus optional bundled scripts/resources. It's a *procedure*, not a tool: it tells
the agent **how** to do something and **composes** tools (Shell, MCP) rather than
replacing them. OpenMate adopts the open SKILL.md standard, so skills authored for
Claude/Cursor/Codex load unchanged.

**Progressive disclosure** keeps context cheap by revealing a skill in three levels:

1. *cards* — name + description in the system prompt (``SkillProvider.system_fragment``),
   all the model needs to *decide* a skill applies.
2. *body* — the full instructions, injected only when ``load_skill(name)`` is called.
3. *resources* — bundled files/scripts, read or run only when the body calls for them.

The frontmatter parser here is a tiny self-contained subset of YAML (no third-party
dependency): ``key: value``, inline ``[a, b]`` lists, and ``- item`` block lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..kernel.types import TextPart
from ..ports.tool import ToolResult, ToolSpec

if TYPE_CHECKING:
    from ..kernel.types import RunContext


# --- frontmatter parsing ------------------------------------------------------
def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in ("'", '"'):
        return s[1:-1]
    return s


def _parse_simple_yaml(lines: list[str]) -> dict:
    """Parse the small YAML subset SKILL.md frontmatter uses."""
    meta: dict = {}
    key: str | None = None
    for raw in lines:
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        stripped = raw.strip()
        if stripped.startswith("- ") and key is not None:
            meta.setdefault(key, [])
            if isinstance(meta[key], list):
                meta[key].append(_unquote(stripped[2:]))
            continue
        if ":" in raw:
            k, _, v = raw.partition(":")
            key = k.strip()
            v = v.strip()
            if v == "":
                meta[key] = []  # a block list (filled by following `- ` items) or empty
            elif v.startswith("[") and v.endswith("]"):
                inner = v[1:-1].strip()
                meta[key] = [_unquote(x) for x in inner.split(",") if x.strip()]
            else:
                meta[key] = _unquote(v)
    return meta


def parse_frontmatter(text: str) -> tuple[dict, str]:
    """Split ``SKILL.md`` into (frontmatter dict, markdown body)."""
    if not text.startswith("---"):
        return {}, text
    lines = text.splitlines()
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
    if end is None:
        return {}, text
    meta = _parse_simple_yaml(lines[1:end])
    body = "\n".join(lines[end + 1 :]).lstrip("\n")
    return meta, body


# --- model -------------------------------------------------------------------
@dataclass
class SkillManifest:
    """Parsed from SKILL.md frontmatter. ``description`` drives soft selection."""

    name: str
    description: str
    version: str | None = None
    tools: list[str] = field(default_factory=list)  # allowlist (least privilege)
    resources: list[str] = field(default_factory=list)  # bundled files, relative paths


@dataclass
class Skill:
    manifest: SkillManifest
    path: Path  # the SKILL.md file

    @property
    def dir(self) -> Path:
        return self.path.parent

    def body(self) -> str:
        """The SKILL.md markdown body (loaded lazily — disclosure level 2)."""
        _, body = parse_frontmatter(self.path.read_text(encoding="utf-8"))
        return body

    def _resource_index(self) -> list[str]:
        if self.manifest.resources:
            return list(self.manifest.resources)
        return [
            str(p.relative_to(self.dir))
            for p in sorted(self.dir.rglob("*"))
            if p.is_file() and p.name != "SKILL.md"
        ]

    def render(self) -> str:
        """The body plus an index of bundled resources (what ``load_skill`` returns)."""
        out = self.body()
        index = self._resource_index()
        if index:
            listing = "\n".join(f"- {r}" for r in index)
            out += (
                "\n\n---\n## Bundled resources\n"
                "Read one with a file tool, or run a script via `shell`, only when needed:\n"
                + listing
            )
        return out

    def resource(self, rel: str) -> bytes:
        """Read a bundled resource (disclosure level 3), confined to the skill dir."""
        target = (self.dir / rel).resolve()
        root = self.dir.resolve()
        if root != target and root not in target.parents:
            raise ValueError(f"resource '{rel}' is outside the skill directory")
        return target.read_bytes()


class SkillRegistry:
    """Discovers skills on disk and serves their cards/bodies."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def discover(self, *paths: str | Path) -> None:
        """Scan paths for ``SKILL.md`` files (parses frontmatter only — cheap).

        A path may be a single skill directory or a root holding many; nested
        layouts (``skills/email/triage-inbox/SKILL.md``) are found recursively.
        """
        for p in paths:
            root = Path(p)
            if not root.exists():
                continue
            for md in sorted(root.rglob("SKILL.md")):
                meta, _ = parse_frontmatter(md.read_text(encoding="utf-8"))
                name = meta.get("name")
                if not name:
                    continue
                manifest = SkillManifest(
                    name=name,
                    description=meta.get("description", ""),
                    version=meta.get("version") or None,
                    tools=list(meta.get("tools") or []),
                    resources=list(meta.get("resources") or []),
                )
                self._skills[name] = Skill(manifest=manifest, path=md)

    def cards(self) -> list[SkillManifest]:
        """The ~100-token metadata for each skill (disclosure level 1)."""
        return [s.manifest for s in self._skills.values()]

    def get(self, name: str) -> Skill:
        return self._skills[name]


class LoadSkillTool:
    """The tool the model calls to pull a skill's full instructions into context."""

    def __init__(self, registry: SkillRegistry) -> None:
        self.registry = registry
        self.spec = ToolSpec(
            name="load_skill",
            description=(
                "Load a skill's full instructions when its listed description matches "
                "the task. Pass the skill name from the available-skills list."
            ),
            parameters={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "the skill name"}},
                "required": ["name"],
            },
            side_effecting=False,
            idempotent=True,
        )

    async def invoke(self, args: dict, ctx: "RunContext") -> ToolResult:
        name = (args or {}).get("name")
        try:
            skill = self.registry.get(name)
        except KeyError:
            available = ", ".join(s.name for s in self.registry.cards()) or "none"
            return ToolResult(
                [TextPart(f"unknown skill '{name}'. Available skills: {available}")],
                is_error=True,
            )
        # Record activation. Full least-privilege tool scoping is docs/14 §Phase 4;
        # the PoC just notes which skill is driving for tracing/inspection.
        try:
            ctx.state.scratch.setdefault("active_skills", []).append(name)
        except Exception:  # noqa: BLE001 — activation tracking is best-effort
            pass
        return ToolResult([TextPart(skill.render())])


class SkillProvider:
    """A ``ToolProvider`` that sources skills: yields ``load_skill`` + injects cards."""

    name = "skills"

    def __init__(self, paths: list[str | Path]) -> None:
        self._paths = list(paths)
        self.registry = SkillRegistry()

    async def setup(self) -> None:
        self.registry.discover(*self._paths)

    async def tools(self) -> list:
        return [LoadSkillTool(self.registry)]

    def system_fragment(self) -> str | None:
        cards = self.registry.cards()
        if not cards:
            return None
        lines = "\n".join(f"- **{c.name}**: {c.description}" for c in cards)
        return (
            "## Available skills\n"
            "Call `load_skill(name)` to load a skill's full instructions when its "
            "description matches the task.\n" + lines
        )

    async def teardown(self) -> None:
        return None
