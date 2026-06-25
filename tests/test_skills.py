"""Skills: frontmatter parsing, discovery, progressive disclosure, load_skill."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from openmate.skills.skill import (
    LoadSkillTool,
    SkillProvider,
    SkillRegistry,
    parse_frontmatter,
)

_EMAIL_SKILLS = Path(__file__).resolve().parent.parent / "skills" / "email"


def test_frontmatter_handles_colons_inline_lists_and_block_lists():
    meta, body = parse_frontmatter(
        "---\n"
        "name: t\n"
        "description: Do X. Use when: asked to X.\n"  # colon inside the value
        "tags: [a, b, c]\n"  # inline list
        "tools:\n  - one\n  - two\n"  # block list
        "---\n"
        "# Body\ntext"
    )
    assert meta["name"] == "t"
    assert meta["description"] == "Do X. Use when: asked to X."
    assert meta["tags"] == ["a", "b", "c"]
    assert meta["tools"] == ["one", "two"]
    assert body.startswith("# Body")


def test_no_frontmatter_returns_empty_meta():
    meta, body = parse_frontmatter("# just markdown")
    assert meta == {} and body == "# just markdown"


def _write_skill(root: Path, name: str, desc: str, *, resource: str | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: {desc}\n---\n# {name}\nbody for {name}",
        encoding="utf-8",
    )
    if resource:
        (d / resource).write_text("reference content", encoding="utf-8")
    return d


def test_discover_cards_and_get(tmp_path):
    _write_skill(tmp_path, "alpha", "the alpha skill")
    _write_skill(tmp_path, "beta", "the beta skill", resource="ref.md")
    reg = SkillRegistry()
    reg.discover(tmp_path)

    assert {c.name for c in reg.cards()} == {"alpha", "beta"}
    assert "body for alpha" in reg.get("alpha").body()
    # a bundled resource shows up in the rendered (level-2) output
    assert "ref.md" in reg.get("beta").render()


def test_resource_read_is_confined_to_skill_dir(tmp_path):
    _write_skill(tmp_path, "alpha", "the alpha skill", resource="ref.md")
    skill = SkillRegistry()
    skill.discover(tmp_path)
    s = skill.get("alpha")
    assert s.resource("ref.md") == b"reference content"
    try:
        s.resource("../../etc/passwd")
    except ValueError:
        return
    raise AssertionError("expected a ValueError for an out-of-dir resource path")


async def test_load_skill_tool_returns_body_and_records_activation():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    tool = LoadSkillTool(reg)
    ctx = SimpleNamespace(state=SimpleNamespace(scratch={}))

    res = await tool.invoke({"name": "triage-inbox"}, ctx)
    assert not res.is_error
    assert "Triage the inbox" in res.content[0].text
    assert ctx.state.scratch["active_skills"] == ["triage-inbox"]


async def test_load_skill_unknown_is_recoverable_error():
    reg = SkillRegistry()
    reg.discover(_EMAIL_SKILLS)
    res = await LoadSkillTool(reg).invoke(
        {"name": "nope"}, SimpleNamespace(state=SimpleNamespace(scratch={}))
    )
    assert res.is_error
    assert "triage-inbox" in res.content[0].text  # lists what's available


async def test_skill_provider_exposes_load_tool_and_cards():
    p = SkillProvider([_EMAIL_SKILLS])
    await p.setup()
    tools = await p.tools()
    assert [t.spec.name for t in tools] == ["load_skill"]
    fragment = p.system_fragment()
    assert "## Available skills" in fragment
    assert "triage-inbox" in fragment and "summarize-thread" in fragment
