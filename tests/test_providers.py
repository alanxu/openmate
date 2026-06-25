"""Tool providers + assemble(): composition, lifecycle, and least-privilege scoping."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from helpers import make_services

from openmate.adapters.models.fake import FakeModel
from openmate.adapters.tools.builtin import calculator
from openmate.adapters.tools.mcp_client import MCPServerSpec
from openmate.agent.assemble import assemble
from openmate.skills.skill import SkillProvider
from openmate.tools.provider import MCPProvider, NativeProvider, ShellProvider

_FAKE_GMAIL = Path(__file__).resolve().parent.parent / "servers" / "gmail" / "fake_server.py"
_EMAIL_SKILLS = Path(__file__).resolve().parent.parent / "skills" / "email"
_needs_mcp = pytest.mark.skipif(
    __import__("importlib").util.find_spec("mcp") is None, reason="mcp SDK not installed"
)


async def test_native_provider_contributes_its_tools():
    p = NativeProvider([calculator])
    await p.setup()
    assert [t.spec.name for t in await p.tools()] == ["calculator"]
    await p.teardown()


async def test_shell_provider_runs_a_command():
    p = ShellProvider()
    tool = (await p.tools())[0]
    assert tool.spec.name == "shell" and tool.spec.side_effecting
    res = await tool.invoke({"command": "echo hello-shell"}, ctx=None)
    assert not res.is_error
    assert "hello-shell" in res.content[0].text
    assert "[exit 0]" in res.content[0].text


async def test_shell_nonzero_exit_is_error():
    res = await ShellProvider().sandbox.run("exit 3", timeout=5)
    assert res[0] == 3


async def test_assemble_unions_tools_and_merges_system_fragments():
    svc, _ = make_services()
    providers = [
        NativeProvider([calculator], system_fragment="## Native\nmath available"),
        SkillProvider([_EMAIL_SKILLS]),
    ]
    async with assemble(
        name="t", system="BASE SYSTEM", model=FakeModel([]), services=svc, providers=providers
    ) as agent:
        names = {t.spec.name for t in agent.tools}
        assert "calculator" in names and "load_skill" in names
        # both the base system and each provider fragment land in instructions
        assert "BASE SYSTEM" in agent.instructions
        assert "## Native" in agent.instructions
        assert "## Available skills" in agent.instructions


async def test_assemble_tears_providers_down_even_on_error():
    events = []

    class SpyProvider:
        name = "spy"

        async def setup(self):
            events.append("setup")

        async def tools(self):
            return []

        def system_fragment(self):
            return None

        async def teardown(self):
            events.append("teardown")

    svc, _ = make_services()
    try:
        async with assemble(
            name="t", system="s", model=FakeModel([]), services=svc, providers=[SpyProvider()]
        ):
            raise RuntimeError("boom inside the context")
    except RuntimeError:
        pass
    assert events == ["setup", "teardown"]  # teardown ran despite the error


async def test_assemble_forwards_policy_to_agent():
    svc, _ = make_services()
    async with assemble(
        name="t",
        system="s",
        model=FakeModel([]),
        services=svc,
        providers=[NativeProvider([calculator])],
        max_steps=5,
    ) as agent:
        assert agent.max_steps == 5


@_needs_mcp
async def test_mcp_provider_scope_allowlist_hides_unlisted_tools():
    gmail = MCPServerSpec(name="gmail", command=[sys.executable, str(_FAKE_GMAIL)])
    provider = MCPProvider([gmail], scope_allowlist=["gmail_search", "gmail_get_message"])
    await provider.setup()
    try:
        names = {t.spec.name for t in await provider.tools()}
        assert names == {"gmail_search", "gmail_get_message"}
        assert "gmail_create_draft" not in names  # excluded by least privilege
    finally:
        await provider.teardown()
