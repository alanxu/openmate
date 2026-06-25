"""MCP integration: annotation→spec mapping, result shaping, and a stdio round-trip."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from openmate.adapters.tools.mcp_client import (
    MCPClient,
    MCPServerSpec,
    MCPToolAdapter,
    result_from_mcp,
    spec_from_mcp_tool,
)

_FAKE_GMAIL = Path(__file__).resolve().parent.parent / "servers" / "gmail" / "fake_server.py"


def _mcp_tool(name, *, read_only=False, idempotent=False, schema=None):
    ann = SimpleNamespace(readOnlyHint=read_only, idempotentHint=idempotent)
    return SimpleNamespace(
        name=name,
        description=f"the {name} tool",
        inputSchema=schema or {"type": "object", "properties": {}},
        annotations=ann,
    )


def test_read_only_annotation_maps_to_non_side_effecting():
    spec = spec_from_mcp_tool(_mcp_tool("gmail_search", read_only=True, idempotent=True))
    assert spec.side_effecting is False  # readOnlyHint -> skips the approval gate
    assert spec.idempotent is True


def test_write_tool_is_side_effecting():
    spec = spec_from_mcp_tool(_mcp_tool("gmail_create_draft", read_only=False))
    assert spec.side_effecting is True


def test_namespacing_is_idempotent_and_collision_safe():
    # prefix applied when missing...
    assert spec_from_mcp_tool(_mcp_tool("search"), namespace_prefix="gmail_").name == "gmail_search"
    # ...but never doubled when the server already namespaced
    assert (
        spec_from_mcp_tool(_mcp_tool("gmail_search"), namespace_prefix="gmail_").name
        == "gmail_search"
    )


def test_result_shaping_text_and_error():
    ok = result_from_mcp(SimpleNamespace(content=[SimpleNamespace(text="hello")], isError=False))
    assert not ok.is_error and ok.content[0].text == "hello"
    bad = result_from_mcp(SimpleNamespace(content=[SimpleNamespace(text="boom")], isError=True))
    assert bad.is_error


def test_result_shaping_falls_back_to_structured_content():
    res = result_from_mcp(
        SimpleNamespace(content=[], isError=False, structuredContent={"a": 1})
    )
    assert '"a": 1' in res.content[0].text


async def test_adapter_dispatches_under_remote_name():
    seen = {}

    class FakeSession:
        async def call_tool(self, name, args):
            seen["name"], seen["args"] = name, args
            return SimpleNamespace(content=[SimpleNamespace(text="ok")], isError=False)

    tool = MCPToolAdapter(
        FakeSession(), _mcp_tool("search"), namespace_prefix="gmail_"
    )
    assert tool.spec.name == "gmail_search"  # model sees the namespaced name
    res = await tool.invoke({"q": "hi"}, ctx=None)
    assert seen["name"] == "search"  # but the server is called by its real name
    assert res.content[0].text == "ok"


async def test_adapter_turns_exceptions_into_recoverable_errors():
    class BoomSession:
        async def call_tool(self, name, args):
            raise RuntimeError("network down")

    tool = MCPToolAdapter(BoomSession(), _mcp_tool("search"))
    res = await tool.invoke({}, ctx=None)
    assert res.is_error and "network down" in res.content[0].text


# --- end-to-end over a real stdio transport (needs the mcp SDK) --------------
pytestmark_e2e = pytest.mark.skipif(
    __import__("importlib").util.find_spec("mcp") is None, reason="mcp SDK not installed"
)


@pytestmark_e2e
async def test_stdio_roundtrip_against_fake_gmail():
    client = MCPClient()
    await client.connect(
        MCPServerSpec(name="gmail", command=[sys.executable, str(_FAKE_GMAIL)])
    )
    try:
        tools = {t.spec.name: t for t in await client.list_tools()}
        # the read tools are discovered and correctly flagged non-side-effecting
        assert tools["gmail_search"].spec.side_effecting is False
        assert tools["gmail_create_draft"].spec.side_effecting is True

        res = await tools["gmail_search"].invoke({"q": "is:unread", "max_results": 5}, ctx=None)
        assert not res.is_error
        assert "m1" in res.content[0].text  # the seeded unread message

        got = await tools["gmail_get_message"].invoke({"id": "m1"}, ctx=None)
        assert "agent loop review" in got.content[0].text
    finally:
        await client.close()


@pytestmark_e2e
async def test_pagination_via_page_token():
    client = MCPClient()
    await client.connect(
        MCPServerSpec(name="gmail", command=[sys.executable, str(_FAKE_GMAIL)])
    )
    try:
        search = {t.spec.name: t for t in await client.list_tools()}["gmail_search"]
        import json

        page1 = json.loads((await search.invoke({"q": "", "max_results": 1}, ctx=None)).content[0].text)
        assert page1["next_page_token"] is not None
        page2 = json.loads(
            (await search.invoke({"q": "", "max_results": 1, "page_token": page1["next_page_token"]}, ctx=None)).content[0].text
        )
        assert page1["items"][0]["id"] != page2["items"][0]["id"]  # a different page
    finally:
        await client.close()
