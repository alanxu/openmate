"""Tools: schema generation, invocation, and the calculator."""

from __future__ import annotations

from openmate.adapters.tools.builtin import calculator, current_time
from openmate.adapters.tools.native import tool


def test_schema_generated_from_signature():
    @tool
    def greet(name: str, times: int = 1) -> str:
        """Greet someone."""
        return "hi " * times + name

    params = greet.spec.parameters
    assert params["properties"]["name"]["type"] == "string"
    assert params["properties"]["times"]["type"] == "integer"
    assert params["required"] == ["name"]  # only args without defaults are required
    assert "Greet someone." in greet.spec.description


async def test_calculator_evaluates():
    res = await calculator.invoke({"expression": "2 * (3 + 4)"}, ctx=None)
    assert not res.is_error
    assert res.content[0].text == "14"


async def test_calculator_rejects_unsafe_input():
    res = await calculator.invoke({"expression": "__import__('os').system('echo hi')"}, ctx=None)
    assert res.is_error  # not valid arithmetic -> recoverable error, never executed


async def test_missing_required_arg_is_error():
    res = await calculator.invoke({}, ctx=None)
    assert res.is_error
    assert "expression" in res.content[0].text


async def test_no_arg_tool():
    res = await current_time.invoke({}, ctx=None)
    assert not res.is_error
    assert "T" in res.content[0].text  # ISO timestamp
