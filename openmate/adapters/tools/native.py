"""Native Python tools.

``@tool`` turns a plain function into a ``Tool``: it derives the JSON-Schema
``parameters`` from the function's type hints and uses the docstring as the
model-facing description. Bad arguments produce a recoverable error result, not
a crash.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, get_type_hints

from ...kernel.types import RunContext, TextPart
from ...ports.tool import Tool, ToolResult, ToolSpec

_JSON_TYPES: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_signature(fn: Callable) -> dict:
    sig = inspect.signature(fn)
    try:
        hints = get_type_hints(fn)
    except Exception:  # noqa: BLE001 — fall back to no hints
        hints = {}
    properties: dict[str, dict] = {}
    required: list[str] = []
    for name, param in sig.parameters.items():
        if name in ("ctx", "self") or param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        py_type = hints.get(name, str)
        json_type = _JSON_TYPES.get(py_type, "string")
        properties[name] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(name)
    return {"type": "object", "properties": properties, "required": required}


class FunctionTool:
    def __init__(self, fn: Callable, spec: ToolSpec) -> None:
        self.fn = fn
        self.spec = spec
        self._wants_ctx = "ctx" in inspect.signature(fn).parameters

    async def invoke(self, args: dict, ctx: RunContext) -> ToolResult:
        call_args = dict(args or {})
        missing = [k for k in self.spec.parameters.get("required", []) if k not in call_args]
        if missing:
            return ToolResult(
                [TextPart(f"missing required argument(s): {', '.join(missing)}")],
                is_error=True,
            )
        if self._wants_ctx:
            call_args["ctx"] = ctx
        try:
            result = self.fn(**call_args)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as e:
            return ToolResult([TextPart(f"invalid arguments: {e}")], is_error=True)
        except Exception as e:  # noqa: BLE001 — a tool failure is recoverable, not a crash
            return ToolResult([TextPart(f"{type(e).__name__}: {e}")], is_error=True)
        return ToolResult([TextPart("" if result is None else str(result))])


def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    side_effecting: bool = True,
    timeout_s: float = 30.0,
    idempotent: bool = False,
) -> Any:
    """Decorator that wraps a function as a ``Tool``.

    Usage::

        @tool(side_effecting=False)
        def calculator(expression: str) -> str:
            "Evaluate an arithmetic expression."
            ...
    """

    def wrap(f: Callable) -> Tool:
        desc = description or (inspect.getdoc(f) or f.__name__)
        spec = ToolSpec(
            name=name or f.__name__,
            description=desc,
            parameters=_schema_from_signature(f),
            side_effecting=side_effecting,
            timeout_s=timeout_s,
            idempotent=idempotent,
        )
        return FunctionTool(f, spec)

    return wrap(fn) if fn is not None else wrap
