"""JSON (de)serialization for the kernel types.

A ``RunState`` is a portable checkpoint: ``from_jsonable(to_jsonable(x)) == x``
holds for every state, which is what enables durability, time-travel, and
replay. Parts use a tagged-union encoding (``{"_t": ...}``).
"""

from __future__ import annotations

from typing import Any

from .types import (
    Message,
    Part,
    RunState,
    TextPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)


def part_to_jsonable(p: Part) -> dict[str, Any]:
    if isinstance(p, TextPart):
        return {"_t": "text", "text": p.text}
    if isinstance(p, ToolCallPart):
        return {"_t": "tool_call", "id": p.id, "name": p.name, "args": p.args}
    if isinstance(p, ToolResultPart):
        return {
            "_t": "tool_result",
            "call_id": p.call_id,
            "content": [part_to_jsonable(c) for c in p.content],
            "is_error": p.is_error,
        }
    raise TypeError(f"cannot serialize part: {p!r}")


def part_from_jsonable(d: dict[str, Any]) -> Part:
    t = d["_t"]
    if t == "text":
        return TextPart(d["text"])
    if t == "tool_call":
        return ToolCallPart(d["id"], d["name"], d["args"])
    if t == "tool_result":
        return ToolResultPart(
            d["call_id"],
            [part_from_jsonable(c) for c in d["content"]],
            d.get("is_error", False),
        )
    raise ValueError(f"unknown part tag: {t!r}")


def message_to_jsonable(m: Message) -> dict[str, Any]:
    return {
        "role": m.role,
        "content": [part_to_jsonable(p) for p in m.content],
        "name": m.name,
        "metadata": m.metadata,
    }


def message_from_jsonable(d: dict[str, Any]) -> Message:
    return Message(
        role=d["role"],
        content=[part_from_jsonable(p) for p in d["content"]],
        name=d.get("name"),
        metadata=d.get("metadata") or {},
    )


def state_to_jsonable(s: RunState) -> dict[str, Any]:
    return {
        "thread_id": s.thread_id,
        "messages": [message_to_jsonable(m) for m in s.messages],
        "scratch": s.scratch,
        "step": s.step,
        "status": s.status,
        "usage": {
            "prompt_tokens": s.usage.prompt_tokens,
            "completion_tokens": s.usage.completion_tokens,
            "cost_usd": s.usage.cost_usd,
            "wall_ms": s.usage.wall_ms,
        },
        "cursor": s.cursor,
        "rev": s.rev,
    }


def state_from_jsonable(d: dict[str, Any]) -> RunState:
    u = d.get("usage") or {}
    return RunState(
        thread_id=d["thread_id"],
        messages=[message_from_jsonable(m) for m in d["messages"]],
        scratch=d.get("scratch") or {},
        step=d.get("step", 0),
        status=d.get("status", "running"),
        usage=Usage(
            prompt_tokens=u.get("prompt_tokens", 0),
            completion_tokens=u.get("completion_tokens", 0),
            cost_usd=u.get("cost_usd", 0.0),
            wall_ms=u.get("wall_ms", 0.0),
        ),
        cursor=d.get("cursor") or {},
        rev=d.get("rev", 0),
    )
