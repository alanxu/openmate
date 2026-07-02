"""Anthropic (Claude) model adapter.

This is the "Claude model" adapter. Because MiniMax ships an **Anthropic-compatible**
endpoint, the exact same adapter talks to a MiniMax model simply by pointing
``base_url`` at ``https://api.minimax.io/anthropic`` and using a MiniMax API key.
That is the whole point of the Model port: the kernel never knows the difference.

    AnthropicModel("MiniMax-M2",
                   api_key="sk-...",                       # a MiniMax key
                   base_url="https://api.minimax.io/anthropic")

All this module owns is the translation layer between OpenMate ``Message``/``Part``
values and the Anthropic Messages API wire format. Keys are read from the
environment at construction, never from prompts or state.
"""

from __future__ import annotations

import json
import os
from typing import Any, AsyncIterator

from ...kernel.errors import ConfigError, ProviderError
from ...kernel.types import (
    Message,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolResultPart,
    Usage,
)
from ...ports.model import ModelCapabilities, ModelRequest, ModelResponse, StreamDelta
from ...ports.tool import ToolSpec

_DEFAULT_CAPS = ModelCapabilities(
    tool_calling=True,
    parallel_tools=True,
    structured_output=False,
    vision=False,
    thinking=True,  # MiniMax-M2 / Claude expose reasoning; captured as ThinkingPart
    prompt_caching=False,
    streaming=True,
    max_context=200_000,
    max_output=8192,
)

_STOP_REASON_MAP = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "max_tokens": "length",
    "tool_use": "tool_calls",
    "refusal": "filter",
}


class AnthropicModel:
    """A ``Model`` backed by the Anthropic Messages API (or any compatible endpoint)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        capabilities: ModelCapabilities | None = None,
        client: Any = None,  # inject a stub in tests
        max_retries: int = 2,
    ) -> None:
        self.name = model
        self._model = model
        self.capabilities = capabilities or _DEFAULT_CAPS

        if client is not None:
            self._client = client
            return

        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")
        base_url = base_url or os.environ.get("ANTHROPIC_BASE_URL")
        if not api_key:
            raise ConfigError(
                "No API key found. Set ANTHROPIC_API_KEY (your MiniMax key) in the "
                "environment or pass api_key=... to AnthropicModel."
            )
        try:
            from anthropic import AsyncAnthropic
        except ImportError as e:  # pragma: no cover
            raise ConfigError(
                "The 'anthropic' package is required. Install it with: pip install anthropic"
            ) from e

        kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": max_retries}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = AsyncAnthropic(**kwargs)

    # --- public port surface ---------------------------------------------------
    async def generate(self, req: ModelRequest) -> ModelResponse:
        system, messages = self._messages_to_wire(req.messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if req.tools:
            kwargs["tools"] = [self._tool_to_wire(t) for t in req.tools]
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop:
            kwargs["stop_sequences"] = req.stop

        try:
            raw = await self._client.messages.create(**kwargs)
        except Exception as e:  # noqa: BLE001 — normalize provider errors
            raise ProviderError(f"{type(e).__name__}: {e}") from e

        return self._response_from_wire(raw, wire=kwargs)

    async def stream(self, req: ModelRequest) -> AsyncIterator[StreamDelta]:
        """Token-level streaming over the same Messages API (``stream=True`` / SSE).

        Yields ``text``/``thinking`` deltas as they arrive, then a terminal
        ``done`` delta carrying the fully-assembled ``ModelResponse`` — identical
        to what ``generate`` returns, so callers reassemble losslessly. The wire
        kwargs are built the same way as ``generate`` (duplicated, not shared, to
        keep ``generate`` untouched).
        """
        system, messages = self._messages_to_wire(req.messages)
        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": req.max_tokens,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if req.tools:
            kwargs["tools"] = [self._tool_to_wire(t) for t in req.tools]
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.stop:
            kwargs["stop_sequences"] = req.stop

        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if getattr(event, "type", None) != "content_block_delta":
                        continue
                    d = event.delta
                    if getattr(d, "type", None) == "text_delta":
                        yield StreamDelta("text", d.text)
                    elif getattr(d, "type", None) == "thinking_delta":
                        yield StreamDelta("thinking", getattr(d, "thinking", "") or "")
                final = await stream.get_final_message()
        except Exception as e:  # noqa: BLE001 — normalize provider errors
            raise ProviderError(f"{type(e).__name__}: {e}") from e

        yield StreamDelta("done", self._response_from_wire(final, wire=kwargs))

    # --- translation layer -----------------------------------------------------
    def _messages_to_wire(self, messages: list[Message]) -> tuple[str, list[dict]]:
        """OpenMate messages -> (system string, Anthropic message list).

        Anthropic carries the system prompt out-of-band and expects tool results
        as ``tool_result`` blocks on a *user* message.
        """
        system_parts: list[str] = []
        wire: list[dict] = []
        for m in messages:
            if m.role == "system":
                if m.text:
                    system_parts.append(m.text)
                continue
            if m.role == "tool":
                blocks = [
                    {
                        "type": "tool_result",
                        "tool_use_id": p.call_id,
                        "content": self._render_result_content(p),
                        "is_error": p.is_error,
                    }
                    for p in m.content
                    if isinstance(p, ToolResultPart)
                ]
                wire.append({"role": "user", "content": blocks})
                continue

            blocks = []
            for p in m.content:
                if isinstance(p, TextPart):
                    if p.text:
                        blocks.append({"type": "text", "text": p.text})
                elif isinstance(p, ToolCallPart):
                    blocks.append(
                        {"type": "tool_use", "id": p.id, "name": p.name, "input": p.args}
                    )
                # ThinkingPart is intentionally not re-sent: replaying thinking blocks
                # needs their signatures and only matters when extended thinking is
                # explicitly enabled. We keep it for display/audit, not on the wire.
            if not blocks:  # Anthropic rejects empty content
                blocks = [{"type": "text", "text": "(no content)"}]
            wire.append({"role": m.role, "content": blocks})

        return "\n\n".join(system_parts), wire

    @staticmethod
    def _render_result_content(p: ToolResultPart) -> str:
        text = "".join(c.text for c in p.content if isinstance(c, TextPart))
        return text or "(empty result)"

    @staticmethod
    def _tool_to_wire(spec: ToolSpec) -> dict:
        return {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.parameters,
        }

    def _response_from_wire(self, raw: Any, *, wire: dict | None = None) -> ModelResponse:
        parts: list[Any] = []
        for block in getattr(raw, "content", []) or []:
            btype = getattr(block, "type", None)
            if btype == "text":
                parts.append(TextPart(getattr(block, "text", "")))
            elif btype == "tool_use":
                parts.append(
                    ToolCallPart(
                        id=getattr(block, "id", ""),
                        name=getattr(block, "name", ""),
                        args=dict(getattr(block, "input", {}) or {}),
                    )
                )
            elif btype == "thinking":
                parts.append(
                    ThinkingPart(
                        getattr(block, "thinking", "") or "",
                        getattr(block, "signature", None),
                    )
                )
            # other block types (e.g. redacted_thinking) are dropped

        usage_obj = getattr(raw, "usage", None)
        usage = Usage(
            prompt_tokens=getattr(usage_obj, "input_tokens", 0) or 0,
            completion_tokens=getattr(usage_obj, "output_tokens", 0) or 0,
        )
        stop_reason = getattr(raw, "stop_reason", None)
        finish = _STOP_REASON_MAP.get(stop_reason, "stop")
        if any(isinstance(p, ToolCallPart) for p in parts):
            finish = "tool_calls"

        # Stash the wire kwargs on the response object so the loop can attach
        # them to the previous ModelRequested event (one wire dict per call).
        if wire is not None:
            try:
                raw._openmate_wire = _jsonable(wire)
            except Exception:  # noqa: BLE001
                pass

        return ModelResponse(
            message=Message("assistant", parts),
            usage=usage,
            finish_reason=finish,  # type: ignore[arg-type]
            raw=raw,
        )


def _jsonable(obj: Any) -> Any:
    """Best-effort conversion to JSON-serializable form for wire kwargs."""
    return json.loads(json.dumps(obj, default=str))
