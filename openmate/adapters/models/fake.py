"""A deterministic fake model for tests, offline demos, and replay.

Construct it with a script of responses (or plain helpers) and it returns them in
order — the basis for the determinism guarantee (same script -> byte-identical
event log), no API key required.
"""

from __future__ import annotations

from ...kernel.types import Message, TextPart, ToolCallPart, Usage
from ...ports.model import ModelCapabilities, ModelRequest, ModelResponse, StreamDelta

_CAPS = ModelCapabilities(
    tool_calling=True,
    parallel_tools=True,
    structured_output=True,
    streaming=True,
    max_context=1_000_000,
)


def text_response(text: str, *, tokens: int = 1) -> ModelResponse:
    return ModelResponse(
        message=Message("assistant", [TextPart(text)]),
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens),
        finish_reason="stop",
    )


def tool_call_response(call_id: str, name: str, args: dict, *, tokens: int = 1) -> ModelResponse:
    return ModelResponse(
        message=Message("assistant", [ToolCallPart(call_id, name, args)]),
        usage=Usage(prompt_tokens=tokens, completion_tokens=tokens),
        finish_reason="tool_calls",
    )


class FakeModel:
    name = "fake"
    capabilities = _CAPS

    def __init__(self, script: list[ModelResponse]) -> None:
        self._script = list(script)
        self._i = 0
        self.requests: list[ModelRequest] = []  # captured for assertions

    async def generate(self, req: ModelRequest) -> ModelResponse:
        self.requests.append(req)
        if self._i >= len(self._script):
            # Out of script -> end the run cleanly rather than looping forever.
            return text_response("(fake model: end of script)")
        resp = self._script[self._i]
        self._i += 1
        return resp

    async def stream(self, req: ModelRequest):
        """Scripted streaming: one text delta (if any) then the terminal ``done``.

        Reuses ``generate`` so the script advances once per call, keeping the
        streamed and non-streamed paths deterministic and in lock-step.
        """
        resp = await self.generate(req)
        if resp.message.text:
            yield StreamDelta("text", resp.message.text)
        yield StreamDelta("done", resp)
