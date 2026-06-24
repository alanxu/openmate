# 03 — Model Port & Provider Abstraction

> The swappable LLM boundary. Part of OpenMate; see [architecture.md §14](architecture.md#14-the-model-llm-port--provider-abstraction). The single rule: **no code outside `adapters/models/*` knows which provider is in use.**

## Scope & responsibilities

The `Model` port normalizes generation, tool calling, structured output, streaming, and usage accounting across providers, and advertises **capabilities** so the loop adapts to differences instead of assuming a floor. Everything provider-specific (wire formats, Responses-vs-Chat conventions, reasoning exposure, prompt-cache controls, token accounting) is absorbed by an adapter. This module is also where routing, fallback, retries, and caching live, since those are model-call concerns.

---

## Core abstractions (class level)

```python
# openmate/ports/model.py
@dataclass(frozen=True)
class ModelCapabilities:
    tool_calling: bool; parallel_tools: bool; structured_output: bool
    vision: bool; audio: bool; thinking: bool; prompt_caching: bool
    streaming: bool; max_context: int; max_output: int

@dataclass
class ModelRequest:
    messages: list[Message]
    tools: list[ToolSpec] | None = None
    output_schema: type | None = None          # structured output target
    temperature: float | None = None
    max_tokens: int | None = None
    stop: list[str] | None = None
    cache_hints: "CacheHints | None" = None     # prompt-cache breakpoints
    extra: dict = field(default_factory=dict)   # escape hatch, namespaced per provider

@dataclass
class ModelResponse:
    message: Message                            # text / tool calls / thinking / data parts
    usage: Usage
    finish_reason: Literal["stop","tool_calls","length","filter"]
    raw: Any = None                             # provider payload, debugging only

class Model(Protocol):
    name: str
    capabilities: ModelCapabilities
    async def generate(self, req: ModelRequest) -> ModelResponse: ...
    def stream(self, req: ModelRequest) -> AsyncIterator["StreamDelta"]: ...

@dataclass
class StreamDelta:
    kind: Literal["text","thinking","tool_call_partial","usage","done"]
    data: Any
```

---

## Phase 0 — PoC (foundational)

**Goal:** one real adapter + a deterministic fake, `generate()` only, with normalized tool calling.

Ship: `Model` protocol, `OpenAICompatibleModel` (covers OpenAI **and** local servers like Ollama/vLLM/LM Studio via the same API — maximum coverage for one adapter), and `FakeModel` for tests.

```python
# openmate/adapters/models/openai_compat.py
class OpenAICompatibleModel(Model):
    def __init__(self, model: str, base_url: str, api_key: str, caps: ModelCapabilities): ...
    async def generate(self, req):
        wire = self._to_wire(req.messages, req.tools)        # OpenMate → provider JSON
        raw = await self._client.chat.completions.create(**wire)
        return self._from_wire(raw)                          # provider JSON → ModelResponse

# openmate/adapters/models/fake.py
class FakeModel(Model):
    def __init__(self, script: list[ModelResponse]): self._script = iter(script)
    async def generate(self, req): return next(self._script)  # scripted, deterministic
```

Key PoC work is the **translation layer**: mapping `Message`/`Part` to the provider's message/tool-call schema and back, including the awkward bits (assistant tool-call messages, `tool` role results keyed by `call_id`). Keys are loaded from env at construction (`ANTHROPIC_API_KEY`, etc.), never from prompts/state.

**PoC acceptance:** the runtime ([02](02-agent-loop-and-runtime.md)) runs identically against `FakeModel` and a live OpenAI-compatible endpoint; swapping is a one-line constructor change.

---

## Phase 1 — Streaming, native adapters & usage

- **Provider adapters:** `AnthropicModel` (Messages API, thinking blocks, prompt-cache breakpoints) and a first-class `OpenAIModel` (Responses API). Each declares its real `ModelCapabilities`.
- **Streaming:** implement `stream()` yielding `StreamDelta`s; the runtime forwards them as `ModelStreamed` events. Partial tool-call assembly (providers stream tool args as fragments) is normalized into complete `ToolCallPart`s.
- **Usage & cost:** every response carries token counts; a `PriceBook` maps `(provider, model) → $/token` to fill `Usage.cost_usd` for budgets ([12](12-production-and-reliability.md)).
- **Tokenizer port:** `Tokenizer.count(messages)` (tiktoken/provider-native/heuristic) for the context budget in [09](09-context-engineering.md).

---

## Phase 2 — Structured output & capability-aware degradation

Robust typed output is a core capability (the Pydantic-AI lesson), so support it three ways and pick by capability:

```python
class StructuredDecoder:
    async def decode(self, model: Model, req: ModelRequest, schema: type[T]) -> T:
        if model.capabilities.structured_output:          # native constrained decoding
            return await self._native(model, req, schema)
        return await self._json_mode_with_retry(model, req, schema)  # fallback

    async def _json_mode_with_retry(self, model, req, schema, max_tries=3):
        for _ in range(max_tries):
            resp = await model.generate(with_json_instructions(req, schema))
            try: return validate(resp.message.text, schema)
            except ValidationError as e:
                req = append_correction(req, resp, e)     # self-correction loop (P6)
        raise StructuredOutputError(...)
```

Other degradations the runtime relies on: **no native tool calling** → prompted ReAct format + parser ([05](05-planning-and-reasoning.md)); **no parallel tools** → serialize calls; **no vision** → caption images via a side model or error clearly.

---

## Phase 3 — Routing, fallback & reliability

A `RouterModel` *is* a `Model`, so routing is invisible to agents (architecture §14.3).

```python
class RouterModel(Model):
    def __init__(self, rules: list["RouteRule"], fallbacks: list[Model]): ...
    async def generate(self, req):
        model = self._select(req)                          # by task tag, size, cost, latency SLA
        return await self._with_resilience(model, req)
    async def _with_resilience(self, model, req):
        for m in [model, *self.fallbacks]:
            try: return await RetryPolicy().run(lambda: m.generate(req))
            except (RateLimit, ProviderError): self._breaker.trip(m); continue
        raise AllProvidersFailed(...)
```

Techniques: **cost routing** (cheap model for routine steps, strong for planning/reflection — pairs with [05](05-planning-and-reasoning.md)); **fallback** across providers on error/rate-limit; **retries** with jittered exponential backoff + timeouts; **circuit breaker** per backend; **client-side rate limiting**/token-bucket. A gateway-backed adapter (e.g. LiteLLM-style proxy) is one valid `RouterModel` implementation — an adapter, never a kernel dependency.

---

## Phase 4 — Caching, multimodal, batch & embeddings

- **Prompt caching:** `CacheHints` mark stable prefixes (system prompt, tool specs, long context) as cache breakpoints; the adapter translates to the provider's mechanism. Big latency/cost win on long-context agents.
- **Semantic response cache:** an optional wrapping `Model` that returns a cached completion for semantically-equal requests (interceptor form in [12](12-production-and-reliability.md)).
- **Multimodal generate:** images/audio in, images/audio out via the rich parts from [01](01-domain-model-and-kernel.md).
- **Batch API adapter:** for offline eval/ingest, a `BatchModel` submits many requests at lower cost/throughput priority ([11](11-observability-and-evaluation.md)).
- **Embeddings port** (sibling): `Embedder.embed(texts) -> list[Vector]`, used by RAG ([07](07-retrieval-rag.md)) and memory ([06](06-memory-and-state.md)); kept here because it's a model concern, with the same routing/caching machinery.

---

## Adapter capability matrix (fill per integration)

| Adapter | tool_calling | parallel | structured | vision | thinking | prompt_cache | streaming |
|---|---|---|---|---|---|---|---|
| `OpenAICompatibleModel` | ✓ | ✓ | json-mode | varies | ✗ | ✗ | ✓ |
| `OpenAIModel` (Responses) | ✓ | ✓ | native | ✓ | partial | ✓ | ✓ |
| `AnthropicModel` | ✓ | ✓ | tool-based | ✓ | ✓ | ✓ | ✓ |
| `FakeModel` | scripted | — | scripted | — | — | — | scripted |

## Testing & verification

- **Record/replay:** `RecordingModel` wraps a live model and captures responses to a cassette; `FakeModel` replays them — gives realistic deterministic tests and powers replay-eval ([11](11-observability-and-evaluation.md)).
- **Contract tests:** one shared suite runs against every adapter (round-trip a tool call, a structured output, a streamed completion) so adapters are interchangeable in fact, not just intent.
- **Degradation tests:** force `structured_output=False` and assert the json-mode+retry path still yields valid objects.

## Trade-offs & open questions

Native structured output vs. tool-based vs. json-mode (use native when available, else the retry decoder). How much of `extra`/provider escape hatches to expose without leaking vendor concepts upward (keep namespaced and rare). Whether embeddings deserve their own top-level module as RAG grows (currently colocated).
