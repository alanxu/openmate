# OpenMate — Design Docs

Detailed design for OpenMate, a from-scratch, provider-agnostic, framework-interoperable AI agent and reference architecture. Start with **[architecture.md](architecture.md)** for the system overview, then read the numbered area docs (roughly in build order).

## How to read these

Every area doc follows the same shape:

- **Scope & responsibilities** — what the module owns and where it sits.
- **Core abstractions (class level)** — the Python `Protocol`s/dataclasses that define the module.
- **Phase 0 — PoC (foundational)** — the most foundational capability; what to build first.
- **Phase 1…N** — escalating techniques layered above the PoC (as many as the area warrants).
- **Testing & verification** and **Trade-offs & open questions**.

Phases are about *capability maturity*, not a delivery timeline — Phase 0 across all areas roughly equals the [architecture roadmap](architecture.md#21-roadmap) milestones M0–M2.

## Index

| # | Doc | Owns | Arch §|
|---|---|---|---|
| — | [architecture.md](architecture.md) | System overview, principles, layers, roadmap | all |
| 01 | [Domain Model & Kernel](01-domain-model-and-kernel.md) | Core types, events, state, serialization | §5 |
| 02 | [Agent Loop & Runtime](02-agent-loop-and-runtime.md) | The loop, interceptors, termination, resume | §6 |
| 03 | [Model Port & Providers](03-model-port-and-providers.md) | LLM abstraction, adapters, routing, structured output | §14 |
| 04 | [Tools & MCP](04-tools-and-mcp.md) | Tool contract, executor, sandbox, registry, MCP, providers | §8 |
| 05 | [Planning & Reasoning](05-planning-and-reasoning.md) | ReAct, plan-execute, reflection, search | §7 |
| 06 | [Memory & State](06-memory-and-state.md) | Working/short/long-term memory, stores | §9 |
| 07 | [Retrieval (RAG)](07-retrieval-rag.md) | Ingestion, retrievers, hybrid/rerank/agentic/graph | §10 |
| 08 | [Multi-Agent Orchestration](08-multi-agent-orchestration.md) | Supervisor, handoff, crew, group-chat, A2A | §11 |
| 09 | [Context Engineering](09-context-engineering.md) | Assembly, compaction, editing, offloading | §12 |
| 10 | [Safety & Guardrails](10-safety-and-guardrails.md) | Input/output guards, policy, HITL, injection | §13 |
| 11 | [Observability & Evaluation](11-observability-and-evaluation.md) | Tracing (OTel), eval harness, replay | §16 |
| 12 | [Production & Reliability](12-production-and-reliability.md) | Durability, scale, cost, reliability, deploy | §17 |
| 13 | [Framework Interoperability](13-framework-interoperability.md) | Absorb/host adapters, concept mapping | §15 |
| 14 | [Skills](14-skills.md) | SKILL.md system, progressive disclosure, authoring, governance | §8 |

## The PoC, in one view

The foundational (Phase 0) slice across modules is a coherent, runnable agent:

> Kernel types ([01](01-domain-model-and-kernel.md)) → async ReAct loop with step cap ([02](02-agent-loop-and-runtime.md)) → one model adapter + `FakeModel` ([03](03-model-port-and-providers.md)) → function tools with sequential dispatch ([04](04-tools-and-mcp.md)) → ReAct strategy ([05](05-planning-and-reasoning.md)) → SQLite checkpointing + key-value memory ([06](06-memory-and-state.md)) → naive RAG ([07](07-retrieval-rag.md)) → agent-as-tool ([08](08-multi-agent-orchestration.md)) → budget-aware context packing ([09](09-context-engineering.md)) → tool allowlist + approval ([10](10-safety-and-guardrails.md)) → stdout/JSONL tracing + golden tests ([11](11-observability-and-evaluation.md)) → budget + loop guard ([12](12-production-and-reliability.md)) → MCP tool absorption ([13](13-framework-interoperability.md)) → wired into a runnable agent by `assemble()` over shell + MCP providers ([04](04-tools-and-mcp.md), [02](02-agent-loop-and-runtime.md)) with skills loaded on demand ([14](14-skills.md)).

Everything above Phase 0 is an additive layer you opt into.
