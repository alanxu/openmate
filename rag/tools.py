"""Retrieval as a tool — the bridge from RAG to the agent loop (docs/07 §Phase 2).

``RetrieveTool`` wraps a :class:`Retriever` as an OpenMate ``Tool``. One retrieval
call is plain RAG; an agent that calls it repeatedly — judging results, refining
the query — *is* agentic RAG ("just an OpenMate agent whose tools are retrievers;
no special engine"). ``RagProvider`` lets ``assemble()`` mount it like any other
provider.
"""

from __future__ import annotations

from openmate.kernel.types import TextPart
from openmate.ports.retriever import Document, Retriever
from openmate.ports.tool import ToolResult, ToolSpec


def format_documents(docs: list[Document]) -> str:
    if not docs:
        return "No matching documents found in the knowledge base."
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(f"[{i}] score={d.score:.3f} · source={d.source}\n{d.text}")
    return "\n\n".join(blocks)


class RetrieveTool:
    """``rag_search(query, k?)`` — retrieve the top-k chunks for a query."""

    def __init__(self, retriever: Retriever, *, k: int = 5, name: str = "rag_search") -> None:
        self.retriever = retriever
        self.k = k
        self.spec = ToolSpec(
            name=name,
            description=(
                "Search the knowledge base for passages relevant to a query and return "
                "the top matches with their sources and scores. Read-only; call it "
                "multiple times with refined queries if the first results are weak."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "what to search for"},
                    "k": {"type": "integer", "description": "how many passages to return"},
                },
                "required": ["query"],
            },
            side_effecting=False,
            idempotent=True,
        )

    async def invoke(self, args: dict, ctx) -> ToolResult:
        query = (args or {}).get("query")
        if not query:
            return ToolResult([TextPart("missing required argument: query")], is_error=True)
        k = int((args or {}).get("k") or self.k)
        docs = await self.retriever.retrieve(query, k=k)
        # Record what was retrieved so an agentic run can report its sources.
        if ctx is not None and getattr(ctx, "state", None) is not None:
            try:
                hits = ctx.state.scratch.setdefault("rag_sources", [])
                hits.extend({"id": d.id, "source": d.source, "score": d.score} for d in docs)
            except Exception:  # noqa: BLE001 — source tracking is best-effort
                pass
        return ToolResult([TextPart(format_documents(docs))])


class RagProvider:
    """A ``ToolProvider`` that contributes the retrieve tool (for ``assemble()``)."""

    name = "rag"

    def __init__(self, retriever: Retriever, *, k: int = 5) -> None:
        self._tool = RetrieveTool(retriever, k=k)

    async def setup(self) -> None:
        return None

    async def tools(self) -> list:
        return [self._tool]

    def system_fragment(self) -> str | None:
        return (
            "## Knowledge base\n"
            "Use `rag_search` to ground answers in the indexed knowledge base. Prefer "
            "retrieved evidence over prior knowledge, and cite sources as [n]."
        )

    async def teardown(self) -> None:
        return None
