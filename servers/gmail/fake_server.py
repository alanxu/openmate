"""A FAKE Gmail MCP server — seeded fixtures, no Google account, no OAuth.

It speaks the exact same tool surface and shaped responses as the real server
(``server.py``), so the email skills, the offline demo, and the MCP end-to-end
tests all run with zero credentials. Launch it over stdio:

    python servers/gmail/fake_server.py

Only depends on the ``mcp`` SDK (``pip install "openmate[mcp]"``) — deliberately
no ``openmate`` import, since an MCP server is a standalone process.
"""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

app = FastMCP("gmail-fake")

# --- seeded mailbox (already in the model-legible shape the real server emits) ---
_LABELS = [
    {"id": "INBOX", "name": "INBOX", "type": "system", "messages_total": 4},
    {"id": "UNREAD", "name": "UNREAD", "type": "system", "messages_total": 3},
    {"id": "IMPORTANT", "name": "IMPORTANT", "type": "system", "messages_total": 1},
    {"id": "Label_1", "name": "Receipts", "type": "user", "messages_total": 1},
]

_MESSAGES = [
    {
        "id": "m1",
        "threadId": "t1",
        "from": [{"name": "Ada Lovelace", "email": "ada@example.com"}],
        "to": [{"name": "Alan", "email": "alan@example.com"}],
        "subject": "Re: agent loop review",
        "date": "2026-06-24T14:21:00-04:00",
        "labels": ["INBOX", "UNREAD", "IMPORTANT"],
        "snippet": "Looks good — let's ship it on Tuesday. One question about the stop policy…",
        "body": (
            "Looks good — let's ship it on Tuesday. One question about the stop "
            "policy: does NoProgress fire before or after the checkpoint?\n\n"
            "> Are we ready to merge the loop refactor?\n> — Alan"
        ),
        "attachments": [],
    },
    {
        "id": "m2",
        "threadId": "t2",
        "from": [{"name": "GitHub", "email": "notifications@github.com"}],
        "to": [{"name": "Alan", "email": "alan@example.com"}],
        "subject": "[openmate] CI passed on main",
        "date": "2026-06-24T13:05:00-04:00",
        "labels": ["INBOX", "UNREAD"],
        "snippet": "All checks have passed for commit b90cc0d.",
        "body": "All checks have passed for commit b90cc0d on branch main.",
        "attachments": [],
    },
    {
        "id": "m3",
        "threadId": "t3",
        "from": [{"name": "Acme Billing", "email": "billing@acme.example"}],
        "to": [{"name": "Alan", "email": "alan@example.com"}],
        "subject": "Your June invoice is ready",
        "date": "2026-06-23T09:00:00-04:00",
        "labels": ["INBOX", "UNREAD", "Label_1"],
        "snippet": "Invoice #4471 for $42.00 is attached.",
        "body": "Invoice #4471 for $42.00 is attached. Payment due July 1.",
        "attachments": [
            {"id": "att_1", "filename": "invoice-4471.pdf", "artifact": "art_9c1a"}
        ],
    },
    {
        "id": "m4",
        "threadId": "t1",
        "from": [{"name": "Alan", "email": "alan@example.com"}],
        "to": [{"name": "Ada Lovelace", "email": "ada@example.com"}],
        "subject": "Re: agent loop review",
        "date": "2026-06-22T18:40:00-04:00",
        "labels": ["SENT"],
        "snippet": "Are we ready to merge the loop refactor?",
        "body": "Are we ready to merge the loop refactor?\n— Alan",
        "attachments": [],
    },
]

_BY_ID = {m["id"]: m for m in _MESSAGES}


def _summary(m: dict) -> dict:
    """The compact item shape returned by search (no full body)."""
    return {
        "id": m["id"],
        "threadId": m["threadId"],
        "from": m["from"],
        "subject": m["subject"],
        "date": m["date"],
        "snippet": m["snippet"],
        "labels": m["labels"],
    }


def _matches(m: dict, q: str) -> bool:
    q = (q or "").strip().lower()
    if not q:
        return "INBOX" in m["labels"]
    ok = True
    for token in q.split():
        if token == "is:unread":
            ok = ok and "UNREAD" in m["labels"]
        elif token.startswith("label:"):
            ok = ok and token.split(":", 1)[1].upper() in {l.upper() for l in m["labels"]}
        elif token.startswith("from:"):
            needle = token.split(":", 1)[1]
            ok = ok and any(needle in a["email"].lower() for a in m["from"])
        else:
            hay = (m["subject"] + " " + m["snippet"] + " " + m["body"]).lower()
            ok = ok and token in hay
    return ok


_READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
_DRAFT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


@app.tool(
    name="gmail_search",
    description=(
        "Search the mailbox. Returns {items, next_page_token}; pass page_token to "
        "continue. Supports Gmail operators like is:unread, label:NAME, from:addr. "
        "Read-only and safe to retry."
    ),
    annotations=_READ,
)
def gmail_search(q: str = "", max_results: int = 10, page_token: str | None = None) -> dict:
    hits = [m for m in _MESSAGES if _matches(m, q)]
    hits.sort(key=lambda m: m["date"], reverse=True)
    start = int(page_token) if page_token else 0
    page = hits[start : start + max_results]
    nxt = start + max_results
    return {
        "items": [_summary(m) for m in page],
        "next_page_token": str(nxt) if nxt < len(hits) else None,
    }


@app.tool(
    name="gmail_get_message",
    description="Fetch one message by id: decoded headers + plaintext body. Read-only.",
    annotations=_READ,
)
def gmail_get_message(id: str, format: str = "full") -> dict:
    m = _BY_ID.get(id)
    if m is None:
        return {"error": f"no message with id '{id}'"}
    if format == "metadata":
        return _summary(m)
    return m


@app.tool(
    name="gmail_get_thread",
    description="Fetch a whole thread by id: its messages in chronological order. Read-only.",
    annotations=_READ,
)
def gmail_get_thread(id: str) -> dict:
    msgs = sorted(
        (m for m in _MESSAGES if m["threadId"] == id), key=lambda m: m["date"]
    )
    if not msgs:
        return {"error": f"no thread with id '{id}'"}
    return {"id": id, "messages": msgs}


@app.tool(
    name="gmail_list_labels",
    description="List mailbox labels with message counts. Read-only.",
    annotations=_READ,
)
def gmail_list_labels() -> dict:
    return {"labels": _LABELS}


@app.tool(
    name="gmail_create_draft",
    description=(
        "Create a DRAFT reply or new email (never sends). Returns the draft id. "
        "Use this to propose a reply for the user to review and send themselves."
    ),
    annotations=_DRAFT,
)
def gmail_create_draft(
    to: list[str], subject: str, body: str, thread_id: str | None = None
) -> dict:
    # The fake doesn't persist; it echoes a plausible draft id so flows complete.
    return {
        "id": "draft_fake_1",
        "message_id": "m_draft_1",
        "to": to,
        "subject": subject,
        "thread_id": thread_id,
    }


if __name__ == "__main__":
    app.run()
