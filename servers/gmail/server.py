"""A real Gmail MCP server — read-focused, OAuth-scoped (docs/04 worked example).

Exposes Gmail as MCP **tools** (acting) and **resources** (browsing). The hard,
unsexy half lives here: OAuth refresh, MIME decoding, header normalization, body
plaintexting, truncation, pagination. The model never sees raw MIME or tokens.

Tools (read-focused; send is intentionally absent):

    gmail_search        read   q, max_results, page_token -> {items, next_page_token}
    gmail_get_message   read   id, format                 -> shaped message
    gmail_get_thread    read   id                          -> ordered messages
    gmail_list_labels   read   -> labels + counts
    gmail_create_draft  write  to, subject, body, thread_id? -> {id} (NEVER sends)

Run it over stdio (this is what an OpenMate ``MCPServerSpec`` launches):

    GOOGLE_CLIENT_ID=… GOOGLE_CLIENT_SECRET=… \
    TOKEN_PATH=~/.config/openmate/gmail/token.json \
    python servers/gmail/server.py

Auth: Google OAuth 2.0 with a per-user refresh token. The server is the OAuth
client, not the agent. ``TOKEN_PATH`` points at an authorized-user token.json
(0600); see README.md for the one-time bootstrap. Tokens live on disk, never in
model context — a leaked token in a tool result is a security incident.

Dependencies (separate from the core package — see requirements.txt):
    mcp  google-api-python-client  google-auth  google-auth-oauthlib
"""

from __future__ import annotations

import base64
import os
from datetime import datetime
from email.utils import parsedate_to_datetime

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

# ----------------------------------------------------------------------------
# OAuth scopes: read everything, plus compose (drafts). NOT gmail.send — sending
# is destructive and non-idempotent; grant it explicitly elsewhere if ever needed.
SCOPES = [
    "https://www.googleapis.com/auth/gmail.readonly",
    "https://www.googleapis.com/auth/gmail.compose",
]
TOKEN_PATH = os.path.expanduser(
    os.environ.get("TOKEN_PATH", "~/.config/openmate/gmail/token.json")
)
MAX_BODY_CHARS = 8000

app = FastMCP("gmail")
_service = None  # lazily built googleapiclient resource


def _gmail():
    """Build (once) the authenticated Gmail client; refresh the token on demand."""
    global _service
    if _service is not None:
        return _service
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as e:  # pragma: no cover - exercised only in a real deploy
        raise RuntimeError(
            "Gmail server needs google libraries — `pip install -r "
            "servers/gmail/requirements.txt`"
        ) from e

    if not os.path.exists(TOKEN_PATH):
        raise RuntimeError(
            f"no OAuth token at {TOKEN_PATH}. Run the bootstrap in servers/gmail/README.md."
        )
    creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)
    if not creds.valid and creds.refresh_token:
        creds.refresh(Request())  # refresh on expiry; google-auth coalesces internally
        with open(TOKEN_PATH, "w", encoding="utf-8") as f:
            f.write(creds.to_json())
    # cache_discovery=False: the discovery doc otherwise leaks ~200KB per restart.
    _service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    return _service


# --- result shaping: make Gmail model-legible -------------------------------
def _decode_b64url(data: str) -> bytes:
    return base64.urlsafe_b64decode(data + "=" * (-len(data) % 4))


def _split_address(raw: str) -> list[dict]:
    """'Ada Lovelace <ada@example.com>, b@x.com' -> [{name, email}, ...]."""
    from email.utils import getaddresses

    return [{"name": name, "email": email} for name, email in getaddresses([raw or ""]) if email]


def _iso_date(raw: str) -> str:
    try:
        dt = parsedate_to_datetime(raw)
        return dt.isoformat() if dt else raw
    except (TypeError, ValueError):
        return raw


def _html_to_text(html: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", "", html)
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    from html import unescape

    return re.sub(r"\n{3,}", "\n\n", unescape(text)).strip()


def _extract_body(payload: dict) -> tuple[str, list[dict]]:
    """Walk the MIME tree: prefer text/plain, fall back to text/html; collect attachments."""
    plain, html, attachments = "", "", []

    def walk(part: dict) -> None:
        nonlocal plain, html
        mime = part.get("mimeType", "")
        body = part.get("body", {})
        filename = part.get("filename")
        if filename:
            attachments.append(
                {
                    "id": body.get("attachmentId"),
                    "filename": filename,
                    "mime": mime,
                    # large blobs become Artifacts referenced by id (docs/01); the
                    # model never reads raw bytes inline.
                    "artifact": f"art_{(body.get('attachmentId') or filename)[:8]}",
                }
            )
        elif mime == "text/plain" and body.get("data"):
            plain += _decode_b64url(body["data"]).decode("utf-8", "replace")
        elif mime == "text/html" and body.get("data"):
            html += _decode_b64url(body["data"]).decode("utf-8", "replace")
        for sub in part.get("parts", []) or []:
            walk(sub)

    walk(payload)
    text = plain.strip() or _html_to_text(html)
    if len(text) > MAX_BODY_CHARS:
        text = text[:MAX_BODY_CHARS] + "\n… (truncated; call gmail_get_message with format=raw)"
    return text, attachments


def _shape_message(msg: dict, *, full: bool = True) -> dict:
    headers = {h["name"].lower(): h["value"] for h in msg.get("payload", {}).get("headers", [])}
    shaped = {
        "id": msg.get("id"),
        "threadId": msg.get("threadId"),
        "from": _split_address(headers.get("from", "")),
        "to": _split_address(headers.get("to", "")),
        "subject": headers.get("subject", ""),
        "date": _iso_date(headers.get("date", "")),
        "labels": msg.get("labelIds", []),
        "snippet": msg.get("snippet", ""),
    }
    if full:
        body, attachments = _extract_body(msg.get("payload", {}))
        shaped["body"] = body
        shaped["attachments"] = attachments
    return shaped


# --- tools -------------------------------------------------------------------
_READ = ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=True)
_DRAFT = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True)


@app.tool(
    name="gmail_search",
    description=(
        "Search the mailbox. Returns {items, next_page_token}; pass page_token to "
        "continue. Supports Gmail operators (is:unread, label:NAME, from:addr, "
        "newer_than:7d). Read-only and safe to retry."
    ),
    annotations=_READ,
)
def gmail_search(q: str = "", max_results: int = 10, page_token: str | None = None) -> dict:
    gmail = _gmail()
    resp = (
        gmail.users()
        .messages()
        .list(userId="me", q=q, maxResults=max_results, pageToken=page_token)
        .execute()
    )
    items = []
    for ref in resp.get("messages", []):
        msg = (
            gmail.users()
            .messages()
            .get(userId="me", id=ref["id"], format="metadata")
            .execute()
        )
        items.append(_shape_message(msg, full=False))
    return {"items": items, "next_page_token": resp.get("nextPageToken")}


@app.tool(
    name="gmail_get_message",
    description="Fetch one message by id: decoded headers + plaintext body. Read-only.",
    annotations=_READ,
)
def gmail_get_message(id: str, format: str = "full") -> dict:
    gmail = _gmail()
    msg = gmail.users().messages().get(userId="me", id=id, format=format).execute()
    return _shape_message(msg, full=format != "metadata")


@app.tool(
    name="gmail_get_thread",
    description="Fetch a whole thread by id: its messages in chronological order. Read-only.",
    annotations=_READ,
)
def gmail_get_thread(id: str) -> dict:
    gmail = _gmail()
    thread = gmail.users().threads().get(userId="me", id=id, format="full").execute()
    msgs = [_shape_message(m) for m in thread.get("messages", [])]
    msgs.sort(key=lambda m: m.get("date", ""))
    return {"id": id, "messages": msgs}


@app.tool(
    name="gmail_list_labels",
    description="List mailbox labels with message counts. Read-only.",
    annotations=_READ,
)
def gmail_list_labels() -> dict:
    gmail = _gmail()
    labels = gmail.users().labels().list(userId="me").execute().get("labels", [])
    out = []
    for label in labels:
        detail = gmail.users().labels().get(userId="me", id=label["id"]).execute()
        out.append(
            {
                "id": label["id"],
                "name": label["name"],
                "type": label.get("type"),
                "messages_total": detail.get("messagesTotal"),
            }
        )
    return {"labels": out}


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
    from email.mime.text import MIMEText

    gmail = _gmail()
    mime = MIMEText(body)
    mime["To"] = ", ".join(to)
    mime["Subject"] = subject
    raw = base64.urlsafe_b64encode(mime.as_bytes()).decode()
    message: dict = {"raw": raw}
    if thread_id:
        message["threadId"] = thread_id
    draft = (
        gmail.users()
        .drafts()
        .create(userId="me", body={"message": message})
        .execute()
    )
    return {"id": draft["id"], "message_id": draft.get("message", {}).get("id")}


# --- resources (browsing) ----------------------------------------------------
@app.resource("gmail://message/{message_id}")
def message_resource(message_id: str) -> str:
    msg = gmail_get_message(message_id)
    return f"From: {msg['from']}\nSubject: {msg['subject']}\nDate: {msg['date']}\n\n{msg['body']}"


@app.resource("gmail://thread/{thread_id}")
def thread_resource(thread_id: str) -> str:
    thread = gmail_get_thread(thread_id)
    blocks = [f"--- {m['date']} · {m['from']}\n{m['body']}" for m in thread["messages"]]
    return f"Thread {thread_id}\n\n" + "\n\n".join(blocks)


if __name__ == "__main__":
    app.run()
