"""A small set of genuinely useful built-in tools.

Most work entirely offline (math, time, filesystem); ``fetch_url`` reaches the
web using only the standard library. File paths are confined to the current
working directory tree as a basic safety boundary.
"""

from __future__ import annotations

import ast
import operator
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .native import tool

_MAX_FILE_BYTES = 100_000
_MAX_FETCH_BYTES = 20_000

# --- calculator: a safe arithmetic evaluator (no eval(), no names) -------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("unsupported expression")


@tool(side_effecting=False, idempotent=True)
def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression (+, -, *, /, //, %, **, parentheses)."""
    tree = ast.parse(expression, mode="eval")
    return str(_eval_node(tree.body))


@tool(side_effecting=False, idempotent=True)
def current_time() -> str:
    """Return the current UTC date and time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_path(path: str) -> Path:
    p = Path(path).expanduser().resolve()
    root = Path.cwd().resolve()
    if root not in p.parents and p != root:
        raise ValueError(f"path '{path}' is outside the working directory")
    return p


@tool(side_effecting=False, idempotent=True)
def read_file(path: str) -> str:
    """Read a UTF-8 text file from within the working directory and return its contents."""
    p = _safe_path(path)
    data = p.read_text(encoding="utf-8", errors="replace")
    if len(data) > _MAX_FILE_BYTES:
        return data[:_MAX_FILE_BYTES] + f"\n...[truncated, {len(data)} bytes total]"
    return data


@tool(side_effecting=True)
def write_file(path: str, content: str) -> str:
    """Write text to a file within the working directory (creates or overwrites it)."""
    p = _safe_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"wrote {len(content)} bytes to {p}"


@tool(side_effecting=False, idempotent=True)
def list_directory(path: str = ".") -> str:
    """List the entries in a directory within the working directory tree."""
    p = _safe_path(path)
    if not p.is_dir():
        return f"not a directory: {path}"
    entries = sorted(os.listdir(p))
    if not entries:
        return "(empty directory)"
    return "\n".join(("📁 " if (p / e).is_dir() else "📄 ") + e for e in entries)


@tool(side_effecting=False, idempotent=True)
def fetch_url(url: str) -> str:
    """Fetch the text content of an http(s) URL (truncated). Use for quick web lookups."""
    if not url.startswith(("http://", "https://")):
        return "error: url must start with http:// or https://"
    req = Request(url, headers={"User-Agent": "OpenMate/0.1"})
    with urlopen(req, timeout=15) as resp:  # noqa: S310 — scheme is validated above
        raw = resp.read(_MAX_FETCH_BYTES + 1)
    text = raw.decode("utf-8", errors="replace")
    if len(raw) > _MAX_FETCH_BYTES:
        text = text[:_MAX_FETCH_BYTES] + "\n...[truncated]"
    return text


def read_only_tools() -> list:
    """The safe, read-only default toolset."""
    return [calculator, current_time, read_file, list_directory, fetch_url]


def all_tools() -> list:
    """Every built-in tool, including the side-effecting ``write_file``."""
    return [*read_only_tools(), write_file]
