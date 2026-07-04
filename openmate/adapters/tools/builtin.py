"""A small set of genuinely useful built-in tools.

Most work entirely offline (math, time, filesystem); ``fetch_url`` reaches the
web using only the standard library. File paths are confined to the current
working directory tree as a basic safety boundary.
"""

from __future__ import annotations

import ast
import operator
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from .native import tool

_MAX_FILE_BYTES = 100_000
_MAX_FETCH_BYTES = 20_000
_MAX_SHELL_OUTPUT = 20_000
_SHELL_TIMEOUT_S = 30

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


def _make_safe_path(extra_roots: "list[str] | None" = None, base: "str | Path | None" = None):
    """Build a path-confinement check rooted at cwd (or ``base``) plus any ``extra_roots``.

    A closure, not a module-level singleton, so callers (e.g. the UI, scoping
    write access per thread/"folder added for editing") can confine a *set* of
    file tools to a caller-chosen allowlist without touching the Tool port or
    the default cwd-only behavior every other caller still gets. Relative paths
    resolve against ``base`` (the task's working directory) when given, else the
    process cwd — so an attached project becomes the *default* directory, not
    merely an allowed one.
    """
    base_dir = Path(base).expanduser().resolve() if base else Path.cwd().resolve()
    roots: list[Path] = []
    for r in [base_dir, Path.cwd().resolve(), *(Path(x).expanduser().resolve() for x in (extra_roots or []))]:
        if r not in roots:
            roots.append(r)

    def safe_path(path: str) -> Path:
        raw = Path(path).expanduser()
        p = raw.resolve() if raw.is_absolute() else (base_dir / raw).resolve()
        for root in roots:
            if root in p.parents or p == root:
                return p
        allowed = ", ".join(str(r) for r in roots)
        raise ValueError(f"path '{path}' is outside the allowed directories ({allowed})")

    return safe_path


def make_file_tools(extra_roots: "list[str] | None" = None, *, default_dir: "str | Path | None" = None) -> list:
    """``read_file``/``write_file``/``list_directory``, confined to cwd + ``extra_roots``.

    The module-level ``read_file``/``write_file``/``list_directory`` below are
    just this factory called with no extra roots — kept as importable names for
    backward compatibility (existing callers, tests) that reference them directly.
    ``default_dir`` sets the base that relative paths resolve against (the task's
    working directory); it defaults to the process cwd.
    """
    safe_path = _make_safe_path(extra_roots, base=default_dir)

    @tool(side_effecting=False, idempotent=True)
    def read_file(path: str) -> str:
        """Read a UTF-8 text file from an allowed directory and return its contents."""
        p = safe_path(path)
        data = p.read_text(encoding="utf-8", errors="replace")
        if len(data) > _MAX_FILE_BYTES:
            return data[:_MAX_FILE_BYTES] + f"\n...[truncated, {len(data)} bytes total]"
        return data

    @tool(side_effecting=True)
    def write_file(path: str, content: str) -> str:
        """Write text to a file within an allowed directory (creates or overwrites it)."""
        p = safe_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"wrote {len(content)} bytes to {p}"

    @tool(side_effecting=False, idempotent=True)
    def list_directory(path: str = ".") -> str:
        """List the entries in a directory within an allowed directory tree."""
        p = safe_path(path)
        if not p.is_dir():
            return f"not a directory: {path}"
        entries = sorted(os.listdir(p))
        if not entries:
            return "(empty directory)"
        return "\n".join(("📁 " if (p / e).is_dir() else "📄 ") + e for e in entries)

    return [read_file, write_file, list_directory]


_read_file, _write_file, _list_directory = make_file_tools()
read_file, write_file, list_directory = _read_file, _write_file, _list_directory


def make_shell_tool(extra_roots: "list[str] | None" = None, *, default_dir: "str | Path | None" = None):
    """A ``run_shell`` tool whose *working directory* is confined to cwd + ``extra_roots``.

    ``default_dir`` sets where a command with no explicit ``cwd`` runs (the task's
    working directory); it defaults to the process cwd.

    Caveat: confinement only covers ``cwd`` — the shell itself is not sandboxed,
    so a command can still ``cd`` or use absolute paths to act outside the
    allowlist (e.g. ``cat /etc/passwd``). This is the same trust boundary as
    giving a human a terminal scoped to a starting directory, not a jail. Good
    enough for a local single-user assistant; do not expose this to untrusted
    multi-tenant input without a real sandbox (container, VM, seccomp, etc.).
    """
    safe_path = _make_safe_path(extra_roots, base=default_dir)

    @tool(side_effecting=True)
    def run_shell(command: str, cwd: str = ".") -> str:
        """Run a shell command. ``cwd`` must be the server's working directory or
        one of the folders added for this thread; defaults to that working
        directory. 30s timeout, output truncated. Use for file edits, git,
        running scripts/tests, etc."""
        work_dir = safe_path(cwd)
        try:
            proc = subprocess.run(
                command,
                shell=True,  # noqa: S602 — intentional: this *is* a shell-command tool
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                timeout=_SHELL_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            return f"error: command timed out after {_SHELL_TIMEOUT_S}s"
        out = proc.stdout or ""
        if proc.stderr:
            out += f"\n[stderr]\n{proc.stderr}"
        out = out.strip() or "(no output)"
        if len(out) > _MAX_SHELL_OUTPUT:
            out = out[:_MAX_SHELL_OUTPUT] + f"\n...[truncated, {len(out)} bytes total]"
        return f"exit={proc.returncode}\n{out}"

    return run_shell


_run_shell = make_shell_tool()
run_shell = _run_shell


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


def read_only_tools(extra_roots: "list[str] | None" = None, *, default_dir: "str | Path | None" = None) -> list:
    """The safe, read-only default toolset, optionally confined to extra directories too.

    No ``calculator`` here on purpose — it's still defined above (and used directly
    by the test suite/examples as a simple deterministic fixture tool), but the
    actual agent toolsets lean on ``run_shell``/a model's own arithmetic instead of
    a second, narrower tool that does the same thing.
    """
    if extra_roots or default_dir:
        rf, _wf, ld = make_file_tools(extra_roots, default_dir=default_dir)
    else:
        rf, ld = read_file, list_directory
    return [current_time, rf, ld, fetch_url]


def all_tools(extra_roots: "list[str] | None" = None, *, default_dir: "str | Path | None" = None) -> list:
    """Every built-in tool, including the side-effecting ``write_file``/``run_shell``.

    No dedicated mkdir or calculator tool on purpose — ``run_shell`` is the
    general-purpose escape hatch for both, and a model that already knows
    ``mkdir -p`` or basic arithmetic doesn't need a second, narrower tool
    teaching it the same thing. ``default_dir`` makes relative paths and shell
    commands default to the task's working directory (e.g. an attached project).
    """
    if extra_roots or default_dir:
        rf, wf, ld = make_file_tools(extra_roots, default_dir=default_dir)
        shell = make_shell_tool(extra_roots, default_dir=default_dir)
    else:
        rf, wf, ld = read_file, write_file, list_directory
        shell = run_shell
    return [current_time, rf, ld, fetch_url, wf, shell]
