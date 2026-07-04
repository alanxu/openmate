"""Built-in tools: ``default_dir`` makes relative paths and ``run_shell`` (with no
explicit cwd) default to a chosen working directory — e.g. an attached project —
instead of the process cwd. Guards the "attaching a project doesn't set the work
dir" regression (ui/server.py:build_agent passes the project dir as default_dir)."""

from __future__ import annotations

import asyncio
import os

import pytest

from openmate.adapters.tools.builtin import (
    _make_safe_path,
    make_file_tools,
    make_shell_tool,
)


def test_default_dir_relative_paths_resolve_against_base(tmp_path):
    sp = _make_safe_path(base=str(tmp_path))
    base = tmp_path.resolve()
    assert sp(".") == base                       # "." is the project dir, not cwd
    assert sp("a.txt") == base / "a.txt"
    (tmp_path / "sub").mkdir()
    assert sp("sub") == base / "sub"


def test_paths_outside_base_and_cwd_are_rejected(tmp_path):
    sp = _make_safe_path(base=str(tmp_path))
    with pytest.raises(ValueError):
        sp("/etc/passwd")


def test_default_dir_none_falls_back_to_cwd():
    sp = _make_safe_path()
    assert str(sp(".")) == os.getcwd()           # unchanged behavior when no base given


def test_shell_and_file_tools_default_to_project_dir(tmp_path):
    (tmp_path / "note.txt").write_text("in project dir")
    shell = make_shell_tool(default_dir=str(tmp_path))
    out = asyncio.run(shell.invoke({"command": "pwd"}, None))
    assert str(tmp_path.resolve()) in out.content[0].text

    rf, _wf, _ld = make_file_tools(default_dir=str(tmp_path))
    r = asyncio.run(rf.invoke({"path": "note.txt"}, None))   # relative → resolves in project dir
    assert "in project dir" in r.content[0].text
