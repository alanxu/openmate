"""Pytest configuration — gates the live (real-model) tests.

All live tests live under ``evals/`` (the metric eval harness plus the live
smoke/integration tests); ``tests/`` is 100% offline. Live tests call a real
model over the network (cost, latency, non-determinism), so they are **skipped
by default**. Enable the live tier explicitly:

    pytest --run-live evals/ -v

A normal ``pytest`` run stays fully offline even when an API key is present in
``.env`` (the live tests are collected but skipped). Live tests additionally
self-skip if no key is configured.
"""

from __future__ import annotations

import os

import pytest


def pytest_addoption(parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="run live tests that call a real model (requires an API key)",
    )
    parser.addoption(
        "--eval-verbose",
        action="store_true",
        default=False,
        help="print a live event / agent-IO trace during eval runs (use with -s)",
    )
    parser.addoption(
        "--log",
        action="store_true",
        default=False,
        help="log the full event stream (raw model I/O) to ~/.openmate/logs",
    )


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "live: test that calls a real model over the network (opt-in)"
    )
    # Bridge flags to the harness's make_services() (a plain function, not a
    # fixture, so it can't read pytest config directly).
    if config.getoption("--eval-verbose"):
        os.environ["OPENMATE_EVAL_VERBOSE"] = "1"
    if config.getoption("--log"):
        os.environ["OPENMATE_LOG"] = "1"


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-live"):
        return  # opted in — let them run
    skip_live = pytest.mark.skip(reason="live test — pass --run-live")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def live_model():
    """The configured real model (MiniMax via the Anthropic-compatible API).

    Skips the test if no API key is set, so live runs degrade gracefully.
    """
    from openmate.config import default_model, load_env

    load_env()
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")):
        pytest.skip("no API key — set ANTHROPIC_API_KEY (or MINIMAX_API_KEY) in .env")
    return default_model()
