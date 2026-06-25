"""Pytest configuration — gates the live (real-model) tests.

The tests in ``test_live.py`` call a real model over the network (cost, latency,
non-determinism), so they are **skipped by default**. Enable them explicitly:

    pytest --run-live tests/test_live.py -v
    OPENMATE_LIVE_TESTS=1 pytest tests/test_live.py -v

A normal ``pytest`` run stays fully offline even when an API key is present in
``.env``. Live tests additionally self-skip if no key is configured.
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


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers", "live: test that calls a real model over the network (opt-in)"
    )


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-live") or os.environ.get("OPENMATE_LIVE_TESTS"):
        return  # opted in — let them run
    skip_live = pytest.mark.skip(
        reason="live test — pass --run-live or set OPENMATE_LIVE_TESTS=1"
    )
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
