"""Default wiring: load configuration from the environment and build a ready-to-run
model + services bundle.

Out of the box this targets a MiniMax model through its Anthropic-compatible
endpoint, so OpenMate runs as a Claude-style agent on a MiniMax key. Every value
is overridable by an environment variable (see ``.env.example``).
"""

from __future__ import annotations

import os
import random
import time
from uuid import uuid4

from .adapters.models.anthropic import AnthropicModel
from .adapters.stores.memory import InMemoryStore
from .adapters.tracers.console import ConsoleTracer
from .kernel.events import EventBus
from .kernel.types import Services

# MiniMax's Anthropic-compatible API. Override with ANTHROPIC_BASE_URL / OPENMATE_MODEL.
DEFAULT_BASE_URL = "https://api.minimax.io/anthropic"
DEFAULT_MODEL = "MiniMax-M2"


def load_env() -> None:
    """Load the local ``.env`` file if python-dotenv is installed (best-effort).

    ``override=True`` makes the project ``.env`` authoritative for local runs, so
    a pre-existing ``ANTHROPIC_BASE_URL``/``ANTHROPIC_API_KEY`` in the surrounding
    shell (e.g. one pointing at the real Anthropic API) does not shadow it.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(override=True)


def default_model(model: str | None = None) -> AnthropicModel:
    """Build the Claude/Anthropic adapter pointed at MiniMax (or whatever the env says)."""
    load_env()
    base_url = os.environ.get("ANTHROPIC_BASE_URL", DEFAULT_BASE_URL)
    name = model or os.environ.get("OPENMATE_MODEL", DEFAULT_MODEL)
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("MINIMAX_API_KEY")
    return AnthropicModel(name, api_key=api_key, base_url=base_url)


def default_services(*, store=None, tracer=None, verbose: bool = False) -> Services:
    """Assemble a ``Services`` bundle with a console tracer wired to the event bus."""
    from .adapters.tracers.jsonl import attach_if_enabled

    bus = EventBus()
    tracer = tracer if tracer is not None else ConsoleTracer(verbose=verbose)
    if isinstance(tracer, ConsoleTracer):
        tracer.attach(bus)
    attach_if_enabled(bus)  # OPENMATE_LOG → file logging, no other wiring needed
    return Services(
        store=store if store is not None else InMemoryStore(),
        tracer=tracer,
        bus=bus,
        clock=time.time,
        rng=random.Random(),
        new_id=lambda: uuid4().hex,
    )
