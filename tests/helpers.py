"""Shared test helpers."""

from __future__ import annotations

from openmate.adapters.stores.memory import InMemoryStore
from openmate.kernel.events import Event, EventBus
from openmate.kernel.types import Services
from openmate.ports.tracer import NullTracer


def make_services() -> tuple[Services, list[Event]]:
    """A deterministic ``Services`` (counter clock + ids) plus a captured event list."""
    bus = EventBus()
    events: list[Event] = []
    bus.subscribe(events.append)

    tick = {"n": 0}

    def clock() -> float:
        tick["n"] += 1
        return float(tick["n"])

    seq = {"n": 0}

    def new_id() -> str:
        seq["n"] += 1
        return f"id{seq['n']}"

    svc = Services(store=InMemoryStore(), tracer=NullTracer(), bus=bus, clock=clock, new_id=new_id)
    return svc, events
