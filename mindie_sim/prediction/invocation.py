"""One device invocation boundary: declared startup and completion floor.

[Role] Constrain one already-prepared finite body program at a device
  invocation boundary.
[Invariants] Startup is a causal delay after original readiness before body
  expansion. Minimum is anchored after that same readiness and runs concurrently
  with startup+body, so duration is max(minimum, startup+body), never
  startup+max(minimum,body). Body work, caps, buffers and queue resources are
  unchanged; the body starts once and owner is forwarded. Provenance is carried
  as data, not proof of measured identity. Missing parameters are zero and are
  never inferred from observed grouped residuals.
[Timeouts & Locks] Finite engine events only; no threads, sleeps, I/O or locks.
  One startup, one floor and one completion observer per invocation.
[Side-Effect Policy] Submits tasks only to the caller's engine; no global state,
  barrier or mutable ledger.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.invocation import InvocationTiming

from .engine import Engine, Event
from .program import Program


@dataclass(frozen=True)
class InvocationProgram:
    """Start one prepared body behind declared invocation timing."""

    body: Program
    timing: InvocationTiming

    def __post_init__(self):
        if not isinstance(self.timing, InvocationTiming):
            raise TypeError("invocation_program_requires_invocation_timing")

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        startup_ms = self.timing.startup_ms
        minimum_ms = self.timing.minimum_ms
        if startup_ms == 0 and ready.done:
            body = self.body.start(engine, name + ":body", ready, owner=owner)
            if minimum_ms == 0:
                return body
            floor = engine.submit(
                name + ":minimum", after=(ready,), latency=minimum_ms, owner=owner,
            )
            return engine.submit(name + ":complete", after=(body, floor), owner=owner)

        completion = engine.future(name + ":complete")
        floor = (
            engine.submit(name + ":minimum", after=(ready,), latency=minimum_ms, owner=owner)
            if minimum_ms else None
        )

        def launch(startup: Event) -> None:
            # Expand the body only after startup has completed. This keeps idle
            # invocation state bounded and preserves the body's input readiness.
            body = self.body.start(engine, name + ":body", startup, owner=owner)
            dependencies = (body,) if floor is None else (body, floor)
            engine.submit(
                name + ":body_complete", after=dependencies, owner=owner,
                on_complete=lambda _: engine.resolve(completion),
            )

        engine.submit(
            name + ":startup", after=(ready,), latency=startup_ms,
            on_complete=launch, owner=owner,
        )
        return completion
