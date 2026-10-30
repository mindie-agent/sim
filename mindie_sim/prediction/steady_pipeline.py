"""Compact steady two-stage pipeline with an exact finite tail.

[Role] Preserve finite producer and consumer work while executing repeated
  identical full items in a constant number of causal Engine tasks.
[Invariants] Full work is N copies of the two declared stages. A two-slot tail
  is one exact nonuniform item. Producer and consumer resource domains are disjoint,
  so aggregation cannot merge contender counts or placements. The steady task
  is coupled: all of its declared resource demands advance with synchronized
  fractional progress instead of letting the fast pipe run ahead.
[Timeouts & Locks] Constant task/event state independent of count; no I/O,
  threads or sleeps.
[Side-Effect Policy] Validation is pure. ``start`` submits ordinary tasks and
  allocates only the caller's explicit buffers on the caller's Engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from .engine import Allocation, Engine, Event
from .program import Stage, scope_allocations


def _resources(stage: Stage) -> set[str]:
    """Every physical resource named by a stage declaration."""
    return set(stage.work) | set(stage.caps) | set(stage.placement)


def _validate_stream(stage: Stage, label: str) -> None:
    if not isinstance(stage, Stage):
        raise ValueError(f"steady_pipeline_{label}_requires_stage")
    if stage.after:
        raise ValueError(f"steady_pipeline_{label}_after_not_supported")
    if stage.previous:
        raise ValueError(f"steady_pipeline_{label}_previous_not_supported")
    if stage.latency != 0:
        raise ValueError(f"steady_pipeline_{label}_latency_not_supported")
    if stage.program is not None:
        raise ValueError(f"steady_pipeline_{label}_program_not_supported")
    if stage.acquire or stage.free:
        raise ValueError(f"steady_pipeline_{label}_allocation_not_supported")
    if getattr(stage, "coupled", False):
        raise ValueError(f"steady_pipeline_{label}_coupled_not_supported")
    for resource, amount in stage.work.items():
        if not isinstance(resource, str) or not resource:
            raise ValueError(f"steady_pipeline_{label}_resource_invalid")
        try:
            finite = math.isfinite(float(amount))
        except (TypeError, ValueError):
            finite = False
        if not finite or float(amount) < 0:
            raise ValueError(f"steady_pipeline_{label}_work_invalid:{resource}")


def _validate_tail(base: Stage, tail: Stage, label: str) -> None:
    _validate_stream(tail, label)
    extra = _resources(tail) - _resources(base)
    if extra:
        names = ",".join(sorted(extra))
        raise ValueError(f"steady_pipeline_{label}_tail_resources_not_in_full:{names}")
    for resource in _resources(tail):
        if base.placement.get(resource) != tail.placement.get(resource):
            raise ValueError(f"steady_pipeline_{label}_tail_placement_mismatch:{resource}")


def _submit_stage(engine: Engine, stage: Stage, name: str, after: tuple[Event, ...],
                  owner: str) -> Event:
    return engine.submit(
        name, stage.work, after=after, caps=stage.caps,
        placement=stage.placement, owner=owner,
    )


@dataclass(frozen=True)
class SteadyPipeline:
    """N uniform full items followed by at most one exact, nonuniform tail.

    The compact form keeps the prologue, the coupled steady interval and the
    drain as ordinary Engine tasks. It never enumerates items or slots. Use a
    detailed ``PipelineProgram`` when item work, stream order or contention
    needs to be inspected tile by tile.
    """

    count: int
    producer: Stage
    consumer: Stage
    slots: int = 2
    tail_producer: Stage | None = None
    tail_consumer: Stage | None = None
    buffers: tuple[Allocation, ...] = ()

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise ValueError("steady_pipeline_count_requires_nonnegative_integer")
        if isinstance(self.slots, bool) or not isinstance(self.slots, int) or self.slots < 2:
            raise ValueError("steady_pipeline_slots_requires_at_least_two")
        _validate_stream(self.producer, "producer")
        _validate_stream(self.consumer, "consumer")
        shared = _resources(self.producer) & _resources(self.consumer)
        if shared:
            names = ",".join(sorted(shared))
            raise ValueError(f"steady_pipeline_producer_consumer_resources_must_be_disjoint:{names}")
        if (self.tail_producer is None) != (self.tail_consumer is None):
            raise ValueError("steady_pipeline_tail_requires_producer_and_consumer_together")
        if self.tail_producer is not None:
            # With more than two slots a tail producer can start before the
            # final full consumer, changing its exact finite completion.
            if self.slots != 2:
                raise ValueError("steady_pipeline_tail_requires_two_slots")
            _validate_tail(self.producer, self.tail_producer, "producer")
            _validate_tail(self.consumer, self.tail_consumer, "consumer")
        buffers = tuple(self.buffers or ())
        names: set[str] = set()
        for buffer in buffers:
            if not isinstance(buffer, Allocation):
                raise ValueError("steady_pipeline_buffers_require_allocations")
            if buffer.name in names:
                raise ValueError(f"steady_pipeline_buffer_name_not_unique:{buffer.name}")
            names.add(buffer.name)
        object.__setattr__(self, "buffers", buffers)

    def _steady_work(self) -> dict[str, float]:
        factor = self.count - 1
        work: dict[str, float] = {}
        for stage in (self.producer, self.consumer):
            for resource, amount in stage.work.items():
                if amount:
                    work[resource] = work.get(resource, 0.0) + amount * factor
        return work

    def _steady_caps(self) -> dict[str, float]:
        return {**self.producer.caps, **self.consumer.caps}

    def _steady_placement(self) -> dict[str, tuple[int, ...]]:
        return {**self.producer.placement, **self.consumer.placement}

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        """Submit the finite compact pipeline and return its completion event."""
        scoped = scope_allocations(name, self.buffers)
        if not self.count and self.tail_producer is None:
            # An empty invocation must not reserve buffers or invent a delay.
            return engine.submit(name + ":complete", after=(ready,), owner=owner)

        current = ready
        if scoped:
            # Acquisition is an ordinary Engine admission gate after readiness.
            current = engine.submit(
                name + ":reserve", after=(ready,), acquire=scoped, owner=owner)

        if self.count:
            current = _submit_stage(engine, self.producer, name + ":prologue",
                                    (current,), owner)
            if self.count > 1:
                # One coupled task carries every remaining full producer and
                # consumer demand. Coupling keeps the fast pipe from consuming
                # its work before the slow stage and before later arrivals.
                current = engine.submit(
                    name + ":steady", self._steady_work(), after=(current,),
                    caps=self._steady_caps(), placement=self._steady_placement(),
                    coupled=True, owner=owner,
                )
            if self.tail_producer is None:
                current = _submit_stage(engine, self.consumer, name + ":drain",
                                        (current,), owner)
            else:
                drain = _submit_stage(engine, self.consumer, name + ":drain",
                                      (current,), owner)
                tail_producer = _submit_stage(
                    engine, self.tail_producer, name + ":tail_producer",
                    (current,), owner)
                current = _submit_stage(
                    engine, self.tail_consumer, name + ":tail_consumer",
                    (drain, tail_producer), owner)
        else:
            # count == 0 with a tail: the single exact tail item is serial.
            current = _submit_stage(engine, self.tail_producer, name + ":tail_producer",
                                    (current,), owner)
            current = _submit_stage(engine, self.tail_consumer, name + ":tail_consumer",
                                    (current,), owner)

        if scoped:
            # The allocation lifetime spans every full drain and the tail
            # consumers; no stage work scales or duplicates this residency.
            current = engine.submit(
                name + ":release", after=(current,), free=scoped, owner=owner)
        return engine.submit(name + ":complete", after=(current,), owner=owner)
