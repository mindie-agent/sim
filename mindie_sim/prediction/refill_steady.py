"""Compact regular refill loops whose producer is a finite resource DAG.

[Role] Preserve a producer DAG and a two-stream consumer's nominal K period.
[Invariants] One real producer prologue and final consumer remain explicit.
  Interior work and physical owners are conserved in a coupled fluid phase;
  no measured duration, fitted rate or model identity enters compilation.
[Timeouts & Locks] Cached finite producer controls and bounded cycle recurrence;
  no K-panel history. No threads, locks or I/O.
[Side-Effect Policy] Nominal producer controls use the same Engine in isolation;
  all actual resource service and buffer lifetimes stay on the caller's Engine.
"""
from functools import lru_cache
from math import inf
from dataclasses import replace

from .engine import Engine, Resource
from .matrix_cycle import periodic_matrix_cycle
from .program import KernelProgram, Stage


def _units(context):
    return {physical: context.resource_units.get(role, 1)
            for role, physical in context.resource_names.items()}


def _duration(context, stage):
    units = _units(context)
    return max(stage.latency, max((amount / min(stage.caps.get(key, inf), context.rates[key] *
                (len(stage.placement[key]) / units[key] if key in stage.placement else 1))
                for key, amount in stage.work.items()), default=0))


@lru_cache(maxsize=512)
def _nominal(resources, records):
    """Execute only one small producer DAG, never the repeated loop."""
    engine = Engine(tuple(Resource(str(i), rate, units=units)
                          for i, (rate, units) in enumerate(resources)))
    stages = tuple(Stage(name, dict(work), after=after, placement=dict(placement),
                         caps=dict(caps), coupled=coupled, latency=latency)
                   for name, work, after, placement, caps, coupled, latency in records)
    end = KernelProgram(stages).start(engine, "producer", engine.submit("ready"))
    engine.advance(until=(end,))
    return end.time


def producer_duration(context, stages):
    # Normalize rank/owner names so equal physical DAGs share a bounded cache.
    # AIV0 and AIV1 remain distinct units of the same physical vector pipe.
    names = sorted({key for stage in stages for key in stage.work})
    identifiers = {name: str(i) for i, name in enumerate(names)}
    declared = _units(context)
    placements = {key: sorted({unit for stage in stages
                              for unit in stage.placement.get(key, ())}) for key in names}
    resources = tuple((context.rates[key] * len(placements[key]) / declared[key],
                       len(placements[key])) if placements[key] else (context.rates[key], 1)
                      for key in names)
    records = tuple((stage.name, tuple((identifiers[key], value) for key, value in stage.work.items()),
                     stage.after, tuple((identifiers[key], tuple(placements[key].index(i) for i in owners))
                                        for key, owners in stage.placement.items() if key in identifiers),
                     tuple((identifiers[key], value) for key, value in stage.caps.items()
                           if key in identifiers), stage.coupled, stage.latency)
                    for stage in stages)
    return _nominal(resources, records)


def compact_refill(context, pattern, repeats, tail=(), *, refill_slots=2, consumer_slots=2,
                   shared_clients=None):
    """Compress a periodic tuple of (producer DAG, [(operand, cube), ...]).

    Producers serialize as in RefillPipeline; both consumer streams and their
    finite slots span panels. External contention is a fluid approximation,
    bounded by unchanged resource demand and the exact nominal loop recurrence.
    Producer/consumer shared-pipe contention remains a resource bound in the
    compact interior; it is not an exact arbitrary-contention schedule.
    """
    active_pattern = pattern if repeats else ()
    first, last = (active_pattern or tail)[0], (tail or active_pattern)[-1]
    first_load, final = first[0], last[1][-1][1]
    # Simultaneously launched owner pipelines share the die's HBM. Computing
    # each private refill period with the whole die rate hides finite-slot
    # stalls when memory is scarce. This nominal fair share sets only pipeline
    # supply caps; actual bytes still contend on the original shared resource.
    nominal = replace(context, rates={key: value / (shared_clients or {}).get(key, 1)
                                      for key, value in context.rates.items()})
    def times(part):
        return (producer_duration(nominal, part[0]),
                tuple((_duration(context, a), _duration(context, b)) for a, b in part[1]))
    cycle = periodic_matrix_cycle(tuple(map(times, active_pattern)), repeats,
                                  refill_slots, consumer_slots, tail=tuple(map(times, tail)))
    work, owners = {}, {}
    for repeat, panels in ((repeats, active_pattern), (1, tail)):
        for panel in panels:
            for stage in (*panel[0], *(s for pair in panel[1] for s in pair)):
                for key, value in stage.work.items():
                    work[key] = work.get(key, 0) + repeat * value
                for key, units in stage.placement.items():
                    owners.setdefault(key, set()).update(units)
    for stage in (*first_load, final):
        for key, value in stage.work.items():
            work[key] -= value
    work = {key: value for key, value in work.items() if value > 0}
    placement = {key: tuple(sorted(units)) for key, units in owners.items() if key in work}
    interior = Stage("steady", work, placement=placement, coupled=True)
    span = max(_duration(context, interior), cycle.span_ms -
               producer_duration(nominal, first_load) - _duration(context, final))
    return KernelProgram((
        Stage("load", program=KernelProgram(first_load)),
        Stage("steady", work, after=("load",), placement=placement, coupled=True,
              caps={key: value / span for key, value in work.items()} if span else {}),
        Stage("cube", final.work, after=("steady",), placement=final.placement,
              caps=final.caps, coupled=final.coupled, latency=final.latency),
    ))
