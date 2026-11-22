"""Compact regular refill loops whose producer is a finite resource DAG.

[Role] Preserve a producer DAG and a two-stream consumer's nominal K period.
[Invariants] One real producer prologue and final consumer remain explicit.
  Interior work and physical owners are conserved in a coupled fluid phase.
  Peer groups fold only after verifying joint literal recurrence and completion.
  No measured duration, fitted rate or model identity enters compilation.
[Timeouts & Locks] Cached finite producer controls and bounded cycle recurrence;
  no K-panel history. No threads, locks or I/O.
[Side-Effect Policy] Nominal producer controls use the same Engine in isolation;
  all actual resource service and buffer lifetimes stay on the caller's Engine.
"""
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, replace
from functools import lru_cache
from math import inf, lcm

from mindie_sim.prediction.engine import Allocation, Engine, ExecutionError, Resource
from .matrix_cycle import periodic_matrix_cycle
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, PipelineProgram, Stage

NOMINAL_PRODUCER_CACHE_SIZE = 512
UNDECLARED_POOL_CAPACITY_BYTES = 1e18


def _units(context):
    return {physical: context.resource_units.get(role, 1)
            for role, physical in context.resource_names.items()}


def _duration(context, stage):
    units = _units(context)
    return max(stage.latency, max((amount / min(stage.caps.get(key, inf), context.rates[key] *
                (len(stage.placement[key]) / units[key] if key in stage.placement else 1))
                for key, amount in stage.work.items()), default=0))


@lru_cache(maxsize=NOMINAL_PRODUCER_CACHE_SIZE)
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


_PROBE_CACHE_MAX = (2 * 2 * 2 * 2) * (2 * 2 * 2 * 2) * (2 * 2)


@lru_cache(maxsize=_PROBE_CACHE_MAX)
def _periodic_probe(resources, pools, slots, head_records, steady_records, tail_records, s_count):
    """Simulate one finite PipelineProgram probe of length ``h_len + s_count + t_len``."""
    @lru_cache(maxsize=None)
    def to_stages(records, slot):
        return tuple(
            Stage(name, dict(work), after=after, previous=previous,
                  acquire=tuple(Allocation(f"{n}_s{slot}", p, s, (0,)) for n, p, s in acq),
                  free=tuple(Allocation(f"{n}_s{slot}", p, s, (0,)) for n, p, s in fr),
                  placement=dict(placement), caps=dict(caps),
                  coupled=coupled, latency=latency)
            for name, work, after, previous, acq, fr, placement, caps, coupled, latency in records
        )
    h_len, t_len = len(head_records), len(tail_records)
    pool_dict = dict(pools)
    pool_units = {k: 1 for k in pool_dict}
    total = h_len + s_count + t_len
    engine = Engine(
        tuple(Resource(name, rate, units=1) for name, rate in resources),
        pools=pool_dict, pool_units=pool_units,
    )
    def tile(idx):
        slot = idx % slots
        if idx < h_len:
            return to_stages(head_records[idx], slot)
        if idx >= total - t_len:
            return to_stages(tail_records[idx - (total - t_len)], slot)
        return to_stages(steady_records, slot)
    done = PipelineProgram(total, slots, tile).start(engine, "p", engine.submit("r"))
    engine.advance(until=(done,))
    return done.time, tuple(sorted(engine.peak_allocated.items()))


_PERIODIC_CACHE_SIZE = 1024
_ONE_TILE = 1; _TWO_FACTOR = 2; _THREE_FACTOR = 3; _FOUR_TILES = 4; _DEFAULT_THRESHOLD = 8
_JOINT_RECURRENCE_REL_TOL = 1e-6
_JOINT_TIME_EPS_MS = 1e-9
_JOINT_WARMUP_PERIODS = (2, 12, 24)
_JOINT_PROBE_TILE_LIMIT = 128


@lru_cache(maxsize=_PERIODIC_CACHE_SIZE)
def _periodic_pair(resources, pools, slots, head_records, steady_records, tail_records, max_steady):
    """Verify 3-point periodic convergence and extract (base_w, t0, slope, peaks, converged)."""
    step = max(2, slots)

    def run_for(s_count):
        return _periodic_probe(resources, pools, slots, head_records, steady_records, tail_records, s_count)

    bn = _FOUR_TILES
    b0 = bn * _TWO_FACTOR
    b1 = b0 * _THREE_FACTOR
    b2 = b1 * _THREE_FACTOR
    b3 = b2 * _TWO_FACTOR
    for base in (bn, b0, b1, b2, b3):
        w = (base // step) * step
        if w + 2 * step > max_steady:
            break
        (t0, peaks), (t1, _), (t2, _) = run_for(w), run_for(w + step), run_for(w + 2 * step)
        d1 = t1 - t0
        d2 = t2 - t1
        if d1 > 0 and abs(d2 - d1) <= 1e-7 * max(1e-9, abs(d1)):
            return w, t0, d1 / step, peaks, True
    return 0, 0.0, 0.0, (), False


def _extract_leaves(stage):
    """Yield (work_dict, placement_dict) leaf stages; reject unsupported nested programs."""
    if stage.program is None:
        yield dict(stage.work), {k: tuple(sorted(set(u))) for k, u in stage.placement.items() if u}
        return
    if isinstance(stage.program, KernelProgram):
        for sub in stage.program.stages:
            yield from _extract_leaves(sub)
        return
    if isinstance(stage.program, ParallelProgram):
        for branch in stage.program.programs:
            if not isinstance(branch, KernelProgram):
                raise ValueError(f"unsupported_nested_stage_program:{type(branch).__name__}")
            for sub in branch.stages:
                yield from _extract_leaves(sub)
        return
    raise ValueError(f"unsupported_nested_stage_program:{type(stage.program).__name__}")


def _extract_stage(stage):
    """Extract (conserved_work, nominal_work, placement) including subblock programs."""
    if stage.program is None:
        w = dict(stage.work)
        return w, w, {k: set(u) for k, u in stage.placement.items()}
    if isinstance(stage.program, KernelProgram):
        c_work, n_work, place = {}, {}, {}
        for sub in stage.program.stages:
            cw, nw, pl = _extract_stage(sub)
            for k, v in cw.items():
                c_work[k] = c_work.get(k, 0.0) + v
            for k, v in nw.items():
                n_work[k] = n_work.get(k, 0.0) + v
            for k, u in pl.items():
                place.setdefault(k, set()).update(u)
        return c_work, n_work, place
    if isinstance(stage.program, ParallelProgram):
        c_work, max_b, place = {}, {}, {}
        for branch in stage.program.programs:
            if not isinstance(branch, KernelProgram):
                raise ValueError(f"unsupported_nested_stage_program:{type(branch).__name__}")
            b_cw, b_nw, b_pl = _extract_stage(Stage("_", program=branch))
            for k, v in b_cw.items():
                c_work[k] = c_work.get(k, 0.0) + v
            for k, v in b_nw.items():
                max_b[k] = max(max_b.get(k, 0.0), v)
            for k, u in b_pl.items():
                place.setdefault(k, set()).update(u)
        n_work = {
            k: (max_b[k] * len(place[k]) if k in place and len(place[k]) > 1 else c_work[k])
            for k in c_work
        }
        return c_work, n_work, place
    raise ValueError(f"unsupported_nested_stage_program:{type(stage.program).__name__}")


def _can_compact_count(
    count: int, slots: int, *, head: int = _ONE_TILE, tail: int = _ONE_TILE,
    threshold: int = _DEFAULT_THRESHOLD,
) -> bool:
    """Return True when a single owner's tile count meets the compaction floor."""
    step = max(_TWO_FACTOR, slots)
    eff_tail = tail + ((count - head - tail) % step) if count > head + tail else tail
    steady_count = max(0, count - head - eff_tail)
    return count > threshold and steady_count > _FOUR_TILES + _TWO_FACTOR * step


def peer_group_can_compact(
    peer_counts, slots: int, *, head: int = _ONE_TILE, tail: int = _ONE_TILE,
    threshold: int = _DEFAULT_THRESHOLD, peer_heads=None, peer_tails=None,
) -> bool:
    """Enforce one group-wide compaction decision across active peer pipelines."""
    counts = [int(c) for c in peer_counts]
    if peer_heads is not None and len(peer_heads) != len(counts):
        return False
    if peer_tails is not None and len(peer_tails) != len(counts):
        return False
    active = [
        (
            c,
            int(peer_heads[idx]) if peer_heads is not None else int(head),
            int(peer_tails[idx]) if peer_tails is not None else int(tail),
        )
        for idx, c in enumerate(counts) if c > 0
    ]
    return _matching_peer_boundaries(active) and all(
        _can_compact_count(c, slots, head=h, tail=t, threshold=threshold)
        for c, h, t in active
    )


def _matching_peer_boundaries(active):
    counts = [count for count, _, _ in active]
    return bool(counts) and (max(counts) - min(counts) <= _ONE_TILE
                             and len({(head, tail) for _, head, tail in active}) == 1)


def _stage_has_latency(stage):
    if stage.latency:
        return True
    if isinstance(stage.program, KernelProgram):
        return any(map(_stage_has_latency, stage.program.stages))
    if isinstance(stage.program, ParallelProgram):
        return any(_stage_has_latency(child) for program in stage.program.programs for child in program.stages)
    return False


def _encode_periodic_stages(stages, id_map, pool_id):
    alloc_map = {}
    for st in stages:
        for a in st.acquire:
            if a.name not in alloc_map:
                alloc_map[a.name] = f"b{len(alloc_map)}"
    encoded = []
    for st in stages:
        _, nw, pl = _extract_stage(st)
        encoded.append((
            st.name,
            tuple(sorted((id_map[k], float(v)) for k, v in nw.items() if v and k in id_map)),
            tuple(st.after),
            tuple(st.previous),
            tuple((alloc_map[a.name], pool_id[a.pool], float(a.size)) for a in st.acquire if a.pool in pool_id),
            tuple((alloc_map[a.name], pool_id[a.pool], float(a.size)) for a in st.free if a.pool in pool_id),
            tuple((id_map[k], (0,)) for k in sorted(pl) if k in id_map),
            tuple(sorted((id_map[k], float(v)) for k, v in st.caps.items() if k in id_map)),
            st.coupled,
            float(st.latency),
        ))
    return tuple(encoded)


def _build_grouped_steady_program(context, by_res, max_groups, span_ms, slot_allocs):
    """Build multi-group steady pipeline stages preserving distinct unit placements."""
    steady_stages_out = []
    steady_names = []
    for idx in range(max_groups):
        g_work = {}
        g_pl = {}
        g_caps = {}
        for k, groups in by_res.items():
            groups_sorted = sorted(groups, key=lambda x: x[0])
            if len(groups_sorted) == 1:
                units, val = groups_sorted[0]
                share = val / max_groups
            elif idx < len(groups_sorted):
                units, share = groups_sorted[idx]
            else:
                continue
            if share > 0:
                g_work[k] = share
                if units:
                    g_pl[k] = units
                if span_ms > 0:
                    g_caps[k] = share / span_ms
        name = f"steady_pipeline_{idx}"
        steady_names.append(name)
        steady_stages_out.append(Stage(
            name, g_work, after=("setup",), caps=g_caps, placement=g_pl, coupled=True,
        ))
    return KernelProgram((
        Stage("setup", latency=context.setup_ms, acquire=slot_allocs),
        *steady_stages_out,
        Stage("steady_release", after=tuple(steady_names), free=slot_allocs),
    ))


def compact_periodic_pipeline(
    context, count, slots, tile_fn, *, head=_ONE_TILE, tail=_ONE_TILE,
    threshold=_DEFAULT_THRESHOLD, resident=(), peer_counts=None,
    peer_heads=None, peer_tails=None, _peer_group_verified=False,
):
    """Compact a periodic PipelineProgram(count, slots, tile_fn) in O(1) time.

    When count exceeds threshold, tiles outside the prologue/epilogue repeat
    identical resource work. The exact period-slots recurrence on an isolated
    unit engine determines the private pipeline supply caps and peak pool
    occupancy, while all total work and buffers contend on the caller's shared
    Engine as a coupled fluid steady-state approximation. A peer group requires
    a joint decision by compact_periodic_peer_group before any candidate runs.
    """
    counts = () if peer_counts is None else tuple(int(count) for count in peer_counts)
    active_counts = [count for count in counts if count > 0]
    shared = len(active_counts) > 1
    if shared and not _peer_group_verified:
        return PipelineProgram(count, slots, tile_fn)
    if peer_counts is not None and not shared:
        if active_counts != [count] or not peer_group_can_compact(
            counts, slots, head=head, tail=tail, threshold=threshold,
            peer_heads=peer_heads, peer_tails=peer_tails,
        ):
            return PipelineProgram(count, slots, tile_fn)
    if not _can_compact_count(count, slots, head=head, tail=tail, threshold=threshold):
        return PipelineProgram(count, slots, tile_fn)
    step = max(2, slots)
    tail += (count - head - tail) % step
    steady_count = count - head - tail
    head_tiles = [tile_fn(i) for i in range(head)]
    steady_stages = tile_fn(head)
    tail_tiles = [tile_fn(i) for i in range(count - tail, count)]

    total_work = {}
    all_placement = {}
    unit_group_work = {}
    for tile_stages, mult in (
        *((t, 1) for t in head_tiles),
        (steady_stages, steady_count),
        *((t, 1) for t in tail_tiles),
    ):
        for stage in tile_stages:
            for lw, lpl in _extract_leaves(stage):
                for k, v in lw.items():
                    if v:
                        total_work[k] = total_work.get(k, 0.0) + v * mult
                        units = lpl.get(k, ())
                        unit_group_work[(k, units)] = unit_group_work.get((k, units), 0.0) + v * mult
                for k, units in lpl.items():
                    all_placement.setdefault(k, set()).update(units)
    all_placement = {k: tuple(sorted(u)) for k, u in all_placement.items() if k in total_work}
    if shared and any(not units for _, units in unit_group_work) and any(
        _stage_has_latency(stage) for group in (*head_tiles, steady_stages, *tail_tiles) for stage in group
    ):
        return PipelineProgram(count, slots, tile_fn)

    pool_units_by_phys = {}
    for s_stages in (steady_stages, *head_tiles, *tail_tiles):
        for stage in s_stages:
            for alloc in stage.acquire:
                pool_units_by_phys[alloc.pool] = alloc.units
    phys_pools = sorted(pool_units_by_phys)
    pool_id = {p: f"p{i}" for i, p in enumerate(phys_pools)}
    id_to_phys = {f"p{i}": p for i, p in enumerate(phys_pools)}
    phys_to_role = {v: k for k, v in context.pool_names.items()}
    resident_by_phys = {}
    for alloc in resident:
        resident_by_phys[alloc.pool] = resident_by_phys.get(alloc.pool, 0.0) + alloc.size
    pools_key = tuple(
        (pool_id[p], float(context.pool_capacities.get(phys_to_role.get(p),
                                                     UNDECLARED_POOL_CAPACITY_BYTES))
         - resident_by_phys.get(p, 0.0))
        for p in phys_pools
    )

    names = sorted(total_work)
    id_map = {name: str(i) for i, name in enumerate(names)}
    declared = _units(context)
    resources_key = tuple(
        (id_map[k], context.rates[k] * (len(all_placement[k]) / declared[k] if k in all_placement else 1.0))
        for k in names
    )

    w, t0, slope, peaks, converged = _periodic_pair(
        resources_key, pools_key, slots,
        tuple(_encode_periodic_stages(t, id_map, pool_id) for t in head_tiles),
        _encode_periodic_stages(steady_stages, id_map, pool_id),
        tuple(_encode_periodic_stages(t, id_map, pool_id) for t in tail_tiles),
        # This bound includes both verification periods, not just the base
        # warm-up. Passing the base alone rejected otherwise valid probes.
        steady_count,
    )
    if not converged:
        return PipelineProgram(count, slots, tile_fn)
    slot_allocs = tuple(
        Allocation(f"compact_{id_to_phys[pid]}", id_to_phys[pid], size, pool_units_by_phys[id_to_phys[pid]])
        for pid, size in peaks if size > 0
    )
    span_ms = t0 + (steady_count - w) * slope
    caps = {k: v / span_ms for k, v in total_work.items()} if span_ms > 0 else {}

    by_res = {}
    for (k, units), val in unit_group_work.items():
        by_res.setdefault(k, []).append((units, val))
    max_groups = max((len(v) for v in by_res.values()), default=1)
    if max_groups <= 1:
        return KernelProgram((
            Stage("setup", latency=context.setup_ms, acquire=slot_allocs),
            Stage("steady_pipeline", total_work, after=("setup",), caps=caps,
                  placement=all_placement, coupled=True, free=slot_allocs),
        ))
    return _build_grouped_steady_program(context, by_res, max_groups, span_ms, slot_allocs)


@dataclass(frozen=True, slots=True)
class PeriodicPipelineSpec:
    """One owner's periodic loop and its finite boundary tiles."""

    count: int
    slots: int
    tile_fn: Callable[[int], Sequence[Stage]]
    head: int = _ONE_TILE
    tail: int = _ONE_TILE
    threshold: int = _DEFAULT_THRESHOLD
    resident: tuple[Allocation, ...] = ()

    def literal(self) -> PipelineProgram:
        return PipelineProgram(self.count, self.slots, self.tile_fn)


def _truncated_peer_pipeline(spec, removed):
    count = spec.count - removed
    tail_start = count - spec.tail

    def tile(index):
        # The omitted span covers whole slot periods. Tail tiles keep their
        # original work, placement, stage dependencies and buffer parity.
        return spec.tile_fn(index if index < tail_start else index + removed)

    return PipelineProgram(count, spec.slots, tile)


def _probe_boundaries(spec):
    if spec.slots != _ONE_TILE:
        return spec
    steady = tuple(spec.tile_fn(spec.head))
    tail = max(_ONE_TILE, spec.tail)
    if all(tuple(spec.tile_fn(index)) == steady
           for boundary in (range(spec.head), range(spec.count - tail, spec.count)) for index in boundary):
        # A uniform serial loop has no distinct fill or drain. Probe its
        # empty baseline and successive real tiles from the same ready event.
        return replace(spec, head=0, tail=0)
    return spec


def _peer_group_times(engine, specs, programs):
    start = engine.now
    ready = engine.submit("ready")
    ends = []
    for index, (spec, program) in enumerate(zip(specs, programs, strict=True)):
        if spec.resident:
            program = KernelProgram((Stage(
                "resident", program=program, acquire=spec.resident, free=spec.resident,
            ),))
        ends.append(program.start(engine, f"peer{index}", ready))
    engine.advance(until=tuple(ends), first=False)
    if engine.allocations:
        raise ExecutionError("periodic_peer_probe_left_live_allocations")
    times = tuple(end.time - start for end in ends)
    # Individual completions can release downstream work before the group's end.
    return (*times, max(times))


def _steady_stage_demand(stage):
    """Summarize stage work and latency independently of unit placement."""
    program = stage.program
    children = ()
    if isinstance(program, KernelProgram):
        children = tuple(map(_steady_stage_demand, program.stages))
    elif isinstance(program, ParallelProgram):
        children = tuple(tuple(map(_steady_stage_demand, branch.stages)) for branch in program.programs)
    return stage.work, stage.latency, stage.caps, children


def _joint_recurrence_matches(context, specs, candidates) -> bool:
    """Validate a symmetric peer fold against the shared-engine recurrence and drain."""
    def demand(spec):
        return spec.slots, tuple(map(_steady_stage_demand, spec.tile_fn(spec.head)))

    first_demand = demand(specs[0])
    if any(demand(spec) != first_demand for spec in specs[1:]):
        return False
    specs = tuple(map(_probe_boundaries, specs))
    uniform_serial = all(spec.slots == _ONE_TILE and not (spec.head or spec.tail) for spec in specs)
    step = _ONE_TILE if uniform_serial else lcm(_TWO_FACTOR, *(spec.slots for spec in specs))
    # Remove the same number of periods from every peer, preserving any
    # one-tile difference between owners and the phase of each boundary.
    periods = min((spec.count - spec.head - spec.tail) // step for spec in specs)
    if periods < _TWO_FACTOR:
        return False
    host = context.resource_names.get("host")
    resources = tuple(Resource(name, context.rates[name], policy="fifo" if name == host else "fair", units=units)
                      for name, units in _units(context).items())
    pools = {name: context.pool_capacities[role] for role, name in context.pool_names.items()
             if role in context.pool_capacities}
    pool_units = {name: context.pool_units.get(role, _ONE_TILE) for role, name in context.pool_names.items()}
    literal_engine = Engine(resources, pools=pools, pool_units=pool_units, backend="python")
    max_count = max(spec.count for spec in specs)
    warmups = (0, *_JOINT_WARMUP_PERIODS) if uniform_serial else _JOINT_WARMUP_PERIODS
    try:
        for warmup in warmups:
            last_probe = warmup + _TWO_FACTOR
            if (last_probe > periods
                    or max_count - (periods - last_probe) * step > _JOINT_PROBE_TILE_LIMIT):
                break
            first, second, third = (
                _peer_group_times(literal_engine, specs, (
                    _truncated_peer_pipeline(spec, (periods - probe) * step) for spec in specs
                ))
                for probe in range(warmup, last_probe + _ONE_TILE)
            )
            early = tuple(b - a for a, b in zip(first, second, strict=True))
            late = tuple(c - b for b, c in zip(second, third, strict=True))
            if not all(delta > 0 and _joint_times_match(previous, delta)
                       for previous, delta in zip(early, late, strict=True)):
                continue
            remaining = periods - warmup - _ONE_TILE
            expected = (b + remaining * delta for b, delta in zip(second, late, strict=True))
            candidate_engine = Engine(resources, pools=pools, pool_units=pool_units, backend="python")
            compacted = _peer_group_times(candidate_engine, specs, candidates)
            return all(_joint_times_match(actual, extrapolated)
                       for actual, extrapolated in zip(compacted, expected, strict=True))
    except ExecutionError:
        # Simultaneous reservations may exceed a shared pool's capacity.
        return False
    return False


def _joint_times_match(actual, expected):
    return abs(actual - expected) <= _JOINT_RECURRENCE_REL_TOL * max(expected, _JOINT_TIME_EPS_MS)


def compact_periodic_peer_group(
    context, specs: Iterable[PeriodicPipelineSpec], *, compactor=compact_periodic_pipeline,
) -> tuple[KernelProgram | PipelineProgram, ...]:
    """Compile all active peers or keep every peer's literal execution."""
    specs = tuple(specs)
    literal = tuple(spec.literal() for spec in specs)
    active = tuple(spec for spec in specs if spec.count > 0)
    boundaries = [(spec.count, spec.head, spec.tail) for spec in active]
    if not _matching_peer_boundaries(boundaries) or not all(
        _can_compact_count(spec.count, spec.slots, head=spec.head, tail=spec.tail, threshold=spec.threshold)
        for spec in active
    ):
        return literal

    counts, heads, tails = map(tuple, zip(*boundaries))
    candidates = tuple(
        compactor(
            context, spec.count, spec.slots, spec.tile_fn,
            head=spec.head, tail=spec.tail, threshold=spec.threshold, resident=spec.resident,
            peer_counts=counts, peer_heads=heads, peer_tails=tails, _peer_group_verified=True,
        ) if spec.count > 0 else KernelProgram(())
        for spec in specs
    )
    if not all(isinstance(program, KernelProgram) for program in candidates):
        return literal
    active_candidates = tuple(program for spec, program in zip(specs, candidates, strict=True) if spec.count > 0)
    if len(active) > 1 and not _joint_recurrence_matches(context, active, active_candidates):
        return literal
    return candidates
