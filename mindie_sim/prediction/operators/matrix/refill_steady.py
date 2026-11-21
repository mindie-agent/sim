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

from mindie_sim.prediction.engine import Allocation, Engine, Resource
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
    special_count = len({
        *range(min(head, count)),
        *range(max(0, count - eff_tail), count),
    })
    steady_count = count - special_count
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
    if not active:
        return False
    if len(active) > 1:
        if len({(h, t) for _, h, t in active}) != 1:
            return False
        if max(c for c, _, _ in active) - min(c for c, _, _ in active) > _ONE_TILE:
            return False
    return all(
        _can_compact_count(c, slots, head=h, tail=t, threshold=threshold)
        for c, h, t in active
    )


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
    peer_heads=None, peer_tails=None,
):
    """Compact a periodic PipelineProgram(count, slots, tile_fn) in O(1) time.

    When count exceeds threshold, tiles outside the prologue/epilogue repeat
    identical resource work. The exact period-slots recurrence on an isolated
    unit engine determines the private pipeline supply caps and peak pool
    occupancy, while all total work and buffers contend on the caller's shared
    Engine as a coupled fluid steady-state approximation. When ``peer_counts``
    names multiple active peers, callers must supply all peers' tail lengths
    (``peer_tails``) so the whole group takes one uniform all-or-nothing
    decision, and shared-resource tiles with serial wall latency retain literal
    service.
    """
    active_peer_count = 1
    if peer_counts is not None:
        active_counts = [int(c) for c in peer_counts if int(c) > 0]
        active_peer_count = len(active_counts)
        if active_peer_count > 1 and peer_tails is None:
            return PipelineProgram(count, slots, tile_fn)
        if not peer_group_can_compact(
            peer_counts, slots, head=head, tail=tail, threshold=threshold,
            peer_heads=peer_heads, peer_tails=peer_tails,
        ):
            return PipelineProgram(count, slots, tile_fn)
        if active_peer_count == 1 and active_counts[0] != count:
            return PipelineProgram(count, slots, tile_fn)
    step = max(2, slots)
    if count > head + tail:
        tail += (count - head - tail) % step
    special_indices = sorted({
        *range(min(head, count)),
        *range(max(0, count - tail), count),
    })
    special_set = set(special_indices)
    steady_count = count - len(special_set)
    if count <= threshold or steady_count <= _FOUR_TILES + _TWO_FACTOR * step:
        return PipelineProgram(count, slots, tile_fn)
    steady_idx = next((i for i in range(head, count - tail) if i not in special_set), None)
    if steady_idx is None:
        return PipelineProgram(count, slots, tile_fn)

    head_tiles = [tile_fn(i) for i in special_indices if i < steady_idx]
    steady_stages = tile_fn(steady_idx)
    tail_tiles = [tile_fn(i) for i in special_indices if i > steady_idx]

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
    if active_peer_count > 1:
        has_unplaced = any(k not in all_placement for k in total_work)
        has_latency = any(
            stage.latency > 0.0
            for group in (*head_tiles, steady_stages, *tail_tiles)
            for stage in group
        )
        if has_unplaced and has_latency:
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
