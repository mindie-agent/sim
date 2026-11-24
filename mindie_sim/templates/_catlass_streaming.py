"""Bounded-ring streaming CATLASS lowering and execution.

[Role] Lower and execute large CATLASS schedules (many tiles or deep K loops)
  using exact panel stages emitted through a bounded wave/ring window while
  immediately retiring completed stages, patterns, and dependency records.
[Invariants] Emits the exact same panel stages, ring credits, owner-specific
  SRAM grants, L2 residency updates, and FusedRegionSummary totals as eager
  unrolling while keeping peak RSS O(active wave * ring window).
[Timeouts & Locks] Finite caller-declared tile/ring schedules; no I/O or locks.
[Side-Effect Policy] Lowering mutates only private admission snapshots;
  execution mutates only the supplied Engine and optional live admission view.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Iterator, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.prediction.program.stages import (
    KernelProgram, Stage, _declares_no_work, scope_allocations,
)
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AdmissionMemoryView, HbmBusSpec, ReuseKey,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import (
    GATHER_SCATTER_POLICIES, MX_SCALE_GROUP, STREAMK_PARTIAL_SLOTS_PER_CORE, dtype_name,
    grouped_start_cores, packed_bytes, streamk_partition,
)
from .contracts import (
    FusedRegionSummary, SemanticWorkItem, TemplateLoweringResult,
    _logical_obligation, _split_entry_residency, _topological,
    _union_obligations, ceil_div, eligible_matrix_engines,
)
from .fuse_region import Lowering, ring_credit
from .fuse_region_live import _LiveAccess, _LiveAllocation

#: Panel-micro threshold above which CATLASS switches to bounded-ring streaming.
STREAMING_PANEL_MICRO_THRESHOLD = 2048
#: Maximum panels emitted per streaming wave window across active cores.
_WAVE_PANEL_BUDGET = 256
#: Extra safety margin above max ring depth for zero-bubble K-chunk triggers.
_RING_TRIGGER_MARGIN = 1


class _StreamingReuseLifetimes:
    """Retain unique template ReuseKeys across streaming windows until kernel finish."""

    def __init__(self, state: AdmissionMemoryView) -> None:
        self.state = state
        self.keys: dict[ReuseKey, tuple[ReuseKey, bool]] = {}

    def resolve(self, pattern):
        key = pattern.reuse_key
        if key is None:
            return pattern
        if key not in self.keys:
            resolved = self.state.resolve_reuse_key(key)
            self.keys[key] = (resolved, self.state.retain_template_key(resolved))
        return replace(pattern, reuse_key=self.keys[key][0])

    def complete(self, key: ReuseKey | None) -> None:
        _ = key

    def close(self) -> None:
        for resolved, owned in self.keys.values():
            if owned:
                self.state.release_template_key(resolved)
        self.keys.clear()


#: Obligation compaction batch ceiling before calling _union_obligations.
_OBLIGATION_COMPACT_THRESHOLD = 4096


def _ordered_window_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    """Topologically order one window's stages treating prior-window deps as satisfied."""
    n = len(stages)
    idx_by_name = {stage.name: i for i, stage in enumerate(stages)}
    if len(idx_by_name) != n:
        raise ModelingError("template_stage_name_duplicate")
    in_order = True
    for i, stage in enumerate(stages):
        for dep in stage.after:
            dep_idx = idx_by_name.get(dep)
            if dep_idx is not None and dep_idx >= i:
                in_order = False
                break
        if not in_order:
            break
    if in_order:
        return tuple(stages)
    indeg = [0] * n
    children: list[list[int]] = [[] for _ in range(n)]
    for i, stage in enumerate(stages):
        if not stage.after:
            continue
        seen_deps: set[int] | None = None
        for dep in stage.after:
            dep_idx = idx_by_name.get(dep)
            if dep_idx is None:
                continue
            if len(stage.after) > 1:
                if seen_deps is None:
                    seen_deps = set()
                if dep_idx in seen_deps:
                    continue
                seen_deps.add(dep_idx)
            children[dep_idx].append(i)
            indeg[i] += 1
    ready = deque(i for i, deg in enumerate(indeg) if deg == 0)
    ordered: list[Stage] = []
    while ready:
        cur = ready.popleft()
        ordered.append(stages[cur])
        for nxt in children[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(ordered) != n:
        raise ModelingError("template_stage_dependency_cycle")
    return tuple(ordered)


class _StreamingLowering(Lowering):
    """Windowed Lowering that binds, accounts, and retires stages wave-by-wave."""

    def __init__(
        self, datapath, *, context=None, memory_view=None, die: int = 0,
        live_admission: bool = False,
    ) -> None:
        super().__init__(
            datapath, context=context, memory_view=memory_view,
            die=die, live_admission=live_admission,
        )
        self._entry_state = deepcopy(self.state)
        self._streaming_lifetimes = _StreamingReuseLifetimes(self.state)
        self.total_stages = 0
        self.resource_work: Counter[str] = Counter()

    def access(
        self, name, tensor, amount, *, direction="read", pipe="mte2_aiv",
        after=(), owner=0, offset=0, runs=None, version=None, private=False,
        bypass=False, working_set=None, obligation=True,
    ) -> str:
        saved_state = self.state
        self.state = self._entry_state
        try:
            return super().access(
                name, tensor, amount, direction=direction, pipe=pipe,
                after=after, owner=owner, offset=offset, runs=runs,
                version=version, private=private, bypass=bypass,
                working_set=working_set, obligation=obligation,
            )
        finally:
            self.state = saved_state

    def add_ring_window(self, ring: RingBuffer, *, after=(), release_slots: bool = False) -> None:
        """Drain pending stages from ``ring`` while retaining only active ring slots."""
        for stage in ring.drain_stages(release_slots=release_slots):
            if stage.acquire or stage.free:
                acquire, free = stage.acquire, stage.free
                if self.context is not None:
                    acquire = tuple(self.context.allocate(a.name, a.pool, a.size, a.units) for a in acquire)
                    free = tuple(self.context.allocate(a.name, a.pool, a.size, a.units) for a in free)
                stage = replace(stage, acquire=acquire, free=free, after=stage.after or after)
            self.add(stage)

    def flush_window(self, *, emit: bool = False) -> tuple[Stage, ...]:
        """Bind current window accesses, compact obligations, and release retired stages."""
        if not self.stages:
            return ()
        stages = _ordered_window_stages(self.stages)
        if self.accesses:
            self.stages = []
            for stage in stages:
                pattern = self.accesses.pop(stage.name, None)
                self.add(
                    self._bind_access(stage, pattern, self._streaming_lifetimes)
                    if pattern is not None else stage
                )
            stages = tuple(self.stages)
        self.total_stages += len(stages)
        for st in stages:
            for role, amount in st.work.items():
                if role != "cube" and amount:
                    self.resource_work[role] += amount
        if len(self.reads) > _OBLIGATION_COMPACT_THRESHOLD:
            self.reads = list(_union_obligations(self.reads))
        if len(self.writes) > _OBLIGATION_COMPACT_THRESHOLD:
            self.writes = list(_union_obligations(self.writes))
        self.stages.clear()
        self.accesses.clear()
        self.obligations.clear()
        return stages if emit else ()

    def close_lifetimes(self) -> None:
        self._streaming_lifetimes.close()


def _ring_boundary_names(rings: Sequence[RingBuffer]) -> list[str]:
    """Return all live distance-d last_use and initial credit names still needed."""
    names: list[str] = []
    for ring in rings:
        names.extend(ring._initial_credits.values())
        for ev in ring._generations.values():
            names.append(ev.last_use)
    return names


def _emit_chunked_ordinary_wave(
    lowerer, plan: _StreamingLowering, tiles: Sequence[Any], after: tuple[str, ...],
    *, emit: bool,
) -> Iterator[tuple[tuple[Stage, ...], frozenset[str], tuple[str, ...], tuple[str, ...]]]:
    """Stream one wave of ordinary tiles in bounded K-panel chunks when K is deep."""
    from .catlass_lower import _compute_b, _cube_dtype, _issued_ops, _scale

    s, p = lowerer.spec, lowerer.p
    l1m, l1n, l1k = s.l1_tile_shape
    l0m, l0n, l0k = s.l0_tile_shape
    max_depth = max(s.stages)
    chunk_k = max(max_depth + 2, _WAVE_PANEL_BUDGET // max(1, len(tiles)))
    tile_states = []
    for tile in tiles:
        name = f"tile:{lowerer._tile_index}"
        lowerer._tile_index += 1
        core = tile.owner
        rings = tuple(
            RingBuffer(
                die=plan.die, owner=core, pool=pool, allocation=f"{name}:{tag}", depth=depth,
                slot_bytes=packed_bytes(elements, dtype)
                + (_scale(s, elements, l1=pool == "l1") if pool != "l0c" else 0)
                if plan.context is not None else 0,
            )
            for (tag, pool, elements, dtype), depth in zip((
                ("l1a", "l1", l1m * l1k, s.dtype_a), ("l1b", "l1", l1n * l1k, _compute_b(s)),
                ("l0a", "l0a", l0m * l0k, s.dtype_a), ("l0b", "l0b", l0n * l0k, _compute_b(s)),
                ("l0c", "l0c", l0m * l0n, s.dtype_acc),
            ), s.stages)
        )
        acc_credit = ring_credit(rings[4], 0)
        previous: tuple[str, ...] = (*after, acc_credit)
        if s.has_bias:
            previous = (plan.access(
                f"{name}:bias", p.c_id + ":bias", packed_bytes(tile.n, s.dtype_acc),
                owner=core, pipe="mte2_aic", offset=packed_bytes(tile.col, s.dtype_acc), after=previous,
            ),)
        tile_states.append([tile, name, rings, previous, 0])

    for k_start_panel in range(0, lowerer.gk, chunk_k):
        k_end_panel = min(lowerer.gk, k_start_panel + chunk_k)
        is_final_k = k_end_panel >= lowerer.gk
        triggers: list[str] = []
        keep: set[str] = set(after)
        for st in tile_states:
            tile, name, rings, previous, g = st
            l1a, l1b, l0a, l0b, accum = rings
            core = tile.owner
            panel_mmas: list[str] = []
            for panel in range(k_start_panel, k_end_panel):
                k_start = panel * lowerer.tk
                k_len = min(lowerer.tk, p.k - k_start)
                last_load = f"{name}:micro:{g + ceil_div(k_len, lowerer.micro_k) - 1}:loaded"
                r1 = []
                for operand, ring in (("a", l1a), ("b", l1b)):
                    deps = lowerer._fill(f"{name}:k{panel}", tile, k_start, k_len, operand, after, ring, panel)
                    r1.append(lower_generation(
                        ring, panel, "mte2", ("mte1",), producer_extra_after=deps,
                        consumer_extra_after={"mte1": (last_load,)},
                    ).publish)
                for start in range(0, k_len, lowerer.micro_k):
                    micro = min(k_len - start, lowerer.micro_k)
                    mma = f"{name}:micro:{g}:mma"
                    r0 = []
                    for rows, dtype, ring, prior in (
                        (tile.m, s.dtype_a, l0a, r1[0]), (tile.n, _compute_b(s), l0b, r1[1]),
                    ):
                        moved = packed_bytes(rows * micro, dtype) + (
                            rows * ceil_div(micro, MX_SCALE_GROUP) if s.mx_scale_group else 0
                        )
                        load = plan.work(f"{ring.allocation}:g{g}:load", "mte1", moved, owner=core,
                                         after=(prior, ring_credit(ring, g)))
                        r0.append(lower_generation(
                            ring, g, "mte1", ("mma",), producer_extra_after=(load,),
                            consumer_extra_after={"mma": (mma,)},
                        ).publish)
                    loaded = plan.add(Stage(f"{name}:micro:{g}:loaded", after=tuple(r0)))
                    issued = _issued_ops(tile.m, tile.n, micro, s, plan.datapath)
                    previous = (plan.work(
                        mma, "cube", issued, after=(*previous, loaded), owner=core,
                        dtype=_cube_dtype(s), metadata={"logical_flops": 2 * tile.m * tile.n * micro},
                    ),)
                    g += 1
                panel_mmas.append(previous[0])
            st[3] = previous
            st[4] = g
            if not is_final_k:
                for ring in rings:
                    plan.add_ring_window(ring, after=after, release_slots=False)
                keep.update(_ring_boundary_names(rings))
                keep.update(previous)
                trig_idx = max(0, len(panel_mmas) - 1 - max_depth - _RING_TRIGGER_MARGIN)
                triggers.append(panel_mmas[trig_idx])
        if not is_final_k:
            yield plan.flush_window(emit=emit), frozenset(keep), tuple(triggers), ()

    terminals: list[str] = []
    for tile, name, rings, previous, _ in tile_states:
        accum = rings[4]
        event = lower_generation(
            accum, 0, "mma", ("fixpipe",), producer_extra_after=previous,
            consumer_extra_after={"fixpipe": (f"{name}:accumulator_consumed",)},
        )
        terminals.append(lowerer.finish(tile, name, event.publish))
        for ring in rings:
            plan.add_ring_window(ring, after=after, release_slots=True)
    term_tuple = tuple(terminals)
    yield plan.flush_window(emit=emit), frozenset(term_tuple), (), term_tuple


def _iter_grouped_slice_windows(
    lowerer, plan: _StreamingLowering, launch: str, spec, problem, cores: int, *, emit: bool,
) -> Iterator[tuple[tuple[Stage, ...], frozenset[str], tuple[str, ...]]]:
    n_cols = problem.n // 2 if spec.epilogue_kind == "grouped_swiglu_mx_quant" else problem.n
    row_start, owner_done = 0, {}
    schedules = iter(grouped_start_cores(problem.groups, n_cols, lowerer.tm, lowerer.tn, cores))
    pending_count = 0
    for rows in problem.groups:
        if not rows:
            continue
        index, start_c, loops = next(schedules)
        for task_idx in range(loops):
            tile = lowerer.tile(
                task_idx, group=index, group_start=row_start,
                rows=rows, start_core=start_c, n_cols=n_cols,
            )
            owner_done[tile.owner] = lowerer.ordinary(tile, (owner_done.get(tile.owner, launch),))
            pending_count += 1
            if pending_count >= cores:
                keep = frozenset({launch, *owner_done.values()})
                yield plan.flush_window(emit=emit), keep, ()
                pending_count = 0
        row_start += rows
    if plan.stages:
        yield plan.flush_window(emit=emit), frozenset(), ()


def _iter_catlass_windows(
    lowerer, plan: _StreamingLowering, *, emit: bool,
) -> Iterator[tuple[tuple[Stage, ...], frozenset[str], tuple[str, ...]]]:
    """Yield ``(stages, keep_names, trigger_names)`` windows for any CATLASS schedule."""
    from .catlass_lower import _is_simt_matrix, _partial, _reduce_partials

    s, p = lowerer.spec, lowerer.p
    launch = plan.launch()
    if s.dispatch_policy in GATHER_SCATTER_POLICIES:
        launch = lowerer.zero_destination(launch)
    yield plan.flush_window(emit=emit), frozenset({launch}), ()

    blocks = ceil_div(p.m, lowerer.tm) * ceil_div(p.n, lowerer.tn)
    cores = s.aic_budget
    can_chunk_k = (
        s.dispatch_policy != "MmadPingpongMutex"
        and s.epilogue_kind != "grouped_swiglu_mx_quant"
        and not _is_simt_matrix(s)
    )

    def _run_normal_range(
        start: int, stop: int, after: tuple[str, ...], *, extra_keep: tuple[str, ...] = (),
    ) -> Iterator[tuple[Any, ...]]:
        nonlocal wave_after
        if s.dispatch_policy == "MmadPingpongMutex":
            for first in range(start, stop, cores):
                batch_end = min(first + cores, stop)
                is_last = batch_end >= stop
                for i in range(first, batch_end):
                    lowerer.ordinary(lowerer.tile(i), after)
                for rings in lowerer.owner_rings.values():
                    for ring in rings:
                        plan.add_ring_window(ring, after=after, release_slots=is_last)
                keep = set(after) | set(extra_keep)
                keep.update(lowerer.mma_tail.values())
                keep.update(lowerer.fixpipe_tail.values())
                for rings in lowerer.owner_rings.values():
                    keep.update(_ring_boundary_names(rings))
                yield plan.flush_window(emit=emit), frozenset(keep), ()
            wave_after = tuple(lowerer.fixpipe_tail.values()) or after
            return
        cur_after = after
        for first in range(start, stop, cores):
            tiles = tuple(lowerer.tile(i) for i in range(first, min(first + cores, stop)))
            if can_chunk_k and len(tiles) * lowerer.gk > _WAVE_PANEL_BUDGET:
                for w_stages, w_keep, w_trig, w_terms in _emit_chunked_ordinary_wave(
                    lowerer, plan, tiles, cur_after, emit=emit,
                ):
                    if w_terms:
                        cur_after = w_terms
                    yield w_stages, (w_keep | frozenset(extra_keep)) if extra_keep else w_keep, w_trig
            else:
                cur_after = tuple(lowerer.ordinary(tile, cur_after) for tile in tiles)
                yield plan.flush_window(emit=emit), frozenset((*cur_after, *extra_keep)), ()
        wave_after = cur_after

    wave_after: tuple[str, ...] = (launch,)
    if s.scheduler == "grouped_slice_m":
        yield from _iter_grouped_slice_windows(lowerer, plan, launch, s, p, cores, emit=emit)
        return

    if s.scheduler in {"streamk", "multi_core_splitk"}:
        parts: dict[int, list[Any]] = defaultdict(list)
        if s.scheduler == "streamk":
            partition = streamk_partition(blocks, lowerer.gk, cores)
            if not partition.b_sk:
                yield from _run_normal_range(0, blocks, (launch,))
                return
            final = max(0, partition.b_norm - cores)
            yield from _run_normal_range(0, final, (launch,))
            pre = wave_after
            stride = packed_bytes(lowerer.tn, s.dtype_acc)
            tile_bytes = lowerer.tm * stride
            part_names: list[str] = []
            for core, (st_idx, en_idx) in enumerate(partition.intervals):
                c_after = pre
                for part in range(partition.partial_counts[core]):
                    block, panel = divmod(st_idx, lowerer.gk)
                    stop_idx = min(en_idx, (block + 1) * lowerer.gk)
                    tile = replace(lowerer.tile(partition.b_norm + block), owner=core)
                    offset = (core * STREAMK_PARTIAL_SLOTS_PER_CORE + part) * tile_bytes
                    _partial(lowerer, parts, block, tile, range(panel, panel + stop_idx - st_idx), c_after, offset, stride)
                    c_after = (parts[block][-1][1],)
                    part_names.append(c_after[0])
                    st_idx = stop_idx
            keep = frozenset({*pre, *part_names})
            yield plan.flush_window(emit=emit), keep, ()
            signal = plan.add(Stage(
                "cross_core_set_flag", after=tuple(part_names),
                metadata={"streamk_partial_count": partition.partial_count},
            ))
            yield plan.flush_window(emit=emit), frozenset({signal}), ()
            # Stream tail normal waves and AIV partial reductions after the same signal
            # without serializing AIV reductions behind the AIC tail wave.
            tail_windows = list(_run_normal_range(final, partition.b_norm, (signal,), extra_keep=(signal,)))
            if tail_windows:
                for idx, (w_stages, w_keep, w_trig) in enumerate(tail_windows):
                    if idx == len(tail_windows) - 1:
                        yield w_stages, w_keep | frozenset({signal}), (signal,)
                    else:
                        yield w_stages, w_keep, w_trig
        else:
            base, rem = divmod(lowerer.gk, s.splitk_factor)
            owner_done: dict[int, str] = {}
            stride = packed_bytes(p.n, s.dtype_acc)
            matrix_bytes = p.m * stride
            total_tasks = s.splitk_factor * blocks
            for first in range(0, total_tasks, cores):
                for task_id in range(first, min(first + cores, total_tasks)):
                    split, block = divmod(task_id, blocks)
                    st_k = split * base + min(split, rem)
                    en_k = st_k + base + (split < rem)
                    owner = task_id % cores
                    tile = replace(lowerer.tile(block), owner=owner)
                    offset = split * matrix_bytes + tile.row * stride + packed_bytes(tile.col, s.dtype_acc)
                    _partial(lowerer, parts, block, tile, range(st_k, en_k),
                             (owner_done.get(owner, launch),), offset, stride)
                    owner_done[owner] = parts[block][-1][1]
                keep = frozenset({launch, *owner_done.values()})
                yield plan.flush_window(emit=emit), keep, ()
            signal = plan.add(Stage("cross_core_set_flag", after=tuple(owner_done.values())))
            yield plan.flush_window(emit=emit), frozenset({signal}), ()
        block_keys = list(parts)
        for first in range(0, len(block_keys), max(1, s.aiv_budget)):
            batch = {b: parts[b] for b in block_keys[first:first + max(1, s.aiv_budget)]}
            _reduce_partials(lowerer, batch, signal)
            yield plan.flush_window(emit=emit), frozenset({signal}), ()
        return

    yield from _run_normal_range(0, blocks, (launch,))


@dataclass(frozen=True)
class _StreamingCatlassProgram:
    """Bounded-ring streaming Program emitting exact CATLASS panel stages wave-by-wave."""

    spec: Any
    problem: Any
    datapath: Any
    context: Any
    initial_state: AdmissionMemoryView | None
    die: int
    hbm_bus: HbmBusSpec | None
    live_state: AdmissionMemoryView | None
    summary: FusedRegionSummary
    resource_work: tuple[tuple[str, float], ...] = ()
    burst_rules: tuple[Any, ...] = ()

    def with_context(self, datapath, context) -> "_StreamingCatlassProgram":
        """Rebind runtime ``datapath`` and ``context``; re-lower summary on L2/access/burst change."""
        probe = Lowering(datapath, context=context, die=self.die)
        if (
            dict(datapath.buffers).get("l2", 0) != dict(self.datapath.buffers).get("l2", 0)
            or datapath.memory_access != self.datapath.memory_access
            or probe.burst_rules != self.burst_rules
        ):
            from .catlass_lower import solve_catlass_tile

            live_adm = self.live_state is not None
            warm_view = (
                self.live_state if live_adm
                else (deepcopy(self.initial_state) if self.initial_state is not None else None)
            )
            sol = solve_catlass_tile(self.spec, self.problem, datapath, memory_view=warm_view, die=self.die)
            return lower_catlass_streaming(
                self.spec, self.problem, datapath, sol,
                context=context, memory_view=warm_view, die=self.die,
                live_admission=live_adm, region_id=self.summary.region_id,
            ).program
        return replace(
            self,
            datapath=datapath,
            context=context,
            hbm_bus=probe.hbm_bus,
        )

    def start(self, engine, name: str, ready, *, owner: str = ""):
        from .catlass_lower import _MatrixLowerer

        bus = self.hbm_bus
        if bus is not None:
            present = (bus.row_r in engine.resources, bus.row_w in engine.resources)
            if not any(present):
                engine.add_hbm_bus(bus)
            elif all(present):
                engine.bind_hbm_bus(bus)
            else:
                raise ModelingError("template_directional_hbm_rows_incomplete")

        live_admission = self.live_state is not None
        plan = _StreamingLowering(
            self.datapath, context=self.context,
            memory_view=self.live_state if live_admission else self.initial_state,
            die=self.die, live_admission=live_admission,
        )
        lowerer = _MatrixLowerer(plan, self.spec, self.problem)
        windows = _iter_catlass_windows(lowerer, plan, emit=True)
        complete = engine.future(f"{name}:complete")
        live_events: dict[str, Any] = {}
        pending_sinks: list[Any] = []
        delivered = Counter()
        obligations: dict[str, list[Any]] = {"read": [], "write": []}
        live_lifetimes = _StreamingReuseLifetimes(self.live_state) if live_admission else None
        window_idx = 0

        def scoped(items):
            return scope_allocations(name, items)

        def finalize_complete(_ev=None) -> None:
            pending_sinks.clear()
            live_events.clear()
            plan.close_lifetimes()
            if live_lifetimes is not None:
                live_lifetimes.close()
                self.summary._record_delivery(
                    delivered,
                    _union_obligations(obligations["read"]),
                    _union_obligations(obligations["write"]),
                )
            engine.resolve(complete)

        def finish_stream(_ev=None) -> None:
            still_running = tuple(dict.fromkeys(ev for ev in pending_sinks if not ev.done))
            if still_running:
                pending_sinks.clear()
                engine.submit(
                    f"{name}:drain", after=still_running, owner=owner,
                    on_complete=finalize_complete, native_hook=True,
                )
                return
            finalize_complete()

        def pump(_ev=None) -> None:
            nonlocal window_idx
            try:
                stages, keep_names, trigger_names = next(windows)
            except StopIteration:
                finish_stream()
                return
            while not stages:
                try:
                    stages, keep_names, trigger_names = next(windows)
                except StopIteration:
                    finish_stream()
                    return
            w_idx = window_idx
            window_idx += 1
            local_events: dict[str, Any] = {}
            consumed: set[str] = set()
            for stage in stages:
                if stage.after:
                    consumed.update(stage.after)
                preds = tuple(
                    local_events[k] if k in local_events else live_events[k]
                    for k in stage.after
                ) or (ready,)
                if live_admission:
                    if isinstance(stage.program, _LiveAccess):
                        stage = replace(stage, program=replace(
                            stage.program, delivered=delivered, lifetimes=live_lifetimes,
                            obligations=obligations, explicit=False,
                        ))
                    elif (engine._accel is not None and (stage.acquire or stage.free)
                            and not (stage.program or stage.work or stage.latency or stage.fragment or stage.deliveries)):
                        scoped_st = replace(
                            stage, acquire=scoped(stage.acquire), free=scoped(stage.free),
                        )
                        stage = Stage(
                            stage.name, after=stage.after,
                            program=_LiveAllocation(f"{name}:{stage.name}", scoped_st),
                        )
                if stage.program is None and _declares_no_work(stage) and len(preds) == 1:
                    local_events[stage.name] = preds[0]
                    continue
                local_events[stage.name] = stage.submit(
                    engine, f"{name}:{stage.name}", preds, scoped, owner,
                )
            sinks = tuple(dict.fromkeys(
                local_events[st.name] for st in stages if st.name not in consumed
            )) or (ready,)
            if trigger_names:
                wait_on = tuple(dict.fromkeys(
                    local_events[k] if k in local_events else live_events[k]
                    for k in trigger_names
                ))
                pending_sinks.extend(ev for ev in sinks if not ev.done)
            else:
                wait_on = sinks
                if pending_sinks:
                    pending_sinks[:] = [ev for ev in pending_sinks if not ev.done]
            next_live = {}
            for k in keep_names:
                if k in local_events:
                    next_live[k] = local_events[k]
                elif k in live_events:
                    next_live[k] = live_events[k]
            live_events.clear()
            live_events.update(next_live)
            local_events.clear()
            engine.submit(
                f"{name}:win{w_idx}:next", after=wait_on, owner=owner,
                on_complete=pump, native_hook=True,
            )

        pump()
        return complete


def lower_catlass_streaming(
    spec, p, datapath, solution, *,
    context=None, memory_view=None, die: int = 0,
    live_admission: bool = False, region_id=None,
) -> TemplateLoweringResult:
    """Lower a large CATLASS schedule using bounded-ring wave/panel streaming."""
    from ._catlass_periodic import lower_catlass_periodic, supports_exact_periodic_catlass
    from .catlass_lower import _MatrixLowerer, _cube_dtype, _is_simt_matrix

    if supports_exact_periodic_catlass(spec, p, live_admission=live_admission):
        return lower_catlass_periodic(
            spec, p, datapath, solution,
            context=context, memory_view=memory_view, die=die, region_id=region_id,
        )

    initial_state = deepcopy(memory_view) if (memory_view is not None and not live_admission) else None
    plan = _StreamingLowering(
        datapath, context=context, memory_view=memory_view,
        die=die, live_admission=live_admission,
    )
    lowerer = _MatrixLowerer(plan, spec, p)
    for _ in _iter_catlass_windows(lowerer, plan, emit=False):
        pass
    plan.close_lifetimes()
    rid = region_id or spec.template_id
    semantic = [SemanticWorkItem(
        rid + ":matrix", "matrix_mac", _cube_dtype(spec), 2 * p.m * p.n * p.k,
        eligible_matrix_engines(p, datapath), (spec.scheduler,),
    )]
    kinds = {
        "vec_alu": "vector_alu", "vec_sfu": "vector_sfu",
        "vec_reduce": "vector_reduce", "aiv_simt": "vector_alu",
    }
    semantic.extend(
        SemanticWorkItem(
            rid + ":" + pipe, kinds[pipe], dtype_name(spec.dtype_acc), int(ops),
            ("aiv_simt",) if pipe == "aiv_simt" else ("aiv_simd",), (spec.scheduler,),
        )
        for pipe, ops in plan.vector.items() if ops
        and not (pipe == "aiv_simt" and _is_simt_matrix(spec))
        and not (pipe == "vec_reduce" and spec.scheduler in {"streamk", "multi_core_splitk"})
    )
    summary = FusedRegionSummary(
        "K2", rid, (rid,), tuple(replace(item, die=die) for item in semantic),
        tuple(sorted(plan.cube.items())),
        tuple(sorted(plan.vector.items())),
        _union_obligations(plan.reads),
        _union_obligations(plan.writes),
        plan.logical, plan.copied, plan.sectors, plan.service, plan.hits, (), plan.launches,
        plan.metadata_ms, tuple((peak.pool, peak.peak_bytes) for peak in solution.pool_peaks), spec,
        None if live_admission else float(plan.sectors), plan.misses,
        None if live_admission else float(plan.service),
    )
    program = _StreamingCatlassProgram(
        spec=spec, problem=p, datapath=datapath, context=context,
        initial_state=initial_state, die=die, hbm_bus=plan.hbm_bus,
        live_state=plan.live_state, summary=summary,
        resource_work=tuple(sorted(plan.resource_work.items())),
        burst_rules=plan.burst_rules,
    )
    return TemplateLoweringResult(program, summary, plan.total_stages)
