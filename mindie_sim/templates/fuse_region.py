"""Compile physically admitted tile-pipeline regions onto the simulator engine.

[Role] Check region legality, resolve routes, solve liveness and emit work.
[Invariants] Physical pool owners, directional placement, and byte ledgers agree.
[Timeouts & Locks] Finite caller-declared schedules. No locks or remote calls.
[Side-Effect Policy] Lowering uses a private admission snapshot. Execution may
  update an explicitly supplied live admission view.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.l2_residency import _access_plan, cache_policy_for_arch, primitive_working_set_sweep
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, BoundAccess, BurstTurnRule, ByteLedger,
    ConstraintTerm, CounterTerm, HbmBusSpec,
    MemoryAccessPattern, ReuseKey, SectorCoverage, VisibilityContract, bind_memory_access,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    AIC_POOLS, LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape, solve,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS

from .contracts import (
    EdgeRouteBinding, FuseRegionSpec, FusedRegionSummary, SemanticWorkItem,
    TemplateLoweringResult, TensorRangeObligation, _external_reads, _logical_obligation, _obligation_runs,
    _record_produced, _region_order, _run_end, _split_entry_residency, _topological,
    _union_obligations, ceil_div,
)
from .fuse_region_live import _BoundTemplateProgram, _LiveAccess, _ReuseLifetimes, _rational_service

_VECTOR_PIPES = {"vector", "vec_alu", "vec_sfu", "vec_reduce", "aiv_simt", "sort_topk"}
_AIC_RESOURCES = {"cube", "fixpipe", "mte2_aic", "mte1"}
_AIV_RESOURCES = _VECTOR_PIPES | {"mte2_aiv", "mte3"}


def validate_live_buffer_owners(buffers: Sequence[LiveBufferSpec]) -> None:
    for buffer in buffers:
        allowed = (
            {"aiv"} if buffer.pool == "ub" else {"aic"} if buffer.pool in AIC_POOLS
            else {"aic", "aiv", "die"} if buffer.pool == "l2" else set()
        )
        if buffer.owner_kind not in allowed:
            raise ModelingError(f"live_buffer_pool_owner_mismatch:{buffer.name}:{buffer.pool}:{buffer.owner_kind}")


def solve_template_tile(datapath, family, candidate, *, state=None, die=0, reserved=0):
    validate_live_buffer_owners(candidate.buffers)
    if candidate.uses_simt and datapath.simt_ub_carveout_bytes <= 0:
        raise ModelingError("template_simt_carveout_required")
    existing = state.reserved_bytes(die) if state is not None else 0
    capacity = dict(datapath.buffers)
    if reserved < 0 or reserved + existing > capacity.get("l2", 0):
        raise ModelingError(f"template_l2_reservation_overflow:{family}")
    streams = max(1, state.active_stream_count(die)) if state is not None else 1
    options = dict(l2_granule_bytes=state.granule_bytes) if state is not None else {}
    pipeline = TilePipelineSpec(
        family, (candidate,),
        simt_ub_carveout_bytes=datapath.simt_ub_carveout_bytes if candidate.uses_simt else 0,
        subtile_waves=datapath.subtile_wave,
        active_l2_streams=streams, l2_reserved_bytes=reserved + existing, **options,
    )
    hw = SimpleNamespace(datapath=datapath, buffers=capacity, buffer_units={"l2": 1})
    solution = solve(pipeline, None, hw)
    if candidate.buffers and not solution.pool_peaks:
        raise ModelingError(f"template_live_buffers_without_pool_peaks:{family}")
    return solution


def ordered_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    by_name = {stage.name: stage for stage in stages}
    if len(by_name) != len(stages):
        raise ModelingError("template_stage_name_duplicate")
    if any(dep not in by_name for stage in stages for dep in stage.after):
        raise ModelingError("template_stage_dependency_unknown")
    order = _topological({stage.name: stage.after for stage in stages}, "template_stage_dependency_cycle")
    return tuple(by_name[name] for name in order)


def ring_credit(ring, generation):
    return ring.initial_credit(ring.slot(generation)) if generation < ring.depth else ring.last_use(generation - ring.depth)


def bind_template_access_ledger(
    pattern: MemoryAccessPattern, ledger: ByteLedger, owner: int, context, hbm_bus: HbmBusSpec | None,
    name: str, *, after: tuple[str, ...] = (), previous: tuple[str, ...] = (),
    acquire: tuple[Any, ...] = (), free: tuple[Any, ...] = (),
) -> Stage:
    """Bind a frozen byte ledger to the supplied resource rates."""
    aligned = ledger.aligned_transfer_bytes
    if aligned == 0:
        return Stage(name, after=after, previous=previous, acquire=acquire, free=free)
    hit, miss = ledger.l2_hit_bytes, ledger.dram_miss_copy_bytes
    sectors, service = ledger.dram_sector_bytes, ledger.dram_bus_transaction_bytes
    if context is not None:
        pipe_role = pattern.pipe
        if pipe_role not in context.resource_names and pipe_role.startswith("mte2"):
            pipe_role = "mte2"
        if pipe_role not in context.resource_names:
            raise ModelingError(f"bind_access_pipe_missing:{pattern.pipe}")
        pipe_row = context.resource_names[pipe_role]
        pipe_units = context._units_for(pattern.pipe, pipe_row, context._units_for(pipe_role, pipe_row, 1))
        nom_pipe = float(context.rates[pipe_row]) / float(pipe_units)
        prim = context._datacopy_cell()
        hit_rate = context._copy_rate(pattern.pipe, "hit", pipe_units, nom_pipe, prim)
        miss_rate = context._copy_rate(pattern.pipe, "miss", pipe_units, nom_pipe, prim)
        arch = getattr(context.datapath, "arch_id", "generic")
        if arch != "generic":
            cache_policy_for_arch(arch, pattern.cache_policy)
        l2_row = context.resource_names.get("l2", "l2")
        l2_rate = float(context.rates[l2_row]) if l2_row in context.rates else hit_rate
        if hbm_bus is None:
            raise ModelingError("bind_access_hbm_rate_missing")
        bus = hbm_bus
    else:
        pipe_row, l2_row = pattern.pipe, "l2"
        nom_pipe = hit_rate = miss_rate = l2_rate = 1.0
        bus = HbmBusSpec(pattern.die, 1.0, 1.0, 0.0, row_r="hbm", row_w="hbm")

    constraints: list[ConstraintTerm] = []
    row_rates: dict[str, float] = {}
    pipe_ms = float(hit) / hit_rate + float(miss) / miss_rate
    if pipe_ms > 0.0:
        constraints.append(ConstraintTerm(pipe_row, pipe_ms))
        row_rates[pipe_row] = nom_pipe
    if hit > 0 and (context is None or l2_row in context.rates):
        constraints.append(ConstraintTerm(l2_row, float(hit) / l2_rate))
        row_rates[l2_row] = l2_rate
    deliveries: list[CounterTerm] = [CounterTerm(pipe_row, float(aligned))]
    if hit > 0:
        deliveries.append(CounterTerm(l2_row, float(hit)))
    if service > 0.0:
        term_r, term_w = bus.terms_for(pattern.direction, service)
        if term_r.ms_per_progress > 0.0:
            constraints.append(term_r)
        if term_w.ms_per_progress > 0.0:
            constraints.append(term_w)
        ctr_id = bus.ctr_r if pattern.direction == "read" else bus.ctr_w
        if sectors:
            deliveries.append(CounterTerm(ctr_id, float(sectors)))
    bound = BoundAccess(
        ledger=ledger, constraints=tuple(constraints), deliveries=tuple(deliveries),
        after=after, owner=owner, pipe_row_id=pipe_row, l2_row_id=l2_row, row_rates=row_rates,
    )
    stage = bound.to_stage(name, after=after)
    if previous or acquire or free:
        stage = replace(stage, previous=previous, acquire=acquire, free=free)
    return stage


class Lowering:
    """Shared stage and byte ledger construction for the catalog lowerers."""

    def __init__(
        self, datapath, *, context: RecipeContext | None = None,
        memory_view: AdmissionMemoryView | None = None, die: int = 0,
        live_admission: bool = False,
    ):
        self.datapath = datapath
        self.context = context
        self.die = die
        memory = datapath.memory_access
        self.state = deepcopy(memory_view) if memory_view is not None else AdmissionMemoryView(
            {die: dict(datapath.buffers).get("l2", 0)}, memory_access=memory,
            ws_sweep=primitive_working_set_sweep(getattr(context, "primitive_table", None)),
        )
        bus_memory = self.state.memory_access
        self.hbm_bus = None
        if context is not None and "hbm" in context.resource_names:
            hbm = context.resource_names["hbm"]
            rate = context.rates[hbm]
            self.hbm_bus = HbmBusSpec(
                die, min(rate, datapath.hbm_read_bytes_per_s / 1000 or rate),
                min(rate, datapath.hbm_write_bytes_per_s / 1000 or rate), datapath.hbm_rw_alpha,
                row_r=hbm + ":r", row_w=hbm + ":w",
                ctr_r=hbm + ":ctr_r", ctr_w=hbm + ":ctr_w",
                memory_access=bus_memory,
            )
        if live_admission and (self.hbm_bus is None or memory_view is None):
            raise ModelingError("template_live_admission_requires_context_and_view")
        self.live_state = memory_view if live_admission else None
        self.state.drain_writebacks(die)
        self.stages: list[Stage] = []
        self.accesses: dict[str, MemoryAccessPattern] = {}
        self.obligations: dict[str, TensorRangeObligation] = {}
        self.cube: Counter[str] = Counter()
        self.vector: Counter[str] = Counter()
        self.reads: list[TensorRangeObligation] = []
        self.writes: list[TensorRangeObligation] = []
        self.logical = self.copied = self.sectors = self.hits = self.misses = 0
        self.service = Fraction()
        self.metadata_ms = Fraction()
        self.launches = 0
        self.l2_reservation = None
        self.sector = memory.sector_bytes if memory is not None else AddressRunPlan().sector_bytes
        self.burst_rules = tuple(
            BurstTurnRule(0, None, rule.min_stride_bytes, rule.max_stride_bytes,
                          rule.penalty_bytes, rule.evidence_id)
            for rule in memory.burst_turn_penalty
        ) if memory is not None else ()

    def add(self, stage: Stage) -> str:
        self.stages.append(stage)
        return stage.name

    def launch(self, name="launch", *, after=()) -> str:
        stars = self.datapath.stars_spec
        latency = 0
        if stars is not None:
            latency = (stars.issue_interval_us + (0 if self.launches else stars.first_issue_us)) / 1000
        self.launches += 1
        return self.add(Stage(name, after=after, latency=latency, metadata={"stars_launch": True}))

    def metadata(self, ms, *, after=()) -> str:
        self.metadata_ms = Fraction(str(ms))
        if self.metadata_ms < 0:
            raise ModelingError("template_metadata_latency_negative")
        return self.add(Stage("aicpu_metadata", after=after, latency=float(self.metadata_ms),
                              metadata={"aicpu_metadata": True}))

    def binding_role(self, role):
        if role in _VECTOR_PIPES - {"vector"}:
            if self.context.service_rate(role) is None and role not in self.context.resource_names:
                return "vector"
        return role

    def work(self, name, role, amount, *, after=(), owner=0, dtype=None, metadata=None) -> str:
        if amount < 0:
            raise ModelingError(f"template_stage_negative_work:{name}")
        if role == "cube":
            self.cube[dtype] += amount
            physical = Work(matrix={dtype: amount})
        else:
            if role in _VECTOR_PIPES:
                self.vector[role] += amount
            physical = Work(resources={role: amount})
        if self.context is not None:
            role_for_binding = self.binding_role(role)
            if role_for_binding != role:
                physical = Work(resources={role_for_binding: amount})
            work = self.context.bind(physical)
            axis_count = self.datapath.aic_count if role in _AIC_RESOURCES else self.datapath.aiv_count
            placement = self.context.owned(owner, axis_count, (role_for_binding,)) if work else {}
            placement = {key: ids for key, ids in placement.items() if key in work}
        else:
            work = {role: amount} if amount else {}
            placement = {role: (owner,)} if work else {}
        return self.add(Stage(name, work, after=after, placement=placement, metadata=metadata or {}))

    def add_ring(self, ring, *, after=()):
        """Attach slot grants to the issuing tile and map their physical pool."""
        for stage in ring.to_stages():
            if stage.acquire or stage.free:
                acquire, free = stage.acquire, stage.free
                if self.context is not None:
                    acquire = tuple(self.context.allocate(a.name, a.pool, a.size, a.units) for a in acquire)
                    free = tuple(self.context.allocate(a.name, a.pool, a.size, a.units) for a in free)
                stage = replace(stage, acquire=acquire, free=free, after=stage.after or after)
            self.add(stage)

    def _account(self, ledger, address_plan, resident):
        self.logical += ledger.logical_tensor_bytes
        self.copied += ledger.aligned_transfer_bytes
        self.sectors += ledger.dram_sector_bytes
        self.hits += ledger.l2_hit_bytes
        self.misses += ledger.dram_miss_copy_bytes
        if ledger.dram_sector_bytes:
            if ledger.dram_bus_transaction_bytes == ledger.dram_sector_bytes:
                self.service += ledger.dram_sector_bytes
            elif self.burst_rules and not self.state._knee_curve:
                resident = resident if ledger.l2_hit_bytes else SectorCoverage()
                self.service += _rational_service(address_plan, resident, self.burst_rules)
            else:
                self.service += Fraction(ledger.dram_bus_transaction_bytes)

    def _writebacks(self, name, after):
        evictions = self.state.drain_writebacks(self.die)
        if not evictions:
            return after
        amount = sum(item.bytes for item in evictions)
        self.sectors += amount
        self.service += amount
        if self.live_state is not None:
            return after
        if self.hbm_bus is not None:
            work = {term.row_id: term.ms_per_progress for term in self.hbm_bus.terms_for("write", amount)
                    if term.ms_per_progress}
        else:
            work = {"hbm": amount}
        deliveries = ((self.hbm_bus.ctr_w, amount),) if self.hbm_bus is not None else ()
        return (self.add(Stage(f"{name}:dirty_writeback", work, after=after, coupled=len(work) > 1,
                               deliveries=deliveries)),)

    def access(
        self, name, tensor, amount, *, direction="read", pipe="mte2_aiv",
        after=(), owner=0, offset=0, runs=None, version=None, private=False,
        bypass=False, working_set=None, obligation=True,
    ) -> str:
        if not amount:
            return self.add(Stage(name, after=after))
        version = (0 if private or direction == "read" else 1) if version is None else version
        key = None if bypass else ReuseKey(self.die, tensor, version, reuse_stream=f"{tensor}:{version}")
        visibility = VisibilityContract(
            scope="die" if private else "device",
            live_out=direction == "write" and not private, private_workspace=private,
        )
        runs = tuple(runs) if runs is not None else (AddressRun(offset, 1, amount, amount),)
        plan = _access_plan(AddressRunPlan(runs, sector_bytes=self.sector), self.state.memory_access)
        pattern = MemoryAccessPattern(
            tensor, version, self.die, owner, direction, pipe, amount, plan, key,
            working_set, visibility,
        )
        self.accesses[name] = pattern
        if not private and obligation and (entry := _logical_obligation(pattern)):
            self.obligations[name] = entry
            if direction == "read":
                if key is not None:
                    pattern = replace(pattern, reuse_key=self.state.resolve_reuse_key(key))
                resident = self.state.resident_coverage_for(pattern, plan.aligned_copy_bytes())
                self.reads.extend(_split_entry_residency(entry, resident))
            else:
                self.writes.append(entry)
        return self.add(Stage(name, after=after))

    def _bind_access(self, placeholder, declared, lifetimes):
        name, after = placeholder.name, placeholder.after
        pattern = lifetimes.resolve(declared)
        plan = pattern.runs
        if self.context is not None:
            bound = self.context.bind_access(
                pattern, self.state, hbm_bus=self.hbm_bus, burst_rules=self.burst_rules, commit=True,
            )
            stage = bound.to_stage(name, after=after)
            if bound.l2_row_id not in self.context.rates and bound.l2_row_id in stage.work:
                stage = replace(stage, work={role: value for role, value in stage.work.items() if role != bound.l2_row_id})
            if self.live_state is not None:
                stage = Stage(name, after=after, program=_LiveAccess(bound, self.live_state, declared, self.obligations.get(name)))
        else:
            # Unit-rate normalization recovers native byte work independently
            # of hardware service rates. An Engine supplies real rates later.
            bus = HbmBusSpec(self.die, 1, 1, 0, row_r="hbm", row_w="hbm",
                             memory_access=self.state.memory_access)
            bound = bind_memory_access(
                pattern, self.state, hbm_bus=bus, copy_hit_bytes_per_ms=1,
                copy_miss_bytes_per_ms=1, l2_bytes_per_ms=1, pipe_row_id=pattern.pipe,
                l2_row_id="l2", burst_rules=self.burst_rules, commit=True,
            )
            stage = bound.to_stage(name, after=after)
        resident = (
            getattr(self.state, "_last_prior_coverage", SectorCoverage())
            if (bound.ledger.l2_hit_bytes and bound.ledger.dram_sector_bytes)
            else SectorCoverage()
        )
        self._account(bound.ledger, plan, resident)
        lifetimes.complete(declared.reuse_key)
        return replace(stage, after=self._writebacks(name, stage.after))

    def result(self, tier, region_id, op_ids, spec, semantic, solution, *, launches=1, comm=()):
        stages = ordered_stages(self.stages)
        if self.accesses:
            lifetimes = _ReuseLifetimes(self.state, self.accesses.values())
            self.stages = []
            for stage in stages:
                pattern = self.accesses.pop(stage.name, None)
                self.add(self._bind_access(stage, pattern, lifetimes) if pattern is not None else stage)
            stages = ordered_stages(self.stages)
        summary = FusedRegionSummary(
            tier, region_id, tuple(op_ids), tuple(replace(item, die=self.die) for item in semantic),
            tuple(sorted(self.cube.items())),
            tuple(sorted(self.vector.items())), _union_obligations(self.reads), _union_obligations(self.writes),
            self.logical, self.copied, self.sectors, self.service, self.hits, tuple(comm), launches,
            self.metadata_ms, tuple((peak.pool, peak.peak_bytes) for peak in solution.pool_peaks), spec,
            None if self.live_state is not None else float(self.service), self.misses,
        )
        program = (
            _BoundTemplateProgram(stages, hbm_bus=self.hbm_bus, live_state=self.live_state,
                                  l2_reservation=self.l2_reservation, summary=summary)
            if self.hbm_bus is not None else KernelProgram(stages)
        )
        return TemplateLoweringResult(program, summary, len(stages))


def _route_edge(datapath, edge, owners, tile, explicit_bytes):
    source, dest = owners
    counts = {"aic": datapath.aic_count, "aiv": datapath.aiv_count}
    if not (0 <= source < counts[edge.src_owner_axis] and 0 <= dest < counts[edge.dst_owner_axis]):
        raise ModelingError(f"fuse_region_owner_out_of_range:{edge.view_id}")
    if edge.route_kind == "gm_spill":
        source_pool = "l0c" if edge.src_owner_axis == "aic" else "ub"
        dest_pool = "l1" if edge.dst_owner_axis == "aic" else "ub"
        datapath.route(source_pool, "gm", edge.representation, source)
        datapath.route("gm", dest_pool, edge.representation, dest)
        return None
    src, dst = edge.route_kind.split("->")
    axes = {"l0c": "aic", "l1": "aic", "ub": "aiv"}
    if (edge.src_owner_axis, edge.dst_owner_axis) != (axes[src], axes[dst]):
        raise ModelingError(f"fuse_region_route_owner_axis_mismatch:{edge.view_id}")
    if src == "l0c" and dst == "l1" and source != dest:
        raise ModelingError(f"fuse_region_private_l1_unreachable:{edge.view_id}")
    if src == dst == "ub" and source == dest:
        # A same-owner UB view is an alias, not a physical inter-owner transfer.
        if explicit_bytes > datapath.usable_bytes("ub"):
            raise ModelingError(f"fuse_region_ub_alias_too_large:{edge.view_id}")
        return None
    if src == dst == "ub" and (source, dest) not in datapath.ub_peer_edges:
        raise ModelingError(f"fuse_region_ub_peer_unreachable:{edge.view_id}")
    if edge.src_owner_axis != edge.dst_owner_axis:
        aic, aiv = (source, dest) if edge.src_owner_axis == "aic" else (dest, source)
        if aiv not in dict(datapath.aic_aiv_pairs).get(aic, ()):
            raise ModelingError(f"fuse_region_owner_pair_unreachable:{edge.view_id}")
    return datapath.route(src, dst, edge.representation, owners, tile)


def _validate_coupling(stage, units):
    if not stage.coupled:
        return
    active = [(len(stage.placement.get(role, range(units[role]))), units[role])
              for role, demand in stage.work.items() if demand and role in units]
    if active:
        count, total = active[0]
        if any(owners * total != count * size for owners, size in active):
            raise ModelingError(f"fuse_region_coupled_placement_not_proportional:{stage.name}")


def _barrier_storage(routed, op_stages, context):
    """Find source allocations that a full-barrier consumer would otherwise hold."""
    releases, moved = {}, set()
    l0c = context.pool_names.get("l0c", "l0c") if context else "l0c"
    for _, edge, _, _, _ in routed:
        if edge.iter_kind != "full_barrier" or edge.src_owner_axis != "aic":
            continue
        acquired = {a.name for a in op_stages[edge.src_op].acquire if a.pool in {"l0c", l0c}}
        for allocation in op_stages[edge.dst_op].free:
            if allocation.name in acquired:
                releases.setdefault(edge.src_op, {})[allocation.name] = allocation
                moved.add((edge.dst_op, allocation.name))
    return releases, moved


def _route_publication(scope, i, edge):
    return f"{scope}:edge:{i}" + (":write" if edge.route_kind == "gm_spill" else "")


def compile_fuse_region(
    region_id: str, op_ids: Sequence[str], datapath: DatapathCapabilitySpec, *,
    dependencies: Mapping[str, Sequence[str]], op_stages: Mapping[str, Stage],
    edge_routes: Sequence[EdgeRouteBinding], edge_bytes: Mapping[str, int],
    live_buffers: Sequence[LiveBufferSpec] = (), tile_shape: TileShape = TileShape(1, 1, 1),
    tile_count: int = 1, reduction_class="resident_1pass", backend="dsl_mix",
    live_in_obligations: Sequence[TensorRangeObligation] = (),
    live_out_obligations: Sequence[TensorRangeObligation] = (),
    reads_by_op: Mapping[str, Sequence[TensorRangeObligation]] | None = None,
    writes_by_op: Mapping[str, Sequence[TensorRangeObligation]] | None = None,
    semantic_work: Sequence[SemanticWorkItem] = (), collective_ops: Sequence[str] = (),
    waits: Sequence[tuple[str, str]] = (),
    numerical_contracts: Mapping[str, tuple[object, object]] | None = None,
    alias_views: Mapping[str, tuple[str, int, int]] | None = None,
    edge_owners: Mapping[str, tuple[int, int]] | None = None,
    route_tiles: Mapping[str, int | tuple[int, ...]] | None = None,
    resource_units: Mapping[str, int] | None = None,
    second_pass_stages: Mapping[str, Stage] | None = None,
    l2_reservation_bytes: int = 0, uses_simt: bool = False,
    context: RecipeContext | None = None, memory_view: AdmissionMemoryView | None = None,
    die: int = 0, live_admission: bool = False,
) -> tuple[FuseRegionSpec, TemplateLoweringResult]:
    """Compile per-tile stage declarations with explicit tensor and alias facts.

    Op stages describe one tile each. Pass-two declarations name the exact
    operations repeated after the global first-pass barrier.
    """
    edges, buffers = tuple(edge_routes), tuple(live_buffers)
    if not region_id or type(tile_count) is not int or tile_count < 1 or backend not in {"catlass_evg", "dsl_mix", "dsl_vec", "tla_simt"}:
        raise ModelingError("fuse_region_geometry_invalid")
    order, parents = _region_order(
        op_ids, dependencies, collective_ops, edges, waits,
        numerical_contracts or {}, alias_views or {},
    )
    if set(order) != op_stages.keys() or any(s.previous or s.program is not None for s in op_stages.values()):
        raise ModelingError("fuse_region_per_tile_stage_required")
    second = dict(second_pass_stages or {})
    if bool(second) != (reduction_class == "row_split_2pass") or not second.keys() <= op_stages.keys():
        raise ModelingError("fuse_region_two_pass_work_required")
    if any(s.previous or s.program is not None for s in second.values()):
        raise ModelingError("fuse_region_per_tile_stage_required")
    for stage in (*op_stages.values(), *second.values()):
        if stage.after and not set(stage.after) <= set(order):
            raise ModelingError("fuse_region_stage_dependency_unknown")

    routed = []
    source_buffers = {}
    for idx, edge in enumerate(edges):
        size = edge_bytes.get(edge.view_id, 0)
        if type(size) is not int or size <= 0:
            raise ModelingError(f"fuse_region_edge_bytes_required:{edge.view_id}")
        dest_pool = edge.route_kind.split("->")[-1]
        owners = (edge_owners or {}).get(edge.view_id, (0, 0))
        route = _route_edge(datapath, edge, owners, (route_tiles or {}).get(edge.view_id, size), size)
        routed.append((idx, edge, size, owners, route))
        if route is not None or edge.route_kind == "ub->ub":
            src_pool = edge.route_kind.split("->")[0]
            key = (edge.src_op, edge.producer_version, src_pool, edge.src_owner_axis)
            source_buffers[key] = max(source_buffers.get(key, 0), size)
            if route is not None:
                buffers += (LiveBufferSpec(
                    f"edge:{idx}", dest_pool, size, 0, 1, owner_kind=edge.dst_owner_axis,
                    slots=tile_count if edge.iter_kind == "full_barrier" else edge.ring_depth,
                ),)
    buffers += tuple(LiveBufferSpec(f"producer:{op}:{version}", pool, size, 0, 1, owner_kind=axis)
                     for (op, version, pool, axis), size in source_buffers.items())
    uses_simt = uses_simt or backend == "tla_simt"
    solution = solve_template_tile(datapath, region_id, TileCandidate(
        reduction_class, tile_shape, max((e.ring_depth for e in edges), default=1),
        (datapath.aic_count, datapath.aiv_count), tile_count, ceil_div(tile_count, max(1, datapath.aic_count)),
        buffers, uses_simt=uses_simt, owner_axis="aiv" if backend in {"dsl_vec", "tla_simt"} else "aic",
    ), state=memory_view, die=die, reserved=l2_reservation_bytes)
    reads = dict(reads_by_op) if reads_by_op is not None else {order[0]: live_in_obligations}
    writes = dict(writes_by_op) if writes_by_op is not None else {order[-1]: live_out_obligations}
    if not (reads.keys() | writes.keys()) <= set(order):
        raise ModelingError("fuse_region_memory_op_outside")
    produced, external = {}, list(live_in_obligations)
    for op in order:
        external.extend(_external_reads(reads.get(op, ()), produced))
        _record_produced(produced, writes.get(op, ()))
    read_ranges = _union_obligations(external)
    write_ranges = _union_obligations((*live_out_obligations, *(entry for rows in writes.values() for entry in rows)))
    spec = FuseRegionSpec(region_id, order, read_ranges, write_ranges, edges, reduction_class, solution, backend)
    plan = Lowering(datapath, context=context, memory_view=memory_view, die=die, live_admission=live_admission)
    if l2_reservation_bytes:
        plan.state.grant_reservation(die, region_id, l2_reservation_bytes)
        plan.l2_reservation = (die, l2_reservation_bytes, datapath.usable_bytes("l2"))
    launch = plan.launch()
    initial = plan._writebacks("reservation", (launch,))
    rings = {
        i: RingBuffer(die=die, owner=owners[-1], pool=edge.route_kind.split("->")[-1],
                      allocation=f"edge:{i}", depth=edge.ring_depth,
                      slot_bytes=size if route is not None and context is not None else 0)
        for i, edge, size, owners, route in routed if edge.iter_kind != "full_barrier"
    }
    units = {**dict.fromkeys(_AIC_RESOURCES, datapath.aic_count),
             **dict.fromkeys(_AIV_RESOURCES, datapath.aiv_count), **(resource_units or {})}
    if context is not None:
        units.update({context.resource_names.get(key, key): value for key, value in context.resource_units.items()})
    plan.reads.extend(read_ranges)
    plan.writes.extend(write_ranges)
    releases, moved = _barrier_storage(routed, op_stages, context)
    passes = (op_stages, second) if second else (op_stages,)
    for pass_idx, declarations in enumerate(passes):
        completions = []
        for tile in range(tile_count):
            scope = f"p{pass_idx}:t{tile}"

            def scoped(allocations):
                pools = context.pool_names if context else {}
                return tuple(replace(a, name=f"{scope}:{a.name}", pool=pools.get(a.pool, a.pool))
                             for a in allocations)

            if not pass_idx:
                for source, allocations in releases.items():
                    publications = tuple(_route_publication(scope, i, edge) for i, edge, _, _, _ in routed
                                         if edge.src_op == source and edge.iter_kind != "chunk_carried")
                    plan.add(Stage(f"{scope}:{source}:released", after=publications, free=scoped(allocations.values())))
            final_names = {op: f"{scope}:{op}:done" for op in declarations}
            for op in order:
                if op not in declarations:
                    continue
                stage = declarations[op]
                _validate_coupling(stage, units)
                after = [*initial, *(final_names[p] for p in sorted(parents[op] | set(stage.after)) if p in declarations)]
                for i, edge, size, owners, route in routed:
                    if pass_idx or edge.dst_op != op:
                        continue
                    src_tile = tile - 1 if edge.iter_kind == "chunk_carried" else tile
                    if src_tile < 0:
                        continue
                    src = f"p{pass_idx}:t{src_tile}:{edge.src_op}:done"
                    generation = src_tile
                    credit = ()
                    if i in rings:
                        ring = rings[i]
                        credit = (ring_credit(ring, generation),)
                    name = f"{scope}:edge:{i}"
                    dependencies_for_route = (src, *credit)
                    if edge.route_kind == "gm_spill":
                        tensor = f"{region_id}:edge:{i}"
                        pipe = "fixpipe" if edge.src_owner_axis == "aic" else "mte3"
                        spill = plan.access(name + ":write", tensor, size, direction="write", pipe=pipe,
                                            after=dependencies_for_route, private=True, owner=owners[0], offset=src_tile * size)
                        dep = spill
                    elif route is not None:
                        owner = owners[0] if route.path.executor == edge.src_owner_axis else owners[1]
                        dep = plan.work(name, route.path.unit, size, after=dependencies_for_route, owner=owner)
                    else:
                        dep = plan.add(Stage(name, after=dependencies_for_route))
                    if edge.iter_kind == "full_barrier":
                        barrier_deps = tuple(
                            f"p{pass_idx}:t{t}:{edge.src_op}:released" if edge.src_op in releases
                            else _route_publication(f"p{pass_idx}:t{t}", i, edge) for t in range(tile_count)
                        )
                        dep = plan.add(Stage(name + ":barrier", after=barrier_deps))
                    if edge.route_kind == "gm_spill":
                        dep = plan.access(name + ":read", tensor, size, pipe=f"mte2_{edge.dst_owner_axis}",
                                          after=(dep,), private=True, owner=owners[1], offset=src_tile * size)
                    if i in rings:
                        event = lower_generation(
                            rings[i], generation, "route", ("op",), producer_extra_after=(dep,),
                            consumer_extra_after={"op": (final_names[op],)},
                        )
                        dep = event.publish
                    after.append(dep)
                for index, obligation in enumerate(reads.get(op, ())):
                    if obligation.initial_residency == "onchip_resident":
                        continue
                    after.append(_obligation_access(plan, scope, op, index, obligation, tile, tile_count, "read", tuple(after)))
                stage = _bind_declared_work(plan, stage, semantic_work)
                freed = tuple(a for a in stage.free if pass_idx or (op, a.name) not in moved)
                run = plan.add(replace(stage, name=f"{scope}:{op}", after=tuple(dict.fromkeys(after)),
                                       acquire=scoped(stage.acquire), free=scoped(freed)))
                done = [run]
                if pass_idx == len(passes) - 1:
                    for index, obligation in enumerate(writes.get(op, ())):
                        done.append(_obligation_access(plan, scope, op, index, obligation, tile, tile_count, "write", (run,)))
                completions.append(plan.add(Stage(final_names[op], after=tuple(done))))
        if len(passes) > 1 and pass_idx == 0:
            initial = (plan.add(Stage("first_pass_complete", after=tuple(completions))),)
    for ring in rings.values():
        plan.add_ring(ring, after=(launch,))
    return spec, plan.result("K3", region_id, order, spec, semantic_work, solution)


def _bind_declared_work(plan, stage, semantic):
    context = plan.context
    resource_names = context.resource_names if context else {}
    cube_row = resource_names.get("cube", "cube")
    work, placement, caps, units = Counter(), {}, {}, {}
    physical_pipes = {resource_names.get(pipe, pipe): pipe for pipe in _VECTOR_PIPES}
    for role, amount in stage.work.items():
        if not amount:
            continue
        row, unit = role, stage.work_units.get(role)
        if role in {"cube", "matrix", cube_row}:
            raw = stage.metadata.get("issued_cube_flops")
            if raw is None:
                if unit == "peak_ms" or (context and role == cube_row and role not in {"cube", "matrix"}):
                    raise ModelingError("fuse_region_bound_cube_stage_requires_issued_flops")
                raw = amount
            if int(raw) != raw or raw <= 0:
                raise ModelingError("fuse_region_cube_work_requires_integer_flops")
            dtypes = {item.dtype for item in semantic if "cube" in item.admitted_engines}
            fallback = next(iter(dtypes)) if len(dtypes) == 1 else getattr(context.op if context else None, "dtype", "bf16")
            dtype = stage.metadata.get("cube_dtype", fallback)
            if not isinstance(dtype, str) or not dtype:
                raise ModelingError("fuse_region_cube_dtype_invalid")
            plan.cube[dtype] += int(raw)
            if context:
                row, bound = next(iter(context.bind(Work(matrix={dtype: raw})).items()))
            else:
                bound = 1000 * raw / plan.datapath.cube_peak(dtype)
            unit = "peak_ms"
        else:
            pipe = physical_pipes.get(role)
            if pipe is not None:
                plan.vector[pipe] += stage.metadata.get(f"issued_{pipe}_ops", amount)
            bound = amount
            if context and (role not in context.rates or context.service_rate(role) is not None):
                role_for_binding = plan.binding_role(role)
                bound_work = context.bind(Work(resources={role_for_binding: amount}))
                if not bound_work:
                    continue
                row, bound = next(iter(bound_work.items()))
                unit = context.work_units.get(context.physical_role(role_for_binding), unit)
            if unit is None:
                unit = "ops" if pipe is not None else NATIVE_WORK_UNITS.get(role)
        work[row] += bound
        if role in stage.placement:
            owners = stage.placement[role]
            if row in placement and placement[row] != owners:
                raise ModelingError(f"fuse_region_conflicting_placement:{stage.name}:{row}")
            placement[row] = owners
        if role in stage.caps:
            caps[row] = stage.caps[role] * bound / amount
        if unit is not None:
            units[row] = unit
    return replace(stage, work=work, placement=placement, caps=caps, work_units=units)


def _obligation_access(plan, scope, op, index, entry, tile, tiles, direction, after):
    if entry.die != plan.die:
        raise ModelingError(f"fuse_region_cross_die_access_requires_collective:{entry.allocation_id}")
    runs = tuple(segment for run in _obligation_runs(entry) for segment in _tile_address_run(run, tile, tiles))
    return plan.access(
        f"{scope}:{op}:{direction}:{index}", entry.allocation_id, sum(run.rows * run.row_bytes for run in runs),
        direction=direction, pipe="mte2_aiv" if direction == "read" else "mte3",
        after=after, runs=runs, version=entry.tensor_version,
        private=entry.required_final_visibility in {"owner", "die"} and direction == "write",
        obligation=False,
    )


def _tile_address_run(run, tile, tiles):
    """Partition unpadded row payload into tiles, retaining all interior rows."""
    width, stride = run.row_bytes, run.stride_bytes
    if stride <= width:
        width = stride = _run_end(run) - run.base_byte
        rows = 1
    else:
        rows = run.rows
    start = rows * width * tile // tiles
    stop = rows * width * (tile + 1) // tiles
    if stop <= start:
        return
    first, skip = divmod(start, width)
    if skip:
        size = min(width - skip, stop - start)
        yield AddressRun(run.base_byte + first * stride + skip, 1, size, size)
        first += 1
        if start + size == stop:
            return
    last, tail = divmod(stop, width)
    if last > first:
        yield AddressRun(run.base_byte + first * stride, last - first, width, stride)
    if tail:
        yield AddressRun(run.base_byte + last * stride, 1, tail, tail)
