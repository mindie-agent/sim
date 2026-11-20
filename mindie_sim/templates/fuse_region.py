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
from dataclasses import dataclass, replace
from fractions import Fraction
from graphlib import CycleError, TopologicalSorter
from types import SimpleNamespace

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, BoundAccess, BurstTurnRule, HbmBusSpec,
    MemoryAccessPattern, ReuseKey, VisibilityContract, bind_memory_access,
    _merge_byte_intervals, _subtract_byte_intervals,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    AIC_POOLS, LiveBufferSpec, TileCandidate, TilePipelineSpec, TileShape, solve,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec

from .contracts import (
    EdgeRouteBinding, FuseRegionSpec, FusedRegionSummary, SemanticWorkItem,
    TemplateLoweringResult, TensorRangeObligation, _union_obligations, ceil_div,
)

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


def _topological(parents, message):
    try:
        return tuple(TopologicalSorter(parents).static_order())
    except CycleError as exc:
        raise ModelingError(message) from exc


def ordered_stages(stages: Sequence[Stage]) -> tuple[Stage, ...]:
    by_name = {stage.name: stage for stage in stages}
    if len(by_name) != len(stages):
        raise ModelingError("template_stage_name_duplicate")
    if any(dep not in by_name for stage in stages for dep in stage.after):
        raise ModelingError("template_stage_dependency_unknown")
    order = _topological({stage.name: stage.after for stage in stages}, "template_stage_dependency_cycle")
    return tuple(by_name[name] for name in order)


@dataclass(frozen=True)
class _BoundTemplateProgram(KernelProgram):
    """Declare directional admission rows before submitting the bound DAG."""

    hbm_bus: HbmBusSpec | None = None
    live_state: AdmissionMemoryView | None = None
    l2_reservation: tuple[int, int, int] | None = None

    def start(self, engine, name, ready, *, owner=""):
        bus = self.hbm_bus
        if bus is not None:
            present = (bus.row_r in engine.resources, bus.row_w in engine.resources)
            if not any(present):
                engine.add_hbm_bus(bus)
            elif not all(present):
                raise ModelingError("template_directional_hbm_rows_incomplete")
        if self.l2_reservation is None or self.live_state is None:
            return super().start(engine, name, ready, owner=owner)
        die, reserved, capacity = self.l2_reservation
        state = self.live_state
        key = name + ":l2_reservation"

        def grant(_event):
            if state.reserved_bytes(die) + reserved > capacity:
                raise ModelingError("fuse_region_live_l2_reservation_overflow")
            state.grant_reservation(die, key, reserved)

        acquired = engine.submit(key, after=(ready,), owner=owner, on_start=grant)
        done = super().start(engine, name, acquired, owner=owner)
        engine.on_complete(done, lambda _event: state.release_reservation(die, key))
        return done


@dataclass(frozen=True, slots=True)
class _LiveAccess:
    bound: BoundAccess
    state: AdmissionMemoryView
    pattern: MemoryAccessPattern

    def start(self, engine, name, ready, *, owner=""):
        return engine.submit_bound_access(
            name, self.bound, after=(ready,), owner=owner,
            memory_state=self.state, pattern=self.pattern,
        )


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
        self.hbm_bus = None
        if context is not None and "hbm" in context.resource_names:
            hbm = context.resource_names["hbm"]
            rate = context.rates[hbm]
            self.hbm_bus = HbmBusSpec(
                die, datapath.hbm_read_bytes_per_s / 1000 or rate,
                datapath.hbm_write_bytes_per_s / 1000 or rate, datapath.hbm_rw_alpha,
                row_r=hbm + ":r", row_w=hbm + ":w",
                ctr_r=hbm + ":ctr_r", ctr_w=hbm + ":ctr_w",
            )
        if live_admission and (context is None or memory_view is None):
            raise ModelingError("template_live_admission_requires_context_and_view")
        self.live_state = memory_view if live_admission else None
        self.state = deepcopy(memory_view) if memory_view is not None else AdmissionMemoryView({
            die: dict(datapath.buffers).get("l2", 0),
        })
        self.state.drain_writebacks(die)
        self.stages: list[Stage] = []
        self.accesses: dict[str, MemoryAccessPattern] = {}
        self.cube: Counter[str] = Counter()
        self.vector: Counter[str] = Counter()
        self.reads: list[TensorRangeObligation] = []
        self.writes: list[TensorRangeObligation] = []
        self.logical = self.copied = self.sectors = self.hits = self.misses = 0
        self.service = Fraction()
        self.metadata_ms = Fraction()
        self.launches = 0
        self.l2_reservation = None
        memory = datapath.memory_access
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
            role_for_binding = role
            if role in {"vec_alu", "vec_sfu", "vec_reduce", "sort_topk"}:
                if self.context.service_rate(role) is None and role not in self.context.resource_names:
                    role_for_binding = "vector"
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
            self.service += _rational_service(
                address_plan, resident if ledger.l2_hit_bytes else (), self.burst_rules,
            )

    def _writebacks(self, name, after):
        evictions = self.state.drain_writebacks(self.die)
        if not evictions:
            return after
        amount = sum(item.bytes for item in evictions)
        self.sectors += amount
        self.service += amount
        if self.hbm_bus is not None:
            work = {term.row_id: term.ms_per_progress for term in self.hbm_bus.terms_for("write", amount)
                    if term.ms_per_progress}
        else:
            work = {"hbm": amount}
        return (self.add(Stage(f"{name}:dirty_writeback", work, after=after, coupled=len(work) > 1)),)

    def access(
        self, name, tensor, amount, *, direction="read", pipe="mte2_aiv",
        after=(), owner=0, offset=0, runs=None, version=None, private=False,
        bypass=False, working_set=None, obligation=True,
    ) -> str:
        if not amount:
            return self.add(Stage(name, after=after))
        version = (0 if private or direction == "read" else 1) if version is None else version
        key = None if bypass else ReuseKey(self.die, tensor, version)
        visibility = VisibilityContract(
            scope="die" if private else "device",
            live_out=direction == "write" and not private, private_workspace=private,
        )
        plan = AddressRunPlan(
            tuple(runs) if runs is not None else (AddressRun(offset, 1, amount, amount),),
            sector_bytes=self.sector,
        )
        pattern = MemoryAccessPattern(
            tensor, version, self.die, owner, direction, pipe, amount, plan, key,
            working_set, visibility,
        )
        self.accesses[name] = pattern
        if not private and obligation:
            ranges = plan.sector_intervals()
            if ranges:
                entry = TensorRangeObligation(
                    self.die, tensor, version, ranges, "hbm_cold", visibility.scope,
                )
                (self.reads if direction == "read" else self.writes).append(entry)
        return self.add(Stage(name, after=after))

    def _bind_access(self, placeholder, pattern):
        name, after = placeholder.name, placeholder.after
        pipe = pattern.pipe
        plan = pattern.runs
        resident = self.state.resident_intervals_for(pattern, plan.aligned_copy_bytes())
        if self.context is not None:
            preview = (
                self.context.bind_access(pattern, self.state, hbm_bus=self.hbm_bus, burst_rules=self.burst_rules)
                if self.live_state is not None else None
            )
            bound = self.context.bind_access(
                pattern, self.state, hbm_bus=self.hbm_bus, burst_rules=self.burst_rules, commit=True,
            )
            stage = bound.to_stage(name, after=after)
            if bound.l2_row_id not in self.context.rates and bound.l2_row_id in stage.work:
                stage = replace(stage, work={role: value for role, value in stage.work.items() if role != bound.l2_row_id})
            if preview is not None:
                stage = Stage(name, after=after, program=_LiveAccess(preview, self.live_state, pattern))
        else:
            # Unit-rate normalization recovers native byte work independently
            # of hardware service rates. An Engine supplies real rates later.
            bus = HbmBusSpec(self.die, 1, 1, 0, row_r="hbm", row_w="hbm")
            bound = bind_memory_access(
                pattern, self.state, hbm_bus=bus, copy_hit_bytes_per_ms=1,
                copy_miss_bytes_per_ms=1, l2_bytes_per_ms=1, pipe_row_id=pipe,
                l2_row_id="l2", burst_rules=self.burst_rules, commit=True,
            )
            stage = bound.to_stage(name, after=after)
        self._account(bound.ledger, plan, resident)
        return replace(stage, after=self._writebacks(name, stage.after))

    def result(self, tier, region_id, op_ids, spec, semantic, solution, *, launches=1, comm=()):
        stages = ordered_stages(self.stages)
        if self.accesses:
            self.stages = []
            for stage in stages:
                pattern = self.accesses.pop(stage.name, None)
                self.add(self._bind_access(stage, pattern) if pattern is not None else stage)
            stages = ordered_stages(self.stages)
        summary = FusedRegionSummary(
            tier, region_id, tuple(op_ids), tuple(semantic), tuple(sorted(self.cube.items())),
            tuple(sorted(self.vector.items())), _union_obligations(self.reads), _union_obligations(self.writes),
            self.logical, self.copied, self.sectors, self.service, self.hits, tuple(comm), launches,
            self.metadata_ms, tuple((peak.pool, peak.peak_bytes) for peak in solution.pool_peaks), spec,
        )
        program = (
            _BoundTemplateProgram(stages, hbm_bus=self.hbm_bus, live_state=self.live_state,
                                  l2_reservation=self.l2_reservation)
            if self.hbm_bus is not None else KernelProgram(stages)
        )
        return TemplateLoweringResult(program, summary, len(stages), self.hits, self.misses)


def _rational_service(plan, resident, rules):
    """Use the runtime's sector geometry with rational turnaround penalties."""
    service = Fraction()
    aliases = set()
    for run in plan.runs:
        if not run.rows or not run.row_bytes:
            continue
        group = run.proven_alias_group
        if group is not None:
            if group in aliases:
                continue
            aliases.add(group)
        if not resident:
            segments = ((run.sector_union_bytes(plan.sector_bytes), run.row_bytes, run.stride_bytes),)
        else:
            intervals = AddressRunPlan((run,), sector_bytes=plan.sector_bytes).sector_intervals()
            missing = _subtract_byte_intervals(intervals, resident)
            segments = ((hi - lo, hi - lo, (
                0 if run.stride_bytes == 0 else run.stride_bytes if run.stride_bytes > run.row_bytes else hi - lo
            )) for lo, hi in missing)
            resident = _merge_byte_intervals((*resident, *missing))
        for sectors, rows, stride in segments:
            penalty = sum((Fraction(str(rule.penalty_bytes)) for rule in rules
                           if rule.matches(rows, stride)), Fraction())
            service += sectors + sectors * penalty / rows
    if service and plan.scale_bytes:
        service += ceil_div(plan.scale_bytes, plan.sector_bytes) * plan.sector_bytes
    return service


def _region_order(op_ids, dependencies, barriers, edge_routes, waits, numerics, views):
    selected = set(op_ids)
    if not selected or len(selected) != len(op_ids) or not selected <= dependencies.keys():
        raise ModelingError("fuse_region_op_ids_invalid")
    if selected.intersection(barriers):
        raise ModelingError("fuse_region_collective_barrier")
    if any(parent not in dependencies for parents in dependencies.values() for parent in parents):
        raise ModelingError("fuse_region_graph_parent_unknown")
    order = _topological(dependencies, "fuse_region_graph_cycle")
    escaped = set()
    for name in order:
        if name in selected:
            if escaped.intersection(dependencies[name]):
                raise ModelingError(f"fuse_region_not_convex:{name}")
        elif (selected | escaped).intersection(dependencies[name]):
            escaped.add(name)
    internal = {name: set(dependencies[name]) & selected for name in selected}
    edge_keys = set()
    for edge in edge_routes:
        if edge.dst_op not in selected or edge.src_op not in internal[edge.dst_op]:
            raise ModelingError(f"fuse_region_edge_not_internal:{edge.view_id}")
        key = (edge.src_op, edge.producer_version, edge.view_id, edge.dst_op, edge.iter_kind)
        if key in edge_keys:
            raise ModelingError(f"fuse_region_duplicate_edge:{edge.view_id}")
        edge_keys.add(key)
    if {(edge.src_op, edge.dst_op) for edge in edge_routes} != {
        (producer, consumer) for consumer, producers in internal.items() for producer in producers
    }:
        raise ModelingError("fuse_region_internal_route_missing")
    for publisher, consumer in waits:
        if publisher not in selected or consumer not in selected:
            raise ModelingError("fuse_region_wait_endpoint_outside")
        internal[consumer].add(publisher)
    _topological(internal, "fuse_region_credit_publish_deadlock")
    if not numerics.keys() <= selected or any(before != after for before, after in numerics.values()):
        raise ModelingError("fuse_region_numerical_equivalence_failed")

    def precedes(earlier, later):
        pending = list(internal[later])
        visited = set()
        while pending:
            predecessor = pending.pop()
            if predecessor == earlier:
                return True
            if predecessor not in visited:
                visited.add(predecessor)
                pending.extend(internal[predecessor])
        return False

    for mutation in (edge for edge in edge_routes if edge.in_place_mutates):
        for reader in edge_routes:
            if reader is mutation:
                continue
            mut_view, read_view = views.get(mutation.view_id), views.get(reader.view_id)
            if mut_view is not None and read_view is not None:
                m_allocation, m_lo, m_hi = mut_view
                r_allocation, r_lo, r_hi = read_view
                aliases = m_allocation == r_allocation and max(m_lo, r_lo) < min(m_hi, r_hi)
            else:
                aliases = mutation.view_id == reader.view_id or mutation.src_op == reader.src_op
            if (aliases and reader.producer_version <= mutation.producer_version
                    and not precedes(reader.dst_op, mutation.dst_op)):
                raise ModelingError(f"fuse_region_in_place_unread_alias:{mutation.view_id}:{reader.dst_op}")
    carried = {(edge.src_op, edge.dst_op) for edge in edge_routes if edge.iter_kind == "chunk_carried"}
    same_tile = {(edge.src_op, edge.dst_op) for edge in edge_routes if edge.iter_kind != "chunk_carried"} | set(waits)
    for src, dst in carried - same_tile:
        internal[dst].discard(src)
    return tuple(name for name in order if name in selected), internal


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
    read_ranges = _union_obligations((*live_in_obligations, *(entry for rows in reads.values() for entry in rows)))
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
    passes = (op_stages, second) if second else (op_stages,)
    for pass_idx, declarations in enumerate(passes):
        completions = []
        for tile in range(tile_count):
            scope = f"p{pass_idx}:t{tile}"
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
                    if edge.iter_kind == "full_barrier":
                        barrier_deps = tuple(f"p{pass_idx}:t{t}:{edge.src_op}:done" for t in range(tile_count))
                        src = plan.add(Stage(f"{scope}:edge:{i}:barrier", after=barrier_deps))
                    credit = ()
                    if i in rings:
                        ring = rings[i]
                        credit = (ring.initial_credit(ring.slot(generation)) if generation < ring.depth
                                  else ring.last_use(generation - ring.depth),)
                    name = f"{scope}:edge:{i}"
                    dependencies_for_route = (src, *credit)
                    if edge.route_kind == "gm_spill":
                        tensor = f"{region_id}:edge:{i}"
                        pipe = "fixpipe" if edge.src_owner_axis == "aic" else "mte3"
                        spill = plan.access(name + ":write", tensor, size, direction="write", pipe=pipe,
                                            after=dependencies_for_route, private=True, owner=owners[0], offset=src_tile * size)
                        dep = plan.access(name + ":read", tensor, size, pipe=f"mte2_{edge.dst_owner_axis}",
                                          after=(spill,), private=True, owner=owners[1], offset=src_tile * size)
                    elif route is not None:
                        owner = owners[0] if route.path.executor == edge.src_owner_axis else owners[1]
                        dep = plan.work(name, route.path.unit, size, after=dependencies_for_route, owner=owner)
                    else:
                        dep = plan.add(Stage(name, after=dependencies_for_route))
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
                _account_declared_work(plan, stage, semantic_work)
                run = plan.add(replace(stage, name=f"{scope}:{op}", after=tuple(dict.fromkeys(after))))
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


def _account_declared_work(plan, stage, semantic):
    context = plan.context
    cube_role = context.resource_names.get("cube", "cube") if context is not None else "cube"
    amount = stage.metadata.get("issued_cube_flops", stage.work.get(cube_role, 0))
    if amount:
        if context is not None and "issued_cube_flops" not in stage.metadata:
            raise ModelingError("fuse_region_bound_cube_stage_requires_issued_flops")
        if int(amount) != amount or amount < 0:
            raise ModelingError("fuse_region_cube_work_requires_integer_flops")
        dtypes = {item.dtype for item in semantic if "cube" in item.admitted_engines}
        default_dtype = next(iter(dtypes)) if len(dtypes) == 1 else getattr(
            context.op if context is not None else None, "dtype", "unspecified",
        )
        dtype = stage.metadata.get("cube_dtype", default_dtype)
        if not isinstance(dtype, str) or not dtype:
            raise ModelingError("fuse_region_cube_dtype_invalid")
        plan.cube[dtype] += int(amount)
    for pipe in _VECTOR_PIPES:
        role = context.resource_names.get(pipe, pipe) if context is not None else pipe
        ops = stage.metadata.get(f"issued_{pipe}_ops", stage.work.get(role, 0))
        if ops:
            plan.vector[pipe] += ops


def _obligation_access(plan, scope, op, index, entry, tile, tiles, direction, after):
    if entry.die != plan.die:
        raise ModelingError(f"fuse_region_cross_die_access_requires_collective:{entry.allocation_id}")
    runs = []
    for low, high in entry.ranges:
        start = low + (high - low) * tile // tiles
        end = low + (high - low) * (tile + 1) // tiles
        if end > start:
            runs.append(AddressRun(start, 1, end - start, end - start))
    return plan.access(
        f"{scope}:{op}:{direction}:{index}", entry.allocation_id, sum(run.row_bytes for run in runs),
        direction=direction, pipe="mte2_aiv" if direction == "read" else "mte3",
        after=after, runs=runs, version=entry.tensor_version,
        private=entry.required_final_visibility in {"owner", "die"} and direction == "write",
        obligation=False,
    )
