"""Compile physically admitted tile-pipeline regions onto the simulator engine.

[Role] Check region legality, resolve routes, solve liveness and emit work.
[Invariants] Physical pool owners, directional placement, and byte ledgers agree.
[Timeouts & Locks] Finite caller-declared schedules. No locks or remote calls.
[Side-Effect Policy] Lowering uses a private admission snapshot. Execution may
  update an explicitly supplied live admission view.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from fractions import Fraction
from types import SimpleNamespace
from typing import Any, Literal

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.physical_recipe_base import (
    IO_HBM_CATEGORY, Phase, _resource_categories, io_weight, phased_program, tiled_phased_program,
)
from mindie_sim.prediction.program.binding import RecipeContext
from mindie_sim.prediction.program.stages import KernelProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine
from mindie_sim.prediction.runtime.engine_types import Resource
from mindie_sim.prediction.runtime.l2_residency import _access_plan, cache_policy_for_arch, primitive_working_set_sweep
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, BoundAccess, BurstTurnRule, ByteLedger,
    ConstraintTerm, CounterTerm, HbmBusSpec,
    MemoryAccessPattern, ReuseKey, SectorCoverage, VisibilityContract, bind_memory_access,
)
from mindie_sim.prediction.runtime.ring_buffer import RingBuffer, lower_generation
from mindie_sim.prediction.runtime.tile_solver import (
    AIC_POOLS, LiveBufferSpec, TileCandidate, TileInfeasibleError, TilePipelineSpec, TileShape, solve,
)
from mindie_sim.primitives.loader import resolve_dispatch_envelope, resolve_sim_rate_overlay
from mindie_sim.primitives.tilelang import (
    TILELANG_DMA_STAGE_DOUBLE,
    TILELANG_OPS_PER_GOP,
    TileLangCalibration,
    load_tilelang_calibration,
    try_load_tilelang_calibration,
)
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import DatapathCapabilitySpec, datapath_for_die, ub_peer_edges_for_rank
from mindie_sim.schema.hardware import NATIVE_WORK_UNITS, Device, Hardware
from mindie_sim.schema.operators import OpSpec

from .contracts import (
    FP32_BYTES,
    TILELANG_BACKENDS,
    BindingState,
    EdgeRouteBinding, FormulaKernelEvaluation, FormulaKernelSpec, FuseRegionSpec, FusedRegionSummary,
    PipeStageSpec, SemanticWorkItem, TemplateLoweringResult, TensorRangeObligation, TileLangLoweringProfile,
    _derive_pipe_stages_from_equations, _derive_pipe_stages_from_ops, _external_reads,
    _infer_op_symbolic_kind, _logical_obligation, _obligation_runs,
    _record_produced, _region_order, _run_end, _split_entry_residency, _topological,
    _union_obligations, ceil_div, formula_elem_bytes,
)
from .fuse_region_live import _BoundTemplateProgram, _LiveAccess, _ReuseLifetimes, _rational_service

_VECTOR_PIPES = {"vector", "vec_alu", "vec_sfu", "vec_reduce", "aiv_simt", "sort_topk"}
_AIC_RESOURCES = {"cube", "fixpipe", "mte2_aic", "mte1"}
_AIV_RESOURCES = _VECTOR_PIPES | {"mte2_aiv", "mte3"}
_VALID_FUSE_BACKENDS = frozenset({"catlass_evg", "dsl_mix", "dsl_vec", "tla_simt", *TILELANG_BACKENDS})
_TILELANG_DISPATCH_MODES = frozenset({"aclgraph", "eager_prealloc", "eager_callee_alloc"})
_TILELANG_DMA_PATTERNS = frozenset({"rw11", "rw21"})
_TILELANG_CUBE_EPILOGUES = frozenset({"fixpipe", "dual_copy", "atomic_add"})
_OPS_PER_MS_PER_GOP = TILELANG_OPS_PER_GOP / 1000.0


def resolve_tilelang_profile(
    datapath: DatapathCapabilitySpec,
    *,
    backend: str = "TileLang",
    uses_simt: bool = False,
    stages: int = TILELANG_DMA_STAGE_DOUBLE,
    dma_pattern: str = "rw11",
    dispatch_mode: str = "aclgraph",
    vector_mode: str | None = None,
    cube_epilogue: str = "fixpipe",
    blockscaled: bool = False,
    hf32: bool = False,
    calibration: TileLangCalibration | None = None,
) -> tuple[TileLangLoweringProfile, TileLangCalibration]:
    """Resolve explicit ``tilelang_simt`` / ``tilelang_simd`` lowering profile from sealed data."""
    if backend not in TILELANG_BACKENDS:
        raise ModelingError(f"tilelang_backend_invalid:{backend}")
    if backend == "tilelang_simd" and uses_simt:
        raise ModelingError("tilelang_simd_incompatible_with_uses_simt")
    eff_simt = True if backend == "tilelang_simt" else (False if backend == "tilelang_simd" else bool(uses_simt))
    api_tier = "tilelang_simt" if eff_simt else "tilelang_simd"
    if dispatch_mode not in _TILELANG_DISPATCH_MODES:
        raise ModelingError(f"tilelang_dispatch_mode_invalid:{dispatch_mode}")
    if dma_pattern not in _TILELANG_DMA_PATTERNS:
        raise ModelingError(f"tilelang_dma_pattern_invalid:{dma_pattern}")
    if cube_epilogue not in _TILELANG_CUBE_EPILOGUES:
        raise ModelingError(f"tilelang_cube_epilogue_invalid:{cube_epilogue}")
    cal = calibration if calibration is not None else load_tilelang_calibration(str(datapath.arch_id))
    dp_ub = int(dict(datapath.buffers).get("ub", 0))
    simd_ceiling = min(dp_ub, cal.simd_usable_ub_bytes) if dp_ub > 0 else 0
    if eff_simt:
        dp_simt_ub = max(0, simd_ceiling - int(datapath.simt_ub_carveout_bytes))
        ub_ceiling = min(dp_simt_ub, cal.simt_usable_ub_bytes)
    else:
        ub_ceiling = simd_ceiling
    eff_vec_mode = vector_mode or ("ub_fma" if eff_simt else "ub_stream")
    profile = TileLangLoweringProfile(
        api_tier=api_tier,
        uses_simt=eff_simt,
        ub_ceiling_bytes=ub_ceiling,
        simd_ub_ceiling_bytes=simd_ceiling,
        pipelined_stages=int(stages),
        dma_pattern=dma_pattern,
        dma_attainment=cal.dma_attainment(int(stages), uses_simt=eff_simt, rw_pattern=dma_pattern),
        dispatch_mode=dispatch_mode,
        dispatch_us=cal.dispatch_floor_us(
            aclgraph_captured=(dispatch_mode == "aclgraph"),
            callee_alloc=(dispatch_mode == "eager_callee_alloc"),
        ),
        vector_mode=eff_vec_mode,
        vector_gops_per_aiv=cal.vector_gops_per_aiv(uses_simt=eff_simt, op_kind=eff_vec_mode),
        cube_epilogue=cube_epilogue,
        blockscaled=bool(blockscaled),
        hf32=bool(hf32),
        calibration_sha256=cal.envelope_sha256,
    )
    return profile, cal


def validate_live_buffer_owners(buffers: Sequence[LiveBufferSpec]) -> None:
    for buffer in buffers:
        allowed = (
            {"aiv"} if buffer.pool == "ub" else {"aic"} if buffer.pool in AIC_POOLS
            else {"aic", "aiv", "die"} if buffer.pool == "l2" else set()
        )
        if buffer.owner_kind not in allowed:
            raise ModelingError(f"live_buffer_pool_owner_mismatch:{buffer.name}:{buffer.pool}:{buffer.owner_kind}")


def solve_template_tile(
    datapath, family, candidate, *, state=None, die=0, reserved=0,
    tilelang_profile: TileLangLoweringProfile | None = None,
):
    validate_live_buffer_owners(candidate.buffers)
    if candidate.uses_simt and datapath.simt_ub_carveout_bytes <= 0:
        raise ModelingError("template_simt_carveout_required")
    existing = state.reserved_bytes(die) if state is not None else 0
    capacity = dict(datapath.buffers)
    simt_carveout = datapath.simt_ub_carveout_bytes if candidate.uses_simt else 0
    if tilelang_profile is not None:
        capacity["ub"] = tilelang_profile.simd_ub_ceiling_bytes
        if candidate.uses_simt:
            simt_carveout = max(
                int(datapath.simt_ub_carveout_bytes),
                max(0, capacity["ub"] - tilelang_profile.ub_ceiling_bytes),
            )
    if reserved < 0 or reserved + existing > capacity.get("l2", 0):
        raise ModelingError(f"template_l2_reservation_overflow:{family}")
    streams = max(1, state.active_stream_count(die)) if state is not None else 1
    options = dict(l2_granule_bytes=state.granule_bytes) if state is not None else {}
    pipeline = TilePipelineSpec(
        family, (candidate,),
        simt_ub_carveout_bytes=simt_carveout,
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


def _scaled_copy_rates(context: RecipeContext, pipe: str, attainment: float) -> tuple[float | None, float | None]:
    if attainment >= 1.0:
        return None, None
    pipe_role = pipe
    if pipe_role not in context.resource_names and pipe_role.startswith("mte2"):
        pipe_role = "mte2"
    if pipe_role not in context.resource_names:
        return None, None
    pipe_row = context.resource_names[pipe_role]
    pipe_units = context._units_for(pipe, pipe_row, context._units_for(pipe_role, pipe_row, 1))
    pipe_rate = float(context.rates[pipe_row]) / float(pipe_units)
    primitive = context._datacopy_cell()
    hit_rate = context._copy_rate(pipe, "hit", pipe_units, pipe_rate, primitive)
    miss_rate = context._copy_rate(pipe, "miss", pipe_units, pipe_rate, primitive)
    return hit_rate * attainment, miss_rate * attainment


class Lowering:
    """Shared stage and byte ledger construction for the catalog lowerers."""

    def __init__(
        self, datapath, *, context: RecipeContext | None = None,
        memory_view: AdmissionMemoryView | None = None, die: int = 0,
        live_admission: bool = False,
        tilelang_profile: TileLangLoweringProfile | None = None,
        tilelang_calibration: TileLangCalibration | None = None,
    ):
        self.datapath = datapath
        self.context = context
        self.die = die
        self.tilelang_profile = tilelang_profile
        self.tilelang_calibration = tilelang_calibration
        dma_eta = tilelang_profile.dma_attainment if tilelang_profile is not None else 1.0
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
                die, min(rate, datapath.hbm_read_bytes_per_s / 1000 or rate) * dma_eta,
                min(rate, datapath.hbm_write_bytes_per_s / 1000 or rate) * dma_eta, datapath.hbm_rw_alpha,
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
        if self.tilelang_profile is not None:
            latency = self.tilelang_profile.dispatch_us / 1000.0
            meta = {
                "stars_launch": True,
                "tilelang_api_tier": self.tilelang_profile.api_tier,
                "tilelang_dispatch_mode": self.tilelang_profile.dispatch_mode,
                "tilelang_dispatch_us": self.tilelang_profile.dispatch_us,
            }
            self.launches += 1
            return self.add(Stage(name, after=after, latency=latency, metadata=meta))
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
        dma_eta = self.tilelang_profile.dma_attainment if self.tilelang_profile is not None else 1.0
        if self.context is not None:
            copy_hit, copy_miss = _scaled_copy_rates(self.context, pattern.pipe, dma_eta)
            bound = self.context.bind_access(
                pattern, self.state, hbm_bus=self.hbm_bus, burst_rules=self.burst_rules,
                copy_hit_bytes_per_ms=copy_hit, copy_miss_bytes_per_ms=copy_miss, commit=True,
            )
            stage = bound.to_stage(name, after=after)
            if bound.l2_row_id not in self.context.rates and bound.l2_row_id in stage.work:
                stage = replace(stage, work={role: value for role, value in stage.work.items() if role != bound.l2_row_id})
            if self.live_state is not None:
                stage = Stage(name, after=after, program=_LiveAccess(bound, self.live_state, declared, self.obligations.get(name)))
        else:
            # Unit-rate normalization recovers native byte work independently
            # of hardware service rates. An Engine supplies real rates later.
            bus = HbmBusSpec(self.die, dma_eta, dma_eta, 0, row_r="hbm", row_w="hbm",
                             memory_access=self.state.memory_access)
            bound = bind_memory_access(
                pattern, self.state, hbm_bus=bus, copy_hit_bytes_per_ms=dma_eta,
                copy_miss_bytes_per_ms=dma_eta, l2_bytes_per_ms=1, pipe_row_id=pattern.pipe,
                l2_row_id="l2", burst_rules=self.burst_rules, commit=True,
            )
            stage = bound.to_stage(name, after=after)
        if self.tilelang_profile is not None:
            stage = replace(stage, metadata={
                **stage.metadata,
                "tilelang_api_tier": self.tilelang_profile.api_tier,
                "tilelang_dma_attainment": dma_eta,
                "tilelang_stages": self.tilelang_profile.pipelined_stages,
            })
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
            None if self.live_state is not None else float(self.sectors), self.misses,
            None if self.live_state is not None else float(self.service),
        )
        program = (
            _BoundTemplateProgram(stages, hbm_bus=self.hbm_bus, live_state=self.live_state,
                                  l2_reservation=self.l2_reservation, summary=summary)
            if self.hbm_bus is not None else KernelProgram(stages)
        )
        return TemplateLoweringResult(program, summary, len(stages))


def _route_edge(datapath, edge, owners, tile, explicit_bytes, *, ub_limit: int | None = None):
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
        eff_ub_limit = datapath.usable_bytes("ub") if ub_limit is None else min(datapath.usable_bytes("ub"), ub_limit)
        if explicit_bytes > eff_ub_limit:
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


def _route_region_edges(
    datapath, edges, edge_bytes, edge_owners, route_tiles, buffers, tile_count,
    *, ub_limit: int | None = None,
):
    routed = []
    source_buffers = {}
    source_slots: dict[tuple, int] = {}
    for idx, edge in enumerate(edges):
        size = edge_bytes.get(edge.view_id, 0)
        if type(size) is not int or size <= 0:
            raise ModelingError(f"fuse_region_edge_bytes_required:{edge.view_id}")
        dest_pool = edge.route_kind.split("->")[-1]
        owners = (edge_owners or {}).get(edge.view_id, (0, 0))
        route = _route_edge(
            datapath, edge, owners, (route_tiles or {}).get(edge.view_id, size), size, ub_limit=ub_limit,
        )
        routed.append((idx, edge, size, owners, route))
        if route is not None or edge.route_kind == "ub->ub":
            src_pool = edge.route_kind.split("->")[0]
            key = (edge.src_op, edge.producer_version, src_pool, edge.src_owner_axis)
            source_buffers[key] = max(source_buffers.get(key, 0), size)
            if route is not None:
                source_slots.setdefault(key, 1)
                buffers += (LiveBufferSpec(
                    f"edge:{idx}", dest_pool, size, 0, 1, owner_kind=edge.dst_owner_axis,
                    slots=tile_count if edge.iter_kind == "full_barrier" else edge.ring_depth,
                ),)
            else:
                needed_slots = (
                    tile_count if edge.iter_kind == "full_barrier"
                    else max(1, min(tile_count, int(edge.ring_depth)))
                )
                source_slots[key] = max(source_slots.get(key, 1), needed_slots)
    buffers += tuple(
        LiveBufferSpec(
            f"producer:{op}:{version}", pool, size, 0, 1,
            owner_kind=axis, slots=source_slots.get((op, version, pool, axis), 1),
        )
        for (op, version, pool, axis), size in source_buffers.items()
    )
    return routed, buffers


def _lower_region_passes(
    plan, order, parents, routed, rings, units, op_stages, second, reads, writes,
    semantic_work, initial, *, region_id, tile_count, context,
):
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
    tilelang_stages: int | None = None,
    tilelang_dma_pattern: str = "rw11",
    tilelang_dispatch_mode: str = "aclgraph",
    tilelang_vector_mode: str | None = None,
    tilelang_cube_epilogue: str = "fixpipe",
    tilelang_blockscaled: bool = False,
    tilelang_hf32: bool = False,
    tilelang_calibration: TileLangCalibration | None = None,
) -> tuple[FuseRegionSpec, TemplateLoweringResult]:
    """Compile per-tile stage declarations with explicit tensor and alias facts.

    Op stages describe one tile each. Pass-two declarations name the exact
    operations repeated after the global first-pass barrier.
    """
    edges, buffers = tuple(edge_routes), tuple(live_buffers)
    if not region_id or type(tile_count) is not int or tile_count < 1 or backend not in _VALID_FUSE_BACKENDS:
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

    tl_profile: TileLangLoweringProfile | None = None
    tl_cal: TileLangCalibration | None = None
    resolved_backend = backend
    if backend in TILELANG_BACKENDS:
        eff_stages = (
            tilelang_stages if tilelang_stages is not None
            else max((e.ring_depth for e in edges), default=TILELANG_DMA_STAGE_DOUBLE)
        )
        tl_profile, tl_cal = resolve_tilelang_profile(
            datapath,
            backend=backend,
            uses_simt=uses_simt,
            stages=eff_stages,
            dma_pattern=tilelang_dma_pattern,
            dispatch_mode=tilelang_dispatch_mode,
            vector_mode=tilelang_vector_mode,
            cube_epilogue=tilelang_cube_epilogue,
            blockscaled=tilelang_blockscaled,
            hf32=tilelang_hf32,
            calibration=tilelang_calibration,
        )
        resolved_backend = tl_profile.api_tier
        uses_simt = tl_profile.uses_simt
    else:
        uses_simt = uses_simt or backend == "tla_simt"

    routed, buffers = _route_region_edges(
        datapath, edges, edge_bytes, edge_owners, route_tiles, buffers, tile_count,
        ub_limit=tl_profile.ub_ceiling_bytes if tl_profile is not None else None,
    )
    has_cube_stage = any(
        role in {"cube", "matrix"}
        for stage in (*op_stages.values(), *second.values())
        for role, amount in stage.work.items() if amount
    )
    owner_axis = (
        "aic" if (tl_profile is not None and has_cube_stage)
        else ("aiv" if resolved_backend in {"dsl_vec", "tla_simt", "tilelang_simt", "tilelang_simd"} else "aic")
    )
    solution = solve_template_tile(datapath, region_id, TileCandidate(
        reduction_class, tile_shape, max((e.ring_depth for e in edges), default=1),
        (datapath.aic_count, datapath.aiv_count), tile_count, ceil_div(tile_count, max(1, datapath.aic_count)),
        buffers, uses_simt=uses_simt, owner_axis=owner_axis,
    ), state=memory_view, die=die, reserved=l2_reservation_bytes, tilelang_profile=tl_profile)
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
    spec = FuseRegionSpec(
        region_id, order, read_ranges, write_ranges, edges, reduction_class, solution,
        resolved_backend, tilelang_profile=tl_profile,
    )
    plan = Lowering(
        datapath, context=context, memory_view=memory_view, die=die, live_admission=live_admission,
        tilelang_profile=tl_profile, tilelang_calibration=tl_cal,
    )
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
    _lower_region_passes(
        plan, order, parents, routed, rings, units, op_stages, second, reads, writes,
        semantic_work, initial, region_id=region_id, tile_count=tile_count, context=context,
    )
    for ring in rings.values():
        plan.add_ring(ring, after=(launch,))
    return spec, plan.result("K3", region_id, order, spec, semantic_work, solution)


def _apply_tilelang_cube_attainment(
    plan: Lowering, stage: Stage, dtype: str, bound: float, stage_meta: dict[str, Any],
) -> float:
    profile, cal = plan.tilelang_profile, plan.tilelang_calibration
    if profile is None or cal is None:
        return bound
    epilogue = str(stage.metadata.get("tilelang_cube_epilogue", profile.cube_epilogue))
    is_blockscaled = bool(
        stage.metadata.get("tilelang_blockscaled", profile.blockscaled or dtype.lower() == "mxfp8")
    )
    is_hf32 = bool(stage.metadata.get("tilelang_hf32", profile.hf32))
    eta = cal.cube_attainment(dtype, epilogue=epilogue, blockscaled=is_blockscaled, hf32=is_hf32)
    stage_meta["tilelang_api_tier"] = profile.api_tier
    stage_meta["tilelang_cube_attainment"] = eta
    stage_meta["tilelang_blockscaled"] = is_blockscaled
    return bound / eta


def _apply_tilelang_vector_rate(
    plan: Lowering, stage: Stage, pipe: str, role_for_binding: str, row: str,
    amount: float, bound: float, stage_meta: dict[str, Any],
) -> float:
    profile, cal = plan.tilelang_profile, plan.tilelang_calibration
    if profile is None or cal is None:
        return bound
    vec_mode = stage.metadata.get("tilelang_vector_mode")
    if vec_mode is None:
        if pipe == "vec_sfu":
            vec_mode = "ub_exp" if profile.uses_simt else "sfu_exp"
        elif pipe == "vec_reduce":
            vec_mode = "reduce"
        else:
            vec_mode = profile.vector_mode
    tl_gops = cal.vector_gops_per_aiv(uses_simt=profile.uses_simt, op_kind=str(vec_mode))
    stage_meta["tilelang_api_tier"] = profile.api_tier
    stage_meta["tilelang_vector_mode"] = str(vec_mode)
    stage_meta["tilelang_vector_gops_per_aiv"] = tl_gops
    context = plan.context
    if context is not None and row in context.rates and float(context.rates[row]) > 0.0:
        row_units = max(1, int(context._units_for(role_for_binding, row, plan.datapath.aiv_count)))
        row_rate_per_aiv = float(context.rates[row]) / float(row_units)
        tl_ops_per_ms_per_aiv = tl_gops * _OPS_PER_MS_PER_GOP
        return float(amount) * (row_rate_per_aiv / tl_ops_per_ms_per_aiv)
    return bound


def _bind_declared_work(plan, stage, semantic):
    context = plan.context
    resource_names = context.resource_names if context else {}
    cube_row = resource_names.get("cube", "cube")
    work, placement, caps, units = Counter(), {}, {}, {}
    stage_meta = dict(stage.metadata)
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
            fallback = next(iter(dtypes)) if len(dtypes) == 1 else (getattr(context.op, "dtype", "") if context is not None else "")
            dtype = stage.metadata.get("cube_dtype", fallback)
            if not isinstance(dtype, str) or not dtype:
                raise ModelingError("fuse_region_cube_dtype_invalid")
            plan.cube[dtype] += int(raw)
            if context:
                row, bound = next(iter(context.bind(Work(matrix={dtype: raw})).items()))
            else:
                bound = 1000 * raw / plan.datapath.cube_peak(dtype)
            bound = _apply_tilelang_cube_attainment(plan, stage, dtype, bound, stage_meta)
            unit = "peak_ms"
        else:
            pipe = physical_pipes.get(role)
            if pipe is not None:
                plan.vector[pipe] += stage.metadata.get(f"issued_{pipe}_ops", amount)
            bound = amount
            role_for_binding = role
            if context and (role not in context.rates or context.service_rate(role) is not None):
                role_for_binding = plan.binding_role(role)
                bound_work = context.bind(Work(resources={role_for_binding: amount}))
                if not bound_work:
                    continue
                row, bound = next(iter(bound_work.items()))
                unit = context.work_units.get(context.physical_role(role_for_binding), unit)
            if pipe is not None:
                bound = _apply_tilelang_vector_rate(
                    plan, stage, pipe, role_for_binding, row, amount, bound, stage_meta,
                )
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
    if stage_meta != stage.metadata:
        return replace(stage, work=work, placement=placement, caps=caps, work_units=units, metadata=stage_meta)
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


# ---------------------------------------------------------------------------
# White-Box Formula Pipeline Lowering, Dual-Tier SRAM Admission & Evaluation
# ---------------------------------------------------------------------------

_MS_PER_SECOND = 1000.0
_US_PER_MS = 1000.0
_US_PER_SECOND = 1_000_000.0
_HALF_SPLIT_RATIO = 0.5
_HBM_RW_ASYMMETRY_RATIO = 1.5
_UB_DOUBLE_BUFFER_SLOTS = 2
_FORMULA_TILE_CAP = 64
_CUBE_TILE_MN_ELEMS = 128
_CUBE_TILE_K_ELEMS = 64
_FORMULA_VEC_OPS_PER_ELEM = 4.0
_FORMULA_MIN_ROW_WORK_BYTES = 2
_PEAK_DTYPE_ALIASES: Mapping[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
    "fp8_e4m3fn": "fp8",
    "fp8_e4m3": "fp8",
    "fp8_e5m2": "fp8",
    "fp4_e2m1": "fp4",
}


@dataclass(frozen=True, slots=True)
class FusedPipelineRecipe:
    """White-box L0-pipe recipe executing composed kernels directly on ``Engine``."""

    slots: int = _UB_DOUBLE_BUFFER_SLOTS
    tile_cap: int = _FORMULA_TILE_CAP
    row_reduction: bool = False
    tile_count: int = 0
    abi_signature_id: str = ""
    binding_state: BindingState = "unbound_formula_projection"
    traffic_rewrite_capable: bool = True
    pipe_stages: tuple[PipeStageSpec, ...] = ()

    def bind_context(self, context: RecipeContext) -> RecipeContext:
        """Bind calibrated L0 HBM R:W asymmetry rates when supplied by ``PrimitiveTable``."""
        op = context.op
        hbm_bytes = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
        read_bytes = float(getattr(op, "mte2_bytes", 0.0) or 0.0)
        write_bytes = float(getattr(op, "mte3_bytes", 0.0) or 0.0)
        if hbm_bytes <= 0.0 or "hbm" not in context.resource_names:
            return context
        hbm_read = max(0.0, hbm_bytes - write_bytes)
        hbm_write = min(hbm_bytes, write_bytes)
        if hbm_read <= 0.0 or hbm_write > _HBM_RW_ASYMMETRY_RATIO * hbm_read:
            dram_rate = context.service_rate("hbm_write_dram") or context.service_rate("mte2_aiv_dram")
        elif hbm_write <= 0.0:
            dram_rate = context.service_rate("mte2_aiv_dram")
        elif read_bytes > _HBM_RW_ASYMMETRY_RATIO * write_bytes:
            dram_rate = context.service_rate("hbm_rw21_dram") or context.service_rate("mte2_aiv_dram")
        else:
            dram_rate = (
                context.service_rate("hbm_rw11_dram")
                or context.service_rate("hbm_rw21_dram")
                or context.service_rate("mte2_aiv_dram")
            )
        if dram_rate is None or dram_rate <= 0.0:
            return context
        hbm_peak = float(context.rates[context.resource_names["hbm"]]) * _MS_PER_SECOND
        updated_modes = dict(context.service_modes)
        updated_modes["hbm"] = ("hbm", min(dram_rate, hbm_peak))
        return replace(context, service_modes=updated_modes)

    def _lower_dag_pipe_stages(
        self, categories: set[str], has_load: bool,
    ) -> tuple[list[Phase], tuple[str, ...]]:
        """Emit compute/communication ``Phase`` DAG preserving true operand-DAG order."""
        active_kinds: set[str] = set()
        if categories & {"cube", "mte1"}:
            active_kinds.add("cube")
        if "vector" in categories:
            active_kinds.add("vector")
        if "link" in categories:
            active_kinds.add("link")
        if not active_kinds:
            return [], ("mte2_load",) if has_load else ()

        raw_stages: list[PipeStageSpec] = list(self.pipe_stages)
        if not raw_stages:
            prev_name: tuple[str, ...] = ()
            if "cube" in active_kinds:
                raw_stages.append(PipeStageSpec("cube_mma", "cube", 1.0, prev_name))
                prev_name = ("cube_mma",)
            if "vector" in active_kinds:
                raw_stages.append(PipeStageSpec("aiv_fp32", "vector", 1.0, prev_name, self.row_reduction))
                prev_name = ("aiv_fp32",)
            if "link" in active_kinds:
                raw_stages.append(PipeStageSpec("link_pipe", "link", 1.0, prev_name))

        active_names = {st.name for st in raw_stages if st.pipe_kind in active_kinds}
        eff_preds: dict[str, tuple[str, ...]] = {}
        for st in raw_stages:
            preds: list[str] = []
            for p_name in st.after:
                if p_name in active_names:
                    preds.append(p_name)
                elif p_name in eff_preds:
                    preds.extend(eff_preds[p_name])
            eff_preds[st.name] = tuple(dict.fromkeys(preds))

        active_stages = [st for st in raw_stages if st.name in active_names]
        covered_kinds = {st.pipe_kind for st in active_stages}
        tail_names = tuple(
            st.name for st in active_stages
            if not any(st.name in eff_preds[other.name] for other in active_stages)
        )
        for kind, default_name in (("cube", "cube_mma"), ("vector", "aiv_fp32"), ("link", "link_pipe")):
            if kind in active_kinds and kind not in covered_kinds:
                st_extra = PipeStageSpec(
                    default_name, kind, 1.0, tail_names,
                    row_reduction=(self.row_reduction if kind == "vector" else False),
                )
                active_stages.append(st_extra)
                eff_preds[st_extra.name] = tail_names
                tail_names = (st_extra.name,)

        share_totals: dict[str, float] = defaultdict(float)
        for st in active_stages:
            share_totals[st.pipe_kind] += max(0.0, float(st.share))

        phases: list[Phase] = []
        term_name: dict[str, str] = {}
        used_as_pred: set[str] = set()
        for st in active_stages:
            total = share_totals[st.pipe_kind]
            norm_share = (max(0.0, float(st.share)) / total) if total > 0.0 else (1.0 / len(active_stages))
            preds = tuple(dict.fromkeys(term_name[p] for p in eff_preds.get(st.name, ()) if p in term_name))
            used_as_pred.update(p for p in eff_preds.get(st.name, ()) if p in term_name)
            if not preds and has_load:
                preds = ("mte2_load",)
            if st.pipe_kind == "cube":
                cube_w = {role: norm_share for role in ("cube", "mte1") if role in categories}
                phases.append(Phase(st.name, cube_w, after=preds))
                term_name[st.name] = st.name
            elif st.pipe_kind == "vector":
                if st.row_reduction:
                    r_name = "aiv_reduce" if st.name == "aiv_fp32" else f"{st.name}_reduce"
                    e_name = "aiv_epilogue" if st.name == "aiv_fp32" else f"{st.name}_epilogue"
                    phases.append(Phase(r_name, {"vector": norm_share * _HALF_SPLIT_RATIO}, after=preds))
                    phases.append(Phase(e_name, {"vector": norm_share * _HALF_SPLIT_RATIO}, after=(r_name,)))
                    term_name[st.name] = e_name
                else:
                    phases.append(Phase(st.name, {"vector": norm_share}, after=preds))
                    term_name[st.name] = st.name
            elif st.pipe_kind == "link":
                phases.append(Phase(st.name, {"link": norm_share}, after=preds))
                term_name[st.name] = st.name
        sinks = tuple(dict.fromkeys(term_name[st.name] for st in active_stages if st.name not in used_as_pred))
        return phases, sinks or (("mte2_load",) if has_load else ())

    def lower(self, context: RecipeContext) -> KernelProgram:
        """Lower declared physical work into a finite multi-pipe tile pipeline."""
        categories = set(_resource_categories(context).values())
        if not categories:
            return KernelProgram((Stage("empty", keep_empty=True),), minimum_ms=context.minimum_ms)
        read_share = io_weight(context, "hbm_read")
        write_share = io_weight(context, "hbm_write")
        if read_share + write_share <= 0.0:
            read_share = write_share = _HALF_SPLIT_RATIO
        phases: list[Phase] = []
        has_load = bool(categories & {"hbm", "mte2"}) and (read_share > 0.0 or "mte2" in categories)
        has_store = bool(categories & {"hbm", "mte3", "fixpipe"}) and (
            write_share > 0.0 or bool(categories & {"mte3", "fixpipe"})
        )
        if has_load:
            load_w: dict[str, float] = {}
            if "hbm" in categories and read_share > 0.0:
                load_w[IO_HBM_CATEGORY] = read_share if has_store else 1.0
            if "mte2" in categories:
                load_w["mte2"] = 1.0
            phases.append(Phase("mte2_load", load_w))
        pipe_phases, store_preds = self._lower_dag_pipe_stages(categories, has_load)
        phases.extend(pipe_phases)
        if has_store:
            store_w: dict[str, float] = {}
            if "hbm" in categories:
                hbm_w = write_share if has_load else 1.0
                if hbm_w > 0.0:
                    store_w[IO_HBM_CATEGORY] = hbm_w
            if "mte3" in categories:
                store_w["mte3"] = 1.0
            if "fixpipe" in categories:
                store_w["fixpipe"] = 1.0
            if store_w:
                phases.append(Phase("mte3_store", store_w, after=store_preds))
        if not phases:
            return phased_program(context, ())
        cap = self.tile_count if self.tile_count > 0 else self.tile_cap
        return tiled_phased_program(context, tuple(phases), slots=max(1, int(self.slots)), cap=cap)


def _die_launch_us(die: Device, datapath: DatapathCapabilitySpec, primitive_table: Any = None) -> tuple[float, float, float]:
    """Read L0 kernel completion minimums from PrimitiveTable or Device declaration."""
    inv = dict(getattr(die, "kernel_invocations", ()) or ())

    def _inv_us(key: str) -> float:
        entry = inv.get(key)
        if isinstance(entry, Mapping) and entry.get("minimum_ms") is not None:
            val = float(entry["minimum_ms"]) * _US_PER_MS
            if val > 0.0:
                return val
        launch = float(getattr(die, "launch_us", 0.0) or 0.0)
        return max(0.0, launch)

    launch_vec = _inv_us("vector")
    launch_cube = _inv_us("cube")
    launch_norm = _inv_us("norm") or launch_vec
    env = resolve_dispatch_envelope(primitive_table, datapath) if primitive_table is not None else None
    if env is not None:
        aiv_us = float(env.aiv_full_die_floor_ms) * _US_PER_MS
        if aiv_us > 0.0:
            launch_vec = max(launch_vec, aiv_us) if launch_vec > 0.0 else aiv_us
            launch_norm = max(launch_norm, aiv_us)
        if launch_cube <= 0.0:
            launch_cube = float(env.mix_aic_floor_ms(0.0)) * _US_PER_MS
    return launch_vec, launch_cube, launch_norm


def _resolve_die_fabric_rate(
    rank: int,
    links: Sequence[Any],
    *,
    active_width: int | None,
    ranks_per_node: int,
    dies_per_package: int,
    fallback_rate: float,
) -> float:
    """Resolve effective per-die link bytes/s strictly from declared ``Hardware.links``."""
    candidates: list[float] = []
    for lnk in links:
        members = tuple(getattr(lnk, "members", ()) or ())
        if members and rank not in members:
            continue
        cap_bpms = float(getattr(lnk, "capacity_bytes_per_ms", 0.0) or 0.0)
        if cap_bpms <= 0.0:
            continue
        sem = str(getattr(lnk, "semantics", "") or "").lower()
        bw_bps = cap_bpms * _MS_PER_SECOND
        if sem == "shared" and len(members) > 1:
            bw_bps /= float(len(members))
        elif sem == "per_pair" and active_width is not None and active_width > 1:
            bw_bps *= float(min(active_width - 1, max(1, len(members) - 1)))
        candidates.append(bw_bps)
    del ranks_per_node, dies_per_package
    return max(candidates) if candidates else fallback_rate


def _extract_hardware_supply(
    hardware: Case | Hardware | Device | DatapathCapabilitySpec,
    primitive_table: Any = None,
) -> tuple[DatapathCapabilitySpec, dict[str, float], dict[str, float], dict[str, tuple[str, float]], float, float, float, float]:
    """Resolve ``(datapath, rates_per_s, peaks, service_modes, launch_vec_us, launch_cube_us, fabric_bps, fabric_lat_us)``."""
    links: tuple[Any, ...] = ()
    channels: tuple[Any, ...] = ()
    die: Device | None = None
    mac: Hardware | None = None
    if isinstance(hardware, Case):
        mac = hardware.machine
        die = mac.dies[0]
        links = tuple(mac.links)
        channels = tuple(mac.transfer_channels)
    elif isinstance(hardware, Hardware):
        mac = hardware
        die = mac.dies[0]
        links = tuple(mac.links)
        channels = tuple(mac.transfer_channels)
    elif isinstance(hardware, Device):
        die = hardware
    elif isinstance(hardware, DatapathCapabilitySpec):
        datapath = hardware
    else:
        raise ModelingError(f"formula_kernel_unsupported_hardware:{type(hardware).__name__}")

    if die is not None:
        ub_edges = ub_peer_edges_for_rank(die.rank, links, channels)
        datapath = datapath_for_die(die, ub_peer_edges=ub_edges)
        rates = dict(die.rates)
        peaks = dict(die.compute_peaks)
        service_modes = dict(die.service_modes)
        launch_vec, launch_cube, _ = _die_launch_us(die, datapath, primitive_table)
    else:
        rates = {}
        hbm_rate = max(float(datapath.hbm_read_bytes_per_s or 0.0), float(datapath.hbm_write_bytes_per_s or 0.0))
        if hbm_rate > 0.0:
            rates["hbm"] = hbm_rate
            rates["mte2"] = hbm_rate
            rates["mte3"] = hbm_rate
        peaks = dict(datapath.compute_peaks)
        service_modes = {}
        stars = datapath.stars
        base_launch = float(stars.issue_interval_us) if stars is not None else 0.0
        launch_vec = launch_cube = base_launch

    orig_hbm_cap = float(rates.get("hbm") or 0.0)
    if primitive_table is not None:
        sku = getattr(die, "sku", "") if die is not None else ""
        overlay = resolve_sim_rate_overlay(primitive_table, datapath, sku=sku)
        for role, val in overlay.rates:
            if val > 0.0:
                rates[role] = min(val, rates[role]) if rates.get(role, 0.0) > 0.0 else val
        for dt, pk in overlay.compute_peaks:
            if pk > 0.0:
                peaks[dt] = pk
        dp_updates: dict[str, Any] = {}
        if overlay.hbm_read_bytes_per_s is not None:
            hbm_r_ov = float(overlay.hbm_read_bytes_per_s)
            dp_updates["hbm_read_bytes_per_s"] = min(hbm_r_ov, orig_hbm_cap) if orig_hbm_cap > 0.0 else hbm_r_ov
            service_modes["mte2_aiv_dram"] = ("mte2_aiv" if "mte2_aiv" in rates else "mte2", hbm_r_ov)
        if overlay.hbm_write_bytes_per_s is not None:
            hbm_w_ov = float(overlay.hbm_write_bytes_per_s)
            dp_updates["hbm_write_bytes_per_s"] = min(hbm_w_ov, orig_hbm_cap) if orig_hbm_cap > 0.0 else hbm_w_ov
            service_modes["hbm_write_dram"] = ("hbm", hbm_w_ov)
        if overlay.hbm_rw_alpha is not None:
            dp_updates["hbm_rw_alpha"] = float(overlay.hbm_rw_alpha)
        if dp_updates:
            datapath = replace(datapath, **dp_updates)
        if overlay.hbm_rw11_bytes_per_s is not None:
            service_modes["hbm_rw11_dram"] = ("hbm", overlay.hbm_rw11_bytes_per_s)
        if overlay.hbm_rw21_bytes_per_s is not None:
            service_modes["hbm_rw21_dram"] = ("hbm", overlay.hbm_rw21_bytes_per_s)
    if orig_hbm_cap > 0.0:
        cap_updates: dict[str, Any] = {}
        if float(datapath.hbm_read_bytes_per_s or 0.0) > orig_hbm_cap:
            cap_updates["hbm_read_bytes_per_s"] = orig_hbm_cap
        if float(datapath.hbm_write_bytes_per_s or 0.0) > orig_hbm_cap:
            cap_updates["hbm_write_bytes_per_s"] = orig_hbm_cap
        if cap_updates:
            datapath = replace(datapath, **cap_updates)

    fabric_rate = float(rates.get("link") or 0.0)
    fabric_lat_us = 0.0
    if links:
        active_w = max(1, int(hardware.runtime.parallel.tp)) if isinstance(hardware, Case) else None
        fabric_rate = _resolve_die_fabric_rate(
            int(die.rank) if die is not None else 0,
            links,
            active_width=active_w,
            ranks_per_node=int(getattr(mac, "ranks_per_node", 0) or 0),
            dies_per_package=int(getattr(mac, "dies_per_package", 1) or 1),
            fallback_rate=fabric_rate,
        )
        fabric_lat_us = min((float(getattr(lnk, "latency_us", 0.0) or 0.0) for lnk in links), default=0.0)
    return datapath, rates, peaks, service_modes, launch_vec, launch_cube, fabric_rate, fabric_lat_us


_MATMUL_SHAPE_RANK = 3


def _typed_port_row_bytes(entries: Any, default_elem_bytes: float) -> dict[tuple[str, str, int], int]:
    """Extract explicit ``(producer, port, version) -> row_bytes`` from typed port metadata."""
    out: dict[tuple[str, str, int], int] = {}
    if not isinstance(entries, (tuple, list)):
        return out
    for item in entries:
        if not isinstance(item, Mapping):
            continue
        prod = str(item.get("producer") or "").strip()
        port = str(item.get("port") or item.get("name") or "out").strip()
        ver = int(item.get("version", 0) or 0)
        shape = item.get("shape")
        if isinstance(shape, (tuple, list)) and shape and isinstance(shape[-1], int) and shape[-1] > 0:
            eb = formula_elem_bytes(str(item.get("dtype"))) if item.get("dtype") else default_elem_bytes
            out[(prod, port, ver)] = int(shape[-1] * eb)
    return out


def _formula_row_sram_bytes(spec: FormulaKernelSpec) -> tuple[bool, bool, int, int, int, int]:
    """Compute ``(has_cube, has_aiv, tile_n, row_fp32_bytes, l1_tile, l0_tile)`` with port-keyed UB liveness."""
    from .contracts import _extract_op_operand_ports

    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    infos = tuple(_infer_op_symbolic_kind(op) for op in spec.atomic_ops)
    has_row_red = bool(spec.row_reduction) or any(info[2] for info in infos)
    has_aiv = (
        any(not info[1] for info in infos)
        or spec.vector_ops_fp32 > 0.0
        or has_row_red
        or not has_cube
    )
    all_shapes = [t.shape for t in (*spec.inputs, *spec.outputs) if t.shape]
    in_b = max((formula_elem_bytes(t.dtype) for t in spec.inputs), default=0.0)
    out_b = max((formula_elem_bytes(t.dtype) for t in spec.outputs), default=0.0)
    internal_red_widths: list[int] = []
    internal_aiv_widths: list[int] = []
    op_work_bytes: list[int] = []
    op_out_row_bytes: list[int] = []
    op_is_mat: list[bool] = []
    idx_by_id = {op.op_id: idx for idx, op in enumerate(spec.atomic_ops)}
    int_last_use: dict[tuple[int, str, int], int] = {}
    int_port_bytes: dict[tuple[int, str, int], int] = {}
    int_out_ports: dict[tuple[int, str], int] = {}
    ext_first_use: dict[tuple[str, str, int], int] = {}
    ext_last_use: dict[tuple[str, str, int], int] = {}
    ext_row_bytes: dict[tuple[str, str, int], int] = {}
    for idx, (op, (_kind, is_mat, is_red, _act, _params)) in enumerate(zip(spec.atomic_ops, infos)):
        meta = dict(op.metadata) if op.metadata else {}
        in_dt = str(meta.get("activation_format") or meta.get("input_dtype") or op.dtype)
        out_dt = str(meta.get("output_format") or meta.get("output_dtype") or op.dtype)
        op_in_raw = formula_elem_bytes(in_dt)
        op_out_raw = formula_elem_bytes(out_dt)
        in_b = max(in_b, op_in_raw)
        out_b = max(out_b, op_out_raw)
        is_3d_mat = is_mat and len(op.local_shape) == _MATMUL_SHAPE_RANK
        op_in_n = int(op.local_shape[2]) if is_3d_mat else (int(op.local_shape[-1]) if op.local_shape else 0)
        op_out_n = int(op.local_shape[1]) if is_3d_mat else (int(op.local_shape[-1]) if op.local_shape else 0)
        explicit_in = _typed_port_row_bytes(meta.get("typed_inputs"), op_in_raw)
        for (_, p_name, _), p_bytes in _typed_port_row_bytes(meta.get("typed_outputs"), op_out_raw).items():
            int_out_ports[(idx, p_name)] = p_bytes
        for prod_id, port_name, ver in _extract_op_operand_ports(op):
            port_key = (prod_id, port_name, ver)
            cons_bytes = explicit_in.get(port_key, int(op_in_n * op_in_raw) if op_in_n > 0 else 0)
            if prod_id in idx_by_id:
                p_idx = idx_by_id[prod_id]
                int_key = (p_idx, port_name, ver)
                int_last_use[int_key] = max(int_last_use.get(int_key, p_idx), idx)
                if cons_bytes > 0:
                    int_port_bytes[int_key] = max(int_port_bytes.get(int_key, 0), cons_bytes)
            else:
                ext_first_use.setdefault(port_key, idx)
                ext_last_use[port_key] = max(ext_last_use.get(port_key, idx), idx)
                if cons_bytes > 0:
                    ext_row_bytes[port_key] = max(ext_row_bytes.get(port_key, 0), cons_bytes)
        op_is_mat.append(is_mat)
        if is_mat or op_out_n <= 0:
            op_work_bytes.append(0)
            op_out_row_bytes.append(int(op_out_n * op_out_raw) if op_out_n > 0 else 0)
            continue
        op_in_b = max(op_in_raw, float(FP32_BYTES))
        op_out_b = max(op_out_raw, float(FP32_BYTES))
        op_out_row_bytes.append(int(op_out_n * op_out_raw))
        if is_red:
            internal_red_widths.append(op_out_n)
            op_work_bytes.append(int(op_out_n * (op_in_b + op_out_b + float(FP32_BYTES))))
        else:
            internal_aiv_widths.append(op_out_n)
            op_work_bytes.append(int(op_out_n * (op_in_b + op_out_b)) if has_row_red else 0)

    ports_per_prod: dict[int, int] = defaultdict(int)
    for p_idx, _, _ in int_last_use:
        ports_per_prod[p_idx] += 1
    internal_peak_bytes = 0
    for j, work_j in enumerate(op_work_bytes):
        if op_is_mat[j] or work_j <= 0:
            continue
        carried = 0
        for (p_idx, port_name, ver), l_idx in int_last_use.items():
            if not (p_idx < j < l_idx):
                continue
            explicit_out = int_out_ports.get((p_idx, port_name), 0)
            if explicit_out > 0:
                carried += explicit_out
            elif ports_per_prod[p_idx] > 1:
                carried += int_port_bytes.get((p_idx, port_name, ver), 0) or op_out_row_bytes[p_idx]
            else:
                carried += op_out_row_bytes[p_idx] or int_port_bytes.get((p_idx, port_name, ver), 0)
        carried_ext = sum(
            ext_row_bytes.get(pkey, 0) for pkey, f_idx in ext_first_use.items()
            if f_idx < j < ext_last_use.get(pkey, -1)
        )
        internal_peak_bytes = max(internal_peak_bytes, work_j + carried + carried_ext)

    width_pool = [int(sh[-1]) for sh in all_shapes] + internal_aiv_widths + internal_red_widths
    inner_dim = max(width_pool, default=_CUBE_TILE_MN_ELEMS)
    if internal_red_widths:
        tile_n = max(internal_red_widths)
    elif has_row_red:
        tile_n = inner_dim
    else:
        tile_n = min(inner_dim, _CUBE_TILE_MN_ELEMS)
    if has_aiv:
        base_row_bytes = int(tile_n * (max(in_b, float(FP32_BYTES)) + max(out_b, float(FP32_BYTES))))
        if has_row_red:
            base_row_bytes += int(tile_n * FP32_BYTES)
        row_fp32_bytes = max(base_row_bytes, internal_peak_bytes)
    else:
        row_fp32_bytes = 0
    if has_cube:
        a_bytes = in_b if in_b > 0.0 else formula_elem_bytes(spec.matrix_flops[0][0])
        b_bytes = out_b if out_b > 0.0 else a_bytes
        l1_tile = int(_CUBE_TILE_MN_ELEMS * _CUBE_TILE_K_ELEMS * (a_bytes + b_bytes) * _UB_DOUBLE_BUFFER_SLOTS)
        l0_tile = int(_CUBE_TILE_MN_ELEMS * _CUBE_TILE_MN_ELEMS * FP32_BYTES)
    else:
        l1_tile = 0
        l0_tile = 0
    return has_cube, has_aiv, tile_n, row_fp32_bytes, l1_tile, l0_tile


def solve_formula_sram(
    spec: FormulaKernelSpec,
    datapath: DatapathCapabilitySpec,
    *,
    cube_budget: int | None = None,
    vec_budget: int | None = None,
) -> tuple[TilePipelineSpec | None, str, str, int, int, int, int, int, int, int, bool]:
    """Solve Ascend ``FP32`` UB/L1/L0 tiling for ``FormulaKernelSpec`` strictly from ``datapath``."""
    bufs = dict(datapath.buffers)
    carveout = max(0, int(datapath.simt_ub_carveout_bytes or 0))
    ub_raw = int(bufs.get("ub", 0) or 0)
    ub_budget = max(0, ub_raw - carveout)
    l1_budget = int(bufs.get("l1", 0) or 0)
    l0_budget = int(bufs.get("l0c", 0) or 0)
    has_cube, has_aiv, tile_n, row_fp32_bytes, l1_tile, l0_tile = _formula_row_sram_bytes(spec)

    if spec.uses_simt:
        if has_aiv and carveout <= 0:
            return (
                None, "simt_unsupported", f"(M=1, N={tile_n}, K={_CUBE_TILE_K_ELEMS if has_cube else 1})",
                row_fp32_bytes, 0, l1_tile, l1_budget, l0_tile, l0_budget, 1, False,
            )
        cal = try_load_tilelang_calibration(str(datapath.arch_id))
        if cal is not None:
            simt_prof, _ = resolve_tilelang_profile(datapath, uses_simt=True, calibration=cal)
            max_ub_budget = simt_prof.ub_ceiling_bytes
        else:
            max_ub_budget = ub_budget
        cand_modes: tuple[tuple[int, bool], ...] = ((_UB_DOUBLE_BUFFER_SLOTS, True), (1, True))
        eff_bufs = {**bufs, "ub": max_ub_budget + carveout}
        eff_dp = replace(datapath, buffers=tuple(eff_bufs.items()))
    else:
        cand_modes = ((_UB_DOUBLE_BUFFER_SLOTS, False), (1, False))
        max_ub_budget = ub_raw
        eff_bufs = bufs
        eff_dp = datapath
    candidates: list[TileCandidate] = []
    owner_part = (cube_budget or datapath.aic_count, vec_budget or datapath.aiv_count)
    for slots, cand_simt in cand_modes:
        live: list[LiveBufferSpec] = []
        if has_aiv and row_fp32_bytes > 0:
            live.append(LiveBufferSpec("ub_fp32_work", "ub", row_fp32_bytes, 0, 1, owner_kind="aiv", slots=slots))
        if has_cube and l1_tile > 0:
            live.append(LiveBufferSpec("l1_cube_tile", "l1", l1_tile, 0, 1, owner_kind="aic", slots=1))
        if has_cube and l0_tile > 0:
            live.append(LiveBufferSpec("l0c_fp32_accum", "l0c", l0_tile, 0, 1, owner_kind="aic", slots=1))
        algo: Literal["resident_1pass", "online_reduce"] = (
            "resident_1pass" if slots == _UB_DOUBLE_BUFFER_SLOTS else "online_reduce"
        )
        candidates.append(TileCandidate(
            algorithm=algo,
            tile_shape=TileShape(1, max(1, tile_n), _CUBE_TILE_K_ELEMS if has_cube else 1),
            depth=slots,
            owner_partition=owner_part,
            work_items=1,
            tiles_per_owner=1,
            buffers=tuple(live),
            uses_simt=cand_simt,
            owner_axis="aic" if (has_cube and not has_aiv) else "aiv",
        ))
    tile_spec = TilePipelineSpec(spec.family or "formula_kernel", tuple(candidates), simt_ub_carveout_bytes=carveout)
    hw = SimpleNamespace(
        datapath=eff_dp,
        buffers=eff_bufs,
        buffer_units={"ub": datapath.aiv_count, "l1": datapath.aic_count, "l0c": datapath.aic_count, "l2": 1},
        stream_owners=None,
    )
    if has_aiv and ub_raw <= 0:
        return tile_spec, "unbounded", f"(M=1, N={tile_n}, K=1)", 0, 0, l1_tile, l1_budget, l0_tile, l0_budget, 1, False
    try:
        solved = solve(tile_spec, None, hw)
        peaks = {p.pool: int(p.peak_bytes) for p in solved.pool_peaks}
        caps = {p.pool: int(p.capacity_bytes) for p in solved.pool_peaks}
        ub_pk = peaks.get("ub", row_fp32_bytes) if has_aiv else 0
        eff_ub_budget = caps.get("ub", max_ub_budget) if has_aiv else max_ub_budget
        l1_pk = peaks.get("l1", l1_tile) if has_cube else 0
        l0_pk = peaks.get("l0c", l0_tile) if has_cube else 0
        ok = (
            (not has_aiv or ub_pk <= eff_ub_budget)
            and (not has_cube or (l1_pk <= l1_budget and l0_pk <= l0_budget))
        )
        sh = solved.tile_shape
        return (
            tile_spec, solved.algorithm, f"(M={sh.m}, N={sh.n}, K={sh.k})",
            ub_pk, eff_ub_budget, l1_pk, l1_budget, l0_pk, l0_budget, solved.depth, ok,
        )
    except TileInfeasibleError:
        fail_algo = "ub_infeasible" if (has_aiv and row_fp32_bytes > max_ub_budget) else "sram_infeasible"
        return (
            tile_spec, fail_algo, f"(M=1, N={tile_n}, K={_CUBE_TILE_K_ELEMS if has_cube else 1})",
            row_fp32_bytes, max_ub_budget, l1_tile, l1_budget, l0_tile, l0_budget, 1, False,
        )


def _compute_peak_for_dtype(peaks: Mapping[str, float], dtype: str) -> float:
    tok = str(dtype or "").strip().lower()
    if not tok:
        raise ModelingError("formula_kernel_compute_dtype_missing")
    canon = _PEAK_DTYPE_ALIASES.get(tok, tok)
    for key in (tok, canon):
        val = float(peaks.get(key, 0.0) or 0.0)
        if val > 0.0:
            return val
    raise ModelingError(f"formula_kernel_compute_peak_missing:{dtype}")


def _resolve_formula_work_bytes(spec: FormulaKernelSpec) -> tuple[float, float, float, float]:
    """Return ``(mte2_bytes, mte3_bytes, hbm_bytes, vector_ops_fp32)`` for ``spec``."""
    in_bytes = sum(t.byte_size for t in (*spec.inputs, *spec.weights) if t.hbm_resident)
    out_bytes = sum(t.byte_size for t in spec.outputs if t.hbm_resident)
    mte2 = float(spec.mte2_bytes_override) if spec.mte2_bytes_override is not None else in_bytes
    mte3 = float(spec.mte3_bytes_override) if spec.mte3_bytes_override is not None else out_bytes
    hbm = float(spec.hbm_bytes_override) if spec.hbm_bytes_override is not None else (mte2 + mte3)
    vec_ops = float(spec.vector_ops_fp32)
    return max(0.0, mte2), max(0.0, mte3), max(0.0, hbm), max(0.0, vec_ops)


def _build_formula_engine_inputs(
    spec: FormulaKernelSpec,
    rates: Mapping[str, float],
    peaks: Mapping[str, float],
    datapath: DatapathCapabilitySpec,
    *,
    mte2: float,
    mte3: float,
    hbm: float,
    vec_ops: float,
    fabric_rate: float,
    aic_eff: int,
    aiv_eff: int,
) -> tuple[list[Resource], dict[str, str], dict[str, float], dict[str, int], dict[str, float], dict[str, float], dict[str, float], float]:
    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    total_flops = sum(float(fl) for _, fl in spec.matrix_flops if fl > 0.0)
    aic_scale = float(aic_eff) / float(max(1, datapath.aic_count))
    aiv_scale = float(aiv_eff) / float(max(1, datapath.aiv_count))
    cube_ms = 0.0
    for dt, fl in spec.matrix_flops:
        if fl <= 0.0:
            continue
        pk = _compute_peak_for_dtype(peaks, dt) * aic_scale
        cube_ms += _MS_PER_SECOND * float(fl) / pk

    fallback_hbm = float(rates.get("hbm") or 0.0)
    hbm_read_bps = float(getattr(datapath, "hbm_read_bytes_per_s", 0.0) or 0.0) or fallback_hbm
    hbm_write_bps = float(getattr(datapath, "hbm_write_bytes_per_s", 0.0) or 0.0) or fallback_hbm
    alpha_rw = max(0.0, min(1.0, float(getattr(datapath, "hbm_rw_alpha", 0.0) or 0.0)))
    if hbm > 0.0 and hbm_read_bps > 0.0 and hbm_write_bps > 0.0:
        if mte2 > 0.0 and mte3 > 0.0:
            r_share = mte2 / (mte2 + mte3)
            eff_r, eff_w = hbm * r_share, hbm * (1.0 - r_share)
        elif mte3 > 0.0 and mte2 <= 0.0:
            eff_r, eff_w = 0.0, hbm
        else:
            eff_r, eff_w = hbm, 0.0
        t_r_s = (eff_r / hbm_read_bps if eff_r > 0.0 else 0.0) + (
            alpha_rw * eff_w / hbm_write_bps if eff_w > 0.0 else 0.0
        )
        t_w_s = (eff_w / hbm_write_bps if eff_w > 0.0 else 0.0) + (
            alpha_rw * eff_r / hbm_read_bps if eff_r > 0.0 else 0.0
        )
        hbm_time_s = max(t_r_s, t_w_s)
        hbm_rate = (hbm / hbm_time_s) if hbm_time_s > 0.0 else max(hbm_read_bps, hbm_write_bps)
    else:
        hbm_rate = max(hbm_read_bps, hbm_write_bps, fallback_hbm)
    vec_rate = float(rates.get("vector") or 0.0) * aiv_scale
    mte2_key = "mte2_aic" if (has_cube and "mte2_aic" in rates) else ("mte2_aiv" if "mte2_aiv" in rates else "mte2")
    mte2_rate = float(rates.get(mte2_key) or hbm_rate or 0.0) * (aic_scale if mte2_key == "mte2_aic" else aiv_scale)
    mte3_rate = float(rates.get("mte3") or hbm_rate or 0.0) * aiv_scale
    if hbm > 0.0 and hbm_rate <= 0.0:
        raise ModelingError("formula_kernel_hbm_rate_missing")
    if vec_ops > 0.0 and vec_rate <= 0.0:
        raise ModelingError("formula_kernel_vector_rate_missing")
    if spec.link_bytes > 0.0 and fabric_rate <= 0.0:
        raise ModelingError("formula_kernel_link_rate_missing")

    res_list: list[Resource] = [Resource("matrix@0", 1.0, "peak_ms", "fair", aic_eff)]
    res_names: dict[str, str] = {"cube": "matrix@0"}
    res_rates: dict[str, float] = {"matrix@0": 1.0}
    res_units: dict[str, int] = {"cube": aic_eff}
    work_map: dict[str, float] = {}
    native_work: dict[str, float] = {}
    if cube_ms > 0.0:
        work_map["matrix@0"] = cube_ms
        native_work["cube"] = total_flops
    pipe_rows = (
        ("hbm", "memory@0", hbm_rate, "bytes", 1, hbm),
        ("mte2", "mte2@0", mte2_rate, "bytes", aic_eff if has_cube else aiv_eff, mte2),
        ("mte3", "mte3@0", mte3_rate, "bytes", aiv_eff, mte3),
        ("vector", "vector@0", vec_rate, "ops", aiv_eff, vec_ops),
        ("link", "link:0", fabric_rate, "bytes", 1, float(spec.link_bytes)),
    )
    for role, rname, rate_s, unit, units, amount in pipe_rows:
        if rate_s <= 0.0 or (role == "link" and amount <= 0.0):
            continue
        res_list.append(Resource(rname, rate_s / _MS_PER_SECOND, unit, "fair", units))
        res_names[role] = rname
        res_rates[rname] = rate_s / _MS_PER_SECOND
        res_units[role] = units
        if amount > 0.0:
            work_map[rname] = amount
            native_work[role] = amount

    pipe_times_us = {
        "cube": cube_ms * _US_PER_MS,
        "vector": (vec_ops / vec_rate) * _US_PER_SECOND if (vec_ops > 0.0 and vec_rate > 0.0) else 0.0,
        "hbm": (hbm / hbm_rate) * _US_PER_SECOND if (hbm > 0.0 and hbm_rate > 0.0) else 0.0,
        "mte2": (mte2 / mte2_rate) * _US_PER_SECOND if (mte2 > 0.0 and mte2_rate > 0.0) else 0.0,
        "mte3": (mte3 / mte3_rate) * _US_PER_SECOND if (mte3 > 0.0 and mte3_rate > 0.0) else 0.0,
        "link": (float(spec.link_bytes) / fabric_rate) * _US_PER_SECOND if (spec.link_bytes > 0.0 and fabric_rate > 0.0) else 0.0,
    }
    return res_list, res_names, res_rates, res_units, work_map, native_work, pipe_times_us, hbm_rate


def evaluate_formula_kernel(
    spec: FormulaKernelSpec,
    hardware: Case | Hardware | Device | DatapathCapabilitySpec,
    *,
    primitive_table: Any = None,
    active_aic: int | None = None,
    active_aiv: int | None = None,
) -> FormulaKernelEvaluation:
    """Evaluate a Layer-1 ``FormulaKernelSpec`` through ``TileSolver`` + ``Engine`` L0 pipes."""
    datapath, rates, peaks, service_modes, launch_vec, launch_cube, fabric_rate, fabric_lat_us = (
        _extract_hardware_supply(hardware, primitive_table=primitive_table)
    )
    (
        tile_spec, algo, shape_str, ub_pk, ub_cap, l1_pk, l1_cap, l0_pk, l0_cap, depth, sram_ok,
    ) = solve_formula_sram(spec, datapath, cube_budget=active_aic, vec_budget=active_aiv)
    mte2, mte3, hbm, vec_ops = _resolve_formula_work_bytes(spec)
    has_cube = any(fl > 0.0 for _, fl in spec.matrix_flops)
    total_flops = sum(float(fl) for _, fl in spec.matrix_flops if fl > 0.0)
    aic_raw = active_aic if active_aic is not None else datapath.aic_count
    aiv_raw = active_aiv if active_aiv is not None else datapath.aiv_count
    if aic_raw <= 0 or aiv_raw <= 0:
        raise ModelingError(f"formula_kernel_invalid_core_counts:aic={aic_raw}:aiv={aiv_raw}")
    aic_eff = int(aic_raw)
    aiv_eff = int(aiv_raw)
    res_list, res_names, res_rates, res_units, work_map, native_work, pipe_times_us, hbm_rate = (
        _build_formula_engine_inputs(
            spec, rates, peaks, datapath,
            mte2=mte2, mte3=mte3, hbm=hbm, vec_ops=vec_ops,
            fabric_rate=fabric_rate, aic_eff=aic_eff, aiv_eff=aiv_eff,
        )
    )
    launch_us = launch_cube if has_cube else launch_vec
    setup_ms = (fabric_lat_us / _US_PER_MS) if spec.link_bytes > 0.0 else 0.0
    op_shape = spec.outputs[0].shape if spec.outputs else (spec.inputs[0].shape if spec.inputs else ())
    op_dtype = (
        spec.outputs[0].dtype if spec.outputs
        else (spec.inputs[0].dtype if spec.inputs else (spec.matrix_flops[0][0] if spec.matrix_flops else ""))
    )
    if not op_dtype:
        raise ModelingError("formula_kernel_dtype_missing")
    op_stub = OpSpec(
        op_id=spec.name, family=spec.family, phase="decode", stage=0, layer=0,
        requests=(), semantic_shape=op_shape, local_shape=op_shape,
        padded_shape=op_shape, launch_shape=op_shape,
        dtype=op_dtype,
        flops=total_flops, matrix_flops=spec.matrix_flops, vector_ops=vec_ops,
        hbm_bytes=hbm, mte2_bytes=mte2, mte3_bytes=mte3, link_bytes=float(spec.link_bytes),
    )
    abi_id = spec.resolved_abi_signature()
    resolved_stages = spec.pipe_stages
    if not resolved_stages and spec.atomic_ops:
        resolved_stages = _derive_pipe_stages_from_ops(spec.atomic_ops)
    elif not resolved_stages and spec.equations:
        resolved_stages = _derive_pipe_stages_from_equations(
            spec.equations,
            has_cube=has_cube,
            has_vector=(vec_ops > 0.0),
            has_link=(spec.link_bytes > 0.0),
            row_reduction=spec.row_reduction,
        )
    recipe = FusedPipelineRecipe(
        slots=depth,
        row_reduction=spec.row_reduction,
        abi_signature_id=abi_id,
        binding_state="unbound_formula_projection",
        pipe_stages=resolved_stages,
    )
    ctx = RecipeContext(
        op=op_stub, work=work_map, resource_names=res_names, rates=res_rates,
        core_count=aic_eff if has_cube else aiv_eff,
        minimum_ms=launch_us / _US_PER_MS, setup_ms=setup_ms,
        resource_units=res_units, compute_peaks=peaks, service_modes=service_modes,
        native_work=native_work, datapath=datapath, primitive_table=primitive_table,
    )
    bound_ctx = recipe.bind_context(ctx)
    if bound_ctx.service_modes.get("hbm") != ctx.service_modes.get("hbm") and "memory@0" in work_map:
        _, eff_hbm_bps = bound_ctx.service_modes["hbm"]
        if eff_hbm_bps > 0.0:
            work_map = {**work_map, "memory@0": hbm * (hbm_rate / eff_hbm_bps)}
            bound_ctx = replace(bound_ctx, work=work_map)

    dom_pipe, l_star_us = max(pipe_times_us.items(), key=lambda kv: kv[1])
    if l_star_us <= 0.0:
        dom_pipe = "launch"

    engine = Engine(res_list, backend="python")
    ready = engine.submit("ready")
    done = recipe.lower(bound_ctx).start(engine, spec.name, ready)
    engine.advance(until=(done,))
    projected_us = max(float(done.time) * _US_PER_MS, l_star_us)
    return FormulaKernelEvaluation(
        name=spec.name, abi_signature_id=abi_id, composed_equations=spec.equations,
        projected_us=projected_us, l_star_us=l_star_us, tile_spec=tile_spec,
        ub_peak_bytes=ub_pk, ub_budget_bytes=ub_cap, l1_peak_bytes=l1_pk, l0_peak_bytes=l0_pk,
        sram_verified=sram_ok, binding_state="unbound_formula_projection",
        has_unbound_formula_ops=True, dominant_pipe=dom_pipe,
        hbm_bytes=hbm, mte2_bytes=mte2, mte3_bytes=mte3, launch_us=launch_us,
        tile_algorithm=algo, tile_shape_str=shape_str,
        l1_budget_bytes=l1_cap, l0_budget_bytes=l0_cap,
    )


simulate_formula_kernel = evaluate_formula_kernel


def resolve_tilelang_dual_tier_bounds(
    *,
    datapath: DatapathCapabilitySpec,
    before_ms: float,
    l_star_ms: float,
    tiled_ms: float,
    ub_live_bytes: int,
    has_cube: bool,
    has_aiv: bool,
    cube_dtype: str = "",
    vector_ops: float = 0.0,
    is_vector_or_norm_region: bool = True,
) -> tuple[float | None, float | None, str | None]:
    """Compute calibrated TileLang ``T.SimtVF`` quick-win vs ``T.SimdVF / AscendC`` ceiling gains."""
    cal = try_load_tilelang_calibration(str(datapath.arch_id))
    if cal is None:
        return None, None, None
    simd_target_ms = max(l_star_ms, tiled_ms)
    ceil_gain_ms = before_ms - l_star_ms
    simd_gain_ms = min(ceil_gain_ms, before_ms - simd_target_ms)
    simt_prof, _ = resolve_tilelang_profile(datapath, uses_simt=True, calibration=cal)
    simt_ub_ceiling = simt_prof.ub_ceiling_bytes
    ub_warn = (
        cal.simt_vs_simd_ub_wall_warning(ub_live_bytes)
        if (is_vector_or_norm_region or has_aiv) else None
    )
    if (is_vector_or_norm_region or has_aiv) and (simt_ub_ceiling <= 0 or ub_live_bytes > simt_ub_ceiling):
        ub_warn = ub_warn or f"simt_ub_ceiling_exceeded:{ub_live_bytes}>{simt_ub_ceiling}"
        return min(0.0, simd_gain_ms), simd_gain_ms, ub_warn

    disp_ms = cal.aclgraph_issue_us / _US_PER_MS
    cube_tl_ms = 0.0
    if has_cube:
        if not cube_dtype:
            raise ModelingError("tilelang_dual_tier_cube_dtype_missing")
        eta_cube = cal.cube_attainment(
            cube_dtype, blockscaled=(cube_dtype.strip().lower() == "mxfp8"),
        )
        cube_tl_ms = tiled_ms / eta_cube
    if has_cube and not has_aiv:
        t_quick_ms = max(simd_target_ms, cube_tl_ms, disp_ms)
    else:
        dma_simt_ms = tiled_ms / cal.dma_attainment(uses_simt=True, rw_pattern="rw21")
        vec_simt_ms = 0.0
        if vector_ops > 0.0:
            aiv_count = max(1, int(datapath.aiv_count))
            simt_ops_per_ms = (
                float(aiv_count)
                * cal.vector_ops_per_s_per_aiv(uses_simt=True, op_kind="ub_fma")
                / _MS_PER_SECOND
            )
            vec_simt_ms = float(vector_ops) / simt_ops_per_ms
        t_quick_ms = max(simd_target_ms, dma_simt_ms, vec_simt_ms, cube_tl_ms, disp_ms)
    raw_simt_gain = before_ms - t_quick_ms
    simt_gain_ms = min(simd_gain_ms, raw_simt_gain)
    return simt_gain_ms, simd_gain_ms, ub_warn
