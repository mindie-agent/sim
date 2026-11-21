"""Exact pair-step periodic CATLASS lowering and cohort-widened execution.

[Role] Lower large double-buffered ``dataparallel`` CATLASS schedules into
  exact pair-step ``PipelineTemplate`` wave descriptors while preserving exact
  per-panel L2 residency, HBM bus accounting, and ring-buffer credit edges.
[Invariants] Every ring credit, L1/L0/L0C allocation lifetime, L2 hit/miss
  byte count, dirty writeback, HBM service term, and stage count matches
  unrolled CATLASS lowering identically; ``PipelineTemplateProgram`` engages
  the Lean-4 proven ``CycleDetector`` without fluid approximation.
[Timeouts & Locks] Finite O(waves * panels) lowering with O(unique phases)
  context rebinding; no I/O, threads, or locks.
[Side-Effect Policy] Lowering mutates only private admission snapshots;
  execution mutates only the caller's Engine.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program import (
    KernelProgram, ParallelProgram, PipelinePhase, PipelineTemplate,
    PipelineTemplateProgram, Stage,
)
from mindie_sim.prediction.runtime.l2_residency import bind_memory_access, cache_policy_for_arch
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, BoundAccess, ByteLedger, ConstraintTerm,
    CounterTerm, HbmBusSpec, MemoryAccessPattern, ReuseKey, SectorCoverage,
    VisibilityContract,
)
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import GATHER_SCATTER_POLICIES, MX_SCALE_GROUP, dtype_name, packed_bytes
from .contracts import (
    FusedRegionSummary, SemanticWorkItem, TemplateLoweringResult,
    _logical_obligation, _split_entry_residency, _union_obligations,
    ceil_div, eligible_matrix_engines,
)
from .fuse_region import Lowering

_PAIR_DEPTH = 2
_L0C_DEPTH = 1
_DOUBLE_BUFFERED_STAGES = (_PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _L0C_DEPTH)
_MIN_PERIODIC_PANELS = 4
_SUPPORTED_MICRO_COUNTS = frozenset({1, _PAIR_DEPTH})
_CANDIDATE_PAIR_PERIODS = (1, 2, 4)
_CANDIDATE_PAIR_PERIODS_MC1 = (1, 2, 4, 8)
_MIN_MULTI_PAIR_REPEATS = 3
_RING_GEN_STAGES = 4
_MICRO_WORK_STAGES = 4
_TILE_FINISH_STAGES = 2
_LAUNCH_STAGES = 1
_OPERAND_PAIR_COUNT = 2
_MS_PER_SECOND = 1000.0
_UNIT_RATE = 1.0
_VIS_READ = VisibilityContract(scope="device", live_out=False, private_workspace=False)
_VIS_WRITE = VisibilityContract(scope="device", live_out=True, private_workspace=False)
_COMPACT_OBLIGATION_THRESHOLD = 4096


def supports_exact_periodic_catlass(spec, p, *, live_admission: bool) -> bool:
    """Return True when ``(spec, p)`` admits an exact pair-step ``PipelineTemplate``."""
    if live_admission or spec.scheduler != "dataparallel" or p.groups:
        return False
    if spec.dispatch_policy == "MmadPingpongMutex" or spec.dispatch_policy in GATHER_SCATTER_POLICIES:
        return False
    if (spec.epilogue_kind != "none" or spec.prologue_a not in {None, "", "none"}
            or spec.prologue_b not in {None, "", "none"} or spec.has_bias or spec.enable_l1_resident):
        return False
    if tuple(spec.stages) != _DOUBLE_BUFFERED_STAGES:
        return False
    if spec.layout_a != "RowMajor" or spec.layout_b != "RowMajor" or spec.layout_c not in {"RowMajor", "ColumnMajor"}:
        return False
    from .catlass_lower import _is_simt_matrix

    if _is_simt_matrix(spec):
        return False
    tk, micro_k = spec.l1_tile_shape[-1], spec.l0_tile_shape[-1]
    if tk <= 0 or micro_k <= 0:
        return False
    return ceil_div(p.k, tk) >= _MIN_PERIODIC_PANELS and ceil_div(tk, micro_k) in _SUPPORTED_MICRO_COUNTS


@dataclass(frozen=True, slots=True)
class _AccessRecord:
    """Exact bound access summary for one operand/scale/output transfer."""
    pattern: MemoryAccessPattern
    ledger: ByteLedger
    writeback_bytes: int = 0


@dataclass(frozen=True, slots=True)
class _PanelDescriptor:
    """Immutable descriptor of one K-panel's bound transfers and compute geometry."""
    tile_m: int
    tile_n: int
    k_len: int
    gm_a: _AccessRecord
    scale_a: _AccessRecord | None
    gm_b: _AccessRecord
    scale_b: _AccessRecord | None


@dataclass(frozen=True, slots=True)
class _StepPhaseDescriptor:
    """Run-length encoded sequence of identical pair-step or trailing single-panel steps."""
    panels: tuple[_PanelDescriptor, ...]
    count: int


@dataclass(frozen=True, slots=True)
class _CohortWaveSpec:
    """One symmetric core cohort inside a CATLASS output wave."""
    cores: tuple[int, ...]
    phases: tuple[_StepPhaseDescriptor, ...]
    drain: _AccessRecord


def _ledger_key(rec: _AccessRecord | None) -> tuple[Any, ...] | None:
    if rec is None:
        return None
    lg = rec.ledger
    return (
        rec.pattern.pipe, rec.pattern.direction, rec.pattern.cache_policy,
        lg.logical_tensor_bytes, lg.aligned_transfer_bytes, lg.l2_hit_bytes,
        lg.dram_miss_copy_bytes, lg.dram_sector_bytes, lg.dram_bus_transaction_bytes, rec.writeback_bytes,
    )


def _panel_key(panel: _PanelDescriptor) -> tuple[Any, ...]:
    return (
        panel.tile_m, panel.tile_n, panel.k_len,
        _ledger_key(panel.gm_a), _ledger_key(panel.scale_a), _ledger_key(panel.gm_b), _ledger_key(panel.scale_b),
    )


def _writeback_stage(
    name: str, amount: int, hbm_bus: HbmBusSpec | None,
    *, after: tuple[str, ...] = (), previous: tuple[str, ...] = (),
) -> Stage:
    if hbm_bus is not None:
        work = {t.row_id: t.ms_per_progress for t in hbm_bus.terms_for("write", amount) if t.ms_per_progress}
        deliveries = ((hbm_bus.ctr_w, amount),)
    else:
        work, deliveries = {"hbm": amount}, ()
    return Stage(name, work, after=after, previous=previous, coupled=len(work) > 1, deliveries=deliveries)


def _rebind_access_stage(
    rec: _AccessRecord, owner: int, context, hbm_bus: HbmBusSpec | None,
    die: int, name: str, *, after: tuple[str, ...] = (), previous: tuple[str, ...] = (),
    acquire: tuple[Any, ...] = (), free: tuple[Any, ...] = (),
) -> Stage:
    """Materialize a ``Stage`` from a pre-admitted ``_AccessRecord`` on ``context`` or unit rates."""
    pattern, ledger = rec.pattern, rec.ledger
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
        memory = getattr(context.datapath, "memory_access", None)
        bus = replace(hbm_bus, memory_access=memory) if (memory is not None and hbm_bus.memory_access is None) else hbm_bus
    else:
        pipe_row, l2_row = pattern.pipe, "l2"
        nom_pipe = hit_rate = miss_rate = l2_rate = _UNIT_RATE
        bus = HbmBusSpec(die, _UNIT_RATE, _UNIT_RATE, 0.0, row_r="hbm", row_w="hbm")

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


def _append_access_stages(
    stages: list[Stage], rec: _AccessRecord, owner: int, context, hbm_bus: HbmBusSpec | None,
    die: int, name: str, *, after: tuple[str, ...] = (), previous: tuple[str, ...] = (),
    acquire: tuple[Any, ...] = (), free: tuple[Any, ...] = (),
) -> None:
    if rec.writeback_bytes > 0:
        wb_name = f"{name}_wb"
        stages.append(_writeback_stage(wb_name, rec.writeback_bytes, hbm_bus, after=after, previous=previous))
        after, previous = (wb_name,), ()
    stages.append(_rebind_access_stage(
        rec, owner, context, hbm_bus, die, name,
        after=after, previous=previous, acquire=acquire, free=free,
    ))


def _widen_stage(stage: Stage, cores: tuple[int, ...]) -> Stage:
    """Widen a single-core ``Stage`` symmetrically across ``cores``."""
    k = len(cores)
    if k <= 1:
        return stage
    acquire = tuple(replace(a, units=cores) for a in stage.acquire)
    free = tuple(replace(a, units=cores) for a in stage.free)
    return replace(
        stage,
        work={r: v * k for r, v in stage.work.items()},
        caps={r: v * k for r, v in stage.caps.items()},
        deliveries=tuple((r, v * k) for r, v in stage.deliveries),
        placement={r: cores for r in stage.placement},
        acquire=acquire,
        free=free,
        cohort_size=k,
        cohort_widened=tuple(stage.placement),
        cohort_members={r: tuple((c,) for c in cores) for r in stage.placement},
    )


def _alloc_or_empty(context, name: str, pool: str, size: int, owner: int) -> tuple[Any, ...]:
    if context is None or size <= 0 or pool not in context.pool_names:
        return ()
    return (context.allocate(name, pool, size, (owner,)),)


def _panel_tag(s_idx: int, total_panels: int) -> str:
    """Stable suffix so the final 2 panels always expose ``p0`` and ``p1`` across phases."""
    if total_panels == 1:
        return "p0"
    if s_idx >= total_panels - _PAIR_DEPTH:
        return f"p{s_idx - (total_panels - _PAIR_DEPTH)}"
    return f"s{s_idx}"


def _append_panel_micro_stages(
    stages: list[Stage], panel: _PanelDescriptor, s_idx: int, total_panels: int, full_mc: int,
    a_ready: str, b_ready: str, l1_allocs: tuple[Any, ...],
    spec, datapath, context, owner: int, l0_sizes: tuple[int, int],
) -> None:
    from .catlass_lower import _compute_b, _cube_dtype, _issued_ops, _scale

    l0a_sz, l0b_sz = l0_sizes
    micro_k = spec.l0_tile_shape[-1]
    full_last = full_mc - 1
    cube_dt = _cube_dtype(spec)
    aic_count = datapath.aic_count
    cur = _panel_tag(s_idx, total_panels)
    prev1 = _panel_tag(s_idx - 1, total_panels) if s_idx >= 1 else ""
    prev2 = _panel_tag(s_idx - _PAIR_DEPTH, total_panels) if s_idx >= _PAIR_DEPTH else ""
    micros = list(range(0, panel.k_len, micro_k))
    last_u = len(micros) - 1
    for u, start in enumerate(micros):
        micro = min(panel.k_len - start, micro_k)
        l0a_alloc = _alloc_or_empty(context, f"l0a_{u}_{cur}", "l0a", l0a_sz, owner)
        l0b_alloc = _alloc_or_empty(context, f"l0b_{u}_{cur}", "l0b", l0b_sz, owner)
        if full_mc == _PAIR_DEPTH:
            l0_after_a = (a_ready, f"mma_{u}_{prev1}") if s_idx >= 1 else (a_ready,)
            l0_after_b = (b_ready, f"mma_{u}_{prev1}") if s_idx >= 1 else (b_ready,)
            l0_prev = (f"mma_{u}_p1",) if s_idx == 0 else ()
        elif s_idx < _PAIR_DEPTH:
            l0_after_a, l0_after_b, l0_prev = (a_ready,), (b_ready,), (f"mma_0_p{s_idx}",)
        else:
            l0_after_a, l0_after_b, l0_prev = (a_ready, f"mma_0_{prev2}"), (b_ready, f"mma_0_{prev2}"), ()
        ma = packed_bytes(panel.tile_m * micro, spec.dtype_a) + _scale(spec, panel.tile_m * micro)
        mb = packed_bytes(panel.tile_n * micro, _compute_b(spec)) + _scale(spec, panel.tile_n * micro)
        if context is not None:
            wa, wb = context.bind(Work(resources={"mte1": ma})), context.bind(Work(resources={"mte1": mb}))
            mte1_pl = {k: v for k, v in context.owned(owner, aic_count, ("mte1",)).items() if k in wa or k in wb}
        else:
            wa, wb, mte1_pl = {"mte1": ma}, {"mte1": mb}, {"mte1": (owner,)}
        stages.append(Stage(f"l0a_{u}_{cur}", wa, after=l0_after_a, previous=l0_prev,
                            placement=mte1_pl, acquire=l0a_alloc))
        stages.append(Stage(f"l0b_{u}_{cur}", wb, after=l0_after_b, previous=l0_prev,
                            placement=mte1_pl, acquire=l0b_alloc))
        l1_free = l1_allocs if u == last_u else ()
        stages.append(Stage(f"loaded_{u}_{cur}", after=(f"l0a_{u}_{cur}", f"l0b_{u}_{cur}"), free=l1_free))
        issued = _issued_ops(panel.tile_m, panel.tile_n, micro, spec, datapath)
        if context is not None:
            wc = context.bind(Work(matrix={cube_dt: issued}))
            cube_pl = {k: v for k, v in context.owned(owner, aic_count, ("cube",)).items() if k in wc}
        else:
            wc, cube_pl = {"cube": issued}, {"cube": (owner,)}
        if u == 0 and s_idx == 0:
            mma_after, mma_prev = (f"loaded_0_{cur}",), (f"mma_{full_last}_p1",)
        elif u == 0:
            mma_after, mma_prev = (f"mma_{full_last}_{prev1}", f"loaded_0_{cur}"), ()
        else:
            mma_after, mma_prev = (f"mma_{u - 1}_{cur}", f"loaded_{u}_{cur}"), ()
        stages.append(Stage(
            f"mma_{u}_{cur}", wc, after=mma_after, previous=mma_prev, placement=cube_pl,
            free=l0a_alloc + l0b_alloc,
            metadata={"logical_flops": _OPERAND_PAIR_COUNT * panel.tile_m * panel.tile_n * micro},
        ))


def _build_step_stages(
    panels: tuple[_PanelDescriptor, ...], spec, datapath, context, hbm_bus: HbmBusSpec | None,
    die: int, owner: int, slot_sizes: tuple[int, int, int, int],
) -> tuple[Stage, ...]:
    """Build exact multi-panel (even period) or trailing single-panel stages for ``PipelinePhase``."""
    l1a_sz, l1b_sz, l0a_sz, l0b_sz = slot_sizes
    full_mc = ceil_div(spec.l1_tile_shape[-1], spec.l0_tile_shape[-1])
    full_last = full_mc - 1
    total_panels = len(panels)
    stages: list[Stage] = []
    for s_idx, panel in enumerate(panels):
        cur = _panel_tag(s_idx, total_panels)
        l1a_alloc = _alloc_or_empty(context, f"l1a_{cur}", "l1", l1a_sz, owner)
        l1b_alloc = _alloc_or_empty(context, f"l1b_{cur}", "l1", l1b_sz, owner)
        if s_idx < _PAIR_DEPTH:
            gm_after, gm_prev = (), (f"loaded_{full_last}_p{s_idx}",)
        else:
            prev2 = _panel_tag(s_idx - _PAIR_DEPTH, total_panels)
            gm_after, gm_prev = (f"loaded_{full_last}_{prev2}",), ()
        _append_access_stages(
            stages, panel.gm_a, owner, context, hbm_bus, die, f"gm_a_{cur}",
            after=gm_after, previous=gm_prev, acquire=l1a_alloc,
        )
        a_ready = f"gm_a_{cur}"
        if panel.scale_a is not None:
            a_ready = f"scale_a_{cur}"
            _append_access_stages(
                stages, panel.scale_a, owner, context, hbm_bus, die, a_ready, after=(f"gm_a_{cur}",),
            )
        _append_access_stages(
            stages, panel.gm_b, owner, context, hbm_bus, die, f"gm_b_{cur}",
            after=gm_after, previous=gm_prev, acquire=l1b_alloc,
        )
        b_ready = f"gm_b_{cur}"
        if panel.scale_b is not None:
            b_ready = f"scale_b_{cur}"
            _append_access_stages(
                stages, panel.scale_b, owner, context, hbm_bus, die, b_ready, after=(f"gm_b_{cur}",),
            )
        _append_panel_micro_stages(
            stages, panel, s_idx, total_panels, full_mc, a_ready, b_ready, l1a_alloc + l1b_alloc,
            spec, datapath, context, owner, (l0a_sz, l0b_sz),
        )
    return tuple(stages)


@dataclass(frozen=True)
class _PeriodicCatlassProgram:
    """Exact pair-step periodic CATLASS program backed by ``PipelineTemplateProgram``."""

    spec: Any
    problem: Any
    datapath: Any
    context: Any
    die: int
    hbm_bus: HbmBusSpec | None
    waves: tuple[tuple[_CohortWaveSpec, ...], ...]
    slot_sizes: tuple[int, int, int, int, int]
    burst_rules: tuple[Any, ...]
    summary: FusedRegionSummary
    resource_work: tuple[tuple[str, float], ...] = ()

    def with_context(self, datapath, context) -> "_PeriodicCatlassProgram":
        """Rebind the compact wave/phase descriptors to ``context`` in O(unique phases)."""
        probe = Lowering(datapath, context=context, die=self.die)
        if (
            dict(datapath.buffers).get("l2", 0) != dict(self.datapath.buffers).get("l2", 0)
            or datapath.memory_access != self.datapath.memory_access
            or probe.burst_rules != self.burst_rules
        ):
            from .catlass_lower import solve_catlass_tile

            sol = solve_catlass_tile(self.spec, self.problem, datapath, die=self.die)
            return lower_catlass_periodic(
                self.spec, self.problem, datapath, sol,
                context=context, die=self.die, region_id=self.summary.region_id,
            ).program
        return replace(self, datapath=datapath, context=context, hbm_bus=probe.hbm_bus)

    def start(self, engine, name: str, ready, *, owner: str = ""):
        bus = self.hbm_bus
        if bus is not None:
            present = (bus.row_r in engine.resources, bus.row_w in engine.resources)
            if not any(present):
                engine.add_hbm_bus(bus)
            elif all(present):
                engine.bind_hbm_bus(bus)
            else:
                raise ModelingError("template_directional_hbm_rows_incomplete")
        stars = self.datapath.stars_spec
        launch_lat = ((stars.issue_interval_us + stars.first_issue_us) / _MS_PER_SECOND
                      if stars is not None else 0.0)
        stages: list[Stage] = [Stage("launch", latency=launch_lat, metadata={"stars_launch": True})]
        prev_stage = "launch"
        l1_l0_sizes, l0c_sz = self.slot_sizes[:4], self.slot_sizes[4]
        for w_idx, wave_cohorts in enumerate(self.waves):
            progs = []
            for cohort in wave_cohorts:
                c0 = cohort.cores[0]
                l0c_alloc = _alloc_or_empty(self.context, "l0c", "l0c", l0c_sz, c0)
                setup = ((_widen_stage(Stage("l0c_acq", acquire=l0c_alloc, keep_empty=bool(l0c_alloc)), cohort.cores),)
                         if l0c_alloc else ())
                phases = tuple(
                    PipelinePhase(
                        f"p{p_idx}",
                        tuple(
                            _widen_stage(st, cohort.cores)
                            for st in _build_step_stages(
                                ph.panels, self.spec, self.datapath, self.context,
                                self.hbm_bus, self.die, c0, l1_l0_sizes,
                            )
                        ),
                        ph.count,
                    )
                    for p_idx, ph in enumerate(cohort.phases)
                )
                drain_stages: list[Stage] = []
                _append_access_stages(
                    drain_stages, cohort.drain, c0, self.context, self.hbm_bus, self.die,
                    "fixpipe", free=l0c_alloc,
                )
                drain = tuple(_widen_stage(st, cohort.cores) for st in drain_stages)
                progs.append(PipelineTemplateProgram(
                    PipelineTemplate(owner=c0, slots=_PAIR_DEPTH, phases=phases, setup=setup, drain=drain),
                ))
            w_name = f"wave_{w_idx}"
            stages.append(Stage(w_name, after=(prev_stage,), program=ParallelProgram(tuple(progs))))
            prev_stage = w_name
        return KernelProgram(tuple(stages)).start(engine, name, ready, owner=owner)


def _make_rect_pattern(
    spec, p, die: int, sector: int, tensor: str, dtype: str,
    row: int, col: int, rows: int, cols: int, width: int, owner: int,
    *, bypass: bool = False,
) -> MemoryAccessPattern:
    stride = packed_bytes(width, dtype)
    rb = packed_bytes(col + cols, dtype) - packed_bytes(col, dtype)
    if rb != packed_bytes(cols, dtype):
        raise ModelingError("catlass_packed_tile_unaligned")
    base = p.b_base_byte if tensor == p.b_id else 0
    version = p.b_version if tensor == p.b_id else 0
    run = AddressRun(base + row * stride + packed_bytes(col, dtype), rows, rb, stride, "RowMajor")
    key = None if bypass else ReuseKey(die, tensor, version, reuse_stream=f"{tensor}:{version}")
    return MemoryAccessPattern(
        tensor, version, die, owner, "read", "mte2_aic", rb * rows,
        AddressRunPlan((run,), sector_bytes=sector), key, None, _VIS_READ,
    )


def _make_scale_pattern(
    p, die: int, sector: int, tensor: str, row: int, col: int, rows: int, cols: int,
    width: int, owner: int, *, is_a: bool, bypass: bool = False,
) -> MemoryAccessPattern:
    per_row = ceil_div(width, MX_SCALE_GROUP) if is_a else ceil_div(p.k, MX_SCALE_GROUP)
    sr, sc = (rows, ceil_div(cols, MX_SCALE_GROUP)) if is_a else (cols, ceil_div(rows, MX_SCALE_GROUP))
    off = row * per_row + col // MX_SCALE_GROUP if is_a else col * per_row + row // MX_SCALE_GROUP
    tid = tensor + ":scale"
    key = None if bypass else ReuseKey(die, tid, 0, reuse_stream=f"{tid}:0")
    return MemoryAccessPattern(
        tid, 0, die, owner, "read", "mte2_aic", sr * sc,
        AddressRunPlan((AddressRun(off, sr, sc, per_row),), sector_bytes=sector), key, None, _VIS_READ,
    )


def _make_output_pattern(spec, p, tile, die: int, sector: int) -> MemoryAccessPattern:
    dtype = spec.dtype_c
    if spec.layout_c == "RowMajor":
        stride = packed_bytes(p.n, dtype)
        rb = packed_bytes(tile.n, dtype)
        run = AddressRun(tile.row * stride + packed_bytes(tile.col, dtype), tile.m, rb, stride)
        amt = rb * tile.m
    else:
        stride = packed_bytes(p.m, dtype)
        cb = packed_bytes(tile.m, dtype)
        run = AddressRun(tile.col * stride + packed_bytes(tile.row, dtype), tile.n, cb, stride)
        amt = cb * tile.n
    key = ReuseKey(die, p.c_id, 1, reuse_stream=f"{p.c_id}:1")
    return MemoryAccessPattern(
        p.c_id, 1, die, tile.owner, "write", "fixpipe", amt,
        AddressRunPlan((run,), sector_bytes=sector), key, None, _VIS_WRITE,
    )


def _exact_tile_stage_count(spec, gk: int, total_micros: int, has_ctx: bool) -> int:
    """Exact unrolled stage count for one double-buffered dataparallel ordinary tile."""
    fills_per_panel = _OPERAND_PAIR_COUNT * (1 + int(bool(spec.mx_scale_group)))
    ring_gens = (gk, gk, total_micros, total_micros, _L0C_DEPTH)
    ring_stages = 0
    for depth, gens in zip(spec.stages, ring_gens):
        slots_used = min(depth, gens)
        ring_stages += slots_used + _RING_GEN_STAGES * gens + (slots_used if has_ctx else 0)
    return fills_per_panel * gk + _MICRO_WORK_STAGES * total_micros + _TILE_FINISH_STAGES + ring_stages


def _encode_panel_phases(
    panels: list[_PanelDescriptor],
    *,
    periods: tuple[int, ...] = _CANDIDATE_PAIR_PERIODS,
) -> tuple[tuple[_StepPhaseDescriptor, ...], tuple[Any, ...]]:
    """Run-length encode repeating 2/4/8/16-panel periods (and optional trailing panel) into phases."""
    pair_end = (len(panels) // _PAIR_DEPTH) * _PAIR_DEPTH
    pairs = [(panels[i], panels[i + 1]) for i in range(0, pair_end, _PAIR_DEPTH)]
    sigs = [(_panel_key(p0), _panel_key(p1)) for p0, p1 in pairs]
    steps: list[_StepPhaseDescriptor] = []
    step_sigs: list[tuple[Any, ...]] = []
    pos, total = 0, len(pairs)
    while pos < total:
        best_p, best_rep, best_cov = 1, 1, 1
        for period in periods:
            if pos + period > total:
                break
            pat = sigs[pos:pos + period]
            rep = 1
            while pos + (rep + 1) * period <= total and sigs[pos + rep * period:pos + (rep + 1) * period] == pat:
                rep += 1
            min_rep = 1 if period == 1 else _MIN_MULTI_PAIR_REPEATS
            cov = period * rep
            if rep >= min_rep and cov > best_cov:
                best_p, best_rep, best_cov = period, rep, cov
        flat = tuple(pd for pair_t in pairs[pos:pos + best_p] for pd in pair_t)
        flat_sig = tuple(sigs[pos:pos + best_p])
        steps.append(_StepPhaseDescriptor(flat, best_rep))
        step_sigs.append((flat_sig, best_rep))
        pos += best_cov
    if pair_end < len(panels):
        tail_t = (panels[pair_end],)
        steps.append(_StepPhaseDescriptor(tail_t, 1))
        step_sigs.append(((_panel_key(tail_t[0]),), 1))
    return tuple(steps), tuple(step_sigs)


def _whole_tensor_obligations(spec, p, die: int, sector: int) -> tuple[list[Any], list[Any]]:
    """Exact full-tensor logical obligations when entry L2 state is empty."""
    reads: list[Any] = []
    a_stride, b_stride = packed_bytes(p.k, spec.dtype_a), packed_bytes(p.n, spec.dtype_b)
    def _pat(tid, ver, dir_, pipe, base, rows, stride, vis):
        return MemoryAccessPattern(
            tid, ver, die, 0, dir_, pipe, rows * stride,
            AddressRunPlan((AddressRun(base, rows, stride, stride, "RowMajor"),), sector_bytes=sector),
            None, None, vis,
        )
    for pat in (
        _pat(p.a_id, 0, "read", "mte2_aic", 0, p.m, a_stride, _VIS_READ),
        _pat(p.b_id, p.b_version, "read", "mte2_aic", p.b_base_byte, p.k, b_stride, _VIS_READ),
    ):
        if ob := _logical_obligation(pat):
            reads.append(ob)
    if spec.mx_scale_group:
        sa_w = ceil_div(p.k, MX_SCALE_GROUP)
        for tid, rows in ((p.a_id + ":scale", p.m), (p.b_id + ":scale", p.n)):
            if ob := _logical_obligation(_pat(tid, 0, "read", "mte2_aic", 0, rows, sa_w, _VIS_READ)):
                reads.append(ob)
    c_stride, c_rows = (
        (packed_bytes(p.n, spec.dtype_c), p.m) if spec.layout_c == "RowMajor"
        else (packed_bytes(p.m, spec.dtype_c), p.n)
    )
    writes = [ob] if (ob := _logical_obligation(_pat(p.c_id, 1, "write", "fixpipe", 0, c_rows, c_stride, _VIS_WRITE))) else []
    return reads, writes


def _admit_periodic_access(
    pat: MemoryAccessPattern, plan: Lowering, entry_state, unit_bus: HbmBusSpec,
    retained_keys: dict[ReuseKey, tuple[ReuseKey, bool]], resource_work: Counter[str],
    context, die: int, *, track_obligations: bool = True,
) -> _AccessRecord:
    if track_obligations and (entry := _logical_obligation(pat)):
        if pat.direction == "read":
            ent_pat = (replace(pat, reuse_key=entry_state.resolve_reuse_key(pat.reuse_key))
                       if pat.reuse_key is not None else pat)
            res_entry = entry_state.resident_coverage_for(ent_pat, pat.runs.aligned_copy_bytes())
            plan.reads.extend(_split_entry_residency(entry, res_entry))
            if len(plan.reads) > _COMPACT_OBLIGATION_THRESHOLD:
                plan.reads = list(_union_obligations(plan.reads))
        else:
            plan.writes.append(entry)
            if len(plan.writes) > _COMPACT_OBLIGATION_THRESHOLD:
                plan.writes = list(_union_obligations(plan.writes))
    if pat.reuse_key is not None:
        if pat.reuse_key not in retained_keys:
            rk = plan.state.resolve_reuse_key(pat.reuse_key)
            retained_keys[pat.reuse_key] = (rk, plan.state.retain_template_key(rk))
        pat = replace(pat, reuse_key=retained_keys[pat.reuse_key][0])
    bound = bind_memory_access(
        pat, plan.state, hbm_bus=unit_bus, copy_hit_bytes_per_ms=_UNIT_RATE,
        copy_miss_bytes_per_ms=_UNIT_RATE, l2_bytes_per_ms=_UNIT_RATE,
        pipe_row_id=pat.pipe, l2_row_id="l2", burst_rules=plan.burst_rules, commit=True,
    )
    lg = bound.ledger
    res_prior = (
        getattr(plan.state, "_last_prior_coverage", SectorCoverage())
        if (lg.l2_hit_bytes and lg.dram_sector_bytes) else SectorCoverage()
    )
    plan._account(lg, pat.runs, res_prior)
    evictions = plan.state.drain_writebacks(die)
    wb_bytes = sum(item.bytes for item in evictions) if evictions else 0
    if wb_bytes:
        plan.sectors += wb_bytes
        plan.service += wb_bytes
        resource_work["hbm"] += wb_bytes
    if lg.aligned_transfer_bytes:
        resource_work[pat.pipe] += lg.aligned_transfer_bytes
    if lg.l2_hit_bytes and (context is None or "l2" in context.rates):
        resource_work["l2"] += lg.l2_hit_bytes
    if lg.dram_bus_transaction_bytes > 0.0:
        resource_work["hbm"] += lg.dram_bus_transaction_bytes
    return _AccessRecord(pat, lg, wb_bytes)


def _lower_periodic_wave(
    tiles: tuple[Any, ...], spec, p, datapath, low, plan: Lowering,
    admit, admit_shared_a, resource_work: Counter[str], cube_dt: str, die: int, has_ctx: bool,
) -> tuple[tuple[_CohortWaveSpec, ...], int]:
    """Admit all K-panels for one output wave and group identical cores into cohorts."""
    from .catlass_lower import _compute_b, _issued_ops, _scale

    bypass_b = spec.l2_cache_hint_b == "disable"
    use_scale = bool(spec.mx_scale_group)
    per_tile_panels: list[list[_PanelDescriptor]] = [[] for _ in tiles]
    for panel_idx in range(low.gk):
        ks = panel_idx * low.tk
        kl = min(low.tk, p.k - ks)
        for t_i, tile in enumerate(tiles):
            gm_a = admit_shared_a(
                (False, tile.row, tile.m, ks, kl), tile.owner,
                lambda _t=tile, _ks=ks, _kl=kl: _make_rect_pattern(
                    spec, p, die, plan.sector, p.a_id, spec.dtype_a,
                    _t.row, _ks, _t.m, _kl, p.k, _t.owner,
                ),
            )
            sc_a = (admit_shared_a(
                (True, tile.row, tile.m, ks, kl), tile.owner,
                lambda _t=tile, _ks=ks, _kl=kl: _make_scale_pattern(
                    p, die, plan.sector, p.a_id, _t.row, _ks, _t.m, _kl, p.k, _t.owner, is_a=True,
                ),
            ) if use_scale else None)
            gm_b = admit(_make_rect_pattern(
                spec, p, die, plan.sector, p.b_id, spec.dtype_b,
                ks, tile.col, kl, tile.n, p.n, tile.owner, bypass=bypass_b,
            ))
            sc_b = (admit(_make_scale_pattern(
                p, die, plan.sector, p.b_id, ks, tile.col, kl, tile.n, p.n, tile.owner,
                is_a=False, bypass=bypass_b,
            )) if use_scale else None)
            per_tile_panels[t_i].append(_PanelDescriptor(tile.m, tile.n, kl, gm_a, sc_a, gm_b, sc_b))
            for start in range(0, kl, low.micro_k):
                micro = min(kl - start, low.micro_k)
                ma = packed_bytes(tile.m * micro, spec.dtype_a) + _scale(spec, tile.m * micro)
                mb = packed_bytes(tile.n * micro, _compute_b(spec)) + _scale(spec, tile.n * micro)
                resource_work["mte1"] += ma + mb
                plan.cube[cube_dt] += _issued_ops(tile.m, tile.n, micro, spec, datapath)

    periods = _CANDIDATE_PAIR_PERIODS_MC1 if ceil_div(low.tk, low.micro_k) == 1 else _CANDIDATE_PAIR_PERIODS
    cohorts: dict[tuple[Any, ...], list[tuple[int, tuple[_StepPhaseDescriptor, ...], _AccessRecord]]] = {}
    wave_stages = 0
    for t_i, tile in enumerate(tiles):
        drain_rec = admit(_make_output_pattern(spec, p, tile, die, plan.sector))
        p_list = per_tile_panels[t_i]
        total_micros = sum(ceil_div(pd.k_len, low.micro_k) for pd in p_list)
        wave_stages += _exact_tile_stage_count(spec, low.gk, total_micros, has_ctx)
        steps, step_sigs = _encode_panel_phases(p_list, periods=periods)
        cohorts.setdefault((step_sigs, _ledger_key(drain_rec)), []).append((tile.owner, steps, drain_rec))
    wave_specs = tuple(
        _CohortWaveSpec(tuple(i[0] for i in items), items[0][1], items[0][2])
        for items in cohorts.values()
    )
    return wave_specs, wave_stages


def _fast_hit_shared_a(
    hit_item: tuple[_AccessRecord, ReuseKey, Any], owner: int,
    plan: Lowering, resource_work: Counter[str], context, die: int,
) -> _AccessRecord | None:
    base_rec, rk, entry = hit_item
    eff = plan.state.effective_l2_bytes(die)
    if (plan.state._resident.get(rk) is not entry or entry.evicted or entry.working_set_bytes > eff
            or plan.state._hit_fraction(entry.working_set_bytes, eff) != _UNIT_RATE):
        return None
    plan.state._activate_key(rk)
    plan.state._touch_resident(entry)
    aligned = base_rec.ledger.aligned_transfer_bytes
    lg = ByteLedger(base_rec.ledger.logical_tensor_bytes, aligned, 0.0, aligned, 0, 0)
    plan.logical += lg.logical_tensor_bytes
    plan.copied += aligned
    plan.hits += aligned
    if aligned:
        resource_work[base_rec.pattern.pipe] += aligned
        if context is None or "l2" in context.rates:
            resource_work["l2"] += aligned
    return _AccessRecord(replace(base_rec.pattern, owner=owner), lg, 0)


def lower_catlass_periodic(
    spec, p, datapath, solution, *,
    context=None, memory_view=None, die: int = 0, region_id=None,
) -> TemplateLoweringResult:
    """Lower an eligible double-buffered ``dataparallel`` CATLASS matmul into exact periodic waves."""
    from .catlass_lower import _MatrixLowerer, _compute_b, _cube_dtype, _scale

    plan = Lowering(datapath, context=context, memory_view=memory_view, die=die, live_admission=False)
    entry_state = deepcopy(plan.state)
    low, unit_bus = _MatrixLowerer(plan, spec, p), HbmBusSpec(die, _UNIT_RATE, _UNIT_RATE, 0.0, row_r="hbm", row_w="hbm")
    resource_work, cube_dt = Counter(), _cube_dtype(spec)
    blocks, cores = ceil_div(p.m, low.tm) * ceil_div(p.n, low.tn), spec.aic_budget
    retained_keys: dict[ReuseKey, tuple[ReuseKey, bool]] = {}
    entry_empty = not getattr(entry_state, "_resident", None)
    if entry_empty:
        whole_reads, whole_writes = _whole_tensor_obligations(spec, p, die, plan.sector)
        plan.reads.extend(whole_reads)
        plan.writes.extend(whole_writes)

    def admit(pat: MemoryAccessPattern) -> _AccessRecord:
        return _admit_periodic_access(
            pat, plan, entry_state, unit_bus, retained_keys, resource_work, context, die,
            track_obligations=not entry_empty,
        )

    a_hits: dict[tuple[Any, ...], tuple[_AccessRecord, ReuseKey, Any]] = {}

    def admit_shared_a(cache_key: tuple[Any, ...], owner: int, make_pat) -> _AccessRecord:
        if entry_empty and (hit_item := a_hits.get(cache_key)) is not None:
            if (fast := _fast_hit_shared_a(hit_item, owner, plan, resource_work, context, die)) is not None:
                return fast
        pat = make_pat()
        rec = admit(pat)
        rk_info = retained_keys.get(pat.reuse_key) if (entry_empty and pat.reuse_key is not None) else None
        if rk_info is not None and (entry := plan.state._resident.get(rk_info[0])) is not None and not entry.evicted:
            a_hits[cache_key] = (rec, rk_info[0], entry)
        return rec

    compiled_waves: list[tuple[_CohortWaveSpec, ...]] = []
    total_stages, plan.launches = _LAUNCH_STAGES, 1
    for first in range(0, blocks, cores):
        tiles = tuple(low.tile(i) for i in range(first, min(first + cores, blocks)))
        wave_specs, wave_stages = _lower_periodic_wave(
            tiles, spec, p, datapath, low, plan, admit, admit_shared_a, resource_work, cube_dt, die, context is not None,
        )
        compiled_waves.append(wave_specs)
        total_stages += wave_stages
    for rk, owned in retained_keys.values():
        if owned:
            plan.state.release_template_key(rk)
    l1m, l1n, l1k = spec.l1_tile_shape
    l0m, l0n, l0k = spec.l0_tile_shape
    slot_sizes = (
        packed_bytes(l1m * l1k, spec.dtype_a) + _scale(spec, l1m * l1k, l1=True),
        packed_bytes(l1n * l1k, _compute_b(spec)) + _scale(spec, l1n * l1k, l1=True),
        packed_bytes(l0m * l0k, spec.dtype_a) + _scale(spec, l0m * l0k),
        packed_bytes(l0n * l0k, _compute_b(spec)) + _scale(spec, l0n * l0k),
        packed_bytes(l0m * l0n, spec.dtype_acc),
    )
    rid = region_id or spec.template_id
    semantic = (SemanticWorkItem(
        rid + ":matrix", "matrix_mac", cube_dt, _OPERAND_PAIR_COUNT * p.m * p.n * p.k,
        eligible_matrix_engines(p, datapath), (spec.scheduler,), die=die,
    ),)
    summary = FusedRegionSummary(
        "K2", rid, (rid,), semantic, tuple(sorted(plan.cube.items())), tuple(sorted(plan.vector.items())),
        _union_obligations(plan.reads), _union_obligations(plan.writes),
        plan.logical, plan.copied, plan.sectors, plan.service, plan.hits, (), plan.launches,
        plan.metadata_ms, tuple((peak.pool, peak.peak_bytes) for peak in solution.pool_peaks), spec,
        float(plan.service), plan.misses,
    )
    program = _PeriodicCatlassProgram(
        spec, p, datapath, context, die, plan.hbm_bus, tuple(compiled_waves), slot_sizes,
        plan.burst_rules, summary, tuple(sorted(resource_work.items())),
    )
    return TemplateLoweringResult(program, summary, total_stages)
