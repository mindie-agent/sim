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

from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from fractions import Fraction
from math import gcd
from typing import Any

from mindie_sim.prediction.mechanism import Work
from mindie_sim.prediction.program.pipeline import (
    PipelinePhase, PipelineTemplate, PipelineTemplateProgram,
)
from mindie_sim.prediction.program.stages import KernelProgram, ParallelProgram, Stage
from mindie_sim.prediction.runtime.l2_residency import _access_plan, admit_memory_ledger
from mindie_sim.prediction.runtime.memory_physics import (
    AddressRun, AddressRunPlan, ByteLedger, HbmBusSpec, MemoryAccessPattern, ReuseKey, SectorCoverage,
    VisibilityContract,
)
from mindie_sim.schema.common import ModelingError

from .catlass_catalog import (
    GATHER_SCATTER_POLICIES, MX_SCALE_GROUP, STREAMK_PARTIAL_SLOTS_PER_CORE,
    dtype_name, packed_bytes, streamk_partition,
)
from .contracts import (
    FusedRegionSummary, SemanticWorkItem, TemplateLoweringResult,
    _logical_obligation, _split_entry_residency, _union_obligations,
    ceil_div, eligible_matrix_engines,
)
from .fuse_region import Lowering, bind_template_access_ledger
from .fuse_region_live import _ReuseLifetimes

_PAIR_DEPTH = 2
_L0C_DEPTH = 1
_DOUBLE_BUFFERED_STAGES = (_PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _L0C_DEPTH)
_MUTEX_DOUBLE_BUFFERED_STAGES = (_PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH, _PAIR_DEPTH)
_MIN_PERIODIC_PANELS = 4
_SUPPORTED_MICRO_COUNTS = frozenset({1, _PAIR_DEPTH})
_CANDIDATE_PAIR_PERIODS = (1, 2, 4)
_CANDIDATE_PAIR_PERIODS_MC1 = (1, 2, 4, 8)
_MIN_MULTI_PAIR_REPEATS = 3
_RING_GEN_STAGES = 4
_MICRO_WORK_STAGES = 4
_TILE_FINISH_STAGES = 2
_LAUNCH_STAGES = 1
_FLAG_STAGE_COUNT = 1
_REDUCE_OVERHEAD_STAGES = 2
_OPERAND_PAIR_COUNT = 2
_MS_PER_SECOND = 1000.0
_VIS_READ = VisibilityContract(scope="device", live_out=False, private_workspace=False)
_VIS_WRITE = VisibilityContract(scope="device", live_out=True, private_workspace=False)
_VIS_PRIVATE = VisibilityContract(scope="die", live_out=False, private_workspace=True)
_COMPACT_OBLIGATION_THRESHOLD = 4096
_DEFAULT_ALIGN_BYTES = 32
_PERIODIC_SCHEDULERS = frozenset({"dataparallel", "streamk", "multi_core_splitk"})


def supports_exact_periodic_catlass(spec, p, *, live_admission: bool) -> bool:
    """Return True when ``(spec, p)`` admits an exact pair-step ``PipelineTemplate``."""
    if live_admission or spec.scheduler not in _PERIODIC_SCHEDULERS or p.groups:
        return False
    if spec.dispatch_policy in GATHER_SCATTER_POLICIES:
        return False
    if (spec.epilogue_kind != "none" or spec.prologue_a not in {None, "", "none"}
            or spec.prologue_b not in {None, "", "none"} or spec.has_bias or spec.enable_l1_resident):
        return False
    is_mutex = spec.dispatch_policy == "MmadPingpongMutex"
    if is_mutex and spec.scheduler != "dataparallel":
        return False
    valid_stages = (
        (_DOUBLE_BUFFERED_STAGES, _MUTEX_DOUBLE_BUFFERED_STAGES)
        if is_mutex else (_DOUBLE_BUFFERED_STAGES,)
    )
    if tuple(spec.stages) not in valid_stages:
        return False
    if spec.layout_a != "RowMajor" or spec.layout_b != "RowMajor" or spec.layout_c not in {"RowMajor", "ColumnMajor"}:
        return False
    from .catlass_lower import _is_simt_matrix

    if _is_simt_matrix(spec):
        return False
    tk, micro_k = spec.l1_tile_shape[-1], spec.l0_tile_shape[-1]
    if tk <= 0 or micro_k <= 0:
        return False
    gk = ceil_div(p.k, tk)
    if spec.scheduler == "multi_core_splitk" and (spec.splitk_factor < 1 or gk < spec.splitk_factor):
        return False
    return gk >= _MIN_PERIODIC_PANELS and ceil_div(tk, micro_k) in _SUPPORTED_MICRO_COUNTS


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
    k_start_mod: int = 0


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


@dataclass(frozen=True, slots=True)
class _StreamkPartialCohortSpec:
    """Symmetric core cohort sharing identical sequential StreamK partial K-slices."""
    cores: tuple[int, ...]
    parts: tuple[tuple[tuple[_StepPhaseDescriptor, ...], _AccessRecord], ...]


@dataclass(frozen=True, slots=True)
class _SplitkReduceSpec:
    """One AIV block reduction fusing partial accumulators into final C."""
    block: int
    owner: int
    partial_reads: tuple[_AccessRecord, ...]
    reduce_ops: int
    output: _AccessRecord


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


def _append_access_stages(
    stages: list[Stage], rec: _AccessRecord, owner: int, context, hbm_bus: HbmBusSpec | None,
    name: str, *, after: tuple[str, ...] = (), previous: tuple[str, ...] = (),
    acquire: tuple[Any, ...] = (), free: tuple[Any, ...] = (),
) -> None:
    if rec.writeback_bytes > 0:
        wb_name = f"{name}_wb"
        stages.append(_writeback_stage(wb_name, rec.writeback_bytes, hbm_bus, after=after, previous=previous))
        after, previous = (wb_name,), ()
    stages.append(bind_template_access_ledger(
        rec.pattern, rec.ledger, owner, context, hbm_bus, name,
        after=after, previous=previous, acquire=acquire, free=free,
    ))


_OWNED_PERIODIC_ROLES = ("cube", "mte1", "mte2", "mte2_aic", "fixpipe")
_OWNED_AIV_ROLES = ("mte2", "mte2_aiv", "mte3", "vec_reduce", "vector")


def _split_colliding_cores(
    cores: tuple[int, ...], context, axis_count: int, roles: tuple[str, ...] = _OWNED_PERIODIC_ROLES,
) -> tuple[tuple[int, ...], ...]:
    """Split a cohort whenever multiple cores map onto a shared physical resource unit."""
    if context is None or len(cores) <= 1:
        return (cores,)
    per_core = [context.owned(c, axis_count, roles) for c in cores]
    for r in {name for mapping in per_core for name in mapping}:
        seen: set[int] = set()
        for mapping in per_core:
            units = mapping.get(r, ())
            if seen.intersection(units):
                return tuple((c,) for c in cores)
            seen.update(units)
    return (cores,)


def _widen_stage(
    stage: Stage, cores: tuple[int, ...], context=None, axis_count: int | None = None,
    roles: tuple[str, ...] = _OWNED_PERIODIC_ROLES,
) -> Stage:
    """Widen a single-core ``Stage`` symmetrically across ``cores``."""
    k = len(cores)
    if k <= 1:
        return stage
    acquire = tuple(replace(a, units=cores) for a in stage.acquire)
    free = tuple(replace(a, units=cores) for a in stage.free)
    if context is not None and axis_count is not None and stage.placement:
        per_core = [context.owned(c, axis_count, roles) for c in cores]
        placement = {}
        cohort_members = {}
        for r, single_units in stage.placement.items():
            members = tuple(mapping.get(r, single_units) for mapping in per_core)
            placement[r] = tuple(dict.fromkeys(u for group in members for u in group))
            cohort_members[r] = members
    else:
        placement = {r: cores for r in stage.placement}
        cohort_members = {r: tuple((c,) for c in cores) for r in stage.placement}
    return replace(
        stage,
        work={r: v * k for r, v in stage.work.items()},
        caps={r: v * k for r, v in stage.caps.items()},
        deliveries=tuple((r, v * k) for r, v in stage.deliveries),
        placement=placement,
        acquire=acquire,
        free=free,
        cohort_size=k,
        cohort_widened=tuple(placement),
        cohort_members=cohort_members,
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
        from .catlass_lower import _mx_scale_group_span
        sc_groups = _mx_scale_group_span(panel.k_start_mod + start, micro)[1] if spec.mx_scale_group else 0
        ma = packed_bytes(panel.tile_m * micro, spec.dtype_a) + panel.tile_m * sc_groups
        mb = packed_bytes(panel.tile_n * micro, _compute_b(spec)) + panel.tile_n * sc_groups
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
    owner: int, slot_sizes: tuple[int, int, int, int],
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
            stages, panel.gm_a, owner, context, hbm_bus, f"gm_a_{cur}",
            after=gm_after, previous=gm_prev, acquire=l1a_alloc,
        )
        a_ready = f"gm_a_{cur}"
        if panel.scale_a is not None:
            a_ready = f"scale_a_{cur}"
            _append_access_stages(
                stages, panel.scale_a, owner, context, hbm_bus, a_ready, after=(f"gm_a_{cur}",),
            )
        _append_access_stages(
            stages, panel.gm_b, owner, context, hbm_bus, f"gm_b_{cur}",
            after=gm_after, previous=gm_prev, acquire=l1b_alloc,
        )
        b_ready = f"gm_b_{cur}"
        if panel.scale_b is not None:
            b_ready = f"scale_b_{cur}"
            _append_access_stages(
                stages, panel.scale_b, owner, context, hbm_bus, b_ready, after=(f"gm_b_{cur}",),
            )
        _append_panel_micro_stages(
            stages, panel, s_idx, total_panels, full_mc, a_ready, b_ready, l1a_alloc + l1b_alloc,
            spec, datapath, context, owner, (l0a_sz, l0b_sz),
        )
    return tuple(stages)


def _build_cohort_programs(
    wave_cohorts: tuple[_CohortWaveSpec, ...], w_idx: int, tag_prefix: str, is_mutex: bool,
    spec, datapath, context, hbm_bus: HbmBusSpec | None,
    l1_l0_sizes: tuple[int, int, int, int], l0c_sz: int,
) -> list[tuple[tuple[int, ...], Any, Any, tuple[Any, ...]]]:
    """Compile one wave's symmetric cohorts into ``PipelineTemplateProgram`` instances."""
    aic_count = datapath.aic_count
    progs: list[tuple[tuple[int, ...], Any, Any, tuple[Any, ...]]] = []
    for cohort_idx, cohort in enumerate(wave_cohorts):
        for sub_idx, sub_cores in enumerate(_split_colliding_cores(cohort.cores, context, aic_count)):
            c0 = sub_cores[0]
            l0c_name = f"l0c_{tag_prefix}{w_idx}_c{cohort_idx}_{sub_idx}" if is_mutex else "l0c"
            l0c_alloc = _alloc_or_empty(context, l0c_name, "l0c", l0c_sz, c0)
            widened_l0c = (
                tuple(replace(a, units=sub_cores) for a in l0c_alloc)
                if len(sub_cores) > 1 else l0c_alloc
            )
            setup = ((_widen_stage(
                Stage("l0c_acq", acquire=l0c_alloc, keep_empty=bool(l0c_alloc)),
                sub_cores, context, aic_count,
            ),) if (l0c_alloc and not is_mutex) else ())
            phases = tuple(
                PipelinePhase(
                    f"p{p_idx}",
                    tuple(
                        _widen_stage(st, sub_cores, context, aic_count)
                        for st in _build_step_stages(
                            ph.panels, spec, datapath, context, hbm_bus, c0, l1_l0_sizes,
                        )
                    ),
                    ph.count,
                )
                for p_idx, ph in enumerate(cohort.phases)
            )
            drain_stages: list[Stage] = []
            _append_access_stages(
                drain_stages, cohort.drain, c0, context, hbm_bus,
                "fixpipe", free=() if is_mutex else l0c_alloc,
            )
            drain = tuple(_widen_stage(st, sub_cores, context, aic_count) for st in drain_stages)
            if is_mutex:
                kloop_prog = PipelineTemplateProgram(
                    PipelineTemplate(owner=c0, slots=_PAIR_DEPTH, phases=phases, setup=(), drain=()),
                )
                progs.append((sub_cores, kloop_prog, KernelProgram(drain), widened_l0c))
            else:
                progs.append((sub_cores, PipelineTemplateProgram(
                    PipelineTemplate(owner=c0, slots=_PAIR_DEPTH, phases=phases, setup=setup, drain=drain),
                ), None, ()))
    return progs


def _emit_periodic_waves(
    stages: list[Stage], waves: tuple[tuple[_CohortWaveSpec, ...], ...],
    owner_prev: dict[int, str], owner_fix_history: dict[int, list[str]],
    default_after: str, tag_prefix: str, is_mutex: bool, l0c_depth: int,
    spec, datapath, context, hbm_bus: HbmBusSpec | None,
    l1_l0_sizes: tuple[int, int, int, int], l0c_sz: int,
) -> None:
    """Append outer ``Stage`` descriptors for a sequence of periodic cohort waves."""
    for w_idx, wave_cohorts in enumerate(waves):
        progs = _build_cohort_programs(
            wave_cohorts, w_idx, tag_prefix, is_mutex,
            spec, datapath, context, hbm_bus, l1_l0_sizes, l0c_sz,
        )
        if is_mutex:
            for c_idx, (cores, kloop_prog, fix_prog, slot_alloc) in enumerate(progs):
                tag = f"{tag_prefix}wave_{w_idx}" if len(progs) == 1 else f"{tag_prefix}wave_{w_idx}_c{c_idx}"
                k_deps: list[str] = []
                f_deps: list[str] = [f"{tag}_mma"]
                for c in cores:
                    k_deps.append(owner_prev.get(c, default_after))
                    hist = owner_fix_history.get(c)
                    if hist:
                        f_deps.append(hist[-1])
                        if len(hist) >= l0c_depth:
                            k_deps.append(hist[-l0c_depth])
                stages.append(Stage(
                    f"{tag}_mma", after=tuple(dict.fromkeys(k_deps)),
                    acquire=slot_alloc, program=kloop_prog,
                ))
                stages.append(Stage(
                    f"{tag}_fix", after=tuple(dict.fromkeys(f_deps)),
                    free=slot_alloc, program=fix_prog,
                ))
                for c in cores:
                    owner_prev[c] = f"{tag}_mma"
                    owner_fix_history.setdefault(c, []).append(f"{tag}_fix")
        elif len(progs) == 1:
            cores, prog, _, _ = progs[0]
            w_name = f"{tag_prefix}wave_{w_idx}"
            w_after = tuple(dict.fromkeys(owner_prev.get(c, default_after) for c in cores))
            stages.append(Stage(w_name, after=w_after, program=ParallelProgram((prog,))))
            for c in cores:
                owner_prev[c] = w_name
        else:
            for c_idx, (cores, prog, _, _) in enumerate(progs):
                c_name = f"{tag_prefix}wave_{w_idx}_c{c_idx}"
                c_after = tuple(dict.fromkeys(owner_prev.get(c, default_after) for c in cores))
                stages.append(Stage(c_name, after=c_after, program=prog))
                for c in cores:
                    owner_prev[c] = c_name


def _emit_splitk_reduce_stages(
    stages: list[Stage], reduce_tasks: tuple[_SplitkReduceSpec, ...], signal_name: str,
    datapath, context, hbm_bus: HbmBusSpec | None,
) -> None:
    """Append independent per-block AIV partial-load, ``vec_reduce``, and final C write stages."""
    aiv_count = datapath.aiv_count
    vec_role = "vec_reduce"
    if context is not None and context.service_rate(vec_role) is None and vec_role not in context.resource_names:
        vec_role = "vector"
    for task in reduce_tasks:
        owner = task.owner
        load_names: list[str] = []
        for idx, rec in enumerate(task.partial_reads):
            lname = f"reduce:{task.block}:partial:{idx}"
            _append_access_stages(stages, rec, owner, context, hbm_bus, lname, after=(signal_name,))
            load_names.append(lname)
        add_name = f"reduce:{task.block}:add"
        if task.reduce_ops > 0:
            if context is not None:
                w_vec = context.bind(Work(resources={vec_role: task.reduce_ops}))
                vec_pl = context.owned(owner, aiv_count, (vec_role,)) if w_vec else {}
                vec_pl = {k: v for k, v in vec_pl.items() if k in w_vec}
            else:
                w_vec, vec_pl = {"vec_reduce": task.reduce_ops}, {"vec_reduce": (owner,)}
            stages.append(Stage(add_name, w_vec, after=tuple(load_names), placement=vec_pl))
        else:
            stages.append(Stage(add_name, after=tuple(load_names)))
        _append_access_stages(
            stages, task.output, owner, context, hbm_bus,
            f"reduce:{task.block}:output", after=(add_name,),
        )


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
    initial_state: Any = None
    streamk_partials: tuple[_StreamkPartialCohortSpec, ...] = ()
    streamk_partial_count: int | None = None
    reduce_tasks: tuple[_SplitkReduceSpec, ...] = ()
    tail_waves: tuple[tuple[_CohortWaveSpec, ...], ...] = ()
    has_cross_core_flag: bool = False
    init_order_invariant: bool = False
    knee_curve: tuple[tuple[float, float], ...] = ()
    granule_bytes: int = 0
    mem_access_die: Any = None
    _rebound_cache: dict[Any, Any] = field(default_factory=dict, compare=False, hash=False, repr=False)

    def with_context(self, datapath, context) -> "_PeriodicCatlassProgram":
        """Rebind the compact wave/phase descriptors to ``context`` in O(unique phases)."""
        probe = Lowering(datapath, context=context, die=self.die)
        l2_cap = dict(datapath.buffers).get("l2", 0)
        has_l2_rate = context is None or "l2" in context.rates
        self_has_l2_rate = self.context is None or "l2" in self.context.rates
        probe_knee = probe.state._knee_curves_by_die.get(self.die, ())
        probe_granule = probe.state._granule_by_die.get(self.die, probe.state.granule_bytes)
        probe_mem_access = probe.state.memory_access_for(self.die)
        if (
            ((self.context is None) == (context is None) or self.init_order_invariant)
            and l2_cap == dict(self.datapath.buffers).get("l2", 0)
            and probe_mem_access == self.mem_access_die
            and probe_knee == self.knee_curve
            and probe_granule == self.granule_bytes
            and probe.burst_rules == self.burst_rules
            and has_l2_rate == self_has_l2_rate
        ):
            return replace(self, datapath=datapath, context=context, hbm_bus=probe.hbm_bus)
        cache_key = (
            context is not None, l2_cap, probe_mem_access, probe_knee,
            probe_granule, probe.burst_rules, has_l2_rate,
        )
        cached = self._rebound_cache.get(cache_key)
        if cached is not None:
            return replace(cached, datapath=datapath, context=context, hbm_bus=probe.hbm_bus)
        from .catlass_lower import solve_catlass_tile

        warm_view = deepcopy(self.initial_state) if self.initial_state is not None else None
        sol = solve_catlass_tile(self.spec, self.problem, datapath, memory_view=warm_view, die=self.die)
        rebuilt = lower_catlass_periodic(
            self.spec, self.problem, datapath, sol,
            context=context, memory_view=warm_view, die=self.die, region_id=self.summary.region_id,
        ).program
        self._rebound_cache[cache_key] = rebuilt
        return rebuilt

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
        is_mutex = self.spec.dispatch_policy == "MmadPingpongMutex"
        l0c_depth = self.spec.stages[-1]
        owner_prev: dict[int, str] = {}
        owner_fix_history: dict[int, list[str]] = {}
        l1_l0_sizes, l0c_sz = self.slot_sizes[:4], self.slot_sizes[4]
        _emit_periodic_waves(
            stages, self.waves, owner_prev, owner_fix_history, "launch", "",
            is_mutex, l0c_depth, self.spec, self.datapath, self.context, self.hbm_bus, l1_l0_sizes, l0c_sz,
        )
        part_names: list[str] = []
        aic_count = self.datapath.aic_count
        for sk_idx, sk_cohort in enumerate(self.streamk_partials):
            for sub_idx, sub_cores in enumerate(_split_colliding_cores(sk_cohort.cores, self.context, aic_count)):
                c0 = sub_cores[0]
                deps = tuple(dict.fromkeys(owner_prev.get(c, "launch") for c in sub_cores))
                for p_idx, (part_phases, part_drain) in enumerate(sk_cohort.parts):
                    l0c_alloc = _alloc_or_empty(self.context, "l0c", "l0c", l0c_sz, c0)
                    setup = ((_widen_stage(
                        Stage("l0c_acq", acquire=l0c_alloc, keep_empty=bool(l0c_alloc)),
                        sub_cores, self.context, aic_count,
                    ),) if l0c_alloc else ())
                    phases = tuple(
                        PipelinePhase(
                            f"p{ph_idx}",
                            tuple(
                                _widen_stage(st, sub_cores, self.context, aic_count)
                                for st in _build_step_stages(
                                    ph.panels, self.spec, self.datapath, self.context,
                                    self.hbm_bus, c0, l1_l0_sizes,
                                )
                            ),
                            ph.count,
                        )
                        for ph_idx, ph in enumerate(part_phases)
                    )
                    drain_stages: list[Stage] = []
                    _append_access_stages(
                        drain_stages, part_drain, c0, self.context, self.hbm_bus, "partial", free=l0c_alloc,
                    )
                    drain = tuple(_widen_stage(st, sub_cores, self.context, aic_count) for st in drain_stages)
                    prog = PipelineTemplateProgram(
                        PipelineTemplate(owner=c0, slots=_PAIR_DEPTH, phases=phases, setup=setup, drain=drain),
                    )
                    s_name = f"sk_g{sk_idx}_s{sub_idx}_p{p_idx}"
                    stages.append(Stage(s_name, after=deps, program=prog))
                    deps = (s_name,)
                part_names.extend(deps)
        if self.has_cross_core_flag:
            flag_after = (
                tuple(part_names) if self.streamk_partials
                else tuple(dict.fromkeys(owner_prev.values())) or ("launch",)
            )
            flag_meta = (
                {"streamk_partial_count": self.streamk_partial_count}
                if self.streamk_partial_count is not None else {}
            )
            stages.append(Stage("cross_core_set_flag", after=flag_after, metadata=flag_meta))
        if self.reduce_tasks:
            _emit_splitk_reduce_stages(
                stages, self.reduce_tasks, "cross_core_set_flag", self.datapath, self.context, self.hbm_bus,
            )
        if self.tail_waves:
            _emit_periodic_waves(
                stages, self.tail_waves, {}, {}, "cross_core_set_flag", "tail_",
                False, l0c_depth, self.spec, self.datapath, self.context, self.hbm_bus, l1_l0_sizes, l0c_sz,
            )
        return KernelProgram(tuple(stages)).start(engine, name, ready, owner=owner)


@dataclass(frozen=True, slots=True)
class _PatternMemo:
    """Precomputed immutable tensor geometry, reuse keys, and die memory-access defaults."""

    sector: int
    l2_line: int | None
    l2_sector: int | None
    burst_granule: int
    scatter_open: int
    a_stride: int
    b_stride: int
    a_elem: int | None
    b_elem: int | None
    key_a: ReuseKey
    key_b: ReuseKey | None
    key_sa: ReuseKey | None
    key_sb: ReuseKey | None
    key_c: ReuseKey


def _build_pattern_memo(spec, p, die: int, sector: int, mem_access) -> _PatternMemo:
    a_one = packed_bytes(1, spec.dtype_a)
    a_elem = a_one if packed_bytes(_OPERAND_PAIR_COUNT, spec.dtype_a) == _OPERAND_PAIR_COUNT * a_one else None
    b_one = packed_bytes(1, spec.dtype_b)
    b_elem = b_one if packed_bytes(_OPERAND_PAIR_COUNT, spec.dtype_b) == _OPERAND_PAIR_COUNT * b_one else None
    bypass_b = spec.l2_cache_hint_b == "disable"
    key_a = ReuseKey(die, p.a_id, 0, reuse_stream=f"{p.a_id}:0")
    key_b = None if bypass_b else ReuseKey(die, p.b_id, p.b_version, reuse_stream=f"{p.b_id}:{p.b_version}")
    key_sa = ReuseKey(die, p.a_id + ":scale", 0, reuse_stream=f"{p.a_id}:scale:0") if spec.mx_scale_group else None
    key_sb = (
        None if (bypass_b or not spec.mx_scale_group)
        else ReuseKey(die, p.b_id + ":scale", 0, reuse_stream=f"{p.b_id}:scale:0")
    )
    return _PatternMemo(
        sector=sector,
        l2_line=getattr(mem_access, "l2_line_bytes", None),
        l2_sector=getattr(mem_access, "l2_sector_bytes", None),
        burst_granule=getattr(mem_access, "hbm_burst_granule_bytes", 0),
        scatter_open=getattr(mem_access, "hbm_scatter_row_open_bytes", 0),
        a_stride=packed_bytes(p.k, spec.dtype_a),
        b_stride=packed_bytes(p.n, spec.dtype_b),
        a_elem=a_elem,
        b_elem=b_elem,
        key_a=key_a,
        key_b=key_b,
        key_sa=key_sa,
        key_sb=key_sb,
        key_c=ReuseKey(die, p.c_id, 1, reuse_stream=f"{p.c_id}:1"),
    )


def _make_rect_pattern(
    spec, p, die: int, sector: int, tensor: str, dtype: str,
    row: int, col: int, rows: int, cols: int, width: int, owner: int,
    *, bypass: bool = False, memo: _PatternMemo | None = None,
) -> MemoryAccessPattern:
    is_b = tensor == p.b_id
    if memo is not None:
        stride = memo.b_stride if is_b else memo.a_stride
        eb = memo.b_elem if is_b else memo.a_elem
        if eb is not None:
            col_off, rb = col * eb, cols * eb
        else:
            col_off = packed_bytes(col, dtype)
            rb = packed_bytes(col + cols, dtype) - col_off
            if rb != packed_bytes(cols, dtype):
                raise ModelingError("catlass_packed_tile_unaligned")
        key = None if bypass else (memo.key_b if is_b else memo.key_a)
        run_plan = AddressRunPlan(
            (AddressRun((p.b_base_byte if is_b else 0) + row * stride + col_off, rows, rb, stride, "RowMajor"),),
            sector_bytes=memo.sector, l2_line_bytes=memo.l2_line, l2_sector_bytes=memo.l2_sector,
            hbm_burst_granule_bytes=memo.burst_granule, hbm_scatter_row_open_bytes=memo.scatter_open,
        )
        return MemoryAccessPattern(
            tensor, p.b_version if is_b else 0, die, owner, "read", "mte2_aic", rb * rows,
            run_plan, key, None, _VIS_READ,
        )
    stride = packed_bytes(width, dtype)
    col_off = packed_bytes(col, dtype)
    rb = packed_bytes(col + cols, dtype) - col_off
    if rb != packed_bytes(cols, dtype):
        raise ModelingError("catlass_packed_tile_unaligned")
    base = p.b_base_byte if is_b else 0
    version = p.b_version if is_b else 0
    run = AddressRun(base + row * stride + col_off, rows, rb, stride, "RowMajor")
    key = None if bypass else ReuseKey(die, tensor, version, reuse_stream=f"{tensor}:{version}")
    return MemoryAccessPattern(
        tensor, version, die, owner, "read", "mte2_aic", rb * rows,
        AddressRunPlan((run,), sector_bytes=sector), key, None, _VIS_READ,
    )


def _make_scale_pattern(
    spec, p, die: int, sector: int, tensor: str, row: int, col: int, rows: int, cols: int,
    width: int, owner: int, *, is_a: bool, bypass: bool = False, memo: _PatternMemo | None = None,
) -> MemoryAccessPattern:
    from .catlass_lower import _mx_scale_group_span, _mx_scale_runs

    del width
    k_scales_total = ceil_div(p.k, MX_SCALE_GROUP)
    g_lo, sc = _mx_scale_group_span(col, cols) if is_a else _mx_scale_group_span(row, rows)
    outer_start, outer_count, outer_total = (row, rows, p.m) if is_a else (col, cols, p.n)
    k_contiguous = (spec.layout_a == "RowMajor") if is_a else (spec.layout_b == "ColumnMajor")
    runs = _mx_scale_runs(
        outer_start, outer_count, outer_total, g_lo, sc, k_scales_total,
        k_contiguous=k_contiguous,
    )
    tid = tensor + ":scale"
    if memo is not None:
        key = None if bypass else (memo.key_sa if is_a else memo.key_sb)
        run_plan = AddressRunPlan(
            runs,
            sector_bytes=memo.sector, l2_line_bytes=memo.l2_line, l2_sector_bytes=memo.l2_sector,
            hbm_burst_granule_bytes=memo.burst_granule, hbm_scatter_row_open_bytes=memo.scatter_open,
        )
        return MemoryAccessPattern(
            tid, 0, die, owner, "read", "mte2_aic", outer_count * sc, run_plan, key, None, _VIS_READ,
        )
    key = None if bypass else ReuseKey(die, tid, 0, reuse_stream=f"{tid}:0")
    return MemoryAccessPattern(
        tid, 0, die, owner, "read", "mte2_aic", outer_count * sc,
        AddressRunPlan(runs, sector_bytes=sector), key, None, _VIS_READ,
    )


def _make_output_pattern(
    spec, p, tile, die: int, sector: int, *, memo: _PatternMemo | None = None,
    pipe: str = "fixpipe", owner: int | None = None,
) -> MemoryAccessPattern:
    dtype = spec.dtype_c
    eff_owner = tile.owner if owner is None else owner
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
    if memo is not None:
        run_plan = AddressRunPlan(
            (run,), sector_bytes=memo.sector, l2_line_bytes=memo.l2_line, l2_sector_bytes=memo.l2_sector,
            hbm_burst_granule_bytes=memo.burst_granule, hbm_scatter_row_open_bytes=memo.scatter_open,
        )
        return MemoryAccessPattern(p.c_id, 1, die, eff_owner, "write", pipe, amt, run_plan, memo.key_c, None, _VIS_WRITE)
    key = ReuseKey(die, p.c_id, 1, reuse_stream=f"{p.c_id}:1")
    return MemoryAccessPattern(
        p.c_id, 1, die, eff_owner, "write", pipe, amt,
        AddressRunPlan((run,), sector_bytes=sector), key, None, _VIS_WRITE,
    )


def _make_partial_pattern(
    spec, p, tile, run: AddressRun, size: int, die: int, sector: int,
    *, direction: str, pipe: str, owner: int, memo: _PatternMemo | None = None,
) -> MemoryAccessPattern:
    scratch = p.c_id + ":splitk"
    key = ReuseKey(die, scratch, 0, reuse_stream=f"{scratch}:0")
    if memo is not None:
        run_plan = AddressRunPlan(
            (run,), sector_bytes=memo.sector, l2_line_bytes=memo.l2_line, l2_sector_bytes=memo.l2_sector,
            hbm_burst_granule_bytes=memo.burst_granule, hbm_scatter_row_open_bytes=memo.scatter_open,
        )
    else:
        run_plan = AddressRunPlan((run,), sector_bytes=sector)
    return MemoryAccessPattern(scratch, 0, die, owner, direction, pipe, size, run_plan, key, None, _VIS_PRIVATE)


def _scale_panel_count(spec, k_start: int, k_stop: int) -> int:
    """Count K panels in ``range(k_start, k_stop)`` that fetch MX scales from GM."""
    if not spec.mx_scale_group or k_stop <= k_start:
        return 0
    factor = spec.l1_scale_factor_k
    return ceil_div(k_stop, factor) - ceil_div(k_start, factor)


def _exact_tile_stage_count(
    spec, gk: int, total_micros: int, has_ctx: bool, *, scale_panels: int | None = None,
) -> int:
    """Exact unrolled stage count for one double-buffered dataparallel ordinary tile."""
    sc_panels = _scale_panel_count(spec, 0, gk) if scale_panels is None else scale_panels
    fill_stages = _OPERAND_PAIR_COUNT * (gk + sc_panels)
    ring_gens = (gk, gk, total_micros, total_micros, _L0C_DEPTH)
    ring_stages = 0
    for depth, gens in zip(spec.stages, ring_gens):
        slots_used = min(depth, gens)
        ring_stages += slots_used + _RING_GEN_STAGES * gens + (slots_used if has_ctx else 0)
    return fill_stages + _MICRO_WORK_STAGES * total_micros + _TILE_FINISH_STAGES + ring_stages


def _exact_mutex_stage_count(
    spec, core_tile_counts: Counter[int], gk: int, total_micros: int, has_ctx: bool,
) -> int:
    """Exact unrolled stage count for persistent-ring ``MmadPingpongMutex`` across all tiles."""
    fill_stages = _OPERAND_PAIR_COUNT * (gk + _scale_panel_count(spec, 0, gk))
    per_tile_gens = _OPERAND_PAIR_COUNT * gk + _OPERAND_PAIR_COUNT * total_micros + _L0C_DEPTH
    per_tile = (
        fill_stages
        + _MICRO_WORK_STAGES * total_micros
        + _TILE_FINISH_STAGES
        + _RING_GEN_STAGES * per_tile_gens
    )
    total = sum(core_tile_counts.values()) * per_tile
    slot_mult = _OPERAND_PAIR_COUNT if has_ctx else 1
    for tile_count in core_tile_counts.values():
        ring_gens = (tile_count * gk, tile_count * gk, tile_count * total_micros, tile_count * total_micros, tile_count)
        for depth, gens in zip(spec.stages, ring_gens):
            total += min(depth, gens) * slot_mult
    return total


class _PanelPhases:
    """Encode pair periods with bounded lookahead, retaining only emitted phases."""

    def __init__(self, periods):
        self.periods = periods
        self.window = _MIN_MULTI_PAIR_REPEATS * max(periods)
        self.pending = []
        self.phases = []
        self.keys = []
        self.repeat = False
        self.tail = None

    def append(self, panel):
        if self.tail is None:
            self.tail = panel
            return
        pair = (self.tail, panel)
        self.tail = None
        key = tuple(map(_panel_key, pair))
        self.pending.append((pair, key))
        if self.repeat:
            keys, count = self.keys[-1]
            if key == keys[len(self.pending) - 1]:
                if len(self.pending) == len(keys):
                    self.phases[-1] = replace(self.phases[-1], count=count + 1)
                    self.keys[-1] = (keys, count + 1)
                    self.pending.clear()
                return
            self.repeat = False
        if len(self.pending) == self.window:
            self._emit(allow_repeat=True)

    def _emit(self, *, allow_repeat=False):
        sigs = [key for _, key in self.pending]
        best_period, best_count, covered = 1, 1, 1
        for period in self.periods:
            if period > len(sigs):
                break
            key, count = sigs[:period], 1
            for start in range(period, len(sigs) - period + 1, period):
                if sigs[start:start + period] != key:
                    break
                count += 1
            if count >= (1 if period == 1 else _MIN_MULTI_PAIR_REPEATS) and period * count > covered:
                best_period, best_count, covered = period, count, period * count
        panels = tuple(panel for pair, _ in self.pending[:best_period] for panel in pair)
        self.phases.append(_StepPhaseDescriptor(panels, best_count))
        self.keys.append((tuple(sigs[:best_period]), best_count))
        # The candidate periods divide one another and the lookahead window.
        # A period covering the window is the first to reach a future mismatch.
        self.repeat = allow_repeat and covered == self.window
        del self.pending[:covered]

    def finish(self):
        while self.pending:
            self._emit()
        if self.tail is not None:
            self.phases.append(_StepPhaseDescriptor((self.tail,), 1))
            self.keys.append(((_panel_key(self.tail),), 1))
        return tuple(self.phases), tuple(self.keys)


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
        from .catlass_lower import _mx_scale_runs

        sa_w = ceil_div(p.k, MX_SCALE_GROUP)
        for tid, rows, k_contig in (
            (p.a_id + ":scale", p.m, spec.layout_a == "RowMajor"),
            (p.b_id + ":scale", p.n, spec.layout_b == "ColumnMajor"),
        ):
            s_runs = _mx_scale_runs(0, rows, rows, 0, sa_w, sa_w, k_contiguous=k_contig)
            s_pat = MemoryAccessPattern(
                tid, 0, die, 0, "read", "mte2_aic", rows * sa_w,
                AddressRunPlan(s_runs, sector_bytes=sector), None, None, _VIS_READ,
            )
            if ob := _logical_obligation(s_pat):
                reads.append(ob)
    c_stride, c_rows = (
        (packed_bytes(p.n, spec.dtype_c), p.m) if spec.layout_c == "RowMajor"
        else (packed_bytes(p.m, spec.dtype_c), p.n)
    )
    writes = [ob] if (ob := _logical_obligation(_pat(p.c_id, 1, "write", "fixpipe", 0, c_rows, c_stride, _VIS_WRITE))) else []
    return reads, writes


def _resolve_lifetime_pattern(lifetimes: _ReuseLifetimes, pat: MemoryAccessPattern) -> MemoryAccessPattern:
    key = pat.reuse_key
    if key is None:
        return pat
    entry = lifetimes.keys.get(key)
    if entry is None:
        resolved = lifetimes.state.resolve_reuse_key(key)
        entry = (resolved, lifetimes.state.retain_template_key(resolved))
        lifetimes.keys[key] = entry
    return pat if entry[0] == key else replace(pat, reuse_key=entry[0])


def _admit_periodic_access(
    pat: MemoryAccessPattern, plan: Lowering, entry_state,
    lifetimes: _ReuseLifetimes, resource_work: Counter[str],
    context, die: int, *, track_obligations: bool = True,
) -> _AccessRecord:
    die_mem_access = plan.state.memory_access_for(die)
    resolved_runs = _access_plan(pat.runs, die_mem_access)
    if resolved_runs is not pat.runs:
        pat = replace(pat, runs=resolved_runs)
    if track_obligations and not pat.visibility.private_workspace and (entry := _logical_obligation(pat)):
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
    declared_key = pat.reuse_key
    pat = _resolve_lifetime_pattern(lifetimes, pat)
    lg, pat, res_prior = admit_memory_ledger(
        pat, plan.state, bus_memory_access=die_mem_access,
        burst_rules=plan.burst_rules, commit=True,
    )
    plan._account(lg, pat.runs, res_prior if (lg.l2_hit_bytes and lg.dram_sector_bytes) else SectorCoverage())
    lifetimes.complete(declared_key)
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


def _count_wave_compute(tiles, spec, p, datapath, low, plan, resource_work, cube_dt) -> int:
    from .catlass_lower import _compute_b, _issued_ops, _mx_scale_group_span

    full, tail = divmod(p.k, low.tk)
    panel_groups: list[tuple[int, int, int]] = []
    if spec.mx_scale_group:
        period = MX_SCALE_GROUP // gcd(low.tk, MX_SCALE_GROUP)
        cycles, rem_panels = divmod(full, period)
        for idx in range(min(full, period)):
            count = cycles + (1 if idx < rem_panels else 0)
            panel_groups.append((low.tk, (idx * low.tk) % MX_SCALE_GROUP, count))
        if tail > 0:
            panel_groups.append((tail, (full * low.tk) % MX_SCALE_GROUP, 1))
    else:
        if full > 0:
            panel_groups.append((low.tk, 0, full))
        if tail > 0:
            panel_groups.append((tail, 0, 1))
    for (m, n), owners in Counter((tile.m, tile.n) for tile in tiles).items():
        for k, k_mod, repetitions in panel_groups:
            repeats = owners * repetitions
            if not repeats:
                continue
            for start in range(0, k, low.micro_k):
                micro = min(k - start, low.micro_k)
                sc_groups = _mx_scale_group_span(k_mod + start, micro)[1] if spec.mx_scale_group else 0
                a = packed_bytes(m * micro, spec.dtype_a) + m * sc_groups
                b = packed_bytes(n * micro, _compute_b(spec)) + n * sc_groups
                resource_work["mte1"] += (a + b) * repeats
                plan.cube[cube_dt] += _issued_ops(m, n, micro, spec, datapath) * repeats
    return sum(ceil_div(k, low.micro_k) * repetitions for k, _, repetitions in panel_groups)


def _panel_patterns(spec, p, tile, ks, kl, die, sector, memo: _PatternMemo | None = None):
    bypass_b = spec.l2_cache_hint_b == "disable"
    tk = spec.l1_tile_shape[-1]
    load_scales = bool(spec.mx_scale_group and (ks // tk) % spec.l1_scale_factor_k == 0)
    scale_kl = min(tk * spec.l1_scale_factor_k, p.k - ks) if load_scales else 0
    gm_a = _make_rect_pattern(
        spec, p, die, sector, p.a_id, spec.dtype_a, tile.row, ks, tile.m, kl, p.k, tile.owner, memo=memo,
    )
    sc_a = (_make_scale_pattern(
        spec, p, die, sector, p.a_id, tile.row, ks, tile.m, scale_kl, p.k, tile.owner, is_a=True, memo=memo,
    ) if load_scales else None)
    gm_b = _make_rect_pattern(
        spec, p, die, sector, p.b_id, spec.dtype_b, ks, tile.col, kl, tile.n, p.n, tile.owner,
        bypass=bypass_b, memo=memo,
    )
    sc_b = (_make_scale_pattern(
        spec, p, die, sector, p.b_id, ks, tile.col, scale_kl, tile.n, p.n, tile.owner, is_a=False,
        bypass=bypass_b, memo=memo,
    ) if load_scales else None)
    return gm_a, sc_a, gm_b, sc_b


def _admit_optional(admit, pat: MemoryAccessPattern | None) -> _AccessRecord | None:
    return admit(pat) if pat is not None else None


def _check_init_order_safe(
    init_safe: list[bool], first_tile_seen: list[bool],
    state, die: int, key_a: ReuseKey, key_b: ReuseKey | None,
    a_recs: list[_AccessRecord], b_recs: list[_AccessRecord],
) -> None:
    """Conservatively verify that ``(A0, B0, A1, B1)`` and ``(A0, A1, B0, B1)`` are L2-equivalent."""
    if not init_safe[0] or key_b is None:
        return
    if any(r.writeback_bytes for r in a_recs):
        init_safe[0] = False
        return
    entry_a = state._resident.get(key_a)
    if entry_a is None or entry_a.evicted:
        init_safe[0] = False
        return
    if not first_tile_seen[0]:
        first_tile_seen[0] = True
        entry_b = state._resident.get(key_b)
        if (
            entry_b is None or entry_b.evicted
            or any(r.writeback_bytes for r in b_recs)
            or state._hit_fraction(entry_a.working_set_bytes, state.effective_l2_bytes(die), die=die) != 1.0
        ):
            init_safe[0] = False
        return
    if any(r.ledger.dram_miss_copy_bytes > 0 for r in a_recs):
        init_safe[0] = False


def _lower_periodic_wave_fast(
    tiles: tuple[Any, ...], p, low, fast_admit_rect, has_ctx: bool, periods: tuple[int, ...],
    *, init_check=None,
) -> list[_PanelPhases]:
    """Bind a zero-scale byte-aligned wave without per-panel pattern object allocation."""
    full_mc = ceil_div(low.tk, low.micro_k)
    step_width = _PAIR_DEPTH if full_mc == 1 else 1
    encoders = [_PanelPhases(periods) for _ in tiles]
    init_count = min(_PAIR_DEPTH, low.gk)
    init_k = tuple((idx * low.tk, min(low.tk, p.k - idx * low.tk)) for idx in range(init_count))
    init_gm: list[list[list[_AccessRecord | None]]] = [
        [[None, None] for _ in range(init_count)] for _ in tiles
    ]
    for t_idx, tile in enumerate(tiles):
        if has_ctx:
            for p_idx, (ks, kl) in enumerate(init_k):
                init_gm[t_idx][p_idx][0] = fast_admit_rect(False, tile, ks, kl)
            for p_idx, (ks, kl) in enumerate(init_k):
                init_gm[t_idx][p_idx][1] = fast_admit_rect(True, tile, ks, kl)
        else:
            for p_idx, (ks, kl) in enumerate(init_k):
                init_gm[t_idx][p_idx][0] = fast_admit_rect(False, tile, ks, kl)
                init_gm[t_idx][p_idx][1] = fast_admit_rect(True, tile, ks, kl)
        if init_check is not None:
            init_check(
                [init_gm[t_idx][p_idx][0] for p_idx in range(init_count)],
                [init_gm[t_idx][p_idx][1] for p_idx in range(init_count)],
            )
    for t_idx, (tile, encoder) in enumerate(zip(tiles, encoders)):
        for p_idx, (_, kl) in enumerate(init_k):
            rec_a, rec_b = init_gm[t_idx][p_idx]
            encoder.append(_PanelDescriptor(tile.m, tile.n, kl, rec_a, None, rec_b, None))

    for batch_start in range(init_count, low.gk, step_width):
        batch_end = min(batch_start + step_width, low.gk)
        batch_k = tuple((idx * low.tk, min(low.tk, p.k - idx * low.tk)) for idx in range(batch_start, batch_end))
        for tile, encoder in zip(tiles, encoders):
            for ks, kl in batch_k:
                rec_a = fast_admit_rect(False, tile, ks, kl)
                rec_b = fast_admit_rect(True, tile, ks, kl)
                encoder.append(_PanelDescriptor(tile.m, tile.n, kl, rec_a, None, rec_b, None))
    return encoders


def _lower_periodic_wave(
    tiles: tuple[Any, ...], spec, p, datapath, low, plan: Lowering,
    admit, resource_work: Counter[str], cube_dt: str, die: int, has_ctx: bool,
    owner_cohort_id: dict[int, int], next_cohort_id: int, memo: _PatternMemo | None = None,
    fast_admit_rect=None, init_check=None,
) -> tuple[tuple[_CohortWaveSpec, ...], int, int, int]:
    """Bind each wave's panels in exact literal Kahn topological dependency order before grouping."""
    full_mc = ceil_div(low.tk, low.micro_k)
    periods = _CANDIDATE_PAIR_PERIODS_MC1 if full_mc == 1 else _CANDIDATE_PAIR_PERIODS
    step_width = _PAIR_DEPTH if full_mc == 1 else 1
    total_micros = _count_wave_compute(tiles, spec, p, datapath, low, plan, resource_work, cube_dt)
    has_scales = bool(spec.mx_scale_group)
    if fast_admit_rect is not None and not has_scales:
        encoders = _lower_periodic_wave_fast(
            tiles, p, low, fast_admit_rect, has_ctx, periods, init_check=init_check,
        )
    else:
        encoders = [_PanelPhases(periods) for _ in tiles]
        init_count = min(_PAIR_DEPTH, low.gk)
        init_pats = [
            [
                (ks, min(low.tk, p.k - ks), _panel_patterns(spec, p, tile, ks, min(low.tk, p.k - ks), die, plan.sector, memo))
                for panel_idx in range(init_count)
                for ks in (panel_idx * low.tk,)
            ]
            for tile in tiles
        ]
        init_gm: list[list[list[_AccessRecord | None]]] = [
            [[None, None, None, None] for _ in range(init_count)] for _ in tiles
        ]
        for t_idx, rows in enumerate(init_pats):
            if has_ctx:
                for p_idx in range(init_count):
                    init_gm[t_idx][p_idx][0] = admit(rows[p_idx][2][0])
                for p_idx in range(init_count):
                    init_gm[t_idx][p_idx][2] = admit(rows[p_idx][2][2])
            else:
                for p_idx in range(init_count):
                    init_gm[t_idx][p_idx][0] = admit(rows[p_idx][2][0])
                    init_gm[t_idx][p_idx][2] = admit(rows[p_idx][2][2])
        if has_scales:
            for t_idx, rows in enumerate(init_pats):
                if has_ctx:
                    for p_idx in range(init_count):
                        init_gm[t_idx][p_idx][1] = _admit_optional(admit, rows[p_idx][2][1])
                    for p_idx in range(init_count):
                        init_gm[t_idx][p_idx][3] = _admit_optional(admit, rows[p_idx][2][3])
                else:
                    for p_idx in range(init_count):
                        init_gm[t_idx][p_idx][1] = _admit_optional(admit, rows[p_idx][2][1])
                        init_gm[t_idx][p_idx][3] = _admit_optional(admit, rows[p_idx][2][3])
        for t_idx, (tile, encoder) in enumerate(zip(tiles, encoders)):
            for p_idx in range(init_count):
                ks, kl, _ = init_pats[t_idx][p_idx]
                encoder.append(_PanelDescriptor(
                    tile.m, tile.n, kl, *init_gm[t_idx][p_idx],
                    k_start_mod=(ks % MX_SCALE_GROUP if has_scales else 0),
                ))

        for batch_start in range(init_count, low.gk, step_width):
            batch_indices = tuple(range(batch_start, min(batch_start + step_width, low.gk)))
            batch_pats = [
                [
                    (ks, min(low.tk, p.k - ks), _panel_patterns(spec, p, tile, ks, min(low.tk, p.k - ks), die, plan.sector, memo))
                    for p_idx in batch_indices
                    for ks in (p_idx * low.tk,)
                ]
                for tile in tiles
            ]
            batch_recs: list[list[list[_AccessRecord | None]]] = [
                [[None, None, None, None] for _ in batch_indices] for _ in tiles
            ]
            for t_idx, rows in enumerate(batch_pats):
                for b_pos, (_, _, pats) in enumerate(rows):
                    batch_recs[t_idx][b_pos][0] = admit(pats[0])
                    batch_recs[t_idx][b_pos][2] = admit(pats[2])
            if has_scales:
                for t_idx, rows in enumerate(batch_pats):
                    for b_pos, (_, _, pats) in enumerate(rows):
                        batch_recs[t_idx][b_pos][1] = _admit_optional(admit, pats[1])
                        batch_recs[t_idx][b_pos][3] = _admit_optional(admit, pats[3])
            for t_idx, (tile, encoder) in enumerate(zip(tiles, encoders)):
                for b_pos, (ks, kl, _) in enumerate(batch_pats[t_idx]):
                    encoder.append(_PanelDescriptor(
                        tile.m, tile.n, kl, *batch_recs[t_idx][b_pos],
                        k_start_mod=(ks % MX_SCALE_GROUP if has_scales else 0),
                    ))

    cohorts = {}
    for tile, encoder in zip(tiles, encoders):
        drain = admit(_make_output_pattern(spec, p, tile, die, plan.sector, memo=memo))
        steps, signatures = encoder.finish()
        key = (owner_cohort_id.get(tile.owner, 0), signatures, _ledger_key(drain))
        owners, _, _ = cohorts.setdefault(key, ([], steps, drain))
        owners.append(tile.owner)
    wave_specs = tuple(_CohortWaveSpec(tuple(owners), steps, drain) for owners, steps, drain in cohorts.values())
    for cohort in wave_specs:
        for owner in cohort.cores:
            owner_cohort_id[owner] = next_cohort_id
        next_cohort_id += 1
    return (
        wave_specs,
        len(tiles) * _exact_tile_stage_count(spec, low.gk, total_micros, has_ctx),
        next_cohort_id,
        total_micros,
    )


def _make_fast_rect_admitter(
    spec, p, plan: Lowering, memo: _PatternMemo, lifetimes: _ReuseLifetimes,
    resource_work: Counter[str], context, die: int, admit,
):
    """Build a zero-allocation read admitter for cold-entry byte-aligned tiles."""
    state = plan.state
    bypass_b = spec.l2_cache_hint_b == "disable"
    tpl_a = _make_rect_pattern(spec, p, die, plan.sector, p.a_id, spec.dtype_a, 0, 0, 1, 1, p.k, 0, memo=memo)
    tpl_b = _make_rect_pattern(
        spec, p, die, plan.sector, p.b_id, spec.dtype_b, 0, 0, 1, 1, p.n, 0, bypass=bypass_b, memo=memo,
    )
    tpl_a = _resolve_lifetime_pattern(lifetimes, tpl_a)
    tpl_b = _resolve_lifetime_pattern(lifetimes, tpl_b)
    rkey_a, rkey_b = tpl_a.reuse_key, tpl_b.reuse_key
    decl_a, decl_b = memo.key_a, memo.key_b
    a_stride, b_stride = memo.a_stride, memo.b_stride
    a_elem, b_elem = memo.a_elem, memo.b_elem
    b_base, b_ver = p.b_base_byte, p.b_version
    sec, l2_line, l2_sec, burst = memo.sector, memo.l2_line, memo.l2_sector, memo.burst_granule
    track_l2_work = context is None or "l2" in context.rates
    service_int = [0]

    def flush_service() -> None:
        if service_int[0]:
            plan.service += service_int[0]
            service_int[0] = 0

    def fast_admit_rect(is_b: bool, tile, ks: int, kl: int) -> _AccessRecord:
        if is_b:
            base = b_base + ks * b_stride + tile.col * b_elem
            rows, rb, stride = kl, tile.n * b_elem, b_stride
            key, ver, decl_key, tpl = rkey_b, b_ver, decl_b, tpl_b
        else:
            base = tile.row * a_stride + ks * a_elem
            rows, rb, stride = tile.m, kl * a_elem, a_stride
            key, ver, decl_key, tpl = rkey_a, 0, decl_a, tpl_a
        lg = state.admit_rect_read_ledger(
            die, key, ver, base, rows, rb, stride,
            alignment_bytes=_DEFAULT_ALIGN_BYTES, sector_bytes=sec,
            l2_line_bytes=l2_line, l2_sector_bytes=l2_sec,
            burst_granule=burst, burst_rules=plan.burst_rules,
        )
        if lg is None:
            pat = _make_rect_pattern(
                spec, p, die, plan.sector,
                p.b_id if is_b else p.a_id,
                spec.dtype_b if is_b else spec.dtype_a,
                ks if is_b else tile.row,
                tile.col if is_b else ks,
                kl if is_b else tile.m,
                tile.n if is_b else kl,
                p.n if is_b else p.k,
                tile.owner,
                bypass=bypass_b if is_b else False,
                memo=memo,
            )
            return admit(pat)
        plan.logical += lg.logical_tensor_bytes
        plan.copied += lg.aligned_transfer_bytes
        plan.hits += lg.l2_hit_bytes
        plan.misses += lg.dram_miss_copy_bytes
        plan.sectors += lg.dram_sector_bytes
        if lg.dram_sector_bytes:
            if lg.dram_bus_transaction_bytes == lg.dram_sector_bytes:
                service_int[0] += lg.dram_sector_bytes
            else:
                plan.service += Fraction(lg.dram_bus_transaction_bytes)
        lifetimes.complete(decl_key)
        evictions = state.drain_writebacks(die)
        wb_bytes = sum(item.bytes for item in evictions) if evictions else 0
        if wb_bytes:
            plan.sectors += wb_bytes
            service_int[0] += wb_bytes
            resource_work["hbm"] += wb_bytes
        if lg.aligned_transfer_bytes:
            resource_work["mte2_aic"] += lg.aligned_transfer_bytes
        if lg.l2_hit_bytes and track_l2_work:
            resource_work["l2"] += lg.l2_hit_bytes
        if lg.dram_bus_transaction_bytes > 0.0:
            resource_work["hbm"] += lg.dram_bus_transaction_bytes
        return _AccessRecord(tpl, lg, wb_bytes)

    return fast_admit_rect, flush_service


def _count_slice_compute(tile, k_start: int, k_stop: int, spec, p, datapath, low, plan, resource_work, cube_dt) -> int:
    """Accumulate exact MTE1 + Cube work for one K-panel slice ``range(k_start, k_stop)``."""
    from .catlass_lower import _compute_b, _issued_ops, _mx_scale_group_span

    total_micros = 0
    if not spec.mx_scale_group:
        has_tail = k_stop == low.gk and (p.k % low.tk != 0)
        full_count = (k_stop - k_start) - int(has_tail)
        groups: list[tuple[int, int, int]] = []
        if full_count > 0:
            groups.append((low.tk, 0, full_count))
        if has_tail:
            groups.append((p.k - (low.gk - 1) * low.tk, 0, 1))
        for k_len, _, reps in groups:
            for start in range(0, k_len, low.micro_k):
                micro = min(k_len - start, low.micro_k)
                a = packed_bytes(tile.m * micro, spec.dtype_a)
                b = packed_bytes(tile.n * micro, _compute_b(spec))
                resource_work["mte1"] += (a + b) * reps
                plan.cube[cube_dt] += _issued_ops(tile.m, tile.n, micro, spec, datapath) * reps
            total_micros += ceil_div(k_len, low.micro_k) * reps
        return total_micros
    for p_idx in range(k_start, k_stop):
        ks = p_idx * low.tk
        k_len = min(low.tk, p.k - ks)
        k_mod = ks % MX_SCALE_GROUP
        for start in range(0, k_len, low.micro_k):
            micro = min(k_len - start, low.micro_k)
            sc_groups = _mx_scale_group_span(k_mod + start, micro)[1]
            a = packed_bytes(tile.m * micro, spec.dtype_a) + tile.m * sc_groups
            b = packed_bytes(tile.n * micro, _compute_b(spec)) + tile.n * sc_groups
            resource_work["mte1"] += a + b
            plan.cube[cube_dt] += _issued_ops(tile.m, tile.n, micro, spec, datapath)
        total_micros += ceil_div(k_len, low.micro_k)
    return total_micros


def _admit_single_panel(
    tile, p_idx: int, spec, p, low, die: int, sector: int, memo, admit, fast_admit_rect,
) -> _PanelDescriptor:
    ks = p_idx * low.tk
    kl = min(low.tk, p.k - ks)
    if fast_admit_rect is not None and not spec.mx_scale_group:
        rec_a = fast_admit_rect(False, tile, ks, kl)
        rec_b = fast_admit_rect(True, tile, ks, kl)
        return _PanelDescriptor(tile.m, tile.n, kl, rec_a, None, rec_b, None)
    pats = _panel_patterns(spec, p, tile, ks, kl, die, sector, memo)
    rec_a = admit(pats[0])
    rec_b = admit(pats[2])
    rec_sa = admit(pats[1]) if pats[1] is not None else None
    rec_sb = admit(pats[3]) if pats[3] is not None else None
    return _PanelDescriptor(
        tile.m, tile.n, kl, rec_a, rec_sa, rec_b, rec_sb,
        k_start_mod=(ks % MX_SCALE_GROUP if spec.mx_scale_group else 0),
    )


def _admit_init_panels_for_task(
    tile, k_start: int, init_n: int, spec, p, low, die: int, sector: int,
    memo, admit, fast_admit_rect, has_ctx: bool, *, init_check=None,
) -> list[_PanelDescriptor]:
    has_scales = bool(spec.mx_scale_group)
    if fast_admit_rect is not None and not has_scales:
        ks_list = tuple((idx * low.tk, min(low.tk, p.k - idx * low.tk)) for idx in range(k_start, k_start + init_n))
        recs: list[list[_AccessRecord | None]] = [[None, None] for _ in range(init_n)]
        if has_ctx:
            for i, (ks, kl) in enumerate(ks_list):
                recs[i][0] = fast_admit_rect(False, tile, ks, kl)
            for i, (ks, kl) in enumerate(ks_list):
                recs[i][1] = fast_admit_rect(True, tile, ks, kl)
        else:
            for i, (ks, kl) in enumerate(ks_list):
                recs[i][0] = fast_admit_rect(False, tile, ks, kl)
                recs[i][1] = fast_admit_rect(True, tile, ks, kl)
        if init_check is not None:
            init_check([recs[i][0] for i in range(init_n)], [recs[i][1] for i in range(init_n)])
        return [_PanelDescriptor(tile.m, tile.n, ks_list[i][1], recs[i][0], None, recs[i][1], None) for i in range(init_n)]
    pats_list = [
        (ks, min(low.tk, p.k - ks), _panel_patterns(spec, p, tile, ks, min(low.tk, p.k - ks), die, sector, memo))
        for p_idx in range(k_start, k_start + init_n)
        for ks in (p_idx * low.tk,)
    ]
    gm: list[list[_AccessRecord | None]] = [[None, None, None, None] for _ in range(init_n)]
    if has_ctx:
        for i in range(init_n):
            gm[i][0] = admit(pats_list[i][2][0])
        for i in range(init_n):
            gm[i][2] = admit(pats_list[i][2][2])
    else:
        for i in range(init_n):
            gm[i][0] = admit(pats_list[i][2][0])
            gm[i][2] = admit(pats_list[i][2][2])
    if has_scales:
        if has_ctx:
            for i in range(init_n):
                gm[i][1] = _admit_optional(admit, pats_list[i][2][1])
            for i in range(init_n):
                gm[i][3] = _admit_optional(admit, pats_list[i][2][3])
        else:
            for i in range(init_n):
                gm[i][1] = _admit_optional(admit, pats_list[i][2][1])
                gm[i][3] = _admit_optional(admit, pats_list[i][2][3])
    return [
        _PanelDescriptor(
            tile.m, tile.n, pats_list[i][1], *gm[i],
            k_start_mod=(pats_list[i][0] % MX_SCALE_GROUP if has_scales else 0),
        )
        for i in range(init_n)
    ]


def _lower_splitk_waves(
    blocks: int, cores: int, spec, p, datapath, low, plan: Lowering, admit,
    resource_work: Counter[str], cube_dt: str, die: int, has_ctx: bool,
    owner_cohort_id: dict[int, int], next_cohort_id: int, memo: _PatternMemo | None,
    fast_admit_rect, init_check=None,
) -> tuple[list[tuple[_CohortWaveSpec, ...]], dict[int, list[tuple[Any, int, AddressRun]]], int, int]:
    """Lower all ``multi_core_splitk`` partial waves in exact global Kahn topological order."""
    full_mc = ceil_div(low.tk, low.micro_k)
    periods = _CANDIDATE_PAIR_PERIODS_MC1 if full_mc == 1 else _CANDIDATE_PAIR_PERIODS
    base, rem = divmod(low.gk, spec.splitk_factor)
    stride = packed_bytes(p.n, spec.dtype_acc)
    matrix_bytes = p.m * stride
    total_tasks = spec.splitk_factor * blocks
    parts: dict[int, list[tuple[Any, int, AddressRun]]] = defaultdict(list)
    owner_depth: dict[int, int] = {}
    events: list[tuple[int, int, str, int, Any]] = []
    encoders: list[_PanelPhases] = []
    task_tiles: list[Any] = []
    stage_count = 0
    order_seq = 0
    for task_id in range(total_tasks):
        split, block = divmod(task_id, blocks)
        st_k = split * base + min(split, rem)
        en_k = st_k + base + (split < rem)
        n_panels = en_k - st_k
        owner = task_id % cores
        tile = replace(low.tile(block), owner=owner)
        offset = split * matrix_bytes + tile.row * stride + packed_bytes(tile.col, spec.dtype_acc)
        row_bytes = packed_bytes(tile.n, spec.dtype_acc)
        if offset + (tile.m - 1) * stride + row_bytes > spec.workspace_bytes:
            raise ModelingError("catlass_partial_exceeds_workspace")
        run = AddressRun(offset, tile.m, row_bytes, stride)
        size = tile.m * row_bytes
        drain_pat = _make_partial_pattern(
            spec, p, tile, run, size, die, plan.sector,
            direction="write", pipe="fixpipe", owner=owner, memo=memo,
        )
        micros = _count_slice_compute(tile, st_k, en_k, spec, p, datapath, low, plan, resource_work, cube_dt)
        stage_count += _exact_tile_stage_count(
            spec, n_panels, micros, has_ctx, scale_panels=_scale_panel_count(spec, st_k, en_k),
        )
        encoders.append(_PanelPhases(periods))
        task_tiles.append(tile)
        d_init, p_depths, d_partial = _partial_kahn_depths(n_panels, full_mc, owner_depth.get(owner, 0), has_ctx)
        init_n = min(_PAIR_DEPTH, n_panels)
        events.append((d_init, order_seq, "init", task_id, (tile, st_k, init_n)))
        order_seq += 1
        for off_idx, p_depth in enumerate(p_depths):
            events.append((p_depth, order_seq, "panel", task_id, (tile, st_k + _PAIR_DEPTH + off_idx)))
            order_seq += 1
        events.append((d_partial, order_seq, "drain", task_id, drain_pat))
        order_seq += 1
        owner_depth[owner] = d_partial + 1
        parts[block].append((tile, size, run))

    events.sort(key=lambda item: (item[0], item[1]))
    drains: list[_AccessRecord | None] = [None] * total_tasks
    for _, _, kind, task_id, payload in events:
        enc = encoders[task_id]
        if kind == "init":
            tile, k_start, init_n = payload
            for desc in _admit_init_panels_for_task(
                tile, k_start, init_n, spec, p, low, die, plan.sector, memo, admit,
                fast_admit_rect, has_ctx, init_check=init_check,
            ):
                enc.append(desc)
        elif kind == "panel":
            tile, p_idx = payload
            enc.append(_admit_single_panel(
                tile, p_idx, spec, p, low, die, plan.sector, memo, admit, fast_admit_rect,
            ))
        else:
            drains[task_id] = admit(payload)

    waves: list[tuple[_CohortWaveSpec, ...]] = []
    for first in range(0, total_tasks, cores):
        cohorts = {}
        for task_id in range(first, min(first + cores, total_tasks)):
            tile = task_tiles[task_id]
            drain = drains[task_id]
            steps, signatures = encoders[task_id].finish()
            key = (owner_cohort_id.get(tile.owner, 0), signatures, _ledger_key(drain))
            owners, _, _ = cohorts.setdefault(key, ([], steps, drain))
            owners.append(tile.owner)
        wave_specs = tuple(_CohortWaveSpec(tuple(owners), steps, drain) for owners, steps, drain in cohorts.values())
        for cohort in wave_specs:
            for owner in cohort.cores:
                owner_cohort_id[owner] = next_cohort_id
            next_cohort_id += 1
        waves.append(wave_specs)
    return waves, parts, stage_count, next_cohort_id


def _partial_kahn_depths(n_panels: int, full_mc: int, base_depth: int, has_ctx: bool) -> tuple[int, list[int], int]:
    """Exact Kahn topological depths for ``(init_depth, panel_depths[2:], partial_drain_depth)``."""
    d_init = base_depth + (_PAIR_DEPTH if has_ctx else 1)
    d_loaded_panel: list[int] = []
    d_mma_hist: list[int] = []
    panel_depths: list[int] = []
    for p_off in range(n_panels):
        d_gm = d_init if p_off < _PAIR_DEPTH else (d_loaded_panel[p_off - _PAIR_DEPTH] + _PAIR_DEPTH)
        if p_off >= _PAIR_DEPTH:
            panel_depths.append(d_gm)
        d_loaded = d_gm
        for u in range(full_mc):
            m_idx = p_off * full_mc + u
            d_l0_credit = (d_mma_hist[m_idx - _PAIR_DEPTH] + _PAIR_DEPTH) if m_idx >= _PAIR_DEPTH else d_init
            d_l0 = (d_gm if d_gm >= d_l0_credit else d_l0_credit) + 1
            d_loaded = d_l0 + 1
            d_mma = ((d_loaded if d_loaded >= d_mma_hist[-1] else d_mma_hist[-1]) + 1) if d_mma_hist else (d_loaded + 1)
            d_mma_hist.append(d_mma)
        d_loaded_panel.append(d_loaded)
    d_partial = d_mma_hist[-1] + _PAIR_DEPTH
    return d_init, panel_depths, d_partial


def _lower_streamk_partials(
    partition, spec, p, datapath, low, plan: Lowering, admit, resource_work: Counter[str],
    cube_dt: str, die: int, has_ctx: bool, owner_cohort_id: dict[int, int],
    memo: _PatternMemo | None, fast_admit_rect, init_check=None,
) -> tuple[tuple[_StreamkPartialCohortSpec, ...], dict[int, list[tuple[Any, int, AddressRun]]], int]:
    """Lower all StreamK partial slices per core in exact Kahn depth order, grouped into cohorts."""
    full_mc = ceil_div(low.tk, low.micro_k)
    periods = _CANDIDATE_PAIR_PERIODS_MC1 if full_mc == 1 else _CANDIDATE_PAIR_PERIODS
    stride = packed_bytes(low.tn, spec.dtype_acc)
    tile_bytes = low.tm * stride
    parts: dict[int, list[tuple[Any, int, AddressRun]]] = defaultdict(list)
    events: list[tuple[int, int, str, int, int, Any]] = []
    core_encs: dict[int, list[_PanelPhases]] = {}
    core_specs: dict[int, list[tuple[tuple[_StepPhaseDescriptor, ...], tuple[Any, ...], _AccessRecord]]] = {}
    stage_count = 0
    order_seq = 0
    for core, (st_idx, en_idx) in enumerate(partition.intervals):
        n_parts = partition.partial_counts[core]
        if not n_parts:
            continue
        core_encs[core] = []
        core_specs[core] = []
        base_depth = 0
        for part in range(n_parts):
            block, panel = divmod(st_idx, low.gk)
            stop_idx = min(en_idx, (block + 1) * low.gk)
            k_stop = panel + stop_idx - st_idx
            n_panels = k_stop - panel
            tile = replace(low.tile(partition.b_norm + block), owner=core)
            offset = (core * STREAMK_PARTIAL_SLOTS_PER_CORE + part) * tile_bytes
            row_bytes = packed_bytes(tile.n, spec.dtype_acc)
            if offset + (tile.m - 1) * stride + row_bytes > spec.workspace_bytes:
                raise ModelingError("catlass_partial_exceeds_workspace")
            run = AddressRun(offset, tile.m, row_bytes, stride)
            size = tile.m * row_bytes
            drain_pat = _make_partial_pattern(
                spec, p, tile, run, size, die, plan.sector,
                direction="write", pipe="fixpipe", owner=core, memo=memo,
            )
            micros = _count_slice_compute(tile, panel, k_stop, spec, p, datapath, low, plan, resource_work, cube_dt)
            stage_count += _exact_tile_stage_count(
                spec, n_panels, micros, has_ctx, scale_panels=_scale_panel_count(spec, panel, k_stop),
            )
            core_encs[core].append(_PanelPhases(periods))
            d_init, p_depths, d_partial = _partial_kahn_depths(n_panels, full_mc, base_depth, has_ctx)
            init_n = min(_PAIR_DEPTH, n_panels)
            events.append((d_init, order_seq, "init", core, part, (tile, panel, init_n)))
            order_seq += 1
            for off_idx, p_depth in enumerate(p_depths):
                events.append((p_depth, order_seq, "panel", core, part, (tile, panel + _PAIR_DEPTH + off_idx)))
                order_seq += 1
            events.append((d_partial, order_seq, "drain", core, part, drain_pat))
            order_seq += 1
            base_depth = d_partial + 1
            parts[block].append((tile, size, run))
            st_idx = stop_idx

    events.sort(key=lambda item: (item[0], item[1]))
    for _, _, kind, core, part, payload in events:
        enc = core_encs[core][part]
        if kind == "init":
            tile, k_start, init_n = payload
            for desc in _admit_init_panels_for_task(
                tile, k_start, init_n, spec, p, low, die, plan.sector, memo, admit,
                fast_admit_rect, has_ctx, init_check=init_check,
            ):
                enc.append(desc)
        elif kind == "panel":
            tile, p_idx = payload
            enc.append(_admit_single_panel(
                tile, p_idx, spec, p, low, die, plan.sector, memo, admit, fast_admit_rect,
            ))
        else:
            drain = admit(payload)
            phases, sigs = enc.finish()
            core_specs[core].append((phases, sigs, drain))
    cohorts: dict[Any, tuple[list[int], tuple[tuple[tuple[_StepPhaseDescriptor, ...], _AccessRecord], ...]]] = {}
    for core, items in core_specs.items():
        if not items:
            continue
        key = (owner_cohort_id.get(core, 0), tuple((sigs, _ledger_key(dr)) for _, sigs, dr in items))
        rep_parts = tuple((ph, dr) for ph, _, dr in items)
        owners, _ = cohorts.setdefault(key, ([], rep_parts))
        owners.append(core)
    compiled = tuple(_StreamkPartialCohortSpec(tuple(owners), rep_parts) for owners, rep_parts in cohorts.values())
    return compiled, parts, stage_count


def _lower_reduce_tasks(
    parts: dict[int, list[tuple[Any, int, AddressRun]]], spec, p, plan: Lowering,
    admit, resource_work: Counter[str], die: int, memo: _PatternMemo | None,
) -> tuple[tuple[_SplitkReduceSpec, ...], int]:
    """Lower AIV partial reduction batches in exact Kahn window order."""
    block_keys = list(parts)
    aiv_step = max(1, spec.aiv_budget)
    tasks: list[_SplitkReduceSpec] = []
    stage_count = 0
    for first in range(0, len(block_keys), aiv_step):
        batch = block_keys[first:first + aiv_step]
        batch_loads: list[tuple[int, int, Any, tuple[_AccessRecord, ...], int]] = []
        for block in batch:
            items = parts[block]
            tile = items[0][0]
            owner = block % spec.aiv_budget
            reads = tuple(
                admit(_make_partial_pattern(
                    spec, p, tile, run, size, die, plan.sector,
                    direction="read", pipe="mte2_aiv", owner=owner, memo=memo,
                ))
                for _, size, run in items
            )
            reduce_ops = (len(items) - 1) * tile.m * tile.n
            if reduce_ops > 0:
                plan.vector["vec_reduce"] += reduce_ops
                resource_work["vec_reduce"] += reduce_ops
            batch_loads.append((block, owner, tile, reads, reduce_ops))
            stage_count += len(items) + _REDUCE_OVERHEAD_STAGES
        for block, owner, tile, reads, reduce_ops in batch_loads:
            out_rec = admit(_make_output_pattern(
                spec, p, tile, die, plan.sector, memo=memo, pipe="mte3", owner=owner,
            ))
            tasks.append(_SplitkReduceSpec(block, owner, reads, reduce_ops, out_rec))
    return tuple(tasks), stage_count


def lower_catlass_periodic(
    spec, p, datapath, solution, *,
    context=None, memory_view=None, die: int = 0, region_id=None,
) -> TemplateLoweringResult:
    """Lower an eligible double-buffered CATLASS matmul into exact periodic waves.

    [Cohesive-Pipeline] Single-pass exact CATLASS periodic wave, StreamK, and Split-K lowering over shared L2 residency state.
    """
    from .catlass_lower import _MatrixLowerer, _compute_b, _cube_dtype, _scale

    saved_initial = deepcopy(memory_view) if memory_view is not None else None
    plan = Lowering(datapath, context=context, memory_view=memory_view, die=die, live_admission=False)
    entry_state = deepcopy(plan.state)
    low = _MatrixLowerer(plan, spec, p)
    die_mem_access = plan.state.memory_access_for(die)
    memo = _build_pattern_memo(spec, p, die, plan.sector, die_mem_access)
    resource_work, cube_dt = Counter(), _cube_dtype(spec)
    blocks, cores = ceil_div(p.m, low.tm) * ceil_div(p.n, low.tn), spec.aic_budget
    lifetimes = _ReuseLifetimes(plan.state, ())
    first_tile = low.tile(0)
    scale_panel_total = _scale_panel_count(spec, 0, low.gk)
    for pattern, count in zip(
        _panel_patterns(spec, p, first_tile, 0, low.tk, die, plan.sector, memo),
        (low.gk, scale_panel_total, low.gk, scale_panel_total),
    ):
        if pattern is not None and pattern.reuse_key is not None:
            lifetimes.pending[pattern.reuse_key] += blocks * count
    lifetimes.pending[_make_output_pattern(spec, p, first_tile, die, plan.sector, memo=memo).reuse_key] += blocks
    partition = streamk_partition(blocks, low.gk, cores) if spec.scheduler == "streamk" else None
    if partition is not None and partition.b_sk:
        scratch_key = ReuseKey(die, p.c_id + ":splitk", 0, reuse_stream=f"{p.c_id}:splitk:0")
        lifetimes.pending[scratch_key] += _OPERAND_PAIR_COUNT * partition.partial_count
    elif spec.scheduler == "multi_core_splitk":
        scratch_key = ReuseKey(die, p.c_id + ":splitk", 0, reuse_stream=f"{p.c_id}:splitk:0")
        lifetimes.pending[scratch_key] += _OPERAND_PAIR_COUNT * spec.splitk_factor * blocks
    entry_empty = not getattr(entry_state, "_resident", None)
    if entry_empty:
        whole_reads, whole_writes = _whole_tensor_obligations(spec, p, die, plan.sector)
        plan.reads.extend(whole_reads)
        plan.writes.extend(whole_writes)

    def admit(pat: MemoryAccessPattern) -> _AccessRecord:
        return _admit_periodic_access(
            pat, plan, entry_state, lifetimes, resource_work, context, die,
            track_obligations=not entry_empty,
        )

    use_fast = (
        entry_empty and not spec.mx_scale_group and not plan.burst_rules
        and memo.a_elem is not None and memo.b_elem is not None
        and hasattr(plan.state, "admit_rect_read_ledger")
    )
    fast_admit_rect, flush_service = (
        _make_fast_rect_admitter(spec, p, plan, memo, lifetimes, resource_work, context, die, admit)
        if use_fast else (None, None)
    )
    init_safe = [use_fast and p.a_id != p.b_id]
    first_tile_seen = [False]
    rkey_a = plan.state.resolve_reuse_key(memo.key_a)
    rkey_b = None if memo.key_b is None else plan.state.resolve_reuse_key(memo.key_b)

    def init_check(a_recs: list[_AccessRecord], b_recs: list[_AccessRecord]) -> None:
        _check_init_order_safe(init_safe, first_tile_seen, plan.state, die, rkey_a, rkey_b, a_recs, b_recs)

    compiled_waves: list[tuple[_CohortWaveSpec, ...]] = []
    tail_waves: list[tuple[_CohortWaveSpec, ...]] = []
    streamk_partials: tuple[_StreamkPartialCohortSpec, ...] = ()
    streamk_partial_count: int | None = None
    reduce_tasks: tuple[_SplitkReduceSpec, ...] = ()
    has_cross_core_flag = False
    total_stages, plan.launches = _LAUNCH_STAGES, 1
    owner_cohort_id: dict[int, int] = {}
    core_tile_counts: Counter[int] = Counter()
    next_cohort_id = 1
    wave_micros = 0
    is_mutex = spec.dispatch_policy == "MmadPingpongMutex"
    has_ctx = context is not None

    def _lower_normal_range(start: int, stop: int, dest: list[tuple[_CohortWaveSpec, ...]]) -> None:
        nonlocal total_stages, next_cohort_id, wave_micros
        for first in range(start, stop, cores):
            tiles = tuple(low.tile(i) for i in range(first, min(first + cores, stop)))
            if is_mutex:
                for tile in tiles:
                    core_tile_counts[tile.owner] += 1
            wave_specs, wave_stages, next_cohort_id, wave_micros = _lower_periodic_wave(
                tiles, spec, p, datapath, low, plan, admit,
                resource_work, cube_dt, die, has_ctx,
                owner_cohort_id, next_cohort_id, memo,
                fast_admit_rect=fast_admit_rect, init_check=init_check if use_fast else None,
            )
            dest.append(wave_specs)
            if not is_mutex:
                total_stages += wave_stages

    if partition is not None and partition.b_sk:
        final = max(0, partition.b_norm - cores)
        _lower_normal_range(0, final, compiled_waves)
        streamk_partials, parts, part_stages = _lower_streamk_partials(
            partition, spec, p, datapath, low, plan, admit, resource_work,
            cube_dt, die, has_ctx, owner_cohort_id, memo, fast_admit_rect,
            init_check=init_check if use_fast else None,
        )
        streamk_partial_count = partition.partial_count
        has_cross_core_flag = True
        total_stages += part_stages + _FLAG_STAGE_COUNT
        reduce_tasks, red_stages = _lower_reduce_tasks(parts, spec, p, plan, admit, resource_work, die, memo)
        total_stages += red_stages
        owner_cohort_id.clear()
        _lower_normal_range(final, partition.b_norm, tail_waves)
    elif spec.scheduler == "multi_core_splitk":
        sk_waves, parts_sk, sk_stages, next_cohort_id = _lower_splitk_waves(
            blocks, cores, spec, p, datapath, low, plan, admit, resource_work,
            cube_dt, die, has_ctx, owner_cohort_id, next_cohort_id, memo,
            fast_admit_rect, init_check=init_check if use_fast else None,
        )
        compiled_waves.extend(sk_waves)
        has_cross_core_flag = True
        total_stages += sk_stages + _FLAG_STAGE_COUNT
        reduce_tasks, red_stages = _lower_reduce_tasks(parts_sk, spec, p, plan, admit, resource_work, die, memo)
        total_stages += red_stages
    else:
        _lower_normal_range(0, blocks, compiled_waves)
        if is_mutex:
            total_stages += _exact_mutex_stage_count(spec, core_tile_counts, low.gk, wave_micros, has_ctx)

    if flush_service is not None:
        flush_service()
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
        float(plan.sectors), plan.misses, float(plan.service),
    )
    program = _PeriodicCatlassProgram(
        spec, p, datapath, context, die, plan.hbm_bus, tuple(compiled_waves), slot_sizes,
        plan.burst_rules, summary, tuple(sorted(resource_work.items())), saved_initial,
        streamk_partials=streamk_partials,
        streamk_partial_count=streamk_partial_count,
        reduce_tasks=reduce_tasks,
        tail_waves=tuple(tail_waves),
        has_cross_core_flag=has_cross_core_flag,
        init_order_invariant=init_safe[0],
        knee_curve=plan.state._knee_curves_by_die.get(die, ()),
        granule_bytes=plan.state._granule_by_die.get(die, plan.state.granule_bytes),
        mem_access_die=die_mem_access,
    )
    return TemplateLoweringResult(program, summary, total_stages)
