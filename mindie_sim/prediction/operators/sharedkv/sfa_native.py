"""Arch35 split-G 2-AIC/4-AIV pipelined sparse MLA attention (BF16 & C8).

[Role] Lower aclnnSparseFlashAttention and aclnnKvQuantSparseFlashAttention
  on npu_arch_3510 using the native split-G (G>64 -> 2 AIC + 4 AIV per query)
  3-slot 128-row pipelined GM/L2 KV ring.
[Invariants] Work declared in OperatorPlan.work matches lowered stages
  identically. Split-G partitions heads as (ceil(G/2), floor(G/2)) across
  adjacent AICs while 4 AIVs gather/dequantize 32 rows each per 128-row tile;
  non-quant SFA routes KV through the 3-slot GM/L2 ring in both G<=64 and
  G>64 modes. Zero hardcoded setup constants or empirical service binders;
  supply rates come from the active datapath or L0 table.
[Timeouts & Locks] Finite O(tiles) pipeline over selected rows (<= topk).
[Side-Effect Policy] Pure planning and program construction.
Upstream-ABI: aclnnKvQuantSparseFlashAttention
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from math import ceil
from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...runtime.engine import Engine, Resource
from ...runtime.l2_residency import (
    AdmissionMemoryView,
    RawDieL2Bytes,
    _engine_primary_pool_namespace,
    _row_pool_namespace,
    resolve_engine_admission_view,
)
from ...runtime.memory_physics import AddressRun, AddressRunPlan, HbmBusSpec, MemoryAccessPattern, ReuseKey
from ..attention.common import (
    BF16_BYTES,
    C8_ABI,
    C8_MODES,
    C8_V1_ABI,
    FP32_BYTES,
    INDEX_BYTES,
    PAGED_LAYOUTS,
    QUANT_MODE_BF16,
    QUANT_MODE_TQ4,
    SCALE_ADDRESS_BYTES,
    SECTOR_BYTES,
    SFA_V1_ABI,
    SOFTMAX_OPS,
    AttentionContract,
    AttentionPhase,
    _stage,
    make_plan,
    owner_pairs,
    packed_gather_run_plans,
    packed_selection_runs,
    read_work,
    unmodeled,
    workspace_write_work,
)
from ..contracts import OperatorPlanningContext

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SFA_TILING_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp",
    "e87df78987dabb50064fc8361b51463dbd3490f7ba1c5d66a0b85c9721df0259",
)
SFA_KERNEL_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla_arch35.h",
    "651400b9a8aa23290fb5646c38aa7c9ebca85e352f484ce409b7fe426f52ec66",
)
C8_TILING_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp",
    "2c75acce70974165d5b087416d724bcd0d09909b2e8dd693370b5860de9dbc6e",
)
C8_KERNEL_SOURCE = SourceRef(
    "cann/ops-transformer",
    "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
    "attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_kernel_mla_arch35.h",
    "89f70d3db0344bdd44bef92b3108007225a84f342e0995ed724654dbada278fa",
)
SOURCES: tuple[SourceRef, ...] = (
    SFA_TILING_SOURCE,
    SFA_KERNEL_SOURCE,
    C8_TILING_SOURCE,
    C8_KERNEL_SOURCE,
)

SFA_TILE_ROWS = 128
SFA_RING_SLOTS = 3
SFA_SPLIT_G_HEADS = 64
C8_PACKED_ROW_BYTES = 656
TQ4_PACKED_ROW_BYTES = 392
C8_DEQUANT_OPS_PER_ELEM = 11
C8_BLOCK_TABLE_ALIGN_BYTES = 512
C8_VEC_UB_RESERVE_BYTES = 8 * 1024
SFA_OUTPUT_MERGE_OPS_PER_ELEM = 3
SFA_SSMAX_EXTRA_OPS = 2
SFA_LSE_STAT_TENSORS = 2
FLOPS_PER_FMA = 2
FRACTAL_ALIGN = 16
FIXPIPE_M_ALIGN = 2
FIXPIPE_N_ALIGN = 8
SPLIT_G_AICS_PER_QUERY = 2
SPLIT_G_AIVS_PER_QUERY = 4
NON_SPLIT_G_AIVS_PER_QUERY = 2
NON_SPLIT_G_ROWS_PER_LOOP = 2
HALF_DIVISOR = 2
AIV_PER_HEAD_PHASES = frozenset({
    "online_softmax", "probability_transfer", "online_output", "normalize", "output",
})


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _l2_capacity_bytes(ctx: OperatorPlanningContext) -> int:
    return int(ctx.geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0)))


def _effective_l2_capacity_bytes(ctx: OperatorPlanningContext, die: int) -> int:
    if ctx.admission_memory_view is not None:
        return int(ctx.admission_memory_view.effective_l2_bytes(die))
    return _l2_capacity_bytes(ctx)


def _split_g_head_slices(num_heads: int, aics_per_group: int) -> tuple[int, ...]:
    """Return exact per-AIC head counts matching ``ComputeSouterParam``."""
    if aics_per_group > 1:
        first_half = ceil(num_heads / HALF_DIVISOR)
        return (first_half, num_heads - first_half)
    return (num_heads,)


def _subblock_head_slices(heads: int, n_aivs: int) -> tuple[int, ...]:
    """Split one AIC's heads across its paired AIV subblocks matching ``ComputeSouterParam``."""
    if n_aivs >= NON_SPLIT_G_AIVS_PER_QUERY:
        first = ceil(heads / HALF_DIVISOR)
        second = heads - first
        return (first, second) if second > 0 else (first,)
    return (heads,)


def _partition_query_groups(
    num_queries: int,
    raw_owners: tuple[tuple[int, tuple[int, ...]], ...],
    desired_aics: int,
) -> tuple[tuple[tuple[tuple[int, tuple[int, ...]], ...], ...], tuple[range, ...]]:
    """Partition queries into contiguous group slices matching ``InitCalcParamsEach``."""
    complete_owners = tuple(
        (aic, aivs) for aic, aivs in raw_owners if len(aivs) >= NON_SPLIT_G_AIVS_PER_QUERY
    )
    if len(complete_owners) < desired_aics:
        if desired_aics > 1:
            raise unmodeled(
                "attention_sfa_split_g_insufficient_aics",
                f"Split-G (num_heads > {SFA_SPLIT_G_HEADS}) requires at least {desired_aics} AICs "
                f"with {NON_SPLIT_G_AIVS_PER_QUERY} paired AIVs each, got {len(complete_owners)}",
            )
        raise unmodeled(
            "attention_sfa_insufficient_aivs",
            f"Arch35 SFA requires at least {NON_SPLIT_G_AIVS_PER_QUERY} paired AIVs per AIC",
        )
    avail_groups = len(complete_owners) // desired_aics
    if num_queries <= 0:
        return (complete_owners[:desired_aics],), (range(0),)
    queries_per_group = ceil(num_queries / avail_groups) if num_queries > avail_groups else 1
    active_groups = ceil(num_queries / queries_per_group)
    groups = tuple(
        complete_owners[g * desired_aics : (g + 1) * desired_aics]
        for g in range(active_groups)
    )
    group_slices = tuple(
        range(g * queries_per_group, min(num_queries, (g + 1) * queries_per_group))
        for g in range(active_groups)
    )
    return groups, group_slices


def _resolve_engine_memory_view(
    engine, die: int, capacity: int, hbm: str, ctx: OperatorPlanningContext,
) -> AdmissionMemoryView:
    req_ns = _row_pool_namespace(hbm)
    ctx_view = ctx.admission_memory_view
    ctx_exact = (
        bool(req_ns)
        and isinstance(ctx_view, AdmissionMemoryView)
        and ctx_view.pool_namespace == req_ns
        and ctx_view.has_die(die)
    )
    attached = getattr(engine, "admission_memory_view", None)
    if attached is not None:
        owner_ns = getattr(attached, "pool_namespace", "") or _engine_primary_pool_namespace(engine)
        attached_compatible = (
            not req_ns
            or owner_ns == req_ns
            or (owner_ns is None and not ctx_exact)
        )
        if attached_compatible and attached.has_die(die):
            return attached
        if ctx_exact:
            return ctx_view
        if isinstance(ctx_view, AdmissionMemoryView) and ctx_view.has_die(die):
            ctx_ns = ctx_view.pool_namespace
            if not req_ns or not ctx_ns or ctx_ns == req_ns:
                return ctx_view
    views = getattr(engine, "_fia_admission", None)
    if views is None:
        views = engine._fia_admission = {}
    key = (hbm, die)
    view = views.get(key)
    if view is None:
        if attached is None and isinstance(ctx_view, AdmissionMemoryView) and ctx_view.has_die(die):
            ctx_ns = ctx_view.pool_namespace
            if not req_ns or not ctx_ns or ctx_ns == req_ns:
                view = copy.deepcopy(ctx_view)
        if view is None:
            view = resolve_engine_admission_view(
                engine,
                die=die,
                raw_l2=RawDieL2Bytes(capacity),
                pool_namespace=req_ns,
            )
            if view.arch_id is None:
                view.arch_id = str(ctx.compute_arch.arch_id)
        views[key] = view
    return view


def _record_delivery(acc: list[int], ledger) -> None:
    acc[0] += int(ledger.aligned_transfer_bytes)
    acc[1] += int(ledger.dram_sector_bytes)
    acc[2] += int(ledger.l2_hit_bytes)
    acc[3] += int(ledger.dram_miss_copy_bytes)


def _packed_row_bytes(contract: AttentionContract) -> int:
    if contract.abi == SFA_V1_ABI:
        if contract.quant_mode != QUANT_MODE_BF16:
            raise unmodeled(
                "attention_sfa_v1_quant_mode_unsupported",
                f"{SFA_V1_ABI} requires non-quantized quant_mode={QUANT_MODE_BF16}, got {contract.quant_mode}",
            )
        return contract.head_dim * BF16_BYTES
    if contract.quant_mode == QUANT_MODE_BF16:
        return contract.head_dim * BF16_BYTES
    if contract.quant_mode == QUANT_MODE_TQ4:
        return TQ4_PACKED_ROW_BYTES
    if contract.quant_mode in C8_MODES:
        return C8_PACKED_ROW_BYTES
    raise unmodeled(
        "attention_sfa_quant_mode_unsupported",
        f"{contract.abi} does not support quant_mode={contract.quant_mode}",
    )


@dataclass(frozen=True, slots=True)
class _SfaSelectionRead:
    pattern: MemoryAccessPattern
    bound: object
    capacity: int
    hbm: str
    ctx: OperatorPlanningContext
    collector: list[int] | None = None

    def start(self, engine, name, ready, *, owner=""):
        view = _resolve_engine_memory_view(engine, self.pattern.die, self.capacity, self.hbm, self.ctx)
        on_delivery = (
            (lambda ledger, _res, _extra, acc=self.collector: _record_delivery(acc, ledger))
            if self.collector is not None else None
        )
        return engine.submit_bound_access(
            name, self.bound, after=(ready,), owner=owner, memory_state=view,
            pattern=self.pattern, on_delivery=on_delivery,
        )


@dataclass(frozen=True, slots=True)
class _SfaRingReservedProgram:
    body: object
    die: int
    ring_bytes: int
    capacity: int
    hbm: str
    ctx: OperatorPlanningContext
    alloc_id: str

    def start(self, engine, name, ready, *, owner=""):
        view = _resolve_engine_memory_view(engine, self.die, self.capacity, self.hbm, self.ctx)
        if self.ring_bytes <= 0 or view is None:
            return self.body.start(engine, name, ready, owner=owner)
        res_id = f"{self.alloc_id}:{name}"

        def _acquire(_ev):
            view.grant_reservation(self.die, res_id, self.ring_bytes)

        def _release(_ev):
            view.release_reservation(self.die, res_id)

        acq = engine.submit(
            f"{name}:ring_reserve", {}, after=(ready,), owner=owner,
            on_complete=_acquire, native_hook=True,
        )
        done = self.body.start(engine, name, acq, owner=owner)
        return engine.submit(
            f"{name}:ring_release", {}, after=(done,), owner=owner,
            on_complete=_release, native_hook=True,
        )


def _partition_tile_rows(
    n_rows: int, aivs: tuple[int, ...], aics_per_group: int,
) -> tuple[tuple[int, int, int], ...]:
    """Partition selected tile rows across AIVs matching ``CalSparseCalSize``."""
    if n_rows <= 0 or not aivs:
        return ()
    if aics_per_group > 1 and len(aivs) >= SPLIT_G_AIVS_PER_QUERY:
        first_aic = ceil(n_rows / HALF_DIVISOR)
        second_aic = n_rows - first_aic
        v0_end = ceil(first_aic / HALF_DIVISOR)
        v2_end = first_aic + ceil(second_aic / HALF_DIVISOR)
        spans = (
            (aivs[0], 0, v0_end),
            (aivs[1], v0_end, first_aic),
            (aivs[2], first_aic, v2_end),
            (aivs[3], v2_end, n_rows),
        )
    elif len(aivs) >= NON_SPLIT_G_AIVS_PER_QUERY:
        s2_loops = ceil(ceil(n_rows / NON_SPLIT_G_AIVS_PER_QUERY) / NON_SPLIT_G_ROWS_PER_LOOP)
        split = min(s2_loops * NON_SPLIT_G_ROWS_PER_LOOP, n_rows)
        spans = ((aivs[0], 0, split), (aivs[1], split, n_rows))
    else:
        spans = ((aivs[0], 0, n_rows),)
    return tuple((aiv, lo, hi) for aiv, lo, hi in spans if hi > lo)


def _selection_aiv_steps(
    selection_slice: tuple[int, ...], contract: AttentionContract, is_quant: bool, row_bytes: int,
) -> tuple[tuple[str, int, AddressRunPlan], ...]:
    """Build per-AIV tensor read steps; BF16 separates ``keyGm`` and ``keyRopeGm``."""
    if not selection_slice:
        return ()
    if is_quant:
        plans = tuple(packed_gather_run_plans(packed_selection_runs(selection_slice, row_bytes)))
        return tuple((contract.kv_tensor_id, row_bytes, plan) for plan in plans)
    latent_b = contract.v_head_dim * BF16_BYTES
    rope_b = max(0, contract.head_dim - contract.v_head_dim) * BF16_BYTES
    lat_plans = tuple(packed_gather_run_plans(packed_selection_runs(selection_slice, latent_b)))
    if rope_b <= 0:
        return tuple((contract.kv_tensor_id, latent_b, plan) for plan in lat_plans)
    rope_addrs = tuple((addr // latent_b) * rope_b for addr in selection_slice)
    rope_plans = tuple(packed_gather_run_plans(packed_selection_runs(rope_addrs, rope_b)))
    rope_tid = f"{contract.kv_tensor_id}:rope"
    steps: list[tuple[str, int, AddressRunPlan]] = []
    for idx in range(max(len(lat_plans), len(rope_plans))):
        if idx < len(lat_plans):
            steps.append((contract.kv_tensor_id, latent_b, lat_plans[idx]))
        if idx < len(rope_plans):
            steps.append((rope_tid, rope_b, rope_plans[idx]))
    return tuple(steps)


def _lower_selection_gather(
    context, selection_slice: tuple[int, ...], grp_aivs: tuple[int, ...],
    aics_per_group: int, contract: AttentionContract, is_quant: bool, row_bytes: int,
    ctx: OperatorPlanningContext, *, collector: list[int] | None = None,
):
    """Lower per-AIV partitioned selection gathers onto Engine completion-time L2."""
    die = int(ctx.geometry.rank or 0)
    hbm_name = context.resource_names["hbm"]
    pipe_role = "mte2_aiv" if "mte2_aiv" in context.resource_names else "mte2"
    pipe_row = context.resource_names[pipe_role]
    pipe_units = max(1, int(context.resource_units.get(
        pipe_row, context.resource_units.get(pipe_role, context.resource_units.get("mte2", 1))
    )))
    pipe_rate = float(context.rates[pipe_row]) / float(pipe_units)
    bus = HbmBusSpec(die, 1, 1, 0, row_r=hbm_name, row_w=hbm_name, ctr_r=hbm_name, ctr_w=hbm_name)
    l2_cap = _l2_capacity_bytes(ctx)
    aiv_programs = []
    for aiv_id, lo, hi in _partition_tile_rows(len(selection_slice), grp_aivs, aics_per_group):
        owner = unit_group(aiv_id, ctx.datapath.aiv_count, pipe_units)[0]
        steps = _selection_aiv_steps(selection_slice[lo:hi], contract, is_quant, row_bytes)
        read_stages = []
        prev_name: str | None = None
        for idx, (alloc_id, width, plan) in enumerate(steps):
            key = ReuseKey(die, alloc_id, contract.kv_version)
            pat = MemoryAccessPattern(
                key.backing_allocation, key.tensor_version, die, owner, "read", pipe_role,
                plan.runs[0].rows * width, plan, reuse_key=key,
            )
            bound = context.bind_access(
                pat, hbm_bus=bus,
                copy_hit_bytes_per_ms=pipe_rate, copy_miss_bytes_per_ms=pipe_rate,
            )
            st_name = "read" if idx == 0 else f"read-{idx}"
            read_stages.append(Stage(
                st_name,
                after=(prev_name,) if prev_name is not None else (),
                program=_SfaSelectionRead(pat, bound, l2_cap, hbm_name, ctx, collector=collector),
            ))
            prev_name = st_name
        if read_stages:
            aiv_programs.append(KernelProgram(tuple(read_stages)))
    if len(aiv_programs) == 1:
        return aiv_programs[0]
    return ParallelProgram(tuple(aiv_programs))


def _place_owners(context, aics: tuple[int, ...], aivs: tuple[int, ...]) -> dict[str, tuple[int, ...]]:
    place: dict[str, tuple[int, ...]] = {}
    for role in ("cube", "mte1", "mte2_aic", "fixpipe"):
        place.update(context.placement(role, aics))
    for role in ("vector", "mte2_aiv", "mte3"):
        place.update(context.placement(role, aivs))
    return place


def _selection_tile_cold_gathers(
    contract: AttentionContract,
    counts: tuple[int, ...],
    groups: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...],
    group_slices: tuple[range, ...],
    aics_per_group: int,
    is_quant: bool,
    row_bytes: int,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Compute exact cold gather bytes per (query, tile) for unnamed selection_lists."""
    q_to_aivs: dict[int, tuple[int, ...]] = {}
    for grp, sl in zip(groups, group_slices, strict=True):
        grp_aivs = tuple(v for p in grp for v in p[1])
        for q_idx in sl:
            q_to_aivs[q_idx] = grp_aivs
    result: list[tuple[tuple[int, int, int, int], ...]] = []
    for q_idx, rows in enumerate(counts):
        if q_idx >= len(contract.selection_lists) or rows <= 0:
            result.append(())
            continue
        q_sel = contract.selection_lists[q_idx]
        grp_aivs = q_to_aivs.get(q_idx, ())
        tiles: list[tuple[int, int, int, int]] = []
        for t_idx in range(ceil(rows / SFA_TILE_ROWS)):
            t_start = t_idx * SFA_TILE_ROWS
            t_slice = q_sel[t_start : t_start + min(SFA_TILE_ROWS, rows - t_start)]
            copy_b = 0
            hbm_b = 0
            for _, lo, hi in _partition_tile_rows(len(t_slice), grp_aivs, aics_per_group):
                for _, _, plan in _selection_aiv_steps(t_slice[lo:hi], contract, is_quant, row_bytes):
                    copy_b += plan.aligned_copy_bytes()
                    hbm_b += plan.dram_sector_and_service()[0]
            tiles.append((copy_b, hbm_b, 0, 0))
        result.append(tuple(tiles))
    return tuple(result)


def _c8_vectorized_block_table_bytes(
    contract: AttentionContract, ctx: OperatorPlanningContext, counts: tuple[int, ...], is_quant: bool,
) -> int:
    """Return full-row page-table transfer bytes when CANN ``CalcVectorizeFlag`` enables ``IS_VEC_S2PHYADDR``."""
    if not is_quant or contract.kv_layout != "PA_BLK_BSND" or not contract.causal:
        return 0
    block_size = contract.block_size
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return 0
    sparse_count = contract.selected_tokens if contract.selected_tokens is not None else max(counts, default=0)
    if sparse_count <= 0 or sparse_count % SFA_TILE_ROWS != 0:
        return 0
    max_kv = max((kv for _, kv in contract.sequences), default=max(counts, default=0))
    max_blocks = ceil(max_kv / block_size)
    if max_blocks <= 0:
        return 0
    ub_size = int(ctx.geometry.buffers.get("ub") or dict(ctx.datapath.buffers).get("ub", 0))
    ub_budget = ub_size - C8_VEC_UB_RESERVE_BYTES
    if ub_budget <= 0:
        return 0
    blk_ub = _align_up(max_blocks * INDEX_BYTES, C8_BLOCK_TABLE_ALIGN_BYTES)
    idx_ub = _align_up(sparse_count * INDEX_BYTES, SECTOR_BYTES)
    phy_ub = _align_up(sparse_count, SFA_TILE_ROWS) * SCALE_ADDRESS_BYTES
    if blk_ub + idx_ub + phy_ub > ub_budget:
        return 0
    return max_blocks * INDEX_BYTES


def _c8_vec_bt_first_visit_queries(
    contract: AttentionContract,
    counts: tuple[int, ...],
    group_slices: tuple[range, ...],
    aics_per_group: int,
) -> frozenset[int]:
    """Identify queries where each participating AIV in a group first loads a batch's page-table row."""
    if contract.sequences:
        batch_ids = [b for b, (q, _) in enumerate(contract.sequences) for _ in range(q)]
    else:
        batch = min(contract.batch_size, len(counts)) if counts else 1
        base_q, extra_q = divmod(len(counts), max(1, batch))
        batch_ids = [b for b in range(batch) for _ in range(base_q + (b < extra_q))]
    vec_cores = SPLIT_G_AIVS_PER_QUERY if aics_per_group > 1 else NON_SPLIT_G_AIVS_PER_QUERY
    first_visits: set[int] = set()
    for sl in group_slices:
        valid_qs = [q_idx for q_idx in sl if counts[q_idx] > 0]
        base, rem = divmod(len(valid_qs), vec_cores)
        pos = 0
        for v_idx in range(vec_cores):
            take = base + (1 if v_idx < rem else 0)
            seen_batches: set[int] = set()
            for q_idx in valid_qs[pos : pos + take]:
                b_id = batch_ids[q_idx] if q_idx < len(batch_ids) else q_idx
                if b_id not in seen_batches:
                    seen_batches.add(b_id)
                    first_visits.add(q_idx)
            pos += take
    return frozenset(first_visits)


@dataclass(frozen=True, slots=True)
class _Arch35SfaProgram:
    contract: AttentionContract
    ctx: OperatorPlanningContext
    dtype: str
    row_bytes: int
    is_quant: bool
    counts: tuple[int, ...]
    groups: tuple[tuple[tuple[int, tuple[int, ...]], ...], ...]
    group_slices: tuple[range, ...]
    use_gm_ring: bool
    ring_bytes: int
    ring_l2_resident: bool
    has_l2_rate: bool
    vectorized_bt_bytes: int = 0
    vec_bt_first_visits: frozenset[int] = frozenset()
    phases: tuple[AttentionPhase, ...] = ()
    tile_gathers: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()
    _preview_collector: dict[tuple[int, int], list[int]] | None = None

    def _per_aic_tile_work(self, phase_name: str, heads: int, rows: int) -> Work | None:
        c, ctx = self.contract, self.ctx
        if phase_name == "kv_ring_read":
            # C8 Split-G reads contiguous 16-row NZ blocks from GM ring (Align16Func(s2RealSize) * D);
            # BF16 SFA uses CopyToL1Nd2Nz reading exact s2RealSize ND rows from GM.
            ring_read_rows = _align_up(rows, FRACTAL_ALIGN) if self.is_quant else rows
            kv_b = ring_read_rows * c.head_dim * BF16_BYTES
            return read_work(
                ctx, kv_b,
                hbm=0 if self.ring_l2_resident else kv_b,
                l2=(kv_b if (self.ring_l2_resident and self.has_l2_rate) else 0),
            )
        pairs = heads * rows
        if phase_name == "qk":
            m_al = _align_up(heads, FRACTAL_ALIGN)
            n_al = _align_up(rows, FRACTAL_ALIGN)
            k_al = _align_up(c.head_dim, FRACTAL_ALIGN)
            return Work(
                {self.dtype: FLOPS_PER_FMA * m_al * n_al * k_al},
                {"mte1": (m_al + n_al) * k_al * BF16_BYTES},
            )
        if phase_name == "score_fixpipe":
            return Work(resources={
                "fixpipe": _align_up(heads, FIXPIPE_M_ALIGN) * _align_up(rows, FIXPIPE_N_ALIGN) * FP32_BYTES,
            })
        if phase_name == "online_softmax":
            sm_ops = SOFTMAX_OPS + (SFA_SSMAX_EXTRA_OPS if c.ssmax else 0)
            return Work(resources={"vector": pairs * sm_ops})
        if phase_name == "probability_transfer":
            return Work(resources={"mte3": heads * SFA_TILE_ROWS * BF16_BYTES})
        if phase_name == "pv":
            m_al = _align_up(heads, FRACTAL_ALIGN)
            k_al = _align_up(rows, FRACTAL_ALIGN)
            n_al = _align_up(c.v_head_dim, FRACTAL_ALIGN)
            return Work(
                {self.dtype: FLOPS_PER_FMA * m_al * n_al * k_al},
                {"mte1": (m_al * k_al + k_al * n_al) * BF16_BYTES},
            )
        if phase_name == "pv_fixpipe":
            return Work(resources={
                "fixpipe": _align_up(heads, FIXPIPE_M_ALIGN) * _align_up(c.v_head_dim, FIXPIPE_N_ALIGN) * FP32_BYTES,
            })
        if phase_name == "online_output":
            return Work(resources={"vector": heads * c.v_head_dim * SFA_OUTPUT_MERGE_OPS_PER_ELEM})
        return None

    def _per_aic_query_work(self, phase_name: str, heads: int) -> Work:
        c = self.contract
        if phase_name == "query_load":
            return read_work(self.ctx, heads * c.head_dim * BF16_BYTES)
        if phase_name == "normalize":
            return Work(resources={"vector": heads * c.v_head_dim})
        lse_slots = SFA_LSE_STAT_TENSORS if c.abi in {SFA_V1_ABI, C8_ABI} else 1
        out_bytes = heads * (
            c.v_head_dim * BF16_BYTES + (lse_slots * FP32_BYTES if c.return_lse else 0)
        )
        return Work(resources={"mte3": out_bytes, "hbm": out_bytes})

    def _sum_per_aic_tile(self, phase_name: str, head_slices: tuple[int, ...], rows: int) -> Work:
        total = Work()
        for heads in head_slices:
            part = self._per_aic_tile_work(phase_name, heads, rows)
            assert part is not None
            total = total + part
        return total

    def _append_pa_phases(
        self, out: list[AttentionPhase], rows: int,
        *, bt_bytes_override: int | None, selection_slice: tuple[int, ...],
    ) -> tuple[str, ...]:
        c, ctx = self.contract, self.ctx
        if c.kv_layout not in PAGED_LAYOUTS:
            return ("sparse_index",)
        if self.vectorized_bt_bytes > 0:
            bt_bytes = bt_bytes_override if bt_bytes_override is not None else 0
            out.append(AttentionPhase("block_table", read_work(ctx, bt_bytes, aiv=True), ()))
            phy_bytes = rows * SCALE_ADDRESS_BYTES
            out.append(AttentionPhase(
                "phy_addr_write",
                Work(resources={"mte3": phy_bytes, "hbm": phy_bytes}),
                ("sparse_index", "block_table"),
            ))
            out.append(AttentionPhase("phy_addr_read", read_work(ctx, phy_bytes, aiv=True), ("phy_addr_write",)))
            return ("phy_addr_read",)
        if selection_slice:
            addr_stride = self.row_bytes if self.is_quant else (c.v_head_dim * BF16_BYTES)
            page_bytes = max(1, c.block_size * addr_stride)
            pages = len({addr // page_bytes for addr in selection_slice})
        else:
            pages = ceil(rows / max(1, c.block_size))
        out.append(AttentionPhase("block_table", read_work(ctx, pages * INDEX_BYTES, aiv=True), ()))
        return ("sparse_index", "block_table")

    def _tile_phases(
        self, rows: int, head_slices: tuple[int, ...],
        *, gather_override: tuple[int, int, int, int] | None = None,
        bt_bytes_override: int | None = None,
        selection_slice: tuple[int, ...] = (),
    ) -> tuple[AttentionPhase, ...]:
        c, ctx = self.contract, self.ctx
        gather_l2 = 0
        if gather_override is not None:
            gather_copy, gather_hbm, gather_l2, _ = gather_override
        elif self.is_quant:
            stride = ceil(self.row_bytes / SECTOR_BYTES) * SECTOR_BYTES
            run = AddressRunPlan((AddressRun(0, rows, self.row_bytes, stride),))
            gather_copy = run.aligned_copy_bytes()
            gather_hbm = run.dram_sector_and_service()[0]
        else:
            latent_b = c.v_head_dim * BF16_BYTES
            rope_b = max(0, c.head_dim - c.v_head_dim) * BF16_BYTES
            lat_run = AddressRunPlan((AddressRun(0, rows, latent_b, ceil(latent_b / SECTOR_BYTES) * SECTOR_BYTES),))
            gather_copy = lat_run.aligned_copy_bytes()
            gather_hbm = lat_run.dram_sector_and_service()[0]
            if rope_b > 0:
                rope_run = AddressRunPlan((AddressRun(0, rows, rope_b, ceil(rope_b / SECTOR_BYTES) * SECTOR_BYTES),))
                gather_copy += rope_run.aligned_copy_bytes()
                gather_hbm += rope_run.dram_sector_and_service()[0]
        kv_bf16_bytes = rows * c.head_dim * BF16_BYTES
        out: list[AttentionPhase] = [
            AttentionPhase("sparse_index", read_work(ctx, rows * INDEX_BYTES, aiv=True), ()),
        ]
        gather_deps = self._append_pa_phases(
            out, rows, bt_bytes_override=bt_bytes_override, selection_slice=selection_slice,
        )
        gather_work = read_work(
            ctx, gather_copy, aiv=True, hbm=gather_hbm,
            l2=gather_l2 if self.has_l2_rate else 0,
        )
        out.append(AttentionPhase("packed_kv_gather", gather_work, gather_deps))
        prev = "packed_kv_gather"
        if self.is_quant:
            out.append(AttentionPhase(
                "kv_dequant",
                Work(resources={"vector": rows * c.v_head_dim * C8_DEQUANT_OPS_PER_ELEM}),
                (prev,),
            ))
            prev = "kv_dequant"
        if self.use_gm_ring:
            out.append(AttentionPhase(
                "kv_publish",
                workspace_write_work(ctx, "mte3", kv_bf16_bytes, l2_resident=self.ring_l2_resident),
                (prev,),
            ))
            out.append(AttentionPhase(
                "kv_ring_read",
                self._sum_per_aic_tile("kv_ring_read", head_slices, rows),
                ("kv_publish",),
            ))
            qk_dep = "kv_ring_read"
        else:
            # Non-split-G C8 (1 AIC + local 2 AIVs): push KV directly UB->L1 without GM workspace.
            out.append(AttentionPhase("kv_to_l1", Work(resources={"mte3": kv_bf16_bytes}), (prev,)))
            qk_dep = "kv_to_l1"
        out.append(AttentionPhase("qk", self._sum_per_aic_tile("qk", head_slices, rows), (qk_dep,)))
        out.append(AttentionPhase("score_fixpipe", self._sum_per_aic_tile("score_fixpipe", head_slices, rows), ("qk",)))
        out.append(AttentionPhase(
            "online_softmax", self._sum_per_aic_tile("online_softmax", head_slices, rows), ("score_fixpipe",),
        ))
        out.append(AttentionPhase(
            "probability_transfer", self._sum_per_aic_tile("probability_transfer", head_slices, rows), ("online_softmax",),
        ))
        out.append(AttentionPhase("pv", self._sum_per_aic_tile("pv", head_slices, rows), ("probability_transfer",)))
        out.append(AttentionPhase("pv_fixpipe", self._sum_per_aic_tile("pv_fixpipe", head_slices, rows), ("pv",)))
        out.append(AttentionPhase("online_output", self._sum_per_aic_tile("online_output", head_slices, rows), ("pv_fixpipe",)))
        return tuple(out)

    def _query_head_load(self) -> AttentionPhase:
        return AttentionPhase("query_load", self._per_aic_query_work("query_load", self.contract.num_heads), ())

    def _query_finish(self) -> tuple[AttentionPhase, ...]:
        return (
            AttentionPhase("normalize", self._per_aic_query_work("normalize", self.contract.num_heads), ()),
            AttentionPhase("output", self._per_aic_query_work("output", self.contract.num_heads), ("normalize",)),
        )

    def _lower_per_head_stage(
        self, context, ph: AttentionPhase, grp: tuple[tuple[int, tuple[int, ...]], ...],
        half_places: tuple[dict[str, tuple[int, ...]], ...],
        aiv_places: tuple[tuple[dict[str, tuple[int, ...]], ...], ...],
        head_slices: tuple[int, ...], *, rows: int | None = None,
    ) -> Stage:
        branches = []
        split_aiv = ph.name in AIV_PER_HEAD_PHASES
        for h_idx, (h_heads, h_place, sub_places) in enumerate(zip(head_slices, half_places, aiv_places, strict=True)):
            if split_aiv and len(grp[h_idx][1]) >= NON_SPLIT_G_AIVS_PER_QUERY:
                sub_slices = _subblock_head_slices(h_heads, len(grp[h_idx][1]))
                for s_idx, s_heads in enumerate(sub_slices):
                    s_work = (
                        self._per_aic_tile_work(ph.name, s_heads, rows)
                        if rows is not None
                        else self._per_aic_query_work(ph.name, s_heads)
                    )
                    assert s_work is not None
                    st_name = f"{ph.name}_{h_idx}_{s_idx}"
                    branches.append(KernelProgram((_stage(context, replace(ph, name=st_name, work=s_work, after=()), sub_places[s_idx]),)))
            else:
                h_work = (
                    self._per_aic_tile_work(ph.name, h_heads, rows)
                    if rows is not None
                    else self._per_aic_query_work(ph.name, h_heads)
                )
                assert h_work is not None
                branches.append(KernelProgram((_stage(context, replace(ph, name=f"{ph.name}_{h_idx}", work=h_work, after=()), h_place),)))
        if len(branches) == 1:
            return Stage(ph.name, after=ph.after, program=branches[0])
        return Stage(ph.name, after=ph.after, program=ParallelProgram(tuple(branches)))

    def __call__(self, context):
        group_places = []
        group_half_places = []
        group_aiv_places = []
        for grp in self.groups:
            aics = tuple(p[0] for p in grp)
            aivs = tuple(v for p in grp for v in p[1])
            group_places.append(_place_owners(context, aics, aivs))
            group_half_places.append(tuple(_place_owners(context, (p[0],), p[1]) for p in grp))
            group_aiv_places.append(tuple(
                tuple(_place_owners(context, (p[0],), (v,)) for v in p[1])
                for p in grp
            ))

        aics_per_group = len(self.groups[0])
        head_slices = _split_g_head_slices(self.contract.num_heads, aics_per_group)
        q_load_phase = self._query_head_load()
        fin_phases = self._query_finish()

        def build_query(q_idx: int, grp_idx: int):
            grp = self.groups[grp_idx]
            place = group_places[grp_idx]
            half_places = group_half_places[grp_idx]
            aiv_places = group_aiv_places[grp_idx]
            rows = self.counts[q_idx]
            n_tiles = max(1, ceil(rows / SFA_TILE_ROWS)) if rows > 0 else 0
            q_st = (
                self._lower_per_head_stage(context, q_load_phase, grp, half_places, aiv_places, head_slices)
                if aics_per_group > 1
                else _stage(context, q_load_phase, place)
            )
            if n_tiles == 0:
                f_st = tuple(
                    self._lower_per_head_stage(
                        context, replace(p, after=(q_st.name,) if i == 0 else p.after),
                        grp, half_places, aiv_places, head_slices,
                    )
                    for i, p in enumerate(fin_phases)
                )
                return KernelProgram((q_st, *f_st))

            q_gathers = self.tile_gathers[q_idx] if q_idx < len(self.tile_gathers) else ()
            raw_selection = (
                self.contract.selection_lists[q_idx]
                if (self.contract.selection_lists and q_idx < len(self.contract.selection_lists))
                else ()
            )
            q_selection = (
                raw_selection
                if (raw_selection and self.contract.kv_tensor_id
                    and self.has_l2_rate and "hbm" in context.resource_names)
                else ()
            )
            grp_aivs = tuple(v for p in grp for v in p[1])

            def tile_fn(
                t_idx: int, q_id=q_idx, total_rows=rows, g_spec=grp, pl=place,
                hpl=half_places, apl=aiv_places, qg=q_gathers, qsel=q_selection,
                rsel=raw_selection, gaivs=grp_aivs,
            ):
                t_start = t_idx * SFA_TILE_ROWS
                t_rows = min(SFA_TILE_ROWS, total_rows - t_start)
                ov = qg[t_idx] if t_idx < len(qg) else None
                t_sel = rsel[t_start : t_start + t_rows] if rsel else ()
                bt_ov = (
                    (self.vectorized_bt_bytes if (t_idx == 0 and q_id in self.vec_bt_first_visits) else 0)
                    if self.vectorized_bt_bytes
                    else None
                )
                tile_acc = (
                    self._preview_collector.get((q_id, t_idx))
                    if self._preview_collector is not None else None
                )
                stages = []
                for ph in self._tile_phases(
                    t_rows, head_slices, gather_override=ov, bt_bytes_override=bt_ov, selection_slice=t_sel,
                ):
                    if ph.name == "packed_kv_gather" and qsel:
                        st = Stage(
                            ph.name, after=ph.after,
                            program=_lower_selection_gather(
                                context, qsel[t_start : t_start + t_rows], gaivs,
                                aics_per_group, self.contract, self.is_quant, self.row_bytes, self.ctx,
                                collector=tile_acc,
                            ),
                        )
                    elif (aics_per_group > 1 or ph.name in AIV_PER_HEAD_PHASES) and (
                        self._per_aic_tile_work(ph.name, head_slices[0], t_rows) is not None
                    ):
                        st = self._lower_per_head_stage(context, ph, g_spec, hpl, apl, head_slices, rows=t_rows)
                    else:
                        st = _stage(context, ph, pl)
                    if ph.name in {"kv_ring_read", "kv_to_l1"}:
                        st = replace(st, previous=("pv",))
                    stages.append(st)
                return tuple(stages)

            loop = Stage("s2_ring", after=(q_st.name,),
                         program=PipelineProgram(n_tiles, min(n_tiles, SFA_RING_SLOTS), tile_fn))
            f0 = self._lower_per_head_stage(
                context, replace(fin_phases[0], after=(loop.name,)), grp, half_places, aiv_places, head_slices,
            )
            f1 = self._lower_per_head_stage(
                context, replace(fin_phases[1], after=(f0.name,)), grp, half_places, aiv_places, head_slices,
            )
            return KernelProgram((q_st, loop, f0, f1))

        body = ParallelProgram(tuple(
            PipelineProgram(
                len(indices), 1,
                lambda i, g=g_idx, indices=indices: (Stage("query", program=build_query(indices[i], g)),),
            )
            for g_idx, indices in enumerate(self.group_slices) if indices
        ))
        if (
            self.ring_l2_resident and self.ring_bytes > 0 and self.has_l2_rate
            and "hbm" in context.resource_names and self.contract.selection_lists and self.contract.kv_tensor_id
        ):
            die = int(self.ctx.geometry.rank or 0)
            return _SfaRingReservedProgram(
                body=body, die=die, ring_bytes=self.ring_bytes,
                capacity=_l2_capacity_bytes(self.ctx), hbm=context.resource_names["hbm"],
                ctx=self.ctx, alloc_id=f"sfa_ring:{context.op.op_id}",
            )
        return body


def _preview_selection_gathers_engine(
    op: OpSpec,
    prog: _Arch35SfaProgram,
    ctx: OperatorPlanningContext,
    view: AdmissionMemoryView,
) -> tuple[tuple[tuple[int, int, int, int], ...], ...]:
    """Dry-run lowered SFA selection gathers on a scratch Engine matching completion-time L2."""
    collector: dict[tuple[int, int], list[int]] = {
        (q_idx, t_idx): [0, 0, 0, 0]
        for q_idx, rows in enumerate(prog.counts)
        for t_idx in range(ceil(rows / SFA_TILE_ROWS) if rows > 0 else 0)
    }
    rates = {**{role: rate / 1000.0 for role, rate in ctx.geometry.rates.items()}, "cube": 1.0}
    units = {
        **ctx.geometry.resource_units,
        "cube": ctx.geometry.resource_units.get("cube", ctx.datapath.aic_count),
    }
    rc = RecipeContext(
        op, {}, {role: role for role in rates}, rates, ctx.geometry.compute_units,
        resource_units=units, compute_peaks=ctx.geometry.compute_peaks,
    )
    engine = Engine(
        [Resource(role, rate, units=units.get(role, 1)) for role, rate in rates.items()],
        trace_limit=0,
    )
    engine.attach_admission_memory_view(view)
    dry = replace(prog, _preview_collector=collector)(rc)
    done = dry.start(engine, "sfa_preview", engine.submit("ready"))
    engine.advance(until=(done,))
    return tuple(
        tuple(
            (a[0], a[1], a[2], a[3])
            for t_idx in range(ceil(rows / SFA_TILE_ROWS) if rows > 0 else 0)
            for a in (collector[(q_idx, t_idx)],)
        )
        for q_idx, rows in enumerate(prog.counts)
    )


def build_arch35_split_g_sfa(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext, sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build split-G 2-AIC/4-AIV 3-slot pipelined SFA plan (BF16 or C8) on npu_arch_3510."""
    if contract.num_kv_heads != 1:
        raise unmodeled(
            "attention_sfa_kv_heads_unsupported",
            f"{contract.abi} requires num_kv_heads == 1, got {contract.num_kv_heads}",
        )
    if contract.abi == SFA_V1_ABI and contract.quant_mode != QUANT_MODE_BF16:
        raise unmodeled(
            "attention_sfa_v1_quant_mode_unsupported",
            f"{SFA_V1_ABI} requires non-quantized quant_mode={QUANT_MODE_BF16}, got {contract.quant_mode}",
        )
    if contract.quant_mode == QUANT_MODE_TQ4:
        raise unmodeled("attention_arch35_tq4_unsupported", "TQ4 packed latent cache is supported on Arch22 only, not Arch35")
    if contract.abi in {C8_ABI, C8_V1_ABI} and contract.quant_mode not in C8_MODES:
        raise unmodeled(
            "attention_c8_quant_mode_unsupported",
            f"{contract.abi} requires a C8 quant_mode, got {contract.quant_mode}",
        )
    is_quant = contract.quant_mode in C8_MODES
    row_bytes = _packed_row_bytes(contract)
    counts = tuple(contract.row_counts())
    desired_aics = SPLIT_G_AICS_PER_QUERY if contract.num_heads > SFA_SPLIT_G_HEADS else 1
    raw_owners = owner_pairs(ctx, max(desired_aics, len(counts) * desired_aics))
    groups, group_slices = _partition_query_groups(len(counts), raw_owners, desired_aics)
    aics_per_query = desired_aics
    head_slices = _split_g_head_slices(contract.num_heads, aics_per_query)
    use_gm_ring = (not is_quant) or (aics_per_query > 1)
    die = int(ctx.geometry.rank or 0)
    ring_rows = 0
    for sl in group_slices:
        raw_rows = max((min(counts[q_idx], SFA_RING_SLOTS * SFA_TILE_ROWS) for q_idx in sl), default=0)
        ring_rows += _align_up(raw_rows, FRACTAL_ALIGN) if (is_quant and raw_rows > 0) else raw_rows
    ring_bytes = ring_rows * contract.head_dim * BF16_BYTES
    l2_cap = _l2_capacity_bytes(ctx)
    eff_l2 = _effective_l2_capacity_bytes(ctx, die)
    ring_l2_resident = use_gm_ring and eff_l2 > 0 and ring_bytes <= eff_l2
    has_l2 = "l2" in ctx.geometry.rates
    vectorized_bt_bytes = _c8_vectorized_block_table_bytes(contract, ctx, counts, is_quant)
    vec_bt_first_visits = (
        _c8_vec_bt_first_visit_queries(contract, counts, group_slices, aics_per_query)
        if vectorized_bt_bytes
        else frozenset()
    )
    prog = _Arch35SfaProgram(
        contract, ctx, op.dtype or "bf16", row_bytes, is_quant, counts, groups, group_slices,
        use_gm_ring, ring_bytes, ring_l2_resident, has_l2, vectorized_bt_bytes, vec_bt_first_visits,
    )
    bucket: dict[str, AttentionPhase] = {}

    def _acc(phase: AttentionPhase, repeats: int = 1) -> None:
        w = phase.work.scaled(repeats) if repeats != 1 else phase.work
        old = bucket.get(phase.name)
        bucket[phase.name] = replace(phase, work=old.work + w if old else w)

    q_load = prog._query_head_load()
    fin = prog._query_finish()
    # Planning must never mutate the live AdmissionMemoryView; dry-run on a private
    # scratch snapshot while Engine delivery settles fills at transfer completion.
    sel_l2: AdmissionMemoryView | None = None
    if contract.selection_lists and contract.kv_tensor_id and has_l2 and "hbm" in ctx.geometry.rates:
        if ctx.admission_memory_view is not None:
            sel_l2 = copy.deepcopy(ctx.admission_memory_view)
        elif l2_cap > 0:
            sel_l2 = AdmissionMemoryView({die: l2_cap}, arch_id=str(ctx.compute_arch.arch_id))
    tile_gathers: tuple[tuple[tuple[int, int, int, int], ...], ...] = ()
    if sel_l2 is not None and contract.selection_lists:
        tile_gathers = _preview_selection_gathers_engine(op, prog, ctx, sel_l2)
    elif contract.selection_lists:
        tile_gathers = _selection_tile_cold_gathers(
            contract, counts, groups, group_slices, aics_per_query, is_quant, row_bytes,
        )
    per_tile_path = bool(tile_gathers or vectorized_bt_bytes or contract.selection_lists)
    for q_idx, rows in enumerate(counts):
        _acc(q_load)
        if per_tile_path:
            n_tiles = ceil(rows / SFA_TILE_ROWS) if rows > 0 else 0
            q_sel = contract.selection_lists[q_idx] if q_idx < len(contract.selection_lists) else ()
            q_gathers = tile_gathers[q_idx] if q_idx < len(tile_gathers) else ()
            for t_idx in range(n_tiles):
                t_start = t_idx * SFA_TILE_ROWS
                t_rows = min(SFA_TILE_ROWS, rows - t_start)
                ov = q_gathers[t_idx] if t_idx < len(q_gathers) else None
                t_sel = q_sel[t_start : t_start + t_rows] if q_sel else ()
                bt_ov = (
                    (vectorized_bt_bytes if (t_idx == 0 and q_idx in vec_bt_first_visits) else 0)
                    if vectorized_bt_bytes
                    else None
                )
                for ph in prog._tile_phases(
                    t_rows, head_slices, gather_override=ov, bt_bytes_override=bt_ov, selection_slice=t_sel,
                ):
                    _acc(ph, 1)
        else:
            full, rem = divmod(rows, SFA_TILE_ROWS)
            if full:
                for ph in prog._tile_phases(SFA_TILE_ROWS, head_slices):
                    _acc(ph, full)
            if rem:
                for ph in prog._tile_phases(rem, head_slices):
                    _acc(ph, 1)
        for ph in fin:
            _acc(ph)

    phases = tuple(bucket.values())
    total_rows = sum(counts)
    cold_hbm = bucket["packed_kv_gather"].work.resources.get("hbm", 0) if "packed_kv_gather" in bucket else 0
    l2_shared = int(bucket["kv_ring_read"].work.resources.get("mte2_aic", 0)) if (
        ring_l2_resident and "kv_ring_read" in bucket
    ) else 0
    leaf_sources = (C8_KERNEL_SOURCE, C8_TILING_SOURCE) if is_quant else (SFA_KERNEL_SOURCE, SFA_TILING_SOURCE)
    plan_sources = leaf_sources + tuple(src for src in sources if src not in leaf_sources)
    metadata = {
        "source_program": leaf_sources[0].to_dict(),
        "kv_logical_bytes": total_rows * row_bytes,
        "kv_sector_bytes": cold_hbm,
        "kv_dram_sector_bytes": cold_hbm,
        "l2_shared_kv_bytes": l2_shared,
        "kv_row_bytes": row_bytes,
        "kv_gather_rows": total_rows,
        "split_g_aics_per_query": aics_per_query,
        "ring_l2_resident": ring_l2_resident,
    }
    return make_plan(
        op, contract, ctx, phases, plan_sources, metadata=metadata,
        program=replace(prog, phases=phases, tile_gathers=tile_gathers),
    )
