"""Native CANN GDN chunk (Arch35 VF) and recurrent (Arch22/Arch35) helpers.

Upstream-ABI: aclnnRecurrentGatedDeltaRule

[Role] Provide exact Arch35 persistent GDN WY vector-VF issue & transfer
  accounting (from ``chunk_gated_delta_rule_{cube,vector}_compute.h`` and
  ``stage{1,2}_vf.h``) plus native ``aclnnRecurrentGatedDeltaRule`` lowering
  (Arch22 single-buffered ``BUFFER_NUM=1`` vs Arch35 ``vStep=64`` synchronous
  ``CopyInState`` + ``SINGLE_BUFFER`` ``stateOutQueue_`` overlapping ``Q`` dot).
[Invariants] Conserves physical FLOPs, ops, and bytes; uses periodic pipeline
  compaction for O(1) Arch35 WY lowering up to 409.6K tokens; zero hardcoded
  timing constants (sync/barrier latencies come from calibrated P12/P13 L0 cells).
[Timeouts & Locks] O(owners) planning; zero locks or external I/O.
[Side-Effect Policy] Pure immutable OperatorPlans and virtual Engine programs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError, align_up
from mindie_sim.schema.datapath import SourceRef

from ...mechanism import OperatorPlan, Work
from ...program import (
    KernelProgram,
    ParallelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from ..contracts import OperatorPlanningContext
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sharedkv.l0_sync import p12_flag_ms, p12_pipe_v_ms, p13_paired_ms
from .common import (
    FP32_BYTES,
    GDN_CHUNK64_TOKENS,
    LinearAttentionContract,
    Phase,
    _ARCH22_ID,
    analyze_state_residency,
    bind_phase,
    ceil_div,
    element_bytes,
    make_plan,
    missing,
    owner_units,
    transfer,
    verify_native_gdn_abi,
)
from .gdn_recurrent import (
    QK_CAST_SCALE_OPS_PER_LANE,
    VECTOR_LANE_ALIGNMENT,
    head_ownership,
    head_runs,
    kernel_work_terms,
    state_value_step,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
OPS_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
RECURRENT_GDN_ABI = "aclnnRecurrentGatedDeltaRule"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision=OPS_REV,
        relative_path="attention/recurrent_gated_delta_rule/op_host/recurrent_gated_delta_rule_tiling.cpp",
        source_sha256="785101bff3cc4454ffdf31661b940b69254850c5eb5fab7232bb118085cb9ed",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision=OPS_REV,
        relative_path="attention/recurrent_gated_delta_rule/op_kernel/arch35/recurrent_gated_delta_rule_arch35.h",
        source_sha256="5590f9a236f83660df7b216af2d535c4cb07a8ccd96fa07ebc6be3f78c9a03c2",
    ),
)
RECURRENT_GDN_SOURCES = SOURCES

# Arch35 VectorCompute (chunk_gated_delta_rule_vector_compute.h & stage{1,2}_vf.h)
# Exact 64-FP32-lane SIMD issue counts per AIV per full 64-token chunk (2 AIVs/head):
#   [2a] CumSumExpExactVF (6 Hillis-Steele rounds + Gather/Cast) = 45 issues
#   [2b] PrepareStage1FullDimSliceVF (32 rows * (5 scalar/exp + 2*25 vec/cast/pack)) = 1760 issues
#   [3a] BuildLowerAndMaskedQktExactSliceVF (32 rows * 19 ops incl. Exp) = 608 issues
#   [3b] InverseAIVPanel8DirectBrcVFImpl (496 Brc+MulAddDst + 32*(Dup+Load+Sub+Store+Cast+Pack+Store)) = 1216 issues
#   [6c] Cast R32->R16 (4096/64) = 64 issues
#   [9a/9b] Cast delta + UpdateFp32StateByExpVF + RowMajorFp32ToCompactZnBf16VF (128*9) = 1152 issues
ARCH35_VF_LANES = 64
ARCH35_VF_GATE_ISSUES_PER_AIV = 1805
ARCH35_VF_INV_ISSUES_PER_AIV = 1824
ARCH35_VF_CAST_ISSUES_PER_AIV = 64
ARCH35_VF_STATE_ISSUES_PER_AIV = 1152
ARCH35_GDN_AIVS_PER_HEAD = 2
# Source-counted cross-core handshakes (StateReady + ResidualReady) and PIPE_V barriers per chunk
ARCH35_GDN_CROSSCORE_PAIRS_PER_CHUNK = 2
ARCH35_GDN_PIPE_BARRIERS_PER_CHUNK = 12
ARCH35_GDN_COMPACT_THRESHOLD = 32
ARCH35_GDN_COMPACT_WINDOW = 2
ARCH35_GDN_AIC_MTE2_QK_PANELS = 2
ARCH35_GDN_AIV_MTE2_QKV_PANELS = 3

# recurrent_gated_delta_rule_arch{22,35}.h exact per-state-element op counts:
#   Both Arch22 (Muls + MatVecMul + MulAddDst) and Arch35 (VecMulMatVF*2 + OuterAddVF) perform
#   decay + k-mul (2*ak), P09_reduce(k) (ak-1), delta + OuterAdd (2*ak+2), state copy/cast (ak),
#   q-mul + attn_cast (ak+1), and P09_reduce(q) (ak-1) per dv row, plus gamma/Q/K/V casts.
RGDR_UPD_CORE_PASSES = 2
RGDR_GAMMA_KEY_OPS_PER_LANE = 1 + 2 + QK_CAST_SCALE_OPS_PER_LANE
RGDR_PIPE_V_PER_VSTEP_TOKEN = 10  # recurrent_gated_delta_rule_arch{22,35}.h: 10 PipeBarrier<PIPE_V> per tile-token
RGDR_QUEUE_SYNCS_PER_HEAD = 10  # CopyInGamaK/QK/V/State + CopyOutState/Attn EnQue/DeQue handshakes
RGDR_MS_PER_S = 1000.0
GDN_REF_HEAD_DIM = 128
GDN_REF_DIM_SUM = float(GDN_REF_HEAD_DIM * 2)
# recurrent_gated_delta_rule_tiling.cpp:630-661 Arch22 CalUbSize constants
ARCH22_MAX_MTP = 8
ARCH22_RESERVED_UB_BYTES = 128
ARCH22_FP32_BLOCK_ELEMS = 8
ARCH22_QKV_QUEUE_K_BYTES = 4
ARCH22_QKV_QUEUE_V_BYTES = 2
ARCH22_GAMA_QUEUE_BYTES = 4
ARCH22_BETA_QUEUE_BYTES = 2
ARCH22_QKV_UB_K_BYTES = 8
ARCH22_QKV_UB_V_BYTES = 4
ARCH22_QKV_UB_BETA_BYTES = 4
ARCH22_STATE_UB_K_COEFF = 8
ARCH22_STATE_UB_FIXED_COEFF = 12


def _rgdr_reduce_ratio(rc: RecipeContext) -> float:
    """Ratio of calibrated P07 vector ALU rate to P09 vector reduce rate."""
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        alu_cell = table.cell("P07_vec_alu")
        red_cell = table.cell("P09_vec_reduce_cast")
        alu = alu_cell.band("vec_alu_ops_per_s_per_aiv").p50
        red = red_cell.band("vec_reduce_ops_per_s_per_aiv").p50
        if alu is None or red is None or float(red) <= 0.0:
            raise ModelingError("rgdr_missing_calibrated_reduce_primitive_band")
        return max(1.0, float(alu) / float(red))
    vec_key = rc.resource_names.get("vector", "vector")
    vec_rate = float(rc.rates.get(vec_key, rc.rates.get("vector", 0.0)) or 0.0) * RGDR_MS_PER_S
    red_rate = rc.service_rate("vector_reduce")
    if vec_rate > 0.0 and red_rate and red_rate > 0.0:
        return max(1.0, vec_rate / float(red_rate))
    raise ModelingError("rgdr_missing_calibrated_vector_reduce_rate")


def _rgdr_cast_ratio(rc: RecipeContext) -> float:
    """Ratio of calibrated P07 vector ALU rate to P09 FP32-equivalent cast element rate."""
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        alu = table.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50
        cast_bps = table.cell("P09_vec_reduce_cast").band("cast_bytes_per_s_per_aiv").p50
        if alu is None or cast_bps is None or float(cast_bps) <= 0.0:
            raise ModelingError("rgdr_missing_calibrated_cast_primitive_band")
        return max(1.0, float(alu) / (float(cast_bps) / float(FP32_BYTES)))
    vec_key = rc.resource_names.get("vector", "vector")
    vec_rate = float(rc.rates.get(vec_key, rc.rates.get("vector", 0.0)) or 0.0) * RGDR_MS_PER_S
    cast_rate = rc.service_rate("cast") or rc.service_rate("vector_cast")
    if vec_rate > 0.0 and cast_rate and float(cast_rate) > 0.0:
        return max(1.0, vec_rate / (float(cast_rate) / float(FP32_BYTES)))
    raise ModelingError("rgdr_missing_calibrated_cast_rate")


def _arch22_head_ownership(
    lengths: tuple[int, ...], heads: int, blocks: int,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Arch22 IsCurrentBlock (arch22.h:148,384): avgload = ceil(total_tokens * Hv / blocks)."""
    target = ceil_div(sum(lengths) * heads, blocks)
    owners: list[list[tuple[int, int]]] = [[] for _ in range(blocks)]
    owner, load = 0, 0
    for seq, q in enumerate(lengths):
        if q <= 0:
            continue
        remaining = heads
        while remaining:
            count = min(remaining, ceil_div(target - load, q))
            owners[owner].append((seq, count))
            remaining -= count
            load += count * q
            if load >= target:
                owner += 1
                load = 0
    return tuple(tuple(segments) for segments in owners if segments)


def _arch22_v_step(
    ctx: OperatorPlanningContext, dk: int, dv: int, nv: int, state_elem: int,
) -> int:
    """Host-selected Arch22 vStep (recurrent_gated_delta_rule_tiling.cpp:630-661)."""
    ub_raw = dict(getattr(ctx.geometry, "buffers", ())).get("ub")
    if ub_raw is None or int(ub_raw) <= 0:
        raise missing("linear_attention_arch22_ub_missing", "Arch22 recurrent GDN requires declared UB capacity")
    ub_size = int(ub_raw)
    a_nv = align_up(nv, VECTOR_LANE_ALIGNMENT)
    a_dv = align_up(dv, VECTOR_LANE_ALIGNMENT)
    a_dk = align_up(dk, VECTOR_LANE_ALIGNMENT)
    used = (
        ARCH22_MAX_MTP * (ARCH22_QKV_QUEUE_K_BYTES * a_dk + ARCH22_QKV_QUEUE_V_BYTES * a_dv)
        + ARCH22_RESERVED_UB_BYTES
        + ARCH22_MAX_MTP * (ARCH22_GAMA_QUEUE_BYTES + ARCH22_BETA_QUEUE_BYTES) * a_nv
        + ARCH22_MAX_MTP * (ARCH22_QKV_UB_K_BYTES * a_dk + ARCH22_QKV_UB_V_BYTES * a_dv + ARCH22_QKV_UB_BETA_BYTES * a_nv)
    )
    coeff = (2 * state_elem + ARCH22_STATE_UB_K_COEFF) * a_dk + ARCH22_STATE_UB_FIXED_COEFF
    raw_step = ((ub_size - used) // coeff // ARCH22_FP32_BLOCK_ELEMS) * ARCH22_FP32_BLOCK_ELEMS
    if raw_step < ARCH22_FP32_BLOCK_ELEMS:
        raise missing("linear_attention_gdn_state_tile_overflow", "Arch22 recurrent GDN state tile exceeds UB")
    rptime = ceil_div(dv, raw_step)
    return align_up(ceil_div(dv, rptime), ARCH22_FP32_BLOCK_ELEMS)


def _owner_compacted_runs(
    lengths: tuple[int, ...], segments: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int, int], ...]:
    """Compact consecutive equal-q segments while counting distinct participating sequences."""
    runs: list[tuple[int, int, int]] = []
    cur_q: int | None = None
    cur_heads = 0
    cur_seqs = 0
    for seq, head_count in segments:
        q = lengths[seq]
        if q == cur_q:
            cur_heads += head_count
            cur_seqs += 1
        else:
            if cur_heads:
                runs.append((int(cur_q), cur_heads, cur_seqs))
            cur_q, cur_heads, cur_seqs = q, head_count, 1
    if cur_heads:
        runs.append((int(cur_q), cur_heads, cur_seqs))
    return tuple(runs)


def _cap_vec_stage(st: Stage, rc: RecipeContext, ratio: float) -> Stage:
    """Scale a vector stage rate by 1/ratio via Stage.caps while conserving exact physical ops."""
    if ratio <= 0.0 or abs(ratio - 1.0) < 1e-6:
        return st
    vec_key = rc.resource_names.get("vector", "vector")
    if vec_key not in st.work:
        return st
    full_rate = float(rc.rates.get(vec_key, 0.0) or 0.0)
    if full_rate <= 0.0:
        return st
    total_units = max(1, int(rc.resource_units.get("vector", 1) or 1))
    placed = len(st.placement.get(vec_key, ())) or total_units
    cap_rate = full_rate * (placed / float(total_units)) / ratio
    return replace(st, caps={**st.caps, vec_key: cap_rate})


def arch35_gdn_vf_vector_ops(rows: int, dk: int, dv: int) -> tuple[int, int, int, int]:
    """Return exact (gate, mask_inv, cast_r, state_upd) 64-lane VF ops per head-chunk."""
    row_scale = rows / float(GDN_CHUNK64_TOKENS)
    dim_scale = (dk * dv) / float(GDN_REF_HEAD_DIM * GDN_REF_HEAD_DIM)
    scale_2a2b = row_scale * (dk + dv) / GDN_REF_DIM_SUM
    # BuildLowerAndMaskedQktExactSliceVF + InverseAIVPanel8DirectBrcVFImpl execute fixed 64x64 VF panels.
    scale_3a3b = 1.0
    gate_ops = int(round(ARCH35_GDN_AIVS_PER_HEAD * ARCH35_VF_GATE_ISSUES_PER_AIV * ARCH35_VF_LANES * scale_2a2b))
    inv_ops = int(round(ARCH35_GDN_AIVS_PER_HEAD * ARCH35_VF_INV_ISSUES_PER_AIV * ARCH35_VF_LANES * scale_3a3b))
    cast_ops = int(round(ARCH35_GDN_AIVS_PER_HEAD * ARCH35_VF_CAST_ISSUES_PER_AIV * ARCH35_VF_LANES * scale_2a2b))
    state_ops = int(round(ARCH35_GDN_AIVS_PER_HEAD * ARCH35_VF_STATE_ISSUES_PER_AIV * ARCH35_VF_LANES * dim_scale))
    return gate_ops, inv_ops, cast_ops, state_ops


def arch35_gdn_chunk_sync_ms(rc: RecipeContext) -> float:
    """Derive Arch35 per-chunk NotifyCube/WaitForCube + PIPE_V sync latency from P12/P13."""
    return (
        ARCH35_GDN_CROSSCORE_PAIRS_PER_CHUNK * p13_paired_ms(rc)
        + ARCH35_GDN_PIPE_BARRIERS_PER_CHUNK * p12_pipe_v_ms(rc)
    )


def arch35_gdn_extra_sync_stage(name: str, duration_ms: float, after: tuple[str, ...] = ()) -> Stage:
    return Stage(name, after=after, latency=duration_ms, keep_empty=True)


def compact_lane_pipeline(
    rc: RecipeContext, count: int, slots: int, tile_fn,
) -> PipelineProgram | KernelProgram:
    """Compact long uniform chunk lanes so 409.6K (6400 chunks) runs in O(1)."""
    if count > ARCH35_GDN_COMPACT_THRESHOLD:
        return compact_periodic_pipeline(
            rc, count, slots, tile_fn,
            head=ARCH35_GDN_COMPACT_WINDOW, tail=ARCH35_GDN_COMPACT_WINDOW,
            threshold=ARCH35_GDN_COMPACT_THRESHOLD,
        )
    return PipelineProgram(count, slots, tile_fn)


def _build_rgdr_run_stages(
    rc: RecipeContext,
    ctx: OperatorPlanningContext,
    *,
    c: LinearAttentionContract,
    q: int,
    head_count: int,
    seq_count: int,
    aiv_id: int,
    is_arch22: bool,
    elem: int,
    gate_elem: int,
    beta_elem: int,
    slot_bytes: int,
    v_tiles: int,
    init_vec_head: int,
    prek_vec_tok: int,
    redk_vec_tok: int,
    upd_core_vec_tok: int,
    state_copy_vec_tok: int,
    qalu_vec_tok: int,
    redq_vec_tok: int,
    red_ratio: float,
    cast_ratio: float,
    pipe_v_ms: float,
    flag_ms: float,
) -> tuple[Stage, ...]:
    """Emit per-owner ordered MTE2/Vector/MTE3 stages for Arch22 or Arch35 RGDR."""
    owner = (aiv_id,)
    gb_mte2 = seq_count * q * c.hv * (gate_elem + beta_elem) if is_arch22 else 0
    run_mte2 = head_count * (slot_bytes + q * (2 * c.dk + c.dv) * elem) + gb_mte2
    run_scalar = 0 if is_arch22 else head_count * q * (gate_elem + beta_elem * v_tiles)
    run_state_mte3 = head_count * slot_bytes
    run_attn_mte3 = head_count * c.dv * elem
    read_work = transfer(ctx, read=run_mte2) + (Work(resources={"hbm": run_scalar}) if run_scalar else Work())
    init_stage = bind_phase(
        Phase("rgdr_init", Work(resources={"vector": head_count * init_vec_head}), ("rgdr_read",)),
        rc, ctx, (), owner,
    )
    if cast_ratio > 1.0:
        init_stage = _cap_vec_stage(init_stage, rc, cast_ratio)
    stages: list[Stage] = [
        bind_phase(Phase("rgdr_read", read_work), rc, ctx, (), owner),
        init_stage,
    ]
    prev_vec = "rgdr_init"
    prev_mte3: str | None = None
    for t in range(q):
        pk_name, rk_name = f"rgdr_prek_{t}", f"rgdr_redk_{t}"
        up_name, sp_name = f"rgdr_upd_{t}", f"rgdr_stprep_{t}"
        qa_name, rq_name = f"rgdr_qalu_{t}", f"rgdr_redq_{t}"
        sw_name, ow_name = f"rgdr_state_write_{t}", f"rgdr_out_write_{t}"
        stages.append(bind_phase(
            Phase(pk_name, Work(resources={"vector": head_count * prek_vec_tok}), (prev_vec,)),
            rc, ctx, (), owner,
        ))
        stages.append(_cap_vec_stage(
            bind_phase(Phase(rk_name, Work(resources={"vector": head_count * redk_vec_tok}), (pk_name,)),
                       rc, ctx, (), owner),
            rc, red_ratio,
        ))
        stages.append(bind_phase(
            Phase(up_name, Work(resources={"vector": head_count * upd_core_vec_tok}), (rk_name,)),
            rc, ctx, (), owner,
        ))
        if is_arch22:
            # arch22.h:295-308,374-376: query dot + reduce finish before state cast/copy and MTE3 stores.
            stages.append(bind_phase(
                Phase(qa_name, Work(resources={"vector": head_count * qalu_vec_tok}), (up_name,)),
                rc, ctx, (), owner,
            ))
            stages.append(_cap_vec_stage(
                bind_phase(Phase(rq_name, Work(resources={"vector": head_count * redq_vec_tok}), (qa_name,)),
                           rc, ctx, (), owner),
                rc, red_ratio,
            ))
            stprep_stage = bind_phase(
                Phase(sp_name, Work(resources={"vector": head_count * state_copy_vec_tok}), (rq_name,)),
                rc, ctx, (), owner,
            )
            if cast_ratio > 1.0:
                stprep_stage = _cap_vec_stage(stprep_stage, rc, cast_ratio)
            stages.append(stprep_stage)
            ow_after = (rq_name, prev_mte3) if prev_mte3 is not None else (rq_name,)
            stages.append(bind_phase(Phase(ow_name, transfer(ctx, write=run_attn_mte3), ow_after), rc, ctx, (), owner))
            stages.append(bind_phase(Phase(sw_name, transfer(ctx, write=run_state_mte3), (sp_name, ow_name)), rc, ctx, (), owner))
            prev_vec = sp_name
            prev_mte3 = sw_name
        else:
            # arch35.h:438-445: CopyOutState overlaps query dot + reduce; CopyAttentionOut strictly follows reduce.
            stprep_stage = bind_phase(
                Phase(sp_name, Work(resources={"vector": head_count * state_copy_vec_tok}), (up_name,)),
                rc, ctx, (), owner,
            )
            if cast_ratio > 1.0:
                stprep_stage = _cap_vec_stage(stprep_stage, rc, cast_ratio)
            stages.append(stprep_stage)
            sw_after = (sp_name, prev_mte3) if prev_mte3 is not None else (sp_name,)
            stages.append(bind_phase(Phase(sw_name, transfer(ctx, write=run_state_mte3), sw_after), rc, ctx, (), owner))
            stages.append(bind_phase(
                Phase(qa_name, Work(resources={"vector": head_count * qalu_vec_tok}), (sp_name,)),
                rc, ctx, (), owner,
            ))
            stages.append(_cap_vec_stage(
                bind_phase(Phase(rq_name, Work(resources={"vector": head_count * redq_vec_tok}), (qa_name,)),
                           rc, ctx, (), owner),
                rc, red_ratio,
            ))
            stages.append(bind_phase(
                Phase(ow_name, transfer(ctx, write=run_attn_mte3), (rq_name, sw_name)),
                rc, ctx, (), owner,
            ))
            prev_vec = rq_name
            prev_mte3 = ow_name
    barrier_ms = head_count * (
        q * v_tiles * RGDR_PIPE_V_PER_VSTEP_TOKEN * pipe_v_ms
        + RGDR_QUEUE_SYNCS_PER_HEAD * flag_ms
    )
    if barrier_ms > 0.0:
        tail_after = (prev_vec, prev_mte3) if prev_mte3 is not None else (prev_vec,)
        stages.append(Stage("rgdr_sync", after=tail_after, latency=barrier_ms, keep_empty=True))
    return tuple(stages)


def build_native_recurrent_gdn(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower native CANN aclnnRecurrentGatedDeltaRule (BF16 or FP32 state, Arch22/Arch35)."""
    verify_native_gdn_abi(c, ctx, abi_id=RECURRENT_GDN_ABI)
    _, aivs = owner_units(ctx)
    if not aivs:
        raise missing("linear_attention_no_physical_owner", "Recurrent GDN requires AIV owners")

    is_arch22 = ctx.compute_arch.arch_id == _ARCH22_ID
    elem = element_bytes(c.dtype)
    state_elem = element_bytes(c.state_dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    total_tokens = sum(c.lengths)
    total_heads = sum(length > 0 for length in c.lengths) * c.hv
    if total_heads == 0:
        return make_plan(
            ctx, RECURRENT_GDN_ABI, (), sources=RECURRENT_GDN_SOURCES,
            owners=0, program=lambda _rc: KernelProgram(()),
            metadata={"template": "native_recurrent_gdn", "empty_active_sequences": True,
                      "active_aivs": 0, "waves": 0, "state_contraction_ops": 0},
        )
    blocks = min(len(aivs), total_heads)
    ownership = (
        _arch22_head_ownership(c.lengths, c.hv, blocks)
        if is_arch22
        else head_ownership(c.lengths, c.hv, blocks)
    )
    active_aivs = len(ownership)
    owner_heads = tuple(sum(count for _, count in segments) for segments in ownership)
    waves = max(owner_heads)
    ak = align_up(c.dk, VECTOR_LANE_ALIGNMENT)
    av = align_up(c.dv, VECTOR_LANE_ALIGNMENT)
    v_step = (
        _arch22_v_step(ctx, c.dk, c.dv, c.hv, state_elem)
        if is_arch22
        else state_value_step(ak)
    )
    if not v_step:
        raise missing("linear_attention_gdn_state_tile_overflow", "Recurrent GDN state tile exceeds UB")
    v_tiles = ceil_div(c.dv, v_step)

    slot_bytes = c.dk * c.dv * state_elem
    # recurrent_gated_delta_rule_arch{22,35}.h: every value-head owner copies Q, K, V and
    # unconditionally reads/writes in-place state, sharing gdn_recurrent.kernel_work_terms.
    mte2_base, _, mte3_total, vec_ops = kernel_work_terms(
        c.lengths, c.hv, c.dk, c.dv, input_bytes=elem, state_bytes=state_elem,
    )
    if is_arch22:
        # arch22.h:165-170,333-354: one whole-sequence gamma/beta MTE2 copy per participating (owner, seq).
        gb_mte2_total = sum(
            c.lengths[seq] * c.hv * (gate_elem + beta_elem)
            for segments in ownership for seq, _ in segments
        )
        mte2_total = mte2_base + gb_mte2_total
        scalar_total = 0
    else:
        mte2_total = mte2_base
        scalar_total = total_tokens * c.hv * (gate_elem + beta_elem * v_tiles)

    needs_state_cast = state_elem < FP32_BYTES
    init_vec_head = ak * c.dv
    prek_vec_tok = RGDR_GAMMA_KEY_OPS_PER_LANE * ak + av + 2 * ak * c.dv
    redk_vec_tok = (ak - 1) * c.dv
    upd_core_vec_tok = (RGDR_UPD_CORE_PASSES * ak + 2) * c.dv
    state_copy_vec_tok = ak * c.dv
    qalu_vec_tok = (ak + 1) * c.dv
    redq_vec_tok = (ak - 1) * c.dv

    read_work = transfer(ctx, read=mte2_total) + (Work(resources={"hbm": scalar_total}) if scalar_total else Work())
    read_phase = Phase("rgdr_read", read_work)
    vec_phase = Phase("rgdr_compute", Work(resources={"vector": vec_ops}), ("rgdr_read",))
    write_phase = Phase("rgdr_write", transfer(ctx, write=mte3_total), ("rgdr_compute",))
    totals = (read_phase, vec_phase, write_phase)
    report = analyze_state_residency(c, ctx, persistent=True)

    def lower(rc: RecipeContext) -> KernelProgram | ParallelProgram:
        red_ratio = _rgdr_reduce_ratio(rc)
        cast_ratio = _rgdr_cast_ratio(rc) if needs_state_cast else 1.0
        pipe_v_ms = p12_pipe_v_ms(rc)
        flag_ms = p12_flag_ms(rc)

        def _owner_program(aiv_id: int, segments: tuple[tuple[int, int], ...]) -> KernelProgram | PipelineProgram:
            runs = _owner_compacted_runs(c.lengths, segments)
            run_kwargs = dict(
                c=c, aiv_id=aiv_id, is_arch22=is_arch22, elem=elem, gate_elem=gate_elem,
                beta_elem=beta_elem, slot_bytes=slot_bytes, v_tiles=v_tiles,
                init_vec_head=init_vec_head, prek_vec_tok=prek_vec_tok,
                redk_vec_tok=redk_vec_tok, upd_core_vec_tok=upd_core_vec_tok,
                state_copy_vec_tok=state_copy_vec_tok, qalu_vec_tok=qalu_vec_tok,
                redq_vec_tok=redq_vec_tok, red_ratio=red_ratio, cast_ratio=cast_ratio,
                pipe_v_ms=pipe_v_ms, flag_ms=flag_ms,
            )
            if len(runs) == 1:
                q_i, h_i, s_i = runs[0]
                return KernelProgram(_build_rgdr_run_stages(rc, ctx, q=q_i, head_count=h_i, seq_count=s_i, **run_kwargs))
            return PipelineProgram(
                len(runs), 1,
                lambda idx, r=runs, kw=run_kwargs: (Stage(
                    "run",
                    program=KernelProgram(_build_rgdr_run_stages(
                        rc, ctx, q=r[idx][0], head_count=r[idx][1], seq_count=r[idx][2], **kw,
                    )),
                ),),
            )

        owner_progs = tuple(_owner_program(aivs[idx], segs) for idx, segs in enumerate(ownership))
        return owner_progs[0] if len(owner_progs) == 1 else ParallelProgram(owner_progs)

    return make_plan(
        ctx, RECURRENT_GDN_ABI, totals, sources=RECURRENT_GDN_SOURCES,
        owners=active_aivs, program=lower,
        metadata={
            "template": "native_recurrent_gdn", "active_aivs": active_aivs, "waves": waves,
            "v_step": v_step, "v_tiles": v_tiles,
            "state_tile_bytes": report.state_tile_bytes, "owner_ub_bytes": report.owner_ub_bytes,
            "state_contraction_ops": 2 * total_tokens * c.hv * c.dk * c.dv,
            "persistent_ub_resident": True,
            "recurrent_head_schedule": {
                "logical_heads": total_heads,
                "active_aivs": active_aivs,
                "owner_heads": owner_heads,
            },
        },
    )
