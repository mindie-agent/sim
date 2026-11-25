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

from mindie_sim.schema.datapath import SourceRef

from ...mechanism import OperatorPlan, Work
from ...program import (
    KernelProgram,
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
#   3 pre-ALU passes + 1 P09_reduce(k) + 1 q-ALU + 1 P09_reduce(q), plus BF16<->FP32 casts at P09_cast.
RGDR_PRE_ALU_PASSES = 3
RGDR_ARCH35_VSTEP_ELEMS = 8192  # recurrent_gated_delta_rule_arch35.h:149: vStep_ = 8192 / alignDK_
RGDR_PIPE_V_PER_VSTEP_TOKEN = 10  # recurrent_gated_delta_rule_arch{22,35}.h: 10 PipeBarrier<PIPE_V> per tile-token
RGDR_QUEUE_SYNCS_PER_HEAD = 10  # CopyInGamaK/QK/V/State + CopyOutState/Attn EnQue/DeQue handshakes
RGDR_MS_PER_S = 1000.0
GDN_REF_HEAD_DIM = 128
GDN_REF_DIM_SUM = float(GDN_REF_HEAD_DIM * 2)


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


def _rgdr_cast_ms_per_aiv(rc: RecipeContext, cast_bytes_per_head: float) -> float:
    """Return calibrated P09_cast latency in ms for one head's BF16<->FP32 cast on one AIV."""
    if cast_bytes_per_head <= 0.0:
        return 0.0
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        cast_bps = table.cell("P09_vec_reduce_cast").band("cast_bytes_per_s_per_aiv").p50
        if cast_bps is None or float(cast_bps) <= 0.0:
            raise ModelingError("rgdr_missing_calibrated_cast_primitive_band")
        return (cast_bytes_per_head / float(cast_bps)) * RGDR_MS_PER_S
    cast_rate = rc.service_rate("cast") or rc.service_rate("vector_cast")
    if cast_rate and float(cast_rate) > 0.0:
        return (cast_bytes_per_head / float(cast_rate)) * RGDR_MS_PER_S
    raise ModelingError("rgdr_missing_calibrated_cast_rate")


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


def build_native_recurrent_gdn(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower native CANN aclnnRecurrentGatedDeltaRule (BF16 or FP32 state, Arch22/Arch35)."""
    verify_native_gdn_abi(c, ctx, abi_id=RECURRENT_GDN_ABI)
    _, aivs = owner_units(ctx)
    if not aivs:
        raise missing("linear_attention_no_physical_owner", "Recurrent GDN requires AIV owners")

    elem = element_bytes(c.dtype)
    state_elem = element_bytes(c.state_dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    n_seqs = len(c.lengths)
    total_tokens = sum(c.lengths)
    max_q = max(c.lengths)
    total_heads = n_seqs * c.hv
    active_aivs = min(len(aivs), total_heads)
    waves = ceil_div(total_heads, active_aivs)
    v_step = max(1, min(c.dv, RGDR_ARCH35_VSTEP_ELEMS // max(1, c.dk)))
    v_tiles = ceil_div(c.dv, v_step)

    slot_bytes = c.dk * c.dv * state_elem
    # recurrent_gated_delta_rule_arch35.h:438,465: CopyInState/CopyOutState execute unconditionally on in-place state.
    state_read = total_heads * slot_bytes
    state_write = total_tokens * c.hv * slot_bytes
    qkv_read = total_tokens * (2 * c.hk * c.dk * elem + c.hv * (c.dv * elem + gate_elem + beta_elem))
    out_write = total_tokens * c.hv * c.dv * elem
    mte2_total = state_read + qkv_read
    mte3_total = state_write + out_write

    needs_state_cast = 1 if state_elem < FP32_BYTES else 0
    state_elems = c.dk * c.dv
    cast_bytes_per_head = float(state_elems * FP32_BYTES) if needs_state_cast else 0.0
    pre_alu_ops = total_tokens * c.hv * RGDR_PRE_ALU_PASSES * state_elems
    red_k_ops = total_tokens * c.hv * state_elems
    q_alu_ops = total_tokens * c.hv * state_elems
    red_q_ops = total_tokens * c.hv * state_elems
    vec_ops = pre_alu_ops + red_k_ops + q_alu_ops + red_q_ops

    read_phase = Phase("rgdr_read", transfer(ctx, read=mte2_total))
    vec_phase = Phase("rgdr_compute", Work(resources={"vector": vec_ops}), ("rgdr_read",))
    write_phase = Phase("rgdr_write", transfer(ctx, write=mte3_total), ("rgdr_compute",))
    totals = (read_phase, vec_phase, write_phase)
    report = analyze_state_residency(c, ctx, persistent=True)
    full_waves, rem_heads = divmod(total_heads, active_aivs)

    def lower(rc: RecipeContext) -> KernelProgram:
        red_ratio = _rgdr_reduce_ratio(rc)
        cast_ms = _rgdr_cast_ms_per_aiv(rc, cast_bytes_per_head)
        barrier_ms = (
            v_tiles * RGDR_PIPE_V_PER_VSTEP_TOKEN * p12_pipe_v_ms(rc)
            + RGDR_QUEUE_SYNCS_PER_HEAD * p12_flag_ms(rc)
        )

        def _wave_stages(k_heads: int) -> tuple[Stage, ...]:
            share = k_heads / float(total_heads)
            wave_aivs = aivs[:k_heads]
            w_mte2 = mte2_total * share
            w_mte3_tok = (mte3_total * share) / float(max_q)
            w_pre_tok = (pre_alu_ops * share) / float(max_q)
            w_rk_tok = (red_k_ops * share) / float(max_q)
            w_qa_tok = (q_alu_ops * share) / float(max_q)
            w_rq_tok = (red_q_ops * share) / float(max_q)

            # CopyInState + CopyInQK/V execute synchronously before the Q token loop (arch{22,35}.h:361,465).
            stages: list[Stage] = [
                bind_phase(Phase("rgdr_read", transfer(ctx, read=w_mte2)), rc, ctx, (), wave_aivs),
            ]
            prev_vec = "rgdr_read"
            if cast_ms > 0.0:
                stages.append(Stage("rgdr_cast_in", after=(prev_vec,), latency=cast_ms, keep_empty=True))
                prev_vec = "rgdr_cast_in"
            prev_mte3: str | None = None
            for t in range(max_q):
                pre_name = f"rgdr_pre_{t}"
                rk_name = f"rgdr_redk_{t}"
                co_name = f"rgdr_castout_{t}"
                wr_name = f"rgdr_write_{t}"
                qa_name = f"rgdr_qalu_{t}"
                rq_name = f"rgdr_redq_{t}"
                stages.append(bind_phase(
                    Phase(pre_name, Work(resources={"vector": w_pre_tok}), (prev_vec,)),
                    rc, ctx, (), wave_aivs,
                ))
                stages.append(_cap_vec_stage(
                    bind_phase(Phase(rk_name, Work(resources={"vector": w_rk_tok}), (pre_name,)),
                               rc, ctx, (), wave_aivs),
                    rc, red_ratio,
                ))
                after_pre = rk_name
                if cast_ms > 0.0:
                    stages.append(Stage(co_name, after=(rk_name,), latency=cast_ms, keep_empty=True))
                    after_pre = co_name
                # CopyOutState (SINGLE_BUFFER MTE3) overlaps with query VecMulMat + ReduceSum (arch35.h:438-445).
                wr_after = (after_pre, prev_mte3) if prev_mte3 is not None else (after_pre,)
                stages.append(bind_phase(
                    Phase(wr_name, transfer(ctx, write=w_mte3_tok), wr_after),
                    rc, ctx, (), wave_aivs,
                ))
                stages.append(bind_phase(
                    Phase(qa_name, Work(resources={"vector": w_qa_tok}), (after_pre,)),
                    rc, ctx, (), wave_aivs,
                ))
                stages.append(_cap_vec_stage(
                    bind_phase(Phase(rq_name, Work(resources={"vector": w_rq_tok}), (qa_name,)),
                               rc, ctx, (), wave_aivs),
                    rc, red_ratio,
                ))
                prev_vec = rq_name
                prev_mte3 = wr_name
            if barrier_ms > 0.0:
                tail_after = (prev_vec, prev_mte3) if prev_mte3 is not None else (prev_vec,)
                stages.append(Stage("rgdr_sync", after=tail_after, latency=barrier_ms, keep_empty=True))
            return tuple(stages)

        if full_waves == 1 and rem_heads == 0:
            return KernelProgram(_wave_stages(active_aivs))
        full_prog = (
            KernelProgram(_wave_stages(active_aivs))
            if full_waves == 1
            else PipelineProgram(full_waves, 1, lambda _i: _wave_stages(active_aivs))
        )
        if rem_heads == 0:
            return KernelProgram((Stage("rgdr_waves", program=full_prog),))
        return KernelProgram((
            Stage("rgdr_waves", program=full_prog),
            Stage("rgdr_tail_wave", after=("rgdr_waves",), program=KernelProgram(_wave_stages(rem_heads))),
        ))

    return make_plan(
        ctx, RECURRENT_GDN_ABI, totals, sources=RECURRENT_GDN_SOURCES,
        owners=active_aivs, program=lower,
        metadata={
            "template": "native_recurrent_gdn", "active_aivs": active_aivs, "waves": waves,
            "state_tile_bytes": report.state_tile_bytes, "owner_ub_bytes": report.owner_ub_bytes,
            "state_contraction_ops": 2 * total_tokens * c.hv * c.dk * c.dv,
            "persistent_ub_resident": True,
        },
    )
