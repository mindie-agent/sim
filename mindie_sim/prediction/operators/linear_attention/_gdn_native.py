"""Native CANN GDN chunk (Arch35 VF) and recurrent (Arch22/Arch35) helpers.

Upstream-ABI: aclnnRecurrentGatedDeltaRule

[Role] Provide exact Arch35 persistent GDN WY vector-VF issue & transfer
  accounting (from ``chunk_gated_delta_rule_{cube,vector}_compute.h`` and
  ``stage{1,2}_vf.h``) plus native ``aclnnRecurrentGatedDeltaRule`` lowering
  (Arch22 single-buffered ``BUFFER_NUM=1`` vs Arch35 double-buffered input /
  single-buffered output queues).
[Invariants] Conserves physical FLOPs, ops, and bytes; uses periodic pipeline
  compaction for O(1) Arch35 WY lowering up to 409.6K tokens; zero hardcoded
  timing constants (sync/barrier latencies come from calibrated P12/P13 L0 cells).
[Timeouts & Locks] O(owners) planning; zero locks or external I/O.
[Side-Effect Policy] Pure immutable OperatorPlans and virtual Engine programs.
"""
from __future__ import annotations

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
from ..sharedkv.l0_sync import p12_pipe_v_ms, p13_paired_ms
from .common import (
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
)

OPS_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
RECURRENT_GDN_ABI = "aclnnRecurrentGatedDeltaRule"

RECURRENT_GDN_SOURCES: tuple[SourceRef, ...] = (
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

RECURRENT_GDN_VEC_OPS_PER_STATE = 6  # Cast(in) + decay + dot(k) + outer + dot(q) + Cast(out)
GDN_REF_HEAD_DIM = 128
GDN_REF_DIM_SUM = float(GDN_REF_HEAD_DIM * 2)


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
    """Lower native CANN aclnnRecurrentGatedDeltaRule (BF16 or FP32 state, A3/A5)."""
    _, aivs = owner_units(ctx)
    if not aivs:
        raise missing("linear_attention_no_physical_owner", "Recurrent GDN requires AIV owners")

    is_arch35 = str(ctx.compute_arch.arch_id) == "npu_arch_3510"
    elem = element_bytes(c.dtype)
    state_elem = element_bytes(c.state_dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    n_seqs = len(c.lengths)
    total_tokens = sum(c.lengths)
    total_heads = n_seqs * c.hv
    active_aivs = min(len(aivs), total_heads)
    waves = ceil_div(total_heads, active_aivs)

    slot_bytes = c.dk * c.dv * state_elem
    # recurrent_gated_delta_rule_arch35.h:438,465: CopyInState/CopyOutState execute unconditionally on in-place state.
    state_read = total_heads * slot_bytes
    state_write = total_tokens * c.hv * slot_bytes
    qkv_read = total_tokens * (2 * c.hk * c.dk * elem + c.hv * (c.dv * elem + gate_elem + beta_elem))
    out_write = total_tokens * c.hv * c.dv * elem
    mte2_total = state_read + qkv_read
    mte3_total = state_write + out_write
    vec_ops = total_tokens * c.hv * RECURRENT_GDN_VEC_OPS_PER_STATE * c.dk * c.dv

    read_phase = Phase("rgdr_read", transfer(ctx, read=mte2_total))
    vec_phase = Phase("rgdr_compute", Work(resources={"vector": vec_ops}), ("rgdr_read",))
    write_phase = Phase("rgdr_write", transfer(ctx, write=mte3_total), ("rgdr_compute",))
    totals = (read_phase, vec_phase, write_phase)
    report = analyze_state_residency(c, ctx, persistent=True)

    full_waves, rem_heads = divmod(total_heads, active_aivs)

    def lower(rc: RecipeContext) -> KernelProgram:
        def _wave_stages(k_heads: int) -> tuple[Stage, ...]:
            share = k_heads / float(total_heads)
            wave_aivs = aivs[:k_heads]
            w_mte2 = mte2_total * share
            w_mte3 = mte3_total * share
            w_vec = vec_ops * share
            if is_arch35:
                rw_stage = bind_phase(
                    Phase("rgdr_arch35_body", Work(resources={
                        "hbm": w_mte2 + w_mte3,
                        "mte2_aiv": w_mte2,
                        "mte3": w_mte3,
                        "vector": w_vec,
                    })),
                    rc, ctx, (), wave_aivs,
                )
                return (rw_stage,)
            # Arch22: BUFFER_NUM = 1 (recurrent_gated_delta_rule_arch22.h:24) serializes MTE2 -> V -> MTE3
            s_read = bind_phase(Phase("rgdr_read", transfer(ctx, read=w_mte2)), rc, ctx, (), wave_aivs)
            s_vec = bind_phase(Phase("rgdr_compute", Work(resources={"vector": w_vec}), ("rgdr_read",)),
                               rc, ctx, (), wave_aivs)
            s_write = bind_phase(Phase("rgdr_write", transfer(ctx, write=w_mte3), ("rgdr_compute",)),
                                 rc, ctx, (), wave_aivs)
            return (s_read, s_vec, s_write)

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
