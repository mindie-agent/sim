"""White-box vLLM-Ascend Triton KDA chunk and recurrent execution models.

Upstream-ABI: vllm_ascend.ops.triton.kda.kda.chunk_kda

[Role] Lower the seven-kernel Triton ``chunk_kda`` call chain (cumsum, eager
  RCP_LN2/zero-init, scaled dot KKT + 4x16x16 diagonal exp2/reduce, triangular
  solve, W/U/Kg recomputation, serial ``grid=(ceil(V/64), N*H)`` state
  recurrence with FP32 IEEE dot, and output GLA merge) and the ``BV=8`` Triton
  ``fused_recurrent_kda`` kernel grounded in pinned vllm-ascend sources.
[Invariants] OperatorPlan.work conserves exact physical FLOPs, ops, and bytes;
  every resource rate comes from the active die/datapath or L0 primitive table;
  zero hardcoded timing constants or reference clock floats.
[Timeouts & Locks] O(1) periodic planning; zero locks or external I/O.
[Side-Effect Policy] Pure immutable OperatorPlans and virtual Engine programs.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

from ...mechanism import OperatorPlan, Work
from ...program import (
    KernelProgram,
    PipelineProgram,
    RecipeContext,
    Stage,
)
from ..contracts import OperatorPlanningContext
from ..linear_attention.common import (
    FP32_BYTES,
    LinearAttentionContract,
    Phase,
    bind_phase,
    ceil_div,
    element_bytes,
    make_plan,
    missing,
    owner_units,
    transfer,
)
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sharedkv.l0_sync import p03_fixpipe_setup_ms, p12_flag_ms, p13_paired_ms
from ..sharedkv.split_core import p01_cold_hbm_group_cap

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
TRITON_CHUNK_KDA_ABI = "vllm_ascend.ops.triton.kda.kda.chunk_kda"
TRITON_RECURRENT_KDA_ABI = "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda"
TRITON_KDA_CHUNK_ABIS = frozenset({TRITON_CHUNK_KDA_ABI, "triton_chunk_kda"})
TRITON_KDA_RECURRENT_ABIS = frozenset({TRITON_RECURRENT_KDA_ABI, "triton_recurrent_kda"})
TRITON_CHUNK_KDA_MAX_K = 256  # kda.py:905 & chunk_delta_h.py:299 assert K <= 256

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/kda/kda.py",
        source_sha256="1e45592a0675af7dfcc879368f3d1f4ec0065455b2f49109c27a11a586516d5e",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/kda/chunk_delta_h.py",
        source_sha256="4475148caaf529664ab6b3b8d9b11771eb2ab94dd66f7df13c966973035e567f",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/kda/solve_tril.py",
        source_sha256="1920c0d8a5e213efe51e1273279b7e4161aac8a0bec3a778c2616435277daecc",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/kda/fused_recurrent_kda.py",
        source_sha256="6d18c60c4ef58e8972b7cbec785d3c1325a5b130e628ab27ab08c7b37a9c6728",
    ),
)
TRITON_KDA_SOURCES: tuple[SourceRef, ...] = SOURCES[:3]
TRITON_RECURRENT_SOURCES: tuple[SourceRef, ...] = (SOURCES[0], SOURCES[3])

# Upstream vllm_ascend/ops/triton/kda tiling & op-count constants (zero fitted timings)
STATE_TILE_BV = 64
STATE_PANEL_BK = 64
CUMSUM_BLOCK_BS = 32  # cumsum.py:118: triton.Config({"BS": 32})
RECURRENT_MAX_BV = 8  # kda.py:67: BV = min(next_power_of_2(V), 8)
STATE_VEC_U_PASSES = 3
TRITON_CHUNK_LAUNCH_COUNT = 11  # 7 Triton kernels + 4 eager ops (Mul, 3x ZerosLike)
CV_ROUNDTRIP_PAIRED_SYNCS = 1  # p13_paired_ms already measures one AIC<->AIV roundtrip
STATE_MASK_CAST_PASSES = 6  # chunk_delta_h.py:161-244: b_w/b_k masks+casts, b_v mask+cast, gk exp2
REC_FLAGS_PER_TOKEN = 16  # 6 MTE2 + 2 l2norm V_S/S_V + 6 PIPE_V barriers + 2 MTE3 stores
SOLVE_TRIL_ROW_OPS_MULT = 4  # solve_tril.py:221-236: neg-load, bcast+mul, reduce, where
INTER_AIV_WRITTEN_K_TILES = 3  # kda.py:736-746: AIV writes b_q, b_k_i, b_k_j to CV workspace
INTER_AIC_READ_K_TILES = 4  # kda.py:745-746: 2 FP32 dots read 2 LHS + 2 RHS K tiles
DOT_WS_TENSORS_PER_MM = 3  # Triton CV workspace per dot: A + B (AIV->AIC) + C (AIC->AIV)
RECOMPUTE_GATE_ALU_MULT = 5  # kda.py:1031-1091: v*b, k*b, k*exp2, gn-gk, k*exp2
RECOMPUTE_GATE_EXP2_MULT = 2  # kda.py:1064,1091: exp2(b_gk) and exp2(b_gn - b_gk)
INTER_PAIR_ALU_MULT = 7  # kda.py:736-746: 3*(sub+mul) + scale
INTER_PAIR_EXP2_MULT = 3  # kda.py:736,739,743: 3 exp2 calls per (pair, i_k)
INTRA_STEP_ALU_MULT = 7  # kda.py:861-865: sub, bcast, 3*mul, 2*where
ZERO_INIT_FP32_MATRICES = 2  # kda.py:914-915: zeros(A), zeros(Aqk)
L2NORM_VECTORS = 2  # fused_recurrent_kda.py:120-121: q and k L2 norm
L2NORM_TREE_STEPS = 7  # log2(128) dependent tree-reduction steps on 1D vector in UB
L2NORM_ALU_MULT = 8  # cast+mask+sq+add_eps+rsqrt+bcast+div+scale per vector
# fused_recurrent_kda.py:76-149 exact per-program [BV, BK] vector op counts:
# prologue: mask_h (3) + p_h0 2D ptr (5) + zeros (1) + masked load+add (2) = 11
REC_PROLOGUE_OPS_PER_STATE = 11
# decay: bcast_gk (1) + mul_exp (1) + bcast_k (1) + mul_k (1) = 4 (exp is SFU)
REC_DECAY_OPS_PER_STATE = 4
# update: bcast_v (1) + bcast_k (1) + mul (1) + add_h (1) + bcast_q (1) + mul_q (1) = 6
REC_UPDATE_OPS_PER_STATE = 6
# epilogue: p_ht 2D ptr (5) + masked store (1) = 6
REC_EPILOGUE_OPS_PER_STATE = 6
REC_SIMD_WIDTH = 64  # Arch22 256B UB vector width in FP32 lanes
REC_SHORT_VEC_OPS = 12  # [BV] masked load/cast/sub/mul/cast/store ops padded to full SIMD width
REC_ASYNC_GROUPS = 2  # Two staggered AIV halves overlap global HBM while keeping each T=1 AIV serial
REC_SMALL_GM_LOADS = 5  # q, k, v, gk, beta separate GM loads per token
TRITON_DIAG_BLOCK = 16
TRITON_INV_MERGE_DOTS = 16  # solve_tril.py:255-287: 16 IEEE FP32 16x16x16 dots
TRITON_SOLVE_DEP_ROWS = 4 * 14  # solve_tril.py:221-236: 4 blocks x 14 dependent row updates
TRITON_OUT_CV_HANDOFFS = 1  # chunk_gla_fwd_kernel_o num_stages=4 fuses 3 dots before 1 final store
MS_PER_S_KDA = 1000.0
COMPACT_THRESHOLD = 4
COMPACT_WINDOW = 2


def _next_power_of_2(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


def _hbm_burst_granule(ctx: OperatorPlanningContext) -> int:
    """Return HBM burst granule from PrimitiveTable or declared Datapath.memory_access."""
    table = getattr(ctx, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        params = table.cell("P01_datacopy").parameters or {}
        gran = int(params.get("hbm_burst_granule_bytes") or 0)
        if gran > 0:
            return gran
    mem = getattr(getattr(ctx, "datapath", None), "memory_access", None)
    gran = int(getattr(mem, "hbm_burst_granule_bytes", 0) or 0) if mem is not None else 0
    if gran > 0:
        return gran
    raise ModelingError("triton_kda_hbm_burst_granule_missing")


def _reduce_rate_ratio(rc: RecipeContext) -> float:
    """Ratio of vector ALU rate to calibrated P09 reduce rate (or vector_reduce service rate)."""
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        alu = table.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50
        red = table.cell("P09_vec_reduce_cast").band("vec_reduce_ops_per_s_per_aiv").p50
        if alu and red and red > 0.0:
            return max(1.0, float(alu) / float(red))
    vec_key = rc.resource_names.get("vector", "vector")
    vec_rate = float(rc.rates.get(vec_key, rc.rates.get("vector", 0.0)) or 0.0) * MS_PER_S_KDA
    red_rate = rc.service_rate("vector_reduce")
    if vec_rate > 0.0 and red_rate and red_rate > 0.0:
        return max(1.0, vec_rate / float(red_rate))
    raise ModelingError("triton_kda_reduce_rate_ratio_requires_primitive_table_or_service_rate")


def _sfu_rate_ratio(rc: RecipeContext) -> float:
    """Ratio of vector ALU rate to calibrated P08 SFU rate when bound."""
    table = getattr(rc, "primitive_table", None)
    if table is None or not hasattr(table, "cell"):
        raise ModelingError("triton_kda_sfu_rate_ratio_requires_primitive_table")
    alu = table.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50
    sfu = table.cell("P08_vec_sfu").band("vec_sfu_ops_per_s_per_aiv").p50
    if alu and sfu and sfu > 0.0:
        return max(1.0, float(alu) / float(sfu))
    raise ModelingError("triton_kda_sfu_rate_ratio_missing")


def _fixed_stage(name: str, duration_ms: float, after: tuple[str, ...] = ()) -> Stage:
    return Stage(name, after=after, latency=duration_ms, keep_empty=True)


def _cap_vec_stage(st: Stage, rc: RecipeContext, ratio: float) -> Stage:
    """Apply P08/P09 rate ratio via Stage.caps while conserving exact physical vector ops."""
    if ratio <= 1.0:
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


def _owner_chunk_bands(
    lengths: tuple[int, ...], bt: int, hv: int, v_tiles: int, n_owners: int,
) -> tuple[tuple[int, int], ...]:
    """Assign each state program to a core round-robin and return ``(steps, active_count)`` bands."""
    owner_chunks = [0] * n_owners
    idx = 0
    for length in lengths:
        n_chunks = ceil_div(length, bt)
        for _ in range(hv * v_tiles):
            owner_chunks[idx % n_owners] += n_chunks
            idx += 1
    owner_chunks.sort(reverse=True)
    levels = sorted({cnt for cnt in owner_chunks if cnt > 0})
    bands: list[tuple[int, int]] = []
    prev = 0
    for level in levels:
        active = sum(1 for cnt in owner_chunks if cnt >= level)
        bands.append((level - prev, active))
        prev = level
    return tuple(bands)


def _state_unit_works(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> tuple[Work, Work, Work, Work]:
    """Return exact per-program-chunk ``(store_h, ws_dot, res_dec, kv_dot)`` sub-works."""
    elem = element_bytes(c.dtype)
    gate_elem = element_bytes(c.gate_dtype)
    bt = c.bt
    prog_bv = min(c.dv, STATE_TILE_BV)
    prog_h_bf16 = prog_bv * c.dk * elem
    prog_h_fp32 = prog_bv * c.dk * FP32_BYTES
    prog_w_bytes = bt * c.dk * elem
    prog_w_fp32 = bt * c.dk * FP32_BYTES
    prog_u_bytes = bt * prog_bv * elem
    prog_v_fp32 = bt * prog_bv * FP32_BYTES
    prog_gk_bytes = c.dk * gate_elem
    prog_dot_flops = 2 * bt * prog_bv * c.dk
    cv_mem = "l2" if "l2" in ctx.geometry.rates else "hbm"
    ws_cv_bytes = 2 * (prog_w_fp32 + prog_h_fp32 + prog_v_fp32)
    kv_cv_bytes = 2 * (prog_w_bytes + prog_u_bytes + prog_h_fp32)
    w_store_h = transfer(ctx, write=prog_h_bf16) + Work(resources={"vector": 2 * prog_bv * c.dk})
    w_ws_dot = transfer(ctx, read=prog_w_bytes) + Work(
        {"fp32": prog_dot_flops},
        {"mte2_aic": prog_w_fp32 + prog_h_fp32, "mte1": prog_w_fp32 + prog_h_fp32,
         "fixpipe": prog_v_fp32, "mte3": prog_w_fp32 + prog_h_fp32, "mte2_aiv": prog_v_fp32,
         "vector": STATE_MASK_CAST_PASSES * bt * c.dk, cv_mem: ws_cv_bytes},
    )
    w_res_dec = transfer(ctx, read=prog_u_bytes + prog_gk_bytes, write=prog_u_bytes) + Work(
        resources={"vector": STATE_MASK_CAST_PASSES * prog_bv * c.dk + STATE_VEC_U_PASSES * bt * prog_bv + c.dk},
    )
    w_kv_dot = transfer(ctx, read=prog_w_bytes) + Work(
        {c.dtype: prog_dot_flops},
        {"mte2_aic": prog_w_bytes + prog_u_bytes, "mte1": prog_w_bytes + prog_u_bytes,
         "fixpipe": prog_h_fp32, "mte2_aiv": prog_h_fp32,
         "vector": STATE_MASK_CAST_PASSES * bt * c.dk + prog_bv * c.dk,
         cv_mem: kv_cv_bytes},
    )
    return w_store_h, w_ws_dot, w_res_dec, w_kv_dot


def _prepare_kernel_works(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
    total_tokens: int, head_chunks: int, total_offdiag_blocks: int,
) -> dict[str, Work]:
    """Partition pre-recurrence work cleanly across the 6 sequential pre-state kernels."""
    elem = element_bytes(c.dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    bt = c.bt
    bc = min(TRITON_DIAG_BLOCK, bt)
    nc = ceil_div(bt, bc)
    k_panels = ceil_div(c.dk, STATE_PANEL_BK)
    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    token_heads = total_tokens * c.hv
    g_bytes = token_heads * c.dk * gate_elem
    qk_bytes = token_heads * c.dk * elem
    v_bytes = token_heads * c.dv * elem
    beta_bytes = token_heads * beta_elem
    score_fp32 = head_chunks * bt * bt * FP32_BYTES
    score_bf16 = head_chunks * bt * bt * elem
    cv_mem = "l2" if "l2" in ctx.geometry.rates else "hbm"
    burst = _hbm_burst_granule(ctx)
    # Strided 64B sub-block writes across H*BT*4 stride amplify HBM write bursts on Arch22.
    row_stride = c.hv * bt * FP32_BYTES
    sub_row_bytes = bc * FP32_BYTES
    strided_burst_amp = max(1, min(burst, row_stride) // max(1, sub_row_bytes))

    # 1. chunk_local_cumsum_vector_kernel (cumsum.py:122): BS=32, 1 FP32 dot per program.
    cumsum_progs = ceil_div(c.dk, CUMSUM_BLOCK_BS) * head_chunks
    cumsum_ws = cumsum_progs * (bt * bt + 2 * bt * CUMSUM_BLOCK_BS) * FP32_BYTES
    w_cumsum = transfer(ctx, read=g_bytes, write=g_bytes) + Work(
        {"fp32": head_chunks * 2 * bt * bt * c.dk},
        {"mte2_aic": cumsum_progs * (bt * bt + bt * CUMSUM_BLOCK_BS) * FP32_BYTES,
         "mte1": cumsum_progs * (bt * bt + bt * CUMSUM_BLOCK_BS) * FP32_BYTES,
         "fixpipe": g_bytes, "mte3": cumsum_progs * bt * bt * FP32_BYTES,
         "vector": cumsum_progs * bt * bt, cv_mem: cumsum_ws},
    )

    # 2. Eager ops: g * RCP_LN2 + zeros(A) + zeros(Aqk) + zeros(Ai).
    w_eager = transfer(
        ctx, read=g_bytes, write=g_bytes + ZERO_INIT_FP32_MATRICES * score_fp32 + score_bf16,
    ) + Work(resources={"vector": token_heads * c.dk})

    # 3. chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter (kda.py:631):
    # 6 off-diag (i_i, i_j) pairs x k_panels x 2 FP32 dots + 3 exp2 + GM workspace.
    offdiag_head_pairs = c.hv * total_offdiag_blocks
    offdiag_tile = bc * bc
    inter_fp32_flops = offdiag_head_pairs * 2 * 2 * offdiag_tile * c.dk
    inter_k_tile = bc * min(c.dk, STATE_PANEL_BK) * FP32_BYTES
    inter_out_tile = offdiag_tile * FP32_BYTES
    inter_ws = offdiag_head_pairs * k_panels * (2 * (INTER_AIV_WRITTEN_K_TILES * inter_k_tile) + 2 * (2 * inter_out_tile))
    inter_alu = offdiag_head_pairs * INTER_PAIR_ALU_MULT * bc * c.dk
    inter_exp2 = offdiag_head_pairs * INTER_PAIR_EXP2_MULT * bc * c.dk
    inter_a_write = (ZERO_INIT_FP32_MATRICES * score_fp32 * (nc - 1)) // max(1, 2 * nc)
    w_inter_io = transfer(
        ctx, read=2 * qk_bytes + g_bytes + beta_bytes,
        write=inter_a_write * min(2, strided_burst_amp),
    ) + Work(
        {"fp32": inter_fp32_flops},
        {"mte2_aic": offdiag_head_pairs * k_panels * INTER_AIC_READ_K_TILES * inter_k_tile,
         "mte1": offdiag_head_pairs * k_panels * INTER_AIC_READ_K_TILES * inter_k_tile,
         "fixpipe": offdiag_head_pairs * k_panels * 2 * inter_out_tile,
         "mte3": offdiag_head_pairs * k_panels * INTER_AIV_WRITTEN_K_TILES * inter_k_tile,
         "mte2_aiv": offdiag_head_pairs * k_panels * 2 * inter_out_tile,
         "vector": inter_alu, cv_mem: inter_ws},
    )

    # 4. chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra (kda.py:782): pure AIV.
    diag_blocks = head_chunks * nc
    intra_alu = diag_blocks * (bc * c.dk + bc * INTRA_STEP_ALU_MULT * bc * c.dk)
    intra_exp2 = diag_blocks * bc * bc * c.dk
    intra_reduce = diag_blocks * bc * 2 * bc * c.dk
    intra_a_write = (ZERO_INIT_FP32_MATRICES * score_fp32) // max(1, nc)
    w_intra_io = transfer(
        ctx, read=2 * qk_bytes + g_bytes + beta_bytes,
        write=intra_a_write * strided_burst_amp,
    ) + Work(resources={"vector": intra_alu})

    # 5. merge_16x16_to_64x64_inverse_kernel_kda (solve_tril.py:175): 16 FP32 dots + 56 dep rows.
    solve_fp32_flops = head_chunks * TRITON_INV_MERGE_DOTS * 2 * offdiag_tile * bc
    solve_ws = head_chunks * TRITON_INV_MERGE_DOTS * (2 * DOT_WS_TENSORS_PER_MM * inter_out_tile)
    solve_alu = head_chunks * TRITON_SOLVE_DEP_ROWS * SOLVE_TRIL_ROW_OPS_MULT * offdiag_tile
    solve_red = head_chunks * TRITON_SOLVE_DEP_ROWS * offdiag_tile
    w_solve_io = transfer(ctx, read=score_fp32, write=score_bf16) + Work(
        {"fp32": solve_fp32_flops},
        {"mte2_aic": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "mte1": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "fixpipe": head_chunks * TRITON_INV_MERGE_DOTS * inter_out_tile,
         "mte3": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "mte2_aiv": head_chunks * TRITON_INV_MERGE_DOTS * inter_out_tile,
         "vector": solve_alu, cv_mem: solve_ws},
    )

    # 6. recompute_w_u_fwd_kernel (kda.py:968): v_tiles FP32 + k_panels BF16 dots + 2*exp2.
    recomp_bf16_flops = head_chunks * 2 * bt * bt * (c.dk + c.dv)
    recomp_dot_out = head_chunks * bt * (c.dv + c.dk) * FP32_BYTES
    recomp_ws = head_chunks * ((v_tiles + k_panels) * bt * bt * elem + 2 * recomp_dot_out // max(1, head_chunks))
    recomp_alu = RECOMPUTE_GATE_ALU_MULT * token_heads * c.dk + token_heads * c.dv
    recomp_exp2 = RECOMPUTE_GATE_EXP2_MULT * token_heads * c.dk
    w_recomp_io = transfer(
        ctx, read=qk_bytes + v_bytes + beta_bytes + score_bf16 + g_bytes,
        write=2 * qk_bytes + v_bytes,
    ) + Work(
        {c.dtype: recomp_bf16_flops},
        {"mte2_aic": score_bf16 + qk_bytes + v_bytes, "mte1": score_bf16 + qk_bytes + v_bytes,
         "fixpipe": recomp_dot_out, "mte3": qk_bytes + v_bytes, "mte2_aiv": recomp_dot_out,
         "vector": recomp_alu, cv_mem: recomp_ws},
    )

    return {
        "cumsum": w_cumsum,
        "eager": w_eager,
        "inter_io": w_inter_io,
        "inter_exp2": Work(resources={"vector": inter_exp2}),
        "intra_io": w_intra_io,
        "intra_exp2": Work(resources={"vector": intra_exp2}),
        "intra_red": Work(resources={"vector": intra_reduce}),
        "solve_io": w_solve_io,
        "solve_red": Work(resources={"vector": solve_red}),
        "recomp_io": w_recomp_io,
        "recomp_exp2": Work(resources={"vector": recomp_exp2}),
    }


def _lower_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    pairs: tuple[tuple[int, tuple[int, ...]], ...],
    active_state_pairs: tuple[tuple[int, tuple[int, ...]], ...],
    prep_works: dict[str, Work], head_chunks: int, total_offdiag_blocks: int,
    state_init_work: Work, unit_works: tuple[Work, Work, Work, Work],
    bands: tuple[tuple[int, int], ...], out_phase: Phase,
) -> KernelProgram:
    """Lower all 7 Triton chunk KDA kernels + eager ops with exact work conservation."""
    all_aics = tuple(aic for aic, _ in pairs)
    all_aivs = tuple(aiv for _, peers in pairs for aiv in peers)
    n_aics = max(1, len(all_aics))
    reduce_ratio = _reduce_rate_ratio(rc)
    sfu_ratio = _sfu_rate_ratio(rc)
    cv_ms = p13_paired_ms(rc)
    fix_ms = p03_fixpipe_setup_ms(rc)
    k_panels = ceil_div(c.dk, STATE_PANEL_BK)
    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    stages: list[Stage] = []

    def _push(st: Stage) -> str:
        stages.append(st)
        return st.name

    # 1. chunk_local_cumsum_vector_kernel
    prev = _push(bind_phase(Phase("kda_cumsum", prep_works["cumsum"]), rc, ctx, all_aics, all_aivs))
    cumsum_waves = ceil_div(ceil_div(c.dk, CUMSUM_BLOCK_BS) * head_chunks, n_aics)
    cumsum_sync = cumsum_waves * (cv_ms + fix_ms)
    if cumsum_sync > 0.0:
        prev = _push(_fixed_stage("kda_cumsum_cv", cumsum_sync, after=(prev,)))

    # 2. Eager RCP_LN2 + zero-inits
    prev = _push(bind_phase(Phase("kda_eager_scale_zeros", prep_works["eager"], (prev,)), rc, ctx, (), all_aivs))

    # 3. intra_sub_inter (6 pairs x k_panels x 2 dots)
    prev = _push(bind_phase(Phase("kda_inter_io", prep_works["inter_io"], (prev,)), rc, ctx, all_aics, all_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_inter_exp2", prep_works["inter_exp2"], (prev,)), rc, ctx, (), all_aivs),
        rc, sfu_ratio,
    ))
    inter_dots_per_wave = ceil_div(c.hv * total_offdiag_blocks, n_aics) * k_panels * 2
    inter_sync = inter_dots_per_wave * (cv_ms + fix_ms)
    if inter_sync > 0.0:
        prev = _push(_fixed_stage("kda_inter_cv", inter_sync, after=(prev,)))

    # 4. intra_sub_intra (pure AIV)
    prev = _push(bind_phase(Phase("kda_intra_io_alu", prep_works["intra_io"], (prev,)), rc, ctx, (), all_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_intra_exp2", prep_works["intra_exp2"], (prev,)), rc, ctx, (), all_aivs),
        rc, sfu_ratio,
    ))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_intra_reduce", prep_works["intra_red"], (prev,)), rc, ctx, (), all_aivs),
        rc, reduce_ratio,
    ))

    # 5. solve_tril (16 FP32 dots + 56 dep rows)
    prev = _push(bind_phase(Phase("kda_solve_io_alu", prep_works["solve_io"], (prev,)), rc, ctx, all_aics, all_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_solve_reduce", prep_works["solve_red"], (prev,)), rc, ctx, (), all_aivs),
        rc, reduce_ratio,
    ))
    solve_waves = ceil_div(head_chunks, n_aics)
    solve_sync = (
        solve_waves * TRITON_INV_MERGE_DOTS * (cv_ms + fix_ms)
        + ceil_div(head_chunks, max(1, len(all_aivs))) * TRITON_SOLVE_DEP_ROWS * p12_flag_ms(rc)
    )
    if solve_sync > 0.0:
        prev = _push(_fixed_stage("kda_solve_cv", solve_sync, after=(prev,)))

    # 6. recompute_w_u (v_tiles + k_panels dots + 2*exp2)
    prev = _push(bind_phase(Phase("kda_recomp_io_alu", prep_works["recomp_io"], (prev,)), rc, ctx, all_aics, all_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_recomp_exp2", prep_works["recomp_exp2"], (prev,)), rc, ctx, (), all_aivs),
        rc, sfu_ratio,
    ))
    recomp_sync = solve_waves * (v_tiles + k_panels) * (cv_ms + fix_ms)
    if recomp_sync > 0.0:
        prev = _push(_fixed_stage("kda_recomp_cv", recomp_sync, after=(prev,)))

    # 7. chunk_delta_h (grid=(ceil(V/64), N*H) serial recurrence; only HBM capped to active cores)
    st_aics = tuple(aic for aic, _ in active_state_pairs)
    st_aivs = tuple(peers[0] for _, peers in active_state_pairs)
    prev = _push(bind_phase(Phase("state_init_final_io", state_init_work, (prev,)), rc, ctx, st_aics, st_aivs))
    cv_sync_per_step_ms = (2 * k_panels) * (CV_ROUNDTRIP_PAIRED_SYNCS * cv_ms + fix_ms)
    w_store_h, w_ws_dot, w_res_dec, w_kv_dot = unit_works

    for band_idx, (steps, k_active) in enumerate(bands):
        b_pairs = active_state_pairs[:k_active]
        b_aics = tuple(aic for aic, _ in b_pairs)
        b_aivs = tuple(peers[0] for _, peers in b_pairs)
        caps = p01_cold_hbm_group_cap(rc, k_active, include_l2=False)
        ws0 = w_store_h.scaled(k_active)
        ws1 = w_ws_dot.scaled(k_active)
        ws2 = w_res_dec.scaled(k_active)
        ws3 = w_kv_dot.scaled(k_active)

        def _cap(st: Stage, cap_map=caps) -> Stage:
            applied = {k: v for k, v in cap_map.items() if k in st.work}
            return replace(st, caps={**st.caps, **applied}) if applied else st

        def _band_step(_idx: int, ws0=ws0, ws1=ws1, ws2=ws2, ws3=ws3, b_aics=b_aics, b_aivs=b_aivs, _cap=_cap):
            s0 = _cap(bind_phase(Phase("h_store_enter", ws0), rc, ctx, b_aics, b_aivs))
            s1 = _cap(bind_phase(Phase("h_ws_fp32_dot", ws1, ("h_store_enter",)), rc, ctx, b_aics, b_aivs))
            s2 = _cap(bind_phase(Phase("h_res_decay", ws2, ("h_ws_fp32_dot",)), rc, ctx, b_aics, b_aivs))
            s3 = _cap(bind_phase(Phase("h_kv_bf16_dot", ws3, ("h_res_decay",)), rc, ctx, b_aics, b_aivs))
            if cv_sync_per_step_ms > 0.0:
                return (s0, s1, s2, s3, _fixed_stage("h_chunk_cv_sync", cv_sync_per_step_ms, after=("h_kv_bf16_dot",)))
            return (s0, s1, s2, s3)

        b_prog = compact_periodic_pipeline(
            rc, steps, 1, _band_step, head=COMPACT_WINDOW, tail=COMPACT_WINDOW, threshold=COMPACT_THRESHOLD,
        )
        prev = _push(Stage(f"state_recurrence_band_{band_idx}", after=(prev,), program=b_prog))

    # 8. chunk_gla_fwd_kernel_o
    prev = _push(bind_phase(Phase(out_phase.name, out_phase.work, (prev,)), rc, ctx, all_aics, all_aivs))
    out_sync = solve_waves * TRITON_OUT_CV_HANDOFFS * (cv_ms + fix_ms)
    if out_sync > 0.0:
        _push(_fixed_stage("kda_out_cv", out_sync, after=(prev,)))
    return KernelProgram(tuple(stages))


def build_triton_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower vllm_ascend.ops.triton.kda.kda.chunk_kda (7 kernels + 4 eager ops)."""
    if c.dk > TRITON_CHUNK_KDA_MAX_K:
        raise missing(
            "triton_chunk_kda_head_dim_k_unsupported",
            f"vllm_ascend Triton chunk_kda requires dk <= {TRITON_CHUNK_KDA_MAX_K} (got dk={c.dk})",
        )
    aics, aivs = owner_units(ctx)
    pairs = tuple(
        (aic, peers[:2]) for aic, peers in ctx.datapath.aic_aiv_pairs
        if aic in aics and len(peers) >= 2 and all(aiv in aivs for aiv in peers[:2])
    )
    if not pairs:
        raise missing("linear_attention_no_mixed_owners", "Triton chunk KDA requires paired AIC/AIV owners")

    elem = element_bytes(c.dtype)
    state_elem = element_bytes(c.state_dtype)
    gate_elem = element_bytes(c.gate_dtype)
    bt = c.bt
    total_tokens = sum(c.lengths)
    total_chunks = sum(ceil_div(length, bt) for length in c.lengths)
    head_chunks = total_chunks * c.hv
    token_heads = total_tokens * c.hv

    total_offdiag_blocks = 0
    for length in c.lengths:
        rem = length
        while rem > 0:
            rows = min(bt, rem)
            n_sub = ceil_div(rows, TRITON_DIAG_BLOCK)
            total_offdiag_blocks += (n_sub * (n_sub - 1)) // 2
            rem -= rows

    prep_works = _prepare_kernel_works(c, ctx, total_tokens, head_chunks, total_offdiag_blocks)
    cumsum_phase = Phase("triton_kda_cumsum", prep_works["cumsum"])
    prep_body_work = (
        prep_works["eager"] + prep_works["inter_io"] + prep_works["inter_exp2"]
        + prep_works["intra_io"] + prep_works["intra_exp2"] + prep_works["intra_red"]
        + prep_works["solve_io"] + prep_works["solve_red"]
        + prep_works["recomp_io"] + prep_works["recomp_exp2"]
    )
    pre_phase = Phase("triton_kda_prepare_alu", prep_body_work, ("triton_kda_cumsum",))

    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    state_progs = v_tiles * len(c.lengths) * c.hv
    active_state_pairs = pairs[:min(len(pairs), state_progs)]
    bands = _owner_chunk_bands(c.lengths, bt, c.hv, v_tiles, len(active_state_pairs))
    unit_works = _state_unit_works(c, ctx)
    one_chunk_state_work = unit_works[0] + unit_works[1] + unit_works[2] + unit_works[3]
    init_state_bytes = len(c.lengths) * c.hv * c.dk * c.dv * state_elem
    state_init_work = transfer(
        ctx, read=init_state_bytes * c.use_initial_state,
        write=init_state_bytes * c.output_final_state,
    )
    total_state_chunks = total_chunks * c.hv * v_tiles
    state_phase = Phase(
        "triton_kda_state_recurrence",
        state_init_work + one_chunk_state_work.scaled(total_state_chunks),
        ("triton_kda_prepare_alu",),
    )

    g_bytes = token_heads * c.dk * gate_elem
    score_fp32 = head_chunks * bt * bt * FP32_BYTES
    score_bf16 = head_chunks * bt * bt * elem
    entering_bank_bytes = head_chunks * c.dk * c.dv * elem
    out_read = token_heads * (c.dk + c.dv) * elem + g_bytes + score_fp32 + entering_bank_bytes
    out_write = token_heads * c.dv * elem
    out_cube_flops = head_chunks * 2 * (bt * c.dk * c.dv + bt * bt * c.dv)
    out_phase = Phase(
        "triton_kda_output_o",
        transfer(ctx, read=out_read, write=out_write) + Work(
            {c.dtype: out_cube_flops},
            {"mte1": token_heads * (c.dk + c.dv) * elem + entering_bank_bytes + score_bf16,
             "fixpipe": out_write, "vector": 2 * token_heads * c.dk},
        ),
        ("triton_kda_state_recurrence",),
    )

    totals = (cumsum_phase, pre_phase, state_phase, out_phase)

    def lower(rc: RecipeContext) -> KernelProgram:
        return _lower_chunk_kda(
            c, ctx, rc, pairs, active_state_pairs, prep_works,
            head_chunks, total_offdiag_blocks, state_init_work,
            unit_works, bands, out_phase,
        )

    return make_plan(
        ctx, TRITON_CHUNK_KDA_ABI, totals, sources=TRITON_KDA_SOURCES,
        launch_count=TRITON_CHUNK_LAUNCH_COUNT, program=lower,
        metadata={
            "template": "triton_chunk_kda", "chunk_tokens": bt,
            "total_chunks": total_chunks, "state_programs": state_progs,
            "active_state_pairs": len(active_state_pairs), "persistent_ub_resident": False,
        },
    )


def _lower_recurrent_kda(
    ctx: OperatorPlanningContext,
    rc: RecipeContext,
    *,
    aivs: tuple[int, ...],
    active_aivs: int,
    waves: int,
    total_progs: int,
    max_q: int,
    total_hbm_read: float,
    total_hbm_write: float,
    total_mte2_aiv: float,
    total_l2_hits: float,
    has_l2_rate: bool,
    decay_alu: float,
    sfu_ops: float,
    dot1_red: float,
    update_alu: float,
    dot2_red: float,
    ht_mte3_total: float,
) -> KernelProgram:
    reduce_ratio = _reduce_rate_ratio(rc)
    sfu_ratio = _sfu_rate_ratio(rc)
    flag_ms = max_q * REC_FLAGS_PER_TOKEN * p12_flag_ms(rc)

    def _block_stages(k_progs: int, grp_aivs: tuple[int, ...], pfx: str = "") -> tuple[Stage, ...]:
        share = k_progs / float(total_progs)
        w_load = Work(resources={
            "hbm": total_hbm_read * share,
            "mte2_aiv": total_mte2_aiv * share,
            **({"l2": total_l2_hits * share} if (total_l2_hits and has_l2_rate) else {}),
        })
        s0 = bind_phase(Phase(f"{pfx}rec_load", w_load), rc, ctx, (), grp_aivs)
        s1 = bind_phase(
            Phase(f"{pfx}rec_decay_alu", Work(resources={"vector": decay_alu * share}), (s0.name,)),
            rc, ctx, (), grp_aivs,
        )
        s2 = _cap_vec_stage(
            bind_phase(Phase(f"{pfx}rec_decay_sfu", Work(resources={"vector": sfu_ops * share}), (s1.name,)),
                       rc, ctx, (), grp_aivs),
            rc, sfu_ratio,
        )
        s3 = _cap_vec_stage(
            bind_phase(Phase(f"{pfx}rec_dot1_reduce", Work(resources={"vector": dot1_red * share}), (s2.name,)),
                       rc, ctx, (), grp_aivs),
            rc, reduce_ratio,
        )
        s4 = bind_phase(
            Phase(f"{pfx}rec_update_alu", Work(resources={"vector": update_alu * share}), (s3.name,)),
            rc, ctx, (), grp_aivs,
        )
        s5 = _cap_vec_stage(
            bind_phase(Phase(f"{pfx}rec_dot2_reduce", Work(resources={"vector": dot2_red * share}), (s4.name,)),
                       rc, ctx, (), grp_aivs),
            rc, reduce_ratio,
        )
        w_store = Work(resources={"hbm": total_hbm_write * share, "mte3": ht_mte3_total * share})
        s6 = bind_phase(Phase(f"{pfx}rec_store", w_store, (s5.name,)), rc, ctx, (), grp_aivs)
        if flag_ms > 0.0:
            return (s0, s1, s2, s3, s4, s5, s6, _fixed_stage(f"{pfx}rec_sync", flag_ms, after=(s6.name,)))
        return (s0, s1, s2, s3, s4, s5, s6)

    def _group_prog(grp_aivs: tuple[int, ...], grp_progs: int, pfx: str) -> KernelProgram:
        g_n = len(grp_aivs)
        g_full, g_rem = divmod(grp_progs, g_n)
        stages_g: list[Stage] = []
        prev_g: tuple[str, ...] = ()
        if g_full > 0:
            body = (
                KernelProgram(_block_stages(g_n, grp_aivs, pfx))
                if g_full == 1
                else compact_periodic_pipeline(
                    rc, g_full, 1, lambda _i, g_n=g_n, ga=grp_aivs, p=pfx: _block_stages(g_n, ga, p),
                    head=COMPACT_WINDOW, tail=COMPACT_WINDOW, threshold=COMPACT_THRESHOLD,
                )
            )
            stages_g.append(Stage(f"{pfx}full", program=body))
            prev_g = (f"{pfx}full",)
        if g_rem > 0:
            stages_g.append(Stage(
                f"{pfx}tail", after=prev_g, program=KernelProgram(_block_stages(g_rem, grp_aivs[:g_rem], f"{pfx}t_")),
            ))
        return KernelProgram(tuple(stages_g))

    if active_aivs < REC_ASYNC_GROUPS or waves <= 1:
        return _group_prog(aivs[:active_aivs], total_progs, "g0_")
    # Two independent AIV halves keep T=1 load->vec->store serial per AIV while staggering HBM.
    mid = active_aivs // REC_ASYNC_GROUPS
    a_aivs, b_aivs = aivs[:mid], aivs[mid:active_aivs]
    a_progs = (total_progs * len(a_aivs)) // active_aivs
    b_progs = total_progs - a_progs
    a_head = _block_stages(len(a_aivs), a_aivs, "ga0_")
    a_rem_progs = a_progs - len(a_aivs)
    if a_rem_progs <= 0:
        return KernelProgram((
            Stage("triton_rec_half_a", program=KernelProgram(a_head)),
            Stage("triton_rec_half_b", program=_group_prog(b_aivs, b_progs, "gb_")),
        ))
    return KernelProgram((
        *a_head,
        Stage("triton_rec_half_a_rest", after=(a_head[-1].name,), program=_group_prog(a_aivs, a_rem_progs, "ga_")),
        Stage("triton_rec_half_b", after=(a_head[0].name,), program=_group_prog(b_aivs, b_progs, "gb_")),
    ))


def _resolve_triton_recurrent_state_counts(c: LinearAttentionContract) -> tuple[int, int]:
    """Return ``(loaded_initial_seqs, stored_checkpoint_tokens)`` per fused_recurrent_kda.py."""
    n_seqs = len(c.lengths)
    total_tokens = sum(c.lengths)
    inplace_final_state = bool(c.metadata.get("inplace_final_state", True))
    raw_indices = c.metadata.get("ssm_state_indices")
    if not isinstance(raw_indices, (tuple, list)):
        return (n_seqs if c.use_initial_state else 0), total_tokens
    if raw_indices and isinstance(raw_indices[0], (tuple, list)):
        rows = tuple(tuple(int(v) for v in row) for row in raw_indices)
    elif len(raw_indices) == total_tokens and n_seqs > 0:
        flat = tuple(int(v) for v in raw_indices)
        rows_list: list[tuple[int, ...]] = []
        cursor = 0
        for seq_len in c.lengths:
            rows_list.append(flat[cursor : cursor + seq_len])
            cursor += seq_len
        rows = tuple(rows_list)
    else:
        rows = tuple((int(v),) for v in raw_indices)
    num_accepted = c.metadata.get("num_accepted_tokens")
    loaded_seqs = 0
    stored_tokens = 0
    for seq_idx, seq_len in enumerate(c.lengths):
        row = rows[seq_idx] if seq_idx < len(rows) else ()
        if c.use_initial_state:
            init_pos = (
                max(0, int(num_accepted[seq_idx]) - 1)
                if isinstance(num_accepted, (tuple, list)) and seq_idx < len(num_accepted)
                else 0
            )
            init_slot = row[init_pos] if init_pos < len(row) else 0
            # fused_recurrent_kda.py:106-107: state_idx <= 0 returns early before h0 load and token loop.
            if init_slot <= 0:
                continue
            loaded_seqs += 1
        if inplace_final_state:
            stored_tokens += sum(1 for v in row[:seq_len] if v > 0)
        else:
            stored_tokens += seq_len
    return loaded_seqs, stored_tokens


def build_triton_recurrent_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda (grid=(NK, NV, N*HV), BV=8)."""
    _, aivs = owner_units(ctx)
    if not aivs:
        raise missing("linear_attention_no_physical_owner", "Triton recurrent KDA requires AIV owners")

    elem = element_bytes(c.dtype)
    state_elem = element_bytes(c.state_dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    bk = _next_power_of_2(c.dk)
    bv = min(_next_power_of_2(c.dv), RECURRENT_MAX_BV)
    nk = ceil_div(c.dk, bk)
    nv = ceil_div(c.dv, bv)
    n_seqs = len(c.lengths)
    total_tokens = sum(c.lengths)
    max_q = max(c.lengths)
    use_l2norm = bool(c.metadata.get("use_qk_l2norm_in_kernel", True))

    total_progs = nk * nv * n_seqs * c.hv
    active_aivs = min(len(aivs), total_progs)
    waves = ceil_div(total_progs, active_aivs)
    burst = _hbm_burst_granule(ctx)

    # fused_recurrent_kda.py:99-148: kernel stores every token checkpoint to ht
    # whenever INPLACE_FINAL_STATE=False, or for every positive ssm_state_idx when True,
    # independent of the wrapper return flag output_final_state; state_idx <= 0 early-returns.
    loaded_seqs, stored_tokens = _resolve_triton_recurrent_state_counts(c)

    head_state_bytes = c.dk * c.dv * state_elem
    h0_read_total = loaded_seqs * c.hv * head_state_bytes
    ht_write_total = stored_tokens * c.hv * head_state_bytes
    qkvgb_per_tok_head = (2 * c.dk + c.dv) * elem + c.dk * gate_elem + beta_elem
    qkvgb_hbm_total = total_tokens * c.hv * qkvgb_per_tok_head
    # Each program issues 5 separate small vector GM loads (q, k, v, gk, beta) -> min burst per load on MTE2.
    qkvgb_mte2_per_prog = max(2 * c.dk * elem + c.dk * gate_elem + beta_elem + bv * elem, REC_SMALL_GM_LOADS * burst)
    qkvgb_mte2_total = total_tokens * c.hv * nv * qkvgb_mte2_per_prog
    out_write_total = total_tokens * c.hv * c.dv * elem
    ht_mte3_total = ht_write_total + total_tokens * c.hv * nv * burst

    # Exact per-program vector/SFU/reduce op counts grounded in fused_recurrent_kda.py:76-149.
    prog_state = bv * bk
    prologue_alu = total_progs * REC_PROLOGUE_OPS_PER_STATE * prog_state
    l2norm_alu = (total_tokens * c.hv * nv * L2NORM_VECTORS * L2NORM_ALU_MULT * bk) if use_l2norm else 0
    l2norm_red = (total_tokens * c.hv * nv * L2NORM_VECTORS * L2NORM_TREE_STEPS * bk) if use_l2norm else 0
    l2norm_sfu = (total_tokens * c.hv * nv * L2NORM_VECTORS * bk) if use_l2norm else 0
    short_vec_alu = total_tokens * c.hv * nv * REC_SHORT_VEC_OPS * max(bv, REC_SIMD_WIDTH)
    decay_alu = prologue_alu + l2norm_alu + short_vec_alu + total_tokens * c.hv * nv * (REC_DECAY_OPS_PER_STATE * prog_state + bk)
    # fused_recurrent_kda.py:127-128: tl.exp(b_gk[None, :]) evaluates exp on [1, BK] before broadcasting across BV.
    sfu_ops = total_tokens * c.hv * nv * bk + l2norm_sfu
    dot1_red = l2norm_red + total_tokens * c.hv * nv * prog_state
    update_alu = total_tokens * c.hv * nv * (REC_UPDATE_OPS_PER_STATE + REC_EPILOGUE_OPS_PER_STATE) * prog_state
    dot2_red = total_tokens * c.hv * nv * prog_state
    total_vec = decay_alu + sfu_ops + dot1_red + update_alu + dot2_red

    total_hbm_read = h0_read_total + qkvgb_hbm_total
    total_hbm_write = ht_write_total + out_write_total
    total_mte2_aiv = h0_read_total + qkvgb_mte2_total
    total_l2_hits = max(0, qkvgb_mte2_total - qkvgb_hbm_total)
    has_l2_rate = "l2" in ctx.geometry.rates

    phases = (
        Phase("load_initial_state", Work(resources={
            "hbm": h0_read_total, "mte2_aiv": h0_read_total,
        })),
        Phase("recurrent_token_loop", Work(resources={
            "hbm": qkvgb_hbm_total + total_hbm_write,
            "mte2_aiv": qkvgb_mte2_total,
            "mte3": ht_mte3_total,
            "vector": total_vec,
            **({"l2": total_l2_hits} if (total_l2_hits and has_l2_rate) else {}),
        }), ("load_initial_state",)),
    )

    def lower(rc: RecipeContext) -> KernelProgram:
        return _lower_recurrent_kda(
            ctx, rc,
            aivs=aivs, active_aivs=active_aivs, waves=waves, total_progs=total_progs,
            max_q=max_q, total_hbm_read=total_hbm_read, total_hbm_write=total_hbm_write,
            total_mte2_aiv=total_mte2_aiv, total_l2_hits=total_l2_hits, has_l2_rate=has_l2_rate,
            decay_alu=decay_alu, sfu_ops=sfu_ops, dot1_red=dot1_red,
            update_alu=update_alu, dot2_red=dot2_red, ht_mte3_total=ht_mte3_total,
        )

    return make_plan(
        ctx, TRITON_RECURRENT_KDA_ABI, phases, sources=TRITON_RECURRENT_SOURCES,
        owners=active_aivs, program=lower,
        metadata={
            "template": "triton_recurrent_kda", "bk": bk, "bv": bv,
            "total_programs": total_progs, "active_aivs": active_aivs, "waves": waves,
            "persistent_ub_resident": True,
        },
    )
