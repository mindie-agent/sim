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
FP32_DTYPES = frozenset({"fp32", "float32"})
STATE_TILE_BV = 64
STATE_PANEL_BK = 64
OUT_TILE_BV = 128  # kda.py:1156: chunk_gla_fwd_kernel_o triton.Config({"BK": 64, "BV": 128})
CUMSUM_BLOCK_BS = 32  # cumsum.py:118: triton.Config({"BS": 32})
L2NORM_TILED_MBLOCK = 32  # l2norm.py:75: l2norm_fwd_tiled_kernel MBLOCK = 32
L2NORM_TILED_LOAD_ALU_OPS = 2  # l2norm.py:46-49: square, add-eps (+1 cast-in if non-FP32)
L2NORM_TILED_STORE_ALU_OPS = 1  # l2norm.py:50: mul-rsqrt (+1 cast-out if non-FP32)
RECURRENT_MAX_BV = 8  # kda.py:67: BV = min(next_power_of_2(V), 8)
STATE_VEC_U_PASSES = 3
TRITON_CHUNK_LAUNCH_COUNT = 11  # 7 Triton kernels + 4 eager ops (Mul, 3x ZerosLike)
CV_ROUNDTRIP_PAIRED_SYNCS = 1  # p13_paired_ms already measures one AIC<->AIV roundtrip
STATE_MASK_CAST_PASSES = 6  # chunk_delta_h.py:161-244: b_w/b_k masks+casts, b_v mask+cast, gk exp2
REC_FLAGS_PER_TOKEN = 16  # 6 MTE2 + 2 l2norm V_S/S_V + 6 PIPE_V barriers + 2 MTE3 stores
SOLVE_TRIL_ROW_OPS_MULT = 4  # solve_tril.py:221-236: neg-load, bcast+mul, reduce, where
TRITON_SOLVE_SKIP_ROWS = 2  # solve_tril.py:221-236: range(block_start + 2, min(block_end, rows))
BETA_SIGMOID_LOAD_ALU_OPS = 1  # kda.py:169: eager sigmoid pre-exp ALU op (neg)
BETA_SIGMOID_STORE_ALU_OPS = 2  # kda.py:169: eager sigmoid post-exp ALU ops (add-1, rcp)
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


def _pair_owners(
    pairs: tuple[tuple[int, tuple[int, ...]], ...], progs: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return active ``(aics, aivs)`` capped to ``min(len(pairs), progs)`` launched programs."""
    active = pairs[:min(len(pairs), max(1, progs))]
    return (
        tuple(aic for aic, _ in active),
        tuple(aiv for _, peers in active for aiv in peers),
    )


def _chunk_live_stats(c: LinearAttentionContract) -> dict[str, object]:
    """Derive exact per-chunk live-row iteration, store, and serial-run stats from ``c.lengths``."""
    bt = c.bt
    bc = min(TRITON_DIAG_BLOCK, bt)

    def _one_chunk(rows: int) -> tuple[int, int, int, int, int, int, int, int, int]:
        n_sub = ceil_div(rows, bc)
        offdiag_blocks = (n_sub * (n_sub - 1)) // 2
        inter_active = int(rows > bc)
        inter_rows = rows if rows > bc else 0
        inter_qb_rows = max(0, rows - bc)
        inter_live_elems = sum(
            i_i * min(bc, rows - i_i * bc) * bc for i_i in range(1, n_sub)
        )
        intra_live_elems = sum(
            min(bc, rows - i_i * bc) ** 2 for i_i in range(n_sub)
        )
        solve_live_elems = sum(
            (i_i + 1) * min(bc, rows - i_i * bc) * bc for i_i in range(n_sub)
        )
        solve_dep_rows = sum(
            max(0, min(bc, rows - b_idx * bc) - TRITON_SOLVE_SKIP_ROWS)
            for b_idx in range(n_sub)
        )
        return (
            offdiag_blocks, inter_active, inter_rows, inter_qb_rows,
            inter_live_elems, n_sub, intra_live_elems, solve_live_elems, solve_dep_rows,
        )

    full_stats = _one_chunk(bt)
    totals = [0] * len(full_stats)
    total_chunks = 0
    inter_runs: list[tuple[int, int]] = []
    solve_runs: list[tuple[int, int]] = []
    for length in c.lengths:
        full, tail = divmod(length, bt)
        if full > 0:
            total_chunks += full
            for idx, val in enumerate(full_stats):
                totals[idx] += full * val
            if full_stats[0] > 0:
                inter_runs.append((full, full_stats[0]))
            solve_runs.append((full, full_stats[8]))
        if tail > 0:
            total_chunks += 1
            t_stats = _one_chunk(tail)
            for idx, val in enumerate(t_stats):
                totals[idx] += val
            if t_stats[0] > 0:
                inter_runs.append((1, t_stats[0]))
            solve_runs.append((1, t_stats[8]))
    return {
        "total_chunks": total_chunks,
        "total_offdiag_blocks": totals[0],
        "inter_active_chunks": totals[1],
        "inter_rows": totals[2],
        "inter_qb_rows": totals[3],
        "inter_live_elems": totals[4],
        "active_diag_blocks": totals[5],
        "intra_live_elems": totals[6],
        "solve_live_elems": totals[7],
        "solve_dep_rows": totals[8],
        "inter_runs": tuple(inter_runs),
        "solve_runs": tuple(solve_runs),
    }


def _max_owner_serial(
    runs: tuple[tuple[int, int], ...], hv: int, n_owners: int,
) -> int:
    """Return peak serial count across ``n_owners`` AICs under round-robin dispatch."""
    if not runs or n_owners <= 0:
        return 0
    loads = [0] * n_owners
    cursor = 0
    for count, weight in runs:
        total_progs = count * hv
        if total_progs <= 0:
            continue
        full_waves, rem = divmod(total_progs, n_owners)
        if full_waves > 0 and weight > 0:
            delta = full_waves * weight
            for idx in range(n_owners):
                loads[idx] += delta
        if weight > 0:
            for _ in range(rem):
                loads[cursor] += weight
                cursor = (cursor + 1) % n_owners
        else:
            cursor = (cursor + rem) % n_owners
    return max(loads)


def _state_chunk_works(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
    rows: int, tile_bv: int,
) -> tuple[Work, Work, Work, Work]:
    """Return exact per-program-chunk ``(store_h, ws_dot, res_dec, kv_dot)`` sub-works."""
    elem = element_bytes(c.dtype)
    bt = c.bt
    # chunk_delta_h.py:32,78-84 always issues static [BV=64, 64] Cube panels per K-block;
    # boundary_check=(0, 1) masks GM loads/stores to live (rows, tile_bv, dk).
    cube_bv = STATE_TILE_BV
    cube_k = ceil_div(c.dk, STATE_PANEL_BK) * STATE_PANEL_BK
    live_h_bf16 = tile_bv * c.dk * elem
    prog_h_fp32 = cube_bv * cube_k * FP32_BYTES
    live_w_bytes = rows * c.dk * elem
    prog_w_bytes = bt * cube_k * elem
    prog_w_fp32 = bt * cube_k * FP32_BYTES
    live_u_bytes = rows * tile_bv * elem
    prog_u_bytes = bt * cube_bv * elem
    prog_v_fp32 = bt * cube_bv * FP32_BYTES
    # gk is produced by chunk_local_cumsum in FP32 (cumsum.py:201).
    prog_gk_bytes = c.dk * FP32_BYTES
    prog_dot_flops = 2 * bt * cube_bv * cube_k
    cv_mem = "l2" if "l2" in ctx.geometry.rates else "hbm"
    ws_cv_bytes = 2 * (prog_w_fp32 + prog_h_fp32 + prog_v_fp32)
    kv_cv_bytes = 2 * (prog_w_bytes + prog_u_bytes + prog_h_fp32)
    w_store_h = transfer(ctx, write=live_h_bf16) + Work(resources={"vector": 2 * tile_bv * c.dk})
    w_ws_dot = transfer(ctx, read=live_w_bytes) + Work(
        {"fp32": prog_dot_flops},
        {"mte2_aic": prog_w_fp32 + prog_h_fp32, "mte1": prog_w_fp32 + prog_h_fp32,
         "fixpipe": prog_v_fp32, "mte3": prog_w_fp32 + prog_h_fp32, "mte2_aiv": prog_v_fp32,
         "vector": STATE_MASK_CAST_PASSES * bt * cube_k, cv_mem: ws_cv_bytes},
    )
    w_res_dec = transfer(ctx, read=live_u_bytes + prog_gk_bytes, write=live_u_bytes) + Work(
        resources={"vector": STATE_MASK_CAST_PASSES * cube_bv * cube_k + STATE_VEC_U_PASSES * bt * cube_bv + c.dk},
    )
    w_kv_dot = transfer(ctx, read=live_w_bytes) + Work(
        {c.dtype: prog_dot_flops},
        {"mte2_aic": prog_w_bytes + prog_u_bytes, "mte1": prog_w_bytes + prog_u_bytes,
         "fixpipe": prog_h_fp32, "mte2_aiv": prog_h_fp32,
         "vector": STATE_MASK_CAST_PASSES * bt * cube_k + cube_bv * cube_k,
         cv_mem: kv_cv_bytes},
    )
    return w_store_h, w_ws_dot, w_res_dec, w_kv_dot


def _assign_state_programs(
    c: LinearAttentionContract, n_owners: int,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Assign ``(length, tile_bv)`` programs round-robin and group consecutive identical runs."""
    v_widths = tuple(
        min(STATE_TILE_BV, c.dv - offset) for offset in range(0, c.dv, STATE_TILE_BV)
    )
    raw: list[list[tuple[int, int]]] = [[] for _ in range(n_owners)]
    idx = 0
    for length in c.lengths:
        for _ in range(c.hv):
            for tile_bv in v_widths:
                raw[idx % n_owners].append((length, tile_bv))
                idx += 1
    grouped: list[tuple[tuple[int, int, int], ...]] = []
    for progs in raw:
        runs: list[tuple[int, int, int]] = []
        for length, tile_bv in progs:
            if runs and runs[-1][1] == length and runs[-1][2] == tile_bv:
                runs[-1] = (runs[-1][0] + 1, length, tile_bv)
            else:
                runs.append((1, length, tile_bv))
        grouped.append(tuple(runs))
    return tuple(grouped)


def _qk_l2norm_works(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, total_tokens: int,
) -> dict[str, Work]:
    """Return causal Q and K tiled L2-norm works (kda.py:1402-1404, l2norm.py:38-84)."""
    elem = element_bytes(c.dtype)
    bd = _next_power_of_2(c.dk)
    needs_cast = c.dtype.lower() not in FP32_DTYPES
    load_mult = L2NORM_TILED_LOAD_ALU_OPS + int(needs_cast)
    store_mult = L2NORM_TILED_STORE_ALU_OPS + int(needs_cast)
    works: dict[str, Work] = {}
    for prefix, rows in (("q", total_tokens * c.hq), ("k", total_tokens * c.hk)):
        progs = ceil_div(rows, L2NORM_TILED_MBLOCK)
        padded_rows = progs * L2NORM_TILED_MBLOCK
        nbytes = rows * c.dk * elem
        works[f"{prefix}_load"] = transfer(ctx, read=nbytes) + Work(
            resources={"vector": padded_rows * bd * load_mult},
        )
        works[f"{prefix}_red"] = Work(resources={"vector": padded_rows * bd})
        works[f"{prefix}_sfu"] = Work(resources={"vector": padded_rows})
        works[f"{prefix}_store"] = transfer(ctx, write=nbytes) + Work(
            resources={"vector": padded_rows * bd * store_mult},
        )
    return works


def _prepare_kernel_works(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
    total_tokens: int, stats: dict[str, object],
) -> dict[str, Work]:
    """Partition pre-recurrence work cleanly across the sequential pre-state kernels."""
    elem = element_bytes(c.dtype)
    gate_elem = element_bytes(c.gate_dtype)
    beta_elem = element_bytes(c.beta_dtype)
    bt = c.bt
    bc = min(TRITON_DIAG_BLOCK, bt)
    k_panels = ceil_div(c.dk, STATE_PANEL_BK)
    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    head_chunks = int(stats["total_chunks"]) * c.hv
    token_heads = total_tokens * c.hv
    g_in_bytes = token_heads * c.dk * gate_elem
    # chunk_local_cumsum always outputs FP32 (cumsum.py:201, 213, 242).
    g_fp32_bytes = token_heads * c.dk * FP32_BYTES
    qk_bytes = token_heads * c.dk * elem
    v_bytes = token_heads * c.dv * elem
    beta_bytes = token_heads * beta_elem
    # A, Aqk, Ai are allocated as (B, T, H, BT) in live token rows (kda.py:914-915, solve_tril.py:384).
    score_fp32 = token_heads * bt * FP32_BYTES
    score_bf16 = token_heads * bt * elem
    cv_mem = "l2" if "l2" in ctx.geometry.rates else "hbm"
    burst = _hbm_burst_granule(ctx)
    # Strided 64B sub-block writes across H*BT*4 stride amplify HBM write bursts on Arch22.
    row_stride = c.hv * bt * FP32_BYTES
    sub_row_bytes = bc * FP32_BYTES
    strided_burst_amp = max(1, min(burst, row_stride) // max(1, sub_row_bytes))

    # 1. chunk_local_cumsum_vector_kernel (cumsum.py:122): BS=32, 1 FP32 dot per program, FP32 output.
    cumsum_progs = ceil_div(c.dk, CUMSUM_BLOCK_BS) * head_chunks
    cumsum_ws = cumsum_progs * (bt * bt + 2 * bt * CUMSUM_BLOCK_BS) * FP32_BYTES
    w_cumsum = transfer(ctx, read=g_in_bytes, write=g_fp32_bytes) + Work(
        {"fp32": head_chunks * 2 * bt * bt * c.dk},
        {"mte2_aic": cumsum_progs * (bt * bt + bt * CUMSUM_BLOCK_BS) * FP32_BYTES,
         "mte1": cumsum_progs * (bt * bt + bt * CUMSUM_BLOCK_BS) * FP32_BYTES,
         "fixpipe": g_fp32_bytes, "mte3": cumsum_progs * bt * bt * FP32_BYTES,
         "vector": cumsum_progs * bt * bt, cv_mem: cumsum_ws},
    )

    # 2. Pre-inter eager ops: g * RCP_LN2 (FP32) + zeros(A) + zeros(Aqk) (kda.py:1327, 914-915).
    w_eager = transfer(
        ctx, read=g_fp32_bytes, write=g_fp32_bytes + ZERO_INIT_FP32_MATRICES * score_fp32,
    ) + Work(resources={"vector": token_heads * c.dk})

    # 3. chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_inter (kda.py:631):
    # Off-diag (i_i, i_j) pairs for chunks with rows > BC; i_i >= 1 loads q/beta on rows - BC only.
    offdiag_head_pairs = c.hv * int(stats["total_offdiag_blocks"])
    offdiag_tile = bc * bc
    inter_out_tile = offdiag_tile * FP32_BYTES
    if offdiag_head_pairs > 0:
        inter_fp32_flops = offdiag_head_pairs * 2 * 2 * offdiag_tile * c.dk
        inter_k_tile = bc * min(c.dk, STATE_PANEL_BK) * FP32_BYTES
        inter_ws = offdiag_head_pairs * k_panels * (
            2 * (INTER_AIV_WRITTEN_K_TILES * inter_k_tile) + 2 * (2 * inter_out_tile)
        )
        inter_alu = offdiag_head_pairs * INTER_PAIR_ALU_MULT * bc * c.dk
        inter_exp2 = offdiag_head_pairs * INTER_PAIR_EXP2_MULT * bc * c.dk
        inter_read = (
            c.hv * int(stats["inter_qb_rows"]) * (c.dk * elem + beta_elem)
            + c.hv * int(stats["inter_rows"]) * (c.dk * elem + c.dk * FP32_BYTES)
        )
        inter_a_write = ZERO_INIT_FP32_MATRICES * c.hv * int(stats["inter_live_elems"]) * FP32_BYTES
        w_inter_io = transfer(
            ctx, read=inter_read, write=inter_a_write * min(2, strided_burst_amp),
        ) + Work(
            {"fp32": inter_fp32_flops},
            {"mte2_aic": offdiag_head_pairs * k_panels * INTER_AIC_READ_K_TILES * inter_k_tile,
             "mte1": offdiag_head_pairs * k_panels * INTER_AIC_READ_K_TILES * inter_k_tile,
             "fixpipe": offdiag_head_pairs * k_panels * 2 * inter_out_tile,
             "mte3": offdiag_head_pairs * k_panels * INTER_AIV_WRITTEN_K_TILES * inter_k_tile,
             "mte2_aiv": offdiag_head_pairs * k_panels * 2 * inter_out_tile,
             "vector": inter_alu, cv_mem: inter_ws},
        )
    else:
        inter_exp2 = 0
        w_inter_io = Work()

    # 4. chunk_kda_scaled_dot_kkt_fwd_kernel_intra_sub_intra (kda.py:782): pure AIV.
    # kda.py:815-816 early-returns inactive 16-row blocks; BK = max(next_pow2(K), 16) (kda.py:913).
    intra_bk = max(_next_power_of_2(c.dk), TRITON_DIAG_BLOCK)
    active_diag_blocks = c.hv * int(stats["active_diag_blocks"])
    intra_alu = active_diag_blocks * bc * intra_bk + token_heads * INTRA_STEP_ALU_MULT * bc * intra_bk
    intra_exp2 = token_heads * bc * intra_bk
    intra_reduce = token_heads * 2 * bc * intra_bk
    intra_a_write = ZERO_INIT_FP32_MATRICES * c.hv * int(stats["intra_live_elems"]) * FP32_BYTES
    w_intra_io = transfer(
        ctx, read=2 * qk_bytes + g_fp32_bytes + beta_bytes,
        write=intra_a_write * strided_burst_amp,
    ) + Work(resources={"vector": intra_alu})

    # 5. solve_tril_kda (solve_tril.py:175, 384): zeros_like(A) before merge kernel;
    # kernel reads/stores only the 10 lower-triangular 16x16 blocks with boundary_check.
    w_solve_zero_ai = transfer(ctx, write=score_bf16)
    total_solve_dep_rows = c.hv * int(stats["solve_dep_rows"])
    solve_fp32_flops = head_chunks * TRITON_INV_MERGE_DOTS * 2 * offdiag_tile * bc
    solve_ws = head_chunks * TRITON_INV_MERGE_DOTS * (2 * DOT_WS_TENSORS_PER_MM * inter_out_tile)
    solve_alu = total_solve_dep_rows * SOLVE_TRIL_ROW_OPS_MULT * offdiag_tile
    solve_red = total_solve_dep_rows * offdiag_tile
    solve_live_elems = c.hv * int(stats["solve_live_elems"])
    w_solve_io = transfer(
        ctx, read=solve_live_elems * FP32_BYTES, write=solve_live_elems * elem,
    ) + Work(
        {"fp32": solve_fp32_flops},
        {"mte2_aic": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "mte1": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "fixpipe": head_chunks * TRITON_INV_MERGE_DOTS * inter_out_tile,
         "mte3": head_chunks * TRITON_INV_MERGE_DOTS * 2 * inter_out_tile,
         "mte2_aiv": head_chunks * TRITON_INV_MERGE_DOTS * inter_out_tile,
         "vector": solve_alu, cv_mem: solve_ws},
    )

    # 6. recompute_w_u_fwd_kernel (kda.py:968): Cube dots issue full [BT,BT]@[BT,BV/BK] tiles.
    recomp_k = k_panels * STATE_PANEL_BK
    recomp_v = v_tiles * STATE_TILE_BV
    recomp_bf16_flops = head_chunks * 2 * bt * bt * (recomp_k + recomp_v)
    recomp_a_tile = head_chunks * bt * bt * elem
    recomp_b_tiles = head_chunks * bt * (recomp_v + recomp_k) * elem
    recomp_dot_out = head_chunks * bt * (recomp_v + recomp_k) * FP32_BYTES
    recomp_ws = head_chunks * (v_tiles + k_panels) * bt * bt * elem + 2 * recomp_dot_out
    recomp_alu = RECOMPUTE_GATE_ALU_MULT * token_heads * c.dk + token_heads * c.dv
    recomp_exp2 = RECOMPUTE_GATE_EXP2_MULT * token_heads * c.dk
    w_recomp_io = transfer(
        ctx, read=qk_bytes + v_bytes + beta_bytes + score_bf16 + g_fp32_bytes,
        write=2 * qk_bytes + v_bytes,
    ) + Work(
        {c.dtype: recomp_bf16_flops},
        {"mte2_aic": recomp_a_tile + recomp_b_tiles, "mte1": recomp_a_tile + recomp_b_tiles,
         "fixpipe": recomp_dot_out, "mte3": recomp_b_tiles, "mte2_aiv": recomp_dot_out,
         "vector": recomp_alu, cv_mem: recomp_ws},
    )

    return {
        "cumsum": w_cumsum,
        "eager": w_eager,
        "inter_io": w_inter_io,
        "inter_exp2": Work(resources={"vector": inter_exp2}) if inter_exp2 else Work(),
        "intra_io": w_intra_io,
        "intra_exp2": Work(resources={"vector": intra_exp2}),
        "intra_red": Work(resources={"vector": intra_reduce}),
        "solve_zero_ai": w_solve_zero_ai,
        "solve_io": w_solve_io,
        "solve_red": Work(resources={"vector": solve_red}) if solve_red else Work(),
        "recomp_io": w_recomp_io,
        "recomp_exp2": Work(resources={"vector": recomp_exp2}),
    }


def _lower_qk_l2norm_stages(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    all_aivs: tuple[int, ...], l2_works: dict[str, Work],
    total_tokens: int, reduce_ratio: float, sfu_ratio: float,
) -> tuple[list[Stage], str]:
    """Lower Q and K tiled L2-norm kernels in causal load -> reduce -> rsqrt -> store order."""
    stages: list[Stage] = []
    prev = ""
    for prefix, rows in (("q", total_tokens * c.hq), ("k", total_tokens * c.hk)):
        progs = ceil_div(rows, L2NORM_TILED_MBLOCK)
        aivs = all_aivs[:min(len(all_aivs), max(1, progs))]
        after = (prev,) if prev else ()
        s_load = bind_phase(Phase(f"kda_{prefix}_l2norm_load", l2_works[f"{prefix}_load"], after), rc, ctx, (), aivs)
        s_red = _cap_vec_stage(
            bind_phase(Phase(f"kda_{prefix}_l2norm_reduce", l2_works[f"{prefix}_red"], (s_load.name,)), rc, ctx, (), aivs),
            rc, reduce_ratio,
        )
        s_sfu = _cap_vec_stage(
            bind_phase(Phase(f"kda_{prefix}_l2norm_sfu", l2_works[f"{prefix}_sfu"], (s_red.name,)), rc, ctx, (), aivs),
            rc, sfu_ratio,
        )
        s_store = bind_phase(
            Phase(f"kda_{prefix}_l2norm_store", l2_works[f"{prefix}_store"], (s_sfu.name,)), rc, ctx, (), aivs,
        )
        stages.extend((s_load, s_red, s_sfu, s_store))
        prev = s_store.name
    return stages, prev


def _lower_state_program(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    length: int, tile_bv: int,
    b_aics: tuple[int, ...], b_aivs: tuple[int, ...],
    owner_cap: dict[str, float], cv_sync_per_step_ms: float,
    chunk_cache: dict[tuple[int, int], tuple[Work, Work, Work, Work]],
) -> KernelProgram:
    """Lower one ``chunk_delta_h`` program ``init_load -> chunk loop -> final_store``."""
    state_elem = element_bytes(c.state_dtype)
    bt = c.bt
    full, tail = divmod(length, bt)

    def _cap(st: Stage) -> Stage:
        applied = {k: v for k, v in owner_cap.items() if k in st.work}
        return replace(st, caps={**st.caps, **applied}) if applied else st

    def _chunk_stages(rows: int) -> tuple[Stage, ...]:
        key = (rows, tile_bv)
        if key not in chunk_cache:
            chunk_cache[key] = _state_chunk_works(c, ctx, rows, tile_bv)
        w0, w1, w2, w3 = chunk_cache[key]
        s0 = _cap(bind_phase(Phase("h_store_enter", w0), rc, ctx, b_aics, b_aivs))
        s1 = _cap(bind_phase(Phase("h_ws_fp32_dot", w1, ("h_store_enter",)), rc, ctx, b_aics, b_aivs))
        s2 = _cap(bind_phase(Phase("h_res_decay", w2, ("h_ws_fp32_dot",)), rc, ctx, b_aics, b_aivs))
        s3 = _cap(bind_phase(Phase("h_kv_bf16_dot", w3, ("h_res_decay",)), rc, ctx, b_aics, b_aivs))
        if cv_sync_per_step_ms > 0.0:
            return (s0, s1, s2, s3, _fixed_stage("h_chunk_cv_sync", cv_sync_per_step_ms, after=("h_kv_bf16_dot",)))
        return (s0, s1, s2, s3)

    stages: list[Stage] = []
    prev: tuple[str, ...] = ()
    if c.use_initial_state:
        init_w = transfer(ctx, read=tile_bv * c.dk * state_elem)
        s_init = bind_phase(Phase("state_init_load", init_w), rc, ctx, b_aics, b_aivs)
        stages.append(s_init)
        prev = (s_init.name,)
    if full > 0:
        body = (
            KernelProgram(_chunk_stages(bt))
            if full == 1
            else compact_periodic_pipeline(
                rc, full, 1, lambda _i: _chunk_stages(bt),
                head=COMPACT_WINDOW, tail=COMPACT_WINDOW, threshold=COMPACT_THRESHOLD,
            )
        )
        st_full = Stage("state_chunks_full", after=prev, program=body)
        stages.append(st_full)
        prev = (st_full.name,)
    if tail > 0:
        st_tail = Stage("state_chunks_tail", after=prev, program=KernelProgram(_chunk_stages(tail)))
        stages.append(st_tail)
        prev = (st_tail.name,)
    if c.output_final_state:
        # chunk_delta_h.py:302 always allocates final_state as torch.float32.
        final_w = transfer(ctx, write=tile_bv * c.dk * FP32_BYTES)
        stages.append(bind_phase(Phase("state_final_store", final_w, prev), rc, ctx, b_aics, b_aivs))
    return KernelProgram(tuple(stages))


def _lower_state_owners(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    active_state_pairs: tuple[tuple[int, tuple[int, ...]], ...],
    owner_runs: tuple[tuple[tuple[int, int, int], ...], ...],
    cv_sync_per_step_ms: float, prev: str,
) -> Stage:
    """Lower per-owner ``chunk_delta_h`` schedules preserving physical owner identity and epilogue order."""
    n_owners = len(active_state_pairs)
    group_cap = p01_cold_hbm_group_cap(rc, n_owners, include_l2=False)
    owner_cap = {k: v / float(n_owners) for k, v in group_cap.items()}
    chunk_cache: dict[tuple[int, int], tuple[Work, Work, Work, Work]] = {}
    owner_stages: list[Stage] = []
    for owner_idx, ((aic, peers), runs) in enumerate(zip(active_state_pairs, owner_runs)):
        if not runs:
            continue
        b_aics = (aic,)
        b_aivs = (peers[0],)
        seq_stages: list[Stage] = []
        seq_prev: tuple[str, ...] = ()
        for run_idx, (count, length, tile_bv) in enumerate(runs):
            prog = _lower_state_program(
                c, ctx, rc, length, tile_bv, b_aics, b_aivs,
                owner_cap, cv_sync_per_step_ms, chunk_cache,
            )
            run_prog = (
                prog if count == 1
                else compact_periodic_pipeline(
                    rc, count, 1, lambda _i, p=prog: (Stage("prog", program=p),),
                    head=COMPACT_WINDOW, tail=COMPACT_WINDOW, threshold=COMPACT_THRESHOLD,
                )
            )
            run_st = Stage(f"prog_{run_idx}", after=seq_prev, program=run_prog)
            seq_stages.append(run_st)
            seq_prev = (run_st.name,)
        owner_stages.append(Stage(f"state_owner_{owner_idx}", program=KernelProgram(tuple(seq_stages))))
    return Stage("triton_kda_state_recurrence", after=(prev,), program=KernelProgram(tuple(owner_stages)))


def _lower_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    pairs: tuple[tuple[int, tuple[int, ...]], ...],
    all_aivs: tuple[int, ...],
    active_state_pairs: tuple[tuple[int, tuple[int, ...]], ...],
    l2_works: dict[str, Work] | None, prep_works: dict[str, Work],
    total_tokens: int, stats: dict[str, object],
    owner_runs: tuple[tuple[tuple[int, int, int], ...], ...],
    out_phase: Phase,
) -> KernelProgram:
    """Lower all Triton chunk KDA kernels + eager ops with exact per-grid placement & conservation."""
    reduce_ratio = _reduce_rate_ratio(rc)
    sfu_ratio = _sfu_rate_ratio(rc)
    cv_ms = p13_paired_ms(rc)
    fix_ms = p03_fixpipe_setup_ms(rc)
    k_panels = ceil_div(c.dk, STATE_PANEL_BK)
    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    head_chunks = int(stats["total_chunks"]) * c.hv
    stages: list[Stage] = []

    def _push(st: Stage) -> str:
        stages.append(st)
        return st.name

    prev_after: tuple[str, ...] = ()
    if l2_works is not None:
        l2_stages, l2_last = _lower_qk_l2norm_stages(
            c, ctx, rc, all_aivs, l2_works, total_tokens, reduce_ratio, sfu_ratio,
        )
        stages.extend(l2_stages)
        prev_after = (l2_last,)

    # 1. chunk_local_cumsum_vector_kernel: grid=(cdiv(K, BS=32), NT, B*H)
    cumsum_progs = ceil_div(c.dk, CUMSUM_BLOCK_BS) * head_chunks
    cumsum_aics, cumsum_aivs = _pair_owners(pairs, cumsum_progs)
    prev = _push(bind_phase(Phase("kda_cumsum", prep_works["cumsum"], prev_after), rc, ctx, cumsum_aics, cumsum_aivs))
    cumsum_sync = ceil_div(cumsum_progs, max(1, len(cumsum_aics))) * (cv_ms + fix_ms)
    if cumsum_sync > 0.0:
        prev = _push(_fixed_stage("kda_cumsum_cv", cumsum_sync, after=(prev,)))

    # 2. Pre-inter eager ops: g * RCP_LN2 + zeros(A) + zeros(Aqk)
    prev = _push(bind_phase(Phase("kda_eager_scale_zeros", prep_works["eager"], (prev,)), rc, ctx, (), all_aivs))

    # 3. intra_sub_inter: grid=(NT, B*H); only chunks with rows > BC do work
    inter_active_progs = c.hv * int(stats["inter_active_chunks"])
    inter_aics, inter_aivs = _pair_owners(pairs, inter_active_progs)
    prev = _push(bind_phase(Phase("kda_inter_io", prep_works["inter_io"], (prev,)), rc, ctx, inter_aics, inter_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_inter_exp2", prep_works["inter_exp2"], (prev,)), rc, ctx, (), inter_aivs),
        rc, sfu_ratio,
    ))
    if int(stats["total_offdiag_blocks"]) > 0:
        inter_peak_blocks = _max_owner_serial(stats["inter_runs"], c.hv, len(inter_aics))
        inter_sync = inter_peak_blocks * k_panels * 2 * (cv_ms + fix_ms)
        if inter_sync > 0.0:
            prev = _push(_fixed_stage("kda_inter_cv", inter_sync, after=(prev,)))

    # 4. intra_sub_intra: grid=(NT, NC, B*H); pure AIV capped to active diag blocks
    active_diag_blocks = c.hv * int(stats["active_diag_blocks"])
    intra_aivs = all_aivs[:min(len(all_aivs), max(1, active_diag_blocks))]
    prev = _push(bind_phase(Phase("kda_intra_io_alu", prep_works["intra_io"], (prev,)), rc, ctx, (), intra_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_intra_exp2", prep_works["intra_exp2"], (prev,)), rc, ctx, (), intra_aivs),
        rc, sfu_ratio,
    ))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_intra_reduce", prep_works["intra_red"], (prev,)), rc, ctx, (), intra_aivs),
        rc, reduce_ratio,
    ))

    # 5. solve_tril: eager zeros_like(A) for Ai, then merge kernel grid=(NT, B*H)
    prev = _push(bind_phase(Phase("kda_solve_zero_ai", prep_works["solve_zero_ai"], (prev,)), rc, ctx, (), all_aivs))
    solve_aics, solve_aivs = _pair_owners(pairs, head_chunks)
    prev = _push(bind_phase(Phase("kda_solve_io_alu", prep_works["solve_io"], (prev,)), rc, ctx, solve_aics, solve_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_solve_reduce", prep_works["solve_red"], (prev,)), rc, ctx, (), solve_aivs),
        rc, reduce_ratio,
    ))
    solve_waves = ceil_div(head_chunks, max(1, len(solve_aics)))
    solve_peak_dep_rows = _max_owner_serial(stats["solve_runs"], c.hv, len(solve_aics))
    solve_sync = solve_waves * TRITON_INV_MERGE_DOTS * (cv_ms + fix_ms) + solve_peak_dep_rows * p12_flag_ms(rc)
    if solve_sync > 0.0:
        prev = _push(_fixed_stage("kda_solve_cv", solve_sync, after=(prev,)))

    # 6. recompute_w_u: grid=(NT, B*H)
    prev = _push(bind_phase(Phase("kda_recomp_io_alu", prep_works["recomp_io"], (prev,)), rc, ctx, solve_aics, solve_aivs))
    prev = _push(_cap_vec_stage(
        bind_phase(Phase("kda_recomp_exp2", prep_works["recomp_exp2"], (prev,)), rc, ctx, (), solve_aivs),
        rc, sfu_ratio,
    ))
    recomp_sync = solve_waves * (v_tiles + k_panels) * (cv_ms + fix_ms)
    if recomp_sync > 0.0:
        prev = _push(_fixed_stage("kda_recomp_cv", recomp_sync, after=(prev,)))

    # 7. chunk_delta_h: per-owner sequential programs (init -> chunk recurrence -> FP32 final store)
    cv_sync_per_step_ms = (2 * k_panels) * (CV_ROUNDTRIP_PAIRED_SYNCS * cv_ms + fix_ms)
    prev = _push(_lower_state_owners(c, ctx, rc, active_state_pairs, owner_runs, cv_sync_per_step_ms, prev))

    # 8. chunk_gla_fwd_kernel_o: grid=(cdiv(V, BV=128), NT, B*H)
    out_progs = ceil_div(c.dv, OUT_TILE_BV) * head_chunks
    out_aics, out_aivs = _pair_owners(pairs, out_progs)
    prev = _push(bind_phase(Phase(out_phase.name, out_phase.work, (prev,)), rc, ctx, out_aics, out_aivs))
    out_sync = ceil_div(out_progs, max(1, len(out_aics))) * TRITON_OUT_CV_HANDOFFS * (cv_ms + fix_ms)
    if out_sync > 0.0:
        _push(_fixed_stage("kda_out_cv", out_sync, after=(prev,)))
    return KernelProgram(tuple(stages))


def build_triton_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower vllm_ascend.ops.triton.kda.kda.chunk_kda (7 kernels + 4 eager ops + optional Q/K L2-norm)."""
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
    bt = c.bt
    total_tokens = sum(c.lengths)
    stats = _chunk_live_stats(c)
    total_chunks = int(stats["total_chunks"])
    head_chunks = total_chunks * c.hv
    token_heads = total_tokens * c.hv
    use_qk_l2norm = bool(c.metadata.get("use_qk_l2norm_in_kernel", False))
    l2_works = _qk_l2norm_works(c, ctx, total_tokens) if use_qk_l2norm else None

    prep_works = _prepare_kernel_works(c, ctx, total_tokens, stats)
    cumsum_after = ("triton_kda_qk_l2norm",) if use_qk_l2norm else ()
    cumsum_phase = Phase("triton_kda_cumsum", prep_works["cumsum"], cumsum_after)
    prep_body_work = (
        prep_works["eager"] + prep_works["inter_io"] + prep_works["inter_exp2"]
        + prep_works["intra_io"] + prep_works["intra_exp2"] + prep_works["intra_red"]
        + prep_works["solve_zero_ai"] + prep_works["solve_io"] + prep_works["solve_red"]
        + prep_works["recomp_io"] + prep_works["recomp_exp2"]
    )
    pre_phase = Phase("triton_kda_prepare_alu", prep_body_work, ("triton_kda_cumsum",))

    v_tiles = ceil_div(c.dv, STATE_TILE_BV)
    state_progs = v_tiles * len(c.lengths) * c.hv
    active_state_pairs = pairs[:min(len(pairs), state_progs)]
    owner_runs = _assign_state_programs(c, len(active_state_pairs))
    chunk_cache: dict[tuple[int, int], tuple[Work, Work, Work, Work]] = {}
    state_total_work = Work()
    for runs in owner_runs:
        for count, length, tile_bv in runs:
            prog_w = Work()
            if c.use_initial_state:
                prog_w += transfer(ctx, read=tile_bv * c.dk * state_elem)
            full, tail = divmod(length, bt)
            if full > 0:
                if (bt, tile_bv) not in chunk_cache:
                    chunk_cache[(bt, tile_bv)] = _state_chunk_works(c, ctx, bt, tile_bv)
                w0, w1, w2, w3 = chunk_cache[(bt, tile_bv)]
                prog_w += (w0 + w1 + w2 + w3).scaled(full)
            if tail > 0:
                if (tail, tile_bv) not in chunk_cache:
                    chunk_cache[(tail, tile_bv)] = _state_chunk_works(c, ctx, tail, tile_bv)
                w0, w1, w2, w3 = chunk_cache[(tail, tile_bv)]
                prog_w += w0 + w1 + w2 + w3
            if c.output_final_state:
                prog_w += transfer(ctx, write=tile_bv * c.dk * FP32_BYTES)
            state_total_work += prog_w.scaled(count)
    state_phase = Phase(
        "triton_kda_state_recurrence",
        state_total_work,
        ("triton_kda_prepare_alu",),
    )

    g_fp32_bytes = token_heads * c.dk * FP32_BYTES
    score_fp32 = token_heads * bt * FP32_BYTES
    score_bf16 = token_heads * bt * elem
    entering_bank_bytes = head_chunks * c.dk * c.dv * elem
    out_v_tiles = ceil_div(c.dv, OUT_TILE_BV)
    out_shared_read = token_heads * c.dk * elem + g_fp32_bytes + score_fp32
    out_private_read = token_heads * c.dv * elem + entering_bank_bytes
    out_mte2_aiv = out_v_tiles * out_shared_read + out_private_read
    out_l2_hits = (out_v_tiles - 1) * out_shared_read
    out_write = token_heads * c.dv * elem
    out_cube_flops = head_chunks * 2 * (bt * c.dk * c.dv + bt * bt * c.dv)
    out_phase = Phase(
        "triton_kda_output_o",
        transfer(ctx, read=out_mte2_aiv, write=out_write, hits=out_l2_hits) + Work(
            {c.dtype: out_cube_flops},
            {"mte1": token_heads * (c.dk + c.dv) * elem + entering_bank_bytes + score_bf16,
             "fixpipe": out_write, "vector": 2 * token_heads * c.dk},
        ),
        ("triton_kda_state_recurrence",),
    )

    totals: list[Phase] = []
    if l2_works is not None:
        l2_total = (
            l2_works["q_load"] + l2_works["q_red"] + l2_works["q_sfu"] + l2_works["q_store"]
            + l2_works["k_load"] + l2_works["k_red"] + l2_works["k_sfu"] + l2_works["k_store"]
        )
        totals.append(Phase("triton_kda_qk_l2norm", l2_total))
    totals.extend((cumsum_phase, pre_phase, state_phase, out_phase))
    launch_count = TRITON_CHUNK_LAUNCH_COUNT + (L2NORM_VECTORS if use_qk_l2norm else 0)

    def lower(rc: RecipeContext) -> KernelProgram:
        return _lower_chunk_kda(
            c, ctx, rc, pairs, aivs, active_state_pairs, l2_works, prep_works,
            total_tokens, stats, owner_runs, out_phase,
        )

    return make_plan(
        ctx, TRITON_CHUNK_KDA_ABI, tuple(totals), sources=TRITON_KDA_SOURCES,
        launch_count=launch_count, program=lower,
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
    beta_prep_works: tuple[Work, Work, Work, Work] | None = None,
    beta_elems: int = 0,
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
        rec_prog = _group_prog(aivs[:active_aivs], total_progs, "g0_")
    else:
        # Two independent AIV halves keep T=1 load->vec->store serial per AIV while staggering HBM.
        mid = active_aivs // REC_ASYNC_GROUPS
        a_aivs, b_aivs = aivs[:mid], aivs[mid:active_aivs]
        a_progs = (total_progs * len(a_aivs)) // active_aivs
        b_progs = total_progs - a_progs
        a_head = _block_stages(len(a_aivs), a_aivs, "ga0_")
        a_rem_progs = a_progs - len(a_aivs)
        if a_rem_progs <= 0:
            rec_prog = KernelProgram((
                Stage("triton_rec_half_a", program=KernelProgram(a_head)),
                Stage("triton_rec_half_b", program=_group_prog(b_aivs, b_progs, "gb_")),
            ))
        else:
            rec_prog = KernelProgram((
                *a_head,
                Stage("triton_rec_half_a_rest", after=(a_head[-1].name,), program=_group_prog(a_aivs, a_rem_progs, "ga_")),
                Stage("triton_rec_half_b", after=(a_head[0].name,), program=_group_prog(b_aivs, b_progs, "gb_")),
            ))

    if beta_prep_works is None:
        return rec_prog
    beta_cast_w, beta_sig_load_w, beta_sig_sfu_w, beta_sig_store_w = beta_prep_works
    beta_aivs = aivs[:min(len(aivs), max(1, ceil_div(beta_elems, REC_SIMD_WIDTH)))]
    pre_stages: list[Stage] = []
    beta_after: tuple[str, ...] = ()
    if beta_cast_w.resources:
        s_cast = bind_phase(Phase("rec_beta_cast", beta_cast_w), rc, ctx, (), beta_aivs)
        pre_stages.append(s_cast)
        beta_after = (s_cast.name,)
    s_sig_load = bind_phase(Phase("rec_beta_sigmoid_load", beta_sig_load_w, beta_after), rc, ctx, (), beta_aivs)
    s_sig_sfu = _cap_vec_stage(
        bind_phase(Phase("rec_beta_sigmoid_sfu", beta_sig_sfu_w, (s_sig_load.name,)), rc, ctx, (), beta_aivs),
        rc, sfu_ratio,
    )
    s_sig_store = bind_phase(
        Phase("rec_beta_sigmoid_store", beta_sig_store_w, (s_sig_sfu.name,)), rc, ctx, (), beta_aivs,
    )
    pre_stages.extend((s_sig_load, s_sig_sfu, s_sig_store))
    return KernelProgram((*pre_stages, Stage("triton_rec_kernel", after=(s_sig_store.name,), program=rec_prog)))


def _resolve_triton_recurrent_state_counts(c: LinearAttentionContract) -> tuple[int, int, int]:
    """Return ``(loaded_initial_seqs, active_tokens, stored_checkpoint_tokens)`` per fused_recurrent_kda.py."""
    n_seqs = len(c.lengths)
    total_tokens = sum(c.lengths)
    inplace_final_state = bool(c.metadata.get("inplace_final_state", True))
    raw_indices = c.metadata.get("ssm_state_indices")
    if not isinstance(raw_indices, (tuple, list)):
        return (n_seqs if c.use_initial_state else 0), total_tokens, total_tokens
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
    active_tokens = 0
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
        active_tokens += seq_len
        if inplace_final_state:
            stored_tokens += sum(1 for v in row[:seq_len] if v > 0)
        else:
            stored_tokens += seq_len
    return loaded_seqs, active_tokens, stored_tokens


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
    sigmoid_beta = bool(c.metadata.get("sigmoid_beta", False))
    is_beta_headwise = bool(c.metadata.get("is_beta_headwise", False))
    beta_width = c.dv if is_beta_headwise else 1
    beta_prog_width = bv if is_beta_headwise else 1
    needs_beta_cast = sigmoid_beta and c.beta_dtype.lower() not in FP32_DTYPES
    # kda.py:168-169: eager beta.float().sigmoid() precedes fused_recurrent_kda_fwd.
    effective_beta_elem = FP32_BYTES if sigmoid_beta else beta_elem
    launch_count = 1 + int(needs_beta_cast) + int(sigmoid_beta)

    total_progs = nk * nv * n_seqs * c.hv
    active_aivs = min(len(aivs), total_progs)
    waves = ceil_div(total_progs, active_aivs)
    burst = _hbm_burst_granule(ctx)

    # fused_recurrent_kda.py:99-148: kernel stores every token checkpoint to ht
    # whenever INPLACE_FINAL_STATE=False, or for every positive ssm_state_idx when True,
    # independent of the wrapper return flag output_final_state; state_idx <= 0 early-returns.
    loaded_seqs, active_tokens, stored_tokens = _resolve_triton_recurrent_state_counts(c)

    head_state_bytes = c.dk * c.dv * state_elem
    h0_read_total = loaded_seqs * c.hv * head_state_bytes
    ht_write_total = stored_tokens * c.hv * head_state_bytes
    qkvgb_per_tok_head = (2 * c.dk + c.dv) * elem + c.dk * gate_elem + beta_width * effective_beta_elem
    qkvgb_hbm_total = active_tokens * c.hv * qkvgb_per_tok_head
    # Each program issues 5 separate small vector GM loads (q, k, v, gk, beta) -> min burst per load on MTE2.
    qkvgb_mte2_per_prog = max(
        2 * c.dk * elem + c.dk * gate_elem + beta_prog_width * effective_beta_elem + bv * elem,
        REC_SMALL_GM_LOADS * burst,
    )
    qkvgb_mte2_total = active_tokens * c.hv * nv * qkvgb_mte2_per_prog
    out_write_total = active_tokens * c.hv * c.dv * elem
    ht_mte3_total = ht_write_total + active_tokens * c.hv * nv * burst

    # Exact per-program vector/SFU/reduce op counts grounded in fused_recurrent_kda.py:76-149.
    prog_state = bv * bk
    prologue_alu = total_progs * REC_PROLOGUE_OPS_PER_STATE * prog_state
    l2norm_alu = (active_tokens * c.hv * nv * L2NORM_VECTORS * L2NORM_ALU_MULT * bk) if use_l2norm else 0
    l2norm_red = (active_tokens * c.hv * nv * L2NORM_VECTORS * L2NORM_TREE_STEPS * bk) if use_l2norm else 0
    l2norm_sfu = (active_tokens * c.hv * nv * L2NORM_VECTORS * bk) if use_l2norm else 0
    short_vec_alu = active_tokens * c.hv * nv * REC_SHORT_VEC_OPS * max(bv, REC_SIMD_WIDTH)
    decay_alu = prologue_alu + l2norm_alu + short_vec_alu + active_tokens * c.hv * nv * (REC_DECAY_OPS_PER_STATE * prog_state + bk)
    # fused_recurrent_kda.py:127-128: tl.exp(b_gk[None, :]) evaluates exp on [1, BK] before broadcasting across BV.
    sfu_ops = active_tokens * c.hv * nv * bk + l2norm_sfu
    dot1_red = l2norm_red + active_tokens * c.hv * nv * prog_state
    update_alu = active_tokens * c.hv * nv * (REC_UPDATE_OPS_PER_STATE + REC_EPILOGUE_OPS_PER_STATE) * prog_state
    dot2_red = active_tokens * c.hv * nv * prog_state
    total_vec = decay_alu + sfu_ops + dot1_red + update_alu + dot2_red

    total_hbm_read = h0_read_total + qkvgb_hbm_total
    total_hbm_write = ht_write_total + out_write_total
    total_mte2_aiv = h0_read_total + qkvgb_mte2_total
    total_l2_hits = max(0, qkvgb_mte2_total - qkvgb_hbm_total)
    has_l2_rate = "l2" in ctx.geometry.rates

    beta_elems = total_tokens * c.hv * beta_width
    beta_prep_works: tuple[Work, Work, Work, Work] | None = None
    phase_list: list[Phase] = []
    init_after: tuple[str, ...] = ()
    if sigmoid_beta:
        beta_in_bytes = beta_elems * beta_elem
        beta_fp32_bytes = beta_elems * FP32_BYTES
        beta_cast_w = (
            transfer(ctx, read=beta_in_bytes, write=beta_fp32_bytes) + Work(resources={"vector": beta_elems})
            if needs_beta_cast else Work()
        )
        beta_sig_load_w = (
            transfer(ctx, read=beta_fp32_bytes)
            + Work(resources={"vector": BETA_SIGMOID_LOAD_ALU_OPS * beta_elems})
        )
        beta_sig_sfu_w = Work(resources={"vector": beta_elems})
        beta_sig_store_w = (
            transfer(ctx, write=beta_fp32_bytes)
            + Work(resources={"vector": BETA_SIGMOID_STORE_ALU_OPS * beta_elems})
        )
        beta_prep_works = (beta_cast_w, beta_sig_load_w, beta_sig_sfu_w, beta_sig_store_w)
        phase_list.append(Phase(
            "sigmoid_beta_preprocess",
            beta_cast_w + beta_sig_load_w + beta_sig_sfu_w + beta_sig_store_w,
        ))
        init_after = ("sigmoid_beta_preprocess",)

    phase_list.extend((
        Phase("load_initial_state", Work(resources={
            "hbm": h0_read_total, "mte2_aiv": h0_read_total,
        }), init_after),
        Phase("recurrent_token_loop", Work(resources={
            "hbm": qkvgb_hbm_total + total_hbm_write,
            "mte2_aiv": qkvgb_mte2_total,
            "mte3": ht_mte3_total,
            "vector": total_vec,
            **({"l2": total_l2_hits} if (total_l2_hits and has_l2_rate) else {}),
        }), ("load_initial_state",)),
    ))

    def lower(rc: RecipeContext) -> KernelProgram:
        return _lower_recurrent_kda(
            ctx, rc,
            aivs=aivs, active_aivs=active_aivs, waves=waves, total_progs=total_progs,
            max_q=max_q, total_hbm_read=total_hbm_read, total_hbm_write=total_hbm_write,
            total_mte2_aiv=total_mte2_aiv, total_l2_hits=total_l2_hits, has_l2_rate=has_l2_rate,
            decay_alu=decay_alu, sfu_ops=sfu_ops, dot1_red=dot1_red,
            update_alu=update_alu, dot2_red=dot2_red, ht_mte3_total=ht_mte3_total,
            beta_prep_works=beta_prep_works, beta_elems=beta_elems,
        )

    return make_plan(
        ctx, TRITON_RECURRENT_KDA_ABI, tuple(phase_list), sources=TRITON_RECURRENT_SOURCES,
        launch_count=launch_count, owners=active_aivs, program=lower,
        metadata={
            "template": "triton_recurrent_kda", "bk": bk, "bv": bv,
            "total_programs": total_progs, "active_aivs": active_aivs, "waves": waves,
            "persistent_ub_resident": True,
        },
    )
