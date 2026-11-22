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
from ..sharedkv.l0_sync import p12_flag_ms, p13_paired_ms
from ..sharedkv.split_core import p01_cold_hbm_group_cap

VLLM_ASCEND_REV = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
TRITON_CHUNK_KDA_ABI = "vllm_ascend.ops.triton.kda.kda.chunk_kda"
TRITON_RECURRENT_KDA_ABI = "vllm_ascend.ops.triton.kda.kda.fused_recurrent_kda"
TRITON_KDA_CHUNK_ABIS = frozenset({TRITON_CHUNK_KDA_ABI, "triton_chunk_kda"})
TRITON_KDA_RECURRENT_ABIS = frozenset({TRITON_RECURRENT_KDA_ABI, "triton_recurrent_kda"})

TRITON_KDA_SOURCES: tuple[SourceRef, ...] = (
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
)
TRITON_RECURRENT_SOURCES: tuple[SourceRef, ...] = (
    TRITON_KDA_SOURCES[0],
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision=VLLM_ASCEND_REV,
        relative_path="vllm_ascend/ops/triton/kda/fused_recurrent_kda.py",
        source_sha256="6d18c60c4ef58e8972b7cbec785d3c1325a5b130e628ab27ab08c7b37a9c6728",
    ),
)

# Upstream vllm_ascend/ops/triton/kda tiling & op-count constants (zero fitted timings)
STATE_TILE_BV = 64
STATE_PANEL_BK = 64
RECURRENT_MAX_BV = 8
STATE_VEC_H_PASSES = 4
STATE_VEC_U_PASSES = 3
DIAG_EXP2_PER_HEAD_CHUNK = 131_072
DIAG_REDUCE_ROWS_PER_HEAD_CHUNK = 2_048
TRITON_CHUNK_LAUNCH_COUNT = 9  # 7 Triton kernels + 2 eager tensor ops (RCP_LN2, zero-init)
CV_ROUNDTRIP_PAIRED_SYNCS = 2  # AIV->AIC (operand ready) + AIC->AIV (Fixpipe ready) per tl.dot
REC_FLAGS_PER_TOKEN = 4  # MTE2_V + V_S + PIPE_V + V_MTE3 per recurrent token iteration
ARCH22_REDUCE_DIVISOR = 8.0
ARCH22_SFU_DIVISOR = 4.0
SOLVE_TRIL_OPS_FACTOR = 2
SOLVE_TRIL_DIVISOR = 3
RECOMPUTE_GATE_OPS_MULT = 3
ZERO_INIT_MATRICES = 2
CUMSUM_PASSES = 2
L2NORM_VECTORS = 2
# fused_recurrent_kda.py:98-136: zero+load mask_h (2) + bcast_g/mul (2) + bcast_k/mul (2) = 6;
# bcast_v/bcast_k/fma (3) + bcast_q/mul (2) + store mask (1) = 6.
REC_DECAY_OPS_PER_STATE = 6
REC_UPDATE_OPS_PER_STATE = 6
REC_WAVE_SLOTS = 1  # Triton-Ascend SPMD persistent grid loop executes each program sequentially per AIV.
TRITON_DIAG_BLOCK = 16
TRITON_OFFDIAG_BLOCKS = 6  # C(4, 2) off-diagonal 16x16 blocks in 64x64 chunk
TRITON_INV_MERGE_DOTS = 16  # solve_tril.py:255: 16 IEEE FP32 16x16x16 dots
TRITON_DIAG_ELEM_OPS_MULT = 4  # kda.py:852: 4*m*b^2*K elementwise ops
TRITON_SOLVE_DEP_ROWS = 4 * 14  # solve_tril.py:221: 4 blocks x 14 dependent row updates
MS_PER_S_KDA = 1000.0
COMPACT_THRESHOLD = 4
COMPACT_WINDOW = 2


def _next_power_of_2(value: int) -> int:
    power = 1
    while power < value:
        power <<= 1
    return power


def _reduce_rate_ratio(rc: RecipeContext) -> float:
    """Ratio of vector ALU rate to calibrated P09 reduce rate (or vector_reduce service rate)."""
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        try:
            alu = table.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50
            red = table.cell("P09_vec_reduce_cast").band("vec_reduce_ops_per_s_per_aiv").p50
            if alu and red and red > 0.0:
                return max(1.0, float(alu) / float(red))
        except Exception:
            pass
    vec_key = rc.resource_names.get("vector", "vector")
    vec_rate = float(rc.rates.get(vec_key, rc.rates.get("vector", 0.0)) or 0.0) * MS_PER_S_KDA
    red_rate = rc.service_rate("vector_reduce")
    if vec_rate > 0.0 and red_rate and red_rate > 0.0:
        return max(1.0, vec_rate / float(red_rate))
    return ARCH22_REDUCE_DIVISOR


def _sfu_rate_ratio(rc: RecipeContext) -> float:
    """Ratio of vector ALU rate to calibrated P08 SFU rate when bound."""
    table = getattr(rc, "primitive_table", None)
    if table is not None and hasattr(table, "cell"):
        try:
            alu = table.cell("P07_vec_alu").band("vec_alu_ops_per_s_per_aiv").p50
            sfu = table.cell("P08_vec_sfu").band("vec_sfu_ops_per_s_per_aiv").p50
            if alu and sfu and sfu > 0.0:
                return max(1.0, float(alu) / float(sfu))
        except Exception:
            pass
    return ARCH22_SFU_DIVISOR


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
    """Assign each state program to a core round-robin and return ``(steps, active_count)`` bands.

    Guarantees ``sum(steps * active_count) == sum(ceil_div(L, bt)) * hv * v_tiles``
    for uniform and ragged sequence batches alike.
    """
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
    w_store_h = transfer(ctx, write=prog_h_bf16) + Work(resources={"vector": prog_bv * c.dk})
    w_ws_dot = transfer(ctx, read=prog_w_bytes) + Work(
        {"fp32": prog_dot_flops},
        {"mte2_aic": prog_w_fp32 + prog_h_fp32, "mte1": prog_w_fp32 + prog_h_fp32,
         "fixpipe": prog_v_fp32, "mte3": prog_w_fp32 + prog_h_fp32, "mte2_aiv": prog_v_fp32,
         cv_mem: ws_cv_bytes},
    )
    w_res_dec = transfer(ctx, read=prog_u_bytes + prog_gk_bytes, write=prog_u_bytes) + Work(
        resources={"vector": 2 * prog_bv * c.dk + STATE_VEC_U_PASSES * bt * prog_bv + c.dk},
    )
    w_kv_dot = transfer(ctx, read=prog_w_bytes) + Work(
        {c.dtype: prog_dot_flops},
        {"mte2_aic": prog_w_bytes + prog_u_bytes, "mte1": prog_w_bytes + prog_u_bytes,
         "fixpipe": prog_h_fp32, "mte2_aiv": prog_h_fp32, "vector": prog_bv * c.dk,
         cv_mem: kv_cv_bytes},
    )
    return w_store_h, w_ws_dot, w_res_dec, w_kv_dot


def _lower_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext, rc: RecipeContext,
    pairs: tuple[tuple[int, tuple[int, ...]], ...],
    active_state_pairs: tuple[tuple[int, tuple[int, ...]], ...],
    pre_io_work: Work, diag_exp2_ops: int, diag_reduce_ops: int, head_chunks: int,
    state_init_work: Work, unit_works: tuple[Work, Work, Work, Work],
    bands: tuple[tuple[int, int], ...], out_phase: Phase,
) -> KernelProgram:
    """Lower Triton chunk KDA with exact work conservation across uniform and ragged batches."""
    all_aics = tuple(aic for aic, _ in pairs)
    all_aivs = tuple(aiv for _, peers in pairs for aiv in peers)
    reduce_ratio = _reduce_rate_ratio(rc)
    sfu_ratio = _sfu_rate_ratio(rc)
    pre_io_stage = bind_phase(Phase("prepare_io_alu", pre_io_work), rc, ctx, all_aics, all_aivs)
    exp2_stage = _cap_vec_stage(
        bind_phase(Phase("prepare_diag_exp2", Work(resources={"vector": diag_exp2_ops}), ("prepare_io_alu",)),
                   rc, ctx, all_aics, all_aivs),
        rc, sfu_ratio,
    )
    red_stage = _cap_vec_stage(
        bind_phase(Phase("prepare_diag_reduce", Work(resources={"vector": diag_reduce_ops}), ("prepare_diag_exp2",)),
                   rc, ctx, all_aics, all_aivs),
        rc, reduce_ratio,
    )
    solve_sync_ms = ceil_div(head_chunks, max(1, len(all_aivs))) * TRITON_SOLVE_DEP_ROWS * p12_flag_ms(rc)
    prep_tail = "prepare_diag_reduce"
    stages: list[Stage] = [pre_io_stage, exp2_stage, red_stage]
    if solve_sync_ms > 0.0:
        stages.append(_fixed_stage("prepare_solve_sync", solve_sync_ms, after=(prep_tail,)))
        prep_tail = "prepare_solve_sync"

    st_aics = tuple(aic for aic, _ in active_state_pairs)
    st_aivs = tuple(peers[0] for _, peers in active_state_pairs)
    init_stage = bind_phase(Phase("state_init_final_io", state_init_work, (prep_tail,)), rc, ctx, st_aics, st_aivs)
    stages.append(init_stage)
    prev_stage = "state_init_final_io"

    k_panels = ceil_div(c.dk, STATE_PANEL_BK)
    cv_sync_per_step_ms = (2 * k_panels) * CV_ROUNDTRIP_PAIRED_SYNCS * p13_paired_ms(rc)
    w_store_h, w_ws_dot, w_res_dec, w_kv_dot = unit_works

    for band_idx, (steps, k_active) in enumerate(bands):
        b_pairs = active_state_pairs[:k_active]
        b_aics = tuple(aic for aic, _ in b_pairs)
        b_aivs = tuple(peers[0] for _, peers in b_pairs)
        caps = p01_cold_hbm_group_cap(rc, k_active, include_l2=True)
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
        b_name = f"state_recurrence_band_{band_idx}"
        stages.append(Stage(b_name, after=(prev_stage,), program=b_prog))
        prev_stage = b_name

    out_stage = bind_phase(Phase(out_phase.name, out_phase.work, (prev_stage,)), rc, ctx, all_aics, all_aivs)
    stages.append(out_stage)
    return KernelProgram(tuple(stages))


def build_triton_chunk_kda(
    c: LinearAttentionContract, ctx: OperatorPlanningContext,
) -> OperatorPlan:
    """Lower vllm_ascend.ops.triton.kda.kda.chunk_kda (7 kernels + 2 eager ops)."""
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
    beta_elem = element_bytes(c.beta_dtype)
    bt = c.bt
    total_tokens = sum(c.lengths)
    total_chunks = sum(ceil_div(length, bt) for length in c.lengths)
    head_chunks = total_chunks * c.hv
    token_heads = total_tokens * c.hv

    g_bytes = token_heads * c.dk * gate_elem
    qkv_bytes = token_heads * (2 * c.dk + c.dv) * elem
    beta_bytes = token_heads * beta_elem
    score_fp32 = head_chunks * bt * bt * FP32_BYTES
    score_bf16 = head_chunks * bt * bt * elem
    wukg_bytes = token_heads * (2 * c.dk + c.dv) * elem

    pre_read = (
        CUMSUM_PASSES * g_bytes
        + qkv_bytes + g_bytes + beta_bytes
        + score_fp32
        + score_bf16 + token_heads * (c.dk + c.dv) * elem + g_bytes + beta_bytes
    )
    pre_write = (
        CUMSUM_PASSES * g_bytes
        + ZERO_INIT_MATRICES * score_fp32 + score_bf16
        + ZERO_INIT_MATRICES * score_fp32
        + score_bf16
        + wukg_bytes
    )
    diag_blocks = max(1, bt // TRITON_DIAG_BLOCK)
    offdiag_tile = TRITON_DIAG_BLOCK * TRITON_DIAG_BLOCK
    inv_block_flops = 2 * offdiag_tile * TRITON_DIAG_BLOCK
    total_offdiag_blocks = 0
    for length in c.lengths:
        rem = length
        while rem > 0:
            rows = min(bt, rem)
            n_sub = ceil_div(rows, TRITON_DIAG_BLOCK)
            total_offdiag_blocks += (n_sub * (n_sub - 1)) // 2
            rem -= rows
    # vllm_ascend/ops/triton/kda/kda.py:538,542,671: off-diag sub_inter (2*tl.dot per pair) + solve_tril merge.
    pre_fp32_flops = (
        c.hv * total_offdiag_blocks * 2 * 2 * offdiag_tile * c.dk
        + head_chunks * TRITON_INV_MERGE_DOTS * inv_block_flops
    )
    pre_bf16_flops = head_chunks * 2 * bt * bt * (c.dk + c.dv)
    diag_exp2_ops = head_chunks * DIAG_EXP2_PER_HEAD_CHUNK
    diag_reduce_ops = head_chunks * DIAG_REDUCE_ROWS_PER_HEAD_CHUNK * c.dk
    diag_elem_ops = head_chunks * TRITON_DIAG_ELEM_OPS_MULT * diag_blocks * (TRITON_DIAG_BLOCK ** 2) * c.dk
    solve_ops = head_chunks * (SOLVE_TRIL_OPS_FACTOR * bt * bt * bt // SOLVE_TRIL_DIVISOR + bt * bt)
    pre_alu_ops = CUMSUM_PASSES * token_heads * c.dk + RECOMPUTE_GATE_OPS_MULT * token_heads * c.dk + solve_ops + diag_elem_ops

    pre_io_work = transfer(ctx, read=pre_read, write=pre_write) + Work(
        {"fp32": pre_fp32_flops, c.dtype: pre_bf16_flops},
        {"mte1": score_bf16 + token_heads * (2 * c.dk + c.dv) * elem, "fixpipe": wukg_bytes,
         "vector": pre_alu_ops},
    )
    pre_phase = Phase(
        "triton_kda_prepare_alu",
        pre_io_work + Work(resources={"vector": diag_exp2_ops + diag_reduce_ops}),
    )

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

    totals = (pre_phase, state_phase, out_phase)

    def lower(rc: RecipeContext) -> KernelProgram:
        return _lower_chunk_kda(
            c, ctx, rc, pairs, active_state_pairs, pre_io_work,
            diag_exp2_ops, diag_reduce_ops, head_chunks, state_init_work,
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

    total_progs = nk * nv * n_seqs * c.hv
    active_aivs = min(len(aivs), total_progs)
    full_waves, rem_progs = divmod(total_progs, active_aivs)
    waves = full_waves + int(rem_progs > 0)

    head_state_bytes = c.dk * c.dv * state_elem
    h0_read_total = n_seqs * c.hv * head_state_bytes * c.use_initial_state
    ht_write_total = total_tokens * c.hv * head_state_bytes * c.output_final_state
    qkvgb_per_tok_head = (2 * c.dk + c.dv) * elem + c.dk * gate_elem + beta_elem
    qkvgb_hbm_total = total_tokens * c.hv * qkvgb_per_tok_head
    qkvgb_mte2_total = total_tokens * c.hv * (nv * (2 * c.dk * elem + c.dk * gate_elem + beta_elem) + c.dv * elem)
    out_write_total = total_tokens * c.hv * c.dv * elem

    decay_alu = total_tokens * c.hv * nv * REC_DECAY_OPS_PER_STATE * bv * bk
    sfu_ops = total_tokens * c.hv * nv * bk
    dot1_red = total_tokens * c.hv * nv * (L2NORM_VECTORS + bv) * bk
    update_alu = total_tokens * c.hv * nv * REC_UPDATE_OPS_PER_STATE * bv * bk
    dot2_red = total_tokens * c.hv * nv * bv * bk
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
            "mte3": total_hbm_write,
            "vector": total_vec,
            **({"l2": total_l2_hits} if (total_l2_hits and has_l2_rate) else {}),
        }), ("load_initial_state",)),
    )

    def lower(rc: RecipeContext) -> KernelProgram:
        reduce_ratio = _reduce_rate_ratio(rc)
        sfu_ratio = _sfu_rate_ratio(rc)
        flag_ms = max_q * REC_FLAGS_PER_TOKEN * p12_flag_ms(rc)

        def _wave_stages(k_progs: int) -> tuple[Stage, ...]:
            share = k_progs / float(total_progs)
            wave_aivs = aivs[:k_progs]
            w_load = Work(resources={
                "hbm": total_hbm_read * share,
                "mte2_aiv": total_mte2_aiv * share,
                **({"l2": total_l2_hits * share} if (total_l2_hits and has_l2_rate) else {}),
            })
            s0 = bind_phase(Phase("rec_load", w_load), rc, ctx, (), wave_aivs)
            s1 = bind_phase(Phase("rec_decay_alu", Work(resources={"vector": decay_alu * share}), ("rec_load",)),
                            rc, ctx, (), wave_aivs)
            s2 = _cap_vec_stage(
                bind_phase(Phase("rec_decay_sfu", Work(resources={"vector": sfu_ops * share}), ("rec_decay_alu",)),
                           rc, ctx, (), wave_aivs),
                rc, sfu_ratio,
            )
            s3 = _cap_vec_stage(
                bind_phase(Phase("rec_dot1_reduce", Work(resources={"vector": dot1_red * share}), ("rec_decay_sfu",)),
                           rc, ctx, (), wave_aivs),
                rc, reduce_ratio,
            )
            s4 = bind_phase(Phase("rec_update_alu", Work(resources={"vector": update_alu * share}), ("rec_dot1_reduce",)),
                            rc, ctx, (), wave_aivs)
            s5 = _cap_vec_stage(
                bind_phase(Phase("rec_dot2_reduce", Work(resources={"vector": dot2_red * share}), ("rec_update_alu",)),
                           rc, ctx, (), wave_aivs),
                rc, reduce_ratio,
            )
            w_store = Work(resources={"hbm": total_hbm_write * share, "mte3": total_hbm_write * share})
            s6 = bind_phase(Phase("rec_store", w_store, ("rec_dot2_reduce",)), rc, ctx, (), wave_aivs)
            if flag_ms > 0.0:
                return (s0, s1, s2, s3, s4, s5, s6, _fixed_stage("rec_sync", flag_ms, after=("rec_store",)))
            return (s0, s1, s2, s3, s4, s5, s6)

        if full_waves == 1 and rem_progs == 0:
            return KernelProgram(_wave_stages(active_aivs))
        full_prog = (
            KernelProgram(_wave_stages(active_aivs))
            if full_waves == 1
            else PipelineProgram(full_waves, REC_WAVE_SLOTS, lambda _i: _wave_stages(active_aivs))
        )
        if rem_progs == 0:
            return KernelProgram((Stage("triton_rec_waves", program=full_prog),))
        return KernelProgram((
            Stage("triton_rec_waves", program=full_prog),
            Stage("triton_rec_tail_wave", after=("triton_rec_waves",), program=KernelProgram(_wave_stages(rem_progs))),
        ))

    return make_plan(
        ctx, TRITON_RECURRENT_KDA_ABI, phases, sources=TRITON_RECURRENT_SOURCES,
        owners=active_aivs, program=lower,
        metadata={
            "template": "triton_recurrent_kda", "bk": bk, "bv": bv,
            "total_programs": total_progs, "active_aivs": active_aivs, "waves": waves,
            "persistent_ub_resident": True,
        },
    )
