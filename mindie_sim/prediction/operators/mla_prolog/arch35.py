"""ACLNN and DSL MLA prolog strategies for the 3510 architecture.

Upstream-ABI: aclnnMlaPrologV4WeightNz

[Role] Route ACLNN split-N/split-M and DSL dual-destination output.
[Invariants] ACLNN uses GM workspace visibility. DSL sizes the exchange from
  declared capacity and only binds AICs with two reachable FixPipe UB destinations.
[Timeouts & Locks] Bounded by physical core count with no I/O or locks.
[Side-Effect Policy] Returns immutable programs and verified source references.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from typing import Literal

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan
from ...runtime.tile_solver import LiveBufferSpec, TileInfeasibleError, check_live_buffers
from ..contracts import ArchStrategy, OperatorPlanningContext
from ._lowering import (
    BF16_BYTES, FP32_BYTES, FRACTAL_K_BYTE, FRACTAL_MN, MX_GROUP_SIZE,
    _GenericTiling, _ResidentTiling, _align, ceil_div,
)
from .common import (
    ABI_V3_NZ, FAMILY, GROUP_COMPUTE_MIN_HEADS, KV_QUANT_PER_TENSOR, WEIGHT_QUANT_INT8,
    WEIGHT_QUANT_MXFP8, PrologContract, aclnn_contract_supported,
    build_plan, contract_for, unsupported, uses_split_m,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
ARCH_ID = ArchId("npu_arch_3510")
UPSTREAM_ABI = "aclnnMlaPrologV4WeightNz"
DSL_SUPPORTED_HEADS: frozenset[int] = frozenset({16, 64})
AIV_PER_BLOCK = 2
DSL_TOKEN_RANGE_UPPERS = (16, 32, 48, 64, 80, 96, 192, 256)
DSL_RESIDENT_GEOMETRIES = (
    (5120, 1280, 16, 512, 64),
    (5120, 1280, 64, 512, 64),
)
DSL_RESIDENT_PANEL_WIDTH = 256
DSL_RESIDENT_TILE_WIDTH = 64
DSL_RESIDENT_K_WINDOW = 320
DSL_RESIDENT_ALIGNMENT_RESERVE = 4 * 1024
DSL_RESIDENT_MAX_ROWS = 96
DSL_RESIDENT_K_SPLITS = 4
DSL_L0C_RESERVE = 16 * 1024
DSL_VECTOR_LANES = 64
DSL_A2_VECTOR_SCRATCH = 5
DSL_DECODE_SPLIT_THRESHOLD = 128
DSL_DECODE_REDUCE_THRESHOLD = 32
DSL_DECODE_SPLIT_CANDIDATES = (2, 4, 8)
DSL_DECODE_K_LIMIT = 640
DSL_LONG_TOKEN_THRESHOLD = 4096
DSL_LONG_K_SPLITS = 20
DSL_PREFILL_MAX_ROWS = 96
DSL_LARGE_DECODE_ROWS = 112
DSL_LARGE_DECODE_COLUMNS = 256
DSL_DECODE_MAX_ROWS = 128
DSL_PREFILL_MAX_COLUMNS = 128
DSL_PREFILL_REDUCE_ROWS = 4
DSL_K_L1_MAX = 512
DSL_K_L0_MAX = 128
DSL_QB_FULL_HEAD_TILE = 512
DSL_QB_GENERIC_MAX_COLUMNS = 256
DSL_UB_ALIGNMENT = 32
DSL_A2_FIXED_SCRATCH_VECTORS = 3
DSL_MAX_EXTRA_PROJECTION_WAVE_PERCENT = 5
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling.cpp",
        source_sha256="1770937bd4a58222c5606ce03bec3c574e23bd563ea341c03a1710f153962909",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling.h",
        source_sha256="ca141526d57ede6e0d87b3c2d3a1c5e126f228cac3a4dd71a55068f072688f3c",
    ),
    SourceRef(
        repository="Ascend/op-plugin",
        revision="48189acf95882e23fae56ea96ceeb89358cb42da",
        relative_path="op_plugin/ops/opapi/MlaPrologV3KernelNpuOpApi.cpp",
        source_sha256="44f2a01f2acd64c6ce8ea1262b8e68cb11d74fd58ea3e8cd457085d3e0715c14",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/attention/mla_v1.py",
        source_sha256="f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_n_arch35.h",
        source_sha256="81d9ca8290a0f65d05bd0c83df45c2581c30cc9007337b0588618603175c9d0e",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch35/mla_prolog_comm_arch35.h",
        source_sha256="7efde023de09dd6186536b2cc45c76091f3fbed3c211c9f8965690efba453e80",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch35/kernel_mla_prolog_split_m.h",
        source_sha256="e4a0688375492cfcd23ff528b10381d3a34223539497cec28814322d3d38bee4",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_kernel/arch35/service_matmul_arch35.h",
        source_sha256="9fb5f6d4057eead70c044a78b3b047825c54c2d54bf973314dd9b3ec3e0441ad",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.h",
        source_sha256="170ae3ebc125ddf56916ed83c37d84ce6bc3b92a0889bfc44a06cb6e30b22729",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/attn_prologue/attn_prologue_tiling.py",
        source_sha256="1f58dca0d1a7679d0f86b1eea6a1afaf446c2bf1815599954725a2b9104ecb8f",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/attn_prologue/attn_prologue_kernel.py",
        source_sha256="84f69f4f8d84c4713a80915b95cdfb00dce67a30653b1f9cb494f503db755d7b",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/attn_prologue/attn_prologue.py",
        source_sha256="80096f1edface568dd8dd9c01f407f40d092d9ecae01b9eaeae3f044c3a27898",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog_v3/op_api/aclnn_mla_prolog_v4_weight_nz.cpp",
        source_sha256="5513befaeb1e3c006f3d4319ad389a23144ca0c783da3b8cea5ffffa9d3582b6",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mla_prolog/op_host/mla_prolog_tiling_check.cpp",
        source_sha256="5e34239dabdf9a159d41b542401fc2bcf7c3551b6e6695c5d761dff5638e6d57",
    ),
)

DSL_SOURCES = tuple(ref for ref in SOURCES if ref.repository == "tilelang-ascend/kernels")
ACLNN_SOURCES = tuple(ref for ref in SOURCES if ref.repository != "tilelang-ascend/kernels")


def _largest_aligned_divisor(value: int, limit: int, alignment: int) -> int:
    return next((n for n in range(limit // alignment * alignment, 0, -alignment) if value % n == 0), 0)


def _dsl_mx_bytes(k: int) -> int:
    return k + k // MX_GROUP_SIZE


def _dsl_prefill_buffers_fit(
    c: PrologContract, ctx: OperatorPlanningContext, columns: int,
    a1_rows: int, a1_k: tuple[int, int], reduce_rows: int,
    qb_rows: int, qb_cols: int, qb_k: tuple[int, int],
    *, resident: bool = False, split: bool = False,
) -> bool:
    a1_window, a1_block = a1_k
    qb_window, qb_block = qb_k
    if not a1_block or not qb_window or not qb_block:
        return False
    rank, head, lanes = c.q_lora_rank, c.dsl_d_width, DSL_VECTOR_LANES
    exp_stride = _align(rank // MX_GROUP_SIZE, DSL_UB_ALIGNMENT // FP32_BYTES)
    scale_stride = _align(rank // MX_GROUP_SIZE, DSL_UB_ALIGNMENT)
    a1_l1 = 2 * (a1_rows * _dsl_mx_bytes(a1_block) + columns * _dsl_mx_bytes(a1_window))
    qb_x = _dsl_mx_bytes(rank) if resident else 2 * _dsl_mx_bytes(qb_block)
    qb_l1 = qb_rows * qb_x + 2 * qb_cols * _dsl_mx_bytes(qb_window)
    ub_a2 = FP32_BYTES * (
        (reduce_rows + 1) * (rank + head + max(exp_stride, lanes))
        + (2 * reduce_rows + DSL_A2_FIXED_SCRATCH_VECTORS) * lanes
    ) + reduce_rows * (rank * (1 + BF16_BYTES) + scale_stride) + 2 * c.kv_row_bytes
    if split:
        ub_a2 += reduce_rows * (2 * rank + head) * FP32_BYTES
    qb_vector_cols = qb_cols // 2 if resident else ceil_div(head, 2)
    ub_b = qb_rows * (qb_vector_cols * (FP32_BYTES + BF16_BYTES) + 2 * lanes * FP32_BYTES)
    qb_l0b_slots = 1 if qb_cols == DSL_QB_FULL_HEAD_TILE else 2
    l0b_a_bytes = 2 * (a1_rows * a1_block + qb_rows * qb_block)
    l0b_b_bytes = 2 * columns * a1_block + qb_l0b_slots * qb_cols * qb_block
    l0c_bytes = (a1_rows * columns + qb_rows * (qb_cols if resident else head)) * FP32_BYTES + DSL_L0C_RESERVE
    # Generic DSL uses the L0B budget for both static operand allocations (separate intervals).
    live = (
        LiveBufferSpec("dsl_l1", "l1", a1_l1 + qb_l1, 0, 0, owner_kind="aic", alignment_bytes=1),
        LiveBufferSpec("dsl_l0b_a", "l0b", l0b_a_bytes, 0, 0, owner_kind="aic", alignment_bytes=1),
        LiveBufferSpec("dsl_l0b_b", "l0b", l0b_b_bytes, 1, 1, owner_kind="aic", alignment_bytes=1),
        LiveBufferSpec("dsl_l0c", "l0c", l0c_bytes, 0, 0, owner_kind="aic", alignment_bytes=1),
        LiveBufferSpec("dsl_ub", "ub", ub_a2 + ub_b + lanes * FP32_BYTES, 0, 0, owner_kind="aiv", alignment_bytes=1),
    )
    try:
        check_live_buffers(live, ctx.geometry, family=FAMILY)
    except TileInfeasibleError:
        return False
    return True


def _dsl_search_tile_pair(predicate, rows: int, qb_rows: int) -> tuple[int, int] | None:
    while not predicate(rows, qb_rows):
        if qb_rows > FRACTAL_MN:
            qb_rows = max(FRACTAL_MN, qb_rows // 2 // FRACTAL_MN * FRACTAL_MN)
        elif rows > FRACTAL_MN:
            rows = max(FRACTAL_MN, rows // 2 // FRACTAL_MN * FRACTAL_MN)
        else:
            return None
    return rows, qb_rows


def _dsl_prefill_tiles(
    c: PrologContract, ctx: OperatorPlanningContext, cores: int, rows: int, columns: int,
) -> tuple[int, int, int] | None:
    """Return (A1 rows, QB rows, split count) after checking DSL prefill buffer ceilings."""
    buffers = dict(ctx.datapath.buffers) | dict(ctx.geometry.buffers)
    l0b = buffers.get("l0b", 0) // 2 * 2
    rank, head = c.q_lora_rank, c.dsl_d_width
    if columns % FRACTAL_K_BYTE:
        return None
    mx_pair = 2 * MX_GROUP_SIZE
    lanes = DSL_VECTOR_LANES

    def k_tiles(reduction: int, cols: int) -> tuple[int, int]:
        window = _largest_aligned_divisor(reduction, min(reduction, DSL_K_L1_MAX), mx_pair)
        block = _largest_aligned_divisor(window, min(window, DSL_K_L0_MAX, l0b // 2 // 2 // cols), mx_pair)
        return window, block

    qb_cols = DSL_QB_FULL_HEAD_TILE
    if head % qb_cols == 0:
        qb_k = DSL_RESIDENT_K_WINDOW, mx_pair
    else:
        qb_cols = _largest_aligned_divisor(head, DSL_QB_GENERIC_MAX_COLUMNS, FRACTAL_K_BYTE)
        if not qb_cols:
            return None
        qb_k = k_tiles(rank, qb_cols)

    single_k = k_tiles(c.hidden_size, columns)
    qb_grid_fits = rank % qb_k[0] == 0
    head_slices = 2 * AIV_PER_BLOCK
    resident_eligible = (
        head % (head_slices * lanes) == 0
        and head // head_slices >= c.qk_rope_head_dim
        and c.num_heads * 2 >= cores
    )
    res_qb_window = _largest_aligned_divisor(rank, min(DSL_RESIDENT_K_WINDOW, rank), mx_pair) if resident_eligible else 0

    if c.tokens >= DSL_LONG_TOKEN_THRESHOLD:
        if c.hidden_size % (DSL_LONG_K_SPLITS * mx_pair) != 0:
            return None
        initial_k = k_tiles(c.hidden_size // DSL_LONG_K_SPLITS, columns)
        found = _dsl_search_tile_pair(
            lambda r, qr: _dsl_prefill_buffers_fit(c, ctx, columns, r, initial_k, 1, qr, qb_cols, qb_k, split=True),
            rows, rows,
        )
        if found is None:
            return None
        fit_rows, fit_qb_rows = found
        if qb_grid_fits and _dsl_prefill_buffers_fit(
            c, ctx, columns, fit_rows, single_k, DSL_PREFILL_REDUCE_ROWS, fit_qb_rows, qb_cols, qb_k,
        ):
            return fit_rows, fit_qb_rows, 1
        if resident_eligible and _dsl_prefill_buffers_fit(
            c, ctx, columns, fit_rows, single_k, DSL_PREFILL_REDUCE_ROWS,
            FRACTAL_MN, head // 2, (res_qb_window, mx_pair), resident=True,
        ):
            return fit_rows, FRACTAL_MN, 1
        return (fit_rows, fit_qb_rows, DSL_LONG_K_SPLITS) if qb_grid_fits else None

    if qb_grid_fits:
        found = _dsl_search_tile_pair(
            lambda r, qr: _dsl_prefill_buffers_fit(
                c, ctx, columns, r, single_k, DSL_PREFILL_REDUCE_ROWS, qr, qb_cols, qb_k,
            ),
            rows, rows,
        )
        if found is not None:
            return found[0], found[1], 1
    if resident_eligible:
        found = _dsl_search_tile_pair(
            lambda r, _: _dsl_prefill_buffers_fit(
                c, ctx, columns, r, single_k, DSL_PREFILL_REDUCE_ROWS,
                FRACTAL_MN, head // 2, (res_qb_window, mx_pair), resident=True,
            ),
            rows, FRACTAL_MN,
        )
        if found is not None:
            return found[0], FRACTAL_MN, 1
    return None


def _dsl_generic_tiling(op: OpSpec, c: PrologContract, ctx: OperatorPlanningContext, cores: int) -> _GenericTiling:
    tokens, hidden = c.tokens, c.hidden_size
    decode = tokens <= DSL_TOKEN_RANGE_UPPERS[-1]
    large_decode = decode and tokens > DSL_DECODE_SPLIT_THRESHOLD
    mx_pair_width = 2 * MX_GROUP_SIZE
    splits = 1
    if large_decode:
        splits = DSL_RESIDENT_K_SPLITS
    elif decode:
        for candidate in DSL_DECODE_SPLIT_CANDIDATES:
            if hidden % (candidate * mx_pair_width) == 0 and c.dsl_d_width // mx_pair_width * candidate >= cores:
                splits = candidate
                if hidden // candidate <= DSL_DECODE_K_LIMIT:
                    break
        if hidden % (DSL_RESIDENT_K_SPLITS * mx_pair_width) == 0:
            splits = DSL_RESIDENT_K_SPLITS
    row_limit = DSL_DECODE_MAX_ROWS if decode else DSL_PREFILL_MAX_ROWS
    token_tile = DSL_LARGE_DECODE_ROWS if large_decode else min(row_limit, ceil_div(tokens, FRACTAL_MN) * FRACTAL_MN)
    column_limit = DSL_RESIDENT_TILE_WIDTH if decode else DSL_PREFILL_MAX_COLUMNS
    width = gcd(c.q_lora_rank, c.dsl_d_width)
    if large_decode:
        column_limit = DSL_LARGE_DECODE_COLUMNS
    column_tile = next((n for n in range(column_limit, 0, -FRACTAL_K_BYTE) if width % n == 0), width)
    qb_token_tile = token_tile
    if not decode:
        tuned = _dsl_prefill_tiles(c, ctx, cores, token_tile, column_tile)
        if tuned is None:
            raise unsupported(op, ctx, "dsl_prefill_tiles_exceed_capacity_or_nz_grid")
        token_tile, qb_token_tile, splits = tuned
    reduce_tile = DSL_PREFILL_REDUCE_ROWS if not decode and splits == 1 else 1
    if decode and DSL_DECODE_REDUCE_THRESHOLD < tokens <= DSL_DECODE_SPLIT_THRESHOLD and splits == DSL_RESIDENT_K_SPLITS:
        reduce_tile = 2
    tile_splits = ceil_div(tokens, token_tile) * splits
    qa_jobs, kv_jobs = (tile_splits * n // column_tile for n in (c.q_lora_rank, c.dsl_d_width))
    shared = ceil_div(qa_jobs + kv_jobs, cores)
    separate = ceil_div(qa_jobs, cores) + ceil_div(kv_jobs, cores)
    split_queue = 100 * (separate - shared) <= DSL_MAX_EXTRA_PROJECTION_WAVE_PERCENT * shared
    return _GenericTiling(token_tile, column_tile, splits, reduce_tile, split_queue, qb_token_tile=qb_token_tile)


def _resident_groups(jobs: int, columns: int, cores: int, weight: int, activation: int) -> int:
    candidates = {min(columns, ceil_div(cores, jobs))}
    candidates.update(group for group in range(1, columns + 1) if columns % group == 0)

    def cost(groups: int) -> tuple[int, int]:
        loads = [0] * cores
        for job in range(jobs * groups):
            loads[job % cores] += ceil_div(columns - job % groups, groups) * weight + activation
        return max(loads), groups

    return min(candidates, key=cost)


def _dsl_resident_tiling(c: PrologContract, ctx: OperatorPlanningContext, cores: int) -> _ResidentTiling | None:
    geometry = (c.hidden_size, c.q_lora_rank, c.num_heads, c.dsl_d_width, c.qk_rope_head_dim)
    if not (0 < c.tokens <= DSL_TOKEN_RANGE_UPPERS[-1] and c.do_rope and geometry in DSL_RESIDENT_GEOMETRIES):
        return None
    buffers = dict(ctx.datapath.buffers) | dict(ctx.geometry.buffers)
    upper = next(limit for limit in DSL_TOKEN_RANGE_UPPERS if c.tokens <= limit)
    split_k, n, panel = DSL_RESIDENT_K_SPLITS, DSL_RESIDENT_TILE_WIDTH, DSL_RESIDENT_PANEL_WIDTH
    rank, head, window = c.q_lora_rank, c.dsl_d_width, DSL_RESIDENT_K_WINDOW
    k = c.hidden_size // split_k
    qb_scales = rank // MX_GROUP_SIZE * head
    qa_panels = 2 * n * (window + window // MX_GROUP_SIZE)
    vector_bytes = DSL_VECTOR_LANES * FP32_BYTES
    ub_a2 = ((split_k + 2) * (rank + head) * FP32_BYTES + 2 * rank + 2 * c.kv_row_bytes
             + DSL_A2_VECTOR_SCRATCH * vector_bytes + 2 * max(rank // MX_GROUP_SIZE, DSL_VECTOR_LANES) * FP32_BYTES
             + 2 * ceil_div(rank // MX_GROUP_SIZE, MX_GROUP_SIZE) * MX_GROUP_SIZE + 2 * rank * BF16_BYTES)
    for limit in range(min(upper, DSL_RESIDENT_MAX_ROWS), 0, -FRACTAL_MN):
        token_tiles = ceil_div(upper, limit)
        rows = ceil_div(ceil_div(upper, token_tiles), FRACTAL_MN) * FRACTAL_MN
        twin = token_tiles == 2
        concurrent_tiles = 2 if twin else 1
        windows = 2 if twin else rank // window
        l1 = concurrent_tiles * rows * (k + k // MX_GROUP_SIZE) + windows * panel * window
        l1 += qb_scales + qa_panels + DSL_RESIDENT_ALIGNMENT_RESERVE
        l0c_columns = 2 * (panel + n) if twin else head + n
        ub_b = rows * (head // 2) * (FP32_BYTES + BF16_BYTES) + (1 + concurrent_tiles * rows) * vector_bytes
        live = (
            LiveBufferSpec("res_l1", "l1", l1, 0, 0, owner_kind="aic", alignment_bytes=1),
            LiveBufferSpec("res_l0a", "l0a", 2 * rows * window, 0, 0, owner_kind="aic", alignment_bytes=1),
            LiveBufferSpec("res_l0b", "l0b", 2 * n * window, 0, 0, owner_kind="aic", alignment_bytes=1),
            LiveBufferSpec("res_l0c", "l0c", rows * l0c_columns * FP32_BYTES + DSL_L0C_RESERVE, 0, 0, owner_kind="aic", alignment_bytes=1),
            LiveBufferSpec("res_ub", "ub", ub_a2 + ub_b, 0, 0, owner_kind="aiv", alignment_bytes=1),
        )
        try:
            check_live_buffers(live, ctx.geometry, family=FAMILY)
        except TileInfeasibleError:
            continue
        a_groups = _resident_groups(token_tiles * split_k, (rank + head) // n, cores, n * k, rows * k)
        b_groups = cores if twin else _resident_groups(token_tiles, c.num_heads, cores, head * rank, rows * rank)
        gm_alignment = ctx.datapath.route("l0c", "gm").path.alignment_bytes
        return _ResidentTiling(rows, token_tiles, split_k, n, panel, a_groups, b_groups, gm_alignment)
    return None


def _dsl_owners(op: OpSpec, ctx: OperatorPlanningContext):
    stream = ctx.stream_owners
    if stream is not None and stream.isolation == "disjoint":
        aics = stream.aic[:ctx.effective_aic_cores]
        aivs = stream.aiv[:ctx.effective_aiv_cores]
    elif stream is not None and stream.pool_aic:
        aics = stream.pool_aic[:ctx.effective_aic_cores]
        aivs = stream.pool_aiv[:ctx.effective_aiv_cores]
    else:
        aics = tuple(range(ctx.effective_aic_cores))
        aivs = tuple(range(ctx.effective_aiv_cores))
    available = dict(ctx.datapath.aic_aiv_pairs)
    fixpipe = ctx.datapath.route("l0c", "ub")
    pairs = []
    for aic in aics:
        peers = tuple(aiv for aiv in available.get(aic, ()) if aiv in aivs)
        if len(peers) != AIV_PER_BLOCK:
            continue
        for aiv in peers:
            ctx.datapath.route("l0c", "ub", owner_group={"aic": aic, "aiv": aiv})
        pairs.append((aic, peers))
    if not pairs:
        raise unsupported(op, ctx, "dsl_requires_two_aivs_per_aic")
    return tuple(pairs), fixpipe.path.alignment_bytes


def _split_m_owners(ctx: OperatorPlanningContext):
    aics, aivs = ctx.effective_aic_cores, ctx.effective_aiv_cores
    if not aics or not aivs:
        return ()
    width, larger = divmod(aivs, aics)
    pairs, start = [], 0
    for owner in range(aics):
        end = start + width + (owner < larger)
        pairs.append((owner, tuple(range(start, end)) or (owner % aivs,)))
        start = end
    return tuple(pairs)


def _supports(c: PrologContract) -> bool:
    if c.is_dsl:
        return (
            c.weight_quant_mode == WEIGHT_QUANT_MXFP8
            and c.kv_cache_quant_mode == KV_QUANT_PER_TENSOR
            and not c.query_quant_mode
            and c.num_heads in DSL_SUPPORTED_HEADS
        )
    return (
        (c.weight_quant_mode != WEIGHT_QUANT_MXFP8 or c.abi in {ABI_V3_NZ, UPSTREAM_ABI})
        and aclnn_contract_supported(c)[0]
    )


@dataclass(frozen=True, slots=True)
class _Arch35Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = UPSTREAM_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        return _supports(contract_for(op, ctx))

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        c = contract_for(op, ctx)
        if not _supports(c):
            raise unsupported(op, ctx, f"unsupported_abi_or_quant_mode:{c.abi}:{c.weight_quant_mode}")
        if c.is_dsl:
            pairs, alignment = _dsl_owners(op, ctx) if c.tokens else ((), 1)
            resident = _dsl_resident_tiling(c, ctx, len(pairs))
            generic = _dsl_generic_tiling(op, c, ctx, len(pairs)) if c.tokens and resident is None else None
            if generic and c.hidden_size % (generic.k_splits * 2 * MX_GROUP_SIZE):
                raise unsupported(op, ctx, "dsl_generic_split_requires_complete_mx_pairs")
            return build_plan(c, ctx, sources=DSL_SOURCES, mode="dsl", pairs=pairs,
                              resident=resident, generic=generic, fixpipe_alignment=alignment)
        if uses_split_m(c):
            return build_plan(c, ctx, sources=ACLNN_SOURCES, mode="split_m", pairs=_split_m_owners(ctx))
        cq_scale = c.weight_quant_mode == WEIGHT_QUANT_INT8 and c.num_heads >= GROUP_COMPUTE_MIN_HEADS
        return build_plan(c, ctx, sources=ACLNN_SOURCES, mode="split_n", cq_scale=cq_scale)


STRATEGY: ArchStrategy = _Arch35Strategy()
