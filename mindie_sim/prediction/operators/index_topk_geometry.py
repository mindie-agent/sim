"""Sparse indexer source geometry and work inventory.

[Role] Normalize the source caller into geometry and work counters.
[Invariants] Score64 and sparse128 geometry, explicit top-k tile and padded work are preserved.
  No durations, hardware-rate fits or runtime source reads enter normalization.
[Timeouts & Locks] Bounded metadata arithmetic; no waits or locks.
[Side-Effect Policy] Immutable declarations; no device or external I/O.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
from mindie_sim.schema.common import align_up, dtype_bytes

IMPLEMENTATION = "triton_block_index_topk"
SOURCE_PATH = (
    "/Users/maoxx241/code/vllm-v0.25.1/"
    "vllm/models/minimax_m3/common/ops/index_topk.py"
)
SOURCE_SHA256 = "1ca8557aed4260d2f4a742b9eda81d6f774e49d4b016ccc555dde40976c5a1a0"
SOURCE_BYTES = 37281
SUPPLEMENTARY_SOURCES = {
    "indexer.py": {
        "path": "/Users/maoxx241/code/vllm-v0.25.1/"
                "vllm/models/minimax_m3/common/indexer.py",
        "sha256": "43787aeb1c8a7fd0f990aa62267e6c47f802b511b8850a00f99de0fad2fb468a",
        "role": "dispatch and supported head/block sizes; Triton indexer is bf16 only",
    },
    "test_minimax_m3.py": {
        "path": "/Users/maoxx241/code/vllm-v0.25.1/"
                "tests/kernels/attention/test_minimax_m3.py",
        "sha256": "0e88afe33ad85542da6ca02ecbaa144af12bbad504546097b062f643a8742bd2",
        "role": "independent dense score/sorted-index reference",
    },
    "amd/ops/index_topk.py": {
        "path": "/Users/maoxx241/code/vllm-v0.25.1/"
                "vllm/models/minimax_m3/amd/ops/index_topk.py",
        "sha256": "6cc2ee849a0d015bca64f4f826be0478c43dd3cd36ae601712b2aa9971d52999",
        "role": "different platform source variant; not selected",
    },
}

# Source constants and wrapper tiles.
SPARSE_BLOCK = 128          # SPARSE_BLOCK_SIZE, source line 24
SCORE_Q_TILE = 64           # BLOCK_SIZE_Q in minimax_m3_index_score, line 678
BLOCK_SIZE_Q = 1            # sample_interval for the M3 top-k, line 739
TOPK_TILES = (64, 128, 256, 512, 1024, 2048)  # autotune configs, lines 174-184
SCORE_KERNEL = "_index_block_score_kernel"
TOPK_KERNEL = "_topk_index_kernel"
SCORE_PHASE = "score"
TOPK_PHASE = "topk"

# Source-level elementary vector work.  These are logical lane operations, not
# calibrated instruction latencies; every kernel still pays scalar, issue,
# reduction and synchronization costs that are not represented here.
SCORE_MASK_OPS = 2          # compare + select per masked qk lane, lines 151-152
SCORE_MAX_OPS = 1           # one max reduction operand per qk lane, line 154
TOPK_CAUSAL_OPS = 3         # compare, load mask, select sentinel, lines 231-234
TOPK_NAN_OPS = 2            # score != score compare + select, line 235
TOPK_INIT_OPS = 3           # compare init, combine causal, select/subtract, 237-240
TOPK_LOCAL_OPS = 3          # compare local, combine causal, select/subtract, 241-244
TOPK_INDEX_OPS = 2          # i + off_k + 1 and causal select, line 246
TOPK_COPY_OPS = 2           # save the previous top-k score/index arrays, 245-246
TOPK_PREPARE_OPS = (
    TOPK_CAUSAL_OPS + TOPK_NAN_OPS + TOPK_INIT_OPS + TOPK_LOCAL_OPS
    + TOPK_INDEX_OPS + TOPK_COPY_OPS
)
CAS_OPS = 6                 # compare-exchange lane: selects, compare, 2 XORs, select
MERGE_COMBINE_OPS = 6       # score and index: two multiplies + add each, 256-261
INIT_FILL_OPS = 2           # fill topk_score and topk_idx, lines 226-227
EXTRACT_MUL_OPS = 1         # reshape multiply per lane, line 270
EXTRACT_ADD_OPS = 1         # reduction add per consumed lane, line 270
STORE_ADDR_OPS = 1          # topk_mask compare over the tile, line 269


def _ceil_div(value: int, divisor: int) -> int:
    return -((-value) // divisor)


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _int(params: Mapping[str, Any], key: str, default: int | None = None) -> int:
    if key not in params:
        if default is None:
            raise ValueError(f"index_topk_parameter_required:{key}")
        return int(default)
    value = params[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"index_topk_parameter_must_be_integer:{key}")
    return int(value)


def _flag(params: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = params.get(key, default)
    if not isinstance(value, bool):
        raise ValueError(f"index_topk_parameter_must_be_boolean:{key}")
    return value


@dataclass(frozen=True)
class _IndexTopk:
    implementation: str
    query: int
    context: int
    batch: int
    heads: int
    head_dim: int
    topk: int
    block_size_q: int
    score_block: int
    topk_block: int
    init_blocks: int
    local_blocks: int
    dtype: str
    mask_init: bool
    mask_local: bool

    @property
    def sequence_length(self) -> int:
        return self.context + self.query

    @property
    def total_q(self) -> int:
        return self.batch * self.query

    @property
    def max_blocks(self) -> int:
        return _ceil_div(self.sequence_length, self.score_block)

    @property
    def score_stride(self) -> int:
        return align_up(self.max_blocks, 16)

    @property
    def score_tiles(self) -> int:
        return _ceil_div(self.query, SCORE_Q_TILE)

    @property
    def block_size_t(self) -> int:
        return _next_power_of_two(self.topk)

    @property
    def log2_tile(self) -> int:
        return self.topk_block.bit_length() - 1


def _normalize(params: Mapping[str, Any]) -> _IndexTopk:
    """Validate the source-selected geometry; reject nearby-but-different paths."""
    allowed = {
        "implementation", "query", "context", "batch", "heads", "head_dim",
        "topk", "block_size_q", "score_block", "topk_block", "init_blocks",
        "local_blocks", "dtype", "mask_init", "mask_local",
    }
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise ValueError("index_topk_unknown_parameter:" + ",".join(unknown))

    implementation = str(params.get("implementation", IMPLEMENTATION))
    if implementation != IMPLEMENTATION:
        raise ValueError(f"unsupported_index_topk_implementation:{implementation}")

    dtype = str(params.get("dtype", "bf16")).strip().lower()
    if dtype != "bf16":
        # indexer.select_indexer_impl_cls rejects non-bf16 for the Triton
        # indexer (indexer.py lines 511-515), and the source file's docstring
        # records the bf16 score-only path.
        raise ValueError(f"unsupported_index_topk_dtype:{dtype}")

    g = _IndexTopk(
        implementation=implementation,
        query=_int(params, "query"),
        context=_int(params, "context"),
        batch=_int(params, "batch", 1),
        heads=_int(params, "heads", 1),
        head_dim=_int(params, "head_dim"),
        topk=_int(params, "topk"),
        block_size_q=_int(params, "block_size_q", BLOCK_SIZE_Q),
        score_block=_int(params, "score_block", SPARSE_BLOCK),
        topk_block=_int(params, "topk_block", 256),
        init_blocks=_int(params, "init_blocks", 0),
        local_blocks=_int(params, "local_blocks", 0),
        dtype=dtype,
        mask_init=_flag(params, "mask_init"),
        mask_local=_flag(params, "mask_local"),
    )
    if g.query < 1:
        raise ValueError("index_topk_query_must_be_positive")
    if g.context < 0:
        raise ValueError("index_topk_context_must_be_nonnegative")
    if g.batch < 1 or g.heads < 1:
        raise ValueError("index_topk_batch_and_heads_must_be_positive")
    if g.head_dim != 128:
        # indexer.py lines 89-90: supported head sizes are [128].
        raise ValueError(f"unsupported_index_head_dim:{g.head_dim}")
    if g.score_block != SPARSE_BLOCK:
        # indexer.py lines 93-94 and source line 24: sparse block is 128.
        raise ValueError(f"unsupported_sparse_block:{g.score_block}")
    if g.block_size_q != BLOCK_SIZE_Q:
        # The M3 top-k wrapper passes sample_interval=1; other intervals are
        # a different query-block path that this source does not select.
        raise ValueError(f"unsupported_block_size_q:{g.block_size_q}")
    if g.topk < 1:
        raise ValueError("index_topk_topk_must_be_positive")
    if g.topk_block not in TOPK_TILES:
        raise ValueError(f"unsupported_topk_tile:{g.topk_block}")
    if g.block_size_t >= g.topk_block:
        # _topk_index_kernel static_assert(BLOCK_SIZE_K > BLOCK_SIZE_T).
        raise ValueError("index_topk_tile_must_exceed_padded_topk")
    if g.init_blocks < 0 or g.local_blocks < 0:
        raise ValueError("index_topk_forced_blocks_must_be_nonnegative")
    return g


def _floor_sum(count: int, block: int) -> int:
    """sum(floor(m / block) for m in range(count))."""
    whole, remainder = divmod(count, block)
    return block * whole * (whole - 1) // 2 + whole * remainder


def _forced_candidates(limit: int, context: int, query: int, block: int) -> int:
    """sum(min(limit, floor((context + j) / block) + 1) for j in range(query))."""
    if limit <= 0:
        return 0
    sequence = context + query
    max_blocks = _ceil_div(sequence, block)
    limit = min(limit, max_blocks)
    # (limit - v)_+ counts the layers t = v .. limit-1; the number of tokens
    # with v <= t is clamp(t*block - context, 0, query).  Only partial layers
    # exist below max_blocks, so this is a two-term arithmetic sum.
    first = context // block + 1
    last = min(limit - 1, _ceil_div(sequence, block) - 1)
    tail = 0
    if last >= first:
        count = last - first + 1
        tail = block * (first + last) * count // 2 - context * count
    return limit * query - tail


def _score_tile_stats(g: _IndexTopk, tile: int) -> dict[str, int]:
    """Physical work of one source score CTA (pid_q=tile, one request/head)."""
    rows = min(SCORE_Q_TILE, g.query - tile * SCORE_Q_TILE)
    q_start = g.context + tile * SCORE_Q_TILE
    hi = min(g.sequence_length, q_start + SCORE_Q_TILE)
    blocks = _ceil_div(hi, g.score_block)
    masked = max(0, blocks - q_start // g.score_block)
    dtype_width = dtype_bytes(g.dtype)
    return {
        "rows": rows,
        "blocks": blocks,
        "masked_blocks": masked,
        "qk_flops": 2 * SCORE_Q_TILE * g.score_block * g.head_dim * blocks,
        "q_bytes": rows * g.head_dim * dtype_width,
        "k_bytes": blocks * g.score_block * g.head_dim * dtype_width,
        "block_table_bytes": blocks * 4,
        "store_bytes": rows * blocks * 4,
        "mask_lanes": SCORE_Q_TILE * g.score_block * masked,
        "max_lanes": SCORE_Q_TILE * (g.score_block - 1) * blocks,
        "addr_lanes": g.score_block * blocks + SCORE_Q_TILE,
    }


def _topk_token_stats(g: _IndexTopk, token: int) -> dict[str, int]:
    """Physical work of one source top-k CTA (pid_q=token, one request/head)."""
    visible = (g.context + token) // g.score_block + 1
    iterations = _ceil_div(visible, g.topk_block)
    extra = iterations - 1
    n = g.log2_tile
    first_stages = n * (n + 1) // 2
    later_stages = n * (n - 1) // 2 + 2 * n
    stages = first_stages + extra * later_stages
    padded = g.topk_block * iterations
    return {
        "visible": visible,
        "iterations": iterations,
        "extra_iterations": extra,
        "sort_stages": stages,
        "cas_lanes": g.topk_block * stages,
        "merge_lanes": g.topk_block * extra,
        "padded_lanes": padded,
        "prepare_ops": TOPK_PREPARE_OPS * padded,
        "sort_ops": CAS_OPS * g.topk_block * stages,
        "merge_ops": MERGE_COMBINE_OPS * g.topk_block * extra,
        "init_ops": INIT_FILL_OPS * g.topk_block,
        "extract_ops": (
            EXTRACT_MUL_OPS * g.topk_block
            + EXTRACT_ADD_OPS * (g.topk_block - g.block_size_t)
            + 2 * g.block_size_t
            + STORE_ADDR_OPS * g.topk_block
        ),
        "score_read_bytes": visible * 4,
        "selected_index_bytes": g.topk * 4,
    }


def _counts(g: _IndexTopk) -> dict[str, Any]:
    """Aggregate physical source work for the selected pairing, in declared units."""
    dtype_width = dtype_bytes(g.dtype)
    score_block, q_tile = g.score_block, SCORE_Q_TILE
    requests, heads = g.batch, g.heads

    qk_flops = 0
    q_bytes = 0
    k_bytes = 0
    block_table_bytes = 0
    score_store_bytes = 0
    score_mask_lanes = 0
    score_max_lanes = 0
    score_addr_lanes = 0
    for tile in range(g.score_tiles):
        rows = min(q_tile, g.query - tile * q_tile)
        q_start = g.context + tile * q_tile
        hi = min(g.sequence_length, q_start + q_tile)
        blocks = _ceil_div(hi, score_block)
        # Blocks whose 128-position window still intersects this q tile.  The
        # source branch is `if q_start < i + BLOCK_SIZE_K` (lines 151-152).
        masked_blocks = max(0, blocks - q_start // score_block)
        qk_flops += 2 * q_tile * score_block * g.head_dim * blocks
        q_bytes += rows * g.head_dim * dtype_width
        k_bytes += blocks * score_block * g.head_dim * dtype_width
        block_table_bytes += blocks * 4
        score_store_bytes += rows * blocks * 4
        score_mask_lanes += q_tile * score_block * masked_blocks
        score_max_lanes += q_tile * (score_block - 1) * blocks
        score_addr_lanes += score_block * blocks + q_tile

    qk_flops *= requests * heads
    q_bytes *= requests * heads
    k_bytes *= requests * heads
    block_table_bytes *= requests * heads
    score_store_bytes *= requests * heads
    score_mask_lanes *= requests * heads
    score_max_lanes *= requests * heads
    score_addr_lanes *= requests * heads

    # Per-query-token visible scores: v_j = floor((context + j) / 128) + 1.
    # The top-k loop is ceil(v_j / topk_block) padded iterations.  Count the
    # tokens in each iteration tail without materializing one event per token:
    # a token reaches tile t when context + j >= 128 * t * topk_block.
    total_iterations = 0
    tile_index = 0
    while True:
        reached = g.query - max(
            0, tile_index * g.topk_block * score_block - g.context
        )
        if reached <= 0:
            break
        total_iterations += reached
        tile_index += 1
    extra_iterations = total_iterations - g.query
    topk_padded_lanes = g.topk_block * total_iterations

    n = g.log2_tile
    first_iteration_stages = n * (n + 1) // 2
    # Lines 252-264: build network, one descending merge, top-half combine,
    # and a second descending merge for every iteration after the first.
    later_iteration_stages = n * (n - 1) // 2 + 2 * n
    sort_stages = (first_iteration_stages * g.query
                   + later_iteration_stages * extra_iterations)
    cas_lanes = g.topk_block * sort_stages
    merge_lanes = g.topk_block * extra_iterations

    valid_reads = (
        _floor_sum(g.context + g.query, score_block)
        - _floor_sum(g.context, score_block)
        + g.query
    )
    selected_valid = g.query + sum(
        max(0, g.query - max(0, limit * score_block - g.context))
        for limit in range(1, g.topk)
    )

    topk_prepare_ops = TOPK_PREPARE_OPS * topk_padded_lanes
    topk_sort_ops = CAS_OPS * cas_lanes
    topk_merge_ops = MERGE_COMBINE_OPS * merge_lanes
    topk_init_ops = INIT_FILL_OPS * g.topk_block * g.query
    # Reshape multiply + reduction add over the padded tile, then the int32
    # store mask/select over block_size_t, plus the topk_mask compare.
    topk_extract_ops = (
        EXTRACT_MUL_OPS * g.topk_block
        + EXTRACT_ADD_OPS * (g.topk_block - g.block_size_t)
        + 2 * g.block_size_t
        + STORE_ADDR_OPS * g.topk_block
    ) * g.query
    score_vector_ops = (
        SCORE_MASK_OPS * score_mask_lanes
        + SCORE_MAX_OPS * score_max_lanes
        + score_addr_lanes
    )

    output_bytes = g.total_q * heads * g.topk * 4
    score_read_bytes = requests * heads * valid_reads * 4
    reads = q_bytes + k_bytes + block_table_bytes + score_read_bytes
    writes = score_store_bytes + output_bytes

    programs = requests * g.score_tiles * heads
    topk_programs = requests * g.query * heads
    return {
        "score_tiles": g.score_tiles,
        "max_blocks": g.max_blocks,
        "score_stride": g.score_stride,
        "block_size_t": g.block_size_t,
        "score_programs": programs,
        "topk_programs": topk_programs,
        "score_qk_flops": qk_flops,
        "score_q_bytes": q_bytes,
        "score_k_bytes": k_bytes,
        "score_block_table_bytes": block_table_bytes,
        "score_store_bytes": score_store_bytes,
        "score_read_bytes": score_read_bytes,
        "score_mask_lanes": score_mask_lanes,
        "score_max_lanes": score_max_lanes,
        "score_addr_lanes": score_addr_lanes,
        "score_vector_ops": score_vector_ops,
        "topk_iterations": total_iterations * requests * heads,
        "topk_extra_iterations": extra_iterations * requests * heads,
        "topk_padded_lanes": topk_padded_lanes * requests * heads,
        "topk_sort_stages": sort_stages * requests * heads,
        "topk_cas_lanes": cas_lanes * requests * heads,
        "topk_merge_lanes": merge_lanes * requests * heads,
        "topk_first_iteration_stages": first_iteration_stages,
        "topk_later_iteration_stages": later_iteration_stages,
        "topk_prepare_ops": topk_prepare_ops * requests * heads,
        "topk_sort_ops": topk_sort_ops * requests * heads,
        "topk_merge_ops": topk_merge_ops * requests * heads,
        "topk_init_ops": topk_init_ops * requests * heads,
        "topk_extract_ops": topk_extract_ops * requests * heads,
        "vector_ops": (
            score_vector_ops
            + (topk_prepare_ops + topk_sort_ops + topk_merge_ops
               + topk_init_ops + topk_extract_ops) * requests * heads
        ),
        "score_causal_mask_lanes": score_mask_lanes,
        "topk_causal_mask_lanes": topk_padded_lanes * requests * heads,
        "topk_init_mask_lanes": topk_padded_lanes * requests * heads,
        "topk_local_mask_lanes": topk_padded_lanes * requests * heads,
        "visible_candidates": valid_reads * requests * heads,
        "init_forced_candidates": (
            _forced_candidates(g.init_blocks, g.context, g.query, score_block)
            * requests * heads
        ),
        "local_forced_candidates": (
            _forced_candidates(g.local_blocks, g.context, g.query, score_block)
            * requests * heads
        ),
        "valid_visible_scores": valid_reads * requests * heads,
        "selected_valid_indices": selected_valid * requests * heads,
        "selected_padded_indices": g.total_q * heads * g.topk,
        "selected_valid_bytes": selected_valid * requests * heads * 4,
        "selected_index_bytes": output_bytes,
        "score_workspace_bytes": heads * g.total_q * g.score_stride * 4,
        "hbm_read_bytes": reads,
        "mte2_bytes": reads,
        "mte3_bytes": writes,
        "onchip_score_tile_bytes": (
            q_tile * g.head_dim * dtype_width
            + score_block * g.head_dim * dtype_width
            + q_tile * score_block * 4
            + q_tile * 4
        ),
        "onchip_sort_bytes_per_program": (
            7 * g.topk_block + g.block_size_t
        ) * 4,
    }


def _dispatch(g: _IndexTopk) -> dict[str, Any]:
    return {
        "implementation": IMPLEMENTATION,
        "mask_init": g.mask_init,
        "mask_local": g.mask_local,
        "mask_branch_selection": (
            "raw_constexpr_branch" if (g.mask_init or g.mask_local)
            else "wrapper_fixed_false_false"
        ),
        "wrapper_mask_init": False,
        "wrapper_mask_local": False,
        "score_kernel": SCORE_KERNEL,
        "topk_kernel": TOPK_KERNEL,
        "kernel_phases": [SCORE_PHASE, TOPK_PHASE],
        "block_size_q": g.block_size_q,
        "score_block": g.score_block,
        "score_q_tile": SCORE_Q_TILE,
        "topk_block": g.topk_block,
        "block_size_t": g.block_size_t,
        "dtype": g.dtype,
        "launch_count": 2,
        "topk_block_selection": "explicit_case_tile",
        "autotune_configs": list(TOPK_TILES),
        "init_mask_semantics": (
            "mask_minus_1e29" if g.mask_init else "force_1e30"
        ),
        "local_mask_semantics": (
            "mask_minus_1e28" if g.mask_local else "force_1e29"
        ),
        "decode_split_k_path": False,
    }
