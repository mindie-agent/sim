"""One complete quantized lightning-indexer invocation.

[Role] Keep QK, weighted head reduction and top-k on one common-engine kernel.
[Invariants] Query/key quantization, rotary transforms, scale producers and host
  metadata remain caller dependencies. The source performs INT8 QK followed by
  FP16 head reduction; per-head scores remain on chip. A resource envelope does
  not identify the captured tiling, owner split, cache residency or binary.
[Timeouts & Locks] O(requests) integer sums; no I/O or locks.
[Side-Effect Policy] Append one operator to the caller-owned builder only.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.schema.common import ModelingError
from .mechanism import OperatorPlan, PlannedRecipe, Work
from .operators.mixed_core import split_reads
from .program import KernelProgram, Stage
from .sparse_indexer_pipeline import (
    INDEXER_WAVE_ROWS, QUANT_LEDGER, Arch35QuantIndexerRecipe,
    arch35_indexer_work, indexer_waves, key_blocks,
)


IMPLEMENTATION = "cann_quant_lightning_indexer_v2_complete"
RECIPE = "quant_lightning_indexer_v2"
#: CANN 9.2.1 arch35 FP8 `quant_mode=1` lowering of the same operator family.
ARCH35_IMPLEMENTATION = "aclnn_quant_lightning_indexer_v2_arch35"
ARCH35_RECIPE = "aclnn_quant_lightning_indexer_v2_arch35"
ARCH35_INDEX_DTYPE = "fp8_e4m3"
ARCH35_SCALE_DTYPE = "fp16"
ARCH35_HEAD_COUNTS = (32, 64)
ARCH35_TOPK_MAX = 2048
ARCH35_COMPRESSION_MAX = 128
FP32_BYTES = 4
INT32_BYTES = 4
SOURCE = {
    "repository": "ops-transformer",
    "revision": "a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
    "cube": {
        "path": "attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_cube_arch22.h",
        "sha256": "87a50714932a4e8e7bd65a230c7bb32476456ac4aaa029e63c1f660244b9c4a6",
        "lines": [492, 577],
    },
    "vector": {
        "path": "attention/quant_lightning_indexer_v2/op_kernel/arch22/quant_lightning_indexer_v2_service_vector_arch22.h",
        "sha256": "d2ba785ac291998012c3b33671d708ad184970bd5ff896f79aa74dc319ae2906",
        "lines": [271, 420],
    },
    # Local matching operator checkout used for the review: its arch22 cube and
    # vector hashes equal the two entries above.  The host tiler, not a sampled
    # shape table, supplies the named implementation's legal domain.
    "host_tiler": {
        "path": "csrc/attention/quant_lightning_indexer_v2/op_host/quant_lightning_indexer_v2_tiling.cpp",
        "sha256": "c87c4e5ffd04802bf22f54d09131f9d54f8017b706854b936cfdc144748f33ec",
        "lines": [339, 355, 928, 955],
        "scope": ("local matching arch22 checkout whose cube/vector hashes equal "
                  "the SOURCE entries above; not a captured binary identity"),
    },
    "capture_binary_identity": "not_established",
}


def recipes():
    return {RECIPE: PlannedRecipe(_build),
            ARCH35_RECIPE: PlannedRecipe(_build_arch35)}


def _build(op, geometry):
    ledger = dict(op.metadata)["quant_indexer_work"]
    work = split_reads(Work(dict(op.matrix_flops), {
        "hbm": op.hbm_bytes,
        "mte1": ledger["mte1_bytes"],
        "mte2": ledger["aic_read_bytes"] + ledger["aiv_read_bytes"],
        "mte3": op.mte3_bytes,
        "vector": op.vector_ops, "fixpipe": ledger["fixpipe_bytes"],
    }), ledger["aic_read_bytes"], geometry.resource_units)
    return OperatorPlan(work, lambda context: KernelProgram((
        Stage("complete_indexer_service", context.work),)), {
            "execution_abstraction": "complete_kernel_resource_envelope",
            "native_dispatch_reconstructed": False,
            "cross_owner_topk_traffic": "not_selected_without_owner_metadata",
        })


def _owned_spans(view):
    """Owned query spans. Empty ``query_ranges`` is the whole local query."""
    ranges = view.query_ranges
    if not ranges:
        return ((0, view.q_len),)
    return tuple((start, end) for start, end in ranges if end > start)


# Declared native arch22 implementation domain (the SOURCE cube/vector files
# are arch22).  These are the host tiler's actual gates, not a sampled tuple:
# N1/N2 = 64, head dim 128, sparse_count <= 2048, cmp_ratio <= 128 and a power
# of two.
NATIVE_HEAD_DIM = 128
NATIVE_HEAD_COUNT = 64
NATIVE_TOPK_MAX = 2048
NATIVE_COMPRESSION_MAX = 128
NATIVE_BLOCK_ROWS_FACTOR = 16
NATIVE_BLOCK_ROWS_MAX = 1024


def _check_representation(plan, quant, parallel):
    """Validate the representation and single-call boundary shared by both paths."""
    if (parallel.cp != 1 or parallel.pcp < 1 or plan.candidate_filter
            or plan.index_kpool != 1
            or quant.index_dtype != "int8" or quant.index_scale_dtype != "fp16"):
        raise ModelingError("quant_lightning_indexer_v2_complete_domain_mismatch")
    for value in (plan.index_heads, plan.index_dim,
                  plan.index_topk, plan.compress_ratio):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelingError("quant_lightning_indexer_v2_complete_domain_mismatch")


def _check_domain(plan, quant, parallel, block_size=None):
    """Validate the selected arch22 native implementation's legal domain.

    The generic analytic path remains available for other legal geometry; this
    check only guards the named ``VllmAscendQuantLightningIndexerV2`` selection.
    """
    _check_representation(plan, quant, parallel)
    if (plan.index_dim != NATIVE_HEAD_DIM
            or plan.index_heads != NATIVE_HEAD_COUNT
            or plan.index_topk > NATIVE_TOPK_MAX
            or plan.compress_ratio > NATIVE_COMPRESSION_MAX
            or (plan.compress_ratio & (plan.compress_ratio - 1))):
        raise ModelingError("quant_lightning_indexer_v2_complete_domain_mismatch")
    if block_size is not None:
        if (type(block_size) is not int or block_size <= 0
                or block_size % NATIVE_BLOCK_ROWS_FACTOR
                or block_size > NATIVE_BLOCK_ROWS_MAX):
            raise ModelingError("quant_lightning_indexer_v2_complete_domain_mismatch")


def _visible_rows(spans, ratio):
    """Rows with at least one visible compressed key (spans are (q, context))."""
    return sum(max(0, q - max(0, ratio - c - 1)) for q, c in spans)


def _selection_comparisons(pairs, topk, visible_rows):
    """TopK selection work; Top1 is a maximum reduction, not zero work."""
    if topk == 1:
        return max(0, pairs - visible_rows)
    return pairs * math.log2(topk)


def generic_work(*, spans, heads, dim, topk, ratio):
    """Representation-agnostic indexer work for legal unsampled geometry.

    This does not claim the arch22 native binary; it lets the analytic path
    predict arbitrary positive head/width/TopK/compression combinations that
    the source formula can express.
    """
    normalized = []
    for q, c in spans:
        if (isinstance(q, bool) or not isinstance(q, int) or q < 1
                or isinstance(c, bool) or not isinstance(c, int) or c < 0):
            raise ModelingError("quant_indexer_generic_geometry_invalid")
        normalized.append((q, c))
    for name, value in (("heads", heads), ("dim", dim),
                        ("topk", topk), ("ratio", ratio)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ModelingError("quant_indexer_generic_geometry_invalid")
    pairs = sum(_floor_sum(q, ratio, 1, c + 1) for q, c in normalized)
    visible = _visible_rows(normalized, ratio)
    return {
        "query_key_pairs": pairs,
        "qk_int8_flops": 2 * pairs * heads * dim,
        "weighted_head_reduce_fp16_flops": 2 * pairs * heads * 16,
        "topk_comparisons": _selection_comparisons(pairs, topk, visible),
        "selection_output_bytes": sum(q for q, _ in normalized) * topk * 4,
    }


def _build_arch35(op, geometry):
    """arch35 FP8 (`quant_mode=1`) block pipeline over the 32 AIC owners."""
    ledger = dict(dict(op.metadata)[QUANT_LEDGER])
    plan = Arch35QuantIndexerRecipe().build(op, geometry)
    metadata = {
        "execution_abstraction": "owner_wave_block_pipeline",
        "native_dispatch_reconstructed": False,
        "cross_owner_topk_traffic": "per_token_rolling_merge_declared",
        "waves": ledger["waves"],
        "key_blocks": ledger["blocks"],
    }
    return OperatorPlan(plan.work, plan.program, metadata=metadata)


def _check_arch35_domain(plan, quant, parallel, keys):
    if (parallel.cp != 1 or parallel.pcp < 1 or plan.candidate_filter
            or plan.index_kpool != 1
            or quant.index_dtype != ARCH35_INDEX_DTYPE
            or quant.index_scale_dtype != ARCH35_SCALE_DTYPE):
        raise ModelingError("quant_lightning_indexer_v2_arch35_domain_mismatch")
    if (plan.index_heads not in ARCH35_HEAD_COUNTS
            or plan.index_dim != NATIVE_HEAD_DIM
            or plan.index_topk > ARCH35_TOPK_MAX
            or plan.compress_ratio > ARCH35_COMPRESSION_MAX
            or (plan.compress_ratio & (plan.compress_ratio - 1))):
        raise ModelingError("quant_lightning_indexer_v2_arch35_domain_mismatch")
    if keys < 1:
        raise ModelingError("quant_lightning_indexer_v2_arch35_domain_mismatch")


def _view_completed_keys(view, ratio):
    """Compressed keys one request's populated cache holds.

    Causal completion counts the appended query tokens, and a compressed key
    exists only after a whole ``ratio`` group completes. ``ratio <= 1`` is the
    dense non-compressed indexer: every context and query token is a key.
    """
    total = (max(0, int(getattr(view, "context", 0) or 0))
             + max(0, int(getattr(view, "q_len", 0) or 0)))
    if ratio <= 1:
        return total
    return total // ratio


def _candidate_keys(views, ratio):
    """Completed keys of the arch35 scan and of its physical cache payload.

    The scan walks the largest populated cache in one owner wave; the payload
    is every distinct cache row. Each private request owns its own rows, so
    two requests with 1,024 keys each fetch 2,048 rows, not the larger of the
    two. Requests naming the same shared prefix key read one physical union.
    """
    shared: dict[str, int] = {}
    private = 0
    largest = 0
    for view in views:
        keys = _view_completed_keys(view, ratio)
        largest = max(largest, keys)
        context = max(0, int(getattr(view, "context", 0) or 0))
        prefix = max(0, min(int(getattr(view, "prefix_tokens", 0) or 0), context))
        token = str(getattr(view, "prefix_cache_key", "") or "")
        if prefix and token:
            shared_keys = min(keys, prefix if ratio <= 1 else (prefix // ratio))
            shared[token] = max(shared.get(token, 0), shared_keys)
            private += max(0, keys - shared_keys)
        else:
            private += keys
    payload = sum(shared.values()) + private
    scan = max(largest, max(shared.values(), default=0))
    return scan, max(1, payload)


def declare_arch35(builder, *, deps, views, plan, quant, parallel,
                   block_size=None):
    """Declare the CANN 9.2.1 arch35 FP8 (`quant_mode=1`) indexer planner.

    `W_T = ceil(T / C_AIC)` owners each walk `ceil(K / 1024)` valid key blocks:
    one FP8 Cube dot of `2 * H * 128 * valid_keys`, then the `H`-independent
    rolling per-token AIV TopK merge of that block. Only the final INT32 token
    indices reach GM; the score plane stays on chip.
    """
    del block_size
    ratio = plan.compress_ratio
    keys, payload_keys = _candidate_keys(views, ratio)
    _check_arch35_domain(plan, quant, parallel, keys)
    tokens = sum(view.q_len for view in views)
    heads, width = plan.index_heads, plan.index_dim
    width = width if isinstance(width, int) and width > 0 else NATIVE_HEAD_DIM
    key_row_bytes = int(index_row_bytes_from(builder.model, width, quant))
    blocks = key_blocks(keys)
    ledger = {
        "query_rows": int(tokens), "wave_rows": INDEXER_WAVE_ROWS,
        "waves": indexer_waves(tokens), "blocks": blocks,
        "heads": heads, "width": width, "keys": blocks * 1024,
        "completed_keys": keys, "payload_keys": payload_keys,
        "key_row_bytes": key_row_bytes,
        "output_bytes": float(tokens) * plan.index_topk * INT32_BYTES,
    }
    work = arch35_indexer_work(ledger)
    result = builder.work(ARCH35_RECIPE, deps, (tokens, heads, width),
        family="indexer", dtype=ARCH35_INDEX_DTYPE,
        metadata=((QUANT_LEDGER, ledger),
                  ("native_source", {**SOURCE, "arch": "arch35", "lines": [1, 1]}),
                  ("native_tensor_view", {
                      "kind": "aclnn_quant_lightning_indexer_v2_arch35",
                      "query_shape": (tokens, heads, width), "query_dtype": ARCH35_INDEX_DTYPE,
                      "cache_dtype": ARCH35_INDEX_DTYPE,
                      "cache_head_shape": (1, width), "scale_dtype": ARCH35_SCALE_DTYPE,
                      "output_shape": (tokens, 1, plan.index_topk), "output_dtype": "int32",
                      "layout": "ND", "quant_mode": 1, "compression_ratio": ratio,
                      "visible_keys": keys}),
                  ("native_implementation", {
                      "native_kind": ARCH35_IMPLEMENTATION,
                      "selection": ARCH35_IMPLEMENTATION,
                      "conditions": {"submitted_kernels": 1, "quant_mode": 1,
                                     "wave_rows": INDEXER_WAVE_ROWS,
                                     "key_blocks": blocks,
                                     "topk_merge_heads_independent": True}})))
    builder.ops[-1] = replace(work.apply(builder.ops[-1]),
                              source_boundary="kernel",
                              coverage="arch35_fp8_wave_block_indexer_program")
    return result


def index_row_bytes_from(model, width, quant):
    """Declared index cache row width for the resolved index representation."""
    declared = int(round(index_row_bytes(model, width)))
    if declared < width:
        from mindie_sim.schema.common import dtype_bytes
        declared = int(width * dtype_bytes(quant.index_dtype))
    return declared


def declare(builder, *, deps, views, plan, quant, parallel, block_size=None):
    """Declare the supported complete boundary without inferring native owners.

    ``block_size`` is the actual KV/index key block size from the runtime. The
    host tiler requires a multiple of 16 in (0, 1024]; T14 must pass it through
    the normal builder option when available. When absent, the domain is not
    silently assumed: the ledger records that the runtime parameter was not
    supplied.
    """
    if quant.index_dtype == ARCH35_INDEX_DTYPE:
        return declare_arch35(builder, deps=deps, views=views, plan=plan,
                              quant=quant, parallel=parallel,
                              block_size=block_size)
    _check_domain(plan, quant, parallel, block_size)
    tokens = sum(view.q_len for view in views)
    # Causal compressed-key counts follow each owned span. PCP chunks pass
    # query_ranges; padding and cache allocation do not create index work.
    ratio = plan.compress_ratio
    spans = tuple((view, start, end) for view in views for start, end in _owned_spans(view))
    pairs = sum(_floor_sum(end - start, ratio, 1, view.context + start + 1)
                for view, start, end in spans)
    heads, width = plan.index_heads, plan.index_dim
    visible_rows = _visible_rows(
        tuple((end - start, view.context + start) for view, start, end in spans), ratio)
    topk_comparisons = _selection_comparisons(pairs, plan.index_topk, visible_rows)
    sentinel_fills = sum(
        short * plan.index_topk - _floor_sum(short, ratio, 1, view.context + start + 1)
        for view, start, end in spans
        if (short := max(0, min(end - start, plan.index_topk * ratio - (view.context + start + 1))))
    )
    # QueryNd2Nz, ProcessVec0 and WeightDmaCopy all execute under the same
    # isFirstS2InnerLoop guard for queries with at least one compressed key.
    active_query_rows = sum(
        end - start for view, start, end in spans if (view.context + end) // ratio > 0
    )
    query_bytes = active_query_rows * heads * width
    key_bytes = pairs * width
    scale_weight_bytes = active_query_rows * heads * 4 + pairs * 2
    # Source Vec0 broadcasts (weight * query_scale) over the M=16 matrix
    # axis once per active query owner on isFirstS2InnerLoop, and writes/reads
    # that small operand through GM. Only the reduced FP32 score is materialized
    # in GM per pair, not the 64-head INT32 QK result.
    weight_workspace_bytes = active_query_rows * heads * 16 * 2
    score_workspace_bytes = pairs * 4
    output_bytes = tokens * plan.index_topk * 4
    aic_read_bytes = query_bytes + key_bytes + weight_workspace_bytes
    aiv_read_bytes = scale_weight_bytes + score_workspace_bytes
    read_bytes = aic_read_bytes + aiv_read_bytes
    mte3_write_bytes = weight_workspace_bytes + output_bytes
    # ProcessQk loads query/key from L1 to L0A/L0B; ProcessWs loads the FP16
    # score tensor (written to L1 by FixpSToL1) and broadcasted FP16 weights.
    mte1_bytes = (aic_read_bytes + 2 * pairs * heads) if pairs else 0
    vector = active_query_rows * heads * 17 + pairs * 2 + topk_comparisons + sentinel_fills
    matrix = {"int8": 2 * pairs * heads * width,
              "fp16": 2 * pairs * heads * 16}
    ledger = {
        "query_key_pairs": pairs, "query_rows": tokens,
        "sequence_partition": tuple((view.q_len, view.context) for view in views),
        "qk_int8_flops": matrix["int8"],
        "weighted_head_reduce_fp16_flops": matrix["fp16"],
        "head_reduce_physical_m": 16,
        "dequant_relu": "source_fixpipe_before_fp16_head_reduce",
        "fixpipe_bytes": 2 * pairs * heads + 4 * pairs,
        "mte1_bytes": mte1_bytes,
        "aic_read_bytes": aic_read_bytes, "aiv_read_bytes": aiv_read_bytes,
        "query_bytes": query_bytes, "key_bytes": key_bytes,
        "scale_and_weight_bytes": scale_weight_bytes,
        "weight_workspace_read_write_bytes": 2 * weight_workspace_bytes,
        "reduced_score_workspace_read_write_bytes": 2 * score_workspace_bytes,
        "selection_output_bytes": output_bytes,
        "sentinel_fill_elements": sentinel_fills,
        "topk_comparisons": topk_comparisons,
        "visible_query_rows": visible_rows,
        "native_domain": {"head_dim": NATIVE_HEAD_DIM,
                          "head_count": NATIVE_HEAD_COUNT,
                          "topk_max": NATIVE_TOPK_MAX,
                          "compression_max": NATIVE_COMPRESSION_MAX,
                          "block_rows_factor": NATIVE_BLOCK_ROWS_FACTOR,
                          "block_rows_max": NATIVE_BLOCK_ROWS_MAX},
        "key_block_size": block_size,
        "key_block_size_source": ("runtime" if block_size is not None
                                  else "not_supplied"),
        "traffic_scope": "query_and_key_once_per_query; native_cache_reloads_unresolved",
        "topk_scope": "logical_comparisons; cross_owner_partial_lists_unresolved",
    }
    result = builder.work(RECIPE, deps, (tokens, 1, plan.index_topk),
        flops=sum(matrix.values()), vector=vector,
        read=read_bytes, write=mte3_write_bytes,
        family="indexer", dtype="int8", metadata=(
            ("native_source", SOURCE), ("quant_indexer_work", ledger),
            ("mte1_bytes", mte1_bytes), ("fixpipe_bytes", ledger["fixpipe_bytes"]),
            ("native_tensor_view", {"kind": "quant_lightning_indexer_v2",
                "query_shape": (tokens, heads, width), "query_dtype": "int8",
                "cache_dtype": "int8", "cache_head_shape": (1, width),
                "scale_dtype": "fp16", "weight_dtype": "fp16",
                "output_shape": (tokens, 1, plan.index_topk), "output_dtype": "int32",
                "layout": "ND", "compression_ratio": ratio,
                "key_block_size": block_size}),
            ("native_implementation", {"native_kind": "VllmAscendQuantLightningIndexerV2",
                "selection": IMPLEMENTATION, "conditions": {"submitted_kernels": 1,
                    "source_boundary": "qk_weighted_head_reduce_topk",
                    "upstream_quantization_and_host_metadata": "outside"}}),
        ))
    # Reduced scores are written to GM by Fixpipe (FixpResToGm), so they
    # contribute to HBM and Fixpipe traffic without consuming MTE3 stores.
    builder.ops[-1] = replace(builder.ops[-1], matrix_flops=tuple(matrix.items()),
        hbm_bytes=read_bytes + mte3_write_bytes + score_workspace_bytes,
        source_boundary="kernel", coverage="complete_quantized_indexer_resource_envelope")
    return result
