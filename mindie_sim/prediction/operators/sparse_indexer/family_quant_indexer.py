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
from mindie_sim.schema.common import ModelingError
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.vector.mixed_core import split_reads
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, Stage
from .sparse_indexer_pipeline import (
    INDEXER_WAVE_ROWS, QUANT_LEDGER, Arch35QuantIndexerRecipe,
    arch35_indexer_work, indexer_waves, key_blocks,
)


FP32_SCALE_WEIGHT_BYTES = 4
FP32_SCORE_BYTES = 4
HEAD_REDUCTION_CUBE_TILE_ROWS = 16
INDEX_ELEMENT_BYTES = 4
QUANT_WEIGHT_OPS_PER_HEAD = 17


IMPLEMENTATION = "cann_quant_lightning_indexer_v2_complete"
RECIPE = "quant_lightning_indexer_v2"
#: CANN 9.2.1 arch35 FP8 `quant_mode=1` lowering of the same operator family.
ARCH35_IMPLEMENTATION = "aclnn_quant_lightning_indexer_v2_arch35"
ARCH35_RECIPE = "aclnn_quant_lightning_indexer_v2_arch35"
ARCH35_INDEX_DTYPE = "fp8_e4m3"
ARCH35_SCALE_DTYPE = "fp32"
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
    aic_read = float(ledger["aic_read_bytes"])
    aiv_read = float(ledger["aiv_read_bytes"])
    work = split_reads(Work(dict(op.matrix_flops), {
        "hbm": op.hbm_bytes,
        "mte1": ledger["mte1_bytes"],
        "mte2": aic_read + aiv_read,
        "mte3": op.mte3_bytes,
        "vector": op.vector_ops, "fixpipe": ledger["fixpipe_bytes"],
    }), aic_read, geometry.resource_units)
    query_rows = max(0, int(ledger.get("query_rows", 0)))
    aic_units = max(1, int(getattr(geometry, "compute_units", 0) or 1))
    active_owners = min(aic_units, query_rows) if query_rows > 0 else 0

    weight_ws = float(ledger.get("weight_workspace_read_write_bytes", 0.0)) / 2.0
    score_ws = float(ledger.get("reduced_score_workspace_read_write_bytes", 0.0)) / 2.0
    scale_w_read = max(0.0, aiv_read - score_ws)
    score_aiv_read = max(0.0, aiv_read - scale_w_read)
    mte1_total = float(ledger["mte1_bytes"])
    qk_mte1 = min(mte1_total, float(ledger.get("query_bytes", 0.0)) + float(ledger.get("key_bytes", 0.0)))
    reduce_mte1 = max(0.0, mte1_total - qk_mte1)
    fixpipe_total = float(ledger["fixpipe_bytes"])
    reduce_fixpipe = min(fixpipe_total, score_ws)
    qk_fixpipe = max(0.0, fixpipe_total - reduce_fixpipe)
    mte3_total = float(op.mte3_bytes)
    reduce_mte3 = min(mte3_total, weight_ws)
    topk_mte3 = max(0.0, mte3_total - reduce_mte3)
    hbm_total = float(op.hbm_bytes)
    load_hbm = min(hbm_total, aic_read + scale_w_read)
    rem_hbm = max(0.0, hbm_total - load_hbm)
    reduce_hbm = min(rem_hbm, reduce_mte3 + reduce_fixpipe)
    topk_hbm = max(0.0, rem_hbm - reduce_hbm)
    vec_total = float(op.vector_ops)
    topk_vec = min(
        vec_total,
        float(ledger.get("topk_comparisons", 0.0))
        + float(ledger.get("sentinel_fill_elements", 0.0))
        + 2.0 * float(ledger.get("query_key_pairs", 0.0)),
    )
    reduce_vec = max(0.0, vec_total - topk_vec)
    matrix_map = dict(op.matrix_flops)
    int8_flops = float(matrix_map.get("int8", 0.0))
    fp16_flops = float(matrix_map.get("fp16", 0.0))

    def program(context):
        if active_owners <= 0:
            return KernelProgram((Stage("empty_quant_indexer", keep_empty=True),))
        has_split_mte2 = ("mte2_aic" in context.resource_names and "mte2_aiv" in context.resource_names)
        read_roles = ("mte2_aic", "mte2_aiv") if has_split_mte2 else ("mte2",)
        owned_roles = ("cube", "mte1", "fixpipe", "vector", "mte3", *read_roles)
        base_rows, rem_rows = divmod(query_rows, active_owners)
        owner_programs = []
        for owner in range(active_owners):
            my_rows = base_rows + (1 if owner < rem_rows else 0)
            share = float(my_rows) / float(query_rows)
            placement = context.owned(owner, aic_units, owned_roles)
            load_res = {"hbm": load_hbm * share}
            if has_split_mte2:
                load_res["mte2_aic"] = aic_read * share
                load_res["mte2_aiv"] = scale_w_read * share
            else:
                load_res["mte2"] = (aic_read + scale_w_read) * share
            topk_res = {
                "hbm": topk_hbm * share,
                "mte3": topk_mte3 * share,
                "vector": topk_vec * share,
                ("mte2_aiv" if has_split_mte2 else "mte2"): score_aiv_read * share,
            }
            owner_programs.append(KernelProgram((
                Stage("load_inputs", context.bind(Work(resources=load_res)), placement=placement),
                Stage("qk_int8_cube", context.bind(Work(
                    {"int8": int8_flops * share} if int8_flops else {},
                    {"mte1": qk_mte1 * share, "fixpipe": qk_fixpipe * share},
                )), after=("load_inputs",), placement=placement),
                Stage("head_reduce_fp16", context.bind(Work(
                    {"fp16": fp16_flops * share} if fp16_flops else {},
                    {"mte1": reduce_mte1 * share, "fixpipe": reduce_fixpipe * share,
                     "mte3": reduce_mte3 * share, "hbm": reduce_hbm * share,
                     "vector": reduce_vec * share},
                )), after=("qk_int8_cube",), placement=placement),
                Stage("topk_select", context.bind(Work(resources=topk_res)),
                      after=("head_reduce_fp16",), placement=placement),
            )))
        return ParallelProgram(tuple(owner_programs))

    return OperatorPlan(work, program, {
        "execution_abstraction": "tiled_cube_vector_service_pipeline",
        "active_aic_owners": active_owners,
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


ARCH35_PRESET = "npu_arch_3510"


def _build_arch35(op, geometry):
    """arch35 FP8 (`quant_mode=1`) block pipeline over the 32 AIC owners."""
    compute_arch = getattr(geometry, "compute_arch", None) or getattr(
        getattr(geometry, "datapath", None), "compute_arch", None
    )
    preset = str(getattr(compute_arch, "preset", "") or getattr(compute_arch, "arch_id", "") or "")
    if preset != ARCH35_PRESET:
        raise ModelingError("quant_lightning_indexer_v2_arch35_requires_arch35")
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


def _check_arch35_domain(plan, quant, parallel, keys, tokens=1):
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
    if keys < 0 or tokens < 1:
        raise ModelingError("quant_lightning_indexer_v2_arch35_domain_mismatch")


def _view_owned_ranges(view):
    """Owned query offsets; empty ``query_ranges`` is the whole local query."""
    ranges = tuple((int(start), int(end))
                   for start, end in (getattr(view, "query_ranges", ()) or ())
                   if int(end) > int(start))
    return ranges or ((0, max(0, int(getattr(view, "q_len", 0) or 0))),)


def _view_row_keys(view, ratio, noncausal=False):
    """Completed compressed keys visible to every owned query row."""
    context = max(0, int(getattr(view, "context", 0) or 0))
    rows = []
    for start, end in _view_owned_ranges(view):
        for offset in range(start, end):
            visible = (max(0, int(getattr(view, "kv_len", 0) or 0))
                       if noncausal else context + offset + 1)
            rows.append(max(0, visible) if ratio <= 1
                        else max(0, visible) // ratio)
    return tuple(rows)


def _view_completed_keys(view, ratio):
    """Largest completed key count any owned query row of this view sees."""
    return max(_view_row_keys(view, ratio), default=0)


def _candidate_keys(views, ratio, maxima=None):
    """Completed keys of the arch35 scan and of its physical cache payload.

    The scan walks the largest populated cache in one owner wave; the payload
    is every distinct cache row. Each private request owns its own rows, so
    two requests with 1,024 keys each fetch 2,048 rows, not the larger of the
    two. Requests naming the same shared prefix key read one physical union.
    """
    shared: dict[str, int] = {}
    private = 0
    largest = 0
    for index, view in enumerate(views):
        keys = (int(maxima[index]) if maxima is not None
                else _view_completed_keys(view, ratio))
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
    return scan, payload


def _wave_key_ranges(row_sets, views, ratio, shared):
    """Distinct interval fragments in each wave, in logical key coordinates."""
    waves, batch, prefixes, private = [], 0, {}, {}

    def flush():
        waves.append(tuple((0, n) for n in prefixes.values()) + tuple(private.values()))
        prefixes.clear()
        private.clear()

    for request, (values, view) in enumerate(zip(row_sets, views)):
        key = view.prefix_cache_key if shared else ""
        extent = min(view.prefix_tokens, view.context) // max(1, ratio) if key else 0
        for n in values:
            p = min(n, extent)
            if p:
                prefixes[key] = max(prefixes.get(key, 0), p)
            if n > p:
                private[request] = (extent, max(n, private.get(request, (0, 0))[1]))
            batch += 1
            if batch % INDEXER_WAVE_ROWS == 0:
                flush()
    if batch % INDEXER_WAVE_ROWS:
        flush()
    return tuple(waves)


def declare_arch35(builder, *, deps, views, plan, quant, parallel,
                   block_size=None, prefix_cache_mode="private"):
    """Declare the CANN 9.2.1 arch35 FP8 (`quant_mode=1`) indexer planner.

    `W_T = ceil(T / C_AIC)` owners each walk `ceil(K / 1024)` valid key blocks:
    one FP8 Cube dot of `2 * H * 128 * valid_keys`, then the `H`-independent
    rolling per-token AIV TopK merge of that block. Only the final INT32 token
    indices reach GM; the score plane stays on chip.
    """
    del block_size
    ratio = plan.compress_ratio
    noncausal = bool(getattr(plan, "noncausal_attention", False))
    row_sets = tuple(_view_row_keys(v, ratio, noncausal) for v in views)
    row_completed = tuple(n for values in row_sets for n in values)
    view_max = tuple(max(values, default=0) for values in row_sets)
    shared = prefix_cache_mode == "shared"
    keys = max(view_max, default=0)
    payload_keys = _candidate_keys(views, ratio, view_max)[1] if shared else sum(view_max)
    wave_ranges = _wave_key_ranges(row_sets, views, ratio, shared)
    tokens = len(row_completed)
    _check_arch35_domain(plan, quant, parallel, keys, tokens)
    heads, width = plan.index_heads, plan.index_dim
    width = width if isinstance(width, int) and width > 0 else NATIVE_HEAD_DIM
    from mindie_sim.catalog.cache_format import CacheFormat, cache_format
    declared = dict(builder.model.metadata).get("cache_representations", {})
    if "index_k" in declared and cache_format(builder.model, "index_k") != CacheFormat(
            ARCH35_INDEX_DTYPE, width, ARCH35_SCALE_DTYPE):
        raise ModelingError("quant_lightning_indexer_v2_arch35_domain_mismatch")
    key_row_bytes = width + FP32_BYTES
    blocks = key_blocks(keys) if keys > 0 else 0
    keys_padded = blocks * 1024
    runs = []
    for value in row_completed:
        if runs and runs[-1][1] == value:
            runs[-1] = (runs[-1][0] + 1, value)
        else:
            runs.append((1, value))
    ledger = {
        "query_rows": int(tokens), "wave_rows": INDEXER_WAVE_ROWS,
        "waves": indexer_waves(tokens), "blocks": blocks,
        "heads": heads, "width": width, "keys": keys_padded,
        "keys_padded": keys_padded,
        "completed_keys": keys, "payload_keys": payload_keys,
        "row_completed_keys": tuple(runs),
        "key_row_bytes": key_row_bytes, "wave_key_ranges": wave_ranges,
        "output_bytes": float(tokens) * plan.index_topk * INT32_BYTES,
    }
    work = arch35_indexer_work(ledger)
    result = builder.work(ARCH35_RECIPE, deps, (tokens, heads, width),
        family="indexer", dtype=ARCH35_INDEX_DTYPE,
        metadata=((QUANT_LEDGER, ledger),
                  ("native_source", {
                      "repository": "ops-transformer", "revision": "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
                      "relative_path": ("attention/quant_lightning_indexer_v2/op_kernel/arch35/"
                                        "quant_lightning_indexer_v2_kernel_arch35.h"),
                      "source_sha256": "37b78aab1ddb5eb0139e8a8c710610bbbccdc32359c4352ec302057153ed8945"}),
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


def declare(builder, *, deps, views, plan, quant, parallel, block_size=None,
            prefix_cache_mode="private"):
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
                              block_size=block_size, prefix_cache_mode=prefix_cache_mode)
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
    scale_weight_bytes = active_query_rows * heads * FP32_SCALE_WEIGHT_BYTES + pairs * 2
    # Source Vec0 broadcasts (weight * query_scale) over the M=16 matrix
    # axis once per active query owner on isFirstS2InnerLoop, and writes/reads
    # that small operand through GM. Only the reduced FP32 score is materialized
    # in GM per pair, not the 64-head INT32 QK result.
    weight_workspace_bytes = active_query_rows * heads * HEAD_REDUCTION_CUBE_TILE_ROWS * 2
    score_workspace_bytes = pairs * FP32_SCORE_BYTES
    output_bytes = tokens * plan.index_topk * INDEX_ELEMENT_BYTES
    aic_read_bytes = query_bytes + key_bytes + weight_workspace_bytes
    aiv_read_bytes = scale_weight_bytes + score_workspace_bytes
    read_bytes = aic_read_bytes + aiv_read_bytes
    mte3_write_bytes = weight_workspace_bytes + output_bytes
    # ProcessQk loads query/key from L1 to L0A/L0B; ProcessWs loads the FP16
    # score tensor (written to L1 by FixpSToL1) and broadcasted FP16 weights.
    mte1_bytes = (aic_read_bytes + 2 * pairs * heads) if pairs else 0
    vector = active_query_rows * heads * QUANT_WEIGHT_OPS_PER_HEAD + pairs * 2 + topk_comparisons + sentinel_fills
    matrix = {"int8": 2 * pairs * heads * width,
              "fp16": 2 * pairs * heads * HEAD_REDUCTION_CUBE_TILE_ROWS}
    ledger = {
        "query_key_pairs": pairs, "query_rows": tokens,
        "sequence_partition": tuple((view.q_len, view.context) for view in views),
        "qk_int8_flops": matrix["int8"],
        "weighted_head_reduce_fp16_flops": matrix["fp16"],
        "head_reduce_physical_m": HEAD_REDUCTION_CUBE_TILE_ROWS,
        "dequant_relu": "source_fixpipe_before_fp16_head_reduce",
        "fixpipe_bytes": 2 * pairs * heads + FP32_SCORE_BYTES * pairs,
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
        source_boundary="kernel", coverage="tiled_quantized_indexer_service_pipeline")
    return result
