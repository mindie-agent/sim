"""BF16 LightningIndexerV2 on the common resource engine.

[Role] Declare one causal compressed-key scan, weighted head reduction and TopK
  as a resource/phase program shared by independent-operator, native-input and
  graph consumers.
[Invariants] BF16 Q/K and FP32 head weights remain distinct from the INT8 QLI
  path; work follows declared tensors, compression, cache and TopK geometry and
  is never selected from shape/measurement tables. Native cache reload and
  partial TopK exchange are analytical mechanisms on the declared call
  boundary, not permanently deferred as an "unresolved lower bound".
[Timeouts & Locks] Closed-form or O(query rows / tile) integer arithmetic; no
  I/O, device contexts or locks.
[Side-Effect Policy] Return immutable operators and pure executable plans only.
"""
from dataclasses import replace
import math

from mindie_sim.catalog.attention_common import _floor_sum
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.operators import OpSpec
from ...cache_path import CacheGeometry as CacheGeometry, CacheState as CacheState
from ._bf16_access import (
    CACHE_LINE_BYTES as CACHE_LINE_BYTES,
    CACHE_TRANSFER_BYTES as CACHE_TRANSFER_BYTES,
    INDEX_ELEMENT_BYTES as _INDEX_ELEMENT_BYTES,
    LOCAL_LIST_ELEMENT_BYTES as LOCAL_LIST_ELEMENT_BYTES,
    QUERY_TILE_ROWS as QUERY_TILE_ROWS,
    _ceil_div, _key_access_plan as _key_access_plan, _panel_pairs as _panel_pairs,
    _panel_selection_ops as _panel_selection_ops, _sentinel_elements,
    _tile_sentinel as _tile_sentinel, _weighted_vector_ops,
)
from ._bf16_program import (
    _build as _build_program, _placement as _placement,
    _resident_allocations as _resident_allocations, _split_loads as _split_loads,
)
from mindie_sim.prediction.operators.sparse_indexer.indexer_source import select_indexer_source
from ...mechanism import Geometry, OperatorPlan as OperatorPlan, PlannedRecipe, Work as Work
from ..attention.common import RECTANGULAR_SEQUENCE_LAYOUTS, sequence_layout_spans, with_sequence_context
from ..contracts import OperatorPlanningContext
from mindie_sim.prediction.operators.vector.mixed_core import read_role as read_role
from ...ownership import unit_group as unit_group
from ...program import KernelProgram as KernelProgram, Stage as Stage


IMPLEMENTATION = "cann_bf16_lightning_indexer_v2_complete"
RECIPE = "bf16_lightning_indexer_v2"
SOURCE = {
    "caller_repository": "vllm-ascend",
    "caller_revision": "3c1b834495af94efb2af1f1a91da2fc8bd18441e",
    "caller_path": "vllm_ascend/models/qwen4_exp/lightning_indexer.py",
    "caller_lines": [155, 268],
    "binding_sha256": "9af7711bcabc3becc4dc4b97dce81551b1307830e8c1d28ae181c5f6c1880b53",
    "boundary": "one native BF16 LightningIndexerV2 MIX_AIC kernel",
    "dispatch_identity": "caller ABI and profiler boundary; native binary identity not captured",
}

_DEFAULT_COMPRESSION = 4
_DEFAULT_QUERY_HEADS = 4
_DEFAULT_HEAD_DIM = 128
_DEFAULT_TOPK = 512
_DEFAULT_KEY_PAGE_ROWS = 128
_QUERY_RANK = 3
_PAGED_KEY_CACHE_RANK = 4
_CAUSAL_MASK_MODE = 3
WEIGHT_DTYPES = frozenset({"fp32", "bf16", "fp16"})
_DTYPE_ALIASES = {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}


def recipes():
    return {RECIPE: PlannedRecipe(_build)}


def _build(op, context: Geometry | OperatorPlanningContext):
    if isinstance(context, OperatorPlanningContext):
        context = with_sequence_context(op, context)
        geometry = context.geometry
    else:
        geometry = context
    if not context.requests:
        return _build_program(op, geometry)

    layout = context.layout or "TND"
    spans = sequence_layout_spans(context.requests, padded_tokens=context.padded_tokens, layout=layout)
    padded_rows = (sum(request.physical_q_len for request in context.requests)
                   if layout.upper() in RECTANGULAR_SEQUENCE_LAYOUTS else sum(span[0] for span in spans))
    metadata = dict(op.metadata)
    if (raw := metadata.get("lightning_indexer_work")) is None:
        return _build_program(op, geometry)
    ledger = dict(raw)
    compression = ledger["compression"]
    lengths = tuple((q, kv - q) for q, kv, _, _, _ in spans)
    queries = sum(q for q, _ in lengths)
    pairs = sum(causal_pairs(q, c, compression) for q, c in lengths if q)
    heads, kv_heads = ledger["heads"], ledger["kv_heads"]
    flops = 2 * pairs * heads * ledger["head_dim"]
    view = context.admission_memory_view
    if view is not None and "l2" in geometry.buffers:
        capacity = view.effective_l2_bytes(geometry.rank or 0)
        declared = ledger.get("l2_capacity_bytes")
        ledger["l2_capacity_bytes"] = capacity if declared is None else min(declared, capacity)
    ledger.update(
        sequence_partition=lengths, query_rows=queries,
        weighted_head_reduce_ops=_weighted_vector_ops(pairs, heads, kv_heads),
        unique_key_bytes=sum((q + c) // compression for q, c in lengths)
        * kv_heads * ledger["head_dim"] * dtype_bytes("bf16"),
        head_weight_bytes=queries * heads * dtype_bytes(ledger["weight_dtype"]),
        metadata_read_bytes=len(spans) * (2 + ledger["block_table_columns"]) * _INDEX_ELEMENT_BYTES if queries else 0,
        selection_output_bytes=queries * kv_heads * ledger["topk"] * _INDEX_ELEMENT_BYTES,
        sequence_layout_spans=spans,
    )
    metadata["lightning_indexer_work"] = ledger
    bound = replace(op, matrix_flops=(("bf16", flops),), metadata=tuple(metadata.items()))
    plan = _build_program(bound, geometry)
    return replace(plan, metadata={**plan.metadata, "sequence_layout_spans": spans,
                                   "layout": layout, "padded_tokens": max(context.padded_tokens, padded_rows)})


def _integer(value, name, *, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ModelingError("bf16_lightning_indexer_invalid_" + name)
    return value


def _compressed_rows(query_rows, context, compression):
    """Materialized compressed K rows for one contiguous span."""
    return (int(context) + int(query_rows)) // int(compression)


def _physical_key_pages(compressed_rows, key_rows_per_page):
    return _ceil_div(compressed_rows, key_rows_per_page)


def causal_pairs(query_rows, context, compression=_DEFAULT_COMPRESSION):
    """Independent rows see floor((context + row + 1) / compression) keys."""
    _integer(query_rows, "query_rows", minimum=1)
    _integer(context, "context")
    _integer(compression, "compression", minimum=1)
    return _floor_sum(query_rows, compression, 1, context + 1)


def _gather_spans(query_rows, context, query_lengths, context_lengths):
    queries = tuple(query_lengths or ())
    contexts = tuple(context_lengths or ())
    if bool(queries) != bool(contexts):
        raise ModelingError("bf16_lightning_indexer_incomplete_sequence_lengths")
    if queries:
        if len(queries) != len(contexts):
            raise ModelingError("bf16_lightning_indexer_incomplete_sequence_lengths")
        spans = []
        for q, c in zip(queries, contexts):
            _integer(q, "query_length", minimum=1)
            _integer(c, "context_length")
            spans.append((q, c))
        total = sum(q for q, _ in spans)
        if query_rows is not None and _integer(query_rows, "query_rows", minimum=1) != total:
            raise ModelingError("bf16_lightning_indexer_query_rows_disagree_with_lengths")
        if context is not None and _integer(context, "context") != max(
                c for _, c in spans):
            raise ModelingError("bf16_lightning_indexer_context_disagrees_with_lengths")
        return tuple(spans)
    if query_rows is None or context is None:
        raise ModelingError("bf16_lightning_indexer_geometry_required")
    return ((_integer(query_rows, "query_rows", minimum=1),
             _integer(context, "context")),)


def _visible_rows(spans, compression):
    """Rows that own at least one visible compressed key."""
    return sum(max(0, q - max(0, compression - c - 1)) for q, c in spans)


def _topk_comparisons(pairs, topk, kv_heads, spans, compression):
    """Selection comparisons per (query row, KV head) domain.

    A heap selects TopK for K>1; Top1 is a maximum reduction over its visible
    candidates and therefore needs ``visible - 1`` comparisons, not zero.
    """
    if topk == 1:
        return max(0, pairs - _visible_rows(spans, compression)) * kv_heads
    return pairs * kv_heads * math.log2(topk)



def _descriptor(value, name, rank, dtypes, layouts):
    if isinstance(value, (tuple, list)) and len(value) == 3:
        shape, dtype, layout = value
    elif isinstance(value, dict):
        shape = value.get("shape")
        dtype = value.get("dtype")
        layout = value.get("layout", "ND")
    else:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_descriptor_invalid")
    if not isinstance(shape, (tuple, list)) or len(shape) != rank:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_rank")
    shape = tuple(shape)
    if any(isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0 for dim in shape):
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_shape")
    dtype = str(dtype).strip().lower()
    dtype = _DTYPE_ALIASES.get(dtype, dtype)
    if dtype not in dtypes:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_dtype")
    if str(layout) not in layouts:
        raise ModelingError("bf16_lightning_indexer_native_" + name + "_layout")
    return shape, dtype, str(layout)


def native_op(*, query, key_cache, weights, block_table, query_lengths,
              context_lengths, topk, compression=_DEFAULT_COMPRESSION, page_size=None,
              declaration=None, cache_owner=0, l2_capacity_bytes=None,
              exchange_grant=None,
              op_id="lightning_indexer", deps=(), stage=0,
              layer=0, requests=(), phase="prefill", stream="",
              binding_revision="unknown", runtime_version="unknown"):
    """Build the BF16 LI operator from its real native tensor inputs.

    ``query`` is TND ``[tokens, heads, dim]`` BF16, ``key_cache`` is paged
    ``[pages, page, kv_heads, dim]`` BF16, ``weights`` is
    ``[tokens, heads]`` FP32 (BF16/FP16 storage is accepted and upcast), and
    ``block_table`` is an INT32 ``[requests, columns]`` page table.  Dynamic
    query/key lengths are explicit request state, never inferred from padding.
    """
    q_shape, q_dtype, q_layout = _descriptor(
        query, "query", _QUERY_RANK, {"bf16"}, {"TND"})
    k_shape, k_dtype, k_layout = _descriptor(
        key_cache, "key_cache", _PAGED_KEY_CACHE_RANK, {"bf16"}, {"ND", "PA_BBND"})
    w_shape, w_dtype, w_layout = _descriptor(
        weights, "weights", 2, WEIGHT_DTYPES, {"ND"})
    b_shape, b_dtype, b_layout = _descriptor(
        block_table, "block_table", 2, {"int32"}, {"ND"})
    if q_shape[1] != w_shape[1] or w_shape[0] != q_shape[0]:
        raise ModelingError("bf16_lightning_indexer_native_weight_geometry")
    if k_shape[3] != q_shape[2]:
        raise ModelingError("bf16_lightning_indexer_native_head_dim")
    compression = _integer(compression, "compression", minimum=1)
    topk = _integer(topk, "topk", minimum=1)
    key_rows_per_page = k_shape[1]
    if page_size is not None and _integer(
            page_size, "page_size", minimum=1) != key_rows_per_page:
        raise ModelingError("bf16_lightning_indexer_native_page_size")
    page_size = key_rows_per_page
    spans = _gather_spans(None, None, query_lengths, context_lengths)
    if len(spans) != b_shape[0]:
        raise ModelingError("bf16_lightning_indexer_native_block_table_requests")
    if q_shape[0] != sum(q for q, _ in spans):
        raise ModelingError("bf16_lightning_indexer_native_query_lengths")
    columns = b_shape[1]
    compressed_rows = tuple(_compressed_rows(q, c, compression) for q, c in spans)
    key_pages = tuple(_physical_key_pages(rows, key_rows_per_page)
                      for rows in compressed_rows)
    logical_pages = tuple(_ceil_div(c + q, key_rows_per_page * compression)
                          for q, c in spans)
    if any(pages > columns or rows > k_shape[0] * key_rows_per_page
           for rows, pages in zip(compressed_rows, key_pages)):
        raise ModelingError("bf16_lightning_indexer_native_block_table_capacity")
    if declaration is not None:
        selected = select_indexer_source(dict(declaration))
        if selected.get("status") != "bf16_li_v2":
            raise ModelingError(
                "bf16_lightning_indexer_native_source_not_bf16:"
                + str(selected.get("status") or "unknown"))
    op = make_op(
        query_rows=q_shape[0], op_id=op_id, deps=deps, stage=stage, layer=layer,
        requests=requests, phase=phase, stream=stream,
        heads=q_shape[1], head_dim=q_shape[2],
        kv_heads=k_shape[2], topk=topk, compression=compression,
        page_size=page_size, query_dtype=q_dtype, cache_dtype=k_dtype,
        weight_dtype=w_dtype, query_lengths=tuple(q for q, _ in spans),
        context_lengths=tuple(c for _, c in spans),
        block_table_columns=columns, cache_owner=cache_owner,
        l2_capacity_bytes=l2_capacity_bytes,
        exchange_grant=exchange_grant,
        binding_revision=binding_revision, runtime_version=runtime_version)
    binding = {
        "kind": "independent_native_tensor_abi",
        "native_kernel": "LightningIndexer",
        "implementation": IMPLEMENTATION,
        "source_boundary": "kernel",
        "inputs": {
            "query": {"shape": q_shape, "dtype": q_dtype, "layout": q_layout},
            "key_cache": {"shape": k_shape, "dtype": k_dtype, "layout": k_layout},
            "weights": {"shape": w_shape, "dtype": w_dtype, "layout": w_layout},
            "block_table": {"shape": b_shape, "dtype": b_dtype, "layout": b_layout},
        },
        "dynamic_state": {
            "query_lengths": [q for q, _ in spans],
            "key_lengths": [c + q for q, c in spans],
        },
        "cache_policy": {
            "owner": cache_owner,
            "l2_capacity_bytes": l2_capacity_bytes,
            "line_bytes": CACHE_LINE_BYTES,
            "transfer_bytes": CACHE_TRANSFER_BYTES,
        },
        "cache_capacity": {
            "key_rows_per_page": key_rows_per_page,
            "compressed_key_rows": list(compressed_rows),
            "physical_key_pages": list(key_pages),
            "logical_token_pages": list(logical_pages),
            "logical_tokens_per_page": key_rows_per_page * compression,
            "block_table_columns": columns,
        },
        "outputs": {"indices": {"shape": (q_shape[0], k_shape[2], int(topk)),
                                "dtype": "int32", "layout": "ND"}},
        "source_selection": ("declared_bf16_li" if declaration is not None
                             else "native_dtypes_bf16"),
        "unresolved_outputs": ("retained tensor contract leaves output[1] "
                               "identity/shape unresolved; index selection consumes output[0]"),
    }
    metadata = dict(op.metadata)
    metadata["native_input_binding"] = binding
    metadata["native_implementation"] = {
        "native_kind": "LightningIndexer", "selection": IMPLEMENTATION,
        "conditions": {"submitted_kernels": 1,
                       "source_boundary": "qk_weighted_head_reduce_topk",
                       "upstream_metadata_generation": "outside"},
    }
    return replace(op, metadata=tuple(sorted(metadata.items())))


def make_op(*, query_rows=None, context=None, op_id="lightning_indexer", deps=(),
            stage=0, layer=0, requests=(), phase="prefill", stream="",
            heads=_DEFAULT_QUERY_HEADS, head_dim=_DEFAULT_HEAD_DIM,
            topk=_DEFAULT_TOPK, compression=_DEFAULT_COMPRESSION,
            page_size=_DEFAULT_KEY_PAGE_ROWS, kv_heads=1,
            query_dtype="bf16", cache_dtype="bf16", weight_dtype="fp32",
            query_lengths=(), context_lengths=(), block_table_columns=None,
            cache_owner=0, l2_capacity_bytes=None, exchange_grant=None,
            binding_revision="unknown", runtime_version="unknown"):
    """One or more contiguous causal spans with an explicit native geometry.

    Disjoint PCP spans need separate calls because each call rereads its own
    metadata and key range.  The caller supplies an observed binding/runtime
    identity explicitly; the analytical operator also works with unknown or
    different binary identity.
    """
    spans = _gather_spans(query_rows, context, query_lengths, context_lengths)
    for name, value in (("heads", heads), ("head_dim", head_dim),
                        ("topk", topk), ("page_size", page_size),
                        ("kv_heads", kv_heads)):
        _integer(value, name, minimum=1)
    if heads % kv_heads:
        raise ModelingError("bf16_lightning_indexer_kv_heads_must_divide_heads")
    cache_owner = _integer(cache_owner, "cache_owner", minimum=0)
    if l2_capacity_bytes is not None:
        l2_capacity_bytes = _integer(
            l2_capacity_bytes, "l2_capacity_bytes", minimum=0)
    if exchange_grant is not None:
        if isinstance(exchange_grant, dict):
            grant_name = str(exchange_grant.get("name")
                             or exchange_grant.get("grant") or "")
            grant_bytes = exchange_grant.get("bytes")
        else:
            grant_name, grant_bytes = str(exchange_grant), None
        if not grant_name:
            raise ModelingError("bf16_lightning_indexer_exchange_grant_invalid")
        if grant_bytes is not None:
            grant_bytes = _integer(grant_bytes, "exchange_grant_bytes")
        exchange_grant = {"name": grant_name, "bytes": grant_bytes}
    if query_dtype != "bf16" or cache_dtype != "bf16":
        raise ModelingError("bf16_lightning_indexer_v2_domain_mismatch")
    weight_dtype = str(weight_dtype).strip().lower()
    if weight_dtype not in WEIGHT_DTYPES:
        raise ModelingError("bf16_lightning_indexer_v2_domain_mismatch")
    rows = sum(q for q, _ in spans)
    pairs = sum(causal_pairs(q, c, compression) for q, c in spans)
    compressed_rows = tuple(_compressed_rows(q, c, compression) for q, c in spans)
    key_rows = sum(compressed_rows)
    # ``page_size`` is key rows per physical compressed-K page, not logical
    # (uncompressed) tokens.  The caller's token page relation is separate.
    key_pages = tuple(_physical_key_pages(count, page_size)
                      for count in compressed_rows)
    logical_pages = tuple(_ceil_div(c + q, page_size * compression)
                          for q, c in spans)
    required_columns = max(key_pages)
    if block_table_columns is None:
        columns = required_columns
    else:
        columns = _integer(block_table_columns, "block_table_columns", minimum=1)
        if columns < required_columns:
            raise ModelingError("bf16_lightning_indexer_block_table_too_short")
    batch = len(spans)
    # Keep native payload and logical arithmetic distinct.  HBM is the unique
    # external tensor payload; native query-tile reloads and local-list
    # exchange are added by the executable plan from declared mechanisms.
    query_bytes = rows * heads * head_dim * dtype_bytes("bf16")
    key_bytes = key_rows * kv_heads * head_dim * dtype_bytes("bf16")
    weight_bytes = rows * heads * dtype_bytes(weight_dtype)
    # Each request supplies two int32 lengths and one block table row.
    metadata_bytes = batch * (2 + columns) * _INDEX_ELEMENT_BYTES
    output_bytes = rows * kv_heads * topk * _INDEX_ELEMENT_BYTES
    weighted_ops = _weighted_vector_ops(pairs, heads, kv_heads)
    comparison_ops = _topk_comparisons(pairs, topk, kv_heads, spans, compression)
    sentinel = _sentinel_elements(spans, compression, topk, kv_heads)
    vector_ops = weighted_ops + comparison_ops + sentinel
    read_bytes = query_bytes + key_bytes + weight_bytes + metadata_bytes
    qk_flops = 2 * pairs * heads * head_dim
    shape = (rows, heads, head_dim)
    ledger = {
        "query_rows": rows, "context": max(c for _, c in spans),
        "query_key_pairs": pairs,
        "visible_compressed_keys": key_rows, "heads": heads,
        "kv_heads": kv_heads, "head_dim": head_dim, "topk": topk,
        "compression": compression, "page_size": page_size,
        "sequence_partition": tuple((q, c) for q, c in spans),
        "block_table_columns": columns,
        "key_rows_per_page": page_size,
        "compressed_key_rows": key_rows,
        "compressed_key_rows_by_request": compressed_rows,
        "physical_key_pages": key_pages,
        "logical_token_pages": logical_pages,
        "logical_tokens_per_page": page_size * compression,
        "query_tile_rows": QUERY_TILE_ROWS,
        "cache_owner": cache_owner,
        "l2_capacity_bytes": l2_capacity_bytes,
        "exchange_grant": exchange_grant,
        "mechanism_parameters": {
            "cache_reload": "one visible-key stream per query tile",
            "local_topk_exchange": "one local list per vector owner, capacity topk",
            "local_list_element_bytes": LOCAL_LIST_ELEMENT_BYTES,
            "exchange_storage": ("one per-tile GM slot reused across tiles; "
                                 "caller grant bound once when supplied"),
            "exchange_receive": ("topk entries per row/KV head; aliases the "
                                 "score slot when it fits"),
        },
        "qk_bf16_flops": qk_flops,
        "weighted_head_reduce_ops": weighted_ops,
        "topk_comparison_ops": comparison_ops,
        "sentinel_fill_elements": sentinel,
        "query_bytes": query_bytes, "unique_key_bytes": key_bytes,
        "head_weight_bytes": weight_bytes, "metadata_read_bytes": metadata_bytes,
        "selection_output_bytes": output_bytes,
        "read_bytes_lower_bound": read_bytes,
        "weight_dtype": weight_dtype,
        "traffic_scope": ("unique HBM payload; query-tile cache reload and "
                          "local TopK exchange are plan mechanisms"),
        "selection_scope": "logical weighted-ReLU/head-sum and TopK comparisons",
    }
    implementation = {
        "name": IMPLEMENTATION, "binding_revision": binding_revision,
        "runtime_version": runtime_version,
    }
    view = {
        "query_shape": shape, "query_dtype": query_dtype,
        "cache_dtype": cache_dtype, "cache_head_shape": (kv_heads, head_dim),
        "weight_dtype": weight_dtype, "layout_q": "TND",
        "layout_k": "PA_BBND", "mask_mode": _CAUSAL_MASK_MODE, "return_value": 0,
        "compression_ratio": compression, "page_size": page_size,
        "key_page_rows": page_size, "block_table_columns": columns,
        "logical_tokens_per_page": page_size * compression,
        "output_shape": (rows, kv_heads, topk), "output_dtype": "int32",
    }
    return OpSpec(
        op_id=op_id, family="lightning_indexer", phase=phase, stage=stage,
        layer=layer, requests=tuple(requests), deps=tuple(deps), stream=stream,
        semantic_shape=shape, local_shape=shape, padded_shape=shape,
        launch_shape=shape, flops=qk_flops, matrix_flops=(("bf16", qk_flops),),
        hbm_bytes=read_bytes + output_bytes, mte2_bytes=read_bytes,
        mte3_bytes=output_bytes, vector_ops=vector_ops,
        recipe=RECIPE, dtype="bf16", layout="TND/PA_BBND",
        source_boundary="kernel", coverage="bf16_li_mechanism_program",
        metadata=(
            ("native_source", SOURCE), ("implementation", implementation),
            ("lightning_indexer_work", ledger),
            ("native_tensor_view", view),
        ),
        notes=("Native scan and TopK only; metadata generation/fills/packing outside",
               "Query-tile key reload and local TopK exchange are modeled plan phases",
               "HBM keeps the unique payload; reload/exchange are on-chip or workspace traffic"),
    )
