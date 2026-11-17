"""Analytical phases for pooled index keys and causal token-index expansion.

[Role] Compose projections, FP32 pool compression, cache updates and selection.
[Invariants] Only the index plane is compressed. Each query scans its own
  complete pools; selected pool IDs expand to token IDs plus the causal tail.
  Score algebra is declared separately from unknown native owner scheduling.
[Timeouts & Locks] O(requests) construction and bounded scalar counts; no I/O.
[Side-Effect Policy] Appends operators; never reads captures or fits timings.
"""
import math
from dataclasses import replace

from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.catalog.index_pool import PoolSelection, pool_selection
from mindie_sim.schema.common import ModelingError, dtype_bytes

from .indexer_representation import index_representation
from .mla_native import index_layer_norm
from .structure_groups import declare_group

FLOPS_PER_FMA = 2
FP32_BYTES = int(dtype_bytes("fp32"))
INT32_BYTES = int(dtype_bytes("int32"))


def _op_dtype(b, op_id: str, fallback: str) -> str:
    op = next((item for item in reversed(b.ops) if item.op_id == op_id), None)
    return str(op.dtype) if (op is not None and op.dtype) else fallback


SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "8c30e4f491f73cba737a1dacc522894e437cdf3d",
    "paths": (
        "vllm_ascend/attention/indexer_kpool.py",
        "vllm_ascend/models/glm5next/attention.py",
        "vllm_ascend/models/glm5next/sparse_attn_indexer_kpool.py",
        "vllm_ascend/ops/triton/glm5_next_kpool_tail_compress.py",
        "vllm_ascend/ops/triton/glm5_next_lightning_indexer.py",
    ),
    "scope": "Triton_backend_functional_semantics; not_captured_native_KeyPool_dispatch_or_binary",
}


def recipes():
    return {}


def _selection_for(view, plan):
    """One request ledger. PCP DualChunkSwap sums each owned query span."""
    tail = plan.index_kpool_always_select_tail
    noncausal = bool(plan.noncausal_attention)
    ranges = getattr(view, "query_ranges", ()) or ()
    if not ranges:
        return pool_selection(view.q_len, view.context, plan.index_topk, plan.index_kpool,
                              always_select_tail=tail, noncausal=noncausal, kv_len=view.kv_len)
    parts = [pool_selection(end - start, view.context + start, plan.index_topk, plan.index_kpool,
                            always_select_tail=tail, noncausal=noncausal, kv_len=view.kv_len)
             for start, end in ranges]
    # The request-owned tail ring stays one pool, not one tail per span.
    retained = min(sum(end - start for start, end in ranges), plan.index_kpool)
    return PoolSelection(
        sum(part.scanned_pools for part in parts),
        sum(part.selected_pools for part in parts),
        sum(part.selected_tokens for part in parts),
        sum(part.causal_tail_tokens for part in parts),
        sum(part.completed_pools for part in parts),
        sum(part.historical_pool_rows for part in parts),
        retained)


def _geometry(views, plan):
    return tuple(_selection_for(view, plan) for view in views)


def _visited_pool_rows(views, plan):
    """Complete pool IDs actually visited, deduplicating shared prefixes only."""
    pool = max(1, int(plan.index_kpool))
    groups = {}
    for position, view in enumerate(views):
        if int(view.q_len) <= 0:
            continue
        rows = (int(view.context) + int(view.q_len)) // pool
        prefix = int(getattr(view, "prefix_tokens", 0) or 0)
        key = (str(getattr(view, "prefix_key", "") or ""),) if prefix > 0 \
            else (position, str(view.request_id))
        groups[key] = max(groups.get(key, 0), rows)
    return sum(groups.values())


def _causal_token_pairs(views, *, noncausal=False):
    """Visible token pairs. Owned spans use their global start, not q_len at context."""
    total = 0
    for view in views:
        ranges = getattr(view, "query_ranges", ()) or ()
        if noncausal:
            q_cnt = sum(end - start for start, end in ranges if end > start) if ranges else view.q_len
            total += q_cnt * view.kv_len
        elif ranges:
            total += sum((end - start) * (view.context + start)
                         + (end - start) * (end - start + 1) // 2
                         for start, end in ranges)
        else:
            query = view.q_len
            total += query * view.context + query * (query + 1) // 2
    return total


def _ring_span(start, rows, pool):
    """Contiguous/wrapped slot intervals actually touched by a tail write."""
    if rows <= 0:
        return ()
    if rows >= pool:
        return ((0, int(pool)),)
    start = int(start) % int(pool)
    stop = start + int(rows)
    if stop <= pool:
        return ((start, stop),)
    return ((start, int(pool)), (0, stop - int(pool)))


def _union_ring_spans(pieces, pool):
    """Union of physical ring intervals from every query range of one request."""
    pool = int(pool)
    intervals = []
    for start, end in pieces:
        start, end = max(0, int(start)), min(pool, int(end))
        if end > start:
            intervals.append((start, end))
    if not intervals:
        return ()
    intervals.sort()
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((start, end) for start, end in merged)


def _lookup(mapping, request, index):
    if mapping is None:
        return None
    if isinstance(mapping, dict):
        value = mapping.get(str(request))
        return mapping.get(request) if value is None else value
    values = tuple(mapping)
    return values[index] if index < len(values) else None


def _tail_state_accesses(views, geometry, plan, width, tail, versions=None,
                         *, tail_op=None, owners=None, occurrences=None,
                         grants=None, readers=None, last_uses=None,
                         storage_prefix=None):
    """Per-request state access from the actual typed K/gate tail output.

    ``tail_op`` is the real ``index_pool_tail_store`` occurrence.  Its typed
    ``(retained, 2, width)`` output fixes the physical row count and width; the
    per-request ring interval is the union over *all* ``query_ranges`` physical
    write slots, including wrap and disjoint ranges.  Persistent storage is
    derived from the stable builder prefix, not from this call's occurrence;
    the occurrence/version/grant identifies the materialization.
    """
    pool = int(plan.index_kpool)
    tail_id = str(getattr(tail_op, "op_id", tail))
    typed_rows = typed_width = None
    state_owner = storage_prefix
    if tail_op is not None:
        shape = tuple(getattr(tail_op, "local_shape", ()) or ())
        if len(shape) == 3:
            typed_rows, planes, typed_width = int(shape[0]), int(shape[1]), int(shape[2])
            if planes != 2:
                raise ModelingError("index_pool_tail_typed_planes_invalid")
        meta = dict(getattr(tail_op, "metadata", ()) or ())
        state_owner = (str(meta.get("state_owner") or "")
                       or storage_prefix or "index_pool_tail")
    if typed_width is not None:
        width = int(typed_width)
    accesses = []
    for index, (view, g) in enumerate(zip(views, geometry)):
        rows = min(int(g.retained_tail_rows), pool)
        if rows <= 0:
            continue
        if typed_rows is not None and rows > typed_rows:
            raise ModelingError("index_pool_tail_typed_rows_smaller_than_retained")
        ranges = tuple(getattr(view, "query_ranges", ()) or ())
        if not ranges:
            ranges = ((0, int(view.q_len)),)
        if rows >= pool:
            # A full-ring overwrite touches every physical slot regardless of
            # how the logical query ranges are partitioned.
            span = ((0, int(pool)),)
        else:
            pieces = []
            remaining = rows
            for start, end in ranges:
                if remaining <= 0:
                    break
                take = min(max(0, int(end) - int(start)), remaining)
                if not take:
                    continue
                ring_start = (int(view.context) + int(start)) % pool
                pieces.extend(_ring_span(ring_start, take, pool))
                remaining -= take
            if remaining:
                # The retained ring can be longer than the declared query
                # ranges only when it already wraps the whole pool.
                last = int(ranges[-1][1])
                pieces.extend(_ring_span((int(view.context) + last) % pool,
                                         remaining, pool))
            span = _union_ring_spans(pieces, pool)
        version = _lookup(versions, view.request_id, index)
        if version is None:
            version = getattr(view, "state_version", None)
        if version is None:
            version = getattr(view, "generation", None)
        version = None if version is None else int(version)
        owner = _lookup(owners, view.request_id, index)
        if owner is None:
            owner = getattr(view, "rank", None)
        if owner is None:
            owner = getattr(view, "owner", None)
        if owner is None:
            owner = getattr(view, "physical_rank", None)
        owner = None if owner is None else int(owner)
        storage = f"index_pool_tail:{state_owner}:{view.request_id}"
        grant = _lookup(grants, view.request_id, index)
        if grant is None:
            grant = storage
        occurrence_token = _lookup(occurrences, view.request_id, index)
        if occurrence_token is None:
            if version is not None:
                token = f"v{version}"
            else:
                token = "s" + "_".join(f"{start}-{end}" for start, end in span)
            occurrence = f"{tail_id}:{view.request_id}:{token}"
        else:
            occurrence = f"{tail_id}:{occurrence_token}"
        reader = _lookup(readers, view.request_id, index) or ""
        last_use = _lookup(last_uses, view.request_id, index) or ""
        accesses.append({
            "occurrence": occurrence, "producer": tail_id,
            "request": str(view.request_id),
            "stage": 0, "layer": 0,
            "direction": "write", "position": "tail",
            "output": "index_pool_tail",
            "storage": storage,
            "state_version": version,
            "span": span,
            "rank": owner,
            "scope": "target",
            "bytes": float(rows * width * 8),
            "source": tail_id,
            "reader": reader,
            "grant": str(grant),
            "last_use": last_use,
        })
    return accesses


def state_accesses(op):
    """The actual per-request tail writes bound by one producer occurrence.

    This is the same tuple already attached as ``metadata['state_accesses']``;
    T04/T08 and SERVICE-CORE can consume it without rebuilding request rows
    from a total retained count.
    """
    return tuple(dict(getattr(op, "metadata", ()) or ()).get("state_accesses", ()))


def packed_parent(op):
    """The real packed index-store K/scale/tail outputs bound by one producer.

    T12/native ScatterPaCache consumes this contract instead of assuming that a
    scale exists, that it lives inside the cache row, or that an FP32 tail is
    part of the packed parent allocation.  A missing scale is explicit.
    """
    meta = dict(getattr(op, "metadata", ()) or ())
    return meta.get("packed_parent")


def producer(b, *, source, views, plan, model, quant, padded_tokens,
             state_versions=None, state_owners=None, state_occurrences=None,
             state_grants=None, state_readers=None, state_last_uses=None):
    """Complete pools before overwriting the bounded request-owned tail ring."""
    if (not plan.index_kpool_compress or plan.index_kpool <= 1
            or plan.index_input != "hidden" or plan.compress_ratio or plan.qk_rope_head_dim):
        raise ModelingError("index_pool_requires_hidden_nope_and_independent_token_kv")
    valid = sum(v.q_len for v in views)
    count, width, pool = padded_tokens, plan.index_dim, plan.index_kpool
    geometry = _geometry(views, plan)
    completed = sum(g.completed_pools for g in geometry)
    retained = sum(g.retained_tail_rows for g in geometry)
    start = len(b.ops)
    hidden = source
    if model.activation_dtype != "fp32":
        hidden = b.vector(
            "index_pool_hidden_fp32", (source,), count * model.hidden_size, 1,
            family="cast", dtype="fp32", shape=(count, model.hidden_size),
            read_bytes_override=count * model.hidden_size * dtype_bytes(model.activation_dtype),
            write_bytes_override=count * model.hidden_size * 4,
            metadata=(("input_format", model.activation_dtype), ("output_format", "fp32")))
    fused_weights = (b.quant.for_category("indexer.pool_projection")
                     == b.quant.for_category("indexer.weights"))
    key = b.matrix("index_k_weights" if fused_weights else "index_k", (hidden,),
                   count, width + (plan.index_heads if fused_weights else 0), model.hidden_size,
                   category="indexer.pool_projection", metadata=(
                       ("key_slice", (0, width)),
                       ("weights_slice", (width, width + plan.index_heads) if fused_weights else None),
                       ("projection_packing", "source_joint_key_weights" if fused_weights
                        else "separate_explicit_operand_formats"),
                       ("slice_layout", "views_of_projection; no_copy"), ("native_source", SOURCE)))
    normalized = index_layer_norm(b, source=key, rows=count, width=width,
                                  input_dtype="fp32", output_dtype="fp32")
    gates = b.matrix("index_pool_gate", (hidden,), count, width, model.hidden_size,
                     category="indexer.pool_gate", metadata=(("native_source", SOURCE),))
    common = (("pool_size", pool), ("native_source", SOURCE), ("valid_query_rows", valid),
              ("padded_query_rows", count),
              ("implementation_basis", "analytic_materialized_phases"),
              ("native_owner_schedule", "not_selected"))
    pooled = b.work(
        "index_pool_compress", (normalized, gates), (completed, pool, width),
        vector=completed * width * (8 * pool - 2),
        read=completed * pool * width * 8 + (pool * width * 4 if completed else 0),
        write=completed * width * dtype_bytes(model.activation_dtype),
        family="index_pool", dtype="fp32", metadata=common + (
            ("operation", "softmax_gate_plus_ape_over_pool_then_weighted_key_sum"),
            ("completed_pools", completed),
            ("historical_input_rows", sum(g.historical_pool_rows for g in geometry)),
            ("current_input_rows", completed * pool - sum(g.historical_pool_rows for g in geometry)),
            ("output_format", model.activation_dtype),
            ("element_service_count", "arithmetic_reductions_exp_divide_and_output_cast")))
    tail = b.work(
        "index_pool_tail_store", (pooled, normalized, gates), (retained, 2, width),
        read=retained * width * 8, write=retained * width * 8,
        family="cache_store", dtype="fp32", metadata=common + (
            ("retained_rows", retained), ("cache_kind", "index_pool_tail"),
            ("index_width", width), ("write_bytes", retained * width * 8),
            ("state_owner", b.prefix),
            ("producer_output", "index_pool_tail"),
            ("state_storage", "index_pool_tail:" + b.prefix),
            ("state_span", ((0, retained),)),
            ("cache_visible", True), ("scope", "contiguous_forward; speculative_rollback_not_reconstructed")))
    # Per-request tail ring facts from the actual views/pool geometry.  The
    # aggregate retained count alone loses multiplicity; SERVICE-CORE can
    # consume this ordered set directly without rebuilding a token total.
    tail_accesses = _tail_state_accesses(
        views, geometry, plan, width, tail, versions=state_versions,
        tail_op=b.ops[-1], owners=state_owners,
        occurrences=state_occurrences, grants=state_grants,
        readers=state_readers, last_uses=state_last_uses,
        storage_prefix="index_pool_tail:" + b.prefix)
    for item in tail_accesses:
        item["stage"] = int(b.common["stage"])
        item["layer"] = int(b.common["layer"])
    b.ops[-1] = replace(
        b.ops[-1],
        metadata=tuple(sorted({
            **dict(b.ops[-1].metadata),
            "state_accesses": tuple(tail_accesses),
            "state_access_total_bytes": float(sum(
                item["bytes"] for item in tail_accesses)),
            "state_access_count": len(tail_accesses),
        }.items())))
    values = index_representation(b, "index_key", pooled, completed, width,
                                  quant=quant, model=model, rotate_basis=False)
    stored = b.store("index_store", (values, tail), completed,
                     index_row_bytes(model, width), quant.index_dtype)
    # Real packed-parent K/scale truth for T12/native ScatterPaCache.  The
    # scale is a side output of the actual quantize occurrence, not a copy
    # inferred from the cache row; the FP32 tail remains a separate output.
    value_op = next((op for op in b.ops if op.op_id == values), None)
    value_meta = dict(value_op.metadata) if value_op is not None else {}
    scale_count = int(value_meta.get("scale_count", 0) or 0)
    packed_elements = int(value_meta.get("packed_elements", 1) or 1)
    scale_present = scale_count > 0
    value_dtype = str(value_meta.get("output_format") or quant.index_dtype)
    scale_dtype = str(value_meta.get("scale_format") or "none")
    physical_row_bytes = int(value_meta.get(
        "physical_row_bytes", index_row_bytes(model, width)))
    scale_row_bytes = (int(scale_count * dtype_bytes(scale_dtype))
                       if scale_present and scale_dtype not in {"", "none"}
                       else 0)
    data_row_bytes = int(max(0, physical_row_bytes - scale_row_bytes))
    materialized = (packed_elements != 1 or scale_present
                    or value_dtype != str(model.activation_dtype))
    packed_parent = {
        "parent_occurrence": stored,
        "value": {
            "producer": values, "output": "index_k",
            "dtype": value_dtype, "packed_elements": packed_elements,
            "physical_row_bytes": physical_row_bytes,
            "data_row_bytes": data_row_bytes,
            "presence": True,
            "materialization": ("packed_cache_row" if materialized
                                else "logical_value"),
        },
        "scale": {
            "producer": values if scale_present else None,
            "output": "index_scales" if scale_present else None,
            "dtype": scale_dtype,
            "shape": tuple(value_meta.get("native_scale_shape")
                           or value_meta.get("scale_shape") or ()),
            "count": scale_count,
            "row_bytes": scale_row_bytes,
            "row_offset": data_row_bytes,
            "presence": scale_present,
            "materialization": ("native_scatter_scale_slot" if scale_present
                                else "absent"),
        },
        "row_materialization": ("packed_k_plus_scale_suffix"
                                if scale_present and scale_row_bytes
                                else "k_only"),
        "tail": {
            "producer": tail, "dtype": "fp32",
            "shape": (int(retained), 2, int(width)),
            "presence": bool(retained),
            "materialization": "separate_fp32_tail",
        },
        "presence": {"value": True, "scale": scale_present,
                     "tail": bool(retained)},
        "alias": {"value": values, "tail": tail,
                  "scale": values if scale_present else None},
        "native_scatter": {
            "parent": stored,
            "value": values,
            "scale": values if scale_present else None,
            "tail": tail,
            "child_contract": (
                "two ScatterPaCache children (packed K + scale) when the scale "
                "is present; one packed K write otherwise; FP32 tail is a "
                "separate state producer"),
        },
    }
    b.ops[-1] = replace(
        b.ops[-1],
        metadata=tuple(sorted({
            **dict(b.ops[-1].metadata),
            "packed_parent": packed_parent,
        }.items())))
    members = tuple(op.op_id for op in b.ops[start:])
    declare_group(
        b.ops, key=b.prefix + ".key_pool", role="key_pool", family="pooled_indexer",
        members=members, entries=(hidden,) if hidden != source else (key, gates), anchor=stored,
        view={"tokens": count, "valid_tokens": valid,
              "hidden": model.hidden_size, "width": width, "pool_size": pool,
              "completed_pools": completed, "tail_rows": retained,
              "cache_dtype": quant.index_dtype, "state_dtype": "fp32"},
        implementation={"native_kind": "KeyPool", "conditions": {
            "scope": "analytical_projection_norm_gate_pool_cache_phases",
            "native_dispatch_and_fusion_not_selected": True}},
        source=SOURCE, coverage={"kind": "analytic_phase_decomposition"})
    return stored, key if fused_weights else None


def selection(b, *, query, weights, cache, views, plan, quant, padded_tokens,
              score_mode="linear_weighted_query",
              score_implementation="declared", l2_bytes=None):
    """Use source linear scoring; per-head MMA is an explicit alternative."""
    if score_mode not in {"per_head_relu", "linear_weighted_query"}:
        raise ModelingError("index_pool_unknown_score_mode")
    if score_implementation not in {"declared", "per_head_mma", "weighted_query"}:
        raise ModelingError("index_pool_unknown_score_implementation")
    geometry = _geometry(views, plan)
    valid = sum(v.q_len for v in views)
    tokens, heads, width = padded_tokens, plan.index_heads, plan.index_dim
    scanned = sum(g.scanned_pools for g in geometry)
    selected = sum(g.selected_pools for g in geometry)
    valid_indices = sum(g.selected_tokens for g in geometry)
    pool_topk = plan.index_topk // plan.index_kpool
    output_width = plan.index_topk + (plan.index_kpool - 1 if plan.index_kpool_always_select_tail else 0)
    row_bytes = float(index_row_bytes(b.model, width))
    unique_pool_rows = _visited_pool_rows(views, plan)
    l2_bounded_pool_read = float(unique_pool_rows) * row_bytes
    if l2_bytes is not None and float(l2_bytes) > 0.0:
        if l2_bounded_pool_read > float(l2_bytes):
            spill = max(0.0, 1 - float(l2_bytes) / l2_bounded_pool_read)
            l2_bounded_pool_read += (
                max(0.0, float(scanned) * row_bytes - l2_bounded_pool_read)
                * spill)
    common = (("pool_size", plan.index_kpool), ("score_mode", score_mode),
              ("score_implementation", score_implementation),
              ("score_pool_row_bytes", row_bytes),
              ("score_unique_pool_rows", unique_pool_rows),
              ("score_logical_pool_read_bytes", float(scanned) * row_bytes),
              ("score_l2_hit_bytes", max(
                  0.0, float(scanned) * row_bytes - l2_bounded_pool_read)),
              ("valid_query_rows", valid), ("padded_query_rows", tokens),
              ("causal_token_pairs", _causal_token_pairs(views, noncausal=bool(plan.noncausal_attention))),
              ("scanned_pools", scanned), ("selected_pools", selected),
              ("selected_token_rows", valid_indices),
              ("native_owner_schedule", "not_selected"),
              ("implementation_basis", "analytic_resource_phases"), ("native_source", SOURCE))
    query_dtype = _op_dtype(b, query, quant.index_dtype)
    weights_dtype = _op_dtype(b, weights, quant.index_dtype)
    query_elem_b = dtype_bytes(query_dtype)
    weights_elem_b = dtype_bytes(weights_dtype)
    score_dtype = "fp32"
    score_elem_b = dtype_bytes(score_dtype)
    members, entries = [], []
    materialized_scores = False
    if score_mode == "linear_weighted_query":
        query_f32 = b.vector(
            "index_pool_query_fp32", (query,), tokens * heads * width, 1,
            family="cast", dtype="fp32", shape=(tokens, heads, width),
            read_bytes_override=tokens * heads * width * query_elem_b,
            write_bytes_override=tokens * heads * width * FP32_BYTES, metadata=common)
        weights_f32 = b.vector(
            "index_pool_weights_fp32", (weights,), tokens * heads, 1,
            family="cast", dtype="fp32", shape=(tokens, heads),
            read_bytes_override=tokens * heads * weights_elem_b,
            write_bytes_override=tokens * heads * FP32_BYTES, metadata=common)
        weighted = b.work(
            "index_pool_weighted_query", (query_f32, weights_f32), (tokens, width),
            vector=tokens * heads * width * FLOPS_PER_FMA,
            read=tokens * heads * (width + 1) * FP32_BYTES,
            write=tokens * width * FP32_BYTES, family="indexer", dtype="fp32",
            metadata=common)
        members.extend((query_f32, weights_f32, weighted)); entries.extend((query_f32, weights_f32))
        # The declared weighted-query implementation is FP32 vector arithmetic;
        # it is not a BF16 Cube matmul and the score tile is materialized unless
        # one executable fused plan owns the tile and the TopK state in UB.
        materialized_scores = True
        scores = b.work(
            "index_pool_scores", (weighted, cache), (tokens, width),
            flops=0.0,
            matrix_flops=(),
            vector=FLOPS_PER_FMA * scanned * width,
            read=float(scanned) * row_bytes + tokens * width * FP32_BYTES,
            hbm_read=l2_bounded_pool_read + tokens * width * FP32_BYTES,
            write=scanned * score_elem_b, family="indexer", dtype=score_dtype,
            metadata=common + (("score_materialization",
                                "separate_fp32_score_then_topk"),))
        entries.append(scores)
    else:
        matrix = ()
        score_flops = 0.0
        if score_implementation == "per_head_mma":
            score_flops = float(FLOPS_PER_FMA * scanned * heads * width)
            matrix = ((query_dtype, score_flops),)
        materialized_scores = not matrix
        qw_bytes = tokens * heads * (width * query_elem_b + weights_elem_b)
        scores = b.work(
            "index_pool_scores", (query, weights, cache), (tokens, heads, width),
            flops=score_flops, matrix_flops=matrix,
            vector=scanned * heads * FLOPS_PER_FMA,
            read=float(scanned) * row_bytes + qw_bytes,
            hbm_read=l2_bounded_pool_read + qw_bytes,
            write=scanned * score_elem_b if materialized_scores else 0.0,
            family="indexer", dtype=query_dtype,
            metadata=common + (("score_materialization",
                                "per_head_mma" if matrix else
                                "separate_fp32_score_then_topk"),))
        entries.append(scores)
    members.append(scores)
    topk_score_read = scanned * score_elem_b if materialized_scores else 0.0
    topk = b.work(
        "index_pool_topk", (scores,), (tokens, pool_topk),
        vector=scanned * math.log2(max(2, pool_topk)),
        read=topk_score_read,
        write=tokens * pool_topk * INT32_BYTES,
        family="topk", dtype=score_dtype, metadata=common + (
            ("selection_output_dtype", "int32"),
            ("score_materialization",
             "separate_fp32_score_then_topk" if materialized_scores
             else "per_head_mma")))
    tail_rows = sum(g.causal_tail_tokens for g in geometry)
    has_output = (selected > 0 or tail_rows > 0) and tokens > 0
    packed = b.work(
        "index_topk", (topk,), (tokens, output_width),
        vector=selected * plan.index_kpool * 2 + (tokens * output_width if has_output else 0),
        read=tokens * pool_topk * INT32_BYTES if (selected > 0 and tokens > 0) else 0,
        write=tokens * output_width * INT32_BYTES if has_output else 0,
        family="topk", dtype="int32", metadata=common + (
            ("selection_output_dtype", "int32"), ("selection_capacity", output_width),
            ("causal_tail_rows", tail_rows),
            ("selection_layout", "selected_pool_ids_expanded_then_contiguous_valid_causal_tail")))
    members.extend((topk, packed))
    declare_group(
        b.ops, key=b.prefix + ".pool_key_indexer", role="pool_key_indexer", family="pooled_indexer",
        members=tuple(members), entries=tuple(entries), anchor=packed,
        view={"tokens": tokens, "valid_tokens": valid,
              "heads": heads, "width": width, "pool_size": plan.index_kpool,
              "pool_topk": pool_topk, "output_width": output_width,
              "query_dtype": quant.index_dtype, "cache_dtype": quant.index_dtype,
              "score_mode": score_mode, "scanned_pools": scanned},
        implementation={"native_kind": "PoolKeyIndexer", "conditions": {
            "scope": "analytical_score_topk_expand_phases", "score_mode": score_mode,
            "captured_native_algebra_and_owner_schedule": "not_established"}},
        source=SOURCE, coverage={"kind": "analytic_phase_decomposition"})
    return packed
