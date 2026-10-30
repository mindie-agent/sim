"""Analytical phases for pooled index keys and causal token-index expansion.

[Role] Compose projections, FP32 pool compression, cache updates and selection.
[Invariants] Only the index plane is compressed. Each query scans its own
  complete pools; selected pool IDs expand to token IDs plus the causal tail.
  Score algebra is declared separately from unknown native owner scheduling.
[Timeouts & Locks] O(requests) construction and bounded scalar counts; no I/O.
[Side-Effect Policy] Appends operators; never reads captures or fits timings.
"""
import math

from mindie_sim.catalog.cache import index_row_bytes
from mindie_sim.catalog.index_pool import pool_selection
from mindie_sim.schema.common import ModelingError, dtype_bytes

from .indexer_representation import index_representation
from .mla_native import index_layer_norm
from .structure_groups import declare_group


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


def _geometry(views, plan):
    return tuple(pool_selection(v.q_len, v.context, plan.index_topk, plan.index_kpool,
                                always_select_tail=plan.index_kpool_always_select_tail)
                 for v in views)


def producer(b, *, source, views, plan, model, quant, padded_tokens):
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
            ("cache_visible", True), ("scope", "contiguous_forward; speculative_rollback_not_reconstructed")))
    values = index_representation(b, "index_key", pooled, completed, width,
                                  quant=quant, model=model, rotate_basis=False)
    stored = b.store("index_store", (values, tail), completed,
                     index_row_bytes(model, width), quant.index_dtype)
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
              score_mode="linear_weighted_query"):
    """Use source linear scoring; per-head ReLU is an explicit alternative."""
    if score_mode not in {"per_head_relu", "linear_weighted_query"}:
        raise ModelingError("index_pool_unknown_score_mode")
    geometry = _geometry(views, plan)
    valid = sum(v.q_len for v in views)
    tokens, heads, width = padded_tokens, plan.index_heads, plan.index_dim
    scanned = sum(g.scanned_pools for g in geometry)
    selected = sum(g.selected_pools for g in geometry)
    valid_indices = sum(g.selected_tokens for g in geometry)
    pool_topk = plan.index_topk // plan.index_kpool
    output_width = plan.index_topk + (plan.index_kpool - 1 if plan.index_kpool_always_select_tail else 0)
    common = (("pool_size", plan.index_kpool), ("score_mode", score_mode),
              ("valid_query_rows", valid), ("padded_query_rows", tokens),
              ("scanned_pools", scanned), ("selected_pools", selected),
              ("selected_token_rows", valid_indices),
              ("native_owner_schedule", "not_selected"),
              ("implementation_basis", "analytic_resource_phases"), ("native_source", SOURCE))
    members, entries = [], []
    if score_mode == "linear_weighted_query":
        query_f32 = b.vector(
            "index_pool_query_fp32", (query,), tokens * heads * width, 1,
            family="cast", dtype="fp32", shape=(tokens, heads, width),
            read_bytes_override=tokens * heads * width * dtype_bytes(quant.index_dtype),
            write_bytes_override=tokens * heads * width * 4, metadata=common)
        weights_f32 = b.vector(
            "index_pool_weights_fp32", (weights,), tokens * heads, 1,
            family="cast", dtype="fp32", shape=(tokens, heads),
            read_bytes_override=tokens * heads * dtype_bytes(quant.index_dtype),
            write_bytes_override=tokens * heads * 4, metadata=common)
        weighted = b.work(
            "index_pool_weighted_query", (query_f32, weights_f32), (tokens, width),
            vector=tokens * heads * width * 2,
            read=tokens * heads * (width + 1) * 4,
            write=tokens * width * 4, family="indexer", dtype="fp32", metadata=common)
        members.extend((query_f32, weights_f32, weighted)); entries.extend((query_f32, weights_f32))
        scores = b.work(
            "index_pool_scores", (weighted, cache), (tokens, width),
            vector=2 * scanned * width,
            read=scanned * index_row_bytes(b.model, width) + tokens * width * 4,
            write=scanned * 4, family="indexer", dtype="fp32", metadata=common)
        entries.append(scores)
    else:
        scores = b.work(
            "index_pool_scores", (query, weights, cache), (tokens, heads, width),
            flops=2 * scanned * heads * width, vector=scanned * heads * 2,
            read=scanned * index_row_bytes(b.model, width)
                 + tokens * heads * (width + 1) * dtype_bytes(quant.index_dtype),
            write=scanned * 4, family="indexer", dtype=quant.index_dtype, metadata=common)
        entries.append(scores)
    members.append(scores)
    topk = b.work(
        "index_pool_topk", (scores,), (tokens, pool_topk),
        vector=scanned * math.log2(max(2, pool_topk)),
        read=scanned * 4, write=tokens * pool_topk * 4,
        family="topk", dtype="fp32", metadata=common + (("selection_output_dtype", "int32"),))
    packed = b.work(
        "index_topk", (topk,), (tokens, output_width),
        vector=selected * plan.index_kpool * 2 + tokens * output_width,
        read=tokens * pool_topk * 4, write=tokens * output_width * 4,
        family="topk", dtype="int32", metadata=common + (
            ("selection_output_dtype", "int32"), ("selection_capacity", output_width),
            ("causal_tail_rows", sum(g.causal_tail_tokens for g in geometry)),
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
