"""Explicit mixed-quant sparse MLA source binding for ordinary decode graphs.

[Role] Replace one batched attention boundary, preserving cache producers.
[Invariants] Packed cache formats are declared by the model and shared with
  capacity accounting. Source selection does not prove a captured binary ABI.
  One predicate decides whether an automatic binding is admissible, so a
  default can never install a geometry the planner rejects.
[Timeouts & Locks] O(active query rows), no file or device access.
[Side-Effect Policy] Returns immutable operators and a shared Engine recipe.
"""
from dataclasses import replace

from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from .mqsmla import (
    COMPRESSED_CACHE_ROW_BYTES,
    LEGACY_NATIVE_KIND,
    LEGACY_RECIPE,
    LEGACY_SOURCE,
    MQSMLA_HEAD_WIDTH,
    MQSMLA_ROPE_WIDTH,
    NATIVE_KIND,
    ORIGINAL_CACHE_ROW_BYTES,
    RECIPE,
    SOURCE,
    SUPPORTED_PHASES,
    supported_heads,
    build,
    query_rows,
    selective_fd_excluded,
)
#: The CANN 9.2.1 source-symbol spelling retained as an explicit alias.
SOURCE_RECIPE = "mixed_quant_sparse_flash_mla"
FP8_WINDOW_GROUP_SIZE = 32
FP4_COMPRESSED_GROUP_SIZE = 16
SINGLE_KV_HEAD = 1
UNIT_CONTEXT_PARALLEL = 1
MIN_ACTIVE_ROWS = 1
QUERY_ORIGIN = 0
NO_COMPRESSED_ROWS = 0
VALID_ADDRESSING = frozenset({"scalar", "auto", "vector"})
FORMATS = {
    "window": {"dtype": "fp8_e4m3", "group_size": FP8_WINDOW_GROUP_SIZE,
               "scale_dtype": "bf16"},
    "compressed_kv": {"dtype": "fp4_e2m1", "group_size": FP4_COMPRESSED_GROUP_SIZE,
                      "scale_dtype": "bf16"},
}


def recipes():
    """The official CANN recipe, its source-name alias and the legacy prototype."""
    return {RECIPE: PlannedRecipe(build), SOURCE_RECIPE: PlannedRecipe(build),
            LEGACY_RECIPE: PlannedRecipe(build)}


def query_shape(views, plan, *, topk_present=False):
    """The exact per-row geometry the builder will consume."""
    effective_ratio = (plan.compress_ratio if topk_present or plan.owns_long_kv
                       else NO_COMPRESSED_ROWS)
    return {
        "sequences": tuple((int(v.q_len), int(v.context)) for v in views),
        "window": plan.window,
        "ratio": effective_ratio,
        "topk": plan.index_topk if topk_present else NO_COMPRESSED_ROWS,
    }


def unsupported_reason(*, views, plan, parallel, model, padded_tokens,
                       addressing="scalar", metadata_support_fd=False,
                       return_softmax_lse=False, cores=None, topk_present=False,
                       implementation=RECIPE):
    """Return the first failed source-domain predicate, or ``None``.

    ``cores`` is the declared AIC owner count; when supplied, the retained
    SELECTIVE_FD_72 wrapper geometry is rejected through the same predicate the
    builder uses.
    """
    if (metadata_support_fd or addressing not in VALID_ADDRESSING
            or return_softmax_lse):
        return "mqsmla_supported_branch_is_whole_row_address_no_lse"
    tp = max(int(getattr(parallel, "tp", SINGLE_KV_HEAD) or SINGLE_KV_HEAD), SINGLE_KV_HEAD)
    cp = int(getattr(parallel, "cp", UNIT_CONTEXT_PARALLEL) or UNIT_CONTEXT_PARALLEL)
    if cp != UNIT_CONTEXT_PARALLEL:
        return "mqsmla_requires_context_parallel_cp1"
    if not supported_heads(plan.num_heads // tp):
        return "mqsmla_local_heads_outside_source_tp_tiling"
    groups = int(getattr(plan, "o_groups", 0) or 0)
    if groups and groups % tp:
        return "mqsmla_output_groups_not_divisible_by_tp"
    if (plan.num_kv_heads != SINGLE_KV_HEAD or plan.head_dim != MQSMLA_HEAD_WIDTH
            or plan.qk_rope_head_dim != MQSMLA_ROPE_WIDTH):
        return "mqsmla_requires_d512_rope64_single_kv_head"
    if model.activation_dtype != "bf16":
        return "mqsmla_requires_bf16_activation"
    valid_rows = sum(int(v.q_len) for v in views)
    if valid_rows < MIN_ACTIVE_ROWS or padded_tokens < valid_rows:
        return "mqsmla_binding_requires_positive_padded_rows"
    if any(v.phase not in SUPPORTED_PHASES or v.padding
           or (v.query_ranges and v.query_ranges != ((QUERY_ORIGIN, v.q_len),))
           for v in views):
        return "mqsmla_binding_requires_contiguous_unpadded_decode_or_draft_rows"
    if cores is not None and implementation == LEGACY_RECIPE:
        # ``SELECTIVE_FD_72`` is a retired ``cannbot-dsl`` geometry. The
        # official CANN program accepts the same rows, so the refusal follows
        # the explicitly selected implementation, never the row count alone.
        rows = tuple(query_rows(query_shape(views, plan, topk_present=topk_present)))
        if selective_fd_excluded(cores, rows):
            return "mqsmla_selective_72_row_fd_branch_not_modeled"
    for region, declaration in FORMATS.items():
        if cache_format(model, region) != CacheFormat(**declaration):
            return "mqsmla_cache_representation_contract_required:" + region
    return None


def validate(*, views, plan, parallel, model, padded_tokens,
             metadata_support_fd=False, addressing="scalar", return_softmax_lse=False,
             cores=None, topk_present=False, implementation=RECIPE):
    reason = unsupported_reason(
        views=views, plan=plan, parallel=parallel, model=model,
        padded_tokens=padded_tokens, addressing=addressing,
        metadata_support_fd=metadata_support_fd,
        return_softmax_lse=return_softmax_lse, cores=cores,
        topk_present=topk_present, implementation=implementation,
    )
    if reason:
        raise ModelingError(reason)


def lower_attention(ops, outputs, *, selection, views, plan, topk_present, prefix,
                    parallel=None):
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    tp_degree = max(int(getattr(parallel, "tp", SINGLE_KV_HEAD) or SINGLE_KV_HEAD),
                    SINGLE_KV_HEAD)
    heads = plan.num_heads // tp_degree
    rows = tuple((v.q_len, v.context) for v in views)
    active_q = sum(q for q, _ in rows)
    effective_ratio = (plan.compress_ratio if topk_present or plan.owns_long_kv
                       else NO_COMPRESSED_ROWS)
    implementation = str(getattr(selection, "name", RECIPE))
    shape = {"sequences": rows, "window": plan.window, "ratio": effective_ratio,
             "topk": plan.index_topk if topk_present else 0, "heads": heads,
             "width": MQSMLA_HEAD_WIDTH, "metadata_support_fd": False,
             "implementation": implementation,
             "addressing": str(dict(getattr(selection, "options", {}) or {}).get(
                 "addressing", "scalar"))}
    query = (active_q, heads, MQSMLA_HEAD_WIDTH)
    legacy = implementation == LEGACY_RECIPE
    meta = {**dict(members[0].metadata), "mqsmla_geometry": shape,
            "logical_pairs": sum(dict(op.metadata).get("logical_pairs", 0) for op in members),
            "kv_sweeps": sum(dict(op.metadata).get("kv_sweeps", 0) for op in members),
            "implementation_binding": {
                "selector": "mqsmla_attention", "implementation": implementation,
                "capture_binary_identity": "not_established",
                "source": LEGACY_SOURCE if legacy else SOURCE,
                "coverage": "conditional_source_program",
                "caller_unmodeled": ["separate_aicpu_metadata_generation",
                                      "original_window_index_generation"]},
            "attention_tensor_view": {
                "kind": "mixed_quant_shared_kv", "query_shape": query,
                "output_shape": query,
                "query_dtype": "bf16", "output_dtype": "bf16",
                "original_cache_row_bytes": ORIGINAL_CACHE_ROW_BYTES,
                "compressed_cache_row_bytes": COMPRESSED_CACHE_ROW_BYTES,
                "compressed_cache": effective_ratio > 0},
            "native_kind": LEGACY_NATIVE_KIND if legacy else NATIVE_KIND}
    result = replace(members[0], op_id=prefix + ".mqsmla",
                     recipe=LEGACY_RECIPE if legacy else RECIPE,
                     deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
                     semantic_shape=query,
                     local_shape=query,
                     padded_shape=query,
                     launch_shape=query,
                     flops=sum(op.flops for op in members),
                     vector_ops=sum(op.vector_ops for op in members),
                     hbm_bytes=sum(op.hbm_bytes for op in members),
                     mte2_bytes=sum(op.mte2_bytes for op in members),
                     mte3_bytes=sum(op.mte3_bytes for op in members),
                     source_boundary="kernel", metadata=tuple(meta.items()))
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.append(result)
    return [result.op_id]
