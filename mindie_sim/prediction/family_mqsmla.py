"""Explicit mixed-quant sparse MLA source binding for ordinary decode graphs.

[Role] Replace one batched attention boundary, preserving cache producers.
[Invariants] Packed cache formats are declared by the model and shared with
  capacity accounting. Source selection does not prove a captured binary ABI.
[Timeouts & Locks] O(active query rows), no file or device access.
[Side-Effect Policy] Returns immutable operators and a shared Engine recipe.
"""
from dataclasses import replace

from mindie_sim.catalog.cache_format import CacheFormat, cache_format
from mindie_sim.schema.common import ModelingError
from .mechanism import PlannedRecipe
from .mqsmla import SOURCE, build

RECIPE = "cannbot_mixed_quant_sparse_flash_mla"
FORMATS = {
    "window": {"dtype": "fp8_e4m3", "group_size": 32, "scale_dtype": "bf16"},
    "compressed_kv": {"dtype": "fp4_e2m1", "group_size": 16, "scale_dtype": "bf16"},
}


def recipes():
    return {RECIPE: PlannedRecipe(build)}


def validate(*, views, plan, parallel, model, padded_tokens,
             metadata_support_fd=False, addressing="scalar", return_softmax_lse=False):
    if metadata_support_fd or addressing != "scalar" or return_softmax_lse:
        raise ModelingError("mqsmla_supported_branch_is_whole_row_scalar_address_no_lse")
    if (parallel.cp != 1 or parallel.pcp != 1 or plan.num_heads // parallel.tp != 64
            or plan.num_kv_heads != 1 or plan.head_dim != 512
            or plan.qk_rope_head_dim != 64 or model.activation_dtype != "bf16"):
        raise ModelingError("mqsmla_requires_bf16_64_heads_d512_rope64_unsharded_cache")
    if (any(v.phase != "decode" or v.draft_proposal_tokens or
            v.query_ranges and v.query_ranges != ((0, v.q_len),) for v in views)
            or padded_tokens != sum(v.q_len for v in views)):
        raise ModelingError("mqsmla_binding_requires_contiguous_unpadded_decode_rows")
    for region, declaration in FORMATS.items():
        if cache_format(model, region) != CacheFormat(**declaration):
            raise ModelingError("mqsmla_cache_representation_contract_required:" + region)


def lower_attention(ops, outputs, *, selection, views, plan, topk_present, prefix):
    ids = set(outputs)
    members = [op for op in ops if op.op_id in ids]
    rows = tuple((v.q_len, v.context) for v in views)
    shape = {"sequences": rows, "window": plan.window, "ratio": plan.compress_ratio,
             "topk": plan.index_topk if topk_present else 0, "heads": 64, "width": 512,
             "metadata_support_fd": False, "addressing": "scalar"}
    meta = {**dict(members[0].metadata), "mqsmla_geometry": shape,
            "logical_pairs": sum(dict(op.metadata).get("logical_pairs", 0) for op in members),
            "kv_sweeps": sum(dict(op.metadata).get("kv_sweeps", 0) for op in members),
            "implementation_binding": {
                "selector": "mqsmla_attention", "implementation": selection.name,
                "capture_binary_identity": "not_established", "source": SOURCE,
                "coverage": "conditional_source_program",
                "caller_unmodeled": ["separate_aicpu_metadata_generation",
                                      "original_window_index_generation"]},
            "attention_tensor_view": {
                "kind": "mixed_quant_shared_kv", "query_shape": (sum(q for q, _ in rows), 64, 512),
                "query_dtype": "bf16", "output_dtype": "bf16",
                "original_cache_row_bytes": 544, "compressed_cache_row_bytes": 320},
            "native_kind": "MqsmlaKernel___call__"}
    result = replace(members[0], op_id=prefix + ".mqsmla", recipe=RECIPE,
                     deps=tuple(dict.fromkeys(dep for op in members for dep in op.deps)),
                     semantic_shape=(sum(q for q, _ in rows), 64, 512),
                     local_shape=(sum(q for q, _ in rows), 64, 512),
                     padded_shape=(sum(q for q, _ in rows), 64, 512),
                     launch_shape=(sum(q for q, _ in rows), 64, 512),
                     flops=sum(op.flops for op in members),
                     vector_ops=sum(op.vector_ops for op in members),
                     hbm_bytes=sum(op.hbm_bytes for op in members),
                     mte2_bytes=sum(op.mte2_bytes for op in members),
                     mte3_bytes=sum(op.mte3_bytes for op in members),
                     source_boundary="kernel", metadata=tuple(meta.items()))
    ops[:] = [op for op in ops if op.op_id not in ids]
    ops.append(result)
    return [result.op_id]
