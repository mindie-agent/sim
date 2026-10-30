"""Tuple-query MLA decode caller on the ordinary latent-attention frontend.

[Role] Compose caller materializations and the existing two-kernel source plan.
[Invariants] This binding selects dense BF16 decode with one query per request.
  Tuple concat and both zero fills are the pinned caller's actual operations.
  Split count is declared dispatch output, never inferred from timing evidence.
[Timeouts & Locks] O(batch) metadata; no device contexts, source reads or locks.
[Side-Effect Policy] Appends immutable graph records to the supplied builder.
"""
from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from .implementations import preserve_graph
from .mechanism import PlannedRecipe
from .operators import mla_decode


IMPLEMENTATION = "vllm_triton_mla_decode_tuple"


def recipes():
    return {mla_decode.IMPLEMENTATION: PlannedRecipe(mla_decode.build, mla_decode.kernels)}


def validate(*, views, plan, parallel, model, quant, padded_tokens, num_kv_splits=None):
    if (parallel.cp != 1 or parallel.pcp != 1
            or any(view.phase != "decode" or view.q_len != 1 for view in views)
            or padded_tokens != len(views)):
        raise ModelingError("mla_decode_requires_unpartitioned_unpadded_single_token_decode_batch")
    if (plan.mla_variant != "latent_mla" or plan.num_kv_heads != 1 or plan.cache_kind != "dense"
            or plan.window or plan.compress_ratio or plan.indexer or plan.index_topk
            or plan.owns_index_k or plan.owns_topk or plan.index_query_per_layer
            or plan.topk_owner_layer_id >= 0 or plan.candidate_filter):
        raise ModelingError("mla_decode_requires_dense_latent_cache_without_sparse_selection")
    if (model.activation_dtype != "bf16" or quant.kv_dtype != "bf16"
            or plan.cache_row_elements != plan.kv_lora_rank + plan.qk_rope_head_dim):
        raise ModelingError("mla_decode_requires_bf16_latent_plus_rope_cache")
    if type(num_kv_splits) is not int or num_kv_splits < 1:
        raise ModelingError("mla_decode_requires_explicit_positive_split_count")


def lower_attention(builder, *, views, plan, heads, runtime, query, rotary, store,
                    num_kv_splits):
    batch, latent, rope = len(views), plan.kv_lora_rank, plan.qk_rope_head_dim
    producer = next(op for op in builder.ops if op.op_id == query)
    if dict(producer.metadata).get("output_format") != "bf16":
        raise ModelingError("mla_decode_requires_bf16_absorbed_query_output")
    caller = {
        "implementation": IMPLEMENTATION, "query_input": "tuple_latent_and_rotary",
        "source": mla_decode.SOURCE_FILES[0], "caller_lines": "199-213",
        "split_selection": "declared_caller_dispatch_result",
    }
    query_bytes = batch * heads * (latent + rope) * 2
    packed = builder.work(
        "mla_query_concat", (query, rotary), (batch, heads, latent + rope),
        read=query_bytes, write=query_bytes, family="tensor_copy", dtype="bf16",
        metadata=(("caller", caller), ("materialization", "torch.cat(dim=-1)")))
    output_zero = builder.work(
        "mla_output_zero", (packed,), (batch, heads, latent),
        write=batch * heads * latent * 2, family="zero_fill", dtype="bf16",
        metadata=(("caller", caller), ("materialization", "torch.zeros(o)")))
    lse_zero = builder.work(
        "mla_lse_zero", (packed,), (batch, heads),
        write=batch * heads * 2, family="zero_fill", dtype="bf16",
        metadata=(("caller", caller), ("materialization", "torch.zeros(lse)")))
    name = builder.work(
        "latent_attention_decode", (packed, output_zero, lse_zero, store),
        (batch, heads, latent + rope, latent), family="attention", dtype="bf16",
        metadata=(("logical_pairs", sum(view.kv_len for view in views)), ("caller", caller)))
    source = mla_decode.make_op({
        "implementation": mla_decode.IMPLEMENTATION, "contexts": [v.kv_len for v in views],
        "heads": heads, "key_heads": 1, "latent_dim": latent, "rope_dim": rope,
        "page_size": runtime.kv_block_size, "num_kv_splits": num_kv_splits,
        "dtype": "bf16", "cache_dtype": "bf16", "is_mla": True, "is_hip": False,
    })
    source = preserve_graph(source, builder.ops[-1])
    builder.ops[-1] = replace(source, metadata=source.metadata + (("implementation_binding", {
        "selector": "mla_decode", "implementation": IMPLEMENTATION,
        "call_batching": "one_batch_of_single_token_decode_requests",
    }),))
    return name
