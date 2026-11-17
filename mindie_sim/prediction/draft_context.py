"""Independent context-KV preparation for external block drafters.

[Role] Separate target-feature context rows from proposal query rows.
[Invariants] Each draft layer computes and stores its own BF16 context KV.
  Context projection is not reused as the query embedding.
[Timeouts & Locks] O(draft layers); no locks or external work.
[Side-Effect Policy] Appends only to the caller-owned operator list.
"""
from mindie_sim.catalog.ops import gemm_op, vector_op, cache_store_op, barrier_op
from mindie_sim.schema.common import dtype_bytes

NORM_OPS = 5
ROPE_OPS = 6


def prepare(ops, *, projection, model, draft, rows, query_rows,
            prefix, stage, requests):
    common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
    norm = vector_op(
        op_id=prefix + "context_norm", deps=(projection.op_id,),
        elements=rows * draft.hidden_size, ops_per_element=NORM_OPS,
        dtype="bf16", family="rmsnorm", recipe="draft_context_norm", **common)
    ops.append(norm)
    stores = []
    for layer in range(draft.layers):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        qkv = gemm_op(
            op_id=name + ".qkv_a", deps=(norm.op_id,),
            m=rows, n=draft.q_lora_rank + draft.kv_lora_rank + draft.qk_rope_head_dim,
            k=draft.hidden_size, quant=model.quant, category="mla",
            recipe="draft_context_qkv_a", **layer_common)
        kv_width = draft.kv_lora_rank + draft.qk_rope_head_dim
        # Source .contiguous() copies the KV suffix out of fused QKV-A.
        copied = vector_op(
            op_id=name + ".kv_copy", deps=(qkv.op_id,),
            elements=rows * kv_width, ops_per_element=0,
            dtype="bf16", family="elementwise", recipe="draft_context_kv_copy",
            **layer_common)
        normalized = vector_op(
            op_id=name + ".norm", deps=(copied.op_id,),
            elements=rows * draft.kv_lora_rank, ops_per_element=NORM_OPS,
            dtype="bf16", family="rmsnorm", recipe="draft_context_kv_norm",
            **layer_common)
        rotary = vector_op(
            op_id=name + ".rope", deps=(normalized.op_id,),
            elements=rows * draft.qk_rope_head_dim, ops_per_element=ROPE_OPS,
            dtype="bf16", family="rotary", recipe="draft_context_rope",
            **layer_common)
        kv_dtype = str(getattr(model, "kv_dtype", "") or "bf16")
        stored = cache_store_op(
            op_id=name + ".store", deps=(rotary.op_id,), rows=rows,
            row_bytes=int(kv_width * dtype_bytes(kv_dtype)), dtype=kv_dtype,
            kind="draft_context",
            recipe="draft_context_store", **layer_common)
        ops.extend((qkv, copied, normalized, rotary, stored))
        stores.append(stored.op_id)
    embedding = vector_op(
        op_id=prefix + "query_embedding", deps=(), elements=query_rows * draft.hidden_size,
        ops_per_element=0, dtype="bf16", family="embedding",
        recipe="draft_query_embedding", **common)
    ready = barrier_op(
        op_id=prefix + "context_query_ready", deps=(*stores, embedding.op_id),
        recipe="draft_context_query_ready", shape=(rows, query_rows), **common)
    ops.extend((embedding, ready))
    return ready.op_id
