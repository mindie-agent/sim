"""Independent GQA context-KV preparation for external block and serial drafters.

[Role]
  Emit explicit context normalization, cross-layer fused KV-bank or per-layer
  quantized QKV projections, K RMSNorm + rotary slices, and paged KV scatter
  for external GQA drafters without MLA LoRA assumptions.

[Invariants]
  * ``fused_kv_bank`` projects ``2 * layers * local_kv_width`` across layers,
    applies per-layer K normalization and RoPE, and writes TP-sharded K+V rows.
  * ``per_layer_qkv`` charges the full executed quantized QKV projection
    (``local_q_width + 2 * local_kv_width``) per layer, including discarded Q,
    and writes only K+V cache slots.
  * ``dcp_sharded == False`` writes replicated context rows across DCP ranks
    while preserving TP KV-head sharding; ``dcp_sharded == True`` writes only
    rank-owned DCP rows.

[Timeouts & Locks]
  O(draft layers); pure deterministic graph construction with no locks or I/O.

[Side-Effect Policy]
  Appends immutable ``OpSpec`` records to the caller-owned operator list.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.ops import barrier_op, cache_store_op, gemm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes

NORM_OPS_PER_ELEMENT: int = 5
ROPE_OPS_PER_ELEMENT: int = 6
SCATTER_RECIPE: str = "draft_context_kv_scatter"
SCATTER_NATIVE_KIND: str = "aclnnScatterPaKvCache"


def _resolve_gqa_projection_mode(draft) -> str:
    declared = str(getattr(draft, "context_projection_mode", "") or "none").strip()
    if declared in {"fused_kv_bank", "per_layer_qkv"}:
        return declared
    if declared not in {"none", ""}:
        raise ModelingError(f"draft_context_gqa_mode_invalid:{declared}")
    return "per_layer_qkv" if getattr(draft, "weight_category", "") else "fused_kv_bank"


def prepare_gqa_context(
    ops,
    *,
    projection,
    model,
    draft,
    rows: int,
    query_rows: int,
    prefix: str,
    stage: int,
    requests: tuple[str, ...],
    parallel=None,
    owned_scatter_rows_fn=None,
    rows_by_view=None,
    views=None,
    contexts=None,
    block: int = 1,
) -> str:
    """Append external GQA context-KV preparation ops and return the ready op_id."""
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    cp = max(1, int(getattr(parallel, "cp", 1) or 1))
    kv_heads_local = local_kv_heads(int(draft.num_kv_heads), tp)
    head_dim = int(draft.head_dim)
    local_kv_width = kv_heads_local * head_dim
    local_q_width = (int(draft.q_width) + tp - 1) // tp
    if local_kv_width <= 0 or int(draft.layers) <= 0:
        raise ModelingError(
            f"draft_context_gqa_geometry_invalid:{draft.layers},{local_kv_width}"
        )
    rotary_dim = int(draft.qk_rope_head_dim) if int(draft.qk_rope_head_dim) > 0 else head_dim
    kv_dtype = str(getattr(model.quant, "kv_dtype", "") or "bf16")
    row_bytes = 2 * local_kv_width * int(dtype_bytes(kv_dtype))
    dcp_sharded = bool(getattr(draft, "dcp_sharded", False))
    if dcp_sharded and owned_scatter_rows_fn is not None:
        store_rows = int(
            owned_scatter_rows_fn(rows, rows_by_view, views, contexts, parallel, block)
        )
        effective_cp = cp
    else:
        store_rows = int(rows)
        effective_cp = 1

    common = dict(stage=stage, layer=-1, requests=requests, phase="draft")
    norm = vector_op(
        op_id=prefix + "context_norm",
        deps=(projection.op_id,),
        elements=rows * int(draft.hidden_size),
        ops_per_element=NORM_OPS_PER_ELEMENT,
        dtype=draft.activation_dtype,
        family="rmsnorm",
        recipe="draft_context_norm",
        **common,
    )
    ops.append(norm)

    mode = _resolve_gqa_projection_mode(draft)
    proj_category = str(draft.weight_category or "attn.qkv")
    bank_op_id: str | None = None
    if mode == "fused_kv_bank":
        bank_width = 2 * int(draft.layers) * local_kv_width
        bank = gemm_op(
            op_id=prefix + "context_kv_bank",
            deps=(norm.op_id,),
            m=rows,
            n=bank_width,
            k=int(draft.hidden_size),
            quant=model.quant,
            category=proj_category,
            family="gemm",
            recipe="draft_context_kv_bank",
            **common,
        )
        bank = replace(
            bank,
            metadata=bank.metadata + (
                ("context_projection_mode", "fused_kv_bank"),
                ("layers", int(draft.layers)),
                ("local_kv_width", local_kv_width),
                ("tp_degree", tp),
            ),
        )
        ops.append(bank)
        bank_op_id = bank.op_id

    stores: list[str] = []
    for layer in range(int(draft.layers)):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        if mode == "per_layer_qkv":
            qkv_width = local_q_width + 2 * local_kv_width
            qkv_proj = gemm_op(
                op_id=name + ".qkv_proj",
                deps=(norm.op_id,),
                m=rows,
                n=qkv_width,
                k=int(draft.hidden_size),
                quant=model.quant,
                category=proj_category,
                family="gemm",
                recipe="draft_context_qkv_proj",
                **layer_common,
            )
            qkv_proj = replace(
                qkv_proj,
                metadata=qkv_proj.metadata + (
                    ("context_projection_mode", "per_layer_qkv"),
                    ("discarded_q_width", local_q_width),
                    ("local_kv_width", local_kv_width),
                    ("tp_degree", tp),
                ),
            )
            ops.append(qkv_proj)
            proj_ready = qkv_proj.op_id
        else:
            assert bank_op_id is not None
            proj_ready = bank_op_id

        k_norm = vector_op(
            op_id=name + ".k_norm",
            deps=(proj_ready,),
            elements=rows * local_kv_width,
            ops_per_element=NORM_OPS_PER_ELEMENT,
            dtype=draft.activation_dtype,
            family="rmsnorm",
            recipe="draft_context_k_norm",
            **layer_common,
        )
        ops.append(k_norm)
        kv_ready = k_norm.op_id
        if rotary_dim > 0:
            k_rope = vector_op(
                op_id=name + ".k_rope",
                deps=(k_norm.op_id,),
                elements=rows * kv_heads_local * rotary_dim,
                ops_per_element=ROPE_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype,
                family="rotary",
                recipe="draft_context_k_rope",
                **layer_common,
            )
            ops.append(k_rope)
            kv_ready = k_rope.op_id

        stored = cache_store_op(
            op_id=name + ".kv_scatter",
            deps=(kv_ready,),
            rows=store_rows,
            row_bytes=row_bytes,
            dtype=kv_dtype,
            kind="draft_context",
            recipe=SCATTER_RECIPE,
            **layer_common,
        )
        stored = replace(
            stored,
            metadata=stored.metadata + (
                ("native_kind", SCATTER_NATIVE_KIND),
                (
                    "cache_epilog",
                    {
                        "kind": SCATTER_NATIVE_KIND,
                        "context_rows": rows,
                        "rank_owned_rows": store_rows,
                        "kv_row_bytes": row_bytes,
                        "row_elements": 2 * local_kv_width,
                    },
                ),
                ("context_projection_mode", mode),
                ("dcp_sharded", dcp_sharded),
                ("dcp_row_shard", store_rows),
                ("dcp_degree", effective_cp),
                ("tp_degree", tp),
            ),
        )
        ops.append(stored)
        stores.append(stored.op_id)

    embedding = vector_op(
        op_id=prefix + "query_embedding",
        deps=(),
        elements=query_rows * int(draft.hidden_size),
        ops_per_element=0,
        dtype=draft.activation_dtype,
        family="embedding",
        recipe="draft_query_embedding",
        **common,
    )
    ready = barrier_op(
        op_id=prefix + "context_query_ready",
        deps=(*stores, embedding.op_id),
        recipe="draft_context_query_ready",
        shape=(rows, query_rows),
        **common,
    )
    ops.extend((embedding, ready))
    return ready.op_id


__all__ = ["prepare_gqa_context"]
