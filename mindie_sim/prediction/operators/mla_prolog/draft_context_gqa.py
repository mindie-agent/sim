"""Independent GQA context-KV preparation for external block and serial drafters.

[Role]
  Emit explicit context normalization, cross-layer fused KV-bank (with
  contiguous transpose and coalesced all-layer RoPE) or per-layer quantized QKV
  projections, K RMSNorm + per-layer rotary slices, and paged KV scatter for
  external GQA drafters without MLA LoRA assumptions.

[Invariants]
  * ``fused_kv_bank`` projects ``2 * layers * local_kv_width`` across layers,
    emits a contiguous transpose (``draft_context_kv_transpose``) before
    per-layer K normalization, coalesces uniform K-RoPE into one flattened
    all-layer call, and writes TP-sharded K+V rows.
  * ``per_layer_qkv`` charges the full executed quantized QKV projection
    (``local_q_width + 2 * local_kv_width``) per layer, including discarded Q,
    applies per-layer K norm and ``qk_rope_head_dim`` RoPE, and writes only
    K+V cache slots.
  * ``dcp_sharded == False`` writes replicated context rows across DCP ranks
    while preserving TP KV-head sharding; ``dcp_sharded == True`` writes only
    rank-owned DCP rows (intersecting per-row slot validity masks when
    provided). Layers with absent or empty ``layer_slot_valid_rows`` skip
    scatter emission.

[Timeouts & Locks]
  O(draft layers); pure deterministic graph construction with no locks or I/O.

[Side-Effect Policy]
  Appends immutable ``OpSpec`` records to the caller-owned operator list.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from mindie_sim.catalog.attention_common import store_dcp_degree
from mindie_sim.catalog.cache import local_kv_heads
from mindie_sim.catalog.ops import barrier_op, cache_store_op, gemm_op, vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes

NORM_OPS_PER_ELEMENT: int = 5
ROPE_OPS_PER_ELEMENT: int = 6
SLOT_INDEX_BYTES: int = 4
SCATTER_RECIPE: str = "draft_context_kv_scatter"
SCATTER_NATIVE_KIND: str = "aclnnScatterPaKvCache"
TRANSPOSE_RECIPE: str = "draft_context_kv_transpose"


def _resolve_gqa_projection_mode(draft) -> str:
    declared = str(getattr(draft, "context_projection_mode", "") or "none").strip()
    if declared in {"fused_kv_bank", "per_layer_qkv"}:
        return declared
    if declared not in {"none", ""}:
        raise ModelingError(f"draft_context_gqa_mode_invalid:{declared}")
    return "per_layer_qkv" if getattr(draft, "weight_category", "") else "fused_kv_bank"


def _layer_rotary_dim(draft, layer: int, head_dim: int) -> int:
    """Resolve per-layer GQA rotary head width, honoring ``layer_plans[layer].qk_rope_head_dim``."""
    layer_plans = getattr(draft, "layer_plans", ()) or ()
    if layer < len(layer_plans):
        plan = layer_plans[layer]
        plan_rope = int(getattr(plan, "qk_rope_head_dim", 0) or 0)
        if plan_rope > 0:
            return min(plan_rope, head_dim)
        if int(getattr(plan, "qk_nope_head_dim", 0) or 0) >= head_dim:
            return 0
    draft_rope = int(getattr(draft, "qk_rope_head_dim", 0) or 0)
    return min(draft_rope, head_dim) if draft_rope > 0 else head_dim


def _owned_masked_rows(valid_flags, rows_by_view, views, contexts, parallel, block: int) -> int:
    """Intersect per-row validity flags with DCP cache-store block ownership across views."""
    degrees = [store_dcp_degree(parallel, view.phase) for view in views]
    max_degree = max(degrees, default=1)
    interleave = max(1, int(block))
    per_rank = [0] * max_degree
    offset = 0
    for start, count, degree in zip(contexts, rows_by_view, degrees):
        span = max(0, int(count))
        view_flags = valid_flags[offset:offset + span]
        offset += span
        if degree <= 1:
            valid_in_view = sum(1 for ok in view_flags if ok)
            for rank in range(max_degree):
                per_rank[rank] += valid_in_view
            continue
        base_pos = int(start)
        for idx, ok in enumerate(view_flags):
            if ok:
                per_rank[((base_pos + idx) // interleave) % degree] += 1
    return max(per_rank, default=0)


def _layer_scatter_rows(
    layer_slot_valid_rows,
    layer: int,
    *,
    rows: int,
    store_rows: int,
    dcp_sharded: bool,
    cp: int,
    owned_scatter_rows_fn,
    rows_by_view,
    views,
    contexts,
    parallel,
    block: int,
) -> int | None:
    """Return per-layer scatter row count (``0`` for non-empty all-PAD mask), or ``None`` to skip."""
    del cp, owned_scatter_rows_fn
    if layer_slot_valid_rows is None:
        return store_rows
    layer_valid = layer_slot_valid_rows[layer] if layer < len(layer_slot_valid_rows) else None
    if layer_valid is None:
        return None
    if isinstance(layer_valid, Sequence) and not isinstance(layer_valid, (str, bytes)):
        clipped = layer_valid[:int(rows)]
        if not clipped:
            return None
        flags = tuple(bool(x) if isinstance(x, bool) else int(x) >= 0 for x in clipped)
        v_rows = sum(1 for ok in flags if ok)
        if v_rows <= 0:
            return 0
        if not dcp_sharded:
            return min(store_rows, v_rows)
        if rows_by_view and views and contexts:
            return min(store_rows, _owned_masked_rows(flags, rows_by_view, views, contexts, parallel, block))
        return min(store_rows, v_rows)
    v_rows = max(0, min(int(rows), int(layer_valid)))
    if v_rows <= 0:
        return None
    return min(store_rows, v_rows)


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
    layer_slot_valid_rows=None,
) -> str:
    """Append external GQA context-KV preparation ops and return the ready op_id."""
    tp = max(1, int(getattr(parallel, "tp", 1) or 1))
    cp = max(1, int(getattr(parallel, "cp", 1) or 1))
    kv_heads_local = local_kv_heads(int(draft.num_kv_heads), tp)
    head_dim = int(draft.head_dim)
    num_layers = int(draft.layers)
    local_kv_width = kv_heads_local * head_dim
    local_q_width = (int(draft.q_width) + tp - 1) // tp
    if local_kv_width <= 0 or num_layers <= 0:
        raise ModelingError(
            f"draft_context_gqa_geometry_invalid:{draft.layers},{local_kv_width}"
        )
    kv_dtype = str(getattr(model.quant, "kv_dtype", "") or "bf16")
    row_bytes = 2 * local_kv_width * int(dtype_bytes(kv_dtype))
    dcp_sharded = bool(getattr(draft, "dcp_sharded", False))
    if dcp_sharded and rows_by_view and views and contexts:
        store_rows = _owned_masked_rows((True,) * int(rows), rows_by_view, views, contexts, parallel, block)
        effective_cp = cp
    elif dcp_sharded and owned_scatter_rows_fn is not None:
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
    layer_rotary_dims = tuple(_layer_rotary_dim(draft, layer, head_dim) for layer in range(num_layers))
    if mode == "fused_kv_bank":
        layer_kv_ready = _emit_fused_bank_kv(
            ops, draft, model, norm, prefix, rows, num_layers, kv_heads_local,
            head_dim, local_kv_width, proj_category, common, tp, layer_rotary_dims,
        )
    else:
        layer_kv_ready = _emit_per_layer_qkv(
            ops, draft, model, norm, prefix, rows, num_layers, kv_heads_local,
            local_q_width, local_kv_width, proj_category, common, tp, layer_rotary_dims,
        )

    ready_deps: list[str] = []
    for layer in range(num_layers):
        layer_store_rows = _layer_scatter_rows(
            layer_slot_valid_rows, layer, rows=rows, store_rows=store_rows,
            dcp_sharded=dcp_sharded, cp=cp, owned_scatter_rows_fn=owned_scatter_rows_fn,
            rows_by_view=rows_by_view, views=views, contexts=contexts,
            parallel=parallel, block=block,
        )
        kv_ready = layer_kv_ready[layer]
        if layer_store_rows is None:
            if kv_ready not in ready_deps:
                ready_deps.append(kv_ready)
            continue
        slot_count = 0
        if layer_store_rows == 0 and layer_slot_valid_rows is not None and layer < len(layer_slot_valid_rows):
            raw_slots = layer_slot_valid_rows[layer]
            if isinstance(raw_slots, Sequence) and not isinstance(raw_slots, (str, bytes)):
                slot_count = len(raw_slots[:int(rows)])
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        ready_deps.append(_scatter_context_kv(
            ops, name, kv_ready, layer_store_rows, row_bytes, kv_dtype, layer_common,
            rows, local_kv_width, mode, dcp_sharded, effective_cp, tp,
            slot_index_count=slot_count,
        ))

    if int(query_rows) > 0:
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
            deps=(*ready_deps, embedding.op_id),
            recipe="draft_context_query_ready",
            shape=(rows, query_rows),
            **common,
        )
        ops.extend((embedding, ready))
        return ready.op_id
    ready = barrier_op(
        op_id=prefix + "context_query_ready",
        deps=tuple(ready_deps),
        recipe="draft_context_query_ready",
        shape=(rows, 0),
        **common,
    )
    ops.append(ready)
    return ready.op_id


def _emit_fused_bank_kv(
    ops, draft, model, norm, prefix, rows, num_layers, kv_heads_local,
    head_dim, local_kv_width, proj_category, common, tp, layer_rotary_dims,
) -> tuple[str, ...]:
    """Emit fused KV-bank GEMM, contiguous transpose (when non-contiguous), K norm, and coalesced K-RoPE."""
    bank_width = 2 * num_layers * local_kv_width
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
            ("layers", num_layers),
            ("local_kv_width", local_kv_width),
            ("tp_degree", tp),
        ),
    )
    ops.append(bank)
    k_norm_dep = bank.op_id
    # (rows, layers, 2, nkv, hd).permute(2, 1, 0, 3, 4) is already contiguous when layers==1 and rows==1.
    if num_layers > 1 or rows > 1:
        bank_out_dtype = str(dict(bank.metadata).get("output_format") or draft.activation_dtype)
        transpose = vector_op(
            op_id=prefix + "context_kv_transpose",
            deps=(bank.op_id,),
            elements=rows * bank_width,
            ops_per_element=0,
            dtype=bank_out_dtype,
            family="copy",
            recipe=TRANSPOSE_RECIPE,
            shape=(2, num_layers, rows, kv_heads_local, head_dim),
            metadata=(
                ("context_projection_mode", "fused_kv_bank"),
                ("layers", num_layers),
                ("local_kv_width", local_kv_width),
                ("tp_degree", tp),
            ),
            **common,
        )
        ops.append(transpose)
        k_norm_dep = transpose.op_id

    k_norm_ids: list[str] = []
    for layer in range(num_layers):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
        k_norm = vector_op(
            op_id=name + ".k_norm",
            deps=(k_norm_dep,),
            elements=rows * local_kv_width,
            ops_per_element=NORM_OPS_PER_ELEMENT,
            dtype=draft.activation_dtype,
            family="rmsnorm",
            recipe="draft_context_k_norm",
            **layer_common,
        )
        ops.append(k_norm)
        k_norm_ids.append(k_norm.op_id)

    if len(set(layer_rotary_dims)) == 1:
        rotary_dim = layer_rotary_dims[0]
        if rotary_dim <= 0:
            return tuple(k_norm_ids)
        fused_rope = vector_op(
            op_id=prefix + "context_k_rope",
            deps=tuple(k_norm_ids),
            elements=num_layers * rows * kv_heads_local * rotary_dim,
            ops_per_element=ROPE_OPS_PER_ELEMENT,
            dtype=draft.activation_dtype,
            family="rotary",
            recipe="draft_context_k_rope",
            shape=(num_layers * rows, local_kv_width),
            metadata=(
                ("context_projection_mode", "fused_kv_bank"),
                ("layers", num_layers),
                ("coalesced_layers", num_layers),
                ("rotary_dim", rotary_dim),
            ),
            **common,
        )
        ops.append(fused_rope)
        return (fused_rope.op_id,) * num_layers

    ready_ids: list[str] = []
    for layer, rotary_dim in enumerate(layer_rotary_dims):
        kv_ready = k_norm_ids[layer]
        if rotary_dim > 0:
            name = prefix + f"context_kv{layer}"
            k_rope = vector_op(
                op_id=name + ".k_rope",
                deps=(kv_ready,),
                elements=rows * kv_heads_local * rotary_dim,
                ops_per_element=ROPE_OPS_PER_ELEMENT,
                dtype=draft.activation_dtype,
                family="rotary",
                recipe="draft_context_k_rope",
                **{**common, "layer": layer},
            )
            ops.append(k_rope)
            kv_ready = k_rope.op_id
        ready_ids.append(kv_ready)
    return tuple(ready_ids)


def _emit_per_layer_qkv(
    ops, draft, model, norm, prefix, rows, num_layers, kv_heads_local,
    local_q_width, local_kv_width, proj_category, common, tp, layer_rotary_dims,
) -> tuple[str, ...]:
    """Emit per-layer quantized QKV projections, K RMSNorm, and per-layer K-RoPE."""
    ready_ids: list[str] = []
    qkv_width = local_q_width + 2 * local_kv_width
    for layer in range(num_layers):
        name = prefix + f"context_kv{layer}"
        layer_common = {**common, "layer": layer}
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
        k_norm = vector_op(
            op_id=name + ".k_norm",
            deps=(qkv_proj.op_id,),
            elements=rows * local_kv_width,
            ops_per_element=NORM_OPS_PER_ELEMENT,
            dtype=draft.activation_dtype,
            family="rmsnorm",
            recipe="draft_context_k_norm",
            **layer_common,
        )
        ops.extend((qkv_proj, k_norm))
        kv_ready = k_norm.op_id
        rotary_dim = layer_rotary_dims[layer]
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
        ready_ids.append(kv_ready)
    return tuple(ready_ids)


def _scatter_context_kv(
    ops, name, kv_ready, store_rows, row_bytes, kv_dtype, layer_common, rows, local_kv_width,
    mode, dcp_sharded, effective_cp, tp, *, slot_index_count: int = 0,
):
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
    if store_rows == 0 and slot_index_count > 0:
        slot_bytes = float(slot_index_count * SLOT_INDEX_BYTES)
        stored = replace(
            stored,
            mte2_bytes=slot_bytes,
            mte3_bytes=0.0,
            hbm_bytes=slot_bytes,
            vector_ops=float(slot_index_count),
        )
    extra_meta = (("slot_index_count", slot_index_count),) if slot_index_count > 0 else ()
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
        ) + extra_meta,
    )
    ops.append(stored)
    return stored.op_id


__all__ = ["prepare_gqa_context"]
