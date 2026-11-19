"""Sparse GQA indexer lowering and native index-cache write delegation.

[Role]
  Emit sparse GQA index projections, index-key cache writes, PCP index gathers,
  prefix-reused score sweeps, and TopK reducers, while consumer layers with
  ``owns_topk == False`` reuse ``topk_owner_layer_id`` without duplicate ops.

[Invariants]
  * Delegates index-key cache writing to ``index_cache_writer.emit_index_cache_write``
    when available outside ``raw_eager``, falling back cleanly to ``cache_store_op``.
  * Only explicit caller bindings (origin ``"caller"``) pin ``user_pinned_idx``.

[Timeouts & Locks]
  O(requests) arithmetic per sparse GQA layer; no locks or I/O.

[Side-Effect Policy]
  Appends immutable ``OpSpec`` records to the caller-owned ``ops`` list.
"""
from __future__ import annotations

from mindie_sim.catalog.attention_common import _long_selected_pairs
from mindie_sim.catalog.cache import (
    completed_rows,
    index_row_bytes,
    index_selection_cost,
)
from mindie_sim.catalog.ops import cache_store_op, comm_op, gemm_op, vector_op
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .batch_prefix_indexer import apply_indexer_prefix_reads
from .generic_latent_attention import index_key_window, visible_parts
from .implementation_selection import is_caller_pinned_binding

try:
    from mindie_sim.prediction.index_cache_writer import emit_index_cache_write
except ImportError:
    emit_index_cache_write = None

INDEX_SCORE_VECTOR_OPS_PER_ROW: int = 3
TOPK_OPS_PER_ELEMENT: int = 4
DEFAULT_KV_BLOCK_SIZE: int = 128
DEFAULT_INDEX_DIM: int = 64
FP32_SCORE_BYTES: int = 4
INT32_INDEX_BYTES: int = 4


def _emit_index_store(
    *,
    prefix: str,
    deps: tuple[str, ...],
    stage: int,
    layer: int,
    requests: tuple[str, ...],
    phase: str,
    rows: int,
    row_bytes: float,
    idx_dim: int,
    plan,
    model,
    quant,
    runtime,
    machine,
) -> tuple[OpSpec, ...]:
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    raw_eager = bool(
        rt_meta.get("raw_eager")
        or (rt_meta.get("execution_policy") or {}).get("raw_eager")
    )
    if emit_index_cache_write is not None and not raw_eager:
        emitted = emit_index_cache_write(
            prefix=prefix,
            deps=deps,
            stage=stage,
            layer=layer,
            requests=requests,
            phase=phase,
            rows=rows,
            row_bytes=row_bytes,
            index_dim=idx_dim,
            plan=plan,
            model=model,
            quant=quant,
            runtime=runtime,
            machine=machine,
        )
        if isinstance(emitted, OpSpec):
            return (emitted,)
        if emitted:
            return tuple(emitted)
    fallback = cache_store_op(
        op_id=prefix + ".index_store",
        deps=deps,
        stage=stage,
        layer=layer,
        requests=requests,
        phase=phase,
        recipe="index_store",
        rows=rows,
        row_bytes=max(1, int(round(row_bytes))),
        dtype=quant.index_dtype,
        kind="index",
    )
    return (fallback,)


def lower_gqa_indexer(
    ops,
    *,
    prefix: str,
    plan,
    views,
    stage: int,
    deps: str,
    model,
    parallel,
    quant,
    mode: str,
    padded_tokens: int,
    valid: int,
    pcp_prefill: bool,
    runtime=None,
    topk_ids=None,
    index_cache_ids=None,
    machine=None,
) -> tuple[int, int, int, str | None, str | None, str | None]:
    """Lower GQA sparse indexer ops; return ``(eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id)``."""
    plans = model.layer_plans
    idx_raw, sel_raw = int(plan.index_k_owner_layer_id), int(plan.topk_owner_layer_id)
    if plan.indexer:
        idx_owner = sel_owner = plan.index
    else:
        idx_owner = idx_raw if 0 <= idx_raw < plan.index else -1
        sel_owner = (
            sel_raw
            if 0 <= sel_raw < min(plan.index, len(plans))
            and (plans[sel_raw].indexer or plans[sel_raw].owns_topk)
            else idx_owner
        )
    has_indexer = bool(
        plan.indexer or plan.owns_topk or idx_owner >= 0 or sel_owner >= 0
    )
    eff_topk = int(plan.index_topk) if has_indexer else 0
    index_store_id = candidate_id = None
    topk_id = (topk_ids or {}).get(sel_owner) if sel_owner >= 0 else None
    idx_ratio = max(1, int(plan.compress_ratio or 1), int(plan.index_kpool or 1))
    sel_topk = max(1, eff_topk // idx_ratio) if idx_ratio > 1 else eff_topk
    block_size = max(
        1,
        int(
            getattr(runtime, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
            or DEFAULT_KV_BLOCK_SIZE
        ),
    )
    user_pinned_idx = (
        is_caller_pinned_binding(runtime, "families", "indexer")
        or is_caller_pinned_binding(runtime, "recipes", "index_scores")
    )
    requests = tuple(view.request_id for view in views)
    if eff_topk > 0 and (plan.indexer or plan.owns_index_k or plan.owns_topk):
        cost = index_selection_cost(
            model, plan, max((v.kv_len for v in views), default=0)
        )
        row_bytes = cost.index_row_bytes or index_row_bytes(model, plan.index_dim)
        idx_heads = max(1, int(plan.index_heads or 1))
        idx_dim = max(1, int(plan.index_dim or DEFAULT_INDEX_DIM))
        cand_cap = (
            plan.candidate_topk_blocks * plan.candidate_block_size
            if plan.candidate_filter
            else 0
        )
        new_idx_rows = (
            sum(
                completed_rows(v.context + e_p, idx_ratio)
                - completed_rows(v.context + s_p, idx_ratio)
                for v in views
                for s_p, e_p in (v.query_ranges or ((0, v.q_len),))
                if e_p > s_p
            )
            if idx_ratio > 1
            else valid
        )
        rows_by_view: list[int] = []
        shared_idx_by_view: list[int] = []
        pairs_by_view: list[int] = []
        for v in views:
            v_ranges = v.query_ranges or ((0, v.q_len),)
            v_scored = int(
                sum(
                    _long_selected_pairs(
                        e_p - s_p,
                        v.context + s_p,
                        idx_ratio,
                        cand_cap,
                        noncausal=bool(plan.noncausal_attention),
                        kv_len=v.kv_len,
                    )
                    for s_p, e_p in v_ranges
                    if e_p > s_p
                )
            )
            raw_pref = max(
                0, min(int(getattr(v, "prefix_tokens", 0) or 0), int(v.context))
            )
            aligned_pref = ((raw_pref // block_size) * block_size) // idx_ratio
            physical_rows, _ = index_key_window(
                visible_parts(v),
                noncausal=bool(plan.noncausal_attention),
                kv_len=v.kv_len,
                compression=idx_ratio,
            )
            pairs_by_view.append(v_scored)
            rows_by_view.append(physical_rows)
            shared_idx_by_view.append(min(physical_rows, aligned_pref))
        scored_rows = sum(pairs_by_view)
        idx_wq = gemm_op(
            op_id=prefix + ".index_wq",
            deps=(deps,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            m=padded_tokens,
            n=idx_heads * idx_dim,
            k=model.hidden_size,
            quant=quant,
            category="indexer",
            phase=mode,
            recipe="index_wq",
            family="index_projection",
        )
        idx_wk = gemm_op(
            op_id=prefix + ".index_wk",
            deps=(deps,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            m=padded_tokens,
            n=idx_dim,
            k=model.hidden_size,
            quant=quant,
            category="indexer",
            phase=mode,
            recipe="index_wk",
            family="index_projection",
        )
        store_ops = _emit_index_store(
            prefix=prefix,
            deps=(idx_wk.op_id,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            phase=mode,
            rows=new_idx_rows,
            row_bytes=row_bytes,
            idx_dim=idx_dim,
            plan=plan,
            model=model,
            quant=quant,
            runtime=runtime,
            machine=machine,
        )
        idx_store_id = store_ops[-1].op_id
        pcp_idx_ops: tuple[OpSpec, ...] = ()
        idx_ready = idx_store_id
        if pcp_prefill:
            if plan.index_kpool > 1:
                global_idx_rows = sum(
                    (v.context + (v.global_q_len or v.q_len)) // plan.index_kpool
                    - v.context // plan.index_kpool
                    for v in views
                    if v.phase == "prefill"
                )
            else:
                global_idx_rows = sum(
                    completed_rows(v.context + (v.global_q_len or v.q_len), idx_ratio)
                    - completed_rows(v.context, idx_ratio)
                    for v in views
                    if v.phase == "prefill"
                )
            idx_gather = comm_op(
                op_id=prefix + ".pcp_index_gather",
                deps=(idx_store_id,),
                stage=stage,
                layer=plan.index,
                requests=requests,
                family="allgather",
                phase=mode,
                recipe="pcp_index_gather",
                link_bytes=float(
                    global_idx_rows * row_bytes * (parallel.pcp - 1) / parallel.pcp
                ),
                shape=(max(1, global_idx_rows), idx_dim),
                metadata=(("group_axis", "pcp"),),
            )
            pcp_idx_ops = (idx_gather,)
            idx_ready = idx_gather.op_id
        q_idx_bytes = valid * idx_heads * idx_dim * dtype_bytes(quant.index_dtype)
        unique_idx_rows = sum(rows_by_view)
        l2_bounded_index_read = float(min(scored_rows, unique_idx_rows)) * row_bytes
        shape = (scored_rows, idx_heads, idx_dim)
        score_bytes = float(scored_rows * FP32_SCORE_BYTES)
        idx_scores = OpSpec(
            op_id=prefix + ".index_scores",
            family="indexer",
            phase=mode,
            stage=stage,
            layer=plan.index,
            requests=requests,
            semantic_shape=shape,
            local_shape=shape,
            padded_shape=shape,
            launch_shape=shape,
            flops=float(2 * scored_rows * idx_heads * idx_dim),
            vector_ops=float(INDEX_SCORE_VECTOR_OPS_PER_ROW * scored_rows * idx_heads),
            hbm_bytes=float(l2_bounded_index_read + q_idx_bytes + score_bytes),
            mte2_bytes=float(scored_rows * row_bytes + q_idx_bytes),
            mte3_bytes=score_bytes,
            deps=(idx_wq.op_id, idx_ready),
            recipe="index_scores",
            dtype=quant.index_dtype,
            metadata=(("score_materialization", "separate_fp32_score_then_topk"),),
        )
        idx_scores = apply_indexer_prefix_reads(
            idx_scores,
            views=views,
            rows_by_view=rows_by_view,
            shared_rows_by_view=shared_idx_by_view,
            row_bytes=float(row_bytes),
            layer_index=plan.index,
            runtime=runtime,
            compression_ratio=idx_ratio,
            kernel_reuse=(
                "native_per_sequence"
                if user_pinned_idx or plan.candidate_filter
                else "tile"
            ),
            allow_recipe_rebind=not user_pinned_idx,
            hbm_component=l2_bounded_index_read,
            mte2_component=float(scored_rows) * row_bytes,
        )
        idx_topk = vector_op(
            op_id=prefix + ".index_topk",
            deps=(idx_scores.op_id,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            elements=scored_rows,
            ops_per_element=TOPK_OPS_PER_ELEMENT,
            dtype="fp32",
            family="topk",
            phase=mode,
            recipe="index_topk",
            shape=(valid, sel_topk),
            read_bytes_override=float(scored_rows * FP32_SCORE_BYTES),
            write_bytes_override=float(valid * sel_topk * INT32_INDEX_BYTES),
            metadata=(("selection_output_dtype", "int32"),),
        )
        ops.extend((idx_wq, idx_wk, *store_ops, *pcp_idx_ops, idx_scores, idx_topk))
        index_store_id, topk_id = idx_store_id, idx_topk.op_id
    elif idx_owner >= 0:
        index_store_id = (index_cache_ids or {}).get(idx_owner)
    return eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id
