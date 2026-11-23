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

from mindie_sim.catalog.attention_common import _long_selected_pairs, local_kv_heads
from mindie_sim.catalog.cache import (
    completed_rows,
    index_row_bytes,
    index_selection_cost,
)
from mindie_sim.catalog.context_parallel import dcp_indexer_sharded
from mindie_sim.catalog.ops import cache_store_op, comm_op, gemm_op, vector_op
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .batch_prefix_indexer import apply_indexer_prefix_reads
from mindie_sim.prediction.operators.mla.generic_latent_attention import (
    index_key_window,
    owned_store_rows,
    sharded_index_visible_rows,
    visible_parts,
)
from mindie_sim.prediction.implementation_selection import is_caller_pinned_binding
from mindie_sim.prediction.structure_groups import declare_group

try:
    from mindie_sim.prediction.operators.sparse_indexer.index_cache_writer import emit_index_cache_write
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
    completed_rows: int | None = None,
    row_bytes: float,
    idx_dim: int,
    idx_kv_heads: int = 1,
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
            op_id=prefix + ".index_store",
            prefix=prefix,
            deps=deps,
            stage=stage,
            layer=layer,
            requests=requests,
            phase=phase,
            rows=rows,
            completed_rows=completed_rows,
            dim=idx_dim,
            index_dim=idx_dim,
            heads=idx_kv_heads,
            row_bytes=row_bytes,
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
    fallback_rows = rows if completed_rows is None else completed_rows
    fallback = cache_store_op(
        op_id=prefix + ".index_store",
        deps=deps,
        stage=stage,
        layer=layer,
        requests=requests,
        phase=phase,
        recipe="index_store",
        rows=fallback_rows,
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
    cand_topk = int(getattr(plan, "candidate_topk_blocks", 0) or 0)
    if _is_block_msa_indexer(plan) and cand_topk > 0:
        sel_topk = cand_topk
    else:
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
        tp = max(1, int(getattr(parallel, "tp", 1) or 1))
        raw_idx_kv_heads = max(
            1,
            int(getattr(plan, "index_kv_heads", 0)
                or getattr(plan, "indexer_kv_heads", 0) or 1),
        )
        local_idx_kv_heads = local_kv_heads(raw_idx_kv_heads, tp)
        eff_dcp = (
            max(1, int(getattr(parallel, "cp", 1) or 1))
            if dcp_indexer_sharded(model, runtime) else 1
        )
        (row_bytes, idx_heads, idx_dim, new_idx_rows, rows_by_view,
         shared_idx_by_view, scored_rows) = _index_view_work(
            model, plan, views, idx_ratio, valid, block_size,
            local_idx_kv_heads=local_idx_kv_heads, parallel=parallel, eff_dcp=eff_dcp,
            tp=tp)
        idx_wq, idx_wk, store_ops, pcp_idx_ops, idx_store_id, idx_ready = _index_cache_inputs(
            prefix, deps, stage, plan, requests, padded_tokens, idx_heads, idx_dim,
            model, quant, mode, valid, new_idx_rows, row_bytes, runtime, machine,
            pcp_prefill, views, idx_ratio, parallel, local_idx_kv_heads=local_idx_kv_heads)
        idx_scores, idx_topk = _build_index_score_and_topk_ops(
            prefix=prefix,
            plan=plan,
            views=views,
            stage=stage,
            requests=requests,
            quant=quant,
            mode=mode,
            valid=valid,
            idx_heads=idx_heads,
            idx_dim=idx_dim,
            row_bytes=row_bytes,
            rows_by_view=rows_by_view,
            shared_idx_by_view=shared_idx_by_view,
            scored_rows=scored_rows,
            idx_wq_id=idx_wq.op_id,
            idx_ready=idx_ready,
            idx_ratio=idx_ratio,
            sel_topk=sel_topk,
            user_pinned_idx=user_pinned_idx,
            runtime=runtime,
        )
        ops.extend((idx_wq, idx_wk, *store_ops, *pcp_idx_ops, idx_scores, idx_topk))
        _declare_gqa_indexer_group(
            ops, prefix=prefix, plan=plan, parallel=parallel, model=model,
            quant=quant, views=views, padded_tokens=padded_tokens,
            idx_heads=idx_heads, idx_dim=idx_dim, sel_topk=sel_topk,
            idx_ratio=idx_ratio, scores_id=idx_scores.op_id, topk_id=idx_topk.op_id,
        )
        index_store_id, topk_id = idx_store_id, idx_topk.op_id
    elif idx_owner >= 0:
        index_store_id = (index_cache_ids or {}).get(idx_owner)
    return eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id


def _build_index_score_and_topk_ops(
    *,
    prefix: str,
    plan,
    views,
    stage: int,
    requests: tuple[str, ...],
    quant,
    mode: str,
    valid: int,
    idx_heads: int,
    idx_dim: int,
    row_bytes: float,
    rows_by_view: list[int],
    shared_idx_by_view: list[int],
    scored_rows: int,
    idx_wq_id: str,
    idx_ready: str,
    idx_ratio: int,
    sel_topk: int,
    user_pinned_idx: bool,
    runtime,
) -> tuple[OpSpec, OpSpec]:
    """Construct the sparse indexer score sweep and TopK selection OpSpecs."""
    q_idx_bytes = valid * idx_heads * idx_dim * dtype_bytes(quant.index_dtype)
    unique_idx_rows = sum(rows_by_view)
    l2_bounded_index_read = float(min(scored_rows, unique_idx_rows)) * row_bytes
    shape = (scored_rows, idx_heads, idx_dim)
    score_bytes = float(scored_rows * FP32_SCORE_BYTES)
    score_meta: tuple[tuple[str, object], ...] = (
        ("score_materialization", "separate_fp32_score_then_topk"),
    )
    if _is_block_msa_indexer(plan):
        score_meta += (
            ("indexer_kind", "msa_block_score"),
            ("native_kind", "aclnnMsaIndexScore"),
        )
    elif _is_region_csa_indexer(plan):
        score_meta += (
            ("indexer_kind", "step5_region_score"),
            ("native_kind", "step5_indexer_region_score_arch35"),
        )
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
        deps=(idx_wq_id, idx_ready),
        recipe="index_scores",
        dtype=quant.index_dtype,
        metadata=score_meta,
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
    return idx_scores, idx_topk


def _is_block_msa_indexer(plan) -> bool:
    """True for block-maxima MSA indexers whose Q heads shard across TP."""
    return (
        int(getattr(plan, "candidate_block_size", 0) or 0) > 0
        and int(getattr(plan, "candidate_topk_blocks", 0) or 0) > 0
        and not bool(getattr(plan, "index_kpool_compress", False))
    )


def _is_region_csa_indexer(plan) -> bool:
    """True for compressed region-score CSA indexers without Q output gating."""
    kpool = int(getattr(plan, "index_kpool", 1) or 1)
    return (
        bool(getattr(plan, "index_kpool_compress", False))
        and kpool > 1
        and int(getattr(plan, "candidate_block_size", 0) or 0) == kpool
        and int(getattr(plan, "candidate_topk_blocks", 0) or 0) > 0
        and not bool(getattr(plan, "q_output_gate", False))
    )


def _declare_gqa_indexer_group(
    ops, *, prefix, plan, parallel, model, quant, views, padded_tokens,
    idx_heads, idx_dim, sel_topk, idx_ratio, scores_id, topk_id,
):
    """Unconditionally declare the GQA sparse indexer semantic subgraph."""
    quantized = quant.index_dtype not in {"bf16", "fp16"}
    if _is_region_csa_indexer(plan):
        role, family, native_kind = "pool_key_indexer", "pooled_indexer", "Step5RegionScore"
    elif _is_block_msa_indexer(plan):
        role, family, native_kind = "lightning_indexer", "dsa", "MsaIndexScore"
    elif int(plan.index_kpool or 1) > 1:
        role, family, native_kind = "pool_key_indexer", "pooled_indexer", "PoolKeyIndexer"
    elif quantized:
        role, family, native_kind = (
            "quant_lightning_indexer", "dsa", "VllmAscendQuantLightningIndexerV2")
    else:
        role, family, native_kind = "lightning_indexer", "dsa", "LightningIndexer"
    partition = tuple(
        {"query": v.q_len, "context": v.context, "kv_length": v.kv_len}
        for v in views
    )
    declare_group(
        ops, key=prefix + "." + role, role=role, family=family,
        members=(scores_id, topk_id), entries=(scores_id,), anchor=topk_id,
        block="attention",
        view={
            "tokens": sum(v.q_len for v in views), "physical_tokens": padded_tokens,
            "sequence_partition": partition, "compression_ratio": max(1, idx_ratio),
            "heads": idx_heads, "width": idx_dim, "topk": sel_topk,
            "query_dtype": quant.index_dtype,
            "weight_dtype": "fp16" if quant.index_dtype == "int8" else model.activation_dtype,
            "cache_dtype": quant.index_dtype,
            "selection_output_dtype": "int32",
            "scale_dtype": quant.index_scale_dtype,
        },
        implementation={
            "native_kind": native_kind,
            "conditions": {
                "cp": int(parallel.cp), "pcp": int(parallel.pcp),
                "candidate_filter": bool(plan.candidate_filter),
                "source_selection": "gqa_semantic_score_and_topk_projection",
                "timing_boundary": "separate_declared_operators",
            },
        },
        source={"kind": "semantic_builder",
                "module": "mindie_sim.prediction.operators.sparse_indexer.gqa_indexer"},
        fusion_status="unfused_decomposed",
        fusion_reason="gqa_indexer_separate_score_and_topk",
        native_fusion_eligible=False,
    )


def _index_view_work(
    model, plan, views, idx_ratio, valid, block_size, *,
    local_idx_kv_heads: int = 1, parallel=None, eff_dcp: int = 1, tp: int = 1,
):
    cost = index_selection_cost(
        model, plan, max((v.kv_len for v in views), default=0)
    )
    row_bytes_per_head = cost.index_row_bytes or index_row_bytes(model, plan.index_dim)
    row_bytes = row_bytes_per_head * max(1, int(local_idx_kv_heads))
    raw_idx_heads = max(1, int(plan.index_heads or 1))
    idx_heads = (
        local_kv_heads(raw_idx_heads, max(1, int(tp)))
        if _is_block_msa_indexer(plan)
        else raw_idx_heads
    )
    idx_dim = max(1, int(plan.index_dim or DEFAULT_INDEX_DIM))
    cand_cap = (
        plan.candidate_topk_blocks * plan.candidate_block_size
        if plan.candidate_filter
        else 0
    )
    new_rows_by_view = [
        sum(
            completed_rows(v.context + e_p, idx_ratio)
            - completed_rows(v.context + s_p, idx_ratio)
            for s_p, e_p in (v.query_ranges or ((0, v.q_len),))
            if e_p > s_p
        )
        if idx_ratio > 1
        else int(v.q_len)
        for v in views
    ]
    idx_block = max(1, block_size // max(1, idx_ratio))
    if eff_dcp > 1 and parallel is not None and any(new_rows_by_view):
        idx_starts = [completed_rows(v.context, idx_ratio) for v in views]
        new_idx_rows = int(owned_store_rows(
            new_rows_by_view, views, parallel, idx_block, contexts=idx_starts))
    else:
        new_idx_rows = sum(new_rows_by_view) if idx_ratio > 1 else valid
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
            cp=eff_dcp,
            interleave=block_size,
            compression=idx_ratio,
        )
        pairs_by_view.append(v_scored)
        rows_by_view.append(physical_rows)
        shared_idx_by_view.append(min(physical_rows, aligned_pref // eff_dcp))
    if eff_dcp > 1 and parallel is not None:
        scored_rows = int(sharded_index_visible_rows(views, parallel, idx_block, idx_ratio))
    else:
        scored_rows = sum(pairs_by_view)
    return row_bytes, idx_heads, idx_dim, new_idx_rows, rows_by_view, shared_idx_by_view, scored_rows


def _index_cache_inputs(
    prefix, deps, stage, plan, requests, padded_tokens, idx_heads, idx_dim, model, quant, mode,
    valid, new_idx_rows, row_bytes, runtime, machine, pcp_prefill, views, idx_ratio, parallel,
    *, local_idx_kv_heads: int = 1,
):
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
        n=max(1, int(local_idx_kv_heads)) * idx_dim,
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
        rows=valid,
        completed_rows=new_idx_rows,
        row_bytes=row_bytes,
        idx_dim=idx_dim,
        idx_kv_heads=max(1, int(local_idx_kv_heads)),
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
    return idx_wq, idx_wk, store_ops, pcp_idx_ops, idx_store_id, idx_ready
