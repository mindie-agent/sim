"""MLA latent-MQA/latent-MLA attention lowering.

[Role]
  Low-rank Q/KV producers, exact local-head widths, absorption, output gate,
  shared cache stores, grouped output and DSA consumer attention.

[Invariants]
  * q_a/kv_a are replicated; q_b/kv_b output heads are sharded once.
  * Kimi/GLM absorption reads the same kv_b weight bytes as a non-absorbed
    projection; v expansion is a real dependent GEMM after attention.
  * Grouped output keeps the group axis and falls back legally.
  * Output projection depends on all request outputs.

[Timeouts & Locks]
  Pure arithmetic; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from .a3_catalog_cache import (
    attention_rows, index_selection_cost, new_completed_rows,
)
from .a3_catalog_attention_common import (
    AttentionResult, _allreduce, _causal_pairs, _group_plan, _long_selected_pairs,
    _requests,
)
from .a3_catalog_attention_dsa import _dsa_attention_view, _index_producer, _long_cache_producer
from .a3_catalog_ops import (
    barrier_op, cache_store_op, comm_op, gemm_op, packed_weight_bytes, vector_op,
)
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec, _op
from .a3_types import A3InputError, ParallelSpec, dtype_bytes


def _mla_attention(
    *, prefix: str, plan: LayerPlan, views: Sequence[Any], stage: int, deps: str,
    model: ModelSpec, parallel: ParallelSpec, quant: QuantSpec, mode: str,
    padded_tokens: int, cache_ids: Mapping[int, str], topk_ids: Mapping[int, str],
    index_cache_ids: Mapping[int, str], candidate_ids: Mapping[int, str],
    stage_of: Any,
) -> AttentionResult:
    ops: list[OpSpec] = []
    h = model.hidden_size
    facts = dict(model.catalog_facts)
    native = str(facts.get("profile", ""))
    local_heads = max(1, plan.num_heads // parallel.tp)
    qk_dim = plan.qk_nope_head_dim + plan.qk_rope_head_dim
    cache_row_bytes = plan.cache_row_elements * dtype_bytes(quant.kv_dtype)
    seq = max((max(1, v.kv_len) for v in views), default=0)
    attention_outputs: list[str] = []
    if plan.mla_variant == "latent_mqa":
        q_a = gemm_op(
            op_id=f"{prefix}.q_a", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=plan.q_lora_rank,
            k=h, quant=quant, category="mla", phase=mode, family="mla_q_a",
            semantic_family="mla_q_a_replicated",
            metadata=(("sharding", "replicated"),),
        )
        q_norm = vector_op(
            op_id=f"{prefix}.q_norm", deps=(q_a.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.q_lora_rank),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_mla_q_norm",
            shape=(padded_tokens, plan.q_lora_rank),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.q_lora_rank),
                      ("norm_type", "rmsnorm"),
                      ("norm_eps", model.norm_eps),
                      ("source_width", plan.q_lora_rank)),
        )
        q_b = gemm_op(
            op_id=f"{prefix}.q_b", deps=(q_norm.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=local_heads * qk_dim, k=plan.q_lora_rank, quant=quant,
            category="mla", phase=mode, family="mla_q_b",
            semantic_family="mla_q_b_local_heads",
            semantic_shape=(padded_tokens, local_heads * qk_dim, plan.q_lora_rank),
            metadata=(("local_heads", local_heads), ("qk_head_dim", qk_dim),
                      ("sharding", "output_heads_tp")),
        )
        kv_a = gemm_op(
            op_id=f"{prefix}.kv_a", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=plan.kv_lora_rank,
            k=h, quant=quant, category="mla", phase=mode, family="mla_kv_a",
            semantic_family="mla_wkv_replicated",
            metadata=(("sharding", "replicated"),),
        )
        kv_norm = vector_op(
            op_id=f"{prefix}.kv_norm", deps=(kv_a.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.kv_lora_rank),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_mla_kv_norm",
            shape=(padded_tokens, plan.kv_lora_rank),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.kv_lora_rank),
                      ("norm_type", "rmsnorm"),
                      ("norm_eps", model.norm_eps),
                      ("source_width", plan.kv_lora_rank)),
        )
        rope = vector_op(
            op_id=f"{prefix}.kv_rope", deps=(kv_norm.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.qk_rope_head_dim),
            ops_per_element=3.0, dtype=model.activation_dtype, family="rope",
            phase=mode, recipe="catalog_mla_rope",
        )
        swa_store = cache_store_op(
            op_id=f"{prefix}.swa_store", deps=(rope.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), phase=mode,
            recipe="catalog_mla_swa_store", rows=padded_tokens,
            row_bytes=cache_row_bytes, dtype=quant.kv_dtype, kind="mla_swa",
        )
        ops.extend([q_a, q_norm, q_b, kv_a, kv_norm, rope, swa_store])
        q_ready = q_b.op_id
        long_store = None
        index_store = None
        long_cache_dep = None
        if plan.owns_long_kv and plan.compress_ratio > 0:
            new_rows = new_completed_rows(
                max(0, min((v.context for v in views), default=0)),
                max(1, sum(v.q_len for v in views)), plan.compress_ratio,
            ) if views else 0
            producer_ops, long_store, index_store = _long_cache_producer(
                prefix=prefix, plan=plan, stage=stage, deps=kv_norm.op_id,
                seq=seq, new_rows=new_rows, input_tokens=padded_tokens,
                model=model, parallel=parallel,
                quant=quant, mode=mode, requests=_requests(views),
            )
            ops.extend(producer_ops)
            long_cache_dep = long_store
            if plan.owns_index_k and index_store is None:
                index_ready = barrier_op(
                    op_id=f"{prefix}.index_cache_ready", deps=(long_store,),
                    stage=stage, layer=plan.index, requests=_requests(views),
                    phase=mode, recipe="catalog_index_cache_no_new_rows",
                    shape=(1, int(facts.get("index_head_dim", 128) or 128)),
                    metadata=(("new_rows", 0), ("cache_reused", True)),
                )
                ops.append(index_ready)
                index_store = index_ready
        elif plan.compress_ratio > 0:
            owner = int(plan.kv_owner_layer_id)
            owner_store = cache_ids.get(owner)
            if owner < 0 or owner_store is None:
                raise A3InputError(f"catalog_long_cache_dependency_missing:layer{plan.index}")
            if stage_of is not None and stage_of(owner) != stage:
                owner_stage = stage_of(owner)
                transfer = comm_op(
                    op_id=f"{prefix}.long_cache_transfer", deps=(owner_store,),
                    stage=stage, layer=plan.index, requests=_requests(views),
                    family="cache_transfer", phase=mode,
                    recipe="catalog_cross_stage_cache_transfer",
                    link_bytes=float(seq * cache_row_bytes),
                    shape=(max(1, seq), plan.cache_row_elements),
                    metadata=(("owner_layer", owner), ("owner_stage", owner_stage),
                              ("consumer_stage", stage),
                              ("storage_shared", "producer_transfer")),
                )
                ops.append(transfer)
                long_cache_dep = transfer.op_id
            else:
                long_cache_dep = owner_store
        kv_ready = long_cache_dep
        topk_id = None
        index_cache_id = None
        read_bytes = 0.0
        index_owner = int(plan.index_k_owner_layer_id)
        if index_store is not None:
            index_cache_id = index_store.op_id if hasattr(index_store, "op_id") else index_store
        elif index_owner >= 0:
            index_cache_id = cache_ids.get(("index", index_owner))
        candidate_id = None
        if plan.indexer or plan.owns_topk:
            candidate_dep = None
            if plan.candidate_filter and plan.candidate_source_layer_id >= 0:
                candidate_dep = candidate_ids.get(plan.candidate_source_layer_id)
                if candidate_dep is None:
                    raise A3InputError(
                        f"catalog_candidate_dependency_missing:layer{plan.index}"
                    )
            idx_ops, topk_id, read_bytes, candidate_id = _index_producer(
                prefix=prefix, plan=plan, stage=stage, deps=q_norm.op_id,
                seq=seq, model=model, parallel=parallel, quant=quant, mode=mode,
                requests=_requests(views), views=views, cache_dep=index_cache_id,
                candidate_dep=candidate_dep,
            )
            ops.extend(idx_ops)
        elif plan.topk_owner_layer_id >= 0:
            owner_topk = topk_ids.get(plan.topk_owner_layer_id)
            if owner_topk is None:
                raise A3InputError(f"catalog_topk_dependency_missing:layer{plan.index}")
            owner = plan.topk_owner_layer_id
            if stage_of is not None and stage_of(owner) != stage:
                transfer = comm_op(
                    op_id=f"{prefix}.topk_transfer", deps=(owner_topk,),
                    stage=stage, layer=plan.index, requests=_requests(views),
                    family="cache_transfer", phase=mode,
                    recipe="catalog_cross_stage_topk_transfer",
                    link_bytes=float(max(1, int(facts.get('index_topk', 0) or 1)) * 4),
                    shape=(max(1, int(facts.get("index_topk", 0) or 1)),),
                    metadata=(("owner_layer", owner), ("cache", "topk_indices")),
                )
                ops.append(transfer)
                topk_id = transfer.op_id
            else:
                topk_id = owner_topk
    else:  # latent_mla (Kimi / GLM)
        qk = plan.qk_nope_head_dim + plan.qk_rope_head_dim
        q_a = gemm_op(
            op_id=f"{prefix}.q_a", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=plan.q_lora_rank,
            k=h, quant=quant, category="mla", phase=mode, family="mla_q_a",
            semantic_family="mla_q_a_replicated",
            metadata=(("sharding", "replicated"),),
        )
        q_norm = vector_op(
            op_id=f"{prefix}.q_a_norm", deps=(q_a.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.q_lora_rank),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_mla_q_a_norm",
            shape=(padded_tokens, plan.q_lora_rank),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.q_lora_rank),
                      ("norm_type", "rmsnorm"),
                      ("norm_eps", model.norm_eps)),
        )
        q_b = gemm_op(
            op_id=f"{prefix}.q_b", deps=(q_norm.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=local_heads * qk, k=plan.q_lora_rank, quant=quant,
            category="mla", phase=mode, family="mla_q_b",
            semantic_family="mla_q_b_local_heads",
            metadata=(("local_heads", local_heads), ("qk_head_dim", qk),
                      ("sharding", "output_heads_tp")),
        )
        kv_a = gemm_op(
            op_id=f"{prefix}.kv_a", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens,
            n=plan.kv_lora_rank + plan.qk_rope_head_dim, k=h, quant=quant,
            category="mla", phase=mode, family="mla_kv_a",
            semantic_family="mla_kv_a_replicated",
            metadata=(("sharding", "replicated"),),
        )
        kv_norm = vector_op(
            op_id=f"{prefix}.kv_a_norm", deps=(kv_a.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.kv_lora_rank),
            ops_per_element=5.0, dtype=model.activation_dtype, family="rmsnorm",
            phase=mode, recipe="catalog_mla_kv_a_norm",
            shape=(padded_tokens, plan.kv_lora_rank),
            metadata=(("tokens", padded_tokens),
                      ("reduction_width", plan.kv_lora_rank),
                      ("norm_type", "rmsnorm"),
                      ("norm_eps", model.norm_eps)),
        )
        rope = vector_op(
            op_id=f"{prefix}.kv_rope", deps=(kv_norm.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * plan.qk_rope_head_dim),
            ops_per_element=3.0, dtype=model.activation_dtype, family="rope",
            phase=mode, recipe="catalog_mla_rope",
        )
        swa_store = cache_store_op(
            op_id=f"{prefix}.mla_store", deps=(rope.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), phase=mode,
            recipe="catalog_mla_store", rows=padded_tokens,
            row_bytes=cache_row_bytes, dtype=quant.kv_dtype, kind="mla_kv",
        )
        # Absorption: q_nope * W_kv_b^T is a real GEMM; values are expanded
        # after attention.  Together they read the same kv_b weight bytes as a
        # non-absorbed projection, not a duplicate parameter.
        q_absorb = gemm_op(
            op_id=f"{prefix}.q_absorb", deps=(q_b.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=local_heads * plan.kv_lora_rank, k=plan.qk_nope_head_dim,
            quant=quant, category="mla", phase=mode, family="mla_q_absorb",
            semantic_family="mla_q_absorb_kv_b_nope",
            weight_bytes_override=packed_weight_bytes(
                quant, local_heads * plan.kv_lora_rank,
                plan.qk_nope_head_dim, "mla",
            ),
            metadata=(("absorption", "q_nope_absorbed"),),
        )
        ops.extend([q_a, q_norm, q_b, kv_a, kv_norm, rope, swa_store, q_absorb])
        q_ready = q_absorb.op_id
        kv_ready = swa_store.op_id
        topk_id = None
        read_bytes = 0.0
        long_store = None
        index_store = None
        index_dep = None
        candidate_id = None
        if native == "glm":
            # Full indexers own a token-level BF16 index plane.  Shared layers
            # read the nearest full producer's plane; both execute selection.
            index_dim = int(facts.get("index_head_dim", 128) or 128)
            if plan.indexer:
                wk = gemm_op(
                    op_id=f"{prefix}.index_wk", deps=(deps,), stage=stage,
                    layer=plan.index, requests=_requests(views), m=padded_tokens,
                    n=index_dim, k=h, quant=quant, category="indexer_fp8",
                    phase=mode, family="gemm",
                    semantic_family="glm_index_k_projection",
                )
                k_norm = vector_op(
                    op_id=f"{prefix}.index_k_norm", deps=(wk.op_id,), stage=stage,
                    layer=plan.index, requests=_requests(views),
                    elements=max(1, padded_tokens * index_dim), ops_per_element=5.0,
                    dtype=model.activation_dtype, family="rmsnorm", phase=mode,
                    recipe="catalog_glm_index_k_norm",
                    shape=(padded_tokens, index_dim),
                    metadata=(("tokens", padded_tokens),
                              ("reduction_width", index_dim),
                              ("norm_type", "rmsnorm"),
                              ("norm_eps", model.norm_eps)),
                )
                index_store = cache_store_op(
                    op_id=f"{prefix}.index_k_store", deps=(k_norm.op_id,),
                    stage=stage, layer=plan.index, requests=_requests(views),
                    phase=mode, recipe="catalog_glm_index_k_store",
                    rows=padded_tokens, row_bytes=index_dim * 2,
                    dtype=quant.kv_dtype, kind="index_k",
                )
                ops.extend([wk, k_norm, index_store])
                index_dep = index_store.op_id
            else:
                owner = int(plan.index_source_layer_id)
                owner_store = index_cache_ids.get(owner)
                if owner < 0 or owner_store is None:
                    raise A3InputError(f"catalog_glm_index_dependency_missing:layer{plan.index}")
                if stage_of is not None and stage_of(owner) != stage:
                    transfer = comm_op(
                        op_id=f"{prefix}.index_transfer", deps=(owner_store,),
                        stage=stage, layer=plan.index, requests=_requests(views),
                        family="cache_transfer", phase=mode,
                        recipe="catalog_cross_stage_index_transfer",
                        link_bytes=float(seq * index_dim * 2),
                        shape=(max(1, seq), index_dim),
                        metadata=(("owner_layer", owner), ("cache", "index_k"),
                                  ("index_source_status", plan.index_source_status)),
                    )
                    ops.append(transfer)
                    index_dep = transfer.op_id
                else:
                    index_dep = owner_store
            if plan.indexer or plan.indexer_type == "shared":
                idx_ops, topk_id, read_bytes, candidate_id = _index_producer(
                    prefix=prefix, plan=plan, stage=stage, deps=q_b.op_id,
                    seq=seq, model=model, parallel=parallel, quant=quant,
                    mode=mode, requests=_requests(views), views=views,
                    cache_dep=index_dep,
                )
                ops.extend(idx_ops)
    # ---- per-view attention over ordinary MLA/compressed cache ----
    attn_ids: list[str] = []
    selected, swa, dense_rows = attention_rows(model, plan, seq)
    for view in views:
        if plan.mla_variant == "latent_mla":
            use_dim = plan.kv_lora_rank + plan.qk_rope_head_dim
            v_dim = plan.kv_lora_rank
        else:
            use_dim = plan.head_dim
            v_dim = plan.head_dim
        if native in {"deepseek_v4", "deepseek_v41"}:
            view_ops, end = _dsa_attention_view(
                prefix=f"{prefix}.attn.{view.request_id}", view=view, plan=plan,
                stage=stage,
                deps=tuple(d for d in (q_ready, kv_ready, swa_store.op_id)
                           if d is not None)
                + ((topk_id,) if topk_id else ()),
                local_heads=local_heads, qk_dim=use_dim if native else qk_dim,
                v_dim=use_dim if plan.mla_variant == "latent_mqa" else v_dim,
                selected=selected, swa=swa, cache_row_bytes=cache_row_bytes,
                model=model,
            )
        else:
            pairs = _causal_pairs(view.q_len, view.context, 0)
            q_bytes = view.q_len * local_heads * use_dim * dtype_bytes(model.activation_dtype)
            kv_read = pairs * cache_row_bytes
            scores = max(1, int(view.q_len * local_heads * view.kv_len))
            qk = _op(
                op_id=f"{prefix}.attn.{view.request_id}.qk", family="attention_qk",
                phase=view.execution_mode, stage=stage, layer=plan.index,
                requests=(view.request_id,),
                semantic_shape=(view.q_len, view.kv_len, local_heads, use_dim),
                local_shape=(view.q_len, view.kv_len, local_heads, use_dim),
                padded_shape=(view.q_len, view.kv_len, local_heads, use_dim),
                launch_shape=(view.q_len, view.kv_len, local_heads, use_dim),
                flops=2.0 * local_heads * pairs * use_dim,
                hbm_bytes=q_bytes + kv_read + scores * 2.0,
                mte2_bytes=q_bytes + kv_read, mte3_bytes=scores * 2.0,
                deps=(q_ready, kv_ready), recipe="catalog_mla_qk",
                dtype=model.activation_dtype,
                metadata=(("absorption", plan.mla_variant == "latent_mla"),
                          ("q_len", view.q_len), ("kv_len", view.kv_len)),
            )
            soft = vector_op(
                op_id=f"{prefix}.attn.{view.request_id}.softmax", deps=(qk.op_id,),
                stage=stage, layer=plan.index, requests=(view.request_id,),
                elements=scores, ops_per_element=5.0, dtype="fp32",
                family="attention_softmax", phase=view.execution_mode,
                recipe="catalog_mla_softmax",
            )
            pv = _op(
                op_id=f"{prefix}.attn.{view.request_id}.pv", family="attention_pv",
                phase=view.execution_mode, stage=stage, layer=plan.index,
                requests=(view.request_id,),
                semantic_shape=(view.q_len, view.kv_len, local_heads, v_dim),
                local_shape=(view.q_len, view.kv_len, local_heads, v_dim),
                padded_shape=(view.q_len, view.kv_len, local_heads, v_dim),
                launch_shape=(view.q_len, view.kv_len, local_heads, v_dim),
                flops=2.0 * local_heads * pairs * v_dim,
                hbm_bytes=kv_read + scores * 2.0 + view.q_len * local_heads * v_dim * 2,
                mte2_bytes=kv_read + scores * 2.0,
                mte3_bytes=view.q_len * local_heads * v_dim * 2,
                deps=(soft.op_id,), recipe="catalog_mla_pv",
                dtype=model.activation_dtype,
            )
            view_ops = [qk, soft, pv]
            end = pv.op_id
        ops.extend(view_ops)
        attn_ids.append(end)
    if plan.mla_variant == "latent_mla":
        v_expand = gemm_op(
            op_id=f"{prefix}.kv_b", deps=tuple(attn_ids), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=local_heads * plan.v_head_dim, k=plan.kv_lora_rank,
            quant=quant, category="mla", phase=mode, family="mla_kv_b",
            semantic_family="mla_kv_b_value_expand_after_absorption",
            weight_bytes_override=packed_weight_bytes(
                quant, local_heads * plan.v_head_dim, plan.kv_lora_rank, "mla",
            ),
            semantic_shape=(padded_tokens, local_heads * plan.v_head_dim, plan.kv_lora_rank),
            metadata=(("absorption", "value_expand_after_attention"),
                      ("depends_all_requests", True)),
        )
        ops.append(v_expand)
        v_width = local_heads * plan.v_head_dim
        attn_ids = [v_expand.op_id]
        if plan.mla_output_gate:
            out_gate = gemm_op(
                op_id=f"{prefix}.mla_out_gate", deps=(deps,), stage=stage,
                layer=plan.index, requests=_requests(views), m=padded_tokens,
                n=v_width, k=h, quant=quant, category="mla", phase=mode,
                family="gemm", semantic_family="mla_output_gate_projection",
                metadata=(("gate", "sigmoid_or_swish"),),
            )
            out_gate_act = vector_op(
                op_id=f"{prefix}.mla_out_gate_act", deps=(out_gate.op_id,),
                stage=stage, layer=plan.index, requests=_requests(views),
                elements=max(1, padded_tokens * v_width), ops_per_element=4.0,
                dtype=model.activation_dtype, family="linear_output_gate",
                phase=mode, recipe="catalog_mla_out_gate_act",
                semantic_family="mla_output_gate_activation",
            )
            out_gate_mul = vector_op(
                op_id=f"{prefix}.mla_out_gated",
                deps=(v_expand.op_id, out_gate_act.op_id), stage=stage,
                layer=plan.index, requests=_requests(views),
                elements=max(1, padded_tokens * v_width), ops_per_element=2.0,
                dtype=model.activation_dtype, family="linear_output_gate",
                phase=mode, recipe="catalog_mla_out_gate_mul",
                semantic_family="mla_output_gate_multiply",
            )
            ops.extend([out_gate, out_gate_act, out_gate_mul])
            attn_ids = [out_gate_mul.op_id]
    else:
        v_width = local_heads * plan.head_dim
    # ---- grouped or ordinary output projection ----
    group_parallel, local_groups = _group_plan(plan, parallel)
    deps_out = tuple(attn_ids)
    if plan.o_groups > 0:
        groups = plan.o_groups
        group_width = plan.num_heads * (plan.v_head_dim if plan.mla_variant == "latent_mla" else plan.head_dim) // groups
        if group_parallel:
            wo_a = gemm_op(
                op_id=f"{prefix}.wo_a", deps=deps_out, stage=stage,
                layer=plan.index, requests=_requests(views),
                m=padded_tokens * local_groups, n=plan.o_lora_rank,
                k=group_width, quant=quant, category="mla", phase=mode,
                family="gemm", semantic_family="grouped_output_wo_a",
                weight_bytes_override=packed_weight_bytes(
                    quant, local_groups * plan.o_lora_rank, group_width, "mla",
                ),
                metadata=(("o_groups", groups), ("local_groups", local_groups),
                          ("group_parallel", True)),
            )
            wo_b = gemm_op(
                op_id=f"{prefix}.wo_b", deps=(wo_a.op_id,), stage=stage,
                layer=plan.index, requests=_requests(views), m=padded_tokens,
                n=h, k=local_groups * plan.o_lora_rank, quant=quant,
                category="mla", phase=mode, family="gemm",
                semantic_family="grouped_output_wo_b",
                weight_bytes_override=packed_weight_bytes(
                    quant, h, local_groups * plan.o_lora_rank, "mla",
                ),
                metadata=(("o_groups", groups), ("local_groups", local_groups)),
            )
            ops.extend([wo_a, wo_b])
            last = wo_b.op_id
            if parallel.tp > 1:
                ar = _allreduce(prefix=prefix, deps=(last,), stage=stage,
                                layer=plan.index, requests=_requests(views),
                                model=model, parallel=parallel, mode=mode,
                                tokens=padded_tokens, suffix="grouped_out")
                ops.append(ar)
                last = ar.op_id
        else:
            # Legal fallback: gather full head outputs, replicate the grouped
            # projection; the group axis is never silently dropped.
            gather = comm_op(
                op_id=f"{prefix}.heads_allgather", deps=deps_out, stage=stage,
                layer=plan.index, requests=_requests(views), family="allgather",
                phase=mode, recipe="catalog_group_fallback_allgather",
                link_bytes=(parallel.tp - 1) / parallel.tp * padded_tokens
                * plan.num_heads * (plan.v_head_dim if plan.mla_variant == "latent_mla" else plan.head_dim)
                * dtype_bytes(model.activation_dtype),
                shape=(padded_tokens, plan.num_heads,
                       plan.v_head_dim if plan.mla_variant == "latent_mla" else plan.head_dim),
                metadata=(("group_parallel", False),),
            )
            wo_a = gemm_op(
                op_id=f"{prefix}.wo_a", deps=(gather.op_id,), stage=stage,
                layer=plan.index, requests=_requests(views),
                m=padded_tokens * groups, n=plan.o_lora_rank, k=group_width,
                quant=quant, category="mla", phase=mode, family="gemm",
                semantic_family="grouped_output_wo_a_fallback",
                weight_bytes_override=packed_weight_bytes(
                    quant, groups * plan.o_lora_rank, group_width, "mla",
                ),
            )
            wo_b = gemm_op(
                op_id=f"{prefix}.wo_b", deps=(wo_a.op_id,), stage=stage,
                layer=plan.index, requests=_requests(views), m=padded_tokens,
                n=h, k=groups * plan.o_lora_rank, quant=quant, category="mla",
                phase=mode, family="gemm",
                semantic_family="grouped_output_wo_b_fallback",
                weight_bytes_override=packed_weight_bytes(
                    quant, h, groups * plan.o_lora_rank, "mla",
                ),
            )
            ops.extend([gather, wo_a, wo_b])
            last = wo_b.op_id
    else:
        o_proj = gemm_op(
            op_id=f"{prefix}.o_proj", deps=deps_out, stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=h, k=v_width, quant=quant, category="mla", phase=mode,
            family="o_projection", semantic_family="mla_o_proj",
            metadata=(("depends_all_requests", True),),
        )
        ops.append(o_proj)
        last = o_proj.op_id
        if parallel.tp > 1:
            ar = _allreduce(prefix=prefix, deps=(last,), stage=stage,
                            layer=plan.index, requests=_requests(views),
                            model=model, parallel=parallel, mode=mode,
                            tokens=padded_tokens, suffix="mla_out")
            ops.append(ar)
            last = ar.op_id
    cache_store = long_store if plan.mla_variant == "latent_mqa" else swa_store
    return AttentionResult(
        ops=ops, last=last, output_ids=(last,),
        cache_store_id=(
            cache_store.op_id if hasattr(cache_store, "op_id") else cache_store
        ) if cache_store is not None else None,
        topk_id=topk_id,
        selected_rows=selected, index_read_bytes=read_bytes,
        cache_write_rows=seq,
        attention_outputs=tuple(attn_ids),
        index_store_id=(
            index_store.op_id if hasattr(index_store, "op_id") else index_store
        ) if index_store is not None else None,
        candidate_id=candidate_id,
    )
