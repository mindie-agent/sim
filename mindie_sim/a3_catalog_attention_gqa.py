"""GQA/gated-GQA attention lowering.

[Role]
  Exact QKV/gate widths, QK/PV over request views, output gate and projection.

[Invariants]
  * K/V heads replicate rather than shard fractionally when TP>KV heads.
  * Output projection depends on all batch request attention outputs.
  * Q and KV producers are both dependencies.

[Timeouts & Locks]
  Pure arithmetic; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .a3_catalog_attention_common import (
    AttentionResult, _allreduce, _causal_pairs, _qkv_widths, _requests,
)
from .a3_catalog_ops import cache_store_op, gemm_op, packed_weight_bytes, vector_op
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec, _op
from .a3_types import ParallelSpec, dtype_bytes


def _gqa_attention(
    *, prefix: str, plan: LayerPlan, views: Sequence[Any], stage: int, deps: str,
    model: ModelSpec, parallel: ParallelSpec, quant: QuantSpec, mode: str,
    padded_tokens: int,
) -> AttentionResult:
    ops: list[OpSpec] = []
    h = model.hidden_size
    local_heads, local_kv, q_width, kv_width = _qkv_widths(plan, parallel)
    n = q_width + 2 * kv_width
    full_q = (2 if plan.q_output_gate else 1) * plan.num_heads * plan.head_dim
    full_kv = plan.num_kv_heads * plan.head_dim
    weight_bytes = (
        packed_weight_bytes(quant, full_q // parallel.tp, h, "attn")
        + 2 * packed_weight_bytes(quant, max(1, math.ceil(full_kv / parallel.tp)), h, "attn")
    )
    qkv = gemm_op(
        op_id=f"{prefix}.qkv", deps=(deps,), stage=stage, layer=plan.index,
        requests=_requests(views), m=padded_tokens, n=n, k=h, quant=quant,
        category="attn", phase=mode, family="qkv_projection",
        semantic_family="gated_gqa_qkv" if plan.q_output_gate else "gqa_qkv",
        semantic_shape=(padded_tokens, n, h), weight_bytes_override=weight_bytes,
        metadata=(("local_heads", local_heads), ("local_kv_heads", local_kv),
                  ("q_width", q_width), ("kv_width", kv_width),
                  ("output_gate", plan.q_output_gate)),
    )
    ops.append(qkv)
    q_ready = qkv.op_id
    gate_id = None
    if plan.q_output_gate:
        gate = vector_op(
            op_id=f"{prefix}.q_gate", deps=(qkv.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * 2 * local_heads * plan.head_dim),
            ops_per_element=4.0, dtype=model.activation_dtype,
            family="qk_gate", phase=mode, recipe="catalog_gqa_q_gate",
            semantic_family="q_output_sigmoid_gate",
            metadata=(("gate", "sigmoid_swish"),),
        )
        ops.append(gate)
        gate_id = gate.op_id
    kv_bytes_per_token = 2 * local_kv * plan.head_dim * dtype_bytes(quant.kv_dtype)
    attn_ids: list[str] = []
    for view in views:
        pairs = (
            max(1, view.q_len) * max(0, view.context)
            + max(1, view.q_len) ** 2
            if plan.noncausal_attention
            else _causal_pairs(view.q_len, view.context, 0)
        )
        qk_dim = plan.head_dim
        v_dim = plan.head_dim
        scores = max(1, int(view.q_len * local_heads * (view.context + view.q_len)))
        qk_bytes = (
            view.q_len * local_heads * qk_dim * dtype_bytes(model.activation_dtype)
            + pairs * kv_bytes_per_token
            + scores * 2.0
        )
        qk = _op(
            op_id=f"{prefix}.attn.{view.request_id}.qk", family="attention_qk",
            phase=view.execution_mode, stage=stage, layer=plan.index,
            requests=(view.request_id,),
            semantic_shape=(view.q_len, view.context, local_heads, qk_dim),
            local_shape=(view.q_len, max(1, view.kv_len), local_heads, qk_dim),
            padded_shape=(max(1, view.q_len), max(1, view.kv_len), local_heads, qk_dim),
            launch_shape=(max(1, view.q_len), max(1, view.kv_len), local_heads, qk_dim),
            flops=2.0 * local_heads * pairs * qk_dim,
            hbm_bytes=qk_bytes, mte2_bytes=qk_bytes, mte3_bytes=scores * 2.0,
            deps=(q_ready,), recipe="catalog_gqa_qk", dtype=model.activation_dtype,
            metadata=(("layer_type", "gated_gqa" if plan.q_output_gate else "gqa"),
                      ("q_len", view.q_len), ("kv_len", view.kv_len)),
        )
        soft = vector_op(
            op_id=f"{prefix}.attn.{view.request_id}.softmax", deps=(qk.op_id,),
            stage=stage, layer=plan.index, requests=(view.request_id,),
            elements=max(1, view.q_len * local_heads * (view.context + view.q_len)),
            ops_per_element=5.0, dtype="fp32", family="attention_softmax",
            phase=view.execution_mode, recipe="catalog_gqa_softmax",
            shape=(view.q_len, local_heads, max(1, view.kv_len)),
        )
        pv = _op(
            op_id=f"{prefix}.attn.{view.request_id}.pv", family="attention_pv",
            phase=view.execution_mode, stage=stage, layer=plan.index,
            requests=(view.request_id,),
            semantic_shape=(view.q_len, view.context, local_heads, v_dim),
            local_shape=(view.q_len, max(1, view.kv_len), local_heads, v_dim),
            padded_shape=(max(1, view.q_len), max(1, view.kv_len), local_heads, v_dim),
            launch_shape=(max(1, view.q_len), max(1, view.kv_len), local_heads, v_dim),
            flops=2.0 * local_heads * pairs * v_dim, hbm_bytes=qk_bytes,
            mte2_bytes=qk_bytes, mte3_bytes=view.q_len * local_heads * v_dim
            * dtype_bytes(model.activation_dtype),
            deps=(soft.op_id,), recipe="catalog_gqa_pv", dtype=model.activation_dtype,
            metadata=(("layer_type", "gqa"),),
        )
        ops.extend([qk, soft, pv])
        attn_ids.append(pv.op_id)
    o_deps = tuple(attn_ids)
    if gate_id is not None:
        gate_out = vector_op(
            op_id=f"{prefix}.attn.gated", deps=o_deps + (gate_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * local_heads * plan.head_dim),
            ops_per_element=3.0, dtype=model.activation_dtype,
            family="attention_output_gate", phase=mode, recipe="catalog_gqa_out_gate",
            semantic_family="attention_output_sigmoid_gate",
        )
        ops.append(gate_out)
        o_deps = (gate_out.op_id,)
    o_proj = gemm_op(
        op_id=f"{prefix}.o_proj", deps=o_deps, stage=stage, layer=plan.index,
        requests=_requests(views), m=padded_tokens, n=h,
        k=local_heads * plan.head_dim, quant=quant, category="attn",
        phase=mode, family="o_projection", semantic_family="gqa_o_proj",
        semantic_shape=(padded_tokens, h, local_heads * plan.head_dim),
        metadata=(("depends_all_requests", True),),
    )
    ops.append(o_proj)
    last = o_proj.op_id
    if parallel.tp > 1:
        ar = _allreduce(prefix=prefix, deps=(last,), stage=stage, layer=plan.index,
                        requests=_requests(views), model=model, parallel=parallel,
                        mode=mode, tokens=padded_tokens, suffix="attn")
        ops.append(ar)
        last = ar.op_id
    swa_store = None
    if plan.window:
        swa_store = cache_store_op(
            op_id=f"{prefix}.swa_store", deps=(qkv.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), phase=mode,
            recipe="catalog_swa_store", rows=padded_tokens,
            row_bytes=2 * local_kv * plan.head_dim * 2, dtype=quant.kv_dtype,
            kind="gqa_swa",
        )
        ops.append(swa_store)
    return AttentionResult(
        ops=ops, last=last, output_ids=(last,),
        cache_store_id=swa_store.op_id if swa_store else None,
        attention_outputs=tuple(last for _ in views),
        cache_write_rows=padded_tokens,
    )
