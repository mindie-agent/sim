"""KDA/GDN linear-attention lowering.

[Role]
  Projection widths, short convolution, gated delta recurrence, per-request
  endpoint state, output gate and output projection.

[Invariants]
  * Recurrent state is heads*Dk*Dv FP32 per request; convolution history is a
    separate bounded BF16 ring.
  * Two independent requests pay two state read/write sets; two tokens of one
    request can reuse one state boundary.
  * GDN key and value heads are independent.

[Timeouts & Locks]
  Pure arithmetic; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

from .a3_catalog_attention_common import AttentionResult, _allreduce, _requests
from .a3_catalog_ops import cache_store_op, gemm_op, packed_weight_bytes, vector_op
from .a3_catalog_types import QuantSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_ops import OpSpec
from .a3_types import A3InputError, ParallelSpec


def _linear_attention(
    *, prefix: str, plan: LayerPlan, views: Sequence[Any], stage: int, deps: str,
    model: ModelSpec, parallel: ParallelSpec, quant: QuantSpec, mode: str,
    padded_tokens: int,
) -> AttentionResult:
    ops: list[OpSpec] = []
    h = model.hidden_size
    tp = max(1, parallel.tp)
    attn = plan.attention
    if attn == "kda":
        d = plan.linear_heads * plan.linear_head_dim
        k_dim = plan.linear_key_heads * plan.linear_key_head_dim
        local_heads = max(1, plan.linear_heads // tp)
        local_k = max(1, plan.linear_key_heads // tp) * plan.linear_key_head_dim
        q = gemm_op(
            op_id=f"{prefix}.lin_q", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=d // tp, k=h,
            quant=quant, category="kda", phase=mode, family="gemm",
            semantic_family="kda_q_projection",
        )
        k = gemm_op(
            op_id=f"{prefix}.lin_k", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=k_dim // tp, k=h,
            quant=quant, category="kda", phase=mode, family="gemm",
            semantic_family="kda_k_projection",
        )
        v = gemm_op(
            op_id=f"{prefix}.lin_v", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=d // tp, k=h,
            quant=quant, category="kda", phase=mode, family="gemm",
            semantic_family="kda_v_projection",
        )
        g = gemm_op(
            op_id=f"{prefix}.lin_g", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=d // tp, k=h,
            quant=quant, category="kda", phase=mode, family="gemm",
            semantic_family="kda_full_rank_gate",
        )
        b = gemm_op(
            op_id=f"{prefix}.lin_b", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=local_heads, k=h,
            quant=quant, category="kda", phase=mode, family="gemm",
            semantic_family="kda_beta_projection",
        )
        f_a = gemm_op(
            op_id=f"{prefix}.lin_f_a", deps=(deps,), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=plan.linear_head_dim, k=h, quant=quant, category="kda",
            phase=mode, family="gemm", semantic_family="kda_f_a_replicated",
        )
        f_b = gemm_op(
            op_id=f"{prefix}.lin_f_b", deps=(f_a.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), m=padded_tokens,
            n=d // tp, k=plan.linear_head_dim, quant=quant, category="kda",
            phase=mode, family="gemm", semantic_family="kda_f_b",
        )
        conv = vector_op(
            op_id=f"{prefix}.conv", deps=(q.op_id, k.op_id, v.op_id), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * 3 * (d // tp)),
            ops_per_element=2.0 * max(1, plan.conv_kernel),
            dtype=model.activation_dtype, family="linear_conv", phase=mode,
            recipe="catalog_kda_conv", semantic_family="kda_short_conv",
            metadata=(("conv_width_per_rank", 3 * (d // tp)),
                      ("kernel", plan.conv_kernel)),
        )
        beta = vector_op(
            op_id=f"{prefix}.beta_gate", deps=(conv.op_id, b.op_id, f_b.op_id,
                                               g.op_id), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * local_heads * plan.linear_head_dim),
            ops_per_element=8.0, dtype="fp32", family="linear_gate", phase=mode,
            recipe="catalog_kda_beta_gate",
            semantic_family="kda_decay_beta_gate",
        )
        state_elements = local_heads * plan.linear_key_head_dim * plan.linear_head_dim
        state = vector_op(
            op_id=f"{prefix}.state", deps=(beta.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * state_elements),
            ops_per_element=4.0, dtype="fp32", family="recurrent_state",
            phase=mode, recipe="catalog_kda_recurrent",
            semantic_family="kda_recurrent_delta_rule",
            hbm_bytes_override=2.0 * state_elements * 4.0 * max(1, len(views)),
            shape=(local_heads, plan.linear_key_head_dim, plan.linear_head_dim),
            metadata=(("state_read_write_once_per_iteration", True),
                      ("state_elements_per_rank", state_elements),
                      ("state_dtype", "fp32")),
        )
        conv_state = cache_store_op(
            op_id=f"{prefix}.conv_state", deps=(conv.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), phase=mode,
            recipe="catalog_kda_conv_state",
            rows=max(1, plan.conv_kernel - 1) * max(1, len(views)),
            row_bytes=3 * (d // tp) * 2, dtype="bf16", kind="kda_conv_ring",
        )
        out_gate = vector_op(
            op_id=f"{prefix}.out_gate", deps=(state.op_id, g.op_id), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * (d // tp)), ops_per_element=4.0,
            dtype=model.activation_dtype, family="linear_output_gate",
            phase=mode, recipe="catalog_kda_out_gate",
            semantic_family="kda_output_gate_norm",
        )
        o_proj = gemm_op(
            op_id=f"{prefix}.lin_o", deps=(out_gate.op_id, conv_state.op_id),
            stage=stage, layer=plan.index, requests=_requests(views),
            m=padded_tokens, n=h, k=d // tp, quant=quant, category="kda",
            phase=mode, family="linear_o_projection",
            semantic_family="kda_o_projection",
        )
        ops.extend([q, k, v, g, b, f_a, f_b, conv, beta, state, conv_state,
                    out_gate, o_proj])
        last = o_proj.op_id
    else:  # GDN
        key_w = plan.linear_key_heads * plan.linear_key_head_dim
        value_w = plan.linear_value_heads * plan.linear_head_dim
        local_value_heads = max(1, plan.linear_value_heads // tp)
        local_key_heads = max(1, plan.linear_key_heads // tp)
        local_key_dim = plan.linear_key_head_dim
        local_key = local_key_heads * local_key_dim
        local_value = local_value_heads * plan.linear_head_dim
        qkvz_n = 2 * local_key + 2 * local_value
        ba_n = 2 * local_value_heads
        qkvz_weight = packed_weight_bytes(quant, qkvz_n, h, "gdn")
        ba_weight = packed_weight_bytes(quant, ba_n, h, "gdn")
        qkvz = gemm_op(
            op_id=f"{prefix}.qkvz", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=qkvz_n, k=h,
            quant=quant, category="gdn", phase=mode, family="gemm",
            semantic_family="gdn_packed_qkvz",
            weight_bytes_override=qkvz_weight,
            metadata=(("packed_group", "qkvz"), ("local_key_width", 2 * local_key),
                      ("local_value_width", 2 * local_value)),
        )
        ba = gemm_op(
            op_id=f"{prefix}.ba", deps=(deps,), stage=stage, layer=plan.index,
            requests=_requests(views), m=padded_tokens, n=ba_n, k=h,
            quant=quant, category="gdn", phase=mode, family="gemm",
            semantic_family="gdn_packed_ba",
            weight_bytes_override=ba_weight,
            metadata=(("packed_group", "ba"),),
        )
        split = vector_op(
            op_id=f"{prefix}.split_norm", deps=(qkvz.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * qkvz_n), ops_per_element=6.0,
            dtype=model.activation_dtype, family="qk_norm",
            phase=mode, recipe="catalog_gdn_split_norm",
            semantic_family="gdn_split_qk_l2norm_mrope",
            shape=(padded_tokens, qkvz_n),
            metadata=(("tokens", padded_tokens),
                      ("packed_qkvz_width", qkvz_n),
                      ("composite", True),
                      ("subprimitives", ("split", "qk_l2norm", "mrope")),
                      ("norm_type", "l2"),
                      ("not_rmsnorm", True),
                      ("l2norm_heads", local_key_heads),
                      ("l2norm_head_dim", local_key_dim),
                      ("rotary_dim", int(round(local_key_dim * 0.25))),
                      ("partial_rotary_factor", 0.25),
                      ("mrope_sections", (local_key_dim // 4,) * 3
                       + (local_key_dim - 3 * (local_key_dim // 4),))),
        )
        conv = vector_op(
            op_id=f"{prefix}.conv", deps=(split.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * (2 * local_key + local_value)),
            ops_per_element=2.0 * max(1, plan.conv_kernel),
            dtype=model.activation_dtype, family="linear_conv", phase=mode,
            recipe="catalog_gdn_conv", semantic_family="gdn_depthwise_conv",
        )
        gate = vector_op(
            op_id=f"{prefix}.gate", deps=(conv.op_id, ba.op_id), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * local_value_heads),
            ops_per_element=8.0, dtype="fp32", family="linear_gate",
            phase=mode, recipe="catalog_gdn_gate",
            semantic_family="gdn_decay_beta_gate",
        )
        state_elements = local_value_heads * plan.linear_key_head_dim * plan.linear_head_dim
        state = vector_op(
            op_id=f"{prefix}.state", deps=(gate.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * state_elements),
            ops_per_element=5.0, dtype="fp32", family="recurrent_state",
            phase=mode, recipe="catalog_gdn_recurrent",
            semantic_family="gdn_chunked_delta_rule",
            hbm_bytes_override=2.0 * state_elements * 4.0 * max(1, len(views)),
            shape=(local_value_heads, plan.linear_key_head_dim, plan.linear_head_dim),
            metadata=(("key_heads", plan.linear_key_heads),
                      ("value_heads", plan.linear_value_heads),
                      ("local_value_heads", local_value_heads),
                      ("state_elements_per_rank", state_elements)),
        )
        conv_state = cache_store_op(
            op_id=f"{prefix}.conv_state", deps=(conv.op_id,), stage=stage,
            layer=plan.index, requests=_requests(views), phase=mode,
            recipe="catalog_gdn_conv_state",
            rows=max(1, plan.conv_kernel - 1) * max(1, len(views)),
            row_bytes=(2 * local_key + local_value) * 2, dtype="bf16",
            kind="gdn_conv_ring",
        )
        out_gate = vector_op(
            op_id=f"{prefix}.out_gate", deps=(state.op_id, split.op_id),
            stage=stage, layer=plan.index, requests=_requests(views),
            elements=max(1, padded_tokens * local_value), ops_per_element=4.0,
            dtype=model.activation_dtype, family="linear_output_gate",
            phase=mode, recipe="catalog_gdn_out_gate",
            semantic_family="gdn_output_swish_gate",
        )
        o_proj = gemm_op(
            op_id=f"{prefix}.lin_o", deps=(out_gate.op_id, conv_state.op_id),
            stage=stage, layer=plan.index, requests=_requests(views),
            m=padded_tokens, n=h, k=local_value, quant=quant, category="gdn",
            phase=mode, family="linear_o_projection",
            semantic_family="gdn_o_projection",
        )
        ops.extend([qkvz, ba, split, conv, gate, state, conv_state, out_gate,
                    o_proj])
        last = o_proj.op_id
    if parallel.tp > 1:
        ar = _allreduce(prefix=prefix, deps=(last,), stage=stage,
                        layer=plan.index, requests=_requests(views),
                        model=model, parallel=parallel, mode=mode,
                        tokens=padded_tokens, suffix="linear")
        ops.append(ar)
        last = ar.op_id
    return AttentionResult(
        ops=ops, last=last, output_ids=(last,),
        cache_store_id=None, attention_outputs=(last,),
        cache_write_rows=padded_tokens,
    )
