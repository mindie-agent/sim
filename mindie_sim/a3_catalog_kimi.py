"""Kimi K3 KDA/MLA/latent-MoE and external DSpark adapters.

[Role]
  Normalize native one-based attention layer lists, KDA recurrent/convolution
  state, latent routed experts, shared experts and residual-block metadata.

[Invariants]
  * Native one-based full/KDA lists are converted exactly once.
  * KDA state is heads*Dk*Dv FP32 plus a separate bounded convolution ring.
  * Routed width 3584 is distinct from residual/shared 7168.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_catalog_config import _int, _require, flatten_config
from .a3_catalog_types import AuxiliarySpec, DraftSpec, merge_source
from .a3_model_types import LayerPlan
from .a3_types import A3InputError


def _kimi_plans(config: Mapping[str, Any]) -> list[LayerPlan]:
    c, _, _ = flatten_config(config)
    h = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    experts = _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_token", _int(c, "num_experts_per_tok"))
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "num_shared_experts")
    dense_i = _int(c, "intermediate_size")
    dense_first = _int(c, "first_k_dense_replace")
    latent = _int(c, "routed_expert_hidden_size", h // 2)
    if latent <= 0 or latent >= h:
        raise A3InputError(f"model_config_kimi_latent_invalid:{latent}")
    linear = c.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise A3InputError("model_config_kimi_linear_attn_missing")
    full_one_based = {int(v) for v in linear.get("full_attn_layers", ())}
    kda_one_based = {int(v) for v in linear.get("kda_layers", ())}
    if not full_one_based and not kda_one_based:
        raise A3InputError("model_config_kimi_attention_layers_missing")
    if full_one_based and kda_one_based and full_one_based & kda_one_based:
        raise A3InputError("model_config_kimi_attention_layers_overlap")
    if full_one_based:
        full_zero = {v - 1 for v in full_one_based}
    else:
        full_zero = set(range(layers)) - {v - 1 for v in kda_one_based}
    kda_zero = set(range(layers)) - full_zero
    if any(v < 0 or v >= layers for v in full_zero | kda_zero):
        raise A3InputError("model_config_kimi_native_layer_out_of_range")
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    q_rank = int(_require(c, "q_lora_rank"))
    kv_rank = int(_require(c, "kv_lora_rank"))
    qk_nope = int(_require(c, "qk_nope_head_dim"))
    rope = int(_require(c, "qk_rope_head_dim"))
    v_dim = int(c.get("v_head_dim", 128) or 128)
    kda_heads = int(linear.get("num_heads", 0) or 0)
    kda_dim = int(linear.get("head_dim", 0) or 0)
    conv_k = int(linear.get("short_conv_kernel_size", 4) or 4)
    if kda_heads <= 0 or kda_dim <= 0:
        raise A3InputError("model_config_kimi_kda_heads_missing")
    conv_width = 3 * kda_heads * kda_dim
    plans: list[LayerPlan] = []
    for index in range(layers):
        ffn = "dense" if index < dense_first else "moe"
        if index in kda_zero:
            plans.append(LayerPlan(
                index=index, attention="kda", architecture="kda", ffn=ffn,
                num_heads=0, num_kv_heads=0, head_dim=0,
                intermediate_size=dense_i if ffn == "dense" else 0,
                num_experts=experts if ffn == "moe" else 0,
                top_k=top_k if ffn == "moe" else 0,
                moe_intermediate_size=moe_i if ffn == "moe" else 0,
                num_shared_experts=shared if ffn == "moe" else 0,
                expert_latent_size=latent if ffn == "moe" else 0,
                router_input_size=h,
                shared_intermediate_size=shared * moe_i if ffn == "moe" else 0,
                router_mode="sigmoid",
                linear_heads=kda_heads, linear_head_dim=kda_dim,
                linear_key_heads=kda_heads, linear_key_head_dim=kda_dim,
                linear_value_heads=kda_heads,
                conv_kernel=conv_k, conv_width=conv_width,
                recurrent_state_elements=kda_heads * kda_dim * kda_dim,
                recurrent_state_dtype="fp32",
                conv_state_elements=conv_width * (conv_k - 1),
                conv_state_dtype="bf16",
                state_bytes=float(kda_heads * kda_dim * kda_dim * 4 + conv_width * (conv_k - 1) * 2),
                residual_block_size=_int(c, "attn_res_block_size"),
                native_layer_id=index + 1, scope="body",
                kv_owner_layer_id=index,
            ))
        else:
            plans.append(LayerPlan(
                index=index, attention="mla", architecture="mla", ffn=ffn,
                num_heads=heads, num_kv_heads=kv_heads,
                head_dim=qk_nope + rope,
                q_lora_rank=q_rank, kv_lora_rank=kv_rank,
                qk_nope_head_dim=qk_nope, qk_rope_head_dim=rope, v_head_dim=v_dim,
                mla_variant="latent_mla", mla_output_gate=bool(c.get("mla_use_output_gate", True)),
                kv_cache_elements=kv_rank + rope,
                intermediate_size=dense_i if ffn == "dense" else 0,
                num_experts=experts if ffn == "moe" else 0,
                top_k=top_k if ffn == "moe" else 0,
                moe_intermediate_size=moe_i if ffn == "moe" else 0,
                num_shared_experts=shared if ffn == "moe" else 0,
                expert_latent_size=latent if ffn == "moe" else 0,
                router_input_size=h,
                shared_intermediate_size=shared * moe_i if ffn == "moe" else 0,
                router_mode="sigmoid",
                residual_block_size=_int(c, "attn_res_block_size"),
                native_layer_id=index + 1, scope="body",
                kv_owner_layer_id=index,
            ))
    return plans


def _aux_kimi(c: Mapping[str, Any], config: Mapping[str, Any]) -> AuxiliarySpec:
    h = int(c["hidden_size"])
    target_ids = (7, 23, 51, 67, 83)
    block = 7
    markov = 256
    dspark = DraftSpec(
        kind="dspark", layers=5, hidden_size=h, intermediate_size=14336,
        num_heads=64, num_kv_heads=16, head_dim=64,
        q_width=64 * 64, kv_width=16 * 64, attention_kind="gqa",
        target_layer_ids=target_ids, block_size=block, markov_rank=markov,
        confidence_enabled=True, confidence_executes=False,
        confidence_input_width=h + markov,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, markov_replicated=True,
        noncausal_query_block=True, pp_capability=(1,),
        vocab_size=int(c.get("vocab_size", 163840)), num_experts=0, top_k=0,
        source=merge_source({"kind": "external_kimi_k3_dspark"}),
        notes=("external five-layer dense draft; selected via explicit draft mode",),
    )
    return AuxiliarySpec(
        dspark=dspark, external_draft_key="kimi-k3-dspark",
        notes=("native K3 num_nextn_predict_layers=0 does not remove selected external DSpark",),
    )
