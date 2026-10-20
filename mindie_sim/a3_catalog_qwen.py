"""Qwen3.8 GDN/gated-GQA and standalone DSpark adapters.

[Role]
  Normalize hybrid layer types, unequal GDN key/value heads, packed qkvz/ba
  widths, dense/MoE FFN and the standalone DSpark draft model.

[Invariants]
  * GDN key and value head counts/dims are independent.
  * Gated Q projection doubles the Q width without changing K/V width.
  * DSpark is recognized by its architecture, not generic model_type=qwen3.

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


def _qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    h = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    layer_types = [str(v) for v in (c.get("layer_types") or ())]
    if len(layer_types) != layers:
        raise A3InputError(f"model_config_qwen_layer_types_mismatch:{len(layer_types)}!={layers}")
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared_i = _int(c, "shared_expert_intermediate_size")
    key_heads = int(_require(c, "linear_num_key_heads"))
    value_heads = int(_require(c, "linear_num_value_heads"))
    key_dim = int(_require(c, "linear_key_head_dim"))
    value_dim = int(_require(c, "linear_value_head_dim"))
    conv_k = int(_require(c, "linear_conv_kernel_dim"))
    conv_width = 2 * key_heads * key_dim + value_heads * value_dim
    gate = bool(c.get("attn_output_gate", False))
    moe = experts > 0
    plans: list[LayerPlan] = []
    for index, kind in enumerate(layer_types):
        if kind == "linear_attention":
            plans.append(LayerPlan(
                index=index, attention="gdn", architecture="gdn",
                ffn="moe" if moe else "dense",
                num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
                intermediate_size=dense_i if not moe else 0,
                num_experts=experts if moe else 0, top_k=top_k if moe else 0,
                moe_intermediate_size=moe_i if moe else 0,
                num_shared_experts=1 if moe else 0,
                shared_intermediate_size=shared_i if moe else 0,
                shared_expert_gate=bool(moe),
                router_input_size=h, router_mode="sigmoid",
                linear_heads=value_heads, linear_head_dim=value_dim,
                linear_key_heads=key_heads, linear_key_head_dim=key_dim,
                linear_value_heads=value_heads,
                conv_kernel=conv_k, conv_width=conv_width,
                recurrent_state_elements=value_heads * key_dim * value_dim,
                recurrent_state_dtype="fp32",
                conv_state_elements=conv_width * (conv_k - 1),
                conv_state_dtype="bf16",
                state_bytes=float(value_heads * key_dim * value_dim * 4 + conv_width * (conv_k - 1) * 2),
                q_output_gate=gate,
                kv_owner_layer_id=index,
                native_layer_id=index, scope="body",
            ))
        elif kind == "full_attention":
            plans.append(LayerPlan(
                index=index, attention="gqa", architecture="gated_gqa" if gate else "gqa",
                ffn="moe" if moe else "dense",
                num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
                intermediate_size=dense_i if not moe else 0,
                num_experts=experts if moe else 0, top_k=top_k if moe else 0,
                moe_intermediate_size=moe_i if moe else 0,
                num_shared_experts=1 if moe else 0,
                shared_intermediate_size=shared_i if moe else 0,
                shared_expert_gate=bool(moe),
                router_input_size=h, router_mode="sigmoid",
                q_output_gate=gate,
                kv_cache_elements=2 * kv_heads * head_dim,
                kv_owner_layer_id=index,
                native_layer_id=index, scope="body",
            ))
        else:
            raise A3InputError(f"model_config_qwen_layer_type_unsupported:{kind}")
    return plans


def _aux_qwen(c: Mapping[str, Any]) -> AuxiliarySpec:
    nextn = _int(c, "mtp_num_hidden_layers")
    if nextn <= 0:
        return AuxiliarySpec()
    h = int(c["hidden_size"])
    heads = int(c["num_attention_heads"])
    kv_heads = int(c["num_key_value_heads"])
    head_dim = int(c["head_dim"])
    gate = bool(c.get("attn_output_gate", False))
    experts = _int(c, "num_experts")
    moe_i = _int(c, "moe_intermediate_size") if experts else _int(c, "intermediate_size")
    mtp = DraftSpec(
        kind="mtp", layers=nextn, hidden_size=h, intermediate_size=moe_i,
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
        q_width=(2 if gate else 1) * heads * head_dim,
        kv_width=kv_heads * head_dim, attention_kind="gqa",
        num_experts=experts, top_k=_int(c, "num_experts_per_tok") if experts else 0,
        expert_intermediate_size=moe_i,
        num_shared_experts=1 if experts else 0,
        block_size=1, markov_rank=0, confidence_enabled=False,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, pp_capability=(1,),
        vocab_size=int(c["vocab_size"]),
        source=merge_source({"kind": "qwen_mtp"}),
        notes=("Qwen body excludes the optional MTP layer",),
    )
    return AuxiliarySpec(mtp=mtp, notes=("qwen_mtp_is_optional",))


def _dspark_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    h = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    dense_i = int(_require(c, "intermediate_size"))
    return [
        LayerPlan(
            index=index, attention="gqa", architecture="gqa", ffn="dense",
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=dense_i,
            noncausal_attention=True,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index,
            native_layer_id=index, scope="body",
        )
        for index in range(layers)
    ]


def _aux_dspark(c: Mapping[str, Any]) -> AuxiliarySpec:
    h = int(c["hidden_size"])
    targets = tuple(int(v) for v in (c.get("dflash_config", {}) or {}).get("target_layer_ids", ()) or ())
    block = _int(c, "block_size", 7)
    markov = _int(c, "markov_rank", 256)
    spec = DraftSpec(
        kind="standalone_draft", layers=int(c["num_hidden_layers"]),
        hidden_size=h, intermediate_size=int(c["intermediate_size"]),
        num_heads=int(c["num_attention_heads"]),
        num_kv_heads=int(c["num_key_value_heads"]),
        head_dim=int(c["head_dim"]),
        q_width=int(c["num_attention_heads"]) * int(c["head_dim"]),
        kv_width=int(c["num_key_value_heads"]) * int(c["head_dim"]),
        attention_kind="gqa", target_layer_ids=targets, block_size=block,
        markov_rank=markov,
        confidence_enabled=bool(c.get("enable_confidence_head", False)),
        confidence_executes=False,
        confidence_input_width=h + markov,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, markov_replicated=True,
        noncausal_query_block=True, pp_capability=(1,),
        vocab_size=int(c["vocab_size"]),
        source=merge_source({"kind": "standalone_dspark_draft"}),
        notes=("DSpark draft model is its own body; target features are model inputs",),
    )
    return AuxiliarySpec(dspark=spec, standalone_draft=True)

