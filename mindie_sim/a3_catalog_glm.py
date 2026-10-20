"""GLM 5.3 MLA/DSA body and MTP adapters.

[Role]
  Normalize dense/MoE schedule, full/shared indexer types and MTP separation.

[Invariants]
  * Body78 and MTP1 remain distinct scopes.
  * Full/shared index ownership keeps its derived-source status explicit.
  * qk and v dimensions remain independent of raw head_dim.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_catalog_config import _int, _require
from .a3_catalog_types import AuxiliarySpec, DraftSpec, merge_source
from .a3_model_types import LayerPlan
from .a3_types import A3InputError


def _glm_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    layers = int(_require(c, "num_hidden_layers"))
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    q_rank = int(_require(c, "q_lora_rank"))
    kv_rank = int(_require(c, "kv_lora_rank"))
    qk_head = int(c.get("qk_head_dim", 0) or 0)
    qk_nope = int(c.get("qk_nope_head_dim", 0) or 0)
    rope = int(c.get("qk_rope_head_dim", 0) or 0)
    v_dim = int(c.get("v_head_dim", 0) or 0)
    if qk_head <= 0 or qk_nope <= 0 or rope <= 0 or v_dim <= 0:
        raise A3InputError("model_config_glm_mla_dims_missing")
    if qk_nope + rope != qk_head:
        raise A3InputError(f"model_config_glm_qk_split_mismatch:{qk_nope}+{rope}!={qk_head}")
    mlp_types = [str(v) for v in (c.get("mlp_layer_types") or ())]
    index_types = [str(v) for v in (c.get("indexer_types") or ())]
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "n_routed_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    full_layers = [index for index, kind in enumerate(index_types) if kind == "full"]
    if len(index_types) > layers:
        index_types = index_types[:layers]
    plans: list[LayerPlan] = []
    for index in range(layers):
        ffn = "dense" if (index < len(mlp_types) and mlp_types[index] == "dense") else "moe"
        indexer_type = index_types[index] if index < len(index_types) else "shared"
        indexer = indexer_type == "full"
        owner = max((v for v in full_layers if v <= index), default=-1)
        plans.append(LayerPlan(
            index=index, attention="mla", architecture="mla_dsa", ffn=ffn,
            indexer=indexer,
            indexer_type=indexer_type if indexer_type in {"full", "shared"} else "shared",
            num_heads=heads, num_kv_heads=int(c.get("num_key_value_heads", heads) or heads),
            head_dim=qk_head, q_lora_rank=q_rank, kv_lora_rank=kv_rank,
            qk_nope_head_dim=qk_nope, qk_rope_head_dim=rope, v_head_dim=v_dim,
            mla_variant="latent_mla", kv_cache_elements=kv_rank + rope,
            intermediate_size=dense_i if ffn == "dense" else 0,
            num_experts=experts if ffn == "moe" else 0,
            top_k=top_k if ffn == "moe" else 0,
            moe_intermediate_size=moe_i if ffn == "moe" else 0,
            num_shared_experts=shared if ffn == "moe" else 0,
            router_input_size=h, router_mode="sigmoid",
            router_norm_topk=bool(c.get("norm_topk_prob", False)),
            router_groups=_int(c, "n_group"),
            router_group_topk=_int(c, "topk_group"),
            shared_intermediate_size=shared * moe_i if ffn == "moe" else 0,
            index_source_layer_id=owner,
            index_k_owner_layer_id=index if indexer else owner,
            owns_index_k=indexer,
            owns_topk=indexer,
            topk_owner_layer_id=owner if not indexer else index,
            index_source_status=(
                "exact_native_full_schedule" if indexer
                else "derived_conservative_nearest_full"
            ),
            kv_owner_layer_id=index,
            native_layer_id=index, scope="body",
        ))
    return plans


def _aux_glm(c: Mapping[str, Any]) -> AuxiliarySpec:
    nextn = _int(c, "num_nextn_predict_layers")
    if nextn <= 0:
        return AuxiliarySpec()
    h = int(c["hidden_size"])
    experts = _int(c, "n_routed_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    mtp = DraftSpec(
        kind="mtp", layers=nextn, hidden_size=h, intermediate_size=moe_i,
        num_heads=int(c["num_attention_heads"]),
        num_kv_heads=int(c.get("num_key_value_heads", 1) or 1),
        head_dim=int(c.get("qk_head_dim", 256) or 256),
        q_width=int(c["num_attention_heads"]) * int(c.get("qk_head_dim", 256) or 256),
        kv_width=int(c.get("qk_head_dim", 256) or 256), attention_kind="mla",
        q_lora_rank=int(c["q_lora_rank"]), kv_lora_rank=int(c["kv_lora_rank"]),
        qk_nope_head_dim=int(c["qk_nope_head_dim"]),
        qk_rope_head_dim=int(c["qk_rope_head_dim"]),
        v_head_dim=int(c["v_head_dim"]), num_experts=experts, top_k=top_k,
        expert_intermediate_size=moe_i, num_shared_experts=shared,
        block_size=1, markov_rank=0, confidence_enabled=False,
        target_shared_embedding=True, target_shared_lm_head=True,
        base_logits=True, pp_capability=(1,),
        vocab_size=int(c["vocab_size"]),
        source=merge_source({"kind": "glm_mtp"}),
        notes=("GLM body is 78 layers; MTP is one optional layer",),
    )
    return AuxiliarySpec(mtp=mtp, notes=("body78_mtp1_distinct",))
