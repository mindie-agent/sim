"""GLM 5.x MLA/DSA and standard GLM-4/GLM-4.5 adapters.

[Role]
  Normalize dense/MoE schedule, full/shared indexer types and MTP separation.
  Both GLM 5 releases share the existing adapter: the 78-layer MLA/DSA body
  and the 45-layer Flash hybrid whose native ``layer_types`` interleaves 34
  KDA linear-attention layers with 11 MLA/DSA layers.  Standard GLM-4 uses
  the modern GQA or legacy ChatGLM aliases; GLM-4.5/Air are GQA+MoE, not MLA.

[Invariants]
  * Body and MTP remain distinct scopes.
  * Full/shared index ownership keeps its derived-source status explicit.
  * qk and v dimensions remain independent of raw head_dim.
  * A hybrid schedule is read from the native ``layer_types`` array, one
    entry per body layer; a missing/short/long schedule is refused rather
    than padded with a neighbouring layer kind.
  * KDA layers own no token KV: their per-request cost is the recurrent state
    plus the bounded convolution ring, exactly like the Kimi/Bailing KDA
    plans.  Only MLA/DSA layers own ``kv_lora_rank + qk_rope`` cache rows.
  * Standard GLM-4/4.5 plans are GQA with ``2*num_kv_heads*head_dim`` cache
    rows, not latent MLA; partial RoPE, bias and extra norms become gaps.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.common import ModelingError


def _glm_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    if c.get("layer_types") or c.get("linear_attn_config"):
        return _glm_hybrid_plans(c)
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
        raise ModelingError("model_config_glm_mla_dims_missing")
    if qk_nope + rope != qk_head:
        raise ModelingError(f"model_config_glm_qk_split_mismatch:{qk_nope}+{rope}!={qk_head}")
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
            router_dtype="fp32" if c.get("moe_router_dtype") == "float32" else "",
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


def _glm_hybrid_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Plans for the Flash release: native ``layer_types`` + KDA config.

    The schedule is the checkpoint's own zero-based array (45 entries:
    ``linear_attention`` / ``deepseek_sparse_attention``).  Every KDA entry is
    an O(1)-in-sequence recurrent layer; every DSA entry is the GLM latent-MLA
    layer with a full indexer that owns its token-level index plane.
    """
    layers = int(_require(c, "num_hidden_layers"))
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(c.get("num_key_value_heads", heads) or heads)
    q_rank = int(_require(c, "q_lora_rank"))
    kv_rank = int(_require(c, "kv_lora_rank"))
    qk_head = int(c.get("qk_head_dim", 0) or 0)
    qk_nope = int(c.get("qk_nope_head_dim", 0) or 0)
    rope = int(c.get("qk_rope_head_dim", 0) or 0)
    v_dim = int(c.get("v_head_dim", 0) or 0)
    if qk_head <= 0 or qk_nope <= 0 or v_dim <= 0:
        raise ModelingError("model_config_glm_mla_dims_missing")
    if qk_nope + rope != qk_head:
        raise ModelingError(
            f"model_config_glm_qk_split_mismatch:{qk_nope}+{rope}!={qk_head}"
        )
    layer_types = [str(v) for v in (c.get("layer_types") or ())]
    if len(layer_types) != layers:
        raise ModelingError(
            f"model_config_glm_layer_schedule_mismatch:{len(layer_types)}!={layers}"
        )
    mlp_types = [str(v) for v in (c.get("mlp_layer_types") or ())]
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "n_routed_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    linear = c.get("linear_attn_config")
    if not isinstance(linear, Mapping):
        raise ModelingError("model_config_glm_linear_attn_missing")
    kda_heads = int(linear.get("num_heads", 0) or 0)
    kda_dim = int(linear.get("head_dim", 0) or 0)
    conv_k = int(linear.get("short_conv_kernel_size", 4) or 4)
    if kda_heads <= 0 or kda_dim <= 0:
        raise ModelingError("model_config_glm_kda_heads_missing")
    conv_width = 3 * kda_heads * kda_dim
    hyper = bool(c.get("mhc"))
    hc_mult = _int(c, "hc_mult") if hyper else 0
    hc_iters = _int(c, "hc_sinkhorn_iters") if hyper else 0
    common = {
        "intermediate_size": dense_i,
        "num_experts": experts,
        "top_k": top_k,
        "moe_intermediate_size": moe_i,
        "num_shared_experts": shared,
        "router_input_size": h,
        "router_mode": "sigmoid",
        "router_dtype": "fp32" if c.get("moe_router_dtype") == "float32" else "",
        "router_norm_topk": bool(c.get("norm_topk_prob", False)),
        "router_groups": _int(c, "n_group"),
        "router_group_topk": _int(c, "topk_group"),
        "hyper_connections": hyper,
        "hc_mult": hc_mult,
        "hc_sinkhorn_iters": hc_iters,
        "native_layer_id": 0,
        "scope": "body",
    }
    plans: list[LayerPlan] = []
    for index, kind in enumerate(layer_types):
        ffn = (
            "dense"
            if (index < len(mlp_types) and mlp_types[index] == "dense")
            else "moe"
        )
        ffn_fields = dict(
            common,
            ffn=ffn,
            intermediate_size=dense_i if ffn == "dense" else 0,
            num_experts=experts if ffn == "moe" else 0,
            top_k=top_k if ffn == "moe" else 0,
            moe_intermediate_size=moe_i if ffn == "moe" else 0,
            num_shared_experts=shared if ffn == "moe" else 0,
            shared_intermediate_size=shared * moe_i if ffn == "moe" else 0,
            native_layer_id=index,
        )
        if kind in {"linear_attention", "kda"}:
            plans.append(LayerPlan(
                index=index,
                attention="kda",
                architecture="kda",
                linear_heads=kda_heads,
                linear_head_dim=kda_dim,
                linear_key_heads=kda_heads,
                linear_key_head_dim=kda_dim,
                linear_value_heads=kda_heads,
                packed_input_projection=True,
                gate_rank=kda_dim,
                projection_dtype="bf16",
                conv_kernel=conv_k,
                conv_width=conv_width,
                recurrent_state_elements=kda_heads * kda_dim * kda_dim,
                recurrent_state_dtype="fp32",
                conv_state_elements=conv_width * (conv_k - 1),
                conv_state_dtype="bf16",
                state_bytes=float(
                    kda_heads * kda_dim * kda_dim * 4
                    + conv_width * (conv_k - 1) * 2
                ),
                kv_owner_layer_id=index,
                **ffn_fields,
            ))
        elif kind in {"deepseek_sparse_attention", "full_attention", "mla"}:
            plans.append(LayerPlan(
                index=index,
                attention="mla",
                architecture="mla_dsa",
                indexer=True,
                indexer_type="full",
                num_heads=heads,
                num_kv_heads=kv_heads,
                head_dim=qk_head,
                q_lora_rank=q_rank,
                kv_lora_rank=kv_rank,
                qk_nope_head_dim=qk_nope,
                qk_rope_head_dim=rope,
                v_head_dim=v_dim,
                mla_variant="latent_mla",
                mla_output_gate=False,
                packed_input_projection=True,
                projection_dtype="bf16",
                kv_cache_elements=kv_rank + rope,
                kv_owner_layer_id=index,
                index_k_owner_layer_id=index,
                owns_index_k=True,
                topk_owner_layer_id=index,
                owns_topk=True,
                index_source_layer_id=index,
                index_source_status="exact_native_full_schedule",
                **ffn_fields,
            ))
        else:
            raise ModelingError(
                f"model_config_glm_layer_kind_unsupported:{index}:{kind}"
            )
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
        weight_category="mtp",
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


def _glm_alias_int(c: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key in c and c[key] is not None:
            return int(c[key])
    raise ModelingError("model_config_missing:" + keys[0])


def _glm_rope_split(c: Mapping[str, Any], head_dim: int) -> tuple[int, int]:
    factor = float(c.get("partial_rotary_factor", 1.0))
    rope = int(head_dim * factor)
    if rope < 0 or rope > head_dim:
        raise ModelingError(f"model_config_glm_partial_rope_invalid:{factor}")
    return head_dim - rope, rope


def _standard_glm_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Body plans for GLM-4 modern/legacy and GLM-4.5/Air GQA checkpoints."""
    model_type = str(c.get("model_type", ""))
    if model_type == "chatglm":
        return _standard_legacy_chatglm_plans(c)
    layers = int(_require(c, "num_hidden_layers"))
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(c.get("num_key_value_heads", heads) or heads)
    head_dim = int(_require(c, "head_dim"))
    nope, rope = _glm_rope_split(c, head_dim)
    dense_i = int(_require(c, "intermediate_size"))
    if min(layers, h, heads, kv_heads, head_dim, dense_i) <= 0:
        raise ModelingError("model_config_glm_standard_geometry_invalid")
    experts_raw = c.get("n_routed_experts")
    experts = _int(c, "n_routed_experts") if experts_raw is not None else 0
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts")
    first_dense = _int(c, "first_k_dense_replace")
    qk_norm = bool(c.get("use_qk_norm", False))
    plans: list[LayerPlan] = []
    for index in range(layers):
        moe = experts_raw is not None and index >= first_dense
        if moe and (experts <= 0 or top_k <= 0 or moe_i <= 0):
            raise ModelingError(f"model_config_glm_moe_dims_invalid:{index}")
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            ffn="moe" if moe else "dense",
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            qk_nope_head_dim=nope, qk_rope_head_dim=rope, qk_norm=qk_norm,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
            intermediate_size=0 if moe else dense_i,
            num_experts=experts if moe else 0,
            top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=shared * moe_i if moe else 0,
            router_input_size=h, router_mode="sigmoid", router_dtype="fp32",
            router_norm_topk=bool(c.get("norm_topk_prob", False)),
            router_groups=_int(c, "n_group"),
            router_group_topk=_int(c, "topk_group"),
            native_layer_id=index, scope="body",
        ))
    return plans


def _standard_legacy_chatglm_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    layers = _glm_alias_int(c, "num_hidden_layers", "num_layers")
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    head_dim = _glm_alias_int(c, "kv_channels", "head_dim")
    multi_query = bool(c.get("multi_query_attention", "multi_query_group_num" in c))
    kv_heads = _glm_alias_int(c, "multi_query_group_num") if multi_query else heads
    dense_i = _glm_alias_int(c, "ffn_hidden_size", "intermediate_size")
    if min(layers, h, heads, kv_heads, head_dim, dense_i) <= 0:
        raise ModelingError("model_config_glm_legacy_geometry_invalid")
    return [LayerPlan(
        index=index, attention="gqa", architecture="gqa", ffn="dense",
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
        # Native ChatGLMModel builds RotaryEmbedding(rotary_dim // 2).
        qk_nope_head_dim=head_dim - head_dim // 2,
        qk_rope_head_dim=head_dim // 2,
        kv_cache_elements=2 * kv_heads * head_dim,
        kv_owner_layer_id=index, owns_long_kv=True,
        intermediate_size=dense_i, native_layer_id=index, scope="body",
    ) for index in range(layers)]


def _aux_standard_glm(c: Mapping[str, Any]) -> AuxiliarySpec | None:
    if _int(c, "num_nextn_predict_layers") > 0:
        return None
    return AuxiliarySpec()


def _standard_glm_gaps(c: Mapping[str, Any]) -> tuple[str, ...]:
    model_type = str(c.get("model_type", ""))
    gaps: list[str] = []
    if model_type == "chatglm":
        head_dim = _glm_alias_int(c, "kv_channels", "head_dim")
        gaps.append(f"partial_rope_{head_dim // 2}_of_{head_dim}_not_in_attention_plan")
        if not bool(c.get("rmsnorm", False)):
            gaps.append("legacy_layernorm_not_rmsnorm_not_modeled")
        if bool(c.get("add_bias_linear", False)):
            gaps.append("linear_bias_not_in_layer_plan")
        if bool(c.get("add_qkv_bias", False)):
            gaps.append("attention_bias_qkv_not_in_layer_plan")
        if bool(c.get("apply_residual_connection_post_layernorm", False)):
            gaps.append("residual_connection_post_layernorm_not_modeled")
        return tuple(gaps)
    head_dim = _int(c, "head_dim")
    factor = float(c.get("partial_rotary_factor", 1.0))
    rope = int(head_dim * factor)
    if factor != 1.0:
        gaps.append(f"partial_rope_{rope}_of_{head_dim}_not_in_attention_plan")
    if bool(c.get("attention_bias", False)):
        gaps.append("attention_bias_qkv_not_in_layer_plan")
    if model_type == "glm4":
        gaps.append("additional_post_self_attn_and_post_mlp_norms_not_in_layer_schema")
    if model_type == "glm4_moe" and c.get("n_routed_experts") is not None:
        gaps.append("router_sigmoid_correction_bias_and_routed_scaling_not_in_layer_plan")
        if _int(c, "num_nextn_predict_layers") > 0:
            gaps.append("optional_mtp_not_modeled")
    return tuple(gaps)
