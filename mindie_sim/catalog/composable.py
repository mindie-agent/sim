"""Compose external model descriptions from supported structural primitives.

[Role] Assemble standard family adapters and partially understood HF configs.
[Invariants] No unknown attention is substituted with GQA/GDN. Unsupported
  cores remain explicit holes; their known projections and FFNs still execute.
  Structural coverage says nothing about measured timing accuracy.
[Timeouts & Locks] Pure bounded metadata arithmetic, no weights or network.
[Side-Effect Policy] Creates immutable ModelSpec/LayerPlan records.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from collections.abc import Mapping

from .config import flatten_config
from .hf_config import normalize_config, positive, resolve_quant, unrecognized_fields
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, MlaSpec, ModelSpec


def _schedule(c, layers, gaps):
    schedule = c.get("layer_attention_schedule", c.get("layer_types"))
    if schedule is not None:
        if len(schedule) != layers:
            raise ModelingError("model_attention_schedule_length_mismatch")
        return list(schedule)
    if "attn_type_list" in c:
        if len(c["attn_type_list"]) != layers:
            raise ModelingError("model_attention_schedule_length_mismatch")
        if str(c.get("model_type", "")) in {"minimax_m1", "minimax_text_01"}:
            # Pinned MiniMax M1/Text01 caller: 0=Lightning, 1=full attention.
            return [{0: "lightning", 1: "gqa"}.get(v, f"custom:{v}")
                    for v in c["attn_type_list"]]
        return [f"unresolved_attention_code:{v}" for v in c["attn_type_list"]]
    kind = c.get("attention_type", "mla" if c.get("kv_lora_rank") else "gqa")
    if "share_q_dim" in c:
        gaps.append("custom_shared_query_projection:attention_geometry_unresolved")
        kind = "unmodeled"
    return [kind] * layers


def generic_plans(c):
    h = positive(c, "hidden_size")
    layers = positive(c, "num_hidden_layers")
    heads = positive(c, "num_attention_heads")
    dim = positive(c, "head_dim", h // heads)
    kv = positive(c, "num_key_value_heads", heads)
    experts = int(c.get("n_routed_experts") or 0)
    topk = int(c.get("num_experts_per_tok") or 0)
    intermediate = positive(c, "intermediate_size", c.get("moe_intermediate_size"))
    moe_i = int(c.get("moe_intermediate_size") or intermediate)
    shared = int(c.get("n_shared_experts") or 0)
    first = int(c.get("first_k_dense_replace") or 0)
    freq = positive(c, "moe_layer_freq", 1)
    if experts and not 0 < topk <= experts:
        raise ModelingError("model_moe_topk_out_of_range")
    if heads % kv:
        raise ModelingError("model_gqa_group_not_integral")
    gaps, plans = [], []
    schedule = _schedule(c, layers, gaps)
    moe_layers = c.get("moe_layers_enum")
    if isinstance(moe_layers, str):
        moe_layers = {int(v.strip()) for v in moe_layers.split(",") if v.strip()}
    dense_layers = set(c.get("mlp_only_layers") or ())
    sparse = c.get("sparse_attention_config") or {}
    for index, original_kind in enumerate(schedule):
        kind = {"full_attention": "gqa", "mha": "gqa", "mqa": "gqa",
                "sliding_attention": "gqa"}.get(str(original_kind), str(original_kind))
        moe = bool(experts and index >= first and index % freq == 0
                   and index not in dense_layers
                   and (moe_layers is None or index in moe_layers))
        fields = dict(
            index=index, native_layer_id=index, attention=kind, architecture=str(original_kind),
            num_heads=heads, num_kv_heads=kv, head_dim=dim,
            ffn="moe" if moe else "dense", intermediate_size=intermediate,
            num_experts=experts if moe else 0, top_k=topk if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=int(c.get("shared_expert_intermediate_size") or 0),
            shared_expert_gate=bool(c.get("shared_expert_gate", False)),
            router_input_size=h, router_mode=str(c.get("scoring_func", "softmax")),
            router_norm_topk=bool(c.get("norm_topk_prob", False)),
            kv_owner_layer_id=index, owns_long_kv=True,
            qk_norm=bool(c.get("use_qk_norm", c.get("qk_norm", False))),
            q_output_gate=bool(c.get("attn_output_gate", False)),
            expert_latent_size=int(c.get("latent_moe_hidden_size") or 0) if moe else 0,
            window=int(c.get("sliding_window") or 0)
            if original_kind == "sliding_attention" or c.get("model_type") == "mistral" else 0,
        )
        if kind == "mla":
            fields.update(
                q_lora_rank=int(c.get("q_lora_rank") or 0),
                kv_lora_rank=positive(c, "kv_lora_rank"),
                qk_nope_head_dim=positive(c, "qk_nope_head_dim"),
                qk_rope_head_dim=int(c.get("qk_rope_head_dim") or 0),
                v_head_dim=positive(c, "v_head_dim", dim), mla_variant="latent_mla",
            )
        elif kind in {"gdn", "kda"}:
            linear = c.get("linear_attention_config") or {}
            kh = int(linear.get("num_k_heads", 0))
            vh = int(linear.get("num_v_heads", 0))
            kd = int(linear.get("head_k_dim", 0))
            vd = int(linear.get("head_v_dim", 0))
            conv = int(linear.get("conv_kernel_size", 0))
            if min(kh, vh, kd, vd, conv) <= 0 or (kind == "kda" and (kh != vh or kd != vd)):
                fields["attention"] = "unmodeled"
                gaps.append(f"recurrent_geometry_unresolved:layer{index}:{kind}")
            else:
                width = 2 * kh * kd + vh * vd
                fields.update(
                    linear_heads=vh, linear_head_dim=vd, linear_key_heads=kh,
                    linear_key_head_dim=kd, linear_value_heads=vh, conv_kernel=conv,
                    conv_width=width, recurrent_state_elements=vh * kd * vd,
                    conv_state_elements=width * (conv - 1),
                )
        elif kind == "lightning":
            # Known MiniMax matrices are [H,3*heads*D], [H,heads*D] gate,
            # and [heads*D,H]. The Lightning recurrence is not GDN.
            fields.update(attention="projection_only", num_kv_heads=heads, q_output_gate=True,
                          cache_kind="recurrent_unmodeled",
                          recurrent_state_elements=heads * dim * dim)
            gaps.append(f"lightning_attention_core_and_gated_norm:layer{index}")
        elif kind not in {"gqa", "unmodeled"}:
            fields.update(attention="projection_only", cache_kind="none")
            gaps.append(f"unsupported_attention_layer:{index}:{kind}")
        elif kind == "unmodeled":
            gaps.append(f"unmodeled_attention_block:layer{index}")
        if sparse and index >= int(sparse.get("sparse_layers_start", 0)):
            fields["attention"] = "projection_only"
            gaps.append(f"sparse_selection_and_attention_core:layer{index}")
        plans.append(LayerPlan(**fields))
    for name in ("num_mtp_layers", "num_nextn_predict_layers", "mtp_num_hidden_layers"):
        if c.get(name):
            gaps.append(f"optional_draft_structure_unresolved:{name}")
    for name in ("ngram_embedding", "zero_expert_num", "dit_config", "postnorm",
                 "sparse_attention_config", "routed_scaling_factor", "swiglu_limit",
                 "attention_bias", "mlp_bias"):
        if c.get(name):
            gaps.append(f"custom_submodule:{name}")
    if c.get("hidden_act", "silu") not in {"silu", "swiglu"}:
        gaps.append(f"ffn_activation:{c['hidden_act']}:silu_proxy")
    if float(c.get("partial_rotary_factor", 1.0)) != 1:
        gaps.append("partial_rotary_dimensions:full_width_proxy")
    if c.get("qkv_bias"):
        gaps.append("custom_submodule:qkv_bias")
    return plans, gaps


def shared_owners(c, plans, gaps):
    """Resolve layer ownership for every adapter, independently of family."""
    sources = c.get("kv_source_layer_ids", c.get("kv_consumer_to_owner"))
    owners = set(c.get("kv_owner_layers") or ())
    if sources is None and not owners:
        return plans
    if sources is not None and not isinstance(sources, Mapping) and len(sources) != len(plans):
        raise ModelingError("model_kv_owner_schedule_length_mismatch")
    previous = None
    result = []
    for plan in plans:
        index = plan.index
        if index in owners:
            previous = index
        if sources is None:
            owner = previous
        elif isinstance(sources, Mapping):
            owner = sources.get(str(index), sources.get(index, index))
        else:
            owner = sources[index]
        if plan.attention not in {"gqa", "projection_only"} or plan.cache_kind != "dense":
            if owner is not None and int(owner) != index:
                gaps.append(f"kv_sharing_for_unsupported_core:layer{index}")
            result.append(plan)
            continue
        if owner is None or int(owner) < 0 or int(owner) > index:
            raise ModelingError(f"model_kv_owner_invalid:{index}:{owner}")
        owner = int(owner)
        owner_plan = plans[owner]
        if owner != index and (
            owner_plan.attention not in {"gqa", "projection_only"}
            or owner_plan.cache_kind != "dense"
            or (owner_plan.num_kv_heads, owner_plan.head_dim) != (plan.num_kv_heads, plan.head_dim)
        ):
            raise ModelingError(f"model_shared_kv_geometry_mismatch:{index}:{owner}")
        # A mapping through another consumer resolves to its physical owner.
        owner = result[owner].kv_owner_layer_id if owner < index else owner
        result.append(replace(plan, kv_owner_layer_id=owner, owns_long_kv=owner == index))
    return result


def build_composable(key, config, family="generic"):
    c = normalize_config(config)
    aux = None
    if family == "standard_deepseek":
        from .deepseek import _standard_deepseek_plans, _standard_deepseek_gaps, _aux_standard_deepseek
        plans = _standard_deepseek_plans(c)
        gaps, aux = list(_standard_deepseek_gaps(c)), _aux_standard_deepseek(c)
    elif family == "standard_glm":
        from .glm import _standard_glm_plans, _standard_glm_gaps, _aux_standard_glm
        if c.get("kv_lora_rank"):
            from .deepseek import _standard_deepseek_plans
            plans = _standard_deepseek_plans(c)
        else:
            plans = _standard_glm_plans(c)
        gaps, aux = list(_standard_glm_gaps(c)), _aux_standard_glm(c)
    elif family == "standard_qwen":
        from .qwen import _standard_qwen_plans, _standard_qwen_gaps, _aux_standard_qwen
        plans = _standard_qwen_plans(c)
        gaps, aux = list(_standard_qwen_gaps(c)), _aux_standard_qwen(c)
    elif family == "standard_kimi":
        from .kimi import _standard_kimi_plans, _standard_kimi_gaps, _aux_standard_kimi
        plans = _standard_kimi_plans(c)
        gaps, aux = list(_standard_kimi_gaps(c)), _aux_standard_kimi(c)
    else:
        plans, gaps = generic_plans(c)
        if str(c.get("model_type", "")) not in {"llama", "mistral"}:
            gaps.append("generic_architecture_semantics_not_fully_verified")
    plans = shared_owners(c, plans, gaps)
    quant, quant_gaps = resolve_quant(c)
    gaps.extend(quant_gaps)
    if c.get("hidden_act", "silu") not in {"silu", "swiglu", "situ"}:
        gaps.append(f"ffn_activation:{c['hidden_act']}:silu_proxy")
    unknown = unrecognized_fields(config)
    gaps.extend("unrecognized_config_field:" + field for field in unknown)
    _, text, vision = flatten_config(config)
    if vision or config.get("vision_tower"):
        gaps.append("vision_encoder_and_projector:language_backbone_only")
    if c.get("rope_scaling"):
        gaps.append("rope_scaling_variant")
    gaps = tuple(sorted(set(gaps)))
    h = positive(c, "hidden_size")
    first = plans[0]
    mla_plan = next((p for p in plans if p.attention == "mla"), None)
    mla = (MlaSpec(mla_plan.q_lora_rank, mla_plan.kv_lora_rank,
                   mla_plan.qk_nope_head_dim, mla_plan.qk_rope_head_dim, mla_plan.v_head_dim)
           if mla_plan else None)
    metadata = {
        "model_coverage": "partial" if gaps else "complete",
        "unmodeled_features": gaps,
        "degraded_submodules": tuple(gap for gap in gaps if "proxy" in gap or
                                    "unmodeled" in gap or "unresolved" in gap or "core" in gap),
        "unrecognized_config_fields": {field: c[field] for field in unknown},
        "source_config_sha256": hashlib.sha256(json.dumps(
            config, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "source_model_type": str(c.get("model_type", "")),
        "declared_quantization": c.get("quantization_config", {}),
    }
    from mindie_sim.schema.serialization import immutable_sequences
    metadata = immutable_sequences(metadata)
    model = ModelSpec(
        name=key, layers=len(plans), body_layers=len(plans), hidden_size=h,
        num_heads=first.num_heads, num_kv_heads=first.num_kv_heads, head_dim=first.head_dim,
        intermediate_size=positive(c, "intermediate_size", c.get("moe_intermediate_size")),
        vocab_size=positive(c, "vocab_size"), attention=first.attention,
        ffn="moe" if any(p.ffn == "moe" for p in plans) else "dense",
        activation_dtype=quant.native_format, compute_dtype=quant.native_format,
        weight_dtype=quant.weight_format, kv_dtype=quant.kv_dtype, quant=quant,
        quantization=quant.name, norm_eps=float(c.get("rms_norm_eps", 1e-6)),
        num_experts=max(p.num_experts for p in plans), top_k=max(p.top_k for p in plans),
        moe_intermediate_size=max(p.moe_intermediate_size for p in plans),
        num_shared_experts=max(p.num_shared_experts for p in plans),
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        expert_routing_source="uniform", layer_plans=tuple(plans), mla=mla,
        auxiliary=aux, metadata=tuple(sorted(metadata.items())),
        state_bytes_per_request=sum(p.state_bytes_total for p in plans),
    )
    from .specs import _set_facts, _model_weight_totals
    from .weights import cache_row_bytes_by_layer
    params, weights, draft_params, draft_weights = _model_weight_totals(model)
    _set_facts(model, (
        ("profile", family), ("model_type", c.get("model_type", "")),
        ("parameter_count_body", params), ("resident_weight_bytes_body", weights),
        ("parameter_count_optional_draft", draft_params), ("optional_draft_weight_bytes", draft_weights),
        ("cache_row_bytes_by_layer", cache_row_bytes_by_layer(model)),
        ("model_coverage", metadata["model_coverage"]),
        ("attention_types", {kind: sum(p.attention == kind for p in plans)
                             for kind in sorted({p.attention for p in plans})}),
        ("ffn_types", {kind: sum(p.ffn == kind for p in plans)
                       for kind in sorted({p.ffn for p in plans})}),
    ))
    return model
