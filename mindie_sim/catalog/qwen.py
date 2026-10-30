"""Qwen3.8 hybrid and standard Qwen2/2.5/3 dense/MoE adapters.

[Role]
  Normalize hybrid layer types, unequal GDN key/value heads, packed qkvz/ba
  widths, dense/MoE FFN, the standalone DSpark draft model and the ordinary
  Qwen2/Qwen2.5/Qwen3 dense/MoE bodies.

[Invariants]
  * GDN key and value head counts/dims are independent.
  * Gated Q projection doubles the Q width without changing K/V width.
  * DSpark is recognized by its architecture, not generic model_type=qwen3.
  * Standard Qwen head expansion, QK-norm, bias gaps, MoE schedule and window
    placement come from the native config/source, not from the hybrid defaults.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require, flatten_config
from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec, merge_source
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.common import ModelingError


def _qwen_normalization(c: Mapping[str, Any]) -> dict[str, Any]:
    """The model class declares Gemma's weight shift independently of hardware."""
    if c.get("model_type") not in {"qwen3_5_text", "qwen3_5_moe_text"}:
        return {}
    return {
        "norm_weight_offset": 1.0,
        "norm_multiply_before_output_cast": True,
        "source": {
            "repository": "vllm",
            "revision": "752a3a504485790a2e8491cacbb35c137339ad34",
            "model_path": "vllm/model_executor/models/qwen3_5.py",
            "model_sha256": "5f47ae4f4a08d0a78dd681d58b290f3298744c73a82f1349f3e2853469ef73e6",
            "model_lines": "39,165-170,248",
            "normalization_path": "vllm/model_executor/layers/layernorm.py",
            "normalization_sha256": "696615d858292fea7a2942480803cd4c7d2d1e88d818bfe5cd9d695ac464f37c",
            "normalization_lines": "129-157",
            "boundary": "declared_model_semantics; not_capture_binary_identity",
        },
    }


def _qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    h = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = int(_require(c, "num_attention_heads"))
    kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim"))
    layer_types = [str(v) for v in (c.get("layer_types") or ())]
    if len(layer_types) != layers:
        raise ModelingError(f"model_config_qwen_layer_types_mismatch:{len(layer_types)}!={layers}")
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
            raise ModelingError(f"model_config_qwen_layer_type_unsupported:{kind}")
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
        weight_category="mtp", qk_norm=True, q_output_gate=gate,
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


# --- standard Qwen2/Qwen2.5/Qwen3 dense and MoE bodies ---------------------

_STANDARD_QWEN_TYPES = frozenset({"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"})


def _standard_qwen_family(c: Mapping[str, Any]) -> str:
    model_type = str(c.get("model_type", ""))
    if model_type not in _STANDARD_QWEN_TYPES:
        raise ModelingError(f"model_config_standard_qwen_unsupported:{model_type}")
    return model_type


def _standard_qwen_head_dim(
    c: Mapping[str, Any], family: str, hidden: int, heads: int,
) -> int:
    """Qwen2/3 honor an explicit head_dim; Qwen2-MoE derives hidden/heads."""
    if heads <= 0:
        raise ModelingError(f"model_config_qwen_heads_invalid:{heads}")
    raw = c.get("head_dim")
    if family == "qwen2_moe" or raw is None:
        if hidden % heads:
            raise ModelingError(
                f"model_config_qwen_head_dim_not_integral:{hidden}%{heads}")
        return hidden // heads
    head_dim = int(raw)
    if head_dim <= 0:
        raise ModelingError(f"model_config_qwen_head_dim_invalid:{head_dim}")
    return head_dim


def _standard_qwen_mlp_only(c: Mapping[str, Any]) -> tuple[set[int], tuple[str, ...]]:
    raw = c.get("mlp_only_layers")
    if raw is None:
        return set(), ()
    try:
        return {int(value) for value in raw}, ()
    except (TypeError, ValueError):
        return set(), ("mlp_only_layers_invalid",)


def _standard_qwen_ffn_config(
    c: Mapping[str, Any], family: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Decode only the declared Qwen MoE fields; no hybrid/Kimi defaults."""
    dense_i = _int(c, "intermediate_size")
    experts = _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    step = _int(c, "decoder_sparse_step", 1)
    if experts > 0 and step <= 0:
        raise ModelingError(f"model_config_qwen_decoder_sparse_step_invalid:{step}")
    mlp_only, issues = _standard_qwen_mlp_only(c)
    shared_count = 0
    shared_i = 0
    shared_gate = False
    if experts > 0 and family == "qwen2_moe":
        # Qwen2Moe always constructs one shared MLP plus its sigmoid gate.
        shared_i = _int(c, "shared_expert_intermediate_size")
        shared_count = 1 if shared_i > 0 else 0
        shared_gate = shared_i > 0
    elif experts > 0 and family == "qwen3_moe":
        # Qwen3Moe has no shared branch in the reference source. An explicit
        # extension is still honored instead of inheriting the Qwen2/hybrid one.
        shared_count = _int(c, "num_shared_experts", _int(c, "n_shared_experts"))
        shared_i = _int(c, "shared_expert_intermediate_size")
        if shared_count > 0 and shared_i <= 0:
            shared_i = shared_count * moe_i
        elif shared_count <= 0 and shared_i > 0:
            shared_count = 1
        if shared_count <= 0:
            shared_i = 0
        shared_gate = bool(c.get("shared_expert_gate", False)) if shared_i > 0 else False
    return {
        "dense_i": dense_i,
        "experts": experts,
        "top_k": top_k,
        "moe_i": moe_i,
        "step": step,
        "mlp_only": mlp_only,
        "shared_count": shared_count,
        "shared_i": shared_i,
        "shared_gate": shared_gate,
    }, issues


def _standard_qwen_layer_moe(index: int, ffn: Mapping[str, Any]) -> bool:
    """Native source schedule: one-based sparse step, dense-only overrides."""
    experts = int(ffn["experts"])
    if experts <= 0:
        return False
    return (
        index not in ffn["mlp_only"]
        and (index + 1) % int(ffn["step"]) == 0
    )


def _standard_qwen_window(
    c: Mapping[str, Any], layers: int,
) -> tuple[list[str], list[int], tuple[str, ...]]:
    """Return explicit layer kinds and actual per-index window widths.

    An explicit ordinary full/sliding ``layer_types`` schedule is authoritative.
    Without one, v4.51.3 FlashAttn activation is reproduced exactly:
    ``use_sliding_window and sliding_window is not None and
    layer_idx >= max_window_layers``. Unknown custom kinds stay explicit as the
    original kind instead of being coerced to full attention.
    """
    issues: list[str] = []
    raw_types = c.get("layer_types")
    if raw_types is not None:
        values = list(raw_types) if isinstance(raw_types, (list, tuple)) else []
        if not isinstance(raw_types, (list, tuple)):
            issues.append("layer_types_invalid")
        if len(values) != layers:
            issues.append(f"layer_types_length_mismatch:{len(values)}!={layers}")
        kinds, windows = [], []
        for index in range(layers):
            kind = str(values[index]) if index < len(values) else "unspecified_attention"
            window = 0
            if kind == "sliding_attention":
                declared = c.get("sliding_window")
                if declared is None:
                    issues.append(f"sliding_layer_type_without_declared_window:{index}")
                else:
                    window = int(declared)
                    if window <= 0:
                        issues.append(f"sliding_window_invalid:{window}")
                        window = 0
            elif kind != "full_attention":
                issues.append(f"unsupported_attention_layer:{index}:{kind}")
            kinds.append(kind)
            windows.append(window)
        return kinds, windows, tuple(dict.fromkeys(issues))

    kinds = ["full_attention"] * layers
    windows = [0] * layers
    if bool(c.get("use_sliding_window", False)):
        declared = c.get("sliding_window")
        if declared is None:
            issues.append("sliding_window_enabled_without_window")
            kinds = ["unresolved_window_schedule"] * layers
        else:
            window = int(declared)
            if window <= 0:
                issues.append(f"sliding_window_invalid:{window}")
                kinds = ["unresolved_window_schedule"] * layers
            elif c.get("max_window_layers") is None:
                issues.append("sliding_window_max_window_layers_missing")
                kinds = ["unresolved_window_schedule"] * layers
            else:
                max_layers = int(c.get("max_window_layers"))
                for index in range(layers):
                    if index >= max_layers:
                        windows[index] = window
    return kinds, windows, tuple(dict.fromkeys(issues))


def _standard_qwen_plans(c: Mapping[str, Any]) -> list[LayerPlan]:
    """Plans for ordinary Qwen2/Qwen2.5 dense, Qwen2-MoE and Qwen3 dense/MoE."""
    family = _standard_qwen_family(c)
    hidden = int(_require(c, "hidden_size"))
    layers = int(_require(c, "num_hidden_layers"))
    heads = _int(c, "num_attention_heads")
    kv_heads = _int(c, "num_key_value_heads", heads)
    head_dim = _standard_qwen_head_dim(c, family, hidden, heads) if heads else 0
    ffn, _ = _standard_qwen_ffn_config(c, family)
    kinds, windows, _ = _standard_qwen_window(c, layers)
    qk_norm = family in {"qwen3", "qwen3_moe"}
    activation = str(c.get("hidden_act", "silu"))
    plans: list[LayerPlan] = []
    for index in range(layers):
        moe = _standard_qwen_layer_moe(index, ffn)
        if moe:
            experts = int(ffn["experts"])
            top_k = int(ffn["top_k"])
            moe_i = int(ffn["moe_i"])
            if top_k <= 0 or top_k > experts or moe_i <= 0:
                raise ModelingError(f"model_config_qwen_moe_dims_invalid:{index}")
            ffn_fields: dict[str, Any] = {
                "ffn": "moe", "intermediate_size": 0,
                "num_experts": experts, "top_k": top_k,
                "moe_intermediate_size": moe_i,
                "num_shared_experts": int(ffn["shared_count"]),
                "shared_intermediate_size": int(ffn["shared_i"]),
                "shared_expert_gate": bool(ffn["shared_gate"]),
                "router_input_size": hidden, "router_mode": "softmax",
                # The gate Linear keeps model precision. Only the subsequent
                # softmax promotes scores to FP32 in the reference forward.
                "router_dtype": "",
                "router_norm_topk": bool(c.get("norm_topk_prob", False)),
            }
        else:
            dense_i = int(ffn["dense_i"])
            if dense_i <= 0:
                raise ModelingError(f"model_config_qwen_dense_intermediate_missing:{index}")
            ffn_fields = {
                "ffn": "dense", "intermediate_size": dense_i,
                "num_experts": 0, "top_k": 0, "moe_intermediate_size": 0,
                "num_shared_experts": 0, "shared_intermediate_size": 0,
                "shared_expert_gate": False, "router_input_size": 0,
                "router_mode": "softmax", "router_norm_topk": False,
            }
        known_core = (kinds[index] == "full_attention"
                      or kinds[index] == "sliding_attention" and windows[index] > 0)
        if not known_core:
            projections_known = heads > 0 and kv_heads > 0 and head_dim > 0
            plans.append(LayerPlan(
                index=index,
                attention="projection_only" if projections_known else "unmodeled",
                architecture=kinds[index],
                num_heads=heads if projections_known else 0,
                num_kv_heads=kv_heads if projections_known else 0,
                head_dim=head_dim if projections_known else 0,
                qk_norm=False, window=0, cache_kind="none",
                ffn_activation=activation,
                native_layer_id=index, scope="body", **ffn_fields,
            ))
            continue
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads:
            raise ModelingError(
                f"model_config_qwen_gqa_group_not_integral:{heads}%{kv_heads}")
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            qk_norm=qk_norm, window=windows[index],
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index, ffn_activation=activation,
            native_layer_id=index, scope="body", **ffn_fields,
        ))
    return plans


def _aux_standard_qwen(c: Mapping[str, Any]) -> AuxiliarySpec:
    """A layer count alone does not specify a standard-Qwen draft architecture."""
    return AuxiliarySpec()


def _standard_qwen_gaps(c: Mapping[str, Any]) -> tuple[str, ...]:
    """Visible unmodeled standard-Qwen fields; never a timing correction."""
    family = _standard_qwen_family(c)
    hidden = int(_require(c, "hidden_size"))
    heads = _int(c, "num_attention_heads")
    layers = int(_require(c, "num_hidden_layers"))
    issues = list(_standard_qwen_window(c, layers)[2])
    issues.extend(_standard_qwen_ffn_config(c, family)[1])
    if family == "qwen2":
        issues.append("qkv_bias_unmodeled")
    elif family == "qwen2_moe":
        if bool(c.get("qkv_bias", True)):
            issues.append("qkv_bias_unmodeled")
        raw = c.get("head_dim")
        if raw is not None:
            try:
                declared = int(raw)
            except (TypeError, ValueError):
                issues.append("head_dim_invalid")
            else:
                derived = hidden // heads if heads and hidden % heads == 0 else 0
                if declared != derived:
                    issues.append(
                        f"head_dim_ignored_by_qwen2_moe_source:{declared}!={derived}")
    elif bool(c.get("attention_bias", False)):
        # Qwen3 applies attention_bias to Q/K/V and the output projection.
        issues.append("attention_bias_unmodeled")
    activation = str(c.get("hidden_act", "silu"))
    if activation != "silu":
        issues.append(f"hidden_act_unsupported:{activation}")
    if bool(c.get("attn_output_gate", False)):
        issues.append("q_output_gate_unsupported_for_standard_qwen")
    if _int(c, "num_nextn_predict_layers") > 0 or _int(c, "mtp_num_hidden_layers") > 0:
        issues.append("optional_mtp_schema_not_modeled")
    return tuple(dict.fromkeys(issues))
