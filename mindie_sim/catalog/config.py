"""Shared catalog config parsing helpers.

[Role]
  Flatten text/vision nesting and validate required integer/dimension fields for
  family adapters.

[Invariants]
  * Required fields raise ModelingError; they never default to zero.
  * Architecture selection prefers explicit architectures over generic model_type.
  * Config provenance is not an admission gate.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError


def _require(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise ModelingError(f"model_config_missing:{key}")
    return config[key]


def _int(config: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = config.get(key, default)
    if value is None:
        return int(default)
    return int(value)


def flatten_config(config: Mapping[str, Any]) -> tuple[dict[str, Any], Mapping[str, Any] | None, Mapping[str, Any] | None]:
    text = config.get("text_config")
    vision = config.get("vision_config")
    merged = {**config, **text} if isinstance(text, Mapping) else dict(config)
    return merged, text if isinstance(text, Mapping) else None, vision if isinstance(vision, Mapping) else None


_STRUCTURAL_SANDWICH_SLIDING_KEYS = frozenset({
    "global_head_dim", "attention_k_eq_v", "enable_moe_block",
    "num_kv_shared_layers", "hidden_size_per_layer_input",
    "use_double_wide_mlp", "num_global_key_value_heads",
    "top_k_experts", "vocab_size_per_layer_input",
})
_STRUCTURAL_REGION_SPARSE_KEYS = frozenset({
    "sparse_config", "use_head_wise_attn_gate", "swiglu_limits",
    "swiglu_limits_shared", "moe_layer_list", "moe_num_experts",
    "num_attention_groups", "need_fp32_gate", "use_moe_router_bias",
    "moe_router_scaling_factor", "share_expert_dim",
    "head_dim_full_attention", "head_dim_sliding_attention",
})


def _profile(config: Mapping[str, Any], merged: Mapping[str, Any]) -> str:
    archs = " ".join(str(v) for v in config.get("architectures", ()) or ())
    model_type = str(merged.get("model_type", config.get("model_type", ""))).lower()
    if "DSparkDraftModel" in archs or "K3DSpark" in archs or model_type == "k3_dspark":
        return "kimi_dspark"
    if model_type in {"gemma4", "gemma4_text"} or "Gemma4" in archs:
        return "gemma4"
    if (model_type == "minimax_m3_vl" or "MiniMaxM3" in archs
            or merged.get("family") == "minimax_m3"):
        return "minimax_m3"
    if model_type in {"step5", "step5_vl"} or "Step5" in archs:
        return "step5"
    if "num_layers" in merged and "num_hidden_layers" not in merged:
        return "generic"
    if "deepseek_v41" in model_type or "DeepseekV41" in archs:
        return "deepseek_v41"
    if "deepseek_v4" in model_type or "DeepseekV4" in archs:
        return "deepseek_v4"
    if "glm_moe_dsa" in model_type or "glm5_next" in model_type or "Glm5Next" in archs:
        # Both GLM releases share the GLM MLA/DSA lowering (full indexers own a
        # token-level index plane); the Flash release adds KDA layers, which
        # ``_glm_plans`` dispatches on its native ``layer_types`` schedule.
        return "glm"
    if "kimi" in model_type or "Kimi" in archs:
        return ("kimi" if merged.get("linear_attn_config") else
                "standard_kimi" if model_type in {"kimi_k2", "kimi_k25", "kimi_vl", "deepseek_v3"}
                else "generic")
    if "deepseek" in model_type or "Deepseek" in archs:
        return ("standard_deepseek" if model_type in {"deepseek_v2", "deepseek_v3", "deepseek_v32"}
                and merged.get("kv_lora_rank") else "generic")
    if "glm" in model_type or "Glm" in archs or "ChatGLM" in archs:
        return ("standard_glm" if model_type in {"glm4", "glm4_moe", "chatglm"}
                else "generic")
    if "qwen" in model_type or "Qwen" in archs:
        return ("qwen" if ("linear_attention" in (merged.get("layer_types") or ())
                           or _int(merged, "full_attention_interval") > 0)
                and merged.get("linear_num_key_heads") else
                "standard_qwen" if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
                else "generic")
    # Structural fallbacks when a renamed/unseen HF config carries unambiguous
    # native schema geometry without a recognized model_type or architecture name.
    if merged.get("layer_types"):
        if any(field in merged for field in _STRUCTURAL_SANDWICH_SLIDING_KEYS):
            return "gemma4"
        if any(field in merged for field in _STRUCTURAL_REGION_SPARSE_KEYS):
            return "step5"
    if (
        "sparse_attention_config" in merged
        or "num_mtp_modules" in merged
        or str(merged.get("hidden_act", "")).strip().lower() == "swigluoai"
    ):
        return "minimax_m3"
    return "generic"
