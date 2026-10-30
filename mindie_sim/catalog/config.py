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


def _profile(config: Mapping[str, Any], merged: Mapping[str, Any]) -> str:
    archs = " ".join(str(v) for v in config.get("architectures", ()) or ())
    model_type = str(merged.get("model_type", config.get("model_type", ""))).lower()
    if "DSparkDraftModel" in archs:
        return "kimi_dspark"
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
                "standard_kimi" if model_type in {"kimi_k2", "kimi_vl", "deepseek_v3"}
                else "generic")
    if "deepseek" in model_type or "Deepseek" in archs:
        return ("standard_deepseek" if model_type in {"deepseek_v2", "deepseek_v3"}
                and merged.get("kv_lora_rank") else "generic")
    if "glm" in model_type or "Glm" in archs or "ChatGLM" in archs:
        return ("standard_glm" if model_type in {"glm4", "glm4_moe", "chatglm"}
                else "generic")
    if "qwen" in model_type or "Qwen" in archs:
        return ("qwen" if "linear_attention" in (merged.get("layer_types") or ())
                and merged.get("linear_num_key_heads") else
                "standard_qwen" if model_type in {"qwen2", "qwen2_moe", "qwen3", "qwen3_moe"}
                else "generic")
    return "generic"
