"""Shared catalog config parsing helpers.

[Role]
  Flatten text/vision nesting and validate required integer/dimension fields for
  family adapters.

[Invariants]
  * Required fields raise A3InputError; they never default to zero.
  * Architecture selection prefers explicit architectures over generic model_type.
  * Config provenance is not an admission gate.

[Timeouts & Locks]
  Pure mapping reads.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_types import A3InputError


def _require(config: Mapping[str, Any], key: str) -> Any:
    if key not in config or config[key] is None:
        raise A3InputError(f"model_config_missing:{key}")
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
    if "deepseek_v41" in model_type or "DeepseekV41" in archs:
        return "deepseek_v41"
    if "deepseek_v4" in model_type or "DeepseekV4" in archs:
        return "deepseek_v4"
    if "glm_moe_dsa" in model_type:
        return "glm"
    if "kimi" in model_type or "Kimi" in archs:
        return "kimi"
    if "qwen3_5" in model_type or "Qwen3_5" in archs or "qwen3" in model_type:
        return "qwen"
    raise A3InputError(f"model_catalog_profile_unsupported:{model_type or archs}")
