"""Catalog loader for the nine se-tools model definitions.

[Role]
  Load the checked-in actual configs and expose independent identities for all
  nine models.  Family adapters in ``a3_catalog_specs`` normalize text/vision
  nesting, exact body schedules, optional draft/MTP, quantization, cache/state
  and vision/Engram records.

[Invariants]
  * Constants come from config JSON or explicit caller inputs; no model-name
    latency constants and no source-SHA admission gate.
  * ``model.layers`` equals the body-layer count; optional draft/MTP never
    becomes a body layer and never runs on a plain forward.
  * DeepSeek V4 Pro and archived V4 10T remain distinct identities while
    sharing the same semantics; the 10T name never invents dimensions.
  * Missing required dimensions raise ``A3InputError``; zero placeholders are
    not returned for weights.
  * Weight, cache and state numbers are declared/semantic geometry, not
    measured traffic.  Resident weights, optional draft weights and node-
    sharded Engram tables remain separate accounting planes.

[Timeouts & Locks]
  Reads local JSON files only; no network, NPU, subprocess or locks.

[Side-Effect Policy]
  No filesystem writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .a3_model_types import LayerPlan, ModelSpec
from .a3_types import A3InputError, ParallelSpec
from .a3_catalog_specs import build_model_spec as _build_model_spec
from .a3_catalog_weights import (
    body_parameter_count, body_weight_bytes, sharded_weight_bytes as _sharded,
    stage_weight_bytes, stage_weight_terms, weight_breakdown,
)

CONFIG_DIR = Path(__file__).with_name("model_configs")
CATALOG: dict[str, str] = {
    "deepseek-v4-10t": "deepseek-v4-10t.json",
    "deepseek-v4-flash-0731": "deepseek-v4-flash-0731.json",
    "deepseek-v4-pro-0813": "deepseek-v4-pro-0813.json",
    "deepseek-v4.1-flash": "deepseek-v4.1-flash.json",
    "glm-5.3": "glm-5.3.json",
    "kimi-k3": "kimi-k3.json",
    "kimi-k3-dspark": "kimi-k3-dspark.json",
    "qwen3.8-2.4t-a95b": "qwen3.8-2.4t-a95b.json",
    "qwen3.8-27b": "qwen3.8-27b.json",
}

MODEL_ALIASES = {
    "DeepSeek-V4-10T": "deepseek-v4-10t",
    "DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731",
    "DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813",
    "DeepSeek-V4.1-Flash": "deepseek-v4.1-flash",
    "GLM-5.3": "glm-5.3",
    "Kimi-K3": "kimi-k3",
    "Kimi-K3-DSpark": "kimi-k3-dspark",
    "Qwen3.8-2.4T-A95B": "qwen3.8-2.4t-a95b",
    "Qwen3.8-27B": "qwen3.8-27b",
}


def build_model_spec(key: str, config: dict[str, Any]) -> ModelSpec:
    return _build_model_spec(key, config)


def load_model(key: str, *, config_root: str | Path | None = None) -> ModelSpec:
    key = MODEL_ALIASES.get(key, key)
    if key not in CATALOG:
        raise A3InputError(f"model_catalog_unknown:{key}")
    root = Path(config_root) if config_root is not None else CONFIG_DIR
    path = root / CATALOG[key]
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise A3InputError(f"model_config_unreadable:{key}:{type(exc).__name__}") from exc
    return _build_model_spec(key, config)


def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
) -> float:
    """Largest actual PP-stage per-rank weight footprint."""
    parallel.validate(model)
    if model.quant is None:
        # Generic (non-catalog) ModelSpec fallback.
        wb = 1.0
        per_stage = model.weight_bytes_per_layer(parallel) * model.layers_per_stage(parallel.pp)
        return float(model.vocab_size * model.hidden_size * wb / max(1, parallel.tp) + per_stage)
    return _sharded(
        model, parallel, node_ranks, include_draft=include_draft,
        include_engram=include_engram,
    )


def model_weight_detail(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1,
) -> dict[str, Any]:
    stages = stage_weight_bytes(model, parallel, node_ranks)
    stages_draft = stage_weight_bytes(
        model, parallel, node_ranks, include_draft=True,
    )
    return {
        "stage_bytes": stages,
        "stage_bytes_with_optional_draft": stages_draft,
        "largest_stage_bytes": max(stages) if stages else 0.0,
        "largest_stage_with_draft_bytes": max(stages_draft) if stages_draft else 0.0,
        "largest_stage": max(range(len(stages)), key=lambda i: stages[i]) if stages else 0,
        "category_bytes": weight_breakdown(model, parallel, node_ranks),
        "logical_body_parameters": body_parameter_count(model),
        "logical_body_weight_bytes": body_weight_bytes(model),
        "node_ranks": node_ranks,
    }


def list_models() -> list[dict[str, Any]]:
    rows = []
    for key in CATALOG:
        model = load_model(key)
        facts = dict(model.catalog_facts)
        rows.append({
            "key": key, "name": model.name, "layers": model.layers,
            "body_layers": model.body_layers,
            "hidden_size": model.hidden_size, "heads": model.num_heads,
            "experts": model.num_experts, "top_k": model.top_k,
            "parameter_count": model.parameter_count(),
            "resident_weight_bytes": model.resident_weight_bytes_override,
            "attention_types": facts.get("attention_types", {}),
            "ffn_types": facts.get("ffn_types", {}),
            "has_optional_draft": model.has_optional_draft,
            "has_vision": model.vision is not None,
            "engram_tables": facts.get("engram_tables", 0),
            "facts": facts,
        })
    return rows


__all__ = [
    "CATALOG", "MODEL_ALIASES", "CONFIG_DIR", "LayerPlan", "load_model",
    "list_models", "sharded_weight_bytes", "model_weight_detail",
    "stage_weight_terms", "body_weight_bytes", "body_parameter_count",
]
