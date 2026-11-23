"""Fleet model configs and native adapters for multinode/workstream families.

[Role]
  Register the seven fleet checkpoint/config identities and lower Gemma-4,
  MiniMax-M3/M3.1 and Step-5-Preview HF configs into the existing supported
  LayerPlan vocabulary (``gqa``/``gdn``/``mla`` with real windows).

[Invariants]
  * Every body layer is an exact supported attention core; no sparse layer is
    degraded to ``projection_only`` or ``unmodeled``.
  * Declared geometry, not checkpoint names, drives layers, experts, indexers,
    hyper-connections and Engram tables.
  * Optional next-token lists are sliced to body layers; MTP never runs on a
    plain forward. Unmodeled modalities stay explicit coverage gaps.

[Timeouts & Locks]
  Pure mapping arithmetic; no I/O, network, NPU or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation; returns immutable ModelSpec records.
"""
from __future__ import annotations

from typing import Any, Mapping

from mindie_sim.catalog.config import _int, _require, flatten_config
from mindie_sim.catalog.hf_config import native_coverage
from mindie_sim.catalog.quant import _quant_bf16
from mindie_sim.catalog.specs import _model_weight_totals, _set_facts
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError

KV_CACHE_PLANES = 2
STANDARD_RMSNORM_OFFSET = 0.0
NORM_WEIGHT_OFFSET = 1.0
GEMMA4_DENSE_LAYER_NORMS = 4
GEMMA4_MOE_LAYER_NORMS = 7
STEP5_NEXTN_BLOCKS = 3
STEP5_PP_CAPABILITY = (1, 2, 4, 8, 16)
DEFAULT_NORM_EPS = 1e-06
#: vLLM's ``swigluoai`` expression uses alpha=1.702, beta=1 unless the
#: checkpoint declares its own scalars (MiniMax M3 declares neither).
SWIGLU_OAI_ALPHA = 1.702
SWIGLU_OAI_BETA = 1.0
_STEP5_NEXTN_BLOCKS = STEP5_NEXTN_BLOCKS

FLEET_CATALOG_ENTRIES: dict[str, str] = {
    key: f"../catalog/configs/{key}.json" for key in (
        "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b",
        "gemma-4-31b", "qwen3.6-35b-a3b", "qwen3.8-flash-next",
        "step-5-preview",
    )
}

CORE_CATALOG_ENTRIES: dict[str, str] = {
    "deepseek-v4-10t": "deepseek-v4-10t.json", "deepseek-v4-flash-0731": "deepseek-v4-flash-0731.json",
    "deepseek-v4-pro-0813": "deepseek-v4-pro-0813.json", "deepseek-v4.1-flash": "deepseek-v4.1-flash.json",
    "deepseek-v2-lite": "deepseek-v2-lite.json", "deepseek-v3.2-prune-3l": "deepseek-v3.2-prune-3l.json",
    "deepseek-v3": "deepseek-v3.json", "deepseek-r1": "deepseek-r1.json", "deepseek-v3.2": "deepseek-v3.2.json",
    "glm-5.3": "glm-5.3.json", "glm-5.3-flash": "glm-5.3-flash.json",
    "glm-5.1": "glm-5.1.json", "glm-5.2": "glm-5.2.json",
    "kimi-k3": "kimi-k3.json", "kimi-k3.1": "kimi-k3.1.json", "kimi-k3-dspark": "kimi-k3-dspark.json",
    "kimi-k3-dspark-block5": "kimi-k3-dspark-block5.json",
    "kimi-k2": "kimi-k2.json", "kimi-k2-thinking": "kimi-k2-thinking.json", "kimi-k2.5": "kimi-k2.5.json",
    "kimi-k2.6": "kimi-k2.6.json",
    "qwen3-30b-a3b": "qwen3-30b-a3b.json", "qwen3-32b": "qwen3-32b.json",
    "qwen3-235b-a22b": "qwen3-235b-a22b.json", "qwen3-next-80b-a3b": "qwen3-next-80b-a3b.json",
    "qwen3.5-27b": "qwen3.5-27b.json", "qwen3.5-35b-a3b": "qwen3.5-35b-a3b.json",
    "qwen3.5-122b-a10b": "qwen3.5-122b-a10b.json",
    "qwen3.5-397b-a17b": "qwen3.5-397b-a17b.json", "qwen3.6-27b": "qwen3.6-27b.json",
    "qwen3.8-2.4t-a95b": "qwen3.8-2.4t-a95b.json", "qwen3.8-27b": "qwen3.8-27b.json",
    **FLEET_CATALOG_ENTRIES,
}

ARCHITECTURE_PRESET_CATALOG: dict[str, str] = {
    "longcat-flash-lite": "longcat-flash-lite.json",
    "minimax-m3": "../catalog/configs/minimax-m3.json",
    "bailing-v3": "bailing-v3.json",
    "qwen3.5-moe": "qwen3.5-moe.json",
    "hunyuan-video-dit": "hunyuan-video-dit.json",
}

ARCHITECTURE_PRESET_ALIASES: dict[str, str] = {
    "LongCat-Flash-Lite": "longcat-flash-lite", "longcat_flash_lite": "longcat-flash-lite",
    "MiniMax-M3": "minimax-m3", "MiniMax-M3-VL": "minimax-m3", "minimax_m3": "minimax-m3",
    "Bailing-V3": "bailing-v3", "Bailing-Moe-V3": "bailing-v3", "bailing_v3": "bailing-v3",
    "Qwen3.5-MoE": "qwen3.5-moe", "Qwen3.5-MoE-A35B": "qwen3.5-moe", "qwen3_5_moe": "qwen3.5-moe",
    "HunyuanVideo": "hunyuan-video-dit", "HunyuanVideo-DiT": "hunyuan-video-dit", "hunyuan_video": "hunyuan-video-dit",
}

FLEET_MODEL_ALIASES: dict[str, str] = {
    # MiniMax-M3 checkpoint and prompt spellings.
    "minimax_m3": "minimax-m3", "MiniMax-M3": "minimax-m3",
    "MiniMax-M3-VL": "minimax-m3", "MiniMax-M3-MXFP8": "minimax-m3",
    "minimax-m3-mxfp8": "minimax-m3", "minimax_m31_flash": "minimax-m3.1-flash",
    "minimax-m31-flash": "minimax-m3.1-flash", "minimax_m3_1_flash": "minimax-m3.1-flash",
    "MiniMax-M3.1-Flash": "minimax-m3.1-flash", "MiniMax-M3.1-Flash-BF16": "minimax-m3.1-flash",
    "Minimax-M3.1-Flash-w8a8-mxfp8": "minimax-m3.1-flash", "MiniMax-M3.1-Flash-w8a8-mxfp8": "minimax-m3.1-flash",
    "MiniMax-M3.1-Flash-W4A4-MXFP4-CeilX-IterSmooth": "minimax-m3.1-flash",
    "minimax-m31-flash-mxfp8": "minimax-m3.1-flash", "minimax_m3dot1_flash": "minimax-m3.1-flash",
    "minimax_m3dot1_flash_mxfp8": "minimax-m3.1-flash", "minimax_m3dot1_mxfp8": "minimax-m3.1-flash",
    # Gemma-4 26B-A4B and 31B.
    "gemma4_26b_a4b": "gemma-4-26b-a4b", "gemma4-26b-a4b": "gemma-4-26b-a4b",
    "Gemma-4-26B-A4B": "gemma-4-26b-a4b", "gemma-4-26B-A4B-it": "gemma-4-26b-a4b",
    "gemma-4-26B-A4B-it-w4a4-mxfp4": "gemma-4-26b-a4b", "gemma4-26b-a4b-mxfp4": "gemma-4-26b-a4b",
    "gemma4_31b": "gemma-4-31b", "gemma4-31b": "gemma-4-31b",
    "Gemma-4-31B": "gemma-4-31b", "gemma-4-31B-it": "gemma-4-31b",
    "gemma-4-31B-it-w4a4-mxfp4": "gemma-4-31b", "gemma4-31b-mxfp4": "gemma-4-31b",
    # Qwen3.6-35B-A3B hybrid and Qwen3.8-Flash-Next (Qwen4Exp).
    "qwen36_35b_a3b": "qwen3.6-35b-a3b", "qwen3_6_35b_a3b": "qwen3.6-35b-a3b",
    "qwen36-35b-a3b": "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B": "qwen3.6-35b-a3b",
    "Qwen3.6-35B-A3B-w8a8-mxfp8": "qwen3.6-35b-a3b", "Qwen3.6-35B-A3B-w8a8c8-mxfp8": "qwen3.6-35b-a3b",
    "Qwen3.6-35B-A3B-w4a8-mxfp": "qwen3.6-35b-a3b", "qwen36-35b-mxfp8": "qwen3.6-35b-a3b",
    "qwen36-35b-w8a8-mxfp8": "qwen3.6-35b-a3b", "qwen36-35b-w4a8-mxfp": "qwen3.6-35b-a3b",
    "qwen38_flash_next": "qwen3.8-flash-next", "qwen3_8_flash_next": "qwen3.8-flash-next",
    "qwen38-flash-next": "qwen3.8-flash-next", "Qwen3.8-Flash-Next": "qwen3.8-flash-next",
    "Qwen3.8-Flash-Next-w8a8-mxfp8": "qwen3.8-flash-next", "Qwen3.8-Flash-Next-w4a4-mxfp4": "qwen3.8-flash-next",
    "Qwen3.8-Flash-Next-w8a8-mtp": "qwen3.8-flash-next", "Qwen3.8-Flash-Next-w4a8": "qwen3.8-flash-next",
    "Qwen3.8-Flash-Next-w4a8-mxfp": "qwen3.8-flash-next", "qwen3.8-flash-next-w8a8-mtp": "qwen3.8-flash-next",
    "qwen3.8-flash-next-w4a8": "qwen3.8-flash-next", "qwen3.8-flash-next-w4a8-mxfp": "qwen3.8-flash-next",
    "qwen38-flash-next-mxfp8": "qwen3.8-flash-next", "qwen38-flash-next-mxfp4": "qwen3.8-flash-next",
    # Step-5-Preview.
    "step5_preview": "step-5-preview", "step5-preview": "step-5-preview",
    "Step-5-Preview": "step-5-preview", "Step-5-Preview-BF16": "step-5-preview",
    "Step-5-Preview-w8a8-mxfp8": "step-5-preview", "step5-preview-mxfp8": "step-5-preview",
    # Existing multinode checkpoint spellings.
    "Qwen3-30B-A3B-MXFP8": "qwen3-30b-a3b", "qwen3-30b-a3b-mxfp8": "qwen3-30b-a3b",
    "Qwen3.6-27B-w8a8-mxfp8": "qwen3.6-27b", "qwen36-27b-mxfp8": "qwen3.6-27b",
    "Qwen3.8-27B-w8a8-mxfp8": "qwen3.8-27b", "Qwen3.8-27B-w8a8-mxfp8-mtp": "qwen3.8-27b",
    "qwen38-27b-bf16": "qwen3.8-27b", "DeepSeek-V4.1-Flash-1017": "deepseek-v4.1-flash",
    "GLM-5.1-w4a4c8-mxfp4": "glm-5.1", "GLM-5.2-w4a4c8-mxfp4": "glm-5.2",
    "GLM-5.3-W8A8C8-mxfp8": "glm-5.3", "GLM-5.3-w4a4c8-mxfp4": "glm-5.3",
    "GLM-5.3-Flash-w8a8-mxfp8": "glm-5.3-flash", "GLM-5.3-Flash-w4a8-mxfp4": "glm-5.3-flash",
    "Qwen3.5-397B-A17B-w8a8-mxfp8": "qwen3.5-397b-a17b", "Qwen3.5-397B-A17B-w4a4-mxfp4": "qwen3.5-397b-a17b",
}

CORE_MODEL_ALIASES: dict[str, str] = {
    "deepseek-v4-flash": "deepseek-v4-flash-0731", "deepseek-v4-pro": "deepseek-v4-pro-0813",
    "deepseek-v4.1": "deepseek-v4.1-flash", "qwen3.8": "qwen3.8-27b",
    "DeepSeek-V4-10T": "deepseek-v4-10t", "DeepSeek-V4-Flash-0731": "deepseek-v4-flash-0731",
    "DeepSeek-V4-Pro-0813": "deepseek-v4-pro-0813", "DeepSeek-V4.1-Flash": "deepseek-v4.1-flash",
    "DeepSeek-V2-Lite": "deepseek-v2-lite", "dsv2lite": "deepseek-v2-lite",
    "DeepSeek-V3.2-Prune-3L": "deepseek-v3.2-prune-3l", "dsv32prune_L3": "deepseek-v3.2-prune-3l",
    "DeepSeek-V3": "deepseek-v3", "DeepSeek-V3-0324": "deepseek-v3", "deepseek-v3-0324": "deepseek-v3", "dsv3": "deepseek-v3",
    "DeepSeek-R1": "deepseek-r1", "DeepSeek-R1-0528": "deepseek-r1", "deepseek-r1-0528": "deepseek-r1", "dsr1": "deepseek-r1",
    "DeepSeek-V3.2": "deepseek-v3.2", "DeepSeek-V3.2-Exp": "deepseek-v3.2", "deepseek-v3.2-exp": "deepseek-v3.2", "dsv3.2": "deepseek-v3.2",
    "GLM-5.3": "glm-5.3", "glm5.3": "glm-5.3", "GLM5.3": "glm-5.3", "GLM-5": "glm-5.3", "glm-5": "glm-5.3",
    "GLM-5.3-Flash": "glm-5.3-flash", "glm5.3-flash": "glm-5.3-flash", "glm53-flash": "glm-5.3-flash",
    "GLM-5.3-W4A8": "glm-5.3", "GLM-5.3-W4A8-INT": "glm-5.3", "glm-5.3-w4a8": "glm-5.3",
    "GLM-5.1": "glm-5.1", "glm5.1": "glm-5.1", "GLM5.1": "glm-5.1", "glm51": "glm-5.1",
    "GLM-5.2": "glm-5.2", "glm5.2": "glm-5.2", "GLM5.2": "glm-5.2", "glm52": "glm-5.2",
    "Kimi-K3": "kimi-k3", "Kimi-K3.1": "kimi-k3.1", "kimi_k3_1": "kimi-k3.1", "kimi_k31": "kimi-k3.1",
    "kimi-k31": "kimi-k3.1", "Kimi-K3-DSpark": "kimi-k3-dspark", "Kimi-K3-DSpark-Block5": "kimi-k3-dspark-block5",
    "Kimi-K2": "kimi-k2", "Kimi-K2-Instruct": "kimi-k2", "kimi-k2-instruct": "kimi-k2", "kimi_k2": "kimi-k2",
    "Kimi-K2-Thinking": "kimi-k2-thinking", "kimi-k2-thinking": "kimi-k2-thinking", "kimi_k2_thinking": "kimi-k2-thinking",
    "Kimi-K2.5": "kimi-k2.5", "kimi_k2.5": "kimi-k2.5", "kimi_k25": "kimi-k2.5", "kimi-k25": "kimi-k2.5",
    "Kimi-K2.6": "kimi-k2.6", "kimi_k2.6": "kimi-k2.6", "kimi_k26": "kimi-k2.6", "kimi-k26": "kimi-k2.6",
    "Kimi-K2.6-w4a8": "kimi-k2.6", "Kimi-K2.6_w4a8": "kimi-k2.6",
    "kimi-k2.6-w4a8": "kimi-k2.6", "kimi-k2.6_w4a8": "kimi-k2.6",
    "Qwen3-30B-A3B": "qwen3-30b-a3b", "qwen30ba3b": "qwen3-30b-a3b",
    "Qwen3-32B": "qwen3-32b", "qwen3_32b": "qwen3-32b",
    "Qwen3-235B-A22B": "qwen3-235b-a22b", "qwen3_235b_a22b": "qwen3-235b-a22b", "qwen3-235b": "qwen3-235b-a22b",
    "Qwen3-Next-80B-A3B": "qwen3-next-80b-a3b", "Qwen3-Next-80B-A3B-Instruct": "qwen3-next-80b-a3b",
    "qwen3-next-80b-a3b-instruct": "qwen3-next-80b-a3b", "qwen3_next_80b_a3b": "qwen3-next-80b-a3b", "qwen3-next-80b": "qwen3-next-80b-a3b",
    "Qwen3.5-27B": "qwen3.5-27b", "qwen35_27b": "qwen3.5-27b",
    "Qwen3.8-2.4T-A95B": "qwen3.8-2.4t-a95b", "Qwen3.8-27B": "qwen3.8-27b",
    "Qwen3.6-27B": "qwen3.6-27b", "qwen3.6-27b": "qwen3.6-27b",
    "Qwen3.5-35B-A3B": "qwen3.5-35b-a3b", "qwen3.5-35b-a3b": "qwen3.5-35b-a3b", "qwen3_5_moe_a3b": "qwen3.5-35b-a3b",
    "Qwen3.5-397B-A17B": "qwen3.5-397b-a17b", "qwen3_5_397b_a17b": "qwen3.5-397b-a17b", "qwen3.5-397b": "qwen3.5-397b-a17b",
    "Qwen3.5-122B-A10B": "qwen3.5-122b-a10b", "qwen3_5_122b_a10b": "qwen3.5-122b-a10b",
    "qwen35_122b_a10b": "qwen3.5-122b-a10b", "qwen35-122b-a10b": "qwen3.5-122b-a10b",
    "qwen3.5-122b": "qwen3.5-122b-a10b", "Qwen3.5-122B-A10B-w8a8-mtp": "qwen3.5-122b-a10b",
    "qwen3.5-122b-a10b-w8a8-mtp": "qwen3.5-122b-a10b",
    **FLEET_MODEL_ALIASES,
}


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


def _facts(family: str, config: Mapping[str, Any], c: Mapping[str, Any],
           plans: list[LayerPlan], **extra: Any) -> list[tuple[str, Any]]:
    facts = [
        ("profile", family), ("model_type", str(c.get("model_type", ""))),
        ("architectures", tuple(str(v) for v in config.get("architectures", ()) or ())),
        ("body_layers", len(plans)),
        ("norm_eps", float(c.get("rms_norm_eps", 1e-6) or 1e-6)),
        ("attention_types", _counts(p.architecture for p in plans)),
        ("ffn_types", _counts(p.ffn for p in plans)),
        ("mla_layers", sum(p.attention == "mla" for p in plans)),
        ("gqa_layers", sum(p.attention == "gqa" for p in plans)),
        ("gdn_layers", sum(p.attention == "gdn" for p in plans)),
        ("kda_layers", sum(p.attention == "kda" for p in plans)),
        ("dense_layers", sum(p.ffn == "dense" for p in plans)),
        ("moe_layers", sum(p.ffn == "moe" for p in plans)),
        ("indexer_layers", sum(p.indexer for p in plans)),
        ("indexer_full_layers", sum(p.indexer and p.indexer_type == "full" for p in plans)),
        ("hyper_connections", any(p.hyper_connections for p in plans)),
        ("hc_mult", max((p.hc_mult for p in plans), default=0)),
        ("sliding_window", max((p.window for p in plans), default=0)),
        ("num_nextn_predict_layers", _int(c, "num_nextn_predict_layers")),
        ("source_scope", "structural_config_no_latency_labels"),
    ]
    facts.append(("hc_lowrank", max((p.hc_lowrank for p in plans), default=0)))
    if bool(c.get("fp32_residual_connection", False)):
        facts.append(("fp32_residual_connection", True))
    if str(c.get("norm_dtype", "")).lower() in {"float32", "fp32"}:
        facts.append(("norm_dtype", "fp32"))
    facts.extend(extra.items())
    return facts


def _finish(model: ModelSpec, facts: list[tuple[str, Any]]) -> ModelSpec:
    from mindie_sim.catalog.weights import cache_row_bytes_by_layer
    _set_facts(model, facts)
    params, weights, draft_params, draft_weights = _model_weight_totals(model)
    _set_facts(model, list(model.catalog_facts) + [
        ("parameter_count_body", params), ("resident_weight_bytes_body", weights),
        ("parameter_count_optional_draft", draft_params),
        ("optional_draft_weight_bytes", draft_weights),
        ("cache_row_bytes_by_layer", cache_row_bytes_by_layer(model)),
    ])
    return model


def _model(key: str, family: str, config: Mapping[str, Any], c: Mapping[str, Any],
           plans: list[LayerPlan], *, attention: str, ffn: str, heads: int,
           kv_heads: int, head_dim: int, hidden: int, intermediate: int,
           vocab: int, experts: int = 0, top_k: int = 0, moe_i: int = 0,
           shared: int = 0, facts: list[tuple[str, Any]] | None = None,
           gaps: tuple[str, ...] = (), vision: Any = None,
           auxiliary: Any = None) -> ModelSpec:
    quant = _quant_bf16(c, name=f"bf16_{family}")
    metadata = native_coverage(config, list(gaps), plans=plans, quant=quant)
    model_facts = facts if facts is not None else _facts(family, config, c, plans)
    model = ModelSpec(
        name=key, layers=len(plans), body_layers=len(plans),
        hidden_size=hidden, num_heads=heads, num_kv_heads=kv_heads,
        head_dim=head_dim, intermediate_size=intermediate, vocab_size=vocab,
        attention=attention, ffn=ffn, activation_dtype=quant.activation_format,
        weight_dtype=quant.weight_format, kv_dtype=quant.kv_dtype,
        compute_dtype=quant.native_format, quantization=quant.name,
        norm_eps=float(c.get("rms_norm_eps", DEFAULT_NORM_EPS) or DEFAULT_NORM_EPS),
        num_experts=experts, top_k=top_k, moe_intermediate_size=moe_i,
        num_shared_experts=shared, quant=quant, vision=vision,
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        layer_plans=tuple(plans), auxiliary=auxiliary,
        state_bytes_per_request=float(sum(p.state_bytes_total for p in plans)),
        catalog_facts=tuple(sorted(model_facts)), metadata=metadata,
    )
    return _finish(model, model_facts)


def _gemma4(key: str, config: Mapping[str, Any]) -> ModelSpec:
    c, _, vision_raw = flatten_config(config)
    layers = int(_require(c, "num_hidden_layers")); hidden = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads")); vocab = int(_require(c, "vocab_size"))
    head_dim = _int(c, "head_dim") or hidden // heads
    global_head_dim = _int(c, "global_head_dim") or 512
    kv_heads = _int(c, "num_key_value_heads") or heads
    global_kv_heads = _int(c, "num_global_key_value_heads") or kv_heads
    intermediate = (_int(c, "intermediate_size") or _int(c, "shared_expert_intermediate_size")
                    or _int(c, "moe_intermediate_size") or 1)
    kinds = [str(v) for v in c.get("layer_types") or ()]
    if len(kinds) != layers or any(k not in {"sliding_attention", "full_attention"} for k in kinds):
        raise ModelingError("model_config_gemma4_layer_types_invalid")
    k_eq_v = bool(c.get("attention_k_eq_v", False))
    k_layers = tuple(i for i, kind in enumerate(kinds) if kind == "full_attention")
    schedule = tuple("global" if kind == "full_attention" else "local" for kind in kinds)
    window = _int(c, "sliding_window") or 1024
    ple = bool(_int(c, "hidden_size_per_layer_input"))
    ple_width = _int(c, "hidden_size_per_layer_input") or _int(c, "ple_width") or hidden
    # Real Gemma4 configs declare ``enable_moe_block`` + ``num_experts``; older
    # HF-shaped callers (profiler <= 0.5.11) declare only a routed-expert count.
    declared_experts = (_int(c, "num_experts") or _int(c, "n_routed_experts")
                        or _int(c, "num_routed_experts") or _int(c, "num_local_experts"))
    moe_flag = c.get("enable_moe_block")
    moe = bool(moe_flag) if moe_flag is not None else bool(declared_experts)
    experts = declared_experts if moe else 0
    top_k = (_int(c, "top_k_experts") or _int(c, "num_experts_per_tok")) if moe else 0
    moe_i = _int(c, "moe_intermediate_size") if moe else 0
    shared = (_int(c, "n_shared_experts") or _int(c, "num_shared_experts") or 1) if moe else 0
    shared_i = (_int(c, "shared_expert_intermediate_size") or intermediate) if moe else 0
    kv_shared = max(0, _int(c, "num_kv_shared_layers"))
    first_shared = max(0, layers - kv_shared)
    donor_by_kind: dict[str, int] = {}
    for donor_idx in range(first_shared):
        donor_by_kind[kinds[donor_idx]] = donor_idx
    from mindie_sim.catalog.hf_config import rotary_dimensions
    rope_params = c.get("rope_parameters") if isinstance(c.get("rope_parameters"), Mapping) else {}
    plans = []
    for index, kind in enumerate(kinds):
        full = kind == "full_attention"
        layer_kv = global_kv_heads if full else kv_heads
        layer_hd = global_head_dim if full else head_dim
        rope_cfg = rope_params.get(kind) if isinstance(rope_params.get(kind), Mapping) else {}
        partial_factor = float(rope_cfg.get("partial_rotary_factor", 1.0) or 1.0)
        rope_dim, nope_dim = rotary_dimensions(layer_hd, partial_factor)
        shares_kv = index >= first_shared and kind in donor_by_kind
        kv_owner = donor_by_kind[kind] if shares_kv else index
        # Two cache planes are retained on KV-owning layers even when V=v_norm(K).
        factor = KV_CACHE_PLANES
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gemma4",
            ffn="moe" if moe else "dense",
            window=0 if full else window,
            num_heads=heads, num_kv_heads=layer_kv, head_dim=layer_hd,
            qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
            intermediate_size=0 if moe else intermediate,
            num_experts=experts,
            top_k=top_k, moe_intermediate_size=moe_i,
            num_shared_experts=shared,
            shared_intermediate_size=shared_i, qk_norm=True,
            ffn_activation="gelu",
            kv_cache_elements=0 if shares_kv else factor * layer_kv * layer_hd,
            kv_owner_layer_id=kv_owner, owns_long_kv=not shares_kv,
            k_eq_v=bool(full and k_eq_v),
            native_layer_id=index, scope="body",
        ))
    norms_per_layer = GEMMA4_MOE_LAYER_NORMS if moe else GEMMA4_DENSE_LAYER_NORMS
    gemma_facts = {
        "gemma4": 1, "gemma4_ple": int(ple), "gemma4_ple_layers": (),
        "gemma4_ple_width": ple_width, "gemma4_ple_tables": 1,
        "gemma4_ple_gated": True,
        "gemma4_ple_vocab": _int(c, "vocab_size_per_layer_input") or vocab,
        "gemma4_k_eq_v_all": 0,
        "gemma4_k_eq_v_layers": k_layers,
        "gemma4_v_norm": 1,
        "gemma4_layer_scalar": 1,
        "gemma4_norms_per_layer": norms_per_layer,
        "gemma4_num_kv_shared_layers": kv_shared,
        "norm_weight_offset": STANDARD_RMSNORM_OFFSET,
        "normalization": {
            "norm_weight_offset": STANDARD_RMSNORM_OFFSET,
            "v_norm": True,
            "layer_scalar": True,
            "norms_per_layer": norms_per_layer,
        },
        "attention_schedule": schedule,
    }
    gaps = ("vision_encoder_and_projector:language_backbone_only",) if vision_raw else ()
    facts = _facts("gemma4", config, c, plans) + list(gemma_facts.items())
    return _model(
        key, "gemma4", config, c, plans, attention="gqa",
        ffn="moe" if moe else "dense", heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, hidden=hidden, intermediate=intermediate,
        vocab=vocab, experts=experts, top_k=top_k, moe_i=moe_i,
        shared=shared, facts=facts, gaps=gaps,
    )


def _minimax_m3(key: str, config: Mapping[str, Any]) -> ModelSpec:
    c, _, vision_raw = flatten_config(config)
    layers = int(_require(c, "num_hidden_layers")); hidden = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads")); kv_heads = int(_require(c, "num_key_value_heads"))
    head_dim = int(_require(c, "head_dim")); vocab = int(_require(c, "vocab_size"))
    experts = _int(c, "num_local_experts") or _int(c, "num_experts")
    top_k = _int(c, "num_experts_per_tok")
    dense_i = _int(c, "dense_intermediate_size") or _int(c, "intermediate_size")
    moe_i = _int(c, "moe_intermediate_size") or _int(c, "intermediate_size")
    shared = _int(c, "n_shared_experts") or _int(c, "num_shared_experts")
    shared_i = _int(c, "shared_intermediate_size") or moe_i
    raw_freq = c.get("moe_layer_freq")
    if isinstance(raw_freq, (list, tuple)):
        freq = [int(v) for v in raw_freq][:layers]; freq += [0] * (layers - len(freq))
        moe_flags = [bool(v) for v in freq]
    elif raw_freq is None:
        moe_flags = [True] * layers
    else:
        step = int(raw_freq)
        moe_flags = [step > 0 and index % step == 0 for index in range(layers)]
    sparse = c.get("sparse_attention_config")
    sparse = dict(sparse) if isinstance(sparse, Mapping) else {}
    sparse_on = bool(sparse.get("use_sparse_attention", False))
    sparse_freq = [int(v) for v in (sparse.get("sparse_attention_freq") or ())]
    start = _int(sparse, "sparse_layers_start", layers); block = _int(sparse, "sparse_block_size")
    topk_blocks = _int(sparse, "sparse_topk_blocks"); local_blocks = _int(sparse, "sparse_local_block")
    index_heads = _int(sparse, "sparse_num_index_heads"); index_dim = _int(sparse, "sparse_index_dim")
    qk_norm = bool(c.get("use_qk_norm", True))
    gate = bool(c.get("attention_output_gate", False))
    from mindie_sim.catalog.hf_config import declared_partial_rotary, rotary_dimensions
    rope_dim, nope_dim = rotary_dimensions(head_dim, declared_partial_rotary(c))
    activation = str(c.get("hidden_act", "silu"))
    if activation == "swigluoai":
        oai_limit = float(c.get("swiglu_limit") or 0)
        if oai_limit <= 0:
            raise ModelingError("model_config_minimax_swigluoai_limit_missing")
        oai_alpha = float(c.get("swiglu_alpha") or SWIGLU_OAI_ALPHA)
        oai_beta = float(c.get("swiglu_beta") or SWIGLU_OAI_BETA)
    else:
        oai_limit = oai_alpha = oai_beta = 0
    shared_limit = float(c.get("swiglu_limits_shared") or oai_limit)
    use_routing_bias = bool(c.get("use_routing_bias", False))
    routed_scale = float(c.get("routed_scaling_factor") or 1.0)
    plans = []
    for index in range(layers):
        moe = moe_flags[index]
        active = sparse_on and ((index < len(sparse_freq) and sparse_freq[index])
                                or index >= start)
        if active and min(block, topk_blocks, index_heads, index_dim) <= 0:
            raise ModelingError(f"model_config_minimax_sparse_dims_missing:{index}")
        noaux = bool(moe and use_routing_bias)
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            ffn="moe" if moe else "dense", num_heads=heads,
            num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=0 if moe else dense_i,
            num_experts=experts if moe else 0,
            top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=shared_i if moe else 0,
            router_input_size=hidden if moe else 0,
            router_mode=str(c.get("scoring_func", "softmax")),
            router_noaux_tc=noaux,
            router_groups=1 if noaux else 0,
            router_group_topk=1 if noaux else 0,
            routed_scaling_factor=routed_scale,
            qk_norm=qk_norm, q_output_gate=gate,
            qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
            ffn_activation=activation,
            swiglu_limit=oai_limit, swiglu_shared_limit=shared_limit,
            swiglu_alpha=oai_alpha, swiglu_beta=oai_beta,
            indexer=active, indexer_type="full" if active else "none",
            owns_index_k=active,
            index_k_owner_layer_id=index if active else (-1),
            owns_topk=active,
            topk_owner_layer_id=index if active else (-1),
            index_heads=index_heads if active else 0,
            index_dim=index_dim if active else 0,
            index_topk=topk_blocks * block if active else 0,
            candidate_block_size=block if active else 1,
            candidate_topk_blocks=topk_blocks if active else 0,
            kv_cache_elements=KV_CACHE_PLANES * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
            native_layer_id=index, scope="body",
        ))
    from mindie_sim.catalog.minimax_draft import build_minimax_auxiliary
    gemma_norm = bool(c.get("use_gemma_norm", False))
    offset = NORM_WEIGHT_OFFSET if gemma_norm else STANDARD_RMSNORM_OFFSET
    minimax_extra: dict[str, Any] = {
        "norm_weight_offset": offset,
        "num_mtp_modules": _int(c, "num_mtp_modules"),
    }
    if gemma_norm:
        minimax_extra["normalization"] = {
            "norm_weight_offset": NORM_WEIGHT_OFFSET,
            "norm_multiply_before_output_cast": True,
        }
    facts = _facts("minimax_m3", config, c, plans, **minimax_extra)
    auxiliary = build_minimax_auxiliary(c, body_layers=layers)
    gaps = ("vision_encoder_and_projector:language_backbone_only",) if vision_raw else ()
    return _model(
        key, "minimax_m3", config, c, plans, attention="gqa", ffn="moe",
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
        intermediate=dense_i, vocab=vocab, experts=experts, top_k=top_k,
        moe_i=moe_i, shared=shared, facts=facts, gaps=gaps,
        auxiliary=auxiliary,
    )


def _step5(key: str, config: Mapping[str, Any]) -> ModelSpec:
    c, _, vision_raw = flatten_config(config)
    layers = int(_require(c, "num_hidden_layers")); view = dict(c)
    for name in ("layer_types", "rope_theta", "partial_rotary_factors",
                 "swiglu_limits", "swiglu_limits_shared"):
        value = view.get(name)
        if isinstance(value, (list, tuple)) and len(value) > layers:
            view[name] = list(value)[:layers]
    kinds = [str(v) for v in view.get("layer_types") or ()]
    if len(kinds) != layers or any(k not in {"sliding_attention", "full_attention"} for k in kinds):
        raise ModelingError("model_config_step5_layer_types_invalid")
    hidden = int(_require(c, "hidden_size")); heads = int(_require(c, "num_attention_heads"))
    kv_heads = _int(c, "num_attention_groups") or _int(c, "num_key_value_heads") or heads
    head_dim = int(_require(c, "head_dim")); vocab = int(_require(c, "vocab_size"))
    window = _int(c, "sliding_window") or 512
    experts = (_int(c, "moe_num_experts") or _int(c, "n_routed_experts")
               or _int(c, "num_routed_experts") or _int(c, "num_experts"))
    top_k = _int(c, "moe_top_k") or _int(c, "num_experts_per_tok")
    moe_i = _int(c, "moe_intermediate_size")
    shared_i = _int(c, "share_expert_dim") or _int(c, "shared_expert_intermediate_size") or moe_i
    raw_moe = c.get("moe_layer_list")
    if isinstance(raw_moe, (list, tuple)):
        moe_set = {int(v) for v in raw_moe}
    else:
        enum = c.get("moe_layers_enum")
        moe_set = {int(v) for v in str(enum).split(",") if v.strip()} if enum is not None else set()
    if bool(c.get("use_moe", False)) and not moe_set:
        raise ModelingError("model_config_step5_moe_schedule_missing")
    if not moe_set and c.get("use_moe") is None and experts > 0:
        # Older HF-shaped callers (profiler <= 0.5.11) declare a routed-expert
        # count without a schedule: every layer is MoE, as before 0.9.11.
        moe_set = set(range(layers))
    if _int(c, "intermediate_size") or len(moe_set) < layers:
        intermediate = int(_require(c, "intermediate_size"))
    else:
        intermediate = shared_i or moe_i
    sparse = c.get("sparse_config")
    sparse_on = isinstance(sparse, Mapping) and bool(sparse.get("enabled", False))
    sparse = dict(sparse) if isinstance(sparse, Mapping) else {}
    apply_types = ({str(v) for v in (sparse.get("apply_to_layer_types") or ("full_attention",))}
                   if sparse_on else set())
    idx_heads = _int(sparse, "sparse_indexer_num_heads")
    idx_dim = _int(sparse, "proxy_dim")
    topk_regions = _int(sparse, "topk")
    block = max(1, _int(sparse, "region_block_size", 1))
    if sparse_on and min(idx_heads, idx_dim, topk_regions) <= 0:
        raise ModelingError("model_config_step5_sparse_dims_missing")
    gate = bool(c.get("use_head_wise_attn_gate", False)); plans = []
    from mindie_sim.catalog.hf_config import declared_partial_rotary, rotary_dimensions

    def step5_activation(index: int) -> tuple[int, int, float, float, float]:
        """Exact rotated width and clamped-SiLU scalars for body/draft layer."""
        factor = declared_partial_rotary(c, index)
        rope_dim, nope_dim = rotary_dimensions(head_dim, factor)
        limits = view.get("swiglu_limits")
        limit = float(limits[index]) if isinstance(limits, (list, tuple)) else 0.0
        shared_limits = view.get("swiglu_limits_shared")
        shared_limit = (float(shared_limits[index])
                        if isinstance(shared_limits, (list, tuple)) else limit)
        return rope_dim, nope_dim, limit, shared_limit, factor

    need_fp32_gate = bool(c.get("need_fp32_gate", False))
    use_router_bias = bool(c.get("use_moe_router_bias", False))
    routed_scale = float(c.get("moe_router_scaling_factor") or 1.0)
    for index, kind in enumerate(kinds):
        full = kind == "full_attention"; moe = index in moe_set
        sparse_layer = bool(sparse_on and kind in apply_types)
        rope_dim, nope_dim, limit, shared_limit, factor = step5_activation(index)
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            ffn="moe" if moe else "dense",
            window=0 if full else window,
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            qk_rope_head_dim=rope_dim, qk_nope_head_dim=nope_dim,
            intermediate_size=0 if moe else intermediate,
            num_experts=experts if moe else 0,
            top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=1 if moe else 0,
            shared_intermediate_size=shared_i if moe else 0,
            router_input_size=hidden if moe else 0,
            router_mode="sigmoid",
            router_dtype="fp32" if (moe and need_fp32_gate) else "",
            router_output_dtype="fp32" if (moe and need_fp32_gate) else "",
            router_noaux_tc=bool(moe and use_router_bias),
            routed_scaling_factor=routed_scale,
            q_output_gate_kind="headwise" if gate else "none",
            q_gate_projection="separate" if gate else "packed_q",
            ffn_activation="silu",
            swiglu_limit=limit, swiglu_shared_limit=shared_limit,
            indexer=sparse_layer,
            indexer_type="full" if sparse_layer else "none",
            owns_index_k=sparse_layer,
            index_k_owner_layer_id=index if sparse_layer else (-1),
            owns_topk=sparse_layer,
            topk_owner_layer_id=index if sparse_layer else (-1),
            index_heads=idx_heads if sparse_layer else 0,
            index_dim=idx_dim if sparse_layer else 0,
            index_topk=topk_regions * block if sparse_layer else 0,
            index_kpool=block if sparse_layer else 1,
            index_kpool_compress=bool(sparse_layer and block > 1),
            candidate_block_size=block if sparse_layer else 1,
            candidate_topk_blocks=topk_regions if sparse_layer else 0,
            kv_cache_elements=KV_CACHE_PLANES * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
            native_layer_id=index, scope="body",
        ))
    gaps = []
    if vision_raw:
        gaps.append("vision_encoder_and_projector:language_backbone_only")
    fp32_residual = bool(c.get("fp32_residual_connection", True))
    step5_facts = _facts(
        "step5", config, c, plans,
        supports_torch_compile=False,
        need_fp32_gate=need_fp32_gate,
        norm_weight_offset=NORM_WEIGHT_OFFSET,
        normalization={
            "norm_weight_offset": NORM_WEIGHT_OFFSET,
            "norm_multiply_before_output_cast": True,
            "fp32_residual": fp32_residual,
        },
    )
    auxiliary = _step5_auxiliary(
        c, plans, hidden=hidden, heads=heads, kv_heads=kv_heads,
        head_dim=head_dim, intermediate=intermediate, vocab=vocab,
        window=window, gate=gate)
    return _model(
        key, "step5", config, c, plans, attention="gqa", ffn="moe",
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
        intermediate=intermediate, vocab=vocab, experts=experts,
        top_k=top_k, moe_i=moe_i, shared=1, gaps=tuple(gaps), facts=step5_facts,
        auxiliary=auxiliary,
    )


def _step5_rotary_width(c, position: int, head_dim: int) -> int:
    """Exact rotated prefix of one body/NextN layer from its declared factor."""
    from mindie_sim.catalog.hf_config import declared_partial_rotary, rotary_dimensions
    return rotary_dimensions(head_dim, declared_partial_rotary(c, position))[0]


def _step5_draft_scalar(c, name: str, base: int, index: int) -> float:
    """Per-layer NextN scalar from the untruncated body+draft schedule."""
    values = c.get(name)
    position = int(base) + int(index)
    if isinstance(values, (list, tuple)) and position < len(values):
        return float(values[position])
    return 0.0


def _step5_auxiliary(c, plans, *, hidden: int, heads: int, kv_heads: int,
                     head_dim: int, intermediate: int, vocab: int,
                     window: int, gate: bool):
    """Declare the native Step-5 NextN blocks from the untruncated schedules.

    Native checkpoint ids ``92..94`` are the three NextN modules.  Each owns a
    dense FFN, a sliding window of 512, its own ``enorm``/``hnorm``/
    ``eh_proj[H, 2H]``/``output_norm`` and its own head; one embedding is
    shared by the three proposals.  ``cyclic_nextn`` executes proposal ``i``
    with block ``i % 3``.
    """
    from mindie_sim.catalog.draft_spec import parse_draft_max_position_embeddings
    from mindie_sim.catalog.types import AuxiliarySpec, DraftSpec
    nextn = _int(c, "num_nextn_predict_layers")
    if nextn <= 0:
        return None
    if nextn != _STEP5_NEXTN_BLOCKS:
        raise ModelingError(f"model_config_step5_nextn_blocks_invalid:{nextn}")
    native_base = int(len(plans))
    targets = tuple(int(v) for v in (c.get("dspark_target_layer_ids") or ()) or ())
    draft_plans = tuple(
        LayerPlan(
            index=index, native_layer_id=native_base + index,
            attention="gqa", architecture="gqa", ffn="dense",
            window=window,
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=intermediate,
            q_output_gate_kind="headwise" if gate else "none",
            q_gate_projection="separate" if gate else "packed_q",
            ffn_activation="silu",
            qk_rope_head_dim=_step5_rotary_width(c, native_base + index, head_dim),
            qk_nope_head_dim=head_dim - _step5_rotary_width(
                c, native_base + index, head_dim),
            swiglu_limit=_step5_draft_scalar(c, "swiglu_limits", native_base, index),
            swiglu_shared_limit=_step5_draft_scalar(
                c, "swiglu_limits_shared", native_base, index),
            scope="draft",
            kv_cache_elements=KV_CACHE_PLANES * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
        )
        for index in range(nextn)
    )
    spec = DraftSpec(
        kind="mtp", layers=nextn, hidden_size=hidden,
        intermediate_size=intermediate,
        num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
        q_width=heads * head_dim, kv_width=kv_heads * head_dim,
        attention_kind="gqa", q_output_gate=bool(gate),
        num_experts=0, top_k=0, expert_intermediate_size=0,
        target_layer_ids=targets or tuple(range(nextn)),
        sliding_window=window, block_size=1,
        target_shared_embedding=False, target_shared_lm_head=False,
        input_mode="normalized_concat", execution_mode="cyclic_nextn",
        embedding_owned=True, embedding_count=1,
        output_head_owned=True, output_head_count=nextn,
        vocab_size=vocab, pp_capability=STEP5_PP_CAPABILITY,
        layer_plans=draft_plans,
        max_position_embeddings=parse_draft_max_position_embeddings(c),
        source=(("kind", "native_step5_nextn_92_94"),),
        notes=("native NextN ids 92..94; cyclic_nextn executes block i%%3",),
    )
    return AuxiliarySpec(mtp=spec)


def build_fleet_model(key: str, config: Mapping[str, Any], family: str) -> ModelSpec:
    """Lower a fleet HF config through the native structural adapter."""
    builders = {"gemma4": _gemma4, "minimax_m3": _minimax_m3, "step5": _step5}
    if family in builders:
        return builders[family](key, config)
    raise ModelingError(f"model_catalog_profile_unsupported:{family}")

__all__ = ["FLEET_CATALOG_ENTRIES", "FLEET_MODEL_ALIASES", "build_fleet_model"]
