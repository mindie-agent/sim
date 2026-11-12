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

FLEET_CATALOG_ENTRIES: dict[str, str] = {
    key: f"../catalog/configs/{key}.json" for key in (
        "minimax-m3", "minimax-m3.1-flash", "gemma-4-26b-a4b",
        "gemma-4-31b", "qwen3.6-35b-a3b", "qwen3.8-flash-next",
        "step-5-preview",
    )
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
           gaps: tuple[str, ...] = (), vision: Any = None) -> ModelSpec:
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
        norm_eps=float(c.get("rms_norm_eps", 1e-6) or 1e-6),
        num_experts=experts, top_k=top_k, moe_intermediate_size=moe_i,
        num_shared_experts=shared, quant=quant, vision=vision,
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        layer_plans=tuple(plans),
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
    plans = []
    for index, kind in enumerate(kinds):
        full = kind == "full_attention"
        layer_kv = global_kv_heads if full else kv_heads
        layer_hd = global_head_dim if full else head_dim
        factor = 1 if (full and k_eq_v) else 2
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gemma4",
            ffn="moe" if moe else "dense", window=0 if full else window,
            num_heads=heads, num_kv_heads=layer_kv, head_dim=layer_hd,
            intermediate_size=0 if moe else intermediate, num_experts=experts,
            top_k=top_k, moe_intermediate_size=moe_i, num_shared_experts=shared,
            shared_intermediate_size=shared_i, qk_norm=True, ffn_activation="gelu",
            kv_cache_elements=factor * layer_kv * layer_hd,
            kv_owner_layer_id=index, owns_long_kv=True, k_eq_v=bool(full and k_eq_v),
            native_layer_id=index, scope="body",
        ))
    gemma_facts = {
        "gemma4": 1, "gemma4_ple": int(ple), "gemma4_ple_layers": (),
        "gemma4_ple_width": ple_width, "gemma4_ple_tables": 1,
        "gemma4_ple_gated": True, "gemma4_k_eq_v_all": 0,
        "gemma4_k_eq_v_layers": k_layers, "norm_weight_offset": 1.0,
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
    plans = []
    for index in range(layers):
        moe = moe_flags[index]
        active = sparse_on and ((index < len(sparse_freq) and sparse_freq[index])
                                or index >= start)
        if active and min(block, topk_blocks, index_heads, index_dim) <= 0:
            raise ModelingError(f"model_config_minimax_sparse_dims_missing:{index}")
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            ffn="moe" if moe else "dense", num_heads=heads,
            num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=0 if moe else dense_i,
            num_experts=experts if moe else 0, top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=shared if moe else 0,
            shared_intermediate_size=shared_i if moe else 0,
            router_input_size=hidden if moe else 0,
            router_mode=str(c.get("scoring_func", "softmax")),
            qk_norm=qk_norm, q_output_gate=gate, ffn_activation="silu",
            indexer=active, indexer_type="full" if active else "none",
            owns_index_k=active, index_k_owner_layer_id=index if active else -1,
            owns_topk=active, topk_owner_layer_id=index if active else -1,
            index_heads=index_heads if active else 0, index_dim=index_dim if active else 0,
            index_topk=(topk_blocks + local_blocks) * block if active else 0,
            candidate_block_size=block if active else 1,
            candidate_topk_blocks=topk_blocks + local_blocks if active else 0,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
            native_layer_id=index, scope="body",
        ))
    gaps = ("vision_encoder_and_projector:language_backbone_only",) if vision_raw else ()
    return _model(
        key, "minimax_m3", config, c, plans, attention="gqa", ffn="moe",
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
        intermediate=dense_i, vocab=vocab, experts=experts, top_k=top_k,
        moe_i=moe_i, shared=shared, gaps=gaps,
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
    for index, kind in enumerate(kinds):
        full = kind == "full_attention"; moe = index in moe_set
        sparse_layer = bool(sparse_on and kind in apply_types)
        plans.append(LayerPlan(
            index=index, attention="gqa", architecture="gqa",
            ffn="moe" if moe else "dense", window=0 if full else window,
            num_heads=heads, num_kv_heads=kv_heads, head_dim=head_dim,
            intermediate_size=0 if moe else intermediate,
            num_experts=experts if moe else 0, top_k=top_k if moe else 0,
            moe_intermediate_size=moe_i if moe else 0,
            num_shared_experts=1 if moe else 0,
            shared_intermediate_size=shared_i if moe else 0,
            router_input_size=hidden if moe else 0, router_mode="softmax",
            q_output_gate=gate, ffn_activation="silu",
            indexer=sparse_layer, indexer_type="full" if sparse_layer else "none",
            owns_index_k=sparse_layer, index_k_owner_layer_id=index if sparse_layer else -1,
            owns_topk=sparse_layer, topk_owner_layer_id=index if sparse_layer else -1,
            index_heads=idx_heads if sparse_layer else 0,
            index_dim=idx_dim if sparse_layer else 0,
            index_topk=topk_regions * block if sparse_layer else 0,
            index_kpool=block if sparse_layer else 1,
            index_kpool_compress=bool(sparse_layer and block > 1),
            candidate_block_size=block if sparse_layer else 1,
            candidate_topk_blocks=topk_regions if sparse_layer else 0,
            kv_cache_elements=2 * kv_heads * head_dim,
            kv_owner_layer_id=index, owns_long_kv=True,
            native_layer_id=index, scope="body",
        ))
    gaps = ["native_partial_rotary_dimensions_unresolved:retained_plan_proxy"]
    if sparse_on:
        gaps.append("csa_block_compress_ssmax_approximated_by_pooled_indexer")
    if vision_raw:
        gaps.append("vision_encoder_and_projector:language_backbone_only")
    return _model(
        key, "step5", config, c, plans, attention="gqa", ffn="moe",
        heads=heads, kv_heads=kv_heads, head_dim=head_dim, hidden=hidden,
        intermediate=intermediate, vocab=vocab, experts=experts,
        top_k=top_k, moe_i=moe_i, shared=1, gaps=tuple(gaps), facts=None,
    )


def build_fleet_model(key: str, config: Mapping[str, Any], family: str) -> ModelSpec:
    """Lower a fleet HF config through the native structural adapter."""
    builders = {"gemma4": _gemma4, "minimax_m3": _minimax_m3, "step5": _step5}
    if family in builders:
        return builders[family](key, config)
    raise ModelingError(f"model_catalog_profile_unsupported:{family}")

__all__ = ["FLEET_CATALOG_ENTRIES", "FLEET_MODEL_ALIASES", "build_fleet_model"]
