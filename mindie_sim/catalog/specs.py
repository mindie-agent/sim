"""Model-family catalog assembly for the nine se-tools models.

[Role]
  Select the family adapter, assemble body-only LayerPlan schedules plus typed
  quantization/draft/vision/Engram records, exact body parameter/weight totals
  and catalog facts.  Family semantics live in per-family modules.

[Invariants]
  * ``model.layers`` and ``layer_plans`` contain body layers only; optional
    draft/MTP is stored in ``auxiliary`` and never runs on a plain forward.
  * DeepSeek Pro and archived 10T share semantics but keep distinct identities.
  * Missing dimensions raise; no zero placeholder weights.
  * Weight/cache/state values are declared geometry, not measured traffic.

[Timeouts & Locks]
  Pure config parsing; no I/O beyond the caller's already-read mapping.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.config import _int, _profile, _require, flatten_config
from mindie_sim.catalog.deepseek import _aux_deepseek, _deepseek_dims, _deepseek_plans
from mindie_sim.catalog.glm import _aux_glm, _glm_plans
from mindie_sim.catalog.kimi import _aux_kimi, _kimi_plans
from mindie_sim.catalog.quant import (
    _quant_bf16, _quant_deepseek, _quant_glm, _quant_kimi,
)
from mindie_sim.catalog.qwen import (
    _aux_dspark, _aux_qwen, _dspark_plans, _qwen_normalization, _qwen_plans,
)
from mindie_sim.catalog.types import (
    AuxiliarySpec, DraftSpec, EngramSpec, QuantSpec, VisionSpec, merge_source,
)
from mindie_sim.catalog.vision import _engram, _vision_kimi, _vision_qwen, _vision_v41
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes


def _set_facts(model: ModelSpec, facts: Sequence[tuple[str, Any]]) -> None:
    # Frozen dataclass: build the final tuple once through object.__setattr__.
    from mindie_sim.schema.serialization import immutable_sequences
    merged = dict(facts)
    merged["state_bytes_per_request"] = model.state_bytes_per_request
    merged["body_layers"] = model.body_layers
    object.__setattr__(model, "catalog_facts", tuple(sorted(immutable_sequences(merged).items())))


def build_model_spec(key: str, config: Mapping[str, Any]) -> ModelSpec:
    c, text, vision_raw = flatten_config(config)
    family = _profile(config, c)
    if family == "generic" or family.startswith("standard_"):
        from .composable import build_composable
        return build_composable(key, config, family)
    if family == "deepseek_v41":
        plans = _deepseek_plans(c, v41=True)
        quant = _quant_deepseek(c, v41=True)
        attention, ffn = "mla", "moe"
        aux = _aux_deepseek(c, v41=True, d=_deepseek_dims(c, v41=True))
        vision = _vision_v41(config, c) if vision_raw else None
        engram = _engram(c)
    elif family == "deepseek_v4":
        plans = _deepseek_plans(c, v41=False)
        quant = _quant_deepseek(c, v41=False)
        attention, ffn = "mla", "moe"
        aux = _aux_deepseek(c, v41=False, d=_deepseek_dims(c, v41=False))
        vision = None
        engram = None
    elif family == "kimi":
        plans = _kimi_plans(config)
        quant = _quant_kimi(c)
        attention, ffn = "mla", "moe"
        aux = _aux_kimi(c, config)
        vision = _vision_kimi(config, c) if vision_raw else None
        engram = None
    elif family == "kimi_dspark":
        plans = _dspark_plans(c)
        quant = _quant_bf16(c, name="bf16_dspark")
        attention, ffn = "gqa", "dense"
        aux = _aux_dspark(c)
        vision = None
        engram = None
    elif family == "glm":
        plans = _glm_plans(c)
        quant = _quant_glm(c)
        attention, ffn = "mla", "moe"
        aux = _aux_glm(c)
        vision = None
        engram = None
    elif family == "qwen":
        plans = _qwen_plans(c)
        quant = _quant_bf16(c, name="bf16_qwen")
        attention = "gdn"
        ffn = "moe" if _int(c, "num_experts") else "dense"
        aux = _aux_qwen(c)
        vision = _vision_qwen(config, c) if vision_raw else None
        engram = None
    else:
        raise ModelingError(f"model_catalog_profile_unsupported:{family}")
    from .hf_config import native_coverage, resolve_quant
    from .composable import shared_owners
    coverage_gaps = []
    declared_quant = c.get("quantization_config") or {}
    quant_method = str(declared_quant.get("quant_method", ""))
    if declared_quant and (family == "qwen" or quant_method not in {"fp8", "compressed-tensors"}):
        quant, quant_gaps = resolve_quant(c)
        coverage_gaps.extend(quant_gaps)
    # Native adapters resolve structure once. Execution consumes only these
    # explicit mechanisms, and never dispatches on a checkpoint/family label.
    plans = [replace(
        plan,
        cache_kind="window_compressed" if family in {"deepseek_v4", "deepseek_v41"} else "dense",
        compressor_kind=("gated_ring" if family == "deepseek_v41" else "weighted_pool")
                        if plan.compress_ratio else "none",
        compressor_width=2 if family == "deepseek_v4" and plan.compress_ratio == 4 else 1,
        compressor_state_rows=(32 if family == "deepseek_v41" and plan.compress_ratio == 2
                               else plan.compress_ratio * (2 if plan.compress_ratio == 4 else 1)),
        index_input="compressed" if family == "deepseek_v41" else "hidden",
        index_query_per_layer=family == "glm" and plan.attention == "mla" and plan.indexer,
        index_heads=_int(c, "index_n_heads"), index_dim=_int(c, "index_head_dim"),
        index_topk=_int(c, "index_topk"),
        candidate_block_size=max(1, _int(c, "candidate_block_size", 8)),
        candidate_topk_blocks=_int(c, "candidate_topk_blocks", 2048),
        qk_norm=family == "qwen" and plan.attention == "gqa",
    ) for plan in plans]
    # V4.1 uses source-layer sets for compressed KV; its adapter has already
    # resolved them. The dense helper expects one owner entry per layer.
    owner_config = c
    if family == "deepseek_v41":
        owner_config = {key: value for key, value in c.items() if key != "kv_source_layer_ids"}
    plans = shared_owners(owner_config, plans, coverage_gaps)
    if engram is not None:
        engram_layers = {table.layer_id for table in engram.tables}
        plans = [
            replace(plan, engram=True) if plan.index in engram_layers else plan
            for plan in plans
        ]
    body_layers = len(plans)
    if body_layers <= 0:
        raise ModelingError("model_catalog_body_layers_empty")
    h = int(_require(c, "hidden_size"))
    heads = int(_require(c, "num_attention_heads"))
    head_dim = int(c.get("head_dim", 0) or (h // max(1, heads)))
    kv_heads = int(c.get("num_key_value_heads", heads) or heads)
    vocab = int(_require(c, "vocab_size"))
    intermediate = int(c.get("intermediate_size", 0) or c.get("moe_intermediate_size", 0) or 0)
    experts = _int(c, "n_routed_experts", _int(c, "num_experts"))
    top_k = _int(c, "num_experts_per_tok", _int(c, "num_experts_per_token"))
    moe_i = _int(c, "moe_intermediate_size")
    shared = _int(c, "n_shared_experts", _int(c, "num_shared_experts"))
    mla = None
    mla_plans = [p for p in plans if p.attention == "mla"]
    if mla_plans:
        from mindie_sim.schema.model import MlaSpec
        p = mla_plans[0]
        if p.kv_lora_rank <= 0:
            raise ModelingError("model_catalog_mla_dims_incomplete")
        mla = MlaSpec(p.q_lora_rank, p.kv_lora_rank, p.qk_nope_head_dim,
                      p.qk_rope_head_dim, p.v_head_dim)
    state_bytes = float(sum(plan.state_bytes_total for plan in plans))
    if engram is not None and quant is not None:
        storage_mode = str(c.get("engram_storage_mode", quant.engram_table_mode))
        if storage_mode not in {"device_bf16", "device_int8_fp32", "host_fp8_e8m0"}:
            raise ModelingError(f"model_config_engram_storage_mode_unsupported:{storage_mode}")
        quant = replace(quant, engram_table_mode=storage_mode)
        engram = replace(engram, storage_mode=storage_mode)
    model = ModelSpec(
        name=key, layers=body_layers, hidden_size=h, num_heads=heads,
        head_dim=head_dim, intermediate_size=intermediate or (moe_i or 1),
        vocab_size=vocab, attention=attention, ffn=ffn, num_kv_heads=kv_heads,
        activation_dtype="bf16", weight_dtype="bf16",
        kv_dtype=quant.kv_dtype, compute_dtype="bf16",
        norm_eps=float(c.get("rms_norm_eps", 1e-6) or 1e-6),
        quantization=quant.name, num_experts=experts, top_k=top_k,
        moe_intermediate_size=moe_i, num_shared_experts=shared,
        expert_routing=(), mla=mla, layer_plans=tuple(plans),
        state_bytes_per_request=state_bytes,
        body_layers=body_layers, quant=quant, vision=vision,
        engram=engram, auxiliary=aux,
        tie_word_embeddings=bool(c.get("tie_word_embeddings", False)),
        metadata=native_coverage(config, coverage_gaps, plans=plans, quant=quant),
    )
    # Set the activation dtype only for vector/residual math. Quantized GEMM
    # ops carry their own activation/compute dtype explicitly.
    facts = [
        ("profile", family),
        ("model_type", str(c.get("model_type", ""))),
        ("architectures", tuple(str(v) for v in config.get("architectures", ()) or ())),
        ("body_layers", body_layers),
        ("norm_eps", float(c.get("rms_norm_eps", 1e-6) or 1e-6)),
        ("normalization", _qwen_normalization(c) if family == "qwen" else {}),
        ("native_one_based_attention", family == "kimi"),
        ("compress_ratios", tuple(int(p.compress_ratio) for p in plans)),
        ("attention_types", _counts(p.architecture for p in plans)),
        ("ffn_types", _counts(p.ffn for p in plans)),
        ("mla_layers", sum(1 for p in plans if p.attention == "mla")),
        ("gqa_layers", sum(1 for p in plans if p.attention == "gqa")),
        ("kda_layers", sum(1 for p in plans if p.attention == "kda")),
        ("gdn_layers", sum(1 for p in plans if p.attention == "gdn")),
        ("linear_layers", sum(1 for p in plans if p.attention in {"kda", "gdn"})),
        ("dense_layers", sum(1 for p in plans if p.ffn == "dense")),
        ("moe_layers", sum(1 for p in plans if p.ffn == "moe")),
        ("latent_moe_layers", sum(1 for p in plans if p.expert_latent_size > 0)),
        ("indexer_layers", sum(1 for p in plans if p.indexer)),
        ("indexer_full_layers", sum(1 for p in plans if p.indexer and p.indexer_type in {"full", "c4"})),
        ("indexer_shared_layers", sum(1 for p in plans if p.indexer_type == "shared")),
        ("mtp_layers", aux.mtp.layers if aux and aux.mtp else 0),
        ("dspark_layers", aux.dspark.layers if aux and aux.dspark else 0),
        ("num_nextn_predict_layers", _int(c, "num_nextn_predict_layers")),
        ("dspark_target_layer_ids", tuple(int(v) for v in c.get("dspark_target_layer_ids", ()) or ())),
        ("dspark_block_size", _int(c, "dspark_block_size")),
        ("dspark_markov_rank", _int(c, "dspark_markov_rank")),
        ("engram_layer_ids", tuple(t.layer_id for t in (engram.tables if engram else ()))),
        ("engram_tables", len(engram.tables) if engram else 0),
        ("engram_storage_mode", engram.storage_mode if engram else ""),
        ("engram_runtime_row_bytes", engram.row_bytes if engram else 0),
        ("engram_source_row_bytes", 264 if engram else 0),
        ("state_bytes_per_request_unsharded", state_bytes),
        ("vision_config", dict(vision_raw) if isinstance(vision_raw, Mapping) else {}),
        ("vision_present", vision is not None),
        ("vision", vision.to_dict() if vision else None),
        ("hc_mult", max((p.hc_mult for p in plans), default=0)),
        ("hc_sinkhorn_iters", max((p.hc_sinkhorn_iters for p in plans), default=0)),
        ("hyper_connections", any(p.hyper_connections for p in plans)),
        ("residual_block_size", max((p.residual_block_size for p in plans), default=0)),
        ("num_hash_layers", _int(c, "num_hash_layers")),
        ("sliding_window", max((p.window for p in plans), default=0)),
        ("o_groups", max((p.o_groups for p in plans), default=0)),
        ("o_lora_rank", max((p.o_lora_rank for p in plans), default=0)),
        ("index_topk", _int(c, "index_topk")),
        ("index_head_dim", _int(c, "index_head_dim")),
        ("index_n_heads", _int(c, "index_n_heads")),
        ("candidate_source_layer_id", _int(c, "candidate_source_layer_id", -1)),
        ("candidate_topk_blocks", _int(c, "candidate_topk_blocks")),
        ("candidate_block_size", _int(c, "candidate_block_size")),
        ("kv_source_layer_ids", tuple(int(v) for v in c.get("kv_source_layer_ids", ()) or ())),
        ("index_source_layer_ids", tuple(int(v) for v in c.get("index_source_layer_ids", ()) or ())),
        ("quantization", quant.to_dict()),
        ("mlp_layer_types", tuple(str(v) for v in c.get("mlp_layer_types", ()) or ())),
        ("layer_types", tuple(str(v) for v in c.get("layer_types", ()) or ())),
        ("indexer_types", tuple(str(v) for v in c.get("indexer_types", ()) or ())),
        ("dspark_n_routed_experts", _int(c, "dspark_n_routed_experts")),
        ("dspark_num_experts_per_tok", _int(c, "dspark_num_experts_per_tok")),
        ("auxiliary", aux.to_dict() if aux else None),
        ("source_scope", "structural_config_no_latency_labels"),
    ]
    _set_facts(model, facts)
    from mindie_sim.schema.parallel import ParallelSpec as _ParallelSpec
    from mindie_sim.catalog.weights import (
        cache_row_bytes_by_layer, external_shared_weight_bytes,
        external_shared_weight_terms,
    )

    total_params, total_bytes, draft_params, draft_bytes = _model_weight_totals(model)
    external_params = sum(
        term.logical_parameters for term in external_shared_weight_terms(model)
    )
    external_bytes = external_shared_weight_bytes(model, _ParallelSpec())
    facts_extra = [
        ("parameter_count_body", total_params),
        ("resident_weight_bytes_body", total_bytes),
        ("parameter_count_optional_draft", draft_params),
        ("optional_draft_weight_bytes", draft_bytes),
        ("external_shared_parameters", external_params),
        ("external_shared_weight_bytes", external_bytes),
        ("cache_row_bytes_by_layer", cache_row_bytes_by_layer(model)),
    ]
    _set_facts(model, list(model.catalog_facts) + facts_extra)
    return model


def _model_weight_totals(model: ModelSpec) -> tuple[float, float, float, float]:
    from mindie_sim.catalog.weights import (
        body_parameter_count, body_weight_bytes, draft_weight_summary,
    )
    params = body_parameter_count(model)
    weights = body_weight_bytes(model)
    if model.is_standalone_draft:
        # Body already includes the five draft layers plus input/Markov/
        # confidence/projector heads; nothing else is optional.
        return float(params), float(weights), 0.0, 0.0
    draft_params, draft_bytes = draft_weight_summary(model)
    return float(params), float(weights), float(draft_params), float(draft_bytes)


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return result


__all__ = ["build_model_spec", "flatten_config", "merge_source"]
