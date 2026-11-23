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
    AuxiliarySpec, EngramSpec, QuantSpec, VisionSpec, merge_source,
)
from mindie_sim.catalog.vision import _engram, _vision_kimi, _vision_qwen, _vision_v41
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError


DEEPSEEK_V4_COMPRESS_RATIO = 4
DEEPSEEK_V41_COMPRESS_RATIO = 2
DEEPSEEK_V4_COMPRESSOR_WIDTH = 2
DEEPSEEK_V41_COMPRESSOR_STATE_ROWS = 32
COMPRESSED_STATE_ROW_FACTOR = 2
DEFAULT_CANDIDATE_BLOCK_SIZE = 8
DEFAULT_CANDIDATE_TOPK_BLOCKS = 2048
ENGRAM_SOURCE_ROW_BYTES = 264


def _set_facts(model: ModelSpec, facts: Sequence[tuple[str, Any]]) -> None:
    # Frozen dataclass: build the final tuple once through object.__setattr__.
    from .hf_config import json_sequences
    merged = dict(facts)
    merged["state_bytes_per_request"] = model.state_bytes_per_request
    merged["body_layers"] = model.body_layers
    object.__setattr__(model, "catalog_facts",
                       tuple(sorted((key, json_sequences(value))
                                    for key, value in merged.items())))


def build_model_spec(key: str, config: Mapping[str, Any]) -> ModelSpec:
    """Lower one raw config and declare its native auxiliary plans once."""
    from .draft import declare_native_auxiliary_plans
    model = _build_model_spec(key, config)
    c, _text, _vision = flatten_config(config)
    facts = dict(model.catalog_facts)
    if "normalization" not in facts and (
            bool(c.get("use_gemma_norm")) or bool(c.get("apply_layernorm_1p"))):
        # Gemma-style ``(1 + gamma)`` normalization is a declared model
        # semantic regardless of the family adapter that assembled the plans.
        _set_facts(model, list(model.catalog_facts) + [(
            "normalization",
            {"norm_weight_offset": 1.0,
             "norm_multiply_before_output_cast": True,
             "source": {"declaration": "use_gemma_norm_or_apply_layernorm_1p"}},
        )])
    return declare_native_auxiliary_plans(model)


def _dispatch_family_adapter(
    family: str, config: Mapping[str, Any], c: Mapping[str, Any],
    vision_raw: Any,
) -> tuple[list[LayerPlan], QuantSpec, str, str, AuxiliarySpec | None, VisionSpec | None, EngramSpec | None]:
    """Select and run the native family adapter for one catalog profile."""
    if family == "deepseek_v41":
        return (
            _deepseek_plans(c, v41=True), _quant_deepseek(c, v41=True), "mla", "moe",
            _aux_deepseek(c, v41=True, d=_deepseek_dims(c, v41=True)),
            _vision_v41(config, c) if vision_raw else None, _engram(c),
        )
    if family == "deepseek_v4":
        return (
            _deepseek_plans(c, v41=False), _quant_deepseek(c, v41=False), "mla", "moe",
            _aux_deepseek(c, v41=False, d=_deepseek_dims(c, v41=False)), None, None,
        )
    if family == "kimi":
        return (
            _kimi_plans(config), _quant_kimi(c), "mla", "moe",
            _aux_kimi(c, config), _vision_kimi(config, c) if vision_raw else None, None,
        )
    if family == "kimi_dspark":
        plans = _dspark_plans(c)
        return (
            plans, _quant_bf16(c, name="bf16_dspark"),
            plans[0].attention if plans else "gqa", "dense", _aux_dspark(c), None, None,
        )
    if family == "glm":
        return (_glm_plans(c), _quant_glm(c), "mla", "moe", _aux_glm(c), None, None)
    if family == "qwen":
        return (
            _qwen_plans(c), _quant_bf16(c, name="bf16_qwen"), "gdn",
            "moe" if _int(c, "num_experts") else "dense", _aux_qwen(c),
            _vision_qwen(config, c) if vision_raw else None, _engram(c),
        )
    raise ModelingError(f"model_catalog_profile_unsupported:{family}")


def _assemble_catalog_facts(
    *, family: str, config: Mapping[str, Any], c: Mapping[str, Any],
    plans: Sequence[LayerPlan], quant: QuantSpec, vision_raw: Any,
    vision: VisionSpec | None, engram: EngramSpec | None,
    aux: AuxiliarySpec | None, state_bytes: float,
) -> list[tuple[str, Any]]:
    """Build the canonical catalog_facts list for one lowered ModelSpec."""
    from .hf_config import declared_context, declared_rope

    facts = [
        ("profile", family),
        ("model_type", str(c.get("model_type", ""))),
        ("architectures", tuple(str(v) for v in config.get("architectures", ()) or ())),
        ("body_layers", len(plans)),
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
        ("routed_expert_hidden_size", (
            None if c.get("routed_expert_hidden_size") is None
            else int(c["routed_expert_hidden_size"])
        )),
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
        ("engram_source_row_bytes", ENGRAM_SOURCE_ROW_BYTES if engram else 0),
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
        ("rope_parameters", declared_rope(c)),
        ("max_position_embeddings",
         int(declared_context(c).get("max_position_embeddings", 0))),
        ("mlp_layer_types", tuple(str(v) for v in c.get("mlp_layer_types", ()) or ())),
        ("layer_types", tuple(str(v) for v in c.get("layer_types", ()) or ())),
        ("indexer_types", tuple(str(v) for v in c.get("indexer_types", ()) or ())),
        ("dspark_n_routed_experts", _int(c, "dspark_n_routed_experts")),
        ("dspark_num_experts_per_tok", _int(c, "dspark_num_experts_per_tok")),
        ("auxiliary", aux.to_dict() if aux else None),
        ("source_scope", "structural_config_no_latency_labels"),
        ("hc_lowrank", max((p.hc_lowrank for p in plans), default=0)),
    ]
    return facts


def _normalize_family_plans(
    family: str, c: Mapping[str, Any], plans: list[LayerPlan],
    engram: EngramSpec | None, coverage_gaps: list[str],
) -> list[LayerPlan]:
    """Attach canonical cache, compressor, indexer, shared-owner, and Engram fields."""
    from .composable import shared_owners

    plans = [replace(
        plan,
        cache_kind="window_compressed" if family in {"deepseek_v4", "deepseek_v41"} else "dense",
        compressor_kind=("gated_ring" if family == "deepseek_v41" else "weighted_pool")
                        if plan.compress_ratio else "none",
        compressor_width=(DEEPSEEK_V4_COMPRESSOR_WIDTH
                          if family == "deepseek_v4"
                          and plan.compress_ratio == DEEPSEEK_V4_COMPRESS_RATIO else 1),
        compressor_state_rows=(
            DEEPSEEK_V41_COMPRESSOR_STATE_ROWS
            if family == "deepseek_v41"
            and plan.compress_ratio == DEEPSEEK_V41_COMPRESS_RATIO
            else plan.compress_ratio
            * (COMPRESSED_STATE_ROW_FACTOR
               if plan.compress_ratio == DEEPSEEK_V4_COMPRESS_RATIO else 1)),
        index_input="compressed" if family == "deepseek_v41" else "hidden",
        index_query_per_layer=family == "glm" and plan.attention == "mla" and plan.indexer,
        index_heads=plan.index_heads or _int(c, "index_n_heads"),
        index_dim=plan.index_dim or _int(c, "index_head_dim"),
        index_topk=plan.index_topk or _int(c, "index_topk"),
        candidate_block_size=max(1, _int(c, "candidate_block_size",
                                                DEFAULT_CANDIDATE_BLOCK_SIZE)),
        candidate_topk_blocks=_int(c, "candidate_topk_blocks",
                                               DEFAULT_CANDIDATE_TOPK_BLOCKS),
        qk_norm=(plan.qk_norm or (family == "qwen" and plan.attention == "gqa")),
    ) for plan in plans]
    if family == "kimi" and c.get("dsa_config"):
        plans = [replace(plan, index_query_per_layer=plan.indexer) for plan in plans]
    owner_config = ({k: v for k, v in c.items() if k != "kv_source_layer_ids"}
                    if family == "deepseek_v41" else c)
    plans = shared_owners(owner_config, plans, coverage_gaps)
    if engram is not None:
        engram_layers = {table.layer_id for table in engram.tables}
        plans = [
            replace(plan, engram=True) if plan.index in engram_layers else plan
            for plan in plans
        ]
    return plans


def _build_model_spec(key: str, config: Mapping[str, Any]) -> ModelSpec:
    c, _text, vision_raw = flatten_config(config)
    family = _profile(config, c)
    if family in {"gemma4", "minimax_m3", "step5"}:
        from .fleet import build_fleet_model
        return build_fleet_model(key, config, family)
    if family == "generic" or family.startswith("standard_"):
        from .composable import build_composable
        return build_composable(key, config, family)
    plans, quant, attention, ffn, aux, vision, engram = _dispatch_family_adapter(
        family, config, c, vision_raw,
    )
    if bool(config.get("language_model_only", c.get("language_model_only", False))):
        aux = replace(aux, include_vision_weights=False) if aux else AuxiliarySpec(include_vision_weights=False)
    from .hf_config import (
        declared_weight_dtype, native_coverage, resolve_native_formats,
        resolve_quant,
    )
    coverage_gaps: list[str] = []
    declared_quant = c.get("quantization_config") or {}
    quant_method = str(declared_quant.get("quant_method", ""))
    if declared_quant and (family == "qwen" or quant_method not in {"fp8", "compressed-tensors"}):
        quant, quant_gaps = resolve_quant(c)
        coverage_gaps.extend(quant_gaps)
    quant = resolve_native_formats(quant, c)
    declared_weight = declared_weight_dtype(c)
    plans = _normalize_family_plans(family, c, plans, engram, coverage_gaps)
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
        mla = MlaSpec(
            p.q_lora_rank, p.kv_lora_rank, p.qk_nope_head_dim,
            p.qk_rope_head_dim, p.v_head_dim,
            o_lora_rank=p.o_lora_rank, o_groups=p.o_groups,
            index_topk=p.index_topk, index_head_dim=p.index_dim,
            index_n_heads=p.index_heads, compress_ratio=p.compress_ratio,
            sliding_window=p.window,
        )
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
        activation_dtype=str(quant.native_format or "bf16"),
        weight_dtype=(declared_weight
                      if declared_weight in {"bf16", "fp16", "fp32"}
                      and not declared_quant else str(quant.native_format or "bf16")),
        kv_dtype=quant.kv_dtype,
        compute_dtype=str(quant.native_format or "bf16"),
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
    if family == "kimi" and c.get("dsa_config"):
        model = replace(model, metadata=model.metadata + (
            ("cache_allocator", "paired_sfa"),
            ("attention_execution", {
                "dcp_indexer_sharded": True,
                "dcp_output_merge": "alltoall", "recurrent_context_sharded": False,
                "index_rope_dim": int(c["dsa_config"].get("index_rope_dim", 0)),
            }),
        ))
    facts = _assemble_catalog_facts(
        family=family, config=config, c=c, plans=plans, quant=quant,
        vision_raw=vision_raw, vision=vision, engram=engram, aux=aux,
        state_bytes=state_bytes,
    )
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
    _set_facts(model, list(model.catalog_facts) + [
        ("parameter_count_body", total_params),
        ("resident_weight_bytes_body", total_bytes),
        ("parameter_count_optional_draft", draft_params),
        ("optional_draft_weight_bytes", draft_bytes),
        ("external_shared_parameters", external_params),
        ("external_shared_weight_bytes", external_bytes),
        ("cache_row_bytes_by_layer", cache_row_bytes_by_layer(model)),
    ])
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
