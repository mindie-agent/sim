"""Structural normalization of external transformer configuration mappings.

[Role] Resolve aliases, declared precision, rope/context geometry and
  per-value formats without loading weights.
[Invariants] Unknown fields remain visible; absent quantization geometry is an
  explicit native-precision proxy, never invented packing or measured timing.
[Timeouts & Locks] Pure bounded config mappings; no locks or device work.
[Side-Effect Policy] Returns new mappings and immutable quantization records.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .config import flatten_config
from .types import QuantSpec
from mindie_sim.schema.common import ModelingError

_NON_LINEAR_EXCLUSION_ROLES = frozenset({
    "lm_head", "embedding", "norm", "router", "indexer_weights", "mtp",
})
_REGEX_META_CHARS = frozenset("|()[]?+{}")

_ROLE_EXCLUSION_RULES: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...] = (
    ("lm_head", ("lm_head", "shared_head.head"), frozenset({"lm_head", "shared_head", "head"})),
    (
        "embedding",
        ("embed_tokens", "embedding", "wte", "word_embeddings"),
        frozenset({"embed_tokens", "embedding", "wte", "word_embeddings"}),
    ),
    (
        "norm",
        (
            "norm", "layernorm", "input_layernorm", "post_attention_layernorm",
            "q_a_layernorm", "kv_a_layernorm", "rms_norm", "k_norm", "q_norm",
            "enorm", "hnorm", "shared_head.norm",
        ),
        frozenset({
            "norm", "layernorm", "input_layernorm", "post_attention_layernorm",
            "q_a_layernorm", "kv_a_layernorm", "rms_norm", "self_attn",
            "k_norm", "q_norm", "enorm", "hnorm", "shared_head", "indexer",
        }),
    ),
    (
        "router",
        (
            "mlp.gate", "mlp.shared_expert_gate", "shared_expert_gate",
            "mlp.gate.e_score_correction_bias", "router",
        ),
        frozenset({
            "mlp", "gate", "shared_expert_gate", "e_score_correction_bias", "router",
        }),
    ),
    (
        "indexer_weights",
        ("self_attn.indexers_proj", "indexers_proj", "indexer.weights_proj", "weights_proj"),
        frozenset({"self_attn", "indexer", "indexers_proj", "weights_proj"}),
    ),
    (
        "mtp",
        ("eh_proj",),
        frozenset({"eh_proj", "mtp"}),
    ),
    ("attention", ("self_attn", "attention"), frozenset({"self_attn", "attention"})),
    (
        "shared_expert",
        ("shared_experts", "shared_expert"),
        frozenset({"mlp", "shared_experts", "shared_expert"}),
    ),
    (
        "routed_expert",
        ("mlp.experts", "experts", "routed_expert"),
        frozenset({"mlp", "experts", "routed_expert"}),
    ),
)


def _classify_exclusion_pattern(pattern: str) -> str | None:
    raw = str(pattern).strip()
    if not raw or any(ch in raw for ch in _REGEX_META_CHARS):
        return None
    cleaned = raw[3:] if raw.startswith("re:") else raw
    lower = cleaned.lower().replace("\\.", ".")
    if "\\" in lower:
        return None
    segments = [
        seg for seg in lower.strip("^$").replace("*", ".").split(".")
        if seg and seg not in {"model", "layers"} and not seg.isdigit()
    ]
    if not segments:
        return None
    joined = ".".join(segments)
    seg_set = set(segments)
    matched: list[str] = []
    for role, triggers, allowed in _ROLE_EXCLUSION_RULES:
        if seg_set.issubset(allowed) and (
            joined in triggers or any(seg in triggers for seg in segments)
        ):
            matched.append(role)
    return matched[0] if len(matched) == 1 else None


def resolve_role_exclusions(
    patterns: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Map unambiguous standard HF module exclusion patterns to simulator roles."""
    roles: list[str] = []
    unresolved: list[str] = []
    for item in patterns or ():
        pattern = str(item)
        role = _classify_exclusion_pattern(pattern)
        if role is None:
            unresolved.append(pattern)
        elif role not in roles:
            roles.append(role)
    return tuple(roles), tuple(unresolved)


ALIASES = {
    "num_hidden_layers": ("num_layers", "n_layer"),
    "hidden_size": ("d_model", "n_embd"),
    "num_attention_heads": ("n_heads", "n_head"),
    "num_key_value_heads": ("multi_query_group_num", "num_attention_groups"),
    "head_dim": ("kv_channels",),
    "intermediate_size": ("ffn_hidden_size", "ffn_dim"),
    "vocab_size": ("padded_vocab_size",),
    "n_routed_experts": ("num_routed_experts", "num_local_experts",
                         "num_experts", "moe_num_experts"),
    "num_experts_per_tok": ("top_k", "num_experts_per_token", "moe_top_k",
                            "top_k_experts"),
    "n_shared_experts": ("num_shared_experts",),
    "first_k_dense_replace": ("first_k_dense_layers",),
    "shared_expert_intermediate_size": ("shared_intermediate_size", "share_expert_dim"),
}

# Training/tokenizer/provenance settings do not add inference work. Other
# unconsumed keys are retained as gaps; an adapter can explicitly claim a key.
PASSIVE_FIELDS = set("""
architectures model_type name model_key family _name_or_path _commit_hash _torchspec_version
_attn_implementation _attn_implementation_autoset
auto_map transformers_version torch_dtype dtype activation_dtype weight_dtype
kv_dtype tie_word_embeddings bos_token_id eos_token_id pad_token_id
initializer_range attention_dropout hidden_dropout classifier_dropout use_cache
return_dict output_attentions output_hidden_states output_router_logits
router_aux_loss_coef aux_loss_alpha seq_aux router_jitter_noise
max_position_embeddings max_seq_len max_position_embedding model_max_length
rope_theta rms_norm_eps layer_norm_epsilon _from_model_config z_loss_weight
attention_bias mlp_bias torchscript pretraining_tp
ep_size rope_interleave indexer_rope_interleave rope_parameters provenance
compress_rope_theta dspark_noise_token_id hc_eps mtp_use_dedicated_embeddings
language_model_only vision_start_token_id vision_end_token_id vision_token_id
image_placeholder ignore_index bad_words_ids
begin_suppress_tokens decoder_start_token_id diversity_penalty do_sample early_stopping
encoder_no_repeat_ngram_size exponential_decay_length_penalty finetuning_task
forced_bos_token_id forced_eos_token_id id2label
label2id length_penalty max_length min_length no_repeat_ngram_size
num_beam_groups num_beams num_return_sequences output_scores prefix
problem_type remove_invalid_values repetition_penalty
return_dict_in_generate sep_token_id suppress_tokens task_specific_params
temperature tf_legacy_loss tokenizer_class top_p
typical_p use_bfloat16 use_unified_vision_chunk video_placeholder media_placeholder_token_id
image_grid_pinpoints image_seq_length img_token_compression_config
multimodal_projector_bias num_reward_heads process_image_mode
projector_hidden_act projector_hidden_size vision_feature_layer
vision_feature_select_strategy audio_config audio_token_id boa_token_id
boi_token_id eoa_token_id eoa_token_index eoi_token_id
vision_soft_tokens_per_image im_end_token im_patch_token im_start_token
image_token_len patch_token_len projector_bias understand_projector_stride
use_im_start_end vision_select_layer att_impl_type attention_other_setting
fp32_residual_connection head_dim_full_attention head_dim_sliding_attention
num_attention_groups_full_attention num_attention_groups_sliding_attention
num_sliding_attention_heads need_fp32_gate norm_dtype norm_expert_weight
yarn_only_types use_double_wide_mlp use_bidirectional_attention
final_logit_softcapping vocab_size_per_layer_input
make_ngram_vocab_size_divisible_by
""".split())

_FALSY_DEFAULT_FIELDS = frozenset({
    "add_cross_attention", "chunk_size_feed_forward",
    "cross_attention_hidden_size", "is_decoder", "is_encoder_decoder",
    "pruned_heads", "tie_encoder_decoder",
})

STRUCTURAL_FIELDS = set(ALIASES) | {v for names in ALIASES.values() for v in names} | set("""
head_dim intermediate_size moe_intermediate_size q_lora_rank kv_lora_rank
qk_nope_head_dim qk_rope_head_dim v_head_dim layer_types layer_attention_schedule
attention_type ffn_type hidden_act first_k_dense_replace moe_layer_freq
mlp_only_layers moe_layers_enum scoring_func norm_topk_prob topk_method
n_group topk_group routed_scaling_factor num_nextn_predict_layers
mtp_num_hidden_layers num_mtp_layers shared_expert_intermediate_size
shared_expert_gate latent_moe_hidden_size routed_expert_hidden_size
linear_attention_config linear_attn_config linear_num_key_heads
linear_num_value_heads linear_key_head_dim linear_value_head_dim
linear_conv_kernel_dim attn_output_gate output_gate_type
full_attention_interval mamba_ssm_dtype qk_norm use_qk_norm
sliding_window use_sliding_window max_window_layers decoder_sparse_step
kv_owner_layers kv_source_layer_ids kv_consumer_to_owner
quantization_config quantization text_config vision_config
rope_scaling qkv_bias partial_rotary_factor
qk_head_dim moe_renormalize moe_router_activation_func num_expert_group
use_grouped_topk block_size markov_rank num_target_layers
sparse_attention_config ngram_embedding zero_expert_num dit_config
double_stream_layers single_stream_layers swiglu_limit
attention_output_gate dense_intermediate_size num_mtp_modules
attention_k_eq_v enable_moe_block global_head_dim hidden_activation
hidden_size_per_layer_input num_global_key_value_heads moe_layer_list
""".split())

NATIVE_ADAPTER_FIELDS = set("""
activation_situ_beta activation_situ_linear_beta attn_res_block_size
candidate_block_size candidate_source_layer_id candidate_topk_blocks compress_ratios
deepstack_visual_indexes dflash_config downsample_ratio dspark_block_size
dspark_markov_rank dspark_n_routed_experts dspark_num_experts_per_tok
dspark_target_layer_ids enable_confidence_head engram_head_dim engram_layer_ids
engram_max_ngram_size engram_n_heads engram_num_embeddings engram_pad_token_id
engram_storage_mode expert_dtype hc_mult hc_sinkhorn_iters image_token_id
index_head_dim index_n_heads index_source_layer_ids index_topk indexer_types
indexer_head_dim indexer_n_heads index_kpool index_kpool_compress index_kpool_always_select_tail
latent_moe_use_norm media_placeholder_token_id mhc mla_use_output_gate
mlp_layer_types moe_router_dtype num_hash_layers o_groups o_lora_rank
video_token_id mla_use_nope index_share_for_mtp_iteration
index_skip_topk_offset index_topk_freq index_topk_pattern engram_vocab_size
engram_compressed_vocab_size confidence_head_with_markov markov_head_type
apply_layernorm_1p linear_expand_v
target_hidden_size target_num_hidden_layers target_layer_ids mask_token_id
draft_vocab_size full_attention_causal sample_from_anchor
hc_count hc_lowrank heads_per_ngram indexer_budget indexer_compress_ratio
indexer_kv_heads mtp ngram_size ngram_vocab_size_base ple_conv_kernel_size
ple_embed_dim ple_layer_ids split_ngram_parts qk_norm_type rotary_dim
use_gemma_norm use_routing_bias swiglu_alpha expert_intermediate_size
num_kv_shared_layers moe_router_scaling_factor partial_rotary_factors
sparse_config swiglu_limits swiglu_limits_shared use_head_wise_attn_gate
use_moe use_moe_router_bias image_token_index video_token_index
""".split())


def _native_quant_gaps(c, quant):
    """Check declarations the retained native quant adapters do not resolve."""
    raw = c.get("quantization_config") or {}
    if not isinstance(raw, Mapping) or not raw:
        return []
    method = str(raw.get("quant_method", ""))
    gaps = []
    expert = {"fp4": "mxfp4", "fp8": "fp8_e4m3"}.get(
        raw.get("expert_dtype"), raw.get("expert_dtype"))
    if expert is not None and expert != quant.expert_weight_format:
        gaps.append("native_quant_expert_format_unresolved:retained_plan_proxy")
    if method == "fp8":
        if (quant.weight_format != "fp8_e4m3"
                or raw.get("fmt", "e4m3") not in {"e4m3", "e4m3fn"}
                or raw.get("activation_scheme", "dynamic") != "dynamic"):
            gaps.append("native_quantization_variant_unresolved:retained_plan_proxy")
        if raw.get("scale_fmt"):
            scale = {"float32": "fp32", "ue8m0": "fp8_e8m0"}.get(
                raw["scale_fmt"], raw["scale_fmt"])
            if scale != quant.scale_format:
                gaps.append("native_quant_scale_format_unresolved:retained_plan_proxy")
        if raw.get("config_groups"):
            gaps.append("native_quant_config_groups_unresolved:retained_plan_proxy")
    elif method == "compressed-tensors":
        if (raw.get("format") != "mxfp4-pack-quantized"
                or quant.expert_weight_format != "mxfp4"
                or quant.weight_format != "bf16"):
            gaps.append("native_quantization_variant_unresolved:retained_plan_proxy")
        for group in (raw.get("config_groups") or {}).values():
            weights = group.get("weights") or {}
            if (group.get("format", raw.get("format")) != "mxfp4-pack-quantized"
                    or weights.get("num_bits") != 4
                    or weights.get("group_size") != 32
                    or weights.get("type") != "float"
                    or weights.get("strategy") != "group"
                    or weights.get("scale_dtype") not in {None, "uint8", "torch.uint8"}
                    or weights.get("zp_dtype") is not None
                    or weights.get("dynamic", False)
                    or not weights.get("symmetric", True)
                    or group.get("targets", ["Linear"]) != ["Linear"]
                    or group.get("input_activations") is not None
                    or group.get("output_activations") is not None):
                gaps.append("native_quant_config_groups_unresolved:retained_plan_proxy")
        if raw.get("kv_cache_scheme"):
            gaps.append("native_quant_kv_cache_scheme_unresolved:retained_plan_proxy")
    excluded = tuple(str(v) for v in (raw.get("modules_to_not_convert") or raw.get("ignore") or ()))
    if excluded:
        resolved_roles, unresolved_patterns = resolve_role_exclusions(excluded)
        if unresolved_patterns or not set(resolved_roles).issubset(_NON_LINEAR_EXCLUSION_ROLES):
            gaps.append("quant_module_exclusions_unresolved")
    return gaps


_FLOAT_DTYPES = frozenset({"bf16", "fp16", "fp32"})


def json_sequences(value: Any) -> Any:
    """Convert provenance sequences to the JSON-native shape they export as.

    Catalog metadata/facts cross a JSON boundary.  Keeping Python tuples here
    would make ``ModelSpec.to_dict()`` export lists while the in-memory model
    retained tuples, so an accepted SCHEMA-MODEL roundtrip could not hold in
    native calls.  Geometry dataclasses keep their own tuple normalization;
    this helper is only for provenance mappings/records.
    """
    if isinstance(value, (list, tuple)):
        return [json_sequences(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): json_sequences(item) for key, item in value.items()}
    return value


def declared_partial_rotary(c: Mapping[str, Any], index: int | None = None) -> float:
    """Declared rotation fraction for one layer; per-layer arrays win.

    A conforming checkpoint declares either one ``partial_rotary_factor`` or a
    ``partial_rotary_factors`` array covering body and draft layers.  Returning
    the factor is not a proxy: callers turn it into an exact rotated width.
    """
    factors = c.get("partial_rotary_factors")
    if isinstance(factors, (list, tuple)) or (
            isinstance(factors, str) and factors.strip()):
        values = (list(factors) if not isinstance(factors, str)
                  else [v for v in factors.replace("[", "").replace("]", "").split(",") if v.strip()])
        position = 0 if index is None else int(index)
        if position >= len(values):
            raise ModelingError(f"partial_rotary_factors_length_mismatch:{position}")
        value = float(values[position])
    else:
        rope = c.get("rope_parameters")
        rope = rope if isinstance(rope, Mapping) else {}
        raw = c.get("partial_rotary_factor", rope.get("partial_rotary_factor"))
        value = float(raw) if raw is not None else 1.0
    if not 0.0 < value <= 1.0:
        raise ModelingError(f"partial_rotary_factor_out_of_range:{value}")
    return value


def rotary_dimensions(head_dim: int, factor: float) -> tuple[int, int]:
    """Exact (rotated, unchanged) prefix widths of one full head dimension."""
    width = int(head_dim)
    if width <= 0:
        # A plan without declared projection geometry rotates nothing.
        return 0, 0
    if factor >= 1.0:
        return width, 0
    rotated = int(round(width * factor))
    if rotated <= 0 or rotated > width:
        raise ModelingError(f"partial_rotary_width_invalid:{rotated}/{width}")
    return rotated, width - rotated


def declared_rope(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return only RoPE attributes that the source actually declares.

    ``rope_parameters`` is authoritative when present; the older top-level
    ``rope_scaling`` mapping and scalar fields fill gaps only.  No type,
    factor, theta or section is inferred from a model name or a default.
    """
    c, _, _ = flatten_config(config)
    record: dict[str, Any] = {}
    params = c.get("rope_parameters")
    if isinstance(params, Mapping):
        record.update(dict(params))
    scaling = c.get("rope_scaling")
    if isinstance(scaling, Mapping):
        for key, value in scaling.items():
            record.setdefault(key, value)
    if "rope_type" not in record:
        kind = record.get("type")
        if isinstance(kind, str) and kind:
            record["rope_type"] = kind
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section",
                "mrope_interleaved", "compress_rope_theta", "mla_use_nope"):
        value = c.get(key)
        if value is not None:
            record.setdefault(key, value)
    return record


def declared_context(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return declared position/window bounds without a model-name default."""
    c, _, _ = flatten_config(config)
    record: dict[str, Any] = {}
    for key in ("max_position_embeddings", "max_seq_len", "max_position_embedding"):
        value = c.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
            record["max_position_embeddings"] = int(value)
            break
    for key in ("sliding_window", "use_sliding_window", "max_window_layers"):
        value = c.get(key)
        if value is not None:
            record[key] = value
    return record


_STORAGE_FORMATS = frozenset({
    "bf16", "fp16", "fp32", "fp8_e4m3", "fp8_e5m2", "int8",
    "mxfp4", "fp4", "int4", "w4a8",
})


def normalize_declared_format(value: Any, *, expert: bool = False) -> str:
    """Canonicalise one explicitly declared representation, or return empty.

    This helper never invents a format.  It only maps spelling aliases that
    the source configs already use and rejects everything else so the caller
    can retain an explicit coverage gap.
    """
    if value is None:
        return ""
    text = str(value).strip().lower().replace("-", "_")
    text = {
        "bfloat16": "bf16", "float16": "fp16", "float32": "fp32",
        "float8_e4m3": "fp8_e4m3", "float8_e4m3fn": "fp8_e4m3",
        "float8_e5m2": "fp8_e5m2", "fp8": "fp8_e4m3",
    }.get(text, text)
    if expert:
        text = {"fp4": "mxfp4", "fp8": "fp8_e4m3"}.get(text, text)
    return text if text in _STORAGE_FORMATS else ""


def declared_weight_dtype(config: Mapping[str, Any]) -> str:
    c, _, _ = flatten_config(config)
    return normalize_declared_format(c.get("weight_dtype"))


def resolve_native_formats(quant, config: Mapping[str, Any]):
    """Bind declared native stream and independent storage formats.

    The activation stream may inherit a declared FP16/FP32/FP8 native format
    only for the unquantized default representation.  Explicit KV, weight,
    expert, index and scale declarations stay independent: equality with the
    old native spelling is never used as proof of inheritance.  A quantized
    plan keeps every weight/expert/scale field; only its native stream label
    and an explicitly declared KV format are updated.
    """
    c, _, _ = flatten_config(config)
    native = normalize_declared_format(native_dtype(c))
    if quant is None:
        return quant
    raw_quant = c.get("quantization_config")
    quantized = isinstance(raw_quant, Mapping) and bool(raw_quant)
    explicit_weight = normalize_declared_format(c.get("weight_dtype"))
    explicit_expert = normalize_declared_format(c.get("expert_dtype"), expert=True)
    explicit_kv = normalize_declared_format(c.get("kv_dtype"))
    kv_from_quant = (isinstance(raw_quant, Mapping)
                     and raw_quant.get("kv_cache_scheme") is not None)
    changes: dict[str, Any] = {}
    if native and quant.native_format != native:
        changes["native_format"] = native
    if explicit_kv and not kv_from_quant and explicit_kv != quant.kv_dtype:
        changes["kv_dtype"] = explicit_kv
    high = {"bf16", "fp16", "fp32"}
    if not quantized:
        if explicit_weight in high:
            changes["weight_format"] = explicit_weight
            changes["expert_weight_format"] = explicit_weight
        if explicit_expert in high:
            changes["expert_weight_format"] = explicit_expert
        if native:
            # An unquantized plan carries native defaults for these four
            # representations.  Explicit declarations have already been
            # applied above; no field is inherited just because its spelling
            # equals the old native value.
            if "weight_format" not in changes and not explicit_weight:
                changes["weight_format"] = native
            if "activation_format" not in changes:
                changes["activation_format"] = native
            if ("expert_weight_format" not in changes
                    and not explicit_weight and not explicit_expert):
                changes["expert_weight_format"] = native
            if "kv_dtype" not in changes and not explicit_kv:
                changes["kv_dtype"] = native
    return replace(quant, **changes) if changes else quant


_MATRIX_VALUE_ROLES = (
    "attn.q", "attn.kv", "attn.o", "mla", "mla_absorb", "dense_mlp",
    "routed_expert", "shared_expert", "router", "indexer", "mtp",
)


def _role_value_formats(quant, role: str, *, native: str,
                        compute_override: str = "",
                        output_override: str = "") -> dict[str, Any]:
    """Resolve one matrix role through its effective category QuantSpec."""
    from .ops import compute_dtype_for, matrix_output_format

    resolved = quant.for_category(role)
    compute, activation = compute_dtype_for(quant, role)
    weight_storage = resolved.source_format_for(role)
    weight_runtime = resolved.format_for(role)
    output = matrix_output_format(
        quant, role, activation_dtype=activation, prequant=native)
    if compute_override:
        compute = compute_override
    if output_override:
        output = output_override
    high = {"bf16", "bfloat16", "fp16", "float16", "fp32", "float32", ""}
    return {
        "weight_storage": weight_storage,
        "weight_runtime": weight_runtime,
        "activation": activation,
        "compute": compute,
        "output": output,
        "weight_scale": ("" if weight_runtime in high
                         else resolved.execution_weight_scale_format()),
        "activation_scale": ("" if activation in high
                             else resolved.execution_activation_scale_format()),
    }


_ATTENTION_MATRIX_ROLES = {
    "gqa": ("attn.qkv", "attn.o"),
    "mha": ("attn.qkv", "attn.o"),
    "mqa": ("attn.qkv", "attn.o"),
    "projection_only": ("attn.qkv", "attn.o"),
    "gdn": ("gdn.qkvz", "gdn.o", "gdn.beta_alpha"),
    "kda": ("kda.q", "kda.k", "kda.v", "kda.b", "kda.f_a", "kda.f_b",
            "kda.g", "kda.g_a", "kda.g_b", "kda.o"),
}


def _plan_matrix_roles(plan) -> set[str]:
    """Actual matrix categories reachable from one parsed LayerPlan."""
    roles: set[str] = set()
    attention = str(getattr(plan, "attention", "") or "")
    if attention == "mla":
        roles.add("mla")
        if int(getattr(plan, "q_lora_rank", 0) or 0):
            roles.update(("mla.q_a", "mla.q_b"))
        else:
            roles.add("mla.q_proj")
        roles.add("mla.kv_a")
        roles.add("mla_absorb")
        if int(getattr(plan, "o_groups", 0) or 0):
            roles.update(("mla.o_a", "mla.o_b"))
        else:
            roles.add("mla.o")
        if bool(getattr(plan, "mla_output_gate", False)):
            roles.add("mla.g")
    else:
        roles.update(_ATTENTION_MATRIX_ROLES.get(attention, ()))
    ffn = str(getattr(plan, "ffn", "") or "")
    if ffn == "moe":
        roles.update(("router", "routed_expert"))
        if int(getattr(plan, "num_shared_experts", 0) or 0):
            roles.add("shared_expert")
        if int(getattr(plan, "expert_latent_size", 0) or 0):
            roles.add("latent_moe")
    elif ffn == "dense":
        roles.add("dense_mlp")
    if (bool(getattr(plan, "indexer", False))
            or bool(getattr(plan, "owns_topk", False))
            or bool(getattr(plan, "index_query_per_layer", False))):
        roles.update(("indexer.weights", "indexer_fp8"))
        if int(getattr(plan, "index_kpool", 0) or 0) > 1:
            roles.update(("indexer.pool_projection", "indexer.pool_gate"))
    if str(getattr(plan, "scope", "") or "") == "mtp":
        roles.add("mtp")
    return roles


def model_value_formats(model) -> dict[str, Any]:
    """Project one resolved ModelSpec into per-layer/role value formats.

    Every layer uses the same pure chain as execution:
    ``projection_quant(model.quant, plan)`` -> ``for_category(category)`` ->
    the shared ``matrix_output_format`` rule.  A role summary is shared only
    when every applicable layer resolves the same complete record; otherwise
    the caller receives the per-layer binding.  QuantSpec explicit output,
    caller override, MX/4-bit and the normal matrix default are all resolved by
    the same helper used by ``gemm_op``.
    """
    from .quant import projection_quant
    from .types import QuantSpec

    native = str(getattr(model, "activation_dtype", "") or "bf16")
    quant = getattr(model, "quant", None)
    compute_override = ""
    if quant is None:
        weight = (normalize_declared_format(getattr(model, "weight_dtype", ""))
                  or native)
        kv = (normalize_declared_format(getattr(model, "kv_dtype", ""))
              or native)
        activation = str(getattr(model, "activation_dtype", "") or native)
        compute_override = str(getattr(model, "compute_dtype", "") or activation)
        quant = QuantSpec(
            name="declared", weight_format=weight, activation_format=activation,
            expert_weight_format=weight, kv_dtype=kv, index_dtype=native,
            native_format=activation,
        )
    plans = tuple(getattr(model, "layer_plans", ()) or ())
    metadata = dict(getattr(model, "metadata", ()) or ())
    roles = {
        role: _role_value_formats(quant, role, native=native,
                                  compute_override=compute_override)
        for role in _MATRIX_VALUE_ROLES
    }
    layer_records: dict[str, dict[int, dict[str, Any]]] = {}
    for plan in plans:
        layer_quant = projection_quant(quant, plan)
        layer_index = int(getattr(plan, "index", 0))
        for role in _plan_matrix_roles(plan):
            record = _role_value_formats(
                layer_quant, role, native=native,
                compute_override=compute_override)
            if role == "router":
                router_dtype = str(getattr(plan, "router_dtype", "") or "")
                router_output = str(getattr(plan, "router_output_dtype", "") or "")
                if router_dtype:
                    record.update(weight_storage=router_dtype,
                                  weight_runtime=router_dtype,
                                  activation=router_dtype, compute=router_dtype,
                                  weight_scale="", activation_scale="")
                if router_output:
                    record["output"] = router_output
                elif router_dtype:
                    record["output"] = router_dtype
            layer_records.setdefault(role, {})[layer_index] = record
    for role, records in layer_records.items():
        if not records:
            continue
        variants = {tuple(sorted(record.items())) for record in records.values()}
        if len(variants) == 1:
            summary = dict(next(iter(records.values())))
            summary["layers"] = tuple(sorted(records))
            roles[role] = summary
        else:
            roles[role] = {
                "heterogeneous": True,
                "by_layer": {layer: dict(records[layer])
                             for layer in sorted(records)},
            }
    rope = metadata.get("rope_parameters")
    if not isinstance(rope, Mapping):
        unconsumed = metadata.get("unconsumed_config_fields")
        rope = (unconsumed.get("rope_parameters")
                if isinstance(unconsumed, Mapping) else None)
    rope = dict(rope) if isinstance(rope, Mapping) else {}
    for key in ("rope_theta", "partial_rotary_factor", "mrope_section",
                "mrope_interleaved", "mla_use_nope"):
        if key not in rope and key in metadata:
            rope[key] = metadata[key]
    partial = rope.get("partial_rotary_factor", 1.0)
    try:
        partial = float(partial)
    except (TypeError, ValueError):
        partial = 1.0
    mla = getattr(model, "mla", None)
    model_head_dim = int(getattr(model, "head_dim", 0) or 0)
    rotary = {}
    for plan in plans:
        attention = str(getattr(plan, "attention", "") or "gqa")
        head_dim = int(getattr(plan, "head_dim", 0) or model_head_dim)
        if attention == "mla":
            dim = int(getattr(plan, "qk_rope_head_dim", 0)
                      or (mla.qk_rope_head_dim if mla is not None else 0))
        elif 0.0 < partial < 1.0:
            dim = int(round(head_dim * partial))
        else:
            dim = head_dim
        rotary.setdefault(attention, set()).add(dim)
    return {
        "activation_stream": native,
        "compute": str(getattr(model, "compute_dtype", "") or native),
        "output_stream": native,
        "kv_cache": str(quant.kv_dtype),
        "index": str(quant.index_dtype),
        "index_scale": str(quant.index_scale_dtype),
        "weight_scale": quant.execution_weight_scale_format(),
        "activation_scale": quant.execution_activation_scale_format(),
        "matrix_roles": roles,
        "heads": {
            "num_attention_heads": int(getattr(model, "num_heads", 0) or 0),
            "num_key_value_heads": int(getattr(model, "num_kv_heads", 0) or 0),
            "head_dim": int(getattr(model, "head_dim", 0) or 0),
            "mla": (mla.to_dict() if mla is not None else None),
        },
        "rotary": {
            "rope_parameters": rope,
            "partial_rotary_factor": partial,
            "rotary_dim_by_attention": {
                key: tuple(sorted(values)) for key, values in sorted(rotary.items())
            },
            "max_position_embeddings": int(metadata.get("max_position_embeddings", 0) or 0),
        },
    }


def native_coverage(config, gaps, *, plans, quant):
    """Retain extra input semantics even on previously specialized adapters."""
    from mindie_sim.schema.serialization import immutable_sequences
    c, _, _ = flatten_config(config)
    unknown = unrecognized_fields(config, consumed=NATIVE_ADAPTER_FIELDS)
    gaps = set(gaps) | {"unrecognized_config_field:" + field for field in unknown}
    unconsumed = {}

    def omitted(reason, *keys):
        gaps.add(reason)
        unconsumed.update({key: c[key] for key in keys if key in c})

    declared_native = native_dtype(c)
    if declared_native not in _FLOAT_DTYPES:
        omitted(f"native_activation_dtype:{declared_native}:bf16_proxy",
                "activation_dtype", "torch_dtype", "dtype")
    for key, modeled in (("weight_dtype", quant.weight_format), ("kv_dtype", quant.kv_dtype)):
        raw = c.get(key)
        declared = normalize_declared_format(raw)
        if raw is not None and not declared:
            omitted(f"native_{key}:{raw}:unresolved_format", key)
        elif declared and declared != modeled:
            omitted(f"native_{key}:{declared}:{modeled}_proxy", key)
    raw_expert = c.get("expert_dtype")
    expert = normalize_declared_format(raw_expert, expert=True)
    if raw_expert is not None and not expert:
        omitted(f"native_expert_dtype:{raw_expert}:unresolved_format", "expert_dtype")
    elif expert and expert != quant.expert_weight_format:
        omitted(f"native_expert_dtype:{expert}:{quant.expert_weight_format}_proxy",
                "expert_dtype")
    activation = c.get("hidden_act", "silu")
    represented = {plan.ffn_activation for plan in plans}
    if activation == "swigluoai":
        # The OAI gate is executed when every layer declares it.
        represented_ok = represented == {"swigluoai"}
    else:
        represented_ok = not ((activation not in {"silu", "swiglu", "situ"})
                              or (activation == "situ" and represented != {"situ"}))
    if not represented_ok:
        omitted(f"native_ffn_activation:{activation}:retained_plan_proxy", "hidden_act")
    for key in ("attention_bias", "mlp_bias", "qkv_bias"):
        if c.get(key):
            omitted(f"projection_bias_not_in_native_layer_plan:{key}", key)
    rope = declared_rope(config)
    partial_rotary = rope.get("partial_rotary_factor")
    if float(partial_rotary if partial_rotary is not None else 1.0) != 1.0:
        # Exact per-layer rotated widths replace the former full-width proxy.
        rotary_unresolved = False
        for plan in plans:
            if str(getattr(plan, "attention", "")) not in {"gqa", "projection_only"}:
                continue
            head_dim = int(getattr(plan, "head_dim", 0) or 0)
            try:
                expected, _ = rotary_dimensions(
                    head_dim, declared_partial_rotary(c, plan.index))
            except ModelingError:
                rotary_unresolved = True
                break
            if int(getattr(plan, "qk_rope_head_dim", 0) or 0) != expected:
                rotary_unresolved = True
                break
        if rotary_unresolved:
            omitted("native_partial_rotary_dimensions_unresolved:retained_plan_proxy",
                    "partial_rotary_factor", "rope_parameters")
    if c.get("rope_scaling"):
        omitted("rope_scaling_variant", "rope_scaling")
    if c.get("sliding_window") and not any(plan.window for plan in plans):
        omitted("native_sliding_window_unresolved:full_attention_proxy",
                "sliding_window", "use_sliding_window", "max_window_layers")
    elif c.get("use_sliding_window") or (
        c.get("use_sliding_window") is not False
        and c.get("max_window_layers") is not None
        and not any(plan.window for plan in plans)
    ):
        omitted("native_window_schedule_unresolved:retained_plan_proxy",
                "sliding_window", "use_sliding_window", "max_window_layers")
    for gap in _native_quant_gaps(c, quant):
        omitted(gap, "quantization_config")
    if c.get("quantization"):
        omitted("native_quantization_field_unresolved:retained_plan_proxy", "quantization")
    context = declared_context(config)
    metadata = {
        "model_coverage": "partial" if gaps else "complete",
        "unmodeled_features": tuple(sorted(gaps)),
        "degraded_submodules": tuple(sorted(gap for gap in gaps if
            any(word in gap for word in ("proxy", "unmodeled", "unresolved")))),
        "unrecognized_config_fields": {field: c[field] for field in unknown},
        "unconsumed_config_fields": unconsumed,
        "declared_quantization": c.get("quantization_config", {}),
        "source_model_type": c.get("model_type", ""),
        **({"routed_scaling_factor": c["routed_scaling_factor"]}
           if isinstance(c.get("routed_scaling_factor"), (int, float))
           and not isinstance(c.get("routed_scaling_factor"), bool) else {}),
        **({"rope_parameters": rope} if rope else {}),
        **({"max_position_embeddings": int(context["max_position_embeddings"])}
           if "max_position_embeddings" in context else {}),
    }
    return tuple(sorted((key, json_sequences(value))
                        for key, value in immutable_sequences(metadata).items()))


def normalize_config(config):
    c, _, _ = flatten_config(config)
    for target, aliases in ALIASES.items():
        if c.get(target) is None:
            for alias in aliases:
                if c.get(alias) is not None:
                    c[target] = c[alias]
                    break
    # The common expert-width spelling in MiniMax is intermediate_size.
    if c.get("n_routed_experts") and not c.get("moe_intermediate_size"):
        c["moe_intermediate_size"] = c.get("intermediate_size", 0)
    if c.get("shared_expert_intermediate_size") and not c.get("n_shared_experts"):
        c["n_shared_experts"] = 1
    if not c.get("head_dim") and c.get("hidden_size") and c.get("num_attention_heads"):
        c["head_dim"] = int(c["hidden_size"]) // int(c["num_attention_heads"])
    return c


def native_dtype(c):
    name = str(c.get("activation_dtype", c.get("torch_dtype", c.get("dtype", "bf16"))))
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(name, name)


def resolve_quant(c):
    """Return a declared plan plus any explicit representation degradation."""
    native = native_dtype(c)
    gaps = []
    if native not in {"bf16", "fp16", "fp32"}:
        gaps.append(f"native_dtype_unsupported:{native}:bf16_proxy")
        native = "bf16"
    base = QuantSpec(
        name=native, weight_format=native, activation_format=native,
        expert_weight_format=native, kv_dtype=str(c.get("kv_dtype", native)),
        index_dtype=native, index_scale_dtype="none", native_format=native,
    )
    raw = c.get("quantization_config") or {}
    method = str(raw.get("quant_method", "")) if isinstance(raw, Mapping) else ""
    weight = str(c.get("weight_dtype", native))
    weight = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(weight, weight)
    if not raw and weight == native:
        return base, gaps
    if method == "compressed-tensors":
        from .compressed_int4 import resolve_compressed_int4
        resolved = resolve_compressed_int4(raw, base)
        if resolved is not None:
            return resolved, gaps
    if method == "fp8":
        if raw.get("fmt", "e4m3") not in {"e4m3", "e4m3fn"} or raw.get(
                "activation_scheme", "dynamic") != "dynamic":
            gaps.append(f"quantization_variant_unresolved:{method}:{native}_proxy")
            return base, gaps
        block = raw.get("weight_block_size")
        if block and len(block) == 2 and all(int(v) > 0 for v in block):
            scale = str(raw.get("scale_fmt", "fp32"))
            scale = {"float32": "fp32", "ue8m0": "fp8_e8m0"}.get(scale, scale)
            if scale not in {"fp32", "fp8_e8m0"}:
                gaps.append(f"quant_scale_format_unmodeled:{scale}:{native}_proxy")
                return base, gaps
            excluded = tuple(str(v) for v in (raw.get("modules_to_not_convert") or raw.get("ignore") or ()))
            resolved_roles, unresolved_patterns = resolve_role_exclusions(excluded)
            exclusions_unresolved = bool(excluded) and (
                bool(unresolved_patterns)
                or not set(resolved_roles).issubset(_NON_LINEAR_EXCLUSION_ROLES)
            )
            if exclusions_unresolved:
                gaps.append("quant_module_exclusions_unresolved")
            base_exclusions = ("norm", "router", "embedding", "lm_head")
            merged_exclusions = tuple(dict.fromkeys((*base_exclusions, *resolved_roles)))
            quant = QuantSpec(
                name="hf_fp8_block", weight_format="fp8_e4m3",
                expert_weight_format="fp8_e4m3", activation_format="fp8_e4m3",
                weight_block=tuple(int(v) for v in block),
                scale_format=scale, scale_bytes=4 if scale == "fp32" else 1,
                activation_group_size=int(block[1]), activation_scale_format=scale,
                kv_dtype=base.kv_dtype, index_dtype=base.index_dtype,
                index_scale_dtype="none", native_format=native,
                exclusions=merged_exclusions,
            )
            if exclusions_unresolved:
                # Preserve the model with an explicit whole-matrix native
                # proxy until exclusions can be attached to concrete roles.
                return base, gaps
            return quant, gaps
    gaps.append(f"quantization_geometry_unresolved:{method or weight}:{native}_proxy")
    return base, gaps


def unrecognized_fields(config, *, consumed=()):
    c, _, _ = flatten_config(config)
    passive = set(PASSIVE_FIELDS)
    for field in _FALSY_DEFAULT_FIELDS:
        if field in c and not c[field]:
            passive.add(field)
    return tuple(sorted(set(c) - passive - STRUCTURAL_FIELDS - set(consumed)))


def positive(c, key, default=None):
    value = c.get(key, default)
    if value is None or int(value) <= 0:
        raise ModelingError(f"model_config_dimension_required:{key}")
    return int(value)
