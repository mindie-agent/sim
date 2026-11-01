"""Structural normalization of external transformer configuration mappings.

[Role] Resolve aliases, declared precision and coverage without loading weights.
[Invariants] Unknown fields remain visible; absent quantization geometry is an
  explicit native-precision proxy, never invented packing or measured timing.
[Timeouts & Locks] Pure bounded config mappings; no locks or device work.
[Side-Effect Policy] Returns new mappings and immutable quantization records.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from .config import flatten_config
from .types import QuantSpec
from mindie_sim.schema.common import ModelingError

_NON_LINEAR_EXCLUSION_ROLES = frozenset({"lm_head", "embedding", "norm", "router"})
_REGEX_META_CHARS = frozenset("|()[]?+{}")

_ROLE_EXCLUSION_RULES: tuple[tuple[str, tuple[str, ...], frozenset[str]], ...] = (
    ("lm_head", ("lm_head",), frozenset({"lm_head"})),
    (
        "embedding",
        ("embed_tokens", "embedding", "wte", "word_embeddings"),
        frozenset({"embed_tokens", "embedding", "wte", "word_embeddings"}),
    ),
    (
        "norm",
        (
            "norm", "layernorm", "input_layernorm", "post_attention_layernorm",
            "q_a_layernorm", "kv_a_layernorm", "rms_norm",
        ),
        frozenset({
            "norm", "layernorm", "input_layernorm", "post_attention_layernorm",
            "q_a_layernorm", "kv_a_layernorm", "rms_norm", "self_attn",
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
    "num_key_value_heads": ("multi_query_group_num",),
    "head_dim": ("kv_channels",),
    "intermediate_size": ("ffn_hidden_size", "ffn_dim"),
    "vocab_size": ("padded_vocab_size",),
    "n_routed_experts": ("num_routed_experts", "num_local_experts",
                         "num_experts", "moe_num_experts"),
    "num_experts_per_tok": ("top_k", "num_experts_per_token", "moe_top_k"),
    "n_shared_experts": ("num_shared_experts",),
    "first_k_dense_replace": ("first_k_dense_layers",),
    "shared_expert_intermediate_size": ("shared_intermediate_size", "share_expert_dim"),
}

# Training/tokenizer/provenance settings do not add inference work. Other
# unconsumed keys are retained as gaps; an adapter can explicitly claim a key.
PASSIVE_FIELDS = set("""
architectures model_type name model_key family _name_or_path _commit_hash
auto_map transformers_version torch_dtype dtype activation_dtype weight_dtype
kv_dtype tie_word_embeddings bos_token_id eos_token_id pad_token_id
initializer_range attention_dropout hidden_dropout classifier_dropout use_cache
return_dict output_attentions output_hidden_states output_router_logits
router_aux_loss_coef aux_loss_alpha seq_aux router_jitter_noise
max_position_embeddings max_seq_len max_position_embedding model_max_length
rope_theta rms_norm_eps layer_norm_epsilon _from_model_config z_loss_weight
attention_bias mlp_bias torchscript pretraining_tp
""".split())

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
linear_conv_kernel_dim attn_output_gate qk_norm use_qk_norm
sliding_window use_sliding_window max_window_layers decoder_sparse_step
kv_owner_layers kv_source_layer_ids kv_consumer_to_owner
quantization_config quantization text_config vision_config
rope_scaling qkv_bias partial_rotary_factor
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
index_kpool index_kpool_compress index_kpool_always_select_tail
latent_moe_use_norm media_placeholder_token_id mhc mla_use_output_gate
mlp_layer_types moe_router_dtype num_hash_layers o_groups o_lora_rank
video_token_id
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

    if native_dtype(c) != "bf16":
        omitted(f"native_activation_dtype:{native_dtype(c)}:bf16_proxy",
                "activation_dtype", "torch_dtype", "dtype")
    for key, modeled in (("weight_dtype", quant.weight_format), ("kv_dtype", quant.kv_dtype)):
        declared = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(
            c.get(key), c.get(key))
        if declared is not None and declared != modeled:
            omitted(f"native_{key}:{declared}:{modeled}_proxy", key)
    expert = {"fp4": "mxfp4", "fp8": "fp8_e4m3"}.get(
        c.get("expert_dtype"), c.get("expert_dtype"))
    if expert is not None and expert != quant.expert_weight_format:
        omitted(f"native_expert_dtype:{expert}:{quant.expert_weight_format}_proxy",
                "expert_dtype")
    activation = c.get("hidden_act", "silu")
    represented = {plan.ffn_activation for plan in plans}
    if ((activation not in {"silu", "swiglu", "situ"})
            or (activation == "situ" and represented != {"situ"})):
        omitted(f"native_ffn_activation:{activation}:retained_plan_proxy", "hidden_act")
    for key in ("attention_bias", "mlp_bias", "qkv_bias"):
        if c.get(key):
            omitted(f"projection_bias_not_in_native_layer_plan:{key}", key)
    if float(c.get("partial_rotary_factor", 1.0)) != 1.0:
        omitted("native_partial_rotary_dimensions_unresolved:retained_plan_proxy",
                "partial_rotary_factor")
    if c.get("rope_scaling"):
        omitted("rope_scaling_variant", "rope_scaling")
    if c.get("sliding_window") and not any(plan.window for plan in plans):
        omitted("native_sliding_window_unresolved:full_attention_proxy",
                "sliding_window", "use_sliding_window", "max_window_layers")
    elif c.get("use_sliding_window") or c.get("max_window_layers") is not None:
        omitted("native_window_schedule_unresolved:retained_plan_proxy",
                "sliding_window", "use_sliding_window", "max_window_layers")
    for gap in _native_quant_gaps(c, quant):
        omitted(gap, "quantization_config")
    if c.get("quantization"):
        omitted("native_quantization_field_unresolved:retained_plan_proxy", "quantization")
    metadata = {
        "model_coverage": "partial" if gaps else "complete",
        "unmodeled_features": tuple(sorted(gaps)),
        "degraded_submodules": tuple(sorted(gap for gap in gaps if
            any(word in gap for word in ("proxy", "unmodeled", "unresolved")))),
        "unrecognized_config_fields": {field: c[field] for field in unknown},
        "unconsumed_config_fields": unconsumed,
        "declared_quantization": c.get("quantization_config", {}),
        "source_model_type": c.get("model_type", ""),
    }
    return tuple(sorted(immutable_sequences(metadata).items()))


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
    return tuple(sorted(set(c) - PASSIVE_FIELDS - STRUCTURAL_FIELDS - set(consumed)))


def positive(c, key, default=None):
    value = c.get(key, default)
    if value is None or int(value) <= 0:
        raise ModelingError(f"model_config_dimension_required:{key}")
    return int(value)
