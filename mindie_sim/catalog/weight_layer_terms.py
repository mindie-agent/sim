"""Per-layer logical weight term builders for catalog models.

[Role]
  Construct exact logical matrix and parameter ``WeightTerm`` records for one
  ``LayerPlan`` across normalization, attention (MLA/GQA/KDA/GDN), mHC,
  sparse indexers, KV compressors, and dense/MoE FFN blocks.

[Invariants]
  * Routed experts are counted once per expert before EP/ETP sharding; latent
    MoE transforms are distinct from expert matrices.
  * Grouped output projections preserve their group axis.
  * Post-load runtime prepared copies are tagged ``source_owned=False`` so
    source parameter conservation remains exact.

[Timeouts & Locks]
  Pure per-layer arithmetic; O(terms per layer), no I/O or locks.

[Side-Effect Policy]
  Returns fresh immutable ``WeightTerm`` lists; never mutates model specs.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _t
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec

MHC_SCALE_ELEMENTS = 3
MX32_WEIGHT_BLOCK = (1, 32)
LINEAR_KDA_FAMILY_KINDS = frozenset({
    "kda", "mamba", "gla", "deltanet", "lightning", "retnet",
})


def _gemma4_norm_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    """Gemma-4 declares four dense norms and seven MoE norms plus scales."""
    h = model.hidden_size
    terms = [_t("pre_ffn_norm", h, 1, "norm"),
             _t("post_ffn_norm", h, 1, "norm")]
    if plan.ffn == "moe":
        terms.extend([_t("pre_experts_norm", h, 1, "norm"),
                      _t("post_experts_norm", h, 1, "norm"),
                      _t("router_norm", h, 1, "norm"),
                      _t("router_scale", h, 1, "param_fp32"),
                      _t("expert_scale", int(plan.num_experts), 1, "param_fp32")])
    return terms


def _norm_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    terms = ([] if plan.hc_lowrank > 0 else
             [_t("input_norm", h, 1, "norm"), _t("post_attn_norm", h, 1, "norm")])
    if plan.attention == "mla":
        if plan.q_lora_rank:
            terms.append(_t("q_a_norm", plan.q_lora_rank, 1, "norm"))
        if plan.kv_lora_rank:
            terms.append(_t("kv_a_norm", plan.kv_lora_rank, 1, "norm"))
    elif plan.attention == "gqa" and plan.qk_norm:
        terms.append(_t("q_norm", plan.head_dim, 1, "norm"))
        if not 0 <= plan.kv_owner_layer_id < plan.index:
            terms.append(_t("k_norm", plan.head_dim, 1, "norm"))
    elif plan.attention == "gdn":
        terms.append(_t("gdn_norm", plan.linear_head_dim, 1, "norm"))
    if plan.architecture == "gemma4":
        terms.extend(_gemma4_norm_terms(plan, model))
    if plan.residual_block_size:
        terms.extend([
            _t("attn_res_norm", h, 1, "norm"),
            _t("attn_res_proj", 1, h, "attn_res_proj"),
            _t("mlp_res_norm", h, 1, "norm"),
            _t("mlp_res_proj", 1, h, "attn_res_proj"),
        ])
    return terms


def gemma4_ple_terms(model: ModelSpec) -> list[WeightTerm]:
    """Nonzero per-layer-embedding tables and their projection/norm."""
    facts = dict(model.catalog_facts)
    width = int(facts.get("gemma4_ple_width", 0) or 0)
    if not facts.get("gemma4_ple") or width <= 0:
        return []
    layers = max(1, int(model.layers))
    vocab = int(facts.get("gemma4_ple_vocab", model.vocab_size) or model.vocab_size)
    h = model.hidden_size
    return [_t("ple_embedding", vocab, layers * width, "ple_table", "embedding"),
            _t("ple_model_projection", layers * width, h, "ple_projection"),
            _t("ple_projection_norm", width, 1, "norm")]


def gemma4_ple_layer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    del plan
    facts = dict(model.catalog_facts)
    width = int(facts.get("gemma4_ple_width", 0) or 0)
    if not facts.get("gemma4_ple") or width <= 0:
        return []
    h = model.hidden_size
    return [_t("ple_gate", width, h, "ple_gate"),
            _t("ple_projection", h, width, "ple_gate"),
            _t("ple_norm", h, 1, "norm")]


def _mhc_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not plan.hyper_connections or plan.hc_mult <= 0:
        return []
    if plan.hc_lowrank > 0:
        from mindie_sim.catalog.qwen4_exp import lowrank_hc_layer_terms
        return lowrank_hc_layer_terms(plan, model)
    mix = (2 + plan.hc_mult) * plan.hc_mult
    hc_dim = plan.hc_mult * model.hidden_size
    terms: list[WeightTerm] = []
    for part in ("attn", "ffn"):
        terms.extend([
            _t(f"hc_{part}_fn", mix, hc_dim, "mhc"),
            _t(f"hc_{part}_base", mix, 1, "param_fp32"),
            _t(f"hc_{part}_scale", MHC_SCALE_ELEMENTS, 1, "param_fp32"),
        ])
    return terms


def _indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not (plan.owns_index_k or plan.owns_topk or plan.indexer or plan.index_query_per_layer):
        return []
    if plan.index_heads <= 0 or plan.index_dim <= 0:
        raise ModelingError(f"indexer_dimensions_missing:{plan.index}")
    if plan.attention == "gqa" and plan.index_kpool > 1:
        from mindie_sim.catalog.qwen4_exp import qsa_indexer_terms
        return qsa_indexer_terms(plan, model)
    terms: list[WeightTerm] = []
    if plan.owns_index_k:
        width = plan.head_dim if plan.index_input == "compressed" else model.hidden_size
        if plan.index_input == "hidden" and plan.compress_ratio > 1:
            expanded = plan.compressor_width * plan.index_dim
            terms.extend([
                _t("indexer_wk", expanded, width, "compressor_bf16_runtime"),
                _t("indexer_gate", expanded, width, "compressor_bf16_runtime"),
                _t("indexer_ape", plan.compress_ratio, expanded, "ape"),
                _t("indexer_k_norm", plan.index_dim, 1,
                   "compressor_fp32_runtime" if plan.compressor_norm_dtype == "fp32" else "norm"),
            ])
        else:
            terms.extend([
                _t("indexer_wk", plan.index_dim, width,
                   "linear_param_bf16" if plan.index_kpool > 1 else "indexer_fp8"),
                _t("indexer_k_norm", plan.index_dim, 1,
                   "param_fp32" if plan.index_kpool > 1 else "norm"),
                _t("indexer_k_norm_bias", plan.index_dim, 1,
                   "param_fp32" if plan.index_kpool > 1 else "norm"),
            ])
            if plan.index_kpool > 1:
                terms.extend([
                    _t("indexer_pool_gate", plan.index_dim, width, "linear_param_bf16"),
                    _t("indexer_pool_ape", plan.index_kpool, plan.index_dim, "param_fp32"),
                ])
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        terms.extend([
            _t("indexer_wq", plan.index_heads * plan.index_dim,
               plan.q_lora_rank or model.hidden_size, "indexer_fp8"),
            _t("indexer_weights", plan.index_heads, model.hidden_size,
               "linear_param_bf16" if plan.index_kpool > 1 else "indexer.weights"),
        ])
    if plan.index_kpool > 1:
        quant = projection_quant(model.quant or QuantSpec("bf16"), plan)
        roles = {"indexer_wk": "indexer.pool_projection", "indexer_weights": "indexer.weights",
                 "indexer_pool_gate": "indexer.pool_gate"}
        if "native_keypool_caller" in dict(model.metadata):
            roles = {"indexer_weights": "indexer.weights"}
        copies = [
            replace(term, name=term.name + "_prepared_" + quant.format_for(roles[term.name]),
                    category=roles[term.name], source_owned=False, runtime_quant=quant)
            for term in terms
            if term.name in roles and quant.format_for(roles[term.name]) != "bf16"
        ]
        terms.extend(copies)
    return terms


def _compressor_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not plan.owns_long_kv or not plan.compress_ratio:
        return []
    d = plan.head_dim * plan.compressor_width
    terms = [_t("compressor_wkv", d, model.hidden_size, "compressor_bf16_runtime")]
    if plan.compress_ratio > 1:
        terms.extend([
            _t("compressor_gate", d, model.hidden_size, "compressor_bf16_runtime"),
            _t("compressor_ape", plan.compress_ratio, d, "ape"),
        ])
    terms.append(_t("compressor_norm", plan.head_dim, 1,
                    "compressor_fp32_runtime" if plan.compressor_norm_dtype == "fp32" else "norm"))
    return terms


def _attention_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    heads = plan.num_heads
    terms: list[WeightTerm] = []
    if plan.attention == "mla":
        qk = plan.qk_nope_head_dim + plan.qk_rope_head_dim
        if plan.mla_variant == "latent_mqa":
            if plan.q_lora_rank:
                terms.extend([
                    _t("q_a", plan.q_lora_rank, h, "mla"),
                    _t("q_b", heads * qk, plan.q_lora_rank, "mla", "out_tp"),
                    _t("wkv", plan.kv_lora_rank, h, "mla"),
                ])
            else:
                terms.extend([
                    _t("q_proj", heads * qk, h, "mla.q_proj", "out_tp"),
                    _t("wkv", plan.kv_lora_rank, h, "mla"),
                ])
            if plan.o_groups > 0:
                group_width = heads * plan.head_dim // plan.o_groups
                terms.extend([
                    _t("wo_a", plan.o_groups * plan.o_lora_rank, group_width,
                       "mla.o_a", "group_out", groups=plan.o_groups),
                    _t("wo_b", h, plan.o_groups * plan.o_lora_rank,
                       "mla.o_b", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.head_dim, "mla", "in_tp"))
        else:
            if plan.q_lora_rank:
                terms.extend([
                    _t("q_a", plan.q_lora_rank, h, "mla.q_a"),
                    _t("q_b", heads * qk, plan.q_lora_rank, "mla.q_b", "out_tp"),
                ])
            else:
                terms.append(_t("q_proj", heads * qk, h, "mla.q_proj", "out_tp"))
            terms.extend([
                _t("kv_a", plan.kv_lora_rank + plan.qk_rope_head_dim, h, "mla.kv_a"),
                _t("kv_b", heads * (plan.qk_nope_head_dim + plan.v_head_dim),
                   plan.kv_lora_rank, "mla_absorb", "out_tp"),
            ])
            if plan.mla_output_gate:
                terms.append(_t("mla_g", heads * plan.v_head_dim, h, "mla.g", "out_tp"))
            if plan.o_groups > 0:
                group_width = heads * plan.v_head_dim // plan.o_groups
                terms.extend([
                    _t("wo_a", plan.o_groups * plan.o_lora_rank, group_width,
                       "mla.o_a", "group_out", groups=plan.o_groups),
                    _t("wo_b", h, plan.o_groups * plan.o_lora_rank,
                       "mla.o_b", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.v_head_dim, "mla.o", "in_tp"))
        prepared = dict(model.metadata).get("mla_head_padding", {}).get("layers", {}).get(str(plan.index))
        if prepared:
            physical = prepared["physical_heads"] * prepared["tp"]
            copies = [
                _t("mlapo_prepared_q_a", plan.q_lora_rank, h, "mla.q_a"),
                _t("mlapo_prepared_kv_a", plan.kv_lora_rank + plan.qk_rope_head_dim, h, "mla.kv_a"),
                _t("mlapo_prepared_q_b", physical * qk, plan.q_lora_rank, "mla.q_b", "out_tp"),
                _t("mlapo_prepared_uk_unpadded", heads * (
                    plan.qk_nope_head_dim + plan.v_head_dim
                    if prepared["logical_heads"] == 1 else plan.qk_nope_head_dim),
                   plan.kv_lora_rank, "mla_absorb", "out_tp"),
                _t("mlapo_prepared_uv", heads * plan.v_head_dim,
                   plan.kv_lora_rank, "mla_absorb", "out_tp"),
            ]
            if prepared["physical_heads"] != prepared["logical_heads"]:
                copies.append(_t("mlapo_prepared_uk", physical * plan.qk_nope_head_dim,
                                 plan.kv_lora_rank, "mla_absorb", "out_tp"))
            terms.extend(replace(term, source_owned=False, runtime_dtype="bf16") for term in copies)
        return terms
    if plan.attention == "unmodeled":
        return []
    if plan.attention in {"gqa", "projection_only"}:
        gate_separate = plan.q_gate_separate
        q_width = plan.q_projection_width(heads, plan.head_dim)
        kv_width = plan.num_kv_heads * plan.head_dim
        terms.extend([
            _t("q_proj", q_width, h, "attn.qkv", "out_tp"),
            _t("k_proj", kv_width, h, "attn.qkv", "kv_out",
               units=plan.num_kv_heads, unit_width=plan.head_dim),
        ])
        if not plan.k_eq_v:
            terms.append(_t("v_proj", kv_width, h, "attn.qkv", "kv_out",
                            units=plan.num_kv_heads, unit_width=plan.head_dim))
        if gate_separate:
            terms.append(_t("g_proj", heads, h, "attn.gate", "out_tp"))
        terms.append(_t("o_proj", h, heads * plan.head_dim, "attn.o", "in_tp"))
        if 0 <= plan.kv_owner_layer_id < plan.index:
            terms = [term for term in terms if term.name not in {"k_proj", "v_proj"}]
        return terms
    if plan.attention in LINEAR_KDA_FAMILY_KINDS:
        d = (plan.linear_value_heads or plan.linear_heads) * plan.linear_head_dim
        dk = plan.linear_key_head_dim or plan.linear_head_dim
        k_dim = (plan.linear_key_heads or plan.linear_heads) * dk
        decay_dim = (plan.linear_value_heads or plan.linear_heads) * dk
        terms.extend([
            _t("kda_q", k_dim, h, "kda.q", "out_tp"),
            _t("kda_k", k_dim, h, "kda.k", "out_tp"),
            _t("kda_v", d, h, "kda.v", "out_tp"),
            _t("kda_b", plan.linear_value_heads or plan.linear_heads, h, "kda.b", "out_tp"),
            _t("kda_f_a", dk, h, "kda.f_a"),
            _t("kda_f_b", decay_dim, dk, "kda.f_b", "out_tp"),
            _t("kda_q_conv", k_dim, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_k_conv", k_dim, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_v_conv", d, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_A_log", plan.linear_value_heads or plan.linear_heads, 1, "a_log", "out_tp"),
            _t("kda_dt_bias", decay_dim, 1, "dt_bias", "out_tp"),
            _t("kda_o_norm", plan.linear_head_dim, 1, "param_fp32"),
            _t("kda_o", h, d, "kda.o", "in_tp"),
        ])
        if plan.gate_rank:
            terms.extend([_t("kda_g_a", plan.gate_rank, h, "kda.g_a"),
                          _t("kda_g_b", d, plan.gate_rank, "kda.g_b", "out_tp")])
        else:
            terms.append(_t("kda_g", d, h, "kda.g", "out_tp"))
        prepared_projection = dict(model.metadata).get("kda_input_projection", {}).get("layers", {}).get(str(plan.index))
        if prepared_projection:
            terms.append(replace(
                _t("kda_composed_f", decay_dim, h, "kda.f", "out_tp"),
                source_owned=False, runtime_dtype=prepared_projection["dtype"]))
        prepared_conv = dict(model.metadata).get("causal_conv_update", {}).get("layers", {}).get(str(plan.index))
        if prepared_conv:
            terms.append(replace(
                _t("kda_prepared_conv", 2 * k_dim + d, plan.conv_kernel, "conv_weight_bf16", "out_tp"),
                source_owned=False, runtime_dtype=prepared_conv["dtype"]))
        return terms
    if plan.attention == "gdn":
        key_w = plan.linear_key_heads * plan.linear_key_head_dim
        value_w = plan.linear_value_heads * plan.linear_head_dim
        conv_w = plan.conv_width or (2 * key_w + value_w)
        terms.extend([
            _t("gdn_qkvz", conv_w + value_w, h, "gdn.qkvz", "out_tp"),
            _t("gdn_ba", 2 * plan.linear_value_heads, h, "gdn.beta_alpha", "out_tp"),
            _t("gdn_conv", conv_w, plan.conv_kernel, "conv_weight_bf16", "out_tp"),
            _t("gdn_o", h, value_w, "gdn.o", "in_tp"),
            _t("gdn_A_log", plan.linear_value_heads, 1, "linear_param_bf16", "out_tp"),
            _t("gdn_dt_bias", plan.linear_value_heads, 1, "linear_param_bf16", "out_tp"),
            _t("gdn_norm", plan.linear_head_dim, 1, "norm"),
        ])
        return terms
    raise ModelingError(f"catalog_attention_kind_unsupported:{plan.attention}")


def _ffn_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    terms: list[WeightTerm] = []
    if plan.ffn == "dense":
        i = plan.intermediate_size
        terms.extend([
            _t("dense_gate", i, h, "dense_mlp", "dense_up"),
            _t("dense_up", i, h, "dense_mlp", "dense_up"),
            _t("dense_down", h, i, "dense_mlp", "dense_down"),
        ])
        return terms
    if plan.num_experts <= 0 or plan.moe_intermediate_size <= 0:
        raise ModelingError(f"catalog_moe_dims_missing:layer{plan.index}")
    expert_input = plan.expert_latent_size or h
    expert_output = plan.expert_latent_size or h
    terms.extend([
        replace(_t("router", plan.num_experts, plan.router_input_size or h, "router"),
                runtime_dtype="" if plan.router_precast_weight_copy else plan.router_dtype),
        _t("expert_gate", plan.moe_intermediate_size, expert_input,
           "routed_expert", "expert_gate_up", count=plan.num_experts),
        _t("expert_up", plan.moe_intermediate_size, expert_input,
           "routed_expert", "expert_gate_up", count=plan.num_experts),
        _t("expert_down", expert_output, plan.moe_intermediate_size,
           "routed_expert", "expert_down", count=plan.num_experts),
    ])
    if plan.router_noaux_tc:
        terms.append(_t("router_correction_bias", plan.num_experts, 1, "param_fp32"))
    if plan.router_precast_weight_copy and plan.router_dtype:
        terms.append(replace(
            _t("router_precast", plan.num_experts, plan.router_input_size or h, "router"),
            source_owned=False, runtime_dtype=plan.router_dtype))
    if plan.expert_latent_size:
        r = plan.expert_latent_size
        terms.extend([
            _t("latent_down", r, h, "latent_moe"),
            _t("latent_norm", r, 1, "norm"),
            _t("latent_up", h, r, "latent_moe"),
        ])
    shared_i = plan.shared_intermediate_size or (
        plan.num_shared_experts * plan.moe_intermediate_size
    )
    if shared_i > 0:
        terms.extend([
            _t("shared_gate", shared_i, h, "shared_expert", "dense_up"),
            _t("shared_up", shared_i, h, "shared_expert", "dense_up"),
            _t("shared_down", h, shared_i, "shared_expert", "dense_down"),
        ])
    if plan.shared_expert_gate:
        terms.append(_t("shared_router", 1, h, "router"))
    return terms


def _tag_projection_roles(terms: list[WeightTerm], plan: LayerPlan,
                          *, scope: str = "target") -> list[WeightTerm]:
    from mindie_sim.catalog.projection_placement import projection_role
    return [replace(term, placement_role=projection_role(
        plan, term.name, scope=scope)) for term in terms]


def layer_weight_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    """Build all tagged weight terms for one model layer."""
    quant = projection_quant(model.quant or QuantSpec("bf16"), plan)
    attention = [replace(term, quant=quant) for term in _attention_terms(plan, model)]
    prepared = dict(model.metadata).get("prepared_mx32_matrix", {}).get("layers", {}).get(str(plan.index), ())
    if prepared:
        attention = [
            replace(term, runtime_quant=replace(
                quant.for_category("mla.q_b" if term.name == "q_b" else term.category),
                weight_block=MX32_WEIGHT_BLOCK, scale_format="fp8_e8m0", scale_bytes=1))
            if term.name in prepared else term
            for term in attention
        ]
    return _tag_projection_roles(
        _norm_terms(plan, model)
        + gemma4_ple_layer_terms(plan, model)
        + attention
        + _mhc_terms(plan, model)
        + [replace(term, quant=quant) for term in _indexer_terms(plan, model)]
        + _compressor_terms(plan, model)
        + _ffn_terms(plan, model),
        plan,
    )


__all__ = [
    "gemma4_ple_layer_terms",
    "gemma4_ple_terms",
    "layer_weight_terms",
]
