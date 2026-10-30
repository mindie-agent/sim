"""Cohesive per-layer/per-stage weight formulas for catalog models.

[Role]
  Build exact logical matrix terms (rows, columns, count, representation) from
  the heterogeneous layer schedules and sum source/runtime weight bytes under
  TP/EP/ETP/PP/node placement.

[Invariants]
  * Every layer contributes its attention, FFN/MoE, normalization, indexer,
    compressor, mHC and optional draft/vision/Engram weight terms explicitly.
  * Routed experts are counted once per expert before EP/ETP sharding; the
    Kimi latent transform (H->R->H) is not an expert matrix.
  * KV-head replication means at least one local KV head; it is never a
    fractional shard.
  * Grouped output projection keeps the group axis; weights are not treated as
    a square HxH linear.
  * PP uses the largest actual stage sum, never the first stage by assumption.
  * The mandatory body excludes optional MTP/DSpark weights.

[Timeouts & Locks]
  Pure arithmetic bounded by layers/components; no I/O or locks.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from dataclasses import replace

import math
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from mindie_sim.catalog.types import DraftSpec, QuantSpec, VisionSpec
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.weight_terms import WeightTerm, _ceil_div, _local_groups, _sharded_shape, term_bytes, term_payload_bytes, _t
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec














def stage_weight_decomposition(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool | None = None,
    layer_partition: Sequence[int] | None = None,
) -> list[dict[str, float]]:
    """Per-PP-stage weight bytes split into payload and quant metadata.

    ``total == payload + metadata`` for every stage, so a caller can report the
    scale/zero-point share as an explicit component without moving a single
    resident byte.  The split is a decomposition of ``stage_weight_bytes``,
    never an additional charge.
    """
    quant = model.quant or QuantSpec("bf16")
    stages: list[dict[str, float]] = []
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft,
        include_engram=include_engram,
        include_engram_tables=include_engram_tables,
        layer_partition=layer_partition,
    ):
        total = 0.0
        payload = 0.0
        for term in terms:
            total += term_bytes(term, quant, parallel, node_ranks)
            payload += term_payload_bytes(term, quant, parallel, node_ranks)
        stages.append({
            "total_bytes": float(total),
            "payload_bytes": float(payload),
            "quant_metadata_bytes": float(total - payload),
        })
    return stages




def _norm_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    terms = [
        _t("input_norm", h, 1, "norm"),
        _t("post_attn_norm", h, 1, "norm"),
    ]
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
    if plan.residual_block_size:
        terms.extend([
            _t("attn_res_norm", h, 1, "norm"),
            _t("attn_res_proj", 1, h, "norm"),
            _t("mlp_res_norm", h, 1, "norm"),
            _t("mlp_res_proj", 1, h, "norm"),
        ])
    return terms


def _mhc_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not plan.hyper_connections or plan.hc_mult <= 0:
        return []
    h = model.hidden_size
    mix = (2 + plan.hc_mult) * plan.hc_mult
    hc_dim = plan.hc_mult * h
    terms = []
    for part in ("attn", "ffn"):
        terms.append(_t(f"hc_{part}_fn", mix, hc_dim, "mhc"))
        terms.append(_t(f"hc_{part}_base", mix, 1, "param_fp32"))
        terms.append(_t(f"hc_{part}_scale", 3, 1, "param_fp32"))
    return terms


def _indexer_terms(plan, model):
    if not (plan.owns_index_k or plan.owns_topk or plan.indexer or plan.index_query_per_layer):
        return []
    if plan.index_heads <= 0 or plan.index_dim <= 0:
        raise ModelingError(f"indexer_dimensions_missing:{plan.index}")
    terms = []
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
            terms.extend([_t("indexer_wk", plan.index_dim, width, "indexer_fp8"),
                          _t("indexer_k_norm", plan.index_dim, 1, "norm"),
                          _t("indexer_k_norm_bias", plan.index_dim, 1, "norm")])
    if plan.indexer or plan.owns_topk or plan.index_query_per_layer:
        terms.extend([_t("indexer_wq", plan.index_heads * plan.index_dim, plan.q_lora_rank, "indexer_fp8"),
                      _t("indexer_weights", plan.index_heads, model.hidden_size, "indexer.weights")])
    return terms


def _compressor_terms(plan, model):
    if not plan.owns_long_kv or not plan.compress_ratio:
        return []
    d = plan.head_dim * plan.compressor_width
    return [_t("compressor_wkv", d, model.hidden_size, "compressor_bf16_runtime"),
            _t("compressor_gate", d, model.hidden_size, "compressor_bf16_runtime"),
            _t("compressor_ape", plan.compress_ratio, d, "ape"),
            _t("compressor_norm", plan.head_dim, 1,
               "compressor_fp32_runtime" if plan.compressor_norm_dtype == "fp32" else "norm")]


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
                       "mla", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.head_dim, "mla", "in_tp"))
        else:  # latent_mla (Kimi / GLM)
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
                terms.append(_t("mla_g", heads * plan.v_head_dim, h,
                                "mla.g", "out_tp"))
            if plan.o_groups > 0:
                group_width = heads * plan.v_head_dim // plan.o_groups
                terms.extend([
                    _t("wo_a", plan.o_groups * plan.o_lora_rank, group_width,
                       "mla.o_a", "group_out", groups=plan.o_groups),
                    _t("wo_b", h, plan.o_groups * plan.o_lora_rank,
                       "mla", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.v_head_dim, "mla.o", "in_tp"))
        prepared = dict(model.metadata).get("mla_head_padding", {}).get("layers", {}).get(str(plan.index))
        if prepared:
            # Source process_weights_after_loading creates these separate NZ
            # tensors while retained-prefill mode keeps the checkpoint weights.
            physical = prepared["physical_heads"] * prepared["tp"]
            copies = [
                _t("mlapo_prepared_q_a", plan.q_lora_rank, h, "mla.q_a"),
                _t("mlapo_prepared_kv_a", plan.kv_lora_rank + plan.qk_rope_head_dim, h, "mla.kv_a"),
                _t("mlapo_prepared_q_b", physical * qk, plan.q_lora_rank, "mla.q_b", "out_tp"),
                # process_weights_after_loading retains kv_b_proj and creates
                # separate contiguous UK/UV tensors before prolog UK padding.
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
        q_width = (2 if plan.q_output_gate else 1) * heads * plan.head_dim
        kv_width = plan.num_kv_heads * plan.head_dim
        terms.extend([
            _t("q_proj", q_width, h, "attn.qkv", "out_tp"),
            _t("k_proj", kv_width, h, "attn.qkv", "kv_out",
               units=plan.num_kv_heads, unit_width=plan.head_dim),
            _t("v_proj", kv_width, h, "attn.qkv", "kv_out",
               units=plan.num_kv_heads, unit_width=plan.head_dim),
            _t("o_proj", h, heads * plan.head_dim, "attn.o", "in_tp"),
        ])
        if 0 <= plan.kv_owner_layer_id < plan.index:
            terms = [term for term in terms if term.name not in {"k_proj", "v_proj"}]
        return terms
    if plan.attention == "kda":
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
        prepared_conv = dict(model.metadata).get("causal_conv_update", {}).get("layers", {}).get(str(plan.index))
        if prepared_conv:
            # The source registers a separate activation-dtype W x 3C tensor;
            # process_weights_after_loading keeps the FP32 conv1d parameter.
            terms.append(replace(
                _t("kda_prepared_conv", 2 * k_dim + d, plan.conv_kernel, "conv_weight_bf16", "out_tp"),
                source_owned=False, runtime_dtype=prepared_conv["dtype"]))
        return terms
    if plan.attention == "gdn":
        key_w = plan.linear_key_heads * plan.linear_key_head_dim
        value_w = plan.linear_value_heads * plan.linear_head_dim
        conv_w = plan.conv_width
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


def layer_weight_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    quant = projection_quant(model.quant or QuantSpec("bf16"), plan)
    attention = [replace(term, quant=quant) for term in _attention_terms(plan, model)]
    prepared = dict(model.metadata).get("prepared_mx32_matrix", {}).get("layers", {}).get(str(plan.index), ())
    if prepared:
        # Only explicit post-load declarations change resident scales. The
        # term's checkpoint quant remains untouched for source=True callers.
        attention = [replace(term, runtime_quant=replace(
                         quant.for_category("mla.q_b" if term.name == "q_b" else term.category),
                         weight_block=(1, 32), scale_format="fp8_e8m0", scale_bytes=1))
                     if term.name in prepared else term for term in attention]
    return (
        _norm_terms(plan, model)
        + attention
        + _mhc_terms(plan, model)
        + [replace(term, quant=quant) for term in _indexer_terms(plan, model)]
        + _compressor_terms(plan, model)
        + _ffn_terms(plan, model)
    )


def vision_weight_terms(model: ModelSpec) -> list[WeightTerm]:
    vision = model.vision
    if vision is None:
        return []
    terms = [
        _t("vision_patch", vision.hidden_size, vision.patch_input_width, "vision"),
        _t("vision_pos", max(1, vision.pos_embed_tokens or 1), vision.hidden_size, "vision_pos"),
    ]
    if vision.bias:
        terms.append(_t("vision_patch_bias", vision.hidden_size, 1, "vision"))
    per = [
        _t("vision_qkv", vision.qkv_width, vision.hidden_size, "vision"),
        _t("vision_wo", vision.hidden_size, vision.query_width, "vision"),
        _t("vision_fc0", (2 if vision.gated_mlp else 1) * vision.intermediate_size,
           vision.hidden_size, "vision"),
        _t("vision_fc1", vision.hidden_size, vision.intermediate_size, "vision"),
        _t("vision_norm1", vision.hidden_size, 1, "norm"),
        _t("vision_norm2", vision.hidden_size, 1, "norm"),
    ]
    if vision.bias:
        per.extend([
            _t("vision_qkv_bias", vision.qkv_width, 1, "vision"),
            _t("vision_wo_bias", vision.hidden_size, 1, "vision"),
            _t("vision_fc0_bias", (2 if vision.gated_mlp else 1) * vision.intermediate_size, 1, "vision"),
            _t("vision_fc1_bias", vision.hidden_size, 1, "vision"),
        ])
    if vision.norm_type == "layernorm":
        per.extend([
            _t("vision_norm1_bias", vision.hidden_size, 1, "norm"),
            _t("vision_norm2_bias", vision.hidden_size, 1, "norm"),
        ])
    for term in per:
        terms.append(WeightTerm(term.name, term.rows, term.cols, term.category,
                                term.shard, count=vision.layers))
    terms.append(_t("vision_final_norm", vision.hidden_size, 1, "norm"))
    if vision.norm_type == "layernorm":
        terms.append(_t("vision_final_norm_bias", vision.hidden_size, 1, "norm"))
    aligner_in = vision.aligner_input_width
    if vision.kind == "deepseek_v41":
        terms.extend([
            _t("aligner_norm", aligner_in, 1, "norm"),
            _t("aligner_w1", vision.out_hidden_size, aligner_in, "projector"),
            _t("aligner_w1_bias", vision.out_hidden_size, 1, "vision"),
            _t("aligner_w2", vision.out_hidden_size, vision.out_hidden_size, "projector"),
            _t("aligner_w2_bias", vision.out_hidden_size, 1, "vision"),
        ])
    elif vision.kind == "kimi_k3":
        terms.extend([
            _t("projector_w0", vision.aligner_hidden_size, aligner_in, "projector"),
            _t("projector_w2", vision.out_hidden_size, vision.aligner_hidden_size, "projector"),
            _t("projector_norm", vision.out_hidden_size, 1, "norm"),
        ])
    else:
        terms.extend([
            _t("merger_norm", aligner_in, 1, "norm"),
            _t("merger_norm_bias", aligner_in, 1, "norm"),
            _t("merger_w1", aligner_in, aligner_in, "projector"),
            _t("merger_w1_bias", aligner_in, 1, "vision"),
            _t("merger", vision.out_hidden_size, aligner_in, "projector"),
            _t("merger_bias", vision.out_hidden_size, 1, "vision"),
        ])
    for name in ("image_start", "image_end", "image_newline"):
        terms.append(_t(name, model.hidden_size, 1, "image_token"))
    return terms


def body_head_terms(model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    terms = [_t("final_norm", h, 1, "norm")]
    if not model.is_standalone_draft:
        terms.insert(0, _t("embedding", model.vocab_size, h, "embedding", "embedding"))
        if not model.tie_word_embeddings:
            terms.append(_t("lm_head", model.vocab_size, h, "lm_head", "out_tp"))
    if any(plan.residual_block_size for plan in model.layer_plans):
        terms.extend([
            _t("output_attn_res_norm", h, 1, "norm"),
            _t("output_attn_res_proj", 1, h, "norm"),
        ])
    if any(plan.hyper_connections for plan in model.layer_plans):
        hc = max((p.hc_mult for p in model.layer_plans), default=0)
        terms.extend([
            _t("hc_head_fn", 4, 4 * h, "mhc"),
            _t("hc_head_base", 4, 1, "param_fp32"),
            _t("hc_head_scale", 1, 1, "param_fp32"),
        ])
    return terms


def engram_table_terms(model: ModelSpec) -> list[WeightTerm]:
    engram = model.engram
    if engram is None:
        return []
    return [
        WeightTerm(f"engram_table_{table.table_id}", table.rows, table.width,
                   "engram_table", "node", 1)
        for table in engram.tables
    ]


def engram_gate_terms_by_layer(model: ModelSpec) -> dict[int, list[WeightTerm]]:
    engram = model.engram
    if engram is None:
        return {}
    hc = max((p.hc_mult for p in model.layer_plans), default=0)
    columns = engram.columns_per_table
    by_layer: dict[int, list[WeightTerm]] = {}
    for table in engram.tables:
        by_layer.setdefault(table.layer_id, []).extend([
            _t("engram_gate_wkv", (hc + 1) * model.hidden_size,
               columns * engram.width, "engram_gate", count=1),
            _t("engram_q", hc, model.hidden_size, "engram_gate", count=1),
            _t("engram_k", hc, model.hidden_size, "engram_gate", count=1),
        ])
    return by_layer


def engram_weight_terms(model: ModelSpec) -> list[WeightTerm]:
    terms = engram_table_terms(model)
    for layer_terms in engram_gate_terms_by_layer(model).values():
        terms.extend(layer_terms)
    return terms


from mindie_sim.catalog.draft_weights import draft_terms, draft_weight_summary

def _body_terms(model: ModelSpec, *, include_engram: bool = True) -> list[WeightTerm]:
    terms: list[WeightTerm] = []
    for plan in model.layer_plans:
        terms.extend(layer_weight_terms(plan, model))
    terms.extend(body_head_terms(model))
    terms.extend(vision_weight_terms(model))
    if include_engram:
        terms.extend(engram_weight_terms(model))
    if model.is_standalone_draft:
        terms.extend(
            term for term in draft_terms(
                model.auxiliary.dspark, model, standalone_heads_only=True,
            ) if term.source_owned
        )
    return terms


def _stage_of(layer_id: int, pp: int, layers: int) -> int:
    return min(max(1, pp) - 1, int(layer_id) * max(1, pp) // max(1, layers))


def _partition_stage_of(partition: tuple[int, ...]) -> Any:
    bounds: list[int] = []
    running = 0
    for count in partition:
        running += int(count)
        bounds.append(running)

    def stage(layer_id: int) -> int:
        for index, end in enumerate(bounds):
            if int(layer_id) < end:
                return index
        return len(bounds) - 1

    return stage


def stage_weight_terms(
    model: ModelSpec, parallel: ParallelSpec, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool | None = None,
    layer_partition: Sequence[int] | None = None,
    draft_kind: str | None = None,
) -> list[list[WeightTerm]]:
    pp = max(1, parallel.pp)
    if layer_partition is None:
        if model.layers % pp != 0:
            raise ModelingError(f"layers_not_divisible_by_pp:{model.layers}%{pp}")
        stage_of = lambda layer_id: _stage_of(layer_id, pp, model.layers)
    else:
        # A handed-over PP plan declares the physical split; the weight ledger
        # must charge the same stages the execution lowering assigns.
        partition = tuple(int(value) for value in layer_partition)
        if len(partition) != pp:
            raise ModelingError(f"pp_partition_count:{len(partition)}!={pp}")
        if sum(partition) != model.layers:
            raise ModelingError(
                f"pp_partition_layers:{sum(partition)}!={model.layers}"
            )
        stage_of = _partition_stage_of(partition)
    stages: list[list[WeightTerm]] = [[] for _ in range(pp)]
    for plan in model.layer_plans:
        stages[stage_of(plan.index)].extend(layer_weight_terms(plan, model))
    for term in body_head_terms(model):
        stages[0 if term.name == "embedding" else pp - 1].append(term)
    if model.vision is not None:
        stages[0].extend(vision_weight_terms(model))
    tables_in_hbm = include_engram if include_engram_tables is None else include_engram_tables
    if include_engram:
        if tables_in_hbm:
            tables = engram_table_terms(model)
            for stage in stages:
                stage.extend(tables)
        # The query/gate projections consume the gathered rows and are ordinary
        # device weights; only the tables themselves move under CPU offload.
        for layer_id, gate_terms in engram_gate_terms_by_layer(model).items():
            stages[stage_of(layer_id)].extend(gate_terms)
    if model.is_standalone_draft:
        stages[pp - 1].extend(
            term for term in draft_terms(
                model.auxiliary.dspark, model, standalone_heads_only=True,
            ) if term.source_owned
        )
    if include_draft and model.auxiliary is not None and not model.auxiliary.standalone_draft:
        from mindie_sim.catalog.draft import optional_draft_spec
        selected = ((optional_draft_spec(model, draft_kind),) if draft_kind else
                    (model.auxiliary.mtp, model.auxiliary.dspark))
        for draft in selected:
            if draft is not None:
                stages[pp - 1].extend(
                    term for term in draft_terms(draft, model)
                    if term.source_owned
                )
    return stages


def stage_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool | None = None,
    source: bool = False, layer_partition: Sequence[int] | None = None,
    draft_kind: str | None = None,
) -> list[float]:
    quant = model.quant or QuantSpec("bf16")
    result: list[float] = []
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft,
        draft_kind=draft_kind,
        include_engram=include_engram,
        include_engram_tables=include_engram_tables,
        layer_partition=layer_partition,
    ):
        result.append(float(sum(
            term_bytes(term, quant, parallel, node_ranks, source=source)
            for term in terms
        )))
    return result


def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool = True,
) -> float:
    return float(max(stage_weight_bytes(
        model, parallel, node_ranks, include_draft=include_draft,
        include_engram=include_engram,
        include_engram_tables=include_engram_tables,
    )))


def cache_row_bytes_by_layer(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    """Per-layer cache row width using the declared KV dtype.

    The row width is a declaration, not a fixed BF16 constant: a KV dtype
    swap must move the catalog facts as well as the runtime cache model.
    """
    quant = model.quant or QuantSpec("bf16")
    kv_bytes = dtype_bytes(quant.kv_dtype)
    rows = []
    for plan in model.layer_plans:
        if (plan.attention in {"gqa", "projection_only"} and plan.cache_kind == "dense"
                and not 0 <= plan.kv_owner_layer_id < plan.index):
            row = 2 * plan.num_kv_heads * plan.head_dim * kv_bytes
        elif plan.attention == "mla":
            row = plan.cache_row_elements * kv_bytes
        else:
            row = 0
        rows.append((plan.index, int(row)))
    return tuple(rows)


def weight_breakdown(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_engram_tables: bool | None = None,
    layer_partition: Sequence[int] | None = None,
) -> dict[str, float]:
    quant = model.quant or QuantSpec("bf16")
    groups: dict[str, float] = {}
    for terms in stage_weight_terms(
        model, parallel, include_engram=True,
        include_engram_tables=include_engram_tables,
        layer_partition=layer_partition,
    ):
        for term in terms:
            groups[term.category] = groups.get(term.category, 0.0) + term_bytes(
                term, quant, parallel, node_ranks,
            )
    return groups


from mindie_sim.catalog.ownership import (
    body_parameter_count, body_weight_bytes,
    external_shared_weight_bytes, external_shared_weight_terms,
)


__all__ = [
    "WeightTerm", "layer_weight_terms", "vision_weight_terms",
    "engram_weight_terms", "body_head_terms", "draft_terms",
    "draft_weight_summary", "body_parameter_count", "body_weight_bytes",
    "stage_weight_terms", "stage_weight_bytes", "sharded_weight_bytes",
    "stage_weight_decomposition", "term_payload_bytes",
    "cache_row_bytes_by_layer", "weight_breakdown", "term_bytes",
    "external_shared_weight_terms", "external_shared_weight_bytes",
]
