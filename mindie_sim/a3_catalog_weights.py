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

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from .a3_catalog_types import DraftSpec, QuantSpec, VisionSpec
from .a3_model_types import LayerPlan, ModelSpec
from .a3_types import A3InputError, ParallelSpec


@dataclass(frozen=True)
class WeightTerm:
    name: str
    rows: int
    cols: int
    category: str
    shard: str = "replicated"
    count: int = 1
    groups: int = 0
    units: int = 0
    unit_width: int = 0
    shared_owner: bool = False
    source_owned: bool = True

    @property
    def logical_parameters(self) -> float:
        return float(self.rows) * float(self.cols) * float(self.count)


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _local_groups(groups: int, tp: int) -> tuple[int, bool]:
    if groups <= 0:
        return 0, False
    if groups % tp == 0:
        return groups // tp, True
    return groups, False


def _sharded_shape(
    term: WeightTerm, parallel: ParallelSpec, node_ranks: int,
) -> tuple[int, int, int]:
    rows, cols, count = int(term.rows), int(term.cols), int(term.count)
    tp, ep = max(1, parallel.tp), max(1, parallel.ep)
    etp = max(1, parallel.etp)
    if term.shard == "replicated":
        return rows, cols, count
    if term.shard == "out_tp":
        return _ceil_div(rows, tp), cols, count
    if term.shard == "in_tp":
        return rows, _ceil_div(cols, tp), count
    if term.shard == "kv_out":
        if term.units <= 0 or term.unit_width <= 0:
            raise A3InputError(f"catalog_kv_shard_granularity_missing:{term.name}")
        local_units = max(1, _ceil_div(term.units, tp))
        return local_units * term.unit_width, cols, count
    if term.shard == "expert_gate_up":
        return _ceil_div(rows, etp), cols, _ceil_div(count, ep)
    if term.shard == "expert_down":
        return rows, _ceil_div(cols, etp), _ceil_div(count, ep)
    if term.shard == "dense_up":
        return _ceil_div(rows, tp), cols, count
    if term.shard == "dense_down":
        return rows, _ceil_div(cols, tp), count
    if term.shard == "embedding":
        return (_ceil_div(rows, tp) if rows % tp == 0 else rows), cols, count
    if term.shard == "node":
        return _ceil_div(rows, max(1, node_ranks)), cols, count
    if term.shard == "group_out":
        local, _ = _local_groups(term.groups, tp)
        if term.groups <= 0:
            raise A3InputError("catalog_group_term_missing_groups")
        return max(1, rows * local // term.groups), cols, count
    if term.shard == "group_in":
        local, _ = _local_groups(term.groups, tp)
        if term.groups <= 0:
            raise A3InputError("catalog_group_term_missing_groups")
        return rows, max(1, cols * local // term.groups), count
    raise A3InputError(f"catalog_weight_shard_unsupported:{term.shard}")


def term_bytes(
    term: WeightTerm, quant: QuantSpec, parallel: ParallelSpec,
    node_ranks: int = 1, *, source: bool = False,
) -> float:
    if source and not term.source_owned:
        return 0.0
    rows, cols, count = _sharded_shape(term, parallel, node_ranks)
    per_matrix = quant.matrix_bytes(rows, cols, term.category, source=source)
    return float(per_matrix) * float(count)


def _t(name: str, rows: int, cols: int, category: str, shard: str = "replicated",
       count: int = 1, groups: int = 0, units: int = 0,
       unit_width: int = 0) -> WeightTerm:
    if rows <= 0 or cols <= 0:
        raise A3InputError(f"catalog_weight_term_invalid:{name}:{rows}x{cols}")
    return WeightTerm(name, int(rows), int(cols), category, shard, int(count),
                      int(groups), int(units), int(unit_width))


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
    elif plan.attention == "gqa":
        terms.append(_t("q_norm", plan.head_dim, 1, "norm"))
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


def _indexer_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not (plan.owns_index_k or plan.owns_topk or plan.indexer):
        return []
    # Indexer dimensions are model-level native fields.  Fall back to explicit
    # plan fields only when the model did not declare a separate indexer.
    facts = dict(model.catalog_facts)
    index_heads = int(facts.get("index_n_heads", 0) or (32 if model.hidden_size == 6144 else 0))
    index_dim = int(facts.get("index_head_dim", 0) or 128)
    q_rank = plan.q_lora_rank
    if index_heads <= 0 or index_dim <= 0 or q_rank <= 0:
        raise A3InputError(f"catalog_indexer_dims_missing:layer{plan.index}")
    terms: list[WeightTerm] = []
    native = str(facts.get("profile", ""))
    if plan.owns_index_k:
        if native == "deepseek_v41":
            terms.append(_t("indexer_wk", index_dim, plan.head_dim, "norm", count=1))
            terms.append(_t("indexer_k_norm", index_dim, 1, "norm"))
        elif native == "glm":
            terms.append(_t("indexer_wk", index_dim, model.hidden_size, "indexer_fp8"))
            terms.append(_t("indexer_k_norm", index_dim, 1, "norm"))
            terms.append(_t("indexer_k_norm_bias", index_dim, 1, "norm"))
    if plan.indexer or plan.owns_topk:
        terms.append(_t("indexer_wq_b", index_heads * index_dim, q_rank, "indexer_fp8"))
        terms.append(_t("indexer_weights_proj", index_heads, model.hidden_size, "norm"))
    if plan.compress_ratio == 4 and native == "deepseek_v4":
        terms.extend([
            _t("indexer_compressor_wkv", 2 * index_dim, model.hidden_size, "norm"),
            _t("indexer_compressor_wgate", 2 * index_dim, model.hidden_size, "norm"),
            _t("indexer_compressor_norm", index_dim, 1, "norm"),
            _t("indexer_compressor_ape", 4, 2 * index_dim, "ape"),
        ])
    return terms


def _compressor_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    if not plan.owns_long_kv or plan.compress_ratio <= 0:
        return []
    native = str(dict(model.catalog_facts).get("profile", ""))
    d = plan.head_dim
    h = model.hidden_size
    if native == "deepseek_v41":
        cat = "compressor_fp32_runtime" if plan.compress_ratio == 2 else "compressor_bf16_runtime"
        terms = [_t("compressor_wkv", d, h, cat),
                 _t("compressor_norm", d, 1, "norm")]
        if plan.compress_ratio == 2:
            terms.append(_t("compressor_wgate", d, h, cat))
        return terms
    if plan.compress_ratio == 4:
        return [
            _t("compressor_wkv", 2 * d, h, "compressor_bf16_runtime"),
            _t("compressor_wgate", 2 * d, h, "compressor_bf16_runtime"),
            _t("compressor_norm", d, 1, "norm"),
            _t("compressor_ape", 4, 2 * d, "ape"),
        ]
    if plan.compress_ratio == 128:
        return [
            _t("compressor_wkv", d, h, "compressor_bf16_runtime"),
            _t("compressor_wgate", d, h, "compressor_bf16_runtime"),
            _t("compressor_norm", d, 1, "norm"),
            _t("compressor_ape", 128, d, "ape"),
        ]
    return []


def _attention_terms(plan: LayerPlan, model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    heads = plan.num_heads
    terms: list[WeightTerm] = []
    if plan.attention == "mla":
        qk = plan.qk_nope_head_dim + plan.qk_rope_head_dim
        if plan.mla_variant == "latent_mqa":
            terms.extend([
                _t("q_a", plan.q_lora_rank, h, "mla"),
                _t("q_b", heads * qk, plan.q_lora_rank, "mla", "out_tp"),
                _t("wkv", plan.kv_lora_rank, h, "mla"),
            ])
            if plan.o_groups > 0:
                group_width = heads * plan.head_dim // plan.o_groups
                terms.extend([
                    _t("wo_a", plan.o_groups * plan.o_lora_rank, group_width,
                       "mla", "group_out", groups=plan.o_groups),
                    _t("wo_b", h, plan.o_groups * plan.o_lora_rank,
                       "mla", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.head_dim, "mla", "in_tp"))
        else:  # latent_mla (Kimi / GLM)
            terms.extend([
                _t("q_a", plan.q_lora_rank, h, "mla"),
                _t("q_b", heads * qk, plan.q_lora_rank, "mla", "out_tp"),
                _t("kv_a", plan.kv_lora_rank + plan.qk_rope_head_dim, h, "mla"),
                _t("kv_b", heads * (plan.qk_nope_head_dim + plan.v_head_dim),
                   plan.kv_lora_rank, "mla", "out_tp"),
            ])
            if plan.mla_output_gate:
                terms.append(_t("mla_g", heads * plan.v_head_dim, h,
                                "mla", "out_tp"))
            if plan.o_groups > 0:
                group_width = heads * plan.v_head_dim // plan.o_groups
                terms.extend([
                    _t("wo_a", plan.o_groups * plan.o_lora_rank, group_width,
                       "mla", "group_out", groups=plan.o_groups),
                    _t("wo_b", h, plan.o_groups * plan.o_lora_rank,
                       "mla", "group_in", groups=plan.o_groups),
                ])
            else:
                terms.append(_t("o_proj", h, heads * plan.v_head_dim, "mla", "in_tp"))
        return terms
    if plan.attention == "gqa":
        q_width = (2 if plan.q_output_gate else 1) * heads * plan.head_dim
        kv_width = plan.num_kv_heads * plan.head_dim
        terms.extend([
            _t("q_proj", q_width, h, "attn", "out_tp"),
            _t("k_proj", kv_width, h, "attn", "kv_out",
               units=plan.num_kv_heads, unit_width=plan.head_dim),
            _t("v_proj", kv_width, h, "attn", "kv_out",
               units=plan.num_kv_heads, unit_width=plan.head_dim),
            _t("o_proj", h, heads * plan.head_dim, "attn", "in_tp"),
        ])
        return terms
    if plan.attention == "kda":
        d = plan.linear_heads * plan.linear_head_dim
        k_dim = plan.linear_key_heads * plan.linear_key_head_dim
        terms.extend([
            _t("kda_q", d, h, "kda", "out_tp"),
            _t("kda_k", k_dim, h, "kda", "out_tp"),
            _t("kda_v", d, h, "kda", "out_tp"),
            _t("kda_g", d, h, "kda", "out_tp"),
            _t("kda_b", plan.linear_heads, h, "kda", "out_tp"),
            _t("kda_f_a", plan.linear_head_dim, h, "kda"),
            _t("kda_f_b", d, plan.linear_head_dim, "kda", "out_tp"),
            _t("kda_q_conv", d, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_k_conv", k_dim, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_v_conv", d, plan.conv_kernel, "conv_weight_f32", "out_tp"),
            _t("kda_A_log", plan.linear_head_dim, 1, "a_log", "out_tp"),
            _t("kda_dt_bias", d, 1, "dt_bias", "out_tp"),
            _t("kda_o_norm", plan.linear_head_dim, 1, "param_fp32"),
            _t("kda_o", h, d, "kda", "in_tp"),
        ])
        return terms
    if plan.attention == "gdn":
        key_w = plan.linear_key_heads * plan.linear_key_head_dim
        value_w = plan.linear_value_heads * plan.linear_head_dim
        conv_w = plan.conv_width
        terms.extend([
            _t("gdn_qkvz", conv_w + value_w, h, "gdn", "out_tp"),
            _t("gdn_ba", 2 * plan.linear_value_heads, h, "gdn", "out_tp"),
            _t("gdn_conv", conv_w, plan.conv_kernel, "conv_weight_bf16", "out_tp"),
            _t("gdn_o", h, value_w, "gdn", "in_tp"),
            _t("gdn_A_log", plan.linear_value_heads, 1, "linear_param_bf16", "out_tp"),
            _t("gdn_dt_bias", plan.linear_value_heads, 1, "linear_param_bf16", "out_tp"),
            _t("gdn_norm", plan.linear_head_dim, 1, "norm"),
        ])
        return terms
    raise A3InputError(f"catalog_attention_kind_unsupported:{plan.attention}")


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
        raise A3InputError(f"catalog_moe_dims_missing:layer{plan.index}")
    expert_input = plan.expert_latent_size or h
    expert_output = plan.expert_latent_size or h
    terms.extend([
        _t("router", plan.num_experts, plan.router_input_size or h, "router"),
        _t("expert_gate", plan.moe_intermediate_size, expert_input,
           "routed_expert", "expert_gate_up", count=plan.num_experts),
        _t("expert_up", plan.moe_intermediate_size, expert_input,
           "routed_expert", "expert_gate_up", count=plan.num_experts),
        _t("expert_down", expert_output, plan.moe_intermediate_size,
           "routed_expert", "expert_down", count=plan.num_experts),
    ])
    if plan.expert_latent_size:
        r = plan.expert_latent_size
        terms.extend([
            _t("latent_down", r, h, "dense_mlp"),
            _t("latent_norm", r, 1, "norm"),
            _t("latent_up", h, r, "dense_mlp"),
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
    return (
        _norm_terms(plan, model)
        + _attention_terms(plan, model)
        + _mhc_terms(plan, model)
        + _indexer_terms(plan, model)
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


def _draft_attention_terms(draft: DraftSpec, quant: QuantSpec) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if draft.attention_kind == "mla":
        qk = draft.qk_nope_head_dim + draft.qk_rope_head_dim
        # latent_mla owns a kv_b projection; latent_mqa uses wkv directly and
        # has head_dim == kv_lora_rank == v_head_dim (V4/V4.1 DSA).
        latent_mla = (
            draft.kv_lora_rank > 0
            and draft.qk_nope_head_dim > 0
            and draft.kv_lora_rank != draft.head_dim
        )
        if latent_mla:
            terms.extend([
                _t("draft_q_a", draft.q_lora_rank, h, "mla"),
                _t("draft_q_b", draft.num_heads * qk, draft.q_lora_rank, "mla", "out_tp"),
                _t("draft_wkv", draft.kv_lora_rank, h, "mla"),
                _t("draft_kv_b", draft.num_heads * (draft.qk_nope_head_dim + draft.v_head_dim),
                   draft.kv_lora_rank, "mla", "out_tp"),
            ])
            if draft.o_groups:
                group_width = draft.num_heads * draft.head_dim // draft.o_groups
                terms.extend([
                    _t("draft_wo_a", draft.o_groups * draft.o_lora_rank,
                       group_width, "mla", "group_out", groups=draft.o_groups),
                    _t("draft_wo_b", h, draft.o_groups * draft.o_lora_rank,
                       "mla", "group_in", groups=draft.o_groups),
                ])
            else:
                terms.append(_t("draft_o", h, draft.num_heads * draft.head_dim, "mla", "in_tp"))
        else:
            terms.extend([
                _t("draft_q_a", draft.q_lora_rank, h, "mla"),
                _t("draft_q_b", draft.num_heads * qk, draft.q_lora_rank, "mla", "out_tp"),
                _t("draft_wkv", draft.kv_lora_rank, h, "mla"),
            ])
            if draft.o_groups:
                group_width = draft.num_heads * draft.head_dim // draft.o_groups
                terms.extend([
                    _t("draft_wo_a", draft.o_groups * draft.o_lora_rank,
                       group_width, "mla", "group_out", groups=draft.o_groups),
                    _t("draft_wo_b", h, draft.o_groups * draft.o_lora_rank,
                       "mla", "group_in", groups=draft.o_groups),
                ])
            else:
                terms.append(_t("draft_o", h, draft.num_heads * draft.head_dim, "mla", "in_tp"))
    else:
        terms.extend([
            _t("draft_q", draft.q_width, h, "attn", "out_tp"),
            _t("draft_k", draft.kv_width, h, "attn", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_v", draft.kv_width, h, "attn", "kv_out",
               units=draft.num_kv_heads, unit_width=draft.head_dim),
            _t("draft_o", h, draft.q_width, "attn", "in_tp"),
        ])
    terms.extend([_t("draft_input_norm", h, 1, "norm"),
                  _t("draft_post_norm", h, 1, "norm"),
                  _t("draft_final_norm", h, 1, "norm")])
    return terms


def _draft_ffn_terms(draft: DraftSpec, model: ModelSpec) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if draft.num_experts > 0:
        terms.extend([
            _t("draft_router", draft.num_experts, h, "router"),
            _t("draft_expert_gate", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_up", draft.expert_intermediate_size, h,
               "dspark_routed_expert", "expert_gate_up", count=draft.num_experts),
            _t("draft_expert_down", h, draft.expert_intermediate_size,
               "dspark_routed_expert", "expert_down", count=draft.num_experts),
        ])
        if draft.num_shared_experts:
            shared_i = draft.expert_intermediate_size * draft.num_shared_experts
            terms.extend([
                _t("draft_shared_gate", shared_i, h, "shared_expert", "dense_up"),
                _t("draft_shared_up", shared_i, h, "shared_expert", "dense_up"),
                _t("draft_shared_down", h, shared_i, "shared_expert", "dense_down"),
            ])
    else:
        i = draft.intermediate_size
        terms.extend([
            _t("draft_gate", i, h, "dense_mlp", "dense_up"),
            _t("draft_up", i, h, "dense_mlp", "dense_up"),
            _t("draft_down", h, i, "dense_mlp", "dense_down"),
        ])
    if draft.mhc_mult:
        mix = (2 + draft.mhc_mult) * draft.mhc_mult
        hc_dim = draft.mhc_mult * h
        for part in ("attn", "ffn"):
            terms.append(_t(f"draft_hc_{part}_fn", mix, hc_dim, "mhc"))
            terms.append(_t(f"draft_hc_{part}_base", mix, 1, "param_fp32"))
            terms.append(_t(f"draft_hc_{part}_scale", 3, 1, "param_fp32"))
    return terms


def draft_terms(
    draft: DraftSpec, model: ModelSpec, *, standalone_heads_only: bool = False,
) -> list[WeightTerm]:
    h = draft.hidden_size
    terms: list[WeightTerm] = []
    if standalone_heads_only and draft.kind == "standalone_draft":
        # The standalone draft always consumes concatenated target features.
        terms.append(_t("draft_input_proj", h, draft.input_width,
                        draft.input_projection_category, "out_tp"))
        terms.extend([_t("draft_input_norm", h, 1, "norm"),
                      _t("draft_final_norm", h, 1, "norm")])
        if draft.target_shared_embedding:
            terms.append(WeightTerm(
                "target_embedding_alias", draft.vocab_size, h,
                "target_alias_embedding", "replicated", 1, 0, 0, 0, True,
                False,
            ))
        if draft.target_shared_lm_head:
            terms.append(WeightTerm(
                "target_lm_head_alias", draft.vocab_size, h,
                "target_alias_lm_head", "replicated", 1, 0, 0, 0, True,
                False,
            ))
    if not standalone_heads_only:
        terms.append(_t("draft_input_proj", h, draft.input_width,
                        draft.input_projection_category, "out_tp"))
        for _ in range(max(0, draft.layers)):
            terms.extend(_draft_attention_terms(draft, model.quant or QuantSpec("bf16")))
            terms.extend(_draft_ffn_terms(draft, model))
    if draft.markov_rank:
        terms.append(_t("markov_embed", draft.vocab_size, draft.markov_rank,
                        "embedding"))
        terms.append(_t("markov_head", draft.vocab_size, draft.markov_rank,
                        "lm_head"))
        if draft.confidence_enabled and draft.confidence_executes:
            width = draft.confidence_input_width or (h + draft.markov_rank)
            terms.append(_t("confidence_head", 1, width, "norm"))
    return terms


def draft_weight_summary(
    model: ModelSpec, *, standalone_heads_only: bool = False,
) -> tuple[float, float]:
    if model.auxiliary is None:
        return 0.0, 0.0
    quant = model.quant or QuantSpec("bf16")
    parallel = ParallelSpec()
    drafts: list[DraftSpec] = []
    aux = model.auxiliary
    if aux.standalone_draft and aux.dspark is not None:
        # Body layers are counted separately; only extra heads/projection here.
        drafts = [aux.dspark]
        standalone_heads_only = True
    else:
        if aux.mtp is not None:
            drafts.append(aux.mtp)
        if aux.dspark is not None and aux.dspark is not aux.mtp:
            drafts.append(aux.dspark)
    params = 0.0
    weights = 0.0
    for draft in drafts:
        for term in draft_terms(draft, model, standalone_heads_only=standalone_heads_only):
            if not term.source_owned:
                continue
            params += term.logical_parameters
            weights += term_bytes(term, quant, parallel, 1)
    return params, weights


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


def stage_weight_terms(
    model: ModelSpec, parallel: ParallelSpec, *,
    include_draft: bool = False, include_engram: bool = True,
) -> list[list[WeightTerm]]:
    pp = max(1, parallel.pp)
    if model.layers % pp != 0:
        raise A3InputError(f"layers_not_divisible_by_pp:{model.layers}%{pp}")
    stages: list[list[WeightTerm]] = [[] for _ in range(pp)]
    for plan in model.layer_plans:
        stages[_stage_of(plan.index, pp, model.layers)].extend(layer_weight_terms(plan, model))
    for term in body_head_terms(model):
        stages[0 if term.name == "embedding" else pp - 1].append(term)
    if model.vision is not None:
        stages[0].extend(vision_weight_terms(model))
    if include_engram:
        tables = engram_table_terms(model)
        for stage in stages:
            stage.extend(tables)
        for layer_id, gate_terms in engram_gate_terms_by_layer(model).items():
            stages[_stage_of(layer_id, pp, model.layers)].extend(gate_terms)
    if model.is_standalone_draft:
        stages[pp - 1].extend(
            term for term in draft_terms(
                model.auxiliary.dspark, model, standalone_heads_only=True,
            ) if term.source_owned
        )
    if include_draft and model.auxiliary is not None and not model.auxiliary.standalone_draft:
        for draft in (model.auxiliary.mtp, model.auxiliary.dspark):
            if draft is not None:
                stages[pp - 1].extend(
                    term for term in draft_terms(draft, model)
                    if term.source_owned
                )
    return stages


def stage_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True, source: bool = False,
) -> list[float]:
    quant = model.quant or QuantSpec("bf16")
    result: list[float] = []
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft, include_engram=include_engram,
    ):
        result.append(float(sum(
            term_bytes(term, quant, parallel, node_ranks, source=source)
            for term in terms
        )))
    return result


def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
) -> float:
    return float(max(stage_weight_bytes(
        model, parallel, node_ranks, include_draft=include_draft,
        include_engram=include_engram,
    )))


def cache_row_bytes_by_layer(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    quant = model.quant or QuantSpec("bf16")
    rows = []
    for plan in model.layer_plans:
        if plan.attention == "gqa":
            row = 2 * plan.num_kv_heads * plan.head_dim * 2
        elif plan.attention == "mla":
            row = plan.cache_row_elements * 2
        else:
            row = 0
        rows.append((plan.index, int(row)))
    return tuple(rows)


def weight_breakdown(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1,
) -> dict[str, float]:
    quant = model.quant or QuantSpec("bf16")
    groups: dict[str, float] = {}
    for terms in stage_weight_terms(model, parallel, include_engram=True):
        for term in terms:
            groups[term.category] = groups.get(term.category, 0.0) + term_bytes(
                term, quant, parallel, node_ranks,
            )
    return groups


from .a3_catalog_ownership import (
    body_parameter_count, body_weight_bytes,
    external_shared_weight_bytes, external_shared_weight_terms,
)


__all__ = [
    "WeightTerm", "layer_weight_terms", "vision_weight_terms",
    "engram_weight_terms", "body_head_terms", "draft_terms",
    "draft_weight_summary", "body_parameter_count", "body_weight_bytes",
    "stage_weight_terms", "stage_weight_bytes", "sharded_weight_bytes",
    "cache_row_bytes_by_layer", "weight_breakdown", "term_bytes",
    "external_shared_weight_terms", "external_shared_weight_bytes",
]
