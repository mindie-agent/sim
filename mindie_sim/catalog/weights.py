"""Cohesive per-layer/per-stage weight formulas for catalog models.

[Role]
  Assemble model-wide and per-PP-stage logical weight terms from per-layer
  builders in ``weight_layer_terms`` and sum source/runtime weight bytes under
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
from typing import Any, Mapping, Sequence

from mindie_sim.catalog.draft_weights import draft_terms, draft_weight_summary
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_layer_terms import (
    _attention_terms,
    _compressor_terms,
    _ffn_terms,
    _gemma4_norm_terms,
    _indexer_terms,
    _layer_resolved_quant,
    _mhc_terms,
    _norm_terms,
    _tag_projection_roles,
    gemma4_ple_layer_terms,
    gemma4_ple_terms,
    layer_weight_terms,
)
from mindie_sim.catalog.weight_terms import (
    WeightTerm,
    _ceil_div,
    _local_groups,
    _sharded_shape,
    _t,
    term_bytes,
    term_payload_bytes,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

MHC_HEAD_STREAMS = 4
TARGET_ALIAS_CATEGORIES = frozenset({
    "target_alias_embedding", "target_alias_lm_head",
})


def _resident_parallel(
    term: WeightTerm, parallel: ParallelSpec, shared_expert_dp: bool,
    *, pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
) -> ParallelSpec:
    """Placement used for one resident term.

    Shared-expert DP executes those matrices with ``tp=1``. Each rank keeps
    the full shared-expert copy instead of a TP shard.  When
    ``enable_pcp_o_proj_weight_sharding`` is enabled with ``pcp > 1``, attention
    output weights (``o_proj`` / ``wo_a`` / ``wo_b``) are sharded across PCP.
    When ``dcp_q_replicate`` is active, dense MLA Q projection weights
    (``mla.q_b`` / ``mla.q_proj``) replicate across ``dcp_tp_gather``.
    """
    placed = parallel
    if shared_expert_dp and term.category == "shared_expert":
        placed = replace(placed, tp=1)
    if (dcp_q_replicate and term.category in {"mla.q_b", "mla.q_proj"}
            and int(placed.dcp_tp_gather) > 1
            and int(placed.dcp_pcp_factor) == 1
            and int(placed.dcp_dp_width) == 1):
        eff_tp = max(1, int(placed.tp) // int(placed.dcp_tp_gather))
        placed = replace(placed, tp=eff_tp, q_proj_tp=1, dcp_linear_tp=1)
    if (pcp_o_proj_weight_sharding and int(placed.pcp) > 1
            and term.placement_role == "attention_output"):
        base_width = max(int(placed.tp), int(placed.effective_o_proj_tp))
        placed = replace(placed, dcp_linear_tp=1,
                         o_proj_tp=base_width * int(placed.pcp))
    return placed


def _expand_resident_terms(
    term: WeightTerm, parallel: ParallelSpec, shared_expert_dp: bool,
    *, pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
    model: ModelSpec | None = None,
) -> tuple[tuple[WeightTerm, ParallelSpec], ...]:
    """Return placed ``(term, parallel)`` pairs, splitting ``W_UK_T`` when replicated under DCP."""
    placed = _resident_parallel(
        term, parallel, shared_expert_dp,
        pcp_o_proj_weight_sharding=pcp_o_proj_weight_sharding,
        dcp_q_replicate=dcp_q_replicate,
    )
    if (dcp_q_replicate and term.name == "kv_b" and term.category == "mla_absorb"
            and int(parallel.dcp_tp_gather) > 1
            and int(parallel.dcp_pcp_factor) == 1
            and int(parallel.dcp_dp_width) == 1):
        uk_rows = term.rows // 2
        if model is not None:
            mla_plan = next((lp for lp in model.layer_plans if lp.attention == "mla"), None)
            if mla_plan is not None and (mla_plan.qk_nope_head_dim + mla_plan.v_head_dim) > 0:
                uk_rows = int(mla_plan.num_heads) * int(mla_plan.qk_nope_head_dim)
        uv_rows = max(0, term.rows - uk_rows)
        if uk_rows > 0 and uv_rows > 0:
            eff_tp = max(1, int(placed.tp) // int(parallel.dcp_tp_gather))
            placed_uk = replace(placed, tp=eff_tp, q_proj_tp=1, dcp_linear_tp=1)
            return (
                (replace(term, name="w_uk_t", rows=uk_rows), placed_uk),
                (replace(term, name="w_uv", rows=uv_rows), placed),
            )
    return ((term, placed),)


def _special_layer_keys(model: ModelSpec) -> tuple[set[str], set[str]]:
    meta = dict(model.metadata)
    by_index: set[str] = set()
    by_native: set[str] = set()
    for name in ("prepared_mx32_matrix", "mla_head_padding",
                 "kda_input_projection", "causal_conv_update",
                 "layer_quant_overrides"):
        block = meta.get(name)
        if isinstance(block, Mapping):
            layers = block.get("layers")
            if isinstance(layers, Mapping):
                by_index.update(str(k) for k in layers)
            elif name == "layer_quant_overrides":
                by_index.update(str(k) for k in block if k != "native_layers")
            native_layers = block.get("native_layers")
            if isinstance(native_layers, Mapping):
                by_native.update(str(k) for k in native_layers)
    quant = getattr(model, "quant", None)
    if quant is not None and getattr(quant, "layer_quant_overrides", ()):
        q_block = quant.layer_quant_overrides_dict()
        q_layers = q_block.get("layers")
        if isinstance(q_layers, Mapping):
            by_index.update(str(k) for k in q_layers)
        else:
            by_index.update(str(k) for k in q_block if k != "native_layers")
        q_native = q_block.get("native_layers")
        if isinstance(q_native, Mapping):
            by_native.update(str(k) for k in q_native)
    return by_index, by_native


def _cache_key(obj: Any) -> Any:
    try:
        hash(obj)
        return obj
    except TypeError:
        return None


def _iter_model_layer_terms(model: ModelSpec):
    special_idx, special_native = _special_layer_keys(model)
    memo: dict[Any, tuple[WeightTerm, ...]] = {}
    for plan in model.layer_plans:
        native_id = int(getattr(plan, "native_layer_id", -1))
        if str(plan.index) in special_idx or (native_id >= 0 and str(native_id) in special_native):
            yield plan.index, tuple(layer_weight_terms(plan, model))
            continue
        shares_kv = 0 <= plan.kv_owner_layer_id < plan.index
        layer_quant = _layer_resolved_quant(model, plan)
        ck = _cache_key((replace(plan, index=0, native_layer_id=0), shares_kv, layer_quant))
        cached = memo.get(ck) if ck is not None else None
        if cached is None:
            cached = tuple(layer_weight_terms(plan, model))
            if ck is not None:
                memo[ck] = cached
        yield plan.index, cached


def stage_weight_decomposition(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool | None = None,
    include_vision: bool | None = None,
    layer_partition: Sequence[int] | None = None,
    enable_shared_expert_dp: bool = False,
    enable_pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
) -> list[dict[str, float]]:
    """Per-PP-stage weight bytes split into payload and quant metadata."""
    quant = model.quant or QuantSpec("bf16")
    stages: list[dict[str, float]] = []
    pair_memo: dict[Any, tuple[float, float]] = {}
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft, include_engram=include_engram,
        include_engram_tables=include_engram_tables, include_vision=include_vision,
        layer_partition=layer_partition,
    ):
        total = 0.0
        payload = 0.0
        for term in terms:
            ck = _cache_key(term)
            pair = pair_memo.get(ck) if ck is not None else None
            if pair is None:
                expanded = _expand_resident_terms(
                    term, parallel, enable_shared_expert_dp,
                    pcp_o_proj_weight_sharding=enable_pcp_o_proj_weight_sharding,
                    dcp_q_replicate=dcp_q_replicate, model=model,
                )
                pair = (
                    sum(term_bytes(t_sub, quant, p_sub, node_ranks) for t_sub, p_sub in expanded),
                    sum(term_payload_bytes(t_sub, quant, p_sub, node_ranks) for t_sub, p_sub in expanded),
                )
                if ck is not None:
                    pair_memo[ck] = pair
            total += pair[0]
            payload += pair[1]
        stages.append({"total_bytes": float(total), "payload_bytes": float(payload),
                       "quant_metadata_bytes": float(total - payload)})
    return stages


def vision_weight_terms(
    model: ModelSpec, *, include_vision: bool | None = None,
) -> list[WeightTerm]:
    vision = model.vision
    if vision is None:
        return []
    if include_vision is False:
        return []
    if include_vision is None and (model.auxiliary is not None and not model.auxiliary.include_vision_weights):
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


def _output_res_quant(model: ModelSpec) -> QuantSpec:
    base = model.quant or QuantSpec("bf16")
    explicit = {key: dict(val) for key, val in base.category_overrides}
    if "attn_res_proj" in explicit:
        return base
    merged = dict(explicit)
    merged["attn_res_proj"] = {
        "weight_format": base.native_format,
        "activation_format": base.native_format,
    }
    return replace(base, category_overrides=merged)


def _ngram_embedding_terms(model: ModelSpec) -> list[WeightTerm]:
    """LongCat-Flash-Lite packed N-gram embedding tables (BF16, out_tp sharded)."""
    cfg = dict(getattr(model, "metadata", ()) or ()).get("ngram_embedding")
    if not isinstance(cfg, Mapping):
        return []
    ratio = int(cfg.get("ngram_vocab_size_ratio") or 0)
    neighbors = int(cfg.get("emb_neighbor_num") or 0)
    splits = int(cfg.get("emb_split_num") or 0)
    if ratio <= 0 or neighbors <= 1 or splits <= 0:
        return []
    num_tables = splits * (neighbors - 1)
    table_width = max(1, int(model.hidden_size) // num_tables)
    base_rows = int(model.vocab_size) * ratio
    total_rows = sum(base_rows + 2 * i + 1 for i in range(num_tables))
    return [_t("ngram_embedding", total_rows, table_width, "embedding", "out_tp")]


def body_head_terms(model: ModelSpec) -> list[WeightTerm]:
    h = model.hidden_size
    lowrank_hc = any(plan.hc_lowrank > 0 for plan in model.layer_plans)
    terms = [] if lowrank_hc else [_t("final_norm", h, 1, "norm")]
    if not model.is_standalone_draft:
        terms.insert(0, _t("embedding", model.vocab_size, h, "embedding", "embedding"))
        terms.extend(_ngram_embedding_terms(model))
        if not model.tie_word_embeddings:
            terms.append(replace(_t("lm_head", model.vocab_size, h, "lm_head", "out_tp"), placement_role="logits"))
    if any(plan.residual_block_size for plan in model.layer_plans):
        res_quant = _output_res_quant(model)
        terms.extend([
            _t("output_attn_res_norm", h, 1, "norm"),
            replace(_t("output_attn_res_proj", 1, h, "attn_res_proj"), quant=res_quant),
        ])
    if lowrank_hc:
        from mindie_sim.catalog.qwen4_exp import lowrank_hc_head_terms
        terms.extend(lowrank_hc_head_terms(model))
    elif any(plan.hyper_connections for plan in model.layer_plans):
        terms.extend([
            _t("hc_head_fn", MHC_HEAD_STREAMS, MHC_HEAD_STREAMS * h, "mhc"),
            _t("hc_head_base", MHC_HEAD_STREAMS, 1, "param_fp32"),
            _t("hc_head_scale", 1, 1, "param_fp32"),
        ])
    return terms


def engram_table_terms(model: ModelSpec) -> list[WeightTerm]:
    engram = model.engram
    if engram is None:
        return []
    from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_table_terms
    if is_qwen4_exp_ple(engram):
        return ple_table_terms(model)
    return [WeightTerm(f"engram_table_{table.table_id}", table.rows, table.width, "engram_table", "node", 1)
            for table in engram.tables]


def engram_gate_terms_by_layer(model: ModelSpec) -> dict[int, list[WeightTerm]]:
    engram = model.engram
    if engram is None:
        return {}
    from mindie_sim.catalog.qwen4_exp import is_qwen4_exp_ple, ple_gate_terms_by_layer
    if is_qwen4_exp_ple(engram):
        raw_ple = ple_gate_terms_by_layer(model)
        plans_by_idx = {int(p.index): p for p in model.layer_plans}
        resolved_ple: dict[int, list[WeightTerm]] = {}
        for layer_id, layer_terms in raw_ple.items():
            plan = plans_by_idx.get(int(layer_id))
            lquant = _layer_resolved_quant(model, plan) if plan is not None else (model.quant or QuantSpec("bf16"))
            resolved_ple[int(layer_id)] = [
                replace(term, quant=term.quant or lquant) for term in layer_terms
            ]
        return resolved_ple
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


def _body_terms(model: ModelSpec, *, include_engram: bool = True) -> list[WeightTerm]:
    terms: list[WeightTerm] = []
    for _, layer_terms in _iter_model_layer_terms(model):
        terms.extend(layer_terms)
    terms.extend(body_head_terms(model))
    terms.extend(gemma4_ple_terms(model))
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
    include_vision: bool | None = None,
    layer_partition: Sequence[int] | None = None,
    draft_kind: str | None = None,
    slot_plans: Mapping[int, Any] | None = None,
    ep_rank: int | None = None,
) -> list[list[WeightTerm]]:
    pp = max(1, parallel.pp)
    if layer_partition is None:
        if model.layers % pp != 0:
            raise ModelingError(f"layers_not_divisible_by_pp:{model.layers}%{pp}")
        stage_of = lambda layer_id: _stage_of(layer_id, pp, model.layers)
    else:
        partition = tuple(int(value) for value in layer_partition)
        if len(partition) != pp:
            raise ModelingError(f"pp_partition_count:{len(partition)}!={pp}")
        if sum(partition) != model.layers:
            raise ModelingError(
                f"pp_partition_layers:{sum(partition)}!={model.layers}"
            )
        stage_of = _partition_stage_of(partition)
    stages: list[list[WeightTerm]] = [[] for _ in range(pp)]
    for layer_idx, layer_terms in _iter_model_layer_terms(model):
        terms: Sequence[WeightTerm] = layer_terms
        slots = (slot_plans or {}).get(layer_idx)
        if slots is not None:
            from mindie_sim.catalog.expert_slots import routed_expert_weight_terms
            terms = routed_expert_weight_terms(
                layer_terms, slots, ep_rank=0 if ep_rank is None else int(ep_rank))
        stages[stage_of(layer_idx)].extend(terms)
    for term in body_head_terms(model):
        stages[0 if term.name == "embedding" else pp - 1].append(term)
    if model.vision is not None:
        stages[0].extend(vision_weight_terms(model, include_vision=include_vision))
    tables_in_hbm = include_engram if include_engram_tables is None else include_engram_tables
    if include_engram:
        if tables_in_hbm:
            tables = engram_table_terms(model)
            for stage in stages:
                stage.extend(tables)
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
                    if term.category not in TARGET_ALIAS_CATEGORIES
                )
    return stages


def stage_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool | None = None,
    include_vision: bool | None = None,
    source: bool = False, layer_partition: Sequence[int] | None = None,
    draft_kind: str | None = None, enable_shared_expert_dp: bool = False,
    enable_pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
    slot_plans: Mapping[int, Any] | None = None, ep_rank: int | None = None,
) -> list[float]:
    quant = model.quant or QuantSpec("bf16")
    result: list[float] = []
    byte_memo: dict[Any, float] = {}
    for terms in stage_weight_terms(
        model, parallel, include_draft=include_draft, draft_kind=draft_kind,
        include_engram=include_engram, include_engram_tables=include_engram_tables,
        include_vision=include_vision, layer_partition=layer_partition,
        slot_plans=slot_plans, ep_rank=ep_rank,
    ):
        stage_total = 0.0
        for term in terms:
            ck = _cache_key(term)
            val = byte_memo.get(ck) if ck is not None else None
            if val is None:
                expanded = _expand_resident_terms(
                    term, parallel, enable_shared_expert_dp,
                    pcp_o_proj_weight_sharding=enable_pcp_o_proj_weight_sharding,
                    dcp_q_replicate=dcp_q_replicate, model=model,
                )
                val = sum(term_bytes(t_sub, quant, p_sub, node_ranks, source=source)
                          for t_sub, p_sub in expanded)
                if ck is not None:
                    byte_memo[ck] = val
            stage_total += val
        result.append(float(stage_total))
    return result


def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_draft: bool = False, include_engram: bool = True,
    include_engram_tables: bool = True, include_vision: bool | None = None,
    enable_shared_expert_dp: bool = False,
    enable_pcp_o_proj_weight_sharding: bool = False,
    dcp_q_replicate: bool = False,
) -> float:
    return float(max(stage_weight_bytes(
        model, parallel, node_ranks, include_draft=include_draft,
        include_engram=include_engram,
        include_engram_tables=include_engram_tables,
        include_vision=include_vision,
        enable_shared_expert_dp=enable_shared_expert_dp,
        enable_pcp_o_proj_weight_sharding=enable_pcp_o_proj_weight_sharding,
        dcp_q_replicate=dcp_q_replicate,
    )))


def cache_row_bytes_by_layer(model: ModelSpec) -> tuple[tuple[int, int], ...]:
    """Per-layer cache row width using each layer's resolved cache ABI."""
    from mindie_sim.catalog.cache import packed_region_row_bytes
    from mindie_sim.catalog.cache_format import gqa_kv_formats
    from mindie_sim.catalog.cache_pools import _packed_mla_row_bytes
    rows = []
    for plan in model.layer_plans:
        if (plan.attention in {"gqa", "projection_only"} and plan.cache_kind == "dense"
                and not 0 <= plan.kv_owner_layer_id < plan.index):
            key_fmt, val_fmt, is_shared = gqa_kv_formats(model, plan=plan)
            if is_shared:
                row = key_fmt.row_bytes(plan.cache_row_elements)
            else:
                side = int(plan.num_kv_heads) * int(plan.head_dim)
                row = key_fmt.row_bytes(side) + val_fmt.row_bytes(side)
        elif plan.attention == "mla":
            if plan.cache_kind == "dense":
                row = _packed_mla_row_bytes(model, plan)
            elif plan.window > 0:
                row = packed_region_row_bytes(
                    model, "window", plan.cache_row_elements, plan=plan,
                )
            elif plan.owns_long_kv:
                row = packed_region_row_bytes(
                    model, "compressed_kv", plan.cache_row_elements, plan=plan,
                )
            else:
                row = 0
        else:
            row = 0
        rows.append((plan.index, int(row)))
    return tuple(rows)


def weight_breakdown(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1, *,
    include_engram_tables: bool | None = None,
    layer_partition: Sequence[int] | None = None,
    enable_shared_expert_dp: bool = False,
) -> dict[str, float]:
    quant = model.quant or QuantSpec("bf16")
    groups: dict[str, float] = {}
    byte_memo: dict[Any, float] = {}
    for terms in stage_weight_terms(
        model, parallel, include_engram=True,
        include_engram_tables=include_engram_tables, layer_partition=layer_partition,
    ):
        for term in terms:
            ck = _cache_key(term)
            val = byte_memo.get(ck) if ck is not None else None
            if val is None:
                placed = _resident_parallel(term, parallel, enable_shared_expert_dp)
                val = term_bytes(term, quant, placed, node_ranks)
                if ck is not None:
                    byte_memo[ck] = val
            groups[term.category] = groups.get(term.category, 0.0) + val
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
    "gemma4_ple_terms", "gemma4_ple_layer_terms",
    "_norm_terms", "_gemma4_norm_terms", "_mhc_terms", "_indexer_terms",
    "_compressor_terms", "_attention_terms", "_ffn_terms",
    "_tag_projection_roles", "_ceil_div", "_local_groups", "_sharded_shape",
]
