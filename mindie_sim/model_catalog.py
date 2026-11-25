"""Catalog loader for the nine se-tools models, measured A5 served checkpoints
and extended architecture presets.

[Role]
  Load the checked-in actual configs and expose independent identities for all
  nine se-tools models plus extended architecture presets (LongCat-Flash-Lite,
  MiniMax-M3, Bailing-V3, Qwen3.5-MoE, HunyuanVideo-DiT).  Validate hybrid
  recurrent (GDN/KDA), cross-layer KV owner sharing, Latent MoE projection
  width, and DiT 3D visual token expansion invariants.

[Invariants]
  * Constants come from config JSON or explicit caller inputs; no model-name
    latency constants and no source-SHA admission gate.
  * ``model.layers`` equals the body-layer count; optional draft/MTP never
    becomes a body layer and never runs on a plain forward.
  * Recurrent/conv state (GDN/KDA/Mamba) is an O(1)-in-seq_len per-sequence
    endpoint; cross-layer KV allocation counts only ``kv_owner_layers``.
  * Missing or invalid dimensions raise ``ModelingError``; zero placeholders are
    not returned for weights.

[Timeouts & Locks]
  Reads local JSON files only; no network, NPU, subprocess or locks.

[Side-Effect Policy]
  No filesystem writes.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.coverage import model_coverage
from mindie_sim.catalog.config import flatten_config
from mindie_sim.catalog.hf_config import model_value_formats
from mindie_sim.catalog.specs import build_model_spec as _build_model_spec
from mindie_sim.catalog.fleet import (
    ARCHITECTURE_PRESET_ALIASES, ARCHITECTURE_PRESET_CATALOG,
    CORE_CATALOG_ENTRIES as CATALOG, CORE_MODEL_ALIASES as MODEL_ALIASES,
    FLEET_CATALOG_ENTRIES, FLEET_MODEL_ALIASES,
)
from mindie_sim.catalog.weights import (
    body_parameter_count, body_weight_bytes, sharded_weight_bytes as _sharded,
    stage_weight_bytes, stage_weight_terms, weight_breakdown,
)

CONFIG_DIR = Path(__file__).with_name("model_configs")

# Raw serving spellings that are not already declared by the fleet registry.
# Kept here so both ``sim`` and ``profiler`` resolve them through one catalog.
MODEL_ALIASES.update({
    "qwen3-30b": "qwen3-30b-a3b",
    "qwen3.8-2.4t": "qwen3.8-2.4t-a95b",
})

_RECURRENT_KINDS = {"gdn", "kda", "mamba"}
_TOKEN_KV_KINDS = {"gqa", "mla", "gqa_sparse_indexer"}
_DIT_ACTIVE_CACHE_METHODS = {"teacache", "fbcache", "taylorseer"}
_DIT_ACTIVE_SPARSE_METHODS = {"topk", "svg"}


def _catalog_key(key: str) -> str:
    """Normalize declared catalog labels while preserving local path spelling."""
    aliases = {**MODEL_ALIASES, **ARCHITECTURE_PRESET_ALIASES}
    if key in aliases:
        return aliases[key]
    normalized = str(key).strip().lower()
    if normalized in aliases:
        return aliases[normalized]
    if normalized in CATALOG or normalized in ARCHITECTURE_PRESET_CATALOG:
        return normalized
    return key


def build_model_spec(key: str, config: dict[str, Any]) -> ModelSpec:
    return _build_model_spec(key, config)


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        key = str(value)
        result[key] = result.get(key, 0) + 1
    return result


def _prune_expert_overrides(
    overrides: tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...],
    num_experts: int,
) -> tuple[tuple[tuple[int, ...], tuple[tuple[str, Any], ...]], ...]:
    kept = []
    for indices, fields in overrides:
        filtered = tuple(int(idx) for idx in indices if 0 <= int(idx) < num_experts)
        if filtered:
            kept.append((filtered, fields))
    return tuple(kept)


def _prune_override_fields(
    fields: tuple[tuple[str, Any], ...], num_experts: int,
) -> tuple[tuple[str, Any], ...]:
    mapping = dict(fields)
    if "expert_quant_overrides" not in mapping and "expert_quant_count" not in mapping:
        return fields
    raw = mapping.get("expert_quant_overrides") or ()
    pruned = _prune_expert_overrides(raw, num_experts)
    if pruned:
        mapping["expert_quant_overrides"] = pruned
        mapping["expert_quant_count"] = int(num_experts)
    else:
        mapping.pop("expert_quant_overrides", None)
        mapping.pop("expert_quant_count", None)
    return tuple(sorted(mapping.items()))


def _prune_raw_expert_overrides(raw: Any, num_experts: int) -> Any:
    if isinstance(raw, Mapping):
        return {k: v for k, v in raw.items() if 0 <= int(k) < num_experts}
    kept = []
    for item in raw:
        if isinstance(item, Mapping):
            ids = [int(i) for i in (item.get("expert_indices") or ()) if 0 <= int(i) < num_experts]
            if ids:
                kept.append({**item, "expert_indices": ids})
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            ids = tuple(int(i) for i in item[0] if 0 <= int(i) < num_experts)
            if ids:
                kept.append((ids, item[1]))
    return kept


def _prune_layer_quant_value(value: Any, num_experts: int) -> Any:
    from mindie_sim.catalog.types import QuantSpec

    if isinstance(value, QuantSpec):
        return _prune_quant_experts(value, num_experts)
    if isinstance(value, Mapping):
        updated = {k: _prune_layer_quant_value(v, num_experts) for k, v in value.items()}
        if "expert_quant_overrides" in updated or "expert_quant_count" in updated:
            pruned = _prune_raw_expert_overrides(updated.get("expert_quant_overrides") or (), num_experts)
            if pruned:
                updated["expert_quant_overrides"] = pruned
                updated["expert_quant_count"] = int(num_experts)
            else:
                updated.pop("expert_quant_overrides", None)
                updated.pop("expert_quant_count", None)
        return updated
    if isinstance(value, list):
        return [_prune_layer_quant_value(v, num_experts) for v in value]
    if isinstance(value, tuple):
        return tuple(_prune_layer_quant_value(v, num_experts) for v in value)
    return value


def _prune_metadata_layer_quant(metadata: tuple[tuple[str, Any], ...], num_experts: int) -> tuple[tuple[str, Any], ...]:
    if not metadata:
        return metadata
    return tuple(
        (k, _prune_layer_quant_value(v, num_experts)) if k == "layer_quant_overrides" else (k, v)
        for k, v in metadata
    )


def _prune_quant_experts(quant: Any, num_experts: int) -> Any:
    """Drop per-expert quant overrides targeting removed experts ``>= num_experts``."""
    from dataclasses import replace

    if quant is None:
        return quant
    pruned_top = _prune_expert_overrides(quant.expert_quant_overrides, num_experts)
    top_count = int(num_experts) if pruned_top else 0
    cat_ov = tuple(
        (cat, _prune_override_fields(fields, num_experts))
        for cat, fields in quant.category_overrides
    )
    layer_ov = (
        _prune_layer_quant_value(quant.layer_quant_overrides_dict(), num_experts)
        if quant.layer_quant_overrides
        else ()
    )
    return replace(
        quant,
        expert_quant_overrides=pruned_top,
        expert_quant_count=top_count,
        category_overrides=cat_ov,
        layer_quant_overrides=layer_ov,
    )


def _prune_moe_plan(
    plan: LayerPlan, e: int, k: int, *, moe_intermediate_size: int | None = None,
) -> LayerPlan:
    from dataclasses import replace

    if plan.ffn != "moe" or plan.num_experts <= 0:
        return plan
    rg, rgt = plan.router_groups, plan.router_group_topk
    if rg > 0:
        if e < rg:
            rg, rgt = e, (min(rgt, e) if rgt > 0 else 1)
        if e % rg != 0 or not 1 <= rgt <= rg or (e // rg) * rgt < k:
            rg, rgt = 0, 0
    m_inter = plan.moe_intermediate_size if moe_intermediate_size is None else moe_intermediate_size
    return replace(
        plan, num_experts=e, top_k=k, moe_intermediate_size=m_inter,
        router_groups=rg, router_group_topk=rgt,
    )


def _prune_draft_spec(
    draft: Any, *, e: int | None, k: int | None,
    moe_intermediate_size: int | None = None,
    body_layers: int | None = None,
) -> Any:
    from dataclasses import replace

    if draft is None:
        return None
    updates: dict[str, Any] = {}
    if body_layers is not None:
        t_len = len(draft.target_layer_ids)
        if t_len:
            t_ids = (
                tuple(i for i in draft.target_layer_ids if i < body_layers)
                if max(draft.target_layer_ids) < body_layers
                else tuple(range(max(0, body_layers - t_len), body_layers))
            )
            updates["target_layer_ids"] = t_ids
    if e is not None and k is not None:
        if draft.num_experts > 0:
            updates["num_experts"] = e
            updates["top_k"] = k
            if moe_intermediate_size is not None:
                updates["expert_intermediate_size"] = moe_intermediate_size
        if draft.layer_plans:
            updates["layer_plans"] = tuple(
                _prune_moe_plan(p, e, k, moe_intermediate_size=moe_intermediate_size)
                for p in draft.layer_plans
            )
    return replace(draft, **updates) if updates else draft


def _refresh_catalog_facts(spec: ModelSpec) -> ModelSpec:
    """Recompute all geometry-dependent ``catalog_facts`` from the final ``ModelSpec``."""
    from mindie_sim.catalog.specs import _model_weight_totals, _set_facts
    from mindie_sim.catalog.weights import (
        cache_row_bytes_by_layer,
        external_shared_weight_bytes,
        external_shared_weight_terms,
    )

    plans = spec.layer_plans
    params, weights, draft_p, draft_w = _model_weight_totals(spec)
    row_bytes = cache_row_bytes_by_layer(spec)
    facts = dict(spec.catalog_facts)
    facts.update({
        "body_layers": len(plans),
        "parameter_count_body": params,
        "resident_weight_bytes_body": weights,
        "parameter_count_optional_draft": draft_p,
        "optional_draft_weight_bytes": draft_w,
        "cache_row_bytes_by_layer": row_bytes,
        "attention_types": _counts(p.architecture or p.attention for p in plans),
        "ffn_types": _counts(p.ffn for p in plans),
        "kv_owner_layers": tuple(int(idx) for idx, b in row_bytes if float(b) > 0.0),
    })
    active_dspark = (
        spec.auxiliary.dspark
        if spec.auxiliary and spec.auxiliary.dspark is not None
        else (spec.auxiliary.mtp if spec.auxiliary else None)
    )
    optional_updates: dict[str, Any] = {
        "mla_layers": sum(1 for p in plans if p.attention == "mla"),
        "gqa_layers": sum(1 for p in plans if p.attention == "gqa"),
        "kda_layers": sum(1 for p in plans if p.attention == "kda"),
        "gdn_layers": sum(1 for p in plans if p.attention == "gdn"),
        "linear_layers": sum(1 for p in plans if p.attention in {"kda", "gdn"}),
        "dense_layers": sum(1 for p in plans if p.ffn == "dense"),
        "moe_layers": sum(1 for p in plans if p.ffn == "moe"),
        "latent_moe_layers": sum(1 for p in plans if p.expert_latent_size > 0),
        "indexer_layers": sum(1 for p in plans if p.indexer),
        "indexer_full_layers": sum(1 for p in plans if p.indexer and p.indexer_type in {"full", "c4"}),
        "indexer_shared_layers": sum(1 for p in plans if p.indexer_type == "shared"),
        "compress_ratios": tuple(int(p.compress_ratio) for p in plans),
        "hc_mult": max((p.hc_mult for p in plans), default=0),
        "hc_lowrank": max((p.hc_lowrank for p in plans), default=0),
        "hc_sinkhorn_iters": max((p.hc_sinkhorn_iters for p in plans), default=0),
        "hyper_connections": any(p.hyper_connections for p in plans),
        "residual_block_size": max((p.residual_block_size for p in plans), default=0),
        "sliding_window": max((p.window for p in plans), default=0),
        "o_groups": max((p.o_groups for p in plans), default=0),
        "o_lora_rank": max((p.o_lora_rank for p in plans), default=0),
        "engram_layer_ids": tuple(t.layer_id for t in (spec.engram.tables if spec.engram else ())),
        "engram_tables": len(spec.engram.tables) if spec.engram else 0,
        "state_bytes_per_request_unsharded": spec.state_bytes_per_request,
        "mtp_layers": spec.auxiliary.mtp.layers if spec.auxiliary and spec.auxiliary.mtp else 0,
        "dspark_layers": spec.auxiliary.dspark.layers if spec.auxiliary and spec.auxiliary.dspark else 0,
        "dspark_target_layer_ids": (
            tuple(active_dspark.target_layer_ids) if active_dspark is not None else ()
        ),
        "dspark_n_routed_experts": (
            int(active_dspark.num_experts) if active_dspark is not None else 0
        ),
        "dspark_num_experts_per_tok": (
            int(active_dspark.top_k) if active_dspark is not None else 0
        ),
        "auxiliary": spec.auxiliary.to_dict() if spec.auxiliary else None,
        "quantization": spec.quant.to_dict() if spec.quant is not None else None,
        "gemma4_k_eq_v_layers": tuple(p.index for p in plans if p.k_eq_v),
        "gemma4_num_kv_shared_layers": sum(
            1 for p in plans if not p.owns_long_kv and p.kv_owner_layer_id != p.index
        ),
        "attention_schedule": tuple("global" if p.window == 0 else "local" for p in plans),
    }
    for key, value in optional_updates.items():
        if key in facts:
            facts[key] = value
    if "external_shared_parameters" in facts:
        facts["external_shared_parameters"] = sum(
            term.logical_parameters for term in external_shared_weight_terms(spec)
        )
    if "external_shared_weight_bytes" in facts:
        facts["external_shared_weight_bytes"] = external_shared_weight_bytes(spec, ParallelSpec())
    _set_facts(spec, list(facts.items()))
    return spec


def derive_expert_pruned(
    base_key: str | ModelSpec, *, num_experts: int, top_k: int,
    moe_intermediate_size: int | None = None, model_id: str | None = None,
) -> ModelSpec:
    """Immutable per-layer expert-prune of a catalog model."""
    from dataclasses import replace
    from mindie_sim.prediction.structure import normalize_model

    base = base_key if isinstance(base_key, ModelSpec) else load_model(base_key)
    base = normalize_model(base)
    e, k = int(num_experts), int(top_k)
    m_inter = int(moe_intermediate_size) if moe_intermediate_size is not None else int(base.moe_intermediate_size)
    if e <= 0 or k <= 0 or k > e or m_inter <= 0:
        raise ModelingError(f"derive_expert_pruned_invalid:E={e}:K={k}:inter={m_inter}")

    plans = tuple(_prune_moe_plan(plan, e, k, moe_intermediate_size=m_inter) for plan in base.layer_plans)
    aux = base.auxiliary
    if aux is not None:
        aux = replace(
            aux,
            mtp=_prune_draft_spec(aux.mtp, e=e, k=k, moe_intermediate_size=m_inter),
            dspark=_prune_draft_spec(aux.dspark, e=e, k=k, moe_intermediate_size=m_inter),
        )
    keep = len(base.expert_routing) == e
    spec = replace(
        base, name=str(model_id) if model_id is not None else base.name,
        layer_plans=plans, num_experts=e, top_k=k, moe_intermediate_size=m_inter,
        quant=_prune_quant_experts(base.quant, e),
        metadata=_prune_metadata_layer_quant(base.metadata, e),
        expert_routing=base.expert_routing if keep else (),
        expert_routing_source=base.expert_routing_source if keep else "uniform", auxiliary=aux,
    )
    return _refresh_catalog_facts(spec)


def customize_model_spec(
    model: str | Mapping[str, Any] | ModelSpec, *,
    layers: int | None = None,
    n_routed_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    promote_dense_to_moe: bool = False,
) -> ModelSpec:
    """Deterministically scale body layer count and MoE expert geometry."""
    from dataclasses import replace
    from mindie_sim.prediction.structure import normalize_model

    base = (model if isinstance(model, ModelSpec)
            else ModelSpec.from_mapping(model) if isinstance(model, Mapping)
            else load_model(str(model)))
    base = normalize_model(base)
    plans = list(base.layer_plans)
    if layers is not None:
        target_l = int(layers)
        if target_l <= 0:
            raise ModelingError(f"customize_model_spec_layers_invalid:{target_l}")
        if target_l <= len(plans):
            plans = plans[:target_l]
        else:
            cycle_pool = [p for p in plans if p.ffn == "moe"] or plans
            while len(plans) < target_l:
                plans.append(cycle_pool[(len(plans) - len(base.layer_plans)) % len(cycle_pool)])
        plans = [
            replace(
                p, index=idx, native_layer_id=idx,
                kv_owner_layer_id=idx if p.kv_owner_layer_id == p.index else (min(p.kv_owner_layer_id, idx) if p.kv_owner_layer_id >= 0 else -1),
                index_k_owner_layer_id=idx if p.index_k_owner_layer_id == p.index else (min(p.index_k_owner_layer_id, idx) if p.index_k_owner_layer_id >= 0 else -1),
                topk_owner_layer_id=idx if p.topk_owner_layer_id == p.index else (min(p.topk_owner_layer_id, idx) if p.topk_owner_layer_id >= 0 else -1),
                candidate_source_layer_id=p.candidate_source_layer_id if p.candidate_source_layer_id < idx else -1,
                index_source_layer_id=idx if p.index_source_layer_id == p.index else (min(p.index_source_layer_id, idx) if p.index_source_layer_id >= 0 else -1),
            )
            for idx, p in enumerate(plans)
        ]

    e = int(n_routed_experts) if n_routed_experts is not None else int(base.num_experts)
    k = int(num_experts_per_tok) if num_experts_per_tok is not None else int(base.top_k)
    # A runtime MoE override on a dense preset promotes the FFN layers once.
    # An already-MoE model is never re-promoted from its dense prefix.
    promoted = bool(
        promote_dense_to_moe
        and n_routed_experts is not None
        and num_experts_per_tok is not None
        and e > 1 and k > 0
        and not any(p.ffn == "moe" for p in plans)
    )
    if promoted:
        if k > e:
            raise ModelingError(f"customize_model_spec_moe_invalid:E={e}:K={k}")
        plans = [
            replace(
                p, ffn="moe", num_experts=e, top_k=k,
                moe_intermediate_size=(
                    p.moe_intermediate_size
                    or base.moe_intermediate_size
                    or p.intermediate_size
                ),
            )
            for p in plans
        ]
    moe_updated = n_routed_experts is not None or num_experts_per_tok is not None
    if moe_updated:
        if e <= 0 or k <= 0 or k > e:
            raise ModelingError(f"customize_model_spec_moe_invalid:E={e}:K={k}")
        plans = [_prune_moe_plan(p, e, k) for p in plans]

    n_l = len(plans)
    aux = base.auxiliary
    if aux is not None:
        aux = replace(
            aux,
            mtp=_prune_draft_spec(
                aux.mtp, e=e if moe_updated else None, k=k if moe_updated else None, body_layers=n_l,
            ),
            dspark=_prune_draft_spec(
                aux.dspark, e=e if moe_updated else None, k=k if moe_updated else None, body_layers=n_l,
            ),
        )
    engram = base.engram
    if engram is not None and layers is not None:
        tables = tuple(t for t in engram.tables if t.layer_id < n_l)
        engram = replace(engram, tables=tables) if tables else None
        plans = [replace(p, engram=any(t.layer_id == p.index for t in (engram.tables if engram else ()))) for p in plans]
    keep_routing = len(base.expert_routing) == e
    has_moe = any(p.ffn == "moe" for p in plans)
    quant = _prune_quant_experts(base.quant, e) if (moe_updated and e > 0) else base.quant
    meta = _prune_metadata_layer_quant(base.metadata, e) if (moe_updated and e > 0) else base.metadata
    spec = replace(
        base, layers=n_l, body_layers=n_l, layer_plans=tuple(plans),
        ffn="moe" if promoted else base.ffn,
        moe_intermediate_size=(
            plans[0].moe_intermediate_size
            if promoted else base.moe_intermediate_size
        ),
        num_experts=e if has_moe else 0, top_k=k if has_moe else 0,
        quant=quant,
        metadata=meta,
        expert_routing=base.expert_routing if keep_routing else (),
        expert_routing_source=base.expert_routing_source if keep_routing else "uniform",
        state_bytes_per_request=float(sum(p.state_bytes_total for p in plans)),
        auxiliary=aux, engram=engram,
    )
    return _refresh_catalog_facts(spec)


@lru_cache(maxsize=None)
def _load_builtin_model(key: str) -> ModelSpec:
    entries = {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}
    path = CONFIG_DIR / entries[key]
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"model_config_unreadable:{key}:{type(exc).__name__}") from exc
    return _build_model_spec(key, config)


def load_model(key: str, *, config_root: str | Path | None = None) -> ModelSpec:
    key = str(key)
    key = _catalog_key(key)
    entries = {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}
    if config_root is None and key in entries:
        return _load_builtin_model(key)
    root = Path(config_root) if config_root is not None else CONFIG_DIR
    path = root / entries[key] if key in entries else Path(key).expanduser()
    if path.is_dir():
        path = path / "config.json"
    if key in entries and config_root is not None and not path.is_file():
        name = Path(entries[key]).name
        fallback = next((p for p in (root / name, Path(__file__).parent / "catalog" / "configs" / name)
                         if p.is_file()), None)
        if fallback is not None:
            path = fallback
    if key not in entries and not path.is_file():
        import difflib
        pool = sorted(set(entries) | set(MODEL_ALIASES) | set(ARCHITECTURE_PRESET_ALIASES))
        matches = difflib.get_close_matches(key.lower(), pool, n=3, cutoff=0.45) or sorted(CATALOG)[:3]
        raise ModelingError(
            f"model_catalog_unknown:{key} — did you mean {', '.join(matches)}? "
            "Or pass a path to HuggingFace config.json or model directory"
        )
    if path.stat().st_size > 50 * 1024 * 1024:
        raise ModelingError(f"model_config_size_exceeds_metadata_limit:{key}")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except TimeoutError:
        raise
    except OSError as exc:
        raise ModelingError(f"model_config_unreadable:{key}:{type(exc).__name__}") from exc
    return _build_model_spec(key, config)

_DEFAULT_NODE_RANKS = 1
def sharded_weight_bytes(
    model: ModelSpec, parallel: ParallelSpec, node_ranks: int = _DEFAULT_NODE_RANKS, *,
    include_draft: bool = False, include_engram: bool = True,
) -> float:
    """Largest actual PP-stage per-rank weight footprint."""
    parallel.validate(model)
    from mindie_sim.prediction.structure import normalize_model
    return _sharded(normalize_model(model), parallel, node_ranks, include_draft=include_draft, include_engram=include_engram)


def model_weight_detail(model: ModelSpec, parallel: ParallelSpec, node_ranks: int = 1) -> dict[str, Any]:
    stages = stage_weight_bytes(model, parallel, node_ranks)
    stages_draft = stage_weight_bytes(model, parallel, node_ranks, include_draft=True)
    return {
        "stage_bytes": stages, "stage_bytes_with_optional_draft": stages_draft,
        "largest_stage_bytes": max(stages) if stages else 0.0,
        "largest_stage_with_draft_bytes": max(stages_draft) if stages_draft else 0.0,
        "largest_stage": max(range(len(stages)), key=lambda i: stages[i]) if stages else 0,
        "category_bytes": weight_breakdown(model, parallel, node_ranks),
        "logical_body_parameters": body_parameter_count(model),
        "logical_body_weight_bytes": body_weight_bytes(model), "node_ranks": node_ranks,
    }


def list_models() -> list[dict[str, Any]]:
    rows = []
    for key in {**CATALOG, **ARCHITECTURE_PRESET_CATALOG}:
        model = load_model(key)
        facts = dict(model.catalog_facts)
        rows.append({
            "key": key, "name": model.name, "layers": model.layers, "body_layers": model.body_layers,
            "hidden_size": model.hidden_size, "heads": model.num_heads, "experts": model.num_experts, "top_k": model.top_k,
            "parameter_count": model.parameter_count(), "resident_weight_bytes": body_weight_bytes(model),
            "attention_types": facts.get("attention_types", {}), "ffn_types": facts.get("ffn_types", {}),
            "has_optional_draft": model.has_optional_draft, "has_vision": model.vision is not None,
            "engram_tables": facts.get("engram_tables", 0), "facts": facts, **model_coverage(model),
        })
    return rows


__all__ = [
    "CATALOG", "MODEL_ALIASES", "ARCHITECTURE_PRESET_CATALOG", "ARCHITECTURE_PRESET_ALIASES", "CONFIG_DIR", "LayerPlan",
        "derive_expert_pruned", "customize_model_spec", "load_model", "list_models",
    "sharded_weight_bytes", "model_weight_detail", "stage_weight_terms", "body_weight_bytes", "body_parameter_count",
]
