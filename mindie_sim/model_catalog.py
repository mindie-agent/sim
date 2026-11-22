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


def derive_expert_pruned(
    base_key: str | ModelSpec, *, num_experts: int, top_k: int,
    moe_intermediate_size: int | None = None, model_id: str | None = None,
) -> ModelSpec:
    """Immutable per-layer expert-prune of a catalog model."""
    from dataclasses import replace

    base = base_key if isinstance(base_key, ModelSpec) else load_model(base_key)
    e, k = int(num_experts), int(top_k)
    m_inter = int(moe_intermediate_size) if moe_intermediate_size is not None else int(base.moe_intermediate_size)
    if e <= 0 or k <= 0 or k > e or m_inter <= 0:
        raise ModelingError(f"derive_expert_pruned_invalid:E={e}:K={k}:inter={m_inter}")

    def _prune_plan(plan: LayerPlan) -> LayerPlan:
        if plan.ffn != "moe":
            return plan
        rg, rgt = plan.router_groups, plan.router_group_topk
        if rg > 0 and (e % rg != 0 or not 1 <= rgt <= rg or (e // rg) * rgt < k):
            rg, rgt = 0, 0
        return replace(plan, num_experts=e, top_k=k, moe_intermediate_size=m_inter, router_groups=rg, router_group_topk=rgt)

    plans = tuple(_prune_plan(plan) for plan in base.layer_plans)
    aux = base.auxiliary
    if aux is not None:
        mtp = replace(aux.mtp, num_experts=e, top_k=k, expert_intermediate_size=m_inter) if aux.mtp and aux.mtp.num_experts > 0 else aux.mtp
        dspark = replace(aux.dspark, num_experts=e, top_k=k, expert_intermediate_size=m_inter) if aux.dspark and aux.dspark.num_experts > 0 else aux.dspark
        aux = replace(aux, mtp=mtp, dspark=dspark)
    keep = len(base.expert_routing) == e
    return replace(
        base, name=str(model_id) if model_id is not None else base.name,
        layer_plans=plans, num_experts=e, top_k=k, moe_intermediate_size=m_inter,
        expert_routing=base.expert_routing if keep else (),
        expert_routing_source=base.expert_routing_source if keep else "uniform", auxiliary=aux,
    )


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
    from mindie_sim.catalog.specs import _set_facts, _model_weight_totals
    from mindie_sim.catalog.weights import cache_row_bytes_by_layer

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
    if n_routed_experts is not None or num_experts_per_tok is not None:
        if e <= 0 or k <= 0 or k > e:
            raise ModelingError(f"customize_model_spec_moe_invalid:E={e}:K={k}")
        for idx, p in enumerate(plans):
            if p.ffn == "moe" and p.num_experts > 0:
                rg, rgt = p.router_groups, p.router_group_topk
                if rg > 0:
                    if e < rg:
                        rg, rgt = e, (min(rgt, e) if rgt > 0 else 1)
                    if e % rg != 0 or not 1 <= rgt <= rg or (e // rg) * rgt < k:
                        rg, rgt = 0, 0
                plans[idx] = replace(p, num_experts=e, top_k=k, router_groups=rg, router_group_topk=rgt)

    n_l = len(plans)
    aux = base.auxiliary
    if aux is not None:
        def _adj_draft(d):
            if d is None:
                return None
            t_len = len(d.target_layer_ids)
            t_ids = (tuple(i for i in d.target_layer_ids if i < n_l)
                     if t_len and max(d.target_layer_ids) < n_l
                     else tuple(range(max(0, n_l - t_len), n_l)) if t_len else ())
            return replace(d, num_experts=e if d.num_experts > 0 else 0,
                           top_k=k if d.num_experts > 0 else 0, target_layer_ids=t_ids)
        aux = replace(aux, mtp=_adj_draft(aux.mtp), dspark=_adj_draft(aux.dspark))
    engram = base.engram
    if engram is not None and layers is not None:
        tables = tuple(t for t in engram.tables if t.layer_id < n_l)
        engram = replace(engram, tables=tables) if tables else None
        plans = [replace(p, engram=any(t.layer_id == p.index for t in (engram.tables if engram else ()))) for p in plans]
    keep_routing = len(base.expert_routing) == e
    has_moe = any(p.ffn == "moe" for p in plans)
    spec = replace(
        base, layers=n_l, body_layers=n_l, layer_plans=tuple(plans),
        ffn="moe" if promoted else base.ffn,
        moe_intermediate_size=(
            plans[0].moe_intermediate_size
            if promoted else base.moe_intermediate_size
        ),
        num_experts=e if has_moe else 0, top_k=k if has_moe else 0,
        expert_routing=base.expert_routing if keep_routing else (),
        expert_routing_source=base.expert_routing_source if keep_routing else "uniform",
        state_bytes_per_request=float(sum(p.state_bytes_total for p in plans)),
        auxiliary=aux, engram=engram,
    )
    params, weights, draft_p, draft_w = _model_weight_totals(spec)
    facts = dict(spec.catalog_facts)
    facts.update({
        "body_layers": n_l, "parameter_count_body": params, "resident_weight_bytes_body": weights,
        "parameter_count_optional_draft": draft_p, "optional_draft_weight_bytes": draft_w,
        "cache_row_bytes_by_layer": cache_row_bytes_by_layer(spec),
        "kv_owner_layers": tuple(p.index for p in plans if p.owns_long_kv or (p.kv_owner_layer_id == p.index and p.compress_ratio == 0)),
    })
    _set_facts(spec, list(facts.items()))
    return spec


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
