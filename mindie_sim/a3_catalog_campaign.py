"""Normalized operator-campaign records extracted from real catalog OpSpecs.

[Role]
  Give the measurement campaign a stable, model-derived view of every op:
  model/layer/primitive/category, semantic and launch shapes, dtype and packed
  weight/compute precision facts, owner/cache metadata and phase.

[Invariants]
  * Primitive/category come from OpSpec metadata and the actual LayerPlan, not
    from model-name switches or family-string guesses.
  * No latency replay or model-specific coefficient lives here.
  * Semantic, local, padded and launch shapes remain distinct.
  * Packed weight bytes and weight/activation/compute formats are carried
    separately so campaign measurements can distinguish precision paths.

[Timeouts & Locks]
  Pure record extraction; no I/O.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

from typing import Any, Iterable

from .a3_model_types import ModelSpec
from .a3_ops import OpSpec


def _meta(op: OpSpec) -> dict[str, Any]:
    return dict(op.metadata or ())


def op_campaign_record(
    model: ModelSpec, op: OpSpec, *, layer_plan: Any | None = None,
) -> dict[str, Any]:
    meta = _meta(op)
    plan = layer_plan
    category = "other"
    if plan is not None:
        category = f"{getattr(plan, 'architecture', getattr(plan, 'attention', 'other'))}/" \
                   f"{getattr(plan, 'ffn', 'other')}"
    if meta.get("quant_category"):
        category = f"{category}/{meta['quant_category']}"
    return {
        "model_key": model.name,
        "op_id": op.op_id,
        "layer": int(op.layer),
        "stage": int(op.stage),
        "phase": op.phase,
        "requests": list(op.requests),
        "family": op.family,
        "recipe": op.recipe,
        "primitive": str(meta.get("semantic_family", op.family)),
        "category": category,
        "attention_kind": getattr(plan, "attention", None) if plan is not None else None,
        "ffn_kind": getattr(plan, "ffn", None) if plan is not None else None,
        "semantic_shape": list(op.semantic_shape),
        "local_shape": list(op.local_shape),
        "padded_shape": list(op.padded_shape),
        "launch_shape": list(op.launch_shape),
        "flops": float(op.flops),
        "hbm_bytes": float(op.hbm_bytes),
        "link_bytes": float(op.link_bytes),
        "dtype": op.dtype,
        "layout": op.layout,
        "weight_format": meta.get("weight_format", ""),
        "activation_format": meta.get("activation_format", ""),
        "compute_format": meta.get("compute_format", ""),
        "quant_category": meta.get("quant_category", ""),
        "packed_weight_bytes": meta.get("packed_weight_bytes"),
        "weight_representation": meta.get("weight_representation", ""),
        "dequant_before_compute": bool(meta.get("dequant_before_compute", False)),
        "cache_kind": meta.get("cache_kind"),
        "shared_owner": meta.get("shared_owner"),
        "cache_visible": meta.get("cache_visible"),
        "owner_layer": meta.get("owner_layer"),
        "collective": meta.get("collective"),
        "payload_width": meta.get("payload_width"),
        "local_heads": meta.get("local_heads"),
        "local_kv_heads": meta.get("local_kv_heads"),
        "witness_only": bool(meta.get("witness_only", False)),
        "not_execution": bool(meta.get("not_execution", False)),
        "metadata": meta,
    }


def iteration_campaign_records(
    model: ModelSpec, work: Any,
) -> tuple[dict[str, Any], ...]:
    by_layer = {plan.index: plan for plan in model.layer_plans}
    return tuple(
        op_campaign_record(model, op, layer_plan=by_layer.get(op.layer))
        for op in getattr(work, "ops", ())
    )


__all__ = ["op_campaign_record", "iteration_campaign_records"]
