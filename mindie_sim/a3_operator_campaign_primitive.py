"""Primitive precision and actual grouped-geometry facts for campaign conditions.

[Role]
  Resolve precision from the actual op evidence instead of global model quant
  defaults, and preserve the launched expert group rather than whole-rank
  routing.  Used by ``a3_operator_campaign_conditions``.

[Invariants]
  * A non-weight primitive (softmax/attention/norm) keeps its actual dtype and
    does not inherit model weight/scale/quant defaults.
  * Genuine INT8/FP8/MX quantize/dequant/packed GEMM metadata is preserved.
  * Actual ``group_size``/``expert_tokens`` drive point axes; rank/global counts
    stay separately labelled.  When actual group metadata is absent, rank
    routing is used explicitly with its source label.
  * Absent facts remain explicit rather than inferred.

[Timeouts & Locks]
  Pure bounded arithmetic; no I/O or locks.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

from typing import Any, Mapping

from .a3_operator_worker_native import _scaled_expert_counts


_WEIGHT_FAMILIES = {
    "gemm", "qkv_projection", "o_projection", "gate_up_projection",
    "down_projection", "mla_q_a", "mla_q_b", "mla_kv_a", "mla_kv_b",
    "mla_q_absorb", "shared_expert_gate_up", "shared_expert_down", "logits",
    "linear_q_projection", "linear_k_projection", "linear_v_projection",
    "linear_o_projection", "moe_gate_up", "moe_down", "embedding",
}


def _precision(op: OpSpec, model: Any) -> dict[str, Any]:
    """Prefer actual per-op metadata; never attach model quant to a primitive.

    A homogeneous FP32 softmax keeps FP32 compute and no weight/scale facts.
    Quantized GEMM/dequant/packed paths retain their declared metadata.
    """
    meta = dict(op.metadata)
    quant = model.quant
    weight_format = str(meta.get("weight_format") or "")
    has_weight = bool(weight_format or str(getattr(op, "family", "")) in _WEIGHT_FAMILIES)
    output_hint = str(meta.get("output_format") or meta.get("quant_output_dtype") or "")
    is_quantize = ("quant" in str(getattr(op, "family", "")).lower()
                   or "quant" in str(getattr(op, "recipe", "")).lower()
                   or output_hint in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2"})
    if not weight_format and has_weight:
        weight_format = str(model.weight_dtype)
    activation = str(meta.get("input_format") or meta.get("activation_format")
                     or op.dtype or model.activation_dtype)
    compute = str(meta.get("compute_format") or (
        op.dtype if not has_weight and op.dtype else model.compute_dtype))
    output = str(meta.get("output_format") or meta.get("quant_output_dtype")
                 or activation)
    quantization = str(meta.get("quantization") or (
        model.quantization if has_weight else
        ("int8_per_row" if (is_quantize and output_hint == "int8")
         and str(meta.get("scale_axis") or meta.get("quant_axis")) == "per_row"
         else "quantized" if is_quantize else "none")))
    scale_format = str(meta.get("scale_format") or meta.get("scale_dtype")
                       or (quant.scale_format if (quant and (has_weight or is_quantize)) else ""))
    try:
        scale_bytes = float(meta.get("scale_bytes")
                            if meta.get("scale_bytes") is not None
                            else (quant.scale_bytes if (quant and (has_weight or is_quantize)) else 0.0))
    except (TypeError, ValueError):
        scale_bytes = 0.0
    return {
        "input_dtype": activation,
        "weight_dtype": weight_format or None,
        "output_dtype": output,
        "compute_dtype": compute,
        "accumulation_dtype": str(meta.get("accumulation_dtype") or "backend_default"),
        "quantization": quantization,
        "weight_format": weight_format or None,
        "activation_format": activation,
        "compute_format": compute,
        "weight_representation": str(meta.get("weight_representation") or op.layout),
        "quant_category": str(meta.get("quant_category") or ""),
        "packed_weight_bytes": float(meta.get("packed_weight_bytes") or 0.0),
        "dequant_before_compute": bool(meta.get("dequant_before_compute", False)),
        "weight_block": list(quant.weight_block) if (quant and has_weight) else [0, 0],
        "expert_block": int(quant.expert_block) if (quant and has_weight) else 0,
        "scale_format": scale_format,
        "scale_bytes": scale_bytes,
        "scale_shape": list(meta.get("scale_shape") or ()),
        "scale_axis": str(meta.get("scale_axis") or meta.get("quant_axis") or ""),
        "quant_output_dtype": str(meta.get("quant_output_dtype") or output),
        "row_width": int(meta.get("row_width") or 0),
        "kv_dtype": str(meta.get("kv_dtype") or (
            quant.kv_dtype if (quant and has_weight) else "")),
    }

def _actual_grouped_geometry(op: OpSpec, work: Any,
                             global_experts: int) -> dict[str, Any] | None:
    """Return the actual kernel group, not whole-rank routing.

    The catalog metadata carries the launched expert group size and token
    histogram.  ``work.routing`` can describe the whole EP rank.  Keep both:
    actual group/point axes drive the worker; rank facts stay labelled
    provenance.  If no actual metadata exists, the rank routing remains the
    only declared source and is used explicitly.
    """
    if work.routing is None:
        return None
    meta = dict(op.metadata)
    routing = work.routing.to_dict() if hasattr(work.routing, "to_dict") else dict(work.routing)
    raw_counts = meta.get("expert_tokens")
    counts = []
    if isinstance(raw_counts, (list, tuple)):
        try:
            counts = [max(0, int(value)) for value in raw_counts]
        except (TypeError, ValueError):
            counts = []
    rank_counts = [max(0, int(value)) for value in (routing.get("local_expert_counts") or ())]
    rank_tokens = max(0, int(float(routing.get("routing_tokens") or 0)))
    rank_experts = max(1, int(routing.get("local_experts") or len(rank_counts) or 1))
    group_size = int(meta.get("group_size") or 0)
    local_shape = [int(v) for v in (getattr(op, "local_shape", ()) or ())]
    semantic_shape = [int(v) for v in (getattr(op, "semantic_shape", ()) or ())]
    local_tokens = (local_shape[0] if local_shape else
                    semantic_shape[0] if semantic_shape else 0)
    if counts and sum(counts) > 0:
        actual_counts = counts
        actual_tokens = sum(counts)
        actual_experts = max(1, group_size or len(counts))
        source = "actual_op_group_size_expert_tokens"
    elif group_size > 0:
        actual_experts = group_size
        actual_tokens = local_tokens if local_tokens > 0 else rank_tokens
        actual_counts, _ = _scaled_expert_counts(
            rank_counts, actual_experts, actual_tokens)
        source = "actual_group_size_rank_histogram_scaled_to_local_shape"
    else:
        actual_counts = rank_counts
        actual_tokens = rank_tokens
        actual_experts = rank_experts
        source = "declared_rank_routing_no_actual_group_metadata"
    return {
        "actual_experts": actual_experts,
        "actual_routed_tokens": actual_tokens,
        "actual_expert_tokens": actual_counts,
        "actual_local_expert_start": int(meta.get("local_expert_start") or 0),
        "global_experts": int(global_experts),
        "rank_experts": rank_experts,
        "rank_routed_tokens": rank_tokens,
        "rank_expert_tokens": rank_counts,
        "source": source,
    }


__all__ = ["_precision", "_actual_grouped_geometry"]
