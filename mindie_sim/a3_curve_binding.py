"""Bridge from cost-model ops to the public train-only curve query API.

[Role]
  Build a complete pre-use physical query from explicit op/case metadata and
  call ``a3_operator_curve_forecast.query_curve_artifact``.  No local fitter,
  observed outcome, hardware transfer or graph/eager normalization lives here.

[Invariants]
  * Missing protocol/features facts remain absent, so the public query returns
    unsupported rather than wildcarding another measurement.
  * Hardware is a physical identity; cross-hardware transfer is never used.
  * Transition gaps stay unresolved; the caller retains its analytic estimate.
  * The operator-owned predictor is the only query implementation.

[Timeouts & Locks]
  Pure bounded query; no I/O/device/locks.

[Side-Effect Policy]
  No side effects.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from .a3_operator_curve_protocol import canonical_protocol, protocol_equal
from .a3_precision import physical_layout, physical_quantization


def build_curve_base_query(model: Any, op: Any, shape: Mapping[str, Any]) -> dict[str, Any]:
    meta = dict(op.metadata)
    explicit = meta.get("curve_query")
    query = dict(explicit) if isinstance(explicit, Mapping) else {}
    precision = dict(meta.get("curve_precision") or {})
    if not precision:
        activation = model.activation_dtype or model.compute_dtype
        weight = model.weight_dtype or model.compute_dtype
        precision = {
            "input_dtype": activation,
            "weight_dtype": weight,
            "output_dtype": activation,
            "compute_dtype": model.compute_dtype,
            "accumulation_dtype": model.accumulation_dtype,
            "quantization": physical_quantization(model.quantization),
            "weight_format": weight,
            "activation_format": activation,
            "compute_format": model.compute_dtype,
        }
    query.setdefault("hardware_key", model.machine_key())
    query.setdefault("family", meta.get("curve_family", op.family))
    query.setdefault("recipe", meta.get("curve_recipe", op.recipe))
    query.setdefault("execution_kind", meta.get("curve_execution_kind"))
    query.setdefault("worker_recipe", meta.get("curve_worker_recipe"))
    query.setdefault("backend", meta.get("curve_backend"))
    query.setdefault("target_match", meta.get("curve_target_match"))
    query.setdefault("coverage_scope", meta.get("curve_coverage_scope"))
    query.setdefault("layout", meta.get("curve_layout", physical_layout(op.layout)))
    query.setdefault(
        "quantization",
        meta.get("curve_quantization", physical_quantization(model.quantization)),
    )
    query.setdefault("precision", precision)
    query.setdefault("features", dict(meta.get("curve_features") or {}))
    query.setdefault("protocol", dict(meta.get("curve_protocol") or {}))
    query.setdefault("shape", {
        str(key): (int(value) if float(value).is_integer() else float(value))
        for key, value in dict(shape).items()
    })
    return query


def _annotate_transition(
    result: dict[str, Any], query: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach the concrete unmet physical condition to a transition result."""
    if str(result.get("status") or "") != "transition_unresolved":
        return result
    shape = {
        str(key): value for key, value in dict(query.get("shape") or {}).items()
    }
    axis = str(result.get("axis") or query.get("axis") or "")
    result["missing_condition"] = {
        "hardware_key": str(query.get("hardware_key") or ""),
        "family": query.get("family"),
        "recipe": query.get("recipe"),
        "axis": axis or None,
        "value": result.get("x"),
        "shape": shape,
        "fixed_shape": {
            key: value for key, value in shape.items() if key != axis
        },
        "protocol": canonical_protocol(query.get("protocol") or {}),
        "precision": dict(query.get("precision") or {}),
        "layout": query.get("layout"),
        "target_match": query.get("target_match"),
        "coverage_scope": query.get("coverage_scope"),
    }
    return result


def query_curve_for_op(
    model: Any, op: Any, shape: Mapping[str, Any],
) -> dict[str, Any] | None:
    artifact = getattr(model, "curve_artifact", None)
    if not artifact:
        return None
    from .a3_operator_curve_forecast import query_curve_artifact

    query = build_curve_base_query(model, op, shape)
    graph_mode = str(getattr(model, "graph_mode", "auto") or "auto").lower()
    if graph_mode in {"force", "graph", "graph_block", "graph_single_repeated"}:
        # A graph execution query must carry the measured block protocol; a
        # An unspecified/eager protocol cannot be promoted by shape alone.
        mode = str((query.get("protocol") or {}).get("mode") or "").lower()
        if not mode.startswith("graph"):
            return None
    axes: list[str] = []
    explicit_axis = dict(op.metadata).get("curve_axis")
    if explicit_axis:
        axes.append(str(explicit_axis))
    for curve in artifact.get("curves", ()):
        axis = str(curve.get("axis") or "")
        if axis and axis not in axes:
            axes.append(axis)
    for axis in axes:
        if axis not in (query.get("shape") or {}):
            continue
        # explicit metadata path
        axis_query = {**query, "axis": axis}
        result = query_curve_artifact(
            artifact,
            axis_query,
            # Hardware is a physical identity, never a duration-transfer
            # wildcard.
            allow_primitive_transfer=False,
        )
        if result.get("status") in {
            "ok", "extrapolation", "transition_unresolved", "ambiguous",
        }:
            return _annotate_transition(result, axis_query)
    return _generic_curve_query(model, op, shape)


__all__ = ["build_curve_base_query", "query_curve_for_op"]


_GENERIC_GEMM_FAMILIES = {
    "gemm", "qkv_projection", "o_projection", "gate_up_projection",
    "down_projection", "mla_q_a", "mla_kv_a", "mla_q_b", "mla_kv_b",
    "moe_gate_up", "moe_down", "shared_expert_gate_up",
    "shared_expert_down", "linear_q_projection", "linear_k_projection",
    "linear_v_projection", "linear_o_projection", "logits",
}


def _generic_curve_query(
    model: Any, op: Any, shape: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Map a plain BF16 catalog GEMM to generic primitive transfer curves.

    This is an approximate physical transfer, not native-kernel equivalence.
    It never injects metadata into the artifact: it enumerates real curve
    physical/protocol identities, filters by the model's declared physical
    precision and runtime execution mode, and refuses multiple protocols.
    """
    from .a3_operator_curve_forecast import query_curve_artifact

    artifact = getattr(model, "curve_artifact", None)
    if not artifact:
        return None
    if op.family == "rmsnorm":
        if not bool(getattr(model, "curve_allow_primitive_transfer", False)):
            return None
        return _native_rms_curve_query(model, op)
    if op.family not in _GENERIC_GEMM_FAMILIES:
        return None
    if str(op.dtype or "").lower() not in {"", "bf16"}:
        return None
    model_quant = physical_quantization(model.quantization)
    if model_quant != "none":
        return None
    graph_mode = str(getattr(model, "graph_mode", "auto") or "auto").lower()
    explicit_protocol = dict(getattr(model, "calibration_protocol", None) or {})
    hardware = model.machine_key()
    shape_map = {
        str(key): (int(value) if float(value).is_integer() else float(value))
        for key, value in dict(shape).items()
    }
    candidates = []
    for curve in artifact.get("curves", ()):
        physical = dict(curve.get("physical") or {})
        protocol = canonical_protocol(curve.get("protocol") or {})
        if str(physical.get("family") or "").lower() != "gemm":
            continue
        if str(physical.get("hardware_key") or "") != hardware:
            continue
        if physical.get("target_match") is False:
            continue
        execution_kind = str(physical.get("execution_kind") or "").lower()
        coverage_scope = str(physical.get("coverage_scope") or "").lower()
        if execution_kind in {"fused_kernel", "reference_decomposition"}:
            continue
        if coverage_scope.startswith("native_operator_semantic"):
            continue
        if physical_quantization(physical.get("quantization")) != "none":
            continue
        if physical_layout(physical.get("layout")) != "contiguous":
            continue
        precision = dict(physical.get("precision") or {})
        for key in ("input_dtype", "weight_dtype", "output_dtype", "compute_dtype"):
            if str(precision.get(key) or "").lower() not in {"", "bf16"}:
                break
        else:
            pass
        if any(
            str(precision.get(key) or "").lower() not in {"", "bf16"}
            for key in ("input_dtype", "weight_dtype", "output_dtype", "compute_dtype")
        ):
            continue
        axis = str(curve.get("axis") or "")
        if not axis or axis not in shape_map:
            continue
        fixed = dict(curve.get("fixed_shape") or {})
        if {k: shape_map.get(k) for k in fixed} != fixed:
            continue
        mode = str(protocol.get("mode") or "").lower()
        if graph_mode in {"force", "graph", "graph_block", "graph_single_repeated"}:
            if not mode.startswith("graph"):
                continue
        elif mode.startswith("graph"):
            continue
        if explicit_protocol:
            # An explicit caller selection is exact; no partial eager/prefix
            # wildcard is allowed.  Approximate transfer remains available only
            # when the caller supplied no physical protocol at all.
            if not protocol_equal(
                protocol, canonical_protocol(explicit_protocol),
            ):
                continue
        candidates.append((curve, physical, protocol, axis))
    if not candidates:
        return None
    protocol_groups = {
        json.dumps(protocol, sort_keys=True, default=str)
        for _, _, protocol, _ in candidates
    }
    if len(protocol_groups) > 1:
        return {
            "status": "ambiguous",
            "prediction_ms": None,
            "reason": "multiple_physical_protocols_require_explicit_selection",
            "matches": sorted(protocol_groups),
        }
    matches: list[dict[str, Any]] = []
    for curve, physical, protocol, axis in candidates:
        query = {
            **physical,
            "axis": axis,
            "shape": shape_map,
            "protocol": protocol,
        }
        result = query_curve_artifact(
            artifact, query, allow_primitive_transfer=False,
        )
        if result.get("status") in {
            "ok", "extrapolation", "transition_unresolved",
        }:
            result["transfer_scope"] = "generic_bf16_primitive_transfer"
            result["production_match"] = False
            matches.append(_annotate_transition(result, query))
    # When the same physical condition has both a numerical segment and a
    # transition-bracket curve, use the numerical support.  A transition result
    # is preserved only when it is the sole supported curve for that condition;
    # this is evidence preference, not first-curve selection or a local fit.
    numerical = [
        result for result in matches
        if result.get("status") in {"ok", "extrapolation"}
        and result.get("prediction_ms") is not None
    ]
    if numerical:
        matches = numerical
    unique = {
        str(result.get("matched_curve")): result for result in matches
    }
    if len(unique) == 1:
        return next(iter(unique.values()))
    if len(unique) > 1:
        return {
            "status": "ambiguous",
            "prediction_ms": None,
            "reason": "multiple_generic_curves_match",
            "matches": sorted(unique),
        }
    return None


def _native_rms_curve_query(
    model: Any, op: Any,
) -> dict[str, Any] | None:
    """Narrow explicit BF16 plain-RMS primitive transfer.

    This is a candidate transfer, not production equivalence.  The caller must
    opt in; the actual op must match hardware, BF16 input/output allocation,
    continuous layout, epsilon and the declared curve geometry.  Composite GDN
    split/L2/mRoPE or any other family is never treated as plain RMS.
    """
    from .a3_operator_curve_forecast import query_curve_artifact

    artifact = getattr(model, "curve_artifact", None)
    if not artifact:
        return None
    if not bool(getattr(model, "curve_allow_primitive_transfer", False)):
        return None
    if str(getattr(op, "family", "")) != "rmsnorm":
        return None
    if str(getattr(op, "dtype", "") or "").lower() not in {"", "bf16"}:
        return None
    if physical_quantization(getattr(model, "quantization", "")) != "none":
        return None
    if physical_layout(getattr(op, "layout", "")) != "contiguous":
        return None
    meta = dict(getattr(op, "metadata", ()) or ())
    if bool(meta.get("not_rmsnorm")) or bool(meta.get("composite")):
        return None
    if str(meta.get("norm_type", "rmsnorm")).lower() != "rmsnorm":
        return None
    if bool(meta.get("in_place", False)):
        return None
    if str(meta.get("input_format", "")).lower() not in {"", "bf16", "bfloat16"}:
        return None
    if str(meta.get("output_format", "")).lower() not in {"", "bf16", "bfloat16"}:
        return None
    epsilon_raw = meta.get("norm_eps", meta.get("epsilon"))
    if epsilon_raw is None:
        return None
    try:
        epsilon = float(epsilon_raw)
    except (TypeError, ValueError):
        return None
    local = tuple(
        int(value) for value in (
            getattr(op, "local_shape", ()) or getattr(op, "launch_shape", ())
        )
    )
    if len(local) != 2 or local[0] <= 0 or local[1] <= 0:
        return None
    shape = {"tokens": local[0], "width": local[1]}
    hardware = str(model.machine_key())
    graph_mode = str(getattr(model, "graph_mode", "auto") or "auto").lower()
    explicit = canonical_protocol(
        getattr(model, "calibration_protocol", None) or {}
    )
    candidates: list[tuple[Mapping[str, Any], Mapping[str, Any], dict[str, Any], str]] = []
    for curve in artifact.get("curves", ()):
        physical = dict(curve.get("physical") or {})
        if str(physical.get("family") or "").lower() != "rmsnorm":
            continue
        if str(physical.get("hardware_key") or "") != hardware:
            continue
        protocol = canonical_protocol(curve.get("protocol") or {})
        # Allocation/cache/timing facts are part of the physical contract.
        # An incomplete historical native export must not be adopted as a
        # current primitive candidate.
        if any(
            key not in protocol for key in (
                "timer", "warmup", "iterations", "trials",
                "preallocated", "cache_policy", "timing_scope",
            )
        ):
            continue
        if explicit and not protocol_equal(protocol, explicit):
            continue
        mode = str(protocol.get("mode") or "").lower()
        if graph_mode in {"force", "graph", "graph_block", "graph_single_repeated"}:
            if not mode.startswith("graph"):
                continue
        elif mode.startswith("graph"):
            continue
        if physical_quantization(physical.get("quantization")) != "none":
            continue
        if physical_layout(physical.get("layout")) != "contiguous":
            continue
        coverage_scope = str(physical.get("coverage_scope") or "").lower()
        if coverage_scope.startswith("native_operator_semantic"):
            continue
        if str(physical.get("execution_kind") or "").lower() == "reference_decomposition":
            continue
        precision = dict(physical.get("precision") or {})
        if any(
            str(precision.get(key) or "").lower() not in {"", "bf16"}
            for key in (
                "input_dtype", "weight_dtype", "output_dtype", "compute_dtype",
            )
        ):
            continue
        features = dict(physical.get("features") or {})
        try:
            curve_epsilon = float(features.get("epsilon"))
        except (TypeError, ValueError):
            continue
        if abs(curve_epsilon - epsilon) > max(1e-12, abs(epsilon) * 1e-9):
            continue
        axis = str(curve.get("axis") or "")
        if not axis or axis not in shape:
            continue
        fixed = dict(curve.get("fixed_shape") or {})
        if {key: shape.get(key) for key in fixed} != fixed:
            continue
        candidates.append((curve, physical, protocol, axis))
    if not candidates:
        return None
    axes = {axis for _, _, _, axis in candidates}
    explicit_axis = str(meta.get("curve_axis") or "")
    if explicit_axis in axes:
        selected_axis = explicit_axis
    elif "tokens" in axes:
        selected_axis = "tokens"
    elif len(axes) == 1:
        selected_axis = next(iter(axes))
    else:
        return {
            "status": "ambiguous",
            "prediction_ms": None,
            "transfer_scope": "explicit_native_primitive_transfer",
            "production_match": False,
            "reason": "primitive_curve_axis_requires_explicit_selection",
            "matches": sorted(axes),
        }
    candidates = [item for item in candidates if item[3] == selected_axis]
    protocol_groups = {
        json.dumps(protocol, sort_keys=True, separators=(",", ":"))
        for _, _, protocol, _ in candidates
    }
    if len(protocol_groups) > 1:
        return {
            "status": "ambiguous",
            "prediction_ms": None,
            "transfer_scope": "explicit_native_primitive_transfer",
            "production_match": False,
            "reason": "multiple_physical_protocols_require_explicit_selection",
            "matches": sorted(protocol_groups),
        }
    matches: list[dict[str, Any]] = []
    for curve, physical, protocol, axis in candidates:
        query = {
            **physical, "axis": axis, "shape": shape, "protocol": protocol,
        }
        result = query_curve_artifact(
            artifact, query, allow_primitive_transfer=False,
        )
        if result.get("status") in {"ok", "extrapolation", "transition_unresolved"}:
            result["transfer_scope"] = "explicit_native_primitive_transfer"
            result["production_match"] = False
            result["candidate"] = True
            matches.append(_annotate_transition(result, query))
    numerical = [
        result for result in matches
        if result.get("status") in {"ok", "extrapolation"}
        and result.get("prediction_ms") is not None
    ]
    if numerical:
        matches = numerical
    unique = {
        str(result.get("matched_curve")): result for result in matches
    }
    if len(unique) == 1:
        return next(iter(unique.values()))
    if len(unique) > 1:
        return {
            "status": "ambiguous",
            "prediction_ms": None,
            "transfer_scope": "explicit_native_primitive_transfer",
            "production_match": False,
            "reason": "multiple_generic_curves_match",
            "matches": sorted(unique),
        }
    return None
