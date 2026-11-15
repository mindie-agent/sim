"""Hardware-aware execution precision for generic packed matrix work.

[Role]
  Resolve an eligible generic MXA8W4 operator (packed FP4/FP8 weight stream,
  declared FP8 activation) onto the executing dies' native FP8 Cube supply at
  hardware binding.  Weight representation, activation/output representation,
  shapes, logical FLOPs, dependencies and source identities are preserved.

[Invariants]
  * ``policy="declared"`` and explicit source/recipe/compute declarations keep
    their declared compute identity; a calibration table never selects a
    precision.
  * Resolution requires a native FP8 operand mode and a finite positive peak on
    *every* executing die; a heterogeneous stage uses a common legal precision.
  * E4M3 activations select native E4M3; E5M2 selects native E5M2.  Without a
    supported native conversion an E5M2 activation stays on the BF16 path with
    ``precision_resolution_reason="fp8_unavailable"``.
  * ``execution_compute_peaks`` supplies a peak for ``op.dtype`` and every
    declared ``matrix_flops`` precision, taking the per-device minimum.  A
    supply alias key is never admitted as a native operand.

[Timeouts & Locks]
  Pure record transformation; no I/O, device contexts or locks.

[Side-Effect Policy]
  Returns a replacement ``OpSpec``; the caller owns caches and provenance.
"""
from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec


ZERO: float = 0.0
PACKED_FP4_FORMATS = frozenset({"mxfp4", "fp4", "fp4_e2m1"})
FP8_ACTIVATION_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2"})
E4M3_ACTIVATIONS = frozenset({"fp8", "fp8_e4m3"})
E5M2_ACTIVATIONS = frozenset({"fp8_e5m2"})
CATALOG_DEFAULT = "catalog_default"
EXPLICIT_ORIGINS = frozenset({"explicit", "source", "typed", "calibrated"})
EXPLICIT_METADATA_KEYS = frozenset({
    "native_operator", "native_kind", "source_implementation",
    "compute_format_override", "calibrated_service",
})
REASON_UNAVAILABLE = "fp8_unavailable"
REASON_ACTIVATION = "activation_not_fp8"
REASON_SOURCE = "explicit_compute_identity"
REASON_E5M2_CONVERSION = "fp8_e5m2_conversion_unsupported"


def _metadata(op: OpSpec) -> dict[str, Any]:
    return dict(op.metadata)


def _declared_weight_format(meta: Mapping[str, Any]) -> str:
    return str(meta.get("weight_format") or meta.get("weight_dtype") or "")


def _declared_activation(meta: Mapping[str, Any]) -> str:
    return str(meta.get("activation_format") or meta.get("input_format") or "")


def _explicit_identity(op: OpSpec, meta: Mapping[str, Any], recipe: Any) -> bool:
    if op.source_boundary:
        return True
    if op.matrix_flops:
        return True
    origin = str(meta.get("compute_format_origin", ""))
    if origin in EXPLICIT_ORIGINS:
        return True
    if any(key in meta for key in EXPLICIT_METADATA_KEYS):
        return True
    if meta.get("implementation"):
        return True
    if meta.get("kernel") or meta.get("kernel_name"):
        return True
    if recipe is not None and bool(getattr(recipe, "source_bound", False)):
        return True
    return False


def _native_peak_checked(device: Any, dtype: str) -> float:
    aliases = dict(getattr(device, "compute_peak_aliases", ()) or ())
    if dtype in aliases:
        raise ModelingError(f"matrix_precision_not_native:{dtype}")
    peak = float(device.peak(dtype))
    if not math.isfinite(peak) or peak <= ZERO:
        raise ModelingError(f"matrix_precision_peak_unusable:{dtype}")
    return peak


def _supported_precision(devices: Sequence[Any], activation: str) -> str | None:
    """Return the common native FP8 dtype, or ``None`` when unavailable."""
    candidates = (("fp8_e4m3",) if activation in E4M3_ACTIVATIONS
                  else ("fp8_e5m2",) if activation in E5M2_ACTIVATIONS
                  else ())
    for dtype in candidates:
        try:
            for device in devices:
                _native_peak_checked(device, dtype)
        except ModelingError:
            continue
        return dtype
    return None


def _with_reason(op: OpSpec, reason: str) -> OpSpec:
    meta = _metadata(op)
    if meta.get("precision_resolution_reason") == reason:
        return op
    meta["precision_resolution_reason"] = reason
    return replace(op, metadata=tuple(sorted(meta.items())))


def resolve_matrix_precision(
    op: OpSpec, *, devices: Sequence[Any], policy: str, recipe: Any,
) -> OpSpec:
    """Resolve one operator's execution compute identity (idempotent)."""
    if policy == "declared":
        return op
    if policy != "auto":
        raise ModelingError(f"matrix_compute_policy_unsupported:{policy}")
    meta = _metadata(op)
    weight = _declared_weight_format(meta).strip().lower()
    if weight not in PACKED_FP4_FORMATS:
        return op
    activation = _declared_activation(meta).strip().lower()
    if activation not in FP8_ACTIVATION_FORMATS:
        return op
    if _explicit_identity(op, meta, recipe):
        return _with_reason(op, REASON_SOURCE)
    target = _supported_precision(devices, activation)
    if target is None:
        return _with_reason(op, REASON_UNAVAILABLE)
    declared = str(meta.get("compute_format") or op.dtype or "")
    if op.dtype == target and meta.get("execution_compute_format") == target:
        return op
    updated = dict(meta)
    updated["declared_compute_format"] = declared
    updated["execution_compute_format"] = target
    updated["compute_format"] = target
    updated["compute_format_origin"] = "resolved_generic_fp8"
    updated["precision_resolution_reason"] = "generic_fp8_dequant"
    return replace(op, dtype=target, metadata=tuple(sorted(updated.items())))


def execution_compute_peaks(
    op: OpSpec, *, devices: Sequence[Any], base_peaks: Mapping[str, float],
) -> dict[str, float]:
    """Return every peak the bound context needs, validated on all dies."""
    aliases = {str(key): str(value) for device in devices
               for key, value in dict(getattr(device, "compute_peak_aliases", ()) or ()).items()}
    if op.dtype and str(op.dtype) in aliases:
        raise ModelingError(f"matrix_precision_not_native:{op.dtype}")
    result = dict(base_peaks)
    metadata = dict(op.metadata)
    dtypes: list[str] = []
    resolved = str(metadata.get("execution_compute_format", ""))
    if op.dtype and (op.dtype in base_peaks or resolved == str(op.dtype)):
        dtypes.append(str(op.dtype))
    dtypes.extend(str(dtype) for dtype, _amount in op.matrix_flops)
    seen: set[str] = set()
    additions: dict[str, float] = {}
    for dtype in dtypes:
        if dtype in seen or dtype in base_peaks:
            continue
        seen.add(dtype)
        additions[dtype] = min(_native_peak_checked(device, dtype)
                               for device in devices)
    if not additions:
        return base_peaks if isinstance(base_peaks, dict) else dict(base_peaks)
    return {**base_peaks, **additions}
