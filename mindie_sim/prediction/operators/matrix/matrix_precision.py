"""Hardware-aware execution precision for generic packed matrix work.

[Role]
  Resolve eligible generic packed-FP4 matrix operators onto the executing dies'
  native FP4 (``mxfp4``/``fp4_e2m1`` for W4A4) or native FP8 (``fp8_e4m3`` /
  ``fp8_e5m2`` for W4A8) Cube supply at hardware binding, keeping W4A16 on
  ``bf16`` and persisting resolved ``dtype`` and ``matrix_flops``.

[Invariants]
  * ``policy="declared"`` and explicit source/recipe/compute declarations keep
    their declared compute identity; a calibration table never selects a
    precision.
  * Resolution requires a native operand mode and a finite positive peak on
    *every* executing die; a heterogeneous stage uses a common legal precision.
  * FP4 activations select native ``mxfp4``/``fp4_e2m1``; E4M3/MXFP8 activations
    select native ``fp8_e4m3``; E5M2 selects native ``fp8_e5m2``; BF16/FP16
    activations stay on ``bf16`` (W4A16).
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


PACKED_FP4_FORMATS = frozenset({"mxfp4", "fp4", "fp4_e2m1"})
FP4_ACTIVATION_FORMATS = frozenset({"fp4", "fp4_e2m1", "mxfp4"})
FP8_ACTIVATION_FORMATS = frozenset({"fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8"})
NATIVE_LOW_PRECISION_FORMATS = frozenset({"fp4_e2m1", "mxfp4", "hifloat8", "mxfp8"})
E4M3_ACTIVATIONS = frozenset({"fp8", "fp8_e4m3", "mxfp8"})
E5M2_ACTIVATIONS = frozenset({"fp8_e5m2"})
CATALOG_DEFAULT = "catalog_default"
ORIGIN_RESOLVED_FP8 = "resolved_generic_fp8"
ORIGIN_RESOLVED_FP4 = "resolved_native_fp4"
RESOLVED_ORIGINS = frozenset({ORIGIN_RESOLVED_FP8, ORIGIN_RESOLVED_FP4})
EXPLICIT_ORIGINS = frozenset({"explicit", "source", "typed", "calibrated"})
EXPLICIT_METADATA_KEYS = frozenset({
    "native_operator", "native_kind", "source_implementation",
    "compute_format_override", "calibrated_service",
})
REASON_UNAVAILABLE = "fp8_unavailable"
REASON_FP4_UNAVAILABLE = "fp4_unavailable"
REASON_SOURCE = "explicit_compute_identity"
REASON_GENERIC_FP8 = "generic_fp8_dequant"
REASON_NATIVE_FP4 = "native_fp4_mmad"


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
    if not math.isfinite(peak) or peak <= 0:
        raise ModelingError(f"matrix_precision_peak_unusable:{dtype}")
    return peak


def _supported_precision(devices: Sequence[Any], activation: str) -> str | None:
    """Return the common native FP4/FP8 dtype, or ``None`` when unavailable."""
    candidates = (
        ("mxfp4", "fp4_e2m1") if activation in FP4_ACTIVATION_FORMATS
        else ("fp8_e4m3",) if activation in E4M3_ACTIVATIONS
        else ("fp8_e5m2",) if activation in E5M2_ACTIVATIONS
        else ()
    )
    for dtype in candidates:
        try:
            for device in devices:
                _native_peak_checked(device, dtype)
        except ModelingError:
            continue
        return dtype
    return None


def _all_dies_support(devices: Sequence[Any], dtype: str) -> bool:
    if not devices or not dtype:
        return False
    try:
        for device in devices:
            _native_peak_checked(device, dtype)
    except ModelingError:
        return False
    return True


def _with_reason(op: OpSpec, reason: str) -> OpSpec:
    meta = _metadata(op)
    if meta.get("precision_resolution_reason") == reason:
        return op
    meta["precision_resolution_reason"] = reason
    return replace(op, metadata=tuple(sorted(meta.items())))


def _fallback_bf16_with_reason(op: OpSpec, devices: Sequence[Any], reason: str) -> OpSpec:
    meta = _metadata(op)
    origin = str(meta.get("compute_format_origin", ""))
    if (
        (origin in ("", CATALOG_DEFAULT) or origin in RESOLVED_ORIGINS)
        and op.dtype in (PACKED_FP4_FORMATS | FP8_ACTIVATION_FORMATS)
        and not _all_dies_support(devices, str(op.dtype))
        and _all_dies_support(devices, "bf16")
    ):
        meta["precision_resolution_reason"] = reason
        meta["compute_format"] = "bf16"
        meta.pop("execution_compute_format", None)
        if origin in RESOLVED_ORIGINS:
            meta["compute_format_origin"] = CATALOG_DEFAULT
        return replace(op, dtype="bf16", matrix_flops=(), metadata=tuple(sorted(meta.items())))
    return _with_reason(op, reason)


def resolve_matrix_precision(
    op: OpSpec, *, devices: Sequence[Any], policy: str, recipe: Any,
) -> OpSpec:
    """Resolve one operator's execution compute identity (idempotent)."""
    if policy == "declared":
        return op
    if policy != "auto":
        raise ModelingError(f"matrix_compute_policy_unsupported:{policy}")
    if not op.metadata:
        return op
    meta = _metadata(op)
    weight = _declared_weight_format(meta).strip().lower()
    if weight not in PACKED_FP4_FORMATS:
        return op
    activation = _declared_activation(meta).strip().lower()
    is_fp4_act = activation in FP4_ACTIVATION_FORMATS
    is_fp8_act = activation in FP8_ACTIVATION_FORMATS
    if not (is_fp4_act or is_fp8_act):
        return op
    origin = str(meta.get("compute_format_origin", ""))
    if origin not in RESOLVED_ORIGINS and _explicit_identity(op, meta, recipe):
        return _with_reason(op, REASON_SOURCE)
    target = _supported_precision(devices, activation)
    if target is None:
        reason = REASON_FP4_UNAVAILABLE if is_fp4_act else REASON_UNAVAILABLE
        return _fallback_bf16_with_reason(op, devices, reason)
    expected_mflops = (
        ((target, op.flops),)
        if (op.flops and (not op.matrix_flops or origin in RESOLVED_ORIGINS))
        else op.matrix_flops
    )
    if (
        op.dtype == target
        and meta.get("execution_compute_format") == target
        and op.matrix_flops == expected_mflops
    ):
        return op
    declared = str(meta.get("declared_compute_format") or meta.get("compute_format") or op.dtype or "")
    updated = dict(meta)
    updated["declared_compute_format"] = declared
    updated["execution_compute_format"] = target
    updated["compute_format"] = target
    updated["compute_format_origin"] = ORIGIN_RESOLVED_FP4 if is_fp4_act else ORIGIN_RESOLVED_FP8
    updated["precision_resolution_reason"] = REASON_NATIVE_FP4 if is_fp4_act else REASON_GENERIC_FP8
    return replace(
        op,
        dtype=target,
        matrix_flops=expected_mflops,
        metadata=tuple(sorted(updated.items())),
    )


def resolve_case_matrix_precision(
    op: OpSpec, case: Any, *, replica: int = 0, recipe: Any = None,
) -> OpSpec:
    """Resolve ``op`` against its executing stage dies from ``case``."""
    dies = tuple(getattr(getattr(case, "machine", None), "dies", ()) or ())
    if not dies:
        return op
    parallel = getattr(getattr(case, "runtime", None), "parallel", None)
    width = (int(parallel.tp) * int(parallel.pcp)) if parallel is not None else 1
    pp_stages = int(parallel.pp) if parallel is not None else 1
    start = (int(replica) * pp_stages + int(getattr(op, "stage", 0))) * width
    stage_dies = dies[start:start + width] or (dies[0],)
    policy = str(getattr(getattr(case, "runtime", None), "matrix_compute_policy", "auto") or "auto")
    return resolve_matrix_precision(op, devices=stage_dies, policy=policy, recipe=recipe)


def execution_compute_peaks(
    op: OpSpec, *, devices: Sequence[Any], base_peaks: Mapping[str, float],
) -> dict[str, float]:
    """Return every peak the bound context needs, validated on all dies."""
    aliases = {str(key): str(value) for device in devices
               for key, value in dict(getattr(device, "compute_peak_aliases", ()) or ()).items()}
    if op.dtype and str(op.dtype) in aliases:
        raise ModelingError(f"matrix_precision_not_native:{op.dtype}")
    metadata = dict(op.metadata)
    dtypes: list[str] = []
    resolved = str(metadata.get("execution_compute_format", ""))
    if op.dtype and (op.dtype in base_peaks or resolved == str(op.dtype)
                     or (op.flops and op.dtype in NATIVE_LOW_PRECISION_FORMATS)):
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
