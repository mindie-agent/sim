"""Primitive units and input parsing.

[Role] Common numerical contracts independent of device vendors.
[Invariants] Units and declared precision are explicit.
[Timeouts & Locks] Pure arithmetic and validation.
[Side-Effect Policy] No I/O or mutable global state.
"""
from __future__ import annotations
from copy import deepcopy
import math
from typing import Any, Mapping

_CASE_PROVENANCE_FIELDS = (
    "source", "revision", "commit", "git_commit", "source_sha", "name",
    "description",
)

_IGNORED_DOCUMENT_FIELDS = {
    "metadata", "notes", "assumptions",
}

_DTYPE_BYTES = {
    "fp32": 4.0, "float32": 4.0, "bf16": 2.0, "bfloat16": 2.0, "fp16": 2.0,
    "float16": 2.0, "fp8": 1.0, "fp8_e4m3": 1.0, "fp8_e5m2": 1.0,
    "int8": 1.0, "int4": 0.5, "int32": 4.0, "bool": 1.0, "fp8_e8m0": 1.0, "e8m0": 1.0,
}

_DTYPE_PEAK = {
    "bf16": 1.0, "bfloat16": 1.0, "fp16": 1.0, "float16": 1.0,
    "fp32": 0.5, "float32": 0.5, "fp8": 2.0, "fp8_e4m3": 2.0,
    "fp8_e5m2": 2.0, "int8": 2.0, "int4": 4.0,
}

FP8_PEAK_DTYPES = frozenset({
    "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "mx_fp8", "mxfp8_e4m3",
})

_PEAK_DTYPE_ALIASES = {
    "fp8": ("fp8", "fp8_e4m3", "fp8_e5m2"),
    "fp8_e4m3": ("fp8_e4m3", "fp8"),
    "fp8_e5m2": ("fp8_e5m2", "fp8"),
    "bf16": ("bf16", "bfloat16"),
    "bfloat16": ("bfloat16", "bf16"),
    "fp16": ("fp16", "float16"),
    "float16": ("float16", "fp16"),
    "fp32": ("fp32", "float32"),
    "float32": ("float32", "fp32"),
}

_BASE_PEAK_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16"})

class ModelingError(ValueError):
    """Invalid or unsupported modeling input."""

    def __init__(self, reason: str) -> None:
        super().__init__(str(reason))
        self.reason = str(reason)

def _mapping(value: Any, reason: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelingError(reason)
    return value

def _unknown(mapping: Mapping[str, Any], allowed: set[str], context: str) -> None:
    for key in mapping:
        if (key not in allowed and key not in _CASE_PROVENANCE_FIELDS
                and key not in _IGNORED_DOCUMENT_FIELDS
                and not str(key).startswith("x_")):
            raise ModelingError(f"unknown_field:{context}.{key}")

def _float(value: Any, reason: str, *, minimum: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if not math.isfinite(result) or result < minimum:
        raise ModelingError(reason)
    return result

def _int(value: Any, reason: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool):
        raise ModelingError(reason)
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if result < minimum or float(value) != float(result):
        raise ModelingError(reason)
    return result

def _positive_float(value: Any, reason: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ModelingError(reason)
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ModelingError(reason) from exc
    if not math.isfinite(result) or result < 0 or (result == 0 and not allow_zero):
        raise ModelingError(reason)
    return result

def _bool(value: Any, reason: str) -> bool:
    if not isinstance(value, bool):
        raise ModelingError(reason)
    return value

def _text(value: Any, reason: str, *, default: str | None = None) -> str:
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(reason)
    return value.strip()

def dtype_bytes(dtype: str) -> float:
    key = str(dtype).strip().lower()
    if key not in _DTYPE_BYTES:
        raise ModelingError(f"unsupported_dtype:{dtype}")
    return _DTYPE_BYTES[key]

def dtype_peak_scale(dtype: str) -> float:
    key = str(dtype).strip().lower()
    if key not in _DTYPE_PEAK:
        raise ModelingError(f"unsupported_compute_dtype:{dtype}")
    return _DTYPE_PEAK[key]

def align_up(value: int, alignment: int) -> int:
    if alignment <= 1:
        return int(value)
    return (int(value) + alignment - 1) // alignment * alignment

def _normalize_dtype(value: Any, reason: str) -> str:
    text = _text(value, reason).lower().replace("-", "_")
    if text not in _DTYPE_BYTES:
        raise ModelingError(f"{reason}:{text}")
    return text

def _metadata(mapping: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    items = {key: str(mapping[key]) for key in _CASE_PROVENANCE_FIELDS
             if mapping.get(key) is not None}
    if isinstance(mapping.get("metadata"), Mapping):
        # Coverage lists and nested source contracts remain structured through
        # normalization; explicit metadata overrides top-level provenance.
        items.update((str(key), deepcopy(value))
                     for key, value in mapping["metadata"].items())
    return tuple(sorted(items.items()))
