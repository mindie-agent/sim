"""Physical precision/layout normalization for cost-model calibration.

[Role]
  Convert model/catalog labels such as ``bf16_qwen`` or a dtype-like layout
  into the physical precision condition used by measured calibration rows.

[Invariants]
  * Model variant labels are not quantization modes.
  * Packed MXFP4/FP8/INT8 identities remain distinct.
  * Unknown layout text is preserved rather than wildcarded.

[Timeouts & Locks]
  Pure string normalization.

[Side-Effect Policy]
  No I/O or global mutation.
"""
from __future__ import annotations

_PHYSICAL_LAYOUT_LABELS = {
    "bf16", "bfloat16", "fp16", "float16", "fp32", "float32",
    "fp8", "fp8_e4m3", "fp8_e5m2", "int8", "int4", "mxfp4",
}
_PHYSICAL_QUANT_TOKENS = (
    "mxfp4", "fp4", "w4a16", "int4", "fp8", "e4m3", "e5m2",
    "int8", "w8a8",
)


def physical_quantization(label: str) -> str:
    value = str(label or "").strip().lower().replace("-", "_")
    if not value or value == "none":
        return "none"
    for token in _PHYSICAL_QUANT_TOKENS:
        if token in value:
            if token in {"e4m3", "e5m2"}:
                return "fp8"
            if token in {"w4a16"}:
                return "int4"
            if token in {"w8a8"}:
                return "int8"
            return token
    return "none"


def physical_layout(label: str) -> str:
    value = str(label or "").strip().lower()
    if not value:
        return ""
    if value in _PHYSICAL_LAYOUT_LABELS:
        return "contiguous"
    return value


__all__ = ["physical_layout", "physical_quantization"]
