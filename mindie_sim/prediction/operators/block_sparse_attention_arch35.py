"""Compatibility imports for the governed attention package.

[Role] Preserve the legacy block-sparse recipe entrypoints.
[Invariants] A single attention package owns all planning and dispatch.
[Timeouts & Locks] Imports only.
[Side-Effect Policy] No I/O or runtime initialization.
"""
from mindie_sim.prediction.operators.attention import (
    GBSA_RECIPE,
    KV_QUANT_SFA_C8_RECIPE,
    QUANT_MODE_BF16,
    QUANT_MODE_FP8_C8,
    QUANT_MODE_INT8_C8,
    QUANT_MODE_MXFP8,
    QUANT_MODE_TQ4,
    SPARSE_FLASH_MLA_NOPE_RECIPE,
    SUPPORTED_QUANT_MODES,
    build as _build,
    make_gbsa_op,
    make_kv_quant_sfa_c8_op,
    recipes,
)


def build(op, geometry):
    """Delegate the legacy planning signature."""
    return _build(op, geometry)


__all__ = [
    "GBSA_RECIPE", "KV_QUANT_SFA_C8_RECIPE", "SPARSE_FLASH_MLA_NOPE_RECIPE",
    "QUANT_MODE_BF16", "QUANT_MODE_INT8_C8", "QUANT_MODE_TQ4", "QUANT_MODE_FP8_C8", "QUANT_MODE_MXFP8",
    "SUPPORTED_QUANT_MODES", "make_gbsa_op", "make_kv_quant_sfa_c8_op", "build", "recipes",
]
