"""Compatibility entry point for the existing MLA prolog recipe identifier.

[Role] Forward legacy MLA prolog recipe callers to the governed package.
[Invariants] The package owns all dispatch, work, and lowering.
[Timeouts & Locks] Static imports with no I/O or locks.
[Side-Effect Policy] Re-exports the existing entry points and contract constants.
"""
from mindie_sim.prediction.operators.mla_prolog import (
    BF16_BYTES,
    C8_BLOCK_SIZE,
    DEFAULT_KV_LORA_RANK,
    DEFAULT_PROLOG_HEADS,
    DEFAULT_PROLOG_HIDDEN,
    DEFAULT_QK_NOPE_DIM,
    DEFAULT_QK_ROPE_DIM,
    DEFAULT_Q_LORA_RANK,
    FP32_BYTES,
    MLA_PROLOG_V4_RECIPE,
    MX_GROUP_SIZE,
    SLOT_INDEX_BYTES,
    SUPPORTED_WEIGHT_MODES,
    WEIGHT_QUANT_BF16,
    WEIGHT_QUANT_INT8,
    WEIGHT_QUANT_MXFP8,
    build,
    make_op,
    recipes,
)
