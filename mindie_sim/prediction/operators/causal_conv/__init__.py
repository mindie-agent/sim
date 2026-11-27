"""Causal 1D convolution and non-contiguous state-copy operators.

[Role] Governed operator package owning ``causal_conv`` device kernel recipes.
[Invariants] Single-owner operator package; no hardware capability sniffing.
[Timeouts & Locks] Pure namespace package; no I/O or locks.
[Side-Effect Policy] Zero import side effects.
"""
from .family_causal_conv import (
    CONV,
    CONV_SOURCE,
    CONV_SPECULATIVE_WIDTH,
    CONV_SUPPORTED_WIDTHS,
    conv_plan,
    conv_schedule,
    conv_spec_terms,
    conv_task_runs,
    conv_token_ops,
    update_geometry,
)

__all__ = [
    "CONV",
    "CONV_SOURCE",
    "CONV_SPECULATIVE_WIDTH",
    "CONV_SUPPORTED_WIDTHS",
    "conv_plan",
    "conv_schedule",
    "conv_spec_terms",
    "conv_task_runs",
    "conv_token_ops",
    "update_geometry",
]
