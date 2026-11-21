"""Causal 1D convolution and non-contiguous state-copy operators.

[Role] Governed operator package owning ``causal_conv`` device kernel recipes.
[Invariants] Single-owner operator package; no hardware capability sniffing.
[Timeouts & Locks] Pure namespace package; no I/O or locks.
[Side-Effect Policy] Zero import side effects.
"""
