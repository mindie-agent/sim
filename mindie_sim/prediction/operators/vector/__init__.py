"""Normalization, quantization, activation, and vector pipeline operators.

[Role] Governed operator package owning ``vector`` device kernel recipes.
[Invariants] Single-owner operator package; no hardware capability sniffing.
[Timeouts & Locks] Local recipe imports; no I/O or locks.
[Side-Effect Policy] Zero import side effects.
"""
from .pointwise import TiledPointwise

__all__ = ["TiledPointwise"]
