"""TOMBSTONE / LEGACY_PURGED socket for Wave W5a per-die memory ledger projection.

[Role] Unplugged W5a Commit #1 socket replacing the deleted 1,661-line shadow
  HBM/quant/cache/concurrency calculator ("两张皮" #1). W5a implementation must
  project directly from canonical ``StageReceipt.memory_receipt`` /
  ``DieResidentReceipt`` with exact 0 B multi-key integer bijection.
[Invariants] Zero shadow weight/quant/cache/concurrency recomputation, zero
  per-request division of fixed C8 channel scales, and zero silent BF16/A5
  fallbacks.
[Timeouts & Locks] Pure stub; no I/O or locks.
[Side-Effect Policy] Side-effect free; raises ``NotImplementedError`` until
  wired to canonical ``DieResidentReceipt``.
"""
from __future__ import annotations

from typing import Any

_LEGACY_PURGED_MSG = (
    "LEGACY_PURGED: project StageReceipt.memory_receipt DieResidentReceipt "
    "directly with 0 B multi-key bijection"
)


def evaluate_weight_storage_bytes(*args: Any, **kwargs: Any) -> tuple[int, int, int, int]:
    """Unplugged W5a socket — shadow weight storage calculator deleted."""
    raise NotImplementedError(_LEGACY_PURGED_MSG)


def compute_memory_quant_ledger(*args: Any, **kwargs: Any) -> Any:
    """Unplugged W5a socket — shadow memory ledger deleted."""
    raise NotImplementedError(_LEGACY_PURGED_MSG)


__all__ = [
    "compute_memory_quant_ledger",
    "evaluate_weight_storage_bytes",
]
