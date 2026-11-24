"""TOMBSTONE / LEGACY_PURGED socket for Wave W5a Sheet 3 memory worksheet writer.

[Role] Unplugged W5a Commit #1 socket replacing shadow Sheet 3 recomputation,
  precision-ratio scaling, SKU substring sniffing, and concurrency-scaled C8
  channel-scale formulas. W5a implementation must render Sheet 3 directly from
  exact per-Die integer ``DieResidentReceipt`` / ``HybridPoolReceipt``.
[Invariants] Zero coupling to shadow ``export/memory_ledger.py`` constants or
  ``WEIGHT_QUANT_FORMATS``; zero SKU token sniffing; zero float ratio scaling.
[Timeouts & Locks] Pure stub; no I/O or locks.
[Side-Effect Policy] Side-effect free; raises ``NotImplementedError`` until
  wired to canonical ``DieResidentReceipt``.
"""
from __future__ import annotations

from typing import Any

#: Sheet 3 structural column count (24 columns A..X).
MEMORY_COLS = 24
#: Default Sheet 3 quant mode preserving every row's evaluated format and scale bytes.
QUANT_MODE_AS_EVALUATED = "As evaluated"

_LEGACY_PURGED_MSG = (
    "LEGACY_PURGED: render DieResidentReceipt from StageReceipt.memory_receipt"
)


def populate_memory_sheet(ws: Any, bundle: Any) -> None:
    """Unplugged W5a socket — shadow Sheet 3 calculator deleted."""
    del ws, bundle
    raise NotImplementedError(_LEGACY_PURGED_MSG)


__all__ = ["MEMORY_COLS", "QUANT_MODE_AS_EVALUATED", "populate_memory_sheet"]
