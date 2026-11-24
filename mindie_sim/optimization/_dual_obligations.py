"""Default per-site compute, vector, link, and boundary obligations for dual bounds.

[Role] Re-export the canonical replica-preserving obligation derivation helpers
  from ``mindie_sim.rewrite.dual_cert`` without duplicating demand accounting.
[Invariants] Every inferred obligation is sound under all valid framework and
  kernel rewrites: matrix MACs, non-absorbable vector ops, multi-rank collective
  link egress, cold weight reads, and independent read (``mte2 > 0``) / terminal
  store (``mte3 > 0``) pure-copy boundaries keyed by ``(replica, op_id)``.
[Timeouts & Locks] Single linear pass over scenario rank graphs; no I/O or locks.
[Side-Effect Policy] Pure read-only derivation returning immutable obligations.
"""
from __future__ import annotations

from mindie_sim.rewrite.dual_cert import (
    _COLD_WEIGHT_FAMILIES,
    _WEIGHT_ID_KEYS,
    _cold_weight_read,
    _pure_copy_boundary,
    _weight_allocation_id,
    derive_default_obligations,
)

__all__ = (
    "_COLD_WEIGHT_FAMILIES",
    "_WEIGHT_ID_KEYS",
    "_cold_weight_read",
    "_pure_copy_boundary",
    "_weight_allocation_id",
    "derive_default_obligations",
)
