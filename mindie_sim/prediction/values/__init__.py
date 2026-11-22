"""Typed tensor value lineage, use-def tracking and value rewriting.

[Role] Own typed value representations, producer/consumer uses, and SSA value rewrites.
[Invariants] Exact producer-consumer edge conservation across rewrites.
[Timeouts & Locks] Pure in-memory analysis; no locks.
[Side-Effect Policy] Pure functional transforms.
"""
from __future__ import annotations
