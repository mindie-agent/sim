"""Operator fusion feasibility, UB residency, graph rewriting and recipe synthesis.

[Role] Group fusion contract, UB buffer budgeting, region analysis, and
  fused recipe synthesis under one cohesive package.
[Invariants] Conserves external tensor edges and physical work.
[Timeouts & Locks] Pure in-memory analysis; no locks.
[Side-Effect Policy] Read-only graph analysis and rewriting.
"""
from __future__ import annotations
