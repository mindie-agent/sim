"""Unified distributed tensor-placement & collective-boundary sharding subsystem.

[Role] Consolidate Sequence Parallelism (SP), Local-TP, Context Parallelism
  (DCP/PCP monoid boundary rewrites), Expert Parallelism / MC2 FFN boundary
  rewrites, and MoE routing/EPLB placement under one canonical package.
[Invariants] Prediction operator families under ``prediction/operators/<family>/``
  hold single-rank device kernels while distributed boundary rewrites live here.
[Timeouts & Locks] Pure graph rewrites; no I/O or locks.
[Side-Effect Policy] Pure functional graph transformations.
"""
from __future__ import annotations

import importlib
from typing import Any

_SUBMODULES = frozenset({
    "context_parallel",
    "ffn_rules",
    "local_tp",
    "routing",
    "sequence_parallel",
})
__all__ = sorted(_SUBMODULES)


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        mod = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = mod
        return mod
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
