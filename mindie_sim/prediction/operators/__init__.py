"""Governed operator family packages and built-in family discovery.

[Role] Expose the single-owner family contracts, registry, dispatcher, and
  explicit built-in family loader for ``mindie_sim/prediction/operators/*``.
[Invariants] Imports strictly family package entrypoints (never internal
  ``arch22``/``arch35`` leaf modules). Importing this namespace executes no
  device code, calibration I/O, or implicit global registration.
[Timeouts & Locks] Pure imports and tuple assembly; no I/O or locks.
[Side-Effect Policy] Zero side effects beyond standard module loading.
"""
from __future__ import annotations

from .contracts import (
    ArchStrategy,
    FamilyResolver,
    OperatorFamilySpec,
    OperatorPlan,
    OperatorPlanningContext,
    ResolvedOperator,
    SourceRef,
)
from .registry import (
    FamilyDispatcher,
    FamilyRegistry,
    builtin_family_registry,
)


def load_builtin_families() -> tuple[OperatorFamilySpec, ...]:
    """Return every migrated governed operator family specification."""
    from . import rotary

    return (rotary.register_family(),)


__all__ = [
    "ArchStrategy",
    "FamilyDispatcher",
    "FamilyRegistry",
    "FamilyResolver",
    "OperatorFamilySpec",
    "OperatorPlan",
    "OperatorPlanningContext",
    "ResolvedOperator",
    "SourceRef",
    "builtin_family_registry",
    "load_builtin_families",
]
