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
    from . import (
        attention,
        grouped_moe,
        hyper_connections,
        linear_attention,
        mc2,
        mla_prolog,
        paged_mla,
        rotary,
        sparse_indexer,
    )

    return (
        rotary.register_family(),
        mla_prolog.register_family(),
        paged_mla.register_family(),
        attention.register_family(),
        sparse_indexer.register_family(),
        grouped_moe.register_family(),
        linear_attention.register_family(),
        hyper_connections.register_family(),
        mc2.register_family(),
    )


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
