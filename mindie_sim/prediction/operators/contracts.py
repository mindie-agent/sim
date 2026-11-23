"""Governed operator-family contracts, planning contexts, and ISA strategy protocols.

[Role] Define the typed interfaces shared by every governed
  ``mindie_sim/prediction/operators/<family>/`` package, its single public
  entrypoint, and any ISA-specific leaf strategies.
[Invariants] Family dispatch consumes explicit ``ComputeArchIdentity`` and
  ``DatapathCapabilitySpec`` records, never SKU strings, ``compute_peaks`` or
  ``service_modes`` sniffing. Every ``ArchStrategy`` declares verified
  ``SourceRef`` provenance.
[Timeouts & Locks] Pure protocol and immutable dataclass declarations; no I/O
  or locks.
[Side-Effect Policy] Zero side effects on import or instantiation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from mindie_sim.schema.datapath import (
    ArchId,
    ComputeArchIdentity,
    DatapathCapabilitySpec,
    SourceRef,
)
from mindie_sim.schema.operators import (
    KernelBinding,
    OpSpec,
    OwnerScheduleSpec,
    TensorPrecisionABI,
)
from mindie_sim.schema.workload import RequestView
from ..mechanism import Geometry, OperatorPlan
from ..program import Recipe, RecipeBook
from ..results import FusedPipelineDescriptor, OpOccurrence
from ..streams import StreamOwners


@dataclass(frozen=True, slots=True)
class OperatorPlanningContext:
    """Immutable planning context delivered to a governed family resolver."""

    occurrence: OpOccurrence
    geometry: Geometry
    datapath: DatapathCapabilitySpec
    recipe_book: RecipeBook | None = None
    requested_abi: str | None = None
    stream_owners: StreamOwners | None = None
    admission_memory_view: Any = None
    allocation_versions: tuple[tuple[str, int], ...] = ()
    requests: tuple[RequestView, ...] = ()
    padded_tokens: int = 0
    layout: str = ""

    @property
    def compute_arch(self) -> ComputeArchIdentity:
        return self.datapath.compute_arch

    @property
    def effective_aic_cores(self) -> int:
        return (
            self.stream_owners.aic_limit
            if self.stream_owners is not None else self.datapath.aic_count
        )

    @property
    def effective_aiv_cores(self) -> int:
        return (
            self.stream_owners.aiv_limit
            if self.stream_owners is not None else self.datapath.aiv_count
        )


@dataclass(frozen=True, slots=True)
class ResolvedOperator:
    """Resolved governed operator binding ready for execution and projection."""

    occurrence: OpOccurrence
    op: OpSpec
    recipe: Recipe
    implementation: str
    sources: tuple[SourceRef, ...] = ()
    descriptor: FusedPipelineDescriptor | None = None
    precision_abi: TensorPrecisionABI | None = None
    kernel_binding: KernelBinding | None = None
    owner_schedule: OwnerScheduleSpec | None = None


class ArchStrategy(Protocol):
    """Contract implemented by governed ``arch22.py`` / ``arch35.py`` leaf modules."""

    @property
    def arch_id(self) -> ArchId: ...

    @property
    def abi(self) -> str: ...

    @property
    def sources(self) -> tuple[SourceRef, ...]: ...

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool: ...

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan: ...


FamilyResolver = Callable[[OpSpec, OperatorPlanningContext], ResolvedOperator]


@dataclass(frozen=True, slots=True)
class OperatorFamilySpec:
    """Single-owner registration record for one governed operator family package."""

    name: str
    families: frozenset[str]
    owned_recipe_ids: frozenset[str]
    resolve: FamilyResolver
    rewrite_rules: tuple[Any, ...] = ()
    traffic_rewrite_capable: bool = False
    delegated_legacy_recipe_ids: frozenset[str] = frozenset()


__all__ = [
    "ArchStrategy",
    "FamilyResolver",
    "OperatorFamilySpec",
    "OperatorPlan",
    "OperatorPlanningContext",
    "ResolvedOperator",
    "SourceRef",
]
