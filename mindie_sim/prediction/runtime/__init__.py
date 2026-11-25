"""Unique causal runtime: frozen contracts, one state root, one Engine.

[Role] Public import surface for every T04 producer/consumer.
[Invariants] Engine/Event are resolved lazily from the single runtime.engine
  implementation; no second engine or compatibility shim is exported.
[Timeouts & Locks] None.
[Side-Effect Policy] Import only.
"""
from __future__ import annotations

from .contracts import (
    AllocationBinding, AllocationKey, BackingKey, BindingFrame, CapacityDomainState,
    CapTerm, CatalogGeneration, EdgeSpec, EventKey, FiniteProgramSpec, GrantResult,
    HostProgramSpec, OccurrenceKey, PlacementSpec, PoolKey, PortBinding, PortSpec,
    ProgramCatalog, ReaderKey, ReleaseResult, ReleaseSpec, Submission,
    TranslationCertificate, VersionBinding, WorkTerm,
)
from .memory_physics import (
    AddressRun, AddressRunPlan, AdmissionMemoryView, BoundAccess, BurstTurnRule,
    ByteLedger, ConstraintTerm, CounterTerm, DirtyWriteback, HbmBusSpec,
    MemoryAccessPattern, ReuseKey, VisibilityContract, bind_memory_access,
    burst_turnaround_efficiency,
)
from .prepared_terms import (
    BoundVectorOp, PreparedCapacityRow, PreparedDelivery, PreparedProgress,
    PreparedTerm, VectorInstructionGroup, allocate_prepared_epoch,
    bind_vector_groups, integrate_prepared_epoch,
)
from .ring_buffer import (
    DepRecord, GenerationEvents, RingBuffer, SlotGenRecord, SlotGenerationId,
    lower_generation,
)
from .tile_solver import (
    LiveBufferSpec, OwnerPoolPeak, SubtileWaveSpec, TileCandidate,
    TileInfeasibleError, TilePipelineSpec, TileRejection, TileShape,
    TileSolution, solve as solve_tiles,
)

_ENGINE_NAMES = frozenset({
    "Allocation", "Engine", "Event", "ExecutionError", "Resource",
    "PreparedStage",
})


def __getattr__(name):
    if name in _ENGINE_NAMES:
        from . import engine
        return getattr(engine, name)
    if name == "PreparedTermsRunner":
        from .accel_terms import PreparedTermsRunner
        return PreparedTermsRunner
    raise AttributeError(name)


__all__ = [
    "AddressRun", "AddressRunPlan", "AdmissionMemoryView", "Allocation",
    "AllocationBinding", "AllocationKey", "BackingKey", "BindingFrame",
    "BoundAccess", "BoundVectorOp",
    "BurstTurnRule", "ByteLedger", "CapTerm", "CapacityDomainState",
    "CatalogGeneration", "ConstraintTerm", "CounterTerm", "DepRecord",
    "DirtyWriteback", "EdgeSpec", "Engine", "Event", "EventKey", "ExecutionError",
    "FiniteProgramSpec", "GenerationEvents",
    "GrantResult", "HbmBusSpec", "HostProgramSpec",
    "LiveBufferSpec", "MemoryAccessPattern", "OccurrenceKey",
    "OwnerPoolPeak", "PlacementSpec", "PoolKey",
    "PortBinding", "PortSpec", "PreparedCapacityRow", "PreparedDelivery",
    "PreparedProgress", "PreparedStage", "PreparedTerm", "PreparedTermsRunner",
    "ProgramCatalog", "ReaderKey", "ReleaseResult",
    "ReleaseSpec", "Resource", "ReuseKey", "RingBuffer",
    "SlotGenRecord", "SlotGenerationId", "Submission", "SubtileWaveSpec",
    "TileCandidate", "TileInfeasibleError", "TilePipelineSpec",
    "TileRejection", "TileShape", "TileSolution",
    "TranslationCertificate", "VectorInstructionGroup", "VersionBinding",
    "VisibilityContract", "WorkTerm",
    "allocate_prepared_epoch", "bind_memory_access", "bind_vector_groups",
    "burst_turnaround_efficiency", "integrate_prepared_epoch",
    "lower_generation", "solve_tiles",
]
