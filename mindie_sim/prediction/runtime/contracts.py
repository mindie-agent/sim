"""Frozen causal-runtime records shared by every T04 producer and consumer.

[Role] Typed identities, immutable bindings and program/catalog records.
[Invariants] Keyword-only construction; frozen records recursively snapshot
  supplied mappings/sequences; no runtime state, cache or policy inference.
[Timeouts & Locks] None: pure values only.
[Side-Effect Policy] None after construction.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, TYPE_CHECKING
if TYPE_CHECKING:
    from .engine import Event
    from mindie_sim.schema.tensors import TensorValue
    from mindie_sim.schema.operators import StateEffect

def empty_map():
    return MappingProxyType({})

@dataclass(frozen=True, slots=True, kw_only=True)
class OccurrenceKey:
    runtime_uid: str
    launch: int
    ordinal: int = 0
    replica: int = 0
    rank: int = 0
    stage: int = 0

@dataclass(frozen=True, slots=True, kw_only=True)
class AllocationKey:
    occurrence: OccurrenceKey
    scope_kind: str
    scope_ordinal: int
    symbol: int
    generation: int = 0

@dataclass(frozen=True, slots=True, kw_only=True)
class BackingKey:
    physical_domain: str
    rank: int
    tier: str
    storage_id: str
    generation: int = 0

@dataclass(frozen=True, slots=True, kw_only=True)
class EventKey:
    runtime_uid: str
    sequence: int
    generation: int = 0


@dataclass(frozen=True)
class LocalBoundaryToken:
    """Completed synthetic boundary marker scoped to the owning RuntimeState."""

    sequence: int
    time: float = 0.0
    runtime_uid: str = ""

    @property
    def done(self) -> bool:
        return True

@dataclass(frozen=True, slots=True, kw_only=True)
class PoolKey:
    physical_domain: str
    rank: int
    tier: str
    name: str
    unit: int = 0
    quantity_unit: str = 'bytes'

@dataclass(frozen=True, slots=True, kw_only=True)
class PortBinding:
    ready: Event
    value: TensorValue | None = None
    allocation: AllocationKey | None = None
    backing: BackingKey | None = None
    byte_ranges: tuple[tuple[int, int], ...] = ()
    access: str = 'read'
    version: int | None = None
    producer: OccurrenceKey | None = None
    output: str = ''

@dataclass(frozen=True, slots=True, kw_only=True)
class PortSpec:
    name: str
    kind: str = 'tensor'
    direction: str = 'in'
    value_descriptor: TensorValue | None = None
    access: str = 'read'
    optional: bool = False
    alias_symbol: int | None = None
    state_target: str | None = None
    state_version_expr: Any = None
    # Output readiness references the actual producing local node, not the parent join.
    producer_node: int | None = None
    producer_boundary: str = 'done'

@dataclass(frozen=True, slots=True, kw_only=True)
class VersionBinding:
    target: str
    version: int
    ready: Event
    value: PortBinding | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class AllocationBinding:
    key: AllocationKey
    pool: PoolKey
    amount: float
    backing: BackingKey | None = None
    byte_ranges: tuple[tuple[int, int], ...] = ()
    reservation_parent: AllocationKey | None = None
    owner: str = ''
    storage: str = ''
    access: str = 'readwrite'
    last_use: Event | None = None
    # False means externally supplied reference; binding alone never reacquires it.
    acquire: bool = True

class ScalarExpr(Protocol):
    def evaluate(self, binding: BindingFrame) -> float: ...
# ScalarExpr implementations must be immutable, deterministic and contain no live runtime.
Scalar = int | float | ScalarExpr

@dataclass(frozen=True, slots=True, kw_only=True)
class EdgeSpec:
    source_node: int | None = None
    external_port: str | None = None
    source_boundary: str = 'done'
    iteration_delta: int = 0
    kind: str = 'data'

@dataclass(frozen=True, slots=True, kw_only=True)
class WorkTerm:
    physical_resource_role: str
    issued: Scalar
    useful: Scalar = 0.0
    unit: str = 'work'
    service_conversion: float = 1.0

@dataclass(frozen=True, slots=True, kw_only=True)
class CapTerm:
    physical_resource_role: str
    rate: Scalar

@dataclass(frozen=True, slots=True, kw_only=True)
class PlacementSpec:
    physical_resource_role: str
    units: tuple[int, ...]

@dataclass(frozen=True, slots=True, kw_only=True)
class AllocationSpec:
    symbol: int
    label: str
    pool_role: str
    amount: Scalar
    scope_kind: str = 'invocation'
    units: tuple[int, ...] = (0,)
    external_port: str | None = None
    condition: str | None = None
    generation: int = 0

@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseSpec:
    symbol: int
    scope_kind: str = 'invocation'
    condition: str | None = None
    last_reader_port: str | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class StageSpec:
    local_id: int
    label: str
    class_id: int = 0
    relative_after: tuple[EdgeSpec, ...] = ()
    stream_symbol: int | None = None
    work: tuple[WorkTerm, ...] = ()
    latency_expr: Scalar = 0.0
    caps: tuple[CapTerm, ...] = ()
    coupled: bool = False
    priority: int = 0
    #: Symmetric owner multiplicity of a merged coupled cohort; the Engine
    #: normalizes joint progress by this count, exactly like ``Stage``.
    cohort_size: int = 1
    cohort_widened: tuple[str, ...] = ()
    placement: tuple[PlacementSpec, ...] = ()
    acquire: tuple[AllocationSpec, ...] = ()
    release: tuple[ReleaseSpec, ...] = ()
    state_effects: tuple[StateEffect, ...] = ()
    output_ports: tuple[str, ...] = ()
    nested_catalog: ProgramCatalog | None = None
    condition: str | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class StageClass:
    class_id: int
    stages: tuple[StageSpec, ...]

@dataclass(frozen=True, slots=True, kw_only=True)
class CatalogGeneration:
    schema_version: str = 'causal-runtime/1'
    registry_generation: str = '0'
    implementation_generation: str = '0'
    hardware_fingerprint: str = ''
    calibration_generation: str = '0'
    structural_guard_fingerprint: str = ''

@dataclass(frozen=True, slots=True, kw_only=True)
class TranslationCertificate:
    rule: str
    producer_generation: str
    region_start: int
    region_end: int
    period: int
    # Pure, canonical immutable producer expressions/rules, never sampled timings.
    coordinate_translation: tuple[tuple[str, int], ...] = ()
    slot_permutation: tuple[int, ...] = ()
    guard_expressions: tuple[Any, ...] = ()
    allocation_translation: tuple[Any, ...] = ()
    external_cut_ports: tuple[str, ...] = ()
    callback_free: bool = True
    hidden_state_free: bool = True

@dataclass(frozen=True, slots=True, kw_only=True)
class FiniteProgramSpec:
    total: int
    slots: int
    classes: tuple[StageClass, ...]
    # Returns class_id, end_exclusive, output_ordinal, panel_ordinal, slot_ordinal.
    index_of: Callable[[int], tuple[int, int, int, int, int]]
    setup: tuple[StageSpec, ...] = ()
    drain: tuple[StageSpec, ...] = ()
    setup_ms: Scalar = 0.0
    minimum_ms: Scalar = 0.0
    translation: TranslationCertificate | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class HostProgramSpec:
    fifo_role: str
    submit_work_ms: Scalar = 0.0
    captured: bool = False
    host_dependency_ports: tuple[str, ...] = ()
    # None means no modeled queue-credit restriction, not zero queue capacity.
    queue_capacity: int | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class ProgramCatalog:
    catalog_id: str
    implementation_id: str
    generation: CatalogGeneration = field(default_factory=CatalogGeneration)
    stages: tuple[StageSpec, ...] = ()
    ports: tuple[PortSpec, ...] = ()
    external_edges: tuple[EdgeSpec, ...] = ()
    allocation_symbols: tuple[AllocationSpec, ...] = ()
    state_schema: tuple[Any, ...] = ()
    finite: FiniteProgramSpec | None = None
    host_spec: HostProgramSpec | None = None
    selected_call_count: int = 1
    semantic_sources: tuple[str, ...] = ()
    guard: Any = None
    translation: TranslationCertificate | None = None

def _freeze_map(value):
    return MappingProxyType({key: _freeze(item) for key, item in dict(value).items()})


def _freeze(value):
    if isinstance(value, Mapping):
        return _freeze_map(value)
    if isinstance(value, tuple):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True, slots=True, kw_only=True)
class BindingFrame:
    occurrence: OccurrenceKey
    geometry: Mapping[str, Any] = field(default_factory=empty_map)
    values: Mapping[str, PortBinding] = field(default_factory=empty_map)
    resources: Mapping[str, str] = field(default_factory=empty_map)
    allocations: Mapping[int, AllocationBinding] = field(default_factory=empty_map)
    state_reads: Mapping[str, VersionBinding] = field(default_factory=empty_map)
    conditions: Mapping[str, bool] = field(default_factory=empty_map)
    placement: Mapping[str, tuple[int, ...]] = field(default_factory=empty_map)
    host_fifo: str | None = None
    stream_bindings: Mapping[int, str] = field(default_factory=empty_map)
    memory_domain: str | None = None
    pool_bindings: Mapping[str, PoolKey] = field(default_factory=empty_map)
    # Reserved runtime coordinates; populated fresh by the finite cursor.
    coordinates: Mapping[str, int] = field(default_factory=empty_map)

    def __post_init__(self):
        object.__setattr__(self, "geometry", _freeze_map(self.geometry))
        object.__setattr__(self, "values", _freeze_map(self.values))
        object.__setattr__(self, "resources", _freeze_map(self.resources))
        object.__setattr__(self, "allocations", _freeze_map(self.allocations))
        object.__setattr__(self, "state_reads", _freeze_map(self.state_reads))
        object.__setattr__(self, "conditions", _freeze_map(self.conditions))
        object.__setattr__(self, "placement", _freeze_map(self.placement))
        object.__setattr__(self, "stream_bindings", _freeze_map(self.stream_bindings))
        object.__setattr__(self, "pool_bindings", _freeze_map(self.pool_bindings))
        object.__setattr__(self, "coordinates", _freeze_map(self.coordinates))

@dataclass(frozen=True, slots=True, kw_only=True)
class Submission:
    occurrence: OccurrenceKey
    start: Event
    done: Event
    ports: Mapping[str, PortBinding] = field(default_factory=empty_map)
    allocations: Mapping[str, AllocationKey] = field(default_factory=empty_map)
    selected_program_id: str = ''

@dataclass(frozen=True, slots=True, kw_only=True)
class ReaderKey:
    allocation: AllocationKey
    event: EventKey
    ordinal: int = 0

@dataclass(frozen=True, slots=True, kw_only=True)
class GrantResult:
    granted: bool
    allocations: tuple[AllocationKey, ...] = ()
    blocked_pools: tuple[PoolKey, ...] = ()
    epoch: int = 0
    reason: str | None = None

@dataclass(frozen=True, slots=True, kw_only=True)
class ReleaseResult:
    released: bool
    deferred: bool = False
    freed_by_pool: tuple[tuple[PoolKey, float], ...] = ()
    epoch: int = 0

# Canonical ledger mutable payload belongs to RuntimeState only. M implements algorithms.
@dataclass(slots=True, kw_only=True)
class CapacityDomainState:
    domain_id: str
    owner_runtime_uid: str | None = None
    capacities: dict[PoolKey, float] = field(default_factory=dict)
    # Grant/backing/reader payloads may use M-private records, referenced only here.
    grants: dict[AllocationKey, Any] = field(default_factory=dict)
    backings: dict[BackingKey, Any] = field(default_factory=dict)
    readers: dict[ReaderKey, Any] = field(default_factory=dict)
    waiters: dict[PoolKey, Any] = field(default_factory=dict)
    current: dict[PoolKey, float] = field(default_factory=dict)
    peaks: dict[PoolKey, float] = field(default_factory=dict)
    epoch: int = 0

# StateEffect/TensorValue remain the original schema classes, not duplicate schemas.
# Event is the actual new runtime Event implementation, never a second protocol wrapper.
