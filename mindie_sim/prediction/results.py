"""Pure immutable domain result and request records for the unified sim SDK.

[Role] Define typed request/evaluation value objects (``StepEvaluation``,
  ``ThreeTierEvaluation``, ``AlignedSequenceTable``, ``FusedPipelineSpec``,
  ``FusedPipelineDescriptor``, ``WaveBreakdown``, SLA/sweep/structure records)
  and deterministic recursive JSON freezing/materialization helpers.
[Invariants] Zero Excel/OOXML or Perfetto serialization code resides here.
  All public result dataclasses are frozen, slotted and finite-number checked.
  ``FusedPipelineSpec`` stays on ``OperatorPlan`` and never enters C11
  ``Stage`` metadata.
[Timeouts & Locks] Pure in-memory value construction and bounded recursive
  JSON projection; no I/O or locks.
[Side-Effect Policy] Never mutates caller mappings or input records.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import Literal, TypeAlias

from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import SourceRef
from .streams import MAIN_STREAM, StreamId, validate_stream_id

Phase = Literal["prefill", "decode"]
Scope = Literal["body", "model"]
Tier = Literal["unmerge", "mainline", "should_achieve"]
WorkUnit = Literal["bytes", "flop", "op", "instruction"]

JsonValue: TypeAlias = (
    "None | bool | int | float | str | tuple[JsonValue, ...] "
    "| Mapping[str, JsonValue]"
)
JsonObject: TypeAlias = Mapping[str, JsonValue]
CaseInput: TypeAlias = Case | Mapping[str, object]
ModelInput: TypeAlias = str | Path | Mapping[str, object]
HardwareInput: TypeAlias = str | Mapping[str, object]

DEFAULT_TARGET_EFFICIENCY = 0.80


def freeze_json(value: object, /) -> JsonValue:
    """Recursively freeze arbitrary JSON-compatible values into immutable types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non_finite_json_float:{value!r}")
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return MappingProxyType({
            f.name: freeze_json(getattr(value, f.name))
            for f in fields(value)
        })
    if isinstance(value, Mapping):
        return MappingProxyType({
            str(key): freeze_json(item) for key, item in value.items()
        })
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(freeze_json(item) for item in sorted(value, key=repr))
    raise TypeError(f"unsupported_json_value:{type(value).__name__}")


def json_payload(value: object, /) -> object:
    """Convert frozen domain dataclasses and immutable proxies into plain JSON objects."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non_finite_json_float:{value!r}")
        return value
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: json_payload(getattr(value, f.name))
            for f in fields(value)
        }
    if isinstance(value, Mapping):
        return {str(key): json_payload(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_payload(item) for item in value]
    if isinstance(value, (set, frozenset)):
        return [json_payload(item) for item in sorted(value, key=repr)]
    raise TypeError(f"unsupported_json_payload:{type(value).__name__}")


@dataclass(frozen=True, slots=True)
class StepRequest:
    """Fixed-geometry step evaluation request."""

    phase: Phase
    tokens: int = 1
    context: int = 0
    sequences: int | tuple[int, ...] = 1
    scope: Scope = "body"
    media_tokens_per_sequence: int | None = None
    draft_prepare: bool = False


@dataclass(frozen=True, slots=True)
class ReplayOptions:
    """Execution replay controls for step and service evaluations."""

    include_submission: bool = False
    isolated: bool = False
    event_limit: int = 0
    deadline_mono: float | None = None


@dataclass(frozen=True, slots=True)
class OpOccurrence:
    """Unique execution-graph occurrence coordinates."""

    replica: int
    stage: int
    rank: int | None
    op_id: str


@dataclass(frozen=True, slots=True)
class WorkItem:
    """Typed single-resource physical work entry."""

    resource: str
    unit: WorkUnit
    amount: float
    dtype: str | None = None


@dataclass(frozen=True, slots=True)
class PipelinePhase:
    """One named stage inside a fused or composite operator pipeline."""

    phase_id: str
    after: tuple[str, ...]
    work: tuple[WorkItem, ...]
    cohort: Literal[0, 1] | None = None
    slot: int | None = None


@dataclass(frozen=True, slots=True)
class FusedPipelineSpec:
    """Pre-lowering pipeline structure attached directly to ``OperatorPlan``."""

    family: str
    implementation: str
    phases: tuple[PipelinePhase, ...]
    sources: tuple[SourceRef, ...] = ()
    repeats: int = 1


@dataclass(frozen=True, slots=True)
class FusedPipelineDescriptor:
    """Evaluated occurrence-bound pipeline descriptor for downstream inspection."""

    owner: OpOccurrence
    family: str
    implementation: str
    member_op_ids: tuple[str, ...]
    absorbed_source_op_ids: tuple[str, ...]
    phases: tuple[PipelinePhase, ...]
    sources: tuple[SourceRef, ...]
    repeats: int = 1


@dataclass(frozen=True, slots=True)
class WaveBreakdown:
    """Exposed critical-path clocks and wave occupancy summary."""

    waves: int
    active_owners: int
    total_owners: int
    tail_owners: int
    cube_ms: float
    vector_ms: float
    memory_ms: float
    communication_ms: float
    control_ms: float
    overlap_ratio: float


@dataclass(frozen=True, slots=True)
class OperatorSequenceRow:
    """One executed operator occurrence on the realized step timeline."""

    occurrence: OpOccurrence
    ordinal: int
    family: str
    implementation: str
    predecessors: tuple[OpOccurrence, ...]
    stream: StreamId
    shape: tuple[int, ...]
    dtype: str
    work: tuple[WorkItem, ...]
    ready_ms: float | None
    start_ms: float | None
    end_ms: float | None
    isolated_ms: float | None = None
    wave: WaveBreakdown | None = None


@dataclass(frozen=True, slots=True)
class AlignedSequenceTable:
    """Ordered operator sequence table aligned across evaluated tiers."""

    graph_id: str
    phase: Phase
    rows: tuple[OperatorSequenceRow, ...]
    tier_spans_ms: tuple[tuple[Tier, float], ...] = ()


@dataclass(frozen=True, slots=True)
class StepEvaluation:
    """Authoritative single-tier step evaluation from ``runtime.Engine``."""

    graph_id: str
    request: StepRequest
    tier: Tier
    span_ms: float
    operators: tuple[OperatorSequenceRow, ...]
    pipelines: tuple[FusedPipelineDescriptor, ...]
    wave_breakdown: WaveBreakdown | None = None
    accel_backend: str = ""
    native_steps: int = 0


@dataclass(frozen=True, slots=True)
class ThreeTierEvaluation:
    """Aligned three-tier (unmerge / mainline / should_achieve) causal evaluation."""

    unmerge: StepEvaluation
    mainline: StepEvaluation
    should_achieve: StepEvaluation
    target_efficiency: float
    aligned_table: AlignedSequenceTable | None = None


ThreeTierComparison = ThreeTierEvaluation


@dataclass(frozen=True, slots=True)
class ServiceEvaluation:
    """Whole-service simulation outcome."""

    summary: JsonObject
    completed_requests: int
    emitted_tokens: int
    span_ms: float


@dataclass(frozen=True, slots=True)
class SlaRequest:
    """SLA concurrency sizing request."""

    step: StepRequest
    latency_ms: float
    max_sequences_per_dp: int
    output_budget: int


@dataclass(frozen=True, slots=True)
class SlaCandidate:
    """One feasible or infeasible SLA sizing candidate."""

    hardware: str
    sequences_per_dp: int
    span_ms: float
    throughput_tokens_s: float
    fits_memory: bool


@dataclass(frozen=True, slots=True)
class SlaEvaluation:
    """SLA solver output with best feasible candidate and full candidate set."""

    best: SlaCandidate | None
    candidates: tuple[SlaCandidate, ...]


@dataclass(frozen=True, slots=True)
class SweepRequest:
    """Topology or P/D disaggregation Pareto sweep request."""

    sla: SlaRequest
    device_counts: tuple[int, ...]
    mode: Literal["topology", "pd"] = "topology"


@dataclass(frozen=True, slots=True)
class TopologyPoint:
    """One Pareto-evaluated parallel deployment point."""

    device_count: int
    parallel: tuple[tuple[str, int], ...]
    decode_parallel: tuple[tuple[str, int], ...] | None
    span_ms: float
    throughput_tokens_s: float
    fits_memory: bool


@dataclass(frozen=True, slots=True)
class TopologySweepResult:
    """Pareto frontier over evaluated topology candidates."""

    evaluated_count: int
    frontier: tuple[TopologyPoint, ...]


@dataclass(frozen=True, slots=True)
class StructureRequest:
    """Pure structural export request without executing timing replay."""

    step: StepRequest
    replicas: tuple[int, ...] | None = None
    expand_calls: bool = True
    native_evidence: bool = False
    source: JsonObject | None = None


@dataclass(frozen=True, slots=True)
class StructureExport:
    """Canonical hashed structure export sealed by ``graph_id``."""

    graph_id: str
    payload: JsonObject


def coerce_stream(value: str | None) -> StreamId:
    """Return a validated stream label, defaulting empty values to ``main``."""
    if value is None or value == "":
        return MAIN_STREAM
    return validate_stream_id(value, "stream")


def aligned_table_for(
    step: StepEvaluation,
    *,
    tier_spans: Sequence[tuple[Tier, float]] = (),
) -> AlignedSequenceTable:
    """Build an ``AlignedSequenceTable`` anchored to one step evaluation."""
    return AlignedSequenceTable(
        graph_id=step.graph_id,
        phase=step.request.phase,
        rows=step.operators,
        tier_spans_ms=tuple(tier_spans),
    )


__all__ = [
    "AlignedSequenceTable",
    "CaseInput",
    "DEFAULT_TARGET_EFFICIENCY",
    "FusedPipelineDescriptor",
    "FusedPipelineSpec",
    "HardwareInput",
    "JsonObject",
    "JsonValue",
    "ModelInput",
    "OpOccurrence",
    "OperatorSequenceRow",
    "Phase",
    "PipelinePhase",
    "ReplayOptions",
    "Scope",
    "ServiceEvaluation",
    "SlaCandidate",
    "SlaEvaluation",
    "SlaRequest",
    "StepEvaluation",
    "StepRequest",
    "StructureExport",
    "StructureRequest",
    "SweepRequest",
    "ThreeTierComparison",
    "ThreeTierEvaluation",
    "Tier",
    "TopologyPoint",
    "TopologySweepResult",
    "WaveBreakdown",
    "WorkItem",
    "WorkUnit",
    "aligned_table_for",
    "coerce_stream",
    "freeze_json",
    "json_payload",
]
