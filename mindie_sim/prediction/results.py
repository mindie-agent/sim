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
from dataclasses import dataclass, field, fields, is_dataclass
from fractions import Fraction
import math
from pathlib import Path
from typing import Literal, TypeAlias

from mindie_sim.schema.case import Case
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.workload import RequestView
from .streams import MAIN_STREAM, StreamId, StreamPlan, validate_stream_id

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


class FrozenJsonMapping(Mapping[str, JsonValue]):
    """Picklable immutable JSON object mapping safe under ``spawn`` multiprocessing."""

    __slots__ = ("_data",)

    def __init__(
        self,
        items: Mapping[str, JsonValue] | Sequence[tuple[str, JsonValue]] = (),
        /,
    ) -> None:
        object.__setattr__(self, "_data", dict(items))

    def __getitem__(self, key: str) -> JsonValue:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenJsonMapping({self._data!r})"

    def __reduce__(self):
        return (FrozenJsonMapping, (tuple(self._data.items()),))


def freeze_json(value: object, /) -> JsonValue:
    """Recursively freeze arbitrary JSON-compatible values into immutable types."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non_finite_json_float:{value!r}")
        return value
    if isinstance(value, Fraction):
        return FrozenJsonMapping({
            "numerator": value.numerator,
            "denominator": value.denominator,
        })
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return FrozenJsonMapping({
            f.name: freeze_json(getattr(value, f.name))
            for f in fields(value)
            if f.metadata.get("wire", True)
        })
    if isinstance(value, Mapping):
        return FrozenJsonMapping({
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
    if isinstance(value, Fraction):
        return {
            "numerator": value.numerator,
            "denominator": value.denominator,
        }
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: json_payload(getattr(value, f.name))
            for f in fields(value)
            if f.metadata.get("wire", True)
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
    requests: tuple[RequestView, ...] = ()
    padded_tokens: int = 0
    layout: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.requests, tuple):
            object.__setattr__(self, "requests", tuple(self.requests))


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
    primitive_table: object | None = field(
        default=None, repr=False, compare=False, metadata={"wire": False},
    )
    streams: tuple[StreamPlan, ...] = field(
        default=(), repr=False, compare=False, metadata={"wire": False},
    )

    def __reduce__(self):
        # ``PrimitiveTable`` wraps ``MappingProxyType`` (unpicklable), so serialize
        # its plain envelope dict alongside any rescaled warnings.
        pt = self.primitive_table
        if hasattr(pt, "to_envelope_dict"):
            pt_state: object = (
                "primitive_table",
                pt.to_envelope_dict(),
                tuple(getattr(pt, "warnings", ())),
            )
        else:
            pt_state = pt
        wire_args = (
            self.graph_id, self.request, self.tier, self.span_ms,
            self.operators, self.pipelines, self.wave_breakdown,
            self.accel_backend, self.native_steps,
        )
        return (_unpickle_step_evaluation, (wire_args, self.streams, pt_state))


def _unpickle_step_evaluation(
    wire_args: tuple[object, ...],
    streams: tuple[StreamPlan, ...],
    pt_state: object,
) -> StepEvaluation:
    if isinstance(pt_state, tuple) and pt_state and pt_state[0] == "primitive_table":
        from mindie_sim.primitives.schema import parse_primitive_table
        _, env_dict, warnings = pt_state
        pt: object | None = parse_primitive_table(env_dict, verify_envelope=False)
        if warnings:
            object.__setattr__(pt, "warnings", tuple(warnings))
    else:
        pt = pt_state
    return StepEvaluation(*wire_args, primitive_table=pt, streams=streams)


@dataclass(frozen=True, slots=True)
class ThreeTierEvaluation:
    """Aligned three-tier (unmerge / mainline / should_achieve) causal evaluation."""

    unmerge: StepEvaluation
    mainline: StepEvaluation
    should_achieve: StepEvaluation
    certified_lower_bound_ms: float = 0.0
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
    "FrozenJsonMapping",
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
