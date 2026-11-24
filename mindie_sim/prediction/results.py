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
from typing import TYPE_CHECKING, Literal, TypeAlias

from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.scenario import ExecutionStage, ScenarioContract
from mindie_sim.schema.stack import STACK_TRACKS, RuntimeQuantPlan, StackLoweringPlan, StackTrack
from mindie_sim.schema.workload import IterationPlan, RequestView
from .streams import MAIN_STREAM, StreamId, StreamPlan, validate_stream_id

if TYPE_CHECKING:
    from mindie_sim.feature_mechanisms import SoftwareStackSpec
    from mindie_sim.prediction.capacity.ledger import ResidentMemoryReceipt

Phase = Literal["prefill", "decode"]
Scope = Literal["body", "model"]
Tier = Literal["unmerge", "mainline", "should_achieve"]
WorkUnit = Literal["bytes", "flop", "op", "instruction"]
OriginKind = Literal["caller", "stack_profile", "body_default"]
DEFAULT_STACK_NAME = "vllm-ascend"
NS_PER_MS = 1_000_000

CANONICAL_STAGE_IDS: tuple[str, ...] = (
    "before_merged",
    "stage1_ops_config",
    "stage2_unmerged_pr",
    "stage3_target_after",
)
CANONICAL_STAGE_DELTA_COUNT = len(CANONICAL_STAGE_IDS) - 1
StageId = Literal[
    "before_merged",
    "stage1_ops_config",
    "stage2_unmerged_pr",
    "stage3_target_after",
]
STAGE_COMMIT_STATUSES: frozenset[str] = frozenset({
    "committed", "infeasible", "rejected_by_policy", "skipped",
})
StageCommitStatus = Literal["committed", "infeasible", "rejected_by_policy", "skipped"]
COMPARISON_POLICIES: frozenset[str] = frozenset({
    "monotonic_optimization", "diagnostic_unconstrained", "simulation_only",
})
ComparisonPolicy = Literal["monotonic_optimization", "diagnostic_unconstrained", "simulation_only"]

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
        default=None, repr=False, compare=False, kw_only=True, metadata={"wire": False},
    )
    streams: tuple[StreamPlan, ...] = field(
        default=(), repr=False, compare=False, kw_only=True, metadata={"wire": False},
    )
    execution_stage: ExecutionStage | None = field(
        default=None, kw_only=True, metadata={"wire": False},
    )
    stack_id: str = field(
        default="", kw_only=True, metadata={"wire": False},
    )
    scenario_digest: str = field(
        default="", kw_only=True, metadata={"wire": False},
    )
    quant_plan_digest: str = field(
        default="", kw_only=True, metadata={"wire": False},
    )
    lowering_plan_digest: str = field(
        default="", kw_only=True, metadata={"wire": False},
    )
    primitive_table_digest: str = field(
        default="", kw_only=True, metadata={"wire": False},
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
        seam_meta = (
            self.execution_stage,
            self.stack_id,
            self.scenario_digest,
            self.quant_plan_digest,
            self.lowering_plan_digest,
            self.primitive_table_digest,
        )
        return (_unpickle_step_evaluation, (wire_args, self.streams, pt_state, seam_meta))


def _unpickle_step_evaluation(
    wire_args: tuple[object, ...],
    streams: tuple[StreamPlan, ...],
    pt_state: object,
    seam_meta: tuple[object, ...] = (None, "", "", "", "", ""),
) -> StepEvaluation:
    if isinstance(pt_state, tuple) and pt_state and pt_state[0] == "primitive_table":
        from mindie_sim.primitives.schema import parse_primitive_table
        _, env_dict, warnings = pt_state
        pt: object | None = parse_primitive_table(env_dict, verify_envelope=False)
        if warnings:
            object.__setattr__(pt, "warnings", tuple(warnings))
    else:
        pt = pt_state
    stage, stack_id, scen_dig, quant_dig, low_dig, pt_dig = seam_meta
    return StepEvaluation(
        *wire_args,
        primitive_table=pt,
        streams=streams,
        execution_stage=stage,  # type: ignore[arg-type]
        stack_id=str(stack_id),
        scenario_digest=str(scen_dig),
        quant_plan_digest=str(quant_dig),
        lowering_plan_digest=str(low_dig),
        primitive_table_digest=str(pt_dig),
    )


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


@dataclass(frozen=True, slots=True)
class RankOps:
    """One replica's immutable operator sequence."""

    replica: int
    ops: tuple[OpSpec, ...]


@dataclass(frozen=True, slots=True)
class LoweringRequest:
    """Unified lowering input covering fixed step, serving iterations, or explicit ops."""

    step: StepRequest | None = None
    iterations: tuple[tuple[int, IterationPlan], ...] = ()
    declared_ops: tuple[RankOps, ...] = ()
    tier: Tier = "mainline"
    apply_body_defaults: bool = True
    production_capture: bool = False
    stack: str | None = None
    scenario_contract: ScenarioContract | None = None
    runtime_quant_plan: RuntimeQuantPlan | None = None
    stack_lowering_plan: StackLoweringPlan | None = None

    def __post_init__(self) -> None:
        active = int(self.step is not None) + int(bool(self.iterations)) + int(bool(self.declared_ops))
        if active != 1:
            raise ModelingError(f"lowering_request_requires_exactly_one_input:{active}")
        if self.tier not in {"unmerge", "mainline", "should_achieve"}:
            raise ModelingError(f"lowering_request_invalid_tier:{self.tier}")
        if self.scenario_contract is not None and not isinstance(self.scenario_contract, ScenarioContract):
            raise ModelingError("lowering_request_invalid_scenario_contract")
        if self.runtime_quant_plan is not None and not isinstance(self.runtime_quant_plan, RuntimeQuantPlan):
            raise ModelingError("lowering_request_invalid_runtime_quant_plan")
        if self.stack_lowering_plan is not None and not isinstance(self.stack_lowering_plan, StackLoweringPlan):
            raise ModelingError("lowering_request_invalid_stack_lowering_plan")
        if (
            self.scenario_contract is not None
            and self.runtime_quant_plan is not None
            and self.scenario_contract.quant_plan.plan_digest != self.runtime_quant_plan.plan_digest
        ):
            raise ModelingError("lowering_request_quant_plan_mismatch")
        if (
            self.scenario_contract is not None
            and self.stack_lowering_plan is not None
            and self.scenario_contract.lowering_plan.plan_digest != self.stack_lowering_plan.plan_digest
        ):
            raise ModelingError("lowering_request_lowering_plan_mismatch")


@dataclass(frozen=True, slots=True)
class ResolvedBase:
    """Resolved immutable case, stack binding, and provenance ready for rewrite."""

    case: Case
    request: LoweringRequest
    stack: SoftwareStackSpec | None
    recipe_origins: tuple[tuple[str, OriginKind], ...]
    sources: tuple[SourceRef, ...]
    scenario_contract: ScenarioContract | None = None
    runtime_quant_plan: RuntimeQuantPlan | None = None
    stack_lowering_plan: StackLoweringPlan | None = None


def _exact_int(value: object, code: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelingError(f"{code}:{value!r}")
    if minimum is not None and value < minimum:
        raise ModelingError(f"{code}:{value!r}")
    return int(value)


def _req_nonempty_str(value: object, code: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelingError(code)
    return value.strip()


@dataclass(frozen=True, slots=True)
class SubStageReceipt:
    """One named sub-stage exposed-time attribution entry inside a ``StageReceipt``."""

    substage_id: str
    owner: str
    exposed_ns: int
    stream: StreamId = MAIN_STREAM

    def __post_init__(self) -> None:
        object.__setattr__(self, "substage_id", _req_nonempty_str(self.substage_id, "substage_receipt_id_required"))
        object.__setattr__(self, "owner", _req_nonempty_str(self.owner, "substage_receipt_owner_required"))
        object.__setattr__(self, "exposed_ns", _exact_int(self.exposed_ns, "substage_receipt_exposed_ns_invalid", minimum=0))
        object.__setattr__(self, "stream", coerce_stream(self.stream))


@dataclass(frozen=True, slots=True)
class StageReceipt:
    """Exact integer-nanosecond execution and conservation receipt for one canonical stage."""

    stage_id: StageId
    stack_track: StackTrack
    start_tick_ns: int
    end_tick_ns: int
    span_ns: int
    l_star_ns: int
    l_star_valid: bool
    execution_key_sha256: str
    plan_sha256: str
    idle_or_wait_ns: int
    exposed_ns_by_occ: tuple[tuple[str, int], ...] = ()
    committed_case_digest: str = ""
    status: StageCommitStatus = "committed"
    primitive_table_digest: str = ""
    step_eval: StepEvaluation | None = None
    memory_receipt: ResidentMemoryReceipt | None = None
    sub_stage_receipts: tuple[SubStageReceipt, ...] = ()
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        if self.stage_id not in CANONICAL_STAGE_IDS:
            raise ModelingError(f"stage_receipt_stage_id_invalid:{self.stage_id!r}")
        if self.stack_track not in STACK_TRACKS:
            raise ModelingError(f"stage_receipt_stack_track_invalid:{self.stack_track!r}")
        if self.status not in STAGE_COMMIT_STATUSES:
            raise ModelingError(f"stage_receipt_status_invalid:{self.status!r}")
        start_ns = _exact_int(self.start_tick_ns, "stage_receipt_start_tick_invalid", minimum=0)
        end_ns = _exact_int(self.end_tick_ns, "stage_receipt_end_tick_invalid", minimum=0)
        span_ns = _exact_int(self.span_ns, "stage_receipt_span_ns_invalid", minimum=0)
        l_star_ns = _exact_int(self.l_star_ns, "stage_receipt_l_star_ns_invalid", minimum=0)
        idle_ns = _exact_int(self.idle_or_wait_ns, "stage_receipt_idle_ns_invalid", minimum=0)
        if not isinstance(self.l_star_valid, bool):
            raise ModelingError("stage_receipt_l_star_valid_flag_invalid")
        if end_ns < start_ns or span_ns != end_ns - start_ns:
            raise ModelingError(
                f"stage_receipt_tick_span_mismatch:start={start_ns}:end={end_ns}:span={span_ns}"
            )
        occ_rows = tuple(
            (_req_nonempty_str(occ_id, "stage_receipt_occ_id_required"),
             _exact_int(ns, "stage_receipt_occ_ns_invalid", minimum=0))
            for occ_id, ns in self.exposed_ns_by_occ
        )
        sub_rows = tuple(self.sub_stage_receipts)
        if not all(isinstance(row, SubStageReceipt) for row in sub_rows):
            raise ModelingError("stage_receipt_sub_stage_receipts_invalid")
        object.__setattr__(self, "exposed_ns_by_occ", occ_rows)
        object.__setattr__(self, "sub_stage_receipts", sub_rows)
        if self.step_eval is not None and not isinstance(self.step_eval, StepEvaluation):
            raise ModelingError("stage_receipt_step_eval_invalid")
        if self.status == "committed":
            _req_nonempty_str(self.execution_key_sha256, "stage_receipt_execution_key_required")
            _req_nonempty_str(self.primitive_table_digest, "stage_receipt_primitive_table_digest_required")
            if (
                self.step_eval is not None
                and self.step_eval.primitive_table_digest != self.primitive_table_digest
            ):
                raise ModelingError("stage_receipt_primitive_table_digest_mismatch")
            if not self.l_star_valid:
                raise ModelingError(f"stage_receipt_committed_requires_valid_l_star:{self.stage_id}")
            if l_star_ns > span_ns:
                raise ModelingError(
                    f"stage_receipt_l_star_exceeds_span:l_star={l_star_ns}>span={span_ns}"
                )
            accounted = sum(ns for _, ns in occ_rows) + idle_ns
            if accounted != span_ns:
                raise ModelingError(
                    f"stage_receipt_observer_conservation_failed:stage={self.stage_id}:"
                    f"accounted={accounted}!=span={span_ns}:residual={span_ns - accounted}"
                )
        else:
            if (
                self.step_eval is not None
                and self.step_eval.primitive_table_digest
                and self.primitive_table_digest
                and self.step_eval.primitive_table_digest != self.primitive_table_digest
            ):
                raise ModelingError("stage_receipt_primitive_table_digest_mismatch")
            if not isinstance(self.rejection_reason, str) or not self.rejection_reason.strip():
                raise ModelingError(f"stage_receipt_non_committed_requires_reason:{self.stage_id}:{self.status}")

    @property
    def span_ms(self) -> float:
        return self.span_ns / float(NS_PER_MS)

    @property
    def l_star_ms(self) -> float:
        return self.l_star_ns / float(NS_PER_MS)

    @property
    def headroom_ns(self) -> int:
        return self.span_ns - self.l_star_ns


@dataclass(frozen=True, slots=True)
class EngineStepReceipt:
    """Single-Engine 4-stage causal waterfall receipt with exact 0 ns integer telescoping."""

    scenario_digest: str
    primitive_table_digest: str
    comparison_policy: ComparisonPolicy
    stages: tuple[StageReceipt, ...]
    telescoping_deltas_ns: tuple[int, ...] = ()
    endpoints: tuple[tuple[str, int], ...] = ()
    final_bound_ns: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "scenario_digest", _req_nonempty_str(self.scenario_digest, "engine_receipt_scenario_digest_required"))
        object.__setattr__(
            self, "primitive_table_digest", _req_nonempty_str(self.primitive_table_digest, "engine_receipt_primitive_digest_required")
        )
        if self.comparison_policy not in COMPARISON_POLICIES:
            raise ModelingError(f"engine_receipt_comparison_policy_invalid:{self.comparison_policy!r}")
        stages = tuple(self.stages)
        if not stages or not all(isinstance(s, StageReceipt) for s in stages):
            raise ModelingError("engine_receipt_stages_invalid")
        stage_ids = [s.stage_id for s in stages]
        if len(set(stage_ids)) != len(stage_ids):
            raise ModelingError("engine_receipt_duplicate_stage_id")
        order_indices = [CANONICAL_STAGE_IDS.index(sid) for sid in stage_ids]
        if order_indices != sorted(order_indices) or stage_ids[0] != CANONICAL_STAGE_IDS[0]:
            raise ModelingError(f"engine_receipt_stage_order_invalid:{stage_ids!r}")
        for stage in stages:
            if stage.status == "committed" and stage.primitive_table_digest != self.primitive_table_digest:
                raise ModelingError(
                    f"engine_receipt_shared_primitive_table_violated:{stage.stage_id}"
                )
            if stage.primitive_table_digest and stage.primitive_table_digest != self.primitive_table_digest:
                raise ModelingError(
                    f"engine_receipt_shared_primitive_table_violated:{stage.stage_id}"
                )
            if (
                stage.step_eval is not None
                and (
                    stage.status == "committed"
                    or bool(stage.step_eval.primitive_table_digest)
                )
                and stage.step_eval.primitive_table_digest != self.primitive_table_digest
            ):
                raise ModelingError(
                    f"engine_receipt_step_eval_primitive_table_violated:{stage.stage_id}"
                )
        object.__setattr__(self, "stages", stages)
        if self.comparison_policy == "simulation_only":
            stage3 = self.stage_for("stage3_target_after")
            if stage3 is not None and stage3.status == "committed":
                raise ModelingError("engine_receipt_simulation_forbids_committed_stage3_target")
        self._finalize_telescoping_and_bounds()
        self.assert_zero_ns_empty_identity()
        if self.comparison_policy == "monotonic_optimization":
            self.assert_stage_monotonicity()

    def _finalize_telescoping_and_bounds(self) -> None:
        by_id = {s.stage_id: s for s in self.stages}
        base = by_id[CANONICAL_STAGE_IDS[0]]
        if base.status != "committed":
            raise ModelingError("engine_receipt_before_merged_must_be_committed")
        spans: list[int] = [base.span_ns]
        last_committed = base
        for sid in CANONICAL_STAGE_IDS[1:]:
            rec = by_id.get(sid)
            if rec is not None and rec.status == "committed":
                spans.append(rec.span_ns)
                last_committed = rec
            else:
                spans.append(spans[-1])
        expected_deltas = tuple(spans[idx] - spans[idx + 1] for idx in range(CANONICAL_STAGE_DELTA_COUNT))
        if not self.telescoping_deltas_ns:
            object.__setattr__(self, "telescoping_deltas_ns", expected_deltas)
        else:
            provided = tuple(_exact_int(d, "engine_receipt_delta_invalid") for d in self.telescoping_deltas_ns)
            if len(provided) != CANONICAL_STAGE_DELTA_COUNT or provided != expected_deltas:
                raise ModelingError(
                    f"engine_receipt_telescoping_mismatch:got={provided}:expected={expected_deltas}"
                )
            object.__setattr__(self, "telescoping_deltas_ns", provided)
        if sum(self.telescoping_deltas_ns) != spans[0] - spans[-1]:
            raise ModelingError("engine_receipt_telescoping_sum_mismatch")
        expected_endpoints = tuple((s.stage_id, s.span_ns) for s in self.stages)
        if not self.endpoints:
            object.__setattr__(self, "endpoints", expected_endpoints)
        else:
            pts = tuple((_req_nonempty_str(k, "engine_receipt_endpoint_key_invalid"), _exact_int(v, "engine_receipt_endpoint_ns_invalid", minimum=0)) for k, v in self.endpoints)
            if pts != expected_endpoints:
                raise ModelingError(
                    f"engine_receipt_endpoints_mismatch:got={pts}:expected={expected_endpoints}"
                )
            object.__setattr__(self, "endpoints", pts)
        bound = _exact_int(self.final_bound_ns, "engine_receipt_final_bound_invalid", minimum=0)
        if bound == 0 and last_committed.l_star_ns > 0:
            bound = last_committed.l_star_ns
            object.__setattr__(self, "final_bound_ns", bound)
        if bound > last_committed.span_ns:
            raise ModelingError(
                f"engine_receipt_final_bound_exceeds_span:bound={bound}>span={last_committed.span_ns}"
            )

    def stage_for(self, stage_id: StageId) -> StageReceipt | None:
        for stage in self.stages:
            if stage.stage_id == stage_id:
                return stage
        return None

    def assert_stage_monotonicity(self) -> bool:
        """Verify non-increasing integer span_ns across committed stages and valid L* bounds."""
        committed = [s for s in self.stages if s.status == "committed"]
        for prev, curr in zip(committed, committed[1:]):
            if curr.span_ns > prev.span_ns:
                raise ModelingError(
                    f"engine_receipt_stage_monotonicity_violated:"
                    f"{prev.stage_id}({prev.span_ns})<{curr.stage_id}({curr.span_ns})"
                )
        for stage in committed:
            if not stage.l_star_valid or stage.l_star_ns > stage.span_ns:
                raise ModelingError(f"engine_receipt_stage_l_star_invalid:{stage.stage_id}")
        return True

    def assert_zero_ns_empty_identity(self) -> bool:
        """Verify identical execution keys and plans yield an exact 0 ns span delta."""
        committed = [s for s in self.stages if s.status == "committed"]
        for prev, curr in zip(committed, committed[1:]):
            same_key = prev.execution_key_sha256 == curr.execution_key_sha256
            same_plan = prev.plan_sha256 == curr.plan_sha256
            if same_key and same_plan and prev.span_ns != curr.span_ns:
                raise ModelingError(
                    f"engine_receipt_empty_identity_nonzero_delta:"
                    f"{prev.stage_id}({prev.span_ns})!={curr.stage_id}({curr.span_ns})"
                )
        return True


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
    "CANONICAL_STAGE_DELTA_COUNT",
    "CANONICAL_STAGE_IDS",
    "COMPARISON_POLICIES",
    "CaseInput",
    "ComparisonPolicy",
    "DEFAULT_STACK_NAME",
    "EngineStepReceipt",
    "FrozenJsonMapping",
    "FusedPipelineDescriptor",
    "FusedPipelineSpec",
    "HardwareInput",
    "JsonObject",
    "JsonValue",
    "LoweringRequest",
    "ModelInput",
    "NS_PER_MS",
    "OpOccurrence",
    "OperatorSequenceRow",
    "OriginKind",
    "Phase",
    "PipelinePhase",
    "RankOps",
    "ReplayOptions",
    "ResolvedBase",
    "STAGE_COMMIT_STATUSES",
    "Scope",
    "ServiceEvaluation",
    "SlaCandidate",
    "SlaEvaluation",
    "SlaRequest",
    "StageCommitStatus",
    "StageId",
    "StageReceipt",
    "StepEvaluation",
    "StepRequest",
    "StructureExport",
    "StructureRequest",
    "SubStageReceipt",
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
