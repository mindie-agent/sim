"""Immutable timeline, idle-decomposition, and diagnosis records for WaveRecorder.

[Role] Define the schema dataclasses (``RecordedPipelineSpec``, ``PipeIdleRow``,
  ``IntervalRun``, ``PeriodicRun``, ``TileWaveTimeline``, ``BottleneckDiagnosis``,
  ``BufferResidency``, ``FusedPipelineDescriptor``) and exact integer-ns interval
  algebra shared by ``wave_recorder.py``.
[Invariants] Every ``PipeIdleRow`` enforces exact integer conservation
  ``busy_ns + fill_ns + steady_bubble_ns + drain_ns + handoff_ns == span_ns``.
  Timestamp conversion uses IEEE-754 round-half-to-even (``round(t_ms * 1e6)``)
  matching C11 ``msim_ms_to_ns_even``.
[Timeouts & Locks] Pure value objects and O(N log N) interval operations; no I/O.
[Side-Effect Policy] Zero mutable global state; pure functional helpers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

NS_PER_MS = 1_000_000.0
SCHEMA_VERSION = "fused-pipeline-descriptor/1"
SPEC_ID_HEX_LEN = 24
CUBE_PIPE_ROLES = frozenset({"cube", "aic", "fixpipe", "mte1"})
VECTOR_PIPE_ROLES = frozenset({"vec", "vector", "vec_issue", "vec_alu", "vec_sfu", "vec_reduce"})
SINGLE_CAUSE_HANDOFF_KINDS = frozenset({
    "publish", "last_use", "cross_core_workspace", "kernel_handoff", "cross_pipe_handoff",
})
DEFAULT_HEAD_TAIL_LIMIT = 256


def _ms_to_ns(t_ms: float) -> int:
    return int(round(float(t_ms) * NS_PER_MS))


def _union_intervals(intervals: Sequence[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    cleaned = [(lo, hi) for lo, hi in intervals if hi > lo]
    if not cleaned:
        return ()
    cleaned.sort()
    merged: list[tuple[int, int]] = [cleaned[0]]
    for lo, hi in cleaned[1:]:
        p_lo, p_hi = merged[-1]
        if lo <= p_hi:
            if hi > p_hi:
                merged[-1] = (p_lo, hi)
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _clamp_intervals(
    intervals: Sequence[tuple[int, int]], start_ns: int, end_ns: int,
) -> tuple[tuple[int, int], ...]:
    if end_ns <= start_ns or not intervals:
        return ()
    clamped = []
    for lo, hi in intervals:
        c_lo = lo if lo > start_ns else start_ns
        c_hi = hi if hi < end_ns else end_ns
        if c_hi > c_lo:
            clamped.append((c_lo, c_hi))
    return _union_intervals(clamped)


def _subtract_intervals(
    base: Sequence[tuple[int, int]], cut: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not base or not cut:
        return tuple(base)
    out: list[tuple[int, int]] = []
    cut_idx = 0
    n_cut = len(cut)
    for lo, hi in base:
        cur = lo
        while cut_idx < n_cut and cut[cut_idx][1] <= cur:
            cut_idx += 1
        j = cut_idx
        while j < n_cut and cut[j][0] < hi:
            c_lo, c_hi = cut[j]
            if c_lo > cur:
                out.append((cur, c_lo))
            if c_hi > cur:
                cur = c_hi
            if cur >= hi:
                break
            j += 1
        if cur < hi:
            out.append((cur, hi))
    return tuple(out)


def _intersect_intervals(
    left: Sequence[tuple[int, int]], right: Sequence[tuple[int, int]],
) -> tuple[tuple[int, int], ...]:
    if not left or not right:
        return ()
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(left) and j < len(right):
        lo = max(left[i][0], right[j][0])
        hi = min(left[i][1], right[j][1])
        if hi > lo:
            out.append((lo, hi))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return tuple(out)


def _measure(intervals: Sequence[tuple[int, int]]) -> int:
    return sum(hi - lo for lo, hi in intervals)


def _pipe_base_role(pipe_name: str) -> str:
    base = pipe_name.split("@", 1)[0]
    if ":" in base:
        base = base.rsplit(":", 1)[-1]
    return base


def _to_json_obj(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if is_dataclass(value) and not isinstance(value, type):
        return {f.name: _to_json_obj(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _to_json_obj(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_to_json_obj(v) for v in value]
    if isinstance(value, (set, frozenset)):
        return [_to_json_obj(v) for v in sorted(value, key=repr)]
    return str(value)


@dataclass(frozen=True, slots=True)
class RecordedPipelineSpec:
    """Pre-coalesce pipeline specification attached to ``OperatorPlan``."""

    schema_version: str
    spec_id: str
    physical_signature: str
    family: str
    recipe_id: str
    native_abi: str
    source_refs: tuple[Any, ...] = ()
    numerical_contract: str = ""
    members: tuple[str, ...] = ()
    public_ports: tuple[Any, ...] = ()
    solution: Any = None
    nodes: tuple[Any, ...] = ()
    edges: tuple[Any, ...] = ()
    stages: tuple[Any, ...] = ()
    kernel_boundaries: tuple[Any, ...] = ()

    @classmethod
    def create(
        cls,
        *,
        physical_signature: str,
        family: str,
        recipe_id: str,
        native_abi: str,
        source_refs: Sequence[Any] = (),
        numerical_contract: str = "",
        members: Sequence[str] = (),
        public_ports: Sequence[Any] = (),
        solution: Any = None,
        nodes: Sequence[Any] = (),
        edges: Sequence[Any] = (),
        stages: Sequence[Any] = (),
        kernel_boundaries: Sequence[Any] = (),
    ) -> RecordedPipelineSpec:
        payload = json.dumps(
            _to_json_obj({
                "physical_signature": physical_signature,
                "family": family,
                "recipe_id": recipe_id,
                "native_abi": native_abi,
                "source_refs": tuple(source_refs),
                "numerical_contract": numerical_contract,
                "members": tuple(members),
                "solution": solution,
            }),
            sort_keys=True,
            separators=(",", ":"),
        )
        spec_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:SPEC_ID_HEX_LEN]
        return cls(
            schema_version=SCHEMA_VERSION,
            spec_id=spec_id,
            physical_signature=physical_signature,
            family=family,
            recipe_id=recipe_id,
            native_abi=native_abi,
            source_refs=tuple(source_refs),
            numerical_contract=numerical_contract,
            members=tuple(members),
            public_ports=tuple(public_ports),
            solution=solution,
            nodes=tuple(nodes),
            edges=tuple(edges),
            stages=tuple(stages),
            kernel_boundaries=tuple(kernel_boundaries),
        )


@dataclass(frozen=True, slots=True)
class PipeIdleRow:
    """Exact 4-bucket idle decomposition for one physical ``(scope_id, pipe_id)``."""

    scope_id: int
    pipe_id: int
    span_ns: int
    busy_ns: int
    fill_ns: int
    steady_bubble_ns: int
    drain_ns: int
    handoff_ns: int
    inactive_in_window: bool
    pipe_name: str = ""
    owner_time_ns: int = 0

    def __post_init__(self) -> None:
        total = (
            self.busy_ns + self.fill_ns + self.steady_bubble_ns
            + self.drain_ns + self.handoff_ns
        )
        if total != self.span_ns:
            raise ModelingError(
                f"pipe_idle_row_conservation_violated:{self.pipe_id}:{total}!={self.span_ns}"
            )


@dataclass(frozen=True, slots=True)
class IntervalRun:
    """One recorded positive-rate service interval in integer ns."""

    start_ns: int
    end_ns: int
    task_id: int
    pipe_id: int
    scope_id: int
    owner: int
    delivered: float


@dataclass(frozen=True, slots=True)
class PeriodicRun:
    """One proved periodic skip record anchored to its exact floating-point period."""

    anchor_ms: float
    period_ms: float
    repeats: int
    template_id: str
    state_digest: str
    service_pattern: tuple[tuple[str, float, float], ...] = ()
    handoff_pattern: tuple[tuple[str, float, float], ...] = ()
    scope_id: int = 0


@dataclass(frozen=True, slots=True)
class TileWaveTimeline:
    """Bounded head/steady/tail timeline with exact per-pipe idle rows."""

    start_ns: int
    end_ns: int
    head: tuple[IntervalRun, ...]
    steady: tuple[PeriodicRun, ...]
    tail: tuple[IntervalRun, ...]
    pipe_rows: tuple[PipeIdleRow, ...]
    cube_vector_overlap_ns: int


@dataclass(frozen=True, slots=True)
class ResourceRowRef:
    row_id: int
    name: str
    scope_id: int


@dataclass(frozen=True, slots=True)
class WaitWitness:
    task_id: int
    blocker_event: str
    pipe_id: int
    start_ns: int
    end_ns: int
    reason: str


@dataclass(frozen=True, slots=True)
class OwnerPoolPeak:
    owner: int
    pool: str
    peak_bytes: int


@dataclass(frozen=True, slots=True)
class BottleneckDiagnosis:
    limiting_rows: tuple[ResourceRowRef, ...]
    exposed_waits: tuple[WaitWitness, ...]
    peak_allocations: tuple[OwnerPoolPeak, ...]
    critical_events: tuple[str, ...]
    uncovered_stars_ns: int = 0


@dataclass(frozen=True, slots=True)
class BufferResidency:
    pool: str
    owner: int
    slot: int
    generation: int
    size_bytes: int
    grant_ns: int
    release_ns: int | None
    alias_of: str | None = None


@dataclass(frozen=True, slots=True)
class MemberBinding:
    member_id: str
    task_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FusedPipelineDescriptor:
    """Occurrence-bound wave timeline and diagnosis ready for JSON export."""

    spec: Any
    occurrence_key: Any
    semantic_group_id: str
    member_bindings: tuple[MemberBinding, ...]
    timing_scope: Literal["graph_occurrence", "isolated_occurrence"]
    submitted_kernel_count: int
    buffers: tuple[BufferResidency, ...]
    timeline: TileWaveTimeline
    diagnosis: BottleneckDiagnosis

    def to_dict(self) -> dict[str, Any]:
        return _to_json_obj(self)
