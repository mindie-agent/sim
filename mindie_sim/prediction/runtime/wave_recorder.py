"""Five-point WaveRecorder and exact 4-bucket idle decomposition for Engine runs.

[Role] Observe Engine execution across the 5 canonical points (submit/coalesce,
  admission, positive-rate service, allocation grant/release, proved periodic
  skip) and emit ``FusedPipelineDescriptor`` with exact integer-ns 4-bucket idle
  accounting (``fill``, ``steady_bubble``, ``drain``, ``handoff``).
[Invariants] For every physical ``PipeIdleRow`` on window ``I = [t0, t1)``,
  ``busy_ns + fill_ns + steady_bubble_ns + drain_ns + handoff_ns == span_ns``
  holds as an exact integer identity. Observer attachment never alters Engine
  scheduling, rates, or completion timestamps. Periodic skips translate pattern
  endpoints from original ``(anchor_ms, period_ms)`` without cumulative rounding
  drift and never paint skipped bubbles as busy. Zero Excel/OOXML/Perfetto deps.
[Timeouts & Locks] Bounded interval union arithmetic; no I/O or locks.
[Side-Effect Policy] Mutates only this recorder's bounded observation buffers.
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


def _pipe_base_role(pipe_name: str) -> str:
    base = pipe_name.split("@", 1)[0]
    if ":" in base:
        base = base.rsplit(":", 1)[-1]
    return base


DEFAULT_HEAD_TAIL_LIMIT = 256


class WaveRecorder:
    """Five-point bounded observer producing ``FusedPipelineDescriptor``."""

    def __init__(
        self,
        *,
        declared_pipes: Sequence[str] = (),
        scope_id: int = 0,
        max_head_tail: int = DEFAULT_HEAD_TAIL_LIMIT,
    ) -> None:
        self.scope_id = int(scope_id)
        self.max_head_tail = max(1, int(max_head_tail))
        self._pipe_ids: dict[str, int] = {}
        self._pipe_names: dict[int, str] = {}
        for name in declared_pipes:
            self.register_pipe(name)
        self._member_tasks: dict[str, list[int]] = {}
        self._submitted_tasks: set[int] = set()
        self._admitted: dict[int, tuple[float, tuple[int, ...], str, int, tuple[str, ...]]] = {}
        self._service_by_pipe: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._owner_time_by_pipe: dict[tuple[int, int], int] = {}
        self._handoff_by_pipe: dict[tuple[int, int], list[tuple[int, int]]] = {}
        self._wait_witnesses: list[WaitWitness] = []
        self._head: list[IntervalRun] = []
        self._tail: list[IntervalRun] = []
        self._steady: list[PeriodicRun] = []
        self._live_allocs: dict[tuple[str, int, int, int], BufferResidency] = {}
        self._closed_allocs: list[BufferResidency] = []
        self._live_pool_bytes: dict[tuple[int, str], int] = {}
        self._peak_pool_bytes: dict[tuple[int, str], int] = {}
        self._critical_events: list[str] = []
        self._min_ns: int | None = None
        self._max_ns: int | None = None

    def register_pipe(self, pipe_name: str) -> int:
        pid = self._pipe_ids.get(pipe_name)
        if pid is None:
            pid = len(self._pipe_ids)
            self._pipe_ids[pipe_name] = pid
            self._pipe_names[pid] = pipe_name
        return pid

    def _note_bounds(self, lo_ns: int, hi_ns: int) -> None:
        if self._min_ns is None or lo_ns < self._min_ns:
            self._min_ns = lo_ns
        if self._max_ns is None or hi_ns > self._max_ns:
            self._max_ns = hi_ns

    # Point 1: submit / coalesce
    def on_submit(
        self,
        task_id: int,
        stage_id: str,
        aliases: Sequence[str] = (),
        predecessors: Sequence[int] = (),
        signals: Sequence[str] = (),
        occurrence: Any = None,
    ) -> None:
        del predecessors, occurrence
        self._submitted_tasks.add(int(task_id))
        members = tuple(aliases) if aliases else (str(stage_id),)
        for member in members:
            self._member_tasks.setdefault(str(member), []).append(int(task_id))
        for sig in signals:
            if sig and sig not in self._critical_events:
                self._critical_events.append(str(sig))

    # Point 2: admission / blocker transition
    def on_admit(
        self,
        task_id: int,
        t_ms: float,
        owners: Sequence[int] = (),
        stream: str = "main",
        generation: int = 0,
        blockers: Sequence[tuple[str, str, float, str]] = (),
    ) -> None:
        """Record task admission and optional exposed single-cause handoff waits.

        Each ``blockers`` entry is ``(pipe_name, blocker_event, ready_ms, reason)``.
        Only when ``len(blockers) == 1`` and ``reason`` is a genuine cross-pipe
        handoff wait does ``[ready_ms, t_ms)`` enter ``H_0``.
        """
        admit_ns = _ms_to_ns(t_ms)
        self._admitted[int(task_id)] = (
            float(t_ms), tuple(int(o) for o in owners), str(stream), int(generation),
            tuple(b[1] for b in blockers),
        )
        if len(blockers) == 1:
            pipe_name, blocker_ev, ready_ms, reason = blockers[0]
            if reason in SINGLE_CAUSE_HANDOFF_KINDS:
                start_ns = _ms_to_ns(ready_ms)
                if admit_ns > start_ns:
                    pid = self.register_pipe(str(pipe_name))
                    key = (self.scope_id, pid)
                    self._handoff_by_pipe.setdefault(key, []).append((start_ns, admit_ns))
                    self._note_bounds(start_ns, admit_ns)
                    self._wait_witnesses.append(WaitWitness(
                        task_id=int(task_id),
                        blocker_event=str(blocker_ev),
                        pipe_id=pid,
                        start_ns=start_ns,
                        end_ns=admit_ns,
                        reason=str(reason),
                    ))

    # Point 3: positive-rate service
    def on_service(
        self,
        task_id: int,
        physical_row: str,
        owner: int,
        t0_ms: float,
        t1_ms: float,
        delivered: float,
        *,
        scope_id: int | None = None,
    ) -> None:
        if delivered <= 0.0 or t1_ms <= t0_ms:
            return
        lo_ns = _ms_to_ns(t0_ms)
        hi_ns = _ms_to_ns(t1_ms)
        if hi_ns <= lo_ns:
            return
        sid = self.scope_id if scope_id is None else int(scope_id)
        pid = self.register_pipe(str(physical_row))
        key = (sid, pid)
        self._service_by_pipe.setdefault(key, []).append((lo_ns, hi_ns))
        self._owner_time_by_pipe[key] = self._owner_time_by_pipe.get(key, 0) + (hi_ns - lo_ns)
        self._note_bounds(lo_ns, hi_ns)
        run = IntervalRun(
            start_ns=lo_ns,
            end_ns=hi_ns,
            task_id=int(task_id),
            pipe_id=pid,
            scope_id=sid,
            owner=int(owner),
            delivered=float(delivered),
        )
        if len(self._head) < self.max_head_tail:
            self._head.append(run)
        else:
            if len(self._tail) >= self.max_head_tail:
                self._tail.pop(0)
            self._tail.append(run)

    # Point 4: allocation grant / release
    def on_allocation(
        self,
        pool: str,
        owner: int,
        slot: int,
        generation: int,
        size_bytes: int,
        t_ms: float,
        action: Literal["grant", "release"],
        *,
        alias_of: str | None = None,
    ) -> None:
        t_ns = _ms_to_ns(t_ms)
        key = (str(pool), int(owner), int(slot), int(generation))
        pool_key = (int(owner), str(pool))
        if action == "grant":
            self._live_allocs[key] = BufferResidency(
                pool=str(pool),
                owner=int(owner),
                slot=int(slot),
                generation=int(generation),
                size_bytes=int(size_bytes),
                grant_ns=t_ns,
                release_ns=None,
                alias_of=alias_of,
            )
            if alias_of is None:
                cur = self._live_pool_bytes.get(pool_key, 0) + int(size_bytes)
                self._live_pool_bytes[pool_key] = cur
                if cur > self._peak_pool_bytes.get(pool_key, 0):
                    self._peak_pool_bytes[pool_key] = cur
        elif action == "release":
            live = self._live_allocs.pop(key, None)
            if live is not None:
                self._closed_allocs.append(BufferResidency(
                    pool=live.pool,
                    owner=live.owner,
                    slot=live.slot,
                    generation=live.generation,
                    size_bytes=live.size_bytes,
                    grant_ns=live.grant_ns,
                    release_ns=t_ns,
                    alias_of=live.alias_of,
                ))
                if live.alias_of is None:
                    cur = max(0, self._live_pool_bytes.get(pool_key, 0) - live.size_bytes)
                    self._live_pool_bytes[pool_key] = cur

    # Point 5: proved periodic skip
    def on_period(
        self,
        anchor_ms: float,
        period_ms: float,
        repeats: int,
        template_id: str,
        state_digest: str,
        *,
        service_pattern: Sequence[tuple[str, float, float]] = (),
        handoff_pattern: Sequence[tuple[str, float, float]] = (),
        scope_id: int | None = None,
    ) -> None:
        """Translate exact periodic service and handoff patterns without rounding drift."""
        if repeats < 1 or period_ms <= 0.0:
            raise ModelingError("wave_recorder_periodic_skip_invalid")
        sid = self.scope_id if scope_id is None else int(scope_id)
        self._steady.append(PeriodicRun(
            anchor_ms=float(anchor_ms),
            period_ms=float(period_ms),
            repeats=int(repeats),
            template_id=str(template_id),
            state_digest=str(state_digest),
        ))
        self._note_bounds(
            _ms_to_ns(anchor_ms),
            _ms_to_ns(anchor_ms + float(repeats) * period_ms),
        )
        for rep in range(repeats):
            base_ms = anchor_ms + float(rep) * period_ms
            for pipe_name, rel_lo_ms, rel_hi_ms in service_pattern:
                lo_ns = _ms_to_ns(base_ms + rel_lo_ms)
                hi_ns = _ms_to_ns(base_ms + rel_hi_ms)
                if hi_ns > lo_ns:
                    pid = self.register_pipe(str(pipe_name))
                    key = (sid, pid)
                    self._service_by_pipe.setdefault(key, []).append((lo_ns, hi_ns))
                    self._owner_time_by_pipe[key] = (
                        self._owner_time_by_pipe.get(key, 0) + (hi_ns - lo_ns)
                    )
            for pipe_name, rel_lo_ms, rel_hi_ms in handoff_pattern:
                lo_ns = _ms_to_ns(base_ms + rel_lo_ms)
                hi_ns = _ms_to_ns(base_ms + rel_hi_ms)
                if hi_ns > lo_ns:
                    pid = self.register_pipe(str(pipe_name))
                    key = (sid, pid)
                    self._handoff_by_pipe.setdefault(key, []).append((lo_ns, hi_ns))

    def finish(
        self,
        spec: Any,
        occurrence_key: Any,
        *,
        window_start_ms: float | None = None,
        window_end_ms: float | None = None,
        semantic_group_id: str = "",
        timing_scope: Literal["graph_occurrence", "isolated_occurrence"] = "graph_occurrence",
        submitted_kernel_count: int | None = None,
        uncovered_stars_ns: int = 0,
    ) -> FusedPipelineDescriptor:
        """Compute exact 4-bucket idle rows and return ``FusedPipelineDescriptor``."""
        start_ns = (
            _ms_to_ns(window_start_ms)
            if window_start_ms is not None else (self._min_ns or 0)
        )
        end_ns = (
            _ms_to_ns(window_end_ms)
            if window_end_ms is not None else (self._max_ns if self._max_ns is not None else start_ns)
        )
        if end_ns < start_ns:
            raise ModelingError(f"wave_recorder_window_reversed:{start_ns}>{end_ns}")
        span_ns = end_ns - start_ns
        window_iv = ((start_ns, end_ns),) if span_ns > 0 else ()

        all_keys = sorted(
            {(self.scope_id, pid) for pid in self._pipe_names}
            | set(self._service_by_pipe)
            | set(self._handoff_by_pipe)
        )
        pipe_rows: list[PipeIdleRow] = []
        cube_intervals: list[tuple[int, int]] = []
        vec_intervals: list[tuple[int, int]] = []
        limiting: list[ResourceRowRef] = []
        max_busy = 0

        for sid, pid in all_keys:
            pname = self._pipe_names.get(pid, f"pipe_{pid}")
            busy_set = _clamp_intervals(self._service_by_pipe.get((sid, pid), ()), start_ns, end_ns)
            h0_set = _clamp_intervals(self._handoff_by_pipe.get((sid, pid), ()), start_ns, end_ns)
            handoff_set = _subtract_intervals(h0_set, busy_set)
            covered = _union_intervals((*busy_set, *handoff_set))
            idle_j = _subtract_intervals(window_iv, covered)

            busy_ns = _measure(busy_set)
            handoff_ns = _measure(handoff_set)
            if busy_set:
                first_a = busy_set[0][0]
                last_b = busy_set[-1][1]
                fill_ns = _measure(_clamp_intervals(idle_j, start_ns, first_a))
                steady_bubble_ns = _measure(_clamp_intervals(idle_j, first_a, last_b))
                drain_ns = _measure(_clamp_intervals(idle_j, last_b, end_ns))
                inactive = False
            else:
                fill_ns = _measure(_subtract_intervals(window_iv, handoff_set))
                steady_bubble_ns = 0
                drain_ns = 0
                inactive = True

            pipe_rows.append(PipeIdleRow(
                scope_id=sid,
                pipe_id=pid,
                span_ns=span_ns,
                busy_ns=busy_ns,
                fill_ns=fill_ns,
                steady_bubble_ns=steady_bubble_ns,
                drain_ns=drain_ns,
                handoff_ns=handoff_ns,
                inactive_in_window=inactive,
                pipe_name=pname,
                owner_time_ns=self._owner_time_by_pipe.get((sid, pid), 0),
            ))
            role = _pipe_base_role(pname)
            if role in CUBE_PIPE_ROLES:
                cube_intervals.extend(busy_set)
            elif role in VECTOR_PIPE_ROLES:
                vec_intervals.extend(busy_set)
            if busy_ns > max_busy:
                max_busy = busy_ns

        for row in pipe_rows:
            if max_busy > 0 and row.busy_ns == max_busy:
                limiting.append(ResourceRowRef(row.pipe_id, row.pipe_name, row.scope_id))

        cube_union = _union_intervals(cube_intervals)
        vec_union = _union_intervals(vec_intervals)
        overlap_ns = _measure(_intersect_intervals(cube_union, vec_union))

        timeline = TileWaveTimeline(
            start_ns=start_ns,
            end_ns=end_ns,
            head=tuple(self._head),
            steady=tuple(self._steady),
            tail=tuple(self._tail),
            pipe_rows=tuple(pipe_rows),
            cube_vector_overlap_ns=overlap_ns,
        )
        peaks = tuple(
            OwnerPoolPeak(owner=owner, pool=pool, peak_bytes=peak)
            for (owner, pool), peak in sorted(self._peak_pool_bytes.items())
        )
        diagnosis = BottleneckDiagnosis(
            limiting_rows=tuple(limiting),
            exposed_waits=tuple(self._wait_witnesses),
            peak_allocations=peaks,
            critical_events=tuple(self._critical_events),
            uncovered_stars_ns=int(uncovered_stars_ns),
        )
        bindings = tuple(
            MemberBinding(member_id=mid, task_ids=tuple(tids))
            for mid, tids in sorted(self._member_tasks.items())
        )
        buffers = tuple((*self._closed_allocs, *self._live_allocs.values()))
        kernel_cnt = (
            len(self._submitted_tasks)
            if submitted_kernel_count is None else int(submitted_kernel_count)
        )
        return FusedPipelineDescriptor(
            spec=spec,
            occurrence_key=occurrence_key,
            semantic_group_id=semantic_group_id or getattr(spec, "spec_id", ""),
            member_bindings=bindings,
            timing_scope=timing_scope,
            submitted_kernel_count=kernel_cnt,
            buffers=buffers,
            timeline=timeline,
            diagnosis=diagnosis,
        )


__all__ = [
    "BottleneckDiagnosis",
    "BufferResidency",
    "FusedPipelineDescriptor",
    "IntervalRun",
    "MemberBinding",
    "NS_PER_MS",
    "OwnerPoolPeak",
    "PeriodicRun",
    "PipeIdleRow",
    "RecordedPipelineSpec",
    "ResourceRowRef",
    "SCHEMA_VERSION",
    "TileWaveTimeline",
    "WaitWitness",
    "WaveRecorder",
]
