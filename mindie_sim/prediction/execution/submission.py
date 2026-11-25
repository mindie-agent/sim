"""Host submission policy, device STARS descriptor queue, and Eager D2H barrier.

[Role]
  Model host-side submission policies (``SubmissionPolicy``,
  ``ScheduledIteration``), device-boundary ``STARS`` FIFO descriptor issue
  queues (``StarsDescriptorQueue``, ``compute_uncovered_stars_ns``), and
  Prefill Eager ``TokenDispatcherWithAll2AllV`` ``D2H`` split-size barriers
  (``EagerSplitBarrier``, ``SplitBarrierSpec``) on the shared ``Engine``.

[Invariants]
  * Each real device kernel carries a unique descriptor identity
    ``(replay, die, stream, kernel_occurrence, descriptor_ordinal)``; shape
    views, graph aliases, and internal stages never consume STARS descriptors.
  * STARS lanes are genuine FIFO Engine resources gated by finite credit pools
    (``queue_depth``) even when ``issue_interval_us == 0``; ``descriptor_ready``
    permits descriptor prefetch while enforcing prior same-stream kernel
    execution order on device.
  * Completed STARS event handles are retired into merged integer-ns interval
    lists on completion so memory stays bounded ``O(queue_depth)`` over long runs.
  * Prefill Eager ``TokenDispatcherWithAll2AllV`` couples each D2H split copy
    to its physical die port + shared ``(node, root_complex_id, "d2h")`` root
    (plus a per-transfer pageable staging cap when ``pinned=False``), chains
    ``repeat_interleave`` after ``allgather_counts`` when ``num_local_experts > 1``,
    and inserts an explicit ``torch.npu.synchronize()`` fence when
    ``num_local_experts <= 1`` matching upstream ``token_dispatcher.py:647-682``.

[Timeouts & Locks]
  Bounded in-memory interval merges and Engine task submission; no I/O or locks.

[Side-Effect Policy]
  Submits tasks and pools only to the caller-supplied Engine.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.device_input import StarsSpec
from ..engine import Allocation, Event, Resource
from .split_barrier import (
    DEFAULT_SPLIT_DTYPE_BYTES,
    EagerSplitBarrier,
    SplitBarrierResult,
    SplitBarrierSpec,
)

MS_PER_S = 1000.0
NS_PER_MS = 1_000_000.0
ZERO_KERNEL_VIEW_FAMILIES = frozenset({
    "view", "reshape", "alias", "slice_view", "transpose_view",
    "permute_view", "no_op", "noop", "metadata_only",
})
_ZERO_WORK_FIELDS = (
    "flops", "hbm_bytes", "mte2_bytes", "mte3_bytes", "vector_ops", "link_bytes",
)
_ZERO_META_WORK_KEYS = ("mte1_bytes", "fixpipe_bytes")


@dataclass(frozen=True)
class SubmissionPolicy:
    graph_replay_ms: float | None = None
    scheduler_ms: float = 0
    completion_ms: float = 0
    kernel_launch_ms: float | None = None
    piecewise_sync_ms: float = 0.0

    def launch_ms(self, dies, *, captured=False):
        """Software submission overrides the explicit legacy device parameter."""
        eager = (self.kernel_launch_ms if self.kernel_launch_ms is not None
                 else max(d.launch_us for d in dies) / MS_PER_S)
        return (self.graph_replay_ms if captured and self.graph_replay_ms is not None
                else eager)


@dataclass
class ScheduledIteration:
    replica: int
    iteration: int
    completion: Event
    start_ms: float
    mode: str
    valid_tokens: int
    padded_tokens: int
    graph_bucket: int | None
    replay_scope: str
    operators: int
    commits: Mapping[str, Event]
    device_complete: Event
    host_results: Mapping[str, Event]
    client_deliveries: Mapping[str, Event]
    implementation_selections: tuple[dict, ...] = ()
    host_submits: tuple[tuple[int, int, Event], ...] = ()
    declared_commits: Mapping[str, Event] = field(default_factory=dict)
    fallback_commits: Mapping[str, Event] = field(default_factory=dict)
    offload_release: Event | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": self.replica, "iteration": self.iteration,
            "start_ms": self.start_ms, "end_ms": self.completion.time,
            "duration_ms": self.completion.time - self.start_ms,
            "mode": self.mode, "valid_tokens": self.valid_tokens,
            "padded_tokens": self.padded_tokens, "graph_bucket": self.graph_bucket,
            "replay_scope": self.replay_scope, "operators": self.operators,
            "device_end_ms": self.device_complete.time,
            "host_result_ready_ms": max((event.time for event in self.host_results.values()), default=None),
            "implementation_selections": list(self.implementation_selections),
            "host_submits": [
                {"replica": replica, "stage": stage, "start_ms": event.time}
                for replica, stage, event in self.host_submits],
            "declared_commit_ms": {
                request: event.time for request, event in self.declared_commits.items()},
            "fallback_commit_ms": {
                request: event.time for request, event in self.fallback_commits.items()},
            "offload_release_ms": (
                None if self.offload_release is None else self.offload_release.time),
        }


def is_zero_kernel_view(op) -> bool:
    """True for zero-kernel views, aliases, and internal non-kernel stages."""
    family = str(getattr(op, "family", "") or "").strip().lower()
    if family in ZERO_KERNEL_VIEW_FAMILIES:
        return True
    meta = dict(getattr(op, "metadata", ()) or ())
    if meta.get("zero_kernel_view") or meta.get("is_view") or meta.get("graph_alias"):
        return True
    if "kernel_count" in meta and int(meta["kernel_count"]) == 0:
        return True
    if meta.get("internal_stage_only"):
        return True
    if family == "tensor_view" and not meta.get("materialized"):
        matrix_flops = getattr(op, "matrix_flops", ()) or ()
        has_matrix = any(float(amount or 0.0) != 0.0 for _, amount in matrix_flops)
        has_meta_work = any(float(meta.get(key, 0.0) or 0.0) != 0.0 for key in _ZERO_META_WORK_KEYS)
        res_work = meta.get("resource_work")
        has_res_work = isinstance(res_work, Mapping) and any(
            float(val or 0.0) != 0.0 for val in res_work.values()
        )
        if not has_matrix and not has_meta_work and not has_res_work and all(
            float(getattr(op, key, 0.0) or 0.0) == 0.0 for key in _ZERO_WORK_FIELDS
        ):
            return True
    return False


def _merge_ns_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    cleaned = sorted((int(a), int(b)) for a, b in intervals if int(b) > int(a))
    if not cleaned:
        return []
    merged = [cleaned[0]]
    for start, end in cleaned[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def _to_ns_intervals(intervals_ms: Iterable[tuple[float, float]]) -> list[tuple[int, int]]:
    raw = []
    for start_ms, end_ms in intervals_ms:
        s_ns = int(round(float(start_ms) * NS_PER_MS))
        e_ns = int(round(float(end_ms) * NS_PER_MS))
        if e_ns > s_ns:
            raw.append((s_ns, e_ns))
    return _merge_ns_intervals(raw)


def interval_union_ns(intervals_ms: Iterable[tuple[float, float]]) -> int:
    """Total wall-clock nanoseconds covered by the union of intervals."""
    return sum(end - start for start, end in _to_ns_intervals(intervals_ms))


def _diff_ns_intervals(
    desc: list[tuple[int, int]], comp: list[tuple[int, int]]
) -> int:
    if not desc:
        return 0
    if not comp:
        return sum(end - start for start, end in desc)
    uncovered = 0
    c_idx = 0
    c_len = len(comp)
    for d_start, d_end in desc:
        cursor = d_start
        while c_idx < c_len and comp[c_idx][1] <= cursor:
            c_idx += 1
        scan = c_idx
        while scan < c_len and comp[scan][0] < d_end:
            c_start, c_end = comp[scan]
            if c_start > cursor:
                uncovered += c_start - cursor
            if c_end > cursor:
                cursor = c_end
            if cursor >= d_end:
                break
            scan += 1
        if cursor < d_end:
            uncovered += d_end - cursor
    return uncovered


def compute_uncovered_stars_ns(
    descriptor_intervals_ms: Iterable[tuple[float, float]],
    compute_intervals_ms: Iterable[tuple[float, float]],
) -> int:
    """Measure ``D \\ C`` in integer ns across all ``N >= 1`` streams."""
    return _diff_ns_intervals(
        _to_ns_intervals(descriptor_intervals_ms),
        _to_ns_intervals(compute_intervals_ms),
    )


@dataclass(frozen=True, slots=True)
class StarsDescriptorId:
    """Unique physical descriptor key inside one device replay."""

    replay: int
    die: int
    stream: str
    kernel_occurrence: int
    descriptor_ordinal: int


@dataclass
class StarsDescriptorRecord:
    """Event handles (cleared on retirement) and frozen timing for one descriptor."""

    descriptor_id: StarsDescriptorId
    lane_index: int
    issue_duration_ms: float
    issue_start_event: Event | None
    issue_done_event: Event | None
    kernel_start_event: Event | None = None
    kernel_done_event: Event | None = None
    retired_issue_interval_ms: tuple[float, float] | None = None
    retired_compute_interval_ms: tuple[float, float] | None = None

    @property
    def issue_interval_ms(self) -> tuple[float, float] | None:
        if self.retired_issue_interval_ms is not None:
            return self.retired_issue_interval_ms
        if self.issue_done_event is None or self.issue_done_event.time is None:
            return None
        end_ms = float(self.issue_done_event.time)
        start_ms = max(0.0, end_ms - self.issue_duration_ms)
        if self.issue_start_event is not None and self.issue_start_event.time is not None:
            start_ms = max(start_ms, float(self.issue_start_event.time))
        return (start_ms, end_ms)

    @property
    def compute_interval_ms(self) -> tuple[float, float] | None:
        if self.retired_compute_interval_ms is not None:
            return self.retired_compute_interval_ms
        if self.kernel_done_event is None or self.kernel_done_event.time is None:
            return None
        end_ms = float(self.kernel_done_event.time)
        if self.kernel_start_event is not None and self.kernel_start_event.time is not None:
            start_ms = float(self.kernel_start_event.time)
        elif self.issue_done_event is not None and self.issue_done_event.time is not None:
            start_ms = float(self.issue_done_event.time)
        else:
            start_ms = 0.0
        return (start_ms, end_ms)

    def retire(self) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Snapshot numeric intervals and release all live Engine Event handles."""
        issue_iv = self.issue_interval_ms
        comp_iv = self.compute_interval_ms
        self.retired_issue_interval_ms = issue_iv
        self.retired_compute_interval_ms = comp_iv
        self.issue_start_event = None
        self.issue_done_event = None
        self.kernel_start_event = None
        self.kernel_done_event = None
        return issue_iv, comp_iv


class StarsDescriptorQueue:
    """Device-boundary STARS FIFO descriptor issue queue on the shared Engine."""

    def __init__(
        self,
        spec: StarsSpec,
        *,
        die_id: int = 0,
        namespace: str = "",
        controller_id: str | None = None,
    ):
        self.spec = spec
        self.die_id = int(die_id)
        self.namespace = str(namespace)
        self.scope_key = (
            str(controller_id)
            if controller_id
            else (f"die{self.die_id}" if spec.queue_scope == "die" else str(spec.queue_scope))
        )
        self.lane_names = tuple(
            f"{self.namespace}stars:{self.scope_key}:lane{idx}"
            for idx in range(int(spec.lanes))
        )
        self.credit_pool = f"{self.namespace}stars:{self.scope_key}:credits"
        self._ordinal = 0
        self._descriptor_count = 0
        self._replay_first_charged: set[tuple[int, int]] = set()
        self._replay_active_counts: dict[tuple[int, int], int] = {}
        self._latest_replay_by_die: dict[int, int] = {}
        self._stream_last_kernel_done: dict[tuple[int, str], Event] = {}
        self._active_records: dict[int, StarsDescriptorRecord] = {}
        self._retired_desc_ns: list[tuple[int, int]] = []
        self._retired_comp_ns: list[tuple[int, int]] = []
        self._folded_desc_ns: int = 0
        self._folded_comp_ns: int = 0
        self._folded_uncovered_ns: int = 0

    @property
    def active(self) -> bool:
        return True

    @property
    def descriptor_count(self) -> int:
        return self._descriptor_count

    @property
    def records(self) -> list[StarsDescriptorRecord]:
        """Currently active (unretired) in-flight descriptor records."""
        return list(self._active_records.values())

    def ensure_engine_resources(self, engine) -> None:
        """Register FIFO STARS lane resources and credit pool on ``engine``."""
        new_resources = [
            Resource(name, 1.0, "ms", "fifo")
            for name in self.lane_names
            if name not in engine.resources
        ]
        new_pools = (
            {self.credit_pool: float(self.spec.queue_depth)}
            if self.credit_pool not in engine.pools
            else {}
        )
        if new_resources or new_pools:
            engine.extend(new_resources, pools=new_pools)

    def issue_descriptor(
        self,
        engine,
        *,
        replay: int,
        stream: str,
        kernel_occurrence: int,
        descriptor_ready: Event,
        owner: str = "",
        die: int | None = None,
    ) -> tuple[StarsDescriptorRecord, Allocation]:
        """Submit one STARS descriptor issue task obeying lane FIFO + credit depth."""
        self.ensure_engine_resources(engine)
        die_num = self.die_id if die is None else int(die)
        ordinal = self._ordinal
        self._ordinal += 1
        self._descriptor_count += 1
        desc_id = StarsDescriptorId(
            replay=int(replay),
            die=die_num,
            stream=str(stream or "main"),
            kernel_occurrence=int(kernel_occurrence),
            descriptor_ordinal=ordinal,
        )
        lane_index = ordinal % len(self.lane_names)
        lane_name = self.lane_names[lane_index]
        duration_ms = float(self.spec.issue_interval_us) / MS_PER_S
        replay_int = int(replay)
        replay_key = (replay_int, die_num)
        prev_latest = self._latest_replay_by_die.get(die_num, -1)
        if replay_int > prev_latest:
            self._latest_replay_by_die[die_num] = replay_int
            for old_key in [
                k for k in self._replay_first_charged
                if k[1] == die_num and k[0] < replay_int
                and k not in self._replay_active_counts
            ]:
                self._replay_first_charged.discard(old_key)
        # Evict retired replay keys for this die when a new replay begins (O(active) state)
        if replay_key not in self._replay_first_charged:
            stale = [
                k for k in self._replay_first_charged
                if k[1] == die_num and self._replay_active_counts.get(k, 0) <= 0
            ]
            for k in stale:
                self._replay_first_charged.discard(k)
        if (replay_key not in self._replay_first_charged
                and float(self.spec.first_issue_us) > 0.0):
            duration_ms += float(self.spec.first_issue_us) / MS_PER_S
            self._replay_first_charged.add(replay_key)
        self._replay_active_counts[replay_key] = (
            self._replay_active_counts.get(replay_key, 0) + 1
        )

        after_events = [descriptor_ready]
        if self.spec.release_policy == "stream_predecessor_done":
            prev_done = self._stream_last_kernel_done.get((die_num, desc_id.stream))
            if prev_done is not None:
                after_events.append(prev_done)

        prefix = (
            f"{self.namespace}stars:{self.scope_key}:r{replay}:d{die_num}:"
            f"{desc_id.stream}:occ{kernel_occurrence}:ord{ordinal}"
        )
        credit_alloc = Allocation(
            f"{prefix}:credit", self.credit_pool, 1.0, (0,)
        )
        issue_done = engine.submit(
            f"{prefix}:issue",
            {lane_name: duration_ms},
            after=tuple(after_events),
            acquire=(credit_alloc,),
            owner=owner,
        )
        issue_start = engine.start_boundary(issue_done, f"{prefix}:issue_start")
        record = StarsDescriptorRecord(
            descriptor_id=desc_id,
            lane_index=lane_index,
            issue_duration_ms=duration_ms,
            issue_start_event=issue_start,
            issue_done_event=issue_done,
        )
        self._active_records[ordinal] = record
        return record, credit_alloc

    def _fold_retired_up_to(self, horizon_ns: int | None) -> None:
        """Fold retired intervals ending <= ``horizon_ns`` into O(1) counters."""
        if not self._retired_desc_ns and not self._retired_comp_ns:
            return
        if horizon_ns is None:
            self._folded_desc_ns += sum(e - s for s, e in self._retired_desc_ns)
            self._folded_comp_ns += sum(e - s for s, e in self._retired_comp_ns)
            self._folded_uncovered_ns += _diff_ns_intervals(
                self._retired_desc_ns, self._retired_comp_ns
            )
            self._retired_desc_ns = []
            self._retired_comp_ns = []
            return
        fold_desc: list[tuple[int, int]] = []
        keep_desc: list[tuple[int, int]] = []
        for s_ns, e_ns in self._retired_desc_ns:
            if e_ns <= horizon_ns:
                fold_desc.append((s_ns, e_ns))
            elif s_ns >= horizon_ns:
                keep_desc.append((s_ns, e_ns))
            else:
                fold_desc.append((s_ns, horizon_ns))
                keep_desc.append((horizon_ns, e_ns))
        fold_comp: list[tuple[int, int]] = []
        keep_comp: list[tuple[int, int]] = []
        for s_ns, e_ns in self._retired_comp_ns:
            if e_ns <= horizon_ns:
                fold_comp.append((s_ns, e_ns))
            elif s_ns >= horizon_ns:
                keep_comp.append((s_ns, e_ns))
            else:
                fold_comp.append((s_ns, horizon_ns))
                keep_comp.append((horizon_ns, e_ns))
        if not fold_desc and not fold_comp:
            return
        self._folded_desc_ns += sum(e - s for s, e in fold_desc)
        self._folded_comp_ns += sum(e - s for s, e in fold_comp)
        self._folded_uncovered_ns += _diff_ns_intervals(fold_desc, fold_comp)
        self._retired_desc_ns = keep_desc
        self._retired_comp_ns = keep_comp

    def _retire_record(self, ordinal: int) -> None:
        rec = self._active_records.pop(ordinal, None)
        if rec is None:
            return
        rkey = (rec.descriptor_id.replay, rec.descriptor_id.die)
        rem = self._replay_active_counts.get(rkey, 0) - 1
        if rem <= 0:
            self._replay_active_counts.pop(rkey, None)
            if rec.descriptor_id.replay < self._latest_replay_by_die.get(rec.descriptor_id.die, -1):
                self._replay_first_charged.discard(rkey)
        else:
            self._replay_active_counts[rkey] = rem
        issue_iv, comp_iv = rec.retire()
        if issue_iv is not None:
            s_ns = int(round(issue_iv[0] * NS_PER_MS))
            e_ns = int(round(issue_iv[1] * NS_PER_MS))
            if e_ns > s_ns:
                self._retired_desc_ns = _merge_ns_intervals(
                    (*self._retired_desc_ns, (s_ns, e_ns))
                )
        if comp_iv is not None:
            s_ns = int(round(comp_iv[0] * NS_PER_MS))
            e_ns = int(round(comp_iv[1] * NS_PER_MS))
            if e_ns > s_ns:
                self._retired_comp_ns = _merge_ns_intervals(
                    (*self._retired_comp_ns, (s_ns, e_ns))
                )
        if not self._active_records:
            self._fold_retired_up_to(None)
        else:
            starts: list[int] = []
            all_started = True
            for active in self._active_records.values():
                s_ev = active.issue_start_event
                if s_ev is None or s_ev.time is None:
                    all_started = False
                    break
                starts.append(int(round(s_ev.time * NS_PER_MS)))
            if all_started and starts:
                self._fold_retired_up_to(min(starts))

    def bind_kernel_execution(
        self,
        engine,
        record: StarsDescriptorRecord,
        credit_alloc: Allocation,
        *,
        kernel_start: Event,
        kernel_done: Event,
        owner: str = "",
    ) -> Event:
        """Bind physical kernel completion to credit release & handle retirement."""
        record.kernel_start_event = kernel_start
        record.kernel_done_event = kernel_done
        desc_id = record.descriptor_id
        self._stream_last_kernel_done[(desc_id.die, desc_id.stream)] = kernel_done
        ordinal = desc_id.descriptor_ordinal
        release_prefix = (
            f"{self.namespace}stars:{self.scope_key}:r{desc_id.replay}:"
            f"d{desc_id.die}:ord{ordinal}"
        )
        return engine.submit(
            f"{release_prefix}:credit_free",
            after=(kernel_done,),
            free=(credit_alloc,),
            owner=owner,
            on_complete=lambda _ev, ord_id=ordinal: self._retire_record(ord_id),
            native_hook=True,
        )

    def submit_kernel(
        self,
        engine,
        *,
        replay: int,
        stream: str,
        kernel_occurrence: int,
        descriptor_ready: Event,
        work: Mapping[str, float],
        data_deps: Iterable[Event] = (),
        owner: str = "",
        die: int | None = None,
        op=None,
    ) -> tuple[Event, StarsDescriptorRecord | None]:
        """Issue descriptor (skipping zero-kernel views) and enforce stream order."""
        if op is not None and is_zero_kernel_view(op):
            deps = (descriptor_ready, *tuple(data_deps))
            view_done = engine.submit(
                f"{self.namespace}view:r{replay}:{stream}:occ{kernel_occurrence}",
                after=deps,
                owner=owner,
            )
            return view_done, None
        die_num = self.die_id if die is None else int(die)
        stream_name = str(stream or "main")
        prev_stream_done = self._stream_last_kernel_done.get((die_num, stream_name))
        record, credit_alloc = self.issue_descriptor(
            engine,
            replay=replay,
            stream=stream_name,
            kernel_occurrence=kernel_occurrence,
            descriptor_ready=descriptor_ready,
            owner=owner,
            die=die_num,
        )
        # Descriptor prefetch overlaps prior compute under descriptor_ready,
        # while device kernel execution strictly preserves FIFO stream order.
        kernel_deps = [record.issue_done_event]
        if prev_stream_done is not None:
            kernel_deps.append(prev_stream_done)
        kernel_deps.extend(data_deps)
        prefix = (
            f"{self.namespace}stars_kernel:{self.scope_key}:r{replay}:d{die_num}:"
            f"{stream_name}:occ{kernel_occurrence}:ord{record.descriptor_id.descriptor_ordinal}"
        )
        kernel_done = engine.submit(
            prefix,
            dict(work),
            after=tuple(kernel_deps),
            owner=owner,
            coupled=len(work) > 1,
        )
        kernel_start = engine.start_boundary(kernel_done, f"{prefix}:start")
        self.bind_kernel_execution(
            engine, record, credit_alloc,
            kernel_start=kernel_start, kernel_done=kernel_done, owner=owner,
        )
        return kernel_done, record

    def summary(self) -> dict[str, Any]:
        """Compute exact integer-ns descriptor, compute union, and uncovered time."""
        active_desc = _to_ns_intervals(
            iv for rec in self._active_records.values()
            if (iv := rec.issue_interval_ms) is not None
        )
        active_comp = _to_ns_intervals(
            iv for rec in self._active_records.values()
            if (iv := rec.compute_interval_ms) is not None
        )
        desc_ns = _merge_ns_intervals((*self._retired_desc_ns, *active_desc))
        comp_ns = _merge_ns_intervals((*self._retired_comp_ns, *active_comp))
        return {
            "scope": self.scope_key,
            "lanes": int(self.spec.lanes),
            "queue_depth": int(self.spec.queue_depth),
            "release_policy": self.spec.release_policy,
            "descriptor_count": self._descriptor_count,
            "active_records": len(self._active_records),
            "descriptor_union_ns": self._folded_desc_ns + sum(e - s for s, e in desc_ns),
            "compute_union_ns": self._folded_comp_ns + sum(e - s for s, e in comp_ns),
            "uncovered_stars_ns": self._folded_uncovered_ns + _diff_ns_intervals(desc_ns, comp_ns),
        }
