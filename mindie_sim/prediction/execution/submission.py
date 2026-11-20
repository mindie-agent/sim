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
    (``queue_depth``): ``s_i = max(t_ready, t_lane_free, t_credit, t_release)``,
    ``q_i = s_i + tau_issue``, ``t_kernel_start >= max(q_i, t_data_deps, t_owner)``.
  * ``include_submission=False`` removes host wrapper latency while retaining
    device-boundary STARS descriptor issue when configured.
  * ``uncovered_stars_ns`` is computed from exact integer-ns interval unions
    across all ``N >= 1`` streams; overlapped descriptors contribute ``0`` ns,
    with zero step-end ``N * gap`` floors.
  * Prefill Eager ``TokenDispatcherWithAll2AllV`` submits ``input_splits`` and
    ``output_splits`` D2H copies over coupled per-die + ``(node, root_complex_id,
    "d2h")`` resources; ``num_local_experts <= 1`` inserts an explicit device
    synchronization fence matching upstream ``token_dispatcher.py:647-682``.

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
from ..collectives.ports import (
    GBPS_TO_BYTES_PER_MS,
    RootComplexKey,
    host_transfer_demand,
    root_complex_resources_for_dies,
)
from ..engine import Allocation, Event, Resource

MS_PER_S = 1000.0
NS_PER_MS = 1_000_000.0
DEFAULT_SPLIT_DTYPE_BYTES = 8
ZERO_KERNEL_VIEW_FAMILIES = frozenset({
    "view", "reshape", "alias", "slice_view", "transpose_view",
    "permute_view", "no_op", "noop", "metadata_only",
})


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
                {"replica": replica, "stage": stage,
                 "start_ms": event.time}
                for replica, stage, event in self.host_submits],
            "declared_commit_ms": {
                request: event.time
                for request, event in self.declared_commits.items()},
            "fallback_commit_ms": {
                request: event.time
                for request, event in self.fallback_commits.items()},
            "offload_release_ms": (
                None if self.offload_release is None
                else self.offload_release.time),
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


def compute_uncovered_stars_ns(
    descriptor_intervals_ms: Iterable[tuple[float, float]],
    compute_intervals_ms: Iterable[tuple[float, float]],
) -> int:
    """Measure ``D \\ C`` in integer ns across all ``N >= 1`` streams.

    Any descriptor issue interval overlapped by active device compute on any
    stream contributes ``0`` ns to ``uncovered_stars_ns``.
    """
    desc = _to_ns_intervals(descriptor_intervals_ms)
    if not desc:
        return 0
    comp = _to_ns_intervals(compute_intervals_ms)
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
    """Event handles for one issued STARS descriptor and its kernel body."""

    descriptor_id: StarsDescriptorId
    lane_index: int
    issue_duration_ms: float
    issue_start_event: Event
    issue_done_event: Event
    kernel_start_event: Event | None = None
    kernel_done_event: Event | None = None

    @property
    def issue_interval_ms(self) -> tuple[float, float] | None:
        if self.issue_done_event.time is None:
            return None
        end_ms = float(self.issue_done_event.time)
        # Single-server non-preemptive FIFO lane: active service occupies
        # [end_ms - issue_duration_ms, end_ms] even when queued earlier.
        start_ms = max(0.0, end_ms - self.issue_duration_ms)
        if self.issue_start_event.time is not None:
            start_ms = max(start_ms, float(self.issue_start_event.time))
        return (start_ms, end_ms)

    @property
    def compute_interval_ms(self) -> tuple[float, float] | None:
        if self.kernel_done_event is None or self.kernel_done_event.time is None:
            return None
        end_ms = float(self.kernel_done_event.time)
        start_ms = (
            float(self.kernel_start_event.time)
            if self.kernel_start_event is not None and self.kernel_start_event.time is not None
            else (float(self.issue_done_event.time)
                  if self.issue_done_event.time is not None else 0.0)
        )
        return (start_ms, end_ms)


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
        self._replay_first_charged: set[int] = set()
        self._stream_last_kernel_done: dict[tuple[int, str], Event] = {}
        self.records: list[StarsDescriptorRecord] = []

    @property
    def active(self) -> bool:
        return bool(self.spec.issue_interval_us > 0.0 or self.spec.first_issue_us > 0.0)

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
        if (int(replay) not in self._replay_first_charged
                and float(self.spec.first_issue_us) > 0.0):
            duration_ms += float(self.spec.first_issue_us) / MS_PER_S
            self._replay_first_charged.add(int(replay))

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
        self.records.append(record)
        return record, credit_alloc

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
        """Bind kernel completion to credit release and stream-predecessor tracking."""
        record.kernel_start_event = kernel_start
        record.kernel_done_event = kernel_done
        desc_id = record.descriptor_id
        self._stream_last_kernel_done[(desc_id.die, desc_id.stream)] = kernel_done
        release_prefix = (
            f"{self.namespace}stars:{self.scope_key}:r{desc_id.replay}:"
            f"ord{desc_id.descriptor_ordinal}"
        )
        return engine.submit(
            f"{release_prefix}:credit_free",
            after=(kernel_done,),
            free=(credit_alloc,),
            owner=owner,
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
        """Issue descriptor (skipping zero-kernel views) and submit kernel work."""
        if op is not None and is_zero_kernel_view(op):
            deps = (descriptor_ready, *tuple(data_deps))
            view_done = engine.submit(
                f"{self.namespace}view:r{replay}:{stream}:occ{kernel_occurrence}",
                after=deps,
                owner=owner,
            )
            return view_done, None
        record, credit_alloc = self.issue_descriptor(
            engine,
            replay=replay,
            stream=stream,
            kernel_occurrence=kernel_occurrence,
            descriptor_ready=descriptor_ready,
            owner=owner,
            die=die,
        )
        kernel_after = (record.issue_done_event, *tuple(data_deps))
        prefix = (
            f"{self.namespace}stars_kernel:{self.scope_key}:r{replay}:"
            f"{stream}:occ{kernel_occurrence}:ord{record.descriptor_id.descriptor_ordinal}"
        )
        kernel_done = engine.submit(
            prefix,
            dict(work),
            after=kernel_after,
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
        desc_intervals = [
            iv for rec in self.records
            if (iv := rec.issue_interval_ms) is not None
        ]
        comp_intervals = [
            iv for rec in self.records
            if (iv := rec.compute_interval_ms) is not None
        ]
        uncovered_ns = compute_uncovered_stars_ns(desc_intervals, comp_intervals)
        return {
            "scope": self.scope_key,
            "lanes": int(self.spec.lanes),
            "queue_depth": int(self.spec.queue_depth),
            "release_policy": self.spec.release_policy,
            "descriptor_count": len(self.records),
            "descriptor_union_ns": interval_union_ns(desc_intervals),
            "compute_union_ns": interval_union_ns(comp_intervals),
            "uncovered_stars_ns": uncovered_ns,
        }


@dataclass(frozen=True, slots=True)
class SplitBarrierSpec:
    """Declared parameters for Prefill Eager ``TokenDispatcherWithAll2AllV`` D2H."""

    ep_size: int
    num_local_experts: int
    split_dtype_bytes: int = DEFAULT_SPLIT_DTYPE_BYTES
    pinned: bool = True
    non_blocking: bool = True
    host_consume_us: float = 0.0
    histogram_ms: float = 0.0
    repeat_interleave_ms: float = 0.0
    allgather_ms: float = 0.0
    sync_overhead_us: float = 0.0

    def __post_init__(self):
        for field_name in ("ep_size", "num_local_experts", "split_dtype_bytes"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 1:
                raise ModelingError(f"split_barrier_{field_name}_invalid:{val}")
        for field_name in ("host_consume_us", "histogram_ms",
                           "repeat_interleave_ms", "allgather_ms", "sync_overhead_us"):
            val = getattr(self, field_name)
            if (isinstance(val, bool) or not isinstance(val, (int, float))
                    or not math.isfinite(float(val)) or float(val) < 0.0):
                raise ModelingError(f"split_barrier_{field_name}_invalid:{val}")

    @property
    def input_splits_bytes(self) -> int:
        return self.ep_size * self.split_dtype_bytes

    @property
    def output_splits_bytes(self) -> int:
        return self.ep_size * self.split_dtype_bytes

    @property
    def total_d2h_bytes(self) -> int:
        return self.input_splits_bytes + self.output_splits_bytes

    @property
    def requires_explicit_device_sync(self) -> bool:
        """Matches upstream ``token_dispatcher.py:678-682``."""
        return self.num_local_experts <= 1


@dataclass(frozen=True, slots=True)
class SplitBarrierResult:
    """Causal events for one Prefill Eager ``TokenDispatcherWithAll2AllV`` barrier."""

    histogram: Event
    input_splits_d2h: Event
    allgather_counts: Event
    output_splits_d2h: Event
    device_branch_done: Event
    explicit_device_sync: Event | None
    host_consume: Event
    dispatch_ready: Event
    input_splits_bytes: int
    output_splits_bytes: int
    d2h_resources: tuple[str, ...]


class EagerSplitBarrier:
    """Prefill Eager ``TokenDispatcherWithAll2AllV`` D2H host-consumption barrier.

    Models upstream ``vllm-ascend/ops/fused_moe/token_dispatcher.py:647-682``:
    ``histogram`` -> local sum -> ``input_splits`` D2H;
    ``ep_counts`` AllGather -> slice/sum -> ``output_splits`` D2H;
    optional ``repeat_interleave`` (``num_local_experts > 1``) OR explicit
    ``torch.npu.synchronize()`` device fence (``num_local_experts <= 1``);
    host consumes splits -> gates ``dispatch AllToAllV``.
    """

    @staticmethod
    def _d2h_channel_params(machine, *, pinned: bool) -> tuple[float, float]:
        channels = dict(getattr(machine, "transfer_channels", ()) or ())
        decl = dict(channels.get("d2h") or ())
        if not pinned and decl.get("pageable_uni_bw_gbps"):
            bw_gbps = float(decl["pageable_uni_bw_gbps"])
        elif hasattr(machine, "host_solo_bw_gbps"):
            bw_gbps = float(machine.host_solo_bw_gbps("d2h") or 0.0)
        else:
            bw_gbps = float(decl.get("effective_uni_bw_gbps") or 0.0)
        if bw_gbps <= 0.0:
            raise ModelingError("eager_split_barrier_d2h_channel_unconfigured")
        alpha_us = float(decl.get("latency_us", decl.get("alpha_us", 0.0)) or 0.0)
        return bw_gbps * GBPS_TO_BYTES_PER_MS, alpha_us / MS_PER_S

    @classmethod
    def ensure_resources(
        cls,
        engine,
        machine,
        die,
        *,
        pinned: bool = True,
        namespace: str = "",
        d2h_rank_resource: str | None = None,
        host_service_resource: str | None = None,
        device_vector_resource: str | None = None,
    ) -> tuple[str, str, str]:
        """Ensure per-die D2H, shared root-complex D2H, host, and device resources exist."""
        rank = int(getattr(die, "rank", 0))
        d2h_name = d2h_rank_resource or f"{namespace}d2h@rank{rank}"
        host_name = host_service_resource or f"{namespace}host@rank{rank}"
        vec_name = device_vector_resource or f"{namespace}vector@rank{rank}"
        d2h_rate, _ = cls._d2h_channel_params(machine, pinned=pinned)
        add_res = []
        if d2h_name not in engine.resources:
            add_res.append(Resource(d2h_name, d2h_rate, "bytes"))
        if host_name not in engine.resources:
            add_res.append(Resource(host_name, 1.0, "ms", "fifo"))
        if vec_name not in engine.resources:
            add_res.append(Resource(vec_name, 1.0, "ms"))
        for rc_res in root_complex_resources_for_dies(
            machine, (die,), namespace=namespace
        ):
            if rc_res.name not in engine.resources:
                add_res.append(rc_res)
        if add_res:
            engine.extend(add_res)
        return d2h_name, host_name, vec_name

    @classmethod
    def submit(
        cls,
        engine,
        machine,
        die,
        spec: SplitBarrierSpec,
        ready: Event,
        *,
        name: str = "eager_split_barrier",
        namespace: str = "",
        d2h_rank_resource: str | None = None,
        host_service_resource: str | None = None,
        device_vector_resource: str | None = None,
        allgather_event: Event | None = None,
        independent_device_events: Iterable[Event] = (),
        owner: str = "",
    ) -> SplitBarrierResult:
        """Submit the Prefill Eager split-size D2H pipeline onto ``engine``."""
        d2h_name, host_name, vec_name = cls.ensure_resources(
            engine, machine, die,
            pinned=spec.pinned,
            namespace=namespace,
            d2h_rank_resource=d2h_rank_resource,
            host_service_resource=host_service_resource,
            device_vector_resource=device_vector_resource,
        )
        _, d2h_latency_ms = cls._d2h_channel_params(machine, pinned=spec.pinned)

        histogram = (
            engine.submit(
                f"{name}:histogram",
                {vec_name: spec.histogram_ms},
                after=(ready,),
                owner=owner,
            )
            if spec.histogram_ms > 0.0
            else engine.submit(f"{name}:histogram", after=(ready,), owner=owner)
        )

        # Branch A: local counts sum -> input_splits D2H
        in_demand = host_transfer_demand(
            machine, die, "d2h", spec.input_splits_bytes,
            namespace=namespace, rank_resource=d2h_name,
        )
        in_svc = engine.submit(
            f"{name}:input_splits_d2h:svc",
            in_demand,
            after=(histogram,),
            owner=owner,
            coupled=len(in_demand) > 1,
        )
        input_d2h = engine.submit(
            f"{name}:input_splits_d2h:arr",
            after=(in_svc,),
            latency=d2h_latency_ms,
            owner=owner,
        )

        # Branch B: EP counts AllGather -> local slice/sum -> output_splits D2H
        if allgather_event is not None:
            ag_done = engine.submit(
                f"{name}:ep_counts_allgather",
                after=(histogram, allgather_event),
                owner=owner,
            )
        else:
            ag_done = engine.submit(
                f"{name}:ep_counts_allgather",
                after=(histogram,),
                latency=spec.allgather_ms,
                owner=owner,
            )
        out_after = (ag_done,) if spec.non_blocking else (ag_done, input_d2h)
        out_demand = host_transfer_demand(
            machine, die, "d2h", spec.output_splits_bytes,
            namespace=namespace, rank_resource=d2h_name,
        )
        out_svc = engine.submit(
            f"{name}:output_splits_d2h:svc",
            out_demand,
            after=out_after,
            owner=owner,
            coupled=len(out_demand) > 1,
        )
        output_d2h = engine.submit(
            f"{name}:output_splits_d2h:arr",
            after=(out_svc,),
            latency=d2h_latency_ms,
            owner=owner,
        )

        # Branch C: repeat_interleave (num_local_experts > 1) vs explicit device fence
        if spec.requires_explicit_device_sync:
            fence_deps = (input_d2h, output_d2h, *tuple(independent_device_events))
            explicit_sync = engine.submit(
                f"{name}:torch_npu_synchronize",
                after=fence_deps,
                latency=spec.sync_overhead_us / MS_PER_S,
                owner=owner,
            )
            device_branch_done = explicit_sync
            host_deps = (explicit_sync,)
        else:
            explicit_sync = None
            device_branch_done = (
                engine.submit(
                    f"{name}:repeat_interleave",
                    {vec_name: spec.repeat_interleave_ms},
                    after=(histogram,),
                    owner=owner,
                )
                if spec.repeat_interleave_ms > 0.0
                else engine.submit(
                    f"{name}:repeat_interleave",
                    after=(histogram,),
                    owner=owner,
                )
            )
            host_deps = (input_d2h, output_d2h)

        host_consume = engine.submit(
            f"{name}:host_consume_splits",
            {host_name: spec.host_consume_us / MS_PER_S},
            after=host_deps,
            owner=owner,
        )
        dispatch_ready = engine.submit(
            f"{name}:dispatch_ready",
            after=(host_consume, device_branch_done),
            owner=owner,
        )
        return SplitBarrierResult(
            histogram=histogram,
            input_splits_d2h=input_d2h,
            allgather_counts=ag_done,
            output_splits_d2h=output_d2h,
            device_branch_done=device_branch_done,
            explicit_device_sync=explicit_sync,
            host_consume=host_consume,
            dispatch_ready=dispatch_ready,
            input_splits_bytes=spec.input_splits_bytes,
            output_splits_bytes=spec.output_splits_bytes,
            d2h_resources=tuple(sorted(in_demand)),
        )
