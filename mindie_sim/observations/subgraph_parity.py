"""Causal 3-layer live NPU subgraph parity verification engine.

[Role] Stream and correlate Ascend profiler traces (``trace_view.json`` and
  sanitized offline A3/A5 capture bundles) across PyTorch/vLLM -> CANN API ->
  NPU device kernels using STRICT unique ``(rank_id, stream_id, flow_id)`` and
  ACLGraph node-table joins, and verify causal DAG parity invariants against
  independent simulator predictions and ``ProbedAbiRegistry`` 1-to-many kernel
  expansions.
[Invariants]
  * Timestamp-proximity guesses are NEVER promoted to exact causal attribution;
    ANY ``ambiguous_proximity`` event fails ``causal_dag_parity``.
  * Every participating observed rank MUST execute >= 1 real NPU device kernel
    AND match a non-empty independent predicted graph (1:1 occurrence matching,
    strict task-type matching, and predecessor completion <= consumer start).
  * Offline histogram/summary captures are labeled ``offline_summary_inventory``
    and can certify inventory/expansion parity in ``offline_inventory_audit``
    mode, NEVER ``causal_parity_certified=True``.
  * Streaming JSON trace ingestion reads bounded chunks (<= 1 MiB) and never
    loads whole files > 50 MiB into RAM.
  * Pure structural/kernel-level comparison; zero model-name branches (AG014-clean).
[Timeouts & Locks] Bounded streaming JSON/DAG correlation and comparison; no
  NPU device context, no subprocesses, no locks.
[Side-Effect Policy] Read-only file parsing returning immutable report dataclasses.
"""
from __future__ import annotations

from array import array
from bisect import bisect_left
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal, TypeAlias

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.schema.common import ModelingError

REAL_NPU_TASK_TYPES: frozenset[str] = frozenset({
    "AI_CORE",
    "MIX_AIC",
    "AI_VECTOR_CORE",
    "MIX_AIV",
    "COMMUNICATION",
})
_NPU_COMPUTE_TASK_TYPES: frozenset[str] = REAL_NPU_TASK_TYPES - {"COMMUNICATION"}
AI_CPU_TASK_TYPE = "AI_CPU"
DEFAULT_MIN_ATTRIBUTION_RATE = 1.0
PROXIMITY_AMBIGUITY_WINDOW_US = 5.0
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_SINGLE_EVENT_BYTES = 8 * 1024 * 1024
_MAX_SKIP_DEPTH = 128
_TIMESTAMP_EPS_US = 1e-6
_MAX_REPORTED_SAMPLES = 64
_STREAM_WINDOW_FLUSH = 256
_FLOW_CATEGORIES: frozenset[str] = frozenset({"hosttodevice", "async_npu", "torch_to_npu", "flow"})
_KERNEL_CATEGORIES: frozenset[str] = frozenset({"kernel", "device_kernel", "hccl", "npu_kernel"})
_ACLGRAPH_CATEGORIES: frozenset[str] = frozenset({"aclgraph_capture", "graph_node", "hosttodevice(graph)"})
_HOST_OP_CATEGORIES: frozenset[str] = frozenset({"cpu_op", "cann_api", "python_function"})

_MISSING_SO_PATTERNS = (
    re.compile(r"undefined symbol[:\s]+([A-Za-z0-9_]+)"),
    re.compile(r"(aclnn[A-Za-z0-9_]+)\s+(?:not found|missing|absent)"),
    re.compile(r"cannot open shared object file[:\s]+([^\s:;]+)"),
)
_NO_MODULE_NAMED_RE = re.compile(r"No module named ['\"]([^'\"]+)['\"]")
_MODULE_NOT_FOUND_FALLBACK_RE = re.compile(r"ModuleNotFoundError[:\s]+([A-Za-z0-9_.]+)")

ParityDiffCode: TypeAlias = Literal[
    "DEPENDENCY_ORDER_MISMATCH",
    "WRONG_RANK_COLLECTIVE",
    "WRONG_STREAM_LANE",
    "MISSING_OR_EXTRA_LAUNCH",
    "EXPANSION_MISMATCH",
    "SHAPE_MISMATCH",
    "UNATTRIBUTED_KERNEL",
    "MISSING_AI_CPU_METADATA",
    "MISSING_SO_SYMBOL",
    "MISSING_PYTHON_PACKAGE",
]
AttributionKind: TypeAlias = Literal[
    "exact_flow_id",
    "aclgraph_node_table",
    "offline_summary_inventory",
    "ambiguous_proximity",
    "unattributed",
]
VerificationMode: TypeAlias = Literal["causal_dag_parity", "offline_inventory_audit"]
PhaseKind: TypeAlias = Literal["prefill", "decode"]
ExecModeKind: TypeAlias = Literal["eager", "aclgraph"]


@dataclass(frozen=True, slots=True)
class ObservedKernelNode:
    """One device kernel node observed in an Ascend trace or capture summary."""

    node_id: str
    rank_id: int
    stream_id: str
    stream_lane: str
    task_type: str
    kernel_name: str
    cann_api: str = ""
    cpu_op: str = ""
    vllm_scope: str = ""
    input_shapes: str = ""
    input_dtypes: str = ""
    dur_us: float = 0.0
    start_us: float = 0.0
    flow_id: str = ""
    attribution_kind: AttributionKind = "exact_flow_id"
    predecessors: tuple[str, ...] = ()
    collective_group: str = ""
    comm_ranks: tuple[int, ...] = ()
    count: int = 1


@dataclass(frozen=True, slots=True)
class ObservedRankDAG:
    """Per-rank causal DAG extracted from a real NPU trace or capture summary."""

    rank_id: int
    phase: PhaseKind
    exec_mode: ExecModeKind
    nodes: tuple[ObservedKernelNode, ...]
    exact_causal_attribution_rate: float
    ambiguous_matches: tuple[str, ...]
    unattributed_device_kernels: tuple[str, ...]
    npu_device_kernel_count: int
    ai_cpu_kernel_count: int
    task_type_counts: tuple[tuple[str, int], ...] = ()
    is_offline_summary: bool = False

    def task_type_dict(self) -> dict[str, int]:
        return dict(self.task_type_counts)


@dataclass(frozen=True, slots=True)
class PredictedKernelNode:
    """Simulator-predicted launch node for causal parity checking."""

    node_id: str
    rank_id: int
    stream_lane: str
    task_type: str
    api_or_symbol: str
    expected_kernels: tuple[str, ...] = ()
    expected_task_types: tuple[str, ...] = ()
    input_shapes: str = ""
    predecessors: tuple[str, ...] = ()
    collective_group: str = ""
    comm_ranks: tuple[int, ...] = ()
    vllm_scope: str = ""
    is_ai_cpu_metadata: bool = False
    min_count: int = 1
    max_count: int | None = None


@dataclass(frozen=True, slots=True)
class ParityDiff:
    """One structured causal parity mismatch."""

    code: ParityDiffCode
    rank_id: int
    node_id: str
    detail: str
    expected: str = ""
    observed: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "rank_id": self.rank_id,
            "node_id": self.node_id,
            "detail": self.detail,
            "expected": self.expected,
            "observed": self.observed,
        }


@dataclass(frozen=True, slots=True)
class LiveSubgraphParityReport:
    """Authoritative causal parity report for one live or offline NPU capture."""

    signature_id: str
    hardware_sku: str
    config_digest: str
    world_ranks: tuple[int, ...]
    phase: PhaseKind
    exec_mode: ExecModeKind
    vllm_ascend_rev: str
    sim_rev: str
    npu_device_kernel_count: int
    exact_causal_attribution_rate: float
    ambiguous_matches: tuple[str, ...]
    unattributed_device_kernels: tuple[str, ...]
    missing_so_symbols: tuple[str, ...]
    missing_python_modules: tuple[str, ...]
    dag_order_diffs: tuple[ParityDiff, ...]
    rank_collective_diffs: tuple[ParityDiff, ...]
    stream_lane_diffs: tuple[ParityDiff, ...]
    expansion_diffs: tuple[ParityDiff, ...]
    shape_diffs: tuple[ParityDiff, ...]
    launch_diffs: tuple[ParityDiff, ...]
    passed: bool
    verification_mode: VerificationMode = "causal_dag_parity"
    causal_parity_certified: bool = False

    def all_diffs(self) -> tuple[ParityDiff, ...]:
        return (
            *self.dag_order_diffs,
            *self.rank_collective_diffs,
            *self.stream_lane_diffs,
            *self.expansion_diffs,
            *self.shape_diffs,
            *self.launch_diffs,
        )

    def assert_parity(self) -> None:
        """Fail fast if real NPU execution was absent or any causal diff occurred."""
        if self.npu_device_kernel_count <= 0:
            raise ModelingError(
                f"live_parity_zero_npu_kernels:{self.signature_id}:{self.hardware_sku}"
            )
        if not self.passed:
            codes = ",".join(sorted({d.code for d in self.all_diffs()}))
            if self.missing_so_symbols:
                codes = f"{codes},MISSING_SO_SYMBOL" if codes else "MISSING_SO_SYMBOL"
            if self.missing_python_modules:
                codes = f"{codes},MISSING_PYTHON_PACKAGE" if codes else "MISSING_PYTHON_PACKAGE"
            raise ModelingError(
                f"live_subgraph_parity_failed:{self.signature_id}:{codes}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "signature_id": self.signature_id,
            "hardware_sku": self.hardware_sku,
            "config_digest": self.config_digest,
            "world_ranks": list(self.world_ranks),
            "phase": self.phase,
            "exec_mode": self.exec_mode,
            "vllm_ascend_rev": self.vllm_ascend_rev,
            "sim_rev": self.sim_rev,
            "verification_mode": self.verification_mode,
            "causal_parity_certified": self.causal_parity_certified,
            "npu_device_kernel_count": self.npu_device_kernel_count,
            "exact_causal_attribution_rate": round(self.exact_causal_attribution_rate, 6),
            "ambiguous_matches": list(self.ambiguous_matches),
            "unattributed_device_kernels": list(self.unattributed_device_kernels),
            "missing_so_symbols": list(self.missing_so_symbols),
            "missing_python_modules": list(self.missing_python_modules),
            "dag_order_diffs": [d.to_dict() for d in self.dag_order_diffs],
            "rank_collective_diffs": [d.to_dict() for d in self.rank_collective_diffs],
            "stream_lane_diffs": [d.to_dict() for d in self.stream_lane_diffs],
            "expansion_diffs": [d.to_dict() for d in self.expansion_diffs],
            "shape_diffs": [d.to_dict() for d in self.shape_diffs],
            "launch_diffs": [d.to_dict() for d in self.launch_diffs],
            "passed": self.passed,
        }


def classify_runtime_errors(
    error_messages: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Separate missing ``.so`` symbols from missing Python packages cleanly (1 hit per error)."""
    missing_so: list[str] = []
    missing_py: list[str] = []
    for raw in error_messages:
        text = str(raw or "")
        for pat in _MISSING_SO_PATTERNS:
            for hit in pat.findall(text):
                if hit not in missing_so:
                    missing_so.append(hit)
        py_hits = _NO_MODULE_NAMED_RE.findall(text)
        if not py_hits:
            py_hits = _MODULE_NOT_FOUND_FALLBACK_RE.findall(text)
        for hit in py_hits:
            if hit not in missing_py:
                missing_py.append(hit)
    return tuple(missing_so), tuple(missing_py)


def _infer_stream_lane(stream_id: str, kernel_name: str, task_type: str) -> str:
    low_stream = stream_id.lower()
    low_kernel = kernel_name.lower()
    if "shared" in low_stream:
        return "shared_expert"
    if "comm" in low_stream or task_type == "COMMUNICATION" or low_kernel.startswith("hcom_"):
        return "comm"
    if "aux" in low_stream:
        return "auxiliary"
    return "main"


def _normalize_task_type(raw: str) -> str:
    upper = str(raw or "").strip().upper()
    if upper in {"AI_CORE", "AIC", "AICORE"}:
        return "AI_CORE"
    if upper in {"MIX_AIC", "MIXAIC"}:
        return "MIX_AIC"
    if upper in {"AI_VECTOR_CORE", "AIV", "VECTOR"}:
        return "AI_VECTOR_CORE"
    if upper in {"MIX_AIV", "MIXAIV"}:
        return "MIX_AIV"
    if upper in {"COMMUNICATION", "HCCL"}:
        return "COMMUNICATION"
    if upper in {"AI_CPU", "AICPU"}:
        return AI_CPU_TASK_TYPE
    return upper or "UNKNOWN"


def _intern(text: str) -> str:
    return sys.intern(text) if text else ""


@dataclass(frozen=True, slots=True)
class _FlowRecord:
    ts: float
    cann_api: str
    cpu_op: str
    vllm_scope: str
    stream_id: str = ""


@dataclass(frozen=True, slots=True)
class _RawDeviceLaunch:
    kernel: str
    stream_id: str
    task_type: str
    fid: str
    gid: str
    ts: float
    dur: float
    cann_api: str
    cpu_op: str
    vllm_scope: str
    lane: str
    input_shapes: str
    input_dtypes: str
    collective_group: str
    comm_ranks: tuple[int, ...]


def _extract_host_record(ev: Mapping[str, Any], *, explicit_stream: str = "") -> _FlowRecord:
    args = ev.get("args") if isinstance(ev.get("args"), Mapping) else {}
    return _FlowRecord(
        ts=float(ev.get("ts", 0.0)),
        cann_api=_intern(str(args.get("cann_api", ev.get("name", ""))).strip()),
        cpu_op=_intern(str(args.get("cpu_op", "")).strip()),
        vllm_scope=_intern(str(args.get("vllm_scope", "")).strip()),
        stream_id=_intern(explicit_stream),
    )


def _extract_device_launch(ev: Mapping[str, Any]) -> _RawDeviceLaunch:
    args = ev.get("args") if isinstance(ev.get("args"), Mapping) else {}
    kernel = _intern(str(ev.get("name", "")).strip())
    stream_id = _intern(str(args.get("stream_id", ev.get("tid", "0"))).strip())
    task_type = _intern(_normalize_task_type(str(args.get("Task Type", args.get("task_type", "")))))
    raw_ranks = args.get("comm_ranks", ())
    comm_ranks = (
        tuple(int(r) for r in raw_ranks)
        if isinstance(raw_ranks, Sequence) and not isinstance(raw_ranks, str)
        else ()
    )
    lane = _intern(str(args.get("stream_lane", "")).strip() or _infer_stream_lane(stream_id, kernel, task_type))
    return _RawDeviceLaunch(
        kernel=kernel,
        stream_id=stream_id,
        task_type=task_type,
        fid=str(args.get("flow_id", ev.get("flow_id", ""))).strip(),
        gid=str(args.get("graph_node_id", "")).strip(),
        ts=float(ev.get("ts", 0.0)),
        dur=float(ev.get("dur", 0.0)),
        cann_api=_intern(str(args.get("cann_api", "")).strip()),
        cpu_op=_intern(str(args.get("cpu_op", "")).strip()),
        vllm_scope=_intern(str(args.get("vllm_scope", "")).strip()),
        lane=lane,
        input_shapes=str(args.get("Input Shapes", args.get("input_shapes", ""))),
        input_dtypes=str(args.get("Input Data Types", args.get("input_dtypes", ""))),
        collective_group=_intern(str(args.get("collective_group", ""))),
        comm_ranks=comm_ranks,
    )


def _is_device_kernel_event(ev: Mapping[str, Any], cat: str, args: Mapping[str, Any]) -> bool:
    if cat in _KERNEL_CATEGORIES:
        return True
    tt = _normalize_task_type(str(args.get("Task Type", args.get("task_type", ""))))
    return tt in (REAL_NPU_TASK_TYPES | {AI_CPU_TASK_TYPE})


def _near_sorted_ts(sorted_ts: array, target: float, window: float) -> bool:
    if not sorted_ts:
        return False
    idx = bisect_left(sorted_ts, target - window)
    return idx < len(sorted_ts) and sorted_ts[idx] <= target + window


class _CausalFlowState:
    """Bounded causal join table supporting exact (rank, stream, flow_id), Chrome s/f, and ACLGraph."""

    __slots__ = (
        "rank_id", "exec_mode", "stream_flows", "unbound_flows", "unbound_tid",
        "finish_by_ts", "finish_by_id", "graph_nodes",
        "consumed_flows", "_consumed_ring", "_max_consumed_ring",
        "host_ts", "host_sorted",
    )

    def __init__(self, rank_id: int, exec_mode: ExecModeKind, *, max_consumed_ring: int = 0) -> None:
        self.rank_id = rank_id
        self.exec_mode = exec_mode
        # (stream_id, fid) -> _FlowRecord or None (poisoned duplicate)
        self.stream_flows: dict[tuple[str, str], _FlowRecord | None] = {}
        # fid -> _FlowRecord or None (Chrome 's' without explicit stream_id, bound only via 'f')
        self.unbound_flows: dict[str, _FlowRecord | None] = {}
        # fid -> host tid recorded alongside unbound Chrome 's' so completion reclaims stream_flows
        self.unbound_tid: dict[str, str] = {}
        # (stream_id, ts) -> fid or None (Chrome 'f' endpoint; duplicate poisoned)
        self.finish_by_ts: dict[tuple[str, float], str | None] = {}
        # (stream_id, fid) -> dest_ts or None (Chrome 'f' endpoint)
        self.finish_by_id: dict[tuple[str, str], float | None] = {}
        # (stream_id, gid) -> _FlowRecord or None (used ONLY when exec_mode == "aclgraph")
        self.graph_nodes: dict[tuple[str, str], _FlowRecord | None] = {}
        self.consumed_flows: set[tuple[str, str]] = set()
        self._consumed_ring: deque[tuple[str, str]] = deque()
        self._max_consumed_ring = max(0, int(max_consumed_ring))
        self.host_ts: array = array("d")
        self.host_sorted: bool = True

    def _mark_consumed(self, fkey: tuple[str, str]) -> None:
        self.consumed_flows.add(fkey)
        if self._max_consumed_ring > 0:
            self._consumed_ring.append(fkey)
            while len(self._consumed_ring) > self._max_consumed_ring:
                old = self._consumed_ring.popleft()
                self.consumed_flows.discard(old)

    def _reclaim_flow(self, fkey: tuple[str, str], fid: str) -> None:
        self._mark_consumed(fkey)
        self.stream_flows.pop(fkey, None)
        self.unbound_flows.pop(fid, None)
        host_tid = self.unbound_tid.pop(fid, "")
        if host_tid:
            self.stream_flows.pop((host_tid, fid), None)
        self.finish_by_id.pop(fkey, None)

    def ingest_non_kernel(self, ev: Mapping[str, Any], ph: str, cat: str, args: Mapping[str, Any]) -> None:
        if ph == "s" and cat in _FLOW_CATEGORIES:
            fid = str(ev.get("id", "")).strip()
            if not fid:
                return
            explicit_sid = str(args.get("stream_id", "")).strip()
            if explicit_sid:
                key = (_intern(explicit_sid), fid)
                self.stream_flows[key] = None if key in self.stream_flows else _extract_host_record(ev, explicit_stream=explicit_sid)
                self.unbound_flows[fid] = None
            else:
                # Standard Chrome 's' on host thread (may pair with 'f' on device stream).
                self.unbound_flows[fid] = None if fid in self.unbound_flows else _extract_host_record(ev)
                tid_sid = _intern(str(ev.get("tid", "")).strip())
                if tid_sid:
                    tkey = (tid_sid, fid)
                    self.stream_flows[tkey] = None if tkey in self.stream_flows else _extract_host_record(ev, explicit_stream=tid_sid)
                    if fid not in self.unbound_tid:
                        self.unbound_tid[fid] = tid_sid
        elif ph == "f" and cat in _FLOW_CATEGORIES:
            fid = str(ev.get("id", "")).strip()
            dest_sid = _intern(str(args.get("stream_id", ev.get("tid", ""))).strip())
            dest_ts = float(ev.get("ts", 0.0))
            if not fid or not dest_sid:
                return
            ts_key = (dest_sid, dest_ts)
            self.finish_by_ts[ts_key] = None if ts_key in self.finish_by_ts else fid
            id_key = (dest_sid, fid)
            self.finish_by_id[id_key] = None if id_key in self.finish_by_id else dest_ts
        elif ph == "X" and cat in _ACLGRAPH_CATEGORIES and self.exec_mode == "aclgraph":
            gid = str(args.get("graph_node_id", ev.get("id", ""))).strip()
            sid = _intern(str(args.get("stream_id", ev.get("tid", "0"))).strip())
            if gid and sid:
                gkey = (sid, gid)
                self.graph_nodes[gkey] = None if gkey in self.graph_nodes else _extract_host_record(ev, explicit_stream=sid)
        elif ph == "X" and cat in _HOST_OP_CATEGORIES:
            ts = float(ev.get("ts", 0.0))
            if self.host_ts and ts < self.host_ts[-1]:
                self.host_sorted = False
            self.host_ts.append(ts)

    def _resolve_flow(self, dev: _RawDeviceLaunch) -> tuple[_FlowRecord | None, str, bool]:
        """Return ``(matched_record, resolved_fid, saw_ambiguous_or_invalid_flow)``."""
        sid = dev.stream_id
        # Path A: explicit flow_id on device kernel event.
        if dev.fid:
            fkey = (sid, dev.fid)
            if fkey in self.consumed_flows:
                return None, dev.fid, True
            rec = self.stream_flows.get(fkey)
            if rec is None and fkey in self.stream_flows:
                return None, dev.fid, True
            if fkey in self.finish_by_id:
                f_ts = self.finish_by_id.pop(fkey)
                if f_ts is None or abs(f_ts - dev.ts) > _TIMESTAMP_EPS_US:
                    return None, dev.fid, True
                self.finish_by_ts.pop((sid, f_ts), None)
                if rec is None:
                    u_rec = self.unbound_flows.get(dev.fid)
                    if u_rec is None:
                        return None, dev.fid, True
                    rec = u_rec
            if rec is not None:
                if rec.ts > dev.ts + _TIMESTAMP_EPS_US:
                    return None, dev.fid, True
                self._reclaim_flow(fkey, dev.fid)
                return rec, dev.fid, False
            return None, dev.fid, False

        # Path B: standard Chrome 's' -> 'f' ending on (dev.stream_id, dev.ts).
        ts_key = (sid, dev.ts)
        if ts_key not in self.finish_by_ts:
            return None, "", False
        fid = self.finish_by_ts.pop(ts_key)
        if not fid:
            return None, "", True
        fkey = (sid, fid)
        self.finish_by_id.pop(fkey, None)
        if fkey in self.consumed_flows:
            return None, fid, True
        rec = self.stream_flows.get(fkey)
        if rec is None and fkey in self.stream_flows:
            return None, fid, True
        if rec is None:
            # Only bind unbound 's' if NO conflicting explicit-stream 's' exists for fid.
            rec = self.unbound_flows.get(fid)
            if rec is None:
                return None, fid, True
        if rec.ts > dev.ts + _TIMESTAMP_EPS_US:
            return None, fid, True
        self._reclaim_flow(fkey, fid)
        return rec, fid, False

    def attribute(self, dev: _RawDeviceLaunch) -> tuple[AttributionKind, str, str, str, str]:
        """Attribute one launch strictly; returns ``(kind, flow_id, cann_api, cpu_op, scope)``."""
        cann_api, cpu_op, scope = dev.cann_api, dev.cpu_op, dev.vllm_scope
        rec, resolved_fid, invalid_flow = self._resolve_flow(dev)
        if rec is not None:
            if scope and rec.vllm_scope and scope != rec.vllm_scope:
                return "ambiguous_proximity", resolved_fid, cann_api or rec.cann_api, cpu_op or rec.cpu_op, scope
            return (
                "exact_flow_id",
                resolved_fid,
                cann_api or rec.cann_api,
                cpu_op or rec.cpu_op,
                scope or rec.vllm_scope,
            )
        if not invalid_flow and dev.gid and self.exec_mode == "aclgraph":
            gkey = (dev.stream_id, dev.gid)
            g_rec = self.graph_nodes.get(gkey)
            if g_rec is not None:
                return (
                    "aclgraph_node_table",
                    "",
                    cann_api or g_rec.cann_api,
                    cpu_op or g_rec.cpu_op,
                    scope or g_rec.vllm_scope,
                )
            if gkey in self.graph_nodes:
                return "ambiguous_proximity", "", cann_api, cpu_op, scope
        if invalid_flow:
            return "ambiguous_proximity", resolved_fid, cann_api, cpu_op, scope
        if not self.host_sorted:
            self.host_ts = array("d", sorted(self.host_ts))
            self.host_sorted = True
        if _near_sorted_ts(self.host_ts, dev.ts, PROXIMITY_AMBIGUITY_WINDOW_US):
            return "ambiguous_proximity", resolved_fid, cann_api, cpu_op, scope
        return "unattributed", resolved_fid, cann_api, cpu_op, scope


def correlate_trace_view_events(
    events: Iterable[Mapping[str, Any]],
    *,
    rank_id: int = 0,
    phase: PhaseKind = "decode",
    exec_mode: ExecModeKind = "eager",
) -> ObservedRankDAG:
    """Strictly join PyTorch/CANN host events to device kernels via unique (rank, stream, flow_id).

    Cross-stream flow fallbacks, host-after-device joins, eager-mode ACLGraph ID
    reuse, duplicate flow endpoints, and proximity guesses never receive exact
    causal attribution.
    """
    state = _CausalFlowState(
        rank_id=rank_id, exec_mode=exec_mode, max_consumed_ring=0,
    )
    device_launches: list[_RawDeviceLaunch] = []

    for ev in events:
        if not isinstance(ev, Mapping):
            continue
        ph = str(ev.get("ph", ""))
        cat = str(ev.get("cat", "")).lower()
        args = ev.get("args") if isinstance(ev.get("args"), Mapping) else {}
        ev_rank_raw = args.get("rank_id")
        if ev_rank_raw is not None and int(ev_rank_raw) != rank_id:
            continue
        if ph == "X" and _is_device_kernel_event(ev, cat, args):
            device_launches.append(_extract_device_launch(ev))
        else:
            state.ingest_non_kernel(ev, ph, cat, args)

    device_launches.sort(key=lambda item: (item.ts, item.kernel))
    nodes: list[ObservedKernelNode] = []
    ambiguous: list[str] = []
    unattributed: list[str] = []
    last_by_stream: dict[str, str] = {}
    task_counts: dict[str, int] = {}
    exact_count = 0
    npu_count = 0
    aicpu_count = 0

    for idx, dev in enumerate(device_launches):
        task_counts[dev.task_type] = task_counts.get(dev.task_type, 0) + 1
        if dev.task_type in REAL_NPU_TASK_TYPES:
            npu_count += 1
        if dev.task_type == AI_CPU_TASK_TYPE:
            aicpu_count += 1
        attribution, resolved_fid, cann_api, cpu_op, vllm_scope = state.attribute(dev)
        if attribution in {"exact_flow_id", "aclgraph_node_table"}:
            exact_count += 1
        elif attribution == "ambiguous_proximity":
            ambiguous.append(f"r{rank_id}:{idx}:{dev.kernel}")
        else:
            unattributed.append(f"r{rank_id}:{idx}:{dev.kernel}")
        node_id = f"r{rank_id}.s{dev.stream_id}.k{idx}"
        preds = (last_by_stream[dev.stream_id],) if dev.stream_id in last_by_stream else ()
        last_by_stream[dev.stream_id] = node_id
        nodes.append(ObservedKernelNode(
            node_id=node_id,
            rank_id=rank_id,
            stream_id=dev.stream_id,
            stream_lane=dev.lane,
            task_type=dev.task_type,
            kernel_name=dev.kernel,
            cann_api=cann_api,
            cpu_op=cpu_op,
            vllm_scope=vllm_scope,
            input_shapes=dev.input_shapes,
            input_dtypes=dev.input_dtypes,
            dur_us=dev.dur,
            start_us=dev.ts,
            flow_id=resolved_fid,
            attribution_kind=attribution,
            predecessors=preds,
            collective_group=dev.collective_group,
            comm_ranks=dev.comm_ranks,
            count=1,
        ))
    device_launches.clear()

    total = len(nodes)
    rate = (exact_count / total) if total > 0 else 0.0
    return ObservedRankDAG(
        rank_id=rank_id,
        phase=phase,
        exec_mode=exec_mode,
        nodes=tuple(nodes),
        exact_causal_attribution_rate=rate,
        ambiguous_matches=tuple(ambiguous),
        unattributed_device_kernels=tuple(unattributed),
        npu_device_kernel_count=npu_count,
        ai_cpu_kernel_count=aicpu_count,
        task_type_counts=tuple(sorted(task_counts.items())),
        is_offline_summary=False,
    )


def _token_matches_node(obs: ObservedKernelNode, tok: str) -> bool:
    """Match device kernel name strictly (exact or explicit underscore token boundary)."""
    if not tok or not obs.kernel_name:
        return False
    if obs.kernel_name == tok:
        return True
    if tok.endswith("_") and obs.kernel_name.startswith(tok):
        return True
    return obs.kernel_name.startswith(tok + "_")


def _node_matches_kernel(obs: ObservedKernelNode, expected_tokens: tuple[str, ...]) -> bool:
    return any(_token_matches_node(obs, tok) for tok in expected_tokens)


def _task_type_compatible(
    pred: PredictedKernelNode,
    obs: ObservedKernelNode,
    *,
    expected_override: str = "",
    allow_multi_compute_fallback: bool = False,
) -> bool:
    expected = _normalize_task_type(expected_override or pred.task_type) if (expected_override or pred.task_type) else ""
    observed = _normalize_task_type(obs.task_type)
    if pred.is_ai_cpu_metadata or expected == AI_CPU_TASK_TYPE:
        return observed == AI_CPU_TASK_TYPE
    if expected == "COMMUNICATION":
        return observed == "COMMUNICATION"
    if not expected:
        return observed in REAL_NPU_TASK_TYPES
    if observed == expected:
        return True
    if allow_multi_compute_fallback and expected in _NPU_COMPUTE_TASK_TYPES:
        return observed in _NPU_COMPUTE_TASK_TYPES
    return False


def _subkernel_task_type(pred: PredictedKernelNode, sub_idx: int) -> str:
    if sub_idx < len(pred.expected_task_types) and pred.expected_task_types[sub_idx]:
        return pred.expected_task_types[sub_idx]
    return pred.task_type


def _expected_expansion_for(
    pred: PredictedKernelNode,
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[str, ...]:
    if pred.expected_kernels:
        return pred.expected_kernels
    exp = registry.device_kernel_expansion(pred.api_or_symbol, device_arch)
    return exp if exp else (pred.api_or_symbol,)


def _candidate_preference_key(
    pred: PredictedKernelNode,
    item: tuple[int, ObservedKernelNode],
) -> tuple[int, int, int]:
    """Prefer exact ``vllm_scope`` and ``input_shapes`` matches before earliest trace index."""
    pos, node = item
    scope_miss = 0 if (pred.vllm_scope and node.vllm_scope and node.vllm_scope == pred.vllm_scope) else (
        1 if (pred.vllm_scope and node.vllm_scope) else 0
    )
    shape_miss = 0 if (
        pred.input_shapes
        and node.input_shapes
        and node.input_shapes != "N/A"
        and node.input_shapes == pred.input_shapes
    ) else (
        1 if (pred.input_shapes and node.input_shapes and node.input_shapes != "N/A") else 0
    )
    return (scope_miss, shape_miss, pos)


def _consume_matching_occurrences(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
) -> tuple[list[tuple[int, ObservedKernelNode]], int, bool]:
    """Consume observed occurrences 1:1 with strict task_type and ordered multi-kernel expansion.

    Returns ``(selected_hits, observed_total, expansion_order_ok)``.
    """
    is_multi = len(expected_kernels) > 1
    offline_multi_fallback = bool(obs_dag.is_offline_summary and is_multi)
    if is_multi and not obs_dag.is_offline_summary:
        # Live trace multi-kernel: check EACH subkernel's exact task_type & strict order!
        scoped_pool = [
            (pos, node)
            for pos, node in enumerate(obs_dag.nodes)
            if remaining_counts.get(pos, 0) > 0
            and _node_matches_kernel(node, expected_kernels)
        ]
        if pred.max_count == 0:
            total_avail = sum(remaining_counts[pos] for pos, _ in scoped_pool)
            return scoped_pool, total_avail, True
        # When vllm_scope is present on both pred and observed nodes, bind to matching scope first.
        if pred.vllm_scope and any(n.vllm_scope == pred.vllm_scope for _, n in scoped_pool):
            scoped_pool = [(pos, n) for pos, n in scoped_pool if n.vllm_scope == pred.vllm_scope]
        cursor = -1
        prev_end_us = -1.0
        ordered_picks: list[tuple[int, ObservedKernelNode]] = []
        for sub_idx, tok in enumerate(expected_kernels):
            exp_tt = _subkernel_task_type(pred, sub_idx)
            found = None
            for pos, node in scoped_pool:
                if pos <= cursor or remaining_counts.get(pos, 0) <= 0:
                    continue
                if prev_end_us > 0.0 and node.start_us + _TIMESTAMP_EPS_US < prev_end_us:
                    continue
                if _token_matches_node(node, tok) and _task_type_compatible(
                    pred, node, expected_override=exp_tt, allow_multi_compute_fallback=False,
                ):
                    found = (pos, node)
                    break
            if found is None:
                return ordered_picks, len(ordered_picks), False
            ordered_picks.append(found)
            cursor = found[0]
            prev_end_us = found[1].start_us + max(0.0, found[1].dur_us)
        for pos, _ in ordered_picks:
            remaining_counts[pos] -= 1
        return ordered_picks, len(ordered_picks), True

    single_exp_tt = _subkernel_task_type(pred, 0) if not is_multi else ""
    candidates: list[tuple[int, ObservedKernelNode]] = [
        (pos, node)
        for pos, node in enumerate(obs_dag.nodes)
        if remaining_counts.get(pos, 0) > 0
        and _node_matches_kernel(node, expected_kernels)
        and _task_type_compatible(
            pred, node,
            expected_override=single_exp_tt,
            allow_multi_compute_fallback=offline_multi_fallback,
        )
    ]
    if is_multi:
        if pred.max_count == 0:
            total_avail = sum(remaining_counts[pos] for pos, _ in candidates)
            return candidates, total_avail, True
        need_per_tok = max(1, pred.min_count)
        selected: list[tuple[int, ObservedKernelNode]] = []
        selected_positions: set[int] = set()
        tok_counts: list[int] = []
        for tok in expected_kernels:
            tok_avail = sum(
                remaining_counts.get(pos, 0) for pos, node in candidates if _token_matches_node(node, tok)
            )
            tok_counts.append(tok_avail)
        min_tok = min(tok_counts) if tok_counts else 0
        max_tok = max(tok_counts) if tok_counts else 0
        if min_tok < need_per_tok:
            present = [(pos, node) for pos, node in candidates if remaining_counts.get(pos, 0) > 0]
            return present, min_tok, False
        if pred.max_count is not None and max_tok > pred.max_count:
            return candidates, max_tok, True
        for tok in expected_kernels:
            rem_need = need_per_tok
            for pos, node in candidates:
                if rem_need <= 0:
                    break
                avail = remaining_counts.get(pos, 0)
                if avail > 0 and _token_matches_node(node, tok):
                    take = min(avail, rem_need)
                    remaining_counts[pos] -= take
                    rem_need -= take
                    if pos not in selected_positions:
                        selected.append((pos, node))
                        selected_positions.add(pos)
        selected.sort(key=lambda item: item[0])
        return selected, need_per_tok, True

    total_available = sum(remaining_counts[pos] for pos, _ in candidates)
    if pred.max_count == 0 or total_available < pred.min_count:
        return candidates, total_available, True
    if pred.max_count is not None and total_available > pred.max_count:
        return candidates, total_available, True

    if not obs_dag.is_offline_summary:
        candidates.sort(key=lambda item: _candidate_preference_key(pred, item))
    need = max(1, pred.min_count)
    selected_single: list[tuple[int, ObservedKernelNode]] = []
    consumed_total = 0
    for pos, node in candidates:
        if need <= 0:
            break
        avail = remaining_counts.get(pos, 0)
        if avail <= 0:
            continue
        take = min(avail, need)
        remaining_counts[pos] -= take
        consumed_total += take
        need -= take
        selected_single.append((pos, node))
    return selected_single, consumed_total, True


def _verify_matched_node_attributes(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    hits: Sequence[tuple[int, ObservedKernelNode]],
    rank_id: int,
    *,
    is_offline_summary: bool,
    exp_diffs: list[ParityDiff],
    lane_diffs: list[ParityDiff],
    comm_diffs: list[ParityDiff],
    shape_diffs: list[ParityDiff],
    launch_diffs: list[ParityDiff],
) -> None:
    """Validate expansion completeness, lane, collective ranks, vllm_scope, and shapes."""
    if len(expected_kernels) > 1 and pred.min_count > 0:
        present = {
            tok for tok in expected_kernels
            if any(_token_matches_node(n, tok) for _, n in hits)
        }
        missing_parts = [tok for tok in expected_kernels if tok not in present]
        if missing_parts:
            exp_diffs.append(ParityDiff(
                code="EXPANSION_MISMATCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"incomplete 1-to-many expansion for {pred.api_or_symbol}: missing {missing_parts}",
                expected=",".join(expected_kernels),
                observed=",".join(sorted(present)),
            ))

    if pred.stream_lane and not all(n.stream_lane == pred.stream_lane for _, n in hits):
        obs_lanes = sorted({n.stream_lane for _, n in hits})
        lane_diffs.append(ParityDiff(
            code="WRONG_STREAM_LANE",
            rank_id=rank_id,
            node_id=pred.node_id,
            detail=f"lane mismatch for {pred.api_or_symbol}: expected {pred.stream_lane}, got {obs_lanes}",
            expected=pred.stream_lane,
            observed=",".join(obs_lanes),
        ))

    if pred.collective_group or pred.comm_ranks:
        group_ok = (not pred.collective_group) or all(
            n.collective_group == pred.collective_group for _, n in hits
        )
        ranks_ok = (not pred.comm_ranks) or all(
            n.comm_ranks == pred.comm_ranks for _, n in hits
        )
        if not (group_ok and ranks_ok):
            comm_diffs.append(ParityDiff(
                code="WRONG_RANK_COLLECTIVE",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=(
                    f"collective mismatch for {pred.api_or_symbol}: "
                    f"expected group={pred.collective_group} ranks={pred.comm_ranks}"
                ),
                expected=f"{pred.collective_group}:{pred.comm_ranks}",
                observed=",".join(f"{n.collective_group}:{n.comm_ranks}" for _, n in hits[:2]),
            ))

    if pred.vllm_scope and not is_offline_summary:
        if not all(n.vllm_scope == pred.vllm_scope for _, n in hits):
            obs_scopes = sorted({n.vllm_scope or "missing" for _, n in hits})
            launch_diffs.append(ParityDiff(
                code="UNATTRIBUTED_KERNEL",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"vllm_scope mismatch for {pred.api_or_symbol}: expected {pred.vllm_scope}, got {obs_scopes}",
                expected=pred.vllm_scope,
                observed=",".join(obs_scopes),
            ))

    if pred.input_shapes:
        obs_shapes = [n.input_shapes for _, n in hits if n.input_shapes and n.input_shapes != "N/A"]
        if not is_offline_summary:
            if len(obs_shapes) != len(hits) or any(s != pred.input_shapes for s in obs_shapes):
                bad_shape = next((s for s in obs_shapes if s != pred.input_shapes), "missing")
                shape_diffs.append(ParityDiff(
                    code="SHAPE_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=f"shape mismatch for {pred.api_or_symbol}: expected {pred.input_shapes}",
                    expected=pred.input_shapes,
                    observed=bad_shape,
                ))
        elif obs_shapes and pred.input_shapes not in obs_shapes:
            shape_diffs.append(ParityDiff(
                code="SHAPE_MISMATCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"shape mismatch for {pred.api_or_symbol}: expected {pred.input_shapes}",
                expected=pred.input_shapes,
                observed=obs_shapes[0],
            ))


def _verify_rank_dependency_order(
    pred_nodes: Sequence[PredictedKernelNode],
    matched_bounds: Mapping[str, tuple[int, int, float, float]],
    rank_id: int,
) -> list[ParityDiff]:
    """Enforce predecessor completion strictly before consumer start across every edge."""
    dag_diffs: list[ParityDiff] = []
    for pred in pred_nodes:
        if pred.node_id not in matched_bounds:
            continue
        cur_start_pos, _, cur_start_us, _ = matched_bounds[pred.node_id]
        for dep_id in pred.predecessors:
            if dep_id not in matched_bounds:
                dag_diffs.append(ParityDiff(
                    code="DEPENDENCY_ORDER_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=f"predecessor {dep_id} was not observed before {pred.node_id}",
                    expected=f"{dep_id} completed before {pred.node_id}",
                    observed="missing_predecessor",
                ))
                continue
            _, dep_end_pos, _, dep_end_us = matched_bounds[dep_id]
            if dep_end_pos >= cur_start_pos:
                dag_diffs.append(ParityDiff(
                    code="DEPENDENCY_ORDER_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=(
                        f"dependency {dep_id} completed at pos {dep_end_pos} "
                        f"after/at consumer {pred.node_id} start pos {cur_start_pos}"
                    ),
                    expected=f"{dep_id}.end_pos < {pred.node_id}.start_pos",
                    observed=f"{dep_end_pos} >= {cur_start_pos}",
                ))
            elif dep_end_us > 0.0 and cur_start_us > 0.0 and dep_end_us > cur_start_us + _TIMESTAMP_EPS_US:
                dag_diffs.append(ParityDiff(
                    code="DEPENDENCY_ORDER_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=(
                        f"dependency {dep_id} ended at {dep_end_us:.3f}us "
                        f"after consumer {pred.node_id} started at {cur_start_us:.3f}us"
                    ),
                    expected=f"{dep_id}.end_us <= {pred.node_id}.start_us",
                    observed=f"{dep_end_us:.3f} > {cur_start_us:.3f}",
                ))
    return dag_diffs


def _check_rank_nodes(
    pred_nodes: Sequence[PredictedKernelNode],
    obs_dag: ObservedRankDAG,
    *,
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[
    list[ParityDiff], list[ParityDiff], list[ParityDiff],
    list[ParityDiff], list[ParityDiff], list[ParityDiff],
]:
    comm_diffs: list[ParityDiff] = []
    lane_diffs: list[ParityDiff] = []
    exp_diffs: list[ParityDiff] = []
    shape_diffs: list[ParityDiff] = []
    launch_diffs: list[ParityDiff] = []

    # Live traces are strictly 1 launch per recorded event; offline summaries honor exact 0 counts.
    remaining_counts: dict[int, int] = {
        pos: (max(0, int(node.count)) if obs_dag.is_offline_summary else 1)
        for pos, node in enumerate(obs_dag.nodes)
    }
    matched_bounds: dict[str, tuple[int, int, float, float]] = {}
    rank_id = obs_dag.rank_id

    for pred in pred_nodes:
        expected_kernels = _expected_expansion_for(pred, device_arch, registry)
        hits, observed_total, order_ok = _consume_matching_occurrences(
            pred, expected_kernels, obs_dag, remaining_counts,
        )

        if len(expected_kernels) > 1 and pred.min_count > 0 and not order_ok:
            exp_diffs.append(ParityDiff(
                code="EXPANSION_MISMATCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=(
                    f"incomplete or out-of-order 1-to-many expansion for {pred.api_or_symbol}: "
                    f"expected ordered {expected_kernels}"
                ),
                expected=",".join(expected_kernels),
                observed=",".join(n.kernel_name for _, n in hits) or "none",
            ))
            continue

        if pred.is_ai_cpu_metadata and observed_total == 0:
            launch_diffs.append(ParityDiff(
                code="MISSING_AI_CPU_METADATA",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"missing AI_CPU metadata kernel {expected_kernels}",
                expected=",".join(expected_kernels),
                observed="0",
            ))
            continue

        if observed_total < pred.min_count:
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=(
                    f"expected>={pred.min_count} launches of {expected_kernels} "
                    f"(task_type={pred.task_type or 'ANY'}), got {observed_total}"
                ),
                expected=f">={pred.min_count}",
                observed=str(observed_total),
            ))
            continue
        if pred.max_count is not None and observed_total > pred.max_count:
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"expected<={pred.max_count} launches of {expected_kernels}, got {observed_total}",
                expected=f"<={pred.max_count}",
                observed=str(observed_total),
            ))
            continue

        if not hits:
            continue

        matched_bounds[pred.node_id] = (
            hits[0][0],
            hits[-1][0],
            min(n.start_us for _, n in hits),
            max(n.start_us + n.dur_us for _, n in hits),
        )
        _verify_matched_node_attributes(
            pred, expected_kernels, hits, rank_id,
            is_offline_summary=obs_dag.is_offline_summary,
            exp_diffs=exp_diffs,
            lane_diffs=lane_diffs,
            comm_diffs=comm_diffs,
            shape_diffs=shape_diffs,
            launch_diffs=launch_diffs,
        )

    # Account for EVERY recorded device kernel launch in both live traces and offline inventory audits.
    for pos, node in enumerate(obs_dag.nodes):
        rem = remaining_counts.get(pos, 0)
        if rem <= 0:
            continue
        if len(launch_diffs) < _MAX_REPORTED_SAMPLES:
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_id,
                node_id=node.node_id,
                detail=f"extra unpredicted device launch {node.kernel_name} ({node.task_type})",
                expected="0 extra launches",
                observed=node.kernel_name,
            ))

    dag_diffs = _verify_rank_dependency_order(pred_nodes, matched_bounds, rank_id)
    return dag_diffs, comm_diffs, lane_diffs, exp_diffs, shape_diffs, launch_diffs


def _validate_observed_rank_headers(
    observed_dags: Sequence[ObservedRankDAG],
    by_rank_pred: Mapping[int, Sequence[PredictedKernelNode]],
    *,
    has_predictions: bool,
    verification_mode: VerificationMode,
    min_attribution_rate: float,
) -> list[ParityDiff]:
    """Validate per-rank uniqueness, real-NPU execution, and causal attribution prerequisites."""
    diffs: list[ParityDiff] = []
    seen_ranks: set[int] = set()
    for obs_dag in observed_dags:
        r = obs_dag.rank_id
        if r in seen_ranks:
            diffs.append(ParityDiff(
                code="WRONG_RANK_COLLECTIVE",
                rank_id=r,
                node_id="duplicate_rank",
                detail=f"duplicate observed trace for rank {r}",
                expected=f"unique trace for rank {r}",
                observed=f"duplicate rank {r}",
            ))
            continue
        seen_ranks.add(r)
        if obs_dag.npu_device_kernel_count <= 0:
            diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=r,
                node_id="rank_execution",
                detail=f"rank {r} recorded zero real NPU device kernels",
                expected=">=1 real NPU kernels",
                observed="0",
            ))
        if has_predictions and r not in by_rank_pred:
            diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=r,
                node_id="predicted_graph",
                detail=f"participating rank {r} has no predicted subgraph",
                expected=f"predicted nodes for rank {r}",
                observed="0",
            ))
        if verification_mode == "causal_dag_parity":
            if obs_dag.is_offline_summary:
                diffs.append(ParityDiff(
                    code="UNATTRIBUTED_KERNEL",
                    rank_id=r,
                    node_id="offline_summary",
                    detail="offline histogram/summary capture cannot certify causal DAG parity",
                    expected="live flow_id/aclgraph trace",
                    observed="offline_summary_inventory",
                ))
            if obs_dag.ambiguous_matches:
                diffs.append(ParityDiff(
                    code="UNATTRIBUTED_KERNEL",
                    rank_id=r,
                    node_id="ambiguous_proximity",
                    detail=(
                        f"rank {r} has {len(obs_dag.ambiguous_matches)} ambiguous proximity "
                        f"match(es); zero ambiguous matches required for causal parity"
                    ),
                    expected="0 ambiguous_matches",
                    observed=str(len(obs_dag.ambiguous_matches)),
                ))
            if obs_dag.npu_device_kernel_count > 0 and obs_dag.exact_causal_attribution_rate < min_attribution_rate:
                diffs.append(ParityDiff(
                    code="UNATTRIBUTED_KERNEL",
                    rank_id=r,
                    node_id="attribution",
                    detail=(
                        f"rank {r} exact causal attribution rate "
                        f"{obs_dag.exact_causal_attribution_rate:.3f} < {min_attribution_rate:.3f}"
                    ),
                    expected=f">={min_attribution_rate:.3f}",
                    observed=f"{obs_dag.exact_causal_attribution_rate:.3f}",
                ))
    return diffs


class SubgraphParityChecker:
    """Compare independent predicted per-rank DAGs against real NPU ``ObservedRankDAG``s."""

    def __init__(self, registry: ProbedAbiRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_abi_registry()

    def compare(
        self,
        *,
        signature_id: str,
        hardware_sku: str,
        device_arch: str,
        config_payload: Mapping[str, Any],
        observed_dags: Sequence[ObservedRankDAG],
        predicted_nodes: Sequence[PredictedKernelNode],
        vllm_ascend_rev: str = "",
        sim_rev: str = "",
        error_messages: Sequence[str] = (),
        min_attribution_rate: float = DEFAULT_MIN_ATTRIBUTION_RATE,
        verification_mode: VerificationMode = "causal_dag_parity",
    ) -> LiveSubgraphParityReport:
        if not observed_dags:
            raise ModelingError(f"live_parity_empty_observed_dags:{signature_id}")
        config_digest = hashlib.sha256(
            json.dumps(config_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        world_ranks = tuple(sorted({d.rank_id for d in observed_dags}))
        phase = observed_dags[0].phase
        exec_mode = observed_dags[0].exec_mode

        total_npu = sum(d.npu_device_kernel_count for d in observed_dags)
        total_nodes = sum(len(d.nodes) for d in observed_dags)
        weighted_attr = (
            sum(d.exact_causal_attribution_rate * len(d.nodes) for d in observed_dags) / total_nodes
            if total_nodes > 0 else 0.0
        )
        all_ambiguous = tuple(item for d in observed_dags for item in d.ambiguous_matches)
        all_unattributed = tuple(item for d in observed_dags for item in d.unattributed_device_kernels)
        missing_so, missing_py = classify_runtime_errors(error_messages)

        by_rank_obs = {d.rank_id: d for d in observed_dags}
        by_rank_pred: dict[int, list[PredictedKernelNode]] = {}
        for pred in predicted_nodes:
            by_rank_pred.setdefault(pred.rank_id, []).append(pred)

        dag_diffs: list[ParityDiff] = []
        comm_diffs: list[ParityDiff] = []
        lane_diffs: list[ParityDiff] = []
        exp_diffs: list[ParityDiff] = []
        shape_diffs: list[ParityDiff] = []
        launch_diffs: list[ParityDiff] = []

        if not predicted_nodes:
            launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=world_ranks[0],
                node_id="predicted_graph",
                detail="empty predicted graph: independent predictions required for parity",
                expected=">=1 predicted nodes per participating rank",
                observed="0",
            ))

        launch_diffs.extend(_validate_observed_rank_headers(
            observed_dags,
            by_rank_pred,
            has_predictions=bool(predicted_nodes),
            verification_mode=verification_mode,
            min_attribution_rate=min_attribution_rate,
        ))

        for pred_rank, nodes in by_rank_pred.items():
            if pred_rank not in by_rank_obs:
                for pnode in nodes:
                    comm_diffs.append(ParityDiff(
                        code="WRONG_RANK_COLLECTIVE",
                        rank_id=pred_rank,
                        node_id=pnode.node_id,
                        detail=f"predicted rank {pred_rank} absent from observed ranks {world_ranks}",
                        expected=str(pred_rank),
                        observed=str(world_ranks),
                    ))
                continue
            d1, d2, d3, d4, d5, d6 = _check_rank_nodes(
                nodes, by_rank_obs[pred_rank], device_arch=device_arch, registry=self._registry,
            )
            dag_diffs.extend(d1)
            comm_diffs.extend(d2)
            lane_diffs.extend(d3)
            exp_diffs.extend(d4)
            shape_diffs.extend(d5)
            launch_diffs.extend(d6)

        has_diffs = bool(
            dag_diffs or comm_diffs or lane_diffs or exp_diffs
            or shape_diffs or launch_diffs or missing_so or missing_py
        )
        all_ranks_executed = all(d.npu_device_kernel_count > 0 for d in observed_dags)
        passed = (total_npu > 0) and all_ranks_executed and not has_diffs
        causal_certified = (
            passed
            and verification_mode == "causal_dag_parity"
            and all(not d.is_offline_summary for d in observed_dags)
            and not all_ambiguous
            and not all_unattributed
        )
        return LiveSubgraphParityReport(
            signature_id=signature_id,
            hardware_sku=hardware_sku,
            config_digest=config_digest,
            world_ranks=world_ranks,
            phase=phase,
            exec_mode=exec_mode,
            vllm_ascend_rev=vllm_ascend_rev,
            sim_rev=sim_rev,
            npu_device_kernel_count=total_npu,
            exact_causal_attribution_rate=weighted_attr,
            ambiguous_matches=all_ambiguous,
            unattributed_device_kernels=all_unattributed,
            missing_so_symbols=missing_so,
            missing_python_modules=missing_py,
            dag_order_diffs=tuple(dag_diffs),
            rank_collective_diffs=tuple(comm_diffs),
            stream_lane_diffs=tuple(lane_diffs),
            expansion_diffs=tuple(exp_diffs),
            shape_diffs=tuple(shape_diffs),
            launch_diffs=tuple(launch_diffs),
            passed=passed,
            verification_mode=verification_mode,
            causal_parity_certified=causal_certified,
        )


def _infer_offline_task_type(kname: str) -> str:
    if kname.startswith("hcom_"):
        return "COMMUNICATION"
    if kname.endswith(("Metadata", "AicpuKernel")) or kname == "SearchSorted":
        return AI_CPU_TASK_TYPE
    if any(tok in kname for tok in ("Attention", "Indexer", "FlashMla", "MlaProlog", "KeyPool", "ChunkKda", "ChunkGatedDelta", "ChunkFwd")):
        return "MIX_AIC"
    if any(tok in kname for tok in ("MoeDistribute", "ClippedSwiglu", "AttnResFwd")):
        return "MIX_AIV"
    if any(tok in kname for tok in ("MatMul", "Matmul", "GroupedMatmul", "BatchMatMul")):
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def load_offline_a3_capture_dags(
    capture_path: Path,
) -> tuple[str, dict[str, ObservedRankDAG], dict[str, tuple[tuple[str, ...], tuple[str, ...]]]]:
    """Parse sanitized A3 histogram capture JSON as honest 1-step ``offline_summary_inventory`` DAGs."""
    raw = json.loads(Path(capture_path).read_text(encoding="utf-8"))
    env = raw.get("environment", {})
    va_rev = str(env.get("vllm_ascend_commit", ""))
    dags: dict[str, ObservedRankDAG] = {}

    for tag, entry in (raw.get("captures") or {}).items():
        cfg = entry.get("config") or {}
        exec_mode: ExecModeKind = "aclgraph" if cfg.get("aclgraph") else "eager"
        rep = entry.get("representative_decode_step") or entry.get("representative_prefill_step") or {}
        phase: PhaseKind = "decode" if entry.get("representative_decode_step") else "prefill"
        sig_meta: dict[str, tuple[str, str, float]] = {}
        sig_rows: list[tuple[str, str, str, int, float]] = []
        for sig in entry.get("top_op_signatures") or ():
            parts = str(sig.get("signature", "")).split("|")
            kname = parts[0] if parts else ""
            cann_api = parts[1] if len(parts) > 1 else ""
            ttype = _normalize_task_type(parts[2] if len(parts) > 2 else "")
            cnt = int(sig.get("count", 1))
            dur = float(sig.get("total_dur_us", 0.0))
            if kname:
                sig_rows.append((kname, cann_api, ttype, cnt, dur))
                sig_meta.setdefault(kname, (cann_api, ttype, dur))

        nodes: list[ObservedKernelNode] = []
        idx = 0
        rep_by_type = rep.get("by_type") or {}
        if rep_by_type:
            # Prefer exact 1-step representative step histogram (enriched with CANN/task_type metadata).
            for kname, cnt in rep_by_type.items():
                cann_api, ttype, dur = sig_meta.get(kname, ("", _infer_offline_task_type(kname), 0.0))
                nodes.append(ObservedKernelNode(
                    node_id=f"a3.{tag}.rep{idx}",
                    rank_id=0,
                    stream_id="0",
                    stream_lane=_infer_stream_lane("0", kname, ttype),
                    task_type=ttype,
                    kernel_name=kname,
                    cann_api=cann_api,
                    dur_us=dur,
                    attribution_kind="offline_summary_inventory",
                    count=int(cnt),
                ))
                idx += 1
        else:
            for kname, cann_api, ttype, cnt, dur in sig_rows:
                nodes.append(ObservedKernelNode(
                    node_id=f"a3.{tag}.k{idx}",
                    rank_id=0,
                    stream_id="0",
                    stream_lane=_infer_stream_lane("0", kname, ttype),
                    task_type=ttype,
                    kernel_name=kname,
                    cann_api=cann_api,
                    dur_us=dur,
                    attribution_kind="offline_summary_inventory",
                    count=cnt,
                ))
                idx += 1
        tt_counts = {
            _normalize_task_type(k): int(v)
            for k, v in (entry.get("task_type_counts") or {}).items()
        }
        # Histogram summaries contain no flow_id edges; report 0.0 flow attribution honestly.
        dags[tag] = ObservedRankDAG(
            rank_id=0,
            phase=phase,
            exec_mode=exec_mode,
            nodes=tuple(nodes),
            exact_causal_attribution_rate=0.0,
            ambiguous_matches=(),
            unattributed_device_kernels=(),
            npu_device_kernel_count=int(entry.get("real_npu_kernels", 0)),
            ai_cpu_kernel_count=int(tt_counts.get(AI_CPU_TASK_TYPE, 0)),
            task_type_counts=tuple(sorted(tt_counts.items())),
            is_offline_summary=True,
        )

    blocked = {
        tag: classify_runtime_errors((str(msg),))
        for tag, msg in (raw.get("runtime_blocked_in_container") or {}).items()
    }
    return va_rev, dags, blocked


def load_offline_a5_capture_dags(
    capture_path: Path,
) -> tuple[str, dict[str, ObservedRankDAG]]:
    """Parse sanitized A5 sequence-summary capture JSON as ``offline_summary_inventory`` DAGs."""
    raw = json.loads(Path(capture_path).read_text(encoding="utf-8"))
    va_rev = str(raw.get("vllm_ascend_commit", ""))
    dags: dict[str, ObservedRankDAG] = {}

    for tag, entry in (raw.get("captures") or {}).items():
        exec_mode: ExecModeKind = "aclgraph" if "aclgraph" in tag else "eager"
        seq = entry.get("decode_op_sequence_summary") or entry.get("prefill_op_sequence_summary") or ()
        phase: PhaseKind = "decode" if entry.get("decode_op_sequence_summary") else "prefill"
        nodes: list[ObservedKernelNode] = []
        prev_id = ""
        for idx, item in enumerate(seq):
            kname = str(item.get("kernel", "")).strip()
            ttype = _normalize_task_type(str(item.get("core", "")))
            nid = f"a5.{tag}.k{idx}"
            nodes.append(ObservedKernelNode(
                node_id=nid,
                rank_id=0,
                stream_id="0",
                stream_lane=_infer_stream_lane("0", kname, ttype),
                task_type=ttype,
                kernel_name=kname,
                cann_api=str(item.get("cann_api", "")),
                cpu_op=str(item.get("cpu_op", "")),
                vllm_scope=str(item.get("vllm_scope", "")),
                input_shapes=str(item.get("sample_in_shapes", "")),
                input_dtypes=str(item.get("sample_in_dtypes", "")),
                dur_us=float(item.get("total_us", 0.0)),
                attribution_kind="offline_summary_inventory",
                predecessors=(prev_id,) if prev_id else (),
                count=int(item.get("count", 1)),
            ))
            prev_id = nid
        # Offline sequence summaries carry aggregate coverage, NEVER per-launch causal flow joins.
        core_counts = {
            _normalize_task_type(k): int(v)
            for k, v in (entry.get("core_counts") or {}).items()
        }
        dags[tag] = ObservedRankDAG(
            rank_id=0,
            phase=phase,
            exec_mode=exec_mode,
            nodes=tuple(nodes),
            exact_causal_attribution_rate=0.0,
            ambiguous_matches=(),
            unattributed_device_kernels=(),
            npu_device_kernel_count=int(entry.get("real_npu_kernels", 0)),
            ai_cpu_kernel_count=int(core_counts.get(AI_CPU_TASK_TYPE, 0)),
            task_type_counts=tuple(sorted(core_counts.items())),
            is_offline_summary=True,
        )
    return va_rev, dags


class _JsonStreamReader:
    """Chunked 1 MiB JSON scanner validating depth-1 ``traceEvents`` and full-document EOF."""

    __slots__ = ("_stream", "_name", "_buf", "_pos", "_eof", "_decoder")

    def __init__(self, stream: Any, name: str) -> None:
        self._stream = stream
        self._name = name
        self._buf = ""
        self._pos = 0
        self._eof = False
        self._decoder = json.JSONDecoder()

    def _refill(self) -> bool:
        if self._eof:
            return False
        chunk = self._stream.read(_STREAM_CHUNK_BYTES)
        if not chunk:
            self._eof = True
            return False
        if self._pos > 0:
            self._buf = self._buf[self._pos:] + chunk
            self._pos = 0
        else:
            self._buf += chunk
        return True

    def _compact(self) -> None:
        if self._pos >= _STREAM_CHUNK_BYTES:
            self._buf = self._buf[self._pos:]
            self._pos = 0

    def skip_ws(self) -> str:
        while True:
            while self._pos < len(self._buf) and self._buf[self._pos] in " \t\r\n":
                self._pos += 1
            if self._pos < len(self._buf):
                self._compact()
                return self._buf[self._pos]
            if not self._refill():
                return ""

    def next_char(self) -> str:
        ch = self.skip_ws()
        if not ch:
            raise ModelingError(f"trace_view_json_truncated:{self._name}")
        self._pos += 1
        return ch

    def expect_char(self, expected: str) -> None:
        ch = self.next_char()
        if ch != expected:
            raise ModelingError(f"trace_view_json_invalid:{self._name}:expected_{expected}_got_{ch}")

    def decode_value(self) -> Any:
        ch = self.skip_ws()
        if not ch:
            raise ModelingError(f"trace_view_json_truncated:{self._name}")
        while True:
            try:
                obj, next_pos = self._decoder.raw_decode(self._buf, self._pos)
                self._pos = next_pos
                self._compact()
                return obj
            except json.JSONDecodeError as exc:
                if self._eof or len(self._buf) - self._pos > _MAX_SINGLE_EVENT_BYTES:
                    code = "trace_view_event_too_large" if len(self._buf) - self._pos > _MAX_SINGLE_EVENT_BYTES else "trace_view_json_truncated"
                    raise ModelingError(f"{code}:{self._name}") from exc
                if not self._refill():
                    raise ModelingError(f"trace_view_json_truncated:{self._name}") from exc

    def _skip_string_body(self) -> None:
        escaped = False
        hex_remaining = 0
        while True:
            while self._pos < len(self._buf):
                ch = self._buf[self._pos]
                self._pos += 1
                if hex_remaining > 0:
                    if ch not in "0123456789abcdefABCDEF":
                        raise ModelingError(f"trace_view_json_invalid:{self._name}:bad_unicode_escape")
                    hex_remaining -= 1
                elif escaped:
                    if ch == "u":
                        hex_remaining = 4
                    elif ch not in '"\\/bfnrt':
                        raise ModelingError(f"trace_view_json_invalid:{self._name}:bad_escape_{ch}")
                    escaped = False
                elif ch == "\\":
                    escaped = True
                elif ch == '"':
                    self._compact()
                    return
                elif ord(ch) < 0x20:
                    raise ModelingError(f"trace_view_json_invalid:{self._name}:control_char_in_string")
            if not self._refill():
                raise ModelingError(f"trace_view_json_truncated:{self._name}")

    def _skip_array(self, depth: int) -> None:
        self.expect_char("[")
        first = True
        while True:
            ch = self.skip_ws()
            if not ch:
                raise ModelingError(f"trace_view_json_truncated:{self._name}")
            if ch == "]":
                self._pos += 1
                self._compact()
                return
            if not first:
                self.expect_char(",")
            first = False
            self.skip_value(depth + 1)

    def _skip_object(self, depth: int) -> None:
        self.expect_char("{")
        first = True
        while True:
            ch = self.skip_ws()
            if not ch:
                raise ModelingError(f"trace_view_json_truncated:{self._name}")
            if ch == "}":
                self._pos += 1
                self._compact()
                return
            if not first:
                self.expect_char(",")
            first = False
            self.expect_char('"')
            self._skip_string_body()
            self.expect_char(":")
            self.skip_value(depth + 1)

    def skip_value(self, depth: int = 0) -> None:
        """Strictly validate and skip any JSON value without materializing containers."""
        if depth > _MAX_SKIP_DEPTH:
            raise ModelingError(f"trace_view_json_invalid:{self._name}:nesting_too_deep")
        ch = self.skip_ws()
        if not ch:
            raise ModelingError(f"trace_view_json_truncated:{self._name}")
        if ch == '"':
            self._pos += 1
            self._skip_string_body()
            return
        if ch == "[":
            self._skip_array(depth)
            return
        if ch == "{":
            self._skip_object(depth)
            return
        self.decode_value()

    def iter_array_elements(self) -> Iterator[Mapping[str, Any]]:
        self.expect_char("[")
        first = True
        while True:
            ch = self.skip_ws()
            if not ch:
                raise ModelingError(f"trace_view_json_truncated:{self._name}")
            if ch == "]":
                self._pos += 1
                return
            if not first:
                if ch != ",":
                    raise ModelingError(f"trace_view_json_invalid:{self._name}:expected_comma_or_bracket")
                self._pos += 1
            first = False
            obj = self.decode_value()
            if isinstance(obj, Mapping):
                yield obj

    def assert_eof(self) -> None:
        ch = self.skip_ws()
        if ch:
            raise ModelingError(f"trace_view_json_trailing_content:{self._name}")


def iter_trace_view_events(trace_path: Path) -> Iterator[Mapping[str, Any]]:
    """Stream Chrome-trace JSON events in 1 MiB chunks with strict depth-1 & EOF validation."""
    path = Path(trace_path)
    with path.open("r", encoding="utf-8") as stream:
        reader = _JsonStreamReader(stream, path.name)
        first = reader.skip_ws()
        if not first:
            raise ModelingError(f"trace_view_events_list_required:{path.name}")
        if first == "[":
            yield from reader.iter_array_elements()
            reader.assert_eof()
            return
        if first != "{":
            raise ModelingError(f"trace_view_events_list_required:{path.name}")

        reader.expect_char("{")
        found_events = False
        first_key = True
        while True:
            ch = reader.skip_ws()
            if not ch:
                raise ModelingError(f"trace_view_json_truncated:{path.name}")
            if ch == "}":
                reader.expect_char("}")
                break
            if not first_key:
                reader.expect_char(",")
            first_key = False
            key = reader.decode_value()
            if not isinstance(key, str):
                raise ModelingError(f"trace_view_json_invalid:{path.name}:non_string_key")
            reader.expect_char(":")
            if key == "traceEvents":
                if found_events:
                    raise ModelingError(f"trace_view_json_invalid:{path.name}:duplicate_traceEvents")
                found_events = True
                yield from reader.iter_array_elements()
            else:
                reader.skip_value()
        if not found_events:
            raise ModelingError(f"trace_view_events_list_required:{path.name}")
        reader.assert_eof()


def load_trace_view_file(
    trace_path: Path,
    *,
    rank_id: int = 0,
    phase: PhaseKind = "decode",
    exec_mode: ExecModeKind = "eager",
) -> ObservedRankDAG:
    """Stream and causally correlate an Ascend ``trace_view.json`` file with bounded memory."""
    return correlate_trace_view_events(
        iter_trace_view_events(trace_path),
        rank_id=rank_id,
        phase=phase,
        exec_mode=exec_mode,
    )


@dataclass(frozen=True, slots=True)
class StreamedRankParityOutcome:
    """Compact per-rank parity summary produced without retaining full trace node lists."""

    rank_id: int
    total_events: int
    exact_count: int
    npu_device_kernel_count: int
    ai_cpu_kernel_count: int
    exact_causal_attribution_rate: float
    ambiguous_matches: tuple[str, ...]
    unattributed_device_kernels: tuple[str, ...]
    dag_diffs: tuple[ParityDiff, ...]
    comm_diffs: tuple[ParityDiff, ...]
    lane_diffs: tuple[ParityDiff, ...]
    exp_diffs: tuple[ParityDiff, ...]
    shape_diffs: tuple[ParityDiff, ...]
    launch_diffs: tuple[ParityDiff, ...]


def _matches_any_needed_token(kname: str, needed_tokens: set[str]) -> bool:
    if not kname or not needed_tokens:
        return False
    for tok in needed_tokens:
        if kname == tok or (tok.endswith("_") and kname.startswith(tok)) or kname.startswith(tok + "_"):
            return True
    return False


def evaluate_trace_view_stream(
    trace_path: Path,
    pred_nodes: Sequence[PredictedKernelNode],
    *,
    rank_id: int,
    phase: PhaseKind,
    exec_mode: ExecModeKind,
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> StreamedRankParityOutcome:
    """Verify one rank's ``trace_view.json`` via 2-pass bounded streaming (100% late-flow safe).

    Pass 1 streams non-kernel flow/ACLGraph/host events into ``_CausalFlowState`` without
    retaining device launches; Pass 2 streams device kernel launches against the completed
    causal table so late-listed flow records receive 100% exact attribution in bounded memory.
    """
    state = _CausalFlowState(
        rank_id=rank_id, exec_mode=exec_mode, max_consumed_ring=_STREAM_WINDOW_FLUSH * 8,
    )
    expected_sets: list[tuple[str, ...]] = [
        _expected_expansion_for(pred, device_arch, registry) for pred in pred_nodes
    ]
    needed_tokens: set[str] = {tok for exp in expected_sets for tok in exp}
    needed_budget = sum(max(1, len(exp), pred.min_count) for pred, exp in zip(pred_nodes, expected_sets))
    retain_limit = max(_MAX_REPORTED_SAMPLES, needed_budget * 4)

    # Pass 1: ingest every non-kernel flow, ACLGraph node, and host timestamp record.
    for ev in iter_trace_view_events(trace_path):
        ph = str(ev.get("ph", ""))
        cat = str(ev.get("cat", "")).lower()
        args = ev.get("args") if isinstance(ev.get("args"), Mapping) else {}
        ev_rank_raw = args.get("rank_id")
        if ev_rank_raw is not None and int(ev_rank_raw) != rank_id:
            continue
        if not (ph == "X" and _is_device_kernel_event(ev, cat, args)):
            state.ingest_non_kernel(ev, ph, cat, args)

    retained_nodes: list[ObservedKernelNode] = []
    ambiguous: list[str] = []
    unattributed: list[str] = []
    extra_launch_diffs: list[ParityDiff] = []
    total_events = 0
    exact_count = 0
    npu_count = 0
    aicpu_count = 0

    # Pass 2: stream device launches against completed causal flow state.
    for ev in iter_trace_view_events(trace_path):
        ph = str(ev.get("ph", ""))
        cat = str(ev.get("cat", "")).lower()
        args = ev.get("args") if isinstance(ev.get("args"), Mapping) else {}
        ev_rank_raw = args.get("rank_id")
        if ev_rank_raw is not None and int(ev_rank_raw) != rank_id:
            continue
        if not (ph == "X" and _is_device_kernel_event(ev, cat, args)):
            continue
        dev = _extract_device_launch(ev)
        idx = total_events
        total_events += 1
        if dev.task_type in REAL_NPU_TASK_TYPES:
            npu_count += 1
        if dev.task_type == AI_CPU_TASK_TYPE:
            aicpu_count += 1
        attribution, resolved_fid, cann_api, cpu_op, vllm_scope = state.attribute(dev)
        if attribution in {"exact_flow_id", "aclgraph_node_table"}:
            exact_count += 1
        elif attribution == "ambiguous_proximity":
            if len(ambiguous) < _MAX_REPORTED_SAMPLES:
                ambiguous.append(f"r{rank_id}:{idx}:{dev.kernel}")
        else:
            if len(unattributed) < _MAX_REPORTED_SAMPLES:
                unattributed.append(f"r{rank_id}:{idx}:{dev.kernel}")

        if _matches_any_needed_token(dev.kernel, needed_tokens) and len(retained_nodes) < retain_limit:
            retained_nodes.append(ObservedKernelNode(
                node_id=f"r{rank_id}.s{dev.stream_id}.k{idx}",
                rank_id=rank_id,
                stream_id=dev.stream_id,
                stream_lane=dev.lane,
                task_type=dev.task_type,
                kernel_name=dev.kernel,
                cann_api=cann_api,
                cpu_op=cpu_op,
                vllm_scope=vllm_scope,
                input_shapes=dev.input_shapes,
                input_dtypes=dev.input_dtypes,
                dur_us=dev.dur,
                start_us=dev.ts,
                flow_id=resolved_fid,
                attribution_kind=attribution,
                collective_group=dev.collective_group,
                comm_ranks=dev.comm_ranks,
                count=1,
            ))
        elif len(extra_launch_diffs) < _MAX_REPORTED_SAMPLES:
            extra_launch_diffs.append(ParityDiff(
                code="MISSING_OR_EXTRA_LAUNCH",
                rank_id=rank_id,
                node_id=f"r{rank_id}.s{dev.stream_id}.k{idx}",
                detail=f"extra unpredicted device launch {dev.kernel} ({dev.task_type})",
                expected="0 extra launches",
                observed=dev.kernel,
            ))

    retained_nodes.sort(key=lambda item: (item.start_us, item.node_id))
    rate = (exact_count / total_events) if total_events > 0 else 0.0
    compact_dag = ObservedRankDAG(
        rank_id=rank_id,
        phase=phase,
        exec_mode=exec_mode,
        nodes=tuple(retained_nodes),
        exact_causal_attribution_rate=rate,
        ambiguous_matches=tuple(ambiguous),
        unattributed_device_kernels=tuple(unattributed),
        npu_device_kernel_count=npu_count,
        ai_cpu_kernel_count=aicpu_count,
        is_offline_summary=False,
    )
    if pred_nodes:
        d1, d2, d3, d4, d5, d6 = _check_rank_nodes(
            pred_nodes, compact_dag, device_arch=device_arch, registry=registry,
        )
    else:
        d1, d2, d3, d4, d5, d6 = [], [], [], [], [], []
    d6.extend(extra_launch_diffs)
    return StreamedRankParityOutcome(
        rank_id=rank_id,
        total_events=total_events,
        exact_count=exact_count,
        npu_device_kernel_count=npu_count,
        ai_cpu_kernel_count=aicpu_count,
        exact_causal_attribution_rate=rate,
        ambiguous_matches=tuple(ambiguous),
        unattributed_device_kernels=tuple(unattributed),
        dag_diffs=tuple(d1),
        comm_diffs=tuple(d2),
        lane_diffs=tuple(d3),
        exp_diffs=tuple(d4),
        shape_diffs=tuple(d5),
        launch_diffs=tuple(d6),
    )


__all__ = [
    "AI_CPU_TASK_TYPE",
    "AttributionKind",
    "DEFAULT_MIN_ATTRIBUTION_RATE",
    "LiveSubgraphParityReport",
    "ObservedKernelNode",
    "ObservedRankDAG",
    "PROXIMITY_AMBIGUITY_WINDOW_US",
    "ParityDiff",
    "ParityDiffCode",
    "PredictedKernelNode",
    "REAL_NPU_TASK_TYPES",
    "StreamedRankParityOutcome",
    "SubgraphParityChecker",
    "VerificationMode",
    "classify_runtime_errors",
    "correlate_trace_view_events",
    "evaluate_trace_view_stream",
    "iter_trace_view_events",
    "load_offline_a3_capture_dags",
    "load_offline_a5_capture_dags",
    "load_trace_view_file",
]
