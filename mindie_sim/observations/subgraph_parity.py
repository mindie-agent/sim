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
[Side-Effect Policy] Read-only file parsing returning immutable report dataclasses;
  CLI entrypoint prints JSON or writes an explicitly requested ``--output`` file.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Literal, TypeAlias

from mindie_sim.backends.abi_registry import ProbedAbiRegistry, default_abi_registry
from mindie_sim.compose import compose_case, resolve_hardware_sku
from mindie_sim.prediction.collectives.topology import Topology
from mindie_sim.prediction.results import LoweringRequest, StepRequest
from mindie_sim.prediction.spine import PredictionSpine, PreparedStep
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from mindie_sim.structure_implementations import operator_birth_certificate

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
_NS_PER_US_INT = 1000
_NS_PER_US = Decimal(_NS_PER_US_INT)
_PROXIMITY_AMBIGUITY_WINDOW_NS = int(PROXIMITY_AMBIGUITY_WINDOW_US * _NS_PER_US_INT)
_STREAM_CHUNK_BYTES = 1024 * 1024
_MAX_SINGLE_EVENT_BYTES = 8 * 1024 * 1024
_MAX_SKIP_DEPTH = 128
_MAX_REPORTED_SAMPLES = 64
_STREAM_WINDOW_FLUSH = 256
_GDN_CHUNK_SIZE = 64
_GDN_PREPARE_ARITY = 7
_RANK_3D = 3
_RANK_4D = 4
_NZ_R0 = 16
_NZ_C0_1B = 32
_NZ_C0_2B = 16
_GROUPED_MATMUL_FIXED_SLOTS = 8
_GROUPED_MATMUL_BIAS_SLOT = 2
_GROUPED_MATMUL_WEIGHT_SCALE_SLOT = 3
_GROUPED_MATMUL_GROUP_LIST_SLOT = 7
_ONE_BYTE_DTYPES: frozenset[str] = frozenset({
    "int8", "uint8", "s8", "u8", "fp8", "fp8_e4m3", "fp8_e4m3fn", "fp8_e5m2", "hif8", "hifloat8", "mxfp8",
})
_GEMM_OP_FAMILIES: frozenset[str] = frozenset({
    "qkv_projection",
    "output_projection",
    "gate_up_projection",
    "down_projection",
    "dense_gemm",
    "gemm",
    "grouped_gemm",
    "linear",
    "router",
})
_GDN_EXPANSION_KERNELS: tuple[str, str, str] = (
    "ChunkGatedDeltaRuleFwdPrepare",
    "ChunkFwdH",
    "ChunkFwdO",
)
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
    start_ns: int | None = None
    dur_ns: int | None = None


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
    expected_input_shapes: tuple[str, ...] = ()


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


def _parse_us_to_ns(value: Any) -> int | None:
    """Convert microsecond capture timestamp/duration to exact integer nanoseconds."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        dec = value
    elif isinstance(value, int):
        dec = Decimal(value)
    elif isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        dec = Decimal(stripped)
    else:
        dec = Decimal(str(value))
    return int((dec * _NS_PER_US).to_integral_value(rounding=ROUND_HALF_EVEN))


def _node_time_bounds_ns(
    node: ObservedKernelNode,
    *,
    dag_has_timing: bool = False,
) -> tuple[int, int] | None:
    """Return ``(start_ns, end_ns)`` when timing was recorded, else ``None``."""
    if node.start_ns is not None:
        dur_ns = max(0, node.dur_ns) if node.dur_ns is not None else max(0, _parse_us_to_ns(node.dur_us) or 0)
        return node.start_ns, node.start_ns + dur_ns
    if dag_has_timing or node.start_us != 0.0 or node.dur_us != 0.0:
        start_ns = _parse_us_to_ns(node.start_us) or 0
        dur_ns = max(0, _parse_us_to_ns(node.dur_us) or 0)
        return start_ns, start_ns + dur_ns
    return None


def _dag_has_timing(obs_dag: ObservedRankDAG) -> bool:
    if obs_dag.is_offline_summary:
        return False
    return any(
        n.start_ns is not None or n.start_us != 0.0 or n.dur_us != 0.0
        for n in obs_dag.nodes
    )


def _format_ns_as_us(ns: int) -> str:
    """Format exact integer nanoseconds as microsecond string with 3 fractional digits."""
    sign = "-" if ns < 0 else ""
    whole, frac = divmod(abs(ns), _NS_PER_US_INT)
    return f"{sign}{whole}.{frac:03d}"


@dataclass(frozen=True, slots=True)
class _FlowRecord:
    ts_ns: int | None
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
    ts_ns: int | None
    dur_ns: int | None
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
        ts_ns=_parse_us_to_ns(ev.get("ts")),
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
    raw_ts = ev.get("ts")
    raw_dur = ev.get("dur")
    ts_ns = _parse_us_to_ns(raw_ts)
    dur_ns = _parse_us_to_ns(raw_dur)
    if ts_ns is not None and dur_ns is None:
        dur_ns = 0
    return _RawDeviceLaunch(
        kernel=kernel,
        stream_id=stream_id,
        task_type=task_type,
        fid=str(args.get("flow_id", ev.get("flow_id", ""))).strip(),
        gid=str(args.get("graph_node_id", "")).strip(),
        ts=float(raw_ts) if ts_ns is not None else 0.0,
        dur=float(raw_dur) if raw_dur is not None and dur_ns is not None else 0.0,
        ts_ns=ts_ns,
        dur_ns=dur_ns,
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


def _near_sorted_ns(sorted_ns: Sequence[int], target: int, window_ns: int) -> bool:
    if not sorted_ns:
        return False
    idx = bisect_left(sorted_ns, target - window_ns)
    return idx < len(sorted_ns) and sorted_ns[idx] <= target + window_ns


class _CausalFlowState:
    """Bounded causal join table supporting exact (rank, stream, flow_id), Chrome s/f, and ACLGraph."""

    __slots__ = (
        "rank_id", "exec_mode", "stream_flows", "unbound_flows", "unbound_tid",
        "finish_by_ts", "finish_by_id", "graph_nodes",
        "consumed_flows", "_consumed_ring", "_max_consumed_ring",
        "host_ts_ns", "host_sorted",
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
        # (stream_id, ts_ns) -> fid or None (Chrome 'f' endpoint; duplicate poisoned)
        self.finish_by_ts: dict[tuple[str, int], str | None] = {}
        # (stream_id, fid) -> dest_ts_ns or None (Chrome 'f' endpoint)
        self.finish_by_id: dict[tuple[str, str], int | None] = {}
        # (stream_id, gid) -> _FlowRecord or None (used ONLY when exec_mode == "aclgraph")
        self.graph_nodes: dict[tuple[str, str], _FlowRecord | None] = {}
        self.consumed_flows: set[tuple[str, str]] = set()
        self._consumed_ring: deque[tuple[str, str]] = deque()
        self._max_consumed_ring = max(0, int(max_consumed_ring))
        self.host_ts_ns: list[int] = []
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
            dest_ts_ns = _parse_us_to_ns(ev.get("ts"))
            if not fid or not dest_sid or dest_ts_ns is None:
                return
            ts_key = (dest_sid, dest_ts_ns)
            self.finish_by_ts[ts_key] = None if ts_key in self.finish_by_ts else fid
            id_key = (dest_sid, fid)
            self.finish_by_id[id_key] = None if id_key in self.finish_by_id else dest_ts_ns
        elif ph == "X" and cat in _ACLGRAPH_CATEGORIES and self.exec_mode == "aclgraph":
            gid = str(args.get("graph_node_id", ev.get("id", ""))).strip()
            sid = _intern(str(args.get("stream_id", ev.get("tid", "0"))).strip())
            if gid and sid:
                gkey = (sid, gid)
                self.graph_nodes[gkey] = None if gkey in self.graph_nodes else _extract_host_record(ev, explicit_stream=sid)
        elif ph == "X" and cat in _HOST_OP_CATEGORIES:
            ts_ns = _parse_us_to_ns(ev.get("ts"))
            if ts_ns is not None:
                if self.host_ts_ns and ts_ns < self.host_ts_ns[-1]:
                    self.host_sorted = False
                self.host_ts_ns.append(ts_ns)

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
                f_ts_ns = self.finish_by_id.pop(fkey)
                if f_ts_ns is None or dev.ts_ns is None or f_ts_ns != dev.ts_ns:
                    return None, dev.fid, True
                self.finish_by_ts.pop((sid, f_ts_ns), None)
                if rec is None:
                    u_rec = self.unbound_flows.get(dev.fid)
                    if u_rec is None:
                        return None, dev.fid, True
                    rec = u_rec
            if rec is not None:
                if rec.ts_ns is not None and dev.ts_ns is not None and rec.ts_ns > dev.ts_ns:
                    return None, dev.fid, True
                self._reclaim_flow(fkey, dev.fid)
                return rec, dev.fid, False
            return None, dev.fid, False

        # Path B: standard Chrome 's' -> 'f' ending on (dev.stream_id, dev.ts_ns).
        if dev.ts_ns is None:
            return None, "", False
        ts_key = (sid, dev.ts_ns)
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
        if rec.ts_ns is not None and rec.ts_ns > dev.ts_ns:
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
            self.host_ts_ns.sort()
            self.host_sorted = True
        if dev.ts_ns is not None and _near_sorted_ns(self.host_ts_ns, dev.ts_ns, _PROXIMITY_AMBIGUITY_WINDOW_NS):
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

    device_launches.sort(key=lambda item: (item.ts_ns is None, item.ts_ns or 0))
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
            start_ns=dev.ts_ns,
            dur_ns=dev.dur_ns,
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


def _parse_operand_shapes(text: str) -> tuple[tuple[int, ...], ...] | None:
    """Parse profiler/predicted semicolon-separated operand shapes into normalized int tuples."""
    cleaned = str(text or "").strip().strip('"').strip("'").strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    operands: list[tuple[int, ...]] = []
    for raw_slot in cleaned.split(";"):
        slot = raw_slot.strip().strip("()[]").strip()
        if not slot:
            operands.append(())
            continue
        dims: list[int] = []
        for token in slot.split(","):
            tok = token.strip()
            if not tok:
                continue
            try:
                dims.append(int(tok))
            except ValueError:
                return None
        operands.append(tuple(dims))
    return tuple(operands) if operands else None


def _format_operand_shapes(operands: Sequence[Sequence[int]]) -> str:
    return ";".join(",".join(str(int(d)) for d in shp) for shp in operands)


def _shapes_match(expected: str, observed: str) -> bool:
    exp_clean = str(expected or "").strip().strip('"').strip("'").strip()
    obs_clean = str(observed or "").strip().strip('"').strip("'").strip()
    if not exp_clean or not obs_clean or obs_clean.upper() == "N/A":
        return False
    exp_parsed = _parse_operand_shapes(exp_clean)
    obs_parsed = _parse_operand_shapes(obs_clean)
    if exp_parsed is not None and obs_parsed is not None:
        return exp_parsed == obs_parsed
    return exp_clean == obs_clean


def _derive_expansion_stage_shapes(
    expected_kernels: tuple[str, ...],
    entry_shapes: str,
) -> tuple[str, ...]:
    """Derive per-stage operand shape descriptors for known multi-kernel expansions."""
    if expected_kernels == _PAD_MULTI_KERNEL:
        return ("", entry_shapes)
    if expected_kernels == _GDN_EXPANSION_KERNELS:
        parsed = _parse_operand_shapes(entry_shapes)
        if parsed is not None and len(parsed) == _GDN_PREPARE_ARITY:
            q, k, v, g, beta, cu_seq, chunk_idx = parsed
            if (
                len(q) == _RANK_4D
                and len(k) == _RANK_4D
                and len(v) == _RANK_4D
                and len(g) == _RANK_3D
                and len(beta) == _RANK_3D
            ):
                b_dim, _, t_dim, k_dim = q
                hv_dim, v_dim = v[1], v[_RANK_3D]
                seq_num = (
                    cu_seq[0] - 1
                    if len(cu_seq) == 1 and cu_seq[0] > 1
                    else b_dim
                )
                num_chunks = (
                    chunk_idx[0] // 2
                    if len(chunk_idx) == 1 and chunk_idx[0] > 0 and chunk_idx[0] % 2 == 0
                    else (t_dim + _GDN_CHUNK_SIZE - 1) // _GDN_CHUNK_SIZE
                )
                stage_h = _format_operand_shapes((
                    k,
                    (b_dim, hv_dim, t_dim, k_dim),
                    (b_dim, hv_dim, t_dim, v_dim),
                    g,
                    (seq_num, hv_dim, k_dim, v_dim),
                    cu_seq,
                    chunk_idx,
                ))
                stage_o = _format_operand_shapes((
                    q,
                    k,
                    (b_dim, hv_dim, t_dim, v_dim),
                    (b_dim, hv_dim, num_chunks, k_dim, v_dim),
                    g,
                    cu_seq,
                    chunk_idx,
                ))
                return (entry_shapes, stage_h, stage_o)
    return (entry_shapes,) + ("",) * (len(expected_kernels) - 1)


def _resolve_stage_expected_shapes(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
) -> tuple[str, ...]:
    stage_count = len(expected_kernels)
    if pred.expected_input_shapes:
        return tuple(
            pred.expected_input_shapes[i] if i < len(pred.expected_input_shapes) else ""
            for i in range(stage_count)
        )
    if not pred.input_shapes:
        return ("",) * stage_count
    if stage_count == 1:
        return (pred.input_shapes,)
    return _derive_expansion_stage_shapes(expected_kernels, pred.input_shapes)


def _offline_unknown_task_budget(obs_dag: ObservedRankDAG) -> dict[str, int] | None:
    """Compute remaining per-task-type capacity for offline ``UNKNOWN`` rows from ``task_type_counts``."""
    if not obs_dag.is_offline_summary or not obs_dag.task_type_counts:
        return None
    budget = {_normalize_task_type(k): int(v) for k, v in obs_dag.task_type_counts}
    for node in obs_dag.nodes:
        tt = _normalize_task_type(node.task_type)
        if tt != "UNKNOWN" and tt in budget:
            budget[tt] = max(0, budget[tt] - max(0, int(node.count)))
    return budget


def _task_type_compatible(
    pred: PredictedKernelNode,
    obs: ObservedKernelNode,
    *,
    expected_override: str = "",
    is_offline_summary: bool = False,
    unknown_task_budget: Mapping[str, int] | None = None,
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
    if is_offline_summary and observed == "UNKNOWN" and expected in REAL_NPU_TASK_TYPES:
        if unknown_task_budget is not None:
            return unknown_task_budget.get(expected, 0) > 0
        return True
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


def _stage_attrs_strict_ok(
    pred: PredictedKernelNode,
    node: ObservedKernelNode,
    exp_shape: str,
    *,
    is_offline_summary: bool,
) -> bool:
    if pred.stream_lane and node.stream_lane != pred.stream_lane:
        return False
    if pred.vllm_scope and not is_offline_summary and node.vllm_scope != pred.vllm_scope:
        return False
    if exp_shape:
        has_obs_shape = bool(node.input_shapes and node.input_shapes != "N/A")
        if is_offline_summary:
            if has_obs_shape and not _shapes_match(exp_shape, node.input_shapes):
                return False
        elif not _shapes_match(exp_shape, node.input_shapes):
            return False
    return True


def _candidate_preference_key(
    pred: PredictedKernelNode,
    item: tuple[int, ObservedKernelNode],
    exp_shape: str = "",
) -> tuple[int, int, int, int]:
    """Prefer exact ``vllm_scope``, normalized ``input_shapes``, and ``stream_lane`` before earliest pos."""
    pos, node = item
    scope_miss = 0 if (pred.vllm_scope and node.vllm_scope and node.vllm_scope == pred.vllm_scope) else (
        1 if (pred.vllm_scope and node.vllm_scope) else 0
    )
    target_shape = exp_shape or pred.input_shapes
    has_obs_shape = bool(node.input_shapes and node.input_shapes != "N/A")
    shape_miss = 0 if (
        target_shape and has_obs_shape and _shapes_match(target_shape, node.input_shapes)
    ) else (1 if (target_shape and has_obs_shape) else 0)
    lane_miss = 0 if (not pred.stream_lane or node.stream_lane == pred.stream_lane) else 1
    return (scope_miss, shape_miss, lane_miss, pos)


def _match_live_multi_from_anchor(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    stage_expected: tuple[str, ...],
    scoped_pool: Sequence[tuple[int, ObservedKernelNode]],
    anchor: tuple[int, ObservedKernelNode],
    remaining_counts: Mapping[int, int],
    *,
    dag_has_timing: bool,
    strict_attrs: bool,
) -> tuple[tuple[tuple[int, ObservedKernelNode], ...] | None, list[tuple[int, ObservedKernelNode]]]:
    a_pos, a_node = anchor
    first_tok = expected_kernels[0]
    first_tok_unique = first_tok not in expected_kernels[1:]
    ordered_picks: list[tuple[int, ObservedKernelNode]] = [(a_pos, a_node)]
    cursor = a_pos
    used_pos: set[int] = {a_pos}
    a_bounds = _node_time_bounds_ns(a_node, dag_has_timing=dag_has_timing)
    prev_end_ns: int | None = a_bounds[1] if a_bounds is not None else None
    for sub_idx in range(1, len(expected_kernels)):
        tok = expected_kernels[sub_idx]
        exp_tt = _subkernel_task_type(pred, sub_idx)
        exp_shape = stage_expected[sub_idx] if sub_idx < len(stage_expected) else ""
        found: tuple[int, ObservedKernelNode, tuple[int, int] | None] | None = None
        for pos, node in scoped_pool:
            if pos <= cursor or pos in used_pos or remaining_counts.get(pos, 0) <= 0:
                continue
            if node.stream_id != a_node.stream_id or node.vllm_scope != a_node.vllm_scope:
                continue
            if strict_attrs and first_tok_unique and _token_matches_node(node, first_tok):
                break
            bounds = _node_time_bounds_ns(node, dag_has_timing=dag_has_timing)
            if prev_end_ns is not None and bounds is not None and bounds[0] < prev_end_ns:
                continue
            if not (_token_matches_node(node, tok) and _task_type_compatible(
                pred, node, expected_override=exp_tt, is_offline_summary=False,
            )):
                continue
            if strict_attrs and not _stage_attrs_strict_ok(
                pred, node, exp_shape, is_offline_summary=False,
            ):
                continue
            found = (pos, node, bounds)
            break
        if found is None:
            return None, ordered_picks
        f_pos, f_node, f_bounds = found
        ordered_picks.append((f_pos, f_node))
        cursor = f_pos
        used_pos.add(f_pos)
        if f_bounds is not None:
            prev_end_ns = f_bounds[1]
    return tuple(ordered_picks), ordered_picks


def _try_consume_one_live_multi(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    stage_expected: tuple[str, ...],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    *,
    dag_has_timing: bool,
    strict_attrs: bool,
) -> tuple[tuple[tuple[int, ObservedKernelNode], ...] | None, list[tuple[int, ObservedKernelNode]]]:
    """Consume one complete ordered, same-call, non-overlapping live multi-kernel invocation."""
    scoped_pool = [
        (pos, node)
        for pos, node in enumerate(obs_dag.nodes)
        if remaining_counts.get(pos, 0) > 0 and _node_matches_kernel(node, expected_kernels)
    ]
    if pred.vllm_scope and any(n.vllm_scope == pred.vllm_scope for _, n in scoped_pool):
        scoped_pool = [(pos, n) for pos, n in scoped_pool if n.vllm_scope == pred.vllm_scope]
    first_tok = expected_kernels[0]
    first_tt = _subkernel_task_type(pred, 0)
    first_shape = stage_expected[0] if stage_expected else ""
    best_partial: list[tuple[int, ObservedKernelNode]] = []
    for pos, node in scoped_pool:
        if not (_token_matches_node(node, first_tok) and _task_type_compatible(
            pred, node, expected_override=first_tt, is_offline_summary=False,
        )):
            continue
        if strict_attrs and not _stage_attrs_strict_ok(
            pred, node, first_shape, is_offline_summary=False,
        ):
            continue
        matched, partial = _match_live_multi_from_anchor(
            pred, expected_kernels, stage_expected, scoped_pool, (pos, node),
            remaining_counts, dag_has_timing=dag_has_timing, strict_attrs=strict_attrs,
        )
        if matched is not None:
            for m_pos, _ in matched:
                remaining_counts[m_pos] -= 1
            return matched, list(matched)
        if len(partial) > len(best_partial):
            best_partial = partial
    return None, best_partial


def _consume_offline_multi(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    stage_expected: tuple[str, ...],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    limit: int,
    *,
    strict_attrs: bool,
    unknown_task_budget: dict[str, int] | None,
) -> tuple[list[tuple[tuple[int, ObservedKernelNode], ...]], int, list[tuple[int, ObservedKernelNode]]]:
    """Consume up to ``limit`` complete offline multi-kernel invocations with per-subkernel task types."""
    invocations: list[tuple[tuple[int, ObservedKernelNode], ...]] = []
    consumed = 0
    partial_sample: list[tuple[int, ObservedKernelNode]] = []
    while consumed < limit:
        spent: dict[int, int] = {}
        unknown_spent: dict[str, int] = {}
        picks: list[tuple[int, ObservedKernelNode]] = []
        complete = True
        for sub_idx, tok in enumerate(expected_kernels):
            exp_tt = _subkernel_task_type(pred, sub_idx)
            exp_shape = stage_expected[sub_idx] if sub_idx < len(stage_expected) else ""
            picked: tuple[int, ObservedKernelNode] | None = None
            for pos, node in enumerate(obs_dag.nodes):
                if remaining_counts.get(pos, 0) - spent.get(pos, 0) <= 0:
                    continue
                if not _token_matches_node(node, tok):
                    continue
                if not _task_type_compatible(
                    pred, node, expected_override=exp_tt, is_offline_summary=True,
                    unknown_task_budget=unknown_task_budget,
                ):
                    continue
                if (
                    unknown_task_budget is not None
                    and _normalize_task_type(node.task_type) == "UNKNOWN"
                    and exp_tt
                    and unknown_task_budget.get(exp_tt, 0) - unknown_spent.get(exp_tt, 0) <= 0
                ):
                    continue
                if strict_attrs and not _stage_attrs_strict_ok(
                    pred, node, exp_shape, is_offline_summary=True,
                ):
                    continue
                picked = (pos, node)
                break
            if picked is None:
                complete = False
                break
            spent[picked[0]] = spent.get(picked[0], 0) + 1
            if unknown_task_budget is not None and _normalize_task_type(picked[1].task_type) == "UNKNOWN" and exp_tt:
                unknown_spent[exp_tt] = unknown_spent.get(exp_tt, 0) + 1
            picks.append(picked)
        if not complete:
            partial_sample = picks
            break
        batch = min(
            limit - consumed,
            min(remaining_counts[pos] // need for pos, need in spent.items()),
        )
        if unknown_task_budget is not None and unknown_spent:
            batch = min(
                batch,
                min(unknown_task_budget.get(tt, 0) // need for tt, need in unknown_spent.items()),
            )
        if batch <= 0:
            break
        for pos, need in spent.items():
            remaining_counts[pos] -= need * batch
        if unknown_task_budget is not None:
            for tt, need in unknown_spent.items():
                unknown_task_budget[tt] -= need * batch
        invocations.append(tuple(picks))
        consumed += batch
    return invocations, consumed, partial_sample


def _consume_single_occurrences(
    pred: PredictedKernelNode,
    tok: str,
    exp_shape: str,
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    limit: int,
    *,
    strict_attrs: bool,
    unknown_task_budget: dict[str, int] | None,
) -> tuple[list[tuple[tuple[int, ObservedKernelNode], ...]], int]:
    """Consume up to ``limit`` occurrences of a single-kernel prediction."""
    exp_tt = _subkernel_task_type(pred, 0)
    candidates: list[tuple[int, ObservedKernelNode]] = [
        (pos, node)
        for pos, node in enumerate(obs_dag.nodes)
        if remaining_counts.get(pos, 0) > 0
        and _token_matches_node(node, tok)
        and _task_type_compatible(
            pred, node, expected_override=exp_tt,
            is_offline_summary=obs_dag.is_offline_summary,
            unknown_task_budget=unknown_task_budget,
        )
        and (
            not strict_attrs
            or _stage_attrs_strict_ok(pred, node, exp_shape, is_offline_summary=obs_dag.is_offline_summary)
        )
    ]
    if not obs_dag.is_offline_summary:
        candidates.sort(key=lambda item: _candidate_preference_key(pred, item, exp_shape))
    need = limit
    invocations: list[tuple[tuple[int, ObservedKernelNode], ...]] = []
    consumed = 0
    for pos, node in candidates:
        if need <= 0:
            break
        avail = remaining_counts.get(pos, 0)
        if avail <= 0:
            continue
        is_unknown = unknown_task_budget is not None and _normalize_task_type(node.task_type) == "UNKNOWN" and exp_tt
        if is_unknown:
            avail = min(avail, unknown_task_budget.get(exp_tt, 0))
            if avail <= 0:
                continue
        take = min(avail, need)
        remaining_counts[pos] -= take
        if is_unknown and unknown_task_budget is not None:
            unknown_task_budget[exp_tt] -= take
        consumed += take
        need -= take
        invocations.append(((pos, node),))
    return invocations, consumed


def _consume_invocations(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    stage_expected: tuple[str, ...],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    limit: int,
    *,
    dag_has_timing: bool,
    strict_attrs: bool,
    unknown_task_budget: dict[str, int] | None,
) -> tuple[list[tuple[tuple[int, ObservedKernelNode], ...]], int, list[tuple[int, ObservedKernelNode]]]:
    """Consume up to ``limit`` complete invocations; return ``(invocations, count, partial_sample)``."""
    if limit <= 0:
        return [], 0, []
    if len(expected_kernels) == 1:
        exp_shape = stage_expected[0] if stage_expected else ""
        invs, count = _consume_single_occurrences(
            pred, expected_kernels[0], exp_shape, obs_dag, remaining_counts, limit,
            strict_attrs=strict_attrs, unknown_task_budget=unknown_task_budget,
        )
        return invs, count, []
    if obs_dag.is_offline_summary:
        return _consume_offline_multi(
            pred, expected_kernels, stage_expected, obs_dag, remaining_counts, limit,
            strict_attrs=strict_attrs, unknown_task_budget=unknown_task_budget,
        )
    invs: list[tuple[tuple[int, ObservedKernelNode], ...]] = []
    partial_sample: list[tuple[int, ObservedKernelNode]] = []
    for _ in range(limit):
        matched, partial = _try_consume_one_live_multi(
            pred, expected_kernels, stage_expected, obs_dag, remaining_counts,
            dag_has_timing=dag_has_timing, strict_attrs=strict_attrs,
        )
        if matched is None:
            partial_sample = partial
            break
        invs.append(matched)
    return invs, len(invs), partial_sample


def _count_remaining_forbidden(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    *,
    unknown_task_budget: Mapping[str, int] | None = None,
) -> int:
    total = 0
    for pos, node in enumerate(obs_dag.nodes):
        rem = remaining_counts.get(pos, 0)
        if rem <= 0:
            continue
        for sub_idx, tok in enumerate(expected_kernels):
            exp_tt = _subkernel_task_type(pred, sub_idx)
            if _token_matches_node(node, tok) and _task_type_compatible(
                pred, node, expected_override=exp_tt,
                is_offline_summary=obs_dag.is_offline_summary,
                unknown_task_budget=unknown_task_budget,
            ):
                total += rem
                break
    return total


def _verify_matched_node_attributes(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    invocations: Sequence[tuple[tuple[int, ObservedKernelNode], ...]],
    rank_id: int,
    *,
    is_offline_summary: bool,
    exp_diffs: list[ParityDiff],
    lane_diffs: list[ParityDiff],
    comm_diffs: list[ParityDiff],
    shape_diffs: list[ParityDiff],
    launch_diffs: list[ParityDiff],
) -> None:
    """Validate expansion completeness, lane, collective ranks, vllm_scope, and per-stage shapes."""
    hits: list[tuple[int, ObservedKernelNode]] = [hit for inv in invocations for hit in inv]
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

    stage_expected = _resolve_stage_expected_shapes(pred, expected_kernels)
    if any(stage_expected):
        _verify_stage_shapes(
            pred, expected_kernels, stage_expected, invocations, rank_id,
            is_offline_summary=is_offline_summary,
            shape_diffs=shape_diffs,
        )


def _verify_stage_shapes(
    pred: PredictedKernelNode,
    expected_kernels: tuple[str, ...],
    stage_expected: tuple[str, ...],
    invocations: Sequence[tuple[tuple[int, ObservedKernelNode], ...]],
    rank_id: int,
    *,
    is_offline_summary: bool,
    shape_diffs: list[ParityDiff],
) -> None:
    if not is_offline_summary:
        for inv in invocations:
            for sub_idx, (_, obs_node) in enumerate(inv):
                exp_shape = stage_expected[sub_idx] if sub_idx < len(stage_expected) else ""
                if not exp_shape or _shapes_match(exp_shape, obs_node.input_shapes):
                    continue
                shape_diffs.append(ParityDiff(
                    code="SHAPE_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=f"shape mismatch for {pred.api_or_symbol} stage {sub_idx} ({obs_node.kernel_name}): expected {exp_shape}",
                    expected=exp_shape,
                    observed=obs_node.input_shapes or "missing",
                ))
                return
        return
    for sub_idx, (_, exp_shape) in enumerate(zip(expected_kernels, stage_expected)):
        if not exp_shape:
            continue
        obs_shapes = [
            inv[sub_idx][1].input_shapes
            for inv in invocations
            if sub_idx < len(inv) and inv[sub_idx][1].input_shapes and inv[sub_idx][1].input_shapes != "N/A"
        ]
        bad_shape = next((s for s in obs_shapes if not _shapes_match(exp_shape, s)), "")
        if bad_shape:
            shape_diffs.append(ParityDiff(
                code="SHAPE_MISMATCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=f"shape mismatch for {pred.api_or_symbol} stage {sub_idx}: expected {exp_shape}",
                expected=exp_shape,
                observed=bad_shape,
            ))
            return


def _verify_rank_dependency_order(
    pred_nodes: Sequence[PredictedKernelNode],
    matched_bounds: Mapping[str, tuple[int, int, int | None, int | None]],
    rank_id: int,
) -> list[ParityDiff]:
    """Enforce predecessor completion strictly before consumer start across every edge."""
    dag_diffs: list[ParityDiff] = []
    for pred in pred_nodes:
        if pred.node_id not in matched_bounds:
            continue
        cur_start_pos, _, cur_start_ns, _ = matched_bounds[pred.node_id]
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
            _, dep_end_pos, _, dep_end_ns = matched_bounds[dep_id]
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
            elif dep_end_ns is not None and cur_start_ns is not None and dep_end_ns > cur_start_ns:
                dep_end_str = _format_ns_as_us(dep_end_ns)
                cur_start_str = _format_ns_as_us(cur_start_ns)
                dag_diffs.append(ParityDiff(
                    code="DEPENDENCY_ORDER_MISMATCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=(
                        f"dependency {dep_id} ended at {dep_end_str}us "
                        f"after consumer {pred.node_id} started at {cur_start_str}us"
                    ),
                    expected=f"{dep_id}.end_us <= {pred.node_id}.start_us",
                    observed=f"{dep_end_str} > {cur_start_str}",
                ))
    return dag_diffs


def _allocation_priority_order(
    pred_nodes: Sequence[PredictedKernelNode],
    expected_list: Sequence[tuple[str, ...]],
    stage_shapes_list: Sequence[tuple[str, ...]],
) -> list[int]:
    """Order more-constrained predictions first (multi-kernel, scoped, shaped) stably."""
    return sorted(
        range(len(pred_nodes)),
        key=lambda idx: (
            0 if len(expected_list[idx]) > 1 else 1,
            -len(expected_list[idx]),
            0 if pred_nodes[idx].vllm_scope else 1,
            0 if any(stage_shapes_list[idx]) else 1,
        ),
    )


def _allocate_rank_invocations(
    pred_nodes: Sequence[PredictedKernelNode],
    expected_list: Sequence[tuple[str, ...]],
    obs_dag: ObservedRankDAG,
    remaining_counts: dict[int, int],
    *,
    dag_has_timing: bool,
    unknown_task_budget: dict[str, int] | None,
) -> tuple[
    list[list[tuple[tuple[int, ObservedKernelNode], ...]]],
    list[int],
    list[bool],
    list[list[tuple[int, ObservedKernelNode]]],
]:
    """Allocate complete invocations across predicted nodes (strict min -> relaxed min -> strict slack)."""
    n_preds = len(pred_nodes)
    stage_shapes_list = [
        _resolve_stage_expected_shapes(pred, exp)
        for pred, exp in zip(pred_nodes, expected_list)
    ]
    order = _allocation_priority_order(pred_nodes, expected_list, stage_shapes_list)
    alloc_invs: list[list[tuple[tuple[int, ObservedKernelNode], ...]]] = [[] for _ in range(n_preds)]
    alloc_counts: list[int] = [0] * n_preds
    order_ok: list[bool] = [True] * n_preds
    partial_samples: list[list[tuple[int, ObservedKernelNode]]] = [[] for _ in range(n_preds)]

    # Pass 1a: satisfy mandatory minimum invocations with strict attribute/shape matching first.
    for idx in order:
        pred = pred_nodes[idx]
        if pred.min_count <= 0 or pred.max_count == 0:
            continue
        invs, count, _ = _consume_invocations(
            pred, expected_list[idx], stage_shapes_list[idx], obs_dag, remaining_counts, pred.min_count,
            dag_has_timing=dag_has_timing, strict_attrs=True, unknown_task_budget=unknown_task_budget,
        )
        alloc_invs[idx].extend(invs)
        alloc_counts[idx] += count

    # Pass 1b: bind remaining unsatisfied mandatory minimums without attribute filtering for exact diffs.
    for idx in order:
        pred = pred_nodes[idx]
        if pred.min_count <= 0 or pred.max_count == 0 or alloc_counts[idx] >= pred.min_count:
            continue
        rem_need = pred.min_count - alloc_counts[idx]
        invs, count, partial = _consume_invocations(
            pred, expected_list[idx], stage_shapes_list[idx], obs_dag, remaining_counts, rem_need,
            dag_has_timing=dag_has_timing, strict_attrs=False, unknown_task_budget=unknown_task_budget,
        )
        alloc_invs[idx].extend(invs)
        alloc_counts[idx] += count
        if len(expected_list[idx]) > 1 and alloc_counts[idx] < pred.min_count and (alloc_counts[idx] == 0 or partial):
            order_ok[idx] = False
            partial_samples[idx] = partial

    # Pass 2: absorb remaining invocations within explicitly declared [min_count, max_count] slack (strict).
    for idx in order:
        pred = pred_nodes[idx]
        if pred.max_count is None or pred.max_count <= alloc_counts[idx]:
            continue
        slack = pred.max_count - alloc_counts[idx]
        invs, count, _ = _consume_invocations(
            pred, expected_list[idx], stage_shapes_list[idx], obs_dag, remaining_counts, slack,
            dag_has_timing=dag_has_timing, strict_attrs=True, unknown_task_budget=unknown_task_budget,
        )
        alloc_invs[idx].extend(invs)
        alloc_counts[idx] += count

    return alloc_invs, alloc_counts, order_ok, partial_samples


def _bounds_for_invocations(
    invocations: Sequence[tuple[tuple[int, ObservedKernelNode], ...]],
    *,
    dag_has_timing: bool,
) -> tuple[int, int, int | None, int | None]:
    flat = [hit for inv in invocations for hit in inv]
    min_pos = min(pos for pos, _ in flat)
    max_pos = max(pos for pos, _ in flat)
    known = [
        tb for _, node in flat
        if (tb := _node_time_bounds_ns(node, dag_has_timing=dag_has_timing)) is not None
    ]
    if known:
        return min_pos, max_pos, min(tb[0] for tb in known), max(tb[1] for tb in known)
    return min_pos, max_pos, None, None


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
    has_timing = _dag_has_timing(obs_dag)
    unknown_budget = _offline_unknown_task_budget(obs_dag)
    expected_list = [_expected_expansion_for(pred, device_arch, registry) for pred in pred_nodes]
    alloc_invs, alloc_counts, order_ok, partial_samples = _allocate_rank_invocations(
        pred_nodes, expected_list, obs_dag, remaining_counts,
        dag_has_timing=has_timing, unknown_task_budget=unknown_budget,
    )

    matched_bounds: dict[str, tuple[int, int, int | None, int | None]] = {}
    rank_id = obs_dag.rank_id

    for idx, (pred, expected_kernels) in enumerate(zip(pred_nodes, expected_list)):
        observed_total = alloc_counts[idx]
        invocations = alloc_invs[idx]

        if pred.max_count == 0:
            forbidden_seen = _count_remaining_forbidden(
                pred, expected_kernels, obs_dag, remaining_counts, unknown_task_budget=unknown_budget,
            )
            if forbidden_seen > 0:
                launch_diffs.append(ParityDiff(
                    code="MISSING_OR_EXTRA_LAUNCH",
                    rank_id=rank_id,
                    node_id=pred.node_id,
                    detail=f"expected<=0 launches of {expected_kernels}, got {forbidden_seen}",
                    expected="<=0",
                    observed=str(forbidden_seen),
                ))
            continue

        if len(expected_kernels) > 1 and pred.min_count > 0 and not order_ok[idx]:
            sample_hits = partial_samples[idx] or [hit for inv in invocations for hit in inv]
            exp_diffs.append(ParityDiff(
                code="EXPANSION_MISMATCH",
                rank_id=rank_id,
                node_id=pred.node_id,
                detail=(
                    f"incomplete or out-of-order 1-to-many expansion for {pred.api_or_symbol}: "
                    f"expected ordered {expected_kernels}"
                ),
                expected=",".join(expected_kernels),
                observed=",".join(n.kernel_name for _, n in sample_hits) or "none",
            ))
            continue

        if pred.is_ai_cpu_metadata and pred.min_count > 0 and observed_total == 0:
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

        if not invocations:
            continue

        matched_bounds[pred.node_id] = _bounds_for_invocations(invocations, dag_has_timing=has_timing)
        _verify_matched_node_attributes(
            pred, expected_kernels, invocations, rank_id,
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


def _infer_offline_task_type(kname: str, *, device_arch: str = "A3") -> str:
    if kname == "PadV3":
        return "AI_VECTOR_CORE" if device_arch == "A5" else "MIX_AIV"
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


def _parse_json_float(raw: str) -> float | Decimal:
    """Preserve exact Decimal precision whenever IEEE-754 float64 would round."""
    dec = Decimal(raw)
    flt = float(raw)
    if Decimal(str(flt)) == dec:
        return flt
    return dec


class _JsonStreamReader:
    """Chunked 1 MiB JSON scanner validating depth-1 ``traceEvents`` and full-document EOF."""

    __slots__ = ("_stream", "_name", "_buf", "_pos", "_eof", "_decoder")

    def __init__(self, stream: Any, name: str) -> None:
        self._stream = stream
        self._name = name
        self._buf = ""
        self._pos = 0
        self._eof = False
        self._decoder = json.JSONDecoder(parse_float=_parse_json_float)

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
    needed_budget = sum(
        max(1, len(exp)) * max(1, pred.max_count if pred.max_count is not None else pred.min_count)
        for pred, exp in zip(pred_nodes, expected_sets)
    )
    retain_limit = max(_MAX_REPORTED_SAMPLES, needed_budget * _RANK_4D)

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
                start_ns=dev.ts_ns,
                dur_ns=dev.dur_ns,
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

    retained_nodes.sort(key=lambda item: (item.start_ns is None, item.start_ns or 0))
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


_A2_SKU = "Ascend910B4"
_A3_SKU = "Ascend910_9392"
_A5_SKU = "Ascend950DT_9582"
_DEFAULT_SKU_BY_ARCH: Mapping[str, str] = {
    "A2": _A2_SKU,
    "A3": _A3_SKU,
    "A5": _A5_SKU,
}
_MAX_CASE_JSON_BYTES = 50 * 1024 * 1024
_RANK_PATH_RE = re.compile(r"(?:^|[^a-z0-9])(?:rank|worker|device)[_-]?(\d+)(?:[^0-9]|$)", re.IGNORECASE)
_DEFAULT_PROMPT_LEN = 128
_DEFAULT_OUTPUT_TOKENS = 8
_A3_DEFAULT_BATCH = 3
_A5_DEFAULT_BATCH = 2
_DEFAULT_LAYERS_MOE = 4
_DEFAULT_LAYERS_DENSE = 2
_SUBMOD_LAYERS = 4
_TP_SINGLE = 1
_TP_PAIR = 2
_TP_QUAD = 4
_VALID_TOPO_AXES: frozenset[str] = frozenset({"tp", "dp", "ep", "pp", "cp", "dcp", "pcp"})
_SUBKERNEL_TASK_TYPE_OVERRIDES: Mapping[str, str] = {
    "MemSet": "AI_VECTOR_CORE",
    "ZerosLike": "AI_VECTOR_CORE",
    "ConcatD": "AI_VECTOR_CORE",
    "Slice": "AI_VECTOR_CORE",
    "ViewCopy": "AI_VECTOR_CORE",
    "ClipByValueV2": "AI_VECTOR_CORE",
    "ClipByValue": "AI_VECTOR_CORE",
    "IndexCheck": "AI_VECTOR_CORE",
    "Index": "AI_VECTOR_CORE",
    "Cast": "AI_VECTOR_CORE",
    "Sort": "AI_VECTOR_CORE",
    "GatherElements": "AI_VECTOR_CORE",
    "GatherElementsV2": "AI_VECTOR_CORE",
    "TransposeBatchMatMul": "AI_CORE",
    "ChunkKdaFwdPrepare": "MIX_AIC",
    "ChunkFwdH": "MIX_AIC",
    "ChunkKdaFwdFinalize": "AI_CORE",
    "ChunkGatedDeltaRuleFwdPrepare": "MIX_AIC",
    "ChunkFwdO": "MIX_AIC",
}
_COMM_FAMILIES: frozenset[str] = frozenset({
    "allreduce",
    "allgather",
    "reduce_scatter",
    "reducescatter",
    "alltoall",
    "alltoallv",
    "mc2",
    "mc2_dispatch",
    "mc2_combine",
    "p2p",
})
_MATRIX_FAMILIES: frozenset[str] = frozenset({
    "qkv_projection",
    "output_projection",
    "gate_up_projection",
    "down_projection",
    "dense_gemm",
    "gemm",
    "grouped_gemm",
    "linear",
    "router",
    "attention",
    "mla",
    "mla_prolog",
    "sparse_attention",
    "sparse_indexer",
    "linear_attention",
})
_SKIP_LOGICAL_FAMILIES: frozenset[str] = frozenset({
    "boundary",
    "input",
    "output",
    "barrier",
    "event",
    "release",
    "tensor_view",
    "attention_metadata",
    "state_publication",
    "state_publish",
    "host_sync",
    "noop",
})
_PAD_MULTI_KERNEL = ("MemSet", "PadV3")
_VARIANT_DEFAULTS: tuple[str, ...] = (
    "GatherV2", "Mul", "Add", "TensorMove", "FusedInferAttentionScore",
    "QuantLightningIndexerV2", "MoeGatingTopK", "TopKV2", "hcom_alltoallv",
    "hcom_send", "ZerosLike", "SwiGlu", "MoeTokenUnpermute",
    "_fused_dcp_lse_combine_kernel", "_pack_dcp_output_lse_kernel",
    "_hc_combine_norm_kernel", "_hc_gate_mix_kernel", "_pack_candidates",
    "_merge_candidates", "l2norm_fwd_kernel2", "fused_gdn_gating_kernel",
    "layer_norm_gated_fwd_kernel", "RotaryPositionEmbedding",
    "ChunkGatedDeltaRuleFwdPrepare", "ChunkFwdO", "ChunkFwdH", "SolveTril",
    "ScatterUpdate", "LayerNormV3", "DequantSituQuant", "SwiGluQuant",
    "GeluQuant", "Dequantize", "GatherD", "Silu", "Gelu", "ReduceSum",
    "SoftmaxV2", "ArgMaxV2", "BatchMatMulV2", "ConcatD", "PadV3",
    "PoolKeyCompress", "RmsNormQuant", "AddRmsNormQuant",
    "InplacePartialRotaryMul", "aclnnKvCompressEpilog",
    "RecoverSsmStateCommit", "MoeHashRouting", "QuantSparseLightningIndexer",
)


_VALID_ARCHES: frozenset[str] = frozenset({"A2", "A3", "A5"})


def _canonical_sku(sku: str) -> str:
    raw = sku.strip()
    if not raw:
        return ""
    try:
        return resolve_hardware_sku(raw)
    except ModelingError:
        return raw


def _arch_from_sku_name(sku: str) -> str | None:
    resolved = _canonical_sku(sku).lower()
    if not resolved:
        return None
    if "950" in resolved or "a5" in resolved or "3510" in resolved:
        return "A5"
    if "910b" in resolved or "a2" in resolved:
        return "A2"
    if "910_93" in resolved or "910c" in resolved or "a3" in resolved:
        return "A3"
    return None


def _declared_arch_from_case(case: Case) -> str | None:
    """Return explicitly declared Ascend architecture family on ``case``, else ``None``."""
    family = str(getattr(case.machine, "arch_family", "") or "").strip().upper()
    if family in _VALID_ARCHES:
        return family
    inferred = _arch_from_sku_name(str(getattr(case.machine, "sku", "") or ""))
    if inferred is not None:
        return inferred
    dies = getattr(case.machine, "dies", ())
    if dies:
        preset = str(getattr(getattr(dies[0], "compute_arch", None), "preset", "") or "")
        preset_arch = _arch_from_sku_name(preset)
        if preset_arch is not None:
            return preset_arch
    return None


def _infer_arch_from_case(case: Case) -> str:
    return _declared_arch_from_case(case) or "A3"


def _resolve_live_target_device(
    parsed_case: Case | None,
    *,
    device_arch: str | None,
    hardware_sku: str | None,
) -> tuple[str, str, str | None]:
    """Derive canonical ``(device_arch, hardware_sku)`` from ``parsed_case`` and validate overrides."""
    case_sku = str(getattr(parsed_case.machine, "sku", "") or "").strip() if parsed_case is not None else ""
    declared_arch = _declared_arch_from_case(parsed_case) if parsed_case is not None else None
    if declared_arch is not None and device_arch is not None and device_arch != declared_arch:
        label = case_sku or declared_arch
        return (
            declared_arch,
            case_sku,
            f"--arch {device_arch} conflicts with case SKU {label} ({declared_arch})",
        )
    if hardware_sku is not None:
        arg_sku = hardware_sku.strip()
        if not arg_sku:
            return device_arch or declared_arch or "A3", "", "--sku must be non-empty"
        if case_sku and _canonical_sku(arg_sku) != _canonical_sku(case_sku):
            return device_arch or declared_arch or "A3", arg_sku, f"--sku {arg_sku} conflicts with case SKU {case_sku}"
        sku_arch = _arch_from_sku_name(arg_sku)
        resolved_arch = device_arch or declared_arch or sku_arch or "A3"
        if sku_arch is not None and sku_arch != resolved_arch:
            return resolved_arch, arg_sku, f"--sku {arg_sku} ({sku_arch}) conflicts with arch {resolved_arch}"
        return resolved_arch, _canonical_sku(arg_sku), None
    resolved_arch = device_arch or declared_arch or "A3"
    canonical_case_sku = _canonical_sku(case_sku) if case_sku else ""
    return resolved_arch, canonical_case_sku or _default_sku_for_arch(resolved_arch), None


def _default_sku_for_arch(device_arch: str) -> str:
    return _DEFAULT_SKU_BY_ARCH.get(device_arch, _A5_SKU)


def _stream_lane_for_op(op: OpSpec) -> str:
    stream = str(op.stream or "").strip().lower()
    if "shared" in stream:
        return "shared_expert"
    if "comm" in stream or op.family in _COMM_FAMILIES:
        return "comm"
    if "aux" in stream:
        return "auxiliary"
    return "main"


def _task_type_for_op(op: OpSpec, *, is_ai_cpu: bool, expected_kernels: tuple[str, ...]) -> str:
    if is_ai_cpu or op.family == "ai_cpu_metadata":
        return AI_CPU_TASK_TYPE
    if op.family in _COMM_FAMILIES or any(k.startswith("hcom_") for k in expected_kernels):
        return "COMMUNICATION"
    if op.family in {"attention", "mla", "sparse_attention", "sparse_indexer", "mla_prolog", "indexer"}:
        return "MIX_AIC"
    if op.family in _MATRIX_FAMILIES:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def _expected_task_types_for_kernels(
    expected_kernels: tuple[str, ...],
    *,
    default_task_type: str,
    device_arch: str,
) -> tuple[str, ...]:
    """Derive exact per-subkernel Ascend task types grounded in A3/A5 captures."""
    out: list[str] = []
    for kernel in expected_kernels:
        if kernel == "PadV3":
            out.append("AI_VECTOR_CORE" if device_arch == "A5" else "MIX_AIV")
        elif kernel.startswith("hcom_"):
            out.append("COMMUNICATION")
        elif kernel in _SUBKERNEL_TASK_TYPE_OVERRIDES:
            out.append(_SUBKERNEL_TASK_TYPE_OVERRIDES[kernel])
        else:
            out.append(default_task_type)
    return tuple(out)


def _op_meta(op: OpSpec) -> dict[str, Any]:
    """Decode ``OpSpec.metadata`` cleanly whether stored as a Mapping or tuple of 2-tuples."""
    raw = getattr(op, "metadata", ())
    if isinstance(raw, Mapping):
        return dict(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        out: dict[str, Any] = {}
        for item in raw:
            if isinstance(item, tuple) and len(item) == _TP_PAIR:
                out[str(item[0])] = item[1]
        return out
    return {}


def _format_shape_tuple(shape: Any) -> str:
    if isinstance(shape, str):
        return shape.strip()
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        if shape and all(isinstance(item, Sequence) and not isinstance(item, (str, bytes)) for item in shape):
            slots = [
                ",".join(str(int(d)) for d in sub if isinstance(d, (int, float)) and not isinstance(d, bool))
                for sub in shape
            ]
            return ";".join(slots)
        dims = [str(int(d)) for d in shape if isinstance(d, (int, float)) and not isinstance(d, bool)]
        return ",".join(dims)
    return ""


def _int_shape_tuple(raw: Any) -> tuple[int, ...] | None:
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        dims = tuple(int(d) for d in raw if isinstance(d, (int, float)) and not isinstance(d, bool))
        return dims if dims else None
    return None


def _gdn_stage_operand_shapes(op: OpSpec, meta: Mapping[str, Any]) -> str:
    """Build 7-operand CANN input descriptor for lowered prefill ``gdn_{prepare,state,output}`` ops."""
    raw_lengths = meta.get("lengths")
    lengths = _int_shape_tuple(raw_lengths)
    if not lengths:
        return ""
    hv_dim = int(meta.get("heads") or 0)
    hk_dim = int(meta.get("key_heads") or hv_dim)
    k_dim = int(meta.get("head_dim_k") or (op.local_shape[-1] if op.local_shape else 0))
    v_dim = int(meta.get("head_dim_v") or k_dim)
    if hv_dim <= 0 or hk_dim <= 0 or k_dim <= 0 or v_dim <= 0:
        return ""
    t_dim = sum(lengths)
    seq_num = len(lengths)
    chunk = max(1, int(meta.get("chunk") or _GDN_CHUNK_SIZE))
    chunks_per_seq = _int_shape_tuple(meta.get("chunks_per_sequence"))
    num_chunks = sum(chunks_per_seq) if chunks_per_seq else sum((ln + chunk - 1) // chunk for ln in lengths)
    cu_seq = (seq_num + 1,)
    chunk_idx = (num_chunks * 2,)
    b_dim = 1
    q_shp = (b_dim, hk_dim, t_dim, k_dim)
    v_shp = (b_dim, hv_dim, t_dim, v_dim)
    g_shp = (b_dim, hv_dim, t_dim)
    if op.family == "gdn_prepare":
        return _format_operand_shapes((q_shp, q_shp, v_shp, g_shp, g_shp, cu_seq, chunk_idx))
    if op.family == "gdn_state":
        return _format_operand_shapes((
            q_shp, (b_dim, hv_dim, t_dim, k_dim), v_shp, g_shp,
            (seq_num, hv_dim, k_dim, v_dim), cu_seq, chunk_idx,
        ))
    if op.family == "gdn_output":
        return _format_operand_shapes((
            q_shp, q_shp, v_shp, (b_dim, hv_dim, num_chunks, k_dim, v_dim),
            g_shp, cu_seq, chunk_idx,
        ))
    return ""


def _gemm_scale_shapes(
    op: OpSpec,
    meta: Mapping[str, Any],
    *,
    n_dim: int,
    k_dim: int,
    groups: int,
    raw_tb: Any,
) -> tuple[tuple[int, ...] | None, tuple[int, ...] | None]:
    abi = op.precision_abi
    w_scale_dtype = str(
        meta.get("weight_scale_format")
        or (abi.weight_scale_dtype if abi is not None else "")
        or ""
    ).strip().lower()
    w_scale_shape = _int_shape_tuple(meta.get("weight_scale_shape"))
    if w_scale_shape is None and w_scale_dtype:
        w_block = abi.weight_block if abi is not None else ()
        act_pack = str(abi.act_scale_pack if abi is not None else "").strip().lower()
        if w_scale_dtype == "fp8_e8m0" and len(w_block) >= 2 and w_block[-1] > 1:
            k_groups = (k_dim + w_block[-1] - 1) // w_block[-1]
            if act_pack == "pack2" or w_scale_dtype == "fp8_e8m0":
                k_packed = (k_groups + 1) // 2
                inner_s: tuple[int, ...] = (n_dim, k_packed, 2) if raw_tb is True else (k_packed, n_dim, 2)
            else:
                inner_s = (n_dim, k_groups) if raw_tb is True else (k_groups, n_dim)
            w_scale_shape = (groups, *inner_s) if groups > 0 else inner_s
        else:
            w_scale_shape = (groups, n_dim) if groups > 0 else (n_dim,)
    a_scale_shape = _int_shape_tuple(meta.get("activation_scale_shape"))
    return w_scale_shape, a_scale_shape


def _gemm_bias_shape(meta: Mapping[str, Any], n_dim: int) -> tuple[int, ...] | None:
    explicit = _int_shape_tuple(meta.get("bias_shape"))
    if explicit is not None:
        return explicit
    if bool(meta.get("has_bias") or meta.get("bias")):
        return (n_dim,)
    return None


def _gemm_operand_shapes(op: OpSpec, meta: Mapping[str, Any]) -> str:
    geom = op.local_shape or op.launch_shape or op.semantic_shape
    if len(geom) not in (_RANK_3D, _RANK_4D):
        return ""
    explicit_weight = _int_shape_tuple(meta.get("weight_shape"))
    trans_a = bool(meta.get("transpose_a", meta.get("trans_a", False)))
    raw_tb = meta.get("transpose_b", meta.get("trans_b"))
    if len(geom) == _RANK_4D:
        batch, m_dim, n_dim, k_dim = (int(d) for d in geom)
        trans_b = True if raw_tb is None else bool(raw_tb)
        a_shape: tuple[int, ...] = (batch, k_dim, m_dim) if trans_a else (batch, m_dim, k_dim)
        b_shape: tuple[int, ...] = (
            explicit_weight
            if explicit_weight is not None
            else ((batch, n_dim, k_dim) if trans_b else (batch, k_dim, n_dim))
        )
        operands: list[tuple[int, ...]] = [a_shape, b_shape]
        bias_shp = _gemm_bias_shape(meta, n_dim)
        if bias_shp is not None:
            operands.append(bias_shp)
        return _format_operand_shapes(operands)

    m_dim, n_dim, k_dim = (int(d) for d in geom)
    a_shape = (k_dim, m_dim) if trans_a else (m_dim, k_dim)
    is_grouped = op.family == "grouped_gemm"
    groups = int(meta.get("matrix_weight_bank_groups") or meta.get("matrix_groups") or 0) if is_grouped else 0
    if explicit_weight is not None:
        b_shape = explicit_weight
    elif is_grouped:
        trans_b = bool(raw_tb) if raw_tb is not None else False
        inner = (n_dim, k_dim) if trans_b else (k_dim, n_dim)
        b_shape = (groups, *inner) if groups > 0 else inner
    else:
        trans_b = True if raw_tb is None else bool(raw_tb)
        w_layout = str(
            meta.get("weight_layout")
            or (op.precision_abi.weight_layout if op.precision_abi is not None else "")
            or ""
        ).strip().upper()
        w_dtype = str(
            meta.get("weight_dtype")
            or (op.precision_abi.weight_dtype if op.precision_abi is not None else "")
            or op.dtype
            or ""
        ).strip().lower()
        if w_layout.startswith("FRACTAL_NZ"):
            if "C0_16" in w_layout:
                c0 = _NZ_C0_2B
            elif "C0_32" in w_layout or w_dtype in _ONE_BYTE_DTYPES:
                c0 = _NZ_C0_1B
            else:
                c0 = _NZ_C0_2B
            if raw_tb is True:
                b_shape = ((k_dim + c0 - 1) // c0, (n_dim + _NZ_R0 - 1) // _NZ_R0, _NZ_R0, c0)
            else:
                b_shape = ((n_dim + c0 - 1) // c0, (k_dim + _NZ_R0 - 1) // _NZ_R0, _NZ_R0, c0)
        else:
            b_shape = (n_dim, k_dim) if trans_b else (k_dim, n_dim)

    bias_shp = _gemm_bias_shape(meta, n_dim)
    w_scale_shp, a_scale_shp = _gemm_scale_shapes(
        op, meta, n_dim=n_dim, k_dim=k_dim, groups=groups, raw_tb=raw_tb,
    )
    if is_grouped and groups > 0:
        slots: list[tuple[int, ...]] = [() for _ in range(_GROUPED_MATMUL_FIXED_SLOTS)]
        slots[0] = a_shape
        slots[1] = b_shape
        if bias_shp is not None:
            slots[_GROUPED_MATMUL_BIAS_SLOT] = bias_shp
        if w_scale_shp is not None:
            slots[_GROUPED_MATMUL_WEIGHT_SCALE_SLOT] = w_scale_shp
        slots[_GROUPED_MATMUL_GROUP_LIST_SLOT] = (groups,)
        if a_scale_shp is not None:
            slots.append(a_scale_shp)
        return _format_operand_shapes(slots)

    operands = [a_shape, b_shape]
    if bias_shp is not None:
        operands.append(bias_shp)
    if w_scale_shp is not None:
        operands.append(w_scale_shp)
    if a_scale_shp is not None:
        operands.append(a_scale_shp)
    return _format_operand_shapes(operands)


def _format_input_shapes(op: OpSpec, meta: Mapping[str, Any]) -> str:
    for candidate in (
        meta.get("operand_shapes"),
        meta.get("input_shapes"),
        meta.get("input_shape"),
    ):
        text = _format_shape_tuple(candidate)
        if text:
            return text
    if op.family in {"gdn_prepare", "gdn_state", "gdn_output"}:
        gdn_shapes = _gdn_stage_operand_shapes(op, meta)
        if gdn_shapes:
            return gdn_shapes
    if meta.get("program_kind") == "matmul" or op.family in _GEMM_OP_FAMILIES:
        gemm_shapes = _gemm_operand_shapes(op, meta)
        if gemm_shapes:
            return gemm_shapes
    for candidate in (
        op.local_shape,
        op.padded_shape,
        op.semantic_shape,
    ):
        text = _format_shape_tuple(candidate)
        if text:
            return text
    return ""


def _expand_symbols(
    symbols: Sequence[str],
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[str, ...]:
    """Unconditionally expand EVERY symbol via ``ProbedAbiRegistry`` without whitelists or truncation."""
    expanded: list[str] = []
    for raw in symbols:
        sym = str(raw).strip()
        if not sym:
            continue
        exp = registry.device_kernel_expansion(sym, device_arch)
        if exp:
            expanded.extend(exp)
        else:
            expanded.append(sym)
    return tuple(expanded)


def _select_soc_variant(
    variants: tuple[str, ...],
    *,
    op: OpSpec,
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> str:
    """Resolve a declared SoC/phase variant set to the exact kernel for ``(device_arch, op.phase)``."""
    if len(variants) == 1:
        return variants[0]
    is_a5 = device_arch == "A5"
    is_decode = str(op.phase or "").strip().lower() == "decode"
    soc_pairs: tuple[tuple[str, str, str], ...] = (
        ("MatMulV3", "MatMulV2", "MatMulV3" if is_a5 else "MatMulV2"),
        ("MoeDistributeDispatchV3", "MoeDistributeDispatchV2", "MoeDistributeDispatchV3" if is_a5 else "MoeDistributeDispatchV2"),
        ("MoeDistributeCombineV3", "MoeDistributeCombineV2", "MoeDistributeCombineV3" if is_a5 else "MoeDistributeCombineV2"),
        ("MoeInitRoutingV3", "MoeInitRoutingV2", "MoeInitRoutingV3" if is_a5 else "MoeInitRoutingV2"),
        ("DynamicMxQuant", "DynamicQuantV2", "DynamicMxQuant" if is_a5 else "DynamicQuantV2"),
        ("ScatterPaKvCache", "ReshapeAndCache", "ScatterPaKvCache" if is_a5 else "ReshapeAndCache"),
        ("CausalConv1dUpdate", "CausalConv1d", "CausalConv1dUpdate" if is_decode else "CausalConv1d"),
        ("RecurrentGatedDeltaRule", "ChunkGatedDeltaRuleFwd", "RecurrentGatedDeltaRule" if is_decode else "ChunkGatedDeltaRuleFwd"),
        ("RecurrentKda", "ChunkKdaFwd", "RecurrentKda" if is_decode else "ChunkKdaFwd"),
        ("QuantBatchMatmulV4", "QuantBatchMatmulV3", "QuantBatchMatmulV3"),
    )
    for a, b, choice in soc_pairs:
        if a in variants and b in variants:
            return choice
    if is_a5 and "AddRmsNormDynamicMxQuant" in variants:
        return "AddRmsNormDynamicMxQuant"
    if is_a5 and "MegaMoeArch35" in variants:
        return "MegaMoeArch35"
    known = registry.known_device_kernels(device_arch)
    matched = [
        cand for cand in variants
        if cand in known or registry.device_kernel_expansion(cand, device_arch)
    ]
    if len(matched) == 1:
        return matched[0]
    for cand in _VARIANT_DEFAULTS:
        if cand in variants:
            return cand
    raise ModelingError(
        f"unresolved_variant_set:{op.family}:{op.recipe}:{device_arch}:{variants}"
    )


def _resolve_op_kernels(
    op: OpSpec,
    *,
    meta: Mapping[str, Any],
    device_arch: str,
    registry: ProbedAbiRegistry,
) -> tuple[str, tuple[str, ...]]:
    """Resolve full ordered device kernel tuple with ZERO whitelists and ZERO truncation."""
    explicit_abi = str(
        meta.get("aclnn_symbol") or meta.get("abi_symbol") or meta.get("abi_id") or ""
    ).strip()
    if not explicit_abi and op.kernel_binding is not None:
        explicit_abi = str(
            getattr(op.kernel_binding, "abi_id", "") or getattr(op.kernel_binding, "symbol", "")
        ).strip()

    if explicit_abi:
        abi_exp = registry.device_kernel_expansion(explicit_abi, device_arch)
        if abi_exp:
            return explicit_abi, abi_exp

    raw_native = meta.get("native_kinds")
    if isinstance(raw_native, Sequence) and not isinstance(raw_native, (str, bytes)):
        native_seq = tuple(str(k).strip() for k in raw_native if str(k).strip())
        if native_seq:
            expanded = _expand_symbols(native_seq, device_arch, registry)
            return (explicit_abi or native_seq[0]), expanded

    raw_single = str(meta.get("native_kind") or meta.get("kernel") or "").strip()
    if raw_single:
        sym = explicit_abi or raw_single
        exp = _expand_symbols((explicit_abi,) if explicit_abi else (raw_single,), device_arch, registry)
        return sym, (exp if exp else (raw_single,))

    if explicit_abi:
        exp = _expand_symbols((explicit_abi,), device_arch, registry)
        return explicit_abi, (exp if exp else (explicit_abi,))

    recipe_name = str(op.recipe or "").strip()
    if recipe_name or op.family:
        op_dict: dict[str, Any] = {
            "op_id": op.op_id,
            "family": op.family,
            "recipe": recipe_name,
            "phase": op.phase,
            "stage": op.stage,
            "local_shape": tuple(op.local_shape),
            "padded_shape": tuple(op.padded_shape),
            "launch_shape": tuple(op.launch_shape),
            "dtype": op.dtype,
            "layout": op.layout,
            "source_boundary": op.source_boundary,
            "metadata": dict(meta),
        }
        cert = operator_birth_certificate(op_dict)
        if cert.backend_stack == "unmodeled" or not cert.native_kinds:
            raise ModelingError(
                f"unmodeled_operator_recipe:{op.family}:{recipe_name}"
            )
        if cert.is_variant_set and len(cert.native_kinds) > 1:
            chosen = _select_soc_variant(
                cert.native_kinds, op=op, device_arch=device_arch, registry=registry,
            )
            exp = _expand_symbols((chosen,), device_arch, registry)
            return chosen, (exp if exp else (chosen,))
        expanded = _expand_symbols(cert.native_kinds, device_arch, registry)
        return cert.native_kinds[0], (expanded if expanded else tuple(cert.native_kinds))

    raise ModelingError(f"unmodeled_operator_recipe:{op.family}:{recipe_name}")


def _collective_group_and_ranks(
    op: OpSpec,
    meta: Mapping[str, Any],
    *,
    topo: Topology | None,
    replica: int,
    rank_id: int,
) -> tuple[str, tuple[int, ...]]:
    axis = str(meta.get("group_axis") or "").strip()
    group_label = axis or str(op.link_scope or "").strip()
    raw_ranks = meta.get("comm_ranks")
    if isinstance(raw_ranks, Sequence) and not isinstance(raw_ranks, (str, bytes)):
        parsed = tuple(int(r) for r in raw_ranks)
        if parsed:
            return group_label, parsed
    if topo is not None and axis and axis in _VALID_TOPO_AXES:
        groups = topo.groups(axis, replica, int(op.stage), metadata=meta)
        for grp in groups:
            if rank_id in grp:
                return axis, tuple(int(r) for r in grp)
    return group_label, ()


def build_predicted_nodes_from_prepared_step(
    prepared: PreparedStep,
    *,
    device_arch: str | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PredictedKernelNode, ...]:
    """Derive independent per-physical-rank ``PredictedKernelNode`` DAGs purely from ``PreparedStep``."""
    reg = registry if registry is not None else default_abi_registry()
    arch = device_arch or _infer_arch_from_case(prepared.case)
    topo = Topology(prepared.case.machine, prepared.case.runtime.parallel)
    out: list[PredictedKernelNode] = []

    for entry in prepared.graph.executable:
        replica = int(entry.replica)
        explicit_rank = getattr(entry, "rank", None)
        stage_ops_by_rank: dict[int, list[tuple[OpSpec, dict[str, Any], str, tuple[str, ...], bool]]] = {}
        for op in entry.ops:
            if op.family in _SKIP_LOGICAL_FAMILIES:
                continue
            meta = _op_meta(op)
            is_ai_cpu = (
                bool(meta.get("ai_cpu_metadata"))
                or op.family == "ai_cpu_metadata"
                or str(meta.get("task_type") or meta.get("execution_domain") or "").upper() == AI_CPU_TASK_TYPE
            )
            api_sym, expected_kernels = _resolve_op_kernels(
                op, meta=meta, device_arch=arch, registry=reg,
            )
            phys_ranks = (
                (int(explicit_rank),)
                if explicit_rank is not None
                else (topo.ranks(replica, int(op.stage)) or (replica,))
            )
            for phys_rank in phys_ranks:
                stage_ops_by_rank.setdefault(int(phys_rank), []).append(
                    (op, meta, api_sym, expected_kernels, is_ai_cpu)
                )

        all_op_deps = {op.op_id: tuple(op.deps) for op in entry.ops}
        for phys_rank in sorted(stage_ops_by_rank):
            rank_ops = stage_ops_by_rank[phys_rank]
            emitted_ids = {op.op_id for op, *_ in rank_ops}
            for op, meta, api_sym, expected_kernels, is_ai_cpu in rank_ops:
                resolved_preds: list[str] = []
                seen_walk: set[str] = set()
                stack = list(reversed(op.deps))
                while stack:
                    dep_id = stack.pop()
                    if dep_id in seen_walk:
                        continue
                    seen_walk.add(dep_id)
                    if dep_id in emitted_ids:
                        resolved_preds.append(dep_id)
                    elif dep_id in all_op_deps:
                        stack.extend(reversed(all_op_deps[dep_id]))
                preds = tuple(resolved_preds)
                group_label, comm_ranks = _collective_group_and_ranks(
                    op, meta, topo=topo, replica=replica, rank_id=phys_rank,
                )
                default_tt = _task_type_for_op(op, is_ai_cpu=is_ai_cpu, expected_kernels=expected_kernels)
                out.append(PredictedKernelNode(
                    node_id=op.op_id,
                    rank_id=phys_rank,
                    stream_lane=_stream_lane_for_op(op),
                    task_type=default_tt,
                    api_or_symbol=api_sym,
                    expected_kernels=expected_kernels,
                    expected_task_types=_expected_task_types_for_kernels(
                        expected_kernels, default_task_type=default_tt, device_arch=arch,
                    ),
                    input_shapes=_format_input_shapes(op, meta),
                    predecessors=preds,
                    collective_group=group_label,
                    comm_ranks=comm_ranks,
                    vllm_scope=op.op_id,
                    is_ai_cpu_metadata=is_ai_cpu,
                    min_count=1,
                ))
    return tuple(out)


def _read_case_file(case_path: Path) -> Mapping[str, Any]:
    path = Path(case_path)
    if path.stat().st_size > _MAX_CASE_JSON_BYTES:
        raise ValueError("subgraph_parity_case_exceeds_50_mib")
    return json.loads(path.read_text(encoding="utf-8"))


def build_predicted_nodes_from_case(
    case_source: Case | Mapping[str, Any] | Path,
    *,
    phase: PhaseKind = "decode",
    tokens: int = 1,
    context: int = 1024,
    sequences: int = 1,
    device_arch: str | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PreparedStep, tuple[PredictedKernelNode, ...]]:
    """Lower ``Case -> PredictionSpine -> PreparedStep`` independently with ZERO trace access."""
    if isinstance(case_source, Case):
        case = case_source
    elif isinstance(case_source, Path):
        case = parse_case(_read_case_file(case_source))
    else:
        case = parse_case(dict(case_source))
    resolved_arch, _, dev_err = _resolve_live_target_device(
        case, device_arch=device_arch, hardware_sku=None,
    )
    if dev_err is not None:
        raise ModelingError(f"live_parity_device_conflict:{dev_err}")
    spine = PredictionSpine()
    prepared = spine.lower(
        case,
        LoweringRequest(step=StepRequest(phase, tokens=tokens, context=context, sequences=sequences)),
    )
    preds = build_predicted_nodes_from_prepared_step(
        prepared, device_arch=resolved_arch, registry=registry,
    )
    return prepared, preds


@dataclass(frozen=True, slots=True)
class OfflineCaptureCaseConfig:
    """Pure experiment input config for an offline capture (ZERO expected kernel names)."""

    model_id: str | None
    layers: int = _DEFAULT_LAYERS_MOE
    tp: int = _TP_SINGLE
    dp: int = _TP_SINGLE
    ep: int | None = None
    dcp: int = _TP_SINGLE
    pcp: int = _TP_SINGLE
    quant: str = "bf16"
    kv_cache_dtype: str = "bf16"
    phase: PhaseKind = "decode"
    tokens: int = 1
    context: int = _DEFAULT_PROMPT_LEN
    sequences: int = _A5_DEFAULT_BATCH
    mtp_proposal_tokens: int = 0
    enable_sfa_dcp_sharded_indexer: bool = False
    enable_flashcomm1: bool = False
    submodule_recipes: tuple[str, ...] = ()
    step_count_upper_bound: int = 1


# Input-only capture manifest mapping capture tags to catalog model IDs and run settings.
# ZERO expected kernel names live here: every expectation is lowered via compose_case -> spine.lower.
_OFFLINE_A3_TAG_CONFIGS: Mapping[str, OfflineCaptureCaseConfig] = {
    "deepseek_v32_dcp4_w4a8_c8_sp": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w4a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "deepseek_v32_mtp_spec": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_SINGLE,
        quant="bf16", sequences=_A3_DEFAULT_BATCH,
        mtp_proposal_tokens=1, enable_flashcomm1=True,
    ),
    "deepseek_v32_pcp2_dcp2_tp2": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_QUAD, dcp=_TP_PAIR, pcp=_TP_PAIR,
        quant="w4a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "deepseek_v41_flash_4layer_c2_swa": OfflineCaptureCaseConfig(
        "deepseek-v4.1-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "gemma4_26b_a4b_swa_moe": OfflineCaptureCaseConfig(
        "gemma-4-26b-a4b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "glm52_dcp4_w4a8_c8_sp": OfflineCaptureCaseConfig(
        "glm-5.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w4a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "glm53_flash_dcp1_w8a8_c8": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_flashcomm1=True,
    ),
    "kimi_k31_dcp4_sharded1_aclgraph": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_sfa_dcp_sharded_indexer=True,
    ),
    "kimi_k31_dcp4_sharded1_c8qli": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", kv_cache_dtype="int8", sequences=_A3_DEFAULT_BATCH,
        enable_sfa_dcp_sharded_indexer=True,
    ),
    "kimi_k3_dcp4_bf16kv_mlapo_mso": OfflineCaptureCaseConfig(
        "kimi-k3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD, dcp=_TP_QUAD,
        quant="w8a8", sequences=_A3_DEFAULT_BATCH,
    ),
    "minimax_m3_gqa_indexer": OfflineCaptureCaseConfig(
        "minimax-m3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "nemotron3_mamba2_attn": OfflineCaptureCaseConfig(None),
    "qwen35_moe_gdn_gqa": OfflineCaptureCaseConfig(
        "qwen3.5-moe", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen38_flash_gdn_gqa": OfflineCaptureCaseConfig(
        "qwen3.8-flash-next", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="bf16", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen3_235b_a22b_moe_mc2": OfflineCaptureCaseConfig(
        "qwen3-235b-a22b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=_TP_QUAD,
        quant="w8a8", sequences=_A3_DEFAULT_BATCH, enable_flashcomm1=True,
    ),
    "qwen3_32b_w8a8mix_c8_fixed": OfflineCaptureCaseConfig(
        "qwen3-32b", layers=_DEFAULT_LAYERS_MOE, tp=_TP_QUAD, ep=None,
        quant="w8a8", sequences=_A3_DEFAULT_BATCH,
    ),
}

_OFFLINE_A5_TAG_CONFIGS: Mapping[str, OfflineCaptureCaseConfig] = {
    "deepseek_v32": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", sequences=_A5_DEFAULT_BATCH,
    ),
    "deepseek_v32_aclgraph": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", sequences=_A5_DEFAULT_BATCH,
    ),
    "deepseek_v32_b3_sp": OfflineCaptureCaseConfig(
        "deepseek-v3.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", sequences=_A3_DEFAULT_BATCH,
    ),
    "deepseek_v41_c2_compressor_submod": OfflineCaptureCaseConfig(
        "deepseek-v4.1-flash", layers=_SUBMOD_LAYERS, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16", sequences=_A5_DEFAULT_BATCH,
        submodule_recipes=("aclnn_compressor", "compressed_store", "compress_pool", "index_compress_pool"),
    ),
    "gemma4_26b_a4b": OfflineCaptureCaseConfig(
        "gemma-4-26b-a4b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "gemma4_31b": OfflineCaptureCaseConfig(
        "gemma-4-31b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=None,
        quant="bf16",
    ),
    "gemma4_ple": OfflineCaptureCaseConfig(
        "gemma-4-31b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=None,
        quant="bf16",
    ),
    "glm_52": OfflineCaptureCaseConfig(
        "glm-5.2", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16",
    ),
    "glm_53_flash_hybrid_tp1": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "glm_53_flash_recover_tp1": OfflineCaptureCaseConfig(
        "glm-5.3-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16", mtp_proposal_tokens=1,
    ),
    "kimi_k31_dcp2_replicated_fp8": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="fp8",
    ),
    "kimi_k31_dcp2_replicated_fp8_aclgraph": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="fp8",
    ),
    "kimi_k31_fp8_tp1": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="fp8",
    ),
    "kimi_k31_sharded_dcp_submod": OfflineCaptureCaseConfig(
        "kimi-k3.1", layers=_SUBMOD_LAYERS, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16", enable_sfa_dcp_sharded_indexer=True,
        submodule_recipes=("candidate_pack", "candidate_merge"),
    ),
    "kimi_k3_dense_mla": OfflineCaptureCaseConfig(
        "kimi-k3", layers=_DEFAULT_LAYERS_MOE, tp=_TP_PAIR, ep=_TP_PAIR, dcp=_TP_PAIR,
        quant="bf16",
    ),
    "longcat_flash_base": OfflineCaptureCaseConfig(
        "longcat-flash-lite", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "mimo_v2": OfflineCaptureCaseConfig(None),
    "minimax_m31_flash": OfflineCaptureCaseConfig(
        "minimax-m3.1-flash", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "nemotron_3_mamba2": OfflineCaptureCaseConfig(None),
    "qwen35_moe": OfflineCaptureCaseConfig(
        "qwen3.5-moe", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "qwen38_flash_next": OfflineCaptureCaseConfig(
        "qwen3.8-flash-next", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "qwen3_235b": OfflineCaptureCaseConfig(
        "qwen3-235b-a22b", layers=_DEFAULT_LAYERS_DENSE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
    "seed_20": OfflineCaptureCaseConfig(None),
    "step5_preview": OfflineCaptureCaseConfig(
        "step-5-preview", layers=_DEFAULT_LAYERS_MOE, tp=_TP_SINGLE, ep=_TP_SINGLE,
        quant="bf16",
    ),
}


def _aggregate_step_nodes_for_offline_audit(
    step_nodes: Sequence[PredictedKernelNode],
    *,
    tag: str,
    rank_id: int,
    step_upper_bound: int,
) -> tuple[PredictedKernelNode, ...]:
    """Collapse per-rank lowered step DAG into tightly bounded ``[min_count, max_count]`` nodes."""
    rank_nodes = [n for n in step_nodes if n.rank_id == rank_id]
    if not rank_nodes:
        raise ModelingError(f"offline_parity_empty_rank_nodes:{tag}:rank={rank_id}")
    grouped: dict[
        tuple[tuple[str, ...], tuple[str, ...], str, bool, str, tuple[str, ...]],
        list[PredictedKernelNode],
    ] = {}
    for node in rank_nodes:
        key = (
            node.expected_kernels,
            node.expected_task_types,
            node.task_type,
            node.is_ai_cpu_metadata,
            node.input_shapes,
            node.expected_input_shapes,
        )
        grouped.setdefault(key, []).append(node)
    if not grouped:
        raise ModelingError(f"offline_parity_empty_expected_inventory:{tag}")

    out: list[PredictedKernelNode] = []
    has_sp_pad = False
    for idx, ((kernels, exp_tts, ttype, is_aicpu, in_shapes, exp_in_shapes), items) in enumerate(sorted(grouped.items())):
        if kernels == _PAD_MULTI_KERNEL or "PadV3" in kernels:
            has_sp_pad = True
        per_step_count = len(items)
        first = items[0]
        out.append(PredictedKernelNode(
            node_id=f"{tag}.agg{idx}.{first.api_or_symbol}",
            rank_id=rank_id,
            stream_lane=first.stream_lane,
            task_type=ttype,
            api_or_symbol=first.api_or_symbol,
            expected_kernels=kernels,
            expected_task_types=exp_tts,
            input_shapes=in_shapes,
            is_ai_cpu_metadata=is_aicpu,
            min_count=per_step_count,
            max_count=per_step_count * max(1, int(step_upper_bound)),
            expected_input_shapes=exp_in_shapes,
        ))
    if not has_sp_pad:
        out.append(PredictedKernelNode(
            node_id=f"{tag}.zero_sp_pad_guard",
            rank_id=rank_id,
            stream_lane="main",
            task_type="AI_VECTOR_CORE",
            api_or_symbol="aclnnConstantPadNd",
            expected_kernels=("PadV3",),
            min_count=0,
            max_count=0,
        ))
    return tuple(out)


def _first_int_field(mapping: Mapping[str, Any] | None, keys: tuple[str, ...]) -> int | None:
    if not mapping:
        return None
    for key in keys:
        val = mapping.get(key)
        if val is not None and not isinstance(val, bool):
            return int(val)
    return None


def _resolve_offline_step_geometry(
    cfg: OfflineCaptureCaseConfig,
    *,
    phase: PhaseKind,
    capture_config: Mapping[str, Any] | None,
) -> tuple[int, int, int]:
    """Resolve independent ``(tokens_per_sequence, context_len, sequences)`` for offline lowering."""
    sequences = _first_int_field(capture_config, ("sequences_per_replica", "sequences", "batch_size"))
    if sequences is None:
        sequences = int(cfg.sequences)
    prompt_len = _first_int_field(capture_config, ("prompt_len", "prompt_tokens"))
    explicit_context = _first_int_field(capture_config, ("context_tokens", "context_len", "context", "prefix_tokens"))
    explicit_tokens = _first_int_field(capture_config, ("tokens_per_sequence", "query_tokens", "tokens"))
    if phase == "prefill":
        # Cold prefill starts with 0 prior cached KV; RequestView.kv_len = q_len + context.
        context = explicit_context if explicit_context is not None else 0
        if explicit_tokens is not None:
            tokens = explicit_tokens
        else:
            tokens = prompt_len if prompt_len is not None else (
                int(cfg.tokens) if cfg.phase == "prefill" else int(cfg.context)
            )
    else:
        context = (
            explicit_context
            if explicit_context is not None
            else (prompt_len if prompt_len is not None else int(cfg.context))
        )
        if explicit_tokens is not None:
            tokens = explicit_tokens
        else:
            tokens = int(cfg.tokens) + max(0, int(cfg.mtp_proposal_tokens))
    return tokens, context, sequences


def _compose_offline_case(
    cfg: OfflineCaptureCaseConfig,
    *,
    device_arch: str,
    capture_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an independent simulator Case dict strictly from declared experiment config."""
    assert cfg.model_id is not None
    sku = _A5_SKU if device_arch == "A5" else _A3_SKU
    layers = int((capture_config or {}).get("layers") or cfg.layers)
    tp = int((capture_config or {}).get("tp") or cfg.tp)
    dcp = int((capture_config or {}).get("dcp") or cfg.dcp)
    raw_ep = (capture_config or {}).get("ep")
    ep = (tp * cfg.dp * cfg.pcp) if raw_ep is True else (None if raw_ep is False else cfg.ep)
    kwargs: dict[str, Any] = {
        "tp": tp,
        "dp": cfg.dp,
        "layers": layers,
        "quant": cfg.quant,
        "kv_cache_dtype": cfg.kv_cache_dtype,
        "prompt_tokens": int((capture_config or {}).get("prompt_len") or cfg.context),
        "output_tokens": _DEFAULT_OUTPUT_TOKENS,
    }
    if ep is not None:
        kwargs["ep"] = ep
    if dcp > 1:
        kwargs["dcp"] = dcp
    if cfg.pcp > 1:
        kwargs["pcp"] = cfg.pcp
    if cfg.mtp_proposal_tokens > 0:
        kwargs["mtp_proposal_tokens"] = cfg.mtp_proposal_tokens
    case_dict = compose_case(cfg.model_id, sku, **kwargs)
    rt_meta = case_dict.setdefault("runtime", {}).setdefault("metadata", {})
    add_cfg = rt_meta.setdefault("additional_config", {})
    if cfg.enable_flashcomm1:
        add_cfg["enable_flashcomm1"] = True
    if cfg.enable_sfa_dcp_sharded_indexer:
        rt_meta["enable_sfa_dcp_sharded_indexer"] = True
    return case_dict


def build_offline_expected_nodes(
    tag: str,
    *,
    device_arch: str,
    rank_id: int = 0,
    capture_phase: PhaseKind | None = None,
    capture_config: Mapping[str, Any] | None = None,
    registry: ProbedAbiRegistry | None = None,
) -> tuple[PredictedKernelNode, ...] | None:
    """Lower ``compose_case -> PredictionSpine.lower -> PreparedStep`` for an offline capture tag.

    Returns ``None`` honestly when ``tag`` belongs to an architecture absent from
    ``mindie_sim/catalog/models/`` (``unmodeled_catalog_model``).
    """
    reg = registry if registry is not None else default_abi_registry()
    table = _OFFLINE_A5_TAG_CONFIGS if device_arch == "A5" else _OFFLINE_A3_TAG_CONFIGS
    cfg = table.get(tag)
    if cfg is None:
        raise ModelingError(f"offline_parity_undeclared_capture_tag:{device_arch}:{tag}")
    if cfg.model_id is None:
        return None
    case_dict = _compose_offline_case(cfg, device_arch=device_arch, capture_config=capture_config)
    eff_phase: PhaseKind = capture_phase if capture_phase is not None else cfg.phase
    tokens, context, sequences = _resolve_offline_step_geometry(
        cfg, phase=eff_phase, capture_config=capture_config,
    )
    prepared, full_preds = build_predicted_nodes_from_case(
        case_dict,
        phase=eff_phase,
        tokens=tokens,
        context=context,
        sequences=sequences,
        device_arch=device_arch,
        registry=reg,
    )
    if cfg.submodule_recipes:
        keep_ids = {
            op.op_id
            for entry in prepared.graph.executable
            for op in entry.ops
            if op.recipe in cfg.submodule_recipes
        }
        full_preds = tuple(p for p in full_preds if p.node_id in keep_ids)
        if not full_preds:
            raise ModelingError(
                f"offline_parity_submodule_filter_empty:{tag}:{cfg.submodule_recipes}"
            )
    return _aggregate_step_nodes_for_offline_audit(
        full_preds,
        tag=tag,
        rank_id=rank_id,
        step_upper_bound=cfg.step_count_upper_bound,
    )


def _audit_offline_arch(
    *,
    arch: str,
    sku: str,
    va_rev: str,
    dags: Mapping[str, Any],
    raw_captures: Mapping[str, Mapping[str, Any]],
    checker: SubgraphParityChecker,
    registry: ProbedAbiRegistry,
) -> tuple[list[LiveSubgraphParityReport], list[dict[str, Any]]]:
    reports: list[LiveSubgraphParityReport] = []
    unmodeled: list[dict[str, Any]] = []
    for tag, dag in sorted(dags.items()):
        cap_cfg = (raw_captures.get(tag) or {}).get("config")
        preds = build_offline_expected_nodes(
            tag,
            device_arch=arch,
            rank_id=dag.rank_id,
            capture_phase=dag.phase,
            capture_config=cap_cfg if isinstance(cap_cfg, Mapping) else None,
            registry=registry,
        )
        if preds is None:
            unmodeled.append({
                "signature_id": f"{arch.lower()}:{tag}",
                "sku": sku,
                "status": "unmodeled_catalog_model",
                "npu_kernels": dag.npu_device_kernel_count,
            })
            continue
        rep = checker.compare(
            signature_id=f"{arch.lower()}:{tag}",
            hardware_sku=sku,
            device_arch=arch,
            config_payload={"arch": arch, "tag": tag, "phase": dag.phase, "exec_mode": dag.exec_mode},
            observed_dags=(dag,),
            predicted_nodes=preds,
            vllm_ascend_rev=va_rev,
            verification_mode="offline_inventory_audit",
        )
        reports.append(rep)
    return reports, unmodeled


def evaluate_offline_bundle(
    *,
    a3_path: Path | None,
    a5_path: Path | None,
    registry: ProbedAbiRegistry | None = None,
) -> dict[str, Any]:
    """Run honest full-lowered offline inventory parity audits across A3 and A5 bundles."""
    reg = registry if registry is not None else default_abi_registry()
    checker = SubgraphParityChecker(reg)
    reports: list[LiveSubgraphParityReport] = []
    unmodeled_models: list[dict[str, Any]] = []
    blocked_summary: dict[str, Any] = {}

    if a3_path is not None:
        raw_a3 = json.loads(Path(a3_path).read_text(encoding="utf-8")).get("captures") or {}
        va_rev, dags, blocked = load_offline_a3_capture_dags(a3_path)
        blocked_summary = {
            tag: {"missing_so": list(so), "missing_py": list(py)}
            for tag, (so, py) in blocked.items()
        }
        r3, u3 = _audit_offline_arch(
            arch="A3", sku=_A3_SKU, va_rev=va_rev, dags=dags,
            raw_captures=raw_a3, checker=checker, registry=reg,
        )
        reports.extend(r3)
        unmodeled_models.extend(u3)

    if a5_path is not None:
        raw_a5 = json.loads(Path(a5_path).read_text(encoding="utf-8")).get("captures") or {}
        va_rev, dags = load_offline_a5_capture_dags(a5_path)
        r5, u5 = _audit_offline_arch(
            arch="A5", sku=_A5_SKU, va_rev=va_rev, dags=dags,
            raw_captures=raw_a5, checker=checker, registry=reg,
        )
        reports.extend(r5)
        unmodeled_models.extend(u5)

    passed_count = sum(1 for r in reports if r.passed)
    failed = [r.to_dict() for r in reports if not r.passed]
    return {
        "verification_mode": "offline_inventory_audit",
        "causal_parity_certified": False,
        "total_signatures": len(reports),
        "passed_signatures": passed_count,
        "failed_signatures": len(failed),
        "unmodeled_catalog_signatures": len(unmodeled_models),
        "all_passed": len(reports) > 0 and len(failed) == 0,
        "unmodeled_catalog_models": unmodeled_models,
        "blocked_container_classification": blocked_summary,
        "failed": failed,
        "signatures": [
            {
                "signature_id": r.signature_id,
                "sku": r.hardware_sku,
                "phase": r.phase,
                "exec_mode": r.exec_mode,
                "npu_kernels": r.npu_device_kernel_count,
                "attr_rate": round(r.exact_causal_attribution_rate, 4),
                "causal_parity_certified": r.causal_parity_certified,
                "passed": r.passed,
                "diff_count": len(r.all_diffs()),
            }
            for r in reports
        ],
    }


def _rank_id_from_trace_path(trace_path: Path, live_dir: Path, fallback_idx: int) -> int:
    """Parse explicit rank ID from path components (e.g. ``rank17/trace_view.json``)."""
    try:
        rel_parts = (live_dir.name, *trace_path.relative_to(live_dir).parts)
    except ValueError:
        rel_parts = trace_path.parts[-3:]
    for part in reversed(rel_parts):
        match = _RANK_PATH_RE.search(part)
        if match is not None:
            return int(match.group(1))
    return fallback_idx


def _accumulate_live_rank_outcome(
    rank_idx: int,
    outcome: StreamedRankParityOutcome,
    *,
    has_expected_preds: bool,
    rank_preds: Sequence[PredictedKernelNode],
    min_attribution_rate: float,
    dag_diffs: list[ParityDiff],
    comm_diffs: list[ParityDiff],
    lane_diffs: list[ParityDiff],
    exp_diffs: list[ParityDiff],
    shape_diffs: list[ParityDiff],
    launch_diffs: list[ParityDiff],
) -> bool:
    """Record per-rank execution, attribution, and parity diffs; return whether rank executed NPU kernels."""
    rank_executed = outcome.npu_device_kernel_count > 0
    if not rank_executed:
        launch_diffs.append(ParityDiff(
            code="MISSING_OR_EXTRA_LAUNCH",
            rank_id=rank_idx,
            node_id="rank_execution",
            detail=f"rank {rank_idx} recorded zero real NPU device kernels",
            expected=">=1 real NPU kernels",
            observed="0",
        ))
    if outcome.ambiguous_matches:
        launch_diffs.append(ParityDiff(
            code="UNATTRIBUTED_KERNEL",
            rank_id=rank_idx,
            node_id="ambiguous_proximity",
            detail=f"rank {rank_idx} has {len(outcome.ambiguous_matches)} ambiguous proximity match(es)",
            expected="0 ambiguous_matches",
            observed=str(len(outcome.ambiguous_matches)),
        ))
    if rank_executed and outcome.exact_causal_attribution_rate < min_attribution_rate:
        launch_diffs.append(ParityDiff(
            code="UNATTRIBUTED_KERNEL",
            rank_id=rank_idx,
            node_id="attribution",
            detail=(
                f"rank {rank_idx} exact causal attribution rate "
                f"{outcome.exact_causal_attribution_rate:.3f} < {min_attribution_rate:.3f}"
            ),
            expected=f">={min_attribution_rate:.3f}",
            observed=f"{outcome.exact_causal_attribution_rate:.3f}",
        ))
    if has_expected_preds and not rank_preds:
        launch_diffs.append(ParityDiff(
            code="MISSING_OR_EXTRA_LAUNCH",
            rank_id=rank_idx,
            node_id="predicted_graph",
            detail=f"participating rank {rank_idx} has no predicted subgraph",
            expected=f"predicted nodes for rank {rank_idx}",
            observed="0",
        ))
    else:
        dag_diffs.extend(outcome.dag_diffs)
        comm_diffs.extend(outcome.comm_diffs)
        lane_diffs.extend(outcome.lane_diffs)
        exp_diffs.extend(outcome.exp_diffs)
        shape_diffs.extend(outcome.shape_diffs)
        launch_diffs.extend(outcome.launch_diffs)
    return rank_executed


def _build_live_parity_report(
    live_dir: Path,
    *,
    hardware_sku: str,
    graph_id: str,
    phase: PhaseKind,
    exec_mode: ExecModeKind,
    observed_ranks: Sequence[int],
    by_rank_pred: Mapping[int, Sequence[PredictedKernelNode]],
    total_npu: int,
    total_nodes: int,
    weighted_exact: float,
    all_ranks_executed: bool,
    all_ambiguous: Sequence[str],
    all_unattributed: Sequence[str],
    dag_diffs: list[ParityDiff],
    comm_diffs: list[ParityDiff],
    lane_diffs: list[ParityDiff],
    exp_diffs: list[ParityDiff],
    shape_diffs: list[ParityDiff],
    launch_diffs: list[ParityDiff],
    error_messages: Sequence[str],
) -> dict[str, Any]:
    """Assemble the serialized live parity report after streaming all observed rank traces."""
    observed_rank_set = set(observed_ranks)
    for pred_rank, nodes in by_rank_pred.items():
        if pred_rank not in observed_rank_set:
            for pnode in nodes:
                comm_diffs.append(ParityDiff(
                    code="WRONG_RANK_COLLECTIVE",
                    rank_id=pred_rank,
                    node_id=pnode.node_id,
                    detail=f"predicted rank {pred_rank} absent from observed ranks {tuple(observed_ranks)}",
                    expected=str(pred_rank),
                    observed=str(tuple(observed_ranks)),
                ))

    missing_so, missing_py = classify_runtime_errors(error_messages)
    has_diffs = bool(
        dag_diffs or comm_diffs or lane_diffs or exp_diffs
        or shape_diffs or launch_diffs or missing_so or missing_py
    )
    passed = (total_npu > 0) and all_ranks_executed and not has_diffs
    weighted_rate = (weighted_exact / total_nodes) if total_nodes > 0 else 0.0
    cfg_payload = {"live_dir": live_dir.name, "ranks": len(observed_ranks), "graph_id": graph_id}
    config_digest = hashlib.sha256(
        json.dumps(cfg_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return LiveSubgraphParityReport(
        signature_id=f"live:{live_dir.name}",
        hardware_sku=hardware_sku,
        config_digest=config_digest,
        world_ranks=tuple(observed_ranks),
        phase=phase,
        exec_mode=exec_mode,
        vllm_ascend_rev="",
        sim_rev=graph_id,
        npu_device_kernel_count=total_npu,
        exact_causal_attribution_rate=weighted_rate,
        ambiguous_matches=tuple(all_ambiguous),
        unattributed_device_kernels=tuple(all_unattributed),
        missing_so_symbols=missing_so,
        missing_python_modules=missing_py,
        dag_order_diffs=tuple(dag_diffs),
        rank_collective_diffs=tuple(comm_diffs),
        stream_lane_diffs=tuple(lane_diffs),
        expansion_diffs=tuple(exp_diffs),
        shape_diffs=tuple(shape_diffs),
        launch_diffs=tuple(launch_diffs),
        passed=passed,
        verification_mode="causal_dag_parity",
        causal_parity_certified=passed and not all_ambiguous and not all_unattributed,
    ).to_dict()


def evaluate_live_dir(
    live_dir: Path,
    *,
    device_arch: str | None = None,
    hardware_sku: str | None = None,
    case: Case | Mapping[str, Any] | Path | None = None,
    predicted_nodes: Sequence[PredictedKernelNode] | None = None,
    phase: PhaseKind = "decode",
    tokens: int = 1,
    context: int = 1024,
    sequences: int = 1,
    exec_mode: ExecModeKind = "eager",
    min_attribution_rate: float = DEFAULT_MIN_ATTRIBUTION_RATE,
    error_messages: Sequence[str] = (),
    registry: ProbedAbiRegistry | None = None,
) -> dict[str, Any]:
    """Stream one rank trace at a time in bounded memory against independent predictions."""
    reg = registry if registry is not None else default_abi_registry()
    parsed_case: Case | None = None
    if case is not None:
        if isinstance(case, Case):
            parsed_case = case
        elif isinstance(case, Path):
            parsed_case = parse_case(_read_case_file(case))
        else:
            parsed_case = parse_case(dict(case))
    resolved_arch, resolved_sku, dev_err = _resolve_live_target_device(
        parsed_case, device_arch=device_arch, hardware_sku=hardware_sku,
    )
    if dev_err is not None:
        raise ModelingError(f"live_parity_device_conflict:{dev_err}")
    if predicted_nodes is not None:
        expected_preds = tuple(predicted_nodes)
        graph_id = ""
    elif parsed_case is not None:
        prepared, expected_preds = build_predicted_nodes_from_case(
            parsed_case,
            phase=phase,
            tokens=tokens,
            context=context,
            sequences=sequences,
            device_arch=resolved_arch,
            registry=reg,
        )
        graph_id = prepared.graph_id
    else:
        raise ModelingError(
            "live_parity_requires_independent_predictions: supply --case or explicit predicted_nodes"
        )

    traces = sorted(live_dir.rglob("trace_view.json"))
    if not traces:
        raise FileNotFoundError(f"no trace_view.json found under {live_dir}")

    by_rank_pred: dict[int, list[PredictedKernelNode]] = {}
    for pred in expected_preds:
        by_rank_pred.setdefault(pred.rank_id, []).append(pred)

    dag_diffs: list[ParityDiff] = []
    comm_diffs: list[ParityDiff] = []
    lane_diffs: list[ParityDiff] = []
    exp_diffs: list[ParityDiff] = []
    shape_diffs: list[ParityDiff] = []
    launch_diffs: list[ParityDiff] = []
    all_ambiguous: list[str] = []
    all_unattributed: list[str] = []
    observed_ranks: list[int] = []
    total_npu = 0
    total_nodes = 0
    weighted_exact = 0.0
    all_ranks_executed = True

    if not expected_preds:
        launch_diffs.append(ParityDiff(
            code="MISSING_OR_EXTRA_LAUNCH",
            rank_id=0,
            node_id="predicted_graph",
            detail="empty predicted graph: independent predictions required for causal parity",
            expected=">=1 predicted nodes",
            observed="0",
        ))

    seen_rank_paths: dict[int, Path] = {}
    for enum_idx, trace_path in enumerate(traces):
        rank_idx = _rank_id_from_trace_path(trace_path, live_dir, enum_idx)
        if rank_idx in seen_rank_paths:
            raise ModelingError(
                f"duplicate_rank_trace_detected:rank={rank_idx}:"
                f"{trace_path.name}:{seen_rank_paths[rank_idx].name}"
            )
        seen_rank_paths[rank_idx] = trace_path
        rank_preds = by_rank_pred.get(rank_idx, ())
        outcome = evaluate_trace_view_stream(
            trace_path,
            rank_preds,
            rank_id=rank_idx,
            phase=phase,
            exec_mode=exec_mode,
            device_arch=resolved_arch,
            registry=reg,
        )
        observed_ranks.append(rank_idx)
        total_npu += outcome.npu_device_kernel_count
        total_nodes += outcome.total_events
        weighted_exact += outcome.exact_causal_attribution_rate * outcome.total_events
        all_ambiguous.extend(outcome.ambiguous_matches)
        all_unattributed.extend(outcome.unattributed_device_kernels)
        if not _accumulate_live_rank_outcome(
            rank_idx,
            outcome,
            has_expected_preds=bool(expected_preds),
            rank_preds=rank_preds,
            min_attribution_rate=min_attribution_rate,
            dag_diffs=dag_diffs,
            comm_diffs=comm_diffs,
            lane_diffs=lane_diffs,
            exp_diffs=exp_diffs,
            shape_diffs=shape_diffs,
            launch_diffs=launch_diffs,
        ):
            all_ranks_executed = False

    return _build_live_parity_report(
        live_dir,
        hardware_sku=resolved_sku,
        graph_id=graph_id,
        phase=phase,
        exec_mode=exec_mode,
        observed_ranks=observed_ranks,
        by_rank_pred=by_rank_pred,
        total_npu=total_npu,
        total_nodes=total_nodes,
        weighted_exact=weighted_exact,
        all_ranks_executed=all_ranks_executed,
        all_ambiguous=all_ambiguous,
        all_unattributed=all_unattributed,
        dag_diffs=dag_diffs,
        comm_diffs=comm_diffs,
        lane_diffs=lane_diffs,
        exp_diffs=exp_diffs,
        shape_diffs=shape_diffs,
        launch_diffs=launch_diffs,
        error_messages=error_messages,
    )


def _build_parity_arg_parser(prog: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog,
        description="Run causal live or offline Ascend NPU subgraph parity verification.",
    )
    parser.add_argument("--offline-a3", type=Path, default=None, help="Sanitized A3 capture JSON path")
    parser.add_argument("--offline-a5", type=Path, default=None, help="Sanitized A5 capture JSON path")
    parser.add_argument("--live-dir", type=Path, default=None, help="Live Ascend profiler directory")
    parser.add_argument("--case", type=Path, default=None, help="Case JSON lowered independently before reading live traces")
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--context", type=int, default=1024)
    parser.add_argument("--sequences", type=int, default=1)
    parser.add_argument("--exec-mode", choices=("eager", "aclgraph"), default="eager")
    parser.add_argument("--arch", choices=("A2", "A3", "A5"), default=None,
                        help="Target Ascend architecture (defaults to parsed --case arch)")
    parser.add_argument("--sku", default=None,
                        help="Target Ascend SKU (defaults to parsed --case SKU or arch default)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Optional JSON report path; prints to stdout when omitted")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    prog: str = "python -m mindie_sim.observations.subgraph_parity",
) -> int:
    """Execute packaged live or offline causal subgraph parity verification."""
    parser = _build_parity_arg_parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.live_dir is not None:
            if args.case is None:
                parser.error("--live-dir requires --case for independent predictions")
            parsed_case = parse_case(_read_case_file(args.case))
            device_arch, hardware_sku, dev_err = _resolve_live_target_device(
                parsed_case, device_arch=args.arch, hardware_sku=args.sku,
            )
            if dev_err is not None:
                parser.error(dev_err)
            result = evaluate_live_dir(
                args.live_dir,
                device_arch=device_arch,
                hardware_sku=hardware_sku,
                case=parsed_case,
                phase=args.phase,
                tokens=args.tokens,
                context=args.context,
                sequences=args.sequences,
                exec_mode=args.exec_mode,
            )
            passed = bool(result["passed"])
        else:
            if args.offline_a3 is None and args.offline_a5 is None:
                parser.error("supply --offline-a3/--offline-a5 or --live-dir")
            result = evaluate_offline_bundle(a3_path=args.offline_a3, a5_path=args.offline_a5)
            passed = bool(result["all_passed"])
        text = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.write_text(text, encoding="utf-8")
        else:
            print(text, end="")
    except TimeoutError:
        raise
    except (ModelingError, OSError, ValueError) as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}), file=sys.stderr)
        return 2
    return 0 if passed else 1


__all__ = [
    "AI_CPU_TASK_TYPE",
    "AttributionKind",
    "DEFAULT_MIN_ATTRIBUTION_RATE",
    "LiveSubgraphParityReport",
    "ObservedKernelNode",
    "ObservedRankDAG",
    "OfflineCaptureCaseConfig",
    "PROXIMITY_AMBIGUITY_WINDOW_US",
    "ParityDiff",
    "ParityDiffCode",
    "PredictedKernelNode",
    "REAL_NPU_TASK_TYPES",
    "StreamedRankParityOutcome",
    "SubgraphParityChecker",
    "VerificationMode",
    "build_offline_expected_nodes",
    "build_predicted_nodes_from_case",
    "build_predicted_nodes_from_prepared_step",
    "classify_runtime_errors",
    "correlate_trace_view_events",
    "evaluate_live_dir",
    "evaluate_offline_bundle",
    "evaluate_trace_view_stream",
    "iter_trace_view_events",
    "load_offline_a3_capture_dags",
    "load_offline_a5_capture_dags",
    "load_trace_view_file",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
