"""Streaming 1:1 Ascend ``msprof`` ``trace_view.json`` exporter for MindStudio Insight.

[Role] Stream Chrome Trace JSON array events matching the native Ascend
  ``msprof`` / ``torch_npu.profiler`` schema directly to disk in O(Archetypes)
  bounded memory:
  - ``PID 0: Python / vLLM-Ascend Framework (Host)`` with 4 top-level tracks:
    ``vLLM Engine & ModelRunner``, ``Model Layer Scope (MSTX)`` (full multi-layer
    Target + Draft schedule), ``Submodule Scope (MSTX)``, and ``CANN Runtime /
    Host Launch`` (``aclGraphLaunch`` / ``aclrtLaunchKernel`` + ``HostToDevice``
    flow arrows);
  - ``Ascend Hardware`` (``Stream 0/1/2`` named with real Ascend/CANN/Triton/
    TileLang/AscendC kernels + physically grounded ``Cube (AIC)``, ``Vector (AIV)``,
    ``MTE2``, ``MTE3``, ``Fixpipe`` pipe durations and ``aic_mac_ratio`` /
    ``aiv_vec_ratio`` / ``mte2_ratio`` / ``mte3_ratio`` / ``fixpipe_ratio``);
  - ``HCCL`` (Planes 0/1/2 named with ``hcom_*`` / MC2 kernels);
  - ``Overlap Analysis`` (``Computing``, ``Communication``,
    ``Communication(Not Overlapped)``, ``Free``);
  - ``Ascend Metrics`` (sampled ``HBM Bandwidth``, ``Cube Utilization``,
    ``Vector Utilization``, ``Active HBM Memory`` counter series).
[Invariants] Root document is a valid JSON array ``[...]`` with Unix LF endings
  and ``allow_nan=False``. Zero arbitrary ``0.75/0.15/0.85`` pipe-fraction
  heuristics and zero banned internal Python recipe class names. Full multi-layer
  models (e.g. 93 Target + 5 Draft layers) stream lazily from compact archetype
  templates without materializing all events in RAM (peak delta ``<< 64 MiB``).
  Every ``ph="s"`` flow event has an exact matching ``ph="f"`` (``bp="e"``) event
  with identical ``id`` and causal ``dst_ts >= src_ts``.
[Timeouts & Locks] Pure single-pass streaming generator; atomic write via
  same-directory temp file + ``os.replace``; no locks or device access.
[Side-Effect Policy] Writes only the caller-requested output file path and cleans
  up its private temp file on error.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest
from .bundle import build_export_bundle, compose_export_case
from .models import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    GIB_BYTES,
    MIN_POSITIVE_SECONDS,
    US_PER_SECOND,
    ExportBundle,
    OperatorExportRow,
    TraceScheduleEntry,
)
from .public_view import format_track_label

#: Schema tag for trace export receipts.
TRACE_EXPORT_RECEIPT_SCHEMA = "mindie-sim.msprof-trace-receipt/2"
#: Stream buffer size (64 KiB) for bounded low-RSS JSON emission.
STREAM_BUFFER_BYTES = 64 * 1024
#: Timestamp decimal precision in microseconds (3 decimal places = 1 ns).
TIMESTAMP_US_DECIMALS = 3
#: Metric rounding decimals for compact readable ``msinsight`` args.
METRIC_DECIMALS = 3
#: Ratio rounding decimals for physical pipe & Gap Radar ratios.
RATIO_DECIMALS = 4
#: Maximum counter sample points emitted per series to keep large traces bounded.
MAX_COUNTER_SAMPLES = 512
#: Bytes per decimal GB for GB/s bandwidth counter reporting.
BYTES_PER_GB = 1_000_000_000.0
#: Percentage scale factor (100.0%) and midpoint ratio for per-layer counter sampling.
PCT_SCALE = 100.0
MIDPOINT_FRACTION = 0.5

#: Top-level Host Framework process ID (always sorted above Ascend Hardware).
PID_FRAMEWORK = 0
TID_FW_ENGINE = 0
TID_FW_LAYER_MSTX = 1
TID_FW_SUBMODULE_MSTX = 2
TID_FW_HOST_LAUNCH = 3

#: Per-die PID block stride guaranteeing disjoint Hardware / HCCL / Overlap / Metrics PIDs.
DIE_PID_STRIDE = 100
PID_OFFSET_HARDWARE = 1
PID_OFFSET_HCCL = 2
PID_OFFSET_OVERLAP = 3
PID_OFFSET_METRICS = 4

#: Stream thread IDs inside the ``Ascend Hardware`` process.
TID_STREAM_COMPUTE = 0
TID_STREAM_MSO_AUX = 1
TID_STREAM_COMM_CSA = 2
#: Physical pipe sub-lane thread IDs inside the ``Ascend Hardware`` process.
TID_PIPE_CUBE = 10
TID_PIPE_VECTOR = 11
TID_PIPE_MTE2 = 12
TID_PIPE_MTE3 = 13
TID_PIPE_FIXPIPE = 14

#: Plane thread IDs inside the ``HCCL`` process.
TID_HCCL_COLLECTIVE = 0
TID_HCCL_EP_MOE = 1
TID_HCCL_TP_SP_CP = 2

#: Lane thread IDs inside the ``Overlap Analysis`` process.
TID_OVERLAP_COMPUTING = 0
TID_OVERLAP_COMM = 1
TID_OVERLAP_COMM_NOT_OVERLAPPED = 2
TID_OVERLAP_FREE = 3

#: Counter thread ID inside the ``Ascend Metrics`` process.
TID_METRICS_COUNTER = 0

#: Minimum visible sub-pipe span (1 ns = 0.001 us).
PIPE_MIN_SUBSPAN_US = 0.001

#: Canonical ``msprof`` process and thread specification exported for introspection.
MSPROF_PROCESS_SPEC: dict[str, tuple[tuple[int, str], ...]] = {
    "Python / vLLM-Ascend Framework (Host)": (
        (TID_FW_ENGINE, "vLLM Engine & ModelRunner"),
        (TID_FW_LAYER_MSTX, "Model Layer Scope (MSTX)"),
        (TID_FW_SUBMODULE_MSTX, "Submodule Scope (MSTX)"),
        (TID_FW_HOST_LAUNCH, "CANN Runtime / Host Launch"),
    ),
    "Ascend Hardware": (
        (TID_STREAM_COMPUTE, "Stream 0 (Compute)"),
        (TID_STREAM_MSO_AUX, "Stream 1 (MSO / Aux 2nd)"),
        (TID_STREAM_COMM_CSA, "Stream 2 (Comm / CSA 3rd)"),
        (TID_PIPE_CUBE, "Cube (AIC)"),
        (TID_PIPE_VECTOR, "Vector (AIV)"),
        (TID_PIPE_MTE2, "MTE2 (GM->L1/UB)"),
        (TID_PIPE_MTE3, "MTE3 (UB->GM)"),
        (TID_PIPE_FIXPIPE, "Fixpipe (L0C->GM/UB)"),
    ),
    "HCCL": (
        (TID_HCCL_COLLECTIVE, "HCCL Plane 0 (Collective)"),
        (TID_HCCL_EP_MOE, "HCCL Plane 1 (EP Dispatch/Combine)"),
        (TID_HCCL_TP_SP_CP, "HCCL Plane 2 (TP / SP / CP)"),
    ),
    "Overlap Analysis": (
        (TID_OVERLAP_COMPUTING, "Computing"),
        (TID_OVERLAP_COMM, "Communication"),
        (TID_OVERLAP_COMM_NOT_OVERLAPPED, "Communication(Not Overlapped)"),
        (TID_OVERLAP_FREE, "Free"),
    ),
    "Ascend Metrics": (
        (TID_METRICS_COUNTER, "NPU Utilization & HBM Counters"),
    ),
}


@dataclass(frozen=True, slots=True)
class TraceExportSummary:
    """Compact receipt returned after streaming a ``trace_view.json`` file."""

    schema: str
    output_path: str
    model_name: str
    hardware_sku: str
    phase: str
    graph_id: str
    total_events: int
    metadata_events: int
    complete_events: int
    flow_pairs: int
    counter_events: int
    step_span_us: float
    step_l_star_lean_us: float
    bytes_written: int
    replayed_layers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "output_path": self.output_path,
            "model_name": self.model_name,
            "hardware_sku": self.hardware_sku,
            "phase": self.phase,
            "graph_id": self.graph_id,
            "total_events": self.total_events,
            "metadata_events": self.metadata_events,
            "complete_events": self.complete_events,
            "flow_pairs": self.flow_pairs,
            "counter_events": self.counter_events,
            "step_span_us": round(self.step_span_us, TIMESTAMP_US_DECIMALS),
            "step_l_star_lean_us": round(self.step_l_star_lean_us, TIMESTAMP_US_DECIMALS),
            "bytes_written": self.bytes_written,
            "replayed_layers": self.replayed_layers,
        }


@dataclass(frozen=True, slots=True)
class _ArchetypeTemplate:
    archetype_id: str
    ops: tuple[OperatorExportRow, ...]
    min_start_us: float
    span_us: float
    sub_spans: tuple[tuple[str, float, float], ...]
    comp_intervals: tuple[tuple[float, float], ...]
    comm_intervals: tuple[tuple[float, float], ...]


def _round_us(value: float) -> float:
    return round(max(0.0, float(value)), TIMESTAMP_US_DECIMALS)


def _round_metric(value: float) -> float:
    return round(float(value), METRIC_DECIMALS)


def _die_coordinates(bundle: ExportBundle) -> tuple[tuple[int, int, int], ...]:
    coords = sorted({
        (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        for op in bundle.operators
    })
    return tuple(coords) if coords else ((0, 0, 0),)


def _pid_base(die_index: int) -> int:
    return int(die_index) * DIE_PID_STRIDE


def _hccl_tid(op: OperatorExportRow) -> int:
    lower = f"{op.op_id} {op.family} {op.recipe} {op.ascend_kernel}".lower()
    if any(tok in lower for tok in ("alltoall", "dispatch", "combine", "expert", "moe", "mc2")):
        return TID_HCCL_EP_MOE
    if any(tok in lower for tok in ("allreduce", "allgather", "reducescatter", "tp", "sp", "cp")):
        return TID_HCCL_TP_SP_CP
    return TID_HCCL_COLLECTIVE


def _op_pid_tid(op: OperatorExportRow, base_pid: int) -> tuple[int, int]:
    if op.task_type == "HCCL":
        return base_pid + PID_OFFSET_HCCL, _hccl_tid(op)
    if op.stream_index == 1:
        return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_MSO_AUX
    if op.stream_index == 2:
        return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_COMM_CSA
    return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_COMPUTE


def _iter_metadata_events(
    coords: Sequence[tuple[int, int, int]],
) -> Iterator[dict[str, Any]]:
    fw_name = "Python / vLLM-Ascend Framework (Host)"
    yield {"ph": "M", "name": "process_name", "pid": PID_FRAMEWORK, "tid": 0, "args": {"name": fw_name}}
    yield {"ph": "M", "name": "process_labels", "pid": PID_FRAMEWORK, "tid": 0, "args": {"labels": "CPU"}}
    yield {"ph": "M", "name": "process_sort_index", "pid": PID_FRAMEWORK, "tid": 0, "args": {"sort_index": PID_FRAMEWORK}}
    for tid, thread_label in MSPROF_PROCESS_SPEC[fw_name]:
        yield {"ph": "M", "name": "thread_name", "pid": PID_FRAMEWORK, "tid": tid, "args": {"name": thread_label}}
        yield {"ph": "M", "name": "thread_sort_index", "pid": PID_FRAMEWORK, "tid": tid, "args": {"sort_index": tid}}

    multi = len(coords) > 1
    process_offsets = (
        ("Ascend Hardware", PID_OFFSET_HARDWARE, "NPU"),
        ("HCCL", PID_OFFSET_HCCL, "NPU"),
        ("Overlap Analysis", PID_OFFSET_OVERLAP, "NPU"),
        ("Ascend Metrics", PID_OFFSET_METRICS, "NPU"),
    )
    for die_idx, (replica, stage, rank) in enumerate(coords):
        base = _pid_base(die_idx)
        suffix = f" (R{replica}/PP{stage}/Die{rank})" if multi else ""
        for proc_key, offset, label in process_offsets:
            pid = base + offset
            yield {"ph": "M", "name": "process_name", "pid": pid, "tid": 0, "args": {"name": f"{proc_key}{suffix}"}}
            yield {"ph": "M", "name": "process_labels", "pid": pid, "tid": 0, "args": {"labels": label}}
            yield {"ph": "M", "name": "process_sort_index", "pid": pid, "tid": 0, "args": {"sort_index": pid}}
            for tid, thread_label in MSPROF_PROCESS_SPEC[proc_key]:
                yield {"ph": "M", "name": "thread_name", "pid": pid, "tid": tid, "args": {"name": thread_label}}
                yield {"ph": "M", "name": "thread_sort_index", "pid": pid, "tid": tid, "args": {"sort_index": tid}}


def _compute_physical_pipe_times(
    op: OperatorExportRow, peaks: Mapping[str, float],
) -> dict[str, float]:
    """Unplugged socket: shadow peak-division pipe times deleted."""
    raise NotImplementedError(
        "LEGACY_PURGED: _compute_physical_pipe_times shadow peak-division + HBM-rate fallback "
        "+ synthetic minimum span clamping deleted; project true pipe service intervals directly "
        "from canonical EngineStepReceipt."
    )


def _physical_pipe_spans(
    op: OperatorExportRow,
    ts0: float,
    pipe_info: Mapping[str, float],
    kernel_name: str,
) -> list[tuple[int, str, float, float, dict[str, Any]]]:
    """Derive strictly physical sub-pipe spans inside ``[ts0, ts0 + op.dur_us]``."""
    if op.dur_us <= 0.0:
        return []
    dur = op.dur_us
    d_mte2 = max(PIPE_MIN_SUBSPAN_US, pipe_info["mte2_us"]) if op.mte2_bytes > 0.0 else 0.0
    d_cube = max(PIPE_MIN_SUBSPAN_US, pipe_info["cube_us"]) if op.flops > 0.0 else 0.0
    d_vec = max(PIPE_MIN_SUBSPAN_US, pipe_info["vec_us"]) if op.vector_ops > 0.0 else 0.0
    d_fp = max(PIPE_MIN_SUBSPAN_US, pipe_info["fp_us"]) if pipe_info["fp_bytes"] > 0.0 else 0.0
    d_mte3 = max(PIPE_MIN_SUBSPAN_US, pipe_info["mte3_us"]) if op.mte3_bytes > 0.0 else 0.0

    d_mte2 = min(dur, d_mte2)
    d_cube = min(dur, d_cube)
    d_vec = min(dur, d_vec)
    d_fp = min(dur, d_fp)
    d_mte3 = min(dur, d_mte3)
    store_tail = max(d_fp, d_mte3)

    spans: list[tuple[int, str, float, float, dict[str, Any]]] = []
    if d_mte2 > 0.0:
        spans.append((
            TID_PIPE_MTE2, f"{kernel_name}:MTE2", ts0, d_mte2,
            {"Pipe": "MTE2", "Bytes": round(op.mte2_bytes, 1), "mte2_ratio": pipe_info["mte2_ratio"], "Kernel": kernel_name},
        ))
    if d_cube > 0.0:
        c_start = ts0 + max(0.0, min(dur - d_cube, (dur - store_tail) - d_cube))
        spans.append((
            TID_PIPE_CUBE, f"{kernel_name}:Cube", c_start, d_cube,
            {"Pipe": "Cube(AIC)", "FLOPs": round(op.flops, 1), "aic_mac_ratio": pipe_info["aic_mac_ratio"], "Dtype": op.dtype, "Kernel": kernel_name},
        ))
    if d_vec > 0.0:
        v_start = ts0 + max(0.0, min(dur - d_vec, (dur - d_mte3) - d_vec))
        spans.append((
            TID_PIPE_VECTOR, f"{kernel_name}:Vector", v_start, d_vec,
            {"Pipe": "Vector(AIV)", "Vector Ops": round(op.vector_ops, 1), "aiv_vec_ratio": pipe_info["aiv_vec_ratio"], "Kernel": kernel_name},
        ))
    if d_fp > 0.0:
        f_start = ts0 + max(0.0, dur - d_fp)
        spans.append((
            TID_PIPE_FIXPIPE, f"{kernel_name}:Fixpipe", f_start, d_fp,
            {"Pipe": "Fixpipe", "Bytes": round(pipe_info["fp_bytes"], 1), "fixpipe_ratio": pipe_info["fixpipe_ratio"], "Kernel": kernel_name},
        ))
    if d_mte3 > 0.0:
        s_start = ts0 + max(0.0, dur - d_mte3)
        spans.append((
            TID_PIPE_MTE3, f"{kernel_name}:MTE3", s_start, d_mte3,
            {"Pipe": "MTE3", "Bytes": round(op.mte3_bytes, 1), "mte3_ratio": pipe_info["mte3_ratio"], "Kernel": kernel_name},
        ))
    return spans


def _operator_args(
    bundle: ExportBundle,
    op: OperatorExportRow,
    task_id: int,
    layer_label: str,
    pipe_info: Mapping[str, float],
    pipeline_info: tuple[int, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "Model Name": bundle.model_name,
        "Hardware SKU": bundle.hardware_sku,
        "Phase": bundle.phase,
        "Task Type": op.task_type,
        "Stream Id": op.stream_index,
        "Task Id": task_id,
        "Layer": layer_label,
        "Archetype": op.archetype_title or op.layer_label,
        "Submodule": op.submodule,
        "Logical Role": op.logical_role,
        "Ascend Kernel": op.ascend_kernel,
        "Kernel Backend": op.kernel_backend,
        "Kernel Status": op.kernel_status,
        "Source Ref": op.source_ref,
        "Shapes & Dtype": op.tensor_contract,
        "FLOPs": round(op.flops, 1),
        "Vector Ops": round(op.vector_ops, 1),
        "HBM Read (B)": round(op.hbm_read_bytes, 1),
        "HBM Write (B)": round(op.hbm_write_bytes, 1),
        "HBM Total (B)": round(op.hbm_total_bytes, 1),
        "Link Bytes (B)": round(op.link_bytes, 1),
        "L2 Hit (%)": _round_metric(op.l2_hit_pct),
        "Bottleneck": op.bound_display or op.dominant_bound,
        "aic_mac_ratio": pipe_info["aic_mac_ratio"],
        "aiv_vec_ratio": pipe_info["aiv_vec_ratio"],
        "mte2_ratio": pipe_info["mte2_ratio"],
        "mte3_ratio": pipe_info["mte3_ratio"],
        "fixpipe_ratio": pipe_info["fixpipe_ratio"],
        "L* Physical Lower Bound (us)": _round_us(op.l_star_lean_us),
        "Sim Pred (us)": _round_us(op.dur_us),
        "Gap vs L* (us)": round(op.gap_us, TIMESTAMP_US_DECIMALS),
        "Physical-Limit Gap Ratio (T_pred / L*)": (
            round(op.gap_ratio, RATIO_DECIMALS) if op.gap_ratio is not None else None
        ),
        "Gap Radar Rank": op.gap_radar_rank,
    }
    if pipeline_info is not None:
        args["Fused Phases"] = pipeline_info[0]
        if pipeline_info[1]:
            args["Fused Members"] = ",".join(pipeline_info[1])
    if op.measured_sum_us is not None:
        args["Measured (us)"] = _round_us(op.measured_sum_us)
        args["Measured Kernels"] = len(op.measured_kernels)
    if op.ape_pct is not None:
        args["APE (%)"] = _round_metric(op.ape_pct)
    return args


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    ordered = sorted((max(0.0, a), max(a, b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _subtract_intervals(
    base: Sequence[tuple[float, float]],
    cut: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    if not base or not cut:
        return list(base)
    result: list[tuple[float, float]] = []
    cut_idx = 0
    for lo, hi in base:
        cur = lo
        while cut_idx < len(cut) and cut[cut_idx][1] <= cur:
            cut_idx += 1
        scan = cut_idx
        while scan < len(cut) and cut[scan][0] < hi:
            c_lo, c_hi = cut[scan]
            if c_lo > cur:
                result.append((cur, min(hi, c_lo)))
            cur = max(cur, c_hi)
            if cur >= hi:
                break
            scan += 1
        if cur < hi:
            result.append((cur, hi))
    return result


def _op_in_entry_repeats(op: OperatorExportRow, sched_count: int) -> int:
    """Unplugged socket: heuristic draft repeat multiplier deleted."""
    raise NotImplementedError(
        "LEGACY_PURGED: heuristic _op_in_entry_repeats / _expand_effective_trace_schedule "
        "draft repeat multiplier deleted; emit explicit schedule directly from EngineStepReceipt."
    )


def _expand_effective_trace_schedule(
    bundle: ExportBundle,
) -> tuple[tuple[TraceScheduleEntry, ...], dict[str, int]]:
    """Unplugged socket: heuristic draft schedule expansion deleted."""
    raise NotImplementedError(
        "LEGACY_PURGED: heuristic _expand_effective_trace_schedule deleted; "
        "read explicit full schedule directly from EngineStepReceipt."
    )


def _build_archetype_templates(
    bundle: ExportBundle,
    sched_counts: Mapping[str, int] | None = None,
) -> dict[str, _ArchetypeTemplate]:
    grouped: dict[str, list[OperatorExportRow]] = defaultdict(list)
    for op in bundle.operators:
        grouped[op.archetype_id or op.layer_label].append(op)
    span_by_arch = {a.archetype_id: a.single_span_us for a in bundle.archetypes}
    counts = sched_counts or {}
    templates: dict[str, _ArchetypeTemplate] = {}
    for aid, ops in grouped.items():
        min_s = min((op.start_us for op in ops), default=0.0)
        s_cnt = max(1, int(counts.get(aid, 1)))
        max_e = min_s
        sub_spans: list[tuple[str, float, float]] = []
        comp_iv: list[tuple[float, float]] = []
        comm_iv: list[tuple[float, float]] = []
        for op in ops:
            in_reps = _op_in_entry_repeats(op, s_cnt)
            r_s = max(0.0, op.start_us - min_s)
            op_span = op.dur_us * float(in_reps) if in_reps > 1 else max(0.0, op.end_us - op.start_us)
            r_e = max(r_s, r_s + op_span)
            max_e = max(max_e, min_s + r_e)
            if r_e > r_s:
                if op.task_type == "HCCL":
                    comm_iv.append((r_s, r_e))
                else:
                    comp_iv.append((r_s, r_e))
            if sub_spans and sub_spans[-1][0] == op.submodule:
                prev_sub, p_lo, p_hi = sub_spans[-1]
                sub_spans[-1] = (prev_sub, min(p_lo, r_s), max(p_hi, r_e))
            else:
                sub_spans.append((op.submodule, r_s, r_e))
        span = span_by_arch.get(aid, max(0.0, max_e - min_s))
        if span <= 0.0:
            span = max(0.0, max_e - min_s)
        templates[aid] = _ArchetypeTemplate(
            archetype_id=aid,
            ops=tuple(ops),
            min_start_us=min_s,
            span_us=span,
            sub_spans=tuple(sub_spans),
            comp_intervals=tuple(_merge_intervals(comp_iv)),
            comm_intervals=tuple(_merge_intervals(comm_iv)),
        )
    return templates


def _emit_host_launch_with_flow(
    *,
    name: str,
    dev_ts: float,
    dev_pid: int,
    dev_tid: int,
    launch_dur_us: float,
    flow_id: int,
    extra_args: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], int]:
    host_ts = max(0.0, dev_ts - launch_dur_us)
    host_dur = max(PIPE_MIN_SUBSPAN_US, min(launch_dur_us, dev_ts - host_ts) if dev_ts > host_ts else PIPE_MIN_SUBSPAN_US)
    flow_src_ts = _round_us(host_ts + host_dur)
    flow_dst_ts = max(flow_src_ts, _round_us(dev_ts))
    f_args = {"launch": name, "sync_kind": "HostToDevice", **extra_args}
    events = [
        {
            "name": name, "cat": "Runtime", "ph": "X",
            "ts": _round_us(host_ts), "dur": _round_us(host_dur),
            "pid": PID_FRAMEWORK, "tid": TID_FW_HOST_LAUNCH,
            "args": dict(f_args),
        },
        {
            "name": f"HostToDevice->{name}", "cat": "HostToDevice", "ph": "s",
            "id": flow_id, "ts": flow_src_ts,
            "pid": PID_FRAMEWORK, "tid": TID_FW_HOST_LAUNCH,
            "args": f_args,
        },
        {
            "name": f"HostToDevice<-{name}", "cat": "HostToDevice", "ph": "f",
            "bp": "e", "id": flow_id, "ts": flow_dst_ts,
            "pid": dev_pid, "tid": dev_tid,
            "args": f_args,
        },
    ]
    return events, flow_id + 1


def _emit_op_and_pipes(
    bundle: ExportBundle,
    op: OperatorExportRow,
    *,
    ts_start: float,
    task_id: int,
    layer_label: str,
    base_pid: int,
    flow_id: int,
    include_pipe_lanes: bool,
    pipeline_info: tuple[int, tuple[str, ...]] | None,
    op_placement: Mapping[Any, tuple[int, int, float, str]],
) -> tuple[list[dict[str, Any]], tuple[int, int, float, str], int]:
    pid, tid = _op_pid_tid(op, base_pid)
    cat = "Communication" if op.task_type == "HCCL" else "Kernel"
    k_name = op.ascend_kernel or op.logical_role or op.canonical_op_key
    event_name = f"{k_name} [{op.canonical_op_key}]" if (op.canonical_op_key and op.canonical_op_key != k_name) else k_name
    pipe_info = _compute_physical_pipe_times(op, bundle.hardware_peaks)
    ts_end = ts_start + op.dur_us
    events: list[dict[str, Any]] = [{
        "name": event_name,
        "cat": cat,
        "ph": "X",
        "ts": _round_us(ts_start),
        "dur": _round_us(op.dur_us),
        "pid": pid,
        "tid": tid,
        "args": _operator_args(bundle, op, task_id, layer_label, pipe_info, pipeline_info),
    }]
    placement = (pid, tid, ts_end, k_name)

    for pred in op.predecessors:
        src = op_placement.get(pred)
        if src is None:
            continue
        src_pid, src_tid, src_end_us, src_name = src
        if src_pid == pid and src_tid == tid:
            continue
        dst_ts = max(src_end_us, ts_start)
        sync_kind = "EventRecord/Wait" if src_pid != pid else "StreamSync"
        f_args = {"src_kernel": src_name, "dst_kernel": k_name, "sync_kind": sync_kind}
        events.append({
            "name": f"SetFlag->{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "s",
            "id": flow_id, "ts": _round_us(src_end_us), "pid": src_pid, "tid": src_tid, "args": f_args,
        })
        events.append({
            "name": f"WaitFlag<-{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "f", "bp": "e",
            "id": flow_id, "ts": _round_us(dst_ts), "pid": pid, "tid": tid, "args": f_args,
        })
        flow_id += 1

    if include_pipe_lanes and op.task_type != "HCCL":
        hw_pid = base_pid + PID_OFFSET_HARDWARE
        for p_tid, p_name, p_ts, p_dur, p_args in _physical_pipe_spans(op, ts_start, pipe_info, k_name):
            events.append({
                "name": p_name, "cat": "Pipe", "ph": "X",
                "ts": _round_us(p_ts), "dur": _round_us(p_dur),
                "pid": hw_pid, "tid": p_tid, "args": p_args,
            })
    return events, placement, flow_id


def _iter_overlap_from_intervals(
    comp_by_die: Mapping[int, Sequence[tuple[float, float]]],
    comm_by_die: Mapping[int, Sequence[tuple[float, float]]],
    die_indices: Sequence[int],
    total_span_us: float,
) -> Iterator[dict[str, Any]]:
    for d_idx in die_indices:
        pid = _pid_base(d_idx) + PID_OFFSET_OVERLAP
        comp = _merge_intervals(comp_by_die.get(d_idx, ()))
        comm = _merge_intervals(comm_by_die.get(d_idx, ()))
        comm_only = _subtract_intervals(comm, comp)
        busy = _merge_intervals([*comp, *comm])
        step_end = max(total_span_us, max((hi for _, hi in busy), default=0.0))
        free = _subtract_intervals([(0.0, step_end)], busy) if step_end > 0.0 else []
        for tid, label, intervals in (
            (TID_OVERLAP_COMPUTING, "Computing", comp),
            (TID_OVERLAP_COMM, "Communication", comm),
            (TID_OVERLAP_COMM_NOT_OVERLAPPED, "Communication(Not Overlapped)", comm_only),
            (TID_OVERLAP_FREE, "Free", free),
        ):
            for lo, hi in intervals:
                if hi <= lo:
                    continue
                yield {
                    "name": label, "cat": "Overlap", "ph": "X",
                    "ts": _round_us(lo), "dur": _round_us(hi - lo),
                    "pid": pid, "tid": tid,
                    "args": {"Overlap Lane": label, "Duration (us)": _round_us(hi - lo)},
                }


def _iter_counter_events_from_samples(
    bundle: ExportBundle,
    samples_by_die: Mapping[int, Sequence[tuple[float, float, float, float, float]]],
) -> Iterator[dict[str, Any]]:
    peaks = bundle.hardware_peaks
    hbm_peak_gbps = float(peaks.get("hbm_peak_bytes_per_s", 0.0) or 0.0) / BYTES_PER_GB
    mem_ledger = bundle.memory_ledger
    resident_gib = (
        (mem_ledger.max_stage_hbm_bytes - sum(b.workspace_peak_bytes for b in mem_ledger.stage_budgets[:1]))
        / float(GIB_BYTES)
        if mem_ledger.stage_budgets else 0.0
    )
    resident_gib = max(0.0, resident_gib)

    for d_idx, samples in sorted(samples_by_die.items()):
        pid = _pid_base(d_idx) + PID_OFFSET_METRICS
        if not samples:
            continue
        stride = max(1, (len(samples) + MAX_COUNTER_SAMPLES - 1) // MAX_COUNTER_SAMPLES)
        for idx, (ts_us, bw_gbps, cube_pct, vec_pct, ws_gib) in enumerate(samples):
            if idx % stride != 0 and idx != len(samples) - 1:
                continue
            bw_val = max(0.0, min(bw_gbps, hbm_peak_gbps if hbm_peak_gbps > 0.0 else bw_gbps))
            yield {
                "name": "HBM Bandwidth (GB/s)", "ph": "C", "ts": _round_us(ts_us),
                "pid": pid, "tid": TID_METRICS_COUNTER, "args": {"HBM_BW_GBps": _round_metric(bw_val)},
            }
            yield {
                "name": "Cube Utilization (%)", "ph": "C", "ts": _round_us(ts_us),
                "pid": pid, "tid": TID_METRICS_COUNTER, "args": {"Cube_Util_Pct": _round_metric(min(PCT_SCALE, max(0.0, cube_pct)))},
            }
            yield {
                "name": "Vector Utilization (%)", "ph": "C", "ts": _round_us(ts_us),
                "pid": pid, "tid": TID_METRICS_COUNTER, "args": {"Vector_Util_Pct": _round_metric(min(PCT_SCALE, max(0.0, vec_pct)))},
            }
            yield {
                "name": "Active HBM Memory (GiB)", "ph": "C", "ts": _round_us(ts_us),
                "pid": pid, "tid": TID_METRICS_COUNTER,
                "args": {
                    "Resident_GiB": _round_metric(resident_gib),
                    "Active_Total_GiB": _round_metric(max(resident_gib, resident_gib + max(0.0, ws_gib))),
                },
            }


def _layer_metric_sample(
    ops: Sequence[OperatorExportRow], ts_mid: float, span_us: float, peaks: Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    dur_s = max(span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    cube_peak = float(peaks.get("cube_peak_flops_per_s", 0.0) or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s", 0.0) or 0.0)
    hbm_tot = sum(op.hbm_total_bytes for op in ops)
    flops = sum(op.flops for op in ops)
    vec_ops = sum(op.vector_ops for op in ops)
    ws_gib = max((op.hbm_write_bytes + op.mte3_bytes for op in ops), default=0.0) / float(GIB_BYTES)
    bw_gbps = (hbm_tot / dur_s) / BYTES_PER_GB
    cube_pct = (PCT_SCALE * (flops / dur_s) / cube_peak) if cube_peak > 0.0 else 0.0
    vec_pct = (PCT_SCALE * (vec_ops / dur_s) / vec_peak) if vec_peak > 0.0 else 0.0
    return ts_mid, bw_gbps, cube_pct, vec_pct, ws_gib


def _resolve_host_launch_spec(
    op_eff: OperatorExportRow,
    entry: Any,
    *,
    is_graph_mode: bool,
    is_h2d_op: bool,
) -> tuple[str, float]:
    """Unplugged socket: hardcoded HOST_GRAPH_LAUNCH_US / HOST_KERNEL_LAUNCH_US deleted."""
    raise NotImplementedError(
        "LEGACY_PURGED: magic HOST_GRAPH_LAUNCH_US=12.0 / HOST_KERNEL_LAUNCH_US=1.5 deleted; "
        "read calibrated STARS/host launch durations from PrimitiveTable."
    )


def _entry_layer_span(
    entry: Any, tpl: Any, per_layer: Mapping[tuple[str, int], Mapping[str, Any]],
) -> float:
    if entry.full_layer_index is not None:
        rec = per_layer.get((entry.scope, int(entry.full_layer_index)))
        if isinstance(rec, Mapping) and float(rec.get("span_us", 0.0) or 0.0) > 0.0:
            return float(rec["span_us"])
    return float(tpl.span_us)


def _emit_entry_op_repeats(
    bundle: ExportBundle,
    op_eff: OperatorExportRow,
    *,
    op_ts: float,
    in_reps: int,
    task_id: int,
    layer_label: str,
    base_pid: int,
    flow_id: int,
    include_pipe_lanes: bool,
    pipeline_info: tuple[int, tuple[str, ...]] | None,
    op_placement: dict[Any, tuple[int, int, float, str]],
) -> tuple[list[dict[str, Any]], int, int]:
    """Emit one or ``in_reps`` consecutive spans for a collapsed tail/micro-split operator."""
    out: list[dict[str, Any]] = []
    placement: tuple[int, int, float, str] | None = None
    for rep_idx in range(in_reps):
        rep_ts = op_ts + float(rep_idx) * op_eff.dur_us
        rep_op = (
            replace(op_eff, logical_role=f"{op_eff.logical_role} (#{rep_idx + 1}/{in_reps})")
            if in_reps > 1 else op_eff
        )
        events, placement, flow_id = _emit_op_and_pipes(
            bundle, rep_op,
            ts_start=rep_ts, task_id=task_id, layer_label=layer_label,
            base_pid=base_pid, flow_id=flow_id, include_pipe_lanes=include_pipe_lanes,
            pipeline_info=pipeline_info,
            op_placement=op_placement if rep_idx == 0 else {},
        )
        out.extend(events)
        task_id += 1
    if placement is not None:
        op_placement[op_eff.occurrence] = placement
    return out, task_id, flow_id


def _iter_scheduled_full_model_trace(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
    pipeline_by_occ: Mapping[Any, tuple[int, tuple[str, ...]]],
    *,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
) -> Iterator[dict[str, Any]]:
    """Stream full multi-layer Target + Draft schedule using native per-layer timings when recorded."""
    from .kernel_catalog import strip_op_prefix

    effective_schedule, sched_counts = _expand_effective_trace_schedule(bundle)
    templates = _build_archetype_templates(bundle, sched_counts)
    per_layer = dict(getattr(bundle, "per_layer_timings", None) or {})
    total_span_us = sum(
        _entry_layer_span(e, templates[e.archetype_id], per_layer)
        for e in effective_schedule if e.archetype_id in templates
    )
    track_display = format_track_label(bundle.feature_track, lang="en")
    yield {
        "name": f"ModelRunner.execute_model [{bundle.model_name} | {bundle.phase.upper()} | graph={bundle.graph_capture_mode}]",
        "cat": "Framework", "ph": "X", "ts": 0.0, "dur": _round_us(total_span_us),
        "pid": PID_FRAMEWORK, "tid": TID_FW_ENGINE,
        "args": {
            "Model": bundle.model_name, "Hardware": bundle.hardware_sku,
            "Target Layers": bundle.total_target_layers, "Draft Layers": bundle.total_draft_layers,
            "Graph Capture": bundle.graph_capture_mode, "Feature Track": track_display,
        },
    }

    is_graph_mode = bundle.graph_capture_mode.lower() not in {"eager", "none", "off"}
    comp_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    comm_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    samples_by_die: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)

    ts_cursor = 0.0
    task_id = 1
    flow_id = 1
    launched_scopes: set[str] = set()
    scope_bounds: dict[str, list[float]] = {}

    for entry in effective_schedule:
        tpl = templates.get(entry.archetype_id)
        if tpl is None or not tpl.ops:
            continue
        layer_rec = (
            per_layer.get((entry.scope, int(entry.full_layer_index)))
            if entry.full_layer_index is not None else None
        )
        layer_ops_map = layer_rec.get("ops", {}) if isinstance(layer_rec, Mapping) else {}
        layer_span = _entry_layer_span(entry, tpl, per_layer)
        l_start = ts_cursor
        l_end = ts_cursor + layer_span
        if entry.scope not in scope_bounds:
            scope_bounds[entry.scope] = [l_start, l_end]
        else:
            scope_bounds[entry.scope][1] = l_end

        yield {
            "name": entry.label, "cat": "MSTX_Layer", "ph": "X",
            "ts": _round_us(l_start), "dur": _round_us(layer_span),
            "pid": PID_FRAMEWORK, "tid": TID_FW_LAYER_MSTX,
            "args": {"Scope": entry.scope, "Archetype": entry.archetype_id, "Layer": entry.full_layer_index},
        }
        scale_span = (layer_span / tpl.span_us) if tpl.span_us > 0.0 else 1.0
        for sub_name, r_lo, r_hi in tpl.sub_spans:
            if r_hi <= r_lo:
                continue
            yield {
                "name": f"{sub_name} ({entry.label})", "cat": "MSTX_Submodule", "ph": "X",
                "ts": _round_us(l_start + r_lo * scale_span), "dur": _round_us((r_hi - r_lo) * scale_span),
                "pid": PID_FRAMEWORK, "tid": TID_FW_SUBMODULE_MSTX,
                "args": {"Submodule": sub_name, "Layer": entry.label},
            }

        op_placement: dict[Any, tuple[int, int, float, str]] = {}
        first_op_for_scope = entry.scope not in launched_scopes
        entry_sched_cnt = max(1, int(sched_counts.get(entry.archetype_id, 1)))
        for op in tpl.ops:
            coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
            d_idx = die_map.get(coord, 0)
            base = _pid_base(d_idx)
            exact_timing = layer_ops_map.get(strip_op_prefix(op.op_id)) if layer_ops_map else None
            if exact_timing is not None:
                op_ts = l_start + float(exact_timing[0])
                op_eff = replace(op, dur_us=float(exact_timing[1]), end_us=op.start_us + float(exact_timing[1]))
            else:
                op_ts = l_start + max(0.0, op.start_us - tpl.min_start_us)
                op_eff = op
            is_h2d_op = op_eff.family == "engram_rh2d" or "host_uva" in op_eff.op_id.lower()
            if first_op_for_scope or not is_graph_mode or is_h2d_op:
                dev_pid, dev_tid = _op_pid_tid(op_eff, base)
                launch_name, l_dur = _resolve_host_launch_spec(
                    op_eff, entry, is_graph_mode=is_graph_mode, is_h2d_op=is_h2d_op,
                )
                l_events, flow_id = _emit_host_launch_with_flow(
                    name=launch_name, dev_ts=op_ts, dev_pid=dev_pid, dev_tid=dev_tid,
                    launch_dur_us=l_dur,
                    flow_id=flow_id, extra_args={"scope": entry.scope, "layer": entry.label, "transfer": "H2D" if is_h2d_op else "launch"},
                )
                yield from l_events
                if first_op_for_scope:
                    launched_scopes.add(entry.scope)
                    first_op_for_scope = False

            in_reps = _op_in_entry_repeats(op_eff, entry_sched_cnt)
            events, task_id, flow_id = _emit_entry_op_repeats(
                bundle, op_eff, op_ts=op_ts, in_reps=in_reps, task_id=task_id,
                layer_label=entry.label, base_pid=base, flow_id=flow_id,
                include_pipe_lanes=include_pipe_lanes,
                pipeline_info=pipeline_by_occ.get(op_eff.occurrence),
                op_placement=op_placement,
            )
            yield from events

        # Accumulate compact per-layer overlap intervals & counter samples (O(1) per layer).
        d0 = die_map.get((tpl.ops[0].occurrence.replica, tpl.ops[0].occurrence.stage, int(tpl.ops[0].occurrence.rank or 0)), 0)
        for r_lo, r_hi in tpl.comp_intervals:
            comp_by_die[d0].append((l_start + r_lo * scale_span, l_start + r_hi * scale_span))
        for r_lo, r_hi in tpl.comm_intervals:
            comm_by_die[d0].append((l_start + r_lo * scale_span, l_start + r_hi * scale_span))
        if include_counters and layer_span > 0.0:
            samples_by_die[d0].append(_layer_metric_sample(tpl.ops, l_start + layer_span * MIDPOINT_FRACTION, layer_span, bundle.hardware_peaks))
        ts_cursor = l_end

    for sc_key, (s_lo, s_hi) in scope_bounds.items():
        if s_hi > s_lo:
            sc_title = (
                f"Target Forward Pass ({bundle.total_target_layers} Layers)" if sc_key == "target"
                else f"Speculative Draft Forward Pass ({bundle.total_draft_layers} Layers)"
            )
            yield {
                "name": sc_title, "cat": "Framework", "ph": "X",
                "ts": _round_us(s_lo), "dur": _round_us(s_hi - s_lo),
                "pid": PID_FRAMEWORK, "tid": TID_FW_ENGINE,
                "args": {"Scope": sc_key},
            }

    if include_overlap:
        yield from _iter_overlap_from_intervals(comp_by_die, comm_by_die, sorted(set(die_map.values())), ts_cursor)
    if include_counters:
        yield from _iter_counter_events_from_samples(bundle, samples_by_die)


def _iter_direct_trace(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
    pipeline_by_occ: Mapping[Any, tuple[int, tuple[str, ...]]],
    *,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
) -> Iterator[dict[str, Any]]:
    """Fallback single-pass streamer when caller passed an explicit ``step_eval``."""
    yield {
        "name": f"ModelRunner.execute_model [{bundle.model_name} | {bundle.phase.upper()}]",
        "cat": "Framework", "ph": "X", "ts": 0.0, "dur": _round_us(bundle.step_span_us),
        "pid": PID_FRAMEWORK, "tid": TID_FW_ENGINE,
        "args": {"Model": bundle.model_name, "Hardware": bundle.hardware_sku},
    }
    comp_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    comm_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    samples_by_die: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)
    op_placement: dict[Any, tuple[int, int, float, str]] = {}
    layer_spans: list[tuple[str, float, float]] = []
    sub_spans: list[tuple[str, str, float, float]] = []
    flow_id = 1
    launched = False

    for task_id, op in enumerate(bundle.operators, start=1):
        coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        d_idx = die_map.get(coord, 0)
        base = _pid_base(d_idx)
        if not launched:
            raise NotImplementedError(
                "LEGACY_PURGED: flat-trace unconditional aclGraphLaunch (even in eager mode) with "
                "magic HOST_GRAPH_LAUNCH_US=12.0 deleted (Astra P1); respect graph_capture_mode "
                "and read launch duration from PrimitiveTable."
            )
        lbl = op.archetype_title or op.layer_label
        events, placement, flow_id = _emit_op_and_pipes(
            bundle, op, ts_start=op.start_us, task_id=task_id, layer_label=lbl,
            base_pid=base, flow_id=flow_id, include_pipe_lanes=include_pipe_lanes,
            pipeline_info=pipeline_by_occ.get(op.occurrence), op_placement=op_placement,
        )
        yield from events
        op_placement[op.occurrence] = placement
        if op.dur_us > 0.0:
            if op.task_type == "HCCL":
                comm_by_die[d_idx].append((op.start_us, op.end_us))
            else:
                comp_by_die[d_idx].append((op.start_us, op.end_us))
            if include_counters:
                samples_by_die[d_idx].append(_layer_metric_sample((op,), op.start_us, op.dur_us, bundle.hardware_peaks))
        if layer_spans and layer_spans[-1][0] == lbl:
            layer_spans[-1] = (lbl, min(layer_spans[-1][1], op.start_us), max(layer_spans[-1][2], op.end_us))
        else:
            layer_spans.append((lbl, op.start_us, op.end_us))
        if sub_spans and sub_spans[-1][0] == lbl and sub_spans[-1][1] == op.submodule:
            sub_spans[-1] = (lbl, op.submodule, min(sub_spans[-1][2], op.start_us), max(sub_spans[-1][3], op.end_us))
        else:
            sub_spans.append((lbl, op.submodule, op.start_us, op.end_us))

    for lbl, lo, hi in layer_spans:
        if hi > lo:
            yield {
                "name": lbl, "cat": "MSTX_Layer", "ph": "X",
                "ts": _round_us(lo), "dur": _round_us(hi - lo),
                "pid": PID_FRAMEWORK, "tid": TID_FW_LAYER_MSTX, "args": {"Layer": lbl},
            }
    for lbl, sub, lo, hi in sub_spans:
        if hi > lo:
            yield {
                "name": f"{sub} ({lbl})", "cat": "MSTX_Submodule", "ph": "X",
                "ts": _round_us(lo), "dur": _round_us(hi - lo),
                "pid": PID_FRAMEWORK, "tid": TID_FW_SUBMODULE_MSTX, "args": {"Submodule": sub, "Layer": lbl},
            }
    if include_overlap:
        yield from _iter_overlap_from_intervals(comp_by_die, comm_by_die, sorted(set(die_map.values())), bundle.step_span_us)
    if include_counters:
        yield from _iter_counter_events_from_samples(bundle, samples_by_die)


def iter_msprof_trace_events(
    bundle: ExportBundle,
    *,
    include_pipe_lanes: bool = True,
    include_counters: bool = True,
    include_overlap: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield 1:1 Ascend ``msprof`` Chrome Trace events lazily with O(Archetypes) memory."""
    coords = _die_coordinates(bundle)
    die_map = {coord: idx for idx, coord in enumerate(coords)}
    yield from _iter_metadata_events(coords)

    pipeline_by_occ = {
        pipe.owner: (len(pipe.phases), pipe.member_op_ids)
        for pipe in bundle.pipelines
    }
    if bundle.full_trace_schedule:
        yield from _iter_scheduled_full_model_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
        )
    else:
        yield from _iter_direct_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
        )


def _stream_trace_events_to_fd(
    fd: int,
    bundle: ExportBundle,
    *,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
) -> tuple[int, dict[str, int]]:
    total = 0
    ph_counts: dict[str, int] = {"M": 0, "X": 0, "s": 0, "C": 0}
    with os.fdopen(
        fd, "w", encoding="utf-8", newline="\n", buffering=STREAM_BUFFER_BYTES,
    ) as handle:
        handle.write("[\n")
        first = True
        for event in iter_msprof_trace_events(
            bundle,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
        ):
            if not first:
                handle.write(",\n")
            first = False
            handle.write(
                json.dumps(event, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
            )
            total += 1
            ph = str(event.get("ph") or "")
            if ph in ph_counts:
                ph_counts[ph] += 1
        handle.write("\n]\n")
    return total, ph_counts


def export_msprof_trace(
    source: ExportBundle | CaseInput | None = None,
    output_path: str | Path = "",
    *,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    include_pipe_lanes: bool = True,
    include_counters: bool = True,
    include_overlap: bool = True,
    model: str | None = None,
    hardware: str | None = None,
    topology_preset: str | None = None,
    dp: int = 1,
    tp: int = 1,
    dcp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS,
    concurrency: int = 1,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    feature_track: str = "shipping",
) -> TraceExportSummary:
    """Stream a 1:1 ``msprof`` ``trace_view.json`` to ``output_path`` atomically."""
    if not str(output_path):
        raise ValueError("export_msprof_trace_requires_output_path")
    if source is None:
        if not model or not hardware:
            raise ValueError("export_msprof_trace_requires_source_or_model_and_hardware")
        raw_case, auto_req = compose_export_case(
            model, hardware, topology_preset=topology_preset,
            dp=dp, tp=tp, dcp=dcp, ep=ep, pp=pp, pcp=pcp,
            prompt_tokens=prompt_tokens, output_tokens=output_tokens,
            concurrency=concurrency, mtp_proposal_tokens=mtp_proposal_tokens,
            mtp_acceptance_length=mtp_acceptance_length, feature_track=feature_track,
        )
        source = raw_case
        request = request or auto_req
    bundle = (
        source if isinstance(source, ExportBundle)
        else build_export_bundle(
            source,
            request=request,
            step_eval=step_eval,
            replay=replay,
            observations_path=observations_path,
        )
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        total, ph_counts = _stream_trace_events_to_fd(
            fd, bundle,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
        )
        bytes_written = int(tmp_path.stat().st_size)
        os.replace(tmp_path, target)
    except TimeoutError:
        _cleanup_temp(tmp_path)
        raise
    except Exception:
        _cleanup_temp(tmp_path)
        raise
    return TraceExportSummary(
        schema=TRACE_EXPORT_RECEIPT_SCHEMA,
        output_path=str(target),
        model_name=bundle.model_name,
        hardware_sku=bundle.hardware_sku,
        phase=bundle.phase,
        graph_id=bundle.graph_id,
        total_events=total,
        metadata_events=ph_counts["M"],
        complete_events=ph_counts["X"],
        flow_pairs=ph_counts["s"],
        counter_events=ph_counts["C"],
        step_span_us=bundle.step_span_us,
        step_l_star_lean_us=bundle.step_l_star_lean_us,
        bytes_written=bytes_written,
        replayed_layers=len(bundle.full_trace_schedule) or len(bundle.archetypes),
    )


def _cleanup_temp(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except TimeoutError:
        raise
    except OSError:
        pass


__all__ = [
    "MSPROF_PROCESS_SPEC",
    "PID_FRAMEWORK",
    "TID_FW_ENGINE",
    "TID_FW_HOST_LAUNCH",
    "TID_FW_LAYER_MSTX",
    "TID_FW_SUBMODULE_MSTX",
    "TRACE_EXPORT_RECEIPT_SCHEMA",
    "TraceExportSummary",
    "export_msprof_trace",
    "iter_msprof_trace_events",
]
