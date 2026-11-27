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
import re
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest
from .bundle import build_export_bundle, compose_export_case
from .kernel_catalog import strip_op_prefix
from .models import (
    DEFAULT_SCENARIO_OUTPUT_TOKENS,
    DEFAULT_SCENARIO_PROMPT_TOKENS,
    GIB_BYTES,
    MIN_POSITIVE_SECONDS,
    NS_PER_US,
    US_PER_SECOND,
    ExportBundle,
    OperatorExportRow,
    TraceScheduleEntry,
    bundle_step_int_ns,
    op_exposed_int_ns,
)
from .public_view import format_track_label

#: Schema tag for trace export receipts.
TRACE_EXPORT_RECEIPT_SCHEMA = "mindie-sim.msprof-trace-receipt/2"
#: Stream buffer size (64 KiB) for bounded low-RSS JSON emission.
STREAM_BUFFER_BYTES = 64 * 1024
#: Number of serialized trace JSON lines batched per buffered write call.
_STREAM_BATCH_EVENTS = 256
#: Maximum distinct ``(dur_us, work_tuple)`` variants cached per archetype template op.
_MAX_OP_VARIANT_CACHE = 16
#: Operator args keys emitted before ``Task Id`` and ``Layer`` (for exact JSON order preservation).
_ARGS_PREFIX_KEYS = ("Model Name", "Hardware SKU", "Phase", "Task Type", "Stream Id")
_ARGS_DYNAMIC_KEYS = frozenset((*_ARGS_PREFIX_KEYS, "Task Id", "Layer"))
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
_SPAN_GAP_EPS_US = 1e-6
_WORK_TUPLE_LEN = 10

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
    """Project physical pipe service durations and ratios from declared work and explicit peaks."""
    dur = max(0.0, float(op.dur_us))
    cube_peak = float(op.cube_peak_flops_per_s or peaks.get("cube_peak_flops_per_s") or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s") or 0.0)
    mte2_peak = float(peaks.get("mte2_peak_bytes_per_s") or 0.0)
    mte3_peak = float(peaks.get("mte3_peak_bytes_per_s") or 0.0)
    fp_peak = float(peaks.get("fixpipe_peak_bytes_per_s") or 0.0)
    fp_bytes = max(0.0, float(op.fixpipe_bytes))

    def _pipe_us(work: float, rate: float) -> float:
        if work <= 0.0 or rate <= 0.0 or dur <= 0.0:
            return 0.0
        return min(dur, (float(work) / float(rate)) * US_PER_SECOND)

    cube_us = _pipe_us(op.flops, cube_peak)
    vec_us = _pipe_us(op.vector_ops, vec_peak)
    mte2_us = _pipe_us(op.mte2_bytes, mte2_peak)
    mte3_us = _pipe_us(op.mte3_bytes, mte3_peak)
    fp_us = _pipe_us(fp_bytes, fp_peak)

    def _ratio(span_us: float) -> float:
        if dur <= 0.0 or span_us <= 0.0:
            return 0.0
        return round(min(1.0, span_us / dur), RATIO_DECIMALS)

    return {
        "cube_us": cube_us,
        "vec_us": vec_us,
        "mte2_us": mte2_us,
        "mte3_us": mte3_us,
        "fp_us": fp_us,
        "fp_bytes": fp_bytes,
        "aic_mac_ratio": _ratio(cube_us),
        "aiv_vec_ratio": _ratio(vec_us),
        "mte2_ratio": _ratio(mte2_us),
        "mte3_ratio": _ratio(mte3_us),
        "fixpipe_ratio": _ratio(fp_us),
    }


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
    """Return exact integer in-entry repeat divisor (zero heuristic rounded multipliers)."""
    cnt = max(1, int(sched_count))
    rep = max(1, int(op.repeat_count))
    if rep > cnt and rep % cnt == 0:
        return rep // cnt
    return 1


def _expand_effective_trace_schedule(
    bundle: ExportBundle,
) -> tuple[tuple[TraceScheduleEntry, ...], dict[str, int]]:
    """Read explicit full-model trace schedule directly from ``ExportBundle``."""
    entries = tuple(bundle.full_trace_schedule or ())
    counts = dict(Counter(e.archetype_id for e in entries))
    return entries, counts


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
    pending_flows: list[tuple[Any, tuple[int, int, float, str], float]] | None = None,
) -> tuple[list[dict[str, Any]], tuple[int, int, float, str], int]:
    pid, tid = _op_pid_tid(op, base_pid)
    cat = "Communication" if op.task_type == "HCCL" else "Kernel"
    k_name = op.ascend_kernel or op.logical_role or op.canonical_op_key
    op_tag = strip_op_prefix(op.op_id) or op.canonical_op_key
    event_name = f"{k_name} [{op_tag}]" if (op_tag and op_tag != k_name) else k_name
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
            if pending_flows is not None:
                pending_flows.append((pred, placement, ts_start))
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

    has_pipe_work = (
        op.flops > 0.0 or op.vector_ops > 0.0
        or op.mte2_bytes > 0.0 or op.mte3_bytes > 0.0 or op.fixpipe_bytes > 0.0
    )
    if include_pipe_lanes and (op.task_type != "HCCL" or has_pipe_work):
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
    *,
    stage_by_die: Mapping[int, int] | None = None,
) -> Iterator[dict[str, Any]]:
    peaks = bundle.hardware_peaks
    hbm_peak_gbps = float(peaks.get("hbm_peak_bytes_per_s", 0.0) or 0.0) / BYTES_PER_GB
    mem_ledger = bundle.memory_ledger
    budget_by_stage = {int(b.stage): b for b in mem_ledger.stage_budgets}

    for d_idx, samples in sorted(samples_by_die.items()):
        pid = _pid_base(d_idx) + PID_OFFSET_METRICS
        if not samples:
            continue
        stage = int(stage_by_die.get(d_idx, 0)) if stage_by_die is not None else 0
        sb = budget_by_stage.get(stage)
        if sb is None and mem_ledger.stage_budgets:
            sb = mem_ledger.stage_budgets[0]
        stage_total_gib = (
            max(0.0, float(sb.total_hbm_bytes) / float(GIB_BYTES))
            if sb is not None else 0.0
        )
        max_ws_gib = (
            max(0.0, float(sb.workspace_peak_bytes) / float(GIB_BYTES))
            if sb is not None
            else max(0.0, float(mem_ledger.max_stage_workspace_bytes) / float(GIB_BYTES))
        )
        resident_gib = max(0.0, stage_total_gib - max_ws_gib) if sb is not None else 0.0
        stride = max(1, (len(samples) + MAX_COUNTER_SAMPLES - 1) // MAX_COUNTER_SAMPLES)
        for idx, (ts_us, bw_gbps, cube_pct, vec_pct, ws_gib) in enumerate(samples):
            if idx % stride != 0 and idx != len(samples) - 1:
                continue
            bw_val = max(0.0, min(bw_gbps, hbm_peak_gbps if hbm_peak_gbps > 0.0 else bw_gbps))
            bounded_ws_gib = min(max(0.0, ws_gib), max_ws_gib) if max_ws_gib > 0.0 else 0.0
            active_total_gib = resident_gib + bounded_ws_gib
            if stage_total_gib > 0.0:
                active_total_gib = min(active_total_gib, stage_total_gib)
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
                    "Active_Total_GiB": _round_metric(max(resident_gib, active_total_gib)),
                },
            }


def _layer_metric_sample(
    ops: Sequence[OperatorExportRow | tuple[OperatorExportRow, int]],
    ts_mid: float,
    span_us: float,
    peaks: Mapping[str, float],
) -> tuple[float, float, float, float, float]:
    dur_s = max(span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    cube_peak = float(peaks.get("cube_peak_flops_per_s", 0.0) or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s", 0.0) or 0.0)
    hbm_tot = 0.0
    cube_busy_s = 0.0
    vec_ops = 0.0
    ws_peak_bytes = 0.0
    for item in ops:
        if isinstance(item, tuple):
            op, mult = item
        else:
            op, mult = item, 1
        scale = float(max(1, int(mult)))
        hbm_tot += op.hbm_total_bytes * scale
        vec_ops += op.vector_ops * scale
        if op.flops > 0.0:
            op_peak = float(op.cube_peak_flops_per_s) if float(op.cube_peak_flops_per_s) > 0.0 else cube_peak
            if op_peak > 0.0:
                cube_busy_s += (op.flops * scale) / op_peak
        ws_peak_bytes = max(ws_peak_bytes, max(op.hbm_write_bytes, op.mte3_bytes))
    ws_gib = ws_peak_bytes / float(GIB_BYTES)
    bw_gbps = (hbm_tot / dur_s) / BYTES_PER_GB
    cube_pct = PCT_SCALE * cube_busy_s / dur_s
    vec_pct = (PCT_SCALE * (vec_ops / dur_s) / vec_peak) if vec_peak > 0.0 else 0.0
    return ts_mid, bw_gbps, cube_pct, vec_pct, ws_gib


def _phase_work_demand_us(
    phase: Any, op: OperatorExportRow, peaks: Mapping[str, float],
) -> tuple[float, bool, bool]:
    """Estimate one ``PipelinePhase`` duration (us) and ``(has_comp, has_comm)``."""
    cube_peak = float(op.cube_peak_flops_per_s or peaks.get("cube_peak_flops_per_s") or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s") or 0.0)
    hbm_peak = float(peaks.get("hbm_peak_bytes_per_s") or 0.0)
    link_peak = float(peaks.get("link_peak_bytes_per_s") or 0.0)
    best_us = 0.0
    has_comp = False
    has_comm = False
    for item in getattr(phase, "work", ()) or ():
        res = str(getattr(item, "resource", "") or "").strip().lower()
        amt = float(getattr(item, "amount", 0.0) or 0.0)
        if amt <= 0.0:
            continue
        if "link" in res or "hccl" in res or "comm" in res:
            has_comm = True
            if link_peak > 0.0:
                best_us = max(best_us, (amt / link_peak) * US_PER_SECOND)
        elif "cube" in res or "aic" in res:
            has_comp = True
            if cube_peak > 0.0:
                best_us = max(best_us, (amt / cube_peak) * US_PER_SECOND)
        elif "vec" in res or "aiv" in res:
            has_comp = True
            if vec_peak > 0.0:
                best_us = max(best_us, (amt / vec_peak) * US_PER_SECOND)
        elif "hbm" in res or "mem" in res or "mte" in res:
            if hbm_peak > 0.0:
                best_us = max(best_us, (amt / hbm_peak) * US_PER_SECOND)
    return best_us, has_comp, has_comm


def _schedule_pipeline_phase_intervals(
    pipeline_desc: Any,
    op: OperatorExportRow,
    ts_start: float,
    span_us: float,
    peaks: Mapping[str, float],
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]] | None:
    phases = tuple(getattr(pipeline_desc, "phases", ()) or ())
    if not phases:
        return None
    starts: dict[str, float] = {}
    ends: dict[str, float] = {}
    phase_flags: list[tuple[str, bool, bool]] = []
    for ph in phases:
        name = str(getattr(ph, "phase_id", "") or getattr(ph, "name", "") or "")
        dur_u, has_comp, has_comm = _phase_work_demand_us(ph, op, peaks)
        s0 = max((ends[dep] for dep in (getattr(ph, "after", ()) or ()) if dep in ends), default=0.0)
        starts[name] = s0
        ends[name] = s0 + max(dur_u, PIPE_MIN_SUBSPAN_US if (has_comp or has_comm) else 0.0)
        phase_flags.append((name, has_comp, has_comm))
    total_end = max(ends.values(), default=0.0)
    if total_end <= 0.0:
        return None
    scale = span_us / total_end
    comp_ivs: list[tuple[float, float]] = []
    comm_ivs: list[tuple[float, float]] = []
    for name, has_comp, has_comm in phase_flags:
        lo = ts_start + starts[name] * scale
        hi = ts_start + ends[name] * scale
        if hi <= lo:
            continue
        if has_comp:
            comp_ivs.append((lo, hi))
        if has_comm:
            comm_ivs.append((lo, hi))
    return (comp_ivs, comm_ivs) if (comp_ivs or comm_ivs) else None


def _trim_subspans_to_budget(
    parts: Sequence[tuple[float, float]], budget_us: float,
) -> list[tuple[float, float]]:
    if budget_us <= 0.0 or not parts:
        return []
    total = sum(hi - lo for lo, hi in parts if hi > lo)
    if total <= budget_us + _SPAN_GAP_EPS_US:
        return [(lo, hi) for lo, hi in parts if hi > lo]
    rem = budget_us
    out: list[tuple[float, float]] = []
    for lo, hi in parts:
        if hi <= lo or rem <= 0.0:
            continue
        take = min(hi - lo, rem)
        out.append((lo, lo + take))
        rem -= take
    return out


def _op_comp_comm_intervals(
    op: OperatorExportRow,
    ts_start: float,
    span_us: float,
    peaks: Mapping[str, float],
    *,
    pipeline_desc: Any = None,
    other_spans: Sequence[tuple[float, float]] = (),
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Project an operator's compute and communication overlap intervals without over-overlap."""
    if span_us <= 0.0:
        return [], []
    if pipeline_desc is not None:
        phased = _schedule_pipeline_phase_intervals(pipeline_desc, op, ts_start, span_us, peaks)
        if phased is not None:
            return phased
    if (op.end_us - op.start_us) > (span_us + _SPAN_GAP_EPS_US) and other_spans:
        cut = _merge_intervals(other_spans)
        parts = _trim_subspans_to_budget(_subtract_intervals([(op.start_us, op.end_us)], cut), span_us)
        if parts:
            if op.task_type == "HCCL":
                return [], parts
            return parts, (list(parts) if op.link_bytes > 0.0 else [])
    if op.task_type == "HCCL":
        comm_ivs = [(ts_start, ts_start + span_us)]
        comp_ivs: list[tuple[float, float]] = []
        if op.flops > 0.0 or op.vector_ops > 0.0:
            pipe_info = _compute_physical_pipe_times(op, peaks)
            comp_dur = min(span_us, max(pipe_info["cube_us"], pipe_info["vec_us"]))
            if comp_dur > 0.0:
                comp_ivs.append((ts_start, ts_start + comp_dur))
        return comp_ivs, comm_ivs
    comp_ivs = [(ts_start, ts_start + span_us)]
    comm_ivs = []
    if op.link_bytes > 0.0:
        link_peak = float(peaks.get("link_peak_bytes_per_s") or 0.0)
        comm_dur = min(span_us, (op.link_bytes / link_peak) * US_PER_SECOND) if link_peak > 0.0 else span_us
        if comm_dur > 0.0:
            comm_ivs.append((ts_start, ts_start + comm_dur))
    return comp_ivs, comm_ivs


def _sweep_direct_counter_samples(
    die_ops: Sequence[tuple[float, float, OperatorExportRow]],
    peaks: Mapping[str, float],
) -> list[tuple[float, float, float, float, float]]:
    """Aggregate concurrent direct-trace operators via piecewise-constant timestamp sweep."""
    if not die_ops:
        return []
    points = sorted({t for lo, hi, _ in die_ops for t in (lo, hi) if hi > lo})
    if len(points) < 2:
        return []
    cube_peak = float(peaks.get("cube_peak_flops_per_s", 0.0) or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s", 0.0) or 0.0)
    samples: list[tuple[float, float, float, float, float]] = []
    for idx in range(len(points) - 1):
        t0, t1 = points[idx], points[idx + 1]
        if t1 <= t0:
            continue
        mid = t0 + (t1 - t0) * MIDPOINT_FRACTION
        bw_gbps = 0.0
        cube_pct = 0.0
        vec_pct = 0.0
        ws_peak_bytes = 0.0
        for lo, hi, op in die_ops:
            if lo <= mid < hi:
                op_dur_s = max((hi - lo) / US_PER_SECOND, MIN_POSITIVE_SECONDS)
                bw_gbps += (op.hbm_total_bytes / op_dur_s) / BYTES_PER_GB
                if op.flops > 0.0:
                    op_pk = float(op.cube_peak_flops_per_s) if float(op.cube_peak_flops_per_s) > 0.0 else cube_peak
                    if op_pk > 0.0:
                        cube_pct += PCT_SCALE * (op.flops / op_pk) / op_dur_s
                if op.vector_ops > 0.0 and vec_peak > 0.0:
                    vec_pct += PCT_SCALE * (op.vector_ops / op_dur_s) / vec_peak
                ws_peak_bytes = max(ws_peak_bytes, max(op.hbm_write_bytes, op.mte3_bytes))
        samples.append((t0, bw_gbps, cube_pct, vec_pct, ws_peak_bytes / float(GIB_BYTES)))
    return samples


def _apply_exact_op_work(
    op: OperatorExportRow,
    work_map: Mapping[str, tuple[float, ...]],
    strip_fn: Any,
) -> OperatorExportRow:
    hit = _lookup_entry_op_timing(op, work_map, strip_fn)
    if hit is None or len(hit) < _WORK_TUPLE_LEN:
        return op
    return replace(
        op,
        flops=float(hit[0]),
        vector_ops=float(hit[1]),
        hbm_read_bytes=float(hit[2]),
        hbm_write_bytes=float(hit[3]),
        hbm_total_bytes=float(hit[4]),
        mte2_bytes=float(hit[5]),
        mte3_bytes=float(hit[6]),
        fixpipe_bytes=float(hit[7]),
        link_bytes=float(hit[8]),
        cube_peak_flops_per_s=float(hit[9]),
    )


def _resolve_host_launch_spec(
    op_eff: OperatorExportRow,
    entry: Any,
    *,
    is_graph_mode: bool,
    is_h2d_op: bool,
    bundle: ExportBundle | None = None,
) -> tuple[str, float]:
    """Resolve host launch API name and calibrated STARS/DMA launch duration (us)."""
    scope_lbl = str(getattr(entry, "scope", "") or op_eff.model_scope or "target")
    if is_h2d_op:
        dur = float(getattr(bundle, "h2d_startup_us", 0.0) or 0.0) if bundle is not None else 0.0
        return f"aclrtMemcpyAsync (PLE/Engram H2D: {op_eff.op_id})", max(0.0, dur)
    if is_graph_mode:
        dur = float(getattr(bundle, "graph_launch_us", 0.0) or 0.0) if bundle is not None else 0.0
        return f"aclGraphLaunch[{scope_lbl}]", max(0.0, dur)
    dur = float(getattr(bundle, "host_launch_us", 0.0) or 0.0) if bundle is not None else 0.0
    kern = op_eff.ascend_kernel or op_eff.implementation or op_eff.op_id
    return f"aclrtLaunchKernel({kern})", max(0.0, dur)


_ENTRY_STEP_RE = re.compile(r"\[Step\s+(\d+)\]")


def _resolve_entry_layer_rec(
    entry: Any, per_layer: Mapping[tuple[str, int], Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    m_step = _ENTRY_STEP_RE.search(str(getattr(entry, "label", "") or ""))
    step_suffix = f"_step{m_step.group(1)}" if m_step is not None else ""
    aid = str(getattr(entry, "archetype_id", "") or "")
    if aid in {"target_embed", "target_tail", "draft_pre", "draft_tail"}:
        if step_suffix:
            rec = per_layer.get((f"{aid}{step_suffix}", -1))
            if isinstance(rec, Mapping):
                return rec
        rec = per_layer.get((aid, -1))
        return rec if isinstance(rec, Mapping) else None
    if entry.full_layer_index is None:
        return None
    lyr = int(entry.full_layer_index)
    if aid == "draft_context_kv":
        if step_suffix:
            rec = per_layer.get((f"draft_context_kv{step_suffix}", lyr))
            if isinstance(rec, Mapping):
                return rec
        rec = per_layer.get(("draft_context_kv", lyr))
        if isinstance(rec, Mapping):
            return rec
    if entry.scope == "draft" and step_suffix:
        rec = per_layer.get((f"draft{step_suffix}", lyr))
        if isinstance(rec, Mapping):
            return rec
    rec = per_layer.get((entry.scope, lyr))
    return rec if isinstance(rec, Mapping) else None


def _entry_layer_span(
    entry: Any, tpl: Any, per_layer: Mapping[tuple[str, int], Mapping[str, Any]],
) -> float:
    rec = _resolve_entry_layer_rec(entry, per_layer)
    if isinstance(rec, Mapping) and float(rec.get("span_us", 0.0) or 0.0) > 0.0:
        return float(rec["span_us"])
    return float(tpl.span_us)


def _compute_scheduled_entry_windows(
    effective_schedule: Sequence[TraceScheduleEntry],
    templates: Mapping[str, _ArchetypeTemplate],
    per_layer: Mapping[tuple[str, int], Mapping[str, Any]],
    bundle: ExportBundle,
) -> tuple[list[tuple[TraceScheduleEntry, _ArchetypeTemplate, float, float, Mapping[str, Any] | None]], float]:
    windows: list[tuple[TraceScheduleEntry, _ArchetypeTemplate, float, float, Mapping[str, Any] | None]] = []
    ts_cursor = 0.0
    all_have_start = True
    for entry in effective_schedule:
        tpl = templates.get(entry.archetype_id)
        if tpl is None or not tpl.ops:
            continue
        rec = _resolve_entry_layer_rec(entry, per_layer)
        if isinstance(rec, Mapping) and "start_us" in rec:
            l_start = float(rec["start_us"])
            layer_span = float(rec.get("span_us", 0.0) or 0.0)
            ts_cursor = max(ts_cursor, l_start + layer_span)
        else:
            all_have_start = False
            l_start = ts_cursor
            layer_span = _entry_layer_span(entry, tpl, per_layer)
            ts_cursor = l_start + layer_span
        windows.append((entry, tpl, l_start, layer_span, rec))
    max_end = max((s + d for _, _, s, d, _ in windows), default=0.0)
    total_span_us = max(max_end, float(bundle.step_span_us or 0.0)) if (windows and all_have_start) else max_end
    return windows, total_span_us


def _lookup_entry_op_timing(
    op: OperatorExportRow,
    layer_ops_map: Mapping[str, tuple[float, float]],
    strip_fn: Any,
) -> tuple[float, float] | None:
    hit = layer_ops_map.get(strip_fn(op.op_id))
    if hit is not None:
        return hit
    if op.op_id.startswith("rewrite:") and op.absorbed_source_ops:
        fused_key = "rewrite_fused:" + "+".join(strip_fn(src) for src in op.absorbed_source_ops)
        return layer_ops_map.get(fused_key)
    return None


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
    mirror_placement: dict[Any, tuple[int, int, float, str]] | None = None,
    pending_flows: list[tuple[Any, tuple[int, int, float, str], float]] | None = None,
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
            pending_flows=pending_flows if rep_idx == 0 else None,
        )
        out.extend(events)
        task_id += 1
    if placement is not None:
        op_placement[op_eff.occurrence] = placement
        if mirror_placement is not None:
            mirror_placement[op_eff.occurrence] = placement
    return out, task_id, flow_id


def _flush_pending_flows(
    pending_flows: Sequence[tuple[Any, tuple[int, int, float, str], float]],
    placements: Mapping[Any, tuple[int, int, float, str]],
    flow_id: int,
) -> tuple[list[dict[str, Any]], int]:
    """Emit deferred ``SetFlag_WaitFlag`` flows whose producer occurred later in trace order."""
    out: list[dict[str, Any]] = []
    for pred_occ, (dst_pid, dst_tid, _dst_end, dst_name), dst_start_us in pending_flows:
        src = placements.get(pred_occ)
        if src is None:
            continue
        src_pid, src_tid, src_end_us, src_name = src
        if src_pid == dst_pid and src_tid == dst_tid:
            continue
        dst_ts = max(src_end_us, dst_start_us)
        sync_kind = "EventRecord/Wait" if src_pid != dst_pid else "StreamSync"
        f_args = {"src_kernel": src_name, "dst_kernel": dst_name, "sync_kind": sync_kind}
        out.append({
            "name": f"SetFlag->{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "s",
            "id": flow_id, "ts": _round_us(src_end_us), "pid": src_pid, "tid": src_tid, "args": f_args,
        })
        out.append({
            "name": f"WaitFlag<-{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "f", "bp": "e",
            "id": flow_id, "ts": _round_us(dst_ts), "pid": dst_pid, "tid": dst_tid, "args": f_args,
        })
        flow_id += 1
    return out, flow_id


@dataclass(slots=True)
class _PreparedOpVariant:
    dur_us: float
    dur_rounded: float
    base_args: dict[str, Any]
    pipe_spans_rel: tuple[tuple[int, str, float, float, dict[str, Any]], ...]
    comp_rel: tuple[tuple[float, float], ...]
    comm_rel: tuple[tuple[float, float], ...]
    hbm_tot: float
    vec_ops: float
    cube_busy_s: float
    ws_peak_bytes: float
    op_json_tpl: tuple[str, str, str, str] | None = None
    pipe_json_tpls: tuple[tuple[str, float, str], ...] = ()


@dataclass(slots=True)
class _PreparedOpSpec:
    op: OperatorExportRow
    d_idx: int
    base_pid: int
    hw_pid: int
    dev_pid: int
    dev_tid: int
    cat: str
    stripped_id: str
    fused_key: str | None
    k_name: str
    event_name: str
    is_h2d_op: bool
    in_reps: int
    rel_start_us: float
    pipeline_info: tuple[int, tuple[str, ...]] | None
    pipeline_desc: Any
    base_work: tuple[float, ...]
    variants: dict[tuple[float, tuple[float, ...]], _PreparedOpVariant]


def _prepare_template_ops(
    templates: Mapping[str, _ArchetypeTemplate],
    sched_counts: Mapping[str, int],
    die_map: Mapping[tuple[int, int, int], int],
    pipeline_by_occ: Mapping[Any, tuple[int, tuple[str, ...]]],
    pipeline_desc_by_occ: Mapping[Any, Any],
) -> dict[str, tuple[_PreparedOpSpec, ...]]:
    prepared: dict[str, tuple[_PreparedOpSpec, ...]] = {}
    for aid, tpl in templates.items():
        entry_sched_cnt = max(1, int(sched_counts.get(aid, 1)))
        specs: list[_PreparedOpSpec] = []
        for op in tpl.ops:
            coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
            d_idx = die_map.get(coord, 0)
            base_pid = _pid_base(d_idx)
            dev_pid, dev_tid = _op_pid_tid(op, base_pid)
            stripped_id = strip_op_prefix(op.op_id)
            fused_key = (
                "rewrite_fused:" + "+".join(strip_op_prefix(src) for src in op.absorbed_source_ops)
                if op.op_id.startswith("rewrite:") and op.absorbed_source_ops
                else None
            )
            k_name = op.ascend_kernel or op.logical_role or op.canonical_op_key
            op_tag = stripped_id or op.canonical_op_key
            event_name = f"{k_name} [{op_tag}]" if (op_tag and op_tag != k_name) else k_name
            specs.append(_PreparedOpSpec(
                op=op,
                d_idx=d_idx,
                base_pid=base_pid,
                hw_pid=base_pid + PID_OFFSET_HARDWARE,
                dev_pid=dev_pid,
                dev_tid=dev_tid,
                cat="Communication" if op.task_type == "HCCL" else "Kernel",
                stripped_id=stripped_id,
                fused_key=fused_key,
                k_name=k_name,
                event_name=event_name,
                is_h2d_op=(op.family == "engram_rh2d" or "host_uva" in op.op_id.lower()),
                in_reps=_op_in_entry_repeats(op, entry_sched_cnt),
                rel_start_us=max(0.0, op.start_us - tpl.min_start_us),
                pipeline_info=pipeline_by_occ.get(op.occurrence),
                pipeline_desc=pipeline_desc_by_occ.get(op.occurrence),
                base_work=(
                    float(op.flops),
                    float(op.vector_ops),
                    float(op.hbm_read_bytes),
                    float(op.hbm_write_bytes),
                    float(op.hbm_total_bytes),
                    float(op.mte2_bytes),
                    float(op.mte3_bytes),
                    float(op.fixpipe_bytes),
                    float(op.link_bytes),
                    float(op.cube_peak_flops_per_s),
                ),
                variants={},
            ))
        prepared[aid] = tuple(specs)
    return prepared


def _encode_variant_json_templates(
    spec: _PreparedOpSpec,
    dur_rounded: float,
    base_args: Mapping[str, Any],
    pipe_spans: Sequence[tuple[int, str, float, float, dict[str, Any]]],
    json_encode: Any,
) -> tuple[tuple[str, str, str, str], tuple[tuple[str, float, str], ...]]:
    args_before = {k: base_args[k] for k in _ARGS_PREFIX_KEYS}
    args_after = {k: v for k, v in base_args.items() if k not in _ARGS_DYNAMIC_KEYS}
    op_head = json_encode({"name": spec.event_name, "cat": spec.cat, "ph": "X"})[:-1] + ',"ts":'
    op_mid1 = (
        f',"dur":{json_encode(dur_rounded)},"pid":{spec.dev_pid},"tid":{spec.dev_tid},"args":'
        + json_encode(args_before)[:-1]
        + ',"Task Id":'
    )
    op_mid2 = ',"Layer":'
    op_tail = "," + json_encode(args_after)[1:] + "}"
    p_tpls: list[tuple[str, float, str]] = []
    for p_tid, p_name, rel_off, p_dur_r, p_args in pipe_spans:
        p_head = json_encode({"name": p_name, "cat": "Pipe", "ph": "X"})[:-1] + ',"ts":'
        p_tail = (
            f',"dur":{json_encode(p_dur_r)},"pid":{spec.hw_pid},"tid":{p_tid},'
            f'"args":{json_encode(p_args)}}}'
        )
        p_tpls.append((p_head, rel_off, p_tail))
    return (op_head, op_mid1, op_mid2, op_tail), tuple(p_tpls)


def _build_prepared_variant(
    bundle: ExportBundle,
    spec: _PreparedOpSpec,
    dur_us: float,
    work_tuple: tuple[float, ...],
    *,
    include_pipe_lanes: bool,
    cube_peak_default: float,
    vec_peak: float,
    mte2_peak: float,
    mte3_peak: float,
    fp_peak: float,
    link_peak: float,
    json_encode: Any = None,
) -> _PreparedOpVariant:
    """[Cohesive-Pipeline] Build one pre-encoded Chrome/msprof trace variant with pipe-lane spans."""
    op = spec.op
    flops, vec_ops, hbm_r, hbm_w, hbm_tot, mte2_b, mte3_b, fp_raw, link_b, op_cube_pk = work_tuple
    dur = max(0.0, float(dur_us))
    cube_peak = op_cube_pk if op_cube_pk > 0.0 else cube_peak_default
    fp_bytes = max(0.0, fp_raw)

    cube_us = min(dur, (flops / cube_peak) * US_PER_SECOND) if (flops > 0.0 and cube_peak > 0.0 and dur > 0.0) else 0.0
    vec_us = min(dur, (vec_ops / vec_peak) * US_PER_SECOND) if (vec_ops > 0.0 and vec_peak > 0.0 and dur > 0.0) else 0.0
    mte2_us = min(dur, (mte2_b / mte2_peak) * US_PER_SECOND) if (mte2_b > 0.0 and mte2_peak > 0.0 and dur > 0.0) else 0.0
    mte3_us = min(dur, (mte3_b / mte3_peak) * US_PER_SECOND) if (mte3_b > 0.0 and mte3_peak > 0.0 and dur > 0.0) else 0.0
    fp_us = min(dur, (fp_bytes / fp_peak) * US_PER_SECOND) if (fp_bytes > 0.0 and fp_peak > 0.0 and dur > 0.0) else 0.0

    aic_ratio = round(min(1.0, cube_us / dur), RATIO_DECIMALS) if (dur > 0.0 and cube_us > 0.0) else 0.0
    aiv_ratio = round(min(1.0, vec_us / dur), RATIO_DECIMALS) if (dur > 0.0 and vec_us > 0.0) else 0.0
    mte2_ratio = round(min(1.0, mte2_us / dur), RATIO_DECIMALS) if (dur > 0.0 and mte2_us > 0.0) else 0.0
    mte3_ratio = round(min(1.0, mte3_us / dur), RATIO_DECIMALS) if (dur > 0.0 and mte3_us > 0.0) else 0.0
    fp_ratio = round(min(1.0, fp_us / dur), RATIO_DECIMALS) if (dur > 0.0 and fp_us > 0.0) else 0.0

    dur_rounded = _round_us(dur_us)
    base_args: dict[str, Any] = {
        "Model Name": bundle.model_name,
        "Hardware SKU": bundle.hardware_sku,
        "Phase": bundle.phase,
        "Task Type": op.task_type,
        "Stream Id": op.stream_index,
        "Task Id": 0,
        "Layer": "",
        "Archetype": op.archetype_title or op.layer_label,
        "Submodule": op.submodule,
        "Logical Role": op.logical_role,
        "Ascend Kernel": op.ascend_kernel,
        "Kernel Backend": op.kernel_backend,
        "Kernel Status": op.kernel_status,
        "Source Ref": op.source_ref,
        "Shapes & Dtype": op.tensor_contract,
        "FLOPs": round(flops, 1),
        "Vector Ops": round(vec_ops, 1),
        "HBM Read (B)": round(hbm_r, 1),
        "HBM Write (B)": round(hbm_w, 1),
        "HBM Total (B)": round(hbm_tot, 1),
        "Link Bytes (B)": round(link_b, 1),
        "L2 Hit (%)": _round_metric(op.l2_hit_pct),
        "Bottleneck": op.bound_display or op.dominant_bound,
        "aic_mac_ratio": aic_ratio,
        "aiv_vec_ratio": aiv_ratio,
        "mte2_ratio": mte2_ratio,
        "mte3_ratio": mte3_ratio,
        "fixpipe_ratio": fp_ratio,
        "L* Physical Lower Bound (us)": _round_us(op.l_star_lean_us),
        "Sim Pred (us)": dur_rounded,
        "Gap vs L* (us)": round(op.gap_us, TIMESTAMP_US_DECIMALS),
        "Physical-Limit Gap Ratio (T_pred / L*)": (
            round(op.gap_ratio, RATIO_DECIMALS) if op.gap_ratio is not None else None
        ),
        "Gap Radar Rank": op.gap_radar_rank,
    }
    if spec.pipeline_info is not None:
        base_args["Fused Phases"] = spec.pipeline_info[0]
        if spec.pipeline_info[1]:
            base_args["Fused Members"] = ",".join(spec.pipeline_info[1])
    if op.measured_sum_us is not None:
        base_args["Measured (us)"] = _round_us(op.measured_sum_us)
        base_args["Measured Kernels"] = len(op.measured_kernels)
    if op.ape_pct is not None:
        base_args["APE (%)"] = _round_metric(op.ape_pct)

    has_pipe_work = (
        flops > 0.0 or vec_ops > 0.0 or mte2_b > 0.0 or mte3_b > 0.0 or fp_raw > 0.0
    )
    pipe_spans: list[tuple[int, str, float, float, dict[str, Any]]] = []
    if include_pipe_lanes and dur_us > 0.0 and (op.task_type != "HCCL" or has_pipe_work):
        k_name = spec.k_name
        d_mte2 = min(dur_us, max(PIPE_MIN_SUBSPAN_US, mte2_us)) if mte2_b > 0.0 else 0.0
        d_cube = min(dur_us, max(PIPE_MIN_SUBSPAN_US, cube_us)) if flops > 0.0 else 0.0
        d_vec = min(dur_us, max(PIPE_MIN_SUBSPAN_US, vec_us)) if vec_ops > 0.0 else 0.0
        d_fp = min(dur_us, max(PIPE_MIN_SUBSPAN_US, fp_us)) if fp_bytes > 0.0 else 0.0
        d_mte3 = min(dur_us, max(PIPE_MIN_SUBSPAN_US, mte3_us)) if mte3_b > 0.0 else 0.0
        store_tail = max(d_fp, d_mte3)
        if d_mte2 > 0.0:
            pipe_spans.append((
                TID_PIPE_MTE2, f"{k_name}:MTE2", 0.0, _round_us(d_mte2),
                {"Pipe": "MTE2", "Bytes": round(mte2_b, 1), "mte2_ratio": mte2_ratio, "Kernel": k_name},
            ))
        if d_cube > 0.0:
            pipe_spans.append((
                TID_PIPE_CUBE, f"{k_name}:Cube", max(0.0, min(dur_us - d_cube, (dur_us - store_tail) - d_cube)),
                _round_us(d_cube),
                {"Pipe": "Cube(AIC)", "FLOPs": round(flops, 1), "aic_mac_ratio": aic_ratio, "Dtype": op.dtype, "Kernel": k_name},
            ))
        if d_vec > 0.0:
            pipe_spans.append((
                TID_PIPE_VECTOR, f"{k_name}:Vector", max(0.0, min(dur_us - d_vec, (dur_us - d_mte3) - d_vec)),
                _round_us(d_vec),
                {"Pipe": "Vector(AIV)", "Vector Ops": round(vec_ops, 1), "aiv_vec_ratio": aiv_ratio, "Kernel": k_name},
            ))
        if d_fp > 0.0:
            pipe_spans.append((
                TID_PIPE_FIXPIPE, f"{k_name}:Fixpipe", max(0.0, dur_us - d_fp), _round_us(d_fp),
                {"Pipe": "Fixpipe", "Bytes": round(fp_bytes, 1), "fixpipe_ratio": fp_ratio, "Kernel": k_name},
            ))
        if d_mte3 > 0.0:
            pipe_spans.append((
                TID_PIPE_MTE3, f"{k_name}:MTE3", max(0.0, dur_us - d_mte3), _round_us(d_mte3),
                {"Pipe": "MTE3", "Bytes": round(mte3_b, 1), "mte3_ratio": mte3_ratio, "Kernel": k_name},
            ))

    comp_rel: list[tuple[float, float]] = []
    comm_rel: list[tuple[float, float]] = []
    if dur_us > 0.0:
        phased = None
        if spec.pipeline_desc is not None:
            op_for_pipe = op if work_tuple == spec.base_work else replace(op, cube_peak_flops_per_s=op_cube_pk)
            phased = _schedule_pipeline_phase_intervals(spec.pipeline_desc, op_for_pipe, 0.0, dur_us, bundle.hardware_peaks)
        if phased is not None:
            comp_rel, comm_rel = phased
        elif op.task_type == "HCCL":
            comm_rel.append((0.0, dur_us))
            if flops > 0.0 or vec_ops > 0.0:
                comp_dur = min(dur_us, max(cube_us, vec_us))
                if comp_dur > 0.0:
                    comp_rel.append((0.0, comp_dur))
        else:
            comp_rel.append((0.0, dur_us))
            if link_b > 0.0:
                comm_dur = min(dur_us, (link_b / link_peak) * US_PER_SECOND) if link_peak > 0.0 else dur_us
                if comm_dur > 0.0:
                    comm_rel.append((0.0, comm_dur))

    cube_busy_s = (flops / cube_peak) if (flops > 0.0 and cube_peak > 0.0) else 0.0
    op_json_tpl: tuple[str, str, str, str] | None = None
    pipe_json_tpls: tuple[tuple[str, float, str], ...] = ()
    if json_encode is not None:
        op_json_tpl, pipe_json_tpls = _encode_variant_json_templates(
            spec, dur_rounded, base_args, pipe_spans, json_encode,
        )
    return _PreparedOpVariant(
        dur_us=dur_us,
        dur_rounded=dur_rounded,
        base_args=base_args,
        pipe_spans_rel=tuple(pipe_spans),
        comp_rel=tuple(comp_rel),
        comm_rel=tuple(comm_rel),
        hbm_tot=hbm_tot,
        vec_ops=vec_ops,
        cube_busy_s=cube_busy_s,
        ws_peak_bytes=max(hbm_w, mte3_b),
        op_json_tpl=op_json_tpl,
        pipe_json_tpls=pipe_json_tpls,
    )


def _iter_scheduled_full_model_trace(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
    pipeline_by_occ: Mapping[Any, tuple[int, tuple[str, ...]]],
    *,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
    pipeline_desc_by_occ: Mapping[Any, Any] = (),
    _json_encode: Any = None,
) -> Iterator[Any]:
    """Stream full multi-layer Target + Draft schedule using native per-layer timings when recorded.

    [Cohesive-Pipeline] Single-pass chronological trace event generator over target and draft layer schedules.
    """
    raw_mode = _json_encode is not None
    effective_schedule, sched_counts = _expand_effective_trace_schedule(bundle)
    templates = _build_archetype_templates(bundle, sched_counts)
    prepared_by_arch = _prepare_template_ops(
        templates, sched_counts, die_map, pipeline_by_occ, pipeline_desc_by_occ,
    )
    per_layer = dict(getattr(bundle, "per_layer_timings", None) or {})
    entry_windows, total_span_us = _compute_scheduled_entry_windows(
        effective_schedule, templates, per_layer, bundle,
    )
    track_display = format_track_label(bundle.feature_track, lang="en")
    root_ev = {
        "name": f"ModelRunner.execute_model [{bundle.model_name} | {bundle.phase.upper()} | graph={bundle.graph_capture_mode}]",
        "cat": "Framework", "ph": "X", "ts": 0.0, "dur": _round_us(total_span_us),
        "pid": PID_FRAMEWORK, "tid": TID_FW_ENGINE,
        "args": {
            "Model": bundle.model_name, "Hardware": bundle.hardware_sku,
            "Target Layers": bundle.total_target_layers, "Draft Layers": bundle.total_draft_layers,
            "Graph Capture": bundle.graph_capture_mode, "Feature Track": track_display,
        },
    }
    yield ("X", _json_encode(root_ev)) if raw_mode else root_ev

    peaks = bundle.hardware_peaks
    cube_peak_default = float(peaks.get("cube_peak_flops_per_s") or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s") or 0.0)
    mte2_peak = float(peaks.get("mte2_peak_bytes_per_s") or 0.0)
    mte3_peak = float(peaks.get("mte3_peak_bytes_per_s") or 0.0)
    fp_peak = float(peaks.get("fixpipe_peak_bytes_per_s") or 0.0)
    link_peak = float(peaks.get("link_peak_bytes_per_s") or 0.0)

    is_graph_mode = bundle.graph_capture_mode.lower() not in {"eager", "none", "off"}
    comp_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    comm_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    samples_by_die: dict[int, list[tuple[float, float, float, float, float]]] = defaultdict(list)

    task_id = 1
    flow_id = 1
    launched_scopes: set[str] = set()
    scope_bounds: dict[str, list[float]] = {}
    global_placement: dict[Any, tuple[int, int, float, str]] = {}
    pending_flows: list[tuple[Any, tuple[int, int, float, str], float]] = []

    for entry, tpl, l_start, layer_span, layer_rec in entry_windows:
        layer_ops_map = layer_rec.get("ops", {}) if isinstance(layer_rec, Mapping) else {}
        layer_work_map = layer_rec.get("ops_work", {}) if isinstance(layer_rec, Mapping) else {}
        l_end = l_start + layer_span
        if entry.scope not in scope_bounds:
            scope_bounds[entry.scope] = [l_start, l_end]
        else:
            scope_bounds[entry.scope][0] = min(scope_bounds[entry.scope][0], l_start)
            scope_bounds[entry.scope][1] = max(scope_bounds[entry.scope][1], l_end)

        layer_ev = {
            "name": entry.label, "cat": "MSTX_Layer", "ph": "X",
            "ts": _round_us(l_start), "dur": _round_us(layer_span),
            "pid": PID_FRAMEWORK, "tid": TID_FW_LAYER_MSTX,
            "args": {"Scope": entry.scope, "Archetype": entry.archetype_id, "Layer": entry.full_layer_index},
        }
        yield ("X", _json_encode(layer_ev)) if raw_mode else layer_ev
        scale_span = (layer_span / tpl.span_us) if tpl.span_us > 0.0 else 1.0
        for sub_name, r_lo, r_hi in tpl.sub_spans:
            if r_hi <= r_lo:
                continue
            sub_ev = {
                "name": f"{sub_name} ({entry.label})", "cat": "MSTX_Submodule", "ph": "X",
                "ts": _round_us(l_start + r_lo * scale_span), "dur": _round_us((r_hi - r_lo) * scale_span),
                "pid": PID_FRAMEWORK, "tid": TID_FW_SUBMODULE_MSTX,
                "args": {"Submodule": sub_name, "Layer": entry.label},
            }
            yield ("X", _json_encode(sub_ev)) if raw_mode else sub_ev

        layer_json = _json_encode(entry.label) if raw_mode else ""
        local_placement: dict[Any, tuple[int, int, float, str]] = {}
        first_op_for_scope = entry.scope not in launched_scopes
        entry_comp: dict[int, list[tuple[float, float]]] = defaultdict(list)
        entry_comm: dict[int, list[tuple[float, float]]] = defaultdict(list)
        entry_metrics: dict[int, list[float]] = {}

        for spec in prepared_by_arch[entry.archetype_id]:
            op = spec.op
            exact_timing = None
            if layer_ops_map:
                exact_timing = layer_ops_map.get(spec.stripped_id)
                if exact_timing is None and spec.fused_key is not None:
                    exact_timing = layer_ops_map.get(spec.fused_key)
                if exact_timing is None:
                    continue
            work_tuple = spec.base_work
            if exact_timing is not None:
                op_ts = l_start + float(exact_timing[0])
                op_dur = float(exact_timing[1])
                if layer_work_map:
                    w_hit = layer_work_map.get(spec.stripped_id)
                    if w_hit is None and spec.fused_key is not None:
                        w_hit = layer_work_map.get(spec.fused_key)
                    if w_hit is not None and len(w_hit) >= _WORK_TUPLE_LEN:
                        h0, h1, h2, h3, h4, h5, h6, h7, h8, h9 = w_hit[:_WORK_TUPLE_LEN]
                        work_tuple = (
                            float(h0), float(h1), float(h2), float(h3), float(h4),
                            float(h5), float(h6), float(h7), float(h8), float(h9),
                        )
            else:
                op_ts = l_start + spec.rel_start_us
                op_dur = op.dur_us

            if first_op_for_scope or not is_graph_mode or spec.is_h2d_op:
                launch_name, l_dur = _resolve_host_launch_spec(
                    op, entry, is_graph_mode=is_graph_mode, is_h2d_op=spec.is_h2d_op, bundle=bundle,
                )
                l_events, flow_id = _emit_host_launch_with_flow(
                    name=launch_name, dev_ts=op_ts, dev_pid=spec.dev_pid, dev_tid=spec.dev_tid,
                    launch_dur_us=l_dur,
                    flow_id=flow_id, extra_args={"scope": entry.scope, "layer": entry.label, "transfer": "H2D" if spec.is_h2d_op else "launch"},
                )
                if raw_mode:
                    for ev in l_events:
                        yield (ev.get("ph"), _json_encode(ev))
                else:
                    yield from l_events
                if first_op_for_scope:
                    launched_scopes.add(entry.scope)
                    first_op_for_scope = False

            use_global_placement = (
                entry.full_layer_index is None
                or op.layer is None
                or int(entry.full_layer_index) == int(op.layer)
            )
            place_primary = global_placement if use_global_placement else local_placement
            place_mirror = local_placement if use_global_placement else global_placement
            d_idx = spec.d_idx

            if spec.in_reps > 1:
                op_eff = op
                if exact_timing is not None:
                    op_eff = replace(op, dur_us=op_dur, end_us=op.start_us + op_dur)
                    if layer_work_map:
                        op_eff = _apply_exact_op_work(op_eff, layer_work_map, strip_op_prefix)
                events, task_id, flow_id = _emit_entry_op_repeats(
                    bundle, op_eff, op_ts=op_ts, in_reps=spec.in_reps, task_id=task_id,
                    layer_label=entry.label, base_pid=spec.base_pid, flow_id=flow_id,
                    include_pipe_lanes=include_pipe_lanes,
                    pipeline_info=spec.pipeline_info,
                    op_placement=place_primary,
                    mirror_placement=place_mirror,
                    pending_flows=pending_flows if use_global_placement else None,
                )
                if raw_mode:
                    for ev in events:
                        yield (ev.get("ph"), _json_encode(ev))
                else:
                    yield from events
                op_span = op_eff.dur_us * float(spec.in_reps)
                if op_span > 0.0:
                    c_ivs, m_ivs = _op_comp_comm_intervals(
                        op_eff, op_ts, op_span, peaks, pipeline_desc=spec.pipeline_desc,
                    )
                    entry_comp[d_idx].extend(c_ivs)
                    entry_comm[d_idx].extend(m_ivs)
                if include_counters:
                    scale = float(spec.in_reps)
                    acc = entry_metrics.get(d_idx)
                    if acc is None:
                        acc = [0.0, 0.0, 0.0, 0.0]
                        entry_metrics[d_idx] = acc
                    acc[0] += op_eff.hbm_total_bytes * scale
                    acc[2] += op_eff.vector_ops * scale
                    if op_eff.flops > 0.0:
                        eff_pk = float(op_eff.cube_peak_flops_per_s) if float(op_eff.cube_peak_flops_per_s) > 0.0 else cube_peak_default
                        if eff_pk > 0.0:
                            acc[1] += (op_eff.flops * scale) / eff_pk
                    acc[3] = max(acc[3], max(op_eff.hbm_write_bytes, op_eff.mte3_bytes))
                continue

            v_key = (op_dur, work_tuple)
            variant = spec.variants.get(v_key)
            if variant is None:
                variant = _build_prepared_variant(
                    bundle, spec, op_dur, work_tuple,
                    include_pipe_lanes=include_pipe_lanes,
                    cube_peak_default=cube_peak_default,
                    vec_peak=vec_peak,
                    mte2_peak=mte2_peak,
                    mte3_peak=mte3_peak,
                    fp_peak=fp_peak,
                    link_peak=link_peak,
                    json_encode=_json_encode,
                )
                if len(spec.variants) < _MAX_OP_VARIANT_CACHE:
                    spec.variants[v_key] = variant

            ts_rounded = _round_us(op_ts)
            if raw_mode:
                if not math.isfinite(ts_rounded):
                    raise ValueError("Out of range float values are not JSON compliant")
                op_head, op_mid1, op_mid2, op_tail = variant.op_json_tpl
                yield ("X", f"{op_head}{ts_rounded}{op_mid1}{task_id}{op_mid2}{layer_json}{op_tail}")
            else:
                op_args = variant.base_args.copy()
                op_args["Task Id"] = task_id
                op_args["Layer"] = entry.label
                yield {
                    "name": spec.event_name,
                    "cat": spec.cat,
                    "ph": "X",
                    "ts": ts_rounded,
                    "dur": variant.dur_rounded,
                    "pid": spec.dev_pid,
                    "tid": spec.dev_tid,
                    "args": op_args,
                }
            task_id += 1
            placement = (spec.dev_pid, spec.dev_tid, op_ts + op_dur, spec.k_name)

            for pred in op.predecessors:
                src = place_primary.get(pred)
                if src is None:
                    if use_global_placement:
                        pending_flows.append((pred, placement, op_ts))
                    continue
                src_pid, src_tid, src_end_us, src_name = src
                if src_pid == spec.dev_pid and src_tid == spec.dev_tid:
                    continue
                dst_ts = max(src_end_us, op_ts)
                sync_kind = "EventRecord/Wait" if src_pid != spec.dev_pid else "StreamSync"
                f_args = {"src_kernel": src_name, "dst_kernel": spec.k_name, "sync_kind": sync_kind}
                s_ev = {
                    "name": f"SetFlag->{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "s",
                    "id": flow_id, "ts": _round_us(src_end_us), "pid": src_pid, "tid": src_tid, "args": f_args,
                }
                f_ev = {
                    "name": f"WaitFlag<-{sync_kind}", "cat": "SetFlag_WaitFlag", "ph": "f", "bp": "e",
                    "id": flow_id, "ts": _round_us(dst_ts), "pid": spec.dev_pid, "tid": spec.dev_tid, "args": f_args,
                }
                if raw_mode:
                    yield ("s", _json_encode(s_ev))
                    yield ("f", _json_encode(f_ev))
                else:
                    yield s_ev
                    yield f_ev
                flow_id += 1

            if raw_mode:
                for p_head, rel_off, p_tail in variant.pipe_json_tpls:
                    p_ts = ts_rounded if rel_off == 0.0 else _round_us(op_ts + rel_off)
                    if not math.isfinite(p_ts):
                        raise ValueError("Out of range float values are not JSON compliant")
                    yield ("X", f"{p_head}{p_ts}{p_tail}")
            else:
                for p_tid, p_name, rel_off, p_dur_r, p_args in variant.pipe_spans_rel:
                    yield {
                        "name": p_name, "cat": "Pipe", "ph": "X",
                        "ts": ts_rounded if rel_off == 0.0 else _round_us(op_ts + rel_off),
                        "dur": p_dur_r,
                        "pid": spec.hw_pid, "tid": p_tid, "args": p_args.copy(),
                    }

            place_primary[op.occurrence] = placement
            place_mirror[op.occurrence] = placement

            if variant.comp_rel:
                c_list = entry_comp[d_idx]
                for r_lo, r_hi in variant.comp_rel:
                    lo = op_ts + r_lo
                    hi = op_ts + r_hi
                    if hi > lo:
                        c_list.append((lo, hi))
            if variant.comm_rel:
                m_list = entry_comm[d_idx]
                for r_lo, r_hi in variant.comm_rel:
                    lo = op_ts + r_lo
                    hi = op_ts + r_hi
                    if hi > lo:
                        m_list.append((lo, hi))
            if include_counters:
                acc = entry_metrics.get(d_idx)
                if acc is None:
                    entry_metrics[d_idx] = [variant.hbm_tot, variant.cube_busy_s, variant.vec_ops, variant.ws_peak_bytes]
                else:
                    acc[0] += variant.hbm_tot
                    acc[1] += variant.cube_busy_s
                    acc[2] += variant.vec_ops
                    if variant.ws_peak_bytes > acc[3]:
                        acc[3] = variant.ws_peak_bytes

        # Accumulate compact per-layer overlap intervals & counter samples strictly from emitted ops.
        for d_k, ivs in entry_comp.items():
            comp_by_die[d_k].extend(_merge_intervals(ivs))
        for d_k, ivs in entry_comm.items():
            comm_by_die[d_k].extend(_merge_intervals(ivs))
        if include_counters and layer_span > 0.0:
            dur_s = max(layer_span / US_PER_SECOND, MIN_POSITIVE_SECONDS)
            ts_mid = l_start + layer_span * MIDPOINT_FRACTION
            for d_k, acc in entry_metrics.items():
                ws_gib = acc[3] / float(GIB_BYTES)
                bw_gbps = (acc[0] / dur_s) / BYTES_PER_GB
                cube_pct = PCT_SCALE * acc[1] / dur_s
                vec_pct = (PCT_SCALE * (acc[2] / dur_s) / vec_peak) if vec_peak > 0.0 else 0.0
                samples_by_die[d_k].append((ts_mid, bw_gbps, cube_pct, vec_pct, ws_gib))

    deferred_events, flow_id = _flush_pending_flows(pending_flows, global_placement, flow_id)
    if raw_mode:
        for ev in deferred_events:
            yield (ev.get("ph"), _json_encode(ev))
    else:
        yield from deferred_events

    for sc_key, (s_lo, s_hi) in scope_bounds.items():
        if s_hi > s_lo:
            sc_title = (
                f"Target Forward Pass ({bundle.total_target_layers} Layers)" if sc_key == "target"
                else f"Speculative Draft Forward Pass ({bundle.total_draft_layers} Layers)"
            )
            sc_ev = {
                "name": sc_title, "cat": "Framework", "ph": "X",
                "ts": _round_us(s_lo), "dur": _round_us(s_hi - s_lo),
                "pid": PID_FRAMEWORK, "tid": TID_FW_ENGINE,
                "args": {"Scope": sc_key},
            }
            yield ("X", _json_encode(sc_ev)) if raw_mode else sc_ev

    stage_by_die = {idx: int(coord[1]) for coord, idx in die_map.items()}
    if include_overlap:
        for ev in _iter_overlap_from_intervals(comp_by_die, comm_by_die, sorted(set(die_map.values())), total_span_us):
            yield (ev.get("ph"), _json_encode(ev)) if raw_mode else ev
    if include_counters:
        for ev in _iter_counter_events_from_samples(bundle, samples_by_die, stage_by_die=stage_by_die):
            yield (ev.get("ph"), _json_encode(ev)) if raw_mode else ev


def _iter_direct_trace(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
    pipeline_by_occ: Mapping[Any, tuple[int, tuple[str, ...]]],
    *,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
    pipeline_desc_by_occ: Mapping[Any, Any] = (),
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
    active_ops_by_die: dict[int, list[tuple[float, float, OperatorExportRow]]] = defaultdict(list)
    spans_by_die: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
    for idx_op, op_item in enumerate(bundle.operators):
        if op_item.dur_us > 0.0:
            coord_i = (op_item.occurrence.replica, op_item.occurrence.stage, int(op_item.occurrence.rank or 0))
            d_i = die_map.get(coord_i, 0)
            spans_by_die[d_i].append((idx_op, op_item.start_us, op_item.start_us + op_item.dur_us))
    op_placement: dict[Any, tuple[int, int, float, str]] = {}
    pending_flows: list[tuple[Any, tuple[int, int, float, str], float]] = []
    layer_spans: list[tuple[str, float, float]] = []
    sub_spans: list[tuple[str, str, float, float]] = []
    flow_id = 1
    is_graph_mode = bundle.graph_capture_mode.lower() not in {"eager", "none", "off"}
    launched = False

    for task_id, op in enumerate(bundle.operators, start=1):
        coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        d_idx = die_map.get(coord, 0)
        base = _pid_base(d_idx)
        is_h2d_op = op.family == "engram_rh2d" or "host_uva" in op.op_id.lower()
        if (not launched) or (not is_graph_mode) or is_h2d_op:
            dev_pid, dev_tid = _op_pid_tid(op, base)
            launch_name, l_dur = _resolve_host_launch_spec(
                op, None, is_graph_mode=is_graph_mode, is_h2d_op=is_h2d_op, bundle=bundle,
            )
            l_events, flow_id = _emit_host_launch_with_flow(
                name=launch_name, dev_ts=op.start_us, dev_pid=dev_pid, dev_tid=dev_tid,
                launch_dur_us=l_dur, flow_id=flow_id,
                extra_args={"scope": op.model_scope, "transfer": "H2D" if is_h2d_op else "launch"},
            )
            yield from l_events
            launched = True
        lbl = op.archetype_title or op.layer_label
        events, placement, flow_id = _emit_op_and_pipes(
            bundle, op, ts_start=op.start_us, task_id=task_id, layer_label=lbl,
            base_pid=base, flow_id=flow_id, include_pipe_lanes=include_pipe_lanes,
            pipeline_info=pipeline_by_occ.get(op.occurrence), op_placement=op_placement,
            pending_flows=pending_flows,
        )
        yield from events
        op_placement[op.occurrence] = placement
        if op.dur_us > 0.0:
            other_ivs = (
                [
                    (s_lo, s_hi)
                    for o_idx, s_lo, s_hi in spans_by_die.get(d_idx, ())
                    if o_idx != (task_id - 1) and s_lo >= op.start_us and s_hi <= op.end_us
                ]
                if (op.end_us - op.start_us) > (op.dur_us + _SPAN_GAP_EPS_US)
                else ()
            )
            c_ivs, m_ivs = _op_comp_comm_intervals(
                op, op.start_us, op.dur_us, bundle.hardware_peaks,
                pipeline_desc=pipeline_desc_by_occ.get(op.occurrence),
                other_spans=other_ivs,
            )
            comp_by_die[d_idx].extend(c_ivs)
            comm_by_die[d_idx].extend(m_ivs)
            if include_counters:
                active_ops_by_die[d_idx].append((op.start_us, op.start_us + op.dur_us, op))
        if layer_spans and layer_spans[-1][0] == lbl:
            layer_spans[-1] = (lbl, min(layer_spans[-1][1], op.start_us), max(layer_spans[-1][2], op.end_us))
        else:
            layer_spans.append((lbl, op.start_us, op.end_us))
        if sub_spans and sub_spans[-1][0] == lbl and sub_spans[-1][1] == op.submodule:
            sub_spans[-1] = (lbl, op.submodule, min(sub_spans[-1][2], op.start_us), max(sub_spans[-1][3], op.end_us))
        else:
            sub_spans.append((lbl, op.submodule, op.start_us, op.end_us))

    deferred_events, flow_id = _flush_pending_flows(pending_flows, op_placement, flow_id)
    yield from deferred_events

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
    stage_by_die = {idx: int(coord[1]) for coord, idx in die_map.items()}
    if include_overlap:
        yield from _iter_overlap_from_intervals(comp_by_die, comm_by_die, sorted(set(die_map.values())), bundle.step_span_us)
    if include_counters:
        samples_by_die = {
            d_k: _sweep_direct_counter_samples(d_ops, bundle.hardware_peaks)
            for d_k, d_ops in active_ops_by_die.items()
        }
        yield from _iter_counter_events_from_samples(bundle, samples_by_die, stage_by_die=stage_by_die)


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
    exposed_sum_ns = sum(op_exposed_int_ns(op, op.repeat_count) for op in bundle.operators)
    step_ns = max(bundle_step_int_ns(bundle), exposed_sum_ns)
    idle_ns = max(0, step_ns - exposed_sum_ns)
    yield {
        "ph": "M",
        "name": "mindie_sim_step_receipt",
        "pid": PID_FRAMEWORK,
        "tid": TID_FW_ENGINE,
        "args": {
            "step_span_ns": step_ns,
            "disjoint_exposed_sum_ns": exposed_sum_ns,
            "idle_or_wait_ns": idle_ns,
            "reconciliation_error_ns": step_ns - (exposed_sum_ns + idle_ns),
        },
    }

    pipeline_by_occ = {
        pipe.owner: (len(pipe.phases), pipe.member_op_ids)
        for pipe in bundle.pipelines
    }
    pipeline_desc_by_occ = {pipe.owner: pipe for pipe in bundle.pipelines}
    if bundle.full_trace_schedule:
        yield from _iter_scheduled_full_model_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
            pipeline_desc_by_occ=pipeline_desc_by_occ,
        )
    else:
        yield from _iter_direct_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
            pipeline_desc_by_occ=pipeline_desc_by_occ,
        )


def _iter_msprof_trace_json_lines(
    bundle: ExportBundle,
    *,
    encode: Any,
    include_pipe_lanes: bool,
    include_counters: bool,
    include_overlap: bool,
) -> Iterator[tuple[Any, str]]:
    """Yield ``(ph, encoded_json_line)`` lazily with exact parity to ``iter_msprof_trace_events``."""
    coords = _die_coordinates(bundle)
    die_map = {coord: idx for idx, coord in enumerate(coords)}
    for ev in _iter_metadata_events(coords):
        yield (ev.get("ph"), encode(ev))
    exposed_sum_ns = sum(op_exposed_int_ns(op, op.repeat_count) for op in bundle.operators)
    step_ns = max(bundle_step_int_ns(bundle), exposed_sum_ns)
    idle_ns = max(0, step_ns - exposed_sum_ns)
    receipt_ev = {
        "ph": "M",
        "name": "mindie_sim_step_receipt",
        "pid": PID_FRAMEWORK,
        "tid": TID_FW_ENGINE,
        "args": {
            "step_span_ns": step_ns,
            "disjoint_exposed_sum_ns": exposed_sum_ns,
            "idle_or_wait_ns": idle_ns,
            "reconciliation_error_ns": step_ns - (exposed_sum_ns + idle_ns),
        },
    }
    yield ("M", encode(receipt_ev))

    pipeline_by_occ = {
        pipe.owner: (len(pipe.phases), pipe.member_op_ids)
        for pipe in bundle.pipelines
    }
    pipeline_desc_by_occ = {pipe.owner: pipe for pipe in bundle.pipelines}
    if bundle.full_trace_schedule:
        yield from _iter_scheduled_full_model_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
            pipeline_desc_by_occ=pipeline_desc_by_occ,
            _json_encode=encode,
        )
    else:
        for ev in _iter_direct_trace(
            bundle, die_map, pipeline_by_occ,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
            pipeline_desc_by_occ=pipeline_desc_by_occ,
        ):
            yield (ev.get("ph"), encode(ev))


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
    encode = json.JSONEncoder(
        ensure_ascii=False, separators=(",", ":"), allow_nan=False,
    ).encode
    chunk: list[str] = []
    with os.fdopen(
        fd, "w", encoding="utf-8", newline="\n", buffering=STREAM_BUFFER_BYTES,
    ) as handle:
        handle.write("[\n")
        first_chunk = True
        for ph, line in _iter_msprof_trace_json_lines(
            bundle,
            encode=encode,
            include_pipe_lanes=include_pipe_lanes,
            include_counters=include_counters,
            include_overlap=include_overlap,
        ):
            chunk.append(line)
            total += 1
            if ph in ph_counts:
                ph_counts[ph] += 1
            if len(chunk) >= _STREAM_BATCH_EVENTS:
                if not first_chunk:
                    handle.write(",\n")
                handle.write(",\n".join(chunk))
                chunk.clear()
                first_chunk = False
        if chunk:
            if not first_chunk:
                handle.write(",\n")
            handle.write(",\n".join(chunk))
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
