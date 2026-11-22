"""Streaming 1:1 Ascend ``msprof`` ``trace_view.json`` exporter for MindStudio Insight.

[Role] Stream Chrome Trace JSON array events matching the native Ascend
  ``msprof`` / ``torch_npu.profiler`` schema (``ph="M"`` metadata, ``ph="X"``
  operator/pipe/overlap spans with Lean-4 ``L*`` Gap Radar metadata, ``ph="s"``
  / ``ph="f"`` ``SetFlag``/``WaitFlag`` flows, and ``ph="C"`` utilization &
  memory counter series) directly to disk in bounded memory.
[Invariants] Root document is a valid JSON array ``[...]`` with Unix LF endings
  and ``allow_nan=False`` (readable natively by MindStudio Insight ``msinsight``,
  Perfetto UI, and ``mindie_profiler.sources_trace.iter_trace_events``). Events
  are streamed incrementally without ever materializing a full event list in RAM
  (peak overhead ``<< 512 MiB``). Every ``ph="s"`` flow event has an exact
  matching ``ph="f"`` (``bp="e"``) event with identical ``id`` and causal
  ``dst_ts >= src_ts``. Zero ``0.80x`` heuristic scaling or host-path leaks.
[Timeouts & Locks] Pure bounded single-pass streaming over operators; atomic
  write via same-directory temp file + ``os.replace``; no locks or device access.
[Side-Effect Policy] Writes only the caller-requested output file path and cleans
  up its private temp file on error.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from mindie_sim.prediction.results import CaseInput, ReplayOptions, StepEvaluation, StepRequest
from .analysis import (
    ExportBundle,
    GIB_BYTES,
    OperatorExportRow,
    build_export_bundle,
)

#: Schema tag for trace export receipts.
TRACE_EXPORT_RECEIPT_SCHEMA = "mindie-sim.msprof-trace-receipt/1"
#: Stream buffer size (64 KiB) for bounded low-RSS JSON emission.
STREAM_BUFFER_BYTES = 64 * 1024
#: Timestamp decimal precision in microseconds (3 decimal places = 1 ns).
TIMESTAMP_US_DECIMALS = 3
#: Metric rounding decimals for compact readable ``msinsight`` args.
METRIC_DECIMALS = 3
#: Ratio rounding decimals for Gap Radar ratios.
RATIO_DECIMALS = 4
#: Maximum counter sample points emitted per series to keep large traces bounded.
MAX_COUNTER_SAMPLES = 512
#: Per-die PID block stride guaranteeing disjoint Ascend Hardware / HCCL / Metrics PIDs.
DIE_PID_STRIDE = 100
PID_OFFSET_HARDWARE = 1
PID_OFFSET_HCCL = 2
PID_OFFSET_OVERLAP = 3
PID_OFFSET_METRICS = 4

#: Stream thread IDs inside the ``Ascend Hardware`` process.
TID_STREAM_COMPUTE = 0
TID_STREAM_MSO_AUX = 1
TID_STREAM_COMM_CSA = 2
#: Pipe sub-lane thread IDs inside the ``Ascend Hardware`` process.
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

#: Phase split shares used only when a single-op event has multiple active sub-pipes
#: without explicit ``FusedPipelineDescriptor.phases`` timestamps.
PIPE_LEAD_FRACTION = 0.25
PIPE_CORE_START_FRACTION = 0.15
PIPE_CORE_END_FRACTION = 0.85
PIPE_TAIL_START_FRACTION = 0.75
PIPE_MIN_SUBSPAN_US = 0.001
FLOW_CAUSAL_EPSILON_US = 1e-6

#: Canonical ``msprof`` process and thread specification exported for introspection.
MSPROF_PROCESS_SPEC: dict[str, tuple[tuple[int, str], ...]] = {
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
        }


def _round_us(value: float) -> float:
    return round(max(0.0, float(value)), TIMESTAMP_US_DECIMALS)


def _round_metric(value: float) -> float:
    return round(float(value), METRIC_DECIMALS)


def _format_shape(shape: Sequence[int]) -> str:
    return "x".join(str(int(dim)) for dim in shape) if shape else "scalar"


def _die_coordinates(bundle: ExportBundle) -> tuple[tuple[int, int, int], ...]:
    coords = sorted({
        (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        for op in bundle.operators
    })
    return tuple(coords) if coords else ((0, 0, 0),)


def _pid_base(die_index: int) -> int:
    return int(die_index) * DIE_PID_STRIDE


def _hccl_tid(op: OperatorExportRow) -> int:
    lower = f"{op.op_id} {op.family} {op.recipe}".lower()
    if any(tok in lower for tok in ("alltoall", "dispatch", "combine", "expert", "moe", "mc2")):
        return TID_HCCL_EP_MOE
    if any(tok in lower for tok in ("allreduce", "allgather", "reducescatter", "tp", "sp", "cp")):
        return TID_HCCL_TP_SP_CP
    return TID_HCCL_COLLECTIVE


def _op_pid_tid(
    op: OperatorExportRow, base_pid: int,
) -> tuple[int, int]:
    if op.submodule == "Collective" or op.task_type == "HCCL":
        return base_pid + PID_OFFSET_HCCL, _hccl_tid(op)
    if op.stream_index == 1:
        return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_MSO_AUX
    if op.stream_index == 2:
        return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_COMM_CSA
    return base_pid + PID_OFFSET_HARDWARE, TID_STREAM_COMPUTE


def _iter_metadata_events(
    coords: Sequence[tuple[int, int, int]],
) -> Iterator[dict[str, Any]]:
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
            yield {
                "ph": "M",
                "name": "process_name",
                "pid": pid,
                "tid": 0,
                "args": {"name": f"{proc_key}{suffix}"},
            }
            yield {
                "ph": "M",
                "name": "process_labels",
                "pid": pid,
                "tid": 0,
                "args": {"labels": label},
            }
            yield {
                "ph": "M",
                "name": "process_sort_index",
                "pid": pid,
                "tid": 0,
                "args": {"sort_index": pid},
            }
            for tid, thread_label in MSPROF_PROCESS_SPEC[proc_key]:
                yield {
                    "ph": "M",
                    "name": "thread_name",
                    "pid": pid,
                    "tid": tid,
                    "args": {"name": thread_label},
                }
                yield {
                    "ph": "M",
                    "name": "thread_sort_index",
                    "pid": pid,
                    "tid": tid,
                    "args": {"sort_index": tid},
                }


def _operator_args(
    bundle: ExportBundle,
    op: OperatorExportRow,
    task_id: int,
    pipeline_info: tuple[int, tuple[str, ...]] | None = None,
) -> dict[str, Any]:
    args: dict[str, Any] = {
        "Model Name": bundle.model_name,
        "Hardware SKU": bundle.hardware_sku,
        "Phase": bundle.phase,
        "Task Type": op.task_type,
        "Stream Id": op.stream_index,
        "Task Id": task_id,
        "Layer": op.layer_label,
        "Submodule": op.submodule,
        "Op Family": op.family,
        "Implementation": op.implementation,
        "Input Shapes": _format_shape(op.shape),
        "Input Data Types": op.dtype,
        "FLOPs": round(op.flops, 1),
        "Vector Ops": round(op.vector_ops, 1),
        "HBM Read (B)": round(op.hbm_read_bytes, 1),
        "HBM Write (B)": round(op.hbm_write_bytes, 1),
        "HBM Total (B)": round(op.hbm_total_bytes, 1),
        "Link Bytes (B)": round(op.link_bytes, 1),
        "L2 Hit (%)": _round_metric(op.l2_hit_pct),
        "Bottleneck": op.dominant_bound,
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


def _active_pipe_spans(
    op: OperatorExportRow,
) -> list[tuple[int, str, float, float, dict[str, Any]]]:
    """Derive bounded sub-pipe spans inside ``[op.start_us, op.end_us]``."""
    if op.dur_us <= 0.0:
        return []
    ts0 = op.start_us
    dur = op.dur_us
    spans: list[tuple[int, str, float, float, dict[str, Any]]] = []
    if op.mte2_bytes > 0.0:
        load_dur = max(PIPE_MIN_SUBSPAN_US, dur * (1.0 - PIPE_LEAD_FRACTION))
        spans.append((
            TID_PIPE_MTE2, f"{op.op_id}:MTE2", ts0, min(dur, load_dur),
            {"Pipe": "MTE2", "Bytes": round(op.mte2_bytes, 1), "Op Id": op.op_id},
        ))
    if op.flops > 0.0:
        c_start = ts0 + (dur * PIPE_CORE_START_FRACTION if op.mte2_bytes > 0.0 else 0.0)
        c_end = ts0 + dur * (PIPE_CORE_END_FRACTION if (op.fixpipe_bytes > 0.0 or op.mte3_bytes > 0.0) else 1.0)
        spans.append((
            TID_PIPE_CUBE, f"{op.op_id}:Cube", c_start, max(PIPE_MIN_SUBSPAN_US, c_end - c_start),
            {"Pipe": "Cube(AIC)", "FLOPs": round(op.flops, 1), "Dtype": op.dtype, "Op Id": op.op_id},
        ))
    if op.vector_ops > 0.0:
        v_start = ts0 + (dur * PIPE_CORE_START_FRACTION if op.mte2_bytes > 0.0 else 0.0)
        v_end = ts0 + dur * (PIPE_CORE_END_FRACTION if op.mte3_bytes > 0.0 else 1.0)
        spans.append((
            TID_PIPE_VECTOR, f"{op.op_id}:Vector", v_start, max(PIPE_MIN_SUBSPAN_US, v_end - v_start),
            {"Pipe": "Vector(AIV)", "Vector Ops": round(op.vector_ops, 1), "Op Id": op.op_id},
        ))
    if op.fixpipe_bytes > 0.0 or (op.flops > 0.0 and op.hbm_write_bytes > 0.0 and op.mte3_bytes <= 0.0):
        fp_bytes = op.fixpipe_bytes if op.fixpipe_bytes > 0.0 else op.hbm_write_bytes
        f_start = ts0 + dur * PIPE_TAIL_START_FRACTION
        spans.append((
            TID_PIPE_FIXPIPE, f"{op.op_id}:Fixpipe", f_start, max(PIPE_MIN_SUBSPAN_US, (ts0 + dur) - f_start),
            {"Pipe": "Fixpipe", "Bytes": round(fp_bytes, 1), "Op Id": op.op_id},
        ))
    if op.mte3_bytes > 0.0:
        s_start = ts0 + dur * PIPE_TAIL_START_FRACTION
        spans.append((
            TID_PIPE_MTE3, f"{op.op_id}:MTE3", s_start, max(PIPE_MIN_SUBSPAN_US, (ts0 + dur) - s_start),
            {"Pipe": "MTE3", "Bytes": round(op.mte3_bytes, 1), "Op Id": op.op_id},
        ))
    return spans


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


def _iter_overlap_events(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
) -> Iterator[dict[str, Any]]:
    comp_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    comm_by_die: dict[int, list[tuple[float, float]]] = defaultdict(list)
    for op in bundle.operators:
        if op.dur_us <= 0.0:
            continue
        coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        d_idx = die_map.get(coord, 0)
        if op.submodule == "Collective" or op.task_type == "HCCL":
            comm_by_die[d_idx].append((op.start_us, op.end_us))
        else:
            comp_by_die[d_idx].append((op.start_us, op.end_us))

    for d_idx in sorted(set(die_map.values())):
        pid = _pid_base(d_idx) + PID_OFFSET_OVERLAP
        comp = _merge_intervals(comp_by_die.get(d_idx, ()))
        comm = _merge_intervals(comm_by_die.get(d_idx, ()))
        comm_only = _subtract_intervals(comm, comp)
        busy = _merge_intervals([*comp, *comm])
        step_end = max(bundle.step_span_us, max((hi for _, hi in busy), default=0.0))
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
                    "name": label,
                    "cat": "Overlap",
                    "ph": "X",
                    "ts": _round_us(lo),
                    "dur": _round_us(hi - lo),
                    "pid": pid,
                    "tid": tid,
                    "args": {"Overlap Lane": label, "Duration (us)": _round_us(hi - lo)},
                }


def _iter_counter_events(
    bundle: ExportBundle,
    die_map: Mapping[tuple[int, int, int], int],
) -> Iterator[dict[str, Any]]:
    peaks = bundle.hardware_peaks
    hbm_peak_gbps = float(peaks.get("hbm_peak_bytes_per_s", 0.0) or 0.0) / 1_000_000_000.0
    cube_peak = float(peaks.get("cube_peak_flops_per_s", 0.0) or 0.0)
    vec_peak = float(peaks.get("vector_peak_ops_per_s", 0.0) or 0.0)
    mem_ledger = bundle.memory_ledger
    resident_gib = (
        (mem_ledger.max_stage_hbm_bytes - sum(
            b.workspace_peak_bytes for b in mem_ledger.stage_budgets[:1]
        )) / float(GIB_BYTES)
        if mem_ledger.stage_budgets else 0.0
    )
    resident_gib = max(0.0, resident_gib)

    ops_by_die: dict[int, list[OperatorExportRow]] = defaultdict(list)
    for op in bundle.operators:
        if op.dur_us <= 0.0:
            continue
        coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        ops_by_die[die_map.get(coord, 0)].append(op)

    for d_idx in sorted(set(die_map.values())):
        pid = _pid_base(d_idx) + PID_OFFSET_METRICS
        die_ops = ops_by_die.get(d_idx, ())
        if not die_ops:
            continue
        deltas: dict[float, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0])
        for op in die_ops:
            dur_s = max(op.dur_us / 1_000_000.0, 1e-12)
            bw_gbps = (op.hbm_total_bytes / dur_s) / 1_000_000_000.0
            cube_pct = (100.0 * (op.flops / dur_s) / cube_peak) if cube_peak > 0.0 else 0.0
            vec_pct = (100.0 * (op.vector_ops / dur_s) / vec_peak) if vec_peak > 0.0 else 0.0
            ws_gib = (op.hbm_write_bytes + op.mte3_bytes) / float(GIB_BYTES)
            t0, t1 = _round_us(op.start_us), _round_us(op.end_us)
            for idx, val in enumerate((bw_gbps, cube_pct, vec_pct, ws_gib)):
                deltas[t0][idx] += val
                deltas[t1][idx] -= val

        timestamps = sorted(deltas)
        stride = max(1, (len(timestamps) + MAX_COUNTER_SAMPLES - 1) // MAX_COUNTER_SAMPLES)
        bw_acc = cube_acc = vec_acc = ws_acc = 0.0
        for step_idx, ts_us in enumerate(timestamps):
            d_bw, d_cube, d_vec, d_ws = deltas[ts_us]
            bw_acc += d_bw
            cube_acc += d_cube
            vec_acc += d_vec
            ws_acc += d_ws
            if step_idx % stride != 0 and step_idx != len(timestamps) - 1:
                continue
            bw_val = max(0.0, min(bw_acc, hbm_peak_gbps if hbm_peak_gbps > 0.0 else bw_acc))
            cube_val = max(0.0, min(100.0, cube_acc))
            vec_val = max(0.0, min(100.0, vec_acc))
            active_gib = max(resident_gib, resident_gib + max(0.0, ws_acc))
            yield {
                "name": "HBM Bandwidth (GB/s)",
                "ph": "C",
                "ts": ts_us,
                "pid": pid,
                "tid": TID_METRICS_COUNTER,
                "args": {"HBM_BW_GBps": _round_metric(bw_val)},
            }
            yield {
                "name": "Cube Utilization (%)",
                "ph": "C",
                "ts": ts_us,
                "pid": pid,
                "tid": TID_METRICS_COUNTER,
                "args": {"Cube_Util_Pct": _round_metric(cube_val)},
            }
            yield {
                "name": "Vector Utilization (%)",
                "ph": "C",
                "ts": ts_us,
                "pid": pid,
                "tid": TID_METRICS_COUNTER,
                "args": {"Vector_Util_Pct": _round_metric(vec_val)},
            }
            yield {
                "name": "Active HBM Memory (GiB)",
                "ph": "C",
                "ts": ts_us,
                "pid": pid,
                "tid": TID_METRICS_COUNTER,
                "args": {
                    "Resident_GiB": _round_metric(resident_gib),
                    "Active_Total_GiB": _round_metric(active_gib),
                },
            }


def iter_msprof_trace_events(
    bundle: ExportBundle,
    *,
    include_pipe_lanes: bool = True,
    include_counters: bool = True,
    include_overlap: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield 1:1 Ascend ``msprof`` Chrome Trace events lazily with O(active) memory."""
    coords = _die_coordinates(bundle)
    die_map = {coord: idx for idx, coord in enumerate(coords)}
    yield from _iter_metadata_events(coords)

    pipeline_by_occ = {
        pipe.owner: (len(pipe.phases), pipe.member_op_ids)
        for pipe in bundle.pipelines
    }
    op_placement: dict[Any, tuple[int, int, float, str]] = {}
    flow_id = 1
    for task_id, op in enumerate(bundle.operators, start=1):
        coord = (op.occurrence.replica, op.occurrence.stage, int(op.occurrence.rank or 0))
        base = _pid_base(die_map.get(coord, 0))
        pid, tid = _op_pid_tid(op, base)
        cat = "Communication" if (op.submodule == "Collective" or op.task_type == "HCCL") else "Kernel"
        yield {
            "name": f"{op.op_id} [{op.family}]",
            "cat": cat,
            "ph": "X",
            "ts": _round_us(op.start_us),
            "dur": _round_us(op.dur_us),
            "pid": pid,
            "tid": tid,
            "args": _operator_args(bundle, op, task_id, pipeline_by_occ.get(op.occurrence)),
        }
        op_placement[op.occurrence] = (pid, tid, op.end_us, op.op_id)

        for pred in op.predecessors:
            src = op_placement.get(pred)
            if src is None:
                continue
            src_pid, src_tid, src_end_us, src_op_id = src
            if src_pid == pid and src_tid == tid:
                continue
            dst_ts = max(src_end_us, op.start_us)
            sync_kind = "EventRecord/Wait" if src_pid != pid else "StreamSync"
            flow_args = {"src_op": src_op_id, "dst_op": op.op_id, "sync_kind": sync_kind}
            yield {
                "name": f"SetFlag->{sync_kind}",
                "cat": "SetFlag_WaitFlag",
                "ph": "s",
                "id": flow_id,
                "ts": _round_us(src_end_us),
                "pid": src_pid,
                "tid": src_tid,
                "args": flow_args,
            }
            yield {
                "name": f"WaitFlag<-{sync_kind}",
                "cat": "SetFlag_WaitFlag",
                "ph": "f",
                "bp": "e",
                "id": flow_id,
                "ts": _round_us(dst_ts),
                "pid": pid,
                "tid": tid,
                "args": flow_args,
            }
            flow_id += 1

        if include_pipe_lanes:
            hw_pid = base + PID_OFFSET_HARDWARE
            pipe_spans = _active_pipe_spans(op)
            for p_tid, p_name, p_ts, p_dur, p_args in pipe_spans:
                yield {
                    "name": p_name,
                    "cat": "Pipe",
                    "ph": "X",
                    "ts": _round_us(p_ts),
                    "dur": _round_us(p_dur),
                    "pid": hw_pid,
                    "tid": p_tid,
                    "args": p_args,
                }
            for prev_span, next_span in zip(pipe_spans, pipe_spans[1:]):
                src_t = _round_us(prev_span[2])
                dst_t = max(src_t, _round_us(next_span[2]))
                p_args = {
                    "src_pipe": prev_span[4]["Pipe"],
                    "dst_pipe": next_span[4]["Pipe"],
                    "op_id": op.op_id,
                    "sync_kind": "SetFlag/WaitFlag",
                }
                yield {
                    "name": "SetFlag->WaitFlag",
                    "cat": "SetFlag_WaitFlag",
                    "ph": "s",
                    "id": flow_id,
                    "ts": src_t,
                    "pid": hw_pid,
                    "tid": prev_span[0],
                    "args": p_args,
                }
                yield {
                    "name": "SetFlag->WaitFlag",
                    "cat": "SetFlag_WaitFlag",
                    "ph": "f",
                    "bp": "e",
                    "id": flow_id,
                    "ts": dst_t,
                    "pid": hw_pid,
                    "tid": next_span[0],
                    "args": p_args,
                }
                flow_id += 1

    if include_overlap:
        yield from _iter_overlap_events(bundle, die_map)
    if include_counters:
        yield from _iter_counter_events(bundle, die_map)


def export_msprof_trace(
    source: ExportBundle | CaseInput,
    output_path: str | Path,
    *,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    include_pipe_lanes: bool = True,
    include_counters: bool = True,
    include_overlap: bool = True,
) -> TraceExportSummary:
    """Stream a 1:1 ``msprof`` ``trace_view.json`` to ``output_path`` atomically."""
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
    total = meta_cnt = complete_cnt = flow_s = counter_cnt = 0
    try:
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
                ph = event.get("ph")
                if ph == "M":
                    meta_cnt += 1
                elif ph == "X":
                    complete_cnt += 1
                elif ph == "s":
                    flow_s += 1
                elif ph == "C":
                    counter_cnt += 1
            handle.write("\n]\n")
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
        metadata_events=meta_cnt,
        complete_events=complete_cnt,
        flow_pairs=flow_s,
        counter_events=counter_cnt,
        step_span_us=bundle.step_span_us,
        step_l_star_lean_us=bundle.step_l_star_lean_us,
        bytes_written=bytes_written,
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
    "TRACE_EXPORT_RECEIPT_SCHEMA",
    "TraceExportSummary",
    "export_msprof_trace",
    "iter_msprof_trace_events",
]
