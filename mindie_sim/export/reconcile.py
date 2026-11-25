"""Streaming ``msprof`` ``kernel_details.csv`` ingress and 1-to-N reconciliation.

[Role] Stream Ascend ``msprof`` ``kernel_details.csv`` or JSON observation
  bundles into typed ``KernelObservationRow`` records and perform
  sample-conserving 1-to-N matching against lowered ``OperatorExportRow`` items.
[Invariants] Every observed kernel row is matched at most once; unmatched
  kernels are accounted for explicitly rather than dropped; bounded JSON reads
  (``<= 50 MiB``) and streaming CSV iteration keep memory bounded.
[Timeouts & Locks] Local read-only file streaming; no locks or device contexts.
[Side-Effect Policy] Read-only over caller observation paths.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any

from mindie_sim.schema.common import ModelingError

from .models import (
    MAX_OBSERVATION_JSON_BYTES,
    MIN_MATCH_TOKEN_LEN,
    P90_QUANTILE,
    RECONCILE_ABS_FLOOR_US,
    RECONCILE_APE_THRESHOLD_PCT,
    KernelObservationRow,
    OperatorExportRow,
    ReconciledKernelGroup,
    ReconciliationSummary,
)

_CSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "kernel_name", "op name", "op_name"),
    "op_type": ("type", "op type", "op_type", "kernel_type"),
    "core": ("accelerator core", "core", "kernel_type", "task type"),
    "start_us": ("start time(us)", "task start time(us)", "task_start(us)", "start_us"),
    "duration_us": ("duration(us)", "task duration(us)", "task_time(us)", "duration_us"),
    "stream_id": ("stream id", "stream_id"),
    "task_id": ("task id", "task_id"),
    "input_shapes": ("input shapes", "input_shapes"),
    "input_dtypes": ("input data types", "input_dtypes"),
}
_FUSED_KERNEL_HINTS: dict[str, tuple[str, ...]] = {
    "qkv_projection": ("qkv", "q_proj", "k_proj", "v_proj", "matmul", "mm"),
    "add_rmsnorm_quant": ("add", "rmsnorm", "rms_norm", "quant", "dynamicquant"),
    "add_rmsnorm": ("add", "rmsnorm", "rms_norm"),
    "mla_prolog": ("mla", "prolog", "rope", "rotary", "quant", "matmul"),
    "gate_up_projection": ("gate", "up", "matmul", "mm"),
    "grouped_gemm": ("grouped", "gmm", "moe", "matmul"),
    "attention": ("flashattention", "mla", "attention", "fia", "ifa", "pfa", "sfa"),
    "allreduce": ("allreduce", "hccl", "hcom"),
    "allgather": ("allgather", "hccl", "hcom"),
    "reduce_scatter": ("reducescatter", "hccl", "hcom"),
    "reducescatter": ("reducescatter", "hccl", "hcom"),
    "alltoall": ("alltoall", "dispatch", "combine", "hccl", "hcom"),
    "rmsnorm": ("rmsnorm", "rms_norm", "norm"),
    "rotary": ("rotary", "rope"),
    "swiglu": ("swiglu", "silu", "mul"),
    "quantize": ("quant", "dynamicquant"),
}


def _clean_text(raw: Any) -> str:
    text = str(raw or "").strip().strip('"').strip()
    return "" if text.upper() in {"N/A", "NA", "NONE", "NULL"} else text


def _parse_float(raw: str, label: str) -> float:
    cleaned = _clean_text(raw)
    if not cleaned:
        raise ModelingError(f"kernel_csv_missing_{label}")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ModelingError(f"kernel_csv_invalid_{label}:{raw!r}") from exc
    if not math.isfinite(value):
        raise ModelingError(f"kernel_csv_nonfinite_{label}:{raw!r}")
    return value


def load_kernel_details_csv(path: str | Path) -> tuple[KernelObservationRow, ...]:
    """Stream an Ascend ``msprof`` ``kernel_details.csv`` into typed observation rows."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"kernel_details_csv_not_found:{target}")
    rows: list[KernelObservationRow] = []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ModelingError(f"kernel_csv_empty_header:{target.name}")
        cols = {str(key).strip().lower(): key for key in reader.fieldnames}
        selected = {
            field: next((cols[alias] for alias in aliases if alias in cols), None)
            for field, aliases in _CSV_COLUMN_ALIASES.items()
        }
        if not selected["name"] or not selected["duration_us"]:
            raise ModelingError(f"kernel_csv_missing_required_columns:{target.name}")
        for idx, raw in enumerate(reader):
            name = _clean_text(raw.get(selected["name"]))
            if not name:
                continue
            dur_us = _parse_float(raw.get(selected["duration_us"], ""), "duration_us")
            if dur_us <= 0.0:
                continue
            start_raw = raw.get(selected["start_us"], "") if selected["start_us"] else ""
            start_us = _parse_float(start_raw, "start_us") if _clean_text(start_raw) else float(idx)
            rows.append(KernelObservationRow(
                index=len(rows),
                name=name,
                op_type=_clean_text(raw.get(selected["op_type"])) if selected["op_type"] else "",
                core=_clean_text(raw.get(selected["core"])) if selected["core"] else "",
                start_us=start_us,
                duration_us=dur_us,
                stream_id=_clean_text(raw.get(selected["stream_id"])) if selected["stream_id"] else "",
                task_id=_clean_text(raw.get(selected["task_id"])) if selected["task_id"] else "",
                input_shapes=_clean_text(raw.get(selected["input_shapes"])) if selected["input_shapes"] else "",
                input_dtypes=_clean_text(raw.get(selected["input_dtypes"])) if selected["input_dtypes"] else "",
            ))
    rows.sort(key=lambda item: (item.start_us, item.index))
    return tuple(rows)


def load_observation_bundle(
    path: str | Path,
) -> tuple[tuple[KernelObservationRow, ...], dict[str, tuple[str, ...]]]:
    """Load an observation JSON bundle or ``kernel_details.csv`` with optional 1-to-N map."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"observation_bundle_not_found:{target}")
    if target.suffix.lower() == ".csv":
        return load_kernel_details_csv(target), {}
    if target.stat().st_size > MAX_OBSERVATION_JSON_BYTES:
        raise ModelingError("observation_json_exceeds_50_mib_limit")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelingError("observation_bundle_must_be_mapping")
    kernels: list[KernelObservationRow] = []
    explicit_map: dict[str, tuple[str, ...]] = {}
    if payload.get("kernel_csv"):
        csv_path = Path(str(payload["kernel_csv"]))
        if not csv_path.is_absolute():
            csv_path = (target.parent / csv_path).resolve()
        kernels.extend(load_kernel_details_csv(csv_path))
    next_index = max((k.index for k in kernels), default=-1) + 1
    for idx, entry in enumerate(payload.get("kernels", ()) or ()):
        if not isinstance(entry, Mapping):
            continue
        dur = float(entry.get("duration_us", 0.0) or 0.0)
        if dur <= 0.0 or not math.isfinite(dur):
            continue
        kernels.append(KernelObservationRow(
            index=next_index,
            name=str(entry.get("name", f"kernel_{idx}")),
            op_type=str(entry.get("op_type", entry.get("type", ""))),
            core=str(entry.get("core", "")),
            start_us=float(entry.get("start_us", idx) or idx),
            duration_us=dur,
            stream_id=str(entry.get("stream_id", "")),
            task_id=str(entry.get("task_id", "")),
            input_shapes=str(entry.get("input_shapes", "")),
            input_dtypes=str(entry.get("input_dtypes", "")),
        ))
        next_index += 1
    for group in (payload.get("mappings") or payload.get("groups") or ()):
        if not isinstance(group, Mapping):
            continue
        sim_id = str(group.get("sim_op_id") or group.get("op_id") or "").strip()
        if not sim_id:
            continue
        names: list[str] = []
        for k_entry in group.get("kernels", ()) or ():
            if isinstance(k_entry, str):
                names.append(k_entry)
            elif isinstance(k_entry, Mapping):
                kname = str(k_entry.get("name", "")).strip()
                kdur = float(k_entry.get("duration_us", 0.0) or 0.0)
                if kname and kdur > 0.0:
                    names.append(kname)
                    kernels.append(KernelObservationRow(
                        index=next_index, name=kname,
                        op_type=str(k_entry.get("op_type", "")),
                        core=str(k_entry.get("core", "")),
                        start_us=float(k_entry.get("start_us", next_index) or next_index),
                        duration_us=kdur,
                        stream_id=str(k_entry.get("stream_id", "")),
                        task_id=str(k_entry.get("task_id", "")),
                        input_shapes=str(k_entry.get("input_shapes", "")),
                        input_dtypes=str(k_entry.get("input_dtypes", "")),
                    ))
                    next_index += 1
        if names:
            explicit_map[sim_id] = tuple(names)
    return tuple(kernels), explicit_map


def _operator_match_tokens(op: OperatorExportRow) -> tuple[str, ...]:
    tokens: list[str] = []
    suffix = op.op_id.rsplit(".", 1)[-1].lower()
    if suffix and len(suffix) >= MIN_MATCH_TOKEN_LEN:
        tokens.append(suffix)
    for source in (op.family, op.recipe, op.ascend_kernel or op.implementation, *op.absorbed_source_ops):
        lower = source.strip().lower()
        if lower and len(lower) >= MIN_MATCH_TOKEN_LEN:
            tokens.append(lower)
            for hint in _FUSED_KERNEL_HINTS.get(lower, ()):
                tokens.append(hint)
    return tuple(dict.fromkeys(tokens))


def _kernel_matches_tokens(kernel: KernelObservationRow, tokens: Sequence[str]) -> bool:
    haystack = f"{kernel.name} {kernel.op_type}".lower()
    return any(tok in haystack for tok in tokens)


def _build_reconciled_group(
    op: OperatorExportRow, matched: tuple[KernelObservationRow, ...],
) -> ReconciledKernelGroup:
    meas_sum = sum(k.duration_us for k in matched)
    if meas_sum <= 0.0:
        return ReconciledKernelGroup(op.op_id, matched, None, None, None)
    abs_err = abs(op.dur_us - meas_sum)
    ape_pct = 100.0 * abs_err / meas_sum
    gate_limit = max((RECONCILE_APE_THRESHOLD_PCT / 100.0) * meas_sum, RECONCILE_ABS_FLOOR_US)
    return ReconciledKernelGroup(
        sim_op_id=op.op_id,
        kernels=matched,
        measured_sum_us=meas_sum,
        ape_pct=ape_pct,
        within_gate=abs_err <= gate_limit,
    )


def _nearest_rank_percentile(sorted_vals: Sequence[float], frac: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(frac * len(sorted_vals)) - 1))
    return float(sorted_vals[idx])


def reconcile_operators_with_kernels(
    operators: Sequence[OperatorExportRow],
    kernels: Sequence[KernelObservationRow],
    *,
    explicit_map: Mapping[str, Sequence[str]] | None = None,
    source_label: str = "kernel_details.csv",
) -> tuple[dict[str, ReconciledKernelGroup], ReconciliationSummary]:
    """Perform sample-conserving 1-to-N reconciliation of fused ``sim`` ops to kernels."""
    consumed: set[int] = set()
    groups: dict[str, ReconciledKernelGroup] = {}
    explicit = dict(explicit_map or {})
    by_name: dict[str, list[KernelObservationRow]] = defaultdict(list)
    for row in kernels:
        by_name[row.name].append(row)

    for op in operators:
        wanted = explicit.get(op.op_id)
        if not wanted:
            continue
        matched: list[KernelObservationRow] = []
        for pattern in wanted:
            chosen = next((k for k in by_name.get(pattern, ()) if k.index not in consumed), None)
            if chosen is None:
                chosen = next((k for k in kernels if k.index not in consumed and pattern.lower() in k.name.lower()), None)
            if chosen is not None:
                consumed.add(chosen.index)
                matched.append(chosen)
        if matched:
            groups[op.op_id] = _build_reconciled_group(op, tuple(matched))

    cursor = 0
    for op in operators:
        if op.op_id in groups:
            continue
        tokens = _operator_match_tokens(op)
        matched = []
        scan_idx = cursor
        max_group = max(1, len(op.absorbed_source_ops) + 1)
        while scan_idx < len(kernels) and len(matched) < max_group:
            cand = kernels[scan_idx]
            scan_idx += 1
            if cand.index in consumed:
                continue
            if _kernel_matches_tokens(cand, tokens):
                consumed.add(cand.index)
                matched.append(cand)
                cursor = scan_idx
            elif matched:
                break
        if matched:
            groups[op.op_id] = _build_reconciled_group(op, tuple(matched))

    unmatched = tuple(k for k in kernels if k.index not in consumed)
    apes = sorted(g.ape_pct for g in groups.values() if g.ape_pct is not None and math.isfinite(g.ape_pct))
    med_ape = float(median(apes)) if apes else None
    p90_ape = _nearest_rank_percentile(apes, P90_QUANTILE) if apes else None
    within_count = sum(1 for g in groups.values() if g.within_gate is True)
    within_pct = (100.0 * within_count / len(groups)) if groups else None
    matched_sim_sum = sum(op.dur_us for op in operators if op.op_id in groups)
    matched_meas_sum = sum(g.measured_sum_us for g in groups.values() if g.measured_sum_us is not None)
    summary = ReconciliationSummary(
        source_label=source_label,
        total_sim_ops=len(operators),
        matched_sim_ops=len(groups),
        total_kernels=len(kernels),
        matched_kernels=len(consumed),
        unmatched_kernels=unmatched,
        matched_sim_sum_us=matched_sim_sum,
        matched_measured_sum_us=matched_meas_sum,
        unmatched_measured_us=sum(k.duration_us for k in unmatched),
        median_ape_pct=med_ape,
        p90_ape_pct=p90_ape,
        within_gate_count=within_count,
        within_gate_pct=within_pct,
    )
    return groups, summary
