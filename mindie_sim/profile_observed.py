"""Measured-profile extraction into canonical calibration observations.

[Role]
  Read C7 summaries, Wave 5 benchmark records, and operator/kernel CSV exports
  into a compact observed-profile record without inventing missing values.

[Invariants]
  * Unmeasured fields remain absent; zero is never substituted for ``None``.
  * CSV inputs are streamed in O(active steps) memory and capped at 2M rows.
  * Observed shares are normalized only over recognized canonical bound names.

[Timeouts & Locks]
  Bounded local reads; no network, subprocess, or locks.

[Side-Effect Policy]
  Read-only; returns fresh dictionaries.
"""
from __future__ import annotations

import csv
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Any

from statistics import median
from .advise_util import (
    CANONICAL_BOTTLENECKS,
    _CSV_MAX_ROWS,
    _candidate_mappings,
    _canonical_bottleneck,
    _classify_name,
    _first_int,
    _first_number,
    _first_text,
    _normalize_shares,
    _positive,
)
from .schema.common import ModelingError

_OBSERVED_KEYS = ("step_from_metrics_ms", "observed_step_ms", "step_duration_ms", "step_ms",
                  "median_step_ms", "steady_step_ms", "p50_step_ms", "step_p50_ms",
                  "step_time_ms")


def _shares_from_operator_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals = {name: 0.0 for name in ("comm_bound", "launch_host_bound", "memory_kv_bound", "compute_bound")}
    for row in rows:
        name = str(row.get("OP Type") or row.get("Name") or row.get("name")
                   or row.get("operator") or "")
        value = _first_number(row, ("Total Time(us)", "duration_us", "total_us", "duration_ms",
                                    "total_ms", "Ratio(%)", "ratio"))
        if value is not None:
            totals[_classify_name(name)] += value
    total = sum(totals.values())
    if total <= 0.0:
        return {}
    return {name: round(value / total, 9) for name, value in totals.items() if value > 0.0}


def _stream_csv_rows(path: Path, columns: set[str] | None = None):
    with path.open(encoding="utf-8", errors="replace", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index >= _CSV_MAX_ROWS:
                raise ModelingError(f"csv_row_limit_exceeded:{path}")
            normalized = {str(key).lstrip("\ufeff"): value for key, value in row.items() if key is not None}
            yield ({key: value for key, value in normalized.items()
                    if columns is None or key in columns})


def _observed_from_csv_files(paths: Sequence[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        if path.name == "op_statistic.csv":
            cols = {"Device_id", "device_id", "OP Type", "Name", "name", "operator",
                    "Total Time(us)", "duration_us", "total_us",
                    "duration_ms", "total_ms", "Ratio(%)", "ratio"}
            totals_by_dev: dict[str, dict[str, float]] = {}
            us_by_dev: dict[str, float] = {}
            for row in _stream_csv_rows(path, cols):
                dev = str(row.get("Device_id") or row.get("device_id") or "0").strip() or "0"
                bucket = totals_by_dev.setdefault(dev, {b: 0.0 for b in CANONICAL_BOTTLENECKS})
                name = str(row.get("OP Type") or row.get("Name") or row.get("name")
                           or row.get("operator") or "")
                val = _first_number(row, ("Total Time(us)", "duration_us", "total_us",
                                          "duration_ms", "total_ms", "Ratio(%)", "ratio"))
                if val is not None and val > 0.0:
                    bucket[_classify_name(name)] += val
                us = _positive(_first_number(row, ("Total Time(us)",)))
                if us is not None:
                    us_by_dev[dev] = us_by_dev.get(dev, 0.0) + us
            if totals_by_dev:
                dev0 = sorted(totals_by_dev)[0]
                tot = sum(totals_by_dev[dev0].values())
                if tot > 0.0:
                    shares = {k: round(v / tot, 9) for k, v in totals_by_dev[dev0].items() if v > 0.0}
                    result["observed_share_by_bound"] = shares
                    result["observed_dominant_bottleneck"] = max(shares, key=shares.__getitem__)
                if us_by_dev.get(dev0, 0.0) > 0.0:
                    result["observed_operator_sum_ms"] = us_by_dev[dev0] / 1000.0
        elif path.name == "kernel_details.csv":
            cols = {"Device_id", "device_id", "Step ID", "Step", "step_id",
                    "Start Time(us)", "Start Time", "start_time_us",
                    "Duration(us)", "Duration", "duration", "duration_us"}
            stats_by_dev: dict[str, dict[str, list[float]]] = {}
            for row in _stream_csv_rows(path, cols):
                dev = str(row.get("Device_id") or row.get("device_id") or "0").strip() or "0"
                step_id = str(row.get("Step ID") or row.get("Step") or row.get("step_id") or "").strip()
                dur = _positive(_first_number(row, ("Duration(us)", "Duration", "duration", "duration_us")))
                start = _positive(_first_number(row, ("Start Time(us)", "Start Time", "start_time_us")))
                dev_map = stats_by_dev.setdefault(dev, {})
                acc = dev_map.setdefault(step_id, [0.0, float("inf"), float("-inf"), 0.0])
                if dur is not None:
                    acc[0] += dur
                    acc[3] += 1.0
                if start is not None:
                    acc[1] = min(acc[1], start)
                    acc[2] = max(acc[2], start + (dur or 0.0))
            if stats_by_dev:
                dev0 = sorted(stats_by_dev)[0]
                by_step = stats_by_dev[dev0]
                explicit_steps = {k: v for k, v in by_step.items() if k != ""}
                groups = list(explicit_steps.values()) if explicit_steps else list(by_step.values())
                step_sums = [g[0] / 1000.0 for g in groups if g[3] > 0.0 and g[0] > 0.0]
                step_spans = [(g[2] - g[1]) / 1000.0 for g in groups if g[2] > g[1] and g[1] < float("inf")]
                if step_sums:
                    result.setdefault("observed_operator_sum_ms", float(median(step_sums)))
                if step_spans:
                    if explicit_steps:
                        result.setdefault("observed_step_ms", float(median(step_spans)))
                    else:
                        result.setdefault("observed_event_span_ms", float(step_spans[0]))
    return result


def _shares_from_blocks(data: Any) -> dict[str, float]:
    busy_ms = _positive(_first_number(data, ("device_busy_union_ms_p50", "device_busy_union_ms", "busy_union_ms")))
    for node in _candidate_mappings(data):
        for key in ("share_by_bound", "bottleneck_shares", "bound_shares",
                    "canonical_bottleneck_shares"):
            raw = node.get(key)
            if isinstance(raw, Mapping):
                shares = _normalize_shares(raw)
                if shares:
                    return shares
        blocks = node.get("semantic_block_breakdown")
        entries: list[tuple[str, Mapping[str, Any]]] = []
        if isinstance(blocks, Mapping):
            entries = [(str(k), v) for k, v in blocks.items() if isinstance(v, Mapping)]
        elif isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
            entries = [(str(r.get("block") or r.get("name") or ""), r)
                       for r in blocks if isinstance(r, Mapping) and (r.get("block") or r.get("name"))]
        if entries:
            raw_frac: dict[str, float] = {}
            raw_ms: dict[str, float] = {}
            for name, row in entries:
                sh = _positive(_first_number(row, ("share",)))
                pct = _positive(_first_number(row, ("share_of_busy_pct", "pct_of_device_time")))
                ms = _positive(_first_number(row, ("block_union_ms", "wall_ms_p50", "total_ms",
                                                   "duration_ms", "union_ms")))
                if sh is not None and sh > 0.0:
                    raw_frac[name] = raw_frac.get(name, 0.0) + sh
                elif pct is not None and pct > 0.0:
                    raw_frac[name] = raw_frac.get(name, 0.0) + (pct / 100.0)
                elif ms is not None and ms > 0.0:
                    raw_ms[name] = raw_ms.get(name, 0.0) + ms
            if raw_ms:
                denom = max(busy_ms or 0.0, sum(raw_ms.values()), 1e-12)
                for name, ms_val in raw_ms.items():
                    raw_frac[name] = raw_frac.get(name, 0.0) + (ms_val / denom)
            shares = _normalize_shares(raw_frac)
            if shares:
                return shares
    return {}


def _explicit_dominant(data: Any) -> str | None:
    for key in ("observed_dominant_bottleneck", "dominant_bottleneck", "dominant_bound",
                "primary_bottleneck"):
        for value in (node.get(key) for node in _candidate_mappings(data)):
            result = _canonical_bottleneck(value)
            if result:
                return result
    return None


def _extract_observed(data: Any, *, path: Path | None = None) -> dict[str, Any]:
    """Extract measured observations; never feed these values into prediction."""
    if not isinstance(data, Mapping):
        return {}
    if {"machine", "model", "workload"} <= set(data):
        nested = dict(data.get("runtime", {}).get("metadata") or {})
        existing = nested.get("observed_profile")
        if isinstance(existing, Mapping) and existing:
            return dict(existing)
        view = {key: value for key, value in data.items()
                if key not in {"machine", "model", "workload", "runtime", "slo"}
                and not str(key).startswith("_")}
        return _extract_observed(view, path=path) if view else {}
    observed: dict[str, Any] = {"schema": "mindie-sim.observed-profile/1"}
    step = _first_number(data, _OBSERVED_KEYS)
    if step is None:
        for node in _candidate_mappings(data):
            sa = node.get("step_accounting")
            if isinstance(sa, Mapping) and str(sa.get("step_status") or "").lower() != "not_provided":
                step = _first_number(sa, ("wall_ms_p50", "step_time_ms", "step_ms"))
                if step is not None:
                    break
    tpot = _first_number(data, ("tpot_true_p50_ms", "tpot_p50_ms", "observed_tpot_p50_ms",
                                "tpot_ms", "median_itl_ms"))
    ttft = _first_number(data, ("ttft_p50_ms", "observed_ttft_p50_ms", "ttft_ms"))
    itl = _first_number(data, ("itl_mean_p50_ms", "observed_itl_p50_ms", "itl_ms", "itl_mean_ms"))
    e2e = _first_number(data, ("e2e_p50_ms", "observed_e2e_p50_ms", "e2e_ms"))
    concurrency = _first_int(data, ("concurrency",), minimum=1)
    for out_key, candidates in (
        ("device_busy_union_ms", ("device_busy_union_ms_p50", "device_busy_union_ms", "busy_union_ms")),
        ("device_activity_sum_ms", ("device_activity_sum_ms_p50", "device_activity_sum_ms")),
        ("exposed_comm_ms", ("exposed_comm_ms_p50", "exposed_comm_ms")),
        ("device_uncovered_ms", ("device_uncovered_ms_p50", "device_uncovered_ms")),
        ("host_gap_ms", ("host_gap_ms_p50", "host_gap_ms")),
        ("observed_event_span_ms", ("observed_event_span_ms_p50", "observed_event_span_ms")),
        ("request_wall_ms", ("request_wall_ms_p50", "request_wall_ms")),
        ("aicpu_share", ("aicpu_exposed_ratio_p50", "aicpu_exposed_ratio", "aicpu_share")),
        ("host_share", ("underfeed_bubble_ratio_p50", "underfeed_bubble_ratio", "host_share")),
    ):
        val = _positive(_first_number(data, candidates))
        if val is not None:
            observed[out_key] = round(val, 9)
    shares = _shares_from_blocks(data)
    if not shares:
        raw: dict[str, float] = {}
        compute_core = (_positive(_first_number(data, ("cube_share",))) or 0.0) + (
            _positive(_first_number(data, ("vector_share",))) or 0.0)
        compute_alt = (_positive(_first_number(data, ("attention_share",))) or 0.0) + (
            _positive(_first_number(data, ("moe_share",))) or 0.0)
        comm_val = _positive(_first_number(data, ("comm_share", "hccl_share")))
        if comm_val:
            raw["comm_bound"] = comm_val
        launch_val = (_positive(_first_number(data, ("aicpu_share", "aicpu_exposed_ratio_p50", "aicpu_exposed_ratio"))) or 0.0) + (
            _positive(_first_number(data, ("host_share", "underfeed_bubble_ratio_p50", "underfeed_bubble_ratio"))) or 0.0)
        if launch_val > 0.0:
            raw["launch_host_bound"] = launch_val
        mem_val = _positive(_first_number(data, ("memory_share",)))
        if mem_val:
            raw["memory_kv_bound"] = mem_val
        if "comm_bound" not in raw and observed.get("exposed_comm_ms") and observed.get("device_busy_union_ms"):
            raw["comm_bound"] = float(observed["exposed_comm_ms"]) / float(observed["device_busy_union_ms"])
        compute = compute_core if compute_core > 0.0 else compute_alt
        if compute > 0.0:
            raw["compute_bound"] = compute
        shares = _normalize_shares(raw)
    if path is not None:
        files: list[Path] = []
        if path.is_dir():
            files = [path / "op_statistic.csv"] if (path / "op_statistic.csv").is_file() else []
            if step is None and (path / "kernel_details.csv").is_file():
                files.append(path / "kernel_details.csv")
            if not files:
                rank_dirs = sorted(d for d in path.glob("rank_*") if d.is_dir())
                if rank_dirs:
                    rdir = rank_dirs[0]
                    if (rdir / "op_statistic.csv").is_file():
                        files.append(rdir / "op_statistic.csv")
                    if step is None and (rdir / "kernel_details.csv").is_file():
                        files.append(rdir / "kernel_details.csv")
        elif path.is_file() and path.parent.is_dir():
            parent = path.parent
            files = [parent / "op_statistic.csv"] if (parent / "op_statistic.csv").is_file() else []
            if step is None and (parent / "kernel_details.csv").is_file():
                files.append(parent / "kernel_details.csv")
        if files:
            csv_observed = _observed_from_csv_files(files)
            csv_shares = csv_observed.pop("observed_share_by_bound", {})
            if csv_shares:
                shares = csv_shares
            for key, value in csv_observed.items():
                if key not in observed or observed[key] is None:
                    observed[key] = value
    dominant = _explicit_dominant(data)
    if dominant is None and shares and (sum(shares.values()) >= 0.5 or max(shares.values()) >= 0.5):
        dominant = max(shares, key=shares.__getitem__)
    for key, value in (("observed_step_ms", step), ("observed_tpot_p50_ms", tpot),
                       ("observed_ttft_p50_ms", ttft), ("observed_itl_p50_ms", itl),
                       ("observed_e2e_p50_ms", e2e)):
        if value is not None:
            observed[key] = round(value, 9)
    if concurrency is not None:
        observed["concurrency"] = concurrency
    ctx = _first_int(data, ("input_len", "prompt_tokens", "context_len", "seq_len", "context_tokens"), minimum=1)
    if ctx is not None:
        observed["context"] = ctx
    scope = _first_text(data, ("scope", "model_scope"))
    if scope and scope.lower() in {"body", "model"}:
        observed["scope"] = scope.lower()
    hw = _first_text(data, ("npu_type", "hardware", "sku", "chip"))
    if hw:
        observed["hardware"] = hw
    impl = _first_text(data, ("vllm_impl", "backend"))
    if impl:
        observed["vllm_impl"] = impl
    boundary = _first_text(data, ("measurement_boundary", "metric_boundary", "boundary"))
    if not boundary and (
        str(data.get("_source_kind") or "") in {"wave5_capture", "wave5_benchmark"}
        or any(
            any(k in node for k in ("benchmark_summary", "run_meta", "tpot_true_p50_ms",
                                    "step_from_metrics_ms", "itl_mean_p50_ms", "e2e_p50_ms"))
            for node in _candidate_mappings(data)
        )
    ) and any(v is not None for v in (tpot, ttft, _first_number(data, ("step_from_metrics_ms",)))):
        boundary = "client"
    if boundary:
        observed["measurement_boundary"] = boundary
    model_id = _first_text(data, ("model_name", "model", "served_model_name", "model_id", "family"))
    if model_id:
        observed["model"] = model_id
    for k, candidates in (("layers", ("num_hidden_layers", "num_layers", "layers")),
                          ("experts", ("num_routed_experts", "n_routed_experts", "num_experts")),
                          ("top_k", ("num_experts_per_tok", "top_k"))):
        val_i = _first_int(data, candidates, minimum=1)
        if val_i is not None:
            observed[k] = val_i
    quant = _first_text(data, ("quantization", "quant"))
    if quant and quant.lower() not in {"none", "auto", "ascend"}:
        observed["quantization"] = quant
    exec_mode = _first_text(data, ("execution_mode", "graph_capture_mode", "graph_mode"))
    if exec_mode is None:
        exec_mode = next(("eager" if h["is_eager"] else "full" for node in _candidate_mappings(data)
                          for h in (node, node.get("workload"), node.get("extra"), node.get("scene"))
                          if isinstance(h, Mapping) and isinstance(h.get("is_eager"), bool)), None)
    if exec_mode and exec_mode.lower() not in {"auto", "unspecified"}:
        observed["execution_mode"] = exec_mode
    for axis in ("tp", "dp", "ep", "pp"):
        ax_val = next((int(h[axis]) for node in _candidate_mappings(data)
                       for h in (node, node.get("parallelism"), node.get("parallel"), node.get("scene"))
                       if isinstance(h, Mapping) and isinstance(h.get(axis), int) and not isinstance(h.get(axis), bool)), None)
        if ax_val is not None:
            observed[axis] = ax_val
    if shares:
        observed["observed_share_by_bound"] = {key: round(value, 9) for key, value in shares.items()}
    if dominant:
        observed["observed_dominant_bottleneck"] = dominant
    phase = None
    for node in _candidate_mappings(data):
        workload = node.get("workload")
        if isinstance(workload, Mapping) and isinstance(workload.get("mode"), str):
            phase = workload["mode"]
            break
    if phase is None:
        phase = _first_text(data, ("phase", "mode"))
    if phase:
        phase = phase.split("_", 1)[0].lower()
    if phase in {"decode", "prefill"}:
        observed["phase"] = phase
    has_metric = any(
        key.startswith("observed_") or key in {
            "device_busy_union_ms", "device_activity_sum_ms", "exposed_comm_ms",
            "device_uncovered_ms", "host_gap_ms", "request_wall_ms", "aicpu_share", "host_share",
        }
        for key in observed
    )
    return observed if (has_metric or shares or dominant) else {}


__all__ = ["_extract_observed", "_observed_from_csv_files", "_shares_from_operator_rows"]
