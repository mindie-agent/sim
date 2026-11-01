"""Shared primitives for profile ingestion and the counterfactual advisor.

[Role]
  Define canonical bottleneck classes, bounded heterogeneous-record lookup,
  finite-number coercion, and text classification used by advisor modules.

[Invariants]
  * Missing values stay ``None``; booleans are never numeric measurements.
  * Canonical bottleneck classes are exactly the four public C7-aligned names.
  * Recursive profile lookup is bounded by depth and node budget.

[Timeouts & Locks]
  Pure in-memory stdlib helpers; no I/O, sleeps, or locks.

[Side-Effect Policy]
  No mutation of inputs; every helper returns new records or scalars.
"""
from __future__ import annotations

from dataclasses import fields
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .schema.case import Software
from .schema.parallel import ParallelSpec

CANONICAL_BOTTLENECKS = ("comm_bound", "launch_host_bound", "memory_kv_bound", "compute_bound")
_PROFILE_MAX_BYTES = 50 * 1024**2
_CSV_MAX_ROWS = 2_000_000
_SLA_SOLVER_MAX_CONTEXT = 16_384
_COMM_TOKENS = ("comm", "link", "hccl", "allreduce", "allgather", "reduce_scatter", "alltoall",
                "all_to_all", "network")
_LAUNCH_TOKENS = ("launch", "host", "dispatch", "submission", "idle", "eager",
                  "cpu", "aicpu", "ai_cpu")
_MEMORY_TOKENS = ("memory", "membound", "kv", "hbm", "mte", "cache", "bandwidth", "dram",
                  "copy", "fill", "gather", "scatter")
_COMPUTE_TOKENS = ("compute", "cube", "vector", "aic", "aiv", "math", "matmul", "gemm",
                   "attention", "moe", "ffn", "expert", "softmax", "norm", "swiglu")


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            return None
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _positive(value: Any) -> float | None:
    result = _number(value)
    return result if result is not None and result >= 0.0 else None


def _pct(delta: float | None, base: float | None) -> float | None:
    if delta is None or base is None or base <= 0.0:
        return None
    return round(delta / base * 100.0, 6)


def _deep_values(mapping: Any, keys: Sequence[str], depth: int = 4, budget: int = 4096) -> list[Any]:
    """Bounded recursive lookup for heterogeneous profiler/capture records."""
    found: list[Any] = []
    skip_children = {"semantic_block_breakdown", "top_operators", "top_hotspot_kernels",
                     "top_operators_by_device_time"}
    stack: list[tuple[Any, int]] = [(mapping, 0)]
    while stack and budget > 0:
        node, level = stack.pop()
        budget -= 1
        if isinstance(node, Mapping):
            for key in keys:
                if key in node:
                    found.append(node[key])
            if level < depth:
                stack.extend((value, level + 1) for k, value in node.items()
                             if k not in skip_children and isinstance(value, (Mapping, list, tuple)))
        elif isinstance(node, (list, tuple)) and level < depth:
            stack.extend((value, level + 1) for value in node[:256]
                         if isinstance(value, (Mapping, list, tuple)))
    return found


def _first_number(mapping: Any, keys: Sequence[str]) -> float | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            result = _number(value)
            if result is not None:
                return result
    return None


def _first_text(mapping: Any, keys: Sequence[str]) -> str | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _first_int(mapping: Any, keys: Sequence[str], minimum: int = 0) -> int | None:
    for key in keys:
        for value in _deep_values(mapping, (key,)):
            if isinstance(value, bool):
                continue
            if isinstance(value, str):
                try:
                    value = int(float(value.strip()))
                except ValueError:
                    continue
            if isinstance(value, int) and value >= minimum:
                return int(value)
    return None


def _sanitize_case(case: Any) -> Any:
    """Move top-level runtime observed/provenance blocks into runtime.metadata."""
    if not isinstance(case, Mapping) or not isinstance(case.get("runtime"), Mapping):
        return case
    runtime = dict(case["runtime"])
    moved = {key: runtime.pop(key) for key in ("observed_profile", "profiler_provenance")
             if key in runtime}
    if moved:
        meta = dict(runtime.get("metadata") or {})
        for key, value in moved.items():
            meta.setdefault(key, value)
        runtime["metadata"] = meta
        case = dict(case)
        case["runtime"] = runtime
    return case


def _has_measured_observation(observed: Any) -> bool:
    if not isinstance(observed, Mapping):
        return False
    numeric_keys = ("observed_step_ms", "step_time_ms", "observed_tpot_p50_ms", "tpot_p50_ms",
                    "observed_ttft_p50_ms", "ttft_p50_ms", "observed_operator_sum_ms",
                    "device_activity_sum_ms", "device_busy_union_ms")
    if any(_number(observed.get(key)) is not None for key in numeric_keys):
        return True
    shares = observed.get("observed_share_by_bound") or observed.get("bottleneck_shares")
    if isinstance(shares, Mapping) and any(_number(value) is not None for value in shares.values()):
        return True
    operators = observed.get("top_operators")
    return isinstance(operators, list) and bool(operators)


def _has_observed_measurement(observed: Any) -> bool:
    if not isinstance(observed, Mapping):
        return False
    numeric_keys = ("observed_step_ms", "step_time_ms", "observed_tpot_p50_ms", "tpot_p50_ms",
                    "observed_ttft_p50_ms", "ttft_p50_ms", "observed_operator_sum_ms",
                    "device_activity_sum_ms", "device_busy_union_ms")
    if any(_number(observed.get(key)) is not None for key in numeric_keys):
        return True
    shares = observed.get("observed_share_by_bound") or observed.get("bottleneck_shares")
    if isinstance(shares, Mapping) and any(_number(value) is not None for value in shares.values()):
        return True
    if isinstance(observed.get("observed_dominant_bottleneck")
                  or observed.get("dominant_bottleneck"), str):
        return True
    operators = observed.get("top_operators")
    return isinstance(operators, list) and bool(operators)


def _candidate_mappings(data: Any) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[int] = set()

    def visit(node: Any, depth: int) -> None:
        if not isinstance(node, Mapping) or id(node) in seen or depth > 3:
            return
        seen.add(id(node))
        result.append(node)
        for key in ("run_meta", "benchmark_summary", "summary", "scene", "capture", "provenance",
                    "metrics", "workload", "parallelism", "parallel", "architecture_override",
                    "design", "config", "case", "attribution", "accounting", "step_accounting",
                    "window_accounting", "summary_document", "attribution_document", "runtime",
                    "metadata", "observed_profile"):
            visit(node.get(key), depth + 1)

    visit(data, 0)
    return result


def _canonical_bottleneck(value: Any) -> str | None:
    if isinstance(value, Mapping):
        for key in ("dominant_bottleneck", "dominant_bound", "bottleneck", "bound", "primary_bound"):
            if key in value:
                result = _canonical_bottleneck(value[key])
                if result:
                    return result
        return None
    if not isinstance(value, str):
        return None
    raw = value.strip().lower()
    if raw in CANONICAL_BOTTLENECKS:
        return raw
    if any(token in raw for token in _COMM_TOKENS):
        return "comm_bound"
    if any(token in raw for token in _LAUNCH_TOKENS):
        return "launch_host_bound"
    if any(token in raw for token in _MEMORY_TOKENS):
        return "memory_kv_bound"
    if any(token in raw for token in _COMPUTE_TOKENS):
        return "compute_bound"
    return None


def _classify_name(name: str) -> str:
    lowered = str(name).lower()
    if any(token in lowered for token in _COMM_TOKENS):
        return "comm_bound"
    if any(token in lowered for token in _LAUNCH_TOKENS):
        return "launch_host_bound"
    if any(token in lowered for token in _MEMORY_TOKENS):
        return "memory_kv_bound"
    return "compute_bound"


def _normalize_shares(shares: Mapping[str, Any]) -> dict[str, float]:
    totals = {name: 0.0 for name in CANONICAL_BOTTLENECKS}
    for key, raw in shares.items():
        value = _positive(raw)
        canonical = _canonical_bottleneck(key) if value is not None else None
        if canonical:
            totals[canonical] += value
    total = sum(totals.values())
    if total <= 0.0:
        return {}
    scale = total if total > 1.000001 else 1.0
    return {name: round(value / scale, 9) for name, value in totals.items() if value > 0.0}


__all__ = [name for name in globals() if not name.startswith("__")]
