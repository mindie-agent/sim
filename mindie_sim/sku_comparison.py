"""Multi-SKU same-config causal performance, efficiency and SLA comparison.

[Role]
  Evaluate two or more hardware SKUs under an identical model, parallelism,
  prompt/context geometry and SLA budget, producing exact latency, 4D
  utilization, VRAM capacity, SLA throughput and operator-family deltas.
[Invariants]
  Zero cross-call result caching: every SKU executes through the real causal
  ``Engine`` in its own fresh ``EvaluationContext``, and the compiled C11
  ``_engine_accel`` library is warmed once in the parent before any worker
  starts.  Each SKU composes its own case (``quant=None`` unless the caller
  overrides it) so a capability-dependent implementation is lowered honestly.
  Work is flattened over ``(sku, scenario)`` into exactly one bounded pool.
[Timeouts & Locks]
  One ``ProcessPoolExecutor`` sized by ``runtime_advisor.resolve_jobs``; a single
  worker runs the direct sequential path.  ``except TimeoutError: raise``
  precedes every ``except OSError:``.
[Side-Effect Policy]
  Pure in-memory evaluation; emits at most one stderr advisory before a heavy
  degraded run.  Public output carries only the compiler basename.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import time
from typing import Any, Mapping, Sequence

from . import runtime_advisor
from .compose import compose_case
from .prediction.efficiency import (
    DEFAULT_MAX_SEQUENCES, DEFAULT_SLA_MS, DEFAULT_SLA_SCENARIOS,
    SLA_SENSITIVITY_BASELINE, SLA_SENSITIVITY_FULL,
    analyze_network_efficiency, apply_sla_baseline_gains, solve_sla_sizing,
)
from .prediction.structure import parse_case

MIN_SKU_COUNT = 2
MS_PER_SECOND = 1000.0
DEFAULT_SCOPE = "body"
DEFAULT_PHASE = "decode"
MICROSECONDS_PER_MS = 1000
DEFAULT_QUERY_TOKENS = 1
PER_WORKER_RAM_GIB = 1.5
FAMILY_KEYS = (
    "attention", "moe_routed", "moe_dispatch_combine", "dense_gemm",
    "collective", "norm_activation",
)


def _resolve_sku_label(spec: str | Mapping[str, Any]) -> str:
    if isinstance(spec, str):
        return spec
    return str(spec.get("name") or spec.get("sku") or "custom_sku")


def _compose_kwargs(model: str, options: Mapping[str, Any]) -> dict[str, Any]:
    kwargs = {"model": model}
    for key in ("tp", "dp", "ep", "pp", "pcp", "num_devices", "ranks_per_node",
                "layers", "max_num_seqs", "max_num_batched_tokens",
                "graph_capture_mode", "async_scheduling", "mtp_proposal_tokens",
                "prompt_tokens", "output_tokens", "concurrency", "prefix_tokens"):
        if options.get(key) is not None:
            kwargs[key] = options[key]
    return kwargs


def _workload_geometry(parsed, *, context: int | None,
                       query_tokens: int | None) -> dict[str, int]:
    """Explicit workload geometry derived from the composed case.

    ``analyze_network_efficiency`` defaults to context zero and one sequence;
    every comparison entry therefore states the geometry it actually priced.
    """
    parallel = parsed.runtime.parallel
    prompts = [int(request.prompt_tokens) for request in parsed.workload.requests]
    resolved_context = (int(context) if context is not None
                        else max(prompts or [0]))
    resolved_query = (int(query_tokens) if query_tokens is not None
                      else DEFAULT_QUERY_TOKENS)
    return {
        "context": resolved_context,
        "query_tokens_per_sequence": resolved_query,
        "sequences_per_dp": max(1, min(int(parsed.runtime.max_num_seqs),
                                      int(parallel.dp))),
    }


def _summary_metrics(effective: Mapping[str, Any]) -> dict[str, Any]:
    families: dict[str, float] = {}
    for row in effective.get("breakdown_by_family", ()) or ():
        if not isinstance(row, Mapping):
            continue
        family = str(row.get("family", ""))
        families[family] = families.get(family, 0.0) + float(
            row.get("total_ms", 0.0))
    return {
        "step_ms": float(effective["graph_duration_ms"]),
        "operator_sum_ms": float(effective.get("operator_sum_ms") or 0.0),
        "bottleneck": effective.get("dominant_bottleneck"),
        "mbu": effective.get("network_mbu"),
        "cube_mfu": effective.get("network_cube_mfu"),
        "vec_mfu": effective.get("network_vec_mfu"),
        "link_util": effective.get("network_link_util"),
        "roofline_efficiency": effective.get("network_roofline_efficiency"),
        "family_breakdown": families,
    }


def _runtime_profile(case_dict: Mapping[str, Any]) -> dict[str, Any]:
    """Declared quantization / kernel profile of this SKU's own composition."""
    runtime = case_dict.get("runtime") or {}
    model = case_dict.get("model") or {}
    quant = model.get("quant") if isinstance(model.get("quant"), Mapping) else None
    metadata = runtime.get("metadata") or {}
    profile = metadata.get("va_profile") if isinstance(metadata, Mapping) else None
    return {
        "quantization": quant.get("name") if quant else None,
        "weight_format": quant.get("weight_format") if quant else None,
        "expert_weight_format": quant.get("expert_weight_format") if quant else None,
        "kv_dtype": quant.get("kv_dtype") if quant else None,
        "hardware_sku": (profile or {}).get("sku") if isinstance(profile, Mapping) else None,
        "graph_capture_mode": runtime.get("graph_capture_mode"),
        "quantization_basis": ((profile or {}).get("quantization_basis")
                               if isinstance(profile, Mapping) else None),
    }


def _evaluate_sku(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One SKU's efficiency point plus its first SLA scenario (picklable)."""
    started = time.perf_counter()
    case_dict = payload["case"]
    parsed = parse_case(case_dict)
    geometry = payload["geometry"]
    effective = analyze_network_efficiency(
        parsed, phase=payload["phase"],
        tokens=geometry["query_tokens_per_sequence"],
        context=geometry["context"],
        sequences=geometry["sequences_per_dp"],
        scope=payload["scope"],
    )
    sla = solve_sla_sizing(
        parsed, sla_ms=payload["sla_ms"], context=geometry["context"],
        tokens_per_seq=geometry["query_tokens_per_sequence"],
        max_sequences_per_dp=payload["max_batch"], phase=payload["phase"],
        scope=payload["scope"], scenarios=[payload["scenario"]],
        sensitivity=payload["sensitivity"], jobs=1, base_b_hint=1,
    )
    return {
        "sku": payload["sku"],
        "elapsed_ms": round((time.perf_counter() - started) * MS_PER_SECOND, 2),
        "metrics": _summary_metrics(effective),
        "runtime_profile": _runtime_profile(case_dict),
        "sla_scenario": sla["scenarios"][0],
        "base_b_hint": int(sla["scenarios"][0].get("base_b_hint") or 1),
        "async_scheduling_memory": sla.get("async_scheduling_memory"),
        "solver": sla.get("solver"),
        "sla_header": {
            key: sla.get(key) for key in
            ("schema", "sla_ms", "context", "phase", "scope", "tokens_per_seq",
             "emitted_tokens_per_seq", "dp", "max_sequences_per_dp")
        },
        "accel_status": effective.get("accel_status"),
        "engine_backend": effective.get("engine_backend"),
    }


def _evaluate_scenario(payload: Mapping[str, Any]) -> dict[str, Any]:
    """One ``(sku, scenario)`` counterfactual in its own worker."""
    parsed = parse_case(payload["case"])
    sla = solve_sla_sizing(
        parsed, sla_ms=payload["sla_ms"], context=payload["context"],
        tokens_per_seq=payload["query_tokens_per_sequence"],
        max_sequences_per_dp=payload["max_batch"], phase=payload["phase"],
        scope=payload["scope"], scenarios=[payload["scenario"]],
        sensitivity=payload["sensitivity"], jobs=1,
        base_b_hint=payload["base_b_hint"],
    )
    return {"sku": payload["sku"],
            "sla_scenario": sla["scenarios"][0],
            "accel_status": None, "engine_backend": None}


def _run_tasks(tasks, workers: int) -> list[dict[str, Any]]:
    if workers <= 1:
        return [fn(payload) for fn, payload in tasks]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = [pool.submit(fn, payload) for fn, payload in tasks]
        return [future.result() for future in futures]


def compare_skus(
    model: str,
    skus: Sequence[str | Mapping[str, Any]],
    *,
    sla_ms: float | None = DEFAULT_SLA_MS,
    max_batch: int = DEFAULT_MAX_SEQUENCES,
    sensitivity: str = SLA_SENSITIVITY_BASELINE,
    scope: str = DEFAULT_SCOPE,
    phase: str = DEFAULT_PHASE,
    context: int | None = None,
    query_tokens_per_sequence: int | None = None,
    quant: str | Mapping[str, Any] | None = None,
    jobs: int | str = "auto",
    emit_advisory: bool = True,
    **compose_options: Any,
) -> dict[str, Any]:
    """Compare ``>= 2`` SKUs under identical model and deployment geometry."""
    if len(skus) < MIN_SKU_COUNT:
        raise ValueError("compare_skus requires at least two SKUs")
    wall_start = time.perf_counter()
    # Prebuild/warm the compiled kernel once in the parent: workers then map the
    # immutable library instead of compiling in a stampede.
    runtime_advisor.probe_host_environment(ensure_accel=True)
    base_kwargs = _compose_kwargs(model, compose_options)
    if quant is not None:
        base_kwargs["quant"] = quant
    resolved: list[tuple[str, dict[str, Any]]] = []
    for spec in skus:
        label = _resolve_sku_label(spec)
        composed = compose_case(hardware=spec if isinstance(spec, str) else
                                str(spec.get("base_hardware") or "a5"),
                                **base_kwargs)
        resolved.append((label, composed))
    geometry = _workload_geometry(parse_case(resolved[0][1]), context=context,
                                  query_tokens=query_tokens_per_sequence)
    scenarios = ([DEFAULT_SLA_SCENARIOS[0]]
                 if str(sensitivity).strip().lower() == SLA_SENSITIVITY_BASELINE
                 else list(DEFAULT_SLA_SCENARIOS))
    has_sla = sla_ms is not None and float(sla_ms) > 0.0
    per_sku_tasks = len(resolved) if has_sla else 0
    work_units = max(1, per_sku_tasks * max(1, len(scenarios))) + len(resolved)
    workers = runtime_advisor.resolve_jobs(
        jobs, task_count=max(1, per_sku_tasks * max(1, len(scenarios))
                             + len(resolved)),
        per_worker_ram_gib=PER_WORKER_RAM_GIB)
    if emit_advisory:
        runtime_advisor.emit_heavy_run_notice(
            "compare-skus", workers=workers, work_units=work_units)
    baseline_tasks = [
        (_evaluate_sku if has_sla else _efficiency_only,
         {"sku": label, "case": case_dict, "geometry": geometry,
          "phase": phase, "scope": scope, "sla_ms": sla_ms or 0.0,
          "max_batch": max_batch, "scenario": scenarios[0],
          "sensitivity": sensitivity})
        for label, case_dict in resolved
    ]
    reports = _run_tasks(baseline_tasks, workers)
    hints = {row["sku"]: int(row.get("base_b_hint") or 1) for row in reports}
    follow_tasks = []
    for label, case_dict in resolved:
        for scenario in scenarios[1:]:
            follow_tasks.append((_evaluate_scenario, {
                "sku": label, "case": case_dict, "sla_ms": sla_ms,
                "context": geometry["context"],
                "query_tokens_per_sequence": geometry["query_tokens_per_sequence"],
                "max_batch": max_batch, "phase": phase, "scope": scope,
                "scenario": scenario, "sensitivity": sensitivity,
                "base_b_hint": hints.get(label, 1)}))
    extra = _run_tasks(follow_tasks, workers) if follow_tasks else []
    by_sku = {row["sku"]: row for row in reports}
    for row in extra:
        target = by_sku[row["sku"]]
        target.setdefault("sla_scenarios", []).append(row["sla_scenario"])
        if row.get("accel_status") is not None:
            target.setdefault("accel_statuses", []).append(row["accel_status"])
    for row in reports:
        ordered = [row["sla_scenario"], *row.get("sla_scenarios", [])]
        row["sla_scenarios"] = ordered
        apply_sla_baseline_gains(ordered, ordered[0])
        row.pop("sla_scenario", None)
    reference = reports[0]
    comparisons = []
    for candidate in reports[1:]:
        comparisons.append(_compare_pair(reference, candidate))
    wall_ms = round((time.perf_counter() - wall_start) * MS_PER_SECOND, 2)
    statuses = [row.get("accel_status") for row in reports
                if row.get("accel_status")]
    coverage = _coverage(statuses)
    env, advisory = runtime_advisor.build_runtime_environment(
        workers_used=workers, estimated_work_units=work_units)
    env["wall_ms"] = wall_ms
    env["workers_used"] = workers
    env["execution_backend"] = coverage["execution_backend"]
    env["native_steps"] = coverage["native_steps"]
    env["python_steps"] = coverage["python_steps"]
    return {
        "schema": "mindie-sim.sku-comparison/1",
        "model": model,
        "scope": scope,
        "phase": phase,
        "quant": quant,
        "geometry": geometry,
        "sla_ms": sla_ms,
        "sensitivity": sensitivity,
        "scenarios_evaluated": [scenario.get("id") for scenario in scenarios],
        "skus": reports,
        "comparisons": comparisons,
        "execution_coverage": coverage,
        "runtime_environment": env,
        "agent_advisory": advisory,
    }


def _efficiency_only(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Efficiency point only, for ``sla_ms=None`` comparisons."""
    started = time.perf_counter()
    parsed = parse_case(payload["case"])
    geometry = payload["geometry"]
    effective = analyze_network_efficiency(
        parsed, phase=payload["phase"],
        tokens=geometry["query_tokens_per_sequence"],
        context=geometry["context"],
        sequences=geometry["sequences_per_dp"], scope=payload["scope"])
    return {
        "sku": payload["sku"],
        "elapsed_ms": round((time.perf_counter() - started) * MS_PER_SECOND, 2),
        "metrics": _summary_metrics(effective),
        "runtime_profile": _runtime_profile(payload["case"]),
        "sla_scenarios": [],
        "accel_status": effective.get("accel_status"),
        "engine_backend": effective.get("engine_backend"),
    }


def _coverage(statuses) -> dict[str, Any]:
    rows = [status for status in statuses if isinstance(status, Mapping)]
    native = sum(int(row.get("native_steps") or 0) for row in rows)
    python = sum(int(row.get("python_steps") or 0) for row in rows)
    if native and python:
        backend = runtime_advisor.BACKEND_MIXED
    elif native:
        backend = runtime_advisor.BACKEND_C11
    else:
        backend = runtime_advisor.BACKEND_PYTHON
    return {"execution_backend": backend, "native_steps": native,
            "python_steps": python, "engines_reported": len(rows)}


def _family_totals(row: Mapping[str, Any]) -> dict[str, float]:
    return dict(row.get("metrics", {}).get("family_breakdown") or {})


def _compare_pair(reference: Mapping[str, Any],
                  candidate: Mapping[str, Any]) -> dict[str, Any]:
    ref_step = max(float(reference["metrics"]["step_ms"]), 1e-12)
    cand_step = max(float(candidate["metrics"]["step_ms"]), 1e-12)
    ref_rows = reference.get("sla_scenarios") or []
    cand_rows = candidate.get("sla_scenarios") or []
    ref_base = ref_rows[0] if ref_rows else {}
    cand_base = cand_rows[0] if cand_rows else {}
    ref_tps = float(ref_base.get("tps_feasible") or 0.0)
    cand_tps = float(cand_base.get("tps_feasible") or 0.0)
    ref_families = _family_totals(reference)
    cand_families = _family_totals(candidate)
    family_delta = {
        key: round(cand_families.get(key, 0.0) - ref_families.get(key, 0.0), 6)
        for key in sorted(set(ref_families) | set(cand_families))
    }
    dp_ref = max(1, int(reference.get("sla_header", {}).get("dp") or 1))
    dp_cand = max(1, int(candidate.get("sla_header", {}).get("dp") or 1))
    return {
        "reference_sku": reference["sku"],
        "candidate_sku": candidate["sku"],
        "step_ms_delta": round(cand_step - ref_step, 6),
        "step_speedup_x": round(ref_step / cand_step, 6),
        "tps_feasible_gain_x": (round(cand_tps / ref_tps, 6)
                                if ref_tps > 0.0 else None),
        "tps_per_die_gain_x": (
            round((cand_tps / dp_cand) / (ref_tps / dp_ref), 6)
            if ref_tps > 0.0 and cand_tps > 0.0 else None),
        "b_feasible_delta": (int(cand_base.get("b_feasible") or 0)
                            - int(ref_base.get("b_feasible") or 0)),
        "b_vram_max_delta": (int(cand_base.get("b_vram_max") or 0)
                            - int(ref_base.get("b_vram_max") or 0)),
        "family_exposed_ms_delta": family_delta,
    }
