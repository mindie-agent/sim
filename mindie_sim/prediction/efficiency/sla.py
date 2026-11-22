"""SLA sizing under hardware counterfactuals.

[Role] Evaluate graph latency and capacity to size concurrency and throughput
  across hardware rate and capacity scenarios.
[Invariants] Every feasible candidate passes latency and lifecycle capacity
  checks on the same evaluated geometry.
[Timeouts & Locks] Finite concurrency probes honor the active evaluation
  deadline. The solver acquires no additional locks.
[Side-Effect Policy] Cases are read-only and the solver owns its probe caches.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError

from ..capacity import Memory
from ..evaluation import check_active_deadline
from ..operator_analysis import target_graphs
from ..structure import parse_case
from .analysis import analyze_network_efficiency, scale_case_hardware
from .attribution import _lower_and_run_graph, _pkg_attr
from .capacity_window import (
    _resolve_speculative_tokens, _vram_max_sequences_per_dp,
    apply_sla_baseline_gains, capacity_window, select_jointly_feasible_candidate,
)
from .peaks import DEFAULT_OUTPUT_BUDGET, _BOUNDS, _SCALE_KEYS, _check_pos_float, _validate_geometry


DEFAULT_SLA_MS = 30.0
DEFAULT_SLA_CONTEXT = 126000
DEFAULT_MAX_SEQUENCES = 256
NON_MONOTONIC_PROBE_CAP = 4
SLA_HEADROOM_PROBE_STEPS = 3
SLA_HEADROOM_MULTIPLIER = 3.0
#: Counterfactual-set selectors for :func:`solve_sla_sizing`.
SLA_SENSITIVITY_BASELINE = "baseline"
SLA_SENSITIVITY_FULL = "full"
#: A per-sequence marginal step below this is flat (or negative): no secant.
SECANT_MIN_SLOPE_MS = 1e-9

DEFAULT_SLA_SCENARIOS: tuple[dict[str, Any], ...] = (
    {"id": "baseline", "label": "1x HBM, 1x Compute", "hbm_bw_scale": 1.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm", "label": "2x HBM, 1x Compute", "hbm_bw_scale": 2.0, "cube_flops_scale": 1.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_compute", "label": "1x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
    {"id": "2x_cube_only", "label": "1x HBM, 2x Cube, 1x Vector", "hbm_bw_scale": 1.0, "cube_flops_scale": 2.0, "vector_ops_scale": 1.0, "link_bw_scale": 1.0},
    {"id": "2x_hbm_2x_compute", "label": "2x HBM, 2x Cube + 2x Vector", "hbm_bw_scale": 2.0, "cube_flops_scale": 2.0, "vector_ops_scale": 2.0, "link_bw_scale": 1.0},
)


def solve_sla_sizing(
    case: CaseSpec | Mapping[str, Any], *, sla_ms: float = DEFAULT_SLA_MS,
    context: int = DEFAULT_SLA_CONTEXT, tokens_per_seq: int | None = None,
    emitted_tokens_per_seq: float | None = None,
    max_sequences_per_dp: int = DEFAULT_MAX_SEQUENCES, phase: str = "decode",
    scope: str = "model", output_budget: int = DEFAULT_OUTPUT_BUDGET,
    scenarios: Sequence[Mapping[str, Any]] | None = None, b1_analysis: Mapping[str, Any] | None = None,
    evaluation_context: Any | None = None,
    deadline_monotonic: float | None = None,
    sensitivity: str = SLA_SENSITIVITY_FULL,
    jobs: int | str = 1,
    fast_span_probes: bool = True,
    base_b_hint: int = 1,
) -> dict[str, Any]:
    """Solve SLA-constrained maximum concurrency B*_SLA and throughput across hardware scenarios."""
    from ..unmodeled_gate import assert_timing_supported
    check_active_deadline()
    _check_pos_float("sla_ms", sla_ms)
    if isinstance(max_sequences_per_dp, bool) or not isinstance(max_sequences_per_dp, int) or max_sequences_per_dp < 1:
        raise ValueError(f"max_sequences_per_dp_must_be_positive_int:{max_sequences_per_dp}")
    if isinstance(output_budget, bool) or not isinstance(output_budget, int) or output_budget < 0:
        raise ValueError(f"output_budget_must_be_nonnegative_int:{output_budget}")

    # ``sensitivity`` only selects the counterfactual *set*; an explicit
    # ``scenarios`` list stays authoritative.  One scenario per call keeps the
    # caller's flat worker pool the only pool (no nested pools).
    if scenarios is not None:
        scenario_source = list(scenarios)
    elif str(sensitivity).strip().lower() == SLA_SENSITIVITY_BASELINE:
        scenario_source = [DEFAULT_SLA_SCENARIOS[0]]
    else:
        scenario_source = list(DEFAULT_SLA_SCENARIOS)
    scenario_list = [dict(scenario) for scenario in scenario_source]
    for scenario in scenario_list:
        if "target_mbu" in scenario:
            raise ValueError("target_efficiency_scaling_retired:use_optimize_ladder")
        for key in _SCALE_KEYS:
            _check_pos_float(f"scenario_scale_{key}", scenario.get(key, 1.0))

    assert_timing_supported(case, entry="solve_sla_sizing")
    parsed = parse_case(case)
    q_tokens, emit_tokens = _resolve_speculative_tokens(parsed, output_budget=output_budget, tokens_per_seq=tokens_per_seq, emitted_tokens_per_seq=emitted_tokens_per_seq)
    # A prefill step processes its query; the following decode step commits its
    # own accepted tokens.  Keep both quantities: the lifecycle capacity must
    # use the decode commit, never the prompt-processing volume.
    decode_q_tokens, decode_emit_tokens = _resolve_speculative_tokens(
        parsed, output_budget=output_budget, tokens_per_seq=None,
        emitted_tokens_per_seq=None)
    _validate_geometry(phase, q_tokens, context, 1, scope)
    batch_budget = int(parsed.runtime.max_num_batched_tokens)
    unchunkable = (not parsed.runtime.enable_chunked_prefill
                   and q_tokens > batch_budget)
    rt_max_b = min(int(parsed.runtime.max_num_seqs), batch_budget // q_tokens)
    if parsed.runtime.enable_chunked_prefill:
        # A chunked prompt may exceed the per-step budget and still admit one
        # sequence; only the unchunked declaration preserves the zero.
        rt_max_b = max(1, rt_max_b)
    if phase == "decode" and parsed.runtime.graph_capture_mode == "full" and parsed.runtime.graph_capture_keys and dict(parsed.runtime.metadata).get("captured_execution_mode") == "FULL_DECODE_ONLY":
        rt_max_b = min(rt_max_b,
                       max(1, max(parsed.runtime.graph_capture_keys) // q_tokens))
    b_max, dp = min(int(max_sequences_per_dp), rt_max_b), parsed.runtime.parallel.dp

    scaled_cases: dict[tuple[float, ...], CaseSpec] = {}
    graphs_by_b: dict[int, Any] = {}
    vram_by_cap: dict[float, tuple[int, int, int]] = {}
    graph_cache: dict[tuple[tuple[float, ...], int, bool], tuple[Any, CaseSpec, dict[str, Any]]] = {}
    fast_spans = bool(fast_span_probes)
    analysis_cache: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}

    def get_scaled(s_key: tuple[float, ...]) -> CaseSpec:
        if s_key not in scaled_cases:
            scaled_cases[s_key] = scale_case_hardware(parsed, **dict(zip(_SCALE_KEYS, s_key)))
        return scaled_cases[s_key]

    baseline_scales = tuple(1.0 for _ in _SCALE_KEYS)

    def get_bundle(s_key: tuple[float, ...], b: int, *,
                   fast: bool = False) -> tuple[Any, CaseSpec, dict[str, Any]]:
        check_active_deadline()
        if (s_key, b, fast) not in graph_cache:
            if b not in graphs_by_b:
                graphs_by_b[b] = _pkg_attr("target_graphs", target_graphs)(parsed, phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope)
            context_reuse = evaluation_context if s_key == baseline_scales else None
            g, ec, gr = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                lowered_graphs=graphs_by_b[b], evaluation_context=context_reuse,
                fast_span_only=bool(fast))
            assert gr is not None
            graph_cache[s_key, b, fast] = (g, ec, gr)
        return graph_cache[s_key, b, fast]

    def get_analysis(s_key: tuple[float, ...], b: int) -> dict[str, Any]:
        check_active_deadline()
        if (s_key, b) not in analysis_cache:
            # ``b1_analysis`` is retained only for API compatibility.  It does not
            # carry the case plan/boundary identity needed to prove the same input,
            # so it may never replace an execution evaluated here.  The graph cache
            # above is the only reuse boundary.
            analysis_cache[s_key, b] = _pkg_attr("analyze_network_efficiency", analyze_network_efficiency)(
                get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                _precomputed_graph=get_bundle(s_key, b, fast=False),
                evaluation_context=(evaluation_context
                                    if s_key == baseline_scales else None))
        return analysis_cache[s_key, b]

    def eval_at(s_key: tuple[float, ...], b: int, *, full: bool = False) -> dict[str, Any]:
        if unchunkable:
            return {"step_ms": 0.0, "step_basis": "geometry_rejected",
                    "step_is_lower_bound": False, "operator_sum_ms": 0.0,
                    "network_mbu": None,
                    "network_cube_mfu": None, "network_vec_mfu": None,
                    "network_link_util": None,
                    "network_roofline_efficiency": None,
                    "bottleneck_shares": {}}
        if not full:
            return {"step_ms": float(
                get_bundle(s_key, b, fast=fast_spans)[2]["duration_ms"])}
        ans = get_analysis(s_key, b)
        shares = {k: ans["breakdown_by_bound"][k]["share_of_operator_sum"] for k in _BOUNDS}
        return {"step_ms": float(ans["graph_duration_ms"]), "step_basis": "graph_evaluation", "step_is_lower_bound": False, "operator_sum_ms": float(ans["operator_sum_ms"]), "network_mbu": float(ans["network_mbu"]), "network_cube_mfu": float(ans["network_cube_mfu"]), "network_vec_mfu": float(ans["network_vec_mfu"]), "network_link_util": float(ans["network_link_util"]), "network_roofline_efficiency": float(ans["network_roofline_efficiency"]), "bottleneck_shares": shares}

    base_b_hint = max(1, int(base_b_hint))
    domain_cap = b_max
    domain_reason = ("prompt_exceeds_batch_budget_without_chunking"
                     if unchunkable else None)

    window_provider: dict[str, Any] = {"fn": None}

    def try_step_ok(s_key: tuple[float, ...], b: int, *, seen: set[int] | None = None,
                    checks: dict[int, dict[str, Any]] | None = None, capacity_max: int | None = None) -> bool:
        nonlocal domain_cap, domain_reason
        if seen is not None:
            seen.add(int(b))
        if b > domain_cap:
            return False
        try:
            step = float(eval_at(s_key, b, full=False)["step_ms"])
            capacity_pass = True
            if checks is not None:
                window_for = window_provider["fn"]
                if window_for is None:
                    raise ModelingError("sla_capacity_window_provider_not_bound")
                current_window, lifecycle_window = window_for(int(b))
                capacity_pass = bool(
                    (capacity_max is None or b <= capacity_max)
                    and current_window["fits"] and lifecycle_window["fits"])
                checks[int(b)] = {
                    "b": int(b), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(capacity_max or 0),
                                 "current_fits": bool(current_window["fits"]),
                                 "lifecycle_fits": bool(lifecycle_window["fits"]),
                                 "pass": capacity_pass},
                    "feasible": bool(step <= sla_ms and capacity_pass)}
            return bool(step <= sla_ms)
        except UnmodeledCapabilityError:
            # An unmodeled core is not an infeasible candidate.
            raise
        except ModelingError as exc:
            if b <= 1:
                raise
            domain_cap, domain_reason = min(domain_cap, b - 1), str(exc)
            return False

    memory_cache: dict[tuple[int, bool | None, float], Memory] = {}

    def memory_for(tokens: int, async_scheduling: bool | None, scale: float) -> Memory:
        key = (int(tokens), async_scheduling, float(scale))
        if key not in memory_cache:
            base_case = (scale_case_hardware(parsed, hbm_capacity_scale=scale)
                         if scale != 1.0 else parsed)
            if async_scheduling is not None:
                base_case = replace(base_case, runtime=replace(
                    base_case.runtime, async_scheduling=bool(async_scheduling)))
            memory_cache[key] = Memory(base_case, workspace_tokens=int(tokens))
        return memory_cache[key]

    def solve_one(scen: Mapping[str, Any]) -> dict[str, Any]:
        nonlocal base_b_hint
        check_active_deadline()
        s_key = tuple(float(scen.get(k, 1.0)) for k in _SCALE_KEYS)
        hbm_capacity_scale = s_key[-1]
        if hbm_capacity_scale not in vram_by_cap and unchunkable:
            vram_by_cap[hbm_capacity_scale] = (0, 0, 0)
        if hbm_capacity_scale not in vram_by_cap:
            b_sync = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, async_scheduling=False, phase=phase, memory=memory_for(q_tokens, False, hbm_capacity_scale))
            b_async = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, async_scheduling=True, phase=phase, memory=memory_for(q_tokens, True, hbm_capacity_scale))
            b_cfg = b_async if parsed.runtime.async_scheduling else b_sync
            if parsed.runtime.memory:
                b_cfg = _vram_max_sequences_per_dp(parsed, context=context, tokens_per_seq=q_tokens, output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale, emitted_tokens_per_seq=emit_tokens, phase=phase, memory=memory_for(q_tokens, None, hbm_capacity_scale))
            vram_by_cap[hbm_capacity_scale] = (b_cfg, b_sync, b_async)
        (b_vram, b_vram_sync, b_vram_async) = vram_by_cap[hbm_capacity_scale]
        if phase == "prefill":
            life_ctx = context + q_tokens
            verify_tokens = decode_q_tokens
            b_life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                      output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale,
                                                      emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=False, phase="decode",
                                                      memory=memory_for(verify_tokens, False, hbm_capacity_scale))
            b_life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=verify_tokens,
                                                       output_budget=output_budget, hbm_capacity_scale=hbm_capacity_scale,
                                                       emitted_tokens_per_seq=decode_emit_tokens, async_scheduling=True, phase="decode",
                                                       memory=memory_for(verify_tokens, True, hbm_capacity_scale))
            b_lifecycle = b_life_async if parsed.runtime.async_scheduling else b_life_sync
        else:
            life_ctx = context
            verify_tokens = q_tokens
            b_life_sync, b_life_async, b_lifecycle = b_vram_sync, b_vram_async, b_vram
        b_joint = min(b_vram, b_lifecycle)

        window_cache: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}

        def rejected_window():
            empty = {
                "supply_bytes": 0.0, "active_kv_bytes": 0.0,
                "retiring_kv_bytes": 0.0, "workspace_bytes": 0.0,
                "peak_bytes": 0.0, "headroom_bytes": 0.0, "fits": False,
            }
            return empty

        def candidate_windows(b: int) -> tuple[dict[str, Any], dict[str, Any]]:
            if b in window_cache:
                return window_cache[b]
            if unchunkable:
                window_cache[b] = (rejected_window(), rejected_window())
                return window_cache[b]
            current_window = capacity_window(
                parsed, phase=phase, context=context, tokens_per_seq=q_tokens,
                sequences=b, output_budget=output_budget,
                emitted_tokens_per_seq=emit_tokens, hbm_capacity_scale=hbm_capacity_scale)
            if phase == "decode":
                lifecycle_window = current_window
            else:
                lifecycle_window = capacity_window(
                    parsed, phase="decode", context=life_ctx, tokens_per_seq=verify_tokens,
                    sequences=b, output_budget=output_budget,
                    emitted_tokens_per_seq=decode_emit_tokens,
                    hbm_capacity_scale=hbm_capacity_scale)
            window_cache[b] = (current_window, lifecycle_window)
            return current_window, lifecycle_window

        window_provider["fn"] = candidate_windows
        b1_ms = (0.0 if unchunkable else
                 float(get_bundle(s_key, 1, fast=fast_spans)[2]["duration_ms"]))
        current_one, lifecycle_one = candidate_windows(1)
        b1_capacity = bool(b_joint >= 1 and current_one["fits"]
                           and lifecycle_one["fits"] and not unchunkable)
        candidate_checks: dict[int, dict[str, Any]] = {
            1: {"b": 1, "step_ms": b1_ms,
                "step_basis": ("geometry_rejected" if unchunkable
                               else "graph_evaluation"),
                "sla": {"limit_ms": float(sla_ms),
                        "pass": bool(b1_ms <= sla_ms and not unchunkable)},
                "capacity": {"b_vram_max": int(b_joint),
                             "current_fits": bool(current_one["fits"]),
                             "lifecycle_fits": bool(lifecycle_one["fits"]),
                             "pass": b1_capacity},
                "feasible": bool(b1_ms <= sla_ms and b1_capacity)}}
        evaluated_b: set[int] = set(candidate_checks)
        eff_max = min(b_max, domain_cap)
        bucket_keys = sorted({int(key) for key in (getattr(parsed.runtime, "graph_capture_keys", ()) or ())}
                             | {int(key) for key in (getattr(parsed.runtime, "piecewise_capture_keys", ()) or ())})

        def candidate_record(b: int, evaluation: Mapping[str, Any]) -> dict[str, Any]:
            sla_pass = float(evaluation["step_ms"]) <= sla_ms
            current_window, lifecycle_window = candidate_windows(int(b))
            capacity_pass = bool(
                b <= b_joint and current_window["fits"]
                and lifecycle_window["fits"] and not unchunkable)
            return {
                "b": int(b), "step_ms": float(evaluation["step_ms"]),
                "step_basis": evaluation.get("step_basis", "graph_evaluation"),
                "input_boundary": {"phase": phase, "tokens": q_tokens, "context": context,
                                   "sequences": int(b), "scope": scope,
                                   "output_budget": output_budget,
                                   "emitted_tokens_per_seq": emit_tokens,
                                   "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
                                   "decode_context_tokens": life_ctx,
                                   "decode_query_tokens": verify_tokens,
                                   "decode_emitted_tokens_per_seq": (
                                       decode_emit_tokens if phase == "prefill" else emit_tokens)},
                "hardware": {"scenario_id": scen.get("id", "custom"),
                             "scales": dict(zip(_SCALE_KEYS, s_key))},
                "plan": {"kind": "target_graphs", "selection": "case_lowering"},
                "sla": {"limit_ms": float(sla_ms), "pass": bool(sla_pass)},
                "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                             "b_vram_lifecycle": int(b_lifecycle), "pass": bool(capacity_pass),
                             "supply_bytes": current_window["supply_bytes"],
                             "active_kv_bytes": current_window["active_kv_bytes"],
                             "retiring_kv_bytes": current_window["retiring_kv_bytes"],
                             "workspace_bytes": current_window["workspace_bytes"],
                             "peak_bytes": current_window["peak_bytes"],
                             "headroom_bytes": current_window["headroom_bytes"],
                             "lifecycle_headroom_bytes": lifecycle_window["headroom_bytes"]},
                "service_lifecycle": {
                    "phase": "decode", "context_tokens": life_ctx,
                    "query_tokens_per_sequence": verify_tokens,
                    "emitted_tokens_per_sequence": (
                        decode_emit_tokens if phase == "prefill" else emit_tokens),
                    "capacity": {
                        "supply_bytes": lifecycle_window["supply_bytes"],
                        "active_kv_bytes": lifecycle_window["active_kv_bytes"],
                        "retiring_kv_bytes": lifecycle_window["retiring_kv_bytes"],
                        "workspace_bytes": lifecycle_window["workspace_bytes"],
                        "peak_bytes": lifecycle_window["peak_bytes"],
                        "headroom_bytes": lifecycle_window["headroom_bytes"],
                        "fits": lifecycle_window["fits"],
                    },
                },
                "feasible": bool(sla_pass and capacity_pass),
                "source": "full_graph_reevaluation",
                "network_mbu": evaluation.get("network_mbu"), "network_cube_mfu": evaluation.get("network_cube_mfu"),
                "network_vec_mfu": evaluation.get("network_vec_mfu"), "network_link_util": evaluation.get("network_link_util"),
                "network_roofline_efficiency": evaluation.get("network_roofline_efficiency"),
                "bottleneck_shares": dict(evaluation.get("bottleneck_shares") or {}),
            }

        bounded_probes = []
        secant_estimate = None
        secant_ok: bool | None = None
        if (0 < base_b_hint <= eff_max and eff_max >= 2 and b1_ms <= sla_ms):
            b2_ms = float(eval_at(s_key, 2, full=False)["step_ms"])
            marginal = b2_ms - b1_ms
            if math.isfinite(marginal) and marginal > SECANT_MIN_SLOPE_MS:
                estimate = 1 + int((sla_ms - b1_ms) / marginal)
                estimate = max(2, min(int(estimate), int(eff_max)))
                if estimate not in evaluated_b:
                    secant_ok = try_step_ok(
                        s_key, estimate, seen=evaluated_b,
                        checks=candidate_checks, capacity_max=b_joint,
                    )
                    secant_estimate = estimate
        if b1_ms > sla_ms:
            probe_ceiling = min(eff_max, NON_MONOTONIC_PROBE_CAP)
            probe_results = {probe: try_step_ok(s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) for probe in range(2, probe_ceiling + 1)}
            bounded_probes = list(probe_results)
            feasible_above = [probe for probe, ok in probe_results.items() if ok]
            if feasible_above:
                b_sla = max(feasible_above)
                eval_sla = eval_at(s_key, b_sla, full=True)
            else:
                b_sla, eval_sla = 0, eval_at(s_key, 1, full=True)
        elif eff_max > 1 and (b1_ms * SLA_HEADROOM_MULTIPLIER <= sla_ms or base_b_hint == eff_max) and try_step_ok(s_key, eff_max, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
            b_sla, eval_sla = eff_max, eval_at(s_key, eff_max, full=True)
        else:
            eff_max = min(b_max, domain_cap)
            low, high = 1, eff_max
            if secant_estimate is not None and secant_ok is not None:
                if secant_ok:
                    low = max(low, secant_estimate)
                else:
                    high = min(high, domain_cap, secant_estimate - 1)
            elif base_b_hint > 1 and all(x >= 1.0 for x in s_key[:-1]):
                hint_probe = min(eff_max, base_b_hint)
                if try_step_ok(s_key, hint_probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                    low = max(low, hint_probe)
                else:
                    high = min(high, domain_cap, hint_probe - 1)
            best_b, lo_i, hi_i = low, low + 1, min(high, domain_cap)
            while lo_i <= hi_i:
                mid = (lo_i + hi_i) // 2
                if try_step_ok(s_key, mid, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                    best_b, lo_i = mid, mid + 1
                else:
                    hi_i = min(hi_i, domain_cap, mid - 1)
            while best_b > 1 and not try_step_ok(s_key, best_b, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint):
                best_b -= 1
            b_sla = best_b if try_step_ok(s_key, best_b, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) else 0
            eval_sla = eval_at(s_key, max(1, b_sla), full=True)

        if b_sla > 0 and b_sla < min(b_max, domain_cap) and bucket_keys:
            ceiling = min(min(b_max, domain_cap), b_sla + SLA_HEADROOM_PROBE_STEPS)
            probe_results = {probe: try_step_ok(s_key, probe, seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint) for probe in range(b_sla + 1, ceiling + 1)}
            bounded_probes.extend(probe_results)
            feasible_above = [probe for probe, ok in probe_results.items() if ok]
            if feasible_above:
                b_sla = max(feasible_above)
                eval_sla = eval_at(s_key, b_sla, full=True)
        # A latency-feasible batch still has to pass both capacity windows.
        # Bound the exact window first (memory-only) and price only its
        # capacity-feasible ceiling, so a near-limit answer does not force one
        # graph evaluation per candidate.
        if b_sla > 0 and b_joint > 0:
            ceiling = min(int(b_sla), int(b_joint), int(b_max), int(domain_cap))
            for probe in range(ceiling, 0, -1):
                current_window, lifecycle_window = candidate_windows(probe)
                if current_window["fits"] and lifecycle_window["fits"]:
                    if try_step_ok(s_key, probe, seen=evaluated_b,
                                   checks=candidate_checks,
                                   capacity_max=b_joint):
                        break
        if b_sla > 0:
            evaluated_b.add(b_sla)
        if scen.get("id") == "baseline" and b_sla > 0:
            base_b_hint = b_sla

        # Probe the capacity ceiling and the latency best as actual candidates even
        # when neither was reached by the binary path; all checks share this set.
        for probe in sorted({b_vram, b_sla}):
            if 0 < probe <= min(b_max, domain_cap) and probe not in candidate_checks:
                step = float(eval_at(s_key, probe, full=False)["step_ms"])
                candidate_checks[probe] = {
                    "b": int(probe), "step_ms": step,
                    "sla": {"limit_ms": float(sla_ms), "pass": step <= sla_ms},
                    "capacity": {"b_vram_max": int(b_joint), "b_vram_current": int(b_vram),
                                 "b_vram_lifecycle": int(b_lifecycle), "pass": probe <= b_joint},
                    "feasible": bool(step <= sla_ms and probe <= b_joint)}
                evaluated_b.add(probe)
        latency_candidate = candidate_record(b_sla, eval_sla) if b_sla > 0 else None
        feasible_check = select_jointly_feasible_candidate(list(candidate_checks.values()))
        b_feas = int(feasible_check["b"]) if feasible_check is not None else 0
        feasible_ev = eval_at(s_key, b_feas, full=True) if b_feas > 0 else None
        feasible_candidate = (candidate_record(b_feas, feasible_ev)
                              if feasible_ev is not None else None)
        if feasible_candidate is not None and not feasible_candidate["feasible"]:
            feasible_candidate = None
            b_feas = 0
        capacity_feasible = b_joint >= 1
        latency_sla_feasible = bool(latency_candidate is not None and latency_candidate["sla"]["pass"])
        jointly_feasible = feasible_candidate is not None
        exhaustive = set(range(1, min(b_max, domain_cap) + 1)) <= set(candidate_checks)
        if domain_cap < 1:
            feasibility_status = "domain_infeasible"
        elif not capacity_feasible:
            feasibility_status = "capacity_infeasible"
        elif jointly_feasible:
            feasibility_status = "feasible" if b_sla <= b_feas else "feasible_capacity_limited"
        elif latency_candidate is not None:
            feasibility_status = ("no_jointly_feasible_candidate_in_evaluated_domain" if exhaustive
                                  else "no_feasible_candidate_in_bounded_search")
        else:
            feasibility_status = ("sla_infeasible_within_evaluated_domain" if exhaustive
                                  else "no_feasible_candidate_in_bounded_search")
        tps_sla = (b_sla * dp * emit_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0
        tps_feas = (b_feas * dp * emit_tokens * 1000.0) / feasible_ev["step_ms"] if jointly_feasible and feasible_ev is not None else 0.0
        step_basis = eval_sla.get("step_basis", "graph_evaluation") if b_sla > 0 else "graph_evaluation_at_b1_reference"
        output_network = feasible_ev if (jointly_feasible and feasible_ev is not None) else eval_sla
        output_network_source = "feasible_candidate" if jointly_feasible else "latency_candidate_not_feasible"
        selected_b = b_feas if b_feas > 0 else (b_sla if b_sla > 0 else 1)
        selected_current_window, selected_lifecycle_window = candidate_windows(selected_b)
        return {
            "scenario_id": scen.get("id", "custom"), "label": scen.get("label", scen.get("id", "custom")), "scales": dict(zip(_SCALE_KEYS, s_key)),
            "b_sla": b_sla, "b_vram_max": b_vram, "b_vram_max_joint": b_joint,
            "b_vram_max_lifecycle_sync": b_life_sync, "b_vram_max_lifecycle_async": b_life_async,
            "b_vram_max_sync": b_vram_sync, "b_vram_max_async": b_vram_async,
            "prefill_tokens_per_seq": q_tokens if phase == "prefill" else None,
            "decode_emitted_tokens_per_seq": decode_emit_tokens if phase == "prefill" else emit_tokens,
            "emitted_tokens_per_seq": emit_tokens,
            "capacity_window": selected_current_window,
            "lifecycle_capacity_window": selected_lifecycle_window,
            "joint_capacity": {
                "sequences": selected_b,
                "current_supply_bytes": selected_current_window["supply_bytes"],
                "current_active_kv_bytes": selected_current_window["active_kv_bytes"],
                "current_retiring_kv_bytes": selected_current_window["retiring_kv_bytes"],
                "current_workspace_bytes": selected_current_window["workspace_bytes"],
                "current_peak_bytes": selected_current_window["peak_bytes"],
                "current_headroom_bytes": selected_current_window["headroom_bytes"],
                "lifecycle_supply_bytes": selected_lifecycle_window["supply_bytes"],
                "lifecycle_active_kv_bytes": selected_lifecycle_window["active_kv_bytes"],
                "lifecycle_retiring_kv_bytes": selected_lifecycle_window["retiring_kv_bytes"],
                "lifecycle_workspace_bytes": selected_lifecycle_window["workspace_bytes"],
                "lifecycle_peak_bytes": selected_lifecycle_window["peak_bytes"],
                "lifecycle_headroom_bytes": selected_lifecycle_window["headroom_bytes"],
                "fits": bool(selected_current_window["fits"] and selected_lifecycle_window["fits"]),
            },
            "delta_b_vram_from_disabling_async": max(0, b_vram_sync - b_vram_async), "hbm_bound": bool(b_joint < b_sla),
            "b_feasible": b_feas, "domain_max_b": domain_cap, "domain_limit_reason": domain_reason,
            "sla_feasible": jointly_feasible, "latency_sla_feasible": latency_sla_feasible,
            "capacity_feasible": capacity_feasible, "jointly_feasible": jointly_feasible, "feasible": jointly_feasible,
            "feasibility_status": feasibility_status,
            "b_sla_lower_bound": b_sla, "b_sla_is_proven_max": False,
            "latency_candidate": latency_candidate, "feasible_candidate": feasible_candidate,
            "search": {"method": "exponential_probe_plus_binary_under_monotone_assumption",
                       "secant_warmup_probe": secant_estimate,
                       "monotonicity_assumed": True, "non_monotonic_dispatch_detected": bool(
                           latency_candidate is not None and latency_candidate["capacity"]["pass"] is False and b1_ms > sla_ms),
                       "non_monotonic_dispatch_check": "bounded_bucket_probe" if bucket_keys else "not_applicable_no_capture_buckets",
                       "bounded_dispatch_probes": bounded_probes, "bounded_search_exhaustive": bool(exhaustive),
                       "evaluated_candidates": sorted(evaluated_b), "b_sla_is_proven_max": False,
                       "candidate_step_source": "full_graph_reevaluation" if jointly_feasible else "none",
                       "lower_bound_can_prove_feasibility": False},
            "total_concurrency": b_sla * dp, "total_feasible_concurrency": b_feas * dp,
            "step_basis": step_basis, "step_is_lower_bound": False,
            "step_ms_at_b1": b1_ms,
            "tps_at_b1": ((dp * emit_tokens * 1000.0) / b1_ms
                          if b1_ms > 0.0 else 0.0),
            "step_ms_at_sla": eval_sla["step_ms"], "tps_at_sla": tps_sla,
            "target_step_tps_at_sla": (b_sla * dp * q_tokens * 1000.0) / eval_sla["step_ms"] if b_sla > 0 else 0.0,
            "step_ms_feasible": feasible_candidate["step_ms"] if jointly_feasible and feasible_candidate is not None else None,
            "tps_feasible": tps_feas,
            "feasible_network_mbu": feasible_candidate["network_mbu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_cube_mfu": feasible_candidate["network_cube_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_vec_mfu": feasible_candidate["network_vec_mfu"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_link_util": feasible_candidate["network_link_util"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_network_roofline_efficiency": feasible_candidate["network_roofline_efficiency"] if jointly_feasible and feasible_candidate is not None else None,
            "feasible_bottleneck_shares": feasible_candidate["bottleneck_shares"] if jointly_feasible and feasible_candidate is not None else None,
            "network_source": output_network_source,
            "network_mbu": output_network["network_mbu"], "network_mbu_at_sla": eval_sla["network_mbu"],
            "network_cube_mfu": output_network["network_cube_mfu"], "network_cube_mfu_at_sla": eval_sla["network_cube_mfu"],
            "network_vec_mfu": output_network["network_vec_mfu"], "network_vec_mfu_at_sla": eval_sla["network_vec_mfu"],
            "network_link_util": output_network["network_link_util"], "network_roofline_efficiency": output_network["network_roofline_efficiency"],
            "bottleneck_shares": output_network["bottleneck_shares"],
        }

    results = [solve_one(scen) for scen in scenario_list]
    base_r = next((row for row in results if row["scenario_id"] == "baseline"), None)
    if base_r is None:
        base_r = next((row for row in results if tuple(row["scales"].values()) == baseline_scales), None)
    if base_r is None:
        base_r = solve_one(DEFAULT_SLA_SCENARIOS[0])
    apply_sla_baseline_gains(results, base_r)
    mem_async_b1 = Memory(replace(parsed, runtime=replace(parsed.runtime, async_scheduling=True)), workspace_tokens=q_tokens)
    prefill = phase == "prefill"
    lifecycle_query = decode_q_tokens if prefill else q_tokens
    lifecycle_emit = decode_emit_tokens if prefill else emit_tokens
    seq_sync_len = max(1, (context + q_tokens) if prefill else (context + max(q_tokens, output_budget)))
    seq_lifecycle_len = max(1, (context + q_tokens + max(lifecycle_query, output_budget)) if prefill else seq_sync_len)
    seq_async_len = max(1, seq_lifecycle_len + mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query)
    if prefill:
        life_ctx = context + q_tokens
        life_sync = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                async_scheduling=False, phase="decode")
        life_async = _vram_max_sequences_per_dp(parsed, context=life_ctx, tokens_per_seq=lifecycle_query,
                                                 output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit,
                                                 async_scheduling=True, phase="decode")
    else:
        life_ctx = context
        life_sync, life_async = base_r["b_vram_max_sync"], base_r["b_vram_max_async"]
    capacity_window_b1 = capacity_window(
        parsed, phase=phase, context=context, tokens_per_seq=q_tokens, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=emit_tokens)
    lifecycle_capacity_window_b1 = capacity_window(
        parsed, phase="decode", context=life_ctx, tokens_per_seq=lifecycle_query, sequences=1,
        output_budget=output_budget, emitted_tokens_per_seq=lifecycle_emit)

    async_mem_summary = {
        "configured_async_scheduling": bool(parsed.runtime.async_scheduling),
        "capacity_window": "prefill_prompt_only" if prefill else "decode_step",
        "lifecycle_window": "prefix_query_plus_output_budget" if prefill else "decode_step",
        "prefill_tokens_per_seq": q_tokens if prefill else None,
        "decode_emitted_tokens_per_seq": lifecycle_emit,
        "b_vram_max_sync": base_r["b_vram_max_sync"],
        "b_vram_max_async": base_r["b_vram_max_async"],
        "b_vram_max_lifecycle_sync": int(life_sync),
        "b_vram_max_lifecycle_async": int(life_async),
        "b_vram_max_joint_sync": int(min(base_r["b_vram_max_sync"], life_sync)),
        "b_vram_max_joint_async": int(min(base_r["b_vram_max_async"], life_async)),
        "delta_b_vram_from_disabling_async": base_r["delta_b_vram_from_disabling_async"],
        "delayed_release_steps_async": mem_async_b1.async_delayed_release_steps,
        "in_flight_lookahead_tokens_async": mem_async_b1.async_in_flight_lookahead_steps * lifecycle_query,
        "kv_bytes_per_seq_sync_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_sync_len, padded=True)],
        "kv_bytes_per_seq_lifecycle_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_lifecycle_len, padded=True)],
        "kv_bytes_per_seq_async_per_stage": [float(value) for value in mem_async_b1.cache_bytes(seq_async_len, padded=True)],
        "async_workspace_overhead_bytes_at_b1_per_stage": [float(value) for value in mem_async_b1.async_workspace],
        "capacity_window_b1": capacity_window_b1,
        "lifecycle_capacity_window_b1": lifecycle_capacity_window_b1,
        "joint_capacity_b1": {
            "supply_bytes": min(capacity_window_b1["supply_bytes"], lifecycle_capacity_window_b1["supply_bytes"]),
            "active_kv_bytes": capacity_window_b1["active_kv_bytes"],
            "retiring_kv_bytes": lifecycle_capacity_window_b1["retiring_kv_bytes"],
            "workspace_bytes": max(capacity_window_b1["workspace_bytes"], lifecycle_capacity_window_b1["workspace_bytes"]),
            "peak_bytes": max(capacity_window_b1["peak_bytes"], lifecycle_capacity_window_b1["peak_bytes"]),
            "headroom_bytes": min(capacity_window_b1["headroom_bytes"], lifecycle_capacity_window_b1["headroom_bytes"]),
            "fits": bool(capacity_window_b1["fits"] and lifecycle_capacity_window_b1["fits"]),
        },
        "hbm_bound_at_baseline": bool(base_r["b_vram_max"] < base_r["b_sla"]),
        "disabling_async_unlocks_concurrency": bool(base_r["b_vram_max_sync"] > base_r["b_vram_max_async"] and base_r["b_vram_max_async"] < base_r["b_sla"]),
    }
    from ..body import describe_included_components
    return {"schema": "mindie-sim.hardware-sla/1", "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope, "timing_boundary": "target_body_only" if scope == "body" else "full_model_step", "included_components": describe_included_components(parsed, scope=scope), "tokens_per_seq": q_tokens, "emitted_tokens_per_seq": emit_tokens, "prefill_tokens_per_seq": q_tokens if prefill else None, "decode_emitted_tokens_per_seq": decode_emit_tokens, "dp": dp, "max_sequences_per_dp": b_max,
            "solver": {"monotonicity_assumed": True, "b_sla_is_proven_max": False,
                       "bounded_non_monotonic_dispatch_probe": True,
                       "status_interpretation": "feasible_candidate_found_or_explicit_sla_capacity_domain_infeasibility"},
            "async_scheduling_memory": async_mem_summary, "scenarios": results}


solve_hardware_sla_throughput = solve_sla_sizing
