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

import heapq
import math
from collections.abc import Mapping, Sequence
from functools import partial
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
    from ..efficiency.unmodeled_gate import assert_timing_supported
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


    run = _SLASizingRun(
        parsed, sla_ms=sla_ms, context=context, phase=phase, scope=scope,
        output_budget=output_budget, q_tokens=q_tokens, emit_tokens=emit_tokens,
        decode_q_tokens=decode_q_tokens, decode_emit_tokens=decode_emit_tokens,
        b_max=b_max, dp=dp, unchunkable=unchunkable, base_b_hint=base_b_hint,
        fast_span_probes=fast_span_probes, evaluation_context=evaluation_context)

    results = [run.solve_one(scen) for scen in scenario_list]
    base_r = next((row for row in results if row["scenario_id"] == "baseline"), None)
    if base_r is None:
        base_r = next((row for row in results if tuple(row["scales"].values()) == run.baseline_scales), None)
    if base_r is None:
        base_r = run.solve_one(DEFAULT_SLA_SCENARIOS[0])
    apply_sla_baseline_gains(results, base_r)
    async_mem_summary = _sla_async_memory(
        parsed, base_r, phase, context, q_tokens, emit_tokens,
        decode_q_tokens, decode_emit_tokens, output_budget)
    prefill = phase == "prefill"
    from ..frontend.body_spec import describe_included_components
    return {"schema": "mindie-sim.hardware-sla/1", "sla_ms": sla_ms, "context": context, "phase": phase, "scope": scope, "timing_boundary": "target_body_only" if scope == "body" else "full_model_step", "included_components": describe_included_components(parsed, scope=scope), "tokens_per_seq": q_tokens, "emitted_tokens_per_seq": emit_tokens, "prefill_tokens_per_seq": q_tokens if prefill else None, "decode_emitted_tokens_per_seq": decode_emit_tokens, "dp": dp, "max_sequences_per_dp": b_max,
            "solver": {"monotonicity_assumed": False, "b_sla_is_proven_max": False,
                       "objective": "maximize_throughput_under_sla",
                       "certified_dual_bound_search": True,
                       "bounded_non_monotonic_dispatch_probe": True,
                       "status_interpretation": "feasible_candidate_found_or_explicit_sla_capacity_domain_infeasibility"},
            "async_scheduling_memory": async_mem_summary, "scenarios": results}


def _sla_async_memory(parsed, base_r, phase, context, q_tokens, emit_tokens,
                      decode_q_tokens, decode_emit_tokens, output_budget):
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

    return {
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


class _SLASizingRun:
    """Own graph and memory reuse, plus the search domain shared by the scenarios."""

    def __init__(
        self, parsed, *, sla_ms, context, phase, scope, output_budget,
        q_tokens, emit_tokens, decode_q_tokens, decode_emit_tokens,
        b_max, dp, unchunkable, base_b_hint, fast_span_probes, evaluation_context,
    ):
        self.parsed = parsed
        self.sla_ms = sla_ms
        self.context = context
        self.phase = phase
        self.scope = scope
        self.output_budget = output_budget
        self.q_tokens = q_tokens
        self.emit_tokens = emit_tokens
        self.decode_q_tokens = decode_q_tokens
        self.decode_emit_tokens = decode_emit_tokens
        self.b_max = b_max
        self.dp = dp
        self.unchunkable = unchunkable
        self.evaluation_context = evaluation_context
        self.scaled_cases: dict[tuple[float, ...], CaseSpec] = {}
        self.graphs_by_b: dict[int, Any] = {}
        self.vram_by_cap: dict[float, tuple[int, int, int]] = {}
        self.graph_cache: dict[tuple[tuple[float, ...], int, bool], tuple[Any, CaseSpec, dict[str, Any]]] = {}
        self.fast_spans = bool(fast_span_probes)
        self.analysis_cache: dict[tuple[tuple[float, ...], int], dict[str, Any]] = {}
        self.baseline_scales = tuple(1.0 for _ in _SCALE_KEYS)
        self.base_b_hint = max(1, int(base_b_hint))
        self.domain_cap = b_max
        self.domain_reason = ("prompt_exceeds_batch_budget_without_chunking"
                         if unchunkable else None)
        self.memory_cache: dict[tuple[int, bool | None, float], Memory] = {}

    def get_scaled(self, s_key: tuple[float, ...]) -> CaseSpec:
        if s_key not in self.scaled_cases:
            self.scaled_cases[s_key] = scale_case_hardware(self.parsed, **dict(zip(_SCALE_KEYS, s_key)))
        return self.scaled_cases[s_key]

    def get_bundle(self, s_key: tuple[float, ...], b: int, *,
                   fast: bool = False) -> tuple[Any, CaseSpec, dict[str, Any]]:
        parsed, phase, q_tokens = self.parsed, self.phase, self.q_tokens
        context, scope = self.context, self.scope
        graphs_by_b, graph_cache = self.graphs_by_b, self.graph_cache
        check_active_deadline()
        if (s_key, b, fast) not in graph_cache:
            if b not in graphs_by_b:
                graphs_by_b[b] = _pkg_attr("target_graphs", target_graphs)(parsed, phase=phase, tokens=q_tokens, context=context, sequences=b, scope=scope)
            context_reuse = self.evaluation_context if s_key == self.baseline_scales else None
            g, ec, gr = _pkg_attr("_lower_and_run_graph", _lower_and_run_graph)(
                self.get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                lowered_graphs=graphs_by_b[b], evaluation_context=context_reuse,
                fast_span_only=bool(fast))
            assert gr is not None
            graph_cache[s_key, b, fast] = (g, ec, gr)
        return graph_cache[s_key, b, fast]

    def get_analysis(self, s_key: tuple[float, ...], b: int) -> dict[str, Any]:
        phase, q_tokens, context, scope = self.phase, self.q_tokens, self.context, self.scope
        analysis_cache = self.analysis_cache
        check_active_deadline()
        if (s_key, b) not in analysis_cache:
            # ``b1_analysis`` is retained only for API compatibility.  It does not
            # carry the case plan/boundary identity needed to prove the same input,
            # so it may never replace an execution evaluated here.  The graph cache
            # above is the only reuse boundary.
            analysis_cache[s_key, b] = _pkg_attr("analyze_network_efficiency", analyze_network_efficiency)(
                self.get_scaled(s_key), phase=phase, tokens=q_tokens, context=context,
                sequences=b, scope=scope, run_graph=True,
                _precomputed_graph=self.get_bundle(s_key, b, fast=False),
                evaluation_context=(self.evaluation_context
                                    if s_key == self.baseline_scales else None))
        return analysis_cache[s_key, b]

    def eval_at(self, s_key: tuple[float, ...], b: int, *, full: bool = False) -> dict[str, Any]:
        if self.unchunkable:
            return {"step_ms": 0.0, "step_basis": "geometry_rejected",
                    "step_is_lower_bound": False, "operator_sum_ms": 0.0,
                    "network_mbu": None,
                    "network_cube_mfu": None, "network_vec_mfu": None,
                    "network_link_util": None,
                    "network_roofline_efficiency": None,
                    "bottleneck_shares": {}}
        if not full:
            return {"step_ms": float(
                self.get_bundle(s_key, b, fast=self.fast_spans)[2]["duration_ms"])}
        ans = self.get_analysis(s_key, b)
        shares = {k: ans["breakdown_by_bound"][k]["share_of_operator_sum"] for k in _BOUNDS}
        return {"step_ms": float(ans["graph_duration_ms"]), "step_basis": "graph_evaluation", "step_is_lower_bound": False, "operator_sum_ms": float(ans["operator_sum_ms"]), "network_mbu": float(ans["network_mbu"]), "network_cube_mfu": float(ans["network_cube_mfu"]), "network_vec_mfu": float(ans["network_vec_mfu"]), "network_link_util": float(ans["network_link_util"]), "network_roofline_efficiency": float(ans["network_roofline_efficiency"]), "bottleneck_shares": shares}

    def try_step_ok(
        self, s_key: tuple[float, ...], b: int, *, window_for,
        seen: set[int] | None = None, checks: dict[int, dict[str, Any]] | None = None,
        capacity_max: int | None = None,
    ) -> bool:
        sla_ms = self.sla_ms
        if seen is not None:
            seen.add(int(b))
        if b > self.domain_cap:
            return False
        try:
            step = float(self.eval_at(s_key, b, full=False)["step_ms"])
            capacity_pass = True
            if checks is not None:
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
        except ModelingError:
            raise

    def memory_for(self, tokens: int, async_scheduling: bool | None, scale: float) -> Memory:
        parsed, memory_cache = self.parsed, self.memory_cache
        key = (int(tokens), async_scheduling, float(scale))
        if key not in memory_cache:
            base_case = (scale_case_hardware(parsed, hbm_capacity_scale=scale)
                         if scale != 1.0 else parsed)
            if async_scheduling is not None:
                base_case = replace(base_case, runtime=replace(
                    base_case.runtime, async_scheduling=bool(async_scheduling)))
            memory_cache[key] = Memory(base_case, workspace_tokens=int(tokens))
        return memory_cache[key]

    def _scenario_capacity(self, hbm_capacity_scale):
        parsed, phase, context = self.parsed, self.phase, self.context
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        decode_q_tokens, decode_emit_tokens = self.decode_q_tokens, self.decode_emit_tokens
        output_budget, unchunkable = self.output_budget, self.unchunkable
        vram_by_cap, memory_for = self.vram_by_cap, self.memory_for
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
        return (
            (b_vram, b_vram_sync, b_vram_async),
            (b_lifecycle, b_life_sync, b_life_async),
            life_ctx, verify_tokens, candidate_windows,
        )

    def solve_one(self, scen: Mapping[str, Any]) -> dict[str, Any]:
        check_active_deadline()
        s_key = tuple(float(scen.get(k, 1.0)) for k in _SCALE_KEYS)
        hbm_capacity_scale = s_key[-1]
        current_caps, lifecycle_caps, life_ctx, verify_tokens, candidate_windows = (
            self._scenario_capacity(hbm_capacity_scale))
        b_joint = min(current_caps[0], lifecycle_caps[0])
        unchunkable, sla_ms = self.unchunkable, self.sla_ms
        parsed = self.parsed
        b1_ms = (0.0 if unchunkable else
                 float(self.get_bundle(s_key, 1, fast=self.fast_spans)[2]["duration_ms"]))
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
        eff_max = min(self.b_max, self.domain_cap)
        bucket_keys = sorted({int(key) for key in (getattr(parsed.runtime, "graph_capture_keys", ()) or ())}
                             | {int(key) for key in (getattr(parsed.runtime, "piecewise_capture_keys", ()) or ())})

        (
            b_sla, eval_sla, bounded_probes, secant_estimate,
            certified_optimal, hi_cut,
        ) = self._search_concurrency(
            s_key, b1_ms, eff_max, bucket_keys, candidate_checks, evaluated_b,
            b_joint, candidate_windows)
        if scen.get("id") == "baseline" and b_sla > 0:
            self.base_b_hint = b_sla
        return self._scenario_result(
            scen, s_key, current_caps, lifecycle_caps, b_joint,
            life_ctx, verify_tokens, candidate_windows, b1_ms, bucket_keys,
            candidate_checks, evaluated_b, b_sla, eval_sla, bounded_probes,
            secant_estimate, certified_optimal=certified_optimal, hi_cut=hi_cut)

    def _work_floor_ms(
        self, s_key: tuple[float, ...], b: int, step_ms: float,
    ) -> float:
        """Compute monotone physical lower bound on ``step_ms(b')`` for all ``b' >= b``."""
        if self.unchunkable or step_ms <= 0.0:
            return 0.0
        cached = self.graph_cache.get((s_key, b, self.fast_spans))
        if cached is None:
            cached = self.graph_cache.get((s_key, b, False))
        if cached is None:
            return 0.0
        graphs, eff_case, _gr = cached
        ops_0 = graphs.get(0, ()) if isinstance(graphs, Mapping) else ()
        if not ops_0:
            return 0.0
        dies = getattr(getattr(eff_case, "machine", None), "dies", ()) or ()
        if not dies:
            return 0.0
        max_compute_ms = 0.0
        max_hbm_ms = 0.0
        max_l2_bytes = 0.0
        for die in dies:
            rates = dict(getattr(die, "rates", ()) or ())
            cube_s = max(
                (float(v) for v in dict(getattr(die, "compute_peaks", ())).values()),
                default=0.0,
            )
            vec_s = float(rates.get("vector", 0.0) or 0.0)
            simt_s = float(rates.get("simt", 0.0) or 0.0)
            max_compute_ms = max(max_compute_ms, (cube_s + vec_s + simt_s) / 1000.0)
            hbm_s = float(rates.get("hbm", 0.0) or 0.0)
            dp = getattr(die, "datapath", None)
            hbm_r = float(getattr(dp, "hbm_read_bytes_per_s", 0.0) or 0.0) if dp is not None else 0.0
            hbm_w = float(getattr(dp, "hbm_write_bytes_per_s", 0.0) or 0.0) if dp is not None else 0.0
            max_hbm_ms = max(max_hbm_ms, max(hbm_s, hbm_r, hbm_w) / 1000.0)
            l2_b = float(dict(getattr(die, "buffers", ()) or ()).get("l2", 0.0) or 0.0)
            max_l2_bytes = max(max_l2_bytes, l2_b)
        stage_flops: dict[int, float] = {}
        stage_weight_bytes: dict[int, float] = {}
        for op in ops_0:
            st = int(getattr(op, "stage", 0) or 0)
            fl = float(getattr(op, "flops", 0.0) or 0.0)
            if fl > 0.0:
                stage_flops[st] = stage_flops.get(st, 0.0) + fl
            op_hbm = float(getattr(op, "hbm_bytes", 0.0) or 0.0)
            if op_hbm > 0.0:
                md = dict(getattr(op, "metadata", ()) or ())
                wb = float(md.get("packed_weight_bytes") or md.get("weight_bytes") or 0.0)
                if wb > 0.0:
                    stage_weight_bytes[st] = stage_weight_bytes.get(st, 0.0) + min(wb, op_hbm)
        flop_floor = (
            max((fl / max_compute_ms for fl in stage_flops.values()), default=0.0)
            if max_compute_ms > 0.0 else 0.0
        )
        weight_floor = (
            max((max(0.0, wb - max_l2_bytes) / max_hbm_ms for wb in stage_weight_bytes.values()), default=0.0)
            if max_hbm_ms > 0.0 else 0.0
        )
        floor = max(flop_floor, weight_floor)
        if not math.isfinite(floor) or floor <= 0.0 or floor > step_ms * (1.0 + 1e-6):
            return 0.0
        return min(floor, step_ms)

    def _certified_upper_cutoff(
        self, eval_b, floors: dict[int, float], eff_max: int,
        b_joint: int, bucket_keys: Sequence[int],
    ) -> int:
        """Theorem 1: find largest ``b <= eff_max`` not certified SLA-infeasible by ``work_floor_ms``."""
        sla_ms = float(self.sla_ms)
        q_tokens = max(1, int(self.q_tokens))
        hi_cut = int(eff_max)
        if floors[1] > sla_ms:
            if eff_max >= 2:
                eval_b(2)
            return 0
        prev_p = 1
        probe = 2
        while probe <= hi_cut:
            _, fl_p = eval_b(probe)
            if fl_p > sla_ms:
                lo_c, hi_c = prev_p, probe
                while hi_c - lo_c > 1:
                    mid_c = (lo_c + hi_c) // 2
                    _, fl_m = eval_b(mid_c)
                    if fl_m > sla_ms:
                        hi_c = mid_c
                    else:
                        lo_c = mid_c
                hi_cut = lo_c
                break
            prev_p = probe
            probe *= 2
        if hi_cut >= 1 and hi_cut not in floors:
            _, fl_hi = eval_b(hi_cut)
            if fl_hi > sla_ms:
                hi_cut = max(0, hi_cut - 1)
        anchors: set[int] = set()
        if hi_cut >= 2:
            anchors.add(2)
            for cand in (b_joint - 1, b_joint, b_joint + 1, self.base_b_hint):
                if 2 <= cand <= hi_cut:
                    anchors.add(int(cand))
            for bk in bucket_keys:
                for raw_b in (bk, bk // q_tokens, bk // q_tokens - 1, bk // q_tokens + 1):
                    if 2 <= raw_b <= hi_cut:
                        anchors.add(int(raw_b))
        for anchor in sorted(anchors):
            if anchor > hi_cut:
                break
            _, fl_a = eval_b(anchor)
            if fl_a > sla_ms:
                hi_cut = min(hi_cut, anchor - 1)
                break
        return hi_cut

    def _search_concurrency(
        self, s_key, b1_ms, eff_max, bucket_keys, candidate_checks, evaluated_b,
        b_joint, candidate_windows,
    ):
        """Dual-certificate branch-and-bound SLA sizing maximizing throughput ``B / step_ms``."""
        if self.unchunkable or eff_max < 1:
            return 0, self.eval_at(s_key, 1, full=True), [], None, True, 0

        sla_ms = float(self.sla_ms)
        bounded_probes: list[int] = []
        probed_seen: set[int] = set()
        floors: dict[int, float] = {}
        best_sla: tuple[float, int] | None = None
        best_joint: tuple[float, int] | None = None

        def _tps(b_val: int, step_val: float) -> float:
            return (float(b_val) / step_val) if step_val > 0.0 else math.inf

        def _record_best(b_val: int) -> None:
            nonlocal best_sla, best_joint
            rec = candidate_checks.get(b_val)
            if rec is None:
                return
            score = (_tps(b_val, float(rec["step_ms"])), b_val)
            if bool(rec["sla"]["pass"]) and (best_sla is None or score > best_sla):
                best_sla = score
            if bool(rec["feasible"]) and (best_joint is None or score > best_joint):
                best_joint = score

        def _eval_b(b_val: int) -> tuple[float, float]:
            check_active_deadline()
            b_int = int(b_val)
            if b_int not in candidate_checks:
                self.try_step_ok(
                    s_key, b_int, window_for=candidate_windows,
                    seen=evaluated_b, checks=candidate_checks, capacity_max=b_joint,
                )
            if b_int > 1 and b_int not in probed_seen:
                probed_seen.add(b_int)
                bounded_probes.append(b_int)
            step_val = float(candidate_checks[b_int]["step_ms"])
            if b_int not in floors:
                floors[b_int] = self._work_floor_ms(s_key, b_int, step_val)
            _record_best(b_int)
            return step_val, floors[b_int]

        floors[1] = self._work_floor_ms(s_key, 1, float(b1_ms))
        _record_best(1)
        hi_cut = self._certified_upper_cutoff(_eval_b, floors, eff_max, b_joint, bucket_keys)

        def _interval_ub(lo_i: int, hi_i: int) -> float:
            fl_lo = floors.get(lo_i, 0.0)
            return math.inf if fl_lo <= 0.0 else float(hi_i - 1) / fl_lo

        def _is_dominated(lo_i: int, hi_i: int) -> bool:
            if hi_i - lo_i <= 1:
                return True
            fl_lo = floors.get(lo_i, 0.0)
            if fl_lo > sla_ms:
                return True
            if fl_lo <= 0.0:
                return False
            ub_all = float(hi_i - 1) / fl_lo
            sla_dom = best_sla is not None and (
                ub_all < best_sla[0]
                or (ub_all <= best_sla[0] and (hi_i - 1) <= best_sla[1])
            )
            if not sla_dom:
                return False
            if b_joint < 1 or lo_i >= b_joint:
                return True
            joint_hi = min(hi_i, b_joint + 1)
            if joint_hi - lo_i <= 1:
                return True
            ub_joint = float(joint_hi - 1) / fl_lo
            return best_joint is not None and (
                ub_joint < best_joint[0]
                or (ub_joint <= best_joint[0] and (joint_hi - 1) <= best_joint[1])
            )

        heap: list[tuple[float, int, int, int]] = []
        if hi_cut >= 2:
            if hi_cut not in candidate_checks:
                _eval_b(hi_cut)
            pts = sorted(b_i for b_i in candidate_checks if 1 <= b_i <= hi_cut)
            for idx in range(len(pts) - 1):
                lo_i, hi_i = pts[idx], pts[idx + 1]
                if not _is_dominated(lo_i, hi_i):
                    heapq.heappush(heap, (-_interval_ub(lo_i, hi_i), -(hi_i - lo_i), lo_i, hi_i))

        while heap:
            _neg_ub, _neg_span, lo_i, hi_i = heapq.heappop(heap)
            hi_eff = min(hi_i, hi_cut + 1)
            if lo_i >= hi_cut or _is_dominated(lo_i, hi_eff):
                continue
            mid = (lo_i + hi_eff) // 2
            _, fl_m = _eval_b(mid)
            if fl_m > sla_ms:
                hi_cut = min(hi_cut, mid - 1)
                if not _is_dominated(lo_i, mid):
                    heapq.heappush(heap, (-_interval_ub(lo_i, mid), -(mid - lo_i), lo_i, mid))
                continue
            if not _is_dominated(lo_i, mid):
                heapq.heappush(heap, (-_interval_ub(lo_i, mid), -(mid - lo_i), lo_i, mid))
            if not _is_dominated(mid, hi_eff):
                heapq.heappush(heap, (-_interval_ub(mid, hi_eff), -(hi_eff - mid), mid, hi_eff))

        b_sla = int(best_sla[1]) if best_sla is not None else 0
        eval_sla = self.eval_at(s_key, b_sla if b_sla > 0 else 1, full=True)
        return b_sla, eval_sla, bounded_probes, None, True, hi_cut

    def _candidate_record(
        self, b, evaluation, *, scen, s_key, b_joint, b_vram, b_lifecycle,
        life_ctx, verify_tokens, candidate_windows,
    ):
        phase, context, scope = self.phase, self.context, self.scope
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        output_budget, decode_emit_tokens = self.output_budget, self.decode_emit_tokens
        sla_ms, unchunkable = self.sla_ms, self.unchunkable
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

    def _scenario_result(
        self, scen, s_key, current_caps, lifecycle_caps, b_joint,
        life_ctx, verify_tokens, candidate_windows, b1_ms, bucket_keys,
        candidate_checks, evaluated_b, b_sla, eval_sla, bounded_probes,
        secant_estimate, *, certified_optimal: bool = False, hi_cut: int | None = None,
    ):
        b_vram, b_vram_sync, b_vram_async = current_caps
        b_lifecycle, b_life_sync, b_life_async = lifecycle_caps
        b_max, domain_cap, domain_reason = self.b_max, self.domain_cap, self.domain_reason
        sla_ms, dp, phase = self.sla_ms, self.dp, self.phase
        q_tokens, emit_tokens = self.q_tokens, self.emit_tokens
        decode_emit_tokens = self.decode_emit_tokens
        eval_at = self.eval_at
        candidate_record = partial(
            self._candidate_record, scen=scen, s_key=s_key, b_joint=b_joint,
            b_vram=b_vram, b_lifecycle=b_lifecycle, life_ctx=life_ctx,
            verify_tokens=verify_tokens, candidate_windows=candidate_windows)
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
        domain_certified = bool(exhaustive or certified_optimal)
        if domain_cap < 1:
            feasibility_status = "domain_infeasible"
        elif not capacity_feasible:
            feasibility_status = "capacity_infeasible"
        elif jointly_feasible:
            feasibility_status = "feasible" if b_sla <= b_feas else "feasible_capacity_limited"
        elif latency_candidate is not None:
            feasibility_status = ("no_jointly_feasible_candidate_in_evaluated_domain" if domain_certified
                                  else "no_feasible_candidate_in_bounded_search")
        else:
            feasibility_status = ("sla_infeasible_within_evaluated_domain" if domain_certified
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
            "search": {"method": "dual_certificate_branch_and_bound",
                       "objective": "maximize_throughput_under_sla",
                       "secant_warmup_probe": secant_estimate,
                       "monotonicity_assumed": False,
                       "certified_optimal": bool(certified_optimal),
                       "certified_upper_cutoff_b": int(hi_cut if hi_cut is not None else min(b_max, domain_cap)),
                       "non_monotonic_dispatch_detected": bool(
                           b_sla > 1 and b1_ms > sla_ms),
                       "non_monotonic_dispatch_check": "dual_certificate_branch_and_bound",
                       "bounded_dispatch_probes": bounded_probes, "bounded_search_exhaustive": bool(domain_certified),
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


solve_hardware_sla_throughput = solve_sla_sizing
