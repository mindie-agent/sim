"""Fixed-plan resource attribution and counterfactual evaluation on an A5 graph.

[Role] Establish the T17 first slice: one lowered plan evaluated through the
  common ordinary engine for baseline and one independent resource/capacity
  intervention, with work/busy/wait/span kept separate.
[Invariants] Nominal MFU/MBU use the graph span, effective service utilization
  uses engine busy time, and neither is inferred by splitting total duration.
  Resource changes re-evaluate the same program; no isolated-sum graph scaling.
[Timeouts & Locks] In-memory composition and bounded graph evaluation only.
[Side-Effect Policy] No files, devices, network, subprocesses or caller mutation.
"""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import pytest

from mindie_sim.compose import compose_case
from mindie_sim.prediction import efficiency as efficiency_module
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _decompose_operator_pipes,
    _vram_max_sequences_per_dp,
    analyze_operator_row,
    capacity_feedback,
    capacity_window,
    decode_query_tokens,
    diagnostic_legacy_invocation_bounds,
    legacy_service_regime_diagnostics,
    analyze_network_efficiency,
    attribute_program,
    capacity_feedback,
    evaluate_fixed_plan,
    evaluate_resource_counterfactual,
    evaluate_target_efficiency,
    extract_hardware_peaks,
    fit_case_roofline_regimes,
    prepare_fixed_plan,
    scale_case_hardware,
    select_jointly_feasible_candidate,
    solve_sla_sizing,
    stage_hbm_bytes,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import OpSpec
from mindie_sim.study import _normalize_ranking_row, _rankings, study
from mindie_sim.sweep import sweep_topologies


def _a5_case(**overrides) -> dict:
    defaults = {"layers": 2, "prompt_tokens": 2048, "output_tokens": 64, "concurrency": 2}
    defaults.update(overrides)
    return compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, dp=1, ep=2, **defaults)


def _capacity_fixture(monkeypatch, *, async_scheduling: bool = True,
                      graph_pool_kib: int = 200, graph_capture: bool = False,
                      budget_kib: int = 2700):
    """One rank/stage paying 1 KiB per KV row in 128-row pages.

    The real T08 ``Memory`` supplies weights, workspace, reserves and the
    admission plan; only the KV page geometry is synthetic so the arithmetic is
    exact and does not depend on a model's layer geometry.
    """
    model_path = Path(__file__).resolve().parents[1] / "mindie_sim" / "model_configs" / "deepseek-v2-lite.json"
    config = json.loads(model_path.read_text())
    config.update({
        "num_hidden_layers": 1, "vocab_size": 1024, "hidden_size": 128,
        "intermediate_size": 256, "moe_intermediate_size": 64,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "kv_lora_rank": 64, "qk_nope_head_dim": 64, "qk_rope_head_dim": 32,
        "v_head_dim": 64,
    })
    raw = compose_case(
        config, hardware="a3", tp=1, dp=1, ep=1, layers=1,
        prompt_tokens=512, output_tokens=128, concurrency=2, block_size=128,
        max_num_seqs=4, max_num_batched_tokens=4096, max_model_len=4096,
        async_scheduling=async_scheduling,
        graph_capture_mode="full" if graph_capture else "none")
    raw["runtime"]["gpu_memory_utilization"] = 1.0
    raw["runtime"]["memory"] = {
        "derive_workspace": False, "workspace_bytes": 0.0,
        "graph_pool_bytes": float(graph_pool_kib * 1024),
        "communication_buffer_bytes": 0.0, "runtime_bytes": 0.0,
        "async_delayed_release_steps": 1 if async_scheduling else 0,
        "async_in_flight_lookahead_steps": 1 if async_scheduling else 0,
        "async_workspace_buffer_multiplier": 0.0,
        "temporary_dequant_bytes": 0.0,
    }
    parsed = parse_case(raw)
    probe = Memory(parsed, workspace_tokens=2)
    fixed = (probe.weights[0] + probe.workspace[0]
             + probe.temporary_dequant[0] + probe.policy.reserved)
    raw["machine"]["device"]["memory_bytes"] = int(fixed + budget_kib * 1024)
    parsed = parse_case(raw)

    def page_bytes(self, tokens, *, padded=True):
        rows = (math.ceil(max(1, int(tokens)) / 128) * 128
                if padded else max(1, int(tokens)))
        return tuple([float(rows * 1024)]
                     + [0.0] * (self.case.runtime.parallel.pp - 1))

    monkeypatch.setattr(Memory, "cache_bytes", page_bytes)
    return parsed


def _patch_bounded_solver(monkeypatch, latency_ms):
    """CPU control: keep the public solver/capacity path, replace only graph latency."""
    monkeypatch.setattr(efficiency_module, "target_graphs", lambda *args, **kwargs: {0: ()})
    monkeypatch.setattr(
        efficiency_module, "_lower_and_run_graph",
        lambda case, *, phase, tokens, context, sequences, scope, **kwargs: (
            {0: ()}, case, {"duration_ms": float(latency_ms(int(sequences)))}))
    def fake_analysis(case, *, phase, tokens, context, sequences, scope, **kwargs):
        step = float(latency_ms(int(sequences)))
        return {
            "graph_duration_ms": step, "operator_sum_ms": step,
            "breakdown_by_bound": {bound: {"share_of_operator_sum": 0.0}
                                   for bound in efficiency_module._BOUNDS},
            "network_mbu": 0.1, "network_cube_mfu": 0.1, "network_vec_mfu": 0.1,
            "network_link_util": 0.1, "network_roofline_efficiency": 0.1,
        }
    monkeypatch.setattr(efficiency_module, "analyze_network_efficiency", fake_analysis)


def test_a5_fixed_plan_baseline_matches_ordinary_analysis_and_separates_phases():
    raw = _a5_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    fixed = evaluate_fixed_plan(raw, plan=plan)
    direct = analyze_network_efficiency(raw, **geometry)

    assert fixed["graph_duration_ms"] == pytest.approx(direct["graph_duration_ms"], rel=1e-12)
    assert fixed["operator_sum_ms"] == pytest.approx(direct["operator_sum_ms"], rel=1e-12)
    assert fixed["total_hbm_bytes"] == pytest.approx(direct["total_hbm_bytes"], rel=1e-12)
    assert fixed["plan"]["mode"] == "fixed"

    attribution = fixed["attribution"]
    assert attribution["timeline_available"] is True
    assert attribution["work_ms"] == pytest.approx(fixed["operator_sum_ms"], rel=1e-12)
    assert attribution["span_ms"] == pytest.approx(fixed["graph_duration_ms"], rel=1e-12)
    assert attribution["active_span_ms"] is not None and attribution["active_span_ms"] >= 0.0
    assert attribution["busy_ms"] is None
    assert attribution["wait_ms"] is None
    assert attribution["service_busy_union_ms"] is None
    assert attribution["busy_basis"] == "no_whole_graph_service_interval_union_available_per_resource_busy_kept"
    assert attribution["wait_basis"] == "service_start_end_or_pmu_required_not_inferable_from_ready_complete"
    assert attribution["resource_service"]
    assert {"hbm", "cube", "vector"}.issubset(attribution["service_utilization_by_role"])

    critical = attribution["critical_path"]
    assert critical is not None
    assert 0.0 <= critical["work_path_ms"] <= attribution["span_ms"] + 1e-9
    assert critical["basis"].startswith(("operator_service", "isolated_costs"))

    for nominal, network in (("cube_mfu", "network_cube_mfu"), ("vec_mfu", "network_vec_mfu"),
                             ("mbu", "network_mbu"), ("link_util", "network_link_util")):
        assert fixed["nominal"][nominal] == pytest.approx(fixed[network], rel=1e-12)
    assert fixed["nominal"]["denominator"] == "graph_span_ms"
    assert fixed["effective_service_utilization"] == attribution["service_utilization_by_role"]
    assert all(0.0 <= value <= 1.0 + 1e-12 for value in fixed["effective_service_utilization"].values())


def test_a5_single_hbm_intervention_reevaluates_same_program_without_work_rescaling():
    raw = _a5_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    baseline = evaluate_fixed_plan(raw, plan=plan)
    result = evaluate_resource_counterfactual(
        raw, resource="hbm_bw", factor=2.0, plan=plan, include_capacity=False)

    assert result["schema"] == "mindie-sim.resource-counterfactual/1"
    assert result["plan"]["mode"] == "fixed"
    assert result["intervention"]["kind"] == "declared_hardware_peak_or_capacity_multiplier"
    assert "T06" in result["intervention"]["calibrated_service_boundary"]
    assert result["baseline"]["graph_duration_ms"] == pytest.approx(
        baseline["graph_duration_ms"], rel=1e-12)
    assert result["counterfactual"]["graph_duration_ms"] < result["baseline"]["graph_duration_ms"]
    assert result["work_conservation"]["conserved"] is True
    for field in ("total_flops", "total_vector_ops", "total_hbm_bytes", "total_link_bytes"):
        assert result["work_conservation"]["fields"][field]["delta_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert result["changed_operators"]
    assert result["changed_operators"][0]["delta_ms"] < 0.0
    assert result["counterfactual"]["attribution"]["work_ms"] < result["baseline"]["attribution"]["work_ms"]
    # The retained plan's declared work is conserved (asserted above).  The
    # attribution's per-row physical_work follows the rate-bound physical
    # mapping, which may re-tile while the declared plan stays fixed.
    with pytest.raises(ValueError, match="t06_normalized_service_api"):
        evaluate_resource_counterfactual(
            raw, resource="effective_rate", factor=1.5, plan=plan, include_capacity=False)
    for resource in ("cube_flops", "vector_ops", "link_bw"):
        other = evaluate_resource_counterfactual(
            raw, resource=resource, factor=2.0, plan=plan, include_capacity=False)
        assert other["work_conservation"]["conserved"] is True


def test_a5_capacity_intervention_changes_feedback_not_program_step():
    raw = _a5_case(max_num_seqs=8192, max_num_batched_tokens=8192, max_model_len=16384)
    raw["machine"]["device"]["memory_bytes"] = 12 * 1024 ** 3
    geometry = {"phase": "decode", "tokens": 1, "context": 8192, "sequences": 1, "scope": "body"}
    result = evaluate_resource_counterfactual(
        raw, resource="hbm_capacity", factor=2.0, include_capacity=True, **geometry)

    assert result["baseline"]["graph_duration_ms"] == pytest.approx(
        result["counterfactual"]["graph_duration_ms"], rel=1e-12)
    assert result["work_conservation"]["conserved"] is True
    base_capacity = result["capacity_feedback"]["baseline"]
    changed_capacity = result["capacity_feedback"]["counterfactual"]
    assert changed_capacity["available_kv_capacity_bytes_per_die"] > base_capacity["available_kv_capacity_bytes_per_die"]
    assert changed_capacity["b_vram_max_sync"] > base_capacity["b_vram_max_sync"]


def test_target_efficiency_is_a_pipe_bound_not_isolated_sum_graph_scaling():
    case = parse_case(compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, dp=1, ep=2,
                                   layers=1, prompt_tokens=1024, output_tokens=32, concurrency=1))
    peaks = extract_hardware_peaks(case)
    hbm, bf16 = peaks["hbm_peak_bytes_per_s"], peaks["compute_peaks"]["bf16"]
    operators = {
        "operators": [
            {"physical_id": "mem", "occurrences": 1, "program": "ResourceEnvelope",
             "duration_ms": 1.0, "total_ms": 1.0,
             "operator": {"op_id": "l0.mem", "family": "custom", "recipe": "parallel",
                          "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16",
                          "local_shape": [1, 4096], "flops": 0.0, "vector_ops": 0.0,
                          "hbm_bytes": 0.8e-3 * hbm, "link_bytes": 0.0}},
            {"physical_id": "cube", "occurrences": 1, "program": "SteadyMatmul",
             "duration_ms": 1.0, "total_ms": 1.0,
             "operator": {"op_id": "l0.cube", "family": "custom", "recipe": "parallel",
                          "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16",
                          "local_shape": [16, 4096], "flops": 0.7e-3 * bf16,
                          "vector_ops": 0.0, "hbm_bytes": 0.0, "link_bytes": 0.0}},
        ]
    }
    base = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=1024, sequences=1,
        operator_report=operators, graph_report={"duration_ms": 1.0})
    evaluated = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=1024, sequences=1,
        target_mbu=0.80, baseline_analysis=base)

    expected_bound = min(evaluated["current_step_duration_ms"],
                         max(evaluated["target_pipe_totals_ms"].values()))
    assert evaluated["target_step_basis"] == "dag_critical_path_and_resource_conservation"
    assert evaluated["target_step_is_lower_bound"] is True
    assert evaluated["target_step_duration_ms"] == pytest.approx(expected_bound, rel=1e-12)
    isolated_sum_scaling = (evaluated["current_step_duration_ms"]
                            * evaluated["target_operator_sum_ms"]
                            / evaluated["current_operator_sum_ms"])
    assert evaluated["target_step_duration_ms"] != pytest.approx(isolated_sum_scaling, rel=1e-9)


def test_roofline_fit_exports_nominal_bounds_without_identifying_pipe():
    raw = _a5_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    baseline = evaluate_fixed_plan(raw, plan=plan)
    fitted = fit_case_roofline_regimes(raw, **geometry, baseline_analysis=baseline)

    assert fitted["identification_scope"] == "total_duration_exports_nominal_work_bounds_but_does_not_identify_pipe_bottleneck"
    for op in fitted["operators"]:
        assert op["bottleneck_pipe"] is None
        assert op["bottleneck_identified"] is False
        assert op["identification"] == "total_duration_does_not_identify_pipe_bottleneck"
        assert all(op[field] is None for field in (
            "effective_mbu", "effective_cube_mfu", "effective_vec_mfu", "effective_link_util"))
        for bound in op["nominal_work_lower_bound_utilization"].values():
            assert bound is None or bound >= 0.0

    # Exact review counterexample: HBM 3ms, Cube 2ms, total 10ms. Both physical
    # explanations (HBM 10ms/Cube 2ms, or HBM 3ms/Cube 10ms) fit one duration.
    decomposed = _decompose_operator_pipes({
        "duration_ms": 10.0, "fixed_ms": 0.0, "minimum_ms": 0.0,
        "t_hbm_ideal_ms": 3.0, "t_cube_ideal_ms": 2.0,
        "t_vec_ideal_ms": 0.0, "t_link_ideal_ms": 0.0,
    })
    assert decomposed["bottleneck_pipe"] is None
    assert decomposed["bottleneck_identified"] is False
    assert decomposed["candidate_pipes"] == ["cube", "hbm"]
    assert decomposed["nominal_work_lower_bound_utilization"]["hbm"] == pytest.approx(0.3, rel=1e-12)
    assert decomposed["nominal_work_lower_bound_utilization"]["cube"] == pytest.approx(0.2, rel=1e-12)
    assert all(decomposed[field] is None for field in (
        "effective_mbu", "effective_cube_mfu", "effective_vec_mfu", "effective_link_util"))




def test_study_and_sweep_label_reselected_plans_and_keep_attribution():
    study_case = compose_case("qwen3.5-35b-a3b", hardware="a3", tp=2, dp=1, ep=2,
                              layers=1, prompt_tokens=1024, output_tokens=16, concurrency=1)
    studied = study(
        [{"id": "a3", "case": study_case}],
        [{"id": "decode", "phase": "decode", "tokens": 1, "context": 1024,
          "sequences": 1, "scope": "body"}],
        level="graph", top_families=3)
    assert studied["evaluation_boundary"]["plan_selection"] == "reselected_case_per_candidate"
    graph = studied["evaluations"][0]["graph"]
    assert graph["plan_mode"] == "reselected_case"
    assert graph["attribution"]["span_ms"] == pytest.approx(graph["duration_ms"], rel=1e-12)
    assert graph["attribution"]["work_ms"] is not None

    swept = sweep_topologies(
        _a5_case(), candidates=[{"dp": 1, "tp": 2, "ep": 2, "pp": 1, "pcp": 1}],
        num_devices=2, sla_ms=50.0, context=2048, phase="decode", scope="body",
        output_budget=64, max_sequences_per_dp=2)
    assert swept["plan_selection"] == "reselected_parallel_per_candidate"
    assert swept["candidates"][0]["plan_mode"] == "reselected_parallel"
    assert swept["candidates"][0]["feasibility_status"] in {
        "feasible", "feasible_capacity_limited",
        "sla_infeasible_within_evaluated_domain",
        "no_jointly_feasible_candidate_in_evaluated_domain",
        "no_feasible_candidate_in_bounded_search",
        "capacity_infeasible", "domain_infeasible"}
    assert swept["candidates"][0]["search"]["b_sla_is_proven_max"] is False


def test_sla_solver_reports_found_candidate_without_claiming_global_proof():
    report = solve_sla_sizing(
        _a5_case(), sla_ms=50.0, context=2048, max_sequences_per_dp=2,
        scenarios=[{"id": "baseline", "hbm_bw_scale": 1.0}])
    assert report["solver"]["b_sla_is_proven_max"] is False
    assert report["solver"]["bounded_non_monotonic_dispatch_probe"] is True
    scenario = report["scenarios"][0]
    assert scenario["feasibility_status"] in {
        "feasible", "feasible_capacity_limited",
        "sla_infeasible_within_evaluated_domain",
        "no_jointly_feasible_candidate_in_evaluated_domain",
        "no_feasible_candidate_in_bounded_search",
        "capacity_infeasible", "domain_infeasible"}
    assert scenario["b_sla_is_proven_max"] is False
    assert scenario["search"]["non_monotonic_dispatch_check"] in {
        "bounded_bucket_probe", "not_applicable_no_capture_buckets"}


def _minimal_copy_case() -> dict:
    return {
        "machine": {
            "name": "declared-gpu", "architecture": "declared", "num_devices": 1,
            "device": {
                "memory_bytes": 1024 ** 3,
                "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                "rates": {"memory": 1e6, "mte2_aic": 1e6, "mte2_aiv": 1e6,
                          "mte3": 1e6, "vector": 1e6},
                "compute_units": 4,
            },
            "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000,
                       "members": [0], "semantics": "shared"}],
        },
        "model": {"name": "geometry", "layers": 2, "hidden_size": 32, "num_heads": 4,
                  "num_kv_heads": 2, "head_dim": 8, "intermediate_size": 64,
                  "vocab_size": 128},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4, "output_tokens": 3}]},
        "runtime": {"dp": 1, "tp": 1, "pp": 1, "ep": 1, "max_num_batched_tokens": 16},
    }


def _minimal_op(name: str, **work) -> OpSpec:
    return OpSpec(name, "test", "prefill", 0, 0, (), (1,), (1,), (1,), (1,), **work)


def test_hbm_rate_intervention_does_not_scale_independent_onchip_copy():
    raw = _minimal_copy_case()
    copy = _minimal_op("onchip_copy", mte3_bytes=2e6)
    hbm_read = _minimal_op("hbm_read", hbm_bytes=2e6)
    scaled = scale_case_hardware(raw, hbm_bw_scale=2.0)

    assert graph_costs(raw, (copy,))["duration_ms"] == 2000.0
    assert graph_costs(scaled, (copy,))["duration_ms"] == 2000.0
    assert graph_costs(raw, (hbm_read,))["duration_ms"] == 2000.0
    assert graph_costs(scaled, (hbm_read,))["duration_ms"] == 1000.0
    rates = dict(parse_case(scaled).machine.dies[0].rates)
    assert rates.get("hbm", rates.get("memory")) == 2e6
    assert rates["mte2_aic"] == 1e6
    assert rates["mte2_aiv"] == 1e6
    assert rates["mte3"] == 1e6


def test_ready_complete_interval_is_not_published_as_service_busy():
    report = {
        "duration_ms": 10.0, "complete_ms": 10.0, "boundary": "device_graph",
        "timeline": [{"id": "r0:delay", "op_id": "delay", "predecessors": [],
                      "ready_ms": 0.0, "complete_ms": 10.0, "elapsed_ms": 10.0}],
        "resources": [],
    }
    rows = [{"op_id": "delay", "occurrences": 1, "duration_ms": 10.0, "total_ms": 10.0,
             "operator": {"op_id": "delay", "family": "test", "recipe": "delay",
                          "hbm_bytes": 0.0, "flops": 0.0,
                          "vector_ops": 0.0, "link_bytes": 0.0}}]
    attribution = attribute_program(report, rows)
    assert attribution["active_span_ms"] == 10.0
    assert attribution["busy_ms"] is None
    assert attribution["wait_ms"] is None
    assert attribution["resource_service"] == []
    assert attribution["service_utilization_by_role"] == {}
    assert attribution["unaccounted_active_gap_ms"] == 0.0


def test_serial_hbm_cube_bound_is_not_a_complete_step():
    case = parse_case(_a5_case(layers=1))
    peaks = extract_hardware_peaks(case)
    hbm, bf16 = peaks["hbm_peak_bytes_per_s"], peaks["compute_peaks"]["bf16"]
    operators = {"operators": [
        {"physical_id": "hbm_stage", "occurrences": 1, "program": "ResourceEnvelope",
         "duration_ms": 1.0, "total_ms": 1.0,
         "operator": {"op_id": "stage.hbm", "family": "custom", "recipe": "serial",
                      "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16",
                      "local_shape": [1, 4096], "flops": 0.0, "vector_ops": 0.0,
                      "hbm_bytes": 1e-3 * hbm, "link_bytes": 0.0}},
        {"physical_id": "cube_stage", "occurrences": 1, "program": "SteadyMatmul",
         "duration_ms": 1.0, "total_ms": 1.0,
         "operator": {"op_id": "stage.cube", "family": "custom", "recipe": "serial",
                      "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16",
                      "local_shape": [16, 4096], "flops": 1e-3 * bf16,
                      "vector_ops": 0.0, "hbm_bytes": 0.0, "link_bytes": 0.0}},
    ]}
    base = analyze_network_efficiency(
        case, phase="decode", tokens=1, context=1024, sequences=1,
        operator_report=operators, graph_report={"duration_ms": 2.0})
    bounded = evaluate_target_efficiency(
        case, phase="decode", tokens=1, context=1024, sequences=1,
        target_mbu=1.0, target_cube_mfu=1.0, target_vec_mfu=1.0, baseline_analysis=base)
    assert bounded["current_step_duration_ms"] == pytest.approx(2.0, rel=1e-12)
    assert bounded["target_step_duration_ms"] == pytest.approx(1.0, rel=1e-12)
    assert bounded["target_step_is_lower_bound"] is True
    assert bounded["target_pipe_totals_ms"]["hbm"] == pytest.approx(1.0, rel=1e-12)
    assert bounded["target_pipe_totals_ms"]["cube"] == pytest.approx(1.0, rel=1e-12)

    scenario = solve_sla_sizing(
        _a5_case(), sla_ms=1e-6, context=2048, max_sequences_per_dp=2,
        scenarios=[{"id": "target", "target_mbu": 1.0,
                    "target_cube_mfu": 1.0, "target_vec_mfu": 1.0}])["scenarios"][0]
    assert scenario["feasibility_status"] == "target_bound_only_unverified"
    assert scenario["sla_feasible"] is False
    assert scenario["b_sla"] == 0 and scenario["b_feasible"] == 0
    assert scenario["target_bound_is_lower_bound"] is True
    assert scenario["target_bound_step_ms_at_b1"] is not None
    assert scenario["step_ms_at_b1"] > 0.0
    assert scenario["search"]["lower_bound_can_prove_feasibility"] is False


def test_prefill_capacity_uses_prompt_window_and_pairs_stages():
    raw = _a5_case(pp=2, layers=2, prompt_tokens=1024, output_tokens=64)
    first = capacity_feedback(
        raw, phase="prefill", tokens=512, context=0, sequences=1, output_budget=1)
    second = capacity_feedback(
        raw, phase="prefill", tokens=512, context=0, sequences=1, output_budget=4096)
    assert first["kv_bytes_per_sequence"] == pytest.approx(
        second["kv_bytes_per_sequence"], rel=1e-12)
    assert first["b_vram_max_sync"] == second["b_vram_max_sync"]

    parsed = parse_case(raw)
    memory = Memory(parsed, workspace_tokens=512)
    kv_tokens = 512
    kv = [float(value) for value in memory.cache_bytes(kv_tokens, padded=True)]
    available = [min(float(row[stage]) for row in memory.capacity)
                 for stage in range(len(kv))]
    expected_headroom = [a - k * 1 for a, k in zip(available, kv)]
    assert len(first["capacity_by_stage"]) == len(expected_headroom)
    for row, expected in zip(first["capacity_by_stage"], expected_headroom):
        assert row["headroom_bytes"] == pytest.approx(expected, rel=1e-12, abs=1e-6)
    assert first["capacity_headroom_bytes"] == pytest.approx(min(expected_headroom), rel=1e-12, abs=1e-6)


def test_b1_over_sla_probes_larger_candidates_and_does_not_claim_infeasible():
    scenario = solve_sla_sizing(
        _a5_case(), sla_ms=1e-9, context=2048, max_sequences_per_dp=4,
        scenarios=[{"id": "baseline", "hbm_bw_scale": 1.0}])["scenarios"][0]
    assert scenario["b_sla"] == 0
    assert scenario["sla_feasible"] is False
    assert 2 in scenario["search"]["bounded_dispatch_probes"]
    assert scenario["search"]["bounded_search_exhaustive"] is True
    assert scenario["search"]["candidate_step_source"] == "none"
    assert scenario["feasibility_status"] == "sla_infeasible_within_evaluated_domain"


def test_prefill_page_boundary_and_workspace_use_actual_query_tokens():
    raw = compose_case("deepseek-v2-lite", hardware="a3", tp=1, dp=1, ep=1,
                       layers=2, prompt_tokens=512, output_tokens=64, concurrency=1)
    parsed = parse_case(raw)
    memory = Memory(parsed, workspace_tokens=512)
    kv_512 = [float(value) for value in memory.cache_bytes(512, padded=True)]
    kv_513 = [float(value) for value in memory.cache_bytes(513, padded=True)]
    assert kv_513 != kv_512

    first = capacity_feedback(
        raw, phase="prefill", tokens=512, context=0, sequences=1, output_budget=1)
    second = capacity_feedback(
        raw, phase="prefill", tokens=512, context=0, sequences=1, output_budget=4096)
    assert first["kv_bytes_per_sequence"] == pytest.approx(max(kv_512), rel=1e-12)
    assert first["kv_bytes_per_sequence"] != pytest.approx(max(kv_513), rel=1e-12)
    assert first["kv_bytes_per_sequence"] == pytest.approx(
        second["kv_bytes_per_sequence"], rel=1e-12)
    expected_b = _vram_max_sequences_per_dp(
        parsed, context=0, tokens_per_seq=512, output_budget=4096,
        emitted_tokens_per_seq=None, async_scheduling=False, phase="prefill")
    assert first["b_vram_max_sync"] == second["b_vram_max_sync"] == expected_b
    assert first["hbm_bytes_per_stage"] == [
        pytest.approx(value) for value in stage_hbm_bytes(parsed)]








def test_joint_candidate_selection_rejects_cross_candidate_stitching():
    latency_only = {"b": 2, "feasible": False,
                    "sla": {"pass": True}, "capacity": {"pass": False},
                    "step_ms": 1.0}
    capacity_only = {"b": 1, "feasible": False,
                     "sla": {"pass": False}, "capacity": {"pass": True},
                     "step_ms": 2.0}
    assert select_jointly_feasible_candidate([latency_only, capacity_only]) is None
    both = {"b": 1, "feasible": True,
            "sla": {"pass": True}, "capacity": {"pass": True},
            "step_ms": 1.0}
    assert select_jointly_feasible_candidate([both, latency_only])["b"] == 1
    # Three-point table: B1 already found but B2 fails SLA and B3 fails capacity.
    three_point = [
        {"b": 1, "feasible": True, "step_ms": 0.4,
         "sla": {"pass": True}, "capacity": {"pass": True}},
        {"b": 2, "feasible": False, "step_ms": 2.0,
         "sla": {"pass": False}, "capacity": {"pass": True}},
        {"b": 3, "feasible": False, "step_ms": 1.0,
         "sla": {"pass": True}, "capacity": {"pass": False}},
    ]
    assert select_jointly_feasible_candidate(three_point)["b"] == 1


def test_solver_and_sweep_publish_metrics_only_from_joint_candidate():
    raw = _a5_case()
    report = solve_sla_sizing(
        raw, sla_ms=50.0, context=2048, max_sequences_per_dp=2,
        scenarios=[{"id": "baseline"}])
    scenario = report["scenarios"][0]
    candidate = scenario["feasible_candidate"]
    assert candidate is not None
    assert candidate["input_boundary"]["sequences"] == candidate["b"]
    assert candidate["hardware"]["scenario_id"] == "baseline"
    assert candidate["plan"]["kind"] == "target_graphs"
    assert scenario["b_feasible"] == candidate["b"]
    assert scenario["step_ms_feasible"] == pytest.approx(candidate["step_ms"], rel=1e-12)
    assert scenario["feasible_network_mbu"] == pytest.approx(candidate["network_mbu"], rel=1e-12)
    expected_tps = (candidate["b"] * report["dp"] * report["emitted_tokens_per_seq"] * 1000.0
                    / candidate["step_ms"])
    assert scenario["tps_feasible"] == pytest.approx(expected_tps, rel=1e-12)

    swept = sweep_topologies(
        raw, candidates=[{"dp": 1, "tp": 2, "ep": 2, "pp": 1, "pcp": 1}],
        num_devices=2, sla_ms=50.0, context=2048, phase="decode", scope="body",
        output_budget=64, max_sequences_per_dp=2)
    row = swept["candidates"][0]
    assert row["feasible_candidate"] is not None
    assert row["feasible"] is True
    assert row["b_feasible"] == row["feasible_candidate"]["b"]
    assert row["network_mbu"] == pytest.approx(row["feasible_candidate"]["network_mbu"], rel=1e-12)
    assert row["step_ms_feasible"] == pytest.approx(row["feasible_candidate"]["step_ms"], rel=1e-12)




def test_scale_hardware_keeps_legacy_service_regimes_byte_identical():
    raw = _a5_case()
    rules = [{"id": "legacy", "match": {"recipe": "rmsnorm"},
              "coordinates": {"bytes": {"path": "work.hbm"}},
              "response": {"kind": "affine", "fixed_us": 10.0,
                           "terms": [{"coordinate": "bytes", "us_per_unit": 1e-6}]}}]
    raw["runtime"]["kernels"] = {"service_regimes": rules}
    scaled = scale_case_hardware(raw, hbm_bw_scale=2.0)
    factor_one = scale_case_hardware(raw, hbm_bw_scale=1.0)
    assert scaled.runtime.kernels["service_regimes"] == rules
    assert factor_one.runtime.kernels["service_regimes"] == rules
    diagnostics = legacy_service_regime_diagnostics(scaled)
    assert diagnostics["production_prediction"] is False
    assert diagnostics["scaled"] is False
    assert diagnostics["rules"] == rules
    assert dict(scaled.runtime.metadata)["legacy_service_regime_diagnostics"]["production_prediction"] is False
    geometry = {"phase": "decode", "tokens": 1, "context": 128, "sequences": 1, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    result = evaluate_resource_counterfactual(
        raw, resource="hbm_bw", factor=2.0, plan=plan, include_capacity=False)
    for side in ("baseline", "counterfactual"):
        diagnostics = result[side]["legacy_service_regime_diagnostics"]
        assert diagnostics["production_prediction"] is False
        assert diagnostics["rules"] == rules


def test_legacy_service_regime_fixed_term_is_diagnostic_only():
    basis = {"service_regimes": [{
        "rule_id": "legacy-memory",
        "queries": {"observable_response": True},
        "response": {"kind": "affine", "fixed_us": 10.0,
                     "terms": [{"coordinate": "bytes", "us_per_unit": 1e-6}]},
    }]}
    replay = diagnostic_legacy_invocation_bounds(basis)
    assert replay["fixed_ms"] == pytest.approx(0.01, rel=1e-12)
    assert replay["production_prediction"] is False
    assert replay["source_rule_ids"] == ["legacy-memory"]

    case = parse_case(compose_case("qwen3.5-35b-a3b", hardware="a5", tp=1, dp=1, ep=1,
                                   layers=1, prompt_tokens=128, output_tokens=16, concurrency=1))
    row = {
        "physical_id": "legacy-row", "occurrences": 1, "program": "Legacy",
        "duration_ms": 20.0, "basis": basis,
        "operator": {"op_id": "legacy.op", "family": "norm", "recipe": "rmsnorm",
                     "phase": "decode", "stage": 0, "layer": 0, "dtype": "bf16",
                     "flops": 0.0, "vector_ops": 0.0, "hbm_bytes": 0.0,
                     "link_bytes": 0.0},
    }
    analysis = analyze_operator_row(row, case, peaks=extract_hardware_peaks(case))
    assert analysis["fixed_ms"] != pytest.approx(0.01, rel=1e-9)


def test_study_zero_or_nonfinite_duration_is_invalid_not_best():
    base = {"probe": "p", "workload_key": "w", "boundary": "b",
            "operator_isolation": None, "operator_isolation_signature": None,
            "physical_devices": 1}
    valid = {**base, "candidate": "valid", "status": "modeled", "duration_ms": 2.0,
             "operators": {"work": {"flops": 100.0}}}
    zero = {**base, "candidate": "zero", "status": "modeled", "duration_ms": 0.0,
            "operators": {"work": {"flops": 100.0}}}
    negative = {**base, "candidate": "neg", "status": "modeled", "duration_ms": -1.0,
                "operators": {"work": {"flops": 100.0}}}
    infinite = {**base, "candidate": "inf", "status": "modeled", "duration_ms": float("inf"),
                "operators": {"work": {"flops": 100.0}}}
    _normalize_ranking_row(zero)
    _normalize_ranking_row(negative)
    _normalize_ranking_row(infinite)
    assert zero["status"] == "invalid" and zero["invalid_reason"] == "invalid_zero_duration_with_positive_work"
    assert negative["status"] == "invalid" and negative["invalid_reason"] == "invalid_negative_duration"
    assert infinite["status"] == "invalid" and infinite["invalid_reason"] == "invalid_nonfinite_duration"
    ranked = _rankings([valid, zero, negative, infinite])
    assert ranked
    candidates = ranked[0]["candidates"]
    assert [row["candidate"] for row in candidates] == ["valid"]
    assert candidates[0]["ratio_to_fastest"] == pytest.approx(1.0, rel=1e-12)

    noop = _normalize_ranking_row({**base, "candidate": "noop", "status": "modeled",
                                   "duration_ms": 0.0, "operators": {"work": {}}})
    assert noop["status"] == "modeled" and noop["zero_work_noop"] is True




def test_public_sla_prefill_keeps_prompt_and_lifecycle_capacity_separate():
    raw = compose_case("deepseek-v2-lite", hardware="a3", tp=1, dp=1, ep=1,
                       layers=1, prompt_tokens=512, output_tokens=131072,
                       concurrency=1, max_num_seqs=1, max_model_len=262144)
    raw["machine"]["device"]["memory_bytes"] = 1150 * 1024 ** 2
    report = solve_sla_sizing(
        raw, sla_ms=50.0, context=0, tokens_per_seq=512,
        output_budget=131072, max_sequences_per_dp=1, phase="prefill",
        scenarios=[{"id": "baseline"}])
    summary = report["async_scheduling_memory"]
    assert summary["capacity_window"] == "prefill_prompt_only"
    assert summary["b_vram_max_sync"] == 1
    assert summary["b_vram_max_lifecycle_sync"] == 0
    assert summary["b_vram_max_joint_sync"] == 0
    assert summary["kv_bytes_per_seq_sync_per_stage"] != summary["kv_bytes_per_seq_async_per_stage"]
    assert summary["kv_bytes_per_seq_lifecycle_per_stage"] != summary["kv_bytes_per_seq_sync_per_stage"]
    scenario = report["scenarios"][0]
    assert scenario["b_vram_max_joint"] == 0
    assert scenario["b_feasible"] == 0


def test_solver_does_not_reuse_b1_analysis_from_another_context():
    raw = _a5_case()
    baseline = solve_sla_sizing(
        raw, sla_ms=50.0, context=2048, max_sequences_per_dp=1,
        scenarios=[{"id": "baseline"}])["scenarios"][0]
    foreign = analyze_network_efficiency(
        raw, phase="decode", tokens=1, context=8192, sequences=1, scope="body")
    reused = solve_sla_sizing(
        raw, sla_ms=50.0, context=2048, max_sequences_per_dp=1,
        scenarios=[{"id": "baseline"}], b1_analysis=foreign)["scenarios"][0]
    assert reused["b_feasible"] == baseline["b_feasible"]
    assert reused["step_ms_feasible"] == pytest.approx(baseline["step_ms_feasible"], rel=1e-12)
    assert reused["tps_feasible"] == pytest.approx(baseline["tps_feasible"], rel=1e-12)




def test_capacity_feedback_keeps_prefill_and_joint_lifecycle_limits_separate():
    raw = compose_case("deepseek-v2-lite", hardware="a3", tp=1, dp=1, ep=1,
                       layers=1, prompt_tokens=512, output_tokens=131072,
                       concurrency=1, max_num_seqs=1, max_model_len=262144)
    raw["machine"]["device"]["memory_bytes"] = 1150 * 1024 ** 2
    feedback = capacity_feedback(
        parse_case(raw), phase="prefill", tokens=512, context=0, sequences=1,
        output_budget=131072)
    assert feedback["b_vram_max_sync"] == 1
    assert feedback["b_vram_max_lifecycle_sync"] == 0
    assert feedback["b_vram_max_joint_sync"] == 0
    assert feedback["is_vram_feasible"] is True
    assert feedback["is_lifecycle_feasible"] is False
    assert feedback["is_joint_feasible"] is False
    assert feedback["lifecycle_capacity_headroom_bytes"] < 0.0


def test_capacity_lifecycle_uses_actual_proposal_query_tokens():
    raw = _a5_case(layers=1, mtp_proposal_tokens=2, mtp_acceptance_length=2.0)
    parsed = parse_case(raw)
    assert decode_query_tokens(parsed, 64) == 3






def test_public_sla_prefill_binds_lifecycle_to_decode_commit(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=True, graph_pool_kib=200)
    _patch_bounded_solver(monkeypatch, lambda b: 1.0)
    report = solve_sla_sizing(
        case, sla_ms=1.5, context=0, tokens_per_seq=512, output_budget=128,
        max_sequences_per_dp=2, phase="prefill", scenarios=[{"id": "baseline"}])
    scenario = report["scenarios"][0]
    assert scenario["b_sla"] == 2
    assert scenario["b_vram_max_joint"] == 2
    assert scenario["b_feasible"] == 2
    assert scenario["prefill_tokens_per_seq"] == 512
    assert scenario["decode_emitted_tokens_per_seq"] == 1.0
    assert scenario["emitted_tokens_per_seq"] == 512.0
    assert scenario["joint_capacity"]["lifecycle_active_kv_bytes"] == pytest.approx(1536 * 1024)
    assert scenario["joint_capacity"]["lifecycle_retiring_kv_bytes"] == pytest.approx(640 * 1024)
    assert scenario["joint_capacity"]["lifecycle_peak_bytes"] == pytest.approx(2560 * 1024)
    assert scenario["joint_capacity"]["lifecycle_headroom_bytes"] == pytest.approx(140 * 1024)
    assert scenario["lifecycle_capacity_window"]["window"]["steps_per_wave"] == 128
    assert report["decode_emitted_tokens_per_seq"] == 1.0
    assert report["prefill_tokens_per_seq"] == 512
    assert report["async_scheduling_memory"]["decode_emitted_tokens_per_seq"] == 1.0
    assert report["async_scheduling_memory"]["prefill_tokens_per_seq"] == 512
    assert report["async_scheduling_memory"]["capacity_window_b1"]["stage_rank"][0]["supply_bytes"] == pytest.approx(2700 * 1024)


def test_public_solver_three_point_non_monotonic_keeps_first_candidate(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=True,
                             graph_pool_kib=200, graph_capture=True)
    _patch_bounded_solver(monkeypatch, lambda b: {1: 0.6, 2: 2.0, 3: 1.0}[b])
    report = solve_sla_sizing(
        case, sla_ms=1.5, context=512, tokens_per_seq=1, output_budget=128,
        max_sequences_per_dp=3, phase="decode", scenarios=[{"id": "baseline"}])
    scenario = report["scenarios"][0]
    assert scenario["b_vram_max_joint"] == 2
    assert scenario["b_sla"] == 3
    assert scenario["b_feasible"] == 1
    assert scenario["step_ms_feasible"] == pytest.approx(0.6)
    assert scenario["feasibility_status"] == "feasible_capacity_limited"
    assert 2 in scenario["search"]["evaluated_candidates"]
    assert 3 in scenario["search"]["evaluated_candidates"]
    assert scenario["latency_candidate"]["b"] == 3
    assert scenario["latency_candidate"]["capacity"]["pass"] is False
    assert scenario["feasible_candidate"]["b"] == 1


def test_no_async_window_has_no_delayed_release_or_peak_over_active(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=False)
    window = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0)
    assert window["window"]["delayed_release_steps"] == 0
    assert window["window"]["lookahead_tokens"] == 0
    assert window["retiring_kv_bytes"] == 0.0
    assert window["wave_retiring_kv_bytes"] == 0.0
    assert window["peak_bytes"] == pytest.approx(window["active_kv_bytes"])
    feedback = capacity_feedback(
        case, phase="decode", context=512, tokens=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0)
    assert feedback["b_vram_max_sync"] == feedback["b_vram_max_async"]
    assert feedback["joint_capacity_headroom_bytes"] == pytest.approx(
        feedback["capacity_headroom_bytes"])


def test_real_proposal_lifecycle_uses_decode_commit_not_prefill_volume():
    raw = compose_case("qwen3.5-35b-a3b", hardware="a5", tp=2, dp=1, ep=2,
                       layers=1, prompt_tokens=512, output_tokens=64,
                       concurrency=1, max_num_seqs=2, max_model_len=4096,
                       mtp_proposal_tokens=2, mtp_acceptance_length=2.0)
    parsed = parse_case(raw)
    feedback = capacity_feedback(
        parsed, phase="prefill", context=0, tokens=512, sequences=1,
        output_budget=64)
    lifecycle = feedback["lifecycle_capacity_window_facts"]
    assert feedback["capacity_window"]["query_tokens_per_sequence"] == 512
    assert feedback["capacity_window"]["emitted_tokens_per_sequence"] == 512.0
    assert lifecycle["query_tokens_per_sequence"] == 3
    assert lifecycle["emitted_tokens_per_sequence"] == pytest.approx(2.0)
    assert lifecycle["window"]["steps_per_wave"] == 32
    assert feedback["b_vram_max_lifecycle_sync"] <= feedback["b_vram_max_sync"]


def test_t08_live_grant_reduces_the_same_projected_supply(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=True, graph_pool_kib=0)
    ledger = Memory(case, workspace_tokens=2)
    before = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0, memory=ledger)
    assert ledger.grant_batch(
        0, [{"name": "t04_last_use", "bytes": 100 * 1024, "stage": 0}],
        owner="t04")
    after = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0, memory=ledger)
    assert after["stage_rank"][0]["fixed_allocations"]["grants"] == pytest.approx(100 * 1024)
    assert after["supply_bytes"] == pytest.approx(before["supply_bytes"] - 100 * 1024)
    assert after["headroom_bytes"] == pytest.approx(before["headroom_bytes"] - 100 * 1024)


def test_t04_last_use_release_restores_the_still_held_supply(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=True, graph_pool_kib=0)
    ledger = Memory(case, workspace_tokens=2)
    assert ledger.resize(0, "request", 10)
    held = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0, memory=ledger)
    assert held["external_held_bytes"] > 0.0
    assert held["supply_bytes"] < 2700 * 1024
    ledger.defer_release(0, "request", steps=1)
    delayed = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0, memory=ledger)
    assert delayed["external_held_bytes"] == pytest.approx(held["external_held_bytes"])
    ledger.step_delayed_releases(0)
    released = capacity_window(
        case, phase="decode", context=512, tokens_per_seq=1, sequences=2,
        output_budget=128, emitted_tokens_per_seq=1.0, memory=ledger)
    assert released["external_held_bytes"] == 0.0
    assert released["supply_bytes"] == pytest.approx(2700 * 1024)
