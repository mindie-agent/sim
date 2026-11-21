"""
[Role] Exercise fixed-plan hardware interventions, pipe bounds and serving capacity.
[Invariants] Independent rates preserve work. The same SLA candidate must satisfy timing and capacity.
[Timeouts & Locks] Bounded graphs and candidate searches on CPU.
[Side-Effect Policy] No devices or repository writes. Test doubles are scoped to each test.
"""
from __future__ import annotations

import json
import math

import pytest
from _support.axes import REPRESENTATIVE_SKUS
from _support.captures import REPO_ROOT
from _support.memory_cases import case as declared_case

from mindie_sim.compose import compose_case
from mindie_sim.prediction import efficiency as efficiency_module
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    _decompose_operator_pipes,
    analyze_network_efficiency,
    analyze_operator_row,
    attribute_program,
    capacity_feedback,
    capacity_window,
    decode_query_tokens, diagnostic_legacy_invocation_bounds,
    legacy_service_regime_diagnostics,
    evaluate_fixed_plan,
    evaluate_resource_counterfactual,
    extract_hardware_peaks,
    fit_case_roofline_regimes,
    prepare_fixed_plan,
    scale_case_hardware,
    select_jointly_feasible_candidate,
    solve_sla_sizing, stage_hbm_bytes,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.study import _normalize_ranking_row, _rankings, study
from mindie_sim.sweep import sweep_topologies


pytestmark = pytest.mark.tier0

HYBRID_MODEL = "qwen3.5-35b-a3b"
LATENT_MODEL = "deepseek-v2-lite"
HYBRID_HARDWARE = REPRESENTATIVE_SKUS["a5"]
LATENT_HARDWARE = REPRESENTATIVE_SKUS["a3"]


def _hybrid_case(**overrides) -> dict:
    defaults = {"layers": 1, "prompt_tokens": 2048, "output_tokens": 64, "concurrency": 2}
    defaults.update(overrides)
    return compose_case(HYBRID_MODEL, hardware=HYBRID_HARDWARE, tp=2, dp=1, ep=2, **defaults)


def _capacity_fixture(monkeypatch, *, async_scheduling: bool = True,
                      graph_pool_kib: int = 200, graph_capture: bool = False,
                      budget_kib: int = 2700):
    """One rank/stage paying 1 KiB per KV row in 128-row pages.

    Memory supplies weights, workspace and reserves. Only KV page geometry is controlled.
    """
    config = json.loads((REPO_ROOT / "mindie_sim/model_configs" / f"{LATENT_MODEL}.json").read_text())
    config.update({
        "num_hidden_layers": 1, "vocab_size": 1024, "hidden_size": 128,
        "intermediate_size": 256, "moe_intermediate_size": 64,
        "num_attention_heads": 2, "num_key_value_heads": 2,
        "kv_lora_rank": 64, "qk_nope_head_dim": 64, "qk_rope_head_dim": 32,
        "v_head_dim": 64,
    })
    raw = compose_case(
        config, hardware=LATENT_HARDWARE, tp=1, dp=1, ep=1, layers=1,
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


def test_fixed_plan_baseline_matches_ordinary_analysis_and_separates_phases():
    raw = _hybrid_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    fixed = evaluate_fixed_plan(raw, plan=plan)
    direct = analyze_network_efficiency(raw, **geometry)

    assert fixed["graph_duration_ms"] == pytest.approx(direct["graph_duration_ms"], rel=1e-12)
    assert fixed["operator_sum_ms"] == pytest.approx(direct["operator_sum_ms"], rel=1e-12)
    assert fixed["total_hbm_bytes"] == pytest.approx(direct["total_hbm_bytes"], rel=1e-12)

    attribution = fixed["attribution"]
    assert attribution["work_ms"] == pytest.approx(fixed["operator_sum_ms"], rel=1e-12)
    assert attribution["span_ms"] == pytest.approx(fixed["graph_duration_ms"], rel=1e-12)
    assert attribution["busy_ms"] is None
    assert attribution["wait_ms"] is None
    assert attribution["service_busy_union_ms"] is None
    assert attribution["resource_service"]
    assert {"hbm", "cube", "vector"}.issubset(attribution["service_utilization_by_role"])

    critical = attribution["critical_path"]
    assert 0.0 <= critical["work_path_ms"] <= attribution["span_ms"] + 1e-9
    assert critical["basis"].startswith(("operator_service", "isolated_costs"))

    for nominal, network in (("cube_mfu", "network_cube_mfu"), ("vec_mfu", "network_vec_mfu"),
                             ("mbu", "network_mbu"), ("link_util", "network_link_util")):
        assert fixed["nominal"][nominal] == pytest.approx(fixed[network], rel=1e-12)
    assert fixed["nominal"]["denominator"] == "graph_span_ms"
    assert fixed["effective_service_utilization"] == attribution["service_utilization_by_role"]
    assert all(0.0 <= value <= 1.0 + 1e-12 for value in fixed["effective_service_utilization"].values())


def test_single_hbm_intervention_reevaluates_same_program_without_work_rescaling():
    raw = _hybrid_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    result = evaluate_resource_counterfactual(
        raw, resource="hbm_bw", factor=2.0, plan=plan, include_capacity=False)

    assert result["counterfactual"]["graph_duration_ms"] < result["baseline"]["graph_duration_ms"]
    assert result["work_conservation"]["conserved"] is True
    for field in ("total_flops", "total_vector_ops", "total_hbm_bytes", "total_link_bytes"):
        assert result["work_conservation"]["fields"][field]["delta_ratio"] == pytest.approx(0.0, abs=1e-9)
    assert result["changed_operators"][0]["delta_ms"] < 0.0
    with pytest.raises(ValueError, match="t06_normalized_service_api"):
        evaluate_resource_counterfactual(
            raw, resource="effective_rate", factor=1.5, plan=plan, include_capacity=False)


def test_capacity_intervention_changes_feedback_not_program_step():
    raw = _hybrid_case(max_num_seqs=8192, max_num_batched_tokens=8192, max_model_len=16384)
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


def test_roofline_fit_exports_nominal_bounds_without_identifying_pipe():
    raw = _hybrid_case()
    geometry = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 2, "scope": "body"}
    fitted = fit_case_roofline_regimes(raw, **geometry)
    assert fitted["operators"]
    assert all(row["bottleneck_pipe"] is None for row in fitted["operators"])

    # Either HBM or cube can account for the same observed duration.
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


def _minimal_op(name: str, **work) -> OpSpec:
    return OpSpec(name, "test", "prefill", 0, 0, (), (1,), (1,), (1,), (1,), **work)


def test_hbm_rate_intervention_does_not_scale_independent_onchip_copy():
    raw = declared_case()
    raw["machine"]["device"]["rates"].update(
        memory=1e6, mte2_aic=1e6, mte2_aiv=1e6, mte3=1e6)
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


@pytest.mark.parametrize("record_key", ["devices", "dies"])
def test_hbm_rate_intervention_scales_directional_datapath_capacities(record_key):
    raw = _hybrid_case()
    machine = raw["machine"]
    defaults = machine["device"]
    read, write = "hbm_read_bytes_per_s", "hbm_write_bytes_per_s"
    override = {write: defaults[write] * 0.5}
    machine[record_key] = [{}, override]
    a5_case = parse_case(raw)
    scaled = scale_case_hardware(a5_case, hbm_bw_scale=2.0)

    for die, scaled_die, rates in zip(a5_case.machine.dies, scaled.machine.dies,
                                     (defaults, {**defaults, **override}), strict=True):
        pairs = ((die, scaled_die), (datapath_for_die(die), datapath_for_die(scaled_die)))
        for original, doubled in pairs:
            assert original.hbm_read_bytes_per_s == rates[read]
            assert original.hbm_write_bytes_per_s == rates[write]
            assert doubled.hbm_read_bytes_per_s == rates[read] * 2.0
            assert doubled.hbm_write_bytes_per_s == rates[write] * 2.0


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


def test_prefill_capacity_uses_prompt_window_and_pairs_stages():
    raw = _hybrid_case(pp=2, layers=2, prompt_tokens=1024, output_tokens=64)
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
    expected_headroom = [a - k for a, k in zip(available, kv)]
    assert first["hbm_bytes_per_stage"] == list(map(float, stage_hbm_bytes(parsed)))
    assert len(first["capacity_by_stage"]) == len(expected_headroom)
    for row, expected in zip(first["capacity_by_stage"], expected_headroom):
        assert row["headroom_bytes"] == pytest.approx(expected, rel=1e-12, abs=1e-6)
    assert first["capacity_headroom_bytes"] == pytest.approx(min(expected_headroom), rel=1e-12, abs=1e-6)


def test_first_candidate_over_sla_still_probes_larger_candidates(monkeypatch):
    _patch_bounded_solver(monkeypatch, lambda batch: float(batch))
    scenario = solve_sla_sizing(
        _hybrid_case(), sla_ms=1e-9, context=2048, max_sequences_per_dp=4,
        scenarios=[{"id": "baseline", "hbm_bw_scale": 1.0}])["scenarios"][0]
    assert scenario["b_sla"] == 0
    assert scenario["b_feasible"] == 0
    assert scenario["tps_at_b1"] > 0
    assert scenario["tps_at_sla"] == scenario["tps_feasible"] == 0
    assert scenario["tps_gain_vs_baseline"] is None
    assert scenario["tps_feasible_gain_vs_baseline"] is None
    assert scenario["sla_feasible"] is False
    assert 2 in scenario["search"]["bounded_dispatch_probes"]
    assert scenario["search"]["bounded_search_exhaustive"] is True
    assert scenario["feasibility_status"] == "sla_infeasible_within_evaluated_domain"


def test_capacity_infeasible_baseline_keeps_latency_sla_gain(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=False, budget_kib=0)
    _patch_bounded_solver(monkeypatch, lambda batch: float(batch))
    scenario = solve_sla_sizing(
        case, sla_ms=1.0, context=0, output_budget=1, max_sequences_per_dp=1,
        scenarios=[{"id": "baseline"}])["scenarios"][0]
    assert scenario["b_sla"] == 1
    assert scenario["b_feasible"] == 0
    assert scenario["tps_at_sla"] > 0
    assert scenario["tps_feasible"] == 0
    assert scenario["tps_gain_vs_baseline"] == 1.0
    assert scenario["tps_feasible_gain_vs_baseline"] is None


@pytest.mark.parametrize("points, expected", [
    ([(False, True), (True, False)], None),
    ([(True, True), (False, True), (True, False)], 1),
])
def test_joint_candidate_selection_rejects_cross_candidate_stitching(points, expected):
    candidates = [
        {"b": batch, "feasible": timing and capacity, "step_ms": float(batch),
         "sla": {"pass": timing}, "capacity": {"pass": capacity}}
        for batch, (timing, capacity) in enumerate(points, 1)
    ]
    selected = select_jointly_feasible_candidate(candidates)
    assert (selected["b"] if selected else None) == expected


def test_solver_and_sweep_publish_metrics_only_from_joint_candidate():
    raw = _hybrid_case()
    report = solve_sla_sizing(
        raw, sla_ms=50.0, context=2048, max_sequences_per_dp=2,
        scenarios=[{"id": "baseline"}])
    scenario = report["scenarios"][0]
    candidate = scenario["feasible_candidate"]
    assert candidate["input_boundary"]["sequences"] == candidate["b"]
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
    assert swept["plan_selection"] == "reselected_parallel_per_candidate"
    assert not report["solver"]["b_sla_is_proven_max"]
    row = swept["candidates"][0]
    assert row["plan_mode"] == "reselected_parallel"
    assert row["b_feasible"] == row["feasible_candidate"]["b"]
    assert row["network_mbu"] == pytest.approx(row["feasible_candidate"]["network_mbu"], rel=1e-12)
    assert row["step_ms_feasible"] == pytest.approx(row["feasible_candidate"]["step_ms"], rel=1e-12)
    result = study(
        [{"id": "control", "case": compose_case(HYBRID_MODEL, LATENT_HARDWARE, tp=2, ep=2, layers=1)}],
        [{"id": "decode", "phase": "decode", "tokens": 1, "context": 1024, "sequences": 1,
          "scope": "body"}], level="graph")
    graph = result["evaluations"][0]["graph"]
    assert graph["plan_mode"] == "reselected_case"
    assert graph["attribution"]["span_ms"] == pytest.approx(graph["duration_ms"])


def test_scale_hardware_preserves_service_regime_records():
    raw = _hybrid_case()
    rules = [{"id": "legacy", "match": {"recipe": "rmsnorm"},
              "coordinates": {"bytes": {"path": "work.hbm"}},
              "response": {"kind": "affine", "fixed_us": 10.0,
                           "terms": [{"coordinate": "bytes", "us_per_unit": 1e-6}]}}]
    raw["runtime"]["kernels"] = {"service_regimes": rules}
    scaled = scale_case_hardware(raw, hbm_bw_scale=2.0)
    assert scaled.runtime.kernels["service_regimes"] == rules
    diagnostics = legacy_service_regime_diagnostics(scaled)
    assert diagnostics["rules"] == rules
    geometry = {"phase": "decode", "tokens": 1, "context": 128, "sequences": 1, "scope": "body"}
    plan = prepare_fixed_plan(raw, **geometry)
    result = evaluate_resource_counterfactual(
        raw, resource="hbm_bw", factor=2.0, plan=plan, include_capacity=False)
    for side in ("baseline", "counterfactual"):
        assert result[side]["legacy_service_regime_diagnostics"]["rules"] == rules


def test_service_regime_fixed_term_is_diagnostic_only():
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

    case = parse_case(compose_case(HYBRID_MODEL, hardware=HYBRID_HARDWARE, tp=1, dp=1, ep=1,
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


@pytest.mark.parametrize("duration, reason", [
    (0.0, "invalid_zero_duration_with_positive_work"),
    (-1.0, "invalid_negative_duration"),
    (float("inf"), "invalid_nonfinite_duration"),
])
def test_study_excludes_invalid_durations_from_ranking(duration, reason):
    valid = {"probe": "p", "workload_key": "w", "boundary": "b", "physical_devices": 1,
             "operator_isolation": None, "operator_isolation_signature": None,
             "candidate": "valid", "status": "modeled", "duration_ms": 2.0,
             "operators": {"work": {"flops": 100.0}}}
    invalid = _normalize_ranking_row({**valid, "candidate": "invalid", "duration_ms": duration})
    assert invalid["status"] == "invalid" and invalid["invalid_reason"] == reason
    assert [row["candidate"] for row in _rankings([valid, invalid])[0]["candidates"]] == ["valid"]
    noop = _normalize_ranking_row({**valid, "duration_ms": 0.0, "operators": {"work": {}}})
    assert noop["status"] == "modeled" and noop["zero_work_noop"] is True


def test_public_sla_prefill_keeps_prompt_and_lifecycle_capacity_separate():
    raw = compose_case(LATENT_MODEL, hardware=LATENT_HARDWARE, tp=1, dp=1, ep=1,
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
    assert scenario["b_vram_max_joint"] == scenario["b_feasible"] == 0


def test_solver_recomputes_analysis_for_different_context():
    raw = _hybrid_case()
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


def test_public_sla_prefill_binds_lifecycle_to_decode_commit(monkeypatch):
    case = _capacity_fixture(monkeypatch, async_scheduling=True, graph_pool_kib=200)
    _patch_bounded_solver(monkeypatch, lambda b: 1.0)
    report = solve_sla_sizing(
        case, sla_ms=1.5, context=0, tokens_per_seq=512, output_budget=128,
        max_sequences_per_dp=2, phase="prefill", scenarios=[{"id": "baseline"}])
    scenario = report["scenarios"][0]
    assert scenario["b_sla"] == scenario["b_vram_max_joint"] == scenario["b_feasible"] == 2
    assert scenario["prefill_tokens_per_seq"] == 512
    assert scenario["decode_emitted_tokens_per_seq"] == 1.0
    assert scenario["emitted_tokens_per_seq"] == 512.0
    assert scenario["joint_capacity"]["lifecycle_active_kv_bytes"] == pytest.approx(1536 * 1024)
    assert scenario["joint_capacity"]["lifecycle_retiring_kv_bytes"] == pytest.approx(640 * 1024)
    assert scenario["joint_capacity"]["lifecycle_peak_bytes"] == pytest.approx(2560 * 1024)
    assert scenario["joint_capacity"]["lifecycle_headroom_bytes"] == pytest.approx(140 * 1024)
    assert scenario["lifecycle_capacity_window"]["window"]["steps_per_wave"] == 128
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
    assert {2, 3} <= set(scenario["search"]["evaluated_candidates"])
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
    raw = compose_case(HYBRID_MODEL, hardware=HYBRID_HARDWARE, tp=2, dp=1, ep=2,
                       layers=1, prompt_tokens=512, output_tokens=64,
                       concurrency=1, max_num_seqs=2, max_model_len=4096,
                       mtp_proposal_tokens=2, mtp_acceptance_length=2.0)
    parsed = parse_case(raw)
    feedback = capacity_feedback(
        parsed, phase="prefill", context=0, tokens=512, sequences=1,
        output_budget=64)
    lifecycle = feedback["lifecycle_capacity_window_facts"]
    assert feedback["capacity_window"]["query_tokens_per_sequence"] == 512
    assert lifecycle["query_tokens_per_sequence"] == decode_query_tokens(parsed, 64) == 3
    assert lifecycle["emitted_tokens_per_sequence"] == pytest.approx(2.0)
    assert lifecycle["window"]["steps_per_wave"] == 32
    assert feedback["b_vram_max_lifecycle_sync"] <= feedback["b_vram_max_sync"]


def test_live_grant_reduces_the_same_projected_supply(monkeypatch):
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


def test_last_use_release_restores_the_still_held_supply(monkeypatch):
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


@pytest.mark.parametrize("factor", [0, -1, float("inf"), float("nan")])
def test_hardware_counterfactual_requires_positive_finite_factor(factor):
    with pytest.raises(ValueError, match="resource_factor"):
        evaluate_resource_counterfactual(declared_case(), resource="hbm_bw", factor=factor)
