"""Candidate studies, SKU comparisons and modeling-priority oracles.

[Role] Verify ``study()`` candidate comparisons, SKU comparison and perf bounds,
  model-convergence cost census, modeling-priority ranking, operator cohort
  screening, direct-mesh collectives, and non-cached efficiency evaluation.
[Invariants] Candidate identity separates memory budget and runtime bindings;
  comparisons only pair equal effective work; every rank/byte conservation is
  an independent analytical formula and no result crosses two public calls.
[Timeouts & Locks] In-memory tier0 studies plus a C11 arena parity run with a
  real compiler; no device context, network, sleeps, or filesystem writes
  outside ``tmp_path``.
[Side-Effect Policy] Read-only over packaged examples and fresh per-test
  mappings; temporary captures live only under pytest ``tmp_path``.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest

from _support.collective_cases import cohort_case, exchange
from _support.serving_cases import STUDY_PROBES, case
from mindie_sim import graph_costs, operator_costs, predict, runtime_advisor, study, target_graphs
from mindie_sim.__main__ import main, parser, run
from mindie_sim.compose import compose_case
from mindie_sim.prediction.efficiency import analyze_network_efficiency, scale_case_hardware, solve_sla_sizing
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.execution.layer_runs import homogeneous_layer_runs
from mindie_sim.prediction.program.physical_key import physical_key
from mindie_sim.prediction.program import KernelProgram, PipelineProgram, Stage
from mindie_sim.prediction.runtime.engine import Engine, Resource
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.sku_comparison import compare_skus
from mindie_sim.sweep import sweep_topologies
from mindie_sim.sweep.engine import compute_sweep_candidate_id
from tools.model_convergence import (compare_runs, fingerprint, representative_probes,
                                     retained_gemm_impact, run_probe, select_probes)
from tools.modeling_priority import cost_ranking, residual_ranking

pytestmark = pytest.mark.tier0

TOL, EXACT, MIB = 1e-9, 1e-12, 1024 * 1024
ROOT = Path(__file__).resolve().parents[2]
DT_SKU = "Ascend950DT_9582"
LONG_MODEL, SECOND_MODEL = "deepseek-v4.1-flash", "kimi-k3.1"
LONG_LAYERS, WIDE_UNITS, WIDE_RESOURCES = 40, 72, 300
SECONDS_BUDGET, COMPARE_BUDGET = 540.0, 240.0
def same(actual, expected, rel=TOL):
    return math.isclose(actual, expected, rel_tol=rel)
def _control():
    return json.loads((ROOT / "examples/source-families/attnres-prefix-control.json").read_text())
def _evaluate(raw, *, geometry=None, matmul=None, balanced_matmul=None, routing=None, level="operators", max_operators=1000):
    geometry = geometry or {"phase": "prefill", "tokens": 4, "context": 0, "sequences": 1, "scope": "model"}
    result = run_probe({"case": raw, "geometry": geometry, "matmul": matmul, "level": level, "balanced_matmul": balanced_matmul, "routing": routing, "top": 100, "max_operators": max_operators})
    effective = deepcopy(raw)
    if routing is not None:
        effective.setdefault("runtime", {})["routing"] = routing
    return {"id": "control", "input_key": fingerprint({"case": effective, "geometry": geometry, "level": level}), **result}
def _pair(recipe, predicted_us, observed_us, calls, *, capture_calls=1, contract_capture_calls=1, contract_model_calls=None):
    return {"recipe": recipe, "shape": [16], "dtype": "bf16", "source_lines": [2], "match": "tensor_contract; dispatch_declared", "predicted_us": predicted_us, "observed_median_us": observed_us, "model_calls": calls, "capture_calls": capture_calls, "contract_model_calls": calls if contract_model_calls is None else contract_model_calls, "contract_capture_calls": contract_capture_calls}
def _comparison(pairs, total):
    return {"boundary": "sum_of_isolated_device_programs", "operator_sum_ms": total, "model": "structural", "hardware": "declared", "software": {}, "phase": "prefill", "tokens_per_sequence": 16, "sequences": 1, "selected_families": [], "comparisons": pairs}
def _allreduce_op(world_size):
    payload = 128 * MIB
    return OpSpec("probe.allreduce", "allreduce", "prefill", 0, 0, (), (1,), (1,), (1,), (1,), link_bytes=2.0 * (world_size - 1) / world_size * payload, metadata=(("group_axis", "tp"), ("steps", 2 * (world_size - 1))))

# --------------------------------------------------------------------------- model convergence
def test_actual_source_selection_and_repetition_conserve_all_reported_costs():
    raw = _control(); costs = _evaluate(raw)["costs"]
    families = {row["recipe"]: row["occurrences"] for row in costs["families"]}
    assert families["attn_res_fwd"] == 7 and families["flash_attention"] == 3
    for collection in ("families", "programs", "top_signatures"):
        assert sum(row["isolated_ms"] for row in costs[collection]) == pytest.approx(costs["operator_sum_ms"])
        assert sum(row["occurrences"] for row in costs[collection]) == costs["operator_occurrences"]
    assert sum(row["occurrences"] for row in costs["programs"] if row["source_boundary"] == "kernel") == 7
    assert costs["remaining_live_tasks"] == costs["remaining_allocations"] == 0
    assert next(row for row in costs["top_signatures"] if row["recipe"] == "qkv_projection")["local_shape"] == [4, 512, 256]
def test_explicit_mapping_change_compares_equal_work_and_preserves_changed_cost():
    raw = _control(); baseline = _evaluate(raw)
    modified = _evaluate(raw, matmul={"kind": "tiled_matmul", "tile_m": 2, "tile_n": 16, "tile_k": 8, "reuse_inputs": False, "reuse_weights": False})
    comparison = compare_runs({"probes": [modified]}, {"probes": [baseline]})["probes"][0]
    assert comparison["status"] == "compared" and comparison["configuration"] == "explicit_matrix_recipe_axes"
    assert {item["resolved_program"] for item in modified["balanced_recipes"].values()} == {"TiledMatmul"}
    assert comparison["semantic_work_unchanged"] and not comparison["executed_service_unchanged"]
    assert any(value > 0 for resource, value in comparison["executed_service_delta"].items() if resource.startswith("memory@"))
    assert comparison["difference_ms"] != 0
    assert comparison["difference_ms"] == pytest.approx(sum(row["difference_ms"] for row in comparison["family_changes"]))
    assert next(row for row in comparison["family_changes"] if row["family"] == "attention")["difference_ms"] == 0
    assert raw == _control()
def test_phase_comparison_keeps_physical_work_separate_from_changed_duration():
    raw = _control(); mapping = {"candidates": [[2, 16, 8]], "compute_m": 2, "compute_n": 8}
    detailed = _evaluate(raw, matmul={"kind": "mapped_matmul", **mapping}); phase = _evaluate(raw, matmul={"kind": "phase_matmul", **mapping})
    comparison = compare_runs({"probes": [phase]}, {"probes": [detailed]})["probes"][0]
    assert comparison["semantic_work_unchanged"] and comparison["executed_service_unchanged"]
    assert comparison["executed_service_max_relative_delta"] < 1e-12 and comparison["difference_ms"] > 0
def test_balanced_recipe_comparison_executes_distinct_groups_without_changing_work():
    raw = case(prompt=4, output=1); raw["model"].update(layers=1, ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    routing = {"policy": "balanced"}
    baseline = _evaluate(raw, routing=routing, balanced_matmul={"kind": "tiled_matmul", "tile_m": 128, "tile_n": 128, "tile_k": 128})
    mapped = _evaluate(raw, routing=routing)
    comparison = compare_runs({"probes": [mapped]}, {"probes": [baseline]})["probes"][0]
    assert comparison["status"] == "compared" and comparison["semantic_work_unchanged"] and comparison["configuration"] == "explicit_matrix_recipe_axes"
    assert baseline["costs"]["semantic_work"] == mapped["costs"]["semantic_work"]
    before = {row["recipe"]: row for row in baseline["costs"]["top_signatures"]}; after = {row["recipe"]: row for row in mapped["costs"]["top_signatures"]}
    for recipe in ("balanced_grouped_gate_up", "balanced_grouped_down"):
        assert before[recipe]["program"] == "TiledMatmul" and after[recipe]["program"] == "MappedMatmul"
        assert after[recipe]["matrix_groups"] == 4 and after[recipe]["occurrences"] == 1 and after[recipe]["matrix_mapping_keys"]
        assert mapped["balanced_recipes"][recipe]["resolved_program"] == "MappedMatmul" and baseline["balanced_recipes"][recipe]["declared"]["tile_m"] == 128
    assert before["qkv_projection"]["isolated_ms"] == after["qkv_projection"]["isolated_ms"] and "routing" not in raw["runtime"]
    rejected = compare_runs({"probes": [mapped]}, {"probes": [_evaluate(raw)]})["probes"][0]
    assert rejected["status"] == "unpaired" and "difference_ms" not in rejected
def test_model_specific_prefill_geometry_produces_actual_equal_expert_groups():
    probes = representative_probes(model_prefill_tokens={"kimi-k3": 252, "glm53": 252}, model_decode_sequences={"glm53": 2})
    selected = [probe for probe in probes if probe["case_id"] in {"qwen35", "kimi-k3", "glm53", "v41-dp4"} and (probe["geometry"]["phase"] == "prefill" or probe["id"] == "glm53-decode")]
    assert selected
    for probe in selected:
        raw = json.loads((ROOT / probe["case_path"]).read_text()); raw["runtime"]["routing"] = {"policy": "balanced"}
        groups = [op for ops in target_graphs(raw, **probe["geometry"]).values() for op in ops if op.family == "grouped_gemm"]
        assert groups
        for op in groups:
            routing = dict(op.metadata)["routing_realization"]
            assert len(set(routing["global_expert_rows"]) - {0}) == 1
            assert sum(routing["local_expert_rows"]) == op.local_shape[0] and dict(op.metadata)["matrix_groups"] == routing["matrix_groups"]
    assert all(probe["geometry"]["tokens"] == 1 for probe in probes if probe["kind"] == "catalog_model" and probe["geometry"]["phase"] == "decode")
    assert next(probe for probe in probes if probe["id"] == "source-grouped-prefill")["geometry"]["tokens"] == 4
def test_capacity_mapping_diagnostics_survive_occurrence_weighting_and_source_selection():
    raw = _control(); raw["model"]["layers"] = 3; result = _evaluate(raw, matmul={"kind": "mapped_matmul"})
    costs = result["costs"]; assert costs["semantic_work"] == _evaluate(raw)["costs"]["semantic_work"]
    mappings = {item["key"]: item for item in costs["matrix_mappings"]}
    assert sum(item["invocations"] for item in mappings.values()) == costs["matrix_mapping_coverage"]["recorded_invocations"] and costs["matrix_mapping_coverage"]["unrecorded_invocations"] == 0
    mapped = [row for row in costs["top_signatures"] if row["matrix_mapping_keys"]]
    assert sum(row["occurrences"] for row in mapped) == sum(item["invocations"] for item in mappings.values())
    for row in mapped:
        diagnostic = mappings[row["matrix_mapping_keys"][0]]["diagnostic"]
        assert diagnostic["shape"] == row["local_shape"] and any(not item["exceeded_pools"] for item in diagnostic["candidates"]) and diagnostic["selected"]["tile_m"] > 0
    source = [row for row in costs["top_signatures"] if row["source_boundary"] == "kernel"]
    assert source and all(not row["matrix_mapping_keys"] for row in source)
def test_mla_batched_matrices_keep_group_geometry_without_rebinding_source_attention():
    raw = {
        "machine": {"name": "synthetic-mla-decode-source-control", "backend": "ascend", "sku": "Ascend910_9362", "num_devices": 1, "ranks_per_node": 1},
        "model": {
            "name": "geometry", "layers": 1, "hidden_size": 32, "num_heads": 16,
            "num_kv_heads": 1, "head_dim": 192, "intermediate_size": 64, "vocab_size": 128,
            "attention": "mla",
            "mla": {"q_lora_rank": 32, "kv_lora_rank": 512, "qk_nope_head_dim": 128, "qk_rope_head_dim": 64, "v_head_dim": 128},
        },
        "workload": {"requests": [{"id": "r", "prompt_tokens": 4, "output_tokens": 3}]},
        "runtime": {
            "dp": 1, "tp": 1, "pp": 1, "ep": 1, "max_num_batched_tokens": 16,
            "kernels": {"lowerings": {"paged_mla_attention": {"implementation": "ascend_paged_mla_decode_arch22"}}},
        },
    }
    geometry = {"phase": "decode", "tokens": 1, "context": 512, "sequences": 1, "scope": "model"}
    baseline = _evaluate(raw, geometry=geometry); mapped = _evaluate(raw, geometry=geometry, matmul={"kind": "mapped_matmul"}, level="graph")
    assert mapped["costs"]["semantic_work"] == baseline["costs"]["semantic_work"]
    rows = {row["recipe"]: row for row in mapped["costs"]["top_signatures"]}
    assert rows["q_absorb"]["matrix_groups"] == rows["v_expand"]["matrix_groups"] == 16
    assert rows["q_absorb"]["local_shape"] == [16, 512, 128] and rows["v_expand"]["local_shape"] == [16, 128, 512]
    before = {row["recipe"]: row for row in baseline["costs"]["top_signatures"] if row["source_boundary"] == "kernel"}
    source = {key: row for key, row in rows.items() if row["source_boundary"] == "kernel"}
    assert source.keys() == before.keys() and source
    for recipe, row in source.items():
        assert not row["matrix_mapping_keys"] and row["isolated_ms"] == pytest.approx(before[recipe]["isolated_ms"])
    assert mapped["graph"]["live_tasks"] == mapped["graph"]["live_programs"] == 0
def test_grouped_quantized_inputs_keep_real_producers_outside_dense_mapping_domain():
    raw = json.loads((ROOT / "examples/cases/qwen35-a5.json").read_text()); geometry = {"phase": "decode", "tokens": 1, "context": 16, "sequences": 1, "scope": "model"}
    baseline = _evaluate(raw, geometry=geometry, max_operators=2000); mapped = _evaluate(raw, geometry=geometry, matmul={"kind": "mapped_matmul"}, max_operators=2000)
    assert mapped["costs"]["semantic_work"] == baseline["costs"]["semantic_work"]
    for ops in target_graphs(raw, **geometry).values():
        by_id = {op.op_id: op for op in ops}; groups = [op for op in ops if op.family == "grouped_gemm" and dict(op.metadata).get("activation_scale_bytes", 0) > 0]; assert groups
        for op in groups:
            meta = dict(op.metadata); producer = by_id[op.deps[0]]; elements = meta["expected_rows"] * op.local_shape[-1]
            assert producer.family == "quantize"
            assert producer.mte2_bytes == pytest.approx(elements * dtype_bytes(producer.dtype))
            assert producer.mte3_bytes == pytest.approx(elements * dtype_bytes(meta["activation_format"]) + meta["activation_scale_bytes"])
    grouped = [row for row in mapped["costs"]["top_signatures"] if row["family"] == "grouped_gemm"]
    assert grouped and all(row["matrix_mapping_keys"] for row in grouped)
    assert any(row["diagnostic"].get("execution_abstraction") == "expected_expert_bank_grid" for row in mapped["costs"]["matrix_mappings"])
    assert all(row["matrix_mapping_domain"] == "aggregate_experts" for row in grouped)
def test_changed_geometry_or_effective_configuration_cannot_be_compared_as_an_improvement():
    raw = _control(); baseline = _evaluate(raw)
    changed_geometry = _evaluate(raw, geometry={"phase": "prefill", "tokens": 5, "context": 0, "sequences": 1, "scope": "model"})
    changed_case = deepcopy(raw); changed_case["model"]["intermediate_size"] *= 2
    for plan in changed_case["model"].get("layer_plans", ()):
        plan["intermediate_size"] *= 2
    for result in (changed_geometry, _evaluate(changed_case)):
        comparison = compare_runs({"probes": [result]}, {"probes": [baseline]})["probes"][0]
        assert comparison["status"] == "unpaired" and "difference_ms" not in comparison
def test_subset_run_preserves_unexecuted_reference_entry_in_comparison():
    raw = _control(); baseline = _evaluate(raw)
    longer = _evaluate(raw, geometry={"phase": "prefill", "tokens": 5, "context": 0, "sequences": 1, "scope": "model"}); longer["id"] = "longer"
    report = compare_runs({"probes": [baseline]}, {"probes": [baseline, longer]})
    missing = next(row for row in report["probes"] if row["probe"] == "longer")
    assert missing["status"] == "unpaired" and missing["reason"] == "missing_in_current" and "difference_ms" not in missing
    assert report["coverage"] == {"reference_probes": 2, "current_probes": 1, "compared": 1, "missing_in_current": ["longer"], "missing_in_reference": []}
def test_expert_cohort_and_graph_span_remain_distinct_from_isolated_sum():
    raw = case(tp=1, dp=2, ep=2, prompt=4, output=1); raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    isolated = _evaluate(raw); composed = _evaluate(raw, level="graph")
    assert isolated["replicas"] == composed["replicas"] == [0, 1] and {row["replica"] for row in isolated["costs"]["top_signatures"]} == {0, 1}
    assert isolated["costs"]["operator_sum_ms"] == pytest.approx(composed["costs"]["operator_sum_ms"])
    assert composed["graph"]["boundary"] == "device_graph" and composed["graph"]["duration_ms"] != composed["costs"]["operator_sum_ms"]
    assert composed["graph"]["live_tasks"] == composed["graph"]["live_programs"] == 0
    assert composed["graph"]["completed_tasks"] > composed["graph"]["peak_live_tasks"] > 0
    assert all(abs(amount) < 1e-8 for amount in composed["graph"]["live_allocated"].values())
def test_explicit_priority_runs_named_entries_first_without_duplicating_family_selection():
    assert [probe["id"] for probe in select_probes(representative_probes(), ["v41-dp4-decode", "qwen35", "v41-dp4"])] == ["v41-dp4-decode", "qwen35-prefill", "qwen35-decode", "v41-dp4-prefill"]
def test_real_retained_tensor_contracts_do_not_double_count_ambiguous_variants():
    prediction = json.loads((ROOT / "tests/fixtures/verification/qwen38-model-tensor-contracts.json").read_text())
    observation = json.loads((ROOT / "tests/fixtures/verification/qwen38-observed-gemm-contracts.json").read_text()); original = deepcopy(observation)
    report = retained_gemm_impact(prediction, observation)
    pair = next(row for row in report["ranked_pairs"] if row["recipe"] == "dense_gate_up")
    op = next(row for row in prediction["operators"] if row["operator"]["recipe"] == "dense_gate_up")
    assert pair["signed_residual_ms"] == pytest.approx(op["occurrences"] * (op["duration_ms"] - pair["observed_median_per_call_ms"]))
    assert observation == original
    prediction["operators"].append(deepcopy(op))
    ambiguous = retained_gemm_impact(prediction, observation)
    assert not any(row["recipe"] == "dense_gate_up" for row in ambiguous["ranked_pairs"])
    assert sum(row["recipe"] == "dense_gate_up" for row in ambiguous["unpaired_or_ambiguous_gemm"]) == 2

# --------------------------------------------------------------------------- modeling priority
def test_cost_ranking_uses_all_occurrences_and_keeps_variant_references():
    def row(family, recipe, calls, duration):
        return {"operator": {"family": family, "recipe": recipe}, "occurrences": calls, "total_ms": calls * duration}
    report = {"boundary": "sum_of_isolated_device_programs", "operator_sum_ms": 25, "operator_occurrences": 104, "operators": [row("gemm", "projection", 2, 3), row("norm", "rms", 100, .1), row("gemm", "projection", 1, 4), row("attention", "online", 1, 5)]}
    before = deepcopy(report); ranked = cost_ranking(report)["ranked_families"]
    assert [row["total_ms"] for row in ranked] == [10, 10, 5] and ranked[0]["operator_indices"] == [0, 2] and ranked[0]["occurrences"] == 3
    assert ranked[1]["occurrences"] == 100 and [row["share_percent"] for row in ranked] == [40, 40, 20]
    assert [row["cumulative_share_percent"] for row in ranked] == [40, 80, 100] and report == before
def test_large_percentage_on_tiny_operator_does_not_outrank_repeated_residual():
    report = _comparison([_pair("large_under", 1000, 1100, 10), _pair("large_over", 1100, 1000, 10), _pair("small", 1, 2, 20)], 25)
    before = deepcopy(report); result = residual_ranking(report)
    assert [row["recipe"] for row in result["ranked_pairs"]] == ["large_under", "large_over", "small"]
    assert result["ranked_pairs"][-1]["operator_error_percent"] == -50
    assert result["signed_residual_ms"] == pytest.approx(-.02) and result["absolute_residual_ms"] == pytest.approx(2.02)
    assert result["absolute_residual_percent_of_reference"] == pytest.approx(8.08) and result["unpaired_or_ambiguous_predicted_ms"] == pytest.approx(3.98)
    assert report == before
def test_ambiguous_contracts_and_missing_observations_do_not_become_zero_error():
    ambiguous = [_pair("same_op", 1000, 1100, 10, capture_calls=5, contract_capture_calls=10), _pair("same_op", 1000, 1200, 10, capture_calls=5, contract_capture_calls=10), _pair("branch_a", 2000, 2100, 6, contract_model_calls=10), _pair("branch_b", 3000, 2100, 4, contract_model_calls=10)]
    result = residual_ranking(_comparison(ambiguous, 34))
    assert result["one_to_one_pairs"] == 0 and result["ambiguous_pairs_excluded_from_sum"] == 4
    assert result["unpaired_or_ambiguous_predicted_ms"] == 34 and result["signed_residual_ms"] is None and result["absolute_residual_ms"] is None
    assert len(result["ranked_pairs"]) == 4
    empty = residual_ranking(_comparison([], 0))
    assert empty["absolute_residual_ms"] is None and empty["paired_predicted_share_percent"] is None

# --------------------------------------------------------------------------- operator cohort screening
def test_isolated_screening_counts_repeated_joint_cohort_once_per_source():
    raw = cohort_case(3)
    graphs = {rank: tuple(replace(exchange(volume, axis="ep"), op_id=f"call{index}") for index in range(2)) for rank, volume in enumerate((0, 4, 8))}
    isolated = operator_costs(raw, graphs, input_ready_ms=7); graph = graph_costs(raw, graphs)
    assert isolated["operator_sum_ms"] == graph["isolated"]["operator_sum_ms"] == 40 and isolated["operator_occurrences"] == 6
    assert len(isolated["collective_cohorts"]) == 1 and isolated["collective_cohorts"][0]["occurrences"] == 2
    assert sum(sum(row["serviced"].values()) * row["occurrences"] for row in isolated["operators"]) == 24
    assert [row["communication"]["wire_bytes"] for row in isolated["operators"]] == [0, 4, 8]
    assert all(not row["service_before_input"] and row["live_tasks"] == 0 for row in isolated["operators"])
    graphs[2] = graphs[2][:1]
    with pytest.raises(ValueError, match="collective_participants_mismatch:call1"):
        operator_costs(raw, graphs)
def test_public_study_and_operators_cli_share_complete_frontend_cohort(tmp_path):
    path = ROOT / "examples/cases/balanced-experts-control.json"; raw = json.loads(path.read_text())
    result = study([{"id": "balanced", "case": raw}], [{"id": "prefill", "phase": "prefill", "tokens": 4}])
    evaluation = result["evaluations"][0]; graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    output = tmp_path / "operators.json"
    assert result["status"] == "completed" and main(["operators", str(path), "--phase", "prefill", "--tokens", "4", "--scope", "model", "-o", str(output)]) == 0
    cli = json.loads(output.read_text())
    assert evaluation["duration_ms"] == pytest.approx(graph["isolated"]["operator_sum_ms"]) and cli["operator_sum_ms"] == pytest.approx(evaluation["duration_ms"])
    assert cli["replicas"] == evaluation["replicas"] == [0, 1] and cli["operator_occurrences"] == graph["operator_occurrences"] and evaluation["graph"] is None
    assert any(row["program"] == "PairwiseCollective" for row in cli["operators"])
    assert sum(row["communication"]["wire_bytes"] * row["occurrences"] for row in cli["operators"] if row["program"] == "PairwiseCollective") == 1536
def test_study_does_not_rank_different_transport_isolation_scopes_as_equal_boundaries():
    raw = json.loads((ROOT / "examples/cases/balanced-experts-control.json").read_text())
    transfers = {"kind": "transfer"}
    candidates = [{"id": "causal", "case": raw}, {"id": "per-replica", "case": raw, "overrides": {"runtime": {"kernels": {"recipes": {"expert_dispatch": transfers, "expert_combine": transfers}}}}}, {"id": "expert_dispatch", "case": raw, "overrides": {"runtime": {"kernels": {"recipes": {"expert_dispatch": transfers}}}}}, {"id": "expert_combine", "case": raw, "overrides": {"runtime": {"kernels": {"recipes": {"expert_combine": transfers}}}}}]
    probes = [{"id": "p4", "phase": "prefill", "tokens": 4}]; isolated = study(candidates, probes)
    assert isolated["status"] == "completed" and {row["operator_isolation"] for row in isolated["evaluations"]} == {"per_replica", "complete_cohort", "mixed"}
    assert len(isolated["rankings"]) == 4 and all(row["status"] == "single_candidate" for row in isolated["rankings"])
    graph = study(candidates, probes, level="graph")
    assert graph["status"] == "completed" and len(graph["rankings"]) == 1 and graph["rankings"][0]["status"] == "ranked"
    assert len(graph["rankings"][0]["candidates"]) == 4 and all(row["operator_isolation"] is None for row in graph["evaluations"])

# --------------------------------------------------------------------------- candidate identity
def _candidate_id(raw):
    return compute_sweep_candidate_id(parse_case(raw), ParallelSpec(tp=1, dp=1, ep=1, pp=1), cand_devices=1, ranks_per_node=1, phase="decode", scope="body", context=8, q_tokens=1, emit_tokens=1.0, output_budget=2, sla_ms=50.0, max_sequences_per_dp=1, lock_parallel=False)
def test_candidate_id_separates_memory_budget_and_runtime_bindings():
    raw = compose_case("qwen3-32b", hardware="a3", tp=1, ep=1, num_devices=1, layers=2, prompt_tokens=8, output_tokens=2, concurrency=1); baseline = _candidate_id(raw)
    tiny = deepcopy(raw); tiny["runtime"]["gpu_memory_utilization"] = 0.001
    rebatched = deepcopy(raw); rebatched["runtime"]["kv_block_size"] = 64
    rebound = deepcopy(raw); rebound["runtime"]["kernels"] = {"lowerings": {"mla_head_padding": "vllm_ascend_mla_head_padding"}}
    assert len({baseline, _candidate_id(tiny), _candidate_id(rebatched), _candidate_id(rebound)}) == 4 and _candidate_id(raw) == baseline

# --------------------------------------------------------------------------- SKU comparison and perf gate
def _backend_case(case_dict, *, context=256, tokens=1, sequences=1):
    return analyze_network_efficiency(case_dict, phase="decode", tokens=tokens, context=context, sequences=sequences, scope="body")
def _resident_mib():
    try:
        with open("/proc/self/statm", encoding="utf-8") as handle:
            return int(handle.read().split()[1]) * 4096 / MIB
    except (OSError, IndexError, ValueError):
        return 0.0
def _fifo_frontier(*, deferred):
    engine = Engine([Resource("fifo", 1.0, "ms", "fifo"), Resource("indep", 1.0, "ms")])
    ready_a = engine.submit("a:ready", release=1.0); ready_b = engine.submit("b:ready", release=0.0)
    program_a = KernelProgram((Stage("a1", {"fifo": 1.0}), Stage("a2", {"indep": 100.0}, after=("a1",))))
    program_b = KernelProgram((Stage("b1", {"indep": 1.0}), Stage("b2", {"fifo": 10.0}, after=("b1",))))
    finished = {}
    def admit(event, program, name):
        finished[name] = program.start(engine, name, event)
    for program, ready, name in ((program_a, ready_a, "a"), (program_b, ready_b, "b")):
        if deferred:
            engine.submit(name + ":begin", after=(ready,), native_hook=True, on_complete=lambda event, p=program, n=name: admit(event, p, n))
        else:
            finished[name] = program.start(engine, name, ready)
    engine.advance(horizon=1000.0)
    return max(event.time for event in finished.values())
def test_fifo_admission_order_is_exact_under_the_eager_optimization(monkeypatch):
    assert _fifo_frontier(deferred=True) == 112.0 and _fifo_frontier(deferred=False) == 102.0
    from mindie_sim.prediction import execution as execution_module
    case_dict = compose_case(LONG_MODEL, hardware="a5", tp=8, ep=8, num_devices=8, layers=4, prompt_tokens=4096, output_tokens=8)
    graphs = target_graphs(case_dict, phase="decode", tokens=1, context=256, sequences=1, scope="body")
    submitted = graph_costs(case_dict, graphs, external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, replay_scope="none")["duration_ms"]
    original = execution_module.Execution._program_uses_fifo
    monkeypatch.setattr(execution_module.Execution, "_program_uses_fifo", lambda self, context, engine: True)
    forced = graph_costs(case_dict, graphs, external_inputs={"analysis_input": 0}, include_submission=True, isolated=False, replay_scope="none")["duration_ms"]
    monkeypatch.setattr(execution_module.Execution, "_program_uses_fifo", original)
    assert abs(submitted - forced) <= abs(forced) * TOL
def test_c11_and_python_engines_agree_on_wide_units_and_many_aiv_frontiers():
    from mindie_sim.prediction.runtime import accel_build
    compiler, reason = accel_build.discover_compiler(); assert compiler is not None, reason
    resources = [Resource(f"r{index}", 100.0 + index, "ms", units=WIDE_UNITS if index == 0 else 1) for index in range(WIDE_RESOURCES)]
    durations = {}; statuses = {}
    for backend in ("python", "c"):
        engine = Engine(resources, backend=backend); events = []
        for index in range(WIDE_RESOURCES):
            units = tuple(range(WIDE_UNITS)) if index == 0 else None
            events.append(engine.submit(f"t{index}", {f"r{index}": 100.0}, placement={f"r{index}": units} if units else None))
        engine.advance(until=tuple(events)); durations[backend] = engine.now; statuses[backend] = engine.accel_status()
    assert durations["c"] == durations["python"] and statuses["c"]["native_steps"] > 0
    assert statuses["python"]["native_steps"] == 0 and statuses["python"]["accel_backend"] == "python"
@pytest.mark.parametrize("layout", ({"tp": 8, "ep": 8, "num_devices": 8}, {"tp": 4, "dp": 4, "ep": 16, "num_devices": 16}))
def test_real_graph_physics_is_backend_independent(monkeypatch, layout):
    from mindie_sim.prediction.runtime import accel_build
    compiler, reason = accel_build.discover_compiler(); assert compiler is not None, reason
    case_dict = compose_case(LONG_MODEL, hardware="a5", layers=LONG_LAYERS, **layout)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "python"); reference = _backend_case(case_dict, context=4096)
    monkeypatch.setenv("MINDIE_SIM_ACCEL", "c"); native = _backend_case(case_dict, context=4096)
    assert abs(native["graph_duration_ms"] - reference["graph_duration_ms"]) / max(abs(reference["graph_duration_ms"]), TOL) < TOL
    assert int(native["accel_status"]["native_steps"]) >= 0 and native["accel_status"]["accel_backend"] in {"c_native", "c_native_ready", "python_fallback"}
def test_two_slot_pipeline_keeps_exact_slot_occupancy():
    from mindie_sim.prediction.program import coalesce_execution_program
    def tile(index):
        return (Stage(f"t{index}.load", {"hbm": 1.0}), Stage(f"t{index}.mma", {"cube": 1.0}, after=(f"t{index}.load",)))
    def makespan(program):
        engine = Engine([Resource("hbm", 1.0), Resource("cube", 1.0)])
        done = program.start(engine, "pipe", engine.submit("input")); engine.advance(until=(done,))
        return done.time
    two_slot = PipelineProgram(count=4, slots=2, tile=tile); reference = makespan(two_slot)
    assert two_slot.slots == 2 and reference > 0.0 and makespan(coalesce_execution_program(two_slot)) == reference
    single_slot = PipelineProgram(count=4, slots=1, tile=tile); flattened = coalesce_execution_program(single_slot)
    assert isinstance(flattened, KernelProgram) and makespan(flattened) == makespan(single_slot)
def test_physical_key_and_layer_runs_collapse_without_false_merges():
    case_dict = compose_case(LONG_MODEL, hardware="a5", tp=8, ep=8, num_devices=8)
    ops = target_graphs(case_dict, phase="decode", tokens=1, context=4096, sequences=1, scope="body")[0]
    signatures = {}
    for op in ops:
        signature = (op.family, op.flops, op.hbm_bytes, op.mte2_bytes, op.mte3_bytes, op.vector_ops, op.link_bytes, op.local_shape)
        assert signatures.setdefault(physical_key(op), signature) == signature
    runs = homogeneous_layer_runs(ops)
    assert len(signatures) <= LONG_LAYERS * 4 and any(run.count >= 3 for run in runs) and sum(run.count for run in runs) >= 30
def test_attnres_layers_never_fold_across_distinct_valid_blocks():
    case_dict = compose_case(SECOND_MODEL, hardware="a5", tp=8, ep=8, num_devices=8, layers=20)
    ops = target_graphs(case_dict, phase="decode", tokens=1, context=256, sequences=1, scope="body")[0]
    block_size = int(case_dict["model"].get("residual_block_size") or 8)
    for run in homogeneous_layer_runs(ops):
        assert len({layer // block_size for layer in run.layers}) == 1
def test_hardware_mutation_always_re_executes_a_fresh_engine():
    case_dict = compose_case(LONG_MODEL, hardware="a5", tp=8, ep=8, num_devices=8, layers=8)
    baseline = _backend_case(case_dict, context=1024)
    scaled = analyze_network_efficiency(scale_case_hardware(parse_case(case_dict), hbm_bw_scale=2.0), phase="decode", tokens=1, context=1024, sequences=1, scope="body")
    assert scaled["graph_duration_ms"] != baseline["graph_duration_ms"]
    assert int(scaled["accel_status"]["native_steps"]) + int(scaled["accel_status"]["python_steps"]) > 0
    assert _backend_case(case_dict, context=1024)["graph_duration_ms"] == baseline["graph_duration_ms"]
def test_long_sequence_dynamic_efficiency_is_seconds_scale():
    case_dict = compose_case(SECOND_MODEL, hardware="a5", tp=1, dp=8, ep=8, num_devices=8, layers=93, moe_comm_mode="dynamic", multistream_overlap_shared_expert=False)
    import time
    started = time.perf_counter(); report = _backend_case(case_dict, context=4096); elapsed = time.perf_counter() - started
    assert 0.0 < float(report["graph_duration_ms"]) and elapsed < SECONDS_BUDGET
def test_compare_skus_matches_standalone_solvers_and_stays_seconds_scale():
    import time
    started = time.perf_counter()
    result = compare_skus(LONG_MODEL, ["a3", "a5"], tp=8, ep=8, num_devices=8, layers=LONG_LAYERS, sla_ms=50.0, sensitivity="baseline", jobs=1, emit_advisory=False)
    elapsed = time.perf_counter() - started
    assert len(result["skus"]) == 2 and result["comparisons"][0]["step_speedup_x"] is not None
    assert "execution_backend" in result["runtime_environment"] and "offload_recommended" in result["agent_advisory"] and result["runtime_environment"]["wall_ms"] > 0.0
    geometry = result["geometry"]
    standalone = analyze_network_efficiency(compose_case(LONG_MODEL, hardware="a3", tp=8, ep=8, num_devices=8, layers=LONG_LAYERS), phase="decode", tokens=1, context=geometry["context"], sequences=geometry["sequences_per_dp"], scope="body")
    assert abs(result["skus"][0]["metrics"]["step_ms"] - standalone["graph_duration_ms"]) / max(abs(standalone["graph_duration_ms"]), TOL) < TOL
    reference_case = compose_case(LONG_MODEL, hardware="a3", tp=8, ep=8, num_devices=8, layers=LONG_LAYERS)
    standalone_sla = solve_sla_sizing(reference_case, sla_ms=50.0, context=geometry["context"], tokens_per_seq=1, max_sequences_per_dp=256, phase="decode", scope="body", sensitivity="baseline", jobs=1)
    assert result["skus"][0]["sla_scenarios"][0]["b_sla"] == standalone_sla["scenarios"][0]["b_sla"] and elapsed < COMPARE_BUDGET
def test_serving_prediction_records_wall_time_coverage_and_peak_rss():
    case_dict = compose_case(LONG_MODEL, hardware="a5", tp=8, ep=8, num_devices=8, layers=LONG_LAYERS, output_tokens=8, concurrency=1, prompt_tokens=128)
    rss_before = _resident_mib(); prediction = predict(case_dict, scope="serving")
    execution = prediction.get("execution") or {}
    assert execution.get("completed_tasks", 0) > 0 and prediction["execution_status"] and int(execution.get("python_steps") or 0) >= 0
    assert max(0.0, _resident_mib() - rss_before) < 512.0
def test_runtime_advisor_bounds_workers_and_reports_truthfully(monkeypatch):
    assert runtime_advisor.resolve_jobs("auto", task_count=1) == 1 and runtime_advisor.resolve_jobs(8, task_count=3) == 3
    assert runtime_advisor.resolve_jobs("auto", task_count=12) >= 1
    host = runtime_advisor.probe_host_environment(ensure_accel=False)
    assert host["compiler"] is not None or host["compiler_available"] is False
    assert "compiler_reason" in host and runtime_advisor.canonical_engine_backend(None) == "python_fallback"
    assert runtime_advisor.canonical_engine_backend({"native_steps": 5, "python_steps": 0}) == "c11_accel"
    assert runtime_advisor.canonical_engine_backend({"native_steps": 5, "python_steps": 3}) == "c11_accel+python"
    assert runtime_advisor.canonical_engine_backend({"requested": "python", "native_steps": 0}) == "python"
    def forbidden(*_args, **_kwargs):
        raise AssertionError("python backend touched the toolchain")
    monkeypatch.setattr("mindie_sim.prediction.runtime.accel_build.discover_compiler", forbidden)
    status = Engine([Resource("r", 10.0)], backend="python").accel_status()
    assert (status["accel_backend"], status["native_steps"], status["python_steps"]) == ("python", 0, 0)
def test_compare_skus_without_sla_publishes_efficiency_only():
    result = compare_skus(LONG_MODEL, ["a3", "a5"], tp=8, ep=8, num_devices=8, layers=4, sla_ms=None, jobs=1, emit_advisory=False)
    assert result["sla_ms"] is None and result["scenarios_evaluated"] == ["baseline"] and [row["sla_scenarios"] for row in result["skus"]] == [[], []]
    assert len(result["comparisons"]) == 1 and result["comparisons"][0]["step_speedup_x"] is not None
def test_cli_compare_skus_without_sla_skips_the_sla_search():
    args = parser().parse_args(["compare-skus", "--model", LONG_MODEL, "--skus", "a3", "a5", "--layers", "4", "--no-sla", "--jobs", "1", "--no-advisory"])
    result = run(args)
    assert result["sla_ms"] is None and [row["sla_scenarios"] for row in result["skus"]] == [[], []]

# --------------------------------------------------------------------------- candidate studies
def test_study_compares_equal_work_and_conserves_cost_concentration():
    raw = case(); original = deepcopy(raw)
    result = study([{"id": "base", "case": raw}, {"id": "memory", "case": raw, "overrides": {"model": {"name": "different-label"}, "machine": {"device": {"rates": {"memory": 2e9}}}}}], STUDY_PROBES, level="graph", top_families=2)
    assert raw == original and result["status"] == "completed" and result["summary"] == {"modeled": 4, "failed": 0, "candidate_count": 2, "probe_count": 2}
    for probe in STUDY_PROBES:
        base, faster = [row for row in result["evaluations"] if row["probe"] == probe["id"]]
        assert base["operators"]["work"] == faster["operators"]["work"] and base["workload_key"] == faster["workload_key"] and faster["duration_ms"] < base["duration_ms"]
        costs = base["operators"]
        assert sum(row["isolated_ms"] for row in costs["top_families"]) + costs["omitted_family_cost_ms"] == pytest.approx(costs["operator_sum_ms"])
        assert sum(row["isolated_ms"] for row in costs["implementations"]) == pytest.approx(costs["operator_sum_ms"]) and costs["calibration"] == ["none"]
        assert [row["candidate"] for row in next(row for row in result["rankings"] if row["probe"] == probe["id"])["candidates"]] == ["memory", "base"]
def test_operator_screening_counts_every_expert_participant_without_graph_execution():
    raw = case(dp=2, ep=2); raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    row = study([{"id": "ep", "case": raw}], [STUDY_PROBES[1]])["evaluations"][0]
    full = graph_costs(raw, target_graphs(raw, phase="decode", tokens=1, context=16))
    assert row["boundary"] == "sum_of_isolated_device_programs" and row["graph"] is None and row["replicas"] == [0, 1]
    assert row["logical_work"]["sequences_in_cohort"] == 2 and row["operators"]["operator_occurrences"] == full["operator_occurrences"]
    assert row["duration_ms"] == pytest.approx(full["isolated"]["operator_sum_ms"])
def test_rankings_do_not_mix_different_models_or_expert_cohort_workloads():
    raw = case(); raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    result = study([{"id": "base", "case": raw}, {"id": "larger", "case": raw, "overrides": {"model": {"moe_intermediate_size": 128}}}, {"id": "ep", "case": raw, "overrides": {"runtime": {"dp": 2, "ep": 2}, "machine": {"num_devices": 2, "links": [{"name": "fabric", "capacity_bytes_per_ms": 1000, "members": [0, 1], "semantics": "shared"}]}}}], [STUDY_PROBES[1]])
    assert result["status"] == "completed" and len(result["rankings"]) == 3 and all(row["status"] == "single_candidate" for row in result["rankings"])
    assert result["evaluations"][2]["logical_work"]["sequences_in_cohort"] == 2
def test_infeasible_serving_does_not_erase_operator_costs_or_disappear_from_constraints():
    raw = case()
    result = study([{"id": "fits", "case": raw}, {"id": "too-small", "case": raw, "overrides": {"runtime": {"memory": {"workspace_bytes": 2 * 1024**3}}}}], [STUDY_PROBES[0]], serving=True)
    assert all(row["status"] == "modeled" for row in result["evaluations"]) and result["evaluations"][0]["duration_ms"] == result["evaluations"][1]["duration_ms"]
    serving = result["candidates"][1]["serving"]
    assert serving["status"] == "infeasible" and serving["boundary"] == "device_token_commit" and serving["failed_requests"][0]["reason"]
    assert result["candidates"][0]["serving"]["status"] == "modeled"
def test_unsupported_implementation_remains_failed_and_cli_preserves_partial_results(tmp_path):
    raw = case(); case_path = tmp_path / "case.json"; case_path.write_text(json.dumps(raw))
    packet = {"candidates": [{"id": "base", "case": "case.json"}, {"id": "unsupported", "case": "case.json", "overrides": {"runtime": {"kernels": {"lowerings": {"rmsnorm": "not-implemented"}}}}}], "probes": [STUDY_PROBES[0]]}
    manifest, output = tmp_path / "study.json", tmp_path / "results.json"; manifest.write_text(json.dumps(packet))
    assert main(["study", str(manifest), "-o", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "partial" and report["summary"]["modeled"] == report["summary"]["failed"] == 1 and report["evaluations"][1]["error"]
    assert "duration_ms" not in report["evaluations"][1] and report["rankings"][0]["status"] == "single_candidate" and report["rankings"][0]["candidates"][0]["candidate"] == "base"
def test_unknown_study_geometry_is_not_silently_ignored():
    result = study([{"id": "base", "case": case()}], [{"id": "d", "phase": "decode", "tokens": 1, "context_tokens": 128}])
    assert result["status"] == "failed" and result["evaluations"][0]["error"] == "unknown_study_probe_fields:context_tokens" and result["rankings"] == []
def test_parallel_override_applies_to_normalized_case_and_changes_real_sharding():
    raw = parse_case(case()).to_dict()
    result = study([{"id": "tp1", "case": raw}, {"id": "tp2", "case": raw, "overrides": {"runtime": {"tp": 2}, "machine": parse_case(case(tp=2)).machine.to_dict()}}], [STUDY_PROBES[0]])
    assert result["status"] == "completed" and result["candidates"][1]["parallel"]["tp"] == 2
    base, shard = result["evaluations"]
    assert shard["operators"]["work"]["flops"] < base["operators"]["work"]["flops"] and shard["physical_devices"] == 2 and base["workload_key"] == shard["workload_key"]
def test_invalid_geometry_and_incomplete_case_leave_successful_candidates_visible():
    incomplete = case(); incomplete.pop("machine")
    result = study([{"id": "base", "case": case()}, {"id": "incomplete", "case": incomplete}], [STUDY_PROBES[0], {"id": "typo", "phase": "decdoe", "tokens": 1}])
    assert result["status"] == "partial" and result["summary"]["modeled"] == 1 and result["summary"]["failed"] == 3 and len(result["rankings"]) == 1
    assert "prefill_or_decode" in result["evaluations"][1]["error"] and "case_missing_fields:machine" in result["evaluations"][2]["error"]
def test_misspelled_override_never_becomes_an_identical_ranked_candidate():
    result = study([{"id": "base", "case": case()}, {"id": "typo", "case": case(), "override": {"runtime": {"tp": 2}}}], [STUDY_PROBES[0]])
    assert result["status"] == "partial" and result["evaluations"][1]["error"] == "unknown_study_candidate_fields:override" and result["rankings"][0]["status"] == "single_candidate"
def test_unreadable_case_file_is_a_visible_manifest_load_error(tmp_path, capsys):
    manifest = tmp_path / "study.json"; manifest.write_text(json.dumps({"candidates": [{"id": "missing", "case": "not-present.json"}], "probes": [STUDY_PROBES[0]]}))
    output = tmp_path / "result.json"; assert main(["study", str(manifest), "-o", str(output)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["status"] == "failed" and "not-present.json" in captured.err and captured.out == "" and not output.exists()
def test_prefill_parallel_alias_updates_the_normalized_axis_before_merging():
    raw = parse_case(case()).to_dict()
    result = study([{"id": "pcp2", "case": raw, "overrides": {"runtime": {"parallel": {"prefill_context_parallel_size": 2, "context_parallel_plan": "pcp"}}, "machine": parse_case(case(tp=2)).machine.to_dict()}}], [STUDY_PROBES[0]], level="graph")
    assert result["status"] == "completed" and result["candidates"][0]["parallel"]["pcp"] == 2 and result["candidates"][0]["parallel"]["cp_plan"] == "pcp"
    assert result["evaluations"][0]["physical_devices"] == 2
def test_hidden_model_geometry_and_misspelled_runtime_cannot_enter_rankings():
    raw = parse_case(case()).to_dict()
    result = study([{"id": "base", "case": raw}, {"id": "typo", "case": raw, "overrides": {"runtim": {"tp": 2}}}, {"id": "shadowed", "case": raw, "overrides": {"model": {"intermediate_size": 128}}}, {"id": "alias", "case": raw, "overrides": {"model": {"ffn_dim": 128}}}], [STUDY_PROBES[0]])
    assert result["status"] == "partial" and "unknown_field:case.runtim" in result["evaluations"][1]["error"]
    assert "requires_explicit_layer_plans" in result["evaluations"][2]["error"] and "requires_explicit_layer_plans" in result["evaluations"][3]["error"]
    assert len(result["rankings"][0]["candidates"]) == 1

# --------------------------------------------------------------------------- review closure
@pytest.mark.parametrize(("world_size", "anchor_ms"), ((4, 1.440), (8, 0.813)))
def test_direct_mesh_allreduce_matches_silicon_anchor(world_size, anchor_ms):
    row = operator_costs(parse_case(compose_case("qwen3.5-35b-a3b", hardware=DT_SKU, tp=world_size, dp=1, ep=1, pp=1, layers=1, prompt_tokens=8, output_tokens=1, concurrency=1)), (_allreduce_op(world_size),))["operators"][0]
    assert row["program"] == "TopologyCollectiveRecipe" and abs(row["duration_ms"] - anchor_ms) / anchor_ms < 0.10 and row["duration_ms"] < anchor_ms * 1.10
@pytest.mark.parametrize("world_size", (2, 4))
def test_public_link_utilization_uses_matching_physical_capacity(world_size):
    report = analyze_network_efficiency(parse_case(compose_case("qwen3.5-35b-a3b", hardware=DT_SKU, tp=world_size, dp=1, ep=1, pp=1, layers=1, prompt_tokens=8, output_tokens=1, concurrency=1)), phase="prefill", tokens=8, context=0, sequences=1, scope="body")
    utilization = float(report["network_link_util"])
    assert math.isfinite(utilization) and 0.0 <= utilization <= 1.0 + 1e-9
    assert all(0.0 <= op["link_util"] <= 1.0 + 1e-9 for op in report["operators"] if op["link_bytes"] > 0)
def test_locked_sweep_reuses_one_plan_or_rejects_unsupported_comparison():
    case_dict = parse_case(compose_case("qwen3.5-35b-a3b", hardware=DT_SKU, tp=8, dp=1, ep=8, pp=1, layers=1, prompt_tokens=64, output_tokens=2, concurrency=1))
    with pytest.raises(Exception, match="fixed_plan_sweep_not_implemented"):
        sweep_topologies(case_dict, num_devices=8, lock_program=True)
    report = sweep_topologies(case_dict, num_devices=8, lock_parallel=True, lock_program=True, sla_ms=30.0, context=4096)
    rows = report["candidates"]
    assert report["plan_selection"] == "fixed_parallel" and report["program_selection"] == "fixed"
    assert rows and all(row.get("plan_identity") for row in rows) and len({row["plan_identity"] for row in rows}) == 1 and len({row["b_retained"] for row in rows}) == 1
def test_enable_dbo_builds_two_distinct_microbatch_partitions():
    from mindie_sim.public.errors import UnmodeledCapabilityError
    case_dict = parse_case(compose_case("qwen3-30b-a3b", hardware="a5", tp=8, ep=8, num_devices=8, ranks_per_node=8, prompt_tokens=64, output_tokens=2, concurrency=16))
    case_dict = replace(case_dict, runtime=replace(case_dict.runtime, enable_dbo=True))
    views = tuple(RequestView(f"r{idx}", 1, 32, True, "decode") for idx in range(16))
    with pytest.raises(UnmodeledCapabilityError, match="unmodeled_in_op_dbo_pending_stage3"):
        Frontend()(IterationPlan(views), case_dict.model, case_dict.runtime, case_dict.machine)
def test_efficiency_evaluation_is_not_served_from_a_cross_call_result_cache():
    import mindie_sim.prediction.efficiency as efficiency
    assert not hasattr(efficiency, "_STRUCTURAL_MEMO") and not hasattr(efficiency, "_EFFICIENCY_MEMO")
    case_dict = parse_case(compose_case("qwen3.5-35b-a3b", hardware=DT_SKU, tp=8, dp=1, ep=8, pp=1, layers=1, prompt_tokens=32, output_tokens=2, concurrency=1))
    calls, original = [], efficiency._lower_and_run_graph
    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    efficiency._lower_and_run_graph = counting
    try:
        for _ in range(2):
            efficiency.analyze_network_efficiency(case_dict, phase="decode", tokens=1, context=64, sequences=1, scope="body")
    finally:
        efficiency._lower_and_run_graph = original
    assert len(calls) == 2

def test_shared_page_release_async_consumer_retirement_and_grant_wakeups():
    from mindie_sim.prediction.capacity import Memory, MemoryPolicy
    from mindie_sim.prediction.scheduler import Scheduler
    from _support.serving_cases import case
    raw = case(requests=[{"id": f"L{i}", "prompt_tokens": 1, "output_tokens": 5} for i in range(3)])
    raw["runtime"].update(kv_block_size=1, gpu_memory_utilization=1.0, memory={"derive_workspace": False})
    base = parse_case(raw)
    cap = int(Memory(base, MemoryPolicy(derive_workspace=False)).weights[0] + 12288)
    parsed = replace(base, machine=replace(base.machine, dies=tuple(replace(d, memory_bytes=cap) for d in base.machine.dies)))
    memory = Memory(parsed, MemoryPolicy(derive_workspace=False))
    sched = Scheduler(parsed, list(parsed.workload.requests), memory)
    assert memory.retain_shared_pages(0, "oa", (256.0,), physical_key="pfx") and memory.retain_shared_pages(0, "ob", (256.0,), physical_key="pfx")
    assert sched.retain_shared_state("zero", (0.0,)) and sched.release_shared_state("zero") is True
    assert sched.release_shared_state("oa") is True and sched.release_shared_state("ob") is True and sched.release_shared_state("none") is False
    for req in ("L0", "L1", "L2"):
        assert memory.resize(0, req, 32)
    assert sched.hold_state_consumer("L0", "reader", storage="gqa_kv", ranges=((0, 32),), generation=7, event="reader:ev")
    sched.retire_state("L0")
    assert memory.is_retired(0, "L0") and not memory.resize(0, "L3", 32) and sched.release_state_consumer("L0", "wrong") is False
    assert sched.release_state_consumer("L0", "reader") is True and memory.resize(0, "L3", 32)
    assert memory.grant_batch(0, [{"name": "dup", "bytes": 64}, {"name": "dup", "bytes": 64}]) is False
    for req in ("L1", "L2", "L3"):
        memory.release(0, req)
    assert memory.grant_batch(0, [{"name": "g0", "bytes": 4096, "event": "g0:ev"}]) is True
    assert memory.hold_for_async_consumer(0, None, "c0", grant="g0", event="c0:ev") and memory.grant_batch(0, [{"name": "g1", "bytes": 9000}]) is False
    assert memory.release_grant_reader(0, "g0", "c0") is True and sched.release_state_grants(("g0",)) == pytest.approx(4096.0)


def test_regime_physical_rate_guards_duplicate_ids_and_member_mismatches():
    from mindie_sim import graph_costs
    from mindie_sim.prediction.service_regime_domains import validate_service_regime_domains
    from mindie_sim.prediction.service_regime_domains.regimes import RegimePolicy
    from mindie_sim.schema.common import ModelingError
    from _support.kernel_invocation import copy_op
    from _support.operator_cases import gqa_decode_hardware, hardware_case

    with pytest.raises(ModelingError, match="service_regime_requires_unique_id"):
        RegimePolicy([{"id": "dup", "response": {"fixed_us": 10}}, {"id": "dup", "response": {"fixed_us": 20}}])
    mk = lambda rid, lo, hi, **kw: {
        "id": rid, "match": kw.get("match", {"recipe": "copy"}), "coordinates": kw.get("coordinates", {"q": {"path": "semantic_shape.0"}}),
        "domain": {"q": {"min": lo, "max": hi}, **kw.get("extra", {})}, "response": {"kind": "affine", "fixed_us": 8000.0},
        "continuity": {"group": "g", "support": {"q": {"min": 1, "max": 200}}, "outside": "analytic"},
        **({"execution": kw["execution"]} if "execution" in kw else {}),
    }
    with pytest.raises(ModelingError, match="members_differ"):
        validate_service_regime_domains([mk("a", 1, 100, match={"recipe": "copy", "dtype": "bf16"}), mk("b", 100, 200, match={"recipe": "copy", "dtype": "fp16"})])
    joint = {"kind": "joint_native", "resource": "copy_service", "reference_rate": 1e9, "native_kernel": "CopyK"}
    with pytest.raises(ModelingError, match="execution_mode"):
        validate_service_regime_domains([mk("a", 1, 100, execution=joint), mk("b", 100, 200)])
    guard_coords = {"q": {"path": "semantic_shape.0"}, "guard": {"path": "semantic_shape.0"}}
    validate_service_regime_domains([mk("a", 1, 100, coordinates=guard_coords, extra={"guard": {"min": 1}}),
                                     mk("b", 100, 200, coordinates=guard_coords, extra={"guard": {"min": 1}}),
                                     {"id": "ext", "match": {"recipe": "copy"}, "domain": {"q": {"min": 300}}, "response": {"fixed_us": 5}}])
    raw_phys = hardware_case()
    raw_phys.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [{"id": "phys-guard", "match": {"physical_rates.resources.hbm": dict(parse_case(raw_phys).machine.dies[0].rates)["hbm"]}, "coordinates": {"m": {"path": "metadata.absent"}}, "response": {"fixed_us": 100}}]
    assert graph_costs(raw_phys, (copy_op(1024),), isolated=False)["duration_ms"] > 0
    raw_req = hardware_case()
    raw_req.setdefault("runtime", {}).setdefault("kernels", {})["service_regimes"] = [mk("req", 1, 200, coordinates={"q": {"path": "metadata.absent"}})]
    with pytest.raises(ModelingError, match="service_regime_requires_declared_coordinate:req"):
        graph_costs(raw_req, (copy_op(1024),), isolated=False)
