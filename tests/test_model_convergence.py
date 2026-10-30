"""Cost census and comparison boundaries using real public model executions."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import target_graphs
from mindie_sim.schema.common import dtype_bytes
from tools.model_convergence import (
    compare_runs, fingerprint, representative_probes, retained_gemm_impact, run_probe,
    select_probes,
)
from test_prediction_system import case


ROOT = Path(__file__).resolve().parents[1]


def control():
    return json.loads((ROOT / "examples/source-families/gqa-norm-control.json").read_text())


def evaluate(raw, *, geometry=None, matmul=None, balanced_matmul=None, routing=None,
             level="operators", max_operators=1000):
    geometry = geometry or {
        "phase": "prefill", "tokens": 4, "context": 0, "sequences": 1, "scope": "model",
    }
    result = run_probe({
        "case": raw, "geometry": geometry, "matmul": matmul, "level": level,
        "balanced_matmul": balanced_matmul, "routing": routing,
        "top": 100, "max_operators": max_operators,
    })
    effective_input = deepcopy(raw)
    if routing is not None:
        effective_input.setdefault("runtime", {})["routing"] = routing
    return {
        "id": "control",
        "input_key": fingerprint({"case": effective_input, "geometry": geometry, "level": level}),
        **result,
    }


def test_actual_source_selection_and_repetition_conserve_all_reported_costs():
    raw = control()
    raw["model"]["layers"] = 3
    result = evaluate(raw)
    costs = result["costs"]
    families = {row["recipe"]: row for row in costs["families"]}
    assert families["triton_unified_attention_2d"]["occurrences"] == 3
    assert families["cuda_norm.rms_norm"]["occurrences"] == 7
    for collection in ("families", "programs", "top_signatures"):
        assert sum(row["isolated_ms"] for row in costs[collection]) == pytest.approx(
            costs["operator_sum_ms"])
        assert sum(row["occurrences"] for row in costs[collection]) == costs["operator_occurrences"]
    source = [row for row in costs["programs"] if row["source_boundary"] == "kernel"]
    assert sum(row["occurrences"] for row in source) == 10
    assert costs["remaining_live_tasks"] == costs["remaining_allocations"] == 0
    qkv = next(row for row in costs["top_signatures"] if row["recipe"] == "qkv_projection")
    assert qkv["local_shape"] == [4, 64, 32]


def test_explicit_mapping_change_compares_equal_work_and_preserves_changed_cost():
    raw = control()
    baseline = evaluate(raw)
    modified = evaluate(raw, matmul={
        "kind": "tiled_matmul", "tile_m": 2, "tile_n": 16, "tile_k": 8,
        "reuse_inputs": False, "reuse_weights": False,
    })
    report = compare_runs({"probes": [modified]}, {"probes": [baseline]})
    comparison = report["probes"][0]
    assert comparison["status"] == "compared"
    # The ordinary matmul choice now also supplies balanced defaults. The
    # census exposes both changed bindings even though this fixture is dense.
    assert comparison["configuration"] == "explicit_matrix_recipe_axes"
    assert {item["resolved_program"] for item in modified["balanced_recipes"].values()} == {"TiledMatmul"}
    assert comparison["semantic_work_unchanged"]
    assert not comparison["executed_service_unchanged"]
    assert any(value > 0 for resource, value in comparison["executed_service_delta"].items()
               if resource.startswith("memory@"))
    assert comparison["difference_ms"] != 0
    assert comparison["difference_ms"] == pytest.approx(sum(
        row["difference_ms"] for row in comparison["family_changes"]))
    attention = next(row for row in comparison["family_changes"] if row["family"] == "attention")
    assert attention["difference_ms"] == 0
    assert raw == control()


def test_phase_comparison_keeps_physical_work_separate_from_changed_duration():
    raw = control()
    mapping = {"candidates": [[2, 16, 8]], "compute_m": 2, "compute_n": 8}
    detailed = evaluate(raw, matmul={"kind": "mapped_matmul", **mapping})
    phase = evaluate(raw, matmul={"kind": "phase_matmul", **mapping})
    comparison = compare_runs({"probes": [phase]}, {"probes": [detailed]})["probes"][0]
    assert comparison["semantic_work_unchanged"] and comparison["executed_service_unchanged"]
    assert comparison["executed_service_max_relative_delta"] < 1e-12
    assert comparison["difference_ms"] > 0


def test_balanced_recipe_comparison_executes_distinct_groups_without_changing_work():
    raw = case(prompt=4, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32)
    routing = {"policy": "balanced"}
    baseline = evaluate(raw, routing=routing, balanced_matmul={
        "kind": "tiled_matmul", "tile_m": 128, "tile_n": 128, "tile_k": 128,
    })
    mapped = evaluate(raw, routing=routing)
    comparison = compare_runs({"probes": [mapped]}, {"probes": [baseline]})["probes"][0]
    assert comparison["status"] == "compared" and comparison["semantic_work_unchanged"]
    assert comparison["configuration"] == "explicit_matrix_recipe_axes"
    assert baseline["costs"]["semantic_work"] == mapped["costs"]["semantic_work"]
    before = {row["recipe"]: row for row in baseline["costs"]["top_signatures"]}
    after = {row["recipe"]: row for row in mapped["costs"]["top_signatures"]}
    for recipe in ("balanced_grouped_gate_up", "balanced_grouped_down"):
        assert before[recipe]["program"] == "TiledMatmul"
        assert after[recipe]["program"] == "MappedMatmul"
        assert after[recipe]["matrix_groups"] == 4
        assert after[recipe]["occurrences"] == 1
        assert after[recipe]["matrix_mapping_keys"]
        assert mapped["balanced_recipes"][recipe]["resolved_program"] == "MappedMatmul"
        assert baseline["balanced_recipes"][recipe]["declared"]["tile_m"] == 128
    assert before["qkv_projection"]["isolated_ms"] == after["qkv_projection"]["isolated_ms"]
    assert "routing" not in raw["runtime"]
    probability = evaluate(raw)
    rejected = compare_runs({"probes": [mapped]}, {"probes": [probability]})["probes"][0]
    assert rejected["status"] == "unpaired" and "difference_ms" not in rejected


def test_model_specific_prefill_geometry_produces_actual_equal_expert_groups():
    probes = representative_probes(
        model_prefill_tokens={"kimi-k3": 252, "glm53": 252},
        model_decode_sequences={"glm53": 2})
    for probe in probes:
        if (probe["case_id"] not in {"qwen35", "kimi-k3", "glm53", "v41-dp4"}
                or probe["geometry"]["phase"] != "prefill" and probe["id"] != "glm53-decode"):
            continue
        raw = json.loads((ROOT / probe["case_path"]).read_text())
        raw["runtime"]["routing"] = {"policy": "balanced"}
        graphs = target_graphs(raw, **probe["geometry"])
        groups = [op for ops in graphs.values() for op in ops if op.family == "grouped_gemm"]
        assert groups
        for op in groups:
            metadata = dict(op.metadata)
            routing = metadata["routing_realization"]
            assert len(set(routing["global_expert_rows"]) - {0}) == 1
            assert sum(routing["local_expert_rows"]) == op.local_shape[0]
            assert metadata["matrix_groups"] == routing["matrix_groups"]
    # Per-sequence decode remains one query token, including the explicit
    # two-request GLM batch; source controls retain their own scenarios.
    assert all(probe["geometry"]["tokens"] == 1 for probe in probes
               if probe["kind"] == "catalog_model" and probe["geometry"]["phase"] == "decode")
    assert next(p for p in probes if p["id"] == "source-grouped-prefill")["geometry"]["tokens"] == 4


def test_capacity_mapping_diagnostics_survive_occurrence_weighting_and_source_selection():
    raw = control()
    raw["model"]["layers"] = 3
    result = evaluate(raw, matmul={"kind": "mapped_matmul"})
    baseline = evaluate(raw)
    costs = result["costs"]
    assert costs["semantic_work"] == baseline["costs"]["semantic_work"]
    mappings = {item["key"]: item for item in costs["matrix_mappings"]}
    assert sum(item["invocations"] for item in mappings.values()) == (
        costs["matrix_mapping_coverage"]["recorded_invocations"])
    assert costs["matrix_mapping_coverage"]["unrecorded_invocations"] == 0
    mapped = [row for row in costs["top_signatures"] if row["matrix_mapping_keys"]]
    assert sum(row["occurrences"] for row in mapped) == sum(
        item["invocations"] for item in mappings.values())
    for row in mapped:
        diagnostic = mappings[row["matrix_mapping_keys"][0]]["diagnostic"]
        assert diagnostic["shape"] == row["local_shape"]
        assert any(not item["exceeded_pools"] for item in diagnostic["candidates"])
        assert diagnostic["selected"]["tile_m"] > 0
    source = [row for row in costs["top_signatures"] if row["source_boundary"] == "kernel"]
    assert source and all(not row["matrix_mapping_keys"] for row in source)


def test_mla_batched_matrices_keep_group_geometry_without_rebinding_source_attention():
    raw = json.loads((ROOT / "examples/source-families/mla-decode-control.json").read_text())
    raw["model"]["num_heads"] = 16
    raw["machine"]["device"].update(l0a_bytes=65536, l0b_bytes=65536,
                                     l0c_bytes=262144, l1_bytes=524288)
    geometry = {"phase": "decode", "tokens": 1, "context": 33,
                "sequences": 1, "scope": "model"}
    baseline = evaluate(raw, geometry=geometry)
    mapped = evaluate(raw, geometry=geometry, matmul={"kind": "mapped_matmul"}, level="graph")
    assert mapped["costs"]["semantic_work"] == baseline["costs"]["semantic_work"]
    rows = {row["recipe"]: row for row in mapped["costs"]["top_signatures"]}
    assert rows["q_absorb"]["matrix_groups"] == rows["v_expand"]["matrix_groups"] == 16
    assert rows["q_absorb"]["local_shape"] == [16, 256, 16]
    assert rows["v_expand"]["local_shape"] == [16, 16, 256]
    before = {row["recipe"]: row for row in baseline["costs"]["top_signatures"]
              if row["source_boundary"] == "kernel"}
    source = {key: row for key, row in rows.items() if row["source_boundary"] == "kernel"}
    assert source.keys() == before.keys()
    assert source
    for recipe, row in source.items():
        assert not row["matrix_mapping_keys"]
        assert row["isolated_ms"] == pytest.approx(before[recipe]["isolated_ms"])
    assert mapped["graph"]["live_tasks"] == mapped["graph"]["live_programs"] == 0


def test_grouped_quantized_inputs_keep_real_producers_outside_dense_mapping_domain():
    raw = json.loads((ROOT / "examples/cases/qwen35-a5.json").read_text())
    geometry = {"phase": "decode", "tokens": 1, "context": 16,
                "sequences": 1, "scope": "model"}
    baseline = evaluate(raw, geometry=geometry, max_operators=2000)
    mapped = evaluate(raw, geometry=geometry, matmul={"kind": "mapped_matmul"},
                      max_operators=2000)
    assert mapped["costs"]["semantic_work"] == baseline["costs"]["semantic_work"]
    for ops in target_graphs(raw, **geometry).values():
        by_id = {op.op_id: op for op in ops}
        groups = [op for op in ops if op.family == "grouped_gemm"]
        assert groups
        for op in groups:
            meta = dict(op.metadata)
            producer = by_id[op.deps[0]]
            rows = meta["expected_rows"]
            elements = rows * op.local_shape[-1]
            assert producer.family == "quantize"
            assert producer.mte2_bytes == pytest.approx(elements * dtype_bytes(producer.dtype))
            assert producer.mte3_bytes == pytest.approx(
                elements * dtype_bytes(meta["activation_format"]) + meta["activation_scale_bytes"])
    grouped = [row for row in mapped["costs"]["top_signatures"]
               if row["family"] == "grouped_gemm"]
    assert grouped and all(row["matrix_mapping_keys"] for row in grouped)
    assert any(row["diagnostic"].get("execution_abstraction") == "expected_expert_bank_grid"
               for row in mapped["costs"]["matrix_mappings"])
    assert all(row["matrix_mapping_domain"] == "aggregate_experts" for row in grouped)


def test_changed_geometry_or_effective_configuration_cannot_be_compared_as_a_fix():
    raw = control()
    baseline = evaluate(raw)
    changed_geometry = evaluate(raw, geometry={
        "phase": "prefill", "tokens": 5, "context": 0, "sequences": 1, "scope": "model",
    })
    changed_case = deepcopy(raw)
    changed_case["model"]["intermediate_size"] *= 2
    changed_model = evaluate(changed_case)
    for result in (changed_geometry, changed_model):
        comparison = compare_runs({"probes": [result]}, {"probes": [baseline]})["probes"][0]
        assert comparison["status"] == "unpaired"
        assert "difference_ms" not in comparison


def test_subset_run_preserves_unexecuted_reference_probe_in_comparison():
    raw = control()
    baseline = evaluate(raw)
    longer = evaluate(raw, geometry={
        "phase": "prefill", "tokens": 5, "context": 0, "sequences": 1, "scope": "model",
    })
    longer["id"] = "longer"
    report = compare_runs({"probes": [baseline]}, {"probes": [baseline, longer]})
    missing = next(row for row in report["probes"] if row["probe"] == "longer")
    assert missing["status"] == "unpaired" and missing["reason"] == "missing_in_current"
    assert "difference_ms" not in missing
    assert report["coverage"] == {
        "reference_probes": 2, "current_probes": 1, "compared": 1,
        "missing_in_current": ["longer"], "missing_in_reference": [],
    }


def test_expert_cohort_and_graph_span_remain_distinct_from_isolated_sum():
    raw = case(tp=1, dp=2, ep=2, prompt=4, output=1)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    isolated = evaluate(raw)
    composed = evaluate(raw, level="graph")
    assert isolated["replicas"] == composed["replicas"] == [0, 1]
    assert {row["replica"] for row in isolated["costs"]["top_signatures"]} == {0, 1}
    assert isolated["costs"]["operator_sum_ms"] == pytest.approx(
        composed["costs"]["operator_sum_ms"])
    assert composed["graph"]["boundary"] == "device_graph"
    assert composed["graph"]["duration_ms"] != composed["costs"]["operator_sum_ms"]
    assert composed["graph"]["live_tasks"] == composed["graph"]["live_programs"] == 0
    assert composed["graph"]["completed_tasks"] > composed["graph"]["peak_live_tasks"] > 0
    assert all(abs(amount) < 1e-8 for amount in composed["graph"]["live_allocated"].values())
    usage = composed["worker_usage"]
    if usage["status"] == "measured":
        assert usage["peak_rss_bytes"] > 0
        assert usage["peak_rss_mib"] == usage["peak_rss_bytes"] / 1024**2


def test_explicit_probe_priority_runs_v4_first_without_duplicating_family_selection():
    selected = select_probes(
        representative_probes(), ["v41-dp4-decode", "qwen35", "v41-dp4"])
    assert [probe["id"] for probe in selected] == [
        "v41-dp4-decode", "qwen35-prefill", "qwen35-decode", "v41-dp4-prefill",
    ]


def test_real_retained_tensor_contracts_do_not_double_count_ambiguous_variants():
    prediction = json.loads((
        ROOT / "tests/fixtures/verification/qwen38-model-tensor-contracts.json"
    ).read_text())
    observation = json.loads((ROOT / "tests/fixtures/verification/qwen38-observed-gemm-contracts.json").read_text())
    original_observation = deepcopy(observation)
    report = retained_gemm_impact(prediction, observation)
    pair = next(row for row in report["ranked_pairs"] if row["recipe"] == "dense_gate_up")
    op = next(row for row in prediction["operators"]
              if row["operator"]["recipe"] == "dense_gate_up")
    expected = op["occurrences"] * (op["duration_ms"] - pair["observed_median_per_call_ms"])
    assert pair["signed_residual_ms"] == pytest.approx(expected)
    assert observation == original_observation
    # A second physical variant with the same tensor contract is not a second
    # independent captured kernel median.
    prediction["operators"].append(deepcopy(op))
    ambiguous = retained_gemm_impact(prediction, observation)
    assert not any(row["recipe"] == "dense_gate_up" for row in ambiguous["ranked_pairs"])
    assert sum(row["recipe"] == "dense_gate_up"
               for row in ambiguous["unpaired_or_ambiguous_gemm"]) == 2
