"""Candidate studies preserve work, comparison scopes and visible failures."""
from copy import deepcopy
import json

import pytest

from mindie_sim import graph_costs, study, target_graphs
from mindie_sim.__main__ import main
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


PROBES = [{"id": "prefill", "phase": "prefill", "tokens": 4},
          {"id": "decode", "phase": "decode", "tokens": 1, "context": 16}]


def test_study_compares_equal_work_and_conserves_cost_concentration():
    raw = case()
    original = deepcopy(raw)
    result = study([
        {"id": "base", "case": raw},
        {"id": "memory", "case": raw, "overrides": {
            "model": {"name": "different-label"},
            "machine": {"device": {"rates": {"memory": 2e9}}},
        }},
    ], PROBES, level="graph", top_families=2)
    assert raw == original
    assert result["status"] == "completed"
    assert result["summary"] == {"modeled": 4, "failed": 0, "candidate_count": 2, "probe_count": 2}
    for probe in PROBES:
        base, faster = [row for row in result["evaluations"] if row["probe"] == probe["id"]]
        assert base["operators"]["work"] == faster["operators"]["work"]
        assert base["workload_key"] == faster["workload_key"]
        assert faster["duration_ms"] < base["duration_ms"]
        costs = base["operators"]
        assert sum(row["isolated_ms"] for row in costs["top_families"]) + costs[
            "omitted_family_cost_ms"] == pytest.approx(costs["operator_sum_ms"])
        assert sum(row["isolated_ms"] for row in costs["implementations"]) == pytest.approx(
            costs["operator_sum_ms"])
        assert costs["calibration"] == ["none"]
        ranking = next(row for row in result["rankings"] if row["probe"] == probe["id"])
        assert [row["candidate"] for row in ranking["candidates"]] == ["memory", "base"]


def test_operator_screening_counts_every_expert_participant_without_graph_execution():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    probe = PROBES[1]
    operators = study([{"id": "ep", "case": raw}], [probe])
    row = operators["evaluations"][0]
    full = graph_costs(raw, target_graphs(raw, phase="decode", tokens=1, context=16))
    assert row["boundary"] == "sum_of_isolated_device_programs"
    assert row["graph"] is None
    assert row["replicas"] == [0, 1]
    assert row["logical_work"]["sequences_in_cohort"] == 2
    assert row["operators"]["operator_occurrences"] == full["operator_occurrences"]
    assert row["duration_ms"] == pytest.approx(full["isolated"]["operator_sum_ms"])


def test_rankings_do_not_mix_different_models_or_expert_cohort_workloads():
    raw = case()
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    result = study([
        {"id": "base", "case": raw},
        {"id": "larger", "case": raw, "overrides": {"model": {"moe_intermediate_size": 128}}},
        {"id": "ep", "case": raw, "overrides": {
            "runtime": {"dp": 2, "ep": 2},
            "machine": {"num_devices": 2, "links": [
                {"name": "fabric", "capacity_bytes_per_ms": 1000,
                 "members": [0, 1], "semantics": "shared"}]},
        }},
    ], [PROBES[1]])
    assert result["status"] == "completed"
    assert len(result["rankings"]) == 3
    assert all(row["status"] == "single_candidate" for row in result["rankings"])
    assert result["evaluations"][2]["logical_work"]["sequences_in_cohort"] == 2


def test_infeasible_serving_does_not_erase_operator_costs_or_disappear_from_constraints():
    raw = case()
    result = study([
        {"id": "fits", "case": raw},
        {"id": "too-small", "case": raw,
         "overrides": {"runtime": {"memory": {"workspace_bytes": 2 * 1024**3}}}},
    ], [PROBES[0]], serving=True)
    assert all(row["status"] == "modeled" for row in result["evaluations"])
    assert result["evaluations"][0]["duration_ms"] == result["evaluations"][1]["duration_ms"]
    serving = result["candidates"][1]["serving"]
    assert serving["status"] == "infeasible"
    assert serving["boundary"] == "device_token_commit"
    assert serving["failed_requests"][0]["reason"]
    assert result["candidates"][0]["serving"]["status"] == "modeled"


def test_unsupported_implementation_remains_failed_and_cli_preserves_partial_results(tmp_path):
    raw = case()
    case_path = tmp_path / "case.json"
    case_path.write_text(json.dumps(raw))
    packet = {
        "candidates": [
            {"id": "base", "case": "case.json"},
            {"id": "unsupported", "case": "case.json", "overrides": {
                "runtime": {"kernels": {"lowerings": {"rmsnorm": "not-implemented"}}}}},
        ],
        "probes": [PROBES[0]],
    }
    manifest, output = tmp_path / "study.json", tmp_path / "results.json"
    manifest.write_text(json.dumps(packet))
    assert main(["study", str(manifest), "-o", str(output)]) == 2
    report = json.loads(output.read_text())
    assert report["status"] == "partial"
    assert report["summary"]["modeled"] == report["summary"]["failed"] == 1
    assert report["evaluations"][1]["error"]
    assert "duration_ms" not in report["evaluations"][1]
    assert report["rankings"][0]["status"] == "single_candidate"
    assert report["rankings"][0]["candidates"][0]["candidate"] == "base"


def test_unknown_probe_geometry_is_not_silently_ignored():
    result = study([{"id": "base", "case": case()}],
                   [{"id": "d", "phase": "decode", "tokens": 1, "context_tokens": 128}])
    assert result["status"] == "failed"
    assert result["evaluations"][0]["error"] == "unknown_study_probe_fields:context_tokens"
    assert result["rankings"] == []


def test_parallel_override_applies_to_normalized_case_and_changes_real_sharding():
    raw = parse_case(case()).to_dict()
    result = study([
        {"id": "tp1", "case": raw},
        {"id": "tp2", "case": raw, "overrides": {
            "runtime": {"tp": 2},
            "machine": parse_case(case(tp=2)).machine.to_dict()}},
    ], [PROBES[0]])
    assert result["status"] == "completed"
    assert result["candidates"][1]["parallel"]["tp"] == 2
    base, shard = result["evaluations"]
    assert shard["operators"]["work"]["flops"] < base["operators"]["work"]["flops"]
    assert shard["physical_devices"] == 2
    assert base["workload_key"] == shard["workload_key"]


def test_invalid_geometry_and_incomplete_case_leave_successful_candidates_visible():
    incomplete = case()
    incomplete.pop("machine")
    result = study([
        {"id": "base", "case": case()}, {"id": "incomplete", "case": incomplete},
    ], [PROBES[0], {"id": "typo", "phase": "decdoe", "tokens": 1}])
    assert result["status"] == "partial"
    assert result["summary"]["modeled"] == 1
    assert result["summary"]["failed"] == 3
    assert len(result["rankings"]) == 1
    assert "prefill_or_decode" in result["evaluations"][1]["error"]
    assert "case_missing_fields:machine" in result["evaluations"][2]["error"]


def test_misspelled_override_never_becomes_an_identical_ranked_candidate():
    result = study([
        {"id": "base", "case": case()},
        {"id": "typo", "case": case(), "override": {"runtime": {"tp": 2}}},
    ], [PROBES[0]])
    assert result["status"] == "partial"
    assert result["evaluations"][1]["error"] == "unknown_study_candidate_fields:override"
    assert result["rankings"][0]["status"] == "single_candidate"


def test_unreadable_case_file_is_a_visible_manifest_load_error(tmp_path, capsys):
    manifest = tmp_path / "study.json"
    manifest.write_text(json.dumps({
        "candidates": [{"id": "missing", "case": "not-present.json"}],
        "probes": [PROBES[0]],
    }))
    output = tmp_path / "result.json"
    assert main(["study", str(manifest), "-o", str(output)]) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err)["status"] == "failed"
    assert "not-present.json" in captured.err
    assert captured.out == "" and not output.exists()


def test_prefill_parallel_alias_updates_the_normalized_axis_before_merging():
    raw = parse_case(case()).to_dict()
    result = study([{"id": "pcp2", "case": raw, "overrides": {
        "runtime": {"parallel": {
            "prefill_context_parallel_size": 2, "context_parallel_plan": "pcp"}},
        "machine": parse_case(case(tp=2)).machine.to_dict(),
    }}], [PROBES[0]], level="graph")
    assert result["status"] == "completed"
    assert result["candidates"][0]["parallel"]["pcp"] == 2
    assert result["candidates"][0]["parallel"]["cp_plan"] == "pcp"
    assert result["evaluations"][0]["physical_devices"] == 2


def test_shadowed_model_geometry_and_misspelled_runtime_cannot_enter_rankings():
    raw = parse_case(case()).to_dict()
    result = study([
        {"id": "base", "case": raw},
        {"id": "typo", "case": raw, "overrides": {"runtim": {"tp": 2}}},
        {"id": "shadowed", "case": raw, "overrides": {"model": {"intermediate_size": 128}}},
        {"id": "alias", "case": raw, "overrides": {"model": {"ffn_dim": 128}}},
    ], [PROBES[0]])
    assert result["status"] == "partial"
    assert "unknown_field:case.runtim" in result["evaluations"][1]["error"]
    assert "requires_explicit_layer_plans" in result["evaluations"][2]["error"]
    assert "requires_explicit_layer_plans" in result["evaluations"][3]["error"]
    assert len(result["rankings"][0]["candidates"]) == 1
