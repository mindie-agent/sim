"""Actual Kimi semantic calls keep prediction and captured activity separate."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from tools.model_scoreboard_kimi import (
    comparison_row, prediction_packet, predict_worker, totals,
)


ROOT = Path(__file__).resolve().parents[1]


def retained_inputs():
    observed = json.loads((
        ROOT / "tests/fixtures/verification/kimi-native-descriptors.json"
    ).read_text())
    case = json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text())
    return observed, case


def packet(mode, observed, case):
    return prediction_packet(
        mode, observed, case, {}, context=32768, kda_tokens=8192,
        matrix_recipe="steady_matmul")


def test_situ_multiple_retained_geometries_execute_full_scope_and_keep_residuals():
    observed, case = retained_inputs()
    groups = [
        next(row for row in observed["groups"]
             if row["kind"] == "GroupedMatmulSituQuant"
             and row["descriptors"]["input_shapes"][0][0] == m)
        for m in (2875, 3286)
    ]
    result = predict_worker(packet("situ", {"groups": groups}, case))
    assert len(result["rows"]) == 2
    compared = []
    for source, predicted in zip(groups, result["rows"]):
        value = predicted["prediction"]
        ledger, work = value["boundary_ledger"], value["logical_work"]
        m = source["descriptors"]["input_shapes"][0][0]
        input_bytes = m * (3584 + 112)
        weight_bytes = 28 * 6144 * (1792 + 112)
        gate_bf16, situ_bf16 = m * 6144 * 2, m * 3072 * 2
        output_bytes = m * (3072 + 96)
        expected_read = input_bytes + weight_bytes + gate_bf16 + situ_bf16
        expected_write = gate_bf16 + situ_bf16 + output_bytes
        assert work["flops"] == 2 * m * 6144 * 3584
        assert work["vector_ops"] == 12 * m * 3072 + 4 * m * 3072 + 2 * m * 96
        assert work["hbm_bytes"] == expected_read + expected_write
        assert value["serviced"]["memory@r0:s0"] == pytest.approx(expected_read + expected_write)
        assert value["serviced"]["mte2@r0:s0"] == pytest.approx(expected_read)
        assert value["serviced"]["mte3@r0:s0"] == pytest.approx(expected_write)
        assert value["execution"]["live_tasks"] == value["execution"]["live_programs"] == 0
        assert value["per_call_ms"] > 0
        assert [op["op_id"] for op in ledger["inventory"]["ops"]] == [
            "situ.gate_up", "situ.activation", "situ.down.quantize"]
        assert ledger["group_list"]["weight_banks"] == 28
        comparison = comparison_row(predicted, source, "situ")
        compared.append(comparison)
        assert comparison["boundary_ledger"] == ledger
        assert comparison["logical_work"] == work
        assert comparison["serviced"] == value["serviced"]
        assert comparison["predicted_sum_ms"] == value["per_call_ms"] * source["count"]
        delta = comparison["predicted_sum_ms"] - source["activity_sum_ms"]
        assert comparison["sum_error"]["signed_gap_ms"] == pytest.approx(delta)
        assert comparison["sum_error"]["absolute_gap_ms"] == pytest.approx(abs(delta))
    family = totals(compared, family="GroupedMatmulSituQuant")
    assert family["descriptor_absolute_residual_sum_ms"] == pytest.approx(
        sum(row["sum_error"]["absolute_gap_ms"] for row in compared))


@pytest.mark.parametrize("mode", ["situ", "kda"])
def test_observed_counts_and_timings_never_change_prediction_packet(mode):
    observed, case = retained_inputs()
    before = packet(mode, observed, case)
    changed = deepcopy(observed)
    for group in changed["groups"]:
        group.update(
            count=987654321, activity_sum_ms=123456789, mean_ms=314159,
            p50_ms=271828, min_ms=161803, max_ms=99999999,
            first_source_line=999999, first_name="duration_count_canary")
    after = packet(mode, changed, case)
    assert before == after
    assert all(set(group) == {"id", "kind", "descriptors"} for group in before["groups"])
    if mode == "situ":
        assert len(before["groups"]) == 361


def test_kda_actual_semantic_prediction_keeps_stage_ledger_without_inventing_calls():
    observed, case = retained_inputs()
    source = next(row for row in observed["groups"] if row["kind"] == "ChunkKdaFwd")
    result = predict_worker(packet("kda", {"groups": [source]}, case))
    original = result["rows"][0]
    row = comparison_row(original, source, "kda")
    ledger = row["stage_ledger"]
    assert row["predicted_ms"] > 0
    assert row["status"] == "not_comparable"
    assert row["occurrences"] == 1104
    assert ledger["geometry"]["tokens"] == 8192
    assert ledger["source_condition"]["stage_scalar_present"] is False
    stages = row["stage_predictions"]
    by_stage = {stage["stage"]: stage for stage in stages}
    pairs, tokens, heads, dk, dv = 524288, 8192, 12, 128, 128
    expected = {
        "GATE_PREPARE": 4 * pairs * heads * dk + (2 / 3) * 128 * 64**3 * heads,
        "POST_WU": 2 * pairs * heads * (dk + dv),
        "FWD_H": 4 * tokens * heads * dk * dv,
        "FINALIZE": 2 * tokens * heads * dk * dv + 2 * pairs * heads * dv,
    }
    for name, flops in expected.items():
        stage = by_stage[name]
        assert stage["semantic_work"]["flops"] == pytest.approx(flops)
        assert stage["predicted_semantic_projection_ms"] > 0
        assert stage["observed_stage_ms"] is stage["cumulative_gap_ms"] is None
        assert stage["execution"]["live_tasks"] == stage["execution"]["live_programs"] == 0
    # This core is a serial semantic DAG; disjoint stage projections cover it.
    assert sum(stage["predicted_semantic_projection_ms"] for stage in stages) == pytest.approx(
        row["predicted_ms"])
    ids = [op_id for stage in stages for op_id in stage["semantic_member_ids"]]
    assert len(ids) == len(set(ids)) == 6
    assert row["execution"]["live_tasks"] == row["execution"]["live_programs"] == 0
    assert all(row[key] is None for key in (
        "predicted_sum_ms", "sum_error", "mean_error", "p50_error"))
    changed = deepcopy(source)
    changed["count"] = 1
    second = comparison_row(original, changed, "kda")
    assert second["stage_ledger"] == ledger
    assert second["predicted_ms"] == row["predicted_ms"]
    assert second["predicted_sum_ms"] is None


def test_kda_source_case_is_inherited_and_only_explicit_override_restores_control():
    observed, case = retained_inputs()
    case["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["kda"] = (
        "vllm_ascend_kda_chunk_state")
    original = deepcopy(case)
    selected = predict_worker(packet("kda", observed, case))["rows"][0]["prediction"]
    assert case == original
    assert selected["kda_implementation"] == "vllm_ascend_kda_chunk_state"
    stages = {row["stage"]: row for row in selected["stage_predictions"]}
    state = stages["FWD_H"]
    assert state["native_mapping_status"] == "source_phase_not_capture_binary_identity"
    assert state["semantic_work"]["hbm_bytes"] == 806879232
    assert state["semantic_work"]["flops"] == 6442450944
    assert sum(row["predicted_semantic_projection_ms"] for row in stages.values()) == pytest.approx(
        selected["per_call_ms"])
    control_packet = packet("kda", observed, case)
    control_packet["kda_implementation"] = "vllm_ascend_kda_in_kernel_gates"
    control = predict_worker(control_packet)["rows"][0]["prediction"]
    assert control["kda_implementation"] == "vllm_ascend_kda_in_kernel_gates"
    assert control["per_call_ms"] < selected["per_call_ms"]
    control_stages = {row["stage"]: row for row in control["stage_predictions"]}
    for name in ("GATE_PREPARE", "POST_WU", "FINALIZE"):
        assert stages[name]["semantic_work"] == control_stages[name]["semantic_work"]
        assert stages[name]["predicted_semantic_projection_ms"] == pytest.approx(
            control_stages[name]["predicted_semantic_projection_ms"])
