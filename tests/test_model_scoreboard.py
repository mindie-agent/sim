"""Scoreboard arithmetic and boundaries from retained A3 model observations."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from mindie_sim.observations.scoreboard import model_scoreboard, prediction_from_probe


def glm_entry():
    # Raw pf4k_b1.csv.gz: 2,392 positive tasks, one p4096/o1/c1 request.
    # The graph value is the independently executed phase probe retained with
    # this scoreboard; observed event spans never populate model parameters.
    conditions = {
        "model": "GLM-5.3-W4A8", "hardware": "Ascend910_9362",
        "parallel": {"tp": 8, "dp": 1, "ep": 1, "pp": 1},
        "phase": "prefill", "tokens": 4096, "context": 0, "sequences": 1,
        "precision": "W4A8 routed / W8A8 attention", "branch": "no_mtp",
    }
    return {
        "id": "glm53", "observed": {
            "conditions": conditions,
            "step": {"duration_ms": 1680.733399, "boundary": "profiled_task_span",
                     "comparison_scope": "single_prefill_iteration"},
            "groups": [
                {"name": "allreduceAicpuKernel", "duration_ms": 1552.089211},
                {"name": "SparseFlashAttention", "duration_ms": 1029.573832},
            ],
        },
        "predicted": {
            "conditions": deepcopy(conditions),
            "step": {"duration_ms": 786.5277604589537, "boundary": "device_graph",
                     "comparison_scope": "single_prefill_iteration"},
        },
        "boundary_relation": {
            "observed": "profiled_task_span", "predicted": "device_graph",
            "assumptions": ["Observed runtime helpers remain inside the full window."],
        },
        "assumptions": ["Balanced routing and serial matrix phases are explicit approximations."],
    }


def test_real_prefill_gap_uses_step_span_not_overlapping_activity_and_keeps_limits():
    entry = glm_entry()
    report = model_scoreboard([entry])
    pair = report["models"][0]["step_comparison"]
    assert pair["status"] == "conditional"
    assert pair["signed_gap_ms"] == pytest.approx(-894.2056385410463)
    assert pair["absolute_gap_ms"] == pytest.approx(894.2056385410463)
    assert pair["signed_gap_percent"] == pytest.approx(-53.20330036120418)
    assert pair["formal_accuracy"] == "not_assessed"
    # Two overlapping activity totals already exceed the actual whole span.
    assert sum(row["duration_ms"] for row in entry["observed"]["groups"]) > pair["observed_ms"]
    assert "runtime helpers" in pair["assumptions"][-1]
    assert report["models"][0]["predicted_top"] == []


def test_capture_span_unknown_context_and_mtp_branch_cannot_manufacture_step_accuracy():
    mixed = glm_entry()
    mixed["id"] = "mixed"
    mixed["observed"]["capture"] = {"span_ms": 6912.686953, "positive_events": 286988}
    del mixed["observed"]["step"]
    mixed["observed"]["unavailable_reasons"] = ["mixed_prefill_decode_capture_without_step_ids"]
    context = glm_entry()
    context["id"] = "context"
    context["observed"]["conditions"]["context"] = None
    branch = glm_entry()
    branch["id"] = "branch"
    branch["observed"]["conditions"]["branch"] = "speculative_verify_and_draft"
    boundary = glm_entry()
    boundary["id"] = "boundary"
    del boundary["boundary_relation"]
    report = model_scoreboard([mixed, context, branch, boundary])
    assert report["coverage"]["conditional_step_comparisons"] == 0
    assert report["descriptive_residuals"]["absolute_gap_ms"] is None
    reasons = [row["step_comparison"]["reasons"] for row in report["models"]]
    assert reasons[0] == ["mixed_prefill_decode_capture_without_step_ids"]
    assert "unknown_condition:context" in reasons[1]
    assert "condition_mismatch:branch" in reasons[2]
    assert reasons[3] == ["timing_boundary_relation_not_declared"]
    assert all(row["step_comparison"]["absolute_gap_ms"] is None for row in report["models"])


def test_opposite_residuals_keep_absolute_error_and_group_contracts_do_not_match_by_name():
    first, second = glm_entry(), glm_entry()
    second["id"] = "second"
    second["predicted"]["step"]["duration_ms"] = (
        2 * second["observed"]["step"]["duration_ms"]
        - first["predicted"]["step"]["duration_ms"])
    identity = {"shape": [4096, 8, 256], "dtype": "bf16", "scope": "single_prefill"}
    first["observed"]["groups"][1].update(
        comparison_key="attention", identity=identity, boundary="kernel_activity_sum")
    first["predicted"]["groups"] = [{
        "name": "SparseFlashAttention", "comparison_key": "attention",
        "identity": {**identity, "dtype": "fp16"}, "boundary": "kernel_activity_sum",
        "duration_ms": 100,
    }]
    report = model_scoreboard([first, second])
    assert report["descriptive_residuals"]["signed_gap_ms"] == pytest.approx(0, abs=1e-9)
    assert report["descriptive_residuals"]["absolute_gap_ms"] == pytest.approx(1788.4112770820926)
    group = report["models"][0]["group_comparisons"][0]
    assert group["status"] == "incomparable"
    assert group["condition_differences"] == [{"field": "dtype", "observed": "bf16", "predicted": "fp16"}]


def test_tool_writes_a_reviewable_partial_scoreboard_without_erasing_available_measurements(tmp_path):
    path = Path(__file__).parents[1] / "tools/model_scoreboard.py"
    spec = importlib.util.spec_from_file_location("scoreboard_tool", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    available = glm_entry()
    missing = glm_entry()
    missing["id"] = "unpredicted"
    del missing["predicted"]
    manifest = tmp_path / "input.json"
    manifest.write_text(json.dumps({"entries": [available, missing]}))
    output, markdown = tmp_path / "output.json", tmp_path / "output.md"
    assert module.main([str(manifest), "-o", str(output), "--markdown", str(markdown)]) == 0
    report = json.loads(output.read_text())
    assert report["coverage"] == {
        "models": 2, "conditional_step_comparisons": 1, "unavailable_or_incomparable": 1}
    assert report["models"][1]["step_comparison"]["observed_ms"] == 1680.733399
    assert report["models"][1]["step_comparison"]["predicted_ms"] is None
    assert report["manifest"]["size_bytes"] == manifest.stat().st_size


def test_nested_unknown_parallelism_does_not_establish_equal_work():
    entry = glm_entry()
    for side in ("observed", "predicted"):
        entry[side]["conditions"]["parallel"]["tp"] = None
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["status"] == "incomparable"
    assert pair["reasons"] == ["unknown_condition:parallel.tp"]
    assert pair["observed_ms"] == 1680.733399
    assert pair["signed_gap_ms"] is None


def test_group_sum_mean_or_occurrence_scope_mismatch_is_not_a_residual():
    entry = glm_entry()
    observed = {
        "name": "attention", "comparison_key": "attention",
        "identity": {"shape": [4096, 8, 256], "dtype": "bf16"},
        "boundary": "kernel_activity_sum", "scope": "one_prefill",
        "statistic": "sum", "occurrences": 10, "duration_ms": 100,
    }
    entry["observed"]["groups"] = [observed]
    for delta in (
        {"statistic": "mean", "occurrences": 1, "duration_ms": 10},
        {"occurrences": 9}, {"scope": "whole_capture"},
    ):
        entry["predicted"]["groups"] = [{**observed, **delta}]
        group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
        assert group["reason"] == "group_statistical_scope_mismatch"
        assert "signed_gap_ms" not in group
    entry["predicted"]["groups"] = [{**observed, "duration_ms": 90}]
    group = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert group["signed_gap_ms"] == -10


def test_executed_probe_identity_cannot_be_relabelled_to_different_geometry(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from model_convergence import run_probe
    from model_scoreboard import execution_identity

    spec = {
        "case": json.loads((root / "examples/quickstart.json").read_text()),
        "geometry": {"phase": "prefill", "tokens": 2, "context": 0,
                     "sequences": 1, "scope": "model"},
        "level": "graph", "max_operators": 1000, "top": 5,
    }
    probe = run_probe(spec)
    probe["execution_identity"] = execution_identity(spec, probe)
    actual = probe["execution_identity"]["conditions"]
    assert actual["tokens"] == 2
    claimed = {**actual, "tokens": 256}
    predicted = prediction_from_probe(
        probe, conditions=claimed, comparison_scope="single_prefill_iteration")
    entry = glm_entry()
    entry["observed"]["conditions"] = claimed
    entry["predicted"] = predicted
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["status"] == "incomparable"
    assert pair["reasons"] == ["prediction_label_mismatch:tokens"]
    assert pair["predicted_ms"] == probe["graph"]["duration_ms"]
    del probe["execution_identity"]
    missing = prediction_from_probe(
        probe, conditions=actual, comparison_scope="single_prefill_iteration")
    assert missing["identity_issues"] == ["prediction_execution_identity_not_retained"]


def test_semantic_family_gap_is_conditioned_and_never_pairs_different_call_counts():
    entry = glm_entry()
    entry["observed"]["groups"] = [{
        "name": "SparseFlashAttention", "duration_ms": 1029.573832,
        "occurrences": 78, "aggregation": "sum", "boundary": "device_activity_durations",
    }]
    entry["predicted"]["groups"] = [{
        "name": "latent_attention0", "duration_ms": 31.60541090909091,
        "occurrences": 78, "aggregation": "sum", "boundary": "isolated",
    }]
    entry["semantic_family_comparisons"] = [{
        "id": "sparse_attention", "observed_groups": ["SparseFlashAttention"],
        "predicted_groups": ["latent_attention0"],
        "assumptions": ["Same model step and 78 logical attention calls; native work is unverified."],
        "boundary_relation": {
            "observed": "device_activity_durations", "predicted": "isolated",
            "assumptions": ["Activity sum and isolated calls differ in scheduling context."],
        },
    }]
    row = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert row["status"] == "conditional"
    assert row["signed_gap_ms"] == pytest.approx(-997.968421090909)
    assert row["absolute_gap_ms"] == pytest.approx(997.968421090909)
    assert row["comparison_level"] == "semantic_family"
    entry["predicted"]["groups"][0]["occurrences"] = 79
    row = model_scoreboard([entry])["models"][0]["group_comparisons"][0]
    assert row["reason"] == "family_occurrence_count_mismatch"
    assert "signed_gap_ms" not in row


def test_reference_capture_does_not_replace_requested_model_coverage():
    missing = glm_entry()
    missing["id"] = "v41"
    missing["observed"].pop("step")
    missing.pop("predicted")
    reference = glm_entry()
    reference.update(id="v4flash", role="reference")
    report = model_scoreboard([missing, reference])
    assert report["coverage"] == {
        "models": 1, "conditional_step_comparisons": 0, "unavailable_or_incomparable": 1}
    assert report["models"][0]["id"] == "v41"
    assert report["references"][0]["step_comparison"]["status"] == "conditional"


def test_complete_scheduled_speculative_iteration_keeps_draft_sample_and_commit(monkeypatch):
    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from model_scoreboard import execution_identity, scheduled_probe

    case = json.loads((root / "examples/cases/qwen35-a3-tp4.json").read_text())
    case["runtime"].setdefault("kernels", {}).setdefault("programs", {})["matmul"] = {
        "kind": "phase_matmul"}
    spec = {
        "case": case, "driver": "scheduled_iteration", "proposal_tokens": 1,
        "geometry": {"phase": "decode", "tokens": 1, "context": 16,
                     "sequences": 1, "scope": "model"},
        "max_operators": 30000, "top": 5,
    }
    probe = scheduled_probe(spec)
    probe["execution_identity"] = execution_identity(spec, probe)
    assert {"sampling", "acceptance", "token_commit"} <= set(probe["iteration"]["families"])
    recipes = {item["recipe"] for item in probe["costs"]["families"]}
    assert "speculative_acceptance" in recipes
    assert any(recipe.startswith("draft_") for recipe in recipes)
    assert probe["graph"]["boundary"] == "device_iteration"
    assert probe["graph"]["live_tasks"] == probe["graph"]["live_programs"] == 0
    assert probe["execution_identity"]["conditions"]["tokens"] == 2
    assert probe["execution_identity"]["conditions"]["branch"] == "full_speculative_step"
    spec["geometry"]["phase"] = "prefill"
    with pytest.raises(ValueError, match="single_token_decode_model_scope"):
        scheduled_probe(spec)


def test_context_endpoints_keep_unknown_actual_context_without_fabricating_a_median():
    entry = glm_entry()
    entry["observed"]["conditions"]["context"] = None
    entry["observed"]["context_range"] = [16384, 16416]
    scenario = entry["predicted"]
    low, high = deepcopy(scenario), deepcopy(scenario)
    low["conditions"]["context"], high["conditions"]["context"] = 16384, 16416
    high["step"]["duration_ms"] += 10
    entry["predicted"] = {"scenarios": [low, high]}
    report = model_scoreboard([entry])
    pair = report["models"][0]["step_comparison"]
    assert pair["status"] == "conditional_range"
    assert pair["predicted_range_ms"] == [786.5277604589537, 796.5277604589537]
    assert pair["predicted_ms"] is None
    assert report["descriptive_residuals"]["signed_gap_ms"] is None
    entry["observed"]["conditions"]["context"] = 0
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["status"] == "incomparable"
    entry["observed"]["conditions"]["context"] = None
    high["conditions"]["context"] = 16400
    pair = model_scoreboard([entry])["models"][0]["step_comparison"]
    assert pair["status"] == "incomparable"
    assert "signed_gap_range_ms" not in pair


def test_family_ranges_keep_native_grouping_and_failed_endpoint_coverage():
    from mindie_sim.observations.scoreboard import render_markdown

    entry = glm_entry()
    entry["observed"]["conditions"]["context"] = None
    entry["observed"]["context_range"] = [16, 32]
    entry["observed"]["family_groups"] = [{
        "name": "matrix", "duration_ms": 10, "statistic": "mean_of_step_activity_sums",
        "occurrences": 7,
    }, {
        "name": "mixed_communication", "duration_ms": 20,
        "statistic": "mean_of_step_activity_sums", "comparable": False,
        "reason": "native_and_control_wrappers_cannot_be_added_as_link_work",
    }]
    low, high = deepcopy(entry["predicted"]), deepcopy(entry["predicted"])
    for scenario, context, duration in ((low, 16, 4), (high, 32, 6)):
        scenario["conditions"]["context"] = context
        scenario["groups"] = [{
            "name": "dense_gate_up", "family": "gemm", "duration_ms": duration,
            "occurrences": 2, "boundary": "sum_of_isolated_device_programs",
        }, {
            "name": "allreduce", "family": "allreduce", "duration_ms": 3,
            "occurrences": 1, "boundary": "sum_of_isolated_device_programs",
        }]
    entry["predicted"] = {"scenarios": [low, high]}
    entry["family_bindings"] = [{
        "id": name, "observed_group": observed, "predicted_groups": [predicted],
        "assumptions": ["Native calls and analytic programs have different grouping."],
        "boundary_assumptions": ["Activity sums are compared with isolated program sums."],
    } for name, observed, predicted in (
        ("matrix", "matrix", "dense_gate_up"),
        ("collective", "mixed_communication", "allreduce"))]
    report = model_scoreboard([entry])
    rows = {row["comparison_key"]: row for row in report["models"][0]["group_comparisons"]}
    assert rows["matrix"]["predicted_range_ms"] == [4, 6]
    assert rows["matrix"]["signed_gap_range_ms"] == [-6, -4]
    assert rows["matrix"]["absolute_gap_range_ms"] == [4, 6]
    assert rows["collective"]["status"] == "incomparable"
    assert rows["collective"]["predicted_range_ms"] == [3, 3]
    assert rows["collective"]["signed_gap_ms"] is None
    # A real failed endpoint must not look like a complete zero-width interval.
    entry["predicted"]["scenarios"][1] = {
        "conditions": high["conditions"], "status": "failed",
        "unavailable_reasons": ["prediction_timeout:40s"],
    }
    partial = model_scoreboard([entry])
    leader = partial["models"][0]["predicted_top"][0]
    assert leader["scenario_coverage"] == "1/2"
    assert leader["statistic"] == "partial_context_predictions_missing_endpoint"
    assert "| 1/2 |" in render_markdown(partial)
    assert "Context 32: prediction_timeout:40s" in render_markdown(partial)
    assert all(row["signed_gap_ms"] is None
               for row in partial["models"][0]["group_comparisons"])


def test_scheduled_prefill_runs_real_draft_preparation_to_graph_completion(monkeypatch):
    from dataclasses import asdict
    from test_draft_and_media import draft_case

    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from model_scoreboard import execution_identity, scheduled_probe

    case = draft_case().to_dict()
    # DraftSpec.to_dict includes derived dimensions; an explicit input uses its
    # constructor fields, as does the historical MTP1 declaration.
    case["model"]["auxiliary"] = asdict(draft_case().model.auxiliary)
    spec = {
        "case": case, "driver": "scheduled_prefill", "proposal_tokens": 0,
        "draft_layers": 1,
        "geometry": {"phase": "prefill", "tokens": 4, "context": 0,
                     "sequences": 1, "scope": "model"},
        "max_operators": 1000, "top": 5,
    }
    probe = scheduled_probe(spec)
    identity = execution_identity(spec, probe)
    assert probe["iteration"]["phase_occurrences"]["draft_prefill"] > 0
    assert probe["iteration"]["plans"][0]["requests"][0]["draft_prepare"] is True
    assert "acceptance" not in probe["iteration"]["families"]
    assert probe["graph"]["live_tasks"] == probe["graph"]["live_programs"] == 0
    assert identity["conditions"]["branch"] == "target_plus_mtp_prefill"
    assert identity["conditions"]["tokens"] == 4
    spec["draft_layers"] = 3
    with pytest.raises(ValueError, match="draft_layer_count_mismatch"):
        scheduled_probe(spec)


def test_scenario_markdown_keeps_timing_boundary_and_context_specific_qualifiers():
    from mindie_sim.observations.scoreboard import render_markdown

    entry = glm_entry()
    entry["assumptions"] = []
    entry["observed"]["conditions"]["context"] = None
    entry["observed"]["context_range"] = [16, 32]
    low, high = deepcopy(entry["predicted"]), deepcopy(entry["predicted"])
    low["conditions"]["context"], high["conditions"]["context"] = 16, 32
    high["assumptions"] = ["Package sources changed during this probe; before/after hashes are retained."]
    entry["predicted"] = {"scenarios": [low, high]}
    report = model_scoreboard([entry])
    assert report["models"][0]["step_comparison"]["status"] == "conditional_range"
    markdown = render_markdown(report)
    assert "Observed runtime helpers remain inside the full window." in markdown
    assert "Context 32: Package sources changed during this probe" in markdown


def test_kimi_selected_mapping_changes_schedule_without_measurement_or_work_changes(monkeypatch):
    from mindie_sim.catalog.types import QuantSpec
    from mindie_sim.prediction.structure import parse_case

    root = Path(__file__).parents[1]
    monkeypatch.syspath_prepend(str(root / "tools"))
    from model_scoreboard_kimi import QUANT, grouped_predictions, prediction_packet

    # Retained native group0: one FP8 8192x7168 times 7168x4608 matrix.
    group = {
        "id": "group0", "kind": "GroupedMatmul", "count": 276,
        "activity_sum_ms": 175.481381,
        "descriptors": {
            "input_shapes": [[8192, 7168], [1, 144, 448, 16, 32], [],
                             [1, 112, 4608, 2], [], [], [], [1], [8192, 112, 2]],
            "output_shapes": [[8192, 4608]],
            "input_dtypes": ["DT_FLOAT8_E4M3FN", "DT_FLOAT8_E4M3FN"],
            "output_dtypes": ["DT_BF16"],
        },
    }
    case_data = json.loads((root / "examples/cases/kimi-k3-a5.json").read_text())
    packet = prediction_packet("grouped", {"groups": [group]}, case_data, {},
                               context=32768, kda_tokens=8192)
    changed = deepcopy(group)
    changed.update(count=1, activity_sum_ms=1000000)
    assert packet == prediction_packet("grouped", {"groups": [changed]}, case_data, {},
                                       context=32768, kda_tokens=8192)
    case, quant = parse_case(case_data), QuantSpec(**QUANT)
    steady = grouped_predictions(case, packet["groups"], quant,
                                 matrix_recipe=packet["matrix_recipe"])[0]["prediction"]
    phase = grouped_predictions(case, packet["groups"], quant,
                                matrix_recipe="phase_matmul")[0]["prediction"]
    assert steady["logical_work"] == phase["logical_work"]
    assert steady["logical_work"]["flops"] == 2 * 8192 * 7168 * 4608
    assert steady["serviced"] == pytest.approx(phase["serviced"])
    assert steady["per_call_ms"] < phase["per_call_ms"]
