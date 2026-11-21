"""
[Role] Check capture-to-cohort comparisons and the scoreboard views they feed.
[Invariants] Count each eligible boundary once, preserve absolute error and keep evidence gaps visible.
[Timeouts & Locks] Read small fixtures and run bounded analytical predictions.
[Side-Effect Policy] Only temporary reports are written. No device or network use.
"""
from __future__ import annotations

import csv
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from statistics import median

import pytest
from _support.captures import REPO_ROOT, captured_report, fixture_json, norm_case
from _support.memory_cases import SPARSE_OFFLOAD_SKU, case as small_case

from mindie_sim.cohort_accuracy import accuracy_fields, accuracy_policy, native_acceptance
from mindie_sim.cohort_comparison import compare_cohorts
from mindie_sim.compose import compose_case
from mindie_sim.observations.accuracy import accuracy_report
from mindie_sim.observations.cohort_scoreboard import render_cohort_campaigns, summarize_cohorts
from mindie_sim.observations.scoreboard import main, native_comparison_scoreboard
from mindie_sim.observations.scoreboard_accuracy import render_native_comparison_markdown
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from tools.accuracy_report import retained_copy_campaign

pytestmark = pytest.mark.tier0

NORM_CAPTURE = "a3_glm53_decode.csv"
MIXED_PRECISION_CASE = "glm53-a3-w4a8.json"
REPLICA_CASE = {"model": "qwen3.5-35b-a3b", "hardware": "Ascend910_9392"}
NATIVE_CAPTURE = ("r64_qbmm_dual_load", "kernel_details.csv")
NATIVE_CONDITIONS = {"hardware": SPARSE_OFFLOAD_SKU, "phase": "prefill",
                     "model": "glm-5.1", "quant": "W8A8_MXFP8"}

@pytest.mark.parametrize("observed, predicted, tier, passed", [
    (50, 65, "absolute_15us", True), (50, 65.001, "absolute_15us", False),
    (50, 35, "absolute_15us", True), (50, 34.999, "absolute_15us", False),
    (50.001, 65.001, "relative_15pct", False), (100, 115, "relative_15pct", True),
    (100, 85, "relative_15pct", True), (100, 115.001, "relative_15pct", False),
    (100, 84.999, "relative_15pct", False), (200, 218, "relative_15pct", True),
    (10, 24, "absolute_15us", True), (10, 0, "absolute_15us", True),
    (50, 0, "absolute_15us", False),
])
def test_accuracy_uses_observed_mean_tier_and_inclusive_tolerance_without_a_floor(
        observed, predicted, tier, passed):
    row = {"status": "compared", "observed": {"duration_ns": {"mean": observed * 1000}},
           "prediction": {"per_call_ms": predicted / 1000}}
    original = deepcopy(row)
    fields = accuracy_fields(row)
    assert fields["accuracy_tier"] == tier and fields["accuracy_pass"] is passed
    assert fields["per_call_diff_us"] == pytest.approx(predicted - observed)
    assert fields["abs_error_per_call_us"] == pytest.approx(abs(predicted - observed))
    assert row == original


@pytest.mark.parametrize("status", ["reference_only", "unscored"])
def test_accuracy_excludes_ineligible_boundaries_even_when_times_are_equal(status):
    fields = accuracy_fields({"status": status, "observed": {"duration_ns": {"mean": 100_000}},
                              "prediction": {"per_call_ms": 0.1}})
    assert fields["accuracy_tier"] == fields["accuracy_status"] == status
    assert fields["accuracy_pass"] is None and fields["accuracy_tolerance"] is None
    assert fields["per_call_diff_us"] == fields["abs_error_per_call_us"] == 0


@pytest.mark.parametrize("observed", [0, -1, float("inf"), float("nan")])
def test_compared_cohort_requires_positive_finite_observation(observed):
    row = {"status": "compared", "observed": {"duration_ns": {"mean": observed}},
           "prediction": {"per_call_ms": 0}}
    with pytest.raises(ValueError, match="requires_finite_per_call_times"):
        accuracy_fields(row)


@pytest.mark.requires_profiler
@pytest.mark.parametrize("clip", [False, True])
def test_profiler_cohort_preserves_complete_activity_and_unbound_occurrences(clip):
    case = norm_case()
    mapping, events = captured_report(case, NORM_CAPTURE, "RmsNorm", clip=clip)
    result = compare_cohorts(case, {"schema_version": "mindie-profile-summary/1",
                                    "structure_mapping": mapping})
    observed = result["windows"][0]["cohorts"][0]
    per_call = observed["prediction"]["per_call_ms"]
    assert per_call > 0
    assert not observed["occurrence_binding_implied"]
    assert observed["multiplicity"]["steps_in_window"] is None
    assert observed["observed"]["complete_event_count"] == len(events) - clip
    assert observed["observed"]["clipped_event_count"] == clip
    assert observed["prediction"]["per_declared_step_ms"] == pytest.approx(
        per_call * observed["declared_instance_count"])
    if clip:
        assert observed["status"] == "unscored" and observed["comparison"] is None
        assert observed["accuracy_pass"] is None
    else:
        measured = sum(event.duration_ns for event in events)
        assert observed["status"] == "compared"
        assert observed["observed"]["activity_sum_ns"] == measured
        assert observed["comparison"]["activity_sum"]["signed_residual_ms"] == pytest.approx(
            per_call * len(events) - measured / 1e6)
        stale = {**observed, "accuracy_tier": "incorrect", "accuracy_pass": None}
        refreshed = accuracy_fields(stale)
        assert refreshed == {key: observed[key] for key in refreshed}
        assert result["accuracy_policy"] == accuracy_policy()


@pytest.mark.requires_profiler
def test_single_capture_rank_folds_symmetric_dp_without_multiplying_instances():
    case = small_case(dp=4, ep=4)
    case["model"].update(norm_case()["model"], num_experts=4, top_k=1, moe_intermediate_size=64)
    mapping, events = captured_report(case, NORM_CAPTURE, "RmsNorm")
    group = mapping["windows"][0]["cohorts"][0]
    assert group["declared_instance_count"] * 4 == len(group["target_ids"])
    row = compare_cohorts(case, mapping)["windows"][0]["cohorts"][0]
    assert row["status"] == "compared" and row["replica"] is None
    assert row["prediction"]["observed_call_count_total_ms"] == row["prediction"]["per_call_ms"] * len(events)
    wrong = deepcopy(mapping)
    wrong["windows"][0]["cohorts"][0]["declared_instance_count"] *= 4
    with pytest.raises(ValueError, match="cohort_declared_instance_count_mismatch"):
        compare_cohorts(case, wrong)
    parsed = parse_case(case)
    dies = parsed.machine.dies
    slower = replace(dies[-1], rates=tuple((key, rate / 2) for key, rate in dies[-1].rates))
    heterogeneous = replace(parsed, machine=replace(parsed.machine, dies=(*dies[:-1], slower)))
    failed = compare_cohorts(heterogeneous, mapping)["windows"][0]["cohorts"][0]
    assert failed["status"] == "prediction_failed" and failed["reason"] == "symmetric_dp_hardware_differs"
    assert failed["observed"]["event_count"] == len(events)
    assert failed["accuracy_pass"] is None and failed["predicted_per_call_us"] is None


def test_declared_case_identity_and_physical_work_cannot_be_replaced_by_matching_shapes(profiler_api):
    case = norm_case()
    mapping, _ = captured_report(case, NORM_CAPTURE, "RmsNorm")
    changed = deepcopy(case)
    changed["model"]["layers"] += 1
    with pytest.raises(ValueError, match="cohort_case_graph_id_mismatch"):
        compare_cohorts(changed, mapping)
    corrupt = deepcopy(mapping)
    corrupt["windows"][0]["cohorts"][0]["physical_work_per_instance"]["vector_ops"] += 1
    with pytest.raises(ValueError, match="cohort_physical_or_tensor_contract_mismatch"):
        compare_cohorts(case, corrupt)


def test_same_shape_different_rank_windows_are_not_summed_as_elapsed_time(profiler_api):
    case = norm_case()
    mapping, _ = captured_report(case, NORM_CAPTURE, "RmsNorm")
    other, _ = captured_report(case, NORM_CAPTURE, "RmsNorm", rank_id="rank_1")
    mapping["windows"].append(other["windows"][0])
    result = compare_cohorts(case, mapping)
    assert len(result["windows"]) == 2
    assert result["distinct_priced_programs"] == 1 and result["status_counts"] == {"compared": 2}
    assert "duration_ms" not in result and "activity_sum_ms" not in result


@pytest.mark.requires_profiler
@pytest.mark.parametrize("kind, status", [
    ("GroupedMatmul", "reference_only"), ("QuantBatchMatmulV3", "compared"),
])
def test_cohort_quantized_epilogue_determines_comparable_boundary(kind, status):
    case = json.loads((REPO_ROOT / "examples/cases" / MIXED_PRECISION_CASE).read_text())
    case["model"]["layers"] = 4
    mapping, events = captured_report(case, "a3_glm53_prefill.csv", kind, phase="prefill", tokens=4096)
    row = compare_cohorts(case, mapping)["windows"][0]["cohorts"][0]
    assert row["status"] == status
    assert row["observed"]["activity_sum_ns"] == sum(event.duration_ns for event in events)
    if status == "compared":
        assert row["prediction"]["physical_work_per_instance"]["vector_ops"] > 0
        assert row["comparison"]["per_call_mean"] is not None
    else:
        assert row["reason"] == "quantized_epilogue_unmodeled"
        assert row["prediction"]["boundary"] == "partial_program_reference"
        assert row["comparison"] is None and row["accuracy_pass"] is None


def test_cohort_comparison_regenerates_a_projected_manifest():
    case = compose_case(**REPLICA_CASE, graph_capture_mode="none", layers=2,
                        dp=2, ep=2, tp=1, prompt_tokens=8, output_tokens=2)
    full = export_structure(
        case, phase="decode", tokens=1, context=8, sequences=2, scope="model",
        replicas=[0])
    report = {
        "schema_version": "mindie-profile-structure-mapping/1",
        "declared_execution": {
            "phase": "decode", "tokens": 1, "context": 8, "sequences": 2,
            "scope": "model", "replica_projection": {"selected": [0]},
        },
        "graph_id": full["graph_id"],
        "windows": [],
    }
    result = compare_cohorts(case, report)
    assert result["graph_id"] == full["graph_id"]


@pytest.fixture
def native_pair(profiler_api):
    import mindie_profiler
    from mindie_profiler.compare_acceptance import compare_native_predictions
    from mindie_profiler.sim_bridge import export_native_observations, native_prediction_requests

    root = Path(mindie_profiler.__file__).resolve().parents[1]
    observed = export_native_observations(
        root.joinpath("tests", "fixtures", *NATIVE_CAPTURE), include_hash=True, **NATIVE_CONDITIONS)
    requests = native_prediction_requests(observed)
    rows = observed["observations"]
    # Set controlled residuals on genuine ABI pairs to isolate acceptance arithmetic.
    timing = {row["abi"]["digest"]: row["timing"]["duration_ms"] for row in rows}
    predictions = []
    for request in requests["requests"]:
        row = {field: request[field] for field in
               ("request_id", "abi", "conditions", "boundary", "occurrence", "clock")}
        predictions.append({**row, "status": "modeled",
                            "duration_ms": timing[request["abi"]["digest"]]})
    return observed, requests, predictions, compare_native_predictions


@pytest.mark.parametrize("ratio, expected, exit_code", [
    (1.05, "incomplete", 4), (1.3, "failed", 3),
])
def test_native_scoreboard_counts_only_pairs_with_duration(native_pair, ratio, expected, exit_code, tmp_path):
    observed, requests, predictions, produce = native_pair
    timed, work_only = predictions
    timed["duration_ms"] *= ratio
    m, k = work_only["abi"]["inputs"][0]["shape"]
    _, n = work_only["abi"]["outputs"][0]["shape"]
    macs = m * n * k
    work_only.update(status="modeled_work_only", duration_ms=None, work={"cube": {"macs": macs}})
    comparison = produce(observed, predictions, requests=requests, diagnostic_tolerance={"relative": 0.5})
    result = native_acceptance(comparison)
    assert result["status"] == expected
    scored = next(row for row in result["items"] if row["duration_available"])
    assert scored["samples"][0]["diagnostic_within"] is True
    work = next(row for row in result["items"] if row["status"] == "work_only")
    assert work["work"]["cube"]["macs"] == macs
    source, output = tmp_path / "comparison.json", tmp_path / "scoreboard.json"
    source.write_text(json.dumps(comparison))
    rendered = tmp_path / "scoreboard.md"
    assert main([str(source), "--output", str(output), "--markdown", str(rendered)]) == exit_code
    board = json.loads(output.read_text())
    accuracy = board["denominators"]["accuracy"]
    assert accuracy["paired_items"] == 2 and accuracy["duration_denominator_items"] == 1
    assert accuracy["work_only"] == 1
    assert (accuracy["within"], accuracy["outside"]) == (int(ratio < 1.1), int(ratio > 1.1))
    report = accuracy_report(comparison)
    assert report["status"] == expected and report["coverage"]["duration_denominator_items"] == 1
    assert {point["status"] for point in report["points"]} == {"paired", "work_only"}
    markdown = render_native_comparison_markdown(board)
    assert "prediction_work_only" in markdown and f"{macs:.6f}" in markdown
    assert rendered.stat().st_size == len(markdown.encode("utf-8"))


def test_native_scoreboard_excludes_unpaired_abi_and_rejects_modified_counts(native_pair):
    observed, requests, predictions, produce = native_pair
    report = produce(observed, predictions[:1], requests=requests)
    accuracy = native_comparison_scoreboard(report)["denominators"]["accuracy"]
    assert accuracy["paired_items"] == accuracy["duration_denominator_items"] == 1
    assert accuracy["unpaired_observations"] == 1
    report["counts"]["paired"] += 1
    with pytest.raises(ValueError, match="count_mismatch:paired"):
        native_acceptance(report)


@pytest.mark.parametrize("field, value, error", [
    ("relative", 0.15, "formal_relative_tolerance_must_stay"),
    ("absolute_ms", 0.015, "formal_absolute_tolerance_must_stay"),
    ("absolute", 0.015, "formal_unknown_field:absolute"),
])
def test_native_formal_tolerance_cannot_be_relabelled(native_pair, field, value, error):
    observed, requests, predictions, produce = native_pair
    report = produce(observed, predictions, requests=requests)
    report["tolerance"]["formal"][field] = value
    with pytest.raises(ValueError, match=error):
        native_acceptance(report)


def retained_cohorts():
    return fixture_json("verification", "qwen38-native-cohorts.json")


def test_native_cohort_scoreboard_conserves_measured_activity_and_counts():
    source = retained_cohorts()
    window = source["windows"][0]
    families = summarize_cohorts(source, label="selected", provenance={})["windows"][0]["families"]
    assert sum(row["complete_calls"] for row in families) == 344
    assert sum(row["passed_cohorts"] for row in families) == 8
    assert sum(row["observed_ms"] for row in families) == pytest.approx(
        sum(row["observed"]["activity_sum_ns"] for row in window["cohorts"]) / 1e6)
    assert all(row["sum_absolute_descriptor_gap_ms"] + 1e-9 >= abs(row["signed_gap_ms"])
               for row in families)


def test_reference_windows_retain_activity_without_inflating_denominator():
    source = retained_cohorts()
    window = source["windows"][0]
    cohort = window["cohorts"][0]
    window["cohorts"] = [{**cohort, "status": "reference_only"}]
    source["windows"].append({**window, "rank_id": "other_rank"})
    report = summarize_cohorts(source, label="reference", provenance={"sha256": "controlled"})
    assert len(report["windows"]) == 2
    for row in report["windows"]:
        family, = row["families"]
        assert family["complete_calls"] == 48
        assert family["passed_cohorts"] == 0
        assert family["comparison_status"] == "reference_only"
        assert family["observed_ms"] == pytest.approx(cohort["observed"]["activity_sum_ns"] / 1e6)
    rendered = render_cohort_campaigns([report])
    assert f"| reference / {window['rank_id']} / 0 |" in rendered
    assert "| reference / other_rank / 1 |" in rendered
    assert rendered.count("| unscored |") == len(report["windows"])
    assert render_cohort_campaigns([]) == ""


def test_clipped_activity_is_not_compared_to_complete_kernel_calls():
    source = retained_cohorts()
    row = source["windows"][0]["cohorts"][0]
    row["observed"]["event_count"] += 1
    row["observed"]["clipped_event_count"] += 1
    row["observed"]["activity_sum_ns"] += 1000000
    report = summarize_cohorts(source, label="clipped", provenance={})
    family = next(f for f in report["windows"][0]["families"]
                  if f["native_kind"] == "MatMulV3")
    assert family["complete_calls"] == 184
    assert family["observed_ms"] == pytest.approx(458.381475)
    assert family["predicted_ms"] == pytest.approx(471.914144, abs=1e-6)


def point(name, predicted, *, tokens=1, configuration=1, occurrences=1, level="operator"):
    return {
        "point_id": name, "duration_ms": predicted, "occurrences": occurrences,
        "identity": {
            "level": level, "boundary": "declared_device_program",
            "hardware": {"resource": "oracle"}, "implementation": {"recipe": "oracle"},
            "phase": "prefill", "shape": {"tokens": tokens},
            "dtype": "bf16", "layout": "contiguous", "execution": {"workers": configuration},
        },
    }


def sample(prediction, observed, *, run="run0", split="heldout", suffix="0"):
    return {
        "sample_id": f"{prediction['point_id']}:{run}:{suffix}",
        "identity": deepcopy(prediction["identity"]), "duration_ms": observed,
        "run_id": run, "split": split,
        "source": f"independent_arithmetic_oracle:{prediction['point_id']}:{run}:{suffix}",
    }


def test_signed_cancellation_preserves_absolute_repeated_error_and_declared_tolerance():
    first, second = point("first", 12, tokens=1, occurrences=3), point("second", 8, tokens=2, occurrences=3)
    report = accuracy_report({
        "predictions": [first, second], "observations": [sample(first, 10), sample(second, 10)],
        "tolerances": {"relative": .1},
        "passed": True,
    })
    totals = report["domain_residuals"][0]
    assert totals["signed_residual_ms"] == 0
    assert totals["absolute_residual_ms"] == 12
    assert totals["selected_observed_sum_ms"] == 60
    assert report["tolerance_outcomes"] == {"within": 0, "outside": 2, "undeclared": 0}
    assert [row["comparison"]["allowed_error_ms"] for row in report["points"]] == [1, 1]


def test_repeated_sampling_cannot_outweigh_independent_runs_and_training_is_excluded():
    prediction = point("p", 21)
    samples = [sample(prediction, 10, run="a", suffix=str(index)) for index in range(20)]
    samples.extend([sample(prediction, 20, run="b"), sample(prediction, 40, run="c"),
                    sample(prediction, 10000, run="training", split="train")])
    used = sample(prediction, 1, run="leaked_training")
    samples.append(used)
    campaign = {
        "predictions": [prediction], "observations": samples,
        "training_sample_ids": [used["sample_id"]],
    }
    first = accuracy_report(campaign)
    paired = first["points"][0]
    assert paired["observed_ms"] == 20
    assert paired["heldout_samples"] == 22 and paired["heldout_runs"] == 3
    assert paired["comparison"]["within_tolerance"] is None
    assert first["coverage"]["excluded_observations"] == {"split_train": 1, "used_for_training": 1}
    campaign["observations"][-2]["duration_ms"] *= 100
    assert accuracy_report(campaign)["points"] == first["points"]
    with pytest.raises(ValueError, match="conflicting_accuracy_sample_or_split"):
        accuracy_report({**campaign, "observations": samples + [{**samples[0], "split": "train"}]})


def test_retained_gpu_batch_uses_raw_holdout_and_frozen_engine(tmp_path):
    path = REPO_ROOT / "data/measurements/gpu/service-curve-validation-20261028.json"
    retained = json.loads(path.read_text())
    altered = tmp_path / "capture.json"
    altered.write_text(json.dumps({**retained, "passed": True, "heldout_engine": [],
                                   "scores_by_work": [{"median_relative_error": 0, "accepted": True}]}))
    campaign = retained_copy_campaign(altered, root=REPO_ROOT, tolerances={"relative": 0.1})
    result = accuracy_report(campaign)
    point, = [row for row in result["points"] if row["status"] == "paired"]
    raw = json.loads((REPO_ROOT / retained["source"]).read_text())
    holdout = [row for row in raw["samples"] if row["split"] == "heldout"]
    assert point["observed_ms"] == pytest.approx(median(row["duration_ms"] * row["repetitions"] for row in holdout))
    assert point["heldout_samples"] == len(holdout) == 5
    assert point["comparison"]["relative_error"] == pytest.approx(-.003238146446031376)
    assert result["coverage"]["excluded_observations"] == {"split_train": 15}



def test_different_boundaries_and_incomplete_identities_remain_unpaired():
    operator = point("operator", 2)
    graph = point("graph", 2, level="subgraph")
    unknown = point("unknown", 3, tokens=2)
    del unknown["identity"]["hardware"]
    unknown_boundary = point("unknown_boundary", 1)
    unknown_boundary["identity"]["boundary"] = None
    wrong_boundary = point("wrapper", 2)
    wrong_boundary["identity"]["boundary"] = "host_wrapper"
    source = sample(operator, 1)
    report = accuracy_report({
        "predictions": [graph, unknown, unknown_boundary, wrong_boundary],
        "observations": [source, sample(unknown_boundary, 1)],
        "tolerances": {"relative": 1},
    })
    assert report["coverage"]["paired_heldout_points"] == 0
    assert report["coverage"]["unpaired_heldout_identities"] == 1
    assert report["points"][0]["reason"] == "no_matching_observation"
    assert report["points"][1]["reason"] == "incomplete_identity"
    assert report["points"][2]["missing_identity_fields"] == ["boundary"]
    assert report["points"][3]["reason"] == "no_matching_observation"
    assert report["coverage"]["excluded_observations"] == {"incomplete_identity": 1}
    assert all(row["comparison"] is None for row in report["points"])
    empty = accuracy_report({"predictions": [operator], "observations": []})
    assert empty["points"][0]["predicted_ms"] == 2
    assert empty["tolerance_outcomes"]["within"] == 0


def test_trend_reports_growth_error_and_reversal_without_using_training_shapes():
    small, large = point("small", 5, tokens=1), point("large", 12, tokens=2)
    campaign = {
        "predictions": [small, large], "observations": [sample(small, 10), sample(large, 20)],
        "relations": [{"id": "tokens", "kind": "trend", "axis": "shape.tokens",
                       "points": ["large", "small"]}],
    }
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "agrees"
    assert relation["segments"][0]["ratio_relative_error"] == pytest.approx(.2)
    large["duration_ms"] = 4
    assert accuracy_report(campaign)["relations"][0]["status"] == "disagrees"
    campaign["observations"][0]["split"] = "train"
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "not_assessed"
    assert relation["unpaired_points"] == ["small"]


def test_configuration_ranking_exposes_reversal_ties_and_uncontrolled_domain_changes():
    first, second, third = (point("one", 7, configuration=1), point("two", 6, configuration=2),
                            point("three", 4, configuration=3))
    campaign = {
        "predictions": [first, second, third],
        "observations": [sample(first, 8), sample(second, 5), sample(third, 5.02)],
        "tolerances": {"order_absolute_ms": .1},
        "relations": [{"id": "workers", "kind": "ranking", "vary": ["execution.workers"],
                       "points": ["one", "two", "three"]}],
    }
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "unresolved_ties"
    assert relation["pair_outcomes"] == {"agrees": 2, "observed_tie": 1}
    first["duration_ms"] = 3
    relation = accuracy_report(campaign)["relations"][0]
    assert relation["status"] == "disagrees"
    assert relation["pair_outcomes"]["reversed"] == 2
    first["identity"]["hardware"] = {"resource": "different"}
    campaign["observations"][0]["identity"] = deepcopy(first["identity"])
    assert accuracy_report(campaign)["relations"][0]["status"] == "domain_mismatch"


@pytest.mark.requires_profiler
@pytest.mark.parametrize("selected", [False, True])
def test_fusion_boundary_selects_physical_program_or_unfused_reference(selected):
    case = json.loads((REPO_ROOT / "examples/cases/v4f-a3-w8a8-tp8.json").read_text())
    case["model"]["layers"] = 4
    if selected:
        case["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})["native_fusion"] = {
            "implementation": "declared_native_fusion",
            "options": {"roles": ["hc_pre"], "traffic_profile": "hc_pre_m_k_split_175ef4d4"},
        }
    structure = export_structure(case, phase="prefill", tokens=4096, scope="body")
    mapping, _ = captured_report(case, "a3_v4f.csv", "HcPre", phase="prefill",
                                 tokens=4096, structure=structure)
    row, = compare_cohorts(case, mapping)["windows"][0]["cohorts"]
    prediction = row["prediction"]
    assert row["target_kind"] == "subgraph" and not row["occurrence_binding_implied"]
    assert prediction["per_call_ms"] > 0
    if not selected:
        assert row["status"] == "reference_only" and row["comparison"] is None
        assert prediction["physical_work_per_instance"]["launches"] > 1
        return
    assert row["status"] == "compared" and row["declared_instance_count"] > 1
    assert prediction["physical_work_per_instance"]["launches"] == 1
    target = next(group for group in structure["subgraphs"]
                  if group["subgraph_id"] == prediction["representative_target_id"])
    assert prediction["representative_operator_ids"] == target["resource_boundary"]["operator_ids"]
    internal = [node for node in structure["nodes"] if node["node_id"] in target["node_ids"]]
    assert len(internal) > 1 and all(not node["event_matchable"] for node in internal)
    assert prediction["physical_work_per_instance"]["hbm_bytes"] < sum(
        node["physical_work"]["hbm_bytes"] for node in internal)


@pytest.mark.parametrize("section, field, value, reason", [
    ("item", "predicted_ms", 0.05, "work_only_predicted_ms_must_stay_null"),
    ("comparison", "within_tolerance", True, "work_only_forbids_scored_field:within_tolerance"),
    ("comparison", "sample_verdicts", [{}], "work_only_must_not_retain_sample_verdicts"),
    ("counts", "work_only", 0, "count_mismatch:work_only"),
])
def test_native_work_only_rejects_invented_duration_verdicts(native_pair, section, field, value, reason):
    observed, requests, predictions, produce = native_pair
    predictions[0].update(status="modeled_work_only", duration_ms=None, work={"cube": {"macs": 1}})
    report = produce(observed, predictions, requests=requests)
    item = next(row for row in report["items"] if row["status"] == "work_only")
    target = {"item": item, "comparison": item["comparison"], "counts": report["counts"]}
    target[section][field] = value
    with pytest.raises(ValueError, match=reason):
        native_acceptance(report)


def test_native_item_verdict_cannot_hide_an_outside_sample(native_pair):
    observed, requests, predictions, produce = native_pair
    report = produce(observed, predictions, requests=requests)
    item = report["items"][0]
    predicted = item["predicted_ms"]
    sample = {"observed_ms": predicted / 2, "relative_error": 1,
              "formal_allowed_error_ms": predicted / 20, "within_tolerance": False}
    item["comparison"]["sample_verdicts"].append(sample)
    with pytest.raises(ValueError, match="verdict_cannot_hide_a_formal_sample_failure"):
        native_acceptance(report)


@pytest.mark.parametrize("partial", [False, True])
def test_native_scoreboard_preserves_unpaired_inputs_and_incomplete_windows(native_pair, partial):
    observed, requests, predictions, produce = native_pair
    if partial:
        observed["truncated"] = True
    else:
        candidate = predictions[-1]
        predictions += [{**candidate, "prediction_id": "duplicate"},
                        {**candidate, "prediction_id": "unlinked", "request_id": "unknown"}]
    inventory = {"records_total": 3} if partial else None
    board = native_comparison_scoreboard(produce(observed, predictions, requests=requests), inventory=inventory)
    counts = board["denominators"]["accuracy"]
    assert counts["outside"] == 0 and counts["within"] == (2 if partial else 1)
    assert counts["duration_denominator_items"] == counts["within"]
    rendered = render_native_comparison_markdown(board)
    if partial:
        assert board["truncated"] and board["status"] != "passed"
        assert "local item evidence only" in rendered
    else:
        assert counts["unpaired_observations"] == counts["unpaired_predictions"] == counts["ambiguous"] == 1
        assert "ambiguous_predictions" in rendered and "unknown_request_id" in rendered


@pytest.mark.parametrize("same_capture", [True, False])
def test_native_scoreboard_links_aliases_by_capture_identity(native_pair, tmp_path, capsys, same_capture):
    observed, requests, predictions, produce = native_pair
    report = produce(observed, predictions, requests=requests)
    raw = report["coverage"]["source_identity"]["path"]
    dispositions = tmp_path / "dispositions.csv"
    with dispositions.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "audit_id", "hardware_tier", "suite", "phase", "primary_raw_source",
            "primary_source_id", "source_ids", "event_source_ids", "evidence_capture_ids",
            "source_statuses", "derived", "derived_key", "source_set_group_id",
            "source_set_group_size", "source_set_role", "native_case_id",
        ))
        for index, capture in enumerate(("capture-0", "capture-0" if same_capture else "capture-1")):
            source = f"source-{index}"
            source_ids = json.dumps([source])
            edge = {"source_id": source, "capture_id": capture, "event_source": raw}
            writer.writerow((
                f"record-{index}", "A5", "controlled", "prefill", raw, source,
                source_ids, source_ids, json.dumps([capture]), json.dumps([edge]),
                False, "", "set", 2, "first_reference" if index == 0 else "duplicate_reference", "",
            ))

    comparison, inventory = tmp_path / "comparison.json", tmp_path / "inventory.json"
    comparison.write_text(json.dumps(report))
    inventory.write_text(json.dumps({"records_total": 2, "a5_records": 2}))
    assert main([str(comparison), "--inventory", str(inventory), "--record-dispositions", str(dispositions)]) == 0
    board = json.loads(capsys.readouterr().out)
    linkage = board["record_linkage"]
    counts = board["denominators"]
    assert counts["accuracy"]["duration_denominator_items"] == 2
    assert counts["inventory"]["validated_fraction_of_ledger"] is None
    rendered = render_native_comparison_markdown(board)
    if same_capture:
        assert linkage["status"] == "all_paired_items_linked"
        assert linkage["independent_evidence"]["matched_audit_records"] == 2
        assert linkage["independent_evidence"]["independent_capture_count"] == 1
        assert linkage["independent_evidence"]["duplicate_reference_records"] == 1
    else:
        assert linkage["comparison"]["ambiguous_items"] == 2
        assert linkage["independent_evidence"]["matched_audit_records"] == 0
        assert "ambiguous_source_alias" in rendered
