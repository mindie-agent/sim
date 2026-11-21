"""
[Role] Verify ingestion, source identity and prediction alignment for retained captures.
[Invariants] Source boundaries and measured work survive projection independently of prediction.
[Timeouts & Locks] Bounded local reads and in-memory comparisons.
[Side-Effect Policy] Only pytest temporary paths receive synthetic captures and reports.
"""
from __future__ import annotations

from copy import deepcopy
import csv
from decimal import Decimal
import gzip
import json
from pathlib import Path
from statistics import mean, median
import tracemalloc

import pytest
from _support.captures import (
    REPO_ROOT,
    activation_protocol,
    fixture_json,
    fixture_path,
    grouped_protocol,
    served_protocol,
)
from _support.memory_cases import case, linear_work

from mindie_sim import Case, predict
from mindie_sim.__main__ import main
from mindie_sim.observations.accuracy import apply_capture_observed_clock, extract_observed_aicore_freq_mhz
from mindie_sim.observations import multilevel_gate as gate
from mindie_sim.observations.calibration import import_profiler
from mindie_sim.observations.call_alignment import compare_calls
from mindie_sim.observations.captures import capture_inventory, iter_conditions, iter_observations
from mindie_sim.observations.comparison import comparison_report
from mindie_sim.observations.store import ObservationIndex, ProfilerCatalogue, digest_file, profiler_rows
from mindie_sim.observations.validation import compare, fit_resource_rate, service_observation
from mindie_sim.prediction.cost import primitive_key
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from tools.analyze_a5_microbench_boundaries import analyze_window, nanoseconds

pytestmark = pytest.mark.tier0

ADAPTER_FIXTURES = ("observations", "capture_adapters")


def test_capture_inventory_counts_source_bound_samples_and_acquisitions():
    protocol = activation_protocol()
    conditions = list(iter_conditions(protocol))
    samples = list(iter_observations(protocol))
    inventory = capture_inventory(protocol, preview_limit=1)
    run_hashes = {capture["sha256"] for capture in protocol["captures"]}
    assert inventory["timing_conditions"] == len(conditions)
    assert inventory["event_or_kernel_sum_samples"] == len(samples)
    assert inventory["acquisition_runs"] == len(run_hashes) == 2
    assert sum(inventory["conditions_by_boundary"].values()) == len(conditions)
    assert len(inventory["condition_preview"]) == 1
    assert "samples" not in inventory["condition_preview"][0]
    assert {sample["run_id"] for sample in samples} == run_hashes
    assert len({sample["sample_id"] for sample in samples}) == len(samples)
    for sample in samples:
        assert sample["sample_id"].startswith(sample["run_id"] + ":")
        assert sample["identity"]["implementation"]["producer_sha256"] == protocol["producer"]["sha256"]


@pytest.mark.parametrize("source_role", ["producer", "supporting_sources", "captures"])
def test_capture_identity_rejects_changed_bytes(source_role, tmp_path):
    protocol = activation_protocol()
    original = protocol["producer"] if source_role != "captures" else protocol["captures"][0]
    path = tmp_path / Path(original["path"]).name
    path.write_bytes(Path(original["path"]).read_bytes() + b"\n")
    altered = {**original, "path": str(path)}
    protocol[source_role] = altered if source_role == "producer" else [altered]
    with pytest.raises(ValueError, match="capture_source_hash_mismatch"):
        list(iter_conditions(protocol))


def test_activation_uses_original_events_and_restores_distinct_burst_boundary():
    manifest = activation_protocol()
    conditions = list(iter_conditions(manifest))
    raw = fixture_json(*ADAPTER_FIXTURES, "activation_a.json")
    first = next(iter(raw["cells"].values()))
    call, burst, hidden_replicate = conditions[0], conditions[1], conditions[2]
    assert [s["duration_ms"] for s in call["samples"]] == first["samples_ms"]
    assert [s["duration_ms"] for s in burst["samples"]] == pytest.approx(
        [value * first["burst_ops"] for value in first["burst_samples_ms"]])
    assert call["identity"]["boundary"] == "npu_event_call"
    assert burst["identity"]["boundary"] == "npu_event_burst"
    assert burst["identity"]["execution"]["invocations"] == first["burst_ops"]
    assert call["identity"] == hidden_replicate["identity"]
    assert call["metadata"]["context"]["hidden"] != hidden_replicate["metadata"]["context"]["hidden"]
    assert conditions[4]["identity"]["level"] == "subgraph"
    assert {s["metadata"]["trial"] for s in call["samples"]} == {0, 1}
    samples = [sample for condition in conditions for sample in condition["samples"]]
    assert len(samples) == 270 and len({sample["run_id"] for sample in samples}) == 2
    assert all(sample["split"] == "unassigned" for sample in samples)


def test_grouped_keeps_actual_histogram_and_whole_graph_replay_events():
    manifest = grouped_protocol()
    conditions = list(iter_conditions(manifest))
    raw = fixture_json(*ADAPTER_FIXTURES, "grouped_probe.json")["rows"][0]
    eager, graph1, graph8 = conditions[:3]
    counts = eager["identity"]["shape"]["expert_counts"]
    assert counts == [4] * 8 + [0] * 248
    assert [s["duration_ms"] for s in eager["samples"]] == raw["eager_per_call"]["raw_samples_ms"]
    assert [s["duration_ms"] for s in graph8["samples"]] == raw["graph_replay_8calls"]["raw_samples_ms"]
    assert graph8["identity"]["execution"]["invocations"] == 8
    assert graph1["identity"]["execution"]["invocations"] == 1
    assert eager["identity"]["boundary"] != graph1["identity"]["boundary"]
    assert graph8["identity"]["execution"]["activation_quant"] == "outside_timed_region"
    assert len({s["run_id"] for c in conditions for s in c["samples"]}) == 1
    assert len([s for c in conditions for s in c["samples"]]) == 240


def test_grouped_preserves_producer_canary_failure_without_shape(tmp_path):
    manifest = grouped_protocol()
    raw = fixture_json(*ADAPTER_FIXTURES, "grouped_probe.json")
    # run_row's early canary-rejected return precedes shape/projection construction.
    failed = {"label": raw["rows"][0]["label"], "status": "failed",
              "error": "canary_rejected", "canary": {"ok": False}}
    raw["rows"].insert(1, failed)
    path = tmp_path / "grouped-with-rejected-canary.json"
    path.write_text(json.dumps(raw))
    manifest["captures"] = [{"path": str(path), "sha256": digest_file(path), "label": "probe"}]
    conditions = list(iter_conditions(manifest))
    assert len(conditions) == 7
    assert conditions[3]["status"] == "failed"
    assert conditions[3]["metadata"]["canary"] == failed["canary"]
    assert conditions[3]["metadata"]["error"] == "canary_rejected"
    assert conditions[3]["identity"] == {} and conditions[3]["samples"] == []
    assert len([sample for condition in conditions for sample in condition["samples"]]) == 240


def test_served_grouped_is_kernel_sum_and_never_includes_contaminated_intervals(profiler_api):
    manifest = served_protocol()
    conditions = list(iter_conditions(manifest))
    assert [condition["status"] for condition in conditions] == ["prefill_contaminated", "measured"]
    assert conditions[0]["samples"] == []
    observed = conditions[1]["samples"][0]
    begin, end = observed["metadata"]["start_ns"], observed["metadata"]["end_ns"]
    oracle, calls = Decimal(0), 0
    with (fixture_path(*ADAPTER_FIXTURES, "served_grouped.csv")).open(newline="") as stream:
        for row in csv.DictReader(stream):
            start = int(Decimal(row["Start Time(us)"].strip()) * 1000)
            if begin <= start < end and row["Name"] in manifest["kernels"]:
                oracle += Decimal(row["Duration(us)"].strip()) / 1000
                calls += 1
    assert calls == 86
    assert observed["metadata"]["source_line_count"] == calls
    assert len(observed["metadata"]["source_line_preview"]) == 5
    assert observed["duration_ms"] == pytest.approx(float(oracle))
    assert observed["identity"]["boundary"] == "profiled_kernel_sum"
    assert observed["duration_ms"] != pytest.approx((end - begin) / 1e6)
    assert "group_list_tensor_values" in observed["metadata"]["unresolved_semantics"]
    inventory = capture_inventory(manifest)
    assert inventory["timing_conditions"] == 2 and inventory["event_or_kernel_sum_samples"] == 1


def test_zero_duration_markers_preserve_intervals_without_entering_kernel_sum(tmp_path, profiler_api):
    from mindie_sim.observations.kernels import kernel_rows

    manifest = served_protocol()
    expected = list(iter_conditions(manifest))
    with (fixture_path(*ADAPTER_FIXTURES, "served_grouped.csv")).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    for row in rows:
        if row["Name"] == manifest["marker"]:
            row["Duration(us)"] = "0"
    path = tmp_path / "zero-duration-markers.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest["captures"][0].update(path=str(path), sha256=digest_file(path))
    actual = list(iter_conditions(manifest))
    assert len(actual) == len(expected) == 2
    for before, after in zip(expected, actual):
        assert after["identity"] == before["identity"]
        assert after["status"] == before["status"]
        assert [sample["duration_ms"] for sample in after["samples"]] == [
            sample["duration_ms"] for sample in before["samples"]]
        for field in ("start_ns", "end_ns", "source_line_preview", "source_line_count"):
            assert after["metadata"][field] == before["metadata"][field]
    assert all(row.name != manifest["marker"] for row in kernel_rows(path))
    retained = [row for row in kernel_rows(path, include_zero_duration=True)
                if row.name == manifest["marker"]]
    assert len(retained) == 3 and all(row.duration_ns == 0 for row in retained)


def test_descriptive_residual_does_not_promote_unassigned_samples_to_validation():
    samples = list(iter_observations(grouped_protocol()))
    identity = samples[0]["identity"]
    prediction = {"point_id": "explicit_arithmetic_test", "identity": identity, "duration_ms": 0.1}
    wrong = deepcopy(prediction)
    wrong.update(point_id="wrong_boundary")
    wrong["identity"]["boundary"] = "profiled_device_kernel"
    report = comparison_report({
        "predictions": [prediction, wrong], "observations": iter(samples),
        "tolerances": {"relative": 10},
    })
    matching = [sample["duration_ms"] for sample in samples if sample["identity"] == identity]
    point = report["descriptive"]["points"][0]
    assert point["observed_ms"] == median(matching)
    assert point["comparison"]["signed_residual_ms"] == pytest.approx(0.1 - median(matching))
    assert "within_tolerance" not in point["comparison"]
    assert point["sample_splits"] == {"unassigned": 40}
    assert report["descriptive"]["points"][1]["reason"] == "no_exact_observation_identity"
    assert report["validation"]["coverage"]["paired_heldout_points"] == 0
    assert report["validation"]["tolerance_outcomes"]["within"] == 0


@pytest.mark.parametrize("reinterpreted_first", [False, True])
def test_duplicate_source_cannot_drop_newly_identified_missing_semantics(reinterpreted_first, profiler_api):
    current = next(iter_observations(served_protocol()))
    current["split"] = "heldout"
    old = deepcopy(current)
    del old["metadata"]["unresolved_semantics"]
    samples = [current, old] if reinterpreted_first else [old, current]
    with pytest.raises(ValueError, match="conflicting_accuracy_sample_or_split"):
        comparison_report({
            "predictions": [{"point_id": "same_partial_descriptor", "identity": current["identity"],
                             "duration_ms": current["duration_ms"]}],
            "observations": samples, "tolerances": {"relative": 0.1},
        })


def test_served_stream_rejects_unordered_time_instead_of_assigning_the_wrong_interval(tmp_path, profiler_api):
    manifest = served_protocol()
    with (fixture_path(*ADAPTER_FIXTURES, "served_grouped.csv")).open(newline="") as stream:
        reader = csv.DictReader(stream)
        columns, rows = reader.fieldnames, list(reader)
    rows[0], rows[-1] = rows[-1], rows[0]
    path = tmp_path / "out-of-order.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    manifest["captures"] = [{"path": str(path), "sha256": digest_file(path), "label": "unordered"}]
    with pytest.raises(ValueError, match="capture_csv_start_times_not_monotone"):
        list(iter_observations(manifest))


def test_missing_served_route_values_cannot_be_validated_by_relabelling_split(profiler_api):
    sample = next(iter_observations(served_protocol()))
    sample["split"] = "heldout"
    report = comparison_report({
        "predictions": [{"point_id": "same_partial_descriptor", "identity": sample["identity"],
                         "duration_ms": sample["duration_ms"]}],
        "observations": [sample], "tolerances": {"relative": 0.1},
    })
    assert report["descriptive"]["points"][0]["reason"] == "unresolved_observation_semantics"
    assert report["validation"]["coverage"]["excluded_observations"] == {"unresolved_semantics": 1}
    assert report["validation"]["tolerance_outcomes"]["within"] == 0

def activation_case():
    return json.loads((REPO_ROOT / "examples/calls/activation-nominal.json").read_text())


def test_real_activation_calls_keep_event_span_and_reuse_physical_predictions():
    protocol = activation_protocol()
    original = list(iter_conditions(protocol))
    result = compare_calls(activation_case(), protocol, include_submission=True)
    assert result["status"] == "completed"
    assert result["summary"]["samples"] == 270
    assert result["summary"]["sample_roles"] == {"unassigned": 270}
    assert result["summary"]["unique_predictions"] < len(original) * 2
    assert result["validation"]["strict_pairs"] == 0
    for row, source in zip(result["conditions"], original):
        assert row["identity"] == source["identity"]
        assert row["canary"] == source["metadata"].get("canary")
        for comparison in row["comparisons"]:
            assert comparison["observation_boundary"] == source["identity"]["boundary"]
            assert comparison["prediction_boundary"] != comparison["observation_boundary"]
            assert not comparison["strict_identity_equivalence"]
            assert comparison["signed_difference_ms"] == pytest.approx(
                comparison["predicted_duration_ms"] - comparison["observed_duration_ms"])
    burst = next(row for row in result["conditions"] if row["boundary"] == "npu_event_burst")
    source = next(row for row in original if row["condition_id"] == burst["condition_id"])
    # The adapter already restored whole-event time; do not multiply it again.
    assert burst["statistics"]["sample_median_ms"] == median(
        sample["duration_ms"] for sample in source["samples"])
    assert all(p["execution"]["live_tasks"] == 0 for p in result["predictions"])


def test_event_durations_and_canaries_never_change_predictions(tmp_path):
    protocol = activation_protocol()
    reference = compare_calls(activation_case(), protocol)
    changed = deepcopy(protocol)
    for index, capture in enumerate(changed["captures"]):
        raw = json.loads(Path(capture["path"]).read_text())
        raw["canaries"] = {name: {"ok": False, "note": "independent canary change"}
                           for name in raw["canaries"]}
        for cell in raw["cells"].values():
            cell["samples_ms"] = [value * 17 for value in cell["samples_ms"]]
            cell["per_pass_ms"] = [[value * 17 for value in trial]
                                   for trial in cell["per_pass_ms"]]
            cell["burst_samples_ms"] = [value * 17 for value in cell["burst_samples_ms"]]
        path = tmp_path / f"capture{index}.json"
        path.write_text(json.dumps(raw))
        capture.update(path=str(path), sha256=digest_file(path))
    altered = compare_calls(activation_case(), changed)
    assert altered["predictions"] == reference["predictions"]
    for before, after in zip(reference["conditions"], altered["conditions"]):
        assert after["statistics"]["sample_median_ms"] == pytest.approx(
            before["statistics"]["sample_median_ms"] * 17)


def test_actual_grouped_representation_is_not_substituted_with_bf16_nd():
    result = compare_calls(activation_case(), grouped_protocol())
    assert result["status"] == "completed"
    assert result["summary"]["samples"] == 240
    assert result["summary"]["unique_predictions"] == 6
    for row in result["conditions"]:
        assert row["alignment"]["quantization"]["compute"] == "int8"
        assert row["alignment"]["api"]["weight"] == "int8 [E,K,N] FRACTAL_NZ"
        assert row["alignment"]["identified_historical_compiled_kernel"] is None
        assert row["alignment"]["matrix"]["core_recipe"] == "mapped_matmul"
    assert {row["identity"]["execution"]["invocations"]
            for row in result["conditions"]} == {1, 8}
    assert result["validation"]["strict_pairs"] == 0


def test_budget_retains_observations_and_cli_returns_partial_report(tmp_path):
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(activation_protocol()))
    output = tmp_path / "comparison.json"
    code = main(["compare-calls", str(REPO_ROOT / "examples/calls/activation-nominal.json"),
                 str(protocol), "--max-predictions", "1", "-o", str(output)])
    result = json.loads(output.read_text())
    assert code == 2
    assert result["summary"]["samples"] == 270
    assert result["summary"]["prediction_evaluations"] == 1
    assert result["summary"]["condition_statuses"]["prediction_failed"] > 0
    assert any(comparison.get("reason") == "prediction_budget_exceeded"
               for row in result["conditions"] for comparison in row["comparisons"])

def test_real_profiler_csv_remains_readable_with_companion_api(profiler_api):
    rows = list(profiler_rows(fixture_path("observations", "profiler.csv")))
    assert rows and all(row["duration_ms"] > 0 and row["kernel"] for row in rows)


def test_profiler_catalogue_exposes_samples_and_variant_extrema():
    with ProfilerCatalogue(fixture_path("observations", "catalogue.sqlite")) as catalogue:
        variants = list(catalogue.variants(limit=5))
        samples = list(catalogue.samples(variants[0]["id"]))
    assert samples and all(row["duration_ms"] > 0 and row["source"] for row in samples)
    assert variants[0]["min_ms"] <= min(row["duration_ms"] for row in samples)
    assert variants[0]["max_ms"] >= max(row["duration_ms"] for row in samples)


def test_shared_profiler_row_preserves_consumer_projection(profiler_api):
    from dataclasses import replace
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.observations.kernels import kernel_rows

    capture = fixture_path("observations", "operator-kernels.csv")
    row = next(kernel_rows(capture))
    projected = next(profiler_rows(capture))
    assert row.start_ns == 1792684744877935379
    assert row.duration_ns == 108051
    assert row.input_shapes == ((16384, 5120), (5120,))
    assert row.output_shapes == ((16384, 5120), (16384, 1))
    assert projected == {
        "line": 2, "duration_ms": .108051, "kernel": "RmsNorm", "operator": "RmsNorm",
        "input_shapes": "16384,5120;5120", "input_dtypes": "DT_BF16;DT_BF16",
        "input_formats": "ND;ND", "output_shapes": "16384,5120;16384,1",
        "output_dtypes": "DT_BF16;FLOAT", "device": "0", "stream": "59",
        "core": "AI_VECTOR_CORE", "block_dim": "64", "boundary": "profiled_device_kernel",
    }
    assert observed_contract(row) == ("rmsnorm", (16384, 5120), "bf16", True)
    assert observed_contract(replace(row, output_shapes=None)) is None
    assert observed_contract(replace(row, output_shapes=((16384, 2560), (16384, 1)))) is None


def test_missing_or_unrecognized_matrix_dtype_does_not_authorize_a_contract(profiler_api):
    from dataclasses import replace
    from mindie_profiler.normalized import row_from_event
    from mindie_profiler.tensor_contracts import observed_contract
    from mindie_sim.observations.kernels import kernel_rows

    row = next(row for row in kernel_rows(fixture_path("observations", "operator-kernels.csv"))
               if row.kind == "MatMulV3")
    assert observed_contract(row) == (
        "matrix", (16384, 16384, 5120), "bf16", "bf16", "bf16", (), "")
    for descriptors in ("", "N/A;DT_BF16", "DT_UNRECOGNIZED;DT_BF16"):
        incomplete = row_from_event(replace(row.event, input_dtypes=descriptors))
        assert incomplete.input_shapes == row.input_shapes
        assert observed_contract(incomplete) is None


def test_bound_raw_samples_can_be_calibrated_without_reinterpreting_predictions():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    key = primitive_key(value, linear_work(plan, value.model, value.runtime, value.machine).ops[0], "none")
    with ProfilerCatalogue(fixture_path("observations", "catalogue.sqlite")) as catalogue:
        variant = next(catalogue.variants(limit=1))
        samples = list(catalogue.samples(variant["id"]))
    # This explicit test binding exercises identity translation only; it does
    # not claim that this fixture's kernel implements the test's linear op.
    with PrimitiveStore(":memory:") as store:
        result = import_profiler(fixture_path("observations", "catalogue.sqlite"), store, [
            {"variant": variant["id"], "key": key.to_dict(), "run_id": "translation",
             "split": "heldout"}])
        assert result["imported_samples"] == len(samples)
        assert store.lookup(key) is None
        assert predict(value, lower=linear_work, primitives=store)["metrics"] == predict(
            value, lower=linear_work)["metrics"]


def test_train_primitive_changes_duration_without_fabricating_resource_work():
    value = Case.from_mapping(case())
    plan = IterationPlan((RequestView("r", 4, 0, True, "prefill"),))
    op = linear_work(plan, value.model, value.runtime, value.machine).ops[0]
    key = primitive_key(value, op, "none")
    before = predict(value, lower=linear_work)
    with PrimitiveStore(":memory:") as store:
        store.record(key, duration_ms=10, run_id="analytic_oracle", repeat_index=0,
                     source="independent timing oracle")
        after = predict(value, lower=linear_work, primitives=store)
    assert after["requests"][0]["ttft_ms"] == 10
    assert after["execution"]["work"] == before["execution"]["work"]
    assert after["execution"]["serviced"]["matrix@r0:s0"] == before["execution"]["serviced"]["matrix@r0:s0"]


def test_service_boundary_mismatch_precludes_calibration_and_acceptance():
    observed = service_observation(fixture_path("observations", "service.json"))
    result = compare(predict(case()), observed, boundary="client_stream")
    assert not result["comparable"]
    assert result["rows"] and all(row["observed"] > 0 for row in result["rows"])
    with pytest.raises(ValueError, match="boundary_conflicts"):
        compare(predict(case()), observed, boundary="device_token_commit")


def test_actual_speculative_capture_uses_tokens_instead_of_sse_chunks_for_tpot():
    raw = fixture_json("observations", "service.json")
    observed = service_observation(fixture_path("observations", "service.json"))
    requests = raw["per_request"]
    expected = mean(row["decode_ms"] / (row["completion_tokens"] - 1) for row in requests)
    assert observed["tpot_ms"]["mean"] == pytest.approx(expected)
    assert observed["tpot_ms"]["mean"] < raw["tpot_ms"]["mean"] / 3
    assert observed["reported_metrics"]["tpot_ms"] == raw["tpot_ms"]
    assert observed["e2e_ms"]["mean"] == pytest.approx(
        mean(row["ttft_ms"] + row["decode_ms"] for row in requests))
    assert observed["output_tps"] == pytest.approx(
        sum(row["completion_tokens"] for row in requests) / raw["wall_s"])


def test_capture_without_usage_does_not_substitute_chunks_or_requested_token_counts(tmp_path):
    raw = fixture_json("observations", "service.json")
    for row in raw["per_request"]:
        row["completion_tokens"] = None
    path = tmp_path / "without-usage.json"
    path.write_text(json.dumps(raw))
    observed = service_observation(path)
    assert observed["tpot_ms"] == {}
    assert observed["output_tps"] is None
    assert observed["reported_metrics"]["tpot_ms"] == raw["tpot_ms"]
    assert observed["ttft_ms"] == raw["ttft_ms"]


def test_historical_source_index_preserves_hashes_and_source_roles(tmp_path):
    with ObservationIndex(tmp_path / "observations.sqlite") as index:
        index.scan(fixture_path("observations"))
        index.attach_profiler_catalogue(fixture_path("observations", "catalogue.sqlite"))
        rows = index.find_sources("profiler.csv")
        assert rows[0]["kind"] == "profiler" and len(rows[0]["sha256"]) == 64
        assert index.inventory()["profiler_catalogue"]["files"] == 1


def test_rate_fit_predicts_independent_measurements():
    root = REPO_ROOT / "data" / "measurements" / "gpu"
    raw = json.loads(next(root.glob("*.stdout")).read_text())
    # The fixture is the actual 4090 Driver API capture, not generated timings.
    samples = raw["samples"]
    fitted = fit_resource_rate(samples)
    changed = [dict(sample, duration_ms=sample["duration_ms"] * 2)
               if sample["split"] == "heldout" else sample for sample in samples]
    assert fitted["bytes_per_ms"] == fit_resource_rate(changed)["bytes_per_ms"]
    assert len(fitted["heldout"]) == 5
    assert max(abs(row["relative_error"]) for row in fitted["heldout"]) < .08

def test_real_matmul_rows_do_not_gain_negative_gaps_from_epoch_float_rounding():
    report = analyze_window(fixture_path("a5_microbench_20261030", "cube_matmul_m1_k4096_n4096.csv"), 4)
    stream = report["streams"][0]
    assert nanoseconds("1793294469786372.021\t") == 1793294469786372021
    assert stream["negative_gap_pairs"] == 0
    assert stream["legacy_float_negative_gap_pairs"] == 1
    assert stream["signed_end_to_start_gap"]["p50_us"] == .022
    assert stream["first_transition_gap_us"] == 84.448
    assert stream["duration_sum_us"] == 50.254
    assert stream["first_start_to_last_end_us"] == 134.746
    assert stream["duration_sum_over_window_span"] == pytest.approx(50.254 / 134.746)


def test_real_vector_rows_retain_first_transition_and_device_work_separately():
    report = analyze_window(fixture_path("a5_microbench_20261030", "vector_add_t1.csv"), 4)
    stream = report["streams"][0]
    assert report["physical_rows"] == 4
    assert report["device_kernel_duration"]["min_us"] == 1.326
    assert stream["first_transition_gap_us"] == 85.347
    assert stream["signed_end_to_start_gap"]["n"] == 3
    assert stream["after_first_transition_gap_diagnostic"]["n"] == 2
    assert stream["after_first_transition_gap_diagnostic"]["mean_us"] == (4.542 + 3.337) / 2
    assert stream["duration_sum_us"] == pytest.approx(2.134 + 1.523 + 1.477 + 1.326)
    assert report["kernel_descriptors"]["Block Num"] == ["8"]
    assert report["kernel_descriptors"]["Accelerator Core"] == ["AI_VECTOR_CORE"]


def _verify_capture(path, *, level="L1", events=("gemm",), ranks=(1,), measured=60_023, operator="elapsed_span"):
    boundary = gate.CaptureBoundary(
        gate.digest_file_streaming(path), "acq", "session", (), events, ranks, "device", events[0], events[-1],
        {"L1": "device_kernel", "L2": "periodic_block", "L3": "model_step"}[level], "ns",
        clock_operator=operator,
    )
    cohort = gate.CohortKey(level, "timing", "device", "decode", "eager", "shape", "tp1")
    item = gate.GateCase("capture", level, "heldout", boundary, "inputs", "stack", "impl", cohort, measured)
    gate.validate_lineage(gate.Manifest("native", "1", (item,)), gate.FrozenFitManifest("empty", "1", ()),
                          capture_files={item.case_id: path}, require_capture_files=True)


@pytest.mark.parametrize("level", ("L1", "L2", "L3"))
def test_native_csv_binds_rank_events_and_duration(tmp_path, level):
    path = tmp_path / "kernel_details.csv"
    path.write_text(
        "Device_id,Task ID,Name,Start Time(us),Duration(us)\n"
        "1,44,gemm,1793296125087346.271,60.023\n"
        "2,44,gemm,1793296125087346.271,548.894\n",
        encoding="utf-8",
    )
    _verify_capture(path, level=level)
    _verify_capture(path, level=level, ranks=(2,), measured=548_894)
    for change, cause in (({"ranks": (999,)}, "capture_rank_mismatch"),
                          ({"events": ("nonexistent",)}, "capture_event_mismatch"),
                          ({"measured": 100_000}, "capture_duration_mismatch")):
        with pytest.raises(ModelingError, match=cause):
            _verify_capture(path, level=level, **change)
    path.write_text("Name,Start Time(us),Duration(us)\ngemm,20,60.023\n", encoding="utf-8")
    with pytest.raises(ModelingError, match="rank_column_absent"):
        _verify_capture(path, level=level)


def test_profiler_rows_preserve_elapsed_and_busy_clock_boundaries(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text("Rank ID,OP Name,Start Time(us),Duration(us)\n1,first,20,5\n1,last,30,7\n")
    events = ("first", "last")
    _verify_capture(path, events=events, measured=17_000)
    _verify_capture(path, events=events, measured=12_000, operator="busy_union")
    path.write_text("Device_id,Task ID,Name,Start Time(us),Duration(us)\n1,44,gemm,20,5\n1,44,gemm,30,7\n")
    with pytest.raises(ModelingError, match="ambiguous_endpoint"):
        _verify_capture(path, events=("44",))
    _verify_capture(path, events=("row=2",), measured=7_000)
    compressed = path.with_suffix(".csv.gz")
    compressed.write_bytes(gzip.compress(path.read_bytes()))
    _verify_capture(compressed, events=("row=2",), measured=7_000)
    path.write_text("Device_id,OP Type,Count,Total Time(us)\n1,gemm,1,60.023\n")
    _verify_capture(path)
    path.write_text("Device_id,OP Type,Count,Total Time(us)\n1,gemm,2,60.023\n")
    with pytest.raises(ModelingError, match="aggregate_event"):
        _verify_capture(path)
    path.write_text("Device_id,Name\n1,gemm\n")
    with pytest.raises(ModelingError, match="csv_schema_unsupported"):
        _verify_capture(path)


def test_json_capture_filters_unrelated_events_while_streaming(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ijson", None)  # Exercise the stdlib fallback directly.
    path = tmp_path / "trace.json"
    with path.open("w", encoding="utf-8") as capture:
        capture.write('{"device_ranks":[1],"events":[')
        for index in range(40_000):
            capture.write(json.dumps({"name": f"unrelated-{index}", "clock_operator": "busy_union",
                                      "detail": "x" * 100}) + ",\n")
        capture.write('{"name":"gemm","clock_operator":"elapsed_span"}]}')
    tracemalloc.start()
    try:
        _verify_capture(path)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 2 * 1024 * 1024


def test_capture_envelope_clock_reaches_canonical_device_configuration():
    path = REPO_ROOT / "data/primitives/captures/arch22_a3_9362.capture.json"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert extract_observed_aicore_freq_mhz(envelope) == 1650
    from mindie_sim.backends.ascend import profiles
    sku = "Ascend910_9362"
    nominal = profiles()[sku]["device"]
    updated = apply_capture_observed_clock({"machine": {"sku": sku, "device": nominal}}, envelope)["machine"]["device"]
    assert updated["observed_aicore_freq_mhz"] == updated["aicore_freq_mhz"] == 1650
    assert updated["rates"]["mte2_aic"] == pytest.approx(nominal["rates"]["mte2_aic"] * 1650 / 1500)
