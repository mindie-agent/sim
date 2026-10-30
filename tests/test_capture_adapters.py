"""Boundary and source-identity regressions from retained A3/A5 captures."""
from copy import deepcopy
import csv
from decimal import Decimal
import json
from pathlib import Path
from statistics import median

import pytest

from mindie_sim.observations.captures import capture_inventory, iter_conditions, iter_observations
from mindie_sim.observations.comparison import comparison_report
from mindie_sim.observations.store import digest_file


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/observations/capture_adapters"


def protocol(name, captures, producer):
    result = json.loads((ROOT / f"data/observations/protocols/{name}-protocol.json").read_text())
    result["producer"] = {"path": str(FIXTURE / producer), "sha256": digest_file(FIXTURE / producer)}
    result["supporting_sources"] = []
    result["captures"] = [
        {"path": str(FIXTURE / filename), "sha256": digest_file(FIXTURE / filename), "label": filename}
        for filename in captures
    ]
    return result


def activation_protocol():
    return protocol("activation-call", ["activation_a.json", "activation_b.json"],
                    "activation_producer.py.txt")


def grouped_protocol():
    return protocol("grouped-call", ["grouped_probe.json"], "grouped_producer.py.txt")


def served_protocol():
    return protocol("grouped-served", ["served_grouped.csv"], "served_reducer.py.txt")


def test_activation_uses_original_events_and_restores_distinct_burst_boundary():
    manifest = activation_protocol()
    conditions = list(iter_conditions(manifest))
    raw = json.loads((FIXTURE / "activation_a.json").read_text())
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
    raw = json.loads((FIXTURE / "grouped_probe.json").read_text())["rows"][0]
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
    raw = json.loads((FIXTURE / "grouped_probe.json").read_text())
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
    with (FIXTURE / "served_grouped.csv").open(newline="") as stream:
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
    with (FIXTURE / "served_grouped.csv").open(newline="") as stream:
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
    with (FIXTURE / "served_grouped.csv").open(newline="") as stream:
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
