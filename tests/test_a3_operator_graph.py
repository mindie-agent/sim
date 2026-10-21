"""CPU-only contract tests for the isolated A3 operator-graph timing lane.

These tests exercise the real manifest/plan/results bookkeeping paths.  They do
not import torch, create a device, mock a graph, or claim capture/replay
evidence.  Device behavior remains a sim_audit lease task.
"""
from __future__ import annotations

import builtins
import copy
import gc
import json
import math
import weakref
from pathlib import Path

import pytest

from mindie_sim.a3_operator_graph import main
from mindie_sim.a3_operator_graph_plan import (
    ManifestError,
    build_plan,
    dry_run_payload,
    load_manifest_file,
    parse_manifest_text,
)
from mindie_sim.a3_operator_graph_results import (
    AtomicResultWriter,
    OutputPathError,
    ResultBook,
    ResultBookError,
    make_raw_observation,
    summarize_samples,
)
from mindie_sim.a3_operator_graph_timing import TimingMixin, warmup_plan

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "operator-graph"
FULL = EXAMPLES / "full-matrix.manifest.json"
CAPABILITY = EXAMPLES / "capability.manifest.json"


def _load(name: Path) -> dict:
    document, _, _ = load_manifest_file(name)
    return document


def _plan(name: Path):
    document, digest, path = load_manifest_file(name)
    return build_plan(document), document, digest, path


def _case(plan: dict, case_id: str) -> dict:
    return next(case for case in plan["cases"] if case["case_id"] == case_id)


def _condition(plan: dict, mode: str, n: int, case_id: str) -> dict:
    return next(condition for condition in plan["conditions"]
                if condition["case_id"] == case_id
                and condition["mode"] == mode and condition["n"] == n)


def _samples(condition: dict, trial_index: int, count: int = 30,
             cpu: float = 1.0, device: float = 2.0,
             wall: float = 3.0) -> list[dict]:
    return [
        make_raw_observation(
            condition, trial_index=trial_index, sample_index=index,
            cpu_submit_ms=cpu, device_interval_ms=device, wall_complete_ms=wall)
        for index in range(count)
    ]


def _trial(condition: dict, trial_index: int, count: int = 30,
           status: str = "measured") -> dict:
    samples = _samples(condition, trial_index, count)
    return {
        "trial_index": trial_index,
        "condition_id": condition["condition_id"],
        "mode": condition["mode"],
        "n": condition["n"],
        "status": status,
        "warmup": {
            "status": "ok",
            "policy": "rewarm_exact_callable_or_replay_before_each_trial",
            "timed": False,
            "excluded_from_samples": True,
            "trial_index": trial_index,
            "block_invocations": condition["block_invocations"],
            "calls": 10 if condition["mode"] == "eager_block" else 0,
            "replays": 0 if condition["mode"] == "eager_block" else 10,
        },
        "samples": samples,
        "summary": summarize_samples(samples),
        "errors": [],
    }


def _mini_plan(*, modes=("eager_block", "graph_single_repeated", "graph_block"),
               ns=(1,), cases=1, trials=2, samples=30) -> dict:
    document = copy.deepcopy(_load(CAPABILITY))
    document["cases"] = document["cases"][:cases]
    for case in document["cases"]:
        case["modes"] = list(modes)
        case["ns"] = list(ns)
    conditions = sum(len(case["modes"]) * len(case["ns"])
                     for case in document["cases"])
    document["protocol"]["trials"] = trials
    document["protocol"]["samples_per_condition"] = samples
    document["expectations"] = {
        "protocol_conditions": conditions,
        "raw_blocks": conditions * trials * samples,
    }
    return build_plan(document)


def _book(plan: dict, tmp_path: Path, name: str = "results.json") -> ResultBook:
    return ResultBook(
        plan, run_id="cpu-test-run", results_path=tmp_path / name,
        manifest_path=str(CAPABILITY), manifest_sha256="0" * 64,
        device="npu:0", source_files={})


def test_full_matrix_schema_geometry_and_exact_protocol_counts() -> None:
    plan, document, _, _ = _plan(FULL)
    assert document["kind"] == "a3_operator_graph_manifest"
    assert plan["kind"] == "a3_operator_graph"
    counts = plan["counts"]
    assert counts["case_count"] == 6
    assert counts["condition_count"] == 54
    assert counts["planned_condition_executions"] == 108
    assert counts["planned_raw_blocks"] == 3240
    assert counts["warmup_iterations"] == 10
    assert counts["trials"] == 2
    assert counts["samples_per_condition"] == 30
    assert counts["planned_warmup_iterations_per_trial"] == 10
    assert counts["planned_warmup_events_per_condition"] == 20
    assert counts["conditions_per_case"] == {
        "rmsnorm-bf16-t8-h5120": 9,
        "rmsnorm-bf16-t512-h5120": 9,
        "swiglu-bf16-t8-f2048": 9,
        "swiglu-bf16-t512-f2048": 9,
        "copy-uint8-rows8-row-bytes1024": 9,
        "copy-uint8-rows512-row-bytes1024": 9,
    }
    assert set(counts["conditions_per_mode"].values()) == {18}
    assert counts["conditions_per_n"] == {"1": 18, "32": 18, "128": 18}
    assert plan["expectations"] == {"protocol_conditions": 54, "raw_blocks": 3240}
    assert plan["budgets"]["case"] == 24.0
    assert plan["budgets"]["global_work"] == 1200.0
    assert plan["budgets"]["setup"] == 60.0
    assert plan["budgets"]["cleanup"] == 120.0
    assert plan["budgets"]["outer_supervisor_seconds_min"] == 1380.0

    rms_small = _case(plan, "rmsnorm-bf16-t8-h5120")["geometry"]
    assert rms_small["tensors"]["input"]["shape"] == [8, 5120]
    assert rms_small["tensors"]["input"]["strides"] == [5120, 1]
    assert rms_small["tensors"]["input"]["nbytes"] == 8 * 5120 * 2
    assert rms_small["tensors"]["gamma"]["nbytes"] == 5120 * 2
    assert rms_small["tensors"]["rstd"]["shape"] == [8, 1]
    assert rms_small["tensors"]["rstd"]["dtype"] == "float32"
    assert rms_small["tensors"]["rstd"]["itemsize"] == 4
    assert rms_small["tensors"]["rstd"]["nbytes"] == 8 * 1 * 4
    assert rms_small["total_declared_tensor_bytes"] == 174112
    assert rms_small["per_invocation_output_bytes_declared"] == 81920 + 32

    rms_large = _case(plan, "rmsnorm-bf16-t512-h5120")["geometry"]
    assert rms_large["tensors"]["input"]["numel"] == 512 * 5120
    assert rms_large["tensors"]["input"]["nbytes"] == 512 * 5120 * 2
    assert rms_large["tensors"]["rstd"]["dtype"] == "float32"
    assert rms_large["tensors"]["rstd"]["nbytes"] == 512 * 1 * 4
    assert rms_large["total_declared_tensor_bytes"] == 10498048

    swiglu = _case(plan, "swiglu-bf16-t512-f2048")["geometry"]
    assert swiglu["tensors"]["packed_input"]["shape"] == [512, 4096]
    assert swiglu["tensors"]["packed_input"]["nbytes"] == 512 * 4096 * 2
    assert swiglu["tensors"]["output"]["shape"] == [512, 2048]
    assert swiglu["tensors"]["output"]["strides"] == [2048, 1]

    copy_small = _case(plan, "copy-uint8-rows8-row-bytes1024")["geometry"]
    assert copy_small["tensors"]["source"]["shape"] == [8, 1024]
    assert copy_small["tensors"]["target"]["nbytes"] == 8 * 1024
    assert copy_small["total_declared_tensor_bytes"] == 2 * 8 * 1024

    eager = _condition(plan, "eager_block", 128, "rmsnorm-bf16-t512-h5120")
    assert eager["capture_invocations"] == 0
    assert eager["host_calls_per_block"] == 128
    assert eager["replay_calls_per_block"] == 0
    single = _condition(plan, "graph_single_repeated", 32,
                        "rmsnorm-bf16-t512-h5120")
    assert single["capture_invocations"] == 1
    assert single["host_calls_per_block"] == 32
    assert single["replay_calls_per_block"] == 32
    block = _condition(plan, "graph_block", 128,
                       "rmsnorm-bf16-t512-h5120")
    assert block["capture_invocations"] == 128
    assert block["host_calls_per_block"] == 1
    assert block["replay_calls_per_block"] == 1
    assert block["declared_capture_output_bytes"] == (
        128 * (512 * 5120 * 2 + 512 * 1 * 4))
    assert block["n_kind"] == (
        "declared_operator_api_invocations_not_measured_kernel_launches")

    rms_reference = single["reference_plan"]
    assert rms_reference["epsilon"] == 1e-6
    assert rms_reference["rstd_expected_shape"] == [512, 1]
    assert rms_reference["rstd_expected_dtype"] == "float32"
    assert rms_reference["rstd_expected_itemsize"] == 4
    assert rms_reference["callable_source"] == (
        "torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)")
    assert rms_reference["output_policy"] == (
        "native_default_outputs_only_no_out_overload")
    swiglu_condition = _condition(plan, "graph_block", 1,
                                  "swiglu-bf16-t8-f2048")
    assert swiglu_condition["reference_plan"]["packed_last_dim"] == 4096
    assert "silu(packed[..., :F]) * packed[..., F:]" == (
        swiglu_condition["reference_plan"]["output"])
    copy_condition = _condition(plan, "eager_block", 1,
                                "copy-uint8-rows8-row-bytes1024")
    assert copy_condition["callable_source"] == "target.copy_(source)"
    assert copy_condition["preallocated_output"] is True
    assert copy_condition["reference_plan"]["comparison"] == (
        "exact_byte_equality_no_tolerance")


def test_capability_example_is_explicitly_minimal_and_not_the_matrix() -> None:
    capability, document, _, _ = _plan(CAPABILITY)
    full, _, _, _ = _plan(FULL)
    assert capability["counts"]["condition_count"] == 9
    assert capability["counts"]["planned_raw_blocks"] == 540
    assert capability["counts"]["planned_condition_executions"] == 18
    assert capability["counts"]["conditions_per_n"] == {"1": 9, "32": 0, "128": 0}
    assert set(capability["counts"]["conditions_per_mode"].values()) == {3}
    assert len(capability["cases"]) == 3
    assert {case["operator"] for case in capability["cases"]} == {
        "rms_norm", "swiglu", "copy"}
    assert "capability" in document["manifest_id"]
    assert "not the full" in document["description"].lower()
    assert set(capability["condition_ids"]) < set(full["condition_ids"])
    assert all(condition["n"] == 1 for condition in capability["conditions"])
    assert not any(case["shape"].get("tokens") == 512
                   or case["shape"].get("rows") == 512
                   for case in capability["cases"])


def test_schedule_is_deterministic_complete_and_counterbalanced() -> None:
    plan, _, _, _ = _plan(FULL)
    schedule = plan["schedule"]
    again = build_plan(_load(FULL))["schedule"]
    assert schedule == again
    assert schedule["seed"] == 20261021
    first = schedule["trial_orders"][0]
    second = schedule["trial_orders"][1]
    assert len(first) == len(second) == 54
    assert set(first) == set(plan["condition_ids"]) == set(second)
    assert len(set(first)) == 54
    assert second == list(reversed(first))
    assert first[0] == second[-1]
    assert first[-1] == second[0]
    for trial in schedule["trial_orders"]:
        modes = [condition_id.split("::")[1] for condition_id in trial]
        ns = [int(condition_id.rsplit("::n", 1)[1]) for condition_id in trial]
        assert set(modes) == {"eager_block", "graph_single_repeated", "graph_block"}
        assert set(ns) == {1, 32, 128}
        assert modes.count("graph_block") == 18
        assert ns.count(128) == 18


@pytest.mark.parametrize("mutation", [
    lambda d: d.update(kind="wrong"),
    lambda d: d.pop("kind"),
    lambda d: d.update(unexpected=True),
    lambda d: d["protocol"].update(trials=0),
    lambda d: d["protocol"].update(warmup_iterations=0),
    lambda d: d["protocol"].update(samples_per_condition=0),
    lambda d: d["protocol"]["budgets_seconds"].update(case=25),
    lambda d: d["protocol"]["budgets_seconds"].update(global_work=1201),
    lambda d: d["protocol"]["budgets_seconds"].update(setup=61),
    lambda d: d["protocol"]["budgets_seconds"].update(cleanup=121),
    lambda d: d["protocol"]["reference"].update(atol=0),
    lambda d: d["protocol"]["reference"].update(rtol=0.5),
    lambda d: d["expectations"].update(protocol_conditions=53),
    lambda d: d["expectations"].update(raw_blocks=1),
    lambda d: d.update(cases=[]),
    lambda d: d["cases"][0].update(operator="unknown_operator"),
    lambda d: d["cases"][0]["shape"].update(hidden=0),
    lambda d: d["cases"][0]["shape"].update(extra=1),
    lambda d: d["cases"][1].update(dtype="float16"),
    lambda d: d["cases"][0].update(epsilon=1e-5),
    lambda d: d["cases"][2].update(epsilon=1e-6),
    lambda d: d["cases"][2].update(operator="rms_norm"),
    lambda d: d["cases"][2]["data"].update(gamma_low=0.5),
    lambda d: d["cases"][4]["shape"].update(row_bytes=0),
    lambda d: d["cases"][4]["data"].update(extra=1),
    lambda d: d["cases"][0].update(ns=[1, 1]),
    lambda d: d["cases"][0].update(ns=[1, 64]),
    lambda d: d["cases"][0].update(modes=["eager_block", "eager_block"]),
    lambda d: d["cases"][0].update(modes=["fast"]),
    lambda d: d["cases"][1].update(case_id=d["cases"][0]["case_id"]),
    lambda d: d["cases"][0].update(case_id="bad id with spaces"),
    lambda d: d["cases"][3].update(layout="strided"),
])
def test_manifest_contract_rejects_unsafe_or_dishonest_variants(mutation) -> None:
    document = copy.deepcopy(_load(FULL))
    mutation(document)
    with pytest.raises(ManifestError) as raised:
        build_plan(document)
    assert raised.value.errors
    assert raised.value.as_dict()["error"] == "manifest_invalid"


def test_json_parser_rejects_duplicate_keys_and_nonfinite_numbers() -> None:
    duplicate = (
        '{"kind":"a3_operator_graph_manifest",'
        '"manifest_id":"a","manifest_id":"b"}')
    with pytest.raises(ManifestError, match="duplicate JSON key"):
        parse_manifest_text(duplicate)
    nonfinite = '{"kind": NaN}'
    with pytest.raises(ManifestError, match="non-finite JSON number"):
        parse_manifest_text(nonfinite)


def test_dry_run_is_stdlib_only_and_writes_no_results(monkeypatch, tmp_path,
                                                     capsys) -> None:
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "torch" or name == "torch_npu" or name.startswith("torch."):
            raise AssertionError(f"dry-run imported {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    payload = dry_run_payload(FULL)
    assert payload["valid"] is True
    assert payload["validation_mode"] == "stdlib_only_no_torch_import"
    assert payload["plan"]["counts"]["condition_count"] == 54
    results_path = tmp_path / "must-not-exist.json"
    assert main([
        "--manifest", str(FULL), "--results", str(results_path),
        "--run-id", "dry-test", "--device", "npu:0", "--dry-run",
    ]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["dry_run"] is True
    assert output["writes_results"] is False
    assert output["plan"]["counts"]["planned_raw_blocks"] == 3240
    assert not results_path.exists()


def test_cli_rejects_bad_device_and_existing_results_without_device_import(
        tmp_path, capsys) -> None:
    assert main([
        "--manifest", str(CAPABILITY), "--results", str(tmp_path / "x.json"),
        "--run-id", "bad-device", "--device", "npu:1", "--dry-run",
    ]) == 2
    assert "npu:0" in capsys.readouterr().err

    existing = tmp_path / "existing.json"
    existing.write_text("prior evidence\n", encoding="utf-8")
    before = existing.read_bytes()
    assert main([
        "--manifest", str(CAPABILITY), "--results", str(existing),
        "--run-id", "would-overwrite", "--device", "npu:0",
    ]) == 2
    assert existing.read_bytes() == before
    assert "refusing to overwrite" in capsys.readouterr().err


def test_raw_clocks_derived_labels_and_anomalies() -> None:
    plan, _, _, _ = _plan(CAPABILITY)
    condition = _condition(plan, "graph_block", 1, "rmsnorm-bf16-t8-h5120")
    sample = make_raw_observation(
        condition, trial_index=0, sample_index=0, cpu_submit_ms=30.0,
        device_interval_ms=60.0, wall_complete_ms=90.0)
    assert sample["cpu_submit_ms"] == 30.0
    assert sample["device_interval_ms"] == 60.0
    assert sample["wall_complete_ms"] == 90.0
    assert sample["n_kind"] == (
        "declared_operator_api_invocations_not_measured_kernel_launches")
    assert sample["derived"]["label"] == "derived"
    assert sample["derived"]["denominator"] == 1
    assert sample["derived"]["denominator_kind"] == (
        "declared_operator_api_invocations_not_measured_kernel_launches")
    assert sample["derived"]["cpu_submit_per_invocation_ms"] == 30.0
    assert sample["derived"]["device_interval_per_invocation_ms"] == 60.0
    assert sample["derived"]["wall_complete_per_invocation_ms"] == 90.0
    assert "analytic_ms" not in sample

    full_plan, _, _, _ = _plan(FULL)
    block = _condition(full_plan, "graph_block", 128,
                       "rmsnorm-bf16-t8-h5120")
    sample = make_raw_observation(
        block, trial_index=1, sample_index=7, cpu_submit_ms=64.0,
        device_interval_ms=128.0, wall_complete_ms=256.0)
    assert sample["derived"]["denominator"] == 128
    assert sample["derived"]["cpu_submit_per_invocation_ms"] == 0.5
    assert sample["derived"]["device_interval_per_invocation_ms"] == 1.0
    assert sample["derived"]["wall_complete_per_invocation_ms"] == 2.0
    assert sample["anomalies"] == []

    negative = make_raw_observation(
        block, trial_index=0, sample_index=0, cpu_submit_ms=-1.0,
        device_interval_ms=1.0, wall_complete_ms=2.0)
    assert "negative_cpu_submit_ms" in negative["anomalies"]
    assert negative["derived"]["cpu_submit_per_invocation_ms"] < 0

    nonfinite = make_raw_observation(
        block, trial_index=0, sample_index=0, cpu_submit_ms=math.inf,
        device_interval_ms=1.0, wall_complete_ms=2.0)
    assert nonfinite["cpu_submit_ms"] is None
    assert "nonfinite_cpu_submit_ms" in nonfinite["anomalies"]
    assert nonfinite["derived"]["cpu_submit_per_invocation_ms"] is None

    summary = summarize_samples([sample, negative, nonfinite])
    assert summary["sample_count"] == 3
    assert summary["clocks"]["device_interval_ms"]["count"] == 3
    assert summary["clocks"]["cpu_submit_ms"]["count"] == 2
    assert summary["anomaly_count"] == 2
    assert "negative_cpu_submit_ms" in summary["anomaly_codes"]
    assert summary["interpretation"] == (
        "descriptive_only_not_analytic_prediction")
    assert "analytic_ms" not in json.dumps(summary)


def test_result_book_completeness_progress_failures_and_partial_persistence(
        tmp_path) -> None:
    plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=3)
    book = _book(plan, tmp_path)
    assert book.doc["status"] == "running"
    assert book.doc["progress"]["requested_conditions"] == 3
    assert book.doc["progress"]["measured_conditions"] == 0
    assert book.doc["progress"]["complete_conditions"] == 0

    measured_condition = plan["conditions"][0]
    other_condition = plan["conditions"][1]
    book.record_trial(measured_condition["condition_id"],
                      _trial(measured_condition, 0))
    assert book.doc["status"] == "partial"
    assert book.doc["progress"]["raw_blocks"] == 30
    assert book.condition(measured_condition["condition_id"])[
        "status"] == "measuring"
    assert book.condition(measured_condition["condition_id"])[
        "measurement_complete"] is False
    book.record_trial(measured_condition["condition_id"],
                      _trial(measured_condition, 1))
    book.update(measured_condition["condition_id"],
                correctness={"passed": True, "checks": {}})
    record = book.condition(measured_condition["condition_id"])
    assert record["status"] == "measured"
    assert record["measurement_complete"] is True
    assert book.doc["progress"]["complete_conditions"] == 1
    assert book.doc["progress"]["raw_blocks"] == 60
    assert len(record["warmup"]["trials"]) == 2
    assert all(item["timed"] is False for item in record["warmup"]["trials"])
    for trial in record["trials"]:
        assert trial["run_id"] == "cpu-test-run"
        assert len(trial["samples"]) == 30
        assert all(sample["run_id"] == "cpu-test-run"
                   for sample in trial["samples"])
        assert [sample["sample_index"] for sample in trial["samples"]] == list(
            range(30))

    book.record_trial(other_condition["condition_id"],
                      _trial(other_condition, 0, count=15))
    assert book.condition(other_condition["condition_id"])[
        "status"] == "measuring"
    book.fail(other_condition["condition_id"], "measurement", {
        "type": "RuntimeError", "message": "exact boom",
        "traceback": "Traceback (fake test only)", "retry": "not_attempted",
    })
    failed = book.condition(other_condition["condition_id"])
    assert failed["status"] == "failed"
    assert failed["measurement_complete"] is False
    assert failed["errors"][0]["error"]["message"] == "exact boom"
    assert book.doc["failures"][0]["error"]["message"] == "exact boom"
    assert book.doc["failures"][0]["retry"] == "not_attempted"
    assert len(failed["trials"][0]["samples"]) == 15

    book.finalize(stop_reason="test_stop_after_two_conditions")
    progress = book.doc["progress"]
    assert book.doc["status"] == "partial"
    assert progress["requested_conditions"] == 3
    assert progress["measured_conditions"] == 1
    assert progress["failed_conditions"] == 1
    assert progress["not_run_conditions"] == 1
    assert progress["terminal_conditions"] == 3
    assert progress["complete_conditions"] == 1
    assert progress["raw_blocks"] == 75
    assert book.doc["stop_reason"] == "test_stop_after_two_conditions"

    writer = AtomicResultWriter(tmp_path / "partial.json")
    writer.preflight()
    writer.write(book.to_dict())
    loaded = json.loads((tmp_path / "partial.json").read_text(encoding="utf-8"))
    assert loaded["status"] == "partial"
    assert loaded["progress"]["raw_blocks"] == 75
    assert loaded["conditions"][0]["trials"][0]["samples"][0]["run_id"] == (
        "cpu-test-run")
    assert loaded["conditions"][1]["status"] == "failed"
    assert "analytic_ms" not in json.dumps(loaded)


def test_result_book_never_marks_failed_checks_or_missing_samples_complete(
        tmp_path) -> None:
    plan = _mini_plan(modes=("graph_block",), ns=(1,), cases=1)
    condition = plan["conditions"][0]
    book = _book(plan, tmp_path)

    book.record_trial(condition["condition_id"], _trial(condition, 0))
    book.record_trial(condition["condition_id"], _trial(condition, 1))
    book.update(condition["condition_id"],
                correctness={"passed": False, "checks": {"output": "bad"}})
    record = book.condition(condition["condition_id"])
    assert record["status"] == "measured"
    assert record["measurement_complete"] is False
    assert "correctness_passed" in record["completeness"]["blockers"]
    assert book.doc["progress"]["measured_conditions"] == 1
    assert book.doc["progress"]["complete_conditions"] == 0
    book.finalize()
    assert book.doc["status"] == "completed_with_incomplete_measurements"
    assert book.doc["progress"]["complete_conditions"] == 0

    short_plan = _mini_plan(modes=("graph_block",), ns=(1,), cases=1)
    short_condition = short_plan["conditions"][0]
    short_book = _book(short_plan, tmp_path, "short.json")
    short_book.record_trial(short_condition["condition_id"],
                            _trial(short_condition, 0, count=1))
    short_book.record_trial(short_condition["condition_id"],
                            _trial(short_condition, 1, count=1))
    short_book.update(short_condition["condition_id"],
                      correctness={"passed": True})
    short = short_book.condition(short_condition["condition_id"])
    assert short["status"] == "failed"
    assert short["measurement_complete"] is False
    assert "all_samples_present" in short["completeness"]["blockers"]


def test_finalize_marks_incomplete_measuring_condition_failed_with_samples(
        tmp_path) -> None:
    plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=2)
    condition = plan["conditions"][0]
    book = _book(plan, tmp_path)
    book.record_trial(condition["condition_id"], _trial(condition, 0))
    assert book.condition(condition["condition_id"])["status"] == "measuring"
    book.finalize(stop_reason="interrupted_during_trial")
    record = book.condition(condition["condition_id"])
    assert record["status"] == "failed"
    assert record["reason"] == "run_stopped_before_condition_completion"
    assert len(record["trials"]) == 1
    assert len(record["trials"][0]["samples"]) == 30
    assert record["errors"][-1]["error"]["type"] == (
        "IncompleteConditionAtFinalize")
    assert record["errors"][-1]["retry"] == "not_attempted"
    assert book.doc["status"] == "partial"
    assert book.doc["progress"]["raw_blocks"] == 30
    assert book.doc["progress"]["failed_conditions"] == 1
    assert book.doc["progress"]["not_run_conditions"] == 1


def test_result_book_rejects_duplicates_and_overwrites(tmp_path) -> None:
    plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=1)
    condition = plan["conditions"][0]
    book = _book(plan, tmp_path)
    book.record_trial(condition["condition_id"], _trial(condition, 0))
    with pytest.raises(ResultBookError, match="next index"):
        book.record_trial(condition["condition_id"],
                          _trial(condition, 0))
    book.record_trial(condition["condition_id"], _trial(condition, 1))
    book.update(condition["condition_id"], correctness={"passed": True})
    assert book.condition(condition["condition_id"])["status"] == "measured"
    with pytest.raises(ResultBookError, match="overwrite"):
        book.fail(condition["condition_id"], "late", {
            "type": "X", "message": "no"})
    with pytest.raises(ResultBookError, match="cannot be initialized"):
        ResultBook({**plan, "conditions": [], "condition_ids": []},
                   run_id="empty", results_path=tmp_path / "empty.json",
                   manifest_path="m", manifest_sha256="0" * 64,
                   device="npu:0")


def test_writer_atomicity_overwrite_guard_and_nonfinite_refusal(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text('{"old": true}\n', encoding="utf-8")
    writer = AtomicResultWriter(path)
    with pytest.raises(OutputPathError, match="refusing to overwrite"):
        writer.preflight()
    overwriting = AtomicResultWriter(path, allow_overwrite=True)
    overwriting.preflight()
    overwriting.write({"old": False, "value": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": False, "value": 1}
    with pytest.raises(ValueError):
        overwriting.write({"bad": math.nan})
    assert json.loads(path.read_text(encoding="utf-8")) == {"old": False, "value": 1}


def test_example_schemas_and_non_measured_results_example_are_aligned() -> None:
    manifest_schema = json.loads(
        (EXAMPLES / "manifest.schema.json").read_text(encoding="utf-8"))
    results_schema = json.loads(
        (EXAMPLES / "results.schema.json").read_text(encoding="utf-8"))
    assert manifest_schema["properties"]["kind"]["const"] == "a3_operator_graph_manifest"
    assert results_schema["properties"]["kind"]["const"] == "a3_operator_graph_timing_results"
    assert "kind" in manifest_schema["required"]
    assert "kind" in results_schema["required"]
    for name in ("full-matrix.manifest.json", "capability.manifest.json"):
        document = json.loads((EXAMPLES / name).read_text(encoding="utf-8"))
        assert document["kind"] == "a3_operator_graph_manifest"

    assert "calibrated_conditions" not in results_schema["properties"][
        "progress"]["required"]
    assert "complete_conditions" in results_schema["properties"]["progress"][
        "required"]

    example = json.loads((EXAMPLES / "results.example.json").read_text(encoding="utf-8"))
    assert example["kind"] == "a3_operator_graph_timing_results"
    assert example["status"] == "failed"
    assert example["cleanup_ok"] is True
    assert example["persistence_ok"] is True
    assert example["progress"]["raw_blocks"] == 0
    assert example["progress"]["measured_conditions"] == 0
    assert example["progress"]["complete_conditions"] == 0
    assert example["stop_reason"] == "documentation_example_setup_failure"
    for condition in example["conditions"]:
        assert condition["status"] == "not_run"
        assert condition["measurement_complete"] is False
        assert condition["completeness"]["complete"] is False
        assert condition["trials"] == []

    def keys(value):
        if isinstance(value, dict):
            for key, child in value.items():
                yield key
                yield from keys(child)
        elif isinstance(value, list):
            for child in value:
                yield from keys(child)

    assert not any("calibrat" in key for key in keys(example))
    assert not any("calibrat" in key for key in keys(results_schema))


def test_measurement_completeness_controls_exit_not_a_success_label(
        tmp_path) -> None:
    from mindie_sim.a3_operator_graph_runtime import _exit_code

    def finished_book(correctness: bool, *, cleanup_ok: bool = True,
                      persistence_ok: bool = True) -> ResultBook:
        plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=1)
        condition = plan["conditions"][0]
        book = _book(plan, tmp_path, f"exit-{correctness}-{cleanup_ok}-{persistence_ok}.json")
        book.record_trial(condition["condition_id"], _trial(condition, 0))
        book.record_trial(condition["condition_id"], _trial(condition, 1))
        book.update(condition["condition_id"],
                    correctness={"passed": correctness})
        book.finalize(cleanup_ok=cleanup_ok, persistence_ok=persistence_ok)
        return book

    complete = finished_book(True)
    assert complete.condition(complete.plan["condition_ids"][0])[
        "measurement_complete"] is True
    assert complete.doc["progress"]["complete_conditions"] == 1
    assert complete.doc["progress"]["measured_conditions"] == 1
    assert complete.doc["status"] == "completed"
    assert _exit_code(complete) == 0
    assert "calibrated" not in complete.to_dict()["conditions"][0]

    correctness_failed = finished_book(False)
    assert correctness_failed.doc["status"] == (
        "completed_with_incomplete_measurements")
    assert correctness_failed.doc["progress"]["complete_conditions"] == 0
    assert _exit_code(correctness_failed) == 1

    cleanup_failed = finished_book(True, cleanup_ok=False)
    assert cleanup_failed.doc["status"] == "completed_with_cleanup_failure"
    assert cleanup_failed.doc["cleanup_ok"] is False
    assert _exit_code(cleanup_failed) == 1

    persistence_failed = finished_book(True, persistence_ok=False)
    assert persistence_failed.doc["status"] == (
        "completed_with_persistence_failure")
    assert persistence_failed.doc["persistence_ok"] is False
    assert _exit_code(persistence_failed) == 1


def test_cleanup_sync_failure_skips_unsafe_reset_and_is_not_ok() -> None:
    from mindie_sim.a3_operator_graph_device_common import BudgetClock

    class Stream:
        def __init__(self, fail_on=()):
            self.calls = 0
            self.fail_on = set(fail_on)

        def synchronize(self):
            self.calls += 1
            if self.calls in self.fail_on:
                raise RuntimeError(f"stream sync failure {self.calls}")

    class Npu:
        def __init__(self, fail=False):
            self.fail = fail

        def synchronize(self):
            if self.fail:
                raise RuntimeError("global sync failure")

    class Torch:
        def __init__(self, fail=False):
            self.npu = Npu(fail)

    class Graph:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    class Runner:
        pass

    def runner(stream, torch, graph):
        value = Runner()
        value.stream = stream
        value.torch = torch
        value._graphs = [("condition", graph)]
        value.budgets = {"cleanup": 120.0}
        return value

    def clock():
        value = BudgetClock({"case": 24, "global_work": 1200,
                             "setup": 60, "cleanup": 120})
        value.start_cleanup()
        return value

    graph = Graph()
    info = TimingMixin.cleanup(
        runner(Stream(fail_on={1}), Torch(), graph), clock())
    assert info["status"] == "failed_synchronize_before_reset"
    assert info["ok"] is False
    assert info["graphs_reset_skipped"] is True
    assert info["graphs_reset"] == 0
    assert graph.resets == 0

    graph = Graph()
    info = TimingMixin.cleanup(
        runner(Stream(), Torch(fail=True), graph), clock())
    assert info["status"] == "failed_global_synchronize_before_reset"
    assert info["ok"] is False
    assert graph.resets == 0

    graph = Graph()
    info = TimingMixin.cleanup(
        runner(Stream(fail_on={2}), Torch(), graph), clock())
    assert info["status"] == "failed_final_synchronize"
    assert info["ok"] is False
    assert graph.resets == 1
    assert info["final_synchronize"]["status"] == "failed"

    graph = Graph()
    info = TimingMixin.cleanup(runner(Stream(), Torch(), graph), clock())
    assert info["status"] == "ok"
    assert info["ok"] is True
    assert graph.resets == 1


def test_cleanup_reset_error_is_surfaced_not_success() -> None:
    from mindie_sim.a3_operator_graph_device_common import BudgetClock

    class Stream:
        def synchronize(self):
            pass

    class Npu:
        def synchronize(self):
            pass

    class Torch:
        npu = Npu()

    class BadGraph:
        def reset(self):
            raise RuntimeError("reset failed")

    class Runner:
        pass

    value = Runner()
    value.stream = Stream()
    value.torch = Torch()
    value._graphs = [("condition", BadGraph())]
    value.budgets = {"cleanup": 120.0}
    clock = BudgetClock({"case": 24, "global_work": 1200,
                         "setup": 60, "cleanup": 120})
    clock.start_cleanup()
    info = TimingMixin.cleanup(value, clock)
    assert info["status"] == "completed_with_reset_errors"
    assert info["ok"] is False
    assert info["reset_errors"][0]["error"]["message"] == "reset failed"


def test_cleanup_deadline_checked_after_final_sync() -> None:
    from mindie_sim.a3_operator_graph_device_common import CleanupBudgetExceeded

    class Stream:
        def synchronize(self):
            pass

    class Npu:
        def synchronize(self):
            pass

    class Torch:
        npu = Npu()

    class Graph:
        def __init__(self):
            self.resets = 0

        def reset(self):
            self.resets += 1

    class Clock:
        def __init__(self):
            self.stages = []

        def check_cleanup(self, stage):
            self.stages.append(stage)
            if stage == "cleanup_complete":
                raise CleanupBudgetExceeded(stage, 121.0, 120.0)

    class Runner:
        pass

    graph = Graph()
    runner = Runner()
    runner.stream = Stream()
    runner.torch = Torch()
    runner._graphs = [("condition", graph)]
    runner.budgets = {"cleanup": 120.0}
    clock = Clock()
    info = TimingMixin.cleanup(runner, clock)
    assert info["status"] == "timeout"
    assert info["ok"] is False
    assert info["cleanup_complete_deadline_exceeded"] is True
    assert info["within_cleanup_budget"] is False
    assert graph.resets == 1
    assert clock.stages[-1] == "cleanup_complete"


def test_warmup_is_declared_before_every_trial_with_explicit_n_semantics() -> None:
    plan, _, _, _ = _plan(FULL)
    eager = _condition(plan, "eager_block", 128, "rmsnorm-bf16-t8-h5120")
    single = _condition(plan, "graph_single_repeated", 32,
                        "rmsnorm-bf16-t8-h5120")
    block = _condition(plan, "graph_block", 128,
                       "rmsnorm-bf16-t8-h5120")
    eager_plan = warmup_plan(eager, 10)
    assert eager_plan["policy"] == (
        "rewarm_exact_callable_or_replay_before_each_trial")
    assert eager_plan["scope"] == "exact_eager_callable_calls_not_block_sized"
    assert eager_plan["calls"] == 10
    assert eager_plan["replays"] == 0
    assert eager_plan["block_invocations"] == 128
    assert eager_plan["timed"] is False
    assert eager_plan["excluded_from_samples"] is True
    single_plan = warmup_plan(single, 10)
    assert single_plan["replays"] == 10
    assert single_plan["captured_invocations_per_replay"] == 1
    assert single_plan["captured_invocations_warmed"] == 10
    assert single_plan["block_invocations"] == 32
    block_plan = warmup_plan(block, 10)
    assert block_plan["replays"] == 10
    assert block_plan["captured_invocations_per_replay"] == 128
    assert block_plan["captured_invocations_warmed"] == 1280
    assert block_plan["block_invocations"] == 128
    with pytest.raises(ValueError, match="positive"):
        warmup_plan(block, 0)


def test_measure_trial_rewarms_before_each_trial_outside_sample_sequence() -> None:
    plan, _, _, _ = _plan(FULL)
    condition = _condition(plan, "graph_single_repeated", 32,
                           "rmsnorm-bf16-t8-h5120")
    artifact = type("Artifact", (), {})()
    artifact.condition = condition
    artifact.mode = condition["mode"]
    artifact.metadata = {"active_case_seconds": 0.0}
    order: list[tuple[str, int]] = []

    class Runner:
        budgets = {"case": 24.0}

        def _check_deadlines(self, start, remaining, stage):
            return None

        def _warmup(self, artifact, trial_index, iterations):
            order.append(("warmup", trial_index))
            return {
                "status": "ok",
                "policy": "rewarm_exact_callable_or_replay_before_each_trial",
                "timed": False,
                "excluded_from_samples": True,
                "trial_index": trial_index,
                "requested_iterations": iterations,
                "calls": 0,
                "replays": iterations,
                "captured_invocations_per_replay": 1,
                "captured_invocations_warmed": iterations,
                "block_invocations": condition["block_invocations"],
                "duration_s": 0.001,
            }

        def _one_block(self, artifact, trial_index, sample_index):
            order.append(("sample", trial_index))
            return make_raw_observation(
                condition, trial_index=trial_index,
                sample_index=sample_index, cpu_submit_ms=1.0,
                device_interval_ms=2.0, wall_complete_ms=3.0)

    runner = Runner()
    first = TimingMixin.measure_trial(runner, artifact, 0, plan["protocol"])
    second = TimingMixin.measure_trial(runner, artifact, 1, plan["protocol"])
    for trial_index, trial in ((0, first), (1, second)):
        assert trial["status"] == "measured"
        assert len(trial["samples"]) == 30
        assert trial["warmup"]["trial_index"] == trial_index
        assert trial["warmup"]["timed"] is False
        assert trial["warmup"]["excluded_from_samples"] is True
    assert order[:2] == [("warmup", 0), ("sample", 0)]
    assert order[31:33] == [("warmup", 1), ("sample", 1)]
    assert sum(1 for item in order if item[0] == "warmup") == 2
    assert sum(1 for item in order if item[0] == "sample") == 60


def test_eager_block_releases_previous_output_and_retains_only_current(
        monkeypatch) -> None:
    import mindie_sim.a3_operator_graph_timing as timing

    class Stream:
        def synchronize(self):
            pass

    class StreamContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class Npu:
        def stream(self, stream):
            return StreamContext()

    class Torch:
        npu = Npu()

    class Runner:
        pass

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    class Output:
        pass

    plan, _, _, _ = _plan(FULL)
    condition = _condition(plan, "eager_block", 32,
                           "rmsnorm-bf16-t8-h5120")
    seen_previous_owner: list[object] = []
    artifact = type("Artifact", (), {})()
    artifact.condition = condition
    artifact.graph = None
    artifact.events = (Event(), Event())
    artifact.metadata = {"allocation": {}}
    artifact.tensors = type("Tensors", (), {})()
    previous = Output()
    previous_ref = weakref.ref(previous)
    artifact.final_output = previous

    def call():
        seen_previous_owner.append(artifact.final_output)
        return Output()

    artifact.tensors.callable = call
    runner = Runner()
    runner.torch = Torch()
    runner.stream = Stream()
    runner._sync = lambda: None
    monkeypatch.setattr(timing, "output_facts",
                        lambda torch, result: {"fake": True})
    del previous
    raw = TimingMixin._one_block(runner, artifact, 0, 0)
    gc.collect()
    assert seen_previous_owner == [None] * 32
    assert previous_ref() is None
    assert artifact.final_output is not None
    assert raw["eager_previous_output_released_before_block"] is True


def test_eager_warmup_does_not_retain_any_warmup_output(monkeypatch) -> None:
    import mindie_sim.a3_operator_graph_timing as timing

    class Stream:
        def synchronize(self):
            pass

    class StreamContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class Npu:
        def stream(self, stream):
            return StreamContext()

    class Torch:
        npu = Npu()

    class Runner:
        pass

    class Output:
        pass

    plan, _, _, _ = _plan(FULL)
    condition = _condition(plan, "eager_block", 32,
                           "rmsnorm-bf16-t8-h5120")
    artifact = type("Artifact", (), {})()
    artifact.condition = condition
    artifact.graph = None
    artifact.tensors = type("Tensors", (), {})()
    outputs: list[Output] = []

    def call():
        value = Output()
        outputs.append(value)
        return value

    artifact.tensors.callable = call
    runner = Runner()
    runner.torch = Torch()
    runner.stream = Stream()
    runner._sync = lambda: None
    monkeypatch.setattr(timing, "output_facts",
                        lambda torch, result: {"fake": True})
    warmup = TimingMixin._warmup(runner, artifact, 1, 10)
    refs = [weakref.ref(value) for value in outputs]
    outputs.clear()
    gc.collect()
    assert warmup["status"] == "ok"
    assert warmup["calls"] == 10
    assert warmup["replays"] == 0
    assert warmup["timed"] is False
    assert warmup["warmup_output_retained"] is False
    assert all(ref() is None for ref in refs)


def test_graph_block_retains_captured_final_output(monkeypatch) -> None:
    import mindie_sim.a3_operator_graph_timing as timing

    class Stream:
        def synchronize(self):
            pass

    class StreamContext:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return None

    class Npu:
        def stream(self, stream):
            return StreamContext()

    class Torch:
        npu = Npu()

    class Runner:
        pass

    class Event:
        def record(self, stream):
            pass

        def synchronize(self):
            pass

        def elapsed_time(self, end):
            return 1.0

    class Output:
        pass

    class Graph:
        def __init__(self):
            self.replays = 0

        def replay(self):
            self.replays += 1

    plan, _, _, _ = _plan(FULL)
    condition = _condition(plan, "graph_block", 1,
                           "rmsnorm-bf16-t8-h5120")
    artifact = type("Artifact", (), {})()
    artifact.condition = condition
    graph = Graph()
    artifact.graph = graph
    artifact.events = (Event(), Event())
    artifact.metadata = {"allocation": {}}
    artifact.tensors = type("Tensors", (), {})()
    retained = Output()
    artifact.final_output = retained
    runner = Runner()
    runner.torch = Torch()
    runner.stream = Stream()
    runner._sync = lambda: None
    monkeypatch.setattr(timing, "output_facts",
                        lambda torch, result: {"fake": True})
    raw = TimingMixin._one_block(runner, artifact, 0, 0)
    assert graph.replays == 1
    assert artifact.final_output is retained
    assert raw["eager_previous_output_released_before_block"] is False


def test_optional_debug_dump_path_is_sanitized_and_optional(tmp_path) -> None:
    from mindie_sim.a3_operator_graph_device import DeviceRunner

    runner = DeviceRunner(
        device="npu:0", device_index=0,
        budgets={"case": 24, "global_work": 1200, "setup": 60,
                 "cleanup": 120},
        debug_dir=str(tmp_path / "debug"))
    runner.debug_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"debug_dump": {"requested": True, "status": "not_attempted"}}
    condition = {"condition_id": "case::graph_block::n128"}

    class WritingGraph:
        def debug_dump(self, path):
            Path(path).write_text("{}\n", encoding="utf-8")

    runner._debug_dump(WritingGraph(), condition, metadata)
    assert metadata["debug_dump"]["status"] == "ok"
    assert metadata["debug_dump"]["path"].endswith(
        "case_graph_block_n128.json")
    assert Path(metadata["debug_dump"]["path"]).exists()

    unsupported = {"debug_dump": {"requested": True, "status": "not_attempted"}}
    runner._debug_dump(object(), condition, unsupported)
    assert unsupported["debug_dump"]["status"] == "unsupported"

    failing = {"debug_dump": {"requested": True, "status": "not_attempted"}}

    class FailingGraph:
        def debug_dump(self, path):
            raise RuntimeError("dump unsupported")

    runner._debug_dump(FailingGraph(), condition, failing)
    assert failing["debug_dump"]["status"] == "error"
    assert "dump unsupported" in failing["debug_dump"]["error"]["message"]


def test_initial_persistence_failure_prevents_device_work_and_reports_stderr(
        monkeypatch, tmp_path, capsys) -> None:
    import mindie_sim.a3_operator_graph_runtime as runtime

    plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=1)
    results = tmp_path / "unwritable.json"
    device_calls: list[str] = []

    class NeverConstructed:
        def __init__(self, *args, **kwargs):
            device_calls.append("constructed")
            raise AssertionError("device runner must not be constructed")

    def fail_write(self, document):
        raise OSError("disk full")

    monkeypatch.setattr(runtime, "DeviceRunner", NeverConstructed)
    monkeypatch.setattr(runtime.AtomicResultWriter, "write", fail_write)
    code = runtime.run_manifest(
        plan, manifest_path=str(CAPABILITY), manifest_sha256="0" * 64,
        run_id="persistence-failure", device="npu:0", results_path=results)
    assert code == 1
    assert device_calls == []
    assert not results.exists()
    error = json.loads(capsys.readouterr().err)
    assert error["error"] == "persistence_failure"
    assert error["durable_final_write"] is False
    assert {item["stage"] for item in error["errors"]} == {
        "initial_requested_conditions", "final"}
    assert all("disk full" in item["error"]["message"]
               for item in error["errors"])


def test_run_manifest_setup_failure_persists_truthful_partial_evidence(
        monkeypatch, tmp_path) -> None:
    import importlib

    from mindie_sim.a3_operator_graph_runtime import run_manifest

    real_import = importlib.import_module

    def unavailable_stack(name, package=None):
        if name in ("torch", "torch_npu"):
            raise ModuleNotFoundError(f"test-forced missing {name}")
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", unavailable_stack)
    plan = _mini_plan(modes=("eager_block", "graph_block"), ns=(1,), cases=1)
    results = tmp_path / "setup-failure.json"
    code = run_manifest(
        plan, manifest_path=str(CAPABILITY), manifest_sha256="0" * 64,
        run_id="setup-failure-test", device="npu:0", results_path=results)
    document = json.loads(results.read_text(encoding="utf-8"))
    assert code == 1
    assert document["status"] == "failed"
    assert document["setup"]["status"] == "failed"
    assert "test-forced missing torch" in document["setup"]["error"]["message"]
    assert document["cleanup"]["status"] == "not_needed_no_device_resources"
    assert document["progress"]["requested_conditions"] == 2
    assert document["progress"]["attempted_conditions"] == 0
    assert document["progress"]["terminal_conditions"] == 2
    assert document["progress"]["not_run_conditions"] == 2
    assert document["progress"]["measured_conditions"] == 0
    assert document["progress"]["complete_conditions"] == 0
    assert document["progress"]["raw_blocks"] == 0
    assert document["stop_reason"] == "setup_failed"
    for condition in document["conditions"]:
        assert condition["status"] == "not_run"
        assert condition["trials"] == []
        assert condition["measurement_complete"] is False
    assert "analytic_ms" not in json.dumps(document)


def test_runtime_device_authorization_guard(tmp_path) -> None:
    from mindie_sim.a3_operator_graph_runtime import run_manifest

    plan = _mini_plan(modes=("eager_block",), ns=(1,), cases=1)
    results = tmp_path / "must-not-be-written.json"
    with pytest.raises(ValueError, match="npu:0"):
        run_manifest(plan, manifest_path="m", manifest_sha256="0" * 64,
                     run_id="guard", device="npu:1", results_path=results)
    assert not results.exists()
