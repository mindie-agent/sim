"""Report/identity logic with constructed evidence; never measured accuracy."""
import copy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim.a3_cost import PredictiveCostModel
from mindie_sim.a3_types import A3InputError, MachineSpec
from mindie_sim.execution_replay import execution_plan_identity, execution_replay_report, identity_sha256
from mindie_sim.operator_estimation import _operator, create_operator_estimator, operator_plan_report
from mindie_sim.operator_plan import EstimateContext


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "examples/execution-replay/mixed.json").read_text(encoding="utf-8"))


def observation(row, *, sample_id="unit-sample", split="heldout", synthetic=False, duration=None):
    return {
        "sample_id": sample_id, "identity": copy.deepcopy(row["identity"]),
        "identity_sha256": row["identity_sha256"], "split": split,
        "source_class": "synthetic" if synthetic else "measured", "synthetic": synthetic,
        "duration_ms": duration if duration is not None else row["fine_isolated_ms"] * 1.25,
        "source": "unit_test_constructed_declared_label_not_a_device_measurement",
    }


def test_reuses_original_execution_and_keeps_predecessor_wait_out_of_extra_span():
    data = fixture()
    report = execution_replay_report(data)
    assert report["operator_plan"] == operator_plan_report(data["operator_plan"])
    assert report["effective_input"] == data
    assert report["coverage"]["fine_plan_records"] == 3
    assert report["operator_plan"]["elapsed_ms"] == pytest.approx(.0550192)
    assert report["accuracy"]["heldout"]["fine_isolated"]["count"] == 0
    assert report["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] is None
    before = 0
    for row in report["operators"]:
        span = row["shared_execution"]
        assert span["dependency_ready_ms"] == pytest.approx(before)
        assert span["span_from_ready_ms"] == pytest.approx(row["fine_isolated_ms"])
        assert row["extra_execution_span_ms"] == pytest.approx(0, abs=1e-14)
        before = span["complete_ms"]
    assert report["operators"][2]["shared_execution"]["complete_ms"] > report["operators"][2]["fine_isolated_ms"]
    # Legacy scalar estimate has additive launch; its old engine isolated
    # reference may overlap launch with work. Preserve and name both witnesses.
    pre = report["operators"][0]
    assert pre["legacy_estimate_ms"] == pytest.approx(.01124)
    assert pre["legacy_isolated_ms"] == pytest.approx(.01024)
    assert pre["fine_isolated_ms"] == pytest.approx(.011752)


def test_shared_independent_kernels_show_extra_span_without_changing_isolated_observation_comparison():
    data = fixture()
    gemm = data["operator_plan"]["operators"][1]
    gemm["deps"] = []
    other = copy.deepcopy(gemm)
    other["op_id"] = "another"
    data["operator_plan"]["operators"] = [gemm, other]
    first = execution_replay_report(data)
    assert all(row["shared_execution"]["dependency_ready_ms"] == 0 for row in first["operators"])
    assert max(row["extra_execution_span_ms"] for row in first["operators"]) > 0
    data["evidence"] = [observation(first["operators"][0], duration=first["operators"][0]["fine_isolated_ms"])]
    replay = execution_replay_report(data)
    assert replay["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] == 0
    # Two op IDs with the same actual physical identity share one observation,
    # rather than counting the same observation twice in accuracy.
    assert replay["evidence"][0]["matched_op_ids"] == ["gemm", "another"]
    assert replay["accuracy"]["heldout"]["fine_isolated"]["count"] == 1
    assert replay["operator_plan"] == first["operator_plan"]
    assert replay["costs_modified_by_evidence"] is False


def test_identity_hash_is_canonical_but_binds_all_execution_details():
    data = fixture()
    plan_input = data["operator_plan"]
    machine = MachineSpec.from_mapping(plan_input["machine"])
    op = _operator(plan_input["operators"][1])
    context = EstimateContext()
    estimator = create_operator_estimator(plan_input["estimators"])
    cost = PredictiveCostModel(machine, op_estimator=estimator).cost(op)
    def identity(plan=cost.execution_plan, operator=op):
        return execution_plan_identity(
            plan, operator, machine, context, estimator_config=plan_input["estimators"],
            protocol=data["protocol"], launch_ms=cost.launch_ms,
        )
    original = identity()
    reordered = {key: original[key] for key in reversed(original)}
    assert identity_sha256(original) == identity_sha256(reordered)
    # Runtime IDs, upstream waits and observation labels are not kernel identity.
    assert identity(operator=replace(op, op_id="renamed", deps=("somewhere",), requests=("another-request",))) == original
    plan = cost.execution_plan
    first = plan.stages[0]
    altered_stage = replace(first, minimum_duration_ms=first.minimum_duration_ms + .001)
    assert identity_sha256(identity(plan=replace(plan, stages=(altered_stage, *plan.stages[1:])))) != identity_sha256(original)
    assert identity_sha256(identity(plan=replace(plan, version="changed"))) != identity_sha256(original)
    assert identity_sha256(identity(plan=replace(plan, backend="another"))) != identity_sha256(original)
    # Returned identities are detached from the mutable caller provenance.
    plan.provenance["later_note"] = "changed after capture"
    assert "later_note" not in original["plan"]["provenance"]


@pytest.mark.parametrize("target,field", [
    ("plan", "version"), ("plan", "backend"), ("hardware", "hardware_key"),
    ("operator", "dtype"), ("operator", "layout"), ("identity", "timing_scope"),
])
def test_evidence_rejects_rehashed_identity_mismatches(target, field):
    data = fixture()
    row = execution_replay_report(data)["operators"][0]
    evidence = observation(row)
    mapping = evidence["identity"] if target == "identity" else evidence["identity"][target]
    mapping[field] = "different"
    evidence["identity_sha256"] = identity_sha256(evidence["identity"])
    data["evidence"] = [evidence]
    report = execution_replay_report(data)
    assert report["evidence"][0]["status"] == "rejected"
    assert report["evidence"][0]["reason"] == "execution_replay_evidence_identity_not_in_current_plans"
    assert report["accuracy"]["heldout"]["fine_isolated"]["count"] == 0


@pytest.mark.parametrize("change", ["profile", "tiling", "scope", "protocol", "shape", "hardware"])
def test_current_configuration_changes_refuse_old_evidence(change):
    data = fixture()
    original = execution_replay_report(data)
    data["evidence"] = [observation(original["operators"][1])]
    plan = data["operator_plan"]
    if change == "profile":
        plan["estimators"]["gemm_profile"]["source"] += " different declaration"
    elif change == "tiling":
        plan["estimators"]["gemm_profile"]["l1_k"] = 32
    elif change == "scope":
        data["evidence"][0]["identity"]["context"]["timing_scope"] = "device_event_warm_batched"
        data["evidence"][0]["identity_sha256"] = identity_sha256(data["evidence"][0]["identity"])
    elif change == "protocol":
        data["protocol"]["cache_state"] = "explicit_cold"
    elif change == "shape":
        for name in ("semantic_shape", "local_shape", "padded_shape", "launch_shape"):
            plan["operators"][1][name] = [16, 48, 80]
        plan["operators"][1]["flops"] = 122880
    elif change == "hardware":
        plan["machine"]["device"]["hbm_bw_bytes_per_s"] = 3000000000
    report = execution_replay_report(data)
    assert report["evidence"][0]["status"] == "rejected"
    assert report["accuracy"]["heldout"]["fine_isolated"]["count"] == 0


def test_heldout_and_train_metrics_are_separate_and_synthetic_never_counts_as_accuracy():
    data = fixture()
    rows = execution_replay_report(data)["operators"]
    data["evidence"] = [
        observation(rows[0], sample_id="heldout", duration=rows[0]["fine_isolated_ms"] * 2),
        observation(rows[1], sample_id="train", split="train", duration=rows[1]["fine_isolated_ms"] * 4),
        observation(rows[2], sample_id="synthetic", synthetic=True, duration=1e6),
    ]
    report = execution_replay_report(data)
    accuracy = report["accuracy"]
    assert accuracy["heldout_sample_ids"] == ["heldout"]
    assert accuracy["train_sample_ids"] == ["train"]
    assert accuracy["synthetic_matched_count"] == 1
    assert accuracy["heldout"]["fine_isolated"]["count"] == 1
    assert accuracy["heldout"]["fine_isolated"]["wape_pct"] == 50
    assert accuracy["train_diagnostics"]["fine_isolated"]["wape_pct"] == 75
    assert report["costs_modified_by_evidence"] is False
    # Changing heldout outcomes cannot influence train diagnostics or any cost.
    data["evidence"][0]["duration_ms"] *= 3
    changed = execution_replay_report(data)
    assert changed["accuracy"]["train_diagnostics"] == accuracy["train_diagnostics"]
    assert changed["operator_plan"] == report["operator_plan"]
    assert changed["operators"] == report["operators"]


@pytest.mark.parametrize("field,value,reason", [
    ("synthetic", True, "source_class_synthetic_mismatch"),
    ("duration_ms", 0, "duration_invalid"),
    ("duration_ms", True, "duration_invalid"),
    ("split", "test", "split_invalid"),
    ("identity", {}, "identity_hash_mismatch"),
    ("identity_sha256", "shape-only-key", "identity_hash_mismatch"),
    ("source", "", "source_missing"),
])
def test_incomplete_or_contradictory_evidence_is_not_used(field, value, reason):
    data = fixture()
    evidence = observation(execution_replay_report(data)["operators"][0])
    evidence[field] = value
    data["evidence"] = [evidence]
    result = execution_replay_report(data)
    assert result["evidence"][0]["status"] == "rejected"
    assert reason in result["evidence"][0]["reason"]
    assert result["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] is None


def test_shape_only_evidence_missing_identity_and_duplicate_samples_are_rejected():
    data = fixture()
    record = observation(execution_replay_report(data)["operators"][0])
    data["evidence"] = [record, copy.deepcopy(record)]
    report = execution_replay_report(data)
    assert report["accuracy"]["rejected_count"] == 2
    assert all("duplicate_sample_id" in row["reason"] for row in report["evidence"])
    del record["identity"]
    data["evidence"] = [record]
    assert "incomplete" in execution_replay_report(data)["evidence"][0]["reason"]
    data["evidence"] = [{"op_id": "pre", "shape": [2560], "duration_ms": 1}]
    assert "fields_invalid" in execution_replay_report(data)["evidence"][0]["reason"]


def test_fallback_has_no_fine_time_or_fingerprint_and_cannot_match_old_plan_evidence():
    data = fixture()
    row = execution_replay_report(data)["operators"][0]
    old = observation(row)
    data["operator_plan"]["estimators"]["strict"] = False
    # Retain declared real baseline work while removing the supported vector flow.
    data["operator_plan"]["operators"][0]["family"] = "rmsnorm"
    data["operator_plan"]["operators"][0]["metadata"] = {}
    data["evidence"] = [old]
    report = execution_replay_report(data)
    fallback = report["operators"][0]
    assert fallback["operator_estimation"]["status"] == "fallback"
    assert fallback["fine_isolated_ms"] is None and fallback["identity"] is None
    assert fallback["identity_sha256"] is None and fallback["extra_execution_span_ms"] is None
    assert fallback["legacy_isolated_ms"] > 0
    assert report["coverage"]["fine_plan_records"] == 2
    assert report["evidence"][0]["status"] == "rejected"


def test_full_hardware_hash_is_available_when_legacy_hardware_key_is_empty():
    data = fixture()
    data["operator_plan"]["machine"]["hardware_key"] = ""
    report = execution_replay_report(data)
    identity = report["operators"][0]["identity"]
    assert identity["hardware"]["hardware_key"] == ""
    assert identity["hardware_sha256"] == identity_sha256(identity["hardware"])
    assert identity["hardware_sha256"].startswith("sha256:")


@pytest.mark.parametrize("metadata", [
    {"fusion_stages": [{"name": "read", "hbm_bytes": 1024}]},
    {"comm_stages": [{"kind": "read", "work": {"local_hbm_read": 1024}}]},
])
def test_staged_fallback_uses_terminal_completion_for_span_and_consumer_readiness(metadata):
    data = fixture()
    data["operator_plan"]["estimators"]["strict"] = False
    consumer = copy.deepcopy(data["operator_plan"]["operators"][0])
    consumer.update(op_id="consumer", deps=["staged-fallback"])
    data["operator_plan"]["operators"] = [
        {"op_id": "staged-fallback", "family": "elementwise", "launch_shape": [16],
         "hbm_bytes": 1024, "metadata": metadata}, consumer,
    ]
    report = execution_replay_report(data)
    staged, consumer_row = report["operators"]
    assert staged["identity"] is None and staged["fine_isolated_ms"] is None
    assert staged["shared_execution"]["complete_ms"] > 0
    assert consumer_row["shared_execution"]["dependency_ready_ms"] == staged["shared_execution"]["complete_ms"]
    assert consumer_row["extra_execution_span_ms"] == pytest.approx(0, abs=1e-14)


def test_extreme_finite_evidence_never_emits_nonfinite_json_metrics():
    data = fixture()
    row = execution_replay_report(data)["operators"][0]
    data["evidence"] = [observation(row, duration=1e-320)]
    tiny = execution_replay_report(data)
    assert tiny["evidence"][0]["status"] == "rejected"
    assert "unrepresentable" in tiny["evidence"][0]["reason"]
    json.dumps(tiny, allow_nan=False)
    data["evidence"] = [observation(row, duration=1e308, sample_id=str(i)) for i in range(2)]
    huge = execution_replay_report(data)
    assert huge["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] == 100
    assert huge["accuracy"]["heldout"]["fine_isolated"]["mean_absolute_error_ms"] == 1e308
    json.dumps(huge, allow_nan=False)
    data["evidence"] = [observation(row, duration=value, sample_id=str(i))
                        for i, value in enumerate((1e-308, 1.0))]
    uneven = execution_replay_report(data)
    assert uneven["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] == pytest.approx(100)
    data["evidence"].reverse()
    reversed_order = execution_replay_report(data)
    assert reversed_order["accuracy"]["heldout"]["fine_isolated"]["wape_pct"] == pytest.approx(100)


@pytest.mark.parametrize("field,value", [
    ("timing_scope", "host_wall"), ("timer", "host_wall"),
    ("execution_mode", "graph"), ("calls_per_sample", 2),
    ("calls_per_sample", True), ("includes_dispatch", False),
    ("completion", "iteration_complete"), ("warmup", "cold"),
])
def test_protocol_scope_is_not_wildcarded_or_silently_normalized(field, value):
    data = fixture()
    data["protocol"][field] = value
    with pytest.raises(A3InputError, match="protocol_"):
        execution_replay_report(data)


def test_empty_protocol_or_context_scope_mismatch_is_rejected():
    data = fixture()
    data["protocol"] = {}
    with pytest.raises(A3InputError, match="protocol_incomplete"):
        execution_replay_report(data)
    data = fixture()
    data["operator_plan"]["context"] = {"timing_scope": "device_event_warm_per_call"}
    with pytest.raises(A3InputError, match="context_scope_mismatch"):
        execution_replay_report(data)


def test_cli_original_json_and_structured_failure(tmp_path):
    output = tmp_path / "replay.json"
    run = subprocess.run([
        sys.executable, "-m", "mindie_sim", "execution-replay", "--input",
        str(ROOT / "examples/execution-replay/mixed.json"), "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["command"] == "execution-replay" and result["status"] == "modeled"
    assert result["input_sha256"]
    bad = fixture()
    bad["protocol"]["includes_dispatch"] = False
    bad_input = tmp_path / "bad.json"
    bad_input.write_text(json.dumps(bad), encoding="utf-8")
    failed = subprocess.run([sys.executable, "-m", "mindie_sim", "execution-replay", "--input", str(bad_input)],
                            cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert failed.returncode == 1
    assert json.loads(failed.stdout)["reason"] == "execution_replay_protocol_unsupported"
