"""A5 communication binding, regime conversion, and streaming controls.

[Role] Verify Pass-1 and Pass-2 A5 communication binding and streaming invariants.
[Invariants] Unconfirmed AICPU/AIV requests never promote to confirmed regimes,
  AICPU and AIV never alias, non-A5 hardware is rejected, and boundaries stay
  strictly isolated across ingestion, curve lookup, recipes, and regimes.
[Timeouts & Locks] In-memory and tmp_path tests only; no locks or unbounded waits.
[Side-Effect Policy] Pure unit/boundary tests; never initializes a device.
"""
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.observations import a5_comm_ingest as ingest_module
from mindie_sim.observations.a5_comm_binding import (
    _A5CommPlan, _iter_json_events, _read_stream_value, _wire_factor,
)
from mindie_sim.observations.a5_comm_calibration import (
    A5_ARCH_FAMILY, A5_HARDWARE_KEY, A5_SKU, A5CommCalibrationBundle,
    bind_service_curve, build_a5_comm_recipe, build_a5_followup_manifest,
    compute_a5_contention_slowdown, evaluate_a5_mc2_chains,
    fit_a5_service_curves, ingest_a5_campaign_data, resolve_expansion_regime,
    to_service_regime,
)
from mindie_sim.observations.calibration import (
    bind_service_curve as reexported_bind_service_curve,
)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.services import ResourceService, service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from test_mechanism_contract import hardware_case
from test_a5_comm_calibration import (
    A5_SOURCE_CASE, COLLECTIVES, HOLDOUT_PAYLOADS, LIVE_A5, MIB, ROOT, STAGES,
    WORLD_SIZES, _a5_case, _a5_comm_op, _campaign, _comm_context, _fit_campaign,
    _mc2_chain, _measurement_op, _mode_conditions, _pipeline_campaign,
    _raw_point, _scope_for, embedded,
)

def test_finding1_explicit_heldout_split_is_never_overwritten():
    document = _pipeline_campaign("heldout-4m", [
        {"collective": "all_gather", "payload_bytes": 1024, "median_ms": 0.05,
         "run_id": "run-1", "repeat_index": 0, "split": "train"},
        {"collective": "all_gather", "payload_bytes": 1048576, "median_ms": 0.06,
         "run_id": "run-1", "repeat_index": 1, "split": "train"},
        {"collective": "all_gather", "payload_bytes": 4194304, "median_ms": 9.0,
         "run_id": "run-2", "repeat_index": 0, "split": "heldout"},
    ])
    ingested = ingest_a5_campaign_data(document)
    heldout_row = next(row for row in ingested["conditions"]
                       if row["payload_bytes"] == 4194304)
    assert heldout_row["split"] == "heldout"
    assert all(row["split"] in {None, "train", "heldout"}
               for row in ingested["conditions"])

    bundle = fit_a5_service_curves(ingested)
    curve = bundle.curves[("all_gather", 2, "intra_node_ub", "bf16", "unspecified")]
    assert 4194304 not in {work for work, _ in curve.knots}
    assert {row["work"] for row in bundle.holdout} == {4194304}

    mutated = deepcopy(ingested)
    next(row for row in mutated["conditions"]
         if row["payload_bytes"] == 4194304)["per_op_ms"] = 999.0
    refit = fit_a5_service_curves(mutated)
    assert refit.curves[("all_gather", 2, "intra_node_ub", "bf16",
                         "unspecified")].model_id == curve.model_id


def test_finding1_conflicting_repeat_split_raises():
    document = _pipeline_campaign("conflicting-split", [
        {"collective": "all_gather", "payload_bytes": 1024, "median_ms": 0.05,
         "run_id": "same-run", "repeat_index": 0, "split": "train"},
        {"collective": "all_gather", "payload_bytes": 1048576, "median_ms": 0.06,
         "run_id": "same-run", "repeat_index": 0, "split": "train"},
        {"collective": "all_gather", "payload_bytes": 1048576, "median_ms": 0.07,
         "run_id": "same-run", "repeat_index": 0, "split": "heldout"},
    ])
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        ingest_a5_campaign_data(document)


def test_finding2_stack_curves_are_partitioned_never_pooled(embedded):
    _, bundle = embedded
    base = ("all_gather", 8, "intra_node_ub", "bf16", "unspecified")
    payload = 128 * MIB

    default = bundle.curves[base]
    rack06 = bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16",
        stack="rack06")
    rack08 = bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16",
        stack="rack08")
    six = bundle.lookup_curve(base + ("rack08",))

    assert default.estimate(payload).duration_ms == pytest.approx(3.059270)
    assert rack06.estimate(payload).duration_ms == pytest.approx(3.059270)
    assert rack08.estimate(payload).duration_ms == pytest.approx(1.931796)
    assert six.model_id == rack08.model_id
    assert default.model_id != rack08.model_id
    assert bundle.stack_curves[base + ("rack06",)].model_id == rack06.model_id
    assert bundle.stack_curves[base + ("rack08",)].model_id == rack08.model_id
    assert rack06.estimate(payload).duration_ms != pytest.approx(2.4955)
    assert bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16",
        cann="9.1.0").model_id == rack06.model_id


def test_finding2_condition_rows_preserve_software_selectors():
    document = _campaign(label="software", stack="rack06",
                         machine={"cann": "9.1.0"}, algorithm="ring",
                         protocol={"kind": "hccl", "algorithm": "ring"})
    ingested = ingest_a5_campaign_data(document)
    assert {row["stack"] for row in ingested["conditions"]} == {"rack06"}
    assert {row["cann"] for row in ingested["conditions"]} == {"9.1.0"}
    assert {row["algorithm"] for row in ingested["conditions"]} == {"ring"}
    assert ingested["conditions"][0]["protocol"] == {
        "kind": "hccl", "algorithm": "ring"}


def test_finding2_ambiguous_custom_stacks_require_selector():
    documents = [
        _campaign(label="custom-a", stack="custom-a", machine={"cann": "9.9"},
                  conditions=[
                      {"collective": "all_reduce", "payload_bytes": 1024,
                       "median_ms": 0.05, "run_id": "a"},
                      {"collective": "all_reduce", "payload_bytes": 1048576,
                       "median_ms": 0.06, "run_id": "a"}]),
        _campaign(label="custom-b", stack="custom-b", machine={"cann": "9.9"},
                  conditions=[
                      {"collective": "all_reduce", "payload_bytes": 1024,
                       "median_ms": 0.07, "run_id": "b"},
                      {"collective": "all_reduce", "payload_bytes": 1048576,
                       "median_ms": 0.08, "run_id": "b"}]),
    ]
    with pytest.raises(
            ModelingError,
            match="a5_incompatible_software_stacks_require_selector"):
        fit_a5_service_curves(ingest_a5_campaign_data(documents))


def test_finding3_bind_service_curve_matches_engine_identity(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    parsed = parse_case(raw)

    curve = bundle.bind_service_curve(parsed, operation)
    assert curve.identity == service_identity(parsed, operation, "none")
    assert curve.resource == "link"
    assert curve.work_unit == "bytes"
    assert curve.boundary == "device_kernel"
    assert curve.active_units == parsed.machine.dies[0].units("link")
    assert isinstance(ResourceService(curve), ResourceService)
    assert bind_service_curve(bundle, parsed, operation).identity == curve.identity
    assert reexported_bind_service_curve is bind_service_curve

    phase_operation = replace(
        operation, metadata=operation.metadata + (("device_kernel_group",
                                                   {"group": "collective"}),))
    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.bind_service_curve(parsed, phase_operation)


def test_pass3_finding3_phase_binding_selects_only_measured_boundary(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    parsed = parse_case(raw)
    phase_operation = replace(
        operation, metadata=operation.metadata + (("device_kernel_group",
                                                   {"group": "collective"}),))

    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.bind_service_curve(parsed, phase_operation)

    conditions = []
    for boundary, intercept in (("device_kernel", 0.05),
                                ("resource_phase", 0.10)):
        for payload in (1024, 1048576, 4194304):
            conditions.append({
                "collective": "all_gather", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "boundary": boundary,
                "dtype": "bf16", "run_id": f"{boundary}-{payload}",
                "split": "train"})
    phase_bundle = _fit_campaign(_campaign(
        label="phase-bind", conditions=conditions, world_size=8,
        placement="intra-node-ub"))
    source = phase_bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16",
        boundary="resource_phase")
    phase_curve = phase_bundle.bind_service_curve(parsed, phase_operation)
    device_curve = phase_bundle.bind_service_curve(parsed, operation)

    assert phase_curve.boundary == "resource_phase"
    assert phase_curve.provenance["bound_a5_curve"] == source.model_id
    assert phase_curve.identity == service_identity(
        parsed, phase_operation, "none")
    assert device_curve.boundary == "device_kernel"
    device_source = phase_bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16",
        boundary="device_kernel")
    payload = operation.link_bytes / _wire_factor("all_gather", 8)
    assert phase_curve.estimate(operation.link_bytes).duration_ms == (
        pytest.approx(source.estimate(payload).duration_ms))
    assert source.estimate(payload).duration_ms > device_source.estimate(
        payload).duration_ms
    assert phase_curve.estimate(operation.link_bytes).duration_ms > (
        device_curve.estimate(operation.link_bytes).duration_ms)


def test_finding3_to_service_regime_changes_operator_and_end_to_end_predict(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    rule = bundle.to_service_regime(raw, operation)
    assert rule["response"]["kind"] == "piecewise_linear"
    assert rule["response"]["knots"][0][1] == pytest.approx(
        bundle.curves[("all_gather", 8, "intra_node_ub", "bf16",
                       "unspecified")].knots[0][1] * 1000.0)
    assert to_service_regime(bundle, raw, operation)["response"] == rule["response"]

    control = graph_costs(raw, (operation,), input_ready_ms=7)
    calibrated = deepcopy(raw)
    calibrated.setdefault("runtime", {}).setdefault("kernels", {})[
        "service_regimes"] = [rule]
    treated = graph_costs(calibrated, (operation,), input_ready_ms=7)
    assert treated["duration_ms"] > control["duration_ms"]
    queries = treated["parameters"]["service_regimes"][0]["queries"]
    assert queries.get("observable_response") == 1

    baseline_prediction = predict(raw)
    calibrated_prediction = predict(calibrated)
    assert calibrated_prediction["metrics"]["e2e_ms"]["p50"] != pytest.approx(
        baseline_prediction["metrics"]["e2e_ms"]["p50"])


@pytest.mark.parametrize("mode", ["aiv", "aicpu"])
def test_finding3_recipe_schedules_aiv_vector_or_aicpu_hbm(embedded, mode):
    _, bundle = embedded
    raw = _a5_case()
    comm = OpSpec("comm", "allgather", "prefill", 0, 0, (), (1,), (1,), (1,),
                  (1,), link_bytes=401408, dtype="bf16", recipe="a5comm",
                  stream="other")
    vector = OpSpec("vector", "vector", "prefill", 0, 0, (), (1,), (1,), (1,),
                    (1,), vector_ops=100_000_000, dtype="bf16", recipe="vector")
    recipe = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True)
    plan = recipe.build(comm, None)
    assert plan.metadata["unspecified_fallback"] is True
    assert build_a5_comm_recipe(
        bundle, "all_gather", world_size=8,
        expansion_mode=mode, allow_unspecified_fallback=True).build is not None
    book = RecipeBook({"a5comm": recipe})

    baseline = graph_costs(raw, (vector,))
    concurrent = graph_costs(raw, (comm, vector), recipes=book)
    vector_row = next(row for row in concurrent["timeline"]
                      if row["op_id"] == "vector")
    comm_row = next(row for row in concurrent["timeline"] if row["op_id"] == "comm")
    base_vector = baseline["execution"]["serviced"]["vector@r0:s0"]
    concurrent_vector = concurrent["execution"]["serviced"]["vector@r0:s0"]
    if mode == "aiv":
        assert concurrent_vector > base_vector
        assert vector_row["complete_ms"] > baseline["timeline"][0]["complete_ms"]
        assert comm_row["complete_ms"] > 0
    else:
        assert concurrent_vector == pytest.approx(base_vector)
        assert vector_row["complete_ms"] == pytest.approx(
            baseline["timeline"][0]["complete_ms"])
        assert comm_row["complete_ms"] > 0


def test_finding4_requested_mode_never_resolves_ambiguous_evidence():
    evidence = ["hcom_allReduce_AicpuKernel", "AI_VECTOR_CORE"]
    for requested in ("aicpu", "aiv", "unspecified"):
        resolution = resolve_expansion_regime(
            requested, "not_observed", profiler_evidence=evidence)
        assert resolution["regime"] == "unspecified"
        assert resolution["evidence_conflict"] is True
        assert resolution["expansion_unconfirmed"] is True
        assert resolution["confirmed_by"] is None


def test_finding5_mc2_observed_regime_never_inferred_from_dtype():
    document = _pipeline_campaign("mc2-regimes", [], chains=[
        _mc2_chain("int8-aicpu", observed_expansion_mode="aicpu"),
        _mc2_chain("int8-aiv", observed_expansion_mode="aiv"),
        _mc2_chain("int8-unconfirmed", requested_expansion_mode="aiv",
                   observed_expansion_mode="not_observed"),
    ])
    ingested = ingest_a5_campaign_data(document)
    by_id = {row["condition"]: row for row in ingested["mc2_chains"]}
    assert by_id["int8-aicpu"]["regime"] == "aicpu"
    assert by_id["int8-aiv"]["regime"] == "aiv"
    assert by_id["int8-unconfirmed"]["regime"] == "unspecified"
    assert by_id["int8-unconfirmed"]["expansion_unconfirmed"] is True
    assert by_id["int8-unconfirmed"]["unconfirmed_tag"] == (
        "requested_aiv_unconfirmed")
    summary = evaluate_a5_mc2_chains(ingested)
    rows = {row["condition"]: row for row in summary["rows"]}
    assert rows["int8-aicpu"]["expansion_regime"] == "aicpu"
    assert rows["int8-aicpu"]["vector_slowdown"] == 1.0
    assert rows["int8-aiv"]["expansion_regime"] == "aiv"
    assert rows["int8-aiv"]["vector_slowdown"] > 1.0
    assert rows["int8-unconfirmed"]["expansion_regime"] == "unspecified"
    assert rows["int8-unconfirmed"]["vector_slowdown"] == 1.0
    assert rows["int8-unconfirmed"]["mte_slowdown"] > 1.0
    assert rows["int8-unconfirmed"]["contention_degradation_multiplier"] > 1.0


def test_finding6_explicit_p2p_scope_and_pair_stay_separate():
    rows = []
    for pair, scope, base in (("0->8", "cross_node_urma", 0.07),
                              ("0->24", "cross_node_urma_far", 0.10)):
        for payload, duration in ((1024, base), (1048576, base * 2)):
            rows.append({"collective": "p2p", "world_size": 32,
                         "payload_bytes": payload, "median_ms": duration,
                         "p2p_pair": pair, "placement_scope": scope,
                         "dtype": "bf16", "run_id": pair, "split": "train"})
    document = _pipeline_campaign("w32-p2p", rows, world_size=32,
                                  placement="a5-ub-mesh-intra-node")
    ingested = ingest_a5_campaign_data(document)
    pairs = {(row["pair"], row["placement_scope"])
             for row in ingested["conditions"]}
    assert pairs == {("0->8", "cross_node_urma"),
                     ("0->24", "cross_node_urma_far")}
    assert all(row["pair"] != "0->1" for row in ingested["conditions"])

    bundle = fit_a5_service_curves(ingested)
    cross = bundle.curves[("p2p_send_recv", 32, "cross_node_urma", "bf16",
                           "unspecified")]
    far = bundle.curves[("p2p_send_recv", 32, "cross_node_urma_far", "bf16",
                         "unspecified")]
    assert cross.identity["p2p_pair"] == "0->8"
    assert far.identity["p2p_pair"] == "0->24"
    assert cross.model_id != far.model_id


def test_finding6_rank_case_explicit_pair_and_scope_survive():
    rank = {"role": "rank", "status": "completed", "label": "w32-rank",
            "world_size": 32, "dtype": "bf16",
            "placement": "a5-ub-mesh-intra-node",
            "cases": [{"case_id": "p2p-far", "collective": "p2p",
                       "payload_bytes": 1024,
                       "per_op_ms_summary": {"median": 0.09},
                       "p2p_pair": [0, 24], "placement_scope": "far",
                       "status": "completed",
                       "correctness": {"expected": 1, "observed": 1}}]}
    ingested = ingest_a5_campaign_data(rank)
    row = ingested["conditions"][0]
    assert row["pair"] == "0->24"
    assert row["placement_scope"] == "cross_node_urma_far"


def test_finding7_reimport_and_distinct_run_ids_do_not_shift_knots():
    rows = [
        {"collective": "all_reduce", "payload_bytes": 1048576, "median_ms": 0.06,
         "run_id": "run-1", "repeat_index": 0},
        {"collective": "all_reduce", "payload_bytes": 4194304, "median_ms": 0.10,
         "run_id": "run-1", "repeat_index": 0},
        {"collective": "all_reduce", "payload_bytes": 1048576, "median_ms": 0.08,
         "run_id": "run-2", "repeat_index": 0},
        {"collective": "all_reduce", "payload_bytes": 4194304, "median_ms": 0.12,
         "run_id": "run-2", "repeat_index": 0},
    ]
    ingested = ingest_a5_campaign_data(_pipeline_campaign("reimport", rows))
    assert len(ingested["conditions"]) == 4
    first = fit_a5_service_curves(ingested)
    twice = deepcopy(ingested)
    twice["conditions"].extend(deepcopy(ingested["conditions"]))
    second = fit_a5_service_curves(twice)
    key = ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")
    assert first.curves[key].knots == second.curves[key].knots
    assert first.curves[key].model_id == second.curves[key].model_id
    knots = dict(first.curves[key].knots)
    assert knots[1048576] == pytest.approx(0.07)
    assert knots[4194304] == pytest.approx(0.11)


def test_finding7_duplicate_identity_with_conflicting_duration_raises():
    rows = [
        {"collective": "all_reduce", "payload_bytes": 1048576, "median_ms": 0.06,
         "run_id": "run-1", "repeat_index": 0},
        {"collective": "all_reduce", "payload_bytes": 4194304, "median_ms": 0.10,
         "run_id": "run-1", "repeat_index": 0},
        {"collective": "all_reduce", "payload_bytes": 4194304, "median_ms": 0.11,
         "run_id": "run-1", "repeat_index": 0},
    ]
    with pytest.raises(ValueError, match="conflicting_primitive_repeat_or_split"):
        ingest_a5_campaign_data(_pipeline_campaign("duplicate-duration", rows))


@pytest.mark.parametrize(("patch", "reason"), [
    ({"status": "failed"}, "a5_campaign_not_completed"),
    ({"canary_ok": False}, "a5_campaign_canary_failed"),
    ({"canary": "mismatch"}, "a5_campaign_canary_failed"),
    ({"correctness": {"expected": 36, "observed": 35}},
     "a5_campaign_canary_failed"),
    ({"hardware_key": "Ascend910_93"},
     "a5_calibration_rejects_non_a5_hardware"),
    ({"sku": "Ascend910B"}, "a5_calibration_rejects_non_a5_hardware"),
    ({"arch_family": "A3"}, "a5_calibration_rejects_non_a5_hardware"),
])
def test_finding8_condition_row_status_canary_and_hardware_are_validated(patch, reason):
    document = _campaign()
    document["conditions"][0].update(patch)
    with pytest.raises((ValueError, ModelingError), match=reason):
        ingest_a5_campaign_data(document)


def test_finding8_rank_and_mc2_rows_are_validated():
    rank = {"role": "rank", "status": "completed", "label": "bad-rank",
            "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
            "cases": [{"case_id": "bad", "collective": "all_reduce",
                       "payload_bytes": 1024,
                       "per_op_ms_summary": {"median": 0.05},
                       "correctness": {"expected": 1, "observed": 2}}]}
    with pytest.raises(ValueError, match="a5_campaign_canary_failed"):
        ingest_a5_campaign_data(rank)

    chain = _pipeline_campaign("bad-chain", [], chains=[
        _mc2_chain("bad-canary", canary_ok=False)])
    with pytest.raises(ValueError, match="a5_campaign_canary_failed"):
        ingest_a5_campaign_data(chain)

    chain = _pipeline_campaign("bad-chain-arch", [], chains=[
        _mc2_chain("bad-arch", arch_family="A3")])
    with pytest.raises(ModelingError,
                       match="a5_calibration_rejects_non_a5_hardware"):
        ingest_a5_campaign_data(chain)


def test_finding9_directory_ingestion_keeps_comm_lane(tmp_path):
    chain = [_mc2_chain("int8-ep2-e64-g512")]
    lane = {"raw_points": [
        {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.5},
        {"dtype": "int8", "ep": 2, "link_bytes": 2048, "ms": 0.7}]}
    (tmp_path / "a5-mc2-quant-chain-validation.json").write_text(
        json.dumps(chain), encoding="utf-8")
    (tmp_path / "a5-mc2-quant-comm-lane.json").write_text(
        json.dumps(lane), encoding="utf-8")

    ingested = ingest_a5_campaign_data(tmp_path)
    assert {row["collective"] for row in ingested["conditions"]} == {"all_to_allv"}
    assert {row["payload_bytes"] for row in ingested["conditions"]} == {1024, 2048}
    assert len(ingested["mc2_chains"]) == 1

    (tmp_path / "a5-collective-calibration.json").write_text(json.dumps(
        _campaign(label="calibration", conditions=[
            {"collective": "all_reduce", "payload_bytes": 1024,
             "median_ms": 0.05},
            {"collective": "all_reduce", "payload_bytes": 1048576,
             "median_ms": 0.06}])), encoding="utf-8")
    (tmp_path / "capture.parent.json").write_text(json.dumps({
        "role": "parent", "label": "dropped-parent", "status": "completed",
        "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub"}), encoding="utf-8")
    with_calibration = ingest_a5_campaign_data(tmp_path)
    assert "dropped-parent" not in with_calibration["campaign_labels"]
    assert "a5-mc2-quant-comm-lane" in with_calibration["campaign_labels"]
    assert {row["payload_bytes"] for row in with_calibration["conditions"]
            if row["collective"] == "all_to_allv"} == {1024, 2048}


def test_finding10_large_json_streams_without_json_load(tmp_path, monkeypatch):
    payload = {
        "schema_version": "mindie-comm-campaign-a5/1",
        "role": "rank", "status": "completed", "label": "large-rank",
        "world_size": 8, "dtype": "bf16", "placement": "intra-node-ub",
        "cases": [{
            "case_id": "large", "collective": "all_gather", "dtype": "bf16",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1},
            "samples": [{"per_op_ms": 0.05} for _ in range(1024)],
            "raw_iterations_us": [50.0 for _ in range(100_000)],
        }],
    }
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert ingest_module._LARGE_JSON_BYTES == 50 * 1024 * 1024
    monkeypatch.setattr(ingest_module, "_LARGE_JSON_BYTES", 64)
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("json.load must not read a file above 50 MiB")
    monkeypatch.setattr(ingest_module.json, "load", forbidden)

    loaded = ingest_module._load_json(path)
    assert calls == []
    assert loaded["cases"][0]["per_op_ms_summary"]["median"] == 0.05
    assert "samples" not in loaded["cases"][0]
    assert "raw_iterations_us" not in loaded["cases"][0]
    ingested = ingest_a5_campaign_data(path)
    assert ingested["conditions"][0]["payload_bytes"] == 1024
    assert calls == []



@pytest.mark.parametrize("mode", ["aiv", "aicpu"])
def test_pass2_finding1_recipe_preserves_link_work_and_phases(embedded, mode):
    _, bundle = embedded
    link_bytes = 401408.0
    op = OpSpec("comm", "allgather", "prefill", 0, 0, (), (1,), (1,), (1,),
                (1,), link_bytes=link_bytes, hbm_bytes=4096, dtype="bf16",
                recipe="a5comm")
    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True).build(op, None)

    assert isinstance(plan, _A5CommPlan)
    assert plan.metadata["unspecified_fallback"] is True
    assert plan.work.apply(op).link_bytes == pytest.approx(link_bytes)
    program = plan.program(_comm_context(op))
    stages = {stage.name: stage for stage in program.stages}
    assert stages["link"].work.get("link:port0") == pytest.approx(link_bytes)
    assert "hbm" in stages
    if mode == "aiv":
        assert stages["vector"].placement
        assert plan.minimum_ms > 0
    else:
        assert "vector" not in stages
        assert plan.minimum_ms > 0

    lowered = plan.lower(_comm_context(op))
    assert lowered.minimum_ms == pytest.approx(
        max(_comm_context(op).minimum_ms, plan.minimum_ms))
    assert lowered.stages[1].program is not None


def test_pass2_finding2_wire_factor_keeps_captured_coordinates(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    source = bundle.lookup_curve(
        "all_gather", 8, placement_scope="intra_node_ub", dtype="bf16")
    bound = bundle.bind_service_curve(parse_case(raw), operation)

    assert _wire_factor("all_gather", 8) == 7
    assert _wire_factor("all_reduce", 8) == pytest.approx(2 * 7 / 8)
    assert _wire_factor("all_to_all", 8) == pytest.approx(7 / 8)
    derived = bundle.lookup_curve(
        "all_to_allv", 8, placement_scope="intra_node_ub", dtype="bf16")
    measured = bundle.lookup_curve(
        "all_to_allv", 2, placement_scope="intra_node_ub", dtype="int8")
    assert derived.provenance["derived_from"] == "all_to_all"
    assert _wire_factor("all_to_allv", 8, derived) == pytest.approx(7 / 8)
    assert _wire_factor("all_to_allv", 2, measured) == 1.0
    assert _wire_factor("p2p_send_recv", 8) == 1.0

    assert bound.provenance["wire_factor"] == pytest.approx(7)
    assert tuple(bound.provenance["raw_knots"]) == tuple(source.knots)
    for (raw_work, _), (wire_work, _) in zip(source.knots, bound.knots):
        assert wire_work == pytest.approx(raw_work * 7)
    rule = bundle.to_service_regime(raw, operation)
    assert rule["coordinates"]["bytes"]["scale"] == pytest.approx(1 / 7)
    assert rule["coordinates"]["bytes"]["path"] == "work.link"
    assert rule["domain"]["bytes"]["min"] == pytest.approx(
        source.knots[0][0])
    assert rule["response"]["knots"][0][0] == pytest.approx(
        source.knots[0][0])

    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode="aicpu",
        allow_unspecified_fallback=True).build(operation, None)
    assert plan.metadata["unspecified_fallback"] is True
    lookup = source.estimate(operation.link_bytes / 7).duration_ms
    contention = compute_a5_contention_slowdown("aicpu")
    assert plan.minimum_ms == pytest.approx(max(
        lookup, contention["transfer_ms"] + contention["sync_ms"]))


def test_pass2_finding3_parent_aggregation_never_mixes_launch_anchors(tmp_path):
    parent = {
        "role": "parent", "label": "anchor-parent", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub",
    }
    (tmp_path / "anchor.parent.json").write_text(
        json.dumps(parent), encoding="utf-8")
    for rank in (0, 1):
        document = {
            "role": "rank", "world_size": 2, "dtype": "bf16",
            "placement": "intra-node-ub", "cases": [
                {"case_id": "steady", "collective": "all_reduce",
                 "payload_bytes": 1024,
                 "per_op_ms_summary": {"median": 0.05},
                 "correctness": {"expected": 1, "observed": 1}},
                {"case_id": "launch", "collective": "all_reduce",
                 "payload_bytes": 1024, "single": True,
                 "per_op_ms_summary": {"median": 10.0},
                 "correctness": {"expected": 1, "observed": 1}},
            ]}
        (tmp_path / f"anchor.rank{rank}.json").write_text(
            json.dumps(document), encoding="utf-8")

    ingested = ingest_a5_campaign_data(tmp_path)

    assert len(ingested["conditions"]) == 1
    assert ingested["conditions"][0]["single"] is False
    assert ingested["conditions"][0]["per_op_ms"] == pytest.approx(0.05)
    assert len(ingested["single_op_anchors"]) == 1
    assert ingested["single_op_anchors"][0]["single"] is True
    assert ingested["single_op_anchors"][0]["per_op_ms"] == pytest.approx(10.0)


def test_pass2_finding4_stream_preserves_sample_level_splits(
        tmp_path, monkeypatch):
    payload = {
        "role": "rank", "status": "completed", "label": "stream-splits",
        "world_size": 8, "dtype": "bf16", "placement": "intra-node-ub",
        "cases": [{
            "case_id": "samples", "collective": "all_gather",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1},
            "samples": [
                {"payload_bytes": 1024, "per_op_ms": 0.05,
                 "split": "train", "run_id": "stream-1", "repeat_index": 0},
                {"payload_bytes": 1048576, "per_op_ms": 0.06,
                 "split": "train", "run_id": "stream-1", "repeat_index": 1},
                {"payload_bytes": 4194304, "per_op_ms": 9.0,
                 "split": "heldout", "run_id": "stream-2", "repeat_index": 0},
            ],
            "raw_iterations_us": [50.0 for _ in range(1000)],
        }],
    }
    path = tmp_path / "rank.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ingest_module, "_LARGE_JSON_BYTES", 64)

    loaded = ingest_module._load_json(path)
    samples = loaded["cases"][0]["samples"]
    assert {sample["split"] for sample in samples} == {"train", "heldout"}
    assert all(set(sample) & {"split", "run_id", "repeat_index"}
               for sample in samples)
    assert "raw_iterations_us" not in loaded["cases"][0]

    ingested = ingest_a5_campaign_data(path)
    by_payload = {row["payload_bytes"]: row for row in ingested["conditions"]}
    assert by_payload[1024]["split"] == "train"
    assert by_payload[1048576]["split"] == "train"
    assert by_payload[4194304]["split"] == "heldout"
    bundle = fit_a5_service_curves(ingested)
    curve = bundle.curves[("all_gather", 8, "intra_node_ub", "bf16",
                           "unspecified")]
    assert 4194304 not in {work for work, _ in curve.knots}
    assert {row["work"] for row in bundle.holdout} == {4194304}


def test_pass2_finding4_stdlib_chunked_stream_is_used_without_ijson(
        tmp_path, monkeypatch):
    payload = {
        "role": "rank", "status": "completed", "label": "fallback",
        "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
        "cases": [{
            "case_id": "case", "collective": "all_reduce",
            "payload_bytes": 1024,
            "per_op_ms_summary": {"median": 0.05},
            "correctness": {"expected": 1, "observed": 1},
            "samples": [
                {"per_op_ms": 0.05, "run_id": "fallback", "split": "train",
                 "repeat_index": 0, "payload_bytes": 1024},
                {"per_op_ms": 0.06, "run_id": "fallback", "split": "train",
                 "repeat_index": 1, "payload_bytes": 1048576},
            ],
        }],
    }
    path = tmp_path / "fallback.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(ingest_module, "_LARGE_JSON_BYTES", 64)
    monkeypatch.setitem(sys.modules, "ijson", None)

    loaded = ingest_module._load_json(path)
    assert loaded["label"] == "fallback"
    assert loaded["cases"][0]["per_op_ms_summary"]["median"] == 0.05
    assert len(loaded["cases"][0]["samples"]) == 2
    ingested = ingest_a5_campaign_data(path)
    assert {row["split"] for row in ingested["conditions"]} == {"train"}


def test_pass2_finding5_link_curve_executes_without_named_link_resource(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    curve = bundle.bind_service_curve(raw, operation)
    assert "link" not in parse_case(raw).machine.dies[0].resource_units

    def predicted(case_raw):
        bound = bundle.bind_service_curve(case_raw, operation)
        row = operator_costs(
            case_raw, (operation,),
            recipes=RecipeBook({operation.recipe: ResourceService(bound)})
        )["operators"][0]
        return row, bound

    row, _ = predicted(raw)
    estimate = curve.estimate(operation.link_bytes)
    assert row["duration_ms"] - estimate.duration_ms > 0
    assert any(name.startswith("link:") for name in row["serviced"])
    assert row["basis"]["service_models"][0]["queries"].get(
        "interpolated") == 1

    link = dict(parse_case(raw).machine.links[0].to_dict())

    def with_latency(latency_us):
        case_raw = deepcopy(raw)
        case_raw["machine"]["links"] = [{**link, "latency_us": latency_us}]
        return case_raw

    faster_raw = with_latency(0.1)
    slower_raw = with_latency(10.0)
    faster_row, _ = predicted(faster_raw)
    slower_row, _ = predicted(slower_raw)

    assert faster_row["duration_ms"] < row["duration_ms"] < slower_row["duration_ms"]
