"""A5 communication campaign ingestion and transport identity controls.

[Role] Verify Pass-2 and Pass-3 A5 campaign ingestion and transport identity.
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

def test_pass2_finding6_binding_rejects_non_a5_everywhere(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    reason = "a5_calibration_rejects_non_a5_hardware"

    with pytest.raises(ModelingError, match=reason):
        bundle.bind_service_curve(parse_case(hardware_case()), operation)
    with pytest.raises(ModelingError, match=reason):
        bundle.to_service_regime(parse_case(hardware_case()), operation)
    with pytest.raises(ModelingError, match=reason):
        bundle.build_a5_comm_recipe(
            "all_gather", world_size=8, hardware_key="Ascend910B")
    with pytest.raises(ModelingError, match=reason):
        bundle.build_a5_comm_recipe(
            "all_gather", world_size=8, arch_family="A3")

    recipe = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode="aicpu",
        allow_unspecified_fallback=True)
    bad_metadata = replace(
        operation,
        metadata=operation.metadata + (("hardware_key", "Ascend910B"),))
    with pytest.raises(ModelingError, match=reason):
        recipe.build(bad_metadata, None)
    plan = recipe.build(operation, None)
    assert plan.metadata["unspecified_fallback"] is True
    with pytest.raises(ModelingError, match=reason):
        plan.program(_comm_context(operation, service_identity={
            "hardware": {"key": "Ascend910B", "architecture": "A3"}}))


def test_pass2_finding7_regime_match_keeps_communication_domain(embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    rule = bundle.to_service_regime(raw, operation)
    identity = service_identity(parse_case(raw), operation, "none")
    metadata = dict(operation.metadata)

    assert rule["match"]["family"] == operation.family
    assert rule["match"]["transport"] == identity["transport"]
    present = [key for key in (
        "collective", "a5_collective", "group_axis", "steps", "world_size",
        "group_size", "ranks", "group_ranks", "placement_scope", "p2p_pair",
        "expansion_mode", "expansion_regime") if key in metadata]
    assert {"collective", "group_axis", "steps"} <= set(present)
    for key in present:
        assert rule["match"][f"metadata.{key}"] == metadata[key]
    assert rule["match"]["recipe"] == operation.recipe
    assert rule["match"]["hardware"] == identity["hardware"]["key"]


def test_pass2_finding8_software_variants_never_overwrite_each_other():
    conditions = []
    for cann, intercept, slope in (("9.1.0", 0.05, 1e-9),
                                   ("9.9", 0.08, 5e-9)):
        for payload in (1024, 1048576, 4194304):
            conditions.append({
                "collective": "all_reduce", "payload_bytes": payload,
                "median_ms": intercept + payload * slope, "cann": cann,
                "algorithm": "ring", "run_id": f"{cann}-{payload}",
                "split": "train"})
    bundle = _fit_campaign(
        _campaign(label="variants", stack="rack06", conditions=conditions))
    base = ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")
    primary = bundle.stack_curves[base + ("rack06",)]
    full = {base + ("rack06", cann, "ring", None): curve
            for cann, curve in (("9.1.0", primary), ("9.9", bundle.lookup_curve(
                "all_reduce", 2, cann="9.9")))}

    assert primary.identity["cann"] == "9.1.0"
    assert bundle.curves[base].identity["cann"] == "9.1.0"
    assert set(full) <= set(bundle.stack_curves)
    assert bundle.lookup_curve(
        "all_reduce", 2, cann="9.1.0").identity["cann"] == "9.1.0"
    assert bundle.lookup_curve(
        "all_reduce", 2, cann="9.9").identity["cann"] == "9.9"
    assert bundle.lookup_curve(
        "all_reduce", 2, cann="9.9").model_id != primary.model_id
    assert len({curve.model_id for key, curve in bundle.stack_curves.items()
                if key[:5] == base}) == 2


def test_pass2_finding9_explicit_p2p_pair_never_falls_back():
    rows = []
    for pair, intercept in (("0->1", 0.05), ("0->8", 0.09)):
        for payload in (1024, 1048576, 4194304):
            rows.append({
                "collective": "p2p", "world_size": 8,
                "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9,
                "p2p_pair": pair, "placement_scope": "intra_node_ub",
                "dtype": "bf16", "run_id": pair, "split": "train"})
    bundle = _fit_campaign(_campaign(
        label="pairs", conditions=rows, world_size=8,
        placement="intra-node-ub", stack="rack06"))
    base = ("p2p_send_recv", 8, "intra_node_ub", "bf16", "unspecified")

    assert bundle.curves[base].identity["p2p_pair"] == "0->1"
    near = bundle.lookup_curve(
        "p2p_send_recv", 8, placement_scope="intra_node_ub", pair="0->1")
    far = bundle.lookup_curve(
        "p2p_send_recv", 8, placement_scope="intra_node_ub", pair="0->8")
    assert near.identity["p2p_pair"] == "0->1"
    assert far.identity["p2p_pair"] == "0->8"
    assert near.model_id != far.model_id
    assert bundle.lookup_curve(
        "p2p_send_recv", 8, placement_scope="intra_node_ub",
        pair=("0", "8")).identity["p2p_pair"] == "0->8"
    with pytest.raises(ModelingError, match="a5_curve_unavailable"):
        bundle.lookup_curve(
            "p2p_send_recv", 8, placement_scope="intra_node_ub", pair="0->4")


def test_pass2_finding10_expanded_samples_and_raw_points_are_validated():
    document = _campaign(conditions=[{
        "collective": "all_reduce", "payload_bytes": 1024,
        "median_ms": 0.05,
        "samples": [{"per_op_ms": 0.05, "status": "failed"}]}])
    with pytest.raises(ValueError, match="a5_campaign_not_completed"):
        ingest_a5_campaign_data(document)

    document = _campaign(conditions=[{
        "collective": "all_reduce", "payload_bytes": 1024,
        "median_ms": 0.05,
        "samples": [{"per_op_ms": 0.05, "canary_ok": False}]}])
    with pytest.raises(ValueError, match="a5_campaign_canary_failed"):
        ingest_a5_campaign_data(document)

    rank = {"role": "rank", "status": "completed", "label": "sample-rank",
            "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
            "cases": [{"case_id": "sample", "collective": "all_reduce",
                       "payload_bytes": 1024,
                       "per_op_ms_summary": {"median": 0.05},
                       "samples": [{"per_op_ms": 0.05,
                                    "correctness": {"expected": 1,
                                                    "observed": 2}}]}]}
    with pytest.raises(ValueError, match="a5_campaign_canary_failed"):
        ingest_a5_campaign_data(rank)

    lane = {
        "label": "raw-points", "status": "completed", "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY, "arch_family": A5_ARCH_FAMILY,
        "world_size": 2, "raw_points": [
            {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.5,
             "run_id": "raw-a", "repeat_index": 0, "split": "train"},
            {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.6,
             "run_id": "raw-a", "repeat_index": 1, "split": "train"},
            {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.7,
             "run_id": "raw-b", "repeat_index": 0, "split": "heldout"},
        ]}
    ingested = ingest_a5_campaign_data(lane)
    assert len(ingested["conditions"]) == 3
    assert {(row["run_id"], row["repeat_index"], row["split"])
            for row in ingested["conditions"]} == {
        ("raw-a", 0, "train"), ("raw-a", 1, "train"), ("raw-b", 0, "heldout")}
    bad = deepcopy(lane)
    bad["raw_points"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="a5_campaign_not_completed"):
        ingest_a5_campaign_data(bad)


def test_pass2_finding11_out_of_domain_extrapolates_without_clamping_work():
    conditions = [
        {"collective": "all_reduce", "payload_bytes": payload,
         "median_ms": 0.05 + payload * 1e-9, "dtype": "bf16",
         "run_id": f"payload-{payload}", "split": "train"}
        for payload in (1024, 4096)]
    bundle = _fit_campaign(_campaign(label="domain", conditions=conditions))
    curve = bundle.curves[("all_reduce", 2, "intra_node_ub", "bf16",
                           "unspecified")]
    x0, t0 = curve.knots[0]
    x1, t1 = curve.knots[-1]
    slope = max((t1 - t0) / (x1 - x0), (t1 - curve.startup_ms) / x1)
    contention = compute_a5_contention_slowdown("aicpu")
    floor = contention["transfer_ms"] + contention["sync_ms"]

    above = OpSpec("comm", "allreduce", "prefill", 0, 0, (), (1,), (1,),
                   (1,), (1,), link_bytes=1_000_000, dtype="bf16",
                   recipe="a5comm")
    plan = bundle.build_a5_comm_recipe(
        "all_reduce", world_size=2, expansion_mode="aicpu",
        allow_unspecified_fallback=True).build(above, None)
    assert plan.metadata["unspecified_fallback"] is True
    assert plan.work.apply(above).link_bytes == pytest.approx(1_000_000)
    assert plan.minimum_ms == pytest.approx(max(
        t1 + slope * (1_000_000 - x1), floor))

    below = OpSpec("comm", "allreduce", "prefill", 0, 0, (), (1,), (1,),
                   (1,), (1,), link_bytes=10, dtype="bf16", recipe="a5comm")
    plan = bundle.build_a5_comm_recipe(
        "all_reduce", world_size=2, expansion_mode="aicpu",
        allow_unspecified_fallback=True).build(below, None)
    assert plan.metadata["unspecified_fallback"] is True
    assert plan.work.apply(below).link_bytes == pytest.approx(10)
    assert plan.minimum_ms == pytest.approx(max(
        curve.startup_ms + (t0 - curve.startup_ms) * 10 / x0, floor))


def test_pass2_finding12_recipe_uses_operator_dtype():
    conditions = []
    for dtype, intercept in (("bf16", 0.05), ("int8", 0.10)):
        for payload in (1024, 4096):
            conditions.append({
                "collective": "all_reduce", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "dtype": dtype,
                "run_id": f"{dtype}-{payload}", "split": "train"})
    bundle = _fit_campaign(_campaign(label="dtypes", conditions=conditions))
    op = OpSpec("comm", "allreduce", "prefill", 0, 0, (), (1,), (1,), (1,),
                (1,), link_bytes=2048, dtype="int8", recipe="a5comm")

    plan = bundle.build_a5_comm_recipe(
        "all_reduce", world_size=2, expansion_mode="aicpu",
        allow_unspecified_fallback=True).build(op, None)
    assert plan.metadata["unspecified_fallback"] is True
    int8 = bundle.lookup_curve("all_reduce", 2, dtype="int8")
    assert plan.minimum_ms == pytest.approx(int8.estimate(2048).duration_ms)

    plan = bundle.build_a5_comm_recipe(
        "all_reduce", world_size=2, expansion_mode="aicpu",
        dtype="bf16", allow_unspecified_fallback=True).build(op, None)
    assert plan.metadata["unspecified_fallback"] is True
    bf16 = bundle.lookup_curve("all_reduce", 2, dtype="bf16")
    assert plan.minimum_ms == pytest.approx(bf16.estimate(2048).duration_ms)


def test_pass2_finding13_boundaries_stay_separate():
    conditions = []
    for boundary, intercept in (("device_kernel", 0.05),
                                ("resource_phase", 0.10)):
        for payload in (1024, 1048576, 4194304):
            conditions.append({
                "collective": "all_reduce", "payload_bytes": payload,
                "median_ms": intercept + payload * 1e-9, "boundary": boundary,
                "dtype": "bf16", "run_id": boundary, "split": "train"})
    bundle = _fit_campaign(_campaign(
        label="boundaries", stack="rack06", conditions=conditions))
    base = ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified")

    assert bundle.curves[base].boundary == "device_kernel"
    phase = bundle.stack_curves[base + ("rack06", "resource_phase")]
    assert phase.boundary == "resource_phase"
    assert bundle.lookup_curve(
        "all_reduce", 2, boundary="device_kernel").boundary == "device_kernel"
    assert bundle.lookup_curve(
        "all_reduce", 2, boundary="resource_phase").model_id == phase.model_id
    assert bundle.lookup_curve("all_reduce", 2).boundary == "device_kernel"
    assert bundle.lookup_curve(
        "all_reduce", 2, stack="rack06").boundary == "device_kernel"
    assert phase.model_id != bundle.curves[base].model_id
    assert phase.estimate(1048576).duration_ms > bundle.curves[
        base].estimate(1048576).duration_ms

    bad = deepcopy(conditions)
    bad[0]["boundary"] = "call"
    with pytest.raises(
            ModelingError,
            match="service_curve_requires_device_or_resource_phase_boundary"):
        ingest_a5_campaign_data({**_campaign(label="bad-boundary"),
                                 "conditions": bad})


def test_pass2_finding14_mc2_dedupe_keeps_independent_observations():
    first = _mc2_chain("shared")
    second = _mc2_chain(
        "shared", legs_ms={"dispatch": 1.1, "gate_up": 1.0,
                           "swiglu_quant": 1.0, "down": 1.0, "combine": 1.0},
        chain_ms=4.1)
    ingested = ingest_a5_campaign_data(_pipeline_campaign(
        "independent-chains", [], chains=[first, second]))
    assert len(ingested["mc2_chains"]) == 2

    one = ingest_a5_campaign_data(_pipeline_campaign("campaign-a", [], chains=[
        _mc2_chain("same")]))
    other = ingest_a5_campaign_data(_pipeline_campaign("campaign-b", [], chains=[
        _mc2_chain("same")]))
    combined = ingest_a5_campaign_data([
        _pipeline_campaign("campaign-a", [], chains=[_mc2_chain("same")]),
        _pipeline_campaign("campaign-b", [], chains=[_mc2_chain("same")]),
    ])
    assert len(one["mc2_chains"]) == len(other["mc2_chains"]) == 1
    assert len(combined["mc2_chains"]) == 2
    assert {row["campaign"] for row in combined["mc2_chains"]} == {
        "campaign-a", "campaign-b"}


def test_pass2_finding4_stdlib_stream_survives_token_chunk_boundaries():
    raw = (b'{"a": [12345678901234567890, 1.2345678901234567, -9.5e12], '
           b'"b": {"c": true, "d": null, "e": "x\\\"y"}, "f": []}')
    expected = {
        "a": [12345678901234567890.0, 1.2345678901234567, -9.5e12],
        "b": {"c": True, "d": None, "e": 'x"y'}, "f": []}

    for chunk_size in (1, 2, 3, 5, 7, 64):
        events = _iter_json_events(io.BytesIO(raw), chunk_size=chunk_size)
        assert _read_stream_value(events) == expected



@pytest.mark.parametrize("mode", ["aicpu", "aiv"])
def test_pass3_finding1_unconfirmed_expansion_requires_explicit_fallback(
        embedded, mode):
    _, bundle = embedded
    op = _measurement_op()
    recipe = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode)
    with pytest.raises(
            ModelingError,
            match=f"a5_expansion_regime_unconfirmed:{mode}"):
        recipe.build(op, None)

    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True).build(op, None)
    fallback = bundle.curves[
        ("all_gather", 8, "intra_node_ub", "bf16", "unspecified")]
    assert plan.metadata == {
        "a5_collective": "all_gather", "world_size": 8,
        "placement_scope": "intra_node_ub", "dtype": "bf16",
        "requested_expansion_mode": mode,
        "curve_expansion_mode": "unspecified",
        "unspecified_fallback": True, "model_id": fallback.model_id,
        "stack": "rack06", "cann": "9.1.0",
        "algorithm": None, "protocol": None}
    direct = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8).build(op, None)
    assert direct.metadata["unspecified_fallback"] is False
    assert direct.metadata["requested_expansion_mode"] == "unspecified"


@pytest.mark.parametrize("mode", ["aiv", "aicpu"])
def test_pass3_finding1_recipe_preserves_mte_and_vector_work(embedded, mode):
    _, bundle = embedded
    op = _measurement_op(mte2_bytes=2048, mte3_bytes=1024,
                         vector_ops=100_000_000)
    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True).build(op, None)
    applied = plan.work.apply(op)

    assert applied.mte2_bytes == 2048
    assert applied.mte3_bytes == 1024
    stages = {stage.name: stage for stage in plan.program(_comm_context(op)).stages}
    assert stages["mte2"].work["mte2@r0:s0"] == pytest.approx(2048)
    assert stages["mte3"].work["mte3@r0:s0"] == pytest.approx(1024)
    assert applied.vector_ops == 100_000_000
    assert stages["vector"].work["vector@r0:s0"] == pytest.approx(
        100_000_000)


def test_pass3_finding2_transport_identity_tracks_communication_parameters(
        embedded):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    parsed = parse_case(raw)
    curve = bundle.bind_service_curve(parsed, operation)
    identity = service_identity(parsed, operation, "none")

    assert curve.identity == identity
    assert identity["transport"]["collective"] == "allgather"
    assert identity["transport"]["group_axis"] == "tp"
    assert identity["transport"]["steps"] == 7

    changed = replace(operation, metadata=operation.metadata + (
        ("world_size", 2), ("expansion_mode", "aiv"),
        ("p2p_pair", "0->4")))
    changed_identity = service_identity(parsed, changed, "none")
    assert changed_identity != identity
    assert changed_identity["transport"]["world_size"] == 2
    assert changed_identity["transport"]["expansion_mode"] == "aiv"
    assert changed_identity["transport"]["p2p_pair"] == "0->4"
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(
            raw, (changed,),
            recipes=RecipeBook(
                {changed.recipe: ResourceService(curve)}))


@pytest.mark.parametrize(("overrides", "key", "value"), [
    ({"world_size": 2}, "world_size", 2),
    ({"expansion_mode": "aiv", "allow_unspecified_fallback": True},
     "expansion_mode", "aiv"),
    ({"collective": "all_reduce"}, "collective", "all_reduce"),
])
def test_pass3_finding2_override_binding_cannot_execute_on_changed_operator(
        embedded, overrides, key, value):
    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    parsed = parse_case(raw)
    bound = bundle.bind_service_curve(parsed, operation, **overrides)

    assert bound.identity["transport"][key] == value
    assert bound.identity != service_identity(parsed, operation, "none")
    with pytest.raises(ModelingError, match="service_identity_mismatch"):
        operator_costs(
            raw, (operation,),
            recipes=RecipeBook({operation.recipe: ResourceService(bound)}))


def test_pass3_finding4_profiler_evidence_is_not_truncated():
    evidence = (["hcom_allReduce_AicpuKernel"] + ["not-a-mode"] * 100
                + ["AI_VECTOR_CORE"])
    resolution = resolve_expansion_regime(
        "unspecified", "not_observed", profiler_evidence=evidence)

    assert resolution["profiler_modes"] == ["aicpu", "aiv"]
    assert resolution["evidence_conflict"] is True
    assert resolution["regime"] == "unspecified"
    assert resolution["expansion_unconfirmed"] is True


@pytest.mark.parametrize("origin", ["rank", "campaign"])
def test_pass3_finding5_sample_level_single_anchors_never_train(origin):
    samples = [{"per_op_ms": 0.05, "single": False},
               {"per_op_ms": 9.0, "single": True}]
    if origin == "rank":
        document = {
            "role": "rank", "status": "completed", "label": "sample-rank",
            "world_size": 2, "dtype": "bf16", "placement": "intra-node-ub",
            "cases": [{"case_id": "sample", "collective": "all_reduce",
                       "payload_bytes": 1024,
                       "per_op_ms_summary": {"median": 0.05},
                       "samples": samples,
                       "correctness": {"expected": 1, "observed": 1}}]}
    else:
        document = _campaign(label="sample-campaign", conditions=[{
            "collective": "all_reduce", "payload_bytes": 1024,
            "median_ms": 0.05, "samples": samples}])

    ingested = ingest_a5_campaign_data(document)

    assert len(ingested["conditions"]) == 1
    assert ingested["conditions"][0]["single"] is False
    assert ingested["conditions"][0]["per_op_ms"] == pytest.approx(0.05)
    assert len(ingested["single_op_anchors"]) == 1
    assert ingested["single_op_anchors"][0]["single"] is True
    assert ingested["single_op_anchors"][0]["per_op_ms"] == pytest.approx(9.0)


def test_pass3_finding5_raw_points_keep_point_level_single_and_regime():
    raw_points = [
        {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.5,
         "single": False, "observed_expansion_mode": "aicpu",
         "boundary": "device_kernel", "run_id": "raw", "repeat_index": 0,
         "split": "train"},
        {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 9.0,
         "single": True, "observed_expansion_mode": "aiv",
         "boundary": "device_kernel", "run_id": "raw", "repeat_index": 0,
         "split": "train"},
        {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": 0.7,
         "single": False, "observed_expansion_mode": "aiv",
         "boundary": "resource_phase", "run_id": "raw", "repeat_index": 0,
         "split": "train"},
    ]
    document = _campaign(label="raw-points-single", conditions=[],
                         raw_points=raw_points)

    ingested = ingest_a5_campaign_data(document)

    assert len(ingested["conditions"]) == 2
    assert len(ingested["single_op_anchors"]) == 1
    by_cell = {(row["observed_expansion_mode"], row["boundary"]): row
               for row in ingested["conditions"]}
    assert set(by_cell) == {("aicpu", "device_kernel"),
                            ("aiv", "resource_phase")}
    assert by_cell[("aicpu", "device_kernel")]["per_op_ms"] == pytest.approx(0.5)
    assert by_cell[("aiv", "resource_phase")]["per_op_ms"] == pytest.approx(0.7)
    anchor = ingested["single_op_anchors"][0]
    assert anchor["single"] is True
    assert anchor["per_op_ms"] == pytest.approx(9.0)
    assert anchor["observed_expansion_mode"] == "aiv"
    assert anchor["boundary"] == "device_kernel"


def test_pass3_finding6_stream_filters_samples_without_raw_array():
    class Payload:
        live = 0
        peak = 0

        def __init__(self, index):
            self.index = index
            Payload.live += 1
            Payload.peak = max(Payload.peak, Payload.live)

        def __del__(self):
            Payload.live -= 1

    def events():
        yield ("start_map", None)
        yield ("map_key", "samples")
        yield ("start_array", None)
        for index in range(200):
            payload = Payload(index)
            yield ("start_map", None)
            yield ("map_key", "junk")
            yield ("string", payload)
            yield ("end_map", None)
        yield ("end_array", None)
        yield ("map_key", "label")
        yield ("string", "filtered")
        yield ("end_map", None)

    # Every iterated sample holds a live payload; a raw samples list would keep
    # all 200 alive at once even though none matches the evidence projection.
    stream = events()
    loaded = _read_stream_value(stream)

    assert loaded == {"label": "filtered"}
    assert Payload.peak < 10
    del stream
    assert Payload.live == 0


def test_pass3_finding8_condition_dedupe_keeps_measurement_boundary():
    document = _campaign(label="boundary-dedupe", conditions=[
        {"collective": "all_reduce", "payload_bytes": 1024,
         "median_ms": 0.05, "run_id": "same", "repeat_index": 0,
         "split": "train", "boundary": "device_kernel"},
        {"collective": "all_reduce", "payload_bytes": 1024,
         "median_ms": 0.05, "run_id": "same", "repeat_index": 0,
         "split": "train", "boundary": "resource_phase"},
    ])

    ingested = ingest_a5_campaign_data(document)

    assert len(ingested["conditions"]) == 2
    assert {row["boundary"] for row in ingested["conditions"]} == {
        "device_kernel", "resource_phase"}


def test_pass3_finding9_parent_aggregation_preserves_independent_trials(tmp_path):
    parent = {
        "role": "parent", "label": "trials", "status": "completed",
        "canary_ok": True, "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY, "world_size": 2, "dtype": "bf16",
        "placement": "intra-node-ub",
    }
    (tmp_path / "trials.parent.json").write_text(
        json.dumps(parent), encoding="utf-8")
    for rank, (first, second) in ((0, (0.05, 0.10)), (1, (0.07, 0.12))):
        document = {
            "role": "rank", "world_size": 2, "dtype": "bf16",
            "placement": "intra-node-ub",
            "cases": [
                {"case_id": "trial-a", "collective": "all_reduce",
                 "payload_bytes": 1024, "run_id": "trial-a",
                 "repeat_index": 0,
                 "per_op_ms_summary": {"median": first},
                 "correctness": {"expected": 1, "observed": 1}},
                {"case_id": "trial-b", "collective": "all_reduce",
                 "payload_bytes": 1024, "run_id": "trial-b",
                 "repeat_index": 1,
                 "per_op_ms_summary": {"median": second},
                 "correctness": {"expected": 1, "observed": 1}},
            ]}
        (tmp_path / f"trials.rank{rank}.json").write_text(
            json.dumps(document), encoding="utf-8")

    ingested = ingest_a5_campaign_data(tmp_path)

    assert len(ingested["conditions"]) == 2
    by_run = {row["run_id"]: row for row in ingested["conditions"]}
    assert set(by_run) == {"trial-a", "trial-b"}
    assert by_run["trial-a"]["per_op_ms"] == pytest.approx(0.06)
    assert by_run["trial-b"]["per_op_ms"] == pytest.approx(0.11)
    assert by_run["trial-a"]["repeat_index"] == 0
    assert by_run["trial-b"]["repeat_index"] == 1


def test_pass3_finding10_mc2_chains_preserve_independent_trial_identity():
    chains = [
        _mc2_chain("same", run_id="run-a", repeat_index=0,
                   chain_id="chain-a"),
        _mc2_chain("same", run_id="run-b", repeat_index=1,
                   chain_id="chain-b"),
    ]
    ingested = ingest_a5_campaign_data(_pipeline_campaign(
        "chain-trials", [], chains=chains))

    assert len(ingested["mc2_chains"]) == 2
    by_chain = {row["chain_id"]: row for row in ingested["mc2_chains"]}
    assert set(by_chain) == {"chain-a", "chain-b"}
    assert (by_chain["chain-a"]["run_id"],
            by_chain["chain-a"]["repeat_index"]) == ("run-a", 0)
    assert (by_chain["chain-b"]["run_id"],
            by_chain["chain-b"]["repeat_index"]) == ("run-b", 1)

    default = ingest_a5_campaign_data(_pipeline_campaign(
        "default-chain", [], chains=[_mc2_chain("same")]))
    row = default["mc2_chains"][0]
    assert row["run_id"] == "default-chain"
    assert row["chain_id"] == "default-chain:same"
    assert row["repeat_index"] == 0


def test_pass3_finding10_embedded_origin_uses_declared_source(monkeypatch):
    documents = ingest_module._source_documents(None)
    assert all(
        origin == (document.get("source")
                   or f"embedded:{document.get('label', 'a5')}")
        for document, origin in documents)

    campaign = deepcopy(ingest_module.EMBEDDED_A5_CAMPAIGNS["comm-rack06"])
    campaign.pop("source", None)
    campaign["label"] = "no-source"
    monkeypatch.setattr(
        ingest_module, "EMBEDDED_A5_CAMPAIGNS", {"no-source": campaign})
    monkeypatch.setattr(ingest_module, "EMBEDDED_A5_MC2_CAMPAIGNS", {})

    assert ingest_module._source_documents(None) == [
        (campaign, "embedded:no-source")]


def test_pass3_finding11_manifest_requires_both_expansion_regimes():
    aicpu = {"collective": "all_reduce", "world_size": 2,
             "placement_scope": "intra_node_ub", "dtype": "bf16",
             "payload_bytes": 1024, "median_ms": 0.05,
             "observed_expansion_mode": "aicpu", "run_id": "aicpu",
             "split": "train"}
    aiv = {**aicpu, "observed_expansion_mode": "aiv", "run_id": "aiv"}
    only_aicpu = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="only-aicpu", conditions=[aicpu])))
    both = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="both-modes", conditions=[aicpu, aiv])))

    assert "expansion_regime:aicpu/aiv" in only_aicpu["pending_cell_ids"]
    assert "expansion_regime:aicpu/aiv" not in both["pending_cell_ids"]
    assert {cell["expansion_mode"] for cell in both["verified_cells"]} == {
        "aicpu", "aiv"}


def test_pass3_finding11_manifest_requires_both_wide_all_to_allv_sizes():
    def row(world_size, duration):
        return {"collective": "all_to_allv", "world_size": world_size,
                "placement_scope": "cross_node_urma", "dtype": "bf16",
                "payload_bytes": 1024, "median_ms": duration,
                "run_id": f"w{world_size}", "split": "train"}

    only_w16 = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="w16", conditions=[row(16, 0.10)])))
    both = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="w16-w32", conditions=[
            row(16, 0.10), row(32, 0.20)])))

    assert "all_to_allv:W16-W32" in only_w16["pending_cell_ids"]
    assert "all_to_allv:W16-W32" not in both["pending_cell_ids"]
    assert {cell["world_size"] for cell in both["verified_cells"]
            if cell["collective"] == "all_to_allv"} == {16, 32}


def test_pass3_finding11_manifest_four_hop_requires_collective_scope_and_pair():
    def row(scope):
        return {"collective": "p2p_send_recv", "world_size": 8,
                "placement_scope": scope, "p2p_pair": "0->4",
                "dtype": "bf16", "payload_bytes": 1024,
                "median_ms": 0.05, "run_id": f"pair-{scope}",
                "split": "train"}

    wrong_scope = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="wrong-scope", conditions=[
            row("cross_node_urma")])))
    matched = build_a5_followup_manifest(ingest_a5_campaign_data(
        _campaign(label="matched", conditions=[row("intra_node_ub")])))

    assert ("p2p_send_recv:0->4:intra_node_ub"
            in wrong_scope["pending_cell_ids"])
    assert ("p2p_send_recv:0->4:intra_node_ub"
            not in matched["pending_cell_ids"])
    cell = next(cell for cell in matched["verified_cells"]
                if cell["collective"] == "p2p_send_recv")
    assert cell["placement_scope"] == "intra_node_ub"
    assert cell["p2p_pair"] == "0->4"


@pytest.mark.parametrize("mode", ["aicpu", "unspecified"])
def test_pass4_finding1_non_aiv_recipe_preserves_existing_vector_work(
        embedded, mode):
    _, bundle = embedded
    op = _measurement_op(vector_ops=100_000_000)
    plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True).build(op, None)
    applied = plan.work.apply(op)
    stages = {stage.name: stage
              for stage in plan.program(_comm_context(op)).stages}

    assert applied.vector_ops == pytest.approx(100_000_000)
    assert stages["vector"].work["vector@r0:s0"] == pytest.approx(
        100_000_000)

    plain = _measurement_op()
    plain_plan = bundle.build_a5_comm_recipe(
        "all_gather", world_size=8, expansion_mode=mode,
        allow_unspecified_fallback=True).build(plain, None)
    assert plain_plan.work.apply(plain).vector_ops == 0
    assert "vector" not in {
        stage.name
        for stage in plain_plan.program(_comm_context(plain)).stages}
