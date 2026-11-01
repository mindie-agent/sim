"""A5 communication calibration admits only measured, strictly keyed evidence.

[Role] Verify A5 communication ingestion, curve fitting, binding, and recipes.
[Invariants] Unconfirmed AICPU/AIV requests never promote to confirmed regimes,
  AICPU and AIV never alias, non-A5 hardware is rejected, AIV shares the vector
  pipe while AICPU does not, and launch-only anchors never train steady curves.
[Timeouts & Locks] Subprocess isolation checks use explicit 10s timeouts.
[Side-Effect Policy] Pure in-memory and tmp_path tests; never touches devices.
"""
import ast
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

ROOT = Path(__file__).resolve().parents[1]
LIVE_A5 = Path("/Volumes/data/mindie-sim-data/a5")
HOLDOUT_PAYLOADS = (65536, 16 * 1024 * 1024)
MIB = 1024 * 1024
COLLECTIVES = ("all_reduce", "reduce_scatter", "all_gather", "all_to_all",
               "all_to_allv")
WORLD_SIZES = (2, 4, 8, 16, 32)
STAGES = ("quant", "all_to_allv_dispatch", "grouped_matmul",
          "all_to_allv_combine")
INIT_DOCSTRING = '''"""[Role] Read observations independently of prediction and calibration admission.
[Invariants] Importing this package performs no simulation or device work.
[Timeouts & Locks] No waits or locks.
[Side-Effect Policy] No I/O.
"""
'''


@pytest.fixture(scope="module")
def embedded():
    """Ingest once; fitting 38 curves is read-only but not free."""
    ingested = ingest_a5_campaign_data()
    return ingested, fit_a5_service_curves(ingested)


def _campaign(label="probe", conditions=None, **overrides):
    document = {
        "label": label,
        "status": "completed",
        "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY,
        "world_size": 2,
        "dtype": "bf16",
        "placement": "intra-node-ub",
        "conditions": conditions if conditions is not None else [
            {"collective": "all_reduce", "payload_bytes": 1024,
             "median_ms": 0.05},
            {"collective": "all_reduce", "payload_bytes": 1048576,
             "median_ms": 0.06},
        ],
    }
    document.update(overrides)
    return document


def _fit_campaign(document):
    return fit_a5_service_curves(ingest_a5_campaign_data(document))


def _mode_conditions(mode, *, evidence, intercept, slope):
    """Three training payloads so the affine fit is identified."""
    return [
        {
            "collective": "all_reduce",
            "payload_bytes": payload,
            "median_ms": intercept + payload * slope,
            "requested_expansion_mode": mode,
            "observed_expansion_mode": "not_observed",
            "profiler_evidence": [evidence] if evidence else None,
            "single": False,
            "dtype": "bf16",
            "world_size": 2,
            "placement": "intra-node-ub",
            "run_id": f"{mode}-{payload}",
        }
        for payload in (1024, 1048576, 4194304)
    ]


def _scope_for(world_size):
    return "intra_node_ub" if world_size <= 8 else "cross_node_urma"


# ---------------------------------------------------------------------------
# 1. Strict expansion-regime honesty
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["aicpu", "aiv"])
def test_unconfirmed_request_is_never_promoted(mode):
    resolution = resolve_expansion_regime(mode, "not_observed")

    assert resolution["requested_expansion_mode"] == mode
    assert resolution["observed_expansion_mode"] == "not_observed"
    assert resolution["regime"] == "unspecified"
    assert resolution["expansion_unconfirmed"] is True
    assert resolution["unconfirmed_tag"] == f"requested_{mode}_unconfirmed"
    assert resolution["evidence_conflict"] is False


@pytest.mark.parametrize(("requested", "observed", "evidence", "expected"), [
    ("unspecified", "aicpu", None, "aicpu"),
    ("unspecified", "aiv", None, "aiv"),
    ("aicpu", "not_observed", "hcom_allReduce_AicpuKernel", "aicpu"),
    ("aiv", "not_observed", "AI_VECTOR_CORE", "aiv"),
    ("unspecified", "not_observed",
     {"kernel": "hcom_allReduce_AicpuKernel", "block": "aicpu"}, "aicpu"),
])
def test_observed_or_profiler_evidence_confirms_regime(
        requested, observed, evidence, expected):
    resolution = resolve_expansion_regime(
        requested, observed, profiler_evidence=evidence)

    assert resolution["regime"] == expected
    assert resolution["expansion_unconfirmed"] is False
    assert resolution["unconfirmed_tag"] is None


def test_conflicting_profiler_evidence_without_matching_request_is_unspecified():
    resolution = resolve_expansion_regime(
        "unspecified", "not_observed",
        profiler_evidence=["hcom_allReduce_AicpuKernel", "AI_VECTOR_CORE"])

    assert resolution["regime"] == "unspecified"
    assert resolution["evidence_conflict"] is True
    assert resolution["expansion_unconfirmed"] is True
    assert resolution["unconfirmed_tag"] is None
    assert resolution["profiler_modes"] == ["aicpu", "aiv"]


# ---------------------------------------------------------------------------
# 2. Campaign ingestion validation and anchor separation
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(("patch", "reason"), [
    ({"status": "failed"}, "a5_campaign_not_completed"),
    ({"state": "aborted"}, "a5_campaign_not_completed"),
    ({"status": "running"}, "a5_campaign_not_completed"),
    ({"canary_ok": False}, "a5_campaign_canary_failed"),
    ({"canary": "mismatch"}, "a5_campaign_canary_failed"),
    ({"correctness": {"expected": 36, "observed": 35}},
     "a5_campaign_canary_failed"),
])
def test_ingest_rejects_failed_or_incomplete_campaign(patch, reason):
    document = _campaign(**patch)

    with pytest.raises(ValueError, match=reason):
        ingest_a5_campaign_data(document)


def test_ingest_rejects_condition_correctness_mismatch():
    document = _campaign()
    document["conditions"][0]["correctness"] = {
        "expected": 36, "observed": 35}

    with pytest.raises(ValueError, match="a5_campaign_canary_failed"):
        ingest_a5_campaign_data(document)


def test_single_launch_anchors_never_enter_conditions_or_training_knots():
    document = _campaign(conditions=[
        {"collective": "all_reduce", "payload_bytes": 1024,
         "median_ms": 0.05},
        {"collective": "all_reduce", "payload_bytes": 1048576,
         "median_ms": 0.06},
        {"collective": "all_reduce", "payload_bytes": 1024,
         "median_ms": 0.90, "single": True},
        {"collective": "all_reduce", "payload_bytes": 2097152,
         "median_ms": 0.95, "single": True},
    ])

    ingested = ingest_a5_campaign_data(document)
    assert len(ingested["conditions"]) == 2
    assert {row["payload_bytes"] for row in ingested["conditions"]} == {
        1024, 1048576}
    assert all(row["single"] is False for row in ingested["conditions"])
    assert len(ingested["single_op_anchors"]) == 2
    assert all(row["single"] is True for row in ingested["single_op_anchors"])

    bundle = fit_a5_service_curves(ingested)
    curve = bundle.curves[("all_reduce", 2, "intra_node_ub", "bf16",
                           "unspecified")]
    knots = dict(curve.knots)
    assert knots[1024] == pytest.approx(0.05)
    assert knots[1048576] == pytest.approx(0.06)
    assert 2097152 not in knots
    assert max(knots.values()) < 0.90
    assert bundle.single_op_anchors == tuple(ingested["single_op_anchors"])


def test_parent_and_rank_json_pairs_merge_without_double_counting(tmp_path):
    parent = {
        "role": "parent",
        "label": "pair-campaign",
        "status": "completed",
        "canary_ok": True,
        "hardware_key": A5_HARDWARE_KEY,
        "arch_family": A5_ARCH_FAMILY,
        "world_size": 2,
        "dtype": "bf16",
        "placement": "intra-node-ub",
        "protocol": {"dtype": "bf16", "p2p_scope": "intra_node_ub"},
    }
    (tmp_path / "pair.parent.json").write_text(
        json.dumps(parent), encoding="utf-8")
    for rank, (small, large) in ((0, (0.05, 0.06)), (1, (0.07, 0.08))):
        rank_document = {
            "role": "rank",
            "world_size": 2,
            "dtype": "bf16",
            "placement": "intra-node-ub",
            "cases": [
                {"case_id": "all-reduce-1k", "collective": "all_reduce",
                 "payload_bytes": 1024,
                 "per_op_ms_summary": {"median": small},
                 "correctness": {"expected": 36, "observed": 36}},
                {"case_id": "all-reduce-1m", "collective": "all_reduce",
                 "payload_bytes": 1048576,
                 "per_op_ms_summary": {"median": large},
                 "correctness": {"expected": 36, "observed": 36}},
            ],
        }
        (tmp_path / f"pair.rank{rank}.json").write_text(
            json.dumps(rank_document), encoding="utf-8")

    ingested = ingest_a5_campaign_data(tmp_path)

    assert ingested["campaign_labels"] == ["pair-campaign"]
    assert len(ingested["conditions"]) == 2
    assert len(ingested["single_op_anchors"]) == 0
    by_payload = {row["payload_bytes"]: row["per_op_ms"]
                  for row in ingested["conditions"]}
    assert by_payload[1024] == pytest.approx(0.06)
    assert by_payload[1048576] == pytest.approx(0.07)
    assert all(row["source"].endswith("pair.parent.json")
               for row in ingested["conditions"])

    bundle = fit_a5_service_curves(ingested)
    assert ("all_reduce", 2, "intra_node_ub", "bf16", "unspecified") in bundle.curves


# ---------------------------------------------------------------------------
# 3. Service-curve fitting across world sizes and P2P placement
# ---------------------------------------------------------------------------

def test_fit_covers_every_collective_and_world_size(embedded):
    _, bundle = embedded

    for collective in COLLECTIVES:
        for world_size in WORLD_SIZES:
            scope = _scope_for(world_size)
            key = (collective, world_size, scope, "bf16", "unspecified")
            assert key in bundle.curves, key
            curve = bundle.lookup_curve(
                collective, world_size, placement_scope=scope, dtype="bf16")
            assert curve is bundle.curves[key]
            assert curve.work_unit == "bytes"
            assert curve.boundary == "device_kernel"
            assert curve.identity["hardware"]["key"] == A5_HARDWARE_KEY
            assert curve.identity["hardware"]["sku"] == A5_SKU
            assert curve.identity["hardware"]["architecture"] == A5_ARCH_FAMILY
            if collective == "all_to_allv":
                assert curve.provenance.get("derived_from") == "all_to_all"
    assert bundle.unfitted == ()


def test_every_curve_has_strict_startup_and_no_holdout_knots(embedded):
    _, bundle = embedded

    for key, curve in bundle.curves.items():
        minimum_duration = min(duration for _, duration in curve.knots)
        assert 0.0 <= curve.startup_ms < minimum_duration, key
        training_payloads = {payload for payload, _ in curve.knots}
        assert training_payloads.isdisjoint(HOLDOUT_PAYLOADS), key


def test_mutating_holdout_durations_does_not_change_models(embedded):
    ingested, bundle = embedded
    mutated = deepcopy(ingested)
    touched = 0
    for row in mutated["conditions"]:
        if row["payload_bytes"] in HOLDOUT_PAYLOADS:
            row["per_op_ms"] *= 2.5
            touched += 1

    assert touched > 0
    refit = fit_a5_service_curves(mutated)
    assert set(refit.curves) == set(bundle.curves)
    for key, curve in bundle.curves.items():
        assert refit.curves[key].model_id == curve.model_id, key
        assert refit.curves[key].knots == curve.knots, key


def test_cross_node_p2p_beats_single_pair_intra_node(embedded):
    _, bundle = embedded
    payload = 128 * MIB

    intra = bundle.lookup_curve(
        "p2p_send_recv", 8, placement_scope="intra_node_ub")
    cross = bundle.lookup_curve(
        "p2p_send_recv", 32, placement_scope="cross_node_urma")
    far = bundle.lookup_curve(
        "p2p_send_recv", 32, placement_scope="far")

    intra_ms = intra.estimate(payload).duration_ms
    cross_ms = cross.estimate(payload).duration_ms
    far_ms = far.estimate(payload).duration_ms

    assert intra_ms == pytest.approx(2.70, rel=0.02)
    assert cross_ms == pytest.approx(0.62, rel=0.03)
    assert far_ms == pytest.approx(0.64, rel=0.03)
    assert cross_ms < intra_ms / 3
    assert cross_ms < far_ms
    assert cross.identity["p2p_pair"] == "0->8"
    assert far.identity["p2p_pair"] == "0->24"
    assert cross.identity["transport"] == "urma"
    assert intra.identity["transport"] == "ub_mesh"


# ---------------------------------------------------------------------------
# 4. Mandatory positive / negative controls
# ---------------------------------------------------------------------------

def test_control_1_aicpu_and_aiv_curves_never_alias():
    conditions = (
        _mode_conditions("aicpu", evidence="hcom_allReduce_AicpuKernel",
                         intercept=0.05, slope=1e-9)
        + _mode_conditions("aiv", evidence="AI_VECTOR_CORE",
                           intercept=0.08, slope=2e-9)
    )
    bundle = _fit_campaign(_campaign(label="control-1", conditions=conditions))
    base = ("all_reduce", 2, "intra_node_ub", "bf16")

    aicpu = bundle.curves[base + ("aicpu",)]
    aiv = bundle.curves[base + ("aiv",)]

    assert aicpu is not aiv
    assert aicpu.model_id != aiv.model_id
    assert aicpu.knots != aiv.knots
    assert aicpu.estimate(1048576).duration_ms < aiv.estimate(
        1048576).duration_ms
    assert aicpu.provenance["expansion_regime"] == "aicpu"
    assert aiv.provenance["expansion_regime"] == "aiv"
    assert aicpu.provenance["expansion_unconfirmed"] is False
    assert aiv.provenance["expansion_unconfirmed"] is False
    assert bundle.lookup_curve(
        "all_reduce", 2, expansion_mode="aicpu").model_id == aicpu.model_id
    assert bundle.lookup_curve(
        "all_reduce", 2, expansion_mode="aiv").model_id == aiv.model_id


@pytest.mark.parametrize("overrides", [
    {"hardware_key": "Ascend910_93"},
    {"hardware_key": "Ascend910B"},
    {"arch_family": "A3"},
])
def test_control_2_ingest_and_fit_reject_non_a5_hardware(overrides):
    document = _campaign(**overrides)
    reason = "a5_calibration_rejects_non_a5_hardware"

    with pytest.raises(ModelingError, match=reason):
        ingest_a5_campaign_data(document)
    with pytest.raises(ModelingError, match=reason):
        fit_a5_service_curves(document)


@pytest.mark.parametrize("guard", [
    {"hardware_key": "Ascend910_93"},
    {"hardware_key": "Ascend910B"},
    {"arch_family": "A3"},
])
def test_control_2_lookup_rejects_non_a5_hardware(embedded, guard):
    _, bundle = embedded

    with pytest.raises(ModelingError,
                       match="a5_calibration_rejects_non_a5_hardware"):
        bundle.lookup_curve("all_reduce", 2, **guard)


def test_control_2_fit_rejects_non_a5_already_ingested_data(embedded):
    ingested, _ = embedded
    contaminated = deepcopy(ingested)
    contaminated["hardware_key"] = "Ascend910_93"

    with pytest.raises(ModelingError,
                       match="a5_calibration_rejects_non_a5_hardware"):
        fit_a5_service_curves(contaminated)


def test_control_3_aiv_shares_vector_resources_aicpu_does_not():
    baseline = compute_a5_contention_slowdown(
        "aiv", concurrent_vector_util=0.2, active_aiv_blocks=8,
        total_aiv_units=64)
    demanded = compute_a5_contention_slowdown(
        "aiv", concurrent_vector_util=0.8, active_aiv_blocks=24,
        total_aiv_units=64)

    assert demanded["vector_slowdown"] > 1.0
    by_blocks = [
        compute_a5_contention_slowdown(
            "aiv", concurrent_vector_util=0.8, active_aiv_blocks=blocks,
            total_aiv_units=64)["vector_slowdown"]
        for blocks in (8, 24, 64)
    ]
    by_demand = [
        compute_a5_contention_slowdown(
            "aiv", concurrent_vector_util=duty, active_aiv_blocks=24,
            total_aiv_units=64)["vector_slowdown"]
        for duty in (0.2, 0.5, 0.8)
    ]
    assert by_blocks == sorted(by_blocks) and len(set(by_blocks)) == 3
    assert by_demand == sorted(by_demand) and len(set(by_demand)) == 3
    assert baseline["vector_slowdown"] > 1.0

    aicpu = compute_a5_contention_slowdown(
        "aicpu", concurrent_vector_util=0.8, concurrent_mte_util=0.6)
    assert aicpu["vector_slowdown"] == 1.0
    assert aicpu["vector_alu_slowdown"] == 1.0
    assert aicpu["transfer_latency_ms"] > 0
    assert aicpu["sync_latency_ms"] > 0
    assert aicpu["mte_slowdown"] > 1.0
    assert aicpu["contention_slowdown"] > 1.0


def test_control_4_unconfirmed_rows_never_contaminate_confirmed_curves():
    confirmed = (
        _mode_conditions("aicpu", evidence="hcom_allReduce_AicpuKernel",
                         intercept=0.05, slope=1e-9)
        + _mode_conditions("aiv", evidence="AI_VECTOR_CORE",
                           intercept=0.08, slope=2e-9)
    )
    unconfirmed = (
        _mode_conditions("aicpu", evidence=None, intercept=5.0, slope=1e-8)
        + _mode_conditions("aiv", evidence=None, intercept=5.5, slope=1e-8)
    )
    base = ("all_reduce", 2, "intra_node_ub", "bf16")

    confirmed_bundle = _fit_campaign(
        _campaign(label="control-4", conditions=confirmed))
    mixed_bundle = _fit_campaign(
        _campaign(label="control-4", conditions=confirmed + unconfirmed))
    unconfirmed_bundle = _fit_campaign(
        _campaign(label="control-4-unconfirmed", conditions=unconfirmed))

    for mode in ("aicpu", "aiv"):
        key = base + (mode,)
        assert mixed_bundle.curves[key].model_id == confirmed_bundle.curves[
            key].model_id
        assert mixed_bundle.curves[key].knots == confirmed_bundle.curves[
            key].knots
        assert mixed_bundle.curves[key].provenance[
            "expansion_unconfirmed"] is False
        assert mixed_bundle.lookup_curve(
            "all_reduce", 2, expansion_mode=mode).model_id == (
                confirmed_bundle.curves[key].model_id)

    unspecified_key = base + ("unspecified",)
    assert unspecified_key in mixed_bundle.curves
    assert mixed_bundle.curves[unspecified_key].provenance[
        "expansion_unconfirmed"] is True
    assert mixed_bundle.curves[unspecified_key].model_id != (
        confirmed_bundle.curves[base + ("aicpu",)].model_id)

    assert unconfirmed_bundle.cells == (unspecified_key,)
    for mode in ("aicpu", "aiv"):
        with pytest.raises(
                ModelingError,
                match=f"a5_expansion_regime_unconfirmed:{mode}"):
            unconfirmed_bundle.lookup_curve(
                "all_reduce", 2, expansion_mode=mode,
                allow_unspecified_fallback=False)
    fallback = unconfirmed_bundle.lookup_curve(
        "all_reduce", 2, expansion_mode="aicpu",
        allow_unspecified_fallback=True)
    assert fallback is unconfirmed_bundle.curves[unspecified_key]


# ---------------------------------------------------------------------------
# 5. MC2 overlap decomposition and deterministic follow-up manifest
# ---------------------------------------------------------------------------

def test_mc2_quantized_overlap_and_stage_decomposition(embedded):
    ingested, _ = embedded
    summary = evaluate_a5_mc2_chains(ingested)

    assert summary["schema"] == "a5-mc2-chain-evaluation/1"
    assert summary["dtypes"] == ["bf16", "fp8-e4m3", "int8"]
    overlap = summary["overlap_ratio"]
    assert 0.63 <= overlap["min"] <= 0.69
    assert 0.94 <= overlap["max"] <= 0.97
    assert overlap["median"] == pytest.approx(0.78, abs=0.01)
    assert summary["by_dtype"]["int8"]["count"] == 24
    assert summary["by_dtype"]["fp8-e4m3"]["count"] == 24
    assert summary["by_dtype"]["bf16"]["count"] == 16

    quantized = [row for row in summary["rows"]
                 if row["dtype"] in {"int8", "fp8-e4m3"}]
    assert len(quantized) == 48
    for row in summary["rows"]:
        legs = row["legs_ms"]
        stages = row["stage_legs_ms"]
        assert row["stages"] == STAGES
        assert tuple(stages) == STAGES
        assert sum(stages.values()) == pytest.approx(row["sum_leg_p50_ms"])
        assert stages["quant"] == pytest.approx(
            legs.get("swiglu_quant", legs.get("swiglu")))
        assert stages["all_to_allv_dispatch"] == pytest.approx(legs["dispatch"])
        assert stages["grouped_matmul"] == pytest.approx(
            legs["gate_up"] + legs["down"])
        assert stages["all_to_allv_combine"] == pytest.approx(legs["combine"])
        assert row["overlap_ratio"] == pytest.approx(
            row["chain_p50_ms"] / row["sum_leg_p50_ms"])
    for row in quantized:
        assert row["contention_degradation_multiplier"] > 1.0
        assert row["expansion_regime"] == "unspecified"
        assert row["vector_slowdown"] == 1.0


def test_followup_manifest_is_deterministic_and_separates_cells(embedded):
    ingested, _ = embedded
    first = build_a5_followup_manifest(ingested)
    second = build_a5_followup_manifest(ingested)

    assert first == second
    assert first["schema"] == "a5-followup-manifest/1"
    assert first["verified_cells"]
    assert first["pending_cells"]
    assert len(first["verified_cell_ids"]) == len(
        set(first["verified_cell_ids"])) == len(first["verified_cells"])
    assert set(first["verified_cell_ids"]).isdisjoint(
        first["pending_cell_ids"])
    assert all(cell["evidence"] == "measured"
               for cell in first["verified_cells"])
    assert all(cell["payload_count"] >= 2
               for cell in first["verified_cells"])
    assert set(first["pending_cell_ids"]) == {
        "expansion_regime:aicpu/aiv",
        "collective:W64-8node",
        "p2p_send_recv:0->4:intra_node_ub",
        "contention:live_sweeps",
        "all_to_allv:W16-W32",
    }
    assert first["summary"]["verified_cells"] == len(first["verified_cells"])
    assert first["summary"]["pending_cells"] == len(first["pending_cells"])
    assert first["summary"]["unconfirmed_conditions"] == sum(
        1 for row in ingested["conditions"] if row["expansion_unconfirmed"])


def test_manifest_verified_cells_ignore_single_launch_anchors():
    document = _campaign(
        label="manifest-anchors",
        profiler_evidence=["hcom_allReduce_AicpuKernel"],
        conditions=[
            {"collective": "all_reduce", "payload_bytes": 1024,
             "median_ms": 0.05},
            {"collective": "all_reduce", "payload_bytes": 1048576,
             "median_ms": 0.06},
            {"collective": "all_reduce", "payload_bytes": 2097152,
             "median_ms": 0.95, "single": True},
        ])
    manifest = build_a5_followup_manifest(ingest_a5_campaign_data(document))

    assert len(manifest["verified_cells"]) == 1
    cell = manifest["verified_cells"][0]
    assert cell["campaigns"] == ["manifest-anchors"]
    assert cell["payload_count"] == 2
    assert cell["expansion_unconfirmed"] is False


# ---------------------------------------------------------------------------
# 6. Live mount consistency with the embedded golden bundle
# ---------------------------------------------------------------------------

def test_live_a5_campaign_matches_embedded_golden(embedded):
    if not LIVE_A5.is_dir():
        pytest.skip(f"live A5 campaign mount not available: {LIVE_A5}")

    ingested, bundle = embedded
    live_ingested = ingest_a5_campaign_data(LIVE_A5)
    live_bundle = fit_a5_service_curves(live_ingested)

    assert len(live_ingested["conditions"]) == len(ingested["conditions"])
    assert len(live_ingested["single_op_anchors"]) == len(
        ingested["single_op_anchors"])
    assert len(live_ingested["mc2_chains"]) == len(ingested["mc2_chains"])
    assert set(live_bundle.curves) == set(bundle.curves)
    assert live_bundle.unfitted == ()
    for key, expected in bundle.stack_curves.items():
        if key in live_bundle.stack_curves:
            assert live_bundle.stack_curves[key].identity["cann"] == expected.identity["cann"]
    for key, expected in bundle.curves.items():
        actual = live_bundle.curves[key]
        assert [payload for payload, _ in actual.knots] == [
            payload for payload, _ in expected.knots], key
        for (_, actual_ms), (_, expected_ms) in zip(
                actual.knots, expected.knots):
            assert actual_ms == pytest.approx(expected_ms, rel=1e-4), key
        assert actual.startup_ms == pytest.approx(
            expected.startup_ms, abs=1e-5), key

    expected_summary = evaluate_a5_mc2_chains(ingested)["overlap_ratio"]
    live_summary = evaluate_a5_mc2_chains(live_ingested)["overlap_ratio"]
    for statistic in ("min", "max", "median", "mean"):
        assert live_summary[statistic] == pytest.approx(
            expected_summary[statistic], rel=1e-4), statistic


# ---------------------------------------------------------------------------
# 7. Package init stays side-effect-free; explicit modules own the re-exports
# ---------------------------------------------------------------------------

def test_observations_package_init_has_no_eager_imports_or_exports():
    source = (ROOT / "mindie_sim/observations/__init__.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    assert len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr)


def test_observations_package_import_does_not_load_a5_calibration():
    code = (
        "import sys\n"
        "import mindie_sim.observations\n"
        "print('mindie_sim.observations.a5_comm_calibration' "
        "in sys.modules)\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], cwd=str(ROOT),
        capture_output=True, text=True, check=True, timeout=10)
    assert completed.stdout.strip() == "False"


def test_a5_public_names_are_reexported_by_the_explicit_modules():
    from mindie_sim.observations import a5_comm_calibration as a5
    from mindie_sim.observations import calibration

    names = {
        "A5_ARCH_FAMILY", "A5_HARDWARE_KEY", "A5_SKU",
        "A5CommCalibrationBundle", "EMBEDDED_A5_CAMPAIGNS",
        "EMBEDDED_A5_MC2_CAMPAIGNS", "bind_service_curve", "build_a5_comm_recipe",
        "build_a5_followup_manifest", "compute_a5_contention_slowdown",
        "evaluate_a5_mc2_chains", "fit_a5_service_curves",
        "ingest_a5_campaign_data", "resolve_expansion_regime", "to_service_regime",
    }
    assert names <= set(a5.__all__)
    for name in names:
        assert getattr(calibration, name) is getattr(a5, name)




A5_SOURCE_CASE = (
    ROOT / "examples/cases/high-concurrency-20261030/kimi-source-complete/"
    "a5_kimi_tp8_e16_bf16_c32_p4k_source_complete.json")


def _a5_case():
    raw = json.loads(A5_SOURCE_CASE.read_text(encoding="utf-8"))
    raw["model"]["layers"] = 2
    return raw


def _a5_comm_op(raw):
    ops = target_graphs(raw, scope="body", phase="decode", tokens=1, context=4104,
                        sequences=32, apply_body_defaults=True)[0]
    return replace(next(op for op in ops if op.op_id == "s0.l0.attention.sp_gather"),
                   deps=())


def _pipeline_campaign(label, rows, **overrides):
    return _campaign(label=label, conditions=rows, **overrides)


def _mc2_chain(condition, dtype="int8", **changes):
    chain = {"condition": condition, "status": "ok", "dtype": dtype, "ranks": 2,
             "experts": 64, "global_tokens": 512,
             "legs_ms": {"dispatch": 1.0, "gate_up": 1.0, "swiglu_quant": 1.0,
                         "down": 1.0, "combine": 1.0},
             "chain_ms": 4.0}
    chain.update(changes)
    return chain


def _comm_context(op, **overrides):
    values = dict(
        op=op, work={"link:port0": float(op.link_bytes)},
        resource_names={"hbm": "memory@r0:s0", "vector": "vector@r0:s0",
                        "mte2": "mte2@r0:s0", "mte3": "mte3@r0:s0"},
        rates={"memory@r0:s0": 1e6, "vector@r0:s0": 1e6,
               "mte2@r0:s0": 1e6, "mte3@r0:s0": 1e6}, core_count=64,
        resource_units={"hbm": 64, "vector": 64, "mte2": 64, "mte3": 64},
        service_identity={"hardware": {"key": A5_HARDWARE_KEY,
                                       "architecture": A5_ARCH_FAMILY}})
    values.update(overrides)
    return RecipeContext(**values)


def _measurement_op(**changes):
    values = dict(
        op_id="comm", family="allgather", phase="prefill", stage=0,
        layer=0, deps=(), requests=(1,), semantic_shape=(1,),
        local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        link_bytes=401408, hbm_bytes=4096, dtype="bf16", recipe="a5comm")
    values.update(changes)
    return OpSpec(**values)


def _raw_point(ms, evidence=None, requested=None, observed=None):
    point = {"dtype": "int8", "ep": 2, "link_bytes": 1024, "ms": ms,
             "single": False, "boundary": "device_kernel", "run_id": "raw",
             "repeat_index": 0, "split": "train"}
    if evidence is not None:
        point["profiler_evidence"] = evidence
    if requested is not None:
        point["requested_expansion_mode"] = requested
    if observed is not None:
        point["observed_expansion_mode"] = observed
    return point
