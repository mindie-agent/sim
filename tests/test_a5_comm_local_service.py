"""A5 communication local service: T06 formula, typed domain and evidence gaps.

[Role] Verify the one identifiable A5 native local service and the collective
  envelope provenance.  The service is the observed AICPU child phase, not the
  COMMUNICATION envelope and not a per-hop link price.
[Invariants] Work is per-rank input bytes; direction and shared domain are
  declared; supply changes cannot cross the measured SKU/stack/topology/source;
  the whole-collective curve is an envelope with link transit explicitly
  unidentified; raw captures are read-only.
[Timeouts & Locks] In-memory and read-only file tests only; no device access.
[Side-Effect Policy] Never launches a device, writes a profile or mutates raw
  evidence.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim.catalog.ops import comm_op
from mindie_sim.observations.a5_comm_calibration import (
    A5_COMM_LOCAL_SERVICE_SCHEMA, A5_HARDWARE_KEY, A5_SKU,
    native_child_stage_evidence,
    a5_local_service_applicability_key, a5_local_service_domain_key,
    a5_local_service_lineage_key, a5_local_service_prediction,
    a5_local_service_supply_change, collect_native_service,
    fit_a5_native_local_service, fit_a5_service_curves, ingest_a5_campaign_data,
    require_a5_local_service_domain,
)
from mindie_sim.observations.a5_comm_binding import (
    _stream_json, build_normalized_service_recipe,
    build_normalized_service_stage, normalized_service_duration_ms,
    service_curve_from_normalized,
)
from mindie_sim.observations.a5_comm_golden import (
    A5_ARCH_FAMILY, A5_NATIVE_GROUP2,
)
from mindie_sim.observations.calibration import (
    load_resource_profile, service_conditions_match,
)
from mindie_sim.observations.resource_services import (
    RESOURCE_SERVICE_PROFILE_SCHEMA, normalize_resource_service,
)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.program import RecipeBook, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.services import service_identity
from mindie_sim.observations.resource_services import RESOURCE_SERVICE_SCHEMA
from mindie_sim.schema.common import ModelingError
from test_a5_comm_calibration import embedded  # noqa: F401  pytest fixture

SOURCE = Path(A5_NATIVE_GROUP2["source"])
ENVIRONMENT = SOURCE.parents[1] / "environment.txt"
QWEN_CASE = (Path(__file__).resolve().parents[2]
             / "profiler/examples/structure/cases/a5-qwen38-tp2-b4-caller-fused.json")
IMAGE = (
    next(
        token.split("=", 1)[1]
        for line in ENVIRONMENT.read_text().splitlines()
        if line.startswith("image_id=")
        for token in line.split()
        if token.startswith("image_id=")
    )
    if ENVIRONMENT.is_file() else None
)
COLLECTIVES = ("all_gather", "reduce_scatter")


def _require_raw_source() -> None:
    if not SOURCE.is_file() or IMAGE is None:
        pytest.skip(f"A5 native campaign raw mount unavailable: {SOURCE}")


def _case_op():
    if not QWEN_CASE.is_file():
        pytest.skip(f"sibling profiler Qwen structure case unavailable: {QWEN_CASE}")
    case = parse_case(json.loads(QWEN_CASE.read_text()))
    local_rows = (1024 + case.runtime.parallel.tp - 1) // case.runtime.parallel.tp
    logical = local_rows * int(case.model.hidden_size) * 2
    op = replace(comm_op(
        op_id="ag", deps=(), stage=0, layer=0, requests=(),
        family="allgather", phase="prefill", recipe="ring_allgather",
        link_bytes=logical, shape=(1024, int(case.model.hidden_size)),
        metadata=(("collective", "allgather"), ("group_axis", "tp"),
                  ("payload_dtype", "bf16"))), dtype="bf16")
    return case, op


@pytest.fixture(scope="module")
def local_services():
    _require_raw_source()
    return {
        collective: fit_a5_native_local_service(
            collective=collective, stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE, group_size=2)
        for collective in COLLECTIVES
    }


def test_local_service_is_one_t06_formula_that_predicts_new_work(local_services):
    for collective, record in local_services.items():
        service = record["service"]
        assert record["schema"] == A5_COMM_LOCAL_SERVICE_SCHEMA
        assert record["normalized_schema"] == RESOURCE_SERVICE_SCHEMA
        assert service["kind"] == "resource_rate"
        assert service["scope"] == "independent_resource"
        assert service["boundary"] == "resource_phase"
        assert service["work_unit"] == "bytes"
        assert service["rate_scope"] == "per_owner"
        assert service["active_units"] == 2
        assert service["effective_rate"] > 0
        assert service["fixed_us"] >= 0
        assert service["identification"]["resource_identified"] is False
        assert service["identification"]["status"] == "declared_effective_service"
        assert record["formula"].startswith("T_us =")

        assert len(record["fit"]["training_work"]) >= 2
        assert set(record["fit"]["training_work"]).isdisjoint(
            record["holdout_payloads"])
        assert record["heldout"]
        assert all(row["basis"] == "formula" for row in record["heldout"])

        unseen = 200_000_000
        prediction = a5_local_service_prediction(
            record, unseen, work_unit="bytes")
        assert prediction["basis"] == "formula"
        assert prediction["extrapolated"] is True
        assert prediction["duration_us"] == pytest.approx(
            service["fixed_us"] + unseen * 1e6 / service["effective_rate"])
        assert prediction["duration_us"] > service["fixed_us"]
        with pytest.raises(ValueError, match="a5_local_service_work_unit_mismatch"):
            a5_local_service_prediction(record, unseen, work_unit="bytes_total")


def test_local_service_declares_units_direction_share_and_source(local_services):
    for collective, record in local_services.items():
        domain = record["domain"]
        conditions = domain
        assert conditions["hardware"] == A5_HARDWARE_KEY
        assert conditions["platform"]["family"] == A5_ARCH_FAMILY
        assert conditions["platform"]["variant"] == "DT"
        assert conditions["platform"]["sku"] == A5_SKU
        assert conditions["platform"]["key"] == A5_HARDWARE_KEY
        assert conditions["topology"]["kind"] == "ub_mesh"
        assert conditions["topology"]["scope"] == "intra_node"
        assert conditions["topology"]["group_size"] == 2
        assert conditions["topology"]["host"] == "rack08os2"
        assert conditions["software"]["stack"] == "rack08"
        assert conditions["software"]["cann"] == "9.2.0"
        assert conditions["software"]["execution_core"] == "aicpu"
        assert conditions["software"]["torch"] == "2.10.0+cpu"
        assert conditions["software"]["torch_npu"] == "2.10.0.post7"
        assert conditions["software"]["hccl_algorithm"] == "unknown"
        assert conditions["software"]["hccl_protocol"] is None
        assert conditions["units"] == {
            "work": "bytes_per_rank_input", "duration": "ms",
            "rate": "bytes_per_second", "rate_scope": "per_owner",
            "active_units": 2, "dtype": "bf16", "service_work_unit": "bytes",
        }
        assert conditions["direction"] == (
            "read" if collective == "all_gather" else "send")
        assert conditions["shared_domain"]["kind"] == "ub_mesh"
        assert conditions["shared_domain"]["scope"] == "intra_node"
        assert conditions["shared_domain"]["members"] == 2
        assert conditions["shared_domain"]["host"] == "rack08os2"
        assert conditions["shared_domain"]["id"] == "ub_mesh:rack08os2:W2"
        window = conditions["transport_window"]
        assert window["kind"] == "native_single_target_call"
        assert window["phase"] == "aicpu_child"
        assert window["warmup_calls"] == 3
        assert window["admitted_repeats"] == 5
        assert window["profiled_ranks"] == [0, 1]
        assert window["concurrent_collectives"] is False
        measurement = domain["measurement_boundary"]
        assert measurement["phase"] == "aicpu_child"
        assert measurement["complete_collective_call"] is False
        assert measurement["link_transit_identified"] is False
        assert measurement["direction_observed"] is False

        source = domain["source"]
        assert source["path"] == str(SOURCE)
        assert source["manifest_sha256"] == A5_NATIVE_GROUP2["manifest_sha256"]
        assert source["environment_image_id"] == IMAGE
        assert source["host"] == "rack08os2"
        assert source["versions_sha256"]
        assert source["topology_sha256"]
        assert source["topology_interpretation"].lower().find("intrahost") >= 0
        assert record["applicability_key"]["software.torch_npu"] == "2.10.0.post7"
        assert record["lineage_key"]["source.manifest_sha256"] == (
            A5_NATIVE_GROUP2["manifest_sha256"])
        gaps = record["evidence_gaps"]
        assert any("link_transit" in gap for gap in gaps)
        assert any("per_hop" in gap for gap in gaps)
        assert any("cross_node" in gap for gap in gaps)


def test_supply_change_never_crosses_the_measured_domain(local_services):
    record = local_services["all_gather"]
    changed = a5_local_service_supply_change(
        record, 20_000_000,
        effective_rate=record["service"]["effective_rate"] * 1.1,
        domain=record["domain"])
    assert changed["domain_checked"] is True
    assert changed["fixed_us"] == pytest.approx(record["service"]["fixed_us"])
    assert changed["basis"] == "explicit_effective_rate_intervention"
    assert changed["duration_us"] < a5_local_service_prediction(
        record, 20_000_000)["duration_us"]
    with pytest.raises(ValueError, match="a5_local_service_work_unit_mismatch"):
        a5_local_service_supply_change(
            record, 20_000_000,
            effective_rate=record["service"]["effective_rate"],
            domain=record["domain"], work_unit="rows")

    reduce_scatter = local_services["reduce_scatter"]
    accepted = a5_local_service_supply_change(
        reduce_scatter, 20_000_000,
        effective_rate=reduce_scatter["service"]["effective_rate"],
        domain=reduce_scatter["domain"])
    assert accepted["domain_checked"] is True

    def rejected_condition(path, value, match="a5_local_service_domain_mismatch"):
        domain = deepcopy(record["domain"])
        target = domain
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ModelingError, match=match):
            a5_local_service_supply_change(
                record, 1024, effective_rate=1.0, domain=domain)

    def rejected_lineage(path, value):
        domain = deepcopy(record["domain"])
        target = domain["source"]
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ModelingError, match="lineage"):
            a5_local_service_supply_change(
                record, 1024, effective_rate=1.0, domain=domain)

    rejected_condition(("software", "stack"), "rack06")
    rejected_condition(("software", "cann"), "9.1.0")
    rejected_condition(("software", "torch_npu"), "2.11.0",
                       "software.torch_npu")
    rejected_condition(("software", "torch"), "2.11.0", "software.torch")
    rejected_condition(("software", "hccl_algorithm"), "ring",
                       "software.hccl_algorithm")
    rejected_condition(("software", "hccl_protocol"), "mesh",
                       "software.hccl_protocol")
    rejected_condition(("topology", "scope"), "cross_node_urma")
    rejected_condition(("topology", "group_size"), 8)
    rejected_condition(("shared_domain", "id"), "ub_mesh:rack08os2:W8")
    rejected_condition(("direction",), "send")
    rejected_lineage(("environment_image_id",), "sha256:" + "0" * 64)
    rejected_lineage(("summary_sha256",), "0" * 64)
    rejected_lineage(("versions_sha256",), "0" * 64)

    flat = a5_local_service_domain_key(record["domain"])
    applicability = a5_local_service_applicability_key(record["domain"])
    lineage = a5_local_service_lineage_key(record["domain"])
    assert flat == {**applicability, **lineage}
    assert flat["units.work"] == "bytes_per_rank_input"
    assert flat["direction"] == "read"
    assert flat["shared_domain.id"] == "ub_mesh:rack08os2:W2"
    assert flat["software.torch_npu"] == "2.10.0.post7"
    assert flat["hardware"] == A5_HARDWARE_KEY
    assert flat["platform.sku"] == A5_SKU
    assert flat["software.hccl_algorithm"] == "unknown"
    assert "source.manifest_sha256" in lineage
    assert "source.manifest_sha256" not in applicability
    missing_direction = {key: value for key, value in flat.items()
                         if key != "direction"}
    with pytest.raises(ModelingError, match="a5_local_service_domain_mismatch"):
        require_a5_local_service_domain(record, missing_direction)

    # Nested and flat representations must execute the same rule.
    changed_flat = dict(flat)
    changed_flat["software.torch_npu"] = "2.11.0"
    with pytest.raises(ModelingError, match="software.torch_npu"):
        a5_local_service_supply_change(
            record, 1024, effective_rate=1.0, domain=changed_flat)
    changed_flat_lineage = dict(flat)
    changed_flat_lineage["source.manifest_sha256"] = "0" * 64
    with pytest.raises(ModelingError, match="lineage"):
        a5_local_service_supply_change(
            record, 1024, effective_rate=1.0, domain=changed_flat_lineage)


def test_hardware_presence_and_identity_conflict_match_t06(local_services):
    record = local_services["all_gather"]
    matching = service_conditions_match(
        record["service"], record["domain"])
    assert matching["applies"] is True

    # hardware is the T06 public key; platform carries the structured identity.
    both = deepcopy(record["domain"])
    both["hardware"] = "Ascend950PR_9579"
    with pytest.raises(ValueError,
                       match="a5_local_service_hardware_identity_conflict"):
        a5_local_service_supply_change(
            record, 20_000_000,
            effective_rate=record["service"]["effective_rate"],
            domain=both)
    assert service_conditions_match(record["service"], both)["applies"] is False

    platform_only = deepcopy(record["domain"])
    platform_only["platform"]["sku"] = "Ascend950PR_9579"
    platform_only["platform"]["key"] = "Ascend950PR_9579"
    with pytest.raises(ValueError,
                       match="a5_local_service_hardware_identity_conflict"):
        a5_local_service_supply_change(
            record, 20_000_000,
            effective_rate=record["service"]["effective_rate"],
            domain=platform_only)
    assert service_conditions_match(
        record["service"], platform_only)["applies"] is False

    hardware_only = deepcopy(record["domain"])
    hardware_only["hardware"] = "Ascend950PR_9579"
    del hardware_only["platform"]
    with pytest.raises(ModelingError, match="applicability:.*hardware"):
        a5_local_service_supply_change(
            record, 20_000_000,
            effective_rate=record["service"]["effective_rate"],
            domain=hardware_only)
    assert service_conditions_match(
        record["service"], hardware_only)["applies"] is False

    # A declared null must be present to match; deleting it is not "same".
    missing_null = deepcopy(record["domain"])
    del missing_null["software"]["hccl_protocol"]
    with pytest.raises(ModelingError, match="software.hccl_protocol"):
        a5_local_service_supply_change(
            record, 20_000_000,
            effective_rate=record["service"]["effective_rate"],
            domain=missing_null)
    t06 = service_conditions_match(record["service"], missing_null)
    assert t06["applies"] is False
    assert any(check["path"] == "software.hccl_protocol"
               and check["present"] is False for check in t06["checks"])

    # Nested and flat keys retain the same hardware/presence result.
    flat = a5_local_service_applicability_key(record["domain"])
    assert flat["hardware"] == A5_HARDWARE_KEY
    assert flat == record["applicability_key"]
    flat_missing = {key: value for key, value in flat.items()
                    if key != "software.hccl_protocol"}
    with pytest.raises(ModelingError, match="software.hccl_protocol"):
        require_a5_local_service_domain(record, flat_missing)


def test_local_service_rejects_environment_unit_and_selector_counterexamples():
    _require_raw_source()
    with pytest.raises(ValueError, match="native_service_environment_stack_mismatch"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack06", cann="9.2.0",
            deployment_image_id=IMAGE)
    with pytest.raises(ValueError, match="native_service_cann_mismatch"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.1.0",
            deployment_image_id=IMAGE)
    with pytest.raises(ValueError, match="native_service_environment_image_mismatch"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.2.0",
            deployment_image_id="sha256:" + "0" * 64)
    with pytest.raises(ValueError, match="native_service_requires_aicpu_active_child"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE, execution_core="aiv")
    with pytest.raises(ValueError, match="native_service_requires_intra_node_ub"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE, placement="cross_node_urma")
    with pytest.raises(ValueError, match="native_local_service_collective_unsupported"):
        fit_a5_native_local_service(
            collective="all_reduce", stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE)
    with pytest.raises(ValueError, match="native_local_service_requires_bf16"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE, dtype="fp8-e4m3")
    with pytest.raises(ValueError, match="native_service_no_compatible_conditions"):
        fit_a5_native_local_service(
            collective="all_gather", stack="rack08", cann="9.2.0",
            deployment_image_id=IMAGE, group_size=3)


def test_exported_service_is_t06_profile_loadable(local_services):
    for record in local_services.values():
        profile = {
            "schema": RESOURCE_SERVICE_PROFILE_SCHEMA,
            "hardware": A5_HARDWARE_KEY,
            "services": [record["service"]],
        }
        loaded = load_resource_profile(profile, hardware=A5_HARDWARE_KEY)
        assert loaded["service_ids"] == [record["service"]["id"]]
        assert service_conditions_match(
            loaded["services"][0], record["domain"])["applies"] is True
        with pytest.raises(ValueError, match="hardware_mismatch"):
            load_resource_profile(profile, hardware="Ascend950PR_9579")


def test_normalized_service_conditions_use_the_same_applicability(local_services):
    for record in local_services.values():
        service = record["service"]
        context = deepcopy(record["domain"])
        matching = service_conditions_match(service, context)
        assert matching["applies"] is True
        changed = deepcopy(context)
        changed["software"]["torch_npu"] = "2.11.0"
        failed = service_conditions_match(service, changed)
        assert failed["applies"] is False
        assert any(check["path"] == "software.torch_npu"
                   and not check["matched"] for check in failed["checks"])


def test_lineage_is_separate_from_target_applicability(local_services):
    record = local_services["all_gather"]
    changed = deepcopy(record["domain"])
    changed["source"]["manifest_sha256"] = "0" * 64
    # Prediction is a target-applicability question; T06 ignores lineage too.
    prediction = a5_local_service_prediction(
        record, 1_048_576, domain=changed, work_unit="bytes")
    assert prediction["basis"] == "formula"
    assert service_conditions_match(record["service"], changed)["applies"] is True
    # A supply respecification also binds the calibration lineage.
    with pytest.raises(ModelingError, match="lineage"):
        a5_local_service_supply_change(
            record, 1_048_576,
            effective_rate=record["service"]["effective_rate"],
            domain=changed)


def test_native_curve_keeps_aicpu_child_window_semantics():
    _require_raw_source()
    bundle = collect_native_service(
        str(SOURCE), stack="rack08", cann="9.2.0", execution_core="aicpu",
        placement="intra_node_ub", deployment_image_id=IMAGE, groups=(2,))
    curve = bundle.lookup_curve(
        "all_gather", 2, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    assert curve.resource == "aicpu_child"
    assert curve.provenance["service_semantics"] == "aicpu_child_collective_phase"
    measurement = curve.provenance["measurement_boundary"]
    assert measurement["phase"] == "aicpu_child"
    assert measurement["resource"] == "aicpu_child"
    assert measurement["work_coordinate"] == "per_rank_input_bytes"
    assert measurement["complete_collective_call"] is False
    assert measurement["complete_directed_call"] is False
    assert curve.identity["measurement_boundary"] == measurement
    assert curve.provenance["transport_window"]["kind"] == "native_single_target_call"
    assert curve.provenance["transport_window"]["phase"] == "aicpu_child"
    assert curve.provenance["link_transit_identified"] is False
    assert curve.provenance["hop_scaling"] == "not_derived"
    assert curve.provenance["direction_observed"] is False
    assert curve.provenance["algorithm_identified"] is False
    assert curve.identity["direction"] == "read"
    assert curve.identity["shared_domain"]["members"] == 2


def _native_child_measurement():
    return {
        "kind": "native_target_call", "phase": "aicpu_child",
        "resource": "aicpu_child",
        "work_coordinate": "per_rank_input_bytes", "duration_unit": "ms",
        "source_kind": "native_ai_cpu_child",
        "service_semantics": "aicpu_child_collective_phase",
        "complete_collective_call": False, "complete_directed_call": False,
        "link_transit_identified": False, "hop_scaling": "not_derived",
        "algorithm_identified": False, "direction_observed": False,
        "explicit": True,
    }


def _native_child_window():
    return {
        "kind": "native_single_target_call", "phase": "aicpu_child",
        "warmup_calls": 3, "admitted_repeats": 5,
        "profiled_ranks": [0, 1], "clock_domain": "device",
        "concurrent_collectives": False,
    }


def _sample_campaign(*, repeats=1):
    from test_a5_comm_calibration import _campaign

    measurement, window = _native_child_measurement(), _native_child_window()
    conditions = []
    for work, duration in ((32768, 0.041113), (10485760, 0.0593185)):
        samples = [{
            "per_op_ms": duration,
            "repeat_index": index,
            "measurement": measurement,
            "transport_window": window,
        } for index in range(repeats)]
        conditions.append({
            "collective": "all_gather", "payload_bytes": work,
            "samples": samples,
        })
    return _campaign(label="sample-child", world_size=2, dtype="bf16",
                     observed_expansion_mode="aicpu", conditions=conditions)


def _assert_sample_child_bundle(bundle):
    curve = bundle.curves[
        ("all_gather", 2, "intra_node_ub", "bf16", "aicpu")]
    assert curve.resource == "aicpu_child"
    assert curve.provenance["measurement_boundary"]["phase"] == "aicpu_child"
    assert curve.provenance["measurement_boundary"]["complete_collective_call"] is False
    assert curve.provenance["transport_window"]["kind"] == "native_single_target_call"
    assert curve.knots == ((32768, 0.041113), (10485760, 0.0593185))

    case, operation = _case_op()
    selectors = dict(expansion_mode="aicpu",
                     placement_scope="intra_node_ub")
    with pytest.raises(ModelingError,
                       match="cannot_serve_complete_collective_call"):
        bundle.to_service_regime(
            case, operation, require_complete_call=True, **selectors)
    rule = bundle.to_service_regime(case, operation, **selectors)
    assert rule["provenance"]["complete_collective_call_response"] is False
    assert rule["provenance"]["diagnostic_only"] is True
    assert rule["provenance"]["production_eligible"] is False


def test_sample_expansion_preserves_child_measurement_and_window():
    document = _sample_campaign()
    ingested = ingest_a5_campaign_data(document)
    for row in ingested["conditions"]:
        assert row["measurement"]["phase"] == "aicpu_child"
        assert row["transport_window"]["phase"] == "aicpu_child"
    _assert_sample_child_bundle(fit_a5_service_curves(ingested))


def test_stream_and_cross_256_preserve_child_measurement(tmp_path):
    document = _sample_campaign(repeats=300)
    path = tmp_path / "sample-child.json"
    path.write_text(json.dumps(document))
    streamed = _stream_json(path)
    for rows in (ingest_a5_campaign_data(document)["conditions"],
                 ingest_a5_campaign_data(streamed)["conditions"]):
        assert rows
        assert all(row["measurement"]["phase"] == "aicpu_child" for row in rows)
        assert all(row["transport_window"]["phase"] == "aicpu_child"
                   for row in rows)
    _assert_sample_child_bundle(
        fit_a5_service_curves(ingest_a5_campaign_data(streamed)))


def test_parent_host_window_cannot_replace_child_local_window():
    document = _sample_campaign()
    host_window = {
        "kind": "campaign_steady_state", "phase": "per_op",
        "warmup_calls": None, "admitted_repeats": None,
        "clock_domain": "host", "concurrent_collectives": True,
    }
    for condition in document["conditions"]:
        condition["samples"][0]["transport_window"] = host_window
    with pytest.raises(ValueError,
                       match="a5_measurement_transport_window_mismatch"):
        ingest_a5_campaign_data(document)


def test_mixed_measurement_boundaries_are_not_aggregated():
    from test_a5_comm_calibration import _campaign

    measurement, window = _native_child_measurement(), _native_child_window()
    document = _campaign(label="mixed-boundary", world_size=2, dtype="bf16",
                         conditions=[
        {"collective": "all_gather", "payload_bytes": 32768,
         "samples": [{"per_op_ms": 0.041113, "measurement": measurement,
                      "transport_window": window}]},
        {"collective": "all_gather", "payload_bytes": 32768,
         "samples": [{"per_op_ms": 0.09}]},
        {"collective": "all_gather", "payload_bytes": 1048576,
         "samples": [{"per_op_ms": 0.05}]},
    ])
    rows = ingest_a5_campaign_data(document)["conditions"]
    assert {(row["payload_bytes"], row["measurement"]["phase"])
            for row in rows} == {
        (32768, "aicpu_child"), (32768, "collective_call"),
        (1048576, "collective_call")}
    with pytest.raises(ValueError, match="a5_measurement_boundary_mixed_sources"):
        fit_a5_service_curves(ingest_a5_campaign_data(document))


def test_native_bundle_export_preserves_measurement_boundary():
    _require_raw_source()
    bundle = collect_native_service(
        str(SOURCE), stack="rack08", cann="9.2.0", execution_core="aicpu",
        placement="intra_node_ub", deployment_image_id=IMAGE, groups=(2,))
    exported = bundle.to_dict()
    curve = exported["curves"]["all_gather:W2:intra_node_ub:bf16:aicpu"]
    assert curve["resource"] == "aicpu_child"
    measurement = curve["provenance"]["measurement_boundary"]
    assert measurement["phase"] == "aicpu_child"
    assert measurement["complete_collective_call"] is False
    assert curve["provenance"]["service_semantics"] == (
        "aicpu_child_collective_phase")


def test_local_service_preserves_per_call_identity(local_services):
    for record in local_services.values():
        records = record["source_records"]
        assert len(records) > 8
        assert all({"native_id", "rank", "repeat_index", "marker", "status",
                    "used_in_fit", "work", "split", "source"} <= set(row)
                   for row in records)
        assert any(row["used_in_fit"] for row in records)
        assert record["fit"]["training_samples"] > 4
        assert len(record["heldout"]) > 4
        assert all("rank" in row["source"] and "rep" in row["source"]
                   for row in record["heldout"])
        assert record["service"]["provenance"]["aggregation"] == (
            "median_of_run_medians_per_rank")
        assert record["service"]["provenance"]["sample_source"] == (
            "native_ai_cpu_child")


def test_collective_envelope_is_not_declared_as_link_transit(embedded):
    _, bundle = embedded
    for key, curve in bundle.curves.items():
        collective, world_size, scope = key[:3]
        provenance = curve.provenance
        measurement = provenance["measurement_boundary"]
        if collective == "p2p_send_recv":
            assert curve.resource == "directed_p2p_call", key
            assert measurement["phase"] == "directed_p2p_call"
            assert measurement["complete_collective_call"] is False
            assert measurement["complete_directed_call"] is True
            assert provenance["direction_observed"] is True
        else:
            assert curve.resource == "collective_call", key
            assert measurement["phase"] == "collective_call"
            assert measurement["complete_collective_call"] is True
            assert measurement["complete_directed_call"] is False
            assert provenance["direction_observed"] is False
        assert provenance["service_semantics"] == measurement["service_semantics"]
        assert provenance["link_transit_identified"] is False
        assert provenance["hop_scaling"] == "not_derived"
        assert provenance["direction"] in {"read", "send", "directed", "0->1",
                                           "0->8", "0->24"}
        assert provenance["shared_domain"]["members"] == world_size
        assert provenance["shared_domain"]["scope"] == scope
        assert provenance["transport_window"]["kind"] in {
            "campaign_steady_state", "native_single_target_call"}
        assert provenance["work_scope"] == "captured_per_rank_payload"


def test_bound_collective_keeps_envelope_semantics_not_a_link_price(embedded):
    from test_a5_comm_calibration import _a5_case, _a5_comm_op

    _, bundle = embedded
    raw = _a5_case()
    operation = _a5_comm_op(raw)
    bound = bundle.bind_service_curve(raw, operation)
    assert bound.provenance["service_semantics"] == (
        "whole_collective_call_envelope")
    assert bound.provenance["consumer_role"] == (
        "collective_call_completion_envelope")
    assert bound.provenance["complete_collective_call_response"] is True
    assert bound.provenance["measurement_boundary"]["phase"] == "collective_call"
    assert bound.provenance["link_transit_identified"] is False
    assert bound.provenance["hop_scaling"] == "not_derived"
    assert bound.provenance["direction"] == "read"
    assert bound.provenance["shared_domain"]["kind"] == "ub_mesh"
    assert bound.provenance["shared_domain"]["members"] >= 2
    assert "wire_coordinate_semantics" in bound.provenance

    rule = bundle.to_service_regime(raw, operation)
    assert rule["provenance"]["boundary"] == "measured A5 collective_call response"
    assert rule["provenance"]["diagnostic_only"] is True
    assert rule["provenance"]["source_bearing"] is True
    assert rule["provenance"]["production_eligible"] is False
    assert rule["provenance"]["link_transit_identified"] is False
    assert rule["provenance"]["complete_collective_call_response"] is True
    assert rule["provenance"]["service_semantics"] == (
        "whole_collective_call_envelope")
    assert rule["provenance"]["hop_scaling"] == "not_derived"
    assert rule["provenance"]["coordinate_basis"] == (
        "captured_per_rank_payload_not_link_transit")
    with pytest.raises(ModelingError,
                       match="measured_a5_call_binding_is_diagnostic_only"):
        bundle.bind_service_curve(raw, operation, production=True)
    with pytest.raises(ModelingError,
                       match="to_service_regime_is_diagnostic_only"):
        bundle.to_service_regime(raw, operation, production=True)
    with pytest.raises(ModelingError,
                       match="measured_call_recipe_is_diagnostic_only"):
        bundle.build_a5_comm_recipe(
            "all_gather", world_size=8, production=True)


def test_native_child_binding_preserves_phase_and_rejects_complete_call():
    _require_raw_source()
    bundle = collect_native_service(
        str(SOURCE), stack="rack08", cann="9.2.0", execution_core="aicpu",
        placement="intra_node_ub", deployment_image_id=IMAGE, groups=(2,))
    case, operation = _case_op()
    selectors = dict(expansion_mode="aicpu", stack="rack08", cann="9.2.0",
                     placement_scope="intra_node_ub")
    native = bundle.lookup_curve(
        "all_gather", 2, placement_scope="intra_node_ub", dtype="bf16",
        expansion_mode="aicpu", stack="rack08", cann="9.2.0")
    assert native.knots[0][1] == pytest.approx(0.041113)

    bound = bundle.bind_service_curve(case, operation, **selectors)
    measurement = bound.provenance["measurement_boundary"]
    assert measurement["phase"] == "aicpu_child"
    assert measurement["complete_collective_call"] is False
    assert bound.provenance["source_service_semantics"] == (
        "aicpu_child_collective_phase")
    assert bound.provenance["service_semantics"] == (
        "aicpu_child_collective_phase")
    assert bound.provenance["consumer_role"] == "aicpu_child_completion_floor"
    assert bound.provenance["complete_collective_call_response"] is False
    assert bound.provenance["work_scope"] == "per_rank_input_bytes"
    assert bound.provenance["measured_resource"] == "aicpu_child"
    assert bound.provenance["consumption_resource"] == "link"
    assert bound.estimate(native.knots[0][0]).duration_ms == pytest.approx(
        0.041113)
    assert bound.estimate(native.knots[0][0]).duration_ms < 0.069943

    with pytest.raises(ModelingError,
                       match="cannot_serve_complete_collective_call"):
        bundle.bind_service_curve(
            case, operation, require_complete_call=True, **selectors)
    with pytest.raises(ModelingError,
                       match="cannot_serve_complete_collective_call"):
        bundle.to_service_regime(
            case, operation, require_complete_call=True, **selectors)
    with pytest.raises(ModelingError,
                       match="cannot_serve_complete_collective_call"):
        bundle.build_a5_comm_recipe(
            "all_gather", world_size=2, require_complete_call=True,
            **selectors).build(operation, None)

    rule = bundle.to_service_regime(case, operation, **selectors)
    assert rule["provenance"]["boundary"] == "measured A5 aicpu_child response"
    assert rule["provenance"]["diagnostic_only"] is True
    assert rule["provenance"]["source_bearing"] is True
    assert rule["provenance"]["production_eligible"] is False
    assert rule["provenance"]["complete_collective_call_response"] is False
    assert rule["provenance"]["source_service_semantics"] == (
        "aicpu_child_collective_phase")
    assert rule["provenance"]["service_semantics"] == (
        "aicpu_child_collective_phase")
    assert rule["provenance"]["consumer_role"] == "aicpu_child_completion_floor"
    assert rule["provenance"]["measured_resource"] == "aicpu_child"
    assert rule["provenance"]["consumption_resource"] == "link"

    recipe = bundle.build_a5_comm_recipe(
        "all_gather", world_size=2, **selectors)
    plan = recipe.build(operation, None)
    assert plan.measurement_boundary["phase"] == "aicpu_child"
    assert plan.consumer_role == "aicpu_child_completion_floor"
    assert plan.complete_collective_call_response is False
    assert plan.link_transit_identified is False
    assert plan.diagnostic_only is True
    assert plan.whole_call_floor is True
    assert plan.production_eligible is False
    assert plan.direction_observed is False
    assert plan.direction_basis == "hccl_source_template_native_direction"


def test_explicit_resource_phase_is_not_a_complete_call():
    from test_a5_comm_calibration import _campaign, _fit_campaign

    conditions = [
        {"collective": "all_gather", "payload_bytes": payload,
         "median_ms": 0.05 + payload * 1e-9}
        for payload in (1024, 1048576, 4194304)]
    bundle = _fit_campaign(_campaign(
        world_size=8, boundary="resource_phase", conditions=conditions))
    curve = next(iter(bundle.curves.values()))
    assert curve.resource == "collective_phase"
    measurement = curve.provenance["measurement_boundary"]
    assert measurement["phase"] == "collective_phase"
    assert measurement["complete_collective_call"] is False
    assert measurement["complete_directed_call"] is False
    assert curve.provenance["service_semantics"] == "collective_phase_service"
    assert curve.provenance["link_transit_identified"] is False


def _h5_service(rate, *, legacy_ms=None, active_units=1,
                rate_scope="aggregate"):
    provenance = {"source": "synthetic_h5_control"}
    if legacy_ms is not None:
        provenance["legacy_target_ms"] = legacy_ms
    return normalize_resource_service({
        "id": "h5-transfer", "scope": "independent_resource",
        "resources": ["link"], "phase": "transfer", "work_unit": "bytes",
        "boundary": "resource_phase", "startup_us": 2, "response_us": 0,
        "active_units": active_units, "rate_scope": rate_scope,
        "supply": {"kind": "absolute_rate", "rate": rate},
        "conditions": {"hardware": "TEST"},
        "provenance": provenance,
    })


def _h5_stage(work, *, active_units=1, work_scope="aggregate",
              resource="link", resources=None):
    stage = {
        "work": work, "work_scope": work_scope,
        "active_units": active_units, "phase": "transfer",
        "work_unit": "bytes", "boundary": "resource_phase",
    }
    if resources is not None:
        stage["resources"] = resources
    else:
        stage["resource"] = resource
    return stage


def _h5_context(work, *, active_units=1):
    return RecipeContext(
        op=SimpleNamespace(dtype="bf16", family="allgather",
                          link_bytes=work, metadata=()),
        work={"link:port0": float(work)}, resource_names={},
        rates={"link:port0": 1e12}, core_count=1,
        resource_units={"link": active_units},
        native_work={"link": float(work)},
        service_identity={"hardware": {"key": "TEST"}},
        work_units={"link": "bytes"}, service_boundary="resource_phase")


def test_h5_normalized_service_stage_fixed_once_and_new_work():
    baseline = _h5_service(96_000_000)
    assert normalized_service_duration_ms(
        baseline, 1536, work_scope="aggregate") == pytest.approx(0.018)
    assert normalized_service_duration_ms(
        baseline, 3072, work_scope="aggregate") == pytest.approx(0.034)
    curve = service_curve_from_normalized(
        baseline, identity={"hardware": {"key": "TEST"}}, resource="link",
        conditions_context={"hardware": "TEST"})
    assert curve.estimate(1536).duration_ms == pytest.approx(0.018)
    assert curve.estimate(3072).duration_ms == pytest.approx(0.034)
    assert curve.provenance["production_stage"] is True
    assert curve.provenance["service_work_scope"] == "aggregate"
    stage = build_normalized_service_stage(
        baseline, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(1536), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert stage.kind == "independent_resource"
    assert stage.physical_resource == "link"
    assert stage.duration_us == pytest.approx(18.0)
    assert stage.fixed_us == pytest.approx(2.0)
    assert stage.stage_cap_work_per_ms == pytest.approx(96_000)
    assert stage.aggregate_cap_work_per_ms == pytest.approx(96_000)
    assert stage.work == pytest.approx(1536)
    assert stage.work_total_aggregate == pytest.approx(1536)
    assert stage.production_eligible is True
    program = stage.resource_service.lower(_h5_context(1536))
    stages = {part.name: part for part in program.stages}
    assert stages["setup"].latency == pytest.approx(0.002)
    assert stages["service"].work["link:port0"] == pytest.approx(1536)
    assert stages["service"].caps["link:port0"] == pytest.approx(96_000)
    same_stage_recipe = build_normalized_service_recipe(
        baseline, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(1536), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert same_stage_recipe.lower(_h5_context(1536)).stages[1].caps[
        "link:port0"] == pytest.approx(96_000)

    second = build_normalized_service_stage(
        baseline, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(3072), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert second.duration_us == pytest.approx(34.0)
    assert second.stage_cap_work_per_ms == pytest.approx(96_000)

    with pytest.raises(ModelingError,
                       match="normalized_service_conditions_mismatch:hardware"):
        build_normalized_service_stage(
            baseline, identity={"hardware": {"key": "TEST"}},
            stage=_h5_stage(1536), conditions_context={"hardware": "OTHER"},
            require_conditions=True)

    faster = _h5_service(128_000_000)
    fast = build_normalized_service_stage(
        faster, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(1536), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert fast.duration_us == pytest.approx(14.0)
    assert fast.stage_cap_work_per_ms == pytest.approx(128_000)
    assert fast.resource_service.lower(_h5_context(1536)).stages[1].caps[
        "link:port0"] == pytest.approx(128_000)
    second_fast = build_normalized_service_stage(
        faster, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(3072), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert second_fast.duration_us == pytest.approx(26.0)

    # Legacy target duration is not an input to the ordinary service binding.
    noisy = _h5_service(96_000_000, legacy_ms=9_999_999)
    noisy_stage = build_normalized_service_stage(
        noisy, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(1536), conditions_context={"hardware": "TEST"},
        require_conditions=True)
    assert noisy_stage.duration_us == pytest.approx(18.0)
    assert noisy_stage.resource_service.model.estimate(
        1536).duration_ms == pytest.approx(0.018)


def test_h5_four_owner_aggregate_and_per_owner_are_equivalent():
    service = _h5_service(24_000_000, active_units=4,
                          rate_scope="per_owner")
    assert normalized_service_duration_ms(
        service, 1536, work_scope="aggregate") == pytest.approx(0.018)
    assert normalized_service_duration_ms(
        service, 3072, work_scope="aggregate") == pytest.approx(0.034)
    assert normalized_service_duration_ms(
        service, 384, work_scope="per_owner") == pytest.approx(0.018)
    assert normalized_service_duration_ms(
        service, 768, work_scope="per_owner") == pytest.approx(0.034)

    aggregate = build_normalized_service_stage(
        service, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(1536, active_units=4, work_scope="aggregate"),
        conditions_context={"hardware": "TEST"}, require_conditions=True)
    assert aggregate.duration_us == pytest.approx(18.0)
    assert aggregate.work == pytest.approx(1536)
    assert aggregate.work_total_aggregate == pytest.approx(1536)
    assert aggregate.owner_private_rate == pytest.approx(24_000_000)
    assert aggregate.aggregate_rate == pytest.approx(96_000_000)
    assert aggregate.stage_cap_work_per_ms == pytest.approx(96_000)
    assert aggregate.aggregate_cap_work_per_ms == pytest.approx(96_000)
    agg_program = aggregate.resource_service.lower(
        _h5_context(1536, active_units=4))
    assert agg_program.stages[1].work["link:port0"] == pytest.approx(1536)
    assert agg_program.stages[1].caps["link:port0"] == pytest.approx(96_000)
    assert agg_program.stages[0].latency == pytest.approx(0.002)

    per_owner = build_normalized_service_stage(
        service, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(384, active_units=4, work_scope="per_owner"),
        conditions_context={"hardware": "TEST"}, require_conditions=True)
    assert per_owner.duration_us == pytest.approx(18.0)
    assert per_owner.work == pytest.approx(384)
    assert per_owner.work_total_aggregate == pytest.approx(1536)
    assert per_owner.stage_cap_work_per_ms == pytest.approx(24_000)
    assert per_owner.aggregate_cap_work_per_ms == pytest.approx(96_000)
    own_program = per_owner.resource_service.lower(
        _h5_context(384, active_units=4))
    assert own_program.stages[1].work["link:port0"] == pytest.approx(384)
    assert own_program.stages[1].caps["link:port0"] == pytest.approx(24_000)
    assert own_program.stages[0].latency == pytest.approx(0.002)

    second = build_normalized_service_stage(
        service, identity={"hardware": {"key": "TEST"}},
        stage=_h5_stage(3072, active_units=4, work_scope="aggregate"),
        conditions_context={"hardware": "TEST"}, require_conditions=True)
    assert second.duration_us == pytest.approx(34.0)
    assert second.stage_cap_work_per_ms == pytest.approx(96_000)

    # r5 P1: the default recipe must bind the aggregate stage work even when
    # the service is stored per_owner; no explicit stage/work_scope.
    default_recipe = build_normalized_service_recipe(
        service, identity={"hardware": {"key": "TEST"}}, resource="link",
        conditions_context={"hardware": "TEST"})
    assert default_recipe.model.estimate(1536).duration_ms == pytest.approx(
        0.018)
    default_program = default_recipe.lower(_h5_context(1536, active_units=4))
    assert default_program.stages[0].latency == pytest.approx(0.002)
    assert default_program.stages[1].work["link:port0"] == pytest.approx(1536)
    assert default_program.stages[1].caps["link:port0"] == pytest.approx(
        96_000)
    default_3072 = build_normalized_service_recipe(
        service, identity={"hardware": {"key": "TEST"}}, resource="link",
        conditions_context={"hardware": "TEST", "dummy": True})
    # An unrelated extra condition is ignored; work stays in aggregate scope.
    assert default_3072.model.estimate(3072).duration_ms == pytest.approx(
        0.034)

    aggregate_equivalent = _h5_service(96_000_000)
    aggregate_recipe = build_normalized_service_recipe(
        aggregate_equivalent, identity={"hardware": {"key": "TEST"}},
        resource="link", conditions_context={"hardware": "TEST"})
    assert aggregate_recipe.model.estimate(1536).duration_ms == pytest.approx(
        0.018)
    assert aggregate_recipe.model.estimate(3072).duration_ms == pytest.approx(
        0.034)


def test_h5_joint_phase_is_not_compressed_to_single_link():
    joint = normalize_resource_service({
        "id": "h5-joint", "scope": "joint_phase",
        "resources": ["hbm", "mte2"], "phase": "joint",
        "work_unit": "bytes", "boundary": "resource_phase",
        "startup_us": 2, "response_us": 0, "active_units": 1,
        "supply": {"kind": "absolute_rate", "rate": 96_000_000},
        "conditions": {"hardware": "TEST"},
    })
    stage = build_normalized_service_stage(
        joint, identity={"hardware": {"key": "TEST"}},
        stage={
            "work": 1536, "work_scope": "aggregate", "active_units": 1,
            "phase": "joint", "work_unit": "bytes",
            "boundary": "resource_phase", "resources": ["hbm", "mte2"],
        },
        conditions_context={"hardware": "TEST"}, require_conditions=True)
    assert stage.kind == "joint_phase"
    assert stage.resource_service is None
    assert stage.physical_resource is None
    assert stage.resources == ("hbm", "mte2")
    assert stage.duration_us == pytest.approx(18.0)
    assert stage.aggregate_cap_work_per_ms == pytest.approx(96_000)
    assert stage.production_eligible is True
    with pytest.raises(ModelingError,
                       match="joint_phase_requires_joint_stage"):
        service_curve_from_normalized(
            joint, identity={"hardware": {"key": "TEST"}}, resource="link",
            conditions_context={"hardware": "TEST"})
    with pytest.raises(ModelingError,
                       match="joint_phase_requires_joint_stage"):
        build_normalized_service_recipe(
            joint, identity={"hardware": {"key": "TEST"}}, resource="link",
            conditions_context={"hardware": "TEST"})


def test_h5_child_resource_cannot_be_renamed_to_link(local_services):
    record = local_services["all_gather"]
    service = normalize_resource_service(record["service"])

    def child_stage(resource):
        return {
            "work": 786432, "work_scope": service.rate_scope,
            "active_units": service.active_units, "phase": service.phase,
            "work_unit": service.work_unit, "boundary": service.boundary,
            "resource": resource,
        }

    with pytest.raises(ModelingError,
                       match="normalized_service_resource_mismatch"):
        service_curve_from_normalized(
            service, identity={"hardware": {"key": "TEST"}},
            resource="link", conditions_context=record["domain"])
    with pytest.raises(ModelingError,
                       match="normalized_service_resource_mismatch"):
        build_normalized_service_stage(
            service, identity={"hardware": {"key": "TEST"}},
            stage=child_stage("link"),
            conditions_context=record["domain"],
            require_conditions=True)
    with pytest.raises(ModelingError,
                       match="normalized_service_resource_not_physical"):
        build_normalized_service_stage(
            service, identity={"hardware": {"key": "TEST"}},
            stage=child_stage("aicpu_child"),
            conditions_context=record["domain"],
            require_conditions=True)


def test_public_stage_entry_binds_normalized_service_and_shares_link():
    from test_a5_comm_calibration import _a5_case, _measurement_op

    raw = _a5_case()
    case = parse_case(raw)
    operation = _measurement_op(link_bytes=1536, hbm_bytes=0)
    second_operation = replace(operation, op_id="comm2")
    active = case.machine.dies[0].compute_units
    service = normalize_resource_service({
        "id": "h5-public", "scope": "independent_resource",
        "resources": ["link"], "phase": "transfer", "work_unit": "bytes",
        "boundary": "device_kernel", "startup_us": 2, "response_us": 0,
        "active_units": active,
        "supply": {"kind": "absolute_rate", "rate": 96_000_000},
        "conditions": {},
    })
    stage = build_normalized_service_stage(
        service, identity=service_identity(case, operation, "none"),
        stage={
            "work": 1536, "work_scope": "aggregate", "active_units": active,
            "phase": "transfer", "work_unit": "bytes",
            "boundary": "device_kernel", "resource": "link",
        },
        conditions_context={}, require_conditions=True)
    single = operator_costs(
        raw, (operation,),
        recipes=RecipeBook({operation.recipe: stage.resource_service}),
    )["operators"][0]
    assert single["duration_ms"] == pytest.approx(0.10200000000000001)
    model = single["basis"]["service_models"][0]
    assert model["provenance"]["service_work_scope"] == "aggregate"
    assert model["provenance"]["normalized_fixed_us"] == pytest.approx(2.0)
    assert model["provenance"]["aggregate_cap_work_per_ms"] == pytest.approx(
        96_000)
    # M2 uses directional per-pair UB resources, not per-port TX/RX aliases.
    # Eight ranks make eight directed pairs and each pair is charged once.
    expected_pairs = {
        f"link:d2d_intra_node:0:pair{rank}>{(rank + 1) % 8}"
        for rank in range(8)}
    assert set(single["declared_service"]) == expected_pairs
    assert set(single["declared_service"].values()) == {1536.0}

    shared = operator_costs(
        raw, (operation, second_operation),
        recipes=RecipeBook({
            operation.recipe: stage.resource_service,
            second_operation.recipe: stage.resource_service,
        }),
    )["operators"]
    assert len(shared) == 1
    assert shared[0]["occurrences"] == 2
    assert shared[0]["duration_ms"] == pytest.approx(0.10200000000000001)
    assert shared[0]["total_ms"] == pytest.approx(0.20400000000000001)


def test_same_engine_shared_link_and_reverse_independent():
    from mindie_sim.prediction.engine import Engine, Resource

    def stage_record(service, work):
        return build_normalized_service_stage(
            service, identity={"hardware": {"key": "TEST"}},
            stage=_h5_stage(work, active_units=4, work_scope="aggregate"),
            conditions_context={"hardware": "TEST"},
            require_conditions=True)

    def submit_flow(engine, name, resource, stage):
        setup = engine.submit(name + ":setup", latency=stage.fixed_us / 1000.0,
                              release=0)
        return engine.submit(
            name, {resource: stage.work}, after=(setup,),
            caps={resource: stage.stage_cap_work_per_ms}, release=0)

    service = _h5_service(24_000_000, active_units=4,
                          rate_scope="per_owner")
    first = stage_record(service, 1536)
    engine = Engine([Resource("link", 96_000, "B", "fair")])
    a = submit_flow(engine, "a", "link", first)
    b = submit_flow(engine, "b", "link", first)
    engine.advance(until=(b,))
    assert a.time == pytest.approx(0.034)
    assert b.time == pytest.approx(0.034)
    assert engine.serviced["link"] == pytest.approx(3072)
    assert engine.summary()["live_tasks"] == 0

    independent = Engine([
        Resource("tx", 96_000, "B", "fair"),
        Resource("rx", 96_000, "B", "fair"),
    ])
    x = submit_flow(independent, "x", "tx", first)
    y = submit_flow(independent, "y", "rx", first)
    independent.advance(until=(y,))
    assert x.time == pytest.approx(0.018)
    assert y.time == pytest.approx(0.018)

    fast = stage_record(_h5_service(32_000_000, active_units=4,
                                    rate_scope="per_owner"), 1536)
    fast_engine = Engine([Resource("link", 128_000, "B", "fair")])
    c = submit_flow(fast_engine, "c", "link", fast)
    d = submit_flow(fast_engine, "d", "link", fast)
    fast_engine.advance(until=(d,))
    assert c.time == pytest.approx(0.026)
    assert d.time == pytest.approx(0.026)
    assert fast_engine.serviced["link"] == pytest.approx(3072)


def test_native_child_stage_evidence_explains_786432_and_marks_tuning():
    _require_raw_source()
    evidence = native_child_stage_evidence(
        collective="all_gather", stack="rack08", cann="9.2.0",
        deployment_image_id=IMAGE, group_size=2, split="heldout",
        used_for_tuning=("native.all_gather.g2.h2048.t192.heldout",))
    assert evidence["schema"] == "a5-native-child-stage-evidence/1"
    assert evidence["aggregation"] == "per_call_no_cross_rank_median"
    assert len(evidence["records"]) == 40
    assert evidence["summary"]["used_for_tuning_count"] == 10
    assert evidence["summary"]["all_child_within_communication"] is True
    assert evidence["service_diagnostic_only"] is True
    assert evidence["service_production_eligible"] is False

    calls = [record for record in evidence["records"]
             if record["native_id"] == "native.all_gather.g2.h2048.t192.heldout"
             and record["rank"] == 0 and record["repeat_index"] == 4]
    assert len(calls) == 1
    call = calls[0]
    assert call["child_us"] == pytest.approx(22.886999999999716)
    assert call["overlap_us"] == pytest.approx(call["child_us"])
    assert call["combined_span_union_us"] == pytest.approx(
        call["communication_union_us"])
    assert call["child_within_communication"] is True
    assert call["predicted_us"] == pytest.approx(40.00526484133041)
    assert call["relative_error"] == pytest.approx(0.7479470809337574)
    assert call["original_split"] == "heldout"
    assert call["evidence_role"] == "heldout_used_for_tuning"
    assert call["used_for_tuning"] is True
    assert evidence["summary"]["fixed_term_exceeds_some_child_calls"] is True
    assert evidence["summary"]["child_by_work"]["786432"]["min_us"] == pytest.approx(
        22.886999999999716)

    untouched = native_child_stage_evidence(
        collective="all_gather", stack="rack08", cann="9.2.0",
        deployment_image_id=IMAGE, group_size=2, split="heldout")
    assert untouched["summary"]["used_for_tuning_count"] == 0
    assert all(record["evidence_role"] == "heldout"
               for record in untouched["records"])


def test_local_service_is_diagnostic_overlapped_not_complete_call(local_services):
    for collective, record in local_services.items():
        assert record["diagnostic_only"] is True
        assert record["production_eligible"] is False
        assert isinstance(record["overlap_with_communication"], bool)
        assert record["child_is_complete_collective"] is False
