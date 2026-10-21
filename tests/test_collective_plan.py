"""Synthetic rendezvous and wire-service checks; no NPU or network needed."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from mindie_sim.a3_resource_engine import ResourceEngine, ResourceSpec
from mindie_sim.a3_types import A3InputError
from mindie_sim.collective_plan import (
    CollectiveDescriptor, CollectiveKey, CollectiveLinkStage,
    collective_plan_report, register_collective,
)


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    return json.loads((ROOT / "examples/collectives/shared-link.json").read_text(encoding="utf-8"))


def one_group():
    data = fixture()
    data["collectives"] = data["collectives"][:1]
    data["consumers"] = data["consumers"][:2]
    data["producers"] = data["producers"][:2]
    return data


def traces(report):
    return {item["task_id"]: item for item in report["execution"]["trace"]}


def test_zero_payload_member_still_delays_all_consumers():
    data = one_group()
    report = collective_plan_report(data)
    group = report["collectives"][0]
    events = traces(report)
    assert group["producer_completion_ms"] == {"0": {"r0.produce": 2}, "1": {"r1.produce": 5}}
    assert group["rendezvous_ready_ms"] == 5
    assert group["complete_ms"] == 10
    assert events["r0.consume"]["start_ms"] == events["r1.consume"]["start_ms"] == 10
    assert report["elapsed_ms"] == 11
    assert group["byte_accounting"]["logical_payload_bytes_by_rank"]["1"] == 0
    assert report["execution"]["serviced"]["link:fabric"] == 500
    for task in [group["task_ids"]["join"], group["task_ids"]["complete"],
                 *group["task_ids"]["consumer_gates"].values()]:
        assert events[task]["demands"] == {}


def test_shared_wire_reallocates_service_without_world_barrier_or_duplicate_billing():
    report = collective_plan_report(fixture())
    a, b = report["collectives"]
    assert (a["rendezvous_ready_ms"], b["rendezvous_ready_ms"]) == (5, 1)
    assert (a["complete_ms"], b["complete_ms"]) == (12, 9)
    assert report["elapsed_ms"] == 13
    events = traces(report)
    a_wire = events[a["task_ids"]["wire_stages"]["wire"]]
    b_wire = events[b["task_ids"]["wire_stages"]["wire"]]
    assert [(s["start_ms"], s["end_ms"], s["rate"]) for s in a_wire["service"]] == [
        (5, 9, 50), (9, 12, 100),
    ]
    # B starts at 1 ms while A's zero-payload member is still computing.
    assert b_wire["start_ms"] == 1
    assert sum((s["end_ms"] - s["start_ms"]) * s["rate"] for s in b_wire["service"]) == 600
    assert report["execution"]["serviced"]["link:fabric"] == 1100
    assert report["byte_accounting"]["declared_collective_physical_link_bytes"] == {"link:fabric": 1100}
    assert report["byte_accounting"]["collectives_conserved"] is True
    assert all(group["byte_accounting"]["conserved"] for group in (a, b))


def test_disjoint_wires_do_not_share_capacity():
    data = fixture()
    data["resources"].append({"name": "link:other", "capacity": 100, "unit": "bytes"})
    data["collectives"][1]["link_stages"][0]["link_bytes"] = {"link:other": 600}
    report = collective_plan_report(data)
    assert [group["complete_ms"] for group in report["collectives"]] == [10, 7]
    assert report["execution"]["serviced"]["link:fabric"] == 500
    assert report["execution"]["serviced"]["link:other"] == 600


@pytest.mark.parametrize("field,value", [("epoch", 1), ("group", "another"), ("stage", 1),
                                       ("layer", 1), ("kind", "all_gather")])
def test_identity_dimensions_do_not_merge(field, value):
    data = one_group()
    original = data["collectives"][0]
    second = copy.deepcopy(original)
    second[field] = value
    data["collectives"].append(second)
    report = collective_plan_report(data)
    first, other = report["collectives"]
    assert first["task_ids"]["complete"] != other["task_ids"]["complete"]
    assert first["complete_ms"] == other["complete_ms"] == 15
    assert report["execution"]["serviced"]["link:fabric"] == 1000


def test_consumer_waits_for_requested_epoch_only():
    data = one_group()
    other = copy.deepcopy(data["collectives"][0])
    other["epoch"] = 1
    other["link_stages"][0]["link_bytes"]["link:fabric"] = 100
    data["collectives"].append(other)
    data["consumers"][0]["wait_for"][0]["epoch"] = 1
    report = collective_plan_report(data)
    assert [group["complete_ms"] for group in report["collectives"]] == [11, 7]
    assert traces(report)["r0.consume"]["start_ms"] == 7
    assert traces(report)["r1.consume"]["start_ms"] == 11


def test_multiple_roots_and_all_terminal_wire_stages_join():
    data = one_group()
    data["resources"].append({"name": "link:other", "capacity": 100, "unit": "bytes"})
    data["collectives"][0]["link_stages"] = [
        {"name": "left", "link_bytes": {"link:fabric": 100}},
        {"name": "right", "link_bytes": {"link:other": 200}},
        {"name": "tail", "link_bytes": {"link:fabric": 300}, "deps": ["left"]},
    ]
    report = collective_plan_report(data)
    group = report["collectives"][0]
    assert group["complete_ms"] == 9
    assert traces(report)["r0.consume"]["start_ms"] == 9
    assert group["byte_accounting"]["declared_physical_link_bytes"] == {"link:fabric": 400, "link:other": 200}


def test_barrier_with_no_wire_work_and_zero_payload_collective():
    data = one_group()
    data["collectives"][0].update(kind="barrier", link_stages=[])
    for consumer in data["consumers"]:
        consumer["wait_for"][0]["kind"] = "barrier"
    report = collective_plan_report(data)
    assert report["collectives"][0]["complete_ms"] == 5
    assert report["execution"]["serviced"]["link:fabric"] == 0
    assert report["elapsed_ms"] == 6
    # Explicit zero-byte service stages still rendezvous and complete.
    data = one_group()
    data["collectives"][0]["link_stages"][0]["link_bytes"] = {"link:fabric": 0}
    report = collective_plan_report(data)
    assert report["collectives"][0]["complete_ms"] == 5
    assert report["collectives"][0]["byte_accounting"]["conserved"] is True


def test_two_member_producers_can_contribute_multiple_tasks():
    data = one_group()
    data["producers"].append({"task_id": "late", "release_ms": 7, "demands": {}})
    data["collectives"][0]["producers_by_rank"]["0"].append("late")
    report = collective_plan_report(data)
    assert report["collectives"][0]["rendezvous_ready_ms"] == 7


def test_public_api_reuses_existing_engine_and_gates_later_work():
    engine = ResourceEngine([ResourceSpec("local", 1, "ms"), ResourceSpec("link:wire", 100, "bytes")])
    engine.add_task("producer0", {"local": 2})
    engine.add_task("producer1", {}, release_ms=5)
    descriptor = CollectiveDescriptor(
        CollectiveKey("all_to_all", "explicit-members", 3, 1, 2), (0, 9),
        ((0, 1000), (9, 0)), ((0, ("producer0",)), (9, ("producer1",))),
        (CollectiveLinkStage("transfer", (("link:wire", 100),)),),
    )
    registration = register_collective(engine, descriptor)
    engine.add_task("consumer", {"local": 1}, dependencies=(registration.consumer_gate(9),))
    result = engine.run()
    assert result["completed_at_ms"][registration.join_task_id] == 5
    assert result["completed_at_ms"]["consumer"] == 7
    assert result["serviced"]["link:wire"] == 100
    assert set(engine.resources) == {"local", "link:wire"}
    with pytest.raises(A3InputError, match="consumer_not_participant"):
        registration.consumer_gate(1)


def test_invalid_public_registration_does_not_partially_mutate_engine():
    engine = ResourceEngine([ResourceSpec("link:wire", 100, "bytes")])
    descriptor = CollectiveDescriptor(
        CollectiveKey("all_reduce", "g", 0, 0, 0), (0,), ((0, 0),), ((0, ("missing",)),),
        (CollectiveLinkStage("transfer", (("link:wire", 100),)),),
    )
    with pytest.raises(A3InputError, match="producer_unknown"):
        register_collective(engine, descriptor)
    assert not engine.tasks
    engine.add_task("missing", {})
    register_collective(engine, descriptor)
    before = set(engine.tasks)
    with pytest.raises(A3InputError, match="identity_or_task_collision"):
        register_collective(engine, descriptor)
    assert set(engine.tasks) == before


@pytest.mark.parametrize("field,remove,extra,reason", [
    ("producers_by_rank", True, False, "producer_members_mismatch:missing=\\[1\\]"),
    ("producers_by_rank", False, True, "producer_members_mismatch:missing=\\[\\]:extra=\\[2\\]"),
    ("payload_bytes_by_rank", True, False, "payload_members_mismatch:missing=\\[1\\]"),
    ("payload_bytes_by_rank", False, True, "payload_members_mismatch:missing=\\[\\]:extra=\\[2\\]"),
])
def test_exact_participant_sets(field, remove, extra, reason):
    data = one_group()
    members = data["collectives"][0][field]
    if remove:
        del members["1"]
    if extra:
        members["2"] = ["r0.produce"] if field == "producers_by_rank" else 0
    with pytest.raises(A3InputError, match=reason):
        collective_plan_report(data)


@pytest.mark.parametrize("mutation,reason", [
    (lambda d: d["collectives"][0].update(participants=[0, 0]), "participants_empty_or_duplicate"),
    (lambda d: d["collectives"][0].update(epoch=True), "epoch_invalid"),
    (lambda d: d["collectives"][0].update(epoch=-1), "epoch_invalid"),
    (lambda d: d["collectives"][0].update(streaming=True), "descriptor_invalid"),
    (lambda d: d["collectives"][0]["producers_by_rank"].update({"1": []}), "producers_empty_or_duplicate"),
    (lambda d: d["collectives"][0]["producers_by_rank"].update({"1": ["unknown"]}), "producer_unknown"),
    (lambda d: d["collectives"][0]["link_stages"][0]["link_bytes"].update({"local:0": 1}), "link_work_invalid"),
    (lambda d: d["collectives"][0]["link_stages"][0]["link_bytes"].update({"link:unknown": 1}), "physical_link_missing"),
    (lambda d: d["collectives"][0]["link_stages"][0].update(deps=["future"]), "dependency_not_prior"),
    (lambda d: d["collectives"][0]["payload_bytes_by_rank"].update({"1": -1}), "payload_invalid"),
    (lambda d: d["collectives"][0]["payload_bytes_by_rank"].update({"1": float("nan")}), "payload_invalid"),
    (lambda d: d["consumers"][0]["wait_for"][0].update(epoch=99), "consumer_identity_unknown"),
    (lambda d: d["consumers"][0].update(rank=9), "consumer_not_participant"),
    (lambda d: d["collectives"].append(copy.deepcopy(d["collectives"][0])), "identity_duplicate"),
])
def test_invalid_contracts_fail_explicitly(mutation, reason):
    data = one_group()
    mutation(data)
    with pytest.raises(A3InputError, match=reason):
        collective_plan_report(data)


def test_empty_collective_list_leaves_independent_work_unchanged():
    data = one_group()
    data["collectives"] = []
    for consumer in data["consumers"]:
        consumer["dependencies"] = [f'r{consumer["rank"]}.produce']
        consumer["wait_for"] = []
    report = collective_plan_report(data)
    baseline = ResourceEngine([
        *(ResourceSpec(f"local:{rank}", 1, "ms") for rank in range(4)),
        ResourceSpec("link:fabric", 100, "bytes"),
    ])
    for rank, work in enumerate((2, 5)):
        baseline.add_task(f"r{rank}.produce", {f"local:{rank}": work}, owner="declared_work")
    for rank in range(2):
        baseline.add_task(f"r{rank}.consume", {f"local:{rank}": 1}, dependencies=(f"r{rank}.produce",),
                          owner="declared_work")
    expected = baseline.run()
    assert report["execution"]["trace"] == expected["trace"]
    assert report["execution"]["serviced"] == expected["serviced"]
    assert report["elapsed_ms"] == 6


def test_report_roundtrip_hash_and_cli_output(tmp_path):
    data = fixture()
    report = collective_plan_report(data)
    assert report["effective_input"] == data
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert report["input_sha256"] == hashlib.sha256(canonical.encode()).hexdigest()
    for original, result in zip(data["collectives"], report["collectives"]):
        descriptor = CollectiveDescriptor.from_mapping(original)
        assert CollectiveDescriptor.from_mapping(descriptor.to_dict()) == descriptor
        assert result["descriptor"] == descriptor.to_dict()
    output = tmp_path / "collective.json"
    run = subprocess.run([
        sys.executable, "-m", "mindie_sim", "collective-plan", "--input",
        str(ROOT / "examples/collectives/shared-link.json"), "--output", str(output),
    ], cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert run.returncode == 0, run.stdout + run.stderr
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_cli_reports_missing_member_as_nonzero_structured_failure(tmp_path):
    data = one_group()
    del data["collectives"][0]["producers_by_rank"]["1"]
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    run = subprocess.run([sys.executable, "-m", "mindie_sim", "collective-plan", "--input", str(path)],
                         cwd=ROOT, text=True, capture_output=True, timeout=30)
    assert run.returncode == 1
    error = json.loads(run.stdout)
    assert error["status"] == "error" and error["command"] == "collective-plan"
    assert "producer_members_mismatch" in error["reason"]
