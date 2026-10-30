"""One native invocation contains several concurrent resource graph phases."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, operator_costs
from mindie_sim.prediction.cost import CostModel, primitive_key
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, RecipeBook, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.operators import IterationWork
from mindie_sim.schema.workload import IterationPlan
from test_graph_analysis import op
from test_invocation_binding import declare
from test_mechanism_contract import hardware_case


def phase(name, *, group="dispatch", entry=False, exit=False, **kwargs):
    operation = op(name, **kwargs)
    return replace(operation, metadata=(("device_kernel_group", {
        "id": group, "kernel": "NativeDispatch", "entry": entry, "exit": exit}),))


def fork(group="dispatch", dependency=()):
    return (
        phase(group + ".entry", group=group, entry=True, memory=1000, deps=dependency),
        replace(phase(group + ".count", group=group, deps=(group + ".entry",),
                      stream="count"), vector_ops=2000),
        phase(group + ".payload", group=group, memory=4000,
              deps=(group + ".entry",), stream="payload"),
        replace(phase(group + ".exit", group=group, exit=True,
                      deps=(group + ".count", group + ".payload")), vector_ops=1000),
    )


@pytest.mark.parametrize("minimum,span", [(0, 9), (8, 9), (12, 12)])
def test_phase_fork_retains_overlap_and_inclusive_one_kernel_minimum(minimum, span):
    raw = declare(hardware_case(), startup=3, minimum=minimum,
                  match={"kernel": "NativeDispatch"})
    graph = graph_costs(raw, fork(dependency=("input",)), input_ready_ms=7,
                        external_inputs={"input": 9}, trace_limit=200)
    # 3 startup + 1 load + max(2 vector, 4 payload) + 1 combine.
    assert graph["complete_ms"] == 9 + span
    timeline = {row["op_id"]: row for row in graph["timeline"]}
    assert timeline["dispatch.count"]["ready_ms"] == 13
    assert timeline["dispatch.payload"]["ready_ms"] == 13
    assert graph["execution"]["serviced"]["memory@r0:s0"] == 5000
    assert graph["execution"]["serviced"]["vector@r0:s0"] == 3000
    assert all(event["start_ms"] >= 12 for event in graph["events"] if event["demand"])
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
    # Public isolated costs describe resource phases, never re-charge native
    # startup on their different phase recipe names.
    isolated = operator_costs(raw, fork())
    assert isolated["operator_sum_ms"] == 8


@pytest.mark.parametrize("scope,hosts,span", [("none", 2, 22), ("full", 1, 23)])
def test_groups_share_one_enqueue_each_and_one_capture_replay(scope, hosts, span):
    raw = declare(hardware_case(), startup=3, minimum=10,
                  match={"kernel": "NativeDispatch"})
    raw["machine"]["device"]["launch_us"] = 2000
    raw["runtime"]["submission"] = {"graph_replay_ms": 3}
    ops = fork("first") + fork("second", ("first.exit",))
    graph = graph_costs(raw, ops, replay_scope=scope, include_submission=True, isolated=False)
    assert graph["duration_ms"] == span
    submission = graph["execution"]["submission"]
    assert submission["eager_invocations"] == (hosts if scope == "none" else 0)
    assert submission["graph_replays"] == (hosts if scope == "full" else 0)
    assert submission["host_service_ms"] == (4 if scope == "none" else 3)


def test_iteration_instances_keep_their_floor_and_external_stream_waits_for_exit():
    raw = declare(hardware_case(), startup=3, minimum=12,
                  match={"kernel": "NativeDispatch"})
    parsed = parse_case(raw)
    execution = Execution(parsed, CostModel(parsed), include_submission=False)
    # Same plan id and group id must not reuse a previous group's ready/floor.
    work = IterationWork(fork(), IterationPlan(()), 1, 0, 0, None, "decode", "group")
    first = execution.add(work, 0, 5)
    second = execution.add(work, 0, 5)
    execution.engine.advance(until=(first.completion, second.completion), first=False)
    assert first.completion.time == 17
    assert second.completion.time == 29
    assert next(iter(execution.cost.invocations.values()))["invocations"] == 2
    assert execution.engine.summary()["live_tasks"] == 0
    # An unrelated op on a branch's stream waits for the kernel's full exit,
    # even if that branch completed early.
    report = graph_costs(raw, fork() + (op("later", memory=1000, stream="count"),))
    assert report["complete_ms"] == 13


def test_interleaved_phase_listing_cannot_reverse_kernel_order_on_a_branch():
    first, second = fork("first"), fork("second")
    operators = (first[0], second[0], second[1], first[1],
                 second[2], first[2], first[3], second[3])
    report = graph_costs(hardware_case(), operators, isolated=False)
    assert report["duration_ms"] == 12
    assert report["execution"]["live_tasks"] == 0


def test_late_external_input_of_a_branch_holds_the_native_kernel_entry():
    raw = declare(hardware_case(), startup=3, match={"kernel": "NativeDispatch"})
    operations = list(fork(dependency=("early",)))
    operations[2] = replace(operations[2], deps=("dispatch.entry", "late"))
    result = graph_costs(raw, operations, external_inputs={"early": 2, "late": 11},
                         trace_limit=200)
    assert result["complete_ms"] == 20
    assert all(event["start_ms"] >= 14 for event in result["events"] if event["demand"])


@pytest.mark.parametrize("minimum,span", [(0, 24), (10, 30)])
def test_prior_branch_stream_tail_is_part_of_whole_invocation_readiness(minimum, span):
    raw = declare(hardware_case(), startup=3, minimum=minimum,
                  match={"kernel": "NativeDispatch"})
    operations = (
        op("prior_payload_producer", memory=20000, stream="payload"),
        phase("entry", entry=True),
        phase("payload", memory=1000, stream="payload", deps=("entry",)),
        phase("exit", exit=True, deps=("payload",)),
    )
    result = graph_costs(raw, operations, isolated=False, trace_limit=200)
    timeline = {row["op_id"]: row for row in result["timeline"]}
    assert timeline["prior_payload_producer"]["complete_ms"] == 20
    assert timeline["entry"]["ready_ms"] == 23
    assert timeline["payload"]["ready_ms"] == 23
    assert result["complete_ms"] == span
    assert result["execution"]["serviced"]["memory@r0:s0"] == 21000
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_source_plans_are_resource_phases_but_whole_group_uses_mixed_hardware_class():
    raw = hardware_case()
    raw["machine"]["device"]["kernel_invocations"] = {
        "cube": {"startup_ms": 3, "minimum_ms": 10},
        "vector": {"startup_ms": 2, "minimum_ms": 8},
    }
    boundaries = []

    def planner(operation, geometry):
        work = Work(matrix={"bf16": 2000}) if operation.op_id == "matrix" else Work(resources={"vector": 1000})

        def lower(context):
            boundaries.append(context.service_boundary)
            assert context.invocation_timing is None
            return KernelProgram((Stage("service", context.bind(work)),))
        return OperatorPlan(work, lower)

    operations = (
        replace(phase("vector", entry=True), recipe="native_phase", source_boundary="kernel"),
        replace(phase("matrix", exit=True, deps=("vector",)), recipe="native_phase", source_boundary="kernel"),
    )
    recipes = RecipeBook({"native_phase": PlannedRecipe(planner)})
    result = graph_costs(raw, operations, recipes=recipes, isolated=False)
    assert result["duration_ms"] == 10
    assert boundaries == ["resource_phase", "resource_phase"]
    assert result["execution"]["serviced"]["matrix@r0:s0"] == 1
    assert result["execution"]["serviced"]["vector@r0:s0"] == 1000


def test_full_kernel_primitive_cannot_leak_into_a_phase_and_external_edges_use_exit():
    raw = hardware_case()
    operations = fork()
    bare = replace(operations[0], metadata=())
    parsed = parse_case(raw)
    assert primitive_key(parsed, operations[0], "none").to_dict()["scope"] == "resource_phase"
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parsed, bare, "none"), duration_ms=100,
                     run_id="independent-kernel", repeat_index=0, source="boundary-control")
        report = graph_costs(raw, operations, primitives=store)
    assert report["duration_ms"] == 6
    with pytest.raises(ValueError, match="external_dependency_requires_exit"):
        graph_costs(raw, operations + (op("wrong_consumer", deps=("dispatch.count",)),))


@pytest.mark.parametrize("timing_basis", ["runtime_rule", "hardware_class"])
def test_cross_dp_count_and_payload_remain_transport_only_under_shared_invocation(timing_basis):
    from test_communication_topology import exchange
    from test_joint_collective_execution import cohort_case

    raw = declare(cohort_case(2), startup=3, minimum=10,
                  match={"kernel": "NativeDispatch"})
    raw["runtime"]["kernels"]["families"]["alltoall"] = {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 1}
    raw["machine"]["device"]["launch_us"] = 2000
    if timing_basis == "hardware_class":
        raw["runtime"]["kernels"].pop("invocations")
        raw["machine"]["device"]["kernel_invocations"] = {
            "vector": {"startup_ms": 3, "minimum_ms": 10}}
    graphs = {}
    for rank in (0, 1):
        entry = replace(phase("entry", entry=True, memory=1000), vector_ops=1000)
        transports = []
        for name, volume in (("count", 2), ("payload", 4)):
            transfer = exchange(volume, axis="ep")
            transports.append(replace(
                transfer, op_id=name, deps=("entry",), stream=name,
                metadata=transfer.metadata + phase(name).metadata))
        finish = phase("exit", exit=True, deps=("count", "payload"))
        graphs[rank] = (entry, *transports, finish)
    report = graph_costs(raw, graphs, include_submission=True, trace_limit=200)
    # Both direct messages contend for one per-pair link: 2 + 4 bytes at1 B/ms.
    # All group entry/startup, network and minimum constraints are still causal.
    assert report["duration_ms"] == pytest.approx(12)
    assert sum(v for k, v in report["execution"]["serviced"].items() if k.startswith("link:")) == 12
    assert report["execution"]["submission"]["eager_invocations"] == 2
    assert report["execution"]["submission"]["host_service_ms"] == 4
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


@pytest.mark.parametrize("volumes", [(0, 4), (0, 0)])
def test_declared_native_kernel_keeps_receive_only_invocation_without_inventing_execution_class(volumes):
    from test_communication_topology import exchange
    from test_joint_collective_execution import cohort_case

    raw = declare(cohort_case(2), startup=3, minimum=10,
                  match={"kernel": "NativeDispatch"})
    raw["runtime"]["kernels"]["families"]["alltoall"] = {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 1}
    graphs = {}
    for rank, volume in enumerate(volumes):
        transfer = exchange(volume, axis="ep")
        graphs[rank] = (
            phase("entry", entry=True),
            replace(transfer, op_id="payload", deps=("entry",),
                    metadata=transfer.metadata + phase("payload").metadata),
            phase("exit", exit=True, deps=("payload",)),
        )
    report = graph_costs(raw, graphs, isolated=False)
    timeline = {row["id"]: row for row in report["timeline"]}
    assert [timeline[f"r{rank}:entry"]["ready_ms"] for rank in (0, 1)] == [3, 3]
    assert [timeline[f"r{rank}:exit"]["complete_ms"] for rank in (0, 1)] == [10, 10]
    assert sum(row["invocations"] for row in report["parameters"]["invocation_models"]) == 2
    assert sum(v for k, v in report["execution"]["serviced"].items() if k.startswith("link:")) == sum(volumes)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

    # A native name proves an invocation, not a Cube/Vector execution class.
    raw["runtime"]["kernels"].pop("invocations")
    raw["machine"]["device"]["kernel_invocations"] = {
        "vector": {"startup_ms": 30, "minimum_ms": 100},
        "cube": {"startup_ms": 40, "minimum_ms": 200},
    }
    unclassified = graph_costs(raw, graphs, isolated=False)
    assert unclassified["duration_ms"] == max(volumes)
    assert not unclassified["parameters"].get("invocation_models")
