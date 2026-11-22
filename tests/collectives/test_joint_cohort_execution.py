"""Cross-replica collective readiness, native binding, and service queues.

[Role] Compare joint plans against rank-local causal schedules and exercise
  both serving observations and the native participant launch boundary.
[Invariants] Every payload has one source, and a peer can wait only for its
  declared predecessors. Observation annotations preserve modeled service.
[Timeouts & Locks] Bounded in-memory Engines and primitive databases.
[Side-Effect Policy] Read-only inputs and isolated simulation state.
"""
import json
import math
import random
from copy import deepcopy
from dataclasses import replace
from itertools import combinations
from types import SimpleNamespace

import pytest
from _support.collective_cases import (
    case, cohort_case, copy_op, exchange, joint_case, per_pair_topology, ready_events, topology, wire,
)
from _support.kernel_invocation import fork, lower_copy, op

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction.frontend.body_spec import body_forward
from mindie_sim.prediction.collectives import PairwiseCollective
from mindie_sim.prediction.efficiency.cost import primitive_key
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution, launch_native_cohort
from mindie_sim.prediction.operators.mc2.mc2_program import Mc2LaunchView
from mindie_sim.prediction.graph_analysis import normalise, select_call
from mindie_sim.prediction.program.primitives import PrimitiveStore
from mindie_sim.prediction.program import KernelProgram, NativeProgram, RecipeContext, Stage
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

pytestmark = pytest.mark.tier0


def test_ring_per_rank_ready_and_completion_keep_predecessor_causality():
    topo = per_pair_topology(4)
    group = topo.ring_plan(exchange(3, family="allgather"), 0).groups[0]
    engine = Engine(topo.resources(), trace_limit=200)
    initial = [100, 0, 0, 0]
    ready = ready_events(engine, initial)
    events = group.start_participants(engine, "ringready", ready)
    engine.advance(until=tuple(events.values()), first=False)
    sent = [max(initial[i], initial[(i + 1) % 4]) + 1 for i in range(4)]
    for _ in range(2):
        sent = [max(sent[i], sent[i - 1]) + 1 for i in range(4)]
    assert [event.time for event in events.values()] == [max(sent[i], sent[i - 1]) for i in range(4)]
    early = next(row for row in engine.trace
                 if ":rank2:round1:hop0:service" in row["task"])
    late = next(row for row in engine.trace
                if ":rank0:round0:hop0:arrival" in row["task"])
    assert early["start_ms"] < late["end_ms"]


def test_pairwise_multihop_waits_for_each_framed_hop():
    raw = case(tp=2)
    raw["machine"]["num_devices"] = 3
    raw["machine"]["links"] = [
        {"name": name, "members": members, "capacity_bytes_per_ms": 1000,
         "transaction_bytes": frame, "protocol_overhead_bytes": header, "latency_us": latency}
        for name, members, frame, header, latency in (
            ("a", [0, 2], 16, 4, 200), ("b", [2, 1], 32, 8, 300))
    ]
    topo = topology(raw)
    op = exchange(2, family="alltoall")
    plan = topo.joint_plan(op, {0: 2}, algorithm="pairwise")
    context = RecipeContext(op, {}, {}, {}, 1, setup_ms=999, native_work={"link": 4},
                            collective_plan=lambda: plan)
    with pytest.raises(ModelingError, match="multihop_requires_store_forward"):
        PairwiseCollective().lower(context)
    engine = Engine(topo.resources(), trace_limit=200)
    program = PairwiseCollective(path_model="store_forward").lower(context)
    complete = program.start(engine, "multihop", engine.submit("entry"))
    engine.advance(until=(complete,))
    assert complete.time == pytest.approx(.02 + .2 + .04 + .3)
    stages = {row["task"].split(":rank0:round0:")[-1]: row
              for row in engine.trace if ":rank0:round0:" in row["task"]}
    assert stages["hop1:service"]["start_ms"] == stages["hop0:arrival"]["end_ms"]
    assert engine.serviced == {"link:a": 40, "link:b": 80}


def test_cross_dp_ring_joint_plan_accounts_complete_groups_once():
    raw = case(dp=4, tp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    topo = topology(raw)
    op = exchange(12, family="allreduce", axis="ep")
    plan = topo.joint_plan(op, {0: 12, 1: 12, 2: 20, 3: 20}, algorithm="ring")

    assert [group.ranks for group in plan.groups] == [
        (0, 1, 2, 3), (4, 5, 6, 7)]
    assert [group.rounds for group in plan.groups] == [6, 6]
    for group, volume in zip(plan.groups, (12, 20)):
        assert [m.flow.payload_bytes for m in group.messages] == [volume / group.rounds] * 4
    with pytest.raises(ModelingError, match="equal_per_rank_volume"):
        topo.joint_plan(op, {0: 12, 1: 16, 2: 20, 3: 20}, algorithm="ring")
    with pytest.raises(ModelingError, match="requires_complete_local_group"):
        topo.ring_plan(op, 0)

    engine = Engine(topo.resources())
    ready = ready_events(engine, [0] * 8)
    events = plan.start_participants(engine, "cross", ready)
    engine.advance(until=tuple(events.values()), first=False)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_joint_plan_rejects_missing_group_members_and_alltoallv():
    raw = case(dp=2, tp=1, ep=2)
    raw["model"].update(ffn="moe", num_experts=2, top_k=1, moe_intermediate_size=32)
    topo = topology(raw)
    with pytest.raises(ModelingError, match="group_membership_missing"):
        topo.joint_plan(exchange(2, family="allreduce", axis="ep"), {0: 2},
                        algorithm="ring")
    with pytest.raises(ModelingError, match="explicit_peer_matrix"):
        topo.joint_plan(exchange(2, family="alltoallv"), {0: 2},
                        algorithm="pairwise")


def test_pairwise_lower_charges_only_transport_and_the_inclusive_minimum():
    topo = per_pair_topology(3)
    op = exchange(3, family="alltoall")
    plan = topo.joint_plan(op, {0: 3}, algorithm="pairwise")
    ctx = RecipeContext(op, {}, {}, {}, 1, minimum_ms=3.5, setup_ms=999,
                        native_work={"link": 6}, collective_plan=lambda: plan)
    engine = Engine(topo.resources())
    program = PairwiseCollective().lower(ctx)
    complete = program.start(engine, "lower", engine.submit("ready"))
    engine.advance(until=(complete,))
    assert complete.time == pytest.approx(3.5)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    with pytest.raises(ModelingError, match="transport_only_work"):
        PairwiseCollective().lower(replace(ctx, native_work={"link": 6, "vector": 1}))
    with pytest.raises(ModelingError, match="path_model_unsupported"):
        PairwiseCollective(path_model="cut_through").lower(ctx)


def test_repeated_pairwise_schedules_keep_bounded_live_state():
    topo = per_pair_topology(5)
    plan = topo.joint_plan(exchange(4, family="alltoall"), {0: 4},
                           algorithm="pairwise")

    def peak(repetitions):
        engine = Engine(topo.resources())
        for repetition in range(repetitions):
            ready = ready_events(engine, [0] * 5)
            events = plan.start_participants(
                engine, f"repeat{repetition}", ready)
            engine.advance(until=tuple(events.values()), first=False)
            assert engine.summary()["live_tasks"] == 0
            assert engine.summary()["live_programs"] == 0
        return engine.peak_tasks

    peaks = [peak(20), peak(200)]
    assert peaks[0] == peaks[1]


@pytest.mark.parametrize("count", range(2, 8))
def test_asymmetric_pairwise_matches_independent_framed_rank_recurrence(count):
    """Each directed pair has its own capacity, so no simulator helper is needed."""
    rng = random.Random(31029 + count)
    for scenario in range(8):
        raw = case(dp=count, tp=1, ep=count)
        raw["model"].update(ffn="moe", num_experts=count, top_k=1,
                            moe_intermediate_size=32)
        initial = [rng.choice((0, 0, 1, 17)) for _ in range(count)]
        volumes = [rng.choice((0, 1, 3, 17)) for _ in range(count)]
        volumes[scenario % count] = 17
        links, specs = [], {}
        for source, destination in combinations(range(count), 2):
            rate = rng.choice((.5, 1, 2, 5))
            latency = rng.choice((0, 1, 9, 40))
            frame = rng.choice((None, 2, 4, 8))
            header = rng.choice((0, 1, 3)) if frame else 0
            spec = (rate, latency, frame, header)
            specs[source, destination] = specs[destination, source] = spec
            link = {
                "name": f"p{source}_{destination}",
                "members": [source, destination], "semantics": "per_pair",
                "capacity_bytes_per_ms": rate, "latency_us": latency * 1000,
            }
            if frame:
                link.update(transaction_bytes=frame, protocol_overhead_bytes=header)
            links.append(link)
        raw["machine"]["links"] = links
        topo = topology(raw)
        plan = topo.joint_plan(
            exchange(0, family="alltoall", axis="ep"),
            dict(enumerate(volumes)), algorithm="pairwise")
        engine = Engine(topo.resources(), trace_limit=2048)
        events = plan.start_participants(
            engine, "asymmetric", ready_events(engine, initial))
        engine.advance(until=tuple(events.values()), first=False)

        last_send = list(initial)
        last_receive = list(initial)
        expected_wire = {}
        for round_index in range(count - 1):
            next_send, next_receive = [0.0] * count, [0.0] * count
            for source in range(count):
                destination = (source + round_index + 1) % count
                rate, latency, frame, header = specs[source, destination]
                payload = volumes[source] / (count - 1)
                wire = (math.ceil(payload / frame) * (frame + header)
                        if frame else payload)
                start = max(initial[source], last_send[source],
                            last_receive[source])
                if payload:
                    start = max(start, initial[destination])
                finish = start + wire / rate
                next_send[source] = finish
                next_receive[destination] = max(
                    last_receive[destination], initial[destination],
                    finish + (latency if payload else 0))
                edge = f"p{min(source, destination)}_{max(source, destination)}"
                expected_wire[f"link:{edge}:pair{source}>{destination}"] = wire
            last_send, last_receive = next_send, next_receive
        expected = {rank: max(last_send[rank], last_receive[rank])
                    for rank in range(count)}

        assert {rank: event.time for rank, event in events.items()} == pytest.approx(expected)
        assert engine.serviced == pytest.approx(expected_wire)
        assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
        for row in engine.trace:
            if row["demand"]:
                source = int(row["task"].split(":rank")[1].split(":")[0])
                round_index = int(row["task"].split(":round")[1].split(":")[0])
                destination = (source + round_index + 1) % count
                assert row["start_ms"] >= max(initial[source], initial[destination])


@pytest.mark.parametrize("algorithm,family,count,volume", [
    ("pairwise", "alltoall", 1, 4), ("pairwise", "alltoall", 4, 0),
    ("ring", "allgather", 4, 0),
])
def test_empty_wire_keeps_each_participant_ready_boundary(algorithm, family, count, volume):
    topo = per_pair_topology(count)
    op = exchange(volume, family=family)
    plan = (topo.pairwise_plan(op, 0) if algorithm == "pairwise"
            else topo.joint_plan(op, {0: volume}, algorithm=algorithm))
    engine = Engine(topo.resources())
    ready = ready_events(engine, [7] * count)
    with pytest.raises(ModelingError, match="ready_coverage_mismatch"):
        plan.start_participants(engine, "incomplete", {count: ready[0]})
    events = plan.start_participants(engine, "empty", ready)
    engine.advance(until=tuple(events.values()), first=False)
    assert all(event.time == 7 for event in events.values())
    assert not any(engine.serviced.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_joint_graph_keeps_idle_receives_local_entry_and_early_replica_completion():
    raw = cohort_case(3)
    graphs = {
        rank: (
            op("producer", flops=2e6 if rank == 1 else 0),
            replace(exchange(volume, axis="ep"), deps=("producer",)),
            op("consumer", flops=1e6 if rank == 0 else 0, deps=("exchange",)),
        )
        for rank, volume in enumerate((0, 4, 8))
    }
    report = graph_costs(raw, graphs, trace_limit=200)
    indexed = {row["id"]: row for row in report["timeline"]}
    assert [indexed[f"r{r}:exchange"]["ready_ms"] for r in range(3)] == [0, 2, 0]
    assert [indexed[f"r{r}:exchange"]["complete_ms"] for r in range(3)] == [6, 8, 8]
    assert indexed["r0:consumer"]["complete_ms"] == 7
    assert report["duration_ms"] == 8 and sum(wire(report).values()) == 12
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

    isolated = report["isolated"]
    rows = [row for row in isolated["operators"] if row["program"] == "PairwiseCollective"]
    # Isolation removes the 2ms producer delay, so rank 0 completes at 4ms.
    assert [row["duration_ms"] for row in rows] == [4, 8, 8]
    assert [row["communication"]["wire_bytes"] for row in rows] == [0, 4, 8]
    assert isolated["collective_cohorts"][0]["duration_ms"] == 8
    assert isolated["operator_sum_ms"] == 23
    assert not rows[0]["serviced"]


def test_primitive_minimum_delays_only_its_own_replica_completion():
    raw = cohort_case(2)
    idle, sender = exchange(0, axis="ep"), exchange(4, axis="ep")
    parsed = parse_case(raw)
    with PrimitiveStore(":memory:") as primitives:
        primitives.record(
            primitive_key(parsed, idle, "none"), duration_ms=20,
            run_id="local-completion-control", repeat_index=0,
            source="independent-test-constraint")
        report = graph_costs(
            raw, {0: (idle,), 1: (sender, op("after", flops=1e6, deps=("exchange",)))},
            primitives=primitives)
    indexed = {row["id"]: row for row in report["timeline"]}
    assert indexed["r0:exchange"]["complete_ms"] == 20
    assert indexed["r1:exchange"]["complete_ms"] == 4
    assert indexed["r1:after"]["complete_ms"] == 5
    assert report["duration_ms"] == 20 and sum(wire(report).values()) == 4
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_pipeline_placement_keeps_each_tensor_parallel_source_once():
    raw = case(dp=2, tp=2, pp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=32)
    raw["machine"]["links"][0].update(
        semantics="per_pair", capacity_bytes_per_ms=1)
    raw["runtime"]["kernels"] = {
        "families": {"alltoall": {"kind": "pairwise_collective"}}}
    transfer = exchange(3, family="alltoall", axis="ep", stage=1)
    report = graph_costs(raw, {0: (transfer,), 1: (transfer,)})
    cohort = report["isolated"]["collective_cohorts"][0]
    # Each of four stage-local sources sends one byte to each of three peers.
    assert cohort["rank_completion_ms"] == {"2": 3, "3": 3, "6": 3, "7": 3}
    assert report["duration_ms"] == 3 and sum(wire(report).values()) == 12
    assert [row["communication"]["wire_bytes"]
            for row in report["isolated"]["operators"]] == [6, 6]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def _case(peer_tokens, *, profile=None):
    raw = cohort_case(2)
    raw["machine"]["links"][0].update(capacity_bytes_per_ms=1e9)
    raw["workload"]["requests"] = [
        {"id": "early", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "peer", "prompt_tokens": peer_tokens, "output_tokens": 1},
    ]
    if profile is not None:
        raw["runtime"].setdefault("metadata", {})[
            "collective_participant_profile"] = profile
    return raw


def _first_dispatch(observations):
    return next(item for item in observations
                if item["op_id"] == "s0.l0.dispatch")


@pytest.mark.parametrize("dtype,resolved", [("float32", True), ("fp8_e4m3", False), (None, True)])
def test_participant_profile_checks_dtype_when_declared(dtype, resolved):
    profile = normalise({"schema": "mindie-sim/collective-participant-profile/1", "calls": [
        {"ordinal": 1, "primitive": "all_to_all", "native_name": "hcom_alltoallv__369_1_1",
         "group_size": 2, "dtype": "float32", "ranks": [{"rank_id": 9, "role": "primary"}]},
    ]})
    source = select_call(profile, primitive="all_to_all", group_size=2, dtype=dtype)
    assert source["resolved"] is resolved
    if not resolved:
        assert "source_dtype_mismatch" in source["reason"]


def test_padded_idle_keeps_physical_graph_without_a_logical_request():
    raw = cohort_case(2)
    raw["runtime"]["graph_mode"] = "full"
    raw["runtime"]["graph_capture_mode"] = "full"
    raw["runtime"]["graph_capture_keys"] = [1, 2, 4]
    out = body_forward(raw, phase="decode", tokens=1, context=4,
                       sequences=[1, 0], event_limit=64, compact=True,
                       padded_idle=True)
    assert out["geometry"]["valid_query_tokens_by_replica"] == {0: 1, 1: 0}
    assert out["geometry"]["graph_query_tokens_by_replica"] == {0: 1, 1: 1}
    assert out["profile"]["by_rank"][1]["operator_events"] > 0
    first = _first_dispatch(
        out["execution"]["joint_collective_observations"])
    assert min(first["ready_ms_by_rank"].values()) > 0


@pytest.mark.parametrize("pairwise", [True, False], ids=["pairwise", "aggregate"])
def test_rendezvous_retains_rank_readiness_and_profiles_preserve_service(pairwise):
    def run(tokens, limit, profile=None):
        raw = _case(tokens, profile=profile)
        if not pairwise:
            raw["runtime"]["kernels"] = {}
        return predict(raw, event_limit=limit, trace_limit=1)

    equal, baseline = run(4, 64), run(32, 64)
    if pairwise:
        raw = _case(4)
        graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4, scope="model"))
        assert wire(graph) == wire(equal) and graph["duration_ms"] > 0
    profile = {"schema": "mindie-sim/collective-participant-profile/1",
               "source": {"capture_id": "real-source"},
               "calls": [{"ordinal": 1, "primitive": "all_to_all", "group_size": 2,
                          "native_name": "hcom_alltoallv__369_1_1",
                          "ranks": [{"rank_id": 9, "role": "primary"},
                                    {"rank_id": 4, "role": "idle"}]}]}
    bounded = run(32, 3)["execution"]
    retained = bounded["joint_collective_observations"]
    assert bounded["joint_collective_stats"]["observations"] > len(retained) == 3
    json.dumps(retained)
    result = run(32, 64, profile)
    execution = result["execution"]
    first = _first_dispatch(execution["joint_collective_observations"])
    source = first["source"]
    assert source["resolved"] and source["native_name"] == profile["calls"][0]["native_name"]
    assert source["dtype_basis"] == "underdetermined_source"
    assert (execution["time_ms"], result["metrics"]) == (baseline["execution"]["time_ms"], baseline["metrics"])
    label = "joint_collective_observations"
    assert label not in run(32, 0)["execution"]
    reference = _first_dispatch(baseline["execution"][label])
    assert not reference["source"]["resolved"]
    ready = first["ready_ms_by_rank"]
    assert ready == reference["ready_ms_by_rank"]
    assert ready[0] == pytest.approx(_first_dispatch(equal["execution"][label])["ready_ms_by_rank"][0])
    assert first["critical_rank"] == 1
    assert ready[0] < ready[1] == pytest.approx(first["rendezvous_ready_ms"])
    assert baseline["requests"][0]["ttft_ms"] > equal["requests"][0]["ttft_ms"]
    if pairwise:
        completion = first["completion_ms_by_rank"]
        assert completion[0] == pytest.approx(completion[1])
    else:
        assert first["barrier_ms"] == pytest.approx(ready[1])
        left, right = sorted(first["participants"], key=lambda item: item["replica"])
        assert left["wait_ms"][0] > 0
        assert right["wait_ms"][1] == pytest.approx(0)
        assert left["parent_complete_ms"] == pytest.approx(right["parent_complete_ms"])

def test_native_launch_uses_local_readiness_and_explicit_peer_events():
    engine = Engine([], trace_limit=32)
    execution = object.__new__(Execution)
    execution.engine = engine
    local, barrier, peer = [engine.submit(name, release=when)
                            for name, when in (("local", 5), ("barrier", 10), ("peer", 20))]
    def start(engine, name, ready, peer_ready, **kwargs):
        send = engine.submit(name + ":send", after=(ready,))
        return engine.submit(name + ":receive", after=(send, peer_ready[1]))
    native = NativeProgram(SimpleNamespace(start=start), ())
    aggregate = KernelProgram((Stage("work", latency=1),))
    done = [execution.start_native_participant(program, name, barrier, local_ready=local, peer_ready={1: peer})
            for name, program in (("native", native), ("aggregate", aggregate))]
    engine.advance(until=tuple(done), first=False)
    assert [event.time for event in done] == [20, 11]
    assert next(row for row in engine.trace if row["task"] == "native:send")["end_ms"] == 5


class _NativeEndpoint:
    """Exercise cohort binding with explicit count and framed payload handles."""

    peer_count = 4
    view = Mc2LaunchView(rank=0, peer_labels=(1, 2, 3), node_ranks=2,
                         intra_latency_ms=0.002, inter_latency_ms=0.005)

    def __init__(self, *, active=True, payload=False):
        names = ("count_status_exchange", "payload_exchange") if payload else ("count_status_exchange",)
        self.phases = tuple(SimpleNamespace(name=name, after=()) for name in names)
        self.link_by_phase = {name: {"link": 1} for name in names} if active else {}

    def start_with_handles(self, engine, name, ready, *, launch, send_placeholders, **kwargs):
        frames = dict(launch.payload_frames)
        peer_events = {}
        phase_complete = {}
        for phase in self.phases:
            events = {}
            payload = phase.name == "payload_exchange"
            for peer in launch.peer_labels:
                edge = (launch.rank, peer, phase.name)
                reverse = (peer, launch.rank, phase.name)
                own = send_placeholders.get(edge)
                incoming = send_placeholders.get(reverse)
                if phase.name in self.link_by_phase and (not payload or peer in frames):
                    amount = frames[peer] * launch.payload_packet_bytes if payload else 1
                    send = engine.submit(f"{name}:{phase.name}:{peer}:send", work={"link": amount},
                                         after=(ready,))
                    index = launch.peer_labels.index(peer)
                    delay = launch.latency_for(launch.class_for(index, peer)) if payload else 0
                    events[peer] = {"send": send, "arrival": engine.submit(
                        f"{name}:{phase.name}:{peer}:arrival", latency=float(delay), after=(send,),
                        on_complete=lambda _, promise=own: engine.resolve(promise) if promise else None)}
                if incoming is not None:
                    events.setdefault(peer, {})["receive"] = engine.submit(
                        f"{name}:{phase.name}:{peer}:receive", after=(ready, incoming))
            peer_events[phase.name] = events
            phase_complete[phase.name] = engine.submit(f"{name}:{phase.name}:done", after=(
                ready, *(event for handles in events.values() for event in handles.values()),
            ))
        done = engine.submit(f"{name}:complete", after=tuple(phase_complete.values()))
        return done, {"peer_events": peer_events, "phase_complete": phase_complete}


@pytest.mark.parametrize("peer_delay,payload,idle", [(1, False, False), (6, False, False),
                                                     (6, False, True), (1, True, False)])
def test_native_cohort_binds_remote_arrival_and_retires_declared_edges(peer_delay, payload, idle):
    count = 4 if payload else 2
    inner = _NativeEndpoint(payload=payload)
    cached = NativeProgram(inner, ())
    original = inner.view
    engine = Engine([Resource("link", 1e6)], trace_limit=128)
    ready = ready_events(engine, [0, peer_delay, 0, 0][:count])
    launches = []
    for rank, event in ready.items():
        program = NativeProgram(_NativeEndpoint(active=False), ()) if rank == 1 and idle else cached
        row = {"program": program, "name": f"r{rank}", "ready": event, "rank_index": rank,
               "peer_count": count, "node_ranks": 2}
        if payload:
            row.update(payload_packet_bytes=6144, resource_names={"link_hccs": "link", "link_roce": "link"},
                       payload_frames=((1, 4), (2, 2)) if rank == 0 else (),
                       payload_basis=("actual" if rank == 0 else
                                      "unknown" if rank == 3 else "actual_zero"))
        launches.append(row)
    if idle:
        launches[0]["fullmesh_phases"] = ("count_status_exchange",)
    result = launch_native_cohort(engine, launches)
    engine.advance(until=(result["join"],))
    handles = result["handles"]
    phase = "payload_exchange" if payload else "count_status_exchange"
    sender = handles["r0"]["peer_events"][phase][1]
    assert sender["send"].time < peer_delay
    if idle:
        assert "receive" not in sender
        assert not handles["r1"]["peer_events"][phase].get(0, {}).get("send")
    else:
        receiver = handles["r1"]["peer_events"][phase][0]
        assert receiver["receive"].time == pytest.approx(max(peer_delay, sender["arrival"].time))
    if payload:
        assert result["directed_edges"] == 2
        assert result["directed_frames"] == 6
        assert result["payload_frames"]["r0"] == {1: 4, 2: 2}
        assert any(item.get("rank") == 3 for item in result["partial"])
        assert engine.serviced["link"] == 6 * 6144 + count * (count - 1)
    assert result["unresolved_edges"] == 0
    assert inner.view is original and not hasattr(inner, "names")
    assert not engine._promises and engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_native_service_keeps_reference_work_and_one_inclusive_boundary():
    raw = joint_case()
    op = copy_op(4000)
    raw["runtime"]["kernels"]["invocations"] = [
        {"match": {"recipe": op.recipe}, "startup_ms": 5, "minimum_ms": 9}]
    raw["workload"]["requests"][0]["arrival_ms"] = 7
    row = operator_costs(raw, (op,), input_ready_ms=7)["operators"][0]
    graph = graph_costs(raw, (op,), input_ready_ms=7)
    serving = predict(raw, lower=lower_copy)
    assert row["duration_ms"] == graph["duration_ms"] == 2
    assert serving["requests"][0]["finish_ms"] == graph["complete_ms"] == 9
    assert row["analytical_reference_work"]["hbm"] == graph["execution"]["work"]["hbm_bytes"] == 4000
    assert row["serviced"] == row["declared_service"] == {"observed_copy@r0:s0": 2e6}
    assert not row["service_delta"] and not row["service_before_input"]


def test_native_service_reports_measured_primitive_as_reference():
    raw, op = joint_case(), copy_op(4000)
    with PrimitiveStore(":memory:") as store:
        store.record(primitive_key(parse_case(raw), op, "none"), duration_ms=20,
                     run_id="reference", repeat_index=0, source="synthetic_control")
        row = operator_costs(raw, (op,), primitives=store)["operators"][0]
    assert row["duration_ms"] == 2
    assert row["joint_native_service"]["reference_primitive_ms"] == 20
    assert row["basis"]["operator_occurrences"] == {
        "measured_primitive": 0, "analytic_work": 0, "joint_native_service": 1}
    assert row["basis"]["joint_native_services"][0]["replaced_primitive_count"] == 1


def test_joint_queue_contends_across_streams_and_keeps_dependences_and_rate_scans():
    operators = tuple(replace(copy_op(4000), op_id=str(i), stream=str(i)) for i in range(3))
    for factor in (.5, 1, 2):
        raw = joint_case()
        raw["machine"]["device"]["rates"]["observed_copy"] *= factor
        result = graph_costs(raw, operators, input_ready_ms=7, isolated=False)
        assert result["duration_ms"] == pytest.approx(6 / factor)
        assert result["execution"]["serviced"]["observed_copy@r0:s0"] == 6e6
    last = replace(operators[2], deps=("0", "1"))
    result = graph_costs(joint_case(), (*operators[:2], last), input_ready_ms=7)
    assert result["duration_ms"] == 6


def test_unmatched_domain_still_executes_analytic_program_and_different_queue_overlaps():
    raw = joint_case()
    miss = graph_costs(raw, (copy_op(5000),), isolated=False)
    assert miss["duration_ms"] == 5
    assert "joint_native_services" not in miss["parameters"]
    device = raw["machine"]["device"]
    for field, value in (("rates", 1e9), ("work_units", "busy_ns"), ("resource_units", 1)):
        device[field]["other_copy"] = value
    rule = deepcopy(raw["runtime"]["kernels"]["service_regimes"][0])
    rule.update(id="other", match={"recipe": "second_copy"})
    rule["execution"]["resource"] = "other_copy"
    raw["runtime"]["kernels"]["service_regimes"].append(rule)
    a = replace(copy_op(4000), op_id="a", stream="a")
    b = replace(copy_op(4000), op_id="b", stream="b", recipe="second_copy")
    result = graph_costs(raw, (a, b), isolated=False)
    assert result["duration_ms"] == 2
    raw = joint_case()
    raw["machine"]["num_devices"] = raw["runtime"]["dp"] = 2
    parallel = graph_costs(raw, {0: (a,), 1: (a,)}, isolated=False)
    assert parallel["duration_ms"] == 2
    assert all(parallel["execution"]["serviced"][f"observed_copy@r{r}:s0"] == 2e6 for r in (0, 1))


def test_joint_service_cannot_hide_native_multiphase_or_network_program():
    raw = joint_case()
    rule = raw["runtime"]["kernels"]["service_regimes"][0]
    rule["match"] = {"kernel": "NativeDispatch"}
    rule.pop("coordinates")
    rule.pop("domain")
    with pytest.raises(ModelingError, match="single_local_kernel"):
        graph_costs(raw, fork(), isolated=False)
    rule["match"] = {"recipe": copy_op(4000).recipe}
    with pytest.raises(ModelingError, match="single_local_kernel"):
            operator_costs(raw, (replace(copy_op(4000), family="alltoall"),))


def test_oneshot_allreduce_replay_accounts_direct_peer_exchange():
    from mindie_sim.prediction.collectives import OneShotCollective, Topology
    from mindie_sim.prediction.execution.operator_replay import cohort_operator_costs
    from mindie_sim.prediction.program import RecipeBook
    from mindie_sim.schema.operators import OpSpec

    ranks = 8
    tensor_bytes = 1024 * 1024
    ring_link_bytes = float(2 * (ranks - 1) * tensor_bytes // ranks)
    raw_tp = case(tp=ranks)
    raw_tp["model"].update(num_heads=ranks, num_kv_heads=ranks)
    raw_tp["machine"]["links"] = [{
        "name": "ub_mesh", "members": list(range(ranks)),
        "semantics": "per_pair", "capacity_bytes_per_ms": 400_000_000, "latency_us": 1.0,
    }]
    op_tp = OpSpec(
        op_id="oneshot_ar_tp", family="allreduce", phase="decode", stage=0, layer=0,
        requests=("r",), semantic_shape=(1, 32), local_shape=(1, 32),
        padded_shape=(1, 32), launch_shape=(1, 32), link_bytes=ring_link_bytes,
        link_scope="tp", dtype="bf16", layout="contiguous", recipe="oneshot_ar",
        metadata=(("collective", "allreduce"), ("group_axis", "tp")),
    )
    book = RecipeBook(recipes={"oneshot_ar": OneShotCollective(path_model="direct")})
    row = operator_costs(raw_tp, (op_tp,), recipes=book)["operators"][0]
    comm = row["communication"]
    assert comm["useful_bytes"] == pytest.approx(ranks * (ranks - 1) * tensor_bytes)
    assert comm["messages"] == ranks * (ranks - 1)
    assert len(comm["resources"]) == ranks * (ranks - 1)
    assert row["service_delta"] == {}
    assert row["serviced"] == {name: item["wire_bytes"] for name, item in comm["resources"].items()}

    raw_ep = cohort_case(ranks)
    raw_ep["machine"]["links"][0].update(capacity_bytes_per_ms=400_000_000, latency_us=1.0)
    op_ep = replace(op_tp, op_id="oneshot_ar_ep", link_scope="ep",
                    metadata=(("collective", "allreduce"), ("group_axis", "ep")))
    cohort = cohort_operator_costs(
        parse_case(raw_ep), {r: op_ep for r in range(ranks)}, recipes=book,
    )
    assert len(cohort["operators"]) == ranks
    assert len(cohort["serviced"]) == ranks * (ranks - 1)
    for op_row in cohort["operators"]:
        assert op_row["communication"]["useful_bytes"] == pytest.approx((ranks - 1) * tensor_bytes)
        assert op_row["communication"]["messages"] == ranks - 1
        assert op_row["service_delta"] == {}

    parsed = parse_case(raw_tp)
    topo = Topology(parsed.machine, parsed.runtime.parallel)
    topo.joint_plan(op_tp, {0: ring_link_bytes}, algorithm="oneshot")
    assert sum(topo.accounting(op_tp, 0).useful_by_rank.values()) == pytest.approx(
        ranks * (ranks - 1) * tensor_bytes
    )
    topo.joint_plan(op_tp, {0: ring_link_bytes}, algorithm="ring")
    assert sum(topo.accounting(op_tp, 0).useful_by_rank.values()) == pytest.approx(
        ranks * ring_link_bytes
    )
