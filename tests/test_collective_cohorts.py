"""Independent recurrences for joint ring and balanced pairwise transport."""
from itertools import combinations
import math
import random

import pytest

from mindie_sim.prediction.collectives import PairwiseCollective
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.program import KernelProgram, RecipeContext
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from test_communication_topology import exchange
from test_prediction_system import case


def topology(raw):
    parsed = parse_case(raw)
    return Topology(parsed.machine, parsed.runtime.parallel)


def per_pair_topology(ranks, *, capacity=1, latency_us=0, framing=None):
    """A complete direct graph so every ordered pair owns one directed link."""
    raw = case(tp=ranks)
    raw["model"].update(num_heads=2 * ranks, num_kv_heads=2)
    raw["machine"]["links"] = [
        {"name": f"p{source}{destination}", "members": [source, destination],
         "semantics": "per_pair", "capacity_bytes_per_ms": capacity,
         "latency_us": latency_us, **(framing or {})}
        for source, destination in combinations(range(ranks), 2)
    ]
    return topology(raw)


def ready_events(engine, delays):
    return {rank: engine.submit(f"ready{rank}", latency=delay)
            for rank, delay in enumerate(delays)}


def pairwise_arrivals(volumes, initial, *, capacity=1.0, latency=None):
    """Scalar schedule: contiguous receive order, lazy N-1 pairwise rounds."""
    count = len(volumes)
    if count <= 1:
        return {0: float(initial[0])} if count else {}
    rounds = count - 1
    latency = latency or (lambda source, destination: 0.0)
    send_service = [[0.0] * rounds for _ in range(count)]
    arrivals = [[0.0] * rounds for _ in range(count)]
    processed = [[0.0] * rounds for _ in range(count)]
    for round_index in range(rounds):
        if round_index:
            for rank in range(count):
                processed[rank][round_index - 1] = max(
                    arrivals[rank][round_index - 1],
                    processed[rank][round_index - 2] if round_index >= 2 else 0.0)
        for source in range(count):
            destination = (source + round_index + 1) % count
            payload = volumes[source] / rounds
            start = initial[source] if not payload else max(
                initial[source], initial[destination])
            if round_index:
                start = max(start, send_service[source][round_index - 1],
                            processed[source][round_index - 1])
            service = start + payload / capacity if payload else start
            send_service[source][round_index] = service
            arrival = max(service, initial[destination])
            if payload:
                arrival += latency(source, destination)
            arrivals[destination][round_index] = arrival
    for rank in range(count):
        processed[rank][-1] = max(
            arrivals[rank][-1],
            processed[rank][-2] if rounds >= 2 else 0.0)
    return {rank: max(send_service[rank][-1], processed[rank][-1])
            for rank in range(count)}


def ring_arrivals(rounds, initial, service, latency):
    """Scalar predecessor recurrence with per-rank initial readiness."""
    count = len(initial)
    sent = [None] * count
    received = [None] * count
    for round_index in range(rounds):
        sent = [
            (max(initial[index], initial[(index + 1) % count])
             if round_index == 0 else max(sent[index],
                                          received[(index - 1) % count]))
            + service[index]
            for index in range(count)
        ]
        received = [sent[index] + latency[index] for index in range(count)]
    return {index: max(sent[index], received[(index - 1) % count])
            for index in range(count)}


def test_pairwise_delayed_rank_does_not_become_a_global_round_barrier():
    topo = per_pair_topology(4)
    plan = topo.joint_plan(exchange(3, family="alltoall"), {0: 3},
                           algorithm="pairwise")
    engine = Engine(topo.resources(), trace_limit=100)
    ready = ready_events(engine, [100, 0, 0, 0])
    events = plan.start_participants(engine, "delayed", ready)
    engine.advance(until=tuple(events.values()), first=False)

    oracle = pairwise_arrivals([3] * 4, [100, 0, 0, 0])
    assert {rank: event.time for rank, event in events.items()} == pytest.approx(oracle)
    # Rank 2 sends round 1 after its own round-0 service and receive. It does
    # not wait for delayed rank 0's round-0 arrival or a collective-wide phase.
    early = next(row for row in engine.trace
                 if ":rank2:round1:hop0:service" in row["task"])
    late = next(row for row in engine.trace
                if ":rank0:round0:hop0:arrival" in row["task"])
    assert early["start_ms"] == 100 < late["end_ms"] == 101
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_pairwise_zero_sender_still_receives_and_participates_logically():
    raw = case(dp=3, tp=1, ep=3)
    raw["model"].update(ffn="moe", num_experts=3, top_k=1, moe_intermediate_size=32)
    raw["machine"]["links"] = [
        {"name": f"p{source}{destination}", "members": [source, destination],
         "semantics": "per_pair", "capacity_bytes_per_ms": 1}
        for source, destination in combinations(range(3), 2)
    ]
    topo = topology(raw)
    plan = topo.joint_plan(exchange(3, family="alltoall", axis="ep"),
                           {0: 0, 1: 3, 2: 3}, algorithm="pairwise")
    assert [group.volumes for group in plan.groups] == [
        ((0, 0.0), (1, 3.0), (2, 3.0))]
    engine = Engine(topo.resources(), trace_limit=100)
    ready = ready_events(engine, [0, 0, 0])
    events = plan.start_participants(engine, "zero_source", ready)
    engine.advance(until=tuple(events.values()), first=False)

    oracle = pairwise_arrivals([0, 3, 3], [0, 0, 0])
    assert {rank: event.time for rank, event in events.items()} == pytest.approx(oracle)
    assert sum(engine.serviced.values()) == 6
    assert events[0].time > 0
    assert not [row for row in engine.trace
                if ":rank0:round" in row["task"] and ":hop" in row["task"]]


def test_singleton_pairwise_group_has_no_transfer_rounds():
    topo = per_pair_topology(1)
    plan = topo.joint_plan(exchange(4, family="alltoall"), {0: 4},
                           algorithm="pairwise")
    assert plan.ranks == (0,)
    assert plan.groups[0].rounds == 0
    engine = Engine(topo.resources())
    ready = engine.submit("only_ready")
    events = plan.start_participants(engine, "singleton", {0: ready})
    engine.advance(until=tuple(events.values()), first=False)
    assert events[0] is ready
    assert not any(engine.serviced.values())


@pytest.mark.parametrize("algorithm,family", (
    ("ring", "allgather"), ("pairwise", "alltoall"),
))
def test_allzero_group_completes_at_ready_without_wire_or_latency(algorithm, family):
    topo = per_pair_topology(4)
    plan = topo.joint_plan(exchange(4, family=family), {0: 0}, algorithm=algorithm)
    engine = Engine(topo.resources(), trace_limit=100)
    ready = ready_events(engine, [7, 7, 7, 7])
    events = plan.start_participants(engine, "zero", ready)
    engine.advance(until=tuple(events.values()), first=False)

    assert {rank: event.time for rank, event in events.items()} == {
        rank: 7 for rank in range(4)}
    assert not any(engine.serviced.values())
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_pairwise_framing_is_charged_once_per_source_destination_message():
    topo = per_pair_topology(
        4, capacity=1000, framing={"transaction_bytes": 16,
                                    "protocol_overhead_bytes": 4})
    plan = topo.joint_plan(exchange(4, family="alltoall"), {0: 4},
                           algorithm="pairwise")
    engine = Engine(topo.resources())
    ready = ready_events(engine, [0, 0, 0, 0])
    events = plan.start_participants(engine, "frames", ready)
    engine.advance(until=tuple(events.values()), first=False)

    # Four useful bytes split over three peers, so every 4/3-byte message
    # needs one 16-byte transaction plus its four-byte header.
    assert len(plan.groups) == 1
    for source in range(4):
        for destination in range(4):
            if source == destination:
                continue
            name = f"link:p{min(source, destination)}{max(source, destination)}"
            resource = topo.resource(topo.links[name[5:]], source, destination)
            assert engine.serviced[resource] == 20
    assert sum(engine.serviced.values()) == 12 * 20


def test_pairwise_per_pair_capacity_does_not_contend_like_a_shared_link():
    shared_raw = case(tp=4)
    shared_raw["machine"]["links"][0].update(
        capacity_bytes_per_ms=1, semantics="shared")
    shared = topology(shared_raw)
    private = per_pair_topology(4, capacity=1)

    def complete(topo):
        plan = topo.joint_plan(exchange(3, family="alltoall"), {0: 3},
                               algorithm="pairwise")
        engine = Engine(topo.resources())
        ready = ready_events(engine, [0, 0, 0, 0])
        events = plan.start_participants(engine, "capacity", ready)
        engine.advance(until=tuple(events.values()), first=False)
        return {rank: event.time for rank, event in events.items()}, sum(
            engine.serviced.values())

    private_times, private_work = complete(private)
    shared_times, shared_work = complete(shared)
    assert private_times == pytest.approx({rank: 3 for rank in range(4)})
    assert shared_times == pytest.approx({rank: 12 for rank in range(4)})
    assert private_work == shared_work == 12


def test_pairwise_multihop_waits_each_arrival_and_rejects_direct_declaration():
    raw = case(tp=2)
    raw["machine"]["num_devices"] = 3
    raw["machine"]["links"] = [
        {"name": "a", "members": [0, 2], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 16, "protocol_overhead_bytes": 4, "latency_us": 200},
        {"name": "b", "members": [2, 1], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 32, "protocol_overhead_bytes": 8, "latency_us": 300},
    ]
    topo = topology(raw)
    op = exchange(2, family="alltoall")
    plan = topo.joint_plan(op, {0: 2}, algorithm="pairwise")
    with pytest.raises(ModelingError, match="multihop_requires_store_forward"):
        plan.validate_path_model("direct")
    with pytest.raises(ModelingError, match="multihop_requires_store_forward"):
        PairwiseCollective().lower(RecipeContext(
            op, {}, {}, {}, 1, native_work={"link": 4},
            collective_plan=lambda: plan))
    plan.validate_path_model("store_forward")

    engine = Engine(topo.resources(), trace_limit=200)
    ready = ready_events(engine, [0, 0])
    events = plan.start_participants(engine, "multi", ready)
    engine.advance(until=tuple(events.values()), first=False)
    # Each direction is one 20B frame on a then one 40B frame on b, with
    # both propagation delays paid in sequence.
    assert {rank: event.time for rank, event in events.items()} == pytest.approx(
        {0: .02 + .2 + .04 + .3, 1: .04 + .3 + .02 + .2})
    stages = {row["task"].split(":rank0:round0:")[-1]: row
              for row in engine.trace if ":rank0:round0:" in row["task"]}
    assert stages["hop1:service"]["start_ms"] == stages["hop0:arrival"]["end_ms"]
    assert engine.serviced["link:a"] == 40
    assert engine.serviced["link:b"] == 80

    program = PairwiseCollective(path_model="store_forward").lower(RecipeContext(
        op, {}, {}, {}, 1, setup_ms=999, native_work={"link": 4},
        collective_plan=lambda: plan))
    lowered = Engine(topo.resources())
    ready = lowered.submit("lower_ready")
    complete = program.start(lowered, "lower_multi", ready)
    lowered.advance(until=(complete,))
    assert complete.time == pytest.approx(.56)


def test_cross_dp_ring_joint_plan_accounts_complete_groups_once():
    raw = case(dp=4, tp=2, ep=4)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    topo = topology(raw)
    op = exchange(12, family="allreduce", axis="ep")
    plan = topo.joint_plan(op, {0: 12, 1: 12, 2: 20, 3: 20}, algorithm="ring")

    assert plan.family == "allreduce"
    assert plan.ranks == tuple(range(8))
    assert [group.ranks for group in plan.groups] == [
        (0, 1, 2, 3), (4, 5, 6, 7)]
    assert [group.rounds for group in plan.groups] == [6, 6]
    assert [message.flow.payload_bytes for group in plan.groups
            for message in group.messages] == [2, 2, 2, 2,
                                               20 / 6, 20 / 6, 20 / 6, 20 / 6]
    with pytest.raises(ModelingError, match="equal_per_rank_volume"):
        topo.joint_plan(op, {0: 12, 1: 16, 2: 20, 3: 20}, algorithm="ring")
    with pytest.raises(ModelingError, match="requires_complete_local_group"):
        topo.ring_plan(op, 0)

    engine = Engine(topo.resources())
    ready = ready_events(engine, [0] * 8)
    events = plan.start_participants(engine, "cross", ready)
    engine.advance(until=tuple(events.values()), first=False)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_local_pairwise_declaration_uses_the_same_causal_schedule():
    topo = per_pair_topology(4)
    plan = topo.pairwise_plan(exchange(3, family="alltoall"), 0)
    assert plan.family == "alltoall"
    assert plan.ranks == (0, 1, 2, 3)
    assert [group.rounds for group in plan.groups] == [3]
    engine = Engine(topo.resources())
    ready = engine.submit("local_ready")
    complete = plan.start(engine, "local_pairwise", ready)
    engine.advance(until=(complete,))
    assert complete.time == pytest.approx(3)


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


def test_ring_per_rank_ready_and_completion_keep_predecessor_causality():
    topo = per_pair_topology(4)
    group = topo.ring_plan(exchange(3, family="allgather"), 0).groups[0]
    engine = Engine(topo.resources(), trace_limit=200)
    ready = ready_events(engine, [100, 0, 0, 0])
    events = group.start_participants(engine, "ringready", ready)
    engine.advance(until=tuple(events.values()), first=False)
    oracle = ring_arrivals(3, [100, 0, 0, 0], [1, 1, 1, 1], [0, 0, 0, 0])
    assert {rank: event.time for rank, event in events.items()} == pytest.approx(oracle)
    early = next(row for row in engine.trace
                 if ":rank2:round1:hop0:service" in row["task"])
    late = next(row for row in engine.trace
                if ":rank0:round0:hop0:arrival" in row["task"])
    assert early["start_ms"] < late["end_ms"]


def test_pairwise_ready_coverage_is_checked_before_submitting_work():
    topo = per_pair_topology(3)
    plan = topo.joint_plan(exchange(3, family="alltoall"), {0: 3},
                           algorithm="pairwise")
    engine = Engine(topo.resources())
    ready = ready_events(engine, [0, 0, 0])
    engine.advance(until=tuple(ready.values()), first=False)
    completed = engine.summary()["completed_tasks"]
    with pytest.raises(ModelingError, match="ready_coverage_mismatch"):
        plan.start_participants(engine, "coverage",
                                {0: ready[0], 1: ready[1], 9: ready[2]})
    assert engine.summary()["completed_tasks"] == completed
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_pairwise_lower_keeps_real_minimum_and_never_adds_aggregate_setup():
    topo = per_pair_topology(3)
    op = exchange(3, family="alltoall")
    plan = topo.joint_plan(op, {0: 3}, algorithm="pairwise")
    context = RecipeContext(
        op, {}, {}, {}, 1, minimum_ms=3.5, setup_ms=999,
        native_work={"link": 6}, collective_plan=lambda: plan,
    )
    program = PairwiseCollective().lower(context)
    assert isinstance(program, KernelProgram)
    assert program.minimum_ms == 3.5
    assert [stage.name for stage in program.stages] == ["pairwise"]
    assert not program.stages[0].work
    engine = Engine(topo.resources())
    ready = engine.submit("ready")
    complete = program.start(engine, "lower", ready)
    engine.advance(until=(complete,))
    assert complete.time == pytest.approx(3.5)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    with pytest.raises(ModelingError, match="transport_only_work"):
        PairwiseCollective().lower(RecipeContext(
            op, {}, {}, {}, 1, native_work={"link": 6, "vector": 1},
            collective_plan=lambda: plan))
    with pytest.raises(ModelingError, match="path_model_unsupported"):
        PairwiseCollective(path_model="cut_through").lower(context)


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


def test_joint_pairwise_zero_local_work_keeps_nonzero_peer_receives():
    raw = case(dp=3, tp=1, ep=3)
    raw["model"].update(ffn="moe", num_experts=3, top_k=1,
                        moe_intermediate_size=32)
    raw["machine"]["links"][0].update(
        semantics="per_pair", capacity_bytes_per_ms=1)
    topo = topology(raw)
    op = exchange(0, family="alltoall", axis="ep")
    plan = topo.joint_plan(op, {0: 0, 1: 3, 2: 3}, algorithm="pairwise")
    program = PairwiseCollective().lower(RecipeContext(
        op, {}, {}, {}, 1, native_work={}, collective_plan=lambda: plan))
    engine = Engine(topo.resources())
    ready = engine.submit("entry")
    complete = program.start(engine, "idle_sender", ready)
    engine.advance(until=(complete,))

    assert complete.time == max(pairwise_arrivals([0, 3, 3], [0, 0, 0]).values())
    assert sum(engine.serviced.values()) == 6
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


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
