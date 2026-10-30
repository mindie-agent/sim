"""Ring causality checked against rank recurrences and public graph consumers."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.collectives import RingCollective
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from test_communication_topology import exchange
from test_prediction_system import case


def ring_case(*, delayed=True):
    raw = case(tp=4, prompt=4, output=1)
    raw["machine"]["links"] = [
        {"name": f"edge{rank}", "members": [rank, (rank + 1) % 4],
         "semantics": "per_pair", "capacity_bytes_per_ms": 1,
         "latency_us": 10000 if delayed and rank == 0 else 0}
        for rank in range(4)
    ]
    return raw


def independent_ring_arrivals(rounds, service, latency):
    """Serial source service and predecessor delivery, with no simulator code."""
    sent = [0.0] * len(service)
    received = [0.0] * len(service)
    for _ in range(rounds):
        sent = [max(sent[i], received[(i - 1) % len(service)]) + service[i]
                for i in range(len(service))]
        received = [sent[i] + latency[i] for i in range(len(service))]
    return received


@pytest.mark.parametrize("family,rounds,expected", (
    ("allgather", 3, 13), ("reduce_scatter", 3, 13), ("allreduce", 6, 26),
))
def test_slow_propagation_does_not_create_a_global_round_barrier(family, rounds, expected):
    raw = ring_case()
    op = exchange(rounds, family=family)
    recipes = RecipeBook(families={family: RingCollective()})
    report = graph_costs(raw, (op,), recipes=recipes, trace_limit=200)
    oracle = independent_ring_arrivals(rounds, [1] * 4, [10, 0, 0, 0])
    assert report["duration_ms"] == max(oracle) == expected
    assert sum(value for name, value in report["execution"]["serviced"].items()
               if name.startswith("link:")) == 4 * rounds
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    early = next(row for row in report["events"]
                 if ":rank2:round1:hop0:service" in row["task"])
    late = next(row for row in report["events"]
                if ":rank0:round0:hop0:arrival" in row["task"])
    assert early["start_ms"] == 1 < late["end_ms"] == 11
    # The previous aggregate latency estimate makes every rank pay every slow
    # edge delay before any bytes move, as if each round had a global barrier.
    aggregate = graph_costs(raw, (op,))
    assert aggregate["duration_ms"] == rounds * 11
    assert aggregate["execution"]["serviced"] == report["execution"]["serviced"]


def test_competing_collectives_share_links_without_becoming_one_round_barrier():
    raw = ring_case()
    first = exchange(3, family="allgather")
    second = replace(first, op_id="second", stream="other")
    report = graph_costs(raw, (first, second), recipes=RecipeBook(
        families={"allgather": RingCollective()}))
    # Symmetric calls each receive half of each link's 1B/ms capacity.
    assert report["duration_ms"] == max(
        independent_ring_arrivals(3, [2] * 4, [10, 0, 0, 0])) == 16
    assert sum(value for name, value in report["execution"]["serviced"].items()
               if name.startswith("link:")) == 24
    assert [row["complete_ms"] for row in report["timeline"]] == [16, 16]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


def test_round_framing_keeps_transaction_tail_and_latency_is_not_charged_twice():
    raw = case(tp=2)
    raw["machine"]["links"][0].update(
        semantics="per_pair", transaction_bytes=16, protocol_overhead_bytes=4,
        latency_us=1000)
    op = replace(exchange(34, family="allreduce"), deps=("upstream",))
    report = graph_costs(raw, (op,), recipes=RecipeBook(families={
        "allreduce": RingCollective()}), input_ready_ms=5, external_inputs={"upstream": 8},
        trace_limit=100)
    # Two 17B messages/rank each need two (16+4)B frames. Every hop adds 1ms
    # propagation after .04ms service. No aggregate 2ms setup is added again.
    assert report["duration_ms"] == pytest.approx(3 + 2 * 1.04)
    assert report["complete_ms"] == pytest.approx(10.08)
    assert sum(value for name, value in report["execution"]["serviced"].items()
               if name.startswith("link:")) == 160
    assert report["isolated"]["operators"][0]["communication"]["wire_bytes"] == 160
    assert all(row["start_ms"] >= 8 for row in report["events"] if row["demand"])


def test_multihop_whole_message_forwarding_is_explicit_and_conserves_hop_work():
    raw = case(tp=2)
    raw["machine"]["num_devices"] = 3
    raw["machine"]["links"] = [
        {"name": "a", "members": [0, 2], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 16, "protocol_overhead_bytes": 4, "latency_us": 200},
        {"name": "b", "members": [2, 1], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 32, "protocol_overhead_bytes": 8, "latency_us": 300},
    ]
    op = exchange(34, family="allreduce")
    with pytest.raises(ValueError, match="multihop_requires_store_forward"):
        graph_costs(raw, (op,), recipes=RecipeBook(families={"allreduce": RingCollective()}))
    report = graph_costs(raw, (op,), recipes=RecipeBook(families={
        "allreduce": RingCollective(path_model="store_forward")}), trace_limit=200)
    assert report["duration_ms"] == pytest.approx(2 * (.04 + .2 + .04 + .3))
    assert report["execution"]["serviced"]["link:a"] == 160
    assert report["execution"]["serviced"]["link:b"] == 160
    stages = {row["task"].split("rank0:round0:")[-1]: row for row in report["events"]
              if ":rank0:round0:" in row["task"]}
    assert stages["hop1:service"]["start_ms"] == stages["hop0:arrival"]["end_ms"]


def test_declared_collective_kind_uses_the_same_framing_for_reports_and_execution():
    raw = case(tp=2)
    raw["machine"]["links"][0].update(
        semantics="per_pair", transaction_bytes=16, protocol_overhead_bytes=4)
    op = exchange(34, family="transport", metadata=(("collective", "allreduce"),))
    report = graph_costs(raw, (op,), recipes=RecipeBook(
        families={"transport": RingCollective()}))
    communication = report["isolated"]["operators"][0]["communication"]
    # The generic transport name cannot turn two 17B frames per rank into one
    # 34B message in accounting while execution still runs two ring rounds.
    assert communication["messages"] == 4
    assert communication["wire_bytes"] == 160
    assert sum(report["execution"]["serviced"].values()) == 160


def test_frontend_ring_binding_reaches_graph_and_serving_through_public_configuration():
    raw = ring_case()
    raw["runtime"]["kernels"] = {"families": {
        family: {"kind": "ring_collective"} for family in ("allreduce", "allgather")
    }}
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    graph = graph_costs(raw, graphs)
    serving = predict(raw, event_limit=1000)
    wire = lambda report: {name: value for name, value in report["execution"]["serviced"].items()
                           if name.startswith("link:")}
    assert wire(graph) == wire(serving)
    assert any(row["program"] == "RingCollective" for row in graph["timeline"])
    assert any(":round1:" in row["task"] for row in serving["events"])
    assert serving["status"] == "modeled"
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0


def test_repeated_ring_steps_retain_active_messages_instead_of_step_history():
    parsed = parse_case(ring_case())
    topology = Topology(parsed.machine, parsed.runtime.parallel)
    group = topology.ring_plan(exchange(3, family="allgather"), 0).groups[0]
    peaks = []
    for repetitions in (100, 1000):
        engine = Engine(topology.resources())
        ready = engine.submit("ready")
        complete = replace(group, rounds=repetitions).start(engine, "loop", ready)
        engine.advance(until=(complete,))
        assert complete.time == max(independent_ring_arrivals(
            repetitions, [1] * 4, [10, 0, 0, 0]))
        peaks.append(engine.peak_tasks)
        assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    assert peaks[0] == peaks[1]


def test_cross_replica_ring_executes_complete_cohort_once():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    op = exchange(2, family="allreduce", axis="ep")
    report = graph_costs(raw, {0: (op,), 1: (op,)}, recipes=RecipeBook(
        families={"allreduce": RingCollective()}), trace_limit=100)
    assert report["duration_ms"] == pytest.approx(.004)
    assert report["execution"]["serviced"]["link:fabric"] == 4
    assert report["isolated"]["operator_sum_ms"] == pytest.approx(.008)
    assert sum(row["communication"]["wire_bytes"]
               for row in report["isolated"]["operators"]) == 4
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    with pytest.raises(ValueError, match="requires_complete_cohort"):
        graph_costs(raw, (op,), recipes=RecipeBook(
            families={"allreduce": RingCollective()}))
