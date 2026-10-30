"""Independent finite-window and public cross-DP direct-peer controls."""
from dataclasses import replace
import heapq
import random

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.peer_traffic import PeerTraffic
from test_collective_cohorts import per_pair_topology, ready_events
from test_communication_topology import exchange
from test_joint_collective_execution import cohort_case, wire
from test_prediction_system import case


def independent_arrivals(matrix, ready, window, capacity=1, latency=.005):
    count = len(matrix)
    completed = list(ready)
    for src in range(count):
        slots = [ready[src]] * min(window, count - 1)
        heapq.heapify(slots)
        for offset in range(1, count):
            dst = (src + offset) % count
            payload = matrix[src][dst]
            if not payload:
                continue
            available = heapq.heappop(slots)
            end = max(available, ready[dst]) + payload / capacity + latency
            heapq.heappush(slots, end)
            completed[src] = max(completed[src], end)
            completed[dst] = max(completed[dst], end)
    return completed


@pytest.mark.parametrize("window", [1, 2, 7])
def test_nonuniform_peer_cells_and_reverse_match_independent_slot_recurrence(window):
    rng = random.Random(291031)
    for count in (2, 3, 7):
        matrix = tuple(tuple(0 if s == d else rng.randrange(8)
                             for d in range(count)) for s in range(count))
        traffic = PeerTraffic.from_mapping({"kind": "matrix", "payload_bytes": matrix}, count)
        for peer in (traffic, traffic.transposed()):
            cells = tuple(tuple(peer.payload(s, d) for d in range(count))
                          for s in range(count))
            initial = [float(rng.randrange(5)) for _ in range(count)]
            topo = per_pair_topology(count, latency_us=5)
            volumes = {s: peer.row_total(s) for s in range(count)}
            from mindie_sim.prediction.direct_peer import DirectPeerGroup
            group = DirectPeerGroup(tuple(range(count)), tuple(volumes.items()),
                                    count - 1, topo, peer, window)
            engine = Engine(topo.resources())
            events = group.start_participants(engine, "direct", ready_events(engine, initial))
            engine.advance(until=tuple(events.values()), first=False)
            expected = independent_arrivals(cells, initial, window)
            assert [events[r].time for r in range(count)] == pytest.approx(expected)
            assert sum(engine.serviced.values()) == pytest.approx(sum(map(sum, cells)))
            assert engine.summary()["live_tasks"] == 0


@pytest.mark.parametrize("semantics", ["shared", "per_port"])
@pytest.mark.parametrize("bandwidth_scale", [.5, 1, 2])
def test_window_removes_round_barriers_without_removing_link_contention(semantics, bandwidth_scale):
    raw = case(tp=8)
    raw["model"]["num_heads"] = 16
    raw["machine"]["links"][0].update(
        semantics=semantics, capacity_bytes_per_ms=1000 * bandwidth_scale, latency_us=5,
        transaction_bytes=16, protocol_overhead_bytes=4)
    raw["runtime"]["kernels"] = {"families": {"alltoall": {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 7}}}
    from mindie_sim import operator_costs
    row = operator_costs(raw, (exchange(70),), input_ready_ms=9)["operators"][0]
    # 8 ranks * 7 messages. Ten useful bytes become twenty wire bytes/message.
    contention = 8 if semantics == "shared" else 1
    assert row["duration_ms"] == pytest.approx(7 * 20 / (1000 * bandwidth_scale) * contention + .005)
    assert sum(row["serviced"].values()) == 8 * 7 * 20 * (2 if semantics == "per_port" else 1)
    assert row["communication"]["wire_bytes"] == 8 * 7 * 20
    assert not row["service_before_input"]
    assert row["communication"]["setup_ms"] == 0
    assert row["communication"]["unused_serial_round_estimate_ms"] == .035
    assert row["live_tasks"] == 0
    raw["runtime"]["kernels"]["families"]["alltoall"]["max_inflight_per_rank"] = 1
    serial = operator_costs(raw, (exchange(70),))["operators"][0]
    assert serial["duration_ms"] == pytest.approx(7 * (20 / (1000 * bandwidth_scale) * contention + .005))
    assert serial["serviced"] == row["serviced"]


def test_idle_sender_late_receiver_and_complete_cross_dp_entry():
    raw = cohort_case(3)
    raw["machine"]["links"][0]["latency_us"] = 5
    raw["runtime"]["kernels"] = {"families": {"alltoall": {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 2}}}
    from test_graph_analysis import op
    graphs = {rank: (op("producer", flops=2e6 if rank == 1 else 0),
                    replace(exchange(volume, axis="ep"), deps=("producer",)))
              for rank, volume in enumerate((0, 4, 8))}
    report = graph_costs(raw, graphs)
    indexed = {r["id"]: r for r in report["timeline"]}
    assert [indexed[f"r{r}:exchange"]["complete_ms"] for r in range(3)] == pytest.approx(
        [4.005, 6.005, 6.005])
    assert sum(wire(report).values()) == 12
    assert report["execution"]["live_tasks"] == 0
    rows = report["isolated"]["operators"]
    direct = [r for r in rows if r["program"] == "DirectPeerCollective"]
    assert [r["communication"]["wire_bytes"] for r in direct] == [0, 4, 8]


def test_multihop_store_forward_reframes_once_and_keeps_link_latency():
    raw = cohort_case(2)
    raw["machine"].update(num_devices=3, links=[
        {"name": "first", "members": [0, 2], "semantics": "per_pair",
         "capacity_bytes_per_ms": 1000, "transaction_bytes": 16,
         "protocol_overhead_bytes": 4, "latency_us": 2},
        {"name": "second", "members": [2, 1], "semantics": "per_pair",
         "capacity_bytes_per_ms": 1000, "transaction_bytes": 32,
         "protocol_overhead_bytes": 8, "latency_us": 3}])
    raw["runtime"]["kernels"] = {"families": {"alltoall": {
        "kind": "direct_peer_collective", "max_inflight_per_rank": 1,
        "path_model": "store_forward"}}}
    report = graph_costs(raw, {0: (exchange(17, axis="ep"),),
                               1: (exchange(0, axis="ep"),)})
    assert report["duration_ms"] == pytest.approx(.04 + .002 + .04 + .003)
    assert sum(wire(report).values()) == 80
    assert report["execution"]["live_tasks"] == 0


def test_ordinary_frontend_and_predict_use_explicit_recipe_without_duplicate_wire():
    raw = cohort_case(2)
    recipe = {"kind": "direct_peer_collective", "max_inflight_per_rank": 2}
    raw["runtime"]["kernels"] = {"recipes": {
        "expert_dispatch": recipe, "expert_combine": recipe}}
    raw["workload"]["requests"] = [
        {"id": "a", "prompt_tokens": 4, "output_tokens": 1},
        {"id": "b", "prompt_tokens": 4, "output_tokens": 1}]
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4))
    serving = predict(raw)
    assert serving["status"] == "modeled"
    assert wire(graph) == wire(serving)
    assert any(row["program"] == "DirectPeerCollective" for row in graph["timeline"])
    assert graph["execution"]["live_tasks"] == serving["execution"]["live_tasks"] == 0
