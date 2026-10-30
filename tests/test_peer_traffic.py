"""Physical directed peer traffic, transpose and causal pairwise controls."""
from itertools import combinations

import pytest

from mindie_sim import graph_costs
from mindie_sim.prediction.collectives import PairwiseCollective
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.peer_traffic import PeerTraffic
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.common import ModelingError
from test_collective_cohorts import per_pair_topology, ready_events, topology
from test_communication_topology import exchange
from test_prediction_system import case


MATRIX = ((0, 0, 0), (2, 0, 2), (4, 4, 0))
TRANSPOSE = ((0, 2, 4), (0, 0, 4), (0, 2, 0))


def peer_case(*, dp=3, tp=1, pp=1, ep=3, devices=8):
    raw = case(dp=dp, tp=tp, pp=pp, ep=ep)
    raw["model"].update(ffn="moe", num_experts=ep * 2, top_k=1,
                        moe_intermediate_size=32)
    raw["machine"]["num_devices"] = devices
    raw["machine"]["links"] = [
        {"name": f"p{source}_{destination}",
         "members": [source, destination], "semantics": "per_pair",
         "capacity_bytes_per_ms": 1}
        for source, destination in combinations(range(devices), 2)
    ]
    return raw


def matrix_of(traffic):
    return tuple(tuple(traffic.payload(source, destination)
                       for destination in range(traffic.size))
                 for source in range(traffic.size))


def explicit_pairwise_arrivals(matrix, initial, *, capacity=1.0, latency=None):
    """Independent scalar recurrence over directed cells, not an engine helper."""
    count = len(matrix)
    rounds = count - 1
    initial = [float(value) for value in initial]
    last_send, last_receive = list(initial), list(initial)
    latency = latency or (lambda source, destination: 0.0)
    for round_index in range(rounds):
        next_send = [0.0] * count
        next_receive = list(last_receive)
        for source in range(count):
            destination = (source + round_index + 1) % count
            payload = float(matrix[source][destination])
            start = max(initial[source], last_send[source], last_receive[source])
            if payload:
                start = max(start, initial[destination])
            finish = start + (payload / capacity if payload else 0.0)
            next_send[source] = finish
            arrival = finish + (latency(source, destination) if payload else 0.0)
            next_receive[destination] = max(
                next_receive[destination], initial[destination], arrival)
        last_send, last_receive = next_send, next_receive
    return {rank: max(last_send[rank], last_receive[rank]) for rank in range(count)}


def directed_wire(matrix):
    result = {}
    for source in range(len(matrix)):
        for destination in range(len(matrix)):
            payload = matrix[source][destination]
            if payload:
                edge = f"p{min(source, destination)}_{max(source, destination)}"
                result[f"link:{edge}:pair{source}>{destination}"] = payload
    return result


def test_uniform_forward_and_return_transpose_keep_the_physical_counterexample():
    traffic = PeerTraffic.from_mapping(
        {"kind": "uniform_peers", "source_bytes": [0, 4, 8]}, 3)
    assert traffic.direction == "forward"
    assert matrix_of(traffic) == MATRIX
    assert [traffic.row_total(rank) for rank in range(3)] == [0, 4, 8]
    assert sum(traffic.payload(source, destination)
               for source in range(3) for destination in range(3)) == 12

    reverse = traffic.transposed()
    assert matrix_of(reverse) == TRANSPOSE
    assert [reverse.row_total(rank) for rank in range(3)] == [6, 4, 2]
    assert sum(reverse.payload(source, destination)
               for source in range(3) for destination in range(3)) == 12
    assert reverse.transposed() == traffic
    # The O(N) declaration is retained; no dense uniform matrix is materialized.
    assert reverse.kind == "uniform_peers" and len(reverse.values) == 3


def test_explicit_matrix_transpose_reads_reversed_cells_not_swapped_totals():
    traffic = PeerTraffic.from_mapping(
        {"kind": "matrix", "payload_bytes": MATRIX}, 3)
    assert matrix_of(traffic.transposed()) == TRANSPOSE
    assert [traffic.transposed().row_total(rank) for rank in range(3)] == [6, 4, 2]
    assert traffic.transposed().transposed() == traffic


def test_representative_group_mean_keeps_unequal_physical_rows_and_readiness():
    raw = peer_case(dp=1, tp=3, ep=1, devices=3)
    raw['model'].update(num_heads=6, num_kv_heads=3)
    topo = topology(raw)
    op = exchange(4, family="alltoallv", axis="tp", metadata=(
        ("peer_traffic", {"kind": "matrix", "payload_bytes": MATRIX}),
        ("peer_volume_basis", "group_mean")))
    accounting = topo.accounting(op, 0)
    assert accounting.useful_by_rank == {1: 4, 2: 8}
    assert accounting.wire_by_resource == directed_wire(MATRIX)
    plan = topo.joint_plan(op, {0: 4}, algorithm="pairwise")
    assert plan.groups[0].volumes == ((0, 0), (1, 4), (2, 8))
    engine = Engine(topo.resources())
    initial = [11, 0, 3]
    events = plan.start_participants(engine, "mean", ready_events(engine, initial))
    engine.advance(until=tuple(events.values()), first=False)
    assert {rank: event.time for rank, event in events.items()} == explicit_pairwise_arrivals(MATRIX, initial)
    assert {k:v for k,v in engine.serviced.items() if v} == directed_wire(MATRIX)
    assert engine.summary()['live_tasks'] == 0
    # The ordinary per-rank contract must still catch this scalar mismatch;
    # only a deliberate mean declaration permits unequal representative rows.
    from dataclasses import replace
    ordinary = replace(op, metadata=tuple((k,v) for k,v in op.metadata
                                         if k != "peer_volume_basis"))
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        topo.joint_plan(ordinary, {0: 4}, algorithm="pairwise")


def test_transpose_keeps_small_remote_flows_beside_a_large_local_source():
    traffic = PeerTraffic.from_mapping({
        "kind": "uniform_peers", "source_bytes": [1e20, 2, 4],
        "direction": "transpose",
    }, 3)
    # Subtracting the 1e20 local source from the sum would erase these flows.
    assert traffic.row_total(0) == 3
    assert traffic.row_total(0) == sum(
        traffic.payload(0, destination) for destination in range(3))


def test_singleton_explicit_zero_traffic_preserves_ready_without_transport():
    topo = per_pair_topology(1)
    op = exchange(0, metadata=(("peer_traffic", {
        "kind": "uniform_peers", "source_bytes": [0],
        "direction": "transpose",
    }),))
    work = topo.accounting(op, 0)
    assert work.messages == work.setup_ms == 0
    assert not work.wire_by_resource
    plan = topo.joint_plan(op, {0: 0}, algorithm="pairwise")
    engine = Engine(topo.resources())
    initial = ready_events(engine, [7])
    events = plan.start_participants(engine, "singleton", initial)
    engine.advance(until=tuple(events.values()), first=False)
    assert events[0] is initial[0] and events[0].time == 7
    assert not any(engine.serviced.values())
    assert engine.summary()["live_tasks"] == 0


def test_asymmetric_cells_match_the_independent_pairwise_recurrence():
    raw = peer_case(dp=3, tp=1, ep=3, devices=3)
    topo = topology(raw)
    declaration = {"kind": "matrix", "payload_bytes": MATRIX}
    op = exchange(0, family="alltoallv", axis="ep",
                  metadata=(("peer_traffic", declaration),))
    plan = topo.joint_plan(op, {0: 0, 1: 4, 2: 8}, algorithm="pairwise")
    assert plan.family == "alltoallv"
    assert [group.volumes for group in plan.groups] == [
        ((0, 0.0), (1, 4.0), (2, 8.0))]
    assert plan.groups[0].peer_traffic is not None

    initial = [11, 0, 3]
    engine = Engine(topo.resources(), trace_limit=200)
    events = plan.start_participants(
        engine, "asymmetric", ready_events(engine, initial))
    engine.advance(until=tuple(events.values()), first=False)

    oracle = explicit_pairwise_arrivals(MATRIX, initial)
    assert {rank: event.time for rank, event in events.items()} == pytest.approx(oracle)
    assert ({name: value for name, value in engine.serviced.items() if value}
            == pytest.approx(directed_wire(MATRIX)))
    assert sum(engine.serviced.values()) == 12
    assert events[0].time > 0
    # A zero row still receives, but creates no hop service or framing work.
    assert not [row for row in engine.trace
                if ":rank0:round" in row["task"] and ":hop" in row["task"]]
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


@pytest.mark.parametrize("direction,matrix,totals,rank_times", (
    ("forward", MATRIX, (0, 4, 8), {"0": 4, "1": 8, "2": 8}),
    ("transpose", TRANSPOSE, (6, 4, 2), {"0": 6, "1": 6, "2": 6}),
))
def test_public_graph_and_isolated_rows_charge_actual_directed_cells(
        direction, matrix, totals, rank_times):
    raw = peer_case(dp=3, tp=1, ep=3, devices=3)
    declaration = {"kind": "matrix", "payload_bytes": MATRIX,
                   "direction": direction}
    graphs = {
        replica: (exchange(volume, family="alltoall", axis="ep",
                           metadata=(("peer_traffic", declaration),)),)
        for replica, volume in enumerate(totals)
    }
    report = graph_costs(
        raw, graphs, recipes=RecipeBook(families={"alltoall": PairwiseCollective()}))
    actual = {name: value for name, value in report["execution"]["serviced"].items()
              if name.startswith("link:") and value}
    assert actual == pytest.approx(directed_wire(matrix))
    assert sum(actual.values()) == 12
    assert report["duration_ms"] == max(rank_times.values())
    assert report["bounds"]["combined_ms"] <= report["duration_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0

    cohort = report["isolated"]["collective_cohorts"][0]
    assert cohort["duration_ms"] == max(rank_times.values())
    assert cohort["rank_completion_ms"] == rank_times
    rows = report["isolated"]["operators"]
    assert [row["communication"]["useful_bytes_by_rank"] for row in rows] == [
        {str(rank): float(amount)} if amount else {}
        for rank, amount in enumerate(totals)]
    assert [row["communication"]["wire_bytes"] for row in rows] == list(totals)
    assert all(not row["service_delta"] for row in rows)
    assert all(row["live_tasks"] == 0 for row in rows)


def test_framing_is_charged_per_nonzero_directed_cell():
    topo = per_pair_topology(
        3, capacity=1000, framing={"transaction_bytes": 16,
                                   "protocol_overhead_bytes": 4})
    matrix = ((0, 17, 1), (18, 0, 0), (1, 17, 0))
    declaration = {"kind": "matrix", "payload_bytes": matrix}
    op = exchange(18, family="alltoall", axis="tp",
                  metadata=(("peer_traffic", declaration),))
    work = topo.accounting(op, 0)
    # Five actual cells, not six uniform splits: 17 and 18 each need two
    # 16-byte transactions; 1 needs one. No zero cell is framed.
    assert work.messages == 5
    assert work.useful_by_rank == {0: 18, 1: 18, 2: 18}
    assert work.to_dict()["wire_bytes"] == 160
    assert sum(work.transactions_by_resource.values()) == 8

    plan = topo.joint_plan(op, {0: 18}, algorithm="pairwise")
    engine = Engine(topo.resources())
    events = plan.start_participants(
        engine, "framed_cells", ready_events(engine, [0, 0, 0]))
    engine.advance(until=tuple(events.values()), first=False)
    assert sum(engine.serviced.values()) == 160
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0


def test_tp2_ep4_stage1_uses_physical_group_relative_rows_and_scalars():
    raw = peer_case(dp=2, tp=2, pp=2, ep=4, devices=8)
    topo = topology(raw)
    matrix = ((0, 1, 2, 0),
              (0, 0, 0, 3),
              (2, 2, 0, 3),
              (2, 1, 4, 0))
    declaration = {"kind": "matrix", "payload_bytes": matrix}
    op = exchange(0, family="alltoallv", axis="ep", stage=1,
                  metadata=(("peer_traffic", declaration),))
    plan = topo.joint_plan(op, {0: 3, 1: 7}, algorithm="pairwise")
    group = plan.groups[0]
    assert group.ranks == (2, 3, 6, 7)
    assert group.volumes == ((2, 3.0), (3, 3.0), (6, 7.0), (7, 7.0))

    local = {
        0: (topo.accounting(op, 0, volume=3), {2, 3}, 6),
        1: (topo.accounting(op, 1, volume=7), {6, 7}, 14),
    }
    for replica, (work, ranks, total) in local.items():
        assert set(work.useful_by_rank) == ranks
        assert set(work.useful_by_rank.values()) == ({3.0} if replica == 0 else {7.0})
        assert work.to_dict()["useful_bytes"] == total
        for name in work.wire_by_resource:
            source, destination = map(int, name.split(":pair")[1].split(">"))
            assert source in group.ranks and destination in group.ranks
    assert sum(row[0].to_dict()["useful_bytes"] for row in local.values()) == 20


def test_one_scalar_per_replica_rejects_unequal_local_source_rows():
    topo = per_pair_topology(3)
    declaration = {"kind": "matrix", "payload_bytes": MATRIX}
    op = exchange(12, family="alltoall", axis="tp",
                  metadata=(("peer_traffic", declaration),))
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        topo.accounting(op, 0)
    with pytest.raises(ModelingError, match="peer_traffic_source_volume_mismatch"):
        topo.joint_plan(op, {0: 12}, algorithm="pairwise")
    with pytest.raises(ModelingError, match="explicit_peer_matrix"):
        topo.joint_plan(exchange(2, family="alltoallv"), {0: 2},
                        algorithm="pairwise")
    with pytest.raises(ModelingError, match="explicit_peer_matrix"):
        topo.accounting(exchange(2, family="alltoallv"), 0)


def test_direct_path_validation_checks_only_actual_positive_cells():
    raw = case(tp=3)
    raw["model"].update(num_heads=3, num_kv_heads=1)
    raw["machine"]["links"] = [
        {"name": "a", "members": [0, 2], "capacity_bytes_per_ms": 1},
        {"name": "b", "members": [1, 2], "capacity_bytes_per_ms": 1},
    ]
    topo = topology(raw)
    assert len(topo.route(0, 1)) == 2
    direct = ((0, 0, 5), (0, 0, 5), (5, 0, 0))
    plan = topo.joint_plan(
        exchange(5, family="alltoall", axis="tp",
                 metadata=(("peer_traffic", {"kind": "matrix",
                                             "payload_bytes": direct}),)),
        {0: 5}, algorithm="pairwise")
    plan.validate_path_model("direct")

    unsupported = ((0, 5, 0), (0, 0, 5), (5, 0, 0))
    bad = topo.joint_plan(
        exchange(5, family="alltoall", axis="tp",
                 metadata=(("peer_traffic", {"kind": "matrix",
                                             "payload_bytes": unsupported}),)),
        {0: 5}, algorithm="pairwise")
    with pytest.raises(ModelingError, match="multihop_requires_store_forward"):
        bad.validate_path_model("direct")


def test_independent_pcp_groups_repeat_the_descriptor_without_cross_traffic():
    raw = case(tp=2)
    raw["runtime"]["pcp"] = 2
    raw["machine"]["num_devices"] = 4
    raw["model"].update(num_heads=2, num_kv_heads=1)
    raw["machine"]["links"] = [
        {"name": f"p{source}_{destination}",
         "members": [source, destination], "semantics": "per_pair",
         "capacity_bytes_per_ms": 1}
        for source, destination in combinations(range(4), 2)
    ]
    topo = topology(raw)
    assert topo.groups("pcp", 0, 0) == ((0, 2), (1, 3))
    op = exchange(4, family="alltoall", axis="pcp",
                  metadata=(("peer_traffic", {"kind": "uniform_peers",
                                              "source_bytes": [4, 4]}),))
    work = topo.accounting(op, 0)
    assert work.messages == 4
    assert work.useful_by_rank == {0: 4, 1: 4, 2: 4, 3: 4}
    for name in work.wire_by_resource:
        source, destination = map(int, name.split(":pair")[1].split(">"))
        assert (source, destination) in {(0, 2), (2, 0), (1, 3), (3, 1)}
