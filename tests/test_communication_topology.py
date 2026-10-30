"""Independent byte/ownership oracles for physical communication declarations."""
import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.operators import OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from test_prediction_system import case


def exchange(volume, *, family="alltoall", axis="tp", stage=0, metadata=()):
    return OpSpec("exchange", family, "prefill", stage, 0, (), (1,), (1,), (1,), (1,),
                  link_bytes=volume, metadata=(("group_axis", axis),) + metadata)


def topology(raw):
    parsed = parse_case(raw)
    return Topology(parsed.machine, parsed.runtime.parallel)


def test_alltoall_frames_each_peer_message_before_aggregation():
    raw = case(tp=4)
    raw["machine"]["links"][0].update(transaction_bytes=16, protocol_overhead_bytes=4)
    work = topology(raw).accounting(exchange(30), 0)
    # Four senders each send ten useful bytes to three peers. Each of those
    # twelve messages occupies one 16-byte transaction and a four-byte header.
    assert work.to_dict()["useful_bytes"] == 120
    assert work.to_dict()["wire_bytes"] == 12 * (16 + 4)
    assert work.messages == 12
    assert work.transactions_by_resource == {"link:fabric": 12}
    assert work.useful_by_rank == {0: 30, 1: 30, 2: 30, 3: 30}
    row = operator_costs(raw, (exchange(30),))["operators"][0]
    assert row["duration_ms"] == pytest.approx(240 / 1000)
    assert row["serviced"] == {"link:fabric": 240}
    assert row["operator"]["link_bytes"] == 30
    assert row["communication"] == work.to_dict()


def test_ring_rounds_keep_their_own_transaction_tails():
    raw = case(tp=4)
    raw["machine"]["links"][0].update(transaction_bytes=16, protocol_overhead_bytes=4,
                                     latency_us=2)
    work = topology(raw).accounting(exchange(42, family="allreduce"), 0)
    # A four-rank ring allreduce has six transmissions per rank. Each carries
    # 7 useful bytes; rounding the aggregate 42 bytes would lose three frames.
    assert work.messages == 4 * 6
    assert work.useful_by_resource == {"link:fabric": 4 * 42}
    assert work.wire_by_resource == {"link:fabric": 4 * 6 * 20}
    assert work.setup_ms == pytest.approx(6 * .002)
    report = graph_costs(raw, (exchange(42, family="allreduce"),))
    assert report["duration_ms"] == pytest.approx(.480 + .012)
    assert report["isolated"]["operators"][0]["communication"]["messages"] == 24


def test_undeclared_granule_keeps_fractional_analytic_routing_volume():
    raw = case(tp=4)
    work = topology(raw).accounting(exchange(10), 0)
    assert work.useful_by_rank == pytest.approx({rank: 10 for rank in range(4)})
    assert work.wire_by_resource == pytest.approx({"link:fabric": 40})
    assert work.transactions_by_resource == {"link:fabric": None}
    # The default still prices the declared analytic work, with no calibration.
    assert operator_costs(raw, (exchange(10),))["operator_sum_ms"] == pytest.approx(.04)


def test_alignment_and_framing_reapply_to_useful_payload_at_each_hop():
    raw = case(pp=2)
    raw["machine"]["num_devices"] = 3
    raw["machine"]["links"] = [
        {"name": "first", "members": [0, 2], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 16, "protocol_overhead_bytes": 4, "latency_us": 2},
        {"name": "second", "members": [2, 1], "capacity_bytes_per_ms": 1000,
         "transaction_bytes": 32, "alignment_bytes": 32,
         "protocol_overhead_bytes": 8, "latency_us": 3},
    ]
    parsed = parse_case(raw)
    restored = Hardware.from_mapping(parsed.machine.to_dict())
    assert restored.links == parsed.machine.links
    work = topology(raw).accounting(exchange(17, family="pp_p2p", axis="pp"), 0)
    assert work.useful_by_rank == {0: 17}
    assert work.useful_by_resource == {"link:first": 17, "link:second": 17}
    assert work.wire_by_resource == {"link:first": 40, "link:second": 40}
    assert work.transactions_by_resource == {"link:first": 2, "link:second": 1}
    assert work.transaction_frames == {"link:first": (16, 20), "link:second": (32, 40)}
    assert work.to_dict()["routed_useful_bytes"] == 34
    assert work.setup_ms == pytest.approx(.005)

    # A third transaction on the first link does not inflate downstream payload.
    larger = topology(raw).accounting(exchange(33, family="pp_p2p", axis="pp"), 0)
    assert larger.wire_by_resource == {"link:first": 60, "link:second": 80}


def test_directed_pair_capacity_and_port_capacity_have_distinct_scaling():
    raw = case(tp=4)
    results = {}
    for semantics in ("shared", "per_port", "per_pair"):
        raw["machine"]["links"][0]["semantics"] = semantics
        report = graph_costs(raw, (exchange(30),))
        results[semantics] = report["duration_ms"]
        assert sum(value for name, value in report["execution"]["serviced"].items()
                   if name.startswith("link:")) == (240 if semantics == "per_port" else 120)
        assert topology(raw).accounting(exchange(30), 0).to_dict()["wire_bytes"] == 120
    # 120B on shared fabric; 30B per balanced Tx/Rx port; 10B per pair.
    assert results == pytest.approx({"shared": .12, "per_port": .03, "per_pair": .01})


def test_pair_wires_contend_only_when_the_same_directed_pair_is_used():
    raw = case(dp=2, tp=2)
    raw["machine"]["links"][0]["semantics"] = "per_pair"
    topo = topology(raw)
    engine = Engine(topo.resources())
    same = [engine.submit(f"same{index}", topo.demand(exchange(10), 0)[0])
            for index in range(2)]
    separate = engine.submit("separate", topo.demand(exchange(10), 1)[0])
    engine.advance(until=(*same, separate), first=False)
    assert [event.time for event in same] == pytest.approx([.02, .02])
    assert separate.time == pytest.approx(.01)
    assert engine.serviced["link:fabric:pair0>1"] == 20
    assert engine.serviced["link:fabric:pair2>3"] == 10


def test_expert_tensor_groups_do_not_join_independent_pcp_copies():
    raw = case(tp=4)
    raw["machine"]["num_devices"] = 8
    raw["machine"]["links"][0].update(members=list(range(8)), semantics="per_pair", latency_us=1)
    topo = Topology(Hardware.from_mapping(raw["machine"]), ParallelSpec(tp=4, ep=2, pcp=2))
    assert topo.groups("etp", 0, 0) == ((0, 2), (1, 3), (4, 6), (5, 7))
    work = topo.accounting(exchange(14, family="allreduce", axis="etp"), 0)
    assert work.messages == 8 * 2
    assert work.setup_ms == pytest.approx(.002)
    for name in work.wire_by_resource:
        source, destination = map(int, name.split(":pair")[1].split(">"))
        assert source // 4 == destination // 4
        assert source % 2 == destination % 2


def test_cross_dp_expert_groups_keep_pcp_stage_and_cohort_ownership():
    raw = case(dp=4, tp=2, pp=2)
    raw["machine"]["num_devices"] = 32
    raw["machine"]["links"][0].update(members=list(range(32)), semantics="per_pair",
                                     transaction_bytes=16, protocol_overhead_bytes=4)
    parallel = ParallelSpec(dp=4, tp=2, ep=4, pp=2, pcp=2)
    topo = Topology(Hardware.from_mapping(raw["machine"]), parallel)
    assert topo.groups("ep", 0, 1) == ((4, 5, 12, 13), (6, 7, 14, 15))
    assert topo.groups("ep", 2, 1) == ((20, 21, 28, 29), (22, 23, 30, 31))
    op = exchange(30, axis="ep", stage=1)
    participants = [topo.accounting(op, replica) for replica in (0, 1)]
    assert [set(row.useful_by_rank) for row in participants] == [{4, 5, 6, 7}, {12, 13, 14, 15}]
    assert sum(row.to_dict()["useful_bytes"] for row in participants) == 8 * 30
    assert sum(row.to_dict()["wire_bytes"] for row in participants) == 8 * 3 * 20
    for row in participants:
        for name in row.wire_by_resource:
            source, destination = map(int, name.split(":pair")[1].split(">"))
            assert source % 4 // 2 == destination % 4 // 2
            assert source // 16 == destination // 16 == 0


def test_frontend_graph_and_serving_charge_the_same_declared_wire_work():
    raw = case(tp=4, prompt=4, output=1)
    raw["machine"]["links"][0].update(transaction_bytes=64, protocol_overhead_bytes=8)
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4, scope="model"))
    served = predict(raw)
    rows = [row for row in graph["isolated"]["operators"] if "communication" in row]
    expected = sum(row["communication"]["wire_bytes"] * row["occurrences"] for row in rows)
    assert expected > 0
    assert graph["execution"]["serviced"]["link:fabric"] == expected
    assert served["execution"]["serviced"]["link:fabric"] == expected
    assert served["execution"]["live_tasks"] == served["execution"]["live_programs"] == 0
