"""Executable full-duplex incast, framing and switch-hop counterexamples."""
import pytest

from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.engine import Engine
from test_communication_topology import topology, exchange
from test_prediction_system import case


def fabric(scale=1, plane=None):
    raw = case(tp=4)
    raw["machine"]["links"][0].update(
        semantics="per_port", capacity_bytes_per_ms=1000 * scale,
        latency_us=2, switch_hops=2, switch_hop_latency_us=3,
        transaction_bytes=16, protocol_overhead_bytes=4,
        switch_capacity_bytes_per_ms=plane)
    return raw


def run(raw, flows):
    topo = topology(raw)
    engine = Engine(topo.resources())
    ready = engine.submit("input", latency=7)
    outputs = []
    for index, (source, destination) in enumerate(flows):
        previous = ready
        for hop in topo.route_message(MessageFlow(source, destination, 10)).hops:
            previous = engine.submit(str(index), hop.demand, after=(previous,), coupled=True)
            previous = engine.submit(str(index) + ":arrival", after=(previous,), latency=hop.latency_ms)
        outputs.append(previous)
    engine.advance(until=tuple(outputs), first=False)
    return max(e.time for e in outputs) - ready.time, engine


def test_incast_uses_receiver_capacity_and_opposite_directions_overlap():
    for scale in (.5, 1, 2):
        # Three independent senders serialize on one Rx, not on their idle Tx.
        incast, engine = run(fabric(scale), [(0, 3), (1, 3), (2, 3)])
        disjoint, _ = run(fabric(scale), [(0, 1), (1, 2), (2, 3)])
        duplex, _ = run(fabric(scale), [(0, 1), (1, 0)])
        assert incast == pytest.approx(.060 / scale + .008)
        assert disjoint == duplex == pytest.approx(.020 / scale + .008)
        assert engine.serviced["link:fabric:rx:port3"] == 60


def test_shared_switch_plane_can_limit_otherwise_disjoint_ports():
    elapsed, engine = run(fabric(plane=500), [(0, 1), (1, 2), (2, 3)])
    assert elapsed == pytest.approx(.120 + .008)
    assert engine.serviced["link:fabric:switch_plane"] == 60


def test_wire_ledger_does_not_count_tx_rx_plane_as_three_transmissions():
    raw = fabric(plane=500)
    topo = topology(raw)
    accounting = topo.accounting(exchange(30), 0).to_dict()
    assert accounting["messages"] == 12
    assert accounting["wire_bytes"] == 240  # 12 messages, one 20-byte frame each.
    assert accounting["routed_useful_bytes"] == 120
    assert accounting["capacity_byte_service"] == 720
    assert accounting["setup_ms"] == pytest.approx(3 * .008)
    from mindie_sim.schema.hardware import Hardware
    assert Hardware.from_mapping(topo.machine.to_dict()) == topo.machine


def test_generated_ascend_links_keep_explicit_switch_and_framing_overrides():
    from mindie_sim.schema.hardware import Hardware
    machine = Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582", "num_devices": 8,
        "transfer_channels": {"d2d_intra_node": {
            "switch_hops": 2, "switch_hop_latency_us": .4,
            "latency_us": 1, "transaction_bytes": 512,
            "alignment_bytes": 64, "protocol_overhead_bytes": 32,
            "switch_capacity_bytes_per_ms": 1e8}}})
    link = next(link for link in machine.links if link.channel == "d2d_intra_node")
    assert link.propagation_us == pytest.approx(1.8)
    assert MessageFlow(0, 1, 513).on_link(link) == (1088, 2)
    assert link.switch_capacity_bytes_per_ms == 1e8
