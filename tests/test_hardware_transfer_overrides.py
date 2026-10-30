"""Machine transfer declarations must reach generated links and the Engine."""
from copy import deepcopy
import math

import pytest

from mindie_sim.backends.ascend import profiles
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec

SKU = "Ascend910_9362"


def declared(**extra):
    return {"backend": "ascend", "sku": SKU, "num_devices": 16,
            "ranks_per_node": 8, **extra}


def hardware(**extra):
    return Hardware.from_mapping(declared(**extra))


def channel_links(machine, channel):
    return [link for link in machine.links if link.channel == channel]


def routed_task(machine, source, destination, payload=25_000_000):
    topology = Topology(machine, ParallelSpec(tp=16))
    path = topology.route(source, destination)
    message = topology.route_message(MessageFlow(source, destination, payload))
    assert len(message.hops) == 1
    hop = message.hops[0]
    engine = Engine(topology.resources())
    service = engine.submit("route", {hop.resource: hop.wire_bytes})
    task = engine.submit("arrival", after=(service,), latency=hop.latency_ms)
    engine.advance(until=(task,), first=False)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return path, hop, task.time


def test_channel_mapping_and_list_merge_preserve_alpha_and_framing():
    original = deepcopy(
        profiles()[SKU]["transfer_channels"]["d2d_inter_node"])
    machine = hardware(transfer_channels={
        "d2d_inter_node": {
            "alpha_us": 9.0, "channel_semantics": "per_pair",
            "transaction_bytes": 64, "alignment_bytes": 32,
            "protocol_overhead_bytes": 8,
        },
    })
    inter = channel_links(machine, "d2d_inter_node")
    assert len(inter) == 1
    link = inter[0]
    assert link.capacity_bytes_per_ms == 25_000_000
    assert link.latency_us == 9.0
    assert link.semantics == "per_pair"
    assert (link.transaction_bytes, link.alignment_bytes,
            link.protocol_overhead_bytes) == (64, 32, 8)
    # The registered profile is read-only input, never a mutation target.
    assert profiles()[SKU]["transfer_channels"]["d2d_inter_node"] == original

    listed = hardware(transfer_channels=[
        {"channel": "d2d_intra_node", "alpha_us": 7.5},
        {"channel": "d2d_inter_node", "uni_peak_bw_gbps": 50.0},
    ])
    assert all(link.latency_us == 7.5
               for link in channel_links(listed, "d2d_intra_node"))
    # An explicit peak is not masked by the profile's older effective value.
    assert [link.capacity_bytes_per_ms
            for link in channel_links(listed, "d2d_inter_node")] == [50_000_000]


def test_standard_gbps_bytes_per_s_and_legacy_units_converge():
    base = channel_links(hardware(), "d2d_inter_node")[0]
    gbps = channel_links(hardware(inter_node_bw_gbps=200), "d2d_inter_node")[0]
    bytes_s = channel_links(
        hardware(inter_node_bw_gbps=100, inter_node_bw_bytes_per_s=25e9),
        "d2d_inter_node")[0]
    assert base.capacity_bytes_per_ms == gbps.capacity_bytes_per_ms == \
        bytes_s.capacity_bytes_per_ms == 25_000_000
    # 25 GB/s legacy, 200 Gb/s, and 25e9 B/s are the same byte rate.
    assert base.capacity_bytes_per_ms * 1e3 == 25e9 == \
        200e9 / 8 == 25e9

    # Explicit channel bandwidth beats same-layer convenience fields.
    explicit = channel_links(
        hardware(transfer_channels={
            "d2d_inter_node": {"effective_uni_bw_gbps": 50.0}},
            inter_node_bw_gbps=200),
        "d2d_inter_node")[0]
    assert explicit.capacity_bytes_per_ms == 50_000_000

    # The convenience fields only touch their own channel.
    intra = channel_links(
        hardware(intra_node_bw_gbps=400), "d2d_intra_node")
    assert intra and all(link.capacity_bytes_per_ms == 50_000_000
                         for link in intra)
    assert channel_links(
        hardware(intra_node_bw_gbps=400), "d2d_inter_node")[0].capacity_bytes_per_ms == \
        25_000_000

def test_explicit_links_replace_generated_links_and_empty_is_explicit():
    explicit = [{"name": "declared", "members": list(range(16)),
                 "capacity_gbps": 25, "transaction_bytes": 64,
                 "alignment_bytes": 32, "protocol_overhead_bytes": 8,
                 "latency_us": 2.0, "semantics": "per_port"}]
    machine = hardware(
        links=explicit,
        transfer_channels={"d2d_inter_node": {"effective_uni_bw_gbps": 50.0}},
        inter_node_bw_gbps=200,
    )
    assert [link.name for link in machine.links] == ["declared"]
    link = machine.links[0]
    assert link.capacity_bytes_per_ms == 25_000_000
    assert (link.transaction_bytes, link.alignment_bytes,
            link.protocol_overhead_bytes, link.latency_us) == (64, 32, 8, 2.0)
    assert hardware(links=[]).links == ()


@pytest.mark.parametrize("declaration", [
    [{"channel": "d2d_inter_node"}, {"channel": "d2d_inter_node"}],
    {"not_a_channel": {"effective_uni_bw_gbps": 1}},
    [{"channel": "d2d_inter_node", "not_a_field": 1}],
    [{"channel": "d2d_inter_node", "channel_semantics": "invented"}],
])
def test_duplicate_unknown_channel_and_unknown_fields_are_rejected(declaration):
    with pytest.raises(ModelingError):
        Hardware.from_mapping(declared(transfer_channels=declaration))


def test_cross_node_route_engine_uses_25e9_or_12_5e9_and_keeps_wire():
    base = hardware()
    half = hardware(inter_node_bw_bytes_per_s=12.5e9)
    same_as_gbps = hardware(inter_node_bw_gbps=200)

    base_path, base_hop, base_time = routed_task(base, 0, 8)
    gbps_path, gbps_hop, gbps_time = routed_task(same_as_gbps, 0, 8)
    half_path, half_hop, half_time = routed_task(half, 0, 8)
    assert base_path == gbps_path == half_path
    assert [hop[0] for hop in base_path] == ["d2d_inter_node:0"]
    assert base_hop.wire_bytes == gbps_hop.wire_bytes == half_hop.wire_bytes
    assert gbps_time == pytest.approx(base_time)
    assert base_time == pytest.approx(1.005)
    assert half_time == pytest.approx(2.005)
    assert base_hop.resource == half_hop.resource

    # Intranode route and service are unchanged by the inter-node override.
    _, intra_base, intra_base_time = routed_task(base, 0, 2)
    _, intra_half, intra_half_time = routed_task(half, 0, 2)
    assert intra_base.resource == intra_half.resource
    assert intra_base.wire_bytes == intra_half.wire_bytes
    assert intra_base_time == pytest.approx(intra_half_time)


def test_framing_and_semantics_survive_generated_link_execution():
    machine = hardware(transfer_channels={"d2d_inter_node": {
        "transaction_bytes": 64, "alignment_bytes": 32,
        "protocol_overhead_bytes": 8, "alpha_us": 0.0,
        "channel_semantics": "per_port",
    }})
    _, hop, time = routed_task(machine, 0, 8, payload=25_000_000)
    expected_transactions = math.ceil(25_000_000 / 64)
    assert hop.transaction_frame == (64, 72)
    assert hop.transactions == expected_transactions
    assert hop.wire_bytes == expected_transactions * 72
    assert time == pytest.approx(hop.wire_bytes / 25_000_000)


def test_default_source_ports_and_explicit_shared_capacity_change_real_contention():
    def concurrent(machine):
        topology = Topology(machine, ParallelSpec(tp=16))
        engine = Engine(topology.resources())
        ends = []
        for source, destination in ((0, 1), (1, 0)):
            message = topology.route_message(MessageFlow(source, destination, 448e6))
            assert len(message.hops) == 1
            hop = message.hops[0]
            assert "d2d_intra_package:0" in hop.resource
            service = engine.submit(str(source), {hop.resource: hop.wire_bytes})
            ends.append(engine.submit(
                f"arrival{source}", after=(service,), latency=hop.latency_ms))
        engine.advance(until=tuple(ends), first=False)
        return max(event.time for event in ends), sum(engine.serviced.values())

    base_time, base_wire = concurrent(hardware())
    # Neutral semantics wins over the legacy channel label when both are
    # explicitly supplied; inherited labels keep the prior source-port model.
    shared_time, shared_wire = concurrent(hardware(transfer_channels={
        "d2d_intra_package": {"channel_semantics": "per_port", "semantics": "shared"}}))
    assert base_wire == shared_wire == 896e6
    latency_ms = channel_links(hardware(), "d2d_intra_package")[0].latency_us / 1000
    assert base_time == pytest.approx(1 + latency_ms)
    assert shared_time == pytest.approx(2 + latency_ms)


def test_declared_channel_latency_changes_arrival_without_changing_wire():
    baseline = hardware()
    changed = hardware(transfer_channels={"d2d_inter_node": {"alpha_us": 17}})
    _, base_hop, base_time = routed_task(baseline, 0, 8)
    _, new_hop, new_time = routed_task(changed, 0, 8)
    assert new_hop.wire_bytes == base_hop.wire_bytes == 25_000_000
    assert base_time == pytest.approx(1.005)
    assert new_time == pytest.approx(1.017)
