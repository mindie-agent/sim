"""Physical link sharing, switch incast, tier geometry, and transport lowering.

[Role]
  Verify ``shared`` / ``per_port`` / ``per_pair`` capacity domains, switch
  serialization and duplex overlap, wide-node single-port defaults,
  rank/die/node placement, configurable multi-tier networks, multi-hop bounds,
  receive credits, and native HCCL/MC2 transport lowering.

[Invariants]
  * One wire volume is priced once; only its sharing domain changes service.
  * Unity ``bw_utilization`` keeps ``Link.to_dict`` byte-compatible; the
    declared peak capacity is preserved while effective service moves.
  * Switches and receive ports serialize only the work that shares them;
    wide nodes default to one per-port resource unless declared otherwise.
  * A multi-hop message is bounded by its widest resource, never by
    payload-times-hop-count; chunking adds no bytes.
  * A finite receive credit changes residency and completion, never work.

[Timeouts & Locks]
  Tier-0 in-memory Engines plus package JSON reads (<5 s); no locks, sleeps,
  threads or device contexts.

[Side-Effect Policy]
  Read-only; constructs local Engines/Hardware and writes nothing.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim import compose_case, graph_costs, predict
from mindie_sim.backends.ascend import materialize as ascend_materialize
from mindie_sim.backends.ascend import profiles
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.operators.collectives.direct_peer import DirectPeerCollective
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.operators.collectives.hccl_graph import NETWORK as HCCL_NETWORK
from mindie_sim.prediction.operators.collectives.hccl_graph import lower as hccl_lower
from mindie_sim.prediction.operators.mc2.mc2_graph import NETWORK as MC2_NETWORK
from mindie_sim.prediction.operators.mc2.mc2_graph import recipes as mc2_recipes
from mindie_sim.prediction.operators.mc2.mc2_graph import select as mc2_select
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.sweep.resize import (
    _first_link_by_channel, materialize_machine_for_devices, project_transfer_channel_input,
)

from _support.axes import COMPOSE_HARDWARE, GEN2_SKU, GEN3_SKU, GEN5_DT_SKU, GEN5_DT_SMALL_SKU, GEN5_PR_SKU
from _support.collective_cases import case as declared_case
from _support.collective_cases import declared, exchange
from _support.collective_cases import topology as build_topology

pytestmark = pytest.mark.tier0

TINY_MODEL = "qwen3.5-35b-a3b"
GENERIC_MODEL = "deepseek-v3"

#: ``(sku, preset, devices, ranks_per_node, tiers, routes)`` where each tier is
#: ``(channel, member_count, service_bytes_per_ms, latency_us)``.
TIER_LADDERS = ( (GEN2_SKU, "a2_standard", 16, None,
     (("d2d_intra_node", 8, 18.2e6, 16.0), ("d2d_inter_node", 16, 25e6 * 0.80, 10.0)), ()),
    (GEN5_PR_SKU, "a5_pcie_16p", 32, None,
     (("d2d_intra_4p", 4, 288e6 * 0.85, None), ("d2d_intra_8p", 8, 128e6 * 0.85, None),
      ("d2d_intra_node", 16, 116e6 * 0.85, None), ("d2d_inter_node", 32, 50e6 * 0.80, None)),
     ((0, 2, "d2d_intra_4p:0"), (0, 6, "d2d_intra_8p:0"), (0, 12, "d2d_intra_node:0"), (0, 20, "d2d_inter_node:0"))),
    (GEN5_DT_SKU, "a5_pod", 128, 64,
     (("d2d_intra_node", 64, 800e6 * 0.85, None), ("d2d_inter_node", 128, 800e6 * 0.80, 4.5)), ()),
    (GEN3_SKU, "a3_superpod", 784, None,
     (("d2d_intra_node", 16, 196e6 * 0.80, None), ("d2d_inter_node", 768, 196e6 * 0.70, 3.5),
      ("d2d_inter_pod", 784, 25e6 * 0.70, 10.0)),
     ((0, 8, "d2d_intra_node:0"), (0, 500, "d2d_inter_node:0"), (0, 768, "d2d_inter_pod:0"))),
)

def _machine(sku: str, *, devices: int = 16, ranks_per_node: int = 8, **extra) -> Hardware:
    return Hardware.from_mapping({ "backend": "ascend", "sku": sku, "num_devices": devices,
        "ranks_per_node": ranks_per_node, **extra})

def _link_machine(links, *, devices: int = 2, ranks_per_node: int | None = None) -> Hardware:
    base = _machine(GEN2_SKU, devices=devices, ranks_per_node=ranks_per_node or devices)
    return replace(base, links=tuple( item if isinstance(item, Link) else Link.from_mapping(item) for item in links))

def _channel_link(machine: Hardware, channel: str, members=None) -> Link:
    candidates = [link for link in machine.links if link.channel == channel]
    if members is not None:
        candidates = [link for link in candidates if tuple(link.members) == tuple(members)]
    assert candidates, f"missing link: {channel} {members}"
    return candidates[0]

def _materialized_link(raw: dict, channel: str) -> dict:
    return next(link for link in raw["links"] if link["channel"] == channel)

def _arrival_ms(machine: Hardware, source: int, destination: int, payload: float) -> float:
    topology = Topology(machine, ParallelSpec(tp=16))
    message = topology.route_message(MessageFlow(source, destination, payload))
    assert len(message.hops) == 1
    hop = message.hops[0]
    engine = Engine(topology.resources())
    service = engine.submit("route", {hop.resource: hop.wire_bytes})
    arrival = engine.submit("arrival", after=(service,), latency=hop.latency_ms)
    engine.advance(until=(arrival,), first=False)
    return arrival.time

def _run_plan(plan) -> dict:
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    run = plan.start(engine, "flow", ready)
    engine.advance(until=run.completions, first=False)
    return run.report()

def _wire(report: dict, needle: str = "link:") -> float:
    return sum(value for name, value in report["execution"]["serviced"].items() if name.startswith(needle))

def _fabric(scale: float = 1, plane: float | None = None) -> dict:
    raw = declared_case(tp=4)
    raw["machine"]["links"][0].update( semantics="per_port", capacity_bytes_per_ms=1000 * scale,
        latency_us=2, switch_hops=2, switch_hop_latency_us=3,
        transaction_bytes=16, protocol_overhead_bytes=4,
        switch_capacity_bytes_per_ms=plane)
    return raw

def _flow_span(raw: dict, flows):
    topology = build_topology(raw)
    engine = Engine(topology.resources())
    ready = engine.submit("input", latency=7)
    outputs = []
    for index, (source, destination) in enumerate(flows):
        previous = ready
        for hop in topology.route_message(MessageFlow(source, destination, 10)).hops:
            previous = engine.submit(str(index), hop.demand, after=(previous,), coupled=True)
            previous = engine.submit(str(index) + ":arrival", after=(previous,), latency=hop.latency_ms)
        outputs.append(previous)
    engine.advance(until=tuple(outputs), first=False)
    return max(event.time for event in outputs) - ready.time, engine

def _h4_plan(semantics: str = "shared", *, endpoint_rates=None, capacity=None, reverse: bool = False):
    rates = endpoint_rates or {rank: 120_000 for rank in range(4)}
    endpoint = tuple(sorted((int(rank), float(rate)) for rank, rate in rates.items()))
    link = Link("shared_forward", 60_000, (0, 1, 2, 3), 2.0, "fabric", semantics, endpoint_rates=endpoint)
    topology = Topology(_link_machine((link,), devices=4), ParallelSpec(tp=1))
    flows = ([MessageFlow(0, 2, 1200), MessageFlow(2, 0, 1200)] if reverse
             else [MessageFlow(0, 2, 1200), MessageFlow(1, 3, 1200)])
    return topology.transport_plan(flows, chunk_bytes=300, hop_startup_us=2, receive_capacity_bytes=capacity)

def test_link_semantics_scale_one_wire_volume_differently():
    """120 B on a shared plane, 30 B per balanced port, 10 B per pair."""
    expected = {"shared": (120, 0.12), "per_port": (240, 0.03), "per_pair": (120, 0.01)}
    for semantics, (serviced, duration) in expected.items():
        raw = declared_case(tp=4)
        raw["machine"]["links"][0]["semantics"] = semantics
        report = graph_costs(raw, (exchange(30),))
        assert sum(value for name, value in report["execution"]["serviced"].items()
                   if name.startswith("link:")) == serviced
        assert report["duration_ms"] == pytest.approx(duration)
        assert build_topology(raw).accounting(exchange(30), 0).to_dict()["wire_bytes"] == 120

    paired = declared_case(dp=2, tp=2)
    paired["machine"]["links"][0]["semantics"] = "per_pair"
    topology = build_topology(paired)
    engine = Engine(topology.resources())
    same = [engine.submit(f"same{index}", topology.demand(exchange(10), 0)[0]) for index in range(2)]
    separate = engine.submit("separate", topology.demand(exchange(10), 1)[0])
    engine.advance(until=(*same, separate), first=False)
    assert [event.time for event in same] == pytest.approx([0.02, 0.02])
    assert separate.time == pytest.approx(0.01)
    assert engine.serviced["link:fabric:pair0>1"] == 20
    assert engine.serviced["link:fabric:pair2>3"] == 10

def test_declared_link_fields_survive_serialization():
    link = Link(name="fabric", capacity_bytes_per_ms=25_000_000, members=(0, 1),
                latency_us=2.0, channel="d2d_inter_node", semantics="per_port")
    historical = {"name": "fabric", "capacity_bytes_per_ms": 25_000_000, "members": (0, 1), "latency_us": 2.0,
                  "channel": "d2d_inter_node", "semantics": "per_port"}
    assert link.to_dict() == historical
    assert link.effective_peak_capacity_bytes_per_ms == 25_000_000
    parsed = Link.from_mapping({"capacity_bytes_per_ms": 12_500_000, "bw_utilization": 0.5, "members": [0, 1]})
    assert parsed.capacity_bytes_per_ms == 12_500_000
    assert parsed.effective_peak_capacity_bytes_per_ms == 25_000_000
    throttled = parsed.to_dict()
    assert throttled["bw_utilization"] == 0.5
    assert throttled["peak_capacity_bytes_per_ms"] == 25_000_000

    generated = Hardware.from_mapping({ "backend": "ascend", "sku": GEN5_DT_SKU, "num_devices": 8,
        "transfer_channels": {"d2d_intra_node": { "switch_hops": 2, "switch_hop_latency_us": .4, "latency_us": 1,
            "transaction_bytes": 512, "alignment_bytes": 64,
            "protocol_overhead_bytes": 32, "switch_capacity_bytes_per_ms": 1e8}}})
    widened = _channel_link(generated, "d2d_intra_node")
    assert widened.propagation_us == pytest.approx(1.8)
    assert MessageFlow(0, 1, 513).on_link(widened) == (1088, 2)
    assert widened.switch_capacity_bytes_per_ms == 1e8

@pytest.mark.parametrize("preset", (None, "a5_pod"))
def test_wide_node_uses_one_port_unless_semantics_is_explicit(preset):
    options = dict( hardware=GEN5_DT_SMALL_SKU, tp=4, dp=8, ep=4, num_devices=32,
        ranks_per_node=16, superpod_size=32, intra_node_bw_gbps=800,
        intra_node_bw_util=.85, intra_node_latency_us=12, graph_capture_mode="none",
        prompt_tokens=8, output_tokens=2)
    if preset is not None:
        options["topology_preset"] = preset
    parsed = parse_case(compose_case(GENERIC_MODEL, **options))
    links = [link for link in parsed.machine.links if link.channel == "d2d_intra_node"]
    assert links
    assert all(link.semantics == "per_port" for link in links)
    assert all(link.capacity_bytes_per_ms == pytest.approx(680e6) for link in links)

@pytest.mark.parametrize("scale", (0.5, 1, 2))
def test_incast_serializes_on_one_receiver_while_duplex_overlaps(scale):
    incast, engine = _flow_span(_fabric(scale), [(0, 3), (1, 3), (2, 3)])
    disjoint, _ = _flow_span(_fabric(scale), [(0, 1), (1, 2), (2, 3)])
    duplex, _ = _flow_span(_fabric(scale), [(0, 1), (1, 0)])
    assert incast == pytest.approx(.060 / scale + .008)
    assert disjoint == duplex == pytest.approx(.020 / scale + .008)
    assert engine.serviced["link:fabric:rx:port3"] == 60

def test_shared_switch_plane_limits_disjoint_ports_and_keeps_one_frame():
    elapsed, engine = _flow_span(_fabric(plane=500), [(0, 1), (1, 2), (2, 3)])
    assert elapsed == pytest.approx(.120 + .008)
    assert engine.serviced["link:fabric:switch_plane"] == 60

    topology = build_topology(_fabric(plane=500))
    accounting = topology.accounting(exchange(30), 0).to_dict()
    assert accounting["messages"] == 12
    assert accounting["wire_bytes"] == 240
    assert accounting["routed_useful_bytes"] == 120
    assert accounting["capacity_byte_service"] == 720
    assert accounting["setup_ms"] == pytest.approx(3 * .008)
    assert Hardware.from_mapping(topology.machine.to_dict()) == topology.machine

def test_declared_utilization_moves_service_and_preserves_peak():
    baseline = _machine(GEN3_SKU)
    throttled = _machine(GEN3_SKU, transfer_channels={ "d2d_inter_node": {"bw_utilization": 0.5, "alpha_us": 17.0}})
    base_link = _channel_link(baseline, "d2d_inter_node")
    half_link = _channel_link(throttled, "d2d_inter_node")
    assert half_link.effective_peak_capacity_bytes_per_ms == 25_000_000
    assert half_link.capacity_bytes_per_ms == pytest.approx( base_link.capacity_bytes_per_ms / 2)
    payload = 25_000_000
    base_ms = _arrival_ms(baseline, 0, 8, payload)
    assert base_ms == pytest.approx(payload / 25_000_000 + 0.005)
    assert _arrival_ms(throttled, 0, 8, payload) == pytest.approx( payload / 12_500_000 + 0.017)

    bounded = _machine(GEN3_SKU, inter_node_bw_util=0.5, inter_node_latency_us=9.0)
    bounded_link = _channel_link(bounded, "d2d_inter_node")
    assert bounded_link.latency_us == 9.0
    assert bounded_link.capacity_bytes_per_ms * 2 == pytest.approx( base_link.capacity_bytes_per_ms)
    assert _arrival_ms(bounded, 0, 8, payload) - base_ms == pytest.approx( 1.0 + 0.009 - 0.005)

    def composed(util):
        return parse_case(compose_case( TINY_MODEL, hardware="a3", tp=8, dp=2, ep=16, ranks_per_node=8,
            layers=1, prompt_tokens=32, output_tokens=4, concurrency=1,
            inter_node_bw_gbps=100.0, inter_node_bw_util=util,
            inter_node_latency_us=4.0))

    whole, half = composed(1.0), composed(0.5)
    assert _channel_link(half.machine, "d2d_inter_node").latency_us == 4.0
    assert _arrival_ms(half.machine, 0, 8, payload) - _arrival_ms( whole.machine, 0, 8, payload) == pytest.approx(
            payload / 50_000_000 - payload / 100_000_000)

    local_half = _machine(GEN3_SKU, intra_node_bw_util=0.5)
    local_link = _channel_link(local_half, "d2d_intra_node", range(8))
    local_peak = _channel_link(baseline, "d2d_intra_node", range(8)).capacity_bytes_per_ms
    assert local_link.effective_peak_capacity_bytes_per_ms == pytest.approx(local_peak)
    assert local_link.capacity_bytes_per_ms == pytest.approx(local_peak / 2)
    assert Hardware.from_mapping(local_half.to_dict()) == local_half

@pytest.mark.parametrize("sku,preset,devices,ranks_per_node,tiers,routes", TIER_LADDERS)
def test_tiered_preset_emits_the_declared_link_ladder(sku, preset, devices, ranks_per_node, tiers, routes):
    machine = Hardware.from_mapping({ "backend": "ascend", "sku": sku, "num_devices": devices,
        "topology_preset": preset})
    for channel, members, capacity, latency in tiers:
        link = _channel_link(machine, channel, range(members))
        assert link.capacity_bytes_per_ms == pytest.approx(capacity)
        assert latency is None or link.latency_us == pytest.approx(latency)
    if ranks_per_node is not None:
        assert machine.ranks_per_node == ranks_per_node
    if preset == "a5_pcie_16p":
        assert machine.dies_per_package == 1
    if preset == "a5_pod":
        assert not [link for link in machine.links if link.channel == "d2d_inter_pod"]
    topology = Topology(machine, ParallelSpec(tp=devices))
    for source, destination, expected in routes:
        assert [hop[0] for hop in topology.route(source, destination)] == [expected]
    assert Hardware.from_mapping(machine.to_dict()) == machine

@pytest.mark.parametrize(("alias", "sku"), [(alias, sku) for alias, sku, _ in COMPOSE_HARDWARE])
def test_public_skus_compose_parse_and_predict_across_nodes(alias, sku):
    case = compose_case(TINY_MODEL, hardware=alias, tp=2, dp=2, ep=2, ranks_per_node=2,
                        layers=1, prompt_tokens=4, output_tokens=2, concurrency=1)
    machine = parse_case(case).machine
    profile = profiles()[sku]
    assert machine.sku == sku and len(machine.dies) == 4
    assert machine.dies[0].memory_bytes == profile["device"]["memory_bytes"]
    for channel, members in (("d2d_intra_node", range(2)), ("d2d_inter_node", range(4))):
        link = _channel_link(machine, channel, members)
        assert link.capacity_bytes_per_ms == pytest.approx(
            profile["transfer_channels"][channel]["effective_uni_bw_gbps"] * 1e6)
    result = predict(case)
    assert result["status"] == "modeled"
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0

def test_peak_and_calibration_overrides_survive_materialization():
    asc = ascend_materialize({ "sku": GEN3_SKU, "num_devices": 16, "ranks_per_node": 8,
        "transfer_channels": {"d2d_inter_node": { "peak_capacity_bytes_per_ms": 50000000.0, "bw_utilization": 0.5}}})
    declared_channel = asc["transfer_channels"]["d2d_inter_node"]
    assert declared_channel["peak_capacity_bytes_per_ms"] == pytest.approx(50000000.0)
    assert declared_channel["effective_uni_bw_gbps"] == pytest.approx(25.0)
    assert _materialized_link(asc, "d2d_inter_node")["capacity_bytes_per_ms"] \
        == pytest.approx(25000000.0)

    calibrated = _machine(GEN3_SKU, transfer_channels={"d2d_inter_node": {
        "calibrated_uni_bw_gbps": 80.0, "bw_utilization": 0.5, "alpha_us": 3.5}})
    link = _channel_link(calibrated, "d2d_inter_node")
    assert link.semantics == "per_port" and link.latency_us == 3.5
    assert link.capacity_bytes_per_ms == pytest.approx(40e6)

    explicit = Hardware.from_mapping({
        "backend": "ascend", "sku": GEN3_SKU, "num_devices": 16, "ranks_per_node": 8,
        "inter_node_bw_util": 0.5,
        "links": [{"name": "inter:0", "members": list(range(16)), "capacity_bytes_per_ms": 100e6, "latency_us": 2.0,
                   "channel": "d2d_inter_node"}]})
    assert explicit.links[0].capacity_bytes_per_ms == pytest.approx(50e6)

    package = Hardware.from_mapping({ "backend": "ascend", "sku": GEN3_SKU, "num_devices": 2, "ranks_per_node": 2,
        "intra_package_bw_gbps": 500.0})
    assert _channel_link(package, "d2d_intra_package") \
        .effective_peak_capacity_bytes_per_ms == pytest.approx(500e6)

    alias = compose_case(TINY_MODEL, hardware={"sku": "a5-server-pr-pcie-16p", "inter_node_bw_util": 0.7},
                         tp=1, dp=1, prompt_tokens=16, output_tokens=2)
    assert alias["machine"]["sku"] == GEN5_PR_SKU
    assert alias["machine"]["topology_preset"] == "a5_pcie_16p"

def test_placement_changes_route_and_sharing_domain():
    def machine(node_of, split):
        devices = [{"rank": rank, "node": node_of(rank), "package": rank} for rank in range(8)]
        intra = [{"name": f"d2d_intra_node:{index}", "members": [4 * index + offset for offset in range(4)],
                  "capacity_bytes_per_ms": 800e6, "latency_us": 1.5,
                  "channel": "d2d_intra_node", "semantics": "per_port"}
                 for index in (range(2) if split else range(1))]
        if not split:
            intra[0]["members"] = list(range(8))
        return Hardware.from_mapping({ "backend": "ascend", "sku": GEN5_DT_SMALL_SKU, "num_devices": 8,
            "ranks_per_node": 4, "devices": devices,
            "links": [*intra, {"name": "d2d_inter_node:0", "members": list(range(8)),
                               "capacity_bytes_per_ms": 50e6, "latency_us": 4.5,
                               "channel": "d2d_inter_node", "semantics": "per_port"}]})

    split = Topology(machine(lambda rank: rank // 4, True), ParallelSpec(tp=8))
    flat = Topology(machine(lambda rank: 0, False), ParallelSpec(tp=8))
    assert split.location(5)["node"] == 1
    assert flat.location(5)["node"] == 0
    assert [hop[0] for hop in split.route(0, 5)] == ["d2d_inter_node:0"]
    assert [hop[0] for hop in flat.route(0, 5)] == ["d2d_intra_node:0"]
    flows = [MessageFlow(0, 5, 1024 * 1024), MessageFlow(5, 0, 1024 * 1024)]
    split_report = _run_plan(split.transport_plan(flows))
    flat_report = _run_plan(flat.transport_plan(flows))
    assert split_report["service_lower_bound_ms"] > flat_report["service_lower_bound_ms"]
    assert split_report["span_ms"] > flat_report["span_ms"]

    reports = {}
    for preset in ("a5_server", "a5_pod"):
        raw = compose_case(TINY_MODEL, hardware=GEN5_DT_SMALL_SKU, topology_preset=preset, tp=8, dp=1, ep=8, layers=1,
                           prompt_tokens=8, output_tokens=1, concurrency=1)
        reports[preset] = graph_costs(raw, (exchange(8 * 1024 * 1024, family="allreduce"),))
        assert reports[preset]["execution"]["live_tasks"] == 0
    server = {name for name in reports["a5_server"]["execution"]["serviced"] if name.startswith("link:")}
    pod = {name for name in reports["a5_pod"]["execution"]["serviced"] if name.startswith("link:")}
    assert any("intra_4p" in name for name in server)
    assert any("intra_node" in name for name in server)
    assert not any("intra_4p" in name for name in pod)
    assert all(":pair" in name for name in server)
    assert any(":tx:port" in name for name in pod)
    server_wire = _wire(reports["a5_server"])
    pod_wire = sum(value for name, value in reports["a5_pod"]["execution"]["serviced"].items()
                   if name.startswith("link:") and ":tx:port" in name)
    assert server_wire == pytest.approx(pod_wire)
    assert reports["a5_server"]["duration_ms"] != reports["a5_pod"]["duration_ms"]

def test_transport_lower_bound_separates_shared_work_from_endpoint_rate():
    shared = _run_plan(_h4_plan("shared"))
    disjoint = _run_plan(_h4_plan("per_pair"))
    reverse = _run_plan(_h4_plan("per_pair", reverse=True))
    assert shared["service_lower_bound_ms"] == pytest.approx(0.04)
    assert shared["shared_resource_work_bytes"] == pytest.approx(2400.0)
    assert shared["path_startup_us"] == pytest.approx(6.0)
    assert shared["span_ms"] == pytest.approx(0.046, rel=1e-6)
    assert disjoint["service_lower_bound_ms"] == pytest.approx(0.02)
    assert disjoint["span_ms"] < shared["span_ms"]
    assert reverse["service_lower_bound_ms"] == pytest.approx(0.02)
    assert len({name for name in reverse["bottleneck_resources"] if ":pair" in name}) == 2
    assert {name for name in shared["bottleneck_resources"] if name == "link:shared_forward"} == {"link:shared_forward"}

    bounded = _run_plan(_h4_plan("shared", endpoint_rates={ 0: 120_000, 1: 120_000, 2: 30_000, 3: 120_000}))
    by_flow = {(row["source"], row["destination"]): row for row in bounded["per_flow_isolated_service_ms"]}
    assert by_flow[(0, 2)]["isolated_service_ms"] == pytest.approx(0.04)
    assert by_flow[(1, 3)]["isolated_service_ms"] == pytest.approx(0.02)
    assert bounded["service_lower_bound_ms"] == pytest.approx(0.04)
    assert bounded["endpoint_bottleneck_work_bytes"] == pytest.approx(1200.0)

def test_transport_plan_bounds_hops_and_receive_credit():
    links = (Link("first", 1000, (0, 1), 1000.0, "fabric", "shared"),
             Link("second", 1000, (1, 2), 1000.0, "fabric", "shared"))
    topology = Topology(_link_machine(links, devices=3), ParallelSpec(tp=1))
    flow = MessageFlow(0, 2, 3000)
    whole = _run_plan(topology.transport_plan([flow]))
    pipelined = _run_plan(topology.transport_plan([flow], chunk_bytes=1000))
    assert whole["service_lower_bound_ms"] == pytest.approx(3.0)
    assert pipelined["service_lower_bound_ms"] == pytest.approx(3.0)
    assert pipelined["chunks"] == 3
    assert whole["span_ms"] == pytest.approx(8.0)
    assert pipelined["span_ms"] < whole["span_ms"]
    assert pipelined["capacity_work_bytes"] == pytest.approx(6000.0)
    assert whole["capacity_work_bytes"] == pytest.approx(6000.0)
    with pytest.raises(ModelingError, match="transport_chunk_expansion_unbounded"):
        topology.transport_plan([MessageFlow(0, 2, 8192)], chunk_bytes=1)

    flow = MessageFlow(0, 1, 1)
    with pytest.raises(ModelingError, match="transport_receive_capacity_zero"):
        topology.transport_plan([flow], receive_capacity_bytes={1: 0})
    # Supply an invalid plan directly to exercise the Engine startup guard.
    zero = replace(topology.transport_plan([flow]), receive_capacity=((1, 0),))
    engine = Engine(zero.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in zero.ranks}
    with pytest.raises(ModelingError, match="transport_receive_capacity_zero"):
        zero.start(engine, "zero", ready)

    baseline_full = _h4_plan("shared")
    baseline = _run_plan(replace(baseline_full, flows=(baseline_full.flows[0],)))
    assert baseline["credit_waits"] == 0
    full = _h4_plan("shared", capacity={2: 600})
    plan = replace(full, flows=(full.flows[0],))
    engine = Engine(plan.resources())
    ready = {rank: engine.submit(f"ready{rank}") for rank in plan.ranks}
    consume = engine.submit("decode_consume", latency=10)
    run = plan.start(engine, "credit", ready, release_events={2: (consume,)})
    engine.advance(until=run.completions, first=False)
    report = run.report()
    assert report["credit_waits"] >= 1
    assert report["span_ms"] > baseline["span_ms"]
    assert all(value <= 600 for value in report["receiver_reserved_peak_bytes"].values())
    assert all(value <= 600 for value in report["receiver_resident_peak_bytes"].values())
    assert report["receiver_resident_byte_us"][2] > 0
    assert report["capacity_work_bytes"] == baseline["capacity_work_bytes"]

def test_native_lowering_uses_declared_chunks_and_records_unknowns():
    raw = declared("AIV", "mesh_oneshot", ranks=4)
    parsed = parse_case(raw)
    baseline = replace(exchange(12 * 1024 * 1024, family="allreduce", axis="tp"),
                       deps=(), metadata=(("group_axis", "tp"),))
    chunked = replace(baseline, metadata=(("group_axis", "tp"), ("transport_chunk_bytes", 65536)))
    reports, names = {}, {}
    for label, op, messages in (("whole", baseline, 12), ("chunked", chunked, 1536)):
        selected = hccl_lower((op,), parsed.runtime, parsed.model.activation_dtype)
        network = next(
            item for item in selected
            if item.recipe == HCCL_NETWORK and item.op_id.endswith(".broadcast.network")
        )
        topology = Topology(parsed.machine, parsed.runtime.parallel)
        ledger = topology.accounting(network, 0)
        # Four ranks each send an 8 MiB payload to three peers.
        assert ledger.physical_wire_bytes == 4 * 3 * 8 * 1024 * 1024
        assert ledger.messages == messages
        reports[label] = graph_costs(raw, selected, input_ready_ms=0, trace_limit=100000)
        names[label] = [row.get("task") or row.get("name") or "" for row in reports[label]["events"]]
    assert not any("chunk" in name for name in names["whole"])
    assert any("chunk" in name for name in names["chunked"])
    assert sum(value for name, value in reports["chunked"]["execution"]["serviced"].items()
               if name.startswith("link:")) == pytest.approx(
        sum(value for name, value in reports["whole"]["execution"]["serviced"].items() if name.startswith("link:")))

    mc2 = {"native_moe_protocol": { "kind": "arch35_fullmesh", "max_inflight_per_rank": 2,
        "chunk_bytes": 1024, "hop_startup_us": 1.0,
        "receive_capacity_bytes_per_peer": 4096}}
    assert mc2_select(mc2)["chunk_bytes"] == 1024
    recipe = mc2_recipes(mc2)[MC2_NETWORK]
    assert isinstance(recipe, DirectPeerCollective)
    assert recipe.plan_options["chunk_bytes"] == 1024
    assert recipe.plan_options["receive_capacity_bytes_per_peer"] == 4096
    assert recipe.transport_unknowns() == {}
    with pytest.raises(ModelingError, match="chunk_bytes"):
        mc2_select({"native_moe_protocol": {"kind": "arch35_fullmesh", "chunk_bytes": 0}})
    bare = mc2_recipes({"native_moe_protocol": {"kind": "arch35_fullmesh"}})[MC2_NETWORK]
    assert "transfer_chunk_bytes" in bare.transport_unknowns()

@pytest.mark.parametrize("hardware", (
    {"sku": GEN3_SKU}, {"sku": GEN5_PR_SKU, "topology_preset": "a5_pcie_16p"}))
def test_sweep_resize_rebuilds_declared_channels_for_requested_dies(hardware):
    raw = compose_case(TINY_MODEL, hardware=hardware, tp=8, dp=1, ep=8, layers=1,
                       prompt_tokens=8, output_tokens=1, concurrency=1)
    parsed = parse_case(raw)
    resized = materialize_machine_for_devices(parsed, raw, 16)
    assert (len(resized.dies), resized.ranks_per_node) == (
        16, min(16, parsed.machine.physical_ranks_per_node))
    assert all(member < 16 for link in resized.links for member in link.members)
    assert Hardware.from_mapping(resized.to_dict()) == resized
    explicit_raw = compose_case(TINY_MODEL, hardware=hardware, tp=8, dp=1, ep=8, ranks_per_node=8,
                                layers=1, prompt_tokens=8, output_tokens=1, concurrency=1)
    explicit_resized = materialize_machine_for_devices(parse_case(explicit_raw), explicit_raw, 16)
    assert (len(explicit_resized.dies), explicit_resized.ranks_per_node) == (16, 8)

def test_channel_projection_keeps_declared_peak_and_drops_placeholders():
    projected = project_transfer_channel_input({
        "semantics": "per_port", "peak_capacity_bytes_per_ms": 50e6,
        "calibrated_uni_bw_gbps": 9.0, "unknown": 1})
    assert (projected["semantics"], projected["uni_peak_bw_gbps"]) == ("per_port", 50.0)
    assert "calibrated_uni_bw_gbps" not in projected and "unknown" not in projected
    assert project_transfer_channel_input({}) is None
    assert project_transfer_channel_input(
        {"semantics": "custom"}, materialized=False)["channel_semantics"] == "custom"
    assert _first_link_by_channel(replace(_machine(GEN3_SKU), links=())) == {}


def test_topology_explicit_scope_no_die_sniffing_and_roce_cross_node():
    """Topology requires explicit P18/link scope (zero die sniffing) & prices cross-node RoCE TP=16."""
    from mindie_sim.schema.hardware import Link

    raw_tp16 = compose_case("qwen3-32b", "950dt-96g", tp=16, dp=1, ep=1, layers=1)
    parsed_tp16 = parse_case(raw_tp16)
    topo_tp16 = Topology(parsed_tp16.machine, parsed_tp16.runtime.parallel)
    assert topo_tp16.p18_allreduce_rate_applies("allreduce", tuple(range(16))) is False
    # Cross-node TP=16 routes across declared cross-node links (not dropped or collapsed to intra-node).
    cross_route = topo_tp16.route(0, 8)
    assert cross_route and any(topo_tp16.link_scope(hop[0]) == "cross_node" for hop in cross_route)

    # Explicit Link.scope takes precedence over channel name heuristics.
    base = _machine(GEN3_SKU)
    tagged_link = Link(
        name="custom_tagged", channel="mystery_unregistered_chan",
        members=(0, 8), semantics="per_port", capacity_bytes_per_ms=10.0, latency_us=1.0,
        scope="cross_node",
    )
    tagged_hw = replace(base, links=(*base.links, tagged_link))
    par_2dp = parse_case(compose_case(TINY_MODEL, GEN3_SKU, tp=8, dp=2, layers=1)).runtime.parallel
    assert Topology(tagged_hw, par_2dp).link_scope("custom_tagged") == "cross_node"

    # Undeclared custom multi-node link scope fails fast at Topology construction (zero silent fallback).
    bad_link = Link(
        name="custom_mystery", channel="mystery_unregistered_chan",
        members=(0, 8), semantics="per_port", capacity_bytes_per_ms=10.0, latency_us=1.0,
    )
    bad_hw = replace(base, links=(*base.links, bad_link))
    with pytest.raises(ModelingError, match="link_scope_undeclared"):
        Topology(bad_hw, par_2dp)
