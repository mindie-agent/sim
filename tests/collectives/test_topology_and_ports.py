"""Topology scoping, port coset contention, host root-complex, and HCCL bands.

[Role]
  Verify physical communication topology routing priorities (package -> node ->
  cross_node), cross-node ``(node_id, package_id)`` locality isolation,
  explicit link-scope validation, single-freeze generator ``flows`` handling,
  shared ``(node, physical_endpoint, rail, tx_or_rx)`` port/switch contention,
  Kunpeng Host-HCCS solo ceiling + shared ``(node, root_complex_id, dir)``
  Engine contention, and production piecewise ``CollectiveBandSpec`` lowering.

[Invariants]
  * Distinct nodes sharing ``package=0`` never match ``scope="package"`` links.
  * Unknown link scopes fail loudly via ``Link``, ``parse_hardware``, and ``Topology``.
  * Shared ports (single-link or cross-link) and shared switches serialize on
    one Engine resource; distinct physical rails overlap concurrently.
  * Piecewise ``CollectiveBandSpec`` drives production ``Execution`` HCCL
    latency/accounting, dispatches declared algorithms and ``expansion_mode``,
    conserves AllGather/AllToAll/AllToAllV byte contracts, and pipelines hops.

[Timeouts & Locks]
  Tier-0 pure in-memory + local JSON manifest checks (<3s); no locks or I/O.

[Side-Effect Policy]
  Read-only over repository profile/observation JSON; constructs local Engines.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mindie_sim import api as sdk_api
from mindie_sim.backends import ascend
from mindie_sim.prediction.collectives import (
    LoweredCollective, PairKey, PortKey, RootComplexKey, SwitchKey, Topology,
    frame_wire_bytes, lower_collective, root_complex_resources_for_dies,
    select_band, submit_lowered_collective,
)
from mindie_sim.prediction.collectives.band_plan import build_band_collective_plan
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.efficiency.cost import CostModel
from mindie_sim.prediction.engine import Allocation, Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.program import KernelProgram, Stage
from mindie_sim.prediction.runtime.wave_recorder import WaveRecorder
from mindie_sim.prediction.topology import MessageFlow as LegacyTopologyMessageFlow
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.hardware_links import CollectiveBandSpec, Link
from mindie_sim.schema.operators import IterationPlan, IterationWork, OpSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "data" / "observations" / "campaign-202611-a2-a3"

SINGLE_DIE_SKU = "Ascend910B4"
DUAL_DIE_SMALL_SKU = "Ascend910_9362"
DUAL_DIE_LARGE_SKU = "Ascend910_9382"
DIRECT_MESH_SKU = "Ascend950DT_9582"
TINY_MODEL_ID = "qwen3-32b"


def _make_machine(sku: str, *, num_devices: int, ranks_per_node: int | None = None, topology_preset: str | None = None) -> Hardware:
    raw: dict = {"sku": sku, "num_devices": num_devices}
    if ranks_per_node is not None:
        raw["ranks_per_node"] = ranks_per_node
    if topology_preset is not None:
        raw["topology_preset"] = topology_preset
    return Hardware.from_mapping(ascend(raw))


def _make_custom_machine(base_sku: str, *, num_devices: int, ranks_per_node: int, links: list[dict]) -> Hardware:
    raw = ascend({"sku": base_sku, "num_devices": num_devices, "ranks_per_node": ranks_per_node})
    raw["links"] = links
    return Hardware.from_mapping(raw)


def test_cross_node_identical_package_id_cannot_use_package_scope_link():
    """Two nodes whose local dies both have package=0 must route cross-node."""
    raw = ascend({"sku": DUAL_DIE_SMALL_SKU, "num_devices": 4, "ranks_per_node": 2})
    base_dev = dict(raw["device"])
    raw["dies"] = [
        {**base_dev, "rank": 0, "node": 0, "package": 0},
        {**base_dev, "rank": 1, "node": 0, "package": 0},
        {**base_dev, "rank": 2, "node": 1, "package": 0},
        {**base_dev, "rank": 3, "node": 1, "package": 0},
    ]
    raw["links"] = [
        {"name": "d2d_intra_package:shared_pkg0", "members": [0, 1, 2, 3],
         "capacity_bytes_per_ms": 200_000_000.0, "semantics": "per_port", "scope": "package"},
        {"name": "d2d_inter_node:0", "members": [0, 1, 2, 3],
         "capacity_bytes_per_ms": 50_000_000.0, "semantics": "per_port", "scope": "cross_node"},
    ]
    machine = Hardware.from_mapping(raw)
    topo = Topology(machine, ParallelSpec(tp=4))

    assert topo.location(0)["package"] == 0 and topo.location(2)["package"] == 0
    assert topo.location(0)["node"] == 0 and topo.location(2)["node"] == 1
    assert topo._scope_admits("package", 0, 1) is True
    assert topo._scope_admits("package", 0, 2) is False
    assert topo._scope_admits("node", 0, 2) is False
    assert topo._scope_admits("cross_node", 0, 2) is True
    assert topo.link_scope(topo.route(0, 1)[0][0]) == "package"
    assert topo.link_scope(topo.route(0, 2)[0][0]) == "cross_node"


def test_routing_priority_and_nonempty_expert_tp_groups():
    """Dual-die prefers package > node > cross_node; etp groups never emit empties."""
    machine = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=16, ranks_per_node=8)
    topo = Topology(machine, ParallelSpec(tp=16))
    small_topo = Topology(
        _make_machine(DUAL_DIE_SMALL_SKU, num_devices=2, ranks_per_node=2),
        ParallelSpec(tp=2),
    )
    assert small_topo.link_scope(small_topo.route(0, 1)[0][0]) == "package"
    assert topo.link_scope(topo.route(0, 1)[0][0]) == "package"
    assert topo.link_scope(topo.route(0, 2)[0][0]) == "node"
    assert topo.link_scope(topo.route(0, 8)[0][0]) == "cross_node"

    # TP=2, PCP=2, EP=4 with explicit rank_table must emit non-empty singleton etp groups
    etp_topo = Topology(
        _make_machine(DUAL_DIE_SMALL_SKU, num_devices=4, ranks_per_node=4),
        ParallelSpec(tp=2, pcp=2, ep=4, cp_plan="pcp", rank_table=((0, 1, 2, 3),)),
    )
    assert etp_topo.groups("etp", 0, 0) == ((0,), (1,), (2,), (3,))


def test_calibrated_mesh_optical_superpod_and_host_hccs_profiles():
    """Verify K8 per_pair calibration, optical superpod inter-node tier, and Host-HCCS."""
    single_node = _make_machine(SINGLE_DIE_SKU, num_devices=8, ranks_per_node=8)
    single_topo = Topology(single_node, ParallelSpec(tp=8))
    single_link_name = single_topo.route(0, 7)[0][0]
    single_link = single_topo.links[single_link_name]
    single_ch = dict(dict(single_node.transfer_channels)[single_link.channel])
    assert single_topo.link_scope(single_link_name) == "node"
    assert single_link.semantics == "per_pair"
    assert math.isclose(float(single_ch["uni_peak_bw_gbps"]), 24.0, rel_tol=1e-9)
    assert math.isclose(float(single_ch["calibrated_uni_bw_gbps"]), 18.2, rel_tol=1e-9)
    assert math.isclose(float(single_ch["alpha_us"]), 16.0, rel_tol=1e-9)
    # Dedicated K8 per_pair resources for (0->1) and (0->2) do not contend
    assert set(single_topo.capacity_resources(single_link, 0, 1)).isdisjoint(
        set(single_topo.capacity_resources(single_link, 0, 2))
    )

    # 32-die dual-die superpod routes 0->16 over optical HCCS (not RoCE)
    sp_machine = _make_machine(
        DUAL_DIE_LARGE_SKU, num_devices=32, ranks_per_node=16, topology_preset="a3_superpod"
    )
    sp_topo = Topology(sp_machine, ParallelSpec(tp=32))
    sp_link = sp_topo.links[sp_topo.route(0, 16)[0][0]]
    sp_ch = dict(dict(sp_machine.transfer_channels)[sp_link.channel])
    assert sp_ch["protocol"] == "HCCS_OPTICAL_SUPERPOD"
    assert sp_ch["transport_mode"] == "optical_hccs"
    sp_band = select_band("allreduce", (0, 16), sp_topo, 65536)
    assert "optical" in sp_band.evidence_id and "roce" not in sp_band.evidence_id
    # Shipped 9382 SIO preserves expansion_mode="direct" across small & mid bands (#2)
    sio_s = select_band("allreduce", (0, 1), sp_topo, 4096, expansion_mode="direct")
    sio_m = select_band("allreduce", (0, 1), sp_topo, 196608, expansion_mode="direct")
    assert sio_s.algorithm == "direct" and sio_s.expansion_mode == "direct" and sio_m.expansion_mode == "direct"

    # Dual-die Host-HCCS preserves peak >= solo and HCCS DMA protocol
    h2d = dict(dict(sp_machine.transfer_channels)["h2d"])
    d2h = dict(dict(sp_machine.transfer_channels)["d2h"])
    assert h2d["protocol"] == "HCCS" and h2d["transport_mode"] == "hccs_dma"
    assert float(h2d["uni_peak_bw_gbps"]) == 64.0 >= float(h2d["solo_effective_uni_bw_gbps"]) == 59.5
    assert float(d2h["uni_peak_bw_gbps"]) == 48.0 >= float(d2h["solo_effective_uni_bw_gbps"]) == 41.5

    mesh_node = _make_machine(DIRECT_MESH_SKU, num_devices=8, ranks_per_node=8)
    mesh_topo = Topology(mesh_node, ParallelSpec(tp=8))
    mesh_link = mesh_topo.links[mesh_topo.route(0, 7)[0][0]]
    assert mesh_link.semantics == "per_pair"


def test_unknown_link_scope_rejected_across_link_hardware_and_topology():
    """Unknown scopes fail loudly in Link, neutral Hardware parsing, and Topology."""
    assert LegacyTopologyMessageFlow is MessageFlow
    with pytest.raises(ModelingError, match="link_scope_unknown:.*bogus_scope"):
        Link(
            name="bad_link", members=(0, 1),
            capacity_bytes_per_ms=100_000_000.0, scope="bogus_scope",
        )

    raw = ascend({"sku": SINGLE_DIE_SKU, "num_devices": 4, "ranks_per_node": 4})
    raw.pop("links", None)
    raw["transfer_channels"]["d2d_intra_node"]["scope"] = "bogus_neutral"
    with pytest.raises(ModelingError, match="link_scope_unknown:bogus_neutral"):
        Hardware.from_mapping(raw)

    topo = Topology(_make_machine(SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4), ParallelSpec(tp=4))
    with pytest.raises(ModelingError, match="link_scope_unknown:unknown"):
        topo._scope_admits("unknown", 0, 1)


def test_transport_plan_freezes_generator_flows_once():
    """Passing a single-use generator to transport_plan matches tuple input."""
    topo = Topology(_make_machine(DUAL_DIE_SMALL_SKU, num_devices=4, ranks_per_node=4), ParallelSpec(tp=4))
    flows_tuple = (
        MessageFlow(0, 1, 65536),
        MessageFlow(0, 2, 131072),
        MessageFlow(2, 3, 262144),
    )
    plan_from_tuple = topo.transport_plan(
        flows_tuple, receive_capacity_bytes=524288, receive_buffer_id="rx_buf"
    )
    plan_from_gen = topo.transport_plan(
        (flow for flow in flows_tuple),
        receive_capacity_bytes=524288, receive_buffer_id="rx_buf",
    )
    assert plan_from_gen == plan_from_tuple
    assert tuple(fp.flow for fp in plan_from_gen.flows) == flows_tuple
    assert dict(plan_from_gen.receive_capacity) == {1: 524288.0, 2: 524288.0, 3: 524288.0}


def test_host_bus_preserves_solo_ceiling_and_contends_per_root_complex():
    """Solo die keeps full solo ceiling; concurrent dies contend on shared RC."""
    machine = _make_machine(DUAL_DIE_LARGE_SKU, num_devices=16, ranks_per_node=8)
    solo_h2d = machine.host_solo_bw_gbps("h2d")
    rc_h2d = machine.host_root_complex_ceiling_gbps("h2d")
    assert solo_h2d is not None and rc_h2d is not None
    assert solo_h2d > rc_h2d / 8.0
    assert math.isclose(machine.effective_host_channel_bw_gbps("h2d"), solo_h2d, rel_tol=1e-9)

    rc_names = {res.name for res in root_complex_resources_for_dies(machine, machine.dies)}
    for node_idx in (0, 1):
        for direction in ("h2d", "d2h"):
            assert RootComplexKey(node_idx, "rc0", direction).resource_name() in rc_names

    case = sdk_api.compose_case(
        TINY_MODEL_ID, hardware=DUAL_DIE_LARGE_SKU,
        options={"dp": 1, "tp": 16, "pp": 1, "ranks_per_node": 8, "layers": 1},
    )
    ex = Execution(case, CostModel(case))
    for node_idx in (0, 1):
        for direction in ("h2d", "d2h"):
            assert RootComplexKey(node_idx, "rc0", direction).resource_name() in ex.engine.resources

    payload_bytes = 64_000_000.0
    solo_rate, rc_rate = solo_h2d * 1_000_000.0, rc_h2d * 1_000_000.0
    rc_key = RootComplexKey(0, "rc0", "h2d").resource_name()
    eng_multi = Engine([
        *(Resource(f"d{i}", solo_rate, "bytes") for i in range(8)),
        Resource(rc_key, rc_rate, "bytes"),
    ])
    evs = [
        eng_multi.submit(f"t{i}", {f"d{i}": payload_bytes, rc_key: payload_bytes}, coupled=True)
        for i in range(8)
    ]
    eng_multi.advance()
    assert math.isclose(max(ev.time for ev in evs), (8.0 * payload_bytes) / rc_rate, rel_tol=1e-9)


def test_shared_port_single_link_and_switch_serialize_while_distinct_rails_overlap():
    """Shared ports (cross-link or single-link) & switches serialize; distinct rails overlap."""
    cap = 1000.0  # 1000 B/ms -> 1000 B = 1.0 ms solo, 2.0 ms when serialized
    payload = 1000.0

    # 1) Shared physical endpoint ("nic0") serializes on BOTH per_port AND per_pair (#4)
    for sem, extra in (("per_port", {}), ("per_pair", {"endpoint_rates": {"0": cap, "1": cap, "2": cap}})):
        links_1 = [{
            "name": "one_link", "members": [0, 1, 2], "capacity_bytes_per_ms": cap * 10.0,
            "semantics": sem, "scope": "node", "rail_id": "rail_s",
            "endpoint_ids": {"0": "nic0", "1": "nic0", "2": "nic2"}, **extra,
        }]
        if sem == "per_port":
            links_1[0]["capacity_bytes_per_ms"] = cap
        t1 = Topology(_make_custom_machine(SINGLE_DIE_SKU, num_devices=3, ranks_per_node=3, links=links_1), ParallelSpec(tp=3))
        r02 = t1.capacity_resources(t1.links["one_link"], 0, 2)
        r12 = t1.capacity_resources(t1.links["one_link"], 1, 2)
        assert PortKey(0, "nic0", "rail_s", "tx").resource_name() in r02
        assert PortKey(0, "nic0", "rail_s", "tx").resource_name() in r12
        e1 = Engine(t1.resources())
        ev0, ev1 = e1.submit("s0", {r: payload for r in r02}, coupled=True), e1.submit("s1", {r: payload for r in r12}, coupled=True)
        e1.advance()
        assert math.isclose(max(ev0.time, ev1.time), 2.0, rel_tol=1e-9)

    # 2) Two links sharing a physical switch_id plane
    sw_links = [
        {"name": "sw_a", "members": [0, 1], "capacity_bytes_per_ms": cap,
         "switch_capacity_bytes_per_ms": cap,
         "semantics": "per_port", "scope": "node", "rail_id": "r0", "switch_id": "sw0"},
        {"name": "sw_b", "members": [2, 3], "capacity_bytes_per_ms": cap,
         "switch_capacity_bytes_per_ms": cap,
         "semantics": "per_port", "scope": "node", "rail_id": "r1", "switch_id": "sw0"},
    ]
    topo_sw = Topology(
        _make_custom_machine(SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4, links=sw_links),
        ParallelSpec(tp=4),
    )
    ra = topo_sw.capacity_resources(topo_sw.links["sw_a"], 0, 1)
    rb = topo_sw.capacity_resources(topo_sw.links["sw_b"], 2, 3)
    assert SwitchKey(0, "sw0").resource_name() in ra and SwitchKey(0, "sw0").resource_name() in rb
    eng_sw = Engine(topo_sw.resources())
    ea = eng_sw.submit("sw0", {r: payload for r in ra}, coupled=True)
    eb = eng_sw.submit("sw1", {r: payload for r in rb}, coupled=True)
    eng_sw.advance()
    assert math.isclose(max(ea.time, eb.time), 2.0, rel_tol=1e-9)

    # 3) Distinct rails with no shared switch overlap concurrently (1.0 ms)
    disjoint = [
        {"name": "d_a", "members": [0, 1], "capacity_bytes_per_ms": cap,
         "semantics": "per_port", "scope": "node", "rail_id": "rail_0",
         "endpoint_ids": {"0": "nic0", "1": "nic1"}},
        {"name": "d_b", "members": [0, 2], "capacity_bytes_per_ms": cap,
         "semantics": "per_port", "scope": "node", "rail_id": "rail_1",
         "endpoint_ids": {"0": "nic0", "2": "nic2"}},
    ]
    topo_d = Topology(
        _make_custom_machine(SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4, links=disjoint),
        ParallelSpec(tp=4),
    )
    dra = topo_d.capacity_resources(topo_d.links["d_a"], 0, 1)
    drb = topo_d.capacity_resources(topo_d.links["d_b"], 0, 2)
    assert set(dra).isdisjoint(set(drb))
    eng_d = Engine(topo_d.resources())
    da = eng_d.submit("r0", {r: payload for r in dra}, coupled=True)
    db = eng_d.submit("r1", {r: payload for r in drb}, coupled=True)
    eng_d.advance()
    assert math.isclose(max(da.time, db.time), 1.0, rel_tol=1e-9)


def test_piecewise_collective_band_boundaries_and_overlap_rejection():
    """Exact payload band thresholds [0, 65537), [65537, 2097152), [2097152, None)."""
    topo = Topology(_make_machine(DUAL_DIE_SMALL_SKU, num_devices=8, ranks_per_node=8), ParallelSpec(tp=8))
    group = (0, 2, 4, 6)
    b_zero = select_band("allreduce", group, topo, 0)
    b_64k = select_band("allreduce", group, topo, 65536)
    b_64k_plus = select_band("allreduce", group, topo, 65537)
    b_512k = select_band("allreduce", group, topo, 524288)
    b_2m_minus = select_band("allreduce", group, topo, 2097151)
    b_2m = select_band("allreduce", group, topo, 2097152)

    assert b_zero == b_64k and b_zero.payload_max_exclusive == 65537
    assert b_64k_plus == b_512k and b_64k_plus.payload_max_exclusive == 2097152 and b_2m_minus.algorithm == "nhr"
    assert b_2m.payload_min_inclusive == 2097152 and b_2m.payload_max_exclusive is None

    overlap_a = CollectiveBandSpec(
        collective="allreduce", stack_id="default", topology_scope="node",
        group_min=2, group_max=8, expansion_mode="ring",
        payload_min_inclusive=0, payload_max_exclusive=131072,
        algorithm="ring", protocol="simple", startup_scope="collective",
        startup_us=5.0, payload_bytes_per_frame=4096, wire_bytes_per_frame=4224,
        credit_bytes=0, evidence_id="test_overlap_a",
    )
    overlap_b = CollectiveBandSpec(
        collective="allreduce", stack_id="default", topology_scope="node",
        group_min=2, group_max=8, expansion_mode="ring",
        payload_min_inclusive=65536, payload_max_exclusive=None,
        algorithm="ring", protocol="simple", startup_scope="collective",
        startup_us=2.0, payload_bytes_per_frame=4096, wire_bytes_per_frame=4096,
        credit_bytes=0, evidence_id="test_overlap_b",
    )
    with pytest.raises(ModelingError, match="collective_band_payload_intervals_overlap"):
        Link(
            name="bad_bands", members=(0, 1), capacity_bytes_per_ms=100_000_000.0,
            scope="node", protocol_bands=(overlap_a, overlap_b),
        )


def test_collective_byte_conventions_zero_payload_and_alltoallv_validation():
    """Verify AllReduce/AllGather/AllToAll exact API conventions and AllToAllV splits."""
    topo = Topology(_make_machine(DUAL_DIE_SMALL_SKU, num_devices=8, ranks_per_node=8), ParallelSpec(tp=8))
    group = (0, 2, 4, 6)

    # Ring AllReduce: 2(P-1)B/P
    ar = lower_collective("allreduce", group, topo, 4 * 1024 * 1024)
    assert isinstance(ar, LoweredCollective)
    assert tuple(p.name for p in ar.phases) == ("reduce_scatter", "all_gather")
    for _, sent in ar.per_rank_payload_bytes:
        assert math.isclose(sent, 2.0 * 3 * (4 * 1024 * 1024) / 4, rel_tol=1e-12)

    # AllGather(P=4, B=4096): each rank sends (P-1)*B = 12,288 B
    ag = lower_collective("allgather", group, topo, 4096)
    for _, sent in ag.per_rank_payload_bytes:
        assert math.isclose(sent, 12288.0, rel_tol=1e-12)

    # AllToAll(P=4, B=4096): each rank sends (P-1)*B/P = 3,072 B external + 1,024 B self
    a2a = lower_collective("alltoall", group, topo, 4096)
    for _, sent in a2a.per_rank_payload_bytes:
        assert math.isclose(sent, 3072.0, rel_tol=1e-12)
    for _, self_b in a2a.self_copy_bytes_by_rank:
        assert math.isclose(self_b, 1024.0, rel_tol=1e-12)

    # Zero-payload rendezvous still runs
    zero_low = lower_collective("allreduce", group, topo, 0)
    assert zero_low.rendezvous_required is True and zero_low.phases == ()
    eng_zero = Engine(topo.resources())
    ready_map = {r: eng_zero.submit(f"r{r}", release=0.05 * i) for i, r in enumerate(group)}
    done_map = submit_lowered_collective(eng_zero, "zero_ar", zero_low, ready_map)
    eng_zero.advance()
    expected_rv = 0.05 * 3 + zero_low.collective_startup_us / 1000.0
    assert math.isclose(done_map[0].time, expected_rv, rel_tol=1e-9)

    # AllToAllV: exact splits validation + zero api_bytes with nonzero matrix rejection
    peer_matrix = (
        (1024, 2048, 0, 4096),
        (512, 2048, 1024, 0),
        (0, 4096, 512, 2048),
        (2048, 0, 1024, 4096),
    )
    a2av = lower_collective(
        "alltoallv", group, topo, 7168,
        peer_matrix=peer_matrix,
        send_splits=(7168, 3584, 6656, 7168),
        recv_splits=(3584, 8192, 2560, 10240),
    )
    assert dict(a2av.self_copy_bytes_by_rank) == {0: 1024, 2: 2048, 4: 512, 6: 4096}
    assert dict(a2av.per_rank_payload_bytes) == {0: 6144.0, 2: 1536.0, 4: 6144.0, 6: 3072.0}
    frames, wire = frame_wire_bytes(4096, a2av.band)
    assert frames >= 1 and wire >= 4096.0
    with pytest.raises(ModelingError, match="alltoallv_api_bytes_zero_with_nonzero_matrix"):
        lower_collective("alltoallv", group, topo, 0, peer_matrix=peer_matrix)
    with pytest.raises(ModelingError, match="alltoallv_send_splits_mismatch"):
        lower_collective("alltoallv", group, topo, 7168, peer_matrix=peer_matrix, send_splits=(1, 1, 1, 1))
    with pytest.raises(ModelingError, match="alltoallv_peer_matrix_entry_invalid"):
        lower_collective("alltoallv", (0, 2), topo, 1024, peer_matrix=((0, -1), (1024, 0)))
    # AllToAllV with omitted splits derives band from validated matrix totals, not api_bytes=1 (#3)
    big_a2av = lower_collective(
        "alltoallv", group, topo, 1,
        peer_matrix=((0, 0, 0, 100_000), (0, 0, 0, 0), (0, 0, 0, 0), (0, 0, 0, 0)),
    )
    assert big_a2av.band.payload_min_inclusive == 65537 and "mid" in big_a2av.band.protocol


def test_band_algorithm_expansion_multihop_credits_and_production_execution():
    """Verify expansion_mode, direct alg, multi-hop + credit_bytes, and Execution HCCL."""
    def _band(coll: str, alg: str, startup_us: float, *, scope: str = "node", exp: str | None = None,
              gmax: int = 8, sscope: str = "collective", cred: int = 0, eid: str = "ev") -> CollectiveBandSpec:
        return CollectiveBandSpec(
            collective=coll, stack_id="default", topology_scope=scope, group_min=2, group_max=gmax,
            expansion_mode=exp or alg, payload_min_inclusive=0, payload_max_exclusive=None,
            algorithm=alg, protocol="simple", startup_scope=sscope, startup_us=startup_us,
            payload_bytes_per_frame=4096, wire_bytes_per_frame=4096, credit_bytes=cred, evidence_id=eid,
        )

    # 1) expansion_mode disambiguation & algorithm="direct" single-phase lowering
    b_ring = _band("allreduce", "ring", 10.0, gmax=4, eid="ev_ring")
    b_direct = _band("allreduce", "direct", 3.0, gmax=4, eid="ev_direct")
    topo_exp = Topology(_make_custom_machine(SINGLE_DIE_SKU, num_devices=2, ranks_per_node=2, links=[{
        "name": "exp_link", "members": [0, 1], "capacity_bytes_per_ms": 100_000_000.0,
        "semantics": "per_port", "scope": "node", "protocol_bands": [b_ring.to_dict(), b_direct.to_dict()],
    }]), ParallelSpec(tp=2))
    assert select_band("allreduce", (0, 1), topo_exp, 4096, expansion_mode="direct") == b_direct
    low_dir = lower_collective("allreduce", (0, 1), topo_exp, 4096, expansion_mode="direct")
    assert tuple(p.name for p in low_dir.phases) == ("direct",)
    # 1b) 4-rank tree AllGather delivers full gathered buffer to ALL 4 ranks incl. 1 & 3 (#2)
    b_tree = _band("allgather", "tree", 1.0, eid="ev_tree")
    topo_tr = Topology(_make_custom_machine(SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4, links=[{
        "name": "tr_link", "members": [0, 1, 2, 3], "capacity_bytes_per_ms": 100_000_000.0,
        "semantics": "per_port", "scope": "node", "protocol_bands": [b_tree.to_dict()],
    }]), ParallelSpec(tp=4))
    low_tr = lower_collective("allgather", (0, 1, 2, 3), topo_tr, 1024)
    assert tuple(p.name for p in low_tr.phases) == ("gather_up", "broadcast_down")
    bcast_dsts = {tr.destination: tr.payload_bytes for tr in low_tr.phases[1].transfers}
    assert bcast_dsts == {1: 4096.0, 2: 4096.0, 3: 4096.0}

    # 2) Multi-hop pipelining + credit_bytes + switch propagation preserved on startup > 0 (#3)
    def _run_mh(startup_us: float, sw_lat_us: float = 0.0) -> float:
        bh = _band("alltoallv", "direct", startup_us, gmax=4, sscope="hop",
                   cred=4096 if sw_lat_us == 0.0 else 0, eid="ev_hop")
        tmh = Topology(_make_custom_machine(SINGLE_DIE_SKU, num_devices=3, ranks_per_node=3, links=[
            {"name": f"hop_{a}{b}", "members": [a, b], "capacity_bytes_per_ms": 1_000_000.0 if sw_lat_us == 0.0 else 1e12,
             "latency_us": 0.0, "switch_hops": 1 if sw_lat_us else 0, "switch_hop_latency_us": sw_lat_us,
             "semantics": "per_port", "scope": "node", "protocol_bands": [bh.to_dict()]}
            for a, b in ((0, 1), (1, 2))
        ]), ParallelSpec(tp=3))
        lmh = lower_collective("alltoallv", (0, 1, 2), tmh, 16384, peer_matrix=((0, 0, 16384), (0, 0, 0), (0, 0, 0)))
        assert len(lmh.phases[0].transfers[0].hops) == 2
        emh = Engine(tmh.resources())
        dmh = submit_lowered_collective(emh, "mh", lmh, {r: emh.submit(f"r{r}") for r in (0, 1, 2)})
        emh.advance()
        return float(dmh[2].time)

    assert _run_mh(100.0, 0.0) > 0.21
    assert math.isclose(_run_mh(0.0, 1000.0), 2.0, abs_tol=1e-4)
    assert math.isclose(_run_mh(1.0, 1000.0), 2.002, abs_tol=1e-4)

    # 3) Production Execution consumes band startup_us & credit_bytes window (#1)
    def _band_hw(sku: str, startup_us: float, *, credit_bytes: int = 0, scope_m: str = "collective"):
        raw_hw = ascend({"sku": sku, "num_devices": 4, "ranks_per_node": 4})
        band = _band("allreduce", "ring", startup_us, scope="any", sscope=scope_m,
                     cred=credit_bytes, eid=f"prod_ar_{int(startup_us)}_{credit_bytes}")
        for ch_key in ("d2d_intra_node", "d2d_intra_package"):
            if ch_key in raw_hw["transfer_channels"]:
                raw_hw["transfer_channels"][ch_key]["protocol_bands"] = [band.to_dict()]
        return raw_hw

    def _run_prod_allreduce(sku: str, startup_us: float, *, credit_bytes: int = 0, scope_m: str = "collective") -> float:
        case = sdk_api.compose_case(TINY_MODEL_ID, hardware=_band_hw(sku, startup_us, credit_bytes=credit_bytes, scope_m=scope_m),
                                    options={"dp": 1, "tp": 4, "pp": 1, "layers": 1})
        ex = Execution(case, CostModel(case), include_submission=False)
        op = OpSpec(op_id="ar_0", family="allreduce", recipe="all_reduce", phase="decode", stage=0, layer=0,
                    stream="main", dtype="bf16", link_bytes=65536, link_scope="tp", requests=("req0",),
                    semantic_shape=(1, 1, 32768), local_shape=(1, 1, 32768), padded_shape=(1, 1, 32768), launch_shape=(1, 1, 32768))
        work = IterationWork(plan=IterationPlan(iteration=0, requests=(RequestView("req0", q_len=1, context=16, samples=True, phase="decode"),)),
                             ops=(op,), stage_count=1, mode="decode", valid_tokens=1, padded_tokens=1, graph_bucket=1, recipe="default", replay_scope="full")
        step = ex.add(work, replica=0, release_ms=0.0)
        ex.engine.advance()
        return float(step.completion.time)

    for sku in (SINGLE_DIE_SKU, DUAL_DIE_SMALL_SKU):
        t_fast, t_slow = _run_prod_allreduce(sku, 2.5), _run_prod_allreduce(sku, 5000.0)
        assert t_fast < 0.5 and t_slow >= 5.0 and (t_slow - t_fast) >= 4.99
    t_nocred = _run_prod_allreduce(DUAL_DIE_SMALL_SKU, 1000.0, credit_bytes=0, scope_m="hop")
    t_cred = _run_prod_allreduce(DUAL_DIE_SMALL_SKU, 1000.0, credit_bytes=4096, scope_m="hop")
    assert t_cred > t_nocred * 2.5

    # 4) lower_with_local_work preserves Stage metadata/deliveries/placement/caps/allocs into Engine + WaveRecorder
    topo_s = Topology(Hardware.from_mapping(_band_hw(DUAL_DIE_SMALL_SKU, 2.5)), ParallelSpec(tp=4))
    plan_s = build_band_collective_plan(topo_s, "allreduce", ((0, 1, 2, 3),), {r: 98304.0 for r in range(4)}, {})
    link_res = next(iter(plan_s.accounting().wire_by_resource))
    alloc = Allocation("pack_buf", "l1", 64.0, (0,))
    base_prog = KernelProgram((Stage("setup", latency=0.05), Stage("body", after=("setup",), program=KernelProgram((
        Stage("pack", work={"hbm@d0": 32768.0}, placement={"hbm@d0": (0,)}, caps={"hbm@d0": 1e9}, acquire=(alloc,),
              free=(alloc,), latency=0.001, coupled=True, priority=2, work_units={"hbm@d0": "bytes"},
              metadata={"generation": 7}, deliveries=(("hbm_read@d0", 32768.0),), latency_from_first_service=True),
        Stage("reduce_scatter", work={link_res: 65536.0, "vec@d0": 1024.0}, after=("pack",), metadata={"generation": 8}),
        Stage("unpack_store", work={"hbm@d0": 32768.0}, after=("reduce_scatter",), deliveries=(("hbm_write@d0", 32768.0),)),
    )))))
    low_prog = plan_s.lower_with_local_work(SimpleNamespace(minimum_ms=0.0), base_prog)
    stgs = {s.name: s for s in low_prog.stages}
    assert stgs["pack"].metadata == {"generation": 7} and stgs["pack"].deliveries == (("hbm_read@d0", 32768.0),)
    assert stgs["pack"].placement == {"hbm@d0": (0,)} and stgs["pack"].caps == {"hbm@d0": 1e9} and stgs["pack"].acquire == (alloc,)
    assert stgs["pack"].free == (alloc,) and stgs["pack"].priority == 2 and stgs["pack"].coupled and stgs["pack"].latency == 0.001
    assert stgs["pack"].work_units == {"hbm@d0": "bytes"} and stgs["pack"].latency_from_first_service
    assert link_res not in stgs["reduce_scatter"].work and stgs["reduce_scatter"].metadata == {"generation": 8}
    for backend in ("python", "c"):
        eng = Engine([*topo_s.resources(), Resource("hbm@d0", 1e9, units=1), Resource("vec@d0", 1e9, units=1)],
                     pools={"l1": 64.0}, pool_units={"l1": 1}, backend=backend)
        rec = WaveRecorder(declared_pipes=("hbm@d0", "vec@d0"))
        eng.attach_wave_recorder(rec)
        eng.advance(until=(low_prog.start(eng, "ar", eng.submit("r")),))
        assert eng.delivered.get("hbm_read@d0") == 32768.0 and eng.delivered.get("hbm_write@d0") == 32768.0
        assert {7, 8} <= {row[3] for row in rec._admitted.values()}


def test_startup_scope_and_key_validation_and_catalog_invariants():
    """Verify startup_scope billing, key errors, 16 profiles l2==1, and campaign files."""
    def _run_scope(scope_mode: str) -> float:
        band = CollectiveBandSpec(
            collective="allreduce", stack_id="default", topology_scope="node",
            group_min=2, group_max=4, expansion_mode="ring",
            payload_min_inclusive=0, payload_max_exclusive=None,
            algorithm="ring", protocol="simple", startup_scope=scope_mode,
            startup_us=100.0, payload_bytes_per_frame=4096, wire_bytes_per_frame=4096,
            credit_bytes=0, evidence_id=f"test_{scope_mode}",
        )
        topo = Topology(
            _make_custom_machine(SINGLE_DIE_SKU, num_devices=3, ranks_per_node=3, links=[{
                "name": "fast_node", "members": [0, 1, 2],
                "capacity_bytes_per_ms": 1_000_000_000_000.0, "latency_us": 0.0,
                "semantics": "per_port", "scope": "node", "protocol_bands": [band.to_dict()],
            }]),
            ParallelSpec(tp=3),
        )
        low = lower_collective("allreduce", (0, 1, 2), topo, 4096)
        eng = Engine(topo.resources())
        done = submit_lowered_collective(
            eng, f"ar_{scope_mode}", low, {r: eng.submit(f"r{r}") for r in (0, 1, 2)}
        )
        eng.advance()
        return float(done[0].time)

    assert math.isclose(_run_scope("collective"), 0.1, abs_tol=1e-4)
    assert math.isclose(_run_scope("phase"), 0.2, abs_tol=1e-4)
    assert math.isclose(_run_scope("hop"), 0.4, abs_tol=1e-4)

    with pytest.raises(ModelingError, match="pair_key_self_loop"):
        PairKey("hccs", 1, 1, 0)
    with pytest.raises(ModelingError, match="port_key_direction_invalid"):
        PortKey(0, "nic0", 0, "bidirectional")
    with pytest.raises(ModelingError, match="root_complex_key_direction_invalid"):
        RootComplexKey(0, "rc0", "tx")

    all_profiles = importlib.import_module("mindie_sim.backends.ascend").profiles()
    assert len(all_profiles) == 16
    for sku_name, profile in all_profiles.items():
        assert (profile["device"].get("buffer_units") or {}).get("l2") == 1
        assert len(_make_machine(sku_name, num_devices=2, ranks_per_node=2).dies) == 2

    manifest = json.loads((CAMPAIGN_DIR / "manifest.json").read_text(encoding="utf-8"))
    assert manifest.get("schema")
    train_files = sorted(p for p in CAMPAIGN_DIR.glob("*.json") if p.name != "manifest.json")
    assert len(train_files) == 4
    for packet_path in train_files:
        assert isinstance(json.loads(packet_path.read_text(encoding="utf-8")), dict)
