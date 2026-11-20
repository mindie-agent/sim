"""Topology scoping, port coset contention, host root-complex, and HCCL bands.

[Role]
  Verify physical communication topology routing priorities (package -> node ->
  cross_node), cross-node ``(node_id, package_id)`` locality isolation,
  explicit link-scope validation, single-freeze generator ``flows`` handling,
  shared ``(node, physical_endpoint, rail, tx_or_rx)`` port coset contention,
  Kunpeng Host-HCCS solo ceiling + shared ``(node, root_complex_id, dir)``
  Engine contention, and piecewise ``CollectiveBandSpec`` lowering.

[Invariants]
  * Two dies on distinct nodes sharing ``package=0`` never match a
    ``scope="package"`` link.
  * Same-package pairs select ``package`` links over ``node`` links; unknown
    scopes fail loudly with ``ModelingError``.
  * Solo host transfers preserve full solo endpoint bandwidth without
    pre-dividing by configured die count; concurrent dies on one root complex
    contend on one shared ``(node, root_complex_id, direction)`` Engine row.
  * Distinct communicators sharing a ``PortKey`` serialize on one Engine row,
    while distinct physical rails overlap concurrently.
  * Piecewise ``CollectiveBandSpec`` selects exact disjoint payload bands,
    preserves zero-payload rendezvous, conserves Ring AllReduce ``2(P-1)B/P``,
    and validates integer AllToAllV peer matrices.

[Timeouts & Locks]
  Tier-0 pure in-memory + local JSON manifest checks (<2s); no locks or I/O
  side effects.

[Side-Effect Policy]
  Read-only over repository profile/observation JSON; constructs local Engines.
"""
from __future__ import annotations

import importlib
import json
import math
from pathlib import Path

import pytest

from mindie_sim import api as sdk_api
from mindie_sim.backends import ascend
from mindie_sim.prediction.collectives import (
    LoweredCollective,
    PairKey,
    PortKey,
    RootComplexKey,
    Topology,
    frame_wire_bytes,
    lower_collective,
    root_complex_resources_for_dies,
    select_band,
    submit_lowered_collective,
)
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.engine import Engine, Resource
from mindie_sim.prediction.execution import Execution
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.hardware_links import CollectiveBandSpec, Link
from mindie_sim.schema.parallel import ParallelSpec

pytestmark = pytest.mark.tier0

REPO_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_DIR = REPO_ROOT / "data" / "observations" / "campaign-202611-a2-a3"

SINGLE_DIE_SKU = "Ascend910B4"
DUAL_DIE_SMALL_SKU = "Ascend910_9362"
DUAL_DIE_LARGE_SKU = "Ascend910_9382"
DIRECT_MESH_SKU = "Ascend950DT_9582"
TINY_MODEL_ID = "qwen3-32b"


def _make_machine(sku: str, *, num_devices: int, ranks_per_node: int | None = None) -> Hardware:
    raw = {"sku": sku, "num_devices": num_devices}
    if ranks_per_node is not None:
        raw["ranks_per_node"] = ranks_per_node
    return Hardware.from_mapping(ascend(raw))


def _make_custom_machine(
    base_sku: str,
    *,
    num_devices: int,
    ranks_per_node: int,
    links: list[dict],
) -> Hardware:
    raw = ascend({"sku": base_sku, "num_devices": num_devices, "ranks_per_node": ranks_per_node})
    raw["links"] = links
    return Hardware.from_mapping(raw)


def test_cross_node_identical_package_id_cannot_use_package_scope_link():
    """Two nodes whose local dies both have package=0 must route cross-node."""
    raw = ascend({"sku": DUAL_DIE_SMALL_SKU, "num_devices": 4, "ranks_per_node": 2})
    base_dev = dict(raw["device"])
    # Explicitly assign identical per-node local package=0 on BOTH node 0 and node 1
    raw["dies"] = [
        {**base_dev, "rank": 0, "node": 0, "package": 0},
        {**base_dev, "rank": 1, "node": 0, "package": 0},
        {**base_dev, "rank": 2, "node": 1, "package": 0},
        {**base_dev, "rank": 3, "node": 1, "package": 0},
    ]
    raw["links"] = [
        {
            "name": "d2d_intra_package:shared_pkg0",
            "members": [0, 1, 2, 3],
            "capacity_bytes_per_ms": 200_000_000.0,
            "semantics": "per_port",
            "scope": "package",
        },
        {
            "name": "d2d_inter_node:0",
            "members": [0, 1, 2, 3],
            "capacity_bytes_per_ms": 50_000_000.0,
            "semantics": "per_port",
            "scope": "cross_node",
        },
    ]
    machine = Hardware.from_mapping(raw)
    topo = Topology(machine, ParallelSpec(tp=4))

    loc0 = topo.location(0)
    loc2 = topo.location(2)
    assert loc0["package"] == 0 and loc2["package"] == 0
    assert loc0["node"] == 0 and loc2["node"] == 1

    # (node=0, pkg=0) vs (node=1, pkg=0) MUST NOT satisfy scope="package"
    assert topo._scope_admits("package", 0, 1) is True
    assert topo._scope_admits("package", 0, 2) is False
    assert topo._scope_admits("node", 0, 2) is False
    assert topo._scope_admits("cross_node", 0, 2) is True

    same_pkg = topo.route(0, 1)
    cross_node = topo.route(0, 2)
    assert topo.link_scope(same_pkg[0][0]) == "package"
    assert topo.link_scope(cross_node[0][0]) == "cross_node"


def test_routing_priority_prefers_package_then_node_then_cross_node():
    """Dual-die cluster selects package link over node link over cross-node link."""
    machine = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=16, ranks_per_node=8)
    topo = Topology(machine, ParallelSpec(tp=16))

    # Even with only 2 dies on a dual-die package, package link beats node link
    small_dual = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=2, ranks_per_node=2)
    small_topo = Topology(small_dual, ParallelSpec(tp=2))
    same_pkg_2 = small_topo.route(0, 1)
    assert small_topo.link_scope(same_pkg_2[0][0]) == "package"
    assert same_pkg_2[0][0].startswith("d2d_intra_package")

    # 16-die dual-node hierarchy: same-pkg -> package, same-node diff-pkg -> node, diff-node -> cross_node
    pkg_path = topo.route(0, 1)
    node_path = topo.route(0, 2)
    xnode_path = topo.route(0, 8)
    assert topo.link_scope(pkg_path[0][0]) == "package"
    assert topo.link_scope(node_path[0][0]) == "node"
    assert topo.link_scope(xnode_path[0][0]) == "cross_node"


def test_direct_mesh_and_single_die_node_fabric_semantics():
    """Single-die node uses per_port node link; direct mesh keeps per_pair semantics."""
    single_node = _make_machine(SINGLE_DIE_SKU, num_devices=8, ranks_per_node=8)
    single_topo = Topology(single_node, ParallelSpec(tp=8))
    single_link_name = single_topo.route(0, 7)[0][0]
    single_link = single_topo.links[single_link_name]
    assert single_topo.link_scope(single_link_name) == "node"
    assert dict(dict(single_node.transfer_channels)[single_link.channel])["scope"] == "node"
    assert single_link.semantics == "per_port"

    mesh_node = _make_machine(DIRECT_MESH_SKU, num_devices=8, ranks_per_node=8)
    mesh_topo = Topology(mesh_node, ParallelSpec(tp=8))
    mesh_link_name = mesh_topo.route(0, 7)[0][0]
    mesh_link = mesh_topo.links[mesh_link_name]
    assert mesh_topo.link_scope(mesh_link_name) == "node"
    assert dict(dict(mesh_node.transfer_channels)[mesh_link.channel])["scope"] == "node"
    assert mesh_link.semantics == "per_pair"


def test_unknown_link_scope_rejected_explicitly():
    """Link constructor and Topology scope admission reject unknown scopes."""
    with pytest.raises(ModelingError, match="link_scope_unknown:.*bogus_scope"):
        Link(
            name="bad_link",
            members=(0, 1),
            capacity_bytes_per_ms=100_000_000.0,
            scope="bogus_scope",
        )

    machine = _make_machine(SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4)
    topo = Topology(machine, ParallelSpec(tp=4))
    with pytest.raises(ModelingError, match="link_scope_unknown:unknown"):
        topo._scope_admits("unknown", 0, 1)


def test_transport_plan_freezes_generator_flows_once():
    """Passing a single-use generator to transport_plan matches tuple input."""
    machine = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=4, ranks_per_node=4)
    topo = Topology(machine, ParallelSpec(tp=4))
    flows_tuple = (
        MessageFlow(0, 1, 65536),
        MessageFlow(0, 2, 131072),
        MessageFlow(2, 3, 262144),
    )

    plan_from_tuple = topo.transport_plan(
        flows_tuple,
        receive_capacity_bytes=524288,
        receive_buffer_id="rx_buf",
    )
    flow_gen = (flow for flow in flows_tuple)
    plan_from_gen = topo.transport_plan(
        flow_gen,
        receive_capacity_bytes=524288,
        receive_buffer_id="rx_buf",
    )
    assert plan_from_gen == plan_from_tuple
    assert tuple(fp.flow for fp in plan_from_gen.flows) == flows_tuple
    assert len(plan_from_gen.flows) == 3
    assert dict(plan_from_gen.receive_capacity) == {1: 524288.0, 2: 524288.0, 3: 524288.0}
    assert dict(plan_from_gen.receive_buffer_ids) == {1: "rx_buf", 2: "rx_buf", 3: "rx_buf"}


def test_host_bus_preserves_solo_ceiling_and_contends_per_root_complex():
    """Solo die keeps full solo ceiling; concurrent dies contend on shared RC."""
    machine = _make_machine(DUAL_DIE_LARGE_SKU, num_devices=16, ranks_per_node=8)
    solo_h2d = machine.host_solo_bw_gbps("h2d")
    rc_h2d = machine.host_root_complex_ceiling_gbps("h2d")
    assert solo_h2d is not None and rc_h2d is not None
    # Solo ceiling is never pre-divided by total die count
    assert solo_h2d > rc_h2d / 8.0
    assert math.isclose(machine.effective_host_channel_bw_gbps("h2d"), solo_h2d, rel_tol=1e-9)

    # Multi-node dies expose distinct (node, root_complex_id, direction) resources
    rc_resources = root_complex_resources_for_dies(machine, machine.dies)
    rc_names = {res.name for res in rc_resources}
    assert RootComplexKey(0, "rc0", "h2d").resource_name() in rc_names
    assert RootComplexKey(1, "rc0", "h2d").resource_name() in rc_names
    assert RootComplexKey(0, "rc0", "d2h").resource_name() in rc_names
    assert RootComplexKey(1, "rc0", "d2h").resource_name() in rc_names

    # Build Execution on a 2-node case and verify ALL unique node RCs register
    case = sdk_api.compose_case(
        TINY_MODEL_ID,
        hardware=DUAL_DIE_LARGE_SKU,
        options={"dp": 1, "tp": 16, "pp": 1, "ranks_per_node": 8, "layers": 1},
    )
    ex = Execution(case, CostModel(case))
    engine_res_names = set(ex.engine.resources)
    assert RootComplexKey(0, "rc0", "h2d").resource_name() in engine_res_names
    assert RootComplexKey(1, "rc0", "h2d").resource_name() in engine_res_names
    assert RootComplexKey(0, "rc0", "d2h").resource_name() in engine_res_names
    assert RootComplexKey(1, "rc0", "d2h").resource_name() in engine_res_names

    # Verify Engine contention: 1 solo die runs at solo_h2d; 8 concurrent dies cap at rc_h2d
    payload_bytes = 64_000_000.0
    solo_rate = solo_h2d * 1_000_000.0
    rc_rate = rc_h2d * 1_000_000.0
    rc_key = RootComplexKey(0, "rc0", "h2d").resource_name()

    eng_solo = Engine([Resource("d0", solo_rate, "bytes"), Resource(rc_key, rc_rate, "bytes")])
    ev_solo = eng_solo.submit("solo", {"d0": payload_bytes, rc_key: payload_bytes}, coupled=True)
    eng_solo.advance()
    assert math.isclose(ev_solo.time, payload_bytes / solo_rate, rel_tol=1e-9)

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


def test_shared_port_cosets_serialize_while_distinct_rails_overlap():
    """Two links sharing (node, endpoint, rail, dir) contend; distinct rails overlap."""
    cap_bytes_per_ms = 100_000_000.0  # 100 GB/s
    payload_bytes = 10_000_000.0  # 10 MB -> 0.1 ms solo

    shared_links = [
        {
            "name": "comm_a",
            "members": [0, 1],
            "capacity_bytes_per_ms": cap_bytes_per_ms,
            "semantics": "per_port",
            "scope": "node",
            "rail_id": "rail_shared",
            "endpoint_ids": {"0": "nic0", "1": "nic1"},
        },
        {
            "name": "comm_b",
            "members": [0, 2],
            "capacity_bytes_per_ms": cap_bytes_per_ms,
            "semantics": "per_port",
            "scope": "node",
            "rail_id": "rail_shared",
            "endpoint_ids": {"0": "nic0", "2": "nic2"},
        },
    ]
    shared_machine = _make_custom_machine(
        SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4, links=shared_links
    )
    shared_topo = Topology(shared_machine, ParallelSpec(tp=4))
    res_a = shared_topo.capacity_resources(shared_topo.links["comm_a"], 0, 1)
    res_b = shared_topo.capacity_resources(shared_topo.links["comm_b"], 0, 2)
    shared_tx = PortKey(0, "nic0", "rail_shared", "tx").resource_name()
    assert shared_tx in res_a and shared_tx in res_b

    eng_shared = Engine(shared_topo.resources())
    done_a = eng_shared.submit(
        "coset_a", {r: payload_bytes for r in res_a}, coupled=True
    )
    done_b = eng_shared.submit(
        "coset_b", {r: payload_bytes for r in res_b}, coupled=True
    )
    eng_shared.advance()
    assert math.isclose(max(done_a.time, done_b.time), 0.2, rel_tol=1e-9)

    disjoint_links = [
        {**shared_links[0], "rail_id": "rail_0"},
        {**shared_links[1], "rail_id": "rail_1"},
    ]
    disjoint_machine = _make_custom_machine(
        SINGLE_DIE_SKU, num_devices=4, ranks_per_node=4, links=disjoint_links
    )
    disjoint_topo = Topology(disjoint_machine, ParallelSpec(tp=4))
    dres_a = disjoint_topo.capacity_resources(disjoint_topo.links["comm_a"], 0, 1)
    dres_b = disjoint_topo.capacity_resources(disjoint_topo.links["comm_b"], 0, 2)
    assert set(dres_a).isdisjoint(set(dres_b))

    eng_disjoint = Engine(disjoint_topo.resources())
    ddone_a = eng_disjoint.submit(
        "rail0", {r: payload_bytes for r in dres_a}, coupled=True
    )
    ddone_b = eng_disjoint.submit(
        "rail1", {r: payload_bytes for r in dres_b}, coupled=True
    )
    eng_disjoint.advance()
    assert math.isclose(max(ddone_a.time, ddone_b.time), 0.1, rel_tol=1e-9)


def test_piecewise_collective_band_boundaries_and_overlap_rejection():
    """Exact payload band thresholds [0, 65537), [65537, 2097152), [2097152, None)."""
    machine = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=8, ranks_per_node=8)
    topo = Topology(machine, ParallelSpec(tp=8))
    group = (0, 2, 4, 6)

    b_zero = select_band("allreduce", group, topo, 0)
    b_64k = select_band("allreduce", group, topo, 65536)
    b_64k_plus = select_band("allreduce", group, topo, 65537)
    b_2m_minus = select_band("allreduce", group, topo, 2097151)
    b_2m = select_band("allreduce", group, topo, 2097152)

    assert b_zero.payload_min_inclusive == 0 and b_zero.payload_max_exclusive == 65537
    assert b_64k == b_zero
    assert b_64k_plus.payload_min_inclusive == 65537
    assert b_64k_plus.payload_max_exclusive == 2097152
    assert b_2m_minus == b_64k_plus
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
            name="bad_bands",
            members=(0, 1),
            capacity_bytes_per_ms=100_000_000.0,
            scope="node",
            protocol_bands=(overlap_a, overlap_b),
        )


def test_collective_lowering_ring_conservation_zero_payload_and_alltoallv():
    """Verify Ring AllReduce 2(P-1)B/P, zero-payload rendezvous, and AllToAllV matrix."""
    machine = _make_machine(DUAL_DIE_SMALL_SKU, num_devices=8, ranks_per_node=8)
    topo = Topology(machine, ParallelSpec(tp=8))
    group = (0, 2, 4, 6)
    api_bytes = 4 * 1024 * 1024

    lowered = lower_collective(
        "allreduce", group, topo, api_bytes, communicator_id="tp_comm", coset_id="c0"
    )
    assert isinstance(lowered, LoweredCollective)
    assert tuple(p.name for p in lowered.phases) == ("reduce_scatter", "all_gather")
    expected_per_rank = 2.0 * (len(group) - 1) * api_bytes / len(group)
    for _, sent_bytes in lowered.per_rank_payload_bytes:
        assert math.isclose(sent_bytes, expected_per_rank, rel_tol=1e-12)
    assert lowered.total_wire_bytes >= lowered.total_payload_bytes

    zero_low = lower_collective("allreduce", group, topo, 0)
    assert zero_low.rendezvous_required is True
    assert zero_low.phases == ()
    assert zero_low.total_wire_bytes == 0.0
    eng_zero = Engine(topo.resources())
    ready_map = {r: eng_zero.submit(f"r{r}", release=0.05 * idx) for idx, r in enumerate(group)}
    done_map = submit_lowered_collective(eng_zero, "zero_ar", zero_low, ready_map)
    eng_zero.advance()
    expected_rv = 0.05 * (len(group) - 1) + zero_low.collective_startup_us / 1000.0
    for r in group:
        assert math.isclose(done_map[r].time, expected_rv, rel_tol=1e-9)

    peer_matrix = (
        (1024, 2048, 0, 4096),
        (512, 2048, 1024, 0),
        (0, 4096, 512, 2048),
        (2048, 0, 1024, 4096),
    )
    a2av = lower_collective("alltoallv", group, topo, 8192, peer_matrix=peer_matrix)
    assert dict(a2av.self_copy_bytes_by_rank) == {0: 1024, 2: 2048, 4: 512, 6: 4096}
    assert dict(a2av.per_rank_payload_bytes) == {0: 6144.0, 2: 1536.0, 4: 6144.0, 6: 3072.0}
    frames, wire = frame_wire_bytes(4096, a2av.band)
    assert frames >= 1 and wire >= 4096.0

    with pytest.raises(ModelingError, match="alltoallv_peer_matrix_entry_invalid"):
        lower_collective(
            "alltoallv", (0, 2), topo, 1024, peer_matrix=((0, -1), (1024, 0))
        )


def test_startup_scope_charges_once_per_collective_phase_or_hop():
    """Startup latency charges strictly once per declared startup_scope."""
    def _run_scope(scope_mode: str) -> float:
        band = CollectiveBandSpec(
            collective="allreduce", stack_id="default", topology_scope="node",
            group_min=2, group_max=4, expansion_mode="ring",
            payload_min_inclusive=0, payload_max_exclusive=None,
            algorithm="ring", protocol="simple", startup_scope=scope_mode,
            startup_us=100.0, payload_bytes_per_frame=4096, wire_bytes_per_frame=4096,
            credit_bytes=0, evidence_id=f"test_{scope_mode}",
        )
        link = {
            "name": "fast_node",
            "members": [0, 1, 2],
            "capacity_bytes_per_ms": 1_000_000_000_000.0,
            "latency_us": 0.0,
            "semantics": "per_port",
            "scope": "node",
            "protocol_bands": [band.to_dict()],
        }
        machine = _make_custom_machine(
            SINGLE_DIE_SKU, num_devices=3, ranks_per_node=3, links=[link]
        )
        topo = Topology(machine, ParallelSpec(tp=3))
        low = lower_collective("allreduce", (0, 1, 2), topo, 4096)
        eng = Engine(topo.resources())
        ready = {r: eng.submit(f"r{r}") for r in (0, 1, 2)}
        done = submit_lowered_collective(eng, f"ar_{scope_mode}", low, ready)
        eng.advance()
        return float(done[0].time)

    t_collective = _run_scope("collective")
    t_phase = _run_scope("phase")
    t_hop = _run_scope("hop")
    # 3-rank ring allreduce has 1 collective, 2 phases, 4 sequential rounds (2 per phase)
    assert math.isclose(t_collective, 0.1, abs_tol=1e-4)
    assert math.isclose(t_phase, 0.2, abs_tol=1e-4)
    assert math.isclose(t_hop, 0.4, abs_tol=1e-4)


def test_pair_port_and_root_complex_key_validation():
    """Key dataclasses reject empty identifiers, self-loops, and invalid dirs."""
    with pytest.raises(ModelingError, match="pair_key_self_loop"):
        PairKey("hccs", 1, 1, 0)
    with pytest.raises(ModelingError, match="port_key_direction_invalid"):
        PortKey(0, "nic0", 0, "bidirectional")
    with pytest.raises(ModelingError, match="root_complex_key_direction_invalid"):
        RootComplexKey(0, "rc0", "tx")


def test_profile_catalog_invariants_and_train_campaign_packets():
    """All 16 hardware profiles preserve l2 buffer_units==1 and campaign packets parse."""
    ascend_mod = importlib.import_module("mindie_sim.backends.ascend")
    all_profiles = ascend_mod.profiles()
    assert len(all_profiles) == 16
    for sku_name, profile in all_profiles.items():
        buf_units = profile["device"].get("buffer_units") or {}
        assert buf_units.get("l2") == 1, f"{sku_name} broke buffer_units.l2 == 1"
        machine = _make_machine(sku_name, num_devices=2, ranks_per_node=2)
        assert len(machine.dies) == 2

    manifest_path = CAMPAIGN_DIR / "manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest.get("schema")
    train_files = sorted(p for p in CAMPAIGN_DIR.glob("*.json") if p.name != "manifest.json")
    assert len(train_files) == 4
    for packet_path in train_files:
        payload = json.loads(packet_path.read_text(encoding="utf-8"))
        assert isinstance(payload, dict) and payload
