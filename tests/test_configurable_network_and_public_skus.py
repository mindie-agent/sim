"""Configurable multi-tier network, public SKU and A5 EP MC2 invariants.

[Role] Verify bandwidth utilization, latency overrides, 4P/8P/SuperPod link
  boundaries, public NVIDIA SKUs, and the A5 EP MC2 analytical model.
[Invariants] Halving utilization doubles transmitted service time while the
  declared peak is preserved; unity utilization keeps the historical Link dict
  unchanged; tier bottlenecks follow (ep - N) / ep; only public SKUs exist.
[Timeouts & Locks] In-memory Tier 0 tests with bounded models, no device or I/O.
[Side-Effect Policy] No writes outside pytest temporary state; no device init.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import pytest

from mindie_sim import compose_case, predict
from mindie_sim.backends.nvidia import profiles as nvidia_profiles
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.prediction.topology_tiers import (
    EP_TIER_PRESETS,
    analyze_ep_mc2_communication,
    compute_ep_effective_bandwidth,
    compute_speculative_accept_step,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Hardware, Link
from mindie_sim.schema.parallel import ParallelSpec


ROOT = Path(__file__).resolve().parents[1]
A3 = "Ascend910_9392"
A5_DT = "Ascend950DT_9581"
A5_PR = "Ascend950PR_9579"


def _ascend_machine(*, devices=16, ranks_per_node=8, **extra):
    return Hardware.from_mapping({
        "backend": "ascend", "sku": A3, "num_devices": devices,
        "ranks_per_node": ranks_per_node, **extra,
    })


def _link(machine, channel, members=None):
    candidates = [link for link in machine.links if link.channel == channel]
    if members is not None:
        candidates = [link for link in candidates
                      if tuple(link.members) == tuple(members)]
    assert candidates, f"missing link: {channel} {members}"
    return candidates[0]


def _arrival_ms(machine, source, destination, payload, tp=16):
    topology = Topology(machine, ParallelSpec(tp=tp))
    message = topology.route_message(MessageFlow(source, destination, payload))
    assert len(message.hops) == 1
    hop = message.hops[0]
    engine = Engine(topology.resources())
    service = engine.submit("route", {hop.resource: hop.wire_bytes})
    arrival = engine.submit("arrival", after=(service,), latency=hop.latency_ms)
    engine.advance(until=(arrival,), first=False)
    return arrival.time, hop


def test_link_to_dict_is_byte_compatible_at_unity_utilization():
    link = Link(name="fabric", capacity_bytes_per_ms=25_000_000,
                members=(0, 1), latency_us=2.0, channel="d2d_inter_node",
                semantics="per_port")
    historical = {
        "name": "fabric", "capacity_bytes_per_ms": 25_000_000,
        "members": (0, 1), "latency_us": 2.0, "channel": "d2d_inter_node",
        "semantics": "per_port",
    }
    assert link.to_dict() == historical
    assert link.effective_peak_capacity_bytes_per_ms == 25_000_000

    parsed = Link.from_mapping({"capacity_bytes_per_ms": 12_500_000,
                                "bw_utilization": 0.5, "members": [0, 1]})
    assert parsed.capacity_bytes_per_ms == 12_500_000
    assert parsed.effective_peak_capacity_bytes_per_ms == 25_000_000
    throttled = parsed.to_dict()
    assert throttled["bw_utilization"] == 0.5
    assert throttled["peak_capacity_bytes_per_ms"] == 25_000_000


def test_transfer_channel_utilization_and_latency_change_real_service():
    baseline = _ascend_machine()
    throttled = _ascend_machine(transfer_channels={
        "d2d_inter_node": {"bw_utilization": 0.5, "alpha_us": 17.0}})
    base_link = _link(baseline, "d2d_inter_node")
    half_link = _link(throttled, "d2d_inter_node")
    assert base_link.effective_peak_capacity_bytes_per_ms == 25_000_000
    assert half_link.effective_peak_capacity_bytes_per_ms == 25_000_000
    assert half_link.capacity_bytes_per_ms == pytest.approx(
        base_link.capacity_bytes_per_ms / 2)

    payload = 25_000_000
    base_ms, base_hop = _arrival_ms(baseline, 0, 8, payload)
    half_ms, half_hop = _arrival_ms(throttled, 0, 8, payload)
    assert half_hop.wire_bytes == base_hop.wire_bytes
    expected_base = payload / 25_000_000 + 0.005
    assert base_ms == pytest.approx(expected_base)
    assert half_ms == pytest.approx(payload / 12_500_000 + 0.017)


def test_machine_bandwidth_utilization_halves_capacity_and_preserves_peak():
    baseline = _ascend_machine()
    throttled = _ascend_machine(inter_node_bw_util=0.5,
                                inter_node_latency_us=9.0)
    base_link = _link(baseline, "d2d_inter_node")
    half_link = _link(throttled, "d2d_inter_node")
    assert half_link.effective_peak_capacity_bytes_per_ms == 25_000_000
    assert half_link.capacity_bytes_per_ms * 2 == pytest.approx(
        base_link.capacity_bytes_per_ms)
    assert half_link.latency_us == 9.0

    base_ms, _ = _arrival_ms(baseline, 0, 8, 25_000_000)
    half_ms, _ = _arrival_ms(throttled, 0, 8, 25_000_000)
    assert half_ms - base_ms == pytest.approx(1.0 + 0.009 - 0.005)


def test_compose_case_network_overrides_reach_parsed_links():
    def composed(util):
        return compose_case(
            "qwen3.5-35b-a3b", hardware="a3", tp=8, dp=2, ep=16,
            ranks_per_node=8, layers=1, prompt_tokens=32, output_tokens=4,
            concurrency=1, inter_node_bw_gbps=100.0,
            inter_node_bw_util=util, inter_node_latency_us=4.0,
        )

    baseline = parse_case(composed(1.0))
    half = parse_case(composed(0.5))
    base_link = _link(baseline.machine, "d2d_inter_node")
    half_link = _link(half.machine, "d2d_inter_node")
    assert base_link.effective_peak_capacity_bytes_per_ms == 100_000_000
    assert half_link.effective_peak_capacity_bytes_per_ms == 100_000_000
    assert half_link.capacity_bytes_per_ms == pytest.approx(
        base_link.capacity_bytes_per_ms / 2)
    assert half_link.latency_us == 4.0
    assert base_link.latency_us == 4.0

    base_ms, _ = _arrival_ms(baseline.machine, 0, 8, 25_000_000)
    half_ms, _ = _arrival_ms(half.machine, 0, 8, 25_000_000)
    assert half_ms - base_ms == pytest.approx(25_000_000 / 50_000_000
                                              - 25_000_000 / 100_000_000)


def test_a2_standard_preset_emits_node_and_inter_links_with_declared_latency():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": "Ascend910B4", "num_devices": 16,
        "topology_preset": "a2_standard",
    })
    node = _link(machine, "d2d_intra_node", range(8))
    inter = _link(machine, "d2d_inter_node")
    assert machine.ranks_per_node == 8
    assert node.capacity_bytes_per_ms == pytest.approx(196e6 * 0.70)
    assert node.effective_peak_capacity_bytes_per_ms == pytest.approx(196e6)
    assert node.latency_us == 2.0
    assert inter.capacity_bytes_per_ms == pytest.approx(25e6 * 0.80)
    assert inter.latency_us == 10.0
    assert len(inter.members) == 16
    assert Hardware.from_mapping(machine.to_dict()) == machine


def test_a5_pcie_16p_has_4p_8p_16p_and_inter_tiers():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": A5_PR, "num_devices": 32,
        "topology_preset": "a5_pcie_16p",
    })
    four = _link(machine, "d2d_intra_4p", range(4))
    eight = _link(machine, "d2d_intra_8p", range(8))
    sixteen = _link(machine, "d2d_intra_node", range(16))
    inter = _link(machine, "d2d_inter_node")
    assert machine.ranks_per_node == 16
    assert machine.dies_per_package == 2
    assert four.capacity_bytes_per_ms == pytest.approx(288e6 * 0.85)
    assert four.effective_peak_capacity_bytes_per_ms == pytest.approx(288e6)
    assert eight.capacity_bytes_per_ms == pytest.approx(128e6 * 0.85)
    assert sixteen.capacity_bytes_per_ms == pytest.approx(116e6 * 0.85)
    assert inter.capacity_bytes_per_ms == pytest.approx(50e6 * 0.80)
    assert len(inter.members) == 32

    topology = Topology(machine, ParallelSpec(tp=32))
    assert [hop[0] for hop in topology.route(0, 2)] == ["d2d_intra_4p:0"]
    assert [hop[0] for hop in topology.route(0, 6)] == ["d2d_intra_8p:0"]
    assert [hop[0] for hop in topology.route(0, 12)] == ["d2d_intra_node:0"]
    assert [hop[0] for hop in topology.route(0, 20)] == ["d2d_inter_node:0"]


def test_a5_pod_superpod_is_the_cross_node_link():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": A5_DT, "num_devices": 128,
        "topology_preset": "a5_pod",
    })
    node = _link(machine, "d2d_intra_node", range(64))
    inter = _link(machine, "d2d_inter_node")
    assert machine.ranks_per_node == 64
    assert node.capacity_bytes_per_ms == pytest.approx(800e6 * 0.85)
    assert inter.capacity_bytes_per_ms == pytest.approx(800e6 * 0.80)
    assert inter.latency_us == 4.5
    assert len(inter.members) == 128
    assert not [link for link in machine.links
                if link.channel == "d2d_inter_pod"]
    assert Hardware.from_mapping(machine.to_dict()) == machine


def test_a3_superpod_distinguishes_intra_and_cross_superpod_traffic():
    machine = Hardware.from_mapping({
        "backend": "ascend", "sku": A3, "num_devices": 768,
        "topology_preset": "a3_superpod",
    })
    node = _link(machine, "d2d_intra_node", range(16))
    superpod = _link(machine, "d2d_inter_node", range(384))
    pod = _link(machine, "d2d_inter_pod")
    assert node.capacity_bytes_per_ms == pytest.approx(392e6 * 0.80)
    assert superpod.capacity_bytes_per_ms == pytest.approx(392e6 * 0.70)
    assert superpod.latency_us == 3.5
    assert pod.capacity_bytes_per_ms == pytest.approx(50e6 * 0.70)
    assert pod.latency_us == 10.0
    assert len(pod.members) == 768

    topology = Topology(machine, ParallelSpec(tp=768))
    assert [hop[0] for hop in topology.route(0, 8)] == ["d2d_intra_node:0"]
    assert [hop[0] for hop in topology.route(0, 100)] == ["d2d_inter_node:0"]
    assert [hop[0] for hop in topology.route(0, 500)] == ["d2d_inter_pod:0"]
    assert Hardware.from_mapping(machine.to_dict()) == machine


def test_public_nvidia_profiles_are_exactly_the_four_released_skus():
    expected = {"NVIDIA_H20", "NVIDIA_H800", "NVIDIA_H200", "NVIDIA_B300"}
    assert set(nvidia_profiles()) == expected
    for sku in expected:
        profile = nvidia_profiles()[sku]
        assert profile["hbm_bw_gbps"] > 0
        assert profile["cube_fp16_tflops"] > 0
        assert profile["ranks_per_node"] == 8


@pytest.mark.parametrize(("alias", "sku", "memory_gib"), [
    ("h20", "NVIDIA_H20", 96),
    ("h800", "NVIDIA_H800", 80),
    ("h200", "NVIDIA_H200", 141),
    ("b300", "NVIDIA_B300", 288),
])
def test_public_nvidia_skus_compose_parse_and_predict(alias, sku, memory_gib):
    case = compose_case(
        "qwen3.5-35b-a3b", hardware=alias, tp=8, dp=2, ep=8,
        layers=1, prompt_tokens=32, output_tokens=4, concurrency=1,
    )
    parsed = parse_case(case)
    assert parsed.machine.sku == sku
    assert len(parsed.machine.dies) == 16
    assert parsed.machine.dies[0].memory_bytes == memory_gib * 1024**3
    node = _link(parsed.machine, "d2d_intra_node", range(8))
    inter = _link(parsed.machine, "d2d_inter_node")
    profile = nvidia_profiles()[sku]
    assert node.capacity_bytes_per_ms == pytest.approx(
        profile["intra_node_bw_gbps"] * 1e6)
    assert inter.capacity_bytes_per_ms == pytest.approx(
        profile["inter_node_bw_gbps"] * 1e6)
    if profile["dies_per_package"] > 1:
        package = _link(parsed.machine, "d2d_intra_package", range(2))
        assert package.capacity_bytes_per_ms == pytest.approx(
            profile["intra_package_bw_gbps"] * 1e6)

    result = predict(case)
    assert result["status"] == "modeled"
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


def test_nvidia_machine_utilization_halves_service_and_preserves_peak():
    baseline = Hardware.from_mapping({
        "backend": "nvidia", "sku": "NVIDIA_H200", "num_devices": 16,
        "ranks_per_node": 8,
    })
    throttled = Hardware.from_mapping({
        "backend": "nvidia", "sku": "NVIDIA_H200", "num_devices": 16,
        "ranks_per_node": 8, "intra_node_bw_util": 0.5,
    })
    base = _link(baseline, "d2d_intra_node", range(8))
    half = _link(throttled, "d2d_intra_node", range(8))
    assert base.effective_peak_capacity_bytes_per_ms == pytest.approx(450e6)
    assert half.effective_peak_capacity_bytes_per_ms == pytest.approx(450e6)
    assert half.capacity_bytes_per_ms == pytest.approx(
        base.capacity_bytes_per_ms / 2)
    assert Hardware.from_mapping(throttled.to_dict()) == throttled


def test_a5_ep_accept_step_matches_spreadsheet():
    assert compute_speculative_accept_step(0.75, 3) == pytest.approx(2.734375)
    assert compute_speculative_accept_step(1.0, 3) == 4.0
    assert compute_speculative_accept_step(0.0, 0) == 1.0
    with pytest.raises(ModelingError):
        compute_speculative_accept_step(1.5, 3)
    with pytest.raises(ModelingError):
        compute_speculative_accept_step(0.5, -1)


def test_a5_ep_effective_bandwidth_matches_spreadsheet():
    pcie32 = compute_ep_effective_bandwidth(32, preset="a5_pcie_16p")
    assert pcie32["outbound_ratio"] == pytest.approx(0.5)
    assert pcie32["raw_effective_bw_gbps"] == pytest.approx(100.0)
    assert pcie32["effective_bw_gbps"] == pytest.approx(100.0)
    assert len(pcie32["tier_breakdown"]) == 4
    assert [row["name"] for row in pcie32["tier_breakdown"]] == [
        "intra_4p", "intra_8p", "intra_16p", "inter_node"]
    assert [row["outbound_ratio"] for row in pcie32["tier_breakdown"]] == [
        pytest.approx(3 / 32), pytest.approx(28 / 32),
        pytest.approx(24 / 32), pytest.approx(16 / 32)]

    assert compute_ep_effective_bandwidth(
        8, preset="a5_pcie_16p")["raw_effective_bw_gbps"] == pytest.approx(256.0)
    assert compute_ep_effective_bandwidth(
        16, preset="a5_pcie_16p")["raw_effective_bw_gbps"] == pytest.approx(
            128 / 0.75)
    assert compute_ep_effective_bandwidth(
        64, preset="a5_ub_pod")["effective_bw_gbps"] == pytest.approx(
            800 * 64 / 63)
    assert compute_ep_effective_bandwidth(
        512, preset="a3_superpod")["raw_effective_bw_gbps"] == pytest.approx(
            50 / 0.25)
    throttled = compute_ep_effective_bandwidth(
        32, preset="a5_pcie_16p", bw_utilization=0.8)
    assert throttled["effective_bw_gbps"] == pytest.approx(80.0)
    assert throttled["raw_effective_bw_gbps"] == pytest.approx(100.0)

    custom = compute_ep_effective_bandwidth(
        16, [{"max_ranks": 8, "bw_gbps": 100.0},
             {"max_ranks": 64, "bw_gbps": 25.0}])
    assert custom["raw_effective_bw_gbps"] == pytest.approx(25 / 0.5)

    with pytest.raises(ModelingError):
        compute_ep_effective_bandwidth(1, preset="a5_pcie_16p")
    with pytest.raises(ModelingError):
        compute_ep_effective_bandwidth(8, preset="a5_pcie_16p",
                                       bw_utilization=1.5)
    with pytest.raises(ModelingError):
        compute_ep_effective_bandwidth(8, preset="not_a_preset")
    assert set(EP_TIER_PRESETS) == {"a5_pcie_16p", "a5_ub_pod", "a3_superpod"}


def test_ep_mc2_communication_matches_analytic_geometry():
    analysis = analyze_ep_mc2_communication()
    assert analysis["accept_step"] == pytest.approx(2.734375)
    assert analysis["startup_latency_us"] == pytest.approx(11.5)
    assert analysis["dispatch_bytes_per_layer"] == pytest.approx(
        32 * 7168 * 8 * (33 / 32) * 1.2)
    assert analysis["combine_bytes_per_layer"] == pytest.approx(
        32 * 7168 * 8 * 2.0 * 1.2)
    assert len(analysis["rows"]) == 8 * 4

    row = next(item for item in analysis["rows"]
               if item["ep"] == 64 and item["bw_utilization"] == 0.75)
    effective = 800 * 64 / 63 * 0.75
    dispatch_us = 11.5 + analysis["dispatch_bytes_per_layer"] / (effective * 1e9) * 1e6
    combine_us = 11.5 + analysis["combine_bytes_per_layer"] / (effective * 1e9) * 1e6
    total_ms = (dispatch_us + combine_us) * 61 / 1000
    assert row["effective_bw_gbps"] == pytest.approx(effective)
    assert row["dispatch_us"] == pytest.approx(dispatch_us)
    assert row["combine_us"] == pytest.approx(combine_us)
    assert row["total_comm_ms"] == pytest.approx(total_ms)
    assert row["tpot_comm_ms"] == pytest.approx(total_ms / 2.734375)

    no_draft = analyze_ep_mc2_communication(draft_tokens=0, ep_values=(2,))
    assert no_draft["accept_step"] == 1.0
    pcie = analyze_ep_mc2_communication(
        ep_values=(16,), preset="a5_pcie_16p",
        bw_utilizations=(1.0,), dispatch_dtype="fp8", combine_dtype="int8")
    assert pcie["rows"][0]["total_comm_ms"] == pytest.approx(
        (11.5 + (32 * 7168 * 8 * 1.0 * 1.2) / (128 / 0.75 * 1e9) * 1e6
         + 11.5 + (32 * 7168 * 8 * 1.0 * 1.2) / (128 / 0.75 * 1e9) * 1e6)
        * 61 / 1000)


def test_no_non_public_hardware_codenames_in_public_inputs():
    forbidden = re.compile(
        r"\b(DV100|DV120|DV121|DV122|DV200|DAYU|YINHE|UBX|ZJ[-_]?UBX|"
        r"HWJ[-_]?590|HWJ[-_]?690|R200[-_]?CPX|A6)\b",
        re.IGNORECASE,
    )
    files = sorted((ROOT / "mindie_sim" / "backends" / "profiles").glob("*.json"))
    files += sorted((ROOT / "mindie_sim" / "backends").glob("*.py"))
    files.append(ROOT / "mindie_sim" / "compose.py")
    assert files
    matches = []
    for path in files:
        text = path.read_text(encoding="utf-8")
        matches.extend(f"{path.name}:{match.group(0)}"
                       for match in forbidden.finditer(text))
    assert matches == []


def test_nvidia_profile_json_is_well_formed_and_uses_declared_units():
    path = ROOT / "mindie_sim" / "backends" / "profiles" / "nvidia.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"NVIDIA_H20", "NVIDIA_H800", "NVIDIA_H200", "NVIDIA_B300"}
    assert data["NVIDIA_H20"]["intra_node_bw_gbps"] == 450
    assert data["NVIDIA_H800"]["intra_node_bw_gbps"] == 200
    assert data["NVIDIA_H200"]["hbm_bw_gbps"] == 4800
    assert data["NVIDIA_B300"]["intra_package_bw_gbps"] == 900
    assert data["NVIDIA_B300"]["cube_int4_tflops"] == 15000
    assert all(not math.isnan(value) for row in data.values()
               for value in row.values() if isinstance(value, (int, float)))


def test_ascend_peak_capacity_override_and_nvidia_channel_merge_and_sweep():
    from mindie_sim import compose_case
    from mindie_sim.backends.ascend import materialize as ascend_materialize
    from mindie_sim.backends.nvidia import materialize as nvidia_materialize
    from mindie_sim.sweep import _materialize_machine_for_devices

    asc = ascend_materialize({
        "sku": A3, "num_devices": 16, "ranks_per_node": 8,
        "transfer_channels": {"d2d_inter_node": {"peak_capacity_bytes_per_ms": 50000000.0, "bw_utilization": 0.5}},
    })
    assert asc["transfer_channels"]["d2d_inter_node"]["peak_capacity_bytes_per_ms"] == pytest.approx(50000000.0)
    assert asc["transfer_channels"]["d2d_inter_node"]["effective_uni_bw_gbps"] == pytest.approx(25.0)
    asc_inter = next(link for link in asc["links"] if link["channel"] == "d2d_inter_node")
    assert asc_inter["capacity_bytes_per_ms"] == pytest.approx(25000000.0)

    nv = nvidia_materialize({
        "sku": "NVIDIA_H200", "num_devices": 16, "ranks_per_node": 8,
        "transfer_channels": {"d2d_inter_node": {"bw_utilization": 0.5, "latency_us": 3.5}},
    })
    assert nv["transfer_channels"]["d2d_inter_node"]["semantics"] == "per_port"
    assert nv["transfer_channels"]["d2d_inter_node"]["latency_us"] == pytest.approx(3.5)
    nv_inter = next(link for link in nv["links"] if link["channel"] == "d2d_inter_node")
    assert nv_inter["capacity_bytes_per_ms"] == pytest.approx(25000000.0)
    assert nv_inter["latency_us"] == pytest.approx(3.5)

    nv_case = compose_case(
        "deepseek-v3.2-prune-3l",
        hardware="H200", tp=1, dp=1, prompt_tokens=16, output_tokens=2,
    )
    resized = _materialize_machine_for_devices(parse_case(nv_case), nv_case, 16)
    assert len(resized.dies) == 16
    assert any(link.channel == "d2d_inter_node" for link in resized.links)

    inverted = compute_ep_effective_bandwidth(
        16,
        [{"name": "inner", "max_ranks": 8, "bw_gbps": 5.0},
         {"name": "outer", "max_ranks": 64, "bw_gbps": 100.0}],
    )
    assert inverted["tier_breakdown"][0]["bottleneck"] is True
    assert inverted["raw_effective_bw_gbps"] == pytest.approx(5.0 / (7 / 16))

    nv_bw_and_util = nvidia_materialize({
        "sku": "NVIDIA_H200", "num_devices": 16, "ranks_per_node": 8,
        "inter_node_bw_gbps": 100.0,
        "transfer_channels": {"d2d_inter_node": {"bw_utilization": 0.5}},
    })
    nv_inter2 = next(link for link in nv_bw_and_util["links"] if link["channel"] == "d2d_inter_node")
    assert nv_inter2["capacity_bytes_per_ms"] == pytest.approx(50e6)

    nv_explicit_links = Hardware.from_mapping({
        "backend": "nvidia", "sku": "NVIDIA_H200", "num_devices": 16, "ranks_per_node": 8,
        "inter_node_bw_util": 0.5,
        "links": [{"name": "inter:0", "members": list(range(16)), "capacity_bytes_per_ms": 100e6,
                   "latency_us": 2.0, "channel": "d2d_inter_node"}],
    })
    assert nv_explicit_links.links[0].capacity_bytes_per_ms == pytest.approx(50e6)

    alias_mapping = compose_case(
        "deepseek-v3.2-prune-3l",
        hardware={"sku": "a5-server-pr-pcie-16p", "inter_node_bw_util": 0.7},
        tp=1, dp=1, prompt_tokens=16, output_tokens=2,
    )
    assert alias_mapping["machine"]["sku"] == A5_PR
    assert alias_mapping["machine"]["topology_preset"] == "a5_pcie_16p"

    custom_rpn = compose_case(
        "deepseek-v3.2-prune-3l",
        hardware={"sku": A3, "topology_preset": "a3_superpod", "ranks_per_node": 8},
        tp=1, dp=1, prompt_tokens=16, output_tokens=2,
    )
    resized_rpn = _materialize_machine_for_devices(parse_case(custom_rpn), custom_rpn, 16)
    assert resized_rpn.ranks_per_node == 8

    asc_pkg = Hardware.from_mapping({
        "backend": "ascend", "sku": A3, "num_devices": 2, "ranks_per_node": 2,
        "intra_package_bw_gbps": 500.0,
    })
    pkg_link = _link(asc_pkg, "d2d_intra_package")
    assert pkg_link.effective_peak_capacity_bytes_per_ms == pytest.approx(500e6)

    nv_cal = nvidia_materialize({
        "sku": "NVIDIA_H200", "num_devices": 16, "ranks_per_node": 8,
        "transfer_channels": {"d2d_inter_node": {"calibrated_uni_bw_gbps": 80.0, "bw_utilization": 0.5}},
    })
    nv_cal_inter = next(link for link in nv_cal["links"] if link["channel"] == "d2d_inter_node")
    assert nv_cal_inter["capacity_bytes_per_ms"] == pytest.approx(40e6)


