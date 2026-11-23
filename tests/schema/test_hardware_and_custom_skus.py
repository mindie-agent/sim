"""Every registered and custom hardware SKU materializes its declared physics.

[Role] Each of the 16 Ascend profiles turns its source declaration into owners,
  buffers, pipe supply and precision peaks; SKU geometry, per-device overrides
  and transfer declarations reach scheduling, links and the Engine; neutral
  custom/private SKU templates compose runnable cases; SKU derivation scales
  only owned supplies.
[Invariants] Expected values come from each profile's ``source.declaration``
  times a per-ISA bytes-per-cycle table, or from relative overrides of the
  parsed baseline; ``buffer_units.l2 == 1`` per die; one MTE1 owner serializes
  its L0A/L0B direction modes; all GM paths share one HBM supply; overrides
  touch only their own layer/channel/placement and never mutate a profile.
[Timeouts & Locks] Tier 0 in-memory work plus ``tmp_path`` JSON; no locks.
[Side-Effect Policy] Writes only pytest temporary files; resets the
  process-local custom-SKU registry around every test.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
import math

import pytest

from mindie_sim import compose_case, predict, resolve_hardware_sku, target_operators
from mindie_sim.backends.ascend import profiles
from mindie_sim.catalog.ops import vector_op
import mindie_sim.custom_hardware as ch
from mindie_sim.hardware_derivation import derive_hardware_sku
from mindie_sim.hardware_derivation import EfficiencyCurve, apply_mte2_ceiling, coerce_curve
from mindie_sim.prediction.communication import MessageFlow
from mindie_sim.prediction.efficiency import analyze_network_efficiency, extract_hardware_peaks
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.operators.vector import TiledPointwise
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.topology import Topology
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.hardware import Device, Hardware, Link
from mindie_sim.schema.parallel import ParallelSpec

from _support.axes import ALL_SKUS, REPRESENTATIVE_SKUS, SKUS
from _support.hardware import (
    EXPECTED_L2_BYTES, MODEL, WORKLOAD, Matrix, OneMode, OwnerMoves, ReadMoves,
    declared_supply, hardware_case, isa_of, machine, op, run, sku_case,
)

pytestmark = pytest.mark.tier0

AIC_ROLES = {"mte1", "mte2_aic", "fixpipe"}
ISA = {
    "npu_arch_2201": {
        "pipes": {"mte1": 128, "mte2_aic": 256, "mte2_aiv": 128, "mte3": 64, "fixpipe": 128, "vector": 32},
        "modes": {"mte1_l0a": 256, "mte1_l0b": 128}, "mte1_split": 2, "mte2_split": 2,
        "peaks": {"int4": 4, "fp32": 1 / 4, "hf32": 1 / 2},
        "buffers": {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 131072, "ub": 196608},
    },
    "npu_arch_3510": {
        "pipes": {"mte1": 256, "mte3": 80, "fixpipe": 128, "vector": 64, "reg_data_writes": 2,
                  "reg_ub_accesses": 2, "reg_ub_stores": 1, "reg_alu_lanes": 128,
                  "reg_special_lanes": 64, "reg_reduce_lanes": 64, "reg_shuffle_lanes": 64},
        "modes": {"mte1_l1_ub": 128, "ub_copy": 256, "reg_issue_slots": 2}, "mte1_split": 1, "mte2_split": 1,
        "peaks": {"fp32": 1 / 16, "hf32": 1 / 2},
        "buffers": {"l1": 524288, "l0a": 65536, "l0b": 65536, "l0c": 262144, "ub": 253952},
    },
}
GIB, AMOUNT = 1024 ** 3, 1_000_000_000
TRANSFER_SKU, CUSTOM_MODEL = REPRESENTATIVE_SKUS["a3"], "deepseek-v3.2-prune-3l"
MOCK = {"sku": "MOCK_CONFIDENTIAL_NPU", "hbm_gb": 192, "hbm_bw_gbps": 6000, "cube_fp16_tflops": 1000}
DOCUMENTED_FIELDS = {"hbm_gb", "hbm_bw_gbps", "cube_fp16_tflops", "cube_fp8_tflops",
                     "intra_node_bw_gbps", "superpod_size", "inter_node_bw_util"}


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("MINDIE_SIM_CUSTOM_SKUS", raising=False)
    ch.clear_custom_hardware_skus()
    yield
    ch.clear_custom_hardware_skus()


def _owners(role: str, aic: int, aiv: int) -> int:
    return aic if role in AIC_ROLES or role.startswith("mte1") else aiv


# --- registered profiles -----------------------------------------------------------------
@pytest.mark.parametrize("sku", ALL_SKUS)
def test_profile_declaration_materializes_owners_buffers_supply_precision_and_links(sku):
    raw = profiles()[sku]
    decl = raw["source"]["declaration"]
    clock, aic, aiv, memory, hbm = declared_supply(sku)
    isa = ISA[isa_of(sku)]
    device = sku_case(sku).machine.dies[0]
    rates, modes, peaks = dict(device.rates), dict(device.service_modes), dict(device.compute_peaks)
    assert len(profiles()) == len(ALL_SKUS) == 16
    assert (device.memory_bytes, device.launch_us, device.compute_units) == (memory, 0, aic)
    assert rates["hbm"] == pytest.approx(hbm) and float(decl["hbm_peak_bw_gbps"]) * 1e9 == pytest.approx(hbm)
    buffers = dict(device.buffers)
    assert buffers.pop("l2") == EXPECTED_L2_BYTES[sku] == float(decl["l2_capacity_mib"]) * 2 ** 20 and buffers == isa["buffers"]
    units = dict(device.buffer_units)
    assert units["l2"] == 1 and units["ub"] == aiv and units["l1"] == aic
    for role, per_cycle in isa["pipes"].items():
        owners = _owners(role, aic, aiv)
        assert device.units(role) == owners and rates[role] == pytest.approx(per_cycle * owners * clock)
    for mode, per_cycle in isa["modes"].items():
        base = "mte1" if mode.startswith("mte1") else "vector"
        assert modes[mode] == (base, pytest.approx(per_cycle * _owners(mode, aic, aiv) * clock))
    for capped in (rates["mte2_aic"], modes["mte1_l0a"][1]):
        assert capped <= 256 * aic * clock * (1 + 1e-12)
    assert modes["mte1_l0a"][1] == pytest.approx(isa["mte1_split"] * modes["mte1_l0b"][1])
    assert rates["mte2_aic"] / aic == pytest.approx(isa["mte2_split"] * rates["mte2_aiv"] / aiv)
    for dtype, ratio in isa["peaks"].items():
        assert peaks[dtype] == pytest.approx(peaks["bf16"] * ratio)
    lanes = isa["pipes"]["vector"]
    assert decl["vec_execution_bytes_per_cycle_per_aiv"] == 4 * lanes
    assert decl["vector_fp32_lane_ops_per_s_per_die"] == pytest.approx(rates["vector"])
    efficiency = extract_hardware_peaks(sku_case(sku))
    assert (efficiency["vector_peak_ops_per_s"], efficiency["vector_fp32_fma_flops_per_s"],
            efficiency["vector_fp16_fma_flops_per_s"]) == (
        pytest.approx(rates["vector"]), pytest.approx(2 * rates["vector"]),
        pytest.approx(4 * rates["vector"]))
    for invocation in raw["device"].get("kernel_invocations", {}).values():
        assert invocation.get("provenance", {}).get("basis") != "user_reported_experience"
    pair = machine(sku, num_devices=2)
    assert pair.links and pair.dies_per_package == raw["dies_per_package"]
    assert pair.package_memory_bytes(0) == raw["dies_per_package"] * memory
    for link in pair.links:
        declared = raw["transfer_channels"][link.channel]
        assert link.members == (0, 1) and link.latency_us == pytest.approx(declared["alpha_us"])
        assert link.capacity_bytes_per_ms == pytest.approx(declared["effective_uni_bw_gbps"] * 1e6)


@pytest.mark.parametrize("sku", ALL_SKUS)
def test_vector_units_determine_finite_pointwise_waves(sku):
    raw = hardware_case()
    raw["machine"] = {"backend": "ascend", "sku": sku, "num_devices": 1}
    hardware = Hardware.from_mapping(raw["machine"])
    device, vector_units = hardware.dies[0], declared_supply(sku)[2]
    rows = vector_units + 1  # one row per unit plus a tail row: two masked waves
    operator = vector_op(op_id="tail", deps=(), stage=0, layer=0, requests=(), elements=rows,
                         ops_per_element=1, dtype="bf16", family="pointwise", phase="prefill",
                         recipe="unit_rows", shape=(rows, 1), read_bytes_override=0,
                         write_bytes_override=0)
    recipes = RecipeBook(recipes={"unit_rows": TiledPointwise(0, (), tile_rows=1, tile_cols=1)})
    result = operator_costs(raw, (operator,), recipes=recipes)["operators"][0]
    assert result["serviced"]["vector@r0:s0"] == 2 * vector_units
    minimum = dict(device.kernel_invocations).get("vector")
    assert result["duration_ms"] == pytest.approx(max(
        1000 * 2 * vector_units / dict(device.rates)["vector"], minimum.minimum_ms if minimum else 0))
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


@pytest.mark.parametrize("sku", ALL_SKUS)
def test_undeclared_host_has_no_launch_service_and_explicit_launch_is_additive(sku):
    raw = {"machine": {"backend": "ascend", "sku": sku}, "model": MODEL, "workload": WORKLOAD,
           "runtime": {}}
    kwargs = {"recipes": RecipeBook({"move": OneMode("vector", 1e6, 0)}), "isolated": False,
              "include_submission": True}
    operation = replace(op("vector-work", "move"), vector_ops=1e6)
    baseline = graph_costs(raw, (operation,), **kwargs)
    raw["machine"]["device"] = {"launch_us": 7}
    explicit = graph_costs(raw, (operation,), **kwargs)
    assert explicit["duration_ms"] - baseline["duration_ms"] == pytest.approx(.007)
    for role, work in baseline["execution"]["serviced"].items():
        assert work == 0 if role.startswith("host") else \
            explicit["execution"]["serviced"][role] == pytest.approx(work)


def test_non_ascend_efficiency_peaks_preserve_declared_fp32_vector_rate():
    peaks = extract_hardware_peaks(Case.from_mapping({
        "machine": {"backend": "nvidia", "sku": "NVIDIA_H200"},
        "model": MODEL, "workload": WORKLOAD, "runtime": {}}))
    assert peaks["vector_peak_ops_per_s"] == peaks["vector_fp32_fma_flops_per_s"] == pytest.approx(67e12)


# --- pipeline supply on the common Engine ----------------------------------------------
@pytest.mark.parametrize("sku", SKUS)
def test_one_mte1_owner_serializes_direction_modes_while_owners_overlap(sku):
    from mindie_sim.prediction.evaluation import EvaluationContext
    case = sku_case(sku)
    nominal = EvaluationContext.create(case, primitive_table=False)
    device = case.machine.dies[0]
    modes, units, base = dict(device.service_modes), device.units("mte1"), dict(device.rates)["mte1"]
    a_rate, b_rate = modes["mte1_l0a"][1] / units, modes["mte1_l0b"][1] / units
    ops = (op("a", "a", "stream-a"), op("b", "b", "stream-b"))
    same = run(case, ops, {"a": OneMode("mte1_l0a", AMOUNT, 0), "b": OneMode("mte1_l0b", AMOUNT, 0)}, evaluation=nominal)
    split = run(case, ops, {"a": OneMode("mte1_l0a", AMOUNT, 0), "b": OneMode("mte1_l0b", AMOUNT, 1)}, evaluation=nominal)
    assert same["duration_ms"] == pytest.approx((AMOUNT / a_rate + AMOUNT / b_rate) * 1000)
    assert split["duration_ms"] == pytest.approx(max(AMOUNT / a_rate, AMOUNT / b_rate) * 1000)
    # Mode bytes convert to base-pipe work through the owner rate.
    assert same["execution"]["serviced"]["mte1@r0:s0"] == pytest.approx(
        AMOUNT * base / modes["mte1_l0a"][1] + AMOUNT * base / modes["mte1_l0b"][1])
    placed = {e["task"].split(":")[-2]: e["placement"]["mte1@r0:s0"]
              for e in split["events"] if e["task"].endswith(":move")}
    assert sorted(placed.values()) == [(0,), (1,)]


@pytest.mark.parametrize("sku", SKUS)
def test_fixed_owner_cannot_borrow_idle_owners_and_overrides_rescale_it(sku):
    from mindie_sim.prediction.evaluation import EvaluationContext
    case = sku_case(sku)
    nominal = EvaluationContext.create(case, primitive_table=False)
    device = case.machine.dies[0]
    for role in ("mte1", "mte3", "fixpipe"):
        owners, rate = device.units(role), dict(device.rates)[role]
        one = run(case, (op("one", "one"),), {"one": OwnerMoves(role, 4e9, 1)}, evaluation=nominal)
        full = run(case, (op("full", "full"),), {"full": OwnerMoves(role, 4e9, owners)}, evaluation=nominal)
        assert full["duration_ms"] == pytest.approx(4e9 / rate * 1000)
        assert one["duration_ms"] == pytest.approx(owners * full["duration_ms"])
        assert one["execution"]["serviced"][f"{role}@r0:s0"] == pytest.approx(4e9)
    mte3, fix = dict(device.rates)["mte3"], dict(device.rates)["fixpipe"]
    tuned = sku_case(sku, {"rates": {"mte3": 2 * mte3}, "resource_units": {"fixpipe": 10}})
    tuned_nominal = EvaluationContext.create(tuned, primitive_table=False)
    for role, owners, rate in (("mte3", device.units("mte3"), 2 * mte3), ("fixpipe", 10, fix)):
        assert (tuned.machine.dies[0].units(role), dict(tuned.machine.dies[0].rates)[role]) == (
            owners, rate)
        assert run(tuned, (op(role, role),), {role: OwnerMoves(role, AMOUNT, 1)}, evaluation=tuned_nominal)["duration_ms"] \
            == pytest.approx(AMOUNT * owners / rate * 1000)


@pytest.mark.parametrize("sku", SKUS)
def test_reads_stores_and_fixpipe_writes_contend_on_one_shared_hbm(sku):
    from mindie_sim.prediction.evaluation import EvaluationContext
    case = sku_case(sku)
    nominal = EvaluationContext.create(case, primitive_table=False)
    device = case.machine.dies[0]
    roles = ("mte2_aic", "mte2_aiv", "mte3", "fixpipe")
    result = run(case, tuple(op(r, r, r) for r in roles),
                 {r: ReadMoves(r, device.units(r) * AMOUNT, device.units(r)) for r in roles},
                 evaluation=nominal)
    total = sum(device.units(r) for r in roles) * AMOUNT
    serviced = result["execution"]["serviced"]
    assert [k for k, v in serviced.items() if v and k.startswith("memory@")] == ["memory@r0:s0"]
    assert serviced["memory@r0:s0"] == pytest.approx(total) and "mte2@r0:s0" not in serviced
    assert result["duration_ms"] == pytest.approx(total / dict(device.rates)["hbm"] * 1000)


@pytest.mark.parametrize("sku", SKUS)
def test_direct_fp32_and_hf32_matrix_service_follow_declared_peaks(sku):
    from mindie_sim.prediction.evaluation import EvaluationContext
    case = sku_case(sku)
    nominal = EvaluationContext.create(case, primitive_table=False)
    peaks = dict(case.machine.dies[0].compute_peaks)
    durations = {dtype: run(case, (op(dtype, dtype),), {dtype: Matrix(dtype, 1e12)}, evaluation=nominal)["duration_ms"]
                 for dtype in ("fp32", "hf32", "bf16")}
    for dtype, duration in durations.items():
        assert duration == pytest.approx(1e12 / peaks[dtype] * 1000)
    ratios = ISA[isa_of(case.machine.dies[0].sku)]["peaks"]
    assert durations["fp32"] / durations["hf32"] == pytest.approx(ratios["hf32"] / ratios["fp32"])


def test_selected_prefill_uses_profile_copy_roles_and_input_readiness():
    raw = {"machine": {"backend": "ascend", "sku": REPRESENTATIVE_SKUS["a3"]}, "model": MODEL,
           "workload": WORKLOAD, "runtime": {"kernels": {"recipes": {"flash_attention_prefill": {
               "kind": "staged_gqa_prefill", "query_tile": 4, "key_tile": 4}}}}}
    attention = tuple(o for o in target_operators(raw, phase="prefill", tokens=5, context=3)
                      if o.family == "attention")
    assert len(attention) == 1
    result = graph_costs(raw, attention, input_ready_ms=7, isolated=False, trace_limit=200,
                         external_inputs={d: 11 for o in attention for d in o.deps})
    serviced = result["execution"]["serviced"]
    query, kv, output = 5 * 8 * 2, 2 * (7 + 8) * 8 * 2, 5 * 8 * 2  # 4x7 and 1x8 causal rectangles
    assert serviced["memory@r0:s0"] == pytest.approx(query + kv + output)
    assert serviced["mte2_aic@r0:s0"] == pytest.approx(query + kv)
    assert serviced["mte3@r0:s0"] == pytest.approx(output)
    assert serviced["mte1@r0:s0"] > 0 and "mte2@r0:s0" not in serviced
    assert all(event["start_ms"] >= 11 for event in result["events"] if event["demand"])


# --- per-device geometry overrides -----------------------------------------------------
def test_native_neutral_and_device_record_unit_overrides_preserve_role_ownership():
    sku = REPRESENTATIVE_SKUS["a5"]
    _, aic, aiv, _, _ = declared_supply(sku)
    for override, vector, mte2 in (({}, aiv, None), ({"aiv_count": 11}, 11, None),
                                   ({"resource_units": {"vector": 7}}, 7, None),
                                   ({"aiv_count": 11, "resource_units": {"vector": 7, "mte2": 3}}, 7, 3)):
        device = machine(sku, device=override).dies[0]
        assert (device.units("vector"), device.compute_units) == (vector, aic)
        assert mte2 is None or device.units("mte2") == mte2
    hardware = machine(sku, num_devices=3,
                       device={"aiv_count": 11, "resource_units": {"vector": 7, "mte2": 3}},
                       devices=[{"aiv_count": 5}, {"aiv_count": 5, "resource_units": {"vector": 2}},
                                {"resource_units": {"mte3": 4}}])
    assert [d.units("vector") for d in hardware.dies] == [5, 2, 7]
    assert [d.units("mte2") for d in hardware.dies] == [3, 3, 3] and hardware.dies[2].units("mte3") == 4
    assert Hardware.from_mapping(hardware.to_dict()) == hardware


# --- machine transfer declarations -----------------------------------------------------
def _fabric(**extra) -> Hardware:
    return machine(TRANSFER_SKU, num_devices=16, ranks_per_node=8, **extra)


def _channel(hardware, channel):
    return [link for link in hardware.links if link.channel == channel]


def _routed(hardware, source, destination, payload=25_000_000):
    topology = Topology(hardware, ParallelSpec(tp=16))
    hop, = topology.route_message(MessageFlow(source, destination, payload)).hops
    engine = Engine(topology.resources())
    task = engine.submit("arrival", after=(engine.submit("route", {hop.resource: hop.wire_bytes}),), latency=hop.latency_ms)
    engine.advance(until=(task,), first=False)
    assert engine.summary()["live_tasks"] == engine.summary()["live_programs"] == 0
    return topology.route(source, destination), hop, task.time


def test_channel_records_merge_framing_into_execution_without_mutating_profile():
    original = deepcopy(profiles()[TRANSFER_SKU]["transfer_channels"])
    framed = _fabric(transfer_channels={"d2d_inter_node": {
        "alpha_us": 0.0, "channel_semantics": "per_pair", "transaction_bytes": 64,
        "alignment_bytes": 32, "protocol_overhead_bytes": 8}})
    link, = _channel(framed, "d2d_inter_node")
    assert (link.latency_us, link.semantics) == (0.0, "per_pair")
    assert (link.transaction_bytes, link.alignment_bytes, link.protocol_overhead_bytes) == (64, 32, 8)
    assert profiles()[TRANSFER_SKU]["transfer_channels"] == original
    _, hop, time = _routed(framed, 0, 8)
    transactions = math.ceil(25_000_000 / 64)
    assert (hop.transaction_frame, hop.transactions) == ((64, 72), transactions)
    assert hop.wire_bytes == transactions * 72
    assert time == pytest.approx(hop.wire_bytes / link.capacity_bytes_per_ms)
    listed = _fabric(transfer_channels=[{"channel": "d2d_intra_node", "alpha_us": 7.5},
                                        {"channel": "d2d_inter_node", "uni_peak_bw_gbps": 50.0}])
    assert all(link.latency_us == 7.5 for link in _channel(listed, "d2d_intra_node"))
    assert [link.capacity_bytes_per_ms for link in _channel(listed, "d2d_inter_node")] == [50e6]


def test_bandwidth_units_converge_and_explicit_channel_beats_convenience_fields():
    def inter(**extra):
        return _channel(_fabric(**extra), "d2d_inter_node")[0].capacity_bytes_per_ms
    base = inter()
    # Canonical GB/s: x GB/s == x * 1e9 B/s; explicit B/s beats GB/s on its layer.
    assert inter(inter_node_bw_gbps=base / 1e6) == pytest.approx(base)
    assert inter(inter_node_bw_gbps=1, inter_node_bw_bytes_per_s=base * 1e3) == pytest.approx(base)
    assert inter(transfer_channels={"d2d_inter_node": {"effective_uni_bw_gbps": 50.0}},
                 inter_node_bw_gbps=200) == 50e6
    intra = _fabric(intra_node_bw_gbps=50)
    assert all(link.capacity_bytes_per_ms == 50e6 for link in _channel(intra, "d2d_intra_node"))
    assert _channel(intra, "d2d_inter_node")[0].capacity_bytes_per_ms == base
    explicit = [{"name": "declared", "members": list(range(16)), "capacity_gbps": 25,
                 "transaction_bytes": 64, "alignment_bytes": 32, "protocol_overhead_bytes": 8,
                 "latency_us": 2.0, "semantics": "per_port"}]
    link, = _fabric(links=explicit, inter_node_bw_gbps=200, transfer_channels={
        "d2d_inter_node": {"effective_uni_bw_gbps": 50.0}}).links
    assert (link.name, link.capacity_bytes_per_ms, link.latency_us) == ("declared", 25e6, 2.0)
    assert _fabric(links=[]).links == ()


@pytest.mark.parametrize("declaration", [
    [{"channel": "d2d_inter_node"}, {"channel": "d2d_inter_node"}],
    {"not_a_channel": {"effective_uni_bw_gbps": 1}},
    [{"channel": "d2d_inter_node", "not_a_field": 1}],
    [{"channel": "d2d_inter_node", "channel_semantics": "invented"}],
])
def test_duplicate_unknown_channel_and_unknown_fields_are_rejected(declaration):
    with pytest.raises(ModelingError):
        _fabric(transfer_channels=declaration)


def test_cross_node_bandwidth_and_latency_change_arrival_but_not_wire_or_route():
    base = _fabric()
    inter = _channel(base, "d2d_inter_node")[0]
    capacity, latency = inter.capacity_bytes_per_ms, inter.latency_us / 1000
    half = _fabric(inter_node_bw_bytes_per_s=capacity * 1e3 / 2)
    slow = _fabric(transfer_channels={"d2d_inter_node": {"alpha_us": 17}})
    (base_path, base_hop, base_t), (half_path, half_hop, half_t), (_, slow_hop, slow_t) = (
        _routed(hardware, 0, 8) for hardware in (base, half, slow))
    assert base_path == half_path and [hop[0] for hop in base_path] == ["d2d_inter_node:0"]
    assert base_hop.wire_bytes == half_hop.wire_bytes == slow_hop.wire_bytes
    assert (base_t, half_t, slow_t) == (pytest.approx(25e6 / capacity + latency),
                                        pytest.approx(2 * 25e6 / capacity + latency),
                                        pytest.approx(25e6 / capacity + 0.017))
    (_, intra_base, intra_base_t), (_, intra_half, intra_half_t) = (_routed(h, 0, 2) for h in (base, half))
    assert (intra_base.resource, intra_base.wire_bytes, intra_base_t) == (
        intra_half.resource, intra_half.wire_bytes, pytest.approx(intra_half_t))


def test_source_ports_and_explicit_shared_capacity_change_real_contention():
    def concurrent(hardware):
        topology = Topology(hardware, ParallelSpec(tp=16))
        engine, ends = Engine(topology.resources()), []
        for source, destination in ((0, 1), (1, 0)):
            hop, = topology.route_message(MessageFlow(source, destination, 448e6)).hops
            assert "d2d_intra_package:0" in hop.resource
            service = engine.submit(str(source), {hop.resource: hop.wire_bytes})
            ends.append(engine.submit(f"arrival{source}", after=(service,), latency=hop.latency_ms))
        engine.advance(until=tuple(ends), first=False)
        return max(event.time for event in ends), sum(engine.serviced.values())
    package = _channel(_fabric(), "d2d_intra_package")[0]
    one_way, latency = 448e6 / package.capacity_bytes_per_ms, package.latency_us / 1000
    base_t, base_wire = concurrent(_fabric())
    # Neutral semantics beats the legacy label when both are explicit.
    shared_t, shared_wire = concurrent(_fabric(transfer_channels={
        "d2d_intra_package": {"channel_semantics": "per_port", "semantics": "shared"}}))
    assert base_wire == shared_wire == 896e6
    assert (base_t, shared_t) == (pytest.approx(one_way + latency), pytest.approx(2 * one_way + latency))


# --- neutral custom / private SKU templates --------------------------------------------
@pytest.mark.parametrize("archetype", tuple(ch.HARDWARE_ARCHETYPE_TEMPLATES))
def test_exported_templates_are_documented_fresh_copies(archetype):
    exported, expected = ch.export_hardware_sku_template(archetype), ch.HARDWARE_ARCHETYPE_TEMPLATES[archetype]
    assert (exported["sku"], exported["archetype"]) == ("CUSTOM_SKU_TEMPLATE", archetype)
    assert {key: exported[key] for key in expected} == expected
    docs = exported["_field_docs"]
    assert DOCUMENTED_FIELDS | set(expected) <= set(docs) and all(isinstance(v, str) and v for v in docs.values())
    for entry in ("register_custom_hardware_sku", "compose_custom_case", "MINDIE_SIM_CUSTOM_SKUS"):
        assert entry in exported["_agent_usage_hint"]
    original, exported["hbm_gb"], docs["hbm_gb"] = docs["hbm_gb"], -1.0, "mutated"
    fresh = ch.export_hardware_sku_template(archetype)
    assert (fresh["hbm_gb"], fresh["_field_docs"]["hbm_gb"]) == (expected["hbm_gb"], original)
    compact = ch.export_hardware_sku_template(archetype, include_schema_docs=False)
    assert "_field_docs" not in compact and "_agent_usage_hint" not in compact
    with pytest.raises(ModelingError, match="unknown_hardware_archetype"):
        ch.export_hardware_sku_template("no_such_archetype")


def test_minimal_spec_materializes_parseable_hardware_with_derived_rates():
    built = ch.build_custom_hardware_machine(MOCK, num_devices=8)
    hardware = ch.parse_hardware(built)
    assert (hardware.sku, hardware.hardware_key, len(hardware.dies)) == (MOCK["sku"], MOCK["sku"], 8)
    assert (built["ranks_per_node"], built["dies_per_package"]) == (64, 2)
    assert [item["package"] for item in built["dies"]] == [0, 0, 1, 1, 2, 2, 3, 3]
    die = hardware.dies[0]
    assert die.memory_bytes == 192e9 and dict(die.rates)["hbm"] == pytest.approx(6000e9)
    assert die.peak("bf16") == die.peak("fp16") == pytest.approx(1000e12)
    assert die.peak("fp8") == die.peak("fp8_e4m3") == die.peak("int8") == pytest.approx(2000e12)
    channels = {link.channel for link in hardware.links}
    assert {"intra_package", "intra_node", "superpod", "inter_node"} <= channels and "inter_pod" not in channels
    low, high = (ch.build_custom_hardware_machine({**MOCK, "hbm_bw_gbps": bw}, num_devices=1) for bw in (3000, 6000))
    for role in ("mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe"):
        assert high["device"]["rates"][role] == pytest.approx(2 * low["device"]["rates"][role])
    pipes = {"mte1": 1234, "mte2": 4321, "mte3": 999, "fixpipe": 888}
    explicit = ch.parse_hardware(ch.build_custom_hardware_machine({
        **MOCK, "cube_fp8_tflops": 700, "cube_int4_tflops": 400, "_agent_usage_hint": "ignored",
        **{f"{role}_bw_gbps": gbps for role, gbps in pipes.items()}}, num_devices=1)).dies[0]
    assert (explicit.peak("fp8"), explicit.peak("int4")) == (pytest.approx(700e12), pytest.approx(400e12))
    for role in ("mte1", "mte2_aic", "mte2_aiv", "mte3", "fixpipe"):
        assert dict(explicit.rates)[role] == pytest.approx(pipes[role.split("_")[0]] * 1e9)
    assert ch.parse_hardware(ch.build_custom_hardware_machine({**MOCK, "fp4_tflops": 250}, num_devices=1)
                             ).dies[0].peak("int4") == pytest.approx(250e12)


@pytest.mark.parametrize("patch,count", [
    ({"hbm_gb": 0}, 4), ({"hbm_bw_gbps": 0}, 4), ({"cube_fp16_tflops": 0}, 4),
    ({"vector_fp32_tflops": 0}, 4), ({"ranks_per_node": 0}, 4), ({"intra_node_bw_util": 0.0}, 4),
    ({"inter_node_bw_util": 1.01}, 4), ({"intra_node_bw_gbps": -1.0}, 4),
    ({"archetype": "no_such_archetype"}, 4), ({"cube_fp8_tflops": -2}, 4),
    ({}, 0), ({}, -1), ({}, True), ({}, 1.5),
])
def test_invalid_specs_and_device_counts_raise_modeling_error(patch, count):
    with pytest.raises(ModelingError):
        ch.build_custom_hardware_machine({**MOCK, **patch}, num_devices=count)


def test_link_hierarchy_superpods_and_package_utilization():
    def links(spec, count, archetype=None):
        built = ch.build_custom_hardware_machine(spec, num_devices=count, archetype=archetype)
        return built, ch.parse_hardware(built).links
    _, two_node = links({**MOCK, "ranks_per_node": 2}, 8)
    node = [link for link in two_node if link.channel == "intra_node"]
    inter, = [link for link in two_node if link.channel == "inter_node"]
    assert len(node) == 4 and all(len(link.members) == 2 for link in node) and len(inter.members) == 8
    assert (node[0].capacity_bytes_per_ms, inter.capacity_bytes_per_ms) == (
        pytest.approx(800 * 0.85 * 1e6), pytest.approx(800 * 0.80 * 1e6))
    built, pods = links({**MOCK, "ranks_per_node": 2, "superpod_size": 4}, 8)
    assert {"superpod", "inter_pod"} <= {l.channel for l in pods} and "inter_node" not in {l.channel for l in pods}
    assert (built["superpod_size"], built["max_dies"]) == (4, 4)
    assert sorted(len(l.members) for l in pods if l.channel == "superpod") == [4, 4]
    _, server = links({}, 16, "ascend_dual_die_server")
    assert {"intra_package", "intra_4p", "intra_8p", "intra_node"} <= {l.channel for l in server}
    assert [len(l.members) for l in server if l.channel == "intra_4p"] == [8, 8]
    _, throttled = links({"intra_package_bw_util": 0.5}, 4, "ascend_dual_die_server")
    package = next(l for l in throttled if l.channel == "intra_package")
    assert package.bw_utilization == pytest.approx(0.5)
    assert package.capacity_bytes_per_ms == pytest.approx(package.effective_peak_capacity_bytes_per_ms * 0.5)
    tie, _ = links({**MOCK, "ranks_per_node": 16, "superpod_size": 64, "superpod_bw_gbps": 400.0,
                    "inter_node_bw_gbps": 50.0}, 64, "ascend_dual_die_server")
    names = [l["name"] for l in tie["links"] if len(l["members"]) == 64]
    assert "superpod:0" in names and all(name >= "superpod:0" for name in names)
    with pytest.raises(ModelingError):
        links({"intra_package_bw_util": 1.5}, 4, "ascend_dual_die_server")


def test_registry_file_and_environment_resolve_copies_without_overwriting(tmp_path, monkeypatch):
    def write(name, payload):
        path = tmp_path / name
        path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")
        return path
    spec = {"hbm_gb": 96, "hbm_bw_gbps": 3000, "cube_fp16_tflops": 500}
    profile = ch.register_custom_hardware_sku("MOCK_PRIVATE", spec, archetype="generic_accelerator",
                                              aliases=("MOCK_ALIAS",))
    assert (profile["sku"], profile["archetype"]) == ("MOCK_PRIVATE", "generic_accelerator")
    assert {"MOCK_PRIVATE", "mock_private", "MOCK_ALIAS", "mock_alias"} <= set(ch.list_custom_hardware_skus())
    ch.resolve_custom_hardware_sku("mock_private")["hbm_gb"] = 1
    assert ch.resolve_custom_hardware_sku("MOCK_ALIAS")["hbm_gb"] == 96
    assert ch.resolve_custom_hardware_sku("missing_sku") is None
    assert ch.unregister_custom_hardware_sku("mock_alias") is True and ch.resolve_custom_hardware_sku("MOCK_PRIVATE") is None
    assert ch.unregister_custom_hardware_sku("MOCK_PRIVATE") is False
    for bad, kwargs in (({**spec, "hbm_gb": 0}, {}), (spec, {"archetype": "bogus"})):
        with pytest.raises(ModelingError):
            ch.register_custom_hardware_sku("MOCK_BAD", bad, **kwargs)
    assert set(ch.load_custom_hardware_skus_from_file(write("direct.json", {"MOCK_FILE": {**spec, "hbm_gb": 80}}))) == {"MOCK_FILE"}
    assert ch.resolve_custom_hardware_sku("mock_file")["hbm_gb"] == 80
    nested = write("nested.json", {"skus": {"MOCK_A": spec, "MOCK_B": {**spec, "hbm_gb": 64, "aliases": ["MOCK_B2"]}}})
    assert set(ch.load_custom_hardware_skus_from_file(nested)) == {"MOCK_A", "MOCK_B"}
    assert ch.resolve_custom_hardware_sku("mock_b2")["hbm_gb"] == 64
    ch.clear_custom_hardware_skus()
    ch.register_custom_hardware_sku("MOCK_ENV", {**spec, "hbm_gb": 12})
    env = write("private.json", {"MOCK_ENV": {**spec, "hbm_gb": 99}, "MOCK_ENV_ONLY": {**spec, "hbm_gb": 32}})
    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(env))
    assert (ch.resolve_custom_hardware_sku("MOCK_ENV")["hbm_gb"], ch.resolve_custom_hardware_sku("MOCK_ENV_ONLY")["hbm_gb"]) == (12, 32)
    ch.clear_custom_hardware_skus()
    monkeypatch.setenv("MINDIE_SIM_CUSTOM_SKUS", str(tmp_path / "missing.json"))
    assert ch.list_custom_hardware_skus() == {}
    with pytest.raises(ModelingError):
        ch.load_custom_hardware_skus_from_file(write("bad.json", "{not json"))


def test_custom_cases_compose_predict_and_move_in_physical_direction():
    decode = {"phase": "decode", "tokens": 1, "context": 2048, "sequences": 1}
    case = ch.compose_custom_case(CUSTOM_MODEL, MOCK, tp=2, dp=2, ep=4, prompt_tokens=128, output_tokens=8,
                                  num_requests=2, max_num_seqs=2, max_num_batched_tokens=512, max_model_len=512)
    assert (case["machine"]["num_devices"], case["runtime"]["parallel"]["cp"]) == (4, 1)
    assert len(parse_case(case).machine.dies) == 4 and predict(case, scope="body", **decode)["duration_ms"] > 0
    assert analyze_network_efficiency(case, **decode)["operator_sum_ms"] > 0
    ch.register_custom_hardware_sku("MOCK_REG", {**MOCK, "sku": "MOCK_REG"}, aliases=("mock-reg",))
    assert resolve_hardware_sku("mock-reg") == "MOCK_REG"
    registered = compose_case(CUSTOM_MODEL, hardware="mock-reg", tp=2, dp=1, prompt_tokens=256,
                              output_tokens=16, inter_node_bw_util=0.75)
    assert predict(registered, scope="body", **decode)["duration_ms"] > 0
    single = ch.compose_custom_case(CUSTOM_MODEL, "MOCK_REG", tp=1, dp=1, hbm_gb=32)
    assert single["machine"]["device"]["memory_bytes"] == 32e9 and analyze_network_efficiency(single, **decode)["operator_sum_ms"] > 0
    inline = {**MOCK, "sku": "MOCK_INLINE", "archetype": "ascend_dual_die_server"}
    assert compose_case(CUSTOM_MODEL, hardware=inline, tp=2, dp=1)["machine"]["sku"] == "MOCK_INLINE"
    with pytest.raises(ModelingError, match="unknown_custom_hardware_sku"):
        ch.compose_custom_case(CUSTOM_MODEL, "MOCK_MISSING")
    with pytest.raises(ModelingError, match="num_devices"):
        ch.compose_custom_case(CUSTOM_MODEL, MOCK, num_devices=8)
    prefill = {"phase": "prefill", "tokens": 1024, "context": 1024, "sequences": 1}
    shape = {"prompt_tokens": 1024, "output_tokens": 16, "num_requests": 1, "max_num_seqs": 1, "max_num_batched_tokens": 2048}
    fast, slow = (predict(ch.compose_custom_case(CUSTOM_MODEL, {**MOCK, "cube_fp16_tflops": tf},
                                                 tp=1, dp=1, **shape), scope="body", **prefill)["duration_ms"]
                  for tf in (2000, 250))
    assert fast < slow
    nets = [ch.compose_custom_case(CUSTOM_MODEL, {**MOCK, "ranks_per_node": 2, "inter_node_bw_util": u},
                                   tp=8, dp=1, ep=8, **shape) for u in (1.0, 0.2)]
    high, low = (predict(net, scope="body", **prefill)["duration_ms"] for net in nets)
    high_eff, low_eff = (analyze_network_efficiency(net, **prefill) for net in nets)
    assert low > high and low_eff["total_link_bytes"] == high_eff["total_link_bytes"]
    assert low_eff["total_link_ideal_ms"] > high_eff["total_link_ideal_ms"]
    assert low_eff["network_link_util"] > high_eff["network_link_util"]


# --- pure hardware-SKU derivation ------------------------------------------------------
def test_core_scaling_touches_only_owned_supplies_and_roundtrips():
    base = machine(REPRESENTATIVE_SKUS["a5"])
    die0, half = base.dies[0], base.dies[0].compute_units // 2
    derived = derive_hardware_sku(base, sku="X", aic_count=half, memory_bytes=64 * GIB)
    die = derived.dies[0]
    assert (derived.sku, die.compute_units, die.memory_bytes) == ("X", half, 64 * GIB)
    for dtype in ("bf16", "fp8_e4m3"):
        assert dict(die.compute_peaks)[dtype] == pytest.approx(dict(die0.compute_peaks)[dtype] / 2)
    assert dict(die.resource_units)["vector"] == dict(die0.resource_units)["vector"] // 2
    assert (dict(die.rates)["vector"], dict(die.rates)["hbm"]) == (
        pytest.approx(dict(die0.rates)["vector"] / 2), pytest.approx(dict(die0.rates)["hbm"]))
    assert (dict(die.buffer_units)["l1"], dict(die.buffers)) == (half, dict(die0.buffers))
    assert [(l.name, l.capacity_bytes_per_ms) for l in derived.links] == [
        (l.name, l.capacity_bytes_per_ms) for l in base.links]
    assert die.capability.supports_native("matrix", "mxfp4", "fp32")
    assert die.peak("mxfp4") == pytest.approx(2 * die.peak("fp8_e4m3"))
    narrow = derive_hardware_sku(base, sku="X", aic_count=16, aiv_count=16)
    nd = narrow.dies[0]
    assert (nd.compute_units, dict(nd.resource_units)["vector"], dict(nd.buffer_units)["ub"]) == (16, 16, 16)
    parsed = Hardware.from_mapping(narrow.to_dict()).dies[0]
    assert (parsed.compute_peaks, parsed.resource_units) == (nd.compute_peaks, nd.resource_units)
    noop = derive_hardware_sku(derived, sku="X").dies[0]
    assert (noop.compute_units, noop.compute_peaks, noop.rates) == (die.compute_units, die.compute_peaks, die.rates)


def test_efficiency_curves_mte2_ceiling_and_core_mapping_bound_derivation():
    base = machine(REPRESENTATIVE_SKUS["a5"])
    curve = EfficiencyCurve("aic_count", ((8, 0.5), (32, 1.0)))
    derived = derive_hardware_sku(base, sku="X", aic_count=16, hbm_peak_bytes_per_s=2e12,
                                  hbm_efficiency=curve)
    assert curve.resolve(16) == pytest.approx(2 / 3)
    assert dict(derived.dies[0].rates)["hbm"] == pytest.approx(2e12 * 2 / 3)
    assert (curve.resolve(1), curve.resolve(64), coerce_curve(0.8).resolve(1)) == (0.5, 1.0, 0.8)
    for knots, axis, reason in ((((8, 0.5), (8, 0.6)), "aic_count", "knots_not_increasing"),
                                (((8, 0.0),), "aic_count", "knot_y_invalid"),
                                (((8, 0.5),), "tokens", "axis_unsupported")):
        with pytest.raises(ModelingError, match=f"efficiency_curve_{reason}"):
            EfficiencyCurve(axis, knots)
    capped = apply_mte2_ceiling({"mte2_aic": 80.0, "mte2_aiv": 40.0}, 60.0)
    assert (capped["mte2_aic"], capped["mte2_aiv"]) == (pytest.approx(40.0), pytest.approx(20.0))
    rates = dict(derive_hardware_sku(base, sku="X", aic_count=16, mte2_ceiling_bytes_per_s=1e12).dies[0].rates)
    assert rates["mte2_aic"] + rates["mte2_aiv"] <= 1e12 + 1e-3
    die = Device(rank=0, memory_bytes=GIB, compute_units=4, compute_peaks=(("bf16", 1e12),),
                 rates=(("hbm", 1e12), ("vector", 1e12)), resource_units=(("vector", 8),),
                 buffer_units=(("ub", 8),), buffers=(("ub", 1024),), launch_us=0.0)
    mixed = Hardware("tiny", (die, replace(die, rank=1, compute_units=8)),
                     (Link("fabric", 1e6, (0, 1), latency_us=0.0),), ranks_per_node=2)
    with pytest.raises(ModelingError, match="requires_per_device_core_mapping"):
        derive_hardware_sku(mixed, sku="H", aic_count=4)
