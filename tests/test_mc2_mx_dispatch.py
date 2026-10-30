"""Actual ordinary graphs preserve the MX producer and asymmetric return rows."""
from copy import deepcopy
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.mc2_phase import contract
from mindie_sim.prediction.operators.mc2_mx_quant import mx32_work
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_native_mc2_graph import case as base_case
from test_joint_collective_execution import wire


def case(quant="mx", width=512):
    raw = base_case()
    raw["model"].update(hidden_size=width, head_dim=width // 4,
                        intermediate_size=width, moe_intermediate_size=256)
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e9
    raw["runtime"]["kernels"]["native_moe_protocol"]["dispatch_quant"] = quant
    parsed = parse_case(raw)
    quantization = QuantSpec(
        "explicit-mx32", weight_format="bf16", expert_weight_format="mxfp4",
        activation_format="fp8_e4m3", expert_block=32, weight_block=(32, 32),
        scale_format="fp8_e8m0", scale_bytes=1, activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2,
        exclusions=("router", "norm"))
    return replace(parsed, model=replace(parsed.model, quant=quantization))


def graphs(raw, sequences=(1, 1)):
    return target_graphs(raw, phase="decode", tokens=1, sequences=sequences,
                         context=128, scope="body")


def protocol_only(ops):
    result = {}
    for rank, graph in ops.items():
        selected = [o for o in graph if "device_kernel_group" in dict(o.metadata)]
        names = {o.op_id for o in selected}
        result[rank] = tuple(replace(o, deps=tuple(d for d in o.deps if d in names))
                             for o in selected)
    return result


@pytest.mark.parametrize("width", [256, 512, 1536, 5120])
def test_mx_work_from_full_register_operations_and_group32_scales(width):
    work = mx32_work(width)
    blocks, scales = width // 256, width // 32
    # Independent vector-element count: one used BF16 constant; AND+pair max+
    # eight 16-lane reductions; seven scale constants+13 operations/scale;
    # multiply+two lane permutations+BF16→FP32→FP8 for every input element.
    assert work == {"max_exponent": 128 + blocks * (2 * 128 + 128 + 8 * 15),
                    "scale_construction": 896 + 13 * scales,
                    "rescale_cast": 5 * width}


def test_ordinary_consumer_has_one_quantized_producer_and_keeps_down_quantization():
    raw = case()
    before = deepcopy(raw)
    selected, plain = graphs(raw), graphs(case("none"))
    ops = {o.op_id: o for o in selected[0]}
    assert raw == before
    assert "s0.l0.expert_up.quantize" not in ops
    assert "s0.l0.expert_up.quantize" in {o.op_id for o in plain[0]}
    assert "s0.l0.expert_down.quantize" in ops
    up = ops["s0.l0.expert_up"]
    plain_up = next(o for o in plain[0] if o.op_id == up.op_id)
    assert (up.flops, up.mte2_bytes, up.mte3_bytes) == (
        plain_up.flops, plain_up.mte2_bytes, plain_up.mte3_bytes)
    assert ops["s0.l0.expert_down"] == next(o for o in plain[0]
                                          if o.op_id == "s0.l0.expert_down")
    assert up.deps == ("s0.l0.dispatch",)
    assert dict(up.metadata)["activation_scale_producer"] == up.deps[0]
    raw_phase = dict(ops["s0.l0.mc2_dispatch.send_read_quantize"].metadata)["mc2_phase"]
    plan = contract(raw_phase, 4)
    send = next(p for p in plan.phases if p.name == "send_read_quantize")
    assert dict(send.work)["vector"] == 2 * sum(mx32_work(512).values())
    assert dict(send.work)["mte2_aiv"] == 2 * 512 * 2
    assert plan.layout.scale_bytes == 16 and plan.layout.packet_bytes == 1024
    assert dict(up.metadata)["activation_scale_bytes"] == 32
    result = graph_costs(raw, selected, input_ready_ms=7, external_inputs={"analysis_input": 7})
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert min(row["ready_ms"] for row in result["timeline"]) >= 7


def test_quantized_consumer_waits_for_late_route_and_received_scale_publication():
    raw = case()
    selected = graphs(raw)
    selected[0] = tuple(replace(o, deps=o.deps + ("late_route",))
                        if o.op_id == "s0.l0.pack" else o for o in selected[0])
    result = graph_costs(raw, selected, external_inputs={"analysis_input": 0, "late_route": 17},
                         isolated=False)
    rows = {r["id"]: r for r in result["timeline"]}
    receive = rows["r0:s0.l0.dispatch"]
    consumer = rows["r0:s0.l0.expert_up"]
    assert receive["ready_ms"] >= 17
    assert consumer["ready_ms"] >= receive["complete_ms"]
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


@pytest.mark.parametrize("sequences", [(1, 0), (0, 3), (1, 3)])
def test_nonuniform_idle_source_keeps_quant_counts_and_return_transpose(sequences):
    raw = case(width=5120)
    declared = graphs(raw, sequences)
    protocol = protocol_only(declared)
    result = graph_costs(raw, protocol, isolated=False)
    # Each row is framed before fractions: MX5120→6144B, BF16return→11264B.
    # Two experts assignments per token, one half remote. Count/status always
    # four32B records/source, including zero-count source ranks.
    expected = sum(sequences) * 2 * (6144 + 11264) / 2 + 2 * 4 * 32 / 2
    assert sum(wire(result).values()) == pytest.approx(expected)
    quant = 0
    for rank, ops in declared.items():
        for op in ops:
            phase = dict(op.metadata).get("mc2_phase", {})
            if phase.get("phase") == "send_read_quantize":
                ledger = contract(phase, 4)
                send = next(p for p in ledger.phases if p.name == "send_read_quantize")
                quant += sum(dict(values).get("vector", 0) for name, values in send.steps
                             if name in mx32_work(5120))
                assert phase["source_assignments"] == sequences[rank] * 2
    assert quant == sum(sequences) * 2 * sum(mx32_work(5120).values())
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0


@pytest.mark.parametrize("role", ["vector", "hbm", "mte2_aiv", "mte3", "link"])
def test_mx_protocol_physical_rates_affect_time_without_changing_work(role):
    times, services = [], []
    raw = case(width=5120)
    for factor in (.5, 1, 2):
        if role == "link":
            link = replace(raw.machine.links[0], capacity_bytes_per_ms=100 * factor)
            modified = replace(raw, machine=replace(raw.machine, links=(link,)))
        else:
            devices = tuple(replace(d, rates=tuple({**dict(d.rates), role: 1e7 * factor}.items()))
                            for d in raw.machine.dies)
            modified = replace(raw, machine=replace(raw.machine, dies=devices))
        result = graph_costs(modified, protocol_only(graphs(modified)), isolated=False)
        times.append(result["duration_ms"])
        services.append(result["execution"]["serviced"])
    assert times[0] > times[1] > times[2]
    assert services[0] == pytest.approx(services[1])
    assert services[1] == pytest.approx(services[2])


def test_other_input_representation_is_not_silently_reused_as_mx():
    raw = case()
    incompatible = replace(raw, model=replace(raw.model,
        quant=replace(raw.model.quant, activation_scale_format="fp32", activation_scale_pack=1)))
    with pytest.raises(ModelingError, match="matching_e4m3_e8m0"):
        graphs(incompatible)
