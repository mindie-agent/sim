"""Ordinary activation producer selection and consumers' scale contracts."""
from dataclasses import asdict

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.operators import cuda_quant
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


def quant_case(*, source=True):
    raw = case(prompt=3, output=2)
    raw["model"].update(layers=1, quant=asdict(quant_preset("w8a8_int8")))
    device = raw["machine"]["device"]
    device["compute_peaks"]["int8"] = 1e9
    device["rates"]["shared_memory"] = 1e9
    device["service_modes"] = {
        role: {"resource": resource, "rate": 1e9}
        for role, resource in cuda_quant.SERVICE_ROLES.items()}
    if source:
        raw["runtime"]["kernels"] = {
            "lowerings": {"quantize": cuda_quant.IMPLEMENTATION}}
    return raw


def test_real_quant_producers_keep_payload_scales_and_double_source_input_passes():
    analytic, source = quant_case(source=False), quant_case()
    a_ops = {op.op_id: op for op in target_operators(analytic, phase="prefill", tokens=3)}
    s_ops = {op.op_id: op for op in target_operators(source, phase="prefill", tokens=3)}
    quantized = [op for op in s_ops.values() if op.family == "quantize"]
    assert quantized
    for op in quantized:
        original = a_ops[op.op_id]
        assert op.plan_required and op.returns == ("activation_scales",)
        assert op.deps == original.deps
        assert op.mte2_bytes == 2 * original.mte2_bytes
        assert op.mte3_bytes == original.mte3_bytes
        rows, width = op.local_shape
        assert op.mte3_bytes == rows * width + rows * 4
        consumers = [consumer for consumer in s_ops.values() if op.op_id in consumer.deps]
        assert consumers
        assert all(dict(consumer.metadata)["activation_scale_bytes"] == rows * 4
                   for consumer in consumers)
    report = operator_costs(source, quantized)
    assert all(row["program"] == "OperatorPlan" for row in report["operators"])
    result = predict(source)
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["parameters"]["calibration"] == "none"


def test_source_quant_boundary_binds_before_generic_norm_quant_fusion():
    raw = quant_case()
    raw["runtime"]["lowering_passes"] = [
        "activation_quantization", "norm_quant", "residual_norm"]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    quantized = [op for op in graphs[0] if op.family == "quantize"]
    assert quantized and all(op.plan_required for op in quantized)
    assert not any(op.family == "rmsnorm_quant" for op in graphs[0])
    report = graph_costs(raw, graphs, input_ready_ms=5)
    by_id = {row["op_id"]: row for row in report["timeline"]}
    for op in quantized:
        for dep in op.deps:
            assert by_id[op.op_id]["ready_ms"] >= by_id[dep]["complete_ms"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())


@pytest.mark.parametrize("dtype", ["fp16", "fp32"])
def test_model_binding_cannot_invent_casts_from_bf16_matrix_outputs(dtype):
    raw = quant_case(source=False)
    raw["model"]["activation_dtype"] = dtype
    ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=3)}
    assert dict(ops["s0.l0.gate_up"].metadata)["output_format"] == "bf16"
    assert ops["s0.l0.swiglu"].dtype == dtype
    raw["runtime"]["kernels"] = {"lowerings": {"quantize": cuda_quant.IMPLEMENTATION}}
    with pytest.raises(ModelingError, match="requires_bf16_producer_chain"):
        target_operators(raw, phase="prefill", tokens=3)


def test_block_scales_are_not_reinterpreted_as_per_token_cuda_quant():
    raw = quant_case()
    raw["model"]["quant"]["activation_group_size"] = 16
    with pytest.raises(ModelingError, match="integer_per_row_int8_fp32_scale"):
        target_operators(raw, phase="prefill", tokens=3)
    del raw["runtime"]["kernels"]
    ops = target_operators(raw, phase="prefill", tokens=3)
    assert any(len(dict(op.metadata)["activation_scale_shape"]) == 2
               for op in ops if op.family == "quantize")


def test_statistical_moe_rows_cannot_be_promoted_to_integer_source_ctas():
    raw = quant_case()
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32,
                        expert_routing_histogram=[.8, .1, .05, .05])
    raw["runtime"].update(tp=2, ep=2)
    raw["machine"]["num_devices"] = 2
    raw["machine"]["links"][0]["members"] = [0, 1]
    with pytest.raises(ModelingError, match="integer_per_row_int8_fp32_scale"):
        target_operators(raw, phase="prefill", tokens=3)
