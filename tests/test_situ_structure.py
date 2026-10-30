"""Kimi's independent activation parameters survive every FFN path."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.gated_activation import gated_activation
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure


ROOT = Path(__file__).resolve().parents[1]


def _case():
    return parse_case(json.loads((ROOT / "examples/cases/kimi-k3-a5.json").read_text()))


def test_kimi_config_changes_dense_shared_and_routed_work_to_situ():
    case = _case()
    ops = target_operators(case, phase="prefill", tokens=8)
    activations = [op for op in ops if op.family == "situ"]
    assert {op.recipe for op in activations} == {"situ", "grouped_situ"}
    assert any(".shared." in op.op_id for op in activations)
    assert any(op.recipe == "situ" and ".shared." not in op.op_id for op in activations)
    assert not any(op.family == "swiglu" for op in ops)
    for op in activations:
        meta = dict(op.metadata)
        assert (meta["activation_beta"], meta["activation_linear_beta"]) == (4, 25)
        assert meta["activation_compute_dtype"] == "fp32"
        elements = op.mte3_bytes / 2
        assert op.vector_ops == pytest.approx(12 * elements)
        assert op.mte2_bytes == pytest.approx(4 * elements)
    # The independent base case has BF16 activations. An observed FP8 row must
    # not silently modify its quantization recipe just to create a match.
    graph = export_structure(case, phase="prefill", tokens=8, scope="body")
    assert not any(group["role"] in {"situ_quant", "grouped_gate_up_situ_quant",
                                     "grouped_gate_up_activation_quant"}
                   for group in graph["subgraphs"])


def test_situ_quantization_boundary_requires_declared_mx_operands():
    case = _case()
    quant = replace(case.model.quant, weight_format="fp8_e4m3",
                    expert_weight_format="fp8_e4m3", activation_format="fp8_e4m3",
                    weight_block=(32, 32),
                    activation_group_size=32, activation_scale_format="fp8_e8m0",
                    activation_scale_pack=2, exclusions=())
    selected = replace(case, model=replace(case.model, quant=quant))
    graph = export_structure(selected, phase="prefill", tokens=8, scope="body")
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    groups = [group for group in graph["subgraphs"] if group["role"] == "situ_quant"]
    assert groups
    for group in groups:
        members = [nodes[node_id]["operator"] for node_id in group["node_ids"]]
        assert {op["family"] for op in members} == {"situ", "quantize"}
        assert group["contract"][:2] == ["situ_mxquant_active", "SituMxQuant"]
        assert all(op["family"] != "gemm" for op in members)
    routed = [group for group in graph["subgraphs"]
              if group["role"] == "grouped_gate_up_situ_quant"]
    assert routed and all(group["contract"] is None for group in routed)
    assert all(group["coverage"]["kind"] == "native_source_domain_mismatch" for group in routed)
    assert not any(group["role"] == "grouped_gate_up_activation_quant"
                   for group in graph["subgraphs"])


def test_situ_parameter_expression_adds_clipping_work_without_extra_gm_tensors():
    plan = _case().model.layer_plans[0]
    common = dict(op_id="activation", deps=("up",), rows=3.5, width=128,
                  dtype="bf16", stage=0, requests=(), phase="prefill", recipe="swiglu")
    situ = gated_activation(plan=plan, **common)
    unclipped = gated_activation(plan=replace(plan, activation_situ_linear_beta=None), **common)
    silu = gated_activation(plan=replace(plan, ffn_activation="silu"), **common)
    assert situ.vector_ops - unclipped.vector_ops == 3 * 3.5 * 128
    assert silu.vector_ops == 4 * 3.5 * 128
    assert situ.mte2_bytes == unclipped.mte2_bytes == silu.mte2_bytes == 4 * 3.5 * 128
    assert situ.mte3_bytes == unclipped.mte3_bytes == silu.mte3_bytes == 2 * 3.5 * 128
