"""Block scopes come from composing builders and survive ordinary lowering."""
import json
from pathlib import Path

import pytest

from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.structure import export_structure
from test_mla_family_selection import mla_case
from test_prediction_system import case


ROOT = Path(__file__).resolve().parents[1]
SCOPE_KEYS = {"structure_block", "structure_role"}


def _case(name):
    return (case() if name == "dense_gqa" else
            json.loads((ROOT / "examples/cases" / f"{name}.json").read_text()))


def _without_scope(value):
    if isinstance(value, dict):
        return {key: _without_scope(item) for key, item in value.items() if key not in SCOPE_KEYS}
    if isinstance(value, (list, tuple)):
        return [_without_scope(item) for item in value]
    return value


@pytest.mark.parametrize("name", [
    "dense_gqa", "qwen35-a3-tp4", "kimi-k3-a5", "v4f-a3-w8a8-tp8",
])
def test_gqa_gdn_kda_dense_moe_and_residual_scopes_leave_work_and_dependencies_unchanged(
    name, monkeypatch,
):
    import mindie_sim.prediction.frontend as frontend

    raw = _case(name)
    tagged = target_operators(raw, phase="prefill", tokens=8)
    report = export_structure(raw, phase="prefill", tokens=8, scope="body")
    # Disable only the metadata annotator. This compares all actual work,
    # tensor fields, placement and dependency edges of the same builders.
    monkeypatch.setattr(frontend, "_scope", lambda ops, block, role="body": ops)
    original = target_operators(raw, phase="prefill", tokens=8)
    assert [_without_scope(op.to_dict()) for op in tagged] == [
        _without_scope(op.to_dict()) for op in original]
    assert all(node["block"] in {"attention", "ffn"} for node in report["nodes"])
    assert all(node["role"] in {"body", "residual"} for node in report["nodes"])
    assert {node["block"] for node in report["nodes"] if node["role"] == "residual"} == {
        "attention", "ffn"}
    for node in report["nodes"]:
        assert node["block"] == node["operator"]["metadata"]["structure_block"]
        assert node["role"] == node["operator"]["metadata"]["structure_role"]
    # A layer's architecture is context shared by attention and FFN nodes;
    # the explicit block, rather than that context, identifies the operation.
    for layer in {node["layer"] for node in report["nodes"]}:
        rows = [node for node in report["nodes"] if node["layer"] == layer]
        assert {node["block"] for node in rows} == {"attention", "ffn"}
        assert all(node["structure"] == rows[0]["structure"] for node in rows)


def test_norm_and_inserted_quantizer_keep_owning_builder_block_and_role():
    raw = _case("qwen35-a3-tp4")
    report = export_structure(raw, phase="prefill", tokens=8, scope="body")
    nodes = {node["op_id"]: node for node in report["nodes"]}
    for layer in range(report["model"]["layers"]):
        for suffix, block in ((".norm1", "attention"), (".norm2", "ffn")):
            node = nodes[f"s0.l{layer}" + suffix]
            assert (node["block"], node["role"]) == (block, "body")
    quantized = [node for node in report["nodes"] if node["family"] == "quantize"]
    assert quantized
    for node in quantized:
        consumers = [other for other in report["nodes"] if node["node_id"] in other["deps"]]
        assert consumers
        assert any((other["block"], other["role"]) == (node["block"], node["role"])
                   for other in consumers)


def test_source_call_expansion_inherits_frontend_scope_for_kernels_and_join():
    raw = mla_case()
    report = export_structure(raw, phase="decode", tokens=1, context=16, scope="body")
    nodes = {node["node_id"]: node for node in report["nodes"]}
    assert report["source_calls"]
    for call in report["source_calls"]:
        assert call["operator"]["metadata"]["structure_block"] == "attention"
        assert call["operator"]["metadata"]["structure_role"] == "body"
        for node_id in (call["node_id"], *call["children"]):
            assert (nodes[node_id]["block"], nodes[node_id]["role"]) == ("attention", "body")
            meta = nodes[node_id]["operator"]["metadata"]
            assert (meta["structure_block"], meta["structure_role"]) == ("attention", "body")
            assert nodes[node_id]["lineage"]["source_call_id"] == call["node_id"]


def test_residual_norm_fusion_keeps_target_norm_scope_and_fused_lineage():
    raw = case()
    raw["runtime"]["lowering_passes"] = [
        "activation_quantization", "norm_quant", "residual_norm"]
    report = export_structure(raw, phase="prefill", tokens=8, scope="body")
    fused = [node for node in report["nodes"] if node["family"] == "add_rmsnorm"]
    assert fused
    assert {node["block"] for node in fused} == {"attention", "ffn"}
    assert all(node["role"] == "body" for node in fused)
    assert all(node["lineage"]["fused_from"] == ["residual_add", "rmsnorm"] for node in fused)
