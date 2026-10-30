"""Native fusion candidates retain real semantic work and ownership."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.structure_groups import declare_group
from mindie_sim.structure import export_op_structure, export_structure
from test_prediction_system import case as small_case


ROOT = Path(__file__).resolve().parents[1]


def _case(name):
    return parse_case(json.loads((ROOT / "examples/cases" / (name + ".json")).read_text()))


def _report(case, *, phase="prefill", tokens=4096, context=0):
    return export_structure(case, phase=phase, tokens=tokens, context=context, scope="body")


def _observed(filename, profiler_api):
    import mindie_profiler
    from mindie_profiler.tensor_contracts import observed_contract

    path = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts" / filename
    if not path.is_file():
        pytest.skip(f"retained fusion fixture unavailable: {path}")
    return [(row, json.loads(json.dumps(observed_contract(row)))) for row in kernel_rows(path)]


@pytest.mark.parametrize("name,filename,roles", [
    ("v4f-a3-w8a8-tp8", "a3_v4f.csv",
     {"quant_lightning_indexer", "hc_pre", "hc_post", "grouped_gate_up_activation_quant"}),
    ("glm53-a3-w4a8", "a3_glm53_prefill.csv", {"grouped_gate_up_activation_quant"}),
])
def test_real_fusion_descriptors_match_independent_case_parameters(name, filename, roles, profiler_api):
    report = _report(_case(name))
    actual = [contract for _, contract in _observed(filename, profiler_api) if contract]
    nodes = {node["node_id"]: node for node in report["nodes"]}
    for role in roles:
        groups = [group for group in report["subgraphs"] if group["role"] == role]
        assert groups and all(group["contract"] in actual for group in groups)
        for group in groups:
            assert set(group["node_ids"]) <= nodes.keys()
            assert set(group["external_inputs"]).isdisjoint(group["node_ids"])
            assert set(group["external_outputs"]) <= set(group["node_ids"])
            assert group["coverage"]["kind"] == "complete_subgraph"
            if role == "grouped_gate_up_activation_quant":
                members = [nodes[node_id] for node_id in group["node_ids"]]
                assert {node["block"] for node in members} == {"ffn"}
                assert {node["family"] for node in members} == {
                    "grouped_gemm", "swiglu", "quantize"}
                assert len([node for node in members if node["family"] == "grouped_gemm"]) == 1
                assert {nodes[node_id]["family"] for node_id in group["external_outputs"]} == {"quantize"}


def test_compressor_precision_override_preserves_source_bytes_and_valid_row_work(profiler_api):
    baseline = _case("v4f-a3-w8a8-tp8")
    # This is a caller declaration of the source-supported FP32 norm branch.
    # No descriptor is used to populate the model or derive row geometry.
    selected = replace(baseline, model=replace(
        baseline.model, layer_plans=tuple(
            replace(plan, compressor_norm_dtype="fp32") for plan in baseline.model.layer_plans)))
    before, after = _report(baseline), _report(selected)
    observed = [contract for row, contract in _observed("a3_v4f.csv", profiler_api)
                if row.kind == "Compressor"]
    old_groups = [g for g in before["subgraphs"] if g["role"] == "compressor"]
    groups = [g for g in after["subgraphs"] if g["role"] == "compressor"]
    assert observed and all(contract for contract in observed)
    assert all(g["contract"] not in observed for g in old_groups)
    assert all(g["contract"] in observed for g in groups)
    nodes = {node["node_id"]: node for node in after["nodes"]}
    assert {g["contract"][-1][0][-1] for g in groups} == {128, 512}
    for group in groups:
        members = [nodes[key] for key in group["node_ids"]]
        ratio = group["contract"][3][2][0][0]
        width = group["contract"][-1][0][0]
        norm = next(node for node in members if node["family"] == "rmsnorm")
        rotary = next(node for node in members if node["family"] == "rotary")
        assert norm["operator"]["local_shape"] == [4096 // ratio, width]
        assert rotary["deps"] == [norm["node_id"]]
        assert group["external_outputs"] == [rotary["node_id"]]
        assert not any(node["family"] == "cache_store" for node in members)
    plan = next(plan for plan in selected.model.layer_plans if plan.owns_long_kv and plan.owns_index_k)
    old = {term.name: term for term in layer_weight_terms(baseline.model.layer_plans[plan.index], baseline.model)}
    new = {term.name: term for term in layer_weight_terms(plan, selected.model)}
    for name in ("compressor_norm", "indexer_k_norm"):
        assert term_bytes(new[name], selected.model.quant, selected.runtime.parallel) == (
            2 * term_bytes(old[name], baseline.model.quant, baseline.runtime.parallel))
        assert term_bytes(new[name], selected.model.quant, selected.runtime.parallel, source=True) == (
            term_bytes(old[name], baseline.model.quant, baseline.runtime.parallel, source=True))
    assert new["indexer_wk"].rows == plan.compressor_width * plan.index_dim
    assert new["indexer_gate"].rows == plan.compressor_width * plan.index_dim


def test_indexer_layernorm_and_quantized_external_producers_keep_distinct_semantics():
    glm = _case("glm53-a3-w4a8")
    ops = target_operators(glm, phase="prefill", tokens=32)
    normalizations = [op for op in ops if op.recipe == "index_norm"]
    assert normalizations
    for op in normalizations:
        width = dict(op.metadata)["reduction_width"]
        assert op.family == "layernorm" and op.vector_ops == 8 * 32 * width
        assert dict(op.metadata)["norm_has_bias"]
        assert op.dtype == "fp32"
        assert op.mte2_bytes == (32 + 2) * width * 4
        by_id = {value.op_id: value for value in ops}
        input_cast = by_id[op.deps[0]]
        assert input_cast.family == "cast" and input_cast.dtype == "fp32"
        assert input_cast.mte2_bytes == 32 * width * 2
        assert input_cast.mte3_bytes == 32 * width * 4
        output_cast = next(value for value in ops
                           if value.recipe == "index_norm_output_cast" and op.op_id in value.deps)
        assert output_cast.dtype == "bf16"
        assert output_cast.mte2_bytes == 32 * width * 4
        assert output_cast.mte3_bytes == 32 * width * 2
    v4 = _report(_case("v4f-a3-w8a8-tp8"), tokens=32)
    nodes = {node["node_id"]: node for node in v4["nodes"]}
    group = next(g for g in v4["subgraphs"] if g["role"] == "quant_lightning_indexer")
    query = next(nodes[key] for key in group["external_inputs"]
                 if nodes[key]["family"] == "quantize")
    assert query["operator"]["metadata"]["output_format"] == "int8"
    assert query["operator"]["metadata"]["scale_format"] == "fp16"
    assert query["node_id"] not in group["node_ids"]
    assert {nodes[key]["family"] for key in group["node_ids"]} == {"indexer", "topk"}


def test_hyper_pre_and_attention_residual_include_source_normalization_work():
    report = _report(_case("v4f-a3-w8a8-tp8"), tokens=8)
    nodes = {node["node_id"]: node for node in report["nodes"]}
    group = next(g for g in report["subgraphs"] if g["role"] == "hc_pre")
    recipes = {nodes[key]["recipe"]: nodes[key] for key in group["node_ids"]}
    inverse, affine = recipes["hc_inverse_rms"], recipes["hc_normalize_scale_bias_gate"]
    assert inverse["node_id"] in affine["deps"]
    assert affine["node_id"] in recipes["hc_sinkhorn"]["deps"]
    assert recipes["hc_sinkhorn"]["node_id"] in recipes["hc_pre_mix"]["deps"]
    report = _report(_case("kimi-k3-a5"), tokens=8)
    nodes = {node["node_id"]: node for node in report["nodes"]}
    group = next(g for g in report["subgraphs"] if g["role"] == "attention_residual")
    members = [nodes[key] for key in group["node_ids"]]
    softmax = next(node for node in members if node["family"] == "softmax")
    mixture = next(node for node in members if node["recipe"] == "attention_residual_mix")
    assert softmax["node_id"] in mixture["deps"]
    assert mixture["operator"]["dtype"] == softmax["operator"]["dtype"] == "fp32"
    output = nodes[group["external_outputs"][0]]
    assert output["recipe"] == "attention_residual_output_cast"
    assert output["operator"]["dtype"] == "bf16"
    assert group["contract"] is None


def test_kda_stage_launches_never_receive_complete_chunk_contract_and_decode_source_is_explicit(profiler_api):
    case = _case("kimi-k3-a5")
    kernels = {**case.runtime.kernels, "lowerings": {
        **case.runtime.kernels.get("lowerings", {}), "kda": "vllm_ascend_kda_in_kernel_gates"}}
    selected = replace(case, runtime=replace(case.runtime, kernels=kernels))
    prefill = _report(selected, tokens=64)
    groups = [g for g in prefill["subgraphs"] if g["role"] == "kda_chunk"]
    assert groups and all(g["contract"] is None for g in groups)
    assert all(g["coverage"]["kind"] == "multi_launch_stage_unknown" for g in groups)
    baseline = _report(case, phase="decode", tokens=1, context=32768)
    assert all(g["contract"] is None for g in baseline["subgraphs"] if g["role"] == "kda_recurrent")
    decode = _report(selected, phase="decode", tokens=1, context=32768)
    observed = [contract for row, contract in _observed("a5_kimi_k3.csv", profiler_api)
                if row.kind == "RecurrentKda"]
    groups = [g for g in decode["subgraphs"] if g["role"] == "kda_recurrent"]
    assert observed and groups and all(g["contract"] in observed for g in groups)
    nodes = {node["node_id"]: node for node in decode["nodes"]}
    for group in groups:
        members = [nodes[key] for key in group["node_ids"]]
        assert any(node["recipe"] == "qk_l2norm" for node in members)
        assert any(node["recipe"] == "delta_gates" for node in members)
        assert all(node["recipe"] != "linear_beta_sigmoid" for node in members)
        assert all(node["recipe"] != "linear_output_norm_gate" for node in members)


def test_gdn_algorithm_correspondence_excludes_external_norm_and_unverified_native_launch():
    report = _report(_case("qwen35-a3-tp4"), tokens=64)
    nodes = {node["node_id"]: node for node in report["nodes"]}
    groups = [group for group in report["subgraphs"] if group["role"] == "gdn_chunk"]
    assert groups
    for group in groups:
        members = [nodes[key] for key in group["node_ids"]]
        assert group["contract"] is None
        assert group["coverage"]["kind"] == "native_boundary_unverified"
        assert {node["family"] for node in members} == {
            "recurrent_local", "triangular_solve", "recurrent_scan", "recurrent_output"}
        external = {nodes[key]["recipe"] for key in group["external_inputs"]}
        assert {"qk_l2norm", "delta_gates"} <= external
        assert group["physical_work"]["launches"] > 1
    raw = _case("qwen35-a3-tp4")
    kernels = {**raw.runtime.kernels, "lowerings": {
        **raw.runtime.kernels.get("lowerings", {}),
        "native_fusion": {"implementation": "declared_native_fusion",
                          "options": {"roles": ["gdn_chunk"]}},
    }}
    with pytest.raises(ValueError, match="requires_complete_single_kernel"):
        _report(replace(raw, runtime=replace(raw.runtime, kernels=kernels)), tokens=64)


def test_cross_layer_fusion_preserves_actual_origin_scopes_without_relabeling():
    raw = small_case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant", "residual_norm"]
    case = parse_case(raw)
    ops = list(target_operators(case, phase="prefill", tokens=8))
    fused = next(op for op in ops if op.op_id == "s0.l1.norm1")
    origins = dict(fused.metadata)["fused_origin_scopes"]
    assert {(origin["layer"], origin["block"]) for origin in origins} == {
        (0, "ffn"), (1, "attention")}
    declare_group(
        ops, key="residual_and_next_norm", role="residual_norm", family="normalization",
        members=(fused.op_id,), entries=(fused.op_id,), anchor=fused.op_id,
        view={}, implementation={"native_kind": "AddRmsNorm"}, block=None)
    report = export_op_structure(
        ops, model=case.model.to_dict(), execution={"phase": "prefill"})
    group = report["subgraphs"][0]
    assert group["layer"] is None and group["block"] is None
    assert group["layer_ids"] == [0, 1]
    assert group["owner_scopes"] == [
        {"replica": 0, "stage": 0, "layer": 0, "block": "ffn"},
        {"replica": 0, "stage": 0, "layer": 1, "block": "attention"}]
    node = next(node for node in report["nodes"] if node["node_id"] == group["node_ids"][0])
    assert (node["layer"], node["block"]) == (1, "attention")


def test_mc2_chunks_keep_fused_ffn_groups_separate_and_share_no_members():
    case = _case("v4f-a3-w8a8-tp8")
    case = replace(case, runtime=replace(case.runtime, moe_recipe="mc2", mc2_chunks=3))
    report = _report(case, tokens=16)
    groups = [group for group in report["subgraphs"]
              if group["role"] == "grouped_gate_up_activation_quant" and group["layer"] == 0]
    assert len(groups) == 3
    seen = set()
    for group in groups:
        assert not seen.intersection(group["node_ids"])
        seen.update(group["node_ids"])
        assert group["external_outputs"] == [
            key for key in group["node_ids"] if key.endswith(".expert_down.quantize")]
        assert all(".expert_down" not in key or key.endswith(".quantize")
                   for key in group["node_ids"])
