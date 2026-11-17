"""A5 structure entry: typed values, nullable semantic refs, native candidates.

The A5 fixture is a real declared case from ``examples/cases``.  It is read as
JSON only: no weights, device context, profiler capture or target duration is
consulted.  Counterexamples distinguish source correspondence from a selected
software plan, preserve existing typed records/views/state facts, and keep a
lone native name unselected when ``selection`` is null.
"""
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.engine import Engine
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.structure_groups import declare_group
from mindie_sim.structure import export_op_structure, export_structure
from mindie_sim.structure_implementations import operator_structure

ROOT = Path(__file__).resolve().parents[1]
A5_CASE = ROOT / "examples/cases/qwen35-a5.json"


@pytest.fixture(scope="module")
def a5_report():
    return export_structure(json.loads(A5_CASE.read_text()), phase="decode",
                            tokens=1, context=17, scope="body")


def _nodes(report):
    return {node["node_id"]: node for node in report["nodes"]}


def _tiny_case():
    return {"machine": {"name": "declared", "architecture": "declared", "num_devices": 1,
                        "device": {"memory_bytes": 1024 ** 3,
                                   "compute_peaks": {"bf16": 1e9, "fp32": 1e9},
                                   "rates": {"memory": 1e9, "vector": 1e9},
                                   "compute_units": 4}},
            "model": {"name": "geometry", "layers": 2, "hidden_size": 32,
                      "num_heads": 4, "num_kv_heads": 2, "head_dim": 8,
                      "intermediate_size": 64, "vocab_size": 128},
            "workload": {"requests": [{"id": "r", "prompt_tokens": 4,
                                       "output_tokens": 3}]},
            "runtime": {"lowering_passes": ["activation_quantization", "norm_quant",
                                           "residual_norm"]}}


def test_a5_matrix_entry_keeps_values_and_two_unselected_native_choices(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["family"] == "gemm"
                and node["values"]["contract_kind"] == "matrix"
                and node["values"]["quantization"]["scheme"] == "w8a8")
    values = node["values"]
    assert values["basis"] == "contract"
    assert values["quantization"]["activation_format"] == "int8"
    assert values["quantization"]["weight_format"] == "int8"
    for value in (*values["inputs"], *values["outputs"]):
        assert {"dtype", "layout", "quant", "attrs", "shape",
                "logical_shape", "storage_shape", "strides", "offset"} <= value.keys()
    activation = next(value for value in values["inputs"] if value["name"] == "activation")
    weight = next(value for value in values["inputs"] if value["name"] == "weight")
    output = values["outputs"][0]
    # A compact (M,N,K) tuple does not prove ND; the explicit input port does
    # supply the actual producer representation for activation and scale.
    assert activation["layout"] == "strided"
    assert activation["producer"] == "s0.l0.linear_qkvz.quantize"
    assert activation["output"] == "payload"
    assert activation["bytes"] == 2048.0
    assert weight["layout"] is None
    assert output["layout"] is None
    assert (activation["dtype"], activation["quant"]) == ("int8", "int8")
    assert (weight["dtype"], weight["quant"]) == ("int8", "int8")
    assert output["dtype"] == "bf16" and output["quant"] == ""
    m, n, k = node["contract"][1]
    assert activation["shape"] == [m, k]
    assert weight["shape"] == [n, k]
    assert output["shape"] == [m, n]
    assert {"value_representation_partial"} <= {
        gap["code"] for gap in values["gaps"]}

    candidates = node["native_candidates"]
    assert [candidate["native"] for candidate in candidates] == [
        "QuantBatchMatmulV3", "QuantBatchMatmulV4"]
    assert all(candidate["selected"] is False for candidate in candidates)
    assert all(candidate["observed"] is False for candidate in candidates)
    execution = node["native_execution"]
    assert execution["kind"] == "candidate_set_only"
    assert execution["selection_state"] == "multiple_compatible_unselected"
    assert execution["selected"] is None
    assert node["semantic_refs"] == [{"kind": "layer", "name": "0",
                                      "role": "attention"}]


def test_a5_scale_roles_keep_runtime_and_storage_formats(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["node_id"].endswith("linear_qkvz"))
    scales = node["values"]["quantization"]["scales"]
    assert scales["activation"]["format"] == "fp32"
    assert scales["weight"]["format"] == "bf16"
    assert scales["weight"]["storage_format"] == "fp32"
    activation_scale = next(value for value in node["values"]["inputs"]
                            if value["name"] == "activation_scale")
    assert activation_scale["dtype"] == "fp32"
    weight_scale = next(value for value in node["values"]["declared"]
                        if value["name"] == "weight_scale")
    assert weight_scale["role"] == "scale"
    assert weight_scale["dtype"] == "bf16"
    assert weight_scale["attrs"]["storage_format"] == "fp32"
    assert weight_scale["attrs"]["association"] == "weight"
    assert {port["output"] for port in node["values"]["input_ports"]} == {
        "payload", "activation_scales"}
    activation_scale = next(value for value in node["values"]["inputs"]
                            if value["name"] == "activation_scale")
    assert activation_scale["producer"] == "s0.l0.linear_qkvz.quantize"
    assert activation_scale["output"] == "activation_scales"
    assert activation_scale["bytes"] == 4.0
    assert activation_scale["attrs"]["payload_producer"] == (
        "s0.l0.linear_qkvz.quantize")
    assert activation_scale["attrs"]["payload_output"] == "payload"
    assert all(port["source_op"] == "s0.l0.norm1"
               for port in node["values"]["input_ports"])


def test_a5_norm_semantic_attrs_keep_declared_epsilon(a5_report):
    norm = next(node for node in a5_report["nodes"]
                if node["family"] == "rmsnorm")
    attrs = {item["name"]: item["value"]
             for item in norm["values"]["semantic_attrs"]}
    assert attrs["norm_eps"] == pytest.approx(1e-6)
    assert attrs["reduction_width"] == 2048


def test_a5_swiglu_input_keeps_producer_shape_not_output_duplicate(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["family"] == "swiglu")
    values = node["values"]
    assert values["inputs"][0]["shape"] == [8, 256]
    assert values["inputs"][0]["producer"] == "s0.l0.expert_up"
    assert values["outputs"][0]["shape"] == [8, 128]
    assert values["inputs"][0]["shape"] != values["outputs"][0]["shape"]
    assert "swiglu_input_unresolved" not in {
        gap["code"] for gap in values["gaps"]}
    # Existing work declarations stay untouched.
    assert node["physical_work"]["mte2_bytes"] == 4096.0
    assert node["physical_work"]["mte3_bytes"] == 2048.0


def test_a5_qkv_view_preserves_source_target_layout_and_producer(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["node_id"] == "rank:0/s0.l11.qkv.value")
    values = node["values"]
    assert values["basis"] == "declared_view"
    source, target = values["inputs"][0], values["outputs"][0]
    assert source["shape"] == [1, 1, 256]
    assert source["storage_shape"] == [1, 2560]
    assert source["strides"] == [2560, 256, 1]
    assert source["offset"] == 2304
    assert (source["dtype"], source["layout"]) == ("bf16", "strided")
    assert source["producer"] == "s0.l11.qkv"
    assert source["bytes"] == 512.0
    assert source["storage_bytes"] == 5120.0
    assert target["shape"] == [1, 1, 256]
    assert target["strides"] == [256, 256, 1]
    assert target["offset"] == 0
    assert target["producer"] == "s0.l11.qkv.value"
    assert target["bytes"] == target["storage_bytes"] == 512.0
    codes = {gap["code"] for gap in values["gaps"]}
    assert "native_tensor_abi_unavailable" in codes
    assert "view_source_layout_missing" not in codes


def test_a5_fused_region_lone_native_name_is_not_a_selection(a5_report):
    group = next(group for group in a5_report["subgraphs"]
                 if group["role"] == "qkv_norm_rope")
    assert group["implementation"]["selection"] is None
    assert len(group["native_candidates"]) == 1
    assert group["native_candidates"][0]["selected"] is False
    assert group["native_execution"]["kind"] == "candidate_set_only"
    assert group["native_execution"]["selected"] is None
    # The semantic boundary still has four launches and no replacement region.
    assert group["physical_work"]["launches"] == 4
    assert group.get("resource_boundary") is None


def test_multi_candidate_region_remains_unselected(a5_report):
    group = next(group for group in a5_report["subgraphs"]
                 if len(group["native_candidates"]) > 1)
    assert len({candidate["native"] for candidate in group["native_candidates"]}) > 1
    assert all(candidate["selected"] is False for candidate in group["native_candidates"])
    assert group["native_execution"]["selected"] is None
    assert group["semantic_refs"]
    assert {ref["name"] for ref in group["semantic_refs"]} == set(group["node_ids"])


def test_explicit_region_selection_marks_one_candidate_and_members_stay_unselected():
    case = parse_case(_tiny_case())
    ops = list(target_operators(case, phase="prefill", tokens=8))
    fused = next(op for op in ops if op.op_id == "s0.l1.norm1")
    declare_group(
        ops, key="residual_and_next_norm", role="residual_norm",
        family="normalization", members=(fused.op_id,), entries=(fused.op_id,),
        anchor=fused.op_id, view={},
        implementation={"native_kind": "AddRmsNorm",
                        "selection": "explicit_caller_implementation"},
        block=None)
    report = export_op_structure(ops, model=case.model.to_dict(),
                                 execution={"phase": "prefill"})
    group = report["subgraphs"][0]
    assert group["native_candidates"][0]["selected"] is True
    assert group["native_candidates"][0]["observed"] is False
    assert group["native_execution"]["kind"] == "declared_native_region"
    assert group["native_execution"]["selected"]["native"] == "AddRmsNorm"
    member = next(node for node in report["nodes"]
                  if node["node_id"] == group["node_ids"][0])
    assert member["native_execution"]["selected"] is None
    assert all(candidate["selected"] is False
               for candidate in member["native_candidates"])


def test_region_without_selection_keeps_its_lone_candidate_unselected():
    case = parse_case(_tiny_case())
    ops = list(target_operators(case, phase="prefill", tokens=8))
    fused = next(op for op in ops if op.op_id == "s0.l1.norm1")
    declare_group(
        ops, key="unselected_region", role="residual_norm",
        family="normalization", members=(fused.op_id,), entries=(fused.op_id,),
        anchor=fused.op_id, view={}, implementation={"native_kind": "AddRmsNorm"},
        block=None)
    report = export_op_structure(ops, model=case.model.to_dict(),
                                 execution={"phase": "prefill"})
    group = report["subgraphs"][0]
    assert group["native_candidates"][0]["selected"] is False
    assert group["native_execution"]["kind"] == "candidate_set_only"
    assert group["native_execution"]["selection_state"] == "single_compatible_unselected"
    assert group["native_execution"]["selected"] is None


def test_add_rmsnorm_preserves_typed_inputs_residual_output_and_epsilon():
    case = parse_case(_tiny_case())
    report = export_structure(case, phase="prefill", tokens=8, scope="body")
    node = next(node for node in report["nodes"]
                if node["family"] == "add_rmsnorm")
    values = node["values"]
    assert values["basis"] == "typed_records"
    outputs = {value["name"]: value for value in values["outputs"]}
    assert set(outputs) == {"norm", "residual"}
    assert outputs["residual"]["role"] == "residual"
    assert outputs["residual"]["materialized"] is True
    assert outputs["residual"]["producer"] == "s0.l0.norm2"
    assert outputs["residual"]["bytes"] == 512.0
    assert outputs["residual"]["declared_bytes"] == 512.0
    assert outputs["norm"]["bytes"] == 512.0
    assert set(values["external_outputs"]) == {"residual", "norm"}
    attrs = {item["name"]: item["value"]
             for item in values["semantic_attrs"]}
    assert attrs["norm_eps"] == pytest.approx(1e-6)
    assert values["selection"] == {"basis": "selected_software_plan",
                                   "kind": "residual_norm"}
    # A selected software plan does not promote its lone native counterpart.
    assert node["native_execution"]["selected"] is None
    assert node["native_execution"]["plan_selection"]["kind"] == "residual_norm"
    assert all(candidate["selected"] is False
               for candidate in node["native_candidates"])


def test_real_lower_fusion_covers_multiple_members_with_one_boundary():
    from test_native_fusion_resources import HC_PROFILE, _case

    case = _case("v4f-a3-w8a8-tp8", ["hc_pre"], HC_PROFILE,
                 peaks={"hf32": 2.0e14})
    report = export_structure(case, phase="prefill", tokens=8, scope="body")
    group = next(group for group in report["subgraphs"]
                 if group.get("resource_boundary"))
    assert len(group["node_ids"]) > 1
    boundary = group["resource_boundary"]
    assert boundary["kind"] == "selected_native_fusion"
    assert len(boundary["operator_ids"]) == 1
    assert group["native_execution"]["selected"] is not None
    assert group["native_execution"]["observed"] is False
    resource = next(op for op in report["resource_operators"]
                    if op["node_id"] == boundary["operator_ids"][0])
    assert (group["native_execution"]["selected"]["native"]
            == resource["native_execution"]["selected"]["native"])
    members = [node for node in report["nodes"]
               if node["node_id"] in group["node_ids"]]
    assert members
    assert all(member["native_execution"]["kind"] == "covered_by_fused_region"
               for member in members)
    assert all(candidate["selected"] is False
               for member in members for candidate in member["native_candidates"])
    assert all(candidate["selected"] is False
               for candidate in group["native_candidates"])


def test_source_call_expansion_keeps_one_call_identity_across_kernels():
    from test_mla_family_selection import mla_case

    raw = mla_case()
    report = export_structure(raw, phase="decode", tokens=1, context=16,
                              sequences=2, native_evidence=True)
    call = next(row for row in report["source_calls"])
    assert call["operator"]["source_boundary"] == "call"
    assert call["children"]
    parent = next((node for node in report["nodes"]
                   if node["node_id"] == call["node_id"]), None)
    if parent is not None:
        assert parent["native_execution"]["selected"] is None
    children = [node for node in report["nodes"]
                if node["node_id"] in call["children"]]
    assert len(children) > 1
    assert all(child["boundary"] == "kernel" for child in children)
    selected = [child["native_execution"]["selected"] for child in children]
    assert all(item is not None for item in selected)
    assert {item["native"] for item in selected} == {"decode_attention_fwd_grouped"}
    assert {item["selected_by"] for item in selected} == {
        "operator.metadata.native_call"}
    # The expanded kernels retain one call identity for the declared call.
    assert all(child["lineage"]["source_call_id"] == call["node_id"]
               for child in children)
    coverage = report["native_call_coverage"]
    assert coverage["denominator"]["emitted_operators"] == len(report["nodes"])


def test_single_native_name_with_null_selection_is_not_selected():
    operator = {"op_id": "one", "family": "rmsnorm", "recipe": "rmsnorm",
                "local_shape": [1, 8], "dtype": "bf16",
                "source_boundary": "semantic",
                "metadata": {"native_implementation": {
                    "native_kind": "LoneNative", "selection": None}}}
    record = operator_structure(operator)
    assert [candidate["native"] for candidate in record["native_candidates"]] == [
        "LoneNative"]
    assert record["native_candidates"][0]["selected"] is False
    assert record["native_execution"]["selected"] is None
    assert record["native_execution"]["selection_state"] == (
        "single_compatible_unselected")


def test_novel_matrix_shape_is_recognized_without_a_measurement_gate(monkeypatch):
    quant = QuantSpec("int8", weight_format="int8", activation_format="int8",
                      kv_dtype="bf16")
    op = gemm_op(
        op_id="novel", deps=("input",), stage=0, layer=-1, requests=(),
        m=7, n=129, k=67, quant=quant, category="dense", phase="prefill",
        recipe="novel_linear", family="gemm")
    declaration = op.to_dict(include_work=False)

    def no_timing(*args, **kwargs):
        raise AssertionError("structure recognition consulted the timing engine")

    monkeypatch.setattr(Engine, "advance", no_timing)
    record = operator_structure(declaration)
    assert record["semantic_refs"] is None
    assert record["values"]["basis"] == "contract"
    assert record["values"]["contract_kind"] == "matrix"
    assert record["values"]["inputs"][0]["shape"] == [7, 67]
    assert record["values"]["inputs"][1]["shape"] == [129, 67]
    assert record["values"]["outputs"][0]["shape"] == [7, 129]
    assert record["values"]["quantization"]["scheme"] == "w8a8"
    assert len(record["native_candidates"]) == 2
    assert all(candidate["selected"] is False
               for candidate in record["native_candidates"])


def test_semantic_refs_are_nullable_when_no_model_relation_is_declared():
    quant = QuantSpec("fp32", weight_format="fp32", activation_format="fp32")
    op = gemm_op(
        op_id="standalone", deps=("input",), stage=0, layer=-1, requests=(),
        m=8, n=16, k=8, quant=quant, category="dense", phase="eager",
        recipe="eager_linear", family="gemm")
    report = export_op_structure(
        (op,), model={"kind": "standalone"},
        execution={"phase": "eager", "scope": "model"})
    node = report["nodes"][0]
    assert node["layer"] == -1
    assert node["semantic_refs"] is None
    assert node["values"]["outputs"][0]["shape"] == [8, 16]
    assert node["native_candidates"] == []


def test_unmodeled_matrix_mechanism_names_the_geometry_that_has_no_abi():
    quant = QuantSpec("int8", weight_format="int8", activation_format="int8",
                      kv_dtype="bf16")
    op = gemm_op(
        op_id="two_banks", deps=("input",), stage=0, layer=-1, requests=(),
        m=7, n=129, k=67, quant=quant, category="dense", phase="prefill",
        recipe="two_bank_projection", family="gemm", weight_count=2)
    record = operator_structure(op.to_dict(include_work=False))
    assert record["contract"] is None
    gap = next(gap for gap in record["values"]["gaps"]
               if gap["code"] == "matrix_tensor_abi_unavailable")
    assert gap["detail"]["matrix_groups"] == 2
    assert gap["detail"]["weight_format"] == "int8"
    assert record["native_candidates"]
    assert all(candidate["selected"] is False
               for candidate in record["native_candidates"])


def test_a5_attention_missing_abi_is_named_without_a_silent_fallback(a5_report):
    attention = [node for node in a5_report["nodes"]
                 if node["family"] == "attention" and node["contract"] is None]
    assert attention
    for node in attention:
        values = node["values"]
        assert values["basis"] == "semantic_geometry"
        assert values["inputs"] and values["outputs"]
        attrs = {item["name"]: item["value"] for item in values["semantic_attrs"]}
        assert attrs["query_tokens"] == 1
        assert attrs["context_tokens"] == 17
        assert (attrs["query_heads"], attrs["kv_heads"], attrs["head_dim"]) == (
            4, 1, 256)
        assert attrs["causal"] is True
        assert values["kv_cache_access"]["key_payload_row_bytes"] == 512
        query = next(value for value in values["inputs"] if value["name"] == "query")
        assert query["producer"] == ""
        assert query["attrs"]["producer_unresolved"] is True
        assert "operand_producer_unresolved" in {
            gap["code"] for gap in node["representation_gaps"]}
        codes = {gap["code"] for gap in node["representation_gaps"]}
        assert "attention_tensor_abi_unavailable" in codes
        assert "attention_native_slots_unresolved" in codes
        assert node["implementation"]["native_kinds"] == ["FusedInferAttentionScore"]
        assert node["native_execution"]["selected"] is None
        assert all(candidate["selected"] is False
                   for candidate in node["native_candidates"])
    assert "duration_ms" not in json.dumps(a5_report)


def test_index_and_target_projection_preserve_the_same_value_semantics():
    from mindie_sim.structure_index import export_structure_index
    from mindie_sim.structure_selection import select_structure_targets

    raw = json.loads(A5_CASE.read_text())
    args = dict(phase="decode", tokens=1, context=17, scope="body")
    full = export_structure(raw, **args)
    index = export_structure_index(raw, **args)
    node_ids = ("rank:0/s0.l11.qkv.value",
                "rank:0/s0.l0.linear_qkvz.quantize")
    for node_id in node_ids:
        full_node = next(node for node in full["nodes"]
                         if node["node_id"] == node_id)
        index_node = next(node for node in index["nodes"]
                          if node["node_id"] == node_id)
        assert index_node["values"] == full_node["values"]
        assert index_node["semantic_refs"] == full_node["semantic_refs"]
        assert index_node["native_candidates"] == full_node["native_candidates"]
        assert index_node["native_execution"] == full_node["native_execution"]
    assert index["graph_id"] == full["graph_id"]
    selected = select_structure_targets(raw, list(node_ids), **args)
    for node_id in node_ids:
        full_node = next(node for node in full["nodes"]
                         if node["node_id"] == node_id)
        assert selected["targets"][node_id]["values"] == full_node["values"]
    assert selected["graph_id"] == full["graph_id"]


def test_a5_recurrent_state_keeps_geometry_versions_and_effects(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["family"] == "recurrent_vector")
    values = node["values"]
    assert values["basis"] == "contract+state"
    inputs = {value["name"]: value for value in values["inputs"]}
    outputs = {value["name"]: value for value in values["outputs"]}
    assert inputs["query"]["shape"] == [1, 1, 4, 128]
    assert inputs["value"]["shape"] == [1, 1, 8, 128]
    assert inputs["value"]["storage_shape"] == [2048]
    assert inputs["value"]["strides"] == [1024, 1024, 128, 1]
    assert inputs["value"]["offset"] == 1024
    assert inputs["value"]["bytes"] == 2048.0
    assert inputs["value"]["storage_bytes"] == 4096.0
    assert inputs["query"]["storage_shape"] == [2048]
    assert inputs["key"]["storage_shape"] == [2048]
    assert inputs["query"]["storage_bytes"] == inputs["key"]["storage_bytes"] == 4096.0
    assert {inputs["query"]["offset"], inputs["key"]["offset"],
            inputs["value"]["offset"]} == {0, 512, 1024}
    assert inputs["state"]["dtype"] == "fp32"
    assert inputs["state"]["shape"] == [1, 8, 128, 128]
    assert inputs["state"]["layout"] == "V,K"
    assert outputs["output"]["shape"] == [1, 8, 128]
    assert outputs["state_out"]["shape"] == [1, 8, 128, 128]
    assert values["state_contract"]["read_versions"] == [17]
    assert [effect["action"] for effect in values["state_effects"]] == [
        "read", "write", "commit"]
    assert inputs["state"]["attrs"]["targets"] == [
        "s0.l0.recurrent_state[analysis0:0]"]
    assert inputs["state"]["attrs"]["slot_source"] == "caller_tensor"
    source_effects = node["operator"]["metadata"]["gdn_state_effects"]
    assert [effect["version"] for effect in values["state_effects"]] == [
        effect["version"] for effect in source_effects]
    assert values["state_effects"][0]["version"] == 17
    assert values["state_effects"][-1]["version"] == 18
    assert "state_shape_not_declared" not in {
        gap["code"] for gap in values["gaps"]}


def test_explicit_single_native_selection_leaves_the_other_candidate_unselected():
    operator = {"op_id": "two", "family": "rmsnorm", "recipe": "rmsnorm",
                "local_shape": [1, 8], "dtype": "bf16",
                "source_boundary": "semantic",
                "metadata": {"native_implementation": {
                    "native_kinds": ["NativeA", "NativeB"],
                    "selection": "NativeA"}}}
    record = operator_structure(operator)
    selected = {candidate["native"]: candidate["selected"]
                for candidate in record["native_candidates"]}
    assert selected == {"NativeA": True, "NativeB": False}
    assert record["native_execution"]["kind"] == "source_selected_native"
    assert record["native_execution"]["selected"]["native"] == "NativeA"
    assert record["native_execution"]["selected"]["candidate_id"].endswith(
        "/native:NativeA")


def test_pure_model_parse_and_program_do_not_fabricate_a_workload():
    from mindie_sim.prediction.operator_analysis import (
        model_graphs, model_program, target_graphs,
    )
    from mindie_sim.prediction.structure import parse_case, parse_model
    from test_prediction_system import case

    raw = case()
    raw.pop("workload", None)
    raw["hardware"] = raw.pop("machine")
    parsed = parse_model(raw)
    assert parsed.workload.requests == ()
    assert parsed.model.layers == 2
    full_without_workload = {"machine": raw["hardware"], "model": raw["model"],
                             "runtime": raw["runtime"]}
    with pytest.raises(Exception, match="case_missing_fields"):
        parse_case(full_without_workload)

    program = model_program(raw, phase="decode", tokens=1, context=8, sequences=1)
    assert program.assembly == "model"
    families = {op.family for op in program.work.ops}
    assert "logits" in families
    assert not ({"sampling", "acceptance", "token_commit"} & families)
    assert program.triggers == ()
    assert program.work.ops
    assert program.ports.tensor_in or program.ports.tensor_out

    model = model_graphs(raw, phase="decode", tokens=1, context=8, sequences=1,
                         scope="model")
    assert type(model).__name__ == "ModelGraphSet"
    assert model.programs.keys() == model.keys() == {0}
    program_ops = [op.op_id for op in model.program(0).work.ops]
    assert [op.op_id for op in model[0]] == program_ops
    assert model.program(0).assembly == "model"
    assert model.ports[0].tensor_out
    assert "logits" in {op.family for op in model[0]}
    body = model_graphs(raw, phase="decode", tokens=1, context=8, sequences=1,
                        scope="body")
    assert type(body).__name__ == "TargetGraphSet"
    assert not ({"logits", "embedding"} & {op.family for op in body[0]})
    explicit = target_graphs(raw, phase="decode", tokens=1, context=8,
                             sequences=1, scope="body", pure_model=True)
    assert type(body).__name__ == "TargetGraphSet"
    assert body.graph_id == explicit.graph_id if hasattr(body, "graph_id") else True
    assert sorted(body) == sorted(explicit)
    assert [op.op_id for op in body[0]] == [op.op_id for op in explicit[0]]


def test_a5_quantize_produces_payload_scale_and_consumer_binds_real_ports(a5_report):
    nodes = _nodes(a5_report)
    quant = nodes["rank:0/s0.l0.linear_qkvz.quantize"]
    matrix = nodes["rank:0/s0.l0.linear_qkvz"]
    qvalues = quant["values"]
    assert len(qvalues["inputs"]) == 1
    source = qvalues["inputs"][0]
    assert (source["shape"], source["dtype"]) == ([1, 2048], "bf16")
    assert (source["producer"], source["output"]) == ("s0.l0.norm1", "norm")
    assert source["bytes"] == 4096.0
    outputs = {value["output"]: value for value in qvalues["outputs"]}
    assert set(outputs) == {"payload", "activation_scales"}
    assert outputs["payload"]["dtype"] == "int8"
    assert outputs["payload"]["bytes"] == 2048.0
    assert outputs["activation_scales"]["dtype"] == "fp32"
    assert outputs["activation_scales"]["bytes"] == 4.0
    # The scale is this quantizer's output, never its own input.
    assert not any(value["name"] in {"activation_scale", "activation_scales"}
                   for value in qvalues["inputs"])
    norm = next(node for node in a5_report["nodes"]
                if node["node_id"] == "rank:0/s0.l0.norm1")
    assert (source["producer"], source["output"]) == (
        "s0.l0.norm1", "norm")
    assert any(value["output"] == "norm" for value in norm["values"]["outputs"])
    assert not any(value["output"].startswith("output0")
                   for value in norm["values"]["outputs"])
    assert quant["physical_work"]["mte2_bytes"] == 4096.0
    assert quant["physical_work"]["mte3_bytes"] == 2052.0
    conflict_source = json.loads(json.dumps(quant["operator"]))
    boundary = conflict_source["metadata"]["fused_origin_scopes"][0]["boundary"]
    boundary["typed_inputs"] = [*boundary.get("typed_inputs", ()), {
        "producer": "s0.l0.norm1", "output": "activation_scales",
        "role": "scale", "dtype": "fp32", "shape": [1],
        "source": "synthetic_conflict_probe"}]
    conflict_record = operator_structure(conflict_source, quant["contract"])
    conflicts = [value for value in conflict_record["values"]["declared"]
                 if value["attrs"].get("conflict_with_output")]
    assert conflicts and conflicts[0]["producer"] == "s0.l0.norm1"
    assert conflicts[0]["output"] == "activation_scales"

    inputs = {value["name"]: value for value in matrix["values"]["inputs"]}
    assert (inputs["activation"]["producer"], inputs["activation"]["output"]) == (
        "s0.l0.linear_qkvz.quantize", "payload")
    assert (inputs["activation_scale"]["producer"],
            inputs["activation_scale"]["output"]) == (
        "s0.l0.linear_qkvz.quantize", "activation_scales")
    assert inputs["activation"]["producer"] != matrix["op_id"]
    ports = {port["output"]: port for port in matrix["values"]["input_ports"]}
    assert set(ports) == {"payload", "activation_scales"}
    assert all(port["op"] == "s0.l0.linear_qkvz.quantize"
               for port in ports.values())


def test_operator_accounting_consumes_selected_transport_options():
    from mindie_sim.prediction.operator_analysis import operator_costs
    from test_a5_topology_communication_slice import case as topology_case, exchange

    raw = topology_case(tp=2, dp=1)
    raw["machine"]["links"][0].update(
        transaction_bytes=64, protocol_overhead_bytes=0,
        capacity_bytes_per_ms=1_000_000)
    raw["runtime"]["kernels"] = {"families": {"alltoall": {
        "kind": "direct_peer_collective",
        "max_inflight_per_rank": 2, "chunk_bytes": 64}}}
    op = exchange(128)
    report = operator_costs(raw, (op,))
    communication = report["operators"][0]["communication"]
    assert communication["max_inflight_per_rank"] == 2
    assert communication["chunk_bytes"] == 64
    assert communication["messages"] == 4
    assert communication["wire_bytes"] == 256.0
    assert report["operators"][0]["serviced"]["link:fabric"] == 256.0


def test_recurrent_state_capacity_is_unknown_until_declared(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["family"] == "recurrent_vector")
    values = node["values"]
    state = next(value for value in values["inputs"] if value["name"] == "state")
    assert state["shape"] == [1, 8, 128, 128]
    assert state["bytes"] == 524288.0
    assert state["storage_shape"] is None
    assert state["storage_bytes"] is None
    assert "state_capacity_not_declared" in {
        gap["code"] for gap in values["gaps"]}

    probe = json.loads(json.dumps(node["operator"]))
    contract = probe["metadata"]["gdn_state_contract"]
    contract["state_capacity"] = 9
    contract["slots"] = 9
    contract["active_slots"] = [0, 1]
    record = operator_structure(probe, node["contract"])
    expanded = next(value for value in record["values"]["inputs"]
                    if value["name"] == "state")
    assert expanded["shape"] == [2, 8, 128, 128]
    assert expanded["bytes"] == 1048576.0
    assert expanded["storage_shape"] == [9, 8, 128, 128]
    assert expanded["storage_bytes"] == 4718592.0
    assert expanded["attrs"]["state_capacity"] == 9
    state_out = next(value for value in record["values"]["outputs"]
                     if value["name"] == "state_out")
    assert state_out["storage_shape"] == [9, 8, 128, 128]
    assert [effect["action"] for effect in record["values"]["state_effects"]] == [
        "read", "write", "commit"]


def test_multi_output_norm_identity_survives_output_order_swap():
    raw = json.loads(A5_CASE.read_text())
    report = export_structure(raw, phase="decode", tokens=1, context=17, scope="body")
    node = next(node for node in report["nodes"]
                if node["node_id"] == "rank:0/s0.l0.norm1")
    values = node["values"]
    assert [(value["name"], value["dtype"], value["bytes"])
            for value in values["outputs"]] == [
        ("norm", "bf16", 4096.0), ("rstd", "fp32", 4.0)]
    produced = json.loads(json.dumps(values["outputs"]))
    swapped = list(node["contract"])
    swapped[2], swapped[3] = swapped[3], swapped[2]
    operator = json.loads(json.dumps(node["operator"]))
    first = operator_structure(operator, node["contract"],
                               context={"produced": produced})
    second = operator_structure(operator, tuple(swapped),
                                context={"produced": produced})
    first_names = {value["name"]: value["output"] for value in first["values"]["outputs"]}
    second_names = {value["name"]: value["output"] for value in second["values"]["outputs"]}
    assert first_names == {"norm": "norm", "rstd": "rstd"}
    assert second_names == first_names
    assert all(not name.startswith("output0") for name in first_names)


def test_body_target_uses_post_insertion_capture_partition_once(monkeypatch):
    from dataclasses import replace
    from test_typed_transforms_a5 import _parsed
    from mindie_sim.prediction.graph import GraphPolicy
    from mindie_sim.prediction.operator_analysis import target_operators

    case = _parsed("w8a8_int8")
    case = replace(case, runtime=replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("rmsnorm",),
        lowering_passes=("activation_quantization",)))
    calls = []
    original = GraphPolicy.apply

    def wrapped(self, ops, scope, selected_runtime):
        calls.append(tuple(ops))
        return original(self, ops, scope, selected_runtime)

    monkeypatch.setattr(GraphPolicy, "apply", wrapped)
    ops = target_operators(case, phase="prefill", tokens=2, scope="body",
                           apply_body_defaults=False)
    assert len(calls) == 1
    assert any(op.family == "quantize" for op in calls[0])
    norm = next(op for op in ops if op.op_id == "s0.l0.norm1")
    quant = next(op for op in ops if op.op_id == "s0.l0.qkv.quantize")
    matrix = next(op for op in ops if op.op_id == "s0.l0.qkv")
    state = lambda op: (dict(op.metadata).get("graph_captured"),
                        dict(op.metadata).get("graph_segment"))
    assert state(norm) == (False, 1)
    assert state(quant) == state(matrix) == (True, 2)
    assert dict(quant.metadata)["graph_replay_scope"] == "piecewise"


def test_model_program_uses_same_post_insertion_partition_as_body():
    from dataclasses import replace
    from test_typed_transforms_a5 import _parsed
    from mindie_sim.prediction.operator_analysis import model_program, target_operators

    case = _parsed("w8a8_int8")
    case = replace(case, runtime=replace(
        case.runtime, graph_mode="auto", graph_capture_mode="piecewise",
        piecewise_capture_keys=(8,), mixed_mode="mixed",
        splitting_ops=("rmsnorm",),
        lowering_passes=("activation_quantization",)))
    body = target_operators(case, phase="prefill", tokens=2, scope="body",
                            apply_body_defaults=False)
    program = model_program(case, phase="prefill", tokens=2)
    state = lambda op: (dict(op.metadata).get("graph_captured"),
                        dict(op.metadata).get("graph_segment"))

    def selected(ops):
        return (next(op for op in ops if op.op_id == "s0.l0.norm1"),
                next(op for op in ops if op.op_id == "s0.l0.qkv.quantize"),
                next(op for op in ops if op.op_id == "s0.l0.qkv"))

    assert program.work.replay_scope == "piecewise"
    for name, ops in (("body", body), ("model", program.work.ops)):
        norm, quant, matrix = selected(ops)
        assert state(norm)[0] is False
        assert state(quant) == state(matrix)
        assert state(quant)[0] is True
        assert state(quant)[1] > state(norm)[1]
        if name == "body":
            assert dict(quant.metadata)["graph_replay_scope"] == "piecewise"


def test_concat_public_output_and_two_source_views_resolve(a5_report):
    nodes = _nodes(a5_report)
    concat = nodes["rank:0/s0.l0.qkv_concatenate"]
    recurrent = nodes["rank:0/s0.l0.decode_recurrent_scan"]
    outputs = concat["values"]["outputs"]
    assert [(value["name"], value["shape"], value["dtype"], value["bytes"])
            for value in outputs] == [("output", [2048], "bf16", 4096.0)]
    public = {value["output"]: value for value in outputs}
    query = next(value for value in recurrent["values"]["inputs"]
                 if value["name"] == "query")
    key = next(value for value in recurrent["values"]["inputs"]
               if value["name"] == "key")
    for view, offset in ((query, 0), (key, 512)):
        assert view["producer"] == "s0.l0.qkv_concatenate"
        assert view["output"] == "output"
        assert view["shape"] == [1, 1, 4, 128]
        assert view["bytes"] == 1024.0
        assert view["storage_shape"] == [2048]
        assert view["storage_bytes"] == 4096.0
        assert view["offset"] == offset
        assert view["attrs"]["root_storage_shape"] == [2048]
        assert view["attrs"]["root_storage_bytes"] == 4096.0
        assert public[view["output"]]["bytes"] == 4096.0


def test_same_producer_two_views_keep_effective_and_root_bytes():
    producer_op = {
        "op_id": "p", "family": "tensor_copy", "deps": ("input",),
        "dtype": "bf16", "semantic_shape": [8], "local_shape": [8],
        "recipe": "copy", "phase": "prefill", "stage": 0, "layer": 0,
        "requests": ("r",), "returns": (),
        "metadata": {"source_layouts": [{
            "shape": [8], "strides": [1], "offset": 0,
            "storage_shape": [8]}]}}
    producer = operator_structure(producer_op)
    outputs = {value["output"]: value for value in producer["values"]["outputs"]}
    assert outputs["output"]["shape"] == [8]
    assert outputs["output"]["bytes"] == 16.0

    def consumer(name, offset):
        return operator_structure({
            "op_id": name, "family": "tensor_copy", "deps": ("p",),
            "dtype": "bf16", "semantic_shape": [4], "local_shape": [4],
            "recipe": "copy", "phase": "prefill", "stage": 0, "layer": 0,
            "requests": ("r",), "returns": (),
            "metadata": {
                "source_output": "output",
                "fused_origin_scopes": ({"kind": "input_ports",
                                         "ports": ({"op": "p", "output": "output"},)},),
                "source_layout": {"shape": [8], "strides": [1], "offset": 0,
                                  "storage_shape": [8]},
                "target_layout": {"shape": [4], "strides": [1], "offset": offset,
                                  "storage_shape": [8]}}},
            context={"outputs_by_id": {"p": producer["values"]["outputs"]}})

    first = consumer("c0", 0)["values"]
    second = consumer("c1", 4)["values"]
    for record, offset in ((first, 0), (second, 4)):
        source = record["inputs"][0]
        target = record["outputs"][0]
        assert (source["producer"], source["output"]) == ("p", "output")
        assert (source["shape"], source["bytes"]) == ([8], 16.0)
        assert target["shape"] == [4]
        assert target["bytes"] == 8.0
        assert target["storage_shape"] == [8]
        assert target["storage_bytes"] == 16.0
        assert target["offset"] == offset


def test_matrix_declared_weight_survives_with_named_unresolved_source():
    from mindie_sim.catalog.ops import gemm_op
    from mindie_sim.catalog.quant import quant_preset

    op = gemm_op(op_id="m", deps=("input",), stage=0, layer=0,
                 requests=("r",), m=2, n=3, k=4, quant=quant_preset("bf16"),
                 category="dense", phase="prefill", recipe="m", family="gemm")
    record = operator_structure(op.to_dict(include_work=False))
    inputs = {value["name"]: value for value in record["values"]["inputs"]}
    assert (inputs["activation"]["shape"], inputs["activation"]["bytes"]) == (
        [2, 4], 16.0)
    assert (inputs["weight"]["shape"], inputs["weight"]["bytes"]) == (
        [3, 4], 24.0)
    assert inputs["weight"]["attrs"]["weight_source_unresolved"] is True
    assert record["values"]["outputs"][0]["shape"] == [2, 3]
    assert record["values"]["outputs"][0]["bytes"] == 12.0
    assert "weight_source_unresolved" in {
        gap["code"] for gap in record["values"]["gaps"]}


def test_attention_output_is_query_only_and_context_independent(a5_report):
    node = next(node for node in a5_report["nodes"]
                if node["family"] == "attention" and node["contract"] is None)
    output = next(value for value in node["values"]["outputs"]
                  if value["name"] == "output")
    assert output["shape"] == [1, 4, 256]
    assert output["bytes"] == 2048.0
    probe = json.loads(json.dumps(node["operator"]))
    probe["metadata"]["gqa_decode"]["uniform_context_tokens"] = 31
    changed = operator_structure(probe)
    changed_output = next(value for value in changed["values"]["outputs"]
                          if value["name"] == "output")
    assert changed_output["shape"] == [1, 4, 256]
    assert changed_output["bytes"] == 2048.0
