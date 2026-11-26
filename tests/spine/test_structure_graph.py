"""Declared structure graphs: topology, scopes, subgraphs and native regions.

[Role] Guard ``export_structure``/``export_op_structure``: declared graph
  topology and payload identity, per-builder block/role scopes, native fusion
  regions and candidates, source-call expansion, subgraph membership and the
  prediction coverage of real retained descriptors.
[Invariants] Every node carries its own typed values, ports and state effects;
  a lone native name is never a selection without a declared or observed
  choice; a fusion keeps real work and its origin scopes; subgraph membership
  is a conservative DAG boundary with its own coverage kind; projection and
  work edits never silently drop nodes, deps or negative diagnostics.
[Timeouts & Locks] Pure in-memory exports and bounded retained-capture reads;
  no device, sleeps, threads or locks.
[Side-Effect Policy] Read-only declared cases and fixtures; every call returns
  a caller-owned snapshot with no host, environment or repository mutation.
"""
from dataclasses import replace
from functools import lru_cache
import json
from pathlib import Path

import pytest

from _support.captures import captured_report
from _support.collective_cases import case, exchange
from _support.operator_cases import hardware_case
from mindie_sim import Case
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_bytes
from mindie_sim.cohort_accuracy import compare_cohorts
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.graph import GraphPolicy
from mindie_sim.prediction.operator_analysis import (
    model_graphs, model_program, operator_costs, target_operators)
from mindie_sim.prediction.operators.attention import AttentionContract, ContextParallelPlan, make_attention_op
from mindie_sim.prediction.operators.attention._cp_tiling import HISTORY_DECODE_SITE, VLLM_REVISION
from mindie_sim.prediction.structure import parse_case, parse_model
from mindie_sim.prediction.structure_groups import declare_group
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from mindie_sim.structure import export_op_structure, export_structure
from mindie_sim.structure_implementations import operator_structure
from mindie_sim.structure.physical import export_structure_index
from mindie_sim.structure.physical import select_structure_targets

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
DECLARED_CASE = ROOT / "examples/cases/qwen35-a5.json"
NATIVE_ROLE = "grouped_gate_up_activation_quant"
ROUNDTRIP_CASES = ("qwen3.8-27b-media", "kimi-k3", "deepseek-v4.1-flash", "glm53-a5",
                   "kimi-k3-dspark")
FUSION_ROLES = (("v4f-a3-w8a8-tp8", "a3_v4f.csv", {"quant_lightning_indexer", "hc_pre",
                                                   "hc_post", NATIVE_ROLE}),
                ("glm53-a3-w4a8", "a3_glm53_prefill.csv", {NATIVE_ROLE}))
SCOPE_KEYS = {"structure_block", "structure_role"}
def _raw(name, *, selected=True):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    if selected:
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
            "mla_prolog"] = "vllm_ascend_mla_prolog_v3"
    return raw

def _case(name):
    return parse_case(_raw(name, selected=False))

def _report(value, *, phase="prefill", tokens=4096, context=0, sequences=1):
    return export_structure(value, phase=phase, tokens=tokens, context=context,
                            scope="body", sequences=sequences)

def _nodes(report):
    return {node["node_id"]: node for node in report["nodes"]}

def _groups(report, *roles):
    return [group for group in report["subgraphs"] if group["role"] in roles]

def _group(report, role):
    return next(group for group in report["subgraphs"] if group["role"] == role)

@lru_cache(maxsize=None)
def _rows(group, name):
    """Retained profiler fixture rows (file, or repository-relative stem)."""
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    base = Path(mindie_profiler.__file__).resolve().parents[1] / "tests" / "fixtures" / group
    target = base / name
    target = target if target.exists() else base / f"{name}.csv"
    assert target.exists(), f"retained profiler fixture unavailable: {target}"
    return tuple(kernel_rows(target))

def _one(rows, kind):
    return next(row for row in rows if row.kind == kind)

def _contract(row):
    from mindie_profiler.tensor_contracts import observed_contract
    return json.loads(json.dumps(observed_contract(row)))

@lru_cache(maxsize=None)
def _observed(group, name, kind):
    return tuple(item for item in (_contract(row) for row in _rows(group, name) if row.kind == kind) if item)

def _tiny_ops():
    raw = case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant",
                                        "residual_norm"]
    parsed = parse_case(raw)
    return parsed, list(target_operators(parsed, phase="prefill", tokens=8))

def _declare_residual(ops, implementation):
    fused = next(op for op in ops if op.op_id == "s0.l1.norm1")
    declare_group(ops, key="residual_and_next_norm", role="residual_norm",
                  family="normalization", members=(fused.op_id,), entries=(fused.op_id,),
                  anchor=fused.op_id, view={}, implementation=implementation, block=None)

def _source_case(name, roles, profile, *, peaks=None):
    raw = json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())
    if peaks:
        raw["machine"].setdefault("device", {}).setdefault("compute_peaks", {}).update(peaks)
    parsed_case = parse_case(raw)
    fusion = {"implementation": "declared_native_fusion", "options": {
        "roles": roles, "traffic_profile": profile}}
    kernels = {**parsed_case.runtime.kernels, "lowerings": {
        **parsed_case.runtime.kernels.get("lowerings", {}), "native_fusion": fusion}}
    return replace(parsed_case, runtime=replace(parsed_case.runtime, kernels=kernels))

def _without_scope(value):
    if not isinstance(value, dict):
        return [_without_scope(item) for item in value] if isinstance(value, (list, tuple)) else value
    out = {}
    for key, item in value.items():
        if key in SCOPE_KEYS:
            continue
        if key == "fused_origin_scopes" and isinstance(item, (list, tuple)):
            out[key] = [{k: _without_scope(v) for k, v in entry.items() if k not in {"block", "role"}}
                        if isinstance(entry, dict) else _without_scope(entry) for entry in item]
        else:
            out[key] = _without_scope(item)
    return out

def _msd_block(*, msd, layers=1):
    parsed_case = _case("glm53-a3-w4a8")
    block = next(plan for plan in parsed_case.model.layer_plans
                 if plan.ffn == "moe" and plan.owns_topk)
    model = replace(parsed_case.model, layers=layers, body_layers=layers, layer_plans=tuple(
        replace(block, index=index, kv_owner_layer_id=index, topk_owner_layer_id=index,
                index_k_owner_layer_id=index,
                candidate_source_layer_id=index if block.candidate_source_layer_id >= 0 else -1)
        for index in range(layers)))
    options = {"roles": [NATIVE_ROLE], "traffic_profile": "preserve_workspaces"}
    routing, machine = {}, parsed_case.machine
    if msd:
        options["program_profiles"] = {NATIVE_ROLE: "ascend_w4a8_msd"}
        routing = {"realizations": [{"scope": "body", "layer": index, "phase": "decode", "tokens": 1,
                                     "expert_rows": [1] * block.top_k + [0] * (block.num_experts - block.top_k)}
                                    for index in range(layers)]}
        machine = replace(machine, dies=tuple(replace(die, compute_peaks=(
            *die.compute_peaks, ("int4", 1_000_000_000_000.0))) for die in machine.dies))
    runtime = replace(parsed_case.runtime, pp_layer_partition=(layers,), routing=routing, kernels={
        **parsed_case.runtime.kernels, "lowerings": {
            **parsed_case.runtime.kernels.get("lowerings", {}), "native_fusion": {
                "implementation": "declared_native_fusion", "options": options}}})
    return replace(parsed_case, model=model, runtime=runtime, machine=machine)
def _same(actual, expected):
    assert actual == expected

@pytest.fixture(scope="module")
def declared_report():
    return export_structure(json.loads(DECLARED_CASE.read_text()), phase="decode",
                            tokens=1, context=17, scope="body")

def test_matrix_entry_and_quantize_ports(declared_report):
    node = next(node for node in declared_report["nodes"] if node["family"] == "gemm"
                and node["values"]["contract_kind"] == "matrix")
    values = node["values"]
    activation = next(value for value in values["inputs"] if value["name"] == "activation")
    weight = next(value for value in values["inputs"] if value["name"] == "weight")
    output, (m, n, k) = values["outputs"][0], node["contract"][1]
    _same((values["basis"], activation["layout"], activation["producer"]), ("contract", "strided", "s0.l0.linear_qkvz.quantize"))
    _same((activation["shape"], weight["shape"], output["shape"]), ([m, k], [n, k], [m, n]))
    _same((activation["quant"], output["dtype"], weight["layout"], output["layout"]), ("int8", "bf16", None, None))
    scales, w_scale = values["quantization"]["scales"], next(v for v in values["declared"] if v["name"] == "weight_scale")
    _same((scales["activation"]["format"], scales["weight"]["format"], scales["weight"]["storage_format"]), ("fp32", "bf16", "fp32"))
    _same((w_scale["role"], w_scale["dtype"], w_scale["attrs"]["storage_format"], w_scale["attrs"]["association"]), ("scale", "bf16", "fp32", "weight"))
    _same([candidate["native"] for candidate in node["native_candidates"]], ["QuantBatchMatmulV3", "QuantBatchMatmulV4"])
    quant = _nodes(declared_report)["rank:0/s0.l0.linear_qkvz.quantize"]
    outputs = {value["output"]: value for value in quant["values"]["outputs"]}
    _same((outputs["payload"]["dtype"], outputs["activation_scales"]["bytes"]), ("int8", 4.0))
    conflict = json.loads(json.dumps(quant["operator"]))
    boundary = conflict["metadata"]["fused_origin_scopes"][0]["boundary"]
    boundary["typed_inputs"] = [*boundary.get("typed_inputs", ()), {"producer": "s0.l0.norm1", "output": "activation_scales", "role": "scale", "dtype": "fp32", "shape": [1], "source": "synthetic_conflict_probe"}]
    conflicts = [value for value in operator_structure(conflict, quant["contract"])["values"]["declared"] if value["attrs"].get("conflict_with_output")]
    assert conflicts[0]["output"] == "activation_scales"

def test_native_regions_and_attention_abi_keep_candidates_unselected(declared_report):
    group = _group(declared_report, "qkv_norm_rope")
    _same((group["implementation"]["selection"], group["native_execution"]["kind"],
           group["physical_work"]["launches"]), (None, "candidate_set_only", 4))
    region = next(group for group in declared_report["subgraphs"]
                  if len(group["native_candidates"]) > 1)
    assert all(candidate["selected"] is False for candidate in region["native_candidates"])
    attention = [node for node in declared_report["nodes"]
                 if node["family"] == "attention" and node["contract"] is None]
    assert attention
    for node in attention:
        attrs = {item["name"]: item["value"] for item in node["values"]["semantic_attrs"]}
        _same((attrs["query_tokens"], attrs["context_tokens"], attrs["query_heads"],
               attrs["causal"]), (1, 17, 4, True))
        assert {"attention_tensor_abi_unavailable", "attention_native_slots_unresolved"} <= {gap["code"] for gap in node["representation_gaps"]}
def test_region_and_native_candidates_require_an_explicit_selection():
    case, ops = _tiny_ops()
    _declare_residual(ops, {"native_kind": "AddRmsNorm",
                            "selection": "explicit_caller_implementation"})
    report = export_op_structure(ops, model=case.model.to_dict(), execution={"phase": "prefill"})
    group = report["subgraphs"][0]
    member = next(node for node in report["nodes"] if node["node_id"] == group["node_ids"][0])
    _same((group["native_execution"]["kind"], group["native_execution"]["selected"]["native"],
           member["native_execution"]["selected"]),
          ("declared_native_region", "AddRmsNorm", None))
    chosen = operator_structure({
        "op_id": "two", "family": "rmsnorm", "recipe": "rmsnorm", "local_shape": [1, 8],
        "dtype": "bf16", "source_boundary": "semantic",
        "metadata": {"native_implementation": {"native_kinds": ["NativeA", "NativeB"],
                                               "selection": "NativeA"}}})
    _same((chosen["native_execution"]["kind"],
           {c["native"]: c["selected"] for c in chosen["native_candidates"]}),
          ("source_selected_native", {"NativeA": True, "NativeB": False}))
def _cp_attention_call_op():
    history = ContextParallelPlan(
        VLLM_REVISION, HISTORY_DECODE_SITE, "decode", "history_attention",
        (("query", (1, 1, 128), 2),), dcp_size=2,
    )
    contract = AttentionContract(1, 1, 1, 128, 128, selected_tokens=33, cp_plan=history)
    return make_attention_op(contract, op_id="s0.l0.attention", deps=("input",), dtype="bf16")


def test_real_lower_fusion_and_source_call_identity():
    case = _source_case("v4f-a3-w8a8-tp8", ["hc_pre"], "hc_pre_m_k_split_175ef4d4",
                        peaks={"hf32": 2.0e14})
    report = export_structure(case, phase="prefill", tokens=8, scope="body")
    group = next(group for group in report["subgraphs"] if group.get("resource_boundary"))
    members = [node for node in report["nodes"] if node["node_id"] in group["node_ids"]]
    _same((len(group["node_ids"]) > 1, group["resource_boundary"]["kind"],
           all(member["native_execution"]["kind"] == "covered_by_fused_region"
               for member in members)), (True, "selected_native_fusion", True))
    calls = export_op_structure((_cp_attention_call_op(),), model={},
                                execution={"phase": "decode"}, native_evidence=True)
    call = next(row for row in calls["source_calls"])
    children = [node for node in calls["nodes"] if node["node_id"] in call["children"]]
    kernel_children = [child for child in children if child["boundary"] == "kernel"]
    _same((call["operator"]["source_boundary"], [child["recipe"] for child in kernel_children],
           all(child["lineage"]["source_call_id"] == call["node_id"] for child in children)),
          ("call", ["fused_infer_attention_score"], True))

def test_index_and_pure_model_projection_preserve_values():
    raw = json.loads(DECLARED_CASE.read_text())
    args = dict(phase="decode", tokens=1, context=17, scope="body")
    full, index = export_structure(raw, **args), export_structure_index(raw, **args)
    node_ids = ("rank:0/s0.l11.qkv.value", "rank:0/s0.l0.linear_qkvz.quantize")
    fields = ("values", "semantic_refs", "native_candidates", "native_execution")
    for node_id in node_ids:
        full_node = next(node for node in full["nodes"] if node["node_id"] == node_id)
        index_node = next(node for node in index["nodes"] if node["node_id"] == node_id)
        assert {field: index_node[field] for field in fields} == { field: full_node[field] for field in fields}
    assert index["graph_id"] == full["graph_id"] == ( select_structure_targets(raw, list(node_ids), **args)["graph_id"])
    raw = case()
    raw.pop("workload", None)
    raw["hardware"] = raw.pop("machine")
    parsed = parse_model(raw)
    with pytest.raises(Exception, match="case_missing_fields"):
        parse_case({"machine": raw["hardware"], "model": raw["model"],
                    "runtime": raw["runtime"]})
    program = model_program(raw, phase="decode", tokens=1, context=8, sequences=1)
    families = {op.family for op in program.work.ops}
    _same((parsed.workload.requests, program.assembly, "logits" in families,
           bool({"sampling", "acceptance", "token_commit"} & families)),
          ((), "model", True, False))
def test_cross_layer_mc2_and_norm_quant_fusions():
    raw = case()
    raw["runtime"]["lowering_passes"] = ["activation_quantization", "norm_quant", "residual_norm"]
    parsed_case = parse_case(raw)
    ops = list(target_operators(parsed_case, phase="prefill", tokens=8))
    fused = next(op for op in ops if op.op_id == "s0.l1.norm1")
    assert {(origin["layer"], origin["block"]) for origin in dict(fused.metadata)["fused_origin_scopes"]} == {(0, "ffn"), (1, "attention")}
    _declare_residual(ops, {"native_kind": "AddRmsNorm"})
    group = export_op_structure(ops, model=parsed_case.model.to_dict(),
                                execution={"phase": "prefill"})["subgraphs"][0]
    assert (group["layer"], group["layer_ids"]) == (None, [0, 1])
    source = _case("v4f-a3-w8a8-tp8")
    chunked = replace(source, runtime=replace(source.runtime, moe_recipe="mc2", mc2_chunks=3))
    groups = [group for group in _groups(_report(chunked, tokens=16), NATIVE_ROLE)
              if group["layer"] == 0]
    assert len(groups) == 3 and len(set().union(*[set(g["node_ids"]) for g in groups])) == sum( len(g["node_ids"]) for g in groups)
@pytest.mark.requires_profiler
def test_compressor_precision_override_preserves_source_bytes_and_valid_row_work(profiler_api):
    baseline = _case("v4f-a3-w8a8-tp8")
    # A caller declaration of the source-supported FP32 norm branch; no
    # descriptor populates the model or derives row geometry.
    selected = replace(baseline, model=replace(baseline.model, layer_plans=tuple(
        replace(plan, compressor_norm_dtype="fp32") for plan in baseline.model.layer_plans)))
    before, after = _report(baseline), _report(selected)
    observed = _observed("fusion_contracts", "a3_v4f.csv", "Compressor")
    old_groups, groups = _groups(before, "compressor"), _groups(after, "compressor")
    assert observed and old_groups and all(group["contract"] in observed for group in old_groups)
    assert all(group["contract"] in observed for group in groups)
    assert {group["contract"][-1][0][-1] for group in groups} == {128, 512}
    plan = next(plan for plan in selected.model.layer_plans
                if plan.owns_long_kv and plan.owns_index_k)
    old = {term.name: term for term in layer_weight_terms(
        baseline.model.layer_plans[plan.index], baseline.model)}
    new = {term.name: term for term in layer_weight_terms(plan, selected.model)}
    for name in ("compressor_norm", "indexer_k_norm"):
        assert term_bytes(new[name], selected.model.quant, selected.runtime.parallel) == ( 2 * term_bytes(old[name], baseline.model.quant, baseline.runtime.parallel))
        assert term_bytes(new[name], selected.model.quant, selected.runtime.parallel, source=True) == term_bytes( old[name], baseline.model.quant, baseline.runtime.parallel, source=True)
def test_indexer_norm_and_residual_source_work_keep_distinct_semantics():
    glm = _case("glm53-a3-w4a8")
    # This pins the retained eager FP32 LayerNorm decomposition: no optional
    # automatic fusion owns the boundary.
    metadata = dict(glm.runtime.metadata)
    metadata["execution_policy"] = {"automatic_optional_fusions": False}
    glm = replace(glm, runtime=replace(glm.runtime, metadata=tuple(metadata.items())))
    ops = target_operators(glm, phase="prefill", tokens=32)
    for op in [op for op in ops if op.recipe == "index_norm"]:
        width = dict(op.metadata)["reduction_width"]
        input_cast = {value.op_id: value for value in ops}[op.deps[0]]
        output_cast = next(value for value in ops if value.recipe == "index_norm_output_cast"
                           and op.op_id in value.deps)
        _same((op.family, op.vector_ops, op.dtype, input_cast.mte3_bytes,
               output_cast.dtype), ("layernorm", 8 * 32 * width, "fp32",
                                    32 * width * 4, "bf16"))
    v4 = _report(_case("v4f-a3-w8a8-tp8"), tokens=32)
    nodes, group = _nodes(v4), _group(v4, "quant_lightning_indexer")
    query = next(nodes[key] for key in group["external_inputs"]
                 if nodes[key]["family"] == "quantize")
    _same((query["operator"]["metadata"]["output_format"],
           {nodes[key]["family"] for key in group["node_ids"]}), ("int8", {"indexer"}))
def test_kda_chunk_boundary_is_multi_launch_unknown():
    case = _case("kimi-k3-a5")
    staged = replace(case, runtime=replace(case.runtime, kernels={**case.runtime.kernels, "lowerings": {**case.runtime.kernels.get("lowerings", {}), "kda": "vllm_ascend_kda_in_kernel_gates"}}))
    groups = _groups(_report(staged, tokens=64), "kda_chunk")
    assert groups and all(group["contract"] is None and group["coverage"]["kind"] == "multi_launch_stage_unknown" for group in groups)

@pytest.mark.requires_profiler
def test_kda_recurrent_baseline_differs_from_in_kernel_gate_capture(profiler_api):
    observed = _observed("fusion_contracts", "a5_kimi_k3.csv", "RecurrentKda")
    assert all(contract[2][1][1] == "fp32" for contract in observed)
    baseline = _groups(_report(_case("kimi-k3-a5"), phase="decode", tokens=1, context=32768), "kda_recurrent")
    assert baseline and all(group["contract"] not in observed for group in baseline)

@pytest.mark.requires_profiler
def test_kda_recurrent_in_kernel_gates_match_captured_contract(profiler_api):
    case = _case("kimi-k3-a5")
    observed = _observed("fusion_contracts", "a5_kimi_k3.csv", "RecurrentKda")
    staged = replace(case, runtime=replace(case.runtime, kernels={**case.runtime.kernels, "lowerings": {**case.runtime.kernels.get("lowerings", {}), "kda": "vllm_ascend_kda_in_kernel_gates"}}))
    decode = _report(staged, phase="decode", tokens=1, context=32768)
    recurrent, nodes = _groups(decode, "kda_recurrent"), _nodes(decode)
    assert observed and recurrent and all(group["contract"] in observed for group in recurrent)
    assert all({"qk_l2norm", "delta_gates"} <= {nodes[key]["recipe"] for key in group["node_ids"]} for group in recurrent)

def test_chunked_gdn_region_boundary_is_explicit():
    raw = _case("qwen35-a3-tp4")
    gdn = _groups(_report(raw, tokens=64), "gdn_chunk")
    assert gdn and all(group["contract"] is None and group["coverage"]["kind"] == "native_boundary_unverified" for group in gdn)
    kernels = {**raw.runtime.kernels, "lowerings": {
        **raw.runtime.kernels.get("lowerings", {}), "native_fusion": {"implementation": "declared_native_fusion", "options": {"roles": ["gdn_chunk"]}}}}
    with pytest.raises(ValueError, match="requires_complete_single_kernel"):
        _report(replace(raw, runtime=replace(raw.runtime, kernels=kernels)), tokens=64)
@pytest.mark.parametrize("name", ("dense_gqa", "qwen35-a3-tp4", "kimi-k3-a5",
                                   "v4f-a3-w8a8-tp8"))
def test_block_and_call_scopes_leave_work_and_dependencies_unchanged(name, monkeypatch):
    import mindie_sim.prediction.frontend as frontend
    raw = case() if name == "dense_gqa" else _raw(name, selected=False)
    tagged = target_operators(raw, phase="prefill", tokens=8)
    report = export_structure(raw, phase="prefill", tokens=8, scope="body")
    # Disable only the metadata annotator: all actual work, tensor fields,
    # placement and dependency edges of the same builders must be identical.
    monkeypatch.setattr(frontend, "_scope", lambda ops, block, role="body": ops)
    original = target_operators(raw, phase="prefill", tokens=8)
    assert [_without_scope(op.to_dict()) for op in tagged] == [ _without_scope(op.to_dict()) for op in original]
    assert all(node["block"] == node["operator"]["metadata"]["structure_block"] and node["role"] == node["operator"]["metadata"]["structure_role"] for node in report["nodes"])
@pytest.mark.requires_profiler
@pytest.mark.parametrize("phase,tokens,context,files", (
    ("prefill", 4096, 0, ("a3_glm53.csv", "a3_glm53_prolog_no_index.csv")),
    ("decode", 1, 4096, ("a3_glm53_decode.csv",))))
def test_real_prolog_and_indexer_match_declared_groups_without_erasing_branches(phase, tokens, context, files, profiler_api):
    report = export_structure(_raw("glm53-a3-w4a8"), phase=phase, tokens=tokens,
                              context=context, scope="body")
    groups = report["subgraphs"]
    full = [g for g in groups if g["role"] == "mla_prolog" and g["structure"]["indexer"]]
    shared = [g for g in groups if g["role"] == "mla_prolog" and not g["structure"]["indexer"]]
    indexers = _groups(report, "lightning_indexer")
    observed = [contract for name in files for row in _rows("attention_cache_contracts", name)
                for contract in (_contract(row),) if contract]
    with_query = next(c for c in observed if c[0] == "mla_prolog_active" and c[-1][-1][0])
    no_query = next(c for c in observed if c[0] == "mla_prolog_active" and not c[-1][-1][0])
    lightning = next(c for c in observed if c[0] == "lightning_indexer_active")
    assert full and shared and indexers
    assert all(g["contract"] == json.loads(json.dumps(with_query)) for g in full)
    assert all(g["contract"] == json.loads(json.dumps(no_query)) for g in shared)
    assert all(g["contract"] == json.loads(json.dumps(lightning)) for g in indexers)
    # The checkpoint's declared BF16 precision is not overridden by a capture.
    if phase == "decode":
        actual = _contract(_one(_rows("body_attention", "kimi_a5_full_model.csv"), "MlaPrologV3"))
        prolog = _groups(export_structure(_raw("kimi-k3-a5"), phase="decode", tokens=1, context=32768, scope="body"), "mla_prolog")
        assert prolog and actual[2][1] == "fp8_e4m3" and all(g["contract"][2][1] == "bf16" and g["contract"] != actual for g in prolog)

def test_prolog_boundary_disabled_conversions_and_custom_lowering_order():
    report = export_structure(_raw("glm53-a3-w4a8"), phase="prefill", tokens=4096,
                              context=0, scope="body")
    for group in _groups(report, "mla_prolog"):
        prefix = f"rank:0/s0.l{group['layer']}"
        outputs = {prefix + suffix for suffix in (".rotary", ".kv_store", ".q_absorb")}
        if group["structure"]["indexer"]:
            outputs.add(prefix + ".q_b.quantize")
        _same((group["external_inputs"], set(group["external_outputs"])),
              ([prefix + ".q_a.quantize"], outputs))
    for passes in ([], ["norm_quant", "residual_norm"]):
        raw = _raw("glm53-a3-w4a8")
        raw["runtime"]["lowering_passes"] = passes
        with pytest.raises(ModelingError,
                           match="quantized_operands_require_activation_quantization"):
            export_structure(raw, phase="prefill", tokens=4096, context=0, scope="body")
    # BF16 projection operands already use the model activation format.
    raw["model"]["quantization"] = "bf16"
    bf16 = export_structure(raw, phase="prefill", tokens=4096, context=0, scope="body")
    full = next(g for g in bf16["subgraphs"]
                if g["role"] == "mla_prolog" and g["structure"]["indexer"])
    _same((full["contract"][-1][-1][1], "rank:0/s0.l0.q_norm" in full["external_outputs"]),
          ("bf16", True))
def test_selected_quantized_prolog_rejects_unquantized_indexer_shared_output():
    case = parse_case(_raw("glm53-a3-w4a8"))
    quant = replace(case.model.quant, exclusions=(*case.model.quant.exclusions, "indexer_fp8"))
    excluded = replace(case, model=replace(case.model, quant=quant))
    with pytest.raises(ModelingError, match="mla_prolog_v3_query_norm_indexer_activation_mismatch"):
        export_structure(excluded, phase="prefill", tokens=4096, context=0, scope="body")
    # The direct frontend entry does not run parse_case.
    with pytest.raises(ModelingError, match="mla_prolog_v3_query_norm_indexer_activation_mismatch"):
        Frontend().body(excluded.model, excluded.runtime,
                        (RequestView("probe", 32, 0, False, "prefill"),), ready="input", machine=excluded.machine)
    # When both projections consume BF16, the real shared producer is q_norm.
    quant = replace(quant, exclusions=(*quant.exclusions, "mla"))
    report = export_structure(replace(excluded, model=replace(excluded.model, quant=quant)),
                              phase="prefill", tokens=32, context=0, scope="body")
    group = next(g for g in report["subgraphs"]
                 if g["role"] == "mla_prolog" and g["structure"]["indexer"])
    _same((group["contract"][-1][-1][1], _nodes(report)["rank:0/s0.l0.index_q"]["deps"]),
          ("bf16", ["rank:0/s0.l0.q_norm"]))

def test_absorption_bytes_and_group_expansion_conserve_declared_work():
    baseline = parse_case(_raw("glm53-a3-w4a8", selected=False))
    selected = parse_case(_raw("glm53-a3-w4a8"))
    before = {op.op_id: op for op in target_operators(baseline, phase="prefill", tokens=32)}
    after = {op.op_id: op for op in target_operators(selected, phase="prefill", tokens=32)}
    quant, parallel = selected.model.quant, selected.runtime.parallel
    assert quant.format_for("mla_absorb") == quant.source_format_for("mla_absorb") == "bf16"
    for plan in selected.model.layer_plans:
        term = next(t for t in layer_weight_terms(plan, selected.model) if t.name == "kv_b")
        expected = term_bytes(term, quant, parallel)
        pair = [after[f"s0.l{plan.index}" + suffix] for suffix in (".q_absorb", ".v_expand")]
        assert expected == term_bytes(term, quant, parallel, source=True) == sum( dict(op.metadata)["packed_weight_bytes"] for op in pair)
        for op in pair:
            _same((op.flops, dict(op.metadata)["weight_format"],
                   op.op_id + ".quantize" in after), (before[op.op_id].flops, "bf16", False))
    for source in (False, True):
        expected_delta = sum(
            term_bytes(next(t for t in layer_weight_terms(plan, selected.model) if t.name == "kv_b"),
                       quant, parallel, source=source)
            - term_bytes(next(t for t in layer_weight_terms(baseline.model.layer_plans[plan.index], baseline.model) if t.name == "kv_b"),
                         baseline.model.quant, parallel, source=source)
            for plan in selected.model.layer_plans
        )
        assert (sum(stage_weight_bytes(selected.model, parallel, source=source)) - sum(stage_weight_bytes(baseline.model, parallel, source=source))) == expected_delta
    call = _cp_attention_call_op()
    expansion = gemm_op(op_id="s0.l0.v_expand", deps=(call.op_id,), stage=0, layer=0,
                        requests=("r0",), m=1, n=128, k=128, quant=quant_preset("bf16"),
                        category="dense", phase="decode", recipe="v_expand", family="projection")
    ops = [call, expansion]
    declaration = dict(model={}, execution={"phase": "decode"})
    original = export_op_structure(ops, **declaration)
    declare_group(ops, key="s0.l0.decode", role="declared_decode", family="mla",
                  members=(call.op_id, expansion.op_id), entries=(call.op_id,),
                  anchor=expansion.op_id, view={},
                  implementation={"source_selection": "explicit_test_of_existing_source_call"})
    report = export_op_structure(ops, **declaration)
    group = report["subgraphs"][0]
    source_call = next(row for row in report["source_calls"]
                       if row["node_id"] == f"rank:0/{call.op_id}")
    assert set(group["node_ids"]) == { *source_call["children"], source_call["node_id"], f"rank:0/{expansion.op_id}"}
    assert [node["node_id"] for node in report["nodes"]] == [ node["node_id"] for node in original["nodes"]]
    changed = export_op_structure(
        [replace(op, flops=op.flops + 1234) for op in ops], **declaration)["subgraphs"]
    assert changed[0]["physical_signature"] != group["physical_signature"]
@pytest.mark.requires_profiler
@pytest.mark.parametrize("msd", (False, True))
def test_real_descriptor_cohorts_preserve_declared_binding(profiler_api, msd):
    case = _msd_block(msd=msd)
    structure = export_structure(case, phase="decode", tokens=1, scope="body")
    group = _group(structure, NATIVE_ROLE)
    gate = next(node for node in structure["nodes"] if node["family"] == "grouped_gemm")
    assert group["physical_work"]["launches"] == 1 and gate["event_matchable"] is False
    assert gate["contract_scope"]["prediction_coverage"]["reason"] == "quantized_epilogue_unmodeled"
    mapping, events = captured_report(case, "a3_glm53_decode.csv",
                                      "GroupedMatmulSwigluQuantV2", structure=structure, repeat=1)
    row = compare_cohorts(case, mapping)["windows"][0]["cohorts"][0]
    _same((row["target_kind"], row["observed"]["activity_sum_ns"]),
          ("subgraph", events[0].duration_ns))
    if msd:
        assert row["resource_boundary"]["program_profile"] == "ascend_w4a8_msd"
        assert row["status"] == "compared"
    else:
        _same((row["status"], row["reason"], row["comparison"]),
              ("reference_only", "quantized_epilogue_unmodeled", None))

@pytest.mark.requires_profiler
@pytest.mark.parametrize("source_down", (False, True))
def test_down_source_program_removes_only_its_implemented_epilogue_gap(profiler_api, source_down):
    case = _msd_block(msd=True)
    if source_down:
        case = replace(case, runtime=replace(case.runtime, kernels={
            **case.runtime.kernels, "lowerings": {
                **case.runtime.kernels["lowerings"],
                "grouped_gemm": {"implementation": "ascend_w4a8_msd_grouped_down",
                                 "recipes": ["grouped_down", "balanced_grouped_down"]}}}))
    structure = export_structure(case, phase="decode", tokens=1, scope="body")
    recipe = "ascend_w4a8_msd_grouped_down" if source_down else "grouped_down"
    down = next(row for row in structure["nodes"] if row["recipe"] == recipe)
    mapping, events = captured_report(case, "a3_glm53_decode.csv", "GroupedMatmul", structure=structure, repeat=1)
    row = compare_cohorts(case, mapping)["windows"][0]["cohorts"][0]
    assert down["physical_work"]["launches"] == 1
    _same((row["target_kind"], row["observed"]["activity_sum_ns"]), ("node", events[0].duration_ns))
    if source_down:
        assert row["status"] == "compared" and down["contract_scope"]["implemented_quantized_epilogue"]["program"] == recipe
        # 8 active experts x m_tile=2 (double_rows) padded to Arch22 INT4 subtile_m=16.
        assert down["physical_work"]["matrix_flops"] == {"int4": 2 * 16 * 8 * 6144 * 256}
    else:
        _same((row["status"], row["reason"], row["comparison"]), ("reference_only", "quantized_epilogue_unmodeled", None))

def test_typed_state_and_matrix_representations_keep_their_gaps(declared_report):
    fused = next(node for node in export_structure(_tiny_ops()[0], phase="prefill", tokens=8, scope="body")["nodes"] if node["family"] == "add_rmsnorm")
    outputs = {value["name"]: value for value in fused["values"]["outputs"]}
    _same((fused["values"]["basis"], outputs["residual"]["role"], fused["values"]["selection"]), ("typed_records", "residual", {"basis": "selected_software_plan", "kind": "residual_norm"}))
    nmap = _nodes(declared_report)
    assert "rank:0/s0.l0.qkv_concatenate" not in nmap
    rin = {v["name"]: v for v in nmap["rank:0/s0.l0.decode_recurrent_scan"]["values"]["inputs"] if v["name"] in {"query", "key"}}
    for view, off in ((rin["query"], 0), (rin["key"], 512)):
        _same((view["producer"], view["output"], view["shape"], view["bytes"], view["storage_shape"], view["storage_bytes"], view["offset"], view["attrs"]["root_storage_shape"], view["attrs"]["root_storage_bytes"]), ("s0.l0.short_convolution", "output", [1, 1, 4, 128], 1024.0, [1, 2048], 4096.0, off, [1, 2048], 4096.0))
    node = next(node for node in declared_report["nodes"] if node["family"] == "recurrent_vector")
    inputs = {value["name"]: value for value in node["values"]["inputs"]}
    _same((inputs["state"]["dtype"], inputs["state"]["shape"], inputs["state"]["storage_shape"], inputs["state"]["storage_bytes"], "state_capacity_not_declared" in {gap["code"] for gap in node["values"]["gaps"]}, [effect["action"] for effect in node["values"]["state_effects"]]), ("fp32", [1, 8, 128, 128], None, None, True, ["read", "write", "commit"]))
    probe = json.loads(json.dumps(node["operator"])); probe["metadata"]["gdn_state_contract"].update(state_capacity=9, slots=9, active_slots=[0, 1])
    expanded = next(value for value in operator_structure(probe, node["contract"])["values"]["inputs"] if value["name"] == "state")
    assert (expanded["shape"], expanded["bytes"], expanded["storage_shape"], expanded["storage_bytes"]) == ([2, 8, 128, 128], 1048576.0, [9, 8, 128, 128], 4718592.0)
    declared = gemm_op(op_id="m", deps=("input",), stage=0, layer=0, requests=("r",), m=2, n=3, k=4, quant=quant_preset("bf16"), category="dense", phase="prefill", recipe="m", family="gemm")
    inputs = {value["name"]: value for value in operator_structure(declared.to_dict(include_work=False))["values"]["inputs"]}
    assert (inputs["weight"]["attrs"]["weight_source_unresolved"], inputs["activation"]["shape"]) == (True, [2, 4])

def test_operator_accounting_and_model_graphs_preserve_declared_transport():
    raw = case(tp=2, dp=1)
    raw["machine"]["links"][0].update(transaction_bytes=64, protocol_overhead_bytes=0, capacity_bytes_per_ms=1_000_000)
    raw["runtime"]["kernels"] = {"families": {"alltoall": {"kind": "direct_peer_collective", "max_inflight_per_rank": 2, "chunk_bytes": 64}}}
    report = operator_costs(raw, (exchange(128),))["operators"][0]
    _same((report["communication"]["max_inflight_per_rank"], report["communication"]["messages"], report["communication"]["wire_bytes"]), (2, 4, 256.0))
    raw = case(); raw.pop("workload"); raw["hardware"] = raw.pop("machine")
    model = model_graphs(raw, phase="decode", tokens=1, context=8, sequences=1, scope="model")
    _same((type(model).__name__, model.programs.keys(), [op.op_id for op in model[0]]), ("ModelGraphSet", {0}, [op.op_id for op in model.program(0).work.ops]))
    assert model.ports[0].tensor_out

def test_piecewise_capture_keys_partition_quantized_segments(monkeypatch):
    raw = hardware_case()
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 2e6
    parsed = parse_case(raw)
    case = replace(parsed, model=replace(parsed.model, quant=quant_preset("w8a8_int8")))
    case = replace(case, runtime=replace(case.runtime, graph_mode="auto", graph_capture_mode="piecewise", piecewise_capture_keys=(8,), mixed_mode="mixed", splitting_ops=("rmsnorm",), lowering_passes=("activation_quantization",)))
    calls, original = [], GraphPolicy.apply

    def wrapped(self, ops, scope, selected_runtime):
        calls.append(tuple(ops))
        return original(self, ops, scope, selected_runtime)

    monkeypatch.setattr(GraphPolicy, "apply", wrapped)
    body = target_operators(case, phase="prefill", tokens=2, scope="body", apply_body_defaults=False)
    state = lambda op: (dict(op.metadata).get("graph_captured"), dict(op.metadata).get("graph_segment"))
    quant = next(op for op in body if op.op_id == "s0.l0.qkv.quantize")
    assert (len(calls), state(quant), dict(quant.metadata)["graph_replay_scope"]) == (1, (True, 2), "piecewise")
    assert model_program(case, phase="prefill", tokens=2).work.replay_scope == "piecewise"

def test_layer_catalog_cache_and_quant_overrides_change_work_together():
    original = Case.from_mapping(case(prompt=8, output=1))
    changed = replace(original, model=replace(original.model, layer_plans=tuple(
        replace(layer, intermediate_size=128) for layer in original.model.layer_plans)))
    plan = IterationPlan((RequestView("r", 8, 0, True, "prefill"),))
    before = Frontend()(plan, original.model, original.runtime, original.machine)
    after = Frontend()(plan, changed.model, changed.runtime, changed.machine)
    expected_weights = 2 * 3 * 32 * (128 - 64) * 2
    assert sum(Memory(changed).weights) - sum(Memory(original).weights) == expected_weights
    assert sum(op.flops for op in after.ops) - sum(op.flops for op in before.ops) == ( 2 * 8 * expected_weights / 2)
    window = LayerPlan(0, attention="mla", cache_kind="window_compressed", window=4,
                       compress_ratio=4, kv_lora_rank=8, kv_owner_layer_id=0, owns_long_kv=True)
    parsed_case = Case.from_mapping(case())
    for tokens in (3, 4, 7, 8, 19):
        result = layer_cache_components(parsed_case.model, window, ParallelSpec(), tokens,
                                        capacity_padded=False)
        _same((result["window"], result["compressed_kv"]),
              (min(tokens, 4) * 8 * 2, (tokens // 4) * 8 * 2))
    common = dict(op_id="m", deps=(), stage=0, layer=0, requests=(), m=3, n=64, k=128,
                  category="routed_expert", phase="decode")
    packed = gemm_op(quant=QuantSpec("int8", weight_format="int8",
                                    expert_weight_format="int8"), **common)
    assert (packed.flops, packed.dtype, gemm_op(quant=QuantSpec("bf16"), **{**common, "m": 0}).flops) == (2 * 3 * 64 * 128, "int8", 0)
