"""Exported node, port, tensor, quant and physical-work contracts.

[Role] Guard the exported contract surface: the projected structure index and
  selected-target projection, independent matrix/native/quant descriptors
  compared with retained captures, the interned JSON payload, and the resource
  identity that separates physical work from occurrence labels.
[Invariants] A projection never edits its caller's case or drops identity:
  ``graph_id``, record counts, native contracts and physical signatures stay
  equal to the full report.  Each descriptor carries its own precision,
  geometry and bank facts, so a captured kernel with different dtype, bank,
  batch or routing evidence is never inherited by similarity.
[Timeouts & Locks] Pure in-memory exports plus bounded retained-capture row
  reads; no device, sleeps, threads or locks.
[Side-Effect Policy] Read-only fixtures and inputs; every builder returns a
  caller-owned snapshot.  No host path, environment or repository mutation.
"""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from _support.collective_cases import case
from _support.kernel_invocation import copy_op, curve
from _support.moe_hc_mc2_fixtures import mc2_case
from _support.operator_cases import hardware_case, mla_case
from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.operator_analysis import operator_costs, target_graphs, target_operators
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.services import service_identity
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_op_structure, export_structure
from mindie_sim.structure_contracts import comparison_scope, predicted_contract
from mindie_sim.structure_index import export_structure_index, index_digest
from mindie_sim.structure_physical import node_physical
from mindie_sim.structure_selection import select_structure_targets

pytestmark = pytest.mark.tier0
ROOT = Path(__file__).resolve().parents[2]
ROLE = "grouped_gate_up_activation_quant"
DENSE_CASES = (("glm53-a3-w4a8", "a3_glm53_prefill", 4096, (21, 78)),
               ("qwen36-a3-tp4", "a3_qwen36", 8192, (48, 64)),
               ("qwen38-a5", "a5_qwen38", 16384, (64, 64)))
GROUPED_DOWN = (("glm53-a3-w4a8", "a3_glm53_prefill", 4096, 75),
                ("qwen35-a3-tp4", "a3_qwen35", 8192, 40))
GATED_NORMS = (("qwen35-a3-tp4", "a3_qwen35.csv", 8192),
               ("qwen36-a3-tp4", "a3_qwen36.csv", 8192),
               ("qwen38-a5", "a5_qwen38.csv", 16384))
GEMMA_NORMS = (("qwen35-a3-tp4", "a3_qwen35.csv"), ("qwen36-a3-tp4", "a3_qwen36.csv"))
CACHE_PROJECTIONS = (("qwen35-a3-tp4", "a3_qwen35.csv", 8192),
                     ("qwen36-a3-tp4", "a3_qwen36.csv", 8192),
                     ("qwen38-a5", "a5_qwen38.csv", 16384))
INDEX_FIELDS = ("contract", "contract_scope", "physical_signature", "physical_work",
                "deps", "lineage", "implementation", "structure")

def case_json(name):
    return json.loads((ROOT / "examples/cases" / (name + ".json")).read_text())

def parsed(name):
    return parse_case(case_json(name))

def operators(name, tokens, *, phase="prefill"):
    return target_operators(parsed(name), phase=phase, tokens=tokens)

def _rows(group, name):
    """Retained profiler fixture rows (file, or directory stem), read in place."""
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    base = Path(mindie_profiler.__file__).resolve().parents[1] / "tests" / "fixtures" / group
    target = base / name
    target = target if target.exists() else base / f"{name}.csv"
    assert target.exists(), f"retained profiler fixture unavailable: {target}"
    return tuple(kernel_rows(target))

def _kind(rows, kind):
    return [row for row in rows if row.kind == kind]

def _one(rows, kind):
    return next(row for row in rows if row.kind == kind)

def _contract(row):
    from mindie_profiler.tensor_contracts import observed_contract
    return observed_contract(row)

def _norm_op(**changes):
    base = vector_op(op_id="layer0.norm", deps=("input",), stage=0, layer=0, requests=(),
                     elements=256, ops_per_element=6, dtype="bf16", family="rmsnorm",
                     phase="prefill", recipe="rmsnorm", shape=(8, 32))
    return replace(base, **changes)

def _moe_case():
    raw = case(dp=2, ep=2)
    raw["model"].update(ffn="moe", num_experts=4, top_k=2, moe_intermediate_size=32)
    return raw

def _msd_block(*, layers=2):
    """The declared MoE block repeated as one MSD source program per layer."""
    parsed_case = parsed("glm53-a3-w4a8")
    block = next(plan for plan in parsed_case.model.layer_plans if plan.ffn == "moe" and plan.owns_topk)
    model = replace(parsed_case.model, layers=layers, body_layers=layers, layer_plans=tuple(
        replace(block, index=index, kv_owner_layer_id=index, topk_owner_layer_id=index,
                index_k_owner_layer_id=index,
                candidate_source_layer_id=index if block.candidate_source_layer_id >= 0 else -1)
        for index in range(layers)))
    routing = {"realizations": [{
        "scope": "body", "layer": index, "phase": "decode", "tokens": 1,
        "expert_rows": [1] * block.top_k + [0] * (block.num_experts - block.top_k),
    } for index in range(layers)]}
    machine = replace(parsed_case.machine, dies=tuple(replace(
        die, compute_peaks=(*die.compute_peaks, ("int4", 1_000_000_000_000.0)))
        for die in parsed_case.machine.dies))
    runtime = replace(parsed_case.runtime, pp_layer_partition=(layers,), routing=routing, kernels={
        **parsed_case.runtime.kernels, "lowerings": {
            **parsed_case.runtime.kernels.get("lowerings", {}),
            "native_fusion": {"implementation": "declared_native_fusion", "options": {
                "roles": [ROLE], "traffic_profile": "preserve_workspaces",
                "program_profiles": {ROLE: "ascend_w4a8_msd"}}}}})
    return replace(parsed_case, model=model, runtime=runtime, machine=machine)

def test_index_preserves_peer_bound_work_and_detects_projected_record_changes():
    raw = _moe_case()
    args = dict(phase="decode", tokens=1, context=8, scope="body", sequences=[1, 3])
    full, index = export_structure(raw, **args), export_structure_index(raw, **args)
    assert (index["graph_id"], index["execution"]) == (full["graph_id"], full["execution"])
    assert all(index[key] == full[key] for key in index["record_counts"])
    assert index_digest(index) == index["index_id"]
    changed = deepcopy(index)
    changed["nodes"][0]["physical_work"]["hbm_bytes"] += 1
    assert index_digest(changed) != index["index_id"]
    assert export_structure_index(raw, **{**args, "sequences": [1, 5]})["graph_id"] != index["graph_id"]

def test_source_plan_is_omitted_but_native_contract_and_physical_identity_remain():
    raw = mla_case()
    raw["runtime"]["kernels"]["lowerings"] = {"mla_decode": {
        "implementation": "vllm_triton_mla_decode_tuple", "options": {"num_kv_splits": 2}}}
    original = deepcopy(raw)
    args = dict(phase="decode", tokens=1, context=33, scope="body", sequences=2)
    full, index = export_structure(raw, **args), export_structure_index(raw, **args)
    assert index["graph_id"] == full["graph_id"]
    assert index["source_calls"] and index["record_counts"]["source_calls"] == len(full["source_calls"])
    for expected, actual in zip(full["nodes"], index["nodes"], strict=True):
        assert {field: actual[field] for field in INDEX_FIELDS} == {
            field: expected[field] for field in INDEX_FIELDS}
    assert raw == original

def test_mc2_execution_tree_is_removed_after_full_peer_identity_is_computed():
    raw = mc2_case()
    args = dict(phase="decode", tokens=1, context=128, scope="body", sequences=[1, 3])
    full, index = export_structure(raw, **args), export_structure_index(raw, **args)
    assert index["graph_id"] == full["graph_id"]
    plans = [node for node in full["nodes"] if "program_execution" in node["operator"]["metadata"]]
    assert plans
    indexed = {node["node_id"]: node for node in index["nodes"]}
    for node in plans:
        expected = deepcopy(node)
        del expected["operator"]["metadata"]["program_execution"]
        assert indexed[node["node_id"]] == expected
    assert index["execution"]["replicas"] == [0, 1]

@pytest.mark.requires_profiler
@pytest.mark.parametrize("name,capture,tokens,expected", DENSE_CASES)
def test_real_quantized_dense_operators_match_independently_declared_cases(name, capture, tokens, expected, profiler_api):
    declared = [predicted_contract(op.to_dict()) for op in operators(name, tokens)]
    contracts = [_contract(row) for row in _kind(_rows("fusion_contracts", capture), "QuantBatchMatmulV3")]
    assert tuple(declared.count(item) for item in contracts) == expected
    assert all(item is not None for item in contracts)

@pytest.mark.requires_profiler
@pytest.mark.parametrize("name,capture,tokens,count", GROUPED_DOWN)
def test_real_grouped_down_preserves_bank_and_unknown_routing_histogram(name, capture, tokens, count, profiler_api):
    actual = _contract(_one(_rows("fusion_contracts", capture), "GroupedMatmul"))
    matches = [op for op in operators(name, tokens) if predicted_contract(op.to_dict()) == actual]
    assert len(matches) == count and actual[0] == "grouped_matrix"
    op = matches[0].to_dict()
    assert "per_expert_row_histogram" in comparison_scope(op)["not_compared"]
    assert (op["metadata"]["matrix_weight_bank_groups"], op["metadata"]["matrix_groups"]) == (256, 1)
    # A bank and a dense tensor can share total M,N,K and precision; their work
    # assignment still differs and must not share a contract.
    dense = deepcopy(op)
    dense["family"] = "gemm"
    dense["metadata"].pop("matrix_mapping_domain")
    assert predicted_contract(dense)[0] == "matrix" and predicted_contract(dense) != actual
    missing_bank = deepcopy(op)
    missing_bank["metadata"].pop("matrix_weight_bank_groups")
    fractional = deepcopy(op)
    fractional["metadata"]["expected_rows"] = op["local_shape"][0] - 0.25
    assert predicted_contract(missing_bank) is None and predicted_contract(fractional) is None

def test_physical_batch_count_cannot_reuse_flat_dense_contract():
    op = next(op for op in operators("glm53-a3-w4a8", 4096) if op.recipe == "index_q")
    assert predicted_contract(op.to_dict())[0] == "matrix"
    batched = replace(op, metadata=tuple(
        (key, 2 if key == "matrix_groups" else value) for key, value in op.metadata))
    assert predicted_contract(batched.to_dict()) is None
    assert predicted_contract(replace(op, source_boundary="call").to_dict()) is None

@pytest.mark.requires_profiler
def test_mxfp4_capture_does_not_override_declared_bf16_compute(profiler_api):
    ops = operators("kimi-k3-a5", 8192)
    declared = [predicted_contract(op.to_dict()) for op in ops]
    assert all(_contract(row) is not None and _contract(row) not in declared and _contract(row)[2:4] == ("fp8_e4m3", "mxfp4") for row in _kind(_rows("fusion_contracts", "a5_kimi_k3"), "QuantBatchMatmulV4"))
    assert any(item and item[0] == "grouped_matrix" and item[2:4] == ("fp8_e4m3", "mxfp4") for item in declared)
    assert all(dict(op.metadata).get("activation_format") == "bf16" for op in ops if dict(op.metadata).get("quant_category") in {"mla.q_a", "shared_expert"})

def test_matrix_resource_scope_keeps_w4_correction_separate_from_w8_and_mx():
    glm = operators("glm53-a3-w4a8", 4096)
    w4 = next(op for op in glm if op.recipe == "grouped_down")
    assert w4.vector_ops == 0
    assert comparison_scope(w4.to_dict())["prediction_coverage"] == {
        "kind": "partial_program", "reason": "quantized_epilogue_unmodeled",
        "missing_work": ["packed_int4_output_rescale", "quantization_offset_correction"],
        "basis": "declared_int4_int8_program_has_no_supported_correction_epilogue"}
    w8 = next(op for op in glm if op.recipe == "index_q")
    assert dict(w8.metadata)["matrix_epilogue"]["operations_per_output"] > 0 and w8.vector_ops > 0
    assert "prediction_coverage" not in comparison_scope(w8.to_dict())
    mx = next(op for op in operators("qwen38-a5", 16384) if op.recipe == "dense_gate_up")
    assert dict(mx.metadata)["activation_scale_bytes"] > 0
    scope = comparison_scope(mx.to_dict())
    assert "prediction_coverage" not in scope and "not separately resolved" in scope[
        "mx_execution_assumption"]

@pytest.mark.requires_profiler
def test_real_matrix_candidate_does_not_select_layer_from_duration(profiler_api):
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.structure_mapping import map_structure
    graph = export_structure(case_json("glm53-a3-w4a8"), phase="prefill", tokens=4096)
    row = _one(_rows("fusion_contracts", "a3_glm53_prefill"), "QuantBatchMatmulV3")
    event = replace(row.event, rank_id="rank_0", rank_index=0)
    path = Path(row.source)
    stat = path.stat()
    source = SourceRef("rank_0", 0, "csv", str(path), path.name, None,
                       stat.st_size, stat.st_mtime_ns, "retained_capture", "kernel_details")

    def mapping(task):
        observation = Observation("rank_0", 0, Window(task.start_ns, task.end_ns), None, [task],
                                  source, QueryStats("rank_0", "matrix_control"))
        return map_structure([observation], graph)

    result = mapping(event)
    mapped = result["event_mappings"][0]
    assert (mapped["status"], mapped["candidate_count"]) == ("ambiguous", 21) and not mapped["node_ids"]
    assert result["windows"][0]["bound_busy_union_ns"] == 0
    assert mapping(replace(event, end_ns=event.end_ns + 100000))["event_mappings"][0][
        "candidate_node_ids"] == mapped["candidate_node_ids"]

def test_exported_structure_interns_repeated_strings_and_keeps_payload():
    source = {"kind": "case_json", "path": "examples/cases/qwen35-a3-tp4.json"}
    report = export_structure(parse_case(case_json("qwen35-a3-tp4")), phase="prefill",
                              tokens=8, scope="body", source=source)
    nodes = report["nodes"]
    phases = {id(node["phase"]) for node in nodes}
    assert phases and len(phases) == 1 and id(report["execution"]["phase"]) in phases
    decoded = json.loads(json.dumps(report, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
    assert decoded == report and decoded["graph_id"] == report["graph_id"] and decoded["source"] == source
    assert [node["node_id"] for node in decoded["nodes"]] == [node["node_id"] for node in nodes]
    assert [group["subgraph_id"] for group in decoded["subgraphs"]] == [
        group["subgraph_id"] for group in report["subgraphs"]]

@pytest.mark.requires_profiler
@pytest.mark.parametrize("name,capture,tokens", GATED_NORMS)
def test_real_gated_norm_abi_matches_independent_model_frontends(name, capture, tokens, profiler_api):
    actual = _contract(_one(_rows("body_fused", capture), "_layer_norm_fwd_1pass_kernel_npu"))
    gates = [op for op in operators(name, tokens) if op.recipe == "linear_output_norm_gate"]
    assert len(gates) > 1 and len({op.op_id for op in gates}) == len(gates)
    assert all(predicted_contract(op.to_dict()) == actual for op in gates)
    assert all(dict(op.metadata)["fused_from"] == ("rmsnorm", "gate")
               and op.source_boundary == "" for op in gates)
    assert "norm_order_not_observed" in comparison_scope(gates[0].to_dict())
    # A captured dtype is evidence; alternate precision cannot inherit the match.
    assert predicted_contract(replace(gates[0], dtype="fp16").to_dict()) != actual

@pytest.mark.requires_profiler
@pytest.mark.parametrize("name,capture", GEMMA_NORMS)
def test_gemma_weight_offset_is_declared_before_matching_real_native_norm(name, capture, profiler_api):
    from mindie_sim.prediction.transforms import residual_norm
    rows = _rows("body_fused", capture)
    actual, actual_add = _contract(_one(rows, "GemmaRmsNorm")), _contract(_one(rows, "AddRmsNormBias"))
    declared = operators(name, 8192)
    norms = [op for op in declared if op.family == "rmsnorm"
             and dict(op.metadata).get("norm_weight_offset") == 1.0]
    assert norms and all(predicted_contract(op.to_dict()) == actual for op in norms)
    ordinary = replace(norms[0], metadata=tuple(
        (key, 0.0 if key == "norm_weight_offset" else value) for key, value in norms[0].metadata))
    assert predicted_contract(ordinary.to_dict()) != actual
    # A controlled opt-in fusion must keep Gemma's shift and not invent AddBias.
    fused = [op for op in residual_norm(declared, None) if op.family == "add_rmsnorm"]
    assert fused and all(predicted_contract(op.to_dict()) is None for op in fused)
    assert actual_add[1] == "AddRmsNormBias"
    assert all(predicted_contract(op.to_dict()) != actual_add for op in fused)

@pytest.mark.requires_profiler
def test_real_kda_norm_matches_declared_chunk_without_filling_missing_rows(profiler_api):
    actual, missing = _rows("body_fused", "kimi_a5_norm_gate.csv")
    declared = operators("kimi-k3-a5", 8192)
    norms = [op for op in declared if op.recipe == "linear_output_norm_gate"]
    assert norms and all(dict(op.metadata)["recurrent_attention_kind"] == "kda" for op in norms)
    assert all(predicted_contract(op.to_dict()) == _contract(actual) for op in norms)
    assert _contract(missing) is None
    assert all("auxiliary_stats_not_compared" in comparison_scope(op.to_dict()) for op in norms)
    # The retained dense FIA keeps unabsorbed Q/K/V; latent absorption cannot
    # claim that different boundary.
    assert all(predicted_contract(op.to_dict()) is None for op in declared if op.family == "attention")

def test_prefill_attention_projection_keeps_kv_heads_and_rejects_paged_assumptions():
    parsed_case = parsed("qwen38-a5")
    attention = next(op for op in target_operators(parsed_case, phase="prefill", tokens=16384) if op.family == "attention")
    contract = predicted_contract(attention.to_dict())
    assert contract[0] == "attention_active_qkv"
    assert contract[2] == (((16384, 24, 256), "bf16", "ND"), ((16384, 4, 256), "bf16", "ND"),
                           ((16384, 4, 256), "bf16", "ND"))
    assert contract[3] == (((16384, 24, 256), "bf16", "ND"),)
    others = (*target_operators(parsed_case, phase="prefill", tokens=64, context=1024),
              *target_operators(parsed_case, phase="decode", tokens=1, context=1024))
    assert all(predicted_contract(op.to_dict()) is None for op in others if op.family == "attention")

@pytest.mark.requires_profiler
@pytest.mark.parametrize("name,capture,tokens", CACHE_PROJECTIONS)
def test_real_attention_cache_projections_match_independent_model_frontends(name, capture, tokens, profiler_api):
    rows = _rows("attention_cache_contracts", capture)
    actual_attention = _contract(_one(rows, "FusedInferAttentionScore"))
    actual_cache = _contract(_one(rows, "ScatterPaKvCache"))
    declared = operators(name, tokens)
    attention = [op for op in declared if op.family == "attention"]
    stores = [op for op in declared if op.family == "cache_store"
              and dict(op.metadata).get("cache_kind") == "gqa"]
    assert len(attention) == len(stores) > 1
    assert all(predicted_contract(op.to_dict()) == actual_attention for op in attention)
    assert all(predicted_contract(op.to_dict()) == actual_cache for op in stores)
    assert len({op.layer for op in stores}) == len(stores)
    assert all("native_storage_layout_cache_capacity_slot_dtype_values" in
               comparison_scope(op.to_dict()) for op in stores)

@pytest.mark.requires_profiler
def test_real_cache_capacity_and_paged_attention_preserve_comparison_boundary(profiler_api):
    rows = _rows("attention_cache_contracts", "a5_qwen38.csv")
    caches, attention = _kind(rows, "ScatterPaKvCache"), _kind(rows, "FusedInferAttentionScore")
    assert len(caches) == len(attention) == 2
    assert caches[0].input_shapes[1] != caches[1].input_shapes[1]
    assert _contract(caches[0]) == _contract(caches[1])
    assert _contract(attention[0])[0] == "attention_active_qkv"
    assert _contract(attention[1])[0] == "attention_paged_active_qkv"
    assert _contract(attention[0]) != _contract(attention[1])
    paged = attention[1]
    changed = replace(paged, input_shapes=(
        paged.input_shapes[0], (1000, *paged.input_shapes[1][1:]),
        (1000, *paged.input_shapes[2][1:]), *paged.input_shapes[3:]))
    assert _contract(changed) == _contract(paged)

@pytest.mark.requires_profiler
def test_model_identity_needs_attention_or_norm_beyond_shared_cache_projection(profiler_api):
    q35 = _rows("attention_cache_contracts", "a3_qwen35.csv")
    q36 = _rows("attention_cache_contracts", "a3_qwen36.csv")
    cache = lambda rows: _contract(_one(rows, "ScatterPaKvCache"))
    attention = lambda rows: _contract(_one(rows, "FusedInferAttentionScore"))
    assert cache(q35) == cache(q36) and attention(q35) != attention(q36)
    declared = target_operators(parsed("qwen35-a3-tp4"), phase="prefill", tokens=8192)
    assert all(predicted_contract(op.to_dict()) != attention(q36)
               for op in declared if op.family == "attention")

def test_cache_declaration_keeps_active_tokens_separate_from_prefix_capacity():
    parsed_case = parsed("qwen38-a5")
    plan = ((target_operators(parsed_case, phase="prefill", tokens=64, context=1024), 64),
            (target_operators(parsed_case, phase="decode", tokens=1, context=1024), 1))
    for declared, tokens in plan:
        stores = [op for op in declared if op.family == "cache_store"
                  and dict(op.metadata).get("cache_kind") == "gqa"]
        assert stores
        assert all(predicted_contract(op.to_dict()) == (
            "kv_cache_store_active_elements", "ScatterPaKvCache",
            ((tokens * 4 * 256, "bf16"), (tokens * 4 * 256, "bf16")), (tokens,),
        ) for op in stores)

@pytest.mark.requires_profiler
def test_real_sparse_attention_matches_declared_latent_and_rope_geometry(profiler_api):
    actual = _contract(_one(_rows("attention_cache_contracts", "a3_glm53.csv"), "SparseFlashAttention"))
    declared = target_operators(parsed("glm53-a3-w4a8"), phase="prefill", tokens=4096)
    attention = [op for op in declared if op.family == "attention"]
    assert len(attention) == 78
    assert all(predicted_contract(op.to_dict()) == actual for op in attention)
    assert all("cache_capacity_page_size_sequence_and_index_tensors" in
               comparison_scope(op.to_dict()) for op in attention)
    # Score and top-k stay separate in the DAG; neither claims the fused ABI.
    assert all(predicted_contract(op.to_dict()) is None
               for op in declared if op.family in {"indexer", "topk"})

@pytest.mark.requires_profiler
def test_real_shared_kv_attention_distinguishes_compressed_cache_presence(profiler_api):
    actual = _contract(_one(_rows("attention_cache_contracts", "a3_deepseek_v4f.csv"),
                            "SparseAttnSharedkv"))
    parsed_case = parsed("deepseek-v4-flash-0731")
    declared = target_operators(parsed_case, phase="prefill", tokens=4096)
    attention = [op for op in declared if op.family == "attention"]
    uncompressed = [op for op in attention if parsed_case.model.layer_plans[op.layer].compress_ratio == 0]
    compressed = [op for op in attention if parsed_case.model.layer_plans[op.layer].compress_ratio > 0]
    assert len(uncompressed) == 2 and len(compressed) == 41
    assert all(predicted_contract(op.to_dict()) == actual for op in uncompressed)
    assert all(predicted_contract(op.to_dict()) != actual for op in compressed)
    compressed_rows = _rows("attention_cache_contracts", "a3_deepseek_v4f_compressed.csv")
    assert len(compressed_rows) == 2
    for row in compressed_rows:
        assert all(predicted_contract(op.to_dict()) == _contract(row) for op in compressed)
        assert all(predicted_contract(op.to_dict()) != _contract(row) for op in uncompressed)
    glm = target_operators(parsed("glm53-a3-w4a8"), phase="prefill", tokens=4096)
    assert all(predicted_contract(op.to_dict()) != actual for op in glm if op.family == "attention")

def test_source_attention_call_keeps_children_instead_of_becoming_one_kernel_contract():
    raw = mla_case()
    unexpanded = export_structure(raw, phase="decode", tokens=1, context=16, expand_calls=False)
    expanded = export_structure(raw, phase="decode", tokens=1, context=16)
    parent_id = "rank:0/s0.l0.latent_attention_decode"
    parent = next(node for node in unexpanded["nodes"] if node["node_id"] == parent_id)
    assert parent["contract"] is None
    assert parent["contract_scope"] == "source_call; expand_to_kernel_boundaries_before_tensor_comparison"
    call = next(row for row in expanded["source_calls"] if row["node_id"] == parent_id)
    children = [node for node in expanded["nodes"] if node["node_id"] in call["children"]]
    assert [node["operator"]["metadata"]["kernel_phase"] for node in children] == ["stage1", "stage2"]
    assert set(children[1]["deps"]) == {children[0]["node_id"], *parent["deps"]}
    assert all(node["lineage"]["source_call_id"] == parent_id for node in children)

def test_occurrence_annotations_do_not_split_resource_identical_layers():
    first = _norm_op(deps=("input0",), requests=("r0",), metadata=(
        ("norm_has_weight", True), ("structure_block", "attention"),
        ("structure_subgraph_memberships", ("layer0.prolog",))))
    repeated = replace(first, op_id="layer1.norm", deps=("input1",), layer=1, requests=("r1",),
                       metadata=tuple({**dict(first.metadata),
                                       "structure_subgraph_memberships": ("layer1.prolog",)}.items()))
    one, two = node_physical(first), node_physical(repeated)
    assert one == two and (one["physical_work"]["launches"],
                           one["physical_work"]["vector_ops"]) == (1, 256 * 6)
    changed = replace(repeated, mte2_bytes=repeated.mte2_bytes + 128, hbm_bytes=repeated.hbm_bytes + 128)
    assert node_physical(changed)["physical_signature"] != one["physical_signature"]
    manifest = export_op_structure([first, repeated], model={"name": "declared-repeated-norm"}, execution={"phase": "prefill"})
    assert len({node["physical_signature"] for node in manifest["nodes"]}) == 1
    assert len({node["node_id"] for node in manifest["nodes"]}) == 2

def test_capture_segment_identity_does_not_change_work_but_capture_mode_does():
    first = _norm_op(metadata=(("graph_captured", True), ("graph_segment", 0)))
    repeated = replace(first, op_id="layer1.norm", layer=1,
                       metadata=tuple({**dict(first.metadata), "graph_segment": 1}.items()))
    assert node_physical(first) == node_physical(repeated)
    eager = replace(repeated, metadata=tuple({**dict(repeated.metadata), "graph_captured": False}.items()))
    assert node_physical(eager)["physical_signature"] != node_physical(first)["physical_signature"]

def test_routing_producer_occurrences_preserve_aliases_without_splitting_layers():
    graphs = target_graphs(parsed("qwen35-a3-tp4"), phase="prefill", tokens=8192, scope="body")
    for recipe in ("route_pack", "route_unpack"):
        ops = [op for op in graphs[0] if op.recipe == recipe]
        assert len(ops) == 40 and len({node_physical(op)["physical_signature"] for op in ops}) == 1
        original = ops[0]
        meta = dict(original.metadata)
        if recipe == "route_pack":
            meta["topk_ids_producer"] = meta["payload_producer"]
        else:
            meta["topk_weights_producer"] = None
        changed = replace(original, metadata=tuple(meta.items()))
        assert (node_physical(changed)["physical_signature"]
                != node_physical(original)["physical_signature"])

def test_semantic_recipe_labels_share_fallback_but_explicit_invocations_do_not():
    first = _norm_op(op_id="norm_a", deps=(), requests=(), recipe="attention_norm")
    second = replace(first, op_id="norm_b", layer=1, recipe="ffn_norm")
    assert node_physical(first) == node_physical(second)
    baseline = operator_costs(case(), [first, second])["operators"]
    assert baseline[0]["duration_ms"] == baseline[1]["duration_ms"]
    declared = case()
    declared["runtime"]["kernels"] = {"invocations": [{
        "match": {"recipe": "attention_norm"}, "startup_ms": 1}]}
    bindings = declared["runtime"]["kernels"]
    assert node_physical(first, recipe_bindings=bindings) != node_physical(second, recipe_bindings=bindings)
    changed = operator_costs(declared, [first, second])["operators"]
    assert changed[0]["duration_ms"] == pytest.approx(
        changed[1]["duration_ms"] + 1, rel=1e-12, abs=1e-12)

def test_family_service_identity_still_distinguishes_kernel_recipe_names():
    parsed_case = parse_case(hardware_case())
    first = copy_op()
    second = replace(first, recipe="other_copy")
    bindings = {"families": {"copy": {"kind": "resource_service", "model": curve().to_dict()}}}
    assert service_identity(parsed_case, first) != service_identity(parsed_case, second)
    assert node_physical(first, recipe_bindings=bindings) != node_physical(second, recipe_bindings=bindings)

def test_catalog_matrix_roles_share_only_an_unbound_selected_program():
    first = next(op for op in target_graphs(case(), phase="prefill", tokens=4, scope="body")[0]
                 if dict(op.metadata).get("program_kind") == "matmul")
    second = replace(
        first, op_id="other_projection", family="gate_up_projection", recipe="dense_gate_up",
        metadata=tuple({**dict(first.metadata), "semantic_family": "gate_up_projection",
                        "quant_category": "shared_expert", "catalog_category": "shared_expert"}.items()))
    assert node_physical(first) == node_physical(second)
    # An explicit family binding is what distinguishes the two roles.
    explicit = {"families": {"gate_up_projection": {"kind": "tiled_matmul"}}}
    book = RecipeBook.from_mapping(explicit)
    assert type(book.select(first)) is not type(book.select(second))
    assert (node_physical(first, recipe_bindings=explicit)
            != node_physical(second, recipe_bindings=explicit))
    invocation = {"invocations": [{"match": {"family": "gate_up_projection"}, "startup_ms": 1}]}
    assert (node_physical(first, recipe_bindings=invocation)
            != node_physical(second, recipe_bindings=invocation))

def test_msd_member_ids_are_local_but_recipe_shape_and_order_remain_physical():
    parsed_case = _msd_block(layers=2)
    ops = [op for op in target_graphs(parsed_case, phase="decode", tokens=1, scope="body")[0]
           if "fusion_semantic_recipes" in dict(op.metadata)]
    assert len(ops) == 2
    first, second = ops
    bindings = parsed_case.runtime.kernels
    original = node_physical(first, recipe_bindings=bindings)
    assert node_physical(second, recipe_bindings=bindings) == original
    assert [entry["op_id"] for entry in dict(first.metadata)["fusion_semantic_recipes"]] != [
        entry["op_id"] for entry in dict(second.metadata)["fusion_semantic_recipes"]]
    for changed_field, value in (("recipe", "distinct_explicit_member_recipe"),
                                 ("local_shape", [1, 2, 3]),
                                 ("family", "distinct_physical_member_family")):
        metadata = deepcopy(dict(second.metadata))
        metadata["fusion_semantic_recipes"][0][changed_field] = value
        changed = replace(second, metadata=tuple(metadata.items()))
        assert node_physical(changed, recipe_bindings=bindings) != original
    reordered = deepcopy(dict(second.metadata))
    reordered["fusion_semantic_recipes"].reverse()
    assert node_physical(replace(second, metadata=tuple(reordered.items())),
                         recipe_bindings=bindings) != original
    # Unknown nested IDs are deliberately outside this narrow normalization.
    left = replace(first, metadata=(*first.metadata, ("custom_source", {"op_id": "left"})))
    right = replace(second, metadata=(*second.metadata, ("custom_source", {"op_id": "right"})))
    assert node_physical(left, recipe_bindings=bindings) != node_physical(right, recipe_bindings=bindings)

def test_mx_dispatch_producer_ids_preserve_scale_payload_aliasing():
    path = next((ROOT / "examples/cases/high-concurrency-20261030"
                 / "v41-composed-arch35-conditional").glob("*c32*"))
    ops = target_graphs(json.loads(path.read_text()), phase="decode", tokens=1, context=2048,
                        sequences=[4] * 8, scope="body")[0]
    up = [op for op in ops if op.op_id.endswith(".expert_up")]
    assert len(up) == 40 and len({node_physical(op)["physical_signature"] for op in up}) == 1
    meta = dict(up[0].metadata)
    assert meta["activation_scale_producer"] == meta["activation_payload_producer"]
    meta["activation_scale_producer"] = "independent_scale_kernel"
    other = replace(up[0], metadata=tuple(meta.items()))
    assert (node_physical(other)["physical_signature"]
            != node_physical(up[0])["physical_signature"])

def _declaration(family, *, output, scale_dtype, scale_shape):
    return {"family": family, "recipe": family, "dtype": "bf16", "local_shape": (1024, 3584),
            "metadata": {"input_format": "bf16", "output_format": output,
                         "activation_scale_shape": scale_shape,
                         "activation_scale_format": scale_dtype, "norm_has_weight": True}}

def test_dynamic_int8_and_mx_have_different_scale_domains():
    int8 = _declaration("quantize", output="int8", scale_dtype="fp32", scale_shape=(1024,))
    mx = _declaration("quantize", output="fp8_e4m3", scale_dtype="fp8_e8m0", scale_shape=(1024, 56, 2))
    assert predicted_contract(int8) == (
        "dynamic_quant_active", "DynamicQuant", ((1024, 3584), "bf16", "ND"),
        ((1024, 3584), "int8", "ND"), ((1024,), "fp32", "ND"))
    assert predicted_contract(mx) == (
        "DynamicMxQuant", (1024, 3584), "bf16", "fp8_e4m3", (1024, 56, 2), "fp8_e8m0")
    int8["metadata"]["activation_scale_shape"] = (1024, 56, 2)
    assert predicted_contract(int8) is None

@pytest.mark.requires_profiler
def test_real_norm_mx_contract_requires_both_norm_and_quantization(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    root = Path(mindie_profiler.__file__).resolve().parents[1] / "tests" / "fixtures" / "fusion_contracts"
    row = next(row for path in sorted(root.glob("*.csv")) for row in kernel_rows(path)
               if row.kind == "RmsNormDynamicMxQuant")
    op = _declaration("rmsnorm_quant", output="fp8_e4m3", scale_dtype="fp8_e8m0",
                      scale_shape=(1024, 56, 2))
    assert predicted_contract(op) == _contract(row)
    op["metadata"]["norm_weight_offset"] = 1.0
    assert predicted_contract(op) is None

@pytest.mark.parametrize("sequences", ([1, 3], [2, 2]))
def test_full_identity_is_preserved_without_retaining_unselected_replicas(sequences):
    raw = _moe_case()
    args = dict(phase="decode", tokens=1, context=8, scope="body", sequences=sequences)
    full = export_structure(raw, **args)
    all_targets = {row["node_id"]: row for row in full["nodes"]}
    all_targets.update((row["subgraph_id"], row) for row in full["subgraphs"])
    wanted = {key for key in all_targets if key.startswith("rank:0/")} | {"rank:1/not-a-node"}
    reduced = select_structure_targets(raw, wanted, **args)
    assert (reduced["graph_id"], reduced["execution"]) == (full["graph_id"], full["execution"])
    assert reduced["targets"] == {key: value for key, value in all_targets.items() if key in wanted}
    assert reduced["record_counts"] == {key: len(full[key]) for key in reduced["record_counts"]}
    # Work on an unselected DP participant still changes full identity.
    altered = select_structure_targets(raw, wanted, **{**args, "sequences": [sequences[0], 5]})
    assert altered["graph_id"] != reduced["graph_id"]

def test_source_native_fusion_and_calls_preserve_full_hash_and_boundaries():
    raw = mla_case()
    raw["runtime"]["kernels"]["lowerings"] = {"mla_decode": {
        "implementation": "vllm_triton_mla_decode_tuple", "options": {"num_kv_splits": 2}}}
    original = deepcopy(raw)
    args = dict(phase="decode", tokens=1, context=33, scope="body", sequences=2)
    full = export_structure(raw, **args)
    wanted = {row["node_id"] for row in full["nodes"]} | {row["subgraph_id"] for row in full["subgraphs"]}
    selected = select_structure_targets(raw, wanted, **args)
    assert selected["graph_id"] == full["graph_id"]
    assert selected["record_counts"]["source_calls"] == len(full["source_calls"]) > 0
    assert selected["record_counts"]["resource_operators"] == len(full["resource_operators"])
    assert raw == original
