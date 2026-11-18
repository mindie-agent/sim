"""Source-defined routing work and independent retained selection contracts."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.structure import export_structure
from test_prediction_system import case as small_case


ROOT = Path(__file__).resolve().parents[1]


def catalog(name):
    return parse_case(json.loads((ROOT / "examples/cases" / f"{name}.json").read_text()))


def router_case(mode):
    raw = small_case()
    raw["model"].update(layers=2, ffn="moe", num_experts=4, top_k=2,
                        moe_intermediate_size=16)
    case = parse_case(raw)
    return replace(case, model=replace(case.model, layer_plans=tuple(
        replace(plan, router_mode=mode, router_dtype="fp32", router_norm_topk=True)
        for plan in case.model.layer_plans)))


@pytest.mark.parametrize("mode,native", [
    # Only a declared hash table executes the hash gating kernel.  Every
    # score-based function (sigmoid / softmax / sqrtsoftplus) executes the
    # score gating kernel and may own its renormalization in-kernel.
    ("hash", "MoeGatingTopKHash"), ("sqrtsoftplus", "MoeGatingTopK"),
    ("sigmoid", "MoeGatingTopK"),
])
def test_selection_boundary_retains_logits_gemm_and_source_normalization(mode, native):
    case = router_case(mode)
    graph = export_structure(case, phase="prefill", tokens=4, scope="body")
    nodes = {node["node_id"]: node for node in graph["nodes"]}
    groups = [group for group in graph["subgraphs"] if group["role"] == "router_selection"]
    assert len(groups) == 2
    for group in groups:
        members = [nodes[key] for key in group["node_ids"]]
        recipes = {node["recipe"] for node in members}
        assert group["contract"][1] == native
        assert all(node["family"] not in {"router", "cast"} for node in members)
        assert ("route_normalize" in recipes) == (native == "MoeGatingTopK")
        assert {nodes[key]["family"] for key in group["external_inputs"]} == {"router"}
        parameters = group["implementation"]["declared_parameters"]
        assert parameters["renorm"] == (native == "MoeGatingTopK")
        assert parameters["external_renorm"] == (native == "MoeGatingTopKHash")
    ops = target_operators(case, phase="prefill", tokens=4, scope="body")
    router = next(op for op in ops if op.family == "router")
    cast = next(op for op in ops if op.recipe == "router_input_cast")
    assert router.dtype == "fp32" and router.deps == (cast.op_id,)
    assert router.flops == 2 * 4 * 4 * 32
    assert dict(router.metadata)["weight_format"] == "fp32"
    assert dict(router.metadata)["packed_weight_bytes"] == 4 * 32 * 4
    assert cast.mte2_bytes == 4 * 32 * 2
    assert cast.mte3_bytes == 4 * 32 * 4


@pytest.mark.parametrize("name,dtype,multiplier", [
    # A retained explicit FP32 declaration keeps its declared operand copy.
    ("v4f-a3-w8a8-tp8", "fp32", 3),
    # ``AscendGateLinear`` Tier 4: BF16-resident weights, FP32 accumulation
    # output, zero precast copy and zero input cast.
    ("glm53-a3-w4a8", "", 1),
])
def test_router_runtime_precision_preserves_checkpoint_storage(name, dtype, multiplier):
    case = catalog(name)
    plan = next(plan for plan in case.model.layer_plans if plan.ffn == "moe")
    assert plan.router_dtype == dtype
    baseline = replace(plan, router_dtype="", router_precast_weight_copy=False)
    old = {term.name: term for term in layer_weight_terms(baseline, case.model)}
    new = {term.name: term for term in layer_weight_terms(plan, case.model)}
    quant, parallel = case.model.quant, case.runtime.parallel
    source = term_bytes(new["router"], quant, parallel, source=True)
    assert source == term_bytes(old["router"], quant, parallel, source=True)
    actual = sum(term_bytes(term, quant, parallel) for key, term in new.items()
                 if key in {"router", "router_precast"})
    assert actual == source * multiplier
    if dtype == "fp32":
        assert plan.router_precast_weight_copy is True
        assert term_bytes(new["router_precast"], quant, parallel, source=True) == 0
    else:
        assert plan.router_output_dtype == "fp32"
        assert plan.router_precast_weight_copy is False
        assert "router_precast" not in new


@pytest.mark.parametrize("name,capture,kind,tokens", [
    ("glm53-a3-w4a8", "a3_glm53_prefill", "MoeGatingTopK", 4096),
    ("qwen35-a3-tp4", "a3_qwen35", "MoeGatingTopK", 8192),
    ("v4f-a3-w8a8-tp8", "a3_v4f", "MoeGatingTopKHash", 512),
])
def test_real_selection_candidates_preserve_repeated_layers(name, capture, kind, tokens, profiler_api):
    import mindie_profiler
    from mindie_profiler.events import Observation, QueryStats, SourceRef, Window
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.structure_mapping import map_structure
    from mindie_profiler.tensor_contracts import observed_contract

    case = catalog(name)
    graph = export_structure(case, phase="prefill", tokens=tokens, scope="body")
    rel = Path(f"tests/fixtures/fusion_contracts/{capture}.csv")
    path = Path(mindie_profiler.__file__).resolve().parents[1] / rel
    if not path.is_file():
        path = Path(__file__).resolve().parents[2] / "profiler" / rel
    if not path.is_file():
        pytest.skip(f"retained fusion fixture unavailable: {path}")
    rows = [row for row in kernel_rows(path, rank_id="rank_0", rank_index=0)
            if row.kind == kind and row.input_shapes[0][0] == tokens]
    # A capture only contains the declared kind's rows; a sqrtsoftplus layer
    # executes score TopK rather than the hash kernel, so select the matching
    # declared groups instead of requiring every layer to share one kind.
    groups = [group for group in graph["subgraphs"]
              if group["role"] == "router_selection"
              and group["contract"][1] == kind]
    contracts = [json.loads(json.dumps(observed_contract(row))) for row in rows]
    assert groups and rows and all(group["contract"] in contracts for group in groups)
    events = [row.event for row in rows]
    stat = path.stat()
    source = SourceRef(
        "rank_0", 0, "csv", str(path), path.name, None, stat.st_size, stat.st_mtime_ns,
        "retained_capture", "kernel_details")
    observation = Observation(
        "rank_0", 0, Window(min(e.start_ns for e in events), max(e.end_ns for e in events)),
        None, events, source, QueryStats("rank_0", "router_retained_rows"))
    result = map_structure([observation], graph)
    # A captured row maps only when the declared boundary shares its kind and
    # selection mode.  A hash-kind row whose table inputs were not captured is
    # reported unmapped instead of being attributed to a score-based layer.
    declared = {(group["contract"][1], group["contract"][2]) for group in groups}
    unmapped = sum(1 for contract in contracts
                   if (contract[1], contract[2]) not in declared)
    expected = ({"ambiguous": len(events)} if not unmapped else
                {"ambiguous": len(events) - unmapped, "unmapped": unmapped})
    assert result["status_counts"] == expected
    for row in result["event_mappings"]:
        if row.get("status") == "unmapped":
            continue
        assert row["candidate_subgraph_ids"] and not row["subgraph_ids"]
        assert row["structural_classification"]["common"]["role"] == "router_selection"
    assert result["windows"][0]["bound_busy_union_ns"] == 0
