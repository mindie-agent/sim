"""Fusion candidates retain the independently declared DAG and output boundary."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_bytes
from mindie_sim.observations.kernels import kernel_rows
from mindie_sim.prediction.structure_groups import declare_group
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.structure import export_op_structure, export_structure
from test_mla_family_selection import mla_case


ROOT = Path(__file__).resolve().parents[1]


def _raw(name="glm53-a3-w4a8", *, selected=True):
    raw = json.loads((ROOT / "examples/cases" / f"{name}.json").read_text())
    if selected:
        raw["runtime"].setdefault("kernels", {}).setdefault("lowerings", {})[
            "mla_prolog"] = "vllm_ascend_mla_prolog_v3"
    return raw


def _export(raw, *, phase="prefill", tokens=4096, context=0):
    return export_structure(raw, phase=phase, tokens=tokens, context=context, scope="body")


def _fixture(name, profiler_api, group="attention_cache_contracts"):
    import mindie_profiler

    path = (Path(mindie_profiler.__file__).resolve().parents[1] /
            "tests/fixtures" / group / name)
    if not path.is_file():
        pytest.skip(f"retained profiler source fixture is unavailable: {path}")
    return tuple(kernel_rows(path))


def _json(value):
    return json.loads(json.dumps(value))


@pytest.mark.parametrize("phase,tokens,context,files", [
    ("prefill", 4096, 0, ("a3_glm53.csv", "a3_glm53_prolog_no_index.csv")),
    ("decode", 1, 4096, ("a3_glm53_decode.csv",)),
])
def test_real_glm_prolog_and_indexer_match_declared_groups_without_erasing_branches(
    phase, tokens, context, files, profiler_api,
):
    from mindie_profiler.tensor_contracts import observed_contract

    structure = _export(_raw(), phase=phase, tokens=tokens, context=context)
    groups = structure["subgraphs"]
    full = [g for g in groups if g["role"] == "mla_prolog"
            and g["structure"]["indexer"]]
    shared = [g for g in groups if g["role"] == "mla_prolog"
              and not g["structure"]["indexer"]]
    indexers = [g for g in groups if g["role"] == "lightning_indexer"]
    captures = [row for name in files for row in _fixture(name, profiler_api)]
    observed = [observed_contract(row) for row in captures]
    with_query = next(c for c in observed if c and c[0] == "mla_prolog_active" and c[-1][-1][0])
    no_query = next(c for c in observed if c and c[0] == "mla_prolog_active" and not c[-1][-1][0])
    lightning = next(c for c in observed if c and c[0] == "lightning_indexer_active")
    assert full and shared and indexers
    assert all(g["contract"] == _json(with_query) for g in full)
    assert all(g["contract"] == _json(no_query) for g in shared)
    assert all(g["contract"] == _json(lightning) for g in indexers)
    assert {g["layer"] for g in indexers} == {g["layer"] for g in full}
    assert {g["layer"] for g in full}.isdisjoint({g["layer"] for g in shared})


@pytest.mark.parametrize("name", ["glm53-a3-w4a8", "glm53f-a5-true128k"])
def test_indexer_execution_and_shared_attention_dependency_follow_catalog_ownership(name):
    case = parse_case(_raw(name, selected=False))
    ops = {op.op_id: op for op in target_operators(case, phase="decode", tokens=1, context=4096)}
    structure = _export(case, phase="decode", tokens=1, context=4096)
    indexers = {g["layer"]: g for g in structure["subgraphs"]
                if g["role"] == "lightning_indexer"}
    for plan in case.model.layer_plans:
        if plan.attention != "mla":
            continue
        assert plan.index_query_per_layer == plan.indexer
        prefix = f"s0.l{plan.index}"
        attention = ops[prefix + ".latent_attention0"]
        if plan.indexer:
            assert prefix + ".index_topk" in attention.deps
            group = indexers[plan.index]
            assert group["node_ids"] == [
                f"rank:0/{prefix}.index_scores", f"rank:0/{prefix}.index_topk"]
            topk = ops[prefix + ".index_topk"]
            assert topk.dtype == "fp32"
            assert dict(topk.metadata)["selection_output_dtype"] == "int32"
        elif plan.topk_owner_layer_id >= 0:
            owner = f"s0.l{plan.topk_owner_layer_id}.index_topk"
            assert owner in attention.deps and owner in ops
            assert prefix + ".index_q" not in ops
            assert plan.index not in indexers
    assert not any(g["role"] == "mla_prolog" for g in structure["subgraphs"])


def test_prolog_external_boundary_follows_quantized_qnorm_and_rope_consumers():
    report = _export(_raw())
    nodes = {node["node_id"]: node for node in report["nodes"]}
    for group in (g for g in report["subgraphs"] if g["role"] == "mla_prolog"):
        prefix = f"rank:0/s0.l{group['layer']}"
        assert group["external_inputs"] == [prefix + ".q_a.quantize"]
        assert prefix + ".q_a.quantize" not in group["node_ids"]
        assert prefix + ".q_b.quantize" in group["node_ids"]
        outputs = {prefix + suffix for suffix in (".rotary", ".kv_store", ".q_absorb")}
        if group["structure"]["indexer"]:
            outputs.add(prefix + ".q_b.quantize")
            assert nodes[prefix + ".index_q"]["deps"] == [prefix + ".q_b.quantize"]
        assert set(group["external_outputs"]) == outputs
        assert prefix + ".q_norm" not in outputs
        assert prefix + ".rotary" in nodes[prefix + ".latent_attention0"]["deps"]
        assert nodes[prefix + ".q_b.quantize"]["operator"]["metadata"]["output_format"] == "int8"


@pytest.mark.parametrize("passes", [[], ["norm_quant", "residual_norm"]])
def test_selected_quantized_prolog_cannot_claim_conversions_that_are_disabled(passes):
    raw = _raw()
    raw["runtime"]["lowering_passes"] = passes
    with pytest.raises(ModelingError, match="quantized_operands_require_activation_quantization"):
        _export(raw)
    # BF16 projection operands already use the model activation format.
    raw["model"]["quantization"] = "bf16"
    report = _export(raw)
    nodes = {node["node_id"]: node for node in report["nodes"]}
    full = next(g for g in report["subgraphs"]
                if g["role"] == "mla_prolog" and g["structure"]["indexer"])
    assert full["contract"][-1][-1][1] == "bf16"
    assert "rank:0/s0.l0.q_norm" in full["external_outputs"]
    assert nodes["rank:0/s0.l0.index_q"]["deps"] == ["rank:0/s0.l0.q_norm"]


@pytest.mark.parametrize("passes", [
    ["norm_quant", "activation_quantization"],
    ["residual_norm", "activation_quantization", "norm_quant"],
    ["activation_quantization", "norm_quant", "residual_norm"],
])
def test_custom_lowering_order_preserves_actual_quantized_prolog_output(passes):
    raw = _raw()
    raw["runtime"]["lowering_passes"] = passes
    report = _export(raw)
    nodes = {node["node_id"]: node for node in report["nodes"]}
    full = next(g for g in report["subgraphs"]
                if g["role"] == "mla_prolog" and g["structure"]["indexer"])
    producer = nodes["rank:0/s0.l0.q_b.quantize"]
    assert producer["node_id"] in full["node_ids"]
    assert producer["node_id"] in full["external_outputs"]
    assert producer["operator"]["metadata"]["output_format"] == full["contract"][-1][-1][1] == "int8"
    assert nodes["rank:0/s0.l0.index_q"]["deps"] == [producer["node_id"]]


def test_selected_quantized_prolog_rejects_unquantized_indexer_shared_output():
    case = parse_case(_raw())
    quant = replace(case.model.quant, exclusions=(*case.model.quant.exclusions, "indexer_fp8"))
    excluded = replace(case, model=replace(case.model, quant=quant))
    with pytest.raises(ModelingError, match="mla_prolog_v3_query_norm_indexer_activation_mismatch"):
        _export(excluded)
    # Validate the direct frontend entry too: it does not run parse_case.
    from mindie_sim.prediction.frontend import Frontend
    from mindie_sim.schema.workload import RequestView

    views = (RequestView("probe", 32, 0, False, "prefill"),)
    with pytest.raises(ModelingError, match="mla_prolog_v3_query_norm_indexer_activation_mismatch"):
        Frontend().body(excluded.model, excluded.runtime, views, ready="input")
    # When both projections consume BF16, the real shared producer is q_norm.
    quant = replace(quant, exclusions=(*quant.exclusions, "mla"))
    bf16 = replace(excluded, model=replace(excluded.model, quant=quant))
    report = _export(bf16, tokens=32)
    group = next(g for g in report["subgraphs"]
                 if g["role"] == "mla_prolog" and g["structure"]["indexer"])
    nodes = {node["node_id"]: node for node in report["nodes"]}
    assert group["contract"][-1][-1][1] == "bf16"
    assert "rank:0/s0.l0.q_norm" in group["external_outputs"]
    assert nodes["rank:0/s0.l0.index_q"]["deps"] == ["rank:0/s0.l0.q_norm"]


def test_selected_unquantized_absorption_agrees_with_checkpoint_and_capacity_bytes():
    baseline = parse_case(_raw(selected=False))
    selected = parse_case(_raw())
    before = {op.op_id: op for op in target_operators(baseline, phase="prefill", tokens=32)}
    after = {op.op_id: op for op in target_operators(selected, phase="prefill", tokens=32)}
    quant, parallel = selected.model.quant, selected.runtime.parallel
    assert quant.format_for("mla") == baseline.model.quant.format_for("mla") == "int8"
    assert quant.format_for("mla_absorb") == quant.source_format_for("mla_absorb") == "bf16"
    expected_delta = 0
    for plan in selected.model.layer_plans:
        term = next(t for t in layer_weight_terms(plan, selected.model) if t.name == "kv_b")
        expected = term_bytes(term, quant, parallel)
        assert expected == term_bytes(term, quant, parallel, source=True)
        prefix = f"s0.l{plan.index}"
        pair = [after[prefix + suffix] for suffix in (".q_absorb", ".v_expand")]
        assert expected == sum(dict(op.metadata)["packed_weight_bytes"] for op in pair)
        for op in pair:
            assert op.flops == before[op.op_id].flops
            assert dict(op.metadata)["weight_format"] == op.dtype == "bf16"
            assert op.op_id + ".quantize" not in after
            assert op.op_id + ".quantize" in before
        old_plan = baseline.model.layer_plans[plan.index]
        old_term = next(t for t in layer_weight_terms(old_plan, baseline.model) if t.name == "kv_b")
        expected_delta += expected - term_bytes(old_term, baseline.model.quant, parallel)
    for source in (False, True):
        old_bytes = sum(stage_weight_bytes(baseline.model, parallel, source=source))
        new_bytes = sum(stage_weight_bytes(selected.model, parallel, source=source))
        assert new_bytes - old_bytes == expected_delta


def test_unselected_absorption_inherits_existing_mla_precision_exclusion():
    case = parse_case(_raw(selected=False))
    quant = replace(case.model.quant, exclusions=(*case.model.quant.exclusions, "mla"))
    case = replace(case, model=replace(case.model, quant=quant))
    assert quant.format_for("mla_absorb") == quant.source_format_for("mla_absorb") == "bf16"
    ops = target_operators(case, phase="prefill", tokens=32)
    absorbed = [op for op in ops if op.recipe in {"q_absorb", "v_expand"}]
    assert absorbed
    assert all(dict(op.metadata)["weight_format"] == op.dtype == "bf16" for op in absorbed)
    term = next(t for t in layer_weight_terms(case.model.layer_plans[0], case.model)
                if t.name == "kv_b")
    assert term_bytes(term, quant, case.runtime.parallel) == sum(
        dict(op.metadata)["packed_weight_bytes"] for op in absorbed if op.layer == 0)


def test_group_expands_real_source_call_members_without_duplicating_work_or_dropping_join():
    case = parse_case(mla_case())
    ops = list(target_operators(case, phase="decode", tokens=1, context=16))
    call = next(op for op in ops if op.source_boundary == "call")
    expansion = next(op for op in ops if op.op_id.endswith(".v_expand"))
    declaration = dict(model=case.model.to_dict(), execution={"phase": "decode"},
                       kernels=case.runtime.kernels)
    original = export_op_structure(ops, **declaration)
    declare_group(
        ops, key="s0.l0.decode", role="declared_decode", family="mla",
        members=(call.op_id, expansion.op_id), entries=(call.op_id,),
        anchor=expansion.op_id, view={},
        implementation={"source_selection": "explicit_test_of_existing_source_call"},
    )
    report = export_op_structure(ops, **declaration)
    group = report["subgraphs"][0]
    source_call = next(row for row in report["source_calls"]
                       if row["node_id"] == f"rank:0/{call.op_id}")
    assert set(group["node_ids"]) == {
        *source_call["children"], source_call["node_id"], f"rank:0/{expansion.op_id}"}
    assert set(group["external_inputs"]) == {f"rank:0/{dep}" for dep in call.deps}
    assert group["external_outputs"] == [f"rank:0/{expansion.op_id}"]
    assert group["lineage"]["source_call_ids"] == [source_call["node_id"]]
    assert [node["node_id"] for node in report["nodes"]] == [
        node["node_id"] for node in original["nodes"]]
    changed_work = [replace(op, flops=op.flops + 1234, hbm_bytes=op.hbm_bytes + 42) for op in ops]
    changed = export_op_structure(changed_work, **declaration)["subgraphs"]
    without_work = lambda groups: [
        {key: value for key, value in group.items()
         if key not in {"physical_signature", "physical_work"}} for group in groups]
    assert without_work(changed) == without_work(report["subgraphs"])
    assert changed[0]["physical_signature"] != group["physical_signature"]
    assert changed[0]["physical_work"]["flops"] > group["physical_work"]["flops"]


def test_norm_quant_fusion_keeps_declared_group_membership_and_lineage():
    from mindie_sim.catalog.ops import gemm_op, vector_op
    from mindie_sim.catalog.quant import quant_preset
    from mindie_sim.prediction.transforms import activation_quantization, norm_quant

    quant = quant_preset("w4a8_dynamic")
    common = dict(stage=0, layer=0, phase="prefill", requests=())
    ops = [
        vector_op(op_id="norm", deps=("input",), elements=8 * 32,
                  ops_per_element=5, dtype="bf16", family="rmsnorm", recipe="rmsnorm",
                  shape=(8, 32), **common),
        gemm_op(op_id="projection", deps=("norm",), m=8, n=64, k=32,
                quant=quant, category="mla", **common),
    ]
    declare_group(
        ops, key="norm_projection", role="declared_norm_projection", family="linear",
        members=("norm", "projection"), entries=("norm",), anchor="projection",
        view={}, implementation={"source_selection": "declared_norm_and_projection"},
    )
    lowered = norm_quant(activation_quantization(ops, "bf16"), "bf16")
    report = export_op_structure(lowered, model={}, execution={"phase": "prefill"})
    group = report["subgraphs"][0]
    nodes = {node["node_id"]: node for node in report["nodes"]}
    quant = nodes["rank:0/projection.quantize"]
    assert quant["family"] == "rmsnorm_quant"
    assert quant["lineage"]["fused_from"] == ["rmsnorm", "quantize"]
    assert quant["node_id"] in group["node_ids"]
    assert group["external_outputs"] == ["rank:0/projection"]
    assert group["external_inputs"] == ["rank:0/input"]
    assert "rank:0/norm" not in nodes


def test_kimi_retains_declared_bf16_gap_against_real_fp8_prolog(profiler_api):
    from mindie_profiler.tensor_contracts import observed_contract

    rows = _fixture("kimi_a5_full_model.csv", profiler_api, "body_attention")
    actual = next(observed_contract(row) for row in rows if row.kind == "MlaPrologV3")
    report = _export(_raw("kimi-k3-a5"), phase="decode", tokens=1, context=32768)
    groups = [g for g in report["subgraphs"] if g["role"] == "mla_prolog"]
    assert groups
    assert actual[2][1] == "fp8_e4m3"
    assert all(g["contract"][2][1] == "bf16" for g in groups)
    assert all(g["contract"] != _json(actual) for g in groups)
    # The selectable source boundary does not override the independent
    # checkpoint's declared precision to fit a captured kernel.
    assert report["model"]["quant"]["weight_format"] == "bf16"
