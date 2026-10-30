"""GQA QKV/output roles preserve observed shapes and independent representations.

[Role] Check frontend, quant producers, weight residency and Engine work together.
[Invariants] The retained Qwen 35B TP4 and 397B TP16 captures establish output
  shapes (256,2048,1024)/(256,4096,512), BF16 operands, and INT8 body QKV.
  Their timing measurements do not enter these representation controls.
[Timeouts & Locks] Local metadata and bounded component execution; no waits.
[Side-Effect Policy] Read retained cases; no devices, remote work or writes.
"""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import operator_costs
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.attention import gqa
from mindie_sim.prediction.partial_attention import projection_only
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.schema.workload import RequestView


BF16 = {"weight_format": "bf16", "activation_format": "bf16"}
INT8 = {"weight_format": "int8", "activation_format": "int8"}
CASES = (
    ("new9/qwen35-35b-a5-tp4-c64-graph", (256, 2560, 2048), (256, 2048, 1024), 10),
    ("new17/qwen35-397b-a3-tp16-c48", (256, 1536, 4096), (256, 4096, 512), 15),
)


def component(name, overrides=None, *, partial=False):
    path = Path(__file__).parents[1] / "examples/cases/high-concurrency-20261030" / (name + ".json")
    case = parse_case(json.loads(path.read_text()))
    if overrides:
        quant = replace(case.model.quant, category_overrides={
            **dict(case.model.quant.category_overrides), **overrides})
        case = replace(case, model=replace(case.model, quant=quant))
    case = replace(case, runtime=replace(case.runtime,
        kernels={"programs": {"matmul": {"kind": "steady_matmul"}}}))
    plan = next(plan for plan in case.model.layer_plans if plan.attention == "gqa")
    builder = projection_only if partial else gqa
    result = builder(prefix="gqa", plan=plan, model=case.model, quant=case.model.quant,
        parallel=case.runtime.parallel, views=(RequestView("r", 256, 4096, True, "decode"),),
        stage=0, deps="analysis_input", mode="decode", padded_tokens=256,
        cache_ids={}, stage_of=lambda _: 0)
    ops = activation_quantization(result.ops, case.model.activation_dtype)
    return case, plan, ops, {op.recipe: op for op in ops
                            if op.recipe in {"qkv_projection", "attention_output"}}


@pytest.mark.parametrize("name,qkv_shape,out_shape,occurrences", CASES)
@pytest.mark.parametrize("partial", (False, True))
def test_native_output_preserves_int8_qkv_producer_and_matching_residency(
        name, qkv_shape, out_shape, occurrences, partial):
    case, plan, ops, selected = component(name, {"attn.o": BF16}, partial=partial)
    qkv, output = selected["qkv_projection"], selected["attention_output"]
    assert qkv.local_shape == qkv_shape and output.local_shape == out_shape
    assert sum(p.attention == "gqa" for p in case.model.layer_plans) == occurrences
    assert qkv.dtype == "int8" and output.dtype == "bf16"
    by_id = {op.op_id: op for op in ops}
    producer = by_id[qkv.deps[0]]
    assert producer.family == "quantize"
    assert producer.mte3_bytes == qkv_shape[0] * qkv_shape[2] + qkv_shape[0] * 4
    assert all(by_id[dep].family != "quantize" for dep in output.deps)
    output_meta = dict(output.metadata)
    assert output_meta["weight_format"] == output_meta["activation_format"] == "bf16"
    assert output_meta["quant_category"] == "attn.o"
    assert not output_meta.get("activation_scale_bytes")
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    assert output_meta["packed_weight_bytes"] == out_shape[1] * out_shape[2] * 2
    assert term_bytes(terms["o_proj"], case.model.quant, case.runtime.parallel) == output_meta["packed_weight_bytes"]
    assert sum(term_bytes(terms[name], case.model.quant, case.runtime.parallel)
               for name in ("q_proj", "k_proj", "v_proj")) == dict(qkv.metadata)["packed_weight_bytes"]
    priced = operator_costs(case, (qkv, output), input_ready_ms=7)
    peaks = dict(case.machine.dies[0].compute_peaks)
    for row in priced["operators"]:
        op = selected[row["operator"]["recipe"]]
        matrix = sum(value for role, value in row["declared_service"].items()
                     if role.startswith("matrix@"))
        assert matrix == pytest.approx(1000 * op.flops / peaks[op.dtype])
        assert not row["service_before_input"] and not row["live_tasks"]


def test_parent_and_child_precision_inherit_without_model_dispatch():
    _, _, ops, selected = component(CASES[0][0], {"attn": BF16, "attn.qkv": INT8})
    assert selected["qkv_projection"].dtype == "int8"
    assert selected["attention_output"].dtype == "bf16"
    quantized = [op for op in ops if op.family == "quantize"]
    assert len(quantized) == 1 and selected["qkv_projection"].deps == (quantized[0].op_id,)


@pytest.mark.parametrize("name,qkv_shape,out_shape,occurrences", CASES)
def test_undeclared_roles_keep_parent_work_and_explicit_bf16_changes_only_output(
        name, qkv_shape, out_shape, occurrences):
    case, _, _, selected = component(name)
    for op in selected.values():
        m, n, k = op.local_shape
        parent = gemm_op(op_id=op.op_id, deps=(), stage=op.stage, layer=op.layer,
            requests=op.requests, m=m, n=n, k=k, quant=case.model.quant,
            category="attn", phase=op.phase, recipe=op.recipe, family=op.family)
        for field in ("dtype", "flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
            assert getattr(op, field) == getattr(parent, field)
    _, _, _, mixed = component(name, {"attn.o": BF16})
    qkv, output = selected["qkv_projection"], selected["attention_output"]
    assert mixed["qkv_projection"] == qkv
    assert mixed["attention_output"].flops == output.flops
    assert mixed["attention_output"].local_shape == output.local_shape
    assert mixed["attention_output"].mte3_bytes == output.mte3_bytes
