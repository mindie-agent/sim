"""Quantization roles, resident formats, and native execution precision.

[Role] Check named presets and role precedence from parsing through lowering,
  matrix service, and weight residency.
[Invariants] Packing and scales follow placed shapes. Format overrides preserve
  logical matrix work and select only supported native execution precision.
[Timeouts & Locks] Bounded component graphs and local pricing. No locks.
[Side-Effect Policy] Read retained case JSON and build cases in memory.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from _support.builders import hardware, quant
from _support.memory_cases import SPARSE_OFFLOAD_SKU, case as memory_case
from mindie_sim import Case, compose_case, graph_costs, operator_costs, target_graphs
from mindie_sim.catalog.ops import compute_dtype_for, gemm_op
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.catalog.types import EngramSpec, EngramTableSpec, QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, term_bytes, term_payload_bytes
from mindie_sim.catalog.weights import layer_weight_terms, stage_weight_bytes, stage_weight_decomposition
from mindie_sim.prediction.operators.attention.attention import gqa
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.operators.matrix.matrix_epilogue import MatrixEpilogue
from mindie_sim.prediction.operators.matrix.matrix_precision import execution_compute_peaks, resolve_matrix_precision
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.prediction.operators.attention.partial_attention import projection_only
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.transforms import activation_quantization
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_peak_aliases import (
    native_compute_peak_items, resolve_compute_peak_name, validate_compute_peak_aliases,
)
from mindie_sim.schema.hardware import Hardware
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView

pytestmark = pytest.mark.tier0

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples/cases"
LEGACY_FP8_LIMITED_SKU = "Ascend910B2"

ROLES = {
    "linear_qkvz": ("gdn.qkvz", "gdn_qkvz"),
    "linear_beta_alpha": ("gdn.beta_alpha", "gdn_ba"),
    "linear_output": ("gdn.o", "gdn_o"),
}

BF16 = {"weight_format": "bf16", "activation_format": "bf16"}
INT8 = {"weight_format": "int8", "activation_format": "int8"}

def gdn_case(*, quantization="w8a8", overrides=None, source_prefill=False):
    path = EXAMPLES / "qwen36-a3-tp4.json"
    raw = json.loads(path.read_text())
    raw["model"].update(layers=1, quantization=quantization)
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "steady_matmul"}}}
    if quantization == "w8a8_mxfp8" or source_prefill:
        raw["machine"] = {"backend": "ascend", "sku": SPARSE_OFFLOAD_SKU, "num_devices": 4}
    if source_prefill:
        raw["runtime"]["kernels"]["lowerings"] = {"gdn_prefill": "gdn_prefill_arch35"}
    case = parse_case(raw)
    if overrides:
        raw["model"]["quant"] = replace(case.model.quant, category_overrides=overrides).to_dict()
        case = parse_case(raw)
    return case

def projections(case, phase):
    graphs = target_graphs(case, phase=phase, tokens=4, context=4096, scope="body")
    ops = graphs[0]
    selected = {op.recipe: op for op in ops if op.recipe in ROLES}
    assert len(selected) == 3
    return graphs, selected

def assert_native_matrix_service(case, ops, input_ready_ms):
    priced = operator_costs(case, tuple(ops), input_ready_ms=input_ready_ms)
    peaks = dict(case.machine.dies[0].compute_peaks)
    by_recipe = {op.recipe: op for op in ops}
    for row in priced["operators"]:
        op = by_recipe[row["operator"]["recipe"]]
        matrix = sum(value for role, value in row["declared_service"].items()
                     if role.startswith("matrix@"))
        assert matrix == pytest.approx(1000 * op.flops / peaks[op.dtype])
        assert not row["service_before_input"] and not row["live_tasks"]

def check_formats_and_producers(case, graphs, selected, expected):
    by_id = {op.op_id: op for op in graphs[0]}
    terms = {term.name: term for term in layer_weight_terms(case.model.layer_plans[0], case.model)}
    for recipe, op in selected.items():
        category, term = ROLES[recipe]
        metadata = dict(op.metadata)
        rows, n, k = op.local_shape
        fmt = expected[recipe]
        assert metadata["quant_category"] == category
        assert metadata["weight_format"] == metadata["activation_format"] == op.dtype == fmt
        assert metadata["packed_weight_bytes"] == term_bytes(
            terms[term], case.model.quant, case.runtime.parallel)
        assert op.flops == 2 * rows * n * k
        if fmt == "bf16":
            assert metadata["packed_weight_bytes"] == 2 * n * k
            assert not metadata.get("activation_scale_bytes")
            assert metadata.get("activation_producer") != "explicit_quantization"
            assert all(by_id[dep].family not in {"quantize", "rmsnorm_quant"}
                       for dep in op.deps if dep in by_id)
        else:
            producer = by_id[op.deps[0]]
            assert producer.family in {"quantize", "rmsnorm_quant"}
            assert metadata["activation_producer"] == "explicit_quantization"
            assert metadata["activation_scale_bytes"] == 4 * rows
            assert producer.mte3_bytes == rows * k + 4 * rows
    assert_native_matrix_service(case, selected.values(), 3)

GDN_FRONTENDS = (("decode", False), ("prefill", False), ("prefill", True))
GDN_SHAPES = {
    "linear_qkvz": (4, 4096, 5120), "linear_beta_alpha": (4, 24, 5120),
    "linear_output": (4, 5120, 1536),
}

@pytest.mark.parametrize("phase,source", GDN_FRONTENDS)
@pytest.mark.parametrize("weight_policy", ("mixed", "excluded"))
def test_gdn_roles_preserve_shapes_and_representation(phase, source, weight_policy):
    mixed = weight_policy == "mixed"
    case = gdn_case(overrides={"gdn.beta_alpha": BF16, "gdn.o": BF16} if mixed else None,
                    quantization="w8a8" if mixed else "w8a8_mxfp8", source_prefill=source)
    if not mixed:
        case = parse_case({**case.to_dict(), "model": {
            "preset": "qwen3.8-27b", "layers": 1, "quantization": "w8a8_mxfp8"}})
    graphs, selected = projections(case, phase)
    expected = dict.fromkeys(ROLES, "bf16")
    if mixed:
        expected["linear_qkvz"] = "int8"
        assert {name: op.local_shape for name, op in selected.items()} == GDN_SHAPES
        result = graph_costs(case, graphs, external_inputs={"analysis_input": 0}, isolated=False)
        assert result["duration_ms"] > 0
        assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    check_formats_and_producers(case, graphs, selected, expected)

def test_gdn_child_override_takes_precedence_over_parent():
    case = gdn_case(overrides={"gdn": BF16, "gdn.qkvz": INT8})
    graphs, selected = projections(case, "decode")
    check_formats_and_producers(case, graphs, selected, {
        "linear_qkvz": "int8", "linear_beta_alpha": "bf16", "linear_output": "bf16"})

def assert_parent_matrix_work(op, quant, category):
    m, n, k = op.local_shape
    original = gemm_op(op_id=op.op_id, deps=(), stage=op.stage, layer=op.layer,
                       requests=op.requests, m=m, n=n, k=k, quant=quant,
                       category=category, phase=op.phase, recipe=op.recipe, family=op.family)
    for field in ("dtype", "flops", "vector_ops", "hbm_bytes", "mte2_bytes", "mte3_bytes"):
        assert getattr(op, field) == getattr(original, field)
    return dict(original.metadata)

@pytest.mark.parametrize("quantization", ("w8a8", "w8a8_mxfp8"))
def test_undeclared_gdn_roles_keep_parent_category_physical_work(quantization):
    case = gdn_case(quantization=quantization)
    _, selected = projections(case, "decode")
    for op in selected.values():
        previous = assert_parent_matrix_work(op, case.model.quant, "gdn")
        metadata = dict(op.metadata)
        for field in ("packed_weight_bytes", "weight_format", "activation_format",
                      "activation_scale_bytes", "matrix_epilogue"):
            assert metadata.get(field) == previous.get(field)

CASES = (
    ("new9/qwen35-35b-a5-tp4-c64-graph", (256, 2560, 2048), (256, 2048, 1024), 10),
    ("new17/qwen35-397b-a3-tp16-c48", (256, 1536, 4096), (256, 4096, 512), 15),
)

def component(name, overrides=None, *, partial=False):
    path = EXAMPLES / "high-concurrency-20261030" / (name + ".json")
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
    assert output_meta["quant_category"] == "attn.o" and not output_meta.get("activation_scale_bytes")
    terms = {term.name: term for term in layer_weight_terms(plan, case.model)}
    assert output_meta["packed_weight_bytes"] == out_shape[1] * out_shape[2] * 2
    assert term_bytes(terms["o_proj"], case.model.quant, case.runtime.parallel) == output_meta["packed_weight_bytes"]
    assert sum(term_bytes(terms[name], case.model.quant, case.runtime.parallel)
               for name in ("q_proj", "k_proj", "v_proj")) == dict(qkv.metadata)["packed_weight_bytes"]
    assert_native_matrix_service(case, (qkv, output), 7)

def test_parent_and_child_precision_inherit_without_model_dispatch():
    _, _, ops, selected = component(CASES[0][0], {"attn": BF16, "attn.qkv": INT8})
    assert selected["qkv_projection"].dtype == "int8"
    assert selected["attention_output"].dtype == "bf16"
    quantized = [op for op in ops if op.family == "quantize"]
    assert len(quantized) == 1 and selected["qkv_projection"].deps == (quantized[0].op_id,)

@pytest.mark.parametrize("name", [row[0] for row in CASES])
def test_undeclared_roles_keep_parent_work_and_explicit_bf16_changes_only_output(name):
    case, _, _, selected = component(name)
    for op in selected.values():
        assert_parent_matrix_work(op, case.model.quant, "attn")
    _, _, _, mixed = component(name, {"attn.o": BF16})
    qkv, output = selected["qkv_projection"], selected["attention_output"]
    assert mixed["qkv_projection"] == qkv
    assert mixed["attention_output"].flops == output.flops
    assert mixed["attention_output"].local_shape == output.local_shape
    assert mixed["attention_output"].mte3_bytes == output.mte3_bytes

def matrix(quant, *, category='dense_mlp', rows=5, columns=7, width=11, groups=1):
    return gemm_op(op_id='projection', deps=('input',), stage=0, layer=0,
                   requests=(), m=rows, n=columns, k=width, weight_count=groups,
                   quant=quant, category=category, phase='prefill')

def test_bf16_scale_only_is_one_column_input_and_fp32_row_scales_stay_separate():
    quant = QuantSpec('declared-native', weight_format='int8', scale_format='bf16',
                      scale_bytes=2, int8_weight_metadata='scale_only')
    op = matrix(quant)
    meta = dict(op.metadata)
    epilogue = MatrixEpilogue.from_op(op)
    assert quant.matrix_payload_bytes(7, 11, 'dense_mlp') == 77
    assert quant.matrix_bytes(7, 11, 'dense_mlp') == 77 + 14
    assert epilogue.column_scale_bytes == 14
    assert epilogue.row_scale_bytes == 5 * 4
    assert epilogue.operands(op) == (5 * 11, 7 * 11)
    assert (op.mte2_bytes, op.mte3_bytes, op.hbm_bytes) == (166, 70, 236)
    assert op.flops == 2 * 5 * 7 * 11
    assert op.vector_ops == 4 * 5 * 7
    assert meta['weight_scale_format'] == 'bf16'
    producer, consumer = activation_quantization((op,), 'bf16')
    assert producer.mte3_bytes == 5 * 11 + 5 * 4
    assert consumer.deps == (producer.op_id,)
    assert epilogue.scales(5, 7, 5, 7, 1, reuse_inputs=True, reuse_weights=True) == (34, 34)

def test_explicit_scale_only_does_not_change_other_quantization_categories():
    legacy = QuantSpec('int8', weight_format='int8')
    selected = replace(legacy, category_overrides={
        'dense_mlp': {'scale_format': 'bf16', 'scale_bytes': 2,
                      'int8_weight_metadata': 'scale_only'}})
    restored = QuantSpec(**json.loads(json.dumps(selected.to_dict())))
    for current in (selected, restored):
        assert matrix(current, category='mla.q_b') == matrix(legacy, category='mla.q_b')
        assert matrix(current).mte2_bytes == 166
        assert current.matrix_bytes(7, 11, 'dense_mlp') == 91
        assert current.matrix_bytes(7, 11, 'mla.q_b') == 77 + 7 * 8
    assert MatrixEpilogue.from_op(matrix(legacy)).column_scale_bytes == 7 * 8

@pytest.mark.parametrize('groups', [1, 3])
def test_grouped_scale_inputs_and_partial_tiles_do_not_duplicate_column_metadata(groups):
    quant = QuantSpec('native', weight_format='int8', scale_bytes=2,
                      scale_format='bf16', int8_weight_metadata='scale_only')
    op = matrix(quant, rows=5 * groups, groups=groups)
    epilogue = MatrixEpilogue.from_op(op)
    assert epilogue.column_scale_bytes == groups * 7 * 2
    assert epilogue.operands(op) == (groups * 5 * 11, groups * 7 * 11)
    tiles = [(5, 3), (5, 4)] * groups
    traffic = sum(epilogue.scales(a, b, 5 * groups, 7, groups,
                                 reuse_inputs=True, reuse_weights=True)[1]
                  for a, b in tiles)
    assert traffic == pytest.approx(epilogue.row_scale_bytes + epilogue.column_scale_bytes)

def test_unknown_column_metadata_cannot_silently_choose_scale_offset():
    with pytest.raises(ModelingError, match='int8_weight_metadata'):
        QuantSpec('invalid', int8_weight_metadata='infer_from_name')

def ascend_dies(sku=SPARSE_OFFLOAD_SKU):
    return Hardware.from_mapping({"backend": "ascend", "sku": sku, "num_devices": 1}).dies

def mxa8w4(activation="fp8_e4m3", *, category="moe.routed"):
    return gemm_op(
        op_id="mx", deps=("x",), stage=0, layer=0, requests=("r",),
        m=4, n=64, k=128, quant=quant(
            weight_format="mxfp4", activation_format=activation),
        category=category, phase="decode")

def test_structural_identity_and_two_argument_dtype_contract():
    op = mxa8w4()
    metadata = dict(op.metadata)
    assert op.dtype == "bf16"
    assert tuple(metadata[key] for key in ("weight_format", "activation_format", "output_format")) == (
        "mxfp4", "fp8_e4m3", "bf16")
    assert compute_dtype_for(quant(weight_format="mxfp4",
                                   activation_format="fp8_e4m3"),
                             "moe.routed") == ("bf16", "fp8_e4m3")
    assert metadata["compute_format_origin"] == "catalog_default"

def test_eligible_generic_work_resolves_to_native_fp8():
    op = mxa8w4()
    resolved = resolve_matrix_precision(op, devices=ascend_dies(), policy="auto",
                                        recipe=None)
    metadata = dict(resolved.metadata)
    assert resolved.dtype == "fp8_e4m3"
    assert resolve_matrix_precision(resolved, devices=ascend_dies(), policy="auto", recipe=None) == resolved
    assert metadata["declared_compute_format"] == "bf16"
    assert metadata["execution_compute_format"] == "fp8_e4m3"
    assert metadata["weight_format"] == "mxfp4"
    assert metadata["activation_format"] == "fp8_e4m3"
    assert (resolved.local_shape, resolved.flops) == (op.local_shape, op.flops)
    peaks = execution_compute_peaks(
        resolved, devices=ascend_dies(),
        base_peaks={"bf16": ascend_dies()[0].peak("bf16")})
    assert peaks["fp8_e4m3"] == pytest.approx(2 * peaks["bf16"])

PRECISION_CAPABILITIES = (
    ("bf16", "fp8_e4m3", "bf16"), ("generic_fp8", "fp8_e4m3", "fp8_e4m3"),
    ("mixed", "fp8_e4m3", "bf16"), ("native_fp8", "fp8_e5m2", "fp8_e5m2"),
    ("simulated", "fp8_e5m2", "bf16"),
)

@pytest.mark.parametrize("capability,activation,dtype", PRECISION_CAPABILITIES)
def test_operator_precision_requires_support_on_every_die(capability, activation, dtype):
    devices = {
        "bf16": hardware(1, peaks=(("bf16", 1e12),)).dies,
        "generic_fp8": hardware(1, peaks=(("bf16", 1e12), ("fp8", 2e12))).dies,
        "mixed": (*ascend_dies(), replace(ascend_dies(LEGACY_FP8_LIMITED_SKU)[0], rank=1)),
        "native_fp8": ascend_dies(),
        "simulated": hardware(1).dies,
    }[capability]
    resolved = resolve_matrix_precision(mxa8w4(activation), devices=devices, policy="auto", recipe=None)
    assert resolved.dtype == dtype
    if capability == "bf16":
        assert dict(resolved.metadata)["precision_resolution_reason"] == "fp8_unavailable"

def test_declared_policy_and_source_bound_identity_are_preserved():
    op = mxa8w4()
    assert resolve_matrix_precision(op, devices=ascend_dies(), policy="declared",
                                    recipe=None) == op
    source = replace(op, source_boundary="kernel")
    kept = resolve_matrix_precision(source, devices=ascend_dies(), policy="auto",
                                    recipe=None)
    assert kept.dtype == "bf16"
    assert dict(kept.metadata)["precision_resolution_reason"] == "explicit_compute_identity"
    explicit = gemm_op(
        op_id="e", deps=(), stage=0, layer=0, requests=(), m=2, n=8, k=8,
        quant=quant(weight_format="mxfp4", activation_format="fp8_e4m3"),
        category="moe.routed", phase="decode",
        metadata=(("compute_format", "bf16"),))
    assert dict(explicit.metadata)["compute_format_origin"] == "explicit"
    assert resolve_matrix_precision(explicit, devices=ascend_dies(), policy="auto",
                                    recipe=None).dtype == "bf16"

def test_supply_alias_is_validated_and_never_a_native_operand():
    base = hardware(1, peaks=(("bf16", 1e12), ("fp8_e4m3", 2e12),
                              ("mxfp4", 2e12)))
    die = replace(base.dies[0], compute_peak_aliases=(("mxfp4", "fp8_e4m3"),))
    peaks = dict(die.compute_peaks)
    validate_compute_peak_aliases(peaks, dict(die.compute_peak_aliases), die.compute_arch)
    assert resolve_compute_peak_name("mxfp4", peaks,
                                     dict(die.compute_peak_aliases)) == "fp8_e4m3"
    assert "mxfp4" not in dict(native_compute_peak_items(
        peaks, dict(die.compute_peak_aliases)))
    assert die.peak("mxfp4") == pytest.approx(die.peak("fp8_e4m3"))
    with pytest.raises(ModelingError, match="matrix_precision_not_native"):
        execution_compute_peaks(replace(mxa8w4(), dtype="mxfp4"),
                                devices=(die,), base_peaks={})

@pytest.mark.parametrize("peaks,aliases,reason", [
    ({"bf16": 1, "fp8_e4m3": 2, "mxfp4": 3}, {"mxfp4": "fp8_e4m3"}, "value_mismatch"),
    ({"bf16": 1, "mxfp4": 1}, {"mxfp4": "fp8_e5m2"}, "unknown_target"),
    ({"a": 1, "b": 1}, {"a": "b", "b": "a"}, "cycle"),
])
def test_compute_supply_aliases_reject_inconsistent_bindings(peaks, aliases, reason):
    with pytest.raises(ModelingError, match=f"compute_peak_alias_{reason}"):
        validate_compute_peak_aliases(peaks, aliases, hardware(1).dies[0].compute_arch)

def quant_plan():
    return QuantSpec(
        "mixed", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1,
        activation_group_size=32, activation_scale_format="fp8_e8m0",
        activation_scale_pack=2,
        category_overrides={
            "projection": {"weight_format": "fp8_e4m3", "activation_format": "fp8_e4m3"},
            "projection.gate": {"weight_format": "bf16", "activation_format": "bf16"},
            "shared_expert": {"weight_format": "mxfp4", "activation_format": "fp8_e4m3"},
        },
    )

def test_role_inheritance_retains_independent_scale_geometry_after_json_roundtrip():
    quant = quant_plan()
    restored = QuantSpec(**json.loads(json.dumps(quant.to_dict())))
    for plan in (quant, restored):
        assert plan.matrix_payload_bytes(5, 65, "projection.q") == 325
        assert plan.matrix_bytes(5, 65, "projection.q") == 340
        assert plan.matrix_bytes(5, 65, "projection.gate") == 650
        assert plan.matrix_payload_bytes(5, 65, "shared_expert") == 165
        assert plan.matrix_bytes(5, 65, "shared_expert") == 180
        assert plan.matrix_bytes(5, 65, "shared_expert", source=True) == 180
    assert quant == restored
    assert replace(quant, exclusions=("projection",)).matrix_bytes(5, 65, "projection.q") == 650

def test_fp4_preserves_declared_fp8_payload_and_real_mx_quantization_producer():
    op = gemm_op(
        op_id="shared", deps=(), stage=0, layer=0, requests=(),
        m=3, n=5, k=65, quant=quant_plan(), category="shared_expert", phase="prefill",
    )
    meta = dict(op.metadata)
    assert op.dtype == "bf16"  # Explicit expanded compute; no FP4 peak inferred.
    assert op.flops == 1950
    assert (meta["weight_format"], meta["activation_format"]) == ("mxfp4", "fp8_e4m3")
    assert (meta["activation_scale_shape"], meta["activation_scale_bytes"]) == ((3, 2, 2), 12)
    assert (op.mte2_bytes, op.mte3_bytes, op.hbm_bytes) == (195 + 12 + 180, 30, 417)
    quantize, matrix = activation_quantization((op,), "bf16")
    assert matrix.deps == (quantize.op_id,)
    assert (quantize.mte2_bytes, quantize.mte3_bytes, quantize.vector_ops) == (390, 207, 798)
    assert dict(quantize.metadata)["activation_scale_format"] == "fp8_e8m0"

def test_checkpoint_roles_use_the_same_representation_in_ops_and_weight_terms():
    path = EXAMPLES / "kimi-k3-a5-current-checkpoint-conditional.json"
    raw = json.loads(path.read_text())
    raw["model"]["layers"] = 4  # KDA+dense, KDA+MoE and MLA+MoE.
    case = parse_case(raw)
    restored = parse_case({**raw, "model": case.model.to_dict()})
    assert case.model.quant == restored.model.quant
    ops = target_operators(restored, phase="decode", tokens=1, context=32)
    expected = {
        "kda.q": ("kda_q", "fp8_e4m3"), "kda.k": ("kda_k", "fp8_e4m3"),
        "kda.v": ("kda_v", "fp8_e4m3"), "kda.g": ("kda_g", "bf16"),
        "kda.b": ("kda_b", "bf16"), "kda.f_a": ("kda_f_a", "bf16"),
        "kda.f_b": ("kda_f_b", "bf16"), "kda.o": ("kda_o", "bf16"),
        "mla.q_a": ("q_a", "fp8_e4m3"), "mla.q_b": ("q_b", "fp8_e4m3"),
        "mla.kv_a": ("kv_a", "fp8_e4m3"), "mla.g": ("mla_g", "bf16"),
        "mla.o": ("o_proj", "bf16"),
    }
    seen = set()
    for op in ops:
        meta = dict(op.metadata)
        category = meta.get("quant_category")
        slices = meta.get("packed_projection_slices") or ()
        categories = ([str(row["category"]) for row in slices] if slices else [category])
        if not set(categories) & set(expected):
            continue
        term_name, fmt = expected[category]
        terms = layer_weight_terms(case.model.layer_plans[op.layer], case.model)
        by_category = {term.category: term for term in terms}
        assert meta["weight_format"] == fmt
        if slices:
            frame = sum(term_bytes(by_category[str(row["category"])],
                                   case.model.quant, case.runtime.parallel)
                        for row in slices)
            for row in slices:
                assert expected[str(row["category"])][1] == meta["activation_format"]
        else:
            frame = term_bytes(next(term for term in terms if term.name == term_name),
                               case.model.quant, case.runtime.parallel)
        assert meta["packed_weight_bytes"] == frame
        if fmt == "fp8_e4m3":
            assert meta["activation_producer"] == "explicit_quantization"
        seen.update(categories)
    assert seen == expected.keys()
    matrices = [op for op in ops if dict(op.metadata).get("quant_category") == "shared_expert"]
    assert matrices
    assert all(dict(op.metadata)["weight_format"] == "mxfp4" for op in matrices)
    assert all(dict(op.metadata)["activation_format"] == "fp8_e4m3" for op in matrices)

def model_case(preset='w8a8_mxfp8'):
    raw = json.loads((ROOT/'examples/quickstart.json').read_text())
    raw['model'].update(layers=1, ffn='moe', num_experts=8, top_k=2,
                        moe_intermediate_size=256, num_shared_experts=1, quantization=preset)
    raw['machine'] = {'backend': 'ascend', 'sku': 'Ascend950DT_9582', 'num_devices': 4}
    raw['runtime'].update(tp=2, ep=4, dp=2, graph_mode='none', routing={'policy': 'balanced'})
    raw['runtime']['kernels'] = {'programs': {'matmul': {'kind': 'steady_matmul'}},
        'recipes': {'balanced_grouped_down': {'kind': 'ascend_mxfp8_grouped_nz'}}}
    return raw

@pytest.mark.parametrize('preset', ['w8a8_mxfp8', 'w8a8-mxfp8', 'mxfp8', 'fp8_mxfp8', 'fp8'])
def test_registered_mx_presets_equal_explicit_operands_but_keep_residuals_bf16(preset):
    raw = model_case(preset)
    explicit = deepcopy(raw)
    explicit['model'].pop('quantization')
    explicit['model']['quant'] = quant_preset(preset).to_dict()
    case, reference = parse_case(raw), parse_case(explicit)
    assert case.model.quant == reference.model.quant
    assert case.model.activation_dtype == case.model.compute_dtype == 'bf16'
    assert case.model.weight_dtype == 'fp8_e4m3'
    kwargs = dict(phase='decode', tokens=1, sequences=2, context=128, scope='body')
    graphs = target_graphs(case, **kwargs)
    assert graphs == target_graphs(reference, **kwargs)
    ops = graphs[0]
    norms = [op for op in ops if op.family == 'rmsnorm']
    assert norms and all(op.dtype == 'bf16' for op in norms)
    matrices = [op for op in ops if dict(op.metadata).get('weight_format') == 'fp8_e4m3']
    assert matrices and all(op.dtype == 'fp8_e4m3' for op in matrices)
    assert all(dict(op.metadata)['activation_scale_format'] == 'fp8_e8m0' for op in matrices)
    assert any(op.family == 'quantize' for op in ops)
    if preset == 'w8a8_mxfp8':
        result = graph_costs(case, graphs, input_ready_ms=7, external_inputs={"analysis_input": 7})
        assert result['duration_ms'] > 0
        assert result['execution']['live_tasks'] == result['execution']['live_programs'] == 0
        assert not any(result['execution']['live_allocated'].values())
        selected = [row for row in result['isolated']['operators']
                    if row['operator']['recipe'] == 'balanced_grouped_down']
        assert selected and all(row['program'] == 'OperatorPlan' for row in selected)

def test_explicit_storage_and_residual_dtypes_keep_precedence_over_preset_defaults():
    raw = model_case()
    raw['model'].update(weight_dtype='fp16', activation_dtype='fp32',
                        compute_dtype='fp32', kv_dtype='fp16')
    model = parse_case(raw).model
    assert model.weight_dtype == model.quant.weight_format == model.quant.expert_weight_format == "fp16"
    assert model.kv_dtype == model.quant.kv_dtype == "fp16" and model.activation_dtype == model.compute_dtype == "fp32"
    explicit = quant_preset('bf16')
    raw['model']['quant'] = explicit.to_dict()
    assert parse_case(raw).model.quant == explicit

@pytest.mark.parametrize('name', ['w8a8_fp8', 'fp8_e4m3', 'custom_fp8_unblocked'])
def test_unregistered_fp8_format_still_requires_a_full_representation(name):
    raw = model_case(name)
    with pytest.raises(ModelingError, match='explicit_quant_block_and_scale_geometry'):
        parse_case(raw)
    raw['model']['quant'] = quant_preset('w8a8_mxfp8').to_dict()
    assert parse_case(raw).model.quant.weight_block == (1, 32)

def test_odd_width_packing_and_scale_tails_are_recomputed_after_tensor_sharding():
    quant = QuantSpec("declared-fp4", expert_weight_format="mxfp4", expert_block=32)
    term = WeightTerm("expert", 3, 65, "routed_expert", "expert_down")
    whole = ParallelSpec()
    shard = ParallelSpec(tp=2)
    assert term_payload_bytes(term, quant, whole) == 3 * 33
    assert term_bytes(term, quant, whole) == 3 * 33 + 3 * 3
    assert term_payload_bytes(term, quant, shard) == 3 * 17
    assert term_bytes(term, quant, shard) == 3 * 17 + 3 * 2
    assert term_bytes(term, quant, shard) * 2 > term_bytes(term, quant, whole)
    grouped_int4 = replace(quant, expert_weight_format="int4", scale_bytes=4)
    assert term_bytes(term, grouped_int4, whole) == 3 * 33 + 3 * 3 * 4
    channel_int4 = replace(grouped_int4, expert_block=1)
    assert term_bytes(term, channel_int4, whole) == 3 * 33 + 3 * 4

def test_fp8_block_scale_grid_drives_gemm_traffic_without_a_compression_multiplier():
    quant = QuantSpec("declared-fp8", weight_format="fp8_e4m3",
                      weight_block=(2, 32), scale_bytes=4)
    op = gemm_op(op_id="m", deps=(), stage=0, layer=0, requests=(), m=2, n=3, k=65,
                 quant=quant, category="attn", phase="prefill")
    assert dict(op.metadata)["packed_weight_bytes"] == 3 * 65 + 2 * 3 * 4
    assert quant.matrix_payload_bytes(3, 65, "attn") == 3 * 65
    assert op.flops == 2 * 2 * 3 * 65

def test_engram_source_and_runtime_payloads_decompose_resident_capacity():
    parsed = Case.from_mapping(memory_case())
    model = replace(
        parsed.model, layer_plans=tuple(replace(layer, hc_mult=1) for layer in parsed.model.layer_plans),
        engram=EngramSpec(tables=(EngramTableSpec(0, 0, 3, 65),), max_ngram=2, heads=1, width=65))
    for mode, payload, metadata in (
        ("device_bf16", 3 * 65 * 2, 0),
        ("device_int8_fp32", 3 * 65, 3 * 3 * 4),
        ("host_fp8_e8m0", 3 * 65, 3 * 3),
    ):
        quant = QuantSpec("declared-table", engram_table_mode=mode)
        selected = replace(model, quant=quant)
        included = stage_weight_decomposition(selected, parsed.runtime.parallel)[0]
        excluded = stage_weight_decomposition(
            selected, parsed.runtime.parallel, include_engram_tables=False)[0]
        assert included["payload_bytes"] - excluded["payload_bytes"] == payload
        assert included["quant_metadata_bytes"] - excluded["quant_metadata_bytes"] == metadata
        assert included["total_bytes"] == included["payload_bytes"] + included["quant_metadata_bytes"]
        assert included["total_bytes"] == stage_weight_bytes(selected, parsed.runtime.parallel)[0]
        assert Memory(replace(parsed, model=selected)).weights[0] == included["total_bytes"]
        assert quant.matrix_payload_bytes(3, 65, "engram_table", source=True) == 3 * 65
        assert quant.matrix_bytes(3, 65, "engram_table", source=True) == 3 * 65 + 3 * 3

def test_named_dtype_preset_controls_each_cache_and_weight_format():
    case = parse_case(compose_case(
        "gemma-4-31b", hardware=SPARSE_OFFLOAD_SKU, layers=4, tp=4, ep=1,
        quant="fp16", prompt_tokens=8, output_tokens=1,
        graph_capture_mode="none"))
    quant = case.model.quant
    assert quant.weight_format == quant.kv_dtype == quant.index_dtype == "fp16"
    graph = target_graphs(case, phase="decode", tokens=1, context=8, scope="model")
    assert graph[0]
