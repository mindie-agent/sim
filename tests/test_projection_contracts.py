"""Real Flash projection contracts and generic mixed-precision counterexamples.

[Role] Verify packed GEMM geometry, native precision and capacity together.
[Invariants] Tests use tensor/source contracts, never measured timings as input.
[Timeouts & Locks] In-memory layer graphs and finite Engine execution.
[Side-Effect Policy] No network, devices or filesystem writes.
"""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, target_graphs
from mindie_sim.catalog.quant import projection_quant
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.projections import kda_projections
from mindie_sim.prediction.structure import parse_case


BF16 = {"weight_format": "bf16", "activation_format": "bf16"}
INT8 = {"weight_format": "int8", "activation_format": "int8"}
FP32 = {"weight_format": "fp32", "activation_format": "fp32"}


def case(overrides=None):
    raw = {
        "model": {"catalog_key": "glm-5.3-flash", "quantization": "w8a8", "layers": 4},
        "machine": {"backend": "ascend", "sku": "Ascend910_9362", "num_devices": 16},
        "runtime": {"tp": 16, "ep": 16, "kernels": {
            "programs": {"matmul": {"kind": "steady_matmul"}}}},
        "workload": {"requests": [{"id": "r", "prompt_tokens": 1, "output_tokens": 1}]},
    }
    parsed = parse_case(raw)
    if overrides:
        raw["model"]["quant"] = replace(parsed.model.quant, category_overrides=overrides).to_dict()
        parsed = parse_case(raw)
    return parsed


def kda_ops(c, *, packed=None, dk=128, dv=128):
    plan = replace(c.model.layer_plans[0], linear_key_head_dim=dk, linear_head_dim=dv)
    if packed is not None:
        plan = replace(plan, packed_input_projection=packed)
    builder = Builder("kda", plan, 0, [], c.model, "decode")
    result = kda_projections(builder, plan=plan, tokens=24, hidden=4096,
                            key_width=4 * dk, value_width=4 * dv,
                            heads=4, dk=dk, deps="analysis_input")
    return plan, builder.ops, result


def test_packed_native_kda_matches_retained_24x1796_and_two_low_rank_matrices():
    c = case()
    plan, ops, (sources, gate, beta, decay) = kda_ops(c)
    first = ops[0]
    assert first.local_shape == (24, 1796, 4096)
    assert first.dtype == "bf16"
    assert [op.local_shape for op in ops[1:]] == [(24, 512, 128)] * 2
    assert sources == [first.op_id] * 3 and beta == first.op_id
    assert gate == ops[1].op_id and decay == ops[2].op_id
    assert ops[1].deps == ops[2].deps == (first.op_id,)
    slices = dict(first.metadata)["packed_projection_slices"]
    assert [row["offset_elements"] for row in slices] == [0, 512, 1024, 1536, 1540, 1668]
    assert [row["shape"][1] for row in slices] == [512, 512, 512, 4, 128, 128]
    assert all(row["strides"] == (1796, 1) for row in slices)
    _, separate, _ = kda_ops(c, packed=False)
    assert sum(op.flops for op in ops) == sum(op.flops for op in separate)
    assert sum(op.mte3_bytes for op in ops) == sum(op.mte3_bytes for op in separate)
    assert sum(op.mte2_bytes for op in separate) - sum(op.mte2_bytes for op in ops) == 5 * 24 * 4096 * 2
    terms = {term.name: term for term in layer_weight_terms(plan, c.model)}
    first_terms = ["kda_q", "kda_k", "kda_v", "kda_b", "kda_f_a", "kda_g_a"]
    assert sum(term_bytes(terms[name], c.model.quant, c.runtime.parallel)
               for name in first_terms) == dict(first.metadata)["packed_weight_bytes"]
    assert term_bytes(terms["kda_g_b"], c.model.quant, c.runtime.parallel) == 512 * 128 * 2
    assert "kda_g" not in terms
    report = graph_costs(c, {0: tuple(ops)}, input_ready_ms=5,
                         external_inputs={"analysis_input": 5}, isolated=False)
    assert report["duration_ms"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert all(row["ready_ms"] >= 5 for row in report["timeline"])


def test_key_value_widths_and_direct_gate_remain_independent():
    c = case()
    plan, ops, _ = kda_ops(c, dk=64, dv=128)
    by_recipe = {op.recipe: op for op in ops}
    assert by_recipe["linear_decay_b"].local_shape == (24, 256, 64)
    assert by_recipe["linear_gate_b"].local_shape == (24, 512, 128)
    direct = replace(plan, packed_input_projection=False, gate_rank=0, projection_dtype="")
    b = Builder("direct", direct, 0, [], c.model, "decode")
    kda_projections(b, plan=direct, tokens=24, hidden=4096, key_width=256,
                    value_width=512, heads=4, dk=64, deps="analysis_input")
    assert next(op for op in b.ops if op.recipe == "linear_gate").local_shape == (24, 512, 4096)
    assert not any(op.recipe == "linear_gate_b" for op in b.ops)
    terms = {term.name: term for term in layer_weight_terms(direct, c.model)}
    assert terms["kda_f_b"].rows == 64 * 64
    assert terms["kda_g"].cols == 4096 and "kda_g_a" not in terms


def test_explicit_parent_then_child_precision_wins_and_incompatible_pack_splits():
    c = case({"kda": INT8, "kda.q": BF16})
    plan, ops, _ = kda_ops(c)
    by_recipe = {op.recipe: op for op in ops}
    assert "linear_qkvbfg_a" not in by_recipe
    assert by_recipe["linear_q"].dtype == "bf16"
    assert by_recipe["linear_k"].dtype == by_recipe["linear_gate_b"].dtype == "int8"
    assert dict(by_recipe["linear_q"].metadata)["projection_packing"] == "separate_explicit_operand_formats"
    terms = {term.name: term for term in layer_weight_terms(plan, c.model)}
    for op, term in ((by_recipe["linear_q"], terms["kda_q"]),
                     (by_recipe["linear_k"], terms["kda_k"])):
        assert dict(op.metadata)["packed_weight_bytes"] == term_bytes(term, c.model.quant, c.runtime.parallel)
    # Native child defaults cannot defeat an explicit parent declaration.
    c2 = case({"indexer": INT8, "indexer.weights": FP32})
    q = projection_quant(c2.model.quant, c2.model.layer_plans[3])
    assert q.format_for("indexer.weights") == "fp32"
    parent = projection_quant(case({"indexer": INT8}).model.quant, c2.model.layer_plans[3])
    assert parent.format_for("indexer.weights") == "int8"


@pytest.mark.parametrize("overrides,packed", [(None, True), ({"mla.q_a": INT8}, False)])
def test_mla_ordinary_frontend_preserves_packing_precision_and_branch_dependencies(overrides, packed):
    c = case(overrides)
    ops = [op for op in target_graphs(c, phase="decode", tokens=6, sequences=4,
                                     context=2048, scope="body")[0] if op.layer == 3]
    by_recipe = {op.recipe: op for op in ops}
    if packed:
        projection = by_recipe["qkv_a"]
        assert projection.local_shape == (24, 2048, 4096) and projection.dtype == "bf16"
        assert by_recipe["q_norm"].deps == by_recipe["kv_norm"].deps == (projection.op_id,)
        assert "q_a" not in by_recipe and "kv_a" not in by_recipe
    else:
        assert "qkv_a" not in by_recipe
        assert by_recipe["q_a"].dtype == "int8" and by_recipe["kv_a"].dtype == "bf16"
        assert by_recipe["q_a"].deps[0].endswith("quantize")
    assert by_recipe["q_b"].deps == (by_recipe["q_norm"].op_id,)


def test_observed_fp32_index_weight_contract_has_input_cast_and_matching_capacity():
    c = case({"indexer.weights": FP32})
    by_recipe = {op.recipe: op for op in target_graphs(
        c, phase="decode", tokens=6, sequences=4, context=2048, scope="body")[0] if op.layer == 3}
    cast, matrix = by_recipe["index_weights_input_cast"], by_recipe["index_weights"]
    assert matrix.local_shape == (24, 32, 4096) and matrix.dtype == "fp32"
    assert matrix.deps == (cast.op_id,)
    assert cast.mte2_bytes == 24 * 4096 * 2 and cast.mte3_bytes == 24 * 4096 * 4
    terms = {term.name: term for term in layer_weight_terms(c.model.layer_plans[3], c.model)}
    assert term_bytes(terms["indexer_weights"], c.model.quant, c.runtime.parallel) == 32 * 4096 * 4


def test_explicit_mla_source_caller_keeps_its_own_projection_boundaries():
    c = case()
    runtime = replace(c.runtime, kernels={
        **c.runtime.kernels,
        "lowerings": {"mla_prolog": "vllm_ascend_mla_prolog_v3"}})
    c = replace(c, runtime=runtime)
    ops = [op for op in target_graphs(c, phase="decode", tokens=6,
                                     sequences=4, context=2048, scope="body")[0] if op.layer == 3]
    recipes = {op.recipe for op in ops}
    assert "q_a" in recipes and "kv_a" in recipes and "qkv_a" not in recipes
    qa = next(op for op in ops if op.recipe == "q_a")
    assert qa.dtype == "bf16"


def test_packed_kda_consumer_copies_have_source_offsets_and_real_dependencies():
    c = case()
    runtime = replace(c.runtime, tensor_layouts={
        'short_convolution': {'input': 'contiguous'},
        'kda_recurrent': {'beta': 'contiguous'}})
    c = replace(c, runtime=runtime)
    ops = [op for op in target_graphs(c, phase='decode', tokens=4, sequences=4,
                                     context=2048, scope='body')[0] if op.layer == 0]
    by_recipe = {op.recipe: op for op in ops}
    packed = by_recipe['linear_qkvbfg_a']
    qkv = by_recipe['convolution_input_contiguous']
    beta = by_recipe['kda_beta_contiguous']
    assert packed.local_shape == (16, 1796, 4096)
    assert qkv.local_shape == (16, 1536)
    assert beta.local_shape == (16, 4)
    assert qkv.mte2_bytes == qkv.mte3_bytes == 16 * 1536 * 2
    assert beta.mte2_bytes == beta.mte3_bytes == 16 * 4 * 2
    assert dict(qkv.metadata)['slice_offsets'] == (0, 0)
    assert dict(beta.metadata)['slice_offsets'] == (0, 1536)
    assert dict(beta.metadata)['packed_input_shape'] == (16, 1796)
    assert qkv.deps == beta.deps == (packed.op_id,)
    assert by_recipe['short_convolution'].deps == (qkv.op_id,)
    assert by_recipe['linear_beta_sigmoid'].deps == (beta.op_id,)
    report = graph_costs(c, {0: tuple(ops)}, input_ready_ms=7,
                         external_inputs={d: 7 for op in ops for d in op.deps
                                          if d not in {node.op_id for node in ops}}, isolated=False)
    assert report['execution']['live_tasks'] == report['execution']['live_programs'] == 0
    assert min(row['ready_ms'] for row in report['timeline']) >= 7


def test_strided_consumers_keep_views_and_separate_projections_do_not_gain_fake_slices():
    c = case()
    old = target_graphs(c, phase='decode', tokens=4, sequences=4, context=2048, scope='body')[0]
    assert not any(op.recipe in {'kda_beta_contiguous', 'convolution_input_contiguous'} for op in old)
    c = case({'kda': INT8, 'kda.q': BF16})
    c = replace(c, runtime=replace(c.runtime, tensor_layouts={
        'short_convolution': {'input': 'contiguous'}, 'kda_recurrent': {'beta': 'contiguous'}}))
    separate = target_graphs(c, phase='decode', tokens=4, sequences=4, context=2048, scope='body')[0]
    assert not any(op.recipe in {'kda_beta_contiguous', 'convolution_input_contiguous'} for op in separate)
