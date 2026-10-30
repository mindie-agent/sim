"""Ordinary MoE routing bindings to source alignment and combination plans."""
from collections import Counter
from dataclasses import replace

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.operator_analysis import operator_costs, target_operators
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


def _ids(tokens, topk, experts=4, start=0):
    return [[(token * topk + lane + start) % experts for lane in range(topk)]
            for token in range(tokens)]


def _counts(ids, experts):
    counts = [0] * experts
    for row in ids:
        for expert in row:
            counts[expert] += 1
    return counts


def _bindings(*, align=True, combine=True, block_size=16, capability=89,
              weight_dtype="fp32"):
    lowerings = {}
    if align:
        lowerings["route_pack"] = {
            "implementation": "cuda_moe_align_block_size",
            "options": {"block_size": block_size, "pad_sorted_ids": True},
        }
    if combine:
        options = {"weight_dtype": weight_dtype}
        if capability is not None:
            options["compute_capability"] = capability
        lowerings["route_unpack"] = {
            "implementation": "triton_moe_fused_mul_sum", "options": options,
        }
    return {"lowerings": lowerings}


def _case(*, tokens=4, topk=2, hidden=32, ids=None, expert_rows=None,
          source=True, align=True, combine=True, block_size=16, capability=89,
          weight_dtype="fp32", output_tokens=1):
    ids = ids or _ids(tokens, topk)
    expert_rows = expert_rows or _counts(ids, 4)
    decode_ids = [[0, 1]] if topk == 2 else _ids(1, topk)
    decode_rows = _counts(decode_ids, 4)
    raw = case(prompt=tokens, output=output_tokens)
    raw["model"].update(
        layers=1, ffn="moe", num_experts=4, top_k=topk,
        hidden_size=hidden, moe_intermediate_size=32,
    )
    prefill = {"scope": "body", "layer": 0, "phase": "prefill",
               "tokens": tokens, "expert_rows": expert_rows}
    if ids is not None:
        prefill["topk_ids"] = ids
    decode = {"scope": "body", "layer": 0, "phase": "decode",
              "tokens": 1, "expert_rows": decode_rows}
    if ids is not None:
        decode["topk_ids"] = decode_ids
    raw["runtime"]["routing"] = {"realizations": [prefill, decode]}
    if source:
        raw["runtime"]["kernels"] = _bindings(
            align=align, combine=combine, block_size=block_size,
            capability=capability, weight_dtype=weight_dtype)
    parsed = parse_case(raw)
    raw["model"] = replace(parsed.model, layer_plans=tuple(
        replace(plan, router_dtype="fp32") for plan in parsed.model.layer_plans)).to_dict()
    return raw


def _ops(raw, *, tokens=4, phase="prefill"):
    return target_operators(raw, phase=phase, tokens=tokens)


def _one(ops, **kwargs):
    selected = [op for op in ops
                if all(dict(op.metadata).get(key) == value
                       for key, value in kwargs.items())]
    assert len(selected) == 1, [(op.op_id, op.family, op.recipe) for op in selected]
    return selected[0]


def test_alignment_is_inserted_before_the_retained_payload_pack_copy():
    ids = [[0, 1], [0, 2], [1, 2], [0, 1]]
    source_ops = _ops(_case(ids=ids))
    analytic_ops = _ops(_case(ids=ids, source=False))
    align = _one(source_ops, semantic_layout="routing_indices_only")
    pack = _one(source_ops, alignment_op=align.op_id)
    analytic_pack = _one(analytic_ops, packed_layout="expert_grouped")
    assert align.family == "moe_align" and align.source_boundary == "call"
    assert align.deps == ("s0.l0.topk",)
    assert align.op_id in pack.deps
    # Alignment only emits sorted ids/expert blocks; it cannot replace the
    # permutation payload copy. The adapter scans padded source indices and
    # compacts only valid entries, preserving the original payload writes.
    assert align.local_shape == (4, 2)
    assert dict(align.metadata)["analysis"]["counts"] == (3, 3, 2, 0)
    assert dict(align.metadata)["analysis"]["path"] == "small_batch_expert"
    contract = dict(pack.metadata)["pack_contract"]
    source_index_read = 48 * 4 + 3 * 4 + 4
    assert pack.mte2_bytes == analytic_pack.mte2_bytes - 4 * 2 * 4 + source_index_read
    assert pack.mte3_bytes == analytic_pack.mte3_bytes
    assert pack.hbm_bytes == pack.mte2_bytes + pack.mte3_bytes
    assert pack.vector_ops == analytic_pack.vector_ops + 48
    assert contract["compact_rows"] == 8
    assert contract["sentinel_slots"] == 40
    assert contract["padded_payload_reads_or_writes"] is False
    assert pack.hbm_bytes > 0 and pack.recipe == "route_pack"
    parsed = parse_case(_case())
    rows = operator_costs(parsed, (align,))["operators"]
    kernels = [row for row in rows if row["operator"]["family"] == "moe_align"]
    assert len(kernels) == 1
    assert kernels[0]["operator"]["source_boundary"] == "kernel"
    assert kernels[0]["operator"]["recipe"] == "cuda_moe_align_block_size"


def test_large_alignment_expands_to_two_ordered_source_kernels():
    tokens = 512
    ids = _ids(tokens, 2)
    raw = _case(tokens=tokens, ids=ids, source=True)
    graph = target_graphs(raw, phase="prefill", tokens=tokens, scope="body")
    report = graph_costs(raw, graph, external_inputs={"analysis_input": 0})
    align_rows = [row for row in report["timeline"]
                  if row["op_id"].endswith("pack.align/align")]
    sort_rows = [row for row in report["timeline"]
                 if row["op_id"].endswith("pack.align/sort")]
    assert len(align_rows) == len(sort_rows) == 1
    assert align_rows[0]["source_boundary"] == sort_rows[0]["source_boundary"] == "kernel"
    assert align_rows[0]["id"] in sort_rows[0]["predecessors"]
    align = _one(_ops(raw, tokens=tokens),
                 semantic_layout="routing_indices_only")
    assert dict(align.metadata)["analysis"]["path"] == "two_kernel_sort"
    assert dict(align.metadata)["dispatch"]["kernel_phases"] == (
        "large_align", "large_sort")


def test_same_histogram_keeps_distinct_explicit_token_mappings():
    first = _ids(8, 2)
    second = [[0, 1], [0, 1], [2, 3], [2, 3],
              [0, 2], [0, 2], [1, 3], [1, 3]]
    assert tuple(_counts(first, 4)) == tuple(_counts(second, 4)) == (4, 4, 4, 4)
    for ids, expected in ((first, (4, 4, 4, 4)), (second, (4, 4, 4, 4))):
        raw = _case(tokens=8, ids=ids)
        align = _one(_ops(raw, tokens=8), semantic_layout="routing_indices_only")
        realization = dict(align.metadata)["routing_realization"]
        assert tuple(tuple(row) for row in realization["topk_ids"]) == tuple(
            tuple(row) for row in ids)
        assert tuple(dict(align.metadata)["analysis"]["counts"]) == expected


def test_zero_experts_remain_explicit_and_do_not_gain_blocks():
    ids = [[0, 2], [0, 2]]
    raw = _case(tokens=2, ids=ids, expert_rows=[2, 0, 2, 0])
    assert dict(raw["runtime"]["routing"]["realizations"][0])["expert_rows"] == [2, 0, 2, 0]
    align = _one(_ops(raw, tokens=2), semantic_layout="routing_indices_only")
    analysis = dict(align.metadata)["analysis"]
    assert analysis["counts"] == (2, 0, 2, 0)
    assert analysis["padded_per_expert"] == (16, 0, 16, 0)
    from mindie_sim.prediction.operators import moe_align
    blocks = list(moe_align.iter_blocks(align))
    assert [block["expert"] for block in blocks] == [0, 2]
    assert all(block["blocks"] == 1 for block in blocks)
    pack = _one(_ops(raw, tokens=2), alignment_op=align.op_id)
    contract = dict(pack.metadata)["pack_contract"]
    assert contract["sorted_index_bytes"] == 32 * 4
    assert contract["expert_index_bytes"] == 2 * 4
    assert contract["compact_rows"] == 4
    assert contract["sentinel_slots"] == 28
    assert pack.mte3_bytes == 4 * (32 * 2 + 4)


def test_source_alignment_fails_visibly_without_explicit_ids():
    raw = _case()
    del raw["runtime"]["routing"]["realizations"][0]["topk_ids"]
    del raw["runtime"]["routing"]["realizations"][1]["topk_ids"]
    with pytest.raises(ModelingError, match="explicit_topk_ids"):
        _ops(raw)
    raw = _case()
    raw["runtime"]["kernels"] = _bindings(align=False)
    del raw["runtime"]["routing"]["realizations"][0]["topk_ids"]
    del raw["runtime"]["routing"]["realizations"][1]["topk_ids"]
    with pytest.raises(ModelingError, match="explicit_topk_ids"):
        _ops(raw)


def test_explicit_ids_must_agree_with_the_declared_histogram():
    raw = _case(tokens=2, ids=[[0, 1], [1, 2]], expert_rows=[1, 1, 1, 1])
    with pytest.raises(ModelingError, match="must_agree_with_expert_rows"):
        _ops(raw, tokens=2)
    raw = _case(tokens=2, ids=[[0, 0], [1, 2]], expert_rows=[2, 1, 1, 0])
    with pytest.raises(ModelingError, match="duplicate_expert"):
        _ops(raw, tokens=2)
    raw = _case(tokens=2, ids=[[0, 1], [1, 2]], expert_rows=[1, 2, 1, 0])
    raw["runtime"]["routing"]["realizations"][0]["topk_ids"] = [[0], [1, 2]]
    with pytest.raises(ModelingError, match="match_top_k"):
        _ops(raw, tokens=2)


def test_combine_uses_the_materialized_source_contract_and_real_tail_tiles():
    ids = _ids(5, 3)
    raw = _case(tokens=5, topk=3, hidden=130, ids=ids,
                expert_rows=_counts(ids, 4))
    ops = _ops(raw, tokens=5)
    materialize = _one(ops, output_layout="tokens_topk_hidden")
    combine = _one(ops, combine_input_layout="tokens_topk_hidden")
    assert materialize.recipe == "route_unpack.materialize"
    assert materialize.local_shape == (5, 3, 130)
    assert materialize.deps == ("s0.l0.expert_down", "s0.l0.pack", "s0.l0.topk")
    assert dict(materialize.metadata)["inverse_index_producer"] == "s0.l0.pack"
    assert dict(materialize.metadata)["mapping_basis"] == "pack_produced_inverse_token_slot"
    assert combine.family == "moe_combine" and combine.source_boundary == "kernel"
    assert combine.deps == (materialize.op_id, "s0.l0.topk")
    metadata = dict(combine.metadata)
    assert metadata["grid"] == (1, 3)
    assert metadata["block_m"] == 2 and metadata["block_k"] == 256
    assert metadata["last_tile_rows"] == 1
    assert metadata["last_tile_cols"] == 130
    assert metadata["output_write_bytes"] == 5 * 130 * 2
    parsed = parse_case(raw)
    rows = operator_costs(parsed, (combine,))["operators"]
    assert [row["operator"]["family"] for row in rows] == ["moe_combine"]
    vector = next(value for key, value in rows[0]["declared_service"].items()
                  if key.startswith("vector"))
    assert vector == metadata["vector_ops_per_program"] * 3


@pytest.mark.parametrize("capability,weight_dtype,message", [
    (None, "fp32", "compute_capability"),
    (90, "fp32", "compute_capability_89"),
    (89, "bf16", "weight_dtype_must_match"),
])
def test_combine_requires_the_declared_capability_and_router_weight_dtype(
        capability, weight_dtype, message):
    raw = _case(combine=True, capability=capability, weight_dtype=weight_dtype)
    with pytest.raises(ModelingError, match=message):
        _ops(raw)


def test_public_graph_and_predict_execute_source_kernels_not_envelopes():
    raw = _case(tokens=4, output_tokens=2)
    graphs = target_graphs(raw, phase="prefill", tokens=4, scope="model")
    report = graph_costs(raw, graphs)
    ids = {row["op_id"] for row in report["timeline"]}
    assert "s0.l0.pack.align/align" in ids
    assert "s0.l0.unpack" in ids
    assert not any(op_id.endswith("pack.align/align/align") for op_id in ids)
    combine = next(row for row in report["timeline"]
                   if row["op_id"] == "s0.l0.unpack")
    assert combine["family"] == "moe_combine"
    assert combine["source_boundary"] == "kernel"
    assert report["duration_ms"] > 0
    result = predict(raw)
    assert result["status"] == "modeled"
    assert result["requests"][0]["generated_tokens"] == 2
    assert result["metrics"]["makespan_ms"] > 0


def test_hash_routing_retains_logits_and_returns_weights_and_ids():
    parsed = parse_case(_case())
    layer = replace(parsed.model.layer_plans[0], router_mode="hash")
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    graph = target_graphs(parsed, phase="prefill", tokens=4, scope="model")
    source = next(op for op in graph[0] if op.recipe == "hash_routing")
    router = next(op for op in graph[0] if op.family == "router")
    align = next(op for op in graph[0] if op.family == "moe_align")
    combine = next(op for op in graph[0] if op.family == "moe_combine")
    assert source.deps == (router.op_id,)
    assert router.flops == 2 * 4 * 4 * 32
    assert source.mte3_bytes == 4 * 2 * (4 + 4)
    assert source.mte2_bytes == 4 * 4 * 4 + 4 * 8 + 4 * 2 * 4
    assert dict(combine.metadata)["topk_weights_producer"] == source.op_id
    assert align.deps == (source.op_id,)
    assert graph_costs(parsed, graph)["execution"]["live_tasks"] == 0


def test_normalized_weights_preserve_the_distinct_id_producer():
    parsed = parse_case(_case())
    layer = replace(parsed.model.layer_plans[0], router_norm_topk=True)
    parsed = replace(parsed, model=replace(parsed.model, layer_plans=(layer,)))
    graph = target_graphs(parsed, phase="prefill", tokens=4, scope="model")
    align = next(op for op in graph[0] if op.family == "moe_align")
    combine = next(op for op in graph[0] if op.family == "moe_combine")
    assert align.deps == ("s0.l0.topk",)
    assert dict(combine.metadata)["topk_weights_producer"] == "s0.l0.route_normalize"
    assert "s0.l0.route_normalize" in combine.deps
    assert graph_costs(parsed, graph)["execution"]["live_tasks"] == 0
