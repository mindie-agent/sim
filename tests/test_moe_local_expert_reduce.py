"""Local-expert reduce caller: source contracts, work and reduction boundaries."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs, target_operators
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.balanced_routing import BalancedRouting
from mindie_sim.prediction.family_grouped_gemm import lower_op
from mindie_sim.prediction.feedforward import MOE_RECIPES
from mindie_sim.prediction.moe_callers import (
    LOCAL_EXPERT_REDUCE, LocalBankRouting, local_expert_reduce_recipe)
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError, dtype_bytes
from test_balanced_experts import balanced_case
from test_prediction_system import case

ROOT = Path(__file__).resolve().parents[1]
A5_FIXTURE = ROOT / "tests/fixtures/moe-a5-q30-e16k4-tp4ep4-structure.json"
# The compact fixture is the raw structural subset of this existing capture case.
A5_SOURCE = {
    "path": "/tmp/mindie-modeling-systematic-20261106/systemic2/a5_q30_e16k4_4k_raw.json",
    "sha256": "0d262d62a9b9db96455b98488021b0bd1b86d864a8c9f40e9baa75daf9144fb0",
    "derivation": "A5 Qwen3.5 Q30 L8 E16 K4 tp4 ep4 dp1 prefill 4k structural fields",
}


def local_case(*, shared=0, gate=False, tokens=32, top_k=2):
    raw = balanced_case(tp=4, ep=4, experts=8, tokens=tokens)
    raw["model"].update(num_shared_experts=shared, top_k=top_k)
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    raw = parse_case(raw).to_dict()
    if gate:
        raw["model"]["layer_plans"][0]["shared_expert_gate"] = True
    return raw


def expected_case(*, top_k=1, tokens=8):
    raw = case(tp=4, ep=4, prompt=tokens, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=top_k,
                        moe_intermediate_size=64,
                        expert_routing=[.5, .5, 0, 0, 0, 0, 0, 0])
    raw["runtime"].update(moe_recipe=LOCAL_EXPERT_REDUCE,
                          max_num_batched_tokens=max(16, tokens))
    return raw


def source_bank_case(counts, *, tokens=8):
    raw = case(tp=4, ep=4, prompt=tokens, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=2,
                        moe_intermediate_size=64)
    raw["runtime"].update(
        moe_recipe=LOCAL_EXPERT_REDUCE, max_num_batched_tokens=16,
        routing={"realizations": [{
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": tokens,
            "expert_rows": list(counts)}]},
        kernels={"lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}})
    return raw


def by_id(raw, *, tokens=32):
    return {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=tokens)}


def work(op):
    return (op.flops, op.vector_ops, op.mte2_bytes, op.mte3_bytes, op.link_bytes)


def test_replicated_source_routes_full_tokens_and_builds_no_phantom_exchange():
    ops = target_operators(local_case(), phase="prefill", tokens=32)
    graph = {op.op_id: op for op in ops}
    # A dp=1 source owns the complete token selection; no TP token slice occurs.
    assert graph["s0.l0.router"].local_shape[0] == 32
    assert graph["s0.l0.topk"].local_shape == (32, 8)
    assert graph["s0.l0.pack"].local_shape == (32, 2)
    assert not any(op.family in {"alltoall", "alltoallv"} for op in ops)
    assert not any(op.recipe in {"expert_dispatch", "expert_combine"} for op in ops)
    assert not any("histogram_gather" in op.op_id or "alltoall_tp_gather" in op.op_id
                   for op in ops)
    reductions = [op for op in ops if op.recipe == "moe_output_reduce"]
    assert len(reductions) == 1
    reduction = reductions[0]
    assert reduction.deps == ("s0.l0.unpack",)
    assert reduction.local_shape == (32, 32)
    assert reduction.link_bytes == 2 * 3 / 4 * 32 * 32 * 2
    assert dict(reduction.metadata)["reduction_scope"] == "routed_and_shared_joined_output"

    pack = graph["s0.l0.pack"]
    metadata = dict(pack.metadata)
    assert metadata["moe_caller"] == LOCAL_EXPERT_REDUCE
    assert metadata["source_partitions"] == 1
    assert metadata["exchange"] == "none"
    assert metadata["allocated_routing_rows"] == 64
    assert metadata["valid_local_rows"] == 16
    assert metadata["local_expert_rows"] == (8, 8)
    assert metadata["histogram_counts"] == (8, 8)
    assert metadata["routing_realization"]["source_partitions"] == 1
    weight_bytes = dtype_bytes(dict(graph["s0.l0.router"].metadata)["output_format"])
    act_bytes = dtype_bytes("bf16")
    allocation, valid, bank, latent, tokens = 64, 16, 2, 32, 32
    assert pack.mte2_bytes == (tokens * latent * act_bytes
                               + allocation * (8 + weight_bytes) + bank * 4)
    assert pack.mte3_bytes == (valid * latent * act_bytes + allocation * 4
                               + allocation * weight_bytes + bank * 12)
    assert pack.vector_ops == 2 * allocation + bank
    assert pack.mte3_bytes < allocation * latent * act_bytes  # no full payload write
    unpack = graph["s0.l0.unpack"]
    assert unpack.mte2_bytes == valid * latent * act_bytes + allocation * (4 + weight_bytes)
    assert unpack.mte3_bytes == tokens * latent * act_bytes
    assert unpack.vector_ops == 2 * tokens * latent * 2 + allocation
    assert unpack.mte2_bytes < allocation * latent * act_bytes  # no undefined payload read
    # Valid local rows drive expert matrices; the eager allocation drives SwiGlu.
    assert graph["s0.l0.expert_up"].local_shape[0] == 16
    assert graph["s0.l0.expert_down"].local_shape[0] == 16
    assert graph["s0.l0.expert_act"].local_shape[0] == 64
    assert graph["s0.l0.expert_up"].flops == 2 * 16 * 2 * 64 * 32
    assert graph["s0.l0.expert_down"].flops == 2 * 16 * 32 * 64
    assert graph["s0.l0.expert_act"].vector_ops == 4 * 64 * 64
    assert graph["s0.l0.expert_act"].mte2_bytes == 2 * 64 * 64 * 2
    assert graph["s0.l0.expert_act"].mte3_bytes == 64 * 64 * 2
    assert not any(".shared.tp" in op.op_id or op.recipe == "expert_tensor_reduce"
                   for op in ops)


@pytest.mark.parametrize(("counts", "local_rows", "physical_ids"), (
    ((4, 8, 4, 0, 0, 0, 0, 0), (4, 8), (0, 1)),
    ((0, 0, 4, 8, 4, 0, 0, 0), (4, 8), (2, 3)),
    ((8, 0, 8, 0, 0, 0, 0, 0), (8, 0), (0, 1)),
    ((0, 0, 0, 0, 8, 0, 8, 0), (8, 0), (4, 5)),
))
def test_source_grouped_lowering_uses_each_local_contiguous_bank(counts, local_rows, physical_ids):
    raw = source_bank_case(counts)
    graph = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=8)}
    assert graph["s0.l0.expert_act"].local_shape[0] == 16
    for op_id, n, k in (("s0.l0.expert_up", 128, 32),
                        ("s0.l0.expert_down", 32, 64)):
        op = graph[op_id]
        metadata = dict(op.metadata)
        assert metadata["local_expert_rows"] == local_rows
        assert metadata["expert_rows"] == local_rows
        assert metadata["group_ids"] == (0, 1)
        assert metadata["local_weight_expert_ids"] == (0, 1)
        assert metadata["representative_expert_ids"] == physical_ids
        assert metadata["global_expert_rows"] == counts
        assert metadata["routing_rows_basis"] == "contiguous_local_bank"
        assert tuple(metadata["semantic_metadata"]["routing_realization"]["expert_rows"]) == counts
        parent = metadata["semantic_metadata"]["routing_realization"]
        assert tuple(parent["local_expert_rows"]) == local_rows
        row = operator_costs(raw, [op])["operators"][0]["operator"]
        rows = sum(local_rows)
        assert tuple(row["local_shape"]) == (rows, n, k)
        assert row["flops"] == 2 * rows * n * k
        n_tiles = (n + 255) // 256
        expected_mte2 = (rows * k * 2 * n_tiles
                         + sum((value + 127) // 128 for value in local_rows) * k * n * 2)
        assert row["mte2_bytes"] == expected_mte2
        assert row["mte3_bytes"] == rows * n * 2
        assert row["hbm_bytes"] == expected_mte2 + rows * n * 2
    assert dict(graph["s0.l0.expert_act"].metadata)["activation_rows"] == 16


def test_source_grouped_lowering_accepts_forced_balanced_contiguous_bank():
    raw = local_case()
    raw["runtime"]["kernels"] = {
        "lowerings": {"grouped_gemm": "ascendc_grouped_nd_bf16"}}
    graph = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=32)}
    metadata = dict(graph["s0.l0.expert_up"].metadata)
    assert metadata["routing_basis"] == "forced_balanced_integer_counts"
    assert metadata["expert_rows"] == (8, 8)
    assert metadata["local_expert_rows"] == (8, 8)
    assert metadata["representative_expert_ids"] == (0, 1)
    assert len(metadata["global_expert_rows"]) == 8
    assert sum(metadata["global_expert_rows"]) == 64


def test_source_grouped_lowering_rejects_empty_local_bank_without_global_substitution():
    base = by_id(local_case())["s0.l0.expert_up"]
    realization = LocalBankRouting(
        expert_rows=(0,) * 8, local_expert_rows=(0, 0),
        rank_rows=(0, 0, 0, 0), representative_rank=0,
        basis="declared_integer_counts", ep=4, parent={})
    metadata = dict(base.metadata)
    metadata.update(routing_basis="declared_integer_counts",
                    routing_realization=realization.to_dict())
    op = replace(base, local_shape=(0, base.local_shape[1], base.local_shape[2]),
                 flops=0.0, metadata=tuple(metadata.items()))
    with pytest.raises(ModelingError, match="local_bank_has_no_valid_rows"):
        lower_op(op)


def test_expected_routing_projects_the_native_contiguous_bank_not_strided_ranks():
    graph = target_operators(expected_case(), phase="prefill", tokens=8)
    by_name = {op.op_id: op for op in graph}
    metadata = dict(by_name["s0.l0.pack"].metadata)
    assert metadata["routing_basis"] == "probability_moments"
    assert metadata["histogram_counts"] is None
    assert metadata["expected_expert_rows"] == pytest.approx((4, 4, 0, 0, 0, 0, 0, 0))
    assert metadata["rank_assignment_rows"] == pytest.approx((8, 0, 0, 0))
    assert metadata["local_expert_rows"] == pytest.approx((4, 4))
    assert metadata["valid_local_rows"] == pytest.approx(8)
    assert 1 < metadata["expected_active_experts"] < 2
    assert by_name["s0.l0.expert_up"].local_shape[0] == 8
    assert by_name["s0.l0.expert_act"].local_shape[0] == 8


def test_forced_balance_tail_uses_contiguous_bank_rows():
    raw = balanced_case(tp=4, ep=4, experts=8, tokens=1)
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    graph = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=1)}
    metadata = dict(graph["s0.l0.pack"].metadata)
    assert metadata["routing_basis"] == "forced_balanced_integer_counts"
    assert metadata["global_expert_rows"] == (1, 1, 0, 0, 0, 0, 0, 0)
    assert metadata["rank_assignment_rows"] == (2, 0, 0, 0)
    assert metadata["local_expert_rows"] == (1, 1)
    assert metadata["histogram_counts"] == (1, 1)
    assert metadata["valid_local_rows"] == 2
    assert metadata["allocated_routing_rows"] == 2
    assert graph["s0.l0.expert_up"].local_shape[0] == 2
    assert graph["s0.l0.expert_act"].local_shape[0] == 2


def test_explicit_nonuniform_histogram_preserves_zero_bank_and_ids():
    raw = case(tp=4, ep=4, prompt=8, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=2,
                        moe_intermediate_size=64)
    raw["runtime"].update(
        moe_recipe=LOCAL_EXPERT_REDUCE, max_num_batched_tokens=16,
        routing={"realizations": [{
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": 8,
            "expert_rows": [8, 0, 8, 0, 0, 0, 0, 0],
            "topk_ids": [[0, 2]] * 8}]})
    graph = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=8)}
    metadata = dict(graph["s0.l0.pack"].metadata)
    assert metadata["routing_basis"] == "declared_integer_counts"
    assert metadata["global_expert_rows"] == (8, 0, 8, 0, 0, 0, 0, 0)
    assert metadata["rank_assignment_rows"] == (8, 8, 0, 0)
    assert metadata["local_expert_rows"] == (8, 0)
    assert metadata["histogram_counts"] == (8, 0)
    assert metadata["valid_local_rows"] == 8
    assert metadata["allocated_routing_rows"] == 16
    assert metadata["representative_expert_ids"] == (0, 1)
    realization = metadata["routing_realization"]
    assert realization["token_identities"] == "declared"
    assert realization["topk_ids"] == ((0, 2),) * 8
    assert graph["s0.l0.expert_up"].local_shape[0] == 8
    assert graph["s0.l0.expert_act"].local_shape[0] == 16


def test_explicit_local_bank_histogram_projects_contiguous_rows_and_capacity():
    case_obj = parse_case(local_case())
    plan, parallel = case_obj.model.layer_plans[0], case_obj.runtime.parallel
    realization = BalancedRouting(
        expert_rows=(12, 8, 8, 8, 4, 8, 8, 8),
        local_expert_rows=(12, 4), ep=4, source_partitions=4,
        global_tokens=32, valid_local_tokens=32, padded_local_tokens=32,
        representative_rank=0, represented_ranks=(0,))
    ops, last = local_expert_reduce_recipe(
        prefix="golden", plan=plan, deps="input", stage=0, requests=("r",),
        tokens=32, model=case_obj.model, parallel=parallel, mode="prefill",
        runtime=case_obj.runtime, global_tokens=None, routing_realization=realization,
        chunks=4)
    graph = {op.op_id: op for op in ops}
    metadata = dict(graph["golden.expert_up"].metadata)
    assert metadata["routing_basis"] == "forced_balanced_integer_counts"
    assert metadata["global_expert_rows"] == (12, 8, 8, 8, 4, 8, 8, 8)
    assert metadata["rank_assignment_rows"] == (20, 16, 12, 16)
    assert metadata["local_expert_rows"] == (12, 8)
    assert metadata["valid_local_rows"] == 20
    assert metadata["histogram_counts"] == (12, 8)
    assert metadata["matrix_group_rows"] == (12, 8)
    assert metadata["matrix_mapping_domain"] == "integer_expert_groups"
    assert metadata["allocated_routing_rows"] == 64
    assert metadata["routing_realization"]["source_partitions"] == 1
    assert metadata["routing_realization"]["local_expert_rows"] == (12, 8)
    assert graph["golden.expert_up"].local_shape[0] == 20
    assert graph["golden.expert_act"].local_shape[0] == 64
    assert graph["golden.expert_down"].local_shape[0] == 20
    assert last == "golden.moe_output_reduce"


def test_expected_routing_stays_fractional_without_an_invented_histogram():
    raw = local_case()
    raw["runtime"].pop("routing")
    graph = by_id(raw)
    metadata = dict(graph["s0.l0.pack"].metadata)
    assert metadata["routing_basis"] == "probability_moments"
    assert metadata["histogram_counts"] is None
    assert metadata["expected_rows"] == pytest.approx(16)
    assert metadata["valid_local_rows"] == pytest.approx(16)
    assert 1 < metadata["expected_active_experts"] < 2  # expected, not integer support
    assert metadata["allocated_routing_rows"] == 64
    assert graph["s0.l0.expert_up"].local_shape[0] == 16
    assert graph["s0.l0.expert_act"].local_shape[0] == 64


def test_ep_tp_etp_cp_transport_and_shared_layout_declarations_fail_explicitly():
    case_obj = parse_case(local_case())
    kwargs = dict(prefix="guard", deps="input", stage=0, requests=(), tokens=32,
                  model=case_obj.model, mode="prefill", chunks=4)
    for parallel, reason in (
        (replace(case_obj.runtime.parallel, dp=2), "requires_dp1"),
        (replace(case_obj.runtime.parallel, ep=2), "requires_ep_equals_tp"),
        (replace(case_obj.runtime.parallel, cp=2), "requires_cp_pcp1"),
    ):
        with pytest.raises(ModelingError, match=reason):
            local_expert_reduce_recipe(plan=case_obj.model.layer_plans[0],
                                       parallel=parallel, runtime=case_obj.runtime, **kwargs)
    for declaration, reason in (
        ("moe_transport", "rejects_transport_declaration"),
        ("native_moe_protocol", "rejects_native_mc2_declaration"),
        ("moe_token_partition", "rejects_token_partition_declaration"),
    ):
        runtime = replace(case_obj.runtime, kernels={
            **case_obj.runtime.kernels, declaration: {"kind": "declared"}})
        with pytest.raises(ModelingError, match=reason):
            local_expert_reduce_recipe(
                plan=case_obj.model.layer_plans[0],
                parallel=case_obj.runtime.parallel, runtime=runtime, **kwargs)
    for metadata in (
        (("enable_shared_expert_dp", True),),
        (("shared_expert_layout", "shared_expert_data_parallel_only"),),
    ):
        runtime = replace(case_obj.runtime, metadata=metadata)
        with pytest.raises(ModelingError, match="shared_expert_data_parallel_layout"):
            local_expert_reduce_recipe(
                plan=case_obj.model.layer_plans[0],
                parallel=case_obj.runtime.parallel, runtime=runtime, **kwargs)
    for layout, reason in (
        ("shared_expert_data_parallel_only", "shared_expert_data_parallel_layout"),
        ("tp_sharded", "rejects_sequence_parallel_declaration"),
    ):
        runtime = replace(case_obj.runtime, kernels={
            **case_obj.runtime.kernels,
            "sequence_parallel": {"shared_expert_layout": layout}})
        with pytest.raises(ModelingError, match=reason):
            local_expert_reduce_recipe(
                plan=case_obj.model.layer_plans[0],
                parallel=case_obj.runtime.parallel, runtime=runtime, **kwargs)
    layer = replace(case_obj.model.layer_plans[0], expert_latent_size=16)
    with pytest.raises(ModelingError, match="rejects_routed_output_transform"):
        local_expert_reduce_recipe(plan=layer, parallel=case_obj.runtime.parallel,
                                   runtime=case_obj.runtime, **kwargs)
    with pytest.raises(ModelingError, match="rejects_sequence_parallel"):
        local_expert_reduce_recipe(
            plan=case_obj.model.layer_plans[0], parallel=case_obj.runtime.parallel,
            runtime=case_obj.runtime, sequence_plan=object(), **kwargs)


@pytest.mark.parametrize("gate", (False, True))
def test_shared_weights_are_tp_sharded_and_join_precedes_one_tp_reduce(gate):
    raw = local_case(shared=1, gate=gate)
    ops = target_operators(raw, phase="prefill", tokens=32)
    graph = {op.op_id: op for op in ops}
    assert not any(".shared.tp" in op.op_id for op in ops)
    assert not any(op.recipe == "expert_tensor_reduce" for op in ops)
    reductions = [op for op in ops if op.recipe == "moe_output_reduce"]
    assert len(reductions) == 1
    join = graph[reductions[0].deps[0]]
    assert join.recipe == "expert_add"
    assert {graph[key].recipe for key in join.deps} == {
        "route_unpack", "shared_gate_apply" if gate else "dense_down"}
    shared_down = graph["s0.l0.shared.down"]
    assert graph["s0.l0.shared.gate_up"].local_shape[1] == 2 * 16
    assert shared_down.local_shape[-1] == 16  # 64-wide shared expert / tp=4
    result = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=32),
                         external_inputs={"analysis_input": 0})
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["bounds"]["combined_ms"] <= result["duration_ms"] + 1e-9


def test_representative_a5_raw_uses_capacity_activation_and_no_exchange():
    raw = json.loads(A5_FIXTURE.read_text())
    assert A5_SOURCE["sha256"]
    raw["runtime"]["moe_recipe"] = LOCAL_EXPERT_REDUCE
    ops = target_operators(raw, phase="prefill", tokens=4096)
    graph = {op.op_id: op for op in ops}
    assert not any(op.family in {"alltoall", "alltoallv"} for op in ops)
    assert not any(op.recipe in {"expert_dispatch", "expert_combine"} for op in ops)
    reductions = [op for op in ops if op.recipe == "moe_output_reduce"]
    assert len(reductions) == 8
    assert sum(op.link_bytes for op in reductions) == 8 * 2 * 3 / 4 * 4096 * 2048 * 2
    for layer in range(8):
        up = graph[f"s0.l{layer}.expert_up"]
        activation = graph[f"s0.l{layer}.expert_act"]
        down = graph[f"s0.l{layer}.expert_down"]
        pack = graph[f"s0.l{layer}.pack"]
        metadata = dict(pack.metadata)
        weight_bytes = dtype_bytes(dict(graph[f"s0.l{layer}.router"].metadata)["output_format"])
        assert up.local_shape[0] == 4096
        assert down.local_shape[0] == 4096
        assert up.flops == 2 * 4096 * 1536 * 2048
        assert down.flops == 2 * 4096 * 2048 * 768
        assert activation.local_shape == (16384, 768)
        assert activation.vector_ops == 4 * 16384 * 768
        assert activation.mte2_bytes == 2 * 16384 * 768 * 2
        assert activation.mte3_bytes == 16384 * 768 * 2
        assert metadata["allocated_routing_rows"] == 16384
        assert metadata["valid_local_rows"] == 4096
        assert metadata["local_expert_rows"] == (1024,) * 4
        assert sum(metadata["histogram_counts"]) == 4096
        assert pack.mte2_bytes == (4096 * 2048 * 2 + 16384 * (8 + weight_bytes) + 4 * 4)
        assert pack.mte3_bytes == (4096 * 2048 * 2 + 16384 * 4 + 16384 * weight_bytes
                                   + 4 * 12)
        assert pack.mte3_bytes < 16384 * 2048 * 2
    result = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=4096,
                                           scope="body"),
                         external_inputs={"analysis_input": 0})
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert result["bounds"]["combined_ms"] <= result["duration_ms"] + 1e-9


def test_old_alltoall_recipe_retains_exchange_and_full_payload():
    old_raw = local_case()
    old_raw["runtime"]["moe_recipe"] = "alltoall"
    old = by_id(old_raw)
    new = by_id(local_case())
    assert {"expert_dispatch", "expert_combine"} <= {op.recipe for op in old.values()}
    assert not any(op.recipe == "moe_output_reduce" for op in old.values())
    assert old["s0.l0.expert_act"].local_shape[0] == 16
    assert new["s0.l0.expert_act"].local_shape[0] == 64
    for suffix in ("router", "topk", "expert_up", "expert_down"):
        assert work(old["s0.l0." + suffix]) == work(new["s0.l0." + suffix])
    # The generic recipe still prices the ordinary full q*K payload exchange.
    assert old["s0.l0.pack"].mte3_bytes == 32 * 2 * (32 * 2 + 4)
    assert old["s0.l0.unpack"].mte2_bytes == 32 * 2 * (32 * 2 + 4)
    assert new["s0.l0.pack"].mte3_bytes < old["s0.l0.pack"].mte3_bytes
    assert new["s0.l0.unpack"].mte2_bytes < old["s0.l0.unpack"].mte2_bytes
    assert MOE_RECIPES[LOCAL_EXPERT_REDUCE] is local_expert_reduce_recipe
    unknown = local_case()
    unknown["runtime"]["moe_recipe"] = "not_a_recipe"
    with pytest.raises(ModelingError, match="registered_recipe"):
        parse_case(unknown)


def test_old_recipe_still_rejects_cross_ep_global_histogram():
    raw = case(tp=4, ep=4, prompt=8, output=1)
    raw["model"].update(layers=1, ffn="moe", num_experts=8, top_k=2,
                        moe_intermediate_size=64)
    raw["runtime"].update(
        moe_recipe="alltoall", max_num_batched_tokens=16,
        routing={"realizations": [{
            "scope": "body", "layer": 0, "phase": "prefill", "tokens": 8,
            "expert_rows": [8, 0, 8, 0, 0, 0, 0, 0]}]})
    with pytest.raises(ModelingError, match="ep1_pcp1"):
        target_operators(raw, phase="prefill", tokens=8)


def test_w8a8_local_expert_reduce_keeps_the_heaviest_bank_until_declared():
    """Routed int8 with a bf16 shared expert is a legal caller.

    The grouped GEMM stays on the heaviest bank unless the runtime declares
    the witnessed full eager allocation. That declaration exposes the larger
    allocation but does not replace the native local bank row histogram with
    per-EP-rank totals. Shared int8 beside floating routed experts, fp8, and
    mixed floats stay rejected.
    """
    base = parse_case(local_case())
    int8 = replace(base.model, quant=QuantSpec(
        "declared-int8", expert_weight_format="int8"))
    graph = by_id(replace(base, model=int8).to_dict())
    assert graph["s0.l0.expert_up"].local_shape[0] == 16
    assert graph["s0.l0.expert_act"].local_shape[0] == 64
    assert graph["s0.l0.router"].local_shape[0] == 32
    assert not any(op.family in {"alltoall", "alltoallv"} for op in graph.values())
    assert not any(op.recipe.startswith("arch35_fullmesh") for op in graph.values())
    declared = replace(base, model=int8).to_dict()
    declared["runtime"]["metadata"] = {"moe_grouped_gemm_rows": "full_eager_allocation"}
    full = by_id(declared)
    up, down, activation = (full["s0.l0.expert_up"], full["s0.l0.expert_down"],
                            full["s0.l0.expert_act"])
    metadata = dict(up.metadata)
    # EP=4 and local bank=2 here, so rank totals (16,16,16,16) are not the
    # local weight-bank histogram (8,8). The activated matrix stays 16 rows;
    # the 64-slot eager allocation is capacity, not extra group-list work.
    assert up.local_shape[0] == down.local_shape[0] == 16
    assert activation.local_shape[0] == 64
    assert metadata["matrix_group_rows"] == (8, 8)
    assert metadata["matrix_groups"] == 2
    assert metadata["local_expert_rows"] == (8, 8)
    assert metadata["rank_assignment_rows"] == (16, 16, 16, 16)
    assert metadata["matrix_active_rows"] == 16
    assert metadata["matrix_capacity_rows"] == 64
    assert metadata["matrix_capacity_basis"] == "eager_routing_allocation_not_activated_rows"
    assert metadata["matrix_input_capacity_shape"] == (64, 32)
    assert metadata["matrix_output_capacity_shape"] == (64, 128)
    assert up.padded_shape[0] == 64
    assert up.flops == 2 * 16 * 128 * 32
    assert full["s0.l0.topk"].local_shape[0] == 32
    assert metadata["grouped_gemm_rows"] == "full_eager_allocation"
    override = replace(base.model, quant=QuantSpec(
        "mixed-routed-override", weight_format="bf16", expert_weight_format="bf16",
        category_overrides=(("routed_expert", {"weight_format": "int8"}),)))
    override_graph = by_id(replace(base, model=override).to_dict())
    assert override_graph["s0.l0.expert_up"].local_shape[0] == 16
    mixed_shared = replace(base.model, quant=QuantSpec(
        "mixed-int8-shared", weight_format="int8", expert_weight_format="bf16"))
    with pytest.raises(ModelingError,
                       match="rejects_quantized_expert_format:shared_expert:int8"):
        target_operators(replace(base, model=mixed_shared).to_dict(),
                         phase="prefill", tokens=32)
    fp8 = replace(base.model, quant=QuantSpec(
        "declared-fp8", expert_weight_format="fp8"))
    with pytest.raises(ModelingError,
                       match="rejects_quantized_expert_format:routed_expert:fp8"):
        target_operators(replace(base, model=fp8).to_dict(),
                         phase="prefill", tokens=32)
    mixed_float = replace(base.model, quant=QuantSpec(
        "mixed-fp16-shared", weight_format="fp16", expert_weight_format="bf16"))
    with pytest.raises(ModelingError,
                       match="rejects_mixed_expert_format:shared_expert:fp16!=bf16"):
        target_operators(replace(base, model=mixed_float).to_dict(),
                         phase="prefill", tokens=32)


def test_fp16_expert_path_remains_supported_without_quantize_insertion():
    base = parse_case(local_case())
    model = replace(base.model, activation_dtype="fp16", weight_dtype="fp16",
                    compute_dtype="fp16", quant=QuantSpec(
                        "declared-fp16", weight_format="fp16",
                        expert_weight_format="fp16"))
    ops = target_operators(replace(base, model=model).to_dict(),
                           phase="prefill", tokens=32)
    graph = {op.op_id: op for op in ops}
    assert graph["s0.l0.expert_up"].local_shape[0] == 16
    assert graph["s0.l0.expert_act"].local_shape[0] == 64
    assert dict(graph["s0.l0.expert_up"].metadata)["input_format"] == "fp16"
    assert not any(op.recipe == "dynamic_quantize" for op in ops)
    assert graph["s0.l0.moe_output_reduce"].link_bytes == 2 * 3 / 4 * 32 * 32 * 2
