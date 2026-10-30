"""TP reduction cutpoints, physical payload and public graph behavior."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, target_graphs, target_operators
from mindie_sim.prediction.structure import parse_case
from test_balanced_experts import balanced_case


def configuration(*, shared=1, gate=False):
    raw = balanced_case(tp=4, tokens=32)
    raw["model"].update(num_shared_experts=shared)
    raw = parse_case(raw).to_dict()
    raw["model"]["layer_plans"][0]["shared_expert_gate"] = gate
    raw["runtime"]["moe_recipe"] = "tp_join_reduce"
    return raw


@pytest.mark.parametrize("shared,gate", ((0, False), (1, False), (1, True)))
def test_reduce_token_rows_only_after_weighted_unpermute_and_shared_join(shared, gate):
    raw = configuration(shared=shared, gate=gate)
    old = deepcopy(raw)
    old["runtime"]["moe_recipe"] = "alltoall"
    early = target_operators(old, phase="prefill", tokens=32)
    late = target_operators(raw, phase="prefill", tokens=32)
    by_id = {op.op_id: op for op in late}
    reduction = next(op for op in late if op.recipe == "moe_output_reduce")
    assert reduction.local_shape == (32, 32)
    assert reduction.link_bytes == 2 * 3 / 4 * 32 * 32 * 2
    assert not any(op.recipe == "expert_tensor_reduce" or ".shared.tp" in op.op_id for op in late)
    predecessor = by_id[reduction.deps[0]]
    assert predecessor.recipe == ("expert_add" if shared else "route_unpack")
    if shared:
        inputs = [by_id[key] for key in predecessor.deps]
        assert {op.recipe for op in inputs} == {
            "route_unpack", "shared_gate_apply" if gate else "dense_down"}
    # This moves only summation. Matrix/activation/packing work is unchanged.
    for field in ("flops", "vector_ops", "hbm_bytes"):
        assert sum(getattr(op, field) for op in early if op.family != "allreduce") == \
               sum(getattr(op, field) for op in late if op.family != "allreduce")
    before = sum(op.link_bytes for op in early if op.family == "allreduce")
    after = sum(op.link_bytes for op in late if op.family == "allreduce")
    assert before - after == reduction.link_bytes * (2 + shared - 1)
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=32))
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0
    assert graph["bounds"]["combined_ms"] <= graph["duration_ms"] + 1e-9


def test_linear_token_combine_commutes_with_tp_sum_but_nonlinear_output_does_not():
    # Two TP ranks, two expert slots, one shared contribution, replicated gates.
    partials = ((2, 5), (7, 11))
    weights, shared, shared_gate = (0.25, 0.75), (3, 13), 0.5
    early = sum(weights[e] * sum(rank[e] for rank in partials) for e in range(2))
    early += shared_gate * sum(shared)
    late = sum(sum(weights[e] * rank[e] for e in range(2)) + shared_gate * shared[r]
               for r, rank in enumerate(partials))
    assert early == late
    assert sum(sum(rank) ** 2 for rank in partials) != sum(map(sum, partials)) ** 2
    raw = configuration()
    raw["model"]["layer_plans"][0]["expert_latent_size"] = 16
    with pytest.raises(ValueError, match="routed_output_transform"):
        target_operators(raw, phase="prefill", tokens=32)


def test_explicit_late_recipe_does_not_replace_cross_rank_exchange():
    raw = balanced_case(tp=4, ep=4, tokens=32)
    raw["runtime"]["moe_recipe"] = "tp_join_reduce"
    with pytest.raises(ValueError, match="tp_only"):
        target_graphs(raw, phase="prefill", tokens=32)


def test_mla_and_moe_reductions_in_one_catalog_layer_keep_distinct_dependencies():
    raw = balanced_case(tp=4, tokens=32)
    raw["model"] = {"catalog_key": "GLM-5.3-W4A8", "layers": 4}
    raw["runtime"]["moe_recipe"] = "tp_join_reduce"
    graphs = target_graphs(raw, phase="decode", tokens=32, context=4096)
    ops = graphs[0]
    assert len({op.op_id for op in ops}) == len(ops)
    mla = next(op for op in ops if op.op_id == "s0.l3.output_reduce")
    moe = next(op for op in ops if op.recipe == "moe_output_reduce")
    assert mla.op_id != moe.op_id
    assert mla.deps != moe.deps


@pytest.mark.parametrize("suffix,batch", (("c48", 64), ("c136", 160)))
def test_retained_tp_only_case_reduces_token_payload_after_shared_join(suffix, batch):
    path = Path(__file__).resolve().parents[1] / "examples/cases/high-concurrency-20261030/new17"
    raw = json.loads((path / f"qwen35-397b-a3-tp16-{suffix}.json").read_text())
    ops = target_operators(raw, phase="decode", tokens=4, sequences=batch,
                           context=4104, scope="body")
    by_id = {op.op_id: op for op in ops}
    reductions = [op for op in ops if op.recipe == "moe_output_reduce"]
    assert len(reductions) == 60
    for op in reductions:
        assert op.local_shape == (batch * 4, 4096)
        assert op.link_bytes == 2 * 15 / 16 * batch * 4 * 4096 * 2
        join = by_id[op.deps[0]]
        assert join.recipe == "expert_add"
        assert {by_id[key].recipe for key in join.deps} == {"route_unpack", "shared_gate_apply"}
    assert not any(op.recipe == "expert_tensor_reduce" for op in ops)
