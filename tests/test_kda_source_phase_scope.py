"""A prefill state implementation cannot certify an unchanged decode kernel."""
from dataclasses import replace

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.family_kda import IMPLEMENTATION as GATES
from mindie_sim.prediction.family_kda_state import IMPLEMENTATION as STATE
from mindie_sim.prediction.recurrent import linear
from mindie_sim.schema.case import Software
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_subgraphs import subgraph_contract


def decode_ops(selection, tokens):
    model = load_model("kimi-k3")
    plan = next(layer for layer in model.layer_plans if layer.attention == "kda")
    lowerings = {"kda": selection} if selection else {}
    return linear(
        prefix="decode", plan=plan,
        views=(RequestView("r", tokens, 2048, False, "decode"),),
        stage=0, deps="input", model=model, parallel=ParallelSpec(tp=8),
        quant=model.quant, mode="decode", padded_tokens=tokens,
        policy=AttentionPolicy(recurrent_chunk=64),
        runtime=Software.from_mapping({"kernels": {"lowerings": lowerings}}),
    ).ops


def declaration(ops):
    return next(
        group for op in ops
        for group in dict(op.metadata).get("structure_subgraphs", ())
        if group["role"] == "kda_recurrent")


@pytest.mark.parametrize("tokens", [1, 2, 4])
def test_prefill_state_selection_preserves_decode_work_without_native_contract(tokens):
    ordinary, selected = decode_ops(None, tokens), decode_ops(STATE, tokens)
    assert tuple(replace(op, metadata=()) for op in selected) == tuple(
        replace(op, metadata=()) for op in ordinary)
    group = declaration(selected)
    assert not group["tensor_view"]["in_kernel_gate_source"]
    assert group["implementation"]["selection"] is None
    assert subgraph_contract(group) is None
    assert all(not op.source_boundary for op in selected)


def test_explicit_decode_gate_selection_keeps_its_declared_contract():
    group = declaration(decode_ops(GATES, 1))
    assert group["tensor_view"]["in_kernel_gate_source"]
    assert group["implementation"]["selection"] == GATES
    assert subgraph_contract(group) is not None
