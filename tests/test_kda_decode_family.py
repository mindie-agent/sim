"""Ordinary KDA caller producer boundaries and real source execution."""
from dataclasses import replace

import pytest

from mindie_sim import graph_costs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.family_kda_recurrent import IMPLEMENTATION
from mindie_sim.prediction.recurrent import linear
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Software
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView
from mindie_sim.structure_subgraphs import subgraph_contract
from test_kda_state_pipeline import hardware_case


def caller(lengths=(1, 4), *, padded=None, dv=128, options=None, phase="decode"):
    model = load_model("kimi-k3")
    plan = next(p for p in model.layer_plans if p.attention == "kda")
    plan = replace(plan, linear_key_heads=1, linear_value_heads=2, linear_heads=2,
                   linear_head_dim=dv, linear_key_head_dim=128)
    model = replace(model, layers=1, hidden_size=512, layer_plans=(plan,),
                    quant=quant_preset("bf16"))
    runtime = Software.from_mapping({"kernels": {"lowerings": {
        "kda": {"implementation": IMPLEMENTATION, "options": options or {}}}}})
    views = tuple(RequestView(str(i), q, 2048, False, phase) for i, q in enumerate(lengths))
    ops = linear(prefix="kda", plan=plan, views=views, stage=0, deps="input",
                 model=model, parallel=ParallelSpec(), quant=model.quant, mode=phase,
                 padded_tokens=padded or sum(lengths), policy=AttentionPolicy(),
                 runtime=runtime).ops
    case = parse_case(hardware_case(units=4, split_copy=True))
    die = replace(case.machine.dies[0], buffers=(("ub", 256*1024),),
                  buffer_units=(("ub", 8),))
    return replace(case, model=model, runtime=runtime,
                   machine=replace(case.machine, dies=(die,))), ops


@pytest.mark.parametrize("dv", (128, 256))
@pytest.mark.parametrize("internal_beta", (False, True))
def test_source_replaces_fused_members_once_and_keeps_actual_producers(dv, internal_beta):
    _, ops = caller(dv=dv, options={"use_beta_sigmoid": internal_beta})
    by_recipe = {op.recipe: op for op in ops}
    kernel = by_recipe[IMPLEMENTATION]
    assert not ({"qk_l2norm", "delta_gates"} & by_recipe.keys())
    assert sum(op.family == "recurrent_scan" for op in ops) == 1
    assert ("linear_beta_sigmoid" in by_recipe) is not internal_beta
    beta = by_recipe["linear_beta" if internal_beta else "linear_beta_sigmoid"]
    assert kernel.deps == (by_recipe["short_convolution"].op_id,
                           by_recipe["linear_decay_b"].op_id, beta.op_id)
    # Key-channel gates do not grow when only the value/state width changes.
    assert by_recipe["linear_decay_b"].local_shape == (5, 256, 128)
    assert by_recipe["linear_gate"].local_shape == (5, 2*dv, 512)
    params = dict(kernel.metadata)["kda_recurrent_params"]
    assert params["lengths"] == (1, 4)
    assert params["beta_dtype"] == ("bf16" if internal_beta else "fp32")
    group, = dict(kernel.metadata)["structure_subgraphs"]
    assert group["op_ids"] == group["entry_op_ids"] == (kernel.op_id,)
    assert group["tensor_view"]["beta_dtype"] == params["beta_dtype"]
    assert subgraph_contract(group) is not None
    available = {"input"}
    for op in ops:
        assert set(op.deps) <= available
        available.add(op.op_id)


def test_physical_projection_padding_does_not_invent_recurrent_state_updates():
    _, ops = caller((1, 4), padded=16)
    by_recipe = {op.recipe: op for op in ops}
    assert by_recipe["linear_q"].local_shape[0] == 16
    assert by_recipe["linear_output"].local_shape[0] == 16
    native = by_recipe[IMPLEMENTATION]
    assert native.local_shape == (5, 2, 128, 128)
    assert dict(native.metadata)["kda_recurrent_params"]["lengths"] == (1, 4)


def test_complete_ordinary_graph_runs_on_the_registered_source_program():
    case, ops = caller()
    result = graph_costs(case, ops, input_ready_ms=7, isolated=False,
                         external_inputs={"input": 7}, trace_limit=1000)
    assert result["duration_ms"] > 0
    execution = result["execution"]
    assert execution["live_tasks"] == execution["live_programs"] == 0
    assert execution["serviced"]["host@r0:s0"] == 0
    assert not any(execution["live_allocated"].values())
    native = next(row for row in result["timeline"] if row["recipe"] == IMPLEMENTATION)
    assert native["ready_ms"] >= 7
    assert native["program"] == "OperatorPlan"
    assert native["source_boundary"] == "kernel"


def test_decode_source_selection_does_not_certify_prefill_algorithm():
    _, ops = caller((64,), phase="prefill")
    assert not any(op.recipe == IMPLEMENTATION for op in ops)
    group = next(group for op in ops for group in dict(op.metadata).get("structure_subgraphs", ())
                 if group["role"] == "kda_chunk")
    assert group["implementation"]["selection"] is None
    assert not group["tensor_view"]["in_kernel_gate_source"]
