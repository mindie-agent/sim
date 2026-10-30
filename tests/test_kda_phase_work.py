"""Independent chunk arithmetic keeps KDA products at their causal stage."""
from dataclasses import replace

import pytest

from mindie_sim.model_catalog import load_model
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.recurrent import linear
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import RequestView


def recurrent_ops(kind, lengths, *, heads=3, dk=16, dv=32, chunk=64):
    model = load_model("kimi-k3")
    plan = replace(
        model.layer_plans[0], attention=kind, linear_heads=heads,
        linear_key_heads=heads, linear_value_heads=heads,
        linear_head_dim=dv, linear_key_head_dim=dk,
    )
    views = tuple(RequestView(str(i), length, 0, False, "prefill",
                              draft_prepare=False) for i, length in enumerate(lengths))
    return linear(
        prefix="core", plan=plan, views=views, stage=0, deps="input",
        model=model, parallel=ParallelSpec(), quant=model.quant,
        mode="prefill", padded_tokens=sum(lengths),
        policy=AttentionPolicy(recurrent_chunk=chunk),
    ).ops


@pytest.mark.parametrize("lengths", [(1,), (64,), (65,), (65, 127)])
def test_kda_qk_kk_prepare_and_aqk_v_finalize_preserve_ragged_work(lengths):
    heads, dk, dv, chunk = 3, 16, 32, 64
    ops = recurrent_ops("kda", lengths, heads=heads, dk=dk, dv=dv, chunk=chunk)
    by_recipe = {op.recipe: op for op in ops}
    for index, tokens in enumerate(lengths):
        pairs = (tokens // chunk) * chunk**2 + (tokens % chunk)**2
        prepare = by_recipe[f"chunk_products{index}"]
        final = by_recipe[f"recurrent_output{index}"]
        scan = by_recipe[f"recurrent_scan{index}"]
        assert prepare.flops == 4 * pairs * heads * dk
        assert final.flops == 2 * tokens * heads * dk * dv + 2 * pairs * heads * dv
        assert prepare.flops + final.flops == (
            2 * tokens * heads * dk * dv + 2 * pairs * heads * (2 * dk + dv))
        assert scan.op_id in final.deps
        assert prepare.op_id in by_recipe[f"triangular_solve{index}"].deps


def test_gdn_retains_its_separate_existing_algorithm_work():
    by_recipe = {op.recipe: op for op in recurrent_ops("gdn", (65,))}
    pairs, heads, dk, dv = 64**2 + 1, 3, 16, 32
    assert by_recipe["chunk_products0"].flops == 2 * pairs * heads * dk
    assert by_recipe["recurrent_output0"].flops == (
        2 * 65 * heads * dk * dv + 2 * pairs * heads * (dk + dv))
