"""Ordinary frontend selection, explicit geometry and graph execution."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.prediction.attention_prefill import StagedGQAPrefill
from mindie_sim.prediction.cost import CostModel
from mindie_sim.prediction.execution import Execution
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def prefill_case(*, decode=False):
    raw = case()
    recipes = {"flash_attention_prefill": {"kind": "staged_gqa_prefill"}}
    if decode:
        recipes["flash_attention_decode"] = {"kind": "staged_gqa_decode"}
    raw["runtime"]["kernels"] = {"recipes": recipes}
    return raw


def test_ordinary_prefill_selection_declares_ragged_request_geometry():
    parsed = parse_case(prefill_case())
    views = (RequestView("first", 3, 5, False, "prefill"),
             RequestView("second", 5, 2, False, "prefill"))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input",
                            tokens=8, mode="prefill")
    attention = [op for op in ops if op.family == "attention" and op.layer == 0]
    assert len(attention) == 1
    op = attention[0]
    assert op.recipe == "flash_attention_prefill"
    assert dict(op.metadata)["gqa_prefill"] == {
        "requests": ((3, 5), (5, 2)), "query_heads": 4, "kv_heads": 2,
        "head_dim": 8, "query_dtype": "bf16", "kv_dtype": "bf16",
        "causal": True, "window": 0, "cp": 1,
    }
    # Geometry declaration does not alter the semantic causal count.
    assert op.flops == 4 * (3 * 5 + 3 * 4 // 2 + 5 * 2 + 5 * 6 // 2) * 4 * 8

    work = Frontend()(IterationPlan(views), parsed.model, parsed.runtime,
                      parsed.machine)
    execution = Execution(parsed, CostModel(parsed))
    step = execution.add(work, 0, 0)
    execution.engine.advance(until=(step.completion,))
    selected = [op for op in work.ops if op.family == "attention"]
    assert selected and selected[0].recipe == "flash_attention_prefill"
    assert execution.engine.summary()["live_tasks"] == 0
    assert execution.engine.summary()["live_programs"] == 0
    assert not any(execution.engine.allocated.values())
    assert execution.engine.serviced["memory@r0:s0"] > 0
    assert execution.recipes_used.get("flash_attention_prefill", 0) >= 1


def test_default_and_decode_selectors_are_untouched():
    parsed = parse_case(case())
    views = (RequestView("prefill", 4, 0, False, "prefill"),)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input",
                            tokens=4, mode="prefill")
    assert {op.recipe for op in ops if op.family == "attention"} == {
        "flash_attention"}
    assert dict(next(op for op in ops if op.family == "attention").metadata)[
        "gqa_prefill"]["requests"] == ((4, 0),)

    parsed = parse_case(prefill_case(decode=True))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input",
                            tokens=4, mode="prefill")
    assert {op.recipe for op in ops if op.family == "attention"} == {
        "flash_attention_prefill"}
    decode_views = (RequestView("decode", 1, 7, False, "decode"),)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, decode_views,
                            ready="input", tokens=1, mode="decode")
    attention = [op for op in ops if op.family == "attention"]
    assert {op.recipe for op in attention} == {"flash_attention_decode"}
    assert "gqa_decode" in dict(attention[0].metadata)


def test_draft_prefill_phase_binds_the_same_explicit_recipe():
    parsed = parse_case(prefill_case())
    views = (RequestView("draft", 2, 4, False, "prefill"),)
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input",
                            tokens=2, mode="draft_prefill")
    attention = [op for op in ops if op.family == "attention"]
    assert {op.recipe for op in attention} == {"flash_attention_prefill"}
    assert all(op.phase == "draft_prefill" for op in attention)
    assert dict(attention[0].metadata)["gqa_prefill"]["requests"] == ((2, 4),)


def test_source_required_binding_precedes_generic_recipe():
    op = make_source_op()
    book = RecipeBook.from_mapping(
        {"recipes": {"flash_attention_prefill": {"kind": "staged_gqa_prefill"}}})
    target = replace(op, recipe="flash_attention_prefill",
                     source_boundary="kernel")
    with pytest.raises(ModelingError, match="source_plan_binding_required"):
        book.select(target)
    # A semantic recipe remains selectable when no source binding is required.
    assert isinstance(book.select(op), StagedGQAPrefill)


def make_source_op():
    from mindie_sim.schema.operators import OpSpec
    metadata = {
        "requests": ((2, 3),), "query_heads": 4, "kv_heads": 2, "head_dim": 8,
        "query_dtype": "bf16", "kv_dtype": "bf16", "causal": True,
        "window": 0, "cp": 1,
    }
    shape = (2, 5, 4, 8)
    return OpSpec("source_attention", "attention", "prefill", 0, 0, ("r",),
                  shape, shape, shape, shape, recipe="flash_attention_prefill",
                  dtype="bf16", metadata=(("gqa_prefill", metadata),))
