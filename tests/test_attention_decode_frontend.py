"""Ordinary decode binding preserves real GQA requests and phase selection."""
from mindie_sim import target_operators
from mindie_sim.prediction.attention import AttentionPolicy
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


def test_decode_recipe_selection_preserves_prefill_and_request_geometry():
    raw = case()
    raw["runtime"]["kernels"] = {
        "recipes": {"flash_attention_decode": {"kind": "staged_gqa_decode"}}
    }
    parsed = parse_case(raw)
    prefill = target_operators(parsed, phase="prefill", tokens=3)
    assert {op.recipe for op in prefill if op.family == "attention"} == {"flash_attention"}
    views = (RequestView("short", 1, 7, False, "decode"),
             RequestView("verify", 3, 11, False, "decode"))
    ops, _ = Frontend().body(parsed.model, parsed.runtime, views, ready="input",
                            tokens=4, mode="decode")
    attention = [op for op in ops if op.family == "attention"]
    assert len(attention) == 2
    for op in attention:
        assert op.recipe == "flash_attention_decode"
        assert dict(op.metadata)["gqa_decode"] == {
            "requests": ((1, 7), (3, 11)), "query_heads": 4, "kv_heads": 2,
            "head_dim": 8, "query_dtype": "bf16", "kv_dtype": "bf16",
            "causal": True, "window": 0, "cp": 1,
        }
        # The new geometry declaration must not change the logical model.
        assert op.flops == 4 * (8 + 12 + 13 + 14) * 4 * 8


def test_unfused_decode_contract_is_per_request():
    parsed = parse_case(case())
    views = (RequestView("first", 1, 3, False, "decode"),
             RequestView("second", 2, 5, False, "decode"))
    frontend = Frontend(attention_policy=AttentionPolicy(fused_batch=False))
    ops, _ = frontend.body(parsed.model, parsed.runtime, views, ready="input",
                           tokens=3, mode="decode")
    attention = [op for op in ops if op.family == "attention" and op.layer == 0]
    assert [dict(op.metadata)["gqa_decode"]["requests"] for op in attention] == [
        ((1, 3),), ((2, 5),),
    ]
    assert {op.recipe for op in attention} == {"flash_attention"}


def test_complete_mtp_iteration_selects_draft_and_verification_attention():
    raw = case()
    raw["model"] = {"catalog_key": "qwen3.8-27b"}
    raw["runtime"].update(
        metadata={"draft_mode": "mtp", "draft_proposal_tokens": 3},
        kernels={"recipes": {
            "flash_attention_decode": {"kind": "staged_gqa_decode"}}})
    parsed = parse_case(raw)
    view = RequestView("request", 1, 16384, True, "decode",
                       draft_proposal_tokens=3)
    work = Frontend()(IterationPlan((view,)), parsed.model, parsed.runtime, parsed.machine)
    attention = [op for op in work.ops if op.family == "attention"]
    assert len(attention) == 19
    assert {op.recipe for op in attention} == {"flash_attention_decode"}
    draft = [op for op in attention if op.phase == "draft"]
    target = [op for op in attention if op.phase == "decode"]
    assert len(draft) == 3 and len(target) == 16
    assert [dict(op.metadata)["gqa_decode"]["requests"] for op in draft] == [
        ((1, 16384),), ((1, 16385),), ((1, 16386),),
    ]
    assert {dict(op.metadata)["gqa_decode"]["requests"] for op in target} == {
        ((4, 16384),)}
