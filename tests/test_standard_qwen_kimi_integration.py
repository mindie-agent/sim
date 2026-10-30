"""Small real HF-config inputs exercise public graphs without loading weights."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, predict, target_graphs
from mindie_sim.prediction.structure import parse_case
from test_prediction_system import case


FIXTURES = Path(__file__).parent / "fixtures" / "standard-qwen-kimi"
NAMES = (
    "qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b",
    "qwen3-8b", "qwen3-30b-a3b", "kimi-k2-instruct", "kimi-vl-a3b-instruct",
)


def small_case(name):
    config = json.loads((FIXTURES / f"{name}.json").read_text())
    text = config.get("text_config", config)
    text.update(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
    if name.startswith("qwen3"):
        # This deliberately differs from hidden/heads, as in Qwen3-30B-A3B.
        text["head_dim"] = 16
    if "num_experts" in text:
        text.update(num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32)
    if "shared_expert_intermediate_size" in text:
        text["shared_expert_intermediate_size"] = 48
    if name.startswith("kimi"):
        text.update(n_routed_experts=4, num_experts_per_tok=2,
                    moe_intermediate_size=32, kv_lora_rank=16,
                    qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8)
        if text.get("q_lora_rank") is not None:
            text["q_lora_rank"] = 16
    raw = case(output=2)
    raw["model"] = config
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e9
    raw["runtime"]["routing"] = {"policy": "balanced"}
    return raw


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("phase,tokens,context", (("prefill", 4, 0), ("decode", 1, 8)))
def test_official_family_small_geometry_runs_public_graph_and_preserves_work(
    name, phase, tokens, context,
):
    raw = small_case(name)
    parsed = parse_case(raw)
    graphs = target_graphs(raw, phase=phase, tokens=tokens, context=context)
    ops = tuple(op for group in graphs.values() for op in group)
    if name.startswith("qwen"):
        width = 16 if name.startswith("qwen3") else 8
        projections = [op for op in ops if op.family == "qkv_projection"]
        assert len(projections) == 2
        assert sum(op.flops for op in projections) == 2 * 2 * tokens * 32 * 8 * width
        qk_norm = [op for op in ops if op.recipe == "qk_norm"]
        assert len(qk_norm) == (2 if name.startswith("qwen3") else 0)
        routers = [op for op in ops if op.family == "router"]
        assert all(op.dtype == "bf16" for op in routers)
        assert not any(op.op_id.endswith("router_cast") for op in ops)
    else:
        assert all(p.expert_latent_size == 0 and p.attention == "mla"
                   for p in parsed.model.layer_plans)
        assert parsed.model.auxiliary.dspark is None
        direct = [op for op in ops if op.op_id.endswith(".q_proj")]
        if name == "kimi-vl-a3b-instruct":
            assert len(direct) == 2
            assert sum(op.flops for op in direct) == 2 * 2 * tokens * 32 * 4 * 12
            assert not any(op.op_id.endswith((".q_a", ".q_b", ".q_norm")) for op in ops)
        else:
            assert not direct
            assert parsed.model.quant.format_for("routed_expert") in {"fp8", "fp8_e4m3"}
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    assert report["bounds"]["combined_ms"] <= report["duration_ms"] + 1e-9
    assert all(abs(value) < 1e-9 for value in report["execution"]["live_allocated"].values())


@pytest.mark.parametrize("name", NAMES)
def test_raw_and_wrapped_hf_inputs_reach_identical_serving_program(name):
    raw = small_case(name)
    wrapped = deepcopy(raw)
    wrapped["model"] = {"hf_config": wrapped["model"], "name": "renamed-geometry"}
    direct_result = predict(raw)
    wrapped_result = predict(wrapped)
    assert direct_result["requests"][0]["status"] == "served"
    assert direct_result["execution"]["work"] == wrapped_result["execution"]["work"]
    assert direct_result["execution"]["serviced"] == wrapped_result["execution"]["serviced"]
    assert direct_result["metrics"] == wrapped_result["metrics"]


def test_custom_attention_retains_known_work_without_claiming_a_gqa_cache():
    raw = small_case("qwen3-8b")
    raw["model"]["layer_types"] = ["full_attention", "custom_attention"]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    unknown = [op for ops in graphs.values() for op in ops if op.layer == 1]
    assert any(op.family == "qkv_projection" for op in unknown)
    assert any(op.recipe == "dense_gate_up" for op in unknown)
    assert any(op.recipe == "unmodeled_attention_core" for op in unknown)
    assert not any(op.recipe in {"kv_store", "flash_attention"} for op in unknown)
    report = graph_costs(raw, graphs)
    assert report["model_coverage"] == "partial"
    assert "unsupported_attention_layer:1:custom_attention" in report["unmodeled_features"]
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0


@pytest.mark.parametrize("name", ("qwen2-57b-a14b", "qwen3-30b-a3b",
                                 "kimi-k2-instruct", "kimi-vl-a3b-instruct"))
def test_nondivisible_tokens_use_the_declared_expected_routing_path(name):
    raw = small_case(name)
    raw["runtime"].pop("routing")
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    report = graph_costs(raw, graphs)
    assert report["duration_ms"] > 0
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
