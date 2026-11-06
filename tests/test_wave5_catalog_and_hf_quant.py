"""Wave 5 catalog expansion, customize_model_spec, and HF role exclusion tests."""
from __future__ import annotations

import pytest

from mindie_sim import operator_costs, target_graphs, target_operators
from mindie_sim.catalog.hf_config import resolve_quant, resolve_role_exclusions
from mindie_sim.coverage import model_coverage
from mindie_sim.model_catalog import (
    CATALOG,
    MODEL_ALIASES,
    body_parameter_count,
    body_weight_bytes,
    build_model_spec,
    customize_model_spec,
    load_architecture_profile,
    load_model,
)
from mindie_sim.schema.common import ModelingError
from test_prediction_system import case


WAVE5_MODELS = {
    "deepseek-v2-lite": ("DeepSeek-V2-Lite", "dsv2lite"),
    "deepseek-v3.2-prune-3l": ("DeepSeek-V3.2-Prune-3L", "dsv32prune_L3"),
    "deepseek-v3": ("DeepSeek-V3", "DeepSeek-V3-0324", "dsv3"),
    "deepseek-r1": ("DeepSeek-R1", "DeepSeek-R1-0528", "dsr1"),
    "deepseek-v3.2": ("DeepSeek-V3.2", "DeepSeek-V3.2-Exp", "dsv3.2"),
    "glm-5.1": ("GLM-5.1", "glm5.1", "glm51"),
    "glm-5.2": ("GLM-5.2", "glm5.2", "glm52"),
    "kimi-k2": ("Kimi-K2", "Kimi-K2-Instruct", "kimi_k2"),
    "kimi-k2-thinking": ("Kimi-K2-Thinking", "kimi_k2_thinking"),
    "kimi-k2.5": ("Kimi-K2.5", "kimi_k2.5", "kimi_k25", "kimi-k25"),
    "qwen3-30b-a3b": ("Qwen3-30B-A3B", "qwen30ba3b"),
    "qwen3-32b": ("Qwen3-32B", "qwen3_32b"),
    "qwen3-235b-a22b": ("Qwen3-235B-A22B", "qwen3_235b_a22b", "qwen3-235b"),
    "qwen3-next-80b-a3b": ("Qwen3-Next-80B-A3B", "Qwen3-Next-80B-A3B-Instruct", "qwen3_next_80b_a3b"),
    "qwen3.5-27b": ("Qwen3.5-27B", "qwen35_27b"),
    "qwen3.5-397b-a17b": ("Qwen3.5-397B-A17B", "qwen3_5_397b_a17b", "qwen3.5-397b"),
}


@pytest.mark.parametrize("key,aliases", list(WAVE5_MODELS.items()))
def test_wave5_catalog_entries_and_aliases_load_complete_with_positive_weights(key, aliases):
    assert key in CATALOG
    spec = load_model(key)
    cov = model_coverage(spec)
    assert not any(f.startswith("unrecognized_config_field:") for f in cov["unmodeled_features"])
    if key in {
        "deepseek-v3.2-prune-3l", "deepseek-v3", "deepseek-r1", "deepseek-v3.2",
        "kimi-k2", "kimi-k2-thinking", "kimi-k2.5", "qwen3-next-80b-a3b", "qwen3.5-397b-a17b",
    }:
        assert cov["model_coverage"] == "partial"
    else:
        assert cov["model_coverage"] == "complete", f"{key} had unmodeled gaps: {cov}"
    params = body_parameter_count(spec)
    weights = body_weight_bytes(spec)
    assert params > 0
    assert weights > 0
    profile = load_architecture_profile(key)
    assert profile.num_layers == spec.layers
    for alias in aliases:
        assert MODEL_ALIASES[alias] == key
        alias_spec = load_model(alias)
        assert alias_spec == spec


def test_wave5_specific_geometry_contracts():
    dsv2 = load_model("deepseek-v2-lite")
    assert dsv2.layers == 27
    assert dsv2.hidden_size == 2048
    assert dsv2.num_experts == 64 and dsv2.top_k == 6
    assert dsv2.layer_plans[0].ffn == "dense"
    assert all(p.ffn == "moe" for p in dsv2.layer_plans[1:])

    dsv32 = load_model("deepseek-v3.2-prune-3l")
    assert dsv32.layers == 3
    assert dsv32.hidden_size == 7168
    assert dsv32.num_experts == 64 and dsv32.top_k == 8
    assert dsv32.layer_plans[0].ffn == "dense"
    assert [p.ffn for p in dsv32.layer_plans] == ["dense", "moe", "moe"]
    assert dsv32.layer_plans[1].router_groups == 8
    assert dsv32.layer_plans[1].router_group_topk == 4

    dsv3 = load_model("deepseek-v3")
    assert dsv3.layers == 61 and dsv3.hidden_size == 7168
    assert dsv3.num_experts == 256 and dsv3.top_k == 8
    assert sum(1 for p in dsv3.layer_plans if p.ffn == "dense") == 3

    dsv32_full = load_model("deepseek-v3.2")
    assert dsv32_full.layers == 61 and dsv32_full.mla is not None
    assert dsv32_full.mla.index_topk == 2048 and dsv32_full.mla.index_n_heads == 64

    glm51 = load_model("glm-5.1")
    assert glm51.layers == 78 and glm51.num_experts == 256 and glm51.top_k == 8
    assert sum(1 for p in glm51.layer_plans if p.ffn == "dense") == 3
    assert all(p.indexer and p.indexer_type == "full" for p in glm51.layer_plans)

    glm52 = load_model("glm-5.2")
    assert glm52.layers == 78 and glm52.num_experts == 256 and glm52.top_k == 8
    assert sum(1 for p in glm52.layer_plans if p.indexer_type == "shared") > 0

    k2 = load_model("kimi-k2")
    assert k2.layers == 61 and k2.num_experts == 384 and k2.top_k == 8

    k25 = load_model("kimi-k2.5")
    assert k25.layers == 61 and k25.num_experts == 384 and k25.top_k == 8
    assert "vision_encoder_and_projector:language_backbone_only" in model_coverage(k25)["unmodeled_features"]

    qwen30b = load_model("qwen3-30b-a3b")
    assert qwen30b.layers == 48
    assert qwen30b.hidden_size == 2048
    assert qwen30b.num_experts == 128 and qwen30b.top_k == 8
    assert all(p.qk_norm and p.ffn == "moe" for p in qwen30b.layer_plans)

    qwen32b = load_model("qwen3-32b")
    assert qwen32b.layers == 64 and qwen32b.hidden_size == 5120 and qwen32b.ffn == "dense"

    qwen235b = load_model("qwen3-235b-a22b")
    assert qwen235b.layers == 94 and qwen235b.num_experts == 128 and qwen235b.top_k == 8

    qwen_next = load_model("qwen3-next-80b-a3b")
    assert qwen_next.layers == 48 and qwen_next.num_experts == 512 and qwen_next.top_k == 10
    assert sum(1 for p in qwen_next.layer_plans if p.attention == "gdn") == 36
    assert sum(1 for p in qwen_next.layer_plans if p.attention == "gqa") == 12
    assert all(p.q_output_gate for p in qwen_next.layer_plans if p.attention == "gqa")

    qwen397b = load_model("qwen3.5-397b-a17b")
    assert qwen397b.layers == 60 and qwen397b.num_experts == 512 and qwen397b.top_k == 10
    assert sum(1 for p in qwen397b.layer_plans if p.attention == "gdn") == 45
    assert sum(1 for p in qwen397b.layer_plans if p.attention == "gqa") == 15

    qwen35 = load_model("qwen3.5-27b")
    assert qwen35.layers == 64
    assert qwen35.hidden_size == 5120
    assert qwen35.intermediate_size == 17408
    assert sum(1 for p in qwen35.layer_plans if p.attention == "gdn") == 48
    assert sum(1 for p in qwen35.layer_plans if p.attention == "gqa") == 16


@pytest.mark.parametrize("name", ("qwen3-next-80b-a3b", "qwen3.5-397b-a17b"))
def test_hybrid_qwen_routing_keeps_expert_probability_normalization(name):
    spec = customize_model_spec(name, layers=4)
    assert all(plan.router_mode == "softmax" and plan.router_norm_topk
               for plan in spec.layer_plans)
    raw = case(prompt=4, output=1)
    raw["model"] = spec.to_dict()
    ops = target_operators(raw, phase="decode", tokens=1, context=4)
    normalized = [op for op in ops if op.recipe == "route_normalize"]
    assert len(normalized) == 4
    assert all(op.vector_ops == 3 * spec.top_k and op.deps for op in normalized)
    raw["model"]["layer_plans"][0]["router_norm_topk"] = False
    control = target_operators(raw, phase="decode", tokens=1, context=4)
    assert sum(op.recipe == "route_normalize" for op in control) == 3


@pytest.mark.parametrize("model_name", ["qwen3-30b-a3b", "deepseek-v2-lite", "deepseek-v4-flash"])
@pytest.mark.parametrize("experts", [16, 32, 48, 64])
@pytest.mark.parametrize("top_k", [2, 4, 6, 8])
def test_customize_model_spec_grid_conserves_parameters_and_executes_graphs(
    model_name, experts, top_k,
):
    spec = customize_model_spec(
        model_name, layers=8, n_routed_experts=experts, num_experts_per_tok=top_k,
    )
    assert spec.layers == 8
    assert len(spec.layer_plans) == 8
    assert spec.num_experts == experts
    assert spec.top_k == top_k
    moe_plans = [p for p in spec.layer_plans if p.ffn == "moe"]
    assert moe_plans
    assert all(p.num_experts == experts and p.top_k == top_k for p in moe_plans)

    # Top-K routing does not alter total body parameter count for fixed E.
    spec_k2 = customize_model_spec(
        model_name, layers=8, n_routed_experts=experts, num_experts_per_tok=2,
    )
    assert body_parameter_count(spec) == body_parameter_count(spec_k2)
    assert body_weight_bytes(spec) == body_weight_bytes(spec_k2)

    raw = case(prompt=4, output=1)
    raw["model"] = spec.to_dict()
    raw["machine"]["device"]["memory_bytes"] = 10**14
    raw["machine"]["device"]["compute_peaks"]["fp8_e4m3"] = 1e12
    raw["machine"]["device"]["compute_peaks"]["int8"] = 1e12
    graphs = target_graphs(raw, phase="prefill", tokens=4)
    assert graphs and all(len(ops) > 0 for ops in graphs.values())
    ops = target_operators(raw, phase="prefill", tokens=4)
    assert sum(op.flops for op in ops) > 0
    report = operator_costs(raw, ops)
    assert report["operator_sum_ms"] > 0 and report["operator_occurrences"] > 0


def test_customize_model_spec_layer_cycling_and_router_group_clamping():
    cycled = customize_model_spec(
        "deepseek-v3.2-prune-3l", layers=5, n_routed_experts=4, num_experts_per_tok=2,
    )
    assert cycled.layers == 5
    assert [p.index for p in cycled.layer_plans] == [0, 1, 2, 3, 4]
    assert [p.ffn for p in cycled.layer_plans] == ["dense", "moe", "moe", "moe", "moe"]
    # n_routed_experts=4 < original n_group=8 updates router_groups safely
    assert all(p.router_groups <= 4 for p in cycled.layer_plans if p.ffn == "moe")

    with pytest.raises(ModelingError):
        customize_model_spec("qwen3-30b-a3b", layers=0)
    with pytest.raises(ModelingError):
        customize_model_spec("qwen3-30b-a3b", n_routed_experts=8, num_experts_per_tok=16)


def test_resolve_role_exclusions_deterministic_mapping_and_quant_resolution():
    patterns = [
        "lm_head",
        "*lm_head*",
        "model.embed_tokens*",
        "*norm*",
        "*.mlp.gate",
        "*.mlp.shared_expert_gate",
        "*self_attn*",
        "*shared_experts*",
        "*mlp.experts*",
        "model.layers.0.unknown_custom_proj",
    ]
    roles, unresolved = resolve_role_exclusions(patterns)
    assert roles == (
        "lm_head",
        "embedding",
        "norm",
        "router",
        "attention",
        "shared_expert",
        "routed_expert",
    )
    assert unresolved == ("model.layers.0.unknown_custom_proj",)

    # Standard non-linear exclusions resolve cleanly without proxy degradation.
    clean_cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": [
                "lm_head",
                "model.embed_tokens*",
                "*norm*",
                "model.layers.*.mlp.gate",
                "*.mlp.shared_expert_gate",
            ],
        },
    }
    quant, gaps = resolve_quant(clean_cfg)
    assert gaps == []
    assert quant.name == "hf_fp8_block"
    assert quant.weight_format == "fp8_e4m3"
    assert set(quant.exclusions) >= {"lm_head", "embedding", "norm", "router"}

    # Linear submodule exclusions (e.g. attention) still trigger explicit proxy gap.
    linear_excluded_cfg = {
        "torch_dtype": "bfloat16",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "activation_scheme": "dynamic",
            "weight_block_size": [128, 128],
            "modules_to_not_convert": ["lm_head", "*self_attn*"],
        },
    }
    quant_proxy, gaps_proxy = resolve_quant(linear_excluded_cfg)
    assert "quant_module_exclusions_unresolved" in gaps_proxy
    assert quant_proxy.weight_format == "bf16"

    # Compound regex alternation and multi-role patterns must stay unresolved.
    compound_roles, compound_unresolved = resolve_role_exclusions(
        ["re:.*(lm_head|self_attn).*", "*lm_head*self_attn*"]
    )
    assert compound_roles == ()
    assert compound_unresolved == ("re:.*(lm_head|self_attn).*", "*lm_head*self_attn*")

    compound_spec = build_model_spec(
        "compound-regex-test",
        {
            "model_type": "qwen3",
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "vocab_size": 256,
            "torch_dtype": "bfloat16",
            "quantization_config": {
                "quant_method": "fp8",
                "fmt": "e4m3",
                "activation_scheme": "dynamic",
                "weight_block_size": [128, 128],
                "modules_to_not_convert": ["re:.*(lm_head|self_attn).*"],
            },
        },
    )
    compound_cov = model_coverage(compound_spec)
    assert compound_cov["model_coverage"] == "partial"
    assert "quant_module_exclusions_unresolved" in compound_cov["unmodeled_features"]
    assert compound_spec.quant.weight_format == "bf16"
