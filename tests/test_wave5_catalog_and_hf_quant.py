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
    "qwen3-30b-a3b": ("Qwen3-30B-A3B", "qwen30ba3b"),
    "qwen3.5-27b": ("Qwen3.5-27B", "qwen35_27b"),
}


@pytest.mark.parametrize("key,aliases", list(WAVE5_MODELS.items()))
def test_wave5_catalog_entries_and_aliases_load_complete_with_positive_weights(key, aliases):
    assert key in CATALOG
    spec = load_model(key)
    cov = model_coverage(spec)
    if key == "deepseek-v3.2-prune-3l":
        assert cov["model_coverage"] == "partial"
        assert set(cov["unmodeled_features"]) == {
            "router_noaux_tc_correction_bias_not_in_layer_plan",
            "router_noaux_tc_group_score_top2_not_in_layer_plan",
        }
    else:
        assert cov["model_coverage"] == "complete", f"{key} had unmodeled gaps: {cov}"
    params = body_parameter_count(spec)
    weights = body_weight_bytes(spec)
    assert params > 0
    assert weights >= params * 2.0
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

    qwen30b = load_model("qwen3-30b-a3b")
    assert qwen30b.layers == 48
    assert qwen30b.hidden_size == 2048
    assert qwen30b.num_experts == 128 and qwen30b.top_k == 8
    assert all(p.qk_norm and p.ffn == "moe" for p in qwen30b.layer_plans)

    qwen35 = load_model("qwen3.5-27b")
    assert qwen35.layers == 64
    assert qwen35.hidden_size == 5120
    assert qwen35.intermediate_size == 17408
    assert sum(1 for p in qwen35.layer_plans if p.attention == "gdn") == 48
    assert sum(1 for p in qwen35.layer_plans if p.attention == "gqa") == 16


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
