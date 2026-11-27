"""Standard dense/MoE/MLA/VLM HF family adapters against config geometry oracles.

[Role]
  Lower the retained standard-family HF fixtures (DeepSeek, GLM, Qwen, Kimi)
  through the real catalog adapters and compare every layer schedule, cache
  width, expert/router term and weight shape with arithmetic derived directly
  from the raw JSON config; then execute each shrunk fixture through the public
  serving graph.

[Invariants]
  * Schedules are computed from ``first_k_dense_replace``/``moe_layer_freq``,
    ``decoder_sparse_step``/``mlp_only_layers`` and explicit ``layer_types``,
    never from catalog names or layer-count guesses.
  * MLA/KV widths, partial-RoPE splits, expert banks, shared banks and router
    biases are independent byte/shape conservation statements.
  * Optional MTP/vision/draft mechanisms are explicit coverage gaps, never
    silently absorbed into the body.
  * Shrunk real fixtures reach the same public frontend/engine and preserve
    work with zero live tasks after execution.

[Timeouts & Locks]
  Bounded fixture parsing and small in-memory graphs; no device, network,
  subprocess or locks.

[Side-Effect Policy]
  Read-only repository fixtures; all execution is in memory.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from _support.hf_configs import case, family_config, small_case
from mindie_sim import graph_costs, target_graphs, target_operators
from mindie_sim.catalog.deepseek import (
    _aux_standard_deepseek, _standard_deepseek_gaps, _standard_deepseek_plans,
)
from mindie_sim.catalog.glm import _aux_standard_glm, _standard_glm_gaps, _standard_glm_plans
from mindie_sim.catalog.kimi import (
    _aux_standard_kimi, _standard_kimi_gaps, _standard_kimi_plans,
)
from mindie_sim.catalog.qwen import (
    _aux_standard_qwen, _standard_qwen_gaps, _standard_qwen_plans,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.structure import parse_case
from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.structure import export_structure

pytestmark = pytest.mark.tier0

DEEPSEEK = ("deepseek-v2", "deepseek-v2-lite", "deepseek-v3", "deepseek-r1")
GLM = ("glm4-0414", "glm4-chat", "glm45", "glm45-air")
QWEN = ("qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b", "qwen3-8b", "qwen3-30b-a3b")
KIMI = ("kimi-k2-instruct", "kimi-vl-a3b-instruct")
STANDARD = QWEN + KIMI


def _flat(config: dict) -> dict:
    text = config.get("text_config")
    return {**config, **text} if isinstance(text, dict) else dict(config)


def _model(name: str, config: dict, plans) -> ModelSpec:
    merged = _flat(config)
    return ModelSpec(
        name=name, layers=len(plans), hidden_size=merged["hidden_size"],
        num_heads=plans[0].num_heads, head_dim=plans[0].head_dim,
        intermediate_size=merged.get("intermediate_size") or merged.get("ffn_hidden_size"),
        vocab_size=merged.get("vocab_size", merged.get("padded_vocab_size")),
        attention=plans[0].attention, ffn=plans[0].ffn,
        num_kv_heads=plans[0].num_kv_heads, quantization="bf16",
        layer_plans=tuple(plans), body_layers=len(plans))


def _terms(plan, spec) -> dict:
    return {term.name: term for term in layer_weight_terms(plan, spec)}


def _modulo_moe_layers(config: dict) -> list[int]:
    merged = _flat(config)
    first = int(merged["first_k_dense_replace"])
    freq = int(merged["moe_layer_freq"])
    return [index for index in range(int(merged["num_hidden_layers"]))
            if index >= first and index % freq == 0]


def _qwen_moe_layers(config: dict) -> list[int]:
    layers = int(config["num_hidden_layers"])
    if int(config.get("num_experts", 0) or 0) <= 0:
        return []
    dense_only = {int(value) for value in (config.get("mlp_only_layers") or ())}
    step = int(config.get("decoder_sparse_step", 1))
    return [index for index in range(layers)
            if index not in dense_only and (index + 1) % step == 0]


@pytest.mark.parametrize("name", DEEPSEEK)
def test_standard_mla_family_schedule_cache_and_weight_arithmetic(name: str) -> None:
    config = family_config(name)
    plans = _standard_deepseek_plans(config)
    moe_layers = _modulo_moe_layers(config)
    assert len(plans) == config["num_hidden_layers"]
    assert [plan.index for plan in plans if plan.ffn == "moe"] == moe_layers
    qk = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]
    for plan in plans:
        assert plan.attention == "mla" and plan.mla_variant == "latent_mla"
        assert plan.num_kv_heads == 1
        assert (plan.head_dim, plan.qk_nope_head_dim, plan.qk_rope_head_dim) == (
            qk, config["qk_nope_head_dim"], config["qk_rope_head_dim"])
        assert plan.kv_cache_elements == config["kv_lora_rank"] + config["qk_rope_head_dim"]
        assert plan.q_lora_rank == (config.get("q_lora_rank") or 0)
        assert (plan.router_dtype, plan.router_mode) == ("fp32", config["scoring_func"])
        if plan.ffn == "dense":
            assert (plan.intermediate_size, plan.num_experts) == (
                config["intermediate_size"], 0)
        else:
            assert plan.intermediate_size == 0
            assert (plan.num_experts, plan.top_k) == (
                config["n_routed_experts"], config["num_experts_per_tok"])
            assert plan.moe_intermediate_size == config["moe_intermediate_size"]
            assert plan.shared_intermediate_size == (
                config["n_shared_experts"] * config["moe_intermediate_size"])
    model = _model(name, config, plans)
    dense, moe = _terms(plans[0], model), _terms(plans[moe_layers[0]], model)
    assert (dense["dense_gate"].rows, dense["dense_down"].rows) == (
        config["intermediate_size"], config["hidden_size"])
    assert (moe["router"].rows, moe["router"].cols) == (
        config["n_routed_experts"], config["hidden_size"])
    assert moe["expert_gate"].count == config["n_routed_experts"]
    assert moe["expert_gate"].rows == config["moe_intermediate_size"]
    if config.get("q_lora_rank") is None:
        assert (moe["q_proj"].rows, moe["q_proj"].cols) == (
            config["num_attention_heads"] * qk, config["hidden_size"])
        assert not {"q_a", "q_b", "q_a_norm"} & moe.keys()
    else:
        rank = config["q_lora_rank"]
        assert (moe["q_a"].rows, moe["q_a"].cols) == (rank, config["hidden_size"])
        assert (moe["q_b"].rows, moe["q_b"].cols) == (
            config["num_attention_heads"] * qk, rank)
        assert moe["q_a_norm"].rows == rank
    modulo = family_config(name)
    modulo["moe_layer_freq"] = 2
    changed = _standard_deepseek_plans(modulo)
    first = int(modulo["first_k_dense_replace"])
    assert [plan.index for plan in changed if plan.ffn == "moe"] == [
        index for index in range(len(changed)) if index >= first and index % 2 == 0]
    assert [plan.index for plan in changed if plan.ffn == "dense"] == [
        index for index in range(len(changed)) if not (index >= first and index % 2 == 0)]


def test_noaux_tc_router_declares_bias_and_charges_physical_vector_work() -> None:
    config = family_config("deepseek-v3")
    gaps = _standard_deepseek_gaps(config)
    assert not any("noaux_tc" in gap or "routed_scaling" in gap for gap in gaps)
    assert "optional_mtp_not_modeled" in gaps
    assert _aux_standard_deepseek(config) is None
    plans = _standard_deepseek_plans(config)
    moe = next(plan for plan in plans if plan.ffn == "moe")
    assert moe.router_noaux_tc is True
    assert (moe.router_groups, moe.router_group_topk) == (
        config["n_group"], config["topk_group"])
    assert moe.routed_scaling_factor == config["routed_scaling_factor"]
    bias = _terms(moe, _model("deepseek-v3", config, plans))["router_correction_bias"]
    assert (bias.rows, bias.cols) == (config["n_routed_experts"], 1)
    assert bias.rows * bias.cols * 4 == config["n_routed_experts"] * 4
    lite = family_config("deepseek-v2-lite")
    assert _standard_deepseek_gaps(lite) == ()
    assert _aux_standard_deepseek(lite) is not None

    plan = LayerPlan(
        index=0, attention="mla", architecture="mla_standard", ffn="moe",
        num_heads=4, num_kv_heads=1, head_dim=24, kv_lora_rank=32,
        qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        mla_variant="latent_mla", kv_cache_elements=40, kv_owner_layer_id=0,
        owns_long_kv=True, num_experts=8, top_k=2, moe_intermediate_size=16,
        router_input_size=32, router_mode="sigmoid", router_dtype="fp32",
        router_norm_topk=True, router_groups=4, router_group_topk=2,
        router_noaux_tc=True, routed_scaling_factor=2.5, native_layer_id=0, scope="body")
    raw = case(prompt=3, output=1)
    raw["model"] = {"name": "direct-noaux", "layers": 1, "hidden_size": 32, "num_heads": 4,
                    "num_kv_heads": 1, "head_dim": 24, "intermediate_size": 16,
                    "vocab_size": 128, "attention": "mla", "ffn": "moe",
                    "layer_plans": [plan.to_dict()]}
    tokens = 3
    ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=tokens)}
    scores, biased = ops["s0.l0.router_scores"], ops["s0.l0.router_bias"]
    group_score, group_topk = ops["s0.l0.router_group_score"], ops["s0.l0.router_group_topk"]
    mask, topk = ops["s0.l0.router_group_mask"], ops["s0.l0.router_expert_topk"]
    gather, scale = ops["s0.l0.router_weight_gather"], ops["s0.l0.route_scale"]
    experts = plan.num_experts
    assert scores.vector_ops == 4 * tokens * experts
    assert biased.vector_ops == tokens * experts
    assert biased.mte2_bytes == (tokens + 1) * experts * 4
    assert dict(biased.metadata)["correction_bias_bytes"] == experts * 4
    assert group_score.mte2_bytes == tokens * experts * 4
    assert group_score.mte3_bytes == tokens * plan.router_groups * 4
    assert group_score.vector_ops == 3 * tokens * (experts - plan.router_groups)
    assert group_topk.vector_ops == 2 * tokens * plan.router_groups
    assert mask.vector_ops == tokens * experts and topk.vector_ops > 0
    # Biased scores never become weights: the gather reads the raw sigmoid.
    assert gather.deps[1] == scores.op_id
    assert dict(gather.metadata)["weight_source"] == "unbiased_sigmoid_scores"
    assert dict(scale.metadata)["routed_scaling_factor"] == 2.5
    assert dict(ops["s0.l0.pack"].metadata)["topk_ids_producer"] == topk.op_id
    assert "s0.l0.group_select" not in ops


@pytest.mark.parametrize("name", GLM)
def test_standard_grouped_query_family_schedule_cache_and_weight_arithmetic(name: str) -> None:
    config = family_config(name)
    plans = _standard_glm_plans(config)
    assert len(plans) == config.get("num_hidden_layers", config.get("num_layers"))
    moe_model = config["model_type"] == "glm4_moe"
    first = int(config.get("first_k_dense_replace", 0)) if moe_model else len(plans)
    expected_moe = [index for index in range(len(plans))
                    if moe_model and index >= first and config.get("n_routed_experts") is not None]
    assert [plan.index for plan in plans if plan.ffn == "moe"] == expected_moe
    for plan in plans:
        assert plan.attention == "gqa"
        assert plan.head_dim == config.get("head_dim", config.get("kv_channels"))
        assert plan.num_heads == config["num_attention_heads"]
        assert plan.num_kv_heads == config.get(
            "num_key_value_heads", config.get("multi_query_group_num"))
        assert plan.kv_cache_elements == 2 * plan.num_kv_heads * plan.head_dim
        if plan.ffn == "dense":
            assert plan.intermediate_size == config.get(
                "intermediate_size", config.get("ffn_hidden_size"))
        else:
            assert plan.num_experts == config["n_routed_experts"]
            assert plan.moe_intermediate_size == config["moe_intermediate_size"]
            assert plan.shared_intermediate_size == (
                config["n_shared_experts"] * config["moe_intermediate_size"])
        if config["model_type"] != "chatglm":
            assert plan.qk_norm is bool(config.get("use_qk_norm", False))
            if plan.ffn == "moe":
                assert (plan.router_mode, plan.router_dtype) == ("sigmoid", "fp32")
    model = _model(name, config, plans)
    terms = _terms(plans[0], model)
    assert (terms["q_proj"].rows, terms["q_proj"].cols) == (
        config["num_attention_heads"] * model.head_dim, config["hidden_size"])
    if expected_moe:
        moe_terms = _terms(plans[expected_moe[0]], model)
        assert moe_terms["router"].rows == config["n_routed_experts"]
        assert moe_terms["expert_gate"].count == config["n_routed_experts"]
    else:
        assert terms["dense_gate"].rows == model.intermediate_size


def test_standard_family_gaps_optional_mtp_and_retained_flags_stay_visible() -> None:
    modern = _standard_glm_gaps(family_config("glm4-0414"))
    assert "additional_post_self_attn_and_post_mlp_norms_not_in_layer_schema" in modern
    assert "partial_rope_64_of_128_not_in_attention_plan" in modern
    assert "attention_bias_qkv_not_in_layer_plan" in modern
    assert _aux_standard_glm(family_config("glm4-0414")) is not None
    moe = _standard_glm_gaps(family_config("glm45"))
    assert "router_sigmoid_correction_bias_and_routed_scaling_not_in_layer_plan" in moe
    assert "optional_mtp_not_modeled" in moe
    assert _aux_standard_glm(family_config("glm45")) is None
    assert _aux_standard_glm(family_config("glm45-air")) is None
    legacy = _standard_glm_gaps(family_config("glm4-chat"))
    assert "attention_bias_qkv_not_in_layer_plan" in legacy
    assert "legacy_layernorm_not_rmsnorm_not_modeled" not in legacy
    changed = family_config("glm4-chat")
    changed.update(rmsnorm=False, add_bias_linear=True,
                   apply_residual_connection_post_layernorm=True)
    assert len(_standard_glm_plans(changed)) == changed["num_layers"]
    assert set(_standard_glm_gaps(changed)) >= {
        "legacy_layernorm_not_rmsnorm_not_modeled",
        "linear_bias_not_in_layer_plan",
        "residual_connection_post_layernorm_not_modeled"}


@pytest.mark.parametrize("name", QWEN)
def test_standard_dense_and_moe_family_geometry_and_weights(name: str) -> None:
    config = family_config(name)
    plans = _standard_qwen_plans(config)
    hidden, heads, kv_heads = (config["hidden_size"], config["num_attention_heads"],
                               config["num_key_value_heads"])
    if config["model_type"] == "qwen2_moe" or config.get("head_dim") is None:
        head_dim = hidden // heads
    else:
        head_dim = int(config["head_dim"])
    moe_layers = _qwen_moe_layers(config)
    qk_norm = config["model_type"] in {"qwen3", "qwen3_moe"}
    assert [plan.index for plan in plans] == list(range(config["num_hidden_layers"]))
    for plan in plans:
        assert plan.attention == "gqa"
        assert (plan.num_heads, plan.num_kv_heads, plan.head_dim) == (heads, kv_heads, head_dim)
        assert plan.qk_norm is qk_norm and plan.window == 0
        assert plan.kv_cache_elements == 2 * kv_heads * head_dim
        assert plan.kv_owner_layer_id == plan.index and plan.expert_latent_size == 0
        if plan.index in moe_layers:
            assert plan.ffn == "moe" and plan.intermediate_size == 0
            assert (plan.num_experts, plan.top_k) == (
                config["num_experts"], config["num_experts_per_tok"])
            assert plan.moe_intermediate_size == config["moe_intermediate_size"]
            assert plan.router_mode == "softmax"
            assert plan.router_norm_topk is bool(config.get("norm_topk_prob", False))
            if config["model_type"] == "qwen2_moe":
                assert (plan.num_shared_experts, plan.shared_expert_gate) == (1, True)
                assert plan.shared_intermediate_size == config["shared_expert_intermediate_size"]
            else:
                assert (plan.num_shared_experts, plan.shared_intermediate_size) == (0, 0)
                assert plan.shared_expert_gate is False
        else:
            assert plan.ffn == "dense" and plan.intermediate_size == config["intermediate_size"]
            assert plan.num_experts == plan.top_k == plan.num_shared_experts == 0
    assert [plan.index for plan in plans if plan.ffn == "moe"] == moe_layers
    model = _model(name, config, plans)
    for plan in plans:
        layer = _terms(plan, model)
        assert (layer["q_proj"].rows, layer["q_proj"].cols) == (heads * head_dim, hidden)
        assert (layer["k_proj"].rows, layer["v_proj"].rows) == (kv_heads * head_dim,) * 2
        assert (layer["o_proj"].rows, layer["o_proj"].cols) == (hidden, heads * head_dim)
        assert (("q_norm" in layer), ("k_norm" in layer)) == (qk_norm, qk_norm)
        if plan.ffn == "dense":
            assert (layer["dense_gate"].rows, layer["dense_down"].cols) == (
                config["intermediate_size"], config["intermediate_size"])
        else:
            assert (layer["router"].rows, layer["router"].cols) == (config["num_experts"], hidden)
            assert layer["expert_gate"].count == config["num_experts"]
            assert (layer["expert_gate"].rows, layer["expert_gate"].cols) == (
                config["moe_intermediate_size"], hidden)
            shared = plan.shared_intermediate_size
            assert ("shared_gate" in layer) == (shared > 0)
            if shared:
                assert layer["shared_gate"].rows == shared
    if moe_layers:
        router = _terms(plans[0], model)["router"]
        assert term_bytes(router, QuantSpec("bf16"), ParallelSpec(), 1) == (
            hidden * config["num_experts"] * 2)


def test_explicit_head_dim_schedule_and_window_overrides_are_declared() -> None:
    config = family_config("qwen3-30b-a3b")
    assert (config["hidden_size"], config["num_attention_heads"]) == (2048, 32)
    assert config["head_dim"] == 128 and _standard_qwen_plans(config)[0].head_dim == 128
    assert _standard_qwen_plans(config)[0].head_dim != 2048 // 32
    model = _model("qwen3-30b-a3b", config, _standard_qwen_plans(config))
    assert _terms(_standard_qwen_plans(config)[0], model)["q_proj"].rows == 32 * 128 == 4096
    assert _standard_qwen_plans(family_config("qwen2-7b"))[0].head_dim == 3584 // 28 == 128

    stepped = dict(family_config("qwen3-30b-a3b"))
    stepped.update(decoder_sparse_step=3, mlp_only_layers=[2, 5, 8, 11])
    plans = _standard_qwen_plans(stepped)
    expected = {index for index in range(48) if (index + 1) % 3 == 0} - {2, 5, 8, 11}
    assert {plan.index for plan in plans if plan.ffn == "moe"} == expected
    assert all(plans[index].ffn == "dense" for index in (2, 5, 8, 11))

    windowed = dict(family_config("qwen2-7b"))
    windowed.update(use_sliding_window=True, sliding_window=4096, max_window_layers=20)
    assert [plan.window for plan in _standard_qwen_plans(windowed)] == [0] * 20 + [4096] * 8
    assert not any("sliding" in gap for gap in _standard_qwen_gaps(windowed))
    windowed["max_window_layers"] = 28
    assert all(plan.window == 0 for plan in _standard_qwen_plans(windowed))
    explicit = dict(family_config("qwen3-8b"))
    explicit.update(sliding_window=8, max_window_layers=0, layer_types=(
        ["full_attention"] * 2 + ["sliding_attention"] * 2 + ["full_attention"] * 32))
    plans = _standard_qwen_plans(explicit)
    assert [plan.window for plan in plans] == [0, 0, 8, 8] + [0] * 32


def test_grouped_query_gaps_aux_and_unknown_attention_stay_explicit() -> None:
    for name in ("qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b"):
        assert "qkv_bias_unmodeled" in _standard_qwen_gaps(family_config(name))
    assert not any("bias" in gap for gap in _standard_qwen_gaps(family_config("qwen3-8b")))
    biased = dict(family_config("qwen3-8b"))
    biased["attention_bias"] = True
    assert "attention_bias_unmodeled" in _standard_qwen_gaps(biased)
    no_bias = dict(family_config("qwen2-57b-a14b"))
    no_bias["qkv_bias"] = False
    assert "qkv_bias_unmodeled" not in _standard_qwen_gaps(no_bias)
    activated = dict(family_config("qwen3-8b"))
    activated["hidden_act"] = "gelu"
    assert _standard_qwen_plans(activated)[0].ffn_activation == "gelu"
    assert "hidden_act_unsupported:gelu" in _standard_qwen_gaps(activated)
    for name in QWEN:
        aux = _aux_standard_qwen(family_config(name))
        assert (aux.mtp, aux.dspark, aux.external_draft_key) == (None, None, "")
    declared = dict(family_config("qwen3-30b-a3b"))
    declared["mtp_num_hidden_layers"] = 1
    assert _aux_standard_qwen(declared).mtp is None
    assert "optional_mtp_schema_not_modeled" in _standard_qwen_gaps(declared)
    undeclared = dict(family_config("qwen3-8b"))
    undeclared["num_nextn_predict_layers"] = 1
    assert _aux_standard_qwen(undeclared).mtp is None
    assert "optional_mtp_schema_not_modeled" in _standard_qwen_gaps(undeclared)

    unknown = dict(family_config("qwen3-8b"))
    unknown["layer_types"] = ["custom_attention"] * unknown["num_hidden_layers"]
    unknown.pop("num_attention_heads")
    plans = _standard_qwen_plans(unknown)
    assert all(plan.attention == "unmodeled" for plan in plans)
    assert all(plan.intermediate_size == unknown["intermediate_size"] for plan in plans)
    assert len(_standard_qwen_gaps(unknown)) == unknown["num_hidden_layers"]
    short = dict(family_config("qwen3-8b"))
    short["layer_types"] = ["custom_attention", "full_attention"]
    plans = _standard_qwen_plans(short)
    assert plans[1].attention == "gqa"
    assert all(plan.attention == "projection_only" for plan in plans[2:])
    assert all(plan.cache_kind == "none" for plan in plans[2:])
    missing = dict(family_config("qwen2-7b"))
    missing.update(use_sliding_window=True, sliding_window=4096)
    missing.pop("max_window_layers", None)
    assert all(plan.attention == "projection_only" for plan in _standard_qwen_plans(missing))
    assert "sliding_window_max_window_layers_missing" in _standard_qwen_gaps(missing)


@pytest.mark.parametrize("name", KIMI)
def test_standard_mla_vlm_family_geometry_and_weight_terms(name: str) -> None:
    config = family_config(name)
    merged = _flat(config)
    plans = _standard_kimi_plans(config)
    expected = _modulo_moe_layers(config)
    assert len(plans) == merged["num_hidden_layers"]
    assert [plan.index for plan in plans if plan.ffn == "moe"] == expected
    qk = merged["qk_nope_head_dim"] + merged["qk_rope_head_dim"]
    for plan in plans:
        assert (plan.attention, plan.architecture) == ("mla", "mla_standard")
        assert plan.mla_variant == "latent_mla" and plan.num_kv_heads == 1
        assert (plan.head_dim, plan.qk_nope_head_dim, plan.qk_rope_head_dim) == (
            qk, merged["qk_nope_head_dim"], merged["qk_rope_head_dim"])
        assert plan.kv_lora_rank == merged["kv_lora_rank"]
        assert plan.q_lora_rank == (merged.get("q_lora_rank") or 0)
        assert plan.kv_cache_elements == merged["kv_lora_rank"] + merged["qk_rope_head_dim"]
        assert plan.kv_owner_layer_id == plan.index and plan.owns_long_kv is True
        assert (plan.router_mode, plan.router_norm_topk) == (
            merged.get("scoring_func", "softmax"), bool(merged.get("norm_topk_prob", False)))
        assert plan.router_noaux_tc is (merged.get("topk_method") == "noaux_tc")
        assert plan.routed_scaling_factor == float(merged.get("routed_scaling_factor") or 1.0)
        # Standard text bodies must not inherit the hybrid K3 mechanisms.
        assert plan.attention not in {"kda", "gdn"}
        assert (plan.linear_heads, plan.conv_kernel, plan.state_bytes) == (0, 0, 0)
        assert plan.hyper_connections is False and plan.hc_mult == 0
        if plan.ffn == "moe":
            assert plan.intermediate_size == 0
            assert (plan.num_experts, plan.top_k) == (
                merged["n_routed_experts"], merged["num_experts_per_tok"])
            assert plan.shared_intermediate_size == (
                merged["n_shared_experts"] * merged["moe_intermediate_size"])
        else:
            assert plan.intermediate_size == merged["intermediate_size"]
            assert plan.num_experts == plan.top_k == 0
    model = _model(name, config, plans)
    hidden, heads = merged["hidden_size"], merged["num_attention_heads"]
    dense, sparse = _terms(plans[0], model), _terms(plans[expected[0]], model)
    if merged.get("q_lora_rank"):
        assert (dense["q_a"].rows, dense["q_a"].cols) == (merged["q_lora_rank"], hidden)
        assert (dense["q_b"].rows, dense["q_b"].cols) == (heads * qk, merged["q_lora_rank"])
        assert "q_proj" not in dense
    else:
        assert (dense["q_proj"].rows, dense["q_proj"].cols) == (heads * qk, hidden)
        assert not {"q_a", "q_b", "q_a_norm"} & dense.keys()
    assert (dense["kv_a"].rows, dense["kv_a"].cols) == (
        merged["kv_lora_rank"] + merged["qk_rope_head_dim"], hidden)
    assert (dense["kv_b"].rows, dense["kv_b"].cols) == (
        heads * (merged["qk_nope_head_dim"] + merged["v_head_dim"]), merged["kv_lora_rank"])
    assert (dense["o_proj"].rows, dense["o_proj"].cols) == (hidden, heads * merged["v_head_dim"])
    shared = merged["n_shared_experts"] * merged["moe_intermediate_size"]
    assert (sparse["shared_gate"].rows, sparse["shared_gate"].cols) == (shared, hidden)
    assert {"shared_router", "kda_q", "gdn_qkvz"}.isdisjoint(sparse)
    bias = sparse["router_correction_bias"]
    assert (bias.rows, bias.cols, bias.category) == (merged["n_routed_experts"], 1, "param_fp32")


@pytest.mark.parametrize("name", KIMI)
def test_vlm_family_gaps_vision_and_normalized_text_type_stay_explicit(name: str) -> None:
    config = family_config(name)
    merged = _flat(config)
    gaps = _standard_kimi_gaps(config)
    assert not any("noaux_tc" in gap or "routed_scaling" in gap for gap in gaps)
    assert not any("fp8" in gap.lower() or "quant" in gap.lower() for gap in gaps)
    aux = _aux_standard_kimi(config)
    assert (aux.mtp, aux.dspark, aux.external_draft_key) == (None, None, "")
    if name == "kimi-vl-a3b-instruct":
        assert "vision_tower_unmodeled" in gaps and "vision_projector_unmodeled" in gaps
        assert merged["q_lora_rank"] is None
        normalized = _flat(config)
        normalized["model_type"] = "deepseek_v3"
        assert _standard_kimi_plans(normalized) == _standard_kimi_plans(config)
        assert "vision_tower_unmodeled" in _standard_kimi_gaps(normalized)
    else:
        assert "vision_tower_unmodeled" not in gaps
    assert "optional_mtp_not_modeled" not in gaps
    custom = dict(config)
    custom.update(linear_attn_config={"full_attn_layers": [1]},
                  routed_expert_hidden_size=1024, dspark_block_size=7)
    custom_gaps = _standard_kimi_gaps(custom)
    assert "kda_fields_present_in_standard_kimi_config" in custom_gaps
    assert "latent_expert_fields_present_in_standard_kimi_config" in custom_gaps
    assert "dspark_fields_present_in_standard_kimi_config" in custom_gaps


def _direct_mla_case(q_lora_rank: int):
    plan = LayerPlan(
        index=0, attention="mla", architecture="mla_standard", ffn="dense",
        num_heads=4, num_kv_heads=1, head_dim=24, q_lora_rank=q_lora_rank,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        mla_variant="latent_mla", kv_cache_elements=40, kv_owner_layer_id=0,
        owns_long_kv=True, intermediate_size=64, native_layer_id=0, scope="body")
    raw = case(prompt=3, output=1)
    raw["model"] = {"name": "direct-mla", "layers": 1, "hidden_size": 32, "num_heads": 4,
                    "num_kv_heads": 1, "head_dim": 24, "intermediate_size": 64,
                    "vocab_size": 128, "attention": "mla", "ffn": "dense",
                    "layer_plans": [plan.to_dict()]}
    return plan, raw


def test_direct_and_lowrank_mla_queries_are_mutually_exclusive() -> None:
    plan, raw = _direct_mla_case(0)
    ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=3)}
    q = ops["s0.l0.q_proj"]
    assert q.semantic_shape == (3, 4 * 24, 32) and q.flops == 2 * 3 * 4 * 24 * 32
    assert not any(op_id.endswith((".q_a", ".q_a_norm", ".q_norm", ".q_b")) for op_id in ops)
    report = graph_costs(raw, tuple(ops.values()), input_ready_ms=7,
                         external_inputs={"analysis_input": 0}, trace_limit=100)
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    model = ModelSpec(name="direct-mla", layers=1, hidden_size=32, num_heads=4, head_dim=24,
                      intermediate_size=64, vocab_size=128, attention="mla",
                      num_kv_heads=1, layer_plans=(plan,), body_layers=1)
    terms = _terms(plan, model)
    assert (terms["q_proj"].rows, terms["q_proj"].cols) == (96, 32)
    assert not {"q_a", "q_b", "q_a_norm"} & terms.keys()
    lowrank_plan, lowrank_raw = _direct_mla_case(8)
    lowrank = {op.op_id: op for op in target_operators(lowrank_raw, phase="prefill", tokens=3)}
    assert {"s0.l0.q_a", "s0.l0.q_norm", "s0.l0.q_b"} <= lowrank.keys()
    assert "s0.l0.q_proj" not in lowrank
    assert lowrank["s0.l0.q_a"].flops == 2 * 3 * 8 * 32
    assert lowrank["s0.l0.q_b"].flops == 2 * 3 * 96 * 8
    lowrank_terms = _terms(lowrank_plan, replace(model, name="lowrank-mla",
                                                 layer_plans=(lowrank_plan,)))
    assert lowrank_terms["q_a_norm"].rows == 8
    assert (lowrank_terms["q_a"].rows, lowrank_terms["q_a"].cols) == (8, 32)
    assert (lowrank_terms["q_b"].rows, lowrank_terms["q_b"].cols) == (96, 8)
    _, refused = _direct_mla_case(0)
    refused_plan = replace(plan, head_dim=192, kv_lora_rank=512, qk_nope_head_dim=128,
                           qk_rope_head_dim=64, v_head_dim=128, kv_cache_elements=576)
    refused["model"]["head_dim"] = 192
    refused["model"]["layer_plans"] = [refused_plan.to_dict()]
    refused["runtime"]["kernels"] = {"lowerings": {"mla_prolog": "vllm_ascend_mla_prolog_v3"}}
    with pytest.raises(ModelingError, match="mla_prolog_v3_direct_q_unsupported"):
        target_operators(refused, phase="prefill", tokens=3)


@pytest.mark.parametrize("name", STANDARD)
@pytest.mark.parametrize("phase,tokens,context", (("prefill", 4, 0), ("decode", 1, 8)))
def test_shrunk_fixture_runs_public_graph_and_preserves_work(
    name: str, phase: str, tokens: int, context: int,
) -> None:
    raw = small_case(name)
    parsed = parse_case(raw)
    graphs = target_graphs(raw, phase=phase, tokens=tokens, context=context)
    ops = tuple(op for group in graphs.values() for op in group)
    if name.startswith("qwen"):
        width = 16 if name.startswith("qwen3") else 8
        projections = [op for op in ops if op.family == "qkv_projection"]
        assert len(projections) == 2
        assert sum(op.flops for op in projections) == 2 * 2 * tokens * 32 * 8 * width
        assert len([op for op in ops if op.recipe in {"q_norm", "k_norm"}]) == (
            4 if name.startswith("qwen3") else 0)
        assert all(op.dtype == "bf16" for op in ops if op.family == "router")
        assert not any(op.op_id.endswith("router_cast") for op in ops)
    else:
        assert all(plan.expert_latent_size == 0 and plan.attention == "mla"
                   for plan in parsed.model.layer_plans)
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


def test_custom_attention_retains_known_work_without_claiming_a_gqa_cache() -> None:
    raw = small_case("qwen3-8b")
    raw["model"]["layer_types"] = ["full_attention", "custom_attention"]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    unknown = [op for ops in graphs.values() for op in ops if op.layer == 1]
    assert any(op.family == "qkv_projection" for op in unknown)
    assert any(op.recipe == "dense_gate_up" for op in unknown)
    assert any(op.recipe == "unmodeled_attention_core" for op in unknown)
    assert not any(op.recipe in {"kv_store", "flash_attention"} for op in unknown)
    structure = export_structure(raw, phase="prefill", tokens=3)
    hints = structure["unmodeled_structure_onboarding_hints"]
    assert hints and hints[0]["schema"] == "mindie-unmodeled-structure-hint/1"
    with pytest.raises(UnmodeledCapabilityError) as raised:
        graph_costs(raw, graphs)
    assert raised.value.onboarding_hint["ssot_owner_repo"] == "mindie-sim"
    assert raised.value.result_available is False
