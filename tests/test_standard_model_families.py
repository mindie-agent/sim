"""Structural adapters for standard DeepSeek V2/V3 and GLM-4/4.5 configs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.deepseek import (
    _aux_standard_deepseek, _standard_deepseek_gaps, _standard_deepseek_plans,
)
from mindie_sim.catalog.glm import (
    _aux_standard_glm, _standard_glm_gaps, _standard_glm_plans,
)
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import target_operators
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.model import LayerPlan, ModelSpec

from test_prediction_system import case

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/hf-standard-families"
DEEPSEEK = ("deepseek-v2", "deepseek-v2-lite", "deepseek-v3", "deepseek-r1")
GLM = ("glm4-0414", "glm4-chat", "glm45", "glm45-air")


def config(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def _model(name: str, plans, dense_i: int) -> ModelSpec:
    cfg = config(name)
    first = plans[0]
    return ModelSpec(
        name=name, layers=len(plans), hidden_size=cfg["hidden_size"],
        num_heads=first.num_heads, head_dim=first.head_dim,
        intermediate_size=dense_i, vocab_size=cfg.get("vocab_size", cfg.get("padded_vocab_size")),
        attention=first.attention, ffn=first.ffn,
        num_kv_heads=first.num_kv_heads, quantization="bf16",
        layer_plans=tuple(plans), body_layers=len(plans),
    )


def _terms(plan, model):
    return {term.name: term for term in layer_weight_terms(plan, model)}


def test_official_fixture_bytes_and_source_manifest_are_pinned():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert {item["id"] for item in manifest} == set(DEEPSEEK + GLM)
    for item in manifest:
        assert len(item["revision"]) == 40
        assert item["revision"] in item["resolved_url"]
        digest = hashlib.sha256((FIXTURES / f"{item['id']}.json").read_bytes()).hexdigest()
        assert digest == item["sha256"]
        fixture = config(item["id"])
        assert all(fixture[key] == value for key, value in item["config"].items())
    sources = json.loads((FIXTURES / "source-manifest.json").read_text())
    assert {item["id"] for item in sources} == {
        "deepseek-v2", "deepseek-v2-lite", "deepseek-v3", "glm4", "glm4_moe", "chatglm"}
    assert all(len(item["sha256"]) == 64 and item["size"] > 0 for item in sources)


@pytest.mark.parametrize("name", DEEPSEEK)
def test_deepseek_standard_schedule_cache_and_weight_arithmetic(name):
    cfg = config(name)
    plans = _standard_deepseek_plans(cfg)
    first = int(cfg["first_k_dense_replace"])
    freq = int(cfg["moe_layer_freq"])
    expected_moe = [i for i in range(len(plans)) if i >= first and i % freq == 0]
    assert len(plans) == cfg["num_hidden_layers"]
    assert [p.index for p in plans if p.ffn == "moe"] == expected_moe
    assert all(p.ffn == "dense" for p in plans[:first])
    qk = cfg["qk_nope_head_dim"] + cfg["qk_rope_head_dim"]
    for plan in plans:
        assert plan.attention == "mla" and plan.mla_variant == "latent_mla"
        assert plan.num_kv_heads == 1
        assert (plan.head_dim, plan.qk_nope_head_dim, plan.qk_rope_head_dim) == (
            qk, cfg["qk_nope_head_dim"], cfg["qk_rope_head_dim"])
        assert plan.v_head_dim == cfg["v_head_dim"]
        assert plan.kv_cache_elements == cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]
        assert plan.q_lora_rank == (cfg.get("q_lora_rank") or 0)
        assert plan.router_dtype == "fp32"
        assert plan.router_mode == cfg["scoring_func"]
        if plan.ffn == "dense":
            assert plan.intermediate_size == cfg["intermediate_size"]
            assert plan.num_experts == 0
        else:
            assert plan.intermediate_size == 0
            assert plan.num_experts == cfg["n_routed_experts"]
            assert plan.top_k == cfg["num_experts_per_tok"]
            assert plan.moe_intermediate_size == cfg["moe_intermediate_size"]
            assert plan.shared_intermediate_size == (
                cfg["n_shared_experts"] * cfg["moe_intermediate_size"])

    model = _model(name, plans, cfg["intermediate_size"])
    dense_plan, moe_plan = plans[0], plans[expected_moe[0]]
    dense_terms = _terms(dense_plan, model)
    assert dense_terms["dense_gate"].rows == cfg["intermediate_size"]
    assert dense_terms["dense_down"].rows == cfg["hidden_size"]
    moe_terms = _terms(moe_plan, model)
    assert (moe_terms["router"].rows, moe_terms["router"].cols) == (
        cfg["n_routed_experts"], cfg["hidden_size"])
    assert moe_terms["expert_gate"].count == cfg["n_routed_experts"]
    assert moe_terms["expert_gate"].rows == cfg["moe_intermediate_size"]
    assert moe_terms["shared_gate"].rows == (
        cfg["n_shared_experts"] * cfg["moe_intermediate_size"])
    if cfg.get("q_lora_rank") is None:
        q = moe_terms["q_proj"]
        assert (q.rows, q.cols) == (
            cfg["num_attention_heads"] * qk, cfg["hidden_size"])
        assert "q_a" not in moe_terms and "q_b" not in moe_terms
        assert "q_a_norm" not in moe_terms
    else:
        rank = cfg["q_lora_rank"]
        assert (moe_terms["q_a"].rows, moe_terms["q_a"].cols) == (rank, cfg["hidden_size"])
        assert (moe_terms["q_b"].rows, moe_terms["q_b"].cols) == (
            cfg["num_attention_heads"] * qk, rank)
        assert moe_terms["q_a_norm"].rows == rank


def test_deepseek_gaps_keep_noaux_tc_and_mtp_explicit():
    v3 = _standard_deepseek_gaps(config("deepseek-v3"))
    assert "router_noaux_tc_correction_bias_not_in_layer_plan" in v3
    assert "router_routed_scaling_factor_not_in_layer_plan" in v3
    assert "optional_mtp_not_modeled" in v3
    assert _aux_standard_deepseek(config("deepseek-v3")) is None
    lite = _standard_deepseek_gaps(config("deepseek-v2-lite"))
    assert lite == ()
    assert _aux_standard_deepseek(config("deepseek-v2-lite")) is not None


def test_deepseek_moe_schedule_uses_native_modulo_not_first_dense_offset():
    cfg = config("deepseek-v3")
    cfg["moe_layer_freq"] = 2
    plans = _standard_deepseek_plans(cfg)
    assert [p.index for p in plans if p.ffn == "moe"] == list(range(4, 61, 2))
    assert [p.index for p in plans if p.ffn == "dense"] == [0, 1, 2, 3, 5, *range(7, 61, 2)]


@pytest.mark.parametrize("name", GLM)
def test_glm_standard_schedule_cache_and_weight_arithmetic(name):
    cfg = config(name)
    plans = _standard_glm_plans(cfg)
    assert len(plans) == cfg.get("num_hidden_layers", cfg.get("num_layers"))
    legacy = cfg["model_type"] == "chatglm"
    moe_model = cfg["model_type"] == "glm4_moe"
    first = int(cfg.get("first_k_dense_replace", 0)) if moe_model else len(plans)
    expected_moe = [
        i for i in range(len(plans))
        if moe_model and i >= first and cfg.get("n_routed_experts") is not None]
    assert [p.index for p in plans if p.ffn == "moe"] == expected_moe
    for plan in plans:
        assert plan.attention == "gqa"
        assert plan.head_dim == cfg.get("head_dim", cfg.get("kv_channels"))
        assert plan.num_heads == cfg["num_attention_heads"]
        assert plan.num_kv_heads == cfg.get(
            "num_key_value_heads", cfg.get("multi_query_group_num"))
        assert plan.kv_cache_elements == 2 * plan.num_kv_heads * plan.head_dim
        if not legacy:
            assert plan.qk_norm is bool(cfg.get("use_qk_norm", False))
        assert plan.qk_rope_head_dim == plan.head_dim // 2
        assert plan.qk_nope_head_dim + plan.qk_rope_head_dim == plan.head_dim
        if plan.ffn == "dense":
            assert plan.intermediate_size == cfg.get(
                "intermediate_size", cfg.get("ffn_hidden_size"))
        else:
            assert plan.num_experts == cfg["n_routed_experts"]
            assert plan.moe_intermediate_size == cfg["moe_intermediate_size"]
            assert plan.shared_intermediate_size == (
                cfg["n_shared_experts"] * cfg["moe_intermediate_size"])
            assert plan.router_mode == "sigmoid" and plan.router_dtype == "fp32"

    model = _model(name, plans, cfg.get("intermediate_size", cfg.get("ffn_hidden_size")))
    terms = _terms(plans[0], model)
    assert (terms["q_proj"].rows, terms["q_proj"].cols) == (
        cfg["num_attention_heads"] * model.head_dim, cfg["hidden_size"])
    assert terms["k_proj"].rows == model.num_kv_heads * model.head_dim
    assert (terms["q_norm"].rows if "q_norm" in terms else 0) == (
        model.head_dim if model.layer_plans[0].qk_norm else 0)
    if expected_moe:
        moe_terms = _terms(plans[expected_moe[0]], model)
        assert moe_terms["router"].rows == cfg["n_routed_experts"]
        assert moe_terms["expert_gate"].count == cfg["n_routed_experts"]
    else:
        assert terms["dense_gate"].rows == model.intermediate_size


def test_glm_gaps_and_optional_mtp_are_not_silently_absorbed():
    modern = _standard_glm_gaps(config("glm4-0414"))
    assert "additional_post_self_attn_and_post_mlp_norms_not_in_layer_schema" in modern
    assert "partial_rope_64_of_128_not_in_attention_plan" in modern
    assert "attention_bias_qkv_not_in_layer_plan" in modern
    assert _aux_standard_glm(config("glm4-0414")) is not None
    moe = _standard_glm_gaps(config("glm45"))
    assert "router_sigmoid_correction_bias_and_routed_scaling_not_in_layer_plan" in moe
    assert "optional_mtp_not_modeled" in moe
    assert _aux_standard_glm(config("glm45")) is None
    assert _aux_standard_glm(config("glm45-air")) is None
    legacy = _standard_glm_gaps(config("glm4-chat"))
    assert "attention_bias_qkv_not_in_layer_plan" in legacy
    assert "partial_rope_64_of_128_not_in_attention_plan" in legacy
    assert "legacy_layernorm_not_rmsnorm_not_modeled" not in legacy


def test_legacy_chatglm_degradable_flags_become_gaps_without_raise():
    cfg = config("glm4-chat")
    cfg.update(rmsnorm=False, add_bias_linear=True,
               apply_residual_connection_post_layernorm=True)
    plans = _standard_glm_plans(cfg)
    assert len(plans) == cfg["num_layers"]
    gaps = _standard_glm_gaps(cfg)
    assert set(gaps) >= {
        "legacy_layernorm_not_rmsnorm_not_modeled",
        "linear_bias_not_in_layer_plan",
        "residual_connection_post_layernorm_not_modeled",
    }


def _direct_case(q_lora_rank: int):
    plan = LayerPlan(
        index=0, attention="mla", architecture="mla_standard", ffn="dense",
        num_heads=4, num_kv_heads=1, head_dim=24, q_lora_rank=q_lora_rank,
        kv_lora_rank=32, qk_nope_head_dim=16, qk_rope_head_dim=8, v_head_dim=16,
        mla_variant="latent_mla", kv_cache_elements=40,
        kv_owner_layer_id=0, owns_long_kv=True, intermediate_size=64,
        native_layer_id=0, scope="body",
    )
    raw = case(prompt=3, output=1)
    raw["model"] = {
        "name": "direct-mla", "layers": 1, "hidden_size": 32, "num_heads": 4,
        "num_kv_heads": 1, "head_dim": 24, "intermediate_size": 64,
        "vocab_size": 128, "attention": "mla", "ffn": "dense",
        "layer_plans": [plan.to_dict()],
    }
    return plan, raw


def test_direct_q_creates_one_real_q_proj_and_releases_live_tasks():
    plan, raw = _direct_case(0)
    ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=3)}
    q = ops["s0.l0.q_proj"]
    assert q.semantic_shape == (3, 4 * 24, 32)
    assert q.flops == 2 * 3 * 4 * 24 * 32
    assert not any(op_id.endswith((".q_a", ".q_a_norm", ".q_norm", ".q_b")) for op_id in ops)
    assert all(min(op.semantic_shape) > 0 for op in ops.values() if op.family == "gemm")
    report = graph_costs(
        raw, tuple(ops.values()), input_ready_ms=7,
        external_inputs={"analysis_input": 0}, trace_limit=100)
    assert report["execution"]["live_tasks"] == 0
    assert report["execution"]["live_programs"] == 0
    assert not any(report["execution"]["live_allocated"].values())
    row = next(row for row in report["timeline"] if row["op_id"] == q.op_id)
    assert row["ready_ms"] >= 7

    model = ModelSpec(
        name="direct-mla", layers=1, hidden_size=32, num_heads=4, head_dim=24,
        intermediate_size=64, vocab_size=128, attention="mla",
        num_kv_heads=1, layer_plans=(plan,), body_layers=1,
    )
    terms = _terms(plan, model)
    assert (terms["q_proj"].rows, terms["q_proj"].cols) == (96, 32)
    assert terms["q_proj"].rows * terms["q_proj"].cols > 0
    assert "q_a" not in terms and "q_b" not in terms and "q_a_norm" not in terms


def test_lowrank_mla_path_is_unchanged():
    plan, raw = _direct_case(8)
    ops = {op.op_id: op for op in target_operators(raw, phase="prefill", tokens=3)}
    assert "s0.l0.q_a" in ops and "s0.l0.q_norm" in ops and "s0.l0.q_b" in ops
    assert "s0.l0.q_proj" not in ops
    assert ops["s0.l0.q_a"].flops == 2 * 3 * 8 * 32
    assert ops["s0.l0.q_b"].flops == 2 * 3 * 96 * 8
    model = ModelSpec(
        name="lowrank-mla", layers=1, hidden_size=32, num_heads=4, head_dim=24,
        intermediate_size=64, vocab_size=128, attention="mla",
        num_kv_heads=1, layer_plans=(plan,), body_layers=1,
    )
    terms = _terms(plan, model)
    assert terms["q_a_norm"].rows == 8
    assert (terms["q_a"].rows, terms["q_a"].cols) == (8, 32)
    assert (terms["q_b"].rows, terms["q_b"].cols) == (96, 8)


def test_direct_q_with_explicit_source_prolog_is_refused_not_faked():
    _, raw = _direct_case(0)
    raw["runtime"]["kernels"] = {
        "lowerings": {"mla_prolog": "vllm_ascend_mla_prolog_v3"}}
    with pytest.raises(ModelingError, match="mla_prolog_v3_direct_q_unsupported"):
        target_operators(raw, phase="prefill", tokens=3)
