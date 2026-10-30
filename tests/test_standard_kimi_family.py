"""Independent structural checks for standard Kimi-K2/Kimi-VL text bodies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.kimi import (
    _aux_standard_kimi, _standard_kimi_gaps, _standard_kimi_plans,
)
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.schema.model import ModelSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/standard-qwen-kimi"
K2 = "kimi-k2-instruct"
VL = "kimi-vl-a3b-instruct"
KIMI = (K2, VL)


def config(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def flat(cfg: dict) -> dict:
    text = cfg.get("text_config")
    return {**cfg, **text} if isinstance(text, dict) else dict(cfg)


def model(name: str, cfg: dict, plans) -> ModelSpec:
    merged = flat(cfg)
    return ModelSpec(
        name=name, layers=len(plans), hidden_size=merged["hidden_size"],
        num_heads=merged["num_attention_heads"], head_dim=plans[0].head_dim,
        intermediate_size=merged["intermediate_size"],
        vocab_size=merged["vocab_size"], attention="mla", ffn=plans[0].ffn,
        num_kv_heads=1, layer_plans=tuple(plans), body_layers=len(plans),
    )


def terms(plan, spec):
    return {term.name: term for term in layer_weight_terms(plan, spec)}


def expected_moe_layers(cfg: dict) -> list[int]:
    merged = flat(cfg)
    first = int(merged["first_k_dense_replace"])
    freq = int(merged["moe_layer_freq"])
    return [i for i in range(int(merged["num_hidden_layers"]))
            if i >= first and i % freq == 0]


def test_official_kimi_fixture_bytes_and_manifest_are_pinned():
    manifest = {item["id"]: item for item in
                json.loads((FIXTURES / "manifest.json").read_text())}
    for name in KIMI:
        item = manifest[name]
        path = FIXTURES / item["file"]
        assert item["revision"] in item["resolved_url"]
        assert "/tmp" not in item["resolved_url"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        fixture = json.loads(path.read_text())
        assert fixture["model_type"] == item["config"]["model_type"]
        assert fixture["architectures"] == item["config"]["architectures"]
    assert manifest[K2]["revision"] == "fd1984e2b7a3350dbf7305fe73a4ede25c14de50"
    assert manifest[VL]["revision"] == "398eede0903cd983a2bfa0cc634e9ac1d843f375"



@pytest.mark.parametrize("name", KIMI)
def test_kimi_text_schedule_mla_geometry_and_no_k3_mechanisms(name):
    cfg = config(name)
    merged = flat(cfg)
    plans = _standard_kimi_plans(cfg)
    assert len(plans) == merged["num_hidden_layers"]
    assert [plan.index for plan in plans] == list(range(len(plans)))
    expected = expected_moe_layers(cfg)
    assert [plan.index for plan in plans if plan.ffn == "moe"] == expected
    assert all(plan.ffn == "dense" for plan in plans if plan.index not in expected)
    qk = merged["qk_nope_head_dim"] + merged["qk_rope_head_dim"]
    for plan in plans:
        assert plan.attention == "mla"
        assert plan.architecture == "mla_standard"
        assert plan.mla_variant == "latent_mla"
        assert plan.num_heads == merged["num_attention_heads"]
        assert plan.num_kv_heads == 1
        assert plan.head_dim == qk
        assert plan.qk_nope_head_dim == merged["qk_nope_head_dim"]
        assert plan.qk_rope_head_dim == merged["qk_rope_head_dim"]
        assert plan.v_head_dim == merged["v_head_dim"]
        assert plan.kv_lora_rank == merged["kv_lora_rank"]
        assert plan.q_lora_rank == (merged.get("q_lora_rank") or 0)
        assert plan.kv_cache_elements == merged["kv_lora_rank"] + merged["qk_rope_head_dim"]
        assert plan.kv_owner_layer_id == plan.index
        assert plan.owns_long_kv is True
        assert plan.router_mode == merged.get("scoring_func", "softmax")
        assert plan.router_norm_topk == bool(merged.get("norm_topk_prob", False))
        assert plan.router_groups == merged.get("n_group", 0)
        assert plan.router_group_topk == merged.get("topk_group", 0)
        # Standard K2/VL text bodies must not inherit the K3 hybrid mechanisms.
        assert plan.attention != "kda" and plan.attention != "gdn"
        assert plan.linear_heads == plan.linear_head_dim == 0
        assert plan.linear_key_heads == plan.linear_key_head_dim == 0
        assert plan.conv_kernel == plan.conv_width == 0
        assert plan.recurrent_state_elements == plan.conv_state_elements == 0
        assert plan.state_bytes == 0
        assert plan.expert_latent_size == 0
        assert plan.hyper_connections is False
        assert plan.hc_mult == plan.hc_sinkhorn_iters == 0
        if plan.ffn == "moe":
            assert plan.intermediate_size == 0
            assert plan.num_experts == merged["n_routed_experts"]
            assert plan.top_k == merged["num_experts_per_tok"]
            assert plan.moe_intermediate_size == merged["moe_intermediate_size"]
            assert plan.num_shared_experts == merged["n_shared_experts"]
            assert plan.shared_intermediate_size == (
                merged["n_shared_experts"] * merged["moe_intermediate_size"])
            assert plan.shared_expert_gate is False
        else:
            assert plan.intermediate_size == merged["intermediate_size"]
            assert plan.num_experts == plan.top_k == 0


@pytest.mark.parametrize("name", KIMI)
def test_kimi_weight_terms_preserve_mla_and_shared_widths(name):
    cfg = config(name)
    merged = flat(cfg)
    plans = _standard_kimi_plans(cfg)
    spec = model(name, cfg, plans)
    h = merged["hidden_size"]
    heads = merged["num_attention_heads"]
    qk = merged["qk_nope_head_dim"] + merged["qk_rope_head_dim"]
    dense = next(plan for plan in plans if plan.ffn == "dense")
    sparse = next(plan for plan in plans if plan.ffn == "moe")

    dense_layer = terms(dense, spec)
    q_rank = merged.get("q_lora_rank") or 0
    if q_rank:
        assert (dense_layer["q_a"].rows, dense_layer["q_a"].cols) == (q_rank, h)
        assert (dense_layer["q_b"].rows, dense_layer["q_b"].cols) == (heads * qk, q_rank)
        assert dense_layer["q_a_norm"].rows == q_rank
        assert "q_proj" not in dense_layer
    else:
        # Kimi-VL declares q_lora_rank=null: one direct Q projection.
        assert (dense_layer["q_proj"].rows, dense_layer["q_proj"].cols) == (
            heads * qk, h)
        assert "q_a" not in dense_layer and "q_b" not in dense_layer
        assert "q_a_norm" not in dense_layer
    assert (dense_layer["kv_a"].rows, dense_layer["kv_a"].cols) == (
        merged["kv_lora_rank"] + merged["qk_rope_head_dim"], h)
    assert (dense_layer["kv_b"].rows, dense_layer["kv_b"].cols) == (
        heads * (merged["qk_nope_head_dim"] + merged["v_head_dim"]),
        merged["kv_lora_rank"])
    assert (dense_layer["o_proj"].rows, dense_layer["o_proj"].cols) == (
        h, heads * merged["v_head_dim"])
    assert "mla_g" not in dense_layer
    assert (dense_layer["dense_gate"].rows, dense_layer["dense_gate"].cols) == (
        merged["intermediate_size"], h)
    assert (dense_layer["dense_down"].rows, dense_layer["dense_down"].cols) == (
        h, merged["intermediate_size"])

    sparse_layer = terms(sparse, spec)
    assert (sparse_layer["router"].rows, sparse_layer["router"].cols) == (
        merged["n_routed_experts"], h)
    assert sparse_layer["expert_gate"].count == merged["n_routed_experts"]
    assert (sparse_layer["expert_gate"].rows, sparse_layer["expert_gate"].cols) == (
        merged["moe_intermediate_size"], h)
    assert (sparse_layer["expert_down"].rows, sparse_layer["expert_down"].cols) == (
        h, merged["moe_intermediate_size"])
    shared = merged["n_shared_experts"] * merged["moe_intermediate_size"]
    assert (sparse_layer["shared_gate"].rows, sparse_layer["shared_gate"].cols) == (
        shared, h)
    assert (sparse_layer["shared_up"].rows, sparse_layer["shared_up"].cols) == (
        shared, h)
    assert (sparse_layer["shared_down"].rows, sparse_layer["shared_down"].cols) == (
        h, shared)
    assert "shared_router" not in sparse_layer
    assert "latent_down" not in sparse_layer and "latent_up" not in sparse_layer
    assert "kda_q" not in sparse_layer and "gdn_qkvz" not in sparse_layer


@pytest.mark.parametrize("name", KIMI)
def test_kimi_standard_aux_never_invents_dspark_or_mtp(name):
    aux = _aux_standard_kimi(config(name))
    assert aux.mtp is None
    assert aux.dspark is None
    assert aux.external_draft_key == ""
    assert aux.standalone_draft is False


@pytest.mark.parametrize("name", KIMI)
def test_kimi_standard_gaps_are_visible_and_vision_is_explicit(name):
    cfg = config(name)
    merged = flat(cfg)
    gaps = _standard_kimi_gaps(cfg)
    # Both fixtures declare noaux_tc and a routed scaling factor.
    assert "router_noaux_tc_correction_bias_not_in_layer_plan" in gaps
    assert "router_noaux_tc_group_score_top2_not_in_layer_plan" in gaps
    assert "router_routed_scaling_factor_not_in_layer_plan" in gaps
    assert not any("fp8" in gap.lower() or "quant" in gap.lower() for gap in gaps)
    if name == VL:
        assert "vision_tower_unmodeled" in gaps
        assert "vision_projector_unmodeled" in gaps
        assert merged["q_lora_rank"] is None
    else:
        assert "vision_tower_unmodeled" not in gaps
        assert "vision_projector_unmodeled" not in gaps
        assert merged["num_nextn_predict_layers"] == 0
    assert "optional_mtp_not_modeled" not in gaps

    custom = dict(cfg)
    custom["linear_attn_config"] = {"full_attn_layers": [1]}
    custom["routed_expert_hidden_size"] = 1024
    custom["dspark_block_size"] = 7
    custom_gaps = _standard_kimi_gaps(custom)
    assert "kda_fields_present_in_standard_kimi_config" in custom_gaps
    assert "latent_expert_fields_present_in_standard_kimi_config" in custom_gaps
    assert "dspark_fields_present_in_standard_kimi_config" in custom_gaps


def test_kimi_vl_normalized_text_type_does_not_change_its_selected_adapter():
    cfg = config(VL)
    before = _standard_kimi_plans(cfg)
    cfg["text_config"]["model_type"] = "deepseek_v3"
    normalized = flat(cfg)
    assert _standard_kimi_plans(normalized) == before
    assert "vision_tower_unmodeled" in _standard_kimi_gaps(normalized)
