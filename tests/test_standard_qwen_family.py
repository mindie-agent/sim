"""Independent structural checks for standard Qwen2/2.5/3 dense and MoE bodies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from mindie_sim.catalog.qwen import (
    _aux_standard_qwen, _standard_qwen_gaps, _standard_qwen_plans,
)
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.catalog.weight_terms import term_bytes
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests/fixtures/standard-qwen-kimi"
QWEN = ("qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b",
        "qwen3-8b", "qwen3-30b-a3b")
ALL = QWEN + ("kimi-k2-instruct", "kimi-vl-a3b-instruct")


def config(name: str) -> dict:
    return json.loads((FIXTURES / f"{name}.json").read_text())


def model(name: str, cfg: dict, plans) -> ModelSpec:
    return ModelSpec(
        name=name, layers=len(plans), hidden_size=cfg["hidden_size"],
        num_heads=cfg["num_attention_heads"],
        head_dim=plans[0].head_dim if plans[0].head_dim else cfg["hidden_size"] // cfg["num_attention_heads"],
        intermediate_size=cfg["intermediate_size"], vocab_size=cfg["vocab_size"],
        attention="gqa", ffn=plans[0].ffn,
        num_kv_heads=cfg["num_key_value_heads"],
        layer_plans=tuple(plans), body_layers=len(plans),
    )


def terms(plan, spec):
    return {term.name: term for term in layer_weight_terms(plan, spec)}


def expected_head_dim(cfg: dict) -> int:
    hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
    if cfg["model_type"] == "qwen2_moe" or cfg.get("head_dim") is None:
        return hidden // heads
    return int(cfg["head_dim"])


def expected_moe_layers(cfg: dict) -> list[int]:
    layers = cfg["num_hidden_layers"]
    experts = cfg.get("num_experts", 0)
    if experts <= 0:
        return []
    dense_only = {int(v) for v in cfg.get("mlp_only_layers", ()) or ()}
    step = int(cfg.get("decoder_sparse_step", 1))
    return [i for i in range(layers)
            if i not in dense_only and (i + 1) % step == 0]


def test_official_fixture_bytes_and_portable_manifest():
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    assert {item["id"] for item in manifest} == set(ALL)
    for item in manifest:
        path = FIXTURES / item["file"]
        assert "/tmp" not in item["resolved_url"]
        assert item["revision"] in item["resolved_url"]
        assert len(item["revision"]) == 40
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
        assert path.stat().st_size == item["bytes"]
        fixture = json.loads(path.read_text())
        text = fixture.get("text_config", {})
        flat = {**fixture, **text}
        for key, value in item["config"].items():
            if key.startswith("text_"):
                assert flat[key[5:]] == value
            elif key == "vision_model_type":
                assert fixture["vision_config"]["model_type"] == value
            else:
                assert flat[key] == value
    sources = json.loads((FIXTURES / "source-manifest.json").read_text())
    assert sources["transformers"]["revision"] == "v4.51.3"
    assert all(len(source["sha256"]) == 64
               for source in sources["transformers"]["fetched_sources"])
    assert sources["models"][0]["repository"] == "moonshotai/Kimi-K2-Instruct"
    assert sources["models"][1]["repository"] == "moonshotai/Kimi-VL-A3B-Instruct"


@pytest.mark.parametrize("name", QWEN)
def test_qwen_exact_geometry_schedule_and_shared_experts(name):
    cfg = config(name)
    plans = _standard_qwen_plans(cfg)
    assert [plan.index for plan in plans] == list(range(cfg["num_hidden_layers"]))
    family = cfg["model_type"]
    head_dim = expected_head_dim(cfg)
    moe_layers = expected_moe_layers(cfg)
    qk_norm = family in {"qwen3", "qwen3_moe"}
    for plan in plans:
        assert plan.attention == "gqa"
        assert plan.num_heads == cfg["num_attention_heads"]
        assert plan.num_kv_heads == cfg["num_key_value_heads"]
        assert plan.head_dim == head_dim
        assert plan.qk_norm is qk_norm
        assert plan.window == 0
        assert plan.kv_cache_elements == 2 * cfg["num_key_value_heads"] * head_dim
        assert plan.kv_owner_layer_id == plan.index
        assert plan.expert_latent_size == 0
        if plan.index in moe_layers:
            assert plan.ffn == "moe"
            assert plan.intermediate_size == 0
            assert plan.num_experts == cfg["num_experts"]
            assert plan.top_k == cfg["num_experts_per_tok"]
            assert plan.moe_intermediate_size == cfg["moe_intermediate_size"]
            assert plan.router_mode == "softmax"
            assert plan.router_norm_topk is bool(cfg.get("norm_topk_prob", False))
            if family == "qwen2_moe":
                assert plan.num_shared_experts == 1
                assert plan.shared_intermediate_size == cfg["shared_expert_intermediate_size"]
                assert plan.shared_expert_gate is True
            else:
                assert plan.num_shared_experts == 0
                assert plan.shared_intermediate_size == 0
                assert plan.shared_expert_gate is False
        else:
            assert plan.ffn == "dense"
            assert plan.intermediate_size == cfg["intermediate_size"]
            assert plan.num_experts == plan.top_k == 0
            assert plan.num_shared_experts == 0
    assert [plan.index for plan in plans if plan.ffn == "moe"] == moe_layers


@pytest.mark.parametrize("name", QWEN)
def test_qwen_weight_terms_follow_declared_head_and_ffn_geometry(name):
    cfg = config(name)
    plans = _standard_qwen_plans(cfg)
    spec = model(name, cfg, plans)
    h = cfg["hidden_size"]
    heads = cfg["num_attention_heads"]
    kv_heads = cfg["num_key_value_heads"]
    head_dim = expected_head_dim(cfg)
    qk_norm = cfg["model_type"] in {"qwen3", "qwen3_moe"}
    for plan in plans:
        layer = terms(plan, spec)
        assert (layer["q_proj"].rows, layer["q_proj"].cols) == (heads * head_dim, h)
        assert (layer["k_proj"].rows, layer["k_proj"].cols) == (kv_heads * head_dim, h)
        assert (layer["v_proj"].rows, layer["v_proj"].cols) == (kv_heads * head_dim, h)
        assert (layer["o_proj"].rows, layer["o_proj"].cols) == (h, heads * head_dim)
        assert (("q_norm" in layer), ("k_norm" in layer)) == (qk_norm, qk_norm)
        if qk_norm:
            assert layer["q_norm"].rows == layer["k_norm"].rows == head_dim
        if plan.ffn == "dense":
            assert (layer["dense_gate"].rows, layer["dense_gate"].cols) == (
                cfg["intermediate_size"], h)
            assert (layer["dense_down"].rows, layer["dense_down"].cols) == (
                h, cfg["intermediate_size"])
            assert "router" not in layer and "expert_gate" not in layer
        else:
            assert (layer["router"].rows, layer["router"].cols) == (
                cfg["num_experts"], h)
            assert layer["expert_gate"].count == cfg["num_experts"]
            assert (layer["expert_gate"].rows, layer["expert_gate"].cols) == (
                cfg["moe_intermediate_size"], h)
            assert (layer["expert_down"].rows, layer["expert_down"].cols) == (
                h, cfg["moe_intermediate_size"])
            shared = plan.shared_intermediate_size
            assert ("shared_gate" in layer) == (shared > 0)
            if shared:
                assert (layer["shared_gate"].rows, layer["shared_gate"].cols) == (
                    shared, h)
                assert ("shared_router" in layer) is plan.shared_expert_gate


def test_qwen_explicit_head_dim_is_authoritative_for_qwen3_30b():
    cfg = config("qwen3-30b-a3b")
    plans = _standard_qwen_plans(cfg)
    assert cfg["hidden_size"] == 2048 and cfg["num_attention_heads"] == 32
    assert cfg["head_dim"] == 128
    assert plans[0].head_dim == 128
    assert plans[0].head_dim != cfg["hidden_size"] // cfg["num_attention_heads"]
    spec = model("qwen3-30b-a3b", cfg, plans)
    layer = terms(plans[0], spec)
    assert layer["q_proj"].rows == 32 * 128 == 4096
    assert layer["o_proj"].cols == 4096
    derived = config("qwen2-7b")
    assert _standard_qwen_plans(derived)[0].head_dim == 3584 // 28 == 128


def test_qwen_moe_schedule_uses_one_based_step_and_dense_only_overrides():
    cfg = dict(config("qwen3-30b-a3b"))
    cfg["decoder_sparse_step"] = 3
    cfg["mlp_only_layers"] = [2, 5, 8, 11]
    plans = _standard_qwen_plans(cfg)
    expected = {i for i in range(48) if (i + 1) % 3 == 0} - {2, 5, 8, 11}
    assert {plan.index for plan in plans if plan.ffn == "moe"} == expected
    assert all(plans[i].ffn == "dense" for i in (2, 5, 8, 11))
    assert all(plans[i].ffn == "dense" for i in range(48) if i not in expected)

    qwen2 = dict(config("qwen2-57b-a14b"))
    qwen2["decoder_sparse_step"] = 2
    qwen2["mlp_only_layers"] = [3, 7]
    plans = _standard_qwen_plans(qwen2)
    expected = {i for i in range(28) if (i + 1) % 2 == 0} - {3, 7}
    assert {plan.index for plan in plans if plan.ffn == "moe"} == expected
    assert plans[0].ffn == "dense" and plans[1].ffn == "moe"


def test_qwen_window_uses_actual_top_layer_flash_attention_condition():
    cfg = dict(config("qwen2-7b"))
    cfg.update(use_sliding_window=True, sliding_window=4096, max_window_layers=20)
    plans = _standard_qwen_plans(cfg)
    assert [plan.window for plan in plans] == [0] * 20 + [4096] * 8
    assert not any("sliding" in gap for gap in _standard_qwen_gaps(cfg))

    # The source compares layer_idx >= max_window_layers. If the threshold is
    # the model depth, the flag is true but no layer is windowed.
    cfg["max_window_layers"] = 28
    assert all(plan.window == 0 for plan in _standard_qwen_plans(cfg))


def test_qwen_explicit_full_sliding_schedule_and_unsupported_kind_gap():
    cfg = dict(config("qwen3-8b"))
    cfg["sliding_window"] = 8
    cfg["max_window_layers"] = 0
    cfg["layer_types"] = (
        ["full_attention"] * 2 + ["sliding_attention"] * 2
        + ["full_attention"] * 32
    )
    plans = _standard_qwen_plans(cfg)
    assert [plan.window for plan in plans] == [0, 0, 8, 8] + [0] * 32
    assert all(plan.attention == "gqa" for plan in plans)

    cfg["layer_types"] = list(cfg["layer_types"])
    cfg["layer_types"][5] = "linear_attention"
    plans = _standard_qwen_plans(cfg)
    gaps = _standard_qwen_gaps(cfg)
    assert len(plans) == 36
    assert plans[5].attention == "projection_only"
    assert plans[5].architecture == "linear_attention"
    assert plans[5].head_dim == 128 and plans[5].kv_cache_elements == 0
    assert plans[5].cache_kind == "none"
    assert all(plan.attention == "gqa" for i, plan in enumerate(plans) if i != 5)
    assert "unsupported_attention_layer:5:linear_attention" in gaps


def test_qwen_missing_window_information_is_a_visible_gap_not_full_attention():
    cfg = dict(config("qwen2-7b"))
    cfg["use_sliding_window"] = True
    cfg["sliding_window"] = 4096
    cfg.pop("max_window_layers", None)
    plans = _standard_qwen_plans(cfg)
    assert all(plan.window == 0 for plan in plans)
    assert all(plan.attention == "projection_only" for plan in plans)
    assert "sliding_window_max_window_layers_missing" in _standard_qwen_gaps(cfg)

    cfg = dict(config("qwen3-8b"))
    cfg["layer_types"] = ["sliding_attention"] + ["full_attention"] * 35
    cfg["sliding_window"] = None
    plans = _standard_qwen_plans(cfg)
    assert plans[0].window == 0
    assert plans[0].attention == "projection_only"
    assert "sliding_layer_type_without_declared_window:0" in _standard_qwen_gaps(cfg)


def test_qwen_bias_and_activation_gaps_keep_body_partially_usable():
    for name in ("qwen2-7b", "qwen2.5-7b-instruct", "qwen2-57b-a14b"):
        assert "qkv_bias_unmodeled" in _standard_qwen_gaps(config(name))
    assert not any("bias" in gap for gap in _standard_qwen_gaps(config("qwen3-8b")))
    assert not any("bias" in gap for gap in _standard_qwen_gaps(config("qwen3-30b-a3b")))

    biased = dict(config("qwen3-8b"))
    biased["attention_bias"] = True
    gaps = _standard_qwen_gaps(biased)
    assert "attention_bias_unmodeled" in gaps
    # Qwen3 declares the same flag for the output projection.
    assert len(_standard_qwen_plans(biased)) == 36

    no_bias = dict(config("qwen2-57b-a14b"))
    no_bias["qkv_bias"] = False
    assert "qkv_bias_unmodeled" not in _standard_qwen_gaps(no_bias)

    activated = dict(config("qwen3-8b"))
    activated["hidden_act"] = "gelu"
    plans = _standard_qwen_plans(activated)
    assert plans[0].ffn_activation == "gelu"
    assert "hidden_act_unsupported:gelu" in _standard_qwen_gaps(activated)


def test_qwen_standard_aux_never_infers_a_draft_from_only_a_layer_count():
    for name in QWEN:
        aux = _aux_standard_qwen(config(name))
        assert aux.mtp is None and aux.dspark is None
        assert aux.external_draft_key == ""
        assert aux.standalone_draft is False

    cfg = dict(config("qwen3-30b-a3b"))
    cfg["mtp_num_hidden_layers"] = 1
    aux = _aux_standard_qwen(cfg)
    assert aux.dspark is None
    assert aux.mtp is None
    assert "optional_mtp_schema_not_modeled" in _standard_qwen_gaps(cfg)

    undeclared_geometry = dict(config("qwen3-8b"))
    undeclared_geometry["num_nextn_predict_layers"] = 1
    assert _aux_standard_qwen(undeclared_geometry).mtp is None
    assert "optional_mtp_schema_not_modeled" in _standard_qwen_gaps(undeclared_geometry)


@pytest.mark.parametrize("name", ("qwen2-57b-a14b", "qwen3-30b-a3b"))
def test_qwen_router_weights_do_not_inherit_the_softmax_fp32_cast(name):
    cfg = config(name)
    plans = _standard_qwen_plans(cfg)
    spec = model(name, cfg, plans)
    router = terms(plans[0], spec)["router"]
    # HF's gate is a normal model-precision Linear. The FP32 softmax happens
    # after its output; promoting gate weights would double residency.
    assert term_bytes(router, QuantSpec("bf16"), ParallelSpec(), 1) == (
        cfg["hidden_size"] * cfg["num_experts"] * 2)


def test_unknown_attention_keeps_ffn_without_inventing_missing_projection_dims():
    cfg = config("qwen3-8b")
    cfg["layer_types"] = ["custom_attention"] * cfg["num_hidden_layers"]
    cfg.pop("num_attention_heads")
    plans = _standard_qwen_plans(cfg)
    assert all(plan.attention == "unmodeled" for plan in plans)
    assert all(plan.intermediate_size == cfg["intermediate_size"] for plan in plans)
    assert len(_standard_qwen_gaps(cfg)) == cfg["num_hidden_layers"]


def test_short_explicit_schedule_preserves_known_entries_instead_of_replacing_all():
    cfg = config("qwen3-8b")
    cfg["layer_types"] = ["custom_attention", "full_attention"]
    plans = _standard_qwen_plans(cfg)
    assert plans[0].architecture == "custom_attention"
    assert plans[0].attention == "projection_only"
    assert plans[1].attention == "gqa"
    assert all(plan.attention == "projection_only" for plan in plans[2:])
    assert all(plan.cache_kind == "none" for plan in plans[2:])
