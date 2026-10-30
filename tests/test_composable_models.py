"""External configs exercise real frontend work and explicit omitted boundaries."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import graph_costs, operator_costs, predict, target_graphs, target_operators
from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.model_catalog import ARCHITECTURE_PRESET_CATALOG, load_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.model import ModelSpec
from mindie_sim.schema.parallel import ParallelSpec
from test_prediction_system import case

FIXTURES = Path(__file__).parent / "fixtures" / "composable-hf"


def config(**changes):
    return dict(model_type="llama", num_hidden_layers=2, hidden_size=32,
                num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                intermediate_size=64, vocab_size=128, **changes)


def run_case(model, **kwargs):
    raw = case(**kwargs)
    raw["model"] = model
    return raw


def test_unknown_fields_do_not_disable_known_graph_and_are_not_silent():
    original = config()
    custom = {**original, "model_type": "step2_structural_test",
              "custom_cross_layer": {"mode": "not_understood"}}
    raw = run_case(custom)
    parsed = parse_case(raw)
    baseline = graph_costs(run_case(original), target_graphs(run_case(original), phase="prefill", tokens=3))
    result = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=3))
    assert result["duration_ms"] == baseline["duration_ms"]
    assert result["model_coverage"] == "partial"
    assert "unrecognized_config_field:custom_cross_layer" in result["unmodeled_features"]
    assert dict(parsed.model.metadata)["unrecognized_config_fields"]["custom_cross_layer"] == custom["custom_cross_layer"]
    serving = predict(raw)
    isolated = operator_costs(raw, target_operators(raw, phase="prefill", tokens=3))
    assert serving["status"] == "modeled"
    assert isolated["model_coverage"] == serving["model_coverage"] == "partial"
    assert ModelSpec.from_mapping(parsed.model.to_dict()) == parsed.model
    assert "model_coverage" not in custom


def test_partial_attention_keeps_projections_ffn_and_marks_core_hole():
    raw = run_case({**config(), "layer_types": ["custom_kernel", "full_attention"]})
    model = parse_case(raw).model
    assert [p.attention for p in model.layer_plans] == ["projection_only", "gqa"]
    ops = target_operators(raw, phase="prefill", tokens=3)
    first = [op for op in ops if op.layer == 0]
    graphs = target_graphs(raw, phase="prefill", tokens=3)
    assert any(op.family == "barrier" and dict(op.metadata).get("unmodeled_component") == "custom_kernel"
               for graph in graphs.values() for op in graph)
    assert not any(op.family == "attention" for op in first)
    projection = next(op for op in first if op.op_id.endswith("known_qkv"))
    assert projection.flops == 2 * 3 * (4 + 2 * 2) * 8 * 32
    assert any(op.family == "swiglu" for op in first)
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=3))
    assert graph["duration_ms"] > 0 and graph["execution"]["live_tasks"] == 0


@pytest.mark.parametrize("pp", [1, 2])
@pytest.mark.parametrize("qk_norm", [False, True])
def test_shared_gqa_has_one_kv_projection_cache_and_real_producer_dependency(pp, qk_norm):
    raw = run_case({**config(), "kv_source_layer_ids": [0, 0], "qk_norm": qk_norm}, pp=pp)
    model = parse_case(raw).model
    owner, borrower = model.layer_plans
    owner_bytes = sum(layer_cache_components(
        model, owner, ParallelSpec(), 16, capacity_padded=False).values())
    assert owner_bytes == 16 * 2 * 2 * 8 * 2
    assert layer_cache_components(model, borrower, ParallelSpec(), 16) == {}
    assert not {"k_proj", "v_proj"} & {term.name for term in layer_weight_terms(borrower, model)}
    assert "k_norm" not in {term.name for term in layer_weight_terms(borrower, model)}
    graphs = target_graphs(raw, phase="decode", tokens=1, context=16)
    ops = [op for graph in graphs.values() for op in graph]
    projections = [op for op in ops if op.family == "qkv_projection"]
    assert [op.local_shape[1] for op in projections] == [64, 32]
    stores = [op for op in ops if op.recipe == "kv_store"]
    assert len(stores) == 1
    shared = next(op for op in ops if op.recipe == "shared_cache_dependency")
    transfers = [op for op in ops if op.recipe == "shared_kv_transfer"]
    assert len(transfers) == pp - 1
    if pp == 1:
        assert stores[0].op_id in shared.deps
    else:
        assert transfers[0].deps == (stores[0].op_id,)
        assert transfers[0].op_id in shared.deps
        assert transfers[0].link_bytes == 17 * 2 * 2 * 8 * 2
    report = graph_costs(raw, target_graphs(raw, phase="decode", tokens=1, context=16),
                         input_ready_ms=7)
    assert all(row["ready_ms"] >= 7 for row in report["timeline"])
    assert report["execution"]["live_tasks"] == report["execution"]["live_programs"] == 0
    if pp > 1:
        assert "cross_pp_shared_kv_receive_residency_and_copy_endpoints" in report["unmodeled_features"]


def test_precision_is_independent_from_kv_and_unknown_quant_has_explicit_proxy():
    native = ModelSpec.from_mapping({**config(), "torch_dtype": "float32", "kv_dtype": "bf16"})
    assert native.activation_dtype == native.compute_dtype == "fp32"
    assert native.quant.matrix_bytes(32, 128, "embedding") == 4 * 32 * 128
    assert native.quant.kv_dtype == "bf16"
    raw = run_case(native.to_dict())
    assert predict(raw)["status"] == "modeled"
    for extra in (
        {"weight_dtype": "fp32"},
        {"quantization_config": {"quant_method": "fp8", "fmt": "e5m2",
                                 "activation_scheme": "static", "weight_block_size": [128, 128]}},
        {"weight_dtype": "fp8"},
    ):
        model = ModelSpec.from_mapping({**config(), **extra})
        assert dict(model.metadata)["model_coverage"] == "partial"
        assert any("proxy" in value for value in dict(model.metadata)["degraded_submodules"])


def test_declared_fp8_blocks_apply_to_experts_and_survive_roundtrip():
    raw = {**config(), "n_routed_experts": 4, "num_experts_per_tok": 2,
           "moe_intermediate_size": 64, "quantization_config": {
               "quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
               "weight_block_size": [128, 128]}}
    model = ModelSpec.from_mapping(raw)
    assert model.quant.matrix_bytes(130, 129, "routed_expert") == 130 * 129 + 2 * 2 * 4
    assert model.quant.format_for("router") == "bf16"
    assert ModelSpec.from_mapping(model.to_dict()).quant == model.quant


def test_global_sliding_window_is_work_not_just_metadata():
    full = run_case(config())
    windowed = run_case({**config(), "model_type": "mistral", "sliding_window": 4})
    get = lambda raw: sum(op.flops for op in target_operators(
        raw, phase="decode", tokens=1, context=32) if op.family == "attention")
    assert get(windowed) / get(full) == pytest.approx(4 / 33)
    biased = parse_case(run_case({**config(), "attention_bias": True})).model
    assert "custom_submodule:attention_bias" in dict(biased.metadata)["unmodeled_features"]


@pytest.mark.parametrize("declared", [{"1": 0}, [0, 0]])
def test_shared_owner_alias_applies_to_standard_and_generic_families(declared):
    for family in ("llama", "qwen3"):
        raw = run_case({**config(), "model_type": family, "kv_consumer_to_owner": declared})
        model = parse_case(raw).model
        assert [p.kv_owner_layer_id for p in model.layer_plans] == [0, 0]
        assert len([op for op in target_operators(raw, phase="prefill", tokens=3)
                    if op.recipe == "kv_store"]) == 1


@pytest.mark.parametrize("family", ["deepseek_future", "glm_future", "qwen_next", "kimi_future"])
def test_unverified_family_variant_is_executable_but_never_claimed_complete(family):
    raw = run_case({**config(), "model_type": family, "hidden_act": "gelu"})
    model = parse_case(raw).model
    assert "ffn_activation:gelu:silu_proxy" in dict(model.metadata)["unmodeled_features"]
    assert predict(raw)["model_coverage"] == "partial"


def test_glm_without_indexer_keeps_explicit_mla_and_direct_query():
    raw = run_case({**config(), "model_type": "glm4_moe", "kv_lora_rank": 16,
                    "q_lora_rank": None, "qk_nope_head_dim": 8,
                    "qk_rope_head_dim": 4, "v_head_dim": 8})
    model = parse_case(raw).model
    assert all(p.attention == "mla" and not p.indexer and p.q_lora_rank == 0
               for p in model.layer_plans)
    assert predict(raw)["status"] == "modeled"


@pytest.mark.parametrize("name", [
    "minimax-m1-80k", "minimax-text-01", "step3", "baichuan2-7b-base",
])
def test_official_external_configs_load_without_weights_and_roundtrip(name):
    raw = json.loads((FIXTURES / (name + ".json")).read_text())
    model = ModelSpec.from_mapping(raw)
    assert model.layers == raw.get("num_hidden_layers", raw.get("text_config", {}).get("num_hidden_layers"))
    assert model.parameter_count() > 0
    assert ModelSpec.from_mapping(model.to_dict()) == model
    assert load_model(str(FIXTURES / (name + ".json"))) == replace(model, name=str(FIXTURES / (name + ".json")))
    if "minimax" in name:
        linear = model.layer_plans[0]
        assert linear.attention == "projection_only"
        assert linear.num_kv_heads == linear.num_heads == 64
        assert linear.recurrent_state_elements == 64 * 128 * 128
    if name == "step3":
        assert model.layer_plans[0].attention == "unmodeled"
        assert model.layer_plans[3].ffn == "dense"
        assert model.layer_plans[4].ffn == "moe"
        assert model.layer_plans[60].ffn == "dense"


@pytest.mark.parametrize("name", ["minimax-m1-80k", "minimax-text-01", "step3"])
def test_external_custom_models_run_known_backbone_with_same_partial_serving_scope(name):
    raw_config = json.loads((FIXTURES / (name + ".json")).read_text())
    # Small dimensional controls preserve each real config's custom branch
    # declaration. They establish execution, not full-checkpoint timing accuracy.
    text = raw_config.get("text_config", raw_config)
    text.update(num_hidden_layers=2, hidden_size=32, intermediate_size=64,
                vocab_size=128, num_attention_heads=4, head_dim=8,
                num_key_value_heads=2, moe_intermediate_size=16)
    if "attn_type_list" in text:
        text.update(attn_type_list=[0, 1], num_local_experts=4, num_experts_per_tok=2)
    else:
        text.update(moe_layers_enum="1", moe_num_experts=4, moe_top_k=2,
                    share_expert_dim=16, share_q_dim=16)
    raw = run_case(raw_config)
    result = predict(raw)
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=3), isolated=False)
    assert result["status"] == "modeled"
    assert result["unmodeled_features"] == graph["unmodeled_features"]
    assert result["model_coverage"] == "partial"
    assert graph["execution"]["live_tasks"] == graph["execution"]["live_programs"] == 0


@pytest.mark.parametrize("key", list(ARCHITECTURE_PRESET_CATALOG))
def test_all_architecture_presets_reach_real_frontend_and_engine(key):
    model = load_model(key)
    # Keep original physical layer geometry; one body layer suffices to prove
    # the preset enters the public engine without thousands of simulated tiles.
    one = replace(model, layers=1, body_layers=1, layer_plans=model.layer_plans[:1], auxiliary=None)
    raw = run_case(one.to_dict(), prompt=1, output=1)
    raw["machine"]["device"]["memory_bytes"] = 10**15
    raw["runtime"]["kernels"] = {"programs": {"matmul": {"kind": "envelope"}}}
    graph = graph_costs(raw, target_graphs(raw, phase="prefill", tokens=1), isolated=False)
    assert graph["duration_ms"] > 0
    assert graph["model_coverage"] == "partial"
    assert predict(raw)["status"] == "modeled"
    assert graph["execution"]["live_tasks"] == 0
