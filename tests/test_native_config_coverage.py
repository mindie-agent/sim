"""Native adapter omissions stay visible without changing retained work."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from mindie_sim import target_operators
from mindie_sim.catalog.weights import body_head_terms, layer_weight_terms
from mindie_sim.model_catalog import CATALOG, load_model
from mindie_sim.schema.model import ModelSpec
from test_prediction_system import case


def hybrid_config(**changes):
    return dict(
        model_type="qwen3_5_text", num_hidden_layers=2, hidden_size=32,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=64, vocab_size=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, **changes,
    )


def physical_work(config):
    raw = case(prompt=3, output=2)
    raw["model"] = config
    return [(op.family, op.flops, op.vector_ops, op.hbm_bytes, op.dtype)
            for op in target_operators(raw, phase="prefill", tokens=3)]


@pytest.mark.parametrize("changes", [
    {"hidden_act": "gelu"}, {"qkv_bias": True}, {"attention_bias": True},
    {"mlp_bias": True}, {"partial_rotary_factor": 0.5},
    {"sliding_window": 4, "use_sliding_window": True, "max_window_layers": 2},
    {"activation_dtype": "fp32"}, {"weight_dtype": "fp16"}, {"kv_dtype": "fp8"},
    {"expert_dtype": "fp8"},
    {"rope_scaling": {"rope_type": "dynamic", "factor": 2}},
])
def test_unconsumed_native_changes_keep_work_and_full_declarations(changes):
    original = hybrid_config()
    modified = {**original, **changes}
    model = ModelSpec.from_mapping(modified)
    metadata = dict(model.metadata)
    assert metadata["model_coverage"] == "partial"
    assert metadata["unconsumed_config_fields"] == changes
    assert physical_work(modified) == physical_work(original)
    assert ModelSpec.from_mapping(model.to_dict()) == model


def test_native_tied_embedding_removes_only_duplicate_resident_head_weights():
    untied = ModelSpec.from_mapping(hybrid_config())
    tied = ModelSpec.from_mapping(hybrid_config(tie_word_embeddings=True))
    assert tied.tie_word_embeddings
    assert {term.name for term in body_head_terms(untied)} == {"embedding", "final_norm", "lm_head"}
    assert {term.name for term in body_head_terms(tied)} == {"embedding", "final_norm"}
    assert untied.parameter_count() - tied.parameter_count() == 128 * 32
    assert dict(untied.catalog_facts)["resident_weight_bytes_body"] - dict(
        tied.catalog_facts)["resident_weight_bytes_body"] == 128 * 32 * 2
    # Tying storage never removes the logits computation.
    assert physical_work(hybrid_config(tie_word_embeddings=True)) == physical_work(hybrid_config())


@pytest.mark.parametrize("name", ["glm-5.3", "kimi-k3"])
@pytest.mark.parametrize("quant", [
    {"quant_method": "fp8", "fmt": "e5m2", "activation_scheme": "static",
     "weight_block_size": [64, 64]},
    {"quant_method": "compressed-tensors", "format": "int-quantized",
     "config_groups": {"group_0": {"weights": {"num_bits": 8}}}},
])
def test_native_quant_method_name_does_not_claim_an_unhandled_variant(name, quant):
    path = Path(__file__).parents[1] / "mindie_sim/model_configs" / (name + ".json")
    config = json.loads(path.read_text())
    config.get("text_config", config)["quantization_config"] = deepcopy(quant)
    model = ModelSpec.from_mapping(config)
    metadata = dict(model.metadata)
    assert metadata["model_coverage"] == "partial"
    assert any("quant" in gap and "unresolved" in gap for gap in metadata["unmodeled_features"])
    # Nested declarations remain available to comparison identity and callers.
    assert json.loads(json.dumps(metadata["unconsumed_config_fields"]["quantization_config"])) == quant
    assert ModelSpec.from_mapping(model.to_dict()) == model


def test_native_packing_does_not_imply_module_targets_or_expert_precision():
    directory = Path(__file__).parents[1] / "mindie_sim/model_configs"
    raw = json.loads((directory / "kimi-k3.json").read_text())
    config = raw.get("text_config", raw)
    config["quantization_config"].pop("ignore")
    group = config["quantization_config"]["config_groups"]["group_0"]
    group["targets"] = ["re:.*self_attn.*"]
    kimi = ModelSpec.from_mapping(raw)
    metadata = dict(kimi.metadata)
    assert kimi.quant.weight_format == "bf16" and kimi.quant.expert_weight_format == "mxfp4"
    assert "native_quant_config_groups_unresolved:retained_plan_proxy" in metadata["unmodeled_features"]
    assert metadata["unconsumed_config_fields"]["quantization_config"]["config_groups"][
        "group_0"]["targets"] == ("re:.*self_attn.*",)

    raw = json.loads((directory / "deepseek-v4.1-flash.json").read_text())
    raw.get("text_config", raw)["expert_dtype"] = "bf16"
    deepseek = ModelSpec.from_mapping(raw)
    metadata = dict(deepseek.metadata)
    assert deepseek.quant.expert_weight_format == "mxfp4"
    assert "native_expert_dtype:bf16:mxfp4_proxy" in metadata["unmodeled_features"]
    assert metadata["unconsumed_config_fields"]["expert_dtype"] == "bf16"


def test_native_dense_kv_sharing_remains_distinct_from_compressed_owner_sets():
    config = hybrid_config()
    config.update(num_hidden_layers=3, layer_types=[
        "linear_attention", "full_attention", "full_attention"],
        kv_source_layer_ids=[0, 1, 1])
    model = ModelSpec.from_mapping(config)
    assert [layer.kv_owner_layer_id for layer in model.layer_plans] == [0, 1, 1]
    consumer = model.layer_plans[2]
    assert not consumer.owns_long_kv
    assert not {"k_proj", "v_proj", "k_norm"} & {
        term.name for term in layer_weight_terms(consumer, model)}

    source = Path(__file__).parents[1] / "mindie_sim/model_configs/deepseek-v4.1-flash.json"
    raw = json.loads(source.read_text())
    text = raw.get("text_config", raw)
    compressed = load_model("deepseek-v4.1-flash")
    sources = text["kv_source_layer_ids"]
    assert len(sources) != compressed.layers
    for layer in compressed.layer_plans:
        if layer.compress_ratio:
            assert layer.kv_owner_layer_id == max(owner for owner in sources if owner <= layer.index)
            assert layer.owns_long_kv == (layer.index in sources)


@pytest.mark.parametrize("name", CATALOG)
def test_native_catalog_roundtrip_preserves_nested_facts_and_all_geometry(name):
    model = load_model(name)
    restored = ModelSpec.from_mapping(json.loads(json.dumps(model.to_dict())))
    assert restored == model
    assert restored.parameter_count() == model.parameter_count()
