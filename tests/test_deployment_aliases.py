"""Catalog declarations preserve representation work and hardware support."""
from copy import deepcopy

import pytest

from mindie_sim import predict, target_graphs
from mindie_sim.catalog.quant import quant_preset
from mindie_sim.model_catalog import load_architecture_profile, load_model
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError


SHORT_MODELS = (
    ("deepseek-v4-flash", "deepseek-v4-flash-0731"),
    ("deepseek-v4-pro", "deepseek-v4-pro-0813"),
    ("deepseek-v4.1", "deepseek-v4.1-flash"),
    ("qwen3.8", "qwen3.8-27b"),
)


def deployment_case(model, quantization, sku, *, tp=8):
    # Keep native hidden/attention/expert dimensions; bound the acceptance graph
    # to the first declared layer and a small LM head, without device workloads.
    return {
        "machine": {"backend": "ascend", "sku": sku, "num_devices": tp},
        "model": {
            "preset": model, "quantization": quantization,
            "layers": 1, "vocab_size": 128, "auxiliary": None,
        },
        "workload": {"requests": [
            {"id": "r", "prompt_tokens": 1, "output_tokens": 1},
        ]},
        "runtime": {
            "tp": tp, "max_num_batched_tokens": 16,
            "kernels": {"programs": {"matmul": {"kind": "steady_matmul"}}},
        },
    }


@pytest.mark.parametrize("short,canonical", SHORT_MODELS)
def test_model_short_label_preserves_exact_geometry_quantization_and_weights(short, canonical):
    model = load_model(canonical)
    assert load_model(short) == model
    assert load_model(short.upper()) == model
    assert load_architecture_profile(short) == load_architecture_profile(canonical)


def test_quant_aliases_preserve_real_packed_bytes_and_checkpoint_exclusions():
    w4 = quant_preset("w4a8")
    assert w4 == quant_preset("w4a8_dynamic")
    # Odd K exercises actual INT4 packing, one channel scale and the mixed
    # non-expert INT8 policy rather than a blanket half-byte estimate.
    assert w4.matrix_bytes(3, 5, "routed_expert") == 3 * 3 + 3 * 4
    assert w4.matrix_bytes(3, 5, "attn") == 3 * 5 + 3 * 4 * 2
    assert quant_preset("w8a8") == quant_preset("w8a8_dynamic")
    assert quant_preset("bf16").matrix_bytes(3, 5, "routed_expert") == 3 * 5 * 2
    fp8 = quant_preset(" FP8 ")
    assert fp8 == quant_preset("w8a8_mxfp8")
    assert fp8.matrix_bytes(3, 33, "dense_mlp") == 3 * 33 + 3 * 2
    assert fp8.matrix_bytes(3, 33, "gdn") == 3 * 33 * 2
    assert fp8.matrix_bytes(3, 33, "mtp") == 3 * 33 * 2
    assert dict(fp8.source)["checkpoint"] == "Eco-Tech/Qwen3.8-27B-w8a8-mxfp8"


@pytest.mark.parametrize("short,canonical", (
    ("w4a8", "w4a8_dynamic"), ("w8a8", "w8a8_dynamic"),
    ("fp8", "w8a8_mxfp8"),
))
def test_alias_and_canonical_execute_identical_public_programs(short, canonical):
    raw = deployment_case("qwen3.8", short, "Ascend950DT_9581")
    reference = deepcopy(raw)
    reference["model"]["quantization"] = canonical
    assert target_graphs(raw, phase="prefill", tokens=1) == target_graphs(
        reference, phase="prefill", tokens=1)
    actual, expected = predict(raw), predict(reference)
    assert actual["requests"][0]["status"] == "served"
    assert actual["execution"]["serviced"] == expected["execution"]["serviced"]
    assert actual["metrics"] == expected["metrics"]


@pytest.mark.parametrize("model", (
    "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-v4.1",
    "qwen3.8", "glm-5.3", "kimi-k3", "minimax-m3",
))
@pytest.mark.parametrize("quantization", ("w8a8", "w4a8", "bf16"))
@pytest.mark.parametrize("sku", ("Ascend910_9391", "Ascend950DT_9581"))
def test_supported_representations_reach_public_serving_and_preserve_coverage(
    model, quantization, sku,
):
    raw = deployment_case(model, quantization, sku)
    parsed = parse_case(raw)
    result = predict(raw)
    assert result["status"] == "modeled"
    assert result["requests"][0]["status"] == "served"
    assert result["execution"]["live_tasks"] == result["execution"]["live_programs"] == 0
    assert all(abs(value) < 1e-6
               for value in result["execution"]["live_allocated"].values())
    assert result["execution"]["work"]["flops"] > 0
    coverage = dict(parsed.model.metadata).get("model_coverage", "complete")
    assert result["model_coverage"] == coverage
    if model == "minimax-m3":
        assert coverage == "partial"
        assert result["unmodeled_features"]


def test_fp8_alias_does_not_grant_a3_an_undeclared_matrix_capability():
    raw = deployment_case("qwen3.8", "fp8", "Ascend910_9391")
    assert parse_case(raw).model.quant == quant_preset("w8a8_mxfp8")
    with pytest.raises(ModelingError, match="compute_precision_rate_required:fp8_e4m3"):
        predict(raw)
    raw["machine"]["sku"] = "Ascend950DT_9581"
    supported = predict(raw)
    assert supported["requests"][0]["status"] == "served"
    assert supported["execution"]["live_tasks"] == 0
