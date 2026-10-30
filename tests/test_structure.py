"""Structural preservation and independent tensor arithmetic."""
from dataclasses import replace
import json
from pathlib import Path

import pytest

from mindie_sim import Case
from mindie_sim.catalog.cache import layer_cache_components, new_completed_rows
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.cost import primitive_key
from mindie_sim.prediction.frontend import Frontend
from mindie_sim.schema.model import LayerPlan
from mindie_sim.schema.parallel import ParallelSpec
from mindie_sim.schema.workload import IterationPlan, RequestView
from test_prediction_system import case


@pytest.mark.parametrize("name", [
    "qwen3.8-27b-media", "kimi-k3", "deepseek-v4.1-flash", "glm53-a5", "kimi-k3-dspark",
])
def test_case_roundtrip_preserves_geometry_runtime_and_primitive_identity(name):
    directory = "components" if name == "kimi-k3-dspark" else "cases"
    path = Path(__file__).parents[1] / "examples" / directory / (name + ".json")
    first = Case.from_mapping(json.loads(path.read_text()))
    second = Case.from_mapping(json.loads(json.dumps(first.to_dict())))
    assert json.dumps(first.to_dict(), sort_keys=True) == json.dumps(second.to_dict(), sort_keys=True)
    op = gemm_op(op_id="p", deps=(), stage=0, layer=0, requests=(), m=3, n=64, k=128,
                 quant=first.model.quant, category="attn", phase="prefill")
    assert primitive_key(first, op, "none") == primitive_key(second, op, "none")


def test_layer_override_changes_arithmetic_and_residency_without_cell_matching():
    original = Case.from_mapping(case(prompt=8, output=1))
    changed = replace(original, model=replace(
        original.model,
        layer_plans=tuple(replace(layer, intermediate_size=128)
                          for layer in original.model.layer_plans)))
    plan = IterationPlan((RequestView("r", 8, 0, True, "prefill"),))
    a = Frontend()(plan, original.model, original.runtime, original.machine)
    b = Frontend()(plan, changed.model, changed.runtime, changed.machine)
    # SwiGLU: two input projections and one output projection per layer.
    expected_weights = 2 * 3 * 32 * (128 - 64) * 2
    assert sum(Memory(changed).weights) - sum(Memory(original).weights) == expected_weights
    assert sum(op.flops for op in b.ops) - sum(op.flops for op in a.ops) == 2 * 8 * expected_weights / 2


def test_cache_owners_windows_compressed_groups_and_state_are_distinct():
    parsed = Case.from_mapping(case())
    plan = LayerPlan(0, attention="mla", cache_kind="window_compressed",
                     window=4, compress_ratio=4, kv_lora_rank=8,
                     kv_owner_layer_id=0, owns_long_kv=True)
    cp = ParallelSpec()
    for tokens in (3, 4, 7, 8, 19):
        result = layer_cache_components(parsed.model, plan, cp, tokens, capacity_padded=False)
        assert result["window"] == min(tokens, 4) * 8 * 2
        assert result["compressed_kv"] == (tokens // 4) * 8 * 2
        consumer = replace(plan, index=1, owns_long_kv=False)
        assert "compressed_kv" not in layer_cache_components(parsed.model, consumer, cp, tokens)
    assert sum(new_completed_rows(start, length, 4) for start, length in ((0, 3), (3, 2), (5, 6))) == 11 // 4
    recurrent = LayerPlan(0, attention="gdn", recurrent_state_elements=32, conv_state_elements=12)
    assert layer_cache_components(parsed.model, recurrent, cp, 1) == layer_cache_components(
        parsed.model, recurrent, cp, 100_000)


def test_context_cache_allocates_whole_owned_blocks_but_transfer_counts_only_valid_rows():
    parsed = Case.from_mapping(case(tp=4))
    plan = replace(parsed.model.layer_plans[0], num_kv_heads=1)
    p = ParallelSpec(tp=4, cp=2)
    padded = layer_cache_components(parsed.model, plan, p, 129, interleave=128)
    payload = layer_cache_components(parsed.model, plan, p, 129, capacity_padded=False)
    assert padded["gqa_kv"] == 128 * 2 * 8 * 2
    assert payload["gqa_kv"] * 4 == 129 * 2 * 8 * 2 * 2


def test_quantization_changes_bytes_and_compute_precision_without_changing_flops():
    common = dict(op_id="m", deps=(), stage=0, layer=0, requests=(), m=3, n=64, k=128,
                  category="routed_expert", phase="decode")
    dense = gemm_op(quant=QuantSpec("bf16"), **common)
    packed = gemm_op(quant=QuantSpec("int8", weight_format="int8", expert_weight_format="int8"), **common)
    assert packed.flops == dense.flops == 2 * 3 * 64 * 128
    assert dict(packed.metadata)["packed_weight_bytes"] < dict(dense.metadata)["packed_weight_bytes"]
    assert packed.dtype == "int8"
    empty = gemm_op(quant=QuantSpec("bf16"), **{**common, "m": 0})
    assert empty.flops == empty.hbm_bytes == empty.vector_ops == 0


def test_catalog_dtype_and_layer_overrides_change_execution_and_capacity_together():
    raw = case()
    raw["model"] = {"catalog_key": "qwen3.8-27b", "layers": 2,
                    "quantization": "w8a8_dynamic", "kv_dtype": "fp32"}
    first = Case.from_mapping(raw)
    raw["model"]["quantization"] = "bf16"
    second = Case.from_mapping(raw)
    assert first.model.quant.kv_dtype == second.model.quant.kv_dtype == "fp32"
    assert sum(Memory(second).weights) > sum(Memory(first).weights)
    assert first.model.compute_dtype == second.model.compute_dtype == "bf16"
    assert first.model.parameter_count() == second.model.parameter_count()


def test_primitive_identity_includes_the_selected_kernel_program():
    original = Case.from_mapping(case())
    changed = replace(original, runtime=replace(original.runtime, kernels={
        "programs": {"matmul": {"kind": "tiled_matmul", "tile_k": 64}}}))
    op = gemm_op(op_id="p", deps=(), stage=0, layer=0, requests=(), m=8, n=64, k=256,
                 quant=original.model.quant, category="attn", phase="prefill")
    assert primitive_key(original, op, "none") != primitive_key(changed, op, "none")
