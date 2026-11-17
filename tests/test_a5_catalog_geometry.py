"""Catalog/config/HF geometry reaches the typed model and subgraph records.

[Role]
  Exercise the catalog-side geometry projection on real A5 catalog sets:
  declared rope/head/context attributes, per-value storage/compute/output
  formats, and the piecewise typed theta that consumes them.

[Invariants]
  * A declared FP16/FP32 source is not replaced by a BF16 proxy.
  * A 4-bit weight role has a high-precision output value; explicit output
    declarations still win.
  * MLA index/head attributes survive catalog assembly.
  * A real A5 model produces typed values with distinct activation, weight,
    output and scale formats; service-only families are absent from a body
    graph.
  * No predicate depends on a model-name latency constant or a measurement.

[Timeouts & Locks]
  Pure CPU, bounded to a few catalog models and one small A5 case.

[Side-Effect Policy]
  Read-only catalog/config access plus one pytest temporary config copy; no
  weights, devices or repository writes.
"""
from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from types import SimpleNamespace

from mindie_sim import target_operators
from mindie_sim.catalog.ops import gemm_op
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weights import layer_weight_terms
from mindie_sim.model_catalog import (
    ARCHITECTURE_PRESET_ALIASES, ARCHITECTURE_PRESET_CATALOG, CATALOG,
    MODEL_ALIASES, build_model_spec, load_model,
    model_value_formats,
)
from mindie_sim.prediction.lowering import Builder
from mindie_sim.prediction.frontend_features import FeatureConfig
from mindie_sim.schema.model import LayerPlan, ModelSpec
from mindie_sim.schema.serialization import apply_model_overrides
from mindie_sim.structure import export_structure
from test_prediction_system import case

ROOT = Path(__file__).parents[1]
EXPECTED_CATALOG_COUNT = 37


def test_catalog_key_config_alias_inventory_is_preserved():
    assert len(CATALOG) == EXPECTED_CATALOG_COUNT
    assert len(ARCHITECTURE_PRESET_CATALOG) == 5
    for key in (*CATALOG, *ARCHITECTURE_PRESET_CATALOG):
        assert load_model(key).layers > 0
    for alias, key in MODEL_ALIASES.items():
        assert key in CATALOG, alias
        assert load_model(alias) == load_model(key)
    for alias, key in ARCHITECTURE_PRESET_ALIASES.items():
        assert key in ARCHITECTURE_PRESET_CATALOG, alias
        assert load_model(alias) == load_model(key)


def test_declared_rope_and_context_reach_feature_metadata():
    deepseek = load_model("deepseek-v4.1-flash")
    rope = dict(deepseek.metadata)["rope_parameters"]
    assert rope["rope_theta"] == 10000
    assert rope["rope_type"] == "yarn"
    assert rope["factor"] == 16
    features = FeatureConfig.from_model(deepseek)
    promoted = dict(features.metadata)
    assert promoted["rope_theta"] == 10000
    assert promoted["rope_type"] == "yarn"
    assert promoted["rope_parameters"]["factor"] == 16

    deepseek_formats = model_value_formats(deepseek)
    expert = deepseek_formats["matrix_roles"]["routed_expert"]
    assert (expert["weight_storage"], expert["activation"],
            expert["compute"], expert["output"]) == (
        "mxfp4", "fp8_e4m3", "bf16", "bf16")
    assert deepseek_formats["heads"]["mla"]["index_topk"] == 512

    qwen = load_model("qwen3.6-27b")
    values = model_value_formats(qwen)
    assert values["rotary"]["rope_parameters"]["partial_rotary_factor"] == 0.25
    assert values["rotary"]["rope_parameters"]["mrope_section"] == [11, 11, 10]
    assert values["rotary"]["rotary_dim_by_attention"]["gqa"] == (64,)
    assert values["rotary"]["max_position_embeddings"] == 262144

    explicit = ModelSpec.from_mapping(dict(
        name="explicit-rope", layers=1, hidden_size=64, num_heads=4,
        num_kv_heads=2, head_dim=16, intermediate_size=64, vocab_size=128,
        layer_plans=[dict(index=0, attention="gqa", ffn="dense")],
        metadata=[("unconsumed_config_fields", {
            "rope_parameters": {"partial_rotary_factor": 0.25}})],
    ))
    assert model_value_formats(explicit)["rotary"]["rotary_dim_by_attention"]["gqa"] == (4,)

    glm = load_model("glm-5.3")
    assert glm.mla is not None
    assert glm.mla.index_topk == 2048
    assert glm.mla.index_head_dim == 128
    assert glm.mla.index_n_heads == 32
    assert glm.layer_plans[0].index_topk == 2048


def test_declared_native_dtype_is_not_replaced_by_bf16_proxy():
    raw = json.loads((ROOT / "mindie_sim/model_configs/qwen3.5-27b.json").read_text())
    raw["torch_dtype"] = "float16"
    model = build_model_spec("external-fp16", raw)
    assert model.activation_dtype == model.weight_dtype == model.compute_dtype == "fp16"
    gaps = dict(model.metadata)["unmodeled_features"]
    assert not any("native_activation_dtype" in gap for gap in gaps)
    features = FeatureConfig.from_model(model)
    assert (features.activation_dtype, features.weight_dtype,
            features.kv_dtype) == ("fp16", "fp16", "fp16")


def test_redundant_operand_dtype_does_not_change_direct_output():
    quant = QuantSpec(
        name="w4a8", weight_format="int4", expert_weight_format="int4",
        activation_format="int8", native_format="bf16")
    args = dict(deps=(), stage=0, layer=0, requests=(), m=1, n=128, k=32,
                quant=quant, category="dense_mlp", phase="decode")
    implicit = gemm_op(op_id="implicit", **args)
    explicit = gemm_op(op_id="explicit", activation_dtype="int8", **args)
    for op in (implicit, explicit):
        meta = dict(op.metadata)
        assert (meta["activation_format"], meta["compute_format"],
                meta["output_format"]) == ("int8", "int8", "bf16")
        assert op.flops == 8192
        assert op.mte3_bytes == 256
    fp32 = gemm_op(op_id="fp32", output_dtype="fp32", **args)
    int8_out = gemm_op(op_id="int8", output_dtype="int8", **args)
    assert fp32.mte3_bytes == 512
    assert int8_out.mte3_bytes == 128

    plan = LayerPlan(index=0, attention="gqa", ffn="dense")
    builder = Builder("op", plan, 0, (),
                      SimpleNamespace(quant=quant, activation_dtype="bf16",
                                      compute_dtype="bf16"), "decode")
    builder.matrix("m", (), 1, 128, 32, category="dense_mlp",
                   activation_dtype="int8")
    built = builder.ops[-1]
    meta = dict(built.metadata)
    assert (meta["activation_format"], meta["output_format"]) == ("int8", "bf16")
    assert built.mte3_bytes == 256


def test_mx_prequant_is_independent_of_operand_override():
    mx8 = QuantSpec(
        name="mx8", weight_format="fp8_e4m3", expert_weight_format="fp8_e4m3",
        activation_format="fp8_e4m3", weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0, activation_group_size=32,
        activation_scale_format="fp8_e8m0", activation_scale_pack=2)
    mx4 = replace(mx8, name="mx4", weight_format="mxfp4",
                  expert_weight_format="mxfp4")
    for quant in (mx8, mx4):
        args = dict(deps=(), stage=0, layer=0, requests=(), m=1, n=128, k=33,
                    quant=quant, category="dense_mlp", phase="decode")
        implicit = gemm_op(op_id="implicit", **args)
        explicit = gemm_op(op_id="explicit", activation_dtype="fp8_e4m3", **args)
        assert dict(implicit.metadata)["output_format"] == "bf16"
        assert dict(explicit.metadata)["output_format"] == "bf16"
        assert implicit.mte3_bytes == explicit.mte3_bytes == 256
        assert implicit.mte2_bytes == explicit.mte2_bytes
        fp16 = gemm_op(op_id="fp16", activation_dtype="fp8_e4m3",
                       prequant_dtype="fp16", **args)
        assert dict(fp16.metadata)["output_format"] == "fp16"
        assert fp16.mte3_bytes == 256
        fp32 = gemm_op(op_id="fp32", output_dtype="fp32",
                       activation_dtype="fp8_e4m3", **args)
        int8_out = gemm_op(op_id="int8", output_dtype="int8",
                           activation_dtype="fp8_e4m3", **args)
        assert fp32.mte3_bytes == 512
        assert int8_out.mte3_bytes == 128
        assert dict(fp32.metadata)["weight_scale_format"] == "fp8_e8m0"
        assert dict(fp32.metadata)["activation_scale_format"] == "fp8_e8m0"


def test_flash_w8a8_layer_projection_and_ffn_formats():
    model = ModelSpec.from_mapping({
        "catalog_key": "glm-5.3-flash", "quantization": "w8a8", "layers": 4})
    formats = model_value_formats(model)
    mla = formats["matrix_roles"]["mla"]
    assert (mla["weight_storage"], mla["activation"], mla["compute"],
            mla["output"]) == ("bf16", "bf16", "bf16", "bf16")
    assert mla["layers"] == (3,)
    assert mla["weight_scale"] == mla["activation_scale"] == ""
    assert formats["matrix_roles"]["mla.q_a"]["weight_storage"] == "bf16"
    assert formats["matrix_roles"]["mla.kv_a"]["weight_storage"] == "bf16"
    assert formats["matrix_roles"]["dense_mlp"]["layers"] == (0, 1, 2)
    assert formats["matrix_roles"]["dense_mlp"]["weight_storage"] == "int8"
    ffn = formats["matrix_roles"]["routed_expert"]
    assert (ffn["weight_storage"], ffn["activation"], ffn["output"]) == (
        "int8", "int8", "bf16")
    assert ffn["layers"] == (3,)
    assert ffn["weight_scale"] and ffn["activation_scale"]

    raw = case(prompt=1, output=1)
    raw["model"] = model.to_dict()
    ops = [op for op in target_operators(raw, phase="decode", tokens=1)
           if op.layer == 3]
    packed = next(op for op in ops if op.recipe == "qkv_a")
    assert (dict(packed.metadata)["weight_format"],
            dict(packed.metadata)["activation_format"],
            dict(packed.metadata)["output_format"]) == ("bf16", "bf16", "bf16")
    expert = next(op for op in ops if op.op_id.endswith(".expert_up"))
    assert (dict(expert.metadata)["weight_format"],
            dict(expert.metadata)["activation_format"],
            dict(expert.metadata)["output_format"]) == ("int8", "int8", "bf16")

    overridden_quant = replace(model.quant, category_overrides=(
        ("mla.q_a", (("weight_format", "int8"), ("activation_format", "int8"))),))
    overridden = replace(model, quant=overridden_quant)
    child = model_value_formats(overridden)["matrix_roles"]
    assert child["mla"]["weight_storage"] == "bf16"
    assert child["mla.q_a"]["weight_storage"] == "int8"
    assert child["mla.kv_a"]["weight_storage"] == "bf16"
    raw["model"] = overridden.to_dict()
    split = [op for op in target_operators(raw, phase="decode", tokens=1)
             if op.layer == 3 and op.family == "gemm"]
    assert not any(op.recipe == "qkv_a" for op in split)
    q_a = next(op for op in split if op.recipe == "q_a")
    kv_a = next(op for op in split if op.recipe == "kv_a")
    assert dict(q_a.metadata)["weight_format"] == "int8"
    assert dict(kv_a.metadata)["weight_format"] == "bf16"
    terms = {term.name: term for term in layer_weight_terms(model.layer_plans[3], overridden)}
    assert terms["q_a"].quant.format_for("mla.q_a") == "int8"
    assert terms["kv_a"].quant.format_for("mla.kv_a") == "bf16"


def test_hybrid_fp32_direct_output_is_512_bytes():
    model = ModelSpec.from_mapping(dict(
        model_type="qwen3_5_text", num_hidden_layers=2, hidden_size=32,
        num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        intermediate_size=64, vocab_size=128,
        layer_types=["linear_attention", "full_attention"],
        linear_num_key_heads=2, linear_num_value_heads=2,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_conv_kernel_dim=4, weight_dtype="fp32",
    ))
    dense = model_value_formats(model)["matrix_roles"]["dense_mlp"]
    assert (dense["weight_storage"], dense["activation"],
            dense["compute"], dense["output"]) == (
        "fp32", "fp32", "fp32", "fp32")
    raw = case(prompt=1, output=1)
    raw["model"] = model.to_dict()
    gate_up = next(op for op in target_operators(raw, phase="decode", tokens=1)
                   if op.op_id.endswith(".gate_up"))
    assert gate_up.local_shape == (1, 128, 32)
    assert dict(gate_up.metadata)["output_format"] == "fp32"
    assert gate_up.mte3_bytes == 512


def test_quant_spec_runtime_output_fields_roundtrip_and_override():
    quant = QuantSpec(
        name="runtime-abi", weight_format="int8", activation_format="int8",
        expert_weight_format="int8", weight_block=(128, 128), scale_bytes=2.0,
        runtime_weight_scale_format="bf16", runtime_weight_scale_bytes=2.0,
        runtime_activation_scale_format="fp32",
        runtime_int8_weight_metadata="scale_only", output_format="bf16",
    )
    restored = QuantSpec(**json.loads(json.dumps(quant.to_dict())))
    assert restored == quant
    assert restored.execution_weight_scale_format() == "bf16"
    assert restored.execution_weight_scale_bytes() == 2.0
    assert restored.execution_activation_scale_format() == "fp32"
    assert restored.execution_int8_weight_metadata() == "scale_only"
    assert restored.output_format == "bf16"
    model = apply_model_overrides(load_model("qwen3.5-27b"),
                                  {"quant": quant.to_dict()})
    assert model.quant == quant
    restored_model = ModelSpec.from_mapping(json.loads(json.dumps(model.to_dict())))
    assert restored_model.quant == quant


def test_custom_config_root_uses_the_same_native_resolution(tmp_path):
    raw = json.loads((ROOT / "mindie_sim/model_configs/qwen3.5-27b.json").read_text())
    raw["torch_dtype"] = "float16"
    (tmp_path / "qwen3.5-27b.json").write_text(json.dumps(raw))
    model = load_model("qwen3.5-27b", config_root=tmp_path)
    assert (model.activation_dtype, model.weight_dtype,
            model.kv_dtype) == ("fp16", "fp16", "fp16")


def test_router_output_and_category_scales_use_the_effective_plan():
    glm = load_model("glm-5.3")
    router = model_value_formats(glm)["matrix_roles"]["router"]
    assert (router["weight_storage"], router["activation"],
            router["compute"], router["output"]) == (
        "fp32", "fp32", "fp32", "fp32")
    moe_layers = tuple(p.index for p in glm.layer_plans if p.ffn == "moe")
    assert router["layers"] == moe_layers

    overridden = replace(glm.quant, category_overrides=(
        ("routed_expert", (
            ("weight_format", "mxfp4"), ("scale_format", "fp8_e8m0"),
            ("scale_bytes", 1.0), ("activation_scale_format", "fp8_e8m0"))),
    ))
    expert = model_value_formats(replace(glm, quant=overridden))["matrix_roles"]["routed_expert"]
    assert expert["weight_storage"] == "mxfp4"
    assert expert["weight_scale"] == expert["activation_scale"] == "fp8_e8m0"


def test_no_quant_mixed_representation_keeps_model_fields():
    model = ModelSpec.from_mapping(dict(
        name="mixed", layers=1, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=64, vocab_size=128,
        activation_dtype="fp16", weight_dtype="bf16", kv_dtype="fp8",
        compute_dtype="fp32",
        layer_plans=[dict(index=0, attention="gqa", ffn="dense")],
    ))
    formats = model_value_formats(model)
    assert (formats["activation_stream"], formats["compute"],
            formats["kv_cache"]) == ("fp16", "fp32", "fp8_e4m3")
    dense = formats["matrix_roles"]["dense_mlp"]
    assert (dense["weight_storage"], dense["activation"],
            dense["compute"], dense["output"]) == (
        "bf16", "bf16", "fp32", "bf16")
    # The FP16 stream value is not the direct BF16 matrix result; a cast to
    # the stream is a separate producer, not a second output declaration.
    assert formats["output_stream"] == "fp16"
    assert dense["weight_scale"] == dense["activation_scale"] == ""


def test_heterogeneous_router_formats_keep_layer_binding():
    model = ModelSpec.from_mapping(dict(
        name="hetero", layers=2, hidden_size=64, num_heads=4, num_kv_heads=2,
        head_dim=16, intermediate_size=64, vocab_size=128,
        num_experts=4, top_k=2, moe_intermediate_size=32,
        layer_plans=[
            dict(index=0, attention="gqa", ffn="moe", num_heads=4,
                 num_kv_heads=2, head_dim=16, num_experts=4, top_k=2,
                 moe_intermediate_size=32, router_dtype="fp32"),
            dict(index=1, attention="gqa", ffn="moe", num_heads=4,
                 num_kv_heads=2, head_dim=16, num_experts=4, top_k=2,
                 moe_intermediate_size=32, router_dtype="bf16"),
        ],
    ))
    router = model_value_formats(model)["matrix_roles"]["router"]
    assert router["heterogeneous"] is True
    assert router["by_layer"][0]["output"] == "fp32"
    assert router["by_layer"][1]["output"] == "bf16"


def test_four_bit_matrix_declares_a_high_precision_output_value():
    quant = QuantSpec(
        name="mxfp4_experts", weight_format="fp8_e4m3",
        activation_format="fp8_e4m3", expert_weight_format="mxfp4",
        weight_block=(128, 128), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=128, activation_scale_format="fp8_e8m0",
    )
    assert quant.output_format_for("routed_expert", prequant="bf16") == "bf16"
    explicit = QuantSpec(
        name="explicit", weight_format="mxfp4", expert_weight_format="mxfp4",
        output_format="int8", scale_format="fp8_e8m0", scale_bytes=1.0,
    )
    assert explicit.output_format_for("routed_expert", prequant="bf16") == "int8"
    plain = QuantSpec(
        name="fp8", weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        weight_block=(128, 128), scale_format="fp32", scale_bytes=4.0,
    )
    assert plain.output_format_for("attn.q", "fp8_e4m3", prequant="bf16") == "fp8_e4m3"


def test_real_a5_model_typed_theta_keeps_per_value_formats():
    model = load_model("kimi-k3")
    values = model_value_formats(model)
    expert = values["matrix_roles"]["routed_expert"]
    assert (expert["weight_storage"], expert["weight_runtime"]) == ("mxfp4", "mxfp4")
    assert expert["activation"] == expert["compute"] == expert["output"] == "bf16"
    assert expert["weight_scale"] == "fp8_e8m0"
    assert values["matrix_roles"]["dense_mlp"]["weight_scale"] == ""
    raw = case(prompt=1, output=1)
    raw["model"] = model.to_dict()
    ops = target_operators(raw, phase="decode", tokens=1)
    expert_op = next(op for op in ops
                     if op.family == "grouped_gemm"
                     and dict(op.metadata).get("quant_category") == "routed_expert")
    expert_meta = dict(expert_op.metadata)
    assert (expert_meta["weight_format"], expert_meta["activation_format"],
            expert_meta["compute_format"], expert_meta["output_format"]) == (
        "mxfp4", "bf16", "bf16", "bf16")
    assert expert_meta["weight_scale_format"] == "fp8_e8m0"
    dense_op = next(op for op in ops
                    if dict(op.metadata).get("quant_category") == "dense_mlp")
    dense_meta = dict(dense_op.metadata)
    assert (dense_meta["weight_format"], dense_meta["activation_format"],
            dense_meta["output_format"]) == ("bf16", "bf16", "bf16")
    assert "weight_scale_format" not in dense_meta


def test_a5_body_typed_graph_has_distinct_value_formats_and_no_service_nodes():
    case = json.loads((ROOT / "examples/cases/qwen36-a5.json").read_text())
    report = export_structure(case, phase="decode", tokens=1, context=17, scope="body")
    assert report["model"]["name"] == "qwen3.6-27b"
    families = {node["family"] for node in report["nodes"]}
    assert not families & {"sampling", "acceptance", "token_commit"}
    matrix = next(
        node for node in report["nodes"]
        if node["family"] == "gemm" and node.get("values", {}).get("basis") == "contract"
    )
    typed = matrix["values"]
    quant = typed["quantization"]
    assert quant["input_format"] == "int8"
    assert quant["weight_format"] == "int8"
    assert quant["compute_format"] == "int8"
    assert quant["output_format"] == "bf16"
    inputs = {value["name"]: value["dtype"] for value in typed["inputs"]}
    outputs = {value["name"]: value["dtype"] for value in typed["outputs"]}
    assert inputs["activation"] == "int8"
    assert inputs["weight"] == "int8"
    assert outputs["output"] == "bf16"
