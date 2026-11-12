"""Typed MX linear result ABI and recipe selection.

[Role]
  Distinguish a stored/header FP8 E4M3 dtype from the MX method: E8M0
  block-32 weight and dynamic-activation scales select the MX
  QuantBatchMatmulV3 path and the result returns the pre-quant BF16 dtype.

[Invariants]
  * Bare E4M3 without group-32 scale method is not MX and keeps fallback.
  * Explicit output_format int8 wins over MX inference.
  * Explicit format name mxfp8 still selects MX; scale-group method selects it
    for header-normalized fp8_e4m3 declarations.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.catalog.types import QuantSpec
from mindie_sim.prediction.piecewise.families_matrix import _gemm_mode
from mindie_sim.schema.subgraphs import StaticSignature


def _mx(name="W8A8_MXFP8"):
    return QuantSpec(
        name=name, weight_format="fp8_e4m3", activation_format="fp8_e4m3",
        weight_block=(1, 32), expert_block=32,
        scale_format="fp8_e8m0", scale_bytes=1.0,
        activation_group_size=32, activation_scale_format="fp8_e8m0",
        activation_scale_pack=2)


def _theta(weight_format="fp8_e4m3", act_format="fp8_e4m3", group=32,
           scale=1.0, output_format="", method="mx_fp8"):
    return StaticSignature(
        role="QuantBatchMatmulV3", matrix=(("K", 16384), ("N", 6144)),
        quant=(("weight_format", weight_format), ("act_format", act_format),
               ("scale_group", group), ("weight_scale", scale),
               ("linear_method", method),
               ("output_format", output_format)))


def test_mx_method_uses_runtime_prequant_dtype_and_regular_fp8_does_not():
    q = _mx()
    assert q.linear_method("gqa_o") == "mx_fp8"
    assert q.output_format_for("gqa_o", prequant="bf16") == "bf16"
    assert q.output_format_for("dense_mlp", prequant="fp16") == "fp16"
    # Regular FP8 even with group32 but FP32 scales is not MX.
    plain = replace(q, weight_block=(1, 32), activation_group_size=32,
                    scale_format="fp32", scale_bytes=4.0,
                    activation_scale_format="fp32")
    assert plain.linear_method("gqa_o") == "fp8"
    assert plain.output_format_for("gqa_o", "fp8_e4m3", prequant="bf16") == "fp8_e4m3"
    # E8M0/group32 weight but group64 activation is not MX.
    group64 = replace(q, activation_group_size=64)
    assert group64.linear_method("gqa_o") == "fp8"
    explicit = replace(q, output_format="int8")
    assert explicit.output_format_for("gqa_o", prequant="bf16") == "int8"


def test_typed_mx_selects_mx_recipe_but_regular_fp8_does_not():
    assert _gemm_mode("QuantBatchMatmulV3", _theta()) == "quant_batch_matmul_v3_mxfp8"
    assert _gemm_mode("QuantBatchMatmulV3", _theta(method="fp8")) == "quant_batch_matmul_v3"
    assert _gemm_mode("QuantBatchMatmulV3", _theta(method="w8a8")) == "quant_batch_matmul_v3"
    assert _gemm_mode("QuantBatchMatmulV3", _theta(method="mx_fp4")) == "quant_batch_matmul_v3"
