"""Declared precision boundaries compared with original native capture rows."""
from pathlib import Path

import pytest

from mindie_sim.structure_contracts import predicted_contract


def declaration(family, *, output, scale_dtype, scale_shape):
    return {
        "family": family, "recipe": family, "dtype": "bf16", "local_shape": (1024, 3584),
        "metadata": {
            "input_format": "bf16", "output_format": output,
            "activation_scale_shape": scale_shape, "activation_scale_format": scale_dtype,
            "norm_has_weight": True,
        },
    }


def test_dynamic_int8_and_mx_have_different_scale_domains():
    int8 = declaration("quantize", output="int8", scale_dtype="fp32", scale_shape=(1024,))
    mx = declaration("quantize", output="fp8_e4m3", scale_dtype="fp8_e8m0",
                     scale_shape=(1024, 56, 2))
    assert predicted_contract(int8) == (
        "dynamic_quant_active", "DynamicQuant", ((1024, 3584), "bf16", "ND"),
        ((1024, 3584), "int8", "ND"), ((1024,), "fp32", "ND"))
    assert predicted_contract(mx) == (
        "DynamicMxQuant", (1024, 3584), "bf16", "fp8_e4m3", (1024, 56, 2), "fp8_e8m0")
    int8["metadata"]["activation_scale_shape"] = (1024, 56, 2)
    assert predicted_contract(int8) is None


def test_real_norm_mx_contract_requires_both_norm_and_quantization(profiler_api):
    import mindie_profiler
    from mindie_profiler.normalized import kernel_rows
    from mindie_profiler.tensor_contracts import observed_contract

    root = Path(mindie_profiler.__file__).resolve().parents[1] / "tests/fixtures/fusion_contracts"
    rows = (row for path in sorted(root.glob("*.csv")) for row in kernel_rows(path))
    row = next(row for row in rows if row.kind == "RmsNormDynamicMxQuant")
    op = declaration("rmsnorm_quant", output="fp8_e4m3", scale_dtype="fp8_e8m0",
                     scale_shape=(1024, 56, 2))
    assert predicted_contract(op) == observed_contract(row)
    op["metadata"]["norm_weight_offset"] = 1.0
    assert predicted_contract(op) is None
