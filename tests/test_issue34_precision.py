"""Generic MXA8W4 execution precision and declared supply aliases."""
from __future__ import annotations

from dataclasses import replace

import pytest

from mindie_sim.catalog.ops import compute_dtype_for, gemm_op
from mindie_sim.prediction.matrix_precision import (
    execution_compute_peaks, resolve_matrix_precision,
)
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.compute_peak_aliases import (
    native_compute_peak_items, resolve_compute_peak_name,
    validate_compute_peak_aliases,
)
from mindie_sim.schema.hardware import Hardware
from issue34_helpers import hardware, quant


def a5():
    return Hardware.from_mapping({"backend": "ascend", "sku": "Ascend950DT_9582",
                                  "num_devices": 1})


def a2():
    return Hardware.from_mapping({"backend": "ascend", "sku": "Ascend910B2",
                                  "num_devices": 1})


def mxa8w4(activation="fp8_e4m3", *, category="moe.routed"):
    return gemm_op(
        op_id="mx", deps=("x",), stage=0, layer=0, requests=("r",),
        m=4, n=64, k=128, quant=quant(
            weight_format="mxfp4", activation_format=activation),
        category=category, phase="decode")


def test_structural_identity_and_two_argument_dtype_contract():
    op = mxa8w4()
    metadata = dict(op.metadata)
    assert op.dtype == "bf16"
    assert metadata["weight_format"] == "mxfp4"
    assert metadata["activation_format"] == "fp8_e4m3"
    assert metadata["output_format"] == "bf16"
    assert compute_dtype_for(quant(weight_format="mxfp4",
                                   activation_format="fp8_e4m3"),
                             "moe.routed") == ("bf16", "fp8_e4m3")
    assert metadata["compute_format_origin"] == "catalog_default"


def test_eligible_generic_work_resolves_to_fp8_on_a5():
    op = mxa8w4()
    resolved = resolve_matrix_precision(op, devices=a5().dies, policy="auto",
                                        recipe=None)
    metadata = dict(resolved.metadata)
    assert resolved.dtype == "fp8_e4m3"
    assert metadata["declared_compute_format"] == "bf16"
    assert metadata["execution_compute_format"] == "fp8_e4m3"
    assert metadata["weight_format"] == "mxfp4"
    assert metadata["activation_format"] == "fp8_e4m3"
    assert resolved.local_shape == op.local_shape
    assert resolved.flops == op.flops
    peaks = execution_compute_peaks(
        resolved, devices=a5().dies,
        base_peaks={"bf16": a5().dies[0].peak("bf16")})
    assert peaks["fp8_e4m3"] == pytest.approx(2 * peaks["bf16"])


def test_bf16_only_fp8_only_and_mixed_dies_keep_a_common_legal_precision():
    bf16_only = hardware(1, peaks=(("bf16", 1e12),))
    op = mxa8w4()
    kept = resolve_matrix_precision(op, devices=bf16_only.dies, policy="auto",
                                    recipe=None)
    assert kept.dtype == "bf16"
    assert dict(kept.metadata)["precision_resolution_reason"] == "fp8_unavailable"

    fp8_only = hardware(1, peaks=(("bf16", 1e12), ("fp8", 2e12)))
    resolved = resolve_matrix_precision(op, devices=fp8_only.dies, policy="auto",
                                        recipe=None)
    assert resolved.dtype == "fp8_e4m3"

    mixed = (replace(a5().dies[0], rank=0), replace(a2().dies[0], rank=1))
    kept = resolve_matrix_precision(op, devices=mixed, policy="auto", recipe=None)
    assert kept.dtype == "bf16"


def test_declared_policy_and_source_bound_identity_are_preserved():
    op = mxa8w4()
    assert resolve_matrix_precision(op, devices=a5().dies, policy="declared",
                                    recipe=None) == op
    source = replace(op, source_boundary="kernel")
    kept = resolve_matrix_precision(source, devices=a5().dies, policy="auto",
                                    recipe=None)
    assert kept.dtype == "bf16"
    assert dict(kept.metadata)["precision_resolution_reason"] == "explicit_compute_identity"
    explicit = gemm_op(
        op_id="e", deps=(), stage=0, layer=0, requests=(), m=2, n=8, k=8,
        quant=quant(weight_format="mxfp4", activation_format="fp8_e4m3"),
        category="moe.routed", phase="decode",
        metadata=(("compute_format", "bf16"),))
    assert dict(explicit.metadata)["compute_format_origin"] == "explicit"
    assert resolve_matrix_precision(explicit, devices=a5().dies, policy="auto",
                                    recipe=None).dtype == "bf16"


def test_e5m2_requires_real_native_support():
    op = mxa8w4("fp8_e5m2")
    resolved = resolve_matrix_precision(op, devices=a5().dies, policy="auto",
                                        recipe=None)
    assert resolved.dtype == "fp8_e5m2"
    no_e5 = hardware(1)
    kept = resolve_matrix_precision(op, devices=no_e5.dies, policy="auto",
                                    recipe=None)
    assert kept.dtype == "bf16"


def test_supply_alias_is_validated_and_never_a_native_operand():
    base = hardware(1, peaks=(("bf16", 1e12), ("fp8_e4m3", 2e12),
                              ("mxfp4", 2e12)))
    die = replace(base.dies[0], compute_peak_aliases=(("mxfp4", "fp8_e4m3"),))
    peaks = dict(die.compute_peaks)
    validate_compute_peak_aliases(peaks, dict(die.compute_peak_aliases), die.compute_arch)
    assert resolve_compute_peak_name("mxfp4", peaks,
                                     dict(die.compute_peak_aliases)) == "fp8_e4m3"
    assert "mxfp4" not in dict(native_compute_peak_items(
        peaks, dict(die.compute_peak_aliases)))
    assert die.peak("mxfp4") == pytest.approx(die.peak("fp8_e4m3"))
    with pytest.raises(ModelingError, match="matrix_precision_not_native"):
        execution_compute_peaks(replace(mxa8w4(), dtype="mxfp4"),
                                devices=(die,), base_peaks={})
    with pytest.raises(ModelingError, match="compute_peak_alias_value_mismatch"):
        validate_compute_peak_aliases({"bf16": 1.0, "fp8_e4m3": 2.0, "mxfp4": 3.0},
                                      {"mxfp4": "fp8_e4m3"}, die.compute_arch)
    with pytest.raises(ModelingError, match="compute_peak_alias_unknown_target"):
        validate_compute_peak_aliases({"bf16": 1.0, "mxfp4": 1.0},
                                      {"mxfp4": "fp8_e5m2"}, die.compute_arch)
    with pytest.raises(ModelingError, match="compute_peak_alias_cycle"):
        validate_compute_peak_aliases({"a": 1.0, "b": 1.0},
                                      {"a": "b", "b": "a"}, die.compute_arch)
