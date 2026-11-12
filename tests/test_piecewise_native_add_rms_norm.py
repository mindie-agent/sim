"""Focused source/typed-gate tests for the R31 AddRmsNorm adapter."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mindie_sim.prediction.native_binding import bind_native_call, declare_native_call
from mindie_sim.prediction.piecewise import native_add_rms_norm as adapter
from mindie_sim.prediction.piecewise import native_calls


class _Theta:
    def __init__(self, values):
        self.values = dict(values)

    def find(self, key, default=None):
        return self.values.get(key, default)


def _compiled(*, kind="N0", name="rmsnorm", siblings=("residual_add", "rmsnorm"),
              image_id=adapter.R31_IMAGE_ID, image=adapter.R31_IMAGE_TAG,
              **theta):
    values = {
        "elements": 5120,
        "gamma_bytes": 10240.0,
        "gamma_format": "bf16",
        "sequence_rows": "shard",
        "sequence_ranks": 2,
        "layernorm": 0,
        "norm_weight_offset": 0.0,
        "norm_arch": "rmsnorm",
        "norm_eps": 1e-6,
        "rstd_bytes": 4.0 if kind == "O0" else None,
        "residual_out_bytes": 10240.0 if kind == "O0" else None,
        "residual_add": 1 if kind == "O0" else None,
        "fusion": "add_rmsnorm" if kind == "O0" else None,
    }
    values.update(theta)
    operator = SimpleNamespace(
        family="elementwise", name=name, role="norm",
        template=SimpleNamespace(role="norm"), theta=_Theta(values))
    return SimpleNamespace(
        kind=kind, layer=0, operator=operator,
        siblings=tuple(SimpleNamespace(name=item) for item in siblings),
        feature_metadata=(("image_id", image_id), ("image", image)),
    )


def _binding(compiled):
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    surface = native_calls._surface(
        compiled, spec,
        source_boundary="call" if len(spec.members) > 1 else "kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None))


def _geometry(tokens=4096, phase="prefill", selected=1):
    return SimpleNamespace(
        effective_tokens=tokens, phase_name=phase,
        producing_lm_head_rows=lambda: selected)


def test_inner_n0_selects_one_add_rms_norm_call():
    spec = adapter.binding_spec(_compiled())
    assert spec is not None
    assert spec.api["entry"] == "aclnnAddRmsNorm"
    assert spec.api["namespace"] == "aclnn"
    assert spec.owner == "fused:add_rmsnorm"
    assert spec.members == ("residual_add", "rmsnorm")
    assert spec.typed["attrs"]["hidden"] == 5120
    assert spec.typed["attrs"]["sequence_rows"] == "shard"
    assert spec.typed["attrs"]["sequence_ranks"] == 2
    assert spec.typed["shapes"]["gamma"] == [5120]
    assert spec.typed["shapes"]["rstd"] == [1]
    assert spec.source["sha256"] == adapter.CANN_LIBOPAPI["sha256"]
    assert spec.source["state"] == "task_artifact"
    assert "python_caller_module_not_retained" in spec.unresolved


def test_terminal_o0_selects_same_kernel_but_own_owner():
    spec = adapter.binding_spec(_compiled(kind="O0", name="final_norm"))
    assert spec is not None
    assert spec.owner == "O0:elementwise:final_norm"
    assert spec.members == ()
    assert spec.api["entry"] == "aclnnAddRmsNorm"


def test_dynamic_prepare_keeps_real_r31_rank_geometry():
    compiled = _compiled()
    binding = _binding(compiled)
    result = adapter.dynamic_call(
        binding, compiled, _geometry(),
        validate=native_calls.validate_native_occurrence)
    assert result is not None
    assert result["actual"]["shapes"]["x"] == [2048, 5120]
    assert result["actual"]["shapes"]["rstd"] == [2048, 1]
    assert result["actual"]["attrs"]["rows"] == 2048
    assert result["actual"]["attrs"]["rstd_bytes"] == 8192
    prepare = result["prepare"]
    assert prepare["call"] == "aclnnAddRmsNorm"
    assert prepare["kernel"]["name"] == "AddRmsNorm"
    assert prepare["shapes"]["x"] == [2048, 5120]
    assert prepare["shapes"]["residual"] == [2048, 5120]
    assert prepare["descriptors"]["gamma"]["shape"] == [5120]
    assert prepare["descriptors"]["rstd"]["dtype"] == "fp32"
    assert prepare["output_count"] == 3
    assert prepare["runtime_executed"] is False
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert "aclnn_positional_slots_not_retained" in prepare["partial_reasons"]
    assert prepare["capture"]["name"] == "aclnnAddRmsNorm_AddRmsNorm_AddRmsNorm"
    assert prepare["capture"]["block_num"] == 64
    assert "captured_rows" not in prepare["attrs"]
    assert "captured_block_num" not in prepare["attrs"]
    assert prepare["attrs"]["physical_rows"] == 2048
    assert prepare["attrs"]["sp_rows"] == "shard"
    assert prepare["attrs"]["sp_ranks"] == 2
    assert prepare["deployment"]["image_id"] == adapter.R31_IMAGE_ID
    assert result["prepare_args"]["shapes"]["x"] == [2048, 5120]


def test_dynamic_decode_and_selected_rows():
    compiled = _compiled()
    binding = _binding(compiled)
    decode = adapter.dynamic_call(
        binding, compiled, _geometry(tokens=1, phase="decode"),
        validate=native_calls.validate_native_occurrence)
    assert decode["prepare"]["shapes"]["x"] == [1, 5120]
    selected = adapter.dynamic_call(
        binding, compiled,
        _geometry(tokens=4096, phase="prefill", selected=3),
        validate=native_calls.validate_native_occurrence)
    assert selected["prepare"]["shapes"]["x"] == [2048, 5120]


def test_selected_o0_uses_producing_rows():
    compiled = _compiled(kind="O0", name="final_norm",
                         sequence_rows="selected")
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    binding = _binding(compiled)
    result = adapter.dynamic_call(
        binding, compiled, _geometry(selected=3),
        validate=native_calls.validate_native_occurrence)
    assert result["prepare"]["shapes"]["x"] == [3, 5120]
    assert result["prepare"]["shapes"]["rstd"] == [3, 1]
    assert result["prepare"]["match"]["row_basis"] == "selected"


def test_width_comes_from_theta_not_fixed_default():
    compiled = _compiled(elements=4096, gamma_bytes=8192.0)
    spec = adapter.binding_spec(compiled)
    assert spec is not None
    assert spec.typed["attrs"]["hidden"] == 4096
    assert spec.typed["shapes"]["gamma"] == [4096]
    binding = _binding(compiled)
    result = adapter.dynamic_call(
        binding, compiled, _geometry(),
        validate=native_calls.validate_native_occurrence)
    assert result["prepare"]["shapes"]["x"] == [2048, 4096]
    assert result["prepare"]["attrs"]["physical_rows"] == 2048


def test_plain_initial_norm_is_not_this_call():
    assert adapter.binding_spec(_compiled(siblings=("rmsnorm",))) is None


@pytest.mark.parametrize("changes", [
    {"image_id": "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"},
    {"image": "r64"},
    {"elements": 6144, "gamma_bytes": 10240.0},
    {"gamma_format": "fp32"},
    {"norm_has_bias": True},
    {"output_format": "int8"},
    {"quantized": 1},
    {"layernorm": 1},
    {"norm_weight_offset": 1.0},
    {"norm_arch": "gemma"},
])
def test_incompatible_norm_sources_stay_unbound(changes):
    values = dict(changes)
    image_id = values.pop("image_id", adapter.R31_IMAGE_ID)
    image = values.pop("image", adapter.R31_IMAGE_TAG)
    compiled = _compiled(image_id=image_id, image=image, **values)
    assert adapter.binding_spec(compiled) is None


def test_o0_without_residual_fusion_stays_unbound():
    assert adapter.binding_spec(_compiled(kind="O0", name="final_norm",
                                          residual_add=0)) is None
    assert adapter.binding_spec(_compiled(kind="O0", name="final_norm",
                                          fusion="")) is None


def test_dynamic_rejects_foreign_binding_and_empty_geometry():
    compiled = _compiled()
    binding = _binding(compiled)
    assert adapter.dynamic_call(
        SimpleNamespace(api={"entry": "other"}, typed={"attrs": {}},
                        source={}, call_id="x"),
        compiled, _geometry(),
        validate=native_calls.validate_native_occurrence) is None
    assert adapter.dynamic_call(
        binding, compiled, _geometry(tokens=0),
        validate=native_calls.validate_native_occurrence) is None
