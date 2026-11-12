"""Focused live-lowering invariants for the R64 AddRmsNormBias adapter.

[Role] Exercise the leaf adapter against the genuine ``family_ascend_norm``
  lowering at two widths, with declared/absent/unresolved optional bias.

[Invariants] Width, epsilon, weight and residual are live lowering fields.  Beta
  is present only when ``norm_has_bias`` is true; a selected source with an
  unknown bias declaration keeps the scaffold with no beta tensor.  The source
  is not selected by model SHA, image tag, capture sample or fixed hidden size.

[Timeouts & Locks] Pure bounded metadata construction; no I/O or locks.

[Side-Effect Policy] Test-local objects only; no device, product mutation or
  mutable global cache dependence.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from mindie_sim.catalog.ops import vector_op
from mindie_sim.prediction.family_ascend_norm import lower_op
from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise.native_calls import (
    _Surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r64_add_rms_norm_bias import (
    ENTRY, binding_spec, dynamic_call)
from mindie_sim.schema.subgraphs import StaticSignature
from mindie_sim.structure_native_contracts import add_rmsnorm_bias_contract

KEEP = (
    "gamma_bytes", "gamma_format", "gamma_role", "norm_arch", "norm_eps",
    "norm_weight_offset", "layernorm", "native_norm_kind", "norm_has_bias",
    "norm_has_weight", "norm_bias_value", "norm_bias_trained",
    "norm_bias_origin", "norm_bias_selector", "norm_source_package_sha256",
    "norm_source_weights_sha256", "native_norm_source", "sequence_rows",
    "sequence_ranks", "returns",
)
BETA_EVIDENCE = (
    ("norm_bias_value", "zero"),
    ("norm_bias_trained", False),
    ("norm_bias_origin", "ascend_rmsnorm_zero_parameter"),
    ("norm_bias_selector", "any_norm_bias_in_quant_description"),
    ("norm_source_package_sha256", "436570c2"),
    ("norm_source_weights_sha256", "7434850d"),
    ("native_norm_source", "AscendRMSNorm.forward_oot residual"),
    ("sequence_rows", ""),
    ("sequence_ranks", 1),
)


def _theta(lowered, hidden: int, *, native_kind="AddRmsNormBias",
           drop_bias=False, drop_value=False):
    extras = {}
    for key, value in lowered.metadata:
        if key in KEEP:
            extras[key] = value
    extras["elements"] = hidden
    extras["returns"] = tuple(lowered.returns)
    if native_kind is None:
        extras.pop("native_norm_kind", None)
    elif native_kind != "AddRmsNormBias":
        extras["native_norm_kind"] = native_kind
    if drop_bias:
        extras.pop("norm_has_bias", None)
    if drop_value:
        extras.pop("norm_bias_value", None)
    # StaticSignature requires immutable scalar/tuple values.
    clean = {}
    for key, value in extras.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            clean[key] = value
        elif isinstance(value, tuple) and all(
                isinstance(item, (str, int, float, bool)) or item is None
                for item in value):
            clean[key] = value
    return StaticSignature("norm", matrix=(("elements", hidden),),
                           quant=(("act_bytes", 1.0),),
                           extras=tuple(clean.items()))


def _compiled(hidden: int, rows: int, has_bias: bool, *,
              native_kind="AddRmsNormBias", drop_bias=False,
              drop_value=False, residual=True):
    op = vector_op(
        op_id="n0.rmsnorm", deps=(), stage=0, layer=0, requests=("r0",),
        elements=rows * hidden, ops_per_element=5, dtype="bf16",
        family="add_rmsnorm", phase="prefill", recipe="add_rmsnorm",
        shape=(rows, hidden),
        read_bytes_override=rows * hidden * 2,
        write_bytes_override=rows * hidden * 2,
        metadata=(
            ("norm_has_weight", True), ("norm_eps", 1e-5),
            ("gamma_bytes", hidden * 2.0), ("gamma_format", "bf16"),
            ("gamma_role", "ffn_norm"), ("norm_arch", "rmsnorm"),
            ("norm_weight_offset", 0.0), ("layernorm", 0),
            ("residual_add", 1),
        ) + BETA_EVIDENCE,
        returns=("norm_output", "residual") if residual else ("norm_output",),
    )
    lowered = lower_op(op, has_bias=has_bias)[-1]
    theta = _theta(lowered, hidden, native_kind=native_kind,
                   drop_bias=drop_bias, drop_value=drop_value)
    operator = SimpleNamespace(
        name="rmsnorm", family=lowered.family, theta=theta,
        local_shape=lowered.local_shape, dtype=lowered.dtype,
        returns=lowered.returns)
    siblings = ((SimpleNamespace(name="residual_add"),) if residual else ())
    return SimpleNamespace(kind="N0", layer=0, operator=operator,
                           siblings=siblings), lowered


def _binding(compiled, rows: int, *, final_rows=None,
             final_rows_source="workload_preselection_local_activation_rows"):
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = _Surface(
        op_id="N0:elementwise:rmsnorm", family=compiled.operator.family,
        recipe="add_rmsnorm", phase="prefill", source_boundary="kernel",
        dtype=spec.typed["dtype"], layout="ND",
        metadata=tuple(sorted(spec.typed["attrs"].items())))
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    binding = bind_native_call(declared)
    assert binding is not None
    if final_rows is not None:
        attr_map = dict(binding.typed.get("attrs") or {})
        attr_map.update({
            "final_rows": int(final_rows),
            "final_rows_source": final_rows_source,
            "selected_lm_head_rows": 1,
            "rank_local_shard_rows": rows,
        })
        binding.typed["attrs"] = attr_map
    geometry = SimpleNamespace(
        effective_tokens=rows, phase_name="prefill",
        selected_tokens=1, selected_row_count=1,
        physical_tokens=rows, operand_tokens=rows)
    dynamic = dynamic_call(binding, compiled, geometry,
                           validate=validate_native_occurrence)
    assert dynamic is not None
    return spec, binding, dynamic


@pytest.mark.parametrize("hidden,rows", [(5120, 64), (6144, 1024)])
def test_live_width_and_zero_beta(hidden: int, rows: int):
    compiled, lowered = _compiled(hidden, rows, True)
    spec, _binding_value, dynamic = _binding(compiled, rows)
    assert spec.typed["shapes"]["x1"] == [hidden]
    assert spec.typed["attrs"]["beta_state"] == "present"
    assert spec.typed["attrs"]["beta_value"] == "zero"
    assert spec.typed["attrs"]["beta_trained"] is False
    assert dynamic["actual"]["shapes"]["x1"] == [rows, hidden]
    assert dynamic["actual"]["shapes"]["beta"] == [hidden]
    prepare = dynamic["prepare"]
    assert prepare["match"]["hidden"] == hidden
    assert prepare["match"]["rows"] == rows
    assert prepare["match"]["beta_value"] == "zero"
    assert prepare["match"]["beta_trained"] is False
    assert prepare["beta"]["tensor"]["shape"] == [hidden]
    assert prepare["beta"]["value"] == "zero"
    assert prepare["complete"] is False
    contract = add_rmsnorm_bias_contract({
        "metadata": dict(lowered.metadata),
        "local_shape": lowered.local_shape,
        "dtype": lowered.dtype,
    })
    assert contract is not None
    assert contract[2][3] == ((hidden,), "bf16", "ND")


def test_declared_absent_bias_has_no_beta_tensor():
    compiled, _lowered = _compiled(5120, 64, False)
    spec, _binding_value, dynamic = _binding(compiled, 64)
    assert spec.api["entry"] == ENTRY
    assert spec.typed["attrs"]["beta_state"] == "absent"
    assert "beta" not in spec.typed["shapes"]
    prepare = dynamic["prepare"]
    assert "beta" not in dynamic["actual"]["shapes"]
    assert prepare["descriptors"]["beta"]["present"] is False
    assert prepare["descriptors"]["beta"]["resolution"] == "declared_absent"
    assert prepare["match"]["input_count"] == 3


def test_selected_source_with_unknown_bias_keeps_unresolved_scaffold():
    compiled, _lowered = _compiled(5120, 64, True, drop_bias=True)
    spec, _binding_value, dynamic = _binding(compiled, 64)
    assert spec.typed["attrs"]["beta_state"] == "unresolved"
    assert "beta" not in spec.typed["shapes"]
    assert "norm_bias_declaration_unresolved" in spec.unresolved
    prepare = dynamic["prepare"]
    assert "beta" not in dynamic["actual"]["shapes"]
    assert prepare["descriptors"]["beta"]["present"] is None
    assert (prepare["descriptors"]["beta"]["resolution"]
            == "not_declared_by_live_lowering")
    assert prepare["optional_slots"]["beta"] == "unresolved_not_declared"
    assert "norm_bias_declaration_unresolved" in prepare["partial_reasons"]
    assert prepare["complete"] is False


def test_present_bias_without_value_is_explicitly_unresolved():
    compiled, _lowered = _compiled(6144, 64, True, drop_value=True)
    spec, _binding_value, dynamic = _binding(compiled, 64)
    assert spec.typed["attrs"]["beta_state"] == "present"
    assert "beta_value" not in spec.typed["attrs"]
    assert "beta_value_unresolved" in spec.unresolved
    prepare = dynamic["prepare"]
    assert prepare["beta"]["value"] is None
    assert "beta_value_unresolved" in prepare["partial_reasons"]
    assert prepare["complete"] is False



def _o0_compiled(*, returns=("x", "residual"), native_kind="AddRmsNormBias",
                 weight_offset=0.0, arch="rmsnorm", has_bias=True,
                 beta_state="present_zero", beta_value="zero",
                 rstd_bytes=4.0, residual_out=12288.0,
                 preselection_rows=None):
    extras = [
        ("elements", 6144),
        ("native_norm_kind", native_kind),
        ("native_norm_source", "AscendRMSNorm.forward_oot final residual branch"),
        ("norm_has_weight", True),
        ("norm_weight_offset", weight_offset),
        ("norm_arch", arch),
        ("layernorm", 0),
        ("norm_eps", 1e-5),
        ("gamma_role", "final"),
        ("gamma_format", "bf16"),
        ("gamma_bytes", 12288.0),
        ("residual_add", 1),
        ("fusion", "add_rmsnorm"),
        ("returns", returns),
        ("rstd_bytes", rstd_bytes),
        ("residual_out_bytes", residual_out),
        ("norm_bias_state", beta_state),
        ("norm_bias_value", beta_value),
        ("norm_bias_trained", False),
        ("norm_bias_origin", "AscendRMSNorm final residual constructor/role source"),
        ("norm_bias_selector", "final residual-present forward_oot"),
        ("norm_source_weights_sha256", "7434850d"),
        ("norm_source_package_sha256", ""),
    ]
    if preselection_rows is not None:
        extras.append(("preselection_rows", int(preselection_rows)))
    if has_bias is True:
        extras.append(("norm_has_bias", True))
    elif has_bias is False:
        extras.append(("norm_has_bias", False))
    theta = StaticSignature("norm", matrix=(("elements", 6144),),
                            quant=(("act_bytes", 1.0),),
                            extras=tuple(extras))
    operator = SimpleNamespace(
        name="final_norm", family="elementwise", theta=theta,
        local_shape=(1, 6144), dtype="bf16", returns=returns)
    return SimpleNamespace(kind="O0", layer=0, operator=operator, siblings=())


def test_o0_final_norm_uses_preselection_geometry_not_rstd():
    compiled = _o0_compiled()
    spec, _binding_value, dynamic = _binding(compiled, 1024)
    assert spec.owner == "O0:elementwise:final_norm"
    assert spec.members == ()
    assert spec.typed["attrs"]["native_norm_scope"] == "o0_final"
    assert spec.typed["shapes"]["x1"] == [6144]
    assert spec.typed["shapes"]["beta"] == [6144]
    actual = dynamic["actual"]
    assert actual["shapes"]["x1"] == [1024, 6144]
    assert actual["shapes"]["rstd"] == [1024, 1]
    assert actual["shapes"]["beta"] == [6144]
    prepare = dynamic["prepare"]
    assert prepare["match"]["native_norm_scope"] == "o0_final"
    assert prepare["match"]["rows"] == 1024
    assert prepare["match"]["row_basis"] == "geometry_effective_tokens"
    assert prepare["match"]["rows_source"] == "geometry_effective_tokens"
    assert prepare["match"]["input_count"] == 4
    assert prepare["fusion"]["members"] == []
    assert prepare["fusion"]["contract"] == "add_rmsnorm"
    assert prepare["pricing"]["charged_owner"] == "O0:elementwise:final_norm"
    assert prepare["beta"]["value"] == "zero"
    assert prepare["beta"]["trained"] is False
    assert prepare["complete"] is False
    # Static rstd/residual bytes are per-row and must not force one row.
    assert prepare["attrs"]["rstd_bytes"] == 1024 * 4
    assert 1024 != 4


def test_o0_final_norm_geometry_drives_rows_not_hardcoded():
    compiled = _o0_compiled()
    _spec, _binding_value, dynamic = _binding(compiled, 7)
    assert dynamic["actual"]["shapes"]["x1"] == [7, 6144]
    assert dynamic["prepare"]["match"]["rows"] == 7
    assert dynamic["prepare"]["match"]["row_basis"] == "geometry_effective_tokens"


def test_o0_final_norm_source_builder_preselection_wins():
    compiled = _o0_compiled(preselection_rows=13)
    _spec, _binding_value, dynamic = _binding(compiled, 1024)
    assert dynamic["prepare"]["match"]["rows"] == 13
    assert (dynamic["prepare"]["match"]["row_basis"]
            == "source_builder_preselection_rows")


def test_o0_final_norm_uses_shared_bind25q_final_rows():
    compiled = _o0_compiled()
    _spec, _binding_value, dynamic = _binding(compiled, 1024, final_rows=13)
    prepare = dynamic["prepare"]
    assert dynamic["actual"]["shapes"]["x1"] == [13, 6144]
    assert prepare["match"]["rows"] == 13
    assert (prepare["match"]["row_basis"]
            == "workload_preselection_local_activation_rows")
    assert prepare["match"]["final_rows_source"] == (
        "workload_preselection_local_activation_rows")
    assert prepare["match"]["selected_lm_head_rows"] == 1


def test_o0_final_norm_requires_residual_return():
    assert binding_spec(_o0_compiled(returns=("x",)), phase="prefill") is None


def test_o0_final_norm_requires_true_class_and_offset():
    assert binding_spec(_o0_compiled(native_kind="AddRmsNorm"),
                        phase="prefill") is None
    assert binding_spec(_o0_compiled(weight_offset=1.0),
                        phase="prefill") is None
    assert binding_spec(_o0_compiled(arch="gemma"),
                        phase="prefill") is None


def test_o0_final_norm_absent_beta_has_no_tensor():
    compiled = _o0_compiled(has_bias=False, beta_state="absent",
                            beta_value="")
    spec, _binding_value, dynamic = _binding(compiled, 1024)
    assert spec.api["entry"] == ENTRY
    assert spec.owner == "O0:elementwise:final_norm"
    assert spec.typed["attrs"]["beta_state"] == "absent"
    assert "beta" not in spec.typed["shapes"]
    assert "beta" not in dynamic["actual"]["shapes"]
    assert dynamic["prepare"]["descriptors"]["beta"]["present"] is False
    assert dynamic["prepare"]["match"]["input_count"] == 3


@pytest.mark.parametrize("kwargs", [
    {"native_kind": None},
    {"native_kind": "AddRmsNormDynamicMxQuant"},
    {"residual": False},
])
def test_other_or_unselected_norm_is_not_attached(kwargs):
    compiled, _lowered = _compiled(5120, 64, True, **kwargs)
    assert binding_spec(compiled, phase="prefill") is None
