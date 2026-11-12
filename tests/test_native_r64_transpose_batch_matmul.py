"""Source-bound R64 aclnnTransposeBatchMatMul value-expansion adapter.

[Role] Exercise the operator adapter against the exact R64 SFA caller blob and
  the original GLM51 PF1kC1 captured geometry.

[Invariants] The call is
  ``torch_npu.npu_transpose_batchmatmul(x, W_UV, perm_x1=(1,0,2),
  perm_y=(1,0,2))`` with BF16 ``[B,N,L] x [N,L,V] -> [B,N,V]``.  The call
  identity is independent of B; B is an occurrence fact.  Other images, roles,
  dtypes, permutations and the separate ``o_projection``/sparse-attention
  calls stay unbound.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

from types import SimpleNamespace

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r64_transpose_batch_matmul import (
    ACLNN_API, CALL_EXPRESSION, ENTRY, KERNEL_NAME, PERM_X1, PERM_Y,
    PROFILER_NAME, R64_HOST_BINARY_SHA, R64_IMAGE_SHA, R64_SOURCE_SHA, ROLE,
    SOURCE, SOURCE_LINES, binding_spec, dynamic_call)
from mindie_sim.prediction.subgraphs.builders_attention import _r64_image
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

R35_IMAGE_SHA = (
    "5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
VIEW = {
    "kind": "transpose_batch_matmul",
    "batch": 64,
    "n": 256,
    "k": 512,
    "input_shape": (1024, 64, 512),
    "weight_shape": (64, 512, 256),
    "output_shape": (1024, 64, 256),
    "input_permutation": (1, 0, 2),
    "output_permutation": (1, 0, 2),
    "permutation_scope": "logical_view; physical_native_layout_not_selected",
    "role": "value_expansion",
}
BF16 = (("act_format", "bf16"), ("weight_format", "bf16"),
        ("output_format", "bf16"), ("quantized", 0))


def _extras(view=None, *, batch=64):
    row = VIEW if view is None else view
    pairs = tuple((key, tuple(value) if isinstance(value, (list, tuple))
                   else value) for key, value in row.items())
    return (("batch", batch), ("native_tensor_view", pairs))


def _compiled(*, image_id=R64_IMAGE_SHA, kind="A1", name="v_expand",
              family="gemm", view=None, extras=None, quant=BF16,
              charge_owner=None):
    template = OperatorTemplate(name=name, family=family, role=ROLE)
    rows = _extras(view) if extras is None else extras
    if charge_owner is not None:
        rows = rows + (("native_charge_owner", charge_owner),)
    signature = StaticSignature(
        role=ROLE, matrix=(("K", 512), ("N", 256)), quant=quant,
        extras=rows)
    operator = BoundOperator(template, signature)
    return SimpleNamespace(
        kind=kind, operator=operator, layer=0,
        feature_metadata=(("image_id", image_id),
                          ("image", "vllm-ascend:...r64")),
        program=SimpleNamespace(annotations=()))


def _binding(compiled):
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(tokens, phase="prefill"):
    return WorkloadGeometry((RequestGeometry(tokens, 0, phase=phase),),
                            phase=phase)


def test_r64_value_expansion_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] == ACLNN_API
    assert spec.source["sha256"] == R64_SOURCE_SHA
    assert spec.source["symbol"] == SOURCE["symbol"]
    assert spec.source["revision"] == SOURCE["revision"]
    assert spec.owner == "v_expand"
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["acl_api"] == ACLNN_API
    assert spec.kernels[0]["profiler_name"] == PROFILER_NAME
    assert spec.typed["shapes"] == {
        "input": [64, 512], "weight": [64, 512, 256],
        "output": [64, 256]}
    assert spec.typed["attrs"]["matrix_batch"] == 64
    assert spec.typed["attrs"]["matrix_k"] == 512
    assert spec.typed["attrs"]["matrix_n"] == 256
    assert spec.typed["attrs"]["rows"] == "runtime_tokens"
    assert spec.typed["attrs"]["source_lines"] == list(SOURCE_LINES)
    assert binding is not None
    assert binding.timing_owner == {
        "op_id": "A1:gemm:v_expand", "scope": "call"}
    assert "python_wrapper_signature_not_retained" in binding.unresolved

    first = dynamic_call(binding, compiled, _geometry(1024),
                         validate=validate_native_occurrence)
    assert first is not None
    assert first["actual"]["shapes"] == {
        "input": [1024, 64, 512],
        "weight": [64, 512, 256],
        "output": [1024, 64, 256],
    }
    prepare = first["prepare"]
    assert prepare["arguments"] == {
        "x1": "input", "x2": "weight",
        "perm_x1": list(PERM_X1), "perm_y": list(PERM_Y)}
    assert prepare["positional_arguments"] == {
        "x1": "input", "x2": "weight"}
    assert prepare["keywords"] == {
        "perm_x1": list(PERM_X1), "perm_y": list(PERM_Y)}
    assert prepare["optional_slots"]["perm_x2"] == "not_passed_source_default"
    assert prepare["prototype"] == CALL_EXPRESSION
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["match"]["tokens"] == 1024
    assert prepare["match"]["input_shape"] == [1024, 64, 512]
    assert prepare["match"]["output_shape"] == [1024, 64, 256]
    assert prepare["match"]["observed_original"]["count"] == 78
    assert prepare["acl_interface"]["acl_api"] == ACLNN_API
    assert (prepare["acl_interface"]["retained_host_tiling_binary"]
            ["sha256"]) == R64_HOST_BINARY_SHA
    assert prepare["binary_selection"]["status"] == "uncertain"
    assert prepare["descriptors"]["weight"]["role_name"] == "W_UV"
    assert prepare["descriptors"]["input"]["native_view_shape"] == [
        64, 1024, 512]
    assert prepare["descriptors"]["output"]["native_view_shape"] == [
        64, 1024, 256]

    held = dynamic_call(binding, compiled, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["input"] == [2048, 64, 512]
    assert held["prepare"]["match"]["tokens"] == 2048
    # A new B is a new occurrence, not a new selected call.
    assert binding.call_id == _binding(_compiled())[1].call_id

    decode = dynamic_call(binding, compiled, _geometry(1, phase="decode"),
                          validate=validate_native_occurrence)
    assert decode is not None
    assert decode["actual"]["shapes"]["output"] == [1, 64, 256]


def test_r64_value_expansion_rejects_other_sources_and_roles():
    compiled = _compiled()
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(kind="A0"), phase="prefill") is None
    assert binding_spec(_compiled(name="q_absorb"), phase="prefill") is None
    assert binding_spec(_compiled(name="mla_attention"),
                        phase="prefill") is None
    assert binding_spec(_compiled(family="mla"), phase="prefill") is None
    assert binding_spec(_compiled(view={**VIEW, "role": "key_absorption"}),
                        phase="prefill") is None
    assert binding_spec(_compiled(view={**VIEW, "kind": "matmul"}),
                        phase="prefill") is None
    assert binding_spec(
        _compiled(view={**VIEW, "input_permutation": (0, 1, 2)}),
        phase="prefill") is None
    assert binding_spec(
        _compiled(view={**VIEW, "batch": 32}), phase="prefill") is None
    assert binding_spec(
        _compiled(quant=(("act_format", "bf16"),
                         ("weight_format", "fp16"),
                         ("output_format", "bf16"), ("quantized", 0))),
        phase="prefill") is None
    assert binding_spec(
        _compiled(quant=(("act_format", "bf16"),
                         ("weight_format", "bf16"),
                         ("output_format", "bf16"), ("quantized", 1))),
        phase="prefill") is None
    assert binding_spec(
        _compiled(extras=(("batch", 64),)), phase="prefill") is None

    spec, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
    from dataclasses import replace
    wrong = replace(binding, api={
        **(binding.api or {}), "entry": "torch_npu.npu_bmm"})
    assert dynamic_call(wrong, compiled, _geometry(1024),
                        validate=validate_native_occurrence) is None


def test_value_expansion_can_declare_parent_attribution_owner():
    compiled = _compiled(charge_owner="A1:mla:mla_attention")
    spec, binding = _binding(compiled)
    assert spec.owner == "A1:mla:mla_attention"
    from dataclasses import replace
    attributed = replace(binding, timing_owner={
        "op_id": spec.owner, "scope": "call"})
    assert attributed.timing_owner == {
        "op_id": "A1:mla:mla_attention", "scope": "call"}
    envelope = dynamic_call(attributed, compiled, _geometry(1024),
                            validate=validate_native_occurrence)
    assert envelope is not None
    assert envelope["prepare"]["match"]["charge_owner"] == \
        "A1:mla:mla_attention"


def test_semantic_v_expand_is_r64_image_only():
    assert _r64_image({"image_id": R64_IMAGE_SHA,
                       "image": "vllm-ascend:...r64"})
    assert not _r64_image({"image_id": R35_IMAGE_SHA,
                           "image": "vllm-ascend:...r35"})
    assert not _r64_image({})
