"""Focused R31 GDN state-gather adapter contract tests.

The adapter is metadata-only: these tests prove the real shared
``native_calls._CallSpec``/``NativeCallBinding`` path, the exact source
identity, the Triton grid/argument/descriptor expansion and the negative
source/branch gates.  They do not allocate tensors, import torch or launch a
kernel.
"""
from dataclasses import replace
from types import SimpleNamespace

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise.geometry import (
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_gather_ssm_states import (
    CAPTURE_PATH, GDN_SHA, KERNEL_NAME, R31_IMAGE_SHA, STATE_INDEX_SHA,
    binding_spec, dynamic_call, source_selected)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

R31_TEXT = (
    "quay.io/ascend/vllm-ascend:v0.30.0-a5-cann9.2.0-r31 "
    f"(sha256:{R31_IMAGE_SHA})")
OTHER_TEXT = "quay.io/ascend/vllm-ascend:v0.30.0rc2-a5 (c3d00a196905)"


def _extras(**overrides):
    values = {
        "state_dtype": "fp32",
        "state_layout": "V,K",
        "heads": 8,
        "dv": 128,
        "dk": 128,
        "index_dtype": "int32",
        "source_kernel": KERNEL_NAME,
    }
    values.update(overrides)
    return tuple((key, value) for key, value in values.items()
                 if value is not None)


def _compiled(*, extras=None, image=R31_TEXT, family="state_gather",
              kind="A3", variant="chunked_prefill",
              name="gather_recurrent_state", row_bytes=8 * 128 * 128 * 4):
    extras = _extras() if extras is None else tuple(extras)
    theta = StaticSignature(
        "gather",
        matrix=(("rows", 1), ("row_bytes", row_bytes)),
        extras=extras)
    template = OperatorTemplate(name=name, family=family, role="gather")
    return SimpleNamespace(
        kind=kind, variant=variant, operator=BoundOperator(template, theta),
        feature_metadata=(("source_image", image),), siblings=())


def _binding(compiled=None):
    compiled = compiled or _compiled()
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = native_calls._Surface(
        op_id="A3:state_gather:gather_recurrent_state",
        family="state_gather", recipe="gather", phase="prefill",
        source_boundary="kernel", dtype="fp32", layout="ND")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(*contexts, phase="prefill"):
    requests = tuple(
        RequestGeometry(q_len=64, context=context, phase=phase)
        for context in contexts)
    return WorkloadGeometry(requests=requests, phase=phase)


def test_binding_spec_is_the_real_r31_triton_call_identity():
    spec, binding = _binding()
    assert type(spec).__name__ == "_CallSpec"
    assert spec.api == {
        "namespace": "triton", "entry": KERNEL_NAME,
        "aclnn_symbol": None, "version": "r31/9e869e8148"}
    assert spec.source["path"] == (
        "vllm_ascend/ops/triton/mamba/state_index.py")
    assert spec.source["sha256"] == STATE_INDEX_SHA
    assert spec.source["symbol"] == "gather_ssm_states"
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["source_sha256"] == STATE_INDEX_SHA
    assert spec.kernels[0]["lines"] == [14, 43]
    assert spec.typed["dtype"] == "fp32"
    assert spec.typed["shapes"] == {
        "state": [8, 128, 128], "output": [8, 128, 128]}
    assert spec.typed["attrs"]["state_row_bytes"] == 8 * 128 * 128 * 4
    assert spec.typed["attrs"]["source_commit"] == (
        "8fd0bc4885cb52e58791aaa8efacc571b9bd115d")
    assert binding.call_id
    assert binding.unresolved == ("aclnn_symbol_unresolved",)


def test_dynamic_call_expands_exact_triton_grid_arguments_and_descriptors():
    compiled = _compiled()
    _, binding = _binding(compiled)
    env = dynamic_call(
        binding, compiled, _geometry(0, 512),
        validate=validate_native_occurrence)
    assert env is not None
    actual = env["actual"]
    assert actual["dtype"] == "fp32" and actual["layout"] == "ND"
    assert actual["shapes"] == {
        "state": [None, 8, 128, 128],
        "indices": [2],
        "has_initial_state": [2],
        "output": [2, 8, 128, 128],
    }
    attrs = actual["attrs"]
    assert attrs["batch"] == 2
    assert attrs["has_initial_state"] == [False, True]
    assert attrs["selected_rows"] == 1
    assert attrs["state_cache_rows"] is None
    assert attrs["grid"] == [128, 2]
    prepare = env["prepare"]
    assert prepare["grid"] == [128, 2]
    assert prepare["block_size"] == 1024
    assert prepare["num_warps"] == 8
    assert prepare["arguments"] == {
        "state": "state", "indices": "indices",
        "has_initial_state": "has_initial_state", "output": "output",
        "stride_state_batch": 131072, "stride_indices": 1,
        "stride_has_initial_state": 1, "row_size": 131072,
        "BLOCK_SIZE": 1024, "num_warps": 8,
    }
    descriptors = prepare["descriptors"]
    assert descriptors["state"]["shape"] == [None, 8, 128, 128]
    assert descriptors["state"]["stride0"] == 131072
    assert descriptors["indices"] == {
        "shape": [2], "dtype": "int32", "layout": "ND", "role": "input",
        "stride0": 1, "pointer": None, "allocation": "external"}
    assert descriptors["has_initial_state"]["dtype"] == "bool"
    assert descriptors["output"]["shape"] == [2, 8, 128, 128]
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["partial_reasons"] == [
        "state_cache_rows_runtime_unresolved",
        "state_stride0_padding_unobserved",
        "index_stride_runtime_unobserved",
        "has_initial_state_stride_runtime_unobserved",
    ]
    assert prepare["match"]["branch"] == "fla_gdn_prefill"
    assert prepare["match"]["selected_rows"] == 1


def test_declared_cache_rows_and_stride_complete_the_prepare():
    compiled = _compiled(extras=_extras(
        state_cache_rows=96, state_stride0=131080,
        stride_indices=1, stride_has_initial_state=1))
    _, binding = _binding(compiled)
    env = dynamic_call(
        binding, compiled, _geometry(7),
        validate=validate_native_occurrence)
    assert env is not None
    prepare = env["prepare"]
    assert env["actual"]["shapes"]["state"] == [96, 8, 128, 128]
    assert prepare["arguments"]["stride_state_batch"] == 131080
    assert prepare["descriptors"]["state"]["stride0"] == 131080
    assert prepare["complete"] is True
    assert prepare["partial"] is False
    assert prepare["partial_reasons"] == []
    assert env["actual"]["attrs"]["has_initial_state"] == [True]


def test_source_selected_is_the_builder_gate():
    assert source_selected((("source_image", R31_TEXT),)) is True
    assert source_selected((("deployment_source_revision", "r31/9e869e8148"),)) is True
    assert source_selected((("source_image", OTHER_TEXT),)) is False
    assert source_selected(()) is False


def test_r31_gate_rejects_other_images_and_other_branches():
    assert binding_spec(_compiled(family="copy"), phase="prefill") is not None
    assert binding_spec(_compiled(family="recurrent"),
                        phase="prefill") is None
    assert binding_spec(_compiled(image=OTHER_TEXT), phase="prefill") is None
    assert binding_spec(_compiled(image=""), phase="prefill") is None
    assert binding_spec(_compiled(family="state_scatter"),
                        phase="prefill") is None
    assert binding_spec(_compiled(kind="A4"), phase="prefill") is None
    assert binding_spec(_compiled(), phase="decode") is None
    assert binding_spec(_compiled(variant="recurrent_decode"),
                        phase="decode") is None
    assert binding_spec(_compiled(extras=_extras(index_dtype=None)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(state_dtype=None)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(output_dtype="bf16")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(source_kernel="other")),
                        phase="prefill") is None
    assert binding_spec(_compiled(row_bytes=7), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(state_cache_rows=96.5)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(state_stride0=131072.5)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(heads=8.5)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(stride_indices=1.5)),
                        phase="prefill") is None


def test_dynamic_call_rejects_non_gather_binding_and_decode_geometry():
    compiled = _compiled()
    _, binding = _binding(compiled)
    assert dynamic_call(
        replace(binding, api={"entry": "not_the_gather"}),
        compiled, _geometry(0), validate=validate_native_occurrence) is None
    wrong_branch = replace(
        binding, typed={
            **binding.typed,
            "attrs": {**binding.typed["attrs"], "call_branch": "other"}})
    assert dynamic_call(
        wrong_branch, compiled, _geometry(0),
        validate=validate_native_occurrence) is None
    assert dynamic_call(
        binding, compiled, _geometry(0, phase="decode"),
        validate=validate_native_occurrence) is None


def test_observed_capture_never_becomes_a_state_cache_dimension():
    compiled = _compiled()
    _, binding = _binding(compiled)
    env = dynamic_call(
        binding, compiled, _geometry(0), validate=validate_native_occurrence)
    assert env is not None
    prepare = env["prepare"]
    assert prepare["observed_capture"]["path"] == CAPTURE_PATH
    assert prepare["observed_capture"]["input_shapes"].startswith("62438,")
    assert prepare["observed_capture"]["relation"].startswith(
        "retained_observed_sample_post_prediction")
    assert env["actual"]["shapes"]["state"][0] is None
    assert "62438" not in str(env["prepare_args"])
    assert env["prepare"]["match"]["caller_sha256"] == GDN_SHA
