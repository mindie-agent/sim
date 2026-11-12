"""Focused checks for the R31 QKV RMSNorm+MRoPE observed producer.

[Role] Establish the true MRoPE admission rule, the exact retained A5 Python
  dispatch chain, the observed input/Q geometry, and the explicit
  Base/ACLNN/kernel/unknown-output candidates; no hardware, no measured
  duration, no whole forecast.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from mindie_sim import model_catalog
from mindie_sim.prediction.piecewise import native_calls
from mindie_sim.prediction.piecewise import (
    native_split_qkv_rmsnorm_mrope as qkv)
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_layer)
from mindie_sim.schema.common import ModelingError

IMAGE = ("sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03"
         "d4e824a233c2ffea")
META = (
    ("image_id", IMAGE),
    ("image", "vllm-ascend-nightly-a5-cann9.2.0-r31"),
    ("mrope_section", (11, 11, 10)),
    ("mrope_interleaved", True),
    ("partial_rotary_factor", 0.25),
    ("norm_eps", 1e-6),
)
FAMILY_ONLY = (("image_id", IMAGE), ("model_identifier", "qwen3_5"))


@pytest.fixture(scope="module")
def a0_instance():
    model = model_catalog.load_model("qwen3.6-27b")
    layer = next(plan for plan in model.layer_plans
                 if plan.attention == "gqa")
    features = FeatureConfig.from_model(
        model, tp=2, dp=4, ep=1, pp=1, cp=1, pcp=1, metadata=META)
    instances = instantiate_layer(model, layer, features)
    return next(item for item in instances.subgraphs if item.kind == "A0")


def _compiled(a0, metadata=META):
    return SimpleNamespace(
        kind="A0", layer=a0.layer, operator=a0.operator("rotary"),
        siblings=a0.operators, instance=a0, feature_metadata=metadata,
        supply=SimpleNamespace(resource_units=(("vector", 64),)),
    )


def _bind(compiled, spec):
    return native_calls.bind_native_call(native_calls.declare_native_call(
        native_calls._Surface(
            op_id="A0:elementwise:rotary", family="elementwise",
            recipe="rope", phase="", source_boundary="call",
            dtype=spec.typed["dtype"], layout=spec.typed["layout"],
            metadata=tuple(sorted(spec.typed["attrs"].items()))),
        api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members))


def test_dispatch_owner_and_observed_geometry(a0_instance, monkeypatch):
    compiled = _compiled(a0_instance)
    spec = qkv.binding_spec(compiled)
    assert spec is not None
    # Observed physical identity, not a Base/ACLNN assertion.
    assert spec.api["namespace"] == "cann_kernel"
    assert spec.api["entry"] == qkv.KERNEL_NAME
    assert spec.api["aclnn_symbol"] is None
    assert spec.api["dispatch_owner"] == "A5DeviceAdaptor"
    assert spec.api["dispatch_status"] == (
        "partial_python_acl_mapping_unresolved")
    candidates = spec.api["python_api_candidates"]
    assert candidates[0]["entry"] == qkv.A5_ENTRY
    assert candidates[0]["status"] == "exact_retained_a5_python_method"
    assert candidates[0]["kernel_name_match"] is False
    assert candidates[1]["entry"] == qkv.BASE_ENTRY
    assert candidates[1]["status"] == "candidate_not_proven"
    # Exact retained A5 Python method; kernel mapping stays partial.
    assert spec.source["symbol"] == "A5DeviceAdaptor.split_qkv_rmsnorm_rope"
    assert spec.source["sha256"] == qkv.DEVICE_OP_SHA
    assert spec.source["lines"] == [1950, 1963]
    assert spec.source["entry"] == qkv.A5_ENTRY
    assert spec.kernels[0]["name"] == qkv.KERNEL_NAME
    assert spec.kernels[0]["evidence"] == "runtime_observed"
    assert spec.kernels[0]["python_api"] == qkv.A5_ENTRY
    assert spec.kernels[0]["python_api_kernel_match"] is False
    assert (spec.kernels[0]["simt_kernel_name_observed_elsewhere"]
            == qkv.SIMT_KERNEL_NAME)
    assert spec.members == ("qk_norm", "rotary")
    assert spec.owner == "fused:qkv_norm_rope"
    assert spec.execution_core == "aiv"
    assert spec.typed["shapes"] == {
        "input": [7168], "q_weight": [256], "k_weight": [256], "q": [3072],
    }
    attrs = spec.typed["attrs"]
    assert (attrs["q_hidden_size"], attrs["kv_hidden_size"],
            attrs["head_dim"]) == (6144, 512, 256)
    assert attrs["mrope_section"] == [11, 11, 10]
    assert attrs["mrope_interleaved"] is True
    assert attrs["rope_dim"] == 64
    assert attrs["eps"] == pytest.approx(1e-6)
    assert attrs["dispatch_owner"] == "A5DeviceAdaptor"
    assert attrs["python_api"] == qkv.A5_ENTRY
    assert attrs["base_candidate_status"] == "candidate_not_proven"
    assert attrs["aclnn_status"] == "candidate_not_proven"
    assert attrs["source_exact_for_r31"] is False
    assert attrs["a5_python_method_exact"] is True
    assert attrs["base_python_method_exact"] is False
    assert attrs["old_interface_source_exact"] is False
    assert attrs["mrope_source_sha256"] == (
        "5fee0bd595394c38d1111ac3e254372d9e25d7871fa79389d6b69adb33cb5adf")
    assert attrs["source_config_sha256"] == (
        "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c")
    orig = native_calls._spec_for
    monkeypatch.setattr(
        native_calls, "_spec_for",
        lambda item, *, phase="": qkv.binding_spec(item, phase=phase)
        or orig(item, phase=phase))
    binding = native_calls.bind_compiled_operator(compiled, phase="prefill")
    assert binding is not None
    assert binding.timing_owner == {
        "op_id": "fused:qkv_norm_rope", "scope": "call"}
    assert binding.member_op_ids == ("qk_norm", "rotary")
    prefill = qkv.dynamic_call(
        binding, compiled,
        SimpleNamespace(effective_tokens=4096, phase_name="prefill"),
        validate=native_calls.validate_native_occurrence)
    assert prefill is not None
    shapes = prefill["actual"]["shapes"]
    assert shapes["input"] == [4096, 7168]
    assert shapes["q_weight"] == [256]
    assert shapes["k_weight"] == [256]
    assert shapes["cos_sin_cache"] == [3, 4096, 64]
    assert shapes["q"] == [4096, 3072]
    assert shapes["k_candidate"] == [4096, 512]
    assert shapes["v_candidate"] == [4096, 512]
    assert shapes["gate_candidate"] == [4096, 3072]
    assert shapes["positions_candidate"] == [3, 4096]
    prepare = prefill["prepare"]
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["source_exact_for_r31"] is False
    assert prepare["a5_python_method_exact"] is True
    assert prepare["base_python_method_exact"] is False
    assert prepare["python_kernel_mapping_exact"] is False
    assert prepare["aclnn_mapping_exact"] is False
    assert prepare["source_exact"] == {
        "a5_python_method": True,
        "base_python_method": False,
        "python_kernel_mapping": False,
        "aclnn_mapping": False,
        "kernel_source": False,
        "kernel_binary": False,
        "k_v_gate_outputs": False,
        "q_weight_transform": False,
        "positions_semantics": False,
        "old_interface": False,
    }
    assert prepare["outputs"]["observed_reference_outputs"] == ["q"]
    assert prepare["outputs"]["observed_reference_shapes"]["q"] == [4096, 3072]
    assert prepare["outputs"]["model_projected_shapes"]["q"] == [4096, 3072]
    assert prepare["outputs"]["candidate_unverified"] == ["k", "v", "gate"]
    assert prepare["outputs"]["current_event_q_basis"] == (
        "model_projected_local_geometry")
    assert prepare["descriptors"]["q"]["role"] == "output_model_projected"
    assert prepare["descriptors"]["q"]["observed_reference_shape"] == [
        4096, 3072]
    assert prepare["outputs"]["observed_reference_basis"].startswith(
        "original R31 PF4kC1 capture record 179")
    assert prepare["mrope_source"]["sha256"] == (
        "5fee0bd595394c38d1111ac3e254372d9e25d7871fa79389d6b69adb33cb5adf")
    assert prepare["launch"]["launched_blocks"] is None
    assert prepare["launch"]["status"] == (
        "unverified_no_retained_tiler_or_active_launch_evidence")
    assert prepare["launch"]["observed_reference"]["block_num"] == 64
    assert prepare["launch"]["observed_reference"]["tokens"] == 4096
    assert prepare["launch"]["observed_reference"]["record"] == 179
    assert "python_dispatch_mapping_partial" in prepare["partial_reasons"]
    assert "a5_python_api_kernel_name_mismatch" in prepare["partial_reasons"]
    assert "base_python_candidate_not_proven" in prepare["partial_reasons"]
    assert "aclnn_mapping_not_proven" in prepare["partial_reasons"]
    assert prepare["observed"]["record"] == 179
    dispatch = prepare["dispatch"]
    assert dispatch["status"] == "partial"
    assert dispatch["owner"] == "A5DeviceAdaptor"
    chain = dispatch["exact_retained_chain"]
    assert chain["hardware_profile"]["sha256"] == qkv.HARDWARE_PROFILE_SHA
    assert chain["hardware_profile"]["lines"] == [333, 338]
    assert "FP8_OPTIMIZED" in chain["hardware_profile"]["fact"]
    assert chain["device_operator"]["lines"] == [2067, 2076]
    assert "A5DeviceAdaptor" in chain["device_operator"]["fact"]
    assert chain["python_method"]["lines"] == [1950, 1963]
    assert chain["python_method"]["entry"] == qkv.A5_ENTRY
    assert dispatch["base_candidate"]["status"] == "candidate_not_proven"
    assert dispatch["aclnn_candidate"]["status"] == "candidate_not_proven"
    assert dispatch["phase"].startswith("static device-adaptor")
    decode = qkv.dynamic_call(
        binding, compiled,
        SimpleNamespace(effective_tokens=1, phase_name="decode"),
        validate=native_calls.validate_native_occurrence)
    assert decode is not None
    assert decode["actual"]["shapes"]["cos_sin_cache"] == [3, 1, 64]
    assert decode["actual"]["shapes"]["q"] == [1, 3072]
    assert decode["prepare"]["launch"]["launched_blocks"] is None
    assert decode["prepare"]["outputs"]["observed_reference_shapes"]["q"] == [
        4096, 3072]
    assert decode["prepare"]["outputs"]["model_projected_shapes"]["q"] == [
        1, 3072]


def test_base_is_not_inferred_from_non_simt_kernel(a0_instance):
    spec = qkv.binding_spec(_compiled(a0_instance))
    assert spec is not None
    assert spec.api["entry"] != qkv.BASE_ENTRY
    assert spec.source["symbol"] != "BaseDeviceAdaptor.split_qkv_rmsnorm_rope"
    assert spec.api["dispatch_status"] == (
        "partial_python_acl_mapping_unresolved")
    assert spec.api["python_api_candidates"][1]["status"] == (
        "candidate_not_proven")
    assert ("not on that chain" in
            spec.api["python_api_candidates"][1]["why_not_selected"])


def test_family_string_alone_and_plain_rope_are_unbound(a0_instance):
    # The real R31 feature metadata has image_id + model_identifier but no
    # true MRoPE declaration until FEATURES normalization is applied.
    assert qkv.binding_spec(_compiled(a0_instance, FAMILY_ONLY)) is None
    plain = (("image_id", IMAGE),
             ("rope_parameters", {"rope_type": "default",
                                  "rope_theta": 10000000.0}))
    assert qkv.binding_spec(_compiled(a0_instance, plain)) is None
    section_only = (("image_id", IMAGE), ("mrope_section", (11, 11, 10)))
    assert qkv.binding_spec(_compiled(a0_instance, section_only)) is None
    interleave_only = (("image_id", IMAGE), ("mrope_interleaved", True))
    assert qkv.binding_spec(_compiled(a0_instance, interleave_only)) is None


def test_true_mrope_without_width_keeps_unverified_facts(a0_instance):
    metadata = (("image_id", IMAGE),
                ("mrope_section", (11, 11, 10)),
                ("mrope_interleaved", True))
    compiled = _compiled(a0_instance, metadata)
    spec = qkv.binding_spec(compiled)
    assert spec is not None
    assert "rope_dim" not in spec.typed["attrs"]
    binding = _bind(compiled, spec)
    env = qkv.dynamic_call(
        binding, compiled,
        SimpleNamespace(effective_tokens=4096, phase_name="prefill"),
        validate=native_calls.validate_native_occurrence)
    assert env is not None
    assert env["actual"]["shapes"]["cos_sin_cache"] == [3, 4096, None]
    assert env["prepare"]["complete"] is False
    assert "rope_dim_from_caller" in env["prepare"]["partial_reasons"]
    assert "eps_from_caller" in env["prepare"]["partial_reasons"]


def test_non_r31_anchor_and_missing_contract_stay_unbound(a0_instance):
    wrong_image = _compiled(a0_instance, (("image_id", "sha256:" + "0" * 64),))
    assert qkv.binding_spec(wrong_image) is None
    qk_norm = SimpleNamespace(**{**_compiled(a0_instance).__dict__,
                                 "operator": a0_instance.operator("qk_norm")})
    assert qkv.binding_spec(qk_norm) is None
    no_contract = SimpleNamespace(**{**_compiled(a0_instance).__dict__,
                                     "instance": replace(
                                         a0_instance,
                                         template=replace(
                                             a0_instance.template,
                                             fusions=()))})
    assert qkv.binding_spec(no_contract) is None


def test_wrong_typed_operands_are_rejected(a0_instance):
    compiled = _compiled(a0_instance)
    spec = qkv.binding_spec(compiled)
    assert spec is not None
    binding = _bind(compiled, spec)
    env = qkv.dynamic_call(
        binding, compiled,
        SimpleNamespace(effective_tokens=4096, phase_name="prefill"),
        validate=native_calls.validate_native_occurrence)
    assert env is not None
    bad = {**env["actual"], "shapes": {
        **env["actual"]["shapes"], "input": [4096, 8192]}}
    with pytest.raises(ModelingError,
                       match="native_call_signature_shape_mismatch:input"):
        native_calls.validate_native_occurrence(binding, bad)
    bad = {**env["actual"], "dtype": "fp16"}
    with pytest.raises(ModelingError,
                       match="native_call_signature_dtype_mismatch"):
        native_calls.validate_native_occurrence(binding, bad)
