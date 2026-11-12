"""Focused source-scoped SwiGlu adapter contract.

[Role] Prove the R31 standalone F0 SwiGlu binding admits exactly the
  BF16 gate/up occurrence, expands its real single-input ``dim=-1`` operands,
  and stays unavailable for another image, another activation or a fused
  quantized row.
[Invariants] The occurrence row count never enters the static signature; the
  two logical gate/up halves exactly reconstruct the input width; the exact
  R31 caller registration and kernel/API references stay explicit; the
  prepare record never claims a completed physical execution.
[Timeouts & Locks] Pure CPU metadata; no device, file or network access.
[Side-Effect Policy] Read-only test data; no product state is mutated.
"""
from __future__ import annotations

from types import SimpleNamespace

from mindie_sim.prediction.piecewise.native_calls import (
    _CallSpec, validate_native_occurrence,
)
from mindie_sim.prediction.piecewise.native_swiglu import (
    ACLNN_SYMBOL, CAPTURE_CSV_SHA256, ENTRY, IMPLEMENTATION, R31_CALLER_SHA256,
    R31_IMAGE_SHA, R31_REVISION, binding_spec, dynamic_call,
)
from mindie_sim.schema.subgraphs import StaticSignature

ELEMENTS = 8704


def _compiled(*, image=R31_IMAGE_SHA, kind="F0", family="elementwise",
              name="activation", elements=ELEMENTS, read_bytes=34816.0,
              write_bytes=17408.0, activation="silu", gated=1,
              lowerings=("gdn_decode",), gate_up_format="bf16"):
    theta = StaticSignature(
        role="activation",
        matrix=(("elements", elements),),
        quant=(("act_bytes", 1.0),),  # ordinary quant-ledger proxy, not the ABI
        extras=(("activation", activation), ("gated", gated),
                ("ops_per_element", 4), ("read_bytes", read_bytes),
                ("write_bytes", write_bytes)),
    )
    operator = SimpleNamespace(
        family=family, name=name, theta=theta,
        template=SimpleNamespace(role="activation"))
    gate_up = SimpleNamespace(
        name="gate_up", family="gemm",
        theta=StaticSignature(role="gate_up",
                              quant=(("act_format", "int8"),
                                     ("output_format", gate_up_format)),
                              extras=(("category", "dense_mlp"),)),
        template=SimpleNamespace(role="gate_up"))
    metadata = [("image_id", "sha256:" + image)]
    if lowerings is not None:
        metadata.append(("source_lowerings", tuple(lowerings)))
    return SimpleNamespace(
        kind=kind, layer=0, variant="silu", siblings=(gate_up,),
        feature_metadata=tuple(metadata), operator=operator)


def _binding(spec):
    return SimpleNamespace(
        api=spec.api, typed={**spec.typed, "signature_hash": "focused"},
        source=spec.source, call_id="focused-swiglu-call",
        unresolved=spec.unresolved)


def test_binding_spec_is_the_exact_r31_standalone_f0_call():
    compiled = _compiled()
    spec = binding_spec(compiled, phase="prefill")
    assert isinstance(spec, _CallSpec)
    assert spec.api == {
        "namespace": "torch", "entry": ENTRY,
        "aclnn_symbol": ACLNN_SYMBOL, "version": R31_REVISION}
    assert spec.source["path"] == "vllm_ascend/utils.py"
    assert spec.source["sha256"] == R31_CALLER_SHA256
    assert spec.source["revision"] == R31_REVISION
    assert spec.typed["dtype"] == "bf16"
    assert spec.typed["layout"] == "ND"
    assert spec.typed["shapes"] == {"input": [17408], "output": [8704]}
    attrs = spec.typed["attrs"]
    assert attrs["implementation"] == IMPLEMENTATION
    assert attrs["native_kind"] == "SwiGlu"
    assert attrs["deployment_image_id"] == R31_IMAGE_SHA
    assert attrs["dim"] == -1
    assert attrs["gate_first"] is True
    assert attrs["gate_order"] == ["gate", "up"]
    assert attrs["compute"] == "silu(gate) * up"
    assert attrs["read_bytes"] == 4 * ELEMENTS
    assert attrs["write_bytes"] == 2 * ELEMENTS
    assert attrs["caller_source_path"] == "vllm_ascend/ops/activation.py"
    assert attrs["caller_source_symbol"] == "AscendSiluAndMul.forward_oot"
    assert attrs["kernel_source_revision"] == "cann-9.1.0-near-source@955f8b33"
    assert spec.owner == "F0:elementwise:activation"
    assert spec.execution_core == "aiv"
    assert len(spec.kernels) == 1
    assert spec.kernels[0]["name"] == "SwiGlu"
    assert spec.kernels[0]["role"] == "vector"
    assert spec.kernels[0]["evidence"] == "source_declared"
    assert "kernel_source_is_cann_9_1_near_source_not_r31_package" in spec.unresolved


def test_dynamic_call_reconstructs_the_captured_prefill_operands():
    compiled = _compiled()
    spec = binding_spec(compiled, phase="prefill")
    binding = _binding(spec)
    geometry = SimpleNamespace(effective_tokens=4096, phase_name="prefill")
    env = dynamic_call(binding, compiled, geometry,
                       validate=validate_native_occurrence)
    assert env is not None
    actual = env["actual"]
    assert actual["dtype"] == "bf16"
    assert actual["layout"] == "ND"
    assert actual["shapes"] == {"input": [4096, 17408],
                                "output": [4096, 8704]}
    assert actual["attrs"]["tokens"] == 4096
    assert actual["attrs"]["phase"] == "prefill"
    prepare = env["prepare"]
    assert prepare["schema"] == "mindie-sim/swiglu-native-prepare/1"
    assert prepare["call"] == ENTRY
    assert prepare["aclnn_symbol"] == ACLNN_SYMBOL
    assert prepare["arguments"] == {"x": "input", "dim": -1, "y": "output"}
    assert prepare["argument_order"] == ["x", "dim", "y"]
    assert prepare["prototype"] == "torch_npu.npu_swiglu(input, dim=-1) -> Tensor"
    assert prepare["descriptors"]["input"] == {
        "shape": [4096, 17408], "dtype": "bf16", "layout": "ND",
        "role": "input", "pointer": None, "allocation": "external"}
    assert prepare["descriptors"]["output"] == {
        "shape": [4096, 8704], "dtype": "bf16", "layout": "ND",
        "role": "output", "pointer": None, "allocation": "external"}
    assert prepare["split"]["gate"] == [4096, 8704]
    assert prepare["split"]["up"] == [4096, 8704]
    assert prepare["split"]["order"] == ["gate", "up"]
    assert prepare["split"]["allocation"] == "view_of_input"
    assert prepare["source"]["sha256"] == R31_CALLER_SHA256
    assert prepare["kernel"]["sha256"] == (
        "4e152645e295a51fe8946881f490a5520c45beb3817a325e94e8f5c6b13254ee")
    assert prepare["api_binary"]["symbol"] == ACLNN_SYMBOL
    assert prepare["source_version"] == R31_REVISION
    assert prepare["capture"]["kernel_details_csv_sha256"] == CAPTURE_CSV_SHA256
    assert prepare["capture"]["input_shape"] == [4096, 17408]
    assert prepare["capture"]["output_shape"] == [4096, 8704]
    assert prepare["capture"]["input_dtype"] == "bf16"
    assert prepare["capture"]["block_num"] == 64
    assert prepare["capture"]["not_used_as_prediction_input"] is True
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["workspace"]["bytes"] is None
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert "no_physical_completeness" in prepare["non_claims"]
    assert "no_numeric_prediction_change" in prepare["non_claims"]


def test_decode_rows_share_the_same_static_signature_and_use_the_occurrence():
    compiled = _compiled()
    spec = binding_spec(compiled, phase="decode")
    binding = _binding(spec)
    env = dynamic_call(
        binding, compiled, SimpleNamespace(effective_tokens=2, phase_name="decode"),
        validate=validate_native_occurrence)
    assert env is not None
    assert env["prepare"]["shapes"] == {"input": [2, 17408],
                                        "output": [2, 8704]}
    assert env["prepare"]["match"]["input_width"] == 17408
    assert env["prepare"]["match"]["output_width"] == 8704
    # The token lane is an occurrence fact, not a second immutable call.
    assert "tokens" not in spec.typed["attrs"]
    assert "phase" not in spec.typed["attrs"]


def test_other_images_activations_and_quantized_rows_are_not_admitted():
    assert binding_spec(_compiled(image="f" * 64)) is None
    assert binding_spec(_compiled(kind="F1")) is None
    assert binding_spec(_compiled(family="gemm")) is None
    assert binding_spec(_compiled(name="down")) is None
    assert binding_spec(_compiled(activation="gelu")) is None
    assert binding_spec(_compiled(gated=0)) is None
    assert binding_spec(_compiled(lowerings=("gdn_decode",
                                             "dequant_swiglu_quant"))) is None
    assert binding_spec(_compiled(lowerings=None)) is None
    assert binding_spec(_compiled(gate_up_format="fp16")) is None
    assert binding_spec(_compiled(lowerings=())) is not None
    # A quantized activation writes/reads one byte per element, not BF16.
    assert binding_spec(_compiled(read_bytes=17408.0,
                                  write_bytes=8704.0)) is None
    assert binding_spec(_compiled(), phase="mixed") is None


def test_wrong_binding_and_empty_geometry_stay_unavailable():
    compiled = _compiled()
    spec = binding_spec(compiled)
    binding = _binding(spec)
    wrong = SimpleNamespace(**{**binding.__dict__,
                               "api": {**binding.api, "entry": "other"}})
    assert dynamic_call(wrong, compiled,
                        SimpleNamespace(effective_tokens=4096,
                                        phase_name="prefill"),
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, compiled,
                        SimpleNamespace(effective_tokens=0,
                                        phase_name="prefill"),
                        validate=validate_native_occurrence) is None
