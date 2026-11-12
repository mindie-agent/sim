"""Focused R31 A3 ``aclnnInplaceZero`` core-output adapter checks.

[Role] Prove the real R31 Qwen3.6 A3 ``core_zeros`` row selects one native
  ``torch.zeros`` / ``aclnnInplaceZero`` / ``ZerosLike`` call identity and
  prepares the exact ``[T,24,128]`` BF16 core-output occurrence.

[Invariants] The R31 deployment image and the source-declared A3 fill row are
  the only selectors; kept value heads, head width and element width stay in
  the binding and only the forward token count changes.  The zero is the
  freshly allocated ``core_attn_out`` accumulator, so it is one in-place
  operand and not the beta/alpha/q/k/v/z slice family.  Other images, phases
  and rows do not validate.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local case instantiation only; no product state
  mutation, tensor allocation or device launch.
"""
from __future__ import annotations

import json
from pathlib import Path

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    AnalysisDomain, ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r31_inplace_zero import (
    ACLNN_API, ACLNN_SYMBOLS, ENTRY, KERNEL_NAME, LIBOPAPI_SHA, PROFILER_NAME,
    R31_IMAGE_SHA, ROLE, SOURCE, SOURCE_LINES, binding_spec, dynamic_call)
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.subgraphs.instantiate import (
    FeatureConfig, instantiate_model)

ROOT = Path(__file__).resolve().parents[1]
QWEN = ROOT / "examples/cases/longctx-decode-20261030/qwen36-a5-tp2.json"
R31 = (("image_id", "sha256:" + R31_IMAGE_SHA),)
R64 = (("image_id",
        "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"),)
UNKNOWN = (("image_id", "sha256:" + "f" * 64),)
DOMAIN = AnalysisDomain(max_tokens=10 ** 9, max_moe_rows=10 ** 9,
                        max_query_tiles=10 ** 9, max_kv_tiles=10 ** 9,
                        max_index_rows=10 ** 12, max_work=10 ** 15,
                        max_q=10 ** 9, max_s_kv=10 ** 9, max_seq=10 ** 9,
                        max_batch=10 ** 6)


def _case():
    case = parse_case(json.loads(QWEN.read_text()))
    features = FeatureConfig.from_model(
        case.model, tp=2, dp=4, ep=1, pp=1, cores=64, token_tile=16,
        kv_tile=16, moe_row_tile=4, split_target=1000, max_splits=4)
    return case, instantiate_model(case.model, features)


def _compiler(case):
    return OperatorPiecewiseCompiler(
        PipeSupply.from_hardware(case.machine, dtype="bf16"),
        ImplementationPolicy(cores=64), domain=DOMAIN)


def _subgraph(inst, kind, name):
    return next(sub for layer in inst.layers for sub in layer.subgraphs
                if sub.kind == kind
                and any(operator.name == name for operator in sub.operators))


def _geometry(tokens, phase="prefill"):
    return WorkloadGeometry(
        (RequestGeometry(tokens, tokens, phase=phase),),
        phase=phase, cohort_batch=1, local_batch=1)


def _static_decode_geometry():
    """The source FEATURES8 static dense-SP decode lane: valid1 -> rows2."""
    return WorkloadGeometry(
        (RequestGeometry(1, 4095, phase="decode"),), phase="decode",
        tp_width=2, dense_sequence_parallel=True, graph_dispatch="full",
        capture_sizes=(2,), cohort_batch=1, local_batch=1)


def _binding(compiled, spec):
    surface = _surface(compiled, spec, source_boundary="kernel")
    return bind_native_call(declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved))


def test_r31_core_zero_prepare_only():
    case, inst = _case()
    compiler = _compiler(case)
    sub = _subgraph(inst, "A3", "core_zeros")
    compiled = compiler.compile(sub, feature_metadata=R31)
    operator = compiled.operator("core_zeros")
    spec = binding_spec(operator, phase="prefill")
    assert spec is not None
    assert spec.api["namespace"] == "torch"
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] == ACLNN_API
    assert spec.source["path"] == SOURCE["path"]
    assert spec.source["sha256"] == SOURCE["sha256"]
    assert spec.source["symbol"] == SOURCE["symbol"]
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["evidence"] == "measured"
    assert spec.kernels[0]["acl_api"] == ACLNN_API
    assert spec.kernels[0]["profiler_name"] == PROFILER_NAME
    assert spec.kernels[0]["acl_symbols"] == list(ACLNN_SYMBOLS)
    assert spec.kernels[0]["acl_symbol_library_sha256"] == LIBOPAPI_SHA
    assert spec.execution_core == "aiv"
    for reason in ("aten_dispatch_to_aclnn_inplace_zero_not_retained",
                   "acl_api_header_not_retained",
                   "acl_tensor_argument_abi_not_retained",
                   "zeros_like_kernel_source_not_retained",
                   "kernel_binary_selection_unresolved"):
        assert reason in spec.unresolved
    assert spec.owner == "core_zeros"
    attrs = spec.typed["attrs"]
    assert attrs["buffer_role"] == ROLE
    assert attrs["allocation"] == "torch.zeros"
    assert attrs["inplace"] is True
    assert attrs["kept_heads"] == 24
    assert attrs["head_dim"] == 128
    assert attrs["elements_per_token"] == 3072
    assert attrs["dtype_bytes"] == 2
    assert attrs["read_bytes"] == 0
    assert attrs["write_bytes"] == 6144
    assert attrs["producer_op_id"] is None
    assert attrs["source_lines"] == [422, 423, 424, 425, 426]
    assert attrs["source_line_allocation"] == SOURCE_LINES["allocation"] == 422
    assert attrs["source_line_shape"] == SOURCE_LINES["shape"] == 423
    assert attrs["consumer_op_ids"] == [
        "A3:recurrent:gated_delta", "A3:elementwise:gated_norm"]
    assert spec.typed["dtype"] == "bf16"
    assert spec.typed["layout"] == "ND"
    assert spec.typed["phase"] == "all"
    assert spec.typed["shapes"] == {
        "self": [24, 128], "output": [24, 128]}
    assert attrs["inner_valid_only_lines"] == [487, 491]
    assert attrs["inner_path_claimed"] is False
    assert attrs["phase_scope"] == "outer_allocation_before_phase"

    binding = _binding(operator, spec)
    assert binding.op_id == "A3:copy:core_zeros"
    assert binding.timing_owner == {
        "op_id": "A3:copy:core_zeros", "scope": "call"}
    first = dynamic_call(binding, operator, _geometry(4096),
                         validate=validate_native_occurrence)
    assert first is not None
    actual = first["actual"]
    assert actual["dtype"] == "bf16"
    assert actual["layout"] == "ND"
    assert actual["shapes"] == {
        "self": [4096, 24, 128], "output": [4096, 24, 128]}
    assert actual["attrs"]["tokens"] == 4096
    assert actual["attrs"]["elements_per_token"] == 3072
    prepare = first["prepare"]
    assert prepare["call"] == ENTRY
    assert prepare["native_entry"] == ACLNN_API
    assert prepare["arguments"] == {"selfRef": "core_attn_out"}
    assert prepare["argument_order"] == ["selfRef"]
    assert prepare["torch_arguments"]["size"] == [4096, 24, 128]
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["match"]["native_type"] == KERNEL_NAME
    assert prepare["match"]["source_lines"] == [422, 426]
    assert prepare["parent_allocation_line"] == 422
    assert prepare["inner_valid_only_lines"] == [487, 491]
    assert prepare["inner_path_claimed"] is False
    assert prepare["phase"] == "prefill"
    assert prepare["match"]["m"] == 4096
    assert prepare["descriptors"]["output"]["alias_group"] == "core_attn_out"
    assert prepare["descriptors"]["output"]["aliases"] == "self"
    assert prepare["shapes"] == {
        "self": [4096, 24, 128], "output": [4096, 24, 128]}

    held = dynamic_call(binding, operator, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["self"] == [2048, 24, 128]
    assert held["prepare"]["match"]["m"] == 2048
    # A new token count is a new occurrence, not a new selected call.
    same = compiled.operator("core_zeros")
    assert binding.call_id == _binding(
        same, binding_spec(same, phase="prefill")).call_id


def test_r31_core_zero_is_phase_neutral_and_uses_shared_lane():
    case, inst = _case()
    compiler = _compiler(case)
    sub = _subgraph(inst, "A3", "core_zeros")
    compiled = compiler.compile(sub, feature_metadata=R31)
    operator = compiled.operator("core_zeros")
    prefill = binding_spec(operator, phase="prefill")
    decode = binding_spec(operator, phase="decode")
    empty = binding_spec(operator, phase="")
    assert prefill is not None and decode is not None and empty is not None
    assert prefill.typed == decode.typed == empty.typed
    assert prefill.typed["phase"] == "all"
    assert binding_spec(operator, phase="mixed") is None

    pf_binding = _binding(operator, prefill)
    dc_binding = _binding(operator, decode)
    assert pf_binding.call_id == dc_binding.call_id

    pf = dynamic_call(pf_binding, operator, _geometry(4096),
                      validate=validate_native_occurrence)
    assert pf is not None
    assert pf["actual"]["shapes"]["self"] == [4096, 24, 128]
    assert pf["prepare"]["phase"] == "prefill"

    # The raw decode geometry has one valid row; FEATURES8's shared
    # operator_layout selects the source role's physical gathered lane (2).
    raw_dc = _static_decode_geometry()
    assert raw_dc.effective_tokens == 1
    assert raw_dc.token_lanes()["gathered_tokens"] == 2
    projected = raw_dc.operator_layout("A3", "", "core_zeros", "copy")
    assert projected.effective_tokens == 2
    dc = dynamic_call(dc_binding, operator, projected,
                      validate=validate_native_occurrence)
    assert dc is not None
    assert dc["actual"]["shapes"] == {
        "self": [2, 24, 128], "output": [2, 24, 128]}
    assert dc["actual"]["attrs"]["phase"] == "decode"
    assert dc["actual"]["attrs"]["token_lane"] == (
        "shared_operator_layout_role_projection")
    assert dc["prepare"]["phase"] == "decode"
    assert dc["prepare"]["inner_path_claimed"] is False


def test_r31_core_zero_rejects_other_sources_and_rows():
    case, inst = _case()
    compiler = _compiler(case)
    sub = _subgraph(inst, "A3", "core_zeros")
    compiled = compiler.compile(sub, feature_metadata=R31)
    operator = compiled.operator("core_zeros")
    assert binding_spec(operator, phase="prefill") is not None

    r64 = compiler.compile(sub, feature_metadata=R64)
    assert binding_spec(r64.operator("core_zeros"), phase="prefill") is None
    assert binding_spec(r64.operator("core_zeros"), phase="decode") is None
    unknown = compiler.compile(sub, feature_metadata=UNKNOWN)
    assert binding_spec(unknown.operator("core_zeros"),
                        phase="prefill") is None

    # The beta/alpha/q/k/v/z slices are separate native calls.
    for name in ("beta_slice", "alpha_slice", "qkv_slice", "q_slice",
                 "k_slice", "v_slice", "z_slice"):
        row = compiler.compile(_subgraph(inst, "A3", name),
                               feature_metadata=R31)
        assert binding_spec(row.operator(name), phase="prefill") is None

    binding = _binding(operator, binding_spec(operator, phase="decode"))
    assert dynamic_call(binding, operator, _geometry(1, phase="draft"),
                        validate=validate_native_occurrence) is None
