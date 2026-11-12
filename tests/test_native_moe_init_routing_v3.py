"""Source-bound R64 incoming MoeInitRoutingV3 adapter invariants.

[Role] Exercise the operator adapter against the declared R64 All2All caller
  identity and check the actual typed occurrence for real per-layer received
  rows, including two different original GLM51 prefill layers.

[Invariants] The incoming call is
  ``torch_npu.npu_moe_init_routing_v2`` with the caller's ``unsqueeze(-1)``
  local-expert index (native top-k 1), FP4/E2M1 payload, E8M0 paired scale,
  INT32 inverse row index and INT64 count over ``[0, E_local)``.  The row
  count is the per-layer received ledger, never ``M*k``; the model top-k stays
  separate.  Other images, phases, payload features and routing rows stay
  unbound.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, OperatorPiecewiseCompiler, PipeSupply,
    RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_moe_init_routing_v3 import (
    CALL_LINES, ENTRY, KERNEL_BODY_SOURCE, PAYLOAD, R64_DEVICE_OP_SHA,
    R64_IMAGE_SHA, R64_SOURCE_COMMIT, SCALE_FORMAT, binding_spec, dynamic_call)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

SUPPLY = PipeSupply((
    ("cube", 1000.0), ("vector", 1000.0), ("hbm", 1000.0),
    ("mte1", 1000.0), ("mte2", 1000.0), ("mte3", 1000.0),
    ("fixpipe", 1000.0), ("link", 1000.0),
    ("link_hccs", 2000.0), ("link_roce", 500.0),
))
POLICY = ImplementationPolicy(mt=16, nt=16, cores=1, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)
R35_IMAGE_SHA = (
    "5461a1a8253c6dc07868956166ad94751f84c82d00ff7791c59a9c1ac497de06")
COMM = (
    ("mode", "dynamic"), ("policy", "capacity_and_world_size"),
    ("capacity", 64), ("world_size", 16), ("tp", 1), ("top_k", 8),
    ("experts_per_device", 16), ("enable_fused_mc2", 1),
    ("mega_moe_supported", 0), ("enable_prefill_mc2", 0),
    ("max_num_batched_tokens", 8192), ("max_num_reqs", 64),
    ("uniform_decode_query_len", 1),
    ("capture_sizes", (1, 2, 4, 8, 16, 32, 64)),
    ("lora", 0), ("is_draft_model", 0), ("draft_quant_supported", 1),
)
EXTRAS = (
    ("E", 256), ("k", 8), ("router_hidden", 6144),
    ("router_received_rows", 341), ("router_payload_bytes", 0.5),
    ("router_scale_cols", 192), ("router_index_bytes", 4),
    ("router_source_dtype", PAYLOAD), ("router_scale_dtype", SCALE_FORMAT),
    ("routed_weight_format", "mxfp4"),
    ("router_selected", "post_dispatch_received_rows"),
    ("router_source_operation", "MoeInitRoutingV3"),
    ("moe_comm", COMM),
)


def _compiled(*, image_id=R64_IMAGE_SHA, kind="F3", name="router_inbound",
              extras=EXTRAS):
    template = OperatorTemplate(name=name, family="router", role="inbound")
    signature = StaticSignature(
        role=name, matrix=(("elements", 8),),
        experts=(("E", 256), ("k", 8)),
        tiles=(("mt", 16), ("cores", 1)), extras=extras)
    operator = BoundOperator(template, signature)
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind=kind,
        feature_metadata=(("image_id", image_id),
                          ("image", "r64-image-8d422528")))


def _binding(compiled):
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(tokens=1024, phase="prefill"):
    return WorkloadGeometry((RequestGeometry(tokens, 0, phase=phase),),
                            phase=phase)


def test_r64_incoming_v3_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] == "aclnnMoeInitRoutingV3"
    assert spec.api["entry_source_sha256"] == R64_DEVICE_OP_SHA
    assert spec.api["caller_source_sha256"] == spec.source["sha256"]
    assert spec.source["sha256"] == (
        "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5")
    assert spec.source["symbol"] == (
        "TokenDispatcherWithAll2AllV._dispatch_postprocess")
    assert spec.kernels[0]["name"] == "MoeInitRoutingV3"
    assert spec.kernels[0]["source_sha256"] == (
        KERNEL_BODY_SOURCE["sha256"])
    assert spec.owner == "router_inbound"
    assert spec.members == ()
    assert binding.timing_owner == {
        "op_id": "F3:router:router_inbound", "scope": "call"}
    attrs = spec.typed["attrs"]
    assert attrs["source_commit"] == R64_SOURCE_COMMIT
    assert attrs["caller_source_lines"] == list(CALL_LINES)
    assert attrs["native_top_k"] == 1
    assert attrs["model_top_k"] == 8
    assert attrs["router_received_rows"] == 341
    assert attrs["expert_num"] == 256
    assert attrs["local_experts"] == 16
    assert attrs["active_expert_range"] == [0, 16]
    assert attrs["router_payload_format"] == PAYLOAD
    assert attrs["router_scale_pairs"] == 96
    assert attrs["expert_tokens_num_type"] == 1
    assert attrs["row_idx_type"] == 0

    first = dynamic_call(binding, compiled, _geometry(1024),
                         validate=validate_native_occurrence)
    assert first is not None
    shapes = first["actual"]["shapes"]
    assert shapes == {
        "x": [341, 6144],
        "expert_idx": [341, 1],
        "scale": [341, 96, 2],
        "expanded_x": [341, 6144],
        "expanded_row_idx": [341],
        "expert_tokens_count_or_cumsum": [16],
        "expanded_scale": [341, 96, 2],
    }
    prepare = first["prepare"]
    assert prepare["arguments"]["active_expert_range"] == [0, 16]
    assert prepare["arguments"]["expert_num"] == 256
    assert prepare["arguments"]["expert_tokens_num_flag"] is True
    assert prepare["arguments"]["expert_tokens_num_type"] == 1
    assert prepare["arguments"]["row_idx_type"] == 0
    assert prepare["arguments"]["quant_mode"] == -1
    assert prepare["arguments"]["x_dtype"] == PAYLOAD
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["charge"] == "one_per_native_call"
    assert prepare["match"]["payload"] == PAYLOAD
    assert prepare["match"]["source_lines"] == list(CALL_LINES)

    # A different original layer is a different occurrence, not a new call.
    other = _compiled(extras=tuple(
        (key, 669 if key == "router_received_rows" else value)
        for key, value in EXTRAS))
    other_spec, other_binding = _binding(other)
    held = dynamic_call(other_binding, other, _geometry(1024),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["x"] == [669, 6144]
    assert held["actual"]["shapes"]["scale"] == [669, 96, 2]
    assert held["prepare"]["match"]["received_rows"] == 669
    assert other_spec.typed["attrs"]["router_received_rows"] == 669
    assert other_binding.call_id != binding.call_id


def test_incoming_v3_never_uses_m_k_as_received_rows():
    compiled = _compiled()
    _, binding = _binding(compiled)
    for tokens in (1, 1024, 2048):
        result = dynamic_call(binding, compiled, _geometry(tokens),
                              validate=validate_native_occurrence)
        assert result is not None
        assert result["actual"]["shapes"]["x"] == [341, 6144]
        assert result["prepare"]["match"]["received_rows"] == 341
        # The model top-k remains static source context, not the native K.
        assert result["prepare"]["match"]["model_top_k"] == 8
        assert result["prepare"]["match"]["native_top_k"] == 1


def test_incoming_v3_rejects_other_sources_and_features():
    compiled = _compiled()
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(kind="F1"), phase="prefill") is None
    assert binding_spec(_compiled(name="router_unweighted_reorder"),
                        phase="prefill") is None
    assert binding_spec(compiled, phase="decode") is None
    assert binding_spec(compiled, phase="") is None
    fp8 = tuple(
        (key, "fp8_e4m3" if key == "router_source_dtype"
         else 1.0 if key == "router_payload_bytes" else value)
        for key, value in EXTRAS)
    assert binding_spec(_compiled(extras=fp8), phase="prefill") is None
    outgoing = tuple(
        (key, "MoeTokenPermute/MoeInitRoutingV2"
         if key == "router_source_operation" else value)
        for key, value in EXTRAS)
    assert binding_spec(_compiled(extras=outgoing), phase="prefill") is None
    wrong_routed_role = tuple(
        (key, "fp8_e4m3" if key == "routed_weight_format" else value)
        for key, value in EXTRAS)
    assert binding_spec(_compiled(extras=wrong_routed_role),
                        phase="prefill") is None
    mc2_comm = tuple(
        (key, "always_mc2" if key == "mode" else value)
        for key, value in COMM)
    mc2 = tuple(
        (key, mc2_comm if key == "moe_comm" else value)
        for key, value in EXTRAS)
    assert binding_spec(_compiled(extras=mc2), phase="prefill") is None
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(1024, phase="decode"),
                        validate=validate_native_occurrence) is None
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
