"""Source-bound R64 outgoing MoeTokenPermute adapter invariants.

[Role] Exercise the operator adapter against the declared R64 caller identity
  and check the actual typed occurrence from the existing F1 physical record.

[Invariants] The outbound call is ``torch_npu.npu_moe_token_permute`` with
  ``num_out_tokens = M*k``, produces BF16 ``[M*k,H]`` + INT32 ``[M*k]``, keeps
  the call identity independent of M, and rejects other images, phases,
  routing rows and the separate ``npu_moe_init_routing_v2`` capability.

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
from mindie_sim.prediction.piecewise.native_moe_token_permute import (
    ACLNN_API, ENTRY, KERNEL_NAME, R64_IMAGE_SHA, R64_SOURCE_COMMIT,
    SOURCE, SOURCE_LINES, binding_spec, dynamic_call)
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
EXTRAS = (
    ("router_hidden", 6144), ("router_dtype_bytes", 2),
    ("router_indices_bytes", 4), ("router_prob_bytes", 2),
    ("router_source_operation", "MoeTokenPermute/MoeInitRoutingV2"),
    ("router_source_dtype", "bf16"),
    ("router_source_device", "AIV"),
    ("router_selected", "prefill_local_outbound"),
)


def _compiled(*, image_id=R64_IMAGE_SHA, kind="F1", name="router_permute",
              extras=EXTRAS):
    template = OperatorTemplate(name=name, family="router", role="permute")
    signature = StaticSignature(
        role=name, matrix=(("elements", 8),),
        experts=(("E", 256), ("k", 8)),
        tiles=(("mt", 128), ("cores", 1)), extras=extras)
    operator = BoundOperator(template, signature)
    return OperatorPiecewiseCompiler(SUPPLY, POLICY).compile_operator(
        operator, kind=kind, feature_metadata=(("image_id", image_id),))


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


def test_r64_outbound_token_permute_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    assert spec.api["entry"] == ENTRY
    assert spec.source["sha256"] == SOURCE["sha256"]
    assert spec.source["symbol"] == SOURCE["symbol"]
    assert spec.owner == "router_permute"
    assert binding.timing_owner == {
        "op_id": "F1:router:router_permute", "scope": "call"}
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.typed["attrs"]["acl_api"] == ACLNN_API
    assert spec.typed["attrs"]["source_commit"] == R64_SOURCE_COMMIT
    assert spec.typed["attrs"]["source_lines"] == list(SOURCE_LINES)
    assert spec.typed["attrs"]["num_out_tokens_source"] == "topk_ids.numel()"
    assert spec.typed["attrs"]["num_out_tokens_expr"] == "M*k"

    first = dynamic_call(binding, compiled, _geometry(1024),
                         validate=validate_native_occurrence)
    assert first is not None
    shapes = first["actual"]["shapes"]
    assert shapes == {
        "tokens": [1024, 6144], "indices": [1024, 8],
        "permuted_tokens": [8192, 6144], "sorted_indices": [8192]}
    prepare = first["prepare"]
    assert prepare["arguments"] == {
        "tokens": "tokens", "indices": "indices", "num_out_tokens": 8192}
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["match"]["op_type"] == KERNEL_NAME
    assert prepare["match"]["source_lines"] == list(SOURCE_LINES)

    held = dynamic_call(binding, compiled, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["tokens"] == [2048, 6144]
    assert held["actual"]["shapes"]["permuted_tokens"] == [16384, 6144]
    assert held["prepare"]["arguments"]["num_out_tokens"] == 2048 * 8
    # A new M is a new occurrence, not a new selected call.
    assert binding.call_id == _binding(_compiled())[1].call_id


def test_r64_outbound_token_permute_rejects_other_sources():
    compiled = _compiled()
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(compiled, phase="decode") is None
    assert binding_spec(compiled, phase="") is None
    assert binding_spec(_compiled(kind="F2"), phase="prefill") is None
    assert binding_spec(_compiled(name="router_topk"), phase="prefill") is None
    init_routing_only = (
        ("router_hidden", 6144), ("router_dtype_bytes", 2),
        ("router_indices_bytes", 4),
        ("router_source_operation", "MoeInitRoutingV2"))
    assert binding_spec(_compiled(extras=init_routing_only),
                        phase="prefill") is None
    fp8_outbound = (
        ("router_hidden", 6144), ("router_dtype_bytes", 1),
        ("router_indices_bytes", 4),
        ("router_source_operation", "MoeTokenPermute/MoeInitRoutingV2"),
        ("router_source_dtype", "fp8_e4m3"),
        ("router_source_device", "AIV"),
        ("router_selected", "prefill_local_outbound"),
    )
    assert binding_spec(_compiled(extras=fp8_outbound),
                        phase="prefill") is None
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(1024, phase="decode"),
                        validate=validate_native_occurrence) is None


def test_outbound_token_permute_requires_dense_geometry():
    compiled = _compiled()
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
