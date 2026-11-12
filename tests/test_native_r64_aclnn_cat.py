"""Source-bound R64 A1 q-latent/q-rotary aclnnCat adapter invariants.

[Role] Exercise the operator adapter against the real R64 caller identity and
  the actual A1 sibling attention widths.

[Invariants] One ``torch.cat([ql_nope, q_pe], dim=-1)`` produces BF16
  ``[M,64,576]`` from ``[M,64,512]`` and ``[M,64,64]``; ``M`` is occurrence
  state; the call identity is stable; other images, operators, widths and
  undeclared lowering stay unbound; ``complete`` remains false because the
  ConcatD source/ABI is not retained.

[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device or locks.

[Side-Effect Policy] Test-local construction only; no product state mutation.
"""
from __future__ import annotations

from mindie_sim.prediction.native_binding import (
    bind_native_call, declare_native_call)
from mindie_sim.prediction.piecewise import (
    ImplementationPolicy, PipeSupply, RequestGeometry, WorkloadGeometry)
from mindie_sim.prediction.piecewise.compiler import CompiledOperator
from mindie_sim.prediction.piecewise.native_calls import (
    _surface, validate_native_occurrence)
from mindie_sim.prediction.piecewise.native_r64_aclnn_cat import (
    ACLNN_API, ENTRY, KERNEL_NAME, PROFILER_NAME, R64_DEVICE_OP_SHA,
    R64_IMAGE_SHA, SOURCE, SOURCE_LINES, binding_spec, dynamic_call)
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


def _compiled(*, image_id=R64_IMAGE_SHA, kind="A1", name="q_concat",
              family="copy", role="concat", heads=64, latent=512, rope=64,
              elements=None, act_bytes=2.0, extras=None, sibling=True):
    if extras is None:
        extras = (("q_concat_heads", heads), ("q_concat_latent", latent),
                  ("q_concat_rope", rope))
    signature = StaticSignature(
        role="concat",
        matrix=(("elements", int(elements if elements is not None
                                 else heads * (latent + rope))),),
        quant=(("act_bytes", act_bytes),),
        tiles=(("mt", 16), ("cores", 1)),
        extras=extras,
    )
    operator = BoundOperator(
        OperatorTemplate(name=name, family=family, role=role), signature)
    siblings = ()
    if sibling:
        attention = BoundOperator(
            OperatorTemplate(name="mla_attention", family="mla",
                             role="attention"),
            StaticSignature(
                role="attention",
                heads=(("q", heads), ("kv", 1), ("dn", 192),
                       ("dr", rope), ("dv", 512), ("rkv", latent)),
            ))
        siblings = (attention,)
    return CompiledOperator(
        operator, SUPPLY, POLICY, kind=kind, siblings=siblings,
        feature_metadata=(("image_id", image_id),))


def _binding(compiled, phase="prefill"):
    spec = binding_spec(compiled, phase=phase)
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


def test_r64_a1_q_concat_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] == ACLNN_API
    assert spec.api["version"] == "r64/8d422528"
    assert spec.source["sha256"] == R64_DEVICE_OP_SHA
    assert spec.source["symbol"] == SOURCE["symbol"]
    assert spec.source["revision"] == "r64/8d422528"
    assert spec.owner == "q_concat"
    assert binding.timing_owner == {"op_id": "A1:copy:q_concat",
                                    "scope": "call"}
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["acl_api"] == ACLNN_API
    assert spec.kernels[0]["profiler_name"] == PROFILER_NAME
    assert spec.typed["dtype"] == "bf16"
    assert spec.typed["layout"] == "NCL"
    assert spec.typed["shapes"] == {
        "ql_nope": [64, 512], "q_pe": [64, 64], "query": [64, 576]}
    assert spec.typed["attrs"]["source_lines"] == list(SOURCE_LINES)
    assert spec.typed["attrs"]["deployment_source_revision"] == "r64"

    out = dynamic_call(binding, compiled, _geometry(1024),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["shapes"] == {
        "ql_nope": [1024, 64, 512], "q_pe": [1024, 64, 64],
        "query": [1024, 64, 576]}
    prepare = out["prepare"]
    assert prepare["arguments"] == {
        "tensors": ["ql_nope", "q_pe"], "dim": -1, "out": None}
    assert prepare["argument_order"] == ["tensors", "dim", "out"]
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["match"]["m"] == 1024
    assert prepare["match"]["latent_width"] == 512
    assert prepare["match"]["rope_width"] == 64
    assert prepare["match"]["output_width"] == 576
    assert prepare["match"]["source_lines"] == list(SOURCE_LINES)
    assert prepare["observed_original"]["input_shapes"] == [
        "1024,64,512", "1024,64,64"]
    assert prepare["observed_original"]["output_shape"] == "1024,64,576"
    assert prepare["observed_original"]["output_dtype"] == "DT_BF16"
    assert prepare["acl_interface"]["status"] == "not_retained"
    assert prepare["source_lowering"]["status"] == (
        "measured_not_source_declared")
    assert "concatd_kernel_source_not_retained" in prepare["unresolved"]


def test_r64_a1_q_concat_geometry_and_binding_stability():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    held = dynamic_call(binding, compiled, _geometry(1, phase="decode"),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"] == {
        "ql_nope": [1, 64, 512], "q_pe": [1, 64, 64],
        "query": [1, 64, 576]}
    assert held["prepare"]["complete"] is False
    assert binding.call_id == _binding(_compiled())[1].call_id
    assert spec.typed["shapes"]["query"] == [64, 576]
    assert binding_spec(compiled, phase="decode") is not None
    assert binding_spec(compiled, phase="") is not None


def test_r64_a1_q_concat_uses_sibling_widths_without_extras():
    compiled = _compiled(extras=())
    spec = binding_spec(compiled, phase="prefill")
    assert spec is not None
    assert spec.typed["shapes"] == {
        "ql_nope": [64, 512], "q_pe": [64, 64], "query": [64, 576]}


def test_r64_a1_q_concat_rejects_other_sources_and_shapes():
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(kind="A0"), phase="prefill") is None
    assert binding_spec(_compiled(name="mla_attention"),
                        phase="prefill") is None
    assert binding_spec(_compiled(role="copy"), phase="prefill") is None
    assert binding_spec(_compiled(), phase="chunked_prefill") is None
    assert binding_spec(_compiled(sibling=False), phase="prefill") is None
    assert binding_spec(_compiled(latent=256, elements=64 * 576),
                        phase="prefill") is None
    assert binding_spec(_compiled(elements=64 * 512), phase="prefill") is None
    assert binding_spec(_compiled(act_bytes=4.0), phase="prefill") is None


def test_r64_a1_q_concat_rejects_empty_geometry():
    compiled = _compiled()
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
