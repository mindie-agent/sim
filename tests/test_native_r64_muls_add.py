"""R64 F4 routed/shared ``muls_add`` source-recipe invariants.

[Role] Exercise the F4 join adapter against the source-selected combine
  recipe: fused ``muls_add`` (scale at join, no double scaling), plain add /
  scale-owned-by-F1 (truthful absent), and the shared token-domain route that
  admits both prefill and decode.
[Invariants] ``muls_add`` requires ``combine_variant=muls_add``,
  ``scale_owner=join``, ``ops_per_element=2`` and a declared scale > 1.
  ``plain_add`` and ``scale_owner=router`` return ``None`` even with a
  declared scale.  The identity scale (Kimi K3 ``routed_scaling_factor=1``)
  is a plain add and is absent.  Fused prefill/decode occurrences use their
  own local token count; the call identity is stable.
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
from mindie_sim.prediction.piecewise.native_r64_muls_add import (
    ENTRY, KERNEL_NAME, binding_spec, dynamic_call)
from mindie_sim.schema.subgraphs import (
    BoundOperator, OperatorTemplate, StaticSignature)

R64_IMAGE = (
    "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
    "nightly-ci-releases-v0.32.0rc-a5-openeuler-cann9.2.1-20261101-r64"
)
R64_IMAGE_ID = (
    "sha256:8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
)
HIDDEN = 6144
RATES = (
    ("cube", 1_000_000.0), ("vector", 1_000_000.0), ("hbm", 1_000_000.0),
    ("mte1", 1_000_000.0), ("mte2", 1_000_000.0), ("mte3", 1_000_000.0),
    ("fixpipe", 1_000_000.0), ("link", 1_000_000.0),
    ("link_hccs", 2_000_000.0), ("link_roce", 500_000.0),
)
POLICY = ImplementationPolicy(mt=16, nt=16, cores=32, moe_row_tile=16,
                              split_target=10 ** 9, max_splits=8)


def _compiled(*, scale=2.5, combine_variant="muls_add",
              scale_owner="join", ops=2, hidden=HIDDEN,
              act_bytes=2.0, vector_cores=64,
              model_identifier="glm_moe_dsa", image=R64_IMAGE,
              image_id=R64_IMAGE_ID):
    signature = StaticSignature(
        role="join",
        matrix=(("elements", int(hidden)),),
        quant=(("act_bytes", float(act_bytes)),),
        tiles=(("mt", 16), ("cores", 32)),
        extras=(("ops_per_element", int(ops)),
                ("read_bytes", 2 * hidden * act_bytes),
                ("write_bytes", hidden * act_bytes),
                ("routed_scaling_factor", scale),
                ("combine_variant", combine_variant),
                ("scale_owner", scale_owner),
                ("token_domain", "shared_routed_combine")),
    )
    operator = BoundOperator(
        OperatorTemplate(name="join", family="elementwise", role="join"),
        signature)
    supply = PipeSupply(RATES,
                        resource_units=(("vector", int(vector_cores)),))
    return CompiledOperator(
        operator, supply, POLICY, kind="F4", variant="join",
        feature_metadata=(("image", image), ("image_id", image_id),
                          ("model_identifier", model_identifier)))


def _binding(compiled, phase):
    spec = binding_spec(compiled, phase=phase)
    assert spec is not None
    surface = _surface(compiled, spec, source_boundary="kernel")
    declared = declare_native_call(
        surface, api=spec.api, source=spec.source, typed=spec.typed,
        kernels=spec.kernels, execution_core=spec.execution_core,
        unresolved=spec.unresolved, members=spec.members or None)
    return spec, bind_native_call(declared)


def _geometry(tokens, phase):
    return WorkloadGeometry((RequestGeometry(tokens, 0, phase=phase),),
                            phase=phase)


def test_glm_fused_muls_add_prefill_and_decode():
    compiled = _compiled()
    spec, binding = _binding(compiled, "prefill")
    assert spec.api["entry"] == ENTRY
    assert spec.api["version"] == "r64/8d422528"
    assert spec.owner == "join"
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert binding.timing_owner == {"op_id": "F4:elementwise:join",
                                    "scope": "call"}
    attrs = spec.typed["attrs"]
    assert attrs["scale"] == 2.5
    assert attrs["combine_variant"] == "muls_add"
    assert attrs["scale_owner"] == "join"
    assert attrs["selected_variant"] == "muls_add"
    assert attrs["token_domain"] == "shared_routed_combine"
    assert attrs["ops_per_element"] == 2
    assert attrs["vector_cores"] == 64
    assert attrs["producer"] == "F4:elementwise:join"
    assert attrs["parent"] == "F4:join:shared_routed_combine"
    assert attrs["double_scaling"] is False

    out = dynamic_call(binding, compiled, _geometry(1024, "prefill"),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["shapes"] == {
        "x": [1024, HIDDEN], "y": [1024, HIDDEN], "output": [1024, HIDDEN]}
    prepare = out["prepare"]
    assert prepare["arguments"]["scale"] == 2.5
    assert prepare["kernel_arguments"]["n_elements"] == 1024 * HIDDEN
    assert prepare["kernel_arguments"]["n_blocks"] == 2048
    assert prepare["kernel_arguments"]["BLOCK_SIZE"] == 3072
    assert prepare["grid"] == [64]
    assert prepare["complete"] is False
    assert prepare["match"]["token_domain"] == "shared_routed_combine"
    assert prepare["match"]["phase"] == "prefill"
    assert prepare["match"]["selected_variant"] == "muls_add"
    assert prepare["attribution"]["scale_owner"] == "join"
    assert prepare["attribution"]["double_scaling"] is False
    assert "glm4_moe" in prepare["source_evidence"]["combine_source"]
    assert "kimi_linear" in prepare["source_evidence"]["combine_source"]

    decode_spec, decode_binding = _binding(compiled, "decode")
    assert decode_spec.typed["attrs"]["token_domain"] == "shared_routed_combine"
    decode = dynamic_call(decode_binding, compiled, _geometry(2, "decode"),
                          validate=validate_native_occurrence)
    assert decode is not None
    assert decode["actual"]["shapes"] == {
        "x": [2, HIDDEN], "y": [2, HIDDEN], "output": [2, HIDDEN]}
    assert decode["prepare"]["grid"] == [4]
    assert decode["prepare"]["match"]["phase"] == "decode"
    assert decode["prepare"]["match"]["selected_variant"] == "muls_add"


def test_k3_identity_scale_is_plain_absent():
    compiled = _compiled(scale=1.0, combine_variant="plain_add",
                         scale_owner="none", ops=1,
                         model_identifier="kimi_k3")
    recipe = compiled.operator.theta.find("combine_variant", "")
    assert recipe == "plain_add"
    assert binding_spec(compiled, phase="prefill") is None
    assert binding_spec(compiled, phase="decode") is None


def test_router_owned_scale_is_absent_no_double_scaling():
    compiled = _compiled(scale=2.5, combine_variant="plain_add",
                         scale_owner="router", ops=1)
    assert binding_spec(compiled, phase="prefill") is None
    assert binding_spec(compiled, phase="decode") is None


def test_unresolved_variant_is_absent():
    compiled = _compiled(scale=None, combine_variant="unresolved",
                         scale_owner="unresolved", ops=2)
    assert binding_spec(compiled, phase="prefill") is None


def test_inconsistent_fused_recipe_is_absent():
    compiled = _compiled(scale=2.5, combine_variant="muls_add",
                         scale_owner="join", ops=1)
    assert binding_spec(compiled, phase="prefill") is None


def test_other_image_is_incompatible():
    compiled = _compiled(image=(
        "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
        "nightly-ci-releases-v0.31.0-a5-cann9.2.0-20261023-r31"),
        image_id=(
            "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"))
    assert binding_spec(compiled, phase="prefill") is None
