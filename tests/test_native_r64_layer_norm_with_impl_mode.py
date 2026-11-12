"""Source-bound R64 I0 indexer-k aclnnLayerNormWithImplMode adapter invariants.

[Role] Exercise the operator adapter against the selected same-image
  ``nn.LayerNorm`` source slot declared by the applied I0 producer, the real
  R64 four-input LayerNormV4 contract and the actual I0 indexer geometry.

[Invariants] One declared ``[M,D]`` input with a one-element INT32
  ``normalized_shape`` array and ``[D]`` gamma/beta produces ``[M,D]`` output
  plus ``[M,1]`` mean/rstd.  ``D``, epsilon and the resolved dtype are the
  selected source slot's declared values, not the original case's
  128/1e-6/FP32; ``M`` and the explicit prefill/decode phase are occurrence
  state.  The row is I0 and is ordered after the K projection, before K
  rope/cache store, then the I1 search; the old after-search blanket I1
  linkage is rejected.  The native ACLNN symbol/kernel are observed prepare
  evidence, not the selector; the ACLNN wrapper/dispatch, implMode value, C
  header and kernel source remain unresolved.

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
from mindie_sim.prediction.piecewise.native_r64_layer_norm_with_impl_mode import (
    ACLNN_API, ENTRY, KERNEL_NAME, LAYER_NORM_SOURCE, PROFILER_NAME,
    R64_IMAGE_SHA, SOURCE_KIND, SOURCE_ORDER, SOURCE_VARIANTS, STAGE,
    binding_spec, dynamic_call)
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


def _bytes(dtype):
    return 4 if dtype == "fp32" else 2


def _extras(width=128, eps=1e-6, dtype="fp32", *, variant="glm_moe_dsa",
            source_kind=SOURCE_KIND, source_stage=STAGE,
            source_path=None, source_symbol=None, source_lines=None,
            source_sha256=None, call_site_path=None, call_site_symbol=None,
            call_site_lines=None, call_site_sha256=None,
            layer_norm_source_path=None, layer_norm_source_symbol=None,
            layer_norm_source_lines=None, layer_norm_source_sha256=None,
            attribution_owner=None, normalized_shape=None, index_dim=None,
            elements=None,
            include=("source_kind", "source_path", "source_symbol",
                     "source_lines", "source_sha256", "call_site_path",
                     "call_site_symbol", "call_site_lines",
                     "call_site_sha256", "layer_norm_source_path",
                     "layer_norm_source_symbol", "layer_norm_source_lines",
                     "layer_norm_source_sha256", "layer_norm_dtype_source",
                     "source_stage", "attribution_owner", "index_dim",
                     "norm_eps", "normalized_shape", "norm_dtype")):
    variant_data = SOURCE_VARIANTS[variant]
    model = variant_data["model"]
    call_site = variant_data["call_site"]
    layer_norm = LAYER_NORM_SOURCE
    nbytes = _bytes(dtype)
    values = {
        "ops_per_element": 5,
        "read_bytes": width * nbytes,
        "write_bytes": width * nbytes + 8,
        "gamma_bytes": width * nbytes,
        "gamma_storage_bytes": width * nbytes,
        "gamma_format": dtype,
        "gamma_storage_format": dtype,
        "gamma_role": "indexer.k_norm",
        "rstd_bytes": 8,
        "source_kind": source_kind,
        "source_path": model["path"] if source_path is None else source_path,
        "source_symbol": (model["symbol"] if source_symbol is None
                          else source_symbol),
        "source_lines": (list(model["lines"]) if source_lines is None
                         else source_lines),
        "source_sha256": (model["sha256"] if source_sha256 is None
                          else source_sha256),
        "call_site_path": (call_site["path"] if call_site_path is None
                           else call_site_path),
        "call_site_symbol": (call_site["symbol"] if call_site_symbol is None
                             else call_site_symbol),
        "call_site_lines": (list(call_site["lines"])
                            if call_site_lines is None else call_site_lines),
        "call_site_sha256": (call_site["sha256"] if call_site_sha256 is None
                             else call_site_sha256),
        "layer_norm_source_path": (layer_norm["path"]
                                   if layer_norm_source_path is None
                                   else layer_norm_source_path),
        "layer_norm_source_symbol": (layer_norm["symbol"]
                                     if layer_norm_source_symbol is None
                                     else layer_norm_source_symbol),
        "layer_norm_source_lines": (list(layer_norm["lines"])
                                    if layer_norm_source_lines is None
                                    else layer_norm_source_lines),
        "layer_norm_source_sha256": (layer_norm["sha256"]
                                     if layer_norm_source_sha256 is None
                                     else layer_norm_source_sha256),
        "layer_norm_dtype_source": layer_norm["dtype_source"],
        "source_stage": source_stage,
        "attribution_owner": (f"{STAGE}:indexer:index_k_norm"
                              if attribution_owner is None
                              else attribution_owner),
        "index_dim": width if index_dim is None else index_dim,
        "norm_eps": eps,
        "normalized_shape": ([width] if normalized_shape is None
                             else list(normalized_shape)),
        "norm_dtype": dtype,
        "deployment_image_id": R64_IMAGE_SHA,
        "deployment_source_revision": "r64",
    }
    return tuple((key, value) for key, value in values.items()
                 if key in include)


EXTRAS = _extras()


def _compiled(*, image_id=R64_IMAGE_SHA, kind=STAGE, name="index_k_norm",
              family="elementwise", role="norm", elements=128, extras=EXTRAS):
    signature = StaticSignature(
        role=role,
        matrix=(("elements", int(elements)),),
        quant=(("act_bytes", 2.0),),
        tiles=(("mt", 16), ("cores", 1)),
        extras=extras,
    )
    operator = BoundOperator(
        OperatorTemplate(name=name, family=family, role=role), signature)
    return CompiledOperator(
        operator, SUPPLY, POLICY, kind=kind,
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


def _work(tokens, width, dtype, phase):
    nbytes = _bytes(dtype)
    return {
        "source": "declared_I0_nn_LayerNorm_shape_dtype",
        "dtype": dtype,
        "dtype_bytes": nbytes,
        "elements": tokens * width,
        "vector_ops": tokens * (width * 5 + -(-width // 64)),
        "read_bytes": tokens * width * nbytes + width * nbytes,
        "write_bytes": tokens * (width * nbytes + 8),
        "gamma_bytes": width * nbytes,
        "y_bytes": tokens * width * nbytes,
        "mean_rstd_bytes": tokens * 8,
        "normalized_shape": [width],
        "source_stage": STAGE,
        "phase": phase,
    }


def test_r64_i0_indexer_k_layer_norm_prepare_only():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    model = SOURCE_VARIANTS["glm_moe_dsa"]["model"]
    call_site = SOURCE_VARIANTS["glm_moe_dsa"]["call_site"]
    assert spec.api["entry"] == ENTRY
    assert spec.api["aclnn_symbol"] == ACLNN_API
    assert spec.api["version"] == "r64/8d422528"
    assert spec.source["path"] == model["path"]
    assert spec.source["symbol"] == model["symbol"]
    assert spec.source["sha256"] == model["sha256"]
    assert spec.source["revision"] == "r64/8d422528"
    assert spec.owner == "index_k_norm"
    assert binding.timing_owner == {"op_id": "I0:elementwise:index_k_norm",
                                    "scope": "call"}
    assert spec.kernels[0]["name"] == KERNEL_NAME
    assert spec.kernels[0]["acl_api"] == ACLNN_API
    assert spec.kernels[0]["profiler_name"] == PROFILER_NAME
    assert spec.typed["dtype"] == "fp32"
    assert spec.typed["layout"] == "ND"
    assert spec.typed["shapes"] == {
        "input": [128], "normalized_shape": [1], "gamma": [128],
        "beta": [128], "output": [128], "mean": [1], "rstd": [1]}
    attrs = spec.typed["attrs"]
    assert attrs["source_kind"] == SOURCE_KIND
    assert attrs["source_stage"] == STAGE
    assert attrs["source_path"] == model["path"]
    assert attrs["source_symbol"] == model["symbol"]
    assert attrs["source_lines"] == list(model["lines"])
    assert attrs["source_sha256"] == model["sha256"]
    assert attrs["call_site_path"] == call_site["path"]
    assert attrs["call_site_symbol"] == call_site["symbol"]
    assert attrs["call_site_lines"] == list(call_site["lines"])
    assert attrs["call_site_sha256"] == call_site["sha256"]
    assert attrs["layer_norm_source_path"] == LAYER_NORM_SOURCE["path"]
    assert attrs["layer_norm_source_symbol"] == LAYER_NORM_SOURCE["symbol"]
    assert attrs["layer_norm_source_lines"] == list(LAYER_NORM_SOURCE["lines"])
    assert attrs["layer_norm_source_sha256"] == LAYER_NORM_SOURCE["sha256"]
    assert attrs["attribution_owner"] == "I0:indexer:index_k_norm"
    assert attrs["caller_status"] == "declared_selected_source_callsite"
    assert attrs["normalized_shape"] == [128]
    assert attrs["index_dim"] == 128
    assert attrs["eps"] == 1e-6
    assert attrs["dtype"] == "fp32"
    assert attrs["impl_mode"] is None
    assert attrs["native_acl_api"] == ACLNN_API
    assert attrs["native_kind"] == KERNEL_NAME
    assert len(attrs["source_variants"]) == 1
    assert len(attrs["candidate_callers"]) == 2
    assert spec.typed["occurrence_attrs"] == {
        "m": "positive_int", "phase": "string"}

    out = dynamic_call(binding, compiled, _geometry(1024),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["dtype"] == "fp32"
    assert out["actual"]["shapes"] == {
        "input": [1024, 128], "normalized_shape": [1], "gamma": [128],
        "beta": [128], "output": [1024, 128], "mean": [1024, 1],
        "rstd": [1024, 1]}
    prepare = out["prepare"]
    assert prepare["source_stage"] == STAGE
    assert prepare["source_order"] == SOURCE_ORDER
    assert prepare["arguments"] == {
        "input": "x", "normalized_shape": [128], "weight": "gamma",
        "bias": "beta", "eps": 1e-6}
    assert prepare["aclnn_arguments"]["epsilon"] == 1e-6
    assert prepare["aclnn_arguments"]["impl_mode"] is None
    assert prepare["work"] == _work(1024, 128, "fp32", "prefill")
    assert prepare["runtime_executed"] is False
    assert prepare["allocation"] == "not_performed"
    assert prepare["complete"] is False
    assert prepare["partial"] is True
    assert prepare["match"]["source_stage"] == STAGE
    assert prepare["match"]["source_path"] == model["path"]
    assert prepare["match"]["call_site_path"] == call_site["path"]
    assert prepare["match"]["layer_norm_source_path"] == (
        LAYER_NORM_SOURCE["path"])
    assert prepare["match"]["attribution_owner"] == "I0:indexer:index_k_norm"
    assert prepare["match"]["phase"] == "prefill"
    assert prepare["match"]["width"] == 128
    assert prepare["match"]["eps"] == 1e-6
    assert prepare["match"]["dtype"] == "fp32"
    assert prepare["match"]["impl_mode"] is None
    assert prepare["source_lowering"]["selected_caller"] == dict(call_site)
    assert prepare["source_lowering"]["selected_caller_status"] == (
        "declared_by_builder_source_slot")
    assert prepare["source_lowering"]["source_stage"] == STAGE
    assert prepare["source_lowering"]["status"] == (
        "source_selected_callsite_acl_wrapper_measured_not_source_declared")
    assert prepare["source_lowering"]["source_order"] == SOURCE_ORDER
    assert len(prepare["source_lowering"]["call_site_candidates"]) == 2
    assert prepare["acl_interface"]["status"] == "not_retained"
    assert prepare["acl_interface"]["impl_mode"] is None
    assert "acl_wrapper_and_dispatch_unresolved" in prepare["unresolved"]
    assert "layer_norm_kernel_source_not_retained" in prepare["unresolved"]
    assert prepare["non_claims"] == [
        "no_runtime_allocation", "no_device_launch", "no_measured_duration",
        "no_acl_wrapper_claim", "no_kernel_source_claim"]
    descriptors = prepare["descriptors"]
    assert descriptors["normalized_shape"]["values"] == [128]
    assert descriptors["input"]["dtype"] == "fp32"
    assert descriptors["mean"]["dtype"] == "fp32"
    assert descriptors["output"]["shape"] == [1024, 128]


def test_r64_i0_indexer_k_layer_norm_glm5next_variant():
    compiled = _compiled(extras=_extras(variant="glm5_next"))
    spec, binding = _binding(compiled)
    model = SOURCE_VARIANTS["glm5_next"]["model"]
    call_site = SOURCE_VARIANTS["glm5_next"]["call_site"]
    assert spec.source["path"] == model["path"]
    assert spec.source["sha256"] == model["sha256"]
    assert spec.typed["attrs"]["source_path"] == model["path"]
    assert spec.typed["attrs"]["call_site_path"] == call_site["path"]
    assert spec.typed["attrs"]["call_site_lines"] == list(call_site["lines"])
    assert spec.typed["attrs"]["attribution_owner"] == (
        "I0:indexer:index_k_norm")
    assert len(spec.typed["attrs"]["source_variants"]) == 5
    assert binding.timing_owner == {"op_id": "I0:elementwise:index_k_norm",
                                    "scope": "call"}
    out = dynamic_call(binding, compiled, _geometry(1024),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["shapes"]["input"] == [1024, 128]
    assert out["prepare"]["source_lowering"]["selected_caller"] == dict(call_site)
    assert out["prepare"]["source_stage"] == STAGE


def test_r64_i0_indexer_k_layer_norm_geometry_phase_and_binding_stability():
    compiled = _compiled()
    spec, binding = _binding(compiled)
    held = dynamic_call(binding, compiled, _geometry(2048),
                        validate=validate_native_occurrence)
    assert held is not None
    assert held["actual"]["shapes"]["input"] == [2048, 128]
    assert held["prepare"]["work"] == _work(2048, 128, "fp32", "prefill")
    decoded = dynamic_call(binding, compiled, _geometry(1, phase="decode"),
                           validate=validate_native_occurrence)
    assert decoded is not None
    assert decoded["actual"]["shapes"] == {
        "input": [1, 128], "normalized_shape": [1], "gamma": [128],
        "beta": [128], "output": [1, 128], "mean": [1, 1], "rstd": [1, 1]}
    assert decoded["prepare"]["match"]["phase"] == "decode"
    assert decoded["prepare"]["work"] == _work(1, 128, "fp32", "decode")
    assert binding.call_id == _binding(_compiled())[1].call_id
    assert spec.typed["shapes"]["input"] == [128]


def test_r64_i0_indexer_k_layer_norm_consumes_declared_width_and_eps():
    compiled = _compiled(elements=64, extras=_extras(64, 2e-5))
    spec, binding = _binding(compiled)
    assert spec.typed["shapes"]["input"] == [64]
    assert spec.typed["attrs"]["normalized_shape"] == [64]
    assert spec.typed["attrs"]["index_dim"] == 64
    assert spec.typed["attrs"]["eps"] == 2e-5
    out = dynamic_call(binding, compiled, _geometry(1024),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["shapes"] == {
        "input": [1024, 64], "normalized_shape": [1], "gamma": [64],
        "beta": [64], "output": [1024, 64], "mean": [1024, 1],
        "rstd": [1024, 1]}
    assert out["prepare"]["arguments"]["normalized_shape"] == [64]
    assert out["prepare"]["arguments"]["eps"] == 2e-5
    assert out["prepare"]["work"] == _work(1024, 64, "fp32", "prefill")


def test_r64_i0_indexer_k_layer_norm_consumes_resolved_dtype():
    compiled = _compiled(extras=_extras(dtype="bf16"))
    spec, binding = _binding(compiled)
    assert spec.typed["dtype"] == "bf16"
    out = dynamic_call(binding, compiled, _geometry(1024),
                       validate=validate_native_occurrence)
    assert out is not None
    assert out["actual"]["dtype"] == "bf16"
    descriptors = out["prepare"]["descriptors"]
    assert descriptors["input"]["dtype"] == "bf16"
    assert descriptors["gamma"]["dtype"] == "bf16"
    assert descriptors["output"]["dtype"] == "bf16"
    assert descriptors["mean"]["dtype"] == "fp32"
    assert out["prepare"]["match"]["dtype"] == "bf16"
    assert out["prepare"]["work"] == _work(1024, 128, "bf16", "prefill")


def test_r64_i0_indexer_k_layer_norm_rejects_i1_and_other_sources():
    assert binding_spec(_compiled(kind="I1"), phase="prefill") is None
    assert binding_spec(_compiled(image_id=R35_IMAGE_SHA),
                        phase="prefill") is None
    assert binding_spec(_compiled(kind="A1"), phase="prefill") is None
    assert binding_spec(_compiled(name="index_score"),
                        phase="prefill") is None
    assert binding_spec(_compiled(family="copy"), phase="prefill") is None
    assert binding_spec(_compiled(role="copy"), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(source_kind="RmsNorm")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(source_stage="I1")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        source_path="vllm/model_executor/models/other.py")),
        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        source_symbol="Other.__init__")), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(source_lines=[1, 2])),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(source_sha256="0" * 64)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        call_site_path="vllm_ascend/attention/other.py")),
        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        call_site_symbol="Other.forward")), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(call_site_lines=[1, 1])),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        call_site_sha256="0" * 64)), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        layer_norm_source_path="vllm/model_executor/layers/other.py")),
        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        layer_norm_source_symbol="OtherNorm")), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        layer_norm_source_lines=[1, 1])), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        layer_norm_source_sha256="0" * 64)), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        attribution_owner="I1:indexer:index_k_norm")),
        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(
        include=("source_kind", "source_stage", "index_dim", "norm_eps",
                 "normalized_shape", "norm_dtype"))),
        phase="prefill") is None
    assert binding_spec(_compiled(elements=64), phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(eps=0.0)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(eps=-1e-6)),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(dtype="int8")),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(normalized_shape=[64, 2])),
                        phase="prefill") is None
    assert binding_spec(_compiled(extras=_extras(index_dim=64)),
                        phase="prefill") is None
    assert binding_spec(_compiled(), phase="") is None
    assert binding_spec(_compiled(), phase="chunked_prefill") is None


def test_r64_i0_indexer_k_layer_norm_rejects_empty_geometry():
    compiled = _compiled()
    _, binding = _binding(compiled)
    assert dynamic_call(binding, compiled, _geometry(0),
                        validate=validate_native_occurrence) is None
