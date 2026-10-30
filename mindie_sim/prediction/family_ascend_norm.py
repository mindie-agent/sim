"""Explicit Ascend normalization ABI over existing semantic fusion passes.

[Role] Select native residual/RMS quantization outputs and optional affine beta.
[Invariants] Residual outputs and rstd stores survive fusion. Gemma's gamma+1
  producer is external. A caller beta option is a declaration, never inference
  from a capture. Resource envelopes preserve work without native tile claims.
[Timeouts & Locks] Pure bounded operator arithmetic; no waits or locks.
[Side-Effect Policy] Returns operators; no source, checkpoint or device access.
"""
from dataclasses import replace

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes


ADD_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "clean_at_revision",
    "path": "csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h",
    "sha256": "5ad66e31f5cdea5c7c61ddf3bde84743deca97965381f049fd197a272a281eed",
    "lines": [125, 218],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
QUANT_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "clean_at_revision",
    "path": "vllm_ascend/compilation/passes/norm_quant_fusion_pass.py",
    "sha256": "e17b2cb3afeb45283885bf807022e7608b70400e153983f15f8967bd76cb0748",
    "lines": [152, 334],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
GAMMA_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "clean_at_revision",
    "path": "vllm_ascend/ops/layernorm.py",
    "sha256": "3782d77a4d1a5493121ee51c676b455304ccfdcd1abccbe58b45f5d4ad861a1e",
    "lines": [98, 102],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}


def recipes():
    # Selected work uses the common vector envelope with its ordinary recipe.
    return {}


def lower_op(op, *, has_bias=None, external_gamma_shift=False):
    meta = dict(op.metadata)
    if (op.dtype not in {"bf16", "fp16", "fp32"} or len(op.local_shape) != 2
            or meta.get("norm_has_weight") is not True):
        raise ModelingError("ascend_native_norm_requires_weighted_row_norm")
    rows, width = op.local_shape
    shift = meta.get("norm_weight_offset", 0.0)
    if shift and (shift != 1.0 or not external_gamma_shift):
        raise ModelingError("ascend_native_norm_requires_explicit_external_gamma_shift")
    if external_gamma_shift and shift != 1.0:
        raise ModelingError("ascend_native_norm_gamma_shift_requires_declared_gemma_weight")
    bias_before = bool(meta.get("norm_has_bias", False))
    bias = bias_before if has_bias is None else has_bias
    if not isinstance(bias, bool) or (bias_before and not bias):
        raise ModelingError("ascend_native_norm_bias_cannot_remove_declared_affine")
    output = meta.get("output_format")
    if op.family == "add_rmsnorm":
        kind, stats_bytes = "AddRmsNormBias", rows * 4
    elif (op.family == "add_rmsnorm_quant" and output == "int8"
          and meta.get("activation_scale_format") == "fp32"
          and tuple(meta.get("activation_scale_shape", ())) == (rows,)):
        kind, stats_bytes = "AddRmsNormDynamicQuant", 0
    elif (op.family in {"rmsnorm_quant", "add_rmsnorm_quant"} and output == "fp8_e4m3"
          and meta.get("activation_scale_format") == "fp8_e8m0"
          and width % 64 == 0
          and tuple(meta.get("activation_scale_shape", ())) == (rows, width // 64, 2)
          and not bias):
        kind = ("RmsNormDynamicMxQuant" if op.family == "rmsnorm_quant"
                else "AddRmsNormDynamicMxQuant")
        stats_bytes = 0
    else:
        raise ModelingError("ascend_native_norm_quantization_domain_not_supported")
    extra_read = width * dtype_bytes(op.dtype) if bias and not bias_before else 0
    extra_work = rows * width if bias and not bias_before else 0
    read, write = op.mte2_bytes + extra_read, op.mte3_bytes + stats_bytes
    meta.update({
        "norm_has_bias": bias, "norm_weight_offset": 0.0,
        "norm_caller_weight_offset": shift, "native_norm_kind": kind,
        "native_source": ADD_SOURCE if kind == "AddRmsNormBias" else QUANT_SOURCE,
        "native_implementation": {
            "native_kinds": (kind,), "selection": "explicit_caller_implementation",
            "basis": "source_semantics_and_declared_fusion",
            "conditions": {"beta": bias, "gamma_shift_external": bool(shift),
                           "cost_model": "semantic_resource_envelope"},
        },
        "norm_auxiliary_output_bytes": stats_bytes,
        "norm_auxiliary_scope": (
            "materialized_fp32_rstd" if stats_bytes else
            "active_output_residual_and_scales; unused_native_auxiliary_stores_not_established"),
        "norm_parameter_basis": "explicit_caller_bias_option" if has_bias is not None
                                else "declared_normalization_semantics",
    })
    deps, prefix = op.deps, []
    if shift:
        gamma = vector_op(
            op_id=op.op_id + ".gamma_shift", deps=(), stage=op.stage, layer=op.layer,
            requests=op.requests, elements=width, ops_per_element=1, dtype=op.dtype,
            family="elementwise", phase=op.phase, recipe="gemma_gamma_shift", shape=(width,),
            read_bytes_override=width * dtype_bytes(op.dtype),
            write_bytes_override=width * dtype_bytes(op.dtype),
            metadata=(("weight_offset", shift), ("native_source", GAMMA_SOURCE),
                      *((key, meta[key]) for key in ("structure_block", "structure_role")
                        if key in meta)),
        )
        prefix.append(replace(gamma, stream=op.stream))
        deps = (*deps, gamma.op_id)
    selected = replace(
        op, deps=deps, vector_ops=op.vector_ops + extra_work,
        mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
        returns=(*op.returns, *(("rstd",) if stats_bytes else ())),
        metadata=tuple(meta.items()),
    )
    return (*prefix, selected)
