"""Explicit Ascend normalization ABI over existing semantic fusion passes.

[Role] Select native residual/RMS normalization, per-token INT8 and group-32
  MX quantization outputs, and optional affine beta.
[Invariants] Residual outputs, rstd stores, quant payload and scale metadata
  survive selection. A caller beta option is a declaration, never inference
  from a capture. Gemma's gamma+1 producer is external. INT8 is one FP32 scale
  per complete row; MX is one E8M0 scale per 32-value group with an even-group
  tail pad. A quantized selection must account for the source register ledger
  of its quantization phase instead of silently reusing a generic envelope.
  Resource envelopes preserve work without native tile claims.
[Timeouts & Locks] Pure bounded operator arithmetic; no waits or locks.
[Side-Effect Policy] Returns operators; no source, checkpoint or device access.
"""
from dataclasses import replace
from typing import Literal

from mindie_sim.catalog.ops import vector_op
from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef

from .mx_dynamic_quant import group32_work

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"

SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-nn",
        revision="238f76a06d0cfc5ee90071e2c08310e2b517c3a6",
        relative_path="norm/add_rms_norm_dynamic_quant/op_host/add_rms_norm_dynamic_quant_def.cpp",
        source_sha256="efabe02000c51b72722e15c05ee00847668f008c0c116b84d44176fc3b5b23a7",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="vllm_ascend/compilation/passes/norm_quant_fusion_pass.py",
        source_sha256="e17b2cb3afeb45283885bf807022e7608b70400e153983f15f8967bd76cb0748",
    ),
    SourceRef(
        repository="vllm-ascend",
        revision="175ef4d49c2072f9233619636f0309f4f88b4e2c",
        relative_path="csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h",
        source_sha256="5ad66e31f5cdea5c7c61ddf3bde84743deca97965381f049fd197a272a281eed",
    ),
)


ADD_SOURCE = {
    "repository": "vllm-ascend",
    "base_revision": "175ef4d49c2072f9233619636f0309f4f88b4e2c",
    "source_state": "clean_at_revision",
    "path": "csrc/moe/add_rms_norm_bias/op_kernel/add_rms_norm_bias.h",
    "sha256": "5ad66e31f5cdea5c7c61ddf3bde84743deca97965381f049fd197a272a281eed",
    "lines": [125, 218],
    "scope": "independent_source_semantics; not_capture_binary_identity",
}
# Three-input (x, residual, gamma) fused residual RMSNorm without a beta slot.
# The retained A5 Qwen3.6 capture names aclnnAddRmsNorm for the residual norms;
# the R31 libopapi record supplies the same symbol's declared tensor contract.
ADD_RMS_SOURCE = {
    "repository": "cann",
    "symbol": "aclnnAddRmsNorm",
    "source_state": "task_artifact",
    "kernel": "AddRmsNorm",
    "libopapi_path": "/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "libopapi_sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "capture": {
        "path": "docs/qwen36-tp1-profile-evidence.json",
        "op_type": "AddRmsNorm",
        "op_name": "aclnnAddRmsNorm_AddRmsNorm_AddRmsNorm",
    },
    "scope": "api_and_caller_semantics; not_capture_binary_identity; timing_not_used",
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
MX_QUANT_SOURCE = {
    "repository": "cann",
    "kernel": "DynamicMxQuantTailAxisFP8",
    "selected_key": 33,
    "source_state": "source_work_model",
    "scope": "group32_e4m3_e8m0_alg0_register_ledger; revision_from_caller",
}

_MX_GROUP = 32
_FP32_BYTES = 4
_QUANT_INPUT_DTYPES = frozenset({"bf16", "fp16"})
# Declared RMSNorm work with zero, one or two fused elementwise operations.
_NORM_OPS_PER_ELEMENT = (5, 6, 7)
_GENERIC_QUANT_OPS_PER_ELEMENT = 4


def recipes():
    # Selected work uses the common vector envelope with its ordinary recipe.
    return {}


def _with(returns, name):
    return returns if name in returns else (*returns, name)


def _mx_layout(width, rows):
    groups = (int(width) + _MX_GROUP - 1) // _MX_GROUP
    padded = (groups + 1) // 2 * 2
    return groups, padded, (int(rows), padded // 2, 2)


def _norm_vector_part(op, rows, width, *, quant_component):
    """Known semantic norm work around a fused quantizer, or None.

    The norm portion is left exactly as the producer declared; only the
    quantizer phase is replaced by its source ledger. An unknown composition
    is reported rather than silently rewritten.
    """
    elements = int(rows) * int(width)
    for count in _NORM_OPS_PER_ELEMENT:
        expected = count * elements
        if op.vector_ops - quant_component == expected:
            return expected
    return None


def _quant_component(op, rows, width, *, group_size, mx):
    """Declared vector work of the quantization phase before replacement."""
    elements = int(rows) * int(width)
    groups = (int(width) + int(group_size) - 1) // int(group_size)
    generic = _GENERIC_QUANT_OPS_PER_ELEMENT * elements + 2 * int(rows) * groups
    norm_work = {count * elements for count in _NORM_OPS_PER_ELEMENT}
    declared = dict(op.metadata).get("mx_vector_ops_per_row") if mx else None
    if declared is not None:
        try:
            declared = int(declared)
        except (TypeError, ValueError) as exc:
            raise ModelingError(
                "ascend_native_norm_mx_vector_ops_per_row_invalid") from exc
        if declared < 1:
            raise ModelingError("ascend_native_norm_mx_vector_ops_per_row_invalid")
        # The subgraph guard declares the producer as a one-op-per-element
        # quantize slot and keeps the true source ledger in metadata.
        if op.vector_ops - elements in norm_work:
            return elements, generic, declared, "declared_mx_vector_ops_per_row"
        if op.vector_ops - generic in norm_work:
            return generic, generic, declared, "declared_mx_vector_ops_per_row"
        return None, generic, declared, "unknown"
    if op.vector_ops - generic in norm_work:
        return generic, generic, None, "generic_activation_quantization"
    return None, generic, None, "unknown"


def _quant_contract(meta, family, rows, width, *, dtype, bias):
    """Validate the selected quantizer ABI and return its materialization."""
    if dtype not in _QUANT_INPUT_DTYPES:
        raise ModelingError(
            f"ascend_native_norm_quant_input_dtype_unsupported:{family}:{dtype}")
    gamma_dtype = str(meta.get("norm_gamma_runtime_format") or "").strip().lower()
    if gamma_dtype and gamma_dtype != dtype:
        raise ModelingError(
            f"ascend_native_norm_quant_gamma_dtype_mismatch:{family}:{dtype}:{gamma_dtype}")
    output = meta.get("output_format")
    elements = int(rows) * int(width)
    if family == "add_rmsnorm_quant" and output == "int8":
        if (meta.get("activation_group_size") != width
                or meta.get("activation_scale_format") != "fp32"
                or tuple(meta.get("activation_scale_shape", ())) != (rows,)
                or meta.get("activation_scale_bytes") != rows * _FP32_BYTES):
            raise ModelingError(
                "ascend_native_norm_int8_quant_requires_complete_row_fp32_scale")
        payload = rows * width
        scale = rows * _FP32_BYTES
        return {
            "kind": "AddRmsNormDynamicQuant",
            "family": "add_rmsnorm_quant",
            "quant_groups": 1,
            "quant_group_size": width,
            "payload_bytes": payload,
            "scale_bytes": scale,
            "scale_shape": (rows,),
            "scale_layout": "ND",
            "quant_component": _GENERIC_QUANT_OPS_PER_ELEMENT * elements + 2 * rows,
            "quant_vector_ops": None,
            "quant_work_source": "semantic_per_group_conversion",
            "source": QUANT_SOURCE,
        }
    if family in {"rmsnorm_quant", "add_rmsnorm_quant"} and output == "fp8_e4m3":
        # Both declared MX ABIs expose only x, residual-or-none and gamma.
        # A beta request is an unmodelled input: reject it instead of dropping
        # the affine term or relabelling the call. Bias belongs to the ordinary
        # AddRmsNormBias/INT8-declared beta paths, not to this MX contract.
        if bias:
            raise ModelingError(
                "ascend_native_norm_mx_quant_beta_not_in_declared_abi")
    elif family == "rmsnorm_quant" and output == "int8":
        raise ModelingError(
            "ascend_native_norm_plain_int8_quant_kernel_not_established")
    else:
        raise ModelingError("ascend_native_norm_quantization_domain_not_supported")
    if (meta.get("activation_group_size") != _MX_GROUP
            or meta.get("activation_scale_format") != "fp8_e8m0"):
        raise ModelingError("ascend_native_norm_mx_quant_requires_group32_e8m0")
    groups, padded, shape = _mx_layout(width, rows)
    if (tuple(meta.get("activation_scale_shape", ())) != shape
            or meta.get("activation_scale_bytes") != rows * padded):
        raise ModelingError(
            "ascend_native_norm_mx_quant_scale_materialization_mismatch")
    return {
        "kind": ("RmsNormDynamicMxQuant" if family == "rmsnorm_quant"
                 else "AddRmsNormDynamicMxQuant"),
        "family": family,
        "quant_groups": groups,
        "padded_scale_groups": padded,
        "quant_group_size": _MX_GROUP,
        "payload_bytes": rows * width,
        "scale_bytes": rows * padded,
        "scale_shape": shape,
        "scale_layout": "NCL",
        "quant_component": _GENERIC_QUANT_OPS_PER_ELEMENT * elements + 2 * rows * groups,
        "quant_vector_ops": None,
        "quant_work_source": "mx_dynamic_quant.group32_work:r64_alg0",
        "source": {**QUANT_SOURCE, "quant_kernel": MX_QUANT_SOURCE},
    }


def _native_vector_work(op, meta, qmeta, rows, width, extra_work):
    vector_ops = op.vector_ops
    quant_vector_ops = None
    norm_vector_ops = None
    quant_component = None
    work_source = "caller_declared"
    if qmeta is not None:
        prior_norm = meta.get("native_norm_vector_ops")
        prior_quant = meta.get("native_norm_quant_vector_ops")
        prior_extra = meta.get("native_norm_extra_vector_ops")
        already = (
            qmeta["kind"].endswith("MxQuant")
            and meta.get("native_norm_kind") == qmeta["kind"]
            and prior_norm is not None and prior_quant is not None)
        if already:
            if op.vector_ops == int(prior_norm) + int(prior_quant) + int(prior_extra or 0):
                vector_ops = op.vector_ops
                extra_work = 0
                norm_vector_ops = int(prior_norm)
                quant_vector_ops = int(prior_quant)
                quant_component = meta.get("native_norm_quant_component_ops")
                work_source = str(meta.get("native_norm_quant_work_source") or
                                  "mx_dynamic_quant.group32_work:r64_alg0")
            else:
                already = False
        if not already:
            component, generic, declared, basis = _quant_component(
                op, rows, width,
                group_size=(width if qmeta["kind"] == "AddRmsNormDynamicQuant"
                            else _MX_GROUP),
                mx=qmeta["kind"].endswith("MxQuant"))
            quant_component = component
            if component is not None:
                norm_vector_ops = _norm_vector_part(
                    op, rows, width, quant_component=component)
            if qmeta["kind"].endswith("MxQuant"):
                algorithm = meta.get("mx_algorithm")
                if algorithm not in {None, "", "ocp_alg0"}:
                    raise ModelingError(
                        "ascend_native_norm_mx_quant_algorithm_ledger_unsupported")
                work = group32_work(width, op.dtype)
                lane_ops = int(work["lane_ops"]["total"])
                per_row = declared if declared is not None else lane_ops
                if component is None:
                    raise ModelingError(
                        "ascend_native_norm_mx_quant_work_composition_unknown")
                vector_ops = op.vector_ops - component + rows * per_row
                quant_vector_ops = rows * per_row
                work_source = (basis if declared is not None else
                               "mx_dynamic_quant.group32_work:r64_alg0")
            elif qmeta["kind"] == "AddRmsNormDynamicQuant":
                if component is not None:
                    quant_vector_ops = component
                    work_source = "semantic_per_group_conversion"
                else:
                    work_source = "caller_declared_unclassified"
    return vector_ops, extra_work, norm_vector_ops, quant_vector_ops, quant_component, work_source


def _update_native_norm_meta(
    meta, op, qmeta, *, bias, shift, kind, source, stats_bytes, has_bias,
    norm_vector_ops, quant_vector_ops, quant_component, work_source,
    residual_bytes, residual_materialized, residual_covered, normalized_bytes,
    extra_read, extra_record, fp32_working_tiles,
):
    meta.update({
        "norm_has_bias": bias, "norm_weight_offset": 0.0,
        "norm_caller_weight_offset": shift, "native_norm_kind": kind,
        "native_norm_fp32_working_tiles": int(fp32_working_tiles),
        "native_source": source,
        "native_implementation": {
            "native_kinds": (kind,), "selection": "explicit_caller_implementation",
            "basis": "source_semantics_and_declared_fusion",
            "conditions": {"beta": bias, "gamma_shift_external": bool(shift),
                           "cost_model": "semantic_resource_envelope", "api": kind},
        },
        "norm_auxiliary_output_bytes": stats_bytes,
        "norm_auxiliary_scope": (
            "materialized_fp32_rstd" if stats_bytes else
            "active_output_residual_and_scales; unused_native_auxiliary_stores_not_established"),
        "norm_parameter_basis": ("explicit_caller_bias_option" if has_bias is not None
                                 else "declared_normalization_semantics"),
        "native_norm_vector_ops": norm_vector_ops,
        "native_norm_quant_vector_ops": quant_vector_ops,
        "native_norm_quant_component_ops": quant_component,
        "native_norm_quant_work_source": work_source,
        "native_norm_residual_bytes": residual_bytes,
        "native_norm_residual_format": op.dtype if residual_materialized else None,
        "native_norm_residual_output": bool(residual_materialized),
        "native_norm_residual_billing": (
            "included_in_mte3" if residual_materialized and residual_covered
            else "declared_output_not_covered_by_producer_mte3"
            if residual_materialized else "absent"),
        "native_norm_store_consistent": bool((not residual_materialized) or residual_covered),
        "native_norm_normalized_output_bytes": normalized_bytes,
        "native_norm_normalized_format": meta.get("output_format") or op.dtype,
        "native_norm_rstd_bytes": stats_bytes,
        "native_norm_rstd_format": "fp32" if stats_bytes else None,
        "native_norm_extra_read_bytes": extra_read,
        "native_norm_extra_vector_ops": extra_record,
    })
    if qmeta is not None:
        meta.update({
            "native_norm_quant_payload_bytes": qmeta["payload_bytes"],
            "native_norm_quant_scale_bytes": qmeta["scale_bytes"],
            "native_norm_quant_scale_format": meta.get("activation_scale_format"),
            "native_norm_quant_scale_shape": qmeta["scale_shape"],
            "native_norm_quant_scale_layout": qmeta["scale_layout"],
            "native_norm_quant_group_size": qmeta["quant_group_size"],
            "native_norm_quant_groups": qmeta["quant_groups"],
            "native_norm_quant_padded_groups": qmeta.get("padded_scale_groups", qmeta["quant_groups"]),
            "native_norm_quant_source": qmeta["source"],
        })


def lower_op(op, *, has_bias=None, external_gamma_shift=False):
    meta = dict(op.metadata)
    if (op.dtype not in {"bf16", "fp16", "fp32"} or len(op.local_shape) != 2
            or meta.get("norm_has_weight") is not True):
        raise ModelingError("ascend_native_norm_requires_weighted_row_norm")
    rows, width = op.local_shape
    if rows < 1 or width < 1:
        raise ModelingError("ascend_native_norm_requires_positive_row_geometry")
    shift = meta.get("norm_weight_offset", 0.0)
    if shift and (shift != 1.0 or not external_gamma_shift):
        raise ModelingError("ascend_native_norm_requires_explicit_external_gamma_shift")
    if external_gamma_shift and shift != 1.0:
        raise ModelingError("ascend_native_norm_gamma_shift_requires_declared_gemma_weight")
    bias_before = bool(meta.get("norm_has_bias", False))
    bias = bias_before if has_bias is None else has_bias
    if not isinstance(bias, bool) or (bias_before and not bias):
        raise ModelingError("ascend_native_norm_bias_cannot_remove_declared_affine")
    qmeta = _quant_contract(meta, op.family, rows, width, dtype=op.dtype, bias=bias) \
        if op.family in {"rmsnorm_quant", "add_rmsnorm_quant"} else None
    declared_api = meta.get("norm_native_api")
    if op.family == "add_rmsnorm":
        if declared_api is None:
            gemma_caller = str(meta.get("rms_norm_source_class") or "") == \
                "AscendGemmaRMSNorm"
            kind = "AddRmsNorm" if gemma_caller and not bias else "AddRmsNormBias"
        elif declared_api in {"AddRmsNorm", "AddRmsNormBias"}:
            kind = str(declared_api)
        else:
            raise ModelingError("ascend_native_norm_native_api_unknown")
        if kind == "AddRmsNorm" and bias:
            raise ModelingError("ascend_native_add_rms_norm_has_no_beta_slot")
        stats_bytes = rows * _FP32_BYTES
        source = ADD_RMS_SOURCE if kind == "AddRmsNorm" else ADD_SOURCE
    else:
        if declared_api is not None:
            raise ModelingError(
                "ascend_native_norm_native_api_only_for_unquantized_add")
        kind = qmeta["kind"]
        stats_bytes = 0
        source = qmeta["source"]
    extra_read = width * dtype_bytes(op.dtype) if bias and not bias_before else 0
    extra_work = rows * width if bias and not bias_before else 0
    extra_record = max(int(meta.get("native_norm_extra_vector_ops") or 0), extra_work)

    (
        vector_ops, extra_work, norm_vector_ops, quant_vector_ops, quant_component, work_source,
    ) = _native_vector_work(op, meta, qmeta, rows, width, extra_work)

    read = op.mte2_bytes + extra_read
    # A second direct lowering of the same selected record must not store rstd
    # twice; production also guards this with implementation_binding, but the
    # family keeps the public call idempotent on its own.
    prior_aux = meta.get("norm_auxiliary_output_bytes")
    prior_rstd = meta.get("native_norm_rstd_bytes")
    stats_already = bool(stats_bytes) and prior_aux == stats_bytes \
        and prior_rstd == stats_bytes
    write = op.mte3_bytes + (0 if stats_already else stats_bytes)
    expected_store = (qmeta["payload_bytes"] + qmeta["scale_bytes"]
                      if qmeta is not None else 0)
    add_family = op.family == "add_rmsnorm" or (
        qmeta is not None and qmeta["family"] == "add_rmsnorm_quant")
    if add_family and qmeta is not None:
        expected_store += rows * width * dtype_bytes(op.dtype)
    if qmeta is not None and op.mte3_bytes + stats_bytes < expected_store:
        raise ModelingError("ascend_native_norm_quant_store_below_declared_outputs")
    normalized_bytes = (qmeta["payload_bytes"] if qmeta is not None
                        else rows * width * dtype_bytes(op.dtype))
    declared_residual = (rows * width * dtype_bytes(op.dtype)
                         if add_family else 0)
    # The producer's mte3 is the single store billing.  An empty return list is
    # the ordinary semantic fused norm before its external outputs are named;
    # only add the residual when that mte3 actually covers both stores.  A
    # non-empty list without "residual" is an explicit absence (for example an
    # O0 final norm): do not invent that output or a virtual byte.
    residual_covered = (not add_family) or (
        op.mte3_bytes >= declared_residual + normalized_bytes)
    residual_materialized = add_family and (
        qmeta is not None or "residual" in op.returns
        or (not op.returns and residual_covered))
    residual_bytes = declared_residual if residual_materialized else 0
    returns = op.returns
    if qmeta is None:
        returns = _with(returns, "rstd")
        if residual_materialized:
            returns = _with(returns, "residual")
    else:
        returns = _with(returns, "activation_scales")
        if add_family:
            returns = _with(returns, "residual")

    fp32_working_tiles = 2 if (add_family and op.dtype in {"bf16", "fp16"}) else 1
    _update_native_norm_meta(
        meta, op, qmeta, bias=bias, shift=shift, kind=kind, source=source,
        stats_bytes=stats_bytes, has_bias=has_bias, norm_vector_ops=norm_vector_ops,
        quant_vector_ops=quant_vector_ops, quant_component=quant_component,
        work_source=work_source, residual_bytes=residual_bytes,
        residual_materialized=residual_materialized, residual_covered=residual_covered,
        normalized_bytes=normalized_bytes, extra_read=extra_read,
        extra_record=extra_record, fp32_working_tiles=fp32_working_tiles)
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
        op, deps=deps, vector_ops=vector_ops + extra_work,
        mte2_bytes=read, mte3_bytes=write, hbm_bytes=read + write,
        returns=returns, metadata=tuple(meta.items()),
    )
    return (*prefix, selected)
