"""Prepare-only R31 A0 output-gate sigmoid invocation.

[Role] Expand the real R31 Qwen A0 ``output_gate`` row in both source
  phases into the actual ACLNN sigmoid call.  The public Python operation is
  ``torch.sigmoid(input, *, out=None)``; the retained R31 ``libopapi.so``
  exports ``aclnnSigmoid`` / ``aclnnSigmoidGetWorkspaceSize`` and the original
  captures name the launched kernel
  ``aclnnSigmoid_SigmoidAiCore_Sigmoid`` with task type ``Sigmoid``.  The same
  R31 image has authenticated prefill [4096,3072] and decode [2,3072] rows;
  the decode capture contains 127 steps x 16 A0 calls (2032 total).
[Invariants] Only the exact R31 image id and the compiled A0
  ``output_gate``/``gate`` row with BF16 byte geometry admit this call; there
  is no prefill-only or decode-only exclusion.  The physical row count is
  the existing role-specific ``geometry.effective_tokens`` already projected
  once by central FEATURES8/``CompiledOperator.evaluate``; this leaf adds no
  padding, 2x row multiplier, sample id, model literal, or hardcoded M.  The
  static gate width comes only from the compiled row.  The exact Python model
  wrapper that owns ``g_proj`` and calls ``torch.sigmoid`` is not retained, the
  ACLNN parameter header and the ``SigmoidAiCore`` binary/source are not
  retained, and the downstream in-place ``aclnnMul`` is a separate physical
  call.  Those facts stay explicit; no tensor, pointer, workspace size, kernel
  binding or complete native execution is fabricated.  The retained
  ``vllm_ascend/attention/sfa_v1.py:2164`` sigmoid is deliberately *not* used
  as attribution: that is the sparse SFA/MLA backend, while the captured
  attention operator is dense ``FusedInferAttentionScore`` GQA.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing R31 image
  identity, compiled A0 gate geometry and original captured operands.  No
  tensor allocation, device launch, runtime import or timing selection.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_VERSION = "cann-9.2.0-r31"
ENTRY = "torch.sigmoid"
ACLNN_API = "aclnnSigmoid"
ACLNN_WORKSPACE_API = "aclnnSigmoidGetWorkspaceSize"
KERNEL_NAME = "Sigmoid"
KERNEL_LABEL = "SigmoidAiCore"
PROFILER_NAME = "aclnnSigmoid_SigmoidAiCore_Sigmoid"
# Original-capture evidence only.  These values never derive the occurrence
# rows or gate width; the occurrence consumes geometry.effective_tokens and
# the compiled row's static gate width.
EVIDENCE_GATE_WIDTH = 3072
EVIDENCE_PREFILL_TOKENS = 4096
EVIDENCE_PREFILL_CALLS = 16
EVIDENCE_BLOCK_NUM = 64
EVIDENCE_TASK_TYPE = "AI_VECTOR_CORE"
EVIDENCE_DECODE_TOKENS = 2
EVIDENCE_DECODE_CALLS = 2032
EVIDENCE_DECODE_STEPS = 127
EVIDENCE_DECODE_BLOCK_NUM = 3

LIBOPAPI = {
    "repository": "cann",
    "path": ("usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/"
             "libopapi.so"),
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": ACLNN_API,
    "revision": R31_SOURCE_VERSION,
    "state": "task_artifact",
}
CAPTURE = {
    "path": ("/Volumes/data/mindie-agent-workstreams/"
             "a5-multinode-fleet-rack06-20261107/out/smoke_qwen36_single/"
             "pf_4k_c1/177.25.28.1/"
             "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz"),
    "sha256": "a523936e982e4fd45b80071e906276d17d18fac03c5bfaca903e2f1158f778d1",
    "retained_orig_sha": "2476b865",
    "observed_name": PROFILER_NAME,
    "observed_input_shape": [EVIDENCE_PREFILL_TOKENS, EVIDENCE_GATE_WIDTH],
    "observed_instance": "smoke_qwen36_single_pf_4k_c1",
    "observed_count": EVIDENCE_PREFILL_CALLS,
}
DC_CAPTURE = {
    "path": ("/Volumes/data/mindie-agent-workstreams/"
             "a5-multinode-fleet-rack06-20261107/out/smoke_qwen36_single/"
             "dc_4k_c1/177.25.28.1/"
             "rank1_694_20261107032749008_ascend_pt/kernel_details.csv.gz"),
    "sha256": "ab243b7d8d83e7b3a22404e50e1946b4b2b3bcc3986540e2056d1235180c8296",
    "retained_orig_sha": "ab243b7d",
    "observed_name": PROFILER_NAME,
    "observed_input_shape": [EVIDENCE_DECODE_TOKENS, EVIDENCE_GATE_WIDTH],
    "observed_instance": "smoke_qwen36_single_dc_4k_c1",
    "observed_count": EVIDENCE_DECODE_CALLS,
    "observed_steps": EVIDENCE_DECODE_STEPS,
    "observed_block_num": EVIDENCE_DECODE_BLOCK_NUM,
}
SFA_CANDIDATE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/attention/sfa_v1.py",
    "sha256": "54357ea7189e849ec0aa62a7c4b6d80fe58f849047e111138be04c2981246aa7",
    "line": 2164,
    "verdict": "rejected_different_deployed_backend",
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _r31_metadata(compiled: Any) -> dict[str, Any]:
    """The exact witnessed R31 deployment image only."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    from mindie_sim.prediction.mx_source_provenance import image_digest
    if not image_digest(image_id).startswith(R31_IMAGE_PREFIX):
        return {}
    return meta


def _gate_width(compiled: Any) -> int | None:
    """Static gate width from the compiled A0 row; never from a kernel name."""
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "name", "") or "") != "output_gate":
        return None
    if str(getattr(operator, "family", "") or "") != "elementwise":
        return None
    if str(getattr(getattr(operator, "template", None), "role", "") or "") != "gate":
        return None
    theta = operator.theta
    width = _int(theta.find("elements", 0))
    if width < 1:
        return None
    dtype_bytes = 2
    read = float(theta.find("read_bytes", 0) or 0)
    write = float(theta.find("write_bytes", 0) or 0)
    if read != float(width * dtype_bytes) or write != float(width * dtype_bytes):
        return None
    return width


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 A0 output-gate sigmoid call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  Every other image, operator or non-BF16 gate geometry returns
    ``None`` so ordinary prediction is unchanged; both prefill and decode are
    admitted because the source A0 gate executes in both phases.
    """
    del phase
    if compiled is None or str(getattr(compiled, "kind", "") or "") != "A0":
        return None
    meta = _r31_metadata(compiled)
    if not meta:
        return None
    width = _gate_width(compiled)
    if width is None:
        return None
    operator = compiled.operator
    theta = operator.theta
    from . import native_calls
    source = native_calls._source(
        LIBOPAPI["repository"], LIBOPAPI["path"],
        sha256=LIBOPAPI["sha256"], symbol=LIBOPAPI["symbol"],
        revision=LIBOPAPI["revision"], state=LIBOPAPI["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", path=LIBOPAPI["path"],
        sha256=LIBOPAPI["sha256"], evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_label=KERNEL_LABEL,
        source_status="kernel_binary_not_retained",
        owner_binary_path=LIBOPAPI["path"],
        owner_binary_sha256=LIBOPAPI["sha256"])
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "both",
        "shapes": {"input": [width], "output": [width]},
        "attrs": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "acl_workspace_api": ACLNN_WORKSPACE_API,
            "kernel_name": KERNEL_NAME,
            "kernel_label": KERNEL_LABEL,
            "profiler_name": PROFILER_NAME,
            "gate_width": width,
            "gate_elements": width,
            "dtype_bytes": 2,
            "read_bytes": float(width * 2),
            "write_bytes": float(width * 2),
            "operator_name": "output_gate",
            "operator_family": "elementwise",
            "operator_recipe": "gate",
            "attribution_scope": "attn_output_gate",
            "deployment_image_id": str(meta.get("image_id") or R31_IMAGE_SHA),
            "deployment_source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "phase_scope": "prefill+decode",
            "analytical_act_bytes": float(
                dict(theta.quant).get("act_bytes", 0.0) or 0.0),
            "analytical_ops_per_element": _int(
                theta.find("ops_per_element", 0)),
            "caller_status": "unresolved",
            "g_proj_projection_status": "unresolved",
            "acl_signature_status": (
                "symbols_retained_parameter_header_not_retained"),
            "kernel_source_status": "kernel_binary_not_retained",
            "attribution": "A0:elementwise:output_gate",
            "downstream_mul_status": (
                "separate_aclnnMul_not_this_call"),
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=R31_SOURCE_VERSION),
        source, typed, (kernel,), (), "output_gate",
        execution_core="aiv",
        unresolved=(
            "python_caller_source_unresolved",
            "g_proj_projection_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
            "aclnn_signature_not_retained",
            "attributable_source_reference_unresolved",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 A0 output-gate sigmoid occurrence.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is the A0 output-gate
    ``aclnnSigmoid`` call itself; the downstream in-place ``aclnnMul`` is not
    prepared or charged here.
    """
    del compiled
    api = binding.api or {}
    if (api.get("entry") != ENTRY
            and api.get("aclnn_symbol") != ACLNN_API):
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("native_type") != KERNEL_NAME
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("deployment_source_revision") != R31_REVISION):
        return None
    from mindie_sim.prediction.mx_source_provenance import image_digest
    image = str(attrs.get("deployment_image_id") or "")
    if not image_digest(image).startswith(R31_IMAGE_PREFIX):
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    width = _int(attrs.get("gate_width"))
    if width < 1:
        return None
    # The central FEATURES8 projection has already selected the source role's
    # physical dense row count; this leaf never pads or multiplies rows.
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if tokens < 1:
        return None
    cache_key = (binding.call_id, tokens, width, phase)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "input": [tokens, width],
        "output": [tokens, width],
    }
    actual_attrs = {
        **attrs,
        "tokens": tokens,
        "m": tokens,
        "rows": tokens,
        "input_rows": tokens,
        "output_rows": tokens,
        "input_shape": [tokens, width],
        "output_shape": [tokens, width],
        "runtime_phase": phase,
    }
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "input": _descriptor([tokens, width], "bf16", "ND", "input"),
        "output": _descriptor([tokens, width], "bf16", "ND", "output"),
    }
    descriptors["output"].update({
        "allocation_owner": "torch.sigmoid_return",
    })
    arguments = {"input": "input", "out": None}
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {
        "name": KERNEL_NAME, "role": "vector", "evidence": "measured"}
    kernel.update({
        "acl_api": ACLNN_API,
        "profiler_name": PROFILER_NAME,
        "kernel_label": KERNEL_LABEL,
        "source_status": "kernel_binary_not_retained",
        "owner_binary_path": LIBOPAPI["path"],
        "owner_binary_sha256": LIBOPAPI["sha256"],
    })
    prepare = {
        "schema": "mindie-sim/sigmoid-native-prepare/1",
        "call": ENTRY,
        "native_entry": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "api": {
            "namespace": "torch",
            "entry": ENTRY,
            "aclnn_symbol": ACLNN_API,
            "version": R31_SOURCE_VERSION,
        },
        "source": source,
        "source_version": R31_SOURCE_VERSION,
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "prototype": "torch.sigmoid(input, *, out=None) -> Tensor",
        "arguments": arguments,
        "argument_order": ["input", "out"],
        "positional_arguments": {"input": "input"},
        "keywords": {"out": None},
        "optional_slots": {"out": "not_passed_default"},
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "return_contract": {
            "shape": "same_as_input",
            "dtype": "same_as_input",
            "layout": "ND",
        },
        "acl_interface": {
            "namespace": "aclnn",
            "symbol": ACLNN_API,
            "workspace_symbol": ACLNN_WORKSPACE_API,
            "symbol_evidence": (
                "R31 libopapi.so exported symbol plus original profiler "
                "row name"),
            "parameter_signature_status": "not_retained",
            "public_torch_prototype": (
                "torch.sigmoid(input, *, out=None) -> Tensor"),
        },
        "observed": {
            "prepared": {
                "phase": phase or "unknown",
                "input_shape": list(actual_shapes["input"]),
                "output_shape": list(actual_shapes["output"]),
                "input_dtype": "DT_BF16",
                "input_format": "ND",
                "output_dtype": "DT_BF16",
                "output_format": "ND",
            },
            "name": PROFILER_NAME,
            "native_type": KERNEL_NAME,
            "kernel_label": KERNEL_LABEL,
            "prefill": {
                "capture": CAPTURE,
                "input_shape": [
                    EVIDENCE_PREFILL_TOKENS, EVIDENCE_GATE_WIDTH],
                "input_dtype": "DT_BF16",
                "input_format": "ND",
                "output_shape": [
                    EVIDENCE_PREFILL_TOKENS, EVIDENCE_GATE_WIDTH],
                "output_dtype": "DT_BF16",
                "output_format": "ND",
                "task_type": EVIDENCE_TASK_TYPE,
                "block_num": EVIDENCE_BLOCK_NUM,
                "count": EVIDENCE_PREFILL_CALLS,
                "device_id": 4,
            },
            "decode": {
                "capture": DC_CAPTURE,
                "input_shape": [
                    EVIDENCE_DECODE_TOKENS, EVIDENCE_GATE_WIDTH],
                "input_dtype": "DT_BF16",
                "input_format": "ND",
                "output_shape": [
                    EVIDENCE_DECODE_TOKENS, EVIDENCE_GATE_WIDTH],
                "output_dtype": "DT_BF16",
                "output_format": "ND",
                "task_type": EVIDENCE_TASK_TYPE,
                "block_num": EVIDENCE_DECODE_BLOCK_NUM,
                "count": EVIDENCE_DECODE_CALLS,
                "steps": EVIDENCE_DECODE_STEPS,
            },
        },
        "caller": {
            "status": "unresolved",
            "path": None,
            "sha256": None,
            "symbol": None,
            "missing": (
                "exact Python model wrapper that owns g_proj and calls "
                "torch.sigmoid on the A0 GQA output"),
            "retained_candidate_checked": dict(SFA_CANDIDATE),
            "retrieval_candidates_not_proven": [
                "vllm/model_executor/models/qwen3_next.py",
                "vllm_ascend/models/qwen3_next.py",
            ],
        },
        "gate_projection": {
            "status": "unresolved",
            "source": None,
            "note": (
                "the g_proj producer is not retained; the captured sigmoid "
                "operand shape/dtype is direct evidence, the projection "
                "binding is not claimed here"),
        },
        "attribution": {
            "timing_owner": "A0:elementwise:output_gate",
            "physical_call": ACLNN_API,
            "gate_projection": (
                "separate g_proj producer; only its output is the sigmoid "
                "operand"),
            "downstream_inplace_mul": (
                "separate aclnnMul_MulAiCore_Mul; not prepared, attributed "
                "or charged by this adapter"),
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R31_IMAGE_SHA),
            "revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "selection": (
                "exact R31 image id + compiled A0 output_gate/gate row "
                "with BF16 byte geometry; authenticated prefill [4096,3072] "
                "and decode [2,3072] examples"),
            "source_path": LIBOPAPI["path"],
            "source_sha256": LIBOPAPI["sha256"],
            "source_symbol": ACLNN_API,
            "caller_path": None,
            "caller_sha256": None,
            "caller_status": "unresolved",
            "kernel_source_path": None,
            "kernel_source_sha256": None,
            "kernel_source_status": "kernel_binary_not_retained",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R31_IMAGE_SHA),
            "source_revision": R31_REVISION,
            "observed_model_identifier": str(
                attrs.get("observed_model_identifier") or ""),
            "gate_width": width,
            "dtype": "bf16",
            "layout": "ND",
            "phase": phase,
            "prepared_input_shape": [tokens, width],
            "source_sha256": LIBOPAPI["sha256"],
            "capture_sha256": CAPTURE["sha256"],
        },
        "binary_selection": {
            "status": "runtime_binary_only",
            "reason": (
                "aclnnSigmoid host API and SigmoidAiCore profiler label are "
                "retained; exact kernel binary/source and Python caller are "
                "not retained"),
            "owner_kernel": KERNEL_NAME,
            "kernel_label": KERNEL_LABEL,
            "acl_symbol": ACLNN_API,
            "runtime_library": LIBOPAPI["path"],
            "runtime_library_sha256": LIBOPAPI["sha256"],
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "python_caller_source_unresolved",
            "g_proj_projection_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
        ],
        "unresolved": [
            "python_caller_source_unresolved",
            "g_proj_projection_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
            "aclnn_signature_not_retained",
            "attributable_source_reference_unresolved",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_physical_completion",
            "no_kernel_source_claim",
            "no_caller_attribution",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": "bf16",
            "layout": "ND",
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


def observed_call() -> dict:
    """The original R31 sigmoid call facts, retained for reconciliation."""
    return {
        "capture": dict(CAPTURE),
        "name": PROFILER_NAME,
        "input_shape": [EVIDENCE_PREFILL_TOKENS, EVIDENCE_GATE_WIDTH],
        "input_dtype": "DT_BF16",
        "input_format": "ND",
        "output_shape": [EVIDENCE_PREFILL_TOKENS, EVIDENCE_GATE_WIDTH],
        "output_dtype": "DT_BF16",
        "output_format": "ND",
        "task_type": EVIDENCE_TASK_TYPE,
        "block_num": EVIDENCE_BLOCK_NUM,
        "count": EVIDENCE_PREFILL_CALLS,
        "decode_capture": dict(DC_CAPTURE),
    }
