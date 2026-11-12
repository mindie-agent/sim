"""Prepare-only R31 A3 ``aclnnInplaceZero`` core-output initialization.

[Role] Expand the real R31 source allocation
  ``core_attn_out = torch.zeros((num_tokens, num_v_heads // tp_size,
  head_v_dim), dtype=hidden_states.dtype, device=hidden_states.device)`` in
  ``vllm_ascend/ops/gdn.py`` into the exact ACLNN in-place ``ZerosLike``
  invocation.  The outer allocation at lines 422-428 happens before the
  prefill/decode branch, so it is admitted for both ordinary phases.  The
  separate ``_forward_core`` valid-only truncation at lines 487-491 is not
  this call and is deliberately not claimed.
[Invariants] The source identity is the exact R31 image digest
  ``9e869e814846...`` plus the retained ``gdn.py`` blob
  ``5e3b0fb0...`` and its line-422 allocation / line-423 shape / line-428
  consumer.  Only the A3 ``core_zeros`` fill row is selected; kept value
  heads and head width stay in the binding and the forward token count lives
  only in the occurrence.  The one operand is the freshly allocated
  ``core_attn_out`` buffer, so the ACLNN in-place zero has no second tensor
  or scalar input.  No tensor is allocated, no pointer or symbol is
  fabricated, no device launch is claimed and no measurement is used to
  price or select work.  The occurrence token rows are the shared FEATURES8
  ``WorkloadGeometry.operator_layout`` role-projected lane consumed as
  ``geometry.effective_tokens``; this leaf performs no local padding,
  TP round-up, request cloning or byte/cost accounting.  The ACLNN symbols
  are proven in the retained R31 ``libopapi.so``; the ATen lowering, ACLNN
  header/argument order and ``ZerosLike`` kernel source are not retained and
  stay explicit.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected source and the
  shared operator-layout geometry; other images, unknown phases and other
  rows stay unbound.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e814846"
R31_SOURCE_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
ENTRY = "torch.zeros"
ACLNN_API = "aclnnInplaceZero"
ACLNN_SYMBOLS = ("aclnnInplaceZeroGetWorkspaceSize", "aclnnInplaceZero")
KERNEL_NAME = "ZerosLike"
PROFILER_NAME = "aclnnInplaceZero_ZerosLikeAiCore_ZerosLike"
ROLE = "core_attn_out_accumulator"
LIBOPAPI_PATH = (
    "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so")
LIBOPAPI_SHA = (
    "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456")
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/gdn.py",
    "sha256": "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed",
    "symbol": "AscendGatedDeltaNetAttention.forward",
    "revision": R31_REVISION,
    "state": "task_artifact",
}
SOURCE_LINES = {
    "allocation": 422,
    "shape": 423,
    "dtype": 424,
    "device": 425,
    "close": 426,
    "tokens_mixed_qkvz": 381,
    "tokens_gqa_layout": 406,
    "consumer_core": 428,
    "consumer_norm": 444,
    "inner_valid_only_start": 487,
    "inner_valid_only_end": 491,
    "inner_valid_qkv": 489,
    "inner_valid_b": 490,
    "inner_valid_a": 491,
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor(shape: list[int], dtype: str, layout: str, role: str,
                *, allocation: str = "torch.zeros") -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": allocation}


def _r31_metadata(compiled: Any) -> dict[str, Any]:
    """The exact witnessed R31 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    from mindie_sim.prediction.mx_source_provenance import image_digest
    if not image_digest(image_id).startswith(R31_IMAGE_PREFIX):
        return {}
    lowerings = meta.get("source_lowerings")
    if isinstance(lowerings, str):
        lowerings = (lowerings,)
    if lowerings is not None and "gdn_decode" not in tuple(lowerings):
        return {}
    return meta


def _core_zeros_geometry(operator: Any) -> dict[str, int] | None:
    """The source-declared A3 core-output fill, or None for another row."""
    if operator is None:
        return None
    if str(getattr(operator, "family", "") or "") != "copy":
        return None
    if str(getattr(operator, "name", "") or "") != "core_zeros":
        return None
    try:
        theta = operator.theta
    except (AttributeError, TypeError, ValueError):
        return None
    if str(theta.find("source_kernel", "") or "") != KERNEL_NAME:
        return None
    if str(theta.find("transfer", "") or "") != "fill":
        return None
    if str(theta.find("execution", "") or "") != "zero_fill":
        return None
    if str(theta.find("row_source", "") or "") != "forward_fill":
        return None
    if str(theta.find("owner_axis", "") or "") != "sequence":
        return None
    if str(theta.find("kv_dtype", "") or "") != "bf16":
        return None
    if _int(theta.find("rank", 0)) != 3:
        return None
    kept_heads = _int(theta.find("kept_heads", 0))
    head_dim = _int(theta.find("head_dim", 0))
    elements = _int(theta.find("elements", 0))
    source_extent = _int(theta.find("source_extent", 0))
    offset = _int(theta.find("offset_elements", 0))
    read_bytes = _int(theta.find("read_bytes", -1))
    write_bytes = _int(theta.find("write_bytes", -1))
    if (kept_heads < 1 or head_dim < 1 or elements < 1
            or elements != kept_heads * head_dim
            or source_extent != elements or offset != 0
            or read_bytes != 0 or write_bytes != elements * 2):
        return None
    return {"kept_heads": kept_heads, "head_dim": head_dim,
            "elements": elements}


def _r31_typed(geometry: dict[str, int], meta: dict[str, Any]) -> dict:
    kept_heads = geometry["kept_heads"]
    head_dim = geometry["head_dim"]
    elements = geometry["elements"]
    return {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "all",
        "shapes": {
            "self": [kept_heads, head_dim],
            "output": [kept_heads, head_dim],
        },
        "attrs": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "acl_symbols": list(ACLNN_SYMBOLS),
            "acl_symbol_library": LIBOPAPI_PATH,
            "acl_symbol_library_sha256": LIBOPAPI_SHA,
            "profiler_name": PROFILER_NAME,
            "python_entry": ENTRY,
            "python_caller": SOURCE["symbol"],
            "source_kernel": KERNEL_NAME,
            "transfer": "fill",
            "execution": "zero_fill",
            "row_source": "forward_fill",
            "owner_axis": "sequence",
            "buffer_role": ROLE,
            "inplace": True,
            "allocation": "torch.zeros",
            "kept_heads": kept_heads,
            "head_dim": head_dim,
            "elements_per_token": elements,
            "dtype_bytes": 2,
            "read_bytes": 0,
            "write_bytes": elements * 2,
            "source_lines": [SOURCE_LINES["allocation"],
                             SOURCE_LINES["shape"],
                             SOURCE_LINES["dtype"],
                             SOURCE_LINES["device"],
                             SOURCE_LINES["close"]],
            "source_line_allocation": SOURCE_LINES["allocation"],
            "source_line_shape": SOURCE_LINES["shape"],
            "num_tokens_source": (
                "hidden_states.size(0) / projected_states_qkvz.size(0)"),
            "producer_op_id": None,
            "consumer_op_ids": ["A3:recurrent:gated_delta",
                                "A3:elementwise:gated_norm"],
            "phase_scope": "outer_allocation_before_phase",
            "inner_valid_only_lines": [
                SOURCE_LINES["inner_valid_only_start"],
                SOURCE_LINES["inner_valid_only_end"]],
            "inner_valid_only_path": (
                "AscendGatedDeltaNetAttention._forward_core truncates "
                "mixed_qkv/b/a to num_actual_tokens"),
            "inner_path_claimed": False,
            "geometry_lane": "shared_operator_layout_role_projection",
            "leaf_padding": False,
            "deployment_image_id": str(
                meta.get("image_id") or R31_IMAGE_SHA),
            "deployment_image": str(meta.get("image") or ""),
            "deployment_source_revision": "r31",
            "kernel_source_status": "kernel_source_not_retained",
            "python_wrapper_status": "source_declared",
            "aten_dispatch_status": "captured_not_executed",
            "acl_api_header_status": "not_retained",
            "acl_tensor_abi_status": "not_retained",
        },
        "occurrence_attrs": {"tokens": "positive_int"},
        "rank_width": 3,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real R31 A3 ``core_zeros`` ``_CallSpec``, or ``None``.

    Selection is the exact R31 deployment image, ``kind == "A3"`` and the
    source-declared outer ``core_zeros`` fill row.  The allocation is before
    phase, so ordinary prefill and decode both qualify; an unknown phase,
    another image, the beta/alpha/q/k/v/z copy rows, the separate valid-only
    inner truncation and every other operator return ``None``.
    """
    if str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    if phase not in {"", "prefill", "decode"}:
        return None
    meta = _r31_metadata(compiled)
    if not meta:
        return None
    siblings = tuple(getattr(compiled, "siblings", ()) or ())
    sibling_names = {str(getattr(item, "name", "") or "")
                     for item in siblings}
    if not {"core_zeros", "gated_delta", "gated_norm"} <= sibling_names:
        return None
    operator = getattr(compiled, "operator", None)
    geometry = _core_zeros_geometry(operator)
    if geometry is None:
        return None
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        acl_symbols=list(ACLNN_SYMBOLS),
        acl_symbol_library=LIBOPAPI_PATH,
        acl_symbol_library_sha256=LIBOPAPI_SHA,
        kernel_source_status="kernel_source_not_retained")
    typed = _r31_typed(geometry, meta)
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=R31_SOURCE_VERSION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=("aten_dispatch_to_aclnn_inplace_zero_not_retained",
                    "acl_api_header_not_retained",
                    "acl_tensor_argument_abi_not_retained",
                    "zeros_like_kernel_source_not_retained",
                    "kernel_binary_selection_unresolved"))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 A3 core-output zero occurrence for the tokens.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is the existing
    ``A3:copy:core_zeros`` row; the later ``qwen_gdn_attention_core`` and
    gated-norm work is attributed to its own rows, not charged here.
    """
    api = binding.api or {}
    if (api.get("entry") != ENTRY
            and api.get("aclnn_symbol") != ACLNN_API):
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("acl_api") != ACLNN_API
            or attrs.get("op_type") != KERNEL_NAME
            or attrs.get("buffer_role") != ROLE
            or attrs.get("deployment_source_revision") != "r31"):
        return None
    phase_name = str(getattr(geometry, "phase_name", "") or "")
    if phase_name not in {"prefill", "decode"}:
        return None
    # FEATURES8's central ``WorkloadGeometry.operator_layout`` has already
    # selected this row's role-specific physical token lane.  Consume it
    # directly; do not round, pad or multiply rows in this leaf.
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    kept_heads = _int(attrs.get("kept_heads", 0))
    head_dim = _int(attrs.get("head_dim", 0))
    if tokens < 1 or kept_heads < 1 or head_dim < 1:
        return None
    cache_key = (binding.call_id, tokens)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "self": [tokens, kept_heads, head_dim],
        "output": [tokens, kept_heads, head_dim],
    }
    actual_attrs = {
        **attrs,
        "tokens": tokens,
        "m": tokens,
        "observed_input_shapes": {"self": list(actual_shapes["self"])},
        "observed_output_shapes": {"output": list(actual_shapes["output"])},
        "phase": phase_name,
        "token_lane": "shared_operator_layout_role_projection",
    }
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "self": _descriptor(actual_shapes["self"], "bf16", "ND",
                            "inplace_input_output"),
        "output": _descriptor(actual_shapes["output"], "bf16", "ND",
                              "output"),
    }
    for name in descriptors:
        descriptors[name]["alias_group"] = "core_attn_out"
        descriptors[name]["storage"] = "core_attn_out"
    descriptors["self"]["parent_allocation"] = "torch.zeros"
    descriptors["output"]["aliases"] = "self"
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "source_status": "kernel_source_not_retained"})
    prepare = {
        "schema": "mindie-sim/inplace-zero-native-prepare/1",
        "call": ENTRY,
        "native_entry": ACLNN_API,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": [SOURCE_LINES["allocation"],
                         SOURCE_LINES["close"]],
        "parent_allocation_line": SOURCE_LINES["allocation"],
        "outer_allocation_lines": [SOURCE_LINES["allocation"],
                                   SOURCE_LINES["close"]],
        "inner_valid_only_lines": [
            SOURCE_LINES["inner_valid_only_start"],
            SOURCE_LINES["inner_valid_only_end"]],
        "inner_valid_only_path": (
            "AscendGatedDeltaNetAttention._forward_core truncates "
            "mixed_qkv/b/a to num_actual_tokens"),
        "inner_path_claimed": False,
        "phase": phase_name,
        "shape_line": SOURCE_LINES["shape"],
        "num_tokens_source_lines": [SOURCE_LINES["tokens_mixed_qkvz"],
                                    SOURCE_LINES["tokens_gqa_layout"]],
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {"selfRef": "core_attn_out"},
        "argument_order": ["selfRef"],
        "positional_arguments": {"selfRef": "core_attn_out"},
        "torch_arguments": {
            "size": [tokens, kept_heads, head_dim],
            "dtype": "hidden_states.dtype",
            "device": "hidden_states.device",
            "layout": "strided",
        },
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "inplace": {
            "self_ref": "core_attn_out",
            "output_aliases_input": True,
            "parent_allocation": "torch.zeros",
            "initialized_before": "torch.ops.vllm.qwen_gdn_attention_core",
            "consumer_line": SOURCE_LINES["consumer_core"],
            "later_norm_line": SOURCE_LINES["consumer_norm"],
        },
        "aliases": {"self": "output",
                    "basis": "one in-place ACLNN operand",
                    "storage_shared": True},
        "optional_slots": {
            "workspace": "acl_executor_owned_not_prepared",
            "stream": "aclrtStream_caller_owned",
            "scalar": "not_present",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "m": tokens,
            "kept_heads": kept_heads,
            "head_dim": head_dim,
            "elements_per_token": kept_heads * head_dim,
            "dtype": "bf16",
            "layout": "ND",
            "inplace": True,
            "phase": phase_name,
            "phase_scope": "outer_allocation_before_phase",
            "inner_path_claimed": False,
            "source_sha256": source.get("sha256"),
            "source_lines": [SOURCE_LINES["allocation"],
                             SOURCE_LINES["close"]],
            "source_version": R31_SOURCE_VERSION,
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R31_IMAGE_SHA),
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R31_IMAGE_SHA),
            "image": str(attrs.get("deployment_image") or ""),
            "revision": R31_REVISION,
            "selection": ("exact R31 image digest + A3 core_zeros row + "
                          "outer gdn.py line-422 torch.zeros allocation "
                          "before phase"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "acl_symbol_library": LIBOPAPI_PATH,
            "acl_symbol_library_sha256": LIBOPAPI_SHA,
            "acl_symbols": list(ACLNN_SYMBOLS),
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_binary_selection_trace_only",
            "acl_api": ACLNN_API,
            "acl_symbols_proven": list(ACLNN_SYMBOLS),
            "acl_symbol_library_sha256": LIBOPAPI_SHA,
            "owner_kernel": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "source_status": "kernel_source_not_retained",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "aten_dispatch_to_aclnn_inplace_zero_not_retained",
            "acl_api_header_not_retained",
            "acl_tensor_argument_abi_not_retained",
            "zeros_like_kernel_source_not_retained",
        ],
        "unresolved": [
            "aten_dispatch_to_aclnn_inplace_zero_not_retained",
            "acl_api_header_not_retained",
            "acl_tensor_argument_abi_not_retained",
            "zeros_like_kernel_source_not_retained",
            "kernel_binary_selection_unresolved",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_acl_tensor_pointer_fabrication",
            "no_acl_header_argument_order_claim",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "bf16", "layout": "ND",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
