"""Prepare-only R64 outgoing MoeTokenPermute invocation.

[Role] Expand the source-selected prefill outbound
  ``torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens)`` call
  from its real R64 caller, ``TokenDispatcherWithAll2AllV._dispatch_preprocess``,
  into the actual BF16 dense-row / INT32 top-k input and expanded-row output
  descriptors for the evaluated token geometry.
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the caller blob that is invariant across every source
  commit whose ``device_op.py`` is the R64 ``2a251259...`` blob.  M and the
  expanded row count ``M*k`` live only in the occurrence; ``k``, H, dtype,
  API, kernel name and output contract stay in the binding.  No tensor is
  allocated, no pointer or symbol is fabricated, no device launch is claimed
  and no measurement is used to price or select work.  The outbound
  TokenPermute is not ``npu_moe_init_routing_v2``; the latter is a different
  API and is deliberately not admitted here.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal geometry; unknown source/ABI stays absent or unresolved.
"""
from __future__ import annotations

from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch_npu.npu_moe_token_permute"
ACLNN_API = "aclnnMoeTokenPermute"
KERNEL_NAME = "MoeInitRoutingV2"
PROFILER_NAME = "aclnnMoeTokenPermute_MoeInitRoutingV2_MoeInit"
SOURCE_LINES = (619, 623)
NUM_OUT_TOKENS_LINE = 647
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/fused_moe/token_dispatcher.py",
    "sha256": "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5",
    "symbol": "TokenDispatcherWithAll2AllV._dispatch_preprocess",
    "revision": REVISION,
    "state": "task_artifact",
}
ROLE = "outbound_local_before_dispatch"
SOURCE_SELECTED = "prefill_local_outbound"
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if source_revision_from_oci(image, image_id) != "r64":
        return {}
    return meta


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 outgoing TokenPermute call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Decode, non-R64 images, another routing
    row and the separate ``npu_moe_init_routing_v2`` capability return
    ``None`` so ordinary prediction is unchanged.
    """
    if str(getattr(compiled, "kind", "") or "") != "F1":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "family", "") or "") != "router"
            or str(getattr(operator, "name", "") or "") != "router_permute"):
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "")
                     or meta.get("phase") or "")
    if phase_name != "prefill":
        return None
    theta = operator.theta
    operation = theta.find("router_source_operation", None)
    if (not isinstance(operation, str)
            or not operation.startswith("MoeTokenPermute")):
        return None
    selected = str(theta.find("router_selected", SOURCE_SELECTED) or "")
    if selected not in {"", SOURCE_SELECTED}:
        return None
    declared_dtype = str(theta.find("router_source_dtype", "bf16") or "bf16")
    dtype_bytes = _int(theta.find("router_dtype_bytes", 2), 2)
    indices_bytes = _int(theta.find("router_indices_bytes", 4), 4)
    hidden = _int(theta.find("router_hidden", 0))
    top_k = _int(theta.find("k", 0))
    experts = _int(theta.find("E", 0))
    if (declared_dtype != "bf16" or dtype_bytes != 2
            or indices_bytes != 4 or hidden < 1 or top_k < 1):
        return None
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_source_status="kernel_binary_not_retained")
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "prefill",
        # `sorted_indices` is a 1-D M*k row vector with no occurrence-
        # independent suffix; the dynamic occurrence carries its exact shape
        # while the other roles keep their static invariant suffixes.
        "shapes": {
            "tokens": [hidden],
            "indices": [top_k],
            "permuted_tokens": [hidden],
        },
        "attrs": {
            "input_contract": ["tokens", "indices", "num_out_tokens"],
            "output_contract": ["permuted_tokens", "sorted_indices"],
            "router_hidden": hidden,
            "hidden": hidden,
            "k": top_k,
            "E": experts,
            "router_dtype_bytes": dtype_bytes,
            "router_indices_bytes": indices_bytes,
            "router_source_operation": operation,
            "router_source_dtype": declared_dtype,
            "router_source_device": "AIV",
            "router_selected": selected or SOURCE_SELECTED,
            "token_permute_role": ROLE,
            "deployment_image_id": str(meta.get("image_id") or
                                       meta.get("deployment_image_id") or
                                       R64_IMAGE_SHA),
            "deployment_source_revision": "r64",
            "source_commit": R64_SOURCE_COMMIT,
            "source_lines": list(SOURCE_LINES),
            "source_num_out_tokens_line": NUM_OUT_TOKENS_LINE,
            "acl_api": ACLNN_API,
            "op_type": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "kernel_source_status": "kernel_binary_not_retained",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "num_out_tokens_source": "topk_ids.numel()",
            "num_out_tokens_expr": "M*k",
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=("aclnn_symbol_unresolved",
                    "moe_token_permute_kernel_source_not_retained",
                    "moe_token_permute_binary_selection_unresolved"))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected outbound TokenPermute.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is the
    ``router_permute`` call itself; the caller's dispatch/collective work is
    not charged here.
    """
    if (binding.api or {}).get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("token_permute_role") != ROLE
            or attrs.get("acl_api") != ACLNN_API
            or attrs.get("op_type") != KERNEL_NAME
            or attrs.get("deployment_source_revision") != "r64"):
        return None
    if not str(attrs.get("router_source_operation") or "").startswith(
            "MoeTokenPermute"):
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    hidden = _int(attrs.get("router_hidden") or attrs.get("hidden"))
    top_k = _int(attrs.get("k"))
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if hidden < 1 or top_k < 1 or tokens < 1:
        return None
    rows = tokens * top_k
    cache_key = (binding.call_id, tokens, hidden, top_k)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "tokens": [tokens, hidden],
        "indices": [tokens, top_k],
        "permuted_tokens": [rows, hidden],
        "sorted_indices": [rows],
    }
    actual_attrs = {
        **attrs,
        "m": tokens,
        "num_out_tokens": rows,
        "padded_mode": False,
    }
    actual = {"dtype": "bf16", "layout": "ND", "shapes": actual_shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "tokens": _descriptor([tokens, hidden], "bf16", "ND", "input"),
        "indices": _descriptor([tokens, top_k], "int32", "ND", "input"),
        "permuted_tokens": _descriptor([rows, hidden], "bf16", "ND",
                                       "output"),
        "sorted_indices": _descriptor([rows], "int32", "ND", "output"),
        "num_out_tokens": {
            "shape": [], "dtype": "int32", "layout": "ND",
            "role": "host_scalar", "value": rows, "pointer": None,
            "allocation": "caller",
        },
    }
    arguments = {
        "tokens": "tokens",
        "indices": "indices",
        "num_out_tokens": rows,
    }
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "source_status": "kernel_binary_not_retained"})
    prepare = {
        "schema": "mindie-sim/moe-token-permute-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R64_SOURCE_COMMIT,
        "source_lines": list(SOURCE_LINES),
        "source_num_out_tokens_line": NUM_OUT_TOKENS_LINE,
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": ["tokens", "indices", "num_out_tokens"],
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "optional_slots": {
            "num_out_tokens": "host_scalar_from_topk_ids_numel",
            "padded_mode": "not_passed_default_false",
        },
        "match": {
            "native_type": "MoeTokenPermute",
            "op_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "m": tokens,
            "k": top_k,
            "hidden": hidden,
            "num_out_tokens": rows,
            "source_sha256": source.get("sha256"),
            "source_lines": list(SOURCE_LINES),
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "source_commit": R64_SOURCE_COMMIT,
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or R64_IMAGE_SHA),
            "revision": REVISION,
            "selection": ("exact R64 image digest + caller blob invariant "
                          "over the R64 device_op.py blob"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "device_op_sha256": R64_DEVICE_OP_SHA,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_nn_kernel_binary_not_retained",
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": ["moe_token_permute_kernel_binary_not_retained",
                            "aclnn_symbol_unresolved"],
        "unresolved": ["aclnn_symbol_unresolved",
                       "moe_token_permute_kernel_source_not_retained",
                       "moe_token_permute_binary_selection_unresolved",
                       "execution_rank_not_selected"],
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration", "no_kernel_source_claim"],
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
