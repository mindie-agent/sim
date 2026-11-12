"""Prepare-only R64 aclnnCat join of MLA q latent and q rotary.

[Role] Expand the source-selected A1 ``torch.cat([ql_nope, q_pe], dim=-1)``
  call from the R64 A5 sparse-attention caller into the actual BF16
  ``[M, q_heads, 512]`` + ``[M, q_heads, 64]`` -> ``[M, q_heads, 576]``
  descriptors for the evaluated token geometry.
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the R64 ``device_op.py`` caller blob.  ``M`` lives only
  in the occurrence; heads, latent width, rope width, API, concat axis and
  dtype stay in the binding.  The Python entry is one real ``torch.cat``.
  ``aclnnCat``/``ConcatD`` are the observed ACLNN call and kernel, not a
  source-declared ABI: the C header, kernel source and binary selection stay
  explicitly unresolved.  No tensor is allocated, no pointer or symbol is
  fabricated, no device launch is claimed and no measured clock selects work.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection,
  the sibling A1 attention signature and normal geometry.  Unknown source,
  widths or ABI stay absent or unresolved.
"""
from __future__ import annotations

from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
REVISION = "r64/8d422528"
ENTRY = "torch.cat"
ACLNN_API = "aclnnCat"
KERNEL_NAME = "ConcatD"
PROFILER_NAME = "aclnnCat_ConcatD_ConcatD"
SOURCE_LINES = (1157, 1157)
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": R64_DEVICE_OP_SHA,
    "symbol": "A5DeviceAdaptor._execute_kv_quant_sparse_flash_attention_vllm",
    "revision": REVISION,
    "state": "task_artifact",
}
ROLE = "concat"
CONCAT_ROLE = "mla_q_latent_plus_rope"
ORIGINAL_FILE = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/glm51_dp16tp1ep16/pf_1k_c1/177.25.21.1/"
    "dp2_pp0_tp0_dcp0_ep2_rank0_1467_20261107094934940_ascend_pt/"
    "kernel_details.csv.gz")
ORIGINAL_SHA = "0223d793"
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


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


def _attention_sibling(compiled: Any) -> Any:
    for operator in getattr(compiled, "siblings", ()) or ():
        if (str(getattr(operator, "name", "") or "") == "mla_attention"
                and str(getattr(operator, "family", "") or "") == "mla"):
            return operator
    return None


def _concat_dims(compiled: Any) -> tuple[int, int, int] | None:
    """The q-latent / q-rotary widths declared by the A1 structure.

    The builder's explicit extras are preferred.  The same values may be
    read from the real sibling ``mla_attention`` signature; the two are
    cross-checked when both exist.  No width is inferred from a trace.
    """
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = operator.theta
    extras = dict(theta.extras)
    heads = _int(extras.get("q_concat_heads",
                            theta.find("q_concat_heads", 0)))
    latent = _int(extras.get("q_concat_latent",
                             theta.find("q_concat_latent", 0)))
    rope = _int(extras.get("q_concat_rope",
                           theta.find("q_concat_rope", 0)))
    sibling = _attention_sibling(compiled)
    declared: dict[str, int] = {}
    if sibling is not None:
        heads_map = dict(sibling.theta.heads)
        declared = {"q": _int(heads_map.get("q", 0)),
                    "rkv": _int(heads_map.get("rkv", 0)),
                    "dr": _int(heads_map.get("dr", 0))}
        if heads < 1:
            heads = declared["q"]
        if latent < 1:
            latent = declared["rkv"]
        if rope < 1:
            rope = declared["dr"]
    if heads < 1 or latent < 1 or rope < 1:
        return None
    if declared:
        if declared["q"] and declared["q"] != heads:
            return None
        if declared["rkv"] and declared["rkv"] != latent:
            return None
        if declared["dr"] and declared["dr"] != rope:
            return None
    elements = _int(theta.find("elements", 0))
    if elements != heads * (latent + rope):
        return None
    try:
        act_bytes = float(theta.find("act_bytes"))
    except (TypeError, ValueError):
        return None
    if act_bytes != 2.0:
        return None
    return heads, latent, rope


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 A1 q-latent/q-rotary Cat identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Any non-R64 image, another operator or
    an inconsistent/undeclared A1 width returns ``None`` so ordinary
    prediction is unchanged.
    """
    if str(getattr(compiled, "kind", "") or "") != "A1":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (str(getattr(operator, "name", "") or "") != "q_concat"
            or str(getattr(operator, "family", "") or "") != "copy"
            or str(getattr(operator, "role", "") or "") != ROLE):
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    phase_name = str(phase or getattr(compiled, "phase", "")
                     or meta.get("phase") or "")
    if phase_name not in {"", "prefill", "decode"}:
        return None
    if _attention_sibling(compiled) is None:
        return None
    dims = _concat_dims(compiled)
    if dims is None:
        return None
    heads, latent, rope = dims
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    from mindie_sim.prediction.mx_source_provenance import image_digest
    digest = image_digest(image_id)
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        accelerator_core="AI_VECTOR_CORE",
        concat_axis="last",
        kernel_source_status="kernel_binary_not_retained")
    typed = {
        "dtype": "bf16",
        "layout": "NCL",
        "shapes": {
            "ql_nope": [heads, latent],
            "q_pe": [heads, rope],
            "query": [heads, latent + rope],
        },
        "attrs": {
            "input_contract": ["ql_nope", "q_pe"],
            "output_contract": ["query"],
            "concat_dim": -1,
            "concat_axis": "last",
            "input_count": 2,
            "q_heads": heads,
            "latent_width": latent,
            "rope_width": rope,
            "output_width": latent + rope,
            "concat_role": CONCAT_ROLE,
            "source_operation": "torch.cat",
            "source_expression": "torch.cat([ql_nope, q_pe], dim=-1)",
            "source_symbol": SOURCE["symbol"],
            "source_path": SOURCE["path"],
            "source_sha256": SOURCE["sha256"],
            "source_lines": list(SOURCE_LINES),
            "deployment_source_revision": "r64",
            "deployment_image_id": digest or "unresolved",
            "acl_api": ACLNN_API,
            "op_type": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "input_formats": ["NCL", "NCL"],
            "output_format": "NCL",
            "accelerator_core": "AI_VECTOR_CORE",
            "kernel_source_status": "kernel_binary_not_retained",
            "source_lowering": "torch_cat_to_aclnnCat_measured",
            "execution_core": "aiv",
        },
        "rank_width": 3,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=(
            "torch_cat_to_aclnnCat_lowering_measured_not_source_declared",
            "aclnnCat_c_interface_not_retained",
            "concatd_kernel_source_not_retained",
            "concatd_binary_selection_unresolved",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R64 A1 Cat.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The physical owner is this ``q_concat``
    call; the following SFA call owns its own attention work.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY or api.get("aclnn_symbol") != ACLNN_API:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("acl_api") != ACLNN_API
            or attrs.get("op_type") != KERNEL_NAME
            or attrs.get("concat_role") != CONCAT_ROLE
            or attrs.get("deployment_source_revision") != "r64"
            or attrs.get("concat_dim") != -1):
        return None
    heads = _int(attrs.get("q_heads"))
    latent = _int(attrs.get("latent_width"))
    rope = _int(attrs.get("rope_width"))
    tokens = _int(getattr(geometry, "effective_tokens", 0) or 0)
    if heads < 1 or latent < 1 or rope < 1 or tokens < 1:
        return None
    if int(attrs.get("output_width") or 0) != latent + rope:
        return None
    cache_key = (binding.call_id, tokens, heads, latent, rope)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "ql_nope": [tokens, heads, latent],
        "q_pe": [tokens, heads, rope],
        "query": [tokens, heads, latent + rope],
    }
    actual_attrs = {
        **attrs,
        "m": tokens,
        "input_shapes": [[tokens, heads, latent], [tokens, heads, rope]],
        "output_shape": [tokens, heads, latent + rope],
    }
    actual = {"dtype": "bf16", "layout": "NCL", "shapes": actual_shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "ql_nope": _descriptor([tokens, heads, latent], "bf16", "NCL",
                               "input"),
        "q_pe": _descriptor([tokens, heads, rope], "bf16", "NCL", "input"),
        "query": _descriptor([tokens, heads, latent + rope], "bf16", "NCL",
                             "output"),
    }
    keywords = {"out": None}
    arguments = {
        "tensors": ["ql_nope", "q_pe"],
        "dim": -1,
        **keywords,
    }
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({"acl_api": ACLNN_API, "profiler_name": PROFILER_NAME,
                   "source_status": "kernel_binary_not_retained"})
    prepare = {
        "schema": "mindie-sim/aclnn-cat-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": list(SOURCE_LINES),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "prototype": "torch.cat(tensors, dim=0, *, out=None)",
        "arguments": arguments,
        "argument_order": ["tensors", "dim", "out"],
        "positional_arguments": {"tensors": ["ql_nope", "q_pe"], "dim": -1},
        "keywords": keywords,
        "optional_slots": {"out": "not_passed"},
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "acl_interface": {
            "symbol": ACLNN_API,
            "evidence": "profiler_name_only",
            "c_header": None,
            "status": "not_retained",
        },
        "source_lowering": {
            "python_entry": ENTRY,
            "acl_symbol": ACLNN_API,
            "kernel": KERNEL_NAME,
            "evidence": "source_line_plus_adjacent_original_rows",
            "status": "measured_not_source_declared",
        },
        "observed_original": {
            "original_example": "glm51_dp16tp1ep16_pf_1k_c1",
            "file": ORIGINAL_FILE,
            "orig_sha": ORIGINAL_SHA,
            "name": PROFILER_NAME,
            "native_type": KERNEL_NAME,
            "accelerator_core": "AI_VECTOR_CORE",
            "input_shapes": ["1024,64,512", "1024,64,64"],
            "output_shape": "1024,64,576",
            "input_dtypes": ["DT_BF16", "DT_BF16"],
            "output_dtype": "DT_BF16",
            "input_formats": ["NCL", "NCL"],
            "output_format": "NCL",
            "leading_dim_counts": {"1024": 78, "1": 1326},
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "acl_api": ACLNN_API,
            "op_type": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "call_id": binding.call_id,
            "m": tokens,
            "q_heads": heads,
            "latent_width": latent,
            "rope_width": rope,
            "output_width": latent + rope,
            "concat_dim": -1,
            "input_shapes": [actual_shapes["ql_nope"],
                             actual_shapes["q_pe"]],
            "output_shape": actual_shapes["query"],
            "source_sha256": source.get("sha256"),
            "source_lines": list(SOURCE_LINES),
            "deployment_image_id": str(
                attrs.get("deployment_image_id") or "unresolved"),
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": REVISION,
            "selection": ("exact R64 image digest + R64 device_op.py caller "
                          "blob + A5 FP8_OPTIMIZED sparse-attention path"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "device_op_sha256": R64_DEVICE_OP_SHA,
        },
        "binary_selection": {
            "status": "observed_name_not_binary_identity",
            "reason": "concatd_kernel_binary_not_retained",
            "owner_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "concatd_kernel_source_not_retained",
            "aclnnCat_c_interface_not_retained",
            "torch_cat_to_aclnnCat_lowering_measured_not_source_declared",
        ],
        "unresolved": [
            "torch_cat_to_aclnnCat_lowering_measured_not_source_declared",
            "aclnnCat_c_interface_not_retained",
            "concatd_kernel_source_not_retained",
            "concatd_binary_selection_unresolved",
            "execution_rank_not_selected",
        ],
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration", "no_kernel_source_claim"],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": "bf16", "layout": "NCL",
                         "shapes": actual_shapes, "attrs": actual_attrs},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
