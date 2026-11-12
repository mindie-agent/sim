"""R31 ``aclnnInplaceCopy`` A3 z-view materialization adapter.

[Role] Bind the real R31 Qwen3.6 prefill A3 copy row ``z_slice`` to its
  deployed native surface.  The source is the actual R31 selected
  ``vllm_ascend/ops/gdn.py`` blob (image ``9e869e814846...``, source sha
  ``5e3b0fb0...``): the packed QKVZ projection is split into ``z`` as a
  non-contiguous view at lines 396-397 and then flattened at line 443
  immediately before the gated norm at line 444.  Crossing the QKVZ row gap
  materializes ``[T, 64, 128] -> [T, 24, 128]`` through the measured
  ``aclnnInplaceCopy`` parent and its ``Slice`` child.  The different
  ``c3d00a196905`` image's ``gdn.py`` blob ``e9ed210b`` is not this source.
[Invariants] Only the exact R31 deployment image and the A3 ``z_slice`` copy
  row select this adapter.  The source head count, kept head count, head
  width, packed extent, offset and genuine non-contiguous packed-view stride
  are source geometry; only ``T`` lives in the occurrence.  The
  non-contiguous view materialization is owned and charged once by
  ``A3:copy:z_slice``.  The Python/ATen dispatch, chosen backend, ACLNN C
  header, exact slice-bound values, kernel source and binary selection are
  not retained and stay explicit.  No other A3 copy row (``beta_slice``,
  ``alpha_slice``, ``core_zeros`` or the decode q/k/v rows) is admitted;
  those are separate native calls and are not silently multiplied from this
  evidence.  A different image, phase or row returns ``None`` so ordinary
  prediction is unchanged.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected source and the
  already-priced A3 copy row.  No tensor, pointer, allocation, device launch
  or measured duration is produced.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e814846"
R31_SOURCE_BLOB_SHA = (
    "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed")
R31_SOURCE_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
R31_WRAPPER_VERSION = R31_SOURCE_VERSION
CANN_VERSION = "cann-9.2.0-r31"
ACLNN_API = "aclnnInplaceCopy"
KERNEL_NAME = "Slice"
PROFILER_NAME = "aclnnInplaceCopy_SliceAiCore_Slice"
COPY_ROLE = "z_norm_materialize"
COPY_OWNER = "A3:copy:z_slice"
SOURCE_ORIGINAL_PATH = "/vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py"
SOURCE_ARCHIVE_PATH = "I-r31/selected.tar.gz"
SOURCE_ARCHIVE_SHA256 = (
    "3917b98e1cf66dc9d73d765bff7c8eca04b102775a30c08849c50722f03ccd19")
SOURCE_MANIFEST_PATH = "I-r31/selected/manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "aeaa8136ef2e2882affc5b50d5c95cc49d87251bdb310b84466922c6df648e1e")
SOURCE_MEMBER_PATH = (
    "vllm-workspace/vllm-ascend/vllm_ascend/ops/gdn.py")
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/gdn.py",
    "sha256": R31_SOURCE_BLOB_SHA,
    "symbol": "AscendGatedDeltaNetAttention.forward",
    "revision": R31_REVISION,
    "state": "task_artifact",
}
SOURCE_LINES = {
    "producer_call": [392, 392],
    "split_view": [396, 397],
    "materialize": [443, 443],
    "consumer": [444, 444],
}
OBSERVED = {
    "cell_id": "smoke_qwen36_single_pf_4k_c1",
    "raw_sha256": (
        "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a"),
    "csv_gzip_sha256": (
        "a523936e982e4fd45b80071e906276d17d18fac03c5bfaca903e2f1158f778d1"),
    "csv_path": (
        "/Volumes/data/mindie-agent-workstreams/"
        "a5-multinode-fleet-rack06-20261107/out/smoke_qwen36_single/"
        "pf_4k_c1/177.25.28.1/"
        "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz"),
    "profiler_name": PROFILER_NAME,
    "task_type": "Slice",
    "accelerator_core": "AI_VECTOR_CORE",
    "occurrences": 48,
    "input_shapes": [[4096, 64, 128], [3], [3]],
    "input_dtypes": ["DT_BF16", "INT64", "INT64"],
    "output_shapes": [[4096, 24, 128]],
    "output_dtypes": ["DT_BF16"],
    "sequence": (
        "aclnnInplaceCopy_TensorMove [4096,24,128] -> "
        "aclnnInplaceCopy_Slice [4096,64,128] -> [4096,24,128] -> "
        "aclnnInplaceCopy_TensorMove [12582912] -> "
        "_layer_norm_fwd_1pass_kernel_npu [98304,128]"),
    "sequence_counts": {
        "immediate_previous_tensor_move": 47,
        "immediate_next_tensor_move": 47,
        "layer_norm_within_next_3": 48,
    },
}
_UNRESOLVED = (
    "python_aten_dispatch_unresolved",
    "acl_api_header_not_retained",
    "slice_bounds_values_not_retained",
    "kernel_source_runtime_binary_only",
    "acl_binary_selection_trace_only",
    "chosen_backend_unresolved",
)
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _image_text(compiled: Any) -> str | None:
    """The declared R31 deployment image, never a model label."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    texts = []
    for key in ("image_id", "oci_image_digest", "deployment_image_id",
                "image", "deployment_image", "oci_image", "source_image"):
        value = meta.get(key)
        if value:
            texts.append(_text(value).strip().lower())
    joined = " ".join(texts)
    if R31_IMAGE_PREFIX in joined:
        return joined
    return None


def _copy_geometry(operator: Any) -> dict[str, int] | None:
    """The source-declared A3 z slice view, or None for another row."""
    if operator is None:
        return None
    if str(getattr(operator, "family", "") or "") != "copy":
        return None
    if str(getattr(operator, "name", "") or "") != "z_slice":
        return None
    try:
        theta = operator.theta
        matrix = dict(theta.matrix)
        extras = dict(theta.extras)
    except (AttributeError, TypeError, ValueError):
        return None
    if str(extras.get("source_kernel", "") or "") != KERNEL_NAME:
        return None
    if str(extras.get("execution", "") or "") != "copy":
        return None
    if str(extras.get("row_source", "") or "") != "forward_multirow":
        return None
    if str(extras.get("owner_axis", "") or "") != "sequence":
        return None
    if str(extras.get("kv_dtype", "") or "") != "bf16":
        return None
    rank = _int(extras.get("rank"), 0)
    head_dim = _int(extras.get("head_dim"), 0)
    source_heads = _int(extras.get("source_heads"), 0)
    kept_heads = _int(extras.get("kept_heads"), 0)
    source_extent = _int(extras.get("source_extent"), 0)
    offset = _int(extras.get("offset_elements"), -1)
    elements = _int(matrix.get("elements"), 0)
    if rank != 3 or head_dim < 1 or source_heads < 1 or kept_heads < 1:
        return None
    if source_extent < 1 or offset < 0 or elements < 1:
        return None
    if elements != kept_heads * head_dim:
        return None
    if source_extent != source_heads * head_dim:
        return None
    if offset % head_dim:
        return None
    if offset + elements != source_extent:
        return None
    if kept_heads > source_heads:
        return None
    return {
        "rank": rank,
        "head_dim": head_dim,
        "source_heads": source_heads,
        "kept_heads": kept_heads,
        "source_extent": source_extent,
        "offset_elements": offset,
        "output_elements": elements,
        "slice_offset_heads": offset // head_dim,
    }


def _typed(geometry: dict[str, int]) -> dict[str, Any]:
    return {
        "dtype": "bf16",
        "layout": "ND",
        "phase": "prefill",
        "shapes": {
            "source": [geometry["source_heads"], geometry["head_dim"]],
            "output": [geometry["kept_heads"], geometry["head_dim"]],
        },
        "attrs": {
            "copy_role": COPY_ROLE,
            "source_kernel": KERNEL_NAME,
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "rank": geometry["rank"],
            "head_dim": geometry["head_dim"],
            "source_heads": geometry["source_heads"],
            "kept_heads": geometry["kept_heads"],
            "source_extent": geometry["source_extent"],
            "offset_elements": geometry["offset_elements"],
            "output_elements": geometry["output_elements"],
            "dtype_bytes": 2,
            "source_row_bytes": geometry["source_extent"] * 2,
            "output_row_bytes": geometry["output_elements"] * 2,
            "source_lines": dict(SOURCE_LINES),
            "source_original_path": SOURCE_ORIGINAL_PATH,
            "source_archive_path": SOURCE_ARCHIVE_PATH,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_member_path": SOURCE_MEMBER_PATH,
            "source_blob_sha256": R31_SOURCE_BLOB_SHA,
            "source_image_id": R31_IMAGE_SHA,
            "deployment_image_id": R31_IMAGE_SHA,
            "deployment_source_revision": R31_REVISION,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "cann_version": CANN_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "python_caller": SOURCE["symbol"],
            "python_wrapper_status": "unresolved",
            "aten_dispatch_status": "unresolved",
            "chosen_backend_status": "unresolved",
            "acl_api_header_status": "not_retained",
            "kernel_source_status": "kernel_binary_not_retained",
            "slice_bounds_status": "shape_derived_values_not_retained",
            "copy_owner": COPY_OWNER,
            "copy_kind": "noncontiguous_view_materialization",
            "source_view_noncontiguous": True,
            "source_view_layout": "packed_qkvz_strided",
            "output_layout": "contiguous_nd",
            "source_stride_elements": [
                geometry["source_extent"], geometry["head_dim"], 1],
            "output_stride_elements": [
                geometry["kept_heads"] * geometry["head_dim"],
                geometry["head_dim"], 1],
            "source_packed_width_elements": geometry["source_extent"],
            "source_offset_heads": geometry["slice_offset_heads"],
            "producer_op_id": "A3:gemm:qkvz_projection",
            "producer_source_lines": [392, 396, 397],
            "consumer_op_id": "A3:elementwise:gated_norm",
            "consumer_source_lines": [443, 444],
        },
        "rank_width": 3,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real R31 A3 ``z_slice`` ``aclnnInplaceCopy`` spec or None.

    Selection is the exact R31 deployment/source image, ``kind == "A3"`` and
    the source-declared ``z_slice`` view geometry.  An empty or non-prefill
    phase, another image, another copy row or the decode q/k/v rows return
    ``None``.  The returned ``_CallSpec`` records the missing Python/ATen
    dispatch, ACLNN header, slice-bound values and kernel source explicitly.
    """
    if str(getattr(compiled, "kind", "") or "") != "A3":
        return None
    if phase != "prefill":
        return None
    if _image_text(compiled) is None:
        return None
    operator = getattr(compiled, "operator", None)
    geometry = _copy_geometry(operator)
    if geometry is None:
        return None
    from . import native_calls

    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "copy", evidence="measured",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_source_status="kernel_binary_not_retained")
    return native_calls._call_spec(
        native_calls._api("aclnn", ACLNN_API, aclnn_symbol=ACLNN_API,
                          version=CANN_VERSION),
        source,
        _typed(geometry),
        (kernel,),
        (),
        str(getattr(operator, "name", "z_slice")),
        execution_core="aiv",
        unresolved=_UNRESOLVED,
    )


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _geometry_from_attrs(attrs: dict[str, Any]) -> dict[str, int] | None:
    try:
        rank = _int(attrs.get("rank"), 0)
        head_dim = _int(attrs.get("head_dim"), 0)
        source_heads = _int(attrs.get("source_heads"), 0)
        kept_heads = _int(attrs.get("kept_heads"), 0)
        source_extent = _int(attrs.get("source_extent"), 0)
        offset = _int(attrs.get("offset_elements"), -1)
        elements = _int(attrs.get("output_elements"), 0)
    except (TypeError, ValueError):
        return None
    if rank != 3 or head_dim < 1 or source_heads < 1 or kept_heads < 1:
        return None
    if source_extent < 1 or offset < 0 or elements < 1:
        return None
    if elements != kept_heads * head_dim:
        return None
    if source_extent != source_heads * head_dim:
        return None
    if offset % head_dim or offset + elements != source_extent:
        return None
    if kept_heads > source_heads:
        return None
    return {
        "rank": rank,
        "head_dim": head_dim,
        "source_heads": source_heads,
        "kept_heads": kept_heads,
        "source_extent": source_extent,
        "offset_elements": offset,
        "output_elements": elements,
        "slice_offset_heads": offset // head_dim,
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 A3 ``z_slice`` occurrence for the evaluated tokens.

    The returned envelope mirrors ``native_qbmm.dynamic_qbmm``.  ``T`` is the
    only dynamic axis; the source head/kept-head/head-width layout and the
    packed-view offset stay in the selected call identity.  The physical
    owner is the existing ``A3:copy:z_slice`` row; no sibling copy row is
    charged or attributed here.
    """
    api = binding.api or {}
    if (api.get("entry") != ACLNN_API
            and api.get("aclnn_symbol") != ACLNN_API):
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("copy_role") != COPY_ROLE:
        return None
    if attrs.get("source_kernel") != KERNEL_NAME:
        return None
    if attrs.get("acl_api") != ACLNN_API:
        return None
    if attrs.get("deployment_source_revision") != R31_REVISION:
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    tokens = _int(getattr(geometry, "effective_tokens", 0))
    if tokens < 1:
        return None
    operator = getattr(compiled, "operator", None)
    selected = _copy_geometry(operator)
    if selected is None:
        return None
    declared = _geometry_from_attrs(attrs)
    if declared is None or declared != selected:
        return None
    cache_key = (binding.call_id, tokens)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    head_dim = selected["head_dim"]
    source_heads = selected["source_heads"]
    kept_heads = selected["kept_heads"]
    source_extent = selected["source_extent"]
    offset = selected["offset_elements"]
    source_shape = [tokens, source_heads, head_dim]
    output_shape = [tokens, kept_heads, head_dim]
    actual_shapes = {"source": source_shape, "output": output_shape}
    actual_attrs = {
        **attrs,
        "tokens": tokens,
        "source_shape": list(source_shape),
        "output_shape": list(output_shape),
        "source_view_offset_elements": offset,
        "source_view_extent_elements": selected["output_elements"],
        "slice_offset_heads": selected["slice_offset_heads"],
        "slice_kept_heads": kept_heads,
        "source_stride_elements": [source_extent, head_dim, 1],
        "output_stride_elements": [kept_heads * head_dim, head_dim, 1],
        "source_view_noncontiguous": True,
        "copy_owner": COPY_OWNER,
        "copy_kind": "noncontiguous_view_materialization",
        "source_view_layout": "packed_qkvz_strided",
        "source_packed_width_elements": source_extent,
        "source_offset_heads": selected["slice_offset_heads"],
    }
    actual = {"dtype": binding.typed.get("dtype"),
              "layout": binding.typed.get("layout"),
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "source": _descriptor(source_shape, "bf16", "ND", "input"),
        "output": _descriptor(output_shape, "bf16", "ND", "output"),
    }
    descriptors["source"].update({
        "storage_shape": list(source_shape),
        "view_offset_elements": offset,
        "view_extent_elements": selected["output_elements"],
        "view_stride_elements": [source_extent, head_dim, 1],
        "producer": "A3:gemm:qkvz_projection",
        "producer_row": "packed_qkvz",
        "noncontiguous": True,
        "view_layout": "packed_qkvz_strided",
        "copy_owner": COPY_OWNER,
    })
    descriptors["output"].update({
        "storage_shape": list(output_shape),
        "view_stride_elements": [kept_heads * head_dim, head_dim, 1],
        "allocation": "destination_materialization",
        "consumer": "A3:elementwise:gated_norm",
        "contiguous": True,
        "copy_owner": COPY_OWNER,
    })
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {
        "name": KERNEL_NAME, "role": "copy", "evidence": "measured"}
    kernel.update({
        "observed_name": PROFILER_NAME,
        "observed_task_type": "Slice",
        "observed_core": "AI_VECTOR_CORE",
        "source_status": "kernel_binary_not_retained",
    })
    prepare = {
        "schema": "mindie-sim/inplace-copy-native-prepare/1",
        "call": ACLNN_API,
        "native_entry": ACLNN_API,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_lines": dict(SOURCE_LINES),
        "source_original_path": SOURCE_ORIGINAL_PATH,
        "source_archive_path": SOURCE_ARCHIVE_PATH,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "source_manifest_path": SOURCE_MANIFEST_PATH,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
        "source_member_path": SOURCE_MEMBER_PATH,
        "source_blob_sha256": R31_SOURCE_BLOB_SHA,
        "source_version": R31_SOURCE_VERSION,
        "cann_version": CANN_VERSION,
        "wrapper_version": R31_WRAPPER_VERSION,
        "selected_host_variant": None,
        "selected_host_variant_status": "unresolved",
        "kernel": kernel,
        "chosen_backend": {
            "status": "unresolved",
            "observed_core": "AI_VECTOR_CORE",
            "observed_child": KERNEL_NAME,
            "reason": "acl_binary_selection_trace_only",
            "claimed_complete": False,
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {
            "destination": "output",
            "source": "source_view",
        },
        "argument_order": ["destination", "source"],
        "argument_order_status": (
            "aclnn_inplace_copy_name_semantics; exact C header/order "
            "not retained"),
        "lowering_arguments": {
            "source_view_offset_elements": offset,
            "source_view_extent_elements": selected["output_elements"],
            "source_packed_width_elements": source_extent,
            "source_offset_heads": selected["slice_offset_heads"],
            "source_view_noncontiguous": True,
            "copy_owner": COPY_OWNER,
            "source_heads": source_heads,
            "kept_heads": kept_heads,
            "head_dim": head_dim,
        },
        "observed_child_operands": [
            {"shape": [3], "dtype": "INT64", "role": "slice_bounds",
             "value_status": "not_retained"},
            {"shape": [3], "dtype": "INT64", "role": "slice_bounds",
             "value_status": "not_retained"},
        ],
        "optional_slots": {
            "destination": "present",
            "source": "present",
            "slice_bounds": "present_child_operands_values_not_retained",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "slice_view": {
            "source": "packed_qkvz_row",
            "source_shape": list(source_shape),
            "output_shape": list(output_shape),
            "offset_elements": offset,
            "extent_elements": selected["output_elements"],
            "source_stride_elements": [source_extent, head_dim, 1],
            "output_stride_elements": [kept_heads * head_dim, head_dim, 1],
            "source_packed_width_elements": source_extent,
            "source_offset_heads": selected["slice_offset_heads"],
            "source_view_noncontiguous": True,
            "source_view_layout": "packed_qkvz_strided",
            "output_layout": "contiguous_nd",
            "copy_owner": COPY_OWNER,
            "ownership": "one_charged_native_call",
            "source_heads": source_heads,
            "kept_heads": kept_heads,
            "head_dim": head_dim,
            "rank": 3,
        },
        "observed": {
            "cell_id": OBSERVED["cell_id"],
            "raw_stream_sha256": OBSERVED["raw_sha256"],
            "csv_gzip_sha256": OBSERVED["csv_gzip_sha256"],
            "csv_path": OBSERVED["csv_path"],
            "profiler_name": PROFILER_NAME,
            "task_type": OBSERVED["task_type"],
            "accelerator_core": OBSERVED["accelerator_core"],
            "occurrences": OBSERVED["occurrences"],
            "sample_tokens": 4096,
            "current_tokens": tokens,
            "sample_matches_current_geometry": tokens == 4096,
            "input_shapes": OBSERVED["input_shapes"],
            "input_dtypes": OBSERVED["input_dtypes"],
            "output_shapes": OBSERVED["output_shapes"],
            "output_dtypes": OBSERVED["output_dtypes"],
            "sequence": OBSERVED["sequence"],
            "sequence_counts": dict(OBSERVED["sequence_counts"]),
        },
        "deployment": {
            "image_id": R31_IMAGE_SHA,
            "source_image_id": R31_IMAGE_SHA,
            "revision": R31_REVISION,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "cann_version": CANN_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "selection": (
                "exact R31 deployment image 9e869e814846 + actual R31 "
                "selected gdn.py blob 5e3b0fb0; the different c3d00a "
                "gdn.py e9ed210b image is not selected"),
            "source_path": SOURCE["path"],
            "source_member_path": SOURCE_MEMBER_PATH,
            "source_original_path": SOURCE_ORIGINAL_PATH,
            "source_archive_path": SOURCE_ARCHIVE_PATH,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "source_sha256": SOURCE["sha256"],
            "source_blob_sha256": R31_SOURCE_BLOB_SHA,
            "source_symbol": SOURCE["symbol"],
            "source_lines": dict(SOURCE_LINES),
            "python_caller_status": "source_declared",
            "python_wrapper_status": "unresolved",
            "aten_dispatch_status": "unresolved",
            "chosen_backend_status": "unresolved",
            "acl_api_header_status": "not_retained",
            "kernel_source_status": "kernel_binary_not_retained",
        },
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ACLNN_API,
            "acl_api": ACLNN_API,
            "call_id": binding.call_id,
            "deployment_image_id": R31_IMAGE_SHA,
            "source_image_id": R31_IMAGE_SHA,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "source_blob_sha256": R31_SOURCE_BLOB_SHA,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_manifest_sha256": SOURCE_MANIFEST_SHA256,
            "tokens": tokens,
            "source_shape": list(source_shape),
            "output_shape": list(output_shape),
            "source_heads": source_heads,
            "kept_heads": kept_heads,
            "head_dim": head_dim,
            "offset_elements": offset,
            "source_offset_heads": selected["slice_offset_heads"],
            "source_packed_width_elements": source_extent,
            "source_view_noncontiguous": True,
            "copy_owner": COPY_OWNER,
            "dtype": "bf16",
            "layout": "ND",
            "source_sha256": SOURCE["sha256"],
            "source_lines": dict(SOURCE_LINES),
            "complete": False,
            "partial": True,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_binary_selection_trace_only",
            "chosen_backend_status": "unresolved",
            "owner_kernel": KERNEL_NAME,
            "profiler_name": PROFILER_NAME,
            "source_status": "kernel_binary_not_retained",
            "acl_symbol": ACLNN_API,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [*_UNRESOLVED, "execution_rank_not_selected"],
        "unresolved": [*_UNRESOLVED, "execution_rank_not_selected"],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_acl_api_header_claim",
            "no_argument_order_claim",
            "no_chosen_backend_claim",
            "no_physical_completion",
        ],
    }
    result = {"actual": validated, "prepare": prepare,
              "prepare_args": {
                  "dtype": validated.get("dtype"),
                  "layout": validated.get("layout"),
                  "shapes": validated.get("shapes"),
                  "attrs": validated.get("attrs"),
              }}
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result


def observed_rows() -> dict[str, dict]:
    """The original R31 PF4kC1 z materialization sample, retained."""
    return {
        "A3:copy:z_slice": {
            "source": [4096, 64, 128],
            "output": [4096, 24, 128],
            "occurrences": 48,
            "profiler_name": PROFILER_NAME,
            "accelerator_core": "AI_VECTOR_CORE",
            "raw_stream_sha256": OBSERVED["raw_sha256"],
            "csv_gzip_sha256": OBSERVED["csv_gzip_sha256"],
            "source_sha256": SOURCE["sha256"],
            "source_blob_sha256": R31_SOURCE_BLOB_SHA,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_lines": dict(SOURCE_LINES),
            "copy_owner": COPY_OWNER,
        }
    }
