"""Prepare-only R64 ScatterPaCache indexer cache write.

[Role] Expand the existing R64 A5 single-stream indexer cache write at
  ``vllm_ascend/device/device_op.py:1537``
  (``A5DeviceAdaptor.indexer_quant_scatter``) into the real fused
  ``torch.ops._C_ascend.indexer_compress_epilog_v2`` call and the two
  captured ``ScatterPaCache`` kernel occurrences (FP8 index cache and FP32
  scale cache) for the evaluated token geometry.  The compiled initiator is
  the existing ``I0:kv_access:index_cache_store`` operator; the census role
  label is "I1 quantized cache write".
[Invariants] The source identity is the exact R64 image digest
  ``8d422528...`` and the retained R64 ``device_op.py`` blob.  The selected
  branch is the single-stream ``indexer_quant_scatter`` call; the
  ``indexer_quant_scatter_part1`` multi-stream branch is explicitly rejected
  because this deployment does not declare ``multistream_attention_indexer``.
  Head/width identity, dtype and source lines stay in the binding; the token
  lane and the two cache-write occurrences live only in the occurrence.  The
  indexer read (``QuantLightningIndexer``) is a separate signed call and is
  never charged here.  The fused op's C++ source and ACLNN symbol are not
  retained, so the wrapper-to-kernel mapping, packed cache row bytes and
  cache capacity stay explicit unresolved fields.  No tensor is allocated, no
  pointer or symbol is fabricated, no device launch is claimed and no
  measured clock prices or selects work.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing source selection
  and normal workload geometry.
"""
from __future__ import annotations

from typing import Any

R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
R64_DEVICE_OP_SHA = "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d"
R64_SOURCE_COMMIT = "5929df13d8dc5808186c19446f12078a4a706f40"
REVISION = "r64/8d422528"
ENTRY = "torch.ops._C_ascend.indexer_compress_epilog_v2"
KERNEL_NAME = "ScatterPaCache"
SOURCE_SYMBOL = "A5DeviceAdaptor.indexer_quant_scatter"
REJECTED_SYMBOL = "A5DeviceAdaptor.indexer_quant_scatter_part1"
SOURCE_LINES = [1537, 1552]
CALL_LINES = [1547, 1552]
REJECTED_LINES = [1556, 1569]
LAYOUT = 2
SELECTED_BRANCH = "single_stream_indexer_quant_scatter"
REJECTED_BRANCH = "multistream_indexer_quant_scatter_part1"
CAPTURE_CASE = "glm51_dp16tp1ep16_pf_1k_c1"
CAPTURE_CSV = (
    "/Volumes/data/mindie-agent-workstreams/"
    "a5-multinode-fleet-rack06-20261107/out/glm51_dp16tp1ep16/"
    "pf_1k_c1/177.25.21.1/dp2_pp0_tp0_dcp0_ep2_rank0_1467_"
    "20261107094934940_ascend_pt/kernel_details.csv.gz"
)
CAPTURE_CSV_SHA_PREFIX = "0223d793"
SOURCE = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": R64_DEVICE_OP_SHA,
    "symbol": SOURCE_SYMBOL,
    "revision": REVISION,
    "state": "task_artifact",
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _descriptor(shape: Any, dtype: Any, layout: Any, role: str, *,
                allocation: str = "external", **extra: Any) -> dict:
    result = {
        "shape": (None if shape is None else
                  [None if item is None else int(item) for item in shape]),
        "dtype": dtype, "layout": layout, "role": role, "pointer": None,
        "allocation": allocation,
    }
    result.update({key: value for key, value in extra.items()
                   if value is not None})
    return result


def _r64_metadata(compiled: Any) -> dict[str, Any]:
    """The exact witnessed R64 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image") or "")
    from mindie_sim.prediction.mx_source_provenance import (
        source_revision_from_oci)
    if source_revision_from_oci(image, image_id) != "r64":
        return {}
    return meta


def _cache_dims(compiled: Any, meta: dict[str, Any]) -> dict[str, int] | None:
    """The source-declared index-cache row identity, or ``None``.

    The row width must be the declared index width at one byte (FP8).  A
    different width, a pooled producer or a fixed valid-row declaration is a
    different invocation and stays unbound; a shared-cache consumer does not
    emit this store operator in the existing I0 builder.
    """
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = operator.theta
    extras = dict(theta.extras)
    dim = _int(extras.get("dim"), 0)
    heads = _int(extras.get("heads"), 0)
    kpool = _int(extras.get("kpool"), 0)
    row_bytes = float(theta.find("row_bytes", 0) or 0)
    valid_rows_mode = str(theta.find("valid_rows_mode", "") or "")
    cache_format = str(theta.find("cache_format", "") or "")
    block_size = _int(meta.get("kv_block_size"), 0)
    if dim < 1 or heads < 1 or block_size < 1:
        return None
    if kpool != 1:
        return None
    if valid_rows_mode != "dynamic":
        return None
    if cache_format != "index_dtype":
        return None
    if row_bytes != float(dim):
        return None
    return {"dim": dim, "heads": heads, "row_bytes": int(row_bytes),
            "block_size": block_size, "kpool": kpool}


def _phase(compiled: Any, phase: str) -> str:
    return str(phase or getattr(compiled, "phase", "") or "").lower()


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R64 indexer ``ScatterPaCache`` cache-write identity or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain.  Non-R64 images, the multi-stream part1 branch, a BF16/absent
    index cache, the separate ``I1:indexer:index_score`` read and every other
    operator return ``None`` so ordinary prediction stays unchanged.
    """
    if compiled is None or str(getattr(compiled, "kind", "") or "") != "I0":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "name", "") or "") != "index_cache_store":
        return None
    if str(getattr(operator, "family", "") or "") != "kv_access":
        return None
    meta = _r64_metadata(compiled)
    if not meta:
        return None
    key_dtype = str(meta.get("indexer_kv_dtype") or "").lower()
    if key_dtype == "fp8":
        key_dtype = "fp8_e4m3"
    if key_dtype != "fp8_e4m3":
        return None
    if meta.get("indexer_query_scale_present") is not True:
        return None
    if str(meta.get("indexer_query_scale_dtype") or "").lower() != "fp32":
        return None
    if bool(meta.get("multistream_attention_indexer")):
        return None
    phase_name = _phase(compiled, phase)
    if phase_name and phase_name not in {"prefill", "pf"}:
        return None
    dims = _cache_dims(compiled, meta)
    if dims is None:
        return None
    dim, heads = dims["dim"], dims["heads"]
    from . import native_calls
    source = native_calls._source(
        SOURCE["repository"], SOURCE["path"], sha256=SOURCE["sha256"],
        symbol=SOURCE["symbol"], revision=SOURCE["revision"],
        state=SOURCE["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="measured",
        profiler_name=KERNEL_NAME,
        kernel_source_status="kernel_source_not_retained",
        kernel_mapping_status="captured_not_source_linked",
        captured_roles=["indexer_k_cache", "indexer_scale_cache"],
        captured_input_dtypes=["fp8_e4m3", "fp8_e4m3", "int32"],
        captured_scale_input_dtypes=["fp32", "fp32", "int32"],
        captured_output_dtypes=["fp8_e4m3", "fp32"],
        captured_input_slots=6,
        captured_defined_input_slots=3,
        captured_optional_input_state="DT_UNDEFINED",
        captured_case=CAPTURE_CASE,
    )
    typed = {
        "dtype": "fp8_e4m3",
        "layout": "PA_BSND",
        "phase": "prefill",
        "shapes": {
            "update": [1, dim],
            "cache": [1, 1, dim],
            "scale_update": [1],
            "scale_cache": [1, 1, 1],
            "output": [1, 1, dim],
            "scale_output": [1, 1, 1],
        },
        "attrs": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "profiler_name": KERNEL_NAME,
            "entry": ENTRY,
            "layout_value": LAYOUT,
            "cache_format": "index_dtype",
            "indexer_kv_dtype": "fp8_e4m3",
            "scale_cache_present": True,
            "scale_dtype": "fp32",
            "index_heads": heads,
            "index_head_dim": dim,
            "block_size": dims["block_size"],
            "kpool": dims["kpool"],
            "valid_rows_mode": "dynamic",
            "source_family": SOURCE_SYMBOL,
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
            "rejected_reason": "multistream_attention_indexer_not_selected",
            "source_lines": list(SOURCE_LINES),
            "call_lines": list(CALL_LINES),
            "rejected_source_lines": list(REJECTED_LINES),
            "deployment_image_id": R64_IMAGE_SHA,
            "deployment_source_revision": "r64",
            "source_commit": R64_SOURCE_COMMIT,
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "kernel_source_status": "kernel_source_not_retained",
            "kernel_mapping_status": "captured_not_source_linked",
            "indexer_read_owner": "I1:indexer:index_score",
            "physical_owner": "I0:kv_access:index_cache_store",
            "input_contract": ["indexer_compress_cache", "x",
                               "slot_mapping"],
            "kernel_argument_order": ["update", "cache", "indices"],
        },
        "rank_width": 4,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, version=REVISION), source, typed,
        (kernel,), (), "I0:kv_access:index_cache_store",
        execution_core="aiv",
        unresolved=(
            "aclnn_symbol_unresolved",
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only occurrence for the selected indexer cache write.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The primary owner is the existing
    ``I0:kv_access:index_cache_store`` row; the two captured ``ScatterPaCache``
    kernel writes and the fused wrapper's argument bindings are attribution
    records.  Cache capacity and packed row bytes are not declared by the
    normal case and stay ``None``; they are never replaced by the captured
    sample.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("native_type") != KERNEL_NAME:
        return None
    if attrs.get("selected_branch") != SELECTED_BRANCH:
        return None
    if attrs.get("deployment_source_revision") != "r64":
        return None
    if str(getattr(geometry, "phase_name", "") or "").lower() not in {
            "prefill", "pf"}:
        return None
    tokens = _int(getattr(geometry, "effective_tokens", 0), 0)
    if tokens < 1:
        return None
    dim = _int(attrs.get("index_head_dim"), 0)
    heads = _int(attrs.get("index_heads"), 0)
    block_size = _int(attrs.get("block_size"), 0)
    if dim < 1 or heads < 1 or block_size < 1:
        return None
    cache_key = (binding.call_id, tokens)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    requests = tuple(getattr(geometry, "requests", ()) or ())
    batch = max(1, len(requests))
    operator = getattr(compiled, "operator", None)
    theta_extras = dict(getattr(getattr(operator, "theta", None),
                                "extras", ()) or ())
    actual_shapes = {
        "update": [tokens, 1, dim],
        "cache": [None, 1, 1, dim],
        "scale_update": [tokens, 1, 1],
        "scale_cache": [None, 1, 1, 1],
        "output": [None, 1, 1, dim],
        "scale_output": [None, 1, 1, 1],
    }
    actual_attrs = {
        **attrs,
        "m": tokens,
        "tokens": tokens,
        "batch": batch,
        "cache_rows": None,
        "scale_cache_rows": None,
        "cache_capacity_status": "not_declared",
        "owner_layer": _int(theta_extras.get("owner_layer", -1), -1),
    }
    actual = {"dtype": "fp8_e4m3", "layout": "PA_BSND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "update": _descriptor([tokens, 1, dim], "fp8_e4m3", "PA_BSND",
                              "input"),
        "cache": _descriptor([None, 1, 1, dim], "fp8_e4m3", "PA_BSND",
                             "input_output"),
        "slot_mapping": _descriptor([tokens], "int32", "ND", "input"),
        "scale_update": _descriptor([tokens, 1, 1], "fp32", "PA_BSND",
                                    "input"),
        "scale_cache": _descriptor([None, 1, 1, 1], "fp32", "PA_BSND",
                                   "input_output"),
        "output": _descriptor([None, 1, 1, dim], "fp8_e4m3", "PA_BSND",
                              "output", alias_of="cache",
                              aliasing="in_place"),
        "scale_output": _descriptor([None, 1, 1, 1], "fp32", "PA_BSND",
                                    "output", alias_of="scale_cache",
                                    aliasing="in_place"),
    }
    wrapper_descriptors = {
        "indexer_compress_cache": _descriptor(
            [None, 1, 1, None], "uint8", "packed_indexer_cache",
            "input_output", packed_row_bytes=None,
            evidence="source_view_uint8",
            interpretation="packed k FP8 + scale FP32; split not declared"),
        "x": _descriptor(
            [tokens, 1, dim], None, "ND", "input",
            evidence="captured_kernel_update_shape",
            wrapper_dtype_status="not_retained"),
        "slot_mapping": _descriptor(
            [tokens], "int32", "ND", "input",
            evidence="source_argument_and_captured_kernel_index_dtype"),
    }
    wrapper_descriptors["indexer_compress_cache"]["packed_row_bytes"] = None
    kernel = dict(binding.kernels[0]) if binding.kernels else {}
    kernel.update({
        "captured_roles": ["indexer_k_cache", "indexer_scale_cache"],
        "captured_input_dtypes": ["fp8_e4m3", "fp8_e4m3", "int32"],
        "captured_output_dtypes": ["fp8_e4m3", "fp32"],
        "captured_case": CAPTURE_CASE,
        "mapping_status": "captured_not_source_linked",
    })
    source = dict(binding.source or {})
    prepare = {
        "schema": "mindie-sim/scatter-pa-cache-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "source": source,
        "source_commit": R64_SOURCE_COMMIT,
        "source_commit_status": "inherited_same_device_op_blob_record",
        "source_lines": list(SOURCE_LINES),
        "call_lines": list(CALL_LINES),
        "selected_branch": SELECTED_BRANCH,
        "rejected_branch": REJECTED_BRANCH,
        "rejected_reason": "multistream_attention_indexer_not_selected",
        "rejected_source_lines": list(REJECTED_LINES),
        "kernel": kernel,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "wrapper_arguments": {
            "indexer_compress_cache": "cache",
            "x": "update",
            "slot_mapping": "slot_mapping",
            "layout": LAYOUT,
        },
        "wrapper_argument_order": ["indexer_compress_cache", "x",
                                   "slot_mapping", "layout"],
        "wrapper_keywords": {
            "indexer_compress_cache": "cache",
            "x": "update",
            "slot_mapping": "slot_mapping",
            "layout": LAYOUT,
        },
        "kernel_arguments": {
            "update": "update", "cache": "cache", "indices": "slot_mapping",
        },
        "kernel_argument_order": ["update", "cache", "indices"],
        "kernel_occurrences": {
            "indexer_k_cache": {
                "kernel": KERNEL_NAME,
                "arguments": ["update", "cache", "indices"],
                "input_shapes": {
                    "update": [tokens, 1, dim],
                    "cache": [None, 1, 1, dim],
                    "indices": [tokens],
                },
                "input_dtypes": ["fp8_e4m3", "fp8_e4m3", "int32"],
                "input_formats": ["ND", "ND", "ND"],
                "optional_input_slots": 3,
                "optional_input_state": "DT_UNDEFINED",
                "output_shape": [None, 1, 1, dim],
                "output_dtype": "fp8_e4m3",
                "output_format": "ND",
                "aliasing": "cache_in_place",
                "evidence": "captured_original_profiler_row",
            },
            "indexer_scale_cache": {
                "kernel": KERNEL_NAME,
                "arguments": ["update", "cache", "indices"],
                "input_shapes": {
                    "update": [tokens, 1, 1],
                    "cache": [None, 1, 1, 1],
                    "indices": [tokens],
                },
                "input_dtypes": ["fp32", "fp32", "int32"],
                "input_formats": ["ND", "ND", "ND"],
                "optional_input_slots": 3,
                "optional_input_state": "DT_UNDEFINED",
                "output_shape": [None, 1, 1, 1],
                "output_dtype": "fp32",
                "output_format": "ND",
                "aliasing": "scale_cache_in_place",
                "evidence": "captured_original_profiler_row",
            },
        },
        "shapes": actual_shapes,
        "descriptors": descriptors,
        "arrays": descriptors,
        "wrapper_descriptors": wrapper_descriptors,
        "captured_example": {
            "case": CAPTURE_CASE,
            "source_csv": CAPTURE_CSV,
            "retained_orig_sha_prefix": CAPTURE_CSV_SHA_PREFIX,
            "retained_orig_sha_note": (
                "retained record prefix, not a full file digest"),
            "optional_input_slots": "three DT_UNDEFINED/NULL slots",
            "prefill_rows_per_cache": 78,
            "k_update_shape": [1024, 1, 128],
            "k_cache_shape": [522240, 1, 1, 128],
            "k_slot_shape": [1024],
            "k_dtype": "fp8_e4m3",
            "scale_update_shape": [1024, 1, 1],
            "scale_cache_shape": [522240, 1, 1, 1],
            "scale_slot_shape": [1024],
            "scale_dtype": "fp32",
            "captured_cache_rows": 522240,
            "note": "captured_sample_evidence_not_a_predictive_dimension",
        },
        "mapping": {
            "status": "captured_not_source_linked",
            "wrapper_source_retained": True,
            "fused_cpp_source_retained": False,
            "kernel_source_retained": False,
            "kernel_name_evidence": "original_profiler_row",
            "reason": ("indexer_compress_epilog_v2 C++ source and ACLNN "
                       "symbol are not retained; the captured ScatterPaCache "
                       "row is not promoted to a source-declared child"),
        },
        "deployment": {
            "image_id": R64_IMAGE_SHA,
            "revision": REVISION,
            "selection": ("exact R64 image digest + explicit "
                          "indexer_kv_dtype=fp8_e4m3 + present fp32 "
                          "query scale + multistream_attention_indexer "
                          "false"),
            "source_path": source.get("path"),
            "source_symbol": source.get("symbol"),
            "source_sha256": source.get("sha256"),
            "source_commit": R64_SOURCE_COMMIT,
            "source_commit_status": "inherited_same_device_op_blob_record",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
        },
        "match": {
            "native_type": KERNEL_NAME,
            "op_type": KERNEL_NAME,
            "profiler_name": KERNEL_NAME,
            "entry": ENTRY,
            "call_id": binding.call_id,
            "source_revision": REVISION,
            "source_sha256": R64_DEVICE_OP_SHA,
            "source_lines": list(SOURCE_LINES),
            "call_lines": list(CALL_LINES),
            "selected_branch": SELECTED_BRANCH,
            "rejected_branch": REJECTED_BRANCH,
            "m": tokens,
            "index_heads": heads,
            "index_head_dim": dim,
            "block_size": block_size,
            "cache_rows": None,
            "cache_dtype": "fp8_e4m3",
            "scale_cache_present": True,
            "scale_dtype": "fp32",
            "layout": "PA_BSND",
            "device_op_sha256": R64_DEVICE_OP_SHA,
            "complete": False,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": ("fused_custom_op_cpp_source_and_acl_nn_symbol_"
                       "not_retained"),
            "owner_kernel": KERNEL_NAME,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
        ],
        "unresolved": [
            "indexer_compress_epilog_v2_cpp_source_not_retained",
            "scatter_pa_cache_kernel_source_not_retained",
            "scatter_pa_cache_kernel_mapping_captured_not_source_linked",
            "packed_indexer_cache_row_bytes_unresolved",
            "indexer_cache_capacity_not_declared",
            "scale_cache_write_not_separately_priced",
            "execution_rank_not_selected",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_kernel_source_claim",
            "no_complete_native_execution",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": "fp8_e4m3",
            "layout": "PA_BSND",
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
