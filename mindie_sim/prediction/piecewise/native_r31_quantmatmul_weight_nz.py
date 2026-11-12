"""R31 W8A8-dynamic ``QuantMatmulWeightNz`` / ``QuantBatchMatmulV3`` adapter.

[Role] Bind the real R31 (CANN 9.2.0 r31) W8A8-dynamic linear projection to
  its deployed native call surface.  The source evidence is the R31
  ``libopapi.so`` export plus the original R31 Qwen3.6 PF4kC1 kernel row
  ``aclnnQuantMatmulWeightNz_QuantBatchMatmulV3_QuantBatchMatmulV3``; the
  per-occurrence row geometry stays in the occurrence.
[Invariants] Only the exact R31 deployment identity plus the W8A8 INT8
  WeightNz family selects this adapter.  The observed source storage scale is
  FP32 (a declared BF16 storage is carried as-is), but the deployed ACLNN row
  consumes one BF16 scale per output channel and one FLOAT scale per
  activation row; those are distinct fields.  The weight
  device shape is the active 32/16 NZ tile shape, never a guessed padded
  logical N.  The R31 Python caller and kernel source are not archived, so
  the identity records them as unresolved instead of borrowing R64/R35.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O, device, lock,
  wait or runtime import.
[Side-Effect Policy] Returns immutable metadata only.  No tensor, pointer,
  allocation or native launch is created.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_IMAGE_PREFIX = "9e869e81484"
R31_REVISION = "r31/9e869e8148"
R31_SOURCE_VERSION = "cann-9.2.0-r31"
R31_WRAPPER_VERSION = "vllm-ascend v0.30.0-a5-cann9.2.0-r31"
_ENTRY = "aclnnQuantMatmulWeightNz"
_NZ_N = 32
_NZ_K = 16
_MISSING_CALLER = (
    "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py")
_LIBOPAPI = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "revision": R31_SOURCE_VERSION,
    "state": "task_artifact",
}
_PACKER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/utils.py",
    "sha256": "69c0b69ef9b80c08b772d191e8e66e8857c5c1ea68e719ddc91b1a197f7ee8c3",
    "symbol": "maybe_trans_nz",
    "revision": R31_REVISION,
    "state": "task_artifact",
    "lines": [394, 406],
    "layout_constant": "ACL_FORMAT_FRACTAL_NZ=29",
}
def _digest_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    return text.split(":", 1)[1] if text.startswith("sha256:") else text


def _r31_image_text(compiled: Any) -> str | None:
    """Return the declared R31 deployment identity, never a model label."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    for key in ("image_id", "oci_image_digest", "deployment_image_id",
                "image", "deployment_image", "oci_image", "source_image"):
        text = _digest_text(meta.get(key))
        if text.startswith(R31_IMAGE_PREFIX):
            return text
    return None


def _r31_dims(compiled: Any) -> tuple[int, int] | None:
    operator = getattr(compiled, "operator", None)
    if operator is None or getattr(operator, "family", "") != "gemm":
        return None
    if getattr(compiled, "kind", "") not in {"A0", "F0", "A3"}:
        return None
    theta = operator.theta
    quant = dict(theta.quant)
    if str(quant.get("linear_method", "")) != "w8a8":
        return None
    if str(quant.get("weight_format", "")) != "int8":
        return None
    if str(quant.get("act_format", "")) != "int8":
        return None
    storage = str(quant.get("weight_scale_format", ""))
    runtime = str(quant.get("runtime_weight_scale_format") or "bf16")
    if storage not in {"fp32", "bf16"} or runtime != "bf16":
        return None
    if int(quant.get("scale_group", 0) or 0) != 1:
        return None
    if not bool(quant.get("quantized", 0)):
        return None
    if bool(theta.find("bias", False)):
        return None
    try:
        k, n = int(dict(theta.matrix)["K"]), int(dict(theta.matrix)["N"])
    except (KeyError, TypeError, ValueError):
        return None
    if k < 1 or n < 1 or n % _NZ_N or k % _NZ_K:
        return None
    if _r31_image_text(compiled) is None:
        return None
    return k, n


def _source_record() -> dict:
    return {
        "repository": _LIBOPAPI["repository"],
        "path": _LIBOPAPI["path"],
        "sha256": _LIBOPAPI["sha256"],
        "symbol": _ENTRY,
        "revision": _LIBOPAPI["revision"],
        "state": _LIBOPAPI["state"],
    }


def _typed(k: int, n: int, image: str, storage: str,
           storage_bytes: float) -> dict:
    tile = [n // _NZ_N, k // _NZ_K, _NZ_K, _NZ_N]
    return {
        "dtype": "int8",
        "layout": "ND",
        "shapes": {
            "x": [k], "weight": [k, n],
            "weight_scale": [n], "output": [n],
        },
        "attrs": {
            "matrix_k": k, "matrix_n": n,
            "quantized": 1, "scale_group": 1,
            "linear_method": "w8a8", "implementation": "W8A8_DYNAMIC",
            "act_format": "int8", "weight_format": "int8",
            "output_format": "bf16", "output_dtype": "bf16",
            "weight_storage_dtype": "int8",
            "weight_logical_shape": [k, n],
            "weight_allocation_shape": None,
            "weight_allocation_status": "python_caller_source_unresolved",
            "weight_pre_nz_layout": "unresolved",
            "weight_device_shape": tile,
            "weight_device_layout": "FRACTAL_NZ",
            "weight_transform": "maybe_trans_nz",
            "weight_packer": _PACKER["symbol"],
            "weight_packer_source_path": _PACKER["path"],
            "weight_packer_source_sha256": _PACKER["sha256"],
            "weight_packer_revision": R31_REVISION,
            "weight_scale_storage_format": storage,
            "weight_scale_storage_bytes": storage_bytes,
            "weight_scale_runtime_format": "bf16",
            "weight_scale_runtime_bytes": 2.0,
            "weight_scale_allocation_shape": None,
            "weight_scale_allocation_status": "python_caller_source_unresolved",
            "weight_scale_device_shape": [n],
            "pertoken_scale_format": "fp32",
            "pertoken_scale_runtime_bytes": 4.0,
            "n_padding": "active_32",
            "active_n": n, "padded_n": n,
            "n_padding_tail": 0,
            "deployment_image_id": image,
            "deployment_source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "bias": False, "offset": None,
        },
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real R31 W8A8 WeightNz call spec, or None.

    Selection is the exact R31 deployment identity plus the compiled W8A8
    INT8 GEMM shape.  No model/cell name, R64/R35 source or measured timing
    chooses this branch.  The returned ``_CallSpec`` deliberately keeps the
    missing R31 Python caller/kernel source in ``unresolved``.
    """
    del phase
    dims = _r31_dims(compiled)
    if dims is None:
        return None
    k, n = dims
    image = _r31_image_text(compiled) or R31_IMAGE_SHA
    quant = dict(compiled.operator.theta.quant)
    storage = str(quant.get("weight_scale_format") or "fp32")
    storage_bytes = 4.0 if storage == "fp32" else 2.0
    from .native_calls import _api, _call_spec, _kernel

    return _call_spec(
        _api("aclnn", _ENTRY, aclnn_symbol=_ENTRY,
             version=R31_SOURCE_VERSION),
        _source_record(),
        _typed(k, n, R31_IMAGE_SHA if image.startswith(R31_IMAGE_PREFIX)
               else image, storage, storage_bytes),
        (_kernel("QuantBatchMatmulV3", "mix", evidence="measured"),),
        (),
        getattr(compiled.operator, "name", "gemm"),
        execution_core="mix",
        unresolved=(
            "python_caller_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
            "attributable_source_reference_unresolved",
        ),
    )


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _actual(binding: Any, k: int, n: int, m: int) -> dict:
    attrs = dict(binding.typed.get("attrs") or {})
    actual_attrs = {
        **attrs,
        "m": m, "tokens": m, "k": k, "n": n,
        "weight_scale_device_shape": [n],
        "pertoken_scale_shape": [m],
        "output_shape": [m, n],
        "active_n": n, "padded_n": n, "n_padding_tail": 0,
    }
    return {
        "dtype": binding.typed.get("dtype"),
        "layout": binding.typed.get("layout"),
        "shapes": {
            "x": [m, k], "weight": [k, n],
            "weight_scale": [n], "pertoken_scale": [m],
            "output": [m, n],
        },
        "attrs": actual_attrs,
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare one R31 W8A8 WeightNz occurrence for the evaluated geometry.

    ``m`` is the only dynamic tensor axis.  K/N, the NZ tile shape and both
    scale formats stay the static identity.  The envelope mirrors
    ``native_qbmm.dynamic_qbmm`` but never claims a complete caller: the R31
    Python caller and kernel source remain explicit partial reasons.
    """
    api = binding.api or {}
    if api.get("entry") != _ENTRY and api.get("aclnn_symbol") != _ENTRY:
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("implementation") != "W8A8_DYNAMIC":
        return None
    image = str(attrs.get("deployment_image_id") or "")
    if not image.startswith(R31_IMAGE_PREFIX):
        return None
    shapes = dict(binding.typed.get("shapes") or {})
    weight = shapes.get("weight")
    if not weight or len(weight) != 2:
        return None
    k, n = int(weight[0]), int(weight[1])
    if k < 1 or n < 1 or n % _NZ_N or k % _NZ_K:
        return None
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1:
        return None
    operator = getattr(compiled, "operator", None)
    if operator is not None and hasattr(operator, "theta"):
        try:
            matrix = dict(operator.theta.matrix)
            if int(matrix["K"]) != k or int(matrix["N"]) != n:
                return None
        except (KeyError, TypeError, ValueError):
            return None
    actual = _actual(binding, k, n, m)
    validated = validate(binding, actual)
    tile = [n // _NZ_N, k // _NZ_K, _NZ_K, _NZ_N]
    descriptors = {
        "x": _descriptor([m, k], "int8", "ND", "input"),
        "weight": _descriptor([k, n], "int8", "ND", "input"),
        "weight_scale": _descriptor([n], "bf16", "ND", "input"),
        "pertoken_scale": _descriptor([m], "fp32", "ND", "input"),
        "output": _descriptor([m, n], "bf16", "ND", "output"),
        "weight_nz": _descriptor(tile, "int8", "FRACTAL_NZ", "input"),
    }
    descriptors["weight"].update({
        "storage_dtype": "int8", "allocation_shape": None,
        "allocation_status": "python_caller_source_unresolved",
        "logical_shape": [k, n], "device_shape": tile,
        "device_layout": "FRACTAL_NZ",
    })
    descriptors["weight_scale"].update({
        "storage_format": attrs.get("weight_scale_storage_format", "fp32"),
        "storage_bytes": attrs.get("weight_scale_storage_bytes", 4.0),
        "runtime_format": "bf16", "runtime_bytes": 2.0,
        "allocation_shape": None,
        "allocation_status": "python_caller_source_unresolved",
        "device_shape": [n],
    })
    descriptors["pertoken_scale"].update({
        "runtime_format": "fp32", "runtime_bytes": 4.0,
        "producer": "unresolved",
        "producer_status": "python_caller_source_unresolved",
    })
    source = dict(binding.source or {})
    kernel = dict(binding.kernels[0]) if binding.kernels else {
        "name": "QuantBatchMatmulV3", "role": "mix",
        "evidence": "measured",
    }
    kernel.update({
        "observed_task_type": "MIX_AIC",
        "owner_binary_path": _LIBOPAPI["path"],
        "owner_binary_sha256": _LIBOPAPI["sha256"],
        "source_status": "runtime_binary_only",
    })
    prepare = {
        "schema": "mindie-sim/qbmm-native-prepare/1",
        "call": _ENTRY,
        "native_entry": _ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "implementation": "W8A8_DYNAMIC",
        "source": source,
        "source_version": R31_SOURCE_VERSION,
        "wrapper_version": R31_WRAPPER_VERSION,
        "selected_host_variant": None,
        "selected_host_variant_status": "unresolved",
        "missing_source": _MISSING_CALLER,
        "kernel": kernel,
        "source_lines": {
            "caller": None, "kernel": None,
            "packer": list(_PACKER["lines"]),
            "packer_layout_constant": _PACKER["layout_constant"],
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {
            "x": "x", "weight": "weight",
            "weight_scale": "weight_scale",
            "pertoken_scale": "pertoken_scale",
            "bias": None, "offset": None,
            "output_dtype": "bf16", "output": "output",
        },
        "argument_order": ["x", "weight", "weight_scale", "pertoken_scale"],
        "observed_acl_tensor_order": [
            "x", "weight", "weight_scale", "pertoken_scale"],
        "observed_acl_tensor_dtypes": [
            "INT8", "INT8", "DT_BF16", "FLOAT"],
        "observed_acl_tensor_formats": [
            "ND", "FRACTAL_NZ", "ND", "ND"],
        "observed_output": {
            "dtype": "DT_BF16", "format": "ND", "shape": [m, n]},
        "optional_slots": {
            "bias": "not_present", "offset": "not_passed",
            "output_dtype": "bf16",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "weight_nz": tile,
        "weight_storage": {
            "logical_shape": [k, n], "allocation_shape": None,
            "allocation_status": "python_caller_source_unresolved",
            "allocation_dtype": "int8", "device_shape": tile,
            "device_layout": "FRACTAL_NZ",
            "transform": "maybe_trans_nz",
            "pre_nz_layout": "unresolved",
            "packer": dict(_PACKER),
        },
        "scale_storage": {
            "weight_scale": {
                "storage_format": attrs.get(
                    "weight_scale_storage_format", "fp32"),
                "storage_bytes": attrs.get("weight_scale_storage_bytes", 4.0),
                "runtime_format": "bf16", "runtime_bytes": 2.0,
                "allocation_shape": None,
                "allocation_status": "python_caller_source_unresolved",
                "device_shape": [n],
            },
            "pertoken_scale": {
                "runtime_format": "fp32", "runtime_bytes": 4.0,
                "producer": "unresolved",
                "producer_status": "python_caller_source_unresolved",
                "shape": [m],
            },
        },
        "deployment": {
            "image_id": R31_IMAGE_SHA,
            "revision": R31_REVISION,
            "wrapper_version": R31_WRAPPER_VERSION,
            "selection": (
                "R31 image identity + compiled linear_method=w8a8, "
                "int8/int8, scale_group=1, "
                f"{attrs.get('weight_scale_storage_format', 'fp32')} "
                "stored weight scale"),
            "source_path": _LIBOPAPI["path"],
            "source_sha256": _LIBOPAPI["sha256"],
            "packer_path": _PACKER["path"],
            "packer_sha256": _PACKER["sha256"],
            "caller_path": _MISSING_CALLER,
            "caller_sha256": None,
            "caller_status": "unresolved",
            "kernel_source_path": None,
            "kernel_source_sha256": None,
            "kernel_source_status": "runtime_binary_only",
        },
        "match": {
            "native_type": "QuantBatchMatmulV3",
            "entry": _ENTRY,
            "call_id": binding.call_id,
            "deployment_image_id": R31_IMAGE_SHA,
            "source_revision": R31_REVISION,
            "source_version": R31_SOURCE_VERSION,
            "implementation": "W8A8_DYNAMIC",
            "k": k, "n": n,
            "active_n": n, "padded_n": n, "n_padding": "active_32",
            "weight_layout": "FRACTAL_NZ",
            "weight_allocation_shape": None,
            "weight_allocation_status": "python_caller_source_unresolved",
            "weight_device_shape": tile,
            "activation_dtype": "int8", "weight_dtype": "int8",
            "weight_scale_storage_format": attrs.get(
                "weight_scale_storage_format", "fp32"),
            "weight_scale_runtime_format": "bf16",
            "pertoken_scale_format": "fp32",
            "output_dtype": "bf16",
            "scale_group": 1, "quantized": 1,
            "source_sha256": _LIBOPAPI["sha256"],
            "packer_sha256": _PACKER["sha256"],
            "caller_sha256": None,
            "kernel_sha256": None,
            "complete": False, "partial": True,
        },
        "binary_selection": {
            "status": "runtime_binary_only",
            "reason": "r31_python_caller_and_kernel_source_not_archived",
            "owner_kernel": "QuantBatchMatmulV3",
            "runtime_library": _LIBOPAPI["path"],
            "runtime_library_sha256": _LIBOPAPI["sha256"],
            "acl_symbol": _ENTRY,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "python_caller_source_unresolved",
            "kernel_source_runtime_binary_only",
            "acl_binary_selection_trace_only",
        ],
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_physical_completion",
        ],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {
                "dtype": validated.get("dtype"),
                "layout": validated.get("layout"),
                "shapes": validated.get("shapes"),
                "attrs": validated.get("attrs"),
            }}
