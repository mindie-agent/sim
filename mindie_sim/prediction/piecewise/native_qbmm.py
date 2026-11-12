"""Prepare-only standalone QuantBatchMatmulV3 invocation.

[Role] Expand the selected R64/R35 W8A8-MXFP8 linear wrapper's standalone
  ``torch_npu.npu_quant_matmul`` call into real operand/scale/bias/output
  descriptors for the evaluated token geometry.
[Invariants] Only the source-selected MXFP8 FP8/E8M0 group-32 branch is
  expanded.  Static identity (K/N, dtypes, layout transform, implementation)
  stays in the binding; M, per-token scale rows and output rows live only in
  the occurrence.  No tensor is allocated, no pointer is fabricated and no
  numeric prediction is changed.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing selection and
  source call arguments.
"""
from __future__ import annotations

from typing import Any

R35_IMAGE_SHA = "5461a1a8253"
_R64_WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
    "sha256": "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79",
    "symbol": "AscendW8A8MXFP8DynamicLinearMethod.apply",
    "revision": "r64/8d422528",
    "state": "task_artifact",
}
_R35_WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py",
    "sha256": "51ece928e77420a7f6600e2d87c46e9397a5140b6acee324aa75fdb97266d009",
    "symbol": "AscendW8A8MXFP8DynamicLinearMethod.apply",
    "revision": "r35/5461a1a8",
    "state": "task_artifact",
}
_R64_KERNEL = {
    "repository": "cann",
    "path": "op_kernel/quant_batch_matmul_v3/arch35/quant_batch_matmul_v3.cpp",
    "sha256": "13daf996890d70e7b1dc9275ab3a464043860e67d6df730e28742393c023ec88",
    "symbol": "QuantBatchMatmulV3",
    "revision": "cann-9.2.1-r64",
    "state": "task_artifact",
    "evidence": "source_declared",
    "source_status": "kernel_source",
}
_R35_KERNEL = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": "QuantBatchMatmulV3",
    "revision": "cann-9.2.0-r35",
    "state": "task_artifact",
    "evidence": "runtime_binary",
    "source_status": "runtime_binary_only",
}
QBMM_SOURCES = {
    "r64": {
        "wrapper": _R64_WRAPPER, "kernel": _R64_KERNEL,
        "version": "r64/8d422528", "image_sha": "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e",
        "source_lines": {"qmm_call": [143, 153],
                         "scale_allocation": [83, 83],
                         "weight_transform": [160, 230]},
        "selected_host_variant": "apply", "rejected_host_variants": [],
    },
    "r35": {
        "wrapper": _R35_WRAPPER, "kernel": _R35_KERNEL,
        "version": "r35/5461a1a8", "image_sha": R35_IMAGE_SHA,
        "source_lines": {"qmm_call": [150, 160],
                         "scale_allocation": [89, 89],
                         "weight_transform": [223, 305],
                         "alternate_gmm": [167, 221]},
        "selected_host_variant": "apply",
        "rejected_host_variants": ["apply_single_group_gmm:GroupedMatmulV3"],
    },
}
QBMM_SOURCE = QBMM_SOURCES["r64"]["kernel"]
QBMM_WRAPPER_SOURCE = QBMM_SOURCES["r64"]["wrapper"]
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _descriptor(shape: list[int], dtype: str, layout: str,
                role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def dynamic_qbmm(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Full prepare-only call for the selected standalone MXFP8 QBMM."""
    api = binding.api or {}
    if api.get("entry") != "torch_npu.npu_quant_matmul":
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if attrs.get("implementation") != "W8A8_MXFP8":
        return None
    selected_source = str(attrs.get("deployment_source_revision") or "")
    source = QBMM_SOURCES.get(selected_source)
    if source is None:
        return None
    shapes = dict(binding.typed.get("shapes") or {})
    if "weight" not in shapes:
        return None
    k, n = (int(item) for item in shapes["weight"])
    m = int(getattr(geometry, "effective_tokens", 0) or 0)
    if m < 1 or k < 1 or n < 1:
        return None
    group = int(attrs.get("group_size", attrs.get("scale_group", 32)) or 32)
    if group < 1:
        return None
    pairs = _ceil_div(_ceil_div(k, group), 2)
    cache_key = (binding.call_id, m)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    act_dtype = str(attrs.get("act_format") or "fp8_e4m3")
    weight_dtype = str(attrs.get("weight_format") or "fp8_e4m3")
    output_dtype = str(attrs.get("output_dtype") or "bf16")
    bias_present = bool(attrs.get("bias"))
    actual_shapes = {
        "x": [m, k], "weight": [k, n],
        "weight_scale": [pairs, n, 2],
        "pertoken_scale": [m, pairs, 2],
        "output": [m, n], "group_sizes": [3],
    }
    if bias_present:
        actual_shapes["bias"] = [n]
    weight_nz = None
    if n % 32 == 0 and k % 16 == 0:
        weight_nz = [n // 32, k // 16, 16, 32]
    descriptors = {
        "x": _descriptor([m, k], act_dtype, "ND", "input"),
        "weight": _descriptor([k, n], weight_dtype, "ND", "input"),
        "weight_scale": _descriptor([pairs, n, 2], "uint8", "ND", "input"),
        "pertoken_scale": _descriptor([m, pairs, 2], "fp8_e8m0", "ND", "input"),
        "output": _descriptor([m, n], output_dtype, "ND", "output"),
        "group_sizes": _descriptor([3], "int64", "ND", "host"),
    }
    descriptors["weight"].update({
        "storage_dtype": weight_dtype, "allocation_shape": [n, k],
        "logical_shape": [k, n], "device_shape": weight_nz,
        "device_layout": "FRACTAL_NZ" if weight_nz else None,
    })
    descriptors["weight_scale"].update({
        "storage_dtype": "uint8", "interpret_dtype": "fp8_e8m0",
        "allocation_shape": [n, (k + group - 1) // group],
        "device_shape": [pairs, n, 2],
    })
    descriptors["pertoken_scale"].update({
        "storage_dtype": "producer_output", "interpret_dtype": "fp8_e8m0",
        "device_shape": [m, pairs, 2],
    })
    if bias_present:
        descriptors["bias"] = _descriptor([n], "fp32", "ND", "input")
    if weight_nz:
        descriptors["weight_nz"] = _descriptor(weight_nz, weight_dtype,
                                                "FRACTAL_NZ", "input")
    producer_inputs = {
        "x_original": _descriptor(
            [m, k], str(attrs.get("prequant_format") or "bf16"), "ND",
            "producer_input"),
    }
    actual_attrs = {
        **attrs, "m": m, "k": k, "n": n,
        "group_sizes": [1, 1, group], "scale_dtype": "fp8_e8m0",
        "pertoken_scale_dtype": "fp8_e8m0",
        "output_dtype": output_dtype,
        "weight_format": weight_dtype, "act_format": act_dtype,
        "linear_method": "mx_fp8", "implementation": "W8A8_MXFP8",
        "weight_transform": "transpose_0_1+maybe_trans_nz",
        "weight_storage_dtype": weight_dtype,
        "weight_allocation_shape": [n, k],
        "weight_logical_shape": [k, n],
        "weight_device_layout": "FRACTAL_NZ" if weight_nz else None,
        "weight_device_shape": weight_nz,
        "weight_scale_storage_dtype": "uint8",
        "weight_scale_interpret_dtype": "fp8_e8m0",
        "weight_scale_allocation_shape": [n, (k + group - 1) // group],
        "weight_scale_device_shape": [pairs, n, 2],
        "bias": bias_present, "offset": None,
    }
    actual = {"dtype": binding.typed.get("dtype"),
              "layout": binding.typed.get("layout"),
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    arguments = {
        "x": "x", "weight": "weight", "weight_scale": "weight_scale",
        "scale_dtype": "fp8_e8m0", "pertoken_scale": "pertoken_scale",
        "pertoken_scale_dtype": "fp8_e8m0",
        "bias": "bias" if bias_present else None,
        "offset": None, "output_dtype": output_dtype,
        "group_sizes": [1, 1, group],
    }
    prepare = {
        "schema": "mindie-sim/qbmm-native-prepare/1",
        "call": "torch_npu.npu_quant_matmul",
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "implementation": "W8A8_MXFP8",
        "source": source["wrapper"],
        "kernel": source["kernel"],
        "source_version": source["version"],
        "source_lines": source["source_lines"],
        "selected_host_variant": source["selected_host_variant"],
        "rejected_host_variants": list(source["rejected_host_variants"]),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "prototype": ("torch_npu.npu_quant_matmul(x1, x2, scale, *, "
                      "offset=None, pertoken_scale=None, bias=None, "
                      "output_dtype=None, x1_dtype=None, x2_dtype=None, "
                      "pertoken_scale_dtype=None, scale_dtype=None, "
                      "group_sizes=None, y_scale=None)"),
        "argument_order": ["x1", "x2", "scale", "offset", "pertoken_scale",
                           "bias", "output_dtype", "x1_dtype", "x2_dtype",
                           "pertoken_scale_dtype", "scale_dtype",
                           "group_sizes", "y_scale"],
        "positional_arguments": {"x1": "x", "x2": "weight",
                                 "scale": "weight_scale"},
        "keywords": {
            "offset": None, "pertoken_scale": "pertoken_scale",
            "bias": "bias" if bias_present else None,
            "output_dtype": output_dtype, "x1_dtype": None, "x2_dtype": None,
            "pertoken_scale_dtype": "fp8_e8m0",
            "scale_dtype": "fp8_e8m0", "group_sizes": [1, 1, group],
            "y_scale": None,
        },
        "optional_slots": {
            "bias": "present" if bias_present else "not_present",
            "offset": "not_passed", "x1_dtype": "not_set",
            "x2_dtype": "not_set", "y_scale": "not_passed",
            "scale_dtype": "fp8_e8m0", "pertoken_scale_dtype": "fp8_e8m0",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "producer_inputs": producer_inputs,
        "weight_nz": weight_nz,
        "weight_storage": {
            "allocation_shape": [n, k], "allocation_dtype": weight_dtype,
            "logical_shape": [k, n], "device_shape": weight_nz,
            "device_layout": "FRACTAL_NZ" if weight_nz else None,
        },
        "scale_storage": {
            "weight_scale": {
                "allocation_shape": [n, (k + group - 1) // group],
                "allocation_dtype": "uint8", "device_shape": [pairs, n, 2],
                "interpret_dtype": "fp8_e8m0",
                "scale_dtype": "fp8_e8m0",
            },
            "pertoken_scale": {
                "device_shape": [m, pairs, 2],
                "interpret_dtype": "fp8_e8m0",
                "storage_dtype": "producer_output",
            },
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "revision": selected_source,
            "selection": "FeatureConfig.metadata.image_id",
            "selected_by": "quant.linear_method=mx_fp8; ordinary linear apply",
            "source_path": source["wrapper"]["path"],
            "source_version": source["version"],
            "source_sha256": source["wrapper"]["sha256"],
            "kernel_sha256": source["kernel"]["sha256"],
        },
        "match": {
            "native_type": "QuantBatchMatmulV3",
            "entry": "torch_npu.npu_quant_matmul",
            "call_id": binding.call_id,
            "deployment_image_id": str(attrs.get("deployment_image_id") or "unresolved"),
            "source_revision": selected_source,
            "source_version": source["version"],
            "implementation": "W8A8_MXFP8",
            "selected_host_variant": source["selected_host_variant"],
            "k": k, "n": n, "weight_layout": "FRACTAL_NZ",
            "activation_dtype": act_dtype, "weight_dtype": weight_dtype,
            "scale_dtype": "fp8_e8m0", "output_dtype": output_dtype,
            "group_size": group,
            "source_sha256": source["wrapper"]["sha256"],
            "kernel_sha256": source["kernel"]["sha256"],
            "source_lines": source["source_lines"],
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "acl_nn_binary_name_not_exported_by_python_wrapper",
            "owner_kernel": "QuantBatchMatmulV3",
        },
        "non_claims": ["no_runtime_allocation", "no_device_launch",
                       "no_measured_duration"],
    }
    result = {"actual": validated, "prepare": prepare,
              "prepare_args": {"dtype": validated.get("dtype"),
                               "layout": validated.get("layout"),
                               "shapes": validated.get("shapes"),
                               "attrs": validated.get("attrs")}}
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
