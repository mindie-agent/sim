"""Narrow executable adapter for one prepared standalone QBMM occurrence.

[Role] Validate a normalized R64/R35 standalone MXFP8 QMM occurrence and
  invoke exactly ``torch_npu.npu_quant_matmul`` once when the caller supplies
  the tensors and an explicit device (or an injected dispatcher for an
  off-device argument test).
[Invariants] No import-time torch/NPU dependency, no default device, no
  string evaluation, no descriptor rewriting and no prediction-path effect.
  Unknown sources/schemas stay unavailable.
[Timeouts & Locks] One dispatcher call; no waits or locks in ordinary analysis.
[Side-Effect Policy] The caller owns tensors/output; this module allocates
  nothing except what the selected native dispatcher returns.
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Mapping

from .native_calls import validate_native_occurrence
from .native_qbmm import QBMM_SOURCES

ENTRY = "torch_npu.npu_quant_matmul"
SCHEMA = "mindie-sim/qbmm-native-prepare/1"
_SOURCE_SHAS = {item["wrapper"]["sha256"] for item in QBMM_SOURCES.values()}


class NativeInvocationUnavailable(RuntimeError):
    """The prepared occurrence is not executable in this environment."""


def prepared_qbmm_plan(occurrence: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the prepared record and return the exact dispatch plan."""
    if not isinstance(occurrence, Mapping):
        raise NativeInvocationUnavailable("occurrence_not_a_mapping")
    prepared = occurrence.get("prepare")
    call = occurrence.get("native_call")
    args = occurrence.get("prepare_args")
    if not isinstance(prepared, Mapping) or prepared.get("schema") != SCHEMA:
        raise NativeInvocationUnavailable("prepared_qbmm_schema_missing")
    if not isinstance(call, Mapping) or not isinstance(args, Mapping):
        raise NativeInvocationUnavailable("prepared_native_call_missing")
    api, source = call.get("api") or {}, call.get("source") or {}
    if api.get("entry") != ENTRY:
        raise NativeInvocationUnavailable("prepared_entry_not_supported")
    if source.get("sha256") not in _SOURCE_SHAS:
        raise NativeInvocationUnavailable("prepared_source_not_supported")
    if prepared.get("source", {}).get("sha256") != source.get("sha256"):
        raise NativeInvocationUnavailable("prepared_source_disagrees")
    descriptors = prepared.get("descriptors") or {}
    required = ("x", "weight", "weight_scale", "pertoken_scale", "output")
    if any(name not in descriptors for name in required):
        raise NativeInvocationUnavailable("prepared_descriptor_missing")
    binding = SimpleNamespace(
        typed=call.get("typed") or {}, call_id=call.get("call_id"))
    try:
        validate_native_occurrence(binding, args)
    except Exception as error:  # normalized validation is the public gate
        raise NativeInvocationUnavailable(
            f"prepared_typed_mismatch:{type(error).__name__}:{error}") from error
    keywords = prepared.get("keywords") or {}
    return {
        "entry": ENTRY, "source_sha256": source.get("sha256"),
        "descriptors": descriptors,
        "output": descriptors["output"],
        "input_names": [name for name in
                        ("x", "weight", "weight_scale", "pertoken_scale", "bias")
                        if name in descriptors
                        and descriptors[name].get("present") is not False],
        "optional_nulls": [name for name, item in descriptors.items()
                           if item.get("present") is False],
        "group_sizes": list(keywords.get("group_sizes") or [1, 1, 32]),
        "scale_dtype": keywords.get("scale_dtype") or "fp8_e8m0",
        "pertoken_scale_dtype": keywords.get("pertoken_scale_dtype") or "fp8_e8m0",
        "output_dtype": keywords.get("output_dtype") or "bf16",
    }


def _torch_module():
    try:
        import torch  # local, deliberately not an import-time dependency
    except ImportError as error:
        raise NativeInvocationUnavailable("torch_unavailable") from error
    return torch


def _torch_dtype(torch: Any, name: str | None) -> Any:
    table = {
        "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
        "fp32": torch.float32, "float32": torch.float32,
        "fp16": torch.float16, "float16": torch.float16,
        "int32": torch.int32, "int64": torch.int64, "uint8": torch.uint8,
        "fp8_e4m3": torch.float8_e4m3fn,
    }
    if name == "fp8_e8m0":
        return torch.uint8  # storage dtype; scale_dtype carries interpretation
    if name not in table:
        raise NativeInvocationUnavailable(f"descriptor_dtype_unsupported:{name}")
    return table[name]


def _check_tensor(torch: Any, name: str, tensor: Any, descriptor: Mapping[str, Any]):
    if not torch.is_tensor(tensor):
        raise NativeInvocationUnavailable(f"tensor_required:{name}")
    expected = descriptor.get("shape")
    device_shape = descriptor.get("device_shape")
    if descriptor.get("device_layout") == "FRACTAL_NZ" and device_shape:
        if list(tensor.shape) not in (list(expected), list(device_shape)):
            raise NativeInvocationUnavailable(f"tensor_shape_mismatch:{name}")
    elif list(tensor.shape) != list(expected):
        raise NativeInvocationUnavailable(f"tensor_shape_mismatch:{name}")
    expected_dtype = _torch_dtype(torch, descriptor.get("dtype"))
    if descriptor.get("dtype") == "fp8_e8m0":
        if tensor.dtype not in {torch.uint8, getattr(torch, "float8_e8m0fnu", torch.uint8)}:
            raise NativeInvocationUnavailable(f"tensor_dtype_mismatch:{name}")
    elif tensor.dtype != expected_dtype:
        raise NativeInvocationUnavailable(f"tensor_dtype_mismatch:{name}")


def invoke_qbmm(occurrence: Mapping[str, Any], tensors: Mapping[str, Any], *,
                device: str | None = None, dispatcher: Any = None) -> Any:
    """Invoke the prepared native QMM once on the caller-selected device."""
    plan = prepared_qbmm_plan(occurrence)
    if dispatcher is None and not device:
        raise NativeInvocationUnavailable("explicit_device_required")
    torch = _torch_module()
    descriptors = plan["descriptors"]
    for name in plan["input_names"]:
        if name not in tensors:
            raise NativeInvocationUnavailable(f"tensor_missing:{name}")
        _check_tensor(torch, name, tensors[name], descriptors[name])
    for name in plan["optional_nulls"]:
        if tensors.get(name) is not None:
            raise NativeInvocationUnavailable(f"optional_null_expected:{name}")
    if dispatcher is None:
        try:
            import torch_npu  # lazy optional runtime dependency
        except ImportError as error:
            raise NativeInvocationUnavailable("torch_npu_unavailable") from error
        scale_dtype = getattr(torch_npu, "float8_e8m0fnu", None)
        if scale_dtype is None:
            raise NativeInvocationUnavailable("torch_npu_e8m0_dtype_unavailable")
        target = torch.device(device)
        for name in plan["input_names"]:
            if tensors[name].device != target:
                raise NativeInvocationUnavailable(f"tensor_device_mismatch:{name}")
        dispatch = torch_npu.npu_quant_matmul
    else:
        scale_dtype = "fp8_e8m0"
        dispatch = dispatcher
    kwargs = {
        "x1": tensors["x"], "x2": tensors["weight"],
        "scale": tensors["weight_scale"],
        "offset": None,
        "pertoken_scale": tensors["pertoken_scale"],
        "bias": tensors.get("bias"),
        "output_dtype": _torch_dtype(torch, plan["output_dtype"]),
        "x1_dtype": None, "x2_dtype": None,
        "pertoken_scale_dtype": scale_dtype,
        "scale_dtype": scale_dtype,
        "group_sizes": plan["group_sizes"],
        "y_scale": None,
    }
    result = dispatch(**kwargs)
    output = plan["output"]
    if not torch.is_tensor(result):
        raise NativeInvocationUnavailable("dispatcher_returned_non_tensor")
    if list(result.shape) != list(output.get("shape") or ()):
        raise NativeInvocationUnavailable("dispatcher_output_shape_mismatch")
    if result.dtype != _torch_dtype(torch, output.get("dtype")):
        raise NativeInvocationUnavailable("dispatcher_output_dtype_mismatch")
    return result
