"""Exact callable construction, tensor metadata and FP32-CPU reference checks.

[Role]
  Build the three exact operator callables used by the timing lane, produce the
  declared deterministic CPU source tensors and mutation tensors, capture actual
  device allocation metadata, and compare sampled native outputs against FP32
  CPU references.  Torch is supplied by the caller; importing this module does
  not import torch.

[Invariants]
  * RMSNorm calls ``torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)``.  SwiGLU
    calls ``torch_npu.npu_swiglu(packed, -1)``.  Copy calls
    ``target.copy_(source)``.  Nothing else is in these callables.
  * RMSNorm/SwiGLU use their installed default outputs; no ``.out`` overload is
    constructed or claimed.  Copy uses one same preallocated destination.
  * CPU sources are deterministic and identical across all mode/N conditions of
    a case; their bytes are hashed and recorded by the runtime.
  * Comparison is read-only.  NaNs/infinities are reported, never repaired, and
    returned RMS ``rstd`` is checked when the installed tuple makes it feasible.
  * Reference transfer and comparison happen outside all timing blocks.

[Timeouts & Locks]
  No locks, threads, timers or retries.  The caller owns budget checks around
  construction, device transfers, synchronizes and comparison.

[Side-Effect Policy]
  Allocates CPU/NPU tensors and enqueues the caller's intended H2D transfers only
  when invoked by the device runtime.  Performs no file, network or lease action.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

from .a3_operator_graph_plan import CALLABLE_SOURCE


@dataclass
class ConditionTensors:
    """Static buffers and exact callable for one condition."""

    callable: Any
    static_cpu: Any
    static_device: Any
    mutation_cpu: Any
    data_sha256: str
    data_hash_method: str
    mutation_sha256: str
    data_contract: dict[str, Any]
    gamma_cpu: Any = None
    gamma_device: Any = None
    target_device: Any = None
    gamma_sha256: str | None = None


def _bf16(torch: Any, values: Any) -> Any:
    return values.to(torch.bfloat16)


def _generic_values(torch: Any, shape: list[int], low: float, high: float,
                    seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.empty(shape, dtype=torch.float32).uniform_(
        low, high, generator=generator)


def _uint8_values(torch: Any, shape: list[int], seed: int) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)


def tensor_sha256(torch: Any, tensor: Any) -> tuple[str, str]:
    """Hash the exact contiguous CPU bytes; never mutate the tensor."""
    contiguous = tensor.detach().contiguous().cpu()
    try:
        payload = contiguous.view(torch.uint8).numpy().tobytes()
        method = "raw_bytes_uint8_view"
    except Exception:
        payload = contiguous.to(torch.float32).numpy().tobytes()
        method = "float32_reinterpretation_fallback"
    return hashlib.sha256(payload).hexdigest(), method


def build_condition_tensors(torch: Any, torch_npu: Any, device: Any,
                            condition: Mapping[str, Any]) -> ConditionTensors:
    """Create deterministic static buffers and the exact no-argument callable."""
    case = condition
    operator = case["operator"]
    shape = case["shape"]
    data = case["data_contract"]
    seed = int(data["seed"])
    mutation_seed = int(data["mutation_seed"])
    if operator == "rms_norm":
        tokens, hidden = shape["tokens"], shape["hidden"]
        static_cpu = _bf16(torch, _generic_values(
            torch, [tokens, hidden], data["input"]["low"],
            data["input"]["high"], seed))
        mutation_cpu = _bf16(torch, _generic_values(
            torch, [tokens, hidden], data["input"]["low"],
            data["input"]["high"], mutation_seed))
        gamma_cpu = _bf16(torch, _generic_values(
            torch, [hidden], data["gamma"]["low"], data["gamma"]["high"],
            seed ^ 0x9E37))
        static_device = static_cpu.to(device=device)
        gamma_device = gamma_cpu.to(device=device)
        mutation_cpu = mutation_cpu.contiguous()

        def call() -> Any:
            return torch_npu.npu_rms_norm(
                static_device, gamma_device, epsilon=1e-6)

        digest, method = tensor_sha256(torch, static_cpu)
        gamma_digest, _ = tensor_sha256(torch, gamma_cpu)
        mutation_digest, _ = tensor_sha256(torch, mutation_cpu)
        return ConditionTensors(
            callable=call,
            static_cpu=static_cpu,
            static_device=static_device,
            mutation_cpu=mutation_cpu,
            data_sha256=digest,
            data_hash_method=method,
            mutation_sha256=mutation_digest,
            data_contract={
                "operator": operator,
                "callable_source": CALLABLE_SOURCE[operator],
                "device_transfer": "contiguous H2D on the explicit measurement stream "
                                   "before capture and timing",
                "input_read_only_during_timing": True,
                "mutation_seed": mutation_seed,
                "mutation_in_place_preserves_address": True,
                "restore_in_place_preserves_address": True,
            },
            gamma_cpu=gamma_cpu,
            gamma_device=gamma_device,
            gamma_sha256=gamma_digest,
        )
    if operator == "swiglu":
        tokens, ffn = shape["tokens"], shape["ffn"]
        static_cpu = _bf16(torch, _generic_values(
            torch, [tokens, 2 * ffn], data["input"]["low"],
            data["input"]["high"], seed))
        mutation_cpu = _bf16(torch, _generic_values(
            torch, [tokens, 2 * ffn], data["input"]["low"],
            data["input"]["high"], mutation_seed))
        static_device = static_cpu.to(device=device)
        mutation_cpu = mutation_cpu.contiguous()

        def call() -> Any:
            return torch_npu.npu_swiglu(static_device, -1)

        digest, method = tensor_sha256(torch, static_cpu)
        mutation_digest, _ = tensor_sha256(torch, mutation_cpu)
        return ConditionTensors(
            callable=call,
            static_cpu=static_cpu,
            static_device=static_device,
            mutation_cpu=mutation_cpu,
            data_sha256=digest,
            data_hash_method=method,
            mutation_sha256=mutation_digest,
            data_contract={
                "operator": operator,
                "callable_source": CALLABLE_SOURCE[operator],
                "packed_last_dim": 2 * ffn,
                "device_transfer": "contiguous H2D on the explicit measurement stream "
                                   "before capture and timing",
                "input_read_only_during_timing": True,
                "mutation_seed": mutation_seed,
                "mutation_in_place_preserves_address": True,
                "restore_in_place_preserves_address": True,
            },
        )
    if operator == "copy":
        rows, row_bytes = shape["rows"], shape["row_bytes"]
        static_cpu = _uint8_values(torch, [rows, row_bytes], seed)
        mutation_cpu = _uint8_values(torch, [rows, row_bytes], mutation_seed)
        static_device = static_cpu.to(device=device)
        target_device = torch.empty([rows, row_bytes], dtype=torch.uint8,
                                    device=device)
        mutation_cpu = mutation_cpu.contiguous()

        def call() -> Any:
            return target_device.copy_(static_device)

        digest, method = tensor_sha256(torch, static_cpu)
        mutation_digest, _ = tensor_sha256(torch, mutation_cpu)
        return ConditionTensors(
            callable=call,
            static_cpu=static_cpu,
            static_device=static_device,
            mutation_cpu=mutation_cpu,
            data_sha256=digest,
            data_hash_method=method,
            mutation_sha256=mutation_digest,
            data_contract={
                "operator": operator,
                "callable_source": CALLABLE_SOURCE[operator],
                "device_transfer": "contiguous H2D on the explicit measurement stream "
                                   "before capture and timing",
                "source_read_only_during_timing": True,
                "destination": "one_same_preallocated_target",
                "mutation_seed": mutation_seed,
                "mutation_in_place_preserves_address": True,
                "restore_in_place_preserves_address": True,
            },
            target_device=target_device,
        )
    raise ValueError(f"unsupported operator {operator!r}")


def tensor_facts(tensor: Any) -> dict[str, Any]:
    """Actual shape/dtype/layout/strides/address metadata, without values."""
    dtype = str(tensor.dtype).replace("torch.", "")
    itemsize = int(tensor.element_size())
    numel = int(tensor.numel())
    return {
        "shape": [int(value) for value in tensor.shape],
        "strides": [int(value) for value in tensor.stride()],
        "dtype": dtype,
        "itemsize": itemsize,
        "numel": numel,
        "nbytes": numel * itemsize,
        "data_ptr": int(tensor.data_ptr()),
        "device": str(tensor.device),
        "layout": "contiguous" if bool(tensor.is_contiguous()) else "strided",
    }


def result_tensors(torch: Any, result: Any) -> list[Any]:
    if isinstance(result, torch.Tensor):
        return [result]
    if isinstance(result, (tuple, list)):
        tensors = [item for item in result if isinstance(item, torch.Tensor)]
        if not tensors or len(tensors) != len(result):
            raise TypeError("native result tuple must contain only tensors")
        return tensors
    raise TypeError(f"native callable returned unsupported type {type(result)!r}")


def output_pointer_sequence(torch: Any, results: list[Any]) -> list[list[int]]:
    sequence: list[list[int]] = []
    for result in results:
        sequence.append([int(tensor.data_ptr())
                         for tensor in result_tensors(torch, result)])
    return sequence


def _reference_rms_norm(torch: Any, tensors: ConditionTensors, source_cpu: Any,
                        epsilon: float) -> dict[str, Any]:
    x = source_cpu.to(torch.float32)
    gamma = tensors.gamma_cpu.to(torch.float32)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    rstd = torch.rsqrt(variance + float(epsilon))
    output = x * rstd * gamma
    return {"output": output, "rstd": rstd}


def _reference_swiglu(torch: Any, tensors: ConditionTensors,
                     source_cpu: Any) -> dict[str, Any]:
    import torch.nn.functional as functional

    packed = source_cpu.to(torch.float32)
    last = packed.shape[-1]
    half = last // 2
    gate = packed[..., :half]
    up = packed[..., half:]
    return {"output": functional.silu(gate) * up}


def reference_outputs(torch: Any, condition: Mapping[str, Any],
                      tensors: ConditionTensors, *, mutated: bool = False) -> dict[str, Any]:
    """Compute the FP32 CPU reference for the original or mutated source."""
    operator = condition["operator"]
    source_cpu = tensors.mutation_cpu if mutated else tensors.static_cpu
    if operator == "rms_norm":
        return _reference_rms_norm(
            torch, tensors, source_cpu, float(condition["epsilon"]))
    if operator == "swiglu":
        return _reference_swiglu(torch, tensors, source_cpu)
    if operator == "copy":
        return {"target": source_cpu.to(torch.uint8).clone()}
    raise ValueError(f"unsupported operator {operator!r}")


def _finite(value: float) -> float | None:
    number = float(value)
    return number if math.isfinite(number) else None


def _compare_float(torch: Any, actual: Any, expected: Any, *,
                   atol: float, rtol: float, name: str) -> dict[str, Any]:
    actual_cpu = actual.detach().to(device="cpu")
    expected_cpu = expected.detach().to(device="cpu").to(torch.float32)
    result: dict[str, Any] = {
        "name": name,
        "actual_shape": [int(value) for value in actual_cpu.shape],
        "expected_shape": [int(value) for value in expected_cpu.shape],
        "actual_dtype": str(actual_cpu.dtype).replace("torch.", ""),
        "reference_dtype": "float32_cpu",
        "atol": float(atol),
        "rtol": float(rtol),
        "comparison_outside_timing": True,
    }
    if list(actual_cpu.shape) != list(expected_cpu.shape):
        result.update({
            "status": "failed",
            "passed": False,
            "reason": "shape_mismatch",
            "mismatch_count": None,
            "max_abs_error": None,
            "max_rel_error": None,
            "nonfinite_count": None,
        })
        return result
    actual_f = actual_cpu.to(torch.float32)
    finite_mask = torch.isfinite(actual_f)
    nonfinite_count = int((~finite_mask).sum().item())
    diff = (actual_f - expected_cpu).abs()
    tolerance = float(atol) + float(rtol) * expected_cpu.abs()
    mismatch = (diff > tolerance) | (~finite_mask)
    mismatch_count = int(mismatch.sum().item())
    max_abs = _finite(diff.max().item()) if diff.numel() else 0.0
    relative = diff / expected_cpu.abs().clamp_min(1e-12)
    max_rel = _finite(relative.max().item()) if relative.numel() else 0.0
    passed = mismatch_count == 0
    result.update({
        "status": "passed" if passed else "failed",
        "passed": passed,
        "mismatch_count": mismatch_count,
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "nonfinite_count": nonfinite_count,
        "max_abs_error_nonfinite_metric": max_abs is None,
        "max_rel_error_nonfinite_metric": max_rel is None,
        "tolerance_formula": "abs(actual-reference) <= atol + rtol*abs(reference)",
    })
    return result


def _compare_exact(torch: Any, actual: Any, expected: Any,
                   name: str) -> dict[str, Any]:
    actual_cpu = actual.detach().to(device="cpu")
    expected_cpu = expected.detach().to(device="cpu").to(torch.uint8)
    same_shape = list(actual_cpu.shape) == list(expected_cpu.shape)
    same_dtype = actual_cpu.dtype == expected_cpu.dtype
    if not same_shape or not same_dtype:
        return {
            "name": name,
            "status": "failed",
            "passed": False,
            "reason": "shape_or_dtype_mismatch",
            "actual_shape": [int(value) for value in actual_cpu.shape],
            "expected_shape": [int(value) for value in expected_cpu.shape],
            "actual_dtype": str(actual_cpu.dtype).replace("torch.", ""),
            "comparison": "exact_byte_equality_no_tolerance",
            "comparison_outside_timing": True,
        }
    mismatch = (actual_cpu != expected_cpu)
    mismatch_count = int(mismatch.sum().item())
    return {
        "name": name,
        "status": "passed" if mismatch_count == 0 else "failed",
        "passed": mismatch_count == 0,
        "actual_shape": [int(value) for value in actual_cpu.shape],
        "expected_shape": [int(value) for value in expected_cpu.shape],
        "actual_dtype": str(actual_cpu.dtype).replace("torch.", ""),
        "mismatch_count": mismatch_count,
        "comparison": "exact_byte_equality_no_tolerance",
        "comparison_outside_timing": True,
    }


def check_against_reference(torch: Any, condition: Mapping[str, Any],
                            tensors: ConditionTensors, result: Any,
                            protocol: Mapping[str, Any], *,
                            mutated: bool = False) -> dict[str, Any]:
    """Compare sampled native output (and feasible RMS rstd) with FP32 CPU."""
    expected = reference_outputs(torch, condition, tensors, mutated=mutated)
    actual = result_tensors(torch, result)
    reference = protocol["reference"]
    atol = float(reference["atol"])
    rtol = float(reference["rtol"])
    checks: dict[str, Any] = {}
    operator = condition["operator"]
    if operator in ("rms_norm", "swiglu"):
        checks["output"] = _compare_float(
            torch, actual[0], expected["output"], atol=atol, rtol=rtol,
            name="output")
    else:
        checks["output"] = _compare_exact(
            torch, actual[0], expected["target"], name="output")
    rstd_checked = False
    if operator == "rms_norm":
        if len(actual) >= 2:
            expected_rstd = expected["rstd"]
            actual_rstd = actual[1]
            if int(actual_rstd.numel()) == int(expected_rstd.numel()):
                actual_rstd_flat = actual_rstd.reshape(-1)
                expected_rstd_flat = expected_rstd.reshape(-1)
                rstd_check = _compare_float(
                    torch, actual_rstd_flat, expected_rstd_flat,
                    atol=float(reference["rstd_atol"]),
                    rtol=float(reference["rstd_rtol"]), name="rstd")
                rstd_check["actual_shape_before_flatten"] = [
                    int(value) for value in actual_rstd.shape]
                rstd_check["expected_shape"] = [
                    int(value) for value in expected_rstd.shape]
                rstd_check["shape_note"] = (
                    "values compared after flattening because rstd element count "
                    "matches; no data repair")
                checks["rstd"] = rstd_check
                rstd_checked = True
            else:
                checks["rstd"] = {
                    "name": "rstd",
                    "status": "unsupported",
                    "passed": None,
                    "reason": "returned rstd element count does not match [T,1]",
                    "actual_numel": int(actual_rstd.numel()),
                    "expected_numel": int(expected_rstd.numel()),
                }
        else:
            checks["rstd"] = {
                "name": "rstd",
                "status": "unsupported",
                "passed": None,
                "reason": "installed callable did not return a second tensor",
            }
    required_passed = checks["output"]["passed"] is True
    rstd_passed = checks.get("rstd", {}).get("passed")
    overall = required_passed and rstd_passed is not False
    return {
        "status": "passed" if overall else "failed",
        "passed": overall,
        "mutated_input": mutated,
        "checks": checks,
        "rstd_checked": rstd_checked,
        "rstd_required": False,
        "reference_dtype": "float32_cpu",
        "comparison_outside_timing": True,
        "data_repair": "none",
    }


def output_facts(torch: Any, result: Any) -> list[dict[str, Any]]:
    return [tensor_facts(tensor) for tensor in result_tensors(torch, result)]


def unique_pointer_count(sequence: list[list[int]]) -> int:
    return len({pointer for group in sequence for pointer in group})
