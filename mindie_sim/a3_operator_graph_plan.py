"""Manifest schema, physical geometry, protocol schedule and CPU-only dry-run plan.

[Role]
  Define the frozen, versioned input contract for the isolated A3 native-operator
  graph timing lane.  Validate a manifest using only the standard library, expand
  its six train-side shapes into explicit mode/N protocol conditions, and compute
  the exact planned condition/trial/sample counts and declared tensor geometry.
  ``--dry-run`` uses this module and never imports torch or torch_npu.

[Invariants]
  * RMSNorm/SwiGLU declare only the installed default-output overloads; there is
    no invented ``.out`` or preallocated-output claim.
  * Native operator math is fixed: RMSNorm epsilon is exactly 1e-6, SwiGLU dim is
    -1 and packed input has a last dimension of 2*F.
  * A condition is one (case, timing mode, N) tuple.  N is the number of declared
    operator API invocations, never asserted to be a hardware kernel count.
  * Trials are complete permutations of all conditions.  Odd trials reverse the
    even-trial order, making the order deterministic and counterbalanced.
  * Budget ceilings cannot be raised above 24 s case / 1200 s work / 60 s setup /
    120 s cleanup.  Declared expectations must equal computed counts.
  * No timing, device fact or measurement-complete status is fabricated by
    validation.

[Timeouts & Locks]
  Pure in-process parsing and arithmetic; no locks, threads or device calls.
  Runtime budget enforcement lives in the execution modules.

[Side-Effect Policy]
  Reads one caller-supplied manifest path in the loader helper.  Validation,
  planning and dry-run are otherwise pure and write no files or device state.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

KIND = "a3_operator_graph"
MANIFEST_KIND = "a3_operator_graph_manifest"

OPERATORS = ("rms_norm", "swiglu", "copy")
MODES = ("eager_block", "graph_single_repeated", "graph_block")
ALLOWED_NS = (1, 32, 128)
DTYPES = {"rms_norm": "bfloat16", "swiglu": "bfloat16", "copy": "uint8"}
ITEMSIZE = {"bfloat16": 2, "uint8": 1, "float32": 4}
SHAPE_FIELDS = {
    "rms_norm": ("tokens", "hidden"),
    "swiglu": ("tokens", "ffn"),
    "copy": ("rows", "row_bytes"),
}
DATA_FIELDS = {
    "rms_norm": ("seed", "input_low", "input_high", "gamma_low", "gamma_high"),
    "swiglu": ("seed", "input_low", "input_high"),
    "copy": ("seed",),
}
MODE_SEMANTICS = {
    "eager_block": "N eager operator invocations per measured block",
    "graph_single_repeated": (
        "capture one operator invocation; replay that one-call graph N times "
        "from Python per measured block"
    ),
    "graph_block": (
        "capture N operator invocations; replay the N-call graph once per "
        "measured block"
    ),
}
MODE_CAPTURE_INVOCATIONS = {"eager_block": 0, "graph_single_repeated": 1}
BUDGET_CEILINGS = {
    "case": 24.0,
    "global_work": 1200.0,
    "setup": 60.0,
    "cleanup": 120.0,
}
OUTER_SUPERVISOR_SECONDS_MIN = 1380.0
CALLABLE_SOURCE = {
    "rms_norm": "torch_npu.npu_rms_norm(x, gamma, epsilon=1e-6)",
    "swiglu": "torch_npu.npu_swiglu(packed, -1)",
    "copy": "target.copy_(source)",
}
OUTPUT_POLICY = {
    "rms_norm": "native_default_outputs_only_no_out_overload",
    "swiglu": "native_default_outputs_only_no_out_overload",
    "copy": "single_same_preallocated_destination",
}
N_KIND = "declared_operator_api_invocations_not_measured_kernel_launches"
_CASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class ManifestError(ValueError):
    """One or more manifest contract violations, without any device work."""

    def __init__(self, errors: Sequence[str], *, path: str | None = None) -> None:
        self.errors = list(errors)
        self.path = path
        prefix = f"{path}: " if path else ""
        super().__init__(prefix + "; ".join(self.errors))

    def as_dict(self) -> dict[str, Any]:
        return {"error": "manifest_invalid", "path": self.path,
                "errors": list(self.errors)}


def _add(errors: list[str], path: str, message: str) -> None:
    errors.append(f"{path}: {message}")


def _object(value: Any, path: str, allowed: Sequence[str],
            required: Sequence[str], errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        _add(errors, path, "expected a JSON object")
        return {}
    for key in value:
        if key not in allowed:
            _add(errors, path, f"unknown field {key!r}")
    for key in required:
        if key not in value:
            _add(errors, path, f"missing required field {key!r}")
    return value


def _integer(value: Any, path: str, minimum: int, maximum: int,
             errors: list[str]) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        _add(errors, path, "expected an integer")
        return None
    if not minimum <= value <= maximum:
        _add(errors, path, f"must be in [{minimum}, {maximum}]")
        return None
    return value


def _number(value: Any, path: str, minimum: float, maximum: float,
            errors: list[str], *, exclusive_min: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _add(errors, path, "expected a finite number")
        return None
    number = float(value)
    if not math.isfinite(number):
        _add(errors, path, "expected a finite number")
        return None
    low_ok = number > minimum if exclusive_min else number >= minimum
    if not low_ok or number > maximum:
        qualifier = ">" if exclusive_min else ">="
        _add(errors, path, f"must be {qualifier} {minimum} and <= {maximum}")
        return None
    return number


def _text(value: Any, path: str, errors: list[str], *,
          max_length: int = 512) -> str | None:
    if not isinstance(value, str) or not value.strip():
        _add(errors, path, "expected a non-empty string")
        return None
    if len(value) > max_length:
        _add(errors, path, f"must be at most {max_length} characters")
        return None
    return value


def _strides(shape: Sequence[int]) -> list[int]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return strides


def tensor_geometry(name: str, shape: Sequence[int], dtype: str) -> dict[str, Any]:
    """Return the declared contiguous physical geometry for one tensor."""
    normalized = [int(value) for value in shape]
    numel = 1
    for value in normalized:
        numel *= value
    itemsize = ITEMSIZE[dtype]
    return {
        "name": name,
        "shape": normalized,
        "strides": _strides(normalized),
        "dtype": dtype,
        "itemsize": itemsize,
        "numel": numel,
        "nbytes": numel * itemsize,
        "layout": "contiguous_row_major",
    }


def _validate_shape(operator: str, value: Any, path: str,
                    errors: list[str]) -> dict[str, int]:
    fields = SHAPE_FIELDS[operator]
    shape = _object(value, path, fields, fields, errors)
    result: dict[str, int] = {}
    for field in fields:
        number = _integer(shape.get(field), f"{path}.{field}", 1, 2**31 - 1,
                          errors)
        if number is not None:
            result[field] = number
    return result


def _validate_data(operator: str, value: Any, path: str,
                   errors: list[str]) -> dict[str, Any]:
    fields = DATA_FIELDS[operator]
    data = _object(value, path, fields, fields, errors)
    result: dict[str, Any] = {}
    seed = _integer(data.get("seed"), f"{path}.seed", 0, 2**63 - 1, errors)
    if seed is not None:
        result["seed"] = seed
    if operator == "copy":
        return result
    low = _number(data.get("input_low"), f"{path}.input_low", -1.0e6, 1.0e6,
                  errors)
    high = _number(data.get("input_high"), f"{path}.input_high", -1.0e6, 1.0e6,
                   errors)
    if low is not None and high is not None and low >= high:
        _add(errors, path, "input_low must be less than input_high")
    if low is not None:
        result["input_low"] = low
    if high is not None:
        result["input_high"] = high
    if operator == "rms_norm":
        gamma_low = _number(data.get("gamma_low"), f"{path}.gamma_low",
                            -1.0e6, 1.0e6, errors)
        gamma_high = _number(data.get("gamma_high"), f"{path}.gamma_high",
                             -1.0e6, 1.0e6, errors)
        if gamma_low is not None and gamma_high is not None and gamma_low >= gamma_high:
            _add(errors, path, "gamma_low must be less than gamma_high")
        if gamma_low is not None:
            result["gamma_low"] = gamma_low
        if gamma_high is not None:
            result["gamma_high"] = gamma_high
    return result


def _reference_plan(operator: str, shape: Mapping[str, int],
                    epsilon: float | None) -> dict[str, Any]:
    if operator == "rms_norm":
        tokens = shape["tokens"]
        return {
            "kind": "fp32_cpu_rms_norm",
            "epsilon": epsilon,
            "normalized_axis": -1,
            "variance": "mean(x*x, axis=-1, keepdim=True)",
            "rstd": "rsqrt(variance + epsilon)",
            "output": "x * rstd * gamma",
            "compared": ["output", "rstd_when_returned"],
            "rstd_expected_shape": [tokens, 1],
            "rstd_expected_dtype": "float32",
            "rstd_expected_itemsize": 4,
            "rstd_required": False,
            "rstd_note": "checked when the installed default overload returns a "
                         "two-tensor tuple with matching element count",
            "reference_dtype": "float32_cpu",
            "installed_schema": (
                "npu::npu_rms_norm(Tensor input, Tensor gamma, float epsilon)"
                " -> (Tensor, Tensor)"
            ),
            "callable_source": CALLABLE_SOURCE[operator],
            "output_policy": OUTPUT_POLICY[operator],
        }
    if operator == "swiglu":
        tokens, ffn = shape["tokens"], shape["ffn"]
        return {
            "kind": "fp32_cpu_swiglu",
            "packed_last_dim": 2 * ffn,
            "split_axis": -1,
            "gate": "silu(packed[..., :F])",
            "up": "packed[..., F:]",
            "output": "silu(packed[..., :F]) * packed[..., F:]",
            "compared": ["output"],
            "output_shape": [tokens, ffn],
            "reference_dtype": "float32_cpu",
            "installed_schema": "npu::npu_swiglu(Tensor input, int dim=-1) -> Tensor",
            "callable_source": CALLABLE_SOURCE[operator],
            "output_policy": OUTPUT_POLICY[operator],
        }
    return {
        "kind": "exact_uint8_copy_equality",
        "operation": "target.copy_(source)",
        "compared": ["output"],
        "comparison": "exact_byte_equality_no_tolerance",
        "destination": "one_same_preallocated_target_for_every_block",
        "scope": "local_contiguous_copy_not_d2d_or_cache_scatter_claim",
        "callable_source": CALLABLE_SOURCE[operator],
        "output_policy": OUTPUT_POLICY[operator],
    }


def _data_contract(operator: str, data: Mapping[str, Any]) -> dict[str, Any]:
    contract: dict[str, Any] = {
        "seed": data["seed"],
        "generator": "torch.Generator(device='cpu'); manual_seed(seed)",
        "transfer": "contiguous CPU source -> NPU before capture, outside timing",
        "pairing": "every mode/N for this case reuses the same CPU source bytes",
        "mutation_seed": int(data["seed"]) ^ 0x5EED,
    }
    if operator == "copy":
        contract["source"] = {
            "dtype": "uint8",
            "generation": "randint(0, 256) on CPU with the declared seed",
        }
    else:
        contract["input"] = {
            "generation": "uniform float32 on CPU then cast to bfloat16",
            "low": data["input_low"],
            "high": data["input_high"],
        }
    if operator == "rms_norm":
        contract["gamma"] = {
            "generation": "uniform float32 on CPU then cast to bfloat16",
            "low": data["gamma_low"],
            "high": data["gamma_high"],
        }
    return contract


def _case_geometry(operator: str, dtype: str, shape: Mapping[str, int]) -> dict[str, Any]:
    if operator == "rms_norm":
        tokens, hidden = shape["tokens"], shape["hidden"]
        tensors = {
            "input": tensor_geometry("input", [tokens, hidden], dtype),
            "gamma": tensor_geometry("gamma", [hidden], dtype),
            "output": tensor_geometry("output", [tokens, hidden], dtype),
            "rstd": tensor_geometry("rstd", [tokens, 1], "float32"),
        }
        output_names = ("output", "rstd")
    elif operator == "swiglu":
        tokens, ffn = shape["tokens"], shape["ffn"]
        tensors = {
            "packed_input": tensor_geometry("packed_input", [tokens, 2 * ffn], dtype),
            "output": tensor_geometry("output", [tokens, ffn], dtype),
        }
        output_names = ("output",)
    else:
        rows, row_bytes = shape["rows"], shape["row_bytes"]
        tensors = {
            "source": tensor_geometry("source", [rows, row_bytes], dtype),
            "target": tensor_geometry("target", [rows, row_bytes], dtype),
        }
        output_names = ("target",)
    per_invocation_output_bytes = sum(tensors[name]["nbytes"] for name in output_names)
    return {
        "operator": operator,
        "dtype": dtype,
        "itemsize": ITEMSIZE[dtype],
        "tensors": tensors,
        "total_declared_tensor_bytes": sum(item["nbytes"] for item in tensors.values()),
        "per_invocation_output_bytes_declared": per_invocation_output_bytes,
        "per_invocation_output_bytes_kind": "shape_derived_declared_not_allocator_measured",
        "layout": "contiguous_row_major",
    }


def _validate_case(raw: Any, index: int, errors: list[str]) -> dict[str, Any] | None:
    path = f"cases[{index}]"
    obj = _object(raw, path, ("case_id", "operator", "dtype", "layout", "shape",
                              "ns", "modes", "data", "epsilon"),
                  ("case_id", "operator", "dtype", "layout", "shape", "ns",
                   "modes", "data"), errors)
    case_id = _text(obj.get("case_id"), f"{path}.case_id", errors, max_length=128)
    if case_id is not None and not _CASE_ID_RE.match(case_id):
        _add(errors, f"{path}.case_id",
             "must match [A-Za-z0-9][A-Za-z0-9._:-]{0,127}")
    operator = obj.get("operator")
    if operator not in OPERATORS:
        _add(errors, f"{path}.operator", f"must be one of {list(OPERATORS)}")
        return None
    dtype = obj.get("dtype")
    if dtype != DTYPES[operator]:
        _add(errors, f"{path}.dtype",
             f"{operator} requires dtype {DTYPES[operator]!r}")
    layout = obj.get("layout")
    if layout != "contiguous":
        _add(errors, f"{path}.layout", "only 'contiguous' row-major is supported")
    shape = _validate_shape(operator, obj.get("shape"), f"{path}.shape", errors)
    ns_value = obj.get("ns")
    ns: list[int] = []
    if not isinstance(ns_value, list) or not ns_value:
        _add(errors, f"{path}.ns", "expected a non-empty list")
    else:
        for ns_index, value in enumerate(ns_value):
            n = _integer(value, f"{path}.ns[{ns_index}]", 1, 2**31 - 1, errors)
            if n is None:
                continue
            if n not in ALLOWED_NS:
                _add(errors, f"{path}.ns[{ns_index}]",
                     f"N must be one of {list(ALLOWED_NS)} for this lane")
            if n in ns:
                _add(errors, f"{path}.ns[{ns_index}]", f"duplicate N {n}")
            ns.append(n)
    modes_value = obj.get("modes")
    modes: list[str] = []
    if not isinstance(modes_value, list) or not modes_value:
        _add(errors, f"{path}.modes", "expected a non-empty list")
    else:
        for mode_index, mode in enumerate(modes_value):
            if mode not in MODES:
                _add(errors, f"{path}.modes[{mode_index}]",
                     f"mode must be one of {list(MODES)}")
            elif mode in modes:
                _add(errors, f"{path}.modes[{mode_index}]", f"duplicate mode {mode!r}")
            else:
                modes.append(mode)
    data = _validate_data(operator, obj.get("data"), f"{path}.data", errors)
    epsilon: float | None = None
    if operator == "rms_norm":
        epsilon = _number(obj.get("epsilon"), f"{path}.epsilon", 0.0, 1.0, errors,
                          exclusive_min=True)
        if epsilon is not None and epsilon != 1e-6:
            _add(errors, f"{path}.epsilon",
                 "installed callable math requires epsilon exactly 1e-6")
    elif "epsilon" in obj:
        _add(errors, f"{path}.epsilon", "only rms_norm may declare epsilon")
    if (case_id is None or operator not in OPERATORS or dtype != DTYPES.get(operator)
            or not shape or len(shape) != len(SHAPE_FIELDS[operator])
            or not ns or not modes or not data):
        return None
    if operator == "rms_norm" and epsilon is None:
        return None
    geometry = _case_geometry(operator, dtype, shape)
    return {
        "case_id": case_id,
        "operator": operator,
        "dtype": dtype,
        "layout": "contiguous",
        "shape": shape,
        "epsilon": epsilon,
        "ns": ns,
        "modes": modes,
        "data": data,
        "geometry": geometry,
        "reference_plan": _reference_plan(operator, shape, epsilon),
        "data_contract": _data_contract(operator, data),
    }


def _validate_protocol(raw: Any, errors: list[str]) -> dict[str, Any] | None:
    path = "protocol"
    obj = _object(raw, path,
                  ("warmup_iterations", "trials", "samples_per_condition",
                   "seed", "reference", "budgets_seconds"),
                  ("warmup_iterations", "trials", "samples_per_condition",
                   "seed", "reference", "budgets_seconds"), errors)
    warmup = _integer(obj.get("warmup_iterations"), f"{path}.warmup_iterations",
                      1, 10000, errors)
    trials = _integer(obj.get("trials"), f"{path}.trials", 1, 100, errors)
    samples = _integer(obj.get("samples_per_condition"),
                       f"{path}.samples_per_condition", 1, 1000, errors)
    seed = _integer(obj.get("seed"), f"{path}.seed", 0, 2**63 - 1, errors)
    reference_obj = _object(obj.get("reference"), f"{path}.reference",
                            ("rtol", "atol", "rstd_rtol", "rstd_atol"),
                            ("rtol", "atol", "rstd_rtol", "rstd_atol"), errors)
    reference: dict[str, float] = {}
    for key in ("rtol", "atol", "rstd_rtol", "rstd_atol"):
        number = _number(reference_obj.get(key), f"{path}.reference.{key}",
                         0.0, 0.25, errors, exclusive_min=True)
        if number is not None:
            reference[key] = number
    budgets_obj = _object(obj.get("budgets_seconds"), f"{path}.budgets_seconds",
                          tuple(BUDGET_CEILINGS),
                          tuple(BUDGET_CEILINGS), errors)
    budgets: dict[str, float] = {}
    for key, ceiling in BUDGET_CEILINGS.items():
        number = _number(budgets_obj.get(key),
                         f"{path}.budgets_seconds.{key}", 0.0, ceiling, errors,
                         exclusive_min=True)
        if number is not None:
            budgets[key] = number
    if budgets.get("case") is not None and budgets.get("global_work") is not None:
        if budgets["case"] > budgets["global_work"]:
            _add(errors, f"{path}.budgets_seconds",
                 "case budget cannot exceed global_work budget")
    if len(reference) != 4 or len(budgets) != 4:
        return None
    if None in (warmup, trials, samples, seed):
        return None
    return {
        "warmup_iterations": warmup,
        "trials": trials,
        "samples_per_condition": samples,
        "seed": seed,
        "reference": reference,
        "budgets_seconds": budgets,
    }


def _compute_counts(cases: Sequence[Mapping[str, Any]],
                    protocol: Mapping[str, Any]) -> dict[str, Any]:
    per_case = {case["case_id"]: len(case["modes"]) * len(case["ns"])
                for case in cases}
    per_mode: dict[str, int] = {mode: 0 for mode in MODES}
    per_n: dict[str, int] = {str(n): 0 for n in ALLOWED_NS}
    for case in cases:
        for mode in case["modes"]:
            per_mode[mode] += len(case["ns"])
        for n in case["ns"]:
            per_n[str(n)] += len(case["modes"])
    conditions = sum(per_case.values())
    trials = int(protocol["trials"])
    samples = int(protocol["samples_per_condition"])
    return {
        "case_count": len(cases),
        "condition_count": conditions,
        "conditions_per_case": per_case,
        "conditions_per_mode": per_mode,
        "conditions_per_n": per_n,
        "trials": trials,
        "samples_per_condition": samples,
        "warmup_iterations": int(protocol["warmup_iterations"]),
        "planned_condition_executions": conditions * trials,
        "planned_raw_blocks": conditions * trials * samples,
        "planned_warmup_iterations_per_trial": int(
            protocol["warmup_iterations"]),
        "planned_warmup_events_per_condition": (
            int(protocol["warmup_iterations"]) * trials),
    }


def validate_manifest(document: Any) -> dict[str, Any]:
    """Validate and normalize a parsed manifest, raising ManifestError."""
    errors: list[str] = []
    obj = _object(document, "manifest",
                  ("kind", "manifest_id", "description", "protocol",
                   "expectations", "cases"),
                  ("kind", "manifest_id", "protocol", "expectations",
                   "cases"), errors)
    if obj.get("kind") != MANIFEST_KIND:
        _add(errors, "manifest.kind", f"must be exactly {MANIFEST_KIND!r}")
    manifest_id = _text(obj.get("manifest_id"), "manifest.manifest_id", errors,
                        max_length=256)
    description = obj.get("description", "")
    if not isinstance(description, str) or len(description) > 4096:
        _add(errors, "manifest.description", "must be a string of at most 4096 characters")
        description = ""
    protocol = _validate_protocol(obj.get("protocol"), errors)
    expectations_obj = _object(obj.get("expectations"), "manifest.expectations",
                               ("protocol_conditions", "raw_blocks"),
                               ("protocol_conditions", "raw_blocks"), errors)
    expectations = {
        "protocol_conditions": _integer(
            expectations_obj.get("protocol_conditions"),
            "manifest.expectations.protocol_conditions", 1, 2**31 - 1, errors),
        "raw_blocks": _integer(
            expectations_obj.get("raw_blocks"),
            "manifest.expectations.raw_blocks", 1, 2**63 - 1, errors),
    }
    raw_cases = obj.get("cases")
    cases: list[dict[str, Any]] = []
    if not isinstance(raw_cases, list) or not raw_cases:
        _add(errors, "manifest.cases", "expected a non-empty list")
    else:
        seen_ids: set[str] = set()
        for index, raw_case in enumerate(raw_cases):
            case = _validate_case(raw_case, index, errors)
            if case is None:
                continue
            if case["case_id"] in seen_ids:
                _add(errors, f"cases[{index}].case_id",
                     f"duplicate case_id {case['case_id']!r}")
            seen_ids.add(case["case_id"])
            cases.append(case)
    if errors:
        raise ManifestError(errors)
    assert protocol is not None
    counts = _compute_counts(cases, protocol)
    if expectations["protocol_conditions"] != counts["condition_count"]:
        errors.append(
            "manifest.expectations.protocol_conditions: declared "
            f"{expectations['protocol_conditions']} but computed {counts['condition_count']}")
    if expectations["raw_blocks"] != counts["planned_raw_blocks"]:
        errors.append(
            "manifest.expectations.raw_blocks: declared "
            f"{expectations['raw_blocks']} but computed {counts['planned_raw_blocks']}")
    if errors:
        raise ManifestError(errors)
    return {
        "kind": MANIFEST_KIND,
        "manifest_id": manifest_id,
        "description": description,
        "protocol": protocol,
        "expectations": {
            "protocol_conditions": int(expectations["protocol_conditions"]),
            "raw_blocks": int(expectations["raw_blocks"]),
        },
        "cases": cases,
    }


def _condition(case: Mapping[str, Any], mode: str, n: int, mode_index: int,
               n_index: int) -> dict[str, Any]:
    capture_invocations = (0 if mode == "eager_block"
                           else 1 if mode == "graph_single_repeated" else n)
    host_calls = n if mode != "graph_block" else 1
    replay_calls = 0 if mode == "eager_block" else n if mode == "graph_single_repeated" else 1
    per_invocation_output = int(
        case["geometry"]["per_invocation_output_bytes_declared"])
    declared_capture_bytes = per_invocation_output * capture_invocations
    condition_id = f"{case['case_id']}::{mode}::n{n}"
    return {
        "condition_id": condition_id,
        "case_id": case["case_id"],
        "operator": case["operator"],
        "dtype": case["dtype"],
        "layout": case["layout"],
        "shape": dict(case["shape"]),
        "epsilon": case["epsilon"],
        "data_seed": case["data"]["seed"],
        "mode": mode,
        "mode_index": mode_index,
        "n": n,
        "n_index": n_index,
        "mode_semantics": MODE_SEMANTICS[mode],
        "n_kind": N_KIND,
        "callable_source": CALLABLE_SOURCE[case["operator"]],
        "output_policy": OUTPUT_POLICY[case["operator"]],
        "preallocated_output": case["operator"] == "copy",
        "capture_invocations": capture_invocations,
        "host_calls_per_block": host_calls,
        "replay_calls_per_block": replay_calls,
        "block_invocations": n,
        "per_invocation_output_bytes_declared": per_invocation_output,
        "declared_capture_output_bytes": declared_capture_bytes,
        "declared_capture_output_bytes_kind":
            "shape_derived_declared_not_allocator_measured",
        "geometry": case["geometry"],
        "reference_plan": case["reference_plan"],
        "data_contract": case["data_contract"],
    }


def build_conditions(cases: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    conditions: list[dict[str, Any]] = []
    for case in cases:
        for mode_index, mode in enumerate(case["modes"]):
            for n_index, n in enumerate(case["ns"]):
                conditions.append(_condition(case, mode, n, mode_index, n_index))
    return conditions


def build_schedule(conditions: Sequence[Mapping[str, Any]],
                   protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Build complete, deterministic, counterbalanced trial permutations."""
    base = [condition["condition_id"] for condition in conditions]
    if not base:
        raise ManifestError(["schedule: no conditions"])
    trial_orders: list[list[str]] = []
    size = len(base)
    for trial in range(int(protocol["trials"])):
        rotation = (trial // 2) % size
        rotated = base[rotation:] + base[:rotation]
        order = list(reversed(rotated)) if trial % 2 else rotated
        trial_orders.append(order)
    return {
        "seed": int(protocol["seed"]),
        "base_order": base,
        "trial_orders": trial_orders,
        "order_rule": (
            "selected order = base order rotated left by (trial_index//2) mod "
            "condition_count; odd trials reverse that rotation. Each trial is a "
            "complete permutation; adjacent trials counterbalance position."
        ),
        "counterbalanced": len(trial_orders) > 1,
    }


def build_plan(document: Any) -> dict[str, Any]:
    """Validate a manifest and return its complete predeclared execution plan."""
    manifest = validate_manifest(document)
    conditions = build_conditions(manifest["cases"])
    schedule = build_schedule(conditions, manifest["protocol"])
    counts = _compute_counts(manifest["cases"], manifest["protocol"])
    budgets = dict(manifest["protocol"]["budgets_seconds"])
    budgets["outer_supervisor_seconds_min"] = OUTER_SUPERVISOR_SECONDS_MIN
    budgets["case_scope"] = (
        "accumulated active prepare+measurement seconds per condition; wall "
        "time between trial passes is not charged to the case")
    budgets["global_work_scope"] = (
        "from after setup through the declared work loop; excludes setup and "
        "the post-work cleanup budget")
    budgets["cleanup_scope"] = (
        "post-work synchronize and graph reset only; never overlaps samples")
    return {
        "kind": KIND,
        "manifest_id": manifest["manifest_id"],
        "description": manifest["description"],
        "protocol": manifest["protocol"],
        "expectations": manifest["expectations"],
        "cases": manifest["cases"],
        "conditions": conditions,
        "condition_ids": [condition["condition_id"] for condition in conditions],
        "schedule": schedule,
        "counts": counts,
        "budgets": budgets,
        "checks": [
            "manifest schema and unknown-field validation",
            "operator/dtype/layout/epsilon/N/mode legality",
            "physical contiguous shape, stride and byte geometry",
            "declared condition expectation equals computed count",
            "declared raw-block expectation equals computed count",
            "every trial is a complete condition permutation",
            "budgets are within the lane ceilings",
        ],
    }


def _pairs_no_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number {value!r} is not allowed")


def parse_manifest_text(text: str, *, path: str | None = None) -> dict[str, Any]:
    try:
        document = json.loads(text, object_pairs_hook=_pairs_no_duplicates,
                              parse_constant=_reject_constant)
    except (json.JSONDecodeError, ValueError) as exc:
        raise ManifestError([f"invalid JSON: {exc}"], path=path) from exc
    if not isinstance(document, dict):
        raise ManifestError(["top-level manifest must be a JSON object"], path=path)
    return document


def load_manifest_file(path: str | Path) -> tuple[dict[str, Any], str, str]:
    """Read one explicit manifest path, returning document, sha256 and path."""
    manifest_path = Path(path)
    try:
        raw = manifest_path.read_bytes()
    except OSError as exc:
        raise ManifestError([f"cannot read manifest: {exc}"], path=str(path)) from exc
    text = raw.decode("utf-8")
    document = parse_manifest_text(text, path=str(path))
    return document, hashlib.sha256(raw).hexdigest(), str(manifest_path)


def dry_run_payload(path: str | Path) -> dict[str, Any]:
    """Validate a manifest with the standard library only; no torch import."""
    document, digest, manifest_path = load_manifest_file(path)
    plan = build_plan(document)
    return {
        "dry_run": True,
        "valid": True,
        "validation_mode": "stdlib_only_no_torch_import",
        "manifest_path": manifest_path,
        "manifest_sha256": digest,
        "plan": plan,
    }
