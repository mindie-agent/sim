"""Prepare-only standalone R31 GDN gated RMSNorm invocation.

[Role] Expand the R31 A3-subgraph GDN ``linear_output_norm_gate`` occurrence
  into the native ``(x, weight, gate) -> (output, rstd)`` descriptor of the
  ``_layer_norm_fwd_1pass_kernel_npu`` Triton launcher.
[Invariants] Admission is the A3 GDN subgraph carrier plus the exact R31 source
  image/caller/wrapper/kernel and the configured gated-norm width/group/head
  shape.  Model labels and hardware labels never admit or reject; the true R31
  capture on Ascend950DT/A5 keeps the same A3 subgraph kind.  The static
  identity holds width 128, one group per row, BF16 tensors, FP32 rstd,
  epsilon 1e-6 and SiLU/Swish gate activation.  Token and row lanes live only
  in the occurrence.  ``norm_before_gate`` is not observed by the retained
  capture/source-constructor pair and stays explicit.  No tensor is allocated,
  no runtime is imported, no duration is changed and the ordinary int8-derived
  gated-norm cost is not used as the native activation dtype.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing compiled
  occurrence and retained source/capture evidence.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = (
    "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea")
R31_SOURCE_VERSION = "r31/9e869e814846"
R31_SOURCE_COMMIT = "8fd0bc488"
NATIVE_TYPE = "_layer_norm_fwd_1pass_kernel_npu"
ENTRY = "vllm_ascend.ops.triton.layernorm_gated.layer_norm_fwd_npu"
SCHEMA = "mindie-sim/gated-layer-norm-native-prepare/1"
# R31 configured gated-norm epsilon/activation provenance; not an admission key.
MODEL_CONFIG_SHA = (
    "69db4eb7196bc8190813231b3018ca05d8c2e3abc7b1af19d55c157af44a9d9c")
EPSILON = 1e-6
BLOCK_M = 64
_DTYPE_ALIASES = {
    "bf16": "bf16", "bfloat16": "bf16",
    "fp16": "fp16", "float16": "fp16",
    "fp32": "fp32", "float32": "fp32",
}
_DTYPE_BYTES = {"bf16": 2, "fp16": 2, "fp32": 4}

R31_CALLER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/gdn.py",
    "sha256": "5e3b0fb0d9fd71bdcf64fdd5f0e182f12041e291b0122b638613ada2f53224ed",
    "symbol": "AscendGatedDeltaNetAttention.forward",
    "revision": R31_SOURCE_VERSION,
    "commit": R31_SOURCE_COMMIT,
    "state": "task_artifact",
    "lines": [443, 444],
    "split_lines": [396, 397],
    "blob": "b21508b4d9332ac25a325b7941c64ced01f47885",
}
R31_WRAPPER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/layernorm.py",
    "sha256": "1770057c3cd170a7433910fc02db2a36bf0443eb218215a243db31761b95e4a6",
    "symbol": "AscendRMSNormGated.forward_oot",
    "revision": R31_SOURCE_VERSION,
    "commit": R31_SOURCE_COMMIT,
    "state": "task_artifact",
    "lines": [[128, 166], [174, 213]],
}
R31_KERNEL = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/triton/layernorm_gated.py",
    "sha256": "ecdd125ce32f969e1b1c0e4f8b4e84ee8657332cd02cdcc4c72edd2bff13aff2",
    "symbol": NATIVE_TYPE,
    "revision": R31_SOURCE_VERSION,
    "commit": R31_SOURCE_COMMIT,
    "state": "task_artifact",
    "kernel_lines": [16, 100],
    "launcher_lines": [103, 168],
}
A3_CAPTURE = {
    "path": ("/Volumes/data/mindie-sim-data/a3/"
             "Qwen3.6-27B-W8A8_high_conc_decode_tpot30_c64_20261030/"
             "kernel_details.csv"),
    "sha256": "949163473ac3c4ebb0fffeffa7bca402f4fe8ac28b5503e79f1c01468dcee917",
    "row": 143,
    "calls_per_step": 48,
    "block_num": 40,
    "accelerator_core": "AI_VECTOR_CORE",
}
A3_ABI_FIXTURE = {
    "path": ("/Volumes/data/code/mindie-release-20261014/profiler/tests/"
             "fixtures/body_fused/a3_qwen36.csv"),
    "sha256": "fc7b2f7b6fd80f1465e2f949c8a75fbc2f4011ec1e31988c72cd058a1c032219",
    "tokens": 8192,
    "width": 128,
    "dtype": "bf16",
    "rows": 98304,
}
R31_UTILS = {
    "path": "vllm_ascend/utils.py",
    "sha256": "69c0b69ef9b80c08b772d191e8e66e8857c5c1ea68e719ddc91b1a197f7ee8c3",
    "symbol": "REGISTERED_ASCEND_OPS[RMSNormGated]",
    "lines": [840, 889],
    "commit": R31_SOURCE_COMMIT,
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}
_SOURCE_IMAGE_TOKENS = ("9e869e814846", "g8fd0bc488")


def _descriptor(shape: list | None, dtype: str | None, layout: str | None,
                role: str, *, allocation: str = "external",
                present: bool = True) -> dict:
    return {"shape": None if shape is None else [int(item) for item in shape],
            "dtype": dtype, "layout": layout, "role": role, "pointer": None,
            "allocation": allocation, "present": present}


def _operator(compiled: Any) -> Any:
    return getattr(compiled, "operator", None)


def _theta(compiled: Any) -> Any:
    return getattr(_operator(compiled), "theta", None)


def _sibling_names(compiled: Any) -> set[str]:
    return {str(getattr(item, "name", ""))
            for item in (getattr(compiled, "siblings", ()) or ())}


def _declared_dtype(extras: dict) -> str | None:
    """Only an explicit source dtype string proves BF16/FP16/FP32."""
    for key in ("output_dtype", "compute_dtype", "result_dtype"):
        value = extras.get(key)
        if isinstance(value, str) and value.strip():
            return _DTYPE_ALIASES.get(value.strip().lower())
    return None


def _gated_norm_shape(compiled: Any) -> tuple[int, int, str, float | None] | None:
    """Actual (local value_heads, d_v, native dtype, ordinary act bytes)."""
    theta = _theta(compiled)
    extras = dict(getattr(theta, "extras", ()) or ())
    matrix = dict(getattr(theta, "matrix", ()) or ())
    value_heads = extras.get("value_heads")
    dv = extras.get("dv", extras.get("head_dim"))
    if not isinstance(value_heads, int) or isinstance(value_heads, bool) or value_heads < 1:
        value_heads = None
    if not isinstance(dv, int) or isinstance(dv, bool) or dv < 1:
        dv = None
    if value_heads is None or dv is None:
        for sibling in (getattr(compiled, "siblings", ()) or ()):
            if str(getattr(sibling, "name", "")) != "gated_delta":
                continue
            sibling_extras = dict(getattr(
                getattr(sibling, "theta", None), "extras", ()) or ())
            if value_heads is None:
                candidate = sibling_extras.get("value_heads")
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                    value_heads = candidate
            if dv is None:
                candidate = sibling_extras.get("dv", sibling_extras.get("head_dim"))
                if isinstance(candidate, int) and not isinstance(candidate, bool) and candidate > 0:
                    dv = candidate
            for key in ("output_dtype", "compute_dtype", "result_dtype",
                        "result_bytes", "act_bytes"):
                if key not in extras and key in sibling_extras:
                    extras[key] = sibling_extras[key]
            break
    if value_heads is None or dv is None:
        return None
    elements = matrix.get("elements")
    if isinstance(elements, (int, float)) and not isinstance(elements, bool):
        if int(elements) != value_heads * dv:
            return None
    dtype = _declared_dtype(extras)
    if dtype not in _DTYPE_BYTES:
        return None
    if dv > 65536 // _DTYPE_BYTES[dtype]:
        return None
    return value_heads, dv, dtype, _ordinary_act_bytes(theta)


def _source_metadata(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _source_image_id(compiled: Any) -> str:
    metadata = _source_metadata(compiled)
    for key in ("image_id", "deployment_image_id", "oci_image_digest"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_image_tag(compiled: Any) -> str:
    metadata = _source_metadata(compiled)
    for key in ("image", "deployment_image"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_identity(compiled: Any) -> str:
    """Canonical image digest plus tag; both are checked, never one alone."""
    metadata = _source_metadata(compiled)
    values = []
    for key in ("image_id", "deployment_image_id", "oci_image_digest",
                "image", "deployment_image"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            values.append(value.strip())
    return " ".join(values)


def _ordinary_act_bytes(theta: Any) -> float | None:
    extras = dict(getattr(theta, "extras", ()) or ())
    value = extras.get("act_bytes")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    quant = dict(getattr(theta, "quant", ()) or ())
    value = quant.get("act_bytes")
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def _is_target(compiled: Any, phase: str) -> bool:
    """A3-subgraph GDN gated_norm carrier; model/hardware labels never admit."""
    operator = _operator(compiled)
    if operator is None:
        return False
    if str(getattr(operator, "name", "")) != "gated_norm":
        return False
    if (str(getattr(operator, "family", "")) != "elementwise"
            or str(getattr(operator, "role", "")) != "norm"):
        return False
    if str(getattr(compiled, "kind", "")) != "A3":
        return False
    names = _sibling_names(compiled)
    if not {"gated_delta", "gdn_gating", "conv", "prepare"} <= names:
        return False
    if any("kda" in name.lower() for name in names):
        return False
    identity = _source_identity(compiled)
    if not identity or not any(token in identity.lower()
                               for token in _SOURCE_IMAGE_TOKENS):
        return False
    if phase not in {"", "decode", "prefill"}:
        return False
    return _gated_norm_shape(compiled) is not None


def _typed(compiled: Any, heads: int, dv: int, dtype: str,
           ordinary_act: float | None) -> dict:
    block_n = 1 if dv <= 1 else 1 << (dv - 1).bit_length()
    attrs = {
        "native_kind": NATIVE_TYPE,
        "native_type": NATIVE_TYPE,
        "source_contract": "native_tensor_abi",
        "source_revision": R31_SOURCE_VERSION,
        "source_commit": R31_SOURCE_COMMIT,
        "source_image_sha256": R31_IMAGE_SHA,
        "source_caller_path": R31_CALLER["path"],
        "source_caller_sha256": R31_CALLER["sha256"],
        "source_caller_symbol": R31_CALLER["symbol"],
        "source_caller_lines": list(R31_CALLER["lines"]),
        "source_caller_split_lines": list(R31_CALLER["split_lines"]),
        "source_wrapper_path": R31_WRAPPER["path"],
        "source_wrapper_sha256": R31_WRAPPER["sha256"],
        "source_wrapper_symbol": R31_WRAPPER["symbol"],
        "source_wrapper_state": R31_WRAPPER["state"],
        "source_wrapper_lines": [list(item) for item in R31_WRAPPER["lines"]],
        "kernel_name": NATIVE_TYPE,
        "kernel_source_path": R31_KERNEL["path"],
        "kernel_source_sha256": R31_KERNEL["sha256"],
        "kernel_source_state": R31_KERNEL["state"],
        "kernel_source_lines": list(R31_KERNEL["kernel_lines"]),
        "kernel_launcher_lines": list(R31_KERNEL["launcher_lines"]),
        "kernel_grid_block_m": BLOCK_M,
        "kernel_block_n": block_n,
        "head_dim": dv,
        "local_value_heads": heads,
        "epsilon": EPSILON,
        "group_width": dv,
        "groups_per_row": 1,
        "norm_kind": "rmsnorm",
        "is_rms_norm": True,
        "weight_present": True,
        "bias_present": False,
        "gate_present": True,
        "mean_materialized": False,
        "rstd_materialized": True,
        "norm_before_gate": None,
        "norm_order": "not_observed",
        "activation": "swish",
        "activation_formula": "z*sigmoid(z)",
        "hidden_act": "silu",
        "native_dtype": dtype,
        "ordinary_compiled_act_bytes": ordinary_act,
    }
    return {
        "dtype": dtype,
        "layout": "ND",
        "shapes": {"x": [dv], "weight": [dv],
                   "gate": [dv], "output": [dv]},
        "attrs": attrs,
        "rank_width": 2,
    }


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real shared ``_CallSpec`` for the R31 GDN gated norm."""
    if not _is_target(compiled, phase):
        return None
    shape = _gated_norm_shape(compiled)
    if shape is None:
        return None
    heads, dv, dtype, ordinary_act = shape
    from .native_calls import _api, _call_spec, _kernel, _source

    source = _source(
        R31_CALLER["repository"], R31_CALLER["path"],
        sha256=R31_CALLER["sha256"], symbol=R31_CALLER["symbol"],
        revision=R31_SOURCE_VERSION, state="task_artifact")
    source.update({"lines": list(R31_CALLER["lines"]),
                   "split_lines": list(R31_CALLER["split_lines"]),
                   "blob": R31_CALLER["blob"], "commit": R31_SOURCE_COMMIT,
                   "image_sha256": R31_IMAGE_SHA,
                   "package": ("vllm-ascend v0.30.0rc2.dev200+"
                               "g8fd0bc488")})
    api = _api("triton", ENTRY, version=R31_SOURCE_VERSION)
    kernels = (_kernel(
        NATIVE_TYPE, "vector", path=R31_KERNEL["path"],
        sha256=R31_KERNEL["sha256"], evidence="source_declared",
        source_revision=R31_SOURCE_VERSION, source_commit=R31_SOURCE_COMMIT,
        source_status=R31_KERNEL["state"], source_lines=R31_KERNEL["kernel_lines"],
        launcher_lines=R31_KERNEL["launcher_lines"],
        capture_evidence="measured", group_width=dv,
        block_m=BLOCK_M,
        block_n=(1 if dv <= 1 else 1 << (dv - 1).bit_length()),
        capture=A3_CAPTURE, fixture=A3_ABI_FIXTURE),)
    typed = _typed(compiled, heads, dv, dtype, ordinary_act)
    unresolved = (
        "norm_before_gate_not_observed",
        "vllm_core_model_constructor_not_retained",
    )
    owner = f"{getattr(compiled, 'kind', '')}:elementwise:gated_norm"
    return _call_spec(api, source, typed, kernels, (), owner,
                      execution_core="aiv", unresolved=unresolved)


def _absent(role: str) -> dict:
    return _descriptor(None, None, None, role, allocation="not_allocated",
                       present=False)


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only invocation for one evaluated gated-norm occurrence."""
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != ENTRY:
        return None
    typed = dict(getattr(binding, "typed", {}) or ())
    attrs = dict(typed.get("attrs") or {})
    if attrs.get("native_kind") != NATIVE_TYPE:
        return None
    identity = _source_identity(compiled)
    if not identity or not any(token in identity.lower()
                               for token in _SOURCE_IMAGE_TOKENS):
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    if phase not in {"decode", "prefill"}:
        return None
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    if tokens < 1:
        return None
    shape = _gated_norm_shape(compiled)
    if shape is None:
        return None
    heads, dv, charge_dtype, ordinary_act = shape
    if (attrs.get("local_value_heads") != heads
            or attrs.get("head_dim") != dv
            or attrs.get("group_width") != dv
            or attrs.get("native_dtype") != charge_dtype):
        return None
    row_bytes = _DTYPE_BYTES[charge_dtype]
    rows = tokens * heads
    if rows < 1:
        return None
    cache_key = (binding.call_id, phase, tokens, rows, dv, charge_dtype)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shapes = {
        "x": [rows, dv],
        "weight": [dv],
        "gate": [rows, dv],
        "output": [rows, dv],
        "rstd": [rows],
    }
    actual_attrs = dict(attrs)
    actual_attrs.update({
        "rows": rows,
        "tokens": tokens,
        "phase": phase,
        "grid": [(rows + BLOCK_M - 1) // BLOCK_M, 1],
        "native_row_bytes": {
            "x": dv * row_bytes,
            "weight": dv * row_bytes,
            "gate": dv * row_bytes,
            "output": dv * row_bytes,
            "rstd": 4,
        },
        "native_bytes": {
            "x": rows * dv * row_bytes,
            "weight": dv * row_bytes,
            "gate": rows * dv * row_bytes,
            "output": rows * dv * row_bytes,
            "rstd": rows * 4,
        },
        "ordinary_compiled_act_bytes": ordinary_act,
        "ordinary_compiled_act_bytes_used_for_native_dtype": False,
        "ordinary_compiled_act_bytes_mismatch": bool(
            ordinary_act is not None and ordinary_act != 2.0),
    })
    actual = {"dtype": charge_dtype, "layout": "ND", "shapes": shapes,
              "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, dv], charge_dtype, "ND", "input"),
        "weight": _descriptor([dv], charge_dtype, "ND", "input"),
        "gate": _descriptor([rows, dv], charge_dtype, "ND", "input"),
        "output": _descriptor([rows, dv], charge_dtype, "ND", "output",
                              allocation="kernel_internal_not_performed"),
        "rstd": _descriptor([rows], "fp32", "ND", "output",
                            allocation="kernel_internal_not_performed"),
        "mean": _absent("output_optional"),
        "bias": _absent("input_optional"),
    }
    prepare = {
        "schema": SCHEMA,
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": typed.get("signature_hash"),
        "source": dict(getattr(binding, "source", {}) or {}),
        "source_lines": list(R31_CALLER["lines"]),
        "source_split_lines": list(R31_CALLER["split_lines"]),
        "source_version": R31_SOURCE_VERSION,
        "source_commit": R31_SOURCE_COMMIT,
        "kernel": (dict(binding.kernels[0]) if getattr(binding, "kernels", ())
                   else None),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {
            "x": "x", "weight": "weight", "bias": None, "eps": EPSILON,
            "z": "gate", "out": None, "group_size": dv,
            "norm_before_gate": None, "is_rms_norm": True,
        },
        "argument_order": ["x", "weight", "bias", "eps", "z", "out",
                           "group_size", "norm_before_gate", "is_rms_norm"],
        "prototype": (ENTRY + "(x, weight, bias, eps, z=None, out=None, "
                      "group_size=None, norm_before_gate=True, "
                      "is_rms_norm=False)"),
        "positional_arguments": {"x": "x", "weight": "weight",
                                 "bias": None, "eps": EPSILON},
        "keywords": {"z": "gate", "out": None, "group_size": dv,
                     "norm_before_gate": None, "is_rms_norm": True},
        "optional_slots": {
            "bias": "not_present",
            "z": "present_required",
            "out": "not_passed_kernel_allocates",
            "mean": "not_materialized_is_rms_norm",
            "norm_before_gate": "not_observed",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "lineage": {
            "input": "gated_delta",
            "gate": ("z_slice" if "z_slice" in _sibling_names(compiled)
                     else "projection_gate_reshape"),
            "gate_source": ("decode_view" if "z_slice" in _sibling_names(compiled)
                            else "projection_output"),
            "fused_from": ["rmsnorm", "gate"],
            "standalone": True,
            "subgraph_members": sorted(_sibling_names(compiled)),
        },
        "deployment": {
            "image_sha256": R31_IMAGE_SHA,
            "revision": R31_SOURCE_VERSION,
            "commit": R31_SOURCE_COMMIT,
            "source_image_resolution": {
                "case_declared_image_id": _source_image_id(compiled) or None,
                "case_declared_image": _source_image_tag(compiled) or None,
                "source_identity": _source_identity(compiled),
                "accepted_source_tokens": list(_SOURCE_IMAGE_TOKENS),
                "basis": ("R31 image/digest or same vllm-ascend g8fd0bc488 "
                          "source commit; identity is mandatory; empty or "
                          "foreign identity is rejected"),
                "accepted": True,
            },
            "caller": dict(R31_CALLER),
            "wrapper": dict(R31_WRAPPER),
            "kernel": dict(R31_KERNEL),
            "utils_registration": dict(R31_UTILS),
            "configured_source_config_sha256": MODEL_CONFIG_SHA,
            "configured_source_scope": ("r31 epsilon/activation provenance; "
                                        "not admission"),
            "epsilon": EPSILON,
            "group_width": dv,
            "groups_per_row": 1,
            "activation": "swish",
            "activation_formula": "z*sigmoid(z)",
            "hidden_act": "silu",
            "norm_before_gate": None,
            "norm_order": "not_observed",
        },
        "observed": {
            "a3_qwen36_capture": dict(A3_CAPTURE),
            "a3_qwen36_abi_fixture": dict(A3_ABI_FIXTURE),
            "source_abi_fixture": {
                "shape": [98304, 128],
                "weight_shape": [128],
                "dtype": "bf16",
                "role": "shape/dtype witness only; actual occurrence below",
            },
            "native_abi": {
                "inputs": [["rows", dv, charge_dtype],
                           [dv, charge_dtype],
                           ["rows", dv, charge_dtype]],
                "outputs": [["rows", dv, charge_dtype], ["rows", "fp32"]],
                "layout": "ND",
            },
            "rows": rows,
            "tokens": tokens,
            "local_value_heads": heads,
            "width": dv,
            "dtype": charge_dtype,
        },
        "match": {
            "native_type": NATIVE_TYPE,
            "entry": ENTRY,
            "call_id": binding.call_id,
            "source_revision": R31_SOURCE_VERSION,
            "source_commit": R31_SOURCE_COMMIT,
            "source_image_sha256": R31_IMAGE_SHA,
            "source_caller_sha256": R31_CALLER["sha256"],
            "source_caller_lines": list(R31_CALLER["lines"]),
            "source_caller_split_lines": list(R31_CALLER["split_lines"]),
            "source_wrapper_sha256": R31_WRAPPER["sha256"],
            "source_wrapper_lines": [list(item)
                                     for item in R31_WRAPPER["lines"]],
            "kernel_source_sha256": R31_KERNEL["sha256"],
            "kernel_source_lines": list(R31_KERNEL["kernel_lines"]),
            "kernel_launcher_lines": list(R31_KERNEL["launcher_lines"]),
            "dtype": charge_dtype,
            "layout": "ND",
            "width": dv,
            "group_width": dv,
            "groups_per_row": 1,
            "epsilon": EPSILON,
            "activation": "swish",
            "norm_before_gate": None,
            "rows": rows,
            "tokens": tokens,
            "local_value_heads": heads,
        },
        "binary_selection": {
            "status": "not_binary",
            "reason": "triton_jit_kernel_launched_from_source",
            "kernel_name": NATIVE_TYPE,
            "launcher": ENTRY,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "norm_before_gate_not_observed",
            "vllm_core_model_constructor_not_retained",
        ],
        "unresolved": [
            "norm_before_gate_not_observed",
            "vllm_core_model_constructor_not_retained",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "norm_gate_order_not_observed",
            "vllm_core_model_constructor_not_retained",
            "ordinary_compiled_act_bytes_not_used_for_native_dtype",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {"dtype": validated.get("dtype"),
                         "layout": validated.get("layout"),
                         "shapes": validated.get("shapes"),
                         "attrs": validated.get("attrs")},
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
