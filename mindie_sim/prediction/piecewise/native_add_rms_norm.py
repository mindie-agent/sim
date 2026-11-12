"""Prepare-only native AddRmsNorm call for the deployed R31 Qwen3.6 path.

[Role] Bind the real ``aclnnAddRmsNorm``/``AddRmsNorm`` call recorded by the
  original R31 rank capture to the normal N0 fused residual+norm occurrence
  and to the terminal O0 normalization.  Expand only the source-recorded
  tensor roles/geometry; do not re-price the existing fused operator cost.
[Invariants] The image is the exact R31 deployment digest, the kernel is the
  captured ``AddRmsNorm`` on AI_VECTOR_CORE with BF16 x/residual/weight and
  BF16/F32/BF16 outputs.  The local width and SP rows come from the compiled
  theta; captured sample dimensions stay in the evidence block only.  The
  initial plain RmsNorm (no residual sibling) is the disjoint OPRMS2 leaf; it
  and every other image, bias/quant norm, or unsupported geometry return
  ``None``.  The prepare is explicitly partial: the Python caller module and
  ACLNN positional slots were not retained, so no complete native execution is
  claimed.
[Timeouts & Locks] Bounded memo of 32 occurrence descriptors; no waits/locks.
[Side-Effect Policy] Pure metadata; no tensor allocation, device import or I/O.
"""
from __future__ import annotations

import math
from typing import Any

R31_IMAGE_ID = "sha256:9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_TAG = (
    "swr.cn-north-12.myhuaweicloud.com/atlas_inference/vllm-ascend:"
    "nightly-ci-releases-v0.30.0-a5-cann9.2.0_20261022115200-20261023-r31"
)
CANN_REVISION = "cann-9.2.0-r31"
CANN_LIBOPAPI = {
    "repository": "cann",
    "path": "/usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": "AddRmsNorm",
    "revision": CANN_REVISION,
    "state": "task_artifact",
}
CAPTURE = {
    "path": "obs-r4/qwen-real-norm-order.json",
    "sha256": "7d1f93bcd26ed8ffb7a7eb925d6c58a010a6e20ad7527c34ed3e665c5a04b5ff",
    "raw_path": (
        "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
        "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
        "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz"
    ),
    "raw_sha256": "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a",
    "sequence_sha256": "ecddce9e05631a21e623c9654269bee2a2b8ed95f640bc24e19e8f07eb420b29",
    "cell": "smoke_qwen36_single_pf_4k_c1",
    "raw_row_index": 76,
    "raw_row_index_base": "1-based_csv_record",
    "raw_row_index_0based": 75,
    "name": "aclnnAddRmsNorm_AddRmsNorm_AddRmsNorm",
    "type": "AddRmsNorm",
    "accelerator_core": "AI_VECTOR_CORE",
    "block_num": 64,
    "mix_block_num": 0,
    "input_shapes": "2048,5120;2048,5120;5120",
    "input_dtypes": "DT_BF16;DT_BF16;DT_BF16",
    "input_formats": "ND;ND;ND",
    "output_shapes": "2048,5120;2048,1;2048,5120",
    "output_dtypes": "DT_BF16;FLOAT;DT_BF16",
    "output_formats": "ND;ND;ND",
    "captured_rows": 2048,
}
_DYNAMIC_CACHE: dict[tuple, dict] = {}
_MAX_CACHE = 32
_ENTRY = "aclnnAddRmsNorm"
_KERNEL = "AddRmsNorm"


def _feature(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _theta(compiled: Any) -> Any:
    return getattr(getattr(compiled, "operator", None), "theta", None)


def _find(compiled: Any, key: str, default: Any = None) -> Any:
    theta = _theta(compiled)
    if theta is None or not hasattr(theta, "find"):
        return default
    return theta.find(key, default)


def _siblings(compiled: Any) -> set[str]:
    return {str(getattr(item, "name", "")) for item in
            (getattr(compiled, "siblings", ()) or ())}


def _norm_gate(compiled: Any) -> bool:
    """Weighted BF16 RMS norm on R31; no bias, quant, Gemma shift or LayerNorm."""
    theta = _theta(compiled)
    if theta is None or not hasattr(theta, "find"):
        return False
    if str(getattr(getattr(compiled, "operator", None), "family", "")) != "elementwise":
        return False
    if int(_find(compiled, "layernorm", 0) or 0):
        return False
    if float(_find(compiled, "norm_weight_offset", 0.0) or 0.0) != 0.0:
        return False
    if str(_find(compiled, "norm_arch", "") or "") not in {"", "rmsnorm"}:
        return False
    if bool(_find(compiled, "norm_has_bias", False)):
        return False
    if str(_find(compiled, "native_norm_kind", "") or "") in {
            "AddRmsNormBias", "AddRmsNormDynamicQuant",
            "AddRmsNormDynamicMxQuant"}:
        return False
    if bool(int(_find(compiled, "quantized", 0) or 0)):
        return False
    if int(_find(compiled, "quant_passes", 0) or 0):
        return False
    if str(_find(compiled, "output_format", "") or "") not in {"", "bf16"}:
        return False
    hidden = int(_find(compiled, "elements", 0) or 0)
    gamma_bytes = float(_find(compiled, "gamma_bytes", 0.0) or 0.0)
    gamma_format = str(_find(compiled, "gamma_format", "") or "")
    if hidden < 1 or gamma_format != "bf16":
        return False
    if gamma_bytes <= 0 or abs(gamma_bytes - hidden * 2) > 1e-6:
        return False
    sequence_rows = str(_find(compiled, "sequence_rows", "") or "")
    if sequence_rows not in {"", "shard", "selected"}:
        return False
    if int(_find(compiled, "sequence_ranks", 1) or 1) < 1:
        return False
    return True


def _select_kind(compiled: Any) -> str | None:
    kind = str(getattr(compiled, "kind", "") or "")
    name = str(getattr(getattr(compiled, "operator", None), "name", "") or "")
    if kind == "N0" and name == "rmsnorm":
        if "residual_add" not in _siblings(compiled):
            return None  # the initial plain RmsNorm is OPRMS, not this call
        return "inner" if _norm_gate(compiled) else None
    if kind == "O0" and name == "final_norm":
        if not _norm_gate(compiled):
            return None
        if int(_find(compiled, "residual_add", 0) or 0) != 1:
            return None
        if str(_find(compiled, "fusion", "") or "") != "add_rmsnorm":
            return None
        if float(_find(compiled, "rstd_bytes", 0.0) or 0.0) <= 0:
            return None
        if float(_find(compiled, "residual_out_bytes", 0.0) or 0.0) <= 0:
            return None
        return "terminal"
    return None


def _r31_source(compiled: Any) -> bool:
    meta = _feature(compiled)
    return (str(meta.get("image_id") or "") == R31_IMAGE_ID
            and "r31" in str(meta.get("image") or "")
            and "cann9.2.0" in str(meta.get("image") or ""))


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real ``_CallSpec`` for R31 N0/O0 AddRmsNorm, else ``None``."""
    if not _r31_source(compiled):
        return None
    kind = _select_kind(compiled)
    if kind is None:
        return None
    from .native_calls import _api, _call_spec, _kernel, _source

    hidden = int(_find(compiled, "elements", 0) or 0)
    if hidden < 1:
        return None
    ranks = int(_find(compiled, "sequence_ranks", 1) or 1)
    sequence_rows = str(_find(compiled, "sequence_rows", "") or "")
    source = _source(
        CANN_LIBOPAPI["repository"], CANN_LIBOPAPI["path"],
        sha256=CANN_LIBOPAPI["sha256"], symbol=CANN_LIBOPAPI["symbol"],
        revision=CANN_LIBOPAPI["revision"], state=CANN_LIBOPAPI["state"])
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {
            "x": [hidden], "residual": [hidden], "gamma": [hidden],
            "output": [hidden], "rstd": [1], "residual_out": [hidden],
        },
        "attrs": {
            "hidden": hidden,
            "sequence_rows": sequence_rows,
            "sequence_ranks": ranks,
            "input_layout": "ND",
            "rstd_dtype": "fp32",
            "output_dtype": "bf16",
            "residual_dtype": "bf16",
            "call_name": _ENTRY,
            "kernel_name": _KERNEL,
        },
        "rank_width": 2,
    }
    kernels = (_kernel(
        _KERNEL, "vector", path=CANN_LIBOPAPI["path"],
        sha256=CANN_LIBOPAPI["sha256"], evidence="runtime_binary"),)
    owner = ("fused:add_rmsnorm" if kind == "inner"
             else "O0:elementwise:final_norm")
    members = ("residual_add", "rmsnorm") if kind == "inner" else ()
    return _call_spec(
        _api("aclnn", _ENTRY, aclnn_symbol=_ENTRY, version="cann-9.2.0"),
        source, typed, kernels, members, owner, execution_core="aiv",
        unresolved=[
            "python_caller_module_not_retained",
            "capture_python_stack_absent",
            "kernel_implementation_source_retained_as_runtime_binary",
            "aclnn_positional_slots_not_retained",
        ])


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _rows(compiled: Any, geometry: Any, sequence_rows: str, ranks: int) -> int:
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    if tokens < 1:
        return 0
    if sequence_rows == "selected":
        selected = getattr(geometry, "producing_lm_head_rows", None)
        if not callable(selected):
            return 0
        return int(selected() or 0)
    if sequence_rows == "shard" and ranks > 1:
        if int(getattr(geometry, "operand_tokens", 0) or 0) > 0:
            return tokens
        return int(math.ceil(tokens / ranks))
    return tokens


def _eps(compiled: Any) -> float | None:
    value = _find(compiled, "norm_eps", None)
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value)
    return None


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R31 AddRmsNorm call."""
    api = dict(getattr(binding, "api", {}) or {})
    if api.get("entry") != _ENTRY:
        return None
    source = dict(getattr(binding, "source", {}) or {})
    if (source.get("sha256") != CANN_LIBOPAPI["sha256"]
            or str(source.get("revision") or "") != CANN_REVISION):
        return None
    attrs = dict((getattr(binding, "typed", {}) or {}).get("attrs") or {})
    hidden = int(attrs.get("hidden", 0) or 0)
    if hidden < 1:
        return None
    sequence_rows = str(attrs.get("sequence_rows", "") or "")
    ranks = int(attrs.get("sequence_ranks", 1) or 1)
    rows = _rows(compiled, geometry, sequence_rows, ranks)
    if rows < 1:
        return None
    phase = str(getattr(geometry, "phase_name", "") or "")
    tokens = int(getattr(geometry, "effective_tokens", 0) or 0)
    eps = _eps(compiled)
    image = str(_feature(compiled).get("image") or R31_IMAGE_TAG)
    cache_key = (str(getattr(binding, "call_id", "")), rows, tokens, phase,
                 sequence_rows, eps, image)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {
        "x": [rows, hidden],
        "residual": [rows, hidden],
        "gamma": [hidden],
        "output": [rows, hidden],
        "rstd": [rows, 1],
        "residual_out": [rows, hidden],
    }
    actual_attrs = {
        **attrs,
        "rows": rows,
        "tokens": tokens,
        "phase": phase,
        "rstd_bytes": rows * 4,
        "residual_bytes": rows * hidden * 2,
        "norm_output_bytes": rows * hidden * 2,
        "physical_rows": rows,
        "sp_rows": sequence_rows or "all",
        "sp_ranks": ranks,
        "row_basis": sequence_rows or "all",
    }
    if eps is not None:
        actual_attrs["eps"] = eps
        actual_attrs["eps_source"] = "model_config_declared"
    actual = {"dtype": "bf16", "layout": "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, hidden], "bf16", "ND", "input"),
        "residual": _descriptor([rows, hidden], "bf16", "ND", "input"),
        "gamma": _descriptor([hidden], "bf16", "ND", "weight"),
        "output": _descriptor([rows, hidden], "bf16", "ND", "output"),
        "rstd": _descriptor([rows, 1], "fp32", "ND", "output"),
        "residual_out": _descriptor([rows, hidden], "bf16", "ND", "output"),
    }
    prepare = {
        "schema": "mindie-sim/add-rms-norm-native-prepare/1",
        "call": _ENTRY,
        "call_id": getattr(binding, "call_id", None),
        "signature_hash": (getattr(binding, "typed", {}) or {}).get("signature_hash"),
        "source": source,
        "kernel": (dict(getattr(binding, "kernels", ())[0])
                   if getattr(binding, "kernels", ()) else
                   {"name": _KERNEL, "role": "vector",
                    "evidence": "runtime_binary"}),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "acl_prepare_only",
            "python_caller_module_not_retained",
            "capture_python_stack_absent",
            "aclnn_positional_slots_not_retained",
        ],
        "shapes": actual_shapes,
        "validated_shapes": validated.get("shapes"),
        "attrs": actual_attrs,
        "descriptors": descriptors,
        "arguments": {
            "input": "x",
            "residual_input": "residual",
            "weight": "gamma",
            "epsilon": eps,
            "norm_output": "output",
            "rstd": "rstd",
            "residual_output": "residual_out",
        },
        "argument_order_status": (
            "tensor_roles_and_dtypes_from_captured_aclnn_call; "
            "positional_slots_not_retained"),
        "outputs": {
            "norm_output": descriptors["output"],
            "rstd": descriptors["rstd"],
            "residual": descriptors["residual_out"],
        },
        "output_count": 3,
        "aliases": {},
        "alias_status": "not_established",
        "capture": dict(CAPTURE),
        "deployment": {
            "image_id": R31_IMAGE_ID,
            "image": image,
            "revision": "r31/9e869e8146",
            "source_sha256": CANN_LIBOPAPI["sha256"],
            "capture_raw_sha256": CAPTURE["raw_sha256"],
            "capture_sequence_sha256": CAPTURE["sequence_sha256"],
        },
        "match": {
            "native_type": _KERNEL,
            "entry": _ENTRY,
            "call_id": getattr(binding, "call_id", None),
            "image_id": R31_IMAGE_ID,
            "source_revision": CANN_REVISION,
            "source_sha256": CANN_LIBOPAPI["sha256"],
            "hidden": hidden,
            "rows": rows,
            "tokens": tokens,
            "row_basis": sequence_rows or "all",
            "dtype": "bf16",
            "gamma_dtype": "bf16",
            "rstd_dtype": "fp32",
            "residual_dtype": "bf16",
            "status": "partial_prepare_not_executed",
        },
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "python_caller_module_not_retained",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": actual["dtype"],
            "layout": actual["layout"],
            "shapes": actual_shapes,
            "attrs": actual_attrs,
        },
    }
    if len(_DYNAMIC_CACHE) >= _MAX_CACHE:
        _DYNAMIC_CACHE.pop(next(iter(_DYNAMIC_CACHE)))
    _DYNAMIC_CACHE[cache_key] = result
    return result
