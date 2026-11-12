"""Prepare-only R31 standalone DynamicQuantV2 invocation.

[Role] Expand the normal graph's ``*_QuantV2`` producer ahead of each
  ``w8a8_dynamic`` INT8 projection into the actual R31 ACLNN
  ``aclnnDynamicQuantV2`` surface: one BF16 row input, an INT8 payload, and
  the separately materialized FP32 per-row scale and offset outputs.
[Invariants] The R31 image digest is required; the call identity is the
  retained CANN 9.2.0 libopapi host surface, never a model label, profiler
  symbol or kernel-name guess.  M lives only in the occurrence.  The exact
  vllm-ascend W8A8_DYNAMIC Python wrapper is not retained for r31, so the
  caller/shape lowering and ``isSymmetrical`` stay explicit unresolved fields;
  this module never claims a complete native execution.  The offset output is
  retained because the independent profiler contract requires the real
  three-output ABI.  Decode and non-r31 images return ``None`` so ordinary
  prediction stays unchanged.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected normal operator
  declaration, the retained R31 host binary and the evaluated geometry.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_IMAGE_PREFIX = "9e869e814846"
R31_REVISION = "r31/9e869e814846"
R31_CANN = "cann-9.2.0-r31"
ENTRY = "torch_npu.npu_dynamic_quant"
ACLNN_API = "aclnnDynamicQuantV2"
KERNEL_NAME = "DynamicQuantV2"
PROFILER_NAME = ("aclnnDynamicQuantV2_DynamicQuantV2_"
                 "DynamicQuantV2")
OP_SURFACE = ("DynamicQuantV2(x, smoothScalesOptional, groupIndexsOptional, "
              "dstType, isSymmetrical, quantMode) -> (y, scale, offset)")
# Retained same-image host side of the ACLNN call.  The device kernel binary
# is not in this bundle; the profiler row proves the measured kernel name and
# the MIX_AIV execution class.
CANNN_LIB = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456",
    "symbol": ACLNN_API,
    "revision": R31_CANN,
    "state": "task_artifact",
}
# Same-image Python declaration of the torch_npu entry.  This is the API
# declaration only; it is not the model's W8A8_DYNAMIC linear wrapper.
DEVICE_OP = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/device/device_op.py",
    "sha256": "f4392f346816c85f138d7e033a924c6b628ae684cd4fcd7a6d8818518351b5c9",
    "symbol": "BaseDeviceAdaptor.npu_dynamic_quant",
    "revision": R31_REVISION,
    "state": "task_artifact",
    "lines": [316, 327],
    "role": "api_declaration_not_model_caller",
}
# Exact R31 caller/shape lowering is not in the retained source set.  Keep the
# likely site named but not admitted; the status records the proof gap.
PYTHON_CALLER = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    "symbol": "AscendW8A8DynamicLinearMethod.apply",
    "revision": R31_REVISION,
    "status": "not_retained",
    "evidence": ("normal case declares quant_method=w8a8_dynamic and each of "
                 "the 208 int8 projections has one preceding QuantV2 creator"),
}
# Same-family wrapper retained for r35, not admitted as r31 source bytes.
R35_PATTERN = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/quantization/methods/w8a8/w8a8_dynamic.py",
    "sha256": "a67601ee596d9aa4ae72ea3638f67ff7236cd8df884564bcd19f33653a1e0910",
    "symbol": "AscendW8A8DynamicLinearMethod.apply",
    "revision": "r35/5461a1a8",
    "status": "same_family_pattern_not_r31_evidence",
}
# Only the structural sites the original r31 capture independently inventories.
_SUPPORTED_SITES = frozenset({
    ("F0", "dense_mlp", "gate_up_QuantV2"),
    ("F0", "dense_mlp", "down_QuantV2"),
    ("A0", "attn.qkv", "qkv_projection_QuantV2"),
    ("A0", "attn.o", "o_projection_QuantV2"),
    ("A3", "gdn", "qkvz_projection_QuantV2"),
})
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _r31_metadata(compiled: Any) -> dict[str, Any]:
    """The witnessed r31 deployment only; other images stay unbound."""
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    identity = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    digest = identity.split(":", 1)[-1].strip().lower()
    if not digest.startswith(R31_IMAGE_PREFIX):
        return {}
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    if image and "-r31" not in image:
        return {}
    return meta


def _phase_name(compiled: Any, phase: str) -> str:
    return str(phase or getattr(compiled, "phase", "")
               or dict(getattr(compiled, "feature_metadata", ()) or ()).get("phase")
               or "")


def _descriptor(shape: list[int], dtype: str, role: str,
                **extra: Any) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": "ND", "role": role, "pointer": None,
            "allocation": "external", **extra}


def _site(compiled: Any) -> tuple[str, str, str, int] | None:
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if str(getattr(operator, "family", "") or "") != "quantize":
        return None
    theta = operator.theta
    extras = dict(theta.extras)
    kind = str(getattr(compiled, "kind", "") or "")
    category = str(extras.get("category", "") or "")
    name = str(getattr(operator, "name", "") or "")
    if (kind, category, name) not in _SUPPORTED_SITES:
        return None
    if extras.get("quant_op") != "QuantV2":
        return None
    if str(extras.get("out_format", "") or "").lower() != "int8":
        return None
    if str(extras.get("source_format", "") or "").lower() != "bf16":
        return None
    if _int(extras.get("source_bytes"), 0) != 2:
        return None
    if _int(extras.get("row_scale_bytes"), 0) != 4:
        return None
    if _int(extras.get("offset_gm_bytes"), -1) != 0:
        return None
    if _int(extras.get("symmetrical"), 0) != 1:
        return None
    if _int(extras.get("has_smooth"), -1) != 0:
        return None
    if _int(extras.get("quant_passes"), 0) != 2:
        return None
    if str(extras.get("block_axis", "") or "") != "row":
        return None
    width = _int(dict(theta.matrix).get("elements"), 0)
    if width < 1:
        return None
    return kind, category, name, width


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The real R31 QuantV2 call identity, or ``None``.

    The shared normal-entry owner calls this from the existing ``_spec_for``
    chain with the evaluated phase.  Only the prefill QuantV2 guard whose
    normal declaration is an int8 per-token BF16 producer on the witnessed
    r31 image is admitted.  Decode, other images and other quant recipes
    return ``None``.
    """
    if _phase_name(compiled, phase) != "prefill":
        return None
    if not _r31_metadata(compiled):
        return None
    site = _site(compiled)
    if site is None:
        return None
    kind, category, name, width = site
    operator = compiled.operator
    from . import native_calls
    source = native_calls._source(
        CANNN_LIB["repository"], CANNN_LIB["path"],
        sha256=CANNN_LIB["sha256"], symbol=ACLNN_API,
        revision=CANNN_LIB["revision"], state=CANNN_LIB["state"])
    kernel = native_calls._kernel(
        KERNEL_NAME, "vector", evidence="runtime_binary",
        acl_api=ACLNN_API, profiler_name=PROFILER_NAME,
        kernel_source_status="kernel_source_not_retained")
    typed = {
        "dtype": "int8",
        "layout": "ND",
        "phase": "prefill",
        "shapes": {"x": [width], "y": [width]},
        "attrs": {
            "acl_api": ACLNN_API,
            "dst_type": "int8",
            "input_arity": 1,
            "is_symmetrical": None,
            "offset_format": "fp32",
            "offset_materialized": True,
            "output_arity": 3,
            "output_format": "int8",
            "outputs": ["y", "scale", "offset"],
            "quant_mode": "pertoken",
            "row_scale_bytes": 4,
            "row_width": width,
            "scale_format": "fp32",
            "source_format": "bf16",
            "source_operation": KERNEL_NAME,
        },
        "rank_width": 2,
    }
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_API,
                          version=R31_REVISION),
        source, typed, (kernel,), (), str(operator.name),
        execution_core="aiv",
        unresolved=(
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
            "dynamic_quant_binary_selection_unresolved",
            "dynamic_quant_is_symmetrical_not_declared",
        ))


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only typed occurrence for the selected R31 QuantV2 creator.

    Returns the same ``{"actual", "prepare", "prepare_args"}`` envelope as
    ``native_qbmm.dynamic_qbmm``.  The timing owner is the existing
    ``*_QuantV2`` row itself; the consuming projection is not charged or
    attributed here.
    """
    api = binding.api or {}
    if api.get("entry") != ENTRY or api.get("aclnn_symbol") != ACLNN_API:
        return None
    if (binding.source or {}).get("sha256") != CANNN_LIB["sha256"]:
        return None
    meta = _r31_metadata(compiled)
    if not meta:
        return None
    if str(getattr(geometry, "phase_name", "") or "") != "prefill":
        return None
    attrs = dict(binding.typed.get("attrs") or {})
    if (attrs.get("source_operation") != KERNEL_NAME
            or attrs.get("output_format") != "int8"
            or attrs.get("outputs") != ["y", "scale", "offset"]):
        return None
    width = _int(attrs.get("row_width"))
    if width < 1:
        return None
    from mindie_sim.prediction.dynamic_quant_schedule import quant_row_demand
    theta = compiled.operator.theta
    tp = _int(dict(theta.extras).get("tp"), 1)
    demand = quant_row_demand(
        geometry, kind=str(getattr(compiled, "kind", "") or ""), tp=tp)
    rows = int(demand.rows)
    if rows < 1:
        return None
    cache_key = (binding.call_id, rows)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    shapes = {"x": [rows, width], "y": [rows, width],
              "scale": [rows], "offset": [rows]}
    actual = {
        "dtype": "int8",
        "layout": "ND",
        "shapes": shapes,
        "attrs": {**attrs, "tokens": rows, "valid_rows": int(demand.valid),
                  "row_source": str(demand.source)},
    }
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, width], "bf16", "input"),
        "y": _descriptor([rows, width], "int8", "output"),
        "scale": _descriptor([rows], "fp32", "output", per_row=True),
        "offset": _descriptor([rows], "fp32", "output", per_row=True,
                              materialized=True),
    }
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or R31_IMAGE_SHA)
    image = str(meta.get("image") or meta.get("deployment_image") or "")
    projection = str(getattr(compiled.operator, "name", "") or "")
    if projection.endswith("_QuantV2"):
        projection = projection[:-len("_QuantV2")]
    prepare = {
        "schema": "mindie-sim/dynamic-quant-native-prepare/1",
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": binding.typed.get("signature_hash"),
        "native_type": KERNEL_NAME,
        "op_id": binding.op_id,
        "source": dict(binding.source or {}),
        "source_declaration": dict(DEVICE_OP),
        "caller": {**PYTHON_CALLER, "status": "not_retained",
                   "r35_pattern": dict(R35_PATTERN)},
        "kernel": {
            "name": KERNEL_NAME, "role": "vector",
            "source_path": CANNN_LIB["path"],
            "source_sha256": CANNN_LIB["sha256"],
            "evidence": "runtime_binary",
            "acl_api": ACLNN_API,
            "profiler_name": PROFILER_NAME,
            "kernel_source_status": "kernel_source_not_retained",
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "op_surface": OP_SURFACE,
        "aclnn_c_abi": "not_retained",
        "arguments": {
            "x": "x",
            "smooth_scales": None,
            "group_indexs": None,
            "dst_type": "int8",
            "is_symmetrical": None,
            "quant_mode": "pertoken",
        },
        "argument_order": ["x", "smooth_scales", "group_indexs", "dst_type",
                           "is_symmetrical", "quant_mode"],
        "optional_slots": {"smooth_scales": "not_present",
                           "group_indexs": "not_present"},
        "outputs": ["y", "scale", "offset"],
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "match": {
            "native_type": KERNEL_NAME,
            "entry": ENTRY,
            "aclnn_symbol": ACLNN_API,
            "call_id": binding.call_id,
            "op_id": binding.op_id,
            "kind": str(getattr(compiled, "kind", "") or ""),
            "projection": projection,
            "tokens": rows,
            "valid_rows": int(demand.valid),
            "row_width": width,
            "source_format": "bf16",
            "output_format": "int8",
            "scale_format": "fp32",
            "offset_format": "fp32",
            "offset_materialized": True,
            "quant_mode": "pertoken",
            "is_symmetrical": None,
            "deployment_image_id": image_id,
            "source_revision": R31_REVISION,
            "source_sha256": CANNN_LIB["sha256"],
            "profiler_name": PROFILER_NAME,
            "kernel": KERNEL_NAME,
            "execution_core": "aiv",
            "python_caller_status": "not_retained",
        },
        "deployment": {
            "image": image,
            "image_id": image_id,
            "revision": R31_REVISION,
            "cann": "9.2.0",
            "host_interface": dict(CANNN_LIB),
            "python_api_declaration": dict(DEVICE_OP),
            "python_caller": {**PYTHON_CALLER, "status": "not_retained",
                              "r35_pattern": dict(R35_PATTERN)},
            "source_selection": (
                "normal quantize declaration quant_op=QuantV2, "
                "out_format=int8, source_format=bf16, row_scale_bytes=4"),
            "shape_lowering": (
                "K from the normal QuantV2 declaration; rows from "
                "dynamic_quant_schedule.quant_row_demand"),
            "attr_selection": (
                "dst_type/quant_mode from captured output contract; "
                "is_symmetrical stays unresolved without the r31 wrapper"),
            "binary_selection": "unresolved",
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_is_symmetrical_not_declared",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
        ],
        "unresolved": [
            "dynamic_quant_python_caller_not_retained",
            "dynamic_quant_is_symmetrical_not_declared",
            "dynamic_quant_kernel_source_not_retained",
            "dynamic_quant_aclnn_c_abi_not_retained",
            "dynamic_quant_binary_selection_unresolved",
        ],
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "not_a_complete_native_execution",
        ],
    }
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": validated.get("dtype"),
            "layout": validated.get("layout"),
            "shapes": dict(validated.get("shapes") or {}),
            "attrs": dict(validated.get("attrs") or {}),
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
