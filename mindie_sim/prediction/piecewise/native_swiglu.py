"""Prepare-only standalone SwiGlu call for the exact R31 Qwen3.6 source.

[Role] Admit only the R31 deployment's dense F0 ``activation`` operator, whose
  ordinary BF16 gate/up tensor is consumed by ``torch_npu.npu_swiglu`` and the
  CANN ``SwiGlu``/``aclnnSwiGlu`` kernel, and expand that occurrence into the
  real single-input/``dim=-1`` operand descriptor for the evaluated token rows.
[Invariants] The source is selected by the exact R31 OCI digest already present
  in ``FeatureConfig.metadata.image_id``.  The static call is one BF16 ND tensor
  of width ``2*intermediate`` and one BF16 ND output of width ``intermediate``;
  the row count lives only in the occurrence.  A fused/quantized activation,
  another image, another dtype, an unknown source or an inconsistent ordinary
  signature returns ``None`` so ordinary prediction stays untouched.  The
  forward Python body and the exact R31 CANN kernel package are not retained:
  the call stays explicitly partial and never claims a complete physical
  execution.  No tensor is allocated and no device/runtime module is imported.
[Timeouts & Locks] Bounded memo of 32 geometry descriptors; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the selected operator and the
  existing request geometry; no numeric prediction, no launch, no measurement.
"""
from __future__ import annotations

from typing import Any

R31_IMAGE_SHA = "9e869e8148467296f56d1001b0dac4fea893c841ac7d0a03d4e824a233c2ffea"
R31_REVISION = "r31/9e869e81"
R31_CALLER_SHA256 = "69c0b69ef9b80c08b772d191e8e66e8857c5c1ea68e719ddc91b1a197f7ee8c3"
R31_LIBOPAPI_SHA256 = "eb1c788e39b7a02f0694450e917a3771c7e65d7d8fb9ea2e03f78c4a5313c456"
CAPTURE_CSV_SHA256 = "2476b865c172821ea816ffff2ca1fb9b11ef2bee07a75887dc443ecc4c55bf2a"
CAPTURE_GZ_SHA256 = "a523936e982e4fd45b80071e906276d17d18fac03c5bfaca903e2f1158f778d1"
CAPTURE_SEQUENCE = "ecddce9e"
CAPTURE_API_CSV_SHA256 = "e19308dea200ead700b417475da5983576ec8884646e3b03e833be6b06e0808e"
CAPTURE_OP_CSV_SHA256 = "5e79a7a4a153006fa1e85ce44b429bf30d0ba5cde7737feb09c04e55beef5f31"
CAPTURE_PATH = (
    "/Volumes/data/mindie-agent-workstreams/a5-multinode-fleet-rack06-20261107/"
    "out/smoke_qwen36_single/pf_4k_c1/177.25.28.1/"
    "rank0_545_20261107032704817_ascend_pt/kernel_details.csv.gz")

ENTRY = "torch_npu.npu_swiglu"
ACLNN_SYMBOL = "aclnnSwiGlu"
SCHEMA = "mindie-sim/swiglu-native-prepare/1"
IMPLEMENTATION = "standalone_swiglu"

#: The exact R31 source that registers ``SiluAndMul -> AscendSiluAndMul``.
#: The method body itself is not retained; the emitted ACLNN entry is observed
#: in the same-rank capture and the symbol is present in the exact R31 binary.
_CALLER_REGISTRATION = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/utils.py",
    "sha256": R31_CALLER_SHA256,
    "symbol": 'REGISTERED_ASCEND_OPS["SiluAndMul"] -> AscendSiluAndMul',
    "revision": R31_REVISION,
    "state": "task_artifact",
    "source_lines": [830, 834, 869],
}
_CALLER_BODY = {
    "repository": "vllm-ascend",
    "path": "vllm_ascend/ops/activation.py",
    "sha256": None,
    "symbol": "AscendSiluAndMul.forward_oot",
    "revision": R31_REVISION,
    "state": "unretained",
    "evidence": "source_body_not_retained_in_task",
}
_R31_API_BINARY = {
    "repository": "cann",
    "path": "usr/local/Ascend/cann-9.2.0/aarch64-linux/lib64/libopapi.so",
    "sha256": R31_LIBOPAPI_SHA256,
    "symbol": ACLNN_SYMBOL,
    "revision": "cann-9.2.0-r31",
    "state": "task_artifact",
    "evidence": "runtime_binary_symbol_table",
}
_KERNEL_SOURCE = {
    "repository": "cann/ops-nn",
    "path": "activation/swi_glu/op_kernel/swi_glu.cpp",
    "sha256": "4e152645e295a51fe8946881f490a5520c45beb3817a325e94e8f5c6b13254ee",
    "symbol": "swi_glu",
    "revision": "cann-9.1.0-near-source@955f8b33",
    "state": "task_artifact",
    "source_scope": "near_source_not_r31_package",
}
_TILING_SOURCE = {
    "repository": "cann/ops-nn",
    "path": "activation/swi_glu/op_host/swi_glu_tiling.cpp",
    "sha256": "7d7076db5f95ca600547224cdd0af260f5bec2ec9da02eaf523cdb4c03f4d80d",
    "symbol": "SwiGluTiling",
    "revision": "cann-9.1.0-near-source@955f8b33",
    "state": "task_artifact",
    "source_scope": "near_source_not_r31_package",
}
_OP_DEF_SOURCE = {
    "repository": "cann/ops-nn",
    "path": "activation/swi_glu/op_host/swi_glu_def.cpp",
    "sha256": "1ca8f924fca9341765e46fe1252aeb02b95d1904e5067485d25d6de4fa364394",
    "symbol": "ops.SwiGlu",
    "revision": "cann-9.1.0-near-source@955f8b33",
    "state": "task_artifact",
    "source_scope": "near_source_not_r31_package",
}
_COMPUTE_SOURCE = {
    "repository": "cann/ops-nn",
    "path": "activation/swi_glu/op_kernel/swi_glu_bf16_std.hpp",
    "sha256": "569bfd8db151edc0643a4eb4c7c9007c93f30fd288b3d12ca0c166d0c189a53e",
    "symbol": "SwigluVectorBF16::Compute",
    "revision": "cann-9.1.0-near-source@955f8b33",
    "state": "task_artifact",
    "source_scope": "near_source_not_r31_package",
}
_SOURCE_LINES = {
    "caller_registration": [830, 834, 869],
    "op_def_input": [23, 27],
    "op_def_output": [28, 31],
    "op_def_dim_attr": [32, 34],
    "kernel_entry": [20, 21],
    "kernel_compute": [55, 100],
    "tiling_db": [396, 438],
    "tiling_key": [528, 529],
}
_UNRESOLVED = (
    "activation_forward_body_source_sha256_not_retained",
    "kernel_source_is_cann_9_1_near_source_not_r31_package",
    "host_tiling_key_not_retained",
    "workspace_bytes_from_runtime_getworkspace",
)
_DYNAMIC_CACHE: dict[tuple, dict] = {}


def _digest(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("sha256:"):
        text = text.split(":", 1)[1]
    return text


def _source_record(record: dict) -> dict:
    return {
        "repository": record.get("repository"),
        "path": record.get("path"),
        "sha256": record.get("sha256"),
        "symbol": record.get("symbol"),
        "revision": record.get("revision"),
        "state": record.get("state") or "unknown",
    }


def _descriptor(shape: list[int], dtype: str, layout: str, role: str) -> dict:
    return {"shape": [int(item) for item in shape], "dtype": dtype,
            "layout": layout, "role": role, "pointer": None,
            "allocation": "external"}


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _theta(compiled: Any) -> Any:
    operator = getattr(compiled, "operator", None)
    return getattr(operator, "theta", None)


def _is_standalone_f0_activation(compiled: Any, phase: str) -> bool:
    if str(getattr(compiled, "kind", "") or "") != "F0":
        return False
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return False
    if str(getattr(operator, "family", "") or "") != "elementwise":
        return False
    if str(getattr(operator, "name", "") or "") != "activation":
        return False
    template = getattr(operator, "template", None)
    if str(getattr(template, "role", "") or "") != "activation":
        return False
    if phase not in {"", "prefill", "decode"}:
        return False
    theta = _theta(compiled)
    if theta is None:
        return False
    if _int(theta.find("gated", 0)) != 1:
        return False
    activation = str(theta.find("activation", "") or "").strip().lower()
    if activation not in {"silu", "swiglu"}:
        return False
    elements = _int(theta.find("elements", 0))
    if elements < 1:
        return False
    # The ordinary F0 signature already carries the real BF16 byte counts.
    # ``quant.act_bytes`` is a quant-ledger proxy and is deliberately not the
    # native boundary: the captured SwiGlu is DT_BF16 in and out.
    read_bytes, write_bytes = (
        _float(theta.find("read_bytes", 0)), _float(theta.find("write_bytes", 0)))
    return (abs(read_bytes - 4.0 * elements) <= 1e-6
            and abs(write_bytes - 2.0 * elements) <= 1e-6)


def _feature_meta(compiled: Any) -> dict:
    return dict(getattr(compiled, "feature_metadata", ()) or ())


def _deployment_image(compiled: Any) -> str:
    meta = _feature_meta(compiled)
    return _digest(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")


def _standalone_lowering(compiled: Any) -> bool:
    """An explicit lowering table must exist and must not select the fusion."""
    meta = _feature_meta(compiled)
    lowerings = meta.get("source_lowerings")
    if not isinstance(lowerings, (list, tuple, set, frozenset)):
        return False
    return "dequant_swiglu_quant" not in {str(value) for value in lowerings}


def _gate_up_output_format(compiled: Any) -> str:
    """The consumed gate/up projection's real output format.

    SwiGlu consumes the preceding F0 ``gate_up`` output; its declared matrix
    output format is the exact dtype boundary, not the quant ledger's
    ``act_bytes`` proxy.
    """
    for operator in getattr(compiled, "siblings", ()) or ():
        if str(getattr(operator, "name", "") or "") != "gate_up":
            continue
        theta = getattr(operator, "theta", None)
        quant = dict(getattr(theta, "quant", ()) or ())
        return str(quant.get("output_format") or "").strip().lower()
    return ""


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the one R31 standalone F0 SwiGlu call identity, or ``None``.

    The returned object is the existing ``native_calls._CallSpec``.  The import
    is local so the common caller can import this module without a cycle.
    """
    if not _is_standalone_f0_activation(compiled, phase):
        return None
    if not _standalone_lowering(compiled):
        return None
    if _gate_up_output_format(compiled) not in {"bf16", "bfloat16"}:
        return None
    digest = _deployment_image(compiled)
    if not digest.startswith(R31_IMAGE_SHA):
        return None
    theta = _theta(compiled)
    elements = _int(theta.find("elements", 0))
    if elements < 1:
        return None
    activation = str(theta.find("activation", "silu") or "silu").strip().lower()
    ops = _int(theta.find("ops_per_element", 4), 4)
    read_bytes = 4.0 * elements
    write_bytes = 2.0 * elements
    operator = compiled.operator
    name = str(getattr(operator, "name", "activation") or "activation")
    family = str(getattr(operator, "family", "elementwise") or "elementwise")
    kind = str(getattr(compiled, "kind", "F0") or "F0")
    typed = {
        "dtype": "bf16",
        "layout": "ND",
        "shapes": {"input": [2 * elements], "output": [elements]},
        "attrs": {
            "elements": elements,
            "gated": 1,
            "activation": activation,
            "gate_up_output_format": "bf16",
            "ops_per_element": ops,
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "input_width": 2 * elements,
            "output_width": elements,
            "dim": -1,
            "gate_first": True,
            "gate_order": ["gate", "up"],
            "compute": "silu(gate) * up",
            "implementation": IMPLEMENTATION,
            "native_kind": "SwiGlu",
            "aclnn_symbol": ACLNN_SYMBOL,
            "deployment_image_id": digest,
            "deployment_source_revision": "r31",
            "source_version": R31_REVISION,
            "source_lines": _SOURCE_LINES,
            "caller_source_path": _CALLER_BODY["path"],
            "caller_source_symbol": _CALLER_BODY["symbol"],
            "caller_registration_path": _CALLER_REGISTRATION["path"],
            "caller_registration_sha256": R31_CALLER_SHA256,
            "kernel_source_path": _KERNEL_SOURCE["path"],
            "kernel_source_sha256": _KERNEL_SOURCE["sha256"],
            "kernel_source_revision": _KERNEL_SOURCE["revision"],
            "capture_csv_sha256": CAPTURE_CSV_SHA256,
            "capture_sequence": CAPTURE_SEQUENCE,
        },
        "rank_width": 2,
    }
    kernels = ({
        "name": "SwiGlu",
        "role": "vector",
        "source_path": _KERNEL_SOURCE["path"],
        "source_sha256": _KERNEL_SOURCE["sha256"],
        "evidence": "source_declared",
        "source_scope": _KERNEL_SOURCE["source_scope"],
        "source_revision": _KERNEL_SOURCE["revision"],
        "actual_runtime_binary_sha256": R31_LIBOPAPI_SHA256,
        "capture_type": "SwiGlu",
        "capture_core": "AI_VECTOR_CORE",
        "compute": "silu(gate) * up",
    },)
    from .native_calls import _CallSpec  # local: avoid an import cycle
    return _CallSpec(
        api={"namespace": "torch", "entry": ENTRY,
             "aclnn_symbol": ACLNN_SYMBOL, "version": R31_REVISION},
        source=_source_record(_CALLER_REGISTRATION),
        typed=typed,
        kernels=kernels,
        members=(),
        owner=f"{kind}:{family}:{name}",
        execution_core="aiv",
        unresolved=_UNRESOLVED,
    )


def _formal_operands(elements: int, tokens: int) -> dict:
    return {
        "input": {"name": "x", "role": "input", "required": True,
                  "dtype_candidates": ["fp16", "fp32", "bf16"],
                  "format": "ND", "selected_dtype": "bf16",
                  "selected_shape": [tokens, 2 * elements],
                  "splits": ["gate", "up"]},
        "attr": {"name": "dim", "type": "int", "optional": True, "value": -1},
        "output": {"name": "y", "role": "output", "required": True,
                   "dtype_candidates": ["fp16", "fp32", "bf16"],
                   "format": "ND", "selected_dtype": "bf16",
                   "selected_shape": [tokens, elements]},
        "shape_relation": ("y.shape = x.shape with the selected dim reduced "
                           "by a factor of two"),
        "selected_width": {"input": 2 * elements, "output": elements},
    }


def _capture_observation(phase: str) -> dict:
    """The exact retained PF4kC1 occurrence, never a phase-unobserved value."""
    if phase not in {"", "prefill"}:
        return {
            "status": "not_retained_for_this_phase",
            "source_call_still_exact": "same R31 SwiGlu call identity",
            "not_used_as_prediction_input": True,
        }
    return {
        "path": CAPTURE_PATH,
        "kernel_details_csv_sha256": CAPTURE_CSV_SHA256,
        "kernel_details_csv_gz_sha256": CAPTURE_GZ_SHA256,
        "sequence_sha256_prefix": CAPTURE_SEQUENCE,
        "api_statistic_sha256": CAPTURE_API_CSV_SHA256,
        "op_statistic_sha256": CAPTURE_OP_CSV_SHA256,
        "api_statistic": {
            "aclnnSwiGlu": {"count": 64, "level": "host", "api": "acl"},
            "aclnnSwiGluGetWorkspaceSize": {"count": 64, "level": "host",
                                            "api": "acl"},
            "aclnnSwiGluTiling": {"count": 1, "level": "host", "api": "acl"},
        },
        "op_statistic": {
            "SwiGlu": {"count": 64, "core_type": "AI_VECTOR_CORE"},
        },
        "csv_record": 82,
        "complete_ordered_groups": 64,
        "input_shape": [4096, 17408],
        "input_dtype": "bf16",
        "input_format": "ND",
        "output_shape": [4096, 8704],
        "output_dtype": "bf16",
        "output_format": "ND",
        "block_num": 64,
        "accelerator_core": "AI_VECTOR_CORE",
        "not_used_as_prediction_input": True,
    }


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Expand one selected R31 SwiGlu occurrence into the prepare-only call."""
    api = getattr(binding, "api", {}) or {}
    if api.get("entry") != ENTRY:
        return None
    typed = getattr(binding, "typed", {}) or {}
    attrs = dict(typed.get("attrs") or {})
    if attrs.get("implementation") != IMPLEMENTATION:
        return None
    if attrs.get("native_kind") != "SwiGlu":
        return None
    if not str(attrs.get("deployment_image_id") or "").startswith(R31_IMAGE_SHA):
        return None
    if str(attrs.get("source_version") or "") != R31_REVISION:
        return None
    elements = _int(attrs.get("elements", 0))
    if elements < 1:
        return None
    shapes = dict(typed.get("shapes") or {})
    if list(shapes.get("output") or ()) != [elements]:
        return None
    m = _int(getattr(geometry, "effective_tokens", 0), 0)
    if m < 1:
        return None
    cache_key = (binding.call_id, m)
    cached = _DYNAMIC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    actual_shapes = {"input": [m, 2 * elements], "output": [m, elements]}
    actual_attrs = {
        **attrs,
        "tokens": m,
        "rows": m,
        "phase": str(getattr(geometry, "phase_name", "") or ""),
    }
    actual = {"dtype": typed.get("dtype") or "bf16",
              "layout": typed.get("layout") or "ND",
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "input": _descriptor([m, 2 * elements], "bf16", "ND", "input"),
        "output": _descriptor([m, elements], "bf16", "ND", "output"),
    }
    source = dict(getattr(binding, "source", {}) or {})
    prepare = {
        "schema": SCHEMA,
        "call": ENTRY,
        "aclnn_symbol": ACLNN_SYMBOL,
        "call_id": binding.call_id,
        "signature_hash": typed.get("signature_hash"),
        "implementation": IMPLEMENTATION,
        "source": source,
        "caller": _CALLER_BODY,
        "caller_registration": _CALLER_REGISTRATION,
        "kernel": _KERNEL_SOURCE,
        "tiling": _TILING_SOURCE,
        "op_def": _OP_DEF_SOURCE,
        "compute_source": _COMPUTE_SOURCE,
        "api_binary": _R31_API_BINARY,
        "source_version": R31_REVISION,
        "source_lines": _SOURCE_LINES,
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": {"x": "input", "dim": -1, "y": "output"},
        "argument_order": ["x", "dim", "y"],
        "positional_arguments": {"x": "input"},
        "keywords": {"dim": -1},
        "optional_slots": {
            "dim": "explicit_selected_-1",
            "workspace": "runtime_getworkspace",
        },
        "prototype": "torch_npu.npu_swiglu(input, dim=-1) -> Tensor",
        "aclnn_arguments": ["x", "dim", "y"],
        "aclnn_runtime_slots": ["workspace", "workspaceSize", "executor", "stream"],
        "formal_operands": _formal_operands(elements, m),
        "shapes": dict(actual_shapes),
        "signature_shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "split": {
            "axis": -1,
            "gate": [m, elements],
            "up": [m, elements],
            "order": ["gate", "up"],
            "compute": "silu(gate) * up",
            "internal": True,
            "allocation": "view_of_input",
        },
        "workspace": {
            "owner": "aclnnSwiGluGetWorkspaceSize",
            "allocation": "caller_owned",
            "bytes": None,
            "status": "unresolved_runtime_query",
        },
        "deployment": {
            "image_id": str(attrs.get("deployment_image_id") or ""),
            "revision": "r31",
            "selection": "FeatureConfig.metadata.image_id",
            "selected_by": ("SiluAndMul caller registration + captured "
                            "aclnnSwiGlu/SwiGlu call"),
            "source_path": _CALLER_REGISTRATION["path"],
            "source_version": R31_REVISION,
            "source_sha256": R31_CALLER_SHA256,
            "caller_body_path": _CALLER_BODY["path"],
            "kernel_sha256": _KERNEL_SOURCE["sha256"],
            "api_binary_sha256": R31_LIBOPAPI_SHA256,
        },
        "binary_selection": {
            "status": "observed_and_symbol_declared",
            "owner_kernel": "SwiGlu",
            "aclnn_symbol": ACLNN_SYMBOL,
            "runtime_binary_sha256": R31_LIBOPAPI_SHA256,
            "evidence": ("R31 libopapi.so symbol table plus same-rank capture "
                         "api_statistic aclnnSwiGlu and op_statistic SwiGlu"),
        },
        "capture": _capture_observation(
            str(getattr(geometry, "phase_name", "") or "")),
        "match": {
            "native_type": "SwiGlu",
            "entry": ENTRY,
            "aclnn_symbol": ACLNN_SYMBOL,
            "call_id": binding.call_id,
            "deployment_image_id": str(attrs.get("deployment_image_id") or ""),
            "source_revision": "r31",
            "source_version": R31_REVISION,
            "implementation": IMPLEMENTATION,
            "kernel": "SwiGlu",
            "input_width": 2 * elements,
            "output_width": elements,
            "activation_dtype": "bf16",
            "output_dtype": "bf16",
            "layout": "ND",
            "source_sha256": source.get("sha256") or R31_CALLER_SHA256,
            "kernel_sha256": _KERNEL_SOURCE["sha256"],
        },
        "complete": False,
        "partial": True,
        "partial_reasons": [
            "caller_forward_body_not_retained",
            "kernel_package_is_cann_9_1_near_source_not_r31",
            "host_tiling_key_not_retained",
            "workspace_bytes_not_retained",
        ],
        "unresolved": list(getattr(binding, "unresolved", ()) or _UNRESOLVED),
        "non_claims": [
            "no_runtime_allocation",
            "no_device_launch",
            "no_measured_duration",
            "no_physical_completeness",
            "no_numeric_prediction_change",
        ],
    }
    if str(getattr(geometry, "phase_name", "") or "") == "decode":
        prepare["partial_reasons"].append(
            "decode_occurrence_capture_not_retained")
    result = {
        "actual": validated,
        "prepare": prepare,
        "prepare_args": {
            "dtype": validated.get("dtype"),
            "layout": validated.get("layout"),
            "shapes": validated.get("shapes"),
            "attrs": validated.get("attrs"),
        },
    }
    if len(_DYNAMIC_CACHE) >= 32:
        _DYNAMIC_CACHE.clear()
    _DYNAMIC_CACHE[cache_key] = result
    return result
