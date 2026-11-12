"""Prepare-only standalone GroupedMatmulSwigluQuantV2 invocation.

[Role] Expand the verified R64 W4A4 routed-expert gate/up composite at
  ``AscendW4A4MXFP4DynamicFusedMoEMethod.apply_gmm1_act_quant`` into the real
  ``torch_npu.npu_grouped_matmul_swiglu_quant_v2`` operand, grouped row, output
  and E8M0 scale descriptors for one evaluated occurrence.
[Invariants] Only the dual-anchored R64 W4A4 MXFP4 source is admitted.  Static
  K/N/local-expert identity stays in the binding; dispatch rows, valid
  assignment rows and group-list values live only in the occurrence.  Packed
  FP4 storage (uint8) and its logical FP4 view are kept distinct.  A total row
  capacity is never invented when the dispatch-row ledger is absent, and no
  default ``swiglu_limit``/``bias`` argument is added because the actual
  caller passes neither.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the pinned caller, declared
  routing and source call surface.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from mindie_sim.prediction.moe_callers import project_expert_receipt

ENTRY = "torch_npu.npu_grouped_matmul_swiglu_quant_v2"
ACLNN_SYMBOL = ("aclnnGroupedMatmulSwigluQuantV2_"
                "GroupedMatmulSwigluQuantV2_GroupedMatmulSwigluQuantV2")
SCHEMA = "mindie-sim/grouped-matmul-swiglu-quant-v2-native-prepare/1"
R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
SOURCE_PATH = "vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py"
SOURCE_SHA = "cbf45467019febc009d26d23492a7a4b5a7784a9f047725c58aa8558998bff0e"
SOURCE_SYMBOL = "AscendW4A4MXFP4DynamicFusedMoEMethod.apply_gmm1_act_quant"
SOURCE_LINES = [383, 408]
SOURCE_CALL_LINES = [394, 408]
SOURCE_ANCHORS = (
    "5929df13d8dc5808186c19446f12078a4a706f40",
    "f7d558657^=ce61785f0b2f551e6068e9bb9259ce73395e0c77",
)
DISPATCHER_PATH = "vllm_ascend/ops/fused_moe/token_dispatcher.py"
DISPATCHER_SHA = "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5"
IMPLEMENTATION = "W4A4_MXFP4"
_FP4 = frozenset({"mxfp4", "mx_fp4", "fp4", "fp4_e2m1", "float4_e2m1",
                  "float4_e2m1fn_x2"})
_E8M0 = frozenset({"fp8_e8m0", "e8m0", "ue8m0", "float8_e8m0fnu"})


def _find(theta: Any, key: str, default: Any = None) -> Any:
    finder = getattr(theta, "find", None)
    if callable(finder):
        try:
            value = finder(key, default)
        except TypeError:
            value = finder(key)
        return default if value is None else value
    for source in (getattr(theta, "extras", ()), getattr(theta, "quant", ()),
                   getattr(theta, "matrix", ())):
        if isinstance(source, Mapping) and key in source:
            return source[key]
    return default


def _ceil_div(value: int, divisor: int) -> int:
    return (int(value) + int(divisor) - 1) // int(divisor)


def _source_meta(compiled: Any) -> tuple[dict[str, Any], str] | None:
    """Only the witnessed R64 OCI image admits this source adapter."""
    from mindie_sim.prediction.mx_source_provenance import (
        R64_IMAGE_SHA as DECLARED_R64, image_digest, source_revision_from_oci)
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    image_id = str(meta.get("image_id") or meta.get("oci_image_digest")
                   or meta.get("deployment_image_id") or "")
    image = str(meta.get("image") or meta.get("deployment_image")
                or meta.get("oci_image") or "")
    digest = image_digest(image_id or image)
    revision = str(meta.get("mx_quant_source_revision") or "")
    if not revision:
        revision = source_revision_from_oci(image, image_id)
    if revision != "r64" or not digest.startswith(DECLARED_R64):
        return None
    return meta, digest


def _static_terms(compiled: Any) -> dict[str, Any] | None:
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    theta = operator.theta
    quant = dict(getattr(theta, "quant", {}) or ())
    extras = dict(getattr(theta, "extras", {}) or ())
    matrix = dict(getattr(theta, "matrix", {}) or ())
    weight_format = str(quant.get("weight_format") or "").lower()
    act_format = str(quant.get("act_format") or "").lower()
    output_format = str(quant.get("output_format")
                        or extras.get("fused_output_format") or "").lower()
    if (weight_format not in _FP4 or act_format not in _FP4
            or output_format not in _FP4
            or int(quant.get("scale_group", 0) or 0) != 32):
        return None
    if str(quant.get("linear_method") or "").lower() not in {
            "mx_fp4", "mxfp4", "w4a4_mxfp4"}:
        return None
    scale_formats = {
        str(quant.get(key) or "").lower()
        for key in ("weight_scale_format", "weight_scale_storage_format",
                    "weight_scale_interpret_dtype")
    }
    if not (scale_formats & _E8M0):
        return None
    if extras.get("activation_source_operation") != "GroupedMatmulSwigluQuantV2":
        return None
    if (int(extras.get("fused_activation_quant", 0) or 0) != 1
            or int(extras.get("activation_typed_outputs", 0) or 0) != 2):
        return None
    try:
        k = int(matrix.get("K", extras.get("latent", 0)) or 0)
        n = int(matrix.get("N", 0) or 0)
        width = int(extras.get("expert_width", n // 2) or 0)
        local = int(extras.get("E_local", extras.get("local", 0)) or 0)
        experts = int(extras.get("E", 0) or 0)
        top_k = int(extras.get("k", 0) or 0)
    except (TypeError, ValueError):
        return None
    if (k < 1 or n < 1 or width < 1 or n != 2 * width
            or local < 1 or top_k < 1 or k % 2 or width % 2):
        return None
    return {"theta": theta, "quant": quant, "extras": extras,
            "matrix": matrix, "K": k, "N": n, "E": width,
            "local": local, "experts": experts, "top_k": top_k}


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """The verified R64 W4A4 grouped gate/up call identity, or None.

    The selection is the dual-anchored local source blob
    (``5929df13d`` and ``f7d558657^`` both sha256 ``cbf45467...``), not the
    raising ``A5DeviceAdaptor.npu_grouped_matmul_swiglu_quant`` wrapper and not
    the W8A8 MXFP8 template; neither is a W4A4 caller.
    """
    from . import native_calls

    if getattr(compiled, "kind", "") != "F3":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (getattr(operator, "name", "") != "expert_gate_up"
            or getattr(operator, "family", "") != "grouped_gemm"):
        return None
    terms = _static_terms(compiled)
    if terms is None:
        return None
    selected = _source_meta(compiled)
    if selected is None:
        return None
    _meta, digest = selected
    k, n, width = terms["K"], terms["N"], terms["E"]
    local, top_k = terms["local"], terms["top_k"]
    pairs = _ceil_div(_ceil_div(k, 32), 2)
    out_pairs = _ceil_div(_ceil_div(width, 32), 2)
    typed = {
        "dtype": "mxfp4", "layout": "ND",
        "shapes": {
            "x": [k],
            "x_scale": [pairs, 2],
            "group_list": [local],
            "weight": [n, k],
            "weight_scale": [n, pairs, 2],
            "output": [width],
            "output_scale": [out_pairs, 2],
        },
        "attrs": {
            "matrix_k": k, "matrix_n": n, "latent": k,
            "expert_width": width, "experts": terms["experts"],
            "experts_per_device": local, "top_k": top_k,
            "scale_group": 32,
            "weight_format": "mxfp4", "act_format": "mxfp4",
            "output_format": "mxfp4",
            "prequant_format": str(terms["quant"].get("prequant_format")
                                   or "bf16"),
            "linear_method": "mx_fp4", "implementation": IMPLEMENTATION,
            "activation_source_operation": "GroupedMatmulSwigluQuantV2",
            "activation_typed_outputs": 2,
            "weight_scale_format": "fp8_e8m0",
            "x_scale_format": "fp8_e8m0",
            "output_scale_format": "fp8_e8m0",
            "weight_storage_dtype": "uint8",
            "weight_storage_layout": "packed_fp4_nd_transposed",
            "weight_packed": True,
            "weight_scale_storage_dtype": "uint8",
            "weight_scale_layout": "packed_e8m0_nd_transposed",
            "x_storage_dtype": "uint8",
            "output_storage_dtype": "uint8",
            "weight_list_length": 1,
            "weight_scale_list_length": 1,
            "output_contract": ["out", "out_scale"],
            "group_list_dtype": "int64",
            "group_list_layout": "cumulative",
            "group_list_source": (
                "cumsum_group_list(mlp_compute_input.group_list, "
                "mlp_compute_input.group_list_type, 0)"),
            "dequant_mode": 2, "quant_mode": 2,
            "dequant_dtype": "fp32", "quant_dtype": "fp4_e2m1",
            "x_dtype": "fp4_e2m1", "weight_dtype": "fp4_e2m1",
            "weight_scale_dtype": "fp8_e8m0",
            "x_scale_dtype": "fp8_e8m0",
            "bias_argument": "not_passed",
            "swiglu_limit_argument": "not_passed",
            "measured_sixth_input": (
                "FLOAT in captured ABI; not named by retained Python caller"),
            "source_revision": "r64", "source_lines": list(SOURCE_LINES),
            "source_call_lines": list(SOURCE_CALL_LINES),
            "source_anchors": list(SOURCE_ANCHORS),
            "dispatcher_source_sha256": DISPATCHER_SHA,
            "deployment_image_id": digest,
        },
        "occurrence_attrs": {
            "capacity_basis": "str",
            "phase": "str",
        },
        "rank_width": 3,
    }
    source = native_calls._source(
        "vllm-ascend", SOURCE_PATH, sha256=SOURCE_SHA,
        symbol=SOURCE_SYMBOL, revision="r64/8d422528", state="source")
    source["verified_anchors"] = list(SOURCE_ANCHORS)
    kernel = native_calls._kernel(
        "GroupedMatmulSwigluQuantV2", "mix", path=SOURCE_PATH,
        sha256=SOURCE_SHA, evidence="source_caller",
        native_call=ACLNN_SYMBOL, source_lines=list(SOURCE_CALL_LINES),
        source_anchors=list(SOURCE_ANCHORS),
        source_status="source_caller_only",
        kernel_source_status="not_retained")
    unresolved = (
        "aclnn_c_abi_not_retained",
        "grouped_matmul_swiglu_quant_v2_kernel_source_not_retained",
        "binary_identity_not_established",
        "compiled_kernel_identity_unpinned",
        "measured_sixth_float_input_not_named_by_python_caller",
    )
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_SYMBOL,
                          version="r64/8d422528"),
        source, typed, (kernel,), (), getattr(operator, "name", ""),
        execution_core="mix", unresolved=unresolved)


def _for_subgraph(geometry: Any, kind: str) -> Any:
    method = getattr(geometry, "for_subgraph", None)
    if callable(method):
        try:
            return method(kind)
        except Exception:
            return geometry
    return geometry


def _descriptor(shape: list, dtype: str, layout: str, role: str,
                **extra: Any) -> dict:
    row = {"shape": [None if item is None else int(item) for item in shape],
           "dtype": dtype, "layout": layout, "role": role, "pointer": None,
           "allocation": "external"}
    row.update(extra)
    return row


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only occurrence for the verified W4A4 grouped gate/up call."""
    api = getattr(binding, "api", None) or {}
    if api.get("entry") != ENTRY:
        return None
    typed = getattr(binding, "typed", None) or {}
    attrs = dict(typed.get("attrs") or {})
    if attrs.get("implementation") != IMPLEMENTATION:
        return None
    if attrs.get("source_revision") != "r64":
        return None
    if getattr(compiled, "kind", "") != "F3":
        return None
    operator = getattr(compiled, "operator", None)
    if getattr(operator, "name", "") != "expert_gate_up":
        return None
    terms = _static_terms(compiled)
    selected = _source_meta(compiled)
    if terms is None or selected is None:
        return None
    _meta, digest = selected
    k, n, width = terms["K"], terms["N"], terms["E"]
    local, top_k = terms["local"], terms["top_k"]
    geom = _for_subgraph(geometry, "F3")
    state = getattr(compiled, "state", None)
    if state is None:
        state = getattr(geom, "state", None)
    receipt = project_expert_receipt(
        compiled, geom, state=state, activation_dtype="fp4_e2m1",
        scale_dtype="fp8_e8m0", expected_scope="grouped_gate_up")
    if receipt is None:
        return None
    rows = receipt["received_rows"]
    upper_bound = receipt["capacity_rows"]
    valid = receipt["valid_rows"]
    cumulative: list[int] | None = receipt["gmm1_group_list"]
    group_basis = receipt["basis"]
    dispatch_basis = group_basis
    phase = str(getattr(geom, "phase_name", "") or "")
    pairs = _ceil_div(_ceil_div(k, 32), 2)
    out_pairs = _ceil_div(_ceil_div(width, 32), 2)
    actual_shapes = {
        "x": [rows, k],
        "x_scale": [rows, pairs, 2],
        "group_list": [local],
        "weight": [local, n, k],
        "weight_scale": [local, n, pairs, 2],
        "output": [rows, width],
        "output_scale": [rows, out_pairs, 2],
    }
    actual_attrs = {
        **attrs, "valid_rows": valid, "capacity_rows": upper_bound,
        "dispatch_rows": rows, "upper_bound": upper_bound,
        "capacity_basis": group_basis,
        "phase": phase, "m": valid,
        "group_list_values": cumulative, "group_list_basis": group_basis,
        "local_expert_ids": receipt["local_expert_ids"],
        "rank_index": receipt["rank"],
        "group_list_provenance": receipt,
    }
    actual = {"dtype": typed.get("dtype"), "layout": typed.get("layout"),
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([rows, k], "fp4_e2m1", "ND", "input",
                         storage_dtype="uint8",
                         storage_shape=[None if rows is None else rows, k // 2],
                         packed=True),
        "x_scale": _descriptor([rows, pairs, 2], "fp8_e8m0", "ND",
                               "input", storage_dtype="uint8",
                               interpret_dtype="fp8_e8m0"),
        "group_list": _descriptor([local], "int64", "ND", "input",
                                  values=cumulative,
                                  layout_semantics="cumulative"),
        "weight": _descriptor([local, n, k], "fp4_e2m1", "ND", "input",
                              storage_dtype="uint8",
                              storage_shape=[local, k // 2, n],
                              logical_shape=[n, k], packed=True,
                              storage_layout="ND_transposed"),
        "weight_scale": _descriptor([local, n, pairs, 2], "fp8_e8m0", "ND",
                                    "input", storage_dtype="uint8",
                                    interpret_dtype="fp8_e8m0",
                                    storage_shape=[local, pairs, n, 2],
                                    logical_shape=[n, pairs, 2],
                                    storage_layout="ND_transposed"),
        "output": _descriptor([rows, width], "fp4_e2m1", "ND", "output",
                              storage_dtype="uint8",
                              storage_shape=[None if rows is None else rows,
                                             width // 2],
                              packed=True),
        "output_scale": _descriptor([rows, out_pairs, 2], "fp8_e8m0",
                                    "ND", "output", storage_dtype="uint8",
                                    interpret_dtype="fp8_e8m0"),
    }
    arguments = {
        "x": "x", "weight": ["weight"], "group_list": "group_list",
        "weight_scale": ["weight_scale"], "x_scale": "x_scale",
        "dequant_mode": 2, "quant_mode": 2,
        "dequant_dtype": "fp32", "quant_dtype": "fp4_e2m1",
        "x_dtype": "fp4_e2m1", "weight_dtype": "fp4_e2m1",
        "weight_scale_dtype": "fp8_e8m0", "x_scale_dtype": "fp8_e8m0",
    }
    argument_order = [
        "x", "weight", "group_list", "weight_scale", "x_scale",
        "dequant_mode", "quant_mode", "dequant_dtype", "quant_dtype",
        "x_dtype", "weight_dtype", "weight_scale_dtype", "x_scale_dtype",
    ]
    shape_fields = receipt
    shape_complete = receipt["gmm1_shape_complete"]
    # Never claim prepare.complete while the captured sixth scalar and the
    # compiled kernel/ACLNN ABI are unresolved.  Shape completeness is a
    # separate field.
    complete = False
    partial_fields: list[str] = []
    partial_reasons: list[str] = []
    if rows is None:
        partial_reasons.append("received_rows_not_declared")
        partial_fields.append("received_rows")
    if receipt["gmm1_group_list"] is None:
        partial_reasons.append(group_basis)
        partial_fields.append("group_list")
    if not receipt["ownership_known"]:
        partial_reasons.append("local_expert_ids_not_declared")
        partial_fields.append("local_expert_ids")
    if upper_bound is None:
        partial_reasons.append("upper_bound_not_declared")
        partial_fields.append("upper_bound")
    if receipt["input_group_list_type"] is None:
        partial_fields.append("input_group_list_type")
    if receipt["gmm2_group_list"] is None:
        partial_fields.append("gmm2_group_list")
    partial_reasons.extend(receipt["unresolved"])
    partial_reasons.extend((
        "measured_sixth_float_input_not_named_by_python_caller",
        "compiled_kernel_identity_unpinned",
    ))
    partial_fields.extend(("sixth_float_input", "compiled_kernel_identity"))
    partial_fields = list(dict.fromkeys(partial_fields))
    unresolved = list(dict.fromkeys(getattr(binding, "unresolved", ()) or ()))
    unresolved.extend(reason for reason in partial_reasons
                      if reason not in unresolved)
    prepare = {
        "schema": SCHEMA,
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": typed.get("signature_hash"),
        "implementation": IMPLEMENTATION,
        "source": {**dict(getattr(binding, "source", None) or {}),
                   "verified_anchors": list(SOURCE_ANCHORS)},
        "source_lines": list(SOURCE_LINES),
        "source_call_lines": list(SOURCE_CALL_LINES),
        "kernel": (dict(binding.kernels[0]) if getattr(binding, "kernels", ())
                   else {"name": "GroupedMatmulSwigluQuantV2", "role": "mix",
                         "evidence": "source_caller"}),
        "dispatcher_source": {
            "path": DISPATCHER_PATH, "sha256": DISPATCHER_SHA,
            "relation": "provides MoEMlpComputeInput.hidden_states/group_list",
        },
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": argument_order,
        "optional_slots": {
            "bias": "not_passed",
            "swiglu_limit": "not_passed",
            "sixth_input": "not_named_by_python_caller",
        },
        "outputs": {"out": "output", "out_scale": "output_scale",
                    "return_arity": 2},
        "list_arguments": {"weight": 1, "weight_scale": 1},
        "format": {
            "x": "fp4_e2m1_packed_uint8",
            "weight": "fp4_e2m1_packed_uint8",
            "weight_scale": "fp8_e8m0",
            "x_scale": "fp8_e8m0",
            "output": "fp4_e2m1_packed_uint8",
            "output_scale": "fp8_e8m0",
            "group_size": 32,
        },
        "surface": {
            "call": ENTRY,
            "argument_order": argument_order,
            "weight_list_length": 1,
            "weight_scale_list_length": 1,
            "outputs": ["out", "out_scale"],
            "returns_tuple": True,
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "shape_fields": shape_fields,
        "group_list": {
            "values": cumulative, "dtype": "int64",
            "layout": "cumulative", "basis": group_basis,
            "group_list_type": 0,
            "selection": "gmm1_cumulative0",
            "gmm1_group_list": receipt["gmm1_group_list"],
            "gmm1_group_list_type": receipt["gmm1_group_list_type"],
            "gmm2_group_list": receipt["gmm2_group_list"],
            "gmm2_group_list_type": receipt["gmm2_group_list_type"],
            "input_group_list": receipt["input_group_list"],
            "input_group_list_type": receipt["input_group_list_type"],
            "provenance": shape_fields,
            "valid_rows": valid,
            "source": (
                "GMM1: cumsum_group_list(input.group_list, "
                "input.group_list_type, 0); GMM2: original list/type"),
        },
        "capacity": {
            "rows": rows, "upper_bound": upper_bound,
            "basis": group_basis, "valid_rows": valid,
            "dispatch_rows": rows,
        },
        "dispatch_rows": rows,
        "upper_bound": upper_bound,
        "deployment": {
            "image_id": digest, "revision": "r64/8d422528",
            "selection": "R64 OCI image + W4A4 routed-expert gate/up op",
            "source_sha256": SOURCE_SHA,
            "source_anchors": list(SOURCE_ANCHORS),
            "dispatcher_sha256": DISPATCHER_SHA,
        },
        "match": {
            "native_type": "GroupedMatmulSwigluQuantV2",
            "entry": ENTRY, "call_id": binding.call_id,
            "deployment_image_id": digest,
            "source_revision": "r64/8d422528",
            "implementation": IMPLEMENTATION,
            "source_sha256": SOURCE_SHA,
            "source_lines": list(SOURCE_LINES),
            "source_call_lines": list(SOURCE_CALL_LINES),
            "source_anchors": list(SOURCE_ANCHORS),
            "k": k, "n": n, "expert_width": width,
            "experts_per_device": local, "top_k": top_k,
            "valid_rows": valid, "dispatch_rows": rows,
            "upper_bound": upper_bound,
            "local_expert_ids": shape_fields["local_expert_ids"],
            "rank_index": receipt["rank"],
            "group_list_basis": group_basis,
            "group_list_provenance": shape_fields,
            "capacity_basis": group_basis,
            "sixth_float_input": "present_in_measured_profile_not_caller_named",
            "shape_contract_complete": shape_complete,
            "gmm1_shape_complete": receipt["gmm1_shape_complete"],
            "gmm2_shape_complete": receipt["gmm2_shape_complete"],
            "pair_shape_complete": receipt["shape_complete"],
            "input_group_list_type": receipt["input_group_list_type"],
            "partial_fields": partial_fields,
            "complete": False,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": ("python_caller_verified_by_dual_git_anchor; "
                       "aclnn_c_abi_and_kernel_binary_not_retained"),
            "owner_kernel": "GroupedMatmulSwigluQuantV2",
            "native_call": ACLNN_SYMBOL,
        },
        "complete": False,
        "shape_contract_complete": shape_complete,
        "gmm1_shape_complete": receipt["gmm1_shape_complete"],
        "gmm2_shape_complete": receipt["gmm2_shape_complete"],
        "pair_shape_complete": receipt["shape_complete"],
        "partial": True,
        "partial_fields": partial_fields,
        "receipt": shape_fields,
        "partial_reasons": list(dict.fromkeys(partial_reasons)),
        "unresolved": list(dict.fromkeys(unresolved)),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration",
            "no_default_swiglu_limit_or_bias_argument",
            "no_fabricated_dispatch_row_capacity",
            "no_global_uniform_routing_as_actual",
            "no_cross_rank_fabricated_counts",
            "no_private_route_ledger_schema",
            "no_kernel_binary_identity",
            "no_gmm2_expert_down",
        ],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {"dtype": validated.get("dtype"),
                             "layout": validated.get("layout"),
                             "shapes": validated.get("shapes"),
                             "attrs": validated.get("attrs")}}
