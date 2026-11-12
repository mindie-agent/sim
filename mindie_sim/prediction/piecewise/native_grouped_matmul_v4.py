"""Prepare-only standalone GroupedMatmulV4 invocation for routed down.

[Role] Expand the selected R64 A5 W4A4 routed-expert down projection into the
  real ``torch_npu.npu_grouped_matmul`` operand, grouped-row, BF16 output and
  E8M0 scale descriptors for one evaluated occurrence.
[Invariants] Only the R64 image's W4A4 MXFP4 ``apply_gmm2`` caller is
  admitted.  Static K/N/local-expert identity stays in the binding.  The
  native row count and group_list values are used only from a declared
  per-layer per-EP route ledger; without that ledger the row count stays
  symbolic/None and the occurrence is partial.  Uniform routing expectations,
  scheduler upper bounds and cross-rank equal averages are never native
  shapes.  The first grouped SwiGLU+quant call is a different owner and is
  never merged into this call.  No tensor is allocated and no pointer or
  numeric prediction is fabricated.
[Timeouts & Locks] Pure bounded metadata arithmetic; no I/O or locks.
[Side-Effect Policy] Pure metadata derived from the existing selected
  operator, declared route ledger and source call template.
"""
from __future__ import annotations

from typing import Any, Mapping

ENTRY = "torch_npu.npu_grouped_matmul"
ACLNN_SYMBOL = "aclnnGroupedMatmulV4_GroupedMatmul_GroupedMatmul"
SCHEMA = "mindie-sim/grouped-matmul-v4-native-prepare/1"
R64_IMAGE_SHA = "8d422528ac20a8babcbbf93762cde3b47d4db6c5f7ab777f4f1f65b32b89ab3e"
SOURCE_PATH = "vllm_ascend/quantization/methods/w4a4/w4a4_mxfp4.py"
SOURCE_SHA = "cbf45467019febc009d26d23492a7a4b5a7784a9f047725c58aa8558998bff0e"
SOURCE_SYMBOL = "AscendW4A4MXFP4DynamicFusedMoEMethod.apply_gmm2"
SOURCE_LINES = [452, 477]
CALL_LINES = [462, 477]
GROUP_LIST_LINES = [469, 471]
DISPATCHER_PATH = "vllm_ascend/ops/fused_moe/token_dispatcher.py"
DISPATCHER_SHA = "1638f673bd4ed58af2e996c20a0197a076e5f6ba7420eba447f0ee75430d96a5"
DISPATCHER_SYMBOL = "TokenDispatcherWithAll2AllV._dispatch_postprocess"
DISPATCHER_LINES = [704, 715]
DISPATCHER_CALL_LINES = [706, 714]
SOURCE_GROUP_LIST_TYPE = 1
IMPLEMENTATION = "W4A4_MXFP4"
_FP4 = frozenset({"mxfp4", "mx_fp4", "fp4", "fp4_e2m1", "float4_e2m1",
                  "float4_e2m1fn_x2"})
_E8M0 = frozenset({"fp8_e8m0", "e8m0", "ue8m0", "float8_e8m0fnu"})
_MAX_ROWS = 1_048_576

# The W4A4 source file is byte-identical at every commit whose tree also
# reproduces the eight independently retained R64 file hashes below.  The
# deployed image is the R64 OCI image sha256:8d422528...; the local git blob
# is therefore the exact source sibling for any R64 tree consistent with the
# retained files.  No remote read or clone was performed.
_R64_PIN = {
    "method": "byte_identical_across_r64_file_hash_intersection",
    "w4a4_blob_sha256": SOURCE_SHA,
    "commits": ["e5f5fafd8", "d9b3281c1", "b6aded6f0", "b3876a1dc",
                "a26cc45a2", "4926e0a22", "2baa19334", "1de10119d"],
    "retained_files": {
        "vllm_ascend/quantization/methods/w8a8/w8a8_mxfp8.py":
            "38de2275cf33ced653b8398e98a07f483489dc8e7353b2b3c238a5ff00782f79",
        "vllm_ascend/device/device_op.py":
            "2a2512595a08b1fd58f69772cf3cc830c0b62f702365f80160c7fa6df3981a5d",
        "vllm_ascend/ascend_forward_context.py":
            "6eb281094ecd732de4969d3e9fb20fd03f47a7fd0a1e5323f50413dc60acf764",
        "vllm_ascend/quantization/utils.py":
            "459625b94d258b8eefb5aefca3525d6f6f64d86097be7593548af19f0a254d85",
        "vllm_ascend/ascend_config.py":
            "6af63d7f66000d2a195732ef25dba3d4d47fd638e96a82f553a937fa0b40b047",
        "vllm_ascend/device/hardware_profile.py":
            "a0b0d882ffcf6cbc5a282ca038811b2a69ce1ddbe557cb54a06b04c76b1842dd",
        "vllm_ascend/attention/sfa_v1.py":
            "82eeef0272977cc2bf0c870699eba005ae71236b1eef7c4b71fe67303a621e68",
        "vllm_ascend/attention/mla_v1.py":
            "55cdabc1bcb85a16905094b03d3eeabeebb108f26e16f6f55e22ad71934cbf45",
    },
}


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
    output_format = str(quant.get("output_format") or "").lower()
    if (weight_format not in _FP4 or act_format not in _FP4
            or output_format not in {"bf16", "bfloat16"}
            or int(quant.get("scale_group", 0) or 0) != 32):
        return None
    if str(quant.get("linear_method") or "").lower() not in {"mx_fp4", "mxfp4"}:
        return None
    if str(quant.get("weight_scale_format")
           or quant.get("weight_scale_storage_format") or "").lower() not in _E8M0:
        return None
    if extras.get("activation_source") != "grouped_gate_up_activation_quant":
        return None
    if float(extras.get("activation_payload_row_bytes", 0) or 0) <= 0:
        return None
    if float(extras.get("activation_scale_bytes", 0) or 0) <= 0:
        return None
    try:
        k = int(matrix.get("K", 0) or 0)
        n = int(matrix.get("N", 0) or 0)
        width = int(extras.get("expert_width", 0) or 0)
        latent = int(extras.get("latent", n) or 0)
        experts = int(extras.get("E", 0) or 0)
        local = int(extras.get("E_local", extras.get("local", 0)) or 0)
        top_k = int(extras.get("k", 0) or 0)
    except (TypeError, ValueError):
        return None
    # Routed down is (K=expert_width, N=latent); the gate/up composite is the
    # opposite orientation and belongs to a different owner.
    if (k < 1 or n < 1 or width < 1 or latent < 1 or k != width or n != latent
            or experts < 1 or local < 1 or local > experts or top_k < 1):
        return None
    return {"theta": theta, "quant": quant, "extras": extras,
            "matrix": matrix, "K": k, "N": n, "expert_width": width,
            "latent": latent, "experts": experts, "local": local,
            "top_k": top_k}


def binding_spec(compiled: Any, *, phase: str = "") -> Any:
    """Return the real shared ``native_calls._CallSpec`` or None.

    Narrow admission: F3 routed-expert down, R64 image, W4A4 MXFP4/E8M0
    group 32, BF16 return.  Every other source/representation keeps the
    existing path unchanged.  The binding is static; no route-ledger row
    count is embedded in it.
    """
    from . import native_calls

    if getattr(compiled, "kind", "") != "F3":
        return None
    operator = getattr(compiled, "operator", None)
    if operator is None:
        return None
    if (getattr(operator, "name", "") != "expert_down"
            or getattr(operator, "family", "") != "grouped_gemm"):
        return None
    terms = _static_terms(compiled)
    if terms is None:
        return None
    selected = _source_meta(compiled)
    if selected is None:
        return None
    _meta, digest = selected
    k, n = terms["K"], terms["N"]
    width, latent = terms["expert_width"], terms["latent"]
    experts, local, top_k = terms["experts"], terms["local"], terms["top_k"]
    pairs = _ceil_div(_ceil_div(k, 32), 2)
    typed = {
        "dtype": "mxfp4", "layout": "ND",
        "shapes": {
            "x": [k],
            "weight": [local, n, k],
            "scale": [local, n, pairs, 2],
            "per_token_scale": [pairs, 2],
            "group_list": [local],
            "output": [n],
        },
        "attrs": {
            "matrix_k": k, "matrix_n": n, "latent": latent,
            "expert_width": width, "experts": experts,
            "experts_per_device": local, "top_k": top_k,
            "scale_group": 32,
            "weight_format": "mxfp4", "act_format": "mxfp4",
            "output_format": "bf16", "prequant_format": "bf16",
            "linear_method": "mx_fp4", "implementation": IMPLEMENTATION,
            "activation_source": "grouped_gate_up_activation_quant",
            "weight_scale_format": "fp8_e8m0",
            "activation_scale_format": "fp8_e8m0",
            "source_revision": "r64", "source_lines": list(SOURCE_LINES),
            "deployment_image_id": digest,
            "group_list_dtype": "int64",
            "bias": False, "offset": None,
            "x_dtype": "fp4_e2m1", "weight_dtype": "fp4_e2m1",
            "output_dtype": "bf16",
            "scale_dtype": "fp8_e8m0",
            "per_token_scale_dtype": "fp8_e8m0",
            "split_item": 2, "group_type": 0,
            "weight_layout": "ND", "activation_layout": "ND",
            "weight_storage_dtype": "uint8",
            "weight_storage_shape": [local, n, k // 2],
            "weight_logical_shape": [local, n, k],
            "weight_scale_storage_dtype": "uint8",
            "weight_scale_interpret_dtype": "fp8_e8m0",
            "call_template_source": SOURCE_PATH,
            "call_template_sha256": SOURCE_SHA,
            "call_template_lines": list(CALL_LINES),
            "group_list_call_lines": list(GROUP_LIST_LINES),
            "dispatcher_path": DISPATCHER_PATH,
            "dispatcher_sha256": DISPATCHER_SHA,
            "dispatcher_symbol": DISPATCHER_SYMBOL,
            "dispatcher_lines": list(DISPATCHER_LINES),
            "dispatcher_call_lines": list(DISPATCHER_CALL_LINES),
        },
        "occurrence_attrs": {
            "received_rows": "optional_positive_int",
            "input_rows": "optional_positive_int",
            "capacity_rows": "optional_positive_int",
            "valid_rows": "optional_positive_int",
            "group_list_values": "optional_int_list",
            "group_list_basis": "optional_string",
            "capacity_basis": "optional_string",
            "input_rows_basis": "optional_string",
            "phase": "string",
        },
        "rank_width": 4,
    }
    source = native_calls._source(
        "vllm-ascend", SOURCE_PATH, sha256=SOURCE_SHA,
        symbol=SOURCE_SYMBOL, revision="r64/8d422528",
        state="r64_local_git_pin")
    source["r64_pin"] = dict(_R64_PIN)
    kernel = native_calls._kernel(
        "GroupedMatmul", "cube", evidence="runtime_caller",
        native_call=ACLNN_SYMBOL, source_status="runtime_caller_only",
        caller_path=SOURCE_PATH, caller_sha256=SOURCE_SHA,
        caller_symbol=SOURCE_SYMBOL, caller_lines=list(CALL_LINES),
        cann_source_status="not_retained")
    unresolved = (
        "aclnn_symbol_from_runtime_caller_not_retained_header",
        "cann_grouped_matmul_v4_kernel_source_not_retained",
        "binary_identity_from_kernel_details_name_only",
        "actual_received_rows_require_shared_expert_receipt",
    )
    return native_calls._call_spec(
        native_calls._api("torch", ENTRY, aclnn_symbol=ACLNN_SYMBOL,
                          version="r64/8d422528"),
        source, typed, (kernel,), (), getattr(operator, "name", ""),
        execution_core="aic", unresolved=unresolved)


def _for_subgraph(geometry: Any, kind: str) -> Any:
    method = getattr(geometry, "for_subgraph", None)
    if callable(method):
        try:
            return method(kind)
        except Exception:
            return geometry
    return geometry


def _expert_receipt(compiled: Any, geometry: Any) -> dict[str, Any] | None:
    """Consume OPGMM1R3's public shared receipt for this layer/rank."""
    from mindie_sim.prediction.moe_callers import project_expert_receipt
    geom = _for_subgraph(geometry, "F3")
    state = getattr(compiled, "state", None)
    if state is None:
        state = getattr(geom, "state", None)
    return project_expert_receipt(
        compiled, geom, state=state, activation_dtype="fp4_e2m1",
        scale_dtype="fp8_e8m0", expected_scope="grouped_gate_up")


def _source_group_list_encoding(terms: Mapping[str, Any], phase: str,
                               meta: Mapping[str, Any]
                               ) -> tuple[int | None, str, bool | None]:
    """Source-selected declaring caller protocol for the incoming group_list.

    ``apply_gmm2`` L469/471 forwards ``input.group_list_type/input.group_list``
    unchanged.  An explicit incoming type wins.  Otherwise the selected
    All2All dispatcher (1638f673) declares type-1 counts; a selected MC2
    decode producer does not declare the same incoming encoding here, so it
    stays unknown instead of being forced to type 1.  This only reads the
    already selected ``moe_comm`` config and metadata; it does not recalculate
    a route, rank or capacity.
    """
    extras = dict(terms.get("extras") or ())
    raw_comm = extras.get("moe_comm")
    comm = dict(raw_comm) if isinstance(raw_comm, Mapping) or (
        isinstance(raw_comm, (list, tuple))) else {}
    enable_fused = bool(int(comm.get("enable_fused_mc2") or 0))
    enable_prefill = bool(int(comm.get("enable_prefill_mc2") or 0))
    if phase == "prefill":
        mc2 = bool(enable_prefill and enable_fused)
    elif phase == "decode":
        mc2 = enable_fused
    else:
        mc2 = None
    raw_type = None
    for source in (meta, extras):
        for key in ("moe_input_group_list_type", "moe_group_list_type",
                    "input_group_list_type"):
            if raw_type is None and source.get(key) is not None:
                raw_type = source.get(key)
    if raw_type is not None:
        parsed = (raw_type if isinstance(raw_type, int)
                  and not isinstance(raw_type, bool) else None)
        if parsed in {0, 1}:
            return parsed, "declared_input_group_list_type", mc2
        return None, "input_group_list_type_invalid", mc2
    if mc2 is True:
        return None, "mc2_producer_group_list_type_unknown", True
    if mc2 is False:
        return SOURCE_GROUP_LIST_TYPE, "dispatcher_1638f673_type1_counts", False
    return None, "source_protocol_unknown", None


def _descriptor(shape: list, dtype: str, layout: str, role: str,
                **extra: Any) -> dict:
    row = {
        "shape": [None if item is None else int(item) for item in shape],
        "dtype": dtype, "layout": layout, "role": role, "pointer": None,
        "allocation": "external",
    }
    row.update(extra)
    return row


def dynamic_call(binding: Any, compiled: Any, geometry: Any, *,
                 validate: Any) -> dict | None:
    """Prepare-only occurrence for the selected W4A4 grouped down.

    The row axis, group_list and upper bound come from OPGMM1R3's single
    ``project_expert_receipt``.  If the receipt has no actual R, the row axis
    stays symbolic/None and the envelope is explicitly partial; no uniform
    expectation or scheduler bound is substituted.
    """
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
    if getattr(operator, "name", "") != "expert_down":
        return None
    terms = _static_terms(compiled)
    selected = _source_meta(compiled)
    if terms is None or selected is None:
        return None
    _meta, digest = selected
    k, n = terms["K"], terms["N"]
    local = terms["local"]
    phase = str(getattr(geometry, "phase_name", "") or "")
    meta = dict(getattr(compiled, "feature_metadata", ()) or ())
    current_rank = meta.get("moe_rank_index", meta.get("rank_index"))
    receipt = _expert_receipt(compiled, geometry)
    received = input_rows = valid = capacity = None
    group_values = group_layout = group_list_type = None
    canonical_group_list = None
    canonical_counts = None
    gmm1_group_list = None
    gmm1_group_list_type = None
    source_type = None
    source_encoding = "original_encoding_not_declared"
    mc2_unknown = False
    expert_ids = expert_rows = None
    receipt_basis = "shared_expert_receipt_unavailable"
    receipt_symbolic = True
    conflicts: tuple = ()
    shape_complete = False
    act_dtype, scale_dtype = "fp4_e2m1", "fp8_e8m0"
    if receipt is not None:
        input_rows = receipt["input_rows"]
        received = receipt["received_rows"]
        valid = receipt["valid_rows"]
        capacity = receipt["capacity_rows"]
        expert_ids = (list(receipt["local_expert_ids"])
                      if receipt["local_expert_ids"] is not None else None)
        expert_rows = (list(receipt["expert_rows"])
                       if receipt["expert_rows"] is not None else None)
        conflicts = tuple(receipt.get("unresolved") or ())
        canonical_group_list = (list(receipt["canonical_group_list"])
                                if receipt.get("canonical_group_list") is not None
                                else (list(receipt["group_list"])
                                      if receipt["group_list"] is not None else None))
        canonical_counts = (list(receipt["canonical_counts"])
                            if receipt.get("canonical_counts") is not None
                            else expert_rows)
        gmm1_group_list = (list(receipt["gmm1_group_list"])
                           if receipt.get("gmm1_group_list") is not None
                           else canonical_group_list)
        gmm1_group_list_type = receipt.get("gmm1_group_list_type")
        if gmm1_group_list_type is None and gmm1_group_list is not None:
            gmm1_group_list_type = 0
        helper_values = (list(receipt["gmm2_group_list"])
                         if receipt.get("gmm2_group_list") is not None else None)
        helper_type = receipt.get("gmm2_group_list_type")
        if helper_type not in {0, 1}:
            helper_type = None
        source_type, source_encoding, mc2_unknown = (
            _source_group_list_encoding(terms, phase, meta))
        if helper_type is not None:
            if (source_encoding == "declared_input_group_list_type"
                    and helper_type != source_type):
                conflicts = tuple(dict.fromkeys(
                    (*conflicts, "shared_receipt_gmm2_type_disagrees")))
                helper_type = None
                helper_values = None
            else:
                source_type = helper_type
                source_encoding = "shared_receipt_gmm2_original_encoding"
                mc2_unknown = False
        group_list_type = source_type
        group_values = None
        group_layout = None
        if helper_values is not None and helper_type is not None:
            group_values = helper_values
            group_layout = "count" if helper_type == 1 else "cumulative"
        elif helper_type is not None:
            group_layout = "count" if helper_type == 1 else "cumulative"
            source_encoding = "shared_receipt_gmm2_type_only"
        elif source_type == 1:
            group_layout = "count"
            counts = canonical_counts
            if (counts is not None and len(counts) == local
                    and (received is None or sum(counts) == received)
                    and (valid is None or sum(counts) == valid)):
                group_values = list(counts)
            elif counts is not None:
                conflicts = tuple(dict.fromkeys(
                    (*conflicts, "counts_do_not_match_received_rows")))
        elif source_type == 0:
            group_layout = "cumulative"
            if canonical_group_list is not None and len(canonical_group_list) == local:
                group_values = list(canonical_group_list)
            elif canonical_group_list is not None:
                conflicts = tuple(dict.fromkeys(
                    (*conflicts, "cumulative_group_list_length_mismatch")))
        receipt_basis = str(receipt["basis"] or "shared_expert_receipt")
        receipt_symbolic = bool(receipt["symbolic"])
        shape_complete = bool(receipt.get("gmm2_shape_complete",
                                          receipt.get("shape_complete", False)))
        raw_act = str(receipt["activation_dtype"] or "fp4_e2m1").lower()
        raw_scale = str(receipt["scale_dtype"] or "fp8_e8m0").lower()
        if raw_act not in _FP4 or raw_scale not in _E8M0:
            receipt_basis = "shared_expert_receipt_stage_dtype_unsupported"
            receipt_symbolic = True
            input_rows = received = valid = capacity = None
            group_values = group_layout = group_list_type = None
            expert_ids = expert_rows = None
            conflicts = tuple(dict.fromkeys((*conflicts,
                                             "stage_dtype_unsupported")))
        else:
            act_dtype, scale_dtype = raw_act, raw_scale
    # The receipt's rank is the current rank supplied by the explicit
    # vocabulary/state.  Refuse it when the actual current rank metadata
    # contradicts it instead of trusting an arbitrary int.
    if (receipt is not None and receipt.get("rank") is not None
            and current_rank is not None):
        try:
            current_rank_i = int(current_rank)
        except (TypeError, ValueError):
            current_rank_i = None
        if current_rank_i is not None and int(receipt["rank"]) != current_rank_i:
            receipt_basis = "shared_expert_receipt_identity_mismatch"
            receipt_symbolic = True
            input_rows = received = valid = capacity = None
            group_values = group_layout = group_list_type = None
            expert_ids = expert_rows = None
            conflicts = tuple(dict.fromkeys((*conflicts,
                                             "receipt_rank_mismatch")))
    if input_rows is None or received is None:
        receipt_symbolic = True
    if input_rows is not None and input_rows > _MAX_ROWS:
        return None
    pairs = _ceil_div(_ceil_div(k, 32), 2)
    actual_shapes = {
        "x": [input_rows, k],
        "weight": [local, n, k],
        "scale": [local, n, pairs, 2],
        "per_token_scale": [input_rows, pairs, 2],
        "group_list": [local],
        "output": [input_rows, n],
    }
    input_rows_basis = ("shared_expert_receipt"
                        if input_rows is not None else "symbolic")
    actual_attrs = {
        **attrs,
        "received_rows": received,
        "input_rows": input_rows,
        "capacity_rows": capacity,
        "valid_rows": valid,
        "dispatch_rows": input_rows,
        "upper_bound": capacity,
        "group_list_values": group_values,
        "group_list_basis": receipt_basis,
        "capacity_basis": receipt_basis,
        "input_rows_basis": input_rows_basis,
        "phase": phase,
        "m": valid,
        "padded_capacity_rows": capacity,
        "group_list_layout": group_layout or "undeclared",
        "group_list_type": (group_list_type
                            if group_list_type is not None
                            else "forwarded_from_caller"),
        "local_expert_ids": expert_ids,
        "expert_rows": expert_rows,
        "stage_activation_dtype": act_dtype,
        "stage_scale_dtype": scale_dtype,
        "receipt_symbolic": receipt_symbolic,
        "receipt_conflicts": list(conflicts),
        "shape_contract_complete": shape_complete,
    }
    actual = {"dtype": typed.get("dtype"), "layout": typed.get("layout"),
              "shapes": actual_shapes, "attrs": actual_attrs}
    validated = validate(binding, actual)
    descriptors = {
        "x": _descriptor([input_rows, k], "fp4_e2m1", "ND", "input",
                         storage_dtype="uint8",
                         storage_shape=[None if input_rows is None
                                        else input_rows, k // 2],
                         interpret_dtype="fp4_e2m1",
                         logical_shape=[input_rows, k],
                         allocation_shape=([capacity, k]
                                           if capacity is not None else None)),
        "weight": _descriptor([local, n, k], "fp4_e2m1", "ND", "input",
                              storage_dtype="uint8",
                              storage_shape=[local, n, k // 2],
                              interpret_dtype="fp4_e2m1",
                              logical_shape=[local, n, k],
                              device_layout="ND"),
        "scale": _descriptor([local, n, pairs, 2], "fp8_e8m0", "ND",
                             "input", storage_dtype="uint8",
                             interpret_dtype="fp8_e8m0",
                             logical_shape=[local, n, pairs, 2]),
        "per_token_scale": _descriptor([input_rows, pairs, 2], "fp8_e8m0",
                                       "ND", "input", storage_dtype="uint8",
                                       interpret_dtype="fp8_e8m0",
                                       allocation_shape=([capacity, pairs, 2]
                                                         if capacity is not None
                                                         else None)),
        "group_list": _descriptor([local], "int64", "ND", "input",
                                  values=group_values,
                                  layout_semantics=(group_layout
                                                    or "undeclared"),
                                  local_expert_ids=expert_ids,
                                  counts=expert_rows),
        "output": _descriptor([input_rows, n], "bf16", "ND", "output",
                              allocation_shape=([capacity, n]
                                                if capacity is not None
                                                else None)),
        "bias": {"shape": None, "dtype": None, "layout": None,
                 "role": "input_optional", "present": False,
                 "pointer": None, "allocation": "external"},
        "offset": {"shape": None, "dtype": None, "layout": None,
                   "role": "input_optional", "present": False,
                   "pointer": None, "allocation": "external"},
    }
    arguments = {
        "x": "x", "weight": "weight", "scale": "scale", "bias": None,
        "per_token_scale": "per_token_scale", "split_item": 2,
        "group_list_type": group_list_type, "group_type": 0,
        "group_list": "group_list", "output_dtype": "bf16",
        "scale_dtype": "fp8_e8m0",
        "per_token_scale_dtype": "fp8_e8m0",
        "x_dtype": "fp4_e2m1", "weight_dtype": "fp4_e2m1",
    }
    argument_order = [
        "x", "weight", "scale", "bias", "per_token_scale", "split_item",
        "group_list_type", "group_type", "group_list", "output_dtype",
        "scale_dtype", "per_token_scale_dtype", "x_dtype", "weight_dtype",
    ]
    partial_reasons = [
        "cann_grouped_matmul_v4_kernel_source_not_retained",
        "binary_identity_from_kernel_details_name_only",
    ]
    if receipt_basis == "shared_expert_receipt_identity_mismatch":
        partial_reasons.append("shared_expert_receipt_identity_mismatch")
        partial_reasons.append("native_shape_uses_symbolic_R")
    elif receipt_basis == "shared_expert_receipt_stage_dtype_unsupported":
        partial_reasons.append(
            "shared_expert_receipt_stage_dtype_unsupported")
        partial_reasons.append("native_shape_uses_symbolic_R")
    if source_type is None:
        partial_reasons.append("original_group_list_type_not_declared")
        if mc2_unknown:
            partial_reasons.append(
                "mc2_incoming_group_list_type_not_declared")
    elif source_type == 1 and group_values is None:
        partial_reasons.append("source_group_list_counts_not_declared")
    elif source_type == 0 and group_values is None:
        partial_reasons.append("canonical_group_list_not_declared")
    if input_rows is None:
        partial_reasons.append("received_rows_not_declared")
    if expert_ids is None:
        partial_reasons.append("local_expert_ids_not_declared")
    if capacity is None:
        partial_reasons.append("upper_bound_not_declared")
    if receipt_symbolic:
        partial_reasons.append("shared_expert_receipt_symbolic")
    if source_type is not None:
        # GMM2 supplies the original type from the dispatcher anchor/shared
        # gmm2 fields; the helper's generic "type not declared" reasons no
        # longer apply.  Absent counts/values remain explicit.
        stale = {"gmm2_group_list_type_not_declared",
                 "input_group_list_type_not_declared"}
        if group_values is not None:
            stale.add("gmm2_group_list_not_declared")
        conflicts = tuple(reason for reason in conflicts if reason not in stale)
    partial_reasons.extend(str(reason) for reason in conflicts)
    prepare = {
        "schema": SCHEMA,
        "call": ENTRY,
        "call_id": binding.call_id,
        "signature_hash": typed.get("signature_hash"),
        "implementation": IMPLEMENTATION,
        "source": dict(getattr(binding, "source", None) or {}),
        "source_lines": list(SOURCE_LINES),
        "call_template": {
            "path": SOURCE_PATH, "sha256": SOURCE_SHA,
            "symbol": SOURCE_SYMBOL, "lines": list(CALL_LINES),
            "group_list_lines": list(GROUP_LIST_LINES),
            "r64_pin": dict(_R64_PIN),
        },
        "dispatcher_source": {
            "path": DISPATCHER_PATH, "sha256": DISPATCHER_SHA,
            "symbol": DISPATCHER_SYMBOL, "lines": list(DISPATCHER_LINES),
            "call_lines": list(DISPATCHER_CALL_LINES),
            "relation": "provides mlp_compute_input.group_list_type/group_list; "
                        "source-declared type1 counts, INT64[16]",
        },
        "kernel": (dict(binding.kernels[0]) if getattr(binding, "kernels", ())
                   else {"name": "GroupedMatmul", "role": "cube",
                         "evidence": "runtime_caller",
                         "native_call": ACLNN_SYMBOL}),
        "stage": "interface_prepared",
        "runtime_executed": False,
        "allocation": "not_performed",
        "arguments": arguments,
        "argument_order": argument_order,
        "optional_slots": {
            "bias": "not_present",
            "offset": "not_passed_by_source",
            "group_list_type": ("source_declared"
                                if group_list_type is not None
                                else "original_encoding_not_declared"),
            "x_dtype": "fp4_e2m1",
            "weight_dtype": "fp4_e2m1",
        },
        "shapes": dict(validated.get("shapes") or {}),
        "descriptors": descriptors,
        "arrays": descriptors,
        "receipt": receipt,
        "group_list": {
            "values": group_values,
            "dtype": "int64",
            "layout": group_layout or "undeclared",
            "basis": receipt_basis,
            "group_list_type": group_list_type,
            "source_encoding": source_encoding,
            "mc2_producer_unknown": bool(mc2_unknown),
            "canonical_cumulative": gmm1_group_list,
            "canonical_group_list_type": gmm1_group_list_type,
            "canonical_counts": canonical_counts,
            "canonical_view_owner": "GMM1 apply_gmm1 L397 cumulative0",
            "valid_rows": valid,
            "local_expert_ids": expert_ids,
            "counts": expert_rows,
            "source": ("project_expert_receipt" if receipt is not None
                       else "unavailable"),
            "note": "GMM2 passes the original apply_gmm2 L469/471 "
                    "group_list_type/group_list; GMM1's canonical view is "
                    "cumulative0, not substituted here",
        },
        "capacity": {
            "rows": capacity,
            "upper_bound": capacity,
            "basis": receipt_basis,
            "input_rows": input_rows,
            "dispatch_rows": input_rows,
            "received_rows": received,
            "valid_rows": valid,
            "declared_for_native_branch": capacity is not None,
        },
        "expected_cost_view": {
            "available": True,
            "basis": "existing_f3_analytical_cost",
            "used_as_native_shape": False,
            "note": "the predicted expected cost remains in the occurrence "
                    "priced view; it is not an input shape",
        },
        "deployment": {
            "image_id": digest, "revision": "r64/8d422528",
            "selection": "R64 OCI image + W4A4 routed-expert down op",
            "source_sha256": SOURCE_SHA,
            "source_pinned_by": dict(_R64_PIN),
        },
        "match": {
            "native_type": "GroupedMatmulV4",
            "entry": ENTRY,
            "aclnn_symbol": ACLNN_SYMBOL,
            "call_id": binding.call_id,
            "deployment_image_id": digest,
            "source_revision": "r64/8d422528",
            "implementation": IMPLEMENTATION,
            "source_sha256": SOURCE_SHA,
            "source_lines": list(SOURCE_LINES),
            "k": k, "n": n, "expert_width": terms["expert_width"],
            "latent": terms["latent"], "experts_per_device": local,
            "top_k": terms["top_k"],
            "received_rows": received, "input_rows": input_rows,
            "capacity_rows": capacity, "valid_rows": valid,
            "local_expert_ids": expert_ids,
            "group_list_basis": receipt_basis,
            "group_list_type": group_list_type,
            "group_list_layout": group_layout,
            "group_list_source_encoding": source_encoding,
            "mc2_group_list_type_unknown": bool(mc2_unknown),
            "canonical_group_list": gmm1_group_list,
            "canonical_group_list_type": gmm1_group_list_type,
            "canonical_counts": canonical_counts,
            "capacity_basis": receipt_basis,
            "input_rows_basis": input_rows_basis,
            "native_shape_basis": ("shared_expert_receipt"
                                   if input_rows is not None else "symbolic"),
            "receipt_symbolic": receipt_symbolic,
            "shape_contract_complete": shape_complete,
            "complete": False,
        },
        "binary_selection": {
            "status": "uncertain",
            "reason": "kernel_details_native_name_without_retained_cann_source",
            "owner_kernel": "GroupedMatmul",
            "native_call": ACLNN_SYMBOL,
        },
        "complete": False,
        "partial": True,
        "partial_reasons": list(dict.fromkeys(partial_reasons)),
        "unresolved": list(dict.fromkeys(
            getattr(binding, "unresolved", ()) or ())),
        "non_claims": [
            "no_runtime_allocation", "no_device_launch",
            "no_measured_duration", "no_complete_cann_kernel_source",
            "no_uniform_moe_counts_as_native_rows",
            "no_scheduler_upper_bound_as_native_rows",
            "no_cross_rank_expert_counts",
            "no_private_route_ledger_schema",
            "no_gmm2_normalization_to_cumulative",
            "symbolic_R_when_shared_expert_receipt_absent",
            "capacity_is_a_declared_bound_not_an_observed_allocation",
            "down_call_is_not_the_first_grouped_swiglu_quant_call",
        ],
    }
    return {"actual": validated, "prepare": prepare,
            "prepare_args": {"dtype": validated.get("dtype"),
                             "layout": validated.get("layout"),
                             "shapes": validated.get("shapes"),
                             "attrs": validated.get("attrs")}}
