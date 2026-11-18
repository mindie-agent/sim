"""Strict projections of the tensor descriptors recorded by native profilers.

[Role] Preserve ordered shape, dtype and storage-format slots for native ABIs;
  publish a content-addressed ABI only when every nonempty slot is complete.
[Invariants] Explicit undefined slots and typed empty shapes remain distinct.
  Tensor descriptors do not identify scalar arguments, tensor values or a model
  occurrence. Missing fields never become a guessed shape, layout or digest.
[Timeouts & Locks] Pure in-memory work proportional to the number of tensor slots.
[Side-Effect Policy] No I/O, hardware context or simulation imports.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence

ABI_SCHEMA = "mindie-native-tensor-abi/1"

DTYPES = {
    "DT_BF16": "bf16", "DT_FLOAT16": "fp16", "FLOAT": "fp32", "INT8": "int8",
    "INT64": "int64", "INT32": "int32", "BOOL": "bool", "DT_UNDEFINED": None,
    "DT_FLOAT8_E4M3FN": "fp8_e4m3", "DT_FLOAT8_E5M2": "fp8_e5m2",
    "DT_FLOAT8_E8M0": "fp8_e8m0", "DT_FLOAT4_E2M1": "fp4_e2m1",
    "DT_INT4": "int4", "INT4": "int4", "DT_UINT4": "uint4", "UINT4": "uint4",
    "UINT64": "uint64", "DT_UINT64": "uint64", "UINT32": "uint32",
    "DT_UINT32": "uint32", "UINT8": "uint8", "DT_UINT8": "uint8",
    "BF16": "bf16", "FP16": "fp16", "DT_FLOAT": "fp32",
    "bf16": "bf16", "fp16": "fp16", "fp32": "fp32", "fp64": "fp64",
    "int8": "int8", "int4": "int4", "int64": "int64", "int32": "int32",
    "uint64": "uint64", "uint32": "uint32", "uint4": "uint4", "bool": "bool",
    "FLOAT16": "fp16", "DT_INT8": "int8", "DT_INT32": "int32",
    "DT_INT64": "int64", "DT_UINT1": "uint1", "UNDEFINED": None,
    "bfloat16": "bf16", "bfloat16_t": "bf16", "float16": "fp16",
    "float32": "fp32", "float64": "fp64", "float8_e4m3fn": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3", "float8_e5m2": "fp8_e5m2", "fp8_e5m2": "fp8_e5m2",
    "fp8_e8m0": "fp8_e8m0", "fp8_e8m0fnu": "fp8_e8m0", "e8m0": "fp8_e8m0",
    "fp4_e2m1": "fp4_e2m1", "dt_float4_e2m1": "fp4_e2m1", "fp4": "fp4_e2m1",
    "mxfp4": "fp4_e2m1", "e2m1": "fp4_e2m1",
    "dt_float8_e4m3fn": "fp8_e4m3", "dt_float8_e8m0": "fp8_e8m0",
    "dt_int4": "int4", "uint8": "uint8", "uint1": "uint1",
    "DT_BFLOAT16": "bf16", "DT_FLOAT32": "fp32", "DT_FLOAT64": "fp64",
    "DT_HIFLOAT8": "hif8", "HIFLOAT8": "hif8", "HIF8": "hif8",
    "hifloat8": "hif8", "hif8": "hif8",
    "FLOAT4_E2M1": "fp4_e2m1", "FP4_E2M1": "fp4_e2m1",
    "MXFP4": "fp4_e2m1", "float4_e2m1": "fp4_e2m1",
}
_MISSING = {"", "N/A", "NULL", "UNDEFINED", "DT_UNDEFINED"}
_EMPTY_DTYPES = {"UNDEFINED", "DT_UNDEFINED"}
_EMPTY_FORMATS = {"NULL", "UNDEFINED"}


def _undefined_dtype(value):
    """True only for an explicitly declared undefined dtype.

    ``None`` is a present JSON null (an explicit undefined declaration).
    A blank/whitespace string or ``N/A`` is a missing field, not a
    declaration: ``TensorSlot.state`` already reports it as incomplete, and
    the ABI must preserve that gap instead of upgrading it to undefined.
    """
    if value is None:
        return True
    return str(value).strip().upper() in {"UNDEFINED", "DT_UNDEFINED", "NULL"}


def _undefined_layout(value):
    """True only for an explicitly declared undefined layout spelling."""
    if value is None:
        return True
    return str(value).strip().upper() in {"NULL", "UNDEFINED", "DT_UNDEFINED"}


def native_contract_issue(row):
    """Return a concrete missing/unsupported descriptor reason, or None."""
    for direction in ("input", "output"):
        shapes = getattr(row, f"{direction}_shapes")
        if shapes is None or not shapes:
            return {
                "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
                "detail": "Native tensor ABI requires captured input and output descriptors.",
                "missing_fields": [f"{direction}_shapes"], "operator_type": row.kind,
            }
        dtypes = getattr(row, f"{direction}_dtypes")
        formats = getattr(row, f"{direction}_formats")
        if len(shapes) != len(dtypes) or len(shapes) != len(formats):
            return {
                "category": "capture_metadata_missing", "reason": "incomplete_native_tensor_slots",
                "detail": "Native shape, dtype and format arrays must describe the same ordered slots.",
                "direction": direction, "slot_counts": {
                    "shapes": len(shapes), "dtypes": len(dtypes), "formats": len(formats)},
                "operator_type": row.kind,
            }
        for index, (shape, dtype, layout) in enumerate(zip(shapes, dtypes, formats)):
            if any(not isinstance(value, int) or value < 0 for value in shape):
                return {
                    "category": "capture_metadata_missing", "reason": "unknown_tensor_dimensions",
                    "detail": "Negative or noninteger native dimensions cannot establish geometry.",
                    "direction": direction, "slot": index, "operator_type": row.kind,
                }
            if not shape and _undefined_dtype(dtype) and _undefined_layout(layout):
                continue
            missing = [field for field, value in (("dtype", dtype), ("format", layout))
                       if value in _MISSING or not str(value or "").strip()]
            if missing:
                return {
                    "category": "capture_metadata_missing", "reason": "missing_tensor_metadata",
                    "detail": "Only explicitly undefined empty slots may omit dtype and format.",
                    "missing_fields": [f"{direction}[{index}].{field}" for field in missing],
                    "operator_type": row.kind,
                }
    return None


def native_tensor_contract(row, *, kind=None):
    """Return full native tensor slots; unknown nonempty dtype names stay literal."""
    if native_contract_issue(row) is not None:
        return None
    sides = []
    for direction in ("input", "output"):
        slots = []
        for shape, dtype, layout in zip(
            getattr(row, f"{direction}_shapes"), getattr(row, f"{direction}_dtypes"),
            getattr(row, f"{direction}_formats"),
        ):
            empty = (not shape and _undefined_dtype(dtype)
                     and _undefined_layout(layout))
            slots.append((tuple(shape), None if empty else DTYPES.get(dtype, dtype),
                          None if empty else layout))
        sides.append(tuple(slots))
    return ("native_tensor_abi", kind or row.kind, *sides)


def native_tensor_diagnostic(row):
    return native_contract_issue(row) or {
        "category": "tensor_contract_available", "reason": "observed_tensor_contract_available",
        "detail": "All captured native tensor slots are preserved; scalar values and dispatch are not.",
        "operator_type": row.kind,
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _normalize_dtype(value):
    """Return (normalized known alias or literal unknown, present)."""
    if not isinstance(value, str):
        return None, False
    text = value.strip()
    if not text or text in _MISSING:
        return None, False
    if text in DTYPES:
        return DTYPES[text], True
    low = text.lower().replace("-", "_")
    if low in DTYPES:
        return DTYPES[low], True
    return text, True


def _slot_inputs(slots, direction):
    """Normalize every original slot without dropping position/presence.

    Records retain the original position, presence flags, raw dtype/layout and
    all known fields.  A missing/invalid field is an issue on that exact slot,
    never a deleted position.  ``present=false`` is an explicit absent slot;
    explicit undefined, typed-empty scalar and missing fields remain distinct.
    """
    records, issues = [], []
    for index, slot in enumerate(slots or ()):
        base = {"position": index}
        if isinstance(slot, Mapping):
            if slot.get("present") is False:
                records.append({
                    **base, "present": False, "shape": None, "dtype": None,
                    "layout": None, "dtype_present": False,
                    "layout_present": False, "quality": "absent", "issues": [],
                })
                continue
            shape_present = "shape" in slot
            shape = slot.get("shape")
            dtype_raw = slot.get("dtype")
            dtype_present = (bool(slot.get("dtype_present"))
                             if "dtype_present" in slot else "dtype" in slot)
            if "layout" in slot:
                layout_raw = slot.get("layout")
                layout_present = (bool(slot.get("layout_present"))
                                  if "layout_present" in slot else True)
            elif "format" in slot:
                layout_raw = slot.get("format")
                layout_present = (bool(slot.get("layout_present"))
                                  if "layout_present" in slot else True)
            else:
                layout_raw, layout_present = None, False
        elif (isinstance(slot, Sequence)
              and not isinstance(slot, (str, bytes)) and len(slot) == 3):
            shape_present = True
            shape, dtype_raw, layout_raw = slot
            dtype_present = layout_present = True
        else:
            records.append({
                **base, "present": True, "shape": None, "dtype": None,
                "layout": None, "dtype_present": False,
                "layout_present": False, "quality": "invalid",
                "issues": [{"reason": "tensor_slot_not_a_descriptor"}],
            })
            issues.append({"slot": f"{direction}[{index}]",
                           "reason": "tensor_slot_not_a_descriptor"})
            continue
        slot_issues = []
        if not shape_present:
            slot_issues.append({"reason": "missing_tensor_shape"})
            shape = None
        if not isinstance(shape, (list, tuple)) or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in shape
        ):
            slot_issues.append({"reason": "unknown_tensor_dimensions"})
            shape = list(shape) if isinstance(shape, (list, tuple)) else None
        else:
            shape = [int(value) for value in shape]
        dtype, dtype_known = _normalize_dtype(dtype_raw)
        if not dtype_present:
            slot_issues.append({"reason": "missing_dtype"})
        if not layout_present:
            slot_issues.append({"reason": "missing_layout"})
        if not slot_issues:
            empty = not shape
            undefined_dtype = _undefined_dtype(dtype_raw)
            undefined_layout = _undefined_layout(layout_raw)
            if empty and undefined_dtype and undefined_layout:
                quality = "undefined"
            elif empty:
                if dtype is None or not isinstance(layout_raw, str) or not layout_raw.strip():
                    slot_issues.append({"reason": "missing_dtype_or_layout"})
                    quality = "missing"
                else:
                    quality = "typed_scalar"
            else:
                if dtype is None:
                    slot_issues.append({"reason": "missing_dtype"})
                if not isinstance(layout_raw, str) or not layout_raw.strip() or layout_raw in _MISSING:
                    slot_issues.append({"reason": "missing_layout"})
                quality = "tensor" if not slot_issues else "missing"
        else:
            quality = "missing"
        record = {
            **base, "present": True, "shape": shape,
            "dtype": dtype if dtype_present else None,
            "layout": layout_raw if layout_present else None,
            "dtype_present": dtype_present, "layout_present": layout_present,
            "dtype_raw": dtype_raw, "layout_raw": layout_raw,
            "quality": quality, "issues": slot_issues,
        }
        records.append(record)
        for issue in slot_issues:
            issues.append({"slot": f"{direction}[{index}]",
                           "reason": issue["reason"]})
    return records, issues


def _canonical_slot(record):
    if record.get("quality") == "absent":
        return {"present": False}
    if record.get("quality") == "undefined":
        # Raw DT_UNDEFINED/NULL spellings stay in dtype_raw/layout_raw; the
        # semantic identity is the same explicit undefined slot.
        return {"shape": [], "dtype": None, "layout": None}
    return {
        "shape": list(record.get("shape") or []),
        "dtype": record.get("dtype"),
        "layout": record.get("layout"),
    }


def native_abi_from_slots(native_operator, inputs, outputs, *, strict_operator=True):
    """Build one lossless, idempotent ABI qualification from ordered slots.

    Every supplied slot position/presence is retained even on failure.  Only a
    complete qualification receives a digest; passing an already-qualified
    complete record back through this function reproduces the same canonical
    slots/digest.  Passing an incomplete record back cannot delete its gap or
    upgrade it to complete.  Known dtype aliases are normalized centrally;
    unknown names stay literal.
    """
    operator = str(native_operator or "").strip()
    in_slots, in_issues = _slot_inputs(inputs or (), "inputs")
    out_slots, out_issues = _slot_inputs(outputs or (), "outputs")
    issues = [*in_issues, *out_issues]
    if not operator:
        issues.insert(0, {"slot": "native_operator",
                          "reason": "missing_native_operator"})
    if not in_slots and not in_issues:
        issues.append({"slot": "inputs", "reason": "missing_native_input_slots"})
    if not out_slots and not out_issues:
        issues.append({"slot": "outputs", "reason": "missing_native_output_slots"})
    if issues:
        return {
            "schema": ABI_SCHEMA,
            "status": "incomplete" if strict_operator else "candidate",
            "native_operator": operator or None,
            "inputs": in_slots, "outputs": out_slots,
            "slot_count": {"inputs": len(in_slots), "outputs": len(out_slots)},
            "issues": issues, "digest": None, "verified": False,
            "declared_digest": None, "computed_digest": None,
        }
    payload = {
        "schema": ABI_SCHEMA, "native_operator": operator,
        "inputs": [_canonical_slot(record) for record in in_slots],
        "outputs": [_canonical_slot(record) for record in out_slots],
    }
    digest = hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()
    return {
        **payload, "status": "complete", "inputs": in_slots,
        "outputs": out_slots,
        "slot_count": {"inputs": len(in_slots), "outputs": len(out_slots)},
        "digest": f"sha256:{digest}", "verified": True,
        "declared_digest": None, "computed_digest": f"sha256:{digest}",
    }


def native_abi_from_contract(contract, *, strict_operator=True):
    """Build one native ABI from a declared ordered tensor contract.

    Captured cohort contracts use ``["native_tensor_abi", operator, inputs,
    outputs]`` where each slot is ``[shape, dtype, layout]``.  Those slots are
    the same ordered positions ``native_abi_from_slots`` already qualifies;
    this adapter only establishes the direction boundary.  A structural family
    contract (``["grouped_matrix", ...]`` or ``["slice_copy", ...]``) is not a
    native ABI and returns ``None`` instead of being promoted to one.
    """
    if not isinstance(contract, (list, tuple)) or len(contract) != 4:
        return None
    if str(contract[0] or "") != "native_tensor_abi":
        return None
    operator = contract[1]
    inputs, outputs = contract[2], contract[3]
    if not isinstance(operator, str) or not operator.strip():
        return None
    if not isinstance(inputs, (list, tuple)) or not isinstance(
            outputs, (list, tuple)):
        return None
    return native_abi_from_slots(operator, inputs, outputs,
                                 strict_operator=strict_operator)


def _row_slots(row, direction):
    shapes = getattr(row, f"{direction}_shapes", None)
    dtypes = getattr(row, f"{direction}_dtypes", None)
    formats = getattr(row, f"{direction}_formats", None)
    if shapes is None:
        return None
    count = max(len(shapes or ()), len(dtypes or ()), len(formats or ()))
    slots = []
    for index in range(count):
        dtype = dtypes[index] if dtypes is not None and index < len(dtypes) else None
        layout = formats[index] if formats is not None and index < len(formats) else None
        slots.append({
            "shape": shapes[index] if index < len(shapes) else None,
            "dtype": dtype,
            "layout": layout,
            "dtype_present": (dtypes is not None and index < len(dtypes)
                              and dtype is not None),
            "layout_present": (formats is not None and index < len(formats)
                               and layout is not None),
        })
    return slots


def native_abi_record(row, *, kind=None):
    """Return observed slots preserving every raw position, or a concrete gap.

    Local tensor gaps keep all other known slots; they are never collapsed to
    ``inputs=None``.  The original canonical payload for a fully known record
    remains unchanged so previously retained digests stay stable.
    """
    operator = kind if kind is not None else getattr(row, "kind", None)
    inputs = _row_slots(row, "input")
    outputs = _row_slots(row, "output")
    if inputs is None and outputs is None:
        return {
            "schema": ABI_SCHEMA, "status": "incomplete",
            "native_operator": operator,
            "inputs": [], "outputs": [],
            "slot_count": {"inputs": 0, "outputs": 0},
            "issues": [{"slot": "inputs",
                        "reason": "missing_tensor_metadata"}],
            "digest": None, "verified": False,
            "declared_digest": None, "computed_digest": None,
        }
    abi = native_abi_from_slots(operator, inputs or (), outputs or ())
    issue = native_contract_issue(row)
    if issue is not None:
        extra = {
            "slot": issue.get("missing_fields") or issue.get("direction") or "abi",
            "reason": issue.get("reason", "incomplete_native_tensor_slots"),
        }
        if issue.get("detail"):
            extra["detail"] = issue["detail"]
        issues = [*abi.get("issues", ()), extra]
        return {
            **abi, "status": "incomplete", "issues": issues,
            "digest": None, "verified": False, "computed_digest": None,
        }
    return abi


def native_abi_digest(record):
    """Return a complete verified digest, or ``None`` for gaps/conflicts."""
    if not isinstance(record, Mapping):
        return None
    if record.get("status") != "complete" or not record.get("verified"):
        return None
    digest = record.get("digest")
    return digest if isinstance(digest, str) and digest.startswith("sha256:") else None


def native_abi_identity(record):
    """Return only the operator and content hash used for exact pairing."""
    if not isinstance(record, Mapping):
        return None
    digest = native_abi_digest(record)
    operator = record.get("native_operator")
    if digest is None or not isinstance(operator, str) or not operator:
        return None
    return {"native_operator": operator, "native_abi_sha256": digest}
