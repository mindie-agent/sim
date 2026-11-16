"""Declared fused matrix output transformations.

[Role] Keep row/column scale traffic and output operations after the K reduction.
[Invariants] Scales never become matrix operands or repeat per K panel.
  This is logical vector service; no hidden HBM accumulator spill or native
  Fixpipe/UB dispatch is inferred from the mathematical transformation.
  A native typed ABI declaration is authoritative for per-slot scale storage:
  an INT8 matrix epilogue must not silently price a default FP32 scale/offset
  when the captured call carries a BF16 scale and no offset.
[Timeouts & Locks] Constant metadata arithmetic; no locks or I/O.
[Side-Effect Policy] Returns immutable work descriptions only.
"""
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes


ZERO = 0
ZERO_FLOAT = 0.0
TOLERANCE = 1e-12
SUPPORTED_EPILOGUES = ("row_column_scale",)
TYPED_SCALE_SLOTS = ("activation", "weight", "weight_scale", "activation_scale")


def _normalize_typed_dtype(value) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key.startswith("dt_"):
        key = key[3:]
    return {
        "float": "fp32", "float32": "fp32", "single": "fp32",
        "bfloat16": "bf16", "half": "fp16", "float16": "fp16",
    }.get(key, key)


def _typed_scale_storage(op):
    """Return typed row/column scale storage from a native ABI declaration.

    T12 normalizes the real call into ``metadata.native_call.typed.attrs``.
    For the W8A8 QuantBatchMatmul family the first four input slots are
    activation, weight, weight_scale and activation_scale.  The slot dtypes
    are the captured ABI representation, not a catalog default.  Responses
    with fewer than four typed inputs cannot establish the scale slots and are
    left to the ordinary declarative path.
    """
    if str(getattr(op, "dtype", "")).strip().lower() not in {"int8", "w8a8"}:
        return None
    metadata = dict(op.metadata)
    native = metadata.get("native_call")
    if not isinstance(native, dict):
        return None
    typed = native.get("typed")
    if not isinstance(typed, dict):
        return None
    attrs = typed.get("attrs")
    if not isinstance(attrs, dict):
        return None
    dtypes = attrs.get("input_dtypes")
    if isinstance(dtypes, (str, bytes)) or not isinstance(dtypes, (list, tuple)):
        return None
    if len(dtypes) < len(TYPED_SCALE_SLOTS):
        return None
    offset = attrs.get("offset")
    if offset not in (None, False, 0, "", "none", "not_present", "absent"):
        raise ModelingError("matrix_epilogue_typed_scale_offset_not_supported")
    slots = tuple(_normalize_typed_dtype(value) for value in dtypes[:4])
    try:
        weight_bytes = dtype_bytes(slots[2])
        activation_bytes = dtype_bytes(slots[3])
    except ModelingError as exc:
        raise ModelingError(
            "matrix_epilogue_typed_scale_dtype_unsupported:"
            + ",".join(slots[2:4])) from exc
    m, n, _k = (int(value) for value in op.local_shape[-3:])
    groups = int(metadata.get("matrix_groups", 1) or 1)
    if m < 0 or n < 0 or groups < 1:
        raise ModelingError("matrix_epilogue_typed_scale_geometry_invalid")
    return {
        "row_scale_bytes": float(m * activation_bytes),
        # One independent per-channel scale per declared output group.
        "column_scale_bytes": float(groups * n * weight_bytes),
        "row_shape": (m,), "column_shape": (groups * n,),
        "weight_scale_format": slots[2],
        "activation_scale_format": slots[3],
        "input_dtypes": list(slots),
        "offset": "not_present_in_four_input_abi",
    }


def _declared_epilogue(op):
    """Parse the explicit metadata record only, without typed cross-checks."""
    raw = dict(op.metadata).get("matrix_epilogue")
    if raw is None:
        return None
    if raw.get("kind") not in SUPPORTED_EPILOGUES:
        raise ModelingError("matrix_epilogue_kind_unsupported")
    if set(raw) != {"kind", "row_scale_bytes", "column_scale_bytes",
                    "operations_per_output"}:
        raise ModelingError("matrix_epilogue_requires_explicit_scale_and_operation_work")
    result = MatrixEpilogue(*(float(raw[key]) for key in
                              ("row_scale_bytes", "column_scale_bytes",
                               "operations_per_output")))
    if any(not math.isfinite(value) or value < ZERO_FLOAT for value in
           (result.row_scale_bytes, result.column_scale_bytes,
            result.operations_per_output)):
        raise ModelingError("matrix_epilogue_work_must_be_finite_nonnegative")
    if not result.operations_per_output:
        raise ModelingError("row_column_scale_requires_positive_output_work")
    m, n, _ = op.local_shape[-3:]
    if not math.isclose(op.vector_ops, result.operations_per_output * m * n,
                        rel_tol=TOLERANCE, abs_tol=TOLERANCE):
        # Full graph capture may pad local rows to a bucket while the declared
        # epilogue work still describes the active call.  Accept exactly the
        # declared active/physical row counts rather than crashing; unknown
        # rows remain an explicit error.
        metadata = dict(op.metadata)
        declared = {
            int(metadata.get("graph_active_query_rows", ZERO) or ZERO),
            int(metadata.get("graph_physical_query_rows", ZERO) or ZERO),
        }
        if not any(row > ZERO and math.isclose(
                op.vector_ops, result.operations_per_output * row * n,
                rel_tol=TOLERANCE, abs_tol=TOLERANCE) for row in declared):
            raise ModelingError(
                "matrix_epilogue_must_account_for_declared_vector_work")
    return result


def _shape_matches(metadata, key, expected):
    shape = metadata.get(key)
    if shape is None:
        # A missing runtime shape is not a complete typed representation.
        return False
    try:
        return tuple(shape) == tuple(expected)
    except TypeError:
        return False


def _runtime_scale_matches(metadata, typed, declared):
    """One criterion for both reconciliation and read-only parsing.

    Byte totals alone are insufficient: the row and column runtime dtype,
    bytes and presence are part of the same typed input contract.  Checkpoint
    storage fields (``*_storage_*``) are deliberately not required to equal the
    runtime representation.
    """
    def same(actual, expected):
        try:
            return math.isclose(float(actual), float(expected),
                                rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False

    return (
        same(declared.row_scale_bytes, typed["row_scale_bytes"])
        and same(declared.column_scale_bytes, typed["column_scale_bytes"])
        and _normalize_typed_dtype(metadata.get("activation_scale_format", ""))
        == typed["activation_scale_format"]
        and same(metadata.get("activation_scale_bytes", 0) or 0,
                 typed["row_scale_bytes"])
        and _normalize_typed_dtype(metadata.get("weight_scale_format", ""))
        == typed["weight_scale_format"]
        and same(metadata.get("weight_scale_bytes", 0) or 0,
                 dtype_bytes(typed["weight_scale_format"]))
        and metadata.get("int8_weight_metadata") == "scale_only"
        and _shape_matches(metadata, "activation_scale_shape", typed["row_shape"])
        and _shape_matches(metadata, "weight_scale_shape", typed["column_shape"])
    )


def _producer_scale_storage(producer):
    if producer is None:
        return None
    metadata = dict(getattr(producer, "metadata", {}) or {})
    row_format = metadata.get("activation_scale_format")
    row_shape = metadata.get("activation_scale_shape")
    row_bytes = metadata.get("activation_scale_bytes")
    if row_format is None and row_shape is None and row_bytes is None:
        return None
    return row_format, row_shape, row_bytes


def _check_producer_scale(producer, typed):
    """A bound producer is authoritative for its own output representation."""
    actual = _producer_scale_storage(producer)
    if actual is None:
        return
    row_format, row_shape, row_bytes = actual
    def same(actual_value, expected):
        try:
            return math.isclose(float(actual_value), float(expected),
                                rel_tol=1e-12, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    if (_normalize_typed_dtype(row_format) != typed["activation_scale_format"]
            or not same(row_bytes, typed["row_scale_bytes"])
            or (row_shape is not None and tuple(row_shape) != tuple(typed["row_shape"]))):
        raise ModelingError(
            "matrix_epilogue_producer_scale_conflict:"
            f"producer={row_format}/{row_shape}/{row_bytes}:"
            f"typed={typed['activation_scale_format']}/{typed['row_shape']}/"
            f"{typed['row_scale_bytes']}")


def reconcile_native_scale_storage(op, *, producer=None):
    """Bind a native typed scale ABI to the common matrix epilogue record.

    The catalog's checkpoint defaults (FP32 scale plus offset) remain valid
    for a case that explicitly declares them.  When T12 supplies typed input
    slots, however, the captured runtime representation wins.  This function
    rewrites only the scale/metadata part of the declared weight footprint and
    the two HBM/MTE2 totals that contain it; FLOPs, output, vector work and the
    semantic shape are untouched.  It is idempotent and returns the original
    op when no typed declaration is present.
    """
    declared = _declared_epilogue(op)
    if declared is None:
        return op
    typed = _typed_scale_storage(op)
    if typed is None:
        return op
    _check_producer_scale(producer, typed)
    row = typed["row_scale_bytes"]
    column = typed["column_scale_bytes"]
    metadata = dict(op.metadata)
    if _runtime_scale_matches(metadata, typed, declared):
        return op
    old_activation_format = metadata.get("activation_scale_format")
    old_activation_bytes = metadata.get("activation_scale_bytes")
    packed = float(metadata.get("packed_weight_bytes", 0) or 0)
    if packed <= 0:
        packed = float(metadata.get("weight_bytes", 0) or 0)
    payload = packed - declared.column_scale_bytes
    if payload < 0:
        raise ModelingError(
            "matrix_epilogue_typed_scale_exceeds_packed_weight:"
            f"{declared.column_scale_bytes}>{packed}")
    new_packed = payload + column
    delta = (row - declared.row_scale_bytes) + (column - declared.column_scale_bytes)
    new_hbm = float(op.hbm_bytes) + delta
    new_mte2 = float(op.mte2_bytes) + delta
    if new_hbm < 0 or new_mte2 < 0:
        raise ModelingError("matrix_epilogue_typed_scale_negative_traffic")
    raw = dict(metadata["matrix_epilogue"])
    raw.update({"row_scale_bytes": row, "column_scale_bytes": column})
    metadata.update({
        "matrix_epilogue": raw,
        "weight_scale_format": typed["weight_scale_format"],
        "weight_scale_bytes": float(dtype_bytes(typed["weight_scale_format"])),
        "activation_scale_format": typed["activation_scale_format"],
        "activation_scale_bytes": row,
        "activation_scale_shape": typed["row_shape"],
        "weight_scale_shape": typed["column_shape"],
        "int8_weight_metadata": "scale_only",
        "column_metadata_policy": "declared_scale_only_storage_traffic",
        "packed_weight_bytes": new_packed,
        "weight_bytes": new_packed,
        **({"weight_payload_bytes": payload}
           if "weight_payload_bytes" in metadata else {}),
        "native_scale_reconciliation": {
            "basis": "native_call.typed.attrs.input_dtypes",
            "input_dtypes": typed["input_dtypes"],
            "offset": typed["offset"],
            "old_row_scale_bytes": declared.row_scale_bytes,
            "old_column_scale_bytes": declared.column_scale_bytes,
            "old_activation_scale_format": old_activation_format,
            "old_activation_scale_bytes": old_activation_bytes,
            "new_row_scale_bytes": row,
            "new_column_scale_bytes": column,
            "new_activation_scale_format": typed["activation_scale_format"],
            "packed_payload_bytes": payload,
            "packed_weight_bytes": new_packed,
            "hbm_delta_bytes": delta,
            "source": "T12 normalized typed input slots",
        },
    })
    return replace(op, hbm_bytes=new_hbm, mte2_bytes=new_mte2,
                   metadata=tuple(metadata.items()))



def reconcile_scale_storage(ops):
    """One graph-level normalization for all matrix ops in declared order.

    The public preparation path can call this once after actual producer ops
    are present.  Each matrix op receives the real producer found through its
    declared dependency edge; no caller-supplied scalar and no synthetic plan
    is needed.  Non-matrix ops and ops without a typed ABI pass through.
    """
    ops = tuple(ops)
    by_id = {op.op_id: op for op in ops}
    result = []
    for op in ops:
        producer = None
        for dependency in op.deps:
            candidate = by_id.get(dependency)
            if candidate is not None and _producer_scale_storage(candidate) is not None:
                producer = candidate
                break
        result.append(reconcile_native_scale_storage(op, producer=producer))
    return tuple(result)


def plan_matrix_epilogue(op, *, producer=None):
    """Return the normalized op and its concrete epilogue for an original planner.

    T10's Tiled/Steady planners consume this pair directly.  This is not a
    work-only projection: the returned op carries the actual packed payload,
    runtime scale bytes, legal local/tail shape and previous program identity,
    and the epilogue is the same object used by ``matrix.lower``.
    """
    normalized = reconcile_native_scale_storage(op, producer=producer)
    return normalized, MatrixEpilogue.from_op(normalized, producer=producer)

@dataclass(frozen=True)
class MatrixEpilogue:
    row_scale_bytes: float = 0
    column_scale_bytes: float = 0
    operations_per_output: float = 0

    @classmethod
    def from_op(cls, op, *, producer=None):
        result = _declared_epilogue(op)
        if result is None:
            return cls()
        typed = _typed_scale_storage(op)
        if typed is not None:
            _check_producer_scale(producer, typed)
        if typed is not None and not _runtime_scale_matches(
                dict(op.metadata), typed, result):
            raise ModelingError(
                "matrix_epilogue_typed_scale_representation_mismatch:"
                f"declared={result.row_scale_bytes}/{result.column_scale_bytes}:"
                f"typed={typed['row_scale_bytes']}/{typed['column_scale_bytes']}:"
                f"row={dict(op.metadata).get('activation_scale_format')}/"
                f"{dict(op.metadata).get('activation_scale_bytes')}")
        return result

    def operands(self, op):
        weight = dict(op.metadata).get("packed_weight_bytes", 0)
        if (self.column_scale_bytes > weight
                or weight + self.row_scale_bytes > op.mte2_bytes):
            raise ModelingError("matrix_epilogue_scales_exceed_declared_inputs")
        return (max(0, op.mte2_bytes - weight - self.row_scale_bytes),
                max(0, weight - self.column_scale_bytes))

    def scales(self, a, b, m, n, groups, *, reuse_inputs, reuse_weights):
        row = self.row_scale_bytes * a / m
        column = self.column_scale_bytes * b / (n * groups)
        fraction = a * b / (m * n)
        hbm = ((self.row_scale_bytes * fraction if reuse_inputs else row)
               + (self.column_scale_bytes * fraction if reuse_weights else column))
        return row + column, hbm
