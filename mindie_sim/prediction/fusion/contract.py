"""Numerical contracts, exact tile partitioning and Cube resource demand.

[Role] Validate operator-level numerical/dtype/layout/group contracts, partition
  physical storage into exact tiles, compute declared AIC operand demand and
  price the capped Cube tile handoff.
[Invariants] ``sum(tiles) == storage`` exactly; alignment padding is separate.
  ``exact_contract`` is returned only for a proven algebraically identical
  contract. A missing rounding/axis/group fact is a coverage gap (relaxed), not
  a silent exact claim. Unknown Cube capacities are rejected by the caller.
[Timeouts & Locks] Finite shape arithmetic; no I/O, clocks or locks.
[Side-Effect Policy] Pure functions over immutable records.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

from mindie_sim.prediction.fusion.graph import (
    CUBE_KINDS, FeasibilityError, OpNode, STATE_KINDS, TensorValue, TypedGraph,
    _align, _numel, dtype_bytes,
)

NUMERICAL_EXACT = "exact"
_CUBE_TILE_M = 128
_CUBE_TILE_K = 128
_CUBE_TILE_N = 64
_CUBE_ACCUMULATOR_BYTES = 4
#: Declared rounding/saturation vocabularies.  A token outside these sets is a
#: coverage gap, never an assumed default: ``"BOGUS"`` must reject.
ROUNDING_MODES = frozenset({"rne", "rz", "rtz", "rna", "rdn", "rup", "round",
                            "round_half_even", "round_to_nearest_even",
                            "round_toward_zero", "nearest", "rnearest",
                            "trunc", "floor", "ceil", "stochastic",
                            "bf16_after_add_and_mix"})
#: A stochastic rounding decision is not a deterministic algebraic identity;
#: it may be admitted only under an explicit relaxed-numerics policy.
NON_DETERMINISTIC_ROUNDING = frozenset({"stochastic"})
SATURATION_MODES = frozenset({"sat", "saturate", "clamp", "wrap", "none"})


def _validate_rounding_mode(rounding) -> tuple[bool, bool]:
    """Shared vocabulary check: ``(known, non_deterministic)``.

    ``cast``, ``kv_cast``, ``quant`` and ``dynamic_quant`` all consume the same
    rounding vocabulary.  A token outside it is unknown; a stochastic mode is
    known but never an exact algebraic identity.
    """
    if rounding is None:
        return True, False
    text = str(rounding)
    return text in ROUNDING_MODES, text in NON_DETERMINISTIC_ROUNDING


@dataclass(frozen=True)
class CubeResources:
    """Declared AIC capacities; absence is a rejected, never an assumed fit."""
    l1: int = 0
    l0a: int = 0
    l0b: int = 0
    l0c: int = 0

    def record(self) -> dict:
        return {"l1": self.l1, "l0a": self.l0a, "l0b": self.l0b, "l0c": self.l0c}

    @property
    def complete(self) -> bool:
        return all(value > 0 for value in (self.l1, self.l0a, self.l0b, self.l0c))


@dataclass(frozen=True)
class ReductionMerge:
    """Explicit cross-tile reduction/merge program for a split width."""
    merge_bytes: int
    merge_ops: float
    tiles: int = 2
    pool: str = "ub"

    def record(self) -> dict:
        return {"merge_bytes": self.merge_bytes, "merge_ops": self.merge_ops,
                "tiles": self.tiles, "pool": self.pool}


@dataclass(frozen=True)
class ContractCheck:
    reasons: tuple[str, ...]
    numerical_policy: str
    reduction_width: int | None = None
    group_sizes: tuple[int, ...] = ()
    scale_shape: tuple[int, ...] = ()

    @property
    def exact(self) -> bool:
        return not self.reasons and self.numerical_policy == NUMERICAL_EXACT


def local_nbytes(value: TensorValue, tensor_parallel: int = 1) -> int:
    """Local rank bytes; a value already sharded is never divided twice."""
    if tensor_parallel < 1:
        raise FeasibilityError("tensor_parallel_invalid")
    if value.sharded or value.role not in {"weight", "parameter"}:
        return value.nbytes
    return int(math.ceil(value.nbytes / tensor_parallel))


def _rect_bytes(total: int, rows: int, cols: int, r0: int, r1: int,
                c0: int, c1: int) -> int:
    """Exact rectangle partition of ``total`` bytes over a rows x cols value."""
    if rows <= 0 or cols <= 0:
        return 0

    def prefix(r: int, c: int) -> int:
        return (total * r * c) // (rows * cols)

    return prefix(r1, c1) - prefix(r0, c1) - prefix(r1, c0) + prefix(r0, c0)


def tile_bytes(value: TensorValue, rows: int, total_rows: int,
               channel_width: int | None = None, start_row: int = 0) -> int:
    """Exact bytes of one tile along axis 0; tails take the exact remainder.

    A rank->=2 value's row spans ``prod(shape[1:])`` elements.  A rank-1 value
    has no explicit row axis: in a one-row graph (or when its leading extent
    equals ``row_count``) it is one physical row over the feature axis, so
    ``(64,)`` BF16 is ``128 B`` rather than 64 rows of ``2 B``.
    """
    if value.role in {"weight", "state", "parameter"}:
        return max(0, value.nbytes)
    shape = value.shape
    if not shape:
        return 0
    storage = (int(value.storage_bytes) if value.storage_bytes is not None
               else None)
    if len(shape) == 1:
        extent = max(0, int(shape[0]))
        if int(total_rows) > 1 and extent == int(total_rows):
            # A true leading row axis: one scalar value per row.
            row_total, total_cols = int(total_rows), 1
        else:
            # One implicit row spanning the feature axis; values are shared by
            # every graph row rather than silently reinterpreted as 2 B rows.
            row_total, total_cols = 1, extent
        cols = total_cols if channel_width is None else max(
            0, min(int(channel_width), total_cols))
        start = max(0, min(int(start_row), row_total))
        take = max(0, min(int(rows), row_total - start))
        if storage is None:
            return take * cols * dtype_bytes(value.dtype)
        return _rect_bytes(storage, row_total, total_cols, start, start + take,
                           0, cols)
    width = int(shape[-1]) if len(shape) > 1 else 0
    if channel_width is not None and width:
        width = max(0, min(int(channel_width), width))
    row_total = int(total_rows) if shape[0] == total_rows else int(shape[0])
    start = max(0, min(int(start_row), row_total))
    rows = max(0, min(int(rows), row_total - start))
    cols = width
    total_cols = int(shape[-1])
    if storage is None:
        inner = _numel(shape[1:-1]) if len(shape) > 2 else 1
        return rows * inner * cols * dtype_bytes(value.dtype)
    return _rect_bytes(storage, row_total, total_cols, start,
                       start + rows, 0, cols)


def tile_padding(value: TensorValue, rows: int, total_rows: int,
                 start_row: int = 0) -> int:
    raw = tile_bytes(value, rows, total_rows, start_row=start_row)
    return _align(raw, value.alignment_bytes) - raw


def _row_geometry(value: TensorValue, total_rows: int
                  ) -> tuple[int, int, int]:
    """``(row_total, total_cols, inner_per_row)`` for a value's row axis."""
    shape = value.shape
    if len(shape) == 1:
        extent = max(0, int(shape[0]))
        if int(total_rows) > 1 and extent == int(total_rows):
            return int(total_rows), 1, 1
        return 1, extent, 1
    row_total = int(total_rows) if shape[0] == total_rows else int(shape[0])
    inner = _numel(shape[1:-1]) if len(shape) > 2 else 1
    return row_total, int(shape[-1]), inner


def max_tile_bytes(value: TensorValue, rows: int, total_rows: int,
                   channel_width: int | None = None) -> int:
    """Maximum exact bytes of any ``rows``-row block; never vary by start.

    The balanced byte partition makes any ``h``-row block either
    ``floor(h*S/R)`` or ``ceil(h*S/R)`` bytes, so the worst block is available
    in closed form.  This replaces per-start tuple expansion for huge graphs.
    """
    if value.role in {"weight", "state", "parameter"}:
        return max(0, value.nbytes)
    if not value.shape:
        return 0
    row_total, total_cols, inner = _row_geometry(value, total_rows)
    take = max(0, min(int(rows), row_total))
    width = total_cols if channel_width is None else max(
        0, min(int(channel_width), total_cols))
    if value.storage_bytes is None:
        return take * inner * width * dtype_bytes(value.dtype)
    if row_total <= 0 or total_cols <= 0:
        return 0
    numerator = int(value.storage_bytes) * width
    denominator = row_total * total_cols
    return (numerator * take + denominator - 1) // denominator


def tile_partition_padding(value: TensorValue, tile_rows: int,
                           total_rows: int,
                           channel_width: int | None = None) -> int:
    """Total alignment padding over every tile, computed in O(1)."""
    if not value.shape:
        return 0
    row_total, total_cols, inner = _row_geometry(value, total_rows)
    tile = max(1, min(int(tile_rows), row_total or 1))
    starts = int(math.ceil(row_total / tile)) if row_total else 1
    if value.role in {"weight", "state", "parameter"}:
        raw = max(0, value.nbytes)
        return starts * (_align(raw, value.alignment_bytes) - raw)
    width = total_cols if channel_width is None else max(
        0, min(int(channel_width), total_cols))
    blocks, tail_rows = divmod(row_total, tile)
    if value.storage_bytes is None:
        per_row = inner * width * dtype_bytes(value.dtype)
        full_raw = blocks * tile * per_row
        tail_raw = tail_rows * per_row
        total_raw = full_raw + tail_raw
        full_q = tile * per_row
        return (blocks * (_align(full_q, value.alignment_bytes) - full_q)
                + _align(tail_raw, value.alignment_bytes) - tail_raw)
    numerator = int(value.storage_bytes) * width
    denominator = row_total * total_cols
    total_raw = (numerator * row_total) // denominator
    full_rows = blocks * tile
    full_raw = (numerator * full_rows) // denominator
    tail_raw = total_raw - full_raw
    q = (numerator * tile) // denominator
    high = full_raw - q * blocks
    aligned = ((blocks - high) * _align(q, value.alignment_bytes)
               + high * _align(q + 1, value.alignment_bytes))
    if tail_rows:
        aligned += _align(tail_raw, value.alignment_bytes)
    return aligned - total_raw




def _fail(bad: list[str], reason: str) -> None:
    if reason not in bad:
        bad.append(reason)


def _check_quant(op: OpNode, values, outs, bad: list[str]) -> ContractCheck:
    out = outs[0]
    axis = op.param("axis")
    group = op.param("group_size")
    rounding = op.param("rounding")
    saturation = op.param("saturation")
    rank = len(values[0].shape) if values else 0
    integer = isinstance(axis, int) and not isinstance(axis, bool)
    normalized: int | None = None
    if not integer or rank == 0 or axis < -rank or axis >= rank:
        _fail(bad, "quant_axis_out_of_range")
    elif integer:
        # A negative axis is the same physical dimension as ``axis + rank``;
        # normalize before matching the reduced scale dimension.
        normalized = int(axis) + rank if axis < 0 else int(axis)
    if not isinstance(group, int) or isinstance(group, bool) or group < 1:
        _fail(bad, "quant_group_size_invalid")
    rounding_known, nondeterministic = _validate_rounding_mode(rounding)
    if not rounding_known:
        _fail(bad, "quant_rounding_unknown")
    if saturation is not None and str(saturation) not in SATURATION_MODES:
        _fail(bad, "quant_saturation_unknown")
    groups = 0
    if normalized is not None and isinstance(group, int) \
            and not isinstance(group, bool) and group >= 1:
        reduced = values[0].shape[normalized]
        groups = int(math.ceil(max(0, reduced) / group))
        if groups < 1:
            _fail(bad, "quant_group_count_zero")
        if outs[1].role != "scale":
            _fail(bad, "quant_scale_output_missing")
        # The scale tensor must be compatible with the operand and group size:
        # every preserved dimension matches (or broadcasts) and the reduced
        # axis is either one scale per group, one, or the full reduced extent.
        scale_shape = outs[1].shape
        operand = values[0].shape
        if len(scale_shape) == len(operand):
            for index, (declared, extent) in enumerate(zip(scale_shape, operand)):
                if index == normalized:
                    if declared not in {groups, 1}:
                        _fail(bad, "quant_scale_shape_mismatch")
                elif declared not in {extent, 1}:
                    _fail(bad, "quant_scale_shape_mismatch")
        elif len(scale_shape) == len(operand) - 1:
            expected = tuple(dim for index, dim in enumerate(operand)
                             if index != normalized)
            if tuple(scale_shape) != expected:
                _fail(bad, "quant_scale_shape_mismatch")
        elif len(scale_shape) == 1:
            if scale_shape[0] not in {groups, int(operand[normalized])}:
                _fail(bad, "quant_scale_shape_mismatch")
        else:
            _fail(bad, "quant_scale_shape_mismatch")
    if out.dtype not in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2", "int4", "fp4"}:
        _fail(bad, "quant_payload_dtype_invalid")
    rounding_known = rounding is not None and saturation is not None
    policy = NUMERICAL_EXACT if op.numerical_policy == NUMERICAL_EXACT \
        and rounding_known and not bad and not nondeterministic else "relaxed"
    width = int(values[0].shape[normalized]) if normalized is not None else None
    return ContractCheck(tuple(bad), policy, width, (), outs[1].shape)


@dataclass(frozen=True)
class HandoffSplit:
    """Directional on-chip transfer bytes at a Cube boundary.

    ``operand`` is the Vector-to-Cube (MTE1/L1) side; ``output`` is the
    Cube-to-Vector (FixPipe/UB) side.  They belong to different causal
    boundaries and may never be collapsed into one unordered phase.
    """
    operand: int = 0
    output: int = 0

    @property
    def total(self) -> int:
        return int(self.operand) + int(self.output)

    def record(self) -> dict:
        return {"operand": int(self.operand), "output": int(self.output),
                "total": self.total}


def all_tile_bytes(value: TensorValue, tile_rows: int,
                   total_rows: int) -> int:
    """Exact bytes of every tile in one full row sweep, not just tile zero.

    A per-tile epilogue handoff transfers the entire retained output storage.
    Charging one tile (the R06 defect) undercounts a BF16 ``M=16`` output with
    ``tile_rows=1`` from ``32 B`` to ``2 B``.  The balanced byte partition makes
    ``full_tiles * full_tile_bytes + tail_tile_bytes`` exactly the value's
    local width bytes, computed in closed form.
    """
    if value.role in {"weight", "state", "parameter"}:
        return max(0, value.nbytes)
    if not value.shape:
        return 0
    row_total, total_cols, inner = _row_geometry(value, total_rows)
    if row_total <= 0 or total_cols <= 0:
        return 0
    tile = max(1, min(int(tile_rows), row_total))
    full_tiles, tail_rows = divmod(row_total, tile)
    full_rows = full_tiles * tile
    if value.storage_bytes is None:
        per_row = inner * total_cols * dtype_bytes(value.dtype)
        return full_rows * per_row + tail_rows * per_row
    numerator = int(value.storage_bytes) * total_cols
    denominator = row_total * total_cols
    total_bytes = numerator // total_cols
    full_bytes = (int(value.storage_bytes) * full_rows * total_cols) // denominator
    return int(full_bytes + (total_bytes - full_bytes))


def handoff_split(graph: TypedGraph, chosen: tuple[str, ...],
                  cube_resources: CubeResources | None,
                  tile_rows: int) -> HandoffSplit:
    """Priced directional tile handoff for every Cube boundary in the region."""
    operand = output = 0
    if cube_resources is None or not cube_resources.complete:
        return HandoffSplit(0, 0)
    for op_id in chosen:
        op = graph.ops[op_id]
        if op.resource != "cube":
            continue
        values = [graph.values[v] for v in op.inputs]
        if len(values) < 2 or len(values[0].shape) < 2 or len(values[1].shape) < 2:
            continue
        left, right = values[0], values[1]
        m_total, k_total, n_total = (int(left.shape[0]), int(left.shape[-1]),
                                     int(right.shape[-1]))
        m = max(1, min(m_total, _CUBE_TILE_M))
        k = max(1, min(k_total, _CUBE_TILE_K))
        n = max(1, min(n_total, _CUBE_TILE_N))
        tiles = (int(math.ceil(max(1, m_total) / m))
                 * int(math.ceil(max(1, k_total) / k))
                 * int(math.ceil(max(1, n_total) / n)))
        elem_a = dtype_bytes(left.dtype)
        elem_b = dtype_bytes(right.dtype)
        operand += tiles * (m * k * elem_a + k * n * elem_b)
        for value_id in op.outputs:
            if any(consumer in chosen and graph.ops[consumer].resource == "vector"
                   for consumer in graph.consumers(value_id)):
                output += all_tile_bytes(graph.values[value_id], tile_rows,
                                         graph.rows)
    return HandoffSplit(int(operand), int(output))


def handoff_bytes(graph: TypedGraph, chosen: tuple[str, ...],
                  cube_resources: CubeResources | None,
                  tile_rows: int) -> int:
    """Priced on-chip tile handoff for every Cube boundary in the region.

    Capped operand tiles are legal only together with the transfer program
    that moves each real tile through on-chip bandwidth; the tile count is
    therefore part of the charged work, never an unreported approximation.
    """
    return handoff_split(graph, chosen, cube_resources, tile_rows).total

def _contract(graph: TypedGraph, op: OpNode) -> ContractCheck:
    values = [graph.values[v] for v in op.inputs]
    outs = [graph.values[v] for v in op.outputs]
    bad: list[str] = []
    relaxed = False
    kind = op.kind
    reduction_width = None
    group_sizes: tuple[int, ...] = ()
    if not outs:
        return ContractCheck(("shape_contract_missing_output",), "relaxed")
    if any(value.layout != "contiguous" for value in (*values, *outs)):
        _fail(bad, "layout_contract_unknown")
    if kind in {"add", "mul", "bias"}:
        if not values or outs[0].shape != values[0].shape or any(
                v.shape != values[0].shape for v in values[1:]):
            _fail(bad, "shape_or_layout_mismatch")
    elif kind == "elementwise":
        # ``elementwise`` also covers declared view/broadcast operators.  The
        # declared output shape is authoritative, but it must be physically
        # reachable from the inputs: an output larger than every input with no
        # equal-shape source is a fabricated declaration and rejects.
        out_shape = outs[0].shape
        out_numel = _numel(out_shape) if out_shape else 0
        if not values or out_numel <= 0 or any(_numel(v.shape) <= 0 for v in values):
            _fail(bad, "shape_or_layout_mismatch")
        elif not any(tuple(v.shape) == tuple(out_shape) for v in values) \
                and all(_numel(v.shape) < out_numel for v in values):
            _fail(bad, "shape_or_layout_mismatch")
    elif kind == "rms_norm":
        if len(values) < 2 or outs[0].shape != values[0].shape or len(
                values[1].shape) != 1 or values[1].shape[0] != values[0].shape[-1]:
            _fail(bad, "shape_or_layout_mismatch")
        epsilon = op.param("epsilon")
        if not isinstance(epsilon, (int, float)) or not math.isfinite(epsilon) \
                or epsilon <= 0:
            _fail(bad, "numerical_contract_unknown")
        reduction_width = int(values[0].shape[-1]) if values else None
        if op.param("reduction_order") is None:
            _fail(bad, "reduction_order_unknown")
    elif kind == "cast":
        if not values or outs[0].shape != values[0].shape:
            _fail(bad, "shape_or_layout_mismatch")
        rounding = op.param("rounding")
        if not op.param("target_dtype"):
            _fail(bad, "numerical_contract_unknown")
        elif op.param("target_dtype") != outs[0].dtype:
            _fail(bad, "cast_target_dtype_mismatch")
        rounding_known, nondeterministic = _validate_rounding_mode(rounding)
        if not rounding_known:
            # Vocabulary, not mere presence: ``rounding="BOGUS"`` is not a
            # declared rounding decision and can never fuse as exact.
            _fail(bad, "cast_rounding_unknown")
        if rounding is None and values and outs[0].dtype != values[0].dtype:
            relaxed = True  # coverage gap, reported as relaxed and never exact
        elif nondeterministic:
            # Stochastic rounding is not reproducible; it may only be admitted
            # under the caller's explicit relaxed policy.
            relaxed = True
    elif kind in {"dynamic_quant", "quant"}:
        if len(outs) < 2 or outs[0].shape != values[0].shape:
            _fail(bad, "shape_or_layout_mismatch")
        else:
            checked = _check_quant(op, values, outs, bad)
            reduction_width = checked.reduction_width
            relaxed = checked.numerical_policy != NUMERICAL_EXACT
    elif kind == "softmax":
        if not values or outs[0].shape != values[0].shape:
            _fail(bad, "shape_or_layout_mismatch")
        reduction_width = int(values[0].shape[-1]) if values else None
        if op.param("reduction_order") is None:
            _fail(bad, "reduction_order_unknown")
    elif kind == "rope":
        if not values or outs[0].shape != values[0].shape:
            _fail(bad, "shape_or_layout_mismatch")
        if op.param("rotate_style") is None:
            _fail(bad, "numerical_contract_unknown")
    elif kind == "kv_cast":
        if not values or outs[0].shape != values[0].shape or outs[0].dtype == values[0].dtype:
            _fail(bad, "numerical_contract_unknown")
        rounding = op.param("rounding")
        rounding_known, nondeterministic = _validate_rounding_mode(rounding)
        if not rounding_known:
            _fail(bad, "kv_cast_rounding_unknown")
        if rounding is None:
            relaxed = True  # coverage gap, reported as relaxed
        elif nondeterministic:
            relaxed = True
    elif kind == "swiglu":
        if not values or outs[0].shape[:-1] != values[0].shape[:-1] or (
                outs[0].shape[-1] * 2 != values[0].shape[-1]):
            _fail(bad, "shape_or_layout_mismatch")
    elif kind == "grouped_matmul" or kind in CUBE_KINDS:
        group_sizes = _check_grouped_and_cube_contracts(op, kind, values, outs, bad)
    elif kind not in STATE_KINDS:
        _fail(bad, "unknown_shape_contract")
    policy = NUMERICAL_EXACT if op.numerical_policy == NUMERICAL_EXACT \
        and not bad and not relaxed else "relaxed"
    return ContractCheck(tuple(bad), policy, reduction_width, group_sizes)


_CONTRACT_FLOAT_DTYPES = frozenset({"bf16", "bfloat16", "fp16", "float16", "fp32", "float32"})
_CONTRACT_QUANT_DTYPES = frozenset({
    "int8", "int4", "w8a8", "w4a8", "fp8", "fp8_e4m3", "fp8_e5m2",
    "mxfp8", "mxfp4", "fp4", "fp4_e2m1",
})


def _validate_matrix_precision_ports(
    op: OpNode, values: list[TensorValue], outs: list[TensorValue], bad: list[str],
) -> None:
    if len(values) < 2 or not outs:
        return
    act_param = str(op.param("activation_format") or "").strip().lower()
    wt_param = str(op.param("weight_format") or "").strip().lower()
    out_param = str(op.param("output_format") or "").strip().lower()
    val0_dt = str(values[0].dtype or "").strip().lower()
    val1_dt = str(values[1].dtype or "").strip().lower()
    out_dt = str(outs[0].dtype or "").strip().lower()
    for declared_dt, port_dt in ((act_param, val0_dt), (wt_param, val1_dt), (out_param, out_dt)):
        if declared_dt and port_dt:
            if (declared_dt in _CONTRACT_FLOAT_DTYPES) != (port_dt in _CONTRACT_FLOAT_DTYPES):
                if declared_dt in _CONTRACT_QUANT_DTYPES or port_dt in _CONTRACT_QUANT_DTYPES:
                    _fail(bad, "precision_abi_mismatch")
    if act_param in _CONTRACT_FLOAT_DTYPES and wt_param in _CONTRACT_FLOAT_DTYPES:
        if op.param("activation_scale_format") and str(op.param("linear_method") or "") in {"", "none", "dense"}:
            _fail(bad, "precision_abi_mismatch")


def _check_grouped_and_cube_contracts(
    op: OpNode, kind: str, values: list[TensorValue], outs: list[TensorValue], bad: list[str],
) -> tuple[int, ...] | None:
    group_sizes: tuple[int, ...] | None = None
    if kind == "grouped_matmul":
        rows = op.param("group_sizes")
        group_list = op.param("group_list")
        if rows is None or group_list is None:
            _fail(bad, "grouped_matmul_group_contract_missing")
        else:
            try:
                group_sizes = tuple(int(item) for item in rows)
            except (TypeError, ValueError):
                group_sizes = ()
                _fail(bad, "grouped_matmul_group_contract_invalid")
            if not group_sizes or any(item < 0 for item in group_sizes):
                _fail(bad, "grouped_matmul_group_contract_invalid")
            if len(values) < 2 or len(values[0].shape) != 2 or len(values[1].shape) != 2:
                _fail(bad, "shape_or_layout_mismatch")
            elif sum(group_sizes) not in {0, values[0].shape[0]} and values[0].shape[0] > 1:
                _fail(bad, "grouped_matmul_group_rows_mismatch")
            if not isinstance(group_list, (tuple, list)) or len(group_sizes) != len(group_list):
                _fail(bad, "grouped_matmul_group_list_mismatch")
        _validate_matrix_precision_ports(op, values, outs, bad)
    elif kind in CUBE_KINDS:
        if len(values) < 2 or any(len(v.shape) != 2 for v in values[:2]) or (
                values[0].shape[-1] != values[1].shape[0]) or (
                outs[0].shape != (values[0].shape[0], values[1].shape[1])):
            _fail(bad, "shape_or_layout_mismatch")
        _validate_matrix_precision_ports(op, values, outs, bad)
    return group_sizes


def _cube_operands(graph: TypedGraph, op: OpNode, tile_rows: int,
                   resources: CubeResources | None) -> dict[str, int]:
    values = [graph.values[v] for v in op.inputs]
    if len(values) < 2 or not resources or not resources.complete:
        return {}
    left, right = values[0], values[1]
    rows = max(1, min(tile_rows, left.shape[0]))
    m = min(rows, _CUBE_TILE_M)
    k = max(1, min(int(left.shape[-1]), _CUBE_TILE_K))
    n = max(1, min(int(right.shape[-1]), _CUBE_TILE_N))
    a = m * k * dtype_bytes(left.dtype)
    b = k * n * dtype_bytes(right.dtype)
    return {"l1": a + b, "l0a": a, "l0b": b, "l0c": m * n * _CUBE_ACCUMULATOR_BYTES}


