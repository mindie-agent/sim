"""Declared fused matrix output transformations.

[Role] Keep row/column scale traffic and output operations after the K reduction.
[Invariants] Scales never become matrix operands or repeat per K panel.
  This is logical vector service; no hidden HBM accumulator spill or native
  Fixpipe/UB dispatch is inferred from the mathematical transformation.
[Timeouts & Locks] Constant metadata arithmetic; no locks or I/O.
[Side-Effect Policy] Returns immutable work descriptions only.
"""
from dataclasses import dataclass
import math

from mindie_sim.schema.common import ModelingError


SUPPORTED_EPILOGUES = ("row_column_scale",)


@dataclass(frozen=True)
class MatrixEpilogue:
    row_scale_bytes: float = 0
    column_scale_bytes: float = 0
    operations_per_output: float = 0

    @classmethod
    def from_op(cls, op):
        raw = dict(op.metadata).get("matrix_epilogue")
        if raw is None:
            return cls()
        if raw.get("kind") not in SUPPORTED_EPILOGUES:
            raise ModelingError("matrix_epilogue_kind_unsupported")
        if set(raw) != {"kind", "row_scale_bytes", "column_scale_bytes",
                       "operations_per_output"}:
            raise ModelingError("matrix_epilogue_requires_explicit_scale_and_operation_work")
        result = cls(*(float(raw[key]) for key in
                       ("row_scale_bytes", "column_scale_bytes", "operations_per_output")))
        if any(not math.isfinite(value) or value < 0 for value in
               (result.row_scale_bytes, result.column_scale_bytes,
                result.operations_per_output)):
            raise ModelingError("matrix_epilogue_work_must_be_finite_nonnegative")
        if not result.operations_per_output:
            raise ModelingError("row_column_scale_requires_positive_output_work")
        m, n, _ = op.local_shape[-3:]
        if not math.isclose(op.vector_ops, result.operations_per_output * m * n,
                            rel_tol=1e-12, abs_tol=1e-12):
            raise ModelingError("matrix_epilogue_must_account_for_declared_vector_work")
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
