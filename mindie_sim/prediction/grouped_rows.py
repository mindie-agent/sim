"""Compose integer expert row classes inside one analytical matrix invocation.

[Role] Reuse equal-group matrix programs for uneven integer expert histograms.
[Invariants] Never pad expert rows or synthesize token IDs. Each active bank is
  read once under the selected reuse policy. One outer invocation contains
  serial row-class phases; this schedule is analytical, not vendor dispatch.
[Timeouts & Locks] O(experts + distinct row classes); no I/O or locks.
[Side-Effect Policy] Returns programs for the caller's ordinary Engine only.
"""
from collections import Counter
from dataclasses import dataclass, replace
import math

from mindie_sim.schema.common import ModelingError


def row_classes(op):
    rows = tuple(dict(op.metadata).get("matrix_group_rows", ()))
    if not rows:
        return ()
    if any(type(row) is not int or row < 0 for row in rows):
        raise ModelingError("matrix_group_rows_require_nonnegative_integers")
    if sum(rows) != op.local_shape[-3]:
        raise ModelingError("matrix_group_rows_must_conserve_declared_rows")
    return tuple(sorted((row, count) for row, count in Counter(rows).items() if row))


def split_context(context, rows_per_expert, experts):
    """Rows distribute activation work; bank counts distribute weight work."""
    from .mechanism import Work
    op = context.op
    total, n, k = op.local_shape[-3:]
    metadata = dict(op.metadata)
    active = sum(count for _, count in row_classes(op))
    rows = rows_per_expert * experts
    row_fraction, bank_fraction = rows / total, experts / active
    weight = metadata["packed_weight_bytes"]
    sub_weight = weight * bank_fraction
    read = (op.mte2_bytes - weight) * row_fraction + sub_weight
    write = op.mte3_bytes * row_fraction
    metadata.update(matrix_groups=experts, matrix_group_rows=(rows_per_expert,) * experts,
                    matrix_mapping_domain="equal_expert_groups", packed_weight_bytes=sub_weight)
    if "weight_bytes" in metadata:
        metadata["weight_bytes"] *= bank_fraction
    if "activation_scale_bytes" in metadata:
        metadata["activation_scale_bytes"] *= row_fraction
        metadata["activation_scale_shape"] = (rows, *metadata["activation_scale_shape"][1:])
    if "matrix_epilogue" in metadata:
        epilogue = dict(metadata["matrix_epilogue"])
        epilogue["row_scale_bytes"] *= row_fraction
        epilogue["column_scale_bytes"] *= bank_fraction
        metadata["matrix_epilogue"] = epilogue
    matrix = {dtype: amount * row_fraction for dtype, amount in
              (op.matrix_flops or ((op.dtype, op.flops),))}
    resources = {role: amount * row_fraction for role, amount in context.native_work.items()
                 if role != "cube"}
    for role, base, part in (("hbm", op.hbm_bytes, read + write),
                             ("mte2", op.mte2_bytes, read),
                             ("mte3", op.mte3_bytes, write)):
        resources[role] = part + (context.native_work.get(role, base) - base) * row_fraction
    phase = context.for_phase(Work(matrix, resources), kernel="integer_expert_row_class")
    # Work.apply canonicalizes extended resource roles. Keep its scaled values,
    # rather than reintroducing the parent invocation's unscaled metadata.
    phase_metadata = dict(phase.op.metadata)
    for key in ("resource_work", "mte1_bytes", "fixpipe_bytes"):
        metadata.pop(key, None)
        if key in phase_metadata:
            metadata[key] = phase_metadata[key]
    shape = (rows, n, k)
    phase_op = replace(
        phase.op, local_shape=shape, semantic_shape=shape, launch_shape=shape,
        padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
        metadata=tuple(metadata.items()))
    return replace(phase, op=phase_op)


@dataclass(frozen=True)
class IntegerGroupedMatmul:
    matrix: object

    def lower(self, context):
        from .program import KernelProgram, Stage
        classes = row_classes(context.op)
        stages = [Stage("setup", latency=context.setup_ms)]
        for rows, experts in classes:
            phase = split_context(context, rows, experts)
            stages.append(Stage(
                f"rows{rows}", after=(stages[-1].name,), program=self.matrix.lower(phase)))
        if context.record_mapping is not None:
            context.record_mapping({
                "execution_abstraction": "integer_expert_row_classes",
                "row_classes": classes,
                "schedule": "serial_row_classes_with_selected_matrix_owner_schedule",
                "timing_basis": "declared_analytical_schedule_not_vendor_dispatch",
                "outer_invocations": 1,
            })
        return KernelProgram(tuple(stages), minimum_ms=context.minimum_ms)


def adapt(recipe, op):
    classes = row_classes(op)
    if len(classes) <= 1:
        return recipe
    from .matrix import HierarchicalMatmul, TiledMatmul
    from .matrix_mapping import MappedMatmul
    if isinstance(recipe, (MappedMatmul, HierarchicalMatmul, TiledMatmul)):
        return IntegerGroupedMatmul(recipe)
    # Other explicit analytical or source planners retain their own program.
    return recipe
