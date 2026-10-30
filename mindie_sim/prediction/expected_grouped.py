"""Keep probabilistic expert banks separate inside an analytical matrix grid.

[Role] Map routed-row and active-bank moments to finite per-bank tile owners.
[Invariants] Original arithmetic, payloads and scale work remain unchanged.
  A total routed-row matrix must never reload the complete bank set per M tile.
  Expected occupancy is not an observed integer histogram or vendor dispatch.
[Timeouts & Locks] Constant metadata work; the selected matrix program is bounded.
[Side-Effect Policy] Returns a local context and ordinary Engine program only.
"""
from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class ExpectedGroupedMatmul:
    matrix: object

    def lower(self, context):
        op = context.op
        meta = dict(op.metadata)
        active = meta["expected_active_experts"]
        rows = meta["expected_rows"]
        _, n, k = op.local_shape[-3:]
        # These are scheduling slots, not a realized routing histogram. A final
        # fractional bank and fractional conditional row are represented by the
        # unchanged expected payload/work over a capacity-rounded owner grid.
        banks = math.ceil(active)
        per_bank = math.ceil(rows / active)
        slots = banks * per_bank
        meta.update(matrix_groups=banks, matrix_mapping_domain="expected_expert_banks")
        if "matrix_epilogue" in meta:
            epilogue = dict(meta["matrix_epilogue"])
            epilogue["operations_per_output"] = op.vector_ops / (slots * n)
            meta["matrix_epilogue"] = epilogue
        shape = (slots, n, k)
        mapped = replace(op, local_shape=shape, launch_shape=shape,
                         padded_shape=tuple(math.ceil(v / 16) * 16 for v in shape),
                         metadata=tuple(meta.items()))
        if context.record_mapping is not None:
            context.record_mapping({
                "execution_abstraction": "expected_expert_bank_grid",
                "expected_rows": rows, "expected_active_experts": active,
                "conditional_mean_rows": rows / active,
                "bank_slots": banks, "rows_per_bank_slot": per_bank,
                "semantic_rows": op.local_shape[-3], "scheduling_rows": slots,
                "work_policy": "preserve_semantic_arithmetic_and_payload_moments",
                "private_service_policy": "derive_owner_copies_and_compute_width_from_rounded_grid",
                "schedule": "capacity_rounded_conditional_mean_not_sampled_histogram",
                "outer_invocations": 1,
            })
        return self.matrix.lower(replace(context, op=mapped))


def adapt(recipe, op):
    from .matrix import HierarchicalMatmul, TiledMatmul
    from .matrix_mapping import MappedMatmul
    meta = dict(op.metadata)
    if (meta.get("matrix_mapping_domain") == "aggregate_experts"
            and meta.get("routing_basis") == "probability_moments"
            and meta.get("expected_active_experts", 0) > 0
            and isinstance(recipe, (TiledMatmul, HierarchicalMatmul, MappedMatmul))):
        return ExpectedGroupedMatmul(recipe)
    return recipe
