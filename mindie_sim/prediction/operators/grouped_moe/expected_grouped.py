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

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
    is_arch35_context,
    primitive_l2_params,
)

ROW_TILE = 128
FRACTAL = 16


def _apply_grouped_weight_cache_policy(
    context, meta, *, active_rows, declared_groups, rows, n, k, dtype,
):
    """Apply upstream CANN GroupedMatmul operand-scoped L2 weight cache policy.

    - DAV_2201 (grouped_matmul_kernel.h:566): sets CACHE_MODE_DISABLE on
      weightGmLocal for every expert with blockDimM == 1 (rows_e <= baseM=128).
    - DAV_3510 (arch35/grouped_no_quant_matmul_tiling.cpp:193): sets
      weightNoL2Cache_ = (m*k + groupNum*k*n + m*n) * elem_bytes >= l2_bytes
      using declared total groupNum = len(group_list).
    """
    if getattr(context, "primitive_table", None) is None:
        return
    admitted = max(0.0, float(meta.get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
    if admitted <= 0.0:
        return
    elem_b = dtype_bytes(meta.get("compute_format", dtype or "bf16"))
    x_bytes = float(rows * k * elem_b)
    l2_bytes, _ = primitive_l2_params(context)
    if is_arch35_context(context):
        total_size = (rows * k + declared_groups * k * n + rows * n) * elem_b
        if l2_bytes > 0 and total_size >= l2_bytes:
            meta["admitted_l2_read_hit_bytes"] = min(admitted, x_bytes)
            meta["l2_weight_cache_disable"] = True
        return
    cacheable_experts = sum(1 for r in active_rows if r > ROW_TILE)
    cacheable_weight_bytes = float(cacheable_experts * k * n * elem_b)
    meta["admitted_l2_read_hit_bytes"] = min(admitted, x_bytes + cacheable_weight_bytes)
    if cacheable_experts == 0:
        meta["l2_weight_cache_disable"] = True


def inclusion_probability(probabilities, *, tokens):
    """Exact ``1 - (1-p)^tokens`` with stable endpoint handling."""
    result = []
    for value in probabilities:
        p = min(1.0, max(0.0, float(value)))
        if p <= 0.0:
            result.append(0.0)
        elif p >= 1:
            result.append(1.0)
        else:
            result.append(-math.expm1(float(tokens) * math.log1p(-p)))
    return tuple(result)


def touched_weight_bytes(probabilities, *, tokens, weight_bytes, scale_bytes=0):
    """Expected cold weight bytes over experts that receive at least one row."""
    touched = inclusion_probability(probabilities, tokens=tokens)
    return math.fsum(a * (float(weight_bytes) + float(scale_bytes))
                     for a in touched), sum(1 for a in touched if a > 0.0)


def row_tile_moments(probabilities, *, tokens, row_tile=ROW_TILE):
    """Expected row-tile count and occupied-row moment per expert."""
    result = []
    for value in probabilities:
        p = min(1.0, max(0.0, float(value)))
        tile_probability = []
        occupied = 0.0
        for start in range(0, int(tokens), int(row_tile)):
            stop = min(int(tokens), start + int(row_tile))
            tile_probability.append(
                1 - (1 - p) ** stop if p < 1 else 1)
            for row in range(start + 1, stop + 1):
                occupied += (1 - (1 - p) ** row) if p < 1 else 1
        result.append((tuple(tile_probability), occupied))
    return tuple(result)


@dataclass(frozen=True)
class ExpectedGroupedMatmul:
    matrix: object

    def lower(self, context):
        op = context.op
        meta = dict(op.metadata)
        _, n, k = op.local_shape[(-3):]
        explicit_hist = meta.get("expert_rows") or meta.get("rows_by_expert")
        aclnn_hist = meta.get("aclnn_group_list")
        if explicit_hist:
            group_rows = tuple(max(0, int(value)) for value in explicit_hist)
            active = sum(1 for value in group_rows if value > 0)
            rows = sum(group_rows)
            banks = len(group_rows)
            per_bank = max(group_rows, default=0)
            slots = max(1, rows)
        elif aclnn_hist:
            full_rows = tuple(max(0, int(value)) for value in aclnn_hist)
            active_rows = tuple(value for value in full_rows if value > 0)
            declared_groups = len(full_rows)
            active = len(active_rows)
            rows = sum(active_rows)
            group_rows = active_rows if active > 0 else (0,)
            banks = max(1, active)
            per_bank = max(group_rows, default=0)
            slots = max(1, rows)
            meta["declared_group_count"] = declared_groups
            _apply_grouped_weight_cache_policy(
                context, meta, active_rows=active_rows,
                declared_groups=declared_groups, rows=rows, n=n, k=k,
                dtype=op.dtype,
            )
        else:
            active = meta["expected_active_experts"]
            rows = meta["expected_rows"]
            group_rows = None
            # These are scheduling slots, not a realized routing histogram. A
            # final fractional bank and fractional conditional row are
            # represented by the unchanged expected work over a rounded grid.
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
                         padded_shape=tuple(math.ceil(v / FRACTAL) * FRACTAL for v in shape),
                         metadata=tuple(meta.items()))
        if context.record_mapping is not None:
            context.record_mapping({
                "execution_abstraction": "expected_expert_bank_grid",
                "expected_rows": rows, "expected_active_experts": active,
                "conditional_mean_rows": (rows / active) if active else 0.0,
                "bank_slots": banks, "rows_per_bank_slot": per_bank,
                "semantic_rows": op.local_shape[(-3)], "scheduling_rows": slots,
                "work_policy": "preserve_semantic_arithmetic_and_payload_moments",
                "private_service_policy": "derive_owner_copies_and_compute_width_from_rounded_grid",
                "schedule": ("exact_histogram_owner_tail" if group_rows is not None
                             else "capacity_rounded_conditional_mean_not_sampled_histogram"),
                "outer_invocations": 1,
            })
        mapped_context = replace(context, op=mapped)
        if group_rows is not None:
            from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
            if isinstance(self.matrix, SteadyMatmul):
                return self.matrix.lower(mapped_context, group_rows=group_rows)
        return self.matrix.lower(mapped_context)


def adapt(recipe, op):
    from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
    from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
    meta = dict(op.metadata)
    if (meta.get("matrix_mapping_domain") == "aggregate_experts"
            and meta.get("routing_basis") == "probability_moments"
            and meta.get("expected_active_experts", 0) > 0
            and isinstance(recipe, (TiledMatmul, HierarchicalMatmul, MappedMatmul))):
        return ExpectedGroupedMatmul(recipe)
    return recipe
