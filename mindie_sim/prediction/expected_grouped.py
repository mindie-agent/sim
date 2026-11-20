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

ROW_TILE = 128
FRACTAL = 16
A5_MXFP4_LOW_OCC_ROWS = 16
A5_MXFP4_DECODE_UNPACK_FRAC = 0.65
A5_MXFP4_FORMATS = frozenset({"mxfp4", "fp4", "fp4_e2m1"})


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
        histogram = meta.get("expert_rows") or meta.get("rows_by_expert")
        if histogram:
            group_rows = tuple(max(0, int(value)) for value in histogram)
            active = sum(1 for value in group_rows if value > 0)
            rows = sum(group_rows)
            banks = len(group_rows)
            per_bank = (max(group_rows) if group_rows else 0)
            slots = max(1, rows)
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
        _, n, k = op.local_shape[(-3):]
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
                "conditional_mean_rows": rows / active,
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
            from .matrix_steady import SteadyMatmul
            if isinstance(self.matrix, SteadyMatmul):
                inner = self.matrix.lower(mapped_context, group_rows=group_rows)
            else:
                inner = self.matrix.lower(mapped_context)
        else:
            inner = self.matrix.lower(mapped_context)
        if (
            "mte2_aic_dram" in context.service_modes
            and op.phase == "decode"
            and str(meta.get("weight_format", "")) in A5_MXFP4_FORMATS
            and per_bank <= A5_MXFP4_LOW_OCC_ROWS
            and "hbm" in context.resource_names
        ):
            hbm_rate = float(context.rates.get(context.resource_names["hbm"], 0.0))
            if hbm_rate > 0.0:
                from .program import KernelProgram, Stage
                extra_ms = A5_MXFP4_DECODE_UNPACK_FRAC * (float(op.hbm_bytes) / hbm_rate)
                return KernelProgram((
                    Stage("a5_mxfp4_unpack", latency=extra_ms),
                    Stage("gmm_body", after=("a5_mxfp4_unpack",), program=inner),
                ))
        return inner


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
