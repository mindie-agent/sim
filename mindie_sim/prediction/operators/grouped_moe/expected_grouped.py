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

from mindie_sim.schema.common import (
    ModelingError,
    dtype_bytes,
    inclusion_probability as _common_inclusion_probability,
)
from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
    is_arch35_context,
    primitive_l2_params,
)

ROW_TILE = 128
FRACTAL_M = 16
FRACTAL = FRACTAL_M


def _dispersed_group_rows(rows: float, banks: int) -> tuple[int, ...]:
    """Zero-demand-safe, row-conserving binomial dispersion across active banks (G2)."""
    if rows <= 0.0:
        return ()
    raw_banks = int(banks)
    if raw_banks <= 0:
        raise ModelingError(f"grouped_moe_zero_banks_nonzero_rows:rows={rows}")
    total_rows = max(1, int(round(rows)))
    bank_count = min(raw_banks, total_rows)
    if bank_count == 1:
        return (total_rows,)
    if total_rows <= bank_count:
        return (1,) * bank_count
    mean_m = float(total_rows) / float(bank_count)
    spread = min(
        mean_m - 1.0,
        math.sqrt(mean_m * (1.0 - 1.0 / float(bank_count))),
    )
    low_m = max(1, math.floor(mean_m - spread))
    high_m = max(1, math.ceil(mean_m + spread))
    half = bank_count // 2
    if bank_count % 2 == 0:
        counts = [low_m] * half + [high_m] * half
    else:
        mid_m = max(1, int(round(mean_m)))
        counts = [low_m] * half + [mid_m] + [high_m] * half
    # Exact largest-remainder conservation so sum(counts) == total_rows strictly.
    diff = total_rows - sum(counts)
    idx = 0
    while diff > 0:
        counts[idx % bank_count] += 1
        diff -= 1
        idx += 1
    while diff < 0:
        pos = (bank_count - 1 - (idx % bank_count))
        if counts[pos] > 1:
            counts[pos] -= 1
            diff += 1
        idx += 1
    return tuple(counts)


def _per_group_padded_m(group_rows: tuple[int, ...]) -> tuple[tuple[int, ...], int]:
    """Pad each active expert group independently to Cube FRACTAL_M = 16 (zero-safe)."""
    padded_groups = tuple(
        math.ceil(group_m / FRACTAL_M) * FRACTAL_M
        for group_m in group_rows if group_m > 0
    )
    if not padded_groups:
        return (), 0
    return padded_groups, sum(padded_groups)


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
    act_fmt = str(meta.get("activation_format") or meta.get("compute_format") or dtype or "").strip()
    wt_fmt = str(meta.get("weight_format") or meta.get("compute_format") or dtype or "").strip()
    out_fmt = str(meta.get("output_format") or dtype or act_fmt).strip()
    if not act_fmt or not wt_fmt or not out_fmt:
        raise ModelingError("expected_grouped_l2_policy_dtype_missing")
    act_b = dtype_bytes(act_fmt)
    wt_b = dtype_bytes(wt_fmt)
    out_b = dtype_bytes(out_fmt)
    x_bytes = float(rows * k) * act_b
    l2_bytes, _ = primitive_l2_params(context)
    if is_arch35_context(context):
        total_size = (
            float(rows * k) * act_b
            + float(declared_groups * k * n) * wt_b
            + float(rows * n) * out_b
        )
        if l2_bytes > 0 and total_size >= l2_bytes:
            meta["admitted_l2_read_hit_bytes"] = min(admitted, x_bytes)
            meta["l2_weight_cache_disable"] = True
        return
    cacheable_experts = sum(1 for r in active_rows if r > ROW_TILE)
    cacheable_weight_bytes = float(cacheable_experts * k * n) * wt_b
    meta["admitted_l2_read_hit_bytes"] = min(admitted, x_bytes + cacheable_weight_bytes)
    if cacheable_experts == 0:
        meta["l2_weight_cache_disable"] = True


def inclusion_probability(probabilities, *, tokens):
    """Exact ``1 - (1-p)^tokens`` delegating to canonical L0 ``schema.common``."""
    return tuple(_common_inclusion_probability(float(tokens), float(value)) for value in probabilities)


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


def _scale_stage_work(stage, factor: float):
    """Weight a conditional tile stage's issued work and latency by execution probability."""
    return replace(
        stage,
        work={key: float(val) * factor for key, val in stage.work.items()},
        latency=float(stage.latency) * factor,
        program=(
            _scale_program_work(stage.program, factor)
            if stage.program is not None else None
        ),
    )


def _scale_program_work(program, factor: float):
    """Weight a conditional execution program's issued work by firing probability ``factor``."""
    from mindie_sim.prediction.program.pipeline import PipelineProgram, PipelineTemplateProgram
    from mindie_sim.prediction.program.stages import KernelProgram, ParallelProgram

    if isinstance(program, KernelProgram):
        return replace(
            program,
            stages=tuple(_scale_stage_work(s, factor) for s in program.stages),
            minimum_ms=float(program.minimum_ms) * factor,
        )
    if isinstance(program, ParallelProgram):
        return replace(
            program,
            programs=tuple(_scale_program_work(p, factor) for p in program.programs),
        )
    if isinstance(program, PipelineProgram):
        orig_tile = program.tile
        return replace(
            program,
            tile=lambda idx, _orig=orig_tile, _f=factor: tuple(
                _scale_stage_work(s, _f) for s in _orig(idx)
            ),
            minimum_ms=float(program.minimum_ms) * factor,
            setup_ms=float(program.setup_ms) * factor,
        )
    if isinstance(program, PipelineTemplateProgram):
        tmpl = program.template
        scaled_phases = tuple(
            replace(ph, stages=tuple(_scale_stage_work(s, factor) for s in ph.stages))
            for ph in tmpl.phases
        )
        scaled_tmpl = replace(
            tmpl,
            setup=tuple(_scale_stage_work(s, factor) for s in tmpl.setup),
            drain=tuple(_scale_stage_work(s, factor) for s in tmpl.drain),
            phases=scaled_phases,
            minimum_ms=float(tmpl.minimum_ms) * factor,
            setup_ms=float(tmpl.setup_ms) * factor,
        )
        return replace(program, template=scaled_tmpl)
    return program


@dataclass(frozen=True)
class ExpectedGroupedMatmul:
    matrix: object

    def lower(self, context):
        op = context.op
        meta = dict(op.metadata)
        _, n, k = op.local_shape[(-3):]
        explicit_hist = meta.get("expert_rows") or meta.get("rows_by_expert")
        aclnn_hist = meta.get("aclnn_group_list")
        is_exact_histogram = False
        if explicit_hist:
            is_exact_histogram = True
            group_rows = tuple(max(0, int(value)) for value in explicit_hist)
            active_rows = tuple(value for value in group_rows if value > 0)
            active = len(active_rows)
            rows = sum(group_rows)
            banks = len(group_rows)
            per_bank = max(group_rows, default=0)
            slots = max(1, rows)
            _, padded_m = _per_group_padded_m(active_rows)
            declared_groups = max(banks, int(meta.get("matrix_weight_bank_groups", 0) or 0))
            meta["declared_group_count"] = declared_groups
            _apply_grouped_weight_cache_policy(
                context, meta, active_rows=active_rows,
                declared_groups=declared_groups, rows=rows, n=n, k=k,
                dtype=op.dtype,
            )
        elif aclnn_hist:
            is_exact_histogram = True
            full_rows = tuple(max(0, int(value)) for value in aclnn_hist)
            active_rows = tuple(value for value in full_rows if value > 0)
            declared_groups = len(full_rows)
            active = len(active_rows)
            rows = sum(active_rows)
            group_rows = active_rows if active > 0 else (0,)
            banks = max(1, active)
            per_bank = max(group_rows, default=0)
            slots = max(1, rows)
            _, padded_m = _per_group_padded_m(active_rows)
            meta["declared_group_count"] = declared_groups
            _apply_grouped_weight_cache_policy(
                context, meta, active_rows=active_rows,
                declared_groups=declared_groups, rows=rows, n=n, k=k,
                dtype=op.dtype,
            )
        else:
            active = float(meta["expected_active_experts"])
            rows = float(meta["expected_rows"])
            # Each active expert bank pads along M to Cube FRACTAL_M = 16
            # independently, with zero-safe binomial row-count dispersion across
            # banks, while logical op.flops stays unpadded (G2).
            sub_unit_prob: float | None = None
            if rows <= 0.0:
                banks = 0
                dispersed_rows = ()
                padded_group_rows, padded_m = (), 0
                expected_padded_m: float = 0.0
            elif active <= 0.0:
                # Fail fast on M > 0, B = 0 via canonical guard.
                _dispersed_group_rows(rows, 0)
                banks = 0
                dispersed_rows = ()
                padded_group_rows, padded_m = (), 0
                expected_padded_m = 0.0
            elif rows < 1.0 and active < 1.0:
                # Sub-unit expected occupancy (e.g. rows=0.5/0.01, active=0.5/0.01):
                # lower the real conditional FRACTAL_M=16 tile schedule (cond_padded rows)
                # and weight its issued program work by execution probability p_active.
                p_active = min(1.0, active)
                cond_padded = math.ceil((rows / p_active) / FRACTAL_M) * FRACTAL_M
                expected_padded_m = p_active * float(cond_padded)
                padded_m = cond_padded
                banks = 1
                dispersed_rows = (max(1, math.ceil(rows / p_active)),)
                padded_group_rows = (cond_padded,)
                sub_unit_prob = p_active
            else:
                banks = math.ceil(active)
                dispersed_rows = _dispersed_group_rows(rows, banks)
                padded_group_rows, padded_m = _per_group_padded_m(dispersed_rows)
                expected_padded_m = float(padded_m)
            group_rows = padded_group_rows
            per_bank = max(padded_group_rows, default=0)
            slots = padded_m
            padding_factor = (expected_padded_m / rows) if rows > 0.0 else 1.0
            declared_groups = max(banks, int(meta.get("matrix_weight_bank_groups", 0) or 0))
            meta["declared_group_count"] = declared_groups
            meta["expected_dispersed_group_rows"] = dispersed_rows
            meta["expected_padded_group_rows"] = padded_group_rows
            meta["expected_padded_m"] = expected_padded_m
            meta["expected_padding_factor"] = padding_factor
            _apply_grouped_weight_cache_policy(
                context, meta, active_rows=dispersed_rows,
                declared_groups=declared_groups, rows=rows, n=n, k=k,
                dtype=op.dtype,
            )
        from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
        use_group_rows = group_rows is not None and isinstance(self.matrix, SteadyMatmul)
        if not use_group_rows and banks > 0 and slots > 0 and slots % banks:
            per_bank = math.ceil(slots / banks)
            slots = banks * per_bank
        meta.update(matrix_groups=banks, matrix_mapping_domain="expected_expert_banks")
        if "matrix_epilogue" in meta:
            epilogue = dict(meta["matrix_epilogue"])
            epilogue["operations_per_output"] = (
                op.vector_ops / (slots * n) if (slots > 0 and n > 0) else 0.0
            )
            meta["matrix_epilogue"] = epilogue
        shape = (slots, n, k)
        padded_shape = (
            padded_m,
            math.ceil(n / FRACTAL_M) * FRACTAL_M,
            math.ceil(k / FRACTAL_M) * FRACTAL_M,
        )
        mapped = replace(op, local_shape=shape, launch_shape=shape,
                         padded_shape=padded_shape,
                         metadata=tuple(meta.items()))
        if context.record_mapping is not None:
            context.record_mapping({
                "execution_abstraction": "expected_expert_bank_grid",
                "expected_rows": rows, "expected_active_experts": active,
                "conditional_mean_rows": (rows / active) if active else 0.0,
                "bank_slots": banks, "rows_per_bank_slot": per_bank,
                "semantic_rows": op.local_shape[(-3)], "scheduling_rows": slots,
                "padded_m": padded_m,
                "work_policy": "preserve_semantic_arithmetic_and_payload_moments",
                "private_service_policy": "derive_owner_copies_and_compute_width_from_rounded_grid",
                "schedule": ("exact_histogram_owner_tail" if is_exact_histogram
                             else "capacity_rounded_conditional_mean_not_sampled_histogram"),
                "outer_invocations": 1,
            })
        mapped_context = replace(context, op=mapped)
        lowered = (
            self.matrix.lower(mapped_context, group_rows=group_rows)
            if use_group_rows
            else self.matrix.lower(mapped_context)
        )
        if not is_exact_histogram and sub_unit_prob is not None:
            return _scale_program_work(lowered, sub_unit_prob)
        return lowered


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
