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
from typing import Literal

from mindie_sim.schema.common import (
    ModelingError,
    dtype_bytes,
    inclusion_probability as _common_inclusion_probability,
)
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.operators.matrix.matrix_mapping_owners import (
    is_arch35_context,
    primitive_l2_params,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp",
        source_sha256="ff47cab5a7fe02b90ee488efcebcc561d1870fcd73ed3600c3db650b624e3940",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="a3925fd49cd256ad478d2faf983bf2961b8ae2c8",
        relative_path="gmm/grouped_matmul/op_kernel/arch35/non_quant/grouped_matmul_kernel.h",
        source_sha256="75cb656bebf4d441edf04c8ab0813a0b8260eddbe949413fde07154f8be6d298",
    ),
)

ROW_TILE = 128
FRACTAL_M = 16
FRACTAL = FRACTAL_M
DAV3510_L2_CACHE_ALIGN_BYTES = 128
DAV3510_DEFAULT_BASE_N = 256
DAV3510_DEFAULT_BASE_K = 128
_NZ_LAYOUTS = frozenset({"NZ", "FRACTAL_NZ", "FRACTAL_NZ_C0_16", "FRACTAL_NZ_C0_32"})
_UNDERFLOW_PMF_CUTOFF = 1e-300
_EXACT_INT_TOL = 1e-9
_CONVEX_WEIGHT_EPS = 1e-12


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
        weight_layout = str(meta.get("weight_layout") or meta.get("layout_b") or "ND").upper()
        weight_nz = weight_layout in _NZ_LAYOUTS
        group_type = int(meta.get("group_type", 0) or 0)
        transpose_weight = bool(meta.get("transpose_weight", meta.get("trans_b", False)))
        inner_bytes = float(k if transpose_weight else n) * wt_b
        inner_aligned = (
            abs(inner_bytes - round(inner_bytes)) <= _EXACT_INT_TOL
            and int(round(inner_bytes)) % DAV3510_L2_CACHE_ALIGN_BYTES == 0
        )
        base_dim = (
            int(meta.get("base_k", DAV3510_DEFAULT_BASE_K) or DAV3510_DEFAULT_BASE_K)
            if transpose_weight
            else int(meta.get("base_n", DAV3510_DEFAULT_BASE_N) or DAV3510_DEFAULT_BASE_N)
        )
        tile_bytes = float(base_dim) * wt_b
        tile_aligned = (
            abs(tile_bytes - round(tile_bytes)) <= _EXACT_INT_TOL
            and int(round(tile_bytes)) % DAV3510_L2_CACHE_ALIGN_BYTES == 0
        )
        can_disable = (group_type == 0) and (weight_nz or (inner_aligned and tile_aligned))
        if l2_bytes > 0 and total_size >= l2_bytes and can_disable:
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


def _binomial_survival_at_thresholds(n: int, p: float, thresholds: tuple[int, ...]) -> tuple[float, ...]:
    """Exact ``P(Binomial(n, p) >= k)`` for each positive threshold ``k``."""
    if not thresholds:
        return ()
    if p <= 0.0:
        return (0.0,) * len(thresholds)
    if p >= 1.0:
        return tuple(1.0 if k <= n else 0.0 for k in thresholds)
    q = 1.0 - p
    mode = min(n, max(0, int((n + 1) * p)))
    log_p_mode = (
        math.lgamma(n + 1) - math.lgamma(mode + 1) - math.lgamma(n - mode + 1)
        + mode * math.log(p) + (n - mode) * math.log1p(-p)
    )
    p_mode = math.exp(log_p_mode)
    lower_targets = {k - 1 for k in thresholds if 1 < k <= mode}
    upper_targets = {k for k in thresholds if mode < k <= n}
    lower_cdf: dict[int, float] = {}
    upper_surv: dict[int, float] = {}
    if lower_targets:
        pmf = p_mode
        pmfs = [0.0] * (mode + 1)
        pmfs[mode] = pmf
        for j in range(mode, 0, -1):
            pmf = pmf * (j * q) / ((n - j + 1) * p)
            if pmf < _UNDERFLOW_PMF_CUTOFF:
                break
            pmfs[j - 1] = pmf
        acc = 0.0
        for j in range(mode + 1):
            acc += pmfs[j]
            if j in lower_targets:
                lower_cdf[j] = acc
    if upper_targets:
        pmf = p_mode
        max_k = max(upper_targets)
        upper_pmfs: list[tuple[int, float]] = []
        for j in range(mode, n):
            pmf = pmf * ((n - j) * p) / ((j + 1) * q)
            if pmf < _UNDERFLOW_PMF_CUTOFF:
                break
            if j + 1 >= min(upper_targets):
                upper_pmfs.append((j + 1, pmf))
            if j + 1 > max_k and pmf < _UNDERFLOW_PMF_CUTOFF:
                break
        acc = 0.0
        tail_by_j: dict[int, float] = {}
        for j, val in reversed(upper_pmfs):
            acc += val
            tail_by_j[j] = acc
        for k in upper_targets:
            upper_surv[k] = tail_by_j.get(k, 0.0)
    out = []
    for k in thresholds:
        if k <= 0:
            out.append(1.0)
        elif k > n:
            out.append(0.0)
        elif k == 1:
            out.append(_common_inclusion_probability(float(n), p))
        elif k <= mode:
            out.append(max(0.0, min(1.0, 1.0 - lower_cdf.get(k - 1, 0.0))))
        else:
            out.append(max(0.0, min(1.0, upper_surv.get(k, 0.0))))
    return tuple(out)


def row_tile_moments(probabilities, *, tokens, row_tile=ROW_TILE):
    """Expected row-tile count and occupied-row moment per expert."""
    total_tokens = int(tokens)
    tile_step = int(row_tile)
    starts = tuple(range(0, total_tokens, tile_step))
    thresholds = tuple(start + 1 for start in starts)
    result = []
    for value in probabilities:
        p = min(1.0, max(0.0, float(value)))
        tile_probability = _binomial_survival_at_thresholds(total_tokens, p, thresholds)
        occupied = float(total_tokens) * p
        result.append((tile_probability, occupied))
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
    from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
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
    if isinstance(program, RefillPipeline):
        orig_panel = program.panel
        def _scaled_panel(idx, _orig=orig_panel, _f=factor):
            load, steps, release = _orig(idx)
            scaled_load = _scale_stage_work(load, _f)
            scaled_steps = (
                (tuple(_scale_stage_work(s, _f) for s in pair), is_last)
                for pair, is_last in steps
            )
            return scaled_load, scaled_steps, release
        return replace(program, panel=_scaled_panel)
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


_SCALED_META_KEYS = (
    "packed_weight_bytes",
    "weight_bytes",
    "routed_weight_read_bytes",
    "admitted_l2_read_hit_bytes",
    "activation_scale_bytes",
)


def _scale_op_and_context(op, context, meta: dict, factor: float):
    """Scale declared op/context/metadata work by ``factor`` before conditional lowering."""
    scaled_meta = dict(meta)
    for key in _SCALED_META_KEYS:
        if key in scaled_meta and scaled_meta[key] is not None:
            scaled_meta[key] = float(scaled_meta[key]) * factor
    if "matrix_epilogue" in scaled_meta:
        epi = dict(scaled_meta["matrix_epilogue"])
        for key in ("row_scale_bytes", "column_scale_bytes"):
            if key in epi and epi[key] is not None:
                epi[key] = float(epi[key]) * factor
        scaled_meta["matrix_epilogue"] = epi
    scaled_matrix = tuple((dt, float(val) * factor) for dt, val in op.matrix_flops)
    scaled_op = replace(
        op,
        flops=float(op.flops) * factor,
        hbm_bytes=float(op.hbm_bytes) * factor,
        mte2_bytes=float(op.mte2_bytes) * factor,
        mte3_bytes=float(op.mte3_bytes) * factor,
        vector_ops=float(op.vector_ops) * factor,
        matrix_flops=scaled_matrix,
        metadata=tuple(scaled_meta.items()),
    )
    scaled_ctx = replace(
        context,
        op=scaled_op,
        work={k: float(v) * factor for k, v in context.work.items()},
        native_work={k: float(v) * factor for k, v in context.native_work.items()},
    )
    return scaled_op, scaled_ctx, scaled_meta


def _realize_bank_branches(
    rows: float, banks: int,
) -> tuple[tuple[float, int, tuple[int, ...], tuple[int, ...], int], ...]:
    """Realize integer row/bank schedules with fractal padding applied before averaging."""
    if rows <= float(banks):
        dispersed = _dispersed_group_rows(float(banks), banks)
        padded_groups, padded_m = _per_group_padded_m(dispersed)
        return ((1.0, banks, dispersed, padded_groups, padded_m),)
    r_lo, r_hi = math.floor(rows), math.ceil(rows)
    f_hi = rows - float(r_lo)
    f_lo = 1.0 - f_hi
    if r_lo == r_hi or f_hi <= _CONVEX_WEIGHT_EPS or f_lo <= _CONVEX_WEIGHT_EPS:
        r_int = r_lo if f_hi <= _CONVEX_WEIGHT_EPS else r_hi
        dispersed = _dispersed_group_rows(float(r_int), banks)
        padded_groups, padded_m = _per_group_padded_m(dispersed)
        return ((1.0, banks, dispersed, padded_groups, padded_m),)
    disp_lo = _dispersed_group_rows(float(r_lo), banks)
    pad_grp_lo, pad_lo = _per_group_padded_m(disp_lo)
    disp_hi = _dispersed_group_rows(float(r_hi), banks)
    pad_grp_hi, pad_hi = _per_group_padded_m(disp_hi)
    return (
        (f_lo, banks, disp_lo, pad_grp_lo, pad_lo),
        (f_hi, banks, disp_hi, pad_grp_hi, pad_hi),
    )


@dataclass(frozen=True)
class ExpectedGroupedMatmul:
    matrix: object

    def _lower_one(self, op, context, meta: dict, *, group_rows, banks: int, slots: int,
                   padded_m: int, n: int, k: int, use_group_rows: bool):
        branch_meta = dict(meta)
        branch_slots = slots
        if not use_group_rows and banks > 0 and branch_slots > 0 and branch_slots % banks:
            per_bank = math.ceil(branch_slots / banks)
            branch_slots = banks * per_bank
        branch_meta.update(matrix_groups=banks, matrix_mapping_domain="expected_expert_banks")
        if "matrix_epilogue" in branch_meta:
            epilogue = dict(branch_meta["matrix_epilogue"])
            epilogue["operations_per_output"] = (
                op.vector_ops / (branch_slots * n) if (branch_slots > 0 and n > 0) else 0.0
            )
            branch_meta["matrix_epilogue"] = epilogue
        shape = (branch_slots, n, k)
        padded_shape = (
            padded_m,
            math.ceil(n / FRACTAL_M) * FRACTAL_M,
            math.ceil(k / FRACTAL_M) * FRACTAL_M,
        )
        mapped = replace(
            op, local_shape=shape, launch_shape=shape,
            padded_shape=padded_shape, metadata=tuple(branch_meta.items()),
        )
        mapped_context = replace(context, op=mapped)
        return (
            self.matrix.lower(mapped_context, group_rows=group_rows)
            if use_group_rows
            else self.matrix.lower(mapped_context)
        ), branch_slots

    def lower(self, context):
        from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
        from mindie_sim.prediction.program.stages import KernelProgram, Stage

        op = context.op
        meta = dict(op.metadata)
        _, n, k = op.local_shape[(-3):]
        explicit_hist = meta.get("expert_rows") or meta.get("rows_by_expert")
        aclnn_hist = meta.get("aclnn_group_list")
        is_exact_histogram = bool(explicit_hist or aclnn_hist)
        use_steady = isinstance(self.matrix, SteadyMatmul)

        if is_exact_histogram:
            raw_hist = explicit_hist if explicit_hist else aclnn_hist
            full_rows = tuple(max(0, int(value)) for value in raw_hist)
            active_rows = tuple(value for value in full_rows if value > 0)
            declared_groups = max(len(full_rows), int(meta.get("matrix_weight_bank_groups", 0) or 0))
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
                declared_groups=declared_groups, rows=rows, n=n, k=k, dtype=op.dtype,
            )
            final_slots = slots
            if not use_steady and banks > 0 and final_slots > 0 and final_slots % banks:
                per_bank = math.ceil(final_slots / banks)
                final_slots = banks * per_bank
            if context.record_mapping is not None:
                context.record_mapping({
                    "execution_abstraction": "expected_expert_bank_grid",
                    "expected_rows": rows, "expected_active_experts": active,
                    "conditional_mean_rows": (rows / active) if active else 0.0,
                    "bank_slots": banks, "rows_per_bank_slot": per_bank,
                    "semantic_rows": op.local_shape[(-3)], "scheduling_rows": final_slots,
                    "padded_m": padded_m,
                    "work_policy": "preserve_semantic_arithmetic_and_payload_moments",
                    "private_service_policy": "derive_owner_copies_and_compute_width_from_rounded_grid",
                    "schedule": "exact_histogram_owner_tail",
                    "outer_invocations": 1,
                })
            lowered, _ = self._lower_one(
                op, context, meta, group_rows=group_rows, banks=banks,
                slots=final_slots, padded_m=padded_m, n=n, k=k, use_group_rows=use_steady,
            )
            return lowered

        active = float(meta["expected_active_experts"])
        rows = float(meta["expected_rows"])
        if rows <= 0.0:
            banks, dispersed_rows, padded_group_rows, padded_m, expected_padded_m = 0, (), (), 0, 0.0
            branches: tuple[tuple[float, int, tuple[int, ...], tuple[int, ...], int], ...] = (
                (1.0, 0, (), (), 0),
            )
            sub_unit_prob: float | None = None
        elif active <= 0.0:
            _dispersed_group_rows(rows, 0)
            return self.matrix.lower(context)
        elif active < 1.0:
            p_active = active
            cond_padded = math.ceil((rows / p_active) / FRACTAL_M) * FRACTAL_M
            expected_padded_m = p_active * float(cond_padded)
            banks, padded_m = 1, cond_padded
            dispersed_rows = (max(1, math.ceil(rows / p_active)),)
            padded_group_rows = (cond_padded,)
            branches = ((p_active, 1, dispersed_rows, padded_group_rows, cond_padded),)
            sub_unit_prob = p_active
        else:
            sub_unit_prob = None
            b_lo, b_hi = math.floor(active), math.ceil(active)
            w_hi = active - float(b_lo)
            w_lo = 1.0 - w_hi
            if b_lo == b_hi or w_hi <= _CONVEX_WEIGHT_EPS or w_lo <= _CONVEX_WEIGHT_EPS:
                banks = b_lo if w_hi <= _CONVEX_WEIGHT_EPS else b_hi
                branches = _realize_bank_branches(rows, banks)
            else:
                lo_parts = tuple(
                    (w_lo * rw, bc, bd, bp, bm)
                    for rw, bc, bd, bp, bm in _realize_bank_branches(rows, b_lo)
                )
                hi_parts = tuple(
                    (w_hi * rw, bc, bd, bp, bm)
                    for rw, bc, bd, bp, bm in _realize_bank_branches(rows, b_hi)
                )
                branches = lo_parts + hi_parts
                banks = b_hi
            expected_padded_m = math.fsum(w * float(bm) for w, _, _, _, bm in branches)
            _, _, dispersed_rows, padded_group_rows, padded_m = branches[-1]

        padding_factor = (expected_padded_m / rows) if rows > 0.0 else 1.0
        declared_groups = max(banks, int(meta.get("matrix_weight_bank_groups", 0) or 0))
        meta["declared_group_count"] = declared_groups
        meta["expected_dispersed_group_rows"] = dispersed_rows
        meta["expected_padded_group_rows"] = padded_group_rows
        meta["expected_padded_m"] = expected_padded_m
        meta["expected_padding_factor"] = padding_factor
        base_meta = dict(meta)
        _apply_grouped_weight_cache_policy(
            context, meta, active_rows=dispersed_rows,
            declared_groups=declared_groups, rows=rows, n=n, k=k, dtype=op.dtype,
        )
        rec_slots = padded_m
        rec_per_bank = max(padded_group_rows, default=0)
        if not use_steady and banks > 0 and rec_slots > 0 and rec_slots % banks:
            rec_per_bank = math.ceil(rec_slots / banks)
            rec_slots = banks * rec_per_bank
        if context.record_mapping is not None:
            context.record_mapping({
                "execution_abstraction": "expected_expert_bank_grid",
                "expected_rows": rows, "expected_active_experts": active,
                "conditional_mean_rows": (rows / active) if active else 0.0,
                "bank_slots": banks, "rows_per_bank_slot": rec_per_bank,
                "semantic_rows": op.local_shape[(-3)], "scheduling_rows": rec_slots,
                "padded_m": expected_padded_m,
                "work_policy": "preserve_semantic_arithmetic_and_payload_moments",
                "private_service_policy": "derive_owner_copies_and_compute_width_from_rounded_grid",
                "schedule": "capacity_rounded_conditional_mean_not_sampled_histogram",
                "outer_invocations": 1,
            })

        if sub_unit_prob is not None:
            work_op, work_ctx, work_meta = _scale_op_and_context(op, context, meta, 1.0 / sub_unit_prob)
            prog, _ = self._lower_one(
                work_op, work_ctx, work_meta, group_rows=padded_group_rows, banks=banks,
                slots=padded_m, padded_m=padded_m, n=n, k=k, use_group_rows=use_steady,
            )
            return _scale_program_work(prog, sub_unit_prob)
        if len(branches) == 1:
            _, b_cnt, _, b_pad_grp, b_pad_m = branches[0]
            lowered, _ = self._lower_one(
                op, context, meta, group_rows=b_pad_grp, banks=b_cnt,
                slots=b_pad_m, padded_m=b_pad_m, n=n, k=k, use_group_rows=use_steady,
            )
            return lowered
        stage_list = []
        for idx, (weight, b_cnt, b_disp, b_pad_grp, b_pad_m) in enumerate(branches):
            b_meta = {
                **base_meta,
                "expected_dispersed_group_rows": b_disp,
                "expected_padded_group_rows": b_pad_grp,
            }
            _apply_grouped_weight_cache_policy(
                context, b_meta, active_rows=b_disp,
                declared_groups=declared_groups, rows=float(sum(b_disp)), n=n, k=k, dtype=op.dtype,
            )
            prog, _ = self._lower_one(
                op, context, b_meta, group_rows=b_pad_grp, banks=b_cnt,
                slots=b_pad_m, padded_m=b_pad_m, n=n, k=k, use_group_rows=use_steady,
            )
            after = (stage_list[-1].name,) if stage_list else ()
            stage_list.append(Stage(f"banks_{idx}", after=after, program=_scale_program_work(prog, weight)))
        return KernelProgram(tuple(stage_list))


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
