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
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from .expected_grouped import FRACTAL, SOURCES as _EXPECTED_SOURCES

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = _EXPECTED_SOURCES


def _parent_hbm_decomposition(context) -> tuple[float, float]:
    """Return ``(effective_read_credit, extra_non_io_hbm)`` for parent HBM conservation."""
    op = context.op
    meta = dict(op.metadata)
    parent_hit = max(0.0, float(meta.get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
    parent_hbm = float(context.native_work.get("hbm", op.hbm_bytes))
    cold_io = float(op.mte2_bytes) + float(op.mte3_bytes)
    extra = parent_hbm - cold_io + parent_hit
    if extra < 0.0:
        return parent_hit - extra, 0.0
    return parent_hit, extra


def _emitted_stage_read_hbm(stage, hbm_name: str, mte3_name: str | None) -> float:
    total = 0.0
    if hbm_name in stage.work and (mte3_name is None or mte3_name not in stage.work):
        total += float(stage.work[hbm_name])
    if stage.program is not None:
        total += _emitted_program_read_hbm(stage.program, hbm_name, mte3_name)
    return total


def _emitted_program_read_hbm(program, hbm_name: str, mte3_name: str | None) -> float:
    """Sum emitted read-side HBM bytes (including operand rereads) in a lowered program."""
    from mindie_sim.prediction.operators.matrix.matrix import _StreamRefillProgram
    from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
    from mindie_sim.prediction.program.pipeline import PipelineProgram, PipelineTemplateProgram
    from mindie_sim.prediction.program.stages import KernelProgram, ParallelProgram

    if isinstance(program, _StreamRefillProgram):
        return _emitted_program_read_hbm(program.pipeline, hbm_name, mte3_name)
    if isinstance(program, KernelProgram):
        return math.fsum(_emitted_stage_read_hbm(s, hbm_name, mte3_name) for s in program.stages)
    if isinstance(program, ParallelProgram):
        return math.fsum(_emitted_program_read_hbm(p, hbm_name, mte3_name) for p in program.programs)
    if isinstance(program, PipelineProgram):
        return math.fsum(
            _emitted_stage_read_hbm(s, hbm_name, mte3_name)
            for idx in range(program.count) for s in program.tile(idx)
        )
    if isinstance(program, RefillPipeline):
        total = 0.0
        for idx in range(program.count):
            load, steps, _ = program.panel(idx)
            total += _emitted_stage_read_hbm(load, hbm_name, mte3_name)
            for pair, _ in steps:
                total += math.fsum(_emitted_stage_read_hbm(s, hbm_name, mte3_name) for s in pair)
        return total
    if isinstance(program, PipelineTemplateProgram):
        tmpl = program.template
        total = math.fsum(_emitted_stage_read_hbm(s, hbm_name, mte3_name) for s in tmpl.setup)
        total += math.fsum(_emitted_stage_read_hbm(s, hbm_name, mte3_name) for s in tmpl.drain)
        if tmpl.indexed_panels is not None:
            for idx in range(tmpl.indexed_panels):
                total += math.fsum(
                    _emitted_stage_read_hbm(s, hbm_name, mte3_name)
                    for s in tmpl.stage_for(idx)
                )
            return total
        for ph in tmpl.phases:
            if ph.residue_stages:
                for idx in range(ph.count):
                    total += math.fsum(
                        _emitted_stage_read_hbm(s, hbm_name, mte3_name)
                        for s in ph.stages_for(idx)
                    )
            else:
                phase_sum = math.fsum(_emitted_stage_read_hbm(s, hbm_name, mte3_name) for s in ph.stages)
                total += float(ph.count) * phase_sum
        return total
    return 0.0


def _scale_stage_read_hbm(stage, hbm_name: str, mte3_name: str | None, factor: float):
    work = stage.work
    if hbm_name in work and (mte3_name is None or mte3_name not in work):
        work = {**work, hbm_name: float(work[hbm_name]) * factor}
    nested = (
        _scale_program_read_hbm(stage.program, hbm_name, mte3_name, factor)
        if stage.program is not None else None
    )
    if work is stage.work and nested is stage.program:
        return stage
    return replace(stage, work=work, program=nested)


def _scale_phase_read_hbm(phase, hbm_name: str, mte3_name: str | None, factor: float):
    scaled_stages = tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, factor) for s in phase.stages)
    scaled_residue = tuple(
        tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, factor) for s in group)
        for group in phase.residue_stages
    )
    return replace(phase, stages=scaled_stages, residue_stages=scaled_residue)


def _scale_program_read_hbm(program, hbm_name: str, mte3_name: str | None, factor: float):
    """Apply partitioned L2 read-hit credit to wrapped custom read stages."""
    from mindie_sim.prediction.operators.matrix.matrix import _StreamRefillProgram
    from mindie_sim.prediction.operators.matrix.refill import RefillPipeline
    from mindie_sim.prediction.program.pipeline import PipelineProgram, PipelineTemplateProgram
    from mindie_sim.prediction.program.stages import KernelProgram, ParallelProgram

    if isinstance(program, _StreamRefillProgram):
        return replace(
            program,
            pipeline=_scale_program_read_hbm(program.pipeline, hbm_name, mte3_name, factor),
        )
    if isinstance(program, KernelProgram):
        return replace(
            program,
            stages=tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, factor) for s in program.stages),
        )
    if isinstance(program, ParallelProgram):
        return replace(
            program,
            programs=tuple(_scale_program_read_hbm(p, hbm_name, mte3_name, factor) for p in program.programs),
        )
    if isinstance(program, PipelineProgram):
        orig_tile = program.tile
        return replace(
            program,
            tile=lambda idx, _orig=orig_tile, _f=factor: tuple(
                _scale_stage_read_hbm(s, hbm_name, mte3_name, _f) for s in _orig(idx)
            ),
        )
    if isinstance(program, RefillPipeline):
        orig_panel = program.panel
        def _scaled_panel(idx, _orig=orig_panel, _f=factor):
            load, steps, release = _orig(idx)
            scaled_load = _scale_stage_read_hbm(load, hbm_name, mte3_name, _f)
            scaled_steps = (
                (tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, _f) for s in pair), is_last)
                for pair, is_last in steps
            )
            return scaled_load, scaled_steps, release
        return replace(program, panel=_scaled_panel)
    if isinstance(program, PipelineTemplateProgram):
        tmpl = program.template
        scaled_phases = tuple(
            _scale_phase_read_hbm(ph, hbm_name, mte3_name, factor)
            for ph in tmpl.phases
        )
        scaled_motif = tuple(
            _scale_phase_read_hbm(ph, hbm_name, mte3_name, factor)
            for ph in tmpl.motif_phases
        )
        scaled_tmpl = replace(
            tmpl,
            setup=tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, factor) for s in tmpl.setup),
            drain=tuple(_scale_stage_read_hbm(s, hbm_name, mte3_name, factor) for s in tmpl.drain),
            phases=scaled_phases,
            motif_phases=scaled_motif,
        )
        return replace(program, template=scaled_tmpl)
    return program


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
    from mindie_sim.prediction.mechanism import Work
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
    read_fraction = (read / float(op.mte2_bytes)) if op.mte2_bytes > 0 else row_fraction
    effective_hit, parent_extra = _parent_hbm_decomposition(context)
    child_hit = effective_hit * read_fraction
    metadata.update(matrix_groups=experts, matrix_group_rows=(rows_per_expert,) * experts,
                    matrix_mapping_domain="equal_expert_groups", packed_weight_bytes=sub_weight)
    if effective_hit > 0.0 or "admitted_l2_read_hit_bytes" in metadata:
        metadata["admitted_l2_read_hit_bytes"] = child_hit
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
    resources["mte2"] = read + (context.native_work.get("mte2", op.mte2_bytes) - op.mte2_bytes) * row_fraction
    resources["mte3"] = write + (context.native_work.get("mte3", op.mte3_bytes) - op.mte3_bytes) * row_fraction
    resources["hbm"] = max(0.0, read + write - child_hit)
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
        padded_shape=tuple(math.ceil(v / FRACTAL) * FRACTAL for v in shape),
        metadata=tuple(metadata.items()))
    return replace(phase, op=phase_op)


@dataclass(frozen=True)
class IntegerGroupedMatmul:
    matrix: object

    def lower(self, context):
        from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
        from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
        from mindie_sim.prediction.operators.matrix.matrix_phase import PhaseMatmul
        from mindie_sim.prediction.operators.matrix.matrix_steady import SteadyMatmul
        from mindie_sim.prediction.program import KernelProgram, Stage

        classes = row_classes(context.op)
        _, parent_extra = _parent_hbm_decomposition(context)
        hbm_name = context.resource_names.get("hbm")
        mte3_name = context.resource_names.get("mte3")
        setup_work = {hbm_name: parent_extra} if (parent_extra > 0.0 and hbm_name) else {}
        stages = [Stage("setup", setup_work, latency=context.setup_ms)]
        native_credit_recipes = (
            SteadyMatmul, MappedMatmul, HierarchicalMatmul, TiledMatmul, PhaseMatmul,
        )
        needs_read_credit_scale = (
            not isinstance(self.matrix, native_credit_recipes) and bool(hbm_name)
        )
        for rows, experts in classes:
            phase = split_context(context, rows, experts)
            child_program = self.matrix.lower(phase)
            if needs_read_credit_scale:
                child_hit = max(0.0, float(dict(phase.op.metadata).get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
                if child_hit > 0.0:
                    emitted_reads = _emitted_program_read_hbm(child_program, hbm_name, mte3_name)
                    if emitted_reads > 0.0:
                        factor = max(0.0, 1.0 - min(child_hit, emitted_reads) / emitted_reads)
                        child_program = _scale_program_read_hbm(child_program, hbm_name, mte3_name, factor)
            stages.append(Stage(
                f"rows{rows}", after=(stages[-1].name,), program=child_program))
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
    from mindie_sim.prediction.operators.matrix.matrix import HierarchicalMatmul, TiledMatmul
    from mindie_sim.prediction.operators.matrix.matrix_mapping import MappedMatmul
    if isinstance(recipe, (MappedMatmul, HierarchicalMatmul, TiledMatmul)):
        return IntegerGroupedMatmul(recipe)
    # Other explicit analytical or source planners retain their own program.
    return recipe
