"""Symmetric cohort coalescing, stage folding, and execution lowering.

[Role] Merge structurally identical programs into explicit cohort stages
  carrying ``cohort_widened`` and per-member ``cohort_members``, fold
  zero-work wrappers and strictly serial integer compute chains, and expose
  ``coalesce_execution_program`` and ``split_cohort_kernel``.
[Invariants] Every merged cohort explicitly records which resources are
  disjointly widened versus shared across members; uncoupled stages mixing
  placed and unplaced work never silently coalesce into distorted shares.
[Timeouts & Locks] Materialization windows are bounded by
  ``PIPELINE_COHORT_WINDOW`` and ``unroll_limit``; no I/O or locks.
[Side-Effect Policy] Pure program transformation; ``_FloorParallelProgram``
  submits tasks only to the caller's ``Engine``.
"""
from __future__ import annotations

from dataclasses import replace

from .pipeline import (
    PIPELINE_COHORT_WINDOW,
    PipelinePhase,
    PipelineProgram,
    PipelineTemplate,
    PipelineTemplateProgram,
)
from .stages import KernelProgram, Stage
from ..engine import Allocation


def _units_mergeable(rows):
    """True when per-program unit tuples are identical or pairwise disjoint."""
    if len({tuple(units) for units in rows}) == 1:
        return True
    seen: set[int] = set()
    for units in rows:
        if any(unit in seen for unit in units):
            return False
        seen.update(units)
    return True


def _allocation_signature(items):
    return tuple(sorted((item.name, item.pool, item.size, len(item.units)) for item in items))


def _stage_signature(stage, *, widen_pipelines: bool = False):
    program = (
        None if stage.program is None
        else program_signature(stage.program, widen_pipelines=widen_pipelines)
    )
    if stage.program is not None and program is None:
        return None
    return (
        stage.name, stage.after, stage.previous, stage.previous_lags,
        stage.latency, stage.coupled, stage.priority, stage.cohort_size,
        stage.keep_empty, stage.cohort_widened, stage.latency_from_first_service,
        stage.deliveries,
        tuple(sorted(stage.metadata.items())),
        tuple(sorted(stage.work_units.items())),
        tuple(sorted(stage.work.items())),
        tuple(sorted(stage.caps.items())),
        tuple(sorted((key, len(units)) for key, units in stage.placement.items())),
        _allocation_signature(stage.acquire),
        _allocation_signature(stage.free),
        program,
        None if stage.fragment is None else stage.fragment.name,
    )


def _template_signature(program):
    template = program.template
    if (
        template.setup or template.drain or template.bindings
        or template.setup_ms or template.minimum_ms
        or template.indexed_panels is not None or template.motif_repeats != 1
    ):
        return None
    phases = []
    for phase in template.phases:
        if phase.residue_stages or len(phase.stages) != 1:
            return None
        signature = _stage_signature(phase.stages[0])
        if signature is None:
            return None
        phases.append((phase.kind, phase.count, phase.key, signature))
    return ("template", template.slots, tuple(phases))


def _pipeline_signature(program, *, widen_pipelines: bool = False):
    """Structural identity of one bounded non-homogeneous tile window."""
    if program.homogeneous or program.setup_ms or program.minimum_ms:
        return None
    if program.count < 0 or program.count > PIPELINE_COHORT_WINDOW:
        return None
    tiles = []
    for index in range(program.count):
        row = []
        for stage in tuple(program.tile(index)):
            signature = _stage_signature(stage, widen_pipelines=widen_pipelines)
            if signature is None:
                return None
            row.append(signature)
        tiles.append(tuple(row))
    return ("pipeline", program.count, program.slots, tuple(tiles))


def program_signature(program, *, widen_pipelines: bool = False):
    """Unit-agnostic structural identity for cohort coalescing, or ``None``."""
    if isinstance(program, KernelProgram):
        stages = []
        for stage in program.stages:
            signature = _stage_signature(stage, widen_pipelines=widen_pipelines)
            if signature is None:
                return None
            stages.append(signature)
        return ("kernel", program.minimum_ms, tuple(stages))
    if isinstance(program, PipelineTemplateProgram):
        return _template_signature(program)
    if widen_pipelines and isinstance(program, PipelineProgram):
        return _pipeline_signature(program, widen_pipelines=True)
    return None


def _merge_allocations(groups):
    """Combine equal-named allocations; ``None`` when identities do not match."""
    order, merged = [], {}
    for items in groups:
        for item in items:
            row = merged.get(item.name)
            if row is None:
                merged[item.name] = [item.pool, item.size, list(item.units), set(item.units)]
                order.append(item.name)
                continue
            if row[0] != item.pool or row[1] != item.size:
                return None
            if any(unit in row[3] for unit in item.units):
                return None
            row[2].extend(item.units)
            row[3].update(item.units)
    return tuple(
        Allocation(name, merged[name][0], merged[name][1], tuple(merged[name][2]))
        for name in order
    )


def _merge_stages(
    stages,
    *,
    scale_coupled_cohort: bool = True,
    widen_pipelines: bool = False,
    allow_disjoint_placement: bool = True,
):
    """Combine N structurally identical stages into one cohort stage."""
    if len(stages) < 2:
        return stages[0] if stages else None
    first = stages[0]
    if any(stage.fragment is not None for stage in stages):
        return None
    if len({
        (
            stage.latency, stage.name, stage.coupled, stage.priority,
            stage.keep_empty, stage.after, stage.previous, stage.previous_lags,
            stage.latency_from_first_service,
            tuple(sorted(stage.work_units.items())),
            tuple(sorted(stage.metadata.items())),
        )
        for stage in stages
    }) != 1:
        return None
    if any(stage.program is not None for stage in stages):
        if any(stage.program is None for stage in stages):
            return None
        if any(
            stage.work or stage.caps or stage.acquire or stage.free
            or stage.placement or stage.latency or stage.deliveries
            for stage in stages
        ):
            return None
        merged = merge_programs(
            [stage.program for stage in stages],
            scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=widen_pipelines,
        )
        if merged is None:
            return None
        return Stage(
            first.name, after=first.after, previous=first.previous,
            previous_lags=first.previous_lags,
            coupled=first.coupled, work_units=first.work_units,
            priority=first.priority, program=merged, keep_empty=first.keep_empty,
            metadata=first.metadata,
        )
    if any(set(stage.placement) != set(first.placement) for stage in stages):
        return None
    work = {key: sum(stage.work[key] for stage in stages) for key in first.work}
    caps = {key: sum(stage.caps[key] for stage in stages) for key in first.caps}
    active_work = {k for k, v in first.work.items() if v > 0.0}
    if not first.coupled and active_work:
        # Merging uncoupled work that shares any unplaced or common-placement
        # resource collapses fair-share contender multiplicity (e.g. 2+1 -> 1+1)
        # or coalesces distinct FIFO grants. Only allow uncoupled merge when
        # every active work resource has strictly disjoint per-member placement.
        if not active_work.issubset(first.placement):
            return None
        for key in active_work:
            rows_k = [tuple(stage.placement[key]) for stage in stages]
            if len(set(rows_k)) == 1:
                return None
    cohort_size = (
        sum(stage.cohort_size for stage in stages)
        if ((first.coupled and scale_coupled_cohort) or (not first.coupled and active_work))
        else first.cohort_size
    )
    has_unplaced_work = (
        not first.coupled and bool(first.placement)
        and bool(set(first.work) - set(first.placement))
    )
    placement: dict[str, tuple[int, ...]] = {}
    widened_keys: list[str] = []
    cohort_members: dict[str, tuple[tuple[int, ...], ...]] = {}
    for key in first.placement:
        rows = [tuple(stage.placement[key]) for stage in stages]
        if not _units_mergeable(rows):
            return None
        if len(set(rows)) == 1:
            # All members share the exact same unit tuple: keep NOT widened.
            placement[key] = rows[0]
            if cohort_size > 1:
                expanded: list[tuple[int, ...]] = []
                for stage in stages:
                    prev_m = stage.cohort_members.get(key)
                    if prev_m:
                        expanded.extend(prev_m)
                    else:
                        expanded.extend(rows[0] for _ in range(stage.cohort_size))
                cohort_members[key] = tuple(expanded)
            continue
        if not allow_disjoint_placement or has_unplaced_work:
            return None
        if len({len(r) for r in rows}) != 1:
            return None
        units: list[int] = []
        expanded_m: list[tuple[int, ...]] = []
        for stage, row in zip(stages, rows):
            units.extend(row)
            prev_m = stage.cohort_members.get(key)
            if prev_m:
                expanded_m.extend(prev_m)
            elif stage.cohort_size == 1:
                expanded_m.append(row)
            else:
                return None
        placement[key] = tuple(units)
        if cohort_size > 1:
            if len(expanded_m) != cohort_size or len({len(m) for m in expanded_m}) != 1:
                return None
            widened_keys.append(key)
            cohort_members[key] = tuple(expanded_m)
    acquire = _merge_allocations([stage.acquire for stage in stages])
    free = _merge_allocations([stage.free for stage in stages])
    if acquire is None or free is None:
        return None
    deliv_sum: dict[str, float] = {}
    for stage in stages:
        for cid, amt in stage.deliveries:
            deliv_sum[cid] = deliv_sum.get(cid, 0.0) + float(amt)
    return Stage(
        first.name, work, after=first.after, latency=first.latency,
        caps=caps, acquire=acquire, free=free, placement=placement,
        previous=first.previous, previous_lags=first.previous_lags,
        coupled=first.coupled, work_units=first.work_units,
        priority=first.priority, cohort_size=cohort_size, keep_empty=first.keep_empty,
        cohort_widened=tuple(widened_keys),
        cohort_members=cohort_members if widened_keys else {},
        metadata=first.metadata,
        deliveries=tuple(deliv_sum.items()),
        latency_from_first_service=first.latency_from_first_service,
    )


def _merge_pipelines(programs, *, scale_coupled_cohort: bool = True, widen_pipelines: bool = False):
    """Merge symmetric owners of one bounded, opaque tile pipeline."""
    first = programs[0]
    if len({
        (p.count, p.slots, p.setup_ms, p.minimum_ms, p.homogeneous, p.producer_bindings is not None)
        for p in programs
    }) != 1:
        return None
    if first.homogeneous or first.setup_ms or first.minimum_ms:
        return None
    if first.count < 0 or first.count > PIPELINE_COHORT_WINDOW:
        return None
    allow_disjoint = first.count <= 1
    tiles = []
    for index in range(first.count):
        rows = [tuple(program.tile(index)) for program in programs]
        if len({len(row) for row in rows}) != 1:
            return None
        stage_disjoint = allow_disjoint and len(rows[0]) <= 1
        stages = []
        for position in range(len(rows[0])):
            merged = _merge_stages(
                [row[position] for row in rows],
                scale_coupled_cohort=scale_coupled_cohort,
                widen_pipelines=widen_pipelines,
                allow_disjoint_placement=(
                    stage_disjoint and not any(r[position].acquire or r[position].free for r in rows)
                ),
            )
            if merged is None:
                return None
            stages.append(merged)
        tiles.append(tuple(stages))
    return PipelineProgram(
        first.count, first.slots, lambda index, tiles=tuple(tiles): tiles[index],
        first.minimum_ms, first.setup_ms, producer_bindings=first.producer_bindings,
    )


def _merge_kernels(programs, *, scale_coupled_cohort: bool = True, widen_pipelines: bool = False):
    first = programs[0]
    if len({p.minimum_ms for p in programs}) != 1 or len({len(p.stages) for p in programs}) != 1:
        return None
    allow_disjoint = (
        len(first.stages) <= 1 and not any(s.acquire or s.free for s in first.stages)
    )
    stages = []
    for index in range(len(first.stages)):
        merged = _merge_stages(
            [program.stages[index] for program in programs],
            scale_coupled_cohort=scale_coupled_cohort,
            widen_pipelines=widen_pipelines,
            allow_disjoint_placement=allow_disjoint,
        )
        if merged is None:
            return None
        stages.append(merged)
    return KernelProgram(tuple(stages), first.minimum_ms)


def _merge_wrapper_stages(
    stages, *, scale_coupled_cohort: bool = True, allow_disjoint_placement: bool = True,
):
    """Merge single-static wrapper stages that own one nested program."""
    if any(stage.program is None for stage in stages):
        return _merge_stages(
            stages,
            scale_coupled_cohort=scale_coupled_cohort,
            allow_disjoint_placement=allow_disjoint_placement,
        )
    if any(
        stage.work or stage.caps or stage.acquire or stage.free
        or stage.placement or stage.latency or stage.deliveries
        for stage in stages
    ):
        return None
    if len({
        (
            s.name, s.after, s.previous, s.previous_lags, s.coupled,
            s.priority, s.keep_empty, tuple(sorted(s.metadata.items())),
        )
        for s in stages
    }) != 1:
        return None
    merged = merge_programs(
        [stage.program for stage in stages], scale_coupled_cohort=scale_coupled_cohort,
    )
    if merged is None:
        return None
    first = stages[0]
    return Stage(
        first.name, after=first.after, previous=first.previous,
        previous_lags=first.previous_lags, program=merged,
        coupled=first.coupled, work_units=first.work_units, priority=first.priority,
        keep_empty=first.keep_empty, metadata=first.metadata,
    )


def _merge_templates(programs, *, scale_coupled_cohort: bool = True):
    first = programs[0].template
    if any(
        p.template.setup or p.template.drain or p.template.bindings
        or p.template.setup_ms or p.template.minimum_ms
        or p.template.indexed_panels is not None or p.template.motif_repeats != 1
        for p in programs
    ):
        return None
    if (
        len({p.template.slots for p in programs}) != 1
        or len({len(p.template.phases) for p in programs}) != 1
    ):
        return None
    total_tiles = sum(phase.count for phase in first.phases)
    allow_disjoint = (
        total_tiles <= 1
        and len(first.phases) <= 1
        and not any(s.acquire or s.free for phase in first.phases for s in phase.stages)
    )
    phases = []
    for index, phase in enumerate(first.phases):
        rows = [program.template.phases[index] for program in programs]
        if (
            len({(row.kind, row.count, row.key, len(row.stages)) for row in rows}) != 1
            or any(row.residue_stages for row in rows)
        ):
            return None
        stage_disjoint = allow_disjoint and len(phase.stages) <= 1
        merged_stages = []
        for stage_index in range(len(phase.stages)):
            merged = _merge_wrapper_stages(
                [row.stages[stage_index] for row in rows],
                scale_coupled_cohort=scale_coupled_cohort,
                allow_disjoint_placement=stage_disjoint,
            )
            if merged is None:
                return None
            merged_stages.append(merged)
        phases.append(PipelinePhase(phase.kind, tuple(merged_stages), phase.count, key=phase.key))
    return PipelineTemplateProgram(PipelineTemplate(
        first.owner, first.slots, tuple(phases),
        setup_ms=first.setup_ms, minimum_ms=first.minimum_ms,
        tile_factory_calls=first.tile_factory_calls,
        phase_objects=len(phases),
        stage_objects=sum(len(phase.stages) for phase in phases),
        static_phase_classes=len(phases),
        dynamic_instances=sum(phase.count for phase in phases),
    ))


def _collect_stage_footprint(stage, out: dict[tuple[str, str], set[int] | None]) -> bool:
    for res, amt in stage.work.items():
        if not amt:
            continue
        key = ("res", res)
        if res not in stage.placement:
            out[key] = None
        elif out.get(key, set()) is not None:
            out.setdefault(key, set()).update(stage.placement[res])
    for alloc in (*stage.acquire, *stage.free):
        key = ("pool", alloc.pool)
        if out.get(key, set()) is not None:
            out.setdefault(key, set()).update(alloc.units or (0,))
    if stage.program is not None:
        return _collect_program_footprint(stage.program, out)
    return True


def _collect_program_footprint(program, out: dict[tuple[str, str], set[int] | None]) -> bool:
    if isinstance(program, KernelProgram):
        return all(_collect_stage_footprint(stage, out) for stage in program.stages)
    if isinstance(program, PipelineTemplateProgram):
        tpl = program.template
        if tpl.indexed_panels is not None:
            return False
        stages = [*tpl.setup, *tpl.drain]
        for phase in tpl.phases:
            for group in (phase.residue_stages or (phase.stages,)):
                stages.extend(group)
        return all(_collect_stage_footprint(stage, out) for stage in stages)
    if isinstance(program, PipelineProgram):
        if program.homogeneous or program.count < 0 or program.count > PIPELINE_COHORT_WINDOW:
            return False
        for idx in range(program.count):
            for stage in tuple(program.tile(idx)):
                if not _collect_stage_footprint(stage, out):
                    return False
        return True
    return False


def _program_footprint(program) -> dict[tuple[str, str], set[int] | None] | None:
    out: dict[tuple[str, str], set[int] | None] = {}
    return out if _collect_program_footprint(program, out) else None


def _footprints_conflict(
    left: dict[tuple[str, str], set[int] | None] | None,
    right: dict[tuple[str, str], set[int] | None] | None,
) -> bool:
    if left is None or right is None:
        return True
    for res in left.keys() & right.keys():
        lu, ru = left[res], right[res]
        if lu is None or ru is None or not lu.isdisjoint(ru):
            return True
    return False


_MISSING_FOOTPRINT = object()


def _merge_footprint_into(
    acc: dict[tuple[str, str], set[int] | None],
    fp: dict[tuple[str, str], set[int] | None] | None,
) -> bool:
    """Fold ``fp`` into ``acc`` in place; return False iff ``fp`` is opaque (``None``)."""
    if fp is None:
        return False
    for res, units in fp.items():
        if units is None:
            acc[res] = None
        elif res not in acc:
            acc[res] = set(units)
        elif acc[res] is not None:
            acc[res].update(units)
    return True


def _last_signature_positions(sigs: list) -> dict | None:
    """Map each hashable non-None signature to its last index, or None if unhashable."""
    last: dict = {}
    try:
        for idx, sig in enumerate(sigs):
            if sig is not None:
                last[sig] = idx
    except TypeError:
        return None
    return last


def coalesce_programs(programs, *, widen_pipelines: bool = False):
    """Combine symmetric owners into cohort programs; keep the exact fallback."""
    programs = tuple(programs)
    n = len(programs)
    if n < 2:
        return programs
    sigs = [program_signature(p, widen_pipelines=widen_pipelines) for p in programs]
    sig_last = _last_signature_positions(sigs)
    footprints: list | None = None
    consumed = [False] * n
    result = []

    def _footprint_at(idx: int):
        nonlocal footprints
        if footprints is None:
            footprints = [_MISSING_FOOTPRINT] * n
        fp = footprints[idx]
        if fp is _MISSING_FOOTPRINT:
            fp = _program_footprint(programs[idx])
            footprints[idx] = fp
        return fp

    for i in range(n):
        if consumed[i]:
            continue
        sig_i = sigs[i]
        if sig_i is None or (sig_last is not None and sig_last[sig_i] <= i):
            consumed[i] = True
            result.append(programs[i])
            continue
        indices = [i]
        pending_intervening: list[int] = []
        intervening_fp: dict[tuple[str, str], set[int] | None] = {}
        intervening_has_any = False
        intervening_opaque = False
        stop = sig_last[sig_i] + 1 if sig_last is not None else n
        for j in range(i + 1, stop):
            if consumed[j]:
                continue
            if sigs[j] != sig_i:
                pending_intervening.append(j)
                continue
            if pending_intervening:
                intervening_has_any = True
                for k in pending_intervening:
                    if not _merge_footprint_into(intervening_fp, _footprint_at(k)):
                        intervening_opaque = True
                        break
                pending_intervening.clear()
            if intervening_opaque:
                break
            if intervening_has_any and _footprints_conflict(_footprint_at(j), intervening_fp):
                pending_intervening.append(j)
                continue
            indices.append(j)
        if len(indices) > 1:
            merged = merge_programs(
                [programs[idx] for idx in indices], widen_pipelines=widen_pipelines,
            )
            if merged is not None:
                for idx in indices:
                    consumed[idx] = True
                result.append(merged)
                continue
        consumed[i] = True
        result.append(programs[i])
    return tuple(result)


def merge_programs(programs, *, scale_coupled_cohort: bool = True, widen_pipelines: bool = False):
    """Merge one cohort of structurally identical programs, or ``None``."""
    if not programs:
        return None
    first = programs[0]
    if isinstance(first, KernelProgram):
        return _merge_kernels(
            programs, scale_coupled_cohort=scale_coupled_cohort, widen_pipelines=widen_pipelines
        )
    if isinstance(first, PipelineTemplateProgram):
        return _merge_templates(programs, scale_coupled_cohort=scale_coupled_cohort)
    if widen_pipelines and isinstance(first, PipelineProgram):
        return _merge_pipelines(
            programs, scale_coupled_cohort=scale_coupled_cohort, widen_pipelines=True
        )
    return None


def split_cohort_kernel(programs):
    """One program per cohort, keeping independently throttled stages split."""
    first = programs[0]
    if not isinstance(first, KernelProgram):
        return None
    signature = program_signature(first)
    if any(program_signature(program) != signature for program in programs[1:]):
        return None
    stages = first.stages
    position = {stage.name: index for index, stage in enumerate(stages)}
    split = [bool(stage.coupled or stage.caps) for stage in stages]
    if not any(split):
        return merge_programs(programs)
    count = len(programs)

    def refs(name, owner=None):
        if not split[position[name]]:
            return (name,)
        if owner is None:
            return tuple(f"{name}#{index}" for index in range(count))
        return (f"{name}#{owner}",)

    rebuilt = []
    for index, stage in enumerate(stages):
        after = tuple(dep for name in stage.after for dep in refs(name))
        previous = tuple(dep for name in stage.previous for dep in refs(name))
        previous_lags = tuple(
            (dep, lag) for name, lag in stage.previous_lags for dep in refs(name)
        )
        if not split[index]:
            merged = _merge_stages([program.stages[index] for program in programs])
            if merged is None:
                return None
            rebuilt.append(replace(
                merged, after=after, previous=previous, previous_lags=previous_lags,
            ))
            continue
        for owner in range(count):
            source = programs[owner].stages[index]
            if source.acquire or source.free or source.program is not None:
                return None
            rebuilt.append(replace(
                source, name=f"{stage.name}#{owner}",
                after=tuple(dep for name in stage.after for dep in refs(name, owner)),
                previous=tuple(dep for name in stage.previous for dep in refs(name, owner)),
                previous_lags=tuple(
                    (dep, lag) for name, lag in stage.previous_lags for dep in refs(name, owner)
                ),
            ))
    return KernelProgram(tuple(rebuilt), first.minimum_ms)
