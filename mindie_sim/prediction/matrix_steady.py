"""Compact finite matrix pipelines with resource-coupled steady progress.

[Role] Compress regular K-panel recurrences without removing their overlap.
[Invariants] Retain selected mapping, physical work, owner placement and finite
  tile buffers. The nominal recurrence preserves refill/consumer slot limits.
  External contention uses fluid backpressure; output tiles are serialized per
  owner. This is an explicit approximation, not arbitrary discrete equivalence.
[Timeouts & Locks] Bounded scalar recurrence and O(active owners) Engine state.
[Side-Effect Policy] Submit to the shared Engine; no observations or I/O.
"""
from dataclasses import asdict, dataclass, replace
from collections import Counter
from functools import lru_cache
import math

from mindie_sim.schema.common import dtype_bytes
from .matrix_cycle import matrix_cycle
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import (MappedMatmul, footprint, matrix_geometry,
                             physical_owner_rate)
from .matrix import operand_transfer
from .ownership import read_role
from .program import (KernelProgram, ParallelProgram, PipelineProgram,
                      PipelineTemplateProgram, Stage)

DEFAULT_SPLIT_K = 1
MS_PER_SECOND = 1000


@lru_cache(maxsize=1024)
def _cycle(load, steps, count, slots, compute_slots, tail_load, tail_steps):
    return matrix_cycle(load, steps, count, slots, compute_slots,
                        tail_load_ms=tail_load, tail_micro_steps=tail_steps)


def _tightened_caps(caps, extra):
    """Add declared ceilings without ever loosening an existing pipeline cap."""
    if not extra:
        return dict(caps)
    combined = dict(caps)
    for key, value in extra.items():
        combined[key] = min(combined[key], value) if key in combined else value
    return combined


@dataclass(frozen=True)
class SteadyMatmul(MappedMatmul):
    """Keep exact nominal K-loop dependencies and a compact contended body."""

    traffic_rewrite_capable: bool = True
    compact_output_tiles: bool = True
    #: Eligible one-tile owners reuse the finite template path.  Multi-tile
    #: owner_run stays its existing real KernelProgram branch.
    template_execution: bool = True
    #: Combine symmetric owners (identical tile count and shape sequence) into
    #: one cohort program: summed work/caps and the union of their units.  The
    #: Engine divides a placement over its units, so the cohort's per-unit
    #: demand, completion time and allocation accounting are the per-owner
    #: values; only task count falls.  ``False`` keeps the detailed per-owner
    #: programs, including asymmetric external contention between owners.
    owner_cohorts: bool = True

    def lower(self, context, *, mapping=None, owner_tiles=None, group_rows=None,
              output=None, directional_mte1=False):
        from .matrix_mapping import (freeze_owner_tiles, grouped_owner_rectangles,
                                     owner_records)
        owner_tiles = (freeze_owner_tiles(owner_tiles)
                       if owner_tiles is not None else None)
        include_record = context.record_mapping is not None
        if mapping is None:
            recipe, record = self.select(context,
                                         include_record=include_record)
        else:
            recipe = mapping
            record = ({
                "basis": "declared_source_owner_mapping", "selected": asdict(mapping),
                "shape": list(context.op.local_shape[-3:]),
            } if include_record else None)
        if owner_tiles is not None:
            sources = [tuple(tiles) for _, tiles in owner_tiles]
            explicit = owner_records(owner_tiles)
        elif group_rows is not None and len(context.op.local_shape) >= 3:
            sources = [shapes for _, shapes in grouped_owner_rectangles(
                group_rows, context.op.local_shape[-2], recipe.tile_m,
                recipe.tile_n, context.core_count)]
            explicit = owner_records(tuple(
                (len(shapes), tuple(shapes)) for shapes in sources))
        else:
            sources = None
            explicit = None
        geometry = matrix_geometry(
            context, recipe, owners=explicit, group_rows=group_rows)
        if include_record:
            record.update({
                "execution_abstraction": "closed_form_k_pipeline",
                "overlap": "finite_K_micro_recurrence; inter_output_tiles_serial_per_owner",
                "contention": "dominant_resource_fair_coupled_fluid_progress",
                "buffer_model": "finite_selected_tile_reservation",
                "output_tile_execution": ("coupled_owner_run" if self.compact_output_tiles
                                          else "individual_serial_tiles"),
                "geometry": geometry.to_dict(),
            })
        dram_rate = context.service_rate("mte2_aic_dram")
        if (getattr(recipe, "k_splits", DEFAULT_SPLIT_K) > DEFAULT_SPLIT_K
                and dram_rate is not None):
            if include_record:
                record.update({
                    "execution_abstraction": "finite_split_k",
                    "split_k": recipe.k_splits,
                    "partial_policy": "explicit_gm_fp32_partials_then_vector_merge",
                })
            from .matrix_splitk import splitk_program
            program = splitk_program(context, recipe.k_splits, recipe=recipe,
                                     geometry=geometry, record=record)
            if include_record:
                context.record_mapping(record)
            return program
        if include_record:
            context.record_mapping(record)
        op, names = context.op, context.resource_names
        total_m, n, k = geometry.local_shape
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        meta, cores = dict(op.metadata), context.core_count
        groups, m = geometry.groups, geometry.m_per_group
        nm, nn = geometry.nm, geometry.nn
        count, tail = divmod(k, recipe.tile_k)
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weights = epilogue.operands(op)
        single_m_hbm_cap = {}
        if nm == 1 and dram_rate is not None and "hbm" in names:
            single_m_hbm_cap[names["hbm"]] = (
                dram_rate / (MS_PER_SECOND * max(1, int(context.core_count))))
        operand_read = read_role(names)
        scale_read = read_role(names, aiv=0)
        width = dtype_bytes(meta.get("compute_format", op.dtype or "bf16"))
        compute = {key: value for key, value in context.work.items()
                   if key not in {names["hbm"], names.get("mte1"), names.get("mte2"),
                                  names.get("mte3"), names.get(operand_read),
                                  names.get(scale_read)}}
        if epilogue.operations_per_output:
            compute.pop(names["vector"], None)

        def transfer(role, amount, *, hbm=None):
            result = {names[role]: amount} if role in names and amount else {}
            if hbm:
                result[names["hbm"]] = result.get(names["hbm"], 0) + hbm
            return result

        def duration(stage):
            return max((amount / min(stage.caps.get(key, math.inf),
                        physical_owner_rate(
                            context, key, stage.placement.get(key)))
                        for key, amount in stage.work.items()), default=0)

        def build(core, index, shape=None):
            if shape is None:
                tile = index * cores + core
                a = min(recipe.tile_m, m - ((tile // nn) % nm) * recipe.tile_m)
                b = min(recipe.tile_n, n - (tile % nn) * recipe.tile_n)
            else:
                a, b = shape
            fraction = a * b / (total_m * n)

            def panel(kd):
                k_fraction = kd / k
                left = inputs * a / total_m * k_fraction
                right = weights * b / (n * groups) * k_fraction
                reads = ((inputs * fraction * k_fraction if recipe.reuse_inputs else left)
                         + (weights * fraction * k_fraction if recipe.reuse_weights else right))
                load = Stage("load", transfer(operand_read, left + right, hbm=reads),
                             placement=context.owned(core, cores, (operand_read,)))
                steps = []
                for mi in range(0, a, recipe.compute_m):
                    for ni in range(0, b, recipe.compute_n):
                        ai, bi = min(recipe.compute_m, a - mi), min(recipe.compute_n, b - ni)
                        inner_a = max(left * ai / a, ai * kd * width)
                        inner_b = max(right * bi / b, bi * kd * width)
                        operand = Stage("operand", operand_transfer(
                            context, inner_a, inner_b, directional=directional_mte1),
                                        placement=context.owned(core, cores, ("mte1",)))
                        matrix = Stage("matrix", {
                            key: value * ai * bi / (total_m * n) * k_fraction
                            for key, value in compute.items()},
                            placement=context.owned(core, cores, ("cube", "vector")))
                        steps.append((operand, matrix))
                return load, tuple(steps)

            full = panel(recipe.tile_k) if count else None
            ending = panel(tail) if tail else None
            first = full or ending
            last = ending or full
            load0 = first[0]
            final = last[1][-1][1]
            full_times = tuple((duration(o), duration(c)) for o, c in full[1]) if full else ()
            tail_times = tuple((duration(o), duration(c)) for o, c in ending[1]) if ending else None
            cycle = _cycle(duration(full[0]) if full else 0, full_times or tail_times, count,
                           recipe.slots, recipe.compute_slots,
                           duration(ending[0]) if ending else None, tail_times)
            work, placement = {}, {}
            for multiplicity, part in ((count, full), (int(bool(tail)), ending)):
                if part is None:
                    continue
                for stage in (part[0], *(stage for pair in part[1] for stage in pair)):
                    for key, value in stage.work.items():
                        work[key] = work.get(key, 0) + multiplicity * value
                    placement.update(stage.placement)
            for stage in (load0, final):
                for key, value in stage.work.items():
                    work[key] -= value
            work = {key: max(0, value) for key, value in work.items() if value > 0}
            steady = Stage("steady", work, placement=placement, coupled=True)
            span = max(duration(steady),
                       cycle.span_ms - duration(load0) - duration(final))
            buffers = tuple(context.allocate(f"tile{index}:{pool}", pool, amount, (core,))
                            for pool, amount in footprint(
                                context, recipe, geometry=geometry).items()
                            if pool in context.pool_names and amount)
            stages = [
                Stage("load", load0.work, acquire=buffers, placement=load0.placement),
                Stage("steady", work, after=("load",), placement=placement, coupled=True,
                      caps=_tightened_caps(
                          {key: amount / span for key, amount in work.items()}
                          if span else {}, single_m_hbm_cap)),
                Stage("matrix", final.work, after=("steady",), placement=final.placement),
            ]
            predecessor = "matrix"
            if output is not None:
                # Source publication consists of ordinary work stages. Native
                # cross-core chunk overlap stays outside this serial tile model.
                publication = output.lower(
                    context, core=core, rows=a, columns=b,
                    buffer_shape=(recipe.tile_m, recipe.tile_n), epilogue=epilogue,
                    reuse_inputs=recipe.reuse_inputs, reuse_weights=recipe.reuse_weights)
                stages.extend(replace(stage, after=stage.after or ("matrix",))
                              for stage in publication.stages)
                predecessor = publication.stages[-1].name
            elif epilogue.operations_per_output:
                copy, hbm = epilogue.scales(
                    a, b, total_m, n, groups, reuse_inputs=recipe.reuse_inputs,
                    reuse_weights=recipe.reuse_weights)
                stages.extend((
                    Stage("scales", transfer(scale_read, copy, hbm=hbm), after=("matrix",),
                          placement=context.owned(core, cores, (scale_read,))),
                    Stage("epilogue", {names["vector"]: epilogue.operations_per_output * a * b},
                          after=("scales",), placement=context.owned(core, cores, ("vector",))),
                ))
                predecessor = "epilogue"
            writes = op.mte3_bytes * fraction
            if output is None:
                stages.append(Stage("store", transfer("mte3", writes, hbm=writes),
                                    after=(predecessor,),
                                    placement=context.owned(core, cores, ("mte3",))))
            if buffers:
                stages[-1] = replace(stages[-1], free=stages[-1].free + buffers)
            return (Stage("tile", program=KernelProgram(tuple(stages))),)

        def owner_run(core, tiles, source_shapes=None):
            # Macro outputs have no inter-tile data edges. Count their at-most
            # four tail geometries without constructing a program per output.
            geometries, representatives = Counter(), {}
            first_shape = last_shape = None
            for index in range(tiles):
                if source_shapes is None:
                    tile = index * cores + core
                    shape = (min(recipe.tile_m, m - ((tile // nn) % nm) * recipe.tile_m),
                             min(recipe.tile_n, n - (tile % nn) * recipe.tile_n))
                else:
                    shape = next(source_shapes)
                    first_shape = first_shape or shape
                    last_shape = shape
                geometries[shape] += 1
                representatives.setdefault(shape, index)
            work, placement, total_span = {}, {}, 0
            first = build(core, 0, first_shape)[0].program.stages
            last = build(core, tiles - 1, last_shape)[0].program.stages
            for shape, multiplicity in geometries.items():
                stages = build(core, representatives[shape], shape)[0].program.stages
                for stage in stages:
                    total_span += multiplicity * duration(stage)
                    for key, value in stage.work.items():
                        work[key] = work.get(key, 0) + multiplicity * value
                    placement.update(stage.placement)
            load, store = first[0], last[-1]
            for stage in (load, store):
                total_span -= duration(stage)
                for key, value in stage.work.items():
                    work[key] -= value
            work = {key: value for key, value in work.items() if value > 0}
            # Per-owner outputs remain nominally serial. External contention
            # shares their coupled fluid body, just like the inner K pipeline.
            caps = _tightened_caps(
                {key: value / total_span for key, value in work.items()},
                single_m_hbm_cap)
            return KernelProgram((
                Stage("load", load.work, acquire=load.acquire, placement=load.placement),
                Stage("owner_run", work, after=("load",), placement=placement,
                      coupled=True, caps=caps),
                Stage("store", store.work, after=("owner_run",), free=load.acquire,
                      placement=store.placement),
            ))

        owners = []
        counts = (tuple(record[0] for record in geometry.owners)
                  if sources is None else tuple(len(shapes) for shapes in sources))
        if self.owner_cohorts:
            from .ownership import unit_group
            from .program import merge_programs
            #: Resource name -> owned-unit count, derived from the same
            #: declaration ``RecipeContext.owned`` consumes.
            name_units = {}
            for role, name in context.resource_names.items():
                name_units[name] = context.resource_units.get(
                    role, context.core_count)

            def shapes_for(core, tiles):
                if sources is not None:
                    return tuple(sources[core])
                return tuple(
                    (min(recipe.tile_m,
                         m - (((index * cores + core) // nn) % nm) * recipe.tile_m),
                     min(recipe.tile_n,
                         n - ((index * cores + core) % nn) * recipe.tile_n))
                    for index in range(tiles))

            def make(core, tiles, shapes):
                if self.compact_output_tiles and tiles > 1:
                    return owner_run(core, tiles, iter(shapes))
                if self.template_execution and tiles == 1:
                    return build(core, 0, shapes[0])[0].program
                return PipelineProgram(
                    tiles, 1, lambda index, core=core, shapes=shapes:
                    build(core, index, shapes[index]), homogeneous=True)

            def clone_stage(stage, core, reference):
                placement = {}
                for name, units in stage.placement.items():
                    declared = name_units.get(name)
                    if declared is None:
                        return None
                    if reference and unit_group(reference, cores, declared) != units:
                        return None
                    placement[name] = unit_group(core, cores, declared)
                acquire = tuple(replace(item, units=(core,))
                                if len(item.units) == 1 else item
                                for item in stage.acquire)
                free = tuple(replace(item, units=(core,))
                             if len(item.units) == 1 else item
                             for item in stage.free)
                nested = None
                if stage.program is not None:
                    nested = clone_program(stage.program, core, reference)
                    if nested is None:
                        return None
                return replace(stage, placement=placement, acquire=acquire,
                               free=free, program=nested)

            def clone_program(program, core, reference=False):
                if isinstance(program, KernelProgram):
                    stages = tuple(clone_stage(stage, core, reference)
                                   for stage in program.stages)
                    if any(stage is None for stage in stages):
                        return None
                    return replace(program, stages=stages)
                if isinstance(program, PipelineTemplateProgram):
                    template = program.template
                    phases = []
                    for phase in template.phases:
                        stages = tuple(clone_stage(stage, core, reference)
                                       for stage in phase.stages)
                        if any(stage is None for stage in stages):
                            return None
                        phases.append(replace(phase, stages=stages))
                    return PipelineTemplateProgram(
                        replace(template, phases=tuple(phases)))
                return None

            grouped: dict[tuple, list[int]] = {}
            for core, tiles in enumerate(counts):
                if not tiles:
                    continue
                grouped.setdefault((tiles, shapes_for(core, tiles)),
                                   []).append(core)
            for (tiles, shapes), members in grouped.items():
                if len(members) == 1:
                    owners.append(make(members[0], tiles, shapes))
                    continue
                # One real lowering per symmetric cohort: identical tile count
                # and per-index shape sequence means identical work, caps and
                # allocation sizes; only the owned units differ.
                representative = make(members[0], tiles, shapes)
                clones = [representative]
                for core in members[1:]:
                    clone = clone_program(representative, core)
                    if clone is None:
                        break
                    clones.append(clone)
                merged = None
                if len(clones) == len(members):
                    merged = merge_programs(
                        clones, scale_coupled_cohort=(tiles > 1))
                if merged is not None:
                    owners.append(merged)
                else:
                    owners.extend(make(core, tiles, shapes) for core in members)
        else:
            for core, tiles in enumerate(counts):
                if not tiles:
                    continue
                source = None if sources is None else iter(sources[core])
                if self.compact_output_tiles and tiles > 1:
                    owners.append(owner_run(core, tiles, source))
                elif source is None:
                    owners.append(PipelineProgram(
                        tiles, 1, lambda index, core=core: build(core, index, None),
                        homogeneous=True))
                else:
                    seq = tuple(sources[core])
                    owners.append(PipelineProgram(
                        tiles, 1, lambda index, core=core, seq=seq:
                        build(core, index, seq[index]), homogeneous=True))
            if self.template_execution:
                owners = [
                    PipelineTemplateProgram(pipe.template(
                        owner=owner, panels_per_output=1))
                    if isinstance(pipe, PipelineProgram) and pipe.count == 1 else pipe
                    for owner, pipe in enumerate(owners)]
        stages = (
            (Stage("setup", latency=context.setup_ms),
             Stage("owners", after=("setup",), program=ParallelProgram(tuple(owners))))
            if context.setup_ms
            else (Stage("owners", program=ParallelProgram(tuple(owners))),)
        )
        return KernelProgram(stages, minimum_ms=context.minimum_ms)
