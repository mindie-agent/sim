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
from .matrix_mapping import MappedMatmul, footprint
from .matrix import operand_transfer
from .ownership import read_role
from .program import KernelProgram, ParallelProgram, PipelineProgram, Stage


@lru_cache(maxsize=1024)
def _cycle(load, steps, count, slots, compute_slots, tail_load, tail_steps):
    return matrix_cycle(load, steps, count, slots, compute_slots,
                        tail_load_ms=tail_load, tail_micro_steps=tail_steps)


@dataclass(frozen=True)
class SteadyMatmul(MappedMatmul):
    """Keep exact nominal K-loop dependencies and a compact contended body."""

    compact_output_tiles: bool = True

    def lower(self, context, *, mapping=None, owner_tiles=None, output=None,
              directional_mte1=False):
        if mapping is None:
            recipe, record = self.select(context)
        else:
            recipe, record = mapping, {
                "basis": "declared_source_owner_mapping", "selected": asdict(mapping),
                "shape": list(context.op.local_shape[-3:]),
            }
        record.update({
            "execution_abstraction": "closed_form_k_pipeline",
            "overlap": "finite_K_micro_recurrence; inter_output_tiles_serial_per_owner",
            "contention": "dominant_resource_fair_coupled_fluid_progress",
            "buffer_model": "finite_selected_tile_reservation",
            "output_tile_execution": ("coupled_owner_run" if self.compact_output_tiles
                                      else "individual_serial_tiles"),
        })
        if context.record_mapping is not None:
            context.record_mapping(record)
        op, names = context.op, context.resource_names
        total_m, n, k = op.local_shape[-3:]
        if not total_m:
            return KernelProgram((Stage("setup", latency=context.setup_ms),),
                                 minimum_ms=context.minimum_ms)
        meta, cores = dict(op.metadata), context.core_count
        groups = int(meta.get("matrix_groups", 1))
        m = total_m // groups
        nm, nn = math.ceil(m / recipe.tile_m), math.ceil(n / recipe.tile_n)
        count, tail = divmod(k, recipe.tile_k)
        epilogue = MatrixEpilogue.from_op(op)
        inputs, weights = epilogue.operands(op)
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
                        context.rates[key] * (len(stage.placement[key]) / context.resource_units.get(
                            next(role for role, resource in names.items() if resource == key),
                            cores) if key in stage.placement else 1))
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
                            for pool, amount in footprint(context, recipe).items()
                            if pool in context.pool_names and amount)
            stages = [
                Stage("reserve", acquire=buffers),
                Stage("load", load0.work, after=("reserve",), placement=load0.placement),
                Stage("steady", work, after=("load",), placement=placement, coupled=True,
                      caps={key: amount / span for key, amount in work.items()} if span else {}),
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
                predecessor = "store"
            stages.append(Stage("release", after=(predecessor,), free=buffers))
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
            load, store = first[1], last[-2]
            for stage in (load, store):
                total_span -= duration(stage)
                for key, value in stage.work.items():
                    work[key] -= value
            work = {key: value for key, value in work.items() if value > 0}
            # Per-owner outputs remain nominally serial. External contention
            # shares their coupled fluid body, just like the inner K pipeline.
            caps = {key: value / total_span for key, value in work.items()}
            return KernelProgram((
                first[0],
                load,
                Stage("owner_run", work, after=("load",), placement=placement,
                      coupled=True, caps=caps),
                Stage("store", store.work, after=("owner_run",), placement=store.placement),
                Stage("release", after=("store",), free=first[0].acquire),
            ))

        owners = []
        counts = ((groups * nm * nn - 1 - core) // cores + 1
                  for core in range(min(cores, groups * nm * nn))) if owner_tiles is None else (
                      count for count, _ in owner_tiles)
        for core, tiles in enumerate(counts):
            if not tiles:
                continue
            source = None if owner_tiles is None else iter(owner_tiles[core][1])
            if self.compact_output_tiles and tiles > 1:
                owners.append(owner_run(core, tiles, source))
            else:
                owners.append(PipelineProgram(
                    tiles, 1, lambda index, core=core, source=source:
                    build(core, index, next(source) if source is not None else None)))
        return KernelProgram((
            Stage("setup", latency=context.setup_ms),
            Stage("owners", after=("setup",), program=ParallelProgram(tuple(owners))),
        ), minimum_ms=context.minimum_ms)
