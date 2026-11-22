"""Compact finite matrix pipelines with resource-coupled steady progress.

[Role] Compress regular K-panel recurrences without removing their overlap.
[Invariants] Retain selected mapping, physical work, owner placement and finite
  tile buffers. The nominal recurrence preserves refill/consumer slot limits.
  External contention uses fluid backpressure; output tiles are serialized per
  owner. This is an explicit approximation, not arbitrary discrete equivalence.
[Timeouts & Locks] Bounded scalar recurrence and O(active owners) Engine state.
[Side-Effect Policy] Submit to the shared Engine; no observations or I/O.
"""
from collections import Counter
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import math

from mindie_sim.schema.common import dtype_bytes
from .matrix_cycle import matrix_cycle
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import (MappedMatmul, footprint, matrix_geometry,
                             physical_owner_rate)
from .matrix_mapping_owners import (
    _FRACTAL_ROWS,
    matrix_kernel_floor_ms,
    matrix_kernel_setup_ms,
    primitive_l2_params,
    primitive_nd2nz_mte2_rate_bps_per_aic,
    primitive_wave_cold_hbm_per_owner_bps,
)
from .matrix import directional_mte1_available, operand_transfer
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import (KernelProgram, ParallelProgram, PipelineProgram,
                      PipelineTemplateProgram, Stage)

DEFAULT_SPLIT_K = 1
MS_PER_SECOND = 1000
FP32_BYTES = 4
#: Minimum L0C accumulator capacity (256 KiB) required for implicit Split-K lowering.
SPLIT_K_MIN_L0C_BYTES = 256 * 1024
#: Upstream CalL1Tiling maximum double-buffered L1 pipeline depth (4 slots).
L1_MAX_PIPELINE_SLOTS = 4


def _resolve_calibrated_hbm_cap(
    context, recipe, geometry, *, dram_rate, width, hbm_read_scale, compulsory_miss,
):
    """Per-owner cold-HBM cap from calibrated P01/P05 wave-summed supply."""
    names = context.resource_names
    if "hbm" not in names or hbm_read_scale <= 0.0:
        return {}
    if getattr(context, "primitive_table", None) is None:
        if geometry.nm == 1 and dram_rate is not None:
            return {names["hbm"]: dram_rate / (MS_PER_SECOND * max(1, int(context.core_count)))}
        return {}
    _, n, k = geometry.local_shape
    active_groups = max(1, sum(1 for r in geometry.group_rows if r > 0))
    cold_tiles = (
        max(1, int(geometry.nn) * active_groups)
        if compulsory_miss else max(1, int(geometry.tiles))
    )
    row_bytes = int(min(recipe.tile_n, n) * width)
    stride_bytes = int(n * width)
    is_grouped = geometry.groups > 1
    active_cold = max(1, min(int(context.core_count), cold_tiles))
    coalesced = is_grouped and active_cold >= max(1, int(geometry.nn))
    per_owner_bps = primitive_wave_cold_hbm_per_owner_bps(
        context, cold_tiles=cold_tiles, row_bytes=row_bytes,
        stride_bytes=stride_bytes, is_grouped=is_grouped,
        is_full_row_coalesced=coalesced, nn_tiles=int(geometry.nn),
        m_tiles=int(geometry.nm), panel_rows=max(1, int(k)),
    )
    if per_owner_bps is None or per_owner_bps <= 0.0:
        if geometry.nm == 1 and dram_rate is not None:
            return {names["hbm"]: dram_rate / (MS_PER_SECOND * max(1, int(context.core_count)))}
        return {}
    return {names["hbm"]: per_owner_bps / MS_PER_SECOND}


def _grouped_owner_rectangles_with_first_m(group_rows, n, tm, tn, cores):
    """Group-major owner rectangles annotated with ``(a, b, is_first_m)``."""
    owners = [[] for _ in range(int(cores))]
    index = 0
    for row in group_rows:
        row = int(row)
        for start in range(0, row, int(tm)):
            a = min(int(tm), row - start)
            first_m = start == 0
            for column in range(0, int(n), int(tn)):
                b = min(int(tn), int(n) - column)
                owners[index % int(cores)].append((a, b, first_m))
                index += 1
    return tuple(tuple(shapes) for shapes in owners)


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


def _build_steady_panel(
    context, recipe, *, core, cores, a, b, kd, k, total_m, n, groups, inputs, weights,
    fraction, width, compulsory_miss, first_m, hbm_read_scale, operand_read,
    load_stage_caps, directional_mte1, compute, names, pt, transfer,
):
    k_fraction = kd / k
    left = inputs * a / total_m * k_fraction
    right = weights * b / (n * groups) * k_fraction
    if compulsory_miss and first_m is not None:
        weight_reads = right if first_m else 0.0
    else:
        weight_reads = weights * fraction * k_fraction if recipe.reuse_weights else right
    reads = (
        (inputs * fraction * k_fraction if recipe.reuse_inputs else left)
        + weight_reads
    ) * hbm_read_scale
    load = Stage("load", transfer(operand_read, left + right, hbm=reads),
                 placement=context.owned(core, cores, (operand_read,)),
                 caps=load_stage_caps)
    steps = []
    cube_key = names.get("cube")
    for mi in range(0, a, recipe.compute_m):
        for ni in range(0, b, recipe.compute_n):
            ai, bi = min(recipe.compute_m, a - mi), min(recipe.compute_n, b - ni)
            inner_a = max(left * ai / a, ai * kd * width)
            inner_b = max(right * bi / b, bi * kd * width)
            operand = Stage("operand", operand_transfer(
                context, inner_a, inner_b, directional=directional_mte1),
                            placement=context.owned(core, cores, ("mte1",)))
            mat_work = {
                key: value * ai * bi / (total_m * n) * k_fraction
                for key, value in compute.items()
            }
            mat_caps = {}
            if pt is not None and cube_key in mat_work and ai < _FRACTAL_ROWS:
                per_core_cube = physical_owner_rate(
                    context, cube_key, context.owned(core, cores, ("cube",)).get(cube_key))
                mat_caps[cube_key] = per_core_cube * (float(ai) / float(_FRACTAL_ROWS))
            matrix = Stage("matrix", mat_work,
                           placement=context.owned(core, cores, ("cube", "vector")),
                           caps=mat_caps)
            steps.append((operand, matrix))
    return load, tuple(steps)


def _append_tile_epilogue(
    stages, context, recipe, *, core, cores, a, b, total_m, n, groups, fraction,
    op, output, epilogue, scale_read, names, buffers, transfer,
):
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


def _make_tile_builder(context, recipe, geometry, dram_rate, output, directional_mte1, duration):
    op, names = context.op, context.resource_names
    total_m, n, k = geometry.local_shape
    meta, cores = dict(op.metadata), context.core_count
    groups, m = geometry.groups, geometry.m_per_group
    nm, nn = geometry.nm, geometry.nn
    count, tail = divmod(k, recipe.tile_k)
    epilogue = MatrixEpilogue.from_op(op)
    inputs, weights = epilogue.operands(op)
    full_hbm_reads = (
        (inputs * (1 if recipe.reuse_inputs else nn))
        + (weights * (1 if recipe.reuse_weights else nm))
    )
    admitted_l2_hit = max(0.0, float(meta.get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
    hbm_read_scale = (
        max(0.0, 1.0 - min(admitted_l2_hit, full_hbm_reads) / full_hbm_reads)
        if (admitted_l2_hit > 0.0 and full_hbm_reads > 0.0) else 1.0
    )
    width = dtype_bytes(meta.get("compute_format", op.dtype or "bf16"))
    pt = getattr(context, "primitive_table", None)
    l2_bytes, _ = primitive_l2_params(context)
    # Compulsory-miss placement: when B is reused across M-tiles, charge cold
    # HBM weight miss on the first M-tile (m_idx == 0) while subsequent M-tiles
    # hit L2. Total HBM bytes are identically conserved.
    compulsory_miss = (
        pt is not None and recipe.reuse_weights and hbm_read_scale > 0.0
        and (groups > 1 or (nm > 1 and (l2_bytes <= 0 or weights <= l2_bytes)))
    )
    single_m_hbm_cap = _resolve_calibrated_hbm_cap(
        context, recipe, geometry, dram_rate=dram_rate, width=width,
        hbm_read_scale=hbm_read_scale, compulsory_miss=compulsory_miss,
    )
    active_owners = max(1, min(int(cores), int(geometry.tiles)))
    weight_layout = str(meta.get("weight_layout", "ND"))
    operand_read = read_role(names)
    scale_read = read_role(names, aiv=0)
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

    def build(core, index, shape=None):
        first_m = None
        if shape is None:
            tile = index * cores + core
            m_idx = (tile // nn) % nm
            a = min(recipe.tile_m, m - m_idx * recipe.tile_m)
            b = min(recipe.tile_n, n - (tile % nn) * recipe.tile_n)
            first_m = m_idx == 0
        elif len(shape) == 3:
            a, b, first_m = shape
        else:
            a, b = shape
        fraction = a * b / (total_m * n)
        tile_mte2_cap = {}
        if pt is not None and operand_read in names:
            eff_bps = primitive_nd2nz_mte2_rate_bps_per_aic(
                context, active_owners,
                inputs * a / total_m, weights * b / (n * groups),
                weight_layout=weight_layout,
                row_bytes=int(min(b, n) * width),
                stride_bytes=int(n * width),
            )
            if eff_bps is not None and eff_bps > 0.0:
                tile_mte2_cap = {names[operand_read]: eff_bps / MS_PER_SECOND}
        tile_load_caps = _tightened_caps(single_m_hbm_cap, tile_mte2_cap)
        load_stage_caps = tile_load_caps if pt is not None else {}

        def panel(kd):
            return _build_steady_panel(
                context, recipe, core=core, cores=cores, a=a, b=b, kd=kd, k=k,
                total_m=total_m, n=n, groups=groups, inputs=inputs, weights=weights,
                fraction=fraction, width=width, compulsory_miss=compulsory_miss,
                first_m=first_m, hbm_read_scale=hbm_read_scale,
                operand_read=operand_read, load_stage_caps=load_stage_caps,
                directional_mte1=directional_mte1, compute=compute, names=names,
                pt=pt, transfer=transfer,
            )

        full = panel(recipe.tile_k) if count else None
        ending = panel(tail) if tail else None
        first = full or ending
        last = ending or full
        load0 = first[0]
        final = last[1][-1][1]
        full_times = tuple((duration(o), duration(c)) for o, c in full[1]) if full else ()
        tail_times = tuple((duration(o), duration(c)) for o, c in ending[1]) if ending else None
        eff_slots = recipe.slots
        if pt is not None:
            l1_cap = int((context.pool_capacities or {}).get("l1", 0))
            panel_l1 = (
                (inputs * a / total_m + weights * b / (n * groups))
                * (min(k, recipe.tile_k) / float(max(1, k)))
            )
            if l1_cap > 0 and panel_l1 > 0.0:
                eff_slots = max(recipe.slots, min(L1_MAX_PIPELINE_SLOTS, int(l1_cap // panel_l1)))
        cycle = _cycle(duration(full[0]) if full else 0, full_times or tail_times, count,
                       eff_slots, recipe.compute_slots,
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
        steady = Stage("steady", work, placement=placement, coupled=True, caps=load_stage_caps)
        span = max(duration(steady),
                   cycle.span_ms - duration(load0) - duration(final))
        buffers = tuple(context.allocate(f"tile{index}:{pool}", pool, amount, (core,))
                        for pool, amount in footprint(
                            context, recipe, geometry=geometry).items()
                        if pool in context.pool_names and amount)
        stages = [
            Stage("load", load0.work, acquire=buffers, placement=load0.placement,
                  caps=load0.caps),
            Stage("steady", work, after=("load",), placement=placement, coupled=True,
                  caps=_tightened_caps(
                      {key: amount / span for key, amount in work.items()}
                      if span else {}, tile_load_caps)),
            Stage("matrix", final.work, after=("steady",), placement=final.placement,
                  caps=final.caps),
        ]
        _append_tile_epilogue(
            stages, context, recipe, core=core, cores=cores, a=a, b=b,
            total_m=total_m, n=n, groups=groups, fraction=fraction, op=op,
            output=output, epilogue=epilogue, scale_read=scale_read, names=names,
            buffers=buffers, transfer=transfer,
        )
        return (Stage("tile", program=KernelProgram(tuple(stages))),)

    return build, single_m_hbm_cap, compulsory_miss


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

    def _build_owner_run(self, core, tiles, source_shapes, *, cores, m, n, nm, nn, recipe, build, duration, single_m_hbm_cap, compulsory_miss=False, continuous_tiles=False):
        geometries, representatives = Counter(), {}
        first_shape = last_shape = None
        for index in range(tiles):
            if source_shapes is None:
                tile = index * cores + core
                m_idx = (tile // nn) % nm
                a = min(recipe.tile_m, m - m_idx * recipe.tile_m)
                b = min(recipe.tile_n, n - (tile % nn) * recipe.tile_n)
                shape = (a, b, m_idx == 0) if compulsory_miss else (a, b)
            else:
                shape = next(source_shapes)
            first_shape = first_shape or shape
            last_shape = shape
            geometries[shape] += 1
            representatives.setdefault(shape, index)
        work, placement, total_span = {}, {}, 0.0
        first = build(core, 0, first_shape)[0].program.stages
        last = build(core, tiles - 1, last_shape)[0].program.stages
        for shape, multiplicity in geometries.items():
            stages = build(core, representatives[shape], shape)[0].program.stages
            if continuous_tiles:
                # Double-buffered L0C + multi-slot L1 overlap tile t's K-drain
                # and Fixpipe store with tile t+1's MTE2 loads. Sum exact stage
                # work while pricing each tile at max(K-body pipes, store).
                k_work, k_place = {}, {}
                for stage in stages:
                    for key, value in stage.work.items():
                        work[key] = work.get(key, 0) + multiplicity * value
                    placement.update(stage.placement)
                    if stage.name != "store":
                        for key, value in stage.work.items():
                            k_work[key] = k_work.get(key, 0) + value
                        k_place.update(stage.placement)
                tile_body = Stage("tile_body", k_work, placement=k_place,
                                  caps=stages[0].caps)
                total_span += multiplicity * max(duration(tile_body), duration(stages[-1]))
            else:
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
        if continuous_tiles:
            # First tile's load0 and last tile's final+store stay outside the
            # pipelined interior, so add back the one final drain+store once.
            total_span += duration(last[-2]) + duration(store)
        work = {key: value for key, value in work.items() if value > 0}
        caps = _tightened_caps(
            {key: value / total_span for key, value in work.items()} if total_span > 0 else {},
            single_m_hbm_cap)
        return KernelProgram((
            Stage("load", load.work, acquire=load.acquire, placement=load.placement,
                  caps=load.caps),
            Stage("owner_run", work, after=("load",), placement=placement,
                  coupled=True, caps=caps),
            Stage("store", store.work, after=("owner_run",), free=load.acquire,
                  placement=store.placement),
        ))

    def lower(self, context, *, mapping=None, owner_tiles=None, group_rows=None,
              output=None, directional_mte1=None):
        from .matrix_mapping import (freeze_owner_tiles, grouped_owner_rectangles,
                                     owner_records)
        pt = getattr(context, "primitive_table", None)
        if directional_mte1 is None:
            directional_mte1 = pt is not None and directional_mte1_available(context)
        owner_tiles = (freeze_owner_tiles(owner_tiles)
                       if owner_tiles is not None else None)
        include_record = context.record_mapping is not None
        if mapping is None:
            recipe, record = self.select(context,
                                         include_record=include_record,
                                         group_rows=group_rows)
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
        l0c_cap = int((context.pool_capacities or {}).get("l0c", 0))
        tile_l0c = int(recipe.tile_m) * int(recipe.tile_n) * FP32_BYTES
        supports_implicit_splitk = (
            l0c_cap >= tile_l0c if pt is not None else l0c_cap >= SPLIT_K_MIN_L0C_BYTES
        )
        if (getattr(recipe, "k_splits", DEFAULT_SPLIT_K) > DEFAULT_SPLIT_K
                and dram_rate is not None and supports_implicit_splitk):
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
        declared_groups = int(dict(context.op.metadata).get("declared_group_count", 1) or 1)
        total_setup_ms = float(context.setup_ms or 0.0) + matrix_kernel_setup_ms(
            context, is_streamk=False, declared_groups=declared_groups)
        eff_min_ms = max(float(context.minimum_ms or 0.0), matrix_kernel_floor_ms(context))
        total_m, n, _ = geometry.local_shape
        if not total_m:
            return KernelProgram((Stage("setup", latency=total_setup_ms),),
                                 minimum_ms=eff_min_ms)
        cores, m = context.core_count, geometry.m_per_group
        nm, nn = geometry.nm, geometry.nn

        def duration(stage):
            return max((amount / min(stage.caps.get(key, math.inf),
                        physical_owner_rate(
                            context, key, stage.placement.get(key)))
                        for key, amount in stage.work.items()), default=0)

        build, single_m_hbm_cap, compulsory_miss = _make_tile_builder(
            context, recipe, geometry, dram_rate, output, directional_mte1, duration,
        )
        if (compulsory_miss and owner_tiles is None
                and group_rows is not None and len(context.op.local_shape) >= 3):
            annotated = _grouped_owner_rectangles_with_first_m(
                group_rows, n, recipe.tile_m, recipe.tile_n, cores)
            sources = [shapes for shapes in annotated if shapes]

        continuous_tiles = (
            pt is not None and output is None
            and not MatrixEpilogue.from_op(context.op).operations_per_output
        )

        def owner_run(core, tiles, source_shapes=None):
            return self._build_owner_run(
                core, tiles, source_shapes, cores=cores, m=m, n=n, nm=nm, nn=nn,
                recipe=recipe, build=build, duration=duration,
                single_m_hbm_cap=single_m_hbm_cap,
                compulsory_miss=compulsory_miss,
                continuous_tiles=continuous_tiles)

        owners = []
        counts = (tuple(record[0] for record in geometry.owners)
                  if sources is None else tuple(len(shapes) for shapes in sources))
        if self.owner_cohorts:
            owners = self._cohort_owner_programs(
                context, recipe, geometry, sources, counts, build, owner_run,
                compulsory_miss=compulsory_miss,
            )
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
            (Stage("setup", latency=total_setup_ms),
             Stage("owners", after=("setup",), program=ParallelProgram(tuple(owners))))
            if total_setup_ms
            else (Stage("owners", program=ParallelProgram(tuple(owners))),)
        )
        return KernelProgram(stages, minimum_ms=eff_min_ms)

    def _cohort_owner_programs(self, context, recipe, geometry, sources, counts, build, owner_run, *, compulsory_miss=False):
        _, n, _ = geometry.local_shape
        cores, m = context.core_count, geometry.m_per_group
        nm, nn = geometry.nm, geometry.nn
        owners = []
        from mindie_sim.prediction.ownership import unit_group
        from mindie_sim.prediction.program import merge_programs
        #: Resource name -> owned-unit count, derived from the same
        #: declaration ``RecipeContext.owned`` consumes.
        name_units = {}
        for role, name in context.resource_names.items():
            name_units[name] = context.resource_units.get(
                role, context.core_count)

        def shapes_for(core, tiles):
            if sources is not None:
                return tuple(sources[core])
            result = []
            for index in range(tiles):
                tile = index * cores + core
                m_idx = (tile // nn) % nm
                a = min(recipe.tile_m, m - m_idx * recipe.tile_m)
                b = min(recipe.tile_n, n - (tile % nn) * recipe.tile_n)
                result.append((a, b, m_idx == 0) if compulsory_miss else (a, b))
            return tuple(result)

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
        return owners
