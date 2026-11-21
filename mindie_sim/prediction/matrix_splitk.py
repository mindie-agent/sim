"""Explicit GM-partials Split-K lowering for underfilled output grids.

[Role] Partition every output tile's K extent across disjoint AIC K-slices,
  spill one FP32 partial accumulator per slice to GM, then reduce the partials
  on the vector pipe and store the final C tile once.
[Invariants] Cube work is issued exactly once: each task owns one output tile
  and one disjoint K slice with ``sum_j K_j = K``, so the split grid issues
  exactly ``2MNK`` matrix FLOPs. Partial write/read bytes and ``(s-1)MN`` merge
  adds are explicit issued work; declared operand/epilogue/final-store work is
  conserved across the same task grid. A declared service mode is bytes/second,
  so its per-owner Stage cap is ``rate/1000`` divided by the active owner count.
  A cap only tightens an existing bound.  The A5 narrow-N router adds an
  explicit FP32 workspace (one partial per K slice) plus its vector merge on
  top of the unchanged steady matrix pipeline; non-Split-K routers keep that
  pipeline alone.
[Timeouts & Locks] O(output_tiles * K_slices) bounded tasks; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .matrix_mapping import matrix_geometry
from .mechanism import PlannedRecipe
from .program import KernelProgram, ParallelProgram, RecipeContext, Stage

FP32_BYTES = 4
MS_PER_SECOND = 1000
DRAM_SERVICE_MODE = "mte2_aic_dram"
LOCAL_SHAPE_DIMS = 3
#: Narrow-N A5 BF16 gate projected onto the 32-AIC grid.  ``AscendGateLinear``
#: issues one 128x256 output tile per AIC wave; the free AIC budget splits K.
ROUTER_TILE_M = 128
ROUTER_TILE_N = 256
#: Split-K slices are nonempty integer K elements; a split grid may never price
#: more partials than the K extent can populate.
ROUTER_K_GRANULE = 1
ROUTER_PROLOGUE_DRAIN_CYCLES = 8250
ROUTER_NOMINAL_CLOCK_HZ = 1.65e9
A5_AIC_UNITS = 32


def _allocate(total: float, weights: list[float]) -> list[float]:
    """Allocate ``total`` by normalized weight, closing rounding last."""
    if not weights:
        return []
    scale = sum(weights)
    if scale <= 0:
        return [0.0 for _ in weights]
    normalized = [weight / scale for weight in weights]
    shares = [float(total) * weight for weight in normalized[:-1]]
    shares.append(float(total) - sum(shares))
    return shares


def _a5_aic_grid(geometry) -> bool:
    units = getattr(geometry, "resource_units", {}) or {}
    return (int(units.get("mte2_aic", 0)) >= A5_AIC_UNITS
            and int(getattr(geometry, "compute_units", 0) or 0) >= A5_AIC_UNITS)


def _router_split(op, geometry) -> bool:
    meta = dict(op.metadata)
    return (bool(meta.get("split_k_required"))
            and meta.get("weight_format") == "bf16"
            and meta.get("output_format") == "fp32"
            and _a5_aic_grid(geometry))


def router_output_tiles(m: int, n: int) -> int:
    """Declared 128x256 router tile grid; zero extents own no tile."""
    if m <= 0 or n <= 0:
        return 0
    return (-(-m // ROUTER_TILE_M)) * (-(-n // ROUTER_TILE_N))


def router_split_count(tiles: int, aics: int = A5_AIC_UNITS) -> int:
    """Split K only across the AIC budget left by the output tile grid."""
    if tiles <= 0:
        return 1
    return max(1, min(int(aics), -(-int(aics) // int(tiles))))


def _router_recipe(op, geometry):
    """Select the exact router mapping the executor will lower."""
    from .matrix_steady import SteadyMatmul
    units = dict(getattr(geometry, "resource_units", {}) or ())
    names = {role: f"{role}@router:0" for role in units}
    names["cube"] = "cube@router:0"
    declared_rates = dict(getattr(geometry, "rates", {}) or ())
    if not declared_rates:
        raise ModelingError("router_mapping_requires_declared_pipe_rates")
    rates = {names[role]: float(rate) / MS_PER_SECOND
             for role, rate in declared_rates.items() if role in names}
    peaks = dict(getattr(geometry, "compute_peaks", ()))
    compute = str(dict(op.metadata).get("compute_format", op.dtype or "bf16"))
    try:
        peak = float(peaks[compute])
    except KeyError:
        cube = 0.0
    else:
        cube = MS_PER_SECOND * float(op.flops) / peak
    context = RecipeContext(
        op, {names["cube"]: cube} if cube else {}, names, rates,
        int(getattr(geometry, "compute_units", A5_AIC_UNITS)),
        resource_units=units, compute_peaks=peaks,
        pool_capacities=dict(getattr(geometry, "buffers", {}) or ()),
        physical_mechanisms=getattr(geometry, "physical_mechanisms", None))
    recipe, _record = SteadyMatmul().select(context, include_record=False)
    return recipe


def router_matrix_operand_bytes(op, geometry, *, recipe=None) -> float:
    """Executed routed BF16 operand fetch for the selected matrix mapping.

    Each output row tile re-reads its A row block and each output column tile
    re-reads its B column block, so the operand pipe services
    ``input_bytes * ceil(N / tile_n) + weight_bytes * ceil(M / tile_m)`` bytes.
    The declared ``op.mte2_bytes`` carries the distinct operand bytes only.
    """
    m, n, k = (int(value) for value in op.local_shape[-LOCAL_SHAPE_DIMS:])
    meta = dict(op.metadata)
    input_bytes = float(m * k * dtype_bytes(
        meta.get("input_format", op.dtype or "bf16")))
    weight_bytes = float(n * k * dtype_bytes(
        meta.get("weight_format", op.dtype or "bf16")))
    recipe = _router_recipe(op, geometry) if recipe is None else recipe
    return (input_bytes * -(-n // recipe.tile_n)
            + weight_bytes * -(-m // recipe.tile_m))


def build_router_splitk(op, geometry):
    """A5 narrow-N BF16 router: generic tile grid plus Split-K workspace.

    The matrix wave keeps the declared operand/epilogue pipeline; every free
    AIC receives a disjoint K slice and one FP32 partial.  The workspace write,
    its vector merge and the last-tile drain are explicit stages whose bytes
    and merge ops are added to the plan work.  A non-Split-K router keeps the
    existing steady matrix recipe unchanged.
    """
    from .matrix_steady import SteadyMatmul
    from .mechanism import OperatorPlan, Work
    from .physical_recipe_base import build_plan, work_from_op
    from .program import KernelProgram, Stage
    if not _router_split(op, geometry):
        return build_plan(SteadyMatmul(), op, geometry)
    base = work_from_op(op)
    declared_mte2 = float(op.mte2_bytes)
    declared_mte3 = float(op.mte3_bytes)
    declared_vector = float(op.vector_ops)
    recipe = _router_recipe(op, geometry)
    matrix_mte2 = router_matrix_operand_bytes(op, geometry, recipe=recipe)
    resources = dict(base.resources)
    resources["mte2"] = matrix_mte2
    base = Work(matrix=base.matrix, resources=resources) + Work(
        matrix={"bf16": float(op.flops)})
    m, n, k = (int(value) for value in op.local_shape[-LOCAL_SHAPE_DIMS:])
    tiles = router_output_tiles(m, n)
    cores = int(getattr(geometry, "compute_units", A5_AIC_UNITS))
    splits = min(router_split_count(tiles, cores),
                 max(1, -(-k // ROUTER_K_GRANULE)))
    partial = float(FP32_BYTES) * float(splits) * float(m) * float(n)
    merge = float(splits - 1) * float(m) * float(n)
    extra = Work(resources={"hbm": 2 * partial, "vector": merge})
    plan_work = base + extra
    clock_ms = ROUTER_NOMINAL_CLOCK_HZ / MS_PER_SECOND

    def program(context):
        extra_demand = context.bind(extra)
        matrix_demand = {key: value - extra_demand.get(key, 0)
                         for key, value in context.work.items()}
        matrix_demand = {key: value for key, value in matrix_demand.items()
                         if value > 0}
        matrix_op = replace(
            context.op, mte2_bytes=declared_mte2,
            hbm_bytes=declared_mte2 + declared_mte3,
            vector_ops=declared_vector)
        matrix = SteadyMatmul().lower(
            replace(context, op=matrix_op, work=matrix_demand),
            mapping=recipe)
        workspace = splitk_program(replace(context, work={}), splits)
        return KernelProgram((
            Stage("router_prologue_drain",
                  latency=ROUTER_PROLOGUE_DRAIN_CYCLES / clock_ms),
            Stage("narrow_matrix", after=("router_prologue_drain",),
                  program=matrix),
            Stage("splitk_workspace", after=("narrow_matrix",),
                  program=workspace),
        ))

    return OperatorPlan(plan_work, program, {
        "execution_abstraction": "finite_split_k_workspace",
        "split_k": splits,
        "output_tiles": tiles,
        "router_tile": (ROUTER_TILE_M, ROUTER_TILE_N),
        "matrix_operand_bytes": matrix_mte2,
        "partial_write_bytes": partial,
        "merge_vector_ops": merge,
    })


@dataclass(frozen=True)
class ScopedRouterRecipe(PlannedRecipe):
    """Register A5 narrow-N Split-K without rebinding ordinary router plans."""

    def prepare(self, op, geometry):
        if not _router_split(op, geometry):
            from .matrix_steady import SteadyMatmul
            return op, SteadyMatmul()
        return super().prepare(op, geometry)


def _split_keys(context, hbm, vector, mte3, cube) -> dict[str, float]:
    """Declared demand serviced by the split tasks, never the MMA or output."""
    split_keys = {}
    excluded = {vector, mte3, cube}
    for key, value in context.work.items():
        if key in excluded or (hbm is not None and key == hbm):
            continue
        split_keys[key] = float(value)
    return split_keys


def _local_geometry(context, recipe, geometry):
    if geometry is None and recipe is not None:
        geometry = matrix_geometry(context, recipe)
    if geometry is not None:
        total_m, n, k = (int(value) for value in geometry.local_shape)
        areas = []
        for count, _m_extent, _n_extent, area in geometry.owners:
            count = int(count)
            if count < 1 or area <= 0:
                continue
            share = float(area) / float(count)
            areas.extend([share] * count)
        return float(total_m), float(n), float(k), areas
    local = tuple(int(value) for value in context.op.local_shape)
    if len(local) < LOCAL_SHAPE_DIMS:
        raise ModelingError("split_k_requires_three_local_dimensions")
    total_m, n, k = local[-LOCAL_SHAPE_DIMS:]
    area = float(total_m) * float(n)
    return float(total_m), float(n), float(k), ([area] if area > 0 else [])


def _k_slices(k: float, count: int) -> tuple[int, ...]:
    """Disjoint integer K slices whose sum is exactly ``K``."""
    total = max(0, int(k))
    parts = max(1, int(count))
    base, extra = divmod(total, parts)
    slices = tuple(base + (1 if index < extra else 0)
                   for index in range(parts))
    return tuple(value for value in slices if value > 0)


def _per_owner_cap(context, owners: int) -> dict[str, float]:
    """Service-mode bytes/second as a bytes/ms cap for one active owner."""
    rate = context.service_rate(DRAM_SERVICE_MODE)
    hbm = context.resource_names.get("hbm")
    if rate is None or hbm is None or owners < 1:
        return {}
    return {hbm: rate / (MS_PER_SECOND * float(int(owners)))}


def _placement(context, index: int, cores: int, roles) -> dict:
    return context.owned(index % cores, cores, tuple(roles))


def _tile_merge_program(context, names, area: float, vector_share,
                        mte3_share, hbm_write_share, partial_read, merges: int,
                        caps, index: int, cores: int) -> KernelProgram:
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    stages = []
    predecessor = ()
    if hbm is not None and partial_read > 0:
        stages.append(Stage("partial_read", {hbm: partial_read}, caps=caps))
        predecessor = ("partial_read",)
    merged = {}
    if vector is not None:
        amount = vector_share + float(area) * float(merges - 1)
        if amount > 0:
            merged[vector] = amount
    if merged:
        stages.append(Stage("reduce", merged, after=predecessor,
                            placement=_placement(context, index, cores,
                                                 ("vector",))))
        predecessor = ("reduce",)
    store = {}
    if mte3 is not None and mte3_share > 0:
        store[mte3] = mte3_share
    if hbm is not None and hbm_write_share > 0:
        store[hbm] = hbm_write_share
    if store:
        stages.append(Stage("final_store", store, after=predecessor, caps=caps,
                            placement=_placement(context, index, cores,
                                                 ("mte3",))))
    return KernelProgram(tuple(stages))


def _record_traffic(record, *, split_count, slices, tasks, partial_bytes,
                    vector_merge, final_write, cube_total, declared_hbm,
                    issued_vector):
    if record is None:
        return
    record.update({
        "execution_abstraction": "finite_split_k",
        "split_k": int(split_count),
        "partial_policy": "explicit_gm_fp32_partials_then_vector_merge",
        "split_k_tasks": len(tasks),
        "split_k_k_slices": list(slices),
        "split_k_traffic": {
            "partial_write_bytes": float(partial_bytes),
            "partial_read_bytes": float(partial_bytes),
            "final_write_bytes": float(final_write),
            "merge_vector_ops": float(vector_merge),
            "merge_latency_ms": 0,
            "declared_cube_work": float(cube_total),
            "issued_cube_work": float(cube_total),
            "issued_hbm_work": float(declared_hbm + partial_bytes + partial_bytes),
            "issued_vector_work": float(issued_vector),
            "tasks": len(tasks),
            "output_tiles": int(len(tasks) // max(1, len(slices))),
            "basis": "explicit_gm_fp32_partials_dp4",
        },
    })


def splitk_program(context, split_count: int, *, recipe=None, geometry=None,
                   record=None) -> KernelProgram:
    """Lower one underfilled output grid onto output-tile x K-slice tasks."""
    count = max(1, int(split_count))
    names = context.resource_names
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    cube = names.get("cube")
    if cube is None:
        cube = next((key for key in context.work if key.startswith("matrix@")),
                    None)
    total_m, n, k, areas = _local_geometry(context, recipe, geometry)
    slices = _k_slices(k, count)
    volume = total_m * n * k
    tasks, weights = [], []
    if volume > 0:
        for tile_index, area in enumerate(areas):
            for k_slice in slices:
                weight = area * float(k_slice) / volume
                if weight > 0:
                    tasks.append((tile_index, area, float(k_slice)))
                    weights.append(weight)
    if not tasks:
        return KernelProgram((Stage("setup", latency=context.setup_ms),),
                             minimum_ms=context.minimum_ms)
    cores = max(1, int(context.core_count))
    splits = len(slices)
    split_keys = _split_keys(context, hbm, vector, mte3, cube)
    hbm_total = float(context.work.get(hbm, 0)) if hbm is not None else 0
    mte3_total = float(context.work.get(mte3, 0)) if mte3 is not None else 0
    hbm_write = min(hbm_total, mte3_total)
    hbm_read = max(0, hbm_total - hbm_write)
    cube_total = float(context.work.get(cube, 0)) if cube is not None else 0
    vector_total = (float(context.work.get(vector, 0))
                    if vector is not None else 0)
    loads = [{} for _ in tasks]
    for key, value in split_keys.items():
        for index, share in enumerate(_allocate(value, weights)):
            if share > 0:
                loads[index][key] = share
    if hbm is not None and hbm_read > 0:
        for index, share in enumerate(_allocate(hbm_read, weights)):
            if share > 0:
                loads[index][hbm] = loads[index].get(hbm, 0) + share
    cubes = _allocate(cube_total, weights)
    partial_write = (float(FP32_BYTES) * float(splits) * total_m * n
                     if hbm is not None else 0)
    vector_merge = (float(splits - 1) * total_m * n
                    if vector is not None else 0)
    split_caps = _per_owner_cap(context, len(tasks))
    merge_caps = _per_owner_cap(context, len(areas))
    load_roles = tuple(role for role in ("mte1", "mte2_aic", "mte2")
                       if role in names)
    mma_roles = ("cube",) if "cube" in names else ()

    def split_program(index: int) -> KernelProgram:
        _tile_index, area, _k_slice = tasks[index]
        stages = [Stage("load", loads[index],
                        caps=(split_caps if loads[index] else {}),
                        placement=_placement(context, index, cores, load_roles))]
        predecessor = "load"
        if cube is not None and cubes[index] > 0:
            stages.append(Stage("mma", {cube: cubes[index]}, after=(predecessor,),
                                placement=_placement(context, index, cores, mma_roles)))
            predecessor = "mma"
        if hbm is not None and area > 0:
            write = float(FP32_BYTES) * area
            stages.append(Stage("partial_write", {hbm: write},
                                after=(predecessor,), caps=split_caps))
        return KernelProgram(tuple(stages))

    tile_areas = [areas[tile] for tile in sorted({task[0] for task in tasks})]
    tile_weights = [area / (total_m * n) for area in tile_areas]
    vector_shares = _allocate(vector_total, tile_weights)
    mte3_shares = _allocate(mte3_total, tile_weights)
    hbm_shares = _allocate(hbm_write, tile_weights)

    def merge_program(index: int) -> KernelProgram:
        area = tile_areas[index]
        partial_read = float(FP32_BYTES) * float(splits) * area
        return _tile_merge_program(
            context, names, area, vector_shares[index], mte3_shares[index],
            hbm_shares[index], partial_read, splits, merge_caps, index, cores)

    merges = tuple(merge_program(index) for index in range(len(tile_areas)))
    _record_traffic(
        record, split_count=count, slices=slices, tasks=tasks,
        partial_bytes=partial_write, vector_merge=vector_merge,
        final_write=hbm_write, cube_total=cube_total,
        declared_hbm=hbm_total, issued_vector=vector_total + vector_merge)
    return KernelProgram((
        Stage("setup", latency=context.setup_ms),
        Stage("splits", after=("setup",),
              program=ParallelProgram(tuple(split_program(index)
                                            for index in range(len(tasks))))),
        Stage("merge", after=("splits",),
              program=ParallelProgram(merges)),
    ), minimum_ms=context.minimum_ms)
