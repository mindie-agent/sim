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
  A cap only tightens an existing bound.
[Timeouts & Locks] O(output_tiles * K_slices) bounded tasks; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError
from .matrix_mapping import matrix_geometry
from .program import KernelProgram, ParallelProgram, Stage

FP32_BYTES = 4
MS_PER_SECOND = 1000
DRAM_SERVICE_MODE = "mte2_aic_dram"
LOCAL_SHAPE_DIMS = 3


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


def _allocate(total: float, weights: list[float]) -> list[float]:
    """Allocate ``total`` by weight with the final share closing any rounding."""
    if not weights:
        return []
    shares = [float(total) * weight for weight in weights[:-1]]
    shares.append(float(total) - sum(shares))
    return shares


def _per_owner_cap(context, owners: int) -> dict[str, float]:
    """Service-mode bytes/second as a bytes/ms cap for one active owner."""
    mode = dict(context.service_modes).get(DRAM_SERVICE_MODE)
    hbm = context.resource_names.get("hbm")
    if not mode or hbm is None or owners < 1:
        return {}
    rate = float(mode[1]) / (MS_PER_SECOND * float(int(owners)))
    return {hbm: rate}


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
