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
  A cap only tightens an existing bound.  The A5 narrow-N router splits K
  only when its 128x256 output grid leaves AICs idle and K has two nonempty
  slices; its matrix demand then runs on the same Split-K task grid (operand
  load, Cube, FP32 partial write, vector merge, final store).  A router that
  fills the AIC grid keeps the steady matrix pipeline with no workspace.
[Timeouts & Locks] O(output_tiles * K_slices) bounded tasks; no I/O or locks.
[Side-Effect Policy] Submits only to the caller's virtual Engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
import re
from typing import Literal

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from .matrix import directional_mte1_available, operand_transfer
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import matrix_geometry
from .matrix_mapping_owners import (
    _FRACTAL_ROWS,
    matrix_kernel_setup_ms,
    physical_owner_rate,
    primitive_cold_hbm_bps,
    primitive_nd2nz_mte2_rate_bps_per_aic,
    primitive_p01_hbm_read_bps,
    streamk_fixup_latency_ms,
)
from mindie_sim.prediction.engine import Allocation
from mindie_sim.prediction.mechanism import PlannedRecipe
from mindie_sim.prediction.ownership import read_role
from mindie_sim.prediction.program import KernelProgram, ParallelProgram, RecipeContext, Stage

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_OPS_TRANSFORMER_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "mc2/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_stream_k_block.h",
        "57d5ad95709e91d15f799aae8fc82b646ddd7efcd297e7699c0d034a504dac69",
    ),
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "mc2/3rd/mat_mul_v3/op_kernel/arch35/mat_mul_stream_k_kernel.h",
        "ad43c08de3113cbaabe4736fe883dd361ae33fce94c6b3bb8b5ba0da01efb66d",
    ),
)

FP32_BYTES = 4
#: DAV_3510 Stream-K AIV row-burst block alignment (mat_mul_stream_k_block.h:214).
STREAMK_AIV_BLOCK_ELEMS = 32
BF16_TO_FP32_FIXPIPE_RATIO = 2.0
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
A5_AIC_UNITS = 32


def _require_clock_hz(context, geometry) -> float:
    for owner in (
        getattr(context, "compute_arch", None),
        getattr(context, "datapath", None),
        getattr(geometry, "compute_arch", None),
    ):
        val = getattr(owner, "clock_hz", None)
        if val is not None and not isinstance(val, bool) and float(val) > 0.0:
            return float(val)
    raise ModelingError("router_splitk_clock_hz_missing")


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
    """True only when the router grid leaves AICs idle for nonempty K slices."""
    meta = dict(op.metadata)
    if not (bool(meta.get("split_k_required"))
            and meta.get("weight_format") == "bf16"
            and meta.get("output_format") == "fp32"
            and _a5_aic_grid(geometry)):
        return False
    if len(op.local_shape) < LOCAL_SHAPE_DIMS:
        return False
    m, n, k = (int(value) for value in op.local_shape[-LOCAL_SHAPE_DIMS:])
    tiles = router_output_tiles(m, n)
    cores = int(getattr(geometry, "compute_units", 0) or 0)
    return 0 < tiles < cores and k > ROUTER_K_GRANULE


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
    """Select the steady router mapping (the non-Split-K executed recipe)."""
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
    compute = str(dict(op.metadata).get("compute_format") or op.dtype or "").strip()
    if not compute:
        raise ModelingError("router_compute_format_missing")
    datapath = getattr(geometry, "datapath", None)
    if datapath is not None and hasattr(datapath, "cube_peak"):
        peak = float(datapath.cube_peak(compute))
    else:
        try:
            peak = float(peaks[compute])
        except KeyError as exc:
            raise ModelingError(f"router_compute_peak_missing:{compute}") from exc
    try:
        cube = MS_PER_SECOND * float(op.flops) / peak
    except ZeroDivisionError as exc:
        raise ModelingError(f"router_compute_peak_invalid:{compute}") from exc
    context = RecipeContext(
        op, {names["cube"]: cube} if cube else {}, names, rates,
        int(getattr(geometry, "compute_units", A5_AIC_UNITS)),
        resource_units=units, compute_peaks=peaks,
        pool_capacities=dict(getattr(geometry, "buffers", {}) or ()),
        physical_mechanisms=getattr(geometry, "physical_mechanisms", None),
        compute_arch=getattr(geometry, "compute_arch", None),
        datapath=getattr(geometry, "datapath", None),
        primitive_table=getattr(geometry, "primitive_table", None))
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
    input_fmt = str(meta.get("input_format") or op.dtype or "").strip()
    weight_fmt = str(meta.get("weight_format") or op.dtype or "").strip()
    if not input_fmt or not weight_fmt:
        raise ModelingError("router_operand_format_missing")
    input_bytes = float(m * k * dtype_bytes(input_fmt))
    weight_bytes = float(n * k * dtype_bytes(weight_fmt))
    recipe = _router_recipe(op, geometry) if recipe is None else recipe
    return (input_bytes * -(-n // recipe.tile_n)
            + weight_bytes * -(-m // recipe.tile_m))


_RESOURCE_RANK_RE = re.compile(r"@r(\d+)(?::|$)")


def _resolve_l2_rank(obj, explicit_rank: int | None = None) -> int:
    if explicit_rank is not None:
        return int(explicit_rank)
    rank = getattr(obj, "rank", None)
    if rank is not None:
        return int(rank)
    names = getattr(obj, "resource_names", None) or {}
    for key in ("l2", "hbm", "cube"):
        res_name = names.get(key)
        if isinstance(res_name, str):
            match = _RESOURCE_RANK_RE.search(res_name)
            if match is not None:
                return int(match.group(1))
    for res_name in names.values():
        if isinstance(res_name, str):
            match = _RESOURCE_RANK_RE.search(res_name)
            if match is not None:
                return int(match.group(1))
    return 0


def _geometry_partials_l2_resident(geometry, partial_bytes: float) -> bool:
    """True when the die's live available L2 budget fits the FP32 partials."""
    units = getattr(geometry, "resource_units", None) or {}
    if "l2" not in units or partial_bytes <= 0.0:
        return False
    view = getattr(geometry, "admission_memory_view", None)
    if view is not None:
        rank = _resolve_l2_rank(geometry)
        avail = int(view.effective_l2_bytes(rank))
        return avail > 0 and partial_bytes <= float(avail)
    l2_bytes = int((getattr(geometry, "buffers", None) or {}).get("l2", 0) or 0)
    if l2_bytes <= 0:
        hierarchy = getattr(getattr(geometry, "physical_mechanisms", None), "memory_hierarchy", None)
        l2_bytes = int(getattr(hierarchy, "effective_l2_bytes", 0) or 0)
    return l2_bytes > 0 and partial_bytes <= float(l2_bytes)


def _partials_l2_resident(context, partial_bytes: float, *, rank: int | None = None) -> bool:
    """True when FP32 Split-K partials fit live available die L2 and L2 pipe exists."""
    names = getattr(context, "resource_names", None) or {}
    if "l2" not in names or partial_bytes <= 0.0:
        return False
    view = getattr(context, "admission_memory_view", None)
    if view is not None:
        eff_rank = _resolve_l2_rank(context, explicit_rank=rank)
        avail = int(view.effective_l2_bytes(eff_rank))
        return avail > 0 and partial_bytes <= float(avail)
    from .matrix_mapping_owners import primitive_l2_params
    l2_bytes, _ = primitive_l2_params(context)
    if l2_bytes <= 0:
        hierarchy = getattr(getattr(context, "physical_mechanisms", None), "memory_hierarchy", None)
        l2_bytes = int(getattr(hierarchy, "effective_l2_bytes", 0) or 0)
    return l2_bytes > 0 and partial_bytes <= float(l2_bytes)


def _router_split_recipe(op, geometry):
    """Steady router mapping re-tiled onto the declared 128x256 router grid."""
    recipe = _router_recipe(op, geometry)
    return replace(
        recipe, tile_m=ROUTER_TILE_M, tile_n=ROUTER_TILE_N,
        compute_m=math.gcd(ROUTER_TILE_M, int(recipe.compute_m)),
        compute_n=math.gcd(ROUTER_TILE_N, int(recipe.compute_n)))


def build_router_splitk(op, geometry):
    """A5 narrow-N BF16 router: 128x256 output tiles x disjoint K slices.

    Only an underfilled router grid splits K.  Operand traffic is priced on
    the same 128x256 tile grid that executes; the matrix demand is distributed
    over output-tile x K-slice tasks, each spilling one FP32 partial, followed
    by the vector merge and one final store.  Partial bytes and merge ops are
    added to the plan work.  A router that fills the AIC grid keeps the steady
    matrix recipe with no workspace.
    """
    from .matrix_steady import SteadyMatmul
    from mindie_sim.prediction.mechanism import OperatorPlan, Work
    from mindie_sim.prediction.physical_recipe_base import build_plan, work_from_op
    from mindie_sim.prediction.program import KernelProgram, Stage
    if not _router_split(op, geometry):
        return build_plan(SteadyMatmul(), op, geometry)
    base = work_from_op(op)
    declared_mte2 = float(op.mte2_bytes)
    declared_mte3 = float(op.mte3_bytes)
    declared_vector = float(op.vector_ops)
    recipe = _router_split_recipe(op, geometry)
    matrix_mte2 = router_matrix_operand_bytes(op, geometry, recipe=recipe)
    resources = dict(base.resources)
    resources["mte2"] = matrix_mte2
    matrix = dict(base.matrix)
    if not any(float(val) > 0.0 for val in matrix.values()) and float(op.flops) > 0.0:
        compute_fmt = str(dict(op.metadata).get("compute_format") or op.dtype or "").strip()
        if not compute_fmt:
            raise ModelingError("router_compute_format_missing")
        matrix = {compute_fmt: float(op.flops)}
    base = Work(matrix=matrix, resources=resources)
    m, n, k = (int(value) for value in op.local_shape[-LOCAL_SHAPE_DIMS:])
    tiles = router_output_tiles(m, n)
    cores = int(getattr(geometry, "compute_units", A5_AIC_UNITS))
    splits = min(router_split_count(tiles, cores),
                 max(1, -(-k // ROUTER_K_GRANULE)))
    partial = float(FP32_BYTES) * float(splits) * float(m) * float(n)
    merge = float(splits - 1) * float(m) * float(n)
    ws_l2 = _geometry_partials_l2_resident(geometry, partial)
    units = getattr(geometry, "resource_units", None) or {}
    extra_res: dict[str, float] = {"vector": merge}
    if partial > 0.0:
        if "fixpipe" in units:
            extra_res["fixpipe"] = partial
        vread_role = (
            "mte2_aiv" if "mte2_aiv" in units
            else ("mte2" if "mte2" in units else None)
        )
        if vread_role is not None:
            extra_res[vread_role] = partial
        if ws_l2:
            extra_res["l2"] = 2.0 * partial
        else:
            extra_res["hbm"] = 2.0 * partial
    extra = Work(resources=extra_res)
    plan_work = base + extra

    def program(context):
        extra_demand = context.bind(extra)
        matrix_demand = {
            key: value - extra_demand.get(key, 0.0)
            for key, value in context.work.items()
        }
        matrix_demand = {
            key: value for key, value in matrix_demand.items() if value > 0.0
        }
        matrix_op = replace(
            context.op,
            mte2_bytes=declared_mte2,
            hbm_bytes=declared_mte2 + declared_mte3,
            vector_ops=declared_vector,
        )
        live_l2 = (
            _partials_l2_resident(context, partial, rank=getattr(geometry, "rank", None))
            if getattr(context, "admission_memory_view", None) is not None
            else ws_l2
        )
        return splitk_program(
            replace(context, op=matrix_op, work=matrix_demand),
            splits,
            recipe=recipe,
            ws_l2_resident=live_l2,
        )

    return OperatorPlan(plan_work, program, {
        "execution_abstraction": "finite_split_k_workspace",
        "split_k": splits,
        "output_tiles": tiles,
        "router_tile": (ROUTER_TILE_M, ROUTER_TILE_N),
        "matrix_operand_bytes": matrix_mte2,
        "partial_write_bytes": partial,
        "merge_vector_ops": merge,
        "workspace_residency": "l2" if ws_l2 else "hbm",
    })


@dataclass(frozen=True)
class ScopedRouterRecipe(PlannedRecipe):
    """Register A5 narrow-N Split-K without rebinding ordinary router plans."""

    def prepare(self, op, geometry):
        if not _router_split(op, geometry):
            from .matrix_steady import SteadyMatmul
            return op, SteadyMatmul()
        return super().prepare(op, geometry)


def _split_keys(context, hbm, vector, mte3, cube, *, excluded_extra=()) -> dict[str, float]:
    """Declared demand serviced by the split tasks, never the MMA or output."""
    split_keys = {}
    excluded = {vector, mte3, cube, *excluded_extra}
    for key, value in context.work.items():
        if key in excluded or (hbm is not None and key == hbm):
            continue
        split_keys[key] = float(value)
    return split_keys


def _local_geometry(context, recipe, geometry):
    """Return ``(total_m, n, k, tile_dims)`` preserving exact ``(tile_m, tile_n)`` per tile."""
    if geometry is None and recipe is not None:
        geometry = matrix_geometry(context, recipe)
    if geometry is not None:
        total_m, n, k = (int(value) for value in geometry.local_shape)
        tm = int(geometry.tile_m)
        tn = int(geometry.tile_n)
        tile_dims: list[tuple[int, int]] = []
        for row_g in geometry.group_rows:
            if row_g <= 0 or n <= 0:
                continue
            for r0 in range(0, int(row_g), tm):
                a = min(tm, int(row_g) - r0)
                for c0 in range(0, n, tn):
                    b = min(tn, n - c0)
                    tile_dims.append((a, b))
        return float(total_m), float(n), float(k), tile_dims
    local = tuple(int(value) for value in context.op.local_shape)
    if len(local) < LOCAL_SHAPE_DIMS:
        raise ModelingError("split_k_requires_three_local_dimensions")
    total_m, n, k = local[-LOCAL_SHAPE_DIMS:]
    dims = [(total_m, n)] if (total_m > 0 and n > 0) else []
    return float(total_m), float(n), float(k), dims


def _k_slices(k: float, count: int) -> tuple[int, ...]:
    """Disjoint integer K slices whose sum is exactly ``K``."""
    total = max(0, int(k))
    parts = max(1, int(count))
    base, extra = divmod(total, parts)
    slices = tuple(base + (1 if index < extra else 0)
                   for index in range(parts))
    return tuple(value for value in slices if value > 0)


def _per_owner_cap(
    context, owners: int, *, row_bytes: int = 0, stride_bytes: int = 0,
    m_tiles: int | None = None, panel_rows: int | None = None,
) -> dict[str, float]:
    """Service-mode bytes/second as a bytes/ms cap for one active owner."""
    hbm = context.resource_names.get("hbm")
    if hbm is None or owners < 1:
        return {}
    if getattr(context, "primitive_table", None) is not None and row_bytes > 0 and stride_bytes > 0:
        active = max(1, min(int(context.core_count), int(owners)))
        cold_bps = primitive_cold_hbm_bps(
            context, active_cores=active, row_bytes=row_bytes,
            stride_bytes=stride_bytes, total_tiles=owners, is_grouped=False,
            m_tiles=m_tiles, panel_rows=panel_rows,
        )
        if cold_bps is not None and cold_bps > 0.0:
            return {hbm: cold_bps / (MS_PER_SECOND * float(active))}
    rate = context.service_rate(DRAM_SERVICE_MODE)
    if rate is None:
        return {}
    return {hbm: rate / (MS_PER_SECOND * float(int(owners)))}


def _merge_owner_cap(
    context, merge_tasks: int, *, total_partial_bytes: float = 0.0,
    l2_resident: bool = False,
) -> tuple[dict[str, float], dict[str, float]]:
    """Per-tile caps ``(partial_read_caps, final_store_caps)`` for AIV Stream-K fixup."""
    hbm = context.resource_names.get("hbm")
    l2 = context.resource_names.get("l2")
    if merge_tasks < 1:
        return {}, {}
    pt = getattr(context, "primitive_table", None)
    read_caps: dict[str, float] = {}
    store_caps: dict[str, float] = {}
    if pt is not None:
        from mindie_sim.primitives.loader import _table_band_p50
        from .matrix_mapping_owners import primitive_p01_scaling_bps
        cores = max(1, int(context.core_count))
        if l2_resident and l2 is not None:
            l2_bps = primitive_p01_scaling_bps(context, cores, warm=True)
            if l2_bps is not None and l2_bps > 0.0:
                read_caps[l2] = l2_bps / (MS_PER_SECOND * float(int(merge_tasks)))
        elif hbm is not None:
            hbm_rd = primitive_p01_hbm_read_bps(context)
            if hbm_rd is not None and hbm_rd > 0.0:
                read_caps[hbm] = hbm_rd / (MS_PER_SECOND * float(int(merge_tasks)))
        if hbm is not None:
            hbm_wr = _table_band_p50(pt, "P01_datacopy", "hbm_write_bytes_per_s")
            if hbm_wr is not None and float(hbm_wr) > 0.0:
                store_caps[hbm] = float(hbm_wr) / (MS_PER_SECOND * float(int(merge_tasks)))
            else:
                hbm_rd = primitive_p01_hbm_read_bps(context)
                if hbm_rd is not None and hbm_rd > 0.0:
                    store_caps[hbm] = hbm_rd / (MS_PER_SECOND * float(int(merge_tasks)))
        return read_caps, store_caps
    fallback = _per_owner_cap(context, merge_tasks)
    return ({} if l2_resident else fallback), fallback


def _split_stage_caps(context, base_caps, load_work, *, used_cores: int,
                      tile_m: float, tile_n: float, k_slice: float,
                      elem_bytes: int, cube_key, core_index: int, cores: int):
    """Calibrated MTE2-AIC and 16-row fractal Cube caps for one Split-K slice."""
    caps = dict(base_caps)
    if getattr(context, "primitive_table", None) is None:
        return caps
    names = context.resource_names
    mte2_key = names.get("mte2_aic", names.get("mte2"))
    if mte2_key and mte2_key in load_work and load_work[mte2_key] > 0.0:
        a_slice = float(tile_m) * float(k_slice) * float(elem_bytes)
        b_slice = float(tile_n) * float(k_slice) * float(elem_bytes)
        eff_bps = primitive_nd2nz_mte2_rate_bps_per_aic(
            context, used_cores, a_slice, b_slice,
            weight_layout=str(dict(context.op.metadata).get("weight_layout", "ND")),
        )
        if eff_bps is not None and eff_bps > 0.0:
            true_bytes = a_slice + b_slice
            ratio = min(1.0, float(load_work[mte2_key]) / true_bytes) if true_bytes > 0.0 else 1.0
            caps[mte2_key] = (eff_bps / MS_PER_SECOND) * ratio
    if cube_key and tile_m < float(_FRACTAL_ROWS):
        per_core_cube = physical_owner_rate(
            context, cube_key,
            context.owned(core_index % cores, cores, ("cube",)).get(cube_key))
        caps[cube_key] = per_core_cube * (float(tile_m) / float(_FRACTAL_ROWS))
    return caps


def _partial_write_caps(
    context, base_caps, *, total_partial_bytes: float = 0.0, active_tasks: int = 1,
    l2_resident: bool = False,
) -> dict[str, float]:
    """Per-AIC Fixpipe and L2/HBM write caps for contiguous FP32 workspace partials."""
    pt = getattr(context, "primitive_table", None)
    names = context.resource_names
    fixpipe = names.get("fixpipe")
    hbm = names.get("hbm")
    l2 = names.get("l2")
    if pt is None:
        return {} if l2_resident else dict(base_caps)
    from mindie_sim.primitives.loader import _table_band_p50
    from .matrix_mapping_owners import primitive_p01_scaling_bps
    fix_out = _table_band_p50(pt, "P03_fixpipe", "fixpipe_out_bytes_per_s_per_aic")
    hbm_wr = _table_band_p50(pt, "P01_datacopy", "hbm_write_bytes_per_s")
    caps: dict[str, float] = {}
    if fix_out is not None and float(fix_out) > 0.0:
        per_aic_fix = float(fix_out) * BF16_TO_FP32_FIXPIPE_RATIO / MS_PER_SECOND
        if fixpipe is not None:
            caps[fixpipe] = per_aic_fix
        elif not l2_resident and hbm is not None:
            caps[hbm] = per_aic_fix
    if l2_resident and l2 is not None:
        cores = max(1, int(context.core_count))
        l2_bps = primitive_p01_scaling_bps(context, cores, warm=True)
        if l2_bps is not None and l2_bps > 0.0:
            caps[l2] = float(l2_bps) / (MS_PER_SECOND * float(max(1, int(active_tasks))))
    elif not l2_resident and hbm is not None and hbm_wr is not None and float(hbm_wr) > 0.0:
        wr_per = float(hbm_wr) / (MS_PER_SECOND * float(max(1, int(active_tasks))))
        caps[hbm] = min(caps[hbm], wr_per) if hbm in caps else wr_per
    if caps or l2_resident:
        return caps
    return dict(base_caps)


def _build_split_slice_stages(load_work, mte1_work, cube_key, cube_val, *,
                              fixpipe, mem_tier, area, k_slice, tile_k,
                              load_caps, write_caps, load_place, mte1_place,
                              mma_place, fixpipe_place, l0c_alloc, pipelined: bool):
    """Build K-panel stages for one Split-K task conserving exact work."""
    n_panels = max(1, math.ceil(k_slice / float(max(1, tile_k)))) if tile_k else 1
    has_partial_write = area > 0 and (fixpipe is not None or mem_tier is not None)
    free_l0c_on_write = l0c_alloc if has_partial_write else ()
    stages = []
    if pipelined and n_panels >= 2 and cube_key is not None and cube_val > 0.0 and load_work:
        p_frac = 1.0 / float(n_panels)
        body_frac = 1.0 - p_frac
        pro_work = {key: val * p_frac for key, val in load_work.items()}
        body_work = {key: val * body_frac for key, val in load_work.items()}
        for key, val in mte1_work.items():
            body_work[key] = body_work.get(key, 0.0) + val * body_frac
        body_work[cube_key] = cube_val * body_frac
        epi_work = {cube_key: cube_val * p_frac}
        for key, val in mte1_work.items():
            epi_work[key] = epi_work.get(key, 0.0) + val * p_frac
        body_place = {**load_place, **mte1_place, **mma_place}
        epi_place = {**mma_place, **mte1_place} if mte1_work else mma_place
        stages.append(Stage("prologue_load", pro_work, caps=load_caps, placement=load_place))
        stages.append(Stage("k_body", body_work, after=("prologue_load",),
                            caps=load_caps, placement=body_place, coupled=True,
                            acquire=l0c_alloc))
        stages.append(Stage("epilogue_mma", epi_work, after=("k_body",),
                            caps=load_caps, placement=epi_place,
                            free=() if has_partial_write else l0c_alloc))
        predecessor = "epilogue_mma"
    else:
        has_mma = cube_key is not None and cube_val > 0.0
        stages.append(Stage(
            "load", load_work,
            caps=(load_caps if load_work else {}),
            placement=load_place,
            acquire=() if has_mma else l0c_alloc,
            free=() if (has_mma or has_partial_write) else l0c_alloc,
        ))
        predecessor = "load"
        if mte1_work:
            stages.append(Stage("operand", mte1_work, after=(predecessor,),
                                placement=mte1_place))
            predecessor = "operand"
        if has_mma:
            stages.append(Stage(
                "mma", {cube_key: cube_val}, after=(predecessor,),
                caps=load_caps, placement=mma_place,
                acquire=l0c_alloc,
                free=() if has_partial_write else l0c_alloc,
            ))
            predecessor = "mma"
    if has_partial_write:
        write = float(FP32_BYTES) * area
        pw_work: dict[str, float] = {}
        if fixpipe is not None:
            pw_work[fixpipe] = write
        if mem_tier is not None:
            pw_work[mem_tier] = write
        stages.append(Stage("partial_write", pw_work,
                            after=(predecessor,), caps=write_caps,
                            placement=fixpipe_place, free=free_l0c_on_write))
    return KernelProgram(tuple(stages))


def _placement(context, index: int, cores: int, roles) -> dict:
    return context.owned(index % cores, cores, tuple(roles))


def _row_burst_partition(context, *, tile_m: int, tile_n: int, splits: int,
                         tile_index: int, cores: int):
    """Row-burst AIV partition per upstream ``mat_mul_stream_k_block.h:212-218``."""
    total_aivs = max(1, int(context.resource_units.get("vector", cores)))
    aiv_ratio = max(1, total_aivs // max(1, int(cores)))
    stream_aivs = context.stream_owner_units("vector")
    available = tuple(int(u) for u in stream_aivs) if stream_aivs else tuple(range(total_aivs))
    max_aiv_for_tile = max(1, min(len(available), int(splits) * aiv_ratio))
    row_align = max(1, math.ceil(STREAMK_AIV_BLOCK_ELEMS / max(1, int(tile_n))))
    raw_burst = max(1, math.ceil(max(1, int(tile_m)) / max_aiv_for_tile))
    m_burst_base = math.ceil(raw_burst / row_align) * row_align
    m_burst_cnt = max(1, min(len(available), math.ceil(max(1, int(tile_m)) / m_burst_base)))
    m_burst_tail = max(1, int(tile_m) - (m_burst_cnt - 1) * m_burst_base)
    start = (tile_index * max_aiv_for_tile) % len(available)
    chosen = tuple(dict.fromkeys(
        available[(start + offset) % len(available)] for offset in range(m_burst_cnt)
    ))
    return chosen, m_burst_base, m_burst_cnt, m_burst_tail


def _role_placement_for_units(context, names, cores: int, units: tuple[int, ...],
                              roles: tuple[str, ...]) -> dict:
    placement: dict[str, tuple[int, ...]] = {}
    for role in roles:
        if not role or role not in names:
            continue
        phys = names[role]
        role_units = max(1, int(context.resource_units.get(role, cores)))
        placement[phys] = tuple(dict.fromkeys(u % role_units for u in units))
    return placement


def _merge_burst_program(context, names, *, units: tuple[int, ...], frac: float,
                         area: float, vector_share: float, mte3_share: float,
                         hbm_write_share: float, partial_read: float,
                         scale_copy: float, merges: int, read_caps: dict,
                         store_caps: dict, cores: int, mem_tier: str | None,
                         vread_role: str | None, vread_name: str | None,
                         scale_role: str | None, scale_name: str | None) -> KernelProgram:
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    scaled_read_caps = {k: v * frac for k, v in read_caps.items()} if frac < 1.0 else read_caps
    scaled_store_caps = {k: v * frac for k, v in store_caps.items()} if frac < 1.0 else store_caps
    stages = []
    predecessor = ()
    if partial_read > 0 and (vread_name is not None or mem_tier is not None):
        pr_work: dict[str, float] = {}
        if vread_name is not None:
            pr_work[vread_name] = partial_read * frac
        if mem_tier is not None:
            pr_work[mem_tier] = partial_read * frac
        read_place = (
            _role_placement_for_units(context, names, cores, units, (vread_role,))
            if vread_role is not None else {}
        )
        stages.append(Stage("partial_read", pr_work, caps=scaled_read_caps, placement=read_place))
        predecessor = ("partial_read",)
    if scale_copy > 0 and scale_name is not None and scale_role is not None:
        scale_place = _role_placement_for_units(context, names, cores, units, (scale_role,))
        stages.append(Stage("scales", {scale_name: scale_copy * frac},
                            after=predecessor, placement=scale_place))
        predecessor = ("scales",)
    merged = {}
    if vector is not None:
        amount = (vector_share + float(area) * float(merges - 1)) * frac
        if amount > 0:
            merged[vector] = amount
    if merged:
        vec_place = _role_placement_for_units(context, names, cores, units, ("vector",))
        stages.append(Stage("reduce", merged, after=predecessor, placement=vec_place))
        predecessor = ("reduce",)
    store = {}
    if mte3 is not None and mte3_share > 0:
        store[mte3] = mte3_share * frac
    if hbm is not None and hbm_write_share > 0:
        store[hbm] = hbm_write_share * frac
    if store:
        st_place = _role_placement_for_units(context, names, cores, units, ("mte3",))
        stages.append(Stage("final_store", store, after=predecessor,
                            caps=scaled_store_caps, placement=st_place))
    return KernelProgram(tuple(stages))


def _tile_merge_program(context, names, tile_m: int, tile_n: int, area: float,
                        vector_share, mte3_share, hbm_write_share, partial_read,
                        scale_copy: float, merges: int, read_caps, store_caps,
                        index: int, cores: int, *,
                        mem_tier: str | None = None,
                        vread_role: str | None = None,
                        vread_name: str | None = None,
                        scale_role: str | None = None,
                        scale_name: str | None = None):
    chosen, m_burst_base, m_burst_cnt, m_burst_tail = _row_burst_partition(
        context, tile_m=tile_m, tile_n=tile_n, splits=merges,
        tile_index=index, cores=cores,
    )
    common = dict(
        area=area, vector_share=vector_share, mte3_share=mte3_share,
        hbm_write_share=hbm_write_share, partial_read=partial_read,
        scale_copy=scale_copy, merges=merges, read_caps=read_caps,
        store_caps=store_caps, cores=cores, mem_tier=mem_tier,
        vread_role=vread_role, vread_name=vread_name,
        scale_role=scale_role, scale_name=scale_name,
    )
    if m_burst_cnt > 1 and m_burst_tail < m_burst_base and len(chosen) > 1:
        full_rows = float((m_burst_cnt - 1) * m_burst_base)
        tail_rows = float(m_burst_tail)
        total_rows = max(1.0, full_rows + tail_rows)
        return ParallelProgram((
            _merge_burst_program(
                context, names, units=chosen[:-1], frac=full_rows / total_rows, **common),
            _merge_burst_program(
                context, names, units=(chosen[-1],), frac=tail_rows / total_rows, **common),
        ))
    return _merge_burst_program(context, names, units=chosen, frac=1.0, **common)


def _record_traffic(record, *, split_count, slices, tasks, partial_bytes,
                    vector_merge, final_write, cube_total, declared_hbm,
                    issued_vector, l2_resident: bool = False,
                    fixpipe_present: bool = False):
    if record is None:
        return
    partial_hbm = 0.0 if l2_resident else float(partial_bytes) * 2.0
    partial_l2 = float(partial_bytes) * 2.0 if l2_resident else 0.0
    record.update({
        "execution_abstraction": "finite_split_k",
        "split_k": int(split_count),
        "partial_policy": (
            "explicit_l2_fp32_partials_then_vector_merge"
            if l2_resident else "explicit_gm_fp32_partials_then_vector_merge"
        ),
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
            "issued_hbm_work": float(declared_hbm + partial_hbm),
            "issued_l2_work": partial_l2,
            "issued_fixpipe_work": float(partial_bytes) if fixpipe_present else 0.0,
            "workspace_residency": "l2" if l2_resident else "hbm",
            "issued_vector_work": float(issued_vector),
            "tasks": len(tasks),
            "output_tiles": int(len(tasks) // max(1, len(slices))),
            "basis": "explicit_gm_fp32_partials_dp4",
        },
    })


def _splitk_task_loads(context, tasks, weights, *, names, hbm, vector, mte3, cube,
                       total_m, n, k, groups, compute_width, directional_mte1,
                       recipe=None):
    """Build per-task load dicts (private A/B operand copies + HBM) and MTE1 dicts."""
    operand_read = read_role(names)
    operand_name = names.get(operand_read)
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    has_private_operands = operand_name is not None and (inputs + weight) > 0.0
    excluded_extra = (
        names.get("mte1"), names.get("fixpipe"),
        operand_name if has_private_operands else None,
    )
    split_keys = _split_keys(
        context, hbm, vector, mte3, cube, excluded_extra=excluded_extra)
    hbm_total = float(context.work.get(hbm, 0)) if hbm is not None else 0.0
    mte3_total = float(context.work.get(mte3, 0)) if mte3 is not None else 0.0
    hbm_write = min(hbm_total, mte3_total)
    hbm_read = max(0.0, hbm_total - hbm_write)
    loads = [{} for _ in tasks]
    for key, value in split_keys.items():
        for index, share in enumerate(_allocate(value, weights)):
            if share > 0:
                loads[index][key] = share
    if hbm is not None and hbm_read > 0:
        for index, share in enumerate(_allocate(hbm_read, weights)):
            if share > 0:
                loads[index][hbm] = loads[index].get(hbm, 0.0) + share
    mte1_loads = []
    rec_cm = int(getattr(recipe, "compute_m", 0) or 0) if recipe is not None else 0
    rec_cn = int(getattr(recipe, "compute_n", 0) or 0) if recipe is not None else 0
    for index, (_tile_index, tm, tn, _area, k_slice) in enumerate(tasks):
        k_frac = k_slice / k if k > 0 else 0.0
        a_copy = inputs * (float(tm) / total_m) * k_frac if total_m > 0 else 0.0
        b_copy = weight * (float(tn) / (n * float(groups))) * k_frac if n > 0 else 0.0
        if has_private_operands and (a_copy + b_copy) > 0.0:
            loads[index][operand_name] = a_copy + b_copy
        inner_m = math.ceil(int(tm) / rec_cm) if rec_cm > 0 else 1
        inner_n = math.ceil(int(tn) / rec_cn) if rec_cn > 0 else 1
        left_l0 = float(inner_n) * max(a_copy, float(tm) * k_slice * compute_width)
        right_l0 = float(inner_m) * max(b_copy, float(tn) * k_slice * compute_width)
        mte1_loads.append(operand_transfer(
            context, left_l0, right_l0, directional=directional_mte1))
    return loads, mte1_loads, hbm_total, mte3_total, hbm_write, epilogue


def splitk_program(context, split_count: int, *, recipe=None, geometry=None,
                   record=None, ws_l2_resident: bool | None = None) -> KernelProgram:
    """Lower one underfilled output grid onto output-tile x K-slice tasks."""
    count = max(1, int(split_count))
    names = context.resource_names
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    l2_res, fixpipe = names.get("l2"), names.get("fixpipe")
    vread_role = (
        "mte2_aiv" if "mte2_aiv" in names
        else ("mte2" if "mte2" in names else None)
    )
    vread_name = names.get(vread_role) if vread_role is not None else None
    cube = names.get("cube")
    if cube is None:
        cube = next((key for key in context.work if key.startswith("matrix@")),
                    None)
    total_m, n, k, tile_dims = _local_geometry(context, recipe, geometry)
    slices = _k_slices(k, count)
    volume = total_m * n * k
    tasks, weights = [], []
    if volume > 0:
        for tile_index, (tm, tn) in enumerate(tile_dims):
            area = float(tm * tn)
            for k_slice in slices:
                weight_share = area * float(k_slice) / volume
                if weight_share > 0:
                    tasks.append((tile_index, tm, tn, area, float(k_slice)))
                    weights.append(weight_share)
    if not tasks:
        return KernelProgram((Stage("setup", latency=context.setup_ms),),
                             minimum_ms=context.minimum_ms)
    cores = max(1, int(context.core_count))
    stream_aic = context.stream_owner_units("cube")
    splits = len(slices)
    pt = getattr(context, "primitive_table", None)
    compute_fmt = str(dict(context.op.metadata).get("compute_format") or context.op.dtype or "").strip()
    if not compute_fmt:
        raise ModelingError("splitk_compute_format_missing")
    compute_width = float(dtype_bytes(compute_fmt))
    elem_bytes = int(compute_width)
    groups = (
        geometry.groups if geometry is not None
        else max(1, int(dict(context.op.metadata).get("matrix_groups", 1) or 1))
    )
    loads, mte1_loads, hbm_total, mte3_total, hbm_write, epilogue = _splitk_task_loads(
        context, tasks, weights, names=names, hbm=hbm, vector=vector,
        mte3=mte3, cube=cube, total_m=total_m, n=n, k=k, groups=groups,
        compute_width=compute_width,
        directional_mte1=(pt is not None and directional_mte1_available(context)),
        recipe=recipe,
    )
    scale_role = read_role(names, aiv=0)
    scale_name = names.get(scale_role)
    reuse_inputs = bool(getattr(recipe, "reuse_inputs", True))
    reuse_weights = bool(getattr(recipe, "reuse_weights", True))
    cube_total = float(context.work.get(cube, 0)) if cube is not None else 0.0
    vector_total = float(context.work.get(vector, 0)) if vector is not None else 0.0
    cubes = _allocate(cube_total, weights)
    has_partial_sink = hbm is not None or l2_res is not None or fixpipe is not None
    partial_write = (float(FP32_BYTES) * float(splits) * total_m * n
                     if has_partial_sink else 0.0)
    l2_resident = (
        _partials_l2_resident(context, partial_write)
        if ws_l2_resident is None
        else (bool(ws_l2_resident) and l2_res is not None)
    )
    mem_tier = l2_res if l2_resident else hbm
    vector_merge = float(splits - 1) * total_m * n if vector is not None else 0.0
    tile_m_dim = min(total_m, float(recipe.tile_m)) if recipe is not None else total_m
    tile_n_dim = min(n, float(recipe.tile_n)) if recipe is not None else n
    tile_k_dim = int(recipe.tile_k) if recipe is not None else 0
    used_cores = max(1, min(cores, len(tasks)))
    m_tiles = max(1, math.ceil(total_m / max(1.0, tile_m_dim)))
    split_caps = _per_owner_cap(
        context, len(tasks),
        row_bytes=int(tile_n_dim) * elem_bytes, stride_bytes=int(n) * elem_bytes,
        m_tiles=m_tiles, panel_rows=max(1, int(k)))
    write_caps = _partial_write_caps(
        context, split_caps, total_partial_bytes=partial_write,
        active_tasks=len(tasks), l2_resident=l2_resident)
    merge_read_caps, merge_store_caps = _merge_owner_cap(
        context, len(tile_dims), total_partial_bytes=partial_write,
        l2_resident=l2_resident)
    load_roles = tuple(role for role in ("mte2_aic", "mte2") if role in names)
    mte1_roles = ("mte1",) if "mte1" in names else ()
    mma_roles = ("cube",) if "cube" in names else ()
    fixpipe_roles = ("fixpipe",) if "fixpipe" in names else ()
    l0c_pool = context.pool_names.get("l0c")

    def split_program(index: int) -> KernelProgram:
        _tile_index, tm, tn, area, k_slice = tasks[index]
        phys_core = (
            int(stream_aic[index % len(stream_aic)]) if stream_aic else index % cores
        )
        load_caps = _split_stage_caps(
            context, split_caps, loads[index], used_cores=used_cores,
            tile_m=float(tm), tile_n=float(tn), k_slice=k_slice,
            elem_bytes=elem_bytes, cube_key=cube, core_index=phys_core, cores=cores)
        l0c_size = float(tm * tn * FP32_BYTES)
        l0c_alloc = (
            (Allocation(f"splitk_l0c_{index}", l0c_pool, l0c_size, (phys_core,)),)
            if (l0c_pool is not None and l0c_size > 0.0) else ()
        )
        return _build_split_slice_stages(
            loads[index], mte1_loads[index], cube, cubes[index],
            fixpipe=fixpipe, mem_tier=mem_tier, area=area,
            k_slice=k_slice, tile_k=tile_k_dim, load_caps=load_caps,
            write_caps=write_caps,
            load_place=_placement(context, phys_core, cores, load_roles),
            mte1_place=_placement(context, phys_core, cores, mte1_roles),
            mma_place=_placement(context, phys_core, cores, mma_roles),
            fixpipe_place=_placement(context, phys_core, cores, fixpipe_roles),
            l0c_alloc=l0c_alloc, pipelined=(pt is not None))

    unique_tiles = sorted({task[0] for task in tasks})
    tile_shapes = [tile_dims[tile] for tile in unique_tiles]
    tile_areas = [float(tm * tn) for tm, tn in tile_shapes]
    tile_weights = [area / (total_m * n) for area in tile_areas]
    vector_shares = _allocate(vector_total, tile_weights)
    mte3_shares = _allocate(mte3_total, tile_weights)
    hbm_shares = _allocate(hbm_write, tile_weights)

    def merge_program(index: int):
        tm, tn = tile_shapes[index]
        area = tile_areas[index]
        partial_read = float(FP32_BYTES) * float(splits) * area
        scale_copy = (
            epilogue.scales(
                tm, tn, total_m, n, groups,
                reuse_inputs=reuse_inputs, reuse_weights=reuse_weights)[0]
            if (epilogue.row_scale_bytes + epilogue.column_scale_bytes) > 0.0 else 0.0
        )
        return _tile_merge_program(
            context, names, tm, tn, area, vector_shares[index], mte3_shares[index],
            hbm_shares[index], partial_read, scale_copy, splits, merge_read_caps,
            merge_store_caps, index, cores, mem_tier=mem_tier,
            vread_role=vread_role, vread_name=vread_name,
            scale_role=scale_role, scale_name=scale_name)

    merges = tuple(merge_program(index) for index in range(len(tile_areas)))
    _record_traffic(
        record, split_count=count, slices=slices, tasks=tasks,
        partial_bytes=partial_write, vector_merge=vector_merge,
        final_write=hbm_write, cube_total=cube_total,
        declared_hbm=hbm_total, issued_vector=vector_total + vector_merge,
        l2_resident=l2_resident, fixpipe_present=(fixpipe is not None))
    total_setup_ms = float(context.setup_ms or 0.0) + matrix_kernel_setup_ms(
        context, is_streamk=(splits > 1))
    sync_ms = streamk_fixup_latency_ms(context, k_splits=splits, used_cores=used_cores)
    if sync_ms > 0.0:
        return KernelProgram((
            Stage("setup", latency=total_setup_ms),
            Stage("splits", after=("setup",),
                  program=ParallelProgram(tuple(split_program(index)
                                                for index in range(len(tasks))))),
            Stage("split_sync", after=("splits",), latency=sync_ms, keep_empty=True),
            Stage("merge", after=("split_sync",),
                  program=ParallelProgram(merges)),
        ), minimum_ms=context.minimum_ms)
    return KernelProgram((
        Stage("setup", latency=total_setup_ms),
        Stage("splits", after=("setup",),
              program=ParallelProgram(tuple(split_program(index)
                                            for index in range(len(tasks))))),
        Stage("merge", after=("splits",),
              program=ParallelProgram(merges)),
    ), minimum_ms=context.minimum_ms)
