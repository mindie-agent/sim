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
from .matrix import directional_mte1_available, operand_transfer, plain_store_role
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping import (
    STREAM_K_K_ALIGN,
    STREAM_K_MIN_K_BYTES,
    STREAM_K_PER_AIC_K_BYTES,
    matrix_geometry,
)
from .matrix_mapping_owners import (
    _FRACTAL_ROWS,
    is_arch35_context,
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
        "mc2/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp",
        "1a1ba337c283965aa9495220b1df8b868c082591768c4b82e6c1d251b2def497",
    ),
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
#: DAV_3510 Stream-K K-slice alignment in bytes (matmul_v3_basic_streamk_tiling.cpp:392).
STREAMK_K_ALIGN_BYTES = 256
#: DAV_3510 ND_FIXPIPE_1_2 workspace N alignment and admission bounds
#: (matmul_v3_basic_streamk_tiling.cpp:26,288-311).
FIXPIPE_N_ALIGN_ELEMS = 32
ND_FIXPIPE_MIN_N = 64
ND_FIXPIPE_MIN_M = 2
ND_FIXPIPE_MIN_AREA = 256
_HALF_OUTPUT_FORMATS = frozenset({"bf16", "bfloat16", "fp16", "float16"})
_PURE_FP32_FORMATS = frozenset({"fp32", "float32"})
#: Payload byte width per element for MX labels whose ``dtype_bytes`` bundles E8M0 scales.
_PAYLOAD_DTYPE_BYTES = {
    "mxfp4": 0.5, "mx_fp4": 0.5, "mxfp4_e2m1": 0.5,
    "mxfp8": 1.0, "mx_fp8": 1.0, "mxfp8_e4m3": 1.0,
}
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


def _payload_dtype_bytes(dtype: str) -> float:
    """Payload element byte width without bundled microscale bytes."""
    key = str(dtype).strip().lower()
    if key in _PAYLOAD_DTYPE_BYTES:
        return _PAYLOAD_DTYPE_BYTES[key]
    return float(dtype_bytes(key))


def _a5_aic_grid(geometry) -> bool:
    units = getattr(geometry, "resource_units", {}) or {}
    cores = int(getattr(geometry, "compute_units", 0) or 0)
    mte2_aic = int(units.get("mte2_aic", 0) or 0)
    if cores < 2 or mte2_aic < cores:
        return False
    has_arch = (
        getattr(geometry, "compute_arch", None) is not None
        or getattr(getattr(geometry, "datapath", None), "compute_arch", None) is not None
        or getattr(geometry, "primitive_table", None) is not None
    )
    return is_arch35_context(geometry) if has_arch else cores >= A5_AIC_UNITS


def _arch35_k_admitted(k: int, *, cores: int, payload_width: float) -> bool:
    """DAV_3510 Stream-K K-axis admission threshold (matmul_v3_basic_streamk_tiling.cpp:257-262)."""
    if k <= 0 or cores <= 0 or payload_width <= 0.0:
        return False
    min_k = max(STREAM_K_MIN_K_BYTES, int(cores) * STREAM_K_PER_AIC_K_BYTES) / float(payload_width)
    k_aligned = math.ceil(int(k) / STREAM_K_K_ALIGN) * STREAM_K_K_ALIGN
    return float(k_aligned) >= min_k


def _streamk_k_align(meta: dict, compute_fmt: str, payload_width: float, *, is_arch35: bool) -> int:
    """DAV_3510 Stream-K singleCoreK element alignment (matmul_v3_basic_streamk_tiling.cpp:263,392)."""
    if not is_arch35 or payload_width <= 0.0:
        return 1
    if compute_fmt.strip().lower() in _PURE_FP32_FORMATS and not bool(meta.get("is_hf32")):
        return 1
    is_a_trans = bool(meta.get("transpose_a") or meta.get("is_a_trans"))
    is_b_trans = bool(meta.get("transpose_b") or meta.get("is_b_trans"))
    if is_a_trans and not is_b_trans:
        return _FRACTAL_ROWS
    return max(1, int(math.ceil(float(STREAMK_K_ALIGN_BYTES) / float(payload_width))))


def _uses_nd_fixpipe_1_2(total_m: float, n: float, *, is_arch35: bool) -> bool:
    """DAV_3510 ``ND_FIXPIPE_1_2`` workspace padding predicate (matmul_v3_basic_streamk_tiling.cpp:288-295)."""
    m_int, n_int = int(total_m), int(n)
    return (
        is_arch35
        and n_int > ND_FIXPIPE_MIN_N
        and (n_int % _FRACTAL_ROWS) != 0
        and m_int > ND_FIXPIPE_MIN_M
        and (m_int * n_int) >= ND_FIXPIPE_MIN_AREA
    )


def _tile_workspace_widths(tile_n: int, *, nd_fixpipe_1_2: bool) -> tuple[int, int, int]:
    """Return ``(ws_tn, fixpipe_tn, clear_tn)`` per ``mat_mul_stream_k_kernel.h:54,271``."""
    tn = int(tile_n)
    if not nd_fixpipe_1_2 or tn <= 0:
        return tn, tn, 0
    ws_tn = math.ceil(tn / FIXPIPE_N_ALIGN_ELEMS) * FIXPIPE_N_ALIGN_ELEMS
    fixpipe_tn = math.ceil(tn / _FRACTAL_ROWS) * _FRACTAL_ROWS
    rem = tn % FIXPIPE_N_ALIGN_ELEMS
    clear_tn = _FRACTAL_ROWS if (0 < rem <= _FRACTAL_ROWS) else 0
    return ws_tn, fixpipe_tn, clear_tn


def _aiv_burst_layout(tile_m: int, ws_tile_n: int, *, splits: int, cores: int,
                      aiv_count: int, align_burst: bool) -> tuple[int, int, int, int, int, int]:
    """Row-burst counts and aligned AIV read/reduce elements (``mat_mul_stream_k_block.h:212-256``)."""
    avail = max(1, int(aiv_count))
    aiv_ratio = max(1, avail // max(1, int(cores)))
    max_aiv = max(1, min(avail, int(splits) * aiv_ratio))
    row_align = max(1, math.ceil(STREAMK_AIV_BLOCK_ELEMS / max(1, int(ws_tile_n))))
    raw_burst = max(1, math.ceil(max(1, int(tile_m)) / max_aiv))
    m_burst_base = math.ceil(raw_burst / row_align) * row_align
    m_burst_cnt = max(1, min(avail, math.ceil(max(1, int(tile_m)) / m_burst_base)))
    m_burst_tail = max(1, int(tile_m) - (m_burst_cnt - 1) * m_burst_base)
    if align_burst:
        per_full = math.ceil((m_burst_base * int(ws_tile_n)) / _FRACTAL_ROWS) * _FRACTAL_ROWS
        full_elems = (m_burst_cnt - 1) * per_full
        tail_elems = math.ceil((m_burst_tail * int(ws_tile_n)) / _FRACTAL_ROWS) * _FRACTAL_ROWS
    else:
        full_elems = (m_burst_cnt - 1) * m_burst_base * int(ws_tile_n)
        tail_elems = m_burst_tail * int(ws_tile_n)
    return m_burst_base, m_burst_cnt, m_burst_tail, max_aiv, full_elems, tail_elems


def _needs_output_cast(op, meta: dict, epilogue: MatrixEpilogue) -> bool:
    if epilogue.operations_per_output > 0.0:
        return False
    abi = getattr(op, "precision_abi", None)
    out_fmt = str(
        meta.get("output_format")
        or (getattr(abi, "output_dtype", "") if abi is not None else "")
        or getattr(op, "dtype", "")
        or ""
    ).strip().lower()
    return out_fmt in _HALF_OUTPUT_FORMATS


def _router_split(op, geometry) -> bool:
    """True only when the router grid leaves AICs idle above the DAV_3510 K threshold."""
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
    if not (0 < tiles < cores and k > ROUTER_K_GRANULE):
        return False
    requested = router_split_count(tiles, cores)
    if requested <= 1:
        return False
    compute_fmt = str(meta.get("compute_format") or op.dtype or "bf16").strip()
    payload_width = _payload_dtype_bytes(compute_fmt)
    if not _arch35_k_admitted(k, cores=cores, payload_width=payload_width):
        return False
    k_align = _streamk_k_align(meta, compute_fmt, payload_width, is_arch35=True)
    return len(_k_slices(k, requested, align=k_align, fixed_stride=True)) > 1


def router_output_tiles(m: int, n: int) -> int:
    """Declared 128x256 router tile grid; zero extents own no tile."""
    if m <= 0 or n <= 0:
        return 0
    return (-(-m // ROUTER_TILE_M)) * (-(-n // ROUTER_TILE_N))


def router_split_count(tiles: int, aics: int = A5_AIC_UNITS) -> int:
    """Split K only within the single-wave AIC budget ``floor(aics / tiles)``."""
    if tiles <= 0:
        return 1
    return max(1, int(aics) // int(tiles))


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
    units = getattr(geometry, "resource_units", None) or {}
    aiv_count = max(1, int(units.get("vector", cores)))
    meta = dict(op.metadata)
    compute_fmt = str(meta.get("compute_format") or op.dtype or "bf16").strip()
    payload_width = _payload_dtype_bytes(compute_fmt)
    k_align = _streamk_k_align(meta, compute_fmt, payload_width, is_arch35=True)
    requested_splits = min(router_split_count(tiles, cores),
                           max(1, -(-k // ROUTER_K_GRANULE)))
    slices = _k_slices(k, requested_splits, align=k_align, fixed_stride=True)
    splits = len(slices)
    nd_fixpipe_1_2 = _uses_nd_fixpipe_1_2(m, n, is_arch35=True)
    fixpipe_total = 0.0
    clear_total = 0.0
    aiv_elems_total = 0.0
    for r0 in range(0, m, ROUTER_TILE_M):
        tm = min(ROUTER_TILE_M, m - r0)
        for c0 in range(0, n, ROUTER_TILE_N):
            tn = min(ROUTER_TILE_N, n - c0)
            ws_tn, fixpipe_tn, clear_tn = _tile_workspace_widths(
                tn, nd_fixpipe_1_2=nd_fixpipe_1_2)
            fixpipe_total += float(FP32_BYTES * splits * tm * fixpipe_tn)
            clear_total += float(FP32_BYTES * splits * tm * clear_tn)
            _, _, _, _, full_e, tail_e = _aiv_burst_layout(
                tm, ws_tn, splits=splits, cores=cores,
                aiv_count=aiv_count, align_burst=True)
            aiv_elems_total += float(full_e + tail_e)
    partial_write = fixpipe_total + clear_total
    partial_read = float(FP32_BYTES) * float(splits) * aiv_elems_total
    ws_mem_total = partial_write + partial_read
    merge = float(splits - 1) * aiv_elems_total
    ws_l2 = _geometry_partials_l2_resident(geometry, max(partial_write, partial_read))
    extra_res: dict[str, float] = {"vector": merge}
    if fixpipe_total > 0.0 and "fixpipe" in units:
        extra_res["fixpipe"] = fixpipe_total
    if clear_total > 0.0 and "mte3" in units:
        extra_res["mte3"] = clear_total
    if partial_read > 0.0:
        vread_role = (
            "mte2_aiv" if "mte2_aiv" in units
            else ("mte2" if "mte2" in units else None)
        )
        if vread_role is not None:
            extra_res[vread_role] = partial_read
    if ws_mem_total > 0.0:
        if ws_l2:
            extra_res["l2"] = ws_mem_total
        else:
            extra_res["hbm"] = ws_mem_total
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
            _partials_l2_resident(context, max(partial_write, partial_read),
                                  rank=getattr(geometry, "rank", None))
            if getattr(context, "admission_memory_view", None) is not None
            else ws_l2
        )
        return splitk_program(
            replace(context, op=matrix_op, work=matrix_demand),
            splits,
            recipe=recipe,
            ws_l2_resident=live_l2,
            k_align=k_align,
            nd_fixpipe_1_2=nd_fixpipe_1_2,
            arch35_streamk=True,
        )

    return OperatorPlan(plan_work, program, {
        "execution_abstraction": "finite_split_k_workspace",
        "split_k": splits,
        "output_tiles": tiles,
        "router_tile": (ROUTER_TILE_M, ROUTER_TILE_N),
        "matrix_operand_bytes": matrix_mte2,
        "partial_write_bytes": partial_write,
        "partial_read_bytes": partial_read,
        "fixpipe_write_bytes": fixpipe_total,
        "workspace_clear_bytes": clear_total,
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


def _k_slices(k: float, count: int, *, align: int = 1, fixed_stride: bool = False) -> tuple[int, ...]:
    """Disjoint integer K slices whose sum is exactly ``K``."""
    total = max(0, int(k))
    if total <= 0:
        return ()
    parts = max(1, int(count))
    granule = max(1, int(align))
    if granule > 1 or fixed_stride:
        raw_k = math.ceil(total / parts)
        single_core_k = math.ceil(raw_k / granule) * granule
        actual = math.ceil(total / single_core_k)
        full = actual - 1
        tail = total - full * single_core_k
        return (single_core_k,) * full + (tail,)
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
                      elem_bytes: float, cube_key, core_index: int, cores: int):
    """Calibrated MTE2-AIC and 16-row fractal Cube caps for one Split-K slice."""
    caps = dict(base_caps)
    if getattr(context, "primitive_table", None) is None:
        return caps
    names = context.resource_names
    mte2_key = names.get("mte2_aic", names.get("mte2"))
    if mte2_key and mte2_key in load_work and load_work[mte2_key] > 0.0:
        a_slice = float(math.ceil(float(tile_m) * float(k_slice) * float(elem_bytes)))
        b_slice = float(math.ceil(float(tile_n) * float(k_slice) * float(elem_bytes)))
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
                              fixpipe, mte3, mem_tier, fixpipe_area, clear_area,
                              k_slice, tile_k, load_caps, write_caps, load_place,
                              mte1_place, mma_place, fixpipe_place, clear_place,
                              l0c_alloc, pipelined: bool,
                              defer_l0c_free: bool = False):
    """Build K-panel stages for one Split-K task conserving exact work."""
    n_panels = max(1, math.ceil(k_slice / float(max(1, tile_k)))) if tile_k else 1
    has_partial_write = fixpipe_area > 0 and (fixpipe is not None or mem_tier is not None)
    has_ws_clear = clear_area > 0 and (mte3 is not None or mem_tier is not None)
    hold_l0c = has_partial_write or defer_l0c_free
    free_l0c_on_write = l0c_alloc if has_partial_write else ()
    stages = []
    if has_ws_clear:
        clear_bytes = float(FP32_BYTES) * clear_area
        clr_work: dict[str, float] = {}
        if mte3 is not None:
            clr_work[mte3] = clear_bytes
        if mem_tier is not None:
            clr_work[mem_tier] = clear_bytes
        stages.append(Stage("ws_clear", clr_work, placement=clear_place))
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
                            free=() if hold_l0c else l0c_alloc))
        predecessor = "epilogue_mma"
    else:
        has_mma = cube_key is not None and cube_val > 0.0
        stages.append(Stage(
            "load", load_work,
            caps=(load_caps if load_work else {}),
            placement=load_place,
            acquire=() if has_mma else l0c_alloc,
            free=() if (has_mma or hold_l0c) else l0c_alloc,
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
                free=() if hold_l0c else l0c_alloc,
            ))
            predecessor = "mma"
    if has_partial_write:
        write = float(FP32_BYTES) * fixpipe_area
        pw_work: dict[str, float] = {}
        if fixpipe is not None:
            pw_work[fixpipe] = write
        if mem_tier is not None:
            pw_work[mem_tier] = write
        pw_after = (predecessor, "ws_clear") if has_ws_clear else (predecessor,)
        stages.append(Stage("partial_write", pw_work,
                            after=pw_after, caps=write_caps,
                            placement=fixpipe_place, free=free_l0c_on_write))
    return KernelProgram(tuple(stages))


def _placement(context, index: int, cores: int, roles) -> dict:
    return context.owned(index % cores, cores, tuple(roles))


def _row_burst_partition(context, *, tile_m: int, tile_n: int, splits: int,
                         tile_index: int, cores: int, align_burst: bool = False):
    """Row-burst AIV partition per upstream ``mat_mul_stream_k_block.h:212-256``."""
    total_aivs = max(1, int(context.resource_units.get("vector", cores)))
    stream_aivs = context.stream_owner_units("vector")
    available = tuple(int(u) for u in stream_aivs) if stream_aivs else tuple(range(total_aivs))
    m_burst_base, m_burst_cnt, m_burst_tail, max_aiv, full_e, tail_e = _aiv_burst_layout(
        tile_m, tile_n, splits=splits, cores=cores,
        aiv_count=len(available), align_burst=align_burst,
    )
    start = (tile_index * max_aiv) % len(available)
    chosen = tuple(dict.fromkeys(
        available[(start + offset) % len(available)] for offset in range(m_burst_cnt)
    ))
    return chosen, m_burst_base, m_burst_cnt, m_burst_tail, full_e, tail_e


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


def _merge_burst_program(context, names, *, units: tuple[int, ...], row_frac: float,
                         burst_elems: float, vector_share: float, mte3_share: float,
                         hbm_write_share: float, partial_read: float,
                         scale_copy: float, merges: int, output_cast: bool,
                         read_caps: dict, store_caps: dict, cores: int,
                         mem_tier: str | None, vread_role: str | None,
                         vread_name: str | None, scale_role: str | None,
                         scale_name: str | None) -> KernelProgram:
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    scaled_read_caps = {k: v * row_frac for k, v in read_caps.items()} if row_frac < 1.0 else read_caps
    scaled_store_caps = {k: v * row_frac for k, v in store_caps.items()} if row_frac < 1.0 else store_caps
    stages = []
    predecessor = ()
    if partial_read > 0 and (vread_name is not None or mem_tier is not None):
        pr_work: dict[str, float] = {}
        if vread_name is not None:
            pr_work[vread_name] = partial_read
        if mem_tier is not None:
            pr_work[mem_tier] = partial_read
        read_place = (
            _role_placement_for_units(context, names, cores, units, (vread_role,))
            if vread_role is not None else {}
        )
        stages.append(Stage("partial_read", pr_work, caps=scaled_read_caps, placement=read_place))
        predecessor = ("partial_read",)
    if scale_copy > 0 and scale_name is not None and scale_role is not None:
        scale_place = _role_placement_for_units(context, names, cores, units, (scale_role,))
        stages.append(Stage("scales", {scale_name: scale_copy * row_frac},
                            after=predecessor, placement=scale_place))
        predecessor = ("scales",)
    merged = {}
    if vector is not None:
        cast_ops = float(burst_elems) if output_cast else 0.0
        amount = vector_share * row_frac + float(burst_elems) * float(merges - 1) + cast_ops
        if amount > 0:
            merged[vector] = amount
    if merged:
        vec_place = _role_placement_for_units(context, names, cores, units, ("vector",))
        stages.append(Stage("reduce", merged, after=predecessor, placement=vec_place))
        predecessor = ("reduce",)
    store = {}
    if mte3 is not None and mte3_share > 0:
        store[mte3] = mte3_share * row_frac
    if hbm is not None and hbm_write_share > 0:
        store[hbm] = hbm_write_share * row_frac
    if store:
        st_place = _role_placement_for_units(context, names, cores, units, ("mte3",))
        stages.append(Stage("final_store", store, after=predecessor,
                            caps=scaled_store_caps, placement=st_place))
    return KernelProgram(tuple(stages))


def _tile_merge_program(context, names, tile_m: int, ws_tile_n: int,
                        vector_share, mte3_share, hbm_write_share,
                        scale_copy: float, merges: int, output_cast: bool,
                        read_caps, store_caps, index: int, cores: int, *,
                        align_burst: bool = False,
                        mem_tier: str | None = None,
                        vread_role: str | None = None,
                        vread_name: str | None = None,
                        scale_role: str | None = None,
                        scale_name: str | None = None):
    chosen, m_burst_base, m_burst_cnt, m_burst_tail, full_e, tail_e = _row_burst_partition(
        context, tile_m=tile_m, tile_n=ws_tile_n, splits=merges,
        tile_index=index, cores=cores, align_burst=align_burst,
    )
    common = dict(
        vector_share=vector_share, mte3_share=mte3_share,
        hbm_write_share=hbm_write_share, scale_copy=scale_copy,
        merges=merges, output_cast=output_cast, read_caps=read_caps,
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
                context, names, units=chosen[:-1], row_frac=full_rows / total_rows,
                burst_elems=float(full_e),
                partial_read=float(FP32_BYTES) * float(merges) * float(full_e),
                **common),
            _merge_burst_program(
                context, names, units=(chosen[-1],), row_frac=tail_rows / total_rows,
                burst_elems=float(tail_e),
                partial_read=float(FP32_BYTES) * float(merges) * float(tail_e),
                **common),
        ))
    total_elems = float(full_e + tail_e)
    return _merge_burst_program(
        context, names, units=chosen, row_frac=1.0, burst_elems=total_elems,
        partial_read=float(FP32_BYTES) * float(merges) * total_elems, **common)


def _record_traffic(record, *, split_count, slices, tasks, partial_write_bytes,
                    partial_read_bytes, fixpipe_write_bytes, clear_write_bytes,
                    vector_merge, vector_cast, final_write, cube_total,
                    declared_hbm, issued_vector, output_tiles: int,
                    dp_tiles: int = 0, sk_tiles: int = 0,
                    l2_resident: bool = False, fixpipe_present: bool = False):
    if record is None:
        return
    if isinstance(record.get("selected"), dict):
        record["selected"] = {**record["selected"], "k_splits": int(split_count)}
    ws_mem_total = float(partial_write_bytes) + float(partial_read_bytes)
    partial_hbm = 0.0 if l2_resident else ws_mem_total
    partial_l2 = ws_mem_total if l2_resident else 0.0
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
            "partial_write_bytes": float(partial_write_bytes),
            "partial_read_bytes": float(partial_read_bytes),
            "workspace_clear_bytes": float(clear_write_bytes),
            "final_write_bytes": float(final_write),
            "merge_vector_ops": float(vector_merge),
            "cast_vector_ops": float(vector_cast),
            "merge_latency_ms": 0,
            "declared_cube_work": float(cube_total),
            "issued_cube_work": float(cube_total),
            "issued_hbm_work": float(declared_hbm + partial_hbm),
            "issued_l2_work": partial_l2,
            "issued_fixpipe_work": float(fixpipe_write_bytes) if fixpipe_present else 0.0,
            "workspace_residency": "l2" if l2_resident else "hbm",
            "issued_vector_work": float(issued_vector),
            "tasks": len(tasks),
            "output_tiles": int(output_tiles),
            "dp_tiles": int(dp_tiles),
            "sk_tiles": int(sk_tiles),
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


def _workspace_totals(context, tile_shapes, *, splits: int, cores: int,
                      nd_fixpipe_1_2: bool, align_burst: bool):
    """Per-tile and aggregate Fixpipe write, AIV clear, and AIV read/reduce volumes."""
    total_aivs = max(1, int(context.resource_units.get("vector", cores)))
    stream_aivs = context.stream_owner_units("vector")
    aiv_count = len(stream_aivs) if stream_aivs else total_aivs
    tile_ws_ns: list[int] = []
    fixpipe_bytes = 0.0
    clear_bytes = 0.0
    aiv_elems = 0.0
    for tm, tn in tile_shapes:
        ws_tn, fixpipe_tn, clear_tn = _tile_workspace_widths(tn, nd_fixpipe_1_2=nd_fixpipe_1_2)
        tile_ws_ns.append(ws_tn)
        fixpipe_bytes += float(FP32_BYTES * splits * tm * fixpipe_tn)
        clear_bytes += float(FP32_BYTES * splits * tm * clear_tn)
        _, _, _, _, full_e, tail_e = _aiv_burst_layout(
            tm, ws_tn, splits=splits, cores=cores,
            aiv_count=aiv_count, align_burst=align_burst)
        aiv_elems += float(full_e + tail_e)
    return tile_ws_ns, fixpipe_bytes, clear_bytes, aiv_elems


def _build_dp_tile_program(load_work, mte1_work, cube_key, cube_val, *,
                           store_name, hbm, vector, scale_name, store_bytes,
                           hbm_bytes, vector_bytes, scale_copy, k_slice, tile_k,
                           load_caps, store_caps, load_place, mte1_place,
                           mma_place, store_place, vec_place, scale_place,
                           l0c_alloc, pipelined: bool) -> KernelProgram:
    """Direct DP tile (Cube->Fixpipe/MTE3->GM) with zero partials or AIV merge."""
    store_work: dict[str, float] = {}
    if store_name is not None and store_bytes > 0.0:
        store_work[store_name] = store_bytes
    if hbm is not None and hbm_bytes > 0.0:
        store_work[hbm] = hbm_bytes
    hold_l0c = bool(store_work)
    base = _build_split_slice_stages(
        load_work, mte1_work, cube_key, cube_val,
        fixpipe=None, mte3=None, mem_tier=None, fixpipe_area=0.0, clear_area=0.0,
        k_slice=k_slice, tile_k=tile_k, load_caps=load_caps, write_caps={},
        load_place=load_place, mte1_place=mte1_place, mma_place=mma_place,
        fixpipe_place={}, clear_place={}, l0c_alloc=l0c_alloc,
        pipelined=pipelined, defer_l0c_free=hold_l0c)
    stages = list(base.stages)
    predecessor = stages[-1].name if stages else ""
    if scale_copy > 0.0 and scale_name is not None:
        stages.append(Stage("dp_scales", {scale_name: scale_copy},
                            after=(predecessor,) if predecessor else (),
                            placement=scale_place))
        predecessor = "dp_scales"
    if vector is not None and vector_bytes > 0.0:
        stages.append(Stage("dp_epilogue", {vector: vector_bytes},
                            after=(predecessor,) if predecessor else (),
                            placement=vec_place))
        predecessor = "dp_epilogue"
    if store_work:
        stages.append(Stage("dp_store", store_work,
                            after=(predecessor,) if predecessor else (),
                            caps=store_caps, placement=store_place,
                            free=l0c_alloc))
    return KernelProgram(tuple(stages))


def _group_dp_waves(dp_entries: list[tuple[int, KernelProgram]]) -> ParallelProgram:
    """Chain DP tiles assigned to the same physical AIC serially to avoid L0C overlap."""
    by_core: dict[int, list[KernelProgram]] = {}
    for phys_core, prog in dp_entries:
        by_core.setdefault(phys_core, []).append(prog)
    branches = []
    for progs in by_core.values():
        if len(progs) == 1:
            branches.append(progs[0])
        else:
            chained = []
            prev = ()
            for idx, prog in enumerate(progs):
                name = f"dp_w{idx}"
                chained.append(Stage(name, after=prev, program=prog))
                prev = (name,)
            branches.append(KernelProgram(tuple(chained)))
    return ParallelProgram(tuple(branches))


def _build_splitk_programs(
    context, tasks, loads, mte1_loads, cubes, *,
    dp_task_indices, sk_task_indices, sk_unique_tiles, sk_tile_shapes,
    tile_ws_ns, all_vector_shares, all_mte3_shares, all_hbm_shares,
    names, cube, hbm, vector, mte3, fixpipe, mem_tier, vread_role, vread_name,
    scale_role, scale_name, dp_store_role, dp_store_name, epilogue,
    total_m, n, groups, reuse_inputs, reuse_weights, payload_width,
    tile_k_dim, splits, output_cast, arch35, eff_nd_fixpipe, cores, eff_cores,
    stream_aic, stream_aiv, dp_split_caps, sk_split_caps, write_caps,
    merge_read_caps, merge_store_caps, used_cores, pipelined: bool,
):
    """Build per-tile DP wave, SK slice, and AIV merge programs."""
    load_roles = tuple(role for role in ("mte2_aic", "mte2") if role in names)
    mte1_roles = ("mte1",) if "mte1" in names else ()
    mma_roles = ("cube",) if "cube" in names else ()
    fixpipe_roles = ("fixpipe",) if "fixpipe" in names else ()
    l0c_pool = context.pool_names.get("l0c")
    total_aivs = max(1, int(context.resource_units.get("vector", cores)))
    aiv_ratio = max(1, total_aivs // cores)
    has_scales = (epilogue.row_scale_bytes + epilogue.column_scale_bytes) > 0.0

    def _slice_l0c(index: int, tm: int, tn: int, phys_core: int):
        size = float(tm * tn * FP32_BYTES)
        if l0c_pool is not None and size > 0.0:
            return (Allocation(f"splitk_l0c_{index}", l0c_pool, size, (phys_core,)),)
        return ()

    def _scale_bytes(tm: int, tn: int) -> float:
        if not has_scales:
            return 0.0
        return epilogue.scales(
            tm, tn, total_m, n, groups,
            reuse_inputs=reuse_inputs, reuse_weights=reuse_weights)[0]

    dp_used = max(1, min(eff_cores, len(dp_task_indices))) if dp_task_indices else 1
    dp_entries: list[tuple[int, KernelProgram]] = []
    for dp_pos, index in enumerate(dp_task_indices):
        tile_idx, tm, tn, _area, k_slice = tasks[index]
        phys_core = int(stream_aic[dp_pos % len(stream_aic)]) if stream_aic else dp_pos % cores
        load_caps = _split_stage_caps(
            context, dp_split_caps, loads[index], used_cores=dp_used,
            tile_m=float(tm), tile_n=float(tn), k_slice=k_slice,
            elem_bytes=payload_width, cube_key=cube, core_index=phys_core, cores=cores)
        prog = _build_dp_tile_program(
            loads[index], mte1_loads[index], cube, cubes[index],
            store_name=dp_store_name, hbm=hbm, vector=vector, scale_name=scale_name,
            store_bytes=all_mte3_shares[tile_idx], hbm_bytes=all_hbm_shares[tile_idx],
            vector_bytes=all_vector_shares[tile_idx], scale_copy=_scale_bytes(tm, tn),
            k_slice=k_slice, tile_k=tile_k_dim, load_caps=load_caps, store_caps={},
            load_place=_placement(context, phys_core, cores, load_roles),
            mte1_place=_placement(context, phys_core, cores, mte1_roles),
            mma_place=_placement(context, phys_core, cores, mma_roles),
            store_place=_placement(
                context, phys_core, cores, (dp_store_role,) if dp_store_role in names else ()),
            vec_place=_placement(context, phys_core, cores, ("vector",) if "vector" in names else ()),
            scale_place=_placement(context, phys_core, cores, (scale_role,) if scale_role in names else ()),
            l0c_alloc=_slice_l0c(index, tm, tn, phys_core), pipelined=pipelined)
        dp_entries.append((phys_core, prog))

    sk_progs = []
    for sk_pos, index in enumerate(sk_task_indices):
        _tile_index, tm, tn, _area, k_slice = tasks[index]
        phys_core = int(stream_aic[sk_pos % len(stream_aic)]) if stream_aic else sk_pos % cores
        load_caps = _split_stage_caps(
            context, sk_split_caps, loads[index], used_cores=used_cores,
            tile_m=float(tm), tile_n=float(tn), k_slice=k_slice,
            elem_bytes=payload_width, cube_key=cube, core_index=phys_core, cores=cores)
        _, fixpipe_tn, clear_tn = _tile_workspace_widths(tn, nd_fixpipe_1_2=eff_nd_fixpipe)
        clear_aiv = (
            int(stream_aiv[(sk_pos * aiv_ratio) % len(stream_aiv)])
            if stream_aiv else (phys_core * aiv_ratio) % total_aivs
        )
        clear_place = (
            _role_placement_for_units(context, names, cores, (clear_aiv,), ("mte3",))
            if clear_tn > 0 else {}
        )
        sk_progs.append(_build_split_slice_stages(
            loads[index], mte1_loads[index], cube, cubes[index],
            fixpipe=fixpipe, mte3=mte3, mem_tier=mem_tier,
            fixpipe_area=float(tm * fixpipe_tn), clear_area=float(tm * clear_tn),
            k_slice=k_slice, tile_k=tile_k_dim, load_caps=load_caps,
            write_caps=write_caps,
            load_place=_placement(context, phys_core, cores, load_roles),
            mte1_place=_placement(context, phys_core, cores, mte1_roles),
            mma_place=_placement(context, phys_core, cores, mma_roles),
            fixpipe_place=_placement(context, phys_core, cores, fixpipe_roles),
            clear_place=clear_place, l0c_alloc=_slice_l0c(index, tm, tn, phys_core),
            pipelined=pipelined))

    merges = tuple(
        _tile_merge_program(
            context, names, tm, tile_ws_ns[sk_i], all_vector_shares[tile_idx],
            all_mte3_shares[tile_idx], all_hbm_shares[tile_idx],
            _scale_bytes(tm, tn), splits, output_cast, merge_read_caps,
            merge_store_caps, sk_i, cores, align_burst=arch35,
            mem_tier=mem_tier, vread_role=vread_role, vread_name=vread_name,
            scale_role=scale_role, scale_name=scale_name)
        for sk_i, (tile_idx, (tm, tn)) in enumerate(zip(sk_unique_tiles, sk_tile_shapes))
    )
    return dp_entries, tuple(sk_progs), merges


def splitk_program(context, split_count: int, *, recipe=None, geometry=None,
                   record=None, ws_l2_resident: bool | None = None,
                   k_align: int | None = None,
                   nd_fixpipe_1_2: bool | None = None,
                   arch35_streamk: bool | None = None) -> KernelProgram:
    """Lower an underfilled output grid (or Arch35 DP+SK tail wave) onto tasks."""
    count = max(1, int(split_count))
    names = context.resource_names
    hbm, vector, mte3 = names.get("hbm"), names.get("vector"), names.get("mte3")
    l2_res, fixpipe = names.get("l2"), names.get("fixpipe")
    vread_role = (
        "mte2_aiv" if "mte2_aiv" in names
        else ("mte2" if "mte2" in names else None)
    )
    vread_name = names.get(vread_role) if vread_role is not None else None
    cube = names.get("cube") or next(
        (key for key in context.work if key.startswith("matrix@")), None)
    total_m, n, k, tile_dims = _local_geometry(context, recipe, geometry)
    cores = max(1, int(context.core_count))
    stream_aic = context.stream_owner_units("cube")
    stream_aiv = context.stream_owner_units("vector")
    eff_cores = len(stream_aic) if stream_aic else cores
    meta = dict(context.op.metadata)
    compute_fmt = str(meta.get("compute_format") or context.op.dtype or "").strip()
    if not compute_fmt:
        raise ModelingError("splitk_compute_format_missing")
    payload_width = _payload_dtype_bytes(compute_fmt)
    arch35 = is_arch35_context(context) if arch35_streamk is None else bool(arch35_streamk)
    tail_tiles = len(tile_dims) % eff_cores
    is_dpsk = (
        arch35 and count > 1 and len(tile_dims) > eff_cores
        and 0 < tail_tiles <= eff_cores // 2
    )
    dp_tile_count = (len(tile_dims) - tail_tiles) if is_dpsk else 0
    eff_k_align = (
        1 if is_dpsk
        else (int(k_align) if k_align is not None
              else _streamk_k_align(meta, compute_fmt, payload_width, is_arch35=arch35))
    )
    eff_nd_fixpipe = (
        False if is_dpsk
        else (bool(nd_fixpipe_1_2) if nd_fixpipe_1_2 is not None
              else _uses_nd_fixpipe_1_2(total_m, n, is_arch35=arch35))
    )
    slices = _k_slices(k, count, align=eff_k_align, fixed_stride=arch35)
    volume = total_m * n * k
    tasks, weights = [], []
    if volume > 0:
        for tile_index, (tm, tn) in enumerate(tile_dims):
            area = float(tm * tn)
            tile_slices = (int(k),) if tile_index < dp_tile_count else slices
            for k_slice in tile_slices:
                weight_share = area * float(k_slice) / volume
                if weight_share > 0:
                    tasks.append((tile_index, tm, tn, area, float(k_slice)))
                    weights.append(weight_share)
    if not tasks:
        return KernelProgram((Stage("setup", latency=context.setup_ms),),
                             minimum_ms=context.minimum_ms)
    splits = len(slices)
    pt = getattr(context, "primitive_table", None)
    groups = (
        geometry.groups if geometry is not None
        else max(1, int(meta.get("matrix_groups", 1) or 1))
    )
    loads, mte1_loads, hbm_total, mte3_total, hbm_write, epilogue = _splitk_task_loads(
        context, tasks, weights, names=names, hbm=hbm, vector=vector,
        mte3=mte3, cube=cube, total_m=total_m, n=n, k=k, groups=groups,
        compute_width=payload_width,
        directional_mte1=(pt is not None and directional_mte1_available(context)),
        recipe=recipe,
    )
    output_cast = _needs_output_cast(context.op, meta, epilogue)
    scale_role = read_role(names, aiv=0)
    scale_name = names.get(scale_role)
    dp_store_role = plain_store_role(names, epilogue)
    dp_store_name = names.get(dp_store_role)
    cube_total = float(context.work.get(cube, 0)) if cube is not None else 0.0
    vector_total = float(context.work.get(vector, 0)) if vector is not None else 0.0
    cubes = _allocate(cube_total, weights)
    all_weights = [float(tm * tn) / (total_m * n) for tm, tn in tile_dims]
    all_vector_shares = _allocate(vector_total, all_weights)
    all_mte3_shares = _allocate(mte3_total, all_weights)
    all_hbm_shares = _allocate(hbm_write, all_weights)
    sk_unique_tiles = sorted({task[0] for task in tasks if task[0] >= dp_tile_count})
    sk_tile_shapes = [tile_dims[tile] for tile in sk_unique_tiles]
    tile_ws_ns, fixpipe_total, clear_total, aiv_elems_total = _workspace_totals(
        context, sk_tile_shapes, splits=splits, cores=cores,
        nd_fixpipe_1_2=eff_nd_fixpipe, align_burst=arch35)
    has_partial_sink = hbm is not None or l2_res is not None or fixpipe is not None
    partial_write = (fixpipe_total + clear_total) if has_partial_sink else 0.0
    partial_read = (float(FP32_BYTES) * float(splits) * aiv_elems_total
                    if (vread_name is not None or hbm is not None or l2_res is not None) else 0.0)
    l2_resident = (
        _partials_l2_resident(context, max(partial_write, partial_read))
        if ws_l2_resident is None
        else (bool(ws_l2_resident) and l2_res is not None)
    )
    mem_tier = l2_res if l2_resident else hbm
    vector_merge = float(splits - 1) * aiv_elems_total if vector is not None else 0.0
    vector_cast = aiv_elems_total if (vector is not None and output_cast) else 0.0
    tile_m_dim = min(total_m, float(recipe.tile_m)) if recipe is not None else total_m
    tile_n_dim = min(n, float(recipe.tile_n)) if recipe is not None else n
    tile_k_dim = int(recipe.tile_k) if recipe is not None else 0
    sk_task_indices = [idx for idx, task in enumerate(tasks) if task[0] >= dp_tile_count]
    dp_task_indices = [idx for idx, task in enumerate(tasks) if task[0] < dp_tile_count]
    used_cores = max(1, min(eff_cores, len(sk_task_indices)))
    m_tiles = max(1, math.ceil(total_m / max(1.0, tile_m_dim)))
    cap_kw = dict(
        row_bytes=math.ceil(tile_n_dim * payload_width),
        stride_bytes=math.ceil(n * payload_width),
        m_tiles=m_tiles, panel_rows=max(1, int(k)),
    )
    sk_split_caps = _per_owner_cap(context, len(sk_task_indices), **cap_kw)
    dp_split_caps = (
        _per_owner_cap(context, min(eff_cores, len(dp_task_indices)), **cap_kw)
        if dp_task_indices else {}
    )
    write_caps = _partial_write_caps(
        context, sk_split_caps, total_partial_bytes=partial_write,
        active_tasks=max(1, len(sk_task_indices)), l2_resident=l2_resident)
    merge_read_caps, merge_store_caps = _merge_owner_cap(
        context, max(1, len(sk_tile_shapes)), total_partial_bytes=partial_read,
        l2_resident=l2_resident)
    dp_entries, sk_progs, merges = _build_splitk_programs(
        context, tasks, loads, mte1_loads, cubes,
        dp_task_indices=dp_task_indices, sk_task_indices=sk_task_indices,
        sk_unique_tiles=sk_unique_tiles, sk_tile_shapes=sk_tile_shapes,
        tile_ws_ns=tile_ws_ns, all_vector_shares=all_vector_shares,
        all_mte3_shares=all_mte3_shares, all_hbm_shares=all_hbm_shares,
        names=names, cube=cube, hbm=hbm, vector=vector, mte3=mte3,
        fixpipe=fixpipe, mem_tier=mem_tier, vread_role=vread_role,
        vread_name=vread_name, scale_role=scale_role, scale_name=scale_name,
        dp_store_role=dp_store_role, dp_store_name=dp_store_name,
        epilogue=epilogue, total_m=total_m, n=n, groups=groups,
        reuse_inputs=bool(getattr(recipe, "reuse_inputs", True)),
        reuse_weights=bool(getattr(recipe, "reuse_weights", True)),
        payload_width=payload_width, tile_k_dim=tile_k_dim, splits=splits,
        output_cast=output_cast, arch35=arch35, eff_nd_fixpipe=eff_nd_fixpipe,
        cores=cores, eff_cores=eff_cores, stream_aic=stream_aic,
        stream_aiv=stream_aiv, dp_split_caps=dp_split_caps,
        sk_split_caps=sk_split_caps, write_caps=write_caps,
        merge_read_caps=merge_read_caps, merge_store_caps=merge_store_caps,
        used_cores=used_cores, pipelined=(pt is not None))
    dp_fixpipe_bytes = (
        sum(all_mte3_shares[tile_idx] for tile_idx in range(dp_tile_count))
        if (is_dpsk and dp_store_role == "fixpipe" and dp_store_name is not None) else 0.0
    )
    _record_traffic(
        record, split_count=splits, slices=slices, tasks=tasks,
        partial_write_bytes=partial_write, partial_read_bytes=partial_read,
        fixpipe_write_bytes=(fixpipe_total if has_partial_sink else 0.0) + dp_fixpipe_bytes,
        clear_write_bytes=clear_total if has_partial_sink else 0.0,
        vector_merge=vector_merge, vector_cast=vector_cast,
        final_write=hbm_write, cube_total=cube_total, declared_hbm=hbm_total,
        issued_vector=vector_total + vector_merge + vector_cast,
        output_tiles=len(tile_dims), dp_tiles=dp_tile_count, sk_tiles=len(sk_unique_tiles),
        l2_resident=l2_resident, fixpipe_present=(fixpipe is not None))
    total_setup_ms = float(context.setup_ms or 0.0) + matrix_kernel_setup_ms(
        context, is_streamk=(splits > 1))
    sync_ms = streamk_fixup_latency_ms(context, k_splits=splits, used_cores=used_cores)
    top_stages = [Stage("setup", latency=total_setup_ms)]
    split_after = ("setup",)
    if dp_entries:
        top_stages.append(Stage("dp_tiles", after=("setup",),
                                program=_group_dp_waves(dp_entries)))
        split_after = ("dp_tiles",)
    top_stages.append(Stage("splits", after=split_after, program=ParallelProgram(sk_progs)))
    if sync_ms > 0.0:
        top_stages.append(Stage("split_sync", after=("splits",), latency=sync_ms, keep_empty=True))
        merge_after = ("split_sync",)
    else:
        merge_after = ("splits",)
    top_stages.append(Stage("merge", after=merge_after, program=ParallelProgram(merges)))
    return KernelProgram(tuple(top_stages), minimum_ms=context.minimum_ms)
