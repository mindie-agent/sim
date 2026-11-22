"""Finite analytical mapping selection for the hierarchical matrix program.

[Role] Derive finite matrix geometry/owner tails, screen declared tile
  candidates, and expose one isolated matrix-operator entry.
[Invariants] Selection uses shape, rates, ownership and capacities, never timings
  or model labels. The selected mapping executes on the ordinary causal Engine.
  Its selection bound is neither the execution duration nor an optimality claim.
  Logical/local/storage/issued extents remain separate values.
[Timeouts & Locks] O(candidates * cores) arithmetic plus bounded explicit
  owner enumeration, no I/O or locks.
[Side-Effect Policy] Emits bounded mapping diagnostics through the caller only;
  the public entry runs only the caller's virtual Engine.
"""
from copy import deepcopy
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction as Fraction
from itertools import count as _count
import math

from mindie_sim.schema.common import ModelingError, dtype_bytes
from .matrix import DEFAULT_COMPUTE_TILE_MN, HierarchicalMatmul
from .matrix_epilogue import MatrixEpilogue
from .matrix_mapping_owners import (
    MS_PER_SECOND,
    _declared_rate,
    critical_output_owner_rate_fraction,
    freeze_owner_tiles,
    gemm_fractal_operand_bytes,
    grouped_owner_rectangles,
    grouped_owner_tiles,
    is_arch35_context,
    mte_owner_unit_count,
    owner_records,
    owner_tiles,
    physical_owner_rate,
    physical_owner_units,
    primitive_cold_hbm_bps,
    primitive_l2_params,
    primitive_nd2nz_mte2_rate_bps_per_aic,
    primitive_p01_scaling_bps,
    streamk_fixup_latency_ms,
)
from mindie_sim.prediction.mechanism import Work as _Work
from mindie_sim.prediction.ownership import read_role
from .matrix_traffic import FP32_BYTES

GEMM_SHAPE_AXES = 3


REFILL_M = 128
MIN_NARROW_N = 32
MAX_SPLIT_K = 32
A5_L2_MTE2_RATE = 1e13
A5_BF16_PEAK_FLOOR = 4e14
_BF16_SUBTILE_REFERENCE = _Work(matrix={"bf16": A5_BF16_PEAK_FLOOR / 1000})
#: Upstream DAV_3510 Stream-K thresholds (arch35/matmul_v3_basic_streamk_tiling.cpp:257-262).
STREAM_K_MIN_K_BYTES = 8192
STREAM_K_PER_AIC_K_BYTES = 256
STREAM_K_K_ALIGN = 256
#: Upstream DAV_2201 Split-K thresholds (matmul_v3_base_tiling.cpp:61, 87, 2571).
DAV2201_SPLITK_K_PER_AIC = 384
DAV2201_SPLITK_LARGE_K = 27392
DAV2201_SPLITK_FP32_MIN_K = 6144
DAV2201_SPLITK_FP32_MAX_MN = 64
SPLIT_K_HALF_CORE_DIVISOR = 2
#: Minimum N/K for DAV_3510 BF16 macro-tile Stream-K vs underfilled subtiles.
ARCH35_MACRO_MIN_N = 256
ARCH35_MACRO_MIN_K = 2048

DEFAULT_CANDIDATES = ((128, 128, 64), (128, 128, 128), (128, 256, 64), (128, 256, 128),
                      (256, 128, 64), (256, 128, 128), (256, 256, 64), (256, 256, 128))
LEGACY_128_CANDIDATES = ((128, 128, 128), (128, 128, 64))
NARROW_CANDIDATES = ((16, 16, 64), (32, 16, 64), (64, 16, 64),
                     (16, 32, 64), (32, 32, 64), (64, 32, 64))
# FP32 projections may expose smaller independent M/N blocks. Ordinary low
# precision projections retain their declared 128-element base mapping: a
# feasible block is not evidence that the deployed kernel dispatches it.
UNDERFILLED_CANDIDATES = tuple((m, n, k) for m in (16, 32, 64)
                              for n in (16, 32, 64) for k in (64, 128))
#: Upstream DAV_2201 CalBaseSize small-M candidates (matmul_v3_base_tiling.cpp:47, 103).
DAV2201_SMALL_M_CANDIDATES = tuple(
    (m, n, k) for m in (16, 32, 64) for n in (128, 256) for k in (64, 128)
)


@dataclass(frozen=True)
class MatrixGeometry:
    """The four distinct matrix geometries plus finite owner/tail assignments.

    ``logical`` is the requested semantic extent, ``local`` is the rank/group
    extent executed here, ``storage`` is the declared padded address extent,
    and ``issued`` is the macro-tile extent that the selected mapping really
    launches.  Keeping them separate prevents a padded storage shape from
    silently becoming useful work, and prevents a logical tail from becoming a
    full tile charge without a declared mapping.
    """

    logical_shape: tuple[int, int, int]
    local_shape: tuple[int, int, int]
    storage_shape: tuple[int, int, int]
    issued_shape: tuple[int, int, int]
    groups: int
    group_rows: tuple[int, ...]
    explicit_groups: bool
    tile_m: int
    tile_n: int
    tile_k: int
    compute_m: int
    compute_n: int
    owners: tuple[tuple[int, int, int, int], ...]

    @property
    def m_per_group(self):
        """Per-group M extent used by the macro tile launcher.

        Equal groups use the exact quotient.  Explicit unequal groups use the
        widest group for issued macro extents; per-owner records retain the
        actual tail rows.
        """
        return max(self.group_rows, default=0)

    @property
    def n(self):
        return self.local_shape[1]

    @property
    def k(self):
        return self.local_shape[2]

    @property
    def total_rows(self):
        return sum(self.group_rows)

    @property
    def nm(self):
        return (math.ceil(self.m_per_group / self.tile_m)
                if self.m_per_group else 0)

    @property
    def nn(self):
        return math.ceil(self.n / self.tile_n) if self.n else 0

    @property
    def nk(self):
        return math.ceil(self.k / self.tile_k) if self.k else 0

    @property
    def tiles(self):
        return sum(record[0] for record in self.owners)

    @property
    def tile_counts(self):
        return {"m": self.nm, "n": self.nn, "k": self.nk}

    @property
    def output_tiles(self):
        return self.tiles

    @property
    def owner_waves(self):
        """Largest finite output-tile run assigned to one owner."""
        return max((record[0] for record in self.owners), default=0)

    @property
    def useful_flops(self):
        return 2 * self.total_rows * self.n * self.k

    @property
    def macro_tile_area(self):
        """Full macro-tile output area, including tail padding."""
        return sum(record[0] * self.tile_m * self.tile_n for record in self.owners)

    @property
    def macro_tile_envelope(self):
        """Full macro-tile arithmetic envelope, including M/N/K tails.

        This is the finite dispatch extent of the selected macro mapping, not
        executed/costed work and not a native instruction schedule.  Each
        fired tile carries the declared tile shape even when its useful tail
        is smaller; ``context.work``/``serviced`` remain the costed work.
        """
        return 2 * self.macro_tile_area * self.nk * self.tile_k

    def to_dict(self):
        return {
            "logical_shape": list(self.logical_shape),
            "local_shape": list(self.local_shape),
            "storage_shape": list(self.storage_shape),
            "issued_shape": list(self.issued_shape),
            "groups": self.groups,
            "group_rows": list(self.group_rows) if self.explicit_groups else [],
            "explicit_groups": self.explicit_groups,
            "tile": [self.tile_m, self.tile_n, self.tile_k],
            "compute_tile": [self.compute_m, self.compute_n],
            "tile_counts": {"m": self.nm, "n": self.nn, "k": self.nk},
            "owner_count": len(self.owners),
            "output_tiles": self.tiles,
            "owner_waves": self.owner_waves,
            "useful_flops": self.useful_flops,
            "macro_tile_area": self.macro_tile_area,
            "macro_tile_envelope": self.macro_tile_envelope,
            "macro_tile_envelope_basis": "full_macro_tile_extent_not_costed_work",
            "owners": [
                {"owner": index, "tiles": count, "m_extent": m_extent,
                 "n_extent": n_extent, "area": area}
                for index, (count, m_extent, n_extent, area)
                in enumerate(self.owners)
            ],
        }


def _validated_shape(value, fallback):
    if value is None or len(value) < 3:
        return fallback
    shape = tuple(int(item) for item in value[-3:])
    if any(item < 0 for item in shape):
        raise ModelingError("matrix_geometry_requires_nonnegative_shapes")
    return shape


def matrix_geometry(context, recipe, *, owners=None, group_rows=None):
    """Derive the common logical/local/storage/issued geometry once.

    ``owners`` accepts finite explicit owner rectangles for an unequal-group or
    source adapter.  Without it, equal group rows use the closed-form
    ``owner_tiles`` recurrence; unequal ``group_rows`` use the bounded explicit
    aggregate.  The returned object is the single shape/owner source consumed
    by mapping bounds, footprint screening and the executing recipes.
    """
    op = context.op
    local = _validated_shape(op.local_shape, ())
    if len(local) != 3:
        raise ModelingError("matrix_geometry_requires_three_local_dimensions")
    total_m, n, k = local
    groups = int(dict(op.metadata).get("matrix_groups", 1))
    if groups < 1:
        raise ModelingError("matrix_groups_require_positive_integer")
    if min(int(recipe.tile_m), int(recipe.tile_n), int(recipe.tile_k),
           int(recipe.compute_m), int(recipe.compute_n)) < 1:
        raise ModelingError("matrix_tile_dimensions_must_be_positive")
    if (int(recipe.tile_m) % int(recipe.compute_m)
            or int(recipe.tile_n) % int(recipe.compute_n)):
        raise ModelingError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
    if owners is not None:
        records = tuple(tuple(int(item) for item in row) for row in owners)
        if len(records) > int(context.core_count):
            raise ModelingError("matrix_explicit_owner_count_exceeds_physical_owners")
        if any(len(row) != 4 or row[0] < 0 or row[1] < 0 or row[2] < 0 or row[3] < 0
               for row in records):
            raise ModelingError("matrix_explicit_owner_records_invalid")
    else:
        records = None

    if group_rows is None:
        if total_m % groups:
            raise ModelingError("matrix_groups_require_equal_integer_rows")
        per_group = total_m // groups
        group_rows = (per_group,) * groups
        explicit_groups = False
    else:
        group_rows = tuple(int(row) for row in group_rows)
        if len(group_rows) != groups or any(row < 0 for row in group_rows):
            raise ModelingError("matrix_group_rows_require_one_nonnegative_row_count_per_group")
        if sum(group_rows) != total_m:
            raise ModelingError("matrix_group_rows_must_sum_to_local_rows")
        explicit_groups = True

    if records is None:
        if len(set(group_rows)) == 1:
            records = owner_tiles(group_rows[0], n, recipe.tile_m, recipe.tile_n,
                                  context.core_count, groups)
        else:
            records = grouped_owner_tiles(
                group_rows, n, recipe.tile_m, recipe.tile_n, context.core_count)
    else:
        if sum(row[3] for row in records) != total_m * n:
            raise ModelingError("matrix_explicit_owner_area_must_cover_local_output")
    if sum(row[3] for row in records) != total_m * n:
        raise ModelingError("matrix_owner_area_must_cover_local_output")

    logical = _validated_shape(op.semantic_shape, local)
    padded = _validated_shape(op.padded_shape, local)
    storage = tuple(max(local[index], padded[index]) for index in range(GEMM_SHAPE_AXES))
    max_rows = max(group_rows, default=0)
    issued = (
        math.ceil(max_rows / recipe.tile_m) * recipe.tile_m,
        math.ceil(n / recipe.tile_n) * recipe.tile_n,
        math.ceil(k / recipe.tile_k) * recipe.tile_k,
    )
    return MatrixGeometry(
        logical_shape=logical, local_shape=local, storage_shape=storage,
        issued_shape=issued, groups=groups, group_rows=group_rows,
        explicit_groups=explicit_groups, tile_m=int(recipe.tile_m),
        tile_n=int(recipe.tile_n), tile_k=int(recipe.tile_k),
        compute_m=int(recipe.compute_m), compute_n=int(recipe.compute_n),
        owners=tuple(records),
    )


def _primitive_mte1_owner_demands(context, left_owners, right_owners, *,
                                  inputs, weight, total_m, n, k, groups, width):
    """Directional L1->L0A / L1->L0B service demands when calibrated modes exist."""
    modes = getattr(context, "service_modes", None) or {}
    l0a_rate = float(modes["mte1_l0a"][1]) / MS_PER_SECOND if "mte1_l0a" in modes else 0.0
    l0b_rate = float(modes["mte1_l0b"][1]) / MS_PER_SECOND if "mte1_l0b" in modes else 0.0
    if l0a_rate <= 0.0 or l0b_rate <= 0.0:
        return None
    units = context.resource_units.get("mte1", context.core_count)
    per_a = l0a_rate / float(max(1, units))
    per_b = l0b_rate / float(max(1, units))
    a_unit = max(inputs / total_m, k * width)
    b_unit = max(weight / (n * groups), k * width)
    return [a[3] * a_unit / per_a + b[3] * b_unit / per_b
            for a, b in zip(left_owners, right_owners)]


def _calibrated_hbm_rate_ms(context, geometry, recipe, *, width, hbm_declared):
    """Effective HBM read rate in bytes/ms from calibrated P01/P05 surfaces."""
    if getattr(context, "primitive_table", None) is None:
        return hbm_declared
    _, n, k = geometry.local_shape
    active = max(1, min(int(context.core_count), int(geometry.tiles)))
    row_bytes = int(min(recipe.tile_n, n) * width)
    stride_bytes = int(n * width)
    is_grouped = geometry.groups > 1
    coalesced = is_grouped and active >= max(1, int(geometry.nn))
    cold_bps = primitive_cold_hbm_bps(
        context, active_cores=active, row_bytes=row_bytes,
        stride_bytes=stride_bytes, total_tiles=geometry.tiles,
        is_grouped=is_grouped, is_full_row_coalesced=coalesced,
        m_tiles=int(geometry.nm), panel_rows=max(1, int(k)),
    )
    if cold_bps is None or cold_bps <= 0.0:
        return hbm_declared
    return min(hbm_declared, cold_bps / MS_PER_SECOND)


def _candidate_split_k(context, *, m, n, k, tk, output_tiles, groups, compute_format):
    """Architecture-grounded Split-K / Stream-K factor for one tile candidate."""
    if output_tiles >= context.core_count or k <= tk:
        return 1
    if groups > 1:
        return 1
    cores = max(1, int(context.core_count))
    half_cores = max(1, cores // SPLIT_K_HALF_CORE_DIVISOR)
    if is_arch35_context(context):
        elem_b = max(1, int(dtype_bytes(compute_format)))
        min_k = max(STREAM_K_MIN_K_BYTES, cores * STREAM_K_PER_AIC_K_BYTES) / float(elem_b)
        k_aligned = math.ceil(k / STREAM_K_K_ALIGN) * STREAM_K_K_ALIGN
        if k_aligned < min_k or output_tiles > half_cores:
            return 1
        return min(k // tk, max(1, cores // max(1, output_tiles)), MAX_SPLIT_K)
    fp32_small = (
        compute_format == "fp32"
        and m <= DAV2201_SPLITK_FP32_MAX_MN and n <= DAV2201_SPLITK_FP32_MAX_MN
        and k >= DAV2201_SPLITK_FP32_MIN_K
    )
    wide_k = (output_tiles <= half_cores and k >= cores * DAV2201_SPLITK_K_PER_AIC)
    if not (wide_k or k >= DAV2201_SPLITK_LARGE_K or fp32_small):
        return 1
    return min(k // tk, max(1, cores // max(1, output_tiles)), MAX_SPLIT_K)


def _arch35_streamk_perf(context, recipe, geometry, *, split_k: int, output_tiles: int) -> float:
    """Upstream DAV_3510 Stream-K perf model (arch35/matmul_v3_basic_streamk_tiling.cpp:70-129)."""
    names = context.resource_names
    cores = max(1, int(context.core_count))
    waves = max(1, math.ceil(int(output_tiles) / cores))
    used = max(1, min(cores, int(output_tiles) * int(split_k)))
    frac = float(used) / float(cores)
    hbm_full = _declared_rate(context, names["hbm"])
    meta = dict(context.op.metadata)
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    total_m, n, k = geometry.local_shape
    row_bytes = int(min(recipe.tile_n, n) * width)
    stride_bytes = int(n * width)
    strided_bps = primitive_cold_hbm_bps(
        context, active_cores=cores,
        row_bytes=row_bytes,
        stride_bytes=stride_bytes,
        total_tiles=max(1, int(output_tiles)), is_grouped=False,
        m_tiles=int(geometry.nm), panel_rows=max(1, int(k)),
    )
    if strided_bps:
        b_hbm_full = min(hbm_full, strided_bps / MS_PER_SECOND)
    elif row_bytes < stride_bytes:
        # Sub-sector strided weight rows still pull complete L2 cache sectors
        # from HBM (e.g. 128 B on DAV_3510), so host tiling must price the
        # same sector efficiency ceiling as calibrated execution.
        _, sector = primitive_l2_params(context)
        sector_eff = min(1.0, float(max(1, row_bytes)) / float(max(1, sector)))
        b_hbm_full = hbm_full * sector_eff
    else:
        b_hbm_full = hbm_full
    l2_agg = primitive_p01_scaling_bps(context, cores, warm=True)
    l2_full = (l2_agg / MS_PER_SECOND) if (l2_agg and l2_agg > 0.0) else hbm_full
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    a_bytes = inputs + epilogue.row_scale_bytes
    b_bytes = weight + epilogue.column_scale_bytes
    l2_extra = max(
        0.0,
        (inputs * max(1, geometry.nn) + weight * max(1, geometry.nm)) - (inputs + weight),
    )
    mte2_ms = (
        a_bytes / (hbm_full * frac)
        + b_bytes / (b_hbm_full * frac)
        + l2_extra / (l2_full * frac)
    )
    if waves > 1:
        # Multi-wave tile grid under-utilizes the tail wave.
        wave_eff = float(output_tiles) / float(waves * cores)
        mte2_ms /= max(wave_eff, 1.0 / float(cores))
    cube_key = names.get("cube")
    cube_total = float(context.work.get(cube_key, 0.0)) if cube_key else 0.0
    cube_units = max(1, int(context.resource_units.get("cube", cores)))
    if waves > 1:
        cube_ms = (cube_total * float(cube_units) * float(waves) / float(max(1, int(output_tiles)))) if cube_total > 0.0 else 0.0
    else:
        cube_ms = (cube_total * float(cube_units) / float(used)) if cube_total > 0.0 else 0.0
    if split_k <= 1:
        return max(mte2_ms, cube_ms)
    partial_bytes = float(2 * FP32_BYTES * split_k * total_m * n)
    vec_r = _declared_rate(context, names["vector"]) if "vector" in names else 0.0
    merge_ops = float((split_k - 1) * total_m * n)
    neg_ms = (
        streamk_fixup_latency_ms(context, k_splits=split_k, used_cores=used)
        + partial_bytes / (hbm_full * frac)
        + (merge_ops / vec_r if vec_r > 0.0 else 0.0)
    )
    return max(mte2_ms, cube_ms) + neg_ms


def _split_k_bound(context, recipe, geometry, *, bound, split_k, output_tiles):
    """Evaluate calibrated Split-K / Stream-K bound (host tiling ignores L2 warmth)."""
    if is_arch35_context(context):
        base_perf = _arch35_streamk_perf(
            context, recipe, geometry, split_k=1, output_tiles=output_tiles)
        sk_perf = _arch35_streamk_perf(
            context, recipe, geometry, split_k=split_k, output_tiles=output_tiles)
        return sk_perf if sk_perf < base_perf else bound + 1.0
    names = context.resource_names
    hbm_r = _declared_rate(context, names["hbm"])
    vec_r = _declared_rate(context, names["vector"]) if "vector" in names else 0.0
    total_m, n, k = geometry.local_shape
    partial_bytes = float(2 * FP32_BYTES * split_k * total_m * n)
    merge_ops = float((split_k - 1) * total_m * n)
    meta = dict(context.op.metadata)
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    hbm_reads = inputs + weight + epilogue.row_scale_bytes + epilogue.column_scale_bytes
    used_cores = max(1, min(int(context.core_count), int(output_tiles) * int(split_k)))
    cold_bps = primitive_cold_hbm_bps(
        context, active_cores=used_cores,
        row_bytes=int(min(recipe.tile_n, n) * width),
        stride_bytes=int(n * width),
        total_tiles=used_cores, is_grouped=False,
        m_tiles=int(geometry.nm), panel_rows=max(1, int(k)),
    )
    eff_hbm_r = min(hbm_r, cold_bps / MS_PER_SECOND) if cold_bps else hbm_r
    hbm_ms = hbm_reads / eff_hbm_r + context.op.mte3_bytes / hbm_r
    fixup_ms = streamk_fixup_latency_ms(
        context, k_splits=split_k, used_cores=used_cores)
    return (max(hbm_ms, bound / split_k) + partial_bytes / hbm_r
            + (merge_ops / vec_r if vec_r > 0.0 else 0.0) + fixup_ms)


def mapping_bound(context, recipe, *, geometry=None, allow_unequal_groups=False,
                  ignore_l2_hits=False):
    """Resource-only lower bound with private copies and actual owner tails."""
    geometry = geometry or matrix_geometry(context, recipe)
    if not allow_unequal_groups and len(set(geometry.group_rows)) > 1:
        raise ModelingError("matrix_mapping_bound_requires_equal_group_rows")
    total_m, n, k = geometry.local_shape
    if not total_m:
        return 0.0
    names, meta = context.resource_names, dict(context.op.metadata)
    groups = geometry.groups
    m = geometry.m_per_group
    epilogue = MatrixEpilogue.from_op(context.op)
    inputs, weight = epilogue.operands(context.op)
    owners = geometry.owners
    private_parts = [(inputs * a / total_m, weight * b / (n * groups))
                     for _, a, b, _ in owners]
    private = [a_b + b_b for a_b, b_b in private_parts]
    scales = [epilogue.row_scale_bytes * a / total_m
              + epilogue.column_scale_bytes * b / (n * groups)
              for _, a, b, _ in owners]
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    if len(set(geometry.group_rows)) == 1:
        left = owner_tiles(m, math.ceil(n / recipe.compute_n), recipe.tile_m,
                           recipe.tile_n // recipe.compute_n, context.core_count, groups)
        right = owner_tiles(math.ceil(m / recipe.compute_m), n,
                            recipe.tile_m // recipe.compute_m, recipe.tile_n, context.core_count, groups)
    else:
        left = grouped_owner_tiles(
            geometry.group_rows, math.ceil(n / recipe.compute_n), recipe.tile_m,
            recipe.tile_n // recipe.compute_n, context.core_count)
        right = grouped_owner_tiles(
            tuple(math.ceil(r / recipe.compute_m) for r in geometry.group_rows),
            n, recipe.tile_m // recipe.compute_m, recipe.tile_n, context.core_count)
    inner = [a[3] * max(inputs / total_m, k * width) + b[3] * max(weight / (n * groups), k * width)
             for a, b in zip(left, right)]
    if recipe.reuse_inputs and recipe.reuse_weights:
        hbm_reads = inputs + weight + epilogue.row_scale_bytes + epilogue.column_scale_bytes
    else:
        hbm_reads = (((inputs + epilogue.row_scale_bytes) *
                      (1 if recipe.reuse_inputs else math.ceil(n / recipe.tile_n)))
                     + ((weight + epilogue.column_scale_bytes) *
                        (1 if recipe.reuse_weights else math.ceil(m / recipe.tile_m))))
    admitted_l2_hit = (
        0.0 if ignore_l2_hits
        else max(0.0, float(meta.get("admitted_l2_read_hit_bytes", 0.0) or 0.0))
    )
    hbm_miss_reads = max(0.0, hbm_reads - min(admitted_l2_hit, hbm_reads))
    hbm_declared = _declared_rate(context, names["hbm"])
    pt = getattr(context, "primitive_table", None)
    active_cores = max(1, min(int(context.core_count), int(geometry.tiles)))
    if pt is not None and hbm_reads > 0.0:
        miss_frac = hbm_miss_reads / hbm_reads
        a_hbm = (inputs + epilogue.row_scale_bytes) * (1 if recipe.reuse_inputs else math.ceil(n / recipe.tile_n)) * miss_frac
        b_hbm = (weight + epilogue.column_scale_bytes) * (1 if recipe.reuse_weights else math.ceil(m / recipe.tile_m)) * miss_frac
        a_cold = primitive_p01_scaling_bps(context, active_cores, warm=False)
        a_rate = min(hbm_declared, a_cold / MS_PER_SECOND) if a_cold else hbm_declared
        b_rate = _calibrated_hbm_rate_ms(
            context, geometry, recipe, width=width, hbm_declared=hbm_declared)
        bound = a_hbm / a_rate + b_hbm / b_rate + context.op.mte3_bytes / hbm_declared
    else:
        bound = hbm_miss_reads / hbm_declared + context.op.mte3_bytes / hbm_declared
    weight_layout = str(meta.get("weight_layout", "ND"))
    dir_mte1 = (
        _primitive_mte1_owner_demands(
            context, left, right, inputs=inputs, weight=weight,
            total_m=total_m, n=n, k=k, groups=groups, width=width)
        if pt is not None else None
    )
    demands = {}
    read_op_role = read_role(names)
    for role, values in (
        (read_op_role, private),
        (read_role(names, aiv=0), scales),
        ("mte1", inner),
        ("mte3", [context.op.mte3_bytes * area / (total_m * n) for _, _, _, area in owners]),
        ("cube", [context.work.get(names["cube"], 0) * area / (total_m * n)
                  for _, _, _, area in owners]),
        ("vector", [epilogue.operations_per_output * area for _, _, _, area in owners]),
    ):
        if role not in names:
            continue
        units = context.resource_units.get(role, context.core_count)
        resource = names[role]
        declared_r = _declared_rate(context, resource)
        for core, value in enumerate(values):
            selected = context.owned(core, context.core_count, (role,))[resource]
            eff_ms = None
            if pt is not None and role == read_op_role and value > 0.0:
                a_b, b_b = private_parts[core]
                eff_bps = primitive_nd2nz_mte2_rate_bps_per_aic(
                    context, active_cores, a_b, b_b, weight_layout=weight_layout,
                    row_bytes=int(min(recipe.tile_n, n) * width),
                    stride_bytes=int(n * width),
                )
                if eff_bps is not None and eff_bps > 0.0:
                    eff_ms = value / min(declared_r / float(max(1, units)),
                                         eff_bps / MS_PER_SECOND)
            elif dir_mte1 is not None and role == "mte1" and core < len(dir_mte1):
                eff_ms = dir_mte1[core]
            for unit in selected:
                key = resource, unit
                inc = (eff_ms / len(selected) if eff_ms is not None
                       else value / len(selected) * units / declared_r)
                demands[key] = demands.get(key, 0.0) + inc
    bound = max(bound, max(demands.values(), default=0))
    return bound


def footprint(context, recipe, *, geometry=None):
    geometry = geometry or matrix_geometry(context, recipe)
    total_m, n, k = geometry.local_shape
    meta = dict(context.op.metadata)
    groups = geometry.groups
    m = geometry.m_per_group
    a, b, kd = min(m, recipe.tile_m), min(n, recipe.tile_n), min(k, recipe.tile_k)
    ai, bi = min(a, recipe.compute_m), min(b, recipe.compute_n)
    inputs, weight = MatrixEpilogue.from_op(context.op).operands(context.op)
    width = dtype_bytes(meta.get("compute_format", context.op.dtype or "bf16"))
    output = dtype_bytes(meta.get("output_format", "bf16"))
    left = inputs * a / total_m * kd / k if total_m else 0
    right = weight * b / (n * groups) * kd / k
    return {"l1": recipe.slots * (left + right), "l0c": a * b * max(FP32_BYTES, output),
            "l0a": recipe.compute_slots * max(left * ai / a if a else 0, ai * kd * width),
            "l0b": recipe.compute_slots * max(right * bi / b, bi * kd * width)}


_SELECT_CACHE = {}
_OPERATOR_SEQUENCE = _count()


def _tuples(items):
    return None if items is None else tuple(tuple(x) for x in items)


def _matmul_recipe_key(recipe):
    return (
        type(recipe), _tuples(recipe.candidates), recipe.compute_m, recipe.compute_n,
        recipe.slots, recipe.compute_slots, recipe.reuse_inputs, recipe.reuse_weights,
        _tuples(recipe.narrow_candidates), _tuples(recipe.underfilled_candidates),
        getattr(recipe, "compact_output_tiles", None),
    )


def _op_matrix_key(op):
    meta = dict(op.metadata)
    rows = meta.get("matrix_group_rows") or meta.get("aclnn_group_list")
    rows_key = tuple(int(v) for v in rows) if rows else ()
    return (
        op.local_shape, op.semantic_shape, op.padded_shape,
        op.dtype, op.hbm_bytes, op.mte2_bytes, op.mte3_bytes, op.vector_ops,
        meta.get("matrix_groups", 1), meta.get("compute_format"), meta.get("output_format"),
        meta.get("packed_weight_bytes", 0), meta.get("row_scale_bytes", 0),
        meta.get("column_scale_bytes", 0), meta.get("epilogue_ops_per_element", 0),
        float(meta.get("admitted_l2_read_hit_bytes", 0.0) or 0.0),
        bool(meta.get("l2_weight_cache_disable", False)),
        rows_key,
        MatrixEpilogue.from_op(op),
    )


def _has_subtile_supply(context) -> bool:
    """Apply the existing copy or BF16 service budget for candidate screening."""
    if float(context.rates.get("mte2_aic", 0)) >= A5_L2_MTE2_RATE:
        return True
    # Screen a fixed amount of BF16 work against a one-ms budget regardless
    # of the operator precision. The binding also works for rate-only devices.
    try:
        demand = context.bind(_BF16_SUBTILE_REFERENCE)
    except (ModelingError, ZeroDivisionError):
        return False
    return 0 <= demand[context.resource_names["cube"]] <= 1


def _resolve_select_candidates(mapper, host_ctx, *, m, n, k, groups, compute_format, rows_key):
    default_grid = (
        sum(math.ceil(r / REFILL_M) for r in rows_key) * math.ceil(n / REFILL_M)
        if rows_key is not None
        else math.ceil(m / REFILL_M) * math.ceil(n / REFILL_M) * groups
    )
    # FP32 always screens the declared subtiles so narrow projections
    # can fill more owners, even when the usual supply budget is unmet.
    fp32_subtiles = mapper.candidates is None and compute_format == "fp32"
    automatic_subtiles = fp32_subtiles or (
        _has_subtile_supply(host_ctx) and (
            mapper.candidates is None
            or m < host_ctx.core_count * REFILL_M
            or default_grid < host_ctx.core_count))
    narrow = (NARROW_CANDIDATES if automatic_subtiles else ()
              ) if mapper.narrow_candidates is None else mapper.narrow_candidates
    narrow = narrow if n <= MIN_NARROW_N else ()
    arch35_macro_streamk = (
        is_arch35_context(host_ctx)
        and compute_format != "fp32" and groups == 1
        and n >= ARCH35_MACRO_MIN_N and k >= ARCH35_MACRO_MIN_K
    )
    underfilled = (
        () if (arch35_macro_streamk and mapper.underfilled_candidates is None)
        else ((UNDERFILLED_CANDIDATES if automatic_subtiles else ())
              if mapper.underfilled_candidates is None else mapper.underfilled_candidates)
    )
    underfilled = underfilled if default_grid < host_ctx.core_count else ()
    use_default_tiling = (
        mapper.candidates is None
        or tuple(tuple(t) for t in mapper.candidates) == LEGACY_128_CANDIDATES
    )
    candidates = [(tile, False) for tile in
                  (DEFAULT_CANDIDATES if use_default_tiling else mapper.candidates)]
    if (not is_arch35_context(host_ctx) and groups > 1
            and use_default_tiling and n >= ARCH35_MACRO_MIN_N):
        # Upstream DAV_2201 SPLIT_M GroupedMatmul fixes baseN = BEST_BASEN (256)
        # (ops-transformer/gmm/grouped_matmul/op_host/op_tiling/grouped_matmul_tiling.cpp:1693-1738).
        candidates = [(tile, flag) for tile, flag in candidates if tile[1] == ARCH35_MACRO_MIN_N]
    if (not is_arch35_context(host_ctx) and groups == 1
            and use_default_tiling and default_grid < host_ctx.core_count):
        candidates.extend((tile, True) for tile in DAV2201_SMALL_M_CANDIDATES)
    candidates.extend((tile, True) for tile in narrow)
    candidates.extend((tile, True) for tile in underfilled if tile not in narrow)
    return candidates, default_grid, automatic_subtiles, arch35_macro_streamk, underfilled


def _evaluate_select_candidate(
    mapper, host_ctx, *, tile, m, n, k, groups, compute_format, rows_key, arch35_macro_streamk,
):
    tm, tn, tk = tile
    cm = min(mapper.compute_m, tm)
    cn = min(mapper.compute_n, tn)
    output_tiles = (
        sum(math.ceil(r / tm) for r in rows_key) * math.ceil(n / tn)
        if rows_key is not None
        else math.ceil(m / tm) * math.ceil(n / tn) * groups
    )
    split_k = _candidate_split_k(
        host_ctx, m=m, n=n, k=k, tk=tk, output_tiles=output_tiles,
        groups=groups, compute_format=compute_format,
    )
    recipe = HierarchicalMatmul(
        tm, tn, tk, cm, cn,
        mapper.slots, mapper.compute_slots, mapper.reuse_inputs,
        mapper.reuse_weights, split_k)
    if min(tm, tn, tk, mapper.compute_m, mapper.compute_n,
           mapper.slots, mapper.compute_slots) < 1:
        raise ModelingError("matrix_tile_dimensions_and_slots_must_be_positive")
    if tm % cm or tn % cn:
        raise ModelingError("matrix_refill_tiles_must_be_multiples_of_compute_tiles")
    geometry = matrix_geometry(host_ctx, recipe, group_rows=rows_key)
    buffers = footprint(host_ctx, recipe, geometry=geometry)
    exceeded = {pool: size for pool, size in buffers.items()
                if pool in host_ctx.pool_capacities and size > host_ctx.pool_capacities[pool]}
    # Host tiling runs from static (M, N, K) and never observes transient L2 warmth.
    if exceeded:
        bound = None
    elif arch35_macro_streamk and split_k > 1:
        base_perf = _arch35_streamk_perf(
            host_ctx, recipe, geometry, split_k=1, output_tiles=output_tiles)
        sk_perf = _arch35_streamk_perf(
            host_ctx, recipe, geometry, split_k=split_k, output_tiles=output_tiles)
        if sk_perf < base_perf:
            bound = sk_perf
        else:
            split_k = 1
            recipe = HierarchicalMatmul(
                tm, tn, tk, cm, cn,
                mapper.slots, mapper.compute_slots, mapper.reuse_inputs,
                mapper.reuse_weights, split_k)
            bound = base_perf
    elif arch35_macro_streamk:
        bound = _arch35_streamk_perf(
            host_ctx, recipe, geometry, split_k=1, output_tiles=output_tiles)
    else:
        bound = mapping_bound(
            host_ctx, recipe, geometry=geometry,
            allow_unequal_groups=rows_key is not None,
            ignore_l2_hits=True)
        if split_k > 1:
            split_bound = _split_k_bound(
                host_ctx, recipe, geometry, bound=bound,
                split_k=split_k, output_tiles=output_tiles)
            if split_bound >= bound:
                split_k = 1
                recipe = HierarchicalMatmul(
                    tm, tn, tk, cm, cn,
                    mapper.slots, mapper.compute_slots, mapper.reuse_inputs,
                    mapper.reuse_weights, split_k)
            else:
                bound = split_bound
    waves = math.ceil(output_tiles / host_ctx.core_count)
    rec = {
        "tile": [tm, tn, tk], "split_k": split_k,
        "footprint_bytes_per_owner": buffers,
        "exceeded_pools": exceeded, "resource_bound_ms": bound,
        "maximum_owner_output_tiles": waves,
        "logical_shape": list(geometry.logical_shape),
        "local_shape": list(geometry.local_shape),
        "storage_shape": list(geometry.storage_shape),
        "issued_shape": list(geometry.issued_shape),
        "useful_flops": geometry.useful_flops,
        "macro_tile_area": geometry.macro_tile_area,
        "macro_tile_envelope": geometry.macro_tile_envelope,
        "output_tiles": geometry.tiles,
    }
    score = (
        (bound, waves, -split_k, -tk, sum(buffers.values()), tm, tn)
        if bound is not None else None
    )
    return recipe, geometry, rec, score


@dataclass(frozen=True)
class MappedMatmul:
    candidates: tuple | None = None
    compute_m: int = DEFAULT_COMPUTE_TILE_MN
    compute_n: int = DEFAULT_COMPUTE_TILE_MN
    slots: int = 2
    compute_slots: int = 2
    reuse_inputs: bool = True
    reuse_weights: bool = True
    narrow_candidates: tuple | None = None
    underfilled_candidates: tuple | None = None

    def select(self, context, *, include_record: bool = True, group_rows=None):
        """Select the declared tile mapping and its explanatory record.

        ``include_record=False`` skips the defensive record copy on the hot
        steady path (a caller that has no mapping observer).  Selection itself
        is identical: the same cached recipe object is returned either way.
        """
        host_ctx = (
            replace(context, primitive_table=None)
            if getattr(context, "primitive_table", None) is not None
            else context
        )
        rows_key = tuple(int(v) for v in group_rows) if group_rows is not None else None
        cache_key = (
            _matmul_recipe_key(self), _op_matrix_key(host_ctx.op), host_ctx.core_count,
            tuple(sorted(host_ctx.resource_names.items())),
            tuple(sorted(host_ctx.rates.items())),
            tuple(sorted(host_ctx.work.items())),
            tuple(sorted(host_ctx.resource_units.items())),
            tuple(sorted(host_ctx.pool_capacities.items())),
            getattr(host_ctx.physical_mechanisms, "signature", None),
            rows_key,
        )
        if cache_key in _SELECT_CACHE:
            recipe, record = _SELECT_CACHE[cache_key]
            return recipe, (deepcopy(record) if include_record else None)
        records = []
        selected = None
        selected_geometry = None
        meta = dict(host_ctx.op.metadata)
        groups = int(meta.get("matrix_groups", 1))
        if rows_key is None:
            if groups < 1 or host_ctx.op.local_shape[(-3)] % groups:
                raise ModelingError("matrix_groups_require_equal_integer_rows")
            m = host_ctx.op.local_shape[(-3)] // groups
        else:
            groups = len(rows_key)
            if groups < 1:
                raise ModelingError("matrix_groups_require_positive_integer")
            m = max(rows_key, default=0)
        compute_format = meta.get("compute_format", host_ctx.op.dtype or "bf16")
        n = host_ctx.op.local_shape[(-2)]
        k = host_ctx.op.local_shape[(-1)]
        candidates, default_grid, automatic_subtiles, arch35_macro_streamk, underfilled = (
            _resolve_select_candidates(
                self, host_ctx, m=m, n=n, k=k, groups=groups,
                compute_format=compute_format, rows_key=rows_key)
        )
        for tile, _is_narrow in candidates:
            recipe, geometry, rec, score = _evaluate_select_candidate(
                self, host_ctx, tile=tile, m=m, n=n, k=k, groups=groups,
                compute_format=compute_format, rows_key=rows_key,
                arch35_macro_streamk=arch35_macro_streamk)
            records.append(rec)
            if score is not None and (selected is None or score < selected[0]):
                selected = (score, recipe)
                selected_geometry = geometry
        if selected is None:
            raise ModelingError(f"matrix_no_feasible_declared_mapping:"
                                f"shape={context.op.local_shape}:groups={groups}:dtype={context.op.dtype}")
        res = selected[1], {
            "basis": "declared_candidates_resource_lower_bound",
            "shape": list(context.op.local_shape[(-3):]),
            "groups": groups,
            "default_128_grid_tiles": default_grid,
            "physical_compute_owners": context.core_count,
            "subtile_policy": ("all_dtype_underfilled_or_explicit"
                               if automatic_subtiles else "explicit_only"),
            "underfilled_grid_candidates": bool(underfilled),
            "geometry": selected_geometry.to_dict(),
            "selected": asdict(selected[1]), "candidates": records,
            "compute_precision_peaks": dict(context.compute_peaks),
            "capacity_screening": "declared_pools_only",
            "declared_pools": dict(context.pool_capacities),
            "read_role_bindings": {
                "matrix_operands": read_role(context.resource_names),
                "output_scales": read_role(context.resource_names, aiv=0),
            },
            "unmodeled_transfer_roles": [role for role in (
                                        "mte1", read_role(context.resource_names),
                                        "mte3")
                                        if role not in context.resource_names],
            "representation_policy": "packed_payload_and_scales_with_compute_width_bound",
            "conversion_latency": "not_inferred_from_transfer_bytes",
        }
        if len(_SELECT_CACHE) >= 1024:
            _SELECT_CACHE.pop(next(iter(_SELECT_CACHE)))
        _SELECT_CACHE[cache_key] = res
        return res[0], (deepcopy(res[1]) if include_record else None)

    def lower(self, context):
        recipe, record = self.select(context)
        if context.record_mapping is not None:
            context.record_mapping(record)
        return recipe.lower(context)


@dataclass(frozen=True)
class MatrixOperatorResult:
    """Public matrix-operator occurrence result.

    This is a comparison/contract result, not a serving latency or a measured
    answer.  Duration, serviced work and completed task count are occurrence
    deltas.  A current-occurrence buffer peak is only observable on a fresh
    Engine; on a shared Engine it is ``None`` with an explicit ``peak_basis``
    instead of reporting whole-engine history as this occurrence's peak.
    """

    duration_ms: float
    geometry: dict
    mapping: dict
    declared_work: dict
    serviced: dict
    peak_allocated: dict | None
    completed_tasks: int
    release_ms: float = 0.0
    ready_ms: float = 0.0
    peak_basis: str = "occurrence_on_fresh_engine"
    historical_peak_allocated: dict = field(default_factory=dict)

    def to_dict(self):
        return {
            "duration_ms": self.duration_ms,
            "geometry": self.geometry,
            "mapping": self.mapping,
            "declared_work": self.declared_work,
            "serviced": self.serviced,
            "peak_allocated": self.peak_allocated,
            "completed_tasks": self.completed_tasks,
            "release_ms": self.release_ms,
            "ready_ms": self.ready_ms,
            "peak_basis": self.peak_basis,
            "historical_peak_allocated": self.historical_peak_allocated,
        }


def _default_engine(context):
    """Build the ordinary Engine from the declared context, never a device."""
    from mindie_sim.prediction.engine import Engine, Resource
    names = set(context.resource_names.values())
    resources = []
    for resource, rate in context.rates.items():
        if resource not in names:
            continue
        units = (1 if resource == context.resource_names.get("hbm")
                 else physical_owner_units(context, resource))
        resources.append(Resource(resource, float(rate), units=units))
    # A prepared ``for_phase``/stage context binds cube FLOPs to peak_ms work
    # but may leave the cube rate in the enclosing execution.  Add that one
    # declared physical resource rather than inventing a compute peak here.
    cube = context.resource_names.get("cube", "cube")
    if context.work.get(cube) and cube not in {item.name for item in resources}:
        resources.append(Resource(cube, 1.0, unit="peak_ms",
                                  units=physical_owner_units(context, cube)))
    if not resources:
        raise ModelingError("matrix_operator_requires_declared_resource_rates")
    pools = {}
    pool_units = {}
    for logical, physical in context.pool_names.items():
        size = context.pool_capacities.get(logical, context.pool_capacities.get(physical))
        if size is None:
            raise ModelingError(f"matrix_operator_requires_pool_capacity:{logical}")
        pools[physical] = float(size)
        pool_units[physical] = int(
            context.pool_units.get(logical, context.pool_units.get(
                physical, context.core_count)))
    return Engine(resources, pools=pools, pool_units=pool_units)


def simulate_matrix_operator(context, *, recipe=None, engine=None, release_ms=0.0):
    """Run one isolated matrix operator through the common causal Engine.

    The entry accepts an already typed ``RecipeContext`` (semantics, tensor
    work, hardware rates/units, buffers and service identity), selects the
    declared mapping if needed, then returns finite work, geometry, buffers and
    completion.  It does not read model labels, observations, serving state or
    a target duration.  Delayed input readiness is an explicit release time and
    is excluded from the returned operator duration just like other public
    isolated-operator checks.

    A caller-provided Engine may be reused serially after previous work has
    completed.  Before submission the entry requires an idle active frontier:
    live/queued tasks, live programs or live allocations are rejected with
    ``matrix_operator_requires_idle_engine_frontier`` before any Engine
    mutation.  This keeps the occurrence serviced/completed delta from
    absorbing other ready work and rejects not-yet-started queued tasks rather
    than misclassifying them as fresh history.  Duration is measured from the
    operator's actual ready event.  Buffer peak is occurrence-only on a fresh
    Engine.  After serial reuse the historical peak would be wrong, so it is
    returned as ``None`` with
    ``peak_basis="unavailable_shared_engine_slice_boundary"``.  Concurrent
    combinations remain available through ``recipe.lower(context)`` plus
    ``program.start(engine, ...)``; this isolated entry does not implement a
    whole-Engine attribution subsystem.
    """
    from mindie_sim.prediction.engine import ExecutionError
    from .matrix_steady import SteadyMatmul
    recipe = recipe or SteadyMatmul()
    records = []
    previous = context.record_mapping

    def record_mapping(value):
        records.append(value)
        if previous is not None:
            previous(value)

    ctx = replace(context, record_mapping=record_mapping)
    engine = engine or _default_engine(ctx)
    frontier = engine.summary()
    if (frontier["live_tasks"] or frontier["live_programs"]
            or any(frontier["live_allocated"].values())):
        raise ExecutionError("matrix_operator_requires_idle_engine_frontier")
    name = f"matrix{next(_OPERATOR_SEQUENCE)}"
    before_serviced = dict(engine.serviced)
    before_allocated = dict(engine.allocated)
    before_completed = engine.completed_count
    before_now = engine.now
    fresh = (before_now == 0 and before_completed == 0
             and not any(before_serviced.values())
             and not any(before_allocated.values()))
    program = recipe.lower(ctx)
    ready = engine.submit(name + ":ready", release=release_ms)
    done = program.start(engine, name, ready)
    engine.advance(until=(done,))
    if not done.done:
        raise ExecutionError("matrix_operator_incomplete")
    if any(key.startswith(name + ":") for key in engine.allocations):
        raise ExecutionError("matrix_operator_left_live_buffers")
    if fresh and (engine.summary()["live_tasks"]
                  or engine.summary()["live_programs"]):
        raise ExecutionError("matrix_operator_left_live_tasks")
    mapping = records[-1] if records else {}
    geometry = mapping.get("geometry") if isinstance(mapping, dict) else None
    if not geometry:
        geometry = matrix_geometry(ctx, recipe).to_dict()
        mapping = {"basis": "explicit_matrix_recipe", "geometry": geometry,
                   "compute_precision_peaks": dict(ctx.compute_peaks)}
    ready_time = ready.time if ready.time is not None else float(release_ms)
    serviced = {
        resource: value - before_serviced.get(resource, 0.0)
        for resource, value in engine.serviced.items()
        if abs(value - before_serviced.get(resource, 0.0)) > 1e-12
    }
    historical_peak = {pool: value for pool, value in engine.peak_allocated.items()
                       if value}
    return MatrixOperatorResult(
        duration_ms=done.time - ready_time,
        geometry=geometry,
        mapping=mapping,
        declared_work=dict(ctx.work),
        serviced=serviced,
        peak_allocated=(dict(historical_peak) if fresh else None),
        completed_tasks=engine.completed_count - before_completed,
        release_ms=float(release_ms),
        ready_ms=ready_time,
        peak_basis=("occurrence_on_fresh_engine" if fresh else
                    "unavailable_shared_engine_slice_boundary"),
        historical_peak_allocated=historical_peak,
    )
