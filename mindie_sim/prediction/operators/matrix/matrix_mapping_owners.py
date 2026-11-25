"""Matrix tile ownership and per-owner resource supply.

[Role] Derive matrix owner rectangles, tail totals and pinned pipe supply.
[Invariants] Owner aggregates retain the exact ragged output extents. A pinned
  stage draws only its assigned fraction of the declared physical pipe rate.
[Timeouts & Locks] Closed-form owner arithmetic and bounded explicit tile
  enumeration; no locks.
[Side-Effect Policy] Pure calculations over caller-supplied shapes and rates.
"""
from collections.abc import Mapping
from fractions import Fraction
from functools import lru_cache
import math
from typing import Literal

from mindie_sim.prediction.runtime.memory_physics import AddressRun
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
_OPS_TRANSFORMER_REV = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "gmm/grouped_matmul/op_host/op_tiling/arch35/grouped_no_quant_matmul_tiling.cpp",
        "f1c1a40e70a11e475467c923698af3297fc55b90588ab8a0bd7ee1b4110d3c5b",
    ),
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "gmm/grouped_matmul/op_kernel/grouped_matmul_kernel.h",
        "b5d4e80ce395887f5d95db83e5f36b6f13edf170208c9a7cf0f927858b0c1025",
    ),
    SourceRef(
        "cann/ops-transformer", _OPS_TRANSFORMER_REV,
        "mc2/3rd/mat_mul_v3/op_host/op_tiling/arch35/matmul_v3_basic_streamk_tiling.cpp",
        "1a1ba337c283965aa9495220b1df8b868c082591768c4b82e6c1d251b2def497",
    ),
)

_FRACTAL_ROWS = 16
#: L2 cache sector widths for DAV_2201 (512 B) and DAV_3510 (128 B).
DAV2201_L2_SECTOR_BYTES = 512
DAV3510_L2_SECTOR_BYTES = 128
#: P05 L2 bank-stride conflict knee (16 KiB) above which non-pow2 strides alias by gcd(stride, 2^k).
L2_BANK_STRIDE_KNEE_BYTES = 16 * 1024
#: DAV_2201 on-the-fly ND->FRACTAL_NZ B transfer runs at 0.5x L0B (64 B/cycle).
DAV2201_ND2NZ_B_L0B_RATIO = 0.5
DAV2201_ND2NZ_B_BYTES_PER_CYCLE = 64.0
#: Ratio threshold separating asymmetric DAV_2201 L0B (0.50x) from symmetric DAV_3510 (1.0x).
DAV2201_ASYMMETRIC_L0B_RATIO_MAX = 0.75
#: Hardware L0A bus width (256 B/cycle).
LOAD_DATA_L0A_BYTES_PER_CYCLE = 256.0
#: Multi-wave grouped HBM cold ramp-up before reaching streaming HBM saturation.
GROUPED_HBM_RAMP_WAVES = 2.0
MS_PER_SECOND = 1000.0
NS_PER_MS = 1_000_000.0
#: Upstream arch35/matmul_v3_basic_streamk_tiling.cpp:124 GetNegPerf cycle model.
STREAMK_NEG_CYCLES_PER_SPLIT = 60.0
STREAMK_NEG_BASE_CYCLES = 1000.0
#: Dual-side AIC + AIV RTS/STARS sub-block launches for MIX_AIC Stream-K kernels.
MIX_AIC_SUBBLOCK_DISPATCHES = 2.0
#: Largest standard MatMul baseM tile in DEFAULT_CANDIDATES (for single-M-tile L2 checks).
MAX_SINGLE_M_TILE = 256
_GEMM_SHAPE_AXES = 3
#: Blocking scalar GM load latency per group in grouped_matmul_kernel.h:314
#: (groupListGm.GetValue(groupIdx) across declared groupNum).
GROUP_LIST_SCALAR_LOAD_CYCLES = 200.0
#: Distinct (row, stride, rows, sector) panel geometries memoized per process.
PANEL_SECTOR_CACHE_ENTRIES = 4096


def mte_owner_unit_count(resource_units):
    """Hardware unit count that pins one matrix output owner.

    Native ``owned(core, core_count, read_role)`` partitions ``mte2_aic``.
    A supply that never declares those units has no per-owner split.
    """
    units = dict(resource_units or ())
    for name in ("mte2_aic", "mte2", "compute_units"):
        value = int(units.get(name, 0) or 0)
        if value > 0:
            return value
    return 0


def gemm_fractal_operand_bytes(tokens, n_tiles, n_pad, k_pad, act_bytes, weight_bytes,
                               *, fractal=_FRACTAL_ROWS):
    """Summed operand bytes of the piecewise GEMM copy, before partials.

    Activation is copied once per N tile at the 16-row fractal height.
    Weights move once at the padded N.  Split-K cancels: each split carries
    ``K / splits``.  This is the family's summed copy, not the slowest owner.
    """
    tokens = int(tokens)
    if tokens <= 0 or int(n_tiles) <= 0 or int(k_pad) <= 0:
        return Fraction(0)
    rows = (tokens + int(fractal) - 1) // int(fractal) * int(fractal)
    act = Fraction(act_bytes)
    weight = Fraction(weight_bytes)
    return (Fraction(int(n_tiles)) * rows * Fraction(int(k_pad)) * act
            + Fraction(int(k_pad)) * Fraction(int(n_pad)) * weight)


def critical_output_owner_rate_fraction(
    m, n, k, tile_m, tile_n, units, *,
    act_bytes, weight_bytes, charged_bytes=None, groups=1,
):
    """Share of an aggregate pipe that elapses as the slowest output owner.

    ``SteadyMatmul`` pins each output tile with ``owned(core, core_count, ...)``.
    Idle units stay idle.  One tile on a 20-unit pipe therefore runs at
    ``1/20`` of the aggregate, and the makespan is the maximum private copy

        act_bytes * m_extent * k + weight_bytes * n_extent * k

    using the exact local tail from ``owner_tiles`` (a 288-wide N on a
    128 tile is 128, 128, 32, not the average 96).  Piecewise stage work
    sums every owner and, for the GEMM family, the fractal row pad.  Passing
    that summed ``charged_bytes`` makes

        charged / (aggregate * fraction) = max_copy / (aggregate / units)

    without rewriting the charged bytes and without reading a duration.
    HBM is unpinned and is not this share.  A missing unit count returns 1.
    """
    units = int(units)
    if units <= 0 or int(m) <= 0 or int(n) <= 0 or int(k) <= 0:
        return Fraction(1)
    owners = owner_tiles(
        int(m), int(n), max(1, int(tile_m)), max(1, int(tile_n)),
        units, int(groups))
    if not owners:
        return Fraction(1)
    act = Fraction(act_bytes)
    weight = Fraction(weight_bytes)
    kk = Fraction(int(k))
    copies = [
        act * Fraction(extent_m) * kk + weight * Fraction(extent_n) * kk
        for _count, extent_m, extent_n, _area in owners
    ]
    peak = max(copies)
    if peak <= 0:
        return Fraction(1)
    # Each output owner copies its own private A and B panels on its pinned
    # MTE2 pipe; idle units stay idle, so the makespan scales with the peak
    # private owner copy ``act * extent_m * k + weight * extent_n * k``.
    basis = Fraction(charged_bytes) if charged_bytes is not None else sum(copies, Fraction(0))
    if basis <= 0:
        return Fraction(1)
    return min(Fraction(1), basis / (peak * Fraction(units)))


def owner_tiles(m, n, tm, tn, cores, groups=1):
    """Return count, summed M extents, N extents and output areas per owner."""
    nm, nn = math.ceil(m / tm), math.ceil(n / tn)
    if not nm:
        return ()
    dm, dn = nm * tm - m, nn * tn - n
    last_m_start = (nm - 1) * nn
    last_n_counts = [0] * cores
    period = cores // math.gcd(nn, cores)
    for row in range(min(nm, period)):
        last_n_counts[(row * nn + nn - 1) % cores] += (nm - 1 - row) // period + 1
    result = []
    for core in range(min(cores, nm * nn)):
        count = (nm * nn - 1 - core) // cores + 1
        before = max(0, (last_m_start - 1 - core) // cores + 1)
        last_m = count - before
        last_n = last_n_counts[core]
        corner = int((nm * nn - 1) % cores == core)
        result.append((count, count * tm - last_m * dm, count * tn - last_n * dn,
                       count * tm * tn - last_m * dm * tn - last_n * dn * tm
                       + corner * dm * dn))
    if groups == 1:
        return tuple(result)
    totals = [[0, 0, 0, 0] for _ in range(cores)]
    period = cores // math.gcd(nm * nn, cores)
    for group in range(min(groups, period)):
        repeats = (groups - 1 - group) // period + 1
        for core, values in enumerate(result):
            target = totals[(core + group * nm * nn) % cores]
            for axis, value in enumerate(values):
                target[axis] += repeats * value
    return tuple(tuple(row) for row in totals if row[0])


def freeze_owner_tiles(tile_sources):
    """Materialize a finite owner-tile declaration exactly once."""
    result = []
    for count, shapes in tile_sources:
        shapes = tuple((int(a), int(b)) for a, b in shapes)
        result.append((int(count), shapes))
    return tuple(result)


def owner_records(tile_sources):
    """Aggregate finite explicit owner rectangles without losing tails.

    ``tile_sources`` is the same shape accepted by ``HierarchicalMatmul``:
    ``((count, ((m, n), ...)), ...)`` for active owners.  The returned record
    is the common ``(count, summed_m, summed_n, output_area)`` tuple used by
    ``owner_tiles``.  This entry lets a grouped/source adapter supply unequal
    group rows without a second matrix geometry implementation.
    """
    result = []
    for count, shapes in tile_sources:
        shapes = tuple((int(a), int(b)) for a, b in shapes)
        if (isinstance(count, bool) or not isinstance(count, int) or count < 0
                or len(shapes) != count):
            raise ModelingError("matrix_explicit_owner_tiles_require_matching_count")
        if any(a < 0 or b < 0 for a, b in shapes):
            raise ModelingError("matrix_explicit_owner_tiles_require_nonnegative_extents")
        result.append((count, sum(a for a, _ in shapes),
                       sum(b for _, b in shapes),
                       sum(a * b for a, b in shapes)))
    return tuple(row for row in result if row[0])


def grouped_owner_rectangles(group_rows, n, tm, tn, cores, *, max_tiles=1_000_000):
    """Finite group-major owner rectangles for explicit group rows.

    Each group contributes its own tail rows; rotating owners never replaces a
    group's row extent by the widest group.  Empty groups consume no tile.  The
    bound prevents a large declaration from becoming an unbounded enumeration.
    This is the executable entry consumed by both the detailed and steady
    matrix recipes; ``grouped_owner_tiles`` is its aggregate projection.
    """
    rows = tuple(int(value) for value in group_rows)
    n, tm, tn, cores = int(n), int(tm), int(tn), int(cores)
    if cores < 1 or tm < 1 or tn < 1 or n < 0 or any(row < 0 for row in rows):
        raise ModelingError("matrix_grouped_owner_geometry_invalid")
    tiles = sum(math.ceil(row / tm) for row in rows) * math.ceil(n / tn)
    if tiles > int(max_tiles):
        raise ModelingError(f"matrix_grouped_owner_enumeration_exceeds_bound:{tiles}")
    owners = [[] for _ in range(cores)]
    index = 0
    for row in rows:
        for start in range(0, row, tm):
            a = min(tm, row - start)
            for column in range(0, n, tn):
                b = min(tn, n - column)
                owners[index % cores].append((a, b))
                index += 1
    return tuple((len(shapes), tuple(shapes)) for shapes in owners if shapes)


def grouped_owner_tiles(group_rows, n, tm, tn, cores, *, max_tiles=1_000_000):
    """Owner/tail aggregate for explicit, possibly unequal group rows."""
    return owner_records(grouped_owner_rectangles(
        group_rows, n, tm, tn, cores, max_tiles=max_tiles))


def physical_owner_units(context, resource):
    """Declared physical units reached by one Engine resource name."""
    values = {int(context.resource_units.get(role, context.core_count))
              for role, name in context.resource_names.items() if name == resource}
    if not values:
        return int(context.core_count)
    if len(values) != 1:
        raise ModelingError(f"matrix_resource_owner_units_conflict:{resource}")
    return values.pop()


def _declared_rate(context, resource):
    """Engine rate for one declared pipe; cube is canonical peak_ms work."""
    if resource in context.rates:
        return float(context.rates[resource])
    if resource == context.resource_names.get("cube"):
        return 1.0
    raise ModelingError(f"matrix_resource_rate_missing:{resource}")


def physical_owner_rate(context, resource, placement=None):
    """Isolated service of one owner's fixed physical unit subset.

    The declared Engine rate is an aggregate over all units.  A stage pinned to
    ``placement`` receives only its fraction of that aggregate; an unpinned
    stage (for example shared HBM) receives the full declared rate.  This is a
    capacity projection, not a calibration factor.
    """
    rate = _declared_rate(context, resource)
    if not placement:
        return rate
    units = physical_owner_units(context, resource)
    if units < 1 or len(placement) > units:
        raise ModelingError(f"matrix_placement_exceeds_resource_units:{resource}")
    return rate * len(placement) / units


def is_arch35_context(context) -> bool:
    """True iff the execution context explicitly targets DAV_3510 via ComputeArch / PrimitiveTable."""
    compute_arch = getattr(context, "compute_arch", None)
    if compute_arch is None:
        datapath = getattr(context, "datapath", None)
        compute_arch = getattr(datapath, "compute_arch", None)
    if compute_arch is not None:
        arch_id = str(getattr(compute_arch, "arch_id", "") or "").strip().lower()
        preset = str(getattr(compute_arch, "preset", "") or "").strip().lower()
        if arch_id or preset:
            return arch_id in {"arch35", "dav_3510", "npu_arch_3510"} or preset == "npu_arch_3510"
    pt = getattr(context, "primitive_table", None)
    if pt is not None:
        arch = str(getattr(pt, "arch_preset", "") or "").strip().lower()
        if arch:
            return arch == "npu_arch_3510"
    return False


def _table_cell(context, primitive_id: str):
    """Return the declared primitive cell when a PrimitiveTable is bound (fail-closed)."""
    pt = getattr(context, "primitive_table", None)
    cell_fn = getattr(pt, "cell", None)
    if cell_fn is None:
        return None
    return cell_fn(primitive_id)


def primitive_core_clock_hz(context) -> float | None:
    """Read AIC core frequency in Hz strictly from declared ComputeArch/Datapath."""
    for owner in (
        getattr(context, "compute_arch", None),
        getattr(context, "datapath", None),
        getattr(getattr(context, "datapath", None), "compute_arch", None),
    ):
        val = getattr(owner, "clock_hz", None)
        if val is None:
            continue
        if isinstance(val, bool) or float(val) <= 0.0:
            raise ModelingError(f"compute_arch_clock_hz_invalid:{val!r}")
        return float(val)
    return None


def primitive_p01_scaling_bps(context, active_cores: int, *, warm: bool) -> float | None:
    """Interpolate P01_datacopy AIC cold/warm scaling curve at ``active_cores``."""
    if active_cores < 1:
        return None
    cell = _table_cell(context, "P01_datacopy")
    if cell is None:
        return None
    params = cell.parameters or {}
    key = "mte2_aic_warm_scaling_bps" if warm else "mte2_aic_cold_scaling_bps"
    curve = params.get(key)
    if not isinstance(curve, Mapping) or not curve:
        return None
    pts = sorted((int(k), float(v)) for k, v in curve.items() if float(v) > 0.0)
    if not pts:
        return None
    c = int(active_cores)
    if c <= pts[0][0]:
        return pts[0][1] * (float(c) / float(pts[0][0]))
    if c >= pts[-1][0]:
        return pts[-1][1]
    for (c0, v0), (c1, v1) in zip(pts, pts[1:]):
        if c0 <= c <= c1:
            return v0 + (float(c - c0) / float(c1 - c0)) * (v1 - v0)
    return pts[-1][1]


def primitive_p01_strided_cold_bps(
    context, row_bytes: int, stride_bytes: int,
) -> float | None:
    """Interpolate calibrated P01 2D strided cold-HBM bandwidth surface."""
    if row_bytes < 1:
        return None
    cell = _table_cell(context, "P01_datacopy")
    if cell is None:
        return None
    params = cell.parameters or {}
    surface = params.get("strided_cold_surface")
    if not surface:
        return None
    rows = sorted({int(entry["row_bytes"]) for entry in surface})
    if not rows:
        return None
    r_sel = max((r for r in rows if r <= int(row_bytes)), default=rows[0])
    pts = sorted(
        (int(entry["stride_bytes"]), float(entry["bps_p50"]))
        for entry in surface
        if int(entry["row_bytes"]) == r_sel and float(entry.get("bps_p50", 0.0)) > 0.0
    )
    if not pts:
        return None
    stride = int(stride_bytes)
    if stride <= pts[0][0]:
        return pts[0][1]
    if stride >= pts[-1][0]:
        return pts[-1][1]
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if s0 <= stride <= s1:
            return v0 + (float(stride - s0) / float(s1 - s0)) * (v1 - v0)
    return pts[-1][1]


def primitive_p05_stride_conflict_bps(context, stride_bytes: int) -> float | None:
    """Interpolate calibrated P05 L2/TLB stride-conflict ceiling on log2 stride."""
    if stride_bytes < 1:
        return None
    cell = _table_cell(context, "P05_l2_residency_reuse")
    if cell is None:
        return None
    params = cell.parameters or {}
    curve = params.get("l2_stride_conflict")
    if not curve:
        return None
    stride = int(stride_bytes)
    aligned = (stride % DAV2201_L2_SECTOR_BYTES) == 0
    if aligned and stride > L2_BANK_STRIDE_KNEE_BYTES and (stride & (stride - 1)) != 0:
        # Multiplying a cache-set stride by an odd factor is a bijection modulo 2^k L2 sets;
        # only its power-of-2 factor gcd(stride, 2^k) = stride & -stride aliases L2 sets.
        stride = stride & -stride
    kinds = ("contiguous", "pow2") if aligned else ("contiguous", "coprime")
    pts = sorted(
        (int(entry["stride_bytes"]), float(entry["bps_p50"]))
        for entry in curve
        if str(entry.get("kind", "")) in kinds and float(entry.get("bps_p50", 0.0)) > 0.0
    )
    if not pts:
        return None
    if stride <= pts[0][0]:
        return pts[0][1]
    if stride >= pts[-1][0]:
        return pts[-1][1]
    log_s = math.log2(float(stride))
    for (s0, v0), (s1, v1) in zip(pts, pts[1:]):
        if s0 <= stride <= s1:
            l0, l1 = math.log2(float(s0)), math.log2(float(s1))
            frac = (log_s - l0) / (l1 - l0) if l1 > l0 else 0.0
            return v0 + frac * (v1 - v0)
    return pts[-1][1]


def primitive_p01_hbm_read_bps(context) -> float | None:
    """Calibrated full-die cold-HBM streaming read rate in bytes/s."""
    pt = getattr(context, "primitive_table", None)
    if pt is None:
        return None
    from mindie_sim.primitives.loader import _table_band_p50
    val = _table_band_p50(pt, "P01_datacopy", "hbm_read_bytes_per_s")
    return float(val) if val is not None and float(val) > 0.0 else None


def primitive_p02_l0_bps_per_aic(context) -> tuple[float, float] | None:
    """Calibrated P02 L1->L0A and L1->L0B rates in bytes/s/AIC."""
    pt = getattr(context, "primitive_table", None)
    if pt is None:
        return None
    from mindie_sim.primitives.loader import _table_band_p50
    l0a = _table_band_p50(pt, "P02_load_data", "l1_to_l0a_bytes_per_s_per_aic")
    l0b = _table_band_p50(pt, "P02_load_data", "l1_to_l0b_bytes_per_s_per_aic")
    if l0a is None or l0b is None or float(l0a) <= 0.0 or float(l0b) <= 0.0:
        return None
    return float(l0a), float(l0b)


def primitive_l2_params(context) -> tuple[int, int]:
    """Return ``(l2_bytes, l2_sector_bytes)`` from P05 or device pools."""
    pt = getattr(context, "primitive_table", None)
    l2_bytes = int((getattr(context, "pool_capacities", None) or {}).get("l2", 0))
    sector = DAV3510_L2_SECTOR_BYTES if is_arch35_context(context) else DAV2201_L2_SECTOR_BYTES
    if pt is not None:
        fp = getattr(getattr(pt, "provenance", None), "silicon_fingerprint", None)
        if fp is not None and int(getattr(fp, "l2_bytes", 0) or 0) > 0:
            l2_bytes = int(fp.l2_bytes)
        p05_cell = _table_cell(context, "P05_l2_residency_reuse")
        p05 = (p05_cell.parameters or {}) if p05_cell is not None else {}
        if int(p05.get("l2_sector_bytes", 0) or 0) > 0:
            sector = int(p05["l2_sector_bytes"])
    return l2_bytes, sector


def _primitive_band_ms(context, primitive_id: str, metric: str) -> float:
    """Calibrated primitive latency band in ms (0.0 when explicitly unmeasured)."""
    pt = getattr(context, "primitive_table", None)
    if pt is None:
        raise ModelingError(f"matrix_primitive_table_required:{primitive_id}:{metric}")
    cell_fn = getattr(pt, "cell", None)
    if cell_fn is None:
        return 0.0
    cell = cell_fn(primitive_id)
    band = cell.band(metric)
    if band.p50 is None:
        return 0.0
    if float(band.p50) <= 0.0:
        raise ModelingError(f"matrix_primitive_band_invalid:{primitive_id}:{metric}")
    return float(band.p50) / NS_PER_MS


def primitive_p13_sync_ms(context, metric: str) -> float:
    """Calibrated P13 cross-core sync latency in ms (0.0 when absent)."""
    return _primitive_band_ms(context, "P13_crosscore_sync", metric)


def primitive_nd2nz_mte2_rate_bps_per_aic(
    context, active_cores: int, left_bytes: float, right_bytes: float, *,
    weight_layout: str = "ND", row_bytes: int = 0, stride_bytes: int = 0,
) -> float | None:
    """Effective per-AIC MTE2 L2->L1 rate from two-term private/L2 roofline.

    Per-AIC private time blends contiguous A at single-core warm MTE2 rate
    with ND2NZ B (64 B/cycle = 0.5 * L0B on asymmetric DAV_2201, with 1-cycle
    per-row turnaround when strided row_bytes < 512 B L2 sector; full rate on
    symmetric DAV_3510 or FRACTAL_NZ). Aggregate die L2 bandwidth
    ``p01_warm(active_cores)`` bounds the combined active-core rate.
    """
    pt = getattr(context, "primitive_table", None)
    if pt is None:
        return None
    total = float(left_bytes) + float(right_bytes)
    if total <= 0.0:
        return None
    cores = max(1, int(active_cores))
    single_bps = primitive_p01_scaling_bps(context, 1, warm=True)
    warm_agg = primitive_p01_scaling_bps(context, cores, warm=True)
    if single_bps is None or single_bps <= 0.0:
        from mindie_sim.primitives.loader import _table_band_p50
        hit = _table_band_p50(pt, "P01_datacopy", "mte2_aic_hit_bytes_per_s_per_core")
        if hit is None or float(hit) <= 0.0:
            return None
        single_bps = float(hit)
    a_bps = single_bps
    b_bps = single_bps
    if not is_arch35_context(context) and str(weight_layout).upper() != "FRACTAL_NZ":
        l0 = primitive_p02_l0_bps_per_aic(context)
        if l0 is not None and (l0[1] / l0[0]) < DAV2201_ASYMMETRIC_L0B_RATIO_MAX:
            b_bps = min(single_bps, DAV2201_ND2NZ_B_L0B_RATIO * l0[1])
            _, sector = primitive_l2_params(context)
            if 0 < int(row_bytes) < int(sector) and int(row_bytes) < int(stride_bytes):
                # 1-cycle burst turnaround on the 64 B/cycle ND2NZ B bus for sub-sector rows.
                b_bps *= float(row_bytes) / (float(row_bytes) + DAV2201_ND2NZ_B_BYTES_PER_CYCLE)
    t_private = float(left_bytes) / a_bps + float(right_bytes) / b_bps
    t_l2_agg = (total * float(cores) / warm_agg) if (warm_agg is not None and warm_agg > 0.0) else 0.0
    t_eff = max(t_private, t_l2_agg)
    return total / t_eff if t_eff > 0.0 else single_bps


def weight_l2_cache_disabled(
    context, *, m_tiles: int | None, row_bytes: int, stride_bytes: int, is_grouped: bool,
) -> bool:
    """Upstream CANN operand-scoped weight B CACHE_MODE_DISABLE rule.

    - Dense MatMulV3 (arch35/matmul_v3_base_tiling_advanced.h:146-189):
      requires total GM >= L2 capacity, single M-tile (baseM >= M and mCnt <= 1),
      and 128B alignment on both inner stride and tile row.
    - A3/DAV_2201 GroupedMatmul (grouped_matmul_kernel.h:566):
      disables weight L2 cache whenever ``blockDimM == 1`` regardless of L2 size.
    - A5/DAV_3510 GroupedMatmul (arch35/grouped_no_quant_matmul_tiling.cpp:186-200):
      requires total GM >= L2 capacity AND (FRACTAL_NZ OR 128B-aligned ND
      stride and panel width).
    """
    op = getattr(context, "op", None)
    meta = dict(getattr(op, "metadata", ()) or ()) if op is not None else {}
    if "l2_weight_cache_disable" in meta:
        return bool(meta["l2_weight_cache_disable"])
    is_arch35 = is_arch35_context(context)
    if is_grouped and not is_arch35:
        if m_tiles is not None:
            return int(m_tiles) == 1
        if op is not None:
            shape = getattr(op, "local_shape", ()) or ()
            if len(shape) >= _GEMM_SHAPE_AXES:
                groups = max(1, int(meta.get("matrix_groups", 1) or 1))
                return (int(shape[-_GEMM_SHAPE_AXES]) // groups) <= MAX_SINGLE_M_TILE
        return False
    l2_bytes, _ = primitive_l2_params(context)
    if l2_bytes <= 0:
        return False
    op_hbm_bytes = float(getattr(op, "hbm_bytes", 0.0) or 0.0) if op is not None else 0.0
    if op_hbm_bytes < float(l2_bytes):
        return False
    weight_nz = str(meta.get("weight_layout", "ND")).strip().upper() == "FRACTAL_NZ"
    if is_grouped and is_arch35:
        if weight_nz:
            return True
        if int(stride_bytes) <= 0 or int(row_bytes) <= 0:
            return False
        return (
            (int(stride_bytes) % DAV3510_L2_SECTOR_BYTES) == 0
            and (int(row_bytes) % DAV3510_L2_SECTOR_BYTES) == 0
        )
    if m_tiles is not None:
        if int(m_tiles) > 1:
            return False
    elif op is not None:
        shape = getattr(op, "local_shape", ()) or ()
        if len(shape) >= _GEMM_SHAPE_AXES:
            groups = max(1, int(meta.get("matrix_groups", 1) or 1))
            if (int(shape[-_GEMM_SHAPE_AXES]) // groups) > MAX_SINGLE_M_TILE:
                return False
    if int(stride_bytes) <= 0 or int(row_bytes) <= 0:
        return False
    return (
        (int(stride_bytes) % DAV3510_L2_SECTOR_BYTES) == 0
        and (int(row_bytes) % DAV3510_L2_SECTOR_BYTES) == 0
    )


@lru_cache(maxsize=PANEL_SECTOR_CACHE_ENTRIES)
def panel_sector_efficiency(*, row_bytes: int, stride_bytes: int, rows: int,
                            sector_bytes: int) -> float:
    """Useful/fetched bytes of one ``rows``-row panel from its L2 sector union.

    Contiguous rows (``row_bytes >= stride_bytes``) coalesce into one run;
    narrow strided rows share a sector whenever ``stride_bytes < sector_bytes``
    (e.g. 32 B every 64 B gives 1/2, not 32/512).  Without a row count, a
    strided row is charged one full sector per row.
    """
    row = max(1, int(row_bytes))
    stride = max(row, int(stride_bytes))
    sector = max(1, int(sector_bytes))
    if rows <= 0:
        return 1.0 if stride == row else min(1.0, float(row) / float(sector))
    union = AddressRun(0, int(rows), row, stride).sector_union_bytes(sector)
    return min(1.0, float(int(rows) * row) / float(max(1, union)))


def primitive_cold_hbm_bps(
    context, *, active_cores: int, row_bytes: int, stride_bytes: int,
    total_tiles: int, is_grouped: bool, is_full_row_coalesced: bool = False,
    m_tiles: int | None = None, weight_cache_disable: bool | None = None,
    panel_rows: int | None = None,
) -> float | None:
    """Effective die cold-HBM read bandwidth from P01 scaling, strided surface and P05 stride cap."""
    pt = getattr(context, "primitive_table", None)
    if pt is None or active_cores < 1:
        return None
    hbm_rd = primitive_p01_hbm_read_bps(context)
    p01_cold = primitive_p01_scaling_bps(context, active_cores, warm=False)
    if p01_cold is None:
        return hbm_rd
    _, sector = primitive_l2_params(context)
    cores = max(1, int(getattr(context, "core_count", 1) or 1))
    if is_grouped and is_full_row_coalesced:
        waves = float(math.ceil(max(1, int(total_tiles)) / cores))
        ramp_bw = min(p01_cold, hbm_rd) if hbm_rd is not None else p01_cold
        if not is_arch35_context(context) and row_bytes < stride_bytes:
            strided = primitive_p01_strided_cold_bps(context, row_bytes, stride_bytes)
            if strided is not None and strided > 0.0:
                ramp_bw = min(ramp_bw, strided)
        if hbm_rd is not None and active_cores >= cores and waves > GROUPED_HBM_RAMP_WAVES:
            ramp = GROUPED_HBM_RAMP_WAVES / ramp_bw
            steady = (waves - GROUPED_HBM_RAMP_WAVES) / hbm_rd
            return waves / (ramp + steady)
        return ramp_bw
    cache_disable = (
        bool(weight_cache_disable)
        if weight_cache_disable is not None
        else weight_l2_cache_disabled(
            context, m_tiles=m_tiles, row_bytes=row_bytes,
            stride_bytes=stride_bytes, is_grouped=is_grouped,
        )
    )
    eff_row = max(1, int(row_bytes))
    eff_stride = max(1, int(stride_bytes))
    rows = int(panel_rows) if (panel_rows is not None and int(panel_rows) > 0) else 0
    if rows <= 0:
        op = getattr(context, "op", None)
        shape = getattr(op, "local_shape", ()) or ()
        if len(shape) >= _GEMM_SHAPE_AXES:
            rows = max(1, int(shape[-1]))
    sector_eff = panel_sector_efficiency(
        row_bytes=eff_row, stride_bytes=eff_stride, rows=rows, sector_bytes=sector)
    cap = p01_cold * sector_eff
    if hbm_rd is not None:
        cap = min(cap, hbm_rd)
    if eff_row < eff_stride:
        if not (cache_disable and is_arch35_context(context)):
            strided = primitive_p01_strided_cold_bps(context, row_bytes, stride_bytes)
            if strided is not None:
                cap = min(cap, strided)
        if not cache_disable:
            stride_conflict = primitive_p05_stride_conflict_bps(context, stride_bytes)
            if stride_conflict is not None:
                cap = min(cap, stride_conflict)
        return cap
    return min(hbm_rd, cap) if hbm_rd is not None else cap


def primitive_wave_cold_hbm_per_owner_bps(
    context, *, cold_tiles: int, row_bytes: int, stride_bytes: int,
    is_grouped: bool, is_full_row_coalesced: bool = False, nn_tiles: int = 1,
    m_tiles: int | None = None, weight_cache_disable: bool | None = None,
    panel_rows: int | None = None,
) -> float | None:
    """Per-owner cold-HBM cap solving the shared-HBM fluid Engine wave equation.

    On the Engine, ``full_waves`` of ``cores`` owners followed by ``tail``
    owners complete in ``F*C/min(H, C*c) + T/min(H, T*c)``. Solving that
    equation for ``c`` against physical ``F*C/bw_full + T/bw_tail`` avoids
    double-throttling tail waves when the full waves already saturate ``H``.
    """
    if getattr(context, "primitive_table", None) is None or cold_tiles < 1:
        return None
    cores = max(1, int(getattr(context, "core_count", 1) or 1))
    full_waves, tail = divmod(int(cold_tiles), cores)
    max_waves = full_waves + (1 if tail > 0 else 0)
    inv_bw = 0.0
    if full_waves > 0:
        bw_full = primitive_cold_hbm_bps(
            context, active_cores=cores, row_bytes=row_bytes,
            stride_bytes=stride_bytes, total_tiles=cold_tiles,
            is_grouped=is_grouped, is_full_row_coalesced=is_full_row_coalesced,
            m_tiles=m_tiles, weight_cache_disable=weight_cache_disable,
            panel_rows=panel_rows,
        )
        if bw_full is None or bw_full <= 0.0:
            return None
        inv_bw += float(full_waves * cores) / bw_full
    if tail > 0:
        tail_coalesced = is_full_row_coalesced and (tail >= max(1, int(nn_tiles)))
        bw_tail = primitive_cold_hbm_bps(
            context, active_cores=tail, row_bytes=row_bytes,
            stride_bytes=stride_bytes, total_tiles=cold_tiles,
            is_grouped=is_grouped, is_full_row_coalesced=tail_coalesced,
            m_tiles=m_tiles, weight_cache_disable=weight_cache_disable,
            panel_rows=panel_rows,
        )
        if bw_tail is None or bw_tail <= 0.0:
            return None
        inv_bw += float(tail) / bw_tail
    if inv_bw <= 0.0:
        return None
    cap = float(max_waves) / inv_bw
    hbm_name = (getattr(context, "resource_names", None) or {}).get("hbm")
    hbm_die_bps = (
        float(context.rates[hbm_name]) * MS_PER_SECOND
        if (hbm_name and hbm_name in getattr(context, "rates", {}))
        else (primitive_p01_hbm_read_bps(context) or 0.0)
    )
    if full_waves > 0 and tail > 0 and hbm_die_bps > 0.0 and (cap * float(cores)) > hbm_die_bps:
        # Full waves are saturated by shared Engine HBM ``hbm_die_bps``; solve
        # remaining tail-wave budget ``inv_bw - F*C/H`` for per-owner cap ``c``.
        tail_budget = inv_bw - float(full_waves * cores) / hbm_die_bps
        if tail_budget > 0.0:
            return min(hbm_die_bps / float(tail), 1.0 / tail_budget)
        return hbm_die_bps / float(tail)
    return cap


def streamk_fixup_latency_ms(
    context, *, k_splits: int, used_cores: int,
) -> float:
    """Upstream DAV_3510 Stream-K AIV fixup loop setup + cross-core sync."""
    if k_splits <= 1:
        return 0.0
    is_arch35 = is_arch35_context(context)
    neg_ms = 0.0
    if is_arch35:
        clock_hz = primitive_core_clock_hz(context)
        cores = max(1, int(getattr(context, "core_count", 1) or 1))
        used = max(1, min(cores, int(used_cores)))
        if clock_hz is not None and clock_hz > 0.0:
            cycles = float(k_splits) * STREAMK_NEG_CYCLES_PER_SPLIT + STREAMK_NEG_BASE_CYCLES
            neg_ms = (cycles / clock_hz) * MS_PER_SECOND * (float(cores) / float(used))
    if getattr(context, "primitive_table", None) is None:
        return neg_ms
    if is_arch35:
        sync_ms = (
            primitive_p13_sync_ms(context, "mode0_aiv_barrier_ns")
            + primitive_p13_sync_ms(context, "mode4_intra_block_ns")
        )
    else:
        sync_ms = primitive_p13_sync_ms(context, "mode0_aic_barrier_ns")
    if sync_ms <= 0.0:
        sync_ms = primitive_p13_sync_ms(context, "mode0_aic_barrier_ns")
    return neg_ms + sync_ms


def _pure_aic_launch_ms(context) -> float:
    """Pure-AIC RTS/STARS dispatch + Cube pipe fill/drain from L0 P02/P03/P15."""
    issue_ms = _primitive_band_ms(context, "P15_rts_stars", "issue_interval_ns")
    pipe_ms = (
        _primitive_band_ms(context, "P02_load_data", "setup_ns")
        + _primitive_band_ms(context, "P03_fixpipe", "setup_ns")
    )
    return issue_ms + pipe_ms


def _mix_aic_launch_ms(context) -> float:
    """MIX_AIC Stream-K dual-side AIC+AIV dispatch + Cube pipe setup from L0 P02/P03/P15.

    Cross-core Stream-K synchronization (1x mode-4 NotifyVector/WaitForCube +
    1x AIV SyncAll per kernel_matmul_streamk.h:187,205-206) is charged once at
    the ``split_sync`` dependency via ``streamk_fixup_latency_ms``.
    """
    issue_ms = _primitive_band_ms(context, "P15_rts_stars", "issue_interval_ns")
    pipe_ms = (
        _primitive_band_ms(context, "P02_load_data", "setup_ns")
        + _primitive_band_ms(context, "P03_fixpipe", "setup_ns")
    )
    return MIX_AIC_SUBBLOCK_DISPATCHES * issue_ms + pipe_ms


def matrix_kernel_setup_ms(
    context, *, is_streamk: bool = False, declared_groups: int = 1,
) -> float:
    """Additive device-side setup from calibrated L0 primitives + groupListGm scalar scan."""
    if getattr(context, "primitive_table", None) is None:
        return 0.0
    if is_streamk:
        base_ms = _mix_aic_launch_ms(context)
    elif is_arch35_context(context):
        base_ms = _pure_aic_launch_ms(context)
    else:
        # On DAV_2201 pure-AIC MatMulV3, AIC launch overlaps steady execution and
        # acts as an inclusive minimum_ms floor (see matrix_kernel_floor_ms).
        base_ms = 0.0
    clock_hz = primitive_core_clock_hz(context)
    scan_ms = (
        (float(declared_groups) * GROUP_LIST_SCALAR_LOAD_CYCLES / clock_hz) * MS_PER_SECOND
        if (int(declared_groups) > 1 and clock_hz is not None and clock_hz > 0.0)
        else 0.0
    )
    return base_ms + scan_ms


def matrix_kernel_floor_ms(context) -> float:
    """Inclusive pure-AIC kernel floor from calibrated L0 P15/P02/P03/P16 primitives."""
    if getattr(context, "primitive_table", None) is None:
        return 0.0
    return _pure_aic_launch_ms(context)
