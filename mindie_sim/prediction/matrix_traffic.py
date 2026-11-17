"""Dense matrix owner scheduling and physical traffic equations.

[Role] Enumerate legal tile/refill/Split-K schedules, separate cold HBM reads
  from L2-origin operand refills, and derive finite owner tails.
[Invariants] Useful and issued arithmetic are separate; a private refill is not
  an HBM read. Split-K partials and the final output are charged exactly once.
  Capacity constraints are checked before a schedule is admitted.
[Timeouts & Locks] Bounded candidate enumeration; no I/O or locks.
[Side-Effect Policy] Pure immutable schedule/traffic records.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Iterator

from mindie_sim.schema.common import ModelingError, dtype_bytes
from mindie_sim.schema.physical_mechanisms import (MatrixPolicySpec,
                                                   MemoryHierarchySpec)

CACHE_LINE = 128
MAX_OWNERS = 32
FP32_BYTES = 4


@dataclass(frozen=True)
class MatrixGeometry:
    m: int
    n: int
    k: int
    groups: int = 1
    input_bytes: float | None = None
    weight_bytes: float | None = None
    output_bytes: float | None = None
    scale_bytes: float = 0.0

    def __post_init__(self):
        if (self.m < 0 or self.n < 0 or self.k < 0 or self.groups < 1
                or self.scale_bytes < 0):
            raise ModelingError("matrix_geometry_invalid")
        for value in (self.input_bytes, self.weight_bytes, self.output_bytes):
            if value is not None and value <= 0:
                raise ModelingError("matrix_geometry_invalid")


def _resolve_matrix_bytes(op, geometry: MatrixGeometry) -> MatrixGeometry:
    """Resolve unset operand byte widths from the operator's declared dtypes."""
    if (geometry.input_bytes is not None and geometry.weight_bytes is not None
            and geometry.output_bytes is not None):
        return geometry
    meta = dict(getattr(op, "metadata", ()) or ())
    op_dtype = str(getattr(op, "dtype", "") or "bf16")
    input_bytes = (
        float(geometry.input_bytes) if geometry.input_bytes is not None
        else float(dtype_bytes(str(meta.get("input_format")
                                  or meta.get("activation_format") or op_dtype))))
    weight_bytes = (
        float(geometry.weight_bytes) if geometry.weight_bytes is not None
        else float(dtype_bytes(str(meta.get("weight_format") or op_dtype))))
    output_bytes = (
        float(geometry.output_bytes) if geometry.output_bytes is not None
        else float(dtype_bytes(str(meta.get("output_format")
                                  or meta.get("output_dtype") or op_dtype))))
    return replace(geometry, input_bytes=input_bytes,
                   weight_bytes=weight_bytes, output_bytes=output_bytes)


@dataclass(frozen=True)
class MatrixSchedule:
    tile_m: int
    tile_n: int
    refill_k: int
    micro_m: int
    micro_n: int
    micro_k: int
    split_k: int
    swizzle_m: int
    swizzle_n: int
    input_slots: int
    compute_slots: int
    output_policy: str

    def __post_init__(self):
        if any(value < 1 for value in (
                self.tile_m, self.tile_n, self.refill_k, self.micro_m,
                self.micro_n, self.micro_k, self.split_k, self.swizzle_m,
                self.swizzle_n, self.input_slots, self.compute_slots)):
            raise ModelingError("matrix_schedule_positive_integers_required")
        if self.tile_m % self.micro_m or self.tile_n % self.micro_n:
            raise ModelingError("matrix_refill_tiles_must_be_multiples_of_microtiles")


@dataclass(frozen=True)
class MatrixTraffic:
    hbm_read_bytes: float
    hbm_write_bytes: float
    l2_hit_bytes: float
    private_refill_bytes: float
    mte1_bytes: float
    partial_bytes: float


@dataclass(frozen=True)
class OwnerSchedule:
    owners: tuple[tuple[int, tuple[tuple[int, int], ...]], ...]
    tail_rectangle: tuple[int, int]
    wave_count: int


def matrix_schedule_candidates(op, geometry, policy: MatrixPolicySpec
                               ) -> Iterator[MatrixSchedule]:
    """Yield capacity-admissible schedules declared by the matrix policy."""
    if not isinstance(geometry, MatrixGeometry):
        raise ModelingError("matrix_schedule_requires_typed_geometry")
    dtype = str(getattr(op, "dtype", "bf16") or "bf16")
    micros = policy.micro_tiles_by_dtype.get(dtype)
    if not micros:
        micros = ((policy.base_tiles[0][0], policy.base_tiles[0][1],
                   policy.refill_k_tiles[0]),)
    for tile_m, tile_n in policy.base_tiles:
        for refill_k in policy.refill_k_tiles:
            for micro in micros:
                micro_m, micro_n, micro_k = micro
                if tile_m % micro_m or tile_n % micro_n:
                    continue
                if micro_m > tile_m or micro_n > tile_n:
                    continue
                output_tiles = (math.ceil(geometry.m / tile_m)
                                * math.ceil(geometry.n / tile_n))
                for split in policy.split_k_candidates:
                    if split > 1:
                        k_per_split = geometry.k // split
                        if k_per_split < policy.min_k_per_split:
                            continue
                    for swizzle in policy.swizzle_windows:
                        yield MatrixSchedule(
                            tile_m, tile_n, refill_k, micro_m, micro_n,
                            micro_k, split, swizzle[0], swizzle[1],
                            policy.input_slots, policy.compute_slots,
                            policy.supported_output_paths[0],
                        )


def _service_bytes(geometry: MatrixGeometry, schedule: MatrixSchedule,
                   hierarchy: MemoryHierarchySpec) -> MatrixTraffic:
    del hierarchy
    m_tiles = max(1, math.ceil(geometry.m / schedule.tile_m))
    n_tiles = max(1, math.ceil(geometry.n / schedule.tile_n))
    k_panels = max(1, math.ceil(geometry.k / schedule.refill_k))
    refill = 0.0
    for m_extent in _extents(geometry.m, schedule.tile_m, m_tiles):
        for n_extent in _extents(geometry.n, schedule.tile_n, n_tiles):
            refill += float(k_panels) * (
                m_extent * geometry.k * geometry.input_bytes
                + n_extent * geometry.k * geometry.weight_bytes)
    cold = (geometry.m * geometry.k * geometry.input_bytes
            + geometry.k * geometry.n * geometry.weight_bytes
            + geometry.m * geometry.n * geometry.output_bytes
            + geometry.scale_bytes)
    partial = (2 * schedule.split_k * geometry.m * geometry.n
               * FP32_BYTES if schedule.split_k > 1 else 0.0)
    return MatrixTraffic(
        hbm_read_bytes=float(geometry.m * geometry.k * geometry.input_bytes
                             + geometry.k * geometry.n * geometry.weight_bytes
                             + geometry.scale_bytes),
        hbm_write_bytes=float(geometry.m * geometry.n * geometry.output_bytes),
        l2_hit_bytes=max(0.0, refill - cold),
        private_refill_bytes=refill,
        mte1_bytes=refill,
        partial_bytes=partial,
    )


def _extents(total: int, tile: int, count: int) -> tuple[int, ...]:
    return tuple(min(tile, total - index * tile) for index in range(count))


def plan_matrix_traffic(op, geometry, schedule: MatrixSchedule,
                        hierarchy: MemoryHierarchySpec) -> MatrixTraffic:
    """Charge cold operands once, L2 refills separately and partials only for Split-K."""
    if not isinstance(geometry, MatrixGeometry):
        raise ModelingError("matrix_traffic_requires_typed_geometry")
    if not isinstance(hierarchy, MemoryHierarchySpec):
        raise ModelingError("matrix_traffic_requires_memory_hierarchy")
    resolved = _resolve_matrix_bytes(op, geometry)
    return _service_bytes(resolved, schedule, hierarchy)


def schedule_matrix_owners(geometry: MatrixGeometry,
                           schedule: MatrixSchedule) -> OwnerSchedule:
    """Assign output tiles round-robin; preserve every real tail rectangle."""
    m_tiles = max(1, math.ceil(geometry.m / schedule.tile_m))
    n_tiles = max(1, math.ceil(geometry.n / schedule.tile_n))
    owners = [[] for _ in range(max(1, min(MAX_OWNERS, m_tiles * n_tiles)))]
    for row in range(m_tiles):
        for column in range(n_tiles):
            extent = (min(schedule.tile_m, geometry.m - row * schedule.tile_m),
                      min(schedule.tile_n, geometry.n - column * schedule.tile_n))
            owner = (row * n_tiles + column) % len(owners)
            owners[owner].append(extent)
    tail = (min(schedule.tile_m, geometry.m - (m_tiles - 1) * schedule.tile_m),
            min(schedule.tile_n, geometry.n - (n_tiles - 1) * schedule.tile_n))
    active = tuple((index, tuple(items)) for index, items in enumerate(owners) if items)
    return OwnerSchedule(active, tail, max(len(items) for _, items in active))
