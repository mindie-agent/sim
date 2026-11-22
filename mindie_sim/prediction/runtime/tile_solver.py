"""Exact on-chip SRAM / L2 peak-liveness TileSolver with algorithm fallback ladder.

[Role] Solve ``(algorithm, tile_shape, depth, owner_partition)`` from source-
  declared tiling candidates against physical UB, L1, L0A/B/C and per-die L2
  capacities, SIMT UB carveouts, compatible buffer aliasing, N-stream L2 knees,
  stream core-control wave quantization, and legal sub-tile instruction blocks.
[Invariants] Peak liveness ``P_{u,p} = max_t sum_{live at t} align(bytes, g_p)``
  charges compatible alias groups once per instant. Shared L2 enforces
  ``buffer_units.l2 == 1``. Issued small-M FLOPs pad only to legal instruction
  blocks without empirical ``subtile_eff`` divisors; logical FLOPs stay exact.
  Infeasible candidates advance strictly along the declared fallback ladder
  (``resident_1pass -> online_reduce -> chunk_recurrence -> row_split_2pass``)
  and surface structured pool/owner/required/available diagnostics if none fits.
[Timeouts & Locks] Finite sweep over candidate intervals; no I/O or locks.
[Side-Effect Policy] Pure functional solver; never mutates inputs.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from mindie_sim.schema.common import ModelingError

AlgorithmKind = Literal[
    "resident_1pass",
    "online_reduce",
    "chunk_recurrence",
    "row_split_2pass",
]
ALGORITHM_LADDER_RANK: Mapping[str, int] = {
    "resident_1pass": 0,
    "online_reduce": 1,
    "chunk_recurrence": 2,
    "row_split_2pass": 3,
}
DEFAULT_POOL_GRANULE_BYTES = 32
DEFAULT_INST_BLOCK_DIM = 16
DEFAULT_L2_GRANULE_BYTES = 128
DEFAULT_SIMT_UB_CARVEOUT_BYTES = 16 * 1024
AIV_UPCAST_BYTES_PER_ELEM = 4
CHECKED_POOLS = ("ub", "l1", "l0a", "l0b", "l0c", "l2", "bt", "fb", "ssbuffer")
_POOL_ALIASES = {"bias_table": "bt", "fixpipe_buffer": "fb"}
_FIXPIPE_BUFFER_ALIGNMENT_BYTES = 64
AIC_POOLS = frozenset({"l1", "l0a", "l0b", "l0c", "bt", "fb", *_POOL_ALIASES})
AIV_POOLS = frozenset({"ub"})


def ub_working_bytes(
    elements: int,
    element_bytes: int = 2,
    *,
    upcast_fp32: bool = False,
) -> int:
    """Compute UB working-set bytes, charging ``4B/elem`` (FP32) on AIV upcast."""
    if elements < 0 or element_bytes < 1:
        raise ModelingError("ub_working_bytes_invalid_dimensions")
    eff_bytes = max(int(element_bytes), AIV_UPCAST_BYTES_PER_ELEM) if upcast_fp32 else int(element_bytes)
    return int(elements) * eff_bytes


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _align_up(value: int, granule: int) -> int:
    if value <= 0 or granule <= 1:
        return max(0, value)
    return ((value + granule - 1) // granule) * granule


@dataclass(frozen=True, slots=True)
class TileShape:
    """Explicit physical tile dimensions."""

    m: int
    n: int
    k: int
    head_dim: int = 0
    v_dim: int = 0

    def __post_init__(self) -> None:
        if self.m < 0 or self.n < 0 or self.k < 0 or self.head_dim < 0 or self.v_dim < 0:
            raise ModelingError("tile_shape_negative_dimension")


@dataclass(frozen=True, slots=True)
class SubtileWaveSpec:
    """Hardware sub-tile issuance granularity for one native compute mode."""

    native_mode: str
    tile_m: int
    tile_n: int
    tile_k: int
    independently_issuable_m: int
    owner_groups: int = 1

    def __post_init__(self) -> None:
        if not self.native_mode:
            raise ModelingError("subtile_wave_mode_required")
        if (
            self.tile_m < 1 or self.tile_n < 1 or self.tile_k < 1
            or self.independently_issuable_m < 1 or self.owner_groups < 1
        ):
            raise ModelingError(f"subtile_wave_dims_invalid:{self.native_mode}")
        if self.independently_issuable_m > self.tile_m:
            raise ModelingError(f"subtile_wave_issuable_m_exceeds_tile:{self.native_mode}")


@dataclass(frozen=True, slots=True)
class LiveBufferSpec:
    """One live buffer allocation contributing to per-owner/per-pool peak liveness."""

    name: str
    pool: str
    bytes_per_owner: int
    live_start: int
    live_end: int
    owner_kind: Literal["aic", "aiv", "die"] = "aiv"
    alias_group: str | None = None
    alignment_bytes: int = DEFAULT_POOL_GRANULE_BYTES
    scale_bytes: int = 0
    slots: int = 1
    upcast_fp32: bool = False
    elements_per_owner: int = 0

    def __post_init__(self) -> None:
        if not self.name or not self.pool:
            raise ModelingError("live_buffer_identity_required")
        if self.bytes_per_owner < 0 or self.scale_bytes < 0 or self.slots < 1 or self.elements_per_owner < 0:
            raise ModelingError(f"live_buffer_size_invalid:{self.name}")
        if self.live_end < self.live_start:
            raise ModelingError(f"live_buffer_interval_reversed:{self.name}")
        if self.alignment_bytes < 1:
            raise ModelingError(f"live_buffer_alignment_invalid:{self.name}")
        if self.pool in AIC_POOLS and self.owner_kind != "aic":
            raise ModelingError(
                f"live_buffer_pool_owner_mismatch:{self.name}:{self.pool}:{self.owner_kind}"
            )
        if self.pool in AIV_POOLS and self.owner_kind != "aiv":
            raise ModelingError(
                f"live_buffer_pool_owner_mismatch:{self.name}:{self.pool}:{self.owner_kind}"
            )
        if self.pool == "l2" and self.owner_kind not in {"aic", "aiv", "die"}:
            raise ModelingError(
                f"live_buffer_pool_owner_mismatch:{self.name}:{self.pool}:{self.owner_kind}"
            )


@dataclass(frozen=True, slots=True)
class TileCandidate:
    """One source-declared numerically equivalent tiling candidate."""

    algorithm: AlgorithmKind
    tile_shape: TileShape
    depth: int
    owner_partition: tuple[int, int]
    work_items: int
    tiles_per_owner: int
    buffers: tuple[LiveBufferSpec, ...]
    logical_flops: int = 0
    issued_flops: int | None = None
    instruction_block: tuple[int, int, int] = (
        DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM, DEFAULT_INST_BLOCK_DIM,
    )
    uses_simt: bool = False
    native_mode: str = ""
    read_bytes: int = 0
    write_bytes: int = 0
    carry_bytes: int = 0
    owner_axis: Literal["aic", "aiv"] = "aic"
    source_guard: str = ""

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHM_LADDER_RANK:
            raise ModelingError(f"tile_candidate_algorithm_invalid:{self.algorithm!r}")
        if self.depth < 1 or self.work_items < 0 or self.tiles_per_owner < 0:
            raise ModelingError(f"tile_candidate_counts_invalid:{self.algorithm}")
        if any(dim < 1 for dim in self.instruction_block):
            raise ModelingError("tile_candidate_instruction_block_invalid")


@dataclass(frozen=True, slots=True)
class TilePipelineSpec:
    """Family-declared candidate sequence and hardware carveout/subtile constraints."""

    family: str
    candidates: tuple[TileCandidate, ...]
    simt_ub_carveout_bytes: int = 0
    subtile_waves: tuple[SubtileWaveSpec, ...] = ()
    pool_granules: Mapping[str, int] = field(default_factory=dict)
    active_l2_streams: int = 1
    l2_reserved_bytes: int = 0
    l2_granule_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.family:
            raise ModelingError("tile_pipeline_family_required")
        if self.simt_ub_carveout_bytes < 0 or self.active_l2_streams < 1 or self.l2_reserved_bytes < 0:
            raise ModelingError("tile_pipeline_spec_invalid_constraints")
        seen_modes: set[str] = set()
        for sw in self.subtile_waves:
            if sw.native_mode in seen_modes:
                raise ModelingError(f"subtile_wave_duplicate_mode:{sw.native_mode}")
            seen_modes.add(sw.native_mode)


@dataclass(frozen=True, slots=True)
class OwnerPoolPeak:
    """Peak live bytes for one ``(owner_kind, pool)`` pair."""

    owner_kind: str
    pool: str
    peak_bytes: int
    capacity_bytes: int


@dataclass(frozen=True, slots=True)
class TileRejection:
    """Structured rejection reason when a tiling candidate overflows a pool."""

    algorithm: str
    pool: str
    owner_kind: str
    required_bytes: int
    available_bytes: int
    source_guard: str


class TileInfeasibleError(ModelingError):
    """Structured error raised when every declared tiling candidate overflows."""

    def __init__(self, family: str, rejections: tuple[TileRejection, ...]) -> None:
        self.family = family
        self.rejections = rejections
        detail = ";".join(
            f"{r.algorithm}:{r.pool}({r.owner_kind})={r.required_bytes}>{r.available_bytes}"
            for r in rejections
        )
        super().__init__(f"tile_infeasible:{family}:{detail}")


@dataclass(frozen=True, slots=True)
class TileSolution:
    """Accepted tiling solution with exact waves, padded issued FLOPs and pool peaks."""

    algorithm: AlgorithmKind
    tile_shape: TileShape
    depth: int
    owner_partition: tuple[int, int]
    waves: int
    active_owners: int
    total_owners: int
    tail_owners: int
    tiles_per_owner: int
    logical_flops: int
    issued_flops: int
    independently_issuable_m: int
    subtile_wave_count: int
    read_bytes: int
    write_bytes: int
    carry_bytes: int
    pool_peaks: tuple[OwnerPoolPeak, ...]
    rejections: tuple[TileRejection, ...] = ()


def _pool_peak_bytes(
    buffers: tuple[LiveBufferSpec, ...],
    owner_kind: str,
    pool: str,
    granule: int,
    *,
    active_by_kind: Mapping[str, int] | None = None,
) -> int:
    """Compute exact ``max_t sum_{live at t} align(bytes, g_p)`` with alias folding."""
    matching = [
        buf for buf in buffers
        if _POOL_ALIASES.get(buf.pool, buf.pool) == pool and (buf.owner_kind == owner_kind or pool == "l2")
    ]
    if not matching:
        return 0
    scales = active_by_kind or {"aic": 1, "aiv": 1, "die": 1}
    time_points = sorted({buf.live_start for buf in matching})
    peak = 0
    for moment in time_points:
        direct_sum = 0
        alias_peaks: dict[str, int] = {}
        for buf in matching:
            if not (buf.live_start <= moment < buf.live_end or (
                buf.live_start == buf.live_end == moment
            )):
                continue
            eff_granule = max(granule, buf.alignment_bytes)
            payload_bytes = buf.bytes_per_owner
            if pool == "ub" and buf.upcast_fp32 and buf.elements_per_owner > 0:
                payload_bytes = max(payload_bytes, buf.elements_per_owner * AIV_UPCAST_BYTES_PER_ELEM)
            per_slot = _align_up(payload_bytes + buf.scale_bytes, eff_granule)
            total_b = per_slot * buf.slots
            if pool == "l2":
                kind_scale = int(scales.get(buf.owner_kind, 1))
                total_b *= max(0, kind_scale)
            if buf.alias_group is not None:
                prev = alias_peaks.get(buf.alias_group, 0)
                if total_b > prev:
                    alias_peaks[buf.alias_group] = total_b
            else:
                direct_sum += total_b
        instant_total = direct_sum + sum(alias_peaks.values())
        if instant_total > peak:
            peak = instant_total
    return peak


def _available_pool_capacity(
    pool: str,
    hw: Any,
    spec: TilePipelineSpec,
    *,
    uses_simt: bool,
    buffers: Mapping[str, int],
    l2_granule: int,
) -> int:
    if pool not in buffers:
        return 0
    raw_cap = int(buffers[pool])
    if pool == "l2":
        buf_units = dict(getattr(hw, "buffer_units", {}) or {})
        if buf_units.get("l2", 1) != 1:
            raise ModelingError("shared_l2_buffer_requires_one_die_owner")
        avail = max(0, raw_cap - spec.l2_reserved_bytes)
        g = max(1, l2_granule)
        return g * (avail // (g * max(1, spec.active_l2_streams)))
    if pool == "ub" and uses_simt:
        dp = getattr(hw, "datapath", None)
        hw_carveout = max(
            int(getattr(hw, "simt_ub_carveout_bytes", 0) or 0),
            int(getattr(dp, "simt_ub_carveout_bytes", 0) or 0) if dp is not None else 0,
        )
        carveout = max(int(spec.simt_ub_carveout_bytes), hw_carveout, DEFAULT_SIMT_UB_CARVEOUT_BYTES)
        return max(0, raw_cap - carveout)
    return max(0, raw_cap)


def _effective_owners(hw: Any, axis: Literal["aic", "aiv"]) -> int:
    stream_owners = getattr(hw, "stream_owners", None)
    if stream_owners is not None:
        return int(stream_owners.aic_limit if axis == "aic" else stream_owners.aiv_limit)
    datapath = getattr(hw, "datapath", None) or hw
    declared = getattr(datapath, f"{axis}_count", None)
    if declared is not None:
        return int(declared)
    if axis == "aic":
        return max(1, int(getattr(hw, "compute_units", 1)))
    res_units = dict(getattr(hw, "resource_units", {}) or {})
    return max(1, int(res_units.get("vector", getattr(hw, "compute_units", 1))))


def _issued_flops_for(
    candidate: TileCandidate,
    subtile_spec: SubtileWaveSpec | None,
) -> tuple[int, int, int]:
    """Return ``(issued_flops, independently_issuable_m, subtile_wave_count)``."""
    shape = candidate.tile_shape
    if subtile_spec is not None:
        bm = subtile_spec.independently_issuable_m
        bn = subtile_spec.tile_n
        bk = subtile_spec.tile_k
        issuable_m = subtile_spec.independently_issuable_m
        subtile_waves = (
            _ceil_div(shape.m, issuable_m) if shape.m > 0 else 0
        )
    else:
        bm, bn, bk = candidate.instruction_block
        issuable_m = bm
        subtile_waves = _ceil_div(shape.m, bm) if shape.m > 0 else 0

    if candidate.issued_flops is not None:
        return int(candidate.issued_flops), issuable_m, subtile_waves
    if candidate.work_items == 0 and candidate.logical_flops == 0:
        return 0, issuable_m, 0
    if shape.m == 0 or shape.n == 0 or shape.k == 0:
        return int(candidate.logical_flops), issuable_m, subtile_waves
    pad_m = _align_up(shape.m, bm)
    pad_n = _align_up(shape.n, bn)
    pad_k = _align_up(shape.k, bk)
    # 2 * pad_m * pad_n * pad_k per tile, scaled across work_items * max(1, tiles_per_owner)
    per_tile_logical = 2 * shape.m * shape.n * shape.k
    per_tile_issued = 2 * pad_m * pad_n * pad_k
    if candidate.logical_flops > 0 and per_tile_logical > 0:
        scale_tiles = _ceil_div(candidate.logical_flops, per_tile_logical)
        issued = scale_tiles * per_tile_issued
    else:
        issued = per_tile_issued * candidate.work_items
    return max(int(candidate.logical_flops), issued), issuable_m, subtile_waves


def solve(spec: TilePipelineSpec, op: Any, hw: Any) -> TileSolution:
    """Select the first source-declared candidate fitting every on-chip and L2 pool."""
    del op
    if not spec.candidates:
        raise TileInfeasibleError(spec.family, ())

    datapath = getattr(hw, "datapath", None) or hw
    buffers = {
        _POOL_ALIASES.get(pool, pool): size
        for source in (datapath, hw)
        for pool, size in dict(getattr(source, "buffers", {}) or {}).items()
    }
    memory = getattr(datapath, "memory_access", None)
    pool_granules = {_POOL_ALIASES.get(pool, pool): size for pool, size in spec.pool_granules.items()}
    l2_granule = (spec.l2_granule_bytes if spec.l2_granule_bytes is not None
                  else getattr(memory, "l2_sector_bytes", DEFAULT_L2_GRANULE_BYTES))
    subtile_by_mode = {sw.native_mode: sw for sw in getattr(datapath, "subtile_wave", ())}
    subtile_by_mode.update((sw.native_mode, sw) for sw in spec.subtile_waves)
    rejections: list[TileRejection] = []
    ordered_candidates = sorted(
        spec.candidates, key=lambda c: ALGORITHM_LADDER_RANK[c.algorithm],
    )

    for candidate in ordered_candidates:
        avail_aic = _effective_owners(hw, "aic")
        avail_aiv = _effective_owners(hw, "aiv")
        avail_owners = avail_aic if candidate.owner_axis == "aic" else avail_aiv
        if avail_owners <= 0:
            rejections.append(TileRejection(
                algorithm=candidate.algorithm,
                pool=candidate.owner_axis,
                owner_kind=candidate.owner_axis,
                required_bytes=1,
                available_bytes=0,
                source_guard=candidate.source_guard or "zero_stream_owners",
            ))
            continue

        part_aic, part_aiv = candidate.owner_partition
        total_aic = min(avail_aic, part_aic) if part_aic > 0 else avail_aic
        total_aiv = min(avail_aiv, part_aiv) if part_aiv > 0 else avail_aiv
        total_owners = total_aic if candidate.owner_axis == "aic" else total_aiv
        u_c = candidate.work_items
        waves = _ceil_div(u_c, total_owners) if u_c > 0 else 0
        active_owners = min(u_c, total_owners) if u_c > 0 else 0
        tail_owners = (u_c - (waves - 1) * total_owners) if waves > 0 else 0
        if u_c == 0:
            active_aic = 0
            active_aiv = 0
        elif candidate.owner_axis == "aic":
            active_aic = active_owners
            if total_aic > 0 and total_aiv >= total_aic:
                active_aiv = min(total_aiv, active_aic * _ceil_div(total_aiv, total_aic))
            else:
                active_aiv = min(u_c, total_aiv)
        else:
            active_aiv = active_owners
            active_aic = min(u_c, total_aic)
        active_by_kind = {"aic": active_aic, "aiv": active_aiv, "die": 1}

        peaks: list[OwnerPoolPeak] = []
        overflow: TileRejection | None = None
        pool_owners = [
            (pool, "die" if pool == "l2" else "aiv" if pool in AIV_POOLS else "aic")
            for pool in CHECKED_POOLS if pool != "ssbuffer"
        ]
        pool_owners.extend(("ssbuffer", kind) for kind in dict.fromkeys(
            buf.owner_kind for buf in candidate.buffers if buf.pool == "ssbuffer"
        ))
        for pool, owner_kind in pool_owners:
            default_granule = _FIXPIPE_BUFFER_ALIGNMENT_BYTES if pool == "fb" else DEFAULT_POOL_GRANULE_BYTES
            granule = int(pool_granules.get(pool, default_granule))
            peak = _pool_peak_bytes(
                candidate.buffers,
                owner_kind,
                pool,
                granule,
                active_by_kind=active_by_kind,
            )
            if peak <= 0:
                continue
            cap = _available_pool_capacity(
                pool, hw, spec, uses_simt=candidate.uses_simt, buffers=buffers, l2_granule=l2_granule,
            )
            peaks.append(OwnerPoolPeak(
                owner_kind=owner_kind,
                pool=pool,
                peak_bytes=peak,
                capacity_bytes=cap,
            ))
            if peak > cap:
                overflow = TileRejection(
                    algorithm=candidate.algorithm,
                    pool=pool,
                    owner_kind=owner_kind,
                    required_bytes=peak,
                    available_bytes=cap,
                    source_guard=("ssbuffer_capacity_open_limit" if pool == "ssbuffer" and pool not in buffers
                                  else candidate.source_guard),
                )
                break

        if overflow is not None:
            rejections.append(overflow)
            continue

        sw = subtile_by_mode.get(candidate.native_mode) if candidate.native_mode else None
        issued_flops, issuable_m, subtile_waves = _issued_flops_for(candidate, sw)
        return TileSolution(
            algorithm=candidate.algorithm,
            tile_shape=candidate.tile_shape,
            depth=candidate.depth,
            owner_partition=candidate.owner_partition,
            waves=waves,
            active_owners=active_owners,
            total_owners=total_owners,
            tail_owners=tail_owners,
            tiles_per_owner=candidate.tiles_per_owner,
            logical_flops=int(candidate.logical_flops),
            issued_flops=issued_flops,
            independently_issuable_m=issuable_m,
            subtile_wave_count=subtile_waves,
            read_bytes=int(candidate.read_bytes),
            write_bytes=int(candidate.write_bytes),
            carry_bytes=int(candidate.carry_bytes),
            pool_peaks=tuple(peaks),
            rejections=tuple(rejections),
        )

    raise TileInfeasibleError(spec.family, tuple(rejections))


__all__ = [
    "AIV_UPCAST_BYTES_PER_ELEM",
    "ALGORITHM_LADDER_RANK",
    "AlgorithmKind",
    "CHECKED_POOLS",
    "DEFAULT_INST_BLOCK_DIM",
    "DEFAULT_POOL_GRANULE_BYTES",
    "DEFAULT_SIMT_UB_CARVEOUT_BYTES",
    "LiveBufferSpec",
    "OwnerPoolPeak",
    "SubtileWaveSpec",
    "TileCandidate",
    "TileInfeasibleError",
    "TilePipelineSpec",
    "TileRejection",
    "TileShape",
    "TileSolution",
    "solve",
    "ub_working_bytes",
]
