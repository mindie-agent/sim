"""Source-derived owner schedule for the A8W4 MSD gate and grouped down.

[Role]
  Turn pinned host tile constants and kernel rolling-owner loops into bounded
  per-owner phase work records.  This module performs arithmetic only.

[Invariants]
  * Gate MSD uses the pinned host A8W4_BASEM/N/K 128/256/256.  Its matrix grid
    is ceil(2*rows/baseM) by ceil(N/baseN) per active expert and the rolling
    cursor continues across experts, so global tile ``t`` belongs to AIC
    ``t % aic_owners``.
  * Gate pre/post partition contiguous input rows by quotient/remainder over
    AIVs.  The post body loads one assist row per owner group segment; an
    expert crossing owners is consequently charged more than once.
  * Down uses the pinned non-whitelist MSD host choice: baseM 64 when
    ``M/expert_count < 32`` else 128, baseN 512 or 256. Its host selects CV11
    or CV12 independently of the physical two-AIV-per-AIC inventory. Every
    participating AIV copies the assist row; UB row chunks rotate from
    vecCount=1, not from the matrix tile ordinal.
  * Owner counts come from the declared engine geometry.  No idle unit lends
    service to a pinned owner; HBM with one unit remains shared.
  * Zero-row experts retain their source bank offsets and issue no matrix
    tile. Down pre-processing still copies and reduces all group counts on
    every participating AIV, including owners with no input row.

[Timeouts & Locks]
  Pure arithmetic, O(experts + owners) retained records; no I/O or locks.

[Side-Effect Policy]
  Returns immutable records only; no engine submission or observation access.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mindie_sim.schema.compute_arch_presets import CAPABILITY_PRESETS
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.prediction.mechanism import Work
FP32_BYTES = 4

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="vllm-project/vllm-ascend",
        revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="csrc/gmm/grouped_matmul_swiglu_quant_v2/op_kernel/grouped_matmul_swiglu_quant_v2_a8w4_msd_mid.h",
        source_sha256="1881d03e65258e846fe9bbfb736f4cbbcd770b84a6730319dd0c9d00abdc4087",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="gmm/grouped_matmul/op_kernel/grouped_matmul_antiquant_a8w4_msd.h",
        source_sha256="b092d276255c441542a5d205dbec5718d2ba5ae0ab7f5e637a2dba5fb20ee619",
    ),
)

NZ_N_ALIGNMENT = 64
COLUMN_SCALE_BYTES = 8
GROUP_COUNT_BYTES = 8
MSD_DIGITS = 2
MSD_FLOPS_PER_K_STEP = MSD_DIGITS * 2
MSD_WORKSPACE_BYTES_PER_OUTPUT = MSD_DIGITS * 2  # One FP16 value per digit.
PRE_OPS_PER_INPUT = 6.5
POST_DEQUANT_OPS_PER_OUTPUT = 6
SWIGLU_OPS_PER_OUTPUT = 5
QUANT_OPS_PER_OUTPUT = 5
CV11_MAX_EXPERT_ROWS = 128
CV11_MAX_EXPERTS = 4
CV11_MAX_K = 8192
CV11_MAX_N = 8192
POST_DEQUANT_UB_ELEMENTS = 8192
POST_DEQUANT_COLUMN_ALIGNMENT = 8

GATE_BASE_M = 128
GATE_BASE_N = 256
GATE_BASE_K = 256
DOWN_SHORT_M = 32
DOWN_BASE_M = 64
DOWN_BASE_N = 512
DOWN_BASE_M_TALL = 128
DOWN_BASE_N_TALL = 256


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def align_up(value: int, granule: int) -> int:
    return ceil_div(value, granule) * granule


def resolve_int4_subtile(subtile_wave=None) -> tuple[int, int, int]:
    """Return registered A2/A3 INT4 Cube subtile ``(tile_m, tile_n, tile_k)``."""
    if subtile_wave is not None:
        return (
            int(subtile_wave.independently_issuable_m),
            int(subtile_wave.tile_n),
            int(subtile_wave.tile_k),
        )
    wave = next(
        w for w in CAPABILITY_PRESETS["npu_arch_2201"]["subtile_waves"]
        if w.native_mode == "int4"
    )
    return int(wave.independently_issuable_m), int(wave.tile_n), int(wave.tile_k)


DEFAULT_INT4_SUBTILE_M, DEFAULT_INT4_SUBTILE_N, DEFAULT_INT4_SUBTILE_K = resolve_int4_subtile()


@dataclass(frozen=True)
class OwnerWork:
    owner: int
    work: Work
    rows: int = 0
    tiles: int = 0
    assist_loads: int = 0


@dataclass(frozen=True)
class ExpertScan:
    index: int
    rows: int
    active: bool
    double_rows: int
    m_tiles: int
    n_tiles: int
    tile_count: int
    last_m_tile: int
    last_n_tile: int
    weight_offset_elements: int
    active_weight_bytes: float


@dataclass(frozen=True)
class GateSchedule:
    m: int
    n: int
    k: int
    base_m: int
    base_n: int
    aic_owners: int
    aiv_owners: int
    experts: tuple[ExpertScan, ...]
    pre: tuple[OwnerWork, ...]
    matrix: tuple[OwnerWork, ...]
    post_dequant: tuple[OwnerWork, ...]
    swiglu: tuple[OwnerWork, ...]
    quantize: tuple[OwnerWork, ...]
    tile_count: int
    expert_scan_count: int
    zero_bank_offsets: tuple[int, ...]
    subtile_m: int = DEFAULT_INT4_SUBTILE_M
    subtile_n: int = DEFAULT_INT4_SUBTILE_N
    subtile_k: int = DEFAULT_INT4_SUBTILE_K

    @property
    def active_experts(self) -> int:
        return sum(expert.active for expert in self.experts)


@dataclass(frozen=True)
class DownSchedule:
    m: int
    n: int
    k: int
    base_m: int
    base_n: int
    aivs_per_aic: int
    aic_owners: int
    aiv_owners: int
    experts: tuple[ExpertScan, ...]
    pre: tuple[OwnerWork, ...]
    matrix: tuple[OwnerWork, ...]
    post_dequant: tuple[OwnerWork, ...]
    output: tuple[OwnerWork, ...]
    tile_count: int
    expert_scan_count: int
    zero_bank_offsets: tuple[int, ...]
    subtile_m: int = DEFAULT_INT4_SUBTILE_M
    subtile_n: int = DEFAULT_INT4_SUBTILE_N
    subtile_k: int = DEFAULT_INT4_SUBTILE_K

    @property
    def active_experts(self) -> int:
        return sum(expert.active for expert in self.experts)


def _acc():
    return {"matrix": {}, "resources": {}}


def _add(acc, *work):
    for item in work:
        for key, amount in item.matrix.items():
            acc["matrix"][key] = acc["matrix"].get(key, 0.0) + amount
        for key, amount in item.resources.items():
            acc["resources"][key] = acc["resources"].get(key, 0.0) + amount


def _finish(acc) -> Work:
    return Work(dict(acc["matrix"]), dict(acc["resources"]))


def _owner_records(acc, counts, tiles=None, rows=None, assists=None):
    records = []
    for owner in sorted(acc):
        work = _finish(acc[owner])
        if not work.matrix and not work.resources:
            continue
        records.append(OwnerWork(
            owner, work,
            rows=(rows or {}).get(owner, 0),
            tiles=(tiles or {}).get(owner, 0),
            assist_loads=(assists or {}).get(owner, 0),
        ))
    return tuple(records)


def source_row_partition(total: int, owners: int):
    """Contiguous ceil/remainder row chunks from the gate pre/post kernels."""
    if total < 0 or owners < 1:
        raise ValueError("row_partition_requires_nonnegative_total_and_owners")
    each = ceil_div(total, owners)
    used = min(total, owners)
    tail = total - (each - 1) * used
    result = []
    for owner in range(used):
        count = each if owner < tail else each - 1
        start = each * owner if owner < tail else (each - 1) * owner + tail
        if count:
            result.append((owner, start, count))
    return tuple(result)


def strided_rows(total: int, owners: int):
    """Row counts for the down pre kernel's block-strided AIV loop."""
    if total < 0 or owners < 1:
        raise ValueError("strided_rows_requires_nonnegative_total_and_owners")
    return tuple((owner, (total - 1 - owner) // owners + 1)
                 for owner in range(min(total, owners)))


def aiv_group(owner: int, aic_owners: int, aiv_owners: int):
    """Physical CV pair; unsupported inventories cannot borrow virtual units."""
    if aiv_owners != 2 * aic_owners:
        raise ValueError("w4a8_source_requires_two_physical_aivs_per_aic")
    return (2 * owner, 2 * owner + 1)


def aiv_pair(owner: int, aic_owners: int, aiv_owners: int):
    return aiv_group(owner, aic_owners, aiv_owners)


def _active_ranges(rows):
    ranges, start = [], 0
    for value in rows:
        end = start + value
        if value:
            ranges.append((start, end))
        start = end
    return ranges


def _assist_loads(owner_rows, group_ranges):
    """Count owner group segments in one monotone owner/group scan."""
    loads = {}
    cursor = 0
    for owner, start, count in owner_rows:
        end = start + count
        while cursor < len(group_ranges) and group_ranges[cursor][1] <= start:
            cursor += 1
        index, count_loads = cursor, 0
        while index < len(group_ranges) and group_ranges[index][0] < end:
            if group_ranges[index][1] > start:
                count_loads += 1
            index += 1
        loads[owner] = count_loads
    return loads


def _scans(rows, n, k, base_m, base_n):
    experts, offset = [], 0
    for index, value in enumerate(rows):
        double_rows = 2 * value
        m_tiles = ceil_div(double_rows, base_m) if value else 0
        n_tiles = ceil_div(n, base_n)
        last_m = double_rows - (m_tiles - 1) * base_m if m_tiles else 0
        last_n = n - (n_tiles - 1) * base_n
        experts.append(ExpertScan(
            index=index, rows=value, active=bool(value), double_rows=double_rows,
            m_tiles=m_tiles, n_tiles=n_tiles, tile_count=m_tiles * n_tiles,
            last_m_tile=last_m, last_n_tile=last_n,
            weight_offset_elements=index * n * k,
            active_weight_bytes=0.5 * n * k if value else 0.0,
        ))
        offset += m_tiles * n_tiles
    return tuple(experts), offset


def _matrix_tiles(rows, n, k, base_m, base_n, aic_owners, fixpipe, *, subtile_dims=None):
    """Aggregate the rolling global tile grid without retaining tile records."""
    sub_m, sub_n, sub_k = subtile_dims or (
        DEFAULT_INT4_SUBTILE_M, DEFAULT_INT4_SUBTILE_N, DEFAULT_INT4_SUBTILE_K,
    )
    issued_k = align_up(k, sub_k)
    owners_acc = {}
    owner_tiles, owner_m, owner_n = {}, {}, {}
    start = 0
    for expert in _scans(rows, n, k, base_m, base_n)[0]:
        if not expert.active:
            continue
        block_n = expert.n_tiles
        for local in range(expert.tile_count):
            m_index, n_index = divmod(local, block_n)
            m_tile = min(base_m, expert.double_rows - m_index * base_m)
            n_tile = min(base_n, n - n_index * base_n)
            owner = (start + local) % aic_owners
            add = owners_acc.setdefault(owner, _acc())
            weight = 0.5 * n_tile * k
            scale = COLUMN_SCALE_BYTES * n_tile
            # Every independently owned N tile submits SetTensorA on its own
            # input slice. No shared L2 hit or multicast is declared here.
            activation = 0.5 * m_tile * k
            resources = {"hbm": activation + weight + scale + 2 * m_tile * n_tile,
                         "mte2": activation + weight + scale}
            if fixpipe:
                resources["fixpipe"] = 2 * m_tile * n_tile
            issued_m = align_up(m_tile, sub_m)
            issued_n = align_up(n_tile, sub_n)
            _add(add, Work({"int4": 2 * issued_m * issued_n * issued_k}, resources),
                 Work(resources={"vector": m_tile * n_tile}))
            owner_tiles[owner] = owner_tiles.get(owner, 0) + 1
            owner_m[owner] = owner_m.get(owner, 0) + m_tile
            owner_n[owner] = owner_n.get(owner, 0) + n_tile
        start += expert.tile_count
    return owners_acc, owner_tiles, owner_m, owner_n, start


def _gate_pre(rows, k, aiv_owners):
    acc = {}
    row_records = source_row_partition(sum(rows), aiv_owners)
    for owner, _, count in row_records:
        acc[owner] = _acc()
        _add(acc[owner], Work(resources={
            "hbm": 2 * count * k, "mte2": count * k, "mte3": count * k,
            "vector": PRE_OPS_PER_INPUT * count * k}))
    return _owner_records(acc, {}, rows={owner: count for owner, _, count in row_records})


def _gate_post(rows, n, k, aiv_owners):
    h = n // 2
    ranges = _active_ranges(rows)
    owner_rows = source_row_partition(sum(rows), aiv_owners)
    loads = _assist_loads(owner_rows, ranges)
    dequant, swiglu, quant = {}, {}, {}
    row_counts = {}
    for owner, _, count in owner_rows:
        assist = loads[owner]
        row_counts[owner] = count
        dequant[owner] = _acc()
        dequant_read_bytes = (MSD_WORKSPACE_BYTES_PER_OUTPUT * n * count
                              + FP32_BYTES * n * assist + FP32_BYTES * count)
        _add(dequant[owner], Work(resources={
            "hbm": dequant_read_bytes,
            "mte2": dequant_read_bytes,
            "vector": POST_DEQUANT_OPS_PER_OUTPUT * n * count,
        }))
        swiglu[owner] = _acc()
        _add(swiglu[owner], Work(resources={"vector": SWIGLU_OPS_PER_OUTPUT * h * count}))
        quant[owner] = _acc()
        _add(quant[owner], Work(resources={
            "hbm": h * count + FP32_BYTES * count,
            "mte3": h * count + FP32_BYTES * count,
            "vector": (QUANT_OPS_PER_OUTPUT * h + 1) * count,
        }))
    return (
        _owner_records(dequant, {}, rows=row_counts, assists=loads),
        _owner_records(swiglu, {}, rows=row_counts, assists=loads),
        _owner_records(quant, {}, rows=row_counts, assists=loads),
    )


def gate_schedule(m, n, k, expert_rows, *, aic_owners, aiv_owners, fixpipe=False, subtile_wave=None):
    """Bounded source-derived owner work for the gate kernel."""
    if min(m, n, k, aic_owners, aiv_owners) < 1 or sum(expert_rows) != m:
        raise ValueError("gate_schedule_requires_conserving_positive_geometry")
    aiv_group(0, aic_owners, aiv_owners)
    sub_m, sub_n, sub_k = resolve_int4_subtile(subtile_wave)
    experts, tile_count = _scans(expert_rows, n, k, GATE_BASE_M, GATE_BASE_N)
    acc, tiles, _, _, _ = _matrix_tiles(
        expert_rows, n, k, GATE_BASE_M, GATE_BASE_N, aic_owners, fixpipe,
        subtile_dims=(sub_m, sub_n, sub_k))
    matrix = _owner_records(acc, {}, tiles=tiles)
    pre = _gate_pre(expert_rows, k, aiv_owners)
    post, swiglu, quant = _gate_post(expert_rows, n, k, aiv_owners)
    zeros = tuple(expert.weight_offset_elements for expert in experts if not expert.active)
    return GateSchedule(
        m=m, n=n, k=k, base_m=GATE_BASE_M, base_n=GATE_BASE_N,
        aic_owners=aic_owners, aiv_owners=aiv_owners, experts=experts,
        pre=pre, matrix=matrix, post_dequant=post, swiglu=swiglu, quantize=quant,
        tile_count=tile_count, expert_scan_count=len(experts),
        zero_bank_offsets=zeros,
        subtile_m=sub_m, subtile_n=sub_n, subtile_k=sub_k,
    )


def _down_pre(rows, k, aic_owners, task_ratio):
    acc = {}
    owners = aic_owners * task_ratio
    counts = dict(strided_rows(sum(rows), owners))
    physical_counts = {}
    for logical_owner in range(owners):
        owner = logical_owner if task_ratio == 2 else 2 * logical_owner
        count = counts.get(logical_owner, 0)
        physical_counts[owner] = count
        acc[owner] = _acc()
        # The explicit DataCopyPad/Cast/ReduceSum/one-result Cast runs before
        # the xloop guard on every launched AIV. Scalar group lookup latency
        # elsewhere is deliberately not inferred from this vector reduction.
        metadata_bytes = GROUP_COUNT_BYTES * len(rows)
        _add(acc[owner], Work(resources={
            "hbm": 2 * count * k + metadata_bytes,
            "mte2": count * k + metadata_bytes, "mte3": count * k,
            "vector": PRE_OPS_PER_INPUT * count * k + 2 * len(rows),
        }))
    return _owner_records(acc, {}, rows=physical_counts)


def down_task_ratio(m, n, k, expert_count):
    """Host CV11 predicate for the declared non-whitelist API-dequant domain."""
    return 1 if (m // expert_count <= CV11_MAX_EXPERT_ROWS and expert_count <= CV11_MAX_EXPERTS
                 and k <= CV11_MAX_K and n <= CV11_MAX_N) else 2


def down_schedule(m, n, k, expert_rows, *, aic_owners, aiv_owners, fixpipe=False, subtile_wave=None):
    """Bounded source-derived owner work for the ordinary grouped-down kernel."""
    if min(m, n, k, aic_owners, aiv_owners) < 1 or sum(expert_rows) != m:
        raise ValueError("down_schedule_requires_conserving_positive_geometry")
    aiv_group(0, aic_owners, aiv_owners)
    sub_m, sub_n, sub_k = resolve_int4_subtile(subtile_wave)
    short = (m // len(expert_rows)) < DOWN_SHORT_M
    base_m = DOWN_BASE_M if short else DOWN_BASE_M_TALL
    base_n = DOWN_BASE_N if short else DOWN_BASE_N_TALL
    experts, tile_count = _scans(expert_rows, n, k, base_m, base_n)
    matrix_acc, tiles, _, _, _ = _matrix_tiles(
        expert_rows, n, k, base_m, base_n, aic_owners, fixpipe,
        subtile_dims=(sub_m, sub_n, sub_k))
    matrix = _owner_records(matrix_acc, {}, tiles=tiles)
    task_ratio = down_task_ratio(m, n, k, len(expert_rows))
    pre = _down_pre(expert_rows, k, aic_owners, task_ratio)

    post, output = {}, {}
    assist_loads, active_tiles = {}, {}
    start = 0
    vec_count = [0] * aic_owners
    for expert in experts:
        if not expert.active:
            continue
        block_n = expert.n_tiles
        for local in range(expert.tile_count):
            m_index, n_index = divmod(local, block_n)
            m_tile = min(base_m, expert.double_rows - m_index * base_m)
            n_tile = min(base_n, n - n_index * base_n)
            owner = (start + local) % aic_owners
            pair = aiv_pair(owner, aic_owners, aiv_owners)[:task_ratio]
            scale = Work(resources={"hbm": FP32_BYTES * n_tile, "mte2": FP32_BYTES * n_tile})
            for unit in pair:
                acc = post.setdefault(unit, _acc())
                _add(acc, scale)
                assist_loads[unit] = assist_loads.get(unit, 0) + 1
            rows_in_tile = m_tile // 2
            # UBCALSIZE=32*256, and the kernel divides it by the aligned
            # host baseN. The counter persists across this AIC's tiles.
            aligned_base_n = (ceil_div(base_n, POST_DEQUANT_COLUMN_ALIGNMENT)
                              * POST_DEQUANT_COLUMN_ALIGNMENT)
            vec_base_m = min(POST_DEQUANT_UB_ELEMENTS // aligned_base_n, rows_in_tile)
            for offset_m in range(0, rows_in_tile, vec_base_m):
                chunk_rows = min(vec_base_m, rows_in_tile - offset_m)
                vec_count[owner] += 1
                unit = pair[vec_count[owner] % task_ratio]
                acc = post.setdefault(unit, _acc())
                dequant_read_bytes = (MSD_WORKSPACE_BYTES_PER_OUTPUT * chunk_rows * n_tile
                                      + FP32_BYTES * chunk_rows)
                _add(acc, Work(resources={
                    "hbm": dequant_read_bytes,
                    "mte2": dequant_read_bytes,
                    "vector": POST_DEQUANT_OPS_PER_OUTPUT * chunk_rows * n_tile}))
                active_tiles[unit] = active_tiles.get(unit, 0) + 1
                out = output.setdefault(unit, _acc())
                _add(out, Work(resources={
                    "hbm": 2 * chunk_rows * n_tile, "mte3": 2 * chunk_rows * n_tile,
                    "vector": chunk_rows * n_tile}))
        start += expert.tile_count
    post_records = _owner_records(post, {}, assists=assist_loads, tiles=active_tiles)
    output_records = _owner_records(output, {}, tiles=active_tiles)
    zeros = tuple(expert.weight_offset_elements for expert in experts if not expert.active)
    return DownSchedule(
        m=m, n=n, k=k, base_m=base_m, base_n=base_n,
        aivs_per_aic=task_ratio, aic_owners=aic_owners, aiv_owners=aiv_owners,
        experts=experts, pre=pre, matrix=matrix, post_dequant=post_records,
        output=output_records, tile_count=tile_count,
        expert_scan_count=len(experts), zero_bank_offsets=zeros,
        subtile_m=sub_m, subtile_n=sub_n, subtile_k=sub_k,
    )


def total_work(*owner_groups) -> Work:
    total = Work()
    for group in owner_groups:
        for item in group:
            total += item.work
    return total


__all__ = [
    "DOWN_BASE_M", "DOWN_BASE_N", "DOWN_BASE_M_TALL", "DOWN_BASE_N_TALL",
    "DownSchedule", "ExpertScan", "GATE_BASE_M", "GATE_BASE_N", "GATE_BASE_K",
    "GateSchedule", "OwnerWork", "aiv_group", "aiv_pair", "down_schedule",
    "gate_schedule", "source_row_partition", "strided_rows", "total_work",
]
