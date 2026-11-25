"""Physical work and finite execution programs for paged MLA.

[Role] Lower tensor extents into owner-local stages and runtime credits.
[Invariants] Keep logical bytes, copy bytes, and DRAM transactions distinct.
[Timeouts & Locks] Stream tiles in a bounded execution window. No locks.
[Side-Effect Policy] Build immutable plans and runtime stages without I/O.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from graphlib import TopologicalSorter
from itertools import accumulate, chain, groupby
from math import gcd, isqrt, lcm
import sys
from typing import Any, Literal

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Program, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.memory_physics import AddressRun, AddressRunPlan, ByteLedger, HbmBusSpec, MemoryAccessPattern, ReuseKey
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..attention.common import AttentionMemoryRead, planning_sequence_spans, sequence_layout_metadata
from ..contracts import OperatorPlanningContext
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sharedkv.l0_sync import (
    p01_gather_v2_cold_bps, p01_hbm_scatter_row_open_bytes, p01_paged_copy_setup_ms,
    p01_strided_cold_bps, p02_l0b_load_factor, p03_fixpipe_setup_ms, p13_syncall_ms,
)
from ..sharedkv.split_core import p01_cold_mte2_aic_cap
from .common import (
    ACLNN_MQS_DATA_BYTES, ACLNN_MQS_INLINE_ROW_BYTES, ACLNN_MQS_SCALE_BYTES, ACLNN_MQS_SEPARATE_SCALE_MODE,
    DSL_FLASH_ABI, DSL_MQS_ABI, FAMILY, FIA_ABI, FLASH_ABI, FP32_BYTES, INDEX_BYTES,
    MQS_ABI, MQS_COMPRESSED_ROW_BYTES, MQS_ORIGINAL_ROW_BYTES,
    MQS_PAGED_LAYOUT, NZ_BLOCK_BYTES, PagedMlaContract, _query_row, align_bytes, capability_gap,
    ceil_div, latent_record_bytes, mla_step_crosscore_ms,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
FIA_ISSUE_ORDER = {
    "qk": (("pv", -2),), "pv": (("qk", 1),),
    "softmax": (("output_update", -2),), "output_update": (("softmax", 1),),
}
AIV_ROLES = frozenset({"mte2_aiv", "mte3", "vector"})
SOFTMAX_OPS_PER_KEY = 5
SOFTMAX_MERGE_SCALAR_OPS = 8
FIA_WORKSPACE_DEPTH = 2
WEIGHTED_MERGE_OPS_PER_VALUE = 3
FP8_DEQUANT_OPS_PER_VALUE = 6
FP4_DEQUANT_OPS_PER_VALUE = 4
SOFTMAX_STATE_SCALARS = 2
# Upstream CANN FIA & FlashMLA broadcast 8 FP32 copies each of max and sum (64 B/head).
FD_SOFTMAX_STATE_SCALARS = 16
DENSE_CANN_ABIS = frozenset({FIA_ABI, FLASH_ABI})
SINGLE_AIV_HEAD_LIMIT = 16
FD_REDUCE_HEAD_GRAIN = 8
# CANN split_core / split_core_v2 host tiling constants (split_core.cpp, split_core_v2.cpp).
FIA_COST_M_WEIGHT = 6
FIA_COST_M_GRAIN = 16
FIA_COST_S2_WEIGHT = 10
FIA_COST_S2_GRAIN = 64
FIA_TOLERANCE_DIVISOR = 2
FIA_MIN_CORE_OFFSET_NUM = 1
FIA_MIN_CORE_OFFSET_DEN = 4
FIA_MIN_CORE_ROUND_NUM = 1
FIA_MIN_CORE_ROUND_DEN = 2
FIA_ARCH22_M_TILE = 256
FIA_ARCH22_M_TILE_MIN = 128
FIA_ARCH22_S2_TILE = 512
FIA_ARCH35_M_TILE = 64
FIA_ARCH35_S2_TILE = 128
FIA_ARCH35_FD_TOLERANCE_BLOCKS = 9
MLA_COMPACT_THRESHOLD = 32
MLA_COMPACT_MIN_KEY_TILE = 128
MQS_REFILL_ROW_ALIGN = 16
# Arch22 FIA MLA cube tiling constants (fia_block_cube_nonquant_mla.h:590-655, 872-918).
# kL1=0 and kL1=1 load [0..512) and [512..1024) of each 1024 B NoPE row back-to-back.
ARCH22_MLA_NL1_SPLIT = 128
ARCH22_MLA_MM1_KL1_LOOPS = 2
ARCH22_MLA_NOPE_ROW_BYTES = 1024
ARCH22_MLA_STRIDE_PROBE_BYTES = 2048
# Arch35 FIA MLA CopyKeyAndRopeTile issues two plane copies (fia_block_cube_noquant_mla.h:257,270).
ARCH35_MLA_PLANE_COPY_CALLS = 2
MS_PER_SECOND_MLA = 1000.0
RowCuts = dict[tuple[int, int], tuple[int, ...]]


def _mla_kv_cold_caps(
    ctx: RecipeContext, active_owners: int, c: PagedMlaContract, layout: PipelineLayout, dtype: str,
) -> dict[str, float]:
    """Per-owner MTE2-AIC / HBM cold cap from P01 active-core scaling + strided/gather surface."""
    caps = dict(p01_cold_mte2_aic_cap(ctx, active_owners, include_hbm=True))
    if c.abi != FIA_ABI or c.mixed_quant or active_owners < 1:
        return caps
    die_bps: float | None = None
    if layout.gm_workspace:
        die_bps = p01_strided_cold_bps(ctx, ARCH22_MLA_NOPE_ROW_BYTES, ARCH22_MLA_STRIDE_PROBE_BYTES)
    else:
        elem = int(dtype_bytes(dtype))
        nope_b = c.latent_dim * elem
        rope_b = c.rope_dim * elem
        bw_nope = p01_gather_v2_cold_bps(ctx, nope_b, scatter=True)
        bw_rope = p01_gather_v2_cold_bps(ctx, rope_b, scatter=True) if rope_b > 0 else bw_nope
        if bw_nope is not None and bw_rope is not None and (nope_b + rope_b) > 0:
            die_bps = float(nope_b + rope_b) / (nope_b / bw_nope + rope_b / bw_rope)
    if die_bps is None or die_bps <= 0.0:
        return caps
    per_owner_ms = die_bps / (MS_PER_SECOND_MLA * float(active_owners))
    for role in ("mte2_aic", "hbm"):
        res_name = ctx.resource_names.get(role)
        if not res_name:
            continue
        prev = caps.get(res_name)
        caps[res_name] = min(prev, per_owner_ms) if prev is not None else per_owner_ms
    return caps


def _mla_step_latency_ms(
    step: str, c: PagedMlaContract, layout: PipelineLayout, keys: int, pages: int,
    base_sync_ms: float, fix_setup_ms: float, page_setup_ms: float,
) -> float:
    """Per-tile sync, Fixpipe command setup, and paged blockTable+ND2NZ setup latency (ms)."""
    lat = base_sync_ms
    if c.abi != FIA_ABI or c.mixed_quant:
        return lat
    if fix_setup_ms > 0.0:
        if step == "score_store":
            calls = ceil_div(keys, ARCH22_MLA_NL1_SPLIT) if layout.gm_workspace else 1
            lat += float(calls) * fix_setup_ms
        elif step == "partial":
            calls = ceil_div(c.dv, ARCH22_MLA_NL1_SPLIT) if layout.gm_workspace else 1
            lat += float(calls) * fix_setup_ms
    if c.kv_layout == "PA_BLK_BSND" and page_setup_ms > 0.0 and pages > 0:
        if layout.gm_workspace:
            # attn_copy_gm_to_l1.h:343,431: each 128-row N_L1 slice issues 1 cold GetBlockIdx in
            # ProcessMm1 (kL1=0); sub-128 pages fragment every kL1 (2x) and nL1 (4x) ND2NZ copy.
            extra_frags = max(0, pages - ceil_div(keys, ARCH22_MLA_NL1_SPLIT))
            if step == "kv_load":
                lookups = pages + ARCH22_MLA_MM1_KL1_LOOPS * extra_frags
                lat += float(lookups) * page_setup_ms
            elif step == "pv_load" and not layout.resident_latent and extra_frags > 0:
                lookups = ceil_div(c.dv, ARCH22_MLA_NL1_SPLIT) * extra_frags
                lat += float(lookups) * page_setup_ms
        elif step == "kv_load":
            lat += float(ARCH35_MLA_PLANE_COPY_CALLS * pages) * page_setup_ms
    return lat


def fixpipe_rectangle_bytes(rows: int, columns: int) -> int:
    """Dual-destination FixPipe divides even M rows across the two AIVs."""
    return align_bytes(rows, 2) * align_bytes(columns * FP32_BYTES)

@dataclass(frozen=True, slots=True)
class PipelineLayout:
    clocks: tuple[tuple[str, ...], ...]
    rings: tuple[tuple[str, int, str, str], ...]
    head_tile: int
    key_tile: int
    gm_workspace: bool = False
    resident_latent: bool = False
    matrix_output: tuple[str, int] = ("fixpipe", NZ_BLOCK_BYTES)
    probability_output: tuple[str, int] = ("mte3", NZ_BLOCK_BYTES)
    pv_panel_width: int = 0
    ws_l2_resident: bool = False
    latent_l2_hit: bool = False
    kv_reuse_hit: float = 0.0
    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(chain.from_iterable(self.clocks))

@dataclass(slots=True)
class _Demand:
    resources: dict[str, float] = field(default_factory=dict)
    flops: int = 0
    metadata: dict[str, int | float | str] = field(default_factory=dict)
    kv_reads: list[tuple[MemoryAccessPattern, ByteLedger]] | None = None

    def add(self, resource: str, amount: int | float) -> None:
        if amount:
            self.resources[resource] = self.resources.get(resource, 0) + amount
    def transfer(self, tensor: str, pipe: str, nbytes: int, plan: AddressRunPlan, *,
                 aligned_copy_bytes: int | None = None, l2_hit_fraction: float = 0.0) -> None:
        if not nbytes:
            return
        direction = "read" if pipe.startswith("mte2") else "write"
        access = MemoryAccessPattern(tensor, 0, 0, 0, direction, pipe, nbytes, plan)
        aligned = access.runs.aligned_copy_bytes() if aligned_copy_bytes is None else aligned_copy_bytes
        sectors, dram = access.runs.dram_sector_and_service()
        hit = max(0.0, min(1.0, float(l2_hit_fraction)))
        l2_bytes = int(round(aligned * hit))
        miss_copy = max(0, aligned - l2_bytes)
        miss_dram = int(round(dram * (1.0 - hit)))
        miss_sectors = int(round(sectors * (1.0 - hit)))
        ledger = ByteLedger(access.logical_bytes, aligned, miss_dram, l2_bytes, miss_copy, miss_sectors)
        if self.kv_reads is not None and tensor in {"key", "value"}:
            self.kv_reads.append((access, ledger))
        self.add(pipe, ledger.aligned_transfer_bytes)
        if ledger.l2_hit_bytes:
            self.add("l2", ledger.l2_hit_bytes)
        if ledger.dram_bus_transaction_bytes:
            self.add("hbm", ledger.dram_bus_transaction_bytes)
        for key, value in (
            ("logical_bytes", ledger.logical_tensor_bytes),
            ("aligned_transfer_bytes", ledger.aligned_transfer_bytes),
            ("l2_hit_bytes", ledger.l2_hit_bytes),
            ("dram_miss_copy_bytes", ledger.dram_miss_copy_bytes),
            ("dram_sector_bytes", ledger.dram_sector_bytes),
            ("dram_bus_transaction_bytes", ledger.dram_bus_transaction_bytes),
            (f"{tensor}_logical_bytes", ledger.logical_tensor_bytes),
        ):
            self.metadata[key] = self.metadata.get(key, 0) + value

    def linear(self, tensor: str, pipe: str, nbytes: int, offset: int = 0, *, l2_hit_fraction: float = 0.0) -> None:
        self.transfer(tensor, pipe, nbytes, AddressRunPlan((AddressRun(offset, 1, nbytes, nbytes),)),
                      l2_hit_fraction=l2_hit_fraction)

    def atomic_add(self, tensor: str, pipe: str, nbytes: int, offset: int, *, l2_hit_fraction: float = 0.0) -> None:
        self.linear(tensor, pipe, nbytes, offset, l2_hit_fraction=l2_hit_fraction)
        if l2_hit_fraction >= 1.0:
            self.add("l2", nbytes)
            self.metadata["atomic_read_bytes"] = nbytes
            return
        read_bytes = self.metadata["dram_bus_transaction_bytes"]
        self.add("hbm", read_bytes)
        self.metadata["dram_bus_transaction_bytes"] += read_bytes
        self.metadata["dram_sector_bytes"] *= 2
        self.metadata["atomic_read_bytes"] = read_bytes
    def work(self, dtype: str) -> Work:
        return Work({dtype: self.flops} if self.flops else {}, self.resources)

def _spans(start: int, count: int, page_size: int) -> Iterator[tuple[int, int, int]]:
    while count:
        page, offset = divmod(start, page_size)
        rows = min(count, page_size - offset)
        yield page, offset, rows
        start += rows
        count -= rows

def _kv_runs(c: PagedMlaContract, spans, width: int, stride: int, table, layout: str,
             *, base_token: int = 0) -> AddressRunPlan:
    runs = []
    for page, offset, rows in spans:
        physical_page = table[page] if table else page
        if layout in ("PA_BLK_BSND", MQS_PAGED_LAYOUT):
            base = (base_token + physical_page * c.page_size + offset) * stride
            runs.append(AddressRun(base, rows, width, stride, layout))
        else:
            base = physical_page * c.page_size * align_bytes(stride)
            column_stride = c.page_size * NZ_BLOCK_BYTES
            full, tail = divmod(width, NZ_BLOCK_BYTES)
            if full:
                runs.append(AddressRun(base + offset * NZ_BLOCK_BYTES, full,
                                       rows * NZ_BLOCK_BYTES, column_stride, layout))
            if tail:
                runs.append(AddressRun(base + full * column_stride + offset * NZ_BLOCK_BYTES,
                                       rows, tail, NZ_BLOCK_BYTES, layout))
    return AddressRunPlan(tuple(runs))

def _sparse_ranges(indices, start: int, count: int):
    if not indices:
        yield start, count
        return
    for _, run in groupby(enumerate(indices[start:start + count]), lambda item: item[1] - item[0]):
        yield next(run)[1], 1 + sum(1 for _ in run)

def _read_sparse_bank(demand, c, branch, spans, table, dsl_width):
    if c.abi != MQS_ABI or c.quant_mode != ACLNN_MQS_SEPARATE_SCALE_MODE:
        width = ACLNN_MQS_INLINE_ROW_BYTES if c.abi == MQS_ABI else dsl_width
        for span in spans:
            demand.transfer(branch, "mte2_aiv", span[2] * width,
                            _kv_runs(c, (span,), width, width, table, c.kv_layout))
        return
    # PA_BBND mode 2 packs the data plane first, then scale + pad for the block.
    row_width = ACLNN_MQS_DATA_BYTES + ACLNN_MQS_SCALE_BYTES
    logical_key = f"{branch}_logical_bytes"
    for page, offset, rows in spans:
        page_base = (table[page] if table else page) * row_width * c.page_size
        for plane, width, plane_offset in (
            ("data", ACLNN_MQS_DATA_BYTES, 0), ("scale", ACLNN_MQS_SCALE_BYTES, ACLNN_MQS_DATA_BYTES),
        ):
            base = page_base + plane_offset * c.page_size + offset * width
            run = AddressRun(base, rows, width, width, MQS_PAGED_LAYOUT)
            # Each scale lands in its own padded UB row.
            demand.transfer(f"{branch}_{plane}", "mte2_aiv", rows * width, AddressRunPlan((run,)),
                            aligned_copy_bytes=rows * align_bytes(width))
        demand.metadata[logical_key] = demand.metadata.get(logical_key, 0) + rows * row_width

def _read_kv(demand: _Demand, c: PagedMlaContract, seq: int, query: int, start: int, keys: int,
             dqk_bytes: int, dv_bytes: int, *, value: bool = False, base_token: int = 0,
             l2_hit_fraction: float = 0.0) -> None:
    if not c.mixed_quant:
        table = c.block_table[seq] if c.block_table else ()
        tensor, width = ("value", dv_bytes) if value else ("key", dqk_bytes)
        stride = dqk_bytes if c.latent_alias else width
        for page, offset, rows in _spans(start, keys, c.page_size):
            plan = _kv_runs(c, ((page, offset, rows),), width, stride, table, c.kv_layout,
                            base_token=0 if table else base_token)
            demand.transfer(tensor, "mte2_aic", rows * width, plan, l2_hit_fraction=l2_hit_fraction)
        first_page = start // c.page_size
        pages = ceil_div(start + keys, c.page_size) - first_page
        demand.linear(f"{tensor}_pages", "mte2_aic", INDEX_BYTES * pages, INDEX_BYTES * first_page,
                      l2_hit_fraction=l2_hit_fraction)
        return
    ori, cmp = c.bank_counts(seq, query)
    for branch, lo, hi, tables, lists, width in (
        ("ori", 0, ori, c.ori_block_table, c.ori_indices, MQS_ORIGINAL_ROW_BYTES),
        ("cmp", ori, ori + cmp, c.cmp_block_table, c.cmp_indices, MQS_COMPRESSED_ROW_BYTES),
    ):
        first, end = max(start, lo), min(start + keys, hi)
        if first >= end:
            continue
        count, offset = end - first, first - lo
        indices = _query_row(lists[seq], query, indices=True) if lists else ()
        ranges = _sparse_ranges(indices, offset, count)
        if c.kv_layout == MQS_PAGED_LAYOUT:
            table = tables[seq] if tables else ()
            spans = tuple(span for base, rows in ranges for span in _spans(base, rows, c.page_size))
            _read_sparse_bank(demand, c, branch, spans, table, width)
            pages = tuple(sorted({page for page, _, _ in spans}))
            for page_idx in pages:
                demand.transfer(
                    f"{branch}_pages", "mte2_aiv", INDEX_BYTES,
                    AddressRunPlan((AddressRun(page_idx * INDEX_BYTES, 1, INDEX_BYTES, INDEX_BYTES),)),
                )
        else:
            packed = ACLNN_MQS_INLINE_ROW_BYTES
            for base, rows in ranges:
                run = AddressRun((base_token + base) * packed, rows, packed, packed, c.kv_layout)
                demand.transfer(branch, "mte2_aiv", rows * packed, AddressRunPlan((run,)))
        demand.linear(f"{branch}_indices", "mte2_aiv", count * INDEX_BYTES if lists else 0,
                      offset * INDEX_BYTES)

def _split_tiles(tiles: int, splits: int, split: int) -> tuple[int, int]:
    width, extra = divmod(tiles, splits)
    return split * width + min(split, extra), width + (split < extra)

def _merge_ops(c: PagedMlaContract, rows: int, joins: int) -> int:
    return rows * joins * (SOFTMAX_MERGE_SCALAR_OPS + WEIGHTED_MERGE_OPS_PER_VALUE * c.dv)


def _live_rows_in_span(start: int, end: int, logical_heads: int, physical_heads: int) -> int:
    def _prefix(n: int) -> int:
        return (n // physical_heads) * logical_heads + min(n % physical_heads, logical_heads)

    return _prefix(end) - _prefix(start)


@dataclass(frozen=True, slots=True)
class _FiaRowSpec:
    seq: int
    m_block: int
    m_rows: int
    tiles: int
    normal_cost: int
    tail_cost: int
    total_cost: int


@dataclass(frozen=True, slots=True)
class _FiaHostPlan:
    """Resolved CANN SplitCore / SplitCore_v2 host schedule for FIA MLA."""

    m_tile: int
    s2_tile: int
    used_cores: int
    fd_enabled: bool
    max_splits: int
    core_slices: tuple[tuple[tuple[int, int, int, int], ...], ...]
    row_splits: dict[tuple[int, int], int]
    fd_reducers: tuple[tuple[tuple[int, int, int, int, int], ...], ...]


def _fia_block_cost(m_rows: int, s2_cols: int) -> int:
    return (
        FIA_COST_M_WEIGHT * ceil_div(m_rows, FIA_COST_M_GRAIN)
        + FIA_COST_S2_WEIGHT * ceil_div(s2_cols, FIA_COST_S2_GRAIN)
    )


def _eval_fia_split_core(
    rows: tuple[_FiaRowSpec, ...],
    batch_ends: tuple[int, ...],
    core_num: int,
    *,
    stream_k: bool,
    cost_limit: int | None = None,
) -> tuple[int, int, tuple[tuple[tuple[int, int, int, int], ...], ...], list[int]]:
    """Replay CANN SplitCore / SplitCore_v2 (AssignByBatch -> Row -> Block -> Force)."""
    n_rows = len(rows)
    total_cost = sum(r.total_cost for r in rows)
    if core_num <= 0 or total_cost <= 0 or n_rows == 0:
        return 1, 0, ((),), [1] * n_rows
    max_m_cost = max(r.total_cost for r in rows)
    batch_of = [0] * n_rows
    batch_costs = [0] * len(batch_ends)
    batch_blocks = [0] * len(batch_ends)
    batch_last = [0] * len(batch_ends)
    prev_end = 0
    for b_idx, b_end in enumerate(batch_ends):
        for r_i in range(prev_end, b_end):
            batch_of[r_i] = b_idx
            batch_costs[b_idx] += rows[r_i].total_cost
            batch_blocks[b_idx] += rows[r_i].tiles
        batch_last[b_idx] = rows[b_end - 1].tail_cost
        prev_end = b_end

    r_idx = tile_idx = 0
    b_idx = 0
    b_rem_cost, b_rem_blocks = batch_costs[0], batch_blocks[0]
    r_rem_cost, r_rem_tiles = rows[0].total_cost, rows[0].tiles
    unassigned = total_cost
    max_cost = 0
    splits = [0] * n_rows
    all_slices: list[tuple[tuple[int, int, int, int], ...]] = []

    def _reload_row() -> None:
        nonlocal b_idx, b_rem_cost, b_rem_blocks, r_rem_cost, r_rem_tiles
        if r_idx >= n_rows:
            return
        new_b = batch_of[r_idx]
        if new_b != b_idx:
            b_idx = new_b
            b_rem_cost, b_rem_blocks = batch_costs[b_idx], batch_blocks[b_idx]
        r_rem_cost, r_rem_tiles = rows[r_idx].total_cost, rows[r_idx].tiles

    for c_idx in range(core_num):
        if cost_limit is not None and max_cost > cost_limit:
            return len(all_slices), max_cost, tuple(all_slices), splits
        if r_idx >= n_rows or unassigned <= 0:
            break
        avg = unassigned // (core_num - c_idx)
        limit = avg if stream_k else max(avg, max_m_cost)
        cur_cost = cur_blocks = 0
        c_slices: list[tuple[int, int, int, int]] = []

        # 1. AssignByBatch
        while r_idx < n_rows and cur_cost + b_rem_cost <= limit + batch_last[b_idx] // FIA_TOLERANCE_DIVISOR:
            b_end = batch_ends[b_idx]
            for k in range(r_idx, b_end):
                st = tile_idx if k == r_idx else 0
                cnt = rows[k].tiles - st
                if cnt > 0:
                    c_slices.append((rows[k].seq, rows[k].m_block, st, cnt))
                    splits[k] += 1
            cur_cost += b_rem_cost
            cur_blocks += b_rem_blocks
            r_idx, tile_idx = b_end, 0
            _reload_row()

        # 2. AssignByRow
        while r_idx < n_rows and cur_cost + r_rem_cost <= limit + rows[r_idx].tail_cost // FIA_TOLERANCE_DIVISOR:
            r = rows[r_idx]
            c_slices.append((r.seq, r.m_block, tile_idx, r_rem_tiles))
            splits[r_idx] += 1
            cur_cost += r_rem_cost
            cur_blocks += r_rem_tiles
            b_rem_cost -= r_rem_cost
            b_rem_blocks -= r_rem_tiles
            r_idx, tile_idx = r_idx + 1, 0
            _reload_row()

        # 3. AssignByBlock
        if stream_k and r_idx < n_rows:
            r = rows[r_idx]
            st = tile_idx
            while tile_idx < r.tiles:
                t_cost = r.tail_cost if tile_idx == r.tiles - 1 else r.normal_cost
                if cur_cost + t_cost <= limit + t_cost // FIA_TOLERANCE_DIVISOR:
                    cur_cost += t_cost
                    cur_blocks += 1
                    tile_idx += 1
                    r_rem_cost -= t_cost
                    r_rem_tiles -= 1
                    b_rem_cost -= t_cost
                    b_rem_blocks -= 1
                else:
                    break
            if tile_idx > st:
                c_slices.append((r.seq, r.m_block, st, tile_idx - st))
                splits[r_idx] += 1
                if tile_idx == r.tiles:
                    r_idx, tile_idx = r_idx + 1, 0
                    _reload_row()

        # 4. ForceAssign
        if cur_blocks == 0 and r_idx < n_rows:
            r = rows[r_idx]
            if stream_k:
                t_cost = r.tail_cost if tile_idx == r.tiles - 1 else r.normal_cost
                c_slices.append((r.seq, r.m_block, tile_idx, 1))
                splits[r_idx] += 1
                cur_cost += t_cost
                cur_blocks = 1
                tile_idx += 1
                r_rem_cost -= t_cost
                r_rem_tiles -= 1
                b_rem_cost -= t_cost
                b_rem_blocks -= 1
                if tile_idx == r.tiles:
                    r_idx, tile_idx = r_idx + 1, 0
                    _reload_row()
            else:
                c_slices.append((r.seq, r.m_block, tile_idx, r_rem_tiles))
                splits[r_idx] += 1
                cur_cost += r_rem_cost
                cur_blocks += r_rem_tiles
                b_rem_cost -= r_rem_cost
                b_rem_blocks -= r_rem_tiles
                r_idx, tile_idx = r_idx + 1, 0
                _reload_row()

        all_slices.append(tuple(c_slices))
        max_cost = max(max_cost, cur_cost)
        unassigned -= cur_cost

    return max(1, len(all_slices)), max_cost, tuple(all_slices), splits


def _assign_fia_fd_reducers(
    fd_tasks: tuple[tuple[int, int, int, int], ...],
    used_cores: int,
    aiv_per_aic: int,
    *,
    is_arch35: bool,
) -> tuple[tuple[tuple[int, int, int, int, int], ...], ...]:
    """Replay CANN SplitFD AIV assignment (split_core.cpp:659, split_core_v2.cpp:625)."""
    if not fd_tasks:
        return ()
    max_vectors = max(1, used_cores * max(1, aiv_per_aic))
    if is_arch35:
        total_load = sum(m_rows * sp for _, _, m_rows, sp in fd_tasks)
        avg_load = max(1, ceil_div(total_load, max_vectors))
        empty_vecs = max(0, max_vectors - len(fd_tasks))
        groups: list[tuple[tuple[int, int, int, int, int], ...]] = []
        for seq, m_block, m_rows, sp in fd_tasks:
            if empty_vecs == 0:
                groups.append(((seq, m_block, 0, m_rows, sp),))
                continue
            v_num = max(1, (sp * m_rows) // avg_load)
            avg_m = ceil_div(m_rows, v_num)
            v_num = min(ceil_div(m_rows, avg_m), empty_vecs + 1)
            for vid in range(v_num):
                h_off = vid * avg_m
                ch_h = min(avg_m, m_rows - h_off)
                groups.append(((seq, m_block, h_off, ch_h, sp),))
            empty_vecs -= v_num - 1
        return tuple(groups)

    units: list[tuple[int, int, int, int, int]] = []
    for seq, m_block, m_rows, sp in fd_tasks:
        parts = ceil_div(m_rows, FD_REDUCE_HEAD_GRAIN)
        for p in range(parts):
            h_off = p * FD_REDUCE_HEAD_GRAIN
            units.append((seq, m_block, h_off, min(FD_REDUCE_HEAD_GRAIN, m_rows - h_off), sp))
    vectors = min(len(units), max_vectors)
    load = sum(u[4] for u in units)
    threshold = load / vectors
    current = 0
    a22_groups: list[tuple[tuple[int, int, int, int, int], ...]] = []
    cur_group: list[tuple[int, int, int, int, int]] = []
    for unit in units:
        sp = unit[4]
        if (current > 0 and len(a22_groups) + 1 < vectors
                and sp > (threshold - current) * FIA_TOLERANCE_DIVISOR):
            a22_groups.append(tuple(cur_group))
            load -= current
            threshold = load / (vectors - len(a22_groups))
            current = 0
            cur_group = []
        cur_group.append(unit)
        current += sp
    if cur_group:
        a22_groups.append(tuple(cur_group))
    return tuple(a22_groups)


def resolve_fia_host_plan(
    *,
    sequences: tuple[tuple[int, int], ...],
    physical_heads: int,
    aic_count: int,
    aiv_per_aic: int = 2,
    is_arch35: bool,
    page_size: int = 0,
    paged: bool = False,
) -> _FiaHostPlan:
    """Replay CANN Arch22/Arch35 FIA MLA tiling, SplitCore, and SplitFD reducer mapping."""
    batch = len(sequences)
    max_s1 = max((q for q, _ in sequences), default=1)
    if is_arch35:
        m_tile, s2_tile, tol = FIA_ARCH35_M_TILE, FIA_ARCH35_S2_TILE, FIA_ARCH35_FD_TOLERANCE_BLOCKS
    else:
        s2_tile = FIA_ARCH22_S2_TILE
        # fia_tiling_nonquant_mla.cpp:277-282 rounds S2 down to a multiple of PA blockSize.
        if paged and page_size > 0 and s2_tile > page_size and s2_tile % page_size != 0:
            s2_tile = (s2_tile // page_size) * page_size
        m_tile = FIA_ARCH22_M_TILE
        same_kv = len({kv for _, kv in sequences}) == 1
        s1_g = max_s1 * physical_heads
        if batch > 0 and aic_count > 0 and batch % aic_count != 0 and same_kv:
            lcm_val = (batch // gcd(batch, aic_count)) * aic_count
            split_of_s1g = lcm_val // batch
            if split_of_s1g > 0 and s1_g % split_of_s1g == 0:
                cand = s1_g // split_of_s1g
                if FIA_ARCH22_M_TILE_MIN <= cand <= m_tile:
                    m_tile = cand
        tol = 0

    rows_list: list[_FiaRowSpec] = []
    batch_ends: list[int] = []
    for seq, (q_len, kv_len) in enumerate(sequences):
        if q_len <= 0 or kv_len <= 0:
            continue
        total_m = q_len * physical_heads
        m_blocks = ceil_div(total_m, m_tile)
        tiles = ceil_div(kv_len, s2_tile)
        tail_s2 = kv_len - (tiles - 1) * s2_tile
        for m_idx in range(m_blocks):
            m_rows = min(m_tile, total_m - m_idx * m_tile)
            n_cost = _fia_block_cost(m_rows, s2_tile)
            t_cost = _fia_block_cost(m_rows, tail_s2)
            rows_list.append(_FiaRowSpec(seq, m_idx, m_rows, tiles, n_cost, t_cost, (tiles - 1) * n_cost + t_cost))
        batch_ends.append(len(rows_list))
    rows = tuple(rows_list)
    b_ends = tuple(batch_ends)
    total_blocks = sum(r.tiles for r in rows)
    if total_blocks <= 0:
        return _FiaHostPlan(m_tile, s2_tile, 1, False, 1, ((),), {}, ())

    max_core = min(aic_count, total_blocks)
    root = isqrt(total_blocks * FIA_MIN_CORE_OFFSET_DEN + FIA_MIN_CORE_OFFSET_NUM)
    min_core = max(1, min((root + FIA_MIN_CORE_ROUND_NUM) // FIA_MIN_CORE_ROUND_DEN, max_core))
    best_used, best_cost, best_slices, best_splits = 1, sys.maxsize, ((),), [1] * len(rows)
    for cores in range(min_core, max_core + 1):
        used, m_cost, slices, splits = _eval_fia_split_core(
            rows, b_ends, cores, stream_k=True, cost_limit=best_cost,
        )
        if m_cost < best_cost:
            best_used, best_cost, best_slices, best_splits = used, m_cost, slices, splits

    fd_on = max(best_splits, default=1) > 1
    if fd_on and tol > 0:
        no_used, no_cost, no_slices, no_splits = _eval_fia_split_core(rows, b_ends, best_used, stream_k=False)
        full_cost = _fia_block_cost(min(max_s1 * physical_heads, m_tile), s2_tile)
        if not ((no_cost - tol * full_cost) > best_cost):
            best_used, best_slices, best_splits, fd_on = no_used, no_slices, no_splits, False

    row_splits = {(r.seq, r.m_block): sp for r, sp in zip(rows, best_splits, strict=True)}
    fd_tasks = tuple(
        (r.seq, r.m_block, r.m_rows, sp)
        for r, sp in zip(rows, best_splits, strict=True) if sp > 1
    )
    fd_reducers = _assign_fia_fd_reducers(fd_tasks, best_used, aiv_per_aic, is_arch35=is_arch35)
    return _FiaHostPlan(
        m_tile, s2_tile, best_used, fd_on, max(best_splits, default=1),
        best_slices, row_splits, fd_reducers,
    )


@dataclass(frozen=True, slots=True)
class _PagedMlaProgramBuilder:
    contract: PagedMlaContract
    layout: PipelineLayout
    dtype: str
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    row_cuts: RowCuts
    sequence_spans: tuple[tuple[int, int, int, int, bool], ...] = ()
    planning: OperatorPlanningContext | None = None
    kv_tensor_id: str = ""
    kv_version: int = 0
    query_offsets: tuple[int, ...] = ()
    fia_host_plan: _FiaHostPlan | None = None

    def _pack_m(self) -> bool:
        return self.contract.abi in DENSE_CANN_ABIS

    def _m_blocks_for_seq(self, seq: int) -> int:
        q = self.contract.sequences[seq][0]
        if not q:
            return 0
        if self._pack_m():
            return ceil_div(q * self.contract.physical_heads, self.layout.head_tile)
        return q * ceil_div(self.contract.physical_heads, self.layout.head_tile)

    def _m_block_span(self, seq: int, m_block: int) -> tuple[int, int, int, int, int]:
        c, layout = self.contract, self.layout
        total_m = c.sequences[seq][0] * c.physical_heads
        m_start = m_block * layout.head_tile
        m_end = min(total_m, m_start + layout.head_tile)
        heads = m_end - m_start
        live = _live_rows_in_span(m_start, m_end, c.logical_heads, c.physical_heads)
        query, head_start = divmod(m_start, c.physical_heads)
        return m_start, heads, live, query, head_start

    def _demand(self, step: str, live: int, *, seq: int = 0, query: int = 0, heads: int = 0,
                start: int = 0, keys: int = 0, first: bool = False, last: bool = False,
                partials: int = 0, mm2_offset: int = 0, head_start: int = 0,
                l0b_factor: float = 1.0) -> _Demand:
        c, layout = self.contract, self.layout
        elem = int(dtype_bytes(self.dtype))
        qk_width, v_width = c.dqk * elem, c.dv * elem
        scores, probs, pv = heads * keys * FP32_BYTES, heads * keys * elem, heads * c.dv * FP32_BYTES
        dense_cann = c.abi in DENSE_CANN_ABIS
        state_heads = heads if (dense_cann or c.abi == DSL_MQS_ABI) else live
        state_scalars = FD_SOFTMAX_STATE_SCALARS if dense_cann else SOFTMAX_STATE_SCALARS
        state = state_heads * (c.dv + state_scalars) * FP32_BYTES
        out_heads = heads if dense_cann else live
        cube_keys = layout.key_tile if (c.abi == DSL_MQS_ABI and keys > 0) else keys
        # Arch35 direct UB->L1 kernels (FIA/FlashMLA/MQS/DSL) publish full s2BaseSize columns.
        prob_cols = layout.key_tile if (not layout.gm_workspace and keys > 0) else keys
        prob_publish = heads * prob_cols * elem
        d = _Demand(kv_reads=[] if self.planning is not None and step in {"kv_load", "pv_load"} else None)
        kv_base = 0
        if self.sequence_spans:
            _, _, q_seq_base, kv_base, _ = self.sequence_spans[seq]
        elif self.query_offsets:
            q_seq_base = self.query_offsets[seq]
        else:
            q_seq_base = sum(q for q, _ in c.sequences[:seq])
        q_base = (q_seq_base + query) * c.physical_heads + head_start
        atomic = layout.gm_workspace and not first
        ws_hit = 1.0 if layout.ws_l2_resident else 0.0
        if step == "sink_load":
            d.linear("sinks", "mte2_aiv", c.physical_heads * FP32_BYTES)
        elif step == "sink_init" and ((first and start == 0) or not partials):
            d.add("vector", heads * (c.dv + SOFTMAX_STATE_SCALARS))
        elif step == "q_load" and (first or layout.gm_workspace):
            d.linear("query", "mte2_aic", heads * qk_width, q_base * qk_width,
                     l2_hit_fraction=0.0 if first else ws_hit)
        elif step == "kv_load":
            _read_kv(d, c, seq, query, start, keys, qk_width, v_width, base_token=kv_base,
                     l2_hit_fraction=layout.kv_reuse_hit)
        elif step == "dequant" and c.mixed_quant:
            ori, _ = c.bank_counts(seq, query)
            fp8_keys = keys if c.abi == MQS_ABI else max(0, min(keys, ori - start))
            d.add("vector", c.dqk * (fp8_keys * FP8_DEQUANT_OPS_PER_VALUE
                                    + (keys - fp8_keys) * FP4_DEQUANT_OPS_PER_VALUE))
        elif step == "kv_to_l1" and c.mixed_quant:
            if c.abi == MQS_ABI:
                d.linear("kv_workspace", "mte3", keys * qk_width, l2_hit_fraction=ws_hit)
            else:
                pipe, alignment = layout.probability_output
                d.add(pipe, align_bytes(cube_keys * qk_width, alignment))
        elif step == "kv_refill" and c.abi == MQS_ABI:
            # mixed_quant_sparse_flash_mla_csa_block_cube.h:203 aligns refill rows up to 16.
            refill_rows = align_bytes(keys, MQS_REFILL_ROW_ALIGN)
            d.linear("kv_workspace", "mte2_aic", refill_rows * qk_width, l2_hit_fraction=ws_hit)
        elif step == "qk_operands":
            q_mult = (ceil_div(keys, ARCH22_MLA_NL1_SPLIT) if keys > 0 else 0) if layout.gm_workspace else 1
            d.add("mte1", heads * q_mult * qk_width + cube_keys * qk_width * l0b_factor)
        elif step == "qk":
            d.flops = 2 * heads * cube_keys * c.dqk
        elif step in {"score_store", "probability", "partial"}:
            tensor, (pipe, alignment), amount = {
                "score_store": ("mm1", layout.matrix_output, scores),
                "probability": ("vec1", layout.probability_output, prob_publish),
                "partial": ("mm2", layout.matrix_output, pv),
            }[step]
            if layout.gm_workspace:
                if step == "partial" and atomic:
                    d.atomic_add(tensor, pipe, amount, mm2_offset, l2_hit_fraction=ws_hit)
                else:
                    d.linear(tensor, pipe, amount, mm2_offset if step == "partial" else 0, l2_hit_fraction=ws_hit)
            else:
                if step == "score_store":
                    amount = fixpipe_rectangle_bytes(heads, cube_keys)
                d.add(pipe, align_bytes(amount, alignment))
        elif step == "score_read" and layout.gm_workspace:
            d.linear("mm1", "mte2_aiv", scores, l2_hit_fraction=ws_hit)
        elif step == "softmax":
            d.add("vector", heads * (SOFTMAX_OPS_PER_KEY * keys - 2))
        elif step == "pv_load":
            if layout.gm_workspace:
                d.linear("vec1", "mte2_aic", probs, l2_hit_fraction=ws_hit)
            if not layout.resident_latent:
                v_hit = 1.0 if layout.latent_l2_hit else layout.kv_reuse_hit
                _read_kv(d, c, seq, query, start, keys, qk_width, v_width, value=True, base_token=kv_base,
                         l2_hit_fraction=v_hit)
        elif step == "pv_operands":
            p_mult = (ceil_div(c.dv, ARCH22_MLA_NL1_SPLIT) if c.dv > 0 else 0) if layout.gm_workspace else 1
            pv_probs = heads * cube_keys * elem
            d.add("mte1", pv_probs * p_mult + cube_keys * v_width * l0b_factor)
        elif step in {"pv", "pv0", "pv1"}:
            columns = layout.pv_panel_width if step != "pv" else c.dv
            d.flops = 2 * heads * cube_keys * columns
        elif step in {"partial0", "partial1"}:
            pipe, alignment = layout.matrix_output
            d.add(pipe, align_bytes(fixpipe_rectangle_bytes(heads, layout.pv_panel_width), alignment))
        elif step in {"release0", "release1"} and not first:
            scalar = SOFTMAX_MERGE_SCALAR_OPS if step == "release0" else 0
            d.add("vector", heads * (WEIGHTED_MERGE_OPS_PER_VALUE * layout.pv_panel_width + scalar))
        elif step == "atomic_rescale" and atomic:
            d.add("vector", _merge_ops(c, heads, 1) - heads * c.dv)
        elif step == "atomic_write" and atomic:
            # Vec1 applies the exponent correction using INT32 atomic add on the MM2 bytes.
            d.atomic_add("mm2", "mte3", pv, mm2_offset, l2_hit_fraction=ws_hit)
        elif step == "output_read" and last:
            d.linear("mm2", "mte2_aiv", pv, mm2_offset, l2_hit_fraction=ws_hit)
        elif step == "output_update" and not layout.gm_workspace and not first:
            d.add("vector", _merge_ops(c, heads, 1))
        elif step == "split_write" and last and partials > 1:
            d.linear("vec2_split_state", "mte3", state)
        elif step == "merge_read":
            plan = AddressRunPlan((AddressRun(0, partials, state, align_bytes(state)),))
            d.transfer("vec2_split_state", "mte2_aiv", partials * state, plan)
        elif step == "merge":
            d.add("vector", _merge_ops(c, out_heads, max(0, partials - 1)))
        elif step == "normalize" and partials:
            d.add("vector", out_heads * (c.dv + (2 if c.return_lse else 0)))
        elif step == "store":
            d.linear("output", "mte3", out_heads * v_width, q_base * v_width)
            if c.return_lse:
                d.linear("lse", "mte3", out_heads * FP32_BYTES, q_base * FP32_BYTES)
        return d

    def _counts(self, seq: int, query: int) -> tuple[int, int]:
        c, width = self.contract, self.layout.key_tile
        if self.fia_host_plan is not None:
            tiles = ceil_div(c.sequences[seq][1], width)
            return tiles, self.fia_host_plan.row_splits.get((seq, query), 1 if tiles else 0)
        if c.abi == DSL_MQS_ABI:
            ori, cmp = c.bank_counts(seq, query)
            tiles = ceil_div(ori, width) + ceil_div(cmp, width)
            return tiles, 1 + len(self.row_cuts.get((seq, query), ())) if tiles else 0
        extent = sum(c.bank_counts(seq, query)) if c.mixed_quant else c.sequences[seq][1]
        tiles = ceil_div(extent, width)
        return tiles, min(tiles, c.splits)

    def _split_extent(self, seq: int, query: int, tiles: int, split: int) -> tuple[int, int]:
        if self.contract.abi == DSL_MQS_ABI:
            bounds = (0, *self.row_cuts.get((seq, query), ()), tiles)
            return bounds[split], bounds[split + 1] - bounds[split]
        return _split_tiles(tiles, self.contract.splits, split)

    def _query_runs(self, seq: int) -> Iterator[tuple[int, int]]:
        cuts = iter(sorted(query for batch, query in self.row_cuts if batch == seq))
        cut = next(cuts, None)
        for start, repeat in self.contract.query_runs(seq):
            end = start + repeat
            while cut is not None and cut < end:
                if start < cut:
                    yield start, cut - start
                yield cut, 1
                start = cut + 1
                cut = next(cuts, None)
            if end > start:
                yield start, end - start

    def _tile_extent(self, seq: int, query: int, tile: int) -> tuple[int, int]:
        c, width = self.contract, self.layout.key_tile
        if c.abi == DSL_MQS_ABI:
            ori, cmp = c.bank_counts(seq, query)
            ori_tiles = ceil_div(ori, width)
            if tile < ori_tiles:
                start = tile * width
                return start, min(width, ori - start)
            offset = (tile - ori_tiles) * width
            return ori + offset, min(width, cmp - offset)
        start = tile * width
        extent = sum(c.bank_counts(seq, query)) if c.mixed_quant else c.sequences[seq][1]
        return start, min(width, extent - start)

    def _tile_demand(self, step, seq, query, heads, live, index, offset, count, partials,
                     mm2_offset=0, head_start=0, *, l0b_factor: float = 1.0):
        start, keys = self._tile_extent(seq, query, offset + index)
        return self._demand(step, live, seq=seq, query=query, heads=heads, start=start, keys=keys,
                            first=index == 0, last=index + 1 == count, partials=partials,
                            mm2_offset=mm2_offset, head_start=head_start, l0b_factor=l0b_factor)

    def _finish_steps(self, partials):
        merge = ("merge_read", "merge") if partials > 1 else ()
        if self.contract.sinks and not partials:
            return ("sink_init", "normalize", "store")
        return (*merge, "normalize", "store")

    def _distribution(self):
        c = self.contract
        head_groups = ceil_div(c.physical_heads, self.layout.head_tile)
        prefix = tuple(accumulate((self._m_blocks_for_seq(seq) for seq in range(len(c.sequences))), initial=0))
        most_splits = max((n for _, _, n in self._group_ranges(head_groups, prefix)), default=0)
        lanes = min(len(self.owners), max(1, most_splits))
        return head_groups, prefix, lanes, len(self.owners) // lanes

    def _group_ranges(self, head_groups, prefix):
        if self._pack_m():
            for seq in range(len(self.contract.sequences)):
                if prefix[seq + 1] > prefix[seq]:
                    yield prefix[seq], prefix[seq + 1], self._counts(seq, 0)[1]
            return
        for seq in range(len(self.contract.sequences)):
            for query, repeat in self._query_runs(seq):
                if repeat:
                    start = prefix[seq] + query * head_groups
                    yield start, start + repeat * head_groups, self._counts(seq, query)[1]

    def _sink_owners(self, head_groups, prefix, lanes, groups_per_wave):
        # Each launched owner loads the shared head vector once, including owners of empty rows.
        slots = set()
        for start, end, partials in self._group_ranges(head_groups, prefix):
            for group in range(start, min(end, start + groups_per_wave)):
                base = group % groups_per_wave * lanes
                slots.update(range(base, base + max(1, min(partials, lanes))))
        return tuple(self.owners[i] for i in sorted(slots))

    def _row_residue_multipliers(self, step: str, repeat: int, row_stride: int) -> tuple[tuple[int, int], ...]:
        if repeat <= 1 or step not in {"q_load", "store"}:
            return ((0, repeat),)
        c = self.contract
        elem = int(dtype_bytes(self.dtype))
        if step == "q_load":
            period = NZ_BLOCK_BYTES // gcd(row_stride * c.dqk * elem, NZ_BLOCK_BYTES)
        else:
            p_out = NZ_BLOCK_BYTES // gcd(row_stride * c.dv * elem, NZ_BLOCK_BYTES)
            p_lse = (
                NZ_BLOCK_BYTES // gcd(row_stride * FP32_BYTES, NZ_BLOCK_BYTES)
                if c.return_lse else 1
            )
            period = lcm(p_out, p_lse)
        if period <= 1:
            return ((0, repeat),)
        return tuple(
            (offset, (repeat - 1 - offset) // period + 1)
            for offset in range(min(repeat, period))
        )

    def _interior_tile_period(self) -> int:
        """Exact sector-alignment period across full interior KV tiles."""
        c, layout = self.contract, self.layout
        elem = int(dtype_bytes(self.dtype))
        if layout.key_tile % c.page_size == 0:
            pt_stride = (layout.key_tile // c.page_size) * INDEX_BYTES
            pt_period = NZ_BLOCK_BYTES // gcd(pt_stride, NZ_BLOCK_BYTES)
        else:
            pt_period = 1
        kv_k_period = NZ_BLOCK_BYTES // gcd(layout.key_tile * c.dqk * elem, NZ_BLOCK_BYTES)
        kv_v_period = NZ_BLOCK_BYTES // gcd(layout.key_tile * c.dv * elem, NZ_BLOCK_BYTES)
        return lcm(pt_period, kv_k_period, kv_v_period)

    def _split_samples(self, count: int, can_fold_mid: bool, mid_period: int) -> tuple[tuple[int, int], ...]:
        if can_fold_mid and count > 2:
            mid_count = count - 2
            return (
                (0, 1),
                *(
                    (1 + r, (mid_count - 1 - r) // mid_period + 1)
                    for r in range(min(mid_count, mid_period))
                ),
                (count - 1, 1),
            )
        return tuple((i, 1) for i in range(count))

    def _fia_explicit_fd_reducers(self) -> tuple[tuple[tuple[int, int, int, int, int], ...], ...]:
        c = self.contract
        if c.abi != FIA_ABI or self.fia_host_plan is not None:
            return ()
        fd_tasks: list[tuple[int, int, int, int]] = []
        for seq in range(len(c.sequences)):
            m_blocks = self._m_blocks_for_seq(seq)
            if not m_blocks:
                continue
            _, partials = self._counts(seq, 0)
            if partials <= 1:
                continue
            for m_block in range(m_blocks):
                _, heads, _, _, _ = self._m_block_span(seq, m_block)
                fd_tasks.append((seq, m_block, heads, partials))
        if not fd_tasks:
            return ()
        aiv_per_aic = min(len(aivs) for _, aivs in self.owners)
        return _assign_fia_fd_reducers(
            tuple(fd_tasks), len(self.owners), aiv_per_aic,
            is_arch35=not self.layout.gm_workspace,
        )

    def _accumulate_fd_reducers(
        self, totals: dict[str, Work],
        reducers: tuple[tuple[tuple[int, int, int, int, int], ...], ...],
    ) -> None:
        c, layout = self.contract, self.layout
        for chunks in reducers:
            for seq, m_block, head_off, ch_heads, partials in chunks:
                m_start = m_block * layout.head_tile + head_off
                q_r, h_r = divmod(m_start, c.physical_heads)
                for step in self._finish_steps(partials):
                    work = self._demand(
                        step, ch_heads, seq=seq, query=q_r, heads=ch_heads,
                        partials=partials, head_start=h_r,
                    ).work(self.dtype)
                    totals[step] = totals.get(step, Work()) + work

    def _accumulate_block_totals(
        self, totals: dict[str, Work], seq: int, m_start: int, heads: int, live: int,
        repeat: int, row_stride: int, tiles: int, partials: int,
        phases: tuple[str, ...], can_fold_mid: bool, mid_period: int, *,
        skip_fd_finish: bool = False,
    ) -> None:
        c = self.contract
        base_query, _ = divmod(m_start, c.physical_heads)
        for split in range(partials):
            offset, count = self._split_extent(seq, base_query, tiles, split)
            for i, mult in self._split_samples(count, can_fold_mid, mid_period):
                for step in phases:
                    for r_off, r_mult in self._row_residue_multipliers(step, repeat, row_stride):
                        q_r, h_r = divmod(m_start + r_off * row_stride, c.physical_heads)
                        demand = self._tile_demand(
                            step, seq, q_r, heads, live, i, offset, count, partials, head_start=h_r,
                        )
                        totals[step] = totals.get(step, Work()) + demand.work(self.dtype).scaled(r_mult * mult)
        if skip_fd_finish and partials > 1:
            return
        for step in self._finish_steps(partials):
            for r_off, r_mult in self._row_residue_multipliers(step, repeat, row_stride):
                q_r, h_r = divmod(m_start + r_off * row_stride, c.physical_heads)
                work = self._demand(
                    step, live, seq=seq, query=q_r, heads=heads, partials=partials, head_start=h_r,
                ).work(self.dtype)
                totals[step] = totals.get(step, Work()) + work.scaled(r_mult)

    def _host_plan_totals(
        self, phases: tuple[str, ...], can_fold_mid: bool, mid_period: int,
    ) -> dict[str, Work]:
        c, host = self.contract, self.fia_host_plan
        assert host is not None
        totals: dict[str, Work] = {}
        for slices in host.core_slices:
            for seq, m_block, offset, count in slices:
                _, heads, live, query, head_start = self._m_block_span(seq, m_block)
                partials = host.row_splits.get((seq, m_block), 1)
                for i, mult in self._split_samples(count, can_fold_mid, mid_period):
                    for step in phases:
                        demand = self._tile_demand(
                            step, seq, query, heads, live, i, offset, count, partials, head_start=head_start,
                        )
                        totals[step] = totals.get(step, Work()) + demand.work(self.dtype).scaled(mult)
                if partials == 1:
                    for step in self._finish_steps(1):
                        work = self._demand(
                            step, live, seq=seq, query=query, heads=heads, partials=1, head_start=head_start,
                        ).work(self.dtype)
                        totals[step] = totals.get(step, Work()) + work
        for seq, (q_len, kv_len) in enumerate(c.sequences):
            if q_len > 0 and kv_len == 0:
                for m_block in range(self._m_blocks_for_seq(seq)):
                    _, heads, live, query, head_start = self._m_block_span(seq, m_block)
                    for step in self._finish_steps(0):
                        work = self._demand(
                            step, live, seq=seq, query=query, heads=heads, partials=0, head_start=head_start,
                        ).work(self.dtype)
                        totals[step] = totals.get(step, Work()) + work
        self._accumulate_fd_reducers(totals, host.fd_reducers)
        return totals

    def totals(self) -> dict[str, Work]:
        c, layout = self.contract, self.layout
        phases = layout.phases
        can_fold_mid = (
            not c.mixed_quant and c.abi != DSL_MQS_ABI and not c.block_table
            and (layout.key_tile % c.page_size == 0 or c.page_size % layout.key_tile == 0)
        )
        mid_period = self._interior_tile_period() if can_fold_mid else 1
        if self.fia_host_plan is not None:
            return self._host_plan_totals(phases, can_fold_mid, mid_period)
        totals: dict[str, Work] = {}
        head_groups, prefix, lanes, groups_per_wave = self._distribution()
        if c.sinks:
            sink_owners = self._sink_owners(head_groups, prefix, lanes, groups_per_wave)
            totals["sink_load"] = self._demand("sink_load", 0).work(self.dtype).scaled(len(sink_owners))
        explicit_fd = self._fia_explicit_fd_reducers()
        for seq in range(len(c.sequences)):
            if self._pack_m():
                q_len = c.sequences[seq][0]
                if not q_len:
                    continue
                tiles, partials = self._counts(seq, 0)
                total_m = q_len * c.physical_heads
                full_blocks, tail_m = divmod(total_m, layout.head_tile)
                if full_blocks:
                    self._accumulate_block_totals(
                        totals, seq, 0, layout.head_tile, layout.head_tile,
                        full_blocks, layout.head_tile, tiles, partials,
                        phases, can_fold_mid, mid_period, skip_fd_finish=bool(explicit_fd),
                    )
                if tail_m:
                    tail_start = full_blocks * layout.head_tile
                    tail_live = _live_rows_in_span(tail_start, total_m, c.logical_heads, c.physical_heads)
                    self._accumulate_block_totals(
                        totals, seq, tail_start, tail_m, tail_live,
                        1, layout.head_tile, tiles, partials,
                        phases, can_fold_mid, mid_period, skip_fd_finish=bool(explicit_fd),
                    )
                continue
            for query, repeat in self._query_runs(seq):
                if not repeat:
                    continue
                tiles, partials = self._counts(seq, query)
                for head_start in range(0, c.physical_heads, layout.head_tile):
                    heads = min(layout.head_tile, c.physical_heads - head_start)
                    live = min(heads, max(0, c.logical_heads - head_start))
                    m_start = query * c.physical_heads + head_start
                    self._accumulate_block_totals(
                        totals, seq, m_start, heads, live,
                        repeat, c.physical_heads, tiles, partials,
                        phases, can_fold_mid, mid_period,
                    )
        if explicit_fd:
            self._accumulate_fd_reducers(totals, explicit_fd)
        return totals

    def _stage(self, context, demand, pair, name, *, after=(), previous=(), previous_lags=(), metadata=None, latency=0.0, cold_caps=None, stage_caps=None):
        aic, aivs = pair
        work = demand.work(self.dtype)
        stages = []
        if demand.kv_reads and "l2" in context.resource_names:
            die = self.planning.geometry.rank or 0
            hbm = context.resource_names["hbm"]
            bus = HbmBusSpec(die, 1, 1, 0, row_r=hbm, ctr_r=hbm)
            resources = dict(work.resources)
            for i, (access, ledger) in enumerate(demand.kv_reads):
                label = "key" if self.contract.latent_alias else access.tensor_id
                tensor = f"{self.kv_tensor_id}:{label}"
                reuse = ReuseKey(die, tensor, self.kv_version)
                pattern = replace(access, tensor_id=tensor, version=self.kv_version, die=die, owner=aic, reuse_key=reuse)
                bound = context.bind_access(pattern, hbm_bus=bus, burst_rules=())
                stages.append(KernelProgram((Stage(f"kv{i}", program=AttentionMemoryRead(
                    pattern, bound, 0, hbm, self.planning,
                )),)))
                resources[access.pipe] -= ledger.aligned_transfer_bytes
                if ledger.l2_hit_bytes:
                    resources["l2"] -= ledger.l2_hit_bytes
                if ledger.dram_bus_transaction_bytes:
                    resources["hbm"] -= ledger.dram_bus_transaction_bytes
            work = Work(work.matrix, {role: value for role, value in resources.items() if value})
        if "l2" in work.resources and "l2" not in context.resource_names:
            work = Work(work.matrix, {role: val for role, val in work.resources.items() if role != "l2"})
        roles = (*work.resources, *(("cube",) if work.matrix else ()))
        placement = {}
        for role in roles:
            if role not in ("hbm", "l2"):
                placement.update(context.placement(role, aivs if role in AIV_ROLES else (aic,)))
        caps = dict(cold_caps) if (cold_caps and work.resources.get("hbm", 0) > 0 and name.startswith("kv_load")) else {}
        if stage_caps:
            caps.update(stage_caps)
        stage = Stage(name, context.bind(work), after=after, previous=previous, previous_lags=previous_lags,
                      placement=placement, coupled=("hbm" in work.resources or "l2" in work.resources), caps=caps,
                      metadata={**demand.metadata, **(metadata or {})})
        if not stages:
            if latency > 0.0 and (work.resources or work.matrix):
                return Stage(name, after=after, previous=previous, previous_lags=previous_lags,
                             metadata=stage.metadata,
                             program=KernelProgram((replace(stage, after=(), previous=(), previous_lags=()),
                                                    Stage("sync", after=(name,), latency=latency, keep_empty=True))))
            return replace(stage, latency=latency) if latency > 0.0 else stage
        if work.resources or work.matrix:
            stages.append(KernelProgram((replace(stage, after=(), previous=(), previous_lags=()),)))
        return Stage(name, after=after, previous=previous, previous_lags=previous_lags,
                     program=ParallelProgram(tuple(stages)), metadata=stage.metadata)

    def _panel_release(self, ctx, demand, aivs, heads):
        # Both subblocks acknowledge independently, even if their M half is empty.
        left = ceil_div(heads, 2)
        per_head = demand.resources.get("vector", 0) // heads
        stages = []
        for aiv, rows in zip(aivs, (left, heads - left)):
            amount = rows * per_head
            work = ctx.bind(Work(resources={"vector": amount})) if amount else {}
            placement = ctx.placement("vector", (aiv,)) if amount else {}
            stages.append(Stage(f"aiv{aiv}", work, placement=placement, keep_empty=True))
        return KernelProgram(tuple(stages))

    def _step_caps(self, ctx, step, demand, *, live, seq, query, heads, start, keys, first, last,
                   partials, mm2_offset, head_start, pages, scatter_open_bytes, l0b_factor, split_cold_caps):
        caps: dict[str, float] = {}
        if step == "kv_load" and scatter_open_bytes > 0 and pages > 0:
            hbm_res = ctx.resource_names.get("hbm")
            b_hbm = demand.work(self.dtype).resources.get("hbm", 0)
            if hbm_res and b_hbm > 0:
                base_cap = (split_cold_caps or {}).get(hbm_res, ctx.rates[hbm_res])
                extra_hbm = ARCH35_MLA_PLANE_COPY_CALLS * pages * scatter_open_bytes
                caps[hbm_res] = base_cap * (b_hbm / (b_hbm + extra_hbm))
        if step in {"qk_operands", "pv_operands"} and l0b_factor != 1.0:
            mte1_res = ctx.resource_names.get("mte1")
            b_mte1 = demand.resources.get("mte1", 0)
            eff_mte1 = self._demand(
                step, live, seq=seq, query=query, heads=heads, start=start, keys=keys,
                first=first, last=last, partials=partials,
                mm2_offset=mm2_offset, head_start=head_start, l0b_factor=l0b_factor,
            ).resources.get("mte1", 0)
            if mte1_res and b_mte1 > 0 and eff_mte1 > 0:
                owner_rate = ctx.rates[mte1_res] / max(1, ctx.resource_units.get("mte1", 1))
                caps[mte1_res] = owner_rate * (b_mte1 / eff_mte1)
        return caps or None

    def _ring_credits(self, owner: int) -> dict[str, list[tuple[str, str, int]]]:
        credits: dict[str, list[tuple[str, str, int]]] = {}
        for region, depth, producer, consumer in self.layout.rings:
            ring = RingBuffer(owner=owner, allocation=region, depth=depth)
            for k in range(depth + 1):
                generation = lower_generation(ring, k, producer, (consumer,))
            credit_index = next(g.index for g in ring.generations if g.last_use == generation.credit)
            credits.setdefault(producer, []).append((region, consumer, generation.index - credit_index))
        return credits

    def _cycle_stage_metadata(
        self, step: str, index: int, offset: int, clock: int, interval: int,
        mm2_slot: int, mm2_offset: int, aiv_owners: str, *, compact_names: bool,
    ) -> dict[str, Any]:
        if compact_names:
            return {}
        layout = self.layout
        metadata: dict[str, Any] = {"kv_tile": offset + index, "clock": clock}
        if layout.gm_workspace:
            metadata.update(task_record_idx=index % len(layout.clocks),
                            workspace_buf_idx=index % FIA_WORKSPACE_DEPTH)
            if step in {"partial", "atomic_rescale", "atomic_write", "output_read"}:
                metadata.update(bn2_idx_in_cur_core=interval, mm2_slot=mm2_slot, mm2_offset_bytes=mm2_offset)
        if layout.pv_panel_width and step in {"pv0", "pv1", "partial0", "partial1", "release0", "release1"}:
            panel = int(step[-1])
            metadata.update(mm2_slot=panel, mm2_generation=index, aiv_owners=aiv_owners,
                            column_start=panel * layout.pv_panel_width, columns=layout.pv_panel_width)
        return metadata

    def _split_program(self, ctx, seq, query, heads, live, pair, offset, count, partials, interval, head_start,
                       *, active_owners: int | None = None):
        layout = self.layout
        aivs = pair[1]
        aiv_owners = ",".join(map(str, aivs)) if layout.pv_panel_width else ""
        aiv_limit = 1 if (layout.gm_workspace and heads <= SINGLE_AIV_HEAD_LIMIT) else heads
        pair = (pair[0], pair[1][:aiv_limit])
        mm2_slot = interval % FIA_WORKSPACE_DEPTH
        mm2_offset = mm2_slot * align_bytes(layout.head_tile * self.contract.dv * FP32_BYTES)
        phases = layout.phases
        delays = {step: delay for delay, steps in enumerate(layout.clocks) for step in steps}
        period = lcm(len(layout.clocks), *(depth for _, depth, _, _ in layout.rings))
        credits = self._ring_credits(pair[0])

        def name(step, index):
            return f"{step}:{index % period}"

        eff_owners = active_owners if active_owners is not None else len(self.owners)
        split_cold_caps = _mla_kv_cold_caps(ctx, eff_owners, self.contract, layout, self.dtype)
        l0b_factor = p02_l0b_load_factor(ctx) if (self.contract.abi == FIA_ABI and not self.contract.mixed_quant) else 1.0
        fix_setup_ms = p03_fixpipe_setup_ms(ctx)
        page_setup_ms = p01_paged_copy_setup_ms(ctx, is_arch35=not layout.gm_workspace)
        scatter_open_bytes = (
            p01_hbm_scatter_row_open_bytes(ctx)
            if (self.contract.abi == FIA_ABI and not layout.gm_workspace
                and not self.contract.mixed_quant and self.contract.kv_layout == "PA_BLK_BSND")
            else 0
        )
        base_sync_by_step = {
            step: mla_step_crosscore_ms(ctx, step, gm_workspace=layout.gm_workspace, abi=self.contract.abi)
            for step in phases
        }

        def _build_cycle(clock: int, *, compact_names: bool) -> tuple[Stage, ...]:
            stages: list[Stage] = []
            for position, step in enumerate(phases):
                index = clock - delays[step]
                if not 0 <= index < count:
                    continue
                deps, prev, prev_lags = [], [], []

                def depend(other: str, tile: int) -> None:
                    if 0 <= tile < count:
                        delta = clock - (tile + delays[other])
                        if compact_names:
                            if delta == 0:
                                deps.append(other)
                            elif delta == 1:
                                prev.append(other)
                            elif delta > 1:
                                prev_lags.append((other, delta))
                        else:
                            (deps if delta == 0 else prev).append(name(other, tile))

                if step in {"release0", "release1"}:
                    depend(f"partial{step[-1]}", index)
                    depend("release1" if step == "release0" else "release0",
                           index - 1 if step == "release0" else index)
                elif layout.gm_workspace and layout.head_tile > SINGLE_AIV_HEAD_LIMIT:
                    gm_dep_map = {
                        "qk": ("kv_load",), "score_store": ("qk_operands", "qk"),
                        "pv": ("pv_load",), "atomic_rescale": ("probability",),
                        "partial": ("pv_operands", "pv"),
                    }
                    for dep_step in gm_dep_map.get(step, (phases[position - 1],) if position else ()):
                        depend(dep_step, index)
                elif position:
                    depend(phases[position - 1], index)
                # Each per-core pipe queue executes successive KV tiles in FIFO order;
                # emitting the explicit lag-1 self-edge keeps literal (period-suffixed)
                # and compacted (unprefixed) pipelines DAG-isomorphic.
                depend(step, index - 1)
                if layout.gm_workspace:
                    for peer, delta in FIA_ISSUE_ORDER.get(step, ()):
                        depend(peer, index + delta)
                if step == "atomic_rescale":
                    depend("partial", index - 1)
                if step == "pv0":
                    depend("partial1", index - 1)
                metadata = self._cycle_stage_metadata(
                    step, index, offset, clock, interval, mm2_slot, mm2_offset, aiv_owners,
                    compact_names=compact_names,
                )
                for region, consumer, depth in credits.get(step, ()):
                    depend(consumer, index - depth)
                    if not compact_names:
                        metadata[f"{region}_slot"] = index % depth
                        metadata[f"{region}_generation"] = index // depth
                start, keys = self._tile_extent(seq, query, offset + index)
                demand = self._demand(step, live, seq=seq, query=query, heads=heads, start=start, keys=keys,
                                      first=index == 0, last=index + 1 == count, partials=partials,
                                      mm2_offset=mm2_offset, head_start=head_start)
                edges = dict(
                    after=tuple(dict.fromkeys(deps)),
                    previous=tuple(dict.fromkeys(prev)),
                    previous_lags=tuple(dict.fromkeys(prev_lags)),
                )
                st_name = step if compact_names else name(step, index)
                if step in {"release0", "release1"}:
                    stages.append(Stage(st_name, program=self._panel_release(ctx, demand, aivs, heads),
                                        metadata=metadata, **edges))
                else:
                    pages = (ceil_div(start + keys, self.contract.page_size) - start // self.contract.page_size) if keys else 0
                    step_caps = self._step_caps(
                        ctx, step, demand, live=live, seq=seq, query=query, heads=heads,
                        start=start, keys=keys, first=index == 0, last=index + 1 == count,
                        partials=partials, mm2_offset=mm2_offset, head_start=head_start,
                        pages=pages, scatter_open_bytes=scatter_open_bytes,
                        l0b_factor=l0b_factor, split_cold_caps=split_cold_caps,
                    )
                    sync_lat = _mla_step_latency_ms(
                        step, self.contract, layout, keys, pages,
                        base_sync_by_step[step], fix_setup_ms, page_setup_ms,
                    )
                    stages.append(self._stage(
                        ctx, demand, pair, st_name, metadata=metadata, latency=sync_lat,
                        cold_caps=split_cold_caps, stage_caps=step_caps, **edges,
                    ))
            by_name = {stage.name: stage for stage in stages}
            order = TopologicalSorter({stage.name: stage.after for stage in stages})
            return tuple(by_name[key] for key in order.static_order())

        slots = len(layout.clocks)
        can_compact = (
            count > MLA_COMPACT_THRESHOLD
            and eff_owners == 1
            and layout.key_tile >= MLA_COMPACT_MIN_KEY_TILE
            and not self.contract.mixed_quant
            and not self.contract.block_table
            and self.planning is None
            and self.contract.kv_layout in {"PA_BLK_BSND", "BNSD"}
            and (layout.key_tile % self.contract.page_size == 0 or self.contract.page_size % layout.key_tile == 0)
            and self._interior_tile_period() == 1
        )
        if can_compact:
            steady_head = max(
                slots + 1,
                *(delays[prod] + depth for _, depth, prod, _ in layout.rings),
            )
            return compact_periodic_pipeline(
                ctx, count + slots - 1, slots,
                lambda clk: _build_cycle(clk, compact_names=True),
                head=steady_head, tail=slots + 1, threshold=MLA_COMPACT_THRESHOLD,
            )
        return PipelineProgram(count + slots - 1, slots, lambda clk: _build_cycle(clk, compact_names=False))

    def _lower_fd_reducers(
        self, context: RecipeContext,
        reducers: tuple[tuple[tuple[int, int, int, int, int], ...], ...],
        used_cores: int,
    ) -> ParallelProgram:
        c, layout = self.contract, self.layout
        active_owners = self.owners[:max(1, min(len(self.owners), used_cores))]
        all_aivs = tuple(aiv for _, aivs in active_owners for aiv in aivs)
        base_aic = active_owners[0][0]
        branches: list[KernelProgram] = []
        for aiv_idx, chunks in enumerate(reducers):
            phys_aiv = all_aivs[aiv_idx % len(all_aivs)]
            aiv_pair = (base_aic, (phys_aiv,))
            aiv_stages: list[Stage] = []
            for ch_idx, (seq, m_block, head_off, ch_heads, partials) in enumerate(chunks):
                m_start = m_block * layout.head_tile + head_off
                q_r, h_r = divmod(m_start, c.physical_heads)
                for step in self._finish_steps(partials):
                    dep = (aiv_stages[-1].name,) if aiv_stages else ()
                    st_name = step if len(chunks) == 1 else f"{step}_{ch_idx}"
                    aiv_stages.append(self._stage(
                        context, self._demand(step, ch_heads, seq=seq, query=q_r, heads=ch_heads,
                                              partials=partials, head_start=h_r),
                        aiv_pair, st_name, after=dep, metadata={"nonempty_splits": partials},
                    ))
            branches.append(KernelProgram(tuple(aiv_stages)))
        return ParallelProgram(tuple(branches))

    def _wrap_global_fd_barrier(
        self, context: RecipeContext, producer: Program,
        reducers: tuple[tuple[tuple[int, int, int, int, int], ...], ...],
        used_cores: int,
    ) -> KernelProgram:
        stages = [Stage("flash_attention", program=producer)]
        if (syncall_ms := p13_syncall_ms(context)) > 0.0:
            stages.append(Stage("fd_syncall", after=(stages[-1].name,), latency=syncall_ms, keep_empty=True))
        stages.append(Stage(
            "fd_reduce", after=(stages[-1].name,),
            program=self._lower_fd_reducers(context, reducers, used_cores),
        ))
        return KernelProgram(tuple(stages))

    def _lower_fia_host_plan(self, context: RecipeContext) -> Program:
        c, layout, host = self.contract, self.layout, self.fia_host_plan
        assert host is not None
        active_cores = min(len(self.owners), len(host.core_slices))
        core_programs: list[KernelProgram] = []
        for core_idx, slices in enumerate(host.core_slices):
            pair = self.owners[core_idx % len(self.owners)]
            c_stages: list[Stage] = []
            for slice_idx, (seq, m_block, offset, count) in enumerate(slices):
                _, heads, live, query, head_start = self._m_block_span(seq, m_block)
                partials = host.row_splits.get((seq, m_block), 1)
                dep = (c_stages[-1].name,) if c_stages else ()
                prog = self._split_program(
                    context, seq, query, heads, live, pair, offset, count, partials,
                    slice_idx, head_start, active_owners=active_cores,
                )
                c_stages.append(Stage(f"slice{slice_idx}", after=dep, program=prog, metadata={"split": slice_idx}))
                if partials == 1:
                    f_pair = (pair[0], pair[1][:1]) if (layout.gm_workspace and heads <= SINGLE_AIV_HEAD_LIMIT) else pair
                    for step in self._finish_steps(1):
                        c_stages.append(self._stage(
                            context, self._demand(step, live, seq=seq, query=query, heads=heads, partials=1,
                                                  head_start=head_start),
                            f_pair, f"{step}_{slice_idx}", after=(c_stages[-1].name,),
                            metadata={"nonempty_splits": 1},
                        ))
            if c_stages:
                core_programs.append(KernelProgram(tuple(c_stages)))
        empty_stages: list[Stage] = []
        for seq, (q_len, kv_len) in enumerate(c.sequences):
            if q_len > 0 and kv_len == 0:
                for m_block in range(self._m_blocks_for_seq(seq)):
                    _, heads, live, query, head_start = self._m_block_span(seq, m_block)
                    pair = self.owners[0]
                    f_pair = (pair[0], pair[1][:1]) if (layout.gm_workspace and heads <= SINGLE_AIV_HEAD_LIMIT) else pair
                    for step in self._finish_steps(0):
                        dep = (empty_stages[-1].name,) if empty_stages else ()
                        empty_stages.append(self._stage(
                            context, self._demand(step, live, seq=seq, query=query, heads=heads, partials=0,
                                                  head_start=head_start),
                            f_pair, f"{step}_empty_{seq}_{m_block}", after=dep,
                            metadata={"nonempty_splits": 0},
                        ))
        if empty_stages:
            core_programs.append(KernelProgram(tuple(empty_stages)))
        if not core_programs:
            return KernelProgram(())
        producer: Program = core_programs[0] if len(core_programs) == 1 else ParallelProgram(tuple(core_programs))
        if host.fd_reducers:
            return self._wrap_global_fd_barrier(context, producer, host.fd_reducers, host.used_cores)
        return producer

    def __call__(self, context: RecipeContext) -> Program:
        c, layout = self.contract, self.layout
        if self.fia_host_plan is not None:
            return self._lower_fia_host_plan(context)
        head_groups, prefix, lanes, groups_per_wave = self._distribution()
        if not prefix[-1]:
            return KernelProgram(())
        explicit_fd = self._fia_explicit_fd_reducers()

        intervals = []
        if layout.gm_workspace:
            # Collapse adjacent sequences with equal split counts for owner-local ordinals.
            for start, end, partials in self._group_ranges(head_groups, prefix):
                if partials and end > start:
                    if intervals and intervals[-1][1:] == (start, partials):
                        start, _, _ = intervals.pop()
                    intervals.append((start, end, partials))

        def owner_interval(group_index, local):
            slot = group_index % groups_per_wave
            prior = 0
            for start, end, partials in intervals:
                if start >= group_index:
                    break
                end = min(end, group_index)
                groups = ceil_div(end - slot, groups_per_wave) - ceil_div(start - slot, groups_per_wave)
                prior += groups * max(0, ceil_div(partials - local, lanes))
            return prior

        def group(group_index, owner_start):
            seq = bisect_right(prefix, group_index) - 1
            local_block = group_index - prefix[seq]
            if self._pack_m():
                _, heads, live, query, head_start = self._m_block_span(seq, local_block)
            else:
                query, head_group = divmod(local_block, head_groups)
                head_start = head_group * layout.head_tile
                heads = min(layout.head_tile, c.physical_heads - head_start)
                live = min(heads, max(0, c.logical_heads - head_start))
            tiles, partials = self._counts(seq, query)

            active_groups = min(groups_per_wave, prefix[-1])
            wave_owners = min(len(self.owners), active_groups * min(lanes, max(1, partials)))

            def wave(index):
                for local in range(min(lanes, partials - index * lanes)):
                    split = index * lanes + local
                    offset, count = self._split_extent(seq, query, tiles, split)
                    program = self._split_program(
                        context, seq, query, heads, live, self.owners[owner_start + local], offset, count, partials,
                        owner_interval(group_index, local) + index, head_start, active_owners=wave_owners,
                    )
                    yield Stage(f"split{local}", program=program, metadata={"split": split})

            stages = [Stage("splits", program=PipelineProgram(ceil_div(partials, lanes), 1, wave))]
            if explicit_fd and partials > 1:
                if prefix[-1] > 1:
                    return KernelProgram(tuple(stages))
                if (syncall_ms := p13_syncall_ms(context)) > 0.0:
                    stages.append(Stage("fd_syncall", after=(stages[-1].name,), latency=syncall_ms, keep_empty=True))
                stages.append(Stage(
                    "fd_reduce", after=(stages[-1].name,),
                    program=self._lower_fd_reducers(context, explicit_fd, len(self.owners)),
                ))
                return KernelProgram(tuple(stages))
            if partials > 1 and (syncall_ms := p13_syncall_ms(context)) > 0.0:
                stages.append(Stage("fd_syncall", after=(stages[-1].name,), latency=syncall_ms, keep_empty=True))
            finish_pair = self.owners[owner_start]
            if layout.gm_workspace and heads <= SINGLE_AIV_HEAD_LIMIT:
                finish_pair = (finish_pair[0], finish_pair[1][:1])
            for step in self._finish_steps(partials):
                stages.append(self._stage(
                    context, self._demand(step, live, seq=seq, query=query, heads=heads, partials=partials,
                                          head_start=head_start), finish_pair,
                    step, after=(stages[-1].name,), metadata={"nonempty_splits": partials},
                ))
            return KernelProgram(tuple(stages))

        def launch(index):
            start = index * groups_per_wave
            return tuple(Stage(f"query{local}", program=group(start + local, local * lanes))
                         for local in range(min(groups_per_wave, prefix[-1] - start)))

        pipeline = PipelineProgram(ceil_div(prefix[-1], groups_per_wave), 1, launch)
        if explicit_fd and prefix[-1] > 1:
            return self._wrap_global_fd_barrier(context, pipeline, explicit_fd, len(self.owners))
        if not c.sinks:
            return pipeline
        loads = tuple(self._stage(context, self._demand("sink_load", 0), pair, f"sink_load{pair[0]}")
                      for pair in self._sink_owners(head_groups, prefix, lanes, groups_per_wave))
        return KernelProgram((*loads, Stage("attention", program=pipeline, after=tuple(s.name for s in loads))))

def owner_pairs(ctx: OperatorPlanningContext, abi: str) -> tuple[tuple[int, tuple[int, ...]], ...]:
    owners = ctx.stream_owners
    aics = (owners.aic or owners.pool_aic) if owners else ()
    aivs = (owners.aiv or owners.pool_aiv) if owners else ()
    active_aic = frozenset((aics or range(ctx.effective_aic_cores))[:ctx.effective_aic_cores])
    active_aiv = frozenset((aivs or range(ctx.effective_aiv_cores))[:ctx.effective_aiv_cores])
    pairs = tuple((aic, active) for aic, siblings in ctx.datapath.aic_aiv_pairs
                  if aic in active_aic and (active := tuple(aiv for aiv in siblings if aiv in active_aiv)))
    if not pairs:
        raise capability_gap("paged_mla_missing_owner_pair", abi, ctx.compute_arch.arch_id)
    return pairs

def make_plan(op: OpSpec, ctx: OperatorPlanningContext, c: PagedMlaContract,
              layout: PipelineLayout, sources: tuple[SourceRef, ...], *, pairs=None,
              row_cuts: RowCuts | None = None, fia_host_plan: _FiaHostPlan | None = None,
              **metadata) -> OperatorPlan:
    dtype = op.dtype or "bf16"
    if dtype not in {"bf16", "fp16"} or (c.mixed_quant and dtype != "bf16"):
        raise capability_gap("paged_mla_unsupported_dtype", c.abi, ctx.compute_arch.arch_id)
    if pairs is None:
        pairs = owner_pairs(ctx, c.abi)
    spans = planning_sequence_spans(ctx, c.kv_layout)
    raw = dict(op.metadata)
    tensor_id = raw.get("kv_tensor_id", "")
    version = dict(ctx.allocation_versions).get(tensor_id, raw.get("kv_version", 0))
    planning = ctx if spans and ctx.admission_memory_view is not None and c.block_table and tensor_id else None
    query_offsets = tuple(accumulate((q for q, _ in c.sequences), initial=0))
    builder = _PagedMlaProgramBuilder(
        c, layout, dtype, pairs, row_cuts or {}, spans, planning, tensor_id, version, query_offsets,
        fia_host_plan,
    )
    phase_totals = builder.totals()
    if "l2" not in ctx.geometry.rates:
        phase_totals = {k: Work(w.matrix, {r: v for r, v in w.resources.items() if r != "l2"}) for k, w in phase_totals.items()}
    total = sum(phase_totals.values(), Work())
    phases = []
    for step in ("sink_load", *layout.phases, "merge_read", "merge", "normalize", "store"):
        if (work := phase_totals.get(step)) is None:
            continue
        items = (tuple(WorkItem("cube", "flop", a, p) for p, a in work.matrix.items())
                 + tuple(WorkItem(r, "op" if r == "vector" else "bytes", a) for r, a in work.resources.items()))
        preds = {"release0": ("partial0",), "release1": ("partial1", "release0")}.get(
            step, (phases[-1].phase_id,) if phases else ())
        phases.append(PipelinePhase(step, preds, items))
    from .common import ABI_PUBLIC_IDENTITY
    public_name, native_entry, native_ns = ABI_PUBLIC_IDENTITY.get(c.abi, (c.abi, "", "aclnn"))
    return OperatorPlan(
        total, builder, metadata={
            "implementation": c.abi, "arch_id": ctx.compute_arch.arch_id, "governed_family": FAMILY,
            "native_kind": public_name, "native_entry": native_entry, "native_namespace": native_ns,
            "s2_base_size": c.s2_base_size, "key_tile": layout.key_tile, "dqk": c.dqk, "dv": c.dv,
            "useful_matrix_flops": c.useful_matrix_flops, "issued_matrix_flops": int(sum(total.matrix.values())),
            "latent_record_bytes": 0 if c.mixed_quant else latent_record_bytes(c, dtype=dtype),
            "scale_encoding": c.scale_encoding if c.mixed_quant else "", **metadata,
            **sequence_layout_metadata(ctx, c.kv_layout), **({"sinks": c.sinks} if c.sinks else {}),
        },
        pipeline_spec=FusedPipelineSpec(FAMILY, c.abi, tuple(phases), sources),
    )

def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(ctx.occurrence, FAMILY, spec.implementation, (op.op_id,),
                                   (), spec.phases, spec.sources, spec.repeats)
