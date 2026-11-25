"""White-box CANN host-tiling & FlashDecode core splitter for FIA GQA and MLA.

[Role] Reproduce CANN ops-transformer split_core / split_core_v2, Arch22/Arch35
  FIA GQA tile selection (including the Arch35 CheckS1OutSplit L2 predicate),
  and FIA MLA M/S2 tile selection without empirical fitting.
[Invariants] Every integer tile size, cost weight, and core-partition rule comes
  from pinned CANN op_host sources. Output core ranges partition all non-empty
  basic (b, kv_head, m_rect, s2_tile) blocks without dropping or duplicating work.
[Timeouts & Locks] Pure integer O(cores * rows) arithmetic; no I/O or locks.
[Side-Effect Policy] Pure immutable tiling descriptors only.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import ceil, gcd, isqrt
import sys
from typing import Literal

from mindie_sim.schema.datapath import SourceRef

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/common/op_host/split_core.cpp",
        "356fd86453726d363ecdf9db2cd8ff3ee37dfc1daadfd60feaa51bb2fdba20c8",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/common/op_host/split_core_v2.cpp",
        "73fc473f37677d43f718ea68a3ff8d4c248f4d7de6808ee68363351759ca8c1c",
    ),
)

COST_M_WEIGHT = 6
COST_M_GRANULARITY = 16
COST_S2_WEIGHT = 10
COST_S2_GRANULARITY = 64
FA_TOLERANCE_RATIO = 2
MIN_CORE_OFFSET_NUM = 1
MIN_CORE_OFFSET_DEN = 4
MIN_CORE_ROUND_NUM = 1
MIN_CORE_ROUND_DEN = 2

ARCH22_GQA_DECODE_MAX_S1 = 16
ARCH22_GQA_GROUP_STEP = 5
ARCH22_GQA_SINNER_TABLE = (8192, 4096, 2048)
ARCH22_GQA_PREFILL_SINNER = 512
ARCH22_GQA_PREFILL_M_BASE = 512
ARCH22_SPLITFUSE_M_BASE = 128
ARCH22_GQA_M_BASE_THRESHOLDS = (
    (512, 512),
    (1024, 256),
    (2048, 128),
    (4096, 64),
)
ARCH22_GQA_M_BASE_FALLBACK = 32
ARCH22_ALIGN_ELEMS = 16

ARCH35_CV_RATIO = 2
ARCH35_SOUTER_64 = 64
ARCH35_SOUTER_32 = 32
ARCH35_SINNER_128 = 128
ARCH35_SINNER_256 = 256
ARCH35_GQA_DIM_128 = 128
ARCH35_L2_PRESSURE_FACTOR = 2
ARCH35_BF16_BYTES = 2
ARCH35_PFA_MERGE_QG_LIMIT = 64
ARCH35_FD_TOLERANCE_BLOCKS = 9
ARCH35_FD_LEAST_BLOCKS = 0

ARCH22_MLA_M_BASE_DEFAULT = 256
ARCH22_MLA_M_BASE_MIN = 128
ARCH22_MLA_S2_BASE = 512
ARCH35_MLA_M_BASE = 64
ARCH35_MLA_S2_BASE = 128


@dataclass(frozen=True, slots=True)
class CorePartition:
    """Resolved CANN host-tiling decision for one FIA GQA or MLA call."""

    m_tile: int
    s2_tile: int
    used_cores: int
    fd_splits_per_row: int
    fd_enabled: bool
    s1_out_split: bool
    pfa_merge: bool
    max_core_cost: int
    total_blocks: int
    max_blocks_per_core: int = 0


def block_cost(m_rows: int, s2_cols: int) -> int:
    """CANN split_core CalcCost(m, n) = 6*ceil(m/16) + 10*ceil(n/64)."""
    return (
        COST_M_WEIGHT * ceil(m_rows / COST_M_GRANULARITY)
        + COST_S2_WEIGHT * ceil(s2_cols / COST_S2_GRANULARITY)
    )


def _align_up(value: int, align: int) -> int:
    return ((value + align - 1) // align) * align


def _min_search_cores(total_blocks: int, max_cores: int) -> int:
    # Exact integer equivalent of floor(sqrt(total_blocks + 0.25) + 0.5)
    root = isqrt(total_blocks * MIN_CORE_OFFSET_DEN + MIN_CORE_OFFSET_NUM)
    candidate = (root + MIN_CORE_ROUND_NUM) // MIN_CORE_ROUND_DEN
    return max(1, min(candidate, max_cores))


@dataclass(frozen=True, slots=True)
class _RowSpec:
    tiles: int
    normal_cost: int
    tail_cost: int

    @property
    def total_cost(self) -> int:
        if self.tiles <= 0:
            return 0
        return (self.tiles - 1) * self.normal_cost + self.tail_cost

    def tile_cost(self, idx: int) -> int:
        return self.tail_cost if idx == self.tiles - 1 else self.normal_cost


def _eval_split_plan(
    rows: tuple[_RowSpec, ...],
    core_num: int,
    *,
    stream_k: bool,
    cost_limit: int | None = None,
) -> tuple[int, int, int]:
    """Replay CANN CalcSplitPlan; return (used_cores, max_cost, max_splits_on_any_row)."""
    total_cost = sum(r.total_cost for r in rows)
    if core_num <= 0 or total_cost <= 0:
        return 1, 0, 1
    max_row_cost = max((r.total_cost for r in rows), default=0)
    row_idx = 0
    tile_idx = 0
    rem_row_tiles = rows[0].tiles
    rem_row_cost = rows[0].total_cost
    unassigned = total_cost
    used = 0
    max_cost = 0
    splits_per_row = [0] * len(rows)

    for c_idx in range(core_num):
        if cost_limit is not None and max_cost > cost_limit:
            return used, max_cost, max(splits_per_row, default=1)
        if row_idx >= len(rows) or unassigned <= 0:
            break
        avg = unassigned // (core_num - c_idx)
        limit = avg if stream_k else max(avg, max_row_cost)
        cur_cost = 0
        cur_blocks = 0

        # 1. Assign whole rows while within half of the row's final block cost.
        while row_idx < len(rows) and tile_idx == 0:
            r_cost = rem_row_cost
            row_tol = rows[row_idx].tail_cost // FA_TOLERANCE_RATIO
            if limit + row_tol >= cur_cost + r_cost:
                cur_cost += r_cost
                cur_blocks += rem_row_tiles
                splits_per_row[row_idx] += 1
                row_idx += 1
                if row_idx < len(rows):
                    tile_idx = 0
                    rem_row_tiles = rows[row_idx].tiles
                    rem_row_cost = rows[row_idx].total_cost
            else:
                break

        # 2. Assign individual S2 tiles when streamK / FlashDecode is active.
        if stream_k and row_idx < len(rows):
            took_cur_row = False
            while row_idx < len(rows):
                t_cost = rows[row_idx].tile_cost(tile_idx)
                if limit + t_cost // FA_TOLERANCE_RATIO >= cur_cost + t_cost:
                    cur_cost += t_cost
                    cur_blocks += 1
                    tile_idx += 1
                    rem_row_tiles -= 1
                    rem_row_cost -= t_cost
                    if rem_row_tiles == 0:
                        splits_per_row[row_idx] += 1
                        took_cur_row = False
                        row_idx += 1
                        if row_idx < len(rows):
                            tile_idx = 0
                            rem_row_tiles = rows[row_idx].tiles
                            rem_row_cost = rows[row_idx].total_cost
                    else:
                        took_cur_row = True
                else:
                    break
            if took_cur_row and row_idx < len(rows):
                splits_per_row[row_idx] += 1

        # 3. Force at least one block (or row if no streamK) to guarantee progress.
        if cur_blocks == 0 and row_idx < len(rows):
            if stream_k:
                t_cost = rows[row_idx].tile_cost(tile_idx)
                cur_cost += t_cost
                cur_blocks += 1
                tile_idx += 1
                rem_row_tiles -= 1
                rem_row_cost -= t_cost
                if rem_row_tiles == 0:
                    splits_per_row[row_idx] += 1
                    row_idx += 1
                    if row_idx < len(rows):
                        tile_idx = 0
                        rem_row_tiles = rows[row_idx].tiles
                        rem_row_cost = rows[row_idx].total_cost
                else:
                    splits_per_row[row_idx] += 1
            else:
                cur_cost += rem_row_cost
                cur_blocks += rem_row_tiles
                splits_per_row[row_idx] += 1
                row_idx += 1
                if row_idx < len(rows):
                    tile_idx = 0
                    rem_row_tiles = rows[row_idx].tiles
                    rem_row_cost = rows[row_idx].total_cost

        used += 1
        max_cost = max(max_cost, cur_cost)
        unassigned -= cur_cost

    # Any trailing unassigned tiles fall onto the final core.
    if row_idx < len(rows) and unassigned > 0:
        max_cost = max(max_cost, unassigned)
    max_splits = max(1, max(splits_per_row, default=1))
    return max(1, used), max_cost, max_splits


def _search_split_core(
    rows: tuple[_RowSpec, ...],
    aic_count: int,
    *,
    stream_k: bool,
    fd_tolerance_blocks: int,
    fd_least_blocks: int,
    full_block_cost: int,
) -> tuple[int, int, int, bool]:
    total_blocks = sum(r.tiles for r in rows)
    if total_blocks <= 0:
        return 1, 0, 1, False
    max_core = min(aic_count, total_blocks)
    min_core = _min_search_cores(total_blocks, max_core)
    best_used = 1
    best_cost = sys.maxsize
    best_splits = 1
    for cores in range(min_core, max_core + 1):
        used, max_cost, max_splits = _eval_split_plan(
            rows, cores, stream_k=stream_k, cost_limit=best_cost,
        )
        if max_cost < best_cost:
            best_used, best_cost, best_splits = used, max_cost, max_splits
    fd_enabled = stream_k and best_splits > 1
    if fd_enabled and (fd_tolerance_blocks > 0 or fd_least_blocks > 0):
        no_used, no_cost, _ = _eval_split_plan(rows, best_used, stream_k=False)
        keep_fd = (
            no_cost > fd_least_blocks * full_block_cost
            and (no_cost - fd_tolerance_blocks * full_block_cost) > best_cost
        )
        if not keep_fd:
            return no_used, no_cost, 1, False
    return best_used, best_cost, best_splits, fd_enabled


def _build_rows(
    sequences: tuple[tuple[int, int], ...],
    num_kv_heads: int,
    group_size: int,
    m_tile: int,
    s2_tile: int,
    *,
    causal: bool = False,
    is_s1g: bool = True,
) -> tuple[_RowSpec, ...]:
    rows: list[_RowSpec] = []
    for q_len, kv_len in sequences:
        if q_len <= 0 or kv_len <= 0:
            continue
        total_m = q_len * group_size
        m_rects = ceil(total_m / m_tile)
        full_tiles = ceil(kv_len / s2_tile)
        tail_s2 = kv_len - (full_tiles - 1) * s2_tile
        for _ in range(num_kv_heads):
            for r_idx in range(m_rects):
                m_first = r_idx * m_tile
                m_rows = min(m_tile, total_m - m_first)
                if causal:
                    # split_core_v2.cpp:125-165 & split_core.cpp:128-166:
                    # right-aligned causal upper bound per M-tile (isS1G vs BNSD).
                    m_last = m_first + m_rows - 1
                    if is_s1g:
                        s1_last = m_last // group_size
                    elif m_first // q_len == m_last // q_len:
                        s1_last = m_last % q_len
                    else:
                        s1_last = q_len
                    s2_last = s1_last + (kv_len - q_len)
                    if s2_last < 0:
                        continue
                    tiles = min(s2_last, kv_len - 1) // s2_tile + 1
                else:
                    tiles = full_tiles
                row_tail_s2 = tail_s2 if tiles == full_tiles else s2_tile
                rows.append(_RowSpec(
                    tiles=tiles,
                    normal_cost=block_cost(m_rows, s2_tile),
                    tail_cost=block_cost(m_rows, row_tail_s2),
                ))
    return tuple(rows)


def resolve_arch22_fia_gqa_tiling(
    *,
    sequences: tuple[tuple[int, int], ...],
    num_heads: int,
    num_kv_heads: int,
    block_size: int,
    paged: bool,
    aic_count: int,
    layout: str,
    causal: bool = False,
) -> CorePartition:
    """Replay Arch22 FiaTilingNonQuant + split_core::SplitCore."""
    group = num_heads // num_kv_heads
    max_s1 = max((q for q, _ in sequences), default=1)
    max_s2 = max((kv for _, kv in sequences), default=1)
    token_layout = layout in {"TND", "NTD"}
    splitfuse_ragged = token_layout and (paged or block_size > 0) and len(sequences) > 1
    if splitfuse_ragged:
        # fused_infer_attention_score_tiling.cpp:1606 & flash_attention_infer_tiling.h:462
        s_inner = min(ARCH22_GQA_PREFILL_SINNER, max(1, max_s2))
        m_base = ARCH22_SPLITFUSE_M_BASE if max_s1 > ARCH22_GQA_DECODE_MAX_S1 else ARCH22_GQA_PREFILL_M_BASE
    elif token_layout:
        s_inner = ARCH22_GQA_PREFILL_SINNER
        m_base = ARCH22_GQA_PREFILL_M_BASE
    elif max_s1 <= ARCH22_GQA_DECODE_MAX_S1:
        idx = min(group // ARCH22_GQA_GROUP_STEP, len(ARCH22_GQA_SINNER_TABLE) - 1)
        s_inner = ARCH22_GQA_SINNER_TABLE[idx]
        if paged and block_size > 0 and s_inner > block_size and s_inner % block_size != 0:
            s_inner = (s_inner // block_size) * block_size
        s_inner = min(s_inner, max(1, max_s2))
        aligned = _align_up(s_inner, ARCH22_ALIGN_ELEMS)
        m_base = ARCH22_GQA_M_BASE_FALLBACK
        for limit, m_val in ARCH22_GQA_M_BASE_THRESHOLDS:
            if aligned <= limit:
                m_base = m_val
                break
    else:
        s_inner = min(ARCH22_GQA_PREFILL_SINNER, max(1, max_s2))
        m_base = ARCH22_GQA_PREFILL_M_BASE

    is_s1g = layout != "BNSD"
    rows = _build_rows(sequences, num_kv_heads, group, m_base, s_inner, causal=causal, is_s1g=is_s1g)
    allow_stream_k = (len(sequences) * num_kv_heads < aic_count) if splitfuse_ragged else True
    used, max_cost, splits, fd_on = _search_split_core(
        rows, aic_count, stream_k=allow_stream_k, fd_tolerance_blocks=0, fd_least_blocks=0,
        full_block_cost=block_cost(min(max_s1 * group, m_base), s_inner),
    )
    return CorePartition(
        m_tile=m_base, s2_tile=s_inner, used_cores=used, fd_splits_per_row=splits,
        fd_enabled=fd_on, s1_out_split=False, pfa_merge=False,
        max_core_cost=max_cost, total_blocks=sum(r.tiles for r in rows),
    )


def resolve_arch35_fia_gqa_tiling(
    *,
    sequences: tuple[tuple[int, int], ...],
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    v_head_dim: int,
    block_size: int,
    paged: bool,
    aic_count: int,
    l2_bytes: int,
    causal: bool,
    is_decode: bool = True,
    layout: str | None = None,
) -> CorePartition:
    """Replay Arch35 FiaTilingNonQuantArch35 + CheckS1OutSplit fallback + split_core_v2."""
    group = num_heads // num_kv_heads
    batch = len(sequences)
    max_s1 = max((q for q, _ in sequences), default=1)
    max_s2 = max((kv for _, kv in sequences), default=1)
    eff_s2 = (ceil(max_s2 / block_size) * block_size) if (paged and block_size > 0) else max_s2

    bn_size = min(batch * num_kv_heads, aic_count)
    l2_footprint = (
        bn_size * eff_s2 * (head_dim + v_head_dim) * ARCH35_BF16_BYTES * ARCH35_L2_PRESSURE_FACTOR
    )
    s1_out_split = (l2_bytes > 0) and (l2_footprint >= l2_bytes) and (not causal)

    if s1_out_split:
        pfa_merge = (group > 1) and (max_s1 * group < ARCH35_PFA_MERGE_QG_LIMIT)
        if pfa_merge:
            # fused_infer_attention_score_tiling_impl.cpp:365: decode fallback uses M=64, S2=256.
            m_base = ARCH35_SOUTER_32 * ARCH35_CV_RATIO
            s_inner = ARCH35_SINNER_256
            used = min(aic_count, batch * num_kv_heads)
            rows = _build_rows(sequences, num_kv_heads, group, m_base, s_inner)
            return CorePartition(
                m_tile=m_base, s2_tile=s_inner, used_cores=max(1, used),
                fd_splits_per_row=1, fd_enabled=False, s1_out_split=True, pfa_merge=True,
                max_core_cost=max((r.total_cost for r in rows), default=0),
                total_blocks=sum(r.tiles for r in rows),
            )
        m_base = ARCH35_SOUTER_64 * ARCH35_CV_RATIO
        s_inner = ARCH35_SINNER_128
        # Long dense fallback: SplitNBSeq distributes B*H*ceil(Q/128) query-head rectangles over AICs.
        rects_per_head = sum(ceil(q / m_base) for q, _ in sequences)
        used = min(aic_count, rects_per_head * num_heads)
        rows = _build_rows(sequences, num_kv_heads, group, m_base, s_inner)
        return CorePartition(
            m_tile=m_base, s2_tile=s_inner, used_cores=max(1, used),
            fd_splits_per_row=1, fd_enabled=False, s1_out_split=True, pfa_merge=False,
            max_core_cost=max((r.total_cost for r in rows), default=0),
            total_blocks=sum(r.tiles for r in rows),
        )

    # Preferred Arch35 GQA tiler (AdjustSinnerAndSouter + split_core_v2).
    if is_decode and max_s1 <= ARCH35_SOUTER_64 and max_s2 > ARCH35_SINNER_128 and v_head_dim <= ARCH35_GQA_DIM_128:
        s_outer = ARCH35_SOUTER_32
        s_inner = ARCH35_SINNER_256
    else:
        s_outer = ARCH35_SOUTER_64
        s_inner = ARCH35_SINNER_128
    m_base = s_outer * ARCH35_CV_RATIO
    is_s1g = layout != "BNSD"
    rows = _build_rows(sequences, num_kv_heads, group, m_base, s_inner, causal=causal, is_s1g=is_s1g)
    m_fd = min(max_s1 * group, m_base)
    used, max_cost, splits, fd_on = _search_split_core(
        rows, aic_count, stream_k=True,
        fd_tolerance_blocks=ARCH35_FD_TOLERANCE_BLOCKS,
        fd_least_blocks=ARCH35_FD_LEAST_BLOCKS,
        full_block_cost=block_cost(m_fd, s_inner),
    )
    return CorePartition(
        m_tile=m_base, s2_tile=s_inner, used_cores=used, fd_splits_per_row=splits,
        fd_enabled=fd_on, s1_out_split=False, pfa_merge=False,
        max_core_cost=max_cost, total_blocks=sum(r.tiles for r in rows),
    )


def resolve_fia_mla_tiling(
    *,
    sequences: tuple[tuple[int, int], ...],
    physical_heads: int,
    aic_count: int,
    is_arch35: bool,
) -> CorePartition:
    """Replay Arch22 / Arch35 FiaTilingNonQuantMla + split_core."""
    batch = len(sequences)
    max_s1 = max((q for q, _ in sequences), default=1)
    max_s2 = max((kv for _, kv in sequences), default=1)
    if is_arch35:
        m_base = ARCH35_MLA_M_BASE
        s_inner = ARCH35_MLA_S2_BASE
        tol = ARCH35_FD_TOLERANCE_BLOCKS
    else:
        m_base = ARCH22_MLA_M_BASE_DEFAULT
        s_inner = min(ARCH22_MLA_S2_BASE, max(1, max_s2))
        same_kv = len({kv for _, kv in sequences}) == 1
        s1_g = max_s1 * physical_heads
        if batch > 0 and aic_count > 0 and batch % aic_count != 0 and same_kv:
            lcm_val = (batch // gcd(batch, aic_count)) * aic_count
            split_of_s1g = lcm_val // batch
            if split_of_s1g > 0 and s1_g % split_of_s1g == 0:
                candidate = s1_g // split_of_s1g
                if ARCH22_MLA_M_BASE_MIN <= candidate <= m_base:
                    m_base = candidate
        tol = 0

    rows = _build_rows(sequences, 1, physical_heads, m_base, s_inner)
    m_fd = min(max_s1 * physical_heads, m_base)
    used, max_cost, splits, fd_on = _search_split_core(
        rows, aic_count, stream_k=True, fd_tolerance_blocks=tol,
        fd_least_blocks=ARCH35_FD_LEAST_BLOCKS,
        full_block_cost=block_cost(m_fd, s_inner),
    )
    return CorePartition(
        m_tile=m_base, s2_tile=s_inner, used_cores=used, fd_splits_per_row=splits,
        fd_enabled=fd_on, s1_out_split=False, pfa_merge=False,
        max_core_cost=max_cost, total_blocks=sum(r.tiles for r in rows),
    )


def compact_arch35_preload_s2(
    context,
    n_tiles: int,
    s2_phases_fn,
    lower_phase,
    place: dict,
    *,
    compact_fn,
    preload_slots: int,
    pipeline_slots: int,
    cycle_slots: int,
    threshold: int,
):
    """Compact Arch35 fia_kernel_noquant_gqa.h:522 ExecuteTask (PRELOAD_N=3) in O(1)."""
    from dataclasses import replace as _replace

    def _split(ti: int):
        mm1, vec1, mm2, vec2 = [], [], [], []
        for p in s2_phases_fn(ti):
            if p.name.startswith(("k_panel_", "qk_panel_", "k_page_", "k_nd2nz_")) or p.name in {"block_table", "score_fixpipe"}:
                mm1.append(p)
            elif p.name in {"online_softmax", "probability_transfer"}:
                vec1.append(p)
            elif p.name.startswith(("v_panel_", "pv_panel_", "v_page_", "v_nd2nz_")) or p.name == "pv_fixpipe":
                mm2.append(p)
            else:
                vec2.append(p)
        return mm1, vec1, mm2, vec2

    def cycle(c: int):
        stages = []
        last_qk = last_fix = last_k_page = None
        if 0 <= c < n_tiles:
            mm1, _, _, _ = _split(c)
            for p in mm1:
                st = lower_phase(p, place)
                prev = ()
                if c >= preload_slots:
                    if p.name.startswith("qk_panel_"):
                        prev = ("pv_panel_0",)
                    elif p.name == "score_fixpipe":
                        prev = ("pv_fixpipe",)
                    elif p.name.startswith("k_panel_0"):
                        prev = ("qk_panel_0",)
                    elif p.name == "k_nd2nz_issue":
                        prev = ("v_panel_0_load",)
                    elif p.name == "k_page_cmd":
                        prev = ("v_page_cmd",)
                stages.append(_replace(st, previous=prev))
                if p.name.startswith("qk_panel_"):
                    last_qk = st.name
                elif p.name == "score_fixpipe":
                    last_fix = st.name
                elif p.name == "k_page_cmd":
                    last_k_page = st.name
        if 0 <= c - 1 < n_tiles:
            _, vec1, _, _ = _split(c - 1)
            for p in vec1:
                st = lower_phase(p, place)
                if p.name == "online_softmax":
                    prev = ("score_fixpipe",) + (("online_output",) if c >= pipeline_slots else ())
                    st = _replace(st, after=(), previous=prev)
                else:
                    st = _replace(st, after=("online_softmax",), previous=())
                stages.append(st)
        if 0 <= c - 2 < n_tiles:
            _, _, mm2, _ = _split(c - 2)
            has_v_issue = any(p.name == "v_nd2nz_issue" for p in mm2)
            has_v_page = any(p.name == "v_page_cmd" for p in mm2)
            for idx_p, p in enumerate(mm2):
                st = lower_phase(p, place)
                if p.name == "v_nd2nz_issue":
                    st = _replace(st, after=("k_panel_0_load",) if (0 <= c < n_tiles) else (), previous=())
                elif p.name == "v_page_cmd":
                    st = _replace(st, after=(last_k_page,) if last_k_page else (), previous=())
                elif p.name.startswith("v_panel_0"):
                    after_v = ("v_nd2nz_issue",) if has_v_issue else ()
                    st = _replace(st, after=after_v, previous=("pv_panel_0",) if c >= preload_slots else ())
                elif p.name.startswith("pv_panel_0"):
                    pv_deps = tuple(x for x in (last_qk, "v_panel_0_load", "v_page_cmd" if has_v_page else None) if x)
                    st = _replace(st, after=pv_deps, previous=("probability_transfer",))
                elif p.name == "pv_fixpipe":
                    st = _replace(st, after=tuple(x for x in (last_fix, mm2[idx_p - 1].name) if x), previous=())
                stages.append(st)
        if 0 <= c - preload_slots < n_tiles:
            _, _, _, vec2 = _split(c - preload_slots)
            for p in vec2:
                st = lower_phase(p, place)
                st = _replace(st, after=("online_softmax",) if (0 <= c - 1 < n_tiles) else (),
                              previous=("pv_fixpipe",))
                stages.append(st)
        return tuple(stages)

    return compact_fn(
        context, n_tiles + preload_slots, cycle_slots, cycle,
        head=pipeline_slots, tail=pipeline_slots, threshold=threshold,
    )


MS_PER_SECOND_SC = 1000.0
P01_CAPACITY_EPS = 1.0 + 1e-6
P01_HBM_SAT_FRAC = 0.96


def _p01_cold_curve_info(context, *, warm: bool = False) -> tuple[list[tuple[int, float]], float] | None:
    """Return validated (sorted_points, hbm_read_p50_bps) from P01_datacopy."""
    pt = getattr(context, "primitive_table", None)
    if pt is None:
        return None
    if any("silicon_profile_conflict" in str(w) for w in (getattr(pt, "warnings", ()) or ())):
        return None
    key = "mte2_aic_warm_scaling_bps" if warm else "mte2_aic_cold_scaling_bps"
    cell = pt.cell("P01_datacopy")
    if getattr(cell, "status", None) != "calibrated":
        return None
    params = cell.parameters or {}
    if params.get("silicon_profile_conflict"):
        return None
    curve = params.get(key)
    if not isinstance(curve, Mapping) or not curve:
        return None
    pts = sorted((int(k), float(v)) for k, v in curve.items() if str(k).isdigit() and float(v) > 0.0)
    if not pts:
        return None
    hbm_band = cell.metrics.get("hbm_read_bytes_per_s")
    hbm_cap = float(hbm_band.certified_upper_capacity) if (hbm_band and hbm_band.certified_upper_capacity) else 0.0
    hbm_read_p50 = float(hbm_band.p50) if (hbm_band and hbm_band.p50) else 0.0
    mte2_band = cell.metrics.get("mte2_aic_hit_bytes_per_s_per_core")
    mte2_cap = float(mte2_band.certified_upper_capacity) if (mte2_band and mte2_band.certified_upper_capacity) else 0.0
    for k_cores, rate in pts:
        if not warm and hbm_cap > 0.0 and rate > hbm_cap * P01_CAPACITY_EPS:
            return None
        if mte2_cap > 0.0 and rate > k_cores * mte2_cap * P01_CAPACITY_EPS:
            return None
    return pts, hbm_read_p50


def _p01_interp_bps(pts: list[tuple[int, float]], active_owners: int) -> float:
    if active_owners <= pts[0][0]:
        return pts[0][1] * (active_owners / pts[0][0])
    val = pts[-1][1]
    for i in range(len(pts) - 1):
        c0, r0 = pts[i]
        c1, r1 = pts[i + 1]
        if c0 <= active_owners <= c1:
            return r0 + (r1 - r0) * (active_owners - c0) / max(1, c1 - c0)
    return val


def _p01_active_core_bps(context, active_owners: int, *, warm: bool = False) -> float | None:
    """Interpolate P01 active-core MTE2-AIC scaling curve (bytes/s) within certified capacity."""
    if active_owners < 1:
        return None
    info = _p01_cold_curve_info(context, warm=warm)
    if info is None:
        return None
    pts, _ = info
    val = _p01_interp_bps(pts, active_owners)
    cell = context.primitive_table.cell("P01_datacopy")
    hbm_band = cell.metrics.get("hbm_read_bytes_per_s")
    hbm_cap = float(hbm_band.certified_upper_capacity) if (hbm_band and hbm_band.certified_upper_capacity) else 0.0
    mte2_band = cell.metrics.get("mte2_aic_hit_bytes_per_s_per_core")
    mte2_cap = float(mte2_band.certified_upper_capacity) if (mte2_band and mte2_band.certified_upper_capacity) else 0.0
    if not warm and hbm_cap > 0.0:
        val = min(val, hbm_cap)
    if mte2_cap > 0.0:
        val = min(val, active_owners * mte2_cap)
    return val


def p01_cold_mte2_aic_cap(
    context, active_owners: int, *, include_hbm: bool = False, hbm_only: bool = False,
) -> dict[str, float]:
    """Per-owner MTE2-AIC (and optional HBM) cold DRAM cap (bytes/ms) from P01.

    Per-core MTE2-AIC retains its unsaturated cold pipe rate even after full-die
    HBM saturates (so strided/burst-amplified transfers do not revert to the L2-hit
    rate), while HBM cap binds only when ``active_owners`` supply is below die HBM.
    """
    if active_owners < 1:
        return {}
    info = _p01_cold_curve_info(context, warm=False)
    if info is None:
        return {}
    pts, hbm_read_p50 = info
    total_bps = _p01_active_core_bps(context, active_owners, warm=False)
    if total_bps is None:
        return {}
    hbm_name = context.resource_names.get("hbm")
    die_hbm_bps = float(context.rates.get(hbm_name, 0.0)) * MS_PER_SECOND_SC if hbm_name else 0.0
    caps: dict[str, float] = {}
    curve_max_bps = max(r for _, r in pts)
    sat_ref = curve_max_bps if curve_max_bps > 0.0 else (hbm_read_p50 if hbm_read_p50 > 0.0 else die_hbm_bps)
    is_saturated = sat_ref > 0.0 and total_bps >= sat_ref * P01_HBM_SAT_FRAC
    if is_saturated:
        unsat = [(k, r) for k, r in pts if r < sat_ref * P01_HBM_SAT_FRAC]
        per_core_bps = (unsat[-1][1] / float(unsat[-1][0])) if unsat else (total_bps / float(active_owners))
    else:
        per_core_bps = total_bps / float(active_owners)
    if not hbm_only:
        mte2_name = context.resource_names.get("mte2_aic")
        if mte2_name:
            # When total_bps saturates the AIC cold DRAM plateau, keep the per-core cold MTE2
            # rate at the highest unsaturated knee rather than dividing saturated HBM by active_owners.
            caps[mte2_name] = per_core_bps / MS_PER_SECOND_SC
    if (include_hbm or hbm_only) and hbm_name and (die_hbm_bps <= 0.0 or total_bps < die_hbm_bps):
        hbm_per_core = total_bps / float(active_owners)
        caps[hbm_name] = hbm_per_core / MS_PER_SECOND_SC
    return caps


def p01_cold_hbm_group_cap(context, active_owners: int, *, include_l2: bool = False) -> dict[str, float]:
    """Aggregate group HBM (and optional L2) cap (bytes/ms) for combined multi-core stages."""
    caps: dict[str, float] = {}
    hbm_name = context.resource_names.get("hbm")
    cold_bps = _p01_active_core_bps(context, active_owners, warm=False)
    if hbm_name and cold_bps is not None:
        caps[hbm_name] = cold_bps / MS_PER_SECOND_SC
    if include_l2:
        l2_name = context.resource_names.get("l2")
        warm_bps = _p01_active_core_bps(context, active_owners, warm=True)
        if l2_name and warm_bps is not None:
            caps[l2_name] = warm_bps / MS_PER_SECOND_SC
    return caps
