"""Executable fused inference attention rectangles and panel loops.

[Role] Lower FIA S1 rectangles and S2 tiles into owner-local programs using
  native CANN host-tiling (Arch22/Arch35 split_core & L2 residency).
[Invariants] Internal Cube panels conserve useful work. FixPipe and cross-core
  dependencies follow the selected FIA template's publication boundary.
  Shared segment descriptors drive both prepared work and lowered stages so
  declared and serviced work match identically.
[Timeouts & Locks] Finite tile traversal and O(1) PipelineTemplateProgram
  compaction for homogeneous interior S2 tiles without locks.
[Side-Effect Policy] Pure planning and program construction.
"""
from __future__ import annotations

from bisect import bisect_right
from collections import deque
from dataclasses import asdict, dataclass, replace
from functools import partial
from itertools import accumulate
from math import ceil, gcd
from typing import Callable, Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_host/fused_infer_attention_score_tiling.cpp",
        source_sha256="bac8846fe180d6b95311a7a80b333e0e37a4a587c81c9cfd798e910c14197be0",
    ),
)
from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...program.stages import scope_allocations
from ...runtime.memory_physics import (
    AddressRun, AddressRunPlan, BurstTurnRule, HbmBusSpec, MemoryAccessPattern, ReuseKey,
)
from ..contracts import OperatorPlanningContext
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sharedkv.l0_sync import (
    p01_hbm_scatter_row_open_bytes,
    p01_strided_cold_bps,
    p02_load_setup_ms,
    p03_fixpipe_setup_ms,
    p13_mode4_ms,
    p13_syncall_ms,
)
from ..sharedkv.split_core import (
    CorePartition,
    _p01_active_core_bps,
    p01_cold_mte2_aic_cap,
    resolve_arch22_fia_gqa_tiling,
    resolve_arch35_fia_gqa_tiling,
)
from .common import (
    ARCH22, BF16_BYTES, FIA_TILE, FP32_BYTES, INDEX_BYTES, PAGED_LAYOUTS, QUANT_MODE_BF16,
    SOFTMAX_OPS, TOKEN_MAJOR_LAYOUTS, AttentionContract, AttentionMemoryRead, AttentionPhase, FiaPanelPlan,
    _allocations, _placement, _stage, attention_owner_waves, bsh_kv_access_plan, make_plan, owner_pairs, panels,
    planning_sequence_spans, read_work, shipping_fia_panel_plan, solve_wide_head_tiling, unmodeled,
    workspace_write_work,
)

FIXPIPE_ROW_ALIGNMENT = 2
FIXPIPE_COLUMN_ALIGNMENT = 8
FIA_PRELOAD_SLOTS = 3
ARCH35_PIPELINE_SLOTS = 4
ARCH22_WORKSPACE_SLOTS = 2
ARCH22_DECODE_M_LIMIT = 32
ARCH35_DECODE_M_LIMIT = 64
ARCH22_WS_SCORE_PROB_BYTES = 6
ARCH22_WS_OUT_BYTES = 8
COMPACT_TILE_THRESHOLD = 24
ARCH22_VEC1_UB_SLICE_BYTES = 3 * 16 * 1024  # fia_block_vec_nonquant.h:483 (3 * 16 KiB per AIV)
AIV_SUBBLOCKS_PER_AIC = 2
DAV_2201_HBM_BURST_BYTES = 512
DAV_3510_HBM_BURST_BYTES = 256
MS_PER_SECOND_FIA = 1000.0


def _bsh_strided_cold_caps(
    context, base_caps: dict[str, float], contract: AttentionContract,
    *, gm_workspace: bool, burst_bytes: int, active_owners: int, total_aics: int,
) -> dict[str, float]:
    """Arch22 token-major BSH/TND strided cold MTE2/HBM cap from P01 strided_cold_surface."""
    if (not gm_workspace or contract.kv_layout not in TOKEN_MAJOR_LAYOUTS
            or contract.num_kv_heads <= 1 or active_owners < 1):
        return base_caps
    row_bytes = contract.head_dim * BF16_BYTES
    if row_bytes >= burst_bytes:
        return base_caps
    stride_bytes = contract.num_kv_heads * row_bytes
    strided_die = p01_strided_cold_bps(context, row_bytes, stride_bytes)
    if strided_die is None or strided_die <= 0.0:
        return base_caps
    full_cold = _p01_active_core_bps(context, max(1, total_aics), warm=False)
    act_cold = _p01_active_core_bps(context, active_owners, warm=False)
    scale = (act_cold / full_cold) if (full_cold and act_cold and full_cold > 0.0) else 1.0
    # Adjacent KV heads share each 512 B burst (burst_bytes / row_bytes peer coalescing).
    coalesce = max(1.0, float(burst_bytes) / float(row_bytes))
    eff_die_bps = strided_die * coalesce * min(1.0, scale)
    per_owner_ms = eff_die_bps / (MS_PER_SECOND_FIA * float(active_owners))
    caps = dict(base_caps)
    for role in ("mte2_aic", "hbm"):
        res_name = context.resource_names.get(role)
        if not res_name:
            continue
        prev = caps.get(res_name)
        # mte2_aic carries burst-amplified copy bytes (coalesce * logical), so its
        # per-owner ceiling scales by coalesce while HBM carries logical bytes (1x).
        target = per_owner_ms * (coalesce if role == "mte2_aic" else 1.0)
        caps[res_name] = min(prev, target) if prev is not None else target
    return caps


def _weighted_contiguous_rect_partition(rects: tuple[_FiaRect, ...], owners: int) -> tuple[tuple[int, ...], ...]:
    """CANN SplitNBSeq weighted contiguous rectangle assignment (tiling_impl.cpp:638)."""
    n_rects = len(rects)
    if n_rects <= 0 or owners <= 0:
        return ()
    weights = [max(1, r.tile_count) for r in rects]
    total = sum(weights)
    out: list[tuple[int, ...]] = []
    idx = 0
    unassigned = total
    for core in range(owners):
        if idx >= n_rects:
            break
        cores_left = owners - core
        limit = unassigned // cores_left
        cur: list[int] = []
        cur_w = 0
        while idx < n_rects:
            w = weights[idx]
            if not cur or cur_w + w <= limit + w // 2:
                cur.append(idx)
                cur_w += w
                idx += 1
            else:
                break
        if core == owners - 1 and idx < n_rects:
            cur.extend(range(idx, n_rects))
            idx = n_rects
        unassigned -= cur_w
        if cur:
            out.append(tuple(cur))
    return tuple(out)


def _fixpipe_bytes(rows: int, columns: int) -> int:
    return ceil(rows / FIXPIPE_ROW_ALIGNMENT) * FIXPIPE_ROW_ALIGNMENT * ceil(columns / FIXPIPE_COLUMN_ALIGNMENT) * FIXPIPE_COLUMN_ALIGNMENT * FP32_BYTES


def _l2_capacity_bytes(ctx: OperatorPlanningContext) -> int:
    if ctx.admission_memory_view is not None:
        return int(ctx.admission_memory_view.effective_l2_bytes(ctx.geometry.rank or 0))
    return int(ctx.geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0)))


@dataclass(frozen=True, slots=True)
class _FiaRect:
    sequence: int
    m: int
    head: int
    kv: int
    row: int
    first: int
    end: int
    rect_in_head: int = 0
    rects_per_head: int = 1
    prev_touched_start: int = 0
    prev_touched_end: int = 0
    tile_ranges: tuple[tuple[int, int], ...] = ()
    prev_touched_spans: tuple[tuple[int, int], ...] = ()

    @property
    def tile_spans(self) -> tuple[tuple[int, int], ...]:
        if self.tile_ranges:
            return self.tile_ranges
        return ((self.first, self.end),) if self.end > self.first else ()

    @property
    def tile_count(self) -> int:
        return sum(hi - lo for lo, hi in self.tile_spans)


@dataclass(frozen=True, slots=True)
class _CoreSegment:
    """One contiguous S2 tile span of one rectangle assigned to one physical AIC."""

    owner_idx: int
    rect_idx: int
    tile_start: int
    tile_end: int


@dataclass(frozen=True, slots=True)
class _ShippingS2:
    """Replay the arch35 MM1/Vec1 lookahead and MM2/Vec2 drain."""

    count: int
    slots: int
    tile: Callable[[int], tuple[Stage, ...]]

    def start(self, engine, name, ready, *, owner=""):
        if not self.count:
            return ready
        done = engine.future(f"{name}:complete")
        cube_tail = vector_tail = ready
        pending = deque()
        next_tile = 0

        def submit(index, stages, events):
            tile_name = f"{name}:tile{index}"
            scoped = partial(scope_allocations, tile_name)
            for stage in stages:
                deps = tuple(events[dep] for dep in stage.after)
                if not deps or stage.after == ("probability_transfer",) or stage.name == "v_panel_0_load":
                    deps += (cube_tail,)
                if stage.name in {"online_softmax", "online_output"}:
                    deps += (vector_tail,)
                events[stage.name] = stage.submit(engine, f"{tile_name}:{stage.name}", deps, scoped, owner)

        def issue(_):
            nonlocal next_tile, cube_tail, vector_tail
            while len(pending) < self.slots and next_tile < self.count:
                stages = self.tile(next_tile)
                vec1_end = next(i for i, stage in enumerate(stages) if stage.name == "probability_transfer") + 1
                events = {}
                submit(next_tile, stages[:vec1_end], events)
                cube_tail, vector_tail = events["score_fixpipe"], events["probability_transfer"]
                pending.append((next_tile, stages[vec1_end:], events))
                next_tile += 1

            while pending:
                index, stages, events = pending.popleft()
                submit(index, stages, events)
                cube_tail, vector_tail = events["pv_fixpipe"], events["online_output"]
                if next_tile < self.count:
                    engine.on_complete(vector_tail, issue)
                    return
            engine.on_complete(vector_tail, lambda _: engine.resolve(done))

        engine.on_complete(ready, issue)
        return done


def _balanced_core_segments(
    rects: tuple[_FiaRect, ...], num_owners: int,
) -> tuple[tuple[_CoreSegment, ...], tuple[int, ...]]:
    """Partition all S2 tiles across num_owners AICs evenly, matching split_core balance."""
    spans = [(r_idx, lo, hi) for r_idx, r in enumerate(rects) for lo, hi in r.tile_spans if hi > lo]
    total_tiles = sum(hi - lo for _, lo, hi in spans)
    if total_tiles <= 0 or num_owners <= 0:
        return (), (1,) * len(rects)
    base, extra = divmod(total_tiles, num_owners)
    segments: list[_CoreSegment] = []
    splits_by_rect = [0] * len(rects)
    s_idx = 0
    t_cursor = spans[0][1]
    for owner_idx in range(num_owners):
        quota = base + (1 if owner_idx < extra else 0)
        while quota > 0 and s_idx < len(spans):
            r_idx, _, s_hi = spans[s_idx]
            avail = s_hi - t_cursor
            if avail <= 0:
                s_idx += 1
                if s_idx < len(spans):
                    t_cursor = spans[s_idx][1]
                continue
            take = min(quota, avail)
            segments.append(_CoreSegment(owner_idx, r_idx, t_cursor, t_cursor + take))
            splits_by_rect[r_idx] += 1
            t_cursor += take
            quota -= take
            if t_cursor >= s_hi:
                s_idx += 1
                if s_idx < len(spans):
                    t_cursor = spans[s_idx][1]
    return tuple(segments), tuple(max(1, s) for s in splits_by_rect)


def _compact_s2_pipeline(context, n_tiles: int, slots: int, tile_fn: Callable[[int], tuple[Stage, ...]]):
    """Execute a long uniform S2 loop in O(1) via exact fill+steady+drain periodic compaction."""
    return compact_periodic_pipeline(
        context, n_tiles, slots, tile_fn, head=1, tail=1, threshold=COMPACT_TILE_THRESHOLD,
    )


def _merge_intervals(intervals: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    """Merge sorted/unsorted half-open ``[lo, hi)`` intervals into disjoint spans."""
    ordered = sorted((lo, hi) for lo, hi in intervals if hi > lo)
    if not ordered:
        return ()
    merged: list[tuple[int, int]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return tuple(merged)


def _slice_query_spans(
    spans: tuple[tuple[int, int], ...], first: int, last: int,
) -> tuple[tuple[int, int], ...]:
    """Return half-open KV coordinate spans covered by local query slice ``[first, last]``."""
    out: list[tuple[int, int]] = []
    offset = 0
    for s_lo, s_hi in spans:
        width = s_hi - s_lo
        ov_lo = max(first, offset)
        ov_hi = min(last + 1, offset + width)
        if ov_hi > ov_lo:
            out.append((s_lo + (ov_lo - offset), s_lo + (ov_hi - offset)))
        offset += width
    return tuple(out)


def _kv_intervals_for_spans(
    q_spans: tuple[tuple[int, int], ...],
    *,
    kv: int,
    causal: bool,
    visible: int,
    window_size: int,
    selected_tokens: int | None,
) -> tuple[tuple[int, int], ...]:
    """Map query position spans ``[p_lo, p_hi)`` to merged visible KV intervals."""
    if visible <= 0 or not q_spans:
        return ()
    raw: list[tuple[int, int]] = []
    if causal:
        for p_lo, p_hi in q_spans:
            lo = max(0, p_lo + 1 - visible)
            hi = min(kv, max(0, p_hi))
            if hi > lo:
                raw.append((lo, hi))
        return _merge_intervals(raw)
    if window_size:
        if selected_tokens is not None and 0 < selected_tokens < 2 * window_size - 1:
            left = min(window_size - 1, (selected_tokens - 1) // 2)
            right = min(window_size, selected_tokens - left)
        else:
            left = window_size - 1
            right = window_size
        for p_lo, p_hi in q_spans:
            lo = max(0, p_lo - left)
            hi = min(kv, (p_hi - 1) + right)
            if hi > lo:
                raw.append((lo, hi))
        return _merge_intervals(raw)
    return ((kv - visible, kv),) if visible > 0 else ()


def _merge_tile_ranges(
    kv_intervals: tuple[tuple[int, int], ...], s2_tile: int,
) -> tuple[tuple[int, int], ...]:
    """Project KV token intervals ``[lo, hi)`` onto merged S2 tile ranges."""
    return _merge_intervals([
        (lo // s2_tile, ceil(hi / s2_tile)) for lo, hi in kv_intervals if hi > lo
    ])


def _tile_query_bounds(
    row: int, m: int, queries: int, group: int, token_major: bool,
) -> tuple[int, int, int]:
    """Return ``(first, last, prev_last)`` local query indices for a tile row."""
    if token_major:
        first, last = row // group, (row + m - 1) // group
        prev_last = (row - 1) // group if row > 0 else 0
        return first, last, prev_last
    prev_last = min(queries - 1, row - 1) if row > 0 else 0
    if row // queries == (row + m - 1) // queries:
        return row % queries, (row + m - 1) % queries, prev_last
    return 0, queries - 1, prev_last


@dataclass(frozen=True, slots=True)
class _FiaLoop:
    contract: AttentionContract
    panels: FiaPanelPlan
    ctx: OperatorPlanningContext
    dtype: str
    gm_workspace: bool
    sequences: tuple[tuple[int, int], ...]
    row_ends: tuple[int, ...]
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    buffers: tuple[tuple[str, int], ...]
    burst_bytes: int
    q_offsets: tuple[int, ...] = ()
    kv_offsets: tuple[int, ...] = ()
    kv_strides: tuple[int, ...] = ()
    m_tile: int = FIA_TILE
    s2_tile: int = FIA_TILE
    effective_splits: int = 1
    ws_l2_resident: bool = False
    kv_reuse_hit_frac: float = 0.0
    fd_segments: tuple[_CoreSegment, ...] = ()
    rect_splits: tuple[int, ...] = ()
    phases: tuple[AttentionPhase, ...] = ()

    def rectangle(self, index):
        c = self.contract
        group = c.num_heads // c.num_kv_heads
        sequence = bisect_right(self.row_ends, index)
        queries, kv = self.sequences[sequence]
        rects_per_head = ceil(queries * group / self.m_tile)
        head, tile = divmod(index - (self.row_ends[sequence - 1] if sequence else 0), rects_per_head)
        row = tile * self.m_tile
        m = min(self.m_tile, queries * group - row)
        visible = min(kv, kv if c.selected_tokens is None else c.selected_tokens, c.window_size or kv)
        spans = (
            c.sequence_query_spans[sequence]
            if (c.sequences and sequence < len(c.sequence_query_spans)) else ()
        )
        prev_spans: tuple[tuple[int, int], ...] = ()
        if c.sequences and (c.causal or c.window_size):
            token_major = (self.ctx.layout or c.kv_layout) != "BNSD"
            first, last, prev_last = _tile_query_bounds(row, m, queries, group, token_major)
            eff_spans = spans if spans else ((kv - queries, kv),)
            cur_q_spans = _slice_query_spans(eff_spans, first, last)
            kv_intervals = _kv_intervals_for_spans(
                cur_q_spans, kv=kv, causal=c.causal, visible=visible,
                window_size=c.window_size, selected_tokens=c.selected_tokens,
            )
            if tile > 0:
                prev_q_spans = _slice_query_spans(eff_spans, 0, prev_last)
                prev_spans = _kv_intervals_for_spans(
                    prev_q_spans, kv=kv, causal=c.causal, visible=visible,
                    window_size=c.window_size, selected_tokens=c.selected_tokens,
                )
        else:
            kv_intervals = ((kv - visible, kv),) if visible > 0 else ()
            if tile > 0:
                prev_spans = kv_intervals
        tile_ranges = _merge_tile_ranges(kv_intervals, self.s2_tile) if visible else ()
        t_first = tile_ranges[0][0] if tile_ranges else 0
        t_end = tile_ranges[-1][1] if tile_ranges else t_first
        prev_start = prev_spans[0][0] if prev_spans else 0
        prev_end = prev_spans[-1][1] if prev_spans else 0
        return _FiaRect(
            sequence, m, head, kv, row, t_first, t_end,
            rect_in_head=tile, rects_per_head=rects_per_head,
            prev_touched_start=prev_start, prev_touched_end=prev_end,
            tile_ranges=tile_ranges, prev_touched_spans=prev_spans,
        )

    def _kv(self, rect, start, n, dim, width, offset, tensor):
        c = self.contract
        page = c.block_size if c.kv_layout in PAGED_LAYOUTS else rect.kv
        base_token = self.kv_offsets[rect.sequence] if self.kv_offsets and not c.block_table else 0
        stride = self.kv_strides[rect.sequence] if self.kv_strides else rect.kv
        orig_start, orig_n = start, n
        runs = []
        while n:
            block, begin = divmod(start, page)
            extent = min(n, page - begin)
            physical = c.block_table[rect.sequence][block] if c.block_table else block
            base = bsh_kv_access_plan(
                c.num_kv_heads, dim, BF16_BYTES, c.kv_layout,
                rows=extent, first_head=rect.head, block_size=page, tokens_per_head=stride,
                base_byte=(base_token + physical * page) * c.num_kv_heads * dim * BF16_BYTES,
            ).runs[0]
            runs.append(AddressRun(base.base_byte + begin * base.stride_bytes + offset * BF16_BYTES,
                                   extent, width * BF16_BYTES, base.stride_bytes, base.layout))
            n, start = n - extent, start + extent
        plan = AddressRunPlan(tuple(runs))
        row_bytes = width * BF16_BYTES
        # Each KV-head owner issues strided row_bytes MTE2 DMA transfers in token-major layouts
        # (BSH/BSND/PA_BLK_BSND/TND, gm_layout.h:199); peer heads coalesce in L2 before HBM when width == dim.
        token_span = (c.num_kv_heads * row_bytes
                      if (c.kv_layout in TOKEN_MAJOR_LAYOUTS and width == dim) else row_bytes)
        hbm_rules = (BurstTurnRule(row_bytes, row_bytes + 1, row_bytes + 1, None,
                                   self.burst_bytes - token_span),) if token_span < self.burst_bytes else ()
        copy = plan.aligned_copy_bytes()
        if c.kv_layout in TOKEN_MAJOR_LAYOUTS and c.num_kv_heads > 1 and row_bytes < self.burst_bytes:
            owner_rules = (BurstTurnRule(row_bytes, row_bytes + 1, row_bytes + 1, None,
                                         self.burst_bytes - row_bytes),)
            copy = max(copy, plan.dram_sector_and_service(owner_rules)[1])
        hbm = plan.dram_sector_and_service(hbm_rules)[1]
        known_addresses = c.block_table or (c.kv_layout not in PAGED_LAYOUTS and self.kv_offsets)
        memory = (plan, hbm_rules, tensor) if c.kv_tensor_id and known_addresses else None
        if memory is None and rect.rect_in_head > 0 and self.kv_reuse_hit_frac > 0.0 and orig_n > 0:
            prev_spans = (
                rect.prev_touched_spans
                if rect.prev_touched_spans
                else ((rect.prev_touched_start, rect.prev_touched_end),)
            )
            reused_toks = sum(
                max(0, min(orig_start + orig_n, p_hi) - max(orig_start, p_lo))
                for p_lo, p_hi in prev_spans
            )
            eff_hit_frac = (reused_toks / float(orig_n)) * self.kv_reuse_hit_frac
            if eff_hit_frac > 0.0:
                hit_bytes = int(round(copy * eff_hit_frac))
                miss_hbm = max(0, int(round(hbm * (1.0 - eff_hit_frac))))
                return read_work(self.ctx, copy, hbm=miss_hbm, l2=hit_bytes), None
        return read_work(self.ctx, copy, hbm=hbm), memory

    def s2_phases(self, rect, tile):
        c, p = self.contract, self.panels
        start = tile * self.s2_tile
        n = min(self.s2_tile, rect.kv - start)
        pairs = rect.m * n
        shipping_ub = not self.gm_workspace and c.panel_mode == "shipping"
        score = _fixpipe_bytes(rect.m, n) if shipping_ub else pairs * FP32_BYTES
        probability = pairs * BF16_BYTES
        ws_l2 = self.ws_l2_resident
        stages = []

        def add(name, work, after=None, slot=None, memory=None, pages: int = 0):
            deps = (stages[-1].name,) if after is None and stages else after or ()
            stages.append(AttentionPhase(name, work, deps, slot, memory, pages))

        tile_pages = ceil((start % c.block_size + n) / c.block_size) if c.kv_layout in PAGED_LAYOUTS else 0
        is_a35_dec = (not self.gm_workspace) and (rect.m <= ARCH35_DECODE_M_LIMIT)
        if c.kv_layout in PAGED_LAYOUTS:
            add("block_table", read_work(self.ctx, tile_pages * INDEX_BYTES, aiv=True))
        k_root = (stages[-1].name,) if stages else ()
        if tile_pages > 0:
            add("k_page_cmd", Work(), k_root, pages=tile_pages)
        if is_a35_dec:
            add("k_nd2nz_issue", Work(), k_root)
        offset = 0
        depth = min(p.k_l1_slots, p.l0b_slots)
        for index, width in enumerate(p.k_panels):
            load, compute = f"k_panel_{index}_load", f"qk_panel_{index}_accumulate"
            slot = index % p.l0b_slots
            if index >= depth:
                ready = (f"qk_panel_{index - depth}_accumulate",)
            elif is_a35_dec and index == 0:
                ready = ("k_nd2nz_issue",)
            else:
                ready = k_root
            previous = (f"qk_panel_{index - 1}_accumulate",) if index else ()
            work, memory = self._kv(rect, start, n, c.head_dim, width, offset, "key")
            add(load, work, ready, slot, memory, pages=tile_pages)
            comp_deps = (load, *previous, *(("k_page_cmd",) if (tile_pages > 0 and index == 0) else ()))
            add(compute, Work({self.dtype: 2 * pairs * width},
                              {"mte1": (rect.m + n) * width * BF16_BYTES}), comp_deps, slot)
            offset += width
        if self.gm_workspace:
            add("score_fixpipe", workspace_write_work(self.ctx, "fixpipe", score, l2_resident=ws_l2))
            score_read = read_work(self.ctx, score, aiv=True, hbm=0 if ws_l2 else score, l2=score if ws_l2 else 0)
            softmax_work = Work(resources={"vector": SOFTMAX_OPS * pairs})
            prob_write = workspace_write_work(self.ctx, "mte3", probability, l2_resident=ws_l2)
            # fia_block_vec_nonquant.h:482-498: when score exceeds 1 UB slice per AIV (48 KiB),
            # ProcessVec1SingleBuf sub-tiles rows and overlaps MTE2 || PIPE_V || MTE3 in UB.
            if score > AIV_SUBBLOCKS_PER_AIC * ARCH22_VEC1_UB_SLICE_BYTES:
                add("online_softmax", score_read + softmax_work + prob_write)
            else:
                add("score_workspace_read", score_read)
                add("online_softmax", softmax_work)
                add("probability_transfer", prob_write)
            add("probability_workspace_read", read_work(self.ctx, probability,
                                                        hbm=0 if ws_l2 else probability,
                                                        l2=probability if ws_l2 else 0))
        else:
            add("score_fixpipe", Work(resources={"fixpipe": score}))
            add("online_softmax", Work(resources={"vector": SOFTMAX_OPS * pairs}))
            add("probability_transfer", Work(resources={"mte3": probability}))
        prob_ready = stages[-1].name
        if self.gm_workspace and rect.m <= ARCH22_DECODE_M_LIMIT:
            v_root = (prob_ready,)
        else:
            v_root = (f"k_panel_{len(p.k_panels) - 1}_load",) if p.k_panels else k_root
        if tile_pages > 0:
            add("v_page_cmd", Work(), ("k_page_cmd",), pages=tile_pages)
        if is_a35_dec:
            add("v_nd2nz_issue", Work(), v_root)
        offset = 0
        depth = min(p.v_l1_slots, p.l0b_slots)
        for index, width in enumerate(p.v_panels):
            load, compute = f"v_panel_{index}_load", f"pv_panel_{index}"
            slot = index % p.l0b_slots
            if index >= depth:
                ready = (f"pv_panel_{index - depth}",)
            elif is_a35_dec and index == 0:
                ready = ("v_nd2nz_issue",)
            else:
                ready = v_root
            previous = prob_ready if index == 0 else stages[-1].name
            work, memory = self._kv(rect, start, n, c.v_head_dim, width, offset, "value")
            add(load, work, ready, slot, memory, pages=tile_pages)
            pv_deps = (load, previous, *(("v_page_cmd",) if (tile_pages > 0 and index == 0) else ()))
            add(compute, Work({self.dtype: 2 * pairs * width},
                              {"mte1": probability + n * width * BF16_BYTES}), pv_deps, slot)
            if not shipping_ub:
                partial = rect.m * width * FP32_BYTES
                if self.gm_workspace:
                    add(f"pv_panel_{index}_fixpipe",
                        workspace_write_work(self.ctx, "fixpipe", partial, l2_resident=ws_l2), slot=slot)
                    add(f"pv_panel_{index}_workspace_read",
                        read_work(self.ctx, partial, aiv=True,
                                  hbm=0 if ws_l2 else partial, l2=partial if ws_l2 else 0), slot=slot)
                else:
                    add(f"pv_panel_{index}_fixpipe", Work(resources={"fixpipe": partial}), slot=slot)
            offset += width
        if shipping_ub:
            add("pv_fixpipe", Work(resources={"fixpipe": _fixpipe_bytes(rect.m, c.v_head_dim)}))
        add("online_output", Work(resources={"vector": rect.m * c.v_head_dim}))
        return tuple(stages)

    def ranges(self, rect):
        spans = rect.tile_spans
        tiles = rect.tile_count
        splits = min(self.effective_splits, max(1, tiles))
        if self.ctx.requests:
            splits = min(splits, len(self.owners))
        if not spans:
            yield range(0, 0)
            return
        if len(spans) == 1:
            t_lo, _ = spans[0]
            for split in range(splits):
                yield range(t_lo + tiles * split // splits, t_lo + tiles * (split + 1) // splits)
            return
        all_tiles = tuple(t for lo, hi in spans for t in range(lo, hi))
        for split in range(splits):
            yield all_tiles[tiles * split // splits:tiles * (split + 1) // splits]

    def query_load(self, rect, tiles):
        c = self.contract
        width = c.head_dim * BF16_BYTES
        size = rect.m * width if tiles else 0
        if not size or not self.q_offsets:
            return AttentionPhase("query_load", read_work(self.ctx, size), ())
        group = c.num_heads // c.num_kv_heads
        base = self.q_offsets[rect.sequence] * c.num_heads
        if (self.ctx.layout or c.kv_layout) == "BNSD":
            stride = self.ctx.requests[rect.sequence].physical_q_len
            queries = self.sequences[rect.sequence][0]
            head, token = divmod(rect.row, queries)
            base += (rect.head * group + head) * stride + token
            block, jump = queries, stride
        else:
            token, head = divmod(rect.row, group)
            base += token * c.num_heads + rect.head * group + head
            token, block, jump = head, group, c.num_heads
        runs = []
        remaining = rect.m
        while remaining:
            rows = min(remaining, block - token)
            runs.append(AddressRun(base * width, rows, width, width))
            base += jump - token
            token = 0
            remaining -= rows
        transfer = AddressRunPlan(tuple(runs))
        work = read_work(self.ctx, transfer.aligned_copy_bytes(), hbm=transfer.dram_sector_and_service()[0])
        return AttentionPhase("query_load", work, ())

    def partial(self, rect, split, after=()):
        c = self.contract
        phase = AttentionPhase("normalize", Work(resources={"vector": rect.m * c.v_head_dim}), after)
        if not split:
            return (phase,)
        payload = rect.m * (c.v_head_dim + 2) * FP32_BYTES
        return (phase, AttentionPhase("split_write", Work(resources={"mte3": payload, "hbm": payload}),
                                      (phase.name,)))

    def finish(self, rect, n_splits):
        c = self.contract
        if n_splits > 1:
            partials = rect.m * n_splits
            payload = partials * (c.v_head_dim + 2) * FP32_BYTES
            yield AttentionPhase("split_read", read_work(self.ctx, payload, aiv=True), ("split_write",))
            yield AttentionPhase("split_merge", Work(resources={
                "vector": partials * (2 * c.v_head_dim + SOFTMAX_OPS),
            }), ("split_read",))
        result = rect.m * (c.v_head_dim * BF16_BYTES + (FP32_BYTES if c.return_lse else 0))
        yield AttentionPhase("output", Work(resources={"mte3": result, "hbm": result}),
                             ("split_merge" if n_splits > 1 else "normalize",))

    def _s2_program(self, context, rect, tiles: range, place, lower_phase):
        n_tiles = len(tiles)
        if n_tiles <= 0:
            return PipelineProgram(0, 1, lambda _: ())

        def tile(i, tiles=tiles, place=place):
            return tuple(lower_phase(phase, place) for phase in self.s2_phases(rect, tiles[i]))

        is_a35_decode = (not self.gm_workspace) and (rect.m <= ARCH35_DECODE_M_LIMIT)
        slots = (
            min(FIA_PRELOAD_SLOTS, self.panels.p_l1_slots) if is_a35_decode
            else (ARCH35_PIPELINE_SLOTS if not self.gm_workspace else ARCH22_WORKSPACE_SLOTS)
        )
        paged = self.contract.kv_layout in PAGED_LAYOUTS
        page = self.contract.block_size if paged else self.s2_tile
        has_live_mem = bool(self.contract.kv_tensor_id and (self.contract.block_table or self.ctx.requests))
        contiguous_tiles = (n_tiles <= 1) or (tiles[-1] - tiles[0] + 1 == n_tiles)
        reuse_uniform = True
        if (not has_live_mem and rect.rect_in_head > 0
                and self.kv_reuse_hit_frac > 0.0 and n_tiles > 2):
            prev_spans = (
                rect.prev_touched_spans
                if rect.prev_touched_spans
                else ((rect.prev_touched_start, rect.prev_touched_end),)
            )
            inner_start = tiles[1] * self.s2_tile
            inner_end = min((tiles[-2] + 1) * self.s2_tile, rect.kv)
            all_inside = any(p_lo <= inner_start and inner_end <= p_hi for p_lo, p_hi in prev_spans)
            all_outside = all(inner_end <= p_lo or inner_start >= p_hi for p_lo, p_hi in prev_spans)
            reuse_uniform = all_inside or all_outside
        uniform_interior = contiguous_tiles and (not has_live_mem) and reuse_uniform and (
            (not paged) or (page % self.s2_tile == 0 or self.s2_tile % page == 0)
        )
        if n_tiles > COMPACT_TILE_THRESHOLD and uniform_interior:
            if not self.gm_workspace and self.contract.panel_mode == "shipping":
                if is_a35_decode:
                    return _ShippingS2(n_tiles, min(FIA_PRELOAD_SLOTS, slots), tile)
                return _compact_s2_pipeline(context, n_tiles, min(n_tiles, slots), tile)
            return _compact_s2_pipeline(context, n_tiles, max(1, slots), tile)
        if min(n_tiles, slots) > 1 and not self.gm_workspace and self.contract.panel_mode == "shipping":
            if is_a35_decode:
                return _ShippingS2(n_tiles, min(FIA_PRELOAD_SLOTS, slots), tile)
            return PipelineProgram(n_tiles, min(n_tiles, slots), tile)
        if min(n_tiles, slots) > 1 and self.gm_workspace:
            return PipelineProgram(n_tiles, slots, tile)
        return PipelineProgram(n_tiles, 1, tile)

    def __call__(self, context):
        datapath = self.ctx.datapath
        places = tuple(_placement(context, pair, datapath.aic_count, datapath.aiv_count) for pair in self.owners)
        allocs = tuple(_allocations(context, self.buffers, pair, datapath.aic_count) for pair in self.owners)
        owners = len(places)
        hbm = context.resource_names.get("hbm")
        l2 = context.resource_names.get("l2")
        capacity = self.ctx.geometry.buffers.get("l2", 0) if l2 in context.rates else 0
        die = self.ctx.geometry.rank or 0

        ragged_kv = len({kv for _, kv in self.contract.sequences}) > 1
        if self.fd_segments:
            active_owners = max(1, min(owners, len(self.fd_segments)))
        elif self.row_ends[-1] > 0:
            splits_0 = 1 if ragged_kv else max(1, len(tuple(self.ranges(self.rectangle(0)))))
            active_owners = max(1, min(owners, self.row_ends[-1] * splits_0))
        else:
            active_owners = max(1, owners)
        cold_kv_caps = p01_cold_mte2_aic_cap(
            context, active_owners, include_hbm=True,
            hbm_only=(self.kv_reuse_hit_frac > 0.0),
        )
        cold_kv_caps = _bsh_strided_cold_caps(
            context, cold_kv_caps, self.contract, gm_workspace=self.gm_workspace,
            burst_bytes=self.burst_bytes, active_owners=active_owners, total_aics=datapath.aic_count,
        )
        paged_kv = self.contract.kv_layout in PAGED_LAYOUTS
        page_cmd_ms = (
            p03_fixpipe_setup_ms(context) if self.gm_workspace else p02_load_setup_ms(context)
        ) if paged_kv else 0.0
        scatter_open_bytes = (
            p01_hbm_scatter_row_open_bytes(context)
            if (paged_kv and not self.gm_workspace) else 0
        )
        group = self.contract.num_heads // self.contract.num_kv_heads
        max_q_m = max((q * group for q, _ in self.sequences), default=1)
        is_a35_decode = (not self.gm_workspace) and max_q_m <= ARCH35_DECODE_M_LIMIT
        a35_decode_fix_ms = (p03_fixpipe_setup_ms(context) + p13_mode4_ms(context)) if is_a35_decode else 0.0
        a35_nd2nz_setup_ms = p02_load_setup_ms(context) if is_a35_decode else 0.0

        def lower_phase(phase, place):
            if phase.name in {"k_page_cmd", "v_page_cmd"}:
                lat = float(phase.pages) * page_cmd_ms if (page_cmd_ms > 0.0 and phase.pages > 0) else 0.0
                prev = ("v_page_cmd",) if (phase.name == "k_page_cmd" and self.gm_workspace) else ()
                return Stage(phase.name, after=phase.after, previous=prev, latency=lat, keep_empty=True)
            if phase.name in {"k_nd2nz_issue", "v_nd2nz_issue"}:
                return Stage(phase.name, after=phase.after, latency=a35_nd2nz_setup_ms, keep_empty=True)
            if not (phase.memory and (capacity or self.ctx.admission_memory_view is not None)
                    and l2 in context.rates and hbm in context.rates):
                is_kv_load = phase.name.endswith("_load") and not phase.name.startswith("probability")
                st = _stage(context, phase, place)
                if self.gm_workspace and phase.name == "k_panel_0_load":
                    st = replace(st, previous=("k_panel_0_load",))
                b_hbm = phase.work.resources.get("hbm", 0)
                if cold_kv_caps and b_hbm > 0 and is_kv_load:
                    st = replace(st, caps={**st.caps, **cold_kv_caps})
                if is_kv_load and scatter_open_bytes > 0 and phase.pages > 0 and hbm and b_hbm > 0:
                    extra_hbm = phase.pages * scatter_open_bytes
                    base_cap = st.caps.get(hbm, context.rates[hbm])
                    st = replace(st, caps={**st.caps, hbm: base_cap * (b_hbm / (b_hbm + extra_hbm))})
                if a35_decode_fix_ms > 0.0 and phase.name in {"score_fixpipe", "pv_fixpipe"}:
                    st = replace(st, latency=max(st.latency, a35_decode_fix_ms))
                return st
            runs, rules, tensor = phase.memory
            pipe = context.resource_names.get("mte2_aic", context.resource_names.get("mte2"))
            units = place.get(pipe, (0,))
            if len(units) != 1:
                return _stage(context, phase, place)
            key = ReuseKey(die, f"{self.contract.kv_tensor_id}:{tensor}", self.contract.kv_version)
            pattern = MemoryAccessPattern(
                key.backing_allocation, key.tensor_version, die, units[0], "read", "mte2_aic",
                sum(run.rows * run.row_bytes for run in runs.runs), runs, reuse_key=key,
            )
            bus = HbmBusSpec(die, 1, 1, 0, row_r=hbm, ctr_r=hbm)
            bound = context.bind_access(pattern, hbm_bus=bus, burst_rules=rules)
            return Stage(phase.name, after=phase.after,
                         program=AttentionMemoryRead(pattern, bound, capacity, hbm, self.ctx),
                         metadata={"operand_slot": phase.slot} if phase.slot is not None else {})

        def build_branch(rect, tiles, n_splits, owner_idx):
            place = places[owner_idx]
            allocations = allocs[owner_idx]
            load = _stage(context, self.query_load(rect, tiles), place, acquire=allocations)
            loop = Stage("s2", after=(load.name,), program=self._s2_program(context, rect, tiles, place, lower_phase))
            partial = self.partial(rect, n_splits > 1, (loop.name,))
            commits = tuple(_stage(context, phase, place, free=allocations if phase is partial[-1] else ())
                            for phase in partial)
            return KernelProgram((load, loop, *commits))

        if self.fd_segments:
            return self._lower_fd_segments(context, places, allocs, owners, lower_phase, build_branch)

        def row(index, core, name="s1"):
            rect = self.rectangle(index)
            ranges = tuple(self.ranges(rect))
            branches = [
                build_branch(rect, tiles, len(ranges),
                             core + split if self.ctx.requests else (core + split) % owners)
                for split, tiles in enumerate(ranges)
            ]
            pre = Stage("partials", program=ParallelProgram(tuple(branches)))
            stages = [pre]
            for phase in self.finish(rect, len(ranges)):
                phase = AttentionPhase(phase.name, phase.work, (stages[-1].name,))
                stages.append(_stage(context, phase, places[core]))
            return Stage(name, program=KernelProgram(tuple(stages)))

        if self.ctx.requests:
            largest = max((ceil(kv / self.s2_tile) for _, kv in self.sequences), default=1)
            lanes = min(owners, self.effective_splits, max(1, largest))
            return attention_owner_waves(self.row_ends[-1], owners, lanes, row)
        if ragged_kv and self.effective_splits == 1 and self.row_ends[-1] > owners:
            rects_all = tuple(self.rectangle(i) for i in range(self.row_ends[-1]))
            core_slices = _weighted_contiguous_rect_partition(rects_all, owners)
            return ParallelProgram(tuple(
                PipelineProgram(len(indices), 1, lambda i, core=core, indices=indices: (row(indices[i], core),))
                for core, indices in enumerate(core_slices) if indices
            ))
        return ParallelProgram(tuple(
            PipelineProgram(len(indices), 1, lambda i, core=core, indices=indices: (row(indices[i], core),))
            for core in range(owners) if (indices := range(core, self.row_ends[-1], owners))
        ))

    def _lower_fd_segments(self, context, places, allocs, owners, lower_phase, build_branch):
        n_rects = self.row_ends[-1]
        rects = tuple(self.rectangle(i) for i in range(n_rects))
        segs_by_owner: list[list[_CoreSegment]] = [[] for _ in range(owners)]
        owner_of_rect: dict[int, int] = {}
        for seg in self.fd_segments:
            owner_of_rect.setdefault(seg.rect_idx, seg.owner_idx)
            segs_by_owner[seg.owner_idx].append(seg)

        def build_owner_program(owner_idx: int, segs: list[_CoreSegment]):
            if len(segs) == 1:
                s = segs[0]
                return build_branch(rects[s.rect_idx], range(s.tile_start, s.tile_end),
                                    self.rect_splits[s.rect_idx], owner_idx)
            place = places[owner_idx]
            allocations = allocs[owner_idx]
            q_work = sum(
                (self.query_load(rects[s.rect_idx], range(s.tile_start, s.tile_end)).work for s in segs),
                Work(),
            )
            load = _stage(context, AttentionPhase("query_load", q_work, ()), place, acquire=allocations)
            seg_stages: list[Stage] = [load]
            for idx_s, s in enumerate(segs):
                s2_prog = self._s2_program(context, rects[s.rect_idx], range(s.tile_start, s.tile_end), place, lower_phase)
                seg_stages.append(Stage(f"s2_{idx_s}", after=(seg_stages[-1].name,), program=s2_prog))
            commit_stages: list[Stage] = []
            prev_dep = (seg_stages[-1].name,)
            for idx_s, s in enumerate(segs):
                part = self.partial(rects[s.rect_idx], self.rect_splits[s.rect_idx] > 1, prev_dep)
                for p_idx, phase in enumerate(part):
                    is_last = (idx_s == len(segs) - 1) and (p_idx == len(part) - 1)
                    st_name = f"s{idx_s}_{phase.name}"
                    dep = prev_dep if p_idx == 0 else (commit_stages[-1].name,)
                    commit_stages.append(_stage(
                        context, AttentionPhase(st_name, phase.work, dep), place,
                        free=allocations if is_last else (),
                    ))
                prev_dep = (commit_stages[-1].name,)
            return KernelProgram((*seg_stages, *commit_stages))

        prod_programs = tuple(
            build_owner_program(o_idx, segs) for o_idx, segs in enumerate(segs_by_owner) if segs
        )
        fin_by_core: list[list[KernelProgram]] = [[] for _ in range(owners)]
        for r_idx, rect in enumerate(rects):
            fc = owner_of_rect.get(r_idx, r_idx % owners)
            f_stages: list[Stage] = []
            for phase in self.finish(rect, self.rect_splits[r_idx]):
                dep = (f_stages[-1].name,) if f_stages else ()
                f_stages.append(_stage(context, AttentionPhase(phase.name, phase.work, dep), places[fc]))
            fin_by_core[fc].append(KernelProgram(tuple(f_stages)))
        fin_programs = tuple(
            PipelineProgram(len(fps), 1, lambda i, fps=fps: (Stage("finish", program=fps[i]),))
            for fps in fin_by_core if fps
        )
        syncall_ms = p13_syncall_ms(context)
        if syncall_ms > 0.0:
            return KernelProgram((
                Stage("fd_partials", program=ParallelProgram(prod_programs)),
                Stage("fd_syncall", after=("fd_partials",), latency=syncall_ms, keep_empty=True),
                Stage("fd_merge", after=("fd_syncall",), program=ParallelProgram(fin_programs)),
            ))
        return KernelProgram((
            Stage("fd_partials", program=ParallelProgram(prod_programs)),
            Stage("fd_merge", after=("fd_partials",), program=ParallelProgram(fin_programs)),
        ))


def _visible_sequences(
    contract: AttentionContract, sequences: tuple[tuple[int, int], ...],
) -> tuple[tuple[int, int], ...]:
    """Bound per-sequence KV spans by selected_tokens, query spans, and sliding window for tiling/L2 math."""
    capped: list[tuple[int, int]] = []
    for idx, (q, kv) in enumerate(sequences):
        selected = kv if contract.selected_tokens is None else contract.selected_tokens
        spans = (
            contract.sequence_query_spans[idx]
            if idx < len(contract.sequence_query_spans) else ()
        )
        eff_kv = (
            min(kv, max((hi for _, hi in spans), default=0))
            if (contract.causal and spans) else kv
        )
        if not contract.window_size:
            win = eff_kv
        elif not spans:
            win = contract.window_size + max(0, q - 1)
        elif contract.causal:
            merged = _merge_intervals([
                (max(0, s_lo + 1 - contract.window_size), min(eff_kv, s_hi))
                for s_lo, s_hi in spans
            ])
            win = sum(hi - lo for lo, hi in merged)
        else:
            radius = contract.window_size - 1
            merged = _merge_intervals([
                (max(0, s_lo - radius), min(kv, s_hi + radius))
                for s_lo, s_hi in spans
            ])
            win = sum(hi - lo for lo, hi in merged)
        capped.append((q, min(eff_kv, selected, win)))
    return tuple(capped)


def _resolve_native_tiling(
    contract: AttentionContract, ctx: OperatorPlanningContext,
    sequences: tuple[tuple[int, int], ...], *, gm_workspace: bool,
    raw_sequences: tuple[tuple[int, int], ...] | None = None,
) -> CorePartition | None:
    if contract.panel_mode != "shipping" or contract.splits > 1 or contract.fia_template is not None:
        return None
    aic_count, paged = ctx.effective_aic_cores, contract.kv_layout in PAGED_LAYOUTS
    # Native FIA tilers (fia_tiling_nonquant_gqa.cpp:429) select row ordering
    # from qLayout, not KV cache storage layout.
    q_layout = str(ctx.layout or contract.kv_layout)
    # Right-aligned causal truncation (s2_last = s1_last + kv - q) is invalid
    # when selected_tokens/windowing shortened kv below the raw KV span; every
    # query tile still sees its bounded visible band and must stay live.
    base_seqs = raw_sequences if raw_sequences is not None else sequences
    shortened_by_selected = (
        contract.selected_tokens is not None
        and any(vkv < okv for (_, okv), (_, vkv) in zip(base_seqs, sequences))
    )
    shortened_below_q = any(
        vkv < q and okv >= q for (q, okv), (_, vkv) in zip(base_seqs, sequences)
    )
    tiling_causal = contract.causal and not shortened_by_selected and not shortened_below_q
    if gm_workspace:
        return resolve_arch22_fia_gqa_tiling(
            sequences=sequences, num_heads=contract.num_heads, num_kv_heads=contract.num_kv_heads,
            block_size=contract.block_size, paged=paged, aic_count=aic_count,
            layout=q_layout, causal=tiling_causal,
        )
    return resolve_arch35_fia_gqa_tiling(
        sequences=sequences, num_heads=contract.num_heads, num_kv_heads=contract.num_kv_heads,
        head_dim=contract.head_dim, v_head_dim=contract.v_head_dim, block_size=contract.block_size,
        paged=paged, aic_count=aic_count, l2_bytes=_l2_capacity_bytes(ctx),
        causal=tiling_causal, is_decode=(contract.phase == "decode"),
        layout=q_layout,
    )


def _compute_kv_l2_reuse(
    contract: AttentionContract, sequences: tuple[tuple[int, int], ...],
    m_tile: int, s2_tile: int, num_owners: int, l2_cap: int,
    ws_bytes: int, ws_l2_resident: bool, gm_workspace: bool,
) -> float:
    group = contract.num_heads // contract.num_kv_heads
    max_s1, max_s2 = max((q for q, _ in sequences), default=1), max((kv for _, kv in sequences), default=0)
    rects_per_head = ceil(max_s1 * group / m_tile)
    if rects_per_head <= 1 or l2_cap <= 0 or max_s2 <= 0:
        return 0.0
    l2_usable = max(0, l2_cap - (ws_bytes if ws_l2_resident else 0))
    kv_head_bytes = max_s2 * (contract.head_dim + contract.v_head_dim) * BF16_BYTES
    concurrent_heads = min(len(sequences) * contract.num_kv_heads, max(1, ceil(num_owners / rects_per_head)))
    wave_rects_per_head = min(rects_per_head, max(1, num_owners // concurrent_heads))
    pipe_slots = ARCH22_WORKSPACE_SLOTS if gm_workspace else ARCH35_PIPELINE_SLOTS
    lockstep_window = concurrent_heads * pipe_slots * s2_tile * (contract.head_dim + contract.v_head_dim) * BF16_BYTES
    in_wave_hit = 1.0 if lockstep_window <= l2_usable else 0.0
    cross_wave_hit = 1.0 if (concurrent_heads * kv_head_bytes) <= l2_usable else 0.0
    total_reuse_rects = max(1, rects_per_head - 1)
    cross_wave_firsts = max(0, ceil(rects_per_head / wave_rects_per_head) - 1)
    in_wave_reuses = max(0, total_reuse_rects - cross_wave_firsts)
    return (in_wave_reuses * in_wave_hit + cross_wave_firsts * cross_wave_hit) / float(total_reuse_rects)


def dense_fia(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext,
    sources: tuple[SourceRef, ...], *, gm_workspace: bool, block_table=None,
) -> OperatorPlan:
    """Issue one physical QK/Vec1/PV rectangle per S2 task with nested D panels."""
    if contract.quant_mode != QUANT_MODE_BF16 or contract.selection_lists or op.dtype not in {"bf16", "fp16"}:
        raise unmodeled("attention_fia_nonquant", "Bind a nonquant dense FIA tensor contract")
    if not contract.query_tokens:
        return make_plan(op, contract, ctx, (), sources)
    buffers = ()
    metadata = {"panel_mode": contract.panel_mode}
    if contract.panel_mode == "shipping":
        plan = shipping_fia_panel_plan(contract, ctx.compute_arch.arch_id)
    else:
        solution = solve_wide_head_tiling(contract, ctx.geometry)
        width = solution.tile_shape.k
        plan = FiaPanelPlan(panels(contract.head_dim, width), panels(contract.v_head_dim, width),
                            1, 2, 2, 1, 1, solution.depth, 1)
        buffers = tuple((peak.pool, peak.peak_bytes) for peak in solution.pool_peaks)
        ub_peaks = tuple(peak.peak_bytes for peak in solution.pool_peaks if peak.pool == "ub")
        metadata["tile_solution"] = {
            "algorithm": solution.algorithm, "panel_width": width,
            "ub_peak_bytes": max(ub_peaks, default=0),
            "rejections": tuple((r.pool, r.required_bytes, r.available_bytes) for r in solution.rejections),
        }
    metadata.update(asdict(plan), k_panels=len(plan.k_panels), v_panels=len(plan.v_panels))
    batch = min(contract.batch_size, contract.query_tokens)
    base, extra = divmod(contract.query_tokens, batch)
    sequences = contract.sequences or tuple((base + (i < extra), contract.selected_tokens) for i in range(batch))
    if block_table is None:
        block_table = dict(op.metadata).get("block_table")
    if block_table is not None:
        contract = replace(contract, block_table=block_table)
    if contract.block_table and (len(contract.block_table) != len(sequences)
                                 or any(len(row) < ceil(kv / contract.block_size)
                                        for row, (_, kv) in zip(contract.block_table, sequences))):
        raise ModelingError("attention_fia_block_table_coverage_invalid")
    group = contract.num_heads // contract.num_kv_heads
    visible_sequences = _visible_sequences(contract, sequences)
    native = _resolve_native_tiling(
        contract, ctx, visible_sequences, gm_workspace=gm_workspace, raw_sequences=sequences,
    )
    m_tile = native.m_tile if native is not None else FIA_TILE
    s2_tile = native.s2_tile if native is not None else FIA_TILE
    eff_splits = native.fd_splits_per_row if native is not None else contract.splits
    ends = tuple(accumulate(ceil(q * group / m_tile) * contract.num_kv_heads for q, _ in sequences))
    n_rects = ends[-1]
    desired_owners = native.used_cores if native is not None else (n_rects * eff_splits)
    owners = owner_pairs(ctx, max(1, desired_owners))
    default_burst = DAV_2201_HBM_BURST_BYTES if ctx.compute_arch.arch_id == ARCH22 else DAV_3510_HBM_BURST_BYTES
    burst = getattr(ctx.datapath.memory_access, "hbm_burst_granule_bytes", 0) or default_burst
    spans = planning_sequence_spans(ctx, contract.kv_layout)
    q_offsets = tuple(span[2] for span in spans)
    kv_offsets = tuple(span[3] for span in spans)
    strides = tuple(request.physical_kv_len if contract.kv_layout == "BNSD" else kv
                    for request, (_, kv) in zip(ctx.requests, sequences))
    l2_cap = _l2_capacity_bytes(ctx)
    max_s1 = max((q for q, _ in sequences), default=1)
    m_live = min(max_s1 * group, m_tile)
    ws_bytes = (
        ARCH22_WORKSPACE_SLOTS * len(owners)
        * (ARCH22_WS_SCORE_PROB_BYTES * m_live * s2_tile + ARCH22_WS_OUT_BYTES * m_live * contract.v_head_dim)
        if gm_workspace else 0
    )
    ws_l2_resident = gm_workspace and l2_cap > 0 and ws_bytes <= l2_cap
    kv_reuse_hit_frac = _compute_kv_l2_reuse(
        contract, visible_sequences, m_tile, s2_tile, len(owners), l2_cap, ws_bytes, ws_l2_resident, gm_workspace,
    )
    use_balanced_fd = (
        native is not None and native.fd_enabled and contract.splits == 1
        and (len(owners) > n_rects or n_rects % len(owners) != 0)
    )
    loop_seed = _FiaLoop(
        contract, plan, ctx, op.dtype, gm_workspace, sequences, ends, owners, buffers, burst,
        q_offsets, kv_offsets, strides,
        m_tile=m_tile, s2_tile=s2_tile, effective_splits=eff_splits,
        ws_l2_resident=ws_l2_resident, kv_reuse_hit_frac=kv_reuse_hit_frac,
    )
    rects = tuple(loop_seed.rectangle(i) for i in range(n_rects))
    fd_segments, rect_splits = _balanced_core_segments(rects, len(owners)) if use_balanced_fd else ((), ())
    if native is not None:
        metadata.update(
            host_m_tile=m_tile, host_s2_tile=s2_tile, host_used_cores=len(owners),
            host_fd_splits=eff_splits, host_s1_out_split=native.s1_out_split,
            host_pfa_merge=native.pfa_merge, ws_l2_resident=ws_l2_resident,
            kv_reuse_hit_frac=round(kv_reuse_hit_frac, 4),
        )
    loop = replace(loop_seed, fd_segments=fd_segments, rect_splits=rect_splits)
    queries, s2, partials, outputs = {}, {}, {}, {}
    page = contract.block_size if contract.kv_layout in PAGED_LAYOUTS else s2_tile
    period = page // gcd(page, s2_tile)
    tiles = 0

    def collect(bucket, phase, repeats=1):
        work = phase.work.scaled(repeats) if repeats != 1 else phase.work
        old = bucket.get(phase.name)
        bucket[phase.name] = replace(phase, work=work + old.work if old else work)

    if fd_segments:
        segs_by_rect: list[list[range]] = [[] for _ in range(n_rects)]
        for seg in fd_segments:
            segs_by_rect[seg.rect_idx].append(range(seg.tile_start, seg.tile_end))
        rect_ranges = tuple(tuple(rs) for rs in segs_by_rect)
    else:
        rect_ranges = tuple(tuple(loop.ranges(rect)) for rect in rects)

    for index, rect in enumerate(rects):
        ranges = rect_ranges[index]
        for span in ranges:
            collect(queries, loop.query_load(rect, span))
            for phase in loop.partial(rect, len(ranges) > 1):
                collect(partials, phase)
        prev_spans = (
            rect.prev_touched_spans
            if rect.prev_touched_spans
            else ((rect.prev_touched_start, rect.prev_touched_end),)
        )
        for t_lo, t_hi in rect.tile_spans:
            tiles += t_hi - t_lo
            full_end = min(t_hi, rect.kv // s2_tile)
            non_periodic = bool(contract.block_table) or (
                rect.rect_in_head > 0 and kv_reuse_hit_frac > 0.0
                and not (
                    any(p_lo <= t_lo * s2_tile and full_end * s2_tile <= p_hi for p_lo, p_hi in prev_spans)
                    or all(full_end * s2_tile <= p_lo or t_lo * s2_tile >= p_hi for p_lo, p_hi in prev_spans)
                )
            )
            periodic_end = full_end if non_periodic else min(full_end, t_lo + period)
            for tile in range(t_lo, periodic_end):
                for phase in loop.s2_phases(rect, tile):
                    collect(s2, phase, 1 if non_periodic else ceil((full_end - tile) / period))
            if t_lo <= full_end < t_hi:
                for phase in loop.s2_phases(rect, full_end):
                    collect(s2, phase)
        for phase in loop.finish(rect, len(ranges)):
            collect(outputs if phase.name == "output" else partials, phase)

    phases = (*queries.values(),
              *(phase if phase.after else replace(phase, after=("query_load",)) for phase in s2.values()),
              replace(partials.pop("normalize"), after=(next(reversed(s2), "query_load"),)),
              *partials.values(),
              replace(outputs["output"], after=(next(reversed(partials), "normalize"),)))
    return make_plan(op, contract, ctx, phases, sources, metadata={**metadata, "s2_tiles": tiles},
                     program=replace(loop, phases=phases))
