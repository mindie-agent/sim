"""Executable fused inference attention rectangles and panel loops.

[Role] Lower FIA S1 rectangles and S2 tiles into owner-local programs.
[Invariants] Internal Cube panels conserve useful work. FixPipe and cross-core
  dependencies follow the selected FIA template's publication boundary.
[Timeouts & Locks] Finite tile traversal and slot allocation without locks.
[Side-Effect Policy] Pure planning and program construction.
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from itertools import accumulate
from math import ceil, gcd

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...runtime.memory_physics import AddressRun, AddressRunPlan
from ..contracts import OperatorPlanningContext
from .common import (
    BF16_BYTES, FIA_TILE, FP32_BYTES, INDEX_BYTES, PAGED_LAYOUTS, QUANT_MODE_BF16,
    SOFTMAX_OPS, TOKEN_MAJOR_LAYOUTS, AttentionContract, AttentionPhase, FiaPanelPlan,
    _allocations, _placement, _stage, bsh_kv_access_plan, make_plan, owner_pairs, panels,
    read_work, shipping_fia_panel_plan, solve_wide_head_tiling, unmodeled,
)

FIXPIPE_ROW_ALIGNMENT = 2
FIXPIPE_COLUMN_ALIGNMENT = 8


def _fixpipe_bytes(rows: int, columns: int) -> int:
    m = ceil(rows / FIXPIPE_ROW_ALIGNMENT) * FIXPIPE_ROW_ALIGNMENT
    n = ceil(columns / FIXPIPE_COLUMN_ALIGNMENT) * FIXPIPE_COLUMN_ALIGNMENT
    return m * n * FP32_BYTES


@dataclass(frozen=True, slots=True)
class _FiaRect:
    m: int
    head: int
    kv: int
    first: int
    end: int


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
    phases: tuple[AttentionPhase, ...] = ()

    def rectangle(self, index):
        c = self.contract
        group = c.num_heads // c.num_kv_heads
        sequence = bisect_right(self.row_ends, index)
        queries, kv = self.sequences[sequence]
        head, tile = divmod(index - (self.row_ends[sequence - 1] if sequence else 0), ceil(queries * group / FIA_TILE))
        row = tile * FIA_TILE
        m = min(FIA_TILE, queries * group - row)
        visible = min(kv, kv if c.selected_tokens is None else c.selected_tokens, c.window_size or kv)
        if c.causal and c.sequences:
            if c.kv_layout in TOKEN_MAJOR_LAYOUTS:
                first, last = row // group, (row + m - 1) // group
            elif row // queries == (row + m - 1) // queries:
                first, last = row % queries, (row + m - 1) % queries
            else:
                first, last = 0, queries - 1
            start = max(0, kv - queries + first + 1 - visible)
            end = max(0, kv - queries + last + 1)
        else:
            start, end = kv - visible, kv
        return _FiaRect(m, head, kv, start // FIA_TILE, ceil(end / FIA_TILE) if visible else start // FIA_TILE)

    def _kv(self, rect, start, n, dim, width, offset):
        c = self.contract
        page = c.block_size if c.kv_layout in PAGED_LAYOUTS else rect.kv
        hbm = copy = 0
        while n:
            begin = start % page
            extent = min(n, page - begin)
            base = bsh_kv_access_plan(
                c.num_kv_heads, dim, BF16_BYTES, c.kv_layout,
                rows=extent, first_head=rect.head, block_size=page, tokens_per_head=rect.kv,
            ).runs[0]
            run = AddressRun(base.base_byte + begin * base.stride_bytes + offset * BF16_BYTES,
                             extent, width * BF16_BYTES, base.stride_bytes, base.layout)
            plan = AddressRunPlan((run,))
            copy += plan.aligned_copy_bytes()
            hbm += plan.dram_sector_and_service()[0]
            n, start = n - extent, start + extent
        return read_work(self.ctx, copy, hbm=hbm)

    def s2_phases(self, rect, tile):
        c, p = self.contract, self.panels
        start = tile * FIA_TILE
        n = min(FIA_TILE, rect.kv - start)
        pairs = rect.m * n
        shipping_ub = not self.gm_workspace and c.panel_mode == "shipping"
        score = _fixpipe_bytes(rect.m, n) if shipping_ub else pairs * FP32_BYTES
        probability = pairs * BF16_BYTES
        stages = []

        def add(name, work, after=None, slot=None):
            deps = (stages[-1].name,) if after is None and stages else after or ()
            stages.append(AttentionPhase(name, work, deps, slot))

        if c.kv_layout in PAGED_LAYOUTS:
            pages = ceil((start % c.block_size + n) / c.block_size)
            add("block_table", read_work(self.ctx, pages * INDEX_BYTES, aiv=True))
        k_root = (stages[-1].name,) if stages else ()
        offset = 0
        depth = min(p.k_l1_slots, p.l0b_slots)
        for index, width in enumerate(p.k_panels):
            load, compute = f"k_panel_{index}_load", f"qk_panel_{index}_accumulate"
            slot = index % p.l0b_slots
            ready = (f"qk_panel_{index - depth}_accumulate",) if index >= depth else k_root
            previous = (f"qk_panel_{index - 1}_accumulate",) if index else ()
            add(load, self._kv(rect, start, n, c.head_dim, width, offset), ready, slot)
            add(compute, Work({self.dtype: 2 * pairs * width},
                              {"mte1": (rect.m + n) * width * BF16_BYTES}), (load, *previous), slot)
            offset += width
        add("score_fixpipe", Work(resources={"fixpipe": score, "hbm": score if self.gm_workspace else 0}))
        if self.gm_workspace:
            add("score_workspace_read", read_work(self.ctx, score, aiv=True))
        add("online_softmax", Work(resources={"vector": SOFTMAX_OPS * pairs}))
        add("probability_transfer", Work(resources={"mte3": probability,
                                                   "hbm": probability if self.gm_workspace else 0}))
        if self.gm_workspace:
            add("probability_workspace_read", read_work(self.ctx, probability))
        v_root = (stages[-1].name,)
        offset = 0
        depth = min(p.v_l1_slots, p.l0b_slots)
        for index, width in enumerate(p.v_panels):
            load, compute = f"v_panel_{index}_load", f"pv_panel_{index}"
            slot = index % p.l0b_slots
            ready = (f"pv_panel_{index - depth}",) if index >= depth else v_root
            previous = stages[-1].name
            add(load, self._kv(rect, start, n, c.v_head_dim, width, offset), ready, slot)
            add(compute, Work({self.dtype: 2 * pairs * width},
                              {"mte1": probability + n * width * BF16_BYTES}), (load, previous), slot)
            if not shipping_ub:
                partial = rect.m * width * FP32_BYTES
                add(f"pv_panel_{index}_fixpipe",
                    Work(resources={"fixpipe": partial, "hbm": partial if self.gm_workspace else 0}), slot=slot)
                if self.gm_workspace:
                    add(f"pv_panel_{index}_workspace_read", read_work(self.ctx, partial, aiv=True))
            offset += width
        if shipping_ub:
            add("pv_fixpipe", Work(resources={"fixpipe": _fixpipe_bytes(rect.m, c.v_head_dim)}))
        add("online_output", Work(resources={"vector": rect.m * c.v_head_dim}))
        return tuple(stages)

    def ranges(self, rect):
        tiles = rect.end - rect.first
        splits = min(self.contract.splits, max(1, tiles))
        for split in range(splits):
            yield range(rect.first + tiles * split // splits, rect.first + tiles * (split + 1) // splits)

    def query_load(self, rect, tiles):
        size = rect.m * self.contract.head_dim * BF16_BYTES if tiles else 0
        return AttentionPhase("query_load", read_work(self.ctx, size), ())

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

    def __call__(self, context):
        datapath = self.ctx.datapath
        places = tuple(_placement(context, pair, datapath.aic_count, datapath.aiv_count) for pair in self.owners)
        allocs = tuple(_allocations(context, self.buffers, pair, datapath.aic_count) for pair in self.owners)
        owners = len(places)

        def row(index, core):
            rect = self.rectangle(index)
            ranges = tuple(self.ranges(rect))
            branches = []
            for split, tiles in enumerate(ranges):
                place = places[(core + split) % owners]
                allocations = allocs[(core + split) % owners]

                def tile(i, tiles=tiles, place=place):
                    return tuple(_stage(context, phase, place) for phase in self.s2_phases(rect, tiles[i]))

                load = _stage(context, self.query_load(rect, tiles), place, acquire=allocations)
                loop = Stage("s2", after=(load.name,), program=PipelineProgram(len(tiles), 1, tile))
                partial = self.partial(rect, len(ranges) > 1, (loop.name,))
                commits = tuple(_stage(context, phase, place, free=allocations if phase is partial[-1] else ())
                                for phase in partial)
                branches.append(KernelProgram((load, loop, *commits)))
            pre = Stage("partials", program=ParallelProgram(tuple(branches)))
            stages = [pre]
            for phase in self.finish(rect, len(ranges)):
                # The branch join publishes all split workspace or the normalized result.
                phase = AttentionPhase(phase.name, phase.work, (stages[-1].name,))
                stages.append(_stage(context, phase, places[core]))
            return (Stage("s1", program=KernelProgram(tuple(stages))),)

        return ParallelProgram(tuple(
            PipelineProgram(len(indices), 1, lambda i, core=core, indices=indices: row(indices[i], core))
            for core in range(owners) if (indices := range(core, self.row_ends[-1], owners))
        ))


def dense_fia(
    op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext,
    sources: tuple[SourceRef, ...], *, gm_workspace: bool,
) -> OperatorPlan:
    """Issue one physical QK/Vec1/PV rectangle per S2 task with nested D panels."""
    if contract.quant_mode != QUANT_MODE_BF16 or contract.selection_lists or op.dtype not in {"bf16", "fp16"}:
        raise unmodeled("attention_fia_nonquant", "Bind a nonquant dense FIA tensor contract")
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
        metadata["tile_solution"] = {
            "algorithm": solution.algorithm, "panel_width": width,
            "rejections": tuple((r.pool, r.required_bytes, r.available_bytes) for r in solution.rejections),
        }
    metadata.update(asdict(plan), k_panels=len(plan.k_panels), v_panels=len(plan.v_panels))
    batch = min(contract.batch_size, contract.query_tokens)
    base, extra = divmod(contract.query_tokens, batch)
    sequences = contract.sequences or tuple((base + (i < extra), contract.selected_tokens) for i in range(batch))
    group = contract.num_heads // contract.num_kv_heads
    ends = tuple(accumulate(ceil(q * group / FIA_TILE) * contract.num_kv_heads for q, _ in sequences))
    owners = owner_pairs(ctx, ends[-1] * contract.splits)
    loop = _FiaLoop(contract, plan, ctx, op.dtype, gm_workspace, sequences, ends, owners, buffers)
    queries, s2, partials, outputs = {}, {}, {}, {}
    page = contract.block_size if contract.kv_layout in PAGED_LAYOUTS else FIA_TILE
    period = page // gcd(page, FIA_TILE)
    tiles = 0

    def collect(bucket, phase, repeats=1):
        work = phase.work.scaled(repeats) if repeats != 1 else phase.work
        old = bucket.get(phase.name)
        bucket[phase.name] = replace(phase, work=work + old.work if old else work)

    for index in range(ends[-1]):
        rect = loop.rectangle(index)
        ranges = tuple(loop.ranges(rect))
        tiles += rect.end - rect.first
        for span in ranges:
            collect(queries, loop.query_load(rect, span))
            for phase in loop.partial(rect, len(ranges) > 1):
                collect(partials, phase)
        full_end = min(rect.end, rect.kv // FIA_TILE)
        # Whole tiles repeat once their page offset repeats. The tail retains its actual width.
        for tile in range(rect.first, min(full_end, rect.first + period)):
            for phase in loop.s2_phases(rect, tile):
                collect(s2, phase, ceil((full_end - tile) / period))
        if rect.first <= full_end < rect.end:
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
