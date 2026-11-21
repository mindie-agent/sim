"""Physical work and finite execution programs for paged MLA.

[Role] Lower tensor extents into owner-local stages and runtime credits.
[Invariants] Keep logical bytes, copy bytes, and DRAM transactions distinct.
[Timeouts & Locks] Stream tiles in a bounded execution window. No locks.
[Side-Effect Policy] Build immutable plans and runtime stages without I/O.
"""
from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from itertools import accumulate, chain, groupby
from math import lcm

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...program import KernelProgram, PipelineProgram, Program, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.memory_physics import AddressRun, AddressRunPlan, ByteLedger, MemoryAccessPattern
from ...runtime.ring_buffer import RingBuffer, lower_generation
from ..contracts import OperatorPlanningContext
from .common import (
    ACLNN_MQS_DATA_BYTES, ACLNN_MQS_INLINE_ROW_BYTES, ACLNN_MQS_SCALE_BYTES, ACLNN_MQS_SEPARATE_SCALE_MODE,
    DSL_MQS_ABI, FAMILY, FP32_BYTES, INDEX_BYTES, MQS_ABI, MQS_COMPRESSED_ROW_BYTES,
    MQS_ORIGINAL_ROW_BYTES, MQS_PAGED_LAYOUT, NZ_BLOCK_BYTES, PagedMlaContract, _query_row, align_bytes,
    capability_gap, ceil_div, latent_record_bytes,
)

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
RowCuts = dict[tuple[int, int], tuple[int, ...]]


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
    @property
    def phases(self) -> tuple[str, ...]:
        return tuple(chain.from_iterable(self.clocks))

@dataclass(slots=True)
class _Demand:
    resources: dict[str, float] = field(default_factory=dict)
    flops: int = 0
    metadata: dict[str, int | float | str] = field(default_factory=dict)

    def add(self, resource: str, amount: int | float) -> None:
        if amount:
            self.resources[resource] = self.resources.get(resource, 0) + amount
    def transfer(self, tensor: str, pipe: str, nbytes: int, plan: AddressRunPlan, *,
                 aligned_copy_bytes: int | None = None) -> None:
        if not nbytes:
            return
        direction = "read" if pipe.startswith("mte2") else "write"
        access = MemoryAccessPattern(tensor, 0, 0, 0, direction, pipe, nbytes, plan)
        aligned = access.runs.aligned_copy_bytes() if aligned_copy_bytes is None else aligned_copy_bytes
        sectors, dram = access.runs.dram_sector_and_service()
        ledger = ByteLedger(access.logical_bytes, aligned, dram, 0, aligned, sectors)
        self.add(pipe, ledger.aligned_transfer_bytes)
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

    def linear(self, tensor: str, pipe: str, nbytes: int, offset: int = 0) -> None:
        self.transfer(tensor, pipe, nbytes, AddressRunPlan((AddressRun(offset, 1, nbytes, nbytes),)))

    def atomic_add(self, tensor: str, pipe: str, nbytes: int, offset: int) -> None:
        self.linear(tensor, pipe, nbytes, offset)
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

def _kv_runs(c: PagedMlaContract, spans, width: int, stride: int, table, layout: str) -> AddressRunPlan:
    runs = []
    for page, offset, rows in spans:
        physical_page = table[page] if table else page
        if layout in ("PA_BLK_BSND", MQS_PAGED_LAYOUT):
            base = (physical_page * c.page_size + offset) * stride
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
             dqk_bytes: int, dv_bytes: int, *, value: bool = False) -> None:
    if not c.mixed_quant:
        table = c.block_table[seq] if c.block_table else ()
        tensor, width = ("value", dv_bytes) if value else ("key", dqk_bytes)
        stride = dqk_bytes if c.latent_alias else width
        for page, offset, rows in _spans(start, keys, c.page_size):
            plan = _kv_runs(c, ((page, offset, rows),), width, stride, table, c.kv_layout)
            demand.transfer(tensor, "mte2_aic", rows * width, plan)
        first_page = start // c.page_size
        pages = ceil_div(start + keys, c.page_size) - first_page
        demand.linear(f"{tensor}_pages", "mte2_aic", INDEX_BYTES * pages, INDEX_BYTES * first_page)
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
                run = AddressRun(base * packed, rows, packed, packed, c.kv_layout)
                demand.transfer(branch, "mte2_aiv", rows * packed, AddressRunPlan((run,)))
        demand.linear(f"{branch}_indices", "mte2_aiv", count * INDEX_BYTES if lists else 0,
                      offset * INDEX_BYTES)

def _split_tiles(tiles: int, splits: int, split: int) -> tuple[int, int]:
    width, extra = divmod(tiles, splits)
    return split * width + min(split, extra), width + (split < extra)

def _merge_ops(c: PagedMlaContract, rows: int, joins: int) -> int:
    return rows * joins * (SOFTMAX_MERGE_SCALAR_OPS + WEIGHTED_MERGE_OPS_PER_VALUE * c.dv)

@dataclass(frozen=True, slots=True)
class _PagedMlaProgramBuilder:
    contract: PagedMlaContract
    layout: PipelineLayout
    dtype: str
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    row_cuts: RowCuts
    def _demand(self, step: str, live: int, *, seq: int = 0, query: int = 0, heads: int = 0,
                start: int = 0, keys: int = 0, first: bool = False, last: bool = False,
                partials: int = 0, mm2_offset: int = 0) -> _Demand:
        c, layout = self.contract, self.layout
        elem = int(dtype_bytes(self.dtype))
        qk_width, v_width = c.dqk * elem, c.dv * elem
        scores, probs, pv = heads * keys * FP32_BYTES, heads * keys * elem, heads * c.dv * FP32_BYTES
        state_heads = heads if c.abi == DSL_MQS_ABI else live
        state = state_heads * (c.dv + SOFTMAX_STATE_SCALARS) * FP32_BYTES
        d = _Demand()
        atomic = layout.gm_workspace and not first
        if step == "sink_load":
            d.linear("sinks", "mte2_aiv", c.physical_heads * FP32_BYTES)
        elif step == "sink_init" and ((first and start == 0) or not partials):
            d.add("vector", heads * (c.dv + SOFTMAX_STATE_SCALARS))
        elif step == "q_load" and (layout.gm_workspace or first):
            d.linear("query", "mte2_aic", heads * qk_width)
        elif step == "kv_load":
            _read_kv(d, c, seq, query, start, keys, qk_width, v_width)
        elif step == "dequant" and c.mixed_quant:
            ori, _ = c.bank_counts(seq, query)
            fp8_keys = keys if c.abi == MQS_ABI else max(0, min(keys, ori - start))
            d.add("vector", c.dqk * (fp8_keys * FP8_DEQUANT_OPS_PER_VALUE
                                    + (keys - fp8_keys) * FP4_DEQUANT_OPS_PER_VALUE))
        elif step == "kv_to_l1" and c.mixed_quant:
            pipe, alignment = layout.probability_output
            d.add(pipe, align_bytes(keys * qk_width, alignment))
        elif step == "qk_operands":
            d.add("mte1", (heads + keys) * qk_width)
        elif step == "qk":
            d.flops = 2 * heads * keys * c.dqk
        elif step in {"score_store", "probability", "partial"}:
            tensor, (pipe, alignment), amount = {
                "score_store": ("mm1", layout.matrix_output, scores),
                "probability": ("vec1", layout.probability_output, probs),
                "partial": ("mm2", layout.matrix_output, pv),
            }[step]
            if layout.gm_workspace:
                if step == "partial" and atomic:
                    d.atomic_add(tensor, pipe, amount, mm2_offset)
                else:
                    d.linear(tensor, pipe, amount, mm2_offset if step == "partial" else 0)
            else:
                if step == "score_store" and layout.pv_panel_width:
                    amount = fixpipe_rectangle_bytes(heads, keys)
                d.add(pipe, align_bytes(amount, alignment))
        elif step == "score_read" and layout.gm_workspace:
            d.linear("mm1", "mte2_aiv", scores)
        elif step == "softmax":
            d.add("vector", heads * (SOFTMAX_OPS_PER_KEY * keys - 2))
        elif step == "pv_load":
            if layout.gm_workspace:
                d.linear("vec1", "mte2_aic", probs)
            if not layout.resident_latent:
                _read_kv(d, c, seq, query, start, keys, qk_width, v_width, value=True)
        elif step == "pv_operands":
            d.add("mte1", probs + keys * v_width)
        elif step in {"pv", "pv0", "pv1"}:
            columns = layout.pv_panel_width if step != "pv" else c.dv
            d.flops = 2 * heads * keys * columns
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
            d.atomic_add("mm2", "mte3", pv, mm2_offset)
        elif step == "output_read" and last:
            d.linear("mm2", "mte2_aiv", pv, mm2_offset)
        elif step == "output_update" and not layout.gm_workspace and not first:
            d.add("vector", _merge_ops(c, heads, 1))
        elif step == "split_write" and last and partials > 1:
            d.linear("vec2_split_state", "mte3", state)
        elif step == "merge_read":
            plan = AddressRunPlan((AddressRun(0, partials, state, align_bytes(state)),))
            d.transfer("vec2_split_state", "mte2_aiv", partials * state, plan)
        elif step == "merge":
            d.add("vector", _merge_ops(c, live, max(0, partials - 1)))
        elif step == "normalize" and partials:
            d.add("vector", live * (c.dv + (2 if c.return_lse else 0)))
        elif step == "store":
            d.linear("output", "mte3", live * v_width)
            if c.return_lse:
                d.linear("lse", "mte3", live * FP32_BYTES)
        return d
    def _counts(self, seq: int, query: int) -> tuple[int, int]:
        c, width = self.contract, self.layout.key_tile
        if c.abi == DSL_MQS_ABI:
            ori, cmp = c.bank_counts(seq, query)
            tiles = ceil_div(ori, width) + ceil_div(cmp, width)
            return tiles, 1 + len(self.row_cuts.get((seq, query), ())) if tiles else 0
        tiles = ceil_div(c.sequences[seq][1], width)
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
        return start, min(width, c.sequences[seq][1] - start)

    def _tile_demand(self, step, seq, query, heads, live, index, offset, count, partials, mm2_offset=0):
        start, keys = self._tile_extent(seq, query, offset + index)
        return self._demand(step, live, seq=seq, query=query, heads=heads, start=start, keys=keys,
                            first=index == 0, last=index + 1 == count, partials=partials,
                            mm2_offset=mm2_offset)

    def _finish_steps(self, partials):
        merge = ("merge_read", "merge") if partials > 1 else ()
        if self.contract.sinks and not partials:
            return ("sink_init", "normalize", "store")
        return (*merge, "normalize", "store")

    def _distribution(self):
        c = self.contract
        head_groups = ceil_div(c.physical_heads, self.layout.head_tile)
        prefix = tuple(accumulate((q * head_groups for q, _ in c.sequences), initial=0))
        most_splits = max((n for _, _, n in self._group_ranges(head_groups, prefix)), default=0)
        lanes = min(len(self.owners), max(1, most_splits))
        return head_groups, prefix, lanes, len(self.owners) // lanes

    def _group_ranges(self, head_groups, prefix):
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

    def totals(self) -> dict[str, Work]:
        c, layout = self.contract, self.layout
        totals: dict[str, Work] = {}
        if c.sinks:
            sink_owners = self._sink_owners(*self._distribution())
            totals["sink_load"] = self._demand("sink_load", 0).work(self.dtype).scaled(len(sink_owners))
        phases = layout.phases
        for seq in range(len(c.sequences)):
            for query, repeat in self._query_runs(seq):
                if not repeat:
                    continue
                tiles, partials = self._counts(seq, query)
                for head_start in range(0, c.physical_heads, layout.head_tile):
                    heads = min(layout.head_tile, c.physical_heads - head_start)
                    live = min(heads, max(0, c.logical_heads - head_start))
                    for split in range(partials):
                        offset, count = self._split_extent(seq, query, tiles, split)
                        for i in range(count):
                            for step in phases:
                                demand = self._tile_demand(step, seq, query, heads, live, i, offset, count, partials)
                                totals[step] = totals.get(step, Work()) + demand.work(self.dtype).scaled(repeat)
                    for step in self._finish_steps(partials):
                        work = self._demand(step, live, heads=heads, partials=partials).work(self.dtype)
                        totals[step] = totals.get(step, Work()) + work.scaled(repeat)
        return totals

    def _stage(self, context, demand, pair, name, *, after=(), previous=(), metadata=None):
        aic, aivs = pair
        work = demand.work(self.dtype)
        roles = (*work.resources, *(("cube",) if work.matrix else ()))
        placement = {}
        for role in roles:
            if role != "hbm":
                placement.update(context.placement(role, aivs if role in AIV_ROLES else (aic,)))
        return Stage(name, context.bind(work), after=after, previous=previous, placement=placement,
                     coupled="hbm" in work.resources,
                     metadata={**demand.metadata, **(metadata or {})})

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

    def _split_program(self, ctx, seq, query, heads, live, pair, offset, count, partials, interval):
        layout = self.layout
        aivs = pair[1]
        aiv_owners = ",".join(map(str, aivs)) if layout.pv_panel_width else ""
        pair = (pair[0], pair[1][:heads])
        mm2_slot = interval % FIA_WORKSPACE_DEPTH
        mm2_offset = mm2_slot * align_bytes(layout.head_tile * self.contract.dv * FP32_BYTES)
        phases = layout.phases
        delays = {step: delay for delay, steps in enumerate(layout.clocks) for step in steps}
        period = lcm(len(layout.clocks), *(depth for _, depth, _, _ in layout.rings))
        credits = {}
        for region, depth, producer, consumer in layout.rings:
            ring = RingBuffer(owner=pair[0], allocation=region, depth=depth)
            for k in range(depth + 1):
                generation = lower_generation(ring, k, producer, (consumer,))
            credit_index = next(g.index for g in ring.generations if g.last_use == generation.credit)
            credits.setdefault(producer, []).append((region, consumer, generation.index - credit_index))

        def name(step, index):
            return f"{step}:{index % period}"

        def cycle(clock):
            stages = []
            for position, step in enumerate(phases):
                index = clock - delays[step]
                if not 0 <= index < count:
                    continue
                deps, prev = [], []
                def depend(other, tile):
                    if 0 <= tile < count:
                        target = deps if tile + delays[other] == clock else prev
                        target.append(name(other, tile))
                if step in {"release0", "release1"}:
                    depend(f"partial{step[-1]}", index)
                    if step == "release0":
                        depend("release1", index - 1)
                    else:
                        depend("release0", index)
                elif position:
                    depend(phases[position - 1], index)
                if step in {"softmax", "output_update", "qk", "pv"}:
                    depend(step, index - 1)
                if layout.gm_workspace:
                    for peer, delta in FIA_ISSUE_ORDER.get(step, ()):
                        depend(peer, index + delta)
                if step == "atomic_rescale":
                    depend("partial", index - 1)
                if step == "pv0":
                    depend("partial1", index - 1)
                metadata = {"kv_tile": offset + index, "clock": clock}
                if layout.gm_workspace:
                    metadata.update(task_record_idx=index % len(layout.clocks), workspace_buf_idx=index % FIA_WORKSPACE_DEPTH)
                    if step in {"partial", "atomic_rescale", "atomic_write", "output_read"}:
                        metadata.update(bn2_idx_in_cur_core=interval, mm2_slot=mm2_slot, mm2_offset_bytes=mm2_offset)
                if layout.pv_panel_width and step in {"pv0", "pv1", "partial0", "partial1", "release0", "release1"}:
                    panel = int(step[-1])
                    metadata.update(mm2_slot=panel, mm2_generation=index, aiv_owners=aiv_owners,
                                    column_start=panel * layout.pv_panel_width, columns=layout.pv_panel_width)
                for region, consumer, depth in credits.get(step, ()):
                    depend(consumer, index - depth)
                    metadata[f"{region}_slot"] = index % depth
                    metadata[f"{region}_generation"] = index // depth
                demand = self._tile_demand(step, seq, query, heads, live, index, offset, count, partials, mm2_offset)
                edges = dict(after=tuple(dict.fromkeys(deps)), previous=tuple(dict.fromkeys(prev)))
                if step in {"release0", "release1"}:
                    stages.append(Stage(name(step, index), program=self._panel_release(ctx, demand, aivs, heads),
                                        metadata=metadata, **edges))
                else:
                    stages.append(self._stage(ctx, demand, pair, name(step, index), metadata=metadata, **edges))
            by_name = {stage.name: stage for stage in stages}
            order = TopologicalSorter({stage.name: stage.after for stage in stages})
            return tuple(by_name[key] for key in order.static_order())
        return PipelineProgram(count + len(layout.clocks) - 1, len(layout.clocks), cycle)

    def __call__(self, context: RecipeContext) -> Program:
        c, layout = self.contract, self.layout
        head_groups, prefix, lanes, groups_per_wave = self._distribution()
        if not prefix[-1]:
            return KernelProgram(())

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
            query, head_group = divmod(group_index - prefix[seq], head_groups)
            head_start = head_group * layout.head_tile
            heads = min(layout.head_tile, c.physical_heads - head_start)
            live = min(heads, max(0, c.logical_heads - head_start))
            tiles, partials = self._counts(seq, query)

            def wave(index):
                for local in range(min(lanes, partials - index * lanes)):
                    split = index * lanes + local
                    offset, count = self._split_extent(seq, query, tiles, split)
                    program = self._split_program(
                        context, seq, query, heads, live, self.owners[owner_start + local], offset, count, partials,
                        owner_interval(group_index, local) + index,
                    )
                    yield Stage(f"split{local}", program=program, metadata={"split": split})

            stages = [Stage("splits", program=PipelineProgram(ceil_div(partials, lanes), 1, wave))]
            for step in self._finish_steps(partials):
                stages.append(self._stage(
                    context, self._demand(step, live, heads=heads, partials=partials), self.owners[owner_start],
                    step, after=(stages[-1].name,), metadata={"nonempty_splits": partials},
                ))
            return KernelProgram(tuple(stages))
        def launch(index):
            start = index * groups_per_wave
            return tuple(Stage(f"query{local}", program=group(start + local, local * lanes))
                         for local in range(min(groups_per_wave, prefix[-1] - start)))

        pipeline = PipelineProgram(ceil_div(prefix[-1], groups_per_wave), 1, launch)
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
              row_cuts: RowCuts | None = None, **metadata) -> OperatorPlan:
    dtype = op.dtype or "bf16"
    if dtype not in {"bf16", "fp16"} or (c.mixed_quant and dtype != "bf16"):
        raise capability_gap("paged_mla_unsupported_dtype", c.abi, ctx.compute_arch.arch_id)
    if pairs is None:
        pairs = owner_pairs(ctx, c.abi)
    if layout.gm_workspace:
        pairs = tuple((aic, aivs[:1]) for aic, aivs in pairs)
    builder = _PagedMlaProgramBuilder(c, layout, dtype, pairs, row_cuts or {})
    phase_totals = builder.totals()
    total = sum(phase_totals.values(), Work())
    phases = []
    for step in ("sink_load", *layout.phases, "merge_read", "merge", "normalize", "store"):
        if (work := phase_totals.get(step)) is None:
            continue
        items = tuple(WorkItem("cube", "flop", amount, precision)
                      for precision, amount in work.matrix.items())
        items += tuple(WorkItem(role, "op" if role == "vector" else "bytes", amount)
                       for role, amount in work.resources.items())
        predecessors = {
            "release0": ("partial0",),
            "release1": ("partial1", "release0"),
        }.get(step, (phases[-1].phase_id,) if phases else ())
        phases.append(PipelinePhase(step, predecessors, items))
    return OperatorPlan(
        total, builder, metadata={
            "implementation": c.abi, "arch_id": ctx.compute_arch.arch_id,
            "governed_family": FAMILY, "s2_base_size": c.s2_base_size, "key_tile": layout.key_tile,
            "dqk": c.dqk, "dv": c.dv, "useful_matrix_flops": c.useful_matrix_flops,
            "latent_record_bytes": 0 if c.mixed_quant else latent_record_bytes(c, dtype=dtype),
            "issued_matrix_flops": c.issued_matrix_flops,
            "scale_encoding": c.scale_encoding if c.mixed_quant else "", **metadata,
            **({"sinks": c.sinks} if c.sinks else {}),
        },
        pipeline_spec=FusedPipelineSpec(FAMILY, c.abi, tuple(phases), sources),
    )

def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(ctx.occurrence, FAMILY, spec.implementation, (op.op_id,),
                                   (), spec.phases, spec.sources, spec.repeats)
