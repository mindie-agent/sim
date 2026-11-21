"""Arch35 fused inference attention and sparse attention strategies.

Upstream-ABI: aclnnGenericBlockSparseAttention

[Role] Model FIA, GBSA, packed SFA, NoPE SFA and the MindIE-SD ABI variants.
[Invariants] Use declared cross-core routes. Preserve distinct offset,
  gather, QK, softmax and PV work and the physical KV representation.
  Refresh resident sectors at admission and publish misses at query completion.
[Timeouts & Locks] Stream selections in query order. No locks or I/O.
[Side-Effect Policy] Bind immutable plans. Classify packed KV reads at Engine admission.
"""
from __future__ import annotations

from bisect import bisect_left
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from functools import partial
from itertools import islice
from math import ceil

from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from ...runtime.memory_physics import (
    AddressRun, AddressRunPlan, BoundAccess, HbmBusSpec, MemoryAccessPattern, ReuseKey, SectorCoverage,
    _intersect_byte_intervals, _measure_byte_intervals, _merge_byte_intervals, _subtract_byte_intervals,
)
from ..contracts import OperatorPlanningContext
from ._fia import dense_fia
from .common import (
    ARCH35, BF16_BYTES, C8_ABI, C8_LATENT_DIM, C8_MODES, C8_ROPE_DIM, C8_ROW_BYTES,
    FIA_ABI, FIA_TILE, FP32_BYTES, GBSA_ABI, INDEX_BYTES, MINDIE_ABIS, MINDIE_V1_ABI, MINDIE_V3_ABI,
    NOPE_ABI, PAGED_LAYOUTS, QUANT_MODE_BF16, QUANT_MODE_FP8_C8, QUANT_MODE_INT8_C8,
    QUANT_MODE_MINDIE_FP8, QUANT_MODE_MXFP4_CX, QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP8,
    QUANT_MODE_TQ4, SCALE_ADDRESS_BYTES, SECTOR_BYTES, SOFTMAX_OPS, SUPPORTED_ABIS, TQ4_ROW_BYTES,
    AttentionContract, AttentionPhase, _placement, _stage, bsh_kv_access_plan, make_plan, output_phases,
    owner_pairs, packed_gather_run_plans, packed_selection_runs, read_work, unmodeled,
)

UPSTREAM_ABI = "aclnnGenericBlockSparseAttention"
OPS_TRANSFORMER_REVISION = "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946"
MINDIE_SD_REVISION = "9fe772dfb90696e21b5e5d4e92ca07dffab84cf4"
VLLM_REVISION = "0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb"
VLLM_MLA_REVISION = "7a7ed073c481d5e3a2fc2194c5b879efb2d373c0"
VLLM_STACK_REVISION = "405617bae227fe5b097ff163efaa60bb7f22c698"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_gqa.h",
        "5ce4ba86dc262777921326ce15f21bbeeb89db8cfe65e864f8951944b4b71723",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/generic_block_sparse_attention/op_kernel/arch35/generic_block_sparse_attention_kernel_arch35.h",
        "8bfaba690655b925bef70ebb418594adbb9bfb4f2c8f56fa860961010d15d69a",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/kv_quant_sparse_flash_attention/op_host/kv_quant_sparse_flash_attention_tiling.cpp",
        "2c75acce70974165d5b087416d724bcd0d09909b2e8dd693370b5860de9dbc6e",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/kv_quant_sparse_flash_attention/op_kernel/arch35/kv_quant_sparse_flash_attention_service_vector_mla_arch35.h",
        "f315c56c898a05fb57a47559df399fabb05f389b3374c68483cbe0e214269c4e",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/sparse_flash_mla/op_kernel/arch35/sparse_flash_mla_swa_kernel_arch35.h",
        "32e2d34a18b674a4d5f4afc48ce206c5e878b796dc8acddec6eb1e03efa1c2a7",
    ),
    SourceRef(
        "MindIE/MindIE-SD", MINDIE_SD_REVISION,
        "csrc/plugin/block_sparse_attention.cpp",
        "6d7106c856e2f3bbb7cc8d272d8cd97ebd95f997ca36ef4fb33bfe3773738664",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/fused_infer_attention_score/op_kernel/arch35/fia_kernel_noquant_gqa.h",
        "f6fbafb900cfca41fd97c0d88edd3d625f9506d95ca2a32907800280709dfba7",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_mxfp4_full_quant.h",
        "e9f20acf0dfeacd5dcd22bfa8d0591957e0ae1852200195d0dc1c9ac61b548fd",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/block_sparse_attention/op_host/block_sparse_attention_tiling.cpp",
        "058c98d73aa2107955eace4f759b6f0003c9d1153d8641bfa70bb5dd1d399bbd",
    ),
    SourceRef(
        "MindIE/MindIE-SD", MINDIE_SD_REVISION,
        "mindiesd/layers/flash_attn/sparse_flash_attn_rf_v3.py",
        "6b4bcedc58b902bcf6c296f22aaef7d09aba39037d51ebf4640d5f988d6cbea3",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/block_sparse_attention/op_kernel/arch35/block_sparse_attention_kernel_arch35_full_quant.h",
        "0439249128d71f67a2146d5ad775c7e340a2cb00e0aa301370065bf4281d1f84",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/attention/attention_v1.py",
        "1194f995eccb90309140d2e19e6dcc2291548cd58d8523c24a5ee1c79836360a",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/attention/context_parallel/common_cp.py",
        "e20eaa698b825c24a9ab37f83870391d944aef7d57fa7e6d680db0c306cc5f65",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/attention/context_parallel/attention_cp.py",
        "d784faa60393e7d543f7c52b2d4f63a8af69810292af231c6ef769ce8de4d02e",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/attention/context_parallel/sfa_cp.py",
        "935ad3ab1a030351aebc7272f7192e30585181d8d757bfff2923eb9d170d13e2",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/attention/mla_v1.py",
        "f7f672e001651a748d2e88ef0f36797ef78db63321ac657f1c7a944c0f561c22",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_MLA_REVISION, "vllm_ascend/attention/mla_v1.py",
        "aba3ec5f7a37bbf10a46ece34140189df4716b61b51eaa19f00d181d506e510c",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_MLA_REVISION, "vllm_ascend/attention/context_parallel/common_cp.py",
        "670f1ffa9c08cd316b5564c9ca03d7a6f4bb1aa6d7e5b81bde5287630903350c",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/attention/context_parallel/attention_cp.py",
        "d784faa60393e7d543f7c52b2d4f63a8af69810292af231c6ef769ce8de4d02e",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/attention/context_parallel/common_cp.py",
        "e20eaa698b825c24a9ab37f83870391d944aef7d57fa7e6d680db0c306cc5f65",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/ops/triton/dcp/dcp_a2a.py",
        "40f1339048c1e61a6de11c5633d70755ccb47ffc15889524d4c6234a89cdd69e",
    ),
    SourceRef(
        "vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/ops/triton/dcp/dcp_a2a.py",
        "40f1339048c1e61a6de11c5633d70755ccb47ffc15889524d4c6234a89cdd69e",
    ),
)
(FIA_SOURCE, GBSA_SOURCE, C8_TILING_SOURCE, C8_VECTOR_SOURCE, NOPE_SOURCE, MINDIE_SOURCE,
 FIA_KERNEL_SOURCE, MXFP4_KERNEL_SOURCE, MXFP4_TILING_SOURCE, MXFP4_CALLER_SOURCE,
 MINDIE_FP8_KERNEL_SOURCE, *CP_SOURCES) = SOURCES
SOURCES_BY_ABI = {
    FIA_ABI: (FIA_SOURCE, FIA_KERNEL_SOURCE), GBSA_ABI: (GBSA_SOURCE,), C8_ABI: (C8_TILING_SOURCE, C8_VECTOR_SOURCE),
    NOPE_ABI: (NOPE_SOURCE,), **{abi: (MINDIE_SOURCE,) for abi in MINDIE_ABIS},
}
MXFP4_SCALE_GROUP = 32
MXFP4_BLOCK_ALIGNMENT = 2 * MXFP4_SCALE_GROUP
MXFP4_HEAD_DIMS = frozenset({64, 128})
FP8_SCALE_GROUP = 128
FP8_MIXED_KV_TILE = 256
FP8_MAX_KV_TILE = 512
MXFP8_SCALE_GROUP = 32
SSMAX_EXTRA_OPS = 2
DEQUANT_OPS = 2
GBSA_OFFSET_WORDS_PER_BLOCK = 2
GBSA_HEAD_GROUP_ROWS = 16


@dataclass(frozen=True, slots=True)
class _C8Read:
    pattern: MemoryAccessPattern
    bound: BoundAccess
    admission: _C8Admission

    def start(self, engine, name, ready, *, owner=""):
        return engine.submit_bound_access(
            name, self.bound, after=(ready,), owner=owner, memory_state=self.admission, pattern=self.pattern,
        )


@dataclass(frozen=True, slots=True)
class _C8Query:
    program: Callable[[_C8Admission], KernelProgram]
    hbm: str
    key: ReuseKey
    capacity: int

    def start(self, engine, name, ready, *, owner=""):
        view = _c8_memory(engine, self.hbm, self.key.die, self.capacity)
        admission = _C8Admission(view)
        program = self.program(admission)
        done = program.start(engine, name, ready, owner=owner)
        _discard_query_programs(engine._compiled_programs, program)
        engine.on_complete(done, lambda _: admission.fill(self.key))
        return done


def _discard_query_programs(cache, program):
    """Release one-shot stage graphs from the Engine's compilation cache."""
    if isinstance(program, KernelProgram):
        cache.pop(id(program), None)
        for stage in program.stages:
            if stage.program is not None:
                _discard_query_programs(cache, stage.program)
    elif isinstance(program, ParallelProgram):
        cache.pop(id(program), None)
        for branch in program.programs:
            _discard_query_programs(cache, branch)


@dataclass(slots=True)
class _C8Admission:
    view: _C8MemoryView
    resident: tuple[tuple[int, int], ...] = ()
    misses: list[tuple[int, int]] = field(default_factory=list)

    def resident_coverage_for(self, pattern, aligned_bytes):
        self.resident = self.view.resident_intervals_for(pattern, aligned_bytes)
        return SectorCoverage(self.resident)

    def admit_access(self, pattern, aligned_bytes):
        resident, self.resident = self.resident, ()
        requested = pattern.runs.sector_intervals()
        hit, miss = AdmissionMemoryView._coverage_hit_miss(SectorCoverage(requested), SectorCoverage(resident),
                                                          aligned_bytes)
        self.misses.extend(_subtract_byte_intervals(requested, resident))
        self.view.touch(pattern.reuse_key, resident)
        return hit, miss

    def fill(self, key):
        self.view.touch(key, _merge_byte_intervals(self.misses), fill=True)
        self.misses.clear()


class _C8MemoryView:
    """Share versioned packed-KV sectors in oldest-to-newest touch order."""

    def __init__(self, capacity):
        self.capacity = capacity // SECTOR_BYTES * SECTOR_BYTES
        self.occupied = 0
        self.ranges: deque[tuple[ReuseKey, int, int]] = deque()

    def resident_intervals_for(self, pattern, aligned_bytes):
        key = pattern.reuse_key
        if key is None or aligned_bytes <= 0:
            return ()
        requested = pattern.runs.sector_intervals()
        if not requested:
            return ()
        lower, upper = requested[0][0], requested[-1][1]
        candidates = _merge_byte_intervals([
            (start, end) for version, start, end in self.ranges
            if version == key and start < upper and lower < end
        ])
        return _intersect_byte_intervals(requested, candidates)

    def touch(self, key, intervals, *, fill=False):
        if not intervals:
            return
        retained = deque()
        lower, upper = intervals[0][0], intervals[-1][1]
        for region in self.ranges:
            version, start, end = region
            if version == key and start < upper and lower < end:
                first = max(0, bisect_left(intervals, (start,)) - 1)
                stop = bisect_left(intervals, (end,), first)
                remainder = _subtract_byte_intervals(((start, end),), intervals[first:stop])
                retained.extend((version, lo, hi) for lo, hi in remainder)
                self.occupied -= end - start - _measure_byte_intervals(remainder)
            elif fill and version != key and version.backing_allocation == key.backing_allocation:
                self.occupied -= end - start
            else:
                retained.append(region)
        for start, end in intervals:
            self.occupied += end - start
            # Compress adjacent sectors when their address order is also their LRU order.
            if retained and retained[-1][0] == key and retained[-1][2] == start:
                _, start, _ = retained.pop()
            retained.append((key, start, end))
        self.ranges = retained
        while self.occupied > self.capacity:
            version, start, end = self.ranges.popleft()
            evicted = min(self.occupied - self.capacity, end - start)
            self.occupied -= evicted
            if start + evicted < end:
                self.ranges.appendleft((version, start + evicted, end))


def _c8_memory(engine, resource, die, capacity):
    views = getattr(engine, "_attention_l2_views", None)
    if views is None:
        views = engine._attention_l2_views = {}
    identity = (resource, die)
    view = views.get(identity)
    if view is None:
        view = views[identity] = _C8MemoryView(capacity)
    return view


@dataclass(frozen=True, slots=True)
class _C8Gather:
    runs: AddressRunPlan
    row_bytes: int
    key: ReuseKey

    def lower(self, context, aiv, aiv_count, admission):
        hbm = context.resource_names["hbm"]
        pipe_role = "mte2_aiv" if "mte2_aiv" in context.resource_names else "mte2"
        pipe_units = context.resource_units.get(pipe_role, 1)
        # The shared HBM resource measures bytes. Directional bus rows
        # otherwise represent occupancy at a unit service rate.
        key = self.key
        bus = HbmBusSpec(key.die, 1, 1, 0, row_r=hbm, ctr_r=hbm)
        owner = unit_group(aiv, aiv_count, pipe_units)[0]
        stages = []
        for index, runs in enumerate(packed_gather_run_plans(self.runs)):
            pattern = MemoryAccessPattern(
                key.backing_allocation, key.tensor_version, key.die, owner, "read", "mte2_aiv",
                runs.runs[0].rows * self.row_bytes, runs, reuse_key=key,
            )
            bound = context.bind_access(pattern, hbm_bus=bus)
            stages.append(Stage("read" if index == 0 else f"read-{index}",
                                program=_C8Read(pattern, bound, admission)))
        return KernelProgram(tuple(stages))


def _l2_capacity(ctx: OperatorPlanningContext) -> int:
    geometry = ctx.geometry
    if geometry.buffer_units.get("l2", 1) != 1:
        raise unmodeled("attention_l2_owner", "Bind one shared L2 allocation owner per die")
    if not (geometry.rates.get("l2") or geometry.resource_units.get("l2")):
        return 0
    return geometry.buffers.get("l2", dict(ctx.datapath.buffers).get("l2", 0))


@dataclass(frozen=True, slots=True)
class _PackedAttention:
    contract: AttentionContract
    ctx: OperatorPlanningContext
    dtype: str
    row_bytes: int
    counts: tuple[int, ...]
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    key: ReuseKey | None
    capacity: int
    phases: tuple[AttentionPhase, ...] = ()

    def intervals(self, rows):
        splits = min(self.contract.splits, max(1, rows))
        base, extra = divmod(rows, splits)
        for index in range(splits):
            start = index * base + min(index, extra)
            yield start, start + base + (index < extra)

    def split_phases(self, query, start, end, splits):
        c, ctx = self.contract, self.ctx
        rows = end - start
        stride = ceil(self.row_bytes / SECTOR_BYTES) * SECTOR_BYTES
        selection = (c.selection_lists[query][start:end] if c.selection_lists else
                     AddressRunPlan((AddressRun(start * stride, rows, self.row_bytes, stride),)))
        runs = packed_selection_runs(selection, self.row_bytes)
        gather = _C8Gather(runs, self.row_bytes, self.key) if self.key and rows else None
        cold_bytes = sum(plan.dram_sector_and_service()[0] for plan in packed_gather_run_plans(runs))
        phases = [AttentionPhase("sparse_index", read_work(ctx, rows * INDEX_BYTES, aiv=True), ())]
        gather_deps = ("sparse_index",)
        if c.kv_layout in PAGED_LAYOUTS:
            pages = ceil((start % c.block_size + rows) / c.block_size) if rows else 0
            phases.append(AttentionPhase("block_table", read_work(ctx, pages * INDEX_BYTES, aiv=True), ()))
            gather_deps += ("block_table",)
        tiles = ceil(rows / FIA_TILE)
        if c.precompute_scale_addr:
            phases.append(AttentionPhase("scale_address", read_work(ctx, tiles * SCALE_ADDRESS_BYTES, aiv=True),
                                         ("sparse_index",)))
            gather_deps += ("scale_address",)
        phases.extend((
            AttentionPhase("packed_kv_gather", read_work(ctx, runs.aligned_copy_bytes(), aiv=True,
                                                        hbm=cold_bytes),
                           gather_deps, memory=gather),
            AttentionPhase("kv_dequant", Work(resources={"vector": rows * c.v_head_dim * DEQUANT_OPS}),
                           ("packed_kv_gather",)),
            AttentionPhase("kv_to_l1", Work(resources={"mte3": rows * c.head_dim * BF16_BYTES}), ("kv_dequant",)),
        ))
        heads, dk, dv = c.num_heads, c.head_dim, c.v_head_dim
        pairs = heads * rows
        query_bytes = heads * dk * BF16_BYTES
        probability = pairs * BF16_BYTES
        phases.extend((
            AttentionPhase("query_load", read_work(ctx, query_bytes), ()),
            AttentionPhase("qk", Work({self.dtype: 2 * pairs * dk},
                                       {"mte1": query_bytes + rows * dk * BF16_BYTES}),
                           ("query_load", "kv_to_l1")),
            AttentionPhase("score_fixpipe", Work(resources={"fixpipe": pairs * FP32_BYTES}), ("qk",)),
            AttentionPhase("online_softmax", Work(resources={
                "vector": pairs * (SOFTMAX_OPS + (SSMAX_EXTRA_OPS if c.ssmax else 0)),
            }), ("score_fixpipe",)),
            AttentionPhase("probability_transfer", Work(resources={"mte3": probability}), ("online_softmax",)),
            AttentionPhase("pv", Work({self.dtype: 2 * pairs * dv}, {"mte1": probability + rows * dv * BF16_BYTES}),
                           ("probability_transfer",)),
            AttentionPhase("pv_fixpipe", Work(resources={"fixpipe": tiles * heads * dv * FP32_BYTES}), ("pv",)),
            AttentionPhase("online_output", Work(resources={"vector": (tiles + 1) * heads * dv}), ("pv_fixpipe",)),
        ))
        if splits > 1:
            payload = heads * (dv + 2) * FP32_BYTES
            phases.append(AttentionPhase("split_write", Work(resources={"mte3": payload, "hbm": payload}),
                                         ("online_output",)))
        return tuple(phases)

    def finish(self, splits):
        c = self.contract
        output = output_phases(self.ctx, c, BF16_BYTES, ("online_output",), splits * c.num_heads, queries=1)
        return output if splits == 1 else output[1:]

    def __call__(self, context):
        datapath = self.ctx.datapath
        places = tuple(_placement(context, pair, datapath.aic_count, datapath.aiv_count) for pair in self.owners)
        owner_count = len(self.owners)

        def query(index, core, admission=None):
            intervals = tuple(self.intervals(self.counts[index]))
            branches = []
            for split, (start, end) in enumerate(intervals):
                owner = (core + split) % owner_count
                aiv = self.owners[owner][1][0]
                stages = []
                for phase in self.split_phases(index, start, end, len(intervals)):
                    if phase.memory is None:
                        stages.append(_stage(context, phase, places[owner]))
                    else:
                        gather = phase.memory.lower(context, aiv, datapath.aiv_count, admission)
                        stages.append(Stage(phase.name, after=phase.after, program=gather))
                branches.append(KernelProgram(tuple(stages)))
            stages = [Stage("splits", program=ParallelProgram(tuple(branches)))]
            for phase in self.finish(len(intervals)):
                stages.append(_stage(context, replace(phase, after=(stages[-1].name,)), places[core]))
            return KernelProgram(tuple(stages))

        def tile(index, core):
            program = (_C8Query(partial(query, index, core), context.resource_names["hbm"], self.key, self.capacity)
                       if self.key else query(index, core))
            return (Stage("query", program=program),)

        return ParallelProgram(tuple(
            PipelineProgram(len(indices), 1, lambda i, core=core, indices=indices: tile(indices[i], core))
            for core in range(owner_count) if (indices := range(core, len(self.counts), owner_count))
        ))


def _packed_kv(op, contract, ctx, sources):
    if (contract.quant_mode not in C8_MODES or contract.num_kv_heads != 1
            or contract.v_head_dim != C8_LATENT_DIM
            or contract.head_dim not in {C8_LATENT_DIM, C8_LATENT_DIM + C8_ROPE_DIM}):
        raise unmodeled("attention_c8_abi_shape", "Bind the packed latent C8 or TQ4 cache geometry")
    size = TQ4_ROW_BYTES if contract.quant_mode == QUANT_MODE_TQ4 else C8_ROW_BYTES
    counts = tuple(contract.row_counts())
    capacity = _l2_capacity(ctx)
    key = None
    if capacity and contract.selection_lists and contract.kv_tensor_id:
        key = ReuseKey(ctx.geometry.rank or 0, contract.kv_tensor_id, contract.kv_version)
    owners = owner_pairs(ctx, sum(min(contract.splits, max(1, count)) for count in counts))
    loop = _PackedAttention(contract, ctx, op.dtype, size, counts, owners, key, capacity)
    producer, finish = {}, {}
    if contract.selection_lists:
        launches = ((query, count, 1) for query, count in enumerate(counts))
    else:
        launches = ((0, count, repeats) for count, repeats in Counter(counts).items())
    for query, count, repeats in launches:
        intervals = tuple(loop.intervals(count))
        for start, end in intervals:
            for phase in loop.split_phases(query, start, end, len(intervals)):
                previous = producer.get(phase.name)
                work = phase.work.scaled(repeats)
                producer[phase.name] = replace(phase, work=work + previous.work if previous else work, memory=None)
        for phase in loop.finish(len(intervals)):
            previous = finish.get(phase.name)
            work = phase.work.scaled(repeats)
            finish[phase.name] = replace(phase, work=work + previous.work if previous else work)
    output = finish.pop("output")
    if finish:
        output = replace(output, after=("split_merge",))
    phases = (*producer.values(), *finish.values(), output)
    cold = producer["packed_kv_gather"].work.resources["hbm"]
    rows = sum(counts)
    metadata = {
        "source_program": sources[0].to_dict(), "kv_logical_bytes": rows * size, "kv_sector_bytes": cold,
        "kv_dram_sector_bytes": cold, "l2_shared_kv_bytes": 0, "kv_row_bytes": size, "kv_gather_rows": rows,
    }
    return make_plan(op, contract, ctx, phases, sources, metadata=metadata, program=replace(loop, phases=phases))


def _sparse_kv_load(
    contract: AttentionContract, ctx: OperatorPlanningContext, dim: int, tensor: str,
    *, native_fp8: bool = False, shared_l2: AdmissionMemoryView | None = None,
) -> Work:
    mode = contract.quant_mode
    elem = BF16_BYTES if mode == QUANT_MODE_BF16 else 1
    if native_fp8:
        scale_bytes = 0  # Native FP8 scalar operands have their own GM loads.
    elif mode == QUANT_MODE_INT8_C8:
        scale_bytes = FP32_BYTES
    elif mode == QUANT_MODE_FP8_C8:
        scale_bytes = ceil(dim / FP8_SCALE_GROUP) * BF16_BYTES
    else:
        scale_bytes = ceil(dim / MXFP8_SCALE_GROUP) if mode == QUANT_MODE_MXFP8 else 0
    hbm = copied = hit = 0
    if shared_l2 is not None and not scale_bytes:
        # The two prefix tensors have independent backing allocations.
        key = ReuseKey(0, tensor, 0, reuse_stream=tensor)
        bus = HbmBusSpec(0, 1, 1, 0)
        for rows in contract.row_counts():
            runs = tuple(
                run for owner in range(contract.num_kv_heads)
                for run in bsh_kv_access_plan(
                    contract.num_kv_heads, dim, elem, contract.kv_layout,
                    rows=rows, first_head=owner, block_size=contract.block_size,
                ).runs
            )
            plan = AddressRunPlan(runs)
            pattern = MemoryAccessPattern(
                tensor, 0, 0, 0, "read", "mte2_aic",
                rows * contract.num_kv_heads * dim * elem, plan, reuse_key=key,
            )
            ledger = bind_memory_access(
                pattern, shared_l2, hbm_bus=bus, copy_hit_bytes_per_ms=1,
                copy_miss_bytes_per_ms=1, l2_bytes_per_ms=1, pipe_row_id="mte2_aic",
                l2_row_id="l2", commit=True,
            ).ledger
            copied += ledger.aligned_transfer_bytes
            hbm += ledger.dram_sector_bytes
            hit += ledger.l2_hit_bytes
        return read_work(ctx, copied, hbm=hbm, l2=hit)

    for rows in contract.row_counts():
        block = contract.block_size
        full, tail = divmod(rows, block)
        for count, repeats in ((block, full), (tail, 1)):
            if not count or not repeats:
                continue
            for owner in range(contract.num_kv_heads):
                plan = bsh_kv_access_plan(contract.num_kv_heads, dim, elem, contract.kv_layout,
                                          rows=count, first_head=owner, block_size=block)
                plan = AddressRunPlan(plan.runs, scale_bytes=count * scale_bytes)
                pattern = MemoryAccessPattern(tensor, 0, 0, owner, "read", "mte2_aic",
                                              count * (dim * elem + scale_bytes), plan)
                hbm += pattern.runs.dram_sector_and_service()[0] * repeats
                copied += pattern.runs.aligned_copy_bytes() * repeats
    return read_work(ctx, copied, hbm=hbm)


def _mxfp4_inputs(contract, q_rows, rows):
    dk, dv = contract.head_dim, contract.v_head_dim
    heads, block = contract.num_kv_heads, contract.block_size
    counts = sum(ceil(n / MXFP4_BLOCK_ALIGNMENT) for n in contract.row_counts()) * heads
    payloads = {
        "query": (q_rows * (dk // 2), q_rows * (dk // MXFP4_SCALE_GROUP)),
        "key": (rows * heads * (dk // 2), rows * heads * (dk // MXFP4_SCALE_GROUP)),
        "value": (rows * heads * (dv // 2), counts * dv * 2),
    }
    return {
        "quant_mode": contract.quant_mode, "dst_type_max": contract.dst_type_max,
        "block_shape": (contract.query_block_size or block, block),
        "packed_dtype": "fp4_e2m1", "scale_dtype": "fp8_e8m0", "output_dtype": "bf16",
        "operand_bytes": {name: data for name, (data, _) in payloads.items()},
        "scale_bytes": {name: scales for name, (_, scales) in payloads.items()},
    }


def _mindie_fp8_scale_bytes(contract):
    # The kernel reads one Q scalar per (batch, head, Q tile), and one K
    # and V scalar per selected S2 tile in that task.
    q_block = contract.query_block_size or contract.block_size
    q_tile = min(q_block, FIA_TILE)
    kv_block = contract.block_size
    preferred = FP8_MIXED_KV_TILE if contract.head_dim <= FIA_TILE else FIA_TILE
    if kv_block == FP8_MAX_KV_TILE and preferred == FP8_MIXED_KV_TILE:
        kv_tile = kv_block
    elif kv_block % preferred == 0:
        kv_tile = preferred
    elif preferred > FIA_TILE and kv_block % FIA_TILE == 0:
        # CANN's mixed FP8 path prefers 128 here, even if a larger divisor fits.
        kv_tile = FIA_TILE
    else:
        kv_tile = next(width for width in range(min(kv_block, preferred), 0, -1) if kv_block % width == 0)
    batch = min(contract.batch_size, contract.query_tokens)
    base, extra = divmod(contract.query_tokens, batch)
    queries = (q for q, _ in contract.sequences) if contract.sequences else (base + (i < extra) for i in range(batch))
    counts = iter(contract.row_counts())
    q_tasks = kv_tasks = 0
    for q in queries:
        for block in range(0, q, q_block):
            for tile in range(block, min(block + q_block, q), q_tile):
                width = min(q_tile, block + q_block - tile, q - tile)
                rows = max(islice(counts, width))
                q_tasks += 1
                kv_tasks += ceil(rows / kv_tile)
    scale_width = contract.num_heads * FP32_BYTES
    return {"query": q_tasks * scale_width, "key": kv_tasks * scale_width, "value": kv_tasks * scale_width}


def _sparse(op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext,
            sources: tuple[SourceRef, ...]) -> OperatorPlan:
    abi, mode = contract.abi, contract.quant_mode
    nope = abi == NOPE_ABI
    mindie = abi in MINDIE_ABIS
    native_fp8 = mindie and mode == QUANT_MODE_MINDIE_FP8
    mxfp4 = mindie and mode in {QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX}
    if mxfp4:
        if abi != MINDIE_V3_ABI:
            raise unmodeled("attention_mindie_mxfp4_version", "MXFP4 requires aclnnBlockSparseAttentionV3")
        if (contract.return_lse or contract.block_size % MXFP4_BLOCK_ALIGNMENT
                or (contract.query_block_size or contract.block_size) % MXFP4_BLOCK_ALIGNMENT):
            raise unmodeled("attention_mindie_mxfp4_block", "MXFP4 requires 64-multiple blocks and no LSE")
        if (contract.head_dim != contract.v_head_dim or contract.head_dim not in MXFP4_HEAD_DIMS
                or (contract.kv_layout == "TND" and not contract.sequences)):
            raise unmodeled("attention_mindie_mxfp4_shape", "Bind matching MXFP4 head and sequence shapes")
        sources = (*sources, MXFP4_KERNEL_SOURCE, MXFP4_TILING_SOURCE, MXFP4_CALLER_SOURCE)
    elif mode == QUANT_MODE_MXFP4_OCP or contract.dst_type_max:
        raise unmodeled("attention_mindie_mxfp4_abi", "Bind an MXFP4 storage format to MindIE-SD V3")
    if op.dtype not in {"bf16", "fp16", "fp8_e4m3"}:
        raise unmodeled("attention_sparse_dtype", f"Bind a supported sparse attention dtype for {abi}")
    if contract.selection_lists:
        raise unmodeled("attention_sparse_addresses", "Bind explicit row addresses to packed SFA")
    if (nope and (mode != QUANT_MODE_BF16 or contract.head_dim != contract.v_head_dim
                  or contract.num_kv_heads != 1)):
        raise unmodeled("attention_nope_shape", "Bind a shared, nonquantized NoPE latent cache")
    if mindie and (contract.kv_layout not in {"TND", "BNSD"}
                   or (not mxfp4 and not native_fp8 and mode != QUANT_MODE_BF16)
                   or (abi == MINDIE_V1_ABI and native_fp8)):
        raise unmodeled("attention_mindie_sd_abi", f"Bind a supported layout and storage format for {abi}")
    if mode == QUANT_MODE_TQ4 and not mxfp4:
        raise unmodeled("attention_tq4_abi", f"TQ4 packed latent cache requires {C8_ABI}")
    dtype = "mxfp4" if mxfp4 else ("fp8_e4m3" if native_fp8 or mode == QUANT_MODE_MXFP8 else op.dtype)
    q_elem = 1 if native_fp8 else BF16_BYTES
    operand_elem = 1 if native_fp8 or mode == QUANT_MODE_MXFP8 else BF16_BYTES
    rows = blocks = tiles = split_rows = 0
    for count in contract.row_counts():
        rows += count
        blocks += ceil(count / contract.block_size)
        tiles += ceil(count / FIA_TILE)
        split_rows += min(contract.splits, max(1, count))
    pairs = rows * contract.num_heads
    kv_rows = rows * contract.num_kv_heads
    q_rows = split_rows * contract.num_heads
    q_bytes = q_rows * contract.head_dim * q_elem
    metadata = {"source_program": sources[0].to_dict()}
    packed_bytes = {}
    if mxfp4:
        binding = _mxfp4_inputs(contract, q_rows, rows)
        metadata.update(binding)
        packed_bytes = binding["operand_bytes"]
        q_bytes = packed_bytes["query"]
    elif native_fp8:
        sources = (*sources, MINDIE_FP8_KERNEL_SOURCE)
        metadata.update(quant_mode=mode, scale_dtype="fp32", scale_bytes=_mindie_fp8_scale_bytes(contract))
    scales = metadata.get("scale_bytes", {})
    phases = [AttentionPhase("query_load", read_work(ctx, q_bytes), ())]
    qk_deps = ("query_load",)
    if scales:
        phases.append(AttentionPhase("query_scale_load", read_work(ctx, scales["query"]), ()))
        qk_deps += ("query_scale_load",)
    v_load = None
    index_count = (rows if nope else blocks) * contract.num_kv_heads
    index_bytes = 0 if contract.metadata_reuse else index_count * INDEX_BYTES
    phases.append(AttentionPhase("sparse_index", read_work(ctx, index_bytes, aiv=True), ()))
    offset_deps = ("sparse_index",)
    if contract.kv_layout in PAGED_LAYOUTS:
        phases.append(AttentionPhase("block_table", read_work(ctx, blocks * INDEX_BYTES, aiv=True), ()))
        offset_deps += ("block_table",)
    offsets = index_count
    phases.append(AttentionPhase("offset_prepare", Work(resources={"vector": offsets}), offset_deps))
    k_deps = ("offset_prepare",)
    if abi == GBSA_ABI:
        size = offsets * GBSA_OFFSET_WORDS_PER_BLOCK * SCALE_ADDRESS_BYTES
        phases.append(AttentionPhase("offset_workspace_write", Work(resources={"mte3": size, "hbm": size}), k_deps))
        phases.append(AttentionPhase("offset_workspace_read", read_work(ctx, size), ("offset_workspace_write",)))
        k_deps = ("offset_workspace_read",)
    shared_l2 = None
    if contract.shared_prefix and mode == QUANT_MODE_BF16:
        capacity = _l2_capacity(ctx)
        if capacity:
            shared_l2 = AdmissionMemoryView({0: capacity})
            shared_l2.enter_stream(0, "key")
            if not nope:
                shared_l2.enter_stream(0, "value")
    k_load = read_work(ctx, packed_bytes["key"]) if mxfp4 else _sparse_kv_load(
        contract, ctx, contract.head_dim, "key", native_fp8=native_fp8, shared_l2=shared_l2,
    )
    phases.append(AttentionPhase("key_load", k_load, k_deps))
    qk_deps += ("key_load",)
    if scales:
        phases.append(AttentionPhase("key_scale_load", read_work(ctx, scales["key"]), k_deps))
        qk_deps += ("key_scale_load",)
    if mxfp4:
        v_load = read_work(ctx, packed_bytes["value"])
    elif not nope:
        v_load = _sparse_kv_load(contract, ctx, contract.v_head_dim, "value",
                                 native_fp8=native_fp8, shared_l2=shared_l2)
    metadata["l2_shared_kv_bytes"] = k_load.resources["l2"] + (v_load.resources["l2"] if v_load else 0)

    if mode in {QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8} and not native_fp8:
        phases.append(AttentionPhase("key_dequant", Work(resources={
            "vector": kv_rows * contract.head_dim * DEQUANT_OPS,
        }), qk_deps))
        qk_deps = ("key_dequant",)
    key_operand_bytes = kv_rows * contract.head_dim * operand_elem
    probability_bytes = pairs * operand_elem
    value_operand_bytes = kv_rows * contract.v_head_dim * operand_elem
    if mxfp4:
        key_operand_bytes = packed_bytes["key"]
        probability_bytes = ceil(pairs / 2)
        value_operand_bytes = packed_bytes["value"]
    l0_scales = scales if mxfp4 else {}
    phases.extend((
        AttentionPhase("qk", Work({dtype: 2 * pairs * contract.head_dim},
                                   {"mte1": q_bytes + key_operand_bytes + l0_scales.get("query", 0)
                                    + l0_scales.get("key", 0)}), qk_deps),
        AttentionPhase("score_fixpipe", Work(resources={"fixpipe": pairs * FP32_BYTES}), ("qk",)),
        AttentionPhase("online_softmax", Work(resources={
            "vector": pairs * (SOFTMAX_OPS + (SSMAX_EXTRA_OPS if contract.ssmax else 0)),
        }), ("score_fixpipe",)),
        AttentionPhase("probability_transfer", Work(resources={"mte3": probability_bytes}), ("online_softmax",)),
    ))
    pv_deps = ("probability_transfer",)
    if v_load is not None:
        phases.append(AttentionPhase("value_load", v_load, pv_deps))
        pv_deps = ("value_load",)
        if scales:
            phases.append(AttentionPhase("value_scale_load", read_work(ctx, scales["value"]), ("probability_transfer",)))
            pv_deps += ("value_scale_load",)
        if mode in {QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8} and not native_fp8:
            phases.append(AttentionPhase("value_dequant", Work(resources={
                "vector": kv_rows * contract.v_head_dim * DEQUANT_OPS,
            }), pv_deps))
            pv_deps = ("value_dequant",)
    phases.extend((
        AttentionPhase("pv", Work({dtype: 2 * pairs * contract.v_head_dim},
                                   {"mte1": probability_bytes + value_operand_bytes + l0_scales.get("value", 0)}), pv_deps),
        AttentionPhase("pv_fixpipe", Work(resources={
            "fixpipe": tiles * contract.num_heads * contract.v_head_dim * FP32_BYTES,
        }), ("pv",)),
        AttentionPhase("online_output", Work(resources={
            "vector": (tiles * contract.num_heads + q_rows) * contract.v_head_dim,
        }), ("pv_fixpipe",)),
    ))
    phases.extend(output_phases(ctx, contract, BF16_BYTES, ("online_output",), q_rows))
    group_rows = (ceil(contract.num_heads / (contract.num_kv_heads * GBSA_HEAD_GROUP_ROWS))
                  if abi == GBSA_ABI else 1)
    work_items = split_rows * contract.num_kv_heads * group_rows
    return make_plan(op, contract, ctx, tuple(phases), sources, metadata=metadata, work_items=work_items)


@dataclass(frozen=True, slots=True)
class _Strategy:
    arch_id: ArchId = ArchId(ARCH35)
    abi: str = UPSTREAM_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        abi = ctx.requested_abi or dict(op.metadata).get("native_kind", self.abi)
        return ctx.compute_arch.arch_id == self.arch_id and abi in SUPPORTED_ABIS

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        contract: AttentionContract = dict(op.metadata)["attention_contract"]
        if not self.accepts(op, ctx):
            raise unmodeled("attention_arch35_abi", "Bind a supported Arch35 attention ABI")
        ctx.datapath.route("l0c", "ub")
        ctx.datapath.route("ub", "l1")
        sources = SOURCES_BY_ABI[contract.abi]
        if contract.abi == FIA_ABI:
            return dense_fia(op, contract, ctx, sources, gm_workspace=False)
        if contract.abi == C8_ABI:
            return _packed_kv(op, contract, ctx, sources)
        return _sparse(op, contract, ctx, sources)


STRATEGY = _Strategy()
