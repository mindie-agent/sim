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

from collections import Counter
from collections.abc import Callable
import copy
from dataclasses import dataclass, replace
from functools import partial
from itertools import islice
from math import ceil, sqrt
from typing import Literal

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec
from ...mechanism import OperatorPlan, Work
from ...ownership import unit_group
from ...program import KernelProgram, ParallelProgram, PipelineProgram, Stage
from ...runtime.l2_residency import AdmissionMemoryView, bind_memory_access
from ...runtime.memory_physics import (
    AddressRun, AddressRunPlan, BoundAccess, HbmBusSpec, MemoryAccessPattern, ReuseKey,
)
from ..contracts import OperatorPlanningContext
from ..sharedkv.sfa_native import SFA_SPLIT_G_HEADS, build_arch35_split_g_sfa
from ._fia import dense_fia
from .common import (
    ARCH35, BF16_BYTES, C8_ABI, C8_LATENT_DIM, C8_MODES, C8_ROPE_DIM, C8_ROW_BYTES, C8_V1_ABI,
    C8_VLLM_ABI, C8_VLLM_MODES, C8_VLLM_ROW_BYTES, FIA_ABI, FIA_TILE, FP32_BYTES, GBSA_ABI, GBSA_LAYOUTS,
    INDEX_BYTES, MINDIE_ABIS, MINDIE_V1_ABI, MINDIE_V3_ABI, NOPE_ABI, PAGED_LAYOUTS, QUANT_MODE_BF16,
    QUANT_MODE_FP8_C8, QUANT_MODE_INT8_C8, QUANT_MODE_MINDIE_FP8, QUANT_MODE_MXFP4_CX, QUANT_MODE_MXFP4_OCP,
    QUANT_MODE_MXFP8, QUANT_MODE_TQ4, SCALE_ADDRESS_BYTES, SECTOR_BYTES, SFA_V1_ABI, SOFTMAX_OPS,
    SOFTMAX_STATS_SLOTS, SUPPORTED_ABIS, TQ4_ROW_BYTES, AttentionContract, AttentionMemoryRead, AttentionPhase,
    _placement, _stage, attention_owner_waves, bsh_kv_access_plan, contract_for_requests, make_plan,
    output_phases, owner_pairs, packed_gather_run_plans, packed_selection_runs, planning_sequence_spans,
    read_work, unmodeled,
)

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
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
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/generic_block_sparse_attention/op_host/generic_block_sparse_attention_tiling.cpp",
        "364f718e71332eb506cdb5e62a857652186049e049d7d9e217296c4e13800790",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/generic_block_sparse_attention/op_host/op_api/aclnn_generic_block_sparse_attention.cpp",
        "b5baedb2018d9fef02332e3e945b95e543d26ca72fe7a8b71f43ad0edc406542",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/generic_block_sparse_attention/op_kernel/arch35/generic_block_sparse_attention_kernel_arch35_mxfp8_full_quant.h",
        "c0c2e6ca31009f228b9f29db310bcc7588bf5584f11722cc1cf072e456435177",
    ),
    SourceRef(
        "cann/ops-transformer", OPS_TRANSFORMER_REVISION,
        "attention/sparse_flash_mla/op_host/sparse_flash_mla_tiling.cpp",
        "080e57921cff3b6ef825581b77b7efe2cb1da31270fd90b6f0d8450eda4b1195",
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
    SourceRef("vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/attention/context_parallel/attention_cp.py", "d784faa60393e7d543f7c52b2d4f63a8af69810292af231c6ef769ce8de4d02e"),
    SourceRef("vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/attention/context_parallel/common_cp.py", "e20eaa698b825c24a9ab37f83870391d944aef7d57fa7e6d680db0c306cc5f65"),
    SourceRef("vllm-project/vllm-ascend", VLLM_REVISION, "vllm_ascend/ops/triton/dcp/dcp_a2a.py", "40f1339048c1e61a6de11c5633d70755ccb47ffc15889524d4c6234a89cdd69e"),
    SourceRef("vllm-project/vllm-ascend", VLLM_STACK_REVISION, "vllm_ascend/ops/triton/dcp/dcp_a2a.py", "40f1339048c1e61a6de11c5633d70755ccb47ffc15889524d4c6234a89cdd69e"),
)
(FIA_SOURCE, GBSA_SOURCE, C8_TILING_SOURCE, C8_VECTOR_SOURCE, NOPE_SOURCE, MINDIE_SOURCE,
 FIA_KERNEL_SOURCE, MXFP4_KERNEL_SOURCE, MXFP4_TILING_SOURCE, MXFP4_CALLER_SOURCE,
 MINDIE_FP8_KERNEL_SOURCE, GBSA_TILING_SOURCE, GBSA_ACLNN_SOURCE, GBSA_MXFP8_KERNEL_SOURCE,
 NOPE_TILING_SOURCE, *CP_SOURCES) = SOURCES
SOURCES_BY_ABI = {
    FIA_ABI: (FIA_SOURCE, FIA_KERNEL_SOURCE),
    GBSA_ABI: (GBSA_SOURCE, GBSA_TILING_SOURCE, GBSA_ACLNN_SOURCE),
    C8_ABI: (C8_TILING_SOURCE, C8_VECTOR_SOURCE),
    C8_V1_ABI: (C8_TILING_SOURCE, C8_VECTOR_SOURCE),
    C8_VLLM_ABI: (C8_TILING_SOURCE, C8_VECTOR_SOURCE),
    SFA_V1_ABI: (NOPE_SOURCE,),
    NOPE_ABI: (NOPE_SOURCE, NOPE_TILING_SOURCE),
    **{abi: (MINDIE_SOURCE,) for abi in MINDIE_ABIS},
}
MXFP4_SCALE_GROUP = 32
MXFP4_BLOCK_ALIGNMENT = 2 * MXFP4_SCALE_GROUP
MXFP4_HEAD_DIMS = frozenset({64, 128})
FP8_SCALE_GROUP = 128
FP8_MIXED_KV_TILE = 256
FP8_MAX_KV_TILE = 512
MXFP8_SCALE_GROUP = 32
MXFP8_V_CHANNEL_SCALE_BYTES = 2
GBSA_MAX_HEAD_DIM = 256
GBSA_MXFP8_HEAD_DIMS = frozenset({192, 256})
GBSA_MXFP8_SMALL_HEAD_DIM = 192
GBSA_MXFP8_FD_BLOCK_SIZES = frozenset({4, 8, 16, 32, 64, 128})
GBSA_MAX_GROUP_SIZE = 128
GBSA_Q_TILE_SMALL = 16
GBSA_Q_TILE_MEDIUM = 32
GBSA_Q_TILE_LARGE = 64
GBSA_KV_TILE_MEDIUM = 256
GBSA_KV_TILE_WIDE = 384
GBSA_KV_TILE_LARGE = 512
GBSA_FD_MIN_BLOCKS = 12
GBSA_FD_UTIL_NUM = 3
GBSA_FD_UTIL_DEN = 10
GBSA_FD_COST_HEAD_TILE = 125
GBSA_FD_COST_BLOCK = 740
GBSA_FD_COST_HEAD_BLOCK = 35
GBSA_FD_LAUNCH_COST = 278
GBSA_OFFSET_HEADER_WORDS = 8
GBSA_OFFSET_ALIGN_WORDS = 4
GBSA_DENSE_RECORD_WORDS = 2
GBSA_PAGED_RECORD_WORDS = 4
C8_BLOCK_TABLE_ALIGN_BYTES = 512
SSMAX_EXTRA_OPS = 2
DEQUANT_OPS = 2


@dataclass(frozen=True, slots=True)
class _C8Read:
    pattern: MemoryAccessPattern
    bound: BoundAccess
    admission: AdmissionMemoryView
    duplicate: bool = False

    def start(self, engine, name, ready, *, owner=""):
        if self.duplicate:
            return self._start_duplicate(engine, name, ready, owner)
        return engine.submit_bound_access(
            name, self.bound, after=(ready,), owner=owner, memory_state=self.admission, pattern=self.pattern,
        )

    def _start_duplicate(self, engine, name, ready, owner):
        """Issue an independent gather whose matching sibling publishes the fill."""
        from ...runtime.engine_submit import _rebind_access

        done = engine.future(name)

        def issue(event):
            view, pattern = self.admission, self.pattern
            identity = (view, pattern.die, pattern.tensor_id, pattern.version)
            dirty = view.pending_writebacks(pattern.die) or getattr(engine, "_inflight_writebacks", {}).get(identity)
            if dirty:
                transfer = engine.submit_bound_access(
                    name, self.bound, after=(event,), owner=owner, memory_state=view, pattern=pattern,
                )
            else:
                live = _rebind_access(self.bound, pattern, view, commit=False)
                transfer = engine.submit_bound_access(name, live, after=(event,), owner=owner)
            engine.on_complete(transfer, lambda _: engine.resolve(done))

        engine.submit(name + ":admit", after=(ready,), owner=owner, on_complete=issue, native_hook=True)
        return done


@dataclass(frozen=True, slots=True)
class _C8Query:
    program: Callable[[AdmissionMemoryView], KernelProgram]
    hbm: str
    key: ReuseKey
    capacity: int
    ctx: OperatorPlanningContext

    def start(self, engine, name, ready, *, owner=""):
        view = self.ctx.admission_memory_view or engine.admission_memory_view
        if view is None:
            raise ModelingError("attention_c8_admission_memory_view_required")
        program = self.program(view)
        done = program.start(engine, name, ready, owner=owner)
        _discard_query_programs(engine._compiled_programs, program)
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


@dataclass(frozen=True, slots=True)
class _C8Gather:
    runs: AddressRunPlan
    row_bytes: int
    key: ReuseKey

    def lower(self, context, aiv, aiv_count, admission, issued_runs):
        hbm = context.resource_names["hbm"]
        pipe_role = "mte2_aiv" if "mte2_aiv" in context.resource_names else "mte2"
        pipe_units = context.resource_units.get(pipe_role, 1)
        pipe_rate = context.rates[context.resource_names[pipe_role]] / pipe_units
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
            bound = context.bind_access(pattern, hbm_bus=bus, copy_hit_bytes_per_ms=pipe_rate,
                                        copy_miss_bytes_per_ms=pipe_rate)
            duplicate = runs in issued_runs
            issued_runs.add(runs)
            stages.append(Stage("read" if index == 0 else f"read-{index}",
                                program=_C8Read(pattern, bound, admission, duplicate)))
        return KernelProgram(tuple(stages))


def _l2_capacity(ctx: OperatorPlanningContext) -> int:
    geometry = ctx.geometry
    if geometry.buffer_units.get("l2", 1) != 1:
        raise unmodeled("attention_l2_owner", "Bind one shared L2 allocation owner per die")
    if not (geometry.rates.get("l2") or geometry.resource_units.get("l2")):
        return 0
    if ctx.admission_memory_view is not None:
        return ctx.admission_memory_view.effective_l2_bytes(geometry.rank or 0)
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
    vectorized_block_table_bytes: int = 0
    vec_bt_first_visit: frozenset[int] = frozenset()

    def intervals(self, rows):
        splits = min(self.contract.splits, max(1, rows))
        if self.ctx.requests:
            splits = min(splits, len(self.owners))
        base, extra = divmod(rows, splits)
        for index in range(splits):
            start = index * base + min(index, extra)
            yield start, start + base + (index < extra)

    def split_phases(self, query, start, end, splits, *, split_index: int = 0):
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
            if self.vectorized_block_table_bytes and rows:
                bt_bytes = (
                    self.vectorized_block_table_bytes
                    if (split_index == 0 and query in self.vec_bt_first_visit)
                    else 0
                )
                phases.append(AttentionPhase("block_table", read_work(ctx, bt_bytes, aiv=True), ()))
                phy_bytes = rows * SCALE_ADDRESS_BYTES
                phases.append(AttentionPhase(
                    "phy_addr_write",
                    Work(resources={"mte3": phy_bytes, "hbm": phy_bytes}),
                    ("sparse_index", "block_table"),
                ))
                phases.append(AttentionPhase("phy_addr_read", read_work(ctx, phy_bytes, aiv=True), ("phy_addr_write",)))
                gather_deps = ("phy_addr_read",)
            else:
                if c.selection_lists and rows:
                    page_bytes = c.block_size * self.row_bytes
                    pages = len({addr // page_bytes for addr in c.selection_lists[query][start:end]})
                else:
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
        lse_slots = SOFTMAX_STATS_SLOTS if c.abi in {C8_ABI, C8_VLLM_ABI} else 1
        output = output_phases(
            self.ctx, c, BF16_BYTES, ("online_output",), splits * c.num_heads,
            queries=1, lse_slots=lse_slots,
        )
        return output if splits == 1 else output[1:]

    def __call__(self, context):
        datapath = self.ctx.datapath
        places = tuple(_placement(context, pair, datapath.aic_count, datapath.aiv_count) for pair in self.owners)
        owner_count = len(self.owners)

        def query(index, core, admission=None):
            intervals = tuple(self.intervals(self.counts[index]))
            issued_runs = set()
            branches = []
            for split, (start, end) in enumerate(intervals):
                owner = core + split if self.ctx.requests else (core + split) % owner_count
                aiv = self.owners[owner][1][0]
                stages = []
                for phase in self.split_phases(index, start, end, len(intervals), split_index=split):
                    if phase.memory is None:
                        stages.append(_stage(context, phase, places[owner]))
                    else:
                        gather = phase.memory.lower(context, aiv, datapath.aiv_count, admission, issued_runs)
                        stages.append(Stage(phase.name, after=phase.after, program=gather))
                branches.append(KernelProgram(tuple(stages)))
            stages = [Stage("splits", program=ParallelProgram(tuple(branches)))]
            for phase in self.finish(len(intervals)):
                stages.append(_stage(context, replace(phase, after=(stages[-1].name,)), places[core]))
            return KernelProgram(tuple(stages))

        def tile(index, core, name="query"):
            program = (_C8Query(partial(query, index, core), context.resource_names["hbm"], self.key,
                                self.capacity, self.ctx)
                       if self.key else query(index, core))
            return Stage(name, program=program)

        if self.ctx.requests:
            splits = min(owner_count, self.contract.splits, max(1, max(self.counts)))
            return attention_owner_waves(len(self.counts), owner_count, splits, tile)
        return ParallelProgram(tuple(
            PipelineProgram(len(indices), 1, lambda i, core=core, indices=indices: (tile(indices[i], core),))
            for core in range(owner_count) if (indices := range(core, len(self.counts), owner_count))
        ))


def _validate_c8_abi(contract: AttentionContract) -> None:
    if contract.quant_mode == QUANT_MODE_TQ4:
        raise unmodeled("attention_arch35_tq4_unsupported", "TQ4 packed latent cache is supported on Arch22 only, not Arch35")
    if contract.abi == C8_V1_ABI and contract.return_lse:
        raise unmodeled("attention_c8_v1_no_lse", "aclnnKvQuantSparseFlashAttention V1 does not expose softmax statistics outputs")
    if contract.abi == C8_VLLM_ABI:
        if (contract.quant_mode not in C8_VLLM_MODES or contract.num_kv_heads != 1
                or contract.v_head_dim != C8_LATENT_DIM
                or contract.head_dim not in {C8_LATENT_DIM, C8_LATENT_DIM + C8_ROPE_DIM}):
            raise unmodeled("attention_c8_vllm_abi_shape", "Bind vLLM packed latent C8 geometry (Dk in {512, 576}, Dv=512)")
        return
    if (contract.quant_mode not in C8_MODES or contract.num_kv_heads != 1
            or contract.v_head_dim != C8_LATENT_DIM
            or contract.head_dim != C8_LATENT_DIM + C8_ROPE_DIM):
        raise unmodeled("attention_c8_abi_shape", "Bind the packed latent C8+RoPE cache geometry (Dk=576, Dv=512)")


def _validate_gbsa_abi(contract: AttentionContract) -> None:
    if contract.kv_layout not in GBSA_LAYOUTS:
        raise unmodeled("attention_gbsa_layout", f"GBSA rejects kv_layout={contract.kv_layout}; bind a valid CANN GBSA layout")
    if contract.head_dim != contract.v_head_dim:
        raise unmodeled("attention_gbsa_head_dim", "GBSA requires head_dim == v_head_dim")
    if contract.quant_mode == QUANT_MODE_MXFP8:
        if contract.head_dim not in GBSA_MXFP8_HEAD_DIMS:
            raise unmodeled("attention_gbsa_mxfp8_head_dim", "Arch35 GBSA MXFP8 requires head_dim in {192, 256}")
    elif contract.head_dim > GBSA_MAX_HEAD_DIM:
        raise unmodeled("attention_gbsa_head_dim", f"Arch35 GBSA requires head_dim <= {GBSA_MAX_HEAD_DIM}")


def _c8_vectorized_block_table_bytes(
    contract: AttentionContract, ctx: OperatorPlanningContext, counts: tuple[int, ...],
) -> int:
    """Return full-row page-table transfer bytes when CANN/vLLM vectorizes PA address lookup."""
    if contract.kv_layout != "PA_BLK_BSND" or not contract.causal:
        return 0
    block_size = contract.block_size
    if block_size <= 0 or (block_size & (block_size - 1)) != 0:
        return 0
    sparse_count = contract.selected_tokens if contract.selected_tokens is not None else max(counts, default=0)
    if sparse_count <= 0:
        return 0
    is_vllm = contract.abi == C8_VLLM_ABI
    if is_vllm:
        if contract.quant_mode != QUANT_MODE_INT8_C8:
            return 0
        split_g = contract.num_heads > SFA_SPLIT_G_HEADS
        no_rope = contract.head_dim == C8_LATENT_DIM
        if split_g and not no_rope:
            return 0
    elif sparse_count % FIA_TILE != 0:
        return 0
    max_kv = max((kv for _, kv in contract.sequences), default=max(counts, default=0))
    max_blocks = ceil(max_kv / block_size)
    if max_blocks <= 0:
        return 0
    ub_size = ctx.geometry.buffers.get("ub") or dict(ctx.datapath.buffers).get("ub", 0)
    if ub_size <= 0:
        return 0
    blk_ub = ceil((max_blocks * INDEX_BYTES) / C8_BLOCK_TABLE_ALIGN_BYTES) * C8_BLOCK_TABLE_ALIGN_BYTES
    aligned_sparse = ceil(sparse_count / FIA_TILE) * FIA_TILE
    if is_vllm:
        ub_need = blk_ub + aligned_sparse * (INDEX_BYTES + SCALE_ADDRESS_BYTES)
    else:
        idx_ub = ceil((sparse_count * INDEX_BYTES) / SECTOR_BYTES) * SECTOR_BYTES
        ub_need = blk_ub + idx_ub + aligned_sparse * SCALE_ADDRESS_BYTES
    if ub_need > ub_size:
        return 0
    return max_blocks * INDEX_BYTES


def _c8_vec_bt_first_visits(
    contract: AttentionContract, ctx: OperatorPlanningContext, counts: tuple[int, ...], owner_count: int,
) -> frozenset[int]:
    """Identify queries where each participating owner first loads a batch's page-table row."""
    if contract.sequences:
        batch_ids = [b for b, (q, _) in enumerate(contract.sequences) for _ in range(q)]
    else:
        batch = min(contract.batch_size, len(counts)) if counts else 1
        base_q, extra_q = divmod(len(counts), batch)
        batch_ids = [b for b in range(batch) for _ in range(base_q + (b < extra_q))]
    splits = min(owner_count, contract.splits, max(1, max(counts, default=1))) if ctx.requests else 1
    groups = max(1, owner_count // splits) if ctx.requests else owner_count
    seen: set[tuple[int, int]] = set()
    first_visits: set[int] = set()
    for q_idx, rows in enumerate(counts):
        if not rows:
            continue
        owner = ((q_idx % groups) * splits) if ctx.requests else (q_idx % owner_count)
        key = (owner, batch_ids[q_idx])
        if key not in seen:
            seen.add(key)
            first_visits.add(q_idx)
    return frozenset(first_visits)


def _packed_kv(op, contract, ctx, sources):
    _validate_c8_abi(contract)
    if contract.abi == C8_VLLM_ABI:
        size = C8_VLLM_ROW_BYTES if contract.head_dim == C8_LATENT_DIM else C8_ROW_BYTES
    elif contract.quant_mode == QUANT_MODE_TQ4:
        size = TQ4_ROW_BYTES
    else:
        size = C8_ROW_BYTES
    counts = tuple(contract.row_counts())
    capacity = _l2_capacity(ctx)
    key = None
    if capacity and contract.selection_lists and contract.kv_tensor_id:
        key = ReuseKey(ctx.geometry.rank or 0, contract.kv_tensor_id, contract.kv_version)
    owners = owner_pairs(ctx, sum(min(contract.splits, max(1, count)) for count in counts))
    vec_bt_bytes = _c8_vectorized_block_table_bytes(contract, ctx, counts)
    first_visits = _c8_vec_bt_first_visits(contract, ctx, counts, len(owners)) if vec_bt_bytes else frozenset()
    loop = _PackedAttention(
        contract, ctx, op.dtype, size, counts, owners, key, capacity,
        vectorized_block_table_bytes=vec_bt_bytes, vec_bt_first_visit=first_visits,
    )
    producer, finish = {}, {}
    if contract.selection_lists or vec_bt_bytes:
        launches = ((query, count, 1) for query, count in enumerate(counts))
    else:
        launches = ((0, count, repeats) for count, repeats in Counter(counts).items())
    for query, count, repeats in launches:
        intervals = tuple(loop.intervals(count))
        for split_idx, (start, end) in enumerate(intervals):
            for phase in loop.split_phases(query, start, end, len(intervals), split_index=split_idx):
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


@dataclass(frozen=True, slots=True)
class _SharedReadWait:
    """Wait for a sibling owner's shared KV transfer when group_rows > 1."""

    key: tuple[int, int]
    reader: AttentionMemoryRead | None

    def start(self, engine, name, ready, *, owner=""):
        events = getattr(engine, "_sparse_shared_kv_events", None)
        if events is None:
            events = engine._sparse_shared_kv_events = {}
        done = events.get(self.key)
        if done is None:
            done = events[self.key] = engine.future(f"{name}:shared_kv")
        if self.reader is not None:
            transfer = self.reader.start(engine, name, ready, owner=owner)
            engine.on_complete(transfer, lambda _: engine.resolve(done))
            return done
        joined = engine.future(f"{name}:wait")
        engine.submit(f"{name}:join", after=(ready, done), owner=owner,
                      on_complete=lambda _: engine.resolve(joined), native_hook=True)
        return joined


@dataclass(frozen=True, slots=True)
class _SparseSharedKvMemory:
    """Defer shared-prefix L2 residency fills to Engine transfer completion."""

    phase_name: str
    patterns: tuple[MemoryAccessPattern, ...]
    ctx: OperatorPlanningContext
    capacity: int
    per_owner_assigned: bool = False

    def __call__(self, context, owner_index, owner_count, core, aic_count, deps):
        if not self.patterns:
            return Stage(self.phase_name, {}, after=deps, keep_empty=True)
        hbm = context.resource_names["hbm"]
        pipe_role = "mte2_aic" if "mte2_aic" in context.resource_names else "mte2"
        pipe_units = context.resource_units.get(pipe_role, 1)
        pipe_rate = context.rates[context.resource_names[pipe_role]] / pipe_units
        owner_unit = unit_group(core, aic_count, pipe_units)[0]
        if self.per_owner_assigned:
            assigned = tuple(pat for pat in self.patterns if pat.owner == owner_index)
            if not assigned:
                return Stage(self.phase_name, {}, after=deps, keep_empty=True)
        elif owner_count > len(self.patterns):
            pat_idx = owner_index % len(self.patterns)
            is_leader = owner_index < len(self.patterns)
            reader = None
            if is_leader:
                pat = replace(self.patterns[pat_idx], owner=owner_unit)
                bus = HbmBusSpec(pat.die, 1, 1, 0, row_r=hbm, ctr_r=hbm)
                bound = context.bind_access(
                    pat, hbm_bus=bus, copy_hit_bytes_per_ms=pipe_rate, copy_miss_bytes_per_ms=pipe_rate,
                )
                reader = AttentionMemoryRead(pat, bound, self.capacity, hbm, self.ctx)
            return Stage(
                self.phase_name, after=deps,
                program=_SharedReadWait((id(self), pat_idx), reader),
                keep_empty=True,
            )
        else:
            assigned = self.patterns[owner_index::owner_count]
        stages = []
        for idx, pattern in enumerate(assigned):
            pat = replace(pattern, owner=owner_unit)
            bus = HbmBusSpec(pat.die, 1, 1, 0, row_r=hbm, ctr_r=hbm)
            bound = context.bind_access(
                pat, hbm_bus=bus, copy_hit_bytes_per_ms=pipe_rate, copy_miss_bytes_per_ms=pipe_rate,
            )
            reader = AttentionMemoryRead(pat, bound, self.capacity, hbm, self.ctx)
            stages.append(Stage(f"{self.phase_name}_{idx}", program=reader))
        if len(stages) == 1:
            return Stage(self.phase_name, after=deps, program=stages[0].program, keep_empty=True)
        return Stage(self.phase_name, after=deps, program=KernelProgram(tuple(stages)), keep_empty=True)


def _sparse_tile_specs(
    contract: AttentionContract, ctx: OperatorPlanningContext, *, tiled: bool,
) -> tuple[tuple[int, int, int, int, int, int | None], ...]:
    """Return (seq_idx, tile_q, tile_kv_rows, tile_pair_rows, base, stride) per task tile."""
    if tiled:
        q_block = contract.query_block_size or contract.block_size
        # CANN MindIE BlockSparseAttention caps each AIC query tile at FIA_TILE (128).
        q_tile = min(q_block, FIA_TILE) if contract.abi in MINDIE_ABIS else q_block
    else:
        q_block = q_tile = 1
    spans = planning_sequence_spans(ctx, contract.kv_layout)
    if spans:
        seq_meta: list[tuple[int, int | None, int, int | None]] = [
            (q, kv, base, request.physical_kv_len if contract.kv_layout == "BNSD" else None)
            for request, (q, kv, _, base, _) in zip(ctx.requests, spans, strict=True)
        ]
    elif contract.sequences:
        seq_meta = [(q, kv, 0, None) for q, kv in contract.sequences]
    else:
        batch = min(contract.batch_size, contract.query_tokens)
        base_q, extra_q = divmod(contract.query_tokens, batch)
        seq_meta = [(base_q + (i < extra_q), None, 0, None) for i in range(batch)]
    counts = iter(contract.row_counts())
    specs = []
    for seq_idx, (seq_q, seq_kv, base, stride) in enumerate(seq_meta):
        for block_start in range(0, seq_q, q_block):
            block_end = min(block_start + q_block, seq_q)
            for tile_start in range(block_start, block_end, q_tile):
                tile_q = min(q_tile, block_end - tile_start)
                tile_counts = tuple(islice(counts, tile_q))
                tile_kv = max(tile_counts, default=0)
                if (tile_kv > 0 and tile_q > 1 and seq_kv is not None
                        and contract.causal and contract.window_size):
                    # End-aligned sliding-window union spans [first_lo, last_end).
                    first_lo = (seq_kv - seq_q + tile_start + 1) - tile_counts[0]
                    union_rows = (seq_kv - seq_q + tile_start + tile_q) - first_lo
                    if contract.selected_tokens is not None:
                        union_rows = min(union_rows, contract.selected_tokens)
                    tile_kv = max(tile_kv, union_rows)
                specs.append((seq_idx, tile_q, tile_kv, sum(tile_counts), base, stride))
    return tuple(specs)


def _gbsa_tile_params(
    contract: AttentionContract, *, gbsa_fp8: bool, gbsa_mxfp8: bool,
) -> tuple[int, int, int]:
    """Return CANN Arch35 GBSA (q_base_tile, kv_base_tile, descriptor_rows)."""
    group_size = contract.num_heads // contract.num_kv_heads
    is_union = contract.query_block_size is not None and contract.query_block_size > 1
    tile_group = group_size * contract.query_block_size if is_union else group_size
    if is_union and gbsa_fp8 and not (tile_group > GBSA_Q_TILE_LARGE and contract.head_dim <= FIA_TILE):
        q_base = kv_base = FIA_TILE
    else:
        if tile_group <= GBSA_Q_TILE_SMALL:
            q_base = GBSA_Q_TILE_SMALL
        elif tile_group <= GBSA_Q_TILE_MEDIUM:
            q_base = GBSA_Q_TILE_MEDIUM
        elif tile_group <= GBSA_Q_TILE_LARGE:
            q_base = GBSA_Q_TILE_LARGE
        else:
            q_base = FIA_TILE
        if gbsa_mxfp8:
            kv_base = GBSA_KV_TILE_LARGE if group_size <= GBSA_Q_TILE_MEDIUM else GBSA_KV_TILE_MEDIUM
        elif contract.head_dim <= GBSA_MAX_HEAD_DIM:
            if q_base <= GBSA_Q_TILE_MEDIUM:
                kv_base = GBSA_KV_TILE_LARGE
            elif q_base <= GBSA_Q_TILE_LARGE or (is_union and gbsa_fp8 and contract.head_dim <= FIA_TILE):
                kv_base = GBSA_KV_TILE_MEDIUM
            else:
                kv_base = FIA_TILE
        else:
            if q_base <= GBSA_Q_TILE_SMALL:
                kv_base = GBSA_KV_TILE_WIDE
            elif q_base <= GBSA_Q_TILE_MEDIUM:
                kv_base = GBSA_KV_TILE_MEDIUM
            else:
                kv_base = FIA_TILE
    # SoftmaxUnion adds exact-mask UB cost; CANN halves kvBaseTile > 256 when D > 128.
    if is_union and kv_base > GBSA_KV_TILE_MEDIUM and contract.head_dim > FIA_TILE:
        kv_base //= 2
    is_paged = contract.kv_layout in PAGED_LAYOUTS
    if gbsa_mxfp8:
        if kv_base == GBSA_KV_TILE_MEDIUM or (contract.head_dim == GBSA_MXFP8_SMALL_HEAD_DIM and is_paged):
            desc_rows = FIA_TILE
        else:
            desc_rows = GBSA_KV_TILE_MEDIUM
    elif contract.head_dim <= FIA_TILE:
        desc_rows = (
            GBSA_KV_TILE_MEDIUM
            if (q_base <= GBSA_Q_TILE_LARGE or (is_union and gbsa_fp8 and kv_base == GBSA_KV_TILE_MEDIUM))
            else FIA_TILE
        )
    else:
        desc_rows = FIA_TILE
    return q_base, kv_base, desc_rows


@dataclass(frozen=True, slots=True)
class _GbsaFdPlan:
    work_items: int
    q_rows: int
    combine_partials: int
    kv_chunks: tuple[int, ...]
    pv_head_tiles: int
    core_extents: tuple[tuple[int, int, int, int, int | None], ...]
    core_visits: frozenset[tuple[int, int, int]]


def _gbsa_fd_schedule(
    contract: AttentionContract,
    ctx: OperatorPlanningContext,
    tile_specs: tuple[tuple[int, int, int, int, int, int | None], ...],
    *,
    kv_tile: int,
    gbsa_mxfp8: bool,
) -> _GbsaFdPlan | None:
    """Return CANN Arch35 GBSA FlashDecode schedule when metadata enables FD."""
    group_size = contract.num_heads // contract.num_kv_heads
    is_union = contract.query_block_size is not None and contract.query_block_size > 1
    if (contract.splits != 1 or is_union or group_size > GBSA_MAX_GROUP_SIZE
            or contract.head_dim > GBSA_MAX_HEAD_DIM):
        return None
    if gbsa_mxfp8 and (contract.head_dim not in GBSA_MXFP8_HEAD_DIMS
                       or contract.block_size not in GBSA_MXFP8_FD_BLOCK_SIZES):
        return None
    base_tasks = [
        (seq_idx, kv_head, tile_kv, base, stride)
        for seq_idx, _, tile_kv, _, base, stride in tile_specs
        for kv_head in range(contract.num_kv_heads)
    ]
    task_blocks = [ceil(tok / contract.block_size) for _, _, tok, _, _ in base_tasks]
    sa_tasks = len(base_tasks)
    total_valid_blocks = sum(task_blocks)
    block_index_stride = (
        ceil(contract.selected_tokens / contract.block_size)
        if contract.selected_tokens is not None else max(task_blocks, default=0)
    )
    if sa_tasks <= 0 or block_index_stride < GBSA_FD_MIN_BLOCKS or total_valid_blocks <= 0:
        return None
    aic_count = len(owner_pairs(ctx, ctx.effective_aic_cores))
    sa_used_cores = min(sa_tasks, aic_count)
    if sa_tasks * GBSA_FD_UTIL_DEN >= aic_count * GBSA_FD_UTIL_NUM:
        return None
    if gbsa_mxfp8:
        if ceil(sum(tok for _, _, tok, _, _ in base_tasks) / kv_tile) <= sa_used_cores:
            return None
        # Metadata scheduler uses conservative PA descriptor boundary even on dense inputs.
        fd_desc_rows = (
            FIA_TILE
            if (kv_tile == GBSA_KV_TILE_MEDIUM or contract.head_dim == GBSA_MXFP8_SMALL_HEAD_DIM)
            else GBSA_KV_TILE_MEDIUM
        )
        blocks_per_unit = max(min(kv_tile, fd_desc_rows) // contract.block_size, 1)
        task_units = [ceil(max(1, b) / blocks_per_unit) for b in task_blocks]
        total_units = sum(task_units)
        units_per_core = ceil(total_units / aic_count)
        active_cores = ceil(total_units / units_per_core)
        if active_cores <= sa_used_cores:
            return None
    else:
        blocks_per_unit = 1
        task_units = [max(1, b) for b in task_blocks]
        total_units = sum(task_units)
        if total_units <= sa_used_cores:
            return None
        head_tiles = ceil(group_size / GBSA_Q_TILE_SMALL)
        total_cost = sum(
            GBSA_FD_COST_HEAD_TILE * head_tiles + GBSA_FD_COST_BLOCK * b
            + GBSA_FD_COST_HEAD_BLOCK * head_tiles * b
            for b in task_blocks if b > 0
        )
        max_core = min(aic_count, total_valid_blocks)
        best_core = max(1, min(ceil(sqrt(0.25 + total_cost / GBSA_FD_LAUNCH_COST) - 0.5), max_core))
        units_per_core = ceil(total_units / best_core)
        active_cores = ceil(total_units / units_per_core)
    q_rows = combine_partials = pv_head_tiles = 0
    kv_chunks: list[int] = []
    core_extents: list[tuple[int, int, int, int, int | None]] = []
    core_visits: set[tuple[int, int, int]] = set()
    prefix = 0
    for (seq_idx, kv_head, tok, base, stride), blocks, units in zip(base_tasks, task_blocks, task_units, strict=True):
        first_core = prefix // units_per_core
        last_core = (prefix + units - 1) // units_per_core
        partials = last_core - first_core + 1
        q_rows += partials * group_size
        if partials > 1:
            combine_partials += partials * group_size
        if tok > 0:
            for core in range(first_core, last_core + 1):
                u_start = max(0, core * units_per_core - prefix)
                u_end = min(units, (core + 1) * units_per_core - prefix)
                b_start = min(blocks, u_start * blocks_per_unit)
                b_end = min(blocks, u_end * blocks_per_unit)
                c_tok = max(0, min(tok, b_end * contract.block_size) - b_start * contract.block_size)
                kv_chunks.append(c_tok)
                if c_tok > 0:
                    core_extents.append((core, kv_head, c_tok, base + b_start * contract.block_size, stride))
                    core_visits.add((core, seq_idx, kv_head))
                    pv_head_tiles += ceil(c_tok / kv_tile) * group_size
        else:
            kv_chunks.append(0)
        prefix += units
    return _GbsaFdPlan(
        active_cores, q_rows, combine_partials, tuple(kv_chunks), pv_head_tiles,
        tuple(core_extents), frozenset(core_visits),
    )


def _gbsa_offset_descriptor_bytes(
    contract: AttentionContract,
    kv_chunks: tuple[int, ...],
    *,
    kv_tile: int,
    descriptor_rows: int,
    gbsa_mxfp8: bool,
) -> int:
    """Size GBSA descriptor MTE3/HBM transfers from coalesced spans plus header and alignment."""
    is_paged = contract.kv_layout in PAGED_LAYOUTS
    # MXFP8 paged cache uses compact 2-word [offset, meta] records shared by K and V.
    record_words = GBSA_DENSE_RECORD_WORDS if (not is_paged or gbsa_mxfp8) else GBSA_PAGED_RECORD_WORDS
    page_size = max(FIA_TILE, contract.block_size)
    total_words = 0
    for chunk_tokens in kv_chunks:
        full_tiles, tail_tile = divmod(chunk_tokens, kv_tile)
        for tile_tokens, repeats in ((kv_tile, full_tiles), (tail_tile, 1)):
            if not tile_tokens or not repeats:
                continue
            full_groups, tail_group = divmod(tile_tokens, descriptor_rows)
            for group_tokens, grp_repeats in ((descriptor_rows, full_groups), (tail_group, 1)):
                if not group_tokens or not grp_repeats:
                    continue
                if is_paged:
                    pages = ceil(group_tokens / page_size)
                    # Regular paged epilogue pairs adjacent full blockY-sized pages into one record.
                    records = ceil(pages / 2) if (not gbsa_mxfp8 and contract.block_size == page_size) else pages
                else:
                    records = 1
                copy_words = (
                    ceil((GBSA_OFFSET_HEADER_WORDS + records * record_words) / GBSA_OFFSET_ALIGN_WORDS)
                    * GBSA_OFFSET_ALIGN_WORDS
                )
                total_words += copy_words * grp_repeats * repeats
    return total_words * SCALE_ADDRESS_BYTES


def _sparse_kv_load(
    contract: AttentionContract, ctx: OperatorPlanningContext, dim: int, tensor: str,
    *, native_fp8: bool = False, gbsa_fp8: bool = False, is_value: bool = False,
    shared_l2: AdmissionMemoryView | None = None,
    extents: tuple[tuple[int, int, int | None], ...] | None = None,
    core_extents: tuple[tuple[int, int, int, int, int | None], ...] | None = None,
    task_heads: int | None = None,
) -> tuple[Work, tuple[MemoryAccessPattern, ...]]:
    mode = contract.quant_mode
    elem = BF16_BYTES if mode == QUANT_MODE_BF16 else 1
    if native_fp8 or gbsa_fp8:
        scale_bytes = 0  # Native/direct FP8 scalar operands have no per-group BF16 scales.
    elif mode == QUANT_MODE_INT8_C8:
        scale_bytes = FP32_BYTES
    elif mode == QUANT_MODE_FP8_C8:
        scale_bytes = ceil(dim / FP8_SCALE_GROUP) * BF16_BYTES
    elif mode == QUANT_MODE_MXFP8:
        # GBSA MXFP8 stores K scales per 32-elem group along D on AIC; V uses
        # per-channel scales loaded separately on AIV (SetValueScale).
        scale_bytes = 0 if (contract.abi == GBSA_ABI and is_value) else ceil(dim / MXFP8_SCALE_GROUP)
    else:
        scale_bytes = 0
    hbm = copied = hit = 0
    row_bytes = contract.num_kv_heads * dim * elem
    head_count = contract.num_kv_heads if task_heads is None else task_heads
    head_group = max(1, head_count // contract.num_kv_heads)
    kv_extents = extents if extents is not None else tuple(
        (rows, base, stride) for _, _, rows, _, base, stride in _sparse_tile_specs(contract, ctx, tiled=False)
    )
    if shared_l2 is not None and not scale_bytes:
        die = ctx.geometry.rank or 0
        key = ReuseKey(die, f"{contract.kv_tensor_id}:{tensor}", contract.kv_version, reuse_stream=tensor)
        bus = HbmBusSpec(die, 1, 1, 0)
        # Preview intra-op shared-prefix hits on a private copy; caller view stays untouched until Engine commit.
        preview_l2 = copy.deepcopy(shared_l2)
        patterns = []
        if core_extents is not None:
            entries = core_extents
        else:
            entries = tuple(
                (owner, owner // head_group, rows, base, stride)
                for rows, base, stride in kv_extents if rows
                for owner in range(head_count)
            )
        for owner, kv_head, rows, base, stride in entries:
            if not rows:
                continue
            plan = bsh_kv_access_plan(
                contract.num_kv_heads, dim, elem, contract.kv_layout,
                rows=rows, first_head=kv_head, block_size=contract.block_size,
                base_byte=base * row_bytes, tokens_per_head=stride,
            )
            pattern = MemoryAccessPattern(
                f"{contract.kv_tensor_id}:{tensor}", contract.kv_version, die, owner, "read", "mte2_aic",
                rows * dim * elem, plan, reuse_key=key,
            )
            ledger = bind_memory_access(
                pattern, preview_l2, hbm_bus=bus, copy_hit_bytes_per_ms=1,
                copy_miss_bytes_per_ms=1, l2_bytes_per_ms=1, pipe_row_id="mte2_aic",
                l2_row_id="l2", commit=True,
            ).ledger
            copied += ledger.aligned_transfer_bytes
            hbm += ledger.dram_sector_bytes
            hit += ledger.l2_hit_bytes
            patterns.append(pattern)
        return read_work(ctx, copied, hbm=hbm, l2=hit), tuple(patterns)

    for rows, base, stride in kv_extents:
        block = contract.block_size
        full, tail = divmod(rows, block)
        for count, repeats in ((block, full), (tail, 1)):
            if not count or not repeats:
                continue
            for owner in range(head_count):
                plan = bsh_kv_access_plan(contract.num_kv_heads, dim, elem, contract.kv_layout,
                                          rows=count, first_head=owner // head_group, block_size=block,
                                          base_byte=base * row_bytes, tokens_per_head=stride)
                plan = AddressRunPlan(plan.runs, scale_bytes=count * scale_bytes)
                pattern = MemoryAccessPattern(tensor, 0, 0, owner, "read", "mte2_aic",
                                              count * (dim * elem + scale_bytes), plan)
                hbm += pattern.runs.dram_sector_and_service()[0] * repeats
                copied += pattern.runs.aligned_copy_bytes() * repeats
    return read_work(ctx, copied, hbm=hbm), ()


def _mxfp4_inputs(contract, q_rows, kv_tile_rows, tile_kv_counts):
    dk, dv = contract.head_dim, contract.v_head_dim
    heads, block = contract.num_heads, contract.block_size
    counts = sum(ceil(n / MXFP4_BLOCK_ALIGNMENT) for n in tile_kv_counts) * heads
    payloads = {
        "query": (q_rows * (dk // 2), q_rows * (dk // MXFP4_SCALE_GROUP)),
        "key": (kv_tile_rows * heads * (dk // 2), kv_tile_rows * heads * (dk // MXFP4_SCALE_GROUP)),
        "value": (kv_tile_rows * heads * (dv // 2), counts * dv * 2),
    }
    return {
        "quant_mode": contract.quant_mode, "dst_type_max": contract.dst_type_max,
        "block_shape": (contract.query_block_size or block, block),
        "packed_dtype": "fp4_e2m1", "scale_dtype": "fp8_e8m0", "output_dtype": "bf16",
        "operand_bytes": {name: data for name, (data, _) in payloads.items()},
        "scale_bytes": {name: scales for name, (_, scales) in payloads.items()},
    }


def _mindie_fp8_kv_tile(contract: AttentionContract) -> int:
    kv_block = contract.block_size
    preferred = FP8_MIXED_KV_TILE if contract.head_dim <= FIA_TILE else FIA_TILE
    if kv_block == FP8_MAX_KV_TILE and preferred == FP8_MIXED_KV_TILE:
        return kv_block
    if kv_block % preferred == 0:
        return preferred
    if preferred > FIA_TILE and kv_block % FIA_TILE == 0:
        # CANN's mixed FP8 path prefers 128 here, even if a larger divisor fits.
        return FIA_TILE
    return next(width for width in range(min(kv_block, preferred), 0, -1) if kv_block % width == 0)


def _mindie_fp8_scale_bytes(contract):
    # The kernel reads one Q scalar per (batch, head, Q tile), and one K
    # and V scalar per selected S2 tile in that task.
    q_block = contract.query_block_size or contract.block_size
    q_tile = min(q_block, FIA_TILE)
    kv_tile = _mindie_fp8_kv_tile(contract)
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


def _load_sparse_kv_pair(
    contract: AttentionContract,
    ctx: OperatorPlanningContext,
    *,
    mode: int,
    nope: bool,
    mxfp4: bool,
    native_fp8: bool,
    gbsa_fp8: bool,
    packed_bytes: dict[str, int],
    extents: tuple[tuple[int, int, int | None], ...],
    core_extents: tuple[tuple[int, int, int, int, int | None], ...] | None = None,
    task_heads: int | None = None,
) -> tuple[Work, Work | None, _SparseSharedKvMemory | None, _SparseSharedKvMemory | None]:
    capacity = _l2_capacity(ctx)
    shared_l2 = (
        ctx.admission_memory_view
        if (contract.shared_prefix and contract.kv_tensor_id and mode == QUANT_MODE_BF16
            and ctx.admission_memory_view is not None and capacity > 0)
        else None
    )
    if mxfp4:
        k_load, k_pats = read_work(ctx, packed_bytes["key"]), ()
    else:
        k_load, k_pats = _sparse_kv_load(
            contract, ctx, contract.head_dim, "key", native_fp8=native_fp8, gbsa_fp8=gbsa_fp8,
            shared_l2=shared_l2, extents=extents, core_extents=core_extents, task_heads=task_heads,
        )
    if mxfp4:
        v_load, v_pats = read_work(ctx, packed_bytes["value"]), ()
    elif not nope:
        v_load, v_pats = _sparse_kv_load(
            contract, ctx, contract.v_head_dim, "value", native_fp8=native_fp8, gbsa_fp8=gbsa_fp8,
            is_value=True, shared_l2=shared_l2, extents=extents, core_extents=core_extents, task_heads=task_heads,
        )
    else:
        v_load, v_pats = None, ()
    per_owner = core_extents is not None
    k_mem = _SparseSharedKvMemory("key_load", k_pats, ctx, capacity, per_owner_assigned=per_owner) if k_pats else None
    v_mem = _SparseSharedKvMemory("value_load", v_pats, ctx, capacity, per_owner_assigned=per_owner) if v_pats else None
    return k_load, v_load, k_mem, v_mem


def _gbsa_mxfp8_v_scale_bytes(
    contract: AttentionContract,
    ctx: OperatorPlanningContext,
    tile_specs: tuple[tuple[int, int, int, int, int, int | None], ...],
    work_items: int,
    gbsa_fd: _GbsaFdPlan | None,
) -> int:
    """Charge per-AIV private-UB SetValueScale loads once per distinct (owner, batch, kv_head)."""
    owners = owner_pairs(ctx, work_items)
    if gbsa_fd is not None:
        visits = gbsa_fd.core_visits
    else:
        owner_count = len(owners)
        visits_set: set[tuple[int, int, int]] = set()
        task_idx = 0
        for seq_idx, _, tile_kv, _, _, _ in tile_specs:
            splits = min(contract.splits, max(1, ceil(tile_kv / contract.block_size)))
            for kv_head in range(contract.num_kv_heads):
                for _ in range(splits):
                    if tile_kv > 0:
                        visits_set.add((task_idx % owner_count, seq_idx, kv_head))
                    task_idx += 1
        visits = frozenset(visits_set)
    aiv_loads = sum(len(owners[owner_idx][1]) for owner_idx, _, _ in visits)
    return aiv_loads * contract.v_head_dim * MXFP8_V_CHANNEL_SCALE_BYTES


def _gbsa_fd_output_phases(
    ctx: OperatorPlanningContext, contract: AttentionContract, combine_partials: int,
) -> tuple[AttentionPhase, ...]:
    """Emit CANN GBSA FlashDecode partial write/read/merge (1 FP32 LSE per partial) + output."""
    phases: list[AttentionPhase] = []
    previous: tuple[str, ...] = ("online_output",)
    if combine_partials > 0:
        payload = combine_partials * (contract.v_head_dim + 1) * FP32_BYTES
        phases.extend((
            AttentionPhase("split_write", Work(resources={"mte3": payload, "hbm": payload}), previous),
            AttentionPhase("split_read", read_work(ctx, payload, aiv=True), ("split_write",)),
            AttentionPhase("split_merge", Work(resources={
                "vector": combine_partials * (2 * contract.v_head_dim + SOFTMAX_OPS),
            }), ("split_read",)),
        ))
        previous = ("split_merge",)
    rows = contract.query_tokens * contract.num_heads
    result = rows * (contract.v_head_dim * BF16_BYTES + (FP32_BYTES if contract.return_lse else 0))
    phases.append(AttentionPhase("output", Work(resources={"mte3": result, "hbm": result}), previous))
    return tuple(phases)


def _validate_sparse_abi(
    op: OpSpec, contract: AttentionContract, sources: tuple[SourceRef, ...],
) -> tuple[SourceRef, ...]:
    abi, mode = contract.abi, contract.quant_mode
    mindie = abi in MINDIE_ABIS
    native_fp8 = mindie and mode == QUANT_MODE_MINDIE_FP8
    mxfp4 = mindie and mode in {QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX}
    if abi == GBSA_ABI:
        _validate_gbsa_abi(contract)
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
    if (abi == NOPE_ABI and (mode != QUANT_MODE_BF16 or contract.head_dim != C8_LATENT_DIM
                             or contract.v_head_dim != C8_LATENT_DIM or contract.num_kv_heads != 1)):
        raise unmodeled("attention_nope_shape", "Bind a shared, nonquantized NoPE latent cache (Dk=Dv=512)")
    if mindie and (contract.kv_layout not in {"TND", "BNSD"}
                   or (not mxfp4 and not native_fp8 and mode != QUANT_MODE_BF16)
                   or (abi == MINDIE_V1_ABI and native_fp8)):
        raise unmodeled("attention_mindie_sd_abi", f"Bind a supported layout and storage format for {abi}")
    if mode == QUANT_MODE_TQ4 and not mxfp4:
        raise unmodeled("attention_tq4_abi", f"TQ4 packed latent cache requires {C8_ABI}")
    return sources


@dataclass(frozen=True, slots=True)
class _SparseSchedule:
    tile_specs: tuple[tuple[int, int, int, int, int, int], ...]
    kv_tile: int
    descriptor_rows: int
    pairs: int
    kv_tile_rows: int
    kv_rows: int
    blocks: int
    task_heads: int
    work_items: int
    q_rows: int
    pv_head_tiles: int
    gbsa_kv_chunks: tuple[int, ...]
    gbsa_fd: _GbsaFdPlan | None


def _build_sparse_schedule(
    contract: AttentionContract, ctx: OperatorPlanningContext, *,
    mindie: bool, native_fp8: bool, gbsa_fp8: bool, gbsa_mxfp8: bool, mxfp4: bool,
) -> _SparseSchedule:
    abi = contract.abi
    tiled_sparse = mindie or (abi == GBSA_ABI and contract.query_block_size is not None)
    tile_specs = _sparse_tile_specs(contract, ctx, tiled=tiled_sparse)
    if abi == GBSA_ABI:
        _, kv_tile, descriptor_rows = _gbsa_tile_params(contract, gbsa_fp8=gbsa_fp8, gbsa_mxfp8=gbsa_mxfp8)
    elif native_fp8:
        kv_tile = descriptor_rows = _mindie_fp8_kv_tile(contract)
    elif mxfp4 and contract.head_dim <= FIA_TILE:
        kv_tile = descriptor_rows = FP8_MIXED_KV_TILE
    else:
        kv_tile = descriptor_rows = FIA_TILE
    pair_rows = kv_tile_rows = blocks = tiles = q_split_rows = task_tiles = 0
    for _, tile_q, tile_kv, tile_pairs, _, _ in tile_specs:
        pair_rows += tile_pairs
        kv_tile_rows += tile_kv
        tile_blocks = ceil(tile_kv / contract.block_size)
        blocks += tile_blocks
        split_cap = max(1, tile_blocks) if abi == GBSA_ABI else max(1, tile_kv)
        splits = min(contract.splits, split_cap)
        if splits > 1 and abi == GBSA_ABI:
            base_c, extra_c = divmod(tile_kv, splits)
            tiles += tile_q * sum(ceil((base_c + (i < extra_c)) / kv_tile) for i in range(splits))
        else:
            tiles += tile_q * ceil(tile_kv / kv_tile)
        q_split_rows += tile_q * splits
        task_tiles += splits
    task_heads = contract.num_heads if mindie else contract.num_kv_heads
    group_rows = (
        ceil(contract.num_heads / (contract.num_kv_heads * GBSA_MAX_GROUP_SIZE))
        if abi == GBSA_ABI else 1
    )
    gbsa_fd = (
        _gbsa_fd_schedule(contract, ctx, tile_specs, kv_tile=kv_tile, gbsa_mxfp8=gbsa_mxfp8)
        if abi == GBSA_ABI else None
    )
    if gbsa_fd is not None:
        work_items, q_rows = gbsa_fd.work_items, gbsa_fd.q_rows
        gbsa_kv_chunks, pv_head_tiles = gbsa_fd.kv_chunks, gbsa_fd.pv_head_tiles
    else:
        work_items = task_tiles * task_heads * group_rows
        q_rows = q_split_rows * contract.num_heads
        pv_head_tiles = tiles * contract.num_heads
        if abi == GBSA_ABI:
            chunks: list[int] = []
            for _, _, tile_kv, _, _, _ in tile_specs:
                splits = min(contract.splits, max(1, ceil(tile_kv / contract.block_size)))
                base_c, extra_c = divmod(tile_kv, splits)
                for _ in range(contract.num_kv_heads * group_rows):
                    chunks.extend(base_c + (i < extra_c) for i in range(splits))
            gbsa_kv_chunks = tuple(chunks)
        else:
            gbsa_kv_chunks = ()
    return _SparseSchedule(
        tile_specs=tile_specs, kv_tile=kv_tile, descriptor_rows=descriptor_rows,
        pairs=pair_rows * contract.num_heads, kv_tile_rows=kv_tile_rows,
        kv_rows=kv_tile_rows * task_heads, blocks=blocks, task_heads=task_heads,
        work_items=work_items, q_rows=q_rows, pv_head_tiles=pv_head_tiles,
        gbsa_kv_chunks=gbsa_kv_chunks, gbsa_fd=gbsa_fd,
    )


def _sparse(op: OpSpec, contract: AttentionContract, ctx: OperatorPlanningContext,
            sources: tuple[SourceRef, ...]) -> OperatorPlan:
    abi, mode = contract.abi, contract.quant_mode
    nope = abi == NOPE_ABI
    mindie = abi in MINDIE_ABIS
    native_fp8 = mindie and mode == QUANT_MODE_MINDIE_FP8
    gbsa_fp8 = abi == GBSA_ABI and mode == QUANT_MODE_FP8_C8
    gbsa_mxfp8 = abi == GBSA_ABI and mode == QUANT_MODE_MXFP8
    mxfp4 = mindie and mode in {QUANT_MODE_MXFP4_OCP, QUANT_MODE_MXFP4_CX}
    sources = _validate_sparse_abi(op, contract, sources)
    dtype = ("mxfp4" if mxfp4 else
             ("fp8_e4m3" if native_fp8 or gbsa_fp8 or mode == QUANT_MODE_MXFP8 else op.dtype))
    q_elem = 1 if (native_fp8 or gbsa_fp8 or gbsa_mxfp8) else BF16_BYTES
    operand_elem = 1 if (native_fp8 or gbsa_fp8 or mode == QUANT_MODE_MXFP8) else BF16_BYTES
    sched = _build_sparse_schedule(
        contract, ctx, mindie=mindie, native_fp8=native_fp8,
        gbsa_fp8=gbsa_fp8, gbsa_mxfp8=gbsa_mxfp8, mxfp4=mxfp4,
    )
    tile_specs, kv_tile, descriptor_rows = sched.tile_specs, sched.kv_tile, sched.descriptor_rows
    pairs, kv_tile_rows, kv_rows = sched.pairs, sched.kv_tile_rows, sched.kv_rows
    blocks, task_heads, work_items = sched.blocks, sched.task_heads, sched.work_items
    q_rows, pv_head_tiles, gbsa_kv_chunks, gbsa_fd = (
        sched.q_rows, sched.pv_head_tiles, sched.gbsa_kv_chunks, sched.gbsa_fd,
    )
    q_bytes = q_rows * contract.head_dim * q_elem
    metadata = {"source_program": sources[0].to_dict()}
    packed_bytes = {}
    if mxfp4:
        tile_kv_counts = tuple(tile_kv for _, _, tile_kv, _, _, _ in tile_specs)
        binding = _mxfp4_inputs(contract, q_rows, kv_tile_rows, tile_kv_counts)
        metadata.update(binding)
        packed_bytes = binding["operand_bytes"]
        q_bytes = packed_bytes["query"]
    elif native_fp8:
        sources = (*sources, MINDIE_FP8_KERNEL_SOURCE)
        metadata.update(quant_mode=mode, scale_dtype="fp32", scale_bytes=_mindie_fp8_scale_bytes(contract))
    elif gbsa_mxfp8:
        sources = (*sources, GBSA_MXFP8_KERNEL_SOURCE)
        q_scale = q_rows * ceil(contract.head_dim / MXFP8_SCALE_GROUP)
        v_scale = _gbsa_mxfp8_v_scale_bytes(contract, ctx, tile_specs, work_items, gbsa_fd)
        metadata.update(quant_mode=mode, scale_dtype="fp8_e8m0", scale_bytes={"query": q_scale, "value": v_scale})
    scales = metadata.get("scale_bytes", {})
    phases = [AttentionPhase("query_load", read_work(ctx, q_bytes), ())]
    qk_deps = ("query_load",)
    if scales.get("query", 0):
        phases.append(AttentionPhase("query_scale_load", read_work(ctx, scales["query"]), ()))
        qk_deps += ("query_scale_load",)
    v_load = None
    index_count = (kv_tile_rows if nope else blocks) * task_heads
    # Reusing cross-layer scheduling metadata skips metadata build, not per-layer sparse index reads.
    index_bytes = index_count * INDEX_BYTES
    phases.append(AttentionPhase("sparse_index", read_work(ctx, index_bytes, aiv=True), ()))
    offset_deps = ("sparse_index",)
    if contract.kv_layout in PAGED_LAYOUTS:
        phases.append(AttentionPhase("block_table", read_work(ctx, blocks * INDEX_BYTES, aiv=True), ()))
        offset_deps += ("block_table",)
    offsets = index_count
    phases.append(AttentionPhase("offset_prepare", Work(resources={"vector": offsets}), offset_deps))
    k_deps = ("offset_prepare",)
    if abi == GBSA_ABI:
        size = _gbsa_offset_descriptor_bytes(
            contract, gbsa_kv_chunks, kv_tile=kv_tile, descriptor_rows=descriptor_rows, gbsa_mxfp8=gbsa_mxfp8,
        )
        phases.append(AttentionPhase("offset_workspace_write", Work(resources={"mte3": size, "hbm": size}), k_deps))
        # Cube consumes KV-offset descriptors via scalar GM loads / D-cache preload, not AIC MTE2.
        phases.append(AttentionPhase("offset_workspace_read", Work(resources={"hbm": size}), ("offset_workspace_write",)))
        k_deps = ("offset_workspace_read",)
    kv_extents = tuple((tile_kv, base, stride) for _, _, tile_kv, _, base, stride in tile_specs)
    k_load, v_load, k_mem, v_mem = _load_sparse_kv_pair(
        contract, ctx, mode=mode, nope=nope, mxfp4=mxfp4, native_fp8=native_fp8, gbsa_fp8=gbsa_fp8,
        packed_bytes=packed_bytes, extents=kv_extents,
        core_extents=gbsa_fd.core_extents if gbsa_fd is not None else None,
        task_heads=task_heads,
    )
    phases.append(AttentionPhase("key_load", k_load, k_deps, memory=k_mem))
    qk_deps += ("key_load",)
    if scales.get("key", 0):
        phases.append(AttentionPhase("key_scale_load", read_work(ctx, scales["key"]), k_deps))
        qk_deps += ("key_scale_load",)
    metadata["l2_shared_kv_bytes"] = k_load.resources["l2"] + (v_load.resources["l2"] if v_load else 0)

    if mode in {QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8} and not native_fp8 and not gbsa_fp8:
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
    l0_scales = scales if mxfp4 else ({"query": scales["query"]} if gbsa_mxfp8 else {})
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
    out_deps = ("pv_fixpipe",)
    if v_load is not None:
        phases.append(AttentionPhase("value_load", v_load, pv_deps, memory=v_mem))
        pv_deps = ("value_load",)
        if scales.get("value", 0):
            if gbsa_mxfp8:
                phases.append(AttentionPhase("value_scale_load", read_work(ctx, scales["value"], aiv=True), ()))
                out_deps = ("pv_fixpipe", "value_scale_load")
            else:
                phases.append(AttentionPhase("value_scale_load", read_work(ctx, scales["value"]), ("probability_transfer",)))
                pv_deps += ("value_scale_load",)
        if mode in {QUANT_MODE_INT8_C8, QUANT_MODE_FP8_C8} and not native_fp8 and not gbsa_fp8:
            phases.append(AttentionPhase("value_dequant", Work(resources={
                "vector": kv_rows * contract.v_head_dim * DEQUANT_OPS,
            }), pv_deps))
            pv_deps = ("value_dequant",)
    phases.extend((
        AttentionPhase("pv", Work({dtype: 2 * pairs * contract.v_head_dim},
                                   {"mte1": probability_bytes + value_operand_bytes + l0_scales.get("value", 0)}), pv_deps),
        AttentionPhase("pv_fixpipe", Work(resources={
            "fixpipe": pv_head_tiles * contract.v_head_dim * FP32_BYTES,
        }), ("pv",)),
        AttentionPhase("online_output", Work(resources={
            "vector": (pv_head_tiles + q_rows) * contract.v_head_dim,
        }), out_deps),
    ))
    if gbsa_fd is not None:
        phases.extend(_gbsa_fd_output_phases(ctx, contract, gbsa_fd.combine_partials))
    else:
        phases.extend(output_phases(ctx, contract, BF16_BYTES, ("online_output",), q_rows))
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
        contract = contract_for_requests(contract, ctx)
        sources = SOURCES_BY_ABI[contract.abi]
        if not contract.query_tokens:
            return make_plan(op, contract, ctx, (), sources)
        ctx.datapath.route("l0c", "ub")
        ctx.datapath.route("ub", "l1")
        if contract.abi == FIA_ABI:
            return dense_fia(op, contract, ctx, sources, gm_workspace=False)
        if contract.abi in {C8_ABI, C8_V1_ABI, C8_VLLM_ABI}:
            _validate_c8_abi(contract)
        if contract.abi == SFA_V1_ABI:
            if contract.splits > 1:
                raise unmodeled(
                    "attention_arch35_sfa_split_kv_unsupported",
                    "aclnnSparseFlashAttention split-KV is supported on Arch22 only, not Arch35",
                )
            return build_arch35_split_g_sfa(op, contract, ctx, sources)
        if (contract.abi in {C8_V1_ABI, C8_ABI} and contract.splits == 1
                and contract.num_heads > SFA_SPLIT_G_HEADS
                and not contract.return_lse):
            return build_arch35_split_g_sfa(op, contract, ctx, sources)
        return _packed_kv(op, contract, ctx, sources) if contract.abi in {C8_ABI, C8_V1_ABI, C8_VLLM_ABI} else _sparse(op, contract, ctx, sources)
STRATEGY = _Strategy()
