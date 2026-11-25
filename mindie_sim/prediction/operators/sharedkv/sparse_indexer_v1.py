"""Native CANN v1 BF16 Lightning Indexer (Arch22 & Arch35) lowering.

Upstream-ABI: aclnnLightningIndexer

[Role] Lower aclnnLightningIndexer v1 using CANN's native SplitCore ownership,
  Arch22 L2-resident alternating GM score slots with dynamic s1BaseSize K-reuse,
  and Arch35 isLDOpen=false per-query S2 scan with trunk radix TopK.
[Invariants] OperatorPlan.work equals the exact sum of lowered stage work.
  Alternating GM score slots omit HBM charges only when aggregate workspace
  fits die L2 capacity; cold key/query reads and output stores stay on HBM.
  Zero hardcoded timing constants; sync latencies come from P12/P13/P15 L0 cells.
[Timeouts & Locks] Uses O(1) compact_periodic_pipeline for long S2 sweeps up
  to 400K+ tokens; zero locks or external I/O.
[Side-Effect Policy] Pure immutable plans and virtual Engine programs.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...ownership import read_role
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...results import FusedPipelineSpec, PipelinePhase, WorkItem
from ..contracts import OperatorPlanningContext
from ..matrix.refill_steady import compact_periodic_pipeline
from ..sparse_indexer._lowering import align, ceil_div
from ..sparse_indexer.common import (
    ARCH22_FLUSH_BLOCKS,
    ARCH22_SMALL_QUERY_MAX,
    ARCH22_SPLIT_RECORD_BYTES,
    ARCH22_TOPK_LIMIT,
    BF16_BYTES,
    DEFAULT_S2_TILE,
    FAMILY,
    FP32_BYTES,
    INT32_BYTES,
    MERGE_FAN_IN,
    PAIR_BYTES,
    SCORE_SLOTS,
    SparseIndexerContract,
    request_contracts,
    unsupported,
)
from .l0_sync import p03_fixpipe_setup_ms, p10_sort_to_alu_op_ratio, p12_flag_ms, p13_paired_ms, p13_syncall_ms

TARGET_BACKEND: Literal["ascend", "hardware_neutral"] = "ascend"
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/lightning_indexer/op_kernel/arch35/lightning_indexer_service_cube_arch35.h",
        "66b9f91fc76c84b048ddcbec39b6c03217786fcf7fd2fdd1c22e19e324c0ec77",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/lightning_indexer/op_kernel/arch22/lightning_indexer_service_cube_arch22.h",
        "edc95c482dbe7147cc00c1c432ca25c469594ac17f26daf8fd10154abf3f6699",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/lightning_indexer/op_kernel/arch22/lightning_indexer_kernel_arch22.h",
        "2f1fb5b9c245e5387d6866f77ee423d4e9cb4ad1869fa0a740a6d243b890728a",
    ),
    SourceRef(
        "cann/ops-transformer",
        "2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        "attention/lightning_indexer/op_kernel/arch22/lightning_indexer_service_vector_arch22.h",
        "33bb65e7a0731dcddadd9ef5d75e7d73dcdfcf67cd28adb6f943310f73249e87",
    ),
)

ARCH35_V1_AIVS_PER_AIC = 2
V1_AIVS_PER_AIC = 2
ARCH35_V1_TILE_ROWS = 128
ARCH35_V1_S1_BASE = 4
ARCH35_V1_S1_BASE_SMALL = 2
ARCH35_V1_PIPE_SLOTS = 3
ARCH35_V1_CUBE_M_BLOCK = 16
ARCH35_V1_FIXPIPE_N_ALIGN = 8
ARCH35_V1_GM_ALIGN_BYTES = 512
ARCH22_V1_S1_BASE = 8
ARCH22_SPARSE_COUNT_8K = 8192
ARCH22_L0_M_BLOCK = 128  # lightning_indexer_service_cube_arch22.h:58 M_BASIC_BLOCK_L0
ARCH22_GM_ALIGN_BYTES = 512  # lightning_indexer_kernel_arch22.h:89 GM_ALIGN_BYTES
ARCH22_PAIR_ELEMS = 2  # lightning_indexer_vector.h:284 VALUE_AND_INDEX_NUM
ARCH35_V1_TRUNK_LEN = 8192
ARCH35_V1_TRUNK_LEN_4K = 4096
ARCH35_V1_TRUNK_LEN_2K = 2048
ARCH35_V1_TOPK_5K = 5120
ARCH35_V1_TOPK_7K = 7168
ARCH35_V1_RADIX_ALIGN = 256
# vf_topk_gather.h:649-656: four 8-bit 256-bin histogram passes on 32-bit sortable scores.
ARCH35_V1_RADIX_PASSES = 4
COMPACT_TILE_THRESHOLD = 48
ARCH35_VEC_OPS_PER_HEAD = 4  # Relu + Cast + MulWeight + ReduceSum (service_vector_arch35.h:312)
# lightning_indexer_service_cube_arch22.h:388 sets Fixpipe reluPre=1;
# lightning_indexer_vector.h:98-111 DoScale + DoReduce runs Mul + Add = 2 * heads - 1 ops/row.
ARCH22_VEC_OPS_PER_HEAD = 2
RADIX_OPS_PER_ITEM = 5  # Histogram + prefix-sum + scatter + trunk merge
FLOPS_PER_FMA = 2
TOPK_MERGE_STRIDE_MULT = 2
ARCH22_K_PANEL_ROWS = 256  # lightning_indexer_service_cube_arch22.h:43
ARCH22_KEY_L1_SLOTS = 3  # lightning_indexer_service_cube_arch22.h:43 KEY_BUF_NUM = 3
ARCH22_FIXPIPE_PANEL_ROWS = 128  # lightning_indexer_service_cube_arch22.h:126 S2_BASIC_BLOCK_L0
# lightning_indexer_service_vector_arch22.h:380 Sort32 + 2 MrgSort4 passes on 512 items = 3 elem-passes.
ARCH22_SORT512_PASSES = 3
ARCH22_INIT_OPS_PER_SLOT = 4  # index seed + padded topk buffer init (service_vector_arch22.h:154)
ARCH22_ENTRY_EXIT_CREDITS = 2  # Cube/Vector entry+exit credits (kernel_arch22.h:665)


def _l2_capacity(ctx: OperatorPlanningContext) -> int:
    hierarchy = getattr(ctx.geometry.physical_mechanisms, "memory_hierarchy", None)
    return int(getattr(hierarchy, "effective_l2_bytes", ctx.geometry.buffers.get("l2", 0)) or 0)


def _has_l2_rate(ctx: OperatorPlanningContext) -> bool:
    return "l2" in ctx.geometry.rates


def _mem_work(nbytes: int, *, l2_resident: bool, has_l2_rate: bool) -> dict[str, int]:
    if nbytes <= 0:
        return {}
    if l2_resident:
        return {"l2": nbytes} if has_l2_rate else {}
    return {"hbm": nbytes}


def _place_aic(ctx: RecipeContext, aic: int, aic_count: int, work: Work) -> dict[str, tuple[int, ...]]:
    roles = {r for r in work.resources if r not in ("hbm", "l2")}
    if work.matrix:
        roles.add("cube")
    return ctx.owned(aic, aic_count, roles)


def _place_aivs(ctx: RecipeContext, aivs: tuple[int, ...], aiv_count: int, work: Work) -> dict[str, tuple[int, ...]]:
    roles = tuple(r for r in work.resources if r not in ("hbm", "l2"))
    placement: dict[str, tuple[int, ...]] = {}
    for aiv in aivs:
        for name, units in ctx.owned(aiv, aiv_count, roles).items():
            placement[name] = tuple(dict.fromkeys((*placement.get(name, ()), *units)))
    return placement


def _acc(bucket: dict[str, Work], name: str, work: Work, repeats: int = 1) -> None:
    if not (work.matrix or work.resources) or repeats <= 0:
        return
    scaled = work.scaled(repeats) if repeats != 1 else work
    bucket[name] = bucket.get(name, Work()) + scaled


@dataclass(frozen=True, slots=True)
class _QueryGroup:
    q_rows: int
    loop_s2_rows: int
    row_valid_s2: tuple[int, ...]
    req_q: int


def _arch35_s1_base(contract: SparseIndexerContract) -> int:
    # lightning_indexer_kernel_arch35.h:177-183: s1BaseSize = 2 when sparseCount > 2048 else 4.
    return ARCH35_V1_S1_BASE_SMALL if contract.topk_items > ARCH22_TOPK_LIMIT else ARCH35_V1_S1_BASE


def _arch22_s1_base(contract: SparseIndexerContract) -> int:
    # lightning_indexer_kernel_arch22.h:183-186: 8192 / sparseCount * 2 when > 2048 else 8.
    if contract.topk_items > ARCH22_TOPK_LIMIT:
        return max(1, (ARCH22_SPARSE_COUNT_8K // contract.topk_items) * 2)
    return ARCH22_V1_S1_BASE


def _arch35_trunk_len(topk_items: int) -> int:
    # lightning_indexer_service_vector_arch35.h:200-202
    if topk_items > ARCH35_V1_TOPK_7K:
        return ARCH35_V1_TRUNK_LEN_2K
    if topk_items > ARCH35_V1_TOPK_5K:
        return ARCH35_V1_TRUNK_LEN_4K
    return ARCH35_V1_TRUNK_LEN


def _invalid_fill_program(
    ctx: RecipeContext,
    owners: tuple[tuple[int, tuple[int, ...]], ...],
    aiv_count: int,
    total_elems: int,
    gm_align_bytes: int,
) -> ParallelProgram:
    """Stripe ProcessInvalid -1 fill across all paired AIVs in GM_ALIGN_BYTES chunks."""
    all_aivs = tuple(aiv for _, aivs in owners for aiv in aivs)
    if total_elems <= 0 or not all_aivs:
        return ParallelProgram(())
    align_elems = max(1, gm_align_bytes // INT32_BYTES)
    per_aiv = align(ceil_div(total_elems, len(all_aivs)), align_elems)
    branches: list[KernelProgram] = []
    for idx, aiv in enumerate(all_aivs):
        base = idx * per_aiv
        if base >= total_elems:
            break
        deal = min(per_aiv, total_elems - base)
        init_w = Work(resources={"vector": deal})
        out_b = deal * INT32_BYTES
        store_w = Work(resources={"mte3": out_b, "hbm": out_b})
        aiv_tuple = (aiv,)
        branches.append(KernelProgram((
            Stage("index_init", ctx.bind(init_w), placement=_place_aivs(ctx, aiv_tuple, aiv_count, init_w)),
            Stage(
                "index_store", ctx.bind(store_w), after=("index_init",),
                placement=_place_aivs(ctx, aiv_tuple, aiv_count, store_w),
            ),
        )))
    return ParallelProgram(tuple(branches))


def _request_q_kv(
    op: OpSpec, contract: SparseIndexerContract, ctx: OperatorPlanningContext | None = None,
) -> tuple[tuple[int, int, bool], ...]:
    if ctx is not None and ctx.requests:
        out_req = tuple(
            (rc.query_tokens, max(0, int(rc.scanned_rows)), rc.causal)
            for rc in request_contracts(contract, ctx)
            if rc.query_tokens > 0
        )
        if out_req:
            return out_req
    q_base, q_rem = divmod(contract.query_tokens, contract.batch_size)
    raw_kv = dict(op.metadata).get("kv_lens")
    if isinstance(raw_kv, (list, tuple)) and len(raw_kv) == contract.batch_size:
        return tuple(
            (q_len, max(0, int(kv)), contract.causal)
            for req_i, kv in enumerate(raw_kv)
            if (q_len := q_base + (1 if req_i < q_rem else 0)) > 0
        )
    return tuple(
        (q_len, max(0, int(contract.scanned_rows)), contract.causal)
        for req_i in range(contract.batch_size)
        if (q_len := q_base + (1 if req_i < q_rem else 0)) > 0
    )


def _query_groups(
    op: OpSpec,
    contract: SparseIndexerContract,
    s1_base: int,
    s2_tile: int,
    ctx: OperatorPlanningContext | None = None,
) -> tuple[_QueryGroup, ...]:
    groups: list[_QueryGroup] = []
    for req_q, req_kv, req_causal in _request_q_kv(op, contract, ctx):
        for s1_offset in range(0, req_q, s1_base):
            q_rows = min(s1_base, req_q - s1_offset)
            if req_kv <= 0:
                groups.append(_QueryGroup(q_rows, 0, (0,) * q_rows, req_q))
            elif not req_causal:
                groups.append(_QueryGroup(q_rows, req_kv, (req_kv,) * q_rows, req_q))
            else:
                # lightning_indexer_v2_kernel_base_arch35.h:39-55 & kernel_arch22.h:244-256
                group_valid_s2 = min(req_kv, max(1, (req_kv - req_q) + s1_offset + s1_base))
                s2_blocks = ceil_div(group_valid_s2, s2_tile)
                loop_s2_rows = min(req_kv, s2_blocks * s2_tile)
                row_valid = tuple(
                    max(0, min(req_kv, (req_kv - req_q) + s1_offset + i + 1))
                    for i in range(q_rows)
                )
                groups.append(_QueryGroup(q_rows, loop_s2_rows, row_valid, req_q))
    return tuple(groups)


def _finalize_plan(
    ctx: OperatorPlanningContext,
    contract: SparseIndexerContract,
    totals: dict[str, Work],
    program: object,
    sources: tuple[SourceRef, ...],
    *,
    tiling: str,
    tile_size: int,
    score_path: str,
    s2_splits: int,
) -> OperatorPlan:
    total = Work()
    phases: list[PipelinePhase] = []
    for name, w in totals.items():
        total += w
        items = tuple(WorkItem("cube", "flop", n, dt) for dt, n in w.matrix.items() if n)
        items += tuple(WorkItem(r, "op" if r == "vector" else "bytes", n) for r, n in w.resources.items() if n)
        phases.append(PipelinePhase(name, (phases[-1].phase_id,) if phases else (), items))
    return OperatorPlan(
        work=total,
        program=program,
        metadata={
            "implementation": contract.abi,
            "abi": contract.abi,
            "arch_id": ctx.compute_arch.arch_id,
            "tiling": tiling,
            "s2_tile_size": tile_size,
            "score_path": score_path,
            "fused_topk": contract.fused_topk,
            "cache_row_bytes": contract.cache_row_bytes,
            "s2_splits": s2_splits,
            "causal": contract.causal,
            "useful_flops": contract.matrix_flops,
        },
        pipeline_spec=FusedPipelineSpec(FAMILY, contract.abi, tuple(phases), sources),
    )


@dataclass(frozen=True, slots=True)
class _Arch35V1Program:
    contract: SparseIndexerContract
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    q_groups: tuple[_QueryGroup, ...]
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int
    score_l2: bool
    weights_l2: bool
    has_l2_rate: bool

    def _tile_works(self, valid: int, q_rows: int, *, cold_weights: bool = False) -> dict[str, Work]:
        c = self.contract
        key_b = valid * c.cache_row_bytes
        heads = q_rows * c.index_heads
        # service_cube_arch35.h:182, 198-210, 363, 378: L1 pads M to 2*gSize; L0A/L0B/Mmad align to 16.
        aligned_m = align(q_rows, 2) * c.index_heads
        issued_m = align(aligned_m, ARCH35_V1_CUBE_M_BLOCK)
        key_l0b_b = align(valid, ARCH35_V1_CUBE_M_BLOCK) * c.cache_row_bytes
        # service_cube_arch35.h:401-403: Fixpipe dualDstCtl writes align(m, 2) x align(n, 8) FP32 scores.
        score_b = align(aligned_m, 2) * align(valid, ARCH35_V1_FIXPIPE_N_ALIGN) * FP32_BYTES
        # lightning_indexer_vector1.h:174-232: MulWeightAndReduceSum executes both 64-lane halves (128 elems).
        vec_ops = ARCH35_V1_TILE_ROWS * (ARCH35_VEC_OPS_PER_HEAD * heads - q_rows)
        # service_vector_arch35.h:324-329: scoreGm store writes full s2BaseSize (128) FP32/row per tile.
        row_b = q_rows * ARCH35_V1_TILE_ROWS * FP32_BYTES
        works: dict[str, Work] = {
            "key_load": Work(resources={self.aic_read: key_b, "hbm": key_b}),
            "cube_mma": Work(
                matrix={c.matrix_dtype: FLOPS_PER_FMA * valid * issued_m * c.index_dim},
                resources={"mte1": key_l0b_b + issued_m * c.cache_row_bytes},
            ),
            "score_handoff": Work(resources={"fixpipe": score_b}),
        }
        if c.has_head_weights:
            # service_vector_arch35.h:299-305: ProcessVec1 reloads BF16 head weights on every S2 tile.
            weight_b = heads * BF16_BYTES
            works["head_weight_load"] = Work(resources={
                self.aiv_read: weight_b,
                **_mem_work(
                    weight_b,
                    l2_resident=(not cold_weights) and self.weights_l2,
                    has_l2_rate=self.has_l2_rate,
                ),
            })
        works["vec_reduce"] = Work(resources={"vector": vec_ops})
        works["score_row_store"] = Work(resources={
            "mte3": row_b, **_mem_work(row_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
        })
        return works

    def _topk_works(self, row_valid_s2: tuple[int, ...]) -> dict[str, Work]:
        c = self.contract
        if not c.fused_topk or not row_valid_s2:
            return {}
        trunk_len = _arch35_trunk_len(c.topk_items)
        hist = align(c.topk_items, ARCH35_V1_RADIX_ALIGN)
        read_b = 0
        radix_ops = 0
        init_ops = 0
        # service_vector_arch35.h:373-540 & vf_topk_gather.h:649-665:
        # validS2Len <= 0: Duplicate(-1, topk) + store; 0 < validS2Len < topk: CreateVecIndex + tail Duplicate(-1);
        # validS2Len >= topk: 4-pass 256-bin radix TopK; when topk > trunk_len & items > hist, loop 0 seeds only.
        for items in row_valid_s2:
            if items >= c.topk_items:
                read_b += items * FP32_BYTES
                if c.topk_items > trunk_len and items > hist:
                    rem_items = items - hist
                    merges = ceil_div(rem_items, trunk_len)
                    cands = merges * hist + align(rem_items, ARCH35_V1_RADIX_ALIGN)
                    radix_ops += (
                        2 * hist
                        + ARCH35_V1_RADIX_PASSES * RADIX_OPS_PER_ITEM * (cands + merges * ARCH35_V1_RADIX_ALIGN)
                    )
                else:
                    trunks = 1 if items <= hist else ceil_div(items, trunk_len)
                    cands = align(items, ARCH35_V1_RADIX_ALIGN) + max(0, trunks - 1) * hist
                    radix_ops += (
                        ARCH35_V1_RADIX_PASSES * RADIX_OPS_PER_ITEM * (cands + trunks * ARCH35_V1_RADIX_ALIGN)
                    )
            else:
                init_ops += c.topk_items
        out_b = len(row_valid_s2) * c.topk_items * INT32_BYTES
        res: dict[str, Work] = {}
        if read_b > 0:
            res["score_row_load"] = Work(resources={
                self.aiv_read: read_b, **_mem_work(read_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            })
        if radix_ops > 0:
            res["trunk_radix_topk"] = Work(resources={"vector": radix_ops})
        if init_ops > 0:
            res["index_init"] = Work(resources={"vector": init_ops})
        res["index_store"] = Work(resources={"mte3": out_b, "hbm": out_b})
        return res

    def _entry_work(self, q_rows: int) -> Work:
        c = self.contract
        q_b = q_rows * c.index_heads * c.cache_row_bytes
        return Work(resources={self.aic_read: q_b, "hbm": q_b})

    def _aiv_topk_stages(
        self, ctx: RecipeContext, aiv: int, row_valid_s2: tuple[int, ...], prev: str = "",
    ) -> tuple[Stage, ...]:
        tk_ws = self._topk_works(row_valid_s2)
        if not tk_ws:
            return ()
        aiv_tuple = (aiv,)
        stages: list[Stage] = []
        cur_prev = prev
        for name in ("score_row_load", "trunk_radix_topk", "index_init", "index_store"):
            if name in tk_ws:
                stages.append(Stage(
                    name, ctx.bind(tk_ws[name]), after=(cur_prev,) if cur_prev else (),
                    placement=_place_aivs(ctx, aiv_tuple, self.aiv_count, tk_ws[name]),
                ))
                cur_prev = name
        return tuple(stages)

    def _group_program(self, ctx: RecipeContext, owner: tuple[int, tuple[int, ...]], group: _QueryGroup) -> KernelProgram:
        aic, aivs = owner
        q_rows = group.q_rows
        scanned_rows = group.loop_s2_rows
        active_aivs = aivs[: min(len(aivs), max(1, q_rows))]
        n_tiles = ceil_div(scanned_rows, ARCH35_V1_TILE_ROWS)
        # service_vector_arch35.h:343-344: AIV0 owns first ceil(q_rows/2) rows, AIV1 owns remaining q_rows//2.
        mid = ceil_div(len(group.row_valid_s2), 2)
        aiv_slices = tuple(
            (aivs[idx], rows)
            for idx, rows in enumerate((group.row_valid_s2[:mid], group.row_valid_s2[mid:]))
            if rows and idx < len(aivs)
        )
        if n_tiles == 0:
            if len(aiv_slices) == 1:
                return KernelProgram(self._aiv_topk_stages(ctx, aiv_slices[0][0], aiv_slices[0][1]))
            tk_branches = tuple(
                KernelProgram(st)
                for aiv, rows in aiv_slices
                if (st := self._aiv_topk_stages(ctx, aiv, rows))
            )
            return KernelProgram((Stage("topk_exit", program=ParallelProgram(tk_branches)),)) if tk_branches else KernelProgram(())
        q_work = self._entry_work(q_rows)
        q_st = Stage("query_load", ctx.bind(q_work), placement=_place_aic(ctx, aic, self.aic_count, q_work))
        tile_sync_ms = p12_flag_ms(ctx, "flag_mte2_v_ns") + p12_flag_ms(ctx, "flag_v_mte3_ns")

        def _tile_stages(idx: int) -> tuple[Stage, ...]:
            valid = min(ARCH35_V1_TILE_ROWS, scanned_rows - idx * ARCH35_V1_TILE_ROWS)
            ws = self._tile_works(valid, q_rows, cold_weights=(idx == 0))
            tile_stages = [
                Stage("key_load", ctx.bind(ws["key_load"]),
                      placement=_place_aic(ctx, aic, self.aic_count, ws["key_load"])),
                Stage("cube_mma", ctx.bind(ws["cube_mma"]), after=("key_load",),
                      placement=_place_aic(ctx, aic, self.aic_count, ws["cube_mma"])),
                Stage("score_handoff", ctx.bind(ws["score_handoff"]), after=("cube_mma",),
                      placement=_place_aic(ctx, aic, self.aic_count, ws["score_handoff"])),
            ]
            vec_after = "score_handoff"
            if "head_weight_load" in ws:
                tile_stages.append(Stage(
                    "head_weight_load", ctx.bind(ws["head_weight_load"]), after=("score_handoff",),
                    placement=_place_aivs(ctx, active_aivs, self.aiv_count, ws["head_weight_load"]),
                ))
                vec_after = "head_weight_load"
            tile_stages.append(Stage(
                "vec_reduce", ctx.bind(ws["vec_reduce"]), after=(vec_after,),
                placement=_place_aivs(ctx, active_aivs, self.aiv_count, ws["vec_reduce"]),
                previous=("score_row_store",),
            ))
            tile_stages.append(Stage(
                "score_row_store", ctx.bind(ws["score_row_store"]), after=("vec_reduce",),
                placement=_place_aivs(ctx, active_aivs, self.aiv_count, ws["score_row_store"]),
                latency=tile_sync_ms,
            ))
            return tuple(tile_stages)

        loop_prog = compact_periodic_pipeline(
            ctx, n_tiles, ARCH35_V1_PIPE_SLOTS, _tile_stages, head=1, tail=1, threshold=COMPACT_TILE_THRESHOLD,
        )
        stages = [q_st, Stage("score_tiles", after=("query_load",), program=loop_prog)]
        if len(aiv_slices) == 1:
            stages.extend(self._aiv_topk_stages(ctx, aiv_slices[0][0], aiv_slices[0][1], prev="score_tiles"))
        elif aiv_slices:
            tk_branches = tuple(
                KernelProgram(st)
                for aiv, rows in aiv_slices
                if (st := self._aiv_topk_stages(ctx, aiv, rows))
            )
            if tk_branches:
                stages.append(Stage("topk_exit", after=("score_tiles",), program=ParallelProgram(tk_branches)))
        return KernelProgram(tuple(stages))

    def __call__(self, ctx: RecipeContext) -> ParallelProgram:
        if self.q_groups and all(g.loop_s2_rows == 0 for g in self.q_groups):
            # lightning_indexer_kernel_arch35.h:473-504: usedCoreNum==0 runs ProcessInvalid across all AIVs.
            if not self.contract.fused_topk:
                return ParallelProgram(())
            total_elems = sum(g.q_rows for g in self.q_groups) * self.contract.topk_items
            return _invalid_fill_program(ctx, self.owners, self.aiv_count, total_elems, ARCH35_V1_GM_ALIGN_BYTES)
        active_groups = tuple(
            g for g in self.q_groups if g.loop_s2_rows > 0 or self._topk_works(g.row_valid_s2)
        )
        n_lanes = min(len(self.owners), len(active_groups))
        if n_lanes == 0:
            return ParallelProgram(())
        # lightning_indexer_kernel_arch35.h:259-341: SplitCore assigns contiguous group intervals per core.
        base_cnt, rem_cnt = divmod(len(active_groups), n_lanes)
        lanes = []
        offset = 0
        for lane in range(n_lanes):
            cnt = base_cnt + (1 if lane < rem_cnt else 0)
            slice_groups = active_groups[offset : offset + cnt]
            offset += cnt
            own = self.owners[lane]
            lanes.append(PipelineProgram(
                len(slice_groups), 1,
                lambda i, own=own, grp=slice_groups: (
                    Stage("q_group", program=self._group_program(ctx, own, grp[i])),
                ),
            ))
        return ParallelProgram(tuple(lanes))


def build_arch35_v1_lightning(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    owners: tuple[tuple[int, tuple[int, ...]], ...], sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build native Arch35 aclnnLightningIndexer v1 (isLDOpen=false, trunk radix TopK)."""
    s1_base = _arch35_s1_base(contract)
    q_groups = _query_groups(op, contract, s1_base, ARCH35_V1_TILE_ROWS, ctx)
    paired_owners = tuple((aic, aivs) for aic, aivs in owners if len(aivs) >= ARCH35_V1_AIVS_PER_AIC)
    if q_groups and not paired_owners:
        raise unsupported(
            "missing_owners",
            "Arch35 aclnnLightningIndexer requires at least one complete (1 AIC, 2 AIV) owner group",
        )
    active = min(len(paired_owners), max(1, len(q_groups))) if q_groups else 0
    score_ws = active * max(
        (g.q_rows * align(g.loop_s2_rows, ARCH35_V1_TILE_ROWS) for g in q_groups), default=0,
    ) * FP32_BYTES
    weight_ws = active * max((g.q_rows for g in q_groups), default=0) * contract.index_heads * BF16_BYTES
    l2_cap = _l2_capacity(ctx)
    score_l2 = l2_cap > 0 and score_ws <= l2_cap
    weights_l2 = l2_cap > 0 and weight_ws <= l2_cap
    has_l2 = _has_l2_rate(ctx)
    aic_r = read_role(ctx.geometry.resource_units)
    aiv_r = read_role(ctx.geometry.resource_units, aiv=0)
    prog = _Arch35V1Program(
        contract, paired_owners, q_groups, aic_r, aiv_r,
        ctx.datapath.aic_count, ctx.datapath.aiv_count, score_l2, weights_l2, has_l2,
    )
    totals: dict[str, Work] = {}
    for g in q_groups:
        full, rem = divmod(g.loop_s2_rows, ARCH35_V1_TILE_ROWS)
        if full or rem:
            _acc(totals, "query_load", prog._entry_work(g.q_rows))
            if full:
                for k, w in prog._tile_works(ARCH35_V1_TILE_ROWS, g.q_rows, cold_weights=True).items():
                    _acc(totals, k, w, 1)
                if full > 1:
                    for k, w in prog._tile_works(ARCH35_V1_TILE_ROWS, g.q_rows, cold_weights=False).items():
                        _acc(totals, k, w, full - 1)
                if rem:
                    for k, w in prog._tile_works(rem, g.q_rows, cold_weights=False).items():
                        _acc(totals, k, w, 1)
            else:
                for k, w in prog._tile_works(rem, g.q_rows, cold_weights=True).items():
                    _acc(totals, k, w, 1)
        for k, w in prog._topk_works(g.row_valid_s2).items():
            _acc(totals, k, w, 1)
    return _finalize_plan(
        ctx, contract, totals, prog, sources,
        tiling="arch35_v1_is_ld_false", tile_size=ARCH35_V1_TILE_ROWS,
        score_path="l0c->ub", s2_splits=1,
    )


@dataclass(frozen=True, slots=True)
class _Arch22V1Program:
    contract: SparseIndexerContract
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    core_runs: tuple[tuple[tuple[int, int, int, bool], ...], ...]
    core_q_rows: tuple[int, ...]
    core_split_q_rows: tuple[int, ...]
    group_row_owner_counts: tuple[tuple[int, ...], ...]
    s2_owners_per_group: int
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int
    score_l2: bool
    has_l2_rate: bool

    def _tile_works(
        self, valid: int, n_owner_tiles: int, q_rows: int, *,
        use_merge_sort: bool = False, sort_ratio: float = 1.0,
    ) -> dict[str, Work]:
        c = self.contract
        qr = max(1, int(q_rows))
        key_b = valid * c.cache_row_bytes
        heads = qr * c.index_heads
        score_b = valid * heads * FP32_BYTES
        w_b = heads * BF16_BYTES if c.has_head_weights else 0
        # service_cube_arch22.h:178-183: each 128-row M L0 panel reloads K into L0B.
        m_l0_panels = ceil_div(heads, ARCH22_L0_M_BLOCK)
        if use_merge_sort:
            # service_vector_arch22.h:370-377 & vector.h:270-313: SortAll(512) + MergeSort(virTopK, 512).
            vir_topk = self._owner_retained_items()
            sort_passes = ARCH22_SORT512_PASSES * DEFAULT_S2_TILE + (vir_topk + DEFAULT_S2_TILE)
            copy_ops = vir_topk * ARCH22_PAIR_ELEMS
        else:
            # service_vector_arch22.h:380-404: 512-item Sort32+MrgSort4 (3 passes)
            # + 4-block MrgBasicBlock (2048/4) + retained SparseTopK (4096/4) on P10_sort_topk;
            # line 396 DataCopy copyback (2048/4) stays on UB copy (1x), not Sort.
            flush_sort_passes = DEFAULT_S2_TILE + (
                (ARCH22_TOPK_LIMIT * TOPK_MERGE_STRIDE_MULT // ARCH22_FLUSH_BLOCKS)
                if n_owner_tiles > ARCH22_FLUSH_BLOCKS else 0
            )
            copy_ops = DEFAULT_S2_TILE if n_owner_tiles > ARCH22_FLUSH_BLOCKS else 0
            sort_passes = ARCH22_SORT512_PASSES * DEFAULT_S2_TILE + flush_sort_passes
        alu_ops = valid * (ARCH22_VEC_OPS_PER_HEAD * heads - qr) + qr * copy_ops
        vec_ops = alu_ops + int(round(qr * sort_passes * sort_ratio))
        return {
            "key_load": Work(resources={self.aic_read: key_b, "hbm": key_b}),
            "cube_mma": Work(
                matrix={c.matrix_dtype: FLOPS_PER_FMA * valid * heads * c.index_dim},
                resources={"mte1": m_l0_panels * key_b + heads * c.cache_row_bytes},
            ),
            "score_handoff": Work(resources={
                "fixpipe": score_b, **_mem_work(score_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            }),
            "score_read": Work(resources={
                self.aiv_read: score_b + w_b,
                **_mem_work(score_b + w_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            }),
            "vec_sort": Work(resources={"vector": vec_ops}),
        }

    def _owner_retained_items(self) -> int:
        # service_vector_arch22.h:137, 484: virTopK = sparseCount if > 2048 else 2048.
        return max(self.contract.topk_items, ARCH22_TOPK_LIMIT)

    def _split_store_work(self, split_q_rows: int) -> Work:
        if split_q_rows <= 0 or not self.contract.fused_topk:
            return Work()
        size = split_q_rows * (self._owner_retained_items() * PAIR_BYTES + ARCH22_SPLIT_RECORD_BYTES)
        init_ops = split_q_rows * self._owner_retained_items() * ARCH22_INIT_OPS_PER_SLOT
        return Work(resources={
            "vector": init_ops,
            "mte3": size, **_mem_work(size, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
        })

    def _query_merge_works(self, n_split_owners: int | None = None, *, sort_ratio: float = 1.0) -> dict[str, Work]:
        c = self.contract
        if not c.fused_topk:
            return {}
        owners_cnt = self.s2_owners_per_group if n_split_owners is None else n_split_owners
        out_b = c.topk_items * INT32_BYTES
        res: dict[str, Work] = {"index_store": Work(resources={"mte3": out_b, "hbm": out_b})}
        if owners_cnt > 1:
            ret = self._owner_retained_items()
            size = owners_cnt * (ret * PAIR_BYTES + ARCH22_SPLIT_RECORD_BYTES)
            merges = owners_cnt - 1
            groups, tail = divmod(merges, MERGE_FAN_IN - 1)
            entries = groups * MERGE_FAN_IN + (tail + 1 if tail else 0)
            passes = groups + int(bool(tail))
            res["split_list_load"] = Work(resources={
                self.aiv_read: size, **_mem_work(size, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            })
            # service_vector_arch22.h:611-630: MrgSort4 runs on P10_sort_topk; DataCopy copyback is 1x.
            merge_ops = int(round(entries * ret * sort_ratio)) + passes * ret
            res["split_topk_merge"] = Work(resources={"vector": merge_ops})
        return res

    def _tile_stages(
        self,
        ctx: RecipeContext,
        aic: int,
        raw_aivs: tuple[int, ...],
        valid: int,
        qr: int,
        n_blks: int,
        *,
        use_merge_sort: bool,
        sort_ratio: float,
        fix_setup_ms: float,
        tile_sync_ms: float,
    ) -> tuple[Stage, ...]:
        aivs = raw_aivs[: min(len(raw_aivs), max(1, qr))]
        ws = self._tile_works(valid, n_blks, qr, use_merge_sort=use_merge_sort, sort_ratio=1.0)
        vec_caps: dict[str, float] = {}
        if sort_ratio != 1.0:
            vec_res = ctx.resource_names.get("vector")
            b_vec = ws["vec_sort"].resources.get("vector", 0)
            eff_vec = self._tile_works(
                valid, n_blks, qr, use_merge_sort=use_merge_sort, sort_ratio=sort_ratio,
            )["vec_sort"].resources.get("vector", 0)
            if vec_res and b_vec > 0 and eff_vec > 0:
                aiv_rate = ctx.rates[vec_res] * len(aivs) / max(1, self.aiv_count)
                vec_caps[vec_res] = aiv_rate * (b_vec / eff_vec)
        fix_lat = ceil_div(valid, ARCH22_FIXPIPE_PANEL_ROWS) * fix_setup_ms
        # service_cube_arch22.h:43: KEY_BUF_NUM=3 triple-buffered 256-row K panels;
        # kernel_arch22.h:665-671: per-tile CrossCoreWaitFlag/SetFlag<MODE2> (P13);
        # service_vector_arch22.h:325: ping-pong score reads overlap vec_sort across tiles.
        sync_stages = (
            (Stage("tile_sync", after=("score_handoff",), previous=("vec_sort",),
                   latency=tile_sync_ms, keep_empty=True),)
            if tile_sync_ms > 0.0 else ()
        )
        vec_after = ("score_read", "tile_sync") if sync_stages else ("score_read",)
        if valid > ARCH22_K_PANEL_ROWS:
            frac0 = ARCH22_K_PANEL_ROWS / float(valid)
            frac1 = 1.0 - frac0
            k0, k1 = ws["key_load"].scaled(frac0), ws["key_load"].scaled(frac1)
            m0, m1 = ws["cube_mma"].scaled(frac0), ws["cube_mma"].scaled(frac1)
            s0 = Stage("key_load_0", ctx.bind(k0), placement=_place_aic(ctx, aic, self.aic_count, k0))
            s1 = Stage("cube_mma_0", ctx.bind(m0), after=("key_load_0",), placement=_place_aic(ctx, aic, self.aic_count, m0))
            s2 = Stage("key_load_1", ctx.bind(k1), after=("key_load_0",), placement=_place_aic(ctx, aic, self.aic_count, k1))
            s3 = Stage("cube_mma_1", ctx.bind(m1), after=("cube_mma_0", "key_load_1"), placement=_place_aic(ctx, aic, self.aic_count, m1))
            s4 = Stage("score_handoff", ctx.bind(ws["score_handoff"]), after=("cube_mma_1",),
                       placement=_place_aic(ctx, aic, self.aic_count, ws["score_handoff"]), latency=fix_lat)
            s5 = Stage("score_read", ctx.bind(ws["score_read"]), after=("score_handoff",),
                       placement=_place_aivs(ctx, aivs, self.aiv_count, ws["score_read"]))
            s6 = Stage("vec_sort", ctx.bind(ws["vec_sort"]), after=vec_after,
                       placement=_place_aivs(ctx, aivs, self.aiv_count, ws["vec_sort"]),
                       caps=vec_caps, previous=("vec_sort",))
            return (s0, s1, s2, s3, s4, s5, *sync_stages, s6)
        s0 = Stage("key_load", ctx.bind(ws["key_load"]),
                   placement=_place_aic(ctx, aic, self.aic_count, ws["key_load"]))
        s1 = Stage("cube_mma", ctx.bind(ws["cube_mma"]), after=("key_load",),
                   placement=_place_aic(ctx, aic, self.aic_count, ws["cube_mma"]))
        s2 = Stage("score_handoff", ctx.bind(ws["score_handoff"]), after=("cube_mma",),
                   placement=_place_aic(ctx, aic, self.aic_count, ws["score_handoff"]), latency=fix_lat)
        s3 = Stage("score_read", ctx.bind(ws["score_read"]), after=("score_handoff",),
                   placement=_place_aivs(ctx, aivs, self.aiv_count, ws["score_read"]))
        s4 = Stage("vec_sort", ctx.bind(ws["vec_sort"]), after=vec_after,
                   placement=_place_aivs(ctx, aivs, self.aiv_count, ws["vec_sort"]),
                   caps=vec_caps, previous=("vec_sort",))
        return (s0, s1, s2, s3, *sync_stages, s4)

    def _query_exit_branches(self, ctx: RecipeContext, sort_ratio: float) -> tuple[KernelProgram, ...]:
        g_rows = self.group_row_owner_counts
        n_m_lanes = min(len(self.owners), len(g_rows))
        m_branches: list[KernelProgram] = []
        for lane in range(n_m_lanes):
            _, raw_aivs = self.owners[lane]
            merge_aivs = raw_aivs[:1]
            m_st: list[Stage] = []
            for gid in range(lane, len(g_rows), n_m_lanes):
                for owners_cnt, q_cnt in sorted(Counter(g_rows[gid]).items()):
                    if q_cnt <= 0:
                        continue
                    m_ws = self._query_merge_works(owners_cnt, sort_ratio=1.0)
                    m_ws_eff = (
                        self._query_merge_works(owners_cnt, sort_ratio=sort_ratio)
                        if sort_ratio != 1.0 else m_ws
                    )
                    for name in ("split_list_load", "split_topk_merge", "index_store"):
                        if name not in m_ws:
                            continue
                        w = m_ws[name].scaled(q_cnt)
                        m_caps: dict[str, float] = {}
                        if name == "split_topk_merge" and sort_ratio != 1.0:
                            vec_res = ctx.resource_names.get("vector")
                            b_vec = w.resources.get("vector", 0)
                            eff_vec = m_ws_eff[name].scaled(q_cnt).resources.get("vector", 0)
                            if vec_res and b_vec > 0 and eff_vec > 0:
                                aiv_rate = ctx.rates[vec_res] * len(merge_aivs) / max(1, self.aiv_count)
                                m_caps[vec_res] = aiv_rate * (b_vec / eff_vec)
                        st_name = f"{name}_{gid}_{owners_cnt}"
                        m_st.append(Stage(
                            st_name, ctx.bind(w), after=(m_st[-1].name,) if m_st else (),
                            placement=_place_aivs(ctx, merge_aivs, self.aiv_count, w),
                            caps=m_caps,
                        ))
            if m_st:
                m_branches.append(KernelProgram(tuple(m_st)))
        return tuple(m_branches)

    def __call__(self, ctx: RecipeContext) -> KernelProgram:
        c = self.contract
        if not self.core_runs:
            if not c.fused_topk:
                return KernelProgram(())
            total_elems = sum(len(rows) for rows in self.group_row_owner_counts) * c.topk_items
            if total_elems <= 0:
                return KernelProgram(())
            return KernelProgram((Stage(
                "query_exit",
                program=_invalid_fill_program(ctx, self.owners, self.aiv_count, total_elems, ARCH22_GM_ALIGN_BYTES),
            ),))
        sort_ratio = p10_sort_to_alu_op_ratio(ctx)
        fix_setup_ms = p03_fixpipe_setup_ms(ctx)
        tile_sync_ms = p13_paired_ms(ctx)
        credit_ms = ARCH22_ENTRY_EXIT_CREDITS * tile_sync_ms
        branches: list[KernelProgram] = []
        for idx, (c_runs, q_rows, split_q_rows) in enumerate(zip(self.core_runs, self.core_q_rows, self.core_split_q_rows)):
            n_blks = sum(cnt for cnt, _, _, _ in c_runs)
            if n_blks <= 0:
                continue
            aic, raw_aivs = self.owners[idx]
            q_b = q_rows * c.index_heads * c.cache_row_bytes
            core_q_w = Work(resources={self.aic_read: q_b, "hbm": q_b})
            st = [Stage("query_load", ctx.bind(core_q_w), placement=_place_aic(ctx, aic, self.aic_count, core_q_w))]
            prev_local = "query_load"
            if credit_ms > 0.0:
                st.append(Stage("entry_credits", after=(prev_local,), latency=credit_ms, keep_empty=True))
                prev_local = "entry_credits"
            for run_i, (cnt, valid, qr, use_ms) in enumerate(c_runs):
                if cnt > 1:
                    loop = compact_periodic_pipeline(
                        ctx, cnt, ARCH22_KEY_L1_SLOTS,
                        lambda _ti, v=valid, q=qr, ms=use_ms, aic=aic, raw_aivs=raw_aivs, n_blks=n_blks: self._tile_stages(
                            ctx, aic, raw_aivs, v, q, n_blks,
                            use_merge_sort=ms, sort_ratio=sort_ratio, fix_setup_ms=fix_setup_ms, tile_sync_ms=tile_sync_ms,
                        ),
                        head=1, tail=1, threshold=COMPACT_TILE_THRESHOLD,
                    )
                    st_name = f"s2_tiles_{run_i}" if len(c_runs) > 1 else "s2_tiles"
                    st.append(Stage(st_name, after=(prev_local,), program=loop))
                    prev_local = st_name
                else:
                    t_name = f"s2_tile_{run_i}"
                    st.append(Stage(
                        t_name, after=(prev_local,),
                        program=KernelProgram(self._tile_stages(
                            ctx, aic, raw_aivs, valid, qr, n_blks,
                            use_merge_sort=use_ms, sort_ratio=sort_ratio, fix_setup_ms=fix_setup_ms, tile_sync_ms=tile_sync_ms,
                        )),
                    ))
                    prev_local = t_name
            store_w = self._split_store_work(split_q_rows)
            if store_w.resources:
                core_max_qr = max((qr for _, _, qr, _ in c_runs), default=1)
                store_aivs = raw_aivs[: min(len(raw_aivs), max(1, core_max_qr))]
                st.append(Stage(
                    "split_list_store", ctx.bind(store_w), after=(prev_local,),
                    placement=_place_aivs(ctx, store_aivs, self.aiv_count, store_w),
                ))
            branches.append(KernelProgram(tuple(st)))

        stages: list[Stage] = []
        prev = ""
        if branches:
            stages.append(Stage("s2_owners", program=ParallelProgram(tuple(branches))))
            prev = "s2_owners"
        barrier_ms = p13_syncall_ms(ctx) or tile_sync_ms
        if branches and self.s2_owners_per_group > 1 and barrier_ms > 0.0:
            stages.append(Stage("split_barrier", after=(prev,), latency=barrier_ms, keep_empty=True))
            prev = "split_barrier"
        if c.fused_topk:
            m_branches = self._query_exit_branches(ctx, sort_ratio)
            if m_branches:
                stages.append(Stage("query_exit", after=(prev,) if prev else (), program=ParallelProgram(m_branches)))
        return KernelProgram(tuple(stages))


def _unpack_block_run(run: tuple[int, ...]) -> tuple[int, int, int, int]:
    gid, cnt, rows, *rest = run
    return gid, cnt, rows, (rest[0] if rest else 1)


def _assign_block_runs(
    runs: Sequence[tuple[int, ...]], active: int, n_groups: int = 0,
) -> tuple[
    tuple[tuple[tuple[int, int, int], ...], ...],
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
]:
    """Distribute ``(group_id, count, valid_rows[, q_rows])`` runs across ``active`` cores."""
    unpacked = tuple(_unpack_block_run(run) for run in runs)
    total_blocks = sum(cnt for _, cnt, _, _ in unpacked)
    if active <= 0 or total_blocks <= 0:
        return (), (), (), ()
    base, rem_cores = divmod(total_blocks, active)
    core_runs: list[tuple[tuple[int, int, int], ...]] = []
    core_groups: list[tuple[int, ...]] = []
    core_q_rows: list[int] = []
    group_count = max(n_groups, max((gid for gid, _, _, _ in unpacked), default=0) + 1)
    group_owners = [0] * group_count
    group_qr = [1] * group_count
    for gid, _, _, qr in unpacked:
        group_qr[gid] = qr
    run_idx = 0
    run_rem = unpacked[0][1] if unpacked else 0
    for core_i in range(active):
        quota = base + (1 if core_i < rem_cores else 0)
        c_runs: list[tuple[int, int, int]] = []
        seen_groups: set[int] = set()
        while quota > 0 and run_idx < len(unpacked):
            gid, _, rows, qr = unpacked[run_idx]
            take = min(quota, run_rem)
            seen_groups.add(gid)
            if c_runs and c_runs[-1][1] == rows and c_runs[-1][2] == qr:
                prev_cnt, _, _ = c_runs[-1]
                c_runs[-1] = (prev_cnt + take, rows, qr)
            else:
                c_runs.append((take, rows, qr))
            quota -= take
            run_rem -= take
            if run_rem == 0:
                run_idx += 1
                run_rem = unpacked[run_idx][1] if run_idx < len(unpacked) else 0
        ordered_groups = tuple(sorted(seen_groups))
        for gid in ordered_groups:
            group_owners[gid] += 1
        core_runs.append(tuple(c_runs))
        core_groups.append(ordered_groups)
        core_q_rows.append(sum(group_qr[gid] for gid in ordered_groups))
    return tuple(core_runs), tuple(core_groups), tuple(core_q_rows), tuple(max(1, c) for c in group_owners)


def _assign_arch22_groups(
    q_groups: tuple[_QueryGroup, ...],
    active_cores: int,
    *,
    over_2k: bool,
) -> tuple[
    tuple[tuple[tuple[int, int, int, bool], ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[tuple[int, ...], ...],
]:
    """Assign Arch22 S2 tiles (or whole groups when over_2k) and derive per-row split participation."""
    n_groups = len(q_groups)
    if active_cores <= 0 or n_groups == 0:
        return (), (), (), tuple((1,) * g.q_rows for g in q_groups)
    core_runs: list[tuple[tuple[int, int, int, bool], ...]] = []
    core_q_rows: list[int] = []
    core_gid_spans: list[dict[int, tuple[int, int]]] = []
    if over_2k:
        # lightning_indexer_kernel_arch22.h:269, 324: isSparseCountOver2K keeps each group on one core.
        active_gids = [gid for gid, g in enumerate(q_groups) if g.loop_s2_rows > 0]
        n_used = min(active_cores, len(active_gids))
        base, rem = divmod(len(active_gids), n_used) if n_used > 0 else (0, 0)
        offset = 0
        for core_i in range(n_used):
            cnt = base + (1 if core_i < rem else 0)
            gids = active_gids[offset : offset + cnt]
            offset += cnt
            c_runs: list[tuple[int, int, int, bool]] = []
            spans: dict[int, tuple[int, int]] = {}
            for gid in gids:
                g = q_groups[gid]
                n_full, tail = divmod(g.loop_s2_rows, DEFAULT_S2_TILE)
                if n_full > 0:
                    c_runs.append((n_full, DEFAULT_S2_TILE, g.q_rows, True))
                if tail > 0:
                    c_runs.append((1, tail, g.q_rows, True))
                spans[gid] = (0, n_full + int(tail > 0))
            core_runs.append(tuple(c_runs))
            core_q_rows.append(sum(q_groups[gid].q_rows for gid in gids))
            core_gid_spans.append(spans)
    else:
        unpacked: list[tuple[int, int, int, int, bool]] = []
        for gid, g in enumerate(q_groups):
            use_ms = g.req_q > ARCH22_SMALL_QUERY_MAX
            n_full, tail = divmod(g.loop_s2_rows, DEFAULT_S2_TILE)
            if n_full > 0:
                unpacked.append((gid, n_full, DEFAULT_S2_TILE, g.q_rows, use_ms))
            if tail > 0:
                unpacked.append((gid, 1, tail, g.q_rows, use_ms))
        total_blocks = sum(cnt for _, cnt, _, _, _ in unpacked)
        n_used = min(active_cores, total_blocks)
        base, rem = divmod(total_blocks, n_used) if n_used > 0 else (0, 0)
        gid_cursor = [0] * n_groups
        run_idx = 0
        run_rem = unpacked[0][1] if unpacked else 0
        for core_i in range(n_used):
            quota = base + (1 if core_i < rem else 0)
            c_runs = []
            spans = {}
            while quota > 0 and run_idx < len(unpacked):
                gid, _, rows, qr, use_ms = unpacked[run_idx]
                take = min(quota, run_rem)
                blk_start = gid_cursor[gid]
                blk_end = blk_start + take
                gid_cursor[gid] = blk_end
                if gid in spans:
                    spans[gid] = (spans[gid][0], blk_end)
                else:
                    spans[gid] = (blk_start, blk_end)
                if c_runs and c_runs[-1][1] == rows and c_runs[-1][2] == qr and c_runs[-1][3] == use_ms:
                    c_runs[-1] = (c_runs[-1][0] + take, rows, qr, use_ms)
                else:
                    c_runs.append((take, rows, qr, use_ms))
                quota -= take
                run_rem -= take
                if run_rem == 0:
                    run_idx += 1
                    run_rem = unpacked[run_idx][1] if run_idx < len(unpacked) else 0
            core_runs.append(tuple(c_runs))
            core_q_rows.append(sum(q_groups[gid].q_rows for gid in spans))
            core_gid_spans.append(spans)
    # service_vector_arch22.h:413-488: per-row split participation depends on row's visible S2 tile count.
    core_split_q_rows = [0] * len(core_runs)
    group_row_owners: list[tuple[int, ...]] = []
    for gid, g in enumerate(q_groups):
        row_counts: list[int] = []
        for valid_s2 in g.row_valid_s2:
            row_tiles = ceil_div(valid_s2, DEFAULT_S2_TILE) if valid_s2 > 0 else 0
            active_owners = [
                ci for ci, spans in enumerate(core_gid_spans)
                if gid in spans and spans[gid][0] < row_tiles
            ]
            if len(active_owners) > 1:
                for ci in active_owners:
                    core_split_q_rows[ci] += 1
            row_counts.append(max(1, len(active_owners)))
        group_row_owners.append(tuple(row_counts))
    return tuple(core_runs), tuple(core_q_rows), tuple(core_split_q_rows), tuple(group_row_owners)


def build_arch22_v1_lightning(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    owners: tuple[tuple[int, tuple[int, ...]], ...], sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build native Arch22 aclnnLightningIndexer v1 (SplitCore + s1BaseSize K-reuse + L2 score slots)."""
    s1_base = _arch22_s1_base(contract)
    over_2k = contract.topk_items > ARCH22_TOPK_LIMIT
    q_groups = _query_groups(op, contract, s1_base, DEFAULT_S2_TILE, ctx)
    total_blocks = sum(ceil_div(g.loop_s2_rows, DEFAULT_S2_TILE) for g in q_groups)
    valid_owners = tuple((aic, aivs) for aic, aivs in owners if len(aivs) >= V1_AIVS_PER_AIC)
    has_work = total_blocks > 0 or (bool(q_groups) and contract.fused_topk)
    if has_work and not valid_owners:
        raise unsupported(
            "missing_owners",
            "Arch22 aclnnLightningIndexer requires at least one complete (1 AIC, 2 AIV) owner group",
        )
    active = min(len(valid_owners), total_blocks) if total_blocks > 0 else 0
    core_runs, core_q_rows, core_split_q_rows, group_row_owners = _assign_arch22_groups(
        q_groups, active, over_2k=over_2k,
    )
    s2_owners_per_group = max(
        (cnt for row_cnts in group_row_owners for cnt in row_cnts), default=1,
    )
    slot_ws = sum(
        SCORE_SLOTS * DEFAULT_S2_TILE * max((qr for _, _, qr, _ in c_runs), default=0) * contract.index_heads * FP32_BYTES
        for c_runs in core_runs
    )
    l2_cap = _l2_capacity(ctx)
    score_l2 = l2_cap > 0 and slot_ws <= l2_cap
    has_l2 = _has_l2_rate(ctx)
    aic_r = read_role(ctx.geometry.resource_units)
    aiv_r = read_role(ctx.geometry.resource_units, aiv=0)
    prog = _Arch22V1Program(
        contract, valid_owners, core_runs, core_q_rows, core_split_q_rows,
        group_row_owners, s2_owners_per_group,
        aic_r, aiv_r, ctx.datapath.aic_count, ctx.datapath.aiv_count, score_l2, has_l2,
    )
    totals: dict[str, Work] = {}
    if not core_runs:
        if contract.fused_topk and q_groups:
            total_elems = sum(g.q_rows for g in q_groups) * contract.topk_items
            out_b = total_elems * INT32_BYTES
            _acc(totals, "index_init", Work(resources={"vector": total_elems}))
            _acc(totals, "index_store", Work(resources={"mte3": out_b, "hbm": out_b}))
    else:
        for c_runs, q_rows, split_q_rows in zip(core_runs, core_q_rows, core_split_q_rows):
            n_blks = sum(cnt for cnt, _, _, _ in c_runs)
            if n_blks <= 0:
                continue
            q_b = q_rows * contract.index_heads * contract.cache_row_bytes
            _acc(totals, "query_load", Work(resources={aic_r: q_b, "hbm": q_b}))
            for cnt, valid, qr, use_ms in c_runs:
                for k, w in prog._tile_works(valid, n_blks, qr, use_merge_sort=use_ms).items():
                    _acc(totals, k, w, cnt)
            store_w = prog._split_store_work(split_q_rows)
            if store_w.resources:
                _acc(totals, "split_list_store", store_w)
        for row_cnts in group_row_owners:
            for owners_cnt, q_cnt in sorted(Counter(row_cnts).items()):
                for k, w in prog._query_merge_works(owners_cnt).items():
                    _acc(totals, k, w, q_cnt)
    return _finalize_plan(
        ctx, contract, totals, prog, sources,
        tiling="arch22_v1_split_core", tile_size=DEFAULT_S2_TILE,
        score_path="l0c->l2->ub" if score_l2 else "l0c->gm->ub",
        s2_splits=s2_owners_per_group,
    )
