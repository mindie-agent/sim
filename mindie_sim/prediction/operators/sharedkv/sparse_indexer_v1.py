"""Native CANN v1 BF16 Lightning Indexer (Arch22 & Arch35) lowering.

Upstream-ABI: aclnnLightningIndexer

[Role] Lower aclnnLightningIndexer v1 using CANN's native SplitCore ownership,
  Arch22 L2-resident alternating GM score slots (256 KiB/core) with s1BaseSize=2
  K-reuse, and Arch35 isLDOpen=false per-query S2 scan with 8192-trunk TopK.
[Invariants] OperatorPlan.work equals the exact sum of lowered stage work.
  Alternating GM score slots omit HBM charges only when aggregate workspace
  fits die L2 capacity; cold key/query reads and output stores stay on HBM.
  Zero hardcoded timing constants; sync latencies come from P12/P13/P15 L0 cells.
[Timeouts & Locks] Uses O(1) compact_periodic_pipeline for long S2 sweeps up
  to 400K+ tokens; zero locks or external I/O.
[Side-Effect Policy] Pure immutable plans and virtual Engine programs.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

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
)
from .l0_sync import p03_fixpipe_setup_ms, p10_sort_to_alu_op_ratio, p12_flag_ms, p13_paired_ms, p13_syncall_ms

ARCH35_V1_TILE_ROWS = 128
ARCH35_V1_S1_BASE = 4
ARCH35_V1_PIPE_SLOTS = 3
ARCH22_V1_S1_BASE = 2
ARCH35_V1_TRUNK_LEN = 8192
ARCH35_V1_RADIX_ALIGN = 256
ARCH35_V1_RADIX_PASSES = 2
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


def _q_groups_for_batch(contract: SparseIndexerContract, s1_base: int) -> tuple[int, ...]:
    q_base, q_rem = divmod(contract.query_tokens, contract.batch_size)
    out: list[int] = []
    for req_i in range(contract.batch_size):
        rem = q_base + (1 if req_i < q_rem else 0)
        while rem > 0:
            take = min(s1_base, rem)
            out.append(take)
            rem -= take
    return tuple(out)


def _q_groups_with_kv(
    op: OpSpec, contract: SparseIndexerContract, s1_base: int, ctx: OperatorPlanningContext | None = None,
) -> tuple[tuple[int, int], ...]:
    if ctx is not None and ctx.requests:
        out_req: list[tuple[int, int]] = []
        for rc in request_contracts(contract, ctx):
            rem = rc.query_tokens
            while rem > 0:
                take = min(s1_base, rem)
                out_req.append((take, max(1, int(rc.scanned_rows))))
                rem -= take
        if out_req:
            return tuple(out_req)
    raw_kv = dict(op.metadata).get("kv_lens")
    if isinstance(raw_kv, (list, tuple)) and len(raw_kv) == contract.batch_size:
        q_base, q_rem = divmod(contract.query_tokens, contract.batch_size)
        out: list[tuple[int, int]] = []
        for req_i, kv in enumerate(raw_kv):
            rem = q_base + (1 if req_i < q_rem else 0)
            while rem > 0:
                take = min(s1_base, rem)
                out.append((take, max(1, int(kv))))
                rem -= take
        return tuple(out)
    return tuple((q, contract.scanned_rows) for q in _q_groups_for_batch(contract, s1_base))


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
    q_groups: tuple[tuple[int, int], ...]
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int
    score_l2: bool
    has_l2_rate: bool

    def _tile_works(self, valid: int, q_rows: int) -> dict[str, Work]:
        c = self.contract
        key_b = valid * c.cache_row_bytes
        heads = q_rows * c.index_heads
        score_b = valid * heads * FP32_BYTES
        vec_ops = valid * (ARCH35_VEC_OPS_PER_HEAD * heads - q_rows)
        row_b = q_rows * valid * FP32_BYTES
        return {
            "key_load": Work(resources={self.aic_read: key_b, "hbm": key_b}),
            "cube_mma": Work(
                matrix={c.matrix_dtype: FLOPS_PER_FMA * valid * heads * c.index_dim},
                resources={"mte1": key_b + heads * c.cache_row_bytes},
            ),
            "score_handoff": Work(resources={"fixpipe": score_b}),
            "vec_reduce": Work(resources={"vector": vec_ops}),
            "score_row_store": Work(resources={
                "mte3": row_b, **_mem_work(row_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            }),
        }

    def _topk_works(self, q_rows: int, items: int) -> dict[str, Work]:
        c = self.contract
        if not c.fused_topk or items <= 0:
            return {}
        read_b = q_rows * items * FP32_BYTES
        trunks = ceil_div(items, ARCH35_V1_TRUNK_LEN)
        hist = align(min(items, c.topk_items), ARCH35_V1_RADIX_ALIGN)
        cands = align(items, ARCH35_V1_RADIX_ALIGN) + max(0, trunks - 1) * hist
        ops = q_rows * ARCH35_V1_RADIX_PASSES * RADIX_OPS_PER_ITEM * (cands + trunks * ARCH35_V1_RADIX_ALIGN)
        out_b = q_rows * c.topk_items * INT32_BYTES
        return {
            "score_row_load": Work(resources={
                self.aiv_read: read_b, **_mem_work(read_b, l2_resident=self.score_l2, has_l2_rate=self.has_l2_rate),
            }),
            "trunk_radix_topk": Work(resources={"vector": ops}),
            "index_store": Work(resources={"mte3": out_b, "hbm": out_b}),
        }

    def _entry_work(self, q_rows: int) -> Work:
        c = self.contract
        q_b = q_rows * c.index_heads * c.cache_row_bytes
        return Work(resources={self.aic_read: q_b, "hbm": q_b})

    def _group_program(self, ctx: RecipeContext, owner: tuple[int, tuple[int, ...]], q_rows: int, scanned_rows: int) -> KernelProgram:
        aic, aivs = owner
        active_aivs = aivs[: min(len(aivs), max(1, q_rows))]
        q_work = self._entry_work(q_rows)
        q_st = Stage("query_load", ctx.bind(q_work), placement=_place_aic(ctx, aic, self.aic_count, q_work))
        n_tiles = ceil_div(scanned_rows, ARCH35_V1_TILE_ROWS)
        if n_tiles == 0:
            return KernelProgram((q_st,))
        tile_sync_ms = p12_flag_ms(ctx, "flag_mte2_v_ns") + p12_flag_ms(ctx, "flag_v_mte3_ns")

        def _tile_stages(idx: int) -> tuple[Stage, ...]:
            valid = min(ARCH35_V1_TILE_ROWS, scanned_rows - idx * ARCH35_V1_TILE_ROWS)
            ws = self._tile_works(valid, q_rows)
            s0 = Stage("key_load", ctx.bind(ws["key_load"]),
                       placement=_place_aic(ctx, aic, self.aic_count, ws["key_load"]))
            s1 = Stage("cube_mma", ctx.bind(ws["cube_mma"]), after=("key_load",),
                       placement=_place_aic(ctx, aic, self.aic_count, ws["cube_mma"]))
            s2 = Stage("score_handoff", ctx.bind(ws["score_handoff"]), after=("cube_mma",),
                       placement=_place_aic(ctx, aic, self.aic_count, ws["score_handoff"]))
            s3 = Stage("vec_reduce", ctx.bind(ws["vec_reduce"]), after=("score_handoff",),
                       placement=_place_aivs(ctx, active_aivs, self.aiv_count, ws["vec_reduce"]),
                       previous=("score_row_store",))
            s4 = Stage("score_row_store", ctx.bind(ws["score_row_store"]), after=("vec_reduce",),
                       placement=_place_aivs(ctx, active_aivs, self.aiv_count, ws["score_row_store"]),
                       latency=tile_sync_ms)
            return (s0, s1, s2, s3, s4)

        loop_prog = compact_periodic_pipeline(
            ctx, n_tiles, ARCH35_V1_PIPE_SLOTS, _tile_stages, head=1, tail=1, threshold=COMPACT_TILE_THRESHOLD,
        )
        stages = [q_st, Stage("score_tiles", after=("query_load",), program=loop_prog)]
        tk_ws = self._topk_works(q_rows, scanned_rows)
        prev = "score_tiles"
        for name in ("score_row_load", "trunk_radix_topk", "index_store"):
            if name in tk_ws:
                stages.append(Stage(
                    name, ctx.bind(tk_ws[name]), after=(prev,),
                    placement=_place_aivs(ctx, active_aivs, self.aiv_count, tk_ws[name]),
                ))
                prev = name
        return KernelProgram(tuple(stages))

    def __call__(self, ctx: RecipeContext) -> ParallelProgram:
        n_own = len(self.owners)
        lanes = []
        for lane in range(min(n_own, len(self.q_groups))):
            idxs = tuple(range(lane, len(self.q_groups), n_own))
            own = self.owners[lane]
            lanes.append(PipelineProgram(
                len(idxs), 1,
                lambda i, own=own, idxs=idxs: (Stage("q_group", program=self._group_program(ctx, own, *self.q_groups[idxs[i]])),),
            ))
        return ParallelProgram(tuple(lanes))


def build_arch35_v1_lightning(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    owners: tuple[tuple[int, tuple[int, ...]], ...], sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build native Arch35 aclnnLightningIndexer v1 (isLDOpen=false, 8192-trunk TopK)."""
    q_groups = _q_groups_with_kv(op, contract, ARCH35_V1_S1_BASE, ctx)
    active = min(len(owners), max(1, len(q_groups)))
    score_ws = active * max((qr * kv for qr, kv in q_groups), default=1) * FP32_BYTES
    l2_cap = _l2_capacity(ctx)
    score_l2 = l2_cap > 0 and score_ws <= l2_cap
    has_l2 = _has_l2_rate(ctx)
    aic_r = read_role(ctx.geometry.resource_units)
    aiv_r = read_role(ctx.geometry.resource_units, aiv=0)
    prog = _Arch35V1Program(
        contract, owners, q_groups, aic_r, aiv_r,
        ctx.datapath.aic_count, ctx.datapath.aiv_count, score_l2, has_l2,
    )
    totals: dict[str, Work] = {}
    for qr, kv in q_groups:
        full, rem = divmod(kv, ARCH35_V1_TILE_ROWS)
        _acc(totals, "query_load", prog._entry_work(qr))
        if full:
            for k, w in prog._tile_works(ARCH35_V1_TILE_ROWS, qr).items():
                _acc(totals, k, w, full)
        if rem:
            for k, w in prog._tile_works(rem, qr).items():
                _acc(totals, k, w, 1)
        for k, w in prog._topk_works(qr, kv).items():
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
    core_runs: tuple[tuple[tuple[int, int, int], ...], ...]
    core_q_rows: tuple[int, ...]
    core_split_q_rows: tuple[int, ...]
    group_q_rows: tuple[int, ...]
    group_s2_owners: tuple[int, ...]
    s2_owners_per_group: int
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int
    score_l2: bool
    has_l2_rate: bool

    def _tile_works(self, valid: int, n_owner_tiles: int, q_rows: int, *, sort_ratio: float = 1.0) -> dict[str, Work]:
        c = self.contract
        qr = max(1, int(q_rows))
        key_b = valid * c.cache_row_bytes
        heads = qr * c.index_heads
        score_b = valid * heads * FP32_BYTES
        w_b = heads * BF16_BYTES if c.has_head_weights else 0
        # lightning_indexer_service_vector_arch22.h:380-404: 512-item Sort32+MrgSort4 (3 passes)
        # + 4-block MrgBasicBlock (2048/4) + retained SparseTopK (4096/4) on P10_sort_topk;
        # line 396 DataCopy copyback (2048/4) stays on UB copy (1x), not Sort.
        flush_sort_passes = DEFAULT_S2_TILE + (
            (ARCH22_TOPK_LIMIT * TOPK_MERGE_STRIDE_MULT // ARCH22_FLUSH_BLOCKS)
            if n_owner_tiles > ARCH22_FLUSH_BLOCKS else 0
        )
        flush_copy_ops = DEFAULT_S2_TILE if n_owner_tiles > ARCH22_FLUSH_BLOCKS else 0
        sort_passes = ARCH22_SORT512_PASSES * DEFAULT_S2_TILE + flush_sort_passes
        alu_ops = valid * (ARCH22_VEC_OPS_PER_HEAD * heads - qr) + qr * flush_copy_ops
        vec_ops = alu_ops + int(round(qr * sort_passes * sort_ratio))
        return {
            "key_load": Work(resources={self.aic_read: key_b, "hbm": key_b}),
            "cube_mma": Work(
                matrix={c.matrix_dtype: FLOPS_PER_FMA * valid * heads * c.index_dim},
                resources={"mte1": key_b + heads * c.cache_row_bytes},
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
        # service_vector_arch22.h:484: each owner writes padded arch22_retained_items (2048 pairs) + 128B meta.
        return self.contract.arch22_retained_items

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
        sort_ratio: float,
        fix_setup_ms: float,
        tile_sync_ms: float,
    ) -> tuple[Stage, ...]:
        aivs = raw_aivs[: min(len(raw_aivs), max(1, qr))]
        ws = self._tile_works(valid, n_blks, qr, sort_ratio=1.0)
        vec_caps: dict[str, float] = {}
        if sort_ratio != 1.0:
            vec_res = ctx.resource_names.get("vector")
            b_vec = ws["vec_sort"].resources.get("vector", 0)
            eff_vec = self._tile_works(valid, n_blks, qr, sort_ratio=sort_ratio)["vec_sort"].resources.get("vector", 0)
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
        g_owners = self.group_s2_owners
        g_q_rows = self.group_q_rows
        n_m_lanes = min(len(self.owners), len(g_owners))
        m_branches: list[KernelProgram] = []
        for lane in range(n_m_lanes):
            _, raw_aivs = self.owners[lane]
            merge_aivs = raw_aivs[:1]
            m_st: list[Stage] = []
            for gid in range(lane, len(g_owners), n_m_lanes):
                q_cnt = g_q_rows[gid] if gid < len(g_q_rows) else 0
                if q_cnt <= 0:
                    continue
                m_ws = self._query_merge_works(g_owners[gid], sort_ratio=1.0)
                m_ws_eff = (
                    self._query_merge_works(g_owners[gid], sort_ratio=sort_ratio)
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
                    st_name = f"{name}_{gid}" if len(g_owners) > n_m_lanes else name
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
        sort_ratio = p10_sort_to_alu_op_ratio(ctx)
        fix_setup_ms = p03_fixpipe_setup_ms(ctx)
        tile_sync_ms = p13_paired_ms(ctx)
        credit_ms = ARCH22_ENTRY_EXIT_CREDITS * tile_sync_ms
        branches: list[KernelProgram] = []
        for idx, (c_runs, q_rows, split_q_rows) in enumerate(zip(self.core_runs, self.core_q_rows, self.core_split_q_rows)):
            n_blks = sum(cnt for cnt, _, _ in c_runs)
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
            for run_i, (cnt, valid, qr) in enumerate(c_runs):
                if cnt > 1:
                    loop = compact_periodic_pipeline(
                        ctx, cnt, ARCH22_KEY_L1_SLOTS,
                        lambda _ti, v=valid, q=qr, aic=aic, raw_aivs=raw_aivs, n_blks=n_blks: self._tile_stages(
                            ctx, aic, raw_aivs, v, q, n_blks,
                            sort_ratio=sort_ratio, fix_setup_ms=fix_setup_ms, tile_sync_ms=tile_sync_ms,
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
                            sort_ratio=sort_ratio, fix_setup_ms=fix_setup_ms, tile_sync_ms=tile_sync_ms,
                        )),
                    ))
                    prev_local = t_name
            store_w = self._split_store_work(split_q_rows)
            if store_w.resources:
                core_max_qr = max((qr for _, _, qr in c_runs), default=1)
                store_aivs = raw_aivs[: min(len(raw_aivs), max(1, core_max_qr))]
                st.append(Stage(
                    "split_list_store", ctx.bind(store_w), after=(prev_local,),
                    placement=_place_aivs(ctx, store_aivs, self.aiv_count, store_w),
                ))
            branches.append(KernelProgram(tuple(st)))

        stages: list[Stage] = [Stage("s2_owners", program=ParallelProgram(tuple(branches)))]
        prev = "s2_owners"
        barrier_ms = p13_syncall_ms(ctx) or tile_sync_ms
        if self.s2_owners_per_group > 1 and barrier_ms > 0.0:
            stages.append(Stage("split_barrier", after=(prev,), latency=barrier_ms, keep_empty=True))
            prev = "split_barrier"
        if c.fused_topk:
            m_branches = self._query_exit_branches(ctx, sort_ratio)
            if m_branches:
                stages.append(Stage("query_exit", after=(prev,), program=ParallelProgram(m_branches)))
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


def build_arch22_v1_lightning(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    owners: tuple[tuple[int, tuple[int, ...]], ...], sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    """Build native Arch22 aclnnLightningIndexer v1 (SplitCore + s1BaseSize K-reuse + L2 score slots)."""
    q_kv_groups = _q_groups_with_kv(op, contract, ARCH22_V1_S1_BASE, ctx)
    group_q_rows = tuple(qr for qr, _ in q_kv_groups)
    n_q_groups = len(q_kv_groups)
    runs: list[tuple[int, int, int, int]] = []
    total_blocks = 0
    for gid, (qr, kv) in enumerate(q_kv_groups):
        n_full, rem = divmod(max(1, int(kv)), DEFAULT_S2_TILE)
        if n_full > 0:
            runs.append((gid, n_full, DEFAULT_S2_TILE, qr))
        if rem > 0:
            runs.append((gid, 1, rem, qr))
        total_blocks += n_full + int(rem > 0)
    active = min(len(owners), total_blocks) if total_blocks > 0 else 0
    core_runs, core_seen_groups, core_q_rows, group_s2_owners = _assign_block_runs(runs, active, n_q_groups)
    core_split_q_rows = tuple(
        sum(group_q_rows[gid] for gid in seen if group_s2_owners[gid] > 1)
        for seen in core_seen_groups
    )
    s2_owners_per_group = max(group_s2_owners, default=1)
    slot_ws = sum(
        SCORE_SLOTS * DEFAULT_S2_TILE * max((qr for _, _, qr in c_runs), default=0) * contract.index_heads * FP32_BYTES
        for c_runs in core_runs
    )
    l2_cap = _l2_capacity(ctx)
    score_l2 = l2_cap > 0 and slot_ws <= l2_cap
    has_l2 = _has_l2_rate(ctx)
    aic_r = read_role(ctx.geometry.resource_units)
    aiv_r = read_role(ctx.geometry.resource_units, aiv=0)
    prog = _Arch22V1Program(
        contract, owners, core_runs, core_q_rows, core_split_q_rows,
        group_q_rows, group_s2_owners, s2_owners_per_group,
        aic_r, aiv_r, ctx.datapath.aic_count, ctx.datapath.aiv_count, score_l2, has_l2,
    )
    totals: dict[str, Work] = {}
    for c_runs, q_rows, split_q_rows in zip(core_runs, core_q_rows, core_split_q_rows):
        n_blks = sum(cnt for cnt, _, _ in c_runs)
        if n_blks <= 0:
            continue
        q_b = q_rows * contract.index_heads * contract.cache_row_bytes
        _acc(totals, "query_load", Work(resources={aic_r: q_b, "hbm": q_b}))
        for cnt, valid, qr in c_runs:
            for k, w in prog._tile_works(valid, n_blks, qr).items():
                _acc(totals, k, w, cnt)
        store_w = prog._split_store_work(split_q_rows)
        if store_w.resources:
            _acc(totals, "split_list_store", store_w)
    for g_cnt, q_cnt in zip(group_s2_owners, group_q_rows):
        if q_cnt <= 0:
            continue
        for k, w in prog._query_merge_works(g_cnt).items():
            _acc(totals, k, w, q_cnt)
    return _finalize_plan(
        ctx, contract, totals, prog, sources,
        tiling="arch22_v1_split_core", tile_size=DEFAULT_S2_TILE,
        score_path="l0c->l2->ub" if score_l2 else "l0c->gm->ub",
        s2_splits=s2_owners_per_group,
    )
