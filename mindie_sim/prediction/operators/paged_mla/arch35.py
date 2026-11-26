"""Arch35 FIA, FlashMLA, and mixed-quant sparse MLA strategies.

[Role] Lower routes, resident KV tiles, result slots, and packed KV gathers.
[Invariants] Validate direct L0C->UB and UB->L1 paths and screen simultaneous
  L1, UB and L0C liveness before building the owner-local execution pipeline.
[Timeouts & Locks] Bounded tiling candidates and no locks.
[Side-Effect Policy] Inspect declared capabilities without device access.
Upstream-ABI: aclnnFlashMla
"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, replace
from typing import Literal

from mindie_sim.schema.common import dtype_bytes
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...runtime.tile_solver import (
    LiveBufferSpec, TileCandidate, TileInfeasibleError, TilePipelineSpec, TileShape, solve,
)
from ..contracts import ArchStrategy, OperatorPlanningContext
from ._lowering import (
    DENSE_CANN_ABIS, SOFTMAX_STATE_SCALARS, PipelineLayout, RowCuts,
    fixpipe_rectangle_bytes, make_plan, owner_pairs, resolve_fia_host_plan,
)
from .common import (
    DSL_FLASH_ABI, DSL_MQS_ABI, FIA_ABI, FP32_BYTES, MQS_ABI,
    MQS_ORIGINAL_ROW_BYTES, SUPPORTED_ABIS, PagedMlaContract, align_bytes, capability_gap, ceil_div, contract_for,
    validate_dense_cann_geometry,
)

TARGET_BACKEND: Literal["ascend"] = "ascend"
ARCH_ID = "npu_arch_3510"
UPSTREAM_ABI = "aclnnFlashMla"
FIA_M_TILE = 64
FIA_S2_TILE = 128
FLASH_M_TILE = 96
FLASH_S2_TILE = 112
FLASH_FIXED_L0C_BYTES = 262_144
MQS_M_TILE = 64
MQS_S2_TILE = 128
Q_DEPTH = 1
FLASH_MM1_DEPTH = 1
FIA_MM1_DEPTH = 2
MM2_DEPTH = 2
KVP_DEPTH = 3
PIPELINE_DRAIN = 3
MQS_GATHER_ROWS = 16
ACLNN_MQS_GATHER_BUFFER_BYTES_PER_ROW = 640
DSL_MQS_SPLIT_SLOTS_PER_CORE = 2
FD_MIN_ROW_TILES = 5
FD_TARGET_TILES_PER_CORE = 4
FD_MIN_SAVED_TILES = 4
FD_M_COST = 24  # Cost of 64 query heads in the DSL metadata ABI.
FD_S2_BLOCK = 64
FD_S2_BLOCK_COST = 10
FD_SELECTIVE_ROWS = 72
FD_SELECTIVE_CORES = 32
FD_SELECTIVE_GROUP_ROWS = 9
FD_SELECTIVE_GROUP_CORES = 4
FD_SELECTIVE_CUT_STRIDE = 11
FD_SELECTIVE_CMP_TILES = 4
PV_L0C_WIDTH = 256
CLOCKS = (
    ("q_load", "kv_load", "dequant", "kv_to_l1", "kv_refill", "qk_operands", "qk", "score_store"),
    ("softmax", "probability", "pv_load", "pv_operands", "pv", "partial"),
    ("output_update", "split_write"),
)
FLASH_CLOCKS = (
    CLOCKS[0],
    ("softmax", "probability", "pv_load", "pv_operands", "pv0", "partial0", "pv1", "partial1"),
    ("release0", "release1", "split_write"),
)
DSL_MQS_CLOCKS = (CLOCKS[0], ("sink_init", *CLOCKS[1]), CLOCKS[2])
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_kernel/arch35/fia_block_cube_noquant_mla.h",
        source_sha256="0ce3c62b573478790b78fe057b5a697d45042c1b7f3d7a9ee582410fe1d1c811",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/fused_infer_attention_score/op_kernel/arch35/fia_block_vec_noquant_mla.h",
        source_sha256="e62a9d1704937b5a67b1b4e532318d81bff5bd39d76fd2cf7f17e6b5ceb8ea7d",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/flash_mla/op_kernel/arch35/flash_mla_block_cube.h",
        source_sha256="a6369c827ddca2d5d6a03e8cd058c0331eb9c63b416e3ebb6d83f633b5bf01da",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_api/aclnn_mixed_quant_sparse_flash_mla.h",
        source_sha256="4b613b72ec6cb36209328b084b74aa76e32ec86ad872976e564d9e4eea60e671",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/flash_mla_with_kvcache/flash_mla_with_kvcache.py",
        source_sha256="044da876085c5b9b16b2ee2c70654f9ae38c3a2a572b7fcaed1cabdb4ca7faaf",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla.py",
        source_sha256="9fd0ad943a677e630ffb13df255b73a7b8889de0ed5a032996c2a3be6fc5ee36",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/flash_mla/op_kernel/arch35/flash_mla_block_vec.h",
        source_sha256="e57170b54c8a97e60d2500604c24a4623d7aef3356be49b55a98411d5e7e7a3f",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_vector.h",
        source_sha256="f60e4f44294b0677522bdeec766f6bb66f61fdc3375ac95f547000eff7c10237",
    ),
    SourceRef(
        repository="tilelang-ascend/kernels",
        revision="5e7ace8cdcc16a5b062ba69d8fdf93d4f1d62223",
        relative_path="samples/attention/mixed_quant_sparse_flash_mla/mixed_quant_sparse_flash_mla_metadata.py",
        source_sha256="51e13bd02d35704f99298dd9fdff36ddd82c941a8106c07d8e668d29664ad40a",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/README_en.md",
        source_sha256="e5f013aca264fd192f5780455eeb4fa9d378a03d9576e34bb31488cdf39e75c0",
    ),
    SourceRef(
        repository="cann/ops-transformer",
        revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/mixed_quant_sparse_flash_mla/op_kernel/arch35/mixed_quant_sparse_flash_mla_csa_block_cube.h",
        source_sha256="df3f7ca12e1bad28f2f52d4bdfe5d43326a176af88d00c7d1121d4cc4e62290e",
    ),
)
ABI_SOURCES = {
    FIA_ABI: SOURCES[:2],
    UPSTREAM_ABI: (SOURCES[2], SOURCES[6]),
    MQS_ABI: (SOURCES[3], SOURCES[7], SOURCES[9], SOURCES[10]),
    DSL_FLASH_ABI: (SOURCES[4],),
    DSL_MQS_ABI: (SOURCES[5], SOURCES[8]),
}


def _fd_tile_cost(keys: int) -> int:
    return FD_M_COST + FD_S2_BLOCK_COST * ceil_div(keys, FD_S2_BLOCK)


@dataclass(frozen=True, slots=True)
class _FdRow:
    ori: int
    cmp: int

    @property
    def tiles(self) -> int:
        return ceil_div(self.ori, MQS_S2_TILE) + ceil_div(self.cmp, MQS_S2_TILE)

    @property
    def cost(self) -> int:
        cost = 0
        for keys in (self.ori, self.cmp):
            full, tail = divmod(keys, MQS_S2_TILE)
            cost += full * _fd_tile_cost(MQS_S2_TILE)
            if tail:
                cost += _fd_tile_cost(tail)
        return cost

    def tile_cost(self, tile: int) -> int:
        ori_tiles = ceil_div(self.ori, MQS_S2_TILE)
        if tile < ori_tiles:
            return _fd_tile_cost(min(MQS_S2_TILE, self.ori - tile * MQS_S2_TILE))
        return _fd_tile_cost(min(MQS_S2_TILE, self.cmp - (tile - ori_tiles) * MQS_S2_TILE))


def _fd_row_runs(c: PagedMlaContract) -> Iterator[tuple[int, int, int, _FdRow]]:
    for seq in range(len(c.sequences)):
        for query, repeat in c.query_runs(seq):
            if repeat:
                yield seq, query, repeat, _FdRow(*c.bank_counts(seq, query))


def _fd_core_cuts(c: PagedMlaContract, blocks: int, total_cost: int) -> RowCuts:
    # Project the native greedy core ranges onto row-local tile boundaries.
    rows = ((seq, query, row) for seq, start, repeat, row in _fd_row_runs(c)
            for query in range(start, start + repeat))
    current = next(rows, None)
    cuts: dict[tuple[int, int], list[int]] = {}
    tile, remaining = 0, current[2].cost if current else 0
    for cores_left in range(blocks, 1, -1):  # The last core takes the remaining rows.
        budget = limit = total_cost // cores_left
        assigned = 0
        whole_rows = True
        while current is not None:
            seq, query, row = current
            tiles = row.tiles
            short_row = tile == 0 and tiles < FD_MIN_ROW_TILES
            if whole_rows and (not tiles or budget + row.tile_cost(tiles - 1) // 2 >= remaining
                               or (short_row and not assigned)):
                cost, taken = remaining, tiles - tile
            elif short_row:
                break
            else:
                whole_rows = False
                cost, taken = row.tile_cost(tile), 1
                if assigned and budget + cost // 2 < cost:
                    break
            assigned += taken
            budget -= cost
            remaining -= cost
            tile += taken
            if tile == tiles:
                current = next(rows, None)
                tile, remaining = 0, current[2].cost if current else 0
        if current is None:
            break
        if tile:
            cuts.setdefault(current[:2], []).append(tile)
        total_cost -= limit - budget
    return {row: tuple(offsets) for row, offsets in cuts.items()}


def _dsl_row_cuts(c: PagedMlaContract, owners: int) -> RowCuts:
    blocks = min(owners, c.splits)
    if blocks <= 1:
        return {}
    rows = sum(q for q, _ in c.sequences)
    total_tiles = total_cost = max_row_tiles = 0
    selective = rows == FD_SELECTIVE_ROWS and blocks == FD_SELECTIVE_CORES
    for _, _, repeat, row in _fd_row_runs(c):
        total_tiles += row.tiles * repeat
        total_cost += row.cost * repeat
        max_row_tiles = max(max_row_tiles, row.tiles)
        selective &= (row.ori, row.cmp) == (MQS_S2_TILE, FD_SELECTIVE_CMP_TILES * MQS_S2_TILE)
    if selective:
        # Native selective plan assigns 11/11/11/12 tiles per four-core group.
        local_cuts = dict(divmod(core * FD_SELECTIVE_CUT_STRIDE, 1 + FD_SELECTIVE_CMP_TILES)
                          for core in range(1, FD_SELECTIVE_GROUP_CORES))
        cuts = {}
        index = 0
        for seq, start, repeat, _ in _fd_row_runs(c):
            for query in range(start, start + repeat):
                if (cut := local_cuts.get(index % FD_SELECTIVE_GROUP_ROWS)) is not None:
                    cuts[seq, query] = (cut,)
                index += 1
        return cuts
    if max_row_tiles < FD_MIN_ROW_TILES:
        return {}
    plan_blocks = min(blocks, max(rows, total_tiles // FD_TARGET_TILES_PER_CORE))
    if ceil_div(rows, blocks) * max_row_tiles - ceil_div(total_tiles, plan_blocks) < FD_MIN_SAVED_TILES:
        return {}
    return _fd_core_cuts(c, plan_blocks, total_cost)


def _solve(op, ctx, c, keys, head_tile, mm1_depth, resident, result_alignment, pairs):
    geometry = replace(
        ctx.geometry, buffers={**dict(ctx.datapath.buffers), **ctx.geometry.buffers},
        datapath=ctx.datapath, stream_owners=ctx.stream_owners,
    )
    elem = int(dtype_bytes(op.dtype or "bf16"))
    aiv_count = min(len(aivs) for _, aivs in pairs)
    candidates = []
    is_flash = c.abi == UPSTREAM_ABI
    is_fia = c.abi == FIA_ABI
    fixed_cube_alloc = is_flash or is_fia
    m_extent = (
        max((q * c.physical_heads for q, _ in c.sequences if q), default=c.physical_heads)
        if fixed_cube_alloc else c.physical_heads
    )
    while head_tile:
        heads = min(head_tile, m_extent)
        vec_heads = ceil_div(heads, aiv_count)
        pv_columns = min(c.dv, PV_L0C_WIDTH)
        scores = fixpipe_rectangle_bytes(heads, keys) // aiv_count
        pv = vec_heads * pv_columns * FP32_BYTES
        # Compiled Arch35 FlashMLA (M96/S2_112, 519,168 B L1) and FIA (M64/S2_128,
        # 516,096 B L1) reserve fixed cube L1 and 256 KiB L0C regardless of live
        # heads/keys (flash_mla_block_cube.h:62-135; fia_kernel_noquant_mla.h:165;
        # fia_block_cube_noquant_mla.h:103-113).
        q_l1 = (FLASH_M_TILE if is_flash else FIA_M_TILE if is_fia else heads) * c.dqk * elem
        kvp = (
            FLASH_S2_TILE * max(c.dqk, c.dv + FLASH_M_TILE) * elem
            if is_flash
            else FIA_S2_TILE * c.dqk * elem
            if is_fia
            else keys * max(c.dqk, c.dv + heads) * elem
        )
        l0c = (
            (FLASH_FIXED_L0C_BYTES // 2)
            if fixed_cube_alloc
            else (heads * max(keys, pv_columns) * FP32_BYTES)
        )
        state = vec_heads * (c.dv + 2) * FP32_BYTES
        buffers = (
            LiveBufferSpec("query", "l1", q_l1, 0, PIPELINE_DRAIN, "aic", slots=Q_DEPTH),
            LiveBufferSpec("kvp", "l1", kvp, 0, PIPELINE_DRAIN, "aic",
                           slots=KVP_DEPTH if (resident or is_fia) else 2),
            LiveBufferSpec("mm1", "ub", scores, 0, 2, alignment_bytes=result_alignment, slots=mm1_depth),
            LiveBufferSpec("mm2", "ub", pv, 1, PIPELINE_DRAIN,
                           alignment_bytes=result_alignment, slots=MM2_DEPTH),
            LiveBufferSpec("online_state", "ub", state, 0, PIPELINE_DRAIN),
            LiveBufferSpec("l0c", "l0c", l0c, 0, PIPELINE_DRAIN, "aic", slots=2),
        )
        if c.mixed_quant:
            gather_pitch = (ACLNN_MQS_GATHER_BUFFER_BYTES_PER_ROW if c.abi == MQS_ABI
                            else MQS_ORIGINAL_ROW_BYTES)
            buffers += (LiveBufferSpec("dequant", "ub", min(keys, MQS_GATHER_ROWS) *
                                       (gather_pitch + c.dqk * elem),
                                       0, PIPELINE_DRAIN, slots=2),)
        if c.sinks:
            buffers += (LiveBufferSpec("sinks", "ub", vec_heads * FP32_BYTES, 0, PIPELINE_DRAIN),)
        candidates.append(TileCandidate(
            "online_reduce", TileShape(heads, keys, c.dqk, c.dqk, c.dv),
            KVP_DEPTH, (ctx.effective_aic_cores, ctx.effective_aiv_cores), 1, 1,
            buffers, issued_flops=2 * heads * keys * (c.dqk + c.dv), source_guard=c.abi,
        ))
        if heads == 1 or c.sinks or fixed_cube_alloc:
            break
        head_tile = heads // 2
    try:
        return solve(TilePipelineSpec("paged_mla", tuple(candidates)), op, geometry)
    except TileInfeasibleError as exc:
        raise capability_gap("paged_mla_tile_infeasible", c.abi, ARCH_ID) from exc


@dataclass(frozen=True, slots=True)
class _DirectStrategy:
    arch_id = ARCH_ID
    abi = UPSTREAM_ABI
    sources = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        return ctx.compute_arch.arch_id == self.arch_id and contract_for(op, ctx).abi in SUPPORTED_ABIS

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /):
        c = contract_for(op, ctx)
        if ctx.compute_arch.arch_id != self.arch_id or c.atomic_split_decode:
            raise capability_gap("paged_mla_unsupported_abi_or_atomic_mode", c.abi, ctx.compute_arch.arch_id)
        validate_dense_cann_geometry(c, self.arch_id)
        if c.abi == FIA_ABI:
            heads, mm1_depth = FIA_M_TILE, FIA_MM1_DEPTH
            key_tile = min(FIA_S2_TILE, c.s2_base_size)
        elif c.mixed_quant:
            heads, mm1_depth = MQS_M_TILE, FLASH_MM1_DEPTH
            key_tile = MQS_S2_TILE if c.abi == DSL_MQS_ABI else min(MQS_S2_TILE, c.s2_base_size)
        else:
            heads, key_tile, mm1_depth = FLASH_M_TILE, FLASH_S2_TILE, FLASH_MM1_DEPTH
        if c.abi == DSL_MQS_ABI:
            max_extent = max((max(c.bank_counts(seq, query)) for seq in range(len(c.sequences))
                              for query, repeat in c.query_runs(seq) if repeat), default=0)
        elif c.mixed_quant:
            max_extent = max((sum(c.bank_counts(seq, query)) for seq in range(len(c.sequences))
                              for query, _ in c.query_runs(seq)), default=0)
        else:
            max_extent = max((kv for q, kv in c.sequences if q), default=0)
        keys = min(key_tile, max_extent)
        resident = c.latent_alias or c.mixed_quant
        peaks = ()
        pairs = owner_pairs(ctx, c.abi)
        panel_width = 0
        if c.abi == self.abi:
            panel_width = PV_L0C_WIDTH
            if c.dv != MM2_DEPTH * panel_width:
                raise capability_gap("paged_mla_flash_value_width", c.abi, self.arch_id)
        if panel_width or c.sinks:
            pairs = tuple((aic, aivs) for aic, aivs in pairs if len(aivs) == 2)
            if not pairs:
                raise capability_gap("paged_mla_requires_two_aivs", c.abi, self.arch_id)
        direct = ctx.datapath.route("l0c", "ub")
        probability = ctx.datapath.route("ub", "l1")
        if keys:
            solution = _solve(op, ctx, c, keys, heads, mm1_depth, resident, direct.path.alignment_bytes, pairs)
            heads = solution.tile_shape.m
            peaks = tuple((p.pool, p.peak_bytes, p.capacity_bytes) for p in solution.pool_peaks)
            elem = int(dtype_bytes(op.dtype or "bf16"))
            for aic, aivs in pairs:
                vec_heads = ceil_div(heads, len(aivs))
                score_bytes = fixpipe_rectangle_bytes(heads, keys) // len(aivs)
                partial_bytes = vec_heads * min(c.dv, PV_L0C_WIDTH) * FP32_BYTES
                probability_bytes = vec_heads * key_tile * elem
                for aiv in aivs:
                    ctx.datapath.route(
                        "l0c", "ub", "fp32", (aic, aiv),
                        align_bytes(max(score_bytes, partial_bytes), direct.path.alignment_bytes),
                    )
                    ctx.datapath.route(
                        "ub", "l1", op.dtype or "bf16", (aiv, aic),
                        align_bytes(probability_bytes, probability.path.alignment_bytes),
                    )
        kvp_depth = KVP_DEPTH if resident else 2
        result_rings = (
            (("mm2_panel0", 1, "partial0", "release0"), ("mm2_panel1", 1, "partial1", "release1"))
            if panel_width else (("mm2", MM2_DEPTH, "partial", "output_update"),)
        )
        rings = (
            ("mm1", mm1_depth, "score_store", "softmax"),
            *result_rings,
            ("kvp", kvp_depth, "kv_load", "pv1" if panel_width else "pv"),
            ("probability", kvp_depth, "probability", "pv_operands"),
            *((("kv_ws", KVP_DEPTH, "kv_to_l1", "kv_refill"),) if c.abi == MQS_ABI else ()),
        )
        clocks = FLASH_CLOCKS if panel_width else DSL_MQS_CLOCKS if c.sinks else CLOCKS
        host_plan = None
        if c.abi == FIA_ABI and c.splits == 1 and max_extent > 0:
            aiv_per_aic = min(len(aivs) for _, aivs in pairs)
            host = resolve_fia_host_plan(
                sequences=c.sequences, physical_heads=c.physical_heads,
                aic_count=len(pairs), aiv_per_aic=aiv_per_aic, is_arch35=True,
                s2_tile=key_tile,
            )
            if host.fd_enabled:
                host_plan = host
        elem_b = int(dtype_bytes(op.dtype or "bf16"))
        l2_cap = ctx.usable_l2_bytes if c.abi == MQS_ABI else 0
        if c.abi in DENSE_CANN_ABIS:
            total_groups = sum(ceil_div(q * c.physical_heads, heads) for q, _ in c.sequences if q)
        else:
            total_groups = sum(q for q, _ in c.sequences) * ceil_div(c.physical_heads, heads)
        most_splits = min(ceil_div(max_extent, key_tile), c.splits) if max_extent else 0
        lanes = min(len(pairs), max(1, most_splits))
        groups_per_wave = len(pairs) // lanes
        active_owners = (
            min(len(pairs), min(groups_per_wave, total_groups) * min(lanes, max(1, most_splits)))
            if total_groups
            else 0
        )
        mqs_ws_per_core = KVP_DEPTH * key_tile * c.dqk * elem_b
        mqs_ws_bytes = max(1, active_owners) * mqs_ws_per_core if active_owners else 0
        mqs_ws_l2_resident = (
            c.abi == MQS_ABI and l2_cap > 0 and active_owners > 0 and mqs_ws_bytes <= l2_cap
        )
        layout = PipelineLayout(
            clocks, rings, heads, key_tile, resident_latent=resident,
            matrix_output=(direct.path.unit, direct.path.alignment_bytes),
            probability_output=(probability.path.unit, probability.path.alignment_bytes),
            pv_panel_width=panel_width, ws_l2_resident=mqs_ws_l2_resident,
        )
        workspace = {}
        row_cuts = None
        if c.abi == MQS_ABI:
            workspace = dict(
                workspace_slots_per_core=KVP_DEPTH,
                workspace_bytes_per_core=mqs_ws_per_core,
                workspace_bytes=mqs_ws_bytes,
            )
        elif c.abi == DSL_MQS_ABI:
            slot_bytes = c.physical_heads * (c.dv + SOFTMAX_STATE_SCALARS) * FP32_BYTES
            per_core = DSL_MQS_SPLIT_SLOTS_PER_CORE * slot_bytes
            workspace = dict(workspace_slots_per_core=DSL_MQS_SPLIT_SLOTS_PER_CORE,
                             workspace_bytes_per_core=per_core, workspace_bytes=len(pairs) * per_core)
            row_cuts = _dsl_row_cuts(c, len(pairs))
        return make_plan(
            op, ctx, c, layout, ABI_SOURCES[c.abi], pairs=pairs, row_cuts=row_cuts,
            fia_host_plan=host_plan,
            buffer_depths=(Q_DEPTH, mm1_depth, MM2_DEPTH), pool_peaks=peaks,
            matrix_output_path=direct.path.name, probability_path=probability.path.name,
            resident_latent=resident, **workspace,
        )


STRATEGY: ArchStrategy = _DirectStrategy()
