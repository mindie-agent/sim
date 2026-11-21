"""Arch22 sparse index scoring with a MemBase score exchange.

Upstream-ABI: aclnnLightningIndexer

[Role] Lower the L0C-to-GM-to-UB path and Arch22 S2 selection cadence.
[Invariants] Typed sorts and explicit SortAll retain their caller ordering.
  Split owners exchange FP32/INT32 pairs through GM.
[Timeouts & Locks] Constant-size planning and lazy tile callbacks; no locks.
[Side-Effect Policy] No device access or external I/O.
"""
from __future__ import annotations

from dataclasses import dataclass

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import ArchId, SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan
from ..contracts import OperatorPlanningContext
from ._lowering import S2Partition
from .common import (
    ARCH22_FLUSH_BLOCKS, ARCH22_SMALL_QUERY_MAX, ARCH22_TOPK_LIMIT, BF16_BYTES, DEFAULT_S2_TILE,
    FP32_BYTES, LIGHTNING_V2_ABI, MERGE_FAN_IN, PAIR_BYTES, QSA_ABI, QSA_OVERLAY_SOURCE, SORT_GROUP,
    Lowering, PhaseWork, SparseIndexerContract,
    _integer, align, build_plan, ceil_div, indexer_contract,
    owner_pairs, pair_merge, sort_ops, unsupported, vector,
)

ARCH_ID = ArchId("npu_arch_2201")
UPSTREAM_ABI = "aclnnLightningIndexer"
_SUPPORTED_ABIS = frozenset({UPSTREAM_ABI, LIGHTNING_V2_ABI, QSA_ABI})
_SCORE_ROUTE = "l0c->gm->ub"
_EXPLICIT_RETAINED_BARRIERS = 3  # After SortAll, after merge, after copyback.
_TYPED_RETAINED_FLUSH_BARRIERS = 4  # After flush, flush copyback, merge, and retained copyback.
SOURCES: tuple[SourceRef, ...] = (
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer/op_kernel/arch22/lightning_indexer_vector.h",
        source_sha256="ef4d26881ddba381df9cf4198e01b39ff4cd223189dd416b51ce468a5b656af4",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer/op_kernel/arch22/lightning_indexer_service_vector_arch22.h",
        source_sha256="33bb65e7a0731dcddadd9ef5d75e7d73dcdfcf67cd28adb6f943310f73249e87",
    ),
    SourceRef(
        repository="cann/ops-transformer", revision="2d3f4f6db0231ac2e9e487cf62e94f6e173eb946",
        relative_path="attention/lightning_indexer_v2/op_host/lightning_indexer_v2_tiling.cpp",
        source_sha256="b8988f5bdc1ccf8900c2a7853c4edf012f8cb99f6e9b5af40af56167bcd9aade",
    ),
    SourceRef(
        repository="Ascend/op-plugin", revision="48189acf95882e23fae56ea96ceeb89358cb42da",
        relative_path="op_plugin/ops/opapi/LightningIndexerKernelNpuOpApi.cpp",
        source_sha256="3b4aefd825b7fe1fc58a89cf67a1d11abe30fa71e95268d884fa9f0b9016d627",
    ),
    SourceRef(
        repository="vllm-project/vllm-ascend", revision="0c99d3fc5ef471224a2109ebb6b6ab848a7e66bb",
        relative_path="vllm_ascend/models/qwen4_exp/lightning_indexer.py",
        source_sha256="1d265521c2a44a904342e24a0d917aec575240469fe0f1dbc5ea61133d0e5736",
    ),
)
LIGHTNING_SOURCES, QSA_SOURCE = SOURCES[:-1], SOURCES[-1]


@dataclass(frozen=True, slots=True)
class SortAllTrace:
    sort32_repeats: int
    mrg_passes: tuple[tuple[int, int], ...]
    passes: int
    needs_copyback: bool
    barriers: int


def trace_sort_all(n: int) -> SortAllTrace:
    """Trace the source Sort32, MrgSort and copyback recurrence."""
    _integer("sort_all_items", n)
    if n % SORT_GROUP:
        raise ModelingError(f"sparse_indexer_sort_all_requires_multiple_of_32:{n}")
    repeats = groups = n // SORT_GROUP
    merges = []
    while groups > MERGE_FAN_IN:
        groups //= MERGE_FAN_IN
        merges.append((MERGE_FAN_IN, groups))
    if groups > 1:
        merges.append((groups, 1))
    passes = len(merges)
    needs_copyback = passes % 2 == 0
    return SortAllTrace(repeats, tuple(merges), passes, needs_copyback,
                        1 + passes + int(needs_copyback))


@dataclass(frozen=True, slots=True)
class FlushMergeTrace:
    block_sorts: int
    flushes: int
    retained_merges: int
    split_merges: int
    sort32_repeats: int
    mrg_passes: int
    barriers: int
    padded_rows: int
    padded_tail_rows: int
    flush_pair_bytes: int = 0


def arch22_flush_period(act_s1_size: int, topk: int, over_2k: bool) -> int:
    if act_s1_size <= ARCH22_SMALL_QUERY_MAX and topk <= ARCH22_TOPK_LIMIT and not over_2k:
        return ARCH22_FLUSH_BLOCKS
    return 1


def trace_arch22_topk_flushes(
    scanned_rows: int, *, act_s1_size: int = 1, topk: int = ARCH22_TOPK_LIMIT,
    tile_size: int = DEFAULT_S2_TILE, s2_splits: int = 1,
    is_sparse_count_over_2k: bool = False,
) -> FlushMergeTrace:
    """Count local sorts, modeled vector barriers and incoming flush payloads."""
    _integer("scanned_rows", scanned_rows, minimum=0)
    for name, value in (("act_s1_size", act_s1_size), ("topk", topk),
                        ("tile_size", tile_size), ("s2_splits", s2_splits)):
        _integer(name, value)
    if tile_size % SORT_GROUP:
        raise ModelingError("sparse_indexer_s2_tile_requires_multiple_of_32")
    period = arch22_flush_period(act_s1_size, topk, is_sparse_count_over_2k)
    explicit = period == 1
    partition = S2Partition(scanned_rows, tile_size, s2_splits)
    flushes = retained = barriers = 0
    sort_all = trace_sort_all(tile_size) if explicit else None
    for rows, count, _ in partition.variants():
        tiles = ceil_div(rows, tile_size)
        owner_flushes = ceil_div(tiles, period)
        flushes += count * owner_flushes
        retained += count * (owner_flushes if explicit else owner_flushes - 1)
        if sort_all is not None:
            barriers += count * tiles * (sort_all.barriers + _EXPLICIT_RETAINED_BARRIERS)
        else:
            later_flushes = owner_flushes - 1
            single_tail = later_flushes > 0 and tiles % period == 1
            barriers += count * (tiles + 1 + _TYPED_RETAINED_FLUSH_BARRIERS * later_flushes - single_tail)
    sorts = partition.tiles
    return FlushMergeTrace(sorts, flushes, retained,
                           max(0, partition.owners - 1),
                           sorts * sort_all.sort32_repeats if sort_all else 0,
                           sorts * sort_all.passes if sort_all else 0, barriers,
                           sorts * tile_size - scanned_rows, tile_size if scanned_rows % tile_size else 0,
                           sorts * tile_size * PAIR_BYTES)


def _flush_period(contract: SparseIndexerContract) -> int:
    act_s1_size = ceil_div(contract.query_tokens, contract.batch_size)
    return arch22_flush_period(act_s1_size, contract.topk_items, contract.is_sparse_count_over_2k)


def _barrier(name: str) -> PhaseWork:
    return PhaseWork(name, metadata={"vector_barriers": 1})


def _select_tile(
    contract: SparseIndexerContract, rows: int, index: int, valid: int, tile_size: int,
) -> tuple[PhaseWork, ...]:
    items = ceil_div(valid, contract.block_size)
    tile_items = tile_size // contract.block_size
    stages = [
        vector("prepare_sort_pairs", items + 2 * (tile_items - items),
               pair_bytes=tile_items * PAIR_BYTES, padded_items=tile_items - items),
    ]
    if _flush_period(contract) == 1:
        stages.extend(_explicit_sort(tile_items))
        stages.append(_barrier("sort_all_complete"))
        stages.extend(_retained_merge(tile_items, contract.arch22_retained_items))
    else:
        stages.extend(_typed_sort(rows, index, tile_size, tile_items))
    return tuple(stages)


def _explicit_sort(items: int) -> tuple[PhaseWork, ...]:
    trace = trace_sort_all(items)
    stages = [
        vector("sort32", trace.sort32_repeats * sort_ops(SORT_GROUP),
               repeats=trace.sort32_repeats, pair_bytes=items * PAIR_BYTES),
        _barrier("sort32_barrier"),
    ]
    run = SORT_GROUP
    for number, (inputs, repeats) in enumerate(trace.mrg_passes, 1):
        pairs = inputs * run * repeats
        stages.extend((
            vector(f"mrg_sort_{number}", pairs, valid_inputs=inputs, repeats=repeats,
                   valid_mask=(1 << inputs) - 1, pair_bytes=pairs * PAIR_BYTES),
            _barrier(f"mrg_sort_barrier_{number}"),
        ))
        run *= inputs
    if trace.needs_copyback:
        stages.extend((
            vector("sort_copyback", items, pair_bytes=items * PAIR_BYTES),
            _barrier("sort_copyback_barrier"),
        ))
    return tuple(stages)


def _typed_sort(rows: int, index: int, tile_size: int, tile_items: int) -> tuple[PhaseWork, ...]:
    stages = [vector("typed_sort", sort_ops(tile_items), repeats=tile_items // SORT_GROUP,
                     pair_bytes=tile_items * PAIR_BYTES), _barrier("typed_sort_barrier")]
    period = ARCH22_FLUSH_BLOCKS
    last_tile = (index + 1) * tile_size >= rows
    if (index + 1) % period and not last_tile:
        return tuple(stages)

    blocks = index % period + 1
    flush_items = blocks * tile_items
    if blocks > 1:
        stages.append(vector("flush_merge", flush_items, valid_inputs=blocks,
                             pair_bytes=flush_items * PAIR_BYTES))
    elif index < period:
        stages.append(vector("flush_copy", flush_items, pair_bytes=flush_items * PAIR_BYTES))
    stages.append(_barrier("flush_barrier"))
    if index < period:
        return tuple(stages)
    if blocks > 1:
        stages.append(vector("flush_copyback", flush_items, pair_bytes=flush_items * PAIR_BYTES))
        stages.append(_barrier("flush_copyback_barrier"))
    stages.extend(_retained_merge(flush_items, ARCH22_TOPK_LIMIT))
    return tuple(stages)


def _retained_merge(incoming: int, retained: int) -> tuple[PhaseWork, ...]:
    return (
        pair_merge("retained_topk_merge", incoming, retained, retained),
        _barrier("retained_merge_barrier"),
        vector("retained_copyback", retained, pair_bytes=retained * PAIR_BYTES),
        _barrier("retained_copyback_barrier"),
    )


def _contract(op: OpSpec, ctx: OperatorPlanningContext) -> SparseIndexerContract:
    contract = indexer_contract(op, ctx, UPSTREAM_ABI)
    if contract.abi not in _SUPPORTED_ABIS or contract.cache_dtype != "bf16":
        raise unsupported("arch22_abi", f"Arch22 cannot lower {contract.abi} with {contract.cache_dtype}")
    return contract


@dataclass(frozen=True, slots=True)
class _Arch22Strategy:
    arch_id: ArchId = ARCH_ID
    abi: str = UPSTREAM_ABI
    sources: tuple[SourceRef, ...] = SOURCES

    def accepts(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> bool:
        if ctx.compute_arch.arch_id != self.arch_id:
            return False
        try:
            _contract(op, ctx)
        except ModelingError:
            return False
        return True

    def build(self, op: OpSpec, ctx: OperatorPlanningContext, /) -> OperatorPlan:
        contract = _contract(op, ctx)
        datapath = ctx.datapath
        if datapath.usable_paths:
            datapath.route("l0c", "gm", "fp32")
            datapath.route("gm", "ub", "fp32")
        owners = owner_pairs(ctx, 1)
        qsa = contract.abi == QSA_ABI
        sources = LIGHTNING_SOURCES
        if qsa:
            sources = (QSA_SOURCE,) if contract.has_head_weights else (QSA_OVERLAY_SOURCE,)
        # Align score blocks so local maxima are disjoint across S2 owners.
        tile_size = align(contract.s2_tile_size, SORT_GROUP * contract.block_size)
        lowering = Lowering(
            contract.abi, "membase_s2_tiles", tile_size, _SCORE_ROUTE,
            tile_selection=_select_tile, flush_period=_flush_period(contract),
            score_align=SORT_GROUP, head_weight_item_bytes=FP32_BYTES if qsa else BF16_BYTES,
        )
        return build_plan(op, ctx, contract, lowering, owners, sources)


STRATEGY = _Arch22Strategy()
