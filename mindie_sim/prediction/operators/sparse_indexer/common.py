"""Shared sparse indexer contract, work ledger, and finite pipeline lowering.

[Role] Convert query and S2 geometry into matrix, transfer, reduction and
  selection work. Bind leaf-specific score handoffs and selection schedules.
[Invariants] The descriptor and executable program use the same phase work;
  padding affects physical transfers and selection, never logical dot FLOPs.
[Timeouts & Locks] Planning uses a bounded set of tile shapes. Execution keeps
  only the active owners and their live tiles; no I/O or locks.
[Side-Effect Policy] Builds immutable plans for the caller's virtual engine.
"""
from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field, fields, replace

from mindie_sim.public.errors import UnmodeledCapabilityError
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import SourceRef
from mindie_sim.schema.operators import OpSpec

from ...mechanism import OperatorPlan, Work
from ...ownership import read_role
from ...program import KernelProgram, ParallelProgram, PipelineProgram, RecipeContext, Stage
from ...results import FusedPipelineDescriptor, FusedPipelineSpec, PipelinePhase, WorkItem
from ...runtime.ring_buffer import RingBuffer
from ..contracts import OperatorPlanningContext

FAMILY = "sparse_indexer"

LIGHTNING_ABI = "aclnnLightningIndexer"
LIGHTNING_V2_ABI = "aclnnLightningIndexerV2"
QLI_V2_ABI = "aclnnQuantLightningIndexerV2"
VLLM_QLI_ABI = "VllmQuantLightningIndexer"
VLLM_QLI_ABIS = frozenset({VLLM_QLI_ABI, "vllm_quant_lightning_indexer_v2", "npu_quant_lightning_indexer_v2"})
QSA_ABI = "aclnnLightningIndexerQSA"
MSA_ABI = "aclnnMsaIndexScore"
MSA_V2_ABI = "aclnnMsaIndexScoreV2"
REGION_ABI = "npu_indexer_region_score"
DSL_QSLI_ABI = "cannbot_dsl.quant_sparse_lightning_indexer"
MSA_ABIS = frozenset({MSA_ABI, MSA_V2_ABI})
QUANT_ABIS = VLLM_QLI_ABIS | {QLI_V2_ABI}
SUPPORTED_ABIS = MSA_ABIS | QUANT_ABIS | {
    LIGHTNING_ABI, LIGHTNING_V2_ABI, QSA_ABI, REGION_ABI, DSL_QSLI_ABI,
}

MSA_INDEX_SCORE_RECIPE = "msa_index_score_arch35"
MSA_INDEX_SCORE_V2_RECIPE = "msa_index_score_v2_arch35"
STEP5_REGION_SCORE_RECIPE = "step5_indexer_region_score_arch35"
LIGHTNING_INDEXER_QSA_RECIPE = "lightning_indexer_qsa_arch35"
SCORER_RECIPES = {
    MSA_ABI: MSA_INDEX_SCORE_RECIPE, MSA_V2_ABI: MSA_INDEX_SCORE_V2_RECIPE,
    REGION_ABI: STEP5_REGION_SCORE_RECIPE, QSA_ABI: LIGHTNING_INDEXER_QSA_RECIPE,
}
_CACHE_ALIASES = {"fp8": "fp8_e4m3", "fp4": "mxfp4", "a4c4": "mxfp4"}
_CACHE_DTYPES = frozenset({"bf16", "fp8_e4m3", "int8", "mxfp4"})
BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
PAIR_BYTES = FP32_BYTES + INT32_BYTES
MX_SCALE_GROUP = 32
MSA_SHORT_BLOCK = 16
MSA_SCORE_STRIDE_ALIGNMENT = 16
MSA_RETURN_INDICES_VALUES = 2
SORT_GROUP = 32
MERGE_FAN_IN = 4
ARCH22_FLUSH_BLOCKS = 4
ARCH22_SMALL_QUERY_MAX = 4
ARCH22_TOPK_LIMIT = 2048
ARCH22_SPLIT_RECORD_BYTES = 128  # Sixteen int64 fields per owner.
QLI_SPLIT_LIST_ALIGNMENT = 16
DEFAULT_S2_TILE = 512
SCORE_SLOTS = 2
SCORE_SLOT_NAME = "index_score"
DEFAULT_INDEX_DIM = 128


def ceil_div(n: int, d: int) -> int:
    return (n + d - 1) // d


def align(n: int, granule: int) -> int:
    return ceil_div(n, granule) * granule


def _integer(name: str, value: int, minimum: int = 1) -> None:
    if type(value) is not int or value < minimum:
        raise ModelingError(f"sparse_indexer_invalid_{name}:{value!r}")


def unsupported(code: str, message: str) -> UnmodeledCapabilityError:
    return UnmodeledCapabilityError(
        capability=FAMILY, entry="sparse_indexer.build",
        code=f"sparse_indexer_{code}", message=message,
    )


@dataclass(frozen=True, slots=True)
class SparseIndexerContract:
    query_tokens: int
    scanned_rows: int
    index_heads: int
    index_dim: int
    block_size: int = 1
    topk_items: int = ARCH22_TOPK_LIMIT
    fused_topk: bool = True
    cache_dtype: str = "bf16"
    has_head_weights: bool = True
    has_mask3_sort: bool = False
    abi: str = LIGHTNING_ABI
    s2_splits: int = 1
    s2_tile_size: int = DEFAULT_S2_TILE
    batch_size: int = 1
    shared_prefix: bool = False
    causal: bool = True
    is_sparse_count_over_2k: bool = False
    return_choice: int | None = None

    def __post_init__(self) -> None:
        if self.abi in MSA_ABIS and self.return_choice is not None:
            object.__setattr__(self, "fused_topk", self.return_choice != 0)

    @property
    def matrix_flops(self) -> int:
        return 2 * self.query_tokens * self.scanned_rows * self.index_heads * self.index_dim

    @property
    def matrix_dtype(self) -> str:
        return self.cache_dtype

    @property
    def cache_row_bytes(self) -> int:
        if self.cache_dtype == "int8" and self.abi in QUANT_ABIS:
            return self.index_dim + BF16_BYTES
        return cache_row_bytes(self.cache_dtype, self.index_dim)

    @property
    def score_items(self) -> int:
        return ceil_div(self.scanned_rows, self.block_size)

    @property
    def arch22_retained_items(self) -> int:
        return self.topk_items if self.is_sparse_count_over_2k else max(self.topk_items, ARCH22_TOPK_LIMIT)

    @property
    def msa_score_heads(self) -> int:
        return 1 if self.block_size == MSA_SHORT_BLOCK else self.index_heads

    @property
    def msa_score_stride(self) -> int:
        return align(max(1, self.score_items), MSA_SCORE_STRIDE_ALIGNMENT)

    @property
    def aiv_scale_bytes(self) -> int:
        if self.abi in QUANT_ABIS and self.cache_dtype in {"fp8_e4m3", "int8"}:
            return self.cache_row_bytes - self.index_dim
        return 0


def cache_row_bytes(cache_dtype: str, dim: int) -> int:
    """Packed E2M1 payload and E8M0 group scales, or the dense row storage."""
    cache_dtype = _CACHE_ALIASES.get(cache_dtype, cache_dtype)
    if cache_dtype == "mxfp4":
        return ceil_div(dim, 2) + ceil_div(dim, MX_SCALE_GROUP)
    if cache_dtype in {"fp8_e4m3", "int8"}:
        return dim + FP32_BYTES  # Per-row quantization scale.
    if cache_dtype == "bf16":
        return dim * BF16_BYTES
    raise unsupported("cache_dtype", f"Unsupported index cache dtype {cache_dtype!r}")


def validate_contract(contract: SparseIndexerContract) -> None:
    for name in ("index_heads", "index_dim", "block_size", "topk_items",
                 "s2_splits", "s2_tile_size", "batch_size"):
        _integer(name, getattr(contract, name))
    _integer("query_tokens", contract.query_tokens, minimum=0)
    _integer("scanned_rows", contract.scanned_rows, minimum=0)
    if contract.abi not in SUPPORTED_ABIS:
        raise unsupported("abi", f"Unsupported sparse indexer ABI {contract.abi!r}")
    if contract.cache_dtype not in _CACHE_DTYPES:
        raise unsupported("cache_dtype", f"Unsupported index cache dtype {contract.cache_dtype!r}")
    if contract.abi in QUANT_ABIS | {LIGHTNING_V2_ABI} and contract.has_head_weights is not True:
        raise ModelingError(f"sparse_indexer_v2_requires_head_weights:{contract.abi}")
    if contract.abi == QSA_ABI and contract.has_head_weights:
        raise ModelingError("sparse_indexer_qsa_is_weight_free")
    choice = contract.return_choice
    if choice is not None and (contract.abi not in MSA_ABIS or type(choice) is not int
                               or not 0 <= choice <= MSA_RETURN_INDICES_VALUES):
        raise ModelingError(f"sparse_indexer_invalid_return_choice:{choice!r}")
    if (contract.abi in MSA_ABIS and contract.block_size == MSA_SHORT_BLOCK
            and contract.fused_topk):
        raise ModelingError("sparse_indexer_msa_block16_disallows_fused_topk")
    if contract.abi == MSA_ABI and contract.fused_topk:
        raise ModelingError("sparse_indexer_msa_v1_returns_scores")


@dataclass(frozen=True, slots=True)
class _S2Partition:
    """Balanced contiguous intervals of whole S2 tiles and one final tail."""

    rows: int
    tile_size: int
    splits: int

    @property
    def tiles(self) -> int:
        return ceil_div(self.rows, self.tile_size)

    @property
    def owners(self) -> int:
        return min(self.splits, self.tiles)

    def owner_rows(self, split: int) -> int:
        base, extra = divmod(self.tiles, self.owners)
        begin = (split * base + min(split, extra)) * self.tile_size
        count = base + (split < extra)
        return min(count * self.tile_size, self.rows - begin)

    def variants(self) -> Iterator[tuple[int, int, bool]]:
        """Distinct owner row counts with multiplicities, independent of S2."""
        if not self.owners:
            return
        base, extra = divmod(self.tiles, self.owners)
        if extra:
            yield (base + 1) * self.tile_size, extra, False
        if self.owners - extra > 1:
            yield base * self.tile_size, self.owners - extra - 1, False
        yield self.owner_rows(self.owners - 1), 1, True


@dataclass(frozen=True, slots=True)
class PhaseWork:
    name: str
    work: Work = field(default_factory=Work)
    aic: bool = False
    metadata: dict[str, int | bool | str] = field(default_factory=dict)


def vector(name: str, ops: int, **metadata: int | bool | str) -> PhaseWork:
    return PhaseWork(name, Work(resources={"vector": ops}), metadata=metadata)


def pair_merge(name: str, incoming: int, retained: int, output: int, **metadata) -> PhaseWork:
    return vector(name, incoming + retained, input_pair_bytes=(incoming + retained) * PAIR_BYTES,
                  output_pair_bytes=output * PAIR_BYTES, **metadata)


def indexer_contract(op: OpSpec, ctx: OperatorPlanningContext, default_abi: str) -> SparseIndexerContract:
    metadata = dict(op.metadata)
    raw = dict(metadata.get("indexer_contract") or metadata.get("scorer_geometry") or {})
    abi = raw.get("abi") or metadata.get("native_kind")
    if abi is not None and ctx.requested_abi is not None and abi != ctx.requested_abi:
        raise unsupported("abi_conflict", f"Requested {ctx.requested_abi!r} for an {abi!r} operator")
    abi = ctx.requested_abi or abi or default_abi
    required = {"query_tokens", "scanned_rows", "index_heads", "index_dim"}
    if not required <= raw.keys():
        raise unsupported("missing_geometry", f"{op.op_id} requires {sorted(required - raw.keys())}")
    raw.setdefault("has_head_weights", abi not in MSA_ABIS | {QSA_ABI})
    dtype = raw.get("cache_dtype", "bf16").lower()
    raw["cache_dtype"] = _CACHE_ALIASES.get(dtype, dtype)
    raw["abi"] = abi
    contract = SparseIndexerContract(**{f.name: raw[f.name] for f in fields(SparseIndexerContract)
                                       if f.name in raw})
    validate_contract(contract)
    return contract


def make_op(
    contract: SparseIndexerContract, *, op_id: str, deps: tuple[str, ...] = (),
    stage: int = 0, layer: int = 0, requests: tuple[str, ...] = (),
    phase: str = "decode", recipe: str | None = None, source: SourceRef | None = None,
) -> OpSpec:
    dtype = _CACHE_ALIASES.get(contract.cache_dtype, contract.cache_dtype)
    if dtype != contract.cache_dtype:
        contract = replace(contract, cache_dtype=dtype)
    validate_contract(contract)
    items = contract.topk_items if contract.fused_topk else contract.score_items
    shape = (contract.query_tokens, items)
    padded_shape = shape
    returns = ()
    meta = {"indexer_contract": asdict(contract), "native_kind": contract.abi}
    if contract.abi in MSA_ABIS:
        shape = (contract.msa_score_heads, *shape)
        padded_shape = shape if contract.fused_topk else (*shape[:-1], contract.msa_score_stride)
        meta["output_shape"] = padded_shape
        if contract.return_choice == MSA_RETURN_INDICES_VALUES:
            returns = ("values",)
            meta.update(values_shape=shape, values_dtype="fp32")
    if source is not None:
        meta["native_source"] = source.to_dict()
        # Existing scorer callers expose this structural record in exports.
        meta["scorer_geometry"] = meta["indexer_contract"].copy()
    return OpSpec(
        op_id=op_id, deps=deps, family="indexer", recipe=recipe or FAMILY,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=padded_shape, launch_shape=padded_shape,
        dtype="int32" if contract.fused_topk else "fp32", flops=contract.matrix_flops,
        matrix_flops=((contract.matrix_dtype, contract.matrix_flops),),
        source_boundary="kernel", returns=returns, metadata=tuple(sorted(meta.items())),
    )


TileSelection = Callable[[SparseIndexerContract, int, int, int, int], tuple[PhaseWork, ...]]
OwnerSelection = Callable[[SparseIndexerContract, int], tuple[PhaseWork, ...]]


@dataclass(frozen=True, slots=True)
class Lowering:
    implementation: str
    tiling: str
    tile_size: int
    score_path: str
    tile_selection: TileSelection | None = None
    owner_selection: OwnerSelection | None = None
    flush_period: int = 1
    score_align: int = 1
    slots: int = 1
    gm_score_row: bool = False
    dual_destination: bool = False
    score_row_item_bytes: int = FP32_BYTES
    owner_score_row: bool = False
    handoff_item_bytes: int = FP32_BYTES
    head_weight_item_bytes: int = BF16_BYTES
    handoff_alignment: int = 1
    handoff_columns: Callable[[int], int] | None = None
    score_slot_bytes: int = 0
    score_row_read_align: int = 1

    def handoff_bytes(self, rows: int, heads: int) -> int:
        destinations = 2 if self.dual_destination else 1
        # A QLI tile has one live query and pads M to two query groups. Each
        # destination receives a full group, including the padded one.
        columns = self.handoff_columns(rows) if self.handoff_columns else align(rows, self.score_align)
        destination_bytes = columns * heads * self.handoff_item_bytes
        return destinations * align(destination_bytes, self.handoff_alignment)


def owner_pairs(ctx: OperatorPlanningContext, aivs_per_aic: int) -> tuple[tuple[int, tuple[int, ...]], ...]:
    owners = ctx.stream_owners
    aics = (owners.aic or owners.pool_aic) if owners else ()
    aivs = (owners.aiv or owners.pool_aiv) if owners else ()
    active_aic = frozenset((aics or range(ctx.datapath.aic_count))[:ctx.effective_aic_cores])
    active_aiv = frozenset((aivs or range(ctx.datapath.aiv_count))[:ctx.effective_aiv_cores])
    return tuple((aic, siblings[:aivs_per_aic]) for aic, paired in ctx.datapath.aic_aiv_pairs
                 if aic in active_aic
                 and len(siblings := tuple(aiv for aiv in paired if aiv in active_aiv)) >= aivs_per_aic)


@dataclass(frozen=True, slots=True)
class _DisjointOwnerProgram:
    """Execute the stages with the physical IDs already bound by the planner."""

    placed: ParallelProgram

    def start(self, engine, name, ready, *, owner=""):
        return self.placed.start(engine, name, ready, owner=owner)


@dataclass(frozen=True, slots=True)
class _SparseIndexerProgramBuilder:
    contract: SparseIndexerContract
    lowering: Lowering
    owners: tuple[tuple[int, tuple[int, ...]], ...]
    l2_reuse: bool
    aic_read: str
    aiv_read: str
    aic_count: int
    aiv_count: int

    @property
    def partition(self) -> _S2Partition:
        return _S2Partition(self.contract.scanned_rows, self.lowering.tile_size,
                            self.contract.s2_splits)

    def owner_entry(self) -> tuple[PhaseWork, ...]:
        c = self.contract
        scales = c.index_heads * c.aiv_scale_bytes
        query = c.index_heads * c.cache_row_bytes - scales
        weights = c.index_heads * self.lowering.head_weight_item_bytes if c.has_head_weights else 0
        stages = [PhaseWork("query_load", Work(resources={"hbm": query, self.aic_read: query}), aic=True)]
        if scales:
            stages.append(PhaseWork("query_scale_load", Work(resources={"hbm": scales, self.aiv_read: scales})))
        if weights and self.lowering.score_path != "l0c->gm->ub":
            stages.append(PhaseWork("head_weight_load", Work(resources={"hbm": weights, self.aiv_read: weights}),
                                    metadata={"head_weight_bytes": weights}))
            if scales:
                stages.append(vector("prescale_head_weights", c.index_heads))
        return tuple(stages)

    def tile(self, rows: int, index: int, *, reuse: bool, final_owner: bool) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        valid = min(g.tile_size, rows - index * g.tile_size)
        scores = valid * c.index_heads
        score_bytes = g.handoff_bytes(valid, c.index_heads)
        operand_bytes = c.cache_row_bytes - c.aiv_scale_bytes
        key = valid * operand_bytes
        memory = "l2" if reuse else "hbm"
        score_to_gm = g.score_path == "l0c->gm->ub"
        stages = [PhaseWork("key_load", Work(resources={memory: key, self.aic_read: key}), True)]
        if (scales := valid * c.aiv_scale_bytes):
            stages.append(PhaseWork("key_scale_load", Work(resources={memory: scales, self.aiv_read: scales})))
        stages.extend((
            PhaseWork("load_operands", Work(resources={"mte1": key + c.index_heads * operand_bytes}), True),
            PhaseWork("cube_scores", Work(matrix={c.matrix_dtype: 2 * scores * c.index_dim}), True),
            PhaseWork("score_handoff", Work(resources={
                "fixpipe": score_bytes, **({"hbm": score_bytes} if score_to_gm else {}),
            }), True, {"route": g.score_path, "score_bytes": score_bytes}),
        ))
        if score_to_gm:
            weights = c.index_heads * BF16_BYTES if c.has_head_weights else 0
            stages.append(PhaseWork("score_read", Work(resources={
                "hbm": score_bytes + weights, self.aiv_read: score_bytes + weights,
            }), metadata={"head_weight_bytes": weights}))
        msa = c.abi in MSA_ABIS
        score_heads = c.msa_score_heads if msa else 1
        if not msa:
            stages.append(vector("rectification", scores))
            if c.has_head_weights:
                stages.append(vector("weighting", scores))
            elif scales:
                stages.append(vector("query_descale", scores))
            stages.append(vector("head_reduction", scores - valid))
            if scales:
                stages.append(vector("key_descale", valid))
        elif score_heads < c.index_heads:
            stages.append(vector("head_maxima", scores - valid))
        candidates = ceil_div(valid, c.block_size)
        if c.block_size > 1:
            stages.append(vector("block_maxima", (valid - candidates) * score_heads))
        if c.has_mask3_sort:
            stages.append(vector("mask3", candidates))
        if msa:
            final_tile = final_owner and (index + 1) * g.tile_size >= rows
            columns = candidates + (c.msa_score_stride - c.score_items if final_tile else 0)
            stages.append(self._msa_score_store(candidates, columns))
        elif g.gm_score_row or g.owner_score_row or not c.fused_topk:
            score_row = candidates * (g.score_row_item_bytes if c.fused_topk else FP32_BYTES)
            stages.append(PhaseWork("score_row_store", Work(resources={"hbm": score_row, "mte3": score_row})))
        elif g.tile_selection is not None:
            stages.extend(g.tile_selection(c, rows, index, valid, g.tile_size))
        return tuple(stages)

    def _msa_score_store(self, valid: int, columns: int) -> PhaseWork:
        c = self.contract
        size = c.msa_score_heads * columns * FP32_BYTES
        return PhaseWork("score_row_store", Work(resources={"hbm": size, "mte3": size}),
                         metadata={"heads": c.msa_score_heads, "valid_items_per_head": valid,
                                   "score_stride": c.msa_score_stride})

    def owner_exit(self, rows: int) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        if not c.fused_topk or g.gm_score_row:
            return ()
        items = ceil_div(rows, c.block_size)
        stages = []
        if g.owner_score_row and (items >= c.topk_items or self.partition.owners > 1):
            size = items * g.score_row_item_bytes
            stages.append(PhaseWork("local_score_row_load", Work(resources={"hbm": size, self.aiv_read: size})))
        if g.owner_selection is not None:
            stages.extend(g.owner_selection(c, items))
        if self.partition.owners > 1:
            count, pairs, records = self._split_storage(rows)
            stages.append(PhaseWork("split_list_store", Work(resources={
                "hbm": pairs + records, "mte3": pairs + records,
            }), metadata={"pairs": count, "pair_bytes": pairs, "record_bytes": records}))
        return tuple(stages)

    def _split_storage(self, rows: int) -> tuple[int, int, int]:
        c = self.contract
        g = self.lowering
        if g.score_path == "l0c->gm->ub":
            count = c.arch22_retained_items
            return count, count * PAIR_BYTES, ARCH22_SPLIT_RECORD_BYTES
        if g.owner_score_row:
            count = align(c.topk_items, QLI_SPLIT_LIST_ALIGNMENT)
            return count, count * (g.score_row_item_bytes + INT32_BYTES), 0
        count = min(c.topk_items, ceil_div(rows, c.block_size))
        return count, count * PAIR_BYTES, 0

    def _split_reduction(self) -> tuple[PhaseWork, ...]:
        pairs = pair_bytes = record_bytes = 0
        for rows, owners, _ in self.partition.variants():
            count, payload, record = self._split_storage(rows)
            pairs += owners * count
            pair_bytes += owners * payload
            record_bytes += owners * record
        size = pair_bytes + record_bytes
        stages = [PhaseWork("split_list_load", Work(resources={"hbm": size, self.aiv_read: size}),
                            metadata={"pair_bytes": pair_bytes, "record_bytes": record_bytes})]
        merges = self.partition.owners - 1
        if self.lowering.score_path != "l0c->gm->ub":
            return (*stages, vector("split_topk_merge", pairs, merges=merges))
        groups, tail = divmod(merges, MERGE_FAN_IN - 1)
        entries = groups * MERGE_FAN_IN + (tail + 1 if tail else 0)
        retained = self.contract.arch22_retained_items
        passes = groups + bool(tail)
        return (*stages,
                vector("split_topk_merge", entries * retained, merges=merges, mrg_passes=passes,
                       input_pair_bytes=entries * retained * PAIR_BYTES),
                vector("split_topk_copyback", passes * retained, vector_barriers=2 * passes,
                       pair_bytes=passes * retained * PAIR_BYTES))

    def query_exit(self) -> tuple[PhaseWork, ...]:
        c = self.contract
        g = self.lowering
        msa = c.abi in MSA_ABIS
        stages = []
        if msa and not c.score_items:
            stages.append(self._msa_score_store(0, c.msa_score_stride))
        if not c.fused_topk:
            return tuple(stages)
        score_heads = c.msa_score_heads if msa else 1
        if g.gm_score_row and c.score_items:
            size = align(c.score_items, g.score_row_read_align) * score_heads * FP32_BYTES
            stages.append(PhaseWork("score_row_load", Work(resources={"hbm": size, self.aiv_read: size})))
            if g.owner_selection is not None:
                stages.extend(g.owner_selection(c, c.score_items))
        elif self.partition.owners > 1:
            stages.extend(self._split_reduction())
        if c.has_mask3_sort:
            stages.append(vector("index_sort", sort_ops(c.topk_items)))
        output_items = score_heads * c.topk_items
        size = output_items * INT32_BYTES
        stages.extend((
            vector("index_extract", output_items, valid_items_per_head=min(c.score_items, c.topk_items),
                   pair_bytes=score_heads * min(c.score_items, c.topk_items)
                   * (g.score_row_item_bytes + INT32_BYTES)),
            PhaseWork("index_store", Work(resources={"hbm": size, "mte3": size})),
        ))
        if c.return_choice == MSA_RETURN_INDICES_VALUES:
            size = output_items * FP32_BYTES
            stages.append(PhaseWork("values_store", Work(resources={"hbm": size, "mte3": size})))
        return tuple(stages)

    def _tile_variants(self, rows: int) -> Iterator[tuple[int, int]]:
        # Interior tiles differ only at the periodic flush. Keep the leading
        # period (first retained list) and the final tile as separate classes.
        period = self.lowering.flush_period
        tiles = ceil_div(rows, self.lowering.tile_size)
        first = min(tiles - 1, period)
        for index in range(first):
            yield index, 1
        interior = tiles - 1 - first
        if interior > 0:
            flushes, ordinary = divmod(interior, period)
            if period > 1:
                ordinary += flushes * (period - 1)
                if ordinary:
                    yield first, ordinary
            if flushes:
                yield first + period - 1, flushes
        yield tiles - 1, 1

    def phase_totals(self) -> dict[str, Work]:
        entry, tiles, owner_exit, query_exit = {}, {}, {}, {}

        def add(totals: dict[str, Work], phases: tuple[PhaseWork, ...], count: int) -> None:
            if count:
                for phase in phases:
                    totals[phase.name] = totals.get(phase.name, Work()) + phase.work.scaled(count)

        c = self.contract
        fresh_queries = 1 if self.l2_reuse else c.query_tokens
        cached_queries = c.query_tokens - fresh_queries
        for rows, owners, final_owner in self.partition.variants():
            add(entry, self.owner_entry(), c.query_tokens * owners)
            for index, count in self._tile_variants(rows):
                add(tiles, self.tile(rows, index, reuse=False, final_owner=final_owner),
                    fresh_queries * owners * count)
                if cached_queries:
                    add(tiles, self.tile(rows, index, reuse=True, final_owner=final_owner),
                        cached_queries * owners * count)
            add(owner_exit, self.owner_exit(rows), c.query_tokens * owners)
        add(query_exit, self.query_exit(), c.query_tokens)
        return {**entry, **tiles, **owner_exit, **query_exit}

    def _placement(self, ctx: RecipeContext, owner, phase: PhaseWork):
        aic, aivs = owner
        if phase.aic:
            return ctx.owned(aic, self.aic_count, (*phase.work.resources, "cube"))
        placement = {}
        for aiv in aivs:
            for name, units in ctx.owned(aiv, self.aiv_count, tuple(phase.work.resources)).items():
                placement[name] = tuple(dict.fromkeys((*placement.get(name, ()), *units)))
        return placement

    def _stages(self, ctx: RecipeContext, owner, phases: tuple[PhaseWork, ...]) -> tuple[Stage, ...]:
        stages = []
        for phase in phases:
            stages.append(Stage(
                phase.name, ctx.bind(phase.work),
                after=(stages[-1].name,) if stages else (),
                placement=self._placement(ctx, owner, phase), metadata=phase.metadata,
            ))
        return tuple(stages)

    def _tile_stages(self, ctx: RecipeContext, owner, rows: int, index: int,
                     reuse: bool, ring: RingBuffer | None, final_owner: bool) -> tuple[Stage, ...]:
        stages = self._stages(ctx, owner, self.tile(rows, index, reuse=reuse, final_owner=final_owner))
        if ring is None:
            return stages
        slot = ring.slot(index)
        release = f"score_release_{slot}"
        dual_dst_ctl = int(self.lowering.handoff_item_bytes == FP32_BYTES)
        metadata = {"score_slot": slot, "score_layout": "row_major",
                    "dual_dst_ctl": dual_dst_ctl, "dual_destination": bool(dual_dst_ctl)}
        bound = []
        alloc = ()
        for stage in stages:
            if stage.name == "cube_scores":
                stage = replace(stage, previous=("score_handoff",))
            elif stage.name == "score_handoff":
                # Credit is returned only after all paired AIVs finish reading
                # the score. The tile window and explicit per-slot edge apply
                # even when two tail tiles retire in a different order.
                slot_bytes = self.lowering.score_slot_bytes
                alloc = tuple(
                    ctx.allocate(f"{ring.allocation}_{slot}_aiv{aiv}", "ub", slot_bytes, (aiv,))
                    for aiv in owner[1]
                ) if "ub" in ctx.pool_names else ()
                stage = replace(stage, previous=(release,), acquire=alloc,
                                metadata={**stage.metadata, **metadata})
                bound.extend((
                    stage,
                    Stage("score_publish", after=(stage.name,), metadata={"score_slot": slot}),
                ))
                continue
            elif stage.name == "rectification":
                stage = replace(stage, after=("score_publish",))
            elif stage.name == "head_reduction":
                bound.append(stage)
                bound.append(Stage(release, after=(stage.name,), free=alloc,
                                   keep_empty=True, metadata={"score_slot": slot}))
                continue
            if bound and bound[-1].name == release:
                stage = replace(stage, after=(release,))
            bound.append(stage)
        return tuple(bound)

    def _owner_program(self, ctx: RecipeContext, owner, split: int, query: int) -> KernelProgram:
        rows = self.partition.owner_rows(split)
        g = self.lowering
        ring = RingBuffer(depth=SCORE_SLOTS, allocation=SCORE_SLOT_NAME) if g.dual_destination else None
        reuse = self.l2_reuse and query > 0
        entry = self._stages(ctx, owner, self.owner_entry())
        tiles = PipelineProgram(
            ceil_div(rows, g.tile_size), ring.depth if ring is not None else g.slots,
            lambda index: self._tile_stages(ctx, owner, rows, index, reuse, ring,
                                            split == self.partition.owners - 1),
        )
        body = Stage("score_tiles", after=(entry[-1].name,), program=tiles)
        end = self._stages(ctx, owner, self.owner_exit(rows))
        if end:
            end = (replace(end[0], after=(body.name,)), *end[1:])
        return KernelProgram((*entry, body, *end))

    def _query_program(self, ctx: RecipeContext, owner_index: int, query: int) -> tuple[Stage, ...]:
        split_count = self.partition.owners
        concurrent = min(split_count, len(self.owners))
        streams = []
        for lane in range(concurrent):
            owner = self.owners[(owner_index + lane) % len(self.owners)]
            streams.append(PipelineProgram(
                ceil_div(split_count - lane, concurrent), 1,
                lambda index, lane=lane, owner=owner: (
                    Stage("s2_owner", program=self._owner_program(ctx, owner, index * concurrent + lane, query)),
                ),
            ))
        exit_stages = self._stages(ctx, self.owners[owner_index], self.query_exit())
        if not streams:
            return exit_stages
        body = Stage("s2_owners", program=ParallelProgram(tuple(streams)))
        if exit_stages:
            exit_stages = (replace(exit_stages[0], after=(body.name,)), *exit_stages[1:])
        return (body, *exit_stages)

    def __call__(self, ctx: RecipeContext) -> ParallelProgram | _DisjointOwnerProgram:
        queries = self.contract.query_tokens
        # Shared-prefix reuse is admitted only if the complete key stream
        # fits, with the first query establishing residency before reuse.
        width = 1 if self.l2_reuse else max(1, len(self.owners) // max(1, self.partition.owners))
        count = min(queries, width)
        placed = ParallelProgram(tuple(
            PipelineProgram(
                ceil_div(queries - lane, count), 1,
                lambda index, lane=lane: self._query_program(
                    ctx, lane * max(1, self.partition.owners), index * count + lane),
            )
            for lane in range(count)
        ))
        # The generic stream binder interprets placement units as lane-local
        # ordinals. These stages already target the lane's physical core IDs.
        if ctx.stream_owners is not None and ctx.stream_owners.isolation == "disjoint":
            return _DisjointOwnerProgram(placed)
        return placed


def sort_ops(items: int) -> int:
    """Pair compare/exchange work for a comparison-based sort."""
    levels = (max(1, items) - 1).bit_length()
    return items * levels


def build_plan(
    op: OpSpec, ctx: OperatorPlanningContext, contract: SparseIndexerContract,
    lowering: Lowering, owners: tuple[tuple[int, tuple[int, ...]], ...],
    sources: tuple[SourceRef, ...],
) -> OperatorPlan:
    if not owners:
        raise unsupported("missing_owners", "Sparse index scoring needs reachable AIC and AIV owners")
    geometry = ctx.geometry
    hierarchy = getattr(geometry.physical_mechanisms, "memory_hierarchy", None)
    l2_capacity = getattr(hierarchy, "effective_l2_bytes", geometry.buffers.get("l2", 0))
    l2_reuse = (
        contract.shared_prefix and contract.query_tokens > 1 and contract.scanned_rows > 0
        and "l2" in geometry.resource_units
        and contract.scanned_rows * contract.cache_row_bytes <= l2_capacity
    )
    builder = _SparseIndexerProgramBuilder(
        contract, lowering, owners, l2_reuse, read_role(geometry.resource_units),
        read_role(geometry.resource_units, aiv=0), ctx.datapath.aic_count, ctx.datapath.aiv_count,
    )
    phases = []
    total = Work()
    for name, work in builder.phase_totals().items():
        total += work
        items = tuple(WorkItem("cube", "flop", n, dtype) for dtype, n in work.matrix.items() if n)
        items += tuple(WorkItem(role, "op" if role == "vector" else "bytes", n)
                       for role, n in work.resources.items() if n)
        phases.append(PipelinePhase(name, (phases[-1].phase_id,) if phases else (), items))
    return OperatorPlan(
        work=total, program=builder,
        metadata={
            "implementation": lowering.implementation, "abi": contract.abi,
            "arch_id": ctx.compute_arch.arch_id, "tiling": lowering.tiling,
            "s2_tile_size": lowering.tile_size, "score_path": lowering.score_path,
            "fused_topk": contract.fused_topk, "cache_row_bytes": contract.cache_row_bytes,
            "s2_splits": builder.partition.owners, "causal": contract.causal,
            **({"score_stride": contract.msa_score_stride,
                "score_workspace_bytes": contract.msa_score_heads * contract.query_tokens
                * contract.msa_score_stride * FP32_BYTES} if contract.abi in MSA_ABIS
                and contract.fused_topk else {}),
        },
        pipeline_spec=FusedPipelineSpec(FAMILY, lowering.implementation, tuple(phases), sources),
    )


def describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )
