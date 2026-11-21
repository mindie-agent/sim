"""Public entry point for the governed sparse indexer operator family.

[Role] Resolve explicit architecture, caller ABI and score contract into one
  plan. Keep legacy scorer recipes and constructors on the same planning path.
[Invariants] Only this package selects its ISA strategies. ABI-specific
  work follows the tensor and route declarations of the selected strategy.
[Timeouts & Locks] Bounded dispatch and planning without I/O or locks.
[Side-Effect Policy] Returns plans and registry records without global writes.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec

from ...mechanism import Geometry, OperatorPlan, PlannedRecipe
from ...results import OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from . import arch22 as _arch22, arch35 as _arch35
from .common import (
    DEFAULT_INDEX_DIM, FAMILY, LIGHTNING_INDEXER_QSA_RECIPE, MSA_ABI,
    MSA_INDEX_SCORE_RECIPE, MSA_INDEX_SCORE_V2_RECIPE, MSA_V2_ABI, QSA_ABI,
    REGION_ABI, SCORER_RECIPES, STEP5_REGION_SCORE_RECIPE,
    SparseIndexerContract, ceil_div, describe, make_op, unsupported,
)

DEFAULT_MSA_HEADS = 4
DEFAULT_MSA_BLOCK = 128
DEFAULT_MSA_TOPK_BLOCKS = 17
DEFAULT_REGION_HEADS = 16
DEFAULT_REGION_BLOCK = 8
DEFAULT_REGION_TOPK = 256
DEFAULT_QSA_HEADS = 32
DEFAULT_QSA_RATIO = 4
DEFAULT_QSA_TOPK = 512

_STRATEGIES = {_arch22.ARCH_ID: _arch22.STRATEGY, _arch35.ARCH_ID: _arch35.STRATEGY}
SortAllTrace = _arch22.SortAllTrace
FlushMergeTrace = _arch22.FlushMergeTrace
trace_sort_all = _arch22.trace_sort_all
trace_arch22_topk_flushes = _arch22.trace_arch22_topk_flushes


def build(op: OpSpec, geometry_or_ctx: Geometry | OperatorPlanningContext) -> OperatorPlan:
    """Build from the governed context or the legacy recipe's physical geometry."""
    if isinstance(geometry_or_ctx, Geometry):
        geometry = geometry_or_ctx
        if geometry.datapath is None:
            raise unsupported("missing_datapath", f"{op.op_id} requires a declared indexer datapath")
        ctx = OperatorPlanningContext(
            occurrence=OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
            geometry=geometry, datapath=geometry.datapath, stream_owners=geometry.stream_owners,
        )
    else:
        ctx = geometry_or_ctx
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise unsupported("architecture", f"No sparse indexer lowering for {ctx.compute_arch.arch_id}")
    return strategy.build(op, ctx)


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    plan = build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(
        occurrence=ctx.occurrence, op=op, recipe=plan, implementation=descriptor.implementation,
        sources=descriptor.sources, descriptor=descriptor,
    )


def register_family() -> OperatorFamilySpec:
    return OperatorFamilySpec(
        name=FAMILY, families=frozenset({FAMILY, "indexer"}),
        owned_recipe_ids=frozenset({FAMILY, *SCORER_RECIPES.values()}),
        resolve=resolve_operator_plan,
    )


def recipes() -> dict[str, PlannedRecipe]:
    """Expose standalone and existing scorer IDs through the governed builder."""
    planner = PlannedRecipe(build)
    return dict.fromkeys((FAMILY, *SCORER_RECIPES.values()), planner)


def make_indexer_op(
    contract: SparseIndexerContract | None = None, *, op_id: str = FAMILY,
    deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", **geometry,
) -> OpSpec:
    """Make an operator from a frozen contract or its named geometry fields."""
    if contract is None:
        if geometry.get("abi") in {MSA_ABI, MSA_V2_ABI, QSA_ABI}:
            geometry.setdefault("has_head_weights", False)
        if geometry.get("abi") == MSA_ABI:
            geometry.setdefault("fused_topk", False)
        contract = SparseIndexerContract(**geometry)
    elif geometry:
        raise TypeError("sparse_indexer_contract_and_geometry_are_mutually_exclusive")
    return make_op(contract, op_id=op_id, deps=deps, stage=stage, layer=layer,
                   requests=requests, phase=phase)


def _scorer_op(contract: SparseIndexerContract, source, variant: str, **location) -> OpSpec:
    op = make_op(contract, recipe=SCORER_RECIPES[contract.abi], source=source, **location)
    metadata = dict(op.metadata)
    metadata["scorer_geometry"]["variant"] = variant
    return replace(op, metadata=tuple(sorted(metadata.items())))


def _batch(batch_size: int | None, requests: tuple[str, ...], queries: int) -> int:
    return max(1, len(requests) or queries) if batch_size is None else batch_size


def make_msa_index_score_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_MSA_HEADS,
    index_dim: int = DEFAULT_INDEX_DIM, block_size: int = DEFAULT_MSA_BLOCK,
    topk_blocks: int = DEFAULT_MSA_TOPK_BLOCKS, return_choice: int = 1,
    cache_dtype: str = "bf16", batch_size: int | None = None, shared_prefix: bool = False,
) -> OpSpec:
    fused = return_choice != 0
    contract = SparseIndexerContract(
        query_tokens, context_tokens, index_heads, index_dim,
        block_size=block_size, topk_items=topk_blocks, fused_topk=fused, cache_dtype=cache_dtype,
        has_head_weights=False, abi=MSA_V2_ABI if fused else MSA_ABI,
        batch_size=_batch(batch_size, requests, query_tokens), shared_prefix=shared_prefix,
        return_choice=return_choice,
    )
    return _scorer_op(
        contract, _arch35.MSA_TOPK_SOURCE if fused else _arch35.MSA_SOURCE,
        "msa_v2" if fused else "msa_v1", op_id=op_id, deps=deps, stage=stage,
        layer=layer, requests=requests, phase=phase,
    )


def make_step5_region_score_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_REGION_HEADS,
    index_dim: int = DEFAULT_INDEX_DIM, region_block_size: int = DEFAULT_REGION_BLOCK,
    topk_regions: int = DEFAULT_REGION_TOPK, batch_size: int | None = None,
    shared_prefix: bool = False,
) -> OpSpec:
    if type(region_block_size) is not int or region_block_size <= 0:
        raise ModelingError("sparse_indexer_invalid_region_block_size")
    contract = SparseIndexerContract(
        query_tokens, ceil_div(context_tokens, region_block_size), index_heads, index_dim,
        topk_items=topk_regions, cache_dtype="fp8_e4m3", has_mask3_sort=True,
        abi=REGION_ABI, batch_size=_batch(batch_size, requests, query_tokens), shared_prefix=shared_prefix,
    )
    return _scorer_op(
        contract, _arch35.REGION_SOURCE, "step5_region_score",
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
    )


def make_lightning_indexer_qsa_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_QSA_HEADS, index_dim: int = DEFAULT_INDEX_DIM,
    compress_ratio: int = DEFAULT_QSA_RATIO, block_topk: int = DEFAULT_QSA_TOPK, cache_dtype: str = "bf16",
    batch_size: int | None = None, shared_prefix: bool = False,
) -> OpSpec:
    if type(compress_ratio) is not int or compress_ratio <= 0:
        raise ModelingError("sparse_indexer_invalid_compress_ratio")
    contract = SparseIndexerContract(
        query_tokens, ceil_div(context_tokens, compress_ratio), index_heads, index_dim,
        topk_items=block_topk, cache_dtype=cache_dtype, has_head_weights=False, abi=QSA_ABI,
        batch_size=_batch(batch_size, requests, query_tokens), shared_prefix=shared_prefix,
    )
    return _scorer_op(
        contract, _arch35.QSA_SOURCE, "qsa_lightning_indexer",
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests, phase=phase,
    )


__all__ = [
    "build", "resolve_operator_plan", "register_family", "SparseIndexerContract",
    "SortAllTrace", "FlushMergeTrace", "trace_sort_all", "trace_arch22_topk_flushes",
    "make_msa_index_score_op", "make_step5_region_score_op", "make_lightning_indexer_qsa_op",
    "make_indexer_op", "recipes", "MSA_INDEX_SCORE_RECIPE", "MSA_INDEX_SCORE_V2_RECIPE",
    "STEP5_REGION_SCORE_RECIPE", "LIGHTNING_INDEXER_QSA_RECIPE",
]
