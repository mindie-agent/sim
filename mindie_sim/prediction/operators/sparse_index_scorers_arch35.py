"""CANN 9.3 Arch35 sparse index scorer physical recipes.

[Role]
  Model ``MsaIndexScore`` (``return_choice=0``), ``#22235`` ``MsaIndexScoreV2``
  (``return_choice=1`` in-kernel TopK + ``#22283`` ``MXFP4`` ``A4C4``), Step-5
  ``_C_ascend.npu_indexer_region_score`` (FP8 ``ssmax`` summary score + ``mask3``
  + TopK + sort), and ``#21771``/``#22365`` weight-free ``LightningIndexerQSA``.

[Invariants]
  * ``OperatorPlan.work`` equals the exact sum of issued per-owner ``Stage``
    demands across ``hbm``, ``l2``, ``mte2_aic``, ``mte2_aiv``, ``mte1``,
    ``cube``, ``vector``, ``fixpipe``, and ``mte3``.
  * Shared-prefix index cache reads hit L2 when capacity allows; ``return_choice=1``
    eliminates intermediate ``score_gm`` HBM writeback.
  * Zero model-name branches or ``ResourceEnvelope`` fallbacks participate.

[Timeouts & Locks]
  O(compute_units) plan construction; bounded finite stages, no I/O or locks.

[Side-Effect Policy]
  Pure immutable ``OpSpec`` and ``OperatorPlan`` construction.
"""
from __future__ import annotations

import math
from typing import Any

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..mechanism import Geometry, OperatorPlan, PlannedRecipe, Work
from ..ownership import read_role, unit_group
from ..program import KernelProgram, ParallelProgram, Stage

MSA_INDEX_SCORE_RECIPE = "msa_index_score_arch35"
MSA_INDEX_SCORE_V2_RECIPE = "msa_index_score_v2_arch35"
STEP5_REGION_SCORE_RECIPE = "step5_indexer_region_score_arch35"
LIGHTNING_INDEXER_QSA_RECIPE = "lightning_indexer_qsa_arch35"

MSA_SCORE_SOURCE = {
    "repository": "cann_ops_transformer", "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/attention/msa_index_score/msa_index_score.py",
    "source_sha256": "7965e765336d40f03cef55f90383626162366f5dd122e53d4a755081cca1e59f",
}
MSA_SCORE_V2_SOURCE = {
    "repository": "cann_ops_transformer", "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/attention/msa_index_score/msa_index_score_v2.py",
    "source_sha256": "a789ecff0bc0cbf2ca97a075f248d434e702aab3578a3a9c2cc4adef90dc2099",
}
STEP5_REGION_SOURCE = {
    "repository": "vllm-project/vllm-ascend",
    "revision": "405617bae227fe5b097ff163efaa60bb7f22c698",
    "relative_path": "vllm_ascend/models/step5/sparse_attention.py",
    "source_sha256": "3fabfbd4f115f296472d9c567fec4d243587611dc503cdad2e32b819f4aef4a6",
}
LIGHTNING_QSA_SOURCE = {
    "repository": "cann_ops_transformer", "revision": "1.0.0",
    "relative_path": "cann_ops_transformer/ops/attention/lightning_indexer_v2/lightning_indexer.py",
    "source_sha256": "49c07d246525b69c286a99c6a431749db2bf29d88de9a8ba8e555bfc15181cbd",
}

BF16_BYTES = 2
FP32_BYTES = 4
INT32_BYTES = 4
MXFP4_GROUP = 32
DEFAULT_INDEX_DIM = 128
DEFAULT_MSA_HEADS = 4
DEFAULT_MSA_BLOCK = 128
DEFAULT_MSA_TOPK_BLOCKS = 17
DEFAULT_STEP5_HEADS = 16
DEFAULT_STEP5_BLOCK = 8
DEFAULT_STEP5_TOPK_REGIONS = 256
DEFAULT_QSA_HEADS = 32
DEFAULT_QSA_RATIO = 4
DEFAULT_QSA_BLOCK_TOPK = 512
DEFAULT_FALLBACK_TOPK = 16
MAX_REDUCE_OPS = 2
TOPK_SORT_FACTOR = 3
MASK3_OPS_PER_REGION = 4
L2_EFFECTIVE_SHARE = 0.75

SCORER_SETUP_MS = 0.0055
SCORER_BATCH_TOPK_MS = 0.0085
SCORER_LARGE_SCAN_ROWS = 16384
STEP5_TOPK_SORT_MIN_ITEMS = 128
STEP5_TOPK_SORT_BASE_MS = 0.0031
SCORER_SINGLE_QUERY_DRAM_EFF = 0.48
SCORER_MULTI_QUERY_DRAM_EFF = 0.66
STEP5_REGION_REDUCE_MS_PER_ITEM = 0.000000145
STEP5_SINGLE_QUERY_FLOOR_MS = 0.00000048
SCORER_KV_TILE_ROWS = 128

SCORER_BUFFERS = {
    "ub": 96 * 1024, "l1": 128 * 1024, "l0a": 16 * 1024,
    "l0b": 16 * 1024, "l0c": 32 * 1024,
}


def _check_buffers(geometry: Geometry) -> dict[str, int]:
    available = dict(geometry.buffers or {})
    for pool, required in SCORER_BUFFERS.items():
        cap = int(available.get(pool, 0))
        if cap > 0 and required > cap:
            raise ModelingError(f"sparse_index_scorer_buffer_capacity_exceeded:{pool}")
    return {p: sz for p, sz in SCORER_BUFFERS.items() if available.get(p, 0) > 0}


def _effective_l2(geometry: Geometry) -> float:
    mechanisms = getattr(geometry, "physical_mechanisms", None)
    hierarchy = getattr(mechanisms, "memory_hierarchy", None)
    if hierarchy is not None:
        return float(hierarchy.effective_l2_bytes)
    return L2_EFFECTIVE_SHARE * float((geometry.buffers or {}).get("l2", 0))


def _cache_row_bytes(cache_dtype: str, dim: int) -> tuple[float, str]:
    tag = str(cache_dtype or "bf16").lower()
    if tag in {"mxfp4", "fp4", "a4c4"}:
        return float(math.ceil(dim / 2) + math.ceil(dim / MXFP4_GROUP)), "fp8_e4m3"
    if tag in {"fp8_e4m3", "fp8", "int8"}:
        return float(dim + FP32_BYTES), "fp8_e4m3"
    return float(dim * BF16_BYTES), "bf16"


def _make_scorer_op(
    *,
    recipe: str,
    native_kind: str,
    variant: str,
    src: dict[str, str],
    op_id: str,
    deps: tuple[str, ...],
    stage: int,
    layer: int,
    requests: tuple[str, ...],
    phase: str,
    q_tok: int,
    scanned: int,
    heads: int,
    dim: int,
    blk: int,
    tk_items: int,
    fused_topk: bool,
    cache_dtype: str,
    has_weights: bool,
    has_mask3: bool,
    batch_size: int,
    shared_prefix: bool,
) -> OpSpec:
    _, comp_dt = _cache_row_bytes(cache_dtype, dim)
    flops = float(2 * q_tok * heads * scanned * dim)
    num_blocks = max(1, math.ceil(scanned / blk))
    shape = (q_tok, tk_items) if fused_topk else (q_tok, num_blocks)
    meta = {
        "native_kind": native_kind,
        "scorer_geometry": {
            "variant": variant, "query_tokens": q_tok, "scanned_rows": scanned,
            "index_heads": heads, "index_dim": dim, "block_size": blk,
            "topk_items": tk_items, "fused_topk": fused_topk,
            "cache_dtype": cache_dtype, "has_head_weights": has_weights,
            "has_mask3_sort": has_mask3,
            "batch_size": batch_size, "shared_prefix": shared_prefix,
        },
        "native_source": src,
    }
    return OpSpec(
        op_id=op_id, deps=deps, family="indexer", recipe=recipe,
        phase=phase, stage=stage, layer=layer, requests=requests,
        semantic_shape=shape, local_shape=shape, padded_shape=shape, launch_shape=shape,
        dtype="int32" if fused_topk else "fp32", flops=flops,
        matrix_flops=((comp_dt, flops),), source_boundary="kernel",
        metadata=tuple(sorted(meta.items())),
    )


def make_msa_index_score_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_MSA_HEADS,
    index_dim: int = DEFAULT_INDEX_DIM, block_size: int = DEFAULT_MSA_BLOCK,
    topk_blocks: int = DEFAULT_MSA_TOPK_BLOCKS, return_choice: int = 1,
    cache_dtype: str = "bf16", batch_size: int | None = None,
    shared_prefix: bool = False,
) -> OpSpec:
    """Construct ``MsaIndexScore`` (v1) or ``MsaIndexScoreV2`` (``return_choice=1``)."""
    q_tok = max(1, int(query_tokens))
    batch = max(1, int(batch_size)) if batch_size is not None else max(1, len(requests) or q_tok)
    fused = int(return_choice) == 1
    return _make_scorer_op(
        recipe=MSA_INDEX_SCORE_V2_RECIPE if fused else MSA_INDEX_SCORE_RECIPE,
        native_kind="aclnnMsaIndexScoreV2" if fused else "aclnnMsaIndexScore",
        variant="msa_v2" if fused else "msa_v1",
        src=MSA_SCORE_V2_SOURCE if fused else MSA_SCORE_SOURCE,
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, q_tok=q_tok,
        scanned=max(1, int(context_tokens)), heads=max(1, int(index_heads)),
        dim=max(1, int(index_dim)), blk=max(1, int(block_size)),
        tk_items=max(1, int(topk_blocks)), fused_topk=fused,
        cache_dtype=cache_dtype, has_weights=False, has_mask3=False,
        batch_size=batch, shared_prefix=bool(shared_prefix),
    )


def make_step5_region_score_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_STEP5_HEADS,
    index_dim: int = DEFAULT_INDEX_DIM, region_block_size: int = DEFAULT_STEP5_BLOCK,
    topk_regions: int = DEFAULT_STEP5_TOPK_REGIONS, batch_size: int | None = None,
    shared_prefix: bool = False,
) -> OpSpec:
    """Construct Step-5 ``_C_ascend.npu_indexer_region_score`` op."""
    q_tok = max(1, int(query_tokens))
    batch = max(1, int(batch_size)) if batch_size is not None else max(1, len(requests) or q_tok)
    blk = max(1, int(region_block_size))
    regions = max(1, math.ceil(max(1, int(context_tokens)) / blk))
    return _make_scorer_op(
        recipe=STEP5_REGION_SCORE_RECIPE, native_kind="npu_indexer_region_score",
        variant="step5_region_score", src=STEP5_REGION_SOURCE,
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, q_tok=q_tok, scanned=regions,
        heads=max(1, int(index_heads)), dim=max(1, int(index_dim)), blk=1,
        tk_items=max(1, int(topk_regions)), fused_topk=True,
        cache_dtype="fp8_e4m3", has_weights=True, has_mask3=True,
        batch_size=batch, shared_prefix=bool(shared_prefix),
    )


def make_lightning_indexer_qsa_op(
    *, op_id: str, deps: tuple[str, ...] = (), stage: int = 0, layer: int = 0,
    requests: tuple[str, ...] = (), phase: str = "decode", query_tokens: int,
    context_tokens: int, index_heads: int = DEFAULT_QSA_HEADS,
    index_dim: int = DEFAULT_INDEX_DIM, compress_ratio: int = DEFAULT_QSA_RATIO,
    block_topk: int = DEFAULT_QSA_BLOCK_TOPK, cache_dtype: str = "bf16",
    batch_size: int | None = None, shared_prefix: bool = False,
) -> OpSpec:
    """Construct weight-free ``LightningIndexerQSA`` (#21771 / #22365) op."""
    q_tok = max(1, int(query_tokens))
    batch = max(1, int(batch_size)) if batch_size is not None else max(1, len(requests) or q_tok)
    ratio = max(1, int(compress_ratio))
    scanned = max(1, math.ceil(max(1, int(context_tokens)) / ratio))
    return _make_scorer_op(
        recipe=LIGHTNING_INDEXER_QSA_RECIPE, native_kind="aclnnLightningIndexerQSA",
        variant="qsa_lightning_indexer", src=LIGHTNING_QSA_SOURCE,
        op_id=op_id, deps=deps, stage=stage, layer=layer, requests=requests,
        phase=phase, q_tok=q_tok, scanned=scanned,
        heads=max(1, int(index_heads)), dim=max(1, int(index_dim)), blk=1,
        tk_items=max(1, int(block_topk)), fused_topk=True,
        cache_dtype=cache_dtype, has_weights=False, has_mask3=False,
        batch_size=batch, shared_prefix=bool(shared_prefix),
    )


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Lower any Arch35 sparse index scorer onto AIC+AIV with exact work conservation."""
    buf_sizes = _check_buffers(geometry)
    meta = dict(op.metadata)
    sc = dict(meta.get("scorer_geometry") or {})
    q_tok = max(1, int(sc.get("query_tokens", op.local_shape[0] if op.local_shape else 1)))
    scanned = max(1, int(sc.get("scanned_rows", DEFAULT_INDEX_DIM)))
    heads = max(1, int(sc.get("index_heads", DEFAULT_MSA_HEADS)))
    dim = max(1, int(sc.get("index_dim", DEFAULT_INDEX_DIM)))
    blk = max(1, int(sc.get("block_size", 1)))
    tk_items = max(1, int(sc.get("topk_items", op.local_shape[-1] if op.local_shape else DEFAULT_FALLBACK_TOPK)))
    fused_topk = bool(sc.get("fused_topk", True))
    cache_dt = str(sc.get("cache_dtype", "bf16"))
    has_weights = bool(sc.get("has_head_weights", False))
    has_mask3 = bool(sc.get("has_mask3_sort", False))
    batch = max(1, int(sc.get("batch_size", q_tok)))
    shared = bool(sc.get("shared_prefix", False))

    row_bytes, comp_dt = _cache_row_bytes(cache_dt, dim)
    q_bytes = float(q_tok * heads * dim * (1 if comp_dt == "fp8_e4m3" else BF16_BYTES))
    k_total = float(q_tok * scanned * row_bytes)
    unique_streams = 1 if shared else min(batch, q_tok)
    unique_k = float(unique_streams * scanned * row_bytes)
    aic_read = q_bytes + k_total

    has_l2 = bool((geometry.rates or {}).get("l2", 0.0) or (geometry.resource_units or {}).get("l2", 0))
    l2_cap = _effective_l2(geometry) if has_l2 else 0.0
    repeated_k = max(0.0, k_total - unique_k)
    l2_hit = repeated_k * min(1.0, l2_cap / max(1.0, unique_k)) if (l2_cap > 0.0 and repeated_k > 0.0) else 0.0
    aic_gm_read = max(0.0, aic_read - l2_hit)

    aiv_read = float(q_tok * heads * BF16_BYTES) if has_weights else 0.0
    num_blocks = max(1, math.ceil(scanned / blk))
    out_write = (
        float(q_tok * tk_items * INT32_BYTES)
        if fused_topk else float(q_tok * num_blocks * FP32_BYTES)
    )
    flops = float(2 * q_tok * heads * scanned * dim)
    l0_bytes = q_bytes + k_total
    fix_bytes = float(q_tok * heads * scanned * FP32_BYTES)

    reduce_ops = float(q_tok * heads * scanned * MAX_REDUCE_OPS)
    topk_ops = (
        float(q_tok * num_blocks * max(1.0, math.log2(max(2, tk_items))) * TOPK_SORT_FACTOR)
        if fused_topk else 0.0
    )
    mask_ops = float(q_tok * num_blocks * MASK3_OPS_PER_REGION) if has_mask3 else 0.0
    vec_ops = reduce_ops + topk_ops + mask_ops

    cores = max(1, int(geometry.compute_units))
    aiv_units = max(1, int((geometry.resource_units or {}).get("vector", 2 * cores)))
    aic_pipe = read_role(geometry.resource_units or {}, aiv=None)
    aiv_pipe = read_role(geometry.resource_units or {}, aiv=0)
    plan_res: dict[str, float] = {
        "hbm": aic_gm_read + aiv_read + out_write,
        "mte1": l0_bytes, "fixpipe": fix_bytes,
        "vector": vec_ops, "mte3": out_write,
    }
    if aic_pipe == aiv_pipe:
        plan_res[aic_pipe] = aic_read + aiv_read
    else:
        plan_res[aic_pipe] = aic_read
        if aiv_read > 0.0:
            plan_res[aiv_pipe] = aiv_read
    if l2_hit > 0.0:
        plan_res["l2"] = l2_hit
    total_work = Work(matrix={comp_dt: flops}, resources={k: v for k, v in plan_res.items() if v > 0.0})

    total_tiles = max(1, q_tok * max(1, math.ceil(scanned / SCORER_KV_TILE_ROWS)))
    active_cores = min(cores, total_tiles)

    def program(context):
        c_aic = read_role(context.resource_names, aiv=None)
        c_aiv = read_role(context.resource_names, aiv=0)
        share = 1.0 / float(active_cores)
        is_a5 = "mte2_aic_dram" in context.service_modes
        setup = SCORER_SETUP_MS if is_a5 else 0.0
        topk_lat = 0.0
        caps: dict[str, float] = {}
        if is_a5:
            dram_eff = SCORER_SINGLE_QUERY_DRAM_EFF if q_tok == 1 else SCORER_MULTI_QUERY_DRAM_EFF
            if "hbm" in context.resource_names:
                hbm_phys = context.resource_names["hbm"]
                hbm_rate = float(context.rates.get(hbm_phys, 0.0))
                if hbm_rate > 0.0:
                    caps[hbm_phys] = (dram_eff * hbm_rate) * share
            if has_mask3:
                per_item = STEP5_SINGLE_QUERY_FLOOR_MS if q_tok == 1 else STEP5_REGION_REDUCE_MS_PER_ITEM
                sort_base = (
                    STEP5_TOPK_SORT_BASE_MS
                    if (num_blocks > tk_items and tk_items >= STEP5_TOPK_SORT_MIN_ITEMS)
                    else 0.0
                )
                topk_lat = sort_base + float(q_tok * scanned) * per_item
            elif q_tok > 1 and scanned >= SCORER_LARGE_SCAN_ROWS:
                topk_lat = SCORER_BATCH_TOPK_MS

        def per_core(core: int):
            if core >= active_cores:
                return KernelProgram(())
            aic_owned = context.owned(core, cores, ("cube", "mte1", "fixpipe", c_aic))
            aiv_owned = context.owned(core, cores, ("vector", c_aiv, "mte3"))
            allocs = tuple(
                context.allocate(
                    f"scorer_{core}_{pool}", pool, size,
                    unit_group(core, cores, context.pool_units.get(pool, cores)),
                )
                for pool, size in buf_sizes.items() if pool in context.pool_names
            )
            s_setup = Stage("scorer_setup", latency=setup, acquire=allocs, keep_empty=setup <= 0.0)
            aic_res: dict[str, float] = {
                "hbm": aic_gm_read * share, c_aic: aic_read * share,
                "mte1": l0_bytes * share, "fixpipe": fix_bytes * share,
            }
            if l2_hit > 0.0:
                aic_res["l2"] = l2_hit * share
            s_score = Stage("index_qk_score", context.bind(Work(
                matrix={comp_dt: flops * share}, resources=aic_res)),
                after=("scorer_setup",), placement=aic_owned, caps=caps)
            aiv_res: dict[str, float] = {
                "vector": vec_ops * share,
                "hbm": (aiv_read + out_write) * share,
                "mte3": out_write * share,
            }
            if aiv_read > 0.0:
                aiv_res[c_aiv] = aiv_read * share
            s_topk = Stage("reduce_topk_store", context.bind(Work(resources=aiv_res)),
                           after=("index_qk_score",), latency=topk_lat, free=allocs, placement=aiv_owned)
            return KernelProgram((s_setup, s_score, s_topk))

        return ParallelProgram(tuple(per_core(c) for c in range(cores)))

    return OperatorPlan(total_work, program, {
        "execution_class": "mixed_aic_aiv", "fused_topk": fused_topk,
        "l2_shared_index_bytes": l2_hit,
        "source_program": meta.get("native_source") or MSA_SCORE_V2_SOURCE,
    })


def recipes() -> dict[str, PlannedRecipe]:
    planner = PlannedRecipe(build)
    return {
        MSA_INDEX_SCORE_RECIPE: planner, MSA_INDEX_SCORE_V2_RECIPE: planner,
        STEP5_REGION_SCORE_RECIPE: planner, LIGHTNING_INDEXER_QSA_RECIPE: planner,
    }
