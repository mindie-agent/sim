"""Explicit analytical online index-score/TopK fusion.

[Role] Remove a closed score workspace only for a selected online algorithm.
[Invariants] Source-bound kernels and incomplete head reductions are preserved.
  Matrix FLOPs by precision, Vector work and the final index write survive; only
  the removed score payload round-trip is deleted.
[Timeouts & Locks] Linear graph walk; no I/O or locks.
[Side-Effect Policy] Returns immutable operators only.
"""
from dataclasses import replace

from .implementations import rewire_replacements
from .rewrite_mhc_rules import merge_meta, users_map


NO_WORK = 0
LOCAL_REDUCTION_SHARDS = 1
NO_PORTS: dict = {}
COMPLETE_LOCAL_REDUCTION = "complete_local"
SCORE_RECIPES = frozenset({"index_scores", "batch_prefix_indexer"})


def online_indexer_topk(ops, source_dtype):
    """Opt-in algorithm; not an assertion about a CANN binary's workspace."""
    del source_dtype
    by_id, users = {item.op_id: item for item in ops}, users_map(ops)
    replacements, aliases, removed = {}, {}, set()
    for topk in ops:
        if (topk.recipe != "index_topk" or topk.plan_required
                or len(topk.deps) != LOCAL_REDUCTION_SHARDS):
            continue
        score = by_id.get(topk.deps[NO_WORK])
        if (score is None or score.plan_required
                or score.recipe not in SCORE_RECIPES
                or users[score.op_id] != [topk.op_id]
                or score.stage != topk.stage or score.stream != topk.stream):
            continue
        metadata = dict(score.metadata)
        if (metadata.get("index_score_shards") != LOCAL_REDUCTION_SHARDS
                or metadata.get("index_context_partitions") != LOCAL_REDUCTION_SHARDS
                or metadata.get("index_candidate_filter")
                or metadata.get("index_head_reduction") != COMPLETE_LOCAL_REDUCTION
                or metadata.get("graph_segment")
                != dict(topk.metadata).get("graph_segment")):
            continue
        score_bytes = float(score.mte3_bytes)
        if score_bytes != float(topk.mte2_bytes):
            continue
        hbm = score.hbm_bytes + topk.hbm_bytes - score_bytes - topk.mte2_bytes
        if hbm < NO_WORK:
            continue
        matrix = {}
        for item in (score, topk):
            work = dict(item.matrix_flops) or ({item.dtype: item.flops}
                                               if item.flops else {})
            for dtype, amount in work.items():
                matrix[dtype] = matrix.get(dtype, NO_WORK) + amount
        fused = replace(
            topk, deps=score.deps, family="indexer", recipe="index_online_topk",
            dtype=score.dtype, flops=score.flops + topk.flops,
            matrix_flops=tuple(sorted(matrix.items())),
            vector_ops=score.vector_ops + topk.vector_ops,
            hbm_bytes=hbm, mte2_bytes=score.mte2_bytes,
            mte3_bytes=topk.mte3_bytes,
            metadata=merge_meta(score, topk, extra=(
                ("score_materialization", "analytical_online_ub"),
                ("native_binary_claimed", False),
                ("output_format", "int32"),
                ("removed_score_roundtrip_bytes", score_bytes + topk.mte2_bytes),
            )))
        replacements[topk.op_id] = fused
        aliases[score.op_id] = topk.op_id
        removed.add(score.op_id)
    if not replacements:
        return list(ops)
    return rewire_replacements(
        ops, replacements=replacements,
        aliases={**{key: key for key in replacements}, **aliases},
        removed=removed, ports=NO_PORTS)
