"""Sparse GQA indexer lowering and native index-cache write delegation.

[Role]
  Emit sparse GQA index projections, index-key cache writes, PCP index gathers,
  prefix-reused score sweeps, and TopK reducers, while consumer layers with
  ``owns_topk == False`` reuse ``topk_owner_layer_id`` without duplicate ops.

[Invariants]
  * Delegates index-key cache writing to ``index_cache_writer.emit_index_cache_write``
    when available outside ``raw_eager``, falling back cleanly to ``cache_store_op``.
  * Only explicit caller bindings (origin ``"caller"``) pin ``user_pinned_idx``.

[Timeouts & Locks]
  O(requests) arithmetic per sparse GQA layer; no locks or I/O.

[Side-Effect Policy]
  Appends immutable ``OpSpec`` records to the caller-owned ``ops`` list.
"""
from __future__ import annotations

import math

from mindie_sim.catalog.attention_common import _floor_sum, _long_selected_pairs, local_kv_heads
from mindie_sim.catalog.cache import (
    completed_rows,
    index_row_bytes,
    index_selection_cost,
)
from mindie_sim.catalog.context_parallel import (
    cp_interleave,
    dcp_indexer_sharded,
    owned_positions,
)
from mindie_sim.catalog.ops import cache_store_op, comm_op, gemm_op, vector_op
from mindie_sim.schema.common import ModelingError, align_up, dtype_bytes
from mindie_sim.schema.operators import OpSpec

from .batch_prefix_indexer import apply_indexer_prefix_reads
from .common import (
    is_block_msa_indexer,
    is_qsa_pooled_indexer,
    is_region_csa_indexer,
)
from mindie_sim.prediction.operators.mla.generic_latent_attention import (
    index_key_window,
    owned_store_rows,
    sharded_index_visible_rows,
    visible_parts,
)
from mindie_sim.prediction.implementation_selection import is_caller_pinned_binding
from mindie_sim.prediction.operators.sparse_indexer.index_cache_writer import (
    _is_arch22_target,
    emit_index_cache_write,
)
from mindie_sim.prediction.structure_groups import declare_group

INDEX_SCORE_VECTOR_OPS_PER_ROW: int = 3
TOPK_OPS_PER_ELEMENT: int = 4
INDEX_RMSNORM_OPS: float = 5.0
INDEX_LAYERNORM_OPS: float = 8.0
INDEX_ROPE_OPS: float = 6.0
INDEX_QUANT_OPS: float = 4.0
WEIGHT_CAST_SCALE_OPS: float = 2.0
DEFAULT_KV_BLOCK_SIZE: int = 128
DEFAULT_INDEX_DIM: int = 64
REGION_CSA_DEFAULT_ROPE_DIM: int = 32
REGION_WS_MIN_STRIDE: int = 8
FP32_SCORE_BYTES: int = 4
INT32_INDEX_BYTES: int = 4
MSA_SCORE_ALIGN_COLS: int = 16
MSA_SUPPORTED_ABIS: frozenset[str] = frozenset({
    "legacy_triton_prefill", "legacy_triton_decode", "fp8_triton", "ascendc",
})
_UNQUANTIZED_INDEX_DTYPES: frozenset[str] = frozenset({"bf16", "bfloat16", "fp16", "float16"})
_REGION_SCORE_NATIVE_DTYPES: frozenset[str] = frozenset({"fp16", "bf16", "fp8_e4m3"})
_DTYPE_ALIASES: dict[str, str] = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "fp8": "fp8_e4m3",
}


def _canonical_dtype(raw: str | None, default: str = "bf16") -> str:
    token = str(raw or "").strip().lower()
    if not token:
        return default
    return _DTYPE_ALIASES.get(token, token)


def _resolve_stage_rank(
    machine,
    physical_rank: int | None,
    stage: int | None,
    parallel,
) -> int | None:
    """Map a replica's stage-0 or explicit physical rank onto the die executing ``stage``."""
    if machine is None or stage is None or parallel is None or not hasattr(machine, "validate_placement"):
        return int(physical_rank) if physical_rank is not None else None
    pp = int(getattr(parallel, "pp", 1) or 1)
    if pp <= 1 and physical_rank is not None:
        return int(physical_rank)
    stage_idx = int(stage)
    if not (0 <= stage_idx < pp):
        return int(physical_rank) if physical_rank is not None else None
    try:
        placement = machine.validate_placement(parallel)
    except ModelingError:
        return int(physical_rank) if physical_rank is not None else None
    if physical_rank is None:
        ranks = placement.stage_ranks(0, stage_idx)
        return ranks[0] if ranks else None
    rank_int = int(physical_rank)
    total_replicas = int(getattr(parallel, "external_dp", 1) or 1) * int(getattr(parallel, "dp", 1) or 1)
    for rep in range(total_replicas):
        rep_stage = placement.stage_ranks(rep, stage_idx)
        if rank_int in rep_stage:
            return rank_int
        for other_stage in range(pp):
            if other_stage == stage_idx:
                continue
            other_ranks = placement.stage_ranks(rep, other_stage)
            if rank_int in other_ranks:
                lane = other_ranks.index(rank_int)
                return rep_stage[lane] if lane < len(rep_stage) else (rep_stage[0] if rep_stage else rank_int)
    return rank_int


def _target_device_for_rank(
    machine,
    physical_rank: int | None,
    *,
    stage: int | None = None,
    parallel=None,
):
    """Resolve the physical die for ``(physical_rank, stage)`` on heterogeneous multi-die machines."""
    if machine is None:
        return None
    dies = tuple(getattr(machine, "dies", ()) or ())
    if not dies:
        return machine
    resolved_rank = _resolve_stage_rank(machine, physical_rank, stage, parallel)
    if resolved_rank is None:
        return machine
    die = next((d for d in dies if getattr(d, "rank", None) == resolved_rank), None)
    if die is None and 0 <= resolved_rank < len(dies) and all(getattr(d, "rank", None) is None for d in dies):
        die = dies[resolved_rank]
    if die is not None and (getattr(die, "compute_arch", None) is not None or getattr(die, "arch_family", "")):
        return die
    return machine


def _machine_supports_fp8(runtime, machine) -> bool:
    """True unless target is Arch22 (A2/A3) or declares a ComputeArchSpec without native FP8."""
    if _is_arch22_target(runtime, machine):
        return False
    if machine is not None:
        compute_arch = getattr(machine, "compute_arch", None)
        if compute_arch is None:
            dies = tuple(getattr(machine, "dies", ()) or ())
            if dies:
                compute_arch = getattr(dies[0], "compute_arch", None)
        if compute_arch is not None and compute_arch.has_native_modes():
            return bool(compute_arch.supports_native("matrix", "fp8_e4m3"))
    return True


def _region_csa_precision(quant, runtime, machine) -> tuple[str, str]:
    """Resolve ``(cache_dtype, scale_dtype)`` for Step-5 region-CSA indexers.

    On FP8-capable targets (A5 / Arch35), Step-5 region-CSA stores unscaled
    ``fp8_e4m3`` summaries and scores with ``npu_indexer_region_score`` (which
    accepts only FP16/BF16/FP8 E4M3FN). On Arch22 (A2/A3) or non-FP8 targets,
    honor ``quant.index_dtype`` (default ``bf16``; an explicit ``fp8_e4m3`` is
    preserved so ``index_cache_writer`` rejects it).
    """
    if _machine_supports_fp8(runtime, machine):
        return "fp8_e4m3", "none"
    raw_dtype = str((getattr(quant, "index_dtype", "") if quant is not None else "") or "").strip().lower()
    raw_scale = str((getattr(quant, "index_scale_dtype", "") if quant is not None else "") or "").strip().lower()
    norm_dtype = _DTYPE_ALIASES.get(raw_dtype, raw_dtype)
    eff_dtype = norm_dtype or "bf16"
    if eff_dtype in _UNQUANTIZED_INDEX_DTYPES:
        eff_scale = "none"
    elif raw_scale:
        eff_scale = raw_scale
    else:
        eff_scale = "fp32"
    return eff_dtype, eff_scale


def owned_compressed_group_count(
    j_lo: int,
    j_hi: int,
    *,
    ratio: int,
    dcp: int,
    interleave: int,
    rank: int,
) -> int:
    """Exact O(1) count of store-on-completion groups j in [j_lo, j_hi) owned by DCP rank.

    Compressed group j closes at raw token t_j = (j + 1) * ratio - 1 and belongs to
    rank floor(t_j / interleave) % dcp. Handles both interleave % ratio == 0 and
    non-divisible interleave % ratio != 0 identically without flooring interleave // ratio.
    """
    if j_hi <= j_lo or dcp <= 1:
        return max(0, j_hi - j_lo)
    r = max(1, int(ratio))
    d = max(1, int(dcp))
    stride = max(1, int(interleave))
    rk = int(rank) % d
    if stride % r == 0:
        return int(owned_positions(j_lo, j_hi - 1, d, stride // r, rk))
    period_raw = math.lcm(r, d * stride)
    period_groups = period_raw // r
    cycles_per_period = period_raw // (d * stride)

    def _prefix_count(end_j: int) -> int:
        if end_j <= 0:
            return 0
        full_periods, rem_groups = divmod(end_j, period_groups)
        per_period = 0
        rem_count = 0
        for cyc in range(cycles_per_period):
            lo_raw = (cyc * d + rk) * stride
            hi_raw = lo_raw + stride
            g_lo = lo_raw // r
            g_hi = hi_raw // r
            if g_hi > g_lo:
                per_period += g_hi - g_lo
                if rem_groups > g_lo:
                    rem_count += min(rem_groups, g_hi) - g_lo
        return full_periods * per_period + rem_count

    return _prefix_count(j_hi) - _prefix_count(j_lo)


def _sum_owned_compressed_prefix(
    n_groups: int,
    *,
    ratio: int,
    dcp: int,
    interleave: int,
    rank: int,
) -> int:
    """Exact O(period) sum_{g=0}^{n_groups - 1} owned_compressed_group_count(0, g)."""
    if n_groups <= 1:
        return 0
    r = max(1, int(ratio))
    d = max(1, int(dcp))
    stride = max(1, int(interleave))
    period_raw = math.lcm(r, d * stride)
    period = max(1, period_raw // r)
    per_period_owned = owned_compressed_group_count(
        0, period, ratio=r, dcp=d, interleave=stride, rank=rank,
    )
    prefix_table = [0]
    for g in range(1, period):
        prefix_table.append(
            owned_compressed_group_count(0, g, ratio=r, dcp=d, interleave=stride, rank=rank)
        )
    full_periods, rem = divmod(n_groups, period)
    sum_table = sum(prefix_table)
    full_sum = (
        period * per_period_owned * full_periods * (full_periods - 1) // 2
        + full_periods * sum_table
    )
    rem_sum = rem * full_periods * per_period_owned + sum(prefix_table[:rem])
    return full_sum + rem_sum


def causal_owned_compressed_pairs(
    views,
    *,
    ratio: int,
    dcp: int,
    interleave: int,
    rank: int,
    noncausal: bool = False,
    cand_cap: int = 0,
) -> int:
    """Exact causally visible (query, rank-owned compressed group) pairs respecting query_ranges."""
    r = max(1, int(ratio))
    cap_groups = (int(cand_cap) // r) if cand_cap > 0 else 0
    total = 0
    for v in views:
        ranges = v.query_ranges or ((0, v.q_len),)
        if noncausal:
            q_cnt = sum(e_p - s_p for s_p, e_p in ranges if e_p > s_p)
            vis = owned_compressed_group_count(
                0, completed_rows(v.kv_len, r),
                ratio=r, dcp=dcp, interleave=interleave, rank=rank,
            )
            if cap_groups > 0:
                vis = min(vis, cap_groups)
            total += q_cnt * vis
            continue
        for s_p, e_p in ranges:
            if e_p <= s_p:
                continue
            lo_t = int(v.context) + int(s_p) + 1
            hi_t = int(v.context) + int(e_p)
            g_lo = lo_t // r
            g_hi = hi_t // r
            if g_lo == g_hi:
                vis = owned_compressed_group_count(
                    0, g_lo, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
                )
                if cap_groups > 0:
                    vis = min(vis, cap_groups)
                total += (hi_t - lo_t + 1) * vis
                continue
            first_cnt = (g_lo + 1) * r - lo_t
            last_cnt = hi_t - g_hi * r + 1
            vis_first = owned_compressed_group_count(
                0, g_lo, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
            )
            vis_last = owned_compressed_group_count(
                0, g_hi, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
            )
            if cap_groups > 0:
                vis_first = min(vis_first, cap_groups)
                vis_last = min(vis_last, cap_groups)
            total += first_cnt * vis_first + last_cnt * vis_last
            if g_hi > g_lo + 1:
                if cap_groups > 0 and vis_last >= cap_groups:
                    for g in range(g_lo + 1, g_hi):
                        vis_g = owned_compressed_group_count(
                            0, g, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
                        )
                        total += r * min(vis_g, cap_groups)
                else:
                    mid = (
                        _sum_owned_compressed_prefix(
                            g_hi, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
                        )
                        - _sum_owned_compressed_prefix(
                            g_lo + 1, ratio=r, dcp=dcp, interleave=interleave, rank=rank,
                        )
                    )
                    total += r * mid
    return total


def _emit_index_store(
    *,
    prefix: str,
    deps: tuple[str, ...],
    stage: int,
    layer: int,
    requests: tuple[str, ...],
    phase: str,
    rows: int,
    completed_rows: int | None = None,
    views=None,
    row_bytes: float,
    idx_dim: int,
    idx_kv_heads: int = 1,
    cache_dtype: str | None = None,
    scale_dtype: str | None = None,
    plan,
    model,
    quant,
    runtime,
    machine,
) -> tuple[OpSpec, ...]:
    rt_meta = dict(getattr(runtime, "metadata", ()) or ())
    spine = getattr(runtime, "single_spine", None) or rt_meta.get("single_spine")
    track_spec = getattr(spine, "track_spec", None)
    if track_spec is None and isinstance(spine, dict):
        track_spec = spine.get("track_spec")
    track = getattr(track_spec, "track", None)
    if track is None and isinstance(track_spec, dict):
        track = track_spec.get("track")
    raw_eager = bool(
        rt_meta.get("raw_eager")
        or (rt_meta.get("execution_policy") or {}).get("raw_eager")
        or str(track or "").strip().lower() == "raw_eager"
    )
    if not raw_eager:
        emitted = emit_index_cache_write(
            op_id=prefix + ".index_store",
            prefix=prefix,
            deps=deps,
            stage=stage,
            layer=layer,
            requests=requests,
            phase=phase,
            rows=rows,
            completed_rows=completed_rows,
            views=views,
            dim=idx_dim,
            index_dim=idx_dim,
            heads=idx_kv_heads,
            cache_dtype=cache_dtype,
            scale_dtype=scale_dtype,
            row_bytes=row_bytes,
            plan=plan,
            model=model,
            quant=quant,
            runtime=runtime,
            machine=machine,
        )
        if isinstance(emitted, OpSpec):
            return (emitted,)
        return tuple(emitted)
    if plan is not None and not (getattr(plan, "owns_index_k", False) or getattr(plan, "indexer", False)):
        return ()
    fallback_rows = rows if completed_rows is None else completed_rows
    fallback = cache_store_op(
        op_id=prefix + ".index_store",
        deps=deps,
        stage=stage,
        layer=layer,
        requests=requests,
        phase=phase,
        recipe="index_store",
        rows=fallback_rows,
        row_bytes=max(1, int(round(row_bytes))),
        dtype=cache_dtype or quant.index_dtype,
        kind="index",
    )
    return (fallback,)


def lower_gqa_indexer(
    ops,
    *,
    prefix: str,
    plan,
    views,
    stage: int,
    deps: str,
    model,
    parallel,
    quant,
    mode: str,
    padded_tokens: int,
    valid: int,
    pcp_prefill: bool,
    runtime=None,
    topk_ids=None,
    index_cache_ids=None,
    machine=None,
    physical_rank: int | None = None,
) -> tuple[int, int, int, str | None, str | None, str | None]:
    """Lower GQA sparse indexer ops; return ``(eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id)``."""
    plans = model.layer_plans
    idx_raw, sel_raw = int(plan.index_k_owner_layer_id), int(plan.topk_owner_layer_id)
    if plan.indexer:
        idx_owner = sel_owner = plan.index
    else:
        idx_owner = idx_raw if 0 <= idx_raw < plan.index else -1
        sel_owner = (
            sel_raw
            if 0 <= sel_raw < min(plan.index, len(plans))
            and (plans[sel_raw].indexer or plans[sel_raw].owns_topk)
            else idx_owner
        )
    has_indexer = bool(
        plan.indexer or plan.owns_topk or idx_owner >= 0 or sel_owner >= 0
    )
    eff_topk = int(plan.index_topk) if has_indexer else 0
    index_store_id = candidate_id = None
    topk_id = (topk_ids or {}).get(sel_owner) if sel_owner >= 0 else None
    idx_ratio = max(1, int(plan.compress_ratio or 1), int(plan.index_kpool or 1))
    cand_topk = int(getattr(plan, "candidate_topk_blocks", 0) or 0)
    if _is_block_msa_indexer(plan) and cand_topk > 0:
        sel_topk = cand_topk
    else:
        sel_topk = max(1, eff_topk // idx_ratio) if idx_ratio > 1 else eff_topk
    block_size = max(
        1,
        int(
            getattr(runtime, "kv_block_size", DEFAULT_KV_BLOCK_SIZE)
            or DEFAULT_KV_BLOCK_SIZE
        ),
    )
    user_pinned_idx = (
        is_caller_pinned_binding(runtime, "families", "indexer")
        or is_caller_pinned_binding(runtime, "recipes", "index_scores")
    )
    requests = tuple(view.request_id for view in views)
    if eff_topk > 0 and (plan.indexer or plan.owns_index_k or plan.owns_topk):
        tp = max(1, int(getattr(parallel, "tp", 1) or 1))
        if _is_region_csa_indexer(plan):
            attn_local_kv = local_kv_heads(max(1, int(getattr(plan, "num_kv_heads", 1) or 1)), tp)
            if attn_local_kv != 1:
                raise ModelingError(
                    f"step5_region_csa_requires_single_local_kv_head:"
                    f"num_kv_heads={plan.num_kv_heads}:tp={tp}:local={attn_local_kv}"
                )
        raw_idx_kv_heads = max(
            1,
            int(getattr(plan, "index_kv_heads", 0)
                or getattr(plan, "indexer_kv_heads", 0) or 1),
        )
        local_idx_kv_heads = local_kv_heads(raw_idx_kv_heads, tp)
        eff_dcp = (
            max(1, int(getattr(parallel, "cp", 1) or 1))
            if dcp_indexer_sharded(model, runtime) else 1
        )
        target_hw = _target_device_for_rank(
            machine, physical_rank, stage=stage, parallel=parallel,
        )
        is_region = _is_region_csa_indexer(plan)
        region_dtype, region_scale_dtype = (
            _region_csa_precision(quant, runtime, target_hw)
            if is_region else (None, None)
        )
        native_region_supported = (
            _machine_supports_fp8(runtime, target_hw)
            and region_dtype in _REGION_SCORE_NATIVE_DTYPES
        ) if is_region else False
        (row_bytes, idx_heads, idx_dim, new_idx_rows, rows_by_view,
         shared_idx_by_view, scored_rows, pairs_by_view) = _index_view_work(
            model, plan, views, idx_ratio, valid, block_size,
            local_idx_kv_heads=local_idx_kv_heads, parallel=parallel, eff_dcp=eff_dcp,
            tp=tp, runtime=runtime,
            region_dtype=region_dtype, region_scale_dtype=region_scale_dtype)
        prep_ops, q_ready_id, weights_ready_id, store_ops, pcp_idx_ops, idx_store_id, idx_ready = _index_cache_inputs(
            prefix, deps, stage, plan, requests, padded_tokens, idx_heads, idx_dim,
            model, quant, mode, valid, new_idx_rows, row_bytes, runtime, target_hw,
            pcp_prefill, views, idx_ratio, parallel, local_idx_kv_heads=local_idx_kv_heads,
            region_dtype=region_dtype, region_scale_dtype=region_scale_dtype)
        tail_ops, scores_anchor, topk_anchor, fused_flag = _build_index_score_and_topk_ops(
            prefix=prefix,
            plan=plan,
            views=views,
            stage=stage,
            requests=requests,
            quant=quant,
            mode=mode,
            valid=valid,
            padded_tokens=padded_tokens,
            idx_heads=idx_heads,
            idx_dim=idx_dim,
            row_bytes=row_bytes,
            rows_by_view=rows_by_view,
            shared_idx_by_view=shared_idx_by_view,
            scored_rows=scored_rows,
            idx_wq_id=q_ready_id,
            idx_weights_id=weights_ready_id,
            idx_ready=idx_ready,
            idx_ratio=idx_ratio,
            sel_topk=sel_topk,
            user_pinned_idx=user_pinned_idx,
            runtime=runtime,
            pairs_by_view=pairs_by_view,
            eff_dcp=eff_dcp,
            cp_stride=(cp_interleave(runtime) if runtime is not None else block_size),
            region_dtype=region_dtype,
            native_region_supported=native_region_supported,
        )
        ops.extend((*prep_ops, *store_ops, *pcp_idx_ops, *tail_ops))
        _declare_gqa_indexer_group(
            ops, prefix=prefix, plan=plan, parallel=parallel, model=model,
            quant=quant, views=views, padded_tokens=padded_tokens,
            idx_heads=idx_heads, idx_dim=idx_dim, sel_topk=sel_topk,
            idx_ratio=idx_ratio, scores_id=scores_anchor, topk_id=topk_anchor,
            fused_topk=fused_flag,
            region_dtype=region_dtype, region_scale_dtype=region_scale_dtype,
            native_region_supported=native_region_supported,
        )
        index_store_id, topk_id = idx_store_id, topk_anchor
    elif idx_owner >= 0:
        index_store_id = (index_cache_ids or {}).get(idx_owner)
    return eff_topk, sel_topk, block_size, index_store_id, topk_id, candidate_id


def _resolve_minimax_abi(runtime, mode: str) -> str:
    rt_meta = dict(getattr(runtime, "metadata", ()) or ()) if runtime is not None else {}
    raw = str(rt_meta.get("minimax_indexer_abi") or "ascendc").strip().lower()
    if raw not in MSA_SUPPORTED_ABIS:
        raise ModelingError(f"unsupported_minimax_indexer_abi:{raw}")
    if raw == "legacy_triton_prefill" and mode != "prefill":
        return "legacy_triton_decode"
    return raw


def _msa_block_score_sizing(
    views, *, cand_blk: int, sel_topk: int, idx_heads: int, abi: str, mode: str,
    phys_q: int,
):
    """Exact 4-ABI rectangular MiniMax FP32 block-score allocation & separate TopK read sizing."""
    # Upstream minimax_m3/msa_m3.py:520 slices index queries to num_actual_tokens (excludes padding).
    actual_views = [v for v in views if not getattr(v, "padding", False)]
    logical_cols = [math.ceil(max(1, int(v.kv_len)) / max(1, cand_blk)) for v in actual_views]
    max_blocks = max(logical_cols, default=0)
    logical_pitch = align_up(max_blocks, MSA_SCORE_ALIGN_COLS) if max_blocks > 0 else 0
    alloc_pitch = (
        align_up(max(max_blocks, sel_topk), MSA_SCORE_ALIGN_COLS)
        if (abi == "fp8_triton" and mode != "prefill")
        else logical_pitch
    )
    valid_q = sum(int(v.q_len) for v in actual_views)
    launch_q = max(valid_q, int(phys_q))
    alloc_elems = idx_heads * launch_q * alloc_pitch
    if abi == "legacy_triton_decode":
        topk_read_elems = idx_heads * valid_q * max_blocks
    elif mode == "prefill":
        topk_read_elems = idx_heads * valid_q * logical_pitch
    else:
        topk_read_elems = idx_heads * valid_q * alloc_pitch
    emit_counts = bool(mode != "prefill" and abi in {"fp8_triton", "ascendc"})
    topk_write_bytes = float(
        idx_heads * valid_q * (sel_topk + (1 if emit_counts else 0)) * INT32_INDEX_BYTES
    )
    topk_id_bytes = float(idx_heads * valid_q * sel_topk * INT32_INDEX_BYTES)
    return (
        alloc_elems, topk_read_elems, topk_write_bytes, topk_id_bytes,
        alloc_pitch, emit_counts, launch_q, valid_q,
    )


def _build_msa_phase_pair(
    *, op_suffix: str, prefix: str, plan, phase_views, stage: int, quant,
    phase_mode: str, phys_q: int, idx_heads: int, idx_dim: int, row_bytes: float,
    phase_rows_by_view: list[int], phase_shared_by_view: list[int],
    phase_scored_rows: int, idx_wq_id: str, idx_ready: str, idx_ratio: int,
    sel_topk: int, user_pinned_idx: bool, runtime,
) -> tuple[OpSpec, OpSpec]:
    abi = _resolve_minimax_abi(runtime, phase_mode)
    cand_blk = max(1, int(getattr(plan, "candidate_block_size", 1) or 1))
    (alloc_elems, topk_read_elems, topk_write_bytes, topk_id_bytes,
     w_alloc, emit_counts, launch_q, valid_q) = _msa_block_score_sizing(
        phase_views, cand_blk=cand_blk, sel_topk=sel_topk,
        idx_heads=idx_heads, abi=abi, mode=phase_mode, phys_q=phys_q,
    )
    reqs = tuple(v.request_id for v in phase_views)
    q_idx_bytes = launch_q * idx_heads * idx_dim * dtype_bytes(quant.index_dtype)
    unique_idx_rows = sum(phase_rows_by_view)
    l2_bounded_read = float(min(phase_scored_rows, unique_idx_rows)) * row_bytes
    score_bytes = float(alloc_elems * FP32_SCORE_BYTES)
    topk_read_bytes = float(topk_read_elems * FP32_SCORE_BYTES)
    shape = (phase_scored_rows, idx_heads, idx_dim)
    score_meta: tuple[tuple[str, object], ...] = (
        ("score_materialization", "separate_fp32_score_then_topk"),
        ("indexer_kind", "msa_block_score"),
        ("native_kind", "aclnnMsaIndexScore" if abi == "ascendc" else abi),
        ("minimax_indexer_abi", abi),
        ("msa_score_heads", idx_heads),
        ("msa_w_alloc", w_alloc),
        ("msa_phys_q", launch_q),
        ("msa_valid_q", valid_q),
        ("msa_score_bytes", score_bytes),
    )
    topk_native = (
        "index_decode_topk_fused" if (abi == "ascendc" and phase_mode != "prefill")
        else "TopKV2" if abi == "ascendc"
        else f"{abi}_topk"
    )
    topk_meta: tuple[tuple[str, object], ...] = (
        ("selection_output_dtype", "int32"),
        ("native_kind", topk_native),
        ("minimax_indexer_abi", abi),
        ("msa_score_heads", idx_heads),
        ("msa_w_alloc", w_alloc),
        ("msa_phys_q", launch_q),
        ("msa_valid_q", valid_q),
        ("msa_topk_read_bytes", topk_read_bytes),
        ("msa_id_write_bytes", topk_id_bytes),
        ("msa_emits_counts", emit_counts),
    )
    idx_scores = OpSpec(
        op_id=f"{prefix}.index_scores{op_suffix}",
        family="indexer", phase=phase_mode, stage=stage, layer=plan.index,
        requests=reqs, semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape,
        flops=float(2 * phase_scored_rows * idx_heads * idx_dim),
        vector_ops=float(INDEX_SCORE_VECTOR_OPS_PER_ROW * phase_scored_rows * idx_heads),
        hbm_bytes=float(l2_bounded_read + q_idx_bytes + score_bytes),
        mte2_bytes=float(phase_scored_rows * row_bytes + q_idx_bytes),
        mte3_bytes=score_bytes,
        deps=(idx_wq_id, idx_ready), recipe="index_scores",
        dtype=quant.index_dtype, metadata=score_meta,
    )
    idx_scores = apply_indexer_prefix_reads(
        idx_scores, views=phase_views, rows_by_view=phase_rows_by_view,
        shared_rows_by_view=phase_shared_by_view, row_bytes=float(row_bytes),
        layer_index=plan.index, runtime=runtime, compression_ratio=idx_ratio,
        kernel_reuse=("native_per_sequence" if user_pinned_idx or plan.candidate_filter else "tile"),
        allow_recipe_rebind=not user_pinned_idx,
        hbm_component=l2_bounded_read, mte2_component=float(phase_scored_rows) * row_bytes,
    )
    idx_topk = vector_op(
        op_id=f"{prefix}.index_topk{op_suffix}",
        deps=(idx_scores.op_id,), stage=stage, layer=plan.index, requests=reqs,
        elements=topk_read_elems, ops_per_element=TOPK_OPS_PER_ELEMENT,
        dtype="fp32", family="topk", phase=phase_mode, recipe="index_topk",
        read_bytes_override=topk_read_bytes, write_bytes_override=topk_write_bytes,
        shape=(idx_heads, valid_q, sel_topk), metadata=topk_meta,
    )
    return idx_scores, idx_topk


def _build_qsa_fused_indexer_op(
    *, prefix: str, stage: int, plan, requests: tuple[str, ...], views, quant, mode: str,
    valid: int, idx_heads: int, idx_dim: int, row_bytes: float, rows_by_view: list[int],
    shared_idx_by_view: list[int], scored_rows: int, idx_wq_id: str, idx_ready: str,
    idx_ratio: int, sel_topk: int, user_pinned_idx: bool, runtime,
    q_idx_bytes: float, l2_bounded_index_read: float, shape: tuple[int, int, int],
) -> OpSpec:
    """Fused QSA aclnnLightningIndexerV2 op (scores + TopK in one kernel, zero TopKV2)."""
    id_write_bytes = float(valid * sel_topk * INT32_INDEX_BYTES)
    weights_read_bytes = float(valid * idx_heads * FP32_SCORE_BYTES)
    idx_scores = OpSpec(
        op_id=prefix + ".index_scores",
        family="indexer",
        phase=mode,
        stage=stage,
        layer=plan.index,
        requests=requests,
        semantic_shape=shape,
        local_shape=shape,
        padded_shape=shape,
        launch_shape=shape,
        flops=float(2 * scored_rows * idx_heads * idx_dim),
        vector_ops=float(
            INDEX_SCORE_VECTOR_OPS_PER_ROW * scored_rows * idx_heads
            + scored_rows * TOPK_OPS_PER_ELEMENT
        ),
        hbm_bytes=float(l2_bounded_index_read + q_idx_bytes + weights_read_bytes + id_write_bytes),
        mte2_bytes=float(scored_rows * row_bytes + q_idx_bytes + weights_read_bytes),
        mte3_bytes=id_write_bytes,
        deps=(idx_wq_id, idx_ready),
        recipe="index_scores",
        dtype=quant.index_dtype,
        metadata=(
            ("indexer_kind", "qsa_lightning_indexer_v2"),
            ("native_kind", "aclnnLightningIndexerV2"),
            ("fused_topk", True),
            ("selection_output_dtype", "int32"),
            ("sel_topk", sel_topk),
        ),
    )
    return apply_indexer_prefix_reads(
        idx_scores, views=views, rows_by_view=rows_by_view,
        shared_rows_by_view=shared_idx_by_view, row_bytes=float(row_bytes),
        layer_index=plan.index, runtime=runtime, compression_ratio=idx_ratio,
        kernel_reuse=("native_per_sequence" if user_pinned_idx or plan.candidate_filter else "tile"),
        allow_recipe_rebind=not user_pinned_idx,
        hbm_component=l2_bounded_index_read, mte2_component=float(scored_rows) * row_bytes,
    )


def _build_msa_score_and_topk_ops(
    *, prefix: str, plan, views, stage: int, requests: tuple[str, ...], quant,
    mode: str, valid: int, padded_tokens: int, idx_heads: int, idx_dim: int,
    row_bytes: float, rows_by_view: list[int], shared_idx_by_view: list[int],
    scored_rows: int, idx_wq_id: str, idx_ready: str, idx_ratio: int,
    sel_topk: int, user_pinned_idx: bool, runtime,
    pairs_by_view: Sequence[int] | None = None,
    eff_dcp: int = 1,
    cp_stride: int = 1,
) -> tuple[tuple[OpSpec, ...], str, str, bool]:
    pre_idx = [i for i, v in enumerate(views) if v.phase == "prefill"]
    dec_idx = [i for i, v in enumerate(views) if v.phase != "prefill"]
    if pre_idx and dec_idx:
        cand_cap = (
            plan.candidate_topk_blocks * plan.candidate_block_size
            if plan.candidate_filter else 0
        )
        dec_views = tuple(views[i] for i in dec_idx)
        pre_views = tuple(views[i] for i in pre_idx)
        dec_valid = sum(int(v.q_len) for v in dec_views)
        pre_valid = sum(int(v.q_len) for v in pre_views)
        pad_extra = max(0, int(padded_tokens) - (dec_valid + pre_valid))
        dec_phys = max(sum(v.physical_q_len for v in dec_views), dec_valid + pad_extra)
        pre_phys = sum(v.physical_q_len for v in pre_views)
        if eff_dcp > 1:
            dec_scored = max(
                causal_owned_compressed_pairs(
                    dec_views, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride,
                    rank=rk, noncausal=bool(plan.noncausal_attention), cand_cap=cand_cap,
                )
                for rk in range(eff_dcp)
            )
            pre_scored = max(
                causal_owned_compressed_pairs(
                    pre_views, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride,
                    rank=rk, noncausal=bool(plan.noncausal_attention), cand_cap=cand_cap,
                )
                for rk in range(eff_dcp)
            )
        elif pairs_by_view is not None and len(pairs_by_view) == len(views):
            dec_scored = int(sum(pairs_by_view[i] for i in dec_idx))
            pre_scored = int(sum(pairs_by_view[i] for i in pre_idx))
        else:
            dec_scored = causal_owned_compressed_pairs(
                dec_views, ratio=idx_ratio, dcp=1, interleave=1, rank=0,
                noncausal=bool(plan.noncausal_attention), cand_cap=cand_cap,
            )
            pre_scored = causal_owned_compressed_pairs(
                pre_views, ratio=idx_ratio, dcp=1, interleave=1, rank=0,
                noncausal=bool(plan.noncausal_attention), cand_cap=cand_cap,
            )
        dec_s, dec_t = _build_msa_phase_pair(
            op_suffix=".decode", prefix=prefix, plan=plan, phase_views=dec_views,
            stage=stage, quant=quant, phase_mode="decode", phys_q=dec_phys,
            idx_heads=idx_heads, idx_dim=idx_dim, row_bytes=row_bytes,
            phase_rows_by_view=[rows_by_view[i] for i in dec_idx],
            phase_shared_by_view=[shared_idx_by_view[i] for i in dec_idx],
            phase_scored_rows=dec_scored, idx_wq_id=idx_wq_id, idx_ready=idx_ready,
            idx_ratio=idx_ratio, sel_topk=sel_topk,
            user_pinned_idx=user_pinned_idx, runtime=runtime,
        )
        pre_s, pre_t = _build_msa_phase_pair(
            op_suffix=".prefill", prefix=prefix, plan=plan, phase_views=pre_views,
            stage=stage, quant=quant, phase_mode="prefill", phys_q=pre_phys,
            idx_heads=idx_heads, idx_dim=idx_dim, row_bytes=row_bytes,
            phase_rows_by_view=[rows_by_view[i] for i in pre_idx],
            phase_shared_by_view=[shared_idx_by_view[i] for i in pre_idx],
            phase_scored_rows=pre_scored, idx_wq_id=idx_wq_id, idx_ready=idx_ready,
            idx_ratio=idx_ratio, sel_topk=sel_topk,
            user_pinned_idx=user_pinned_idx, runtime=runtime,
        )
        join = vector_op(
            op_id=f"{prefix}.index_topk",
            deps=(dec_t.op_id, pre_t.op_id), stage=stage, layer=plan.index,
            requests=requests, elements=0, ops_per_element=0, dtype="int32",
            family="tensor_view", phase=mode, recipe="index_topk_merge",
            read_bytes_override=0.0, write_bytes_override=0.0,
            shape=(idx_heads, valid, sel_topk),
            metadata=(("selection_output_dtype", "int32"), ("mixed_phase_split", True)),
        )
        return (dec_s, dec_t, pre_s, pre_t, join), dec_s.op_id, join.op_id, False
    phase_mode = "prefill" if (pre_idx and not dec_idx) else ("decode" if dec_idx else mode)
    phys_q = max(sum(v.physical_q_len for v in views), int(padded_tokens))
    idx_s, idx_t = _build_msa_phase_pair(
        op_suffix="", prefix=prefix, plan=plan, phase_views=views,
        stage=stage, quant=quant, phase_mode=phase_mode, phys_q=phys_q,
        idx_heads=idx_heads, idx_dim=idx_dim, row_bytes=row_bytes,
        phase_rows_by_view=rows_by_view, phase_shared_by_view=shared_idx_by_view,
        phase_scored_rows=scored_rows, idx_wq_id=idx_wq_id, idx_ready=idx_ready,
        idx_ratio=idx_ratio, sel_topk=sel_topk,
        user_pinned_idx=user_pinned_idx, runtime=runtime,
    )
    return (idx_s, idx_t), idx_s.op_id, idx_t.op_id, False


def _qsa_pack_read_groups(views, *, idx_ratio: int, sel_topk: int) -> int:
    """Exact group IDs loaded by _pack_qsa_group_indices_kernel across live query rows."""
    ratio = max(1, int(idx_ratio))
    total = 0
    for v in views:
        if getattr(v, "padding", False) or int(v.q_len) <= 0:
            continue
        seq_cap = min(sel_topk, int(v.kv_len) // ratio)
        if seq_cap <= 0:
            continue
        for s_p, e_p in (v.query_ranges or ((0, int(v.q_len)),)):
            if e_p <= s_p:
                continue
            lo_pos = int(v.context) + int(s_p)
            hi_pos = int(v.context) + int(e_p)
            sat_pos = seq_cap * ratio - 1
            ramp_hi = min(hi_pos, max(lo_pos, sat_pos))
            if ramp_hi > lo_pos:
                total += _floor_sum(ramp_hi - lo_pos, ratio, 1, lo_pos + 1)
            if hi_pos > ramp_hi:
                total += (hi_pos - ramp_hi) * seq_cap
    return total


def _region_csa_reload_rows(views, *, idx_ratio: int, sel_topk: int) -> tuple[int, int]:
    """Return (max_regions_per_token, sum_reload_rows_above_topk) for Step-5 region CSA."""
    ratio = max(1, int(idx_ratio))
    max_regions = 0
    reload_rows = 0
    threshold_pos = (sel_topk + 1) * ratio
    for v in views:
        if getattr(v, "padding", False) or int(v.q_len) <= 0:
            continue
        for s_p, e_p in (v.query_ranges or ((0, int(v.q_len)),)):
            if e_p <= s_p:
                continue
            lo_pos = int(v.context) + int(s_p)
            hi_pos = int(v.context) + int(e_p)
            max_regions = max(max_regions, (hi_pos - 1) // ratio)
            sel_lo = max(lo_pos, threshold_pos)
            if hi_pos > sel_lo:
                reload_rows += _floor_sum(hi_pos - sel_lo, ratio, 1, sel_lo)
    return max_regions, reload_rows


def _build_region_csa_score_op(
    *,
    prefix: str,
    plan,
    views,
    stage: int,
    requests: tuple[str, ...],
    mode: str,
    valid: int,
    idx_heads: int,
    idx_dim: int,
    row_bytes: float,
    rows_by_view: list[int],
    shared_idx_by_view: list[int],
    scored_rows: int,
    idx_wq_id: str,
    idx_ready: str,
    idx_ratio: int,
    sel_topk: int,
    user_pinned_idx: bool,
    runtime,
    idx_weights_id: str | None,
    l2_bounded_index_read: float,
    topk_write_bytes: float,
    shape: tuple[int, int, int],
    region_dtype: str = "fp8_e4m3",
    native_region_supported: bool = True,
) -> tuple[tuple[OpSpec, ...], str, str, bool]:
    # Upstream indexer_region_score.cpp:52-70 + select.h:474-503 allocates a
    # minimum 8-float stride workspace, writes FP32 scores, and reloads for
    # radix TopK only on rows with history > topk (FillUnderKIds otherwise).
    region_q_bytes = float(valid * idx_heads * idx_dim) * dtype_bytes(region_dtype) if scored_rows else 0.0
    weights_read_bytes = float(valid * idx_heads * FP32_SCORE_BYTES) if scored_rows else 0.0
    max_req_regions, reload_score_rows = _region_csa_reload_rows(
        views, idx_ratio=idx_ratio, sel_topk=sel_topk,
    )
    ws_stride = max(REGION_WS_MIN_STRIDE, align_up(max_req_regions, REGION_WS_MIN_STRIDE))
    score_ws_capacity = float(valid * ws_stride * FP32_SCORE_BYTES)
    score_write_bytes = float(scored_rows * FP32_SCORE_BYTES)
    score_reload_bytes = float(reload_score_rows * FP32_SCORE_BYTES)
    score_flops = float(2 * scored_rows * idx_heads * idx_dim)
    score_deps = tuple(dep for dep in (idx_wq_id, idx_weights_id, idx_ready) if dep)
    abi_meta: tuple[tuple[str, object], ...] = (
        (("native_kind", "npu_indexer_region_score"),)
        if native_region_supported
        else (("readiness", "synthesized_target"),)
    )
    fused_meta: tuple[tuple[str, object], ...] = (
        ("score_materialization", "gm_score_workspace_then_radix_topk"),
        ("gm_score_workspace_bytes", score_ws_capacity),
        ("gm_score_reload_bytes", score_reload_bytes),
        ("selection_output_dtype", "int32"),
        ("fused_topk", True),
        ("indexer_kind", "region_csa_score"),
        *abi_meta,
    )
    idx_scores = OpSpec(
        op_id=prefix + ".index_scores", family="indexer", phase=mode, stage=stage,
        layer=plan.index, requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape,
        flops=score_flops,
        matrix_flops=((region_dtype, score_flops),) if score_flops else (),
        vector_ops=float(
            INDEX_SCORE_VECTOR_OPS_PER_ROW * idx_heads * scored_rows
            + TOPK_OPS_PER_ELEMENT * reload_score_rows
        ),
        hbm_bytes=float(
            l2_bounded_index_read + region_q_bytes + weights_read_bytes
            + score_write_bytes + score_reload_bytes + topk_write_bytes
        ),
        mte2_bytes=float(
            scored_rows * row_bytes + region_q_bytes + weights_read_bytes + score_reload_bytes
        ),
        mte3_bytes=float(score_write_bytes + topk_write_bytes),
        deps=score_deps, recipe="index_scores",
        dtype=region_dtype, metadata=fused_meta,
    )
    idx_scores = apply_indexer_prefix_reads(
        idx_scores, views=views, rows_by_view=rows_by_view,
        shared_rows_by_view=shared_idx_by_view, row_bytes=float(row_bytes),
        layer_index=plan.index, runtime=runtime, compression_ratio=idx_ratio,
        kernel_reuse=("native_per_sequence" if user_pinned_idx or plan.candidate_filter else "tile"),
        allow_recipe_rebind=not user_pinned_idx,
        hbm_component=l2_bounded_index_read, mte2_component=float(scored_rows) * row_bytes,
    )
    return (idx_scores,), idx_scores.op_id, idx_scores.op_id, True


def _build_index_score_and_topk_ops(
    *,
    prefix: str,
    plan,
    views,
    stage: int,
    requests: tuple[str, ...],
    quant,
    mode: str,
    valid: int,
    padded_tokens: int,
    idx_heads: int,
    idx_dim: int,
    row_bytes: float,
    rows_by_view: list[int],
    shared_idx_by_view: list[int],
    scored_rows: int,
    idx_wq_id: str,
    idx_ready: str,
    idx_ratio: int,
    sel_topk: int,
    user_pinned_idx: bool,
    runtime,
    idx_weights_id: str | None = None,
    pairs_by_view: Sequence[int] | None = None,
    eff_dcp: int = 1,
    cp_stride: int = 1,
    region_dtype: str | None = None,
    native_region_supported: bool = True,
) -> tuple[tuple[OpSpec, ...], str, str, bool]:
    """Construct the sparse indexer score sweep and TopK selection OpSpecs."""
    q_idx_bytes = valid * idx_heads * idx_dim * dtype_bytes(quant.index_dtype)
    unique_idx_rows = sum(rows_by_view)
    l2_bounded_index_read = float(min(scored_rows, unique_idx_rows)) * row_bytes
    shape = (scored_rows, idx_heads, idx_dim)
    is_region = _is_region_csa_indexer(plan)
    if _is_qsa_pooled_indexer(plan):
        fused = _build_qsa_fused_indexer_op(
            prefix=prefix, stage=stage, plan=plan, requests=requests, views=views,
            quant=quant, mode=mode, valid=valid, idx_heads=idx_heads, idx_dim=idx_dim,
            row_bytes=row_bytes, rows_by_view=rows_by_view,
            shared_idx_by_view=shared_idx_by_view, scored_rows=scored_rows,
            idx_wq_id=idx_wq_id, idx_ready=idx_ready, idx_ratio=idx_ratio,
            sel_topk=sel_topk, user_pinned_idx=user_pinned_idx, runtime=runtime,
            q_idx_bytes=q_idx_bytes, l2_bounded_index_read=l2_bounded_index_read, shape=shape,
        )
        packed_width = sel_topk + max(1, idx_ratio) - 1
        wrapper_output_width = sel_topk * max(1, idx_ratio) + max(1, idx_ratio) - 1
        read_groups = _qsa_pack_read_groups(views, idx_ratio=idx_ratio, sel_topk=sel_topk)
        group_read_bytes = float(read_groups * INT32_INDEX_BYTES)
        packed_write_bytes = float(valid * packed_width * INT32_INDEX_BYTES)
        pack = vector_op(
            op_id=prefix + ".index_pack",
            deps=(fused.op_id,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            elements=valid * packed_width,
            ops_per_element=1.0,
            dtype="int32",
            family="elementwise",
            phase=mode,
            recipe="index_pack",
            read_bytes_override=group_read_bytes,
            write_bytes_override=packed_write_bytes,
            shape=(valid, packed_width),
            metadata=(
                ("native_kind", "_pack_qsa_group_indices_kernel"),
                ("selection_output_dtype", "int32"),
                ("packed_width", packed_width),
                ("wrapper_output_width", wrapper_output_width),
                ("valid_group_reads", read_groups),
                ("compress_ratio", idx_ratio),
                ("sel_topk", sel_topk),
            ),
        )
        return (fused, pack), fused.op_id, pack.op_id, True
    if _is_block_msa_indexer(plan):
        return _build_msa_score_and_topk_ops(
            prefix=prefix, plan=plan, views=views, stage=stage, requests=requests,
            quant=quant, mode=mode, valid=valid, padded_tokens=padded_tokens,
            idx_heads=idx_heads, idx_dim=idx_dim, row_bytes=row_bytes,
            rows_by_view=rows_by_view, shared_idx_by_view=shared_idx_by_view,
            scored_rows=scored_rows, idx_wq_id=idx_wq_id, idx_ready=idx_ready,
            idx_ratio=idx_ratio, sel_topk=sel_topk,
            user_pinned_idx=user_pinned_idx, runtime=runtime,
            pairs_by_view=pairs_by_view, eff_dcp=eff_dcp, cp_stride=cp_stride,
        )
    topk_write_bytes = float(valid * sel_topk * INT32_INDEX_BYTES)
    if is_region:
        return _build_region_csa_score_op(
            prefix=prefix, plan=plan, views=views, stage=stage, requests=requests,
            mode=mode, valid=valid, idx_heads=idx_heads, idx_dim=idx_dim,
            row_bytes=row_bytes, rows_by_view=rows_by_view,
            shared_idx_by_view=shared_idx_by_view, scored_rows=scored_rows,
            idx_wq_id=idx_wq_id, idx_ready=idx_ready, idx_ratio=idx_ratio,
            sel_topk=sel_topk, user_pinned_idx=user_pinned_idx, runtime=runtime,
            idx_weights_id=idx_weights_id,
            l2_bounded_index_read=l2_bounded_index_read,
            topk_write_bytes=topk_write_bytes, shape=shape,
            region_dtype=region_dtype or quant.index_dtype or "bf16",
            native_region_supported=native_region_supported,
        )
    score_bytes = float(scored_rows * FP32_SCORE_BYTES)
    topk_read_bytes = score_bytes
    score_meta: tuple[tuple[str, object], ...] = (
        ("score_materialization", "separate_fp32_score_then_topk"),
    )
    idx_scores = OpSpec(
        op_id=prefix + ".index_scores", family="indexer", phase=mode, stage=stage,
        layer=plan.index, requests=requests, semantic_shape=shape, local_shape=shape,
        padded_shape=shape, launch_shape=shape,
        flops=float(2 * scored_rows * idx_heads * idx_dim),
        vector_ops=float(INDEX_SCORE_VECTOR_OPS_PER_ROW * scored_rows * idx_heads),
        hbm_bytes=float(l2_bounded_index_read + q_idx_bytes + score_bytes),
        mte2_bytes=float(scored_rows * row_bytes + q_idx_bytes),
        mte3_bytes=score_bytes, deps=(idx_wq_id, idx_ready), recipe="index_scores",
        dtype=quant.index_dtype, metadata=score_meta,
    )
    idx_scores = apply_indexer_prefix_reads(
        idx_scores, views=views, rows_by_view=rows_by_view,
        shared_rows_by_view=shared_idx_by_view, row_bytes=float(row_bytes),
        layer_index=plan.index, runtime=runtime, compression_ratio=idx_ratio,
        kernel_reuse=("native_per_sequence" if user_pinned_idx or plan.candidate_filter else "tile"),
        allow_recipe_rebind=not user_pinned_idx,
        hbm_component=l2_bounded_index_read, mte2_component=float(scored_rows) * row_bytes,
    )
    idx_topk = vector_op(
        op_id=prefix + ".index_topk", deps=(idx_scores.op_id,), stage=stage,
        layer=plan.index, requests=requests, elements=scored_rows,
        ops_per_element=TOPK_OPS_PER_ELEMENT, dtype="fp32", family="topk",
        phase=mode, recipe="index_topk", shape=(valid, sel_topk),
        read_bytes_override=topk_read_bytes, write_bytes_override=topk_write_bytes,
        metadata=(("selection_output_dtype", "int32"),),
    )
    return (idx_scores, idx_topk), idx_scores.op_id, idx_topk.op_id, False


_is_block_msa_indexer = is_block_msa_indexer
_is_region_csa_indexer = is_region_csa_indexer
_is_qsa_pooled_indexer = is_qsa_pooled_indexer


def _declare_gqa_indexer_group(
    ops, *, prefix, plan, parallel, model, quant, views, padded_tokens,
    idx_heads, idx_dim, sel_topk, idx_ratio, scores_id, topk_id,
    fused_topk: bool = False,
    region_dtype: str | None = None,
    region_scale_dtype: str | None = None,
    native_region_supported: bool = True,
):
    """Unconditionally declare the GQA sparse indexer semantic subgraph."""
    quantized = quant.index_dtype not in {"bf16", "fp16"}
    is_region = _is_region_csa_indexer(plan)
    if is_region:
        role, family = "pool_key_indexer", "pooled_indexer"
        native_kind = "IndexerRegionScore" if native_region_supported else "AnalyticalRegionScore"
    elif _is_block_msa_indexer(plan):
        role, family, native_kind = "lightning_indexer", "dsa", "MsaIndexScore"
    elif _is_qsa_pooled_indexer(plan):
        role, family, native_kind = "pool_key_indexer", "pooled_indexer", "LightningIndexerV2"
    elif int(plan.index_kpool or 1) > 1:
        role, family, native_kind = "pool_key_indexer", "pooled_indexer", "PoolKeyIndexer"
    elif quantized:
        role, family, native_kind = (
            "quant_lightning_indexer", "dsa", "VllmAscendQuantLightningIndexerV2")
    else:
        role, family, native_kind = "lightning_indexer", "dsa", "LightningIndexer"
    partition = tuple(
        {"query": v.q_len, "context": v.context, "kv_length": v.kv_len}
        for v in views
    )
    members = (scores_id,) if scores_id == topk_id else (scores_id, topk_id)
    single_op = fused_topk and scores_id == topk_id
    is_qsa = _is_qsa_pooled_indexer(plan)
    eff_query_dtype = (region_dtype or quant.index_dtype) if is_region else quant.index_dtype
    eff_scale_dtype = (
        region_scale_dtype
        if (is_region and region_scale_dtype is not None)
        else quant.index_scale_dtype
    )
    eff_weight_dtype = (
        "fp32" if (is_region or is_qsa)
        else ("fp16" if quant.index_dtype == "int8" else model.activation_dtype)
    )
    declare_group(
        ops, key=prefix + "." + role, role=role, family=family,
        members=members, entries=(scores_id,), anchor=topk_id,
        block="attention",
        view={
            "tokens": sum(v.q_len for v in views), "physical_tokens": padded_tokens,
            "sequence_partition": partition, "compression_ratio": max(1, idx_ratio),
            "heads": idx_heads, "width": idx_dim, "topk": sel_topk,
            "query_dtype": eff_query_dtype,
            "weight_dtype": eff_weight_dtype,
            "cache_dtype": eff_query_dtype,
            "selection_output_dtype": "int32",
            "scale_dtype": eff_scale_dtype,
        },
        implementation={
            "native_kind": native_kind,
            "conditions": {
                "cp": int(parallel.cp), "pcp": int(parallel.pcp),
                "candidate_filter": bool(plan.candidate_filter),
                "source_selection": "gqa_fused_score_topk" if fused_topk else "gqa_semantic_score_and_topk_projection",
                "timing_boundary": "fused_single_op" if single_op else "separate_declared_operators",
            },
        },
        source={"kind": "semantic_builder",
                "module": "mindie_sim.prediction.operators.sparse_indexer.gqa_indexer"},
        fusion_status="fused_single_op" if single_op else "unfused_decomposed",
        fusion_reason=(
            ("region_csa_single_op" if native_region_supported else "region_csa_analytical_arch22")
            if (single_op and is_region)
            else ("qsa_lightning_indexer_v2_fused_topk_then_pack" if fused_topk else "gqa_indexer_separate_score_and_topk")
        ),
        native_fusion_eligible=(single_op and (native_region_supported or not is_region)),
    )


def _index_view_work(
    model, plan, views, idx_ratio, valid, block_size, *,
    local_idx_kv_heads: int = 1, parallel=None, eff_dcp: int = 1, tp: int = 1,
    runtime=None, region_dtype: str | None = None, region_scale_dtype: str | None = None,
):
    cost = index_selection_cost(
        model, plan, max((v.kv_len for v in views), default=0)
    )
    idx_dim = max(1, int(plan.index_dim or DEFAULT_INDEX_DIM))
    if _is_region_csa_indexer(plan):
        eff_region_dt = region_dtype or str(getattr(model.quant, "index_dtype", "bf16") or "bf16")
        scale_dt = (
            region_scale_dtype
            if region_scale_dtype is not None
            else str(getattr(model.quant, "index_scale_dtype", "none") or "none")
        ).strip().lower()
        scale_b = float(dtype_bytes(scale_dt)) if scale_dt not in {"", "none"} else 0.0
        row_bytes_per_head = float(idx_dim) * float(dtype_bytes(eff_region_dt)) + scale_b
    else:
        row_bytes_per_head = cost.index_row_bytes or index_row_bytes(model, plan.index_dim)
    row_bytes = row_bytes_per_head * max(1, int(local_idx_kv_heads))
    raw_idx_heads = max(1, int(plan.index_heads or 1))
    idx_heads = (
        local_kv_heads(raw_idx_heads, max(1, int(tp)))
        if _is_block_msa_indexer(plan)
        else raw_idx_heads
    )
    cand_cap = (
        plan.candidate_topk_blocks * plan.candidate_block_size
        if plan.candidate_filter
        else 0
    )
    new_rows_by_view = [
        sum(
            completed_rows(v.context + e_p, idx_ratio)
            - completed_rows(v.context + s_p, idx_ratio)
            for s_p, e_p in (v.query_ranges or ((0, v.q_len),))
            if e_p > s_p
        )
        if idx_ratio > 1
        else int(v.q_len)
        for v in views
    ]
    cp_stride = cp_interleave(runtime) if runtime is not None else block_size
    if eff_dcp > 1 and parallel is not None and any(new_rows_by_view):
        new_idx_rows = max(
            sum(
                sum(
                    owned_compressed_group_count(
                        completed_rows(v.context + s_p, idx_ratio),
                        completed_rows(v.context + e_p, idx_ratio),
                        ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride, rank=rk,
                    )
                    for s_p, e_p in (v.query_ranges or ((0, v.q_len),))
                    if e_p > s_p
                )
                for v in views
            )
            for rk in range(eff_dcp)
        )
    else:
        new_idx_rows = sum(new_rows_by_view) if idx_ratio > 1 else valid
    rows_by_view: list[int] = []
    shared_idx_by_view: list[int] = []
    pairs_by_view: list[int] = []
    is_region = _is_region_csa_indexer(plan)
    for v in views:
        v_ranges = v.query_ranges or ((0, v.q_len),)
        if is_region:
            v_scored = int(
                sum(
                    _floor_sum(e_p - s_p, idx_ratio, 1, int(v.context) + s_p)
                    for s_p, e_p in v_ranges
                    if e_p > s_p
                )
            )
        else:
            v_scored = int(
                sum(
                    _long_selected_pairs(
                        e_p - s_p,
                        v.context + s_p,
                        idx_ratio,
                        cand_cap,
                        noncausal=bool(plan.noncausal_attention),
                        kv_len=v.kv_len,
                    )
                    for s_p, e_p in v_ranges
                    if e_p > s_p
                )
            )
        raw_pref = max(
            0, min(int(getattr(v, "prefix_tokens", 0) or 0), int(v.context))
        )
        p_raw = (raw_pref // block_size) * block_size
        pref_groups = completed_rows(p_raw, idx_ratio)
        if eff_dcp > 1 and cp_stride % idx_ratio != 0:
            total_groups = completed_rows(v.kv_len, idx_ratio)
            physical_rows = max(
                owned_compressed_group_count(
                    0, total_groups, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride, rank=rk,
                )
                for rk in range(eff_dcp)
            )
            owned_shared = max(
                owned_compressed_group_count(
                    0, pref_groups, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride, rank=rk,
                )
                for rk in range(eff_dcp)
            )
        else:
            physical_rows, _ = index_key_window(
                visible_parts(v),
                noncausal=bool(plan.noncausal_attention),
                kv_len=v.kv_len,
                cp=eff_dcp,
                interleave=cp_stride,
                compression=idx_ratio,
            )
            owned_shared = (
                max(
                    owned_compressed_group_count(
                        0, pref_groups, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride, rank=rk,
                    )
                    for rk in range(eff_dcp)
                )
                if eff_dcp > 1 else pref_groups
            )
        pairs_by_view.append(v_scored)
        rows_by_view.append(physical_rows)
        shared_idx_by_view.append(min(physical_rows, owned_shared))
    if eff_dcp > 1 and parallel is not None:
        scored_rows = max(
            causal_owned_compressed_pairs(
                views, ratio=idx_ratio, dcp=eff_dcp, interleave=cp_stride,
                rank=rk, noncausal=bool(plan.noncausal_attention), cand_cap=cand_cap,
            )
            for rk in range(eff_dcp)
        )
    else:
        scored_rows = sum(pairs_by_view)
    return (
        row_bytes, idx_heads, idx_dim, new_idx_rows, rows_by_view,
        shared_idx_by_view, scored_rows, pairs_by_view,
    )


def _region_csa_prep_ops(
    *, prefix: str, deps: str, stage: int, plan, requests: tuple[str, ...],
    padded_tokens: int, idx_heads: int, idx_dim: int, local_k_heads: int,
    idx_wq: OpSpec, idx_wk: OpSpec, model, quant, mode: str,
    region_dtype: str = "fp8_e4m3",
) -> tuple[tuple[OpSpec, ...], str, str, tuple[str, ...]]:
    """Emit Step-5 region CSA Z/weight projections, Q/K norms, RoPE, and query quant/cast."""
    act_dtype = str(getattr(model, "activation_dtype", "bf16") or "bf16")
    act_bytes = float(dtype_bytes(act_dtype))
    idx_wz = gemm_op(
        op_id=prefix + ".index_wz",
        deps=(deps,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        m=padded_tokens,
        n=local_k_heads * idx_dim,
        k=model.hidden_size,
        quant=quant,
        category="indexer",
        phase=mode,
        recipe="index_wz",
        family="index_projection",
    )
    idx_weights = gemm_op(
        op_id=prefix + ".index_weights",
        deps=(deps,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        m=padded_tokens,
        n=idx_heads,
        k=model.hidden_size,
        quant=quant,
        category="indexer.weights",
        phase=mode,
        recipe="index_weights",
        family="index_projection",
    )
    prep: list[OpSpec] = [idx_wq, idx_wk, idx_wz, idx_weights]
    w_elems = padded_tokens * idx_heads
    weights_ready_op = vector_op(
        op_id=prefix + ".index_weights_fp32",
        deps=(idx_weights.op_id,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        elements=w_elems,
        ops_per_element=WEIGHT_CAST_SCALE_OPS,
        dtype="fp32",
        family="cast",
        phase=mode,
        recipe="index_weights_fp32",
        read_bytes_override=float(w_elems) * act_bytes,
        write_bytes_override=float(w_elems) * float(FP32_SCORE_BYTES),
        shape=(padded_tokens, idx_heads),
        metadata=(("input_format", act_dtype), ("output_format", "fp32")),
    )
    prep.append(weights_ready_op)
    q_elems = padded_tokens * idx_heads * idx_dim
    k_elems = padded_tokens * local_k_heads * idx_dim
    q_norm = vector_op(
        op_id=prefix + ".index_q_norm",
        deps=(idx_wq.op_id,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        elements=q_elems,
        ops_per_element=INDEX_RMSNORM_OPS,
        dtype=act_dtype,
        family="rmsnorm",
        phase=mode,
        recipe="index_q_norm",
        read_bytes_override=float(q_elems) * act_bytes + float(idx_dim * FP32_SCORE_BYTES),
        write_bytes_override=float(q_elems) * act_bytes,
        shape=(padded_tokens, idx_heads, idx_dim),
    )
    k_norm = vector_op(
        op_id=prefix + ".index_k_norm",
        deps=(idx_wk.op_id,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        elements=k_elems,
        ops_per_element=INDEX_LAYERNORM_OPS,
        dtype=act_dtype,
        family="layernorm",
        phase=mode,
        recipe="index_k_norm",
        read_bytes_override=float(k_elems) * act_bytes + float(2 * idx_dim * FP32_SCORE_BYTES),
        write_bytes_override=float(k_elems) * act_bytes,
        shape=(padded_tokens, local_k_heads, idx_dim),
    )
    qk_elems = q_elems + k_elems
    declared_rope = int(
        getattr(plan, "index_rope_dim", 0)
        or getattr(plan, "sparse_indexer_rope_dim", 0)
        or REGION_CSA_DEFAULT_ROPE_DIM
    )
    rope_dim = min(idx_dim, max(1, declared_rope))
    rope_elems = padded_tokens * (idx_heads + local_k_heads) * rope_dim
    qk_rope = vector_op(
        op_id=prefix + ".index_qk_rope",
        deps=(q_norm.op_id, k_norm.op_id),
        stage=stage,
        layer=plan.index,
        requests=requests,
        elements=rope_elems,
        ops_per_element=INDEX_ROPE_OPS,
        dtype=act_dtype,
        family="rotary",
        phase=mode,
        recipe="rotary",
        read_bytes_override=float(qk_elems) * act_bytes + float(padded_tokens * rope_dim) * act_bytes,
        write_bytes_override=float(qk_elems) * act_bytes,
        shape=(padded_tokens, idx_heads + local_k_heads, idx_dim),
    )
    prep.extend((q_norm, k_norm, qk_rope))
    q_ready_id = qk_rope.op_id
    eff_region_dt = _canonical_dtype(region_dtype, default=_canonical_dtype(act_dtype))
    if eff_region_dt != _canonical_dtype(act_dtype):
        is_quant = eff_region_dt not in _UNQUANTIZED_INDEX_DTYPES
        q_convert = vector_op(
            op_id=prefix + (".index_q_quant" if is_quant else ".index_q_cast"),
            deps=(qk_rope.op_id,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            elements=q_elems,
            ops_per_element=INDEX_QUANT_OPS if is_quant else 1.0,
            dtype=eff_region_dt,
            family="quantize" if is_quant else "cast",
            phase=mode,
            recipe="index_q_quant" if is_quant else "index_q_cast",
            read_bytes_override=float(q_elems) * act_bytes,
            write_bytes_override=float(q_elems) * float(dtype_bytes(eff_region_dt)),
            shape=(padded_tokens, idx_heads, idx_dim),
            metadata=(("input_format", act_dtype), ("output_format", eff_region_dt)),
        )
        prep.append(q_convert)
        q_ready_id = q_convert.op_id
    return tuple(prep), q_ready_id, weights_ready_op.op_id, (qk_rope.op_id, idx_wz.op_id)


def _index_cache_inputs(
    prefix, deps, stage, plan, requests, padded_tokens, idx_heads, idx_dim, model, quant, mode,
    valid, new_idx_rows, row_bytes, runtime, machine, pcp_prefill, views, idx_ratio, parallel,
    *, local_idx_kv_heads: int = 1,
    region_dtype: str | None = None,
    region_scale_dtype: str | None = None,
):
    local_k_heads = max(1, int(local_idx_kv_heads))
    idx_wq = gemm_op(
        op_id=prefix + ".index_wq",
        deps=(deps,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        m=padded_tokens,
        n=idx_heads * idx_dim,
        k=model.hidden_size,
        quant=quant,
        category="indexer",
        phase=mode,
        recipe="index_wq",
        family="index_projection",
    )
    idx_wk = gemm_op(
        op_id=prefix + ".index_wk",
        deps=(deps,),
        stage=stage,
        layer=plan.index,
        requests=requests,
        m=padded_tokens,
        n=local_k_heads * idx_dim,
        k=model.hidden_size,
        quant=quant,
        category="indexer",
        phase=mode,
        recipe="index_wk",
        family="index_projection",
    )
    is_region = _is_region_csa_indexer(plan)
    if is_region:
        eff_region_dt, eff_region_scale = (
            (region_dtype, region_scale_dtype)
            if (region_dtype is not None and region_scale_dtype is not None)
            else _region_csa_precision(quant, runtime, machine)
        )
        prep_ops, q_ready_id, weights_ready_id, store_deps = _region_csa_prep_ops(
            prefix=prefix, deps=deps, stage=stage, plan=plan, requests=requests,
            padded_tokens=padded_tokens, idx_heads=idx_heads, idx_dim=idx_dim,
            local_k_heads=local_k_heads, idx_wq=idx_wq, idx_wk=idx_wk,
            model=model, quant=quant, mode=mode, region_dtype=eff_region_dt,
        )
        k_ready_id = store_deps[0]
        store_cache_dtype: str | None = eff_region_dt
        store_scale_dtype: str | None = eff_region_scale
    else:
        prep_ops = (idx_wq, idx_wk)
        q_ready_id = idx_wq.op_id
        weights_ready_id = None
        k_ready_id = idx_wk.op_id
        store_deps = (k_ready_id,)
        store_cache_dtype = None
        store_scale_dtype = None
    store_ops = _emit_index_store(
        prefix=prefix,
        deps=store_deps,
        stage=stage,
        layer=plan.index,
        requests=requests,
        phase=mode,
        rows=valid,
        completed_rows=new_idx_rows,
        views=views,
        row_bytes=row_bytes,
        idx_dim=idx_dim,
        idx_kv_heads=local_k_heads,
        cache_dtype=store_cache_dtype,
        scale_dtype=store_scale_dtype,
        plan=plan,
        model=model,
        quant=quant,
        runtime=runtime,
        machine=machine,
    )
    idx_store_id = store_ops[-1].op_id if store_ops else None
    pcp_idx_ops: tuple[OpSpec, ...] = ()
    idx_ready = idx_store_id or k_ready_id
    if pcp_prefill:
        if plan.index_kpool > 1:
            global_idx_rows = sum(
                (v.context + (v.global_q_len or v.q_len)) // plan.index_kpool
                - v.context // plan.index_kpool
                for v in views
                if v.phase == "prefill"
            )
        else:
            global_idx_rows = sum(
                completed_rows(v.context + (v.global_q_len or v.q_len), idx_ratio)
                - completed_rows(v.context, idx_ratio)
                for v in views
                if v.phase == "prefill"
            )
        idx_gather = comm_op(
            op_id=prefix + ".pcp_index_gather",
            deps=(idx_ready,),
            stage=stage,
            layer=plan.index,
            requests=requests,
            family="allgather",
            phase=mode,
            recipe="pcp_index_gather",
            link_bytes=float(
                global_idx_rows * row_bytes * (parallel.pcp - 1) / parallel.pcp
            ),
            shape=(max(1, global_idx_rows), idx_dim),
            metadata=(("group_axis", "pcp"),),
        )
        pcp_idx_ops = (idx_gather,)
        idx_ready = idx_gather.op_id
    return prep_ops, q_ready_id, weights_ready_id, store_ops, pcp_idx_ops, idx_store_id, idx_ready
