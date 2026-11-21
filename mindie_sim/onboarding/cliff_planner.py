"""Disjoint 2D cliff-plan synthesis with declarative TilingAxes ABI guards.

[Role] Synthesize strictly disjoint ``train`` + 2D ``(B/q x Context x Parallel)``
  off-grid cliff ``heldout`` + ``abi_rejection`` benchmark plans from declared
  ``TilingAxes`` / ``OperatorCliffParameters`` (Prevention Lock 3), and
  re-export sealed ``GateCase -> EvaluationRunner -> EngineRunReceipt`` scoring
  from ``_sealed_scorer``.
[Invariants]
  * ``train`` vs ``heldout`` contexts, point coordinates, and acquisition
    lineage roots are strictly disjoint; ``262144`` is ``train`` and
    ``400000`` (``heldout``) is distinct from ``409600`` (``train``).
  * All candidate cliffs are filtered through ``validate_abi``; illegal ABI
    shapes (e.g., ``topk=2049``) route to ``abi_rejection_points`` while legal
    performance cliffs (e.g., ``context=2049, topk=2048``) stay in ``heldout``.
[Timeouts & Locks] Pure in-memory plan synthesis; zero subprocesses, network
  calls, or device contexts.
[Side-Effect Policy] Zero filesystem writes or global state mutations.
"""
from __future__ import annotations

import hashlib
from typing import Any

from ._sealed_scorer import (
    STANDARD_LEVEL_BOUNDARY_KINDS,
    WhiteboxSpineRunner,
    score_sealed_gate_cases,
    validate_disjoint_lineage,
)
from .contracts import (
    ANCHOR_CONTEXT_GRID,
    CLIFF_PLAN_SCHEMA,
    COLD_RING_L2_MULTIPLIER,
    CONTEXT_MAX_TOKENS,
    CONTEXT_MIN_TOKENS,
    DEFAULT_SPARSE_TOPK,
    OFF_GRID_CLIFF_HELDOUT_GRID,
    SHORT_DIGEST_LEN,
    CliffBenchPlan,
    CliffPointSpec,
    OperatorCliffParameters,
    assert_sanitized_payload,
    canonical_json_bytes,
    sanitize_public_alias,
)
from .tiling_axes import (
    CROSS_SAMPLE_BATCH_SIZES,
    CROSS_SAMPLE_CHUNK_Q,
    CROSS_SAMPLE_MTP_Q,
    CROSS_SAMPLE_PARALLEL_CONTEXT_MIN,
    TilingAxes,
    default_topk_abi_rule,
    validate_abi,
)

_DEFAULT_SKU_TILE_DIM = 128
_CROSS_BATCH_CONTEXTS: tuple[int, ...] = (4097, 24576, 98304)
_CROSS_MTP_CONTEXTS: tuple[int, ...] = (2049, 24576, 98304)
_CROSS_CHUNK_CONTEXTS: tuple[int, ...] = (98304, 200000)
_CROSS_PARALLEL_SPECS: tuple[tuple[int, str], ...] = (
    (200000, "pcp2_dcp2"),
    (400000, "pcp4_dcp2"),
)


def _clamp_in_domain(value: int, *, low: int = 1, high: int = CONTEXT_MAX_TOKENS) -> int:
    return max(low, min(high, int(value)))


def _resolve_tiling_axes(params: OperatorCliffParameters) -> TilingAxes:
    if params.tiling_axes is not None:
        return params.tiling_axes
    thresholds: list[tuple[str, int]] = []
    rules = []
    if params.topk_threshold is not None:
        thresholds.append(("topk", int(params.topk_threshold)))
        rules.append(default_topk_abi_rule(above_threshold=int(params.topk_threshold)))
    return TilingAxes(
        tile_dims=(int(params.tile_dim),),
        core_product_axes=("tokens", "batch_x_kv_heads"),
        phase_thresholds=tuple(thresholds),
        abi_divisibility=tuple(rules),
    )


def _append_guarded_point(
    *,
    heldout: list[CliffPointSpec],
    abi_rejections: list[CliffPointSpec],
    seen_perf: set[tuple[str, int, int, int, int | None, str]],
    seen_reject: set[tuple[str, int, int | None]],
    train_keys: set[tuple[str, int]],
    axes: TilingAxes,
    point_id: str,
    grid_kind: str,
    axis: str,
    value: int,
    regime_tag: str,
    rationale: str,
    batch_size: int = 1,
    query_tokens: int = 1,
    topk: int | None = None,
    parallel_topology: str = "tp1",
) -> None:
    if value < 1:
        return
    probe_shape: dict[str, Any] = {
        "axis": axis,
        "value": value,
        "batch_size": batch_size,
        "query_tokens": query_tokens,
    }
    if axis == "context_len":
        probe_shape["context_len"] = value
    if topk is not None:
        probe_shape["topk"] = topk
    ok_abi, reject_reason = validate_abi(probe_shape, axes)
    if not ok_abi:
        rkey = (axis, value, topk)
        if rkey not in seen_reject:
            seen_reject.add(rkey)
            abi_rejections.append(
                CliffPointSpec(
                    point_id=f"abi_reject_{point_id}",
                    split="abi_rejection",
                    grid_kind="abi_rejection",
                    axis=axis,
                    value=value,
                    regime_tag=f"abi_reject_{regime_tag}",
                    rationale=rationale,
                    batch_size=batch_size,
                    query_tokens=query_tokens,
                    topk=topk,
                    parallel_topology=parallel_topology,
                    abi_expected_error=reject_reason,
                )
            )
        return
    if (axis, value) in train_keys:
        return
    pkey = (axis, value, batch_size, query_tokens, topk, parallel_topology)
    if pkey in seen_perf:
        return
    seen_perf.add(pkey)
    heldout.append(
        CliffPointSpec(
            point_id=point_id,
            split="heldout",
            grid_kind=grid_kind,  # type: ignore[arg-type]
            axis=axis,
            value=value,
            regime_tag=regime_tag,
            rationale=rationale,
            batch_size=batch_size,
            query_tokens=query_tokens,
            topk=topk,
            parallel_topology=parallel_topology,
        )
    )


def synthesize_operator_cliff_plan(
    params: OperatorCliffParameters,
    *,
    sku: str,
    campaign_id: str = "onboard-op",
) -> CliffBenchPlan:
    """Synthesize disjoint train + 2D off-grid cliff heldout + ABI-rejection plan."""
    clean_sku = sanitize_public_alias(sku, "sku")
    clean_campaign = sanitize_public_alias(campaign_id, "onboard")
    axes = _resolve_tiling_axes(params)

    train_points: list[CliffPointSpec] = []
    train_keys: set[tuple[str, int]] = set()
    for ctx in ANCHOR_CONTEXT_GRID:
        if params.context_min <= ctx <= params.context_max:
            train_keys.add(("context_len", ctx))
            train_points.append(
                CliffPointSpec(
                    point_id=f"anchor_ctx_{ctx}",
                    split="train",
                    grid_kind="anchor",
                    axis="context_len",
                    value=ctx,
                    regime_tag="anchor_train",
                    rationale="canonical_anchor_context_grid_512_to_409600",
                )
            )
    if not train_points:
        train_keys.add(("context_len", params.context_min))
        train_points.append(
            CliffPointSpec(
                point_id=f"anchor_ctx_{params.context_min}",
                split="train",
                grid_kind="anchor",
                axis="context_len",
                value=params.context_min,
                regime_tag="anchor_train",
                rationale="domain_min_anchor",
            )
        )

    heldout: list[CliffPointSpec] = []
    abi_rejections: list[CliffPointSpec] = []
    seen_perf: set[tuple[str, int, int, int, int | None, str]] = set()
    seen_reject: set[tuple[str, int, int | None]] = set()

    def _add(**kw: Any) -> None:
        _append_guarded_point(
            heldout=heldout, abi_rejections=abi_rejections,
            seen_perf=seen_perf, seen_reject=seen_reject,
            train_keys=train_keys, axes=axes, **kw,
        )

    for ctx in OFF_GRID_CLIFF_HELDOUT_GRID:
        if params.context_min <= ctx <= params.context_max:
            _add(
                point_id=f"offgrid_ctx_{ctx}", grid_kind="off_grid_cliff",
                axis="context_len", value=ctx, regime_tag="off_grid_heldout",
                rationale="canonical_off_grid_cliff_heldout_grid",
            )

    for batch, ctx in zip(CROSS_SAMPLE_BATCH_SIZES, _CROSS_BATCH_CONTEXTS):
        if params.context_min <= ctx <= params.context_max:
            _add(
                point_id=f"cross_b{batch}_ctx_{ctx}", grid_kind="cross_sample_2d",
                axis="context_len", value=ctx, regime_tag="high_batch_decode_2d",
                rationale="2d_high_batch_decode_core_wave_cliff",
                batch_size=batch, query_tokens=1,
            )

    for q_tok, ctx in zip(CROSS_SAMPLE_MTP_Q, _CROSS_MTP_CONTEXTS):
        if params.context_min <= ctx <= params.context_max:
            _add(
                point_id=f"cross_mtp_q{q_tok}_ctx_{ctx}", grid_kind="cross_sample_2d",
                axis="context_len", value=ctx, regime_tag="mtp_decode_2d",
                rationale="2d_mtp_multi_query_decode_cliff",
                batch_size=1, query_tokens=q_tok,
            )

    for q_tok, ctx in zip(CROSS_SAMPLE_CHUNK_Q, _CROSS_CHUNK_CONTEXTS):
        if params.context_min <= ctx <= params.context_max:
            _add(
                point_id=f"cross_chunk_q{q_tok}_ctx_{ctx}", grid_kind="cross_sample_2d",
                axis="context_len", value=ctx, regime_tag="chunked_prefill_2d",
                rationale="2d_long_context_chunked_prefill_cliff",
                batch_size=1, query_tokens=q_tok,
            )

    for ctx, topo in _CROSS_PARALLEL_SPECS:
        if ctx >= CROSS_SAMPLE_PARALLEL_CONTEXT_MIN and params.context_min <= ctx <= params.context_max:
            _add(
                point_id=f"cross_{topo}_ctx_{ctx}", grid_kind="cross_sample_2d",
                axis="context_len", value=ctx, regime_tag="multi_rank_cp_2d",
                rationale="2d_multi_rank_pcp_dcp_long_context_cliff",
                batch_size=1, query_tokens=1, parallel_topology=topo,
            )

    for tile in axes.tile_dims:
        for delta, tag in ((-1, "minus1"), (0, "exact"), (1, "plus1")):
            _add(
                point_id=f"tile_{tile}_{tag}_{tile + delta}", grid_kind="boundary_cliff",
                axis="tokens", value=tile + delta, regime_tag="tile_boundary",
                rationale=f"tile_dim({tile}){delta:+d}_tail_wave_boundary",
            )
        full_wave = params.core_count * tile
        for delta, tag in ((-1, "minus1"), (1, "plus1")):
            _add(
                point_id=f"cores_wave_{tile}_{tag}_{full_wave + delta}", grid_kind="boundary_cliff",
                axis="tokens", value=full_wave + delta, regime_tag="core_wave_cliff",
                rationale=f"cores({params.core_count})*tile({tile}){delta:+d}_wave_step_cliff",
            )

    if params.topk_threshold is not None:
        tk = int(params.topk_threshold)
        _add(
            point_id=f"topk_ctx_{tk + 1}_k_{tk}", grid_kind="boundary_cliff",
            axis="context_len", value=tk + 1, regime_tag="topk_perf_cliff",
            rationale=f"legal_sparse_topk_perf_cliff:ctx={tk + 1}:topk={tk}",
            topk=tk,
        )
        _add(
            point_id=f"topk_ctx_{tk - 1}_k_{tk - 1}", grid_kind="boundary_cliff",
            axis="context_len", value=tk - 1, regime_tag="topk_pre_threshold",
            rationale=f"legal_sparse_topk_below_threshold:ctx={tk - 1}:topk={tk - 1}",
            topk=tk - 1,
        )
        _add(
            point_id=f"topk_attr_{tk + 1}", grid_kind="boundary_cliff",
            axis="context_len", value=tk + 1, regime_tag="topk_threshold_plus1",
            rationale=f"topk_threshold({tk})+1_abi_alignment_guard",
            topk=tk + 1,
        )

    if params.bytes_per_token > 0:
        l2_knee = _clamp_in_domain(params.l2_bytes // params.bytes_per_token, low=2)
        for delta, tag in ((-1, "minus1"), (1, "plus1")):
            _add(
                point_id=f"l2_knee_{tag}_{l2_knee + delta}", grid_kind="boundary_cliff",
                axis="context_len", value=l2_knee + delta, regime_tag="l2_capacity_knee",
                rationale=f"l2_capacity_knee({l2_knee}){delta:+d}_residency_cliff",
            )

    body = {
        "schema": CLIFF_PLAN_SCHEMA,
        "target_id": params.operator_id,
        "sku": sku,
        "campaign_id": clean_campaign,
        "cold_ring_l2_multiplier": COLD_RING_L2_MULTIPLIER,
        "train_points": [pt.to_dict() for pt in train_points],
        "heldout_points": [pt.to_dict() for pt in heldout],
        "abi_rejection_points": [pt.to_dict() for pt in abi_rejections],
        "tiling_axes": axes.to_dict(),
    }
    plan_sha256 = hashlib.sha256(canonical_json_bytes(body)).hexdigest()
    short = plan_sha256[:SHORT_DIGEST_LEN]
    plan = CliffBenchPlan(
        schema=CLIFF_PLAN_SCHEMA,
        plan_id=f"plan-{clean_sku}-{short}",
        target_id=params.operator_id,
        sku=sku,
        train_lineage_root=f"acq-train-{clean_campaign}-{short}",
        heldout_lineage_root=f"acq-heldout-{clean_campaign}-{short}",
        cold_ring_l2_multiplier=COLD_RING_L2_MULTIPLIER,
        train_points=tuple(train_points),
        heldout_points=tuple(heldout),
        plan_sha256=plan_sha256,
        abi_rejection_points=tuple(abi_rejections),
        tiling_axes=axes,
    )
    assert_sanitized_payload(plan.to_dict(), field_name="cliff_bench_plan")
    return plan


def synthesize_sku_512_400k_plan(
    *,
    sku: str,
    core_count: int,
    l2_bytes: int,
    tile_dim: int = _DEFAULT_SKU_TILE_DIM,
    topk_threshold: int = DEFAULT_SPARSE_TOPK,
    campaign_id: str = "onboard-sku",
) -> CliffBenchPlan:
    """Synthesize canonical 512..400K disjoint Anchor + 2D Cliff Heldout plan for a SKU."""
    params = OperatorCliffParameters(
        operator_id=f"sku:{sku}:512-400k",
        family="sku_suite",
        tile_dim=int(tile_dim),
        core_count=max(1, int(core_count)),
        l2_bytes=max(1, int(l2_bytes)),
        topk_threshold=int(topk_threshold),
        context_min=CONTEXT_MIN_TOKENS,
        context_max=CONTEXT_MAX_TOKENS,
    )
    return synthesize_operator_cliff_plan(params, sku=sku, campaign_id=campaign_id)


__all__ = [
    "STANDARD_LEVEL_BOUNDARY_KINDS",
    "WhiteboxSpineRunner",
    "score_sealed_gate_cases",
    "synthesize_operator_cliff_plan",
    "synthesize_sku_512_400k_plan",
    "validate_disjoint_lineage",
]
