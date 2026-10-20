"""Finite online-softmax attention tile pipeline.

[Role]
  Lower a single-consumer QK -> softmax -> PV chain into bounded query, KV and
  head tiles.  Each KV tile performs QK, a running-max/sum/alpha softmax update,
  PV, and an accumulator update; the last tile is normalized and written.  The
  plan exposes finite KV and query/head slot rings with real reuse edges.

[Invariants]
  * Useful QK and PV FLOPs and the original softmax vector work are preserved.
    Online-softmax accumulator rescaling and final normalization are charged as
    explicit extra vector work; score/probability/PV partials remain on chip.
  * Exact query/KV/head tails come from the actual final slice extents.  No
    blanket 64-element assumption is used for a legal tail.
  * A KV slot is reused only after the accumulator update that consumes its
    score, probability, PV partial and alpha.  Query/head group slots are reused
    only after final output write.  These edges are task dependencies, not
    after-the-fact metadata.
  * Admission is one/two finite KV slots plus finite group slots under the
    declared memory budget; total context length does not enlarge the pool.
  * The tile graph is executed by the shared Cube/Vector/MTE/HBM engine.  The
    plan does not carry a scalar speedup or serial roofline substitute.

[Timeouts & Locks]
  Pure construction bounded by emitted tiling stages.

[Side-Effect Policy]
  No I/O, device, global mutation, or scheduler engine.  Buffer ownership is
  represented only in the returned plan.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .a3_fusion_common import (
    MemoryBudget, PipelinePlan, add_release_join, ceil_div, pressure,
    tile_sizes,
)
from .a3_types import dtype_bytes


@dataclass(frozen=True)
class AttentionTiling:
    q_tile: int
    kv_tile: int
    head_tile: int
    kv_slots: int
    group_slots: int
    q_tiles: int
    kv_tiles: int
    head_tiles: int
    required_by_pool: Mapping[str, float]
    stage_estimate: int


@dataclass(frozen=True)
class AttentionRequest:
    q_len: int
    kv_len: int
    heads: int
    q_dim: int
    v_dim: int
    dtype: str
    qk: Any
    soft: Any
    pv: Any


def _request(qk: Any, soft: Any, pv: Any) -> AttentionRequest | None:
    qk_shape = tuple(int(value) for value in getattr(qk, "launch_shape", ()))
    pv_shape = tuple(int(value) for value in getattr(pv, "launch_shape", ()))
    if len(qk_shape) < 4 or len(pv_shape) < 4:
        return None
    q_len, kv_len, heads, q_dim = (
        max(1, qk_shape[0]), max(1, qk_shape[1]), max(1, qk_shape[2]),
        max(1, qk_shape[3]),
    )
    v_dim = max(1, pv_shape[3])
    return AttentionRequest(
        q_len=q_len, kv_len=kv_len, heads=heads, q_dim=q_dim,
        v_dim=v_dim, dtype=str(getattr(qk, "dtype", "") or "bf16"),
        qk=qk, soft=soft, pv=pv,
    )


def _tiling_pressure(request: AttentionRequest, tiling: AttentionTiling,
                     *, physical: bool = False) -> dict[str, float]:
    b = float(dtype_bytes(request.dtype))
    q = tiling.q_tile * tiling.head_tile
    kv = tiling.kv_tile * tiling.head_tile
    score = tiling.q_tile * tiling.kv_tile * tiling.head_tile * 4.0
    pv = tiling.q_tile * tiling.head_tile * request.v_dim * 4.0
    slots = {
        "l1": kv * request.q_dim * b + kv * request.v_dim * b,
        "ub": score + score,
        "l0c": pv,
    }
    groups = {
        "l1": q * request.q_dim * b,
        "ub": q * request.v_dim * 4.0 + q * 4.0 + q * 4.0 + q * 4.0,
    }
    if physical:
        # L0A/L0B MMAD operands and L0C score/PV outputs are bounded per slot;
        # probability is cast to BF16 UB before entering L0A, and the final
        # output is cast to a BF16 UB buffer before the HBM write.
        slots["l0a"] = max(
            tiling.q_tile * tiling.head_tile * request.q_dim * b,
            tiling.q_tile * tiling.kv_tile * tiling.head_tile * b,
        )
        slots["l0b"] = max(
            tiling.kv_tile * tiling.head_tile * request.q_dim * b,
            tiling.kv_tile * tiling.head_tile * request.v_dim * b,
        )
        slots["l0c"] = max(
            tiling.q_tile * tiling.kv_tile * tiling.head_tile * 4.0,
            pv,
        )
        slots["ub"] += tiling.q_tile * tiling.kv_tile * tiling.head_tile * b
        # Explicit pv_partial_ub copy lives from pv.copy through the
        # accumulator update; it coexists with running output/state.
        slots["ub"] += pv
        groups["ub"] += q * request.v_dim * b
    return pressure(
        slots, tiling.kv_slots, groups=groups, group_count=tiling.group_slots,
    )


def _stage_estimate(r: AttentionRequest, t: AttentionTiling) -> int:
    return (
        t.q_tiles * t.kv_tiles * t.head_tiles * 5
        + t.q_tiles * t.head_tiles * 2
    )


def _choose_tiling(
    request: AttentionRequest, budget: MemoryBudget, *,
    physical: bool = False,
) -> tuple[AttentionTiling | None, dict[str, Any]]:
    q_candidates = tile_sizes(
        request.q_len, (512, 256, 128, 64, 32, 16, 8, 4, 2),
    )
    kv_candidates = tile_sizes(
        request.kv_len, (1024, 512, 256, 128, 64, 32, 16, 8, 4, 2),
    )
    head_candidates = tile_sizes(request.heads, (8, 4, 2))
    best: tuple[float, AttentionTiling] | None = None
    minimal: dict[str, Any] | None = None
    b = float(dtype_bytes(request.dtype))
    for kv_slots in (2, 1):
        for group_slots in (2, 1):
            for q_tile in q_candidates:
                for kv_tile in kv_candidates:
                    for head_tile in head_candidates:
                        tiling = AttentionTiling(
                            q_tile=q_tile, kv_tile=kv_tile, head_tile=head_tile,
                            kv_slots=kv_slots, group_slots=group_slots,
                            q_tiles=ceil_div(request.q_len, q_tile),
                            kv_tiles=ceil_div(request.kv_len, kv_tile),
                            head_tiles=ceil_div(request.heads, head_tile),
                            required_by_pool={}, stage_estimate=0,
                        )
                        required = _tiling_pressure(
                            request, tiling, physical=physical,
                        )
                        estimate = _stage_estimate(request, tiling)
                        tiling = AttentionTiling(
                            **{**tiling.__dict__, "required_by_pool": required,
                               "stage_estimate": estimate},
                        )
                        if minimal is None or sum(required.values()) < sum(
                            minimal.get("required_by_pool", {}).values()
                        ):
                            minimal = {
                                "required_by_pool": required,
                                "q_tile": q_tile, "kv_tile": kv_tile,
                                "head_tile": head_tile,
                                "kv_slots": kv_slots,
                                "group_slots": group_slots,
                            }
                        details = budget.details(required)
                        if not details["fits"]:
                            continue
                        # Prefer two KV slots (real overlap) but never let a
                        # one-slot fallback hide a legal long-context tile if
                        # its emission cost is drastically smaller.
                        penalty = 1.0 if kv_slots > 1 else 3.0
                        score = float(estimate) * penalty
                        if best is None or score < best[0]:
                            best = (score, tiling)
    if best is None:
        return None, {
            "reason": "live_buffer_capacity_exceeded",
            "minimal_required_by_pool": dict(
                (minimal or {}).get("required_by_pool", {})
            ),
            "minimal_tiling": minimal,
            "budget": budget.details((minimal or {}).get("required_by_pool", {})),
        }
    return best[1], {}


def _tile_offset(index: int, tile: int) -> int:
    return int(index) * int(tile)


def _fused_hbm(request: AttentionRequest) -> float:
    qk, soft, pv = request.qk, request.soft, request.pv
    value = (
        float(getattr(qk, "hbm_bytes", 0.0) or 0.0)
        + float(getattr(pv, "hbm_bytes", 0.0) or 0.0)
        - float(getattr(qk, "mte3_bytes", 0.0) or 0.0)
        - float(getattr(pv, "mte2_bytes", 0.0) or 0.0)
        - float(getattr(soft, "hbm_bytes", 0.0) or 0.0)
    )
    return float(max(0.0, value))


def plan_attention_pipeline(
    qk: Any, soft: Any, pv: Any, *, budget: MemoryBudget,
    physical: bool = False,
) -> tuple[PipelinePlan | None, dict[str, Any]]:
    """Build a finite online-softmax pipeline plan or an exact rejection."""
    request = _request(qk, soft, pv)
    if request is None:
        return None, {"reason": "attention_extent_missing"}
    tiling, rejection = _choose_tiling(request, budget, physical=physical)
    if tiling is None:
        return None, rejection

    r = request
    qk_flops = float(getattr(qk, "flops", 0.0) or 0.0)
    pv_flops = float(getattr(pv, "flops", 0.0) or 0.0)
    soft_ops = float(getattr(soft, "vector_ops", 0.0) or 0.0)
    qk_mte2 = float(max(0.0, getattr(qk, "mte2_bytes", 0.0) or 0.0))
    pv_mte2 = float(max(0.0, getattr(pv, "mte2_bytes", 0.0) or 0.0))
    soft_mte2 = float(max(0.0, getattr(soft, "mte2_bytes", 0.0) or 0.0))
    b = float(dtype_bytes(r.dtype))
    total_q_elements = float(r.q_len * r.heads * r.q_dim)
    total_kv_elements = float(r.kv_len * r.heads * r.q_dim)
    total_out_elements = float(r.q_len * r.heads * r.v_dim)
    total_score = float(r.q_len * r.kv_len * r.heads)
    physical_q = total_q_elements * b
    physical_k = float(r.kv_len * r.heads * r.q_dim) * b
    physical_v = float(r.kv_len * r.heads * r.v_dim) * b
    physical_out = total_out_elements * b
    qk_hbm = float(max(0.0, getattr(qk, "hbm_bytes", 0.0) or 0.0))
    pv_hbm = float(max(0.0, getattr(pv, "hbm_bytes", 0.0) or 0.0))
    soft_hbm = float(max(0.0, getattr(soft, "hbm_bytes", 0.0) or 0.0))
    pv_mte3 = float(max(0.0, getattr(pv, "mte3_bytes", 0.0) or 0.0))
    declared_io = (
        qk_mte2 + pv_mte2 + pv_mte3 + qk_hbm + pv_hbm + soft_hbm
        + float(max(0.0, getattr(qk, "mte3_bytes", 0.0) or 0.0))
    )
    if declared_io > 0.0:
        # Q/K/V are independent required reads.  qk.mte2 covers Q+K (including
        # cache/layout overhead); PV's V read is not implied by qk.mte2 and
        # must be retained.  Only the score/probability intermediate (write +
        # PV read) is removable.
        score_write = float(max(0.0, getattr(qk, "mte3_bytes", 0.0) or 0.0))
        score_read = min(pv_mte2, max(score_write, soft_mte2))
        q_load_total = physical_q
        k_load_total = max(physical_k, max(0.0, qk_mte2 - physical_q))
        v_load_total = max(physical_v, max(0.0, pv_mte2 - score_read))
        out_bytes = max(pv_mte3, physical_out)
        fused_mte2 = q_load_total + k_load_total + v_load_total
        hbm_from_ops = qk_hbm + pv_hbm - score_write - score_read
        fused_hbm = max(fused_mte2 + out_bytes, float(max(0.0, hbm_from_ops)))
    else:
        # A deliberately normalized zero-traffic fixture models no HBM path.
        q_load_total = k_load_total = v_load_total = 0.0
        out_bytes = 0.0
        fused_mte2 = 0.0
        fused_hbm = 0.0
    kv_load_total = k_load_total + v_load_total
    load_units = fused_mte2
    output_units = out_bytes
    load_hbm_pool = max(0.0, fused_hbm - output_units)
    output_hbm_pool = output_units
    # The old coarse softmax vector count is not a conservation law.  Charge
    # actual primitive work: max/exp/sum per score element for each update,
    # one normalization divide per output element, and one initial copy per
    # output element.  These may make the fused total larger than the old sum.
    SOFTMAX_OPS_PER_SCORE = 3.0
    soft_update_pool = max(soft_ops, SOFTMAX_OPS_PER_SCORE * total_score)
    norm_work_total = float(total_out_elements)

    plan = PipelinePlan(kind="attention_online_softmax")
    q_tiles = tiling.q_tiles
    kv_tiles = tiling.kv_tiles
    head_tiles = tiling.head_tiles
    kv_slots = tiling.kv_slots
    group_slots = tiling.group_slots
    slot_free: list[int | None] = [None] * kv_slots
    group_free: list[int | None] = [None] * group_slots
    global_iteration = 0
    stage_estimate = _stage_estimate(r, tiling)

    for q_index in range(q_tiles):
        q_offset = _tile_offset(q_index, tiling.q_tile)
        q_size = min(tiling.q_tile, r.q_len - q_offset)
        for h_index in range(head_tiles):
            h_offset = _tile_offset(h_index, tiling.head_tile)
            h_size = min(tiling.head_tile, r.heads - h_offset)
            group_index = q_index * head_tiles + h_index
            group_slot = group_index % group_slots
            group_start_dep = group_free[group_slot]
            owner = f"q{q_offset}.h{h_offset}"
            q_bytes = q_size * h_size * r.q_dim * b
            run_bytes = q_size * h_size * r.v_dim * 4.0
            state_bytes = q_size * h_size * 4.0
            group_q = plan.add_buffer(
                f"attn.q.{owner}", size_bytes=q_bytes, pool="l1", owner=owner,
                shape=(q_size, h_size, r.q_dim), dtype=r.dtype, slot=group_slot,
                category="query_tile",
            )
            group_run = plan.add_buffer(
                f"attn.run.{owner}", size_bytes=run_bytes, pool="ub", owner=owner,
                shape=(q_size, h_size, r.v_dim), dtype="fp32", slot=group_slot,
                category="running_output",
            )
            group_max = plan.add_buffer(
                f"attn.max.{owner}", size_bytes=state_bytes, pool="ub", owner=owner,
                shape=(q_size, h_size), dtype="fp32", slot=group_slot,
                category="running_max",
            )
            group_sum = plan.add_buffer(
                f"attn.sum.{owner}", size_bytes=state_bytes, pool="ub", owner=owner,
                shape=(q_size, h_size), dtype="fp32", slot=group_slot,
                category="running_sum",
            )
            group_alpha = plan.add_buffer(
                f"attn.alpha.{owner}", size_bytes=state_bytes, pool="ub",
                owner=owner, shape=(q_size, h_size), dtype="fp32",
                slot=group_slot, category="alpha",
            )
            group_out = plan.add_buffer(
                f"attn.out.{owner}",
                size_bytes=q_size * h_size * r.v_dim * b, pool="ub",
                owner=owner, shape=(q_size, h_size, r.v_dim), dtype=r.dtype,
                slot=group_slot, category=f"output_{r.dtype}",
            ) if physical else None
            prev_soft: int | None = None
            prev_accum: int | None = None
            prev_qk: int | None = None
            last_qk: int | None = None
            last_soft: int | None = None
            q_load_stage: int | None = None
            q_created = False
            run_created = False
            state_created = False
            qk_consumers: list[int] = []
            soft_consumers: list[int] = []
            accum_consumers: list[int] = []
            run_consumers: list[int] = []
            for kv_index in range(kv_tiles):
                kv_offset = _tile_offset(kv_index, tiling.kv_tile)
                kv_size = min(tiling.kv_tile, r.kv_len - kv_offset)
                slot = global_iteration % kv_slots
                slot_dep = slot_free[slot]
                deps = tuple(
                    item for item in (slot_dep, group_start_dep) if item is not None
                )
                prefetch = plan.add_stage(
                    f"prefetch.{owner}.k{kv_offset}",
                    "attention_prefetch", deps=deps,
                    launch=False,
                    kind="prefetch",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                    },
                    tile_shape=(q_size, kv_size),
                )
                # Q is loaded once per query/head group, then read by every QK
                # stage in the group.
                if not q_created:
                    q_share = q_load_total * (q_size * h_size) / max(1.0, float(r.q_len * r.heads))
                    q_load_stage = plan.add_stage(
                        f"load.q.{owner}", "attention_load",
                        deps=(prefetch,), kind="q_load",
                        extents={
                            "q": [q_offset, q_size], "head": [h_offset, h_size],
                            "q_dim": r.q_dim,
                        },
                        tile_shape=(q_size, h_size),
                        mte2_bytes=q_share,
                        hbm_bytes=(
                            load_hbm_pool * (q_share / load_units)
                            if load_units > 0 else 0.0
                        ),
                        creates=(group_q,), writes=(group_q,),
                    )
                    q_created = True
                k_id = plan.add_buffer(
                    f"attn.k.{owner}.k{kv_offset}", size_bytes=kv_size * h_size * r.q_dim * b,
                    pool="l1", owner=owner, shape=(kv_size, h_size, r.q_dim),
                    dtype=r.dtype, slot=slot, category="kv_key",
                )
                v_id = plan.add_buffer(
                    f"attn.v.{owner}.k{kv_offset}", size_bytes=kv_size * h_size * r.v_dim * b,
                    pool="l1", owner=owner, shape=(kv_size, h_size, r.v_dim),
                    dtype=r.dtype, slot=slot, category="kv_value",
                )
                score_id = plan.add_buffer(
                    f"attn.score.{owner}.k{kv_offset}",
                    size_bytes=q_size * kv_size * h_size * 4.0,
                    pool="ub", owner=owner, shape=(q_size, kv_size, h_size),
                    dtype="fp32", slot=slot, category="score_tile",
                )
                prob_id = plan.add_buffer(
                    f"attn.prob.{owner}.k{kv_offset}",
                    size_bytes=q_size * kv_size * h_size * 4.0,
                    pool="ub", owner=owner, shape=(q_size, kv_size, h_size),
                    dtype="fp32", slot=slot, category="probability_tile",
                )
                if physical:
                    prob_bf16 = plan.add_buffer(
                        f"attn.prob_operand.{owner}.k{kv_offset}",
                        size_bytes=q_size * kv_size * h_size * b,
                        pool="ub", owner=owner, shape=(q_size, kv_size, h_size),
                        dtype=r.dtype, slot=slot, category="probability_operand",
                    )
                    pv_l0c = plan.add_buffer(
                        f"attn.pv_l0c.{owner}.k{kv_offset}",
                        size_bytes=q_size * h_size * r.v_dim * 4.0,
                        pool="l0c", owner=owner, shape=(q_size, h_size, r.v_dim),
                        dtype="fp32", slot=slot, category="pv_l0c",
                    )
                    pv_id = plan.add_buffer(
                        f"attn.pv_partial.{owner}.k{kv_offset}",
                        size_bytes=q_size * h_size * r.v_dim * 4.0,
                        pool="ub", owner=owner, shape=(q_size, h_size, r.v_dim),
                        dtype="fp32", slot=slot, category="pv_partial_ub",
                    )
                else:
                    pv_id = plan.add_buffer(
                        f"attn.pv_partial.{owner}.k{kv_offset}",
                        size_bytes=q_size * h_size * r.v_dim * 4.0,
                        pool="l0c", owner=owner, shape=(q_size, h_size, r.v_dim),
                        dtype="fp32", slot=slot, category="pv_partial",
                    )
                kv_share_mte = kv_load_total * (kv_size * h_size) / max(
                    1.0, float(r.kv_len * r.heads)
                )
                kv_load_stage = plan.add_stage(
                    f"prefetch.kv.{owner}.k{kv_offset}", "attention_load",
                    deps=(prefetch,), kind="kv_load",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                    },
                    tile_shape=(q_size, kv_size),
                    mte2_bytes=kv_share_mte,
                    hbm_bytes=(
                        load_hbm_pool * (kv_share_mte / load_units)
                        if load_units > 0 else 0.0
                    ),
                    creates=(k_id, v_id), writes=(k_id, v_id),
                )
                qk_weight = q_size * kv_size * h_size
                if physical:
                    q_l0a = plan.add_buffer(
                        f"qk.l0a.{owner}.k{kv_offset}",
                        size_bytes=q_size * h_size * r.q_dim * b, pool="l0a",
                        owner=owner, shape=(q_size, h_size, r.q_dim),
                        dtype=r.dtype, slot=slot, category="qk_l0a",
                    )
                    k_l0b = plan.add_buffer(
                        f"qk.l0b.{owner}.k{kv_offset}",
                        size_bytes=kv_size * h_size * r.q_dim * b, pool="l0b",
                        owner=owner, shape=(kv_size, h_size, r.q_dim),
                        dtype=r.dtype, slot=slot, category="qk_l0b",
                    )
                    score_l0c = plan.add_buffer(
                        f"attn.score_l0c.{owner}.k{kv_offset}",
                        size_bytes=q_size * kv_size * h_size * 4.0, pool="l0c",
                        owner=owner, shape=(q_size, kv_size, h_size),
                        dtype="fp32", slot=slot, category="score_l0c",
                    )
                    qk_reads = (group_q, k_id, q_l0a, k_l0b)
                    qk_writes = (score_l0c,)
                    qk_creates = (q_l0a, k_l0b, score_l0c)
                    qk_releases = (k_id, q_l0a, k_l0b)
                    qk_resources = {
                        "mte1": float(q_size * h_size * r.q_dim * b
                                      + kv_size * h_size * r.q_dim * b),
                    }
                else:
                    qk_reads = (group_q, k_id)
                    qk_writes = (score_id,)
                    qk_creates = (score_id,)
                    qk_releases = (k_id,)
                    qk_resources = {}
                qk_stage = plan.add_stage(
                    f"qk.{owner}.k{kv_offset}", "attention_qk",
                    deps=tuple(dict.fromkeys(
                        item for item in (kv_load_stage, q_load_stage, prev_qk)
                        if item is not None
                    )), launch=(global_iteration == 0), kind="qk",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                        "l0a_bytes": q_size * h_size * r.q_dim * b if physical else 0.0,
                        "l0b_bytes": kv_size * h_size * r.q_dim * b if physical else 0.0,
                        "l0a_pool": "l0a", "l0b_pool": "l0b",
                        "l0c_pool": "l0c", "src_dtype": r.dtype,
                    },
                    tile_shape=(q_size, kv_size),
                    flops=qk_flops * qk_weight / max(1.0, total_score),
                    reads=qk_reads, writes=qk_writes, creates=qk_creates,
                    releases=qk_releases, resources=qk_resources,
                )
                if physical:
                    score_ready = plan.add_stage(
                        f"qk.copy.{owner}.k{kv_offset}", "attention_qk",
                        deps=(qk_stage,), kind="l0c_to_ub_score",
                        extents={
                            "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                            "head": [h_offset, h_size],
                            "src_pool": "l0c", "dst_pool": "ub",
                            "src_dtype": "fp32", "dst_dtype": "fp32",
                        },
                        tile_shape=(q_size, kv_size),
                        vector_ops=float(q_size * kv_size * h_size),
                        reads=(score_l0c,), writes=(score_id,),
                        creates=(score_id,), releases=(score_l0c,),
                    )
                else:
                    score_ready = qk_stage
                last_qk = qk_stage
                prev_qk = qk_stage
                qk_consumers.append(qk_stage)
                soft_deps = tuple(
                    dict.fromkeys(
                        (score_ready, prev_soft, prev_accum)
                        if prev_soft is not None and prev_accum is not None
                        else (score_ready,)
                    )
                )
                creates_soft = [prob_id]
                if not state_created:
                    creates_soft.extend([group_max, group_sum, group_alpha])
                soft_stage = plan.add_stage(
                    (
                        f"softmax.q{q_offset}" if kv_index == 0 and h_index == 0
                        else f"softmax_update.{owner}.k{kv_offset}"
                    ),
                    "attention_softmax_update", deps=soft_deps, kind="softmax_update",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                    },
                    tile_shape=(q_size, kv_size),
                    vector_ops=(
                        soft_update_pool * qk_weight / max(1.0, total_score)
                    ),
                    reads=(score_id, group_max, group_sum, group_alpha),
                    writes=(prob_id, group_max, group_sum, group_alpha),
                    creates=tuple(creates_soft), releases=(score_id,),
                )
                if not state_created:
                    state_created = True
                last_soft = soft_stage
                soft_consumers.append(soft_stage)
                if physical:
                    prob_ready = plan.add_stage(
                        f"prob.cast.{owner}.k{kv_offset}", "attention_cast",
                        deps=(soft_stage,), kind=f"fp32_prob_to_{r.dtype}",
                        extents={
                            "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                            "head": [h_offset, h_size],
                            "src_dtype": "fp32", "dst_dtype": r.dtype,
                        },
                        tile_shape=(q_size, kv_size),
                        vector_ops=float(q_size * kv_size * h_size),
                        reads=(prob_id,), writes=(prob_bf16,),
                        creates=(prob_bf16,), releases=(prob_id,),
                    )
                    pv_prob = prob_bf16
                else:
                    prob_ready = soft_stage
                    pv_prob = prob_id
                if physical:
                    pv_l0a = plan.add_buffer(
                        f"pv.l0a.{owner}.k{kv_offset}",
                        size_bytes=q_size * kv_size * h_size * b, pool="l0a",
                        owner=owner, shape=(q_size, kv_size, h_size),
                        dtype=r.dtype, slot=slot, category="pv_l0a",
                    )
                    pv_l0b = plan.add_buffer(
                        f"pv.l0b.{owner}.k{kv_offset}",
                        size_bytes=kv_size * h_size * r.v_dim * b, pool="l0b",
                        owner=owner, shape=(kv_size, h_size, r.v_dim),
                        dtype=r.dtype, slot=slot, category="pv_l0b",
                    )
                    pv_reads = (pv_prob, v_id, pv_l0a, pv_l0b)
                    pv_writes = (pv_l0c,)
                    pv_creates = (pv_l0a, pv_l0b, pv_l0c)
                    pv_releases = (pv_prob, v_id, pv_l0a, pv_l0b)
                    pv_resources = {
                        "mte1": float(q_size * kv_size * h_size * b
                                      + kv_size * h_size * r.v_dim * b),
                    }
                else:
                    pv_reads = (prob_id, v_id)
                    pv_writes = (pv_id,)
                    pv_creates = (pv_id,)
                    pv_releases = (prob_id, v_id)
                    pv_resources = {}
                pv_stage = plan.add_stage(
                    f"pv.{owner}.k{kv_offset}", "attention_pv",
                    deps=(prob_ready,), kind="pv",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                        "l0a_bytes": q_size * kv_size * h_size * b if physical else 0.0,
                        "l0b_bytes": kv_size * h_size * r.v_dim * b if physical else 0.0,
                        "l0a_pool": "l0a", "l0b_pool": "l0b",
                        "l0c_pool": "l0c",
                        "src_dtype": r.dtype,
                    },
                    tile_shape=(q_size, kv_size),
                    flops=pv_flops * qk_weight / max(1.0, total_score),
                    reads=pv_reads, writes=pv_writes, creates=pv_creates,
                    releases=pv_releases, resources=pv_resources,
                )
                if physical:
                    pv_ready = plan.add_stage(
                        f"pv.copy.{owner}.k{kv_offset}", "attention_pv",
                        deps=(pv_stage,), kind="l0c_to_ub_pv",
                        extents={
                            "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                            "head": [h_offset, h_size],
                            "src_pool": "l0c", "dst_pool": "ub",
                            "src_dtype": "fp32", "dst_dtype": "fp32",
                        },
                        tile_shape=(q_size, kv_size),
                        vector_ops=float(q_size * h_size * r.v_dim),
                        reads=(pv_l0c,), writes=(pv_id,),
                        creates=(pv_id,), releases=(pv_l0c,),
                    )
                else:
                    pv_ready = pv_stage
                acc_deps = tuple(
                    dict.fromkeys((pv_ready, prev_accum) if prev_accum is not None
                                  else (pv_ready,))
                )
                # The first accumulator update copies the initial PV partial
                # into the running output; later updates rescale and add.
                accum_ops = (
                    2.0 * q_size * h_size * r.v_dim if kv_index > 0
                    else 1.0 * q_size * h_size * r.v_dim
                )
                run_actions = [group_run]
                writes_run = (group_run,)
                creates_acc = (group_run,) if not run_created else ()
                acc_stage = plan.add_stage(
                    f"accumulator_update.{owner}.k{kv_offset}",
                    "attention_accumulator", deps=acc_deps,
                    kind="accumulator_update",
                    extents={
                        "q": [q_offset, q_size], "kv": [kv_offset, kv_size],
                        "head": [h_offset, h_size], "q_dim": r.q_dim,
                        "v_dim": r.v_dim, "kv_tail": kv_size != tiling.kv_tile,
                    },
                    tile_shape=(q_size, kv_size),
                    vector_ops=accum_ops,
                    reads=tuple(run_actions) + (pv_id, group_alpha),
                    writes=writes_run,
                    creates=creates_acc,
                    releases=(pv_id,),
                )
                if not run_created:
                    run_created = True
                accum_consumers.append(acc_stage)
                run_consumers.append(acc_stage)
                prev_soft = soft_stage
                prev_accum = acc_stage
                slot_free[slot] = acc_stage
                global_iteration += 1

            assert prev_accum is not None and last_qk is not None and last_soft is not None
            normalize_stage = plan.add_stage(
                f"final_normalize.{owner}", "attention_normalize",
                deps=(prev_accum,), kind="normalize",
                extents={
                    "q": [q_offset, q_size], "head": [h_offset, h_size],
                    "q_dim": r.q_dim, "v_dim": r.v_dim,
                },
                tile_shape=(q_size, r.v_dim),
                vector_ops=(
                    norm_work_total * q_size * h_size * r.v_dim
                    / max(1.0, total_out_elements)
                ),
                reads=(group_run, group_sum), writes=(group_run,),
            )
            output_share = out_bytes * q_size * h_size * r.v_dim / max(
                1.0, total_out_elements
            )
            if physical:
                output_cast = plan.add_stage(
                    f"output.cast.{owner}", "attention_cast",
                    deps=(normalize_stage,), kind=f"fp32_output_to_{r.dtype}",
                    extents={
                        "q": [q_offset, q_size], "head": [h_offset, h_size],
                        "q_dim": r.q_dim, "v_dim": r.v_dim,
                        "src_dtype": "fp32", "dst_dtype": r.dtype,
                    },
                    tile_shape=(q_size, h_size, r.v_dim),
                    vector_ops=float(q_size * h_size * r.v_dim),
                    reads=(group_run,), writes=(group_out,),
                    creates=(group_out,), releases=(group_run,),
                )
                write_input = group_out
                write_deps = (output_cast,)
            else:
                write_input = group_run
                write_deps = (normalize_stage,)
            output_write = plan.add_stage(
                f"output_write.{owner}", "attention_write",
                deps=write_deps, kind="output_write",
                extents={
                    "q": [q_offset, q_size], "head": [h_offset, h_size],
                    "q_dim": r.q_dim, "v_dim": r.v_dim,
                    "dtype": r.dtype,
                },
                tile_shape=(q_size, h_size, r.v_dim),
                mte3_bytes=output_share,
                hbm_bytes=(
                    output_hbm_pool * q_size * h_size * r.v_dim
                    / max(1.0, total_out_elements)
                ),
                reads=(write_input,), writes=(),
                releases=(write_input,) if physical else (),
            )
            # Slot release is a join of every fan-out consumer, not the
            # highest stage index: parallel consumers can finish later.
            add_release_join(
                plan, name=f"release.q.{owner}", family="attention_release",
                consumers=qk_consumers, releases=(group_q,),
                kind="q_slot_release_join",
                extents={"q": [q_offset, q_size], "head": [h_offset, h_size]},
            )
            add_release_join(
                plan, name=f"release.max.{owner}", family="attention_release",
                consumers=soft_consumers, releases=(group_max,),
                kind="max_slot_release_join",
                extents={"q": [q_offset, q_size], "head": [h_offset, h_size]},
            )
            add_release_join(
                plan, name=f"release.alpha.{owner}", family="attention_release",
                consumers=tuple(soft_consumers) + tuple(accum_consumers),
                releases=(group_alpha,), kind="alpha_slot_release_join",
                extents={"q": [q_offset, q_size], "head": [h_offset, h_size]},
            )
            add_release_join(
                plan, name=f"release.sum.{owner}", family="attention_release",
                consumers=tuple(soft_consumers) + (normalize_stage,),
                releases=(group_sum,), kind="sum_slot_release_join",
                extents={"q": [q_offset, q_size], "head": [h_offset, h_size]},
            )
            if physical:
                group_free[group_slot] = output_write
            else:
                run_release = add_release_join(
                    plan, name=f"release.run.{owner}", family="attention_release",
                    consumers=tuple(run_consumers) + (normalize_stage, output_write),
                    releases=(group_run,), kind="run_slot_release_join",
                    extents={
                        "q": [q_offset, q_size], "head": [h_offset, h_size],
                        "v_dim": r.v_dim,
                    },
                )
                group_free[group_slot] = run_release

    # Exact final-slice flags travel with every stage extent.
    for stage in plan.stages:
        extents = stage["extents"]
        if "q" in extents:
            q_start, q_size = extents["q"]
            extents["q_tail"] = bool(q_size < tiling.q_tile)
        if "kv" in extents:
            kv_start, kv_size = extents["kv"]
            extents["kv_tail"] = bool(kv_size < tiling.kv_tile)
        if "head" in extents:
            head_start, head_size = extents["head"]
            extents["head_tail"] = bool(head_size < tiling.head_tile)
    # Recompute exact pressure from the declared finite slot sizes.  This is
    # the conservative admission number, independent of stage order.
    plan.finish(required_by_pool=tiling.required_by_pool)
    plan.notes.update({
        "tiling": {
            "q_tile": tiling.q_tile, "kv_tile": tiling.kv_tile,
            "head_tile": tiling.head_tile, "q_tiles": q_tiles,
            "kv_tiles": kv_tiles, "head_tiles": head_tiles,
            "kv_slots": kv_slots, "group_slots": group_slots,
            "query_tail": r.q_len % tiling.q_tile or tiling.q_tile,
            "kv_tail": r.kv_len % tiling.kv_tile or tiling.kv_tile,
            "head_tail": r.heads % tiling.head_tile or tiling.head_tile,
        },
        "stage_estimate": stage_estimate,
        "actual_stage_count": len(plan.stages),
        "finite_kv_slots": kv_slots,
        "finite_group_slots": group_slots,
        "recurrence": "max/sum/alpha/pv/accumulator_update/final_normalize",
        "reference": "online softmax with exact tail extents",
        "physical_path": bool(physical),
        "physical_primitives": (
            f"l1->l0a/l0b, mmad->l0c, l0c->ub, fp32 prob->{r.dtype} ub, "
            f"{r.dtype} mmad, fp32 output->{r.dtype} ub"
        ) if physical else "logical_unified_no_physical_l0_path",
    })
    return plan, {}


__all__ = ["AttentionRequest", "AttentionTiling", "plan_attention_pipeline"]
