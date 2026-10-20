"""Finite multiworker partition of row-independent norm/GEMM work.

[Role]
  Split complete row tiles across declared workers.  Each worker owns a
  private set of row/weight/output/L0/UB buffers and a contiguous row range;
  workers never alias a local buffer.  The merged plan exposes fixed worker
  ids and the same exact work/tails as the single-worker builder.

[Invariants]
  * Work is partitioned by complete rows; total useful FLOPs, norm vector
    work, X reads, weight loads and output bytes equal the single-worker plan.
  * A worker's local buffer ids are disjoint from every other worker's.  No
    shared normalized row is aliased across workers.
  * Per-pool admission is the maximum per-worker requirement, not a sum of
    private workers.  Physical feasibility requires each worker to fit.
  * Stage indices/dependencies are shifted once; worker id is explicit and
    later core assignment binds every stage of one worker to one core.

[Timeouts & Locks]
  Pure finite construction bounded by workers and emitted tile count.

[Side-Effect Policy]
  No I/O, scheduler, engine or global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from .a3_fusion_common import PipelinePlan, ceil_div
from .a3_fusion_attention import (
    _choose_tiling as _choose_attention_tiling,
    _request as _attention_request,
    plan_attention_pipeline,
)
from .a3_fusion_norm import (
    _choose_tiling, _plan_full_row, _plan_two_pass, _plan_single_tile,
    _request,
)


def _scale_op(op: Any, *, shape: tuple[int, ...], fraction: float) -> Any:
    return replace(
        op,
        semantic_shape=shape,
        local_shape=shape,
        padded_shape=shape,
        launch_shape=shape,
        flops=float(getattr(op, "flops", 0.0) or 0.0) * fraction,
        hbm_bytes=float(getattr(op, "hbm_bytes", 0.0) or 0.0) * fraction,
        mte2_bytes=float(getattr(op, "mte2_bytes", 0.0) or 0.0) * fraction,
        mte3_bytes=float(getattr(op, "mte3_bytes", 0.0) or 0.0) * fraction,
        vector_ops=float(getattr(op, "vector_ops", 0.0) or 0.0) * fraction,
        link_bytes=float(getattr(op, "link_bytes", 0.0) or 0.0) * fraction,
    )


def _prefix_worker(plan: PipelinePlan, *, worker: int, offset: int,
                   axis_offsets: dict[str, int] | None = None,
                   ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    axis_offsets = dict(axis_offsets or {})
    prefix = f"w{worker}."
    stages = []
    for stage in plan.stages:
        payload = dict(stage)
        payload["stage_index"] = offset + int(stage["stage_index"])
        payload["deps"] = tuple(offset + int(item) for item in stage.get("deps", ()))
        payload["name"] = prefix + str(stage["name"])
        payload["worker_id"] = worker
        extents = dict(stage.get("extents", {}))
        for axis, delta in axis_offsets.items():
            if axis in extents:
                start, size = extents[axis]
                extents[axis] = [int(delta) + int(start), int(size)]
        payload["extents"] = extents
        actions = {}
        for action, records in dict(stage.get("buffers", {})).items():
            actions[action] = [
                {**dict(record), "id": prefix + str(record["id"]),
                 "worker": worker}
                for record in records
            ]
        payload["buffers"] = actions
        stages.append(payload)
    buffers = []
    for buffer in plan.buffers:
        payload = dict(buffer)
        payload["id"] = prefix + str(buffer["id"])
        payload["worker"] = worker
        payload["owner"] = f"w{worker}." + str(buffer.get("owner", ""))
        if payload.get("created_stage") is not None:
            payload["created_stage"] = offset + int(payload["created_stage"])
        payload["consumers"] = [
            offset + int(item) for item in payload.get("consumers", ())
        ]
        if payload.get("release_stage") is not None:
            payload["release_stage"] = offset + int(payload["release_stage"])
        if payload.get("last_consumer_stage") is not None:
            payload["last_consumer_stage"] = (
                offset + int(payload["last_consumer_stage"])
            )
        if payload.get("lifetime"):
            payload["lifetime"] = [
                offset + int(item) for item in payload["lifetime"]
            ]
        buffers.append(payload)
    return stages, buffers


def plan_norm_multiworker(
    norm: Any, gemm: Any, *, budget: Any, physical: bool,
    workers: int,
) -> tuple[PipelinePlan | None, dict[str, Any]]:
    """Build a private-buffer row partition or an exact rejection."""
    request = _request(norm, gemm)
    if request is None:
        return None, {"reason": "gemm_extent_missing"}
    tiling, rejection = _choose_tiling(request, budget, physical=physical)
    if tiling is None:
        return None, rejection
    total_row_tiles = max(1, ceil_div(request.m, tiling.row_tile))
    active_workers = max(1, min(int(workers), total_row_tiles))
    if active_workers <= 1:
        if tiling.mode == "full_row":
            return _plan_full_row(request, tiling, physical=physical)
        return _plan_two_pass(request, tiling, physical=physical)
    base = 0
    quotient, remainder = divmod(total_row_tiles, active_workers)
    merged_stages: list[dict[str, Any]] = []
    merged_buffers: list[dict[str, Any]] = []
    required: dict[str, float] = {}
    worker_rows: list[dict[str, int]] = []
    all_physical = True
    for worker in range(active_workers):
        tile_count = quotient + (1 if worker < remainder else 0)
        start_row = base * tiling.row_tile
        row_count = min(request.m - start_row, tile_count * tiling.row_tile)
        fraction = float(row_count) / float(request.m)
        if row_count <= 0:
            continue
        sub_norm = _scale_op(
            norm, shape=(row_count, request.k), fraction=fraction,
        )
        sub_gemm = _scale_op(
            gemm, shape=(row_count, request.n, request.k), fraction=fraction,
        )
        sub_request = _request(sub_norm, sub_gemm)
        if sub_request is None:
            return None, {"reason": "gemm_extent_missing"}
        sub_tiling = replace(
            tiling,
            row_tiles=ceil_div(row_count, tiling.row_tile),
            n_groups=(
                ceil_div(tiling.n_tiles, tiling.group_tiles)
                if tiling.mode == "two_pass" and tiling.group_tiles else 1
            ),
        )
        if sub_tiling.row_tiles == 1 and sub_tiling.n_tiles == 1 \
                and sub_tiling.k_tiles == 1 and not physical:
            sub_plan, _ = _plan_single_tile(sub_request, sub_tiling)
        elif sub_tiling.mode == "full_row":
            sub_plan, reject = _plan_full_row(
                sub_request, sub_tiling, physical=physical,
            )
            if sub_plan is None:
                return None, reject
        else:
            sub_plan, reject = _plan_two_pass(
                sub_request, sub_tiling, physical=physical,
            )
            if sub_plan is None:
                return None, reject
        offset = len(merged_stages)
        stages, buffers = _prefix_worker(
            sub_plan, worker=worker, offset=offset,
            axis_offsets={"m": start_row},
        )
        merged_stages.extend(stages)
        merged_buffers.extend(buffers)
        all_physical = all_physical and bool(
            sub_plan.notes.get("physical_path")
        )
        for pool, amount in sub_plan.required_by_pool.items():
            required[str(pool)] = max(required.get(str(pool), 0.0), float(amount))
        worker_rows.append({
            "worker": worker, "row_start": start_row, "rows": row_count,
            "row_tiles": sub_tiling.row_tiles,
        })
        base += tile_count
    merged = PipelinePlan(
        stages=merged_stages,
        buffers=merged_buffers,
        kind="norm_gemm_multiworker",
        notes={
            "tiling": dict(tiling.__dict__),
            "workers": worker_rows,
            "active_workers": len(worker_rows),
            "partition": "contiguous_complete_rows",
            "private_local_buffers": True,
            "required_by_pool_per_worker": dict(required),
            "physical_path": bool(physical and all_physical),
            "all_worker_physical_paths": bool(all_physical),
            "physical_primitives": (
                "per-worker ub/l1/l0a/l0b/l0c path; no cross-worker buffer alias"
                if physical and all_physical
                else "logical_unified_no_physical_l0_path"
            ),
        },
    )
    merged.required_by_pool = dict(required)
    merged.notes["peak_live_bytes_per_worker"] = float(sum(required.values()))
    return merged, {}


def plan_attention_multiworker(
    qk: Any, soft: Any, pv: Any, *, budget: Any, physical: bool,
    workers: int,
) -> tuple[PipelinePlan | None, dict[str, Any]]:
    """Partition independent head slices across private-buffer workers."""
    request = _attention_request(qk, soft, pv)
    if request is None:
        return None, {"reason": "attention_extent_missing"}
    total_heads = max(1, int(request.heads))
    active_workers = max(1, min(int(workers), total_heads))
    if active_workers <= 1:
        return plan_attention_pipeline(
            qk, soft, pv, budget=budget, physical=physical,
        )
    base_tiling, _ = _choose_attention_tiling(
        request, budget, physical=physical,
    )
    quotient, remainder = divmod(total_heads, active_workers)
    merged_stages: list[dict[str, Any]] = []
    merged_buffers: list[dict[str, Any]] = []
    required: dict[str, float] = {}
    worker_heads: list[dict[str, int]] = []
    all_physical = True
    head_start = 0
    for worker in range(active_workers):
        head_count = quotient + (1 if worker < remainder else 0)
        if head_count <= 0:
            continue
        fraction = float(head_count) / float(total_heads)
        q_shape = (request.q_len, request.kv_len, head_count, request.q_dim)
        v_shape = (request.q_len, request.kv_len, head_count, request.v_dim)
        sub_qk = _scale_op(qk, shape=q_shape, fraction=fraction)
        sub_soft = _scale_op(soft, shape=q_shape, fraction=fraction)
        sub_pv = _scale_op(pv, shape=v_shape, fraction=fraction)
        sub_plan, reject = plan_attention_pipeline(
            sub_qk, sub_soft, sub_pv, budget=budget, physical=physical,
        )
        if sub_plan is None:
            return None, reject
        offset = len(merged_stages)
        stages, buffers = _prefix_worker(
            sub_plan, worker=worker, offset=offset,
            axis_offsets={"head": head_start},
        )
        merged_stages.extend(stages)
        merged_buffers.extend(buffers)
        all_physical = all_physical and bool(
            sub_plan.notes.get("physical_path")
        )
        for pool, amount in sub_plan.required_by_pool.items():
            required[str(pool)] = max(required.get(str(pool), 0.0), float(amount))
        worker_heads.append({
            "worker": worker, "head_start": head_start, "heads": head_count,
        })
        head_start += head_count
    merged = PipelinePlan(
        stages=merged_stages,
        buffers=merged_buffers,
        kind="attention_multiworker",
        notes={
            "tiling": (
                dict(base_tiling.__dict__) if base_tiling is not None else {}
            ),
            "workers": worker_heads,
            "active_workers": len(worker_heads),
            "partition": "contiguous_heads",
            "private_local_buffers": True,
            "required_by_pool_per_worker": dict(required),
            "physical_path": bool(physical and all_physical),
            "all_worker_physical_paths": bool(all_physical),
            "physical_primitives": (
                "per-worker q/k/v/ub/l0 path; no cross-worker buffer alias"
                if physical and all_physical
                else "logical_unified_no_physical_l0_path"
            ),
        },
    )
    merged.required_by_pool = dict(required)
    merged.notes["peak_live_bytes_per_worker"] = float(sum(required.values()))
    return merged, {}


__all__ = ["plan_attention_multiworker", "plan_norm_multiworker"]
