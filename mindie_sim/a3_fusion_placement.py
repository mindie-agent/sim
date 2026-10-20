"""Full-work placement of independent physical fused kernels.

[Role]
  After per-rule fusion rewrites a full work DAG, assign each physical fused
  kernel to a legal fixed worker bundle, or serialize colliding kernels on a
  reused bundle.  This prevents independent kernels from simultaneously
  holding local L1/UB/L0 buffers on the same core beyond capacity.

[Invariants]
  * A physical fused kernel keeps all its stages on its fixed worker bundle;
    within-kernel Cube/Vector/MTE overlap and buffer lifetimes are untouched.
  * If the caller supplied a fixed core_base, no kernel is silently moved.
    Independent kernels that collide on that bundle get a real predecessor
    dependency (whole prior kernel completion) before reuse.
  * Without a fixed core_base, independent kernels receive disjoint bundles
    when core_count has room; otherwise they share a bundle and are serialized.
  * An added edge is predecessor -> successor, only between kernels with no
    existing path in either direction, so semantic order cannot cycle.
  * Per-worker local capacities remain the per-kernel admission basis.

[Timeouts & Locks]
  Pure DAG/placement computation bounded by fused-op count.

[Side-Effect Policy]
  No I/O, engine, resource or global mutation.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

from .a3_ops import OpSpec


def _metadata(op: OpSpec) -> dict[str, Any]:
    return dict(op.metadata)


def _worker_width(meta: Mapping[str, Any]) -> int:
    contract = dict(meta.get("core_contract", {}) or {})
    workers = contract.get("worker_cores") or {}
    width = max((int(key) + 1 for key in workers), default=0)
    return max(1, width, int(meta.get("worker_count", 1) or 1))


def _worker_ids(meta: Mapping[str, Any], width: int) -> tuple[int, ...]:
    contract = dict(meta.get("core_contract", {}) or {})
    workers = contract.get("worker_cores") or {}
    ids = tuple(sorted(int(key) for key in workers))
    return ids if ids else tuple(range(width))


def _reachable(parents: Mapping[str, Sequence[str]], start: str, target: str) -> bool:
    if start == target:
        return True
    seen: set[str] = set()
    stack = list(parents.get(start, ()))
    while stack:
        current = stack.pop()
        if current == target:
            return True
        if current not in seen:
            seen.add(current)
            stack.extend(parents.get(current, ()))
    return False


def _rewrite_core(op: OpSpec, *, start: int, core_count: int,
                  width: int, placement_mode: str) -> OpSpec:
    meta = _metadata(op)
    workers = _worker_ids(meta, width)
    contract = dict(meta.get("core_contract", {}) or {})
    worker_cores = {
        worker: (int(start) + worker) % max(1, int(core_count))
        for worker in workers
    }
    contract.update({
        "core_base": int(start),
        "worker_cores": {str(key): value for key, value in worker_cores.items()},
        "placement_mode": placement_mode,
        "placement_bundle": int(start),
        "declared_cores": int(core_count),
        "all_stages_same_worker": len(worker_cores) <= 1,
        "workers_with_work": len(worker_cores),
    })
    stages = []
    for stage in dict(meta).get("fusion_stages", ()):
        item = dict(stage)
        assignment = dict(item.get("core_assignment", {}) or {})
        worker = int(item.get("worker_id", assignment.get("worker_id", 0)) or 0)
        if int(assignment.get("count", 0) or 0) > 0:
            core = worker_cores.get(worker, (int(start) + worker)
                                    % max(1, int(core_count)))
            assignment["core_ids"] = (core,)
            assignment["worker_id"] = worker
            assignment["partition"] = "single_core_tile"
            assignment["scope"] = "whole_worker_pipeline_fixed_core"
            item["core_assignment"] = assignment
        stages.append(item)
    meta["core_contract"] = contract
    meta["fusion_stages"] = tuple(stages)
    meta["full_work_placement"] = {
        "worker_cores": contract["worker_cores"],
        "bundle_start": int(start),
        "mode": placement_mode,
    }
    return replace(op, metadata=tuple(sorted(meta.items())))


def place_full_work_fusions(
    ops: Sequence[OpSpec], *,
    core_count: int | None,
    core_base: int | None,
    fixed_core_base: bool,
) -> tuple[list[OpSpec], dict[str, Any]]:
    """Assign/serialize full-work physical fused kernels; return updated ops."""
    items = list(ops)
    if core_count is None:
        return items, {"mode": "unplaced_no_core_metadata", "kernels": []}
    core_total = max(1, int(core_count))
    fused = [
        op for op in items
        if op.coverage == "fused"
        and bool(_metadata(op).get("physical_feasible"))
    ]
    if not fused:
        return items, {"mode": "no_physical_fused_kernel", "kernels": []}
    widths = {op.op_id: _worker_width(_metadata(op)) for op in fused}
    starts: dict[str, int] = {}
    modes: dict[str, str] = {}
    # Each physical PP stage owns an independent local-pool/allocation scope.
    # Its cursor advances around the declared core range; after the range is
    # exhausted it wraps to a legal start (core0) and subsequent kernels can
    # reuse every bundle instead of pinning all later kernels to start0.
    next_start_by_scope: dict[int, int] = {}
    if fixed_core_base:
        base = (0 if core_base is None else int(core_base)) % core_total
        for op in fused:
            starts[op.op_id] = base
            modes[op.op_id] = "fixed_core_base"
    else:
        for op in fused:
            width = min(int(widths[op.op_id]), core_total)
            scope = int(getattr(op, "stage", 0) or 0)
            cursor = next_start_by_scope.get(scope, 0)
            if cursor + width <= core_total:
                start = cursor
                mode = "disjoint_bundle"
            else:
                start = 0
                mode = "reused_bundle"
            next_start = start + width
            next_start_by_scope[scope] = (
                0 if next_start >= core_total else next_start
            )
            starts[op.op_id] = start
            modes[op.op_id] = mode
    # Existing semantic parent graph, used for cycle-safe serialization.
    parents = {op.op_id: tuple(op.deps) for op in items}
    order_index = {op.op_id: index for index, op in enumerate(items)}
    rewritten: dict[str, OpSpec] = {}
    added_edges: list[dict[str, Any]] = []
    # Last physical kernel known to use each (PP stage, physical core) scope.
    # Mixed worker widths are compared by their actual core sets, not by equal
    # bundle_start values.  Each reused core must wait for its last user; a
    # wider kernel waits for every distinct last user across its core set.
    last_by_scope_core: dict[tuple[int, int], str] = {}
    for op in sorted(fused, key=lambda item: order_index[item.op_id]):
        meta = _metadata(op)
        width = widths[op.op_id]
        start = starts[op.op_id]
        workers = _worker_ids(meta, width)
        core_set = sorted({
            (start + worker) % core_total for worker in workers
        })
        scope = int(getattr(op, "stage", 0) or 0)
        predecessors: list[str] = []
        for core in core_set:
            prior = last_by_scope_core.get((scope, core))
            if prior is not None and prior not in predecessors:
                predecessors.append(prior)
        current = op
        for prior_id in predecessors:
            if _reachable(parents, prior_id, current.op_id) or \
                    _reachable(parents, current.op_id, prior_id):
                continue
            current = replace(
                current, deps=tuple((*current.deps, prior_id)),
            )
            parents[current.op_id] = tuple(current.deps)
            added_edges.append({
                "from": prior_id, "to": current.op_id,
                "reason": "shared_physical_core_local_buffer_serialization",
                "core_set": core_set,
                "stage_scope": scope,
            })
        rewritten[current.op_id] = current
        for core in core_set:
            last_by_scope_core[(scope, core)] = current.op_id
    output: list[OpSpec] = []
    placement_rows: list[dict[str, Any]] = []
    for op in items:
        if op.op_id in starts:
            current = rewritten.get(op.op_id, op)
            current = _rewrite_core(
                current, start=starts[op.op_id], core_count=core_total,
                width=widths[op.op_id], placement_mode=modes[op.op_id],
            )
            output.append(current)
            contract = dict(_metadata(current).get("core_contract", {}) or {})
            placement_rows.append({
                "op_id": current.op_id,
                "mode": modes[op.op_id],
                "bundle_start": starts[op.op_id],
                "worker_cores": dict(contract.get("worker_cores", {})),
                "deps": list(current.deps),
            })
        else:
            output.append(op)
    return output, {
        "mode": "fixed_core_base" if fixed_core_base else "auto_disjoint",
        "core_count": core_total,
        "core_base": core_base,
        "added_serialization_edges": added_edges,
        "kernels": placement_rows,
    }


__all__ = ["place_full_work_fusions"]
