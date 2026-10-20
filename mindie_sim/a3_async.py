"""Explicit ahead-of-forward preparation on the shared resource engine.

[Role]
  Turn ``PrepareSpec`` records into resource tasks with input dependencies,
  release/known-at constraints, deadlines, lifetime metadata and one shared
  timeline with current and future forward work.

[Invariants]
  * ``known_at=startup|arrival|previous_forward`` only; unknown future token,
    routing or speculative-accept data is never used early.
  * Request-specific preparation cannot start before its request arrival.
  * Background preparation shares HBM/MTE/link with token work; it is not
    hidden by ``max(0, prep-compute)`` subtraction.
  * A later token run stops at its token-critical tasks, so a background task
    finishing later does not retroactively delay an already completed token.
  * Deadline/slack is reported from actual completion timestamps.

[Timeouts & Locks]
  Adds bounded tasks to an existing engine; no sleeps/locks.

[Side-Effect Policy]
  In-memory task registration only.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .a3_prepare_residency import PrepareResidencyLedger, ledger_for_engine
from .a3_resources import ResourceEngine, ResourceSpec
from .a3_types import A3InputError, PrepareSpec, RuntimeSpec


def add_prepare_tasks(
    engine: ResourceEngine,
    specs: Sequence[PrepareSpec],
    stage_map: Mapping[int, Mapping[str, str]],
    requests: Sequence[Any],
    *,
    now_ms: float = 0.0,
    prefix: str = "prep",
    added: set[str] | None = None,
    external_deps: Sequence[str] = (),
) -> list[str]:
    """Register preparation tasks on ``engine`` and return their task ids.

    A task keeps its request/iteration/owner scope.  Capacity reservations are
    separate from HBM/MTE service; MTE read and write directions remain
    distinct.  A reusable task is reused only under the same semantic key.
    """
    added = added if added is not None else set()
    if not specs:
        return []
    residency = ledger_for_engine(engine)
    if "prep" not in engine.resources:
        engine.resources["prep"] = ResourceSpec("prep", 1.0, "ms")
        engine.serviced["prep"] = 0.0
    arrivals = {
        getattr(request, "request", request).request_id:
        float(getattr(request, "request", request).arrival_ms)
        for request in requests
    }
    first_stage = stage_map.get(0, {})
    task_ids: list[str] = []
    for index, spec in enumerate(specs):
        semantic_key = (
            spec.cost_once_key
            or f"{spec.name}:{spec.request_id or ''}:{spec.iteration}:{spec.owner_stage}"
        )
        if spec.reusable:
            added_key = semantic_key
        else:
            added_key = f"{semantic_key}:i{spec.iteration}"
        if spec.reusable and added_key in added:
            existing = next(
                (
                    task_id for task_id, task in engine.tasks.items()
                    if dict(task.metadata).get("prepare_key") == added_key
                ),
                "",
            )
            task_ids.append(existing)
            continue
        if spec.request_id is not None:
            if spec.request_id not in arrivals:
                raise A3InputError(f"prepare_request_unknown:{spec.name}:{spec.request_id}")
            arrival = arrivals[spec.request_id]
        else:
            arrival = (
                float(spec.arrival_ms)
                if spec.arrival_ms > 0 else min(arrivals.values(), default=0.0)
            )
        if spec.known_at == "startup":
            release = 0.0
        elif spec.known_at == "arrival":
            release = max(arrival, float(spec.arrival_ms))
        elif spec.known_at == "previous_forward":
            release = max(float(now_ms), arrival, float(spec.arrival_ms))
        else:
            raise A3InputError(f"prepare_known_at_invalid:{spec.name}:{spec.known_at}")
        if spec.request_id is not None and release < arrival - 1e-12:
            raise A3InputError(f"prepare_before_request_arrival:{spec.name}")
        stage_index = int(spec.owner_stage)
        stage_names = stage_map.get(stage_index)
        if stage_names is None:
            stage_names = stage_map.get(0, {})
        hbm_name = stage_names.get("hbm")
        mte2_name = stage_names.get("mte2")
        mte3_name = stage_names.get("mte3")
        link_name = next(
            (value for value in stage_names.values() if value.startswith("link:")),
            None,
        )
        demands: dict[str, float] = {}
        reservations: dict[str, float] = {}
        if spec.duration_ms > 0:
            resource = spec.resource or "prep"
            if resource not in engine.resources:
                if resource not in {"prep", "cpu", "dma", "metadata"}:
                    raise A3InputError(f"prepare_resource_unknown:{spec.name}:{resource}")
                engine.resources[resource] = ResourceSpec(resource, 1.0, "ms")
                engine.serviced[resource] = 0.0
            demands[resource] = float(spec.duration_ms)
        hbm_service = float(spec.service_bytes or spec.bytes or 0.0) if (
            spec.capacity_bytes <= 0 or spec.service_bytes > 0
        ) else 0.0
        if hbm_service > 0:
            if hbm_name is None:
                raise A3InputError(f"prepare_hbm_resource_missing:{spec.name}")
            demands[hbm_name] = demands.get(hbm_name, 0.0) + hbm_service
        residency_entry = None
        if spec.capacity_bytes > 0:
            capacity_name = stage_names.get("hbm_capacity")
            if capacity_name is None:
                raise A3InputError(f"prepare_hbm_capacity_resource_missing:{spec.name}")
            capacity_resource = engine.resources.get(capacity_name)
            if capacity_resource is not None:
                residency.pool_capacity[capacity_name] = float(
                    capacity_resource.capacity
                )
            lifetime = str(spec.lifetime or "workspace").strip().lower()
            if lifetime in {"persistent", "model", "static"}:
                lifetime = "model_resident"
            if lifetime not in {
                "workspace", "model_resident", "request_resident",
                "request_prompt_iteration", "iteration",
            }:
                lifetime = "workspace"
            residency_entry = residency.admit(
                pool=capacity_name,
                capacity_bytes=int(spec.capacity_bytes),
                lifetime=lifetime,
                owner=f"prepare:{spec.name}:{spec.request_id or 'global'}",
                admitted_ms=float(release),
                request_id=spec.request_id,
                iteration=int(spec.iteration),
                accounted_elsewhere=bool(spec.accounted_elsewhere),
            )
            if residency_entry.status in {"overcommit", "missing_capacity"}:
                raise A3InputError(
                    f"prepare_capacity_{residency_entry.status}:"
                    f"{spec.name}:{capacity_name}"
                )
            if residency_entry.status == "admitted":
                reservations[capacity_name] = float(spec.capacity_bytes)
        if spec.link_bytes > 0:
            if link_name is None:
                raise A3InputError(f"prepare_link_resource_missing:{spec.name}")
            demands[link_name] = float(spec.link_bytes)
        read_bytes = float(spec.read_bytes or 0.0)
        write_bytes = float(spec.write_bytes or 0.0)
        if read_bytes <= 0 and spec.mte_bytes > 0:
            read_bytes = float(spec.mte_bytes)
        if read_bytes > 0:
            if mte2_name is None:
                raise A3InputError(f"prepare_mte2_resource_missing:{spec.name}")
            demands[mte2_name] = demands.get(mte2_name, 0.0) + read_bytes
        if write_bytes > 0:
            if mte3_name is None:
                raise A3InputError(f"prepare_mte3_resource_missing:{spec.name}")
            demands[mte3_name] = demands.get(mte3_name, 0.0) + write_bytes
        task_id = f"{prefix}:{spec.name}"
        suffix = index
        while task_id in engine.tasks:
            suffix += 1
            task_id = f"{prefix}:{spec.name}:{suffix}"
        engine.add_task(
            task_id, demands, release_ms=release, dependencies=tuple(external_deps),
            reservations=reservations or None,
            deadline_ms=spec.deadline_ms,
            owner=f"prepare:{spec.name}:{spec.request_id or 'global'}",
            metadata={
                "kind": spec.kind, "known_at": spec.known_at,
                "reusable": spec.reusable, "request_id": spec.request_id,
                "lifetime_ms": spec.lifetime_ms, "lifetime": spec.lifetime,
                "prepare_key": added_key, "iteration": spec.iteration,
                "owner_stage": stage_index, "owner_rank": spec.owner_rank,
                "consumer_op_ids": list(spec.consumer_op_ids),
                "first_consumer_op_ids": list(spec.first_consumer_op_ids),
                "replace_op_ids": list(spec.replace_op_ids),
                "capacity_bytes": spec.capacity_bytes,
                "read_bytes": read_bytes, "write_bytes": write_bytes,
                "service_bytes": hbm_service,
                "arrival_ms": spec.arrival_ms,
                "residency_entry_id": (
                    residency_entry.entry_id if residency_entry is not None else None
                ),
                "residency_status": (
                    residency_entry.status if residency_entry is not None else None
                ),
                "capacity_accounted_elsewhere": bool(
                    spec.accounted_elsewhere
                ),
            },
        )
        if residency_entry is not None:
            residency_entry.task_id = task_id
        added.add(added_key)
        task_ids.append(task_id)
    return task_ids


def release_prepare_residency(
    engine: ResourceEngine,
    *,
    request_id: str | None = None,
    iteration: int | None = None,
    now_ms: float | None = None,
) -> int:
    """Release request/iteration output residency at the actual lifetime end."""
    ledger = getattr(engine, "prepare_residency", None)
    if not isinstance(ledger, PrepareResidencyLedger):
        return 0
    timestamp = float(engine.now_ms if now_ms is None else now_ms)
    if request_id is None:
        return 0
    if iteration is not None:
        return ledger.release_iteration(request_id, int(iteration), timestamp)
    return ledger.release_request(request_id, timestamp)


def prepare_summary(
    result: Mapping[str, Any], token_end_ms: float,
    prep_task_ids: Sequence[str] = (),
    residency: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Report token-ready timestamp, drain tail and deadlines honestly."""
    slack = dict(result.get("deadline_slack_ms") or {})
    remaining = list(result.get("remaining_tasks") or [])
    prep_tasks = [task for task in prep_task_ids if task]
    return {
        "token_ready_ms": token_end_ms,
        "drain_makespan_ms": result.get("makespan_ms"),
        "background_remaining": remaining,
        "drain_tail_ms": max(0.0, float(result.get("makespan_ms") or token_end_ms) - token_end_ms)
        if not remaining else None,
        "deadline_slack_ms": slack,
        "deadline_misses": list(result.get("deadline_misses") or []),
        "prep_tasks": prep_tasks,
        "residency": dict(residency) if residency is not None else None,
        "note": "token-ready and engine-drain timestamps are separate; future work is not charged to a completed token",
    }


def prepare_residency_report(engine: ResourceEngine) -> dict[str, Any] | None:
    ledger = getattr(engine, "prepare_residency", None)
    return ledger.report() if isinstance(ledger, PrepareResidencyLedger) else None


__all__ = [
    "add_prepare_tasks", "prepare_summary", "release_prepare_residency",
    "prepare_residency_report",
]


def prepare_specs_from_candidates(
    candidates: Sequence[Any],
    *,
    host_rates: Mapping[str, float] | None = None,
    max_hbm_bytes: int | None = None,
) -> tuple[tuple[PrepareSpec, ...], tuple[dict[str, Any], ...]]:
    """Translate catalog ``PrepareCandidate`` records into executable specs.

    Candidate ownership, request/iteration scope, capacity, read/write
    direction and lifetime are preserved.  Capacity-only objects reserve HBM
    without inventing service traffic.  Missing host rates are explicit gaps;
    unsharded whole-model quantities are never divided or clamped here.
    """
    rates = dict(host_rates or {})
    specs: list[PrepareSpec] = []
    gaps: list[dict[str, Any]] = []
    for candidate in candidates:
        resources = dict(getattr(candidate, "resources", ()) or ())
        request_id = str(getattr(candidate, "request_id", "") or "") or None
        arrival_ms = float(getattr(candidate, "arrival_ms", 0.0) or 0.0)
        capacity_bytes = int(max(
            0.0,
            float(getattr(candidate, "capacity_bytes", 0.0) or 0.0),
            float(resources.get("hbm_reserved_bytes", 0.0) or 0.0),
        ))
        read_bytes = int(max(
            0.0,
            float(getattr(candidate, "read_bytes", 0.0) or 0.0),
            float(resources.get("mte2_bytes", 0.0) or 0.0),
        ))
        write_bytes = int(max(
            0.0,
            float(getattr(candidate, "write_bytes", 0.0) or 0.0),
            float(resources.get("mte3_bytes", 0.0) or 0.0),
        ))
        service_bytes = int(max(
            0.0,
            float(getattr(candidate, "service_bytes", 0.0) or 0.0),
            float(resources.get("hbm_bytes", 0.0) or 0.0),
        ))
        # Capacity is a residency budget, not transfer service.
        if max_hbm_bytes is not None and capacity_bytes > int(max_hbm_bytes):
            gaps.append({
                "candidate_id": getattr(candidate, "candidate_id", ""),
                "reason": "prepared_capacity_exceeds_declared_hbm_budget",
                "bytes": capacity_bytes, "budget_bytes": int(max_hbm_bytes),
                "owner_stage": int(getattr(candidate, "owner_stage", -1) or -1),
            })
            continue
        cpu_ops = float(resources.get("cpu_ops", 0.0) or 0.0)
        rate = rates.get("cpu_ops_per_s")
        duration_ms = 0.0
        if cpu_ops > 0:
            if not rate or rate <= 0:
                gaps.append({
                    "candidate_id": getattr(candidate, "candidate_id", ""),
                    "reason": "missing_host_service_rate:cpu_ops_per_s",
                    "operations": cpu_ops,
                    "required_work_not_zero": True,
                })
                continue
            duration_ms = cpu_ops / float(rate) * 1000.0
        known = str(getattr(candidate, "known_at", "runtime_static"))
        known_at = {
            "model_load": "startup", "runtime_static": "startup",
            "request_arrival": "arrival",
            "previous_forward": "previous_forward",
        }.get(known)
        if known_at is None:
            gaps.append({
                "candidate_id": getattr(candidate, "candidate_id", ""),
                "reason": f"prepare_known_at_unsupported:{known}",
            })
            continue
        selector = str(getattr(candidate, "consumer_selector", "background") or "")
        # Engram preparation is consumed by the actual Engram query before
        # sampling; it gates the forward consumer, not token visibility only.
        consumer = "forward" if selector or read_bytes or service_bytes else "background"
        kind = str(getattr(candidate, "kind", "prepare"))
        resource = "cpu" if cpu_ops > 0 else (
            "hbm" if service_bytes > 0 else "prep"
        )
        specs.append(PrepareSpec(
            name=str(getattr(candidate, "candidate_id", "prepare")),
            kind=kind, bytes=0,
            duration_ms=float(duration_ms), known_at=known_at,
            reusable=bool(getattr(candidate, "reusable", True)),
            request_id=request_id,
            deadline_ms=None, lifetime_ms=None,
            link_bytes=int(max(0.0, float(resources.get("link_bytes", 0.0) or 0.0))),
            mte_bytes=0,
            consumer=consumer,
            resource=resource,
            consumer_op_ids=tuple(
                str(v) for v in (getattr(candidate, "consumer_op_ids", ()) or ())
            ),
            first_consumer_op_ids=tuple(
                str(v) for v in (
                    getattr(candidate, "first_consumer_op_ids", ()) or ()
                )
            ),
            replace_op_ids=tuple(
                str(v) for v in (getattr(candidate, "replace_op_ids", ()) or ())
            ),
            owner_stage=int(getattr(candidate, "owner_stage", -1)),
            owner_rank=str(getattr(candidate, "owner_rank", "") or ""),
            iteration=int(getattr(candidate, "iteration", -1)),
            capacity_bytes=capacity_bytes,
            read_bytes=read_bytes,
            write_bytes=write_bytes,
            service_bytes=service_bytes,
            mte2_bytes=read_bytes,
            mte3_bytes=write_bytes,
            cost_once_key=str(getattr(candidate, "cost_once_key", "") or ""),
            lifetime=str(getattr(candidate, "lifetime", "") or ""),
            arrival_ms=arrival_ms,
        ))
    return tuple(specs), tuple(gaps)
