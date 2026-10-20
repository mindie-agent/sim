"""Normalized CV/tiled-pipeline algorithm fixtures.

These are normalized arithmetic fixtures, not A3 measurements.  They execute
the same ResourceEngine used by the production paths and are kept out of the
should-cost core so the latter stays focused on model semantics.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .a3_resources import ResourceEngine, ResourceSpec
from .a3_types import A3InputError


def schedule_shared_cv(
    *,
    c_memory: float = 6.0,
    c_compute: float = 8.0,
    v_memory: float = 6.0,
    v_compute: float = 2.0,
    v_start_ms: float = 0.0,
    serialize_v_after_c: bool = False,
) -> dict[str, Any]:
    """Execute the normalized C-read/C-cube + V-read/V-vector DAG."""
    resources = {
        "memory": ResourceSpec("memory", 1.0),
        "cube": ResourceSpec("cube", 1.0),
        "vector": ResourceSpec("vector", 1.0),
    }
    engine = ResourceEngine(resources)
    engine.add_task("c_read", {"memory": c_memory})
    engine.add_task("c_compute", {"cube": c_compute}, dependencies=("c_read",))
    v_deps = ("c_compute",) if serialize_v_after_c else ()
    engine.add_task(
        "v_read", {"memory": v_memory}, dependencies=v_deps,
        release_ms=v_start_ms,
    )
    engine.add_task("v_compute", {"vector": v_compute}, dependencies=("v_read",))
    result = engine.run()
    lower_bound = max(c_memory + v_memory, c_compute, v_compute)
    return {
        "status": "modeled",
        "makespan_ms": result["makespan_ms"],
        "resource_lower_bound_ms": lower_bound,
        "trace": result["trace"],
        "completed_at_ms": result["completed_at_ms"],
        "scenario": {
            "c_memory": c_memory, "c_compute": c_compute,
            "v_memory": v_memory, "v_compute": v_compute,
            "v_start_ms": v_start_ms, "serialize_v_after_c": serialize_v_after_c,
        },
    }


def normalized_shared_cv_search(
    plans: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Search actual engine schedules for the normalized shared-memory DAG."""
    candidates = list(plans or (
        {"name": "simultaneous_fair", "v_start_ms": 0.0},
        {"name": "serial_v_after_c", "serialize_v_after_c": True},
        {"name": "v_delayed_6ms", "v_start_ms": 6.0},
    ))
    evaluated = []
    for plan in candidates:
        result = schedule_shared_cv(
            v_start_ms=float(plan.get("v_start_ms", 0.0)),
            serialize_v_after_c=bool(plan.get("serialize_v_after_c", False)),
        )
        evaluated.append({
            "name": str(plan.get("name", "plan")),
            "target_ms": result["makespan_ms"],
            "resource_lower_bound_ms": result["resource_lower_bound_ms"],
            "trace": result["trace"],
            "completed_at_ms": result["completed_at_ms"],
            "scenario": result["scenario"],
        })
    lower = evaluated[0]["resource_lower_bound_ms"] if evaluated else None
    feasible = [item for item in evaluated if lower is not None and item["target_ms"] >= lower - 1e-9]
    best = min(feasible, key=lambda item: item["target_ms"]) if feasible else None
    return {
        "status": "modeled" if best else "infeasible",
        "normalized_fixture": True,
        "not_a3_measurement": True,
        "resource_lower_bound_ms": lower,
        "resource_lower_bound_is_feasible_target": False,
        "plans": evaluated,
        "feasible_plans": len(feasible),
        "selected": best,
        "selected_target_ms": best["target_ms"] if best else None,
        "witness": best["trace"] if best else None,
        "note": "Schedules are executed by ResourceEngine; no production answer constants.",
    }


def chunked_pipeline_makespan(
    stage_work_ms: Sequence[float], chunks: int, *, per_chunk_overhead_ms: float = 0.2,
) -> dict[str, Any]:
    """Execute FIFO stage/chunk tasks with per-stage per-chunk overhead."""
    if not stage_work_ms or chunks < 1:
        raise A3InputError("chunked_pipeline_input_invalid")
    names = ("producer", "transfer", "consumer")
    resources = {
        name: ResourceSpec(name, 1.0, policy="fifo")
        for name in names
    }
    engine = ResourceEngine(resources)
    work = [float(value) for value in stage_work_ms]
    stage_tasks: list[list[str]] = [[] for _ in work]
    for chunk in range(chunks):
        dependency: str | None = None
        for stage, duration in enumerate(work):
            task_id = f"{names[stage]}_{chunk}"
            engine.add_task(
                task_id,
                {names[stage]: duration / chunks + float(per_chunk_overhead_ms)},
                dependencies=(dependency,) if dependency else (),
                priority=chunk,
            )
            stage_tasks[stage].append(task_id)
            dependency = task_id
    result = engine.run()
    return {
        "chunks": chunks,
        "per_chunk_overhead_ms": float(per_chunk_overhead_ms),
        "stage_chunk_ms": [value / chunks + float(per_chunk_overhead_ms) for value in work],
        "makespan_ms": result["makespan_ms"],
        "timeline": result["trace"],
    }


def normalized_chunked_pipeline_search(
    stage_work_ms: Sequence[float] = (8.0, 6.0, 4.0),
    chunks: Sequence[int] = (1, 2, 4, 8, 16),
) -> dict[str, Any]:
    plans = [
        {"chunks": n, **chunked_pipeline_makespan(stage_work_ms, n)}
        for n in chunks
    ]
    best = min(plans, key=lambda item: item["makespan_ms"])
    return {
        "status": "modeled",
        "normalized_fixture": True,
        "not_a3_measurement": True,
        "plans": [
            {"chunks": plan["chunks"], "makespan_ms": plan["makespan_ms"]}
            for plan in plans
        ],
        "selected": {"chunks": best["chunks"], "makespan_ms": best["makespan_ms"]},
        "witness": best["timeline"],
        "note": "Finite chunking has per-chunk overhead; the optimum is finite.",
    }



__all__ = [
    "schedule_shared_cv", "normalized_shared_cv_search",
    "chunked_pipeline_makespan", "normalized_chunked_pipeline_search",
]
