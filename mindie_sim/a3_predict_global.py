"""One event timeline for all predictive DP replicas.

The engine schedules disjoint local resources plus shared physical links and
stops at the earliest replica completion, so no global step barrier is
introduced.  Runner construction is imported lazily to avoid a cycle with the
framework request scheduler.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .a3_async import (
    add_prepare_tasks, prepare_residency_report, prepare_summary,
)
from .a3_resources import (
    ResourceEngine, ResourceSpec, add_costed_ops, build_resource_specs,
)
from .a3_types import CaseSpec
from .a3_workload import memory_plan


def predict_global_replicas(
    case: CaseSpec,
    states_by_replica: Sequence[Sequence[Any]],
    memory: MemoryPlan,
    replica_links: Sequence[Mapping[str, Any]],
    *,
    cost_backend: Any = None,
) -> list[dict[str, Any]]:
    """One event timeline for all DP replicas, including one shared link.

    Per-request ``media_encoded`` and explicit draft acceptance live in
    ``RequestState``; ``schedule_next``/``complete_step`` apply tower-once
    media state and accepted/rejected token/KV transitions.  This loop only
    arbitrates replica tasks on the shared physical links.
    """
    from .a3_framework import _ReplicaRunner, _backend_duration

    active = [index for index, states in enumerate(states_by_replica) if states]
    if not active:
        return []
    link_by = {item["replica"]: item for item in replica_links}
    runners: dict[int, _ReplicaRunner] = {}
    stage_maps: dict[int, dict[int, dict[str, str]]] = {}
    all_resources: dict[str, ResourceSpec] = {}
    if cost_backend is not None:
        all_resources["simulated_time"] = ResourceSpec("simulated_time", 1.0, "ms")
    for index in active:
        runner = _ReplicaRunner(
            case, states_by_replica[index], index, memory,
            cost_backend=cost_backend, replica_link=link_by[index],
        )
        runners[index] = runner
        link = link_by[index]
        resources, stage_map = build_resource_specs(
            case.machine, case.runtime.parallel.pp,
            compute_dtype=case.model.compute_dtype,
            link_capacity_scale=1.0, link=link.get("link"), ranks=link.get("ranks"),
        )
        renamed: dict[int, dict[str, str]] = {}
        for name, spec in resources.items():
            if name.startswith("link:"):
                all_resources.setdefault(name, spec)
                continue
            local = f"r{index}:{name}"
            all_resources[local] = ResourceSpec(
                local, spec.capacity, spec.unit, spec.scope, spec.policy,
            )
        for stage, names in stage_map.items():
            renamed[stage] = {
                key: (value if value.startswith("link:") else f"r{index}:{value}")
                for key, value in names.items()
            }
        stage_maps[index] = renamed
    engine = ResourceEngine(all_resources)
    prepare_ids: dict[int, list[str]] = {index: [] for index in active}
    prepared_names: dict[int, set[str]] = {index: set() for index in active}
    model_prepare_gaps: list[dict[str, Any]] = []
    model_prepare_specs: tuple[Any, ...] = ()
    if (case.runtime.model_prepare
            or (getattr(case.model, "quant", None) is not None
                and getattr(case.model, "layer_plans", ()))):
        from .a3_catalog_prepare import catalog_prepare_candidates
        from .a3_async import prepare_specs_from_candidates

        candidates = catalog_prepare_candidates(
            case.model, case.runtime, case.runtime.parallel,
            workload=case.workload,
        )
        model_prepare_specs, gap_rows = prepare_specs_from_candidates(
            candidates,
            host_rates=dict(case.runtime.prepare_host_rates),
            max_hbm_bytes=int(case.machine.dies[0].hbm_bytes),
        )
        model_prepare_gaps.extend(gap_rows)
    forward_prepare: dict[int, list[str]] = {index: [] for index in active}
    sample_prepare: dict[int, list[str]] = {index: [] for index in active}
    forward_prepare_specific: dict[
        int, dict[str, list[str]]
    ] = {index: {} for index in active}
    prepare_owner: dict[int, dict[str, str | None]] = {index: {} for index in active}
    prepare_replacements: dict[
        int, dict[str, list[tuple[str | None, str]]]
    ] = {index: {} for index in active}
    for index in active:
        runner = runners[index]
        owned = {state.request.request_id for state in runner.states}
        early = tuple(
            item for item in case.runtime.prepare
            if item.known_at != "previous_forward"
            and (item.request_id is None or item.request_id in owned)
        ) + tuple(
            spec for spec in model_prepare_specs
            if spec.known_at != "previous_forward"
            and (spec.request_id is None or spec.request_id in owned)
        )
        ids = add_prepare_tasks(
            engine, early, stage_maps[index], runner.states,
            now_ms=0.0, prefix=f"prep.r{index}",
            added=prepared_names[index],
        )
        prepare_ids[index].extend(ids)
        for task, spec in zip(ids, early):
            prepare_owner[index][task] = spec.request_id
            for op_id in spec.replace_op_ids:
                prepare_replacements[index].setdefault(op_id, []).append(
                    (spec.request_id, task)
                )
            if spec.consumer == "forward":
                consumers = tuple(dict.fromkeys(
                    (*spec.consumer_op_ids, *spec.first_consumer_op_ids)
                ))
                if consumers:
                    for op_id in consumers:
                        forward_prepare_specific[index].setdefault(
                            op_id, []
                        ).append(task)
                else:
                    forward_prepare[index].append(task)
            elif spec.consumer == "sample":
                sample_prepare[index].append(task)
    pending: dict[int, dict[str, Any]] = {}
    last_draft_next: dict[int, tuple[str, ...]] = {index: () for index in active}
    next_wait: float | None = None
    while True:
        for index in active:
            runner = runners[index]
            if index in pending or all(state.finished for state in runner.states):
                continue
            if runner.iteration >= runner._max_iterations:
                continue
            step = runner.schedule_next()
            if step is None:
                continue
            if "wait_until" in step:
                runner.now = float(step["wait_until"])
                next_wait = runner.now if next_wait is None else min(next_wait, runner.now)
                continue
            owned = {state.request.request_id for state in runner.states}
            late = tuple(
                item for item in case.runtime.prepare
                if item.known_at == "previous_forward"
                and (item.request_id is None or item.request_id in owned)
            ) + tuple(
                spec for spec in model_prepare_specs
                if spec.known_at == "previous_forward"
                and (spec.request_id is None or spec.request_id in owned)
            )
            if late and runner.iteration > 0:
                late_ids = add_prepare_tasks(
                    engine, late, stage_maps[index], runner.states,
                    now_ms=runner.now,
                    prefix=f"prep.r{index}.s{runner.iteration}",
                    added=prepared_names[index],
                )
                prepare_ids[index].extend(late_ids)
                for task, spec in zip(late_ids, late):
                    prepare_owner[index][task] = spec.request_id
                    for op_id in spec.replace_op_ids:
                        prepare_replacements[index].setdefault(op_id, []).append(
                            (spec.request_id, task)
                        )
                    if spec.consumer == "forward":
                        consumers = tuple(dict.fromkeys(
                            (*spec.consumer_op_ids, *spec.first_consumer_op_ids)
                        ))
                        if consumers:
                            for op_id in consumers:
                                forward_prepare_specific[index].setdefault(
                                    op_id, []
                                ).append(task)
                        else:
                            forward_prepare[index].append(task)
                    elif spec.consumer == "sample":
                        sample_prepare[index].append(task)
            prefix = step["prefix"]
            selected_ids = {
                state.request.request_id for state in step["selected"]
            }
            active_forward = [
                task for task in forward_prepare[index]
                if prepare_owner[index].get(task) is None
                or prepare_owner[index].get(task) in selected_ids
            ]
            active_sample = [
                task for task in sample_prepare[index]
                if prepare_owner[index].get(task) is None
                or prepare_owner[index].get(task) in selected_ids
            ]
            active_forward_specific: dict[str, tuple[str, ...]] = {}
            for op_id, entries in forward_prepare_specific[index].items():
                tasks = tuple(
                    task for task in entries
                    if prepare_owner[index].get(task) is None
                    or prepare_owner[index].get(task) in selected_ids
                )
                if tasks:
                    active_forward_specific[op_id] = tuple(dict.fromkeys(tasks))
            active_replacements: dict[str, tuple[str, ...]] = {}
            for op_id, entries in prepare_replacements[index].items():
                chosen = tuple(
                    task for owner, task in entries
                    if owner is None or owner in selected_ids
                )
                if chosen:
                    active_replacements[op_id] = tuple(dict.fromkeys(chosen))
            task_count_before = len(engine.tasks)
            if cost_backend is not None:
                duration = _backend_duration(cost_backend, step["work"], case)
                task_id = f"{prefix}:backend"
                engine.add_task(
                    task_id, {"simulated_time": duration},
                    dependencies=tuple(active_forward),
                    release_ms=runner.now,
                )
                targets = (task_id,)
                forward_gate = f"{prefix}:forward_gate"
                engine.add_task(forward_gate, {}, dependencies=targets)
                step["targets"] = targets
                step["forward_gate"] = forward_gate
            else:
                specific_deps = dict(active_forward_specific)
                if last_draft_next.get(index):
                    specific_deps["draft.prev_proposals_ready"] = (
                        last_draft_next[index]
                    )
                id_map = add_costed_ops(
                    engine, step["costed"], stage_maps[index], prefix=prefix,
                    start_ms=runner.now,
                    external_deps=tuple(active_forward),
                    replacement_tasks=active_replacements,
                    op_external_deps=specific_deps,
                    core_counts={
                        "aic": case.machine.dies[0].aic_count,
                        "aiv": case.machine.dies[0].aiv_count,
                    },
                )
                last_draft_next[index] = tuple(
                    id_map[item.op.op_id] for item in step["costed"]
                    if item.op.op_id.startswith("draft.next.")
                    and item.op.op_id in id_map
                )
                targets = tuple(id_map.values())
                forward_gate = f"{prefix}:forward_gate"
                engine.add_task(forward_gate, {}, dependencies=targets)
                step["targets"] = targets
                step["forward_gate"] = forward_gate
            sample_ids = tuple(active_sample)
            if sample_ids:
                token_gate = f"{prefix}:token_gate"
                engine.add_task(
                    token_gate, {},
                    dependencies=(step["forward_gate"], *sample_ids),
                )
            else:
                token_gate = step["forward_gate"]
            step["gate"] = token_gate
            step["events"] = []
            step["task_ids"] = set(list(engine.tasks)[task_count_before:]) | {step["gate"]}
            pending[index] = step
        if not pending:
            if next_wait is not None and any(
                not all(state.finished for state in runners[index].states)
                for index in active
            ):
                # No current work is admissible; advance to the next arrival.
                next_wait = None
                for index in active:
                    runner = runners[index]
                    if all(state.finished for state in runner.states):
                        continue
                    future = [
                        state.request.arrival_ms for state in runner.states
                        if not state.finished and not state.started
                        and state.request.arrival_ms > runner.now
                    ]
                    if future:
                        runner.now = min(future)
                continue
            break
        gates = tuple(step["gate"] for step in pending.values())
        start = min(runners[index].now for index in pending)
        result = engine.run(
            start_ms=start, until=gates, until_any=True, compact=True,
        )
        for step in pending.values():
            seen = {item["task_id"] for item in step["events"]}
            step["events"].extend(
                item for item in result.get("trace", ())
                if item["task_id"] in step["task_ids"]
                and item["task_id"] not in seen
            )
        for index, step in list(pending.items()):
            if step["gate"] not in result["completed_at_ms"]:
                continue
            end = float(result["completed_at_ms"][step["gate"]])
            step_tasks = set(step["targets"]) | {step["gate"]}
            starts = [item["start_ms"] for item in step["events"]]
            step["start_ms"] = min(starts, default=step["start_ms"])
            events = list(step["events"])
            runners[index].complete_step(
                step, end, events=events,
                background_remaining=result.get("remaining_tasks", ()),
                deadline_slack=result.get("deadline_slack_ms", {}),
            )
            del pending[index]
    drain = engine.run(start_ms=max(
        (runner.now for runner in runners.values()), default=0.0,
    ))
    token_end = max(
        (state.finish_ms or runner.now
         for runner in runners.values() for state in runner.states),
        default=0.0,
    )
    results = []
    for index in active:
        runner = runners[index]
        served = [state for state in runner.states if state.finished]
        unserved = [state for state in runner.states if not state.finished]
        results.append({
            "replica": index,
            "makespan_ms": max(
                (state.finish_ms or runner.now for state in runner.states), default=0.0,
            ),
            "iterations": runner.iteration,
            "served": len(served), "unserved": len(unserved),
            "requests": [state.to_dict() for state in runner.states],
            "traces": runner.traces,
            "prepare": prepare_summary(
                drain,
                max(
                    (state.finish_ms or runner.now for state in runner.states),
                    default=runner.now,
                ),
                prep_task_ids=tuple(prepare_ids[index]),
                residency=prepare_residency_report(engine),
            ),
            "unserved_reasons": {
                state.request.request_id: state.unserved_reason for state in unserved
            },
        })
    if model_prepare_gaps:
        for result in results:
            result["model_prepare_gaps"] = list(model_prepare_gaps)
    return results



__all__ = ["predict_global_replicas"]
