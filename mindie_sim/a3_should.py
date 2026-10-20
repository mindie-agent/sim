"""Normative should-cost: semantic lower bound L and feasible target U.

[Role]
  Answer "what should this model/shape cost on this hardware?" independently of
  the current framework's scheduler, copies, padding and waits.  Derive the
  required compute/memory/communication work from model semantics and request
  token counts, compute an optimistic resource lower bound ``L``, then
  enumerate legal placement/tiling/fusion/recipe plans and build a witness
  timeline for the best feasible target ``U``.

[Invariants]
  * This path is not ``predictive * efficiency``.  It never reads a calibrated
    predictive cost; ``L`` uses raw peaks and unique semantic work.
  * ``L`` is a lower bound for the declared semantic workload, not a runnable
    schedule and not a global lower bound over unknown implementations.
  * ``U`` is the best-known target in the *enumerated plan family*, with a
    legal witness timeline.  It is not called a global lower bound.
  * U includes legal tile padding, launch and dependent communication.  The
    result must satisfy ``L <= U``; a violation is reported, never hidden.
  * Framework copies/padding/waits are not inherited by L.  Graph padding is
    only a legal plan choice for U.
  * Changing the current predictive TP/EP/DP config does not change the
    normative plan family; the family is derived from machine/model/workload
    and optional explicit should-cost constraints.

[Timeouts & Locks]
  Bounded enumeration (<=1024 plans) and pure arithmetic/event simulation.
  No I/O, locks, sleeps, provider import or device access.

[Side-Effect Policy]
  No filesystem/network/global mutation.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from .a3_cost import CalibrationStore, PredictiveCostModel
from .a3_async import (
    add_prepare_tasks, prepare_residency_report, prepare_summary,
)
from .a3_fusion import apply_fusion
from .a3_resources import (
    ResourceEngine, ResourceSpec, add_costed_ops, build_resource_specs,
    fusion_pool_capacities, plan_replica_links,
)
from .a3_fixtures import (
    chunked_pipeline_makespan, normalized_chunked_pipeline_search,
    normalized_shared_cv_search, schedule_shared_cv,
)
from .a3_request import RequestState, runtime_with_draft_next_counts
from .a3_types import (
    A3InputError, CaseSpec, MachineSpec, ModelSpec, ParallelSpec, RuntimeSpec,
    SloSpec, WorkloadSpec, align_up, dtype_bytes,
)
from .a3_workload import IterationPlan, RequestView, derive_iteration_work, memory_plan

from .a3_semantics import SemanticWork, hardware_lower_bound, semantic_work
from .a3_plans import PlanOption, ShouldResult, _candidate_plans




def _plan_runtime(
    case: CaseSpec, plan: PlanOption, *, constraints: Mapping[str, Any] | None = None,
) -> RuntimeSpec:
    raw = dict(constraints or {})
    parallel = ParallelSpec(dp=plan.dp, tp=plan.tp, ep=plan.ep, pp=plan.pp)
    max_len = max(request.prompt_tokens + request.output_tokens for request in case.workload.requests)
    max_prompt = max(request.prompt_tokens for request in case.workload.requests)
    bucket = align_up(max(1, len(case.workload.requests)), 64)
    return RuntimeSpec(
        max_num_batched_tokens=max(1, max_prompt),
        max_num_seqs=max(1, len(case.workload.requests)),
        max_model_len=max_len,
        enable_chunked_prefill=True,
        kv_block_size=int(raw.get("kv_block_size", 128)),
        gpu_memory_utilization=float(raw.get("gpu_memory_utilization", 0.9)),
        parallel=parallel,
        graph_buckets=(bucket,) if plan.graph_padding else (),
        graph_mode="force" if plan.graph_padding else "eager",
        fusion=plan.fusion,
        prepare=case.runtime.prepare,
        model_prepare=case.runtime.model_prepare,
        prepare_host_rates=case.runtime.prepare_host_rates,
        metadata=case.runtime.metadata,
        moe_recipe=plan.moe_recipe if case.model.ffn == "moe" else "dense",
    )


def _assign_requests(case: CaseSpec, dp: int) -> list[list[Any]]:
    replicas: list[list[Any]] = [[] for _ in range(max(1, dp))]
    for index, request in enumerate(case.workload.requests):
        replicas[index % max(1, dp)].append(request)
    return replicas



def _evaluate_normative_plan(
    case: CaseSpec, plan: PlanOption, constraints: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_constraints = dict(constraints or {})
    placement_constraint = raw_constraints.get("placement")
    if isinstance(placement_constraint, Mapping):
        core_base_raw = placement_constraint.get("core_base")
    else:
        core_base_raw = raw_constraints.get("core_base")
    core_base = None if core_base_raw is None else int(core_base_raw)
    try:
        plan.parallel.validate(case.model)  # type: ignore[attr-defined]
    except AttributeError:
        pass
    parallel = ParallelSpec(dp=plan.dp, tp=plan.tp, ep=plan.ep, pp=plan.pp)
    try:
        parallel.validate(case.model)
    except A3InputError as exc:
        return {"status": "infeasible", "reason": exc.reason, "plan": plan.to_dict()}
    if parallel.world_size > len(case.machine.dies):
        return {"status": "infeasible", "reason": "world_size_exceeds_devices", "plan": plan.to_dict()}
    runtime = _plan_runtime(case, plan, constraints=constraints)
    memory = memory_plan(case.model, runtime, case.machine)
    if memory.status != "fits":
        return {"status": "infeasible", "reason": "memory_infeasible", "plan": plan.to_dict(),
                "memory": memory.to_dict()}
    replicas = _assign_requests(case, plan.dp)
    active_replicas = [index for index, group in enumerate(replicas) if group]
    replica_links = plan_replica_links(
        case.machine, parallel, active_replicas=active_replicas,
    )
    link_by_replica = {item["replica"]: item for item in replica_links}
    # Required live KV is a peak-resident per-request block count, not the sum
    # of token lifetimes.
    for requests in replicas:
        if not requests:
            continue
        required_tokens = sum(request.prompt_tokens + request.output_tokens for request in requests)
        required_blocks = sum(
            math.ceil((request.prompt_tokens + request.output_tokens) / max(1, runtime.kv_block_size))
            for request in requests
        )
        if required_tokens > memory.kv_capacity_tokens or required_blocks > memory.kv_blocks:
            return {
                "status": "infeasible", "reason": "replica_peak_kv_infeasible",
                "plan": plan.to_dict(),
                "required_tokens": required_tokens,
                "required_blocks": required_blocks,
                "kv_blocks": memory.kv_blocks,
                "kv_capacity_tokens": memory.kv_capacity_tokens,
                "live_kv_tokens_peak": required_tokens,
                "memory": memory.to_dict(),
                "offload_modeled": False, "recompute_modeled": False,
            }

    # One global resource engine: local replica resources stay disjoint while
    # the physical link is shared by every replica.  Equal-share is applied by
    # the live active set, so an idle/sequential replica does not reserve
    # bandwidth that nobody else is using.
    all_resources: dict[str, ResourceSpec] = {}
    replica_stage_maps: dict[int, dict[int, dict[str, str]]] = {}
    link_names: set[str] = set()
    for replica_index in active_replicas:
        replica_link = link_by_replica[replica_index]
        resources, stage_map = build_resource_specs(
            case.machine, parallel.pp,
            compute_dtype=case.model.compute_dtype,
            link_capacity_scale=1.0,
            link=replica_link.get("link"),
            ranks=replica_link.get("ranks"),
        )
        renamed: dict[int, dict[str, str]] = {}
        for name, spec in resources.items():
            if name.startswith("link:"):
                all_resources.setdefault(name, spec)
                link_names.add(name)
                continue
            local_name = f"r{replica_index}:{name}"
            all_resources[local_name] = ResourceSpec(
                local_name, spec.capacity, spec.unit, spec.scope, spec.policy,
            )
        for stage, names in stage_map.items():
            renamed[stage] = {
                key: (value if value.startswith("link:") else f"r{replica_index}:{value}")
                for key, value in names.items()
            }
        replica_stage_maps[replica_index] = renamed
    engine = ResourceEngine(all_resources)

    cost_model = PredictiveCostModel(
        case.machine, calibration=CalibrationStore(),
        compute_dtype=case.model.compute_dtype,
        weight_dtype=case.model.weight_dtype,
        activation_dtype=case.model.activation_dtype,
        quantization=case.model.quantization,
        # Normative should is independent of measured residuals and operator
        # curve observations; only the predictive product may consume them.
    )
    prepare_ids: dict[int, list[str]] = {}
    forward_prepare: dict[int, list[str]] = {}
    sample_prepare: dict[int, list[str]] = {}
    prepared_names: dict[int, set[str]] = {index: set() for index in active_replicas}
    model_prepare_specs: tuple[Any, ...] = ()
    model_prepare_gaps: list[dict[str, Any]] = []
    if (runtime.model_prepare
            or (getattr(case.model, "quant", None) is not None
                and getattr(case.model, "layer_plans", ()))):
        from .a3_async import prepare_specs_from_candidates
        from .a3_catalog_prepare import catalog_prepare_candidates

        candidates = catalog_prepare_candidates(
            case.model, runtime, parallel, workload=case.workload,
        )
        model_prepare_specs, gap_rows = prepare_specs_from_candidates(
            candidates,
            host_rates=dict(runtime.prepare_host_rates),
            max_hbm_bytes=int(case.machine.dies[0].hbm_bytes),
        )
        model_prepare_gaps.extend(gap_rows)
    prepare_owner: dict[int, dict[str, str | None]] = {
        index: {} for index in active_replicas
    }
    prepare_replacements: dict[
        int, dict[str, list[tuple[str | None, str]]]
    ] = {index: {} for index in active_replicas}
    for replica_index in active_replicas:
        forward_prepare[replica_index] = []
        sample_prepare[replica_index] = []
        owned = {request.request_id for request in replicas[replica_index]}
        early_prepare = tuple(
            item for item in runtime.prepare
            if item.known_at != "previous_forward"
            and (item.request_id is None or item.request_id in owned)
        ) + tuple(
            spec for spec in model_prepare_specs
            if spec.known_at != "previous_forward"
        )
        early_ids = add_prepare_tasks(
            engine, early_prepare, replica_stage_maps[replica_index],
            replicas[replica_index], now_ms=0.0,
            prefix=f"prep.r{replica_index}",
            added=prepared_names[replica_index],
        )
        prepare_ids[replica_index] = early_ids
        for task, spec in zip(early_ids, early_prepare):
            prepare_owner[replica_index][task] = spec.request_id
            for op_id in spec.replace_op_ids:
                prepare_replacements[replica_index].setdefault(
                    op_id, []
                ).append((spec.request_id, task))
            if spec.consumer == "forward":
                forward_prepare[replica_index].append(task)
            elif spec.consumer == "sample":
                sample_prepare[replica_index].append(task)
    replica_steps: dict[int, list[dict[str, Any]]] = {}
    replica_task_ids: dict[int, list[list[str]]] = {}
    replica_gates: dict[int, list[str]] = {}
    for replica_index in active_replicas:
        requests = replicas[replica_index]
        steps: list[dict[str, Any]] = []
        generated = {request.request_id: 0 for request in requests}
        draft_states = {}
        for request in requests:
            state = RequestState(request)
            state.configure_draft(runtime, case.model)
            draft_states[request.request_id] = state
        prefill_views = []
        for request in requests:
            remaining = request.prompt_tokens - request.prefix_tokens
            if remaining > 0:
                prefill_views.append(RequestView(
                    request.request_id, remaining, request.prefix_tokens, True,
                    "prefill", request.prefix_tokens, request.priority,
                    semantic_phase="prefill", execution_mode="prefill",
                    media_encoded=bool(request.media_encoded),
                    draft_verify=False,
                ))
        step_index = 0
        if prefill_views:
            steps.append({"views": tuple(prefill_views), "step": step_index, "phase": "prefill"})
            step_index += 1
            for request in requests:
                if request.prompt_tokens - request.prefix_tokens > 0:
                    generated[request.request_id] = 1
        while any(generated[request.request_id] < request.output_tokens for request in requests):
            views = []
            advances: dict[str, int] = {}
            draft_rows: list[dict[str, Any]] = []
            for request in requests:
                if generated[request.request_id] >= request.output_tokens:
                    continue
                # Before the first decode, all prompt tokens are already in
                # cache; after that the just-generated token is also in KV.
                g = generated[request.request_id]
                context = request.prompt_tokens + (g if g == 0 else g - 1)
                state = draft_states[request.request_id]
                remaining = request.output_tokens - g
                verify = bool(
                    state.draft_mode and remaining > 1
                    and state.draft_proposal_tokens > 0
                )
                proposal = state.draft_proposal_tokens if verify else 0
                accepted = state.peek_draft_accept() if verify else 0
                advance = min(remaining, accepted + 1) if verify else 1
                advances[request.request_id] = advance
                if verify:
                    state.draft_accept_cursor += 1
                    state.draft_proposed_total += proposal
                    state.draft_accepted_total += min(accepted, remaining - 1)
                    state.draft_rollback_tokens += max(0, proposal - accepted)
                    if accepted == 0:
                        state.draft_fallback_tokens += 1
                    draft_rows.append({
                        "request_id": request.request_id,
                        "proposal_tokens": proposal,
                        "accepted_tokens": accepted,
                        "advanced_tokens": advance,
                        "rollback_tokens": max(0, proposal - accepted),
                    })
                views.append(RequestView(
                    request.request_id, 1, context, True, "decode",
                    request.prefix_tokens, request.priority,
                    semantic_phase="decode", execution_mode="decode",
                    media_encoded=bool(
                        request.media_encoded or request.media_tokens > 0
                    ),
                    draft_verify=verify,
                    draft_proposal_tokens=proposal,
                    draft_accepted_tokens=accepted,
                ))
            if not views:
                break
            steps.append({
                "views": tuple(views), "step": step_index, "phase": "decode",
                "draft_steps": draft_rows,
            })
            step_index += 1
            for view in views:
                generated[view.request_id] += advances[view.request_id]
        replica_steps[replica_index] = steps

    # Add every step's costed DAG to the one global engine.  A zero-demand gate
    # serializes a replica's steps; sibling replicas are independent and share
    # only physical link resources.
    for replica_index in active_replicas:
        previous_tasks: list[str] = []
        replica_task_ids[replica_index] = []
        replica_gates[replica_index] = []
        first_arrival = min(
            request.arrival_ms for request in replicas[replica_index]
        )
        arrival_gate = f"norm.r{replica_index}.arrival_gate"
        engine.add_task(
            arrival_gate, {}, release_ms=float(first_arrival),
            owner="request_arrival_gate",
        )
        for step_index, step in enumerate(replica_steps[replica_index]):
            prefix = f"norm.r{replica_index}.s{step_index}"
            external: tuple[str, ...] = (
                (arrival_gate,) if step_index == 0 else ()
            )
            if previous_tasks:
                gate = f"{prefix}.gate"
                engine.add_task(gate, {}, dependencies=tuple(previous_tasks), owner="step_gate")
                replica_gates[replica_index].append(gate)
                external = (gate,)
            late_prepare = tuple(
                item for item in runtime.prepare if item.known_at == "previous_forward"
                and (item.request_id is None or item.request_id in owned)
            ) + tuple(
                spec for spec in model_prepare_specs
                if spec.known_at == "previous_forward"
                and (spec.request_id is None or spec.request_id in owned)
            )
            if late_prepare and step_index > 0:
                late_ids = add_prepare_tasks(
                    engine, late_prepare, replica_stage_maps[replica_index],
                    replicas[replica_index], now_ms=0.0,
                    prefix=f"prep.r{replica_index}.s{step_index}",
                    external_deps=external,
                    added=prepared_names[replica_index],
                )
                prepare_ids[replica_index].extend(late_ids)
                for task, spec in zip(late_ids, late_prepare):
                    prepare_owner[replica_index][task] = spec.request_id
                    for op_id in spec.replace_op_ids:
                        prepare_replacements[replica_index].setdefault(
                            op_id, []
                        ).append((spec.request_id, task))
                    if spec.consumer == "forward":
                        forward_prepare[replica_index].append(task)
                    elif spec.consumer == "sample":
                        sample_prepare[replica_index].append(task)
            selected_ids = {view.request_id for view in step["views"]}
            active_forward = [
                task for task in forward_prepare[replica_index]
                if prepare_owner[replica_index].get(task) is None
                or prepare_owner[replica_index].get(task) in selected_ids
            ]
            active_sample = [
                task for task in sample_prepare[replica_index]
                if prepare_owner[replica_index].get(task) is None
                or prepare_owner[replica_index].get(task) in selected_ids
            ]
            if active_forward:
                external = external + tuple(active_forward)
            active_replacements: dict[str, tuple[str, ...]] = {}
            for op_id, entries in prepare_replacements[replica_index].items():
                chosen = tuple(
                    task for owner, task in entries
                    if owner is None or owner in selected_ids
                )
                if chosen:
                    active_replacements[op_id] = tuple(dict.fromkeys(chosen))
            task_count_before = len(engine.tasks)
            step_runtime = runtime_with_draft_next_counts(
                runtime, tuple(step["views"]),
                tuple(draft_states.values()),
            )
            work = derive_iteration_work(
                IterationPlan(step["views"], iteration=step_index),
                case.model, step_runtime, case.machine, runtime.parallel,
            )
            fusion = apply_fusion(
                work, enabled=plan.fusion,
                scratch_capacity_bytes=(
                    None if fusion_pool_capacities(case.machine.dies[0])
                    else (
                        case.machine.dies[0].l2_bytes
                        if case.machine.dies[0].l2_bytes > 0 else 0
                    )
                ),
                pool_capacities=fusion_pool_capacities(case.machine.dies[0]),
                **({
                    "core_count": int(case.machine.dies[0].aic_count),
                    "per_core_rates": {
                        "cube": case.machine.peak_flops_for(
                            case.model.compute_dtype
                        ) / max(1, int(case.machine.dies[0].aic_count)),
                        "vector": case.machine.dies[0].vector_ops_per_s
                        / max(1, int(case.machine.dies[0].aiv_count)),
                        "mte2": case.machine.dies[0].mte_bw_bytes_per_s / 1000.0,
                        "mte3": case.machine.dies[0].mte_bw_bytes_per_s / 1000.0,
                        "hbm": case.machine.dies[0].hbm_bw_bytes_per_s / 1000.0,
                    },
                    "core_base": core_base,
                    "worker_count": max(1, int(plan.worker_count)),
                } if case.machine.dies[0].aic_count > 0
                  and case.machine.dies[0].aiv_count > 0 else {}),
                cube_flops_per_s=case.machine.peak_flops_for(case.model.compute_dtype),
                vector_ops_per_s=case.machine.dies[0].vector_ops_per_s,
                model_features={
                    "weight_dtype": case.model.weight_dtype,
                    "quantization": case.model.quantization,
                },
            )
            fusion_payload = fusion.to_dict()
            placement = (
                (fusion_payload.get("witness") or {}).get("full_work_placement")
            )
            if plan.fusion and plan.worker_count > 1:
                witness = fusion_payload.get("witness", {})
                feasible_by_task = dict(
                    witness.get("physical_feasible_by_task", {}) or {}
                )
                if (
                    not witness.get("physical_mode")
                    or not all(bool(value) for value in feasible_by_task.values())
                ):
                    return {
                        "status": "infeasible",
                        "reason": "fusion_worker_partition_not_physical",
                        "plan": plan.to_dict(),
                    }
            work = fusion.work
            costed = cost_model.cost_iteration(work)
            id_map = add_costed_ops(
                engine, costed, replica_stage_maps[replica_index],
                prefix=prefix, start_ms=0.0, external_deps=external,
                replacement_tasks=active_replacements,
                core_counts={
                    "aic": case.machine.dies[0].aic_count,
                    "aiv": case.machine.dies[0].aiv_count,
                },
            )
            step_tasks = list(engine.tasks)[task_count_before:]
            forward_gate = f"{prefix}.forward_gate"
            engine.add_task(
                forward_gate, {}, dependencies=tuple(step_tasks),
                owner="forward_completion_gate",
            )
            sample_ids = tuple(active_sample)
            if sample_ids:
                token_gate = f"{prefix}.token_gate"
                engine.add_task(
                    token_gate, {},
                    dependencies=(forward_gate, *sample_ids),
                    owner="sample_completion_gate",
                )
            else:
                token_gate = forward_gate
            task_ids = list(dict.fromkeys((*step_tasks, forward_gate, token_gate)))
            replica_task_ids[replica_index].append(task_ids)
            step["work"] = work
            step["fusion"] = fusion
            step["prefix"] = prefix
            previous_tasks = [forward_gate]
    engine_result = engine.run(start_ms=0.0, compact=True)
    completed_at = engine_result["completed_at_ms"]
    trace_by_task = {item["task_id"]: item for item in engine_result["trace"]}

    replica_results: list[dict[str, Any]] = []
    for replica_index in active_replicas:
        requests = replicas[replica_index]
        steps_out = []
        generated = {request.request_id: 0 for request in requests}
        draft_states = {}
        for request in requests:
            state = RequestState(request)
            state.configure_draft(runtime, case.model)
            draft_states[request.request_id] = state
        first_token: dict[str, float] = {}
        token_times: dict[str, list[float]] = {request.request_id: [] for request in requests}
        finish: dict[str, float] = {}
        for step_record, task_ids in zip(
            replica_steps[replica_index], replica_task_ids[replica_index]
        ):
            times = [completed_at[task_id] for task_id in task_ids if task_id in completed_at]
            step_end = max(times, default=0.0)
            step_start = min(
                (trace_by_task[task_id]["start_ms"] for task_id in task_ids
                 if task_id in trace_by_task),
                default=step_end,
            )
            work = step_record["work"]
            fusion = step_record["fusion"]
            step_out = {
                "step": step_record["step"],
                "phase": work.mode,
                "recipe": work.recipe,
                "valid_tokens": work.valid_tokens,
                "padded_tokens": work.padded_tokens,
                "start_ms": step_start,
                "end_ms": step_end,
                "op_count": len(work.ops),
                "fusion": fusion.to_dict() if plan.fusion else None,
                "events": [
                    trace_by_task[task_id] for task_id in task_ids
                    if task_id in trace_by_task
                ],
                "graph_bucket": work.graph_bucket,
                "routing": work.routing.to_dict() if work.routing is not None else None,
                "draft_steps": list(step_record.get("draft_steps", ())),
            }
            steps_out.append(step_out)
            for view in step_record["views"]:
                request_id = view.request_id
                if step_record["phase"] == "prefill":
                    generated[request_id] = 1
                else:
                    request = next(
                        item for item in requests if item.request_id == request_id
                    )
                    remaining = request.output_tokens - generated[request_id]
                    advance = (
                        min(remaining, int(view.draft_accepted_tokens) + 1)
                        if bool(view.draft_verify) else 1
                    )
                    generated[request_id] += max(1, advance)
                if request_id not in first_token:
                    first_token[request_id] = step_end
                token_times[request_id].append(step_end)
                request = next(item for item in requests if item.request_id == request_id)
                if generated[request_id] >= request.output_tokens:
                    finish[request_id] = step_end
        timeline_origin = min(request.arrival_ms for request in requests)
        absolute_finish = max(finish.values(), default=timeline_origin)
        service_finish = {
            request.request_id: max(0.0, finish[request.request_id] - request.arrival_ms)
            for request in requests if request.request_id in finish
        }
        service_first = {
            request.request_id: max(0.0, first_token[request.request_id] - request.arrival_ms)
            for request in requests if request.request_id in first_token
        }
        service_makespan = max(service_finish.values(), default=0.0)
        wall_first = {
            request.request_id: first_token[request.request_id]
            for request in requests if request.request_id in first_token
        }
        wall_finish = {
            request.request_id: finish[request.request_id]
            for request in requests if request.request_id in finish
        }
        draft_summary: dict[str, dict[str, Any]] = {
            request.request_id: {
                "proposal_tokens": 0, "accepted_tokens": 0,
                "rollback_tokens": 0, "steps": 0,
            }
            for request in requests
        }
        for step in steps_out:
            for row in step.get("draft_steps", ()):
                summary = draft_summary[row["request_id"]]
                summary["proposal_tokens"] += int(row["proposal_tokens"])
                summary["accepted_tokens"] += int(row["accepted_tokens"])
                summary["rollback_tokens"] += int(row["rollback_tokens"])
                summary["steps"] += 1
        replica_results.append({
            "replica": replica_index,
            "requests": [request.request_id for request in requests],
            "draft_requests": draft_summary,
            "makespan_ms": service_makespan,
            "wall_makespan_ms": max(wall_finish.values(), default=service_makespan),
            "first_token_ms": wall_first,
            "service_first_token_ms": service_first,
            "token_times_ms": token_times,
            "finish_ms": wall_finish,
            "service_finish_ms": finish,
            "steps": steps_out,
            "arrival_ms": {request.request_id: request.arrival_ms for request in requests},
        })
    if not replica_results:
        return {"status": "infeasible", "reason": "no_requests", "plan": plan.to_dict()}
    service_makespan = max(
        (result["makespan_ms"] for result in replica_results), default=0.0,
    )
    wall_makespan = max(
        (result["wall_makespan_ms"] for result in replica_results), default=0.0,
    )
    makespan = wall_makespan
    ttft = [
        value for result in replica_results for value in result["service_first_token_ms"].values()
    ]
    wall_ttft = [
        value for result in replica_results for value in result["first_token_ms"].values()
    ]
    itl = []
    for result in replica_results:
        for times in result["token_times_ms"].values():
            itl.extend(times[index] - times[index - 1] for index in range(1, len(times)))
    applied_rules = sorted({
        rule for result in replica_results for step in result["steps"]
        for rule in ((step.get("fusion") or {}).get("applied_rules") or [])
    })
    removed = sorted({
        item for result in replica_results for step in result["steps"]
        for item in ((step.get("fusion") or {}).get("removed_intermediates") or [])
    })
    fusion_changes = {
        key: sum(
            float(((step.get("fusion") or {}).get("changes") or {}).get(key, 0.0))
            for result in replica_results for step in result["steps"]
        )
        for key in ("flops", "hbm_bytes", "mte2_bytes", "mte3_bytes",
                    "vector_ops", "launch", "peak_scratch_bytes", "serial_ms")
    }
    from .a3_async import release_prepare_residency
    for request in case.workload.requests:
        release_prepare_residency(
            engine, request_id=request.request_id,
            now_ms=float(engine_result.get("makespan_ms") or 0.0),
        )
    link_events = [item for item in engine_result["trace"] if any(
        name in item["demands"] for name in link_names
    )]
    return {
        "status": "modeled",
        "plan": {**plan.to_dict(), "placement": placement},
        "target_ms": makespan,
        "intrinsic_target_ms": service_makespan,
        "wall_makespan_ms": wall_makespan,
        "ttft_ms": (sum(wall_ttft) / len(wall_ttft)) if wall_ttft else None,
        "intrinsic_ttft_ms": (sum(ttft) / len(ttft)) if ttft else None,
        "wall_ttft_ms": (sum(wall_ttft) / len(wall_ttft)) if wall_ttft else None,
        "itl_ms": (sum(itl) / len(itl)) if itl else None,
        "replicas": replica_results,
        "steps": [step for result in replica_results for step in result["steps"]],
        "memory": memory.to_dict(),
        "live_kv_tokens_peak": max(
            (sum(request.prompt_tokens + request.output_tokens for request in group)
             for group in replicas),
            default=0,
        ),
        "live_kv_blocks_peak": max(
            (sum(math.ceil((request.prompt_tokens + request.output_tokens) / max(1, runtime.kv_block_size))
                 for request in group)
             for group in replicas),
            default=0,
        ),
        "offload_modeled": False,
        "recompute_modeled": False,
        "witness_kind": "global_per_replica_per_layer_token_step_resource_schedule",
        "shared_resource_timeline": {
            "one_engine": True,
            "link_names": sorted(link_names),
            "link_events": len(link_events),
            "dynamic_capacity": "equal_share_over_live_active_tasks",
            "static_bandwidth_share": False,
        },
        "fusion_effect_claimed": bool(applied_rules),
        "fusion_summary": {
            "applied_rules": applied_rules,
            "removed_intermediates": removed,
            "changes": fusion_changes,
            "resource_dependency_witness": next(
                (
                    (step.get("fusion") or {}).get("witness")
                    for result in replica_results for step in result["steps"]
                    if (step.get("fusion") or {}).get("witness")
                ),
                None,
            ),
        },
        "prepare": {
            "replicas": [
                {"replica": replica_index, **prepare_summary(
                    engine_result, next(
                        result["wall_makespan_ms"] for result in replica_results
                        if result["replica"] == replica_index
                    ), prepare_ids.get(replica_index, ()),
                    residency=prepare_residency_report(engine),
                )}
                for replica_index in active_replicas
            ],
            "model_derived_gaps": list(model_prepare_gaps),
            "note": "preparation/drain is separate from target token completion",
        },
        "engine_drain_makespan_ms": engine_result["makespan_ms"],
        "note": "U is a feasible schedule/witness in this enumerated family; it is not a global lower bound.",
    }


def _coerce_should_case(case: CaseSpec | Mapping[str, Any]) -> CaseSpec:
    if isinstance(case, CaseSpec):
        machine, model, workload, slo = case.machine, case.model, case.workload, case.slo
    else:
        raw = dict(case)
        machine = MachineSpec.from_mapping(raw.get("machine"))
        model = ModelSpec.from_mapping(raw.get("model"))
        workload = WorkloadSpec.from_mapping(raw.get("workload"))
        slo = SloSpec.from_mapping(raw.get("slo")) if raw.get("slo") is not None else SloSpec()
    max_len = max(
        request.prompt_tokens + request.output_tokens for request in workload.requests
    )
    # The normative path must not inherit current chunking/parallel legality.
    source_runtime = case.runtime if isinstance(case, CaseSpec) else RuntimeSpec.from_mapping(
        dict(case).get("runtime") or {}
    )
    media_meta = {}
    try:
        from .a3_catalog_media import media_runtime_metadata

        media_meta = media_runtime_metadata(workload)
    except ImportError:
        media_meta = {}
    merged_metadata = dict(source_runtime.metadata)
    merged_metadata.update(media_meta)
    runtime = RuntimeSpec(
        max_num_batched_tokens=max(1, max_len),
        max_num_seqs=max(1, len(workload.requests)),
        max_model_len=max(1, max_len),
        enable_chunked_prefill=True,
        prepare=source_runtime.prepare,
        model_prepare=source_runtime.model_prepare,
        prepare_host_rates=source_runtime.prepare_host_rates,
        communication_backend=source_runtime.communication_backend,
        communication_observed_backend=source_runtime.communication_observed_backend,
        metadata=tuple(sorted(merged_metadata.items())),
    )
    return CaseSpec(machine=machine, model=model, workload=workload, runtime=runtime, slo=slo)


def _diverse_plan_order(plans: Sequence[PlanOption]) -> list[PlanOption]:
    """Round-robin topology groups, then optimization variants within each.

    Small-device plans no longer consume the entire finite budget before a
    larger-device topology is attempted.  The order is deterministic and keeps
    fusion/graph/recipe variants adjacent inside every topology group.
    """
    groups: dict[tuple[Any, ...], list[PlanOption]] = {}
    for plan in plans:
        key = (plan.devices, plan.tp, plan.dp, plan.pp, plan.ep)
        groups.setdefault(key, []).append(plan)
    for key in groups:
        groups[key].sort(key=lambda item: (
            item.fusion, item.graph_padding, item.moe_recipe,
        ))
    sorted_keys = sorted(groups, key=lambda key: (
        key[0], key[1], key[3], key[4], key[2],
    ))
    # Alternate small and large topology groups so a finite budget cannot
    # exclude every larger/faster device topology before it is attempted.
    ordered_keys: list[tuple[Any, ...]] = []
    left, right = 0, len(sorted_keys) - 1
    while left <= right:
        ordered_keys.append(sorted_keys[left])
        if left != right:
            ordered_keys.append(sorted_keys[right])
        left += 1
        right -= 1
    ordered: list[PlanOption] = []
    depth = 0
    while True:
        added = False
        for key in ordered_keys:
            bucket = groups[key]
            if depth < len(bucket):
                ordered.append(bucket[depth])
                added = True
        if not added:
            break
        depth += 1
    return ordered


def should_cost(
    case: CaseSpec | Mapping[str, Any],
    *,
    constraints: Mapping[str, Any] | None = None,
    search_budget: int | None = None,
    budget: int | None = None,
) -> dict[str, Any]:
    if search_budget is not None or budget is not None:
        constraints = dict(constraints or {})
        constraints["search_budget"] = (
            search_budget if search_budget is not None else budget
        )
    case = _coerce_should_case(case)
    from .a3_request import with_media_runtime

    case = with_media_runtime(case)
    semantic = semantic_work(case.model, case.workload, ParallelSpec())
    lower = hardware_lower_bound(semantic, case.machine, case.model, ParallelSpec())
    lb = float(lower["hardware_lower_bound_ms"])
    plans = _candidate_plans(case.model, case.machine, case.workload, constraints)
    family_size = len(plans)
    filtered: list[PlanOption] = []
    prefiltered_infeasible: list[dict[str, Any]] = []
    for candidate in plans:
        try:
            candidate.parallel.validate(case.model)
            runtime_probe = _plan_runtime(case, candidate, constraints=constraints)
            memory_probe = memory_plan(case.model, runtime_probe, case.machine)
            if memory_probe.status == "fits":
                filtered.append(candidate)
            else:
                prefiltered_infeasible.append({
                    "status": "infeasible", "reason": "memory_infeasible",
                    "plan": candidate.to_dict(), "memory": memory_probe.to_dict(),
                })
        except A3InputError as exc:
            prefiltered_infeasible.append({
                "status": "infeasible", "reason": exc.reason,
                "plan": candidate.to_dict(),
            })
            continue
    plans = filtered
    eligible_after_prefilter = len(plans)
    # Caller-controlled finite budget.  An explicit topology domain is
    # preserved in full unless the caller also asks for a budget.  Without
    # constraints the simulator evaluates a deterministic topology-diverse
    # sample and reports attempted/feasible/unexplored counts explicitly.
    explicit_topology = bool(
        constraints and any(
            constraints.get(key) is not None
            for key in (
                "tp", "pp", "dp", "ep", "fusion", "graph_padding",
                "moe_recipe", "search_budget", "max_plans", "budget",
            )
        )
    )
    budget_raw = None
    if constraints:
        budget_raw = constraints.get(
            "search_budget",
            constraints.get(
                "max_plans",
                constraints.get("budget", constraints.get("plan_budget")),
            ),
        )
    if budget_raw is not None:
        try:
            limit = int(budget_raw)
        except (TypeError, ValueError) as exc:
            raise A3InputError("should_search_budget_invalid") from exc
        if limit < 1:
            raise A3InputError("should_search_budget_invalid")
    else:
        limit = len(plans) if explicit_topology else 64
    ordered_plans = _diverse_plan_order(plans)
    truncated = len(plans) > limit
    attempted_plans = min(len(plans), limit)
    plans = ordered_plans[:limit]
    evaluated: list[dict[str, Any]] = []
    infeasible: list[dict[str, Any]] = []
    for plan in plans:
        try:
            result = _evaluate_normative_plan(case, plan, constraints)
        except A3InputError as exc:
            if not str(exc).startswith("prepare_capacity_"):
                raise
            result = {
                "status": "infeasible", "reason": str(exc),
                "plan": plan.to_dict(),
            }
        if result.get("status") != "modeled":
            infeasible.append(result)
            continue
        target = float(result["target_ms"])
        if target + 1e-9 < lb:
            result["status"] = "infeasible"
            result["reason"] = "target_below_hardware_lower_bound"
            result["lower_bound_ms"] = lb
            infeasible.append(result)
            continue
        evaluated.append(result)
    evaluated.sort(key=lambda item: item["target_ms"])
    if not evaluated:
        return ShouldResult(
            status="infeasible", hardware_lower_bound_ms=lb, feasible_target_ms=None,
            gap_ms=None, gap_ratio=None, optimal_in_family=False,
            plan_family_size=family_size, evaluated_plans=0, selected_plan=None,
            plans=tuple((prefiltered_infeasible + infeasible)[:20]), work=semantic,
            reason="no_legal_plan",
        ).to_dict() | {
            "lower_bound_detail": lower,
            "infeasible_plans": (prefiltered_infeasible + infeasible)[:20],
            "plans_generated": family_size,
            "plans_eligible_after_prefilter": eligible_after_prefilter,
            "plans_attempted": attempted_plans,
            "plans_evaluated": 0,
            "plans_feasible": 0,
            "plans_successful": 0,
            "plans_infeasible_attempted": attempted_plans,
            "plans_pruned_memory_or_legality": len(prefiltered_infeasible),
            "plans_unexplored_after_beam": max(
                0, eligible_after_prefilter - attempted_plans
            ),
            "plan_family_truncated": truncated,
            "search_budget": budget_raw if budget_raw is not None else None,
            "search_budget_effective": limit,
        }
    best = evaluated[0]
    target = float(best["target_ms"])
    selected = PlanOption(**{
        key: best["plan"][key] for key in (
            "tp", "ep", "dp", "pp", "fusion", "graph_padding", "moe_recipe",
            "batch_tokens", "worker_count",
        )
    })
    public_plans = []
    for item in evaluated[:20]:
        public_plans.append({
            **{key: value for key, value in item.items() if key != "replicas"},
            "replica_summary": [
                {
                    "replica": replica["replica"],
                    "requests": replica["requests"],
                    "makespan_ms": replica["makespan_ms"],
                    "step_count": len(replica["steps"]),
                    "draft_requests": replica.get("draft_requests", {}),
                }
                for replica in item["replicas"]
            ],
        })
    result = ShouldResult(
        status="modeled", hardware_lower_bound_ms=lb, feasible_target_ms=target,
        gap_ms=target - lb, gap_ratio=(target - lb) / lb if lb > 0 else None,
        optimal_in_family=not truncated, plan_family_size=family_size,
        evaluated_plans=len(evaluated), selected_plan=selected,
        plans=tuple(public_plans), work=semantic,
    ).to_dict()
    result["lower_bound_detail"] = lower
    selected_placement = best["plan"].get("placement")
    result["selected_plan"]["placement"] = selected_placement
    result["infeasible_plans"] = (prefiltered_infeasible + infeasible)[:20]
    result["bound_consistency"] = {
        "lower_bound_le_target": lb <= target + 1e-9,
        "tolerance_ms": 1e-9,
    }
    result["plan_family_truncated"] = truncated
    result["plans_pruned_memory_or_legality"] = len(prefiltered_infeasible)
    result["plans_attempted"] = attempted_plans
    result["plans_evaluated"] = attempted_plans
    result["plans_feasible"] = len(evaluated)
    result["plans_successful"] = len(evaluated)
    result["plans_infeasible_attempted"] = max(
        0, attempted_plans - len(evaluated)
    )
    result["plans_unexplored_after_beam"] = max(
        0, eligible_after_prefilter - attempted_plans
    )
    result["plans_generated"] = family_size
    result["plans_eligible_after_prefilter"] = eligible_after_prefilter
    result["search_budget"] = budget_raw if budget_raw is not None else None
    result["search_budget_effective"] = limit
    result["intrinsic_target_ms"] = best.get("intrinsic_target_ms")
    result["intrinsic_ttft_ms"] = best.get("intrinsic_ttft_ms")
    result["prepare"] = best.get("prepare")
    result["engine_drain_makespan_ms"] = best.get("engine_drain_makespan_ms")
    result["fusion_effect_claimed"] = bool(
        result["selected_witness"].get("applied_rules")
    ) if "selected_witness" in result else bool(
        best.get("fusion_effect_claimed", False)
    )
    result["selected_witness"] = {
        "kind": best["witness_kind"],
        "target_ms": best["target_ms"],
        "replicas": [
            {
                "replica": replica["replica"], "requests": replica["requests"],
                "draft_requests": replica.get("draft_requests", {}),
                "makespan_ms": replica["makespan_ms"],
                "wall_makespan_ms": replica["wall_makespan_ms"],
                "first_token_ms": replica["first_token_ms"],
                "service_first_token_ms": replica["service_first_token_ms"],
                "finish_ms": replica["finish_ms"],
                "service_finish_ms": replica["service_finish_ms"],
                "steps": [
                    {
                        **{key: value for key, value in step.items() if key != "events"},
                        "events": step.get("events", []),
                        "event_count": len(step.get("events", [])),
                    }
                    for step in replica["steps"]
                ],
            }
            for replica in best["replicas"]
        ],
        "applied_rules": best.get("fusion_summary", {}).get("applied_rules", []),
        "removed_intermediates": best.get("fusion_summary", {}).get("removed_intermediates", []),
        "fusion_changes": best.get("fusion_summary", {}).get("changes", {}),
        "resource_dependency_witness": best.get("fusion_summary", {}).get("resource_dependency_witness"),
        "timeline_basis": "intrinsic_service_time_origin_zero",
        "arrival_policy": "normative service time only; wall arrivals remain in predictive",
        "note": "Per-replica prefill/decode steps and per-layer resource events; no pooled DP*TP*PP resource.",
    }
    result["selected_witness"]["placement"] = selected_placement
    worker_contracts = []
    for replica in best.get("replicas", ()):
        for step in replica.get("steps", ()):
            contract = (
                ((step.get("fusion") or {}).get("witness") or {})
                .get("core_contract")
            )
            if contract:
                worker_contracts.append(contract)
    result["finite_worker_plan"] = {
        "requested_worker_count": int(selected.worker_count),
        "actual_worker_counts": sorted({
            int(item.get("worker_count", 1)) for item in worker_contracts
        }),
        "core_contracts": worker_contracts,
        "basis": "bounded private-buffer worker partitions consumed by the actual fusion stage bindings",
    }
    return result


# ---------------------------------------------------------------------------
# Normalized algorithm acceptance fixtures.  These are NOT A3 measurements.
# ---------------------------------------------------------------------------



__all__ = [
    "SemanticWork", "PlanOption", "ShouldResult", "semantic_work",
    "hardware_lower_bound", "should_cost", "normalized_shared_cv_search",
    "chunked_pipeline_makespan", "normalized_chunked_pipeline_search",
]
