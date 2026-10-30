"""Compose scheduler cohorts and pipeline microbatches.

[Role] Share one dispatch path between colocated and disaggregated serving.
[Invariants] Idle EP participants reach the same collectives; PP microbatches
  share actual stage streams and resources.
[Timeouts & Locks] Bounded by the active batch, no I/O.
[Side-Effect Policy] Submits to the caller's execution engine.
"""
from dataclasses import replace

from mindie_sim.schema.workload import IterationPlan


def cohorts(parallel):
    width = parallel.ep // parallel.tp if parallel.ep_spans_dp else 1
    return [tuple(range(first, first + width)) for first in range(0, parallel.dp, width)]


def submit_batch(case, schedulers, plans, execution, lower, now):
    from .frontend import Frontend
    count = min(case.runtime.pipeline_microbatches, max(len(p.requests) for p in plans.values() if p))
    steps = {replica: [] for replica in plans}
    for microbatch in range(count):
        local = {
            replica: replace(plan, requests=plan.requests[microbatch::count],
                             iteration=plan.iteration * case.runtime.pipeline_microbatches + microbatch)
            if plan else IterationPlan((), iteration=schedulers[replica].iteration)
            for replica, plan in plans.items()
        }
        global_views = tuple(view for plan in local.values() for view in plan.requests)
        context = ({"global_views": global_views, "global_tokens": sum(
            view.q_len + view.draft_proposal_tokens for view in global_views)} if len(plans) > 1 else {})
        sp_padding = case.runtime.kernels.get("sequence_parallel", {}).get("padding")
        if len(plans) > 1 and (case.runtime.routing.get("policy") == "balanced"
                               or sp_padding == "right_to_tp"):
            from .balanced_routing import cohort_padded_tokens
            context["global_tokens"] = cohort_padded_tokens(
                (plan.requests for plan in local.values()), case.runtime)
        # The built-in frontend owns physical shard selection. Custom lowerers
        # keep their existing callable contract and their own placement policy.
        works = {replica: lower(plan, case.model, schedulers[replica].next_runtime(plan), case.machine,
                               **context, **({"replica": replica} if isinstance(lower, Frontend) else {}))
                 for replica, plan in local.items()}
        emitted = {}
        for replica, plan in local.items():
            emitted[replica] = {}
            for view in plan.requests:
                state = schedulers[replica].by_id[view.request_id]
                count = (int(view.draft_advanced_tokens or 1) if view.phase == "decode"
                         else int(view.samples and not state.generated))
                count = min(count, state.request.output_tokens - state.generated)
                if count:
                    emitted[replica][view.request_id] = count
        submitted = execution.add_group(works, now, emitted_tokens=emitted) if len(plans) > 1 else {
            replica: execution.add(work, replica, now, emitted_tokens=emitted[replica])
            for replica, work in works.items()}
        for replica, step in submitted.items():
            steps[replica].append(step)
            record_deliveries(execution, schedulers[replica], step, emitted[replica])
    result = {}
    for replica, items in steps.items():
        completion = execution.engine.submit(
            f"{execution.namespace}r{replica}:batch_complete", after=tuple(s.completion for s in items))
        result[replica] = replace(
            items[0], completion=completion, operators=sum(s.operators for s in items),
            valid_tokens=sum(s.valid_tokens for s in items), padded_tokens=sum(s.padded_tokens for s in items),
            commits={request: event for item in items for request, event in item.commits.items()},
            host_results={request: event for item in items for request, event in item.host_results.items()},
            client_deliveries={request: event for item in items for request, event in item.client_deliveries.items()},
            implementation_selections=tuple(
                {**decision, "microbatch": item.iteration}
                for item in items for decision in item.implementation_selections),
            device_complete=execution.engine.submit(
                f"{execution.namespace}r{replica}:batch_device_complete",
                after=tuple(s.device_complete for s in items)),
        )
        plan = plans[replica]
        advances = {view.request_id: int(view.draft_advanced_tokens or view.samples)
                    for view in plan.requests} if plan else {}
        handoffs = ({view.request_id for view in plan.requests if view.samples}
                    if plan and schedulers[replica].handoff_after_prefill else set())
        if any(not state.done and schedulers[replica].eligible(state)
               and state.request.request_id not in handoffs
               and state.generated + advances.get(state.request.request_id, 0) < state.request.output_tokens
               for state in schedulers[replica].states):
            execution.prepare_next(replica)
    return result


def record_deliveries(execution, scheduler, step, emitted):
    """Observe real event callbacks without retaining a token history."""
    policy = execution.serving_io
    for boundary, events in (("host_result", step.host_results),
                             ("client_delivery", step.client_deliveries)):
        for request, event in events.items():
            state = scheduler.by_id[request]
            count = emitted[request]

            def record(done, state=state, count=count, boundary=boundary, event=event):
                timeline = getattr(state, boundary)
                timeline.record(count, done.time)
                if boundary == "client_delivery" and policy.client_tails.get(state.request.request_id) is event:
                    del policy.client_tails[state.request.request_id]
                release_boundary = "client_delivery" if policy.client_delivery_ms is not None else "host_result"
                successor = scheduler.successors.get(state.request.request_id)
                if (policy.enabled and boundary == release_boundary and successor is not None
                        and timeline.tokens >= state.request.output_tokens):
                    successor.release_ms = max(successor.request.arrival_ms,
                                               done.time + successor.request.gap_ms)

            execution.engine.submit(event.name + ":record:" + request, after=(event,), on_complete=record)
