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
        submitted = execution.add_group(works, now) if len(plans) > 1 else {
            replica: execution.add(work, replica, now) for replica, work in works.items()}
        for replica, step in submitted.items():
            steps[replica].append(step)
    result = {}
    for replica, items in steps.items():
        completion = execution.engine.submit(
            f"{execution.namespace}r{replica}:batch_complete", after=tuple(s.completion for s in items))
        result[replica] = replace(
            items[0], completion=completion, operators=sum(s.operators for s in items),
            valid_tokens=sum(s.valid_tokens for s in items), padded_tokens=sum(s.padded_tokens for s in items),
            commits={request: event for item in items for request, event in item.commits.items()},
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
