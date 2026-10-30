"""Public prediction using one causal request and resource timeline.

[Role] Compose semantics, recipes, placement, cache and scheduling policies.
[Invariants] TTFT/TPOT/TPS come from committed events, never model-specific
  corrections. Calibration supplies primitive parameters. Independent
  validation is a separate operation and does not gate simulation.
[Timeouts & Locks] Finite request work and bounded diagnostic history; no I/O.
[Side-Effect Policy] Inputs remain unchanged; only local simulation state mutates.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Callable, Mapping, Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan
from .capacity import Memory, MemoryPolicy
from .cost import CostModel
from .execution import Execution, ScheduledIteration, SubmissionPolicy
from .metrics import summarize
from .primitives import PrimitiveStore
from .program import RecipeBook
from .scheduler import Scheduler
from .frontend import Frontend
from .attention import AttentionPolicy
from .structure import parse_case
from mindie_sim.coverage import model_coverage


def predict(
    case: CaseSpec | Mapping[str, Any], *, primitives: PrimitiveStore | None = None,
    recipes: RecipeBook | None = None, submission: SubmissionPolicy | None = None,
    memory_policy: MemoryPolicy | None = None, lower: Callable | None = None,
    scheduler_factory: Callable = Scheduler, trace_limit: int = 2,
    event_limit: int = 256, scope: str = "serving", phase: str | None = None,
    tokens: int | None = None, context: int | None = None, sequences: int | None = None,
) -> dict[str, Any]:
    if scope == "body":
        if lower is not None:
            raise ModelingError("body_scope_uses_target_frontend_not_custom_serving_lowerer")
        from .body import body_forward, pd_body_forward
        probe = pd_body_forward if isinstance(case, Mapping) and "pools" in case else body_forward
        return probe(case, phase=phase or "decode", tokens=1 if tokens is None else tokens,
                     context=0 if context is None else context,
                     sequences=1 if sequences is None else sequences,
                     primitives=primitives, recipes=recipes, event_limit=event_limit)
    if scope != "serving" or any(v is not None for v in (phase, tokens, context, sequences)):
        raise ModelingError("prediction_geometry_requires_body_scope")
    if isinstance(case, Mapping) and "pools" in case:
        from .disaggregated import predict_pd
        return predict_pd(case, primitives=primitives, recipes=recipes, submission=submission,
                          memory_policy=memory_policy, lower=lower, trace_limit=trace_limit,
                          event_limit=event_limit)
    case = parse_case(case)
    if case.model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    lower = lower or Frontend(attention_policy=AttentionPolicy(**case.runtime.attention_policy))
    identifiers = [request.request_id for request in case.workload.requests]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("request_ids_must_be_unique")
    memory = Memory(case, memory_policy)
    parallel = case.runtime.parallel
    groups = [[] for _ in range(parallel.dp)]
    chains = {}
    for index, request in enumerate(sorted(case.workload.requests,
                                          key=lambda r: (r.arrival_ms, r.request_id))):
        replica = chains.setdefault(request.chain_id, index % parallel.dp) if request.chain_id else index % parallel.dp
        groups[replica].append(request)
    schedulers = [scheduler_factory(case, group, memory, replica)
                  for replica, group in enumerate(groups)]
    cost = CostModel(case, primitives)
    execution = Execution(case, cost, recipes=recipes, submission=submission, trace_limit=event_limit)
    engine = execution.engine
    engine.now = min((r.arrival_ms for r in case.workload.requests), default=0)
    in_flight: dict[int, ScheduledIteration] = {}
    traces, modes, scopes = [], {}, {}
    iterations = 0
    from .dispatch import cohorts, submit_batch
    groups_to_schedule = cohorts(parallel)
    while not all(scheduler.done for scheduler in schedulers):
        for cohort in groups_to_schedule:
            if any(replica in in_flight for replica in cohort) or all(schedulers[r].done for r in cohort):
                continue
            plans = {r: schedulers[r].schedule(engine.now) if not schedulers[r].done else None for r in cohort}
            if not any(plans.values()):
                continue
            steps = submit_batch(case, schedulers, plans, execution, lower, engine.now)
            for replica, step in steps.items():
                iterations += 1
                modes[step.mode] = modes.get(step.mode, 0) + 1
                scopes[step.replay_scope] = scopes.get(step.replay_scope, 0) + 1
                in_flight[replica] = step
        future = [arrival for replica, scheduler in enumerate(schedulers)
                  if replica not in in_flight
                  if (arrival := scheduler.next_arrival(engine.now)) is not None]
        arrival = min(future) if future else float("inf")
        if not in_flight:
            if arrival == float("inf"):
                break
            engine.now = arrival
            continue
        engine.advance(until=tuple(step.completion for step in in_flight.values()), horizon=arrival)
        for replica, step in list(in_flight.items()):
            if not step.completion.done:
                continue
            schedulers[replica].commit(step.completion.time,
                                      {key: event.time for key, event in step.commits.items()})
            if len(traces) < trace_limit:
                traces.append(step.to_dict())
            del in_flight[replica]
    serving_end = engine.now
    # Async scheduling can leave an unused finite CPU preparation after an
    # admission failure. Drain actual work without extending the token window.
    engine.advance()
    rows = sorted((state.to_dict() for scheduler in schedulers for state in scheduler.states),
                  key=lambda row: (row["arrival_ms"], row["request_id"]))
    metrics = summarize(rows, case.slo, now_ms=serving_end)
    return {
        "schema": "mindie-sim.predict/2",
        **model_coverage(case.model),
        "status": "modeled" if all(row["status"] == "served" for row in rows) else "infeasible",
        "execution_status": "completed" if all(row["status"] == "served" for row in rows) else "blocked",
        "metrics": metrics, "requests": rows, "memory": memory.to_dict(),
        "execution": {**execution.summary(), "iterations": iterations,
                      "iteration_modes": modes, "graph_scopes": scopes},
        "parameters": cost.summary(),
        "assumptions": {
            "speculative_acceptance": {key: policy.basis for scheduler in schedulers
                                      for key, policy in scheduler.drafts.items()},
        },
        "traces": traces, "events": list(engine.trace),
        "plan": {"model": case.model.to_dict(), "runtime": case.runtime.to_dict(),
                 "machine": case.machine.to_dict()},
        "metric_boundary": "device_token_commit",
    }
