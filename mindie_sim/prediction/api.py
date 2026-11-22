"""Public prediction using one causal request and resource timeline.

[Role] Compose semantics, recipes, placement, cache and scheduling policies.
[Invariants] TTFT/TPOT/TPS come from committed events, never model-specific
  corrections. Calibration supplies primitive parameters. Independent
  validation is a separate operation and does not gate simulation.
[Timeouts & Locks] Finite request work and bounded diagnostic history; no I/O.
[Side-Effect Policy] Inputs remain unchanged; only local simulation state mutates.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from hashlib import sha256
import json
import math
from collections.abc import Mapping as _AbcMapping
from typing import Callable, Mapping, Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.workload import IterationPlan
from .capacity import Memory, MemoryPolicy
from .cost import CostModel
from .execution import Execution, ScheduledIteration, SubmissionPolicy
from .metrics import serving_boundaries, summarize
from .primitives import PrimitiveStore
from .program import RecipeBook
from .scheduler import Scheduler
from .frontend import Frontend
from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from .structure import parse_case
from mindie_sim.coverage import model_coverage

ZERO = 0
DEFAULT_TRACE_LIMIT = 2
DEFAULT_EVENT_LIMIT = 256


def _runtime_identity(case: CaseSpec) -> str:
    """Stable public runtime label derived from the parsed declaration."""
    payload = json.dumps(case.to_dict(), sort_keys=True,
                         separators=(",", ":"), allow_nan=False)
    return sha256(payload.encode("utf-8")).hexdigest()


def _event_pending(event) -> bool:
    return event is not None and getattr(event, "done", True) is False


def _memory_pools(execution) -> tuple[str, ...]:
    """Physical buffer pools declared by this Execution's stage bindings."""
    pools: list[str] = []
    for mapping in dict(getattr(execution, "stage_pools", {}) or {}).values():
        for declared in dict(mapping):
            if declared not in pools:
                pools.append(str(declared))
    return tuple(pools) or ("hbm",)


def _bind_service_context(case, execution, schedulers, memory):
    """Validate every caller/Scheduler/conductor/pool identity, then bind once.

    No field of the caller's Execution is changed before all checks pass.  A
    different already-bound ledger is a conflict, never a silent replacement;
    another pool already active in the same Engine therefore keeps its work.
    """
    if memory is None:
        raise ModelingError("cohort_requires_memory_ledger")
    engine = getattr(execution, "engine", None)
    if engine is None:
        raise ModelingError("cohort_requires_execution_engine")
    dp = int(case.runtime.parallel.dp)
    if len(schedulers) != dp:
        raise ModelingError(
            f"cohort_scheduler_count_mismatch:{len(schedulers)}:{dp}")
    bound = getattr(execution, "memory", None)
    if bound is not None and bound is not memory:
        raise ModelingError("execution_memory_ledger_conflict")
    ledger = getattr(execution, "memory_ledger", None)
    if ledger is not None and ledger is not memory:
        raise ModelingError("execution_memory_ledger_conflict")
    if getattr(engine, "memory", None) is not None and engine.memory is not memory:
        raise ModelingError("engine_memory_ledger_conflict")
    for key, conductor in dict(
            getattr(execution, "fragment_conductors", {}) or {}).items():
        conductor_memory = getattr(conductor, "memory", None)
        if conductor_memory is not None and conductor_memory is not memory:
            raise ModelingError(f"conductor_memory_ledger_conflict:{key}")
    for replica, scheduler in enumerate(schedulers):
        if int(getattr(scheduler, "replica", replica)) != replica:
            raise ModelingError(
                f"cohort_scheduler_replica_mismatch:{replica}")
        if getattr(scheduler, "memory", None) is not memory:
            raise ModelingError("scheduler_memory_ledger_conflict")
        scheduler_case = getattr(scheduler, "case", None)
        if scheduler_case is not None and scheduler_case is not case:
            raise ModelingError(
                f"cohort_scheduler_case_mismatch:{replica}")
    # Every identity above is known to be compatible. Use T04's idempotent
    # binding entry instead of assigning execution/engine fields directly.
    if bound is None:
        if execution.bind_memory(
                memory, replica=0, stage=0, pools=_memory_pools(execution)) is None:
            raise ModelingError("execution_memory_ledger_conflict")
    # Existing caller-visible alias of the same ledger (not a second ledger).
    if getattr(execution, "memory_ledger", None) is None:
        execution.memory_ledger = memory
    if getattr(engine, "memory", None) is None:
        if not engine.bind_memory(memory, pools=_memory_pools(execution)):
            raise ModelingError("engine_memory_ledger_conflict")
    execution.schedulers = schedulers
    return engine


def _selected_demand(effects) -> float:
    return sum(float(getattr(effect, "size", 0.0))
               for effect in effects
               if getattr(effect, "action", "acquire") == "acquire")


def _selected_error(memory, effects, replica: int, stage: int):
    demand = _selected_demand(effects)
    capacity = float(memory.capacity[replica][stage])
    used = (memory.live[replica][stage]
            + (memory.staging_live[replica] if stage == 0 else 0.0)
            + memory.granted[replica][stage]
            + memory.shared_reserved[replica][stage]
            + memory.selected_reserved[replica][stage])
    available = max(0.0, capacity - used)
    reason = ("permanent_capacity" if demand > capacity
              else "waiting_for_release")
    detail = ("demand_exceeds_capacity" if demand > capacity
              else "no_pending_release_event")
    return ModelingError(
        "memory_selected_allocation_rejected:"
        f"{reason}:resource=hbm@r{replica}:s{stage}:"
        f"demand={demand:.0f}:available={available:.0f}:"
        f"capacity={capacity:.0f}:permanent=1:detail={detail}")


def _memory_wake_events(memory, replicas) -> tuple:
    """Real pending Events that can change this service's admission."""
    found: dict[int, Any] = {}

    def add(value):
        if isinstance(value, _AbcMapping):
            value = value.get("event")
        if _event_pending(value):
            found[id(value)] = value

    for (replica, _name), record in dict(
            getattr(memory, "grant_records", {}) or {}).items():
        if replica in replicas:
            add(dict(record).get("event"))
    for (replica, _name), binding in dict(
            getattr(memory, "grant_bindings", {}) or {}).items():
        if replica in replicas:
            add(dict(binding).get("event"))
    for (replica, _name), readers in dict(
            getattr(memory, "grant_readers", {}) or {}).items():
        if replica in replicas:
            for snapshot in dict(readers).values():
                add(dict(snapshot).get("event"))
    for (replica, _request), holds in dict(
            getattr(memory, "async_consumer_holds", {}) or {}).items():
        if replica in replicas:
            for snapshot in dict(holds).values():
                binding = getattr(snapshot, "binding", None)
                add(dict(binding or {}).get("event"))
    return tuple(found.values())


def _service_tails(execution, in_flight, tails: set) -> tuple:
    """Own host/client tails plus real copy/last-use release boundaries."""
    found: dict[int, Any] = {}
    for event in tails:
        if _event_pending(event):
            found[id(event)] = event
    for event in dict(
            getattr(execution.serving_io, "client_tails", {}) or {}).values():
        if _event_pending(event):
            found[id(event)] = event
    for step in in_flight.values():
        if _event_pending(getattr(step, "offload_release", None)):
            event = step.offload_release
            found[id(event)] = event
        for event in dict(getattr(step, "host_results", {}) or {}).values():
            if _event_pending(event):
                found[id(event)] = event
        for event in dict(getattr(step, "client_deliveries", {}) or {}).values():
            if _event_pending(event):
                found[id(event)] = event
    return tuple(found.values())


def _quota_exhausted(state, quota) -> bool:
    request_id = state.request.request_id
    return request_id in quota and state.generated >= quota[request_id]


def _quota_boundary(schedulers, quota) -> bool:
    if not quota:
        return False
    unfinished = False
    for scheduler in schedulers:
        for state in scheduler.states:
            if state.done or _quota_exhausted(state, quota):
                continue
            if state.release_ms == math.inf:
                predecessor = next(
                    (request_id for request_id, successor
                     in scheduler.successors.items() if successor is state),
                    None)
                if predecessor is not None:
                    prior = scheduler.by_id.get(predecessor)
                    if prior is not None and _quota_exhausted(prior, quota):
                        continue
            unfinished = True
            break
        if unfinished:
            break
    return not unfinished


@contextmanager
def _quota_scope(schedulers, quota):
    """Apply the observation/emission quota only around schedule().

    ``Progress.request.output_tokens`` keeps its original request value at all
    commit and retirement boundaries. The cap exists only while Scheduler
    forms the next plan, so a bounded window cannot turn a short remainder into
    a completed request or release its closed-loop successor.
    """
    saved_release = []
    saved_request = []
    try:
        if quota:
            for scheduler in schedulers:
                for state in scheduler.states:
                    request_id = state.request.request_id
                    cap = quota.get(request_id)
                    if cap is None:
                        continue
                    if state.generated >= cap:
                        if state.release_ms != math.inf:
                            saved_release.append((state, state.release_ms))
                            state.release_ms = math.inf
                    elif state.request.output_tokens > cap:
                        saved_request.append((state, state.request))
                        state.request = replace(
                            state.request, output_tokens=int(cap))
        yield
    finally:
        for state, value in saved_release:
            state.release_ms = value
        for state, request in saved_request:
            state.request = request


def run_cohort(case, execution, schedulers, memory, lower, *, trace_limit=DEFAULT_TRACE_LIMIT,
               max_model_steps=None, observer=None, token_quotas=None,
               selected_allocations=None, selected_replica: int = ZERO,
               selected_stage: int = ZERO):
    """Event-driven cohort frontier shared by predict/service/assemble adapters.

    The caller owns Execution, Scheduler/Memory and the lowerer. This function
    only advances the existing Engine: schedule -> dispatch -> commit ->
    feedback, with a bounded stop/observer hook. It never constructs a second
    Engine or converts isolated model duration into delay.

    ``token_quotas`` is a per-request observation/emission boundary, not a
    replacement ``output_tokens``. ``selected_allocations`` may wait for a real
    Event owned by this service; permanent refusal happens before any issue.
    """
    from .implementation_selection import combine_summaries
    engine = _bind_service_context(case, execution, schedulers, memory)
    replicas = set(range(len(schedulers)))
    quota = {str(key): max(0, int(value))
             for key, value in dict(token_quotas or {}).items()}
    pending_selected = tuple(selected_allocations or ())
    selected_attached = not pending_selected

    def attach_selected() -> bool:
        nonlocal pending_selected, selected_attached
        if selected_attached:
            return True
        if memory.attach_allocation_effects(
                pending_selected, replica=selected_replica,
                stage=selected_stage):
            selected_attached = True
            pending_selected = ()
            return True
        return False

    if pending_selected:
        attach_selected()

    parallel = case.runtime.parallel
    in_flight: dict[int, ScheduledIteration] = {}
    traces, modes, scopes = [], {}, {}
    selections: list = []
    iterations = 0
    model_steps = 0
    logical_steps = 0
    tails: set = set()
    bindings: list[dict] = []
    assembly = {"declared_commits": 0, "fallback_commits": 0,
                "host_submits": 0, "microbatches": case.runtime.pipeline_microbatches,
                "batch_bindings": bindings}
    scheduler_feedback = {}
    stop_reason = "drained"
    from .dispatch import cohorts, submit_batch
    groups_to_schedule = cohorts(parallel, case.machine)

    while True:
        # 1. Commit only completed actual steps and remove their callbacks.
        for replica, step in list(in_flight.items()):
            if not step.completion.done:
                continue
            schedulers[replica].commit_batch(step)
            scheduler_feedback[replica] = schedulers[replica].feedback(
                engine.now, include_done=schedulers[replica].done)
            assembly["declared_commits"] += len(getattr(step, "declared_commits", {}))
            assembly["fallback_commits"] += len(getattr(step, "fallback_commits", {}))
            assembly["host_submits"] += len(getattr(step, "host_submits", ()))
            if len(bindings) > 256:
                del bindings[:len(bindings) - 256]
            tails.update(getattr(step, "host_results", {}).values())
            tails.update(getattr(step, "client_deliveries", {}).values())
            if _event_pending(getattr(step, "offload_release", None)):
                tails.add(step.offload_release)
            if step.mode != "retire":
                model_steps += 1
                if observer is not None:
                    observer(step)
                if len(traces) < trace_limit:
                    traces.append(step.to_dict())
            del in_flight[replica]

        real_done = all(scheduler.done for scheduler in schedulers)
        cap_reached = (max_model_steps is not None
                       and logical_steps >= max_model_steps)
        quota_reached = _quota_boundary(schedulers, quota)
        if real_done or cap_reached or quota_reached:
            if not in_flight:
                # The global submit cap is a caller-visible window boundary
                # even when the last allowed step also finished every request.
                if cap_reached:
                    stop_reason = "max_model_steps"
                elif real_done:
                    stop_reason = "drained"
                else:
                    stop_reason = "token_quota"
                break

        # 2. Selected allocation is a precondition to any new issue. While it
        # is unsatisfied only real release events/arrivals may advance time.
        if not selected_attached:
            attach_selected()
            if not selected_attached:
                wake = _memory_wake_events(memory, replicas)
                if not wake and not in_flight:
                    raise _selected_error(
                        memory, pending_selected, selected_replica, selected_stage)

        # 3. Form and submit new work only when no service precondition blocks.
        if selected_attached:
            for cohort in groups_to_schedule:
                if max_model_steps is not None and logical_steps >= max_model_steps:
                    break
                if (any(replica in in_flight for replica in cohort)
                        or all(schedulers[r].done for r in cohort)):
                    continue
                with _quota_scope(schedulers, quota):
                    plans = {
                        r: (schedulers[r].schedule(engine.now)
                            if not schedulers[r].done else None)
                        for r in cohort
                    }
                if not any(plans.values()):
                    for r in cohort:
                        if not schedulers[r].done and memory.has_delayed_releases(r):
                            prep = execution.prepared.pop(r, None)
                            sched_ms = (execution.submission.scheduler_ms
                                        if execution.include_submission else 0.0)
                            comp_ms = (execution.submission.completion_ms
                                       if execution.include_submission else 0.0)
                            sched_ev = engine.submit(
                                f"{execution.namespace}r{r}:retire_schedule:{iterations}",
                                {f"{execution.namespace}scheduler@r{r}": sched_ms},
                                after=(prep,) if prep is not None else (),
                                release=engine.now, owner=f"replica:{r}",
                            )
                            comp_ev = engine.submit(
                                f"{execution.namespace}r{r}:retire_complete:{iterations}",
                                after=(sched_ev,), latency=comp_ms,
                                release=engine.now, owner=f"replica:{r}",
                            )
                            in_flight[r] = ScheduledIteration(
                                r, schedulers[r].iteration, comp_ev, engine.now,
                                "retire", 0, 0, None, "none", 0, {}, comp_ev,
                                {}, {}, (),
                            )
                    continue
                steps = submit_batch(
                    case, schedulers, plans, execution, lower, engine.now,
                    bindings_out=bindings, selections_out=selections)
                # One atomic EP/cohort submission counts once against the global
                # decode budget; PP microbatch work remains inside it.
                if any(step.mode != "retire" for step in steps.values()):
                    logical_steps += 1
                for replica, step in steps.items():
                    iterations += 1
                    modes[step.mode] = modes.get(step.mode, 0) + 1
                    scopes[step.replay_scope] = scopes.get(step.replay_scope, 0) + 1
                    in_flight[replica] = step
                    tails.update(getattr(step, "host_results", {}).values())
                    tails.update(getattr(step, "client_deliveries", {}).values())
        # 4. Next arrival is a real clock boundary. Only this service's own
        # pending events can preempt it.
        future = [arrival for replica, scheduler in enumerate(schedulers)
                  if replica not in in_flight
                  if (arrival := scheduler.next_arrival(engine.now)) is not None]
        arrival = min(future) if future else math.inf
        targets = list(_service_tails(execution, in_flight, tails))
        targets.extend(_memory_wake_events(memory, replicas))
        targets.extend(step.completion for step in in_flight.values()
                       if _event_pending(step.completion))
        if not targets and arrival == math.inf:
            if not real_done:
                stop_reason = "blocked"
            break

        before = engine.now
        completed = engine.advance(until=tuple(targets),
                                   horizon=arrival)
        if (not completed and arrival != math.inf and engine.now < arrival
                and not any(step.completion.done for step in in_flight.values())):
            # No task existed and no target fired before the arrival horizon.
            engine.now = arrival
        if engine.now < before:
            raise ModelingError("cohort_time_moved_backwards")

    window_end = engine.now
    if selected_attached:
        # Terminal drain: every delivered host/client/copy tail this service
        # created, including successors released by a delivery callback. A
        # completed tail is removed and never wakes the same instant again.
        while True:
            pending = tuple(event for event in tails if _event_pending(event))
            pending += tuple(
                event for event in dict(
                    getattr(execution.serving_io, "client_tails", {}) or {}).values()
                if _event_pending(event))
            if not pending:
                break
            engine.advance(until=tuple(pending))
        if max_model_steps is None and not quota:
            memory.flush_delayed_releases()
    return {
        "engine": engine, "in_flight": in_flight, "iterations": iterations,
        "model_steps": model_steps, "logical_steps": logical_steps,
        "modes": modes, "scopes": scopes,
        "assembly": assembly, "scheduler_feedback": scheduler_feedback,
        "traces": traces, "serving_end": engine.now, "window_end": window_end,
        "stop_reason": stop_reason, "token_quotas": dict(quota),
        "selected_allocation_attached": bool(selected_attached),
        "operator_selection_summary": combine_summaries(selections),
    }


def predict(
    case: CaseSpec | Mapping[str, Any], *, primitives: PrimitiveStore | None = None,
    recipes: RecipeBook | None = None, submission: SubmissionPolicy | None = None,
    memory_policy: MemoryPolicy | None = None, lower: Callable | None = None,
    scheduler_factory: Callable = Scheduler,
    trace_limit: int = DEFAULT_TRACE_LIMIT,
    event_limit: int = DEFAULT_EVENT_LIMIT, scope: str = "serving",
    phase: str | None = None,
    tokens: int | None = None, context: int | None = None, sequences: int | None = None,
    max_model_steps: int | None = None, window=None, observer=None,
    include_submission: bool = True, resource_fusion: bool = True,
    memory: Memory | None = None, selected_allocations=None,
    memory_replica: int = ZERO, memory_stage: int = ZERO,
    token_quotas: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    # A declared attention core without a physical model may be inspected
    # structurally, but no latency/throughput entry may fabricate its work.
    # The gate precedes every early body/PD return.
    from .unmodeled_gate import assert_timing_supported
    gate_case = case
    if isinstance(case, _AbcMapping) and "pools" not in case:
        # Layer overrides are resolved during parse_case.  Gate the resolved
        # ModelSpec, not only the raw override dictionary, before a caller's
        # custom lowerer or a baseline can produce a timing number.
        try:
            gate_case = parse_case(case)
        except (ModelingError, ValueError, TypeError, KeyError):
            gate_case = case
    assert_timing_supported(case, entry="predict")
    if gate_case is not case:
        assert_timing_supported(gate_case, entry="predict")
    has_geometry = any(v is not None for v in (phase, tokens, context, sequences))
    if scope in {"body", "model"} or (scope == "serving" and has_geometry):
        if lower is not None:
            raise ModelingError("body_scope_uses_target_frontend_not_custom_serving_lowerer")
        from .body import body_forward, pd_body_forward
        resolved_scope = "model" if scope == "model" else "body"
        probe = pd_body_forward if isinstance(case, _AbcMapping) and "pools" in case else body_forward
        return probe(case, phase=phase or "decode", tokens=1 if tokens is None else tokens,
                     context=0 if context is None else context,
                     sequences=1 if sequences is None else sequences,
                     scope=resolved_scope,
                     primitives=primitives, recipes=recipes, event_limit=event_limit)
    if scope != "serving":
        raise ModelingError(f"invalid_prediction_scope:{scope}")
    if isinstance(case, _AbcMapping) and "pools" in case:
        from .disaggregated import predict_pd
        return predict_pd(case, primitives=primitives, recipes=recipes, submission=submission,
                          memory_policy=memory_policy, lower=lower, trace_limit=trace_limit,
                          event_limit=event_limit)
    case = parse_case(case)
    if case.model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    if lower is None and recipes is None:
        # A serving step lowers the same declared operators as the body entry.
        # Supply the same omitted physical program defaults so both scopes
        # select one compact finite-K representation; an explicit caller
        # recipe book still wins.
        from .body_defaults import service_program_defaults
        from mindie_sim.compose_operator_defaults import serving_recipe_inventory
        case = serving_recipe_inventory(service_program_defaults(case))
    lower = lower or Frontend(
        attention_policy=AttentionPolicy(**case.runtime.attention_policy),
        resource_fusion=resource_fusion)
    identifiers = [request.request_id for request in case.workload.requests]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("request_ids_must_be_unique")
    quota = {str(key): max(0, int(value))
             for key, value in dict(token_quotas or {}).items()}
    unknown = sorted(set(quota) - set(identifiers))
    if unknown:
        raise ModelingError("token_quota_unknown_request:" + ",".join(unknown))
    memory = memory or Memory(case, memory_policy)
    parallel = case.runtime.parallel
    groups = [[] for _ in range(parallel.dp)]
    chains = {}
    for index, request in enumerate(sorted(case.workload.requests,
                                          key=lambda r: (r.arrival_ms, r.request_id))):
        replica = chains.setdefault(request.chain_id, index % parallel.dp) if request.chain_id else index % parallel.dp
        groups[replica].append(request)
    schedulers = [scheduler_factory(case, group, memory, replica)
                  for replica, group in enumerate(groups)]
    if window:
        for scheduler in schedulers:
            for request_id, values in dict(window).items():
                state = scheduler.by_id.get(request_id)
                if state is None:
                    continue
                if "computed" in values:
                    state.computed = int(values["computed"])
                if "generated" in values:
                    state.generated = int(values["generated"])
                if "release_ms" in values:
                    state.release_ms = max(state.request.arrival_ms,
                                           float(values["release_ms"]))
    cost = CostModel(case, primitives)
    execution = Execution(
        case, cost, ledger=cost.ledger, recipes=recipes, submission=submission,
        trace_limit=event_limit, include_serving_io=True,
        include_submission=include_submission, memory_policy=memory.policy,
        memory=memory,
    )
    # A bounded summary ring (the default 256 rows) must not silently disable
    # exact physical-work coalescing.  A caller asking for a wider per-event
    # trace is a diagnostic request: keep the uncoalesced decomposition so the
    # published event rows stay one per declared program task.
    execution._compact_step = (not execution.legacy_diagnostics
                               and event_limit <= DEFAULT_EVENT_LIMIT)
    execution.engine._compact_step = execution._compact_step
    execution.serving_io.record_events = event_limit > 0
    engine = execution.engine
    engine.now = min((r.arrival_ms for r in case.workload.requests), default=0)
    run = run_cohort(
        case, execution, schedulers, memory, lower,
        trace_limit=trace_limit, max_model_steps=max_model_steps,
        observer=observer, token_quotas=quota,
        selected_allocations=selected_allocations,
        selected_replica=memory_replica, selected_stage=memory_stage,
    )
    iterations = run["iterations"]
    modes = run["modes"]
    scopes = run["scopes"]
    assembly = run["assembly"]
    scheduler_feedback = run["scheduler_feedback"]
    traces = run["traces"]
    serving_end = run["window_end"]
    states = [state for scheduler in schedulers for state in scheduler.states]
    rows, boundaries = serving_boundaries(
        states, case.slo, host_results=execution.serving_io.enabled,
        client_delivery=execution.serving_io.client_delivery_ms is not None)
    rows.sort(key=lambda row: (row["arrival_ms"], row["request_id"]))
    metrics = summarize(rows, case.slo, now_ms=serving_end)
    occurrences = []
    for prefix, item in list(execution.serving_io.batch_events.items())[:max(0, event_limit)]:
        entry = {"prefix": prefix, "replica": item.get("replica")}
        for key in ("schedule", "host_prepare", "input_ready", "output_ready",
                    "d2h_ready", "host_result", "client_propagation"):
            event = item.get(key)
            entry[key + "_ms"] = None if event is None else event.time
        entry["deliveries_ms"] = {
            request: event.time for request, event in item.get("deliveries", {}).items()}
        occurrences.append(entry)
    stopped = run["stop_reason"] in {"max_model_steps", "token_quota"}
    return {
        "schema": "mindie-sim.predict/2",
        **model_coverage(case.model),
        "status": ("incomplete" if stopped
                   else "modeled" if all(row["status"] == "served" for row in rows)
                   else "infeasible"),
        "execution_status": ("incomplete" if stopped
                             else "completed" if all(row["status"] == "served" for row in rows)
                             else "blocked"),
        "metrics": metrics, "requests": rows, "memory": memory.to_dict(),
        "metric_boundaries": boundaries,
        "throughput_window_boundary": "legacy_scheduler_completion",
        "execution": {**execution.summary(),
                      "runtime_uid": _runtime_identity(case),
                      "iterations": iterations,
                      "iteration_modes": modes, "graph_scopes": scopes,
                      "assembly": assembly, "cohort_stop": run["stop_reason"],
                      "model_steps": run["model_steps"],
                      "logical_steps": run["logical_steps"],
                      "scheduler_feedback": scheduler_feedback,
                      "token_quotas": dict(quota)},
        "parameters": cost.summary(),
        "assumptions": {
            "speculative_acceptance": {key: policy.basis for scheduler in schedulers
                                      for key, policy in scheduler.drafts.items()},
        },
        "traces": traces, "events": list(engine.trace),
        "framework_occurrences": occurrences,
        "operator_selection_summary": run["operator_selection_summary"],
        "plan": {"model": case.model.to_dict(), "runtime": case.runtime.to_dict(),
                 "machine": case.machine.to_dict()},
        "metric_boundary": "device_token_commit",
    }


def simulate_service(*args, **kwargs):
    """Production cohort-loop entry over the shared predict/run_cohort path."""
    result = predict(*args, **kwargs)
    result["production_entry"] = "simulate_service"
    return result


__all__ = ["predict", "simulate_service", "run_cohort"]
