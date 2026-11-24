"""Prefill and decode serving on the shared Engine.

[Role] Orchestrate prefill, destination admission, cache handoff and decode.
[Invariants] Destination capacity is reserved before transfer. Source state stays live
  through completion.
[Timeouts & Locks] Simulation advances finite requests and active transfers without
  external waits.
[Side-Effect Policy] Owns and mutates the per-run serving state.
"""
import math
from collections import defaultdict

from mindie_sim.catalog.cache import layer_cache_components
from mindie_sim.catalog.context_parallel import cp_interleave
from mindie_sim.coverage import combine_coverage
from mindie_sim.schema.common import ModelingError

from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
from ..capacity import Memory
from ..efficiency.cost import CostModel
from ..evaluation import EvaluationContext
from ..execution.dispatch import cohorts, submit_batch
from ..engine import Engine
from ..execution import Execution
from ..frontend import Frontend
from ..efficiency.metrics import serving_boundaries, summarize
from ..pd_connector import PDConnector
from ..scheduler import Scheduler
from ..structure import parse_case
from .handoff import (
    _pd_handoff_binding,
    _pending_release_targets,
    _register_consumer_release,
)

#: Bounded default diagnostic trace sizes for the direct PD entry.
DEFAULT_PD_TRACE_LIMIT = 2
DEFAULT_PD_EVENT_LIMIT = 256


def _endpoint_execution(execution, memory, pool):
    """Attach the endpoint's own T08 ledger to its real T04 tasks.

    T04's per-task entry is ``Engine.submit(memory=...)``: every
    ``Execution.add``/``_start_program`` path forwards ``self.memory`` there,
    so each endpoint's Scheduler and Execution submit to the same T08 ledger
    while sharing one Engine.  ``Engine.bind_memory`` intentionally keeps one
    *default* ledger; constructing a second Execution with a differing default
    would replace that process-wide slot, so PD passes ``memory=None`` at
    construction and binds the endpoint ledger here.  This replaces the old
    name-prefix router and its decode default.
    """
    if memory is None:
        raise ModelingError("pd_endpoint_memory_required:" + pool)
    execution.memory = memory
    return memory


def _memory_binding_report(engine, memory, execution):
    """Expose the real per-endpoint ledgers without a name router or default."""
    return {
        "engine": id(engine),
        "same_engine": all(pool.engine is engine for pool in execution.values()),
        "pools": {name: id(ledger) for name, ledger in memory.items()},
        "executions": {name: id(pool.memory)
                       for name, pool in execution.items()},
        "bound": {name: execution[name].memory is memory[name]
                  for name in memory},
        "name_router": False,
        "default_ledger": None if engine.memory is None else id(engine.memory),
        "budget_versions": {name: int(getattr(ledger, "budget_version", 0))
                            for name, ledger in memory.items()},
        "grant_records": {name: len(getattr(ledger, "grant_records", {}))
                          for name, ledger in memory.items()},
    }


def predict_pd(raw, *, primitives=None, recipes=None, submission=None, memory_policy=None,
               lower=None, trace_limit=DEFAULT_PD_TRACE_LIMIT,
               event_limit=DEFAULT_PD_EVENT_LIMIT, scope="serving", phase=None,
               tokens=None, context=None, sequences=None, published=None,
               primitive_table=None):
    # The direct PD entry enforces the same structural timing gate as predict.
    from ..efficiency.unmodeled_gate import assert_timing_supported
    assert_timing_supported(raw, entry="predict")
    has_geometry = any(v is not None for v in (phase, tokens, context, sequences))
    if scope in {"body", "model"} or (scope == "serving" and has_geometry):
        from ..api import predict
        resolved_scope = "model" if scope == "model" else "body"
        return predict(raw, scope=resolved_scope, phase=phase, tokens=tokens, context=context,
                       sequences=sequences, primitives=primitives, recipes=recipes,
                       lower=lower, event_limit=event_limit, primitive_table=primitive_table)
    if scope != "serving":
        raise ModelingError(f"invalid_prediction_scope:{scope}")
    pools = raw["pools"]
    common = {key: value for key, value in raw.items() if key not in {"pools", "connector"}}
    cases = {name: parse_case({**common, **pools[name], "model": raw["model"], "workload": raw["workload"]})
             for name in ("prefill", "decode")}
    if cases["prefill"].model.is_standalone_draft:
        raise ModelingError("draft_requires_target_model:attach_auxiliary_to_target;catalog_inspection_available")
    identifiers = [r.request_id for r in cases["prefill"].workload.requests]
    if len(identifiers) != len(set(identifiers)):
        raise ModelingError("request_ids_must_be_unique")
    connector = PDConnector(raw["connector"])
    engine = Engine(connector.resources(), trace_limit=event_limit)
    memory = {name: Memory(case, memory_policy) for name, case in cases.items()}
    client_declared = [case.runtime.serving_io.get("client_delivery_ms") is not None
                       for case in cases.values()]
    if any(client_declared) and not all(client_declared):
        raise ModelingError("pd_client_delivery_boundary_requires_both_pool_declarations")
    client_tails = {}
    evals = {
        name: EvaluationContext.create(
            case, recipes, primitives, primitive_table=primitive_table,
        )
        for name, case in cases.items()
    }
    execution = {name: Execution(case, evals[name].cost, ledger=evals[name].cost.ledger,
                                recipes=recipes, submission=submission, engine=engine,
                                namespace=name + ":", include_serving_io=True,
                                client_tails=client_tails,
                                memory_policy=memory[name].policy, memory=None,
                                bindings=evals[name].bindings, evaluation=evals[name])
                 for name, case in cases.items()}
    for name in cases:
        _endpoint_execution(execution[name], memory[name], name)
    lowerers = {name: lower or Frontend(attention_policy=AttentionPolicy(**case.runtime.attention_policy))
                for name, case in cases.items()}
    schedulers, states, placement = {}, [], {}
    for name, case in cases.items():
        groups = [[] for _ in range(case.runtime.parallel.dp)]
        chain_placement = {}
        for index, request in enumerate(case.workload.requests):
            replica = chain_placement.setdefault(request.chain_id, index % len(groups)) if request.chain_id else index % len(groups)
            groups[replica].append(request)
            placement[name, request.request_id] = replica
        schedulers[name] = [Scheduler(case, group, memory[name], replica,
                           eligible=lambda state, pool=name: state.location == pool,
                           handoff_after_prefill=name == "prefill")
                           for replica, group in enumerate(groups)]
    states = [state for scheduler in schedulers["prefill"] for state in scheduler.states]
    shared = {state.request.request_id: state for state in states}
    for scheduler in schedulers["decode"]:
        scheduler.states = [shared[state.request.request_id] for state in scheduler.states]
        scheduler.by_id = {state.request.request_id: state for state in scheduler.states}
        scheduler.successors = {key: shared[value.request.request_id] for key, value in scheduler.successors.items()}
    engine.now = min((r.request.arrival_ms for r in states), default=0)
    serving_end, iterations, handoffs, handoff_bytes, handoff_records, traces = _run_pd_loop(
        cases, engine, connector, memory, execution, lowerers, schedulers,
        states, shared, placement, client_tails, published, trace_limit)
    rows, boundaries = serving_boundaries(
        states, cases["prefill"].slo,
        host_results=any(pool.serving_io.enabled for pool in execution.values()),
        client_delivery=all(client_declared))
    return {
        "schema": "mindie-sim.predict/2", "deployment": "disaggregated",
        **combine_coverage(case.model for case in cases.values()),
        "status": "modeled" if all(state.finish_ms is not None for state in states) else "infeasible",
        "metrics": summarize(rows, cases["prefill"].slo, now_ms=serving_end),
        "metric_boundaries": boundaries,
        "throughput_window_boundary": "legacy_scheduler_completion",
        "requests": rows, "memory": {name: ledger.to_dict() for name, ledger in memory.items()},
        "execution": {**engine.summary(), "iterations": iterations,
                      "handoffs": handoffs, "handoff_bytes": handoff_bytes,
                      "connector": connector.summary(),
                      "memory_binding": _memory_binding_report(engine, memory, execution),
                      "pd_handoffs": [
                          {key: value for key, value in record.items()
                           if key != "transport_event"}
                          | {"complete_ms": record["transport_event"].time}
                          for record in handoff_records],
                      "submission": {name: pool.submission_summary()
                                     for name, pool in execution.items()},
                      "serving_io": {name: pool.serving_io.summary()
                                     for name, pool in execution.items()},
                      "step_summary": {name: pool.step_summary.to_dict()
                                       for name, pool in execution.items()},
                      "transfer_layout": "destination_shards"},
        "traces": traces, "events": list(engine.trace), "metric_boundary": "device_token_commit",
    }


def _run_pd_loop(cases, engine, connector, memory, execution, lowerers,
                 schedulers, states, shared, placement, client_tails,
                 published, trace_limit):
    flights, transfers, traces = {}, {}, []
    # Real event handles for the three distinct PD boundaries.  They are not
    # fabricated sleeps: source is the prefill device-complete event, grant is
    # the decode admission event, and release is resolved by the first actual
    # decode batch that consumes the handoff.
    source_ready, consumer_release = {}, {}
    consumer_callbacks = {}
    source_release_state = {}
    consumer_registered = set()
    handoff_records = []
    handoff_bytes, handoffs, iterations = 0, 0, 0

    def source_release_callback(source_rank, source_released, prefill_replica):
        def release_source(rank=source_rank, done=source_released,
                           replica=prefill_replica):
            if done[0]:
                return
            done[0] = True
            memory["prefill"].release(replica, request)
        return release_source

    while not all(state.done for state in states):
        # Admit to the destination before making any external cache service
        # ready. Waiting transfers retain source ownership and exert backpressure.
        for state in states:
            if state.done or state.location != "waiting_transfer":
                continue
            request = state.request.request_id
            volume = _admit_pd_handoff(
                state, request, cases, engine, connector, memory, execution,
                schedulers, placement, published, source_ready, consumer_release,
                consumer_callbacks, source_release_state, source_release_callback,
                transfers, handoff_records)
            if volume is not None:
                handoff_bytes += volume
                handoffs += 1
        for name, case in cases.items():
            for group in cohorts(case.runtime.parallel):
                if any((name, replica) in flights for replica in group):
                    continue
                pending_release = any(state.location in {"waiting_transfer", "transferring"}
                                      and not state.done for state in states)
                plans = {r: schedulers[name][r].schedule(
                    engine.now, defer_capacity_failure=name == "prefill" and pending_release)
                         if not schedulers[name][r].done else None
                         for r in group}
                if any(plans.values()):
                    steps = submit_batch(case, schedulers[name], plans, execution[name], lowerers[name], engine.now)
                    flights.update({(name, replica): step for replica, step in steps.items()})
                    if name == "decode":
                        for request, step in ((request, steps[replica])
                                              for replica in steps
                                              for request in steps[replica].commits):
                            pending = {
                                rank: release
                                for rank, release in (consumer_release.get(request) or {}).items()
                                if not release.done
                                and (request, rank) not in consumer_registered}
                            consumer_registered.update(
                                (request, rank) for rank in pending)
                            _register_consumer_release(
                                engine, request, pending, step.device_complete,
                                on_consumer=lambda rank, request=request,
                                registry=consumer_callbacks:
                                    registry.get((request, rank)))
                    iterations += len(steps)
                else:
                    _submit_pd_retirements(
                        name, group, schedulers, memory, execution,
                        engine, flights, iterations)
        arrivals = [time for name, group in schedulers.items() for replica, scheduler in enumerate(group)
                    if (name, replica) not in flights if (time := scheduler.next_arrival(engine.now)) is not None]
        horizon = min(arrivals, default=math.inf)
        targets = (tuple(s.completion for s in flights.values())
                   + tuple(transfers.values())
                   + tuple(client_tails.values())
                   + _pending_release_targets(memory))
        if not targets:
            if math.isfinite(horizon):
                engine.advance(horizon=horizon)
                engine.now = horizon
                continue
            for state in states:
                if not state.done:
                    state.reason = "decode_destination_capacity"
                    for name in cases:
                        memory[name].release(placement[name, state.request.request_id], state.request.request_id)
            break
        engine.advance(until=targets, horizon=horizon)
        for key, step in list(flights.items()):
            if not step.completion.done:
                continue
            name, replica = key
            scheduler = schedulers[name][replica]
            scheduler.commit(step.completion.time, {key: event.time for key, event in step.commits.items()})
            if name == "prefill":
                for request in step.commits:
                    source_ready.setdefault(request, step.device_complete)
                for state in scheduler.states:
                    if not state.done and state.first_token_ms is not None and state.location == "prefill":
                        state.location = "waiting_transfer"
            if step.mode != "retire" and len(traces) < trace_limit:
                traces.append({"pool": name, **step.to_dict()})
            del flights[key]
        for request, event in list(transfers.items()):
            if event.done:
                # The actual-rank path releases the source exactly at its
                # per-rank last-use callback; the pool-endpoint path keeps its
                # existing whole-handoff retirement.
                if source_release_state.get(request) is None:
                    memory["prefill"].release(
                        placement["prefill", request], request)
                shared[request].location = "decode"
                del transfers[request]
    # A request that never reaches decode must not leave a transport credit
    # allocation or consumer-release task unresolved.  This explicit release
    # is recorded at the actual end of the serving loop, not as elapsed sleep.
    final_releases = []
    for request, releases in list(consumer_release.items()):
        _schedule_pd_final_releases(
            engine, memory, placement, request, releases, consumer_callbacks,
            source_release_state, final_releases)
    serving_end = _drain_pd_tails(
        engine, memory, consumer_release, transfers, client_tails, final_releases)
    return serving_end, iterations, handoffs, handoff_bytes, handoff_records, traces


def _admit_pd_handoff(
    state, request, cases, engine, connector, memory, execution, schedulers,
    placement, published, source_ready, consumer_release, consumer_callbacks,
    source_release_state, source_release_callback, transfers, handoff_records,
):
    target = placement["decode", request]
    active = sum(s.location in {"decode", "transferring"} and not s.done
                 for s in schedulers["decode"][target].states)
    if active >= cases["decode"].runtime.max_num_seqs:
        return None
    if not memory["decode"].resize(target, request, state.computed + 1):
        return None
    source_event = source_ready.get(request)
    if source_event is None:
        raise ModelingError(
            "pd_handoff_missing_source_ready:" + request)
    del (
        connector, engine, execution, published, consumer_release,
        consumer_callbacks, source_release_state, source_release_callback,
        transfers, handoff_records,
    )
    raise NotImplementedError(
        "LEGACY_PURGED: deleted duplicate target-stage volume loop omitting dcp_dp_width "
        "and two-endpoint overlapping-rank fallback; consume canonical integer PDHandoff "
        "plan and per-rank PDTransfer graph directly from disaggregated.handoff (S6-W5-14)"
    )


def _submit_rank_handoff(
    engine, connector, handoff, request, source_ranks, destination_ranks,
    source_event, target, state, memory, placement, consumer_release,
    consumer_callbacks, source_release_state, source_release_callback, storage_holds,
):
    grant_by_rank = {
        rank: engine.submit(
            f"pd:{request}:destination_grant:r{rank}",
            owner=f"decode:r{target}",
            metadata={"boundary": "pd_destination_admission",
                      "request_id": request, "rank": rank,
                      "receive_buffer_id": dict(
                          handoff.receive_buffer_ids)[rank],
                      "tokens": state.computed + 1})
        for rank in destination_ranks}
    releases = {
        rank: (engine.future(
            f"pd:{request}:consumer_release:r{rank}"),)
        for rank in destination_ranks}
    consumer_release[request] = {
        rank: events[0] for rank, events in releases.items()}
    destination_ranges = defaultdict(list)
    for item in handoff.hbm_transfers():
        if item.destination_byte_range:
            destination_ranges[item.destination_rank].append(
                item.destination_byte_range)
    release_callbacks = {}
    for rank, events in releases.items():
        buffer_id = dict(handoff.receive_buffer_ids)[rank]
        consumer_name = f"pd:{request}:decode:r{rank}"
        generation = memory["decode"].request_versions.get(
            (target, request), 0)
        hold_event = events[0]
        rank_ranges = tuple(destination_ranges.get(rank) or ())
        if not rank_ranges:
            raise ModelingError(f"pd_handoff_destination_ranges_missing:{request}:r{rank}")
        held = memory["decode"].hold_for_async_consumer(
            target, request, consumer_name,
            storage=buffer_id,
            ranges=rank_ranges,
            generation=generation, event=hold_event)
        if not held:
            raise ModelingError(
                f"pd_decode_memory_hold_rejected:{request}:r{rank}")
        storage_holds[rank] = buffer_id
        keep = (memory["decode"], target, request, consumer_name,
                generation, hold_event)
        def release_hold(keep=keep):
            pool, replica, req, consumer, gen, event = keep
            return pool.release_async_consumer(
                replica, req, consumer,
                generation=gen, event=event)
        # Fixed (pool, replica, request, rank, generation, event)
        # identity; never re-derived from a display name later.
        consumer_callbacks[request, rank] = release_hold
        release_callbacks[rank] = release_hold
    source_released = [False]
    source_release_state[request] = source_released
    source_callbacks = {}
    prefill_replica = placement["prefill", request]
    for source_rank in source_ranks:
        source_callbacks[source_rank] = source_release_callback(
            source_rank, source_released, prefill_replica)
    return connector.submit_handoff(
        engine, handoff,
        source_ready_by_rank={rank: source_event
                              for rank in source_ranks},
        destination_ready_by_rank=grant_by_rank,
        release_events=releases,
        source_release_callbacks=source_callbacks,
        release_callbacks_by_rank=release_callbacks,
        name="pd:" + request)


def _submit_pd_retirements(name, group, schedulers, memory, execution,
                           engine, flights, iterations):
    for r in group:
        if not schedulers[name][r].done and memory[name].has_delayed_releases(r):
            pool_exec = execution[name]
            prep = pool_exec.prepared.pop(r, None)
            sched_ms = pool_exec.submission.scheduler_ms if pool_exec.include_submission else 0.0
            comp_ms = pool_exec.submission.completion_ms if pool_exec.include_submission else 0.0
            sched_ev = engine.submit(
                f"{pool_exec.namespace}r{r}:retire_schedule:{iterations}",
                {f"{pool_exec.namespace}scheduler@r{r}": sched_ms},
                after=(prep,) if prep is not None else (),
                release=engine.now, owner=f"replica:{r}",
            )
            comp_ev = engine.submit(
                f"{pool_exec.namespace}r{r}:retire_complete:{iterations}",
                after=(sched_ev,), latency=comp_ms, release=engine.now, owner=f"replica:{r}",
            )
            from ..execution import ScheduledIteration
            flights[name, r] = ScheduledIteration(
                r, schedulers[name][r].iteration, comp_ev, engine.now,
                "retire", 0, 0, None, "none", 0, {}, comp_ev, {}, {}, (),
            )


def _schedule_pd_final_releases(
    engine, memory, placement, request, releases, consumer_callbacks,
    source_release_state, final_releases,
):
    for rank, release in releases.items():
        if release.done:
            continue
        suffix = ("" if len(releases) == 1 and rank == 1
                  else f":r{rank}")
        cancel = consumer_callbacks.get((request, rank))
        source_done = source_release_state.get(request)
        def finish(_event, target=release, cancel=cancel,
                   request=request, source_state=source_done):
            if not target.done:
                engine.resolve(target)
            if cancel is not None:
                cancel()
            if source_state is not None and not source_state[0]:
                source_state[0] = True
                memory["prefill"].release(
                    placement["prefill", request], request)
        final_releases.append(engine.submit(
            f"pd:{request}:final_release{suffix}",
            metadata={"boundary": "pd_unconsumed_final_release",
                      "request_id": request, "rank": rank},
            on_complete=finish))


def _drain_pd_tails(engine, memory, consumer_release, transfers, client_tails,
                    final_releases):
    serving_end = engine.now
    # Complete only this service's causal tail.  An unrelated long timer is
    # not a target and stays pending; same-time service descendants (receive
    # release decisions, consumer cancellation) settle at serving_end.
    device_tail = (tuple(final_releases)
                   + tuple(consumer_release.get(request, {}).get(rank, None)
                           for request in consumer_release
                           for rank in consumer_release[request])
                   + tuple(transfers.values()))
    device_tail = tuple(event for event in device_tail if event is not None)
    if device_tail:
        engine.advance(until=device_tail, first=False, horizon=serving_end)
    engine.advance(horizon=serving_end)
    delivery_tail = tuple(event for event in client_tails.values() if event is not None)
    if delivery_tail:
        engine.advance(until=delivery_tail, first=False)
        engine.advance(horizon=engine.now)
    for ledger in memory.values():
        ledger.flush_delayed_releases()
    return serving_end
