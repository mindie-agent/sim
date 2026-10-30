"""Prefill/decode pools on the same causal engine.

[Role] Schedule prefill, destination admission, cache handoff and decode.
[Invariants] Destination KV is reserved before transfer. The source retains
  state until the transfer completes. TTFT and token intervals include actual
  arrival, queue and handoff times. Transfer bytes are physical destination
  shards, independent of overlap. No post-hoc latency addition.
[Timeouts & Locks] Finite requests and active transfers only; no external work.
[Side-Effect Policy] Mutates only per-run state.
"""
import math

from .attention import AttentionPolicy
from .capacity import Memory
from .cost import CostModel
from .dispatch import cohorts, submit_batch
from .engine import Engine
from .execution import Execution
from .frontend import Frontend
from .metrics import serving_boundaries, summarize
from .pd_connector import PDConnector
from .scheduler import Scheduler
from .structure import parse_case
from mindie_sim.schema.common import ModelingError
from mindie_sim.coverage import combine_coverage


def predict_pd(raw, *, primitives=None, recipes=None, submission=None, memory_policy=None,
               lower=None, trace_limit=2, event_limit=256, scope="serving", phase=None,
               tokens=None, context=None, sequences=None):
    if scope == "body":
        from .api import predict
        return predict(raw, scope=scope, phase=phase, tokens=tokens, context=context,
                       sequences=sequences, primitives=primitives, recipes=recipes,
                       lower=lower, event_limit=event_limit)
    if scope != "serving" or any(v is not None for v in (phase, tokens, context, sequences)):
        raise ModelingError("prediction_geometry_requires_body_scope")
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
    execution = {name: Execution(case, CostModel(case, primitives), recipes=recipes,
                                submission=submission, engine=engine, namespace=name + ":",
                                include_serving_io=True, client_tails=client_tails)
                 for name, case in cases.items()}
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
    flights, transfers, traces = {}, {}, []
    handoff_bytes, handoffs, iterations = 0, 0, 0
    while not all(state.done for state in states):
        # Admit to the destination before making any external cache service
        # ready. Waiting transfers retain source ownership and exert backpressure.
        for state in states:
            if state.done or state.location != "waiting_transfer":
                continue
            request = state.request.request_id
            target = placement["decode", request]
            active = sum(s.location in {"decode", "transferring"} and not s.done
                         for s in schedulers["decode"][target].states)
            if active >= cases["decode"].runtime.max_num_seqs:
                continue
            if not memory["decode"].resize(target, request, state.computed + 1):
                continue
            p = cases["decode"].runtime.parallel
            volume = sum(memory["decode"].cache_bytes(state.computed, padded=False)) * p.tp * p.pcp
            demands = {}
            for pool in ("prefill", "decode"):
                replica = placement[pool, request]
                parallel = cases[pool].runtime.parallel
                local = memory[pool].cache_bytes(state.computed, padded=False)
                for stage, stage_bytes in enumerate(local):
                    roles = execution[pool].stage_names[replica, stage]
                    demands[roles["hbm"]] = stage_bytes
            transfers[request] = connector.submit(engine, "pd:" + request, volume, demands)
            state.location = "transferring"
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
                    iterations += len(steps)
        arrivals = [time for name, group in schedulers.items() for replica, scheduler in enumerate(group)
                    if (name, replica) not in flights if (time := scheduler.next_arrival(engine.now)) is not None]
        horizon = min(arrivals, default=math.inf)
        targets = (tuple(s.completion for s in flights.values()) + tuple(transfers.values())
                   + tuple(client_tails.values()))
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
                for state in scheduler.states:
                    if not state.done and state.first_token_ms is not None and state.location == "prefill":
                        state.location = "waiting_transfer"
            if len(traces) < trace_limit:
                traces.append({"pool": name, **step.to_dict()})
            del flights[key]
        for request, event in list(transfers.items()):
            if event.done:
                memory["prefill"].release(placement["prefill", request], request)
                shared[request].location = "decode"
                del transfers[request]
    serving_end = engine.now
    engine.advance()
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
                      "submission": {name: pool.submission_summary()
                                     for name, pool in execution.items()},
                      "serving_io": {name: pool.serving_io.summary()
                                     for name, pool in execution.items()},
                      "step_summary": {name: pool.step_summary.to_dict()
                                       for name, pool in execution.items()},
                      "transfer_layout": "destination_shards"},
        "traces": traces, "events": list(engine.trace), "metric_boundary": "device_token_commit",
    }
