"""Service, model, and framework fragment assembly driver for ``compose``.

[Role]
  Submit model and no-layer serving framework fragments into a single ``Execution``
  context (or delegate production ``scope="service"`` runs to the canonical cohort
  driver) with exact host-launch, commit, and client-delivery event boundaries.

[Invariants]
  * ``scope="model"`` executes pure forward programs without sampling/commit.
  * ``scope="framework"`` executes only declared Serving I/O with zero model ops.
  * Ordinary decode steps advance one token per active request; speculative steps
    follow ``DraftPolicy`` acceptance credits.

[Timeouts & Locks]
  Bounded discrete-event execution over declared iteration/token windows; no I/O.

[Side-Effect Policy]
  Pure in-memory simulation returning JSON-serializable ``mindie-sim.assembly/1``.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.common import ModelingError

DRAFT_PREPARE_MIN_REMAINING = 2
MAX_OBSERVED_STEPS = 256
MS_PER_SECOND = 1000.0


def _assembly_request_specs(case, requests):
    from mindie_sim.schema.model import RequestSpec
    if requests is None:
        return tuple(case.workload.requests)
    return tuple(item if isinstance(item, RequestSpec) else RequestSpec.from_mapping(item, idx)
                 for idx, item in enumerate(requests))


def _assembly_emit_counts(specs, generated, override):
    if override is not None:
        return {str(k): int(v) for k, v in override.items() if int(v) > 0}
    return {s.request_id: 1 for s in specs if s.output_tokens - generated[s.request_id] > 0}


def _assembly_metric_rows(specs, rows):
    records = {s.request_id: {"commits": [], "host": [], "deliv": [], "gen": 0} for s in specs}
    for row in rows:
        for req, count in row["emitted_tokens"].items():
            rec = records[req]
            rec["gen"] += count
            if row["commit_ms"].get(req) is not None:
                rec["commits"].append((row["commit_ms"][req], count))
        for req, at_ms in row["host_result_ms"].items():
            records[req]["host"].append(at_ms)
        for req, at_ms in row["delivery_ms"].items():
            records[req]["deliv"].append(at_ms)
    result = []
    for spec in specs:
        rec = records[spec.request_id]
        commits, host, delivered = sorted(rec["commits"]), sorted(rec["host"]), sorted(rec["deliv"])
        first_c, last_c = (commits[0][0] if commits else None), (commits[-1][0] if commits else None)
        gen, intervals = rec["gen"], max(0, rec["gen"] - 1)
        tpot = (last_c - first_c) / intervals if intervals and first_c is not None else None
        window = last_c - spec.arrival_ms if last_c is not None else None
        result.append({
            "request_id": spec.request_id, "arrival_ms": spec.arrival_ms, "generated_tokens": gen,
            "first_commit_ms": first_c, "last_commit_ms": last_c,
            "ttft_ms": first_c - spec.arrival_ms if first_c is not None else None, "tpot_ms": tpot, "e2e_ms": window,
            "output_tps": gen * MS_PER_SECOND / window if window and window > 0 else None,
            "first_host_result_ms": host[0] if host else None, "last_host_result_ms": host[-1] if host else None,
            "first_delivery_ms": delivered[0] if delivered else None, "last_delivery_ms": delivered[-1] if delivered else None,
            "delivery_e2e_ms": delivered[-1] - spec.arrival_ms if delivered else None,
        })
    return result


def _assemble_cohort_service(
    parsed, specs, *, scope, iterations, tokens, phase, context, explicit_window,
    full_service, include_submission, trace_limit, resource_fusion, primitives, recipes, lower,
) -> dict[str, Any]:
    from mindie_sim.prediction.api import simulate_service
    quota = int(iterations or 1) * int(tokens or 1) if (explicit_window and phase == "decode") else None
    quotas = {s.request_id: quota for s in specs} if quota is not None else None
    window_case = replace(parsed, workload=replace(parsed.workload, requests=tuple(specs)))
    window = ({s.request_id: {"computed": int(context) if context is not None else s.prompt_tokens, "generated": 0}
               for s in specs} if phase == "decode" else None)
    steps = None if full_service or not explicit_window or phase == "decode" else int(iterations or 1)
    observed_steps, observed_total = [], [0]

    def observe(step):
        observed_total[0] += 1
        if len(observed_steps) < MAX_OBSERVED_STEPS:
            observed_steps.append(step.to_dict())

    result = simulate_service(
        window_case, primitives=primitives, recipes=recipes, trace_limit=trace_limit,
        max_model_steps=steps, window=window, token_quotas=quotas,
        include_submission=scope != "model" if include_submission is None else bool(include_submission),
        resource_fusion=resource_fusion, lower=lower, observer=observe)
    original_outputs = {s.request_id: s.output_tokens for s in specs}
    unfinished = [rid for rid, out_tok in original_outputs.items()
                  if next((r["generated_tokens"] for r in result["requests"] if r["request_id"] == rid), 0) < out_tok]
    result.update({
        "steps": observed_steps, "steps_total": observed_total[0],
        "assemble_backend": "cohort_driver", "assembly_scope": scope,
        "assembly": result.get("execution", {}).get("assembly"),
        "window": {
            "iterations": int(iterations) if iterations is not None else None,
            "tokens": int(tokens) if tokens is not None else None,
            "phase": phase, "context": context, "request_ids": [s.request_id for s in specs],
            "original_output_tokens": original_outputs, "per_request_token_quota": quota,
            "token_quota_by_request": dict(quotas or {}), "unfinished_request_ids": unfinished,
            "max_model_steps": steps, "cohort_stop": result.get("execution", {}).get("cohort_stop"),
        },
    })
    if unfinished:
        result["status"] = result["execution_status"] = "incomplete"
    return result


def _host_event_dict(item):
    replica, stage, ev = item
    return {"replica": replica, "stage": stage, "event": ev.name, "sequence": ev.sequence, "submit_ms": ev.time}


def _io_time(io: dict[str, Any], key: str):
    ev = io.get(key)
    return ev.time if ev is not None and ev.time is not None else None


def _assembly_request_views(
    parsed, specs, generated, computed_kv, model_draft_policies, model_draft_states,
    model_token_quota, *, scope, phase, context, step_index,
):
    from mindie_sim.schema.workload import RequestView

    views = []
    model_budget = int(parsed.runtime.max_num_batched_tokens) if model_draft_policies else 0
    for spec in specs:
        target_tok = min(spec.output_tokens, model_token_quota) if model_token_quota is not None else spec.output_tokens
        remaining = target_tok - generated[spec.request_id]
        if remaining <= 0:
            continue
        proposal, prepare, advance = 0, False, 1
        if phase == "prefill" and step_index == 0:
            q_len, view_phase, count = spec.prompt_tokens, "prefill", 1
        else:
            q_len, view_phase, count = 1, "decode", 1
            if model_draft_policies:
                if model_budget <= 0:
                    continue
                state, policy = model_draft_states[spec.request_id], model_draft_policies.get(spec.request_id)
                if policy is not None:
                    proposal = min(policy.proposal, max(0, remaining - 1), max(0, model_budget - 1))
                    advance = min(remaining, policy.committed(state, proposal) if proposal else 1)
                    prepare = bool(policy.mode) and remaining > DRAFT_PREPARE_MIN_REMAINING
                model_budget -= 1 + proposal
        view_context = (0 if view_phase == "prefill" else
                        (int(context) if context is not None else spec.prompt_tokens + generated[spec.request_id])
                        if scope == "framework" else computed_kv[spec.request_id])
        views.append(RequestView(
            spec.request_id, q_len, view_context, True, view_phase,
            draft_verify=bool(proposal), draft_proposal_tokens=proposal, draft_prepare=prepare,
            draft_advanced_tokens=advance if model_draft_policies else count,
            draft_accepted_tokens=max(0, advance - 1) if model_draft_policies else 0,
            prompt_tokens=spec.prompt_tokens))
    return views


def _record_diagnostic_step(
    execution, serving, plan, program, step, views, emit, active_reqs, before_sub,
    computed_kv, generated, model_draft_states, *, scope, phase, step_index,
    decode_tokens, release_ms, include_submission,
):
    """Advance request state and record the completed diagnostic step."""
    io = serving.pop_batch_events(f"{execution.namespace}r0:i{plan.iteration}") or {}
    launches = [(r, s, ev) for r, s, ev in step.host_submits if ev.time is not None]
    htimes = [ev.time for _, _, ev in launches]
    sub_ms, sub_last = (min(htimes) if htimes else None), (max(htimes) if htimes else None)
    hfirst = min(launches, key=lambda x: (x[2].time, x[2].sequence), default=None)
    hlast = max(launches, key=lambda x: (x[2].time, x[2].sequence), default=None)
    sub_basis = ("actual_step_host_submissions" if htimes else
                 ("submission_not_included" if not include_submission else "no_engine_launch"))
    decl_t = {r: e.time for r, e in step.declared_commits.items() if e.time is not None}
    fall_t = {r: e.time for r, e in step.fallback_commits.items() if e.time is not None}
    merg_t = {r: e.time for r, e in step.commits.items() if e.time is not None}
    host_t = {r: e.time for r, e in step.host_results.items() if e.time is not None} if serving.enabled else {}
    deliv_t = {r: e.time for r, e in step.client_deliveries.items() if e.time is not None} if serving.enabled else {}
    commit_at = step.completion.time if step.completion.time is not None else release_ms
    for view in views:
        if scope == "framework":
            continue
        if view.phase == "prefill":
            computed_kv[view.request_id] = view.prompt_tokens
        elif view.request_id in model_draft_states:
            st = model_draft_states[view.request_id]
            st.commit(view, commit_at)
            computed_kv[view.request_id], generated[view.request_id] = st.computed, st.generated
        else:
            computed_kv[view.request_id] += view.q_len
    return {
        "iteration": step_index,
        "outer_iteration": step_index // decode_tokens if phase == "decode" else step_index,
        "decode_step_in_iteration": step_index % decode_tokens if phase == "decode" else 0,
        "mode": step.mode, "valid_tokens": step.valid_tokens, "padded_tokens": step.padded_tokens,
        "graph_bucket": step.graph_bucket, "replay_scope": step.replay_scope, "release_ms": release_ms,
        "submitted_ms": sub_ms, "submitted_first_ms": sub_ms, "submitted_last_ms": sub_last,
        "submitted_basis": sub_basis, "submitted_scope": "actual_step_host_submissions" if htimes else "none",
        "host_submit_first": _host_event_dict(hfirst) if hfirst else None,
        "host_submit_last": _host_event_dict(hlast) if hlast else None,
        "host_submit_event_count": len(launches), "input_ready_ms": _io_time(io, "input_ready"),
        "host_prepare_done_ms": _io_time(io, "host_prepare"), "device_done_ms": step.device_complete.time,
        "step_ready_ms": step.completion.time, "commit_ms": decl_t, "declared_commit_ms": decl_t,
        "fallback_commit_ms": fall_t, "merged_commit_ms": merg_t, "host_result_ms": host_t,
        "host_complete_ms": _io_time(io, "host_result"), "client_propagation_ms": _io_time(io, "client_propagation"),
        "delivery_ms": deliv_t, "emitted_tokens": dict(emit or {}), "active_requests": sorted(active_reqs),
        "context_tokens": {v.request_id: v.context for v in views},
        "computed_kv_tokens_after": ({v.request_id: computed_kv[v.request_id] for v in views}
                                     if scope != "framework" else None),
        "output_ready_ms": _io_time(io, "output_ready"), "input_bytes": io.get("input_bytes", 0.0),
        "d2h_bytes": io.get("d2h_bytes", 0.0), "declared_commit": bool(step.declared_commits),
        "program_assembly": program.assembly if program is not None else "framework",
        "graph_replay_submissions": execution.submission_counts["graph_replays"] - before_sub["graph_replays"],
        "eager_submissions": execution.submission_counts["eager_invocations"] - before_sub["eager_invocations"],
    }


def assemble(
    case, *, scope="service", iterations=None, phase=None, tokens=None, context=None,
    requests=None, emitted_tokens=None, include_submission=None, trace_limit=0,
    resource_fusion=True, primitives=None, recipes=None, full_service=False,
    diagnostic_window=False, lower=None,
):
    """Submit model and framework fragments into one execution context."""
    from mindie_sim.prediction.operators.attention.attention import AttentionPolicy
    from mindie_sim.prediction.efficiency.cost import CostModel
    from mindie_sim.prediction.execution import Execution
    from mindie_sim.prediction.frontend import Frontend, SERVICE_FAMILIES
    from mindie_sim.schema.operators import IterationWork
    from mindie_sim.schema.workload import IterationPlan, selected_mode

    if scope not in {"model", "service", "framework"}:
        raise ModelingError(f"assembly_scope_unknown:{scope}")
    explicit_window = iterations is not None or tokens is not None
    if phase is None:
        phase = "prefill" if scope == "service" and not explicit_window else "decode"
    if phase not in {"decode", "prefill"}:
        raise ModelingError(f"assembly_phase_unknown:{phase}")
    if iterations is not None and int(iterations) < 1:
        raise ModelingError("assembly_requires_positive_iterations")
    if tokens is not None and int(tokens) < 1:
        raise ModelingError("assembly_requires_positive_tokens")
    if context is not None and int(context) < 0:
        raise ModelingError("assembly_context_requires_nonnegative_tokens")
    parsed = parse_case(case)
    specs = _assembly_request_specs(parsed, requests)
    if scope == "service" and not diagnostic_window:
        return _assemble_cohort_service(
            parsed, specs, scope=scope, iterations=iterations, tokens=tokens, phase=phase,
            context=context, explicit_window=explicit_window, full_service=full_service,
            include_submission=include_submission, trace_limit=trace_limit,
            resource_fusion=resource_fusion, primitives=primitives, recipes=recipes, lower=lower)
    iterations = 1 if iterations is None else int(iterations)
    tokens = 1 if tokens is None else int(tokens)
    if parsed.runtime.parallel.dp != 1:
        raise ModelingError("diagnostic_assembly_window_requires_one_data_replica")
    if not specs:
        raise ModelingError("assembly_requires_at_least_one_request")
    include_submission = scope != "model" if include_submission is None else bool(include_submission)
    execution = Execution(parsed, CostModel(parsed, primitives), recipes=recipes,
                          include_submission=include_submission, trace_limit=trace_limit,
                          include_serving_io=scope != "model")
    frontend = Frontend(attention_policy=AttentionPolicy(**parsed.runtime.attention_policy))
    if scope == "framework" and not execution.serving_io.enabled:
        raise ModelingError("framework_assembly_requires_declared_serving_io")
    execution.serving_io.record_events = True
    generated = {s.request_id: 0 for s in specs}
    computed_kv = {s.request_id: int(context) if context is not None else s.prompt_tokens for s in specs}
    release_ms = max(s.arrival_ms for s in specs)
    rows, program_count, first_program = [], 0, None
    service_counts = {family: 0 for family in SERVICE_FAMILIES}
    serving = execution.serving_io
    model_draft_policies, model_draft_states, model_token_quota = {}, {}, None
    if scope == "model" and phase == "decode":
        from mindie_sim.prediction.scheduler import DraftPolicy, Progress
        for spec in specs:
            policy = DraftPolicy.for_request(parsed, spec)
            if policy.mode:
                model_draft_policies[spec.request_id] = policy
        if model_draft_policies:
            model_token_quota = int(iterations) * int(tokens)
            for spec in specs:
                state = Progress(spec)
                state.computed, state.started = computed_kv[spec.request_id], True
                model_draft_states[spec.request_id] = state
    decode_tokens = int(tokens) if phase == "decode" else 1
    total_steps = int(iterations) * decode_tokens if phase == "decode" else int(iterations)
    if model_draft_policies:
        decode_tokens = 1
        total_steps = sum(min(s.output_tokens, int(model_token_quota)) for s in specs)
    for step_index in range(total_steps):
        views = _assembly_request_views(
            parsed, specs, generated, computed_kv, model_draft_policies, model_draft_states,
            model_token_quota, scope=scope, phase=phase, context=context, step_index=step_index)
        if not views:
            break
        plan = IterationPlan(tuple(views), iteration=step_index,
                             output_tokens=tuple((s.request_id, s.output_tokens) for s in specs))
        if scope == "framework":
            q_sum = sum(v.logical_q_len for v in views)
            work = IterationWork((), plan, parsed.runtime.parallel.pp, q_sum, q_sum,
                                 None, selected_mode(plan), "framework_only", replay_scope="none")
            program = None
        else:
            program = frontend.program(plan, parsed.model, parsed.runtime, parsed.machine,
                                       assembly="model" if scope == "model" else "service",
                                       resource_fusion=resource_fusion)
            work, program_count = program.work, program_count + 1
            first_program = first_program or program
            for family in SERVICE_FAMILIES:
                service_counts[family] += sum(op.family == family for op in work.ops)
        override = ({} if emitted_tokens is None else emitted_tokens) if scope == "framework" else None
        before_sub = dict(execution.submission_counts)
        emit = _assembly_emit_counts(specs, generated, override) if scope != "model" else None
        active_reqs = {v.request_id for v in views}
        if emit is not None:
            emit = {r: c for r, c in emit.items() if r in generated and r in active_reqs}
        step = execution.add(work, 0, release_ms, emitted_tokens=emit)
        execution.engine.advance(until=(step.completion, *step.client_deliveries.values()), first=False)
        row = _record_diagnostic_step(
            execution, serving, plan, program, step, views, emit, active_reqs, before_sub,
            computed_kv, generated, model_draft_states, scope=scope, phase=phase,
            step_index=step_index, decode_tokens=decode_tokens, release_ms=release_ms,
            include_submission=include_submission)
        rows.append(row)
        for req, count in (emit or {}).items():
            if req in generated:
                generated[req] += count
        release_ms = max(e for e in (step.completion.time, *row["delivery_ms"].values()) if e is not None)
    execution.engine.advance()
    return _finalize_assembly_report(
        parsed, execution, serving, rows, specs, first_program, program_count, service_counts,
        scope=scope, phase=phase, include_submission=include_submission,
        iterations=iterations, decode_tokens=decode_tokens, total_steps=total_steps)


def _finalize_assembly_report(
    parsed, execution, serving, rows, specs, first_program, program_count, service_counts,
    *, scope, phase, include_submission, iterations, decode_tokens, total_steps,
) -> dict[str, Any]:
    from mindie_sim.prediction.frontend import SERVICE_FAMILIES

    metrics = _assembly_metric_rows(specs, rows)
    fragments = serving.fragments(
        h2d_hbm=execution.stage_names.get((0, 0), {}).get("hbm"),
        d2h_hbm=execution.stage_names.get((0, parsed.runtime.parallel.pp - 1), {}).get("hbm"))
    has_h2d, has_d2h = serving.transfers.get("h2d") is not None, serving.transfers.get("d2h") is not None
    occurrence = {
        "host_prepare": sum(r["host_prepare_done_ms"] is not None for r in rows),
        "input_transfer": len(rows) if has_h2d else 0,
        "output_ready": sum(r["output_ready_ms"] is not None for r in rows),
        "output_transfer": sum(bool(r["emitted_tokens"]) for r in rows) if has_d2h else 0,
        "host_result": sum(r["host_complete_ms"] is not None for r in rows),
        "client_delivery": sum(r["client_propagation_ms"] is not None for r in rows),
        "client_order": sum(len(r["delivery_ms"]) for r in rows),
    }
    framework = [{**f.to_dict(), "occurrences": occurrence.get(f.name, 0)} for f in fragments]
    ops = first_program.work.ops if first_program is not None else ()
    model_fragment = ({
        **first_program.to_dict(include_ops=False), "op_count": len(ops),
        "families": sorted({op.family for op in ops}),
        "service_families_present": sorted({op.family for op in ops} & SERVICE_FAMILIES),
        "iterations": program_count,
    } if first_program is not None else None)
    total_tokens = sum(r["generated_tokens"] for r in metrics if r["generated_tokens"])
    decl_ev, fall_ev = sum(len(r["declared_commit_ms"]) for r in rows), sum(len(r["fallback_commit_ms"]) for r in rows)
    merg_ev = sum(len(r["merged_commit_ms"]) for r in rows)
    return {
        "schema": "mindie-sim.assembly/1", "scope": scope, "phase": phase, "same_context": True,
        "independent_model_invocations": 0, "submission_included": include_submission,
        "iterations_requested": int(iterations), "decode_tokens_per_iteration": decode_tokens,
        "steps_requested": total_steps, "iterations_executed": len(rows), "steps_executed": len(rows),
        "model_fragment": model_fragment, "framework_fragments": framework, "framework_occurrences": occurrence,
        "triggers": {
            "model_invocation": len(rows) if scope != "framework" else 0,
            "batch_prepare": occurrence["host_prepare"], "input_transfer": occurrence["input_transfer"],
            "output_ready": occurrence["output_ready"], "output_transfer": occurrence["output_transfer"],
            "host_result": occurrence["host_result"], "client_delivery_batches": occurrence["client_delivery"],
            "client_order_events": occurrence["client_order"], "request_commit_callbacks": decl_ev,
            "fallback_completion_callbacks": fall_ev, "merged_commit_callbacks": merg_ev,
            "sampling": service_counts["sampling"], "acceptance": service_counts["acceptance"],
            "token_commit_nodes": service_counts["token_commit"],
            "graph_replay_submissions": sum(r["graph_replay_submissions"] for r in rows),
            "eager_submissions": sum(r["eager_submissions"] for r in rows),
        },
        "state_contract": {
            "decode_step": "one_token_per_active_request", "release": "previous_step_commit_and_delivery_event",
            "context": "actual_computed_kv_positions_before_each_forward; prefill writes prompt_tokens, decode adds its query rows",
            "commit": "declared_token_commit_event_per_step",
        },
        "steps": rows, "requests": metrics,
        "metrics": {
            "boundary": "declared commit and delivery events in this context", "output_tokens": total_tokens,
            "window_end_ms": max((r["last_commit_ms"] for r in metrics if r["last_commit_ms"] is not None), default=None),
            "delivery_window_end_ms": max((r["last_delivery_ms"] for r in metrics if r["last_delivery_ms"] is not None), default=None),
            "commit_events": decl_ev, "fallback_completion_events": fall_ev, "merged_commit_events": merg_ev,
            "client_propagation_events": sum(r["client_propagation_ms"] is not None for r in rows),
            "delivery_events": sum(len(r["delivery_ms"]) for r in rows),
            "model_elapsed_ms": sum(r["step_ready_ms"] - r["release_ms"] for r in rows),
            "host_submit_events": sum(r["host_submit_event_count"] for r in rows),
        },
        "event_boundaries": {
            "ready": "input value/state producer completion", "done": "all declared device work complete",
            "submit": "actual engine host-launch event when observed; scheduler completion is not a submit boundary",
            "commit": "declared token_commit event, else step completion fallback",
            "delivery": "client-visible completion after host result",
        },
        "execution": execution.summary(),
    }
