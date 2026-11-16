"""Compose scheduler cohorts and pipeline microbatches.

[Role] Share one dispatch path between colocated and disaggregated serving.
[Invariants] Idle EP participants reach the same collectives; PP microbatches
  share actual stage streams and resources. Every microbatch carries the
  original scheduled slot, request identity, state version and replica; a
  cached template is rebound to this round's actual work instead of a local
  list position.
[Timeouts & Locks] Bounded by the active batch, no I/O.
[Side-Effect Policy] Submits to the caller's execution engine.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import inspect
import math
from collections.abc import Mapping as _AbcMapping
from typing import Any, Mapping

from mindie_sim.schema.workload import IterationPlan

ZERO = 0
ONE = 1


@dataclass(frozen=True)
class BatchBinding:
    """One real PP/EP microbatch's original scheduler slots and versions."""

    replica: int
    microbatch: int
    microbatches: int
    original_iteration: int
    plan_iteration: int
    slots: tuple[tuple[str, int], ...]
    state_versions: tuple[tuple[str, int | None], ...]

    def slot_of(self, request_id: str) -> int | None:
        return dict(self.slots).get(str(request_id))

    def version_of(self, request_id: str) -> int | None:
        return dict(self.state_versions).get(str(request_id))

    def to_dict(self) -> dict[str, Any]:
        return {
            "replica": self.replica,
            "microbatch": self.microbatch,
            "microbatches": self.microbatches,
            "original_iteration": self.original_iteration,
            "plan_iteration": self.plan_iteration,
            "slots": {request: slot for request, slot in self.slots},
            "state_versions": {
                request: version for request, version in self.state_versions},
        }


def cohorts(parallel, machine=None):
    if machine is not None:
        from .local_tp_groups import local_tp_replica_cohorts, local_tp_spans_dp
        if local_tp_spans_dp(parallel):
            return list(local_tp_replica_cohorts(machine, parallel))
    width = parallel.ep_dp_width
    if int(parallel.tp) == ONE:
        for candidate in (
            int(parallel.o_proj_tp),
            int(parallel.linear_proj_tp),
            int(parallel.lm_head_tp),
        ):
            if candidate > ONE:
                combined = math.lcm(width, candidate)
                if int(parallel.dp) % combined == ZERO:
                    width = combined
    return [tuple(range(first, first + width)) for first in range(ZERO, parallel.dp, width)]


_CALLABLE_BINDING_CACHE: dict[Any, tuple[Any, str | None]] = {}


def _lower_with_binding(lower, plan, model, runtime, machine, context,
                        binding: BatchBinding, *, fast_step: bool = False):
    """Pass the microbatch binding only to a producer that declares it.

    A cached template owner asks for this keyword explicitly; an unrelated
    custom lower keeps its existing callable contract and is never called with
    an unexpected argument.
    """
    from .frontend import Frontend
    if isinstance(lower, Frontend):
        return lower(plan, model, runtime, machine, **context,
                     dispatch_binding=binding, dispatch_fast_step=fast_step)
    cache_id = id(lower)
    cached = _CALLABLE_BINDING_CACHE.get(cache_id)
    if cached is not None and cached[0] is lower:
        keyword = cached[1]
    else:
        try:
            parameters = inspect.signature(lower).parameters.values()
        except (TypeError, ValueError):
            parameters = ()
        keyword = None
        variadic = False
        for parameter in parameters:
            if parameter.name in {"dispatch_binding", "batch_binding"}:
                keyword = parameter.name
                break
            if parameter.kind is inspect.Parameter.VAR_KEYWORD:
                variadic = True
        if keyword is None and variadic:
            keyword = "dispatch_binding"
        if len(_CALLABLE_BINDING_CACHE) >= 64:
            _CALLABLE_BINDING_CACHE.clear()
        _CALLABLE_BINDING_CACHE[cache_id] = (lower, keyword)
    if keyword is not None:
        context = {**context, keyword: binding}
    return lower(plan, model, runtime, machine, **context)


def _record_binding(record, op, binding: BatchBinding, scheduler) -> Mapping:
    if not isinstance(record, _AbcMapping):
        return record
    data = dict(record)
    op_requests = tuple(
        str(item) for item in getattr(op, "requests", ()) or ()
        if not str(item).endswith(":padding"))
    request = op_requests[0] if len(op_requests) == 1 else str(
        data.get("request") or "")
    slot = binding.slot_of(request)
    if slot is None:
        return record
    version = binding.version_of(request)
    stale = data.get("request_slot") not in (None, slot)
    data["request_slot"] = slot
    data["microbatch"] = binding.microbatch
    data["microbatches"] = binding.microbatches
    data["replica"] = binding.replica
    data.setdefault("stage", int(getattr(op, "stage", 0)))
    data["request"] = request
    if version is not None:
        # A template hit carries a previous round's slot/version. Rebind the
        # current round; an actual producer that already names this slot and
        # version keeps its own explicit binding.
        template = (bool(data.get("template"))
                    or str(data.get("binding") or "") == "template")
        if template:
            data["occurrence"] = str(op.op_id)
            if str(data.get("direction") or "read") == "read":
                data["reader"] = str(op.op_id)
            else:
                data["output"] = str(op.op_id)
        if template or data.get("state_version") is None or stale:
            data["state_version"] = version
        data.setdefault("request_version", version)
    return data


_REBIND_META_KEYS = frozenset({
    "state_accesses", "memory_accesses", "memory_bindings", "memory_binding",
})
_LAST_NO_MB_OPS: list[tuple] = [()]
_LAST_NO_REV_OPS: list[tuple] = [()]


def _rebind_work(work, binding: BatchBinding, scheduler, *,
                 fast_step: bool = False):
    """Bind this microbatch's actual occurrences to original slot/version.

    Returns the rebound work plus the exact per-occurrence maps passed through
    ``Execution.add``/``add_group``. Only producer-declared accesses are
    rebound; T14 never fabricates a state read/write.
    """
    from .execution import inherit_meta_id_key, meta_scan_flags
    use_fast = fast_step and work.replay_scope == "none" and work.graph_bucket is None
    if use_fast:
        prev_nomb = _LAST_NO_MB_OPS[0]
        if len(prev_nomb) == len(work.ops):
            all_nomb = True
            for op, pop in zip(work.ops, prev_nomb):
                if op is pop:
                    continue
                rm = op.metadata
                if rm and meta_scan_flags(rm)[2]:
                    all_nomb = False
                    break
            if all_nomb:
                _LAST_NO_MB_OPS[0] = work.ops
                return work, None, None
    ops = None
    state_accesses: dict[str, tuple] = {}
    memory_rows: list[dict] = []
    for index, op in enumerate(work.ops):
        raw_meta = op.metadata
        if not raw_meta:
            if ops is not None:
                ops.append(op)
            continue
        flags = meta_scan_flags(raw_meta)
        if not (flags[2] if use_fast else flags[0]):
            if ops is not None:
                ops.append(op)
            continue
        metadata = dict(raw_meta)
        op_changed = False
        mem_bind_changed = False
        if not use_fast:
            for key in ("state_accesses", "memory_accesses"):
                if key not in metadata:
                    continue
                records = tuple(metadata.get(key) or ())
                rebound = tuple(
                    _record_binding(record, op, binding, scheduler)
                    for record in records)
                metadata[key] = rebound
                state_accesses[str(op.op_id)] = rebound
                op_changed = True
        for key in ("memory_bindings", "memory_binding"):
            raw = metadata.get(key)
            if not isinstance(raw, _AbcMapping):
                continue
            rebound = {
                str(name): _record_binding(value, op, binding, scheduler)
                for name, value in raw.items()
            }
            metadata[key] = rebound
            for name, value in rebound.items():
                if isinstance(value, _AbcMapping):
                    memory_rows.append({
                        "occurrence": str(op.op_id), "allocation": str(name),
                        **dict(value)})
            op_changed = True
            mem_bind_changed = True
        if op_changed:
            if ops is None:
                ops = list(work.ops[:index])
            rebound_meta = tuple(metadata.items())
            if not mem_bind_changed:
                inherit_meta_id_key(raw_meta, rebound_meta)
            op = replace(op, metadata=rebound_meta)
        if ops is not None:
            ops.append(op)
    if ops is None:
        if use_fast and work.ops.__class__ is tuple:
            _LAST_NO_MB_OPS[0] = work.ops
        return work, None, None
    return (replace(work, ops=tuple(ops)),
            (state_accesses or None),
            (memory_rows or None))


def submit_batch(case, schedulers, plans, execution, lower, now, *,
                 bindings_out=None):
    from .frontend import Frontend
    count = min(case.runtime.pipeline_microbatches,
                max((len(plan.requests) for plan in plans.values() if plan),
                    default=1))
    count = max(1, int(count))
    steps = {replica: [] for replica in plans}
    for microbatch in range(count):
        local = {}
        batch_bindings = {}
        for replica, plan in plans.items():
            if plan:
                indexes = tuple(range(microbatch, len(plan.requests), count))
                requests = tuple(plan.requests[index] for index in indexes)
                slots = tuple(
                    (view.request_id, index)
                    for view, index in zip(requests, indexes, strict=True))
                versions = tuple(
                    (view.request_id,
                     (int(schedulers[replica].by_id[view.request_id].state_version)
                      if view.request_id in schedulers[replica].by_id else None))
                    for view in requests)
                binding = BatchBinding(
                    replica=replica,
                    microbatch=microbatch,
                    microbatches=count,
                    original_iteration=int(plan.iteration),
                    plan_iteration=(int(plan.iteration) * count + microbatch),
                    slots=slots,
                    state_versions=versions,
                )
                local[replica] = replace(
                    plan, requests=requests,
                    iteration=binding.plan_iteration)
            else:
                binding = BatchBinding(
                    replica=replica, microbatch=microbatch,
                    microbatches=count,
                    original_iteration=int(getattr(
                        schedulers[replica], "iteration", 0)),
                    plan_iteration=int(getattr(
                        schedulers[replica], "iteration", 0)),
                    slots=(), state_versions=())
                local[replica] = IterationPlan(
                    (), iteration=binding.plan_iteration)
            batch_bindings[replica] = binding
            if bindings_out is not None:
                bindings_out.append(binding.to_dict())
        logical_local = dict(local)
        phase = next((view.phase for plan in local.values()
                      for view in plan.requests), "decode")
        if isinstance(lower, Frontend):
            from .operator_analysis import graph_padded_idle
            from .graph import GraphPolicy
            from mindie_sim.schema.workload import RequestView
            counts = {replica: len(plan.requests)
                      for replica, plan in local.items()}
            if graph_padded_idle(case, phase, counts):
                context0 = next(
                    (view.context for plan in logical_local.values()
                     for view in plan.requests), 0)
                for replica, plan in local.items():
                    if plan.requests:
                        continue
                    probe = RequestView(
                        f"serving{replica}:bucket_probe", 0, context0,
                        False, phase, padding=True, draft_verify=False,
                        draft_prepare=False)
                    bucket, _ = GraphPolicy().select((probe,), case.runtime)
                    physical = int(bucket or 0)
                    if physical:
                        local[replica] = replace(plan, requests=(RequestView(
                            f"serving{replica}:padding", physical, context0,
                            False, phase, padding=True, draft_verify=False,
                            draft_prepare=False),))
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
        source_rows_by_replica = tuple(
            sum(view.q_len + view.draft_proposal_tokens
                for view in logical_local[replica].requests)
            for replica in local)
        if isinstance(lower, Frontend):
            # Physical graph padding and the valid routed workload are separate
            # declarations: the former shapes the graph, the latter owns the
            # routing/F3 projection.
            valid_global = sum(source_rows_by_replica)
            context["routing_global_tokens"] = valid_global
            if len(local) > 1 and "sequence_parallel" in case.runtime.kernels:
                from .sequence_parallel import select as select_sp
                physical_rows = []
                for replica, plan in local.items():
                    rows = sum(view.q_len + view.draft_proposal_tokens
                               for view in logical_local[replica].requests)
                    runtime = schedulers[replica].next_runtime(plan)
                    sp = select_sp(
                        runtime, case.model, logical_local[replica].requests,
                        tokens=rows, global_tokens=valid_global)
                    if sp is not None:
                        physical_rows.extend(sp.valid_per_tp)
                if physical_rows:
                    source_rows_by_replica = tuple(physical_rows)
            context["source_rows_by_replica"] = source_rows_by_replica
        cohort_views = tuple(
            view for replica in logical_local for view in logical_local[replica].requests)
        route_context = ({"routing_views": cohort_views}
                         if isinstance(lower, Frontend) and len(plans) > 1
                         else {})
        works = {}
        state_accesses = {}
        memory_bindings = {}
        fast_step = (
            isinstance(lower, Frontend)
            and len(plans) == 1
            and count == 1
            and execution.lazy_frontier
            and execution._prev_add_topology is not None
            and execution.offload_policy is None
            and not execution.readiness_report
            and execution.on_operator is None
            and not execution.legacy_diagnostics
        )
        for replica, plan in local.items():
            binding = batch_bindings[replica]
            call_context = {**context, **route_context}
            if isinstance(lower, Frontend):
                call_context["replica"] = replica
            work = _lower_with_binding(
                lower, plan, case.model, schedulers[replica].next_runtime(plan),
                case.machine, call_context, binding, fast_step=fast_step)
            work, accesses, bindings = _rebind_work(
                work, binding, schedulers[replica], fast_step=fast_step)
            works[replica] = work
            if accesses is not None:
                state_accesses[replica] = accesses
            if bindings is not None:
                memory_bindings[replica] = bindings
        if isinstance(lower, Frontend):
            from .execution import mark_no_rev_collective, meta_scan_flags
            has_any_rev = False
            prev_norev = _LAST_NO_REV_OPS[0]
            for work in works.values():
                wops = work.ops
                if len(prev_norev) == len(wops):
                    work_rev = False
                    for op, pop in zip(wops, prev_norev):
                        if op is pop:
                            continue
                        if op.metadata and meta_scan_flags(op.metadata)[1]:
                            work_rev = True
                            break
                else:
                    work_rev = False
                    for op in wops:
                        if op.metadata and meta_scan_flags(op.metadata)[1]:
                            work_rev = True
                            break
                if work_rev:
                    has_any_rev = True
                else:
                    mark_no_rev_collective(wops)
                    if wops.__class__ is tuple:
                        _LAST_NO_REV_OPS[0] = wops
            if has_any_rev:
                # Serving and target_graphs share the same ordinary cohort producer;
                # never leave serving with only the generic uniform peer bytes.
                from .cohort_traffic import bind_cohort_traffic
                bound = bind_cohort_traffic(
                    case, {replica: work.ops for replica, work in works.items()})
                works = {replica: replace(work, ops=bound[replica])
                         for replica, work in works.items()}
        # SERVICE-CORE owns the integer emission projection. Use the same
        # accept policy that commit_batch will apply after the step completes.
        emitted = {
            replica: schedulers[replica].emissions(logical_local[replica])
            for replica in local
        }
        kwargs = {
            "emitted_tokens": emitted,
            "state_accesses": state_accesses or None,
            "memory_bindings": memory_bindings or None,
        }
        if len(plans) > 1:
            submitted = execution.add_group(works, now, **kwargs)
        else:
            submitted = {
                replica: execution.add(
                    work, replica, now,
                    emitted_tokens=emitted[replica],
                    state_accesses=(state_accesses or {}).get(replica),
                    memory_bindings=(memory_bindings or {}).get(replica))
                for replica, work in works.items()}
        for replica, step in submitted.items():
            # A graph-padding lane is physical work, not a request identity.
            # It never publishes a commit/host/delivery row for row 0.
            logical_ids = {
                view.request_id for view in logical_local[replica].requests
                if not getattr(view, "padding", False)}
            step = replace(
                step,
                commits={request: event for request, event in step.commits.items()
                         if request in logical_ids},
                declared_commits={
                    request: event for request, event in step.declared_commits.items()
                    if request in logical_ids},
                fallback_commits={
                    request: event for request, event in step.fallback_commits.items()
                    if request in logical_ids},
                host_results={
                    request: event for request, event in step.host_results.items()
                    if request in logical_ids},
                client_deliveries={
                    request: event for request, event in step.client_deliveries.items()
                    if request in logical_ids},
            )
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
            declared_commits={request: event for item in items
                              for request, event in item.declared_commits.items()},
            fallback_commits={request: event for item in items
                              for request, event in item.fallback_commits.items()},
            host_results={request: event for item in items for request, event in item.host_results.items()},
            client_deliveries={request: event for item in items for request, event in item.client_deliveries.items()},
            host_submits=tuple(event for item in items for event in item.host_submits),
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
