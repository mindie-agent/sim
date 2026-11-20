"""Offload scheduling, STARS initialization, stream-plan, and step helpers.

[Role]
  Provide modular setup and per-step offload/readiness/lazy-eligibility helpers
  for ``mindie_sim.prediction.execution.runner`` so the core ``Execution``
  runner stays compact and well below the module size budget.

[Invariants]
  * Offload read/write/tail bindings honor exact producer/reader access events
    and never fabricate copy completions.
  * STARS queues are instantiated strictly for stages whose ``StarsSpec``
    declares positive ``issue_interval_us`` or ``first_issue_us``.

[Timeouts & Locks]
  Bounded in-memory setup and Engine task submission; no I/O or locks.

[Side-Effect Policy]
  Submits tasks only to the caller's ``Engine`` and updates bounded caches.
"""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from typing import Any

from mindie_sim.schema.common import ModelingError
from ..engine import Resource
from ..execution_state import bind_op_accesses, mandatory_edges
from .streams import (
    _collective_native_basis,
    _declaration_of,
    _metadata_dict,
)
from .submission import (
    DEFAULT_SPLIT_DTYPE_BYTES,
    EagerSplitBarrier,
    ScheduledIteration,
    SplitBarrierSpec,
    StarsDescriptorQueue,
)

LAZY_SAFE_CACHE_LIMIT = 16384
DEFAULT_FAST_LAZY_STARS_DEPTH = 64
EAGER_MOE_DISPATCH_RECIPES = frozenset({
    "expert_dispatch", "expert_quantized_dispatch",
})


def _stars_controller_id(spec, replica: int, stage: int, die) -> str:
    scope = str(getattr(spec, "queue_scope", "die") or "die")
    if scope == "die":
        return f"die{int(die.rank)}"
    if scope == "package":
        return f"n{int(getattr(die, 'node', 0))}:pkg{int(getattr(die, 'package', 0))}"
    if scope == "node":
        return f"n{int(getattr(die, 'node', 0))}"
    if scope == "replica":
        return f"r{int(replica)}"
    return f"r{int(replica)}:s{int(stage)}"


def _stars_is_noop_placeholder(spec) -> bool:
    """True only for the uncalibrated default profile placeholder."""
    return (
        float(spec.issue_interval_us) == 0.0
        and float(spec.first_issue_us) == 0.0
        and str(spec.release_policy) == "descriptor_ready"
        and int(spec.queue_depth) >= DEFAULT_FAST_LAZY_STARS_DEPTH
        and str(getattr(spec, "evidence_id", "")) == "pending_raw_dispatch_trace"
    )


def build_stars_queues(
    stage_profiles, namespace: str
) -> dict[tuple[int, int], tuple[tuple[int, StarsDescriptorQueue], ...]]:
    """Build representative-shard STARS queues for every active ``StarsSpec``.

    Instantiates active queues even when ``issue_interval_us == 0`` (e.g.
    explicit finite ``queue_depth`` or non-placeholder evidence) so credit
    depth and ``release_policy`` govern concurrency, sharing controllers across
    replicas/stages that map to the same ``queue_scope``.
    """
    by_controller: dict[str, StarsDescriptorQueue] = {}
    stage_queues: dict[tuple[int, int], tuple[tuple[int, StarsDescriptorQueue], ...]] = {}
    for key, profile in stage_profiles.items():
        datapath = getattr(profile, "datapath", None)
        spec = getattr(datapath, "stars_spec", None)
        if spec is None or _stars_is_noop_placeholder(spec):
            continue
        replica, stage = key
        dies = profile.dies or ()
        rep_die = (
            dies[0] if dies
            else type("_D", (), {"rank": replica, "node": 0, "package": 0})()
        )
        die_rank = int(rep_die.rank)
        ctrl = _stars_controller_id(spec, replica, stage, rep_die)
        queue = by_controller.get(ctrl)
        if queue is None:
            queue = StarsDescriptorQueue(
                spec, die_id=die_rank, namespace=namespace, controller_id=ctrl
            )
            by_controller[ctrl] = queue
        stage_queues[key] = ((die_rank, queue),)
    return stage_queues


def attach_stars_queues(execution) -> None:
    """Build and register STARS queues/pools on ``execution.engine``."""
    execution.stars_queues = build_stars_queues(
        execution.stage_profiles, execution.namespace
    )
    seen: set[int] = set()
    for entries in execution.stars_queues.values():
        for _die_rank, queue in entries:
            if id(queue) not in seen:
                seen.add(id(queue))
                queue.ensure_engine_resources(execution.engine)
    execution._stars_replay_seq = {}
    execution._stars_occ_seq = 0
    execution.eager_split_barriers = []


def build_stars_summary(execution) -> dict[str, Any]:
    """Assemble per-stage and per-die STARS descriptor summaries."""
    out: dict[str, Any] = {}
    for (replica, stage), entries in sorted(execution.stars_queues.items()):
        dies_out = {
            f"d{die_rank}": q.summary()
            for die_rank, q in entries if q.descriptor_count > 0
        }
        if dies_out:
            primary = next(iter(dies_out.values()))
            out[f"r{replica}:s{stage}"] = {
                **primary, "die_count": len(dies_out), "dies": dies_out,
            }
    return out


def bind_stream_plans(stream_plans, stage_profiles, namespace: str):
    """Resolve N-stream owner partitions and quota resources for every stage."""
    stream_owners: dict[tuple[int, int, str], Any] = {}
    stream_quota_by_res: dict[tuple[str, str], str] = {}
    extra_resources: list[Resource] = []
    if not stream_plans:
        return stream_owners, stream_quota_by_res, extra_resources
    from ..streams import bind_stream_owners, stream_quota_resources
    plans_seq = (
        tuple(stream_plans.values())
        if isinstance(stream_plans, _AbcMapping)
        else tuple(stream_plans)
    )
    for plan in plans_seq:
        matched = False
        for (rep, stg), profile in stage_profiles.items():
            if rep != int(plan.rank):
                continue
            matched = True
            bound = bind_stream_owners(plan, profile.datapath)
            for stream_id, owners in bound.items():
                stream_owners[(rep, stg, stream_id)] = owners
            stage_rates = {res.name: float(res.rate) for res in profile.resources}
            qres, qmap = stream_quota_resources(
                bound,
                profile.datapath,
                resource_names=profile.names,
                rates=stage_rates,
                work_units=profile.work_units,
                replica=rep,
                stage=stg,
                namespace=namespace,
            )
            extra_resources.extend(qres)
            for (sid, phys), qname in qmap.items():
                stream_quota_by_res[(phys, sid)] = qname
        if not matched:
            raise ModelingError(f"stream_plan_unmatched_rank:{plan.rank}")
    return stream_owners, stream_quota_by_res, extra_resources


def _stars_allows_lazy_fastpath(execution, physical_ops) -> bool:
    if not execution.stars_queues:
        return True
    for entries in execution.stars_queues.values():
        for _die_rank, queue in entries:
            spec = queue.spec
            if (spec.issue_interval_us > 0.0
                    or spec.first_issue_us > 0.0
                    or spec.release_policy != "descriptor_ready"
                    or int(spec.queue_depth) < DEFAULT_FAST_LAZY_STARS_DEPTH):
                return False
    streams = {op.stream or "main" for op in physical_ops}
    return len(streams) <= 1


def _has_eager_moe_split_barrier(execution, physical_ops) -> bool:
    if not execution.include_submission or int(execution.case.runtime.parallel.ep) <= 1:
        return False
    machine = execution.case.machine
    if not (hasattr(machine, "host_solo_bw_gbps") and machine.host_solo_bw_gbps("d2h")):
        return False
    return any(
        op.recipe in EAGER_MOE_DISPATCH_RECIPES
        and str(getattr(op, "phase", "")).strip().lower() == "prefill"
        for op in physical_ops
    )


def try_lazy_add(execution, work, replica, release_ms, *, scope, schedule,
                 prepared, input_ready, physical_ops, prefix, external,
                 emitted_tokens, memory_bindings, state_accesses):
    """Delegate eligible single-replica steps to active-frontier lazy submit."""
    if not (execution.lazy_frontier and physical_ops
            and scope == "none" and execution._collectives is None
            and execution.offload_policy is None
            and not execution.readiness_report
            and _stars_allows_lazy_fastpath(execution, physical_ops)
            and not _has_eager_moe_split_barrier(execution, physical_ops)):
        return None
    safe_cache = execution._lazy_safe_id_cache

    def _lazy_safe(op):
        if not op.metadata:
            return True
        hit = safe_cache.get(id(op))
        if hit is not None and hit[0] is op:
            return hit[1]
        meta = _metadata_dict(op)
        ok = (_declaration_of(op) is None
              and meta.get("group_axis") not in {"ep", "local_tp"}
              and not meta.get("peer_traffic")
              and not meta.get("mc2_phase")
              and not meta.get("reverse_of_collective"))
        if len(safe_cache) >= LAZY_SAFE_CACHE_LIMIT:
            safe_cache.clear()
        safe_cache[id(op)] = (op, ok)
        return ok

    p_ops_t = physical_ops if physical_ops.__class__ is tuple else tuple(physical_ops)
    prev_top = execution._prev_add_topology
    if (prev_top is not None and not external and not prev_top[1]
            and len(p_ops_t) == len(prev_top[0])):
        same_top = True
        for op, pop in zip(p_ops_t, prev_top[0]):
            if op is pop:
                continue
            if (op.op_id != pop.op_id or op.deps != pop.deps
                    or (op.family == "token_commit") != (pop.family == "token_commit")
                    or not _lazy_safe(op)):
                same_top = False
                break
        if same_top:
            execution._prev_add_topology = (p_ops_t, (), prev_top[2])
            return execution._add_lazy(
                work, replica, release_ms, scope=scope, schedule=schedule,
                prepared=prepared, input_ready=input_ready, physical_ops=p_ops_t,
                prefix=prefix, external_inputs=external,
                emitted_tokens=emitted_tokens, memory_bindings=memory_bindings,
                state_accesses=state_accesses, verified_dag=prev_top[2])
    known = {op.op_id for op in p_ops_t}
    deps_known = all(
        dep in known or dep in external
        for op in p_ops_t for dep in op.deps)
    if deps_known and all(_lazy_safe(op) for op in p_ops_t):
        res_step = execution._add_lazy(
            work, replica, release_ms, scope=scope, schedule=schedule,
            prepared=prepared, input_ready=input_ready, physical_ops=p_ops_t,
            prefix=prefix, external_inputs=external,
            emitted_tokens=emitted_tokens, memory_bindings=memory_bindings,
            state_accesses=state_accesses)
        if not external and execution._prev_lazy_dag is not None:
            execution._prev_add_topology = (p_ops_t, (), execution._prev_lazy_dag[1])
        return res_step
    return None


def observe_collective_readiness(execution, op, replica, ready):
    """Record pipeline readiness observations for a local collective op."""
    from ..collective_readiness import bind_pipeline_ready
    ranks = execution.topology.ranks(replica, op.stage)
    ready_by_rank = {rank: ready for rank in ranks}
    collective, expected_bytes, expected_dtype = _collective_native_basis(op, len(ranks))
    meta = dict(op.metadata)
    try:
        readiness_context = bind_pipeline_ready(
            execution.readiness_report, ready_by_rank,
            expected_collective=collective,
            expected_bytes=(int(expected_bytes)
                            if expected_bytes is not None
                            and expected_bytes.denominator == 1 else None),
            expected_dtype=expected_dtype,
            expected_group_size=len(ranks),
            connection_id=meta.get("collective_connection_id"),
            ordinal=meta.get("collective_call_ordinal"),
            call_id=meta.get("collective_call_id"))
    except ModelingError as exc:
        execution.readiness_unresolved.append({"op_id": op.op_id, "reason": str(exc)})
    else:
        execution.readiness_observations.append({
            "op_id": op.op_id, "replica": replica,
            "ready_by_rank": tuple(sorted(ready_by_rank)),
            "context": readiness_context,
        })


def schedule_op_offload_reads(
    execution, op, work, replica, prefix, name, input_ready, deps, ids,
    layer_entries, expert_layer_events, last_expert_done, kv_layer_events,
    last_kv_done, state_accesses, all_bound_accesses,
):
    """Bind expert and KV read offloads before an operator starts."""
    policy = execution.offload_policy
    if policy is not None and policy.expert_ddr_ratio > 0.0:
        from ..offload import is_routed_expert_entry_op, schedule_expert_layer_offload
        if is_routed_expert_entry_op(op):
            exp_key = (op.stage, op.layer)
            if exp_key not in expert_layer_events:
                expert_layer_events[exp_key] = schedule_expert_layer_offload(
                    execution, prefix, replica, op.stage, op.layer, work.valid_tokens,
                    last_expert_done.get(op.stage, input_ready), op=op)
            if expert_layer_events[exp_key] is not None:
                deps.append(expert_layer_events[exp_key])
    bound_accesses = (
        () if (policy is None and state_accesses is None)
        else bind_op_accesses(work, op, state_accesses=state_accesses))
    if bound_accesses:
        all_bound_accesses.extend(bound_accesses)
    if policy is not None and (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) > 0.0:
        from ..offload import schedule_kv_layer_offload
        reads: dict[str, list] = {}
        for bound in bound_accesses:
            access = bound.access
            if access.direction == "read" and access.bytes > 0.0:
                reads.setdefault(access.reader or op.op_id, []).append(bound)
        for reader, group in reads.items():
            accesses = tuple(bound.access for bound in group)
            edges = tuple(dict.fromkeys(
                edge for bound in group
                for edge in mandatory_edges(
                    bound.access, ids, bound.memory_dict())))
            mandatory = (execution.engine.boundary(
                f"{name}:access_ready:{reader}", after=edges)
                if edges else None)
            copy_events = schedule_kv_layer_offload(
                execution, prefix, replica, op.stage, op.layer, work,
                input_ready, tuple(deps), layer_entries,
                prev_kv_done=last_kv_done.get(op.stage),
                state_accesses=accesses, source_ready=mandatory)
            if copy_events:
                deps.extend(copy_events)
                kv_layer_events.setdefault((op.stage, op.layer), []).extend(copy_events)
    return bound_accesses


def schedule_op_offload_writes(
    execution, op, work, replica, prefix, complete, bound_accesses,
    last_expert_done, last_kv_done, kv_writeback_events, owned_offload,
    last_draft_done, pending_draft,
):
    """Bind expert completion and KV/draft writebacks after an operator finishes."""
    policy = execution.offload_policy
    if policy is None:
        return
    if policy.expert_ddr_ratio > 0.0:
        from ..offload import is_routed_expert_entry_op
        if is_routed_expert_entry_op(op) or op.recipe in {"grouped_down", "balanced_grouped_down"}:
            last_expert_done[op.stage] = complete
    if (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) > 0.0:
        from ..offload import schedule_kv_layer_writeback
        target_writes = tuple(
            bound.access for bound in bound_accesses
            if bound.access.direction == "write" and bound.access.bytes > 0.0
            and bound.access.scope != "draft" and bound.access.producer == op.op_id)
        if target_writes:
            last_kv_done[op.stage] = complete
            copy_events = schedule_kv_layer_writeback(
                execution, prefix, replica, op.stage, op.layer, work, complete,
                state_accesses=target_writes)
            if copy_events:
                kv_writeback_events.extend(copy_events)
                owned_offload.extend(copy_events)
        draft_writes = tuple(
            bound.access for bound in bound_accesses
            if bound.access.direction == "write" and bound.access.bytes > 0.0
            and bound.access.scope == "draft" and bound.access.producer == op.op_id)
        if draft_writes:
            last_draft_done[op.stage] = complete
            pending_draft.setdefault(op.stage, []).extend(draft_writes)


def schedule_step_offload_tails(
    execution, work, replica, prefix, input_ready, physical_ops, ids,
    pending_draft, last_draft_done, all_bound_accesses, kv_layer_events,
    kv_writeback_layers, kv_writeback_events, owned_offload,
):
    """Flush remaining draft/iteration KV writebacks at step tail."""
    policy = execution.offload_policy
    if policy is None or (policy.kvc_ddr_ratio + policy.kvc_ssd_ratio) <= 0.0:
        return
    from ..offload import schedule_draft_kv_writeback, schedule_kv_iteration_offload
    for stage, writes in sorted(pending_draft.items()):
        producer_done = last_draft_done.get(stage)
        if producer_done is not None:
            copy_events = schedule_draft_kv_writeback(
                execution, prefix, replica, stage, work, producer_done,
                state_accesses=tuple(writes))
            if copy_events:
                kv_writeback_events.extend(copy_events)
                owned_offload.extend(copy_events)
    access_union = tuple(bound.access for bound in all_bound_accesses)
    iteration_events = schedule_kv_iteration_offload(
        execution, prefix, replica, work, input_ready, physical_ops, ids,
        scheduled_layers=set(kv_layer_events) | kv_writeback_layers,
        state_accesses=access_union, scheduled_accesses=None)
    if iteration_events:
        kv_writeback_events.extend(iteration_events)
        owned_offload.extend(iteration_events)


def finalize_step(
    execution, work, replica: int, release_ms: float, *, prefix: str,
    scope: str, input_ready, physical_ops, ids, declared_commits,
    host_submits, kv_layer_events, kv_writeback_events, emitted_tokens,
) -> ScheduledIteration:
    """Emit device completion, offload release, serving output, and step summary."""
    engine = execution.engine
    owner = f"replica:{replica}"
    offload_tails = (
        tuple(ev for evs in kv_layer_events.values() for ev in evs)
        + tuple(kv_writeback_events))
    device_complete = engine.submit(
        prefix + ":device_complete", owner=owner,
        after=(tuple(ids[op.op_id] for op in physical_ops)
               + offload_tails) or (input_ready,))
    from ..offload import release_offload_access
    offload_release = engine.submit(
        prefix + ":offload_release",
        after=(device_complete, *offload_tails),
        owner=owner, native_hook=True,
        on_complete=lambda _event: release_offload_access(execution, work))
    completion = engine.submit(
        prefix + ":complete", after=(device_complete,),
        latency=execution.submission.completion_ms if execution.include_submission else 0,
        owner=owner)
    fallback_commits = {
        view.request_id: completion for view in work.plan.requests
        if view.request_id not in declared_commits}
    commits = {**fallback_commits, **declared_commits}
    host_results, clients, host_complete = execution.serving_io.output(
        execution, prefix, replica, work, commits, completion,
        emitted_tokens if emitted_tokens is not None else {
            view.request_id: int(view.draft_advanced_tokens or 1)
            for view in work.plan.requests if view.samples})
    if execution.serving_io.enabled:
        completion = engine.submit(
            prefix + ":step_ready", after=(completion, host_complete),
            owner=owner)
    step = ScheduledIteration(
        replica, work.plan.iteration, completion, release_ms,
        work.mode, work.valid_tokens, work.padded_tokens,
        work.graph_bucket, scope, len(physical_ops), commits,
        device_complete, host_results, clients,
        work.implementation_selections,
        tuple(dict.fromkeys(host_submits)),
        dict(declared_commits), dict(fallback_commits),
        offload_release)
    engine.submit(prefix + ":step_metrics", after=(completion,),
                  on_complete=lambda _: execution.step_summary.record(step),
                  native_hook=True)
    return step


def build_submission_summary(execution) -> dict[str, Any]:
    """Assemble the host submission summary block."""
    return {
        "included": execution.include_submission,
        **execution.submission_counts,
        "host_service_ms": sum(
            execution.engine.serviced.get(names["host"], 0)
            for names in execution.stage_names.values()),
        "stages": [
            {"replica": replica, "stage": stage,
             "kernel_launch_ms": execution.submission.launch_ms(dies),
             "graph_replay_ms": execution.submission.launch_ms(dies, captured=True),
             "kernel_parameter": (f"{execution.submission_source}.kernel_launch_ms"
                                  if execution.submission.kernel_launch_ms is not None
                                  else "machine.device.launch_us"),
             "replay_parameter": (f"{execution.submission_source}.graph_replay_ms"
                                  if execution.submission.graph_replay_ms is not None
                                  else "kernel_launch_parameter")}
            for (replica, stage), dies in execution.stage_dies.items()
        ],
        "boundary": "host enqueue service; may overlap ready device work",
        "zero_parameter": "excludes undeclared host cost; not measured zero overhead",
    }


def schedule_eager_moe_split_barrier(
    execution, op, replica: int, scope: str, name: str, deps: list, ids: dict,
):
    """Wire ``EagerSplitBarrier`` into host+device Prefill Eager MoE dispatch."""
    if (not execution.include_submission
            or scope != "none"
            or op.recipe not in EAGER_MOE_DISPATCH_RECIPES
            or str(getattr(op, "phase", "")).strip().lower() != "prefill"):
        return None
    meta = _metadata_dict(op)
    ep_size = int(meta.get("moe_ep_size") or execution.case.runtime.parallel.ep or 1)
    if ep_size <= 1:
        return None
    machine = execution.case.machine
    if not (hasattr(machine, "host_solo_bw_gbps") and machine.host_solo_bw_gbps("d2h")):
        return None
    stage_dies = execution.stage_dies.get((replica, op.stage), ())
    if not stage_dies:
        return None
    num_experts = int(
        meta.get("moe_num_experts")
        or getattr(execution.case.model, "num_experts", 0)
        or ep_size
    )
    num_local_experts = max(1, num_experts // ep_size)
    dtype_bytes = int(meta.get("split_dtype_bytes") or DEFAULT_SPLIT_DTYPE_BYTES)
    spec = SplitBarrierSpec(
        ep_size=ep_size,
        num_local_experts=num_local_experts,
        split_dtype_bytes=dtype_bytes,
        pinned=bool(meta.get("split_pinned", True)),
        non_blocking=bool(meta.get("split_non_blocking", True)),
    )
    engine = execution.engine
    owner = f"replica:{replica}"
    barrier_entry = engine.submit(
        f"{name}:eager_split:entry", after=tuple(deps), owner=owner
    )
    ag_event = next(
        (ev for op_id, ev in ids.items() if op_id.endswith(".moe_histogram_gather")),
        None,
    )
    indep_events = (
        tuple(ev for op_id, ev in ids.items() if op_id not in op.deps)
        if spec.requires_explicit_device_sync
        else ()
    )
    result = EagerSplitBarrier.submit(
        engine,
        machine,
        stage_dies[0],
        spec,
        barrier_entry,
        name=f"{name}:eager_split",
        namespace=execution.namespace,
        allgather_event=ag_event,
        independent_device_events=indep_events,
        owner=owner,
    )
    deps.append(result.dispatch_ready)
    execution.eager_split_barriers.append({
        "op_id": op.op_id,
        "replica": replica,
        "stage": op.stage,
        "ep_size": ep_size,
        "num_local_experts": num_local_experts,
        "input_splits_bytes": result.input_splits_bytes,
        "output_splits_bytes": result.output_splits_bytes,
        "explicit_device_sync": result.explicit_device_sync is not None,
        "d2h_resources": list(result.d2h_resources),
    })
    return barrier_entry


def build_execution_summary(execution) -> dict[str, Any]:
    """Assemble the complete Execution summary dict."""
    native_groups = execution.native_group_summary()
    eng_summary = execution.engine.summary()
    serviced = {
        k: v for k, v in eng_summary["serviced"].items()
        if (not execution.stream_quota_by_res or "stream_quota@" not in k)
        and not ("host_rc:" in k and v == 0)
        and not ("stars:" in k and v == 0)
    }
    busy_ms = {
        k: v for k, v in eng_summary["resource_busy_ms"].items()
        if (not execution.stream_quota_by_res or "stream_quota@" not in k)
        and not ("host_rc:" in k and v == 0)
        and not ("stars:" in k and v == 0)
    }
    if (len(serviced) != len(eng_summary["serviced"])
            or len(busy_ms) != len(eng_summary["resource_busy_ms"])):
        eng_summary = {
            **eng_summary,
            "serviced": serviced,
            "resource_busy_ms": busy_ms,
        }
    stars = execution.stars_summary()
    return {
        **eng_summary, "work": dict(execution.work),
        **({"work_basis": "logical_and_analytical_reference; executed_service_is_in_serviced"}
           if execution.cost.joint_native else {}),
        "recipes": dict(execution.recipes_used),
        "submission": execution.submission_summary(),
        **({"stars": stars} if stars else {}),
        **({"eager_split_barriers": list(execution.eager_split_barriers)}
           if execution.eager_split_barriers else {}),
        "serving_io": execution.serving_io.summary(),
        "step_summary": execution.step_summary.to_dict(),
        "rank_model": "representative_local_shard_per_stage",
        **({"native_group_count": len(execution.native_group_launches),
            "native_groups": native_groups} if native_groups else {}),
        **({"collective_readiness_observations": list(execution.readiness_observations),
            "collective_readiness_unresolved": list(execution.readiness_unresolved)}
           if (execution.readiness_observations or execution.readiness_unresolved) else {}),
        **({"joint_collective_observations": list(
                execution._joint_collective_trace.values()),
            "joint_collective_stats": dict(execution.joint_collective_stats)}
           if (execution._joint_collective_trace is not None
               and execution.joint_collective_stats["observations"]) else {}),
    }
