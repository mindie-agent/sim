"""Execution runner, step evaluation and cohort launch orchestration.

[Role] Execute single-step and cohort replay graphs through the canonical
  runtime Engine and assemble StepEvaluation results.
[Invariants] Strict wall-clock deadline enforcement; exact physical conservation.
[Timeouts & Locks] Honors ReplayOptions.deadline_mono; no locks.
[Side-Effect Policy] Pure simulation."""
from __future__ import annotations

from collections.abc import Mapping as _AbcMapping
from typing import Any

from mindie_sim.schema.common import ModelingError
from ..engine import Resource
from ..execution.state import bind_op_accesses, mandatory_edges
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


NS_PER_US = 1000.0
NS_PER_MS = 1_000_000.0
_ACLGRAPH_SCOPES = frozenset({"full", "aclgraph", "cudagraph", "graph", "piecewise"})


def _is_raw_eager_runtime(runtime) -> bool:
    """True when runtime metadata explicitly enables raw_eager execution."""
    if runtime is None:
        return False
    raw_meta = getattr(runtime, "metadata", None) or {}
    meta = dict(raw_meta) if not isinstance(raw_meta, _AbcMapping) else raw_meta
    if bool(meta.get("raw_eager", False)):
        return True
    policy = meta.get("execution_policy")
    if isinstance(policy, _AbcMapping) and bool(policy.get("raw_eager", False)):
        return True
    return str(meta.get("feature_track") or "").strip().lower() == "raw_eager"


def _calibrated_host_submit_ms(primitive_table) -> float:
    """Return calibrated P15_rts_stars host_submit_ns in ms, or 0.0 when unmeasured."""
    if primitive_table is None or not hasattr(primitive_table, "cell"):
        return 0.0
    from mindie_sim.primitives.loader import _table_band_p50
    host_ns = _table_band_p50(primitive_table, "P15_rts_stars", "host_submit_ns")
    if host_ns is None or host_ns <= 0.0:
        return 0.0
    return float(host_ns) / NS_PER_MS


def _stars_is_noop_placeholder(spec) -> bool:
    """True only for the uncalibrated default profile placeholder."""
    return (
        float(spec.issue_interval_us) == 0.0
        and float(spec.first_issue_us) == 0.0
        and str(spec.release_policy) == "descriptor_ready"
        and int(spec.queue_depth) >= DEFAULT_FAST_LAZY_STARS_DEPTH
        and str(getattr(spec, "evidence_id", "")) == "pending_raw_dispatch_trace"
    )


def _calibrated_stars_spec(
    spec, primitive_table, *, submission_mode: str = "eager", owner_kind: str = "aic",
):
    """Bind dispatch-only P15 STARS cadence matching owner kind and submission mode."""
    if primitive_table is None or not hasattr(primitive_table, "cell"):
        return None
    from dataclasses import replace as _replace
    from mindie_sim.primitives.loader import _table_band_p50
    from mindie_sim.schema.common import ModelingError

    try:
        p15 = primitive_table.cell("P15_rts_stars")
    except ModelingError:
        return None
    params = p15.parameters or {}
    if params.get("issue_interval_includes_min_kernel", False):
        return None
    measured_owner = str(params.get("issue_interval_owner") or "all").strip().lower()
    if measured_owner not in {"all", owner_kind}:
        return None
    interval_metric = "npugraph_issue_ns" if submission_mode == "aclgraph" else "issue_interval_ns"
    interval_ns = _table_band_p50(primitive_table, "P15_rts_stars", interval_metric)
    if interval_ns is None or interval_ns <= 0.0:
        return None
    first_ns = (
        None
        if params.get("first_issue_includes_min_kernel", False)
        else _table_band_p50(primitive_table, "P15_rts_stars", f"{owner_kind}_first_issue_ns")
    )
    first_extra_us = (
        max(0.0, (first_ns - interval_ns) / NS_PER_US)
        if (first_ns is not None and first_ns > interval_ns)
        else 0.0
    )
    return _replace(
        spec,
        issue_interval_us=float(interval_ns) / NS_PER_US,
        first_issue_us=float(first_extra_us),
        release_policy=spec.release_policy,
        evidence_id=f"p15:{primitive_table.sku}:{primitive_table.canonical_envelope_sha256()[:12]}",
    )


def build_stars_queues(
    stage_profiles, namespace: str, *, include_submission: bool = False,
    submission_mode: str = "eager",
) -> dict[tuple[int, int], tuple[tuple[int, StarsDescriptorQueue], ...]]:
    """Build per-die STARS queues for every active ``StarsSpec``.

    Instantiates active queues across every participating die even when
    ``issue_interval_us == 0`` (e.g. finite ``queue_depth`` or calibrated
    evidence), sharing controllers strictly across dies in the same
    ``queue_scope``.
    """
    by_controller: dict[str, StarsDescriptorQueue] = {}
    stage_queues: dict[tuple[int, int], tuple[tuple[int, StarsDescriptorQueue], ...]] = {}
    for key, profile in stage_profiles.items():
        datapath = getattr(profile, "datapath", None)
        spec = getattr(datapath, "stars_spec", None)
        if spec is None:
            continue
        if _stars_is_noop_placeholder(spec):
            if not include_submission:
                continue
            spec = _calibrated_stars_spec(
                spec, getattr(profile, "primitive_table", None),
                submission_mode=submission_mode, owner_kind="aic",
            )
            if spec is None:
                continue
        replica, stage = key
        dies = profile.dies or ()
        entries: list[tuple[int, StarsDescriptorQueue]] = []
        if not dies:
            stub = type("_D", (), {"rank": replica, "node": 0, "package": 0})()
            ctrl = _stars_controller_id(spec, replica, stage, stub)
            queue = by_controller.setdefault(
                ctrl, StarsDescriptorQueue(spec, die_id=int(replica), namespace=namespace, controller_id=ctrl)
            )
            entries.append((int(replica), queue))
        else:
            for die in dies:
                die_rank = int(die.rank)
                ctrl = _stars_controller_id(spec, replica, stage, die)
                queue = by_controller.get(ctrl)
                if queue is None:
                    queue = StarsDescriptorQueue(
                        spec, die_id=die_rank, namespace=namespace, controller_id=ctrl
                    )
                    by_controller[ctrl] = queue
                entries.append((die_rank, queue))
        stage_queues[key] = tuple(entries)
    return stage_queues


def attach_stars_queues(execution) -> None:
    """Build and register STARS queues/pools on ``execution.engine``."""
    rt = getattr(getattr(execution, "case", None), "runtime", None)
    raw_scope = (
        getattr(rt, "replay_scope", None)
        or getattr(rt, "graph_capture_mode", None)
        or "none"
    )
    scope = str(raw_scope).strip().lower()
    sub_mode = "aclgraph" if scope in _ACLGRAPH_SCOPES else "eager"
    execution.stars_queues = build_stars_queues(
        execution.stage_profiles,
        execution.namespace,
        include_submission=bool(getattr(execution, "include_submission", False)),
        submission_mode=sub_mode,
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
            seen_q: dict[int, dict[str, Any]] = {}
            for _die_rank, q in entries:
                if q.descriptor_count > 0 and id(q) not in seen_q:
                    seen_q[id(q)] = q.summary()
            primary = next(iter(seen_q.values()))
            out[f"r{replica}:s{stage}"] = {
                **primary,
                "descriptor_count": sum(s["descriptor_count"] for s in seen_q.values()),
                "total_descriptor_ns": sum(s["descriptor_union_ns"] for s in seen_q.values()),
                "die_count": len(dies_out),
                "dies": dies_out,
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
    from ..collectives.readiness import bind_pipeline_ready
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


def _stage_submission_entry(execution, stage_key: tuple[int, int], dies) -> dict[str, Any]:
    replica, stage = stage_key
    base_launch = execution.submission.launch_ms(dies)
    eff_launch = execution._stage_launch_ms(stage_key, captured=False)
    if execution.submission.kernel_launch_ms is not None:
        kernel_param = f"{execution.submission_source}.kernel_launch_ms"
    elif base_launch <= 0.0 < eff_launch:
        kernel_param = "primitive_table.P15_rts_stars.host_submit_ns"
    else:
        kernel_param = "machine.device.launch_us"
    replay_param = (
        f"{execution.submission_source}.graph_replay_ms"
        if execution.submission.graph_replay_ms is not None
        else "kernel_launch_parameter"
    )
    return {
        "replica": replica,
        "stage": stage,
        "kernel_launch_ms": eff_launch,
        "graph_replay_ms": execution._stage_launch_ms(stage_key, captured=True),
        "kernel_parameter": kernel_param,
        "replay_parameter": replay_param,
    }


def build_submission_summary(execution) -> dict[str, Any]:
    """Assemble the host submission summary block."""
    return {
        "included": execution.include_submission,
        **execution.submission_counts,
        "host_service_ms": sum(
            execution.engine.serviced.get(names["host"], 0)
            for names in execution.stage_names.values()),
        "stages": [
            _stage_submission_entry(execution, (replica, stage), dies)
            for (replica, stage), dies in execution.stage_dies.items()
        ],
        "boundary": "host enqueue service; may overlap ready device work",
        "zero_parameter": "excludes undeclared host cost; not measured zero overhead",
    }


def _submit_eager_counts_allgather(
    execution, op, replica: int, stage_dies, *, ep_size: int, dtype_bytes: int,
    name: str, barrier_entry, ids: dict, owner: str,
) -> tuple[dict[int, Any], str]:
    """Resolve current-dispatch counts AllGather op or lower it on the EP group."""
    layer_prefix = op.op_id.rsplit(".", 1)[0] if "." in str(op.op_id) else ""
    expected_hist = f"{layer_prefix}.moe_histogram_gather" if layer_prefix else ""
    if expected_hist and expected_hist in ids:
        ev = ids[expected_hist]
        return {int(d.rank): ev for d in stage_dies}, "op"
    from ..collectives.protocol import lower_collective, submit_lowered_collective
    topo = execution.topology
    stage_ranks = tuple(int(d.rank) for d in stage_dies)
    ep_groups = topo.groups("ep", replica, op.stage, metadata=_metadata_dict(op))
    ep_ranks = next(
        (grp for grp in ep_groups if set(stage_ranks).intersection(grp)),
        stage_ranks,
    )
    if len(ep_ranks) <= 1:
        return {}, "none"
    try:
        lowered = lower_collective(
            "allgather", ep_ranks, topo, float(ep_size * dtype_bytes),
            communicator_id=f"{name}:counts_ag",
        )
    except ModelingError:
        return {}, "fallback"
    engine = execution.engine
    missing_res = [
        res for res in topo.resources()
        if res.name not in engine.resources and any(
            res.name in tr.resources
            for ph in lowered.phases for tr in ph.transfers
        )
    ]
    if missing_res:
        engine.extend(missing_res)
    ready_map = {rank: barrier_entry for rank in ep_ranks}
    done_map = submit_lowered_collective(
        engine, f"{name}:eager_split:counts_ag", lowered, ready_map, owner=owner
    )
    return {int(d.rank): done_map[int(d.rank)] for d in stage_dies if int(d.rank) in done_map}, "lowered_collective"


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
    hist_entry = engine.submit(
        f"{name}:eager_split:histogram_prep", after=(barrier_entry,), owner=owner
    )
    ag_by_rank, ag_mode = _submit_eager_counts_allgather(
        execution, op, replica, stage_dies,
        ep_size=ep_size, dtype_bytes=dtype_bytes,
        name=name, barrier_entry=hist_entry, ids=ids, owner=owner,
    )
    indep_events = (
        tuple(ev for op_id, ev in ids.items() if op_id not in op.deps)
        if spec.requires_explicit_device_sync
        else ()
    )
    results = []
    all_d2h_resources: list[str] = []
    for die in stage_dies:
        die_rank = int(die.rank)
        res = EagerSplitBarrier.submit(
            engine,
            machine,
            die,
            spec,
            barrier_entry,
            name=f"{name}:eager_split:d{die_rank}",
            namespace=execution.namespace,
            allgather_event=ag_by_rank.get(die_rank),
            independent_device_events=indep_events,
            owner=owner,
        )
        results.append(res)
        deps.append(res.dispatch_ready)
        for rname in res.d2h_resources:
            if rname not in all_d2h_resources:
                all_d2h_resources.append(rname)
    total_in = sum(r.input_splits_bytes for r in results)
    total_out = sum(r.output_splits_bytes for r in results)
    execution.eager_split_barriers.append({
        "op_id": op.op_id,
        "replica": replica,
        "stage": op.stage,
        "ep_size": ep_size,
        "num_local_experts": num_local_experts,
        "participating_ranks": [int(d.rank) for d in stage_dies],
        "input_splits_bytes": results[0].input_splits_bytes,
        "output_splits_bytes": results[0].output_splits_bytes,
        "total_input_splits_bytes": total_in,
        "total_output_splits_bytes": total_out,
        "total_d2h_bytes": total_in + total_out,
        "counts_allgather_mode": ag_mode,
        "explicit_device_sync": any(r.explicit_device_sync is not None for r in results),
        "d2h_resources": all_d2h_resources,
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


from collections.abc import Mapping
from dataclasses import replace
from functools import lru_cache, partial
from typing import Any

from mindie_sim.schema.case import Case as CaseSpec
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import IterationWork
from ..execution.cohort_traffic import bind_cohort_traffic
from ..collectives.topology import Topology
from ..efficiency.cost import CostModel
from ..engine import Engine, Event, Resource
from ..execution.state import memory_bindings_for
from ..mechanism import Geometry
from ..program import RecipeBook
from ..runtime.l2_residency import AdmissionMemoryView, primitive_working_set_sweep
from .serving_io import ServingIO
from ..efficiency.step_metrics import StepSummary
from .cohort import CollectiveParticipant, _CohortMixin
from .context import (
    INLINE_FRONTIER_TASK_CAP,
    StageProfile,
    _ContextMixin,
    build_stage_profiles,
)
from .lazy_submit import _LazySubmitMixin
from .streams import (
    META_CACHE_LIMIT,
    _NO_REV_OPS_CACHE,
    ResourceRates,
    _LedgerBoundCost,
    _isolated_script_key,
    _ledger_recorder,
    _rank_token,
    mark_no_rev_collective,
    meta_scan_flags,
)
from .submission import (
    ScheduledIteration,
    SubmissionPolicy,
    is_zero_kernel_view,
)

HOST_LOOKAHEAD_DEFAULT = 8
FRAGMENT_CACHE_LIMIT_DEFAULT = 64


@lru_cache
def _sku_working_set_sweep(sku):
    from mindie_sim.primitives.loader import load_manifest, load_primitives

    for key, entry in load_manifest()["tables"].items():
        if sku in (key, entry.get("sku"), *entry.get("compatible_skus", ())):
            return primitive_working_set_sweep(load_primitives(sku))
    return {}


class Execution(_ContextMixin, _CohortMixin, _LazySubmitMixin):
    schedule_replay = False
    _isolated_scripts: dict = {}

    def __init__(
        self, case: CaseSpec, cost: CostModel, *, recipes: RecipeBook | None = None,
        submission: SubmissionPolicy | None = None, trace_limit: int = 0,
        engine: Engine | None = None, namespace: str = "",
        include_submission: bool = True, on_operator=None,
        include_serving_io: bool = False,
        client_tails=None, memory_policy=None,
        legacy_diagnostics: bool | None = None,
        fragment_state=None, memory=None,
        lazy_frontier: bool = True, host_lookahead: int = HOST_LOOKAHEAD_DEFAULT,
        bindings=None, ledger=None, isolated_replica: int | None = None,
        evaluation=None, accel_eager_attach: bool = False,
        stream_plans=None,
    ):
        from ..efficiency.cost import CostLedger
        from ..program.invocation import InvocationPolicy
        from ..offload import resolve_case_offload_policy
        from ..graph_analysis import normalise as _normalise_profile
        from ..service_regime_domains.regimes import RegimePolicy
        from mindie_sim.observations.resource_services import normalize_resource_service

        self.case, self.bindings = case, bindings
        self.evaluation = evaluation
        self._shared_eval_cache = (
            None if evaluation is None else evaluation.prepared_programs)
        self.ledger = ledger if ledger is not None else CostLedger()
        self.cost_model = cost
        self.cost = _LedgerBoundCost(cost, self.ledger)
        self.namespace = namespace
        self.memory = memory
        self._active_requests: tuple = ()
        self._active_padded_tokens: int = 0
        self._active_layout: str = ""
        metadata = getattr(case.runtime, "metadata", {}) or {}
        if not isinstance(metadata, Mapping):
            metadata = dict(metadata)
        if legacy_diagnostics is None:
            legacy_diagnostics = bool(
                metadata.get("legacy_diagnostics", False)
                or getattr(cost, "primitives", None) is not None)
        self.legacy_diagnostics = bool(legacy_diagnostics)
        self.include_submission = include_submission
        self._raw_eager_mode = _is_raw_eager_runtime(case.runtime)
        self.on_operator = on_operator
        self.lazy_frontier = bool(lazy_frontier)
        self.host_lookahead = max(1, int(host_lookahead))
        self.accel_eager_attach = bool(accel_eager_attach)
        self.recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
        self.invocations = InvocationPolicy(case.runtime.kernels.get("invocations", ()))
        self.regimes = RegimePolicy(case.runtime.kernels.get("service_regimes", ()))
        self.submission = submission or SubmissionPolicy(**case.runtime.submission)
        self.submission_source = "api.submission" if submission is not None else "runtime.submission"
        self.serving_io = ServingIO(case.runtime.serving_io if include_serving_io else {},
                                    namespace, case.runtime.parallel.dp, client_tails)
        self.offload_policy = resolve_case_offload_policy(case, memory_policy)
        self._offload_memory = None
        self.step_summary = StepSummary()
        self.stage_names: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_pools: dict[tuple[int, int], dict[str, str]] = {}
        self.stage_dies: dict[tuple[int, int], tuple[Any, ...]] = {}
        self.streams: dict[tuple[int, int, str], Event] = {}
        self.hosts: dict[tuple[int, int], Event] = {}
        self.submission_counts = {"eager_invocations": 0, "graph_replays": 0}
        self.prepared = {}
        self.schedules = {}

        resources, pools, pool_units = self._build_engine_resources(
            case, bindings, namespace, isolated_replica, stream_plans)
        self._bind_engine(engine, trace_limit, resources, pools, pool_units)

        attach_stars_queues(self)

        self.resource_rates = ResourceRates(self.engine.resources)
        self.stage_geometry = {
            key: replace(
                Geometry.from_devices(
                    dies, links=case.machine.links,
                    transfer_channels=case.machine.transfer_channels,
                ),
                datapath=self.stage_profiles[key].datapath,
            )
            for key, dies in self.stage_dies.items()
        }
        self._stage_geom_keys: dict = {}
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self.work = dict.fromkeys(("flops", "vector_ops", "hbm_bytes", "link_bytes"), 0.0)
        self.conflicts = 0
        self.recipes_used: dict[str, int] = {}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins: dict = {}
        self.native_group_launches: dict = {}
        self._program_cache: dict = {}
        self._prepare_cache: dict = {}
        self._lazy_op_cache: dict = {}
        self._lazy_op_id_cache: dict = {}
        self._lazy_stage_id_caches: dict = {}
        self._lazy_safe_id_cache: dict = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None
        self._compact_step: bool = False
        self._last_invocation_record = None
        self._service_identity_cache: dict = {}
        self._op_key_cache: dict = {}
        self._transport_options: dict = {}
        self.replay_counts = {"record": 0, "shadow": 0, "replay": 0, "real": 0}
        self._trace_key = None
        self._trace_shadows: list = []
        self.readiness_report = metadata.get("collective_readiness")
        self.readiness_observations: list = []
        self.readiness_unresolved: list = []
        self._joint_collective_limit = max(0, int(trace_limit))
        self._joint_collective_trace = {} if self._joint_collective_limit else None
        self.joint_collective_stats = (
            {"observations": 0, "retained": 0,
             "max_ready_spread_ms": 0.0, "max_rendezvous_delay_ms": 0.0,
             "critical_rank": None}
            if self._joint_collective_trace is not None else None)
        self._joint_collective_ordinals: dict = {}
        self.participant_profile = _normalise_profile(
            metadata.get("collective_participant_profile"))
        self.fragment_conductors: dict = {}
        self.engine.ensure_state_ledger(fragment_state)
        self.state_accesses = None
        self.scheduled_accesses = None
        self.resource_services = {}
        profile_svc = metadata.get("resource_service_profile")
        for record in (dict(profile_svc or {}).get("services", ())
                       if isinstance(profile_svc, Mapping) else ()):
            service = normalize_resource_service(record)
            self.resource_services[service.service_id] = service

    def _build_engine_resources(
        self, case, bindings, namespace, isolated_replica, stream_plans,
    ):
        from ..offload import offload_resources

        resources, pools, pool_units = [], {}, {}
        resources.extend(self.serving_io.resources())
        resources.extend(offload_resources(
            self.offload_policy, namespace, case.runtime.parallel.dp, case.runtime.parallel.pp))
        p = case.runtime.parallel
        selected_replica = None if isolated_replica is None else int(isolated_replica)
        self.stage_profiles: dict[tuple[int, int], StageProfile] = {}
        profiles = None
        if (bindings is not None and not namespace
                and getattr(bindings, "stage_profiles", ())):
            profiles = tuple(
                profile for profile in bindings.stage_profiles
                if selected_replica is None or profile.replica == selected_replica)
        if not profiles:
            profiles = build_stage_profiles(
                case, namespace,
                replicas=(None if selected_replica is None else (selected_replica,)),
                primitive_table=getattr(bindings, "primitive_table", None))
        if selected_replica is not None and not profiles:
            raise ModelingError("isolated_replica_out_of_range")

        active_ranks: set[int] = set()
        scheduler_replicas: set[int] = set()
        seen_host_rc: set[str] = set()
        for profile in profiles:
            replica, stage = profile.replica, profile.stage
            if replica not in scheduler_replicas:
                resources.append(Resource(
                    f"{namespace}scheduler@r{replica}", 1, "ms", "fifo"))
                scheduler_replicas.add(replica)
            self.stage_dies[replica, stage] = profile.dies
            self.stage_names[replica, stage] = profile.names
            self.stage_pools[replica, stage] = profile.stage_pools
            self.stage_profiles[replica, stage] = profile
            resources.extend(profile.resources)
            for rc_res in getattr(profile, "host_rc_resources", ()):
                if rc_res.name not in seen_host_rc:
                    seen_host_rc.add(rc_res.name)
                    resources.append(rc_res)
            pools.update(profile.pools)
            pool_units.update(profile.pool_units)
            active_ranks.update(int(die.rank) for die in profile.dies)

        self.stream_owners, self.stream_quota_by_res, stream_res = bind_stream_plans(
            stream_plans, self.stage_profiles, self.namespace)
        resources.extend(stream_res)
        self.topology = (
            bindings.topology
            if bindings is not None and bindings.topology is not None and not namespace
            else Topology(
                case.machine, p, namespace,
                collective_startup_mode=case.runtime.collective_startup_mode,
                primitive_table=getattr(bindings, "primitive_table", None),
            ))
        if selected_replica is None:
            resources.extend(self.topology.resources())
        else:
            resources.extend(self.topology.resources_for_ranks(active_ranks))

        return resources, pools, pool_units

    def _bind_engine(self, engine, trace_limit, resources, pools, pool_units):
        self._new_engine = partial(Engine, tuple(resources), pools=dict(pools),
                                   pool_units=dict(pool_units), trace_limit=int(trace_limit),
                                   accel_eager_attach=self.accel_eager_attach)
        if engine is None and self.memory is not None:
            bound = getattr(self.memory, "runtime_state", None)
            if isinstance(bound, Engine):
                engine = bound
        if engine is not None:
            engine.extend(resources, pools, pool_units)
        self._attach_engine(engine if engine is not None else self._new_engine(), self.memory)

    def _attach_engine(self, engine, memory):
        if memory is not None and not engine.bind_memory(memory, pools=("hbm",)):
            raise ModelingError("execution_memory_ledger_conflict")
        view = engine.admission_memory_view
        if view is None:
            view = self._build_admission_view()
            engine.attach_admission_memory_view(view)
        self.engine, self.memory = engine, memory
        self.admission_memory_view = view
        self.allocation_versions = engine._allocation_versions

    def _build_admission_view(self) -> AdmissionMemoryView:
        from mindie_sim.primitives.loader import is_sku_compatible_with_table

        dies = self.case.machine.dies
        capacities = {int(die.rank): int(dict(die.buffers).get("l2", 0)) for die in dies}
        access = {int(die.rank): die.memory_access for die in dies}
        sweeps = {int(die.rank): _sku_working_set_sweep(die.sku) for die in dies if die.sku}
        for profile in self.stage_profiles.values():
            sweep = primitive_working_set_sweep(profile.primitive_table)
            if sweep:
                sweeps.update((int(die.rank), sweep) for die in profile.dies
                              if is_sku_compatible_with_table(profile.primitive_table, die.sku))
        first_rank = int(dies[0].rank)
        capacities.setdefault(0, capacities[first_rank])
        access.setdefault(0, access[first_rank])
        sweeps.setdefault(0, sweeps.get(first_rank, {}))
        return AdmissionMemoryView(capacities, memory_access_by_die=access, ws_sweep_by_die=sweeps)

    def reset_engine(self) -> None:
        """Start the next isolated price with cold memory and fresh bindings."""
        from ..capacity import Memory
        from ..efficiency.cost import CostLedger

        memory = None if self.memory is None else Memory(self.memory.case, self.memory.policy)
        ledger = CostLedger()
        cost = _LedgerBoundCost(self.cost_model, ledger)
        engine = self._new_engine()
        engine.ensure_state_ledger(None)
        self._attach_engine(engine, memory)
        self.ledger, self.cost = ledger, cost
        self._offload_memory = None
        self.recipes_used = {}
        for cache in (self.streams, self.hosts, self.prepared, self.schedules,
                      self._prepare_cache, self._program_cache, self._lazy_op_cache,
                      self._lazy_stage_id_caches, self.fragment_conductors):
            cache.clear()
        self._shared_eval_cache = None
        attach_stars_queues(self)
        self.resource_rates = ResourceRates(self.engine.resources)
        self.stage_contexts = {key: self._stage_context(key, dies)
                               for key, dies in self.stage_dies.items()}
        self._collectives = None
        self._cohort_views = None
        self.native_group_joins = {}
        self.native_group_launches = {}
        self._service_identity_cache = {}
        self._prev_lazy_dag = None
        self._prev_lazy_entries = None
        self._prev_add_topology = None

    def _op_key(self, op):
        key = self._op_key_cache.get(id(op))
        if key is not None and key[0] is op:
            return key[1]
        from ..program.physical_key import operator_metadata_key
        value = operator_metadata_key(op.metadata, op.stage, op.layer)
        if len(self._op_key_cache) >= META_CACHE_LIMIT:
            self._op_key_cache.clear()
        self._op_key_cache[id(op)] = (op, value)
        return value

    def ledger_recorder(self, method):
        """Bind one CostModel recorder to this execution's private ledger."""
        return _ledger_recorder(method, self.ledger)

    @property
    def fragment_state(self):
        """Live view of the one typed state ledger owned by this runtime."""
        return self.engine.ensure_state_ledger()

    def submit_catalog(self, catalog, binding, *, name: str, after=(),
                       owner: str = ""):
        """Submit one immutable catalog on this Execution's RuntimeState."""
        self.engine.ensure_state_ledger()
        return self.engine.submit_catalog(
            catalog, binding, name=name, after=tuple(after), owner=owner)

    def bind_memory(self, memory, *, replica: int = 0, stage: int = 0,
                    pools=("hbm",)):
        """Idempotently bind this Execution's real T08 ledger."""
        if memory is None:
            return self.memory
        if self.memory is not None and self.memory is not memory:
            return None
        if not self.engine.bind_memory(memory, pools=pools):
            return None
        self.memory = memory
        return memory

    def fragment_conductor(
        self, *, replica: int = 0, stage: int = 0,
        resource_bindings=None, pool_bindings=None, unit_conversions=None,
        state=None, detail: bool = False,
        cache_limit: int = FRAGMENT_CACHE_LIMIT_DEFAULT,
        repeat_limit: int = 1_000_000, prefix: str | None = None,
        memory=None, memory_pools=("hbm",), memory_bindings=None,
        memory_owner=None, memory_storage=None, memory_ranges=None,
        memory_generation=None, memory_event=None, memory_host_bytes: float = 0.0,
        memory_unit: str | None = None,
    ):
        """One shared-context conductor for typed model/framework fragments."""
        from ..scheduling import FragmentConductor
        names = dict(self.stage_names.get((replica, stage), {}))
        names.update(dict(resource_bindings or {}))
        pools = dict(self.stage_pools.get((replica, stage), {}))
        pools.update(dict(pool_bindings or {}))
        if state is None:
            state = self.fragment_state
        ledger = memory if memory is not None else self.memory
        physical_memory_pools = tuple(
            pools.get(str(item), str(item)) for item in memory_pools)
        return FragmentConductor(
            self.engine, resource_bindings=names, pool_bindings=pools,
            unit_conversions=unit_conversions, state=state, detail=detail,
            memory=ledger, memory_replica=replica, memory_stage=stage,
            memory_pools=physical_memory_pools, memory_bindings=memory_bindings,
            memory_owner=memory_owner, memory_storage=memory_storage,
            memory_ranges=memory_ranges, memory_generation=memory_generation,
            memory_event=memory_event, memory_host_bytes=memory_host_bytes,
            memory_unit=memory_unit,
            cache_limit=cache_limit, repeat_limit=repeat_limit,
            prefix=prefix or f"fragment@{replica}:s{stage}:",
        )

    def submit_fragment(self, fragment, *, replica: int = 0, stage: int = 0,
                        **submit_kwargs):
        """Submit into the cached shared-context conductor for a stage."""
        key = (replica, stage)
        conductor = self.fragment_conductors.get(key)
        if conductor is None:
            conductor = self.fragment_conductor(replica=replica, stage=stage)
            self.fragment_conductors[key] = conductor
        return conductor.submit(fragment, **submit_kwargs)

    def stage_submit_boundary(self, replica: int = 0, stage: int = 0,
                              name: str = "host_submit", *, step=None):
        """Start event of a real host launch, or ``None``."""
        if step is not None:
            matches = [event for rep, stg, event in step.host_submits
                       if rep == replica and stg == stage]
            return matches[-1] if matches else None
        launch = self.hosts.get((replica, stage))
        if launch is None:
            return None
        return self.engine.start_boundary(launch, name)

    def _start_program(self, context, recipe, name, ready, *, owner,
                       local_ready=None, peer_ready=None, rank_index=None,
                       launch=None, source_send=None,
                       send_placeholders=None, replica=None,
                       memory_ledger=None, memory_pools=None,
                       memory_bindings=None, on_resolve=None, begin_after=None):
        """Keep per-core programs resident only while their operator can run."""
        engine = self.engine
        entry = {"ready": ready}
        inherited = local_ready if local_ready is not None else ready
        stage = int(context.op.stage)

        if (begin_after is None and on_resolve is None and memory_ledger is None
                and not memory_bindings and not Execution.schedule_replay
                and len(engine._tasks) < INLINE_FRONTIER_TASK_CAP
                and not self._program_uses_fifo(context, engine)):
            op = context.op
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            if not replayable:
                (program, mapped_recs, svc_calls), _from_cache = (
                    self._lower_operator_program(
                        context, recipe, engine, replica, name, prog_key, do_compact))
                from ..program import is_pure_static_program
                if is_pure_static_program(program):
                    if mapped_recs and context.record_mapping is not None:
                        for mrec in mapped_recs:
                            context.record_mapping(mrec)
                    if svc_calls and context.record_service is not None:
                        for sargs, skwargs in svc_calls:
                            context.record_service(*sargs, **skwargs)
                    native_resources = self._native_resource_names(context, program)
                    self._record_single_native(
                        context, program, replica, name, transport=transport)
                    with engine.memory_scope(
                            memory_ledger,
                            replica=(0 if replica is None else replica),
                            stage=stage,
                            pools=(memory_pools if memory_pools is not None else ("hbm",)),
                            bindings=memory_bindings):
                        return self._start_engine_program(
                            program, engine, name, inherited, owner,
                            local_ready=local_ready, peer_ready=peer_ready,
                            rank_index=rank_index, launch=launch,
                            source_send=source_send,
                            send_placeholders=send_placeholders,
                            resource_names=native_resources,
                            transport=transport)

        complete = engine.future(name + ":program_complete")

        def begin(_):
            op = context.op
            cur_ready = entry["ready"]
            transport, prog_key, do_compact, replayable = self._program_inputs(
                context, recipe, engine, replica, op, owner)
            token = _rank_token(context)
            if replayable:
                script_key = _isolated_script_key(prog_key, context, token)
                if (engine._trace_prefix is not None and engine._trace_private
                        and engine._trace_clean and engine._trace_token != token
                        and not self._outside_trace(engine)):
                    self._trace_shadows.append((token, complete))
                    self.replay_counts["shadow"] += 1
                    return
                if engine._trace_prefix is None and engine.replay_ready():
                    engine.begin_trace(name, token)
                    self._trace_key = script_key
                    self._trace_shadows = []
                    self.replay_counts["record"] += 1
                else:
                    self.replay_counts["real"] += 1
            else:
                self.replay_counts["real"] += 1

            recording = getattr(engine, "_trace_prefix", None) == name

            def finished(_event=None):
                if recording and getattr(engine, "_trace_prefix", None) == name:
                    self._seal_trace(engine, complete)
                else:
                    engine.resolve(complete)
                if on_resolve is not None:
                    on_resolve()

            (program, mapped_recs, svc_calls), _from_cache = (
                self._lower_operator_program(
                    context, recipe, engine, replica, name, prog_key, do_compact))
            if mapped_recs and context.record_mapping is not None:
                for mrec in mapped_recs:
                    context.record_mapping(mrec)
            if svc_calls and context.record_service is not None:
                for sargs, skwargs in svc_calls:
                    context.record_service(*sargs, **skwargs)
            native_resources = self._native_resource_names(context, program)
            self._record_single_native(
                context, program, replica, name, transport=transport)
            finish = self._start_engine_program(
                program, engine, name, cur_ready, owner,
                local_ready=local_ready, peer_ready=peer_ready,
                rank_index=rank_index, launch=launch,
                source_send=source_send,
                send_placeholders=send_placeholders,
                resource_names=native_resources, transport=transport)
            engine.on_complete(finish, finished, native_hook=True)

        def begin_with_binding(event):
            with engine.memory_scope(
                    memory_ledger, replica=(0 if replica is None else replica),
                    stage=stage,
                    pools=(memory_pools if memory_pools is not None else ("hbm",)),
                    bindings=memory_bindings):
                begin(event)

        if begin_after is None:
            engine.submit(name + ":begin", after=(inherited,), owner=owner,
                          on_complete=begin_with_binding, native_hook=True)
        else:
            boundary = engine.submit(
                name + ":begin", after=tuple(begin_after), owner=owner,
                on_complete=begin_with_binding, native_hook=True)
            entry["ready"] = boundary
        return complete

    def add(self, work: IterationWork, replica: int, release_ms: float, *,
            external_inputs=None, emitted_tokens=None,
            memory_bindings=None, state_accesses=None) -> ScheduledIteration:
        work = self._prepare_iteration_work(work, replica)
        engine = self.engine
        prefix = f"{self.namespace}r{replica}:i{work.plan.iteration}"
        ids: dict[str, Event] = dict(external_inputs or {})
        declared_commits: dict[str, Event] = {}
        host_submits: list[tuple[int, int, Event]] = []
        scope = work.replay_scope or ("full" if work.graph_bucket is not None else "none")
        if self.include_submission and self.case.runtime.async_scheduling and replica in self.prepared:
            schedule = self.prepared.pop(replica)
        else:
            scheduler_ms = self.submission.scheduler_ms if self.include_submission else 0
            schedule = engine.submit(prefix + ":schedule", {f"{self.namespace}scheduler@r{replica}": scheduler_ms},
                                     release=release_ms, owner=f"replica:{replica}")
        self.schedules[replica] = schedule
        prepared, input_ready = self.serving_io.input(self, prefix, replica, schedule, work)
        physical_ops = self.recipes.expand(work.ops)
        external = dict(external_inputs or {})

        lazy_result = try_lazy_add(
            self, work, replica, release_ms, scope=scope, schedule=schedule,
            prepared=prepared, input_ready=input_ready, physical_ops=physical_ops,
            prefix=prefix, external=external, emitted_tokens=emitted_tokens,
            memory_bindings=memory_bindings, state_accesses=state_accesses)
        if lazy_result is not None:
            return lazy_result

        from ..program.kernel_groups import KernelGroups, declaration
        from ..sharding.local_tp import collective_requires_cohort
        graph_launch: dict[object, tuple[Event, Event]] = {}
        layer_entries: dict[int, Event] = {}
        expert_layer_events: dict[tuple[int, int], Event | None] = {}
        last_expert_done: dict[int, Event] = {}
        kv_layer_events: dict[tuple[int, int], list[Event]] = {}
        last_kv_done: dict[int, Event] = {}
        kv_writeback_layers: set[tuple[int, int]] = set()
        kv_writeback_events: list[Event] = []
        all_bound_accesses: list = []
        owned_offload: list[Event] = []
        pending_draft: dict[int, list] = {}
        last_draft_done: dict[int, Event] = {}
        kernel_groups = KernelGroups(self, physical_ops, replica, prefix)
        step_replay_ids: dict[tuple[int, int], int] = {}

        for op in physical_ops:
            original = op
            op, recipe = self.prepare(op, replica)
            stage_key = replica, op.stage
            name = prefix + ":" + op.op_id
            names = self.stage_names[stage_key]
            deps = [input_ready, *(ids[key] for key in op.deps)]
            bound_accesses = schedule_op_offload_reads(
                self, op, work, replica, prefix, name, input_ready, deps, ids,
                layer_entries, expert_layer_events, last_expert_done,
                kv_layer_events, last_kv_done, state_accesses, all_bound_accesses)
            deps.extend(kernel_groups.entry_dependencies(op, ids))
            stream_key = replica, op.stage, op.stream or "main"
            stream_tail = kernel_groups.stream_tail(op, stream_key)
            if stream_tail is not None:
                deps.append(stream_tail)
            context = self._context(op, replica, scope, recipe=recipe)
            context = replace(context, group_buffers=kernel_groups.buffer_scope(op))
            owner = f"replica:{replica}"
            captured = scope == "full" or (
                scope == "piecewise" and bool(dict(op.metadata).get("graph_captured")))
            capture_key = (
                op.stage, dict(op.metadata).get("graph_segment", 0) if scope == "piecewise" else 0)
            kernel_group = declaration(op)
            submit = (bool(kernel_group.get("entry")) if kernel_group else
                      bool(context.work or getattr(recipe, "algorithm", None)))
            host_launch = None
            if submit and self.include_submission:
                host_launch = self._submit_eager_host_launch(
                    engine, stage_key, name, names, owner, prepared, deps,
                    captured, capture_key, scope, graph_launch, host_submits)
                deps.append(host_launch)

            stars_issued = self._issue_eager_descriptors(
                engine, op, stage_key, owner, submit, captured, capture_key,
                host_launch, prepared, deps, step_replay_ids)

            barrier_entry = schedule_eager_moe_split_barrier(
                self, op, replica, scope, name, deps, ids
            )
            ready = engine.submit(name + ":ready", after=tuple(deps) or (schedule,), owner=owner)
            if (self.readiness_report
                    and op.family in {"collective", "allgather", "reduce_scatter",
                                      "allreduce", "transport"}
                    and op.link_bytes > 0
                    and dict(op.metadata).get("group_axis") not in {"ep", "local_tp"}):
                observe_collective_readiness(self, op, replica, ready)
            if op.layer >= 0 and op.layer not in layer_entries:
                layer_entries[op.layer] = ready
            ready = kernel_groups.phase_ready(op, ready, name)
            recipe_name = op.recipe or type(recipe).__name__
            self.recipes_used[recipe_name] = self.recipes_used.get(recipe_name, 0) + 1
            cross_dp = collective_requires_cohort(op, self.topology)
            if cross_dp and getattr(recipe, "algorithm", None) and self._collectives is None:
                raise ModelingError("joint_collective_requires_complete_cohort")
            if self._collectives is not None and cross_dp:
                complete = engine.future(name + ":collective")
                op_meta = dict(op.metadata)
                grp_ranks = op_meta.get("group_ranks")
                coll_key = (
                    f"{op.op_id}@{','.join(map(str, grp_ranks))}"
                    if grp_ranks and op_meta.get("group_axis") == "local_tp"
                    else op.op_id)
                self._collectives.setdefault(coll_key, []).append(CollectiveParticipant(
                    replica, ready, context, recipe, name, complete))
            else:
                physical_hbm = self.stage_pools.get(stage_key, {}).get("hbm", "hbm")
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(op, memory_bindings))
            complete = kernel_groups.finish(op, context, recipe, complete, stream_key)
            if stars_issued:
                group = kernel_groups.group(op)
                physical_done = group.completion if group is not None else complete
                for s_queue, s_rec, s_alloc in stars_issued:
                    s_queue.bind_kernel_execution(
                        engine, s_rec, s_alloc,
                        kernel_start=ready, kernel_done=physical_done, owner=owner)
            ids[op.op_id] = complete
            schedule_op_offload_writes(
                self, op, work, replica, prefix, complete, bound_accesses,
                last_expert_done, last_kv_done, kv_writeback_events,
                owned_offload, last_draft_done, pending_draft)
            if self.on_operator is not None:
                op_start = barrier_entry if barrier_entry is not None else ready
                self.on_operator(original, replica, "JointNativeService" if context.joint_native_service
                                 is not None else type(recipe).__name__, op_start, complete)
            if op.family == "token_commit":
                declared_commits.update((request, complete) for request in op.requests)
            for key in self.work:
                self.work[key] += float(getattr(op, key))

        schedule_step_offload_tails(
            self, work, replica, prefix, input_ready, physical_ops, ids,
            pending_draft, last_draft_done, all_bound_accesses, kv_layer_events,
            kv_writeback_layers, kv_writeback_events, owned_offload)
        return finalize_step(
            self, work, replica, release_ms, prefix=prefix, scope=scope,
            input_ready=input_ready, physical_ops=physical_ops, ids=ids,
            declared_commits=declared_commits, host_submits=host_submits,
            kv_layer_events=kv_layer_events,
            kv_writeback_events=kv_writeback_events,
            emitted_tokens=emitted_tokens)

    def _prepare_iteration_work(self, work: IterationWork, replica: int) -> IterationWork:
        if self._collectives is None:
            if _NO_REV_OPS_CACHE.get(id(work.ops)) is not work.ops:
                has_rev = any(
                    op.metadata and meta_scan_flags(op.metadata)[1]
                    for op in work.ops)
                if has_rev:
                    work = replace(work, ops=bind_cohort_traffic(
                        self.case, {replica: work.ops})[replica])
                else:
                    mark_no_rev_collective(work.ops)
        work = self._lower_local_tp_work(work, replica)
        if (getattr(work, "recipe", None) != "source"
                and getattr(work, "plan", None) is not None
                and getattr(work.plan, "requests", ())):
            self._active_requests = tuple(work.plan.requests)
        if getattr(work, "padded_tokens", 0):
            self._active_padded_tokens = int(work.padded_tokens)
        return work

    def _stage_launch_ms(self, stage_key: tuple[int, int], *, captured: bool = False) -> float:
        dies = self.stage_dies[stage_key]
        duration = self.submission.launch_ms(dies, captured=captured)
        if (duration <= 0.0 and not captured and self._raw_eager_mode
                and self.submission.kernel_launch_ms is None):
            profile = self.stage_profiles.get(stage_key)
            p15_ms = _calibrated_host_submit_ms(
                getattr(profile, "primitive_table", None) if profile is not None else None
            )
            if p15_ms > 0.0:
                return p15_ms
        return duration

    def _submit_eager_host_launch(
        self, engine, stage_key, name, names, owner, prepared, deps,
        captured, capture_key, scope, graph_launch, host_submits,
    ):
        if captured and capture_key in graph_launch:
            host_launch, _start = graph_launch[capture_key]
        else:
            duration = self._stage_launch_ms(stage_key, captured=captured)
            host_deps = (prepared,) + ((self.hosts[stage_key],) if stage_key in self.hosts else ())
            if captured and scope == "piecewise" and self.submission.piecewise_sync_ms > 0.0:
                if deps:
                    host_deps = tuple(deps) + host_deps
                if self.submission_counts["graph_replays"] > 0:
                    duration += float(self.submission.piecewise_sync_ms)
            host_launch = engine.submit(name + ":submit", {names["host"]: duration},
                                        after=host_deps, owner=owner)
            self.hosts[stage_key] = host_launch
            self.submission_counts["graph_replays" if captured else "eager_invocations"] += 1
            start = engine.start_boundary(host_launch, name + ":host_submit")
            if captured:
                graph_launch[capture_key] = (host_launch, start)
            host_submits.append((*stage_key, start))
        return host_launch

    def _issue_eager_descriptors(
        self, engine, op, stage_key, owner, submit, captured, capture_key,
        host_launch, prepared, deps, step_replay_ids,
    ):
        stage_stars = self.stars_queues.get(stage_key, ())
        stars_issued: list[tuple[Any, Any, Any]] = []
        if stage_stars and submit and not is_zero_kernel_view(op):
            replay_slot = (
                (stage_key, "graph", capture_key)
                if captured else (stage_key, "eager", op.op_id)
            )
            if replay_slot not in step_replay_ids:
                rep_id = self._stars_replay_seq.get(stage_key, 0)
                self._stars_replay_seq[stage_key] = rep_id + 1
                step_replay_ids[replay_slot] = rep_id
            self._stars_occ_seq += 1
            desc_ready = host_launch if host_launch is not None else prepared
            for die_rank, s_queue in stage_stars:
                s_rec, s_alloc = s_queue.issue_descriptor(
                    engine, replay=step_replay_ids[replay_slot],
                    stream=op.stream or "main",
                    kernel_occurrence=self._stars_occ_seq,
                    descriptor_ready=desc_ready, owner=owner, die=die_rank)
                stars_issued.append((s_queue, s_rec, s_alloc))
                deps.append(s_rec.issue_done_event)
        return stars_issued

    def prepare_next(self, replica):
        if self.include_submission and self.case.runtime.async_scheduling:
            self.prepared[replica] = self.engine.submit(
                f"{self.namespace}r{replica}:prepare_next",
                {f"{self.namespace}scheduler@r{replica}": self.submission.scheduler_ms},
                after=(self.schedules[replica],), owner=f"replica:{replica}")

    def submission_summary(self) -> dict:
        return build_submission_summary(self)

    def stars_summary(self) -> dict[str, Any]:
        """Return STARS descriptor/uncovered summaries per stage and die."""
        return build_stars_summary(self)

    def summary(self) -> dict:
        return build_execution_summary(self)
