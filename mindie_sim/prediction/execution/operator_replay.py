"""Isolated device-program replay for operators and transport cohorts.

[Role] Price one operator or one complete replica->operator cohort on a fresh
  execution engine, before any serving policy is applied.
[Invariants] An isolated span is not a concurrent graph span; operator names,
  layers and request identities never change physical cost.  Collective wire is
  attributed to its physical source ranks and must conserve the engine service
  totals exactly.  A homogeneously priced cached row restores the current
  declaration identity without repricing physical work.
[Timeouts & Locks] Bounded by the declared operators in one active graph; no
  I/O, locks, device contexts or measurements.
[Side-Effect Policy] Pure in-memory evaluation of immutable inputs; returns new
  rows and never mutates the caller's operators, recipes or case.
"""
from dataclasses import replace
from collections.abc import Mapping
from hashlib import sha256
import json
import math
from mindie_sim.coverage import model_coverage

from mindie_sim.prediction.execution.cohort_traffic import bind_cohort_traffic
from mindie_sim.prediction.execution import CollectiveParticipant, Execution
from mindie_sim.prediction.structure import parse_case
from mindie_sim.prediction.body_defaults import consume_target_declaration


_WARM_DURATION_EPS_MS = 1e-12


def _make_warm_entry_view(src):
    """Clone ``src`` into a first-touch-warm ``AdmissionMemoryView`` for lower bounds."""
    from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView, _ResidentL2Entry
    from mindie_sim.prediction.runtime.memory_physics import (
        SectorCoverage, _subtract_byte_intervals,
    )

    class _WarmEntryAdmissionView(AdmissionMemoryView):
        def _unseen_coverage(self, pattern, key, coverage, working_set, eff):
            if pattern.direction != "read" or eff <= 0 or coverage.empty:
                return SectorCoverage()
            if not self._hit_fraction(working_set, eff, die=pattern.die):
                return SectorCoverage()
            seen = self._seen_by_key.get((key, pattern.version))
            if seen is None:
                return coverage
            if not coverage.periodic and not seen.periodic:
                u_ivs = (
                    _subtract_byte_intervals(coverage.intervals, seen.intervals)
                    if seen.intervals else coverage.intervals
                )
                u_scale = (
                    _subtract_byte_intervals(coverage.scale_intervals, seen.scale_intervals)
                    if seen.scale_intervals else coverage.scale_intervals
                )
                return SectorCoverage(intervals=u_ivs, scale_intervals=u_scale, _canonical=True)
            return SectorCoverage() if seen == coverage else coverage

        def resident_coverage_for(self, pattern, aligned_bytes=None):
            base = super().resident_coverage_for(pattern, aligned_bytes)
            eff_aligned = pattern.runs.aligned_copy_bytes() if aligned_bytes is None else aligned_bytes
            key = self._key(pattern, eff_aligned)
            if key is not None and pattern.direction == "read":
                eff = self._preview_effective_l2(key)
                coverage, working_set = self._geometry(pattern, eff_aligned)
                unseen = self._unseen_coverage(pattern, key, coverage, working_set, eff)
                if not unseen.empty:
                    return unseen if base.empty else base.union(unseen)
            return base

        def _classify_access_with_overlap(
            self, pattern, aligned_bytes, eff, coverage, working_set, entry,
        ):
            if pattern.direction == "read" and eff > 0:
                key = self._key(pattern, aligned_bytes)
                if key is not None:
                    unseen = self._unseen_coverage(pattern, key, coverage, working_set, eff)
                    if not unseen.empty:
                        if entry is None and unseen == coverage:
                            frac = self._hit_fraction(working_set, eff, die=pattern.die)
                            hit = aligned_bytes if frac == 1.0 else round(aligned_bytes * frac)
                            return hit, aligned_bytes - hit, None
                        merged = unseen if entry is None else entry.coverage_snapshot().union(unseen)
                        synth = _ResidentL2Entry(
                            key=key, working_set_bytes=working_set, version=pattern.version,
                            dirty=False, private_workspace=False,
                            resident_coverage=merged, resident_bytes=merged.measure_bytes(),
                            cache_policy=pattern.cache_policy,
                            resident_scale_intervals=merged.scale_intervals,
                        )
                        return super()._classify_access_with_overlap(
                            pattern, aligned_bytes, eff, coverage, working_set, synth,
                        )
            return super()._classify_access_with_overlap(
                pattern, aligned_bytes, eff, coverage, working_set, entry,
            )

        def _admit_access_with_prior(self, pattern, aligned_bytes):
            key = self._key(pattern, aligned_bytes)
            coverage = SectorCoverage()
            if key is not None and pattern.direction == "read":
                coverage, working_set = self._geometry(pattern, aligned_bytes)
                eff = self.effective_l2_bytes(key.die)
                existing = self._existing(pattern, key, eff, preview=False)
                unseen = self._unseen_coverage(pattern, key, coverage, working_set, eff)
                if existing is not None and not unseen.empty:
                    merged = existing.coverage_snapshot().union(unseen)
                    existing.resident_intervals_mut = None
                    existing.resident_coverage = merged
                    existing.resident_bytes = merged.measure_bytes()
                    existing.resident_scale_intervals = merged.scale_intervals
                    existing.working_set_bytes = max(existing.working_set_bytes, working_set)
                    self._evict_over_capacity(key.die, protect_key=key)
            res = super()._admit_access_with_prior(pattern, aligned_bytes)
            if key is not None and not coverage.empty:
                prev = self._seen_by_key.get((key, pattern.version))
                self._seen_by_key[(key, pattern.version)] = (
                    coverage if prev is None else prev.union(coverage)
                )
            return res

    view = _WarmEntryAdmissionView(
        dict(src._l2_capacity),
        memory_access=src._memory_access_default,
        memory_access_by_die=dict(src._memory_access_by_die),
        granule_bytes=src.granule_bytes,
        arch_id=src.arch_id,
    )
    view._knee_curves_by_die = dict(src._knee_curves_by_die)
    view._granule_by_die = dict(src._granule_by_die)
    view._knee_curve = src._knee_curve
    view._seen_by_key = {}
    return view


def _probe_warm_entry_duration(execution, program, input_ready_ms: float) -> float:
    """Measure first-touch-warm lower-bound duration on a fresh silent probe Engine."""
    cold_view = execution.engine.admission_memory_view
    warm_engine = execution._new_engine()
    warm_view = _make_warm_entry_view(cold_view)
    warm_engine.attach_admission_memory_view(warm_view)
    orig_submit_bound = warm_engine.submit_bound_access

    def _silent_warm_bound_access(
        name, bound, *, after=(), release=None, owner="", priority=0,
        memory_state=None, pattern=None, on_delivery=None,
    ):
        del on_delivery
        target_state = warm_view if memory_state is cold_view else memory_state
        return orig_submit_bound(
            name, bound, after=after, release=release, owner=owner, priority=priority,
            memory_state=target_state, pattern=pattern, on_delivery=None,
        )

    warm_engine.submit_bound_access = _silent_warm_bound_access
    warm_ready = warm_engine.submit("operator_warm:input_ready", release=input_ready_ms)
    warm_complete = program.start(warm_engine, "operator_warm", warm_ready)
    warm_engine.advance(until=(warm_complete,))
    return float(warm_complete.time) - float(input_ready_ms)


def physical_signature(op):
    """Deterministic serialization of the canonical physical key."""
    from mindie_sim.prediction.program.physical_key import physical_signature as _signature
    return _signature(op)


def physical_id(op):
    from mindie_sim.prediction.program.physical_key import physical_id as _identifier
    return _identifier(op)


def requires_collective_cohort(case, op, recipe):
    from mindie_sim.prediction.collectives.waves import startup_waves_recipe
    from mindie_sim.prediction.sharding.local_tp import local_tp_spans_dp

    recipe = startup_waves_recipe(op, recipe, case.runtime.collective_startup_mode)
    axis = dict(op.metadata).get("group_axis")
    return bool(
        getattr(recipe, "algorithm", None)
        and ((case.runtime.parallel.ep_spans_dp and axis == "ep")
             or (local_tp_spans_dp(case.runtime.parallel) and axis == "local_tp")
             or (case.runtime.parallel.cp_spans_dp
                 and axis in {"cp", "dcp_dp", "dcp_a2a"})))


def _service_regime_row(context, duration):
    """Selected response evidence, with response and executed duration distinct."""
    selected = context.service_regime or context.joint_native_service
    if not selected:
        return {}
    fields = ("rule_id", "continuity_group", "execution_mode", "response_basis",
              "coordinate_values", "response_duration_ms", "exterior_policy",
              "reference_program", "reference_primitive_ms")
    row = {key: selected[key] for key in fields if key in selected}
    row["executed_duration_ms"] = duration
    return {"service_regime": row}


def _cost_row(execution, op, recipe, context, *, identity, count, duration,
              input_ready_ms, serviced, early_service):
    engine = execution.engine
    return {
        "physical_id": identity,
        "operator": replace(op, deps=(), requests=()).to_dict(),
        "occurrences": count, "program": ("JointNativeService" if context.joint_native_service
                                          is not None else type(recipe).__name__),
        "declared_service": dict(context.work),
        **({"joint_native_service": dict(context.joint_native_service),
            "analytical_reference_work": dict(context.native_work),
            "work_basis": "native_joint_service; operator_work_is_unexecuted_analytical_reference"}
           if context.joint_native_service is not None else {}),
        **_service_regime_row(context, duration),
        "service_delta": {
            key: serviced.get(key, 0) - context.work.get(key, 0)
            for key in serviced.keys() | context.work.keys()
            if serviced.get(key, 0) != context.work.get(key, 0)},
        "duration_ms": duration, "total_ms": duration * count,
        "input_ready_ms": input_ready_ms, "service_before_input": early_service,
        "serviced": {key: value for key, value in serviced.items() if value},
        "peak_buffers": engine.peak_allocated,
        "peak_buffers_per_unit": engine.summary()["peak_allocated_per_unit"],
        "remaining_buffers": dict(engine.allocated),
        "live_tasks": engine.summary()["live_tasks"],
        "basis": execution.cost.summary(),
    }


def _remap_replica_refs(value, source: int, target: int):
    """Rename one replica's resource scope, recursively, without touching work."""
    if source == target:
        return value
    old, new = f"@r{source}:", f"@r{target}:"
    if isinstance(value, str):
        return value.replace(old, new)
    if isinstance(value, Mapping):
        return {_remap_replica_refs(key, source, target):
                _remap_replica_refs(item, source, target)
                for key, item in value.items()}
    if isinstance(value, list):
        return [_remap_replica_refs(item, source, target) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_replica_refs(item, source, target) for item in value)
    return value


_REPLICA_SCOPED_FIELDS = (
    "declared_service", "serviced", "service_before_input", "service_delta",
    "peak_buffers", "remaining_buffers", "peak_buffers_per_unit",
)


def _remap_replica_scopes(row, source: int, target: int):
    """Rename only resource-keyed row fields; operator metadata is current."""
    if source == target:
        return row
    old, new = f"@r{source}:", f"@r{target}:"
    for key in _REPLICA_SCOPED_FIELDS:
        value = row.get(key)
        if isinstance(value, Mapping):
            row[key] = {str(name).replace(old, new): item
                        for name, item in value.items()}
    if "communication" in row:
        row["communication"] = _remap_replica_refs(
            row["communication"], int(source), int(target))
    return row


def _cached_operator_row(cached, op, identity, replica, count):
    """Restore current declaration identity on a homogeneously priced row."""
    row = dict(cached)
    row["operator"] = replace(op, deps=(), requests=()).to_dict()
    row["physical_id"] = identity
    row = _remap_replica_scopes(row, 0, int(replica))
    basis = row.get("basis")
    old_id = str(dict(cached.get("operator") or {}).get("op_id") or "")
    if isinstance(basis, Mapping) and old_id and old_id != op.op_id:
        occurrences = basis.get("operator_occurrences")
        if isinstance(occurrences, Mapping) and old_id in occurrences:
            occurrences = dict(occurrences)
            occurrences[op.op_id] = occurrences.pop(old_id)
            basis = dict(basis)
            basis["operator_occurrences"] = occurrences
            row["basis"] = basis
    row["occurrences"] = int(count)
    row["total_ms"] = float(cached["duration_ms"]) * int(count)
    return row


def _communication_description(accounting, recipe):
    describe = getattr(recipe, "accounting_description", None)
    return describe(accounting) if describe else accounting.to_dict()


def cohort_operator_costs(case, ops, *, recipes, primitives=None,
                          replay_scope="none", input_ready_ms=0,
                          evaluation=None, stream_plans=None):
    """Isolate one complete transport cohort, attributing wire to source ranks."""
    from mindie_sim.prediction.efficiency.cost import CostLedger
    from mindie_sim.prediction.evaluation import EvaluationContext, context_for
    ctx = (context_for(case, evaluation, recipes) if evaluation is not None
           else EvaluationContext.create(case, recipes, primitives))
    if stream_plans:
        from mindie_sim.prediction.streams import validate_stream_plan_ranks
        stream_plans = validate_stream_plan_ranks(stream_plans, {int(r) for r in ops})
    execution = Execution(
        case, ctx.cost, recipes=recipes,
        bindings=ctx.bindings, ledger=CostLedger(), evaluation=ctx,
        stream_plans=stream_plans)
    execution._compact_step = not execution.legacy_diagnostics
    execution.engine._compact_step = execution._compact_step
    engine, participants, identities, replica_scopes = execution.engine, [], {}, {}
    for replica, original in ops.items():
        identities[replica] = physical_id(original)
        op, recipe = execution.prepare(original, replica)
        scope = replay_scope[replica] if isinstance(replay_scope, Mapping) else replay_scope
        replica_scopes[replica] = scope
        context = execution._context(op, replica, scope, recipe=recipe)
        ready = engine.submit(f"r{replica}:input", release=input_ready_ms)
        participants.append(CollectiveParticipant(
            replica, ready, context, recipe, f"r{replica}:isolated",
            engine.future(f"r{replica}:complete")))
    rank_events = execution._start_joint_collective("isolated", participants)
    engine.advance(horizon=input_ready_ms)
    early_service = {key: value for key, value in engine.serviced.items()
                     if value and "stream_quota@" not in key}
    engine.advance(until=tuple(item.completion for item in participants), first=False)
    rows, attributed = [], {}
    duration = max(item.completion.time for item in participants) - input_ready_ms
    for item in participants:
        context = item.context
        transport = execution.transport_options_for(
            context.op, item.replica, replica_scopes.get(item.replica, "none"))
        accounting = execution.topology.accounting(
            context.op, item.replica, volume=context.native_work.get("link", 0),
            transport=transport, recipe=item.recipe)
        for resource, amount in accounting.wire_by_resource.items():
            attributed[resource] = attributed.get(resource, 0) + amount
        row_context = (
            replace(context, work=dict(accounting.wire_by_resource))
            if (getattr(item.recipe, "algorithm", None) == "oneshot"
                and context.joint_native_service is None)
            else context
        )
        row = _cost_row(
            execution, context.op, item.recipe, row_context,
            identity=identities[item.replica], count=1,
            duration=item.completion.time - input_ready_ms,
            input_ready_ms=input_ready_ms, serviced=accounting.wire_by_resource,
            early_service=early_service)
        rows.append({
            "replica": item.replica, **row,
            "communication": _communication_description(accounting, item.recipe),
            "isolation_boundary": "complete_collective_cohort",
            "cohort_replicas": list(ops), "cohort_duration_ms": duration,
            "service_attribution": "physical_source_ranks",
        })
    actual = {key: value for key, value in engine.serviced.items()
              if value and "stream_quota@" not in key}
    if any(not math.isclose(attributed.get(key, 0), actual.get(key, 0),
                            rel_tol=1e-12, abs_tol=1e-9)
           for key in attributed.keys() | actual.keys()):
        raise ValueError("collective_cohort_source_wire_accounting_mismatch")
    return {
        "boundary": "complete_collective_cohort",
        "duration_ms": duration, "replicas": list(ops),
        "rank_completion_ms": {str(rank): event.time for rank, event in rank_events.items()},
        "serviced": actual, "operators": rows,
    }


def _cohort_costs(case, graphs, recipes, primitives, replay_scope, input_ready_ms,
                  evaluation=None, stream_plans=None):
    """Isolate joint transports once, retaining each replica's local programs."""
    from mindie_sim.prediction.execution.dispatch import cohorts

    if stream_plans:
        from mindie_sim.prediction.streams import validate_stream_plan_ranks
        stream_plans = validate_stream_plan_ranks(stream_plans, {int(r) for r in graphs})
    group_for = {
        rank: group
        for group in cohorts(case.runtime.parallel, case.machine)
        for rank in group
    }
    expanded = {rank: tuple(recipes.expand(items)) for rank, items in graphs.items()}
    expanded = bind_cohort_traffic(case, expanded)
    scopes, joins, joint_nodes, isolation = {}, {}, set(), []
    for rank, items in expanded.items():
        if rank not in group_for:
            raise ValueError("operator_replica_outside_declared_placement")
        scopes[rank] = _replay_scope(items, replay_scope)
        for op in items:
            joint = requires_collective_cohort(case, op, recipes.select(op))
            if (case.runtime.parallel.ep_spans_dp
                    and dict(op.metadata).get("group_axis") == "ep"):
                isolation.append((rank, op.stage, op.op_id,
                                  "complete_cohort" if joint else "per_replica"))
            if not joint:
                continue
            group = group_for[rank]
            if set(group) - expanded.keys():
                raise ValueError("operator_collective_requires_complete_cohort")
            members = joins.setdefault((group, op.op_id), {})
            if rank in members:
                raise ValueError(f"operator_duplicate_collective:{op.op_id}")
            members[rank] = op
            joint_nodes.add((rank, op.op_id))

    rows, cache = [], {}
    for (group, identifier), members in joins.items():
        if set(members) != set(group):
            raise ValueError(f"operator_collective_participants_mismatch:{identifier}")
        key = tuple((rank, op.stage, physical_id(op))
                    for rank, op in sorted(members.items()))
        if key not in cache:
            result = cohort_operator_costs(
                case, dict(sorted(members.items())), recipes=recipes, primitives=primitives,
                replay_scope={rank: scopes[rank] for rank in members},
                input_ready_ms=input_ready_ms, evaluation=evaluation,
                stream_plans=stream_plans)
            result.update(id=f"cohort{len(cache)}", occurrences=0)
            for row in result["operators"]:
                row.update(cohort_id=result["id"], occurrences=0, total_ms=0)
                rows.append(row)
            cache[key] = result
        result = cache[key]
        result["occurrences"] += 1
        for row in result["operators"]:
            row["occurrences"] += 1
            row["total_ms"] += row["duration_ms"]

    for rank, items in expanded.items():
        local = tuple(op for op in items if (rank, op.op_id) not in joint_nodes)
        rank_plans = (
            tuple(p for p in stream_plans if int(p.rank) == int(rank))
            if stream_plans else None
        )
        result = operator_costs(
            case, local, recipes=recipes, primitives=primitives,
            replay_scope=scopes[rank], input_ready_ms=input_ready_ms, replica=rank,
            evaluation=evaluation, stream_plans=rank_plans or None)
        rows.extend({"replica": rank, **row} for row in result["operators"])
    isolation_modes = {item[-1] for item in isolation}
    return {
        "schema": "mindie-sim.operators/1",
        **model_coverage(case.model),
        "boundary": "sum_of_isolated_device_programs",
        "replicas": list(graphs),
        "cross_dp_transport_isolation": (
            "mixed" if len(isolation_modes) > 1 else
            next(iter(isolation_modes), "per_replica")),
        # Dispatch and combine can select different scopes. A summary label alone
        # cannot establish comparable isolation when only one call is replaced.
        "transport_isolation_signature": sha256(
            json.dumps(sorted(isolation), separators=(",", ":")).encode()).hexdigest(),
        "operator_occurrences": sum(row["occurrences"] for row in rows),
        "operator_sum_ms": sum(row["total_ms"] for row in rows),
        "operators": rows,
        **({"collective_cohorts": [
            {key: value for key, value in result.items() if key != "operators"}
            for result in cache.values()],
            "sum_interpretation": "sum_of_replica_representative_completion_spans"}
           if cache else {}),
    }


def _replay_scope(ops, declared):
    if declared is not None:
        return declared
    scopes = {dict(op.metadata)["graph_replay_scope"] for op in ops
              if "graph_replay_scope" in dict(op.metadata)}
    if len(scopes) > 1:
        raise ValueError("operator_replay_scopes_differ")
    return next(iter(scopes), "none")


def _evaluate_isolated_group(
    execution, op, *, replica: int, replay_scope: str, input_ready_ms: float,
    identity: str, count: int, compute_warm: bool = False,
):
    from mindie_sim.prediction.program import NativeProgram

    op, recipe = execution.prepare(op, replica)
    if requires_collective_cohort(execution.case, op, recipe):
        raise ValueError("operator_collective_requires_complete_cohort:use_graph_costs")
    context = execution._context(op, replica, replay_scope, recipe=recipe)
    engine = execution.engine
    l2_view = getattr(engine, "admission_memory_view", None)
    l2_fp_fn = getattr(l2_view, "residency_fingerprint", None)
    l2_before = str(l2_fp_fn()) if callable(l2_fp_fn) else ""
    ready = engine.submit("input_ready", release=input_ready_ms)
    program = execution.lower_program(context, recipe)
    cp_binding = getattr(context, "collective_plan", None)
    cp_plan = getattr(cp_binding, "plan", None) if cp_binding is not None else None
    if (cp_plan is not None
            and hasattr(cp_plan, "lower_with_local_work")
            and getattr(recipe, "algorithm", None) is None
            and getattr(cp_binding, "algorithm", None) != "direct_mesh"
            and context.joint_native_service is None):
        program = cp_plan.lower_with_local_work(context, program)
    program = execution._apply_resource_service_program(context, program)
    program = execution._apply_stream_owners_program(context, program)
    if isinstance(program, NativeProgram):
        rank_index = execution.participant_rank_index(replica, op.stage)
        complete = execution.start_native_participant_with_handles(
            program, "operator", ready, rank_index=rank_index,
            resource_names=execution.native_resource_names(
                context, program, rank=rank_index, replica=replica),
            transport=execution.native_transport(op, replica))[0]
    else:
        complete = program.start(engine, "operator", ready)
    engine.advance(horizon=input_ready_ms)
    early_service = {k: v for k, v in engine.serviced.items() if v and "stream_quota@" not in k}
    engine.advance(until=(complete,))
    duration = complete.time - input_ready_ms
    hw_serviced = {k: v for k, v in engine.serviced.items() if "stream_quota@" not in k}
    l2_after = str(l2_fp_fn()) if callable(l2_fp_fn) else ""
    comm_accounting = (
        execution.topology.accounting(
            op, replica, volume=context.native_work["link"],
            transport=execution.transport_options_for(op, replica, replay_scope),
            recipe=recipe)
        if context.native_work.get("link") else None
    )
    row_context = (
        replace(context, work=dict(comm_accounting.wire_by_resource))
        if (comm_accounting is not None
            and getattr(recipe, "algorithm", None) == "oneshot"
            and context.joint_native_service is None)
        else context
    )
    row = {
        **_cost_row(
            execution, op, recipe, row_context, identity=identity,
            count=count, duration=duration,
            input_ready_ms=input_ready_ms, serviced=hw_serviced,
            early_service=early_service),
        **({"communication": _communication_description(comm_accounting, recipe)}
           if comm_accounting is not None else {}),
    }
    if compute_warm and l2_after != l2_before and not isinstance(program, NativeProgram):
        from mindie_sim.prediction.runtime.l2_residency import AdmissionMemoryView
        if isinstance(l2_view, AdmissionMemoryView):
            warm_duration = _probe_warm_entry_duration(execution, program, input_ready_ms)
            if warm_duration + _WARM_DURATION_EPS_MS < duration:
                row["warm_duration_ms"] = warm_duration
    return row


def operator_costs(case, ops, *, recipes=None, primitives=None, replay_scope=None,
                    input_ready_ms=0, replica=0, evaluation=None, stream_plans=None,
                    _compute_warm: bool = False):
    """Price local operators or complete replica->operators cohort mappings."""
    from mindie_sim.prediction.program import RecipeBook
    if input_ready_ms < 0 or not math.isfinite(input_ready_ms):
        raise ValueError("operator_input_ready_time_must_be_finite_and_nonnegative")
    from mindie_sim.prediction.evaluation import EvaluationContext, context_for
    case = consume_target_declaration(parse_case(case), ops)
    ctx = (context_for(case, evaluation, recipes) if evaluation is not None
           else EvaluationContext.create(case, recipes, primitives))
    recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    from mindie_sim.prediction.efficiency.unmodeled_gate import assert_timing_supported
    if isinstance(ops, Mapping):
        for items in ops.values():
            assert_timing_supported(case, tuple(recipes.expand(tuple(items))),
                                    entry="operator_costs")
        from mindie_sim.prediction.efficiency.exposed_breakdown import unavailable_breakdown
        result = _cohort_costs(
            case, ops, recipes, primitives, replay_scope, input_ready_ms,
            evaluation=ctx, stream_plans=stream_plans)
        result.update(unavailable_breakdown(
            "operator_only", boundary="operator_sum"))
        return result
    ops = tuple(recipes.expand(ops))
    # Refuse to fabricate an isolated operator duration for a core hole. This
    # precedes the barrier-family skip below.
    assert_timing_supported(case, ops, entry="operator_costs")
    unbound_inverse = any(dict(op.metadata).get("reverse_of_collective")
                          and "peer_traffic" not in dict(op.metadata) for op in ops)
    if unbound_inverse and case.runtime.parallel.ep_spans_dp:
        raise ValueError("inverse_traffic_requires_complete_dispatch_cohort:pass_replica_mapping")
    if unbound_inverse:
        ops = bind_cohort_traffic(case, {replica: ops})[replica]
    replay_scope = _replay_scope(ops, replay_scope)
    groups = {}
    for op in ops:
        if op.family == "barrier":
            continue
        stream_key = (op.stream or "main") if stream_plans else ""
        key = op.stage, physical_signature(op), stream_key
        if key not in groups:
            groups[key] = {"op": op, "count": 0}
        groups[key]["count"] += 1
    rows = []
    isolated_cache = getattr(ctx, "isolated_results", None)
    execution = None
    stream_owners_lookup = {}
    if stream_plans:
        from mindie_sim.prediction.streams import bind_stream_owners, validate_stream_plan_ranks
        plans_seq = validate_stream_plan_ranks(stream_plans, {int(replica)})
        stream_plans = plans_seq
        for plan in plans_seq:
            for profile in ctx.bindings.stage_profiles:
                if profile.replica == int(replica):
                    bound = bind_stream_owners(plan, profile.datapath)
                    for sid, owners in bound.items():
                        stream_owners_lookup[(int(replica), profile.stage, sid)] = owners
    for group in groups.values():
        op = group["op"]
        signature = physical_signature(op)
        identity = physical_id(op)
        profile_key = ctx.bindings.profile_key(replica, op.stage)
        op_owners = stream_owners_lookup.get((int(replica), int(op.stage), op.stream or "main"))
        link_scope = (
            (int(replica),)
            if (float(getattr(op, "link_bytes", 0.0) or 0.0) > 0.0
                or dict(op.metadata).get("group_axis"))
            else ()
        )
        cache_key = (profile_key, int(op.stage), signature, replay_scope,
                     float(input_ready_ms), op_owners, link_scope, bool(_compute_warm))
        if isolated_cache is not None:
            cached = isolated_cache.get(cache_key)
            if cached is not None:
                rows.append(_cached_operator_row(
                    cached, op, identity, replica, int(group["count"])))
                continue
        from mindie_sim.prediction.efficiency.cost import CostLedger
        if execution is None:
            execution = Execution(
                case, ctx.cost, recipes=recipes,
                bindings=ctx.bindings, ledger=CostLedger(),
                isolated_replica=int(replica), evaluation=ctx,
                stream_plans=stream_plans)
            execution._compact_step = not execution.legacy_diagnostics
        else:
            execution.reset_engine()
        execution.engine._compact_step = execution._compact_step
        row = _evaluate_isolated_group(
            execution, op, replica=replica, replay_scope=replay_scope,
            input_ready_ms=input_ready_ms, identity=identity, count=group["count"],
            compute_warm=bool(_compute_warm),
        )
        rows.append(row)
        if isolated_cache is not None:
            isolated_cache.put(
                cache_key,
                _remap_replica_scopes(dict(row), int(replica), 0))
    from mindie_sim.prediction.efficiency.exposed_breakdown import unavailable_breakdown
    return {
        "schema": "mindie-sim.operators/1",
        **model_coverage(case.model),
        "boundary": "sum_of_isolated_device_programs",
        "operator_occurrences": sum(row["occurrences"] for row in rows),
        "operator_sum_ms": sum(row["total_ms"] for row in rows),
        "operators": rows,
        **unavailable_breakdown("operator_only", boundary="operator_sum"),
    }
