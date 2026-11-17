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

from .cohort_traffic import bind_cohort_traffic
from .execution import CollectiveParticipant, Execution
from .structure import parse_case
from .body_defaults import consume_target_declaration


def physical_signature(op):
    """Deterministic serialization of the canonical physical key."""
    from .physical_key import physical_signature as _signature
    return _signature(op)


def physical_id(op):
    from .physical_key import physical_id as _identifier
    return _identifier(op)


def requires_collective_cohort(case, op, recipe):
    from .local_tp_groups import local_tp_spans_dp

    axis = dict(op.metadata).get("group_axis")
    return bool(
        getattr(recipe, "algorithm", None)
        and ((case.runtime.parallel.ep_spans_dp and axis == "ep")
             or (local_tp_spans_dp(case.runtime.parallel) and axis == "local_tp")))


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
    "serviced", "service_before_input", "peak_buffers", "remaining_buffers",
    "peak_buffers_per_unit",
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
                          evaluation=None):
    """Isolate one complete transport cohort, attributing wire to source ranks."""
    from .cost import CostLedger
    from .evaluation import EvaluationContext, context_for
    ctx = (context_for(case, evaluation, recipes) if evaluation is not None
           else EvaluationContext.create(case, recipes, primitives))
    execution = Execution(
        case, ctx.cost, recipes=recipes,
        bindings=ctx.bindings, ledger=CostLedger(), evaluation=ctx)
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
    early_service = {key: value for key, value in engine.serviced.items() if value}
    engine.advance(until=tuple(item.completion for item in participants), first=False)
    rows, attributed = [], {}
    duration = max(item.completion.time for item in participants) - input_ready_ms
    for item in participants:
        context = item.context
        transport = execution.transport_options_for(
            context.op, item.replica, replica_scopes.get(item.replica, "none"))
        accounting = execution.topology.accounting(
            context.op, item.replica, volume=context.native_work.get("link", 0),
            transport=transport)
        for resource, amount in accounting.wire_by_resource.items():
            attributed[resource] = attributed.get(resource, 0) + amount
        row = _cost_row(
            execution, context.op, item.recipe, context,
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
    actual = {key: value for key, value in engine.serviced.items() if value}
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
                  evaluation=None):
    """Isolate joint transports once, retaining each replica's local programs."""
    from .dispatch import cohorts

    group_for = {rank: group for group in cohorts(case.runtime.parallel) for rank in group}
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
                input_ready_ms=input_ready_ms, evaluation=evaluation)
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
        result = operator_costs(
            case, local, recipes=recipes, primitives=primitives,
            replay_scope=scopes[rank], input_ready_ms=input_ready_ms, replica=rank,
            evaluation=evaluation)
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


def operator_costs(case, ops, *, recipes=None, primitives=None, replay_scope=None,
                    input_ready_ms=0, replica=0, evaluation=None):
    """Price local operators or complete replica->operators cohort mappings."""
    from .program import RecipeBook
    if input_ready_ms < 0 or not math.isfinite(input_ready_ms):
        raise ValueError("operator_input_ready_time_must_be_finite_and_nonnegative")
    from .evaluation import EvaluationContext, context_for
    case = consume_target_declaration(parse_case(case), ops)
    ctx = (context_for(case, evaluation, recipes) if evaluation is not None
           else EvaluationContext.create(case, recipes, primitives))
    recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    from .unmodeled_gate import assert_timing_supported
    if isinstance(ops, Mapping):
        for items in ops.values():
            assert_timing_supported(case, tuple(recipes.expand(tuple(items))),
                                    entry="operator_costs")
        from .exposed_breakdown import unavailable_breakdown
        result = _cohort_costs(
            case, ops, recipes, primitives, replay_scope, input_ready_ms,
            evaluation=ctx)
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
        key = op.stage, physical_signature(op)
        if key not in groups:
            groups[key] = {"op": op, "count": 0}
        groups[key]["count"] += 1
    rows = []
    isolated_cache = getattr(ctx, "isolated_results", None)
    execution = None
    for group in groups.values():
        op = group["op"]
        signature = physical_signature(op)
        identity = physical_id(op)
        profile_key = ctx.bindings.profile_key(replica, op.stage)
        cache_key = (profile_key, int(op.stage), signature, replay_scope,
                     float(input_ready_ms))
        if isolated_cache is not None:
            cached = isolated_cache.get(cache_key)
            if cached is not None:
                rows.append(_cached_operator_row(
                    cached, op, identity, replica, int(group["count"])))
                continue
        from .cost import CostLedger
        if execution is None:
            execution = Execution(
                case, ctx.cost, recipes=recipes,
                bindings=ctx.bindings, ledger=CostLedger(),
                isolated_replica=int(replica), evaluation=ctx)
            execution._compact_step = not execution.legacy_diagnostics
            execution.engine._compact_step = execution._compact_step
        else:
            execution.reset_engine()
        op, recipe = execution.prepare(op, replica)
        if requires_collective_cohort(case, op, recipe):
            raise ValueError(
                "operator_collective_requires_complete_cohort:use_graph_costs")
        context = execution._context(op, replica, replay_scope, recipe=recipe)
        engine = execution.engine
        ready = engine.submit("input_ready", release=input_ready_ms)
        program = execution.lower_program(context, recipe)
        from .program import NativeProgram
        if isinstance(program, NativeProgram):
            # An isolated native call still uses its declared physical rank
            # ports; it is the same launch view, not an unbound synthetic one.
            rank_index = execution.participant_rank_index(replica, op.stage)
            complete = execution.start_native_participant_with_handles(
                program, "operator", ready, rank_index=rank_index,
                resource_names=execution.native_resource_names(
                    context, program, rank=rank_index, replica=replica),
                transport=execution.native_transport(op, replica))[0]
        else:
            complete = program.start(engine, "operator", ready)
        engine.advance(horizon=input_ready_ms)
        early_service = {key: value for key, value in engine.serviced.items() if value}
        engine.advance(until=(complete,))
        duration = complete.time - input_ready_ms
        row = {
            **_cost_row(
                execution, op, recipe, context, identity=identity,
                count=group["count"], duration=duration,
                input_ready_ms=input_ready_ms, serviced=engine.serviced,
                early_service=early_service),
            **({"communication": _communication_description(execution.topology.accounting(
                op, replica, volume=context.native_work["link"],
                transport=execution.transport_options_for(
                    op, replica, replay_scope)), recipe)}
               if context.native_work.get("link") else {}),
        }
        rows.append(row)
        if isolated_cache is not None:
            isolated_cache.put(
                cache_key,
                _remap_replica_scopes(dict(row), int(replica), 0))
    from .exposed_breakdown import unavailable_breakdown
    return {
        "schema": "mindie-sim.operators/1",
        **model_coverage(case.model),
        "boundary": "sum_of_isolated_device_programs",
        "operator_occurrences": sum(row["occurrences"] for row in rows),
        "operator_sum_ms": sum(row["total_ms"] for row in rows),
        "operators": rows,
        **unavailable_breakdown("operator_only", boundary="operator_sum"),
    }
