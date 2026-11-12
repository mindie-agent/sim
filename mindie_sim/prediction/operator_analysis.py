"""Independent device-program costs before serving policy is applied.

[Role] Run each distinct physical operator through the shared execution engine.
[Invariants] The sum of isolated operator times is not a concurrent graph span
  or a serving latency. Names, layers and requests do not change physical cost.
[Timeouts & Locks] Bounded by the distinct operators in one active graph.
[Side-Effect Policy] No measurements, device contexts or external effects.
"""
from dataclasses import replace
from collections.abc import Mapping
from hashlib import sha256
import json
import math
from mindie_sim.coverage import model_coverage

from .cost import CostModel
from .cohort_traffic import bind_cohort_traffic
from .execution import CollectiveParticipant, Execution
from .structure import parse_case
from .body_defaults import (TargetGraphSet, TargetOperators, consume_target_declaration,
                            effective_body_declaration)
from .graph import annotate_geometry


def _selected_group(case, selected_replica=None, cohort=None):
    from mindie_sim.prediction.dispatch import cohorts
    groups = [tuple(group) for group in cohorts(case.runtime.parallel)]
    if cohort is not None:
        selected = tuple(int(value) for value in cohort)
        if selected not in groups:
            raise ValueError("target_graphs_cohort_is_not_a_declared_participant_group")
        return selected
    if selected_replica is not None:
        replica = int(selected_replica)
        group = next((group for group in groups if replica in group), None)
        if group is None:
            raise ValueError("target_graphs_selected_replica_out_of_range")
        return group
    if not groups:
        raise ValueError("target_graphs_requires_a_participant_group")
    return groups[0]


def _group_sequences(sequences, group, parallel):
    if type(sequences) is int:
        return sequences
    values = tuple(sequences)
    if len(values) == len(group):
        return values
    if len(values) == parallel.dp:
        return tuple(values[replica] for replica in group)
    raise ValueError("cohort_sequences_require_positive_scalar_or_complete_nonnegative_counts")


def target_operators(case, *, phase, tokens, context=0, sequences=1, scope="body",
                     media_tokens_per_sequence=None, resource_fusion=True,
                     draft_prepare=False, apply_body_defaults=True,
                     routing_views=None, physical_rank=None, query_position=None,
                     selected_replica=None, cohort=None):
    from mindie_sim.schema.workload import RequestView
    from .dispatch import cohorts

    case = parse_case(case)
    _check_geometry(phase, tokens, sequences, context, scope)
    case, selection = effective_body_declaration(
        case, phase, tokens=tokens, scope=scope, enabled=apply_body_defaults)
    _check_draft_preparation(case, phase, scope, draft_prepare)
    group = _selected_group(case, selected_replica, cohort)
    sequences = _group_sequences(sequences, group, case.runtime.parallel)
    selected = (int(selected_replica) if selected_replica is not None
                else group[0])
    view_count = (int(sequences) if type(sequences) is int else
                  int(sequences[group.index(selected)]))
    views = tuple(RequestView(f"analysis{index}", tokens, context, scope == "model", phase,
                              draft_prepare=draft_prepare)
                  for index in range(max(0, view_count)))
    case = _media_case(case, views, scope, media_tokens_per_sequence)
    if (len(group) > 1
            and case.runtime.kernels.get("sequence_parallel")):
        # A single rank cannot resolve cross-DP SP without the complete
        # cohort's query totals. Delegate only the cohort graph binding.
        graphs = target_graphs(
            case, phase=phase, tokens=tokens, context=context,
            sequences=sequences, scope=scope,
            media_tokens_per_sequence=media_tokens_per_sequence,
            resource_fusion=resource_fusion, draft_prepare=draft_prepare,
            apply_body_defaults=False, routing_views=routing_views,
            physical_rank=physical_rank, query_position=query_position,
            selected_replica=selected_replica, cohort=group,
        )
        selected = (int(selected_replica) if selected_replica is not None
                    else group[0])
        result = tuple(op for op in graphs[selected] if op.family != "barrier")
        return (TargetOperators(result, kernels=case.runtime.kernels, selection=selection)
                if scope == "body" else result)
    replica = selected
    physical = _physical_rank_for(case, replica)
    result = tuple(op for op in _target_ops(
        case, views, scope, resource_fusion=resource_fusion,
        routing_views=routing_views, physical_rank=physical,
        query_position=query_position, replica=replica)
                   if op.family != "barrier")
    return (TargetOperators(result, kernels=case.runtime.kernels, selection=selection)
            if scope == "body" else result)


def _check_geometry(phase, tokens, sequences, context, scope):
    if phase not in {"prefill", "decode"}:
        raise ValueError("operator_analysis_requires_prefill_or_decode")
    if (any(not isinstance(value, int) or isinstance(value, bool)
            for value in (tokens, sequences, context))
            or min(tokens, sequences) < 1 or context < 0 or scope not in {"body", "model"}):
        raise ValueError("operator_analysis_requires_positive_geometry_and_body_or_model_scope")


def cohort_sequence_counts(sequences, participants):
    """Scalar means each DP replica; a list declares the complete EP cohort."""
    if type(sequences) is int and sequences > 0:
        return {replica: sequences for replica in participants}
    if (not isinstance(sequences, (list, tuple)) or len(sequences) != len(participants)
            or any(type(value) is not int or value < 0 for value in sequences)
            or not any(sequences)):
        raise ValueError("cohort_sequences_require_positive_scalar_or_complete_nonnegative_counts")
    return dict(zip(participants, sequences))


def _check_draft_preparation(case, phase, scope, enabled):
    if enabled:
        from mindie_sim.catalog.draft import optional_draft_spec
        if phase != "prefill" or scope != "model":
            raise ValueError("draft_preparation_requires_prefill_model_scope")
        mode = dict(case.runtime.metadata).get("draft_mode")
        if not mode:
            raise ValueError("draft_preparation_requires_declared_runtime_draft_mode")
        optional_draft_spec(case.model, mode)


def _target_ops(case, views, scope, *, global_views=None, resource_fusion=True,
                cohort_tokens=None, phase=None, replica=0, active_rows=None,
                source_rows_by_replica=None, routing_global_tokens=None,
                routing_views=None, physical_rank=None, query_position=None):
    from mindie_sim.schema.workload import IterationPlan
    from .attention import AttentionPolicy
    from .frontend import Frontend
    from .subgraphs.instantiate import FeatureConfig
    from .transforms import lower_implementations
    from .implementation_selection import resolve, annotate

    features = FeatureConfig.from_runtime(
        case.runtime, case.model, machine=case.machine)
    frontend = Frontend(
        attention_policy=AttentionPolicy(**case.runtime.attention_policy),
        features=features)
    global_tokens = (cohort_tokens if cohort_tokens is not None else
                     sum(v.q_len + v.draft_proposal_tokens for v in global_views)
                     if global_views is not None else None)
    if scope == "model":
        work = frontend(IterationPlan(views), case.model, case.runtime, case.machine,
                        global_tokens=global_tokens, global_views=global_views,
                        resource_fusion=resource_fusion, replica=replica,
                        source_rows_by_replica=source_rows_by_replica,
                        routing_global_tokens=routing_global_tokens,
                        routing_views=routing_views,
                        physical_rank=physical_rank,
                        query_position=query_position)
        result, aliases = [], {}
        for op in work.ops:
            dependencies = tuple(dict.fromkeys(
                parent for dep in op.deps for parent in aliases.get(dep, (dep,))))
            if op.family in {"sampling", "token_commit"}:
                # Draft cache preparation follows target completion, but this
                # public boundary excludes sampling and the serving commit.
                aliases[op.op_id] = dependencies
                continue
            metadata = op.metadata + (("graph_replay_scope", work.replay_scope),) \
                if op.op_id == "iteration_input" else op.metadata
            result.append(replace(op, deps=dependencies, metadata=metadata))
        return tuple(result)
    views = tuple(replace(v, q_len=v.q_len + v.draft_proposal_tokens) for v in views)
    bucket, replay_scope = frontend.graph.select(views, case.runtime) if views else (None, "none")
    padded_tokens = max(sum(v.q_len for v in views), bucket or 0)
    runtime, selections = resolve(case.runtime, views, padded_tokens=padded_tokens,
                                  replay_scope=replay_scope, global_views=global_views)
    case = replace(case, runtime=runtime)
    local, local_tokens, pcp_prefill = frontend.partition_views(
        views, case.runtime, padded_tokens)
    pcp_cohort = case.runtime.parallel.pcp > 1 and any(
        v.phase == "prefill" for v in (global_views or views))
    body_global_tokens = (
        frontend.pcp_cohort_tokens(views, local, local_tokens, case.runtime,
                                   global_tokens=global_tokens, global_views=global_views)
        if pcp_cohort else global_tokens
    )
    ops, _ = frontend.body(case.model, case.runtime, local, ready="analysis_input",
                          tokens=local_tokens, mode=views[0].phase if views else phase,
                          global_tokens=body_global_tokens, replica=replica,
                          source_rows_by_replica=source_rows_by_replica,
                          routing_global_tokens=routing_global_tokens,
                          routing_views=routing_views,
                          physical_rank=physical_rank,
                          query_position=query_position)
    # Source lowerings consume the same active/storage distinction as the
    # model path, including MC2's mask over padded graph inputs.
    logical_rows = (sum(v.logical_q_len for v in views) if active_rows is None
                    else int(active_rows))
    ops = annotate_geometry(ops, active_rows=logical_rows,
                            physical_rows=padded_tokens)
    ops = frontend.graph.apply(lower_implementations(
        ops, case.runtime, case.model.activation_dtype, resource_fusion=False,
        local_tokens=local_tokens), replay_scope, case.runtime)
    if resource_fusion:
        from .native_fusion import lower_native_fusions
        ops = lower_native_fusions(ops, case.runtime, local_tokens=local_tokens)
    logical_rows = (sum(v.logical_q_len for v in views) if active_rows is None
                    else int(active_rows))
    ops = annotate_geometry(ops, active_rows=logical_rows,
                            physical_rows=padded_tokens)
    return annotate(tuple(replace(op, metadata=op.metadata + (("graph_replay_scope", replay_scope),))
                          for op in ops), selections)


def _media_case(case, views, scope, media_tokens):
    from .frontend import _media_counts

    if scope != "model" or not any(v.phase == "prefill" for v in views):
        if media_tokens:
            raise ValueError("media_analysis_requires_prefill_model_scope")
        return case
    if media_tokens is None:
        if any(_media_counts(case.runtime).values()):
            raise ValueError("analysis_requires_explicit_media_tokens_per_sequence:0_excludes_tower")
        return case
    if not isinstance(media_tokens, int) or media_tokens < 0:
        raise ValueError("analysis_media_tokens_must_be_nonnegative_integer")
    metadata = dict(case.runtime.metadata)
    metadata["media_tokens_by_request"] = {view.request_id: media_tokens for view in views}
    return replace(case, runtime=replace(case.runtime, metadata=tuple(metadata.items())))


def graph_padded_idle(case, phase: str, counts) -> bool:
    """Whether a genuine declared graph protocol physically pads zero lanes.

    This is source/runtime selection, not a caller flag: full decode capture
    with declared keys (or an explicit piecewise capture) runs captured graph
    rows even when the logical lane count is zero.  Eager/none keeps zero work.
    """
    if (phase != "decode" or not counts
            or not any(int(v) > 0 for v in counts.values())
            or all(int(v) > 0 for v in counts.values())):
        return False
    runtime = case.runtime
    if str(runtime.graph_mode).lower() in {"none", "off", "eager", "disabled"}:
        return False
    if runtime.deepep_high_throughput:
        return False
    full = (str(runtime.graph_capture_mode).lower() == "full"
            and not runtime.invalid_full_modes)
    keys = ((runtime.graph_capture_keys or runtime.graph_buckets)
            if full else runtime.piecewise_capture_keys)
    return bool(keys)


def _physical_rank_for(case, replica):
    from .topology import Topology
    try:
        return Topology(case.machine, case.runtime.parallel).ranks(replica, 0)[0]
    except Exception:
        return None


def _routing_views_for(value, replica):
    if value is None:
        return None
    if isinstance(value, Mapping):
        return value.get(replica)
    return value


def target_graphs(case, *, phase, tokens, context=0, sequences=1, scope="model",
                  media_tokens_per_sequence=None, resource_fusion=True,
                  draft_prepare=False, apply_body_defaults=True,
                  padded_idle=False, routing_views=None, physical_rank=None,
                  query_position=None, selected_replica=None, cohort=None):
    """One DP cohort; scalar sequences repeat, lists declare every replica.

    ``padded_idle`` retains the declared physical capture rows for a replica
    whose logical sequence count is zero.  The logical ledger stays empty; the
    physical padding view launches the actual graph bucket and EP kernel
    participation without inventing a logical request.
    """
    from mindie_sim.schema.workload import RequestView
    from .dispatch import cohorts

    case = parse_case(case)
    _check_geometry(phase, tokens, 1, context, scope)
    case, selection = effective_body_declaration(
        case, phase, tokens=tokens, scope=scope, enabled=apply_body_defaults)
    _check_draft_preparation(case, phase, scope, draft_prepare)
    participants = _selected_group(case, selected_replica, cohort)
    sequences = _group_sequences(sequences, participants, case.runtime.parallel)
    counts = cohort_sequence_counts(sequences, participants)
    pad_idle = padded_idle or graph_padded_idle(case, phase, counts)
    logical_views = {}
    views = {}
    for replica in participants:
        count = counts[replica]
        logical_views[replica] = tuple(RequestView(
            f"analysis{replica}:{i}", tokens, context, scope == "model",
            phase, draft_prepare=draft_prepare,
        ) for i in range(count))
        if count == 0 and pad_idle:
            # A physical graph row is not a logical request.  A zero-length
            # padded view lets GraphPolicy select the declared capture bucket;
            # the frontend then lowers that many physical rows while the
            # logical ledger remains zero.
            from .graph import GraphPolicy
            probe = RequestView(
                f"analysis{replica}:bucket_probe", 0, context, False, phase,
                padding=True, draft_verify=False, draft_prepare=False,
            )
            bucket, _replay = GraphPolicy().select(
                (probe,), case.runtime)
            physical = int(bucket or 0)
            views[replica] = ((RequestView(
                f"analysis{replica}:padding", physical, context, False, phase,
                padding=True, draft_verify=False, draft_prepare=False,
            ),) if physical else ())
        else:
            views[replica] = logical_views[replica]
    active_rows_by_replica = {
        replica: sum(v.q_len for v in logical_views[replica])
        for replica in participants
    }
    case = _media_case(case, tuple(v for group in logical_views.values() for v in group),
                       scope, media_tokens_per_sequence)
    global_views = (tuple(v for group in logical_views.values() for v in group)
                    if len(logical_views) > 1 else None)
    padded_cohort = None
    sp_padding = case.runtime.kernels.get("sequence_parallel", {}).get("padding")
    if len(logical_views) > 1 and (case.runtime.routing.get("policy") == "balanced"
                                   or sp_padding == "right_to_tp"):
        from .balanced_routing import cohort_padded_tokens
        padded_cohort = cohort_padded_tokens(logical_views.values(), case.runtime)
    # The ordinary selector consumes the source rows each physical EP rank
    # actually owns.  With SP, valid_per_tp is that exact ownership; flattening
    # the DP replicas gives the EP-rank vector.  Without SP the vector remains
    # DP-scoped and the lowering keeps its declared uniform peer policy.
    source_rows_by_replica = tuple(
        active_rows_by_replica[p] for p in participants)
    routing_global_tokens = (sum(active_rows_by_replica.values())
                             if global_views is not None else None)
    if len(participants) > 1 and "sequence_parallel" in case.runtime.kernels:
        from .sequence_parallel import select as select_sp
        total_global = sum(active_rows_by_replica.values())
        physical_rows = []
        for replica in participants:
            plan = select_sp(
                case.runtime, case.model, logical_views[replica],
                tokens=active_rows_by_replica[replica],
                global_tokens=total_global)
            if plan is not None:
                physical_rows.extend(plan.valid_per_tp)
        if physical_rows:
            source_rows_by_replica = tuple(physical_rows)
    explicit_routing = any(
        isinstance(item, Mapping) and any(key in item for key in (
            "requests", "request_views", "expert_placement", "source_owners",
            "source_owner", "query_position"))
        for item in (case.runtime.routing.get("realizations") or ()))
    uniform_replicas = (
        len(set(counts.values())) <= 1
        and case.runtime.routing.get("policy") != "balanced"
        and not explicit_routing
        and not any(
            isinstance(item, Mapping) and "replica" in item
            for item in (case.runtime.routing.get("realizations") or ())
        )
    )
    if uniform_replicas and participants:
        first_r = participants[0]
        base_ops = _target_ops(
            case, views[first_r], scope, global_views=global_views,
            resource_fusion=resource_fusion, cohort_tokens=padded_cohort,
            phase=phase, replica=first_r,
            active_rows=active_rows_by_replica[first_r],
            source_rows_by_replica=source_rows_by_replica,
            routing_global_tokens=routing_global_tokens,
            routing_views=_routing_views_for(routing_views, first_r),
            physical_rank=_physical_rank_for(case, first_r),
            query_position=query_position,
        )
        raw_graphs = {first_r: base_ops}
        for r in participants[1:]:
            req_ids = tuple(v.request_id for v in views[r])
            raw_graphs[r] = tuple(
                replace(op, requests=req_ids) if op.requests else op
                for op in base_ops
            )
    else:
        raw_graphs = {
            r: _target_ops(case, group, scope, global_views=global_views,
                           resource_fusion=resource_fusion, cohort_tokens=padded_cohort,
                           phase=phase, replica=r,
                           active_rows=active_rows_by_replica[r],
                           source_rows_by_replica=source_rows_by_replica,
                           routing_global_tokens=routing_global_tokens,
                           routing_views=_routing_views_for(routing_views, r),
                           physical_rank=_physical_rank_for(case, r),
                           query_position=query_position)
            for r, group in views.items()
        }
    graphs = bind_cohort_traffic(case, raw_graphs)
    return (TargetGraphSet(graphs, kernels=case.runtime.kernels, selection=selection)
            if scope == "body" else graphs)


def physical_signature(op):
    """All work/implementation axes, excluding only graph identity."""
    data = op.to_dict()
    for key in ("op_id", "deps", "stage", "layer", "requests", "stream", "notes"):
        data.pop(key, None)
    for key in ("collective_id", "reverse_of_collective"):
        data["metadata"].pop(key, None)
    return json.dumps(data, sort_keys=True, separators=(",", ":"))


def physical_id(op):
    return sha256(physical_signature(op).encode()).hexdigest()


def requires_collective_cohort(case, op, recipe):
    return bool(
        getattr(recipe, "algorithm", None)
        and case.runtime.parallel.ep_spans_dp
        and dict(op.metadata).get("group_axis") == "ep")


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
        "service_delta": {key: serviced.get(key, 0) - context.work.get(key, 0)
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


def _communication_description(accounting, recipe):
    describe = getattr(recipe, "accounting_description", None)
    return describe(accounting) if describe else accounting.to_dict()


def cohort_operator_costs(case, ops, *, recipes, primitives=None,
                          replay_scope="none", input_ready_ms=0):
    """Isolate one complete transport cohort, attributing wire to source ranks."""
    execution = Execution(case, CostModel(case, primitives), recipes=recipes)
    engine, participants, identities = execution.engine, [], {}
    for replica, original in ops.items():
        identities[replica] = physical_id(original)
        op, recipe = execution.prepare(original, replica)
        scope = replay_scope[replica] if isinstance(replay_scope, Mapping) else replay_scope
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
        accounting = execution.topology.accounting(
            context.op, item.replica, volume=context.native_work.get("link", 0))
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


def _cohort_costs(case, graphs, recipes, primitives, replay_scope, input_ready_ms):
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
                replay_scope={rank: scopes[rank] for rank in members}, input_ready_ms=input_ready_ms)
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
            replay_scope=scopes[rank], input_ready_ms=input_ready_ms, replica=rank)
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
                    input_ready_ms=0, replica=0):
    """Price local operators or complete replica->operators cohort mappings."""
    from .program import RecipeBook
    if input_ready_ms < 0 or not math.isfinite(input_ready_ms):
        raise ValueError("operator_input_ready_time_must_be_finite_and_nonnegative")
    case = consume_target_declaration(parse_case(case), ops)
    recipes = recipes or RecipeBook.from_mapping(case.runtime.kernels)
    if isinstance(ops, Mapping):
        return _cohort_costs(
            case, ops, recipes, primitives, replay_scope, input_ready_ms)
    ops = tuple(recipes.expand(ops))
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
    for group in groups.values():
        op = group["op"]
        execution = Execution(case, CostModel(case, primitives), recipes=recipes)
        identity = physical_id(op)
        op, recipe = execution.prepare(op, replica)
        if requires_collective_cohort(case, op, recipe):
            raise ValueError(
                "operator_collective_requires_complete_cohort:use_graph_costs")
        context = execution._context(op, replica, replay_scope, recipe=recipe)
        engine = execution.engine
        ready = engine.submit("input_ready", release=input_ready_ms)
        program = execution.lower_program(context, recipe)
        from .piecewise.compiler import NativeProgram
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
        rows.append({
            **_cost_row(
                execution, op, recipe, context, identity=identity,
                count=group["count"], duration=duration,
                input_ready_ms=input_ready_ms, serviced=engine.serviced,
                early_service=early_service),
            **({"communication": _communication_description(execution.topology.accounting(
                op, replica, volume=context.native_work["link"]), recipe)}
               if context.native_work.get("link") else {}),
        })
    return {
        "schema": "mindie-sim.operators/1",
        **model_coverage(case.model),
        "boundary": "sum_of_isolated_device_programs",
        "operator_occurrences": sum(row["occurrences"] for row in rows),
        "operator_sum_ms": sum(row["total_ms"] for row in rows),
        "operators": rows,
    }
