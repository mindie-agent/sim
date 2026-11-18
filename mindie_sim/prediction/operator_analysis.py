"""Independent device-program costs before serving policy is applied.

[Role] Run each distinct physical operator through the shared execution engine.
[Invariants] The sum of isolated operator times is not a concurrent graph span
  or a serving latency. Names, layers and requests do not change physical cost.
[Timeouts & Locks] Bounded by the distinct operators in one active graph.
[Side-Effect Policy] No measurements, device contexts or external effects.
"""
from copy import deepcopy
from dataclasses import replace
from collections.abc import Mapping

from .analysis_inputs import (
    FIRST_PARTICIPANT, UNIFORM_COUNT_CARDINALITY,
    analysis_view, rebind_analysis_requests,
)
from .cohort_traffic import bind_cohort_traffic
from .frontend_features import FeatureConfig
from .local_tp_groups import local_tp_spans_dp
from .structure import parse_case, parse_model
from .body_defaults import (TargetGraphSet, TargetOperators,
                            effective_body_declaration)
from .operator_replay import (
    cohort_operator_costs, operator_costs, physical_id, physical_signature,
    requires_collective_cohort, _replay_scope,
)
from .graph import annotate_geometry



def _analysis_prefix_fields(case, index: int, context: int) -> dict:
    requests = tuple(getattr(getattr(case, "workload", None), "requests", ()) or ())
    if not requests:
        return {}
    spec = requests[index % len(requests)]
    declared = max(0, min(int(context), int(getattr(spec, "prefix_tokens", 0) or 0)))
    key = str(getattr(spec, "prefix_cache_key", "") or "")
    if not declared and not key:
        return {}
    return {"prefix_tokens": declared, "prefix_cache_key": key}


def _selected_group(case, selected_replica=None, cohort=None):
    from mindie_sim.prediction.dispatch import cohorts
    groups = [tuple(group) for group in cohorts(case.runtime.parallel, case.machine)]
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


MODEL_SCOPE = "model"


class _LazyPrograms(Mapping):
    """Replica programs rebuilt on first access, never during lowering."""

    def __init__(self, seeds, bindings=None, owners=None):
        self._seeds = dict(seeds)
        self._bindings = dict(bindings or {})
        self._owners = dict(owners or {})
        self._cache = {}

    def __getitem__(self, replica):
        key = int(replica)
        if key in self._cache:
            return self._cache[key]
        program = self._seeds[key]
        binding = self._bindings.get(key)
        if binding is not None:
            program = rebind_analysis_requests(program, *binding)
        if key in self._owners:
            work = replace(program.work,
                           ops=_rebind_ep_owner(program.work.ops,
                                                self._owners[key]))
            program = replace(program, work=work)
        self._cache[key] = program
        return program

    def __iter__(self):
        return iter(self._seeds)

    def __len__(self):
        return len(self._seeds)


class ModelGraphSet(dict):
    """Replica graph mapping from the same Frontend.program occurrences.

    The mapping keeps the existing ``TargetOperators`` tuple contract.  The
    attached executable is materialized lazily from the final cohort-bound
    replica graph so Local-TP rewrites and collective traffic are visible in
    ``work.ops`` without rebinding every replica during lowering.
    """

    def __init__(self, values, *, programs, bindings=None, owners=None,
                 kernels, selection):
        self.programs = _LazyPrograms(programs, bindings, owners)
        self._attached = {}
        self.scope = MODEL_SCOPE
        self.effective_kernels = deepcopy(kernels)
        self.default_selection = deepcopy(selection)
        super().__init__((replica, TargetOperators._with_declaration(
            ops, self.effective_kernels, self.default_selection, MODEL_SCOPE))
            for replica, ops in values.items())

    @property
    def ports(self):
        return {replica: self.program(replica).ports for replica in self}

    def program(self, replica=0):
        key = int(replica)
        if key not in self._attached:
            from .frontend import program_attached_to
            self._attached[key] = program_attached_to(
                self.programs[key], tuple(self[key]))
        return self._attached[key]


def target_operators(case, *, phase, tokens, context=0, sequences=1, scope="body",
                     media_tokens_per_sequence=None, resource_fusion=True,
                     draft_prepare=False, apply_body_defaults=True,
                     routing_views=None, physical_rank=None, query_position=None,
                     selected_replica=None, cohort=None, pure_model=False,
                     layer_limit=None):
    from .dispatch import cohorts

    case = parse_model(case) if pure_model else parse_case(case)
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
    views = tuple(
        analysis_view(
            case, f"analysis{index}", tokens, context, scope, phase,
            index=index, draft_prepare=draft_prepare,
            **_analysis_prefix_fields(case, index, context),
        )
        for index in range(max(0, view_count))
    )
    case = _media_case(case, views, scope, media_tokens_per_sequence)
    from .local_tp_groups import local_tp_spans_dp
    if (len(group) > 1
            and (case.runtime.kernels.get("sequence_parallel")
                 or local_tp_spans_dp(case.runtime.parallel))):
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
            layer_limit=layer_limit,
        )
        selected = (int(selected_replica) if selected_replica is not None
                    else group[0])
        result = tuple(op for op in graphs[selected] if op.family != "barrier")
        return (TargetOperators(result, kernels=case.runtime.kernels,
                                selection=selection, scope=scope)
                if scope == "body" else result)
    replica = selected
    physical = _physical_rank_for(case, replica)
    result = tuple(op for op in _target_ops(
        case, views, scope, resource_fusion=resource_fusion,
        routing_views=routing_views, physical_rank=physical,
        query_position=query_position, replica=replica, layer_limit=layer_limit)
                   if op.family != "barrier")
    return (TargetOperators(result, kernels=case.runtime.kernels,
                            selection=selection, scope=scope)
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
                routing_views=None, cohort_views_by_replica=None,
                physical_rank=None, query_position=None,
                model_programs=None, layer_limit=None):
    from mindie_sim.schema.workload import IterationPlan
    from .attention import AttentionPolicy
    from .frontend import Frontend
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
        program = frontend.program(
            IterationPlan(views), case.model, case.runtime, case.machine,
            assembly="model", global_tokens=global_tokens,
            global_views=global_views, resource_fusion=resource_fusion,
            replica=replica, source_rows_by_replica=source_rows_by_replica,
            routing_global_tokens=routing_global_tokens,
            routing_views=routing_views,
            cohort_views_by_replica=cohort_views_by_replica,
            physical_rank=physical_rank,
            query_position=query_position)
        if model_programs is not None:
            model_programs[replica] = program
        work = program.work
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
    from .frontend import mark_analytical_runtime
    runtime = mark_analytical_runtime(runtime, selections)
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
                          cohort_views_by_replica=cohort_views_by_replica,
                          physical_rank=physical_rank,
                          query_position=query_position,
                          machine=case.machine, layer_limit=layer_limit)
    # Source lowerings consume the same active/storage distinction as the
    # model path, including MC2's mask over padded graph inputs.
    logical_rows = (sum(v.logical_q_len for v in views) if active_rows is None
                    else int(active_rows))
    ops = annotate_geometry(ops, active_rows=logical_rows,
                            physical_rows=padded_tokens)
    from .implementation_selection import LoweringRegion, lower_regions

    logical_rows = (sum(v.logical_q_len for v in views) if active_rows is None
                    else int(active_rows))
    # T11 owns representation insertion, the post-insertion partition and
    # capture inheritance.  The independent body entry consumes the same
    # lower_regions result as the model path; no pre-insertion graph partition
    # or old-family suffix is used here.
    region = LoweringRegion(
        tuple(ops), case.runtime, case.model.activation_dtype, local_tokens,
        selections=tuple(selections or ()))
    lowered, _ = lower_regions(
        [region], graph=frontend.graph, replay_scope=replay_scope,
        runtime=case.runtime, resource_fusion=resource_fusion)
    lowered = annotate_geometry(lowered, active_rows=logical_rows,
                                physical_rows=padded_tokens)
    return tuple(replace(op, metadata=op.metadata + (
        ("graph_replay_scope", replay_scope),)) for op in lowered)


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
    if str(runtime.graph_mode).lower() in {"none", "off", "eager", "disabled", "auto"}:
        return False
    if runtime.deepep_high_throughput:
        return False
    full = (str(runtime.graph_capture_mode).lower() == "full"
            and not runtime.invalid_full_modes)
    keys = ((runtime.graph_capture_keys or runtime.graph_buckets)
            if full else runtime.piecewise_capture_keys)
    return bool(keys)


def _rebind_ep_owner(ops, owner):
    """Rebind the physical EP owner of a cloned probability-routed graph."""
    result = []
    for op in ops:
        metadata = dict(op.metadata)
        if "ep_rank" in metadata:
            metadata["ep_rank"] = int(owner)
            op = replace(op, metadata=tuple(sorted(metadata.items())))
        result.append(op)
    return tuple(result)


def _build_topology(case):
    from .topology import Topology
    try:
        return Topology(case.machine, case.runtime.parallel)
    except Exception:
        return None


def _physical_rank_for(case, replica, topology=None):
    active = _build_topology(case) if topology is None else topology
    if active is None:
        return None
    try:
        return active.ranks(replica, FIRST_PARTICIPANT)[FIRST_PARTICIPANT]
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
                  query_position=None, selected_replica=None, cohort=None,
                  pure_model=False, layer_limit=None):
    """One DP cohort; scalar sequences repeat, lists declare every replica.

    ``padded_idle`` retains the declared physical capture rows for a replica
    whose logical sequence count is zero.  The logical ledger stays empty; the
    physical padding view launches the actual graph bucket and EP kernel
    participation without inventing a logical request.
    """
    from mindie_sim.schema.workload import RequestView
    from .dispatch import cohorts

    case = parse_model(case) if pure_model else parse_case(case)
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
        logical_views[replica] = tuple(
            analysis_view(
                case, f"analysis{replica}:{i}", tokens, context, scope,
                phase, index=i, draft_prepare=draft_prepare,
                **_analysis_prefix_fields(case, i, context),
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
        replica: sum(v.q_len + v.draft_proposal_tokens
                     for v in logical_views[replica])
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
    from .frontend import Frontend
    cohort_views_by_replica = tuple(
        Frontend.partition_views(
            logical_views[p], case.runtime, active_rows_by_replica[p],
            model=case.model)[0]
        for p in participants)
    routing_global_tokens = (
        padded_cohort if padded_cohort is not None
        else (sum(active_rows_by_replica.values())
              if global_views is not None else None))
    if len(participants) > UNIFORM_COUNT_CARDINALITY and "sequence_parallel" in case.runtime.kernels:
        from .graph import GraphPolicy
        from .sequence_parallel import select as select_sp
        total_global = (padded_cohort if padded_cohort is not None
                        else sum(active_rows_by_replica.values()))
        physical_rows = []
        for replica in participants:
            expanded = tuple(
                replace(v, q_len=v.q_len + v.draft_proposal_tokens)
                for v in logical_views[replica])
            rows = active_rows_by_replica[replica]
            padded = rows
            if sp_padding == "right_to_tp" and expanded:
                bucket, _ = GraphPolicy().select(expanded, case.runtime)
                padded = max(rows, int(bucket or FIRST_PARTICIPANT))
            plan = select_sp(
                case.runtime, case.model, expanded,
                tokens=padded, global_tokens=total_global)
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
        len(set(counts.values())) <= UNIFORM_COUNT_CARDINALITY
        and (scope != "model" or (
            case.machine.homogeneous
            and case.runtime.parallel.rank_table is None
            and not case.runtime.routing
            and not case.runtime.kernels.get("force_eplb_source")
            and not local_tp_spans_dp(case.runtime.parallel)
            and routing_views is None
            and query_position is None
            and all(
                tuple(replace(v, request_id=base.request_id)
                      for v, base in zip(
                          group, views[participants[FIRST_PARTICIPANT]],
                          strict=True))
                == views[participants[FIRST_PARTICIPANT]]
                for group in views.values())
        ))
        and case.runtime.routing.get("policy") != "balanced"
        and int(getattr(case.runtime.parallel, "redundant_experts_per_rank", 0) or 0) <= 0
        and "expert_slot_map" not in case.runtime.routing
        and (int(case.model.num_experts) <= 0
             or int(getattr(case.runtime.parallel, "ep", 1) or 1) <= 0
             or int(case.model.num_experts)
             % max(1, int(case.runtime.parallel.ep)) == 0)
        and not explicit_routing
        and not any(
            isinstance(item, Mapping) and "replica" in item
            for item in (case.runtime.routing.get("realizations") or ())
        )
    )
    model_programs = {}
    program_bindings = {}
    program_owners = {}
    cohort_topology = _build_topology(case)
    ep_degree = max(UNIFORM_COUNT_CARDINALITY, int(case.runtime.parallel.ep))
    if uniform_replicas and participants:
        first_r = participants[FIRST_PARTICIPANT]
        base_ops = _target_ops(
            case, views[first_r], scope, global_views=global_views,
            resource_fusion=resource_fusion, cohort_tokens=padded_cohort,
            phase=phase, replica=first_r,
            active_rows=active_rows_by_replica[first_r],
            source_rows_by_replica=source_rows_by_replica,
            routing_global_tokens=routing_global_tokens,
            routing_views=_routing_views_for(routing_views, first_r),
            cohort_views_by_replica=cohort_views_by_replica,
            physical_rank=_physical_rank_for(case, first_r, cohort_topology),
            query_position=query_position,
            model_programs=model_programs, layer_limit=layer_limit,
        )
        raw_graphs = {first_r: base_ops}
        for r in participants[UNIFORM_COUNT_CARDINALITY:]:
            owner_rank = int(
                _physical_rank_for(case, r, cohort_topology) or r) % ep_degree
            if scope == "model":
                # Only the graph is rebound here.  The attached program is
                # materialized from this replica's final cohort-bound graph on
                # first ``program(r)`` access.
                raw_graphs[r] = _rebind_ep_owner(
                    rebind_analysis_requests(base_ops, views[first_r], views[r]),
                    owner_rank)
                program_bindings[r] = (views[first_r], views[r])
                program_owners[r] = owner_rank
            else:
                req_ids = tuple(v.request_id for v in views[r])
                raw_graphs[r] = _rebind_ep_owner(
                    tuple(replace(op, requests=req_ids) if op.requests else op
                          for op in base_ops),
                    owner_rank)
    else:
        raw_graphs = {
            r: _target_ops(case, group, scope, global_views=global_views,
                           resource_fusion=resource_fusion, cohort_tokens=padded_cohort,
                           phase=phase, replica=r,
                           active_rows=active_rows_by_replica[r],
                           source_rows_by_replica=source_rows_by_replica,
                           routing_global_tokens=routing_global_tokens,
                           routing_views=_routing_views_for(routing_views, r),
                           cohort_views_by_replica=cohort_views_by_replica,
                           physical_rank=_physical_rank_for(case, r, cohort_topology),
                           query_position=query_position,
                           model_programs=model_programs,
                           layer_limit=layer_limit)
            for r, group in views.items()
        }
    from .local_projection_tp import lower_local_projection_tp_cohort
    graphs = bind_cohort_traffic(case, raw_graphs)
    graphs = lower_local_projection_tp_cohort(
        graphs, model=case.model, runtime=case.runtime, machine=case.machine)
    reuse_mode = str(getattr(case.runtime, "homogeneous_graph_reuse", "off"))
    if len(graphs) > 1 and reuse_mode != "off":
        # The same ordinary cohort producer proves and rebinds replicas; a
        # "require" declaration therefore fails before pricing an unproved
        # asymmetric cohort instead of silently running full duplicates.
        from .cohort_reuse import prepare_cohort_graphs
        prepared = prepare_cohort_graphs(case, graphs, reuse=reuse_mode)
        graphs = {replica: tuple(prepared.graphs[replica])
                  for replica in graphs}
    if scope == "body":
        return TargetGraphSet(graphs, kernels=case.runtime.kernels,
                              selection=selection, scope=scope)
    if uniform_replicas and participants:
        base_program = model_programs[participants[0]]
        program_seeds = {replica: base_program for replica in participants}
    else:
        program_seeds = dict(model_programs)
    return ModelGraphSet(graphs, programs=program_seeds,
                         bindings=program_bindings, owners=program_owners,
                         kernels=case.runtime.kernels, selection=selection)


def model_graphs(value, *, phase, tokens, context=0, sequences=1, scope="model",
                 selected_replica=0, cohort=None, **options):
    """Target graphs for a pure model declaration without a workload.

    The declaration supplies model/runtime/machine (or hardware); query
    geometry is explicit.  No RequestSpec, SLO, scheduler or fabricated
    workload is created.  Returns the existing named ``TargetGraphSet`` for
    body scope and the replica mapping for model scope.
    """
    return target_graphs(
        value, phase=phase, tokens=tokens, context=context, sequences=sequences,
        scope=scope, selected_replica=selected_replica, cohort=cohort,
        pure_model=True, **options)


def model_program(value, *, phase, tokens, context=0, sequences=1,
                  assembly="model", selected_replica=0, cohort=None,
                  apply_body_defaults=False, **options):
    """Assemble a pure-model forward through the existing ``Frontend.program``.

    This consumes explicit RequestView geometry and the selected runtime/model
    plan; it does not create a workload, service scheduler, or device state.
    The returned ``ModelProgram`` keeps typed ports, occurrence triggers,
    replay scope and named outputs for model/body consumers.
    """
    from mindie_sim.schema.workload import IterationPlan
    from .attention import AttentionPolicy
    from .frontend import Frontend

    case = parse_model(value)
    _check_geometry(phase, tokens, 1, context, "model")
    case, _selection = effective_body_declaration(
        case, phase, tokens=tokens, scope="model", enabled=apply_body_defaults)
    participants = _selected_group(case, selected_replica, cohort)
    counts = cohort_sequence_counts(sequences, participants)
    selected = (int(selected_replica) if selected_replica is not None
                else participants[0])
    views_by_replica = {
        replica: tuple(
            analysis_view(
                case, f"model{replica}:{index}", tokens, context, assembly,
                phase, index=index, samples=True,
                **_analysis_prefix_fields(case, index, context),
            ) for index in range(counts[replica]))
        for replica in participants
    }
    plan = IterationPlan(views_by_replica[selected])
    global_views = (tuple(view for replica in participants
                          for view in views_by_replica[replica])
                    if len(participants) > 1 else None)
    features = FeatureConfig.from_runtime(case.runtime, case.model,
                                           machine=case.machine)
    frontend = Frontend(
        attention_policy=AttentionPolicy(**case.runtime.attention_policy),
        features=features)
    program = frontend.program(
        plan, case.model, case.runtime, case.machine, assembly=assembly,
        global_views=global_views,
        global_tokens=(sum(v.q_len + v.draft_proposal_tokens
                           for v in global_views) if global_views else None),
        replica=selected, **options)
    return program

