"""Lower one scheduling iteration into executable operator work.

[Role] Assemble request views, draft work, the target body, and service events.
[Invariants] Direct and homogeneous lowering preserve scheduled rows and readiness.
[Timeouts & Locks] Bounded by active views and graph size. No locks or I/O.
[Side-Effect Policy] Updates the frontend's most recent lowering cache.
"""
from __future__ import annotations

from . import frontend as _frontend


_SPECULATIVE_ACCEPTANCE_OPS_PER_PROPOSAL = 6


def _reject_unmodeled_dbo(runtime, entry: str) -> None:
    if not bool(getattr(runtime, "enable_dbo", False)):
        return
    from mindie_sim.public.errors import UnmodeledCapabilityError

    raise UnmodeledCapabilityError(
        capability="dbo",
        entry=entry,
        code="unmodeled_in_op_dbo_pending_stage3",
        message=(
            "Whole-graph DBO microbatch splitting has been removed; "
            "in-operator DBO belongs to operators/grouped_moe and "
            "operators/mc2 (Stage 3)."
        ),
        affected_results=("latency", "operator_timeline", "throughput"),
        minimal_replay_input={"enable_dbo": True},
    )


# Service-only fragments. A pure-model assembly removes these from the selected
# program; the generic execution graph still prices them as ordinary nodes.
SERVICE_FAMILIES = frozenset({"sampling", "acceptance", "token_commit"})
_TRIGGERS = {
    "sampling": "per_invocation",
    "acceptance": "per_invocation_when_proposals",
    "token_commit": "per_request_per_invocation",
}


def _without_service_fragments(regions):
    """Remove service-only ops from ordered regions and reconnect consumers.

    Dependencies are rewired to the removed op's own predecessors, so a pure
    model keeps the same shared work and readiness edges without inventing a
    layer or a zero-work service marker. Regions are processed in execution
    order because draft/commit boundaries can cross a region edge.
    """
    aliases = {}
    result = []
    for region in regions:
        kept = []
        for op in region.ops:
            deps = tuple(dict.fromkeys(
                parent for dep in op.deps for parent in aliases.get(dep, (dep,))))
            if op.family in _frontend.SERVICE_FAMILIES and op.family != "acceptance":
                aliases[op.op_id] = deps
                continue
            kept.append(_frontend.replace(op, deps=deps))
        result.append(_frontend.replace(region, ops=tuple(kept)))
    return tuple(result)


ANALYTICAL_SELECTIONS = "analytical_implementations"


def mark_analytical_runtime(runtime, selections):
    """Retain policy-resolved analytical decisions on the immutable runtime.

    A policy may remove a selector after choosing its analytical fallback.
    Downstream source-default resolvers must see that decision instead of
    treating the absent binding as permission to install a native source.
    """
    names = {str(record.get("selector"))
             for record in (selections or ())
             if record.get("implementation") == "analytical"}
    names.discard("")
    if not names:
        return runtime
    metadata = dict(getattr(runtime, "metadata", ()) or ())
    previous = tuple(metadata.get(_frontend.ANALYTICAL_SELECTIONS, ()) or ())
    metadata[_frontend.ANALYTICAL_SELECTIONS] = tuple(sorted(set(previous) | names))
    return _frontend.replace(runtime, metadata=tuple(metadata.items()))


def _media_counts(runtime):
    import json

    raw = dict(runtime.metadata).get("media_tokens_by_request", "{}")
    return json.loads(raw) if isinstance(raw, str) else raw


def _append_service_tail(ops, plan, views, previous, *, stage, mode, requests):
    """Add speculative acceptance when needed, then add the commit boundary."""
    proposal_count = sum(view.draft_proposal_tokens for view in plan.requests)
    if proposal_count:
        accept = _frontend.vector_op(
            op_id="accept", deps=(previous,), stage=stage, layer=-1,
            requests=requests, elements=proposal_count, ops_per_element=_SPECULATIVE_ACCEPTANCE_OPS_PER_PROPOSAL,
            dtype="fp32", family="acceptance", phase=mode, recipe="speculative_acceptance",
        )
        ops.append(accept)
        previous = accept.op_id
    committed = _frontend.barrier_op(
        op_id="token_commit", deps=(previous,), stage=stage, layer=-1,
        requests=tuple(view.request_id for view in views if view.samples),
        phase=mode, recipe="token_commit", shape=(len(requests),),
    )
    ops.append(_frontend.replace(committed, family="token_commit"))

    return proposal_count, committed


def _selection_matches_any(op, records) -> bool:
    rec_name = op.recipe
    impl = dict(op.metadata).get("implementation") if op.metadata else None
    for record in records:
        sel_impl = (record.get("selected") or {}).get("implementation")
        if sel_impl and impl == sel_impl:
            return True
        if rec_name and rec_name in ((record.get("plan") or {}).get("recipes") or ()):
            return True
    return False


def _finalize_repeated_selections(frontend, raw_final_ops, selections, prev_sel_cache):
    """Attach implementation receipts to the expanded iteration graph."""
    reused_sel = False
    if prev_sel_cache is not None:
        prev_sel_in, prev_raw_ops, prev_final_sels, prev_ann_first = prev_sel_cache
        if (prev_sel_in == selections
                and len(prev_raw_ops) == len(raw_final_ops)
                and not any(
                    _frontend._selection_matches_any(p_op, selections)
                    or _frontend._selection_matches_any(n_op, selections)
                    for p_op, n_op in zip(prev_raw_ops, raw_final_ops)
                    if p_op is not n_op)):
            final_selections = prev_final_sels
            if raw_final_ops and raw_final_ops[0] is prev_raw_ops[0]:
                final_ops = (prev_ann_first, *raw_final_ops[1:])
            else:
                final_ops = _frontend.annotate_selections(raw_final_ops, final_selections)
            frontend._prev_work_ops = final_ops
            reused_sel = True
    if not reused_sel:
        final_selections = _frontend.selection_delivery(raw_final_ops, selections)
        final_ops = _frontend.annotate_selections(raw_final_ops, final_selections)
        final_ops = frontend._reuse_identical_ops(final_ops)
    sel_cache = (
        selections, raw_final_ops, final_selections,
        final_ops[0] if final_ops else None,
    )
    return final_ops, final_selections, sel_cache


class _FrontendIteration:
    @staticmethod
    def partition_views(views, runtime, padded, *, model=None, lane=None):
        degree = runtime.parallel.pcp
        if degree == 1 or not any(v.phase == "prefill" for v in views):
            return views, padded, False
        plans = getattr(model, "layer_plans", ()) if model is not None else ()
        attn_plans = [
            lp for lp in plans
            if lp.attention in {"gqa", "mla"} and lp.cache_kind != "recurrent_unmodeled"
        ]
        costs = [0.0] * degree
        for view in views:
            if view.phase != "prefill":
                continue
            if attn_plans:
                for lp in attn_plans:
                    win = (int(lp.window)
                           if int(lp.window) > 0 and not lp.compress_ratio and not lp.index_topk
                           else 0)
                    split = _frontend.split_pcp_pairs(view.q_len, view.context, degree, window=win)
                    costs = [old + current for old, current in zip(costs, split.per_rank)]
            else:
                split = _frontend.split_pcp_pairs(view.q_len, view.context, degree, window=0)
                costs = [old + current for old, current in zip(costs, split.per_rank)]
        rank = max(range(degree), key=costs.__getitem__) if lane is None else lane
        ranges = [_frontend.pcp_rank_chunk_ranges(view.q_len, degree, rank) for view in views]
        local = tuple(_frontend.replace(view, q_len=sum(end - start for start, end in parts),
                              global_q_len=view.q_len, query_ranges=parts)
                      if view.phase == "prefill" else view for view, parts in zip(views, ranges))
        extra = max(0, padded - sum(view.q_len for view in views))
        padding = sum(hi - lo for lo, hi in _frontend.pcp_rank_chunk_ranges(extra, degree, rank))
        return local, sum(view.q_len for view in local) + padding, True

    @staticmethod
    def pcp_cohort_tokens(views, local_views, local_tokens, runtime, global_tokens=None, global_views=None):
        p = runtime.parallel
        if global_tokens is None and not getattr(p, "ep_spans_pcp", False):
            return None
        pcp_in_ep = min(p.pcp, max(1, p.ep // p.tp)) if getattr(p, "ep_spans_pcp", False) else 1
        gviews = global_views if global_views is not None else views
        expand = global_views is not None and (
            global_tokens is None or sum(v.q_len for v in gviews) < global_tokens)
        q_of = lambda v: v.q_len + (v.draft_proposal_tokens if expand else 0)
        dec_g = sum(q_of(v) for v in gviews if v.phase != "prefill")
        pf_views = tuple(v for v in gviews if v.phase == "prefill")
        pf_unpadded = sum(q_of(v) for v in pf_views)
        rank_totals = [
            sum(sum(e - s for s, e in _frontend.pcp_rank_chunk_ranges(q_of(v), p.pcp, r))
                for v in pf_views)
            for r in range(p.pcp)
        ]
        pf_local_sum = max(
            (sum(rank_totals[g:g + pcp_in_ep]) for g in range(0, p.pcp, pcp_in_ep)),
            default=0,
        )
        pad_extra = max(0, (global_tokens - dec_g - pf_unpadded) if global_tokens is not None else 0)
        return max(local_tokens, dec_g + pf_local_sum + (pad_extra * pcp_in_ep + p.pcp - 1) // p.pcp)

    def program(self, plan, model, runtime, machine, *, assembly="service", **options):
        """Return executable model work with typed ports and occurrence triggers.

        ``assembly="model"`` selects only the forward plan and its necessary
        implementation support. ``assembly="service"`` keeps sampling,
        acceptance and token commit as declared service fragments.
        """
        work = self(plan, model, runtime, machine, assembly=assembly, **options)
        counts = _frontend.Counter(op.family for op in work.ops)
        triggers = tuple((family, counts[family], _frontend._TRIGGERS[family])
                         for family in sorted(_frontend.SERVICE_FAMILIES) if counts[family])
        return _frontend.ModelProgram(
            work=work,
            ports=_frontend._ports_for_work(work),
            triggers=triggers,
            assembly=assembly,
            layer_scope=("per_model_layer" if any(op.layer >= 0 for op in work.ops)
                         else "none"),
        )

    def __call__(self, plan, model, runtime, machine, *, global_tokens=None, global_views=None,
                 resource_fusion=None, replica=0, source_rows_by_replica=None,
                 routing_global_tokens=None, routing_views=None,
                 cohort_views_by_replica=None,
                 physical_rank=None, explicit_physical_rank: bool = False,
                 query_position=None, assembly="service",
                 dispatch_binding=None, dispatch_fast_step: bool = False):
        # ``dispatch_binding`` is the T14 PP/EP microbatch envelope (original
        # slot/version/replica). The ordinary producer keeps using the views in
        # ``plan``; a cached template owner can read this envelope without
        # inferring a slot from local list positions.
        requested_rank = physical_rank
        _frontend._reject_unmodeled_dbo(runtime, "frontend")
        if assembly not in {"model", "service"}:
            raise _frontend.ModelingError(f"assembly_scope_unknown:{assembly}")
        if resource_fusion is None:
            resource_fusion = self.resource_fusion
        s_env = self._prev_call_env
        if (s_env is not None and s_env[_frontend._ENV_MODEL] is model
                and (s_env[_frontend._ENV_RUNTIME] is runtime
                     or s_env[_frontend._ENV_RUNTIME] == runtime)
                and s_env[_frontend._ENV_MACHINE] is machine
                and s_env[_frontend._ENV_REPLICA] == replica
                and s_env[_frontend._ENV_FEATURES] is self.features):
            _, _, _, _, _, model, auto_rank, features = s_env
            if physical_rank is None:
                physical_rank = auto_rank
        else:
            raw_model = model
            model = _frontend.normalize_model(model)
            auto_rank = None
            try:
                from .topology import Topology
                auto_rank = Topology(machine, runtime.parallel).ranks(
                    replica, 0)[0]
            except Exception:
                auto_rank = None
            features = (self.features if self.features is not None else
                        _frontend.FeatureConfig.from_runtime(runtime, model, machine=machine))
            self._prev_call_env = (
                raw_model, runtime, machine, replica, self.features,
                model, auto_rank, features,
            )
            if physical_rank is None:
                physical_rank = auto_rank
        # An empty local query set can still receive expert rows. Its execution
        # phase belongs to the active cohort, not a fictitious "idle" kernel.
        mode = _frontend.selected_mode(plan if plan.requests else _frontend.replace(
            plan, requests=tuple(global_views or ())))
        if assembly == "model":
            from math import floor
            mean = float(runtime.mtp_acceptance_length)
            plan = _frontend.replace(plan, requests=tuple(
                _frontend.replace(view,
                        draft_advanced_tokens=min(
                            view.q_len + view.draft_proposal_tokens,
                            max(1, floor((index + 1) * mean) - floor(index * mean))))
                if view.draft_proposal_tokens and not view.draft_advanced_tokens
                else view for index, view in enumerate(plan.requests)))
        views = tuple(_frontend.replace(view, q_len=view.q_len + view.draft_proposal_tokens)
                      for view in plan.requests)
        valid = sum(view.logical_q_len for view in views)
        physical = sum(view.q_len for view in views)
        bucket, scope = self.graph.select(views, runtime) if views else (None, "none")
        padded = max(physical, bucket or 0)
        declared_runtime = runtime
        runtime, selections = _frontend.resolve_implementations(
            runtime, views, padded_tokens=padded, replay_scope=scope,
            global_views=global_views)
        runtime = _frontend.mark_analytical_runtime(runtime, selections)
        # A speculative decode leaves state publication to the service
        # acceptance boundary. The homogeneous fast path splices one layer
        # template across copies, which would lose those deferred intents.
        deferred_commits = bool(_frontend.recover_ssm_enabled(runtime)) and any(
            view.phase == "decode" and view.q_len > 1
            and not getattr(view, "padding", False) for view in views)
        requests = tuple(view.request_id for view in views)
        ops = []
        entry = _frontend.barrier_op(
            op_id="iteration_input", deps=(), stage=0, layer=-1, requests=requests, phase=mode,
            recipe="input_ready", shape=(valid, model.hidden_size),
        )
        previous = entry.op_id
        before, after = [], []
        if any(view.draft_proposal_tokens for view in (global_views or plan.requests)):
            from .speculation import draft_program
            before, previous = draft_program(
                self, model, declared_runtime, plan.requests, previous,
                global_views=global_views, replica=replica, replay_scope=scope,
                machine=machine)
        media_counts = {str(request): int(count) for request, count in
                        _frontend._media_counts(runtime).items()}
        for index, view in enumerate(views):
            if view.media_encoded or view.phase != "prefill" or not media_counts.get(view.request_id):
                continue
            media = _frontend.lower_media(
                model=model, runtime=runtime, parallel=runtime.parallel, quant=model.quant,
                mode=mode, views=(view,), media_tokens_override=media_counts[view.request_id],
            )
            rename = {op.op_id: f"media{index}:{op.op_id}" for op in media.ops}
            ops.extend(_frontend.replace(op, op_id=rename[op.op_id],
                               deps=tuple(rename[dep] for dep in op.deps) or (previous,))
                       for op in media.ops)
            previous = rename[media.last]
        from .rank_ledger import selected_lane
        local_views, local_tokens, pcp_prefill = self.partition_views(
            views, runtime, padded, model=model,
            lane=selected_lane(runtime.parallel, machine, replica, requested_rank))
        if valid:
            embedding = _frontend.vector_op(
                op_id="embedding", deps=(previous,), stage=0, layer=-1, requests=requests,
                elements=local_tokens * model.hidden_size, ops_per_element=1,
                dtype=model.activation_dtype, family="embedding", phase=mode, recipe="embedding",
            )
            ops.append(embedding)
            previous = embedding.op_id
        pcp_cohort = runtime.parallel.pcp > 1 and any(
            v.phase == "prefill" for v in (global_views or views))
        body_global_tokens = (
            self.pcp_cohort_tokens(views, local_views, local_tokens, runtime,
                                   global_tokens=global_tokens, global_views=global_views)
            if pcp_cohort else global_tokens
        )
        eff_routing_views = routing_views if routing_views is not None else views
        eff_query_pos = (
            query_position if query_position is not None
            else next((v.context for v in (routing_views or views) if not v.padding), None)
        )
        preparation_views = tuple(view for view in plan.requests if view.phase == "prefill"
                                  and view.draft_prepare)
        global_preparation = tuple(view for view in (global_views or plan.requests)
                                   if view.phase == "prefill" and view.draft_prepare)
        has_after_draft = bool(
            dict(runtime.metadata).get("draft_mode")
            and (preparation_views or global_views and global_preparation)
        )
        fast_result = None if deferred_commits else self._try_lower_homogeneous(
            entry=entry, pre_ops=tuple(ops), body_ready=previous,
            plan=plan, model=model, runtime=runtime,
            views=views, local_views=local_views,
            valid=valid, padded=padded, local_tokens=local_tokens,
            bucket=bucket, scope=scope, mode=mode, requests=requests,
            selections=selections, resource_fusion=resource_fusion,
            assembly=assembly, replica=replica, features=features,
            body_global_tokens=body_global_tokens,
            source_rows_by_replica=source_rows_by_replica,
            routing_global_tokens=routing_global_tokens,
            routing_views=eff_routing_views,
            physical_rank=physical_rank,
            query_position=eff_query_pos,
            before=before, has_after_draft=has_after_draft,
            pcp_prefill=pcp_prefill,
            dispatch_fast_step=dispatch_fast_step,
            machine=machine,
        )
        if fast_result is not None:
            return fast_result
        body, previous = self.body(model, runtime, local_views, ready=previous, tokens=local_tokens,
                                   mode=mode, global_tokens=body_global_tokens, replica=replica,
                                   features=features,
                                   source_rows_by_replica=source_rows_by_replica,
                                   routing_global_tokens=routing_global_tokens,
                                   routing_views=eff_routing_views,
                                   cohort_views_by_replica=cohort_views_by_replica,
                                   physical_rank=physical_rank,
                                   explicit_physical_rank=(explicit_physical_rank or requested_rank is not None),
                                   query_position=eff_query_pos,
                                   include_recover_commits=(assembly == "service" or any(
                                       view.draft_proposal_tokens for view in plan.requests)),
                                   machine=machine)
        ops.extend(body)
        if pcp_prefill:
            if runtime.final_norm_scope == "all":
                finals, previous = self.final_mixer(model, previous, prefix="", mode=mode,
                    stage=runtime.parallel.pp + _frontend.FINAL_LAYER, tokens=local_tokens, requests=requests)
                ops.extend(finals)
            select = self.select_hidden(model, views, previous, mode=mode, stage=runtime.parallel.pp - 1)
            ops.append(select)
            previous = select.op_id
            rows = sum(v.q_len if v.phase == "decode" else 1 for v in views if v.samples)
            gather = _frontend.comm_op(
                op_id="pcp_hidden_gather", deps=(previous,), stage=runtime.parallel.pp - 1,
                layer=-1, requests=requests, family="allgather", phase=mode,
                recipe="pcp_hidden_gather", link_bytes=rows * model.hidden_size * _frontend.dtype_bytes(model.activation_dtype)
                * (runtime.parallel.pcp - 1) / runtime.parallel.pcp,
                shape=(rows, model.hidden_size), metadata=(("group_axis", "pcp"),))
            ops.append(gather)
            previous = gather.op_id
        heads, previous = self.logits(
            model, runtime, views, previous, mode=mode,
            input_tokens=rows if pcp_prefill else padded,
            normalized=pcp_prefill and runtime.final_norm_scope == "all")
        ops.extend(heads)
        proposal_count, committed = _append_service_tail(
            ops, plan, views, previous, stage=runtime.parallel.pp - 1,
            mode=mode, requests=requests)
        if deferred_commits:
            intents = tuple(op for op in ops if _frontend.is_recover_ssm_intent(op))
            if intents and assembly == "model" and not proposal_count:
                ops = list(_frontend._strip_recover_intents(ops))
            elif intents:
                spec_views = tuple(
                    view for view in views
                    if view.phase == "decode" and view.q_len > 1
                    and not bool(getattr(view, "padding", False)))
                anchor = next(
                    (op for op in reversed(ops)
                     if op.family in {"acceptance", "sampling"}), None)
                if anchor is None:
                    raise _frontend.ModelingError(
                        "recover_ssm_commit_requires_acceptance_boundary")
                kept = [op for op in ops if not _frontend.is_recover_ssm_intent(op)]
                bound = _frontend.bind_recover_ssm_commits(
                    intents, views=spec_views,
                    acceptance_events={view.request_id: anchor.op_id
                                       for view in spec_views},
                    state_bindings=None, runtime=runtime)
                position = next(
                    index for index, op in enumerate(kept)
                    if op.op_id == anchor.op_id) + 1
                kept[position:position] = bound
                commit_ids = tuple(op.op_id for op in bound)
                ops = [
                    _frontend.replace(op, deps=tuple(dict.fromkeys(
                        (*op.deps, *commit_ids))))
                    if op.family == "token_commit" else op
                    for op in kept
                ]
        if has_after_draft:
            from .speculation import draft_program
            after, _ = draft_program(
                self, model, declared_runtime, preparation_views, committed.op_id,
                global_views=global_preparation if global_views else None, prefill=True,
                replica=replica, replay_scope=scope, machine=machine)
        # A capture descriptor is distinct from each kernel's active/sliced
        # tensor. Preserve both so native responses cannot alias two buckets
        # merely because their logical attention shapes are equal.
        ops = _frontend.annotate_geometry(ops, active_rows=valid, physical_rows=padded)
        regions = (*before, _frontend.LoweringRegion(tuple(ops), runtime, model.activation_dtype,
                                          local_tokens, selections), *after)
        if assembly == "model":
            # Scope selection happens before lowering so service-only nodes do
            # not enter native fusion, capture or implementation decisions.
            regions = _frontend._without_service_fragments(regions)
        ops, selections = _frontend.lower_regions(
            regions, graph=self.graph, replay_scope=scope, runtime=runtime,
            prefix=(entry,), resource_fusion=resource_fusion)
        ops = self._reuse_identical_ops(ops)
        return _frontend.IterationWork(
            ops=ops,
            plan=plan, stage_count=runtime.parallel.pp,
            valid_tokens=valid, padded_tokens=padded, graph_bucket=bucket, mode=mode,
            recipe="structural", replay_scope=scope,
            implementation_selections=selections,
        )

    def _reuse_identical_ops(self, ops: tuple) -> tuple:
        prev = self._prev_work_ops
        if prev is not None and len(prev) == len(ops):
            reused = tuple(
                p_op if p_op == n_op else n_op
                for p_op, n_op in zip(prev, ops)
            )
            self._prev_work_ops = reused
            return reused
        self._prev_work_ops = ops
        return ops
