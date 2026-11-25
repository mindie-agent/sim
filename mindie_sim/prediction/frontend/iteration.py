"""Lower one scheduling iteration into executable operator work.

[Role] Assemble request views, draft work, the target body, and service events.
[Invariants] Direct and homogeneous lowering preserve scheduled rows and readiness.
[Timeouts & Locks] Bounded by active views and graph size. No locks or I/O.
[Side-Effect Policy] Updates the frontend's most recent lowering cache.
"""
from __future__ import annotations

from mindie_sim.prediction import frontend as _frontend
from mindie_sim.prediction.frontend.body import (
    _expand_target_cohort_views,
    _resolve_hc_final_collapse,
    _verify_query_len,
)


_SPECULATIVE_ACCEPTANCE_OPS_PER_PROPOSAL = 6
_NON_NEURAL_SPEC_METHODS = frozenset({"ngram", "ngram_gpu"})
_NGRAM_HISTORY_TOKEN_BYTES = 4
_NGRAM_MATCH_INDEX_BYTES = 8
_NGRAM_DEFAULT_MIN_N = 2
_NGRAM_DEFAULT_MAX_N = 4


def _spec_method(runtime) -> str:
    meta = dict(getattr(runtime, "metadata", ()) or ())
    raw = meta.get("spec_method") or meta.get("draft_mode") or ""
    return str(raw).strip().lower()


def _is_non_neural_spec(runtime) -> bool:
    return _spec_method(runtime) in _NON_NEURAL_SPEC_METHODS


def _restore_target_pcp_hidden(
    frontend, ops, previous, *, views, model, runtime, mode, local_tokens, padded, requests,
) -> tuple[str, int]:
    """Gather and unpermute full physical PCP rows before target logits row selection."""
    from mindie_sim.prediction.sharding.context_parallel import (
        emit_pcp_hidden_gather_restore,
        pcp_uniform_peer_rows,
    )
    final_stage = runtime.parallel.pp + _frontend.FINAL_LAYER
    if runtime.final_norm_scope == "all":
        finals, previous = frontend.final_mixer(
            model, previous, prefix="", mode=mode,
            stage=final_stage, tokens=local_tokens, requests=requests,
        )
        ops.extend(finals)
    pcp = int(runtime.parallel.pcp)
    per_peer_rows, global_rows = pcp_uniform_peer_rows(
        views, pcp, local_rows=local_tokens,
    )
    restored_rows = max(global_rows, int(padded))
    extra = max(0, int(padded) - sum(int(v.q_len) for v in views))
    if extra > 0:
        decode_rows = sum(int(v.q_len) for v in views if v.phase != "prefill")
        prefill_views = tuple(v for v in views if v.phase == "prefill")
        rank_totals = (
            decode_rows
            + sum(
                sum(hi - lo for lo, hi in _frontend.pcp_rank_chunk_ranges(
                    int(getattr(v, "global_q_len", 0) or v.q_len), pcp, r))
                for v in prefill_views
            )
            + sum(hi - lo for lo, hi in _frontend.pcp_rank_chunk_ranges(extra, pcp, r))
            for r in range(pcp)
        )
        per_peer_rows = max(per_peer_rows, max(rank_totals, default=0))
    per_peer_rows = max(per_peer_rows, (restored_rows + pcp - 1) // pcp)
    hc_mult, collapse_mode = _resolve_hc_final_collapse(model)
    has_lowrank = any(getattr(plan, "hc_lowrank", 0) > 0 for plan in model.layer_plans)
    gather_hidden = (
        hc_mult * model.hidden_size
        if runtime.final_norm_scope != "all" and hc_mult > 1 and (has_lowrank or collapse_mode != "none")
        else model.hidden_size
    )
    pcp_ops: list = []
    previous = emit_pcp_hidden_gather_restore(
        pcp_ops, prefix="", gather_name="pcp_hidden_gather",
        restore_name="pcp_hidden_restore", stage=final_stage,
        layer=_frontend.FINAL_LAYER, requests=requests, mode=mode,
        dep=previous, local_rows=local_tokens, global_rows=restored_rows,
        hidden=gather_hidden, pcp=int(runtime.parallel.pcp),
        act_dtype=model.activation_dtype, runtime=runtime,
        source="vllm_ascend.attention.context_parallel.pcp_manager",
        per_peer_rows=per_peer_rows,
    )
    rename = {
        op.op_id: op.op_id.lstrip(".")
        for op in pcp_ops if op.op_id.startswith(".")
    }
    if rename:
        pcp_ops = [
            _frontend.replace(
                op,
                op_id=rename.get(op.op_id, op.op_id),
                deps=tuple(rename.get(dep, dep) for dep in op.deps),
            )
            for op in pcp_ops
        ]
        previous = rename.get(previous, previous)
    ops.extend(pcp_ops)
    return previous, restored_rows


def _is_view_eos(view, output_tokens_by_req: dict[str, int]) -> bool:
    """Detect whether a prefill or decode view reaches EOS on the current step."""
    if getattr(view, "padding", False):
        return False
    req_id = str(getattr(view, "request_id", ""))
    target_out = output_tokens_by_req.get(req_id)
    if target_out is None or int(target_out) <= 0:
        return False
    phase = str(getattr(view, "phase", ""))
    if phase == "prefill":
        return bool(getattr(view, "samples", False)) and int(target_out) <= 1
    if phase != "decode":
        return False
    prompt = int(getattr(view, "prompt_tokens", 0) or 0)
    if prompt <= 0:
        return False
    generated_before = max(0, int(getattr(view, "context", 0) or 0) - prompt) + 1
    advance = max(1, int(getattr(view, "draft_advanced_tokens", 0) or 0))
    return (generated_before + advance) >= int(target_out)


def _configured_spec_k(runtime) -> int:
    """Configured speculative proposal count from runtime metadata/DSpark (independent of prior p_i)."""
    from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens

    meta = dict(getattr(runtime, "metadata", ()) or ())
    explicit_method = str(meta.get("spec_method") or "").strip().lower()
    if explicit_method == "none" or str(meta.get("draft_mode") or "").strip().lower() == "none":
        return 0
    if explicit_method == "medusa":
        raise _frontend.ModelingError("unsupported_speculative_method_on_ascend_v2:medusa")
    resolved_k = 0
    for key in ("draft_proposal_tokens", "num_speculative_tokens"):
        raw = meta.get(key)
        if raw is not None and raw != "":
            resolved_k = int(raw)
            break
    else:
        dspark = getattr(runtime, "dspark", None)
        if dspark is not None:
            resolved_k = int(dspark.step_spec().draft_len)
        else:
            raw_spec = meta.get("speculative")
            if isinstance(raw_spec, str) and raw_spec.strip():
                import json
                raw_spec = json.loads(raw_spec)
            if isinstance(raw_spec, dict) and raw_spec.get("draft_len") is not None:
                resolved_k = int(raw_spec["draft_len"])
    if resolved_k != 0:
        return validate_fia_speculative_tokens(resolved_k)
    return 0


def _next_proposal_width_for_view(view, *, runtime, output_tokens_by_req: dict[str, int]) -> int:
    """Determine next-step draft proposal width independently of the previous verify width."""
    if getattr(view, "padding", False) or _spec_method(runtime) == "none":
        return 0
    if _is_view_eos(view, output_tokens_by_req):
        return 0
    explicit = getattr(view, "next_draft_proposal_tokens", None)
    if explicit is not None:
        from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens
        return validate_fia_speculative_tokens(int(explicit))
    cfg_k = _configured_spec_k(runtime)
    phase = str(getattr(view, "phase", ""))
    req_id = str(getattr(view, "request_id", ""))
    target_out = output_tokens_by_req.get(req_id)
    if phase == "prefill":
        if not bool(getattr(view, "samples", False)):
            return 0
        cap = cfg_k
        if target_out is not None and int(target_out) > 0:
            cap = min(cap, max(0, int(target_out) - 2))
        return max(0, cap)
    if phase == "decode":
        base_k = cfg_k if cfg_k > 0 else max(0, int(getattr(view, "draft_proposal_tokens", 0) or 0))
        if base_k <= 0:
            return 0
        from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens
        base_k = validate_fia_speculative_tokens(base_k)
        prompt = int(getattr(view, "prompt_tokens", 0) or 0)
        if prompt > 0 and target_out is not None and int(target_out) > 0:
            generated_before = max(0, int(getattr(view, "context", 0) or 0) - prompt) + 1
            advance = max(1, int(getattr(view, "draft_advanced_tokens", 0) or 0))
            rem_after = max(0, int(target_out) - (generated_before + advance))
            return min(base_k, max(0, rem_after - 1))
        return base_k
    return 0


def _validate_decode_ready_state(runtime, mode: str) -> None:
    """Verify decode-only pd_decode runs carry a valid ReadyStateAssumption seed."""
    if mode != "decode":
        return
    meta = dict(getattr(runtime, "metadata", ()) or ())
    spine = meta.get("single_spine")
    if isinstance(spine, dict):
        scenario = spine.get("scenario_contract")
        if isinstance(scenario, dict) and scenario.get("workload_mode") == "pd_decode":
            ready = scenario.get("ready_state")
            if not isinstance(ready, dict) or int(ready.get("context_tokens", 0) or 0) < 1 or not str(ready.get("layout", "")).strip():
                raise _frontend.ModelingError("decode_only_requires_ready_state_assumption")
    if str(meta.get("role", "")).strip().lower() == "pd_decode" and "ready_state" in meta:
        ready = meta.get("ready_state")
        if ready is None:
            raise _frontend.ModelingError("decode_only_requires_ready_state_assumption")
        if isinstance(ready, dict) and (int(ready.get("context_tokens", 0) or 0) < 1 or not str(ready.get("layout", "")).strip()):
            raise _frontend.ModelingError("decode_only_requires_ready_state_assumption")


def _recover_state_owner(op) -> str:
    meta = dict(getattr(op, "metadata", ()) or ())
    return str(meta.get("state_owner") or meta.get("recurrent_state_owner") or f"layer{op.layer}")


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


def _append_non_neural_ngram_propose(
    ops: list,
    *,
    spec_method: str,
    declared_runtime,
    runtime,
    committed,
    ngram_propose_views,
    next_proposal_widths,
    mode: str,
) -> list:
    """Append NgramGPUSpeculator scan+finalize ops (MRV2 default for ngram & ngram_gpu) or MRV1 host metadata."""
    rt_meta = dict(getattr(declared_runtime, "metadata", ()) or ())
    explicit_window = rt_meta.get("ngram_history_window")
    window_cap = max(1, int(explicit_window)) if (explicit_window is not None and int(explicit_window) > 0) else 0
    min_n = max(1, int(rt_meta.get("prompt_lookup_min", _NGRAM_DEFAULT_MIN_N) or _NGRAM_DEFAULT_MIN_N))
    max_n = max(min_n, int(rt_meta.get("prompt_lookup_max", _NGRAM_DEFAULT_MAX_N) or _NGRAM_DEFAULT_MAX_N))
    num_n = max_n - min_n + 1
    hist_lens: list[int] = []
    for v in ngram_propose_views:
        if str(getattr(v, "phase", "")) == "prefill":
            base_len = int(getattr(v, "context", 0) or 0) + int(getattr(v, "global_q_len", 0) or getattr(v, "q_len", 0) or 0) + 1
        else:
            adv = max(1, int(getattr(v, "draft_advanced_tokens", 0) or 0))
            base_len = int(getattr(v, "context", 0) or 0) + adv
        h_len = max(1, base_len)
        if window_cap > 0:
            h_len = min(h_len, window_cap)
        hist_lens.append(h_len)
    next_pairs = tuple(
        (v.request_id, int(next_proposal_widths.get(v.request_id, 0)))
        for v in ngram_propose_views
    )
    hist_pairs = tuple((v.request_id, h) for v, h in zip(ngram_propose_views, hist_lens))
    use_gpu_speculator = (
        spec_method == "ngram_gpu"
        or rt_meta.get("model_runner_v2", True) is not False
    )
    if use_gpu_speculator:
        stage = runtime.parallel.pp - 1
        reqs = tuple(v.request_id for v in ngram_propose_views)
        batch = len(ngram_propose_views)
        total_hist = sum(hist_lens)
        scan_comparisons = sum(
            max(0, h_len - n) * n
            for h_len in hist_lens
            for n in range(min_n, max_n + 1)
        )
        scan_read = float(total_hist * _NGRAM_HISTORY_TOKEN_BYTES)
        scan_write = float(batch * num_n * _NGRAM_MATCH_INDEX_BYTES)
        k_max = max(
            1,
            _configured_spec_k(declared_runtime),
            max((k for _, k in next_pairs), default=0),
        )
        finalize_read = float(batch * (num_n * _NGRAM_MATCH_INDEX_BYTES + k_max * _NGRAM_HISTORY_TOKEN_BYTES))
        finalize_write = float(batch * (k_max + 1) * _NGRAM_HISTORY_TOKEN_BYTES)
        scan_op = _frontend.vector_op(
            op_id="ngram_gpu_scan", deps=(committed.op_id,),
            stage=stage, layer=-1, requests=reqs,
            elements=max(1, total_hist), ops_per_element=0,
            dtype="int32", family="elementwise", phase=mode,
            recipe="ngram_gpu_scan",
            read_bytes_override=scan_read, write_bytes_override=scan_write,
            shape=(batch, max(hist_lens, default=1)),
            metadata=(
                ("ngram_speculator", "NgramGPUSpeculator"),
                ("ngram_stage", "scan"),
                ("ngram_history_lengths", hist_pairs),
                ("prompt_lookup_min", min_n),
                ("prompt_lookup_max", max_n),
            ),
        )
        scan_op = _frontend.replace(scan_op, vector_ops=float(scan_comparisons))
        finalize_op = _frontend.vector_op(
            op_id="ngram_gpu_propose", deps=(scan_op.op_id,),
            stage=stage, layer=-1, requests=reqs,
            elements=batch * (num_n + k_max), ops_per_element=1,
            dtype="int32", family="elementwise", phase=mode,
            recipe="ngram_gpu_propose",
            read_bytes_override=finalize_read, write_bytes_override=finalize_write,
            shape=(batch, k_max),
            metadata=(
                ("ngram_speculator", "NgramGPUSpeculator"),
                ("ngram_stage", "finalize"),
                ("ngram_next_proposals", next_pairs),
                ("ngram_history_lengths", hist_pairs),
                ("prompt_lookup_min", min_n),
                ("prompt_lookup_max", max_n),
            ),
        )
        ops.extend((scan_op, finalize_op))
        return ops
    return [
        _frontend.replace(
            op,
            metadata=op.metadata + (
                ("ngram_host_propose", True),
                ("ngram_next_proposals", next_pairs),
                ("ngram_history_lengths", hist_pairs),
            ),
        )
        if op.op_id == committed.op_id else op
        for op in ops
    ]


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

    def _resolve_call_environment(self, model, runtime, machine, replica, physical_rank):
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
            if (
                machine is not None
                and getattr(runtime, "parallel", None) is not None
                and getattr(machine, "dies", None)
            ):
                from mindie_sim.prediction.topology import Topology
                topo = Topology(machine, runtime.parallel)
                ranks_for_replica = topo.ranks(replica, 0) if 0 <= int(replica) < int(runtime.parallel.dp) else ()
                if ranks_for_replica:
                    auto_rank = ranks_for_replica[0]
            features = (self.features if self.features is not None else
                        _frontend.FeatureConfig.from_runtime(runtime, model, machine=machine))
            self._prev_call_env = (
                raw_model, runtime, machine, replica, self.features,
                model, auto_rank, features,
            )
            if physical_rank is None:
                physical_rank = auto_rank
        return model, physical_rank, features

    def _lower_iteration_inputs(
        self, plan, model, runtime, declared_runtime, machine, views, *,
        global_views, replica, requested_rank, mode, scope, valid, padded, requests,
    ):
        ops = []
        entry = _frontend.barrier_op(
            op_id="iteration_input", deps=(), stage=0, layer=-1, requests=requests, phase=mode,
            recipe="input_ready", shape=(valid, model.hidden_size),
        )
        previous = entry.op_id
        for pub in dict.fromkeys(
            str(getattr(v, "previous_state_publication", "") or "")
            for v in views
            if not getattr(v, "padding", False) and str(getattr(v, "previous_state_publication", "") or "")
        ):
            pub_reqs = tuple(
                str(v.request_id) for v in views
                if not getattr(v, "padding", False)
                and str(getattr(v, "previous_state_publication", "") or "") == pub
            )
            ops.append(_frontend.barrier_op(
                op_id=f"state_input.{pub}", deps=(entry.op_id,), stage=0, layer=-1,
                requests=pub_reqs, phase=mode, recipe="state_input_ready", shape=(len(pub_reqs),),
                metadata=(("state_publication_handle", pub),),
            ))
        before = []
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
        from mindie_sim.prediction.execution.rank_ledger import selected_lane
        local_views, local_tokens, pcp_prefill = self.partition_views(
            views, runtime, padded, model=model,
            lane=selected_lane(runtime.parallel, machine, replica, requested_rank))
        if valid or (scope == "full" and local_tokens > 0):
            embedding = _frontend.vector_op(
                op_id="embedding", deps=(previous,), stage=0, layer=-1, requests=requests,
                elements=local_tokens * model.hidden_size, ops_per_element=1,
                dtype=model.activation_dtype, family="embedding", phase=mode, recipe="embedding",
            )
            ops.append(embedding)
            previous = embedding.op_id
            tp = int(runtime.parallel.tp)
            if tp > 1 and local_tokens > 0:
                reduction = _frontend.replace(
                    _frontend.comm_op(
                        op_id="embedding_allreduce", deps=(previous,), stage=0, layer=-1,
                        requests=requests, family="allreduce", phase=mode, recipe="ring_allreduce",
                        link_bytes=(
                            2 * (tp - 1) / tp * local_tokens * model.hidden_size
                            * _frontend.dtype_bytes(model.activation_dtype)
                        ),
                        shape=(local_tokens, model.hidden_size),
                        metadata=(("group_axis", "tp"), ("steps", 2 * (tp - 1))),
                    ),
                    dtype=model.activation_dtype,
                )
                ops.append(reduction)
                previous = reduction.op_id
        return entry, before, ops, previous, local_views, local_tokens, pcp_prefill

    def _lower_iteration_outputs(
        self, ops, previous, *, entry, plan, views, model, runtime, declared_runtime,
        machine, before, assembly, replica, requested_rank, scope, mode, valid, padded, bucket,
        local_tokens, requests, pcp_prefill, deferred_commits, has_prefill_draft,
        has_decode_draft, decode_proposal_views, global_decode_proposals,
        preparation_views, global_views, global_preparation, selections, resource_fusion,
        next_proposal_widths, ngram_propose_views,
    ):
        restored_rows = padded
        if pcp_prefill:
            previous, restored_rows = _restore_target_pcp_hidden(
                self, ops, previous, views=views, model=model, runtime=runtime,
                mode=mode, local_tokens=local_tokens, padded=padded, requests=requests,
            )
        heads, previous = self.logits(
            model, runtime, views, previous, mode=mode,
            input_tokens=restored_rows if pcp_prefill else padded,
            normalized=pcp_prefill and runtime.final_norm_scope == "all")
        ops.extend(heads)
        proposal_count, committed = _append_service_tail(
            ops, plan, views, previous, stage=runtime.parallel.pp - 1,
            mode=mode, requests=requests)
        accept_op = next(
            (op for op in reversed(ops) if op.family == "acceptance"), None)
        commit_ids: tuple[str, ...] = ()
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
                raw_bindings = dict(getattr(runtime, "metadata", ()) or ()).get(
                    "recover_ssm_state_bindings", {})
                rt_state_bindings = dict(raw_bindings) if isinstance(raw_bindings, dict) else {}
                state_bindings = {
                    _recover_state_owner(intent): str(rt_state_bindings[_recover_state_owner(intent)])
                    for intent in intents
                    if _recover_state_owner(intent) in rt_state_bindings
                }
                bound = _frontend.bind_recover_ssm_commits(
                    intents, views=spec_views,
                    acceptance_events={view.request_id: anchor.op_id
                                       for view in spec_views},
                    state_bindings=state_bindings, runtime=runtime,
                    machine=machine, shared_kda_launch=True)
                position = next(
                    index for index, op in enumerate(kept)
                    if op.op_id == anchor.op_id) + 1
                kept[position:position] = bound
                pub_ids = tuple(
                    op.op_id for op in bound
                    if dict(op.metadata).get("recover_ssm_stage") == "state_publish"
                )
                commit_ids = pub_ids or tuple(op.op_id for op in bound)
                ops = [
                    _frontend.replace(op, deps=tuple(dict.fromkeys(
                        (*op.deps, *commit_ids))))
                    if op.family == "token_commit" else op
                    for op in kept
                ]
        after: list = []
        transfer_producers: list = []
        if has_decode_draft or has_prefill_draft:
            from .speculation import append_draft_proposal_transfer, lower_draft_regions
            if has_decode_draft:
                decode_anchor = accept_op.op_id if accept_op is not None else previous
                decode_ready: str | tuple[str, ...] = (
                    tuple(dict.fromkeys((decode_anchor, *commit_ids)))
                    if commit_ids else decode_anchor
                )
                dec_regions, _dec_prev, dec_last = lower_draft_regions(
                    self, model, declared_runtime, decode_proposal_views, decode_ready,
                    global_views=global_decode_proposals if global_views else None,
                    prefill=False, replica=replica, replay_scope=scope, machine=machine,
                    assembly=assembly, next_proposal_widths=next_proposal_widths,
                    physical_rank=requested_rank)
                base_idx = len(after)
                after.extend(dec_regions)
                if dec_last is not None:
                    transfer_producers.append((
                        decode_proposal_views, False, (base_idx + dec_last[0], dec_last[1]),
                    ))
            if has_prefill_draft:
                pre_regions, _pre_prev, pre_last = lower_draft_regions(
                    self, model, declared_runtime, preparation_views, committed.op_id,
                    global_views=global_preparation if global_views else None, prefill=True,
                    replica=replica, replay_scope=scope, machine=machine,
                    assembly=assembly, next_proposal_widths=next_proposal_widths,
                    physical_rank=requested_rank)
                base_idx = len(after)
                after.extend(pre_regions)
                if pre_last is not None:
                    transfer_producers.append((
                        preparation_views, True, (base_idx + pre_last[0], pre_last[1]),
                    ))
            if after and transfer_producers:
                append_draft_proposal_transfer(
                    after, producers=transfer_producers, runtime=declared_runtime,
                    previous=previous, widths=next_proposal_widths,
                )
        spec_m = _spec_method(declared_runtime)
        if spec_m in _NON_NEURAL_SPEC_METHODS and ngram_propose_views and assembly == "service":
            ops = _append_non_neural_ngram_propose(
                ops, spec_method=spec_m, declared_runtime=declared_runtime,
                runtime=runtime, committed=committed,
                ngram_propose_views=ngram_propose_views,
                next_proposal_widths=next_proposal_widths, mode=mode,
            )
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

    def _resolve_speculative_draft_plan(self, plan, runtime, declared_runtime, global_views, assembly: str):
        output_tokens_by_req = {str(k): int(v) for k, v in (getattr(plan, "output_tokens", ()) or ())}
        spec_disabled = (_spec_method(declared_runtime) == "none")
        non_neural_spec = _is_non_neural_spec(declared_runtime)
        all_plan_views = tuple(plan.requests) + tuple(global_views or ())
        if not spec_disabled:
            from mindie_sim.catalog.draft_spec import validate_fia_speculative_tokens
            for view in all_plan_views:
                if not getattr(view, "padding", False):
                    prop = int(getattr(view, "draft_proposal_tokens", 0) or 0)
                    if prop != 0:
                        validate_fia_speculative_tokens(prop)
        next_proposal_widths = {
            str(view.request_id): (
                0 if spec_disabled else _next_proposal_width_for_view(
                    view, runtime=declared_runtime, output_tokens_by_req=output_tokens_by_req,
                )
            )
            for view in all_plan_views
            if not getattr(view, "padding", False)
        }
        decode_proposal_views = () if spec_disabled else tuple(
            view for view in plan.requests
            if view.phase == "decode" and next_proposal_widths.get(str(view.request_id), 0) > 0
        )
        global_decode_proposals = () if spec_disabled else tuple(
            view for view in (global_views or plan.requests)
            if view.phase == "decode" and next_proposal_widths.get(str(view.request_id), 0) > 0
        )
        has_decode_draft = (not spec_disabled) and (not non_neural_spec) and bool(
            decode_proposal_views or (global_views and global_decode_proposals)
        )
        preparation_views = () if spec_disabled else tuple(
            view for view in plan.requests if view.phase == "prefill" and view.draft_prepare
        )
        global_preparation = () if spec_disabled else tuple(
            view for view in (global_views or plan.requests) if view.phase == "prefill" and view.draft_prepare
        )
        has_prefill_draft = (not spec_disabled) and (not non_neural_spec) and bool(
            (dict(runtime.metadata).get("draft_mode") or _spec_method(declared_runtime))
            and (preparation_views or (global_views and global_preparation))
        )
        configured_k = 0 if spec_disabled else _configured_spec_k(declared_runtime)
        ngram_propose_views = () if spec_disabled else tuple(
            view for view in plan.requests
            if not getattr(view, "padding", False)
            and bool(getattr(view, "samples", False))
            and not _is_view_eos(view, output_tokens_by_req)
            and (
                configured_k > 0
                or next_proposal_widths.get(str(view.request_id), 0) > 0
            )
        )
        has_ngram_propose = bool(non_neural_spec and not spec_disabled and assembly == "service" and ngram_propose_views)
        return (
            next_proposal_widths, decode_proposal_views, global_decode_proposals,
            has_decode_draft, preparation_views, global_preparation,
            has_prefill_draft, ngram_propose_views, has_ngram_propose,
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
        model, physical_rank, features = self._resolve_call_environment(
            model, runtime, machine, replica, physical_rank)
        # An empty local query set can still receive expert rows. Its execution
        # phase belongs to the active cohort, not a fictitious "idle" kernel.
        mode = _frontend.selected_mode(plan if plan.requests else _frontend.replace(
            plan, requests=tuple(global_views or ())))
        _validate_decode_ready_state(runtime, mode)
        if assembly == "model":
            from math import floor
            mean = float(runtime.mtp_acceptance_length)
            plan = _frontend.replace(plan, requests=tuple(
                _frontend.replace(view,
                        draft_advanced_tokens=min(
                            _verify_query_len(view),
                            max(1, floor((index + 1) * mean) - floor(index * mean))))
                if view.draft_proposal_tokens and not view.draft_advanced_tokens
                else view for index, view in enumerate(plan.requests)))
        views = tuple(_frontend.replace(view, q_len=_verify_query_len(view))
                      for view in plan.requests)
        cohort_target_views = _expand_target_cohort_views(cohort_views_by_replica)
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
        (entry, before, ops, previous, local_views, local_tokens,
         pcp_prefill) = self._lower_iteration_inputs(
            plan, model, runtime, declared_runtime, machine, views,
            global_views=global_views, replica=replica, requested_rank=requested_rank,
            mode=mode, scope=scope, valid=valid, padded=padded, requests=requests,
        )
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
        (next_proposal_widths, decode_proposal_views, global_decode_proposals,
         has_decode_draft, preparation_views, global_preparation,
         has_prefill_draft, ngram_propose_views, has_ngram_propose) = self._resolve_speculative_draft_plan(
            plan, runtime, declared_runtime, global_views, assembly,
        )
        has_after_draft = has_prefill_draft or has_decode_draft or has_ngram_propose
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
                                   cohort_views_by_replica=cohort_target_views,
                                   physical_rank=physical_rank,
                                   explicit_physical_rank=(explicit_physical_rank or requested_rank is not None),
                                   query_position=eff_query_pos,
                                   include_recover_commits=(assembly == "service" or any(
                                       view.draft_proposal_tokens for view in plan.requests)),
                                   machine=machine)
        ops.extend(body)
        return self._lower_iteration_outputs(
            ops, previous, entry=entry, plan=plan, views=views, model=model,
            runtime=runtime, declared_runtime=declared_runtime, machine=machine,
            before=before, assembly=assembly, replica=replica,
            requested_rank=requested_rank, scope=scope,
            mode=mode, valid=valid, padded=padded, bucket=bucket,
            local_tokens=local_tokens, requests=requests, pcp_prefill=pcp_prefill,
            deferred_commits=deferred_commits, has_prefill_draft=has_prefill_draft,
            has_decode_draft=has_decode_draft,
            decode_proposal_views=decode_proposal_views,
            global_decode_proposals=global_decode_proposals,
            preparation_views=preparation_views, global_views=global_views,
            global_preparation=global_preparation, selections=selections,
            resource_fusion=resource_fusion,
            next_proposal_widths=next_proposal_widths,
            ngram_propose_views=ngram_propose_views,
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
