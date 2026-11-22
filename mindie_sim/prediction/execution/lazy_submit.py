"""Bounded lazy-frontier DAG materialization for single-replica steps.

[Role]
  Materialize only the active dependency frontier and a bounded host-launch
  lookahead window ahead of the device frontier, keeping working set
  ``O(Active Frontier)`` while executing the exact same causal Engine tasks
  as eager DAG submission.

[Invariants]
  * Dependency cycles raise ``ModelingError("execution_dependency_cycle")``
    before any task is admitted.
  * Per-stage op/program caches are strictly bounded (``STAGE_ID_CACHE_LIMIT``,
    ``LAZY_OP_CACHE_LIMIT``) and store only declaration lowerings, never
    predicted durations.
  * Every stream ``stream_id`` maintains exact per-stream ordering across
    arbitrary ``N >= 1`` streams.

[Timeouts & Locks]
  Bounded active frontier queue and cache sizes; no I/O or locks.

[Side-Effect Policy]
  Submits tasks only to ``self.engine`` and updates bounded per-Execution caches.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from ..engine import Event
from ..execution_state import memory_bindings_for
from ..program import KernelProgram
from .lazy_dag import (
    build_verified_dag,
    collect_begin_after,
    run_cached_bundle,
    stage_memory_info,
)
from .streams import _meta_diff, _meta_patch
from .submission import ScheduledIteration

COMPACT_SUBMIT_MIN_OPS = 32
STAGE_ID_CACHE_LIMIT = 16384
LAZY_OP_CACHE_LIMIT = 4096


class _LazySubmitMixin:
    """Active-frontier DAG submission fast path for ``Execution``."""

    def _add_lazy(self, work, replica, release_ms, *, scope, schedule,
                  prepared, input_ready, physical_ops, prefix,
                  external_inputs, emitted_tokens, memory_bindings,
                  state_accesses, verified_dag=None):
        """Materialize only the active dependency frontier."""
        engine = self.engine
        if self.offload_policy is not None:
            raise ModelingError("lazy_frontier_requires_no_offload_policy")
        n_ops = len(physical_ops)
        if verified_dag is not None:
            packed = verified_dag
        else:
            packed, self._prev_lazy_dag = build_verified_dag(
                physical_ops, self._prev_lazy_dag)
        (order, succ_idx, init_unmet_t, dep_specs,
         root_idx, sink_idx, fut_indices) = packed

        unmet = list(init_unmet_t)
        op_events: list[Event | None] = [None] * n_ops
        op_futures: list[Event | None] = [None] * n_ops
        for fi in fut_indices:
            fut = engine.future(f"{prefix}:{physical_ops[fi].op_id}:lazy_complete")
            op_futures[fi] = fut
            op_events[fi] = fut
        ext_map = dict(external_inputs) if external_inputs else None
        declared_commits = {}
        for fi in fut_indices:
            fop = physical_ops[fi]
            if fop.family == "token_commit":
                fut = op_futures[fi]
                for request in fop.requests:
                    declared_commits[request] = fut

        host_info: list[Any] = [None] * n_ops
        host_cursor = 0
        host_submits: list[tuple[int, int, Event]] = []
        step_holder: dict[str, Any] = {}
        owner = f"replica:{replica}"
        compact_exec = (
            not self.legacy_diagnostics
            and (not engine.trace.maxlen or n_ops > engine.trace.maxlen)
        )
        compact_submit = compact_exec and n_ops > COMPACT_SUBMIT_MIN_OPS
        self._compact_step = compact_exec
        if hasattr(engine, "_compact_step"):
            engine._compact_step = compact_exec
        lazy_fast = (
            not self.legacy_diagnostics
            and not self.regimes.rules
            and not type(self).schedule_replay
        )
        stage_rates_map = (
            {stg: tuple(engine.resources[rn].rate for rn in s_names.values())
             for (rep, stg), s_names in self.stage_names.items()
             if rep == replica}
            if lazy_fast else {}
        )
        include_sub = self.include_submission
        stage_launch_ms = (
            {stg: self.submission.launch_ms(dies)
             for (rep, stg), dies in self.stage_dies.items()
             if rep == replica}
            if include_sub else {}
        )
        stage_host_prep = (
            {stg: engine._prepare(
                ":submit", {self.stage_names[(replica, stg)]["host"]: dur})
             for stg, dur in stage_launch_ms.items()}
            if include_sub else {}
        )
        stage_zero_host_prep = (
            {stg: engine._prepare(
                ":submit", {self.stage_names[(replica, stg)]["host"]: 0.0})
             for stg in stage_launch_ms}
            if (include_sub and compact_submit) else {}
        )
        stage_mem_info = stage_memory_info(self, replica)
        shared_zero_submit: dict[tuple[int, int], tuple[Event, Event]] = {}
        prefix_colon = prefix + ":"
        host_lookahead = self.host_lookahead
        observer = self.on_operator
        recipes_id = id(self.recipes)
        scope_str = str(scope)
        stage_id_caches = (
            {
                stg: self._lazy_stage_id_caches.setdefault(
                    (replica, scope, compact_exec, recipes_id, lrates), {}
                )
                for stg, lrates in stage_rates_map.items()
            }
            if lazy_fast else {}
        )
        step_env = (
            (replica, scope, compact_exec, recipes_id,
             tuple(sorted(stage_rates_map.items())))
            if lazy_fast else None
        )
        prev_le = self._prev_lazy_entries
        if (step_env is not None and prev_le is not None
                and prev_le[0] == step_env and len(prev_le[1]) == n_ops):
            prev_l_ops, prev_l_entries = prev_le[1], prev_le[2]
        else:
            prev_l_ops = prev_l_entries = None
        cur_l_entries: list[Any] | None = [None] * n_ops if lazy_fast else None
        if step_env is not None:
            self._prev_lazy_entries = (step_env, physical_ops, cur_l_entries)

        lazy_cache = self._lazy_op_cache
        cost_counts = self.cost.counts
        cost_invs = self.cost.invocations
        trans_opts = self._transport_options
        streams = self.streams
        recipes_used = self.recipes_used
        exec_work = self.work
        seen_host_stages: set[tuple[int, int]] = set()

        def update_host_submits():
            step = step_holder.get("step")
            if step is not None:
                step.host_submits = tuple(host_submits)

        def _submit_host_launch(stage_key, name):
            stg_id = stage_key[1]
            duration = stage_launch_ms[stg_id]
            if compact_submit and duration == 0.0:
                shared = shared_zero_submit.get(stage_key)
                if shared is None:
                    prev_h = self.hosts.get(stage_key)
                    host_deps = (prepared, prev_h) if prev_h is not None else prepared
                    launch, start = engine._submit_host_launch(
                        stage_zero_host_prep[stg_id], name + ":submit",
                        name + ":host_submit", host_deps, owner)
                    self.hosts[stage_key] = launch
                    seen_host_stages.add(stage_key)
                    host_submits.append((replica, stg_id, start))
                    shared_zero_submit[stage_key] = (launch, start)
                else:
                    launch, start = shared
                self.submission_counts["eager_invocations"] += 1
                return launch, start
            prev_h = self.hosts.get(stage_key)
            if stage_key in seen_host_stages:
                host_deps = prev_h
            else:
                seen_host_stages.add(stage_key)
                host_deps = (prepared, prev_h) if prev_h is not None else prepared
            launch, start = engine._submit_host_launch(
                stage_host_prep[stg_id], name + ":submit",
                name + ":host_submit", host_deps, owner)
            self.hosts[stage_key] = launch
            self.submission_counts["eager_invocations"] += 1
            host_submits.append((replica, stg_id, start))
            return launch, start

        def materialize_one(idx: int, op):
            nonlocal host_cursor
            entry = None
            lazy_key = None
            stg_id_cache = None
            same_step_op = False
            if lazy_fast:
                if (prev_l_ops is not None and prev_l_ops[idx] is op
                        and (pe := prev_l_entries[idx]) is not None):
                    entry = pe
                    same_step_op = True
                else:
                    stg_id_cache = stage_id_caches[op.stage]
                    hit = stg_id_cache.get(id(op))
                    if hit is not None and hit[0] is op:
                        entry = hit[1]
                    else:
                        live_rates = stage_rates_map[op.stage]
                        lazy_key = (
                            replica, scope, compact_exec, recipes_id, live_rates,
                            op.stage, op.family, op.recipe, op.phase, op.dtype, op.layout,
                            op.semantic_shape, op.local_shape, op.padded_shape, op.launch_shape,
                            op.flops, op.matrix_flops, op.mte2_bytes, op.mte3_bytes,
                            op.vector_ops, op.hbm_bytes, op.link_bytes, op.link_scope,
                            op.source_boundary, op.returns, self._op_key(op),
                            op.stream or "main",
                            self.stream_owners.get((replica, op.stage, op.stream or "main")),
                        )
                        entry = lazy_cache.get(lazy_key)
                        if entry is not None and entry is not False:
                            if len(stg_id_cache) >= STAGE_ID_CACHE_LIMIT:
                                stg_id_cache.clear()
                            stg_id_cache[id(op)] = (op, entry)
                if (entry is not None and entry is not False
                        and entry[1] is not False):
                    if cur_l_entries is not None:
                        cur_l_entries[idx] = entry
                    (res_op_tmpl, _mdiff, _rmem, _rec, _ctx_tmpl,
                     submit, transport_options, inv_cell, _rname, _oname,
                     _wdeltas) = entry[0]
                    cost_counts["analytic_work"] += 1
                    if not same_step_op:
                        trans_opts[(scope_str, op.op_id, replica)] = transport_options
                    if inv_cell is not None:
                        if inv_cell[2] is cost_invs and inv_cell[3] is not None:
                            inv_cell[3]["invocations"] += 1
                        elif inv_cell[0] in cost_invs:
                            li = cost_invs[inv_cell[0]]
                            inv_cell[2] = cost_invs
                            inv_cell[3] = li
                            li["invocations"] += 1
                        else:
                            self.cost.record_invocation(inv_cell[1])
                            inv_cell[2] = cost_invs
                            inv_cell[3] = cost_invs.get(inv_cell[0])
                    name = prefix_colon + op.op_id
                    launch = start = None
                    if submit and include_sub:
                        launch, start = _submit_host_launch(
                            (replica, res_op_tmpl.stage), name)
                    host_info[idx] = (entry, op, name, launch, start, None)
                    host_cursor += 1
                    return

            res_op, recipe = self.prepare(op, replica)
            stage_key = replica, res_op.stage
            name = prefix_colon + res_op.op_id
            context = self._context(res_op, replica, scope, recipe=recipe)
            submit = bool(context.work or getattr(recipe, "algorithm", None))
            launch = start = None
            if submit and include_sub:
                launch, start = _submit_host_launch(stage_key, name)
            fast_created = None
            if lazy_fast and lazy_key is not None and entry is None:
                orig_meta = dict(op.metadata)
                res_meta = dict(res_op.metadata)
                meta_diff = _meta_diff(orig_meta, res_meta) if orig_meta != res_meta else None
                raw_mm = res_meta.get("memory_bindings", res_meta.get("memory_binding"))
                raw_meta_mem = (
                    {str(mn): dict(mv) for mn, mv in raw_mm.items()
                     if isinstance(mv, Mapping)}
                    if isinstance(raw_mm, Mapping) else None
                )
                transport_options = trans_opts[(scope_str, res_op.op_id, replica)]
                inv_record = self._last_invocation_record
                inv_cell = (
                    [inv_record["id"], inv_record, cost_invs,
                     cost_invs.get(inv_record["id"])]
                    if inv_record is not None else None
                )
                recipe_name = res_op.recipe or type(recipe).__name__
                obs_prog_name = (
                    "JointNativeService"
                    if context.joint_native_service is not None
                    else type(recipe).__name__
                )
                work_deltas = tuple(
                    (k, v) for k in exec_work
                    if (v := float(getattr(res_op, k))) != 0.0
                )
                prep_bundle = (
                    res_op, meta_diff, raw_meta_mem, recipe, context,
                    submit, transport_options, inv_cell, recipe_name, obs_prog_name, work_deltas,
                )
                fast_created = [prep_bundle, None]
                if len(lazy_cache) >= LAZY_OP_CACHE_LIMIT:
                    lazy_cache.pop(next(iter(lazy_cache)))
                lazy_cache[lazy_key] = fast_created
                if stg_id_cache is not None:
                    if len(stg_id_cache) >= STAGE_ID_CACHE_LIMIT:
                        stg_id_cache.clear()
                    stg_id_cache[id(op)] = (op, fast_created)
                if cur_l_entries is not None:
                    cur_l_entries[idx] = fast_created
            if fast_created is not None:
                host_info[idx] = (fast_created, op, name, launch, start, None)
            else:
                host_info[idx] = (None, op, name, launch, start, (res_op, recipe, context))
            host_cursor += 1

        def materialize_through(limit_idx: int):
            nonlocal host_cursor
            end = n_ops - 1 if limit_idx >= n_ops else limit_idx
            if host_cursor > end:
                return
            prev_len = len(host_submits)
            while host_cursor <= end:
                cur = host_cursor
                if host_info[cur] is None:
                    materialize_one(cur, physical_ops[cur])
                else:
                    host_cursor = cur + 1
            if step_holder and len(host_submits) != prev_len and host_cursor >= n_ops:
                update_host_submits()

        def _run_bundle(op, name, ready_ev, fast_entry, res_op_tmpl, meta_diff, context_tmpl):
            return run_cached_bundle(
                self, engine, replica, owner, compact_exec,
                op, name, ready_ev, fast_entry, res_op_tmpl, meta_diff, context_tmpl)

        def activate(idx: int):
            info = host_info[idx]
            if info is None:
                materialize_through(idx)
                info = host_info[idx]
            fast_entry, orig_op, name, launch, _start, slow_pack = info
            op = physical_ops[idx]
            if fast_entry is not None and fast_entry[1] is False:
                res_op, recipe = self.prepare(op, replica)
                context = self._context(res_op, replica, scope, recipe=recipe)
                slow_pack = (res_op, recipe, context)
                fast_entry = None
            if fast_entry is not None:
                (res_op_tmpl, meta_diff, raw_meta_mem, _recipe, context_tmpl,
                 _submit, _topt, _inv, recipe_name, obs_prog_name, work_deltas) = fast_entry[0]
                res_stage = res_op_tmpl.stage
                stream_key = (replica, res_stage, op.stream or res_op_tmpl.stream or "main")
                tail = streams.get(stream_key)
                begin_after = collect_begin_after(
                    idx, op, launch, tail, dep_specs, op_events,
                    order, ext_map, input_ready, schedule)
                mem_info = stage_mem_info.get(res_stage)
                if mem_info is not None:
                    physical_hbm, mctx_fast_dict, mfast_tuple = mem_info
                else:
                    physical_hbm = self.stage_pools.get(
                        (replica, res_stage), {}).get("hbm", "hbm")
                    mctx_fast_dict = mfast_tuple = None
                if memory_bindings is None:
                    mem_bind = (
                        {mn: dict(mv) for mn, mv in raw_meta_mem.items()}
                        if raw_meta_mem is not None else None)
                else:
                    meta = op.metadata
                    if meta_diff is not None:
                        patched = _meta_patch(dict(op.metadata), meta_diff)
                        meta = tuple(sorted(patched.items()))
                    op_for_mem = replace(res_op_tmpl, op_id=op.op_id, metadata=meta)
                    mem_bind = memory_bindings_for(op_for_mem, memory_bindings)

                now = engine.now
                sync_ev = None
                all_done = True
                for ev in begin_after:
                    ev_t = ev.time
                    if ev_t is None or ev_t > now:
                        all_done = False
                        sync_ev = None
                        break
                    if ev_t == now:
                        sync_ev = ev
                pending_evs = None
                if all_done and sync_ev is None and compact_exec and begin_after:
                    sync_ev = engine._instant_done_event(
                        name + ":begin" if observer is None else name + ":ready")
                elif not all_done and compact_exec and observer is None:
                    pending_evs = [
                        ev for ev in begin_after if ev.time is None or ev.time > now]
                    pb = fast_entry[1]
                    if pb is not None and not pb[4] and pb[5] is None:
                        prog = pb[0].program if pb[3] else pb[0]
                        if prog.__class__ is KernelProgram and prog.minimum_ms <= 0.0:
                            comp_c = getattr(engine, "_compiled_programs", None)
                            chit = comp_c.get(id(prog)) if comp_c is not None else None
                            if (chit is not None and chit[0] is prog
                                    and chit[1] is not False and not chit[1][3]):
                                if len(pending_evs) == 1:
                                    sync_ev = pending_evs[0]
                                elif len(pending_evs) > 1 and chit[1][0]:
                                    sync_ev = tuple(pending_evs)
                if sync_ev is not None:
                    if mem_bind is None and mfast_tuple is not None:
                        prev_mc, prev_mf = engine._memory_context, engine._memory_context_fast
                        engine._memory_context = mctx_fast_dict
                        engine._memory_context_fast = mfast_tuple
                        try:
                            complete = _run_bundle(
                                op, name, sync_ev, fast_entry, res_op_tmpl, meta_diff, context_tmpl)
                        finally:
                            engine._memory_context = prev_mc
                            engine._memory_context_fast = prev_mf
                    else:
                        with engine.memory_scope(
                                self.memory, replica=replica, stage=int(res_stage),
                                pools=(physical_hbm,), bindings=mem_bind):
                            complete = _run_bundle(
                                op, name, sync_ev, fast_entry, res_op_tmpl, meta_diff, context_tmpl)
                    streams[stream_key] = complete
                    op_events[idx] = complete
                    recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
                    if observer is not None:
                        observer(orig_op, replica, obs_prog_name, sync_ev, complete)
                    for key, delta in work_deltas:
                        exec_work[key] += delta
                    if host_cursor < n_ops:
                        materialize_through(idx + host_lookahead)
                    if complete.time is not None:
                        if op_futures[idx] is not None or succ_idx[idx]:
                            completed(idx)
                    elif op_futures[idx] is not None or succ_idx[idx]:
                        t_comp = engine._tasks.get(complete.sequence)
                        cb = lambda _ev, _i=idx: completed(_i)
                        if t_comp is not None and t_comp.callback is None:
                            t_comp.callback = cb
                        else:
                            engine.on_complete(complete, cb)
                    return

                complete = engine.future(name + ":program_complete")

                def async_begin(boundary_ev, stg=int(res_stage), phbm=physical_hbm,
                                mb=mem_bind, mcd=mctx_fast_dict, mft=mfast_tuple,
                                comp=complete, _i=idx, _op=op, _nm=name,
                                _fe=fast_entry, _rt=res_op_tmpl, _md=meta_diff,
                                _ct=context_tmpl):
                    if mb is None and mft is not None:
                        prev_mc, prev_mf = engine._memory_context, engine._memory_context_fast
                        engine._memory_context = mcd
                        engine._memory_context_fast = mft
                        try:
                            finish = _run_bundle(_op, _nm, boundary_ev, _fe, _rt, _md, _ct)
                        finally:
                            engine._memory_context = prev_mc
                            engine._memory_context_fast = prev_mf
                    else:
                        with engine.memory_scope(
                                self.memory, replica=replica, stage=stg,
                                pools=(phbm,), bindings=mb):
                            finish = _run_bundle(_op, _nm, boundary_ev, _fe, _rt, _md, _ct)

                    def done_cb(_ev=None):
                        engine.resolve(comp)
                        completed(_i)

                    engine.on_complete(finish, done_cb, native_hook=True)

                if compact_exec and pending_evs is None:
                    pending_evs = [
                        ev for ev in begin_after if ev.time is None or ev.time > now]
                if pending_evs is not None and len(pending_evs) == 1:
                    ready_marker = pending_evs[0]
                    engine.on_complete(ready_marker, async_begin, native_hook=True)
                else:
                    ready_marker = engine.submit(
                        name + ":begin" if observer is None else name + ":ready",
                        after=begin_after, owner=owner, on_complete=async_begin,
                        native_hook=True)
                streams[stream_key] = complete
                op_events[idx] = complete
                recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
                if observer is not None:
                    observer(orig_op, replica, obs_prog_name, ready_marker, complete)
                for key, delta in work_deltas:
                    exec_work[key] += delta
                if host_cursor < n_ops:
                    materialize_through(idx + host_lookahead)
                return

            res_op, recipe, context = slow_pack
            stage_key = replica, res_op.stage
            stream_key = replica, res_op.stage, res_op.stream or "main"
            tail = streams.get(stream_key)
            begin_after = collect_begin_after(
                idx, op, launch, tail, dep_specs, op_events,
                order, ext_map, input_ready, schedule)
            physical_hbm = self.stage_pools.get(stage_key, {}).get("hbm", "hbm")
            if observer is None:
                ready = None
                complete = self._start_program(
                    context, recipe, name, None, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: completed(_i), begin_after=begin_after)
            else:
                ready = engine.submit(name + ":ready", after=begin_after, owner=owner)
                complete = self._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=self.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: completed(_i))
            streams[stream_key] = complete
            op_events[idx] = complete
            recipe_name = res_op.recipe or type(recipe).__name__
            recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
            if observer is not None:
                observer(
                    orig_op, replica,
                    ("JointNativeService" if context.joint_native_service
                     is not None else type(recipe).__name__),
                    ready, complete)
            for key in exec_work:
                exec_work[key] += float(getattr(res_op, key))
            materialize_through(idx + host_lookahead)

        activating = False
        pending_activations = deque()

        def drain_activations():
            nonlocal activating
            if activating:
                return
            activating = True
            try:
                while pending_activations:
                    activate(pending_activations.popleft())
            finally:
                activating = False

        def completed(idx: int):
            nonlocal activating
            fut = op_futures[idx]
            if fut is not None:
                engine.resolve(fut)
            succs = succ_idx[idx]
            n_s = len(succs)
            if n_s == 1:
                s0 = succs[0]
                u0 = unmet[s0] - 1
                unmet[s0] = u0
                if not u0:
                    if activating:
                        pending_activations.append(s0)
                    else:
                        activating = True
                        try:
                            activate(s0)
                            while pending_activations:
                                activate(pending_activations.popleft())
                        finally:
                            activating = False
                return
            if not n_s:
                return
            ready_ops = [s for s in succs if (unmet.__setitem__(s, unmet[s] - 1) or unmet[s] == 0)]
            if ready_ops:
                if len(ready_ops) > 1:
                    ready_ops.sort()
                pending_activations.extend(ready_ops)
                drain_activations()

        materialize_through(
            (root_idx[-1] + host_lookahead) if root_idx else host_lookahead)
        if root_idx:
            pending_activations.extend(root_idx)
            drain_activations()
        device_complete = engine.submit(
            prefix + ":device_complete", owner=owner,
            after=tuple(op_futures[si] for si in sink_idx) or (input_ready,))
        completion = engine.submit(
            prefix + ":complete", after=(device_complete,),
            latency=(self.submission.completion_ms if self.include_submission else 0),
            owner=owner)
        fallback_commits = {
            view.request_id: completion for view in work.plan.requests
            if view.request_id not in declared_commits}
        commits = {**fallback_commits, **declared_commits}
        host_results, clients, host_complete = self.serving_io.output(
            self, prefix, replica, work, commits, completion,
            emitted_tokens if emitted_tokens is not None else {
                view.request_id: int(view.draft_advanced_tokens or 1)
                for view in work.plan.requests if view.samples})
        if self.serving_io.enabled:
            completion = engine.submit(
                prefix + ":step_ready", after=(completion, host_complete), owner=owner)
        from ..offload import release_offload_access
        offload_release = engine.submit(
            prefix + ":offload_release", after=(device_complete,),
            owner=owner, native_hook=True,
            on_complete=lambda _event: release_offload_access(self, work))
        step = ScheduledIteration(
            replica, work.plan.iteration, completion, release_ms,
            work.mode, work.valid_tokens, work.padded_tokens,
            work.graph_bucket, scope, len(physical_ops), commits,
            device_complete, host_results, clients,
            work.implementation_selections,
            tuple(host_submits),
            dict(declared_commits), dict(fallback_commits),
            offload_release)
        step_holder["step"] = step
        update_host_submits()

        def _record_step(_):
            if hasattr(engine, "_compact_step"):
                engine._compact_step = False
            update_host_submits()
            self.step_summary.record(step)

        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=_record_step, native_hook=True)
        return step
