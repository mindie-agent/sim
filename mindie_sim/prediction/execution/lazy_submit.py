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
from functools import partial
from typing import Any, Mapping

from mindie_sim.schema.common import ModelingError
from ..engine import Event
from ..execution.state import memory_bindings_for
from ..program import KernelProgram
from ..program.invocation import _declared_launch_count
from .lazy_dag import (
    INV_CELL_LAUNCHES_IDX,
    _lower_and_start_uncached,
    build_verified_dag,
    collect_begin_after,
    live_stage_l2_state,
    run_cached_bundle,
    stage_memory_info,
    undo_invocation_record,
    undo_superseded_invocation,
)
from .streams import _freeze_key, _meta_diff, _meta_patch
from .submission import ScheduledIteration

COMPACT_SUBMIT_MIN_OPS = 32
STAGE_ID_CACHE_LIMIT = 16384
LAZY_OP_CACHE_LIMIT = 4096
_PACKED_DAG_ARITY = 9
_PACKED_DAG_STREAM_IDX = 7


class _LazyFrontier:
    """Live dependency and host-lookahead state for one submitted iteration."""

    def __init__(
        self, execution, replica, *, scope, schedule, prepared, input_ready,
        physical_ops, prefix, external_inputs, memory_bindings, verified_dag,
    ):
        engine = execution.engine
        if execution.offload_policy is not None:
            raise ModelingError("lazy_frontier_requires_no_offload_policy")
        n_ops = len(physical_ops)
        if (
            verified_dag is not None
            and len(verified_dag) == _PACKED_DAG_ARITY
            and len(verified_dag[_PACKED_DAG_STREAM_IDX]) == n_ops
            and all(
                (op.stage, op.stream or "main") == sk
                for op, sk in zip(physical_ops, verified_dag[_PACKED_DAG_STREAM_IDX])
            )
        ):
            packed = verified_dag
        else:
            packed, execution._prev_lazy_dag = build_verified_dag(
                physical_ops, execution._prev_lazy_dag)
        (order, succ_idx, init_unmet_t, dep_specs,
         root_idx, sink_idx, fut_indices, _stream_sig, stream_tails) = packed

        self.unmet = list(init_unmet_t)
        op_events: list[Event | None] = [None] * n_ops
        op_futures: list[Event | None] = [None] * n_ops
        for fi in fut_indices:
            fut = engine.future(f"{prefix}:{physical_ops[fi].op_id}:lazy_complete")
            op_futures[fi] = fut
            op_events[fi] = fut
        self.ext_map = dict(external_inputs) if external_inputs else None
        declared_commits = {}
        for fi in fut_indices:
            fop = physical_ops[fi]
            if fop.family == "token_commit":
                fut = op_futures[fi]
                for request in fop.requests:
                    declared_commits[request] = fut

        self.host_info: list[Any] = [None] * n_ops
        self._op_host_submit: list[tuple[int, int, Event] | None] = [None] * n_ops
        self.owner = f"replica:{replica}"
        self._step_active_requests = tuple(getattr(execution, "_active_requests", ()) or ())
        self._step_active_padded_tokens = int(getattr(execution, "_active_padded_tokens", 0) or 0)
        self._step_active_layout = str(getattr(execution, "_active_layout", "") or "")
        self._step_active_sig = (
            self._step_active_requests,
            self._step_active_padded_tokens,
            self._step_active_layout,
        )
        compact_exec = (
            not execution.legacy_diagnostics
            and (not engine.trace.maxlen or n_ops > engine.trace.maxlen)
        )
        compact_submit = compact_exec and n_ops > COMPACT_SUBMIT_MIN_OPS
        execution._compact_step = compact_exec
        if hasattr(engine, "_compact_step"):
            engine._compact_step = compact_exec
        lazy_fast = (
            not execution.legacy_diagnostics
            and not execution.regimes.rules
            and not type(execution).schedule_replay
        )
        stage_rates_map = (
            {stg: tuple(engine.resources[rn].rate for rn in s_names.values())
             for (rep, stg), s_names in execution.stage_names.items()
             if rep == replica}
            if lazy_fast else {}
        )
        include_sub = execution.include_submission
        stage_launch_ms = (
            {stg: execution._stage_launch_ms((rep, stg), captured=False)
             for (rep, stg) in execution.stage_dies
             if rep == replica}
            if include_sub else {}
        )
        self.stage_host_prep = (
            {stg: engine._prepare(
                ":submit", {execution.stage_names[(replica, stg)]["host"]: dur})
             for stg, dur in stage_launch_ms.items()}
            if include_sub else {}
        )
        self.stage_zero_host_prep = (
            {stg: engine._prepare(
                ":submit", {execution.stage_names[(replica, stg)]["host"]: 0.0})
             for stg in stage_launch_ms}
            if (include_sub and compact_submit) else {}
        )
        self.stage_mem_info = stage_memory_info(execution, replica)
        self.shared_zero_submit: dict[tuple[int, int], tuple[Event, Event]] = {}
        self.prefix_colon = prefix + ":"
        self.host_lookahead = execution.host_lookahead
        self.observer = execution.on_operator
        recipes_id = id(execution.recipes)
        self.scope_str = str(scope)
        self.stage_id_caches = (
            {
                stg: execution._lazy_stage_id_caches.setdefault(
                    (replica, scope, compact_exec, recipes_id, lrates, self._step_active_sig), {}
                )
                for stg, lrates in stage_rates_map.items()
            }
            if lazy_fast else {}
        )
        step_env = (
            (replica, scope, compact_exec, recipes_id,
             tuple(sorted(stage_rates_map.items())), self._step_active_sig)
            if lazy_fast else None
        )
        prev_le = execution._prev_lazy_entries
        if (step_env is not None and prev_le is not None
                and prev_le[0] == step_env and len(prev_le[1]) == n_ops):
            prev_l_ops, prev_l_entries = prev_le[1], prev_le[2]
        else:
            prev_l_ops = prev_l_entries = None
        cur_l_entries: list[Any] | None = [None] * n_ops if lazy_fast else None
        if step_env is not None:
            execution._prev_lazy_entries = (step_env, physical_ops, cur_l_entries)

        self.lazy_cache = execution._lazy_op_cache
        self.cost_counts = execution.cost.counts
        self.cost_invs = execution.cost.invocations
        self.trans_opts = execution._transport_options
        self.streams: dict[tuple[int, int, str], Event] = {}
        for (stg, stm), tail_idx in stream_tails:
            skey = (replica, stg, stm)
            prev_ev = execution.streams.get(skey)
            if prev_ev is not None:
                self.streams[skey] = prev_ev
            execution.streams[skey] = op_futures[tail_idx]
        self.recipes_used = execution.recipes_used
        self.exec_work = execution.work
        self.seen_host_stages: set[tuple[int, int]] = set()

        self.execution = execution
        self.engine = engine
        self.replica = replica
        self.scope = scope
        self.prepared = prepared
        self.input_ready = input_ready
        self.physical_ops = physical_ops
        self.prefix = prefix
        self.memory_bindings = memory_bindings
        self.schedule = schedule
        self.n_ops = n_ops
        self.order = order
        self.succ_idx = succ_idx
        self.dep_specs = dep_specs
        self.root_idx = root_idx
        self.sink_idx = sink_idx
        self.op_events = op_events
        self.op_futures = op_futures
        self.declared_commits = declared_commits
        self.compact_exec = compact_exec
        self.compact_submit = compact_submit
        self.lazy_fast = lazy_fast
        self.stage_rates_map = stage_rates_map
        self.include_sub = include_sub
        self.stage_launch_ms = stage_launch_ms
        self.recipes_id = recipes_id
        self.prev_l_ops = prev_l_ops
        self.prev_l_entries = prev_l_entries
        self.cur_l_entries = cur_l_entries
        self.step = None
        self.activating = False
        self.pending_activations = deque()
        self._init_host_queues()

    def _init_host_queues(self):
        self._stage_op_indices: dict[int, list[int]] = {}
        self._stage_host_pos: dict[int, int] = {}
        self._stage_host_last_pos: dict[int, int] = {}
        self._stage_host_in_flight: dict[int, int] = {}
        self._stage_host_tail: dict[int, Event | None] = {}
        self._stage_host_before_last: dict[int, Event | None] = {}
        self._stage_host_before_last_resolved: dict[int, bool] = {}
        self._stage_host_fifo_base: dict[int, int] = {}
        self._stage_compact_zero: set[int] = set()
        self._prepared_pending_host: dict[int, tuple] = {}
        self._awaiting_host: set[int] = set()
        if not self.include_sub:
            self._host_pumped_stages: frozenset[int] = frozenset()
            return
        for idx, op in enumerate(self.physical_ops):
            self._stage_op_indices.setdefault(op.stage, []).append(idx)
        base_seq = self.engine._sequence
        for stg_id in self._stage_op_indices:
            self._stage_host_tail[stg_id] = self.execution.hosts.get((self.replica, stg_id))
            self._stage_host_fifo_base[stg_id] = base_seq
            if self.compact_submit and self.stage_launch_ms.get(stg_id, 0.0) == 0.0:
                self._stage_compact_zero.add(stg_id)
                continue
            self._stage_host_pos[stg_id] = 0
            self._stage_host_last_pos[stg_id] = -1
            self._stage_host_in_flight[stg_id] = 0
            self._stage_host_before_last[stg_id] = None
            self._stage_host_before_last_resolved[stg_id] = False
        self._host_pumped_stages = frozenset(self._stage_host_pos)

    def _current_host_submits(self) -> tuple[tuple[int, int, Event], ...]:
        return tuple(item for item in self._op_host_submit if item is not None)

    def _update_host_submits(self):
        if self.step is not None:
            self.step.host_submits = self._current_host_submits()

    def _prepare_op_context(self, op):
        execution = self.execution
        prev_req = execution._active_requests
        prev_pad = execution._active_padded_tokens
        prev_lay = execution._active_layout
        execution._active_requests = self._step_active_requests
        execution._active_padded_tokens = self._step_active_padded_tokens
        execution._active_layout = self._step_active_layout
        try:
            res_op, recipe = execution.prepare(op, self.replica)
            context = execution._context(res_op, self.replica, self.scope, recipe=recipe)
        finally:
            execution._active_requests = prev_req
            execution._active_padded_tokens = prev_pad
            execution._active_layout = prev_lay
        return res_op, recipe, context

    def _submit_stage_host_task(
        self, stg_id: int, idx: int, pos: int, name: str, *,
        host_deps, attach_pump: bool, publish_global: bool,
    ) -> tuple[Event, Event]:
        engine = self.engine
        launch, start = engine._submit_host_launch(
            self.stage_host_prep[stg_id], name + ":submit",
            name + ":host_submit", host_deps, self.owner)
        task = engine._tasks.get(launch.sequence)
        if task is not None:
            task.fifo_order = (self._stage_host_fifo_base[stg_id], pos)
        if attach_pump:
            self._stage_host_tail[stg_id] = launch
            self._stage_host_in_flight[stg_id] += 1
            engine.on_complete(
                launch, partial(self._on_host_launch_done, stg_id=stg_id),
                native_hook=True)
        if publish_global:
            self.execution.hosts[(self.replica, stg_id)] = launch
        self.execution.submission_counts["eager_invocations"] += 1
        self._op_host_submit[idx] = (self.replica, stg_id, start)
        return launch, start

    def _submit_compact_zero_launch(
        self, stg_id: int, idx: int, name: str,
    ) -> tuple[Event, Event]:
        stage_key = (self.replica, stg_id)
        shared = self.shared_zero_submit.get(stage_key)
        if shared is None:
            prev_h = self._stage_host_tail.get(stg_id)
            host_deps = (self.prepared, prev_h) if prev_h is not None else self.prepared
            launch, start = self.engine._submit_host_launch(
                self.stage_zero_host_prep[stg_id], name + ":submit",
                name + ":host_submit", host_deps, self.owner)
            task = self.engine._tasks.get(launch.sequence)
            if task is not None:
                task.fifo_order = (self._stage_host_fifo_base[stg_id], 0)
            self._stage_host_tail[stg_id] = launch
            self.execution.hosts[stage_key] = launch
            self.seen_host_stages.add(stage_key)
            self._op_host_submit[idx] = (self.replica, stg_id, start)
            shared = (launch, start)
            self.shared_zero_submit[stage_key] = shared
        self.execution.submission_counts["eager_invocations"] += 1
        return shared

    def _prime_host_queues(self):
        for stg_id in self._stage_compact_zero:
            for idx in self._stage_op_indices[stg_id]:
                pack = self._prepare_entry(idx, self.physical_ops[idx])
                if pack[3]:
                    launch, start = self._submit_compact_zero_launch(stg_id, idx, pack[2])
                    self.host_info[idx] = (pack[0], pack[1], pack[2], launch, start, pack[4])
                    break
                self.host_info[idx] = (pack[0], pack[1], pack[2], None, None, pack[4])
        for stg_id in self._host_pumped_stages:
            indices = self._stage_op_indices[stg_id]
            last_pos = -1
            last_idx = -1
            last_pack = None
            for cand_pos in range(len(indices) - 1, -1, -1):
                cand_idx = indices[cand_pos]
                pack = self._prepare_entry(cand_idx, self.physical_ops[cand_idx])
                if pack[3]:
                    last_pos, last_idx, last_pack = cand_pos, cand_idx, pack
                    break
                self.host_info[cand_idx] = (pack[0], pack[1], pack[2], None, None, pack[4])
            if last_pos < 0:
                self._stage_host_pos[stg_id] = len(indices)
                self._stage_host_last_pos[stg_id] = -1
                continue
            self._stage_host_last_pos[stg_id] = last_pos
            self._pump_stage_host(stg_id, resolve_before_last=False)
            stage_key = (self.replica, stg_id)
            if self._stage_host_in_flight[stg_id] > 0:
                before_last = self.engine.future(f"{self.prefix}:s{stg_id}:host_before_last")
                self._stage_host_before_last[stg_id] = before_last
                host_deps = before_last
            else:
                prev_h = self._stage_host_tail[stg_id]
                self.seen_host_stages.add(stage_key)
                host_deps = (self.prepared, prev_h) if prev_h is not None else self.prepared
            launch_last, start_last = self._submit_stage_host_task(
                stg_id, last_idx, last_pos, last_pack[2],
                host_deps=host_deps, attach_pump=False, publish_global=True)
            self.host_info[last_idx] = (
                last_pack[0], last_pack[1], last_pack[2], launch_last, start_last, last_pack[4])

    def _on_host_launch_done(self, _event, *, stg_id: int):
        self._stage_host_in_flight[stg_id] = max(0, self._stage_host_in_flight[stg_id] - 1)
        self._pump_stage_host(stg_id)

    def _pump_stage_host(self, stg_id: int, *, resolve_before_last: bool = True):
        indices = self._stage_op_indices[stg_id]
        last_pos = self._stage_host_last_pos[stg_id]
        woken: list[int] = []
        while self._stage_host_in_flight[stg_id] < 1 and self._stage_host_pos[stg_id] < last_pos:
            pos = self._stage_host_pos[stg_id]
            idx = indices[pos]
            if self.host_info[idx] is None:
                pack = self._prepared_pending_host.pop(idx, None)
                if pack is None:
                    pack = self._prepare_entry(idx, self.physical_ops[idx])
                if pack[3]:
                    stage_key = (self.replica, stg_id)
                    prev_h = self._stage_host_tail[stg_id]
                    if stage_key in self.seen_host_stages:
                        host_deps = prev_h
                    else:
                        self.seen_host_stages.add(stage_key)
                        host_deps = (self.prepared, prev_h) if prev_h is not None else self.prepared
                    launch, start = self._submit_stage_host_task(
                        stg_id, idx, pos, pack[2],
                        host_deps=host_deps, attach_pump=True, publish_global=False)
                else:
                    launch = start = None
                self.host_info[idx] = (pack[0], pack[1], pack[2], launch, start, pack[4])
            self._stage_host_pos[stg_id] = pos + 1
            if idx in self._awaiting_host:
                self._awaiting_host.discard(idx)
                woken.append(idx)
        if resolve_before_last and self._stage_host_pos[stg_id] >= last_pos and self._stage_host_in_flight[stg_id] == 0:
            before_last = self._stage_host_before_last[stg_id]
            if before_last is not None and not self._stage_host_before_last_resolved[stg_id]:
                self._stage_host_before_last_resolved[stg_id] = True
                self._update_host_submits()
                self.engine.resolve(before_last)
            else:
                self._update_host_submits()
        if woken:
            self.pending_activations.extend(woken)
            self._drain_activations()

    @staticmethod
    def _entry_matches_live_l2(entry, live_view, live_l2_key) -> bool:
        return (
            entry is not None
            and entry is not False
            and len(entry) >= 3
            and entry[2] == live_l2_key
            and entry[0][4].admission_memory_view is live_view
        )

    def _prepare_entry(self, idx: int, op):
        execution = self.execution
        replica, scope = self.replica, self.scope
        compact_exec, recipes_id = self.compact_exec, self.recipes_id
        prefix_colon = self.prefix_colon
        lazy_fast = self.lazy_fast
        prev_l_ops, prev_l_entries, cur_l_entries = self.prev_l_ops, self.prev_l_entries, self.cur_l_entries
        stage_id_caches, stage_rates_map = self.stage_id_caches, self.stage_rates_map
        lazy_cache = self.lazy_cache
        cost_counts, cost_invs, trans_opts = self.cost_counts, self.cost_invs, self.trans_opts
        scope_str, exec_work = self.scope_str, self.exec_work
        entry = None
        lazy_key = None
        stg_id_cache = None
        same_step_op = False
        live_view = None
        live_l2_key = None
        if lazy_fast:
            live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
            if (prev_l_ops is not None and prev_l_ops[idx] is op
                    and self._entry_matches_live_l2(
                        (pe := prev_l_entries[idx]), live_view, live_l2_key)):
                entry = pe
                same_step_op = True
            else:
                stg_id_cache = stage_id_caches[op.stage]
                hit = stg_id_cache.get(id(op))
                if (hit is not None and hit[0] is op
                        and self._entry_matches_live_l2(hit[1], live_view, live_l2_key)):
                    entry = hit[1]
                else:
                    live_rates = stage_rates_map[op.stage]
                    lazy_key = (
                        replica, scope, compact_exec, recipes_id, live_rates,
                        self._step_active_sig, live_l2_key,
                        id(live_view) if live_view is not None else None,
                        op.stage, op.family, op.recipe, op.phase, op.dtype, op.layout,
                        op.requests,
                        op.semantic_shape, op.local_shape, op.padded_shape, op.launch_shape,
                        op.flops, op.matrix_flops, op.mte2_bytes, op.mte3_bytes,
                        op.vector_ops, op.hbm_bytes, op.link_bytes, op.link_scope,
                        op.source_boundary, op.returns,
                        _freeze_key(op.precision_abi),
                        _freeze_key(op.kernel_binding),
                        _freeze_key(op.owner_schedule),
                        _freeze_key(op.logical_precision_reqs),
                        op.coverage,
                        execution._op_key(op),
                        op.stream or "main",
                        execution.stream_owners.get((replica, op.stage, op.stream or "main")),
                    )
                    cand = lazy_cache.get(lazy_key)
                    if self._entry_matches_live_l2(cand, live_view, live_l2_key):
                        entry = cand
                        if len(stg_id_cache) >= STAGE_ID_CACHE_LIMIT:
                            stg_id_cache.clear()
                        stg_id_cache[id(op)] = (op, entry)
            if (entry is not None and entry is not False
                    and entry[1] is not False):
                if cur_l_entries is not None:
                    cur_l_entries[idx] = entry
                (_res_op_tmpl, _mdiff, _rmem, _rec, _ctx_tmpl,
                 submit, transport_options, inv_cell, _rname, _oname,
                 _wdeltas) = entry[0]
                cost_counts["analytic_work"] += 1
                if not same_step_op:
                    trans_opts[(scope_str, op.op_id, replica)] = transport_options
                if inv_cell is not None:
                    launches = (
                        max(0, int(inv_cell[INV_CELL_LAUNCHES_IDX]))
                        if len(inv_cell) > INV_CELL_LAUNCHES_IDX else 1
                    )
                    if launches > 0:
                        if (inv_cell[2] is cost_invs and inv_cell[3] is not None
                                and inv_cell[3]["invocations"] > 0):
                            inv_cell[3]["invocations"] += launches
                        elif inv_cell[0] in cost_invs:
                            li = cost_invs[inv_cell[0]]
                            inv_cell[2] = cost_invs
                            inv_cell[3] = li
                            li["invocations"] += launches
                        else:
                            for _ in range(launches):
                                execution.cost.record_invocation(inv_cell[1])
                            inv_cell[2] = cost_invs
                            inv_cell[3] = cost_invs.get(inv_cell[0])
                name = prefix_colon + op.op_id
                return (entry, op, name, submit, None)

        if not lazy_fast:
            live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
        res_op, recipe, context = self._prepare_op_context(op)
        name = prefix_colon + res_op.op_id
        submit = bool(context.work or getattr(recipe, "algorithm", None))
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
            inv_record = execution._last_invocation_record
            inv_launches = (
                _declared_launch_count(context.program_recipe or recipe, context)
                if inv_record is not None else 0
            )
            inv_cell = (
                [inv_record["id"], inv_record, cost_invs,
                 cost_invs.get(inv_record["id"]), inv_launches]
                if inv_record is not None and inv_launches > 0 else None
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
            fast_created = [prep_bundle, None, live_l2_key]
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
            return (fast_created, op, name, submit, None)
        return (
            None, op, name, submit,
            (res_op, recipe, context, live_view, live_l2_key, execution._last_invocation_record),
        )

    def _materialize_non_pumped(self, idx: int, op):
        pack = self._prepare_entry(idx, op)
        launch = start = None
        if pack[3] and self.include_sub and op.stage in self._stage_compact_zero:
            launch, start = self._submit_compact_zero_launch(op.stage, idx, pack[2])
        self.host_info[idx] = (pack[0], pack[1], pack[2], launch, start, pack[4])

    def _run_bundle_in_scope(
        self, op, name, boundary, entry, op_template, meta_diff, context_template,
        stage, hbm, bindings, memory_context, fast_memory_context,
    ):
        engine, replica, owner, compact_exec = self.engine, self.replica, self.owner, self.compact_exec
        execution = self.execution
        if bindings is None and fast_memory_context is not None:
            previous_context = engine._memory_context
            previous_fast_context = engine._memory_context_fast
            engine._memory_context = memory_context
            engine._memory_context_fast = fast_memory_context
            try:
                return run_cached_bundle(
                    execution, engine, replica, owner, compact_exec, op, name, boundary,
                    entry, op_template, meta_diff, context_template, scope=self.scope,
                    prepare_op_context=self._prepare_op_context)
            finally:
                engine._memory_context = previous_context
                engine._memory_context_fast = previous_fast_context
        with engine.memory_scope(
                execution.memory, replica=replica, stage=int(stage),
                pools=(hbm,), bindings=bindings):
            return run_cached_bundle(
                execution, engine, replica, owner, compact_exec, op, name, boundary,
                entry, op_template, meta_diff, context_template, scope=self.scope,
                prepare_op_context=self._prepare_op_context)

    def _complete_async(self, _event, *, complete, idx):
        self.engine.resolve(complete)
        self._completed(idx)

    def _begin_async(
        self, boundary, *, stage, hbm, bindings, memory_context, fast_memory_context,
        complete, idx, op, name, entry, op_template, meta_diff, context_template,
    ):
        live_view, live_l2_key = live_stage_l2_state(self.execution, self.replica, op.stage)
        if not self._entry_matches_live_l2(entry, live_view, live_l2_key):
            old_deltas = entry[0][10]
            undo_superseded_invocation(self.execution, entry[0][7])
            pack = self._prepare_entry(idx, op)
            if pack[0] is not None:
                entry = pack[0]
                op_template, meta_diff, raw_meta_mem, _rec, context_template = entry[0][:5]
                new_deltas = entry[0][10]
                if new_deltas != old_deltas:
                    for key, delta in old_deltas:
                        self.exec_work[key] -= delta
                    for key, delta in new_deltas:
                        self.exec_work[key] += delta
                if self.memory_bindings is None:
                    bindings = (
                        {mn: dict(mv) for mn, mv in raw_meta_mem.items()}
                        if raw_meta_mem is not None else None
                    )
                else:
                    meta = op.metadata
                    if meta_diff is not None:
                        patched = _meta_patch(dict(op.metadata), meta_diff)
                        meta = tuple(sorted(patched.items()))
                    op_for_mem = replace(op_template, op_id=op.op_id, metadata=meta)
                    bindings = memory_bindings_for(op_for_mem, self.memory_bindings)
        finish = self._run_bundle_in_scope(
            op, name, boundary, entry, op_template, meta_diff, context_template,
            stage, hbm, bindings, memory_context, fast_memory_context)
        self.engine.on_complete(
            finish, partial(self._complete_async, complete=complete, idx=idx), native_hook=True)

    def _activate_cached(self, idx, op, orig_op, name, launch, fast_entry):
        execution = self.execution
        engine, replica, owner = self.engine, self.replica, self.owner
        compact_exec, observer = self.compact_exec, self.observer
        streams, dep_specs = self.streams, self.dep_specs
        op_events, order, ext_map = self.op_events, self.order, self.ext_map
        input_ready, schedule = self.input_ready, self.schedule
        stage_mem_info, memory_bindings = self.stage_mem_info, self.memory_bindings
        recipes_used, exec_work = self.recipes_used, self.exec_work
        n_ops, host_lookahead = self.n_ops, self.host_lookahead
        op_futures, succ_idx = self.op_futures, self.succ_idx
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
            physical_hbm = execution.stage_pools.get(
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
        elif (not all_done and compact_exec and observer is None
              and execution._live_admission_memory_view() is None):
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
            complete = self._run_bundle_in_scope(
                op, name, sync_ev, fast_entry, res_op_tmpl, meta_diff, context_tmpl,
                res_stage, physical_hbm, mem_bind, mctx_fast_dict, mfast_tuple)
            streams[stream_key] = complete
            op_events[idx] = complete
            recipes_used[recipe_name] = recipes_used.get(recipe_name, 0) + 1
            if observer is not None:
                observer(orig_op, replica, obs_prog_name, sync_ev, complete)
            for key, delta in work_deltas:
                exec_work[key] += delta
            if complete.time is not None:
                if op_futures[idx] is not None or succ_idx[idx]:
                    self._completed(idx)
            elif op_futures[idx] is not None or succ_idx[idx]:
                t_comp = engine._tasks.get(complete.sequence)
                cb = lambda _ev, _i=idx: self._completed(_i)
                if engine._accel is None and t_comp is not None and t_comp.callback is None:
                    t_comp.callback = cb
                else:
                    engine.on_complete(complete, cb, native_hook=True)
            return

        complete = engine.future(name + ":program_complete")
        async_begin = partial(
            self._begin_async, stage=int(res_stage), hbm=physical_hbm,
            bindings=mem_bind, memory_context=mctx_fast_dict, fast_memory_context=mfast_tuple,
            complete=complete, idx=idx, op=op, name=name, entry=fast_entry,
            op_template=res_op_tmpl, meta_diff=meta_diff, context_template=context_tmpl,
        )
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

    def _activate_slow_async(
        self, idx: int, op, orig_op, name: str, begin_after, physical_hbm: str,
        res_op, recipe, context, prep_view, prep_l2_key, prep_inv_rec,
    ):
        execution, engine, replica, owner = self.execution, self.engine, self.replica, self.owner
        observer, memory_bindings, exec_work = self.observer, self.memory_bindings, self.exec_work
        complete = engine.future(name + ":program_complete")
        ready_marker = engine.submit(
            name + (":begin" if observer is None else ":ready"),
            after=begin_after, owner=owner, native_hook=True,
        )

        def _begin_slow(_event):
            cur_res_op, cur_recipe, cur_ctx = res_op, recipe, context
            cur_inv_rec = prep_inv_rec
            cur_view, cur_l2_key = live_stage_l2_state(execution, replica, op.stage)
            if cur_view is not prep_view or cur_l2_key != prep_l2_key:
                prep_launches = (
                    _declared_launch_count(context.program_recipe or recipe, context)
                    if prep_inv_rec is not None else 0
                )
                undo_invocation_record(execution, prep_inv_rec, launch_count=prep_launches)
                cur_res_op, cur_recipe, cur_ctx = self._prepare_op_context(op)
                cur_inv_rec = execution._last_invocation_record
                for key in exec_work:
                    delta = float(getattr(cur_res_op, key)) - float(getattr(res_op, key))
                    if delta != 0.0:
                        exec_work[key] += delta
            if not type(execution).schedule_replay:
                with engine.memory_scope(
                    execution.memory, replica=replica, stage=int(cur_res_op.stage),
                    pools=(physical_hbm,),
                    bindings=memory_bindings_for(cur_res_op, memory_bindings),
                ):
                    finish = _lower_and_start_uncached(
                        execution, engine, replica, owner, self.compact_exec,
                        cur_res_op, cur_recipe, cur_ctx, name, ready_marker,
                        orig_op=op, scope=self.scope, live_view=cur_view,
                        live_l2_key=cur_l2_key, inv_record=cur_inv_rec,
                        prepare_op_context=self._prepare_op_context,
                    )
            else:
                finish = execution._start_program(
                    cur_ctx, cur_recipe, name, ready_marker, owner=owner, replica=replica,
                    memory_ledger=execution.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(cur_res_op, memory_bindings),
                )
            engine.on_complete(
                finish, partial(self._complete_async, complete=complete, idx=idx), native_hook=True,
            )

        engine.on_complete(ready_marker, _begin_slow, native_hook=True)
        return ready_marker, complete

    def _activate(self, idx: int):
        execution = self.execution
        engine, replica, owner = self.engine, self.replica, self.owner
        host_info, physical_ops, streams = self.host_info, self.physical_ops, self.streams
        dep_specs, op_events, order = self.dep_specs, self.op_events, self.order
        ext_map, input_ready, schedule = self.ext_map, self.input_ready, self.schedule
        observer, memory_bindings = self.observer, self.memory_bindings
        recipes_used, exec_work = self.recipes_used, self.exec_work
        op = physical_ops[idx]
        info = host_info[idx]
        if info is None:
            if op.stage in self._host_pumped_stages:
                pack = self._prepared_pending_host.get(idx)
                if pack is None:
                    pack = self._prepare_entry(idx, op)
                if pack[3]:
                    self._prepared_pending_host[idx] = pack
                    self._awaiting_host.add(idx)
                    return
                info = (pack[0], pack[1], pack[2], None, None, pack[4])
                host_info[idx] = info
            else:
                self._materialize_non_pumped(idx, op)
                info = host_info[idx]
        fast_entry, orig_op, name, launch, _start, slow_pack = info
        if fast_entry is not None and fast_entry[1] is not False:
            live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
            if not self._entry_matches_live_l2(fast_entry, live_view, live_l2_key):
                undo_superseded_invocation(execution, fast_entry[0][7])
                pack = self._prepare_entry(idx, op)
                fast_entry, orig_op, name, slow_pack = pack[0], pack[1], pack[2], pack[4]
                host_info[idx] = (fast_entry, orig_op, name, launch, _start, slow_pack)
        if fast_entry is not None and fast_entry[1] is False:
            live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
            res_op, recipe, context = self._prepare_op_context(op)
            slow_pack = (res_op, recipe, context, live_view, live_l2_key, execution._last_invocation_record)
            fast_entry = None
        if fast_entry is not None:
            self._activate_cached(idx, op, orig_op, name, launch, fast_entry)
            return

        res_op, recipe, context, prep_view, prep_l2_key, prep_inv_rec = slow_pack
        stage_key = replica, res_op.stage
        stream_key = replica, res_op.stage, res_op.stream or "main"
        tail = streams.get(stream_key)
        begin_after = collect_begin_after(
            idx, op, launch, tail, dep_specs, op_events,
            order, ext_map, input_ready, schedule)
        physical_hbm = execution.stage_pools.get(stage_key, {}).get("hbm", "hbm")
        live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
        has_pending = any(ev.time is None or ev.time > engine.now for ev in begin_after)
        if live_view is not None and (has_pending or not type(execution).schedule_replay):
            ready, complete = self._activate_slow_async(
                idx, op, orig_op, name, begin_after, physical_hbm,
                res_op, recipe, context, prep_view, prep_l2_key, prep_inv_rec,
            )
        else:
            if live_view is not None and (live_view is not prep_view or live_l2_key != prep_l2_key):
                prep_launches = (
                    _declared_launch_count(context.program_recipe or recipe, context)
                    if prep_inv_rec is not None else 0
                )
                undo_invocation_record(execution, prep_inv_rec, launch_count=prep_launches)
                res_op, recipe, context = self._prepare_op_context(op)
            if observer is None:
                ready = None
                complete = execution._start_program(
                    context, recipe, name, None, owner=owner, replica=replica,
                    memory_ledger=execution.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: self._completed(_i), begin_after=begin_after)
            else:
                ready = engine.submit(name + ":ready", after=begin_after, owner=owner)
                complete = execution._start_program(
                    context, recipe, name, ready, owner=owner, replica=replica,
                    memory_ledger=execution.memory, memory_pools=(physical_hbm,),
                    memory_bindings=memory_bindings_for(res_op, memory_bindings),
                    on_resolve=lambda _i=idx: self._completed(_i))
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

    def _drain_activations(self):
        if self.activating:
            return
        self.activating = True
        try:
            while self.pending_activations:
                self._activate(self.pending_activations.popleft())
        finally:
            self.activating = False

    def _completed(self, idx: int):
        engine, op_futures, succ_idx, unmet, pending_activations = self.engine, self.op_futures, self.succ_idx, self.unmet, self.pending_activations
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
                if self.activating:
                    pending_activations.append(s0)
                else:
                    self.activating = True
                    try:
                        self._activate(s0)
                        while pending_activations:
                            self._activate(pending_activations.popleft())
                    finally:
                        self.activating = False
            return
        if not n_s:
            return
        ready_ops = [s for s in succs if (unmet.__setitem__(s, unmet[s] - 1) or unmet[s] == 0)]
        if ready_ops:
            if len(ready_ops) > 1:
                ready_ops.sort()
            pending_activations.extend(ready_ops)
            self._drain_activations()

    def finish(self, work, release_ms, emitted_tokens):
        engine, prefix, owner = self.engine, self.prefix, self.owner
        replica, scope = self.replica, self.scope
        input_ready, physical_ops = self.input_ready, self.physical_ops
        op_futures, sink_idx = self.op_futures, self.sink_idx
        declared_commits = self.declared_commits
        execution = self.execution
        self._prime_host_queues()
        if self.root_idx:
            self.pending_activations.extend(self.root_idx)
            self._drain_activations()
        device_complete = engine.submit(
            prefix + ":device_complete", owner=owner,
            after=tuple(op_futures[si] for si in sink_idx) or (input_ready,))
        completion = engine.submit(
            prefix + ":complete", after=(device_complete,),
            latency=(execution.submission.completion_ms if execution.include_submission else 0),
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
                prefix + ":step_ready", after=(completion, host_complete), owner=owner)
        from ..offload import release_offload_access
        offload_release = engine.submit(
            prefix + ":offload_release", after=(device_complete,),
            owner=owner, native_hook=True,
            on_complete=lambda _event: release_offload_access(execution, work))
        step = ScheduledIteration(
            replica, work.plan.iteration, completion, release_ms,
            work.mode, work.valid_tokens, work.padded_tokens,
            work.graph_bucket, scope, len(physical_ops), commits,
            device_complete, host_results, clients,
            work.implementation_selections,
            self._current_host_submits(),
            dict(declared_commits), dict(fallback_commits),
            offload_release)
        self.step = step
        self._update_host_submits()
        engine.submit(prefix + ":step_metrics", after=(completion,),
                      on_complete=self._record_step, native_hook=True)
        return step

    def _record_step(self, _event):
        if hasattr(self.engine, "_compact_step"):
            self.engine._compact_step = False
        self._update_host_submits()
        self.execution.step_summary.record(self.step)


class _LazySubmitMixin:
    """Active-frontier DAG submission fast path for ``Execution``."""

    def _add_lazy(self, work, replica, release_ms, *, scope, schedule,
                  prepared, input_ready, physical_ops, prefix,
                  external_inputs, emitted_tokens, memory_bindings,
                  state_accesses, verified_dag=None):
        """Materialize only the active dependency frontier."""
        frontier = _LazyFrontier(
            self, replica, scope=scope, schedule=schedule, prepared=prepared,
            input_ready=input_ready, physical_ops=physical_ops, prefix=prefix,
            external_inputs=external_inputs, memory_bindings=memory_bindings,
            verified_dag=verified_dag,
        )
        return frontier.finish(work, release_ms, emitted_tokens)
