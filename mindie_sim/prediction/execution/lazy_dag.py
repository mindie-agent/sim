"""Verified DAG builders and cached bundle execution helpers for lazy submit.

[Role]
  Provide verified DAG successor/predecessor indexing, stage memory-scope
  tuples, dependency event collection, and cached bundle execution for
  ``mindie_sim.prediction.execution.lazy_submit``.

[Invariants]
  * Dependency cycles raise ``ModelingError("execution_dependency_cycle")``
    before admitting any Engine task.
  * Lowered program caches store immutable programs and mapping/service calls
    only, never simulated durations.

[Timeouts & Locks]
  Bounded in-memory graph traversal and dict lookups; no I/O or locks.

[Side-Effect Policy]
  Submits tasks only to the caller's ``Engine`` and updates bounded caches.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from typing import Any

from mindie_sim.schema.common import ModelingError
from ..program import KernelProgram, NativeProgram
from ..program.invocation import _declared_launch_count
from .context import _admission_l2_cache_key
from .streams import _meta_patch

INV_CELL_LAUNCHES_IDX = 4


def live_stage_l2_state(execution, replica: int, stage: int):
    """Return ``(view, l2_cache_key)`` for the live stage admission memory view."""
    view = execution._live_admission_memory_view()
    if view is None:
        return None, None
    stage_geom = execution.stage_geometry.get((replica, stage))
    die_rank = getattr(stage_geom, "rank", None)
    if die_rank is None:
        dies = execution.stage_dies.get((replica, stage))
        if dies:
            die_rank = int(dies[0].rank)
    die_id = 0 if die_rank is None else int(die_rank)
    return view, _admission_l2_cache_key(view, die_id)


def undo_invocation_record(execution, inv_record, *, launch_count: int = 1) -> None:
    """Roll back one superseded invocation count by invocation record dict."""
    launches = max(0, int(launch_count))
    if inv_record is not None and launches > 0:
        inv_id = inv_record.get("id")
        if inv_id is not None:
            cost_invs = execution.cost.invocations
            rec = cost_invs.get(inv_id)
            if rec is not None and rec["invocations"] > 0:
                remaining = rec["invocations"] - launches
                if remaining > 0:
                    rec["invocations"] = remaining
                else:
                    rec["invocations"] = 0
                    cost_invs.pop(inv_id, None)
    if execution.cost.counts.get("analytic_work", 0) > 0:
        execution.cost.counts["analytic_work"] -= 1


def undo_superseded_invocation(execution, inv_cell) -> None:
    """Roll back one speculative invocation count when L2 state supersedes a prep."""
    if inv_cell is not None:
        launches = (
            max(0, int(inv_cell[INV_CELL_LAUNCHES_IDX]))
            if len(inv_cell) > INV_CELL_LAUNCHES_IDX else 1
        )
        if launches > 0:
            cost_invs = execution.cost.invocations
            rec = inv_cell[3] if inv_cell[2] is cost_invs else cost_invs.get(inv_cell[0])
            if rec is not None and rec["invocations"] > 0:
                remaining = rec["invocations"] - launches
                if remaining > 0:
                    rec["invocations"] = remaining
                else:
                    rec["invocations"] = 0
                    cost_invs.pop(inv_cell[0], None)
                    if inv_cell[2] is cost_invs:
                        inv_cell[3] = None
    if execution.cost.counts.get("analytic_work", 0) > 0:
        execution.cost.counts["analytic_work"] -= 1


def _invocation_startup_ms(prog) -> float:
    return max(
        (prog.timing.startup_ms, *(c.startup_ms for c in prog.constraints)),
        default=prog.timing.startup_ms,
    )


def _extract_boundary_stage_bodies(program, *, compact_exec: bool) -> dict[str, Any]:
    """Map named boundary stages from a freshly lowered program to single-stage bodies."""
    if not isinstance(program, KernelProgram):
        return {}
    stages = program.stages
    if (len(stages) == 2 and stages[0].name == "setup"
            and stages[1].name == "body" and isinstance(stages[1].program, KernelProgram)):
        stages = stages[1].program.stages
    coalesce = None
    if compact_exec:
        from ..program import coalesce_execution_program
        coalesce = coalesce_execution_program
    bodies: dict[str, Any] = {}
    for stage in stages:
        body = KernelProgram((replace(stage, after=()),))
        bodies[stage.name] = coalesce(body) if coalesce is not None else body
    return bodies


def _wrap_boundary_startup_program(
    program, *, execution, replica: int, scope: str, compact_exec: bool,
    op, prev_res_op, inv_record, live_view, live_l2_key,
    prepare_op_context=None, launch_count: int = 1,
):
    """Attach shared live L2 rebind resolvers to named multi-launch boundary stages."""
    from ..program.invocation import InvocationProgram, _FloorJoinProgram

    if isinstance(program, _FloorJoinProgram):
        wrapped_inner = _wrap_boundary_startup_program(
            program.body, execution=execution, replica=replica, scope=scope,
            compact_exec=compact_exec, op=op, prev_res_op=prev_res_op,
            inv_record=inv_record, live_view=live_view, live_l2_key=live_l2_key,
            prepare_op_context=prepare_op_context, launch_count=launch_count,
        )
        return program if wrapped_inner is program.body else replace(program, body=wrapped_inner)
    if not isinstance(program, KernelProgram):
        return program
    if (len(program.stages) == 2 and program.stages[0].name == "setup"
            and program.stages[1].name == "body" and program.stages[1].program is not None):
        wrapped_body = _wrap_boundary_startup_program(
            program.stages[1].program, execution=execution, replica=replica, scope=scope,
            compact_exec=compact_exec, op=op, prev_res_op=prev_res_op,
            inv_record=inv_record, live_view=live_view, live_l2_key=live_l2_key,
            prepare_op_context=prepare_op_context, launch_count=launch_count,
        )
        if wrapped_body is program.stages[1].program:
            return program
        return replace(program, stages=(program.stages[0], replace(program.stages[1], program=wrapped_body)))
    if not any(
        isinstance(st.program, InvocationProgram) and _invocation_startup_ms(st.program) > 0.0
        for st in program.stages
    ):
        return program

    state: dict[str, Any] = {
        "view": live_view,
        "key": live_l2_key,
        "prev_res_op": prev_res_op,
        "inv_record": inv_record,
        "launch_count": launch_count,
        "fresh_stages": None,
    }

    def _make_stage_resolver(stage_name: str, orig_body):
        def resolve_stage_body():
            cur_view, cur_l2_key = live_stage_l2_state(execution, replica, op.stage)
            if cur_view is state["view"] and cur_l2_key == state["key"]:
                fresh_map = state["fresh_stages"]
                return fresh_map.get(stage_name, orig_body) if fresh_map is not None else orig_body
            undo_invocation_record(execution, state["inv_record"], launch_count=state["launch_count"])
            if prepare_op_context is not None:
                cur_res_op, cur_recipe, cur_ctx = prepare_op_context(op)
            else:
                cur_res_op, cur_recipe = execution.prepare(op, replica)
                cur_ctx = execution._context(cur_res_op, replica, scope, recipe=cur_recipe)
            for key in execution.work:
                delta = float(getattr(cur_res_op, key)) - float(getattr(state["prev_res_op"], key))
                if delta != 0.0:
                    execution.work[key] += delta
            fresh = cur_recipe.lower(replace(cur_ctx, minimum_ms=0, invocation_timing=None))
            fresh = execution._apply_resource_service_program(cur_ctx, fresh)
            fresh = execution._apply_stream_owners_program(cur_ctx, fresh)
            fresh_map = _extract_boundary_stage_bodies(fresh, compact_exec=compact_exec)
            cur_inv = execution._last_invocation_record
            state["view"] = cur_view
            state["key"] = cur_l2_key
            state["prev_res_op"] = cur_res_op
            state["inv_record"] = cur_inv
            state["launch_count"] = (
                _declared_launch_count(cur_ctx.program_recipe or cur_recipe, cur_ctx)
                if cur_inv is not None else 0
            )
            state["fresh_stages"] = fresh_map
            return fresh_map.get(stage_name, orig_body)

        return resolve_stage_body

    new_stages = tuple(
        replace(
            st,
            program=replace(st.program, body_resolver=_make_stage_resolver(st.name, st.program.body)),
        )
        if (isinstance(st.program, InvocationProgram) and _invocation_startup_ms(st.program) > 0.0)
        else st
        for st in program.stages
    )
    return replace(program, stages=new_stages)


def _wrap_live_startup_program(
    program, *, execution, replica: int, scope: str, compact_exec: bool,
    op, prev_res_op, inv_record, live_view, live_l2_key,
    prepare_op_context=None, launch_count: int = 1,
):
    """Re-bind an InvocationProgram body after startup if live L2 state changed."""
    if live_view is None:
        return program
    from ..program.invocation import InvocationProgram
    if not isinstance(program, InvocationProgram):
        return _wrap_boundary_startup_program(
            program, execution=execution, replica=replica, scope=scope,
            compact_exec=compact_exec, op=op, prev_res_op=prev_res_op,
            inv_record=inv_record, live_view=live_view, live_l2_key=live_l2_key,
            prepare_op_context=prepare_op_context, launch_count=launch_count,
        )
    if _invocation_startup_ms(program) <= 0.0:
        return program

    def resolve_body():
        cur_view, cur_l2_key = live_stage_l2_state(execution, replica, op.stage)
        if cur_view is live_view and cur_l2_key == live_l2_key:
            return program.body
        undo_invocation_record(execution, inv_record, launch_count=launch_count)
        if prepare_op_context is not None:
            cur_res_op, cur_recipe, cur_ctx = prepare_op_context(op)
        else:
            cur_res_op, cur_recipe = execution.prepare(op, replica)
            cur_ctx = execution._context(cur_res_op, replica, scope, recipe=cur_recipe)
        for key in execution.work:
            delta = float(getattr(cur_res_op, key)) - float(getattr(prev_res_op, key))
            if delta != 0.0:
                execution.work[key] += delta
        fresh = cur_recipe.lower(replace(cur_ctx, minimum_ms=0, invocation_timing=None))
        fresh = execution._apply_resource_service_program(cur_ctx, fresh)
        fresh = execution._apply_stream_owners_program(cur_ctx, fresh)
        if compact_exec:
            from ..program import coalesce_execution_program
            fresh = coalesce_execution_program(fresh)
        return fresh

    return replace(program, body_resolver=resolve_body)


def build_verified_dag(physical_ops, prev_dag):
    """Build or reuse verified DAG successor/predecessor indices."""
    n_ops = len(physical_ops)
    stream_sig = tuple((op.stage, op.stream or "main") for op in physical_ops)
    dag_sig = tuple(
        (op.op_id, op.deps, stream_sig[idx], op.family == "token_commit")
        for idx, op in enumerate(physical_ops)
    )
    if prev_dag is not None and prev_dag[0] == dag_sig:
        return prev_dag[1], prev_dag
    order = {op.op_id: index for index, op in enumerate(physical_ops)}
    succ_lists: list[list[int]] = [[] for _ in range(n_ops)]
    init_unmet = [0] * n_ops
    dep_specs_list: list[Any] = [None] * n_ops
    last_on_stream: dict[tuple[int, str], int] = {}
    for idx, op in enumerate(physical_ops):
        stream_key = stream_sig[idx]
        stream_pred = last_on_stream.get(stream_key)
        last_on_stream[stream_key] = idx
        raw_deps = op.deps
        if not raw_deps and stream_pred is None:
            dep_specs_list[idx] = ()
            continue
        if not raw_deps and stream_pred is not None:
            init_unmet[idx] = 1
            succ_lists[stream_pred].append(idx)
            dep_specs_list[idx] = stream_pred
            continue
        if len(raw_deps) == 1:
            j0 = order.get(raw_deps[0])
            if j0 is not None and (stream_pred is None or stream_pred == j0):
                init_unmet[idx] = 1
                succ_lists[j0].append(idx)
                dep_specs_list[idx] = j0
                continue
        int_d: list[int] = []
        seen_int: set[int] = set()
        all_internal = True
        for dep in raw_deps:
            j = order.get(dep)
            if j is not None:
                if j not in seen_int:
                    seen_int.add(j)
                    int_d.append(j)
                    succ_lists[j].append(idx)
            else:
                all_internal = False
        if stream_pred is not None and stream_pred not in seen_int:
            seen_int.add(stream_pred)
            int_d.append(stream_pred)
            succ_lists[stream_pred].append(idx)
        init_unmet[idx] = len(int_d)
        dep_specs_list[idx] = (
            int_d[0] if all_internal and len(int_d) == 1
            else (tuple(int_d) if all_internal else None)
        )
    root_idx = tuple(i for i in range(n_ops) if not init_unmet[i])
    if n_ops and not root_idx:
        raise ModelingError("execution_dependency_cycle")
    rem_in = list(init_unmet)
    reach_count = 0
    stack = list(root_idx)
    while stack:
        cur = stack.pop()
        reach_count += 1
        for nxt in succ_lists[cur]:
            rem_in[nxt] -= 1
            if rem_in[nxt] == 0:
                stack.append(nxt)
    if reach_count != n_ops:
        raise ModelingError("execution_dependency_cycle")
    sink_idx = tuple(i for i in range(n_ops) if not succ_lists[i])
    stream_tail_indices = set(last_on_stream.values())
    fut_indices = tuple(
        i for i in range(n_ops)
        if not succ_lists[i] or i in stream_tail_indices
        or physical_ops[i].family == "token_commit"
    )
    packed = (
        order, tuple(tuple(s) for s in succ_lists),
        tuple(init_unmet), tuple(dep_specs_list),
        root_idx, sink_idx, fut_indices, stream_sig,
        tuple(last_on_stream.items()),
    )
    return packed, (dag_sig, packed)


def stage_memory_info(execution, replica: int):
    """Build per-stage memory-scope tuples for fast-path execution."""
    engine = execution.engine
    has_fast_mem = hasattr(engine, "_memory_context_fast")
    default_ledger = (
        execution.memory if execution.memory is not None
        else getattr(engine, "memory", None)
    )
    if has_fast_mem and default_ledger is not None:
        engine._memory_ledgers[id(default_ledger)] = default_ledger
    result: dict[int, tuple[str, dict | None, tuple | None]] = {}
    for (rep, stg) in execution.stage_names:
        if rep != replica:
            continue
        phbm = execution.stage_pools.get((rep, stg), {}).get("hbm", "hbm")
        if has_fast_mem:
            pfroz = (
                engine._intern_pool_set((phbm,))
                if hasattr(engine, "_intern_pool_set")
                else frozenset((str(phbm),))
            )
            rep_i, stg_i = int(replica), int(stg)
            mctx = {
                "memory": default_ledger, "pools": pfroz,
                "replica": rep_i, "stage": stg_i, "bindings": {},
            }
            mfast = (default_ledger, pfroz, None, rep_i, stg_i)
            result[stg] = (phbm, mctx, mfast)
        else:
            result[stg] = (phbm, None, None)
    return result


def collect_begin_after(idx, op, launch, tail, dep_specs, op_events,
                        order, ext_map, input_ready, schedule):
    """Gather causal predecessor Events for one lazily activated op."""
    dspec = dep_specs[idx]
    if dspec.__class__ is int:
        d_ev = op_events[dspec]
        if d_ev is not None:
            if tail is d_ev:
                tail = None
            if launch is not None and tail is not None:
                return (input_ready, d_ev, launch, tail)
            if launch is not None:
                return (input_ready, d_ev, launch)
            if tail is not None:
                return (input_ready, d_ev, tail)
            return (input_ready, d_ev)
        if launch is not None and tail is not None:
            return (input_ready, launch, tail)
        if launch is not None:
            return (input_ready, launch)
        if tail is not None:
            return (input_ready, tail)
        return (input_ready,)
    deps = [input_ready]
    if dspec is not None:
        for j in dspec:
            ev = op_events[j]
            if ev is not None:
                deps.append(ev)
    else:
        for dep in op.deps:
            j = order.get(dep)
            if j is not None:
                ev = op_events[j]
            elif ext_map is not None:
                ev = ext_map.get(dep)
            else:
                ev = None
            if ev is not None:
                deps.append(ev)
    if launch is not None:
        deps.append(launch)
    if tail is not None:
        deps.append(tail)
    return tuple(deps) or (schedule,)


def _lower_and_start_uncached(
    execution, engine, replica, owner, compact_exec,
    cur_res_op, recipe, ctx, name, ready_ev, *,
    orig_op=None, scope="none", live_view=None, live_l2_key=None,
    inv_record=None, prepare_op_context=None,
):
    """Lower and start one op directly when cached L2 residency state is stale."""
    execution.replay_counts["real"] += 1
    program = execution.lower_program(ctx, recipe)
    program = execution._apply_resource_service_program(ctx, program)
    program = execution._apply_stream_owners_program(ctx, program)
    if compact_exec:
        from ..program import coalesce_execution_program
        program = coalesce_execution_program(program)
    is_native_prog = isinstance(program, NativeProgram)
    has_handles = is_native_prog and hasattr(
        program.program, "start_with_handles")
    transport = (
        execution.native_transport(cur_res_op, replica)
        if getattr(recipe, "native", False) else None
    )
    if has_handles and cur_res_op.op_id not in execution.native_group_launches:
        execution._record_single_native(
            ctx, program, replica, name, transport=transport)
    if is_native_prog:
        if transport is None:
            return program.program.start(engine, name, ready_ev)
        return program.start(engine, name, ready_ev, transport=transport)
    inv_launches = (
        _declared_launch_count(ctx.program_recipe or recipe, ctx)
        if inv_record is not None else 0
    )
    program = _wrap_live_startup_program(
        program, execution=execution, replica=replica, scope=scope,
        compact_exec=compact_exec,
        op=cur_res_op if orig_op is None else orig_op,
        prev_res_op=cur_res_op, inv_record=inv_record,
        live_view=live_view, live_l2_key=live_l2_key,
        prepare_op_context=prepare_op_context, launch_count=inv_launches,
    )
    return program.start(engine, name, ready_ev, owner=owner)


def run_cached_bundle(execution, engine, replica, owner, compact_exec,
                      op, name, ready_ev, fast_entry, res_op_tmpl,
                      meta_diff, context_tmpl, *, scope="none",
                      prepare_op_context=None):
    """Execute one cached lazy-submit operator bundle, lowering on first miss."""
    live_view, live_l2_key = live_stage_l2_state(execution, replica, op.stage)
    if ((len(fast_entry) >= 3 and fast_entry[2] != live_l2_key)
            or context_tmpl.admission_memory_view is not live_view):
        undo_superseded_invocation(execution, fast_entry[0][7])
        if prepare_op_context is not None:
            cur_res_op, recipe, ctx = prepare_op_context(op)
        else:
            cur_res_op, recipe = execution.prepare(op, replica)
            ctx = execution._context(cur_res_op, replica, scope, recipe=recipe)
        for key in execution.work:
            delta = float(getattr(cur_res_op, key)) - float(getattr(res_op_tmpl, key))
            if delta != 0.0:
                execution.work[key] += delta
        return _lower_and_start_uncached(
            execution, engine, replica, owner, compact_exec,
            cur_res_op, recipe, ctx, name, ready_ev,
            orig_op=op, scope=scope, live_view=live_view, live_l2_key=live_l2_key,
            inv_record=execution._last_invocation_record,
            prepare_op_context=prepare_op_context,
        )
    inv_cell = fast_entry[0][7]
    inv_rec = inv_cell[1] if inv_cell is not None else None
    inv_launches = (
        max(0, int(inv_cell[INV_CELL_LAUNCHES_IDX]))
        if (inv_cell is not None and len(inv_cell) > INV_CELL_LAUNCHES_IDX)
        else (1 if inv_cell is not None else 0)
    )
    prog_bundle = fast_entry[1]
    if prog_bundle is not None:
        (program, mapped_entries, svc_calls,
         is_native_prog, has_handles, transport) = prog_bundle
        execution.replay_counts["real"] += 1
        if mapped_entries and context_tmpl.record_mapping is not None:
            mappings = execution.cost.mappings
            for m_item in mapped_entries:
                if m_item[2] is mappings and m_item[3] is not None:
                    m_item[3]["invocations"] += 1
                elif m_item[0] in mappings:
                    lm = mappings[m_item[0]]
                    m_item[2] = mappings
                    m_item[3] = lm
                    lm["invocations"] += 1
                else:
                    context_tmpl.record_mapping(m_item[1])
                    m_item[2] = mappings
                    m_item[3] = mappings.get(m_item[0])
        if svc_calls and context_tmpl.record_service is not None:
            for sargs, skwargs in svc_calls:
                context_tmpl.record_service(*sargs, **skwargs)
        if has_handles and op.op_id not in execution.native_group_launches:
            meta = op.metadata
            if meta_diff is not None:
                patched = _meta_patch(dict(op.metadata), meta_diff)
                meta = tuple(sorted(patched.items()))
            cur_res_op = (
                res_op_tmpl if res_op_tmpl.op_id == op.op_id
                else replace(res_op_tmpl, op_id=op.op_id,
                             layer=op.layer, metadata=meta)
            )
            execution._record_single_native(
                replace(context_tmpl, op=cur_res_op),
                program, replica, name, transport=transport)
        if is_native_prog:
            if transport is None:
                return program.program.start(engine, name, ready_ev)
            return program.start(engine, name, ready_ev, transport=transport)
        run_prog = _wrap_live_startup_program(
            program, execution=execution, replica=replica, scope=scope,
            compact_exec=compact_exec, op=op, prev_res_op=res_op_tmpl,
            inv_record=inv_rec, live_view=live_view, live_l2_key=live_l2_key,
            prepare_op_context=prepare_op_context, launch_count=inv_launches,
        )
        return run_prog.start(engine, name, ready_ev, owner=owner)

    recipe = fast_entry[0][3]
    meta = op.metadata
    if meta_diff is not None:
        patched = _meta_patch(dict(op.metadata), meta_diff)
        meta = tuple(sorted(patched.items()))
    cur_res_op = (
        res_op_tmpl if res_op_tmpl.op_id == op.op_id
        else replace(
            res_op_tmpl, op_id=op.op_id, layer=op.layer,
            deps=op.deps, requests=op.requests,
            stream=op.stream, notes=op.notes, metadata=meta)
    )
    ctx = replace(context_tmpl, op=cur_res_op, admission_memory_view=live_view)
    mapped_list = []
    svc_list = []
    orig_rec = ctx.record_mapping
    orig_svc = ctx.record_service

    def cap_map(rec):
        mapped_list.append(deepcopy(rec))
        if orig_rec is not None:
            orig_rec(rec)

    def cap_svc(*args, **kwargs):
        svc_list.append((deepcopy(args), deepcopy(kwargs)))
        if orig_svc is not None:
            orig_svc(*args, **kwargs)

    run_ctx = replace(
        ctx,
        record_mapping=cap_map if orig_rec is not None else None,
        record_service=cap_svc if orig_svc is not None else None,
    )
    execution.replay_counts["real"] += 1
    program = execution.lower_program(run_ctx, recipe)
    program = execution._apply_resource_service_program(ctx, program)
    program = execution._apply_stream_owners_program(ctx, program)
    if compact_exec:
        from ..program import coalesce_execution_program
        program = coalesce_execution_program(program)
    is_native_prog = isinstance(program, NativeProgram)
    has_handles = is_native_prog and hasattr(
        program.program, "start_with_handles")
    transport = (
        execution.native_transport(cur_res_op, replica)
        if getattr(recipe, "native", False) else None
    )
    mappings_init = execution.cost.mappings
    mapped_entries = tuple(
        [mk := json.dumps(m, sort_keys=True, separators=(",", ":")),
         m, mappings_init, mappings_init.get(mk)]
        for m in mapped_list
    )
    fast_entry[1] = (
        program, mapped_entries, tuple(svc_list),
        is_native_prog, has_handles, transport,
    )
    if has_handles and cur_res_op.op_id not in execution.native_group_launches:
        execution._record_single_native(
            ctx, program, replica, name, transport=transport)
    if is_native_prog:
        return program.start(engine, name, ready_ev, transport=transport)
    run_prog = _wrap_live_startup_program(
        program, execution=execution, replica=replica, scope=scope,
        compact_exec=compact_exec, op=op, prev_res_op=cur_res_op,
        inv_record=inv_rec, live_view=live_view, live_l2_key=live_l2_key,
        prepare_op_context=prepare_op_context, launch_count=inv_launches,
    )
    return run_prog.start(engine, name, ready_ev, owner=owner)
