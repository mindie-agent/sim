"""Execution coalescing, stage folding, and floor-parallel program execution.

[Role] Inline pure wrapper stages, fold zero-work acquire/free wrappers and
  strictly serial integer compute chains, execute ``_FloorParallelProgram``,
  and expose ``is_pure_static_program`` and ``coalesce_execution_program``.
[Invariants] Folding conserves every resource total, buffer grant/release,
  and completion instant; uncoupled stages mixing placed and unplaced work
  never remain coalesced into distorted fair shares.
[Timeouts & Locks] Materialization windows are bounded by
  ``PIPELINE_COHORT_WINDOW`` and ``unroll_limit``; no I/O or locks.
[Side-Effect Policy] Pure program transformation; ``_FloorParallelProgram``
  submits tasks only to the caller's ``Engine``.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..engine import Allocation, Engine, Event
from .lowering import coalesce_programs
from .pipeline import PIPELINE_COHORT_WINDOW, PipelineProgram
from .stages import (
    PIPELINE_UNROLL_LIMIT,
    KernelProgram,
    ParallelProgram,
    Program,
    Stage,
    _COMPILED_CACHE_LIMIT,
    _compile_leaf_kernel_ops,
    _declares_no_work,
    _is_pure_root_delay,
    _run_compiled_ops,
)

_EXACT_INT_SUM_LIMIT = 4e15


def _is_empty_queued_store(program) -> bool:
    return (
        type(program).__name__ == "QueuedStore"
        and int(getattr(program, "outstanding", 0)) >= 1
        and int(getattr(program, "granule_bytes", 0)) >= 1
        and int(getattr(program, "rows", 1)) * int(getattr(program, "cols", 1)) == 0
    )


def _inline_wrapper_stages(stages, *, protected_names=()):
    sink_alias: dict[str, str] = {}
    out: list[Stage] = []
    for stage in stages:
        if sink_alias:
            after = tuple(sink_alias.get(d, d) for d in stage.after)
            prev = tuple(sink_alias.get(d, d) for d in stage.previous)
            if after != stage.after or prev != stage.previous:
                stage = replace(stage, after=after, previous=prev)
        if (
            stage.name not in protected_names
            and isinstance(stage.program, KernelProgram)
            and stage.program.minimum_ms == 0.0
            and not stage.work and not stage.caps and not stage.acquire and not stage.free
            and stage.latency <= 0.0 and not stage.placement
            and not stage.previous and not stage.keep_empty and stage.fragment is None
        ):
            inner = stage.program.stages
            if inner and not any(
                s.acquire or s.free or s.previous or s.keep_empty or s.fragment is not None
                for s in inner
            ):
                consumed = {d for s in inner for d in s.after}
                sinks = [s.name for s in inner if s.name not in consumed]
                if len(sinks) == 1:
                    prefix = stage.name
                    for inner_stage in inner:
                        s_after = (
                            tuple(f"{prefix}:{d}" for d in inner_stage.after)
                            if inner_stage.after else stage.after
                        )
                        out.append(replace(
                            inner_stage, name=f"{prefix}:{inner_stage.name}", after=s_after
                        ))
                    sink_alias[prefix] = f"{prefix}:{sinks[0]}"
                    continue
        out.append(stage)
    return tuple(out)


@dataclass(frozen=True)
class _FloorParallelProgram:
    """Parallel cohort program with a single floor-released completion join."""

    programs: tuple[Program, ...]
    minimum_ms: float
    single_body: bool = False

    def start(self, engine: Engine, name: str, ready: Event, *, owner: str = "") -> Event:
        if ready.time is None:
            from ..invocation import InvocationProgram, InvocationTiming
            return InvocationProgram(
                self.programs[0] if self.single_body else ParallelProgram(self.programs),
                InvocationTiming(minimum_ms=self.minimum_ms),
            ).start(engine, name, ready, owner=owner)
        comp_cache = getattr(engine, "_compiled_programs", None)
        if comp_cache is not None:
            hit = comp_cache.get(id(self))
            if hit is not None and hit[0] is self:
                plan = hit[1]
            else:
                all_ops, branch_sinks, next_slot, ok, any_root_delay = [], [], 1, True, False
                for index, program in enumerate(self.programs):
                    if not isinstance(program, KernelProgram):
                        ok = False
                        break
                    g_pref = ":body" if self.single_body else f":group{index}"
                    compiled = _compile_leaf_kernel_ops(engine, program, g_pref, next_slot)
                    if compiled is None:
                        ok = False
                        break
                    b_ops, b_sink, next_slot, _had, hrd = compiled
                    if hrd:
                        any_root_delay = True
                    all_ops.extend(b_ops)
                    branch_sinks.append(b_sink)
                plan = (tuple(all_ops), tuple(branch_sinks), any_root_delay) if ok else False
                if len(comp_cache) >= _COMPILED_CACHE_LIMIT:
                    comp_cache.clear()
                comp_cache[id(self)] = (self, plan)
            if plan is not False:
                plan_ops, sink_slots, any_root_delay = plan
                if not any_root_delay or (
                    (not engine._trace_maxlen or getattr(engine, "_compact_step", False))
                    and not engine._periodic_runs and not engine.occurrence_metrics
                ):
                    ev = _run_compiled_ops(engine, name, ready, owner, plan_ops) if plan_ops else [ready]
                    n_sinks = len(sink_slots)
                    if n_sinks == 1:
                        after_events = ev[sink_slots[0]]
                    elif n_sinks == 2:
                        after_events = (ev[sink_slots[0]], ev[sink_slots[1]])
                    elif n_sinks:
                        after_events = tuple(ev[s] for s in sink_slots)
                    else:
                        after_events = ready
                    return engine._submit_prepared(
                        engine._empty_task, name + ":complete",
                        after=after_events, release=ready.time + self.minimum_ms, owner=owner,
                    )
        if self.single_body:
            body_ev = self.programs[0].start(engine, f"{name}:body", ready, owner=owner)
            return engine.submit(
                name + ":complete", after=(body_ev,),
                release=ready.time + self.minimum_ms, owner=owner,
            )
        events = tuple(
            program.start(engine, f"{name}:group{index}", ready, owner=owner)
            for index, program in enumerate(self.programs)
        )
        return engine.submit(
            name + ":complete", after=events or (ready,),
            release=ready.time + self.minimum_ms, owner=owner,
        )


def _is_fair_compute_resource(name: str) -> bool:
    return (
        name in {"vector", "aic", "cube"}
        or name.startswith(("vector@", "aic@", "cube@"))
        or ":vector@" in name or ":aic@" in name or ":cube@" in name
    )


def _fold_kernel_stages(stages: tuple[Stage, ...]) -> tuple[Stage, ...]:
    """Fold zero-work acquire/free wrappers and strictly serial integer compute chains."""
    if len(stages) < 2 or any(s.previous or s.keep_empty or s.fragment is not None for s in stages):
        return stages
    cur = list(stages)
    while len(cur) >= 2:
        consumers: dict[str, list[int]] = {s.name: [] for s in cur}
        by_name: dict[str, int] = {}
        for idx, s in enumerate(cur):
            if s.name in by_name:
                return tuple(cur)
            by_name[s.name] = idx
            for dep in s.after:
                if dep in consumers:
                    consumers[dep].append(idx)
        changed = False
        for idx, s in enumerate(cur):
            empty_ctrl = (
                s.program is None and not any(s.work.values()) and not any(s.caps.values())
                and s.latency <= 0.0 and not s.coupled and s.priority == 0
                and not s.metadata and not s.deliveries
            )
            if empty_ctrl and s.acquire and not s.free and len(consumers[s.name]) == 1:
                succ_idx = consumers[s.name][0]
                succ = cur[succ_idx]
                if succ.program is None and succ.after == (s.name,) and not succ.acquire:
                    cur[succ_idx] = replace(succ, after=s.after, acquire=s.acquire)
                    del cur[idx]
                    changed = True
                    break
            if (
                empty_ctrl and s.free and not s.acquire and not consumers[s.name]
                and len(s.after) == 1 and s.after[0] in by_name
            ):
                pred_idx = by_name[s.after[0]]
                pred = cur[pred_idx]
                if (
                    pred.program is None and consumers[pred.name] == [idx]
                    and not any(x.name in {y.name for y in pred.free} for x in s.free)
                ):
                    cur[pred_idx] = replace(pred, free=pred.free + s.free)
                    del cur[idx]
                    changed = True
                    break
            if (
                s.program is None and not s.coupled and s.latency <= 0.0
                and not s.acquire and not s.free and len(s.work) == 1
                and len(consumers[s.name]) == 1
            ):
                succ_idx = consumers[s.name][0]
                succ = cur[succ_idx]
                if (
                    succ.program is None and not succ.coupled and succ.latency <= 0.0
                    and not succ.acquire and not succ.free and succ.after == (s.name,)
                    and len(succ.work) == 1 and s.work.keys() == succ.work.keys()
                    and s.caps == succ.caps and s.placement == succ.placement
                    and s.priority == succ.priority and s.work_units == succ.work_units
                    and s.metadata == succ.metadata
                    and not s.deliveries and not succ.deliveries
                    and not s.latency_from_first_service and not succ.latency_from_first_service
                ):
                    rname = next(iter(s.work))
                    w1, w2 = s.work[rname], succ.work[rname]
                    if (
                        _is_fair_compute_resource(rname)
                        and w1 == int(w1) and w2 == int(w2)
                        and abs(w1) + abs(w2) < _EXACT_INT_SUM_LIMIT
                    ):
                        merged_name = s.name
                        cur[idx] = replace(s, work={rname: w1 + w2})
                        for c_idx in consumers[succ.name]:
                            c_stage = cur[c_idx]
                            cur[c_idx] = replace(
                                c_stage,
                                after=tuple(merged_name if d == succ.name else d for d in c_stage.after),
                            )
                        del cur[succ_idx]
                        changed = True
                        break
        if not changed:
            break
    return tuple(cur)


def is_pure_static_program(program) -> bool:
    """True for a callback-free finite DAG of plain kernel stages."""
    if isinstance(program, KernelProgram):
        return not any(
            stage.program is not None or stage.acquire or stage.free
            for stage in program.stages
        )
    if isinstance(program, (ParallelProgram, _FloorParallelProgram)):
        return bool(program.programs) and all(
            is_pure_static_program(child) for child in program.programs
        )
    return False


def coalesce_execution_program(program, *, unroll_limit: int = PIPELINE_UNROLL_LIMIT):
    """Unwrap single-tile pipelines and coalesce symmetric cohorts for Execution."""
    if type(program).__name__ == "_FloorJoinProgram":
        body = coalesce_execution_program(program.body, unroll_limit=unroll_limit)
        eff_min = (
            program.effective_minimum()
            if hasattr(program, "effective_minimum") else program.minimum_ms
        )
        if eff_min <= 0.0:
            return body
        if isinstance(body, ParallelProgram):
            if len(body.programs) == 1:
                body = body.programs[0]
            elif len(body.programs) > 1:
                return _FloorParallelProgram(body.programs, eff_min)
        if isinstance(body, KernelProgram) and body.minimum_ms == 0.0:
            real_stages = [
                s for s in body.stages
                if not (s.program is None and _declares_no_work(s) and len(s.after) <= 1)
            ]
            if len(real_stages) == 1:
                s0 = real_stages[0]
                if (
                    not s0.acquire and not s0.free and s0.program is None
                    and s0.latency == 0.0 and not s0.previous
                    and not s0.keep_empty and s0.fragment is None
                ):
                    return KernelProgram((replace(s0, after=(), latency=eff_min),))
            elif len(real_stages) == 2 and len(body.stages) == 2:
                s0, s1 = real_stages
                if (
                    _is_pure_root_delay(s0, {s0.name}, body.stages)
                    and s0.latency >= eff_min and s1.after == (s0.name,)
                    and s1.program is None and s1.fragment is None
                ):
                    return body
            return _FloorParallelProgram((body,), eff_min, single_body=True)
        return program if body is program.body else replace(program, body=body)

    if type(program).__name__ == "NativeProgram":
        inner = coalesce_execution_program(program.program, unroll_limit=unroll_limit)
        return program if inner is program.program else replace(program, program=inner)

    def unwrap_stage(stage: Stage) -> Stage:
        if stage.program is None:
            return stage
        if (
            _is_empty_queued_store(stage.program) and not stage.work and not stage.caps
            and not stage.acquire and not stage.free and stage.latency <= 0.0
            and not stage.placement and stage.fragment is None
        ):
            return replace(stage, program=None)
        nested = coalesce_execution_program(stage.program, unroll_limit=unroll_limit)
        return stage if nested is stage.program else replace(stage, program=nested)

    if isinstance(program, KernelProgram):
        unwrapped = tuple(unwrap_stage(stage) for stage in program.stages)
        inlined = _fold_kernel_stages(_inline_wrapper_stages(unwrapped))
        if program.minimum_ms == 0.0 and len(inlined) == 1:
            s0 = inlined[0]
            if (
                s0.program is not None and not s0.work and not s0.caps
                and not s0.acquire and not s0.free and s0.latency <= 0.0
                and not s0.placement and not s0.after and not s0.previous
                and not s0.keep_empty and s0.fragment is None
            ):
                return s0.program
        return program if inlined == program.stages else KernelProgram(inlined, program.minimum_ms)

    if isinstance(program, PipelineProgram):
        if (
            program.homogeneous or program.setup_ms or program.minimum_ms
            or program.producer_bindings is not None
        ):
            return program
        if program.count == 1:
            raw_stages = tuple(program.tile(0))
            seen: set[str] = set()
            flat: list[Stage] = []
            valid = True
            for stage in raw_stages:
                if stage.name in seen:
                    valid = False
                    break
                seen.add(stage.name)
                flat.append(unwrap_stage(replace(stage, previous=())))
            if valid:
                return KernelProgram(_fold_kernel_stages(_inline_wrapper_stages(flat)))
        if 1 < program.count <= PIPELINE_COHORT_WINDOW:
            raw_tiles = [tuple(program.tile(i)) for i in range(program.count)]
            if (
                program.slots == 1
                and program.count * len(raw_tiles[0]) <= unroll_limit
                and all(len({s.name for s in tile}) == len(tile) for tile in raw_tiles)
            ):
                unrolled: list[Stage] = []
                prev_sinks: tuple[str, ...] = ()
                for t_idx, tile in enumerate(raw_tiles):
                    t_stages = _fold_kernel_stages(_inline_wrapper_stages(
                        tuple(unwrap_stage(s) for s in tile)
                    ))
                    if not t_stages:
                        continue
                    local_both = (
                        {a.name for s in t_stages for a in s.acquire}
                        & {a.name for s in t_stages for a in s.free}
                    )
                    pfx = f"t{t_idx}:"
                    consumed_t = {d for s in t_stages for d in s.after}
                    for s in t_stages:
                        new_after = tuple(pfx + d for d in s.after) if s.after else prev_sinks
                        new_acq = (
                            tuple(
                                Allocation(pfx + a.name, a.pool, a.size, a.units)
                                if a.name in local_both else a
                                for a in s.acquire
                            )
                            if local_both and s.acquire else s.acquire
                        )
                        new_free = (
                            tuple(
                                Allocation(pfx + a.name, a.pool, a.size, a.units)
                                if a.name in local_both else a
                                for a in s.free
                            )
                            if local_both and s.free else s.free
                        )
                        unrolled.append(replace(
                            s, name=pfx + s.name, after=new_after, previous=(),
                            acquire=new_acq, free=new_free,
                        ))
                    prev_sinks = tuple(
                        pfx + s.name for s in t_stages if s.name not in consumed_t
                    ) or prev_sinks
                return KernelProgram(_fold_kernel_stages(tuple(unrolled)))
            protected = {p for tile in raw_tiles for s in tile for p in s.previous}
            protected.update(s.name for tile in raw_tiles for s in tile)
            allow_inline = (
                {p for tile in raw_tiles for s in tile for p in s.previous}
                if program.slots == 1 else protected
            )
            new_tiles = tuple(
                _fold_kernel_stages(_inline_wrapper_stages(
                    tuple(unwrap_stage(s) for s in tile), protected_names=allow_inline
                ))
                for tile in raw_tiles
            )
            return PipelineProgram(
                program.count, program.slots, lambda idx, tiles=new_tiles: tiles[idx],
                program.minimum_ms, program.setup_ms,
                producer_bindings=program.producer_bindings,
            )
        return program

    if isinstance(program, ParallelProgram):
        children = tuple(
            coalesce_execution_program(child, unroll_limit=unroll_limit)
            for child in program.programs
        )
        merged = coalesce_programs(children, widen_pipelines=True)
        reverted_unplaced = (
            len(merged) < len(children) and any(
                isinstance(m, KernelProgram)
                and any(
                    (not s.coupled) and bool(s.placement) and bool(set(s.work) - set(s.placement))
                    for s in m.stages
                )
                for m in merged
            )
        )
        if reverted_unplaced:
            merged = children
        if (
            len(merged) > 1
            and sum(len(m.stages) for m in merged if isinstance(m, KernelProgram)) > unroll_limit
        ):
            child_limit = unroll_limit // max(1, len(program.programs))
            children = tuple(
                coalesce_execution_program(child, unroll_limit=child_limit)
                for child in program.programs
            )
            merged = children if reverted_unplaced else coalesce_programs(children, widen_pipelines=True)
        if (
            len(merged) == 1 and isinstance(merged[0], KernelProgram)
            and not any(s.acquire or s.free for s in merged[0].stages)
        ):
            return merged[0]
        return program if merged == program.programs else ParallelProgram(merged)

    return program
