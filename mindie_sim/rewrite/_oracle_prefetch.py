"""Prefetch binding and compatible multi-stream event wiring for rewrite replay.

[Role] Materialize declared ``WeightPrefetch`` leases into causal prefetch and
  release operators while aligning synthetic ops with active ``StreamPlan``
  assignments and explicit cross-stream ``StreamEvent`` edges.
[Invariants] Every prefetch op depends causally on its window anchor and runs on
  its target operator's stream; cross-stream anchors receive explicit wait
  events so prefetch and multi-stream schedules remain sound and race-free.
[Timeouts & Locks] Pure bounded graph rewriting; no I/O, threads or locks.
[Side-Effect Policy] Returns new operator tuples, recipe books and stream plans.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.prediction.fusion.bridge import (
    CanonicalPlan, add_prefetch, apply_prefetch_rewrite, settle_prefetch_rewrite,
)
from mindie_sim.prediction.fusion.pool import register_recipe
from mindie_sim.prediction.streams import MAIN_STREAM, StreamEvent
from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision
from mindie_sim.schema.common import ModelingError

PREFETCH_PRIORITY = 10
DISABLED_PRIMITIVE_TABLE_TOKEN = "__primitive_table_disabled__"


def bind_prefetch_ops(case, replica, ops, book, leases, hardware):
    """Lower prefetch leases onto ``ops`` and register replica-scoped recipes."""
    plan = CanonicalPlan(case, ops, None, hardware, (), book)
    for lease in leases:
        target = next((op for op in plan.ops if op.op_id == lease.target_op), None)
        if target is None:
            continue
        if not getattr(plan.book.select(target), "traffic_rewrite_capable", False):
            raise ModelingError(f"rewrite_prefetch_recipe_unbound:{target.op_id}")
        amount = lease.max_bytes
        weight = int(dict(target.metadata).get("weight_bytes", amount))
        decision = PrefetchDecision(
            target.op_id, (lease.port_name,), weight, amount, amount, weight - amount,
            amount, 0, hardware.hbm_bytes_per_ms, 0, True, "", (),
            priority=PREFETCH_PRIORITY,
        )
        if (added := add_prefetch(plan, decision, hardware)) is None:
            raise ModelingError(f"rewrite_prefetch_duplicate_target:{target.op_id}")
        plan, prefetch_id = added
        if (settled := settle_prefetch_rewrite(plan, decision, prefetch_id, amount, hardware)) is None:
            raise ModelingError(f"rewrite_prefetch_settlement_failed:{target.op_id}")
        plan = apply_prefetch_rewrite(plan, settled, prefetch_id)
        plan = replace(plan, ops=tuple(
            replace(op, deps=(lease.window_anchor_op,), stream=target.stream)
            if op.op_id == prefetch_id else (
                replace(op, stream=target.stream)
                if op.family == "release" and target.op_id in op.deps else op
            )
            for op in plan.ops
        ))
    generated = {op.recipe for op in plan.ops if op.family in {"prefetch", "release"}}
    book = plan.book
    for name in generated:
        book = register_recipe(book, f"{name}:r{replica}", plan.book.recipes[name])
    return tuple(
        replace(op, recipe=f"{op.recipe}:r{replica}") if op.recipe in generated else op
        for op in plan.ops
    ), book


def bind_prefetch_streams(graphs, streams, leases):
    """Attach generated prefetch/release ops and cross-stream anchor events to ``streams``."""
    if not streams or not leases:
        return tuple(streams)
    lease_by_target = {lease.target_op: lease for lease in leases}
    updated = []
    for plan in streams:
        ops = graphs.get(plan.rank, ())
        if not ops:
            updated.append(plan)
            continue
        assign = dict(plan.assignments)
        events = list(plan.events)
        existing_edges = {(ev.record_after, wait) for ev in events for wait in ev.wait_before}
        for op in ops:
            if op.family == "prefetch":
                target = next((t for t in lease_by_target if any(o.op_id == t and op.op_id in o.deps for o in ops)), None)
                if target is None:
                    continue
                target_stream = assign.get(target, MAIN_STREAM)
                assign[op.op_id] = target_stream
                anchor = lease_by_target[target].window_anchor_op
                anchor_stream = assign.get(anchor, MAIN_STREAM)
                if anchor_stream != target_stream and (anchor, op.op_id) not in existing_edges:
                    events.append(StreamEvent(
                        event_id=f"prefetch:{anchor}->{op.op_id}",
                        record_stream=anchor_stream,
                        record_after=anchor,
                        wait_stream=target_stream,
                        wait_before=(op.op_id,),
                    ))
                    existing_edges.add((anchor, op.op_id))
            elif op.family == "release":
                target = next((dep for dep in op.deps if dep in lease_by_target), None)
                if target is not None:
                    assign[op.op_id] = assign.get(target, MAIN_STREAM)
        updated.append(replace(
            plan,
            assignments=tuple(assign.items()),
            events=tuple(events),
        ))
    return tuple(updated)


def die_placements_for_case(case) -> tuple[tuple[int, tuple[int, int]], ...]:
    """Return canonical ``(die_rank, (replica, pp_stage))`` tuples for a case."""
    parallel = case.runtime.parallel
    width = int(parallel.tp) * int(parallel.pcp)
    pp = int(parallel.pp)
    limit = int(parallel.dp) * pp * width
    return tuple(
        (int(die.rank), (idx // (pp * width), (idx // width) % pp))
        for idx, die in enumerate(case.machine.dies)
        if idx < limit
    )


def primitive_table_digest_token(primitive_table):
    """Return a deterministic content digest token for ``primitive_table``."""
    if primitive_table is False:
        return DISABLED_PRIMITIVE_TABLE_TOKEN
    if primitive_table is None:
        return None
    canonical = getattr(primitive_table, "canonical_envelope_sha256", None)
    if callable(canonical):
        return canonical()
    return getattr(primitive_table, "envelope_sha256", None) or getattr(primitive_table, "entries", None)
