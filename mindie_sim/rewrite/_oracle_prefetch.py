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
from mindie_sim.prediction.fusion.pool import PrefetchPoolProgram, register_recipe, units_for_pool
from mindie_sim.prediction.fusion.ports import replace_scalar_metadata
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.operators.matrix.matrix_epilogue import MatrixEpilogue
from mindie_sim.prediction.streams import MAIN_STREAM, StreamEvent
from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision
from mindie_sim.schema.common import ModelingError

PREFETCH_PRIORITY = 10
DISABLED_PRIMITIVE_TABLE_TOKEN = "__primitive_table_disabled__"


def _static_weight_bytes(meta: dict, fallback: int) -> int:
    for key in ("packed_weight_bytes", "weight_bytes", "local_weight_bytes", "weight_read_bytes"):
        val = meta.get(key)
        if type(val) in {int, float} and val > 0:
            return int(val)
    return int(fallback)


def _displaced_weight_mte2(op, amount: int) -> tuple[float, int | float | None]:
    """Compute MTE2 bytes moved to prefetch DMA and remaining ``packed_weight_bytes``."""
    mte2 = float(op.mte2_bytes)
    if mte2 <= 0.0 or amount <= 0:
        return 0.0, None
    meta = dict(op.metadata)
    packed = meta.get("packed_weight_bytes")
    if type(packed) in {int, float} and packed > 0:
        ep = MatrixEpilogue.from_op(op)
        headroom_w = max(0.0, float(packed) - float(ep.column_scale_bytes))
        headroom_m = max(0.0, mte2 - float(ep.row_scale_bytes))
        moved = min(float(amount), headroom_w, headroom_m)
        rem = max(float(ep.column_scale_bytes), float(packed) - moved)
        rem_val: int | float = int(round(rem)) if isinstance(packed, int) and rem.is_integer() else rem
        return moved, rem_val
    return min(mte2, float(amount)), None


def _rewrite_prefetch_recipe(
    decision: PrefetchDecision,
    holder,
    amount: int,
    mte2_moved: float,
    pool: str,
    unit_ids: tuple[int, ...],
    units: int,
) -> PlannedRecipe:
    """Emit HBM + MTE2 copy work bound to the lease's physical pool units."""
    base_res = {"hbm": float(amount)}
    if mte2_moved > 0.0:
        base_res["mte2"] = float(mte2_moved)
    work = Work(resources=base_res)

    def program(context):
        names = context.resource_names
        copy_work = {names["hbm"]: float(amount)}
        if mte2_moved > 0.0:
            mte2_key = names.get("mte2_aic") or names.get("mte2")
            if mte2_key:
                copy_work[mte2_key] = float(mte2_moved)
        eff_units = len(unit_ids) if unit_ids else (units if units > 0 else units_for_pool(context, pool))
        return PrefetchPoolProgram(
            holder, float(amount), eff_units,
            caps={names["hbm"]: float(decision.spare_bytes_per_ms)},
            priority=decision.priority, work_resource=copy_work,
            pool_name=dict(context.pool_names).get(pool, pool),
            unit_ids=unit_ids or None,
        )

    def build(op, geometry):
        del op, geometry
        return OperatorPlan(work, program, metadata={"optimizer_prefetch": True})

    return PlannedRecipe(build)


def bind_prefetch_ops(case, replica, ops, book, leases, hardware):
    """Lower prefetch leases onto ``ops`` and register replica-scoped recipes."""
    scoped_leases = tuple(
        lease for lease in leases if getattr(lease, "replica", replica) == replica
    )
    if not scoped_leases:
        return tuple(ops), book
    plan = CanonicalPlan(ops=tuple(ops), book=book)
    for lease in scoped_leases:
        target = next((op for op in plan.ops if op.op_id == lease.target_op), None)
        if target is None:
            raise ModelingError(f"rewrite_prefetch_target_missing:{lease.target_op}")
        if not any(op.op_id == lease.window_anchor_op for op in plan.ops):
            raise ModelingError(f"rewrite_prefetch_anchor_missing:{lease.window_anchor_op}")
        if not getattr(plan.book.select(target), "traffic_rewrite_capable", False):
            raise ModelingError(f"rewrite_prefetch_recipe_unbound:{target.op_id}")
        amount = lease.max_bytes
        weight = _static_weight_bytes(dict(target.metadata), amount)
        stage_hw = hardware(target.stage) if callable(hardware) else hardware
        decision = PrefetchDecision(
            target.op_id, (lease.port_name,), weight, amount, amount, max(0, weight - amount),
            amount, 0, stage_hw.hbm_bytes_per_ms, 0, True, "", (),
            priority=PREFETCH_PRIORITY,
        )
        if (added := add_prefetch(plan, decision, stage_hw)) is None:
            raise ModelingError(f"rewrite_prefetch_duplicate_target:{target.op_id}")
        plan, prefetch_id = added
        if (settled := settle_prefetch_rewrite(plan, decision, prefetch_id, amount)) is None:
            raise ModelingError(f"rewrite_prefetch_settlement_failed:{target.op_id}")
        plan = apply_prefetch_rewrite(plan, settled, prefetch_id)
        consumer = next((op for op in plan.ops if op.op_id == target.op_id), None)
        mte2_moved, rem_packed = _displaced_weight_mte2(consumer, amount) if consumer is not None else (0.0, None)
        prefetch_op = next((op for op in plan.ops if op.op_id == prefetch_id), None)
        holder = dict(plan.pool_holders).get(prefetch_id)
        if prefetch_op is not None and holder is not None:
            unit_ids = tuple(int(u) for u in (getattr(lease, "physical_unit_ids", ()) or ()))
            units = int(getattr(lease, "physical_units", 0) or 0)
            dma_recipe = _rewrite_prefetch_recipe(
                decision, holder, amount, mte2_moved, lease.pool, unit_ids, units,
            )
            plan = replace(plan, book=register_recipe(plan.book, prefetch_op.recipe, dma_recipe))
        new_ops = []
        for op in plan.ops:
            if op.op_id == prefetch_id:
                new_ops.append(replace(
                    op, deps=(lease.window_anchor_op,), stream=target.stream,
                    hbm_bytes=float(amount), mte2_bytes=mte2_moved,
                ))
            elif op.op_id == target.op_id and mte2_moved > 0.0:
                meta_out = (
                    replace_scalar_metadata(op.metadata, packed_weight_bytes=rem_packed)
                    if rem_packed is not None else op.metadata
                )
                new_ops.append(replace(
                    op, mte2_bytes=max(0.0, float(op.mte2_bytes) - mte2_moved),
                    metadata=meta_out,
                ))
            elif op.family == "release" and target.op_id in op.deps:
                new_ops.append(replace(op, stream=target.stream))
            else:
                new_ops.append(op)
        plan = replace(plan, ops=tuple(new_ops))
    generated = {op.recipe for op in plan.ops if op.family in {"prefetch", "release"}}
    book = plan.book
    for name in generated:
        book = register_recipe(book, f"{name}:r{replica}", plan.book.recipes[name])
    return tuple(
        replace(op, recipe=f"{op.recipe}:r{replica}") if op.recipe in generated else op
        for op in plan.ops
    ), book


def bind_prefetch_streams(graphs, streams, leases):
    """Attach generated prefetch/release ops and cross-stream anchor events scoped by ``(replica, op_id)``."""
    if not streams or not leases:
        return tuple(streams)
    lease_by_site = {
        (getattr(lease, "replica", 0), lease.target_op): lease for lease in leases
    }
    updated = []
    for plan in streams:
        replica = plan.rank
        ops = graphs.get(replica, ())
        if not ops:
            updated.append(plan)
            continue
        replica_targets = {
            target_op: lease
            for (lease_rep, target_op), lease in lease_by_site.items()
            if lease_rep == replica
        }
        if not replica_targets:
            updated.append(plan)
            continue
        assign = dict(plan.assignments)
        events = list(plan.events)
        existing_edges = {(ev.record_after, wait) for ev in events for wait in ev.wait_before}
        for op in ops:
            if op.family == "prefetch":
                target = next(
                    (t for t in replica_targets if any(o.op_id == t and op.op_id in o.deps for o in ops)),
                    None,
                )
                if target is None:
                    continue
                target_stream = assign.get(target, MAIN_STREAM)
                assign[op.op_id] = target_stream
                anchor = replica_targets[target].window_anchor_op
                anchor_stream = assign.get(anchor, MAIN_STREAM)
                if anchor_stream != target_stream and (anchor, op.op_id) not in existing_edges:
                    events.append(StreamEvent(
                        event_id=f"prefetch:r{replica}:{anchor}->{op.op_id}",
                        record_stream=anchor_stream,
                        record_after=anchor,
                        wait_stream=target_stream,
                        wait_before=(op.op_id,),
                    ))
                    existing_edges.add((anchor, op.op_id))
            elif op.family == "release":
                target = next((dep for dep in op.deps if dep in replica_targets), None)
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
