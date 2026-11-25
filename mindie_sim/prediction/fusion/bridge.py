"""Materialize prefetch leases on the selected operator sequence.

[Role] Derive prefetch hardware limits and bind copy and release recipes.
[Invariants] A settled copy moves bytes from the consumer to the prefetch.
  Its L1 allocation stays live until both operations complete.
[Timeouts & Locks] Finite in-memory graph rewriting; no locks.
[Side-Effect Policy] Returns new operator tuples and private recipe books.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from mindie_sim.prediction.execution.weight_prefetch import PrefetchDecision
from mindie_sim.prediction.fusion.pool import (
    PoolHolder, PrefetchPoolProgram, ReleasePoolProgram, register_recipe, units_for_pool,
)
from mindie_sim.prediction.fusion.ports import (
    full_operand_ports, replace_input_ports, replace_scalar_metadata, rewritten_dependencies,
)
from mindie_sim.prediction.mechanism import OperatorPlan, PlannedRecipe, Work
from mindie_sim.prediction.program import RecipeBook
from mindie_sim.schema.hardware import Device
from mindie_sim.schema.operators import OpSpec


@dataclass(frozen=True)
class HardwareProfile:
    """Per-die HBM bandwidth and aggregate L1 capacity across prefetch units."""

    hbm_bytes_per_ms: float
    prefetch_capacity_bytes: int
    prefetch_units: int

    def __post_init__(self):
        if self.hbm_bytes_per_ms <= 0 or not math.isfinite(self.hbm_bytes_per_ms):
            raise ValueError("hardware_rate_invalid:hbm_bytes_per_ms")
        if self.prefetch_capacity_bytes < 0 or self.prefetch_units < 1:
            raise ValueError("hardware_capacity_invalid")

    @classmethod
    def from_die(cls, die: Device, **overrides) -> HardwareProfile:
        has_l1 = "l1" in dict(die.buffers)
        fields = dict(hbm_bytes_per_ms=die.total_rate("hbm") / 1000,
                      prefetch_capacity_bytes=die.total_buffer_bytes("l1") if has_l1 else 0,
                      prefetch_units=die.pool_units("l1") if has_l1 else 1)
        return cls(**(fields | overrides))


def _canonical_hardware(case, hardware: HardwareProfile | None,
                        options: Mapping[str, Any]) -> HardwareProfile:
    if hardware is not None:
        return hardware
    if not case.machine.dies:
        raise ValueError("hardware_case_machine_missing")
    return HardwareProfile.from_die(case.machine.dies[0], **options.get("hardware", {}))


@dataclass(frozen=True)
class CanonicalPlan:
    ops: tuple[OpSpec, ...]
    book: RecipeBook
    pool_holders: tuple[tuple[str, PoolHolder], ...] = ()


@dataclass(frozen=True)
class PrefetchRewrite:
    ops: tuple[OpSpec, ...]
    recipe: PlannedRecipe


def _prefetch_recipe(decision: PrefetchDecision, holder: PoolHolder,
                     amount: int) -> PlannedRecipe:
    work = Work(resources={"hbm": float(amount)})

    def program(context):
        names = context.resource_names
        return PrefetchPoolProgram(
            holder, float(amount), units_for_pool(context, "l1"),
            caps={names["hbm"]: float(decision.spare_bytes_per_ms)},
            priority=decision.priority, work_resource=names["hbm"],
            pool_name=dict(context.pool_names).get("l1", "l1"))

    def build(op, geometry):
        del op, geometry
        return OperatorPlan(work, program, metadata={"optimizer_prefetch": True})

    return PlannedRecipe(build)


def _release_recipe(holder: PoolHolder) -> PlannedRecipe:
    def program(context):
        # The release must resolve the same per-(replica, stage) physical pool
        # namespace the copy acquired; a bare holder lookup would free a peer
        # replica's grant in a shared DP/PP Engine.
        return ReleasePoolProgram(
            holder, str(dict(context.pool_names).get("l1", "l1")))

    def build(op, geometry):
        del op, geometry
        return OperatorPlan(Work(), program, metadata={"optimizer_pool_release": True})

    return PlannedRecipe(build)


def add_prefetch(plan: CanonicalPlan, decision: PrefetchDecision,
                 hardware: HardwareProfile) -> tuple[CanonicalPlan, str] | None:
    """Add a low-priority prefetch and its consumer-completion pool release."""
    consumer = next((op for op in plan.ops if op.op_id == decision.op_id), None)
    if consumer is None:
        return None
    prefetch_id = f"prefetch_W_{decision.op_id}"
    release_id = f"release_W_{decision.op_id}"
    existing = {op.op_id for op in plan.ops}
    if prefetch_id in existing or release_id in existing:
        return None
    holder = PoolHolder(prefetch_id)
    raw_layout = str(consumer.layout or "contiguous").strip().lower()
    if raw_layout == "fractal_nz":
        prefetch_layout = "nz"
    elif raw_layout in {"contiguous", "nd", "nz", "strided", "view"}:
        prefetch_layout = raw_layout
    else:
        prefetch_layout = "contiguous"
    prefetch = OpSpec(
        op_id=prefetch_id, family="prefetch", phase=consumer.phase,
        stage=consumer.stage, layer=consumer.layer, requests=(),
        semantic_shape=(1,), local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        hbm_bytes=float(decision.target_bytes), matrix_flops=(),
        dtype=consumer.dtype, layout=prefetch_layout, recipe=prefetch_id,
        coverage="analytical_prototype", stream="prefetch",
        metadata=(("prefetch_target", decision.op_id),
                  ("prefetch_priority", decision.priority),
                  ("prefetch_caps", {"hbm": float(decision.spare_bytes_per_ms)}),
                  ("prefetch_pool", "l1"),
                  ("prefetch_units", int(hardware.prefetch_units)),
                  ("prefetch_bytes", int(decision.target_bytes)),
                  ("pointer_stable", True)),
    )
    release = OpSpec(
        op_id=release_id, family="release", phase=consumer.phase,
        stage=consumer.stage, layer=consumer.layer, requests=(),
        semantic_shape=(1,), local_shape=(1,), padded_shape=(1,), launch_shape=(1,),
        # Hold the L1 grant until both the consumer kernel and the producer
        # copy have retired; the consumer alone can outlive a delayed copy.
        deps=(consumer.op_id, prefetch_id), recipe=release_id, coverage="pool_release",
        dtype=consumer.dtype, layout=prefetch_layout, stream=consumer.stream or "main",
        metadata=(("prefetch_release_for", prefetch_id),
                  ("prefetch_target", decision.op_id),
                  ("prefetch_priority", decision.priority),
                  ("prefetch_release_bytes", int(decision.target_bytes)),
                  ("prefetch_release_units", int(hardware.prefetch_units)),
                  ("prefetch_pool", "l1"),
                  ("pointer_stable", True)),
    )
    book = register_recipe(plan.book, prefetch_id,
                           _prefetch_recipe(decision, holder, decision.target_bytes))
    book = register_recipe(book, release_id, _release_recipe(holder))
    index = next(i for i, item in enumerate(plan.ops) if item.op_id == consumer.op_id)
    before = plan.ops[:index]
    after = plan.ops[index + 1:]
    ops = (*before, prefetch, consumer, release, *after)
    updated = replace(plan, ops=ops, book=book,
                      pool_holders=(*plan.pool_holders, (prefetch_id, holder)))
    return updated, prefetch_id


def settle_prefetch_rewrite(plan: CanonicalPlan, decision: PrefetchDecision,
                            prefetch_id: str, completed_bytes: int) -> PrefetchRewrite | None:
    done = max(0, min(int(completed_bytes), int(decision.target_bytes)))
    if done <= 0:
        return None
    prefetch = next((op for op in plan.ops if op.op_id == prefetch_id), None)
    consumer = next((op for op in plan.ops if op.op_id == decision.op_id), None)
    release_id = f"release_W_{decision.op_id}"
    if prefetch is None or consumer is None:
        return None
    if done > int(consumer.hbm_bytes) + 1:
        raise ValueError("prefetch_exceeds_consumer_hbm")
    holder = dict(plan.pool_holders).get(prefetch_id)
    if holder is None:
        raise ValueError("prefetch_pool_holder_missing")
    resource_work = dict(dict(consumer.metadata).get("resource_work", {}))
    resource_work["mte1"] = float(resource_work.get("mte1", 0.0) + done)
    metadata = replace_scalar_metadata(
        consumer.metadata, resource_work=resource_work, prefetch_hit_bytes=float(done))
    # The prefetch id is a scheduling dependency, but appending it to a
    # deduplicated ``deps`` tuple must not regroup or delete the consumer's
    # positional data operands.  Materialize the complete sequence first.
    by_name = {op.op_id: op for op in plan.ops}
    positions = full_operand_ports(consumer, by_name)
    entries = (*positions, (prefetch_id, f"{prefetch_id}:out"))
    scheduled = (*rewritten_dependencies(consumer, positions, {}, by_name),
                 prefetch_id)
    consumer = replace(
        consumer, hbm_bytes=float(consumer.hbm_bytes - done),
        metadata=replace_input_ports(metadata, entries),
        deps=tuple(dict.fromkeys(scheduled)))
    ops = []
    for op in plan.ops:
        if op.op_id == consumer.op_id:
            ops.append(consumer)
        elif op.op_id == prefetch_id:
            ops.append(replace(op, hbm_bytes=float(done),
                               metadata=replace_scalar_metadata(
                                   op.metadata, prefetch_bytes=int(done))))
        elif op.op_id == release_id:
            ops.append(replace(op, metadata=replace_scalar_metadata(
                op.metadata, prefetch_release_bytes=int(done))))
        else:
            ops.append(op)
    return PrefetchRewrite(tuple(ops), _prefetch_recipe(decision, holder, done))


def apply_prefetch_rewrite(plan: CanonicalPlan,
                           rewrite: PrefetchRewrite, prefetch_id: str) -> CanonicalPlan:
    book = register_recipe(plan.book, prefetch_id, rewrite.recipe)
    return replace(plan, ops=rewrite.ops, book=book)
