"""Single public entrypoint and dispatcher for the governed ``rotary`` family.

[Role] Register the ``rotary`` operator family in ``FamilyRegistry`` and
  resolve rotary ``OpSpec`` occurrences against explicit ``DatapathCapabilitySpec``
  and ABI declarations.
[Invariants] Unified rotate-half formulation shared across ``npu_arch_2201`` and
  ``npu_arch_3510``; validates declared ``gm->ub`` and ``ub->gm`` routes when
  datapath paths are present and rejects unsupported explicit ABI requests.
[Timeouts & Locks] O(1) capability checks and plan resolution; no I/O or locks.
[Side-Effect Policy] Pure immutable ``OperatorFamilySpec`` and ``ResolvedOperator``.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.operators import OpSpec
from ..contracts import (
    OperatorFamilySpec,
    OperatorPlanningContext,
    ResolvedOperator,
)
from .common import IMPLEMENTATION, SOURCES, SUPPORTED_ABIS, build, describe


def resolve_operator_plan(
    op: OpSpec,
    ctx: OperatorPlanningContext,
    /,
) -> ResolvedOperator:
    """Validate ABI and datapath prerequisites, then build the rotary plan."""
    if ctx.requested_abi is not None and ctx.requested_abi not in SUPPORTED_ABIS:
        raise ModelingError(f"rotary_abi_unsupported:{ctx.requested_abi}")
    if ctx.datapath.usable_paths:
        if not ctx.datapath.has_path("gm", "ub") or not ctx.datapath.has_path("ub", "gm"):
            raise ModelingError(
                f"rotary_datapath_missing_gm_ub:{ctx.compute_arch.arch_id}"
            )
    plan = build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(
        occurrence=ctx.occurrence,
        op=op,
        recipe=plan,
        implementation=IMPLEMENTATION,
        sources=SOURCES,
        descriptor=descriptor,
    )


def register_family() -> OperatorFamilySpec:
    """Return the single-owner ``OperatorFamilySpec`` for ``rotary``."""
    return OperatorFamilySpec(
        name="rotary",
        families=frozenset({"rotary"}),
        owned_recipe_ids=frozenset({"rotary", "rotary_position_embedding"}),
        resolve=resolve_operator_plan,
        traffic_rewrite_capable=True,
    )


__all__ = [
    "register_family",
    "resolve_operator_plan",
]
