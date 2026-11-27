"""Public dispatcher and registration for governed paged MLA.

[Role] Resolve a paged MLA contract against the declared ISA and ABI.
[Invariants] Each architecture keeps its own source-derived pipeline strategy.
[Timeouts & Locks] Constant-time architecture selection. No locks.
[Side-Effect Policy] Return plans and registry records without global mutation.
"""
from __future__ import annotations

from mindie_sim.schema.operators import OpSpec

from ...mechanism import Geometry, OperatorPlan
from ...results import OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from ..attention.common import with_sequence_context
from . import arch22, arch35
from ._lowering import describe
from .common import (
    DSL_FLASH_ABI, DSL_MQS_ABI, FIA_ABI, FLASH_ABI, MQS_ABI,
    FAMILY, PagedMlaContract, capability_gap, latent_record_bytes,
    make_op, mqsmla_payload_bytes, validate_contract,
)

_STRATEGIES = {
    arch22.ARCH_ID: arch22.STRATEGY,
    arch35.ARCH_ID: arch35.STRATEGY,
}


def build(op: OpSpec, geometry_or_ctx: Geometry | OperatorPlanningContext, /) -> OperatorPlan:
    """Build a plan from either the governed context or explicit geometry."""
    if isinstance(geometry_or_ctx, OperatorPlanningContext):
        ctx = geometry_or_ctx
    else:
        geo = geometry_or_ctx
        if geo.datapath is None:
            raise capability_gap(
                "paged_mla_datapath_required", dict(op.metadata).get("requested_abi", ""),
                str(getattr(geo.compute_arch, "preset", "")),
            )
        ctx = OperatorPlanningContext(
            OpOccurrence(geo.replica, op.stage, geo.rank, op.op_id), geo, geo.datapath,
            requested_abi=dict(op.metadata).get("requested_abi"), stream_owners=geo.stream_owners,
        )
    ctx = with_sequence_context(op, ctx)
    arch_id = ctx.compute_arch.arch_id
    strategy = _STRATEGIES.get(arch_id)
    if strategy is None:
        raise capability_gap(
            "paged_mla_unsupported_arch", ctx.requested_abi or dict(op.metadata).get("requested_abi", ""),
            arch_id,
        )
    return strategy.build(op, ctx)


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    plan = build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(ctx.occurrence, op, plan, descriptor.implementation,
                            descriptor.sources, descriptor)


def register_family() -> OperatorFamilySpec:
    """Return the paged MLA registration record for a caller-owned registry."""
    return OperatorFamilySpec(
        name=FAMILY, families=frozenset({FAMILY}), owned_recipe_ids=frozenset({FAMILY}),
        resolve=resolve_operator_plan,
    )


__all__ = [
    "DSL_FLASH_ABI",
    "DSL_MQS_ABI",
    "FIA_ABI",
    "FLASH_ABI",
    "MQS_ABI",
    "PagedMlaContract",
    "build",
    "latent_record_bytes",
    "make_op",
    "mqsmla_payload_bytes",
    "register_family",
    "resolve_operator_plan",
    "validate_contract",
]
