"""Public entrypoint for the governed hyper-connections operator family.

[Role] Register recipe ownership and resolve the physical ISA strategy.
[Invariants] Dispatch uses explicit datapath identity and preserves the caller's
  occurrence, tensor contract, and verified source provenance.
[Timeouts & Locks] Pure bounded lookup and planning. No locks.
[Side-Effect Policy] Import and planning do not register global state or use
  device runtimes.
"""
from __future__ import annotations

from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...results import FusedPipelineDescriptor, OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from .arch22 import STRATEGY as MEMBASE_STRATEGY
from .arch35 import STRATEGY as REGBASE_STRATEGY
from .common import (
    FAMILY,
    OWNED_RECIPES,
    AttnResContract,
    EngramGateContract,
    MhcContract,
    TextGatedGuardDecision,
    TextGatedResidualContract,
    attnres_max_resident_rows,
    evaluate_text_gated_guard,
    gap,
    sinkhorn_vector_ops,
)

_STRATEGIES = {
    strategy.arch_id: strategy for strategy in (MEMBASE_STRATEGY, REGBASE_STRATEGY)
}


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext) -> ResolvedOperator:
    """Resolve one occurrence against an explicitly declared device datapath."""
    arch_id = ctx.compute_arch.arch_id
    strategy = _STRATEGIES.get(arch_id)
    if strategy is None:
        raise gap("hyper_connections_arch_unsupported", str(arch_id))
    plan = strategy.build(op, ctx)
    spec = plan.pipeline_spec
    descriptor = FusedPipelineDescriptor(
        owner=ctx.occurrence,
        family=spec.family,
        implementation=spec.implementation,
        member_op_ids=(op.op_id,),
        absorbed_source_op_ids=(),
        phases=spec.phases,
        sources=spec.sources,
        repeats=spec.repeats,
    )
    return ResolvedOperator(ctx.occurrence, op, plan, spec.implementation, spec.sources, descriptor)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build a plan through the same resolver used by the family registry."""
    if geometry.datapath is None:
        raise gap("hyper_connections_requires_datapath", op.op_id)
    context = OperatorPlanningContext(
        OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry,
        geometry.datapath,
        requested_abi=dict(op.metadata).get("requested_abi"),
        stream_owners=geometry.stream_owners,
    )
    return resolve_operator_plan(op, context).recipe


def register_family() -> OperatorFamilySpec:
    """Return the single ownership declaration for these operator recipes."""
    return OperatorFamilySpec(
        name=FAMILY,
        families=frozenset({"mhc", "attention_residual", "engram_gate", FAMILY}),
        owned_recipe_ids=OWNED_RECIPES,
        resolve=resolve_operator_plan,
        delegated_legacy_recipe_ids=frozenset({
            "hc_inverse_rms",
            "hc_normalize_scale_bias_gate",
            "hc_sinkhorn",
            "hc_pre_mix",
            "hc_post_mix",
        }),
    )


__all__ = [
    "AttnResContract",
    "EngramGateContract",
    "MhcContract",
    "TextGatedGuardDecision",
    "TextGatedResidualContract",
    "attnres_max_resident_rows",
    "build",
    "evaluate_text_gated_guard",
    "register_family",
    "resolve_operator_plan",
    "sinkhorn_vector_ops",
]
