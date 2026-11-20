"""Public entrypoint for the governed linear-attention family.

[Role] Resolve tensor and ABI bindings against a declared ISA and datapath.
[Invariants] Each ISA has one strategy. Generic convolution and RecoverSSM
  share source-independent contracts and keep explicit ABI requests intact.
[Timeouts & Locks] Bounded in-memory planning on the selected strategy.
[Side-Effect Policy] Imports and planning perform no device or external I/O.
"""
from __future__ import annotations

from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...results import OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from .arch22 import STRATEGY as _ARCH22
from .arch35 import STRATEGY as _ARCH35
from .common import (
    ABI_OPERATIONS,
    OPERATIONS,
    RECOVER_SSM_FAMILIES,
    CausalConvContract,
    LinearAttentionContract,
    RankOneTransition,
    RecoverSsmCommitPlan,
    RecoverSsmVerifyPlan,
    StateResidencyReport,
    WyChunkWork,
    analyze_state_residency,
    build_causal_conv1d_plan,
    build_recover_ssm_commit,
    build_recover_ssm_plan,
    build_recover_ssm_verify,
    compute_wy_chunk_work,
    describe,
    missing,
    rank_one_transition,
    select_flash_kda_template,
    validate_contract,
)

_STRATEGIES = {_ARCH22.arch_id: _ARCH22, _ARCH35.arch_id: _ARCH35}


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext) -> ResolvedOperator:
    """Bind a governed operation to its executable plan and source descriptor."""
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise missing("linear_attention_arch_unsupported", f"Unsupported ISA {ctx.compute_arch.arch_id}")
    contract = validate_contract(op, ctx)
    abi = ctx.requested_abi or dict(op.metadata).get("abi") or op.recipe or op.family
    if isinstance(contract, CausalConvContract):
        if abi == "causal_conv1d_update":
            if strategy is not _ARCH35:
                raise missing("linear_attention_arch35_abi_required", f"{abi} requires Arch35")
            plan = strategy.build(op, ctx)
        else:
            plan = build_causal_conv1d_plan(contract, ctx)
    elif contract.operation.startswith("recover_ssm_"):
        if contract.abi == "commit_recurrent_kda":
            if strategy is not _ARCH35:
                raise missing("linear_attention_arch35_abi_required", f"{abi} requires Arch35")
            plan = strategy.build(op, ctx)
        else:
            plan = build_recover_ssm_plan(contract, ctx)
    else:
        plan = strategy.build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(ctx.occurrence, op, plan, descriptor.implementation,
                            descriptor.sources, descriptor)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build the same plan through the legacy analytical planner entrypoint."""
    if geometry.datapath is None:
        raise missing("linear_attention_datapath_required", "Geometry must declare the ISA datapath")
    ctx = OperatorPlanningContext(
        OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry, geometry.datapath, stream_owners=geometry.stream_owners,
    )
    return resolve_operator_plan(op, ctx).recipe


def register_family() -> OperatorFamilySpec:
    """Declare recipe ownership for explicit registration in a FamilyRegistry."""
    return OperatorFamilySpec(
        name="linear_attention", families=OPERATIONS | {"linear_attention", "recover_ssm"},
        owned_recipe_ids=OPERATIONS | frozenset(ABI_OPERATIONS), resolve=resolve_operator_plan,
    )


__all__ = [
    "CausalConvContract",
    "LinearAttentionContract",
    "RECOVER_SSM_FAMILIES",
    "RankOneTransition",
    "RecoverSsmCommitPlan",
    "RecoverSsmVerifyPlan",
    "StateResidencyReport",
    "WyChunkWork",
    "analyze_state_residency",
    "build",
    "build_recover_ssm_commit",
    "build_recover_ssm_verify",
    "compute_wy_chunk_work",
    "rank_one_transition",
    "register_family",
    "resolve_operator_plan",
    "select_flash_kda_template",
]
