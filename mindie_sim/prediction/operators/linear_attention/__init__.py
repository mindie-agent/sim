"""Public entrypoint for the governed linear-attention family.

[Role] Resolve tensor and ABI bindings against a declared ISA and datapath.
[Invariants] Each ISA has one strategy. Generic convolution and RecoverSSM
  share source-independent contracts and keep explicit ABI requests intact.
[Timeouts & Locks] Bounded in-memory planning on the selected strategy.
[Side-Effect Policy] Imports and planning perform no device or external I/O.
"""
from __future__ import annotations

from dataclasses import replace

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import GENERIC_ARCH_ID, ArchId
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...program import RecipeContext
from ...results import OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from . import gdn_decode
from .arch22 import STRATEGY as _ARCH22
from .arch35 import STRATEGY as _ARCH35
from .common import (
    ABI_OPERATIONS,
    CHUNK32_KERNELS,
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
    make_plan,
    missing,
    rank_one_transition,
    request_layout_metadata,
    request_lengths,
    select_flash_kda_template,
    validate_contract,
)

_STRATEGIES = {_ARCH22.arch_id: _ARCH22, _ARCH35.arch_id: _ARCH35}


def chunk32_timing_policy(
    geometry: Geometry, context: RecipeContext, step_sync_ms: float,
) -> tuple[float, float]:
    """Resolve launch setup and per-step synchronization for the Chunk32 plan."""
    if geometry.datapath is not None:
        arch_id = geometry.datapath.compute_arch.arch_id
    else:
        spec = geometry.compute_arch
        arch_id = ArchId(spec.preset) if spec is not None and spec.preset else GENERIC_ARCH_ID
    if arch_id == GENERIC_ARCH_ID and not getattr(geometry.compute_arch, "preset", ""):
        return 0.0, 0.0
    strategy = _STRATEGIES.get(arch_id)
    if strategy is None:
        raise ModelingError(f"linear_attention_arch_unsupported:{arch_id}")
    return strategy.chunk32_timing_policy(context, step_sync_ms)


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext) -> ResolvedOperator:
    """Bind a governed operation to its executable plan and source descriptor."""
    if ctx.requests and not ctx.layout:
        ctx = replace(ctx, layout=op.layout)
    if op.recipe in gdn_decode.RECIPE_IDS:
        plan = gdn_decode.build(op, ctx.geometry, ctx=ctx)
        return ResolvedOperator(ctx.occurrence, op, plan, op.recipe)
    if op.recipe in CHUNK32_KERNELS:
        from . import gdn_chunk32
        from ...physical_recipe_registry import uses_retained_source_plan

        if uses_retained_source_plan(op):
            source_op = op
            if ctx.requests:
                lengths = tuple(length for length in request_lengths(ctx.requests) if length)
                if not lengths:
                    plan = make_plan(ctx, op.recipe, (), metadata={"total_chunks": 0})
                    return ResolvedOperator(ctx.occurrence, op, plan, op.recipe)
                source_op = replace(op, metadata=tuple({**dict(op.metadata), "lengths": lengths}.items()))
            plan = gdn_chunk32.build(source_op, ctx.geometry, ctx=ctx)
            if ctx.requests:
                plan = replace(plan, metadata={**plan.metadata, **request_layout_metadata(ctx)})
            return ResolvedOperator(ctx.occurrence, op, plan, op.recipe)
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
    """Build through the governed resolver from physical geometry."""
    if op.recipe in gdn_decode.RECIPE_IDS:
        return gdn_decode.build(op, geometry)
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
        owned_recipe_ids=OPERATIONS | frozenset(ABI_OPERATIONS) | gdn_decode.RECIPE_IDS,
        resolve=resolve_operator_plan,
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
    "chunk32_timing_policy",
    "compute_wy_chunk_work",
    "rank_one_transition",
    "register_family",
    "resolve_operator_plan",
    "select_flash_kda_template",
]
