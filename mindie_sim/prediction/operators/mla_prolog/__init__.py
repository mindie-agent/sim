"""Public entry point for the governed MLA prolog family.

[Role] Bind an explicit compute architecture and ABI to one MLA prolog plan.
[Invariants] Each ISA leaf stays behind the package dispatcher. Legacy recipe
  callers use the same planning and lowering path as governed occurrences.
[Timeouts & Locks] Bounded by the selected strategy. Uses no I/O or locks.
[Side-Effect Policy] Returns plans and registration records without global mutation.
"""
from __future__ import annotations

from mindie_sim.schema.datapath import GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec
from mindie_sim.schema.operators import OpSpec

from ...mechanism import Geometry, OperatorPlan, PlannedRecipe
from ...results import FusedPipelineDescriptor, OpOccurrence
from ..contracts import OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from . import arch22 as _arch22, arch35 as _arch35
from .common import (
    BF16_BYTES,
    C8_BLOCK_SIZE,
    DEFAULT_KV_LORA_RANK,
    DEFAULT_PROLOG_HEADS,
    DEFAULT_PROLOG_HIDDEN,
    DEFAULT_QK_NOPE_DIM,
    DEFAULT_QK_ROPE_DIM,
    DEFAULT_Q_LORA_RANK,
    FAMILY,
    FP32_BYTES,
    MLA_PROLOG_V4_RECIPE,
    MX_GROUP_SIZE,
    SLOT_INDEX_BYTES,
    SUPPORTED_WEIGHT_MODES,
    WEIGHT_QUANT_BF16,
    WEIGHT_QUANT_INT8,
    WEIGHT_QUANT_MXFP8,
    PrologContract,
    make_op,
    unsupported,
    uses_split_m,
    validate_contract,
    w_uk_ownership,
)

_STRATEGIES = {"npu_arch_2201": _arch22.STRATEGY, "npu_arch_3510": _arch35.STRATEGY}


def _planning_context(op: OpSpec, geometry: Geometry) -> OperatorPlanningContext:
    datapath = geometry.datapath
    if datapath is None:
        declared = geometry.compute_arch
        identity = ComputeArchIdentity(
            ArchId(declared.preset) if (declared and declared.preset) else GENERIC_ARCH_ID,
            declared,
        )
        datapath = DatapathCapabilitySpec(
            compute_arch=identity, aic_count=geometry.compute_units,
            aiv_count=int(geometry.resource_units.get("vector", geometry.compute_units)),
            aic_aiv_pairs=(), usable_paths=(), matrix_output_paths=frozenset(),
            ub_peer_edges=(), stars=None,
        )
    return OperatorPlanningContext(
        occurrence=OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry=geometry, datapath=datapath, stream_owners=geometry.stream_owners,
    )


def build(op: OpSpec, geometry_or_ctx: Geometry | OperatorPlanningContext) -> OperatorPlan:
    ctx = (_planning_context(op, geometry_or_ctx)
           if isinstance(geometry_or_ctx, Geometry) else geometry_or_ctx)
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise unsupported(op, ctx, "unsupported_architecture")
    return strategy.build(op, ctx)


def _describe(op: OpSpec, ctx: OperatorPlanningContext, plan: OperatorPlan) -> FusedPipelineDescriptor:
    spec = plan.pipeline_spec
    return FusedPipelineDescriptor(
        ctx.occurrence, spec.family, spec.implementation, (op.op_id,), (),
        spec.phases, spec.sources, spec.repeats,
    )


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext) -> ResolvedOperator:
    plan = build(op, ctx)
    descriptor = _describe(op, ctx, plan)
    return ResolvedOperator(
        occurrence=ctx.occurrence, op=op, recipe=plan, implementation=descriptor.implementation,
        sources=descriptor.sources, descriptor=descriptor,
    )


def register_family() -> OperatorFamilySpec:
    return OperatorFamilySpec(
        name=FAMILY, families=frozenset({FAMILY}),
        owned_recipe_ids=frozenset({FAMILY, MLA_PROLOG_V4_RECIPE}),
        resolve=resolve_operator_plan,
    )


def recipes() -> dict[str, PlannedRecipe]:
    return {MLA_PROLOG_V4_RECIPE: PlannedRecipe(build)}


__all__ = [
    "PrologContract", "build", "make_op", "register_family", "resolve_operator_plan",
    "uses_split_m", "validate_contract", "w_uk_ownership",
]
