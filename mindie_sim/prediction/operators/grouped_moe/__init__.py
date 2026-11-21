"""Public resolver and registration for the governed grouped MoE family.

[Role] Select one architecture strategy from the declared compute identity.
[Invariants] Family ownership and pipeline provenance are explicit.
[Timeouts & Locks] Constant-time dispatch before finite planning; no I/O.
[Side-Effect Policy] Imports and entrypoints construct metadata only.
"""
from __future__ import annotations

from mindie_sim.schema.datapath import (
    GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec,
)
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...ownership import unit_group
from ...results import OpOccurrence
from ..contracts import ArchStrategy, OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from .arch22 import STRATEGY as _ARCH22
from .arch35 import STRATEGY as _ARCH35
from .common import (
    BlazeGroupSchedule,
    BlazeGroupSlice,
    GroupedMoeContract,
    MoECohortSplit,
    capability_gap,
    describe,
    fractal_issues,
    fractal_k0,
    normalize_group_rows,
    schedule_blaze_groups,
)

from .lowering import DualBatchMoEPipeline

_STRATEGIES: dict[ArchId, ArchStrategy] = {
    _ARCH22.arch_id: _ARCH22,
    _ARCH35.arch_id: _ARCH35,
}


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    """Bind the plan and occurrence descriptor to the same source evidence."""
    metadata = dict(op.metadata)
    if op.recipe == "megamoe_arch35" and "megamoe_geometry" in metadata and not {
        "matrix_group_rows", "group_rows", "expert_rows", "group_list",
    }.intersection(metadata):
        from ..mc2.megamoe import NATIVE_KIND, build as build_megamoe_composite

        plan = build_megamoe_composite(op, ctx.geometry)
        source = (plan.metadata or {}).get("source_program")
        return ResolvedOperator(
            ctx.occurrence, op, plan, NATIVE_KIND, (source,) if source else (), None,
        )
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise capability_gap(op, ctx, "unsupported_compute_architecture")
    plan = strategy.build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(
        ctx.occurrence, op, plan, descriptor.implementation, descriptor.sources, descriptor
    )


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build the standalone recipe from explicit physical geometry."""
    datapath = geometry.datapath
    if datapath is None:
        arch = geometry.compute_arch
        paths = arch.paths if arch is not None else ()
        arch_id = ArchId(arch.preset) if arch is not None and arch.preset else GENERIC_ARCH_ID
        aic, aiv = geometry.compute_units, geometry.resource_units.get("vector", 0)
        datapath = DatapathCapabilitySpec(
            compute_arch=ComputeArchIdentity(arch_id, arch),
            aic_count=aic,
            aiv_count=aiv,
            aic_aiv_pairs=tuple((owner, unit_group(owner, aic, aiv)) for owner in range(aic)) if aiv else (),
            usable_paths=paths,
            matrix_output_paths=frozenset(path.target for path in paths if path.source == "l0c"),
            ub_peer_edges=(),
            stars=None,
            buffers=tuple(sorted(geometry.buffers.items())),
        )
    metadata = dict(op.metadata)
    abi = metadata.get("requested_abi") or metadata.get("abi")
    ctx = OperatorPlanningContext(
        OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry, datapath, requested_abi=abi, stream_owners=geometry.stream_owners,
        admission_memory_view=geometry.admission_memory_view,
    )
    return resolve_operator_plan(op, ctx).recipe


def register_family() -> OperatorFamilySpec:
    """Return the package's family and recipe ownership declaration."""
    return OperatorFamilySpec(
        name="grouped_moe",
        families=frozenset({"dense_gemm", "grouped_gemm", "grouped_moe", "dual_batch_moe"}),
        owned_recipe_ids=frozenset({
            "dense_gemm", "grouped_moe", "grouped_gemm", "gmm_swiglu_quant", "dequant_swiglu_quant",
            "mega_moe", "dual_batch_moe", "megamoe_arch35",
        }),
        resolve=resolve_operator_plan,
    )


__all__ = [
    "BlazeGroupSchedule",
    "BlazeGroupSlice",
    "DualBatchMoEPipeline",
    "GroupedMoeContract",
    "MoECohortSplit",
    "build",
    "fractal_issues",
    "fractal_k0",
    "normalize_group_rows",
    "register_family",
    "resolve_operator_plan",
    "schedule_blaze_groups",
]
