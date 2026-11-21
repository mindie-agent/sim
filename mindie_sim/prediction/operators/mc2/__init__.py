"""Public entrypoint and family ownership for governed MC2 operators.

[Role] Select the MC2 strategy from the declared compute architecture.
[Invariants] Every operator has one selected caller, transport, and topology.
[Timeouts & Locks] Constant-time dispatch before finite planning; no I/O.
[Side-Effect Policy] Imports and entrypoints construct metadata only.
"""
from __future__ import annotations

from mindie_sim.schema.datapath import (
    GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec,
)
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...results import OpOccurrence
from ..contracts import ArchStrategy, OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from .arch22 import STRATEGY as _ARCH22
from .arch35 import STRATEGY as _ARCH35
from .common import (
    Mc2CohortSlice,
    Mc2Contract,
    Mc2RowLayout,
    build_cohort_transport_stages,
    capability_gap,
    combine_row_layout,
    describe,
    dispatch_row_layout,
)

_STRATEGIES: dict[ArchId, ArchStrategy] = {
    _ARCH22.arch_id: _ARCH22,
    _ARCH35.arch_id: _ARCH35,
}


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    """Resolve the caller and attach the occurrence-bound pipeline descriptor."""
    strategy = _STRATEGIES.get(ctx.compute_arch.arch_id)
    if strategy is None:
        raise capability_gap(op, ctx, "unsupported_compute_architecture")
    plan = strategy.build(op, ctx)
    descriptor = describe(op, ctx, plan)
    return ResolvedOperator(ctx.occurrence, op, plan, descriptor.implementation, descriptor.sources, descriptor)


def build(op: OpSpec, geometry: Geometry) -> OperatorPlan:
    """Build an MC2 plan using the placement and architecture in geometry."""
    datapath = geometry.datapath
    if datapath is None:
        arch = geometry.compute_arch
        arch_id = ArchId(arch.preset) if arch is not None and arch.preset else GENERIC_ARCH_ID
        paths = arch.paths if arch is not None else ()
        datapath = DatapathCapabilitySpec(
            compute_arch=ComputeArchIdentity(arch_id, arch),
            aic_count=geometry.compute_units,
            aiv_count=geometry.resource_units.get("vector", 0),
            aic_aiv_pairs=(),
            usable_paths=paths,
            matrix_output_paths=frozenset(path.target for path in paths if path.source == "l0c"),
            ub_peer_edges=(),
            stars=None,
        )
    ctx = OperatorPlanningContext(
        OpOccurrence(geometry.replica, op.stage, geometry.rank, op.op_id),
        geometry, datapath, stream_owners=geometry.stream_owners,
    )
    return resolve_operator_plan(op, ctx).recipe


def register_family() -> OperatorFamilySpec:
    """Declare the MC2 families and owned recipe IDs."""
    return OperatorFamilySpec(
        name="mc2",
        families=frozenset({"mc2", "all_gather_matmul", "matmul_reduce_scatter", "moe_dispatch_v2", "moe_combine_v2"}),
        owned_recipe_ids=frozenset({
            "mc2_all_gather_matmul", "mc2_matmul_reduce_scatter", "mc2_dispatch_v2", "mc2_combine_v2",
            "all_gather_matmul_v2_arch35", "matmul_reduce_scatter_v2_arch35", "arch22_v2_caller_local",
        }),
        resolve=resolve_operator_plan,
    )


__all__ = [
    "Mc2CohortSlice",
    "Mc2Contract",
    "Mc2RowLayout",
    "build",
    "build_cohort_transport_stages",
    "combine_row_layout",
    "dispatch_row_layout",
    "register_family",
    "resolve_operator_plan",
]
