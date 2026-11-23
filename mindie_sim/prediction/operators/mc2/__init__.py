"""Public entrypoint and family ownership for governed MC2 operators.

[Role] Select the MC2 strategy from the declared architecture or protocol.
[Invariants] Every operator has one selected caller, transport, and topology.
[Timeouts & Locks] Constant-time dispatch before finite planning; no I/O.
[Side-Effect Policy] Imports and entrypoints construct metadata only.
"""
from __future__ import annotations

from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import (
    GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec,
)
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...results import OpOccurrence
from ..contracts import ArchStrategy, OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from . import arch22, arch35
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
    leaf.STRATEGY.arch_id: leaf.STRATEGY for leaf in (arch22, arch35)
}
_NATIVE_PROTOCOLS = {leaf.NATIVE_PROTOCOL_KIND: leaf for leaf in (arch22, arch35)}
_DEFAULT_NATIVE_PROTOCOL = arch35
NATIVE_PROTOCOL_NETWORK = _DEFAULT_NATIVE_PROTOCOL.NETWORK
_NATIVE_PROTOCOL_FIELDS = frozenset({
    "kind", "max_inflight_per_rank", "dispatch_quant", "global_bs", "chunk_bytes",
    "hop_startup_us", "receive_capacity_bytes_per_peer",
})


def select_native_protocol(kernels):
    """Validate and return the caller's explicit protocol declaration."""
    raw = kernels.get("native_moe_protocol")
    if raw is None:
        return None
    fields = set(raw)
    if "kind" not in fields or fields - _NATIVE_PROTOCOL_FIELDS:
        raise ModelingError("native_moe_protocol_requires_supported_caller")
    strategy = _NATIVE_PROTOCOLS.get(raw["kind"])
    if strategy is None:
        raise ModelingError("native_moe_protocol_requires_supported_caller")
    strategy.validate_native_protocol(raw)
    return raw


def _protocol_module_for_arch(arch_family: str = "", moe_comm=None):
    """Select the native MC2 protocol leaf from declared architecture/policy."""
    arch = str(arch_family or "").strip().upper()
    if not arch and moe_comm:
        comm_map = dict(moe_comm)
        arch = str(comm_map.get("arch_family", "") or "").strip().upper()
        if not arch:
            policy = str(comm_map.get("policy", "") or "").strip().lower()
            if policy in {"fused_or_capacity", "capacity_and_expert_density"}:
                arch = "A3"
            elif policy == "capacity_and_world_size":
                arch = "A5"
    if arch in {"A2", "A3"}:
        return arch22
    return arch35


def native_protocol_recipes(kernels):
    """Register recipes for the declared or automatically selected protocol."""
    selected = select_native_protocol(kernels)
    if selected is None:
        return {
            **arch35.native_protocol_recipes(None),
            arch22.mc2_local_caller.LOCAL: arch22.mc2_local_caller.PlannedRecipe(
                arch22.mc2_local_caller.build
            ),
        }
    return _NATIVE_PROTOCOLS[selected["kind"]].native_protocol_recipes(selected)


def common_native_protocol(chosen, fused_level, moe_comm, *, arch_family: str = ""):
    """Describe the protocol used by common MoE communication selection."""
    leaf = _protocol_module_for_arch(arch_family, moe_comm)
    return leaf.common_native_protocol(chosen, fused_level, moe_comm)


def lower_native_protocol(ops, *, selected, **caller):
    """Apply the selected protocol's graph lowering."""
    return _NATIVE_PROTOCOLS[selected["kind"]].lower_native_protocol(
        ops, selected=selected, **caller)


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    """Resolve the caller and attach the occurrence-bound pipeline descriptor."""
    if op.recipe == "arch22_v2_caller_local" and "mc2_v2_local" in dict(op.metadata):
        from mindie_sim.prediction.operators.mc2.mc2_local_caller import LOCAL, SOURCE as LOCAL_SOURCE, build as build_local_caller

        plan = build_local_caller(op, ctx.geometry)
        return ResolvedOperator(ctx.occurrence, op, plan, LOCAL, (LOCAL_SOURCE,), None)
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
        families=frozenset({
            "mc2", "all_gather_matmul", "matmul_reduce_scatter", "matmul_all_reduce",
            "moe_dispatch_v2", "moe_combine_v2",
        }),
        owned_recipe_ids=frozenset({
            "mc2_all_gather_matmul", "mc2_matmul_reduce_scatter", "mc2_matmul_all_reduce",
            "mc2_dispatch_v2", "mc2_combine_v2",
            "all_gather_matmul_v2_arch35", "matmul_reduce_scatter_v2_arch35",
            "matmul_all_reduce_arch35", "arch22_v2_caller_local",
        }),
        resolve=resolve_operator_plan,
    )


__all__ = [
    "Mc2CohortSlice",
    "Mc2Contract",
    "Mc2RowLayout",
    "NATIVE_PROTOCOL_NETWORK",
    "build",
    "build_cohort_transport_stages",
    "combine_row_layout",
    "common_native_protocol",
    "dispatch_row_layout",
    "lower_native_protocol",
    "native_protocol_recipes",
    "register_family",
    "resolve_operator_plan",
    "select_native_protocol",
]
