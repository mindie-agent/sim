"""Public resolver and registration for the governed grouped MoE family.

[Role] Select one architecture strategy from the declared compute identity.
[Invariants] Family ownership and pipeline provenance are explicit.
[Timeouts & Locks] Constant-time dispatch before finite planning; no I/O.
[Side-Effect Policy] Imports and entrypoints construct metadata only.
"""
from __future__ import annotations

from typing import Literal

from mindie_sim.schema.datapath import (
    GENERIC_ARCH_ID, ArchId, ComputeArchIdentity, DatapathCapabilitySpec, SourceRef,
)
from mindie_sim.schema.operators import OpSpec
from ...mechanism import Geometry, OperatorPlan
from ...ownership import unit_group
from ...results import OpOccurrence
from ..contracts import ArchStrategy, OperatorFamilySpec, OperatorPlanningContext, ResolvedOperator
from .arch22 import STRATEGY as _ARCH22
from .arch35 import STRATEGY as _ARCH35, fit_ub as _arch35_fit_ub
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

TARGET_BACKEND: Literal["ascend"] = "ascend"
SOURCES: tuple[SourceRef, ...] = _ARCH22.sources + _ARCH35.sources

_PEAK_DTYPE_ALIASES = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3fn": "fp8_e4m3",
    "fp4": "fp4_e2m1",
    "hif8": "hifloat8",
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}

_STRATEGIES: dict[ArchId, ArchStrategy] = {
    _ARCH22.arch_id: _ARCH22,
    _ARCH35.arch_id: _ARCH35,
}


def _validate_megamoe_composite_target(op: OpSpec, ctx: OperatorPlanningContext, metadata: dict) -> None:
    if ctx.compute_arch.arch_id != _ARCH35.arch_id or not ctx.datapath.has_l0c_to_ub:
        raise capability_gap(op, ctx, "megamoe_arch35_requires_arch35_l0c_to_ub")
    geom = dict(metadata.get("megamoe_geometry") or ())
    spec = ctx.geometry.compute_arch
    if spec is None or not spec.native_modes:
        return
    native_matrix = {
        operand
        for mode in spec.native_modes
        if mode.kind == "matrix"
        for operand in mode.operands
    }
    dtypes = [geom.get("compute_dtype") or op.dtype or "bf16"]
    if geom.get("shared_intermediate") or geom.get("shared_inter"):
        shared_dt = geom.get("shared_compute_dtype") or dtypes[0]
        dtypes.append(shared_dt)
        if geom.get("shared_weight_dtype"):
            dtypes.append(geom["shared_weight_dtype"])
    for raw_dtype in dtypes:
        key = str(raw_dtype).lower()
        canonical = _PEAK_DTYPE_ALIASES.get(key, key)
        if canonical not in native_matrix:
            raise capability_gap(op, ctx, f"megamoe_arch35_unsupported_compute_dtype:{raw_dtype}")


def resolve_operator_plan(op: OpSpec, ctx: OperatorPlanningContext, /) -> ResolvedOperator:
    """Bind the plan and occurrence descriptor to the same source evidence."""
    metadata = dict(op.metadata)
    if op.recipe == "megamoe_arch35" and "megamoe_geometry" in metadata and not {
        "matrix_group_rows", "group_rows", "expert_rows", "group_list",
    }.intersection(metadata):
        _validate_megamoe_composite_target(op, ctx, metadata)
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


def strategy_for_arch(arch_id: ArchId) -> ArchStrategy | None:
    """Return the governed grouped-MoE strategy registered for ``arch_id``."""
    return _STRATEGIES.get(arch_id)


def fit_arch35_ub(contract: GroupedMoeContract, ctx: OperatorPlanningContext) -> tuple[int, int]:
    """Return the Arch35 UB tile rows and slot byte budget for ``contract``."""
    return _arch35_fit_ub(contract, ctx)


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
    "fit_arch35_ub",
    "fractal_issues",
    "fractal_k0",
    "normalize_group_rows",
    "register_family",
    "resolve_operator_plan",
    "schedule_blaze_groups",
    "strategy_for_arch",
]
