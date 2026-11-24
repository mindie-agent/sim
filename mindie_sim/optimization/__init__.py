"""Constructive optimization and certified lower bounds for MindIE Sim.

[Role] Publish the optimization inputs, certificates, solvers and reports.
[Invariants] Core arithmetic can be used before the rewrite oracle is loaded.
[Timeouts & Locks] No import-time replay or locks.
[Side-Effect Policy] Re-exports only.
"""
from mindie_sim.prediction.runtime.exposed_observer import (
    ServiceGrantObserver,
    attribute_engine_exposure,
)
from .causal_tail import (
    MoERouteDecision,
    compute_acceptance_tail_closure,
    moe_route_legality,
    resolve_barrier_predecessors,
)
from .contracts import (
    BilinearContractionSpec, CatlassKernelSpec, DslKernelSpec, EdgeRouteBinding,
    EvgVisitorNode, FuseRegionSpec, FusedRegionSummary, PrimitiveCapacityEntry,
    PrimitiveCapacityTable, SemanticWorkItem, TensorRangeObligation, UpperCapacityProvider,
)
from .dual_bound import (
    BINARY64_ULP_SHIFT, C11_ULP_GUARD_FACTOR, ENDPOINT_ROUND_GUARD_NS, DagEdge,
    DualSubgradientReport, EpochAccount, EventDagSpec, FarkasCertificate,
    OptimalityCertificate, ResourceChoiceProgram, build_dual_inputs_from_scenario,
    certify_lower_bound, check_farkas_dual, check_optimality_certificate, eval_checked_path,
    inward_floor_ns, solve_checked_critical_path, solve_farkas_dual, verify_epoch_trace,
)
from .framework_solver import (
    CandidateSearchStats, CoreLatticePoint, FrameworkSolveResult, generate_gcd_core_lattice, solve_framework,
)
from .ladder import (
    FIVE_LEVEL_REPORT_SCHEMA, FiveLevelLadderMetrics, FiveLevelReport, FiveLevelWaterfall,
    LooMarginalImpact, OrderedTelescopeStep, RegionWallAttribution, partition_step_wall_ns,
    solve_five_level_ladder,
)

__all__ = (
    "MoERouteDecision", "compute_acceptance_tail_closure",
    "moe_route_legality", "resolve_barrier_predecessors",
    "ServiceGrantObserver", "attribute_engine_exposure",
    "BilinearContractionSpec", "CatlassKernelSpec", "DslKernelSpec", "EdgeRouteBinding",
    "EvgVisitorNode", "FuseRegionSpec", "FusedRegionSummary", "PrimitiveCapacityEntry",
    "PrimitiveCapacityTable", "SemanticWorkItem", "TensorRangeObligation", "UpperCapacityProvider",
    "BINARY64_ULP_SHIFT", "C11_ULP_GUARD_FACTOR",
    "ENDPOINT_ROUND_GUARD_NS", "DagEdge", "DualSubgradientReport", "EpochAccount",
    "EventDagSpec", "FarkasCertificate", "OptimalityCertificate", "ResourceChoiceProgram",
    "build_dual_inputs_from_scenario", "certify_lower_bound", "check_farkas_dual",
    "check_optimality_certificate", "eval_checked_path", "inward_floor_ns",
    "solve_checked_critical_path", "solve_farkas_dual", "verify_epoch_trace",
    "CandidateSearchStats", "CoreLatticePoint", "FrameworkSolveResult", "generate_gcd_core_lattice", "solve_framework",
    "FIVE_LEVEL_REPORT_SCHEMA", "FiveLevelLadderMetrics", "FiveLevelReport", "FiveLevelWaterfall",
    "LooMarginalImpact", "OrderedTelescopeStep", "RegionWallAttribution", "partition_step_wall_ns",
    "solve_five_level_ladder",
)
