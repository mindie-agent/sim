"""Constructive optimization and certified lower bounds for MindIE Sim.

[Role] Publish the optimization inputs, certificates, solvers and reports.
[Invariants] Core arithmetic can be used before the rewrite oracle is loaded.
[Timeouts & Locks] No import-time replay or locks.
[Side-Effect Policy] Re-exports only.
"""
from .contracts import (
    BilinearContractionSpec, CatlassKernelSpec, DslKernelSpec, EdgeRouteBinding,
    EvgVisitorNode, FuseRegionSpec, FusedRegionSummary, PrimitiveCapacityEntry,
    PrimitiveCapacityTable, SemanticWorkItem, TensorRangeObligation, UpperCapacityProvider,
)
from .dsl_solver import (
    DslSolveResult, select_catlass_candidates, select_dsl_candidates, solve_dsl,
    streamk_crossing_partials,
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
from .reload_bounds import (
    BITS_PER_BYTE, MAX_EXPLICIT_STRIDED_ROWS, SECTOR_BYTES, CompulsoryHbmCut,
    PebbleReloadResult, SurplusCouplingSite, build_surplus_coupling_sites,
    compute_compulsory_hbm_cut, direct_mac_pebble_reload_bound,
)

__all__ = (
    "BilinearContractionSpec", "CatlassKernelSpec", "DslKernelSpec", "EdgeRouteBinding",
    "EvgVisitorNode", "FuseRegionSpec", "FusedRegionSummary", "PrimitiveCapacityEntry",
    "PrimitiveCapacityTable", "SemanticWorkItem", "TensorRangeObligation", "UpperCapacityProvider",
    "DslSolveResult", "select_catlass_candidates", "select_dsl_candidates", "solve_dsl",
    "streamk_crossing_partials", "BINARY64_ULP_SHIFT", "C11_ULP_GUARD_FACTOR",
    "ENDPOINT_ROUND_GUARD_NS", "DagEdge", "DualSubgradientReport", "EpochAccount",
    "EventDagSpec", "FarkasCertificate", "OptimalityCertificate", "ResourceChoiceProgram",
    "build_dual_inputs_from_scenario", "certify_lower_bound", "check_farkas_dual",
    "check_optimality_certificate", "eval_checked_path", "inward_floor_ns",
    "solve_checked_critical_path", "solve_farkas_dual", "verify_epoch_trace",
    "CandidateSearchStats", "CoreLatticePoint", "FrameworkSolveResult", "generate_gcd_core_lattice", "solve_framework",
    "FIVE_LEVEL_REPORT_SCHEMA", "FiveLevelLadderMetrics", "FiveLevelReport", "FiveLevelWaterfall",
    "LooMarginalImpact", "OrderedTelescopeStep", "RegionWallAttribution", "partition_step_wall_ns",
    "solve_five_level_ladder", "BITS_PER_BYTE", "MAX_EXPLICIT_STRIDED_ROWS", "SECTOR_BYTES",
    "CompulsoryHbmCut", "PebbleReloadResult", "SurplusCouplingSite", "build_surplus_coupling_sites",
    "compute_compulsory_hbm_cut", "direct_mac_pebble_reload_bound",
)
