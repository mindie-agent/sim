"""Extensible mechanism simulator.

[Role] Publish the single prediction engine and neutral input contracts.
[Invariants] Simulation, primitive calibration and validation are independent.
[Timeouts & Locks] Imports perform no hardware access or file writes.
[Side-Effect Policy] Pure declarations.
"""
from . import observations
from .schema.case import Case, Software
from .schema.hardware import Hardware, Device, Link
from .schema.model import ModelSpec, LayerPlan, RequestSpec, WorkloadSpec, SloSpec
from .schema.parallel import ParallelSpec
from .schema.common import ModelingError
from .prediction.api import predict
from .prediction.disaggregated import predict_pd
from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs, target_operators
from .prediction.structure import parse_case
from .study import study
from .prediction.call import call_costs
from .prediction.call_builders import build_call
from .compose import compose_case, resolve_hardware_sku
from .feature_mechanisms import (
    OptimizationMechanism,
    list_optimization_mechanisms,
    register_optimization_mechanism,
)
from .feature_waterfall import compare_feature_tracks
from .stack_tracks import (
    RuntimeStackProfile,
    StackMechanismBinding,
    load_runtime_stack_manifest,
    register_runtime_stack,
)
from .hardware_derivation import derive_hardware_sku
from .hardware_efficiency import EfficiencyCurve
from .custom_hardware import (
    HARDWARE_ARCHETYPE_TEMPLATES,
    build_custom_hardware_machine,
    clear_custom_hardware_skus,
    compose_custom_case,
    export_hardware_sku_template,
    list_custom_hardware_skus,
    load_custom_hardware_skus_from_file,
    register_custom_hardware_sku,
    resolve_custom_hardware_sku,
    unregister_custom_hardware_sku,
)
from .model_catalog import customize_model_spec
from .prediction.deepep import (
    analyze_shared_expert_overlap,
    deepep_group_volume,
    deepep_intra_volume,
    standard_alltoall_volume,
)
from .prediction.efficiency import (
    analyze_network_efficiency,
    analyze_operator_row,
    evaluate_target_efficiency,
    fit_efficiency_targets,
    scale_case_hardware,
    solve_sla_sizing,
)
from .prediction.offload import OffloadPolicy
from .prediction.speculation import compute_speculative_accept_step
from .sweep import (
    compute_pareto_frontier,
    enumerate_parallel_candidates,
    reconstruct_sweep_summary,
    solve_hardware_sla_throughput,
    sweep_pd_disaggregation_ratios,
    sweep_topologies,
    sweep_topologies_iter,
    sweep_topology_grid,
    sweep_topology_grid_iter,
)
from .public import (
    CORE_PROVIDER,
    ENTRY_POINT_GROUP,
    INTERFACE_VERSION,
    ExecutionEnvironment,
    Extension,
    ExtensionContext,
    ExtensionError,
    Provider,
    USER_PROFILE_SCHEMA,
    UserProfile,
    load_extensions,
    load_user_profile,
    simulate_aclnn_call,
    simulate_graph,
    simulate_model,
    simulate_native_call,
    simulate_operator,
    simulate_service,
)

__version__ = "0.9.18"
__all__ = [
    "observations",
    "predict", "predict_pd", "graph_costs", "operator_costs",
    "target_graphs", "target_operators", "parse_case", "study",
    "call_costs", "build_call",
    "compose_case", "resolve_hardware_sku", "derive_hardware_sku",
    "EfficiencyCurve", "customize_model_spec",
    "HARDWARE_ARCHETYPE_TEMPLATES", "export_hardware_sku_template",
    "build_custom_hardware_machine", "register_custom_hardware_sku",
    "unregister_custom_hardware_sku", "clear_custom_hardware_skus",
    "list_custom_hardware_skus", "load_custom_hardware_skus_from_file",
    "resolve_custom_hardware_sku", "compose_custom_case",
    "analyze_network_efficiency", "analyze_operator_row",
    "evaluate_target_efficiency", "fit_efficiency_targets",
    "scale_case_hardware", "solve_sla_sizing",
    "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
    "analyze_shared_expert_overlap", "OffloadPolicy",
    "compute_speculative_accept_step",
    "enumerate_parallel_candidates", "sweep_topologies",
    "sweep_topology_grid", "sweep_topologies_iter",
    "sweep_topology_grid_iter", "reconstruct_sweep_summary",
    "solve_hardware_sla_throughput",
    "sweep_pd_disaggregation_ratios", "compute_pareto_frontier",
    "CORE_PROVIDER", "ENTRY_POINT_GROUP", "INTERFACE_VERSION",
    "ExecutionEnvironment", "USER_PROFILE_SCHEMA",
    "Extension", "ExtensionContext", "ExtensionError", "Provider", "UserProfile",
    "load_extensions", "load_user_profile",
    "simulate_operator", "simulate_native_call", "simulate_aclnn_call",
    "simulate_graph", "simulate_model", "simulate_service",
    "Case", "Software", "Hardware", "Device", "Link",
    "ModelSpec", "LayerPlan", "RequestSpec", "WorkloadSpec", "SloSpec",
    "ParallelSpec", "ModelingError",
    "OptimizationMechanism", "register_optimization_mechanism",
    "list_optimization_mechanisms",
    "RuntimeStackProfile", "StackMechanismBinding",
    "register_runtime_stack", "load_runtime_stack_manifest",
    "compare_feature_tracks",
]
