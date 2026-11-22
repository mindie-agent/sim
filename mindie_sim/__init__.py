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
from .public.errors import CapabilityGap, UnmodeledCapabilityError
from .prediction.api import predict
from .prediction.disaggregated import predict_pd
from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs, target_operators
from .prediction.structure import parse_case
from .study import study
from .prediction.call_builders import call_costs
from .prediction.call_builders import build_call
from .compose import compose_case, resolve_hardware_sku
from .feature_mechanisms import (
    OptimizationMechanism,
    list_optimization_mechanisms,
    register_optimization_mechanism,
)
from .feature_mechanisms import compare_feature_tracks
from .stack_tracks import (
    RuntimeStackProfile,
    StackMechanismBinding,
    load_runtime_stack_manifest,
    register_runtime_stack,
)
from .hardware_derivation import derive_hardware_sku
from .hardware_derivation import EfficiencyCurve
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
from mindie_sim.prediction.operators.moe_transport.deepep import (
    analyze_shared_expert_overlap,
    deepep_group_volume,
    deepep_intra_volume,
    standard_alltoall_volume,
)
from .prediction.efficiency import (
    analyze_network_efficiency,
    analyze_operator_row,
    scale_case_hardware,
    solve_sla_sizing,
)
from .prediction.offload import OffloadPolicy
from .prediction.frontend.speculation import compute_speculative_accept_step
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
from .api import optimize_ladder
from .public import (
    CORE_PROVIDER,
    ENTRY_POINT_GROUP,
    INTERFACE_VERSION,
    ExecutionEnvironment,
    Extension,
    ExtensionContext,
    ExtensionError,
    FormulaKernelEvaluation,
    FormulaKernelSpec,
    FormulaTensorSpec,
    Provider,
    USER_PROFILE_SCHEMA,
    UserProfile,
    bind_kernel_implementation,
    evaluate_formula_kernel,
    load_extensions,
    load_user_profile,
    model_target_ceiling,
    simulate_aclnn_call,
    simulate_formula_kernel,
    simulate_graph,
    simulate_model,
    simulate_native_call,
    simulate_operator,
    simulate_service,
)

__version__ = "0.9.19"
__all__ = [
    "observations",
    "CapabilityGap", "UnmodeledCapabilityError",
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
    "FormulaTensorSpec", "FormulaKernelSpec", "FormulaKernelEvaluation",
    "evaluate_formula_kernel", "simulate_formula_kernel",
    "bind_kernel_implementation", "model_target_ceiling",
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
    "compare_feature_tracks", "optimize_ladder",
]

_LAZY_SUBMODULES = frozenset({
    "api", "cli", "cohort_accuracy", "stack_registry", "structure",
})


_LAZY_ALIASES = {
    "cli_export": "mindie_sim.cli.export",
    "cohort_comparison": "mindie_sim.cohort_accuracy",
}


def __getattr__(name: str):
    if name in _LAZY_SUBMODULES:
        import importlib
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _LAZY_ALIASES:
        import importlib
        module = importlib.import_module(_LAZY_ALIASES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__) | _LAZY_SUBMODULES)
