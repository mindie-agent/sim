"""Extensible mechanism simulator.

[Role] Publish the single prediction engine and neutral input contracts.
[Invariants] Simulation, primitive calibration and validation are independent.
[Timeouts & Locks] Imports perform no hardware access or file writes.
[Side-Effect Policy] Pure declarations.
"""
from .schema.case import Case, Software
from .schema.hardware import Hardware, Device, Link
from .schema.model import ModelSpec, LayerPlan, RequestSpec, WorkloadSpec, SloSpec
from .schema.parallel import ParallelSpec
from .schema.common import ModelingError
from .prediction.api import predict
from .prediction.disaggregated import predict_pd
from .prediction.graph_analysis import graph_costs
from .prediction.operator_analysis import operator_costs, target_graphs, target_operators
from .study import study
from .prediction.call import call_costs
from .prediction.call_builders import build_call
from .compose import compose_case, resolve_hardware_sku
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
from .advise import advise_case, load_case_from_profile
from .model_catalog import customize_model_spec
from .prediction.afd import effective_pool_link, predict_afd, solve_afd_pareto
from .prediction.deepep import (
    analyze_deepep_collectives,
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
    solve_hardware_sla_throughput,
)
from .prediction.offload import (
    OffloadPolicy,
    analyze_kv_cache_offload,
    prepare_expert_offload,
)
from .prediction.skew_analysis import (
    analyze_chunked_prefill_decode_interference,
    analyze_moe_routing_imbalance,
    analyze_multi_dp_barrier_skew,
)
from .prediction.topology_tiers import (
    EP_TIER_PRESETS,
    analyze_ep_mc2_communication,
    compute_ep_effective_bandwidth,
    compute_speculative_accept_step,
)
from .sweep import (
    compute_pareto_frontier,
    enumerate_parallel_candidates,
    sweep_pd_disaggregation_ratios,
    sweep_topology_grid,
)

__version__ = "0.9.9"
__all__ = ["predict", "predict_pd", "graph_costs", "operator_costs", "target_graphs", "target_operators", "study",
           "call_costs", "build_call",
           "compose_case", "resolve_hardware_sku", "customize_model_spec",
           "HARDWARE_ARCHETYPE_TEMPLATES", "export_hardware_sku_template",
           "build_custom_hardware_machine", "register_custom_hardware_sku",
           "unregister_custom_hardware_sku", "clear_custom_hardware_skus",
           "list_custom_hardware_skus", "load_custom_hardware_skus_from_file",
           "resolve_custom_hardware_sku", "compose_custom_case",
           "advise_case", "load_case_from_profile",
           "analyze_network_efficiency", "analyze_operator_row", "evaluate_target_efficiency",
           "fit_efficiency_targets", "scale_case_hardware", "solve_hardware_sla_throughput",
           "analyze_multi_dp_barrier_skew", "analyze_moe_routing_imbalance",
           "analyze_chunked_prefill_decode_interference",
           "EP_TIER_PRESETS", "compute_speculative_accept_step",
           "compute_ep_effective_bandwidth", "analyze_ep_mc2_communication",
           "deepep_group_volume", "deepep_intra_volume", "standard_alltoall_volume",
           "analyze_deepep_collectives", "analyze_shared_expert_overlap",
           "effective_pool_link", "predict_afd", "solve_afd_pareto",
           "OffloadPolicy", "prepare_expert_offload", "analyze_kv_cache_offload",
           "enumerate_parallel_candidates", "sweep_topology_grid", "sweep_pd_disaggregation_ratios",
           "compute_pareto_frontier",
           "Case", "Software", "Hardware", "Device", "Link",
           "ModelSpec", "LayerPlan", "RequestSpec", "WorkloadSpec", "SloSpec",
           "ParallelSpec", "ModelingError"]
