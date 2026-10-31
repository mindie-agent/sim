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
from .prediction.efficiency import (
    analyze_network_efficiency,
    analyze_operator_row,
    evaluate_target_efficiency,
    fit_efficiency_targets,
    scale_case_hardware,
    solve_hardware_sla_throughput,
)

__version__ = "0.8.2"
__all__ = ["predict", "predict_pd", "graph_costs", "operator_costs", "target_graphs", "target_operators", "study",
           "call_costs", "build_call",
           "analyze_network_efficiency", "analyze_operator_row", "evaluate_target_efficiency",
           "fit_efficiency_targets", "scale_case_hardware", "solve_hardware_sla_throughput",
           "Case", "Software", "Hardware", "Device", "Link",
           "ModelSpec", "LayerPlan", "RequestSpec", "WorkloadSpec", "SloSpec",
           "ParallelSpec", "ModelingError"]
