"""mindie_sim: Ascend NPU four-layer simulation and boundary solver.

[Role]
  Public entry points for Layer 1 hardware/buffer/thermal analysis, Layer 2
  periodic microstage pipeline solving, Layer 3 subgraph fusion analysis and
  Layer 4 whole-network parallel decomposition.

[Invariants]
  * Layer modules are independently importable; the whole-network bridge is
    loaded lazily so hardware/pipeline/subgraph use does not require msmodeling.
  * Uncalibrated inputs never acquire fabricated numeric latency.
  * No module in this package imports torch at import time or starts services.

[Timeouts & Locks]
  Import-time pure computation only.  The CLI performs explicit file reads and
  bounded provider/git probes; no lock is held across calls.

[Side-Effect Policy]
  Imports do not write files, import msmodeling/torch, or change device state.
"""
from __future__ import annotations

from .hardware import (
    AMBIGUOUS_SKU_REASON,
    BUILTIN_SKU_CATALOG,
    A5_CANN_SIM_LINK_CEILINGS_BPC,
    A5_REFERENCE_PEAKS,
    A5_FLASHMLA_C8_GOLDEN_WORKLOAD,
    BufferAllocation,
    ClockAndTopologySpec,
    CubeSpec,
    DeviceCapabilityProfile,
    HardwareError,
    MemoryHierarchySpec,
    NpuSmiChip,
    NpuSmiDevice,
    NpuSmiSnapshot,
    PlatformConfig,
    PcieLinkSnapshot,
    TRANSFER_CHANNELS,
    TransferChannelSpec,
    NpuTopologySnapshot,
    VectorSpec,
    check_mx_scale_layout,
    cross_check_capacities,
    cube_reference_cycles,
    cycles_to_us,
    default_transfer_channels,
    evaluate_transfer_plan,
    hardware_report,
    infer_arch_family,
    is_ambiguous_family_token,
    link_reference_bounds,
    load_sku_profile,
    parse_cannsim_link_ceilings,
    parse_npu_smi_topo,
    parse_pcie_link_info,
    parse_platform_ini,
    parse_platform_ini_text,
    reference_time_us,
    resolve_device_capability_profile,
    sensitivity_matrix,
    solve_collective_bound,
    solve_transfer_bound,
    thermal_bounds,
    unmodeled_thermal_bounds,
    validate_buffer_allocations,
)
from .pipeline import (
    PipelineError,
    PipelineGraph,
    compare_pipeline_variants,
    expand_pipeline_graph,
    normalize_stage_durations,
    parse_pipeline_graph,
    pipeline_self_check,
    solve_periodic_pipeline,
)
from .subgraph import SubgraphError, analyze_subgraph_fusion
from .a3_types import A3InputError, CaseSpec, MachineSpec, ModelSpec, RuntimeSpec, WorkloadSpec
from .a3_cost import CalibrationStore, PredictiveCostModel, attention_flops, gemm_flops
from .a3_workload import derive_iteration_work, memory_plan
from .a3_resources import ResourceEngine, ResourceSpec, simulate_microbatch_pipeline
from .a3_comm_identity import (
    CommIdentityRecord, identity_fixture, parse_profile_identity,
)
from .a3_framework import TokenLinearCostBackend, predict, run_predictive
from .a3_should import should_cost, normalized_shared_cv_search, normalized_chunked_pipeline_search
from .a3_search import search
from .a3_calibration_store import SQLiteCalibrationStore
from .a3_measurement import record_prediction_misses
from .a3_measurement import (
    CpuMatmulRunner, TorchNpuRunner, run_manifest_file, run_pending,
)
from .model_catalog import CATALOG, list_models, load_model
from .a3_real_data import (
    CommunicationCalibration, CvCalibration, import_a3_measurements,
    import_communication_measurements, machine_from_environment,
)

__version__ = "0.7.0"


def net_report(*args, **kwargs):
    """Lazy Layer 4 bridge; imports mindie_profiler only when called."""
    from .net import net_report as _net_report

    return _net_report(*args, **kwargs)


__all__ = [
    "__version__",
    "A5_CANN_SIM_LINK_CEILINGS_BPC",
    "A5_REFERENCE_PEAKS",
    "A5_FLASHMLA_C8_GOLDEN_WORKLOAD",
    "BufferAllocation",
    "HardwareError",
    "NpuSmiChip",
    "NpuSmiDevice",
    "NpuSmiSnapshot",
    "PlatformConfig",
    "check_mx_scale_layout",
    "cross_check_capacities",
    "cube_reference_cycles",
    "cycles_to_us",
    "hardware_report",
    "link_reference_bounds",
    "parse_cannsim_link_ceilings",
    "parse_platform_ini",
    "parse_platform_ini_text",
    "reference_time_us",
    "sensitivity_matrix",
    "thermal_bounds",
    "unmodeled_thermal_bounds",
    "validate_buffer_allocations",
    "PipelineError",
    "PipelineGraph",
    "compare_pipeline_variants",
    "expand_pipeline_graph",
    "normalize_stage_durations",
    "parse_pipeline_graph",
    "pipeline_self_check",
    "solve_periodic_pipeline",
    "SubgraphError",
    "analyze_subgraph_fusion",
    "net_report",
    "A3InputError",
    "CaseSpec",
    "MachineSpec",
    "ModelSpec",
    "RuntimeSpec",
    "WorkloadSpec",
    "CalibrationStore",
    "PredictiveCostModel",
    "attention_flops",
    "gemm_flops",
    "derive_iteration_work",
    "memory_plan",
    "ResourceEngine",
    "ResourceSpec",
    "simulate_microbatch_pipeline",
    "TokenLinearCostBackend",
    "CommIdentityRecord",
    "identity_fixture",
    "parse_profile_identity",
    "predict",
    "run_predictive",
    "should_cost",
    "normalized_shared_cv_search",
    "normalized_chunked_pipeline_search",
    "search",
    "SQLiteCalibrationStore",
    "record_prediction_misses",
    "CpuMatmulRunner",
    "TorchNpuRunner",
    "run_pending",
    "run_manifest_file",
    "CATALOG",
    "list_models",
    "load_model",
    "CommunicationCalibration",
    "CvCalibration",
    "import_a3_measurements",
    "import_communication_measurements",
    "machine_from_environment",
]
