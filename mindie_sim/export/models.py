"""Canonical immutable data models and domain taxonomies for ``mindie_sim.export``.

[Role] Define all shared export constants, multi-ecosystem Ascend kernel backend
  and readiness taxonomies, canonical submodule and macro-module vocabularies,
  hardware/topology aliases, closed-form quantization storage rules, and frozen
  dataclass records exchanged across ``kernel_catalog``, ``memory_ledger``,
  ``reconcile``, ``bundle``, ``xlsx_exporter``, and ``trace_exporter``.
[Invariants] Strictly data and constants only: zero engine evaluation, zero
  filesystem I/O, zero OOXML or Chrome Trace serialization, and zero circular
  imports into sibling export modules.
[Timeouts & Locks] Pure in-memory declarations; no runtime locks or devices.
[Side-Effect Policy] Zero filesystem, network, or global-state side effects.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from mindie_sim.prediction.results import FusedPipelineDescriptor, OpOccurrence

#: Export bundle schema version.
EXPORT_BUNDLE_SCHEMA = "mindie-sim.export-bundle/2"
#: Binary GiB, MiB, KiB, and decimal TB divisors.
GIB_BYTES = 1024 * 1024 * 1024
MIB_BYTES = 1024 * 1024
KIB_BYTES = 1024
TB_BYTES = 1e12
#: Maximum read size for structured observation JSON manifests.
MAX_OBSERVATION_JSON_BYTES = 50 * MIB_BYTES
#: Microseconds per millisecond, nanoseconds per microsecond, microseconds per second.
US_PER_MS = 1000.0
NS_PER_US = 1000.0
US_PER_SECOND = 1_000_000.0
#: Positive denominator floors.
MIN_POSITIVE_SECONDS = 1e-12
MIN_DENOM_EPS = 1e-9
#: Default number of top gap-radar entries and top existing kernels to optimize.
DEFAULT_GAP_RADAR_TOP_K = 20
DEFAULT_TOP_OPTIMIZE_KERNELS = 6
#: Absolute tolerance floor (us) and relative APE threshold (%) for reconciliation.
RECONCILE_ABS_FLOOR_US = 2.0
RECONCILE_APE_THRESHOLD_PCT = 10.0
#: Quantile constants for measured APE summary statistics.
P50_QUANTILE = 0.5
P90_QUANTILE = 0.9
#: Minimum semantic token length for automated kernel_details.csv matching.
MIN_MATCH_TOKEN_LEN = 2
#: Default context, batching, model-length, scenario, and SLO parameters.
DEFAULT_EXPORT_CONTEXT_TOKENS = 1024
DEFAULT_MIN_BATCHED_TOKENS = 768
DEFAULT_MODEL_LEN_HEADROOM = 1024
DEFAULT_MIN_MODEL_LEN = 4096
DEFAULT_SCENARIO_PROMPT_TOKENS = 4096
DEFAULT_SCENARIO_OUTPUT_TOKENS = 128
DEFAULT_TPOT_SLO_MS = 50.0
DEFAULT_TTFT_SLO_MS = 5000.0
#: Default sparse DSA Lightning Indexer top-k fallback when unspecified on layer plan.
DEFAULT_SPARSE_INDEX_TOPK = 2048
#: Fallback sort key for any unrecognized archetype ordering.
ARCHETYPE_ORDER_FALLBACK = 999
#: Maximum character length for compact concatenated source-ref strings in Sheet 4.
MAX_SOURCE_REF_SUMMARY_LEN = 160
#: Closed-form MXFP and KV quantization group widths.
MXFP_GROUP_SIZE = 32
KV_QUANT_GROUP_SIZE = 128
#: Element width and per-channel auxiliary storage constants in bytes.
BF16_BYTES = 2
FP32_BYTES = 4
C8_INDEX_SCALE_BYTES = 2
FP8_INDEX_SCALE_BYTES = 4
INT8_CHANNEL_AUX_BYTES = 8
INT4_CHANNEL_AUX_BYTES = 16
#: Resource tuple slot count for single-op / whole-step Lean L* certificates.
CERT_RESOURCE_COUNT = 5
#: Microsecond threshold below which a Stream-1 op is considered fully overlapped.
OVERLAP_EXPOSED_EPS_US = 1e-3
#: Launch-bound diagnostic thresholds (us) and active-work floor ratio.
LAUNCH_BOUND_SHORT_US = 6.0
LAUNCH_BOUND_WORK_SHARE = 0.25
#: Maximum displayed layer indices in a compact archetype label before ellipsis.
MAX_EXPLICIT_LAYER_LABELS = 4
#: Graph digest hex truncation length.
GRAPH_DIGEST_HEX_LEN = 16
#: Tensor shape rank constants for contract formatting.
GEMM_SHAPE_RANK = 3
MATRIX_2D_RANK = 2
ATTN_4D_RANK = 4
ATTN_5D_RANK = 5

#: Strictly banned internal Python simulator recipe class names (never exported).
BANNED_IMPLEMENTATION_NAMES = frozenset({
    "AIVTiledVectorAndGatherRecipe",
    "SteadyMatmul",
    "ExpectedGroupedMatmul",
    "OperatorPlan",
    "TopologyCollectiveRecipe",
    "DirectPeerCollective",
    "ControlSyncRecipe",
    "TiledFlashAndPagedAttentionRecipe",
    "TiledSparseIndexerRecipe",
    "PhysicalRecipe",
    "Recipe",
})

#: Kernel backend / ecosystem taxonomy.
BACKEND_CANN = "CANN aclnn/AscendC"
BACKEND_CANNBOT_DSL = "cannbot-dsl"
BACKEND_TRITON = "Triton-Ascend"
BACKEND_TORCH_NPU = "torch_npu Custom Op"
BACKEND_HCCL_MC2 = "HCCL / MC2"
#: Normalized mapping from core ``structure.py`` ``backend_stack`` tokens.
STRUCTURE_BACKEND_MAP: dict[str, str] = {
    "aclnn": BACKEND_CANN,
    "ascendc": BACKEND_CANN,
    "cann": BACKEND_CANN,
    "cannbot_dsl": BACKEND_CANNBOT_DSL,
    "cannbot-dsl": BACKEND_CANNBOT_DSL,
    "triton": BACKEND_TRITON,
    "triton_ascend": BACKEND_TRITON,
    "triton-ascend": BACKEND_TRITON,
    "torch_npu": BACKEND_TORCH_NPU,
    "hccl": BACKEND_HCCL_MC2,
    "mc2": BACKEND_HCCL_MC2,
    "view_reshape": BACKEND_CANN,
}

#: Kernel readiness / development status labels.
KERNEL_STATUS_EXISTING = "现成可用 (Existing CANN/HCCL)"
KERNEL_STATUS_FRAMEWORK = "框架适配使能 (Framework Enablement)"
KERNEL_STATUS_NEW_OP = "[需新开发算子]"
KERNEL_STATUS_EXTEND_TILING = "[需扩展算子 Tiling]"
KERNEL_STATUS_NEW_OR_EXTEND = "[需新开发/扩展算子]"
#: Normalized mapping from core ``structure.py`` ``readiness`` tokens.
STRUCTURE_READINESS_MAP: dict[str, tuple[str, bool]] = {
    "existing": (KERNEL_STATUS_EXISTING, False),
    "framework_enable": (KERNEL_STATUS_FRAMEWORK, False),
    "new_kernel_required": (KERNEL_STATUS_NEW_OP, True),
    "extend_tiling": (KERNEL_STATUS_EXTEND_TILING, True),
    "new_or_extend": (KERNEL_STATUS_NEW_OR_EXTEND, True),
}

#: Pending kernel categories rendered in Sheet 4 (``Pending_Kernel_Details``).
PENDING_KIND_TO_IMPLEMENT = "待新开发 / 扩展 (To-Implement)"
PENDING_KIND_TO_OPTIMIZE = "重点待优化 (Top Gap To-Optimize)"

#: Canonical domain-standard Ascend submodule labels (zero "Other").
SUB_EMBED_TAIL = "Embedding / Logits / Spec Verify"
SUB_ATTN_RES = "AttnRes / Residual Norm"
SUB_MLA_PROLOG = "MLA Prolog (Q/KV Low-Rank & RoPE/NoPE)"
SUB_DSA_LI = "DSA Lightning Indexer (LI)"
SUB_LATENT_ATTN = "Latent Attention & DCP"
SUB_GATED_MLA_O = "Gated-MLA O-Proj & TP Comm"
SUB_KDA = "KDA Linear Attention"
SUB_GDN = "GDN Linear Attention"
SUB_STD_ATTN = "Standard GQA / SWA Attention"
SUB_MOE_ROUTER = "MoE Router & Dispatch"
SUB_MOE_ROUTED = "MoE Routed Experts"
SUB_MOE_SHARED = "MoE Shared Experts (MSO Stream 1)"
SUB_DENSE_FFN = "Dense FFN"
SUB_ENGRAM_PLE = "Engram / PLE Memory"
SUB_VISION = "Vision / Multimodal Encoder"
CANONICAL_EXPORT_SUBMODULES = frozenset({
    SUB_EMBED_TAIL,
    SUB_ATTN_RES,
    SUB_MLA_PROLOG,
    SUB_DSA_LI,
    SUB_LATENT_ATTN,
    SUB_GATED_MLA_O,
    SUB_KDA,
    SUB_GDN,
    SUB_STD_ATTN,
    SUB_MOE_ROUTER,
    SUB_MOE_ROUTED,
    SUB_MOE_SHARED,
    SUB_DENSE_FFN,
    SUB_ENGRAM_PLE,
    SUB_VISION,
})

#: Canonical architectural macro-modules for Sheet 1 Block 3.
MACRO_MOD_KDA_GDN = "KDA / GDN Linear Attention"
MACRO_MOD_MLA_PROLOG = "MLA Prolog (Q/KV Low-Rank & Absorb)"
MACRO_MOD_DSA_LI = "DSA / Sparse Indexer (LI)"
MACRO_MOD_LATENT_ATTN = "Latent / Core Attention & DCP"
MACRO_MOD_GATED_O_PROJ = "Gated-MLA / Attention O-Proj"
MACRO_MOD_MOE_ROUTER = "MoE Router & Dispatch/Combine"
MACRO_MOD_MOE_ROUTED = "MoE Routed Experts (GMM1 / Act / GMM2)"
MACRO_MOD_MOE_SHARED = "MoE Shared Experts (MSO Overlapped vs Exposed)"
MACRO_MOD_DENSE_NORM = "Dense FFN & AttnRes / Norm"
MACRO_MOD_COLLECTIVES = "Exposed TP/SP/DCP Collectives"
MACRO_MOD_SPEC_HEAD = "Speculative Draft (MTP/DSpark) & Verify / LM Head"
MACRO_MOD_ENGRAM = "Engram / PLE Memory"
MACRO_MOD_VISION = "Vision / Multimodal Encoder"

#: Display ordering of Sheet 1 Block 3 macro-modules (optional modules appended if present).
MACRO_MODULE_CORE_ORDER: tuple[str, ...] = (
    MACRO_MOD_KDA_GDN,
    MACRO_MOD_MLA_PROLOG,
    MACRO_MOD_DSA_LI,
    MACRO_MOD_LATENT_ATTN,
    MACRO_MOD_GATED_O_PROJ,
    MACRO_MOD_MOE_ROUTER,
    MACRO_MOD_MOE_ROUTED,
    MACRO_MOD_MOE_SHARED,
    MACRO_MOD_DENSE_NORM,
    MACRO_MOD_COLLECTIVES,
    MACRO_MOD_SPEC_HEAD,
)
MACRO_MODULE_OPTIONAL_ORDER: tuple[str, ...] = (
    MACRO_MOD_ENGRAM,
    MACRO_MOD_VISION,
)

#: Hardware + topology preset aliases for zero-boilerplate CLI/Python UX.
HARDWARE_PRESET_ALIASES: dict[str, tuple[str, str | None]] = {
    "950dt-96g": ("Ascend950DT_9582", None),
    "950dt-96g-pod": ("Ascend950DT_9582", "a5_pod"),
    "a5-96g-pod": ("Ascend950DT_9582", "a5_pod"),
    "a5-pod": ("Ascend950DT_9582", "a5_pod"),
    "950dt-144g": ("Ascend950DT_9581", None),
    "950dt-144g-pod": ("Ascend950DT_9581", "a5_pod"),
    "a5-144g-pod": ("Ascend950DT_9581", "a5_pod"),
    "a3-16p": ("Ascend910C_A3", "a3_16p"),
    "a3-pod": ("Ascend910C_A3", "a3_pod"),
    "a2-8p": ("Ascend910B4_A2", "a2_8p"),
}

#: Deployment mode canonical display labels.
DEPLOYMENT_MODE_LABELS: dict[str, str] = {
    "pd_unpooled": "PD 分离 [非池化]",
    "pd_pooled": "PD 分离 [池化]",
    "colocated": "混部 Colocated",
}

#: Closed-form storage specification matching post-TP local shapes.
WEIGHT_QUANT_FORMATS: dict[str, dict[str, float | int]] = {
    "BF16": {"bytes": 2.0, "group": 0, "scale": 0, "aux": 0},
    "FP32": {"bytes": 4.0, "group": 0, "scale": 0, "aux": 0},
    "FP8": {"bytes": 1.0, "group": 0, "scale": 0, "aux": 0},
    "W8A8": {"bytes": 1.0, "group": 0, "scale": 0, "aux": INT8_CHANNEL_AUX_BYTES},
    "W4A8": {"bytes": 0.5, "group": 0, "scale": 0, "aux": INT4_CHANNEL_AUX_BYTES},
    "MXFP8": {"bytes": 1.0, "group": MXFP_GROUP_SIZE, "scale": 1, "aux": 0},
    "MXFP4": {"bytes": 0.5, "group": MXFP_GROUP_SIZE, "scale": 1, "aux": 0},
    "INT8": {"bytes": 1.0, "group": 0, "scale": 0, "aux": INT8_CHANNEL_AUX_BYTES},
    "INT4": {"bytes": 0.5, "group": 0, "scale": 0, "aux": INT4_CHANNEL_AUX_BYTES},
}

SUPPORTED_WEIGHT_QUANTS = ("BF16", "W8A8", "W4A8", "MXFP8", "MXFP4")
SUPPORTED_CACHE_DTYPES = ("BF16", "C8", "FP8")

COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter", "alltoall",
    "alltoallv", "mc2", "mc2_transport", "mc2_protocol", "dispatch", "combine",
    "collective", "hccl", "p2p", "tp_allreduce", "engram_ids", "engram_broadcast",
})

BOUND_DISPLAY_LABELS: dict[str, str] = {
    "memory_bound": "HBM Weight/KV BW Bound",
    "cube_bound": "Cube AIC Compute Bound",
    "vector_bound": "Vector AIV Compute Bound",
    "link_bound": "Exposed HCCL Link Bound",
    "launch_bound": "Kernel Launch / Latency Bound",
}


@dataclass(frozen=True, slots=True)
class KernelObservationRow:
    """One unfused hardware kernel record parsed from ``msprof`` ``kernel_details.csv``."""

    index: int
    name: str
    op_type: str
    core: str
    start_us: float
    duration_us: float
    stream_id: str = ""
    task_id: str = ""
    input_shapes: str = ""
    input_dtypes: str = ""


@dataclass(frozen=True, slots=True)
class ReconciledKernelGroup:
    """1-to-N binding between one fused ``sim`` operator and ``N`` ``msprof`` kernels."""

    sim_op_id: str
    kernels: tuple[KernelObservationRow, ...]
    measured_sum_us: float | None
    ape_pct: float | None
    within_gate: bool | None


@dataclass(frozen=True, slots=True)
class LayerArchetypeSpec:
    """Canonical model scope + layer archetype + exact full-model repeat count."""

    archetype_id: str
    scope: str
    title: str
    short_label: str
    repeat_count: int
    layer_indices: tuple[int, ...]
    layer_range_text: str
    rep_layer_index: int | None
    single_span_us: float = 0.0


@dataclass(frozen=True, slots=True)
class TraceScheduleEntry:
    """One lightweight entry in the full-model streaming trace replay schedule."""

    archetype_id: str
    scope: str
    label: str
    full_layer_index: int | None


@dataclass(frozen=True, slots=True)
class AscendKernelDescriptor:
    """Human-native Ascend kernel identity, ecosystem backend, readiness & contract."""

    logical_role: str
    ascend_kernel: str
    kernel_backend: str
    kernel_status: str
    source_ref: str
    tensor_contract: str
    needs_new_kernel: bool
    new_kernel_spec: str
    canonical_op_key: str


@dataclass(frozen=True, slots=True)
class OperatorExportRow:
    """Complete physical, timing, Ascend kernel identity, and Lean ``L*`` row per op."""

    ordinal: int
    occurrence: OpOccurrence
    op_id: str
    layer: int | None
    layer_label: str
    submodule: str
    family: str
    recipe: str
    implementation: str
    stream: str
    stream_index: int
    task_type: str
    shape: tuple[int, ...]
    dtype: str
    predecessors: tuple[OpOccurrence, ...]
    start_us: float
    end_us: float
    dur_us: float
    flops: float
    vector_ops: float
    hbm_read_bytes: float
    hbm_write_bytes: float
    hbm_total_bytes: float
    mte2_bytes: float
    mte3_bytes: float
    fixpipe_bytes: float
    link_bytes: float
    l2_hit_pct: float
    dominant_bound: str
    l_star_lean_us: float
    gap_us: float
    gap_ratio: float | None
    gap_radar_rank: int
    roofline_efficiency: float | None
    absorbed_source_ops: tuple[str, ...]
    measured_kernels: tuple[KernelObservationRow, ...] = ()
    measured_sum_us: float | None = None
    ape_pct: float | None = None
    archetype_id: str = ""
    archetype_title: str = ""
    model_scope: str = "target"
    repeat_count: int = 1
    logical_role: str = ""
    ascend_kernel: str = ""
    kernel_backend: str = BACKEND_CANN
    kernel_status: str = KERNEL_STATUS_EXISTING
    source_ref: str = ""
    tensor_contract: str = ""
    is_overlapped: bool = False
    exposed_dur_us: float = 0.0
    cube_peak_flops_per_s: float = 0.0
    bound_display: str = ""
    needs_new_kernel: bool = False
    new_kernel_spec: str = ""
    canonical_op_key: str = ""


@dataclass(frozen=True, slots=True)
class AlignedComparisonRow:
    """Aligned Before <-> After operator group (1->1, N->1 fusion, 1->N, added/eliminated)."""

    archetype_id: str
    archetype_title: str
    repeat_count: int
    submodule: str
    before_ops: tuple[OperatorExportRow, ...]
    after_ops: tuple[OperatorExportRow, ...]
    relation: str
    action_note: str


@dataclass(frozen=True, slots=True)
class MacroModuleRow:
    """Whole-network exposed duration & share % aggregated by architectural macro-module."""

    module_name: str
    layer_calls_label: str
    layer_count: int
    before_ms: float
    before_share_pct: float
    after_ms: float
    after_share_pct: float
    saved_ms: float
    lstar_ms: float
    dominant_bound: str
    raw_before_ms: float = 0.0
    raw_after_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class NewKernelBacklogItem:
    """Pre-filled hardware/contract spec + open Agent design slot for a pending kernel."""

    priority: int
    kernel_key: str
    target_kernel: str
    suggested_backend: str
    status: str
    replaces_chain: str
    source_refs: str
    archetype_summary: str
    total_calls_per_step: int
    expected_gain_ms: float
    tensor_contract: str
    microarch_target: str
    default_design_slot: str
    default_acceptance_slot: str
    category: str = PENDING_KIND_TO_IMPLEMENT
    logical_role: str = ""
    submodule: str = ""
    before_bound: str = ""
    after_bound: str = ""
    before_model_ms: float = 0.0
    after_model_ms: float = 0.0
    lstar_model_ms: float = 0.0
    gap_vs_lstar_ms: float = 0.0


#: Canonical alias used by Sheet 4 (``Pending_Kernel_Details``).
PendingKernelItem = NewKernelBacklogItem


@dataclass(frozen=True, slots=True)
class GapRadarRow:
    """Ranked Physical-Limit Gap Radar entry ordered by model-scaled ``T_pred - L*``."""

    rank: int
    layer_label: str
    submodule: str
    op_id: str
    family: str
    implementation: str
    dominant_bound: str
    l_star_lean_us: float
    pred_us: float
    gap_us: float
    gap_ratio: float | None
    share_of_step_pct: float
    repeat_count: int = 1
    logical_role: str = ""
    kernel_status: str = ""


@dataclass(frozen=True, slots=True)
class WeightLedgerRow:
    """One structural tensor family in the closed-form per-die Weight/Quant table."""

    group: str
    name: str
    category: str
    global_rows: int
    global_cols: int
    shard_rule: str
    shard_axis: int
    local_rows: int
    local_cols: int
    stage_repeats: tuple[int, ...]
    max_stage_repeats: int
    quant_format: str
    payload_bytes: int
    scale_bytes: int
    aux_bytes: int
    per_die_bytes: int
    is_engram_or_ple: bool
    is_quant_excluded: bool = False
    quant_note: str = ""


@dataclass(frozen=True, slots=True)
class CacheLedgerRow:
    """One closed-form cache/state component in the Memory/Quant calculator."""

    component: str
    cache_kind: str
    active_layers_per_stage: tuple[int, ...]
    max_stage_layers: int
    bytes_per_token_per_layer: float
    per_request_bytes: float
    batch_total_bytes: float
    scales_with_tokens: bool


@dataclass(frozen=True, slots=True)
class StageBudgetRow:
    """Per-PP-stage HBM & Host DRAM memory budget summary."""

    stage: int
    owned_layers: int
    weight_payload_bytes: float
    weight_scale_aux_bytes: float
    engram_ple_hbm_bytes: float
    engram_ple_host_bytes: float
    cache_bytes: float
    workspace_peak_bytes: float
    reserve_bytes: float
    total_hbm_bytes: float
    usable_hbm_budget_bytes: float
    headroom_bytes: float
    fits_hbm: bool


@dataclass(frozen=True, slots=True)
class MemoryCalculatorLedger:
    """Closed-form Memory / Quant / Cache / Engram-PLE / PP HBM budget ledger."""

    model_name: str
    hardware_sku: str
    phase: str
    weight_quant: str
    cache_dtype: str
    engram_host_offload: bool
    dp: int
    tp: int
    ep: int
    pp: int
    dcp: int
    pcp: int
    local_tp: int
    shared_tp: int
    total_dies: int
    seq_len: int
    concurrency: int
    block_size: int
    spec_tokens: int
    local_cache_tokens: int
    die_hbm_capacity_bytes: float
    gpu_memory_utilization: float
    usable_hbm_bytes: float
    weight_rows: tuple[WeightLedgerRow, ...]
    cache_rows: tuple[CacheLedgerRow, ...]
    stage_budgets: tuple[StageBudgetRow, ...]
    max_stage_hbm_bytes: float
    max_stage_host_offload_bytes: float
    fits_hbm: bool
    sim_memory_weight_bytes_by_stage: tuple[float, ...]
    sim_memory_cache_bytes_by_stage: tuple[float, ...]
    weight_quant_display: str = ""
    quant_exclusions: tuple[str, ...] = ()
    weight_footprint_per_die_bytes: float = 0.0
    per_request_cache_bytes_max_stage: float = 0.0
    max_hbm_concurrency_per_dp: int = 0


@dataclass(frozen=True, slots=True)
class ReconciliationSummary:
    """Summary of 1-to-N reconciliation between ``sim`` ops and ``msprof`` kernels."""

    source_label: str
    total_sim_ops: int
    matched_sim_ops: int
    total_kernels: int
    matched_kernels: int
    unmatched_kernels: tuple[KernelObservationRow, ...]
    matched_sim_sum_us: float
    matched_measured_sum_us: float
    unmatched_measured_us: float
    median_ape_pct: float | None
    p90_ape_pct: float | None
    within_gate_count: int
    within_gate_pct: float | None


@dataclass(frozen=True, slots=True)
class ExportBundle:
    """Single O(Unique Archetypes) source of truth for XLSX and streaming Trace exports."""

    schema: str
    model_name: str
    hardware_sku: str
    phase: str
    scope: str
    graph_id: str
    tokens_per_seq: int
    context_tokens: int
    sequences: int
    step_span_us: float
    operator_sum_us: float
    step_l_star_lean_us: float
    step_certificate_valid: bool
    step_tight_resources: tuple[str, ...]
    throughput_tokens_per_s: float
    throughput_per_die_tokens_per_s: float
    network_cube_mfu: float
    network_vec_mfu: float
    network_mbu: float
    network_link_util: float
    network_roofline_efficiency: float
    exposed_breakdown_us: Mapping[str, float]
    bottleneck_mix: Mapping[str, Mapping[str, float]]
    hardware_peaks: Mapping[str, float]
    operators: tuple[OperatorExportRow, ...]
    pipelines: tuple[FusedPipelineDescriptor, ...]
    gap_radar: tuple[GapRadarRow, ...]
    memory_ledger: MemoryCalculatorLedger
    reconciliation: ReconciliationSummary | None = None
    archetypes: tuple[LayerArchetypeSpec, ...] = ()
    full_trace_schedule: tuple[TraceScheduleEntry, ...] = ()
    total_target_layers: int = 0
    total_draft_layers: int = 0
    graph_capture_mode: str = "full"
    feature_track: str = ""
    hardware_die_summary: str = ""
    topology_preset: str = ""
    deployment_mode: str = ""
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS
    tpot_slo_ms: float = DEFAULT_TPOT_SLO_MS
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS
    spec_method: str = ""
    spec_proposals: int = 0
    avg_accepted_tokens: float = 1.0
    sequence_parallel: bool = False
