"""Authoritative simulation data extraction, archetype analysis, and memory ledger.

[Role] Build a single presentation-agnostic, O(Unique Archetypes) ``ExportBundle``
  from ``mindie_sim`` ``Case`` / ``PredictionSpine`` evaluations, classify every
  operator into domain-accurate Ascend submodules (zero ``Other``), map every
  operator to real Ascend/CANN/cannbot-dsl/Triton/HCCL kernels + readiness status
  + source references + rank-local tensor contracts, align Before-vs-After fusion
  groups (N->1 / 1->N), construct the New-Kernel Design Backlog, reconcile 1-to-N
  ``msprof`` ``kernel_details.csv`` observations, and compute the closed-form
  per-die Memory/Quant/PP ledger honoring ``QuantSpec.exclusions``.
[Invariants] Zero Excel/OOXML or Chrome Trace serialization lives here. Zero
  internal Python recipe class names (``SteadyMatmul``, ``OperatorPlan``,
  ``AIVTiledVectorAndGatherRecipe``, etc.) escape into exported rows. Zero
  ``0.80x`` heuristic fudge factors or silent hardware defaults. Whole-step
  ``L*`` is verified via ``check_optimality_certificate``. Every measured kernel
  is assigned at most once. Full multi-layer models evaluate ONLY unique layer
  archetypes once (O(Archetypes) RAM) while conserving exact full-model repeat
  counts and layer schedules.
[Timeouts & Locks] Single-pass bounded evaluation with streaming CSV ingress;
  honors ``ReplayOptions.deadline_mono``; no locks or device contexts.
[Side-Effect Policy] Read-only over caller cases and observation files; never
  mutates inputs or writes files.
"""
from __future__ import annotations

from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass, replace
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import median
from typing import Any

from mindie_sim.catalog.cache import (
    DEFAULT_CACHE_BLOCK_SIZE,
    INDEX_INTERLEAVE_TOKENS,
    layer_cache_components,
)
from mindie_sim.catalog.types import QuantSpec
from mindie_sim.catalog.weight_terms import WeightTerm, _sharded_shape
from mindie_sim.catalog.weights import engram_table_terms, stage_weight_terms
from mindie_sim.optimization.dual_bound import (
    NS_PER_SECOND,
    DagEdge,
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
    inward_floor_ns,
)
from mindie_sim.prediction.body_defaults import consume_target_declaration
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import extract_hardware_peaks
from mindie_sim.prediction.evaluation import EvaluationContext
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.results import (
    CaseInput,
    FusedPipelineDescriptor,
    OpOccurrence,
    ReplayOptions,
    StepEvaluation,
    StepRequest,
)
from mindie_sim.prediction.rewrite import ReplayRecipeBook
from mindie_sim.prediction.spine import default_spine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest
from mindie_sim.structure_contracts import predicted_contract
from mindie_sim.structure_implementations import native_implementation

#: Export bundle schema version.
EXPORT_BUNDLE_SCHEMA = "mindie-sim.export-bundle/2"
#: Binary GiB divisor in bytes.
GIB_BYTES = 1024 * 1024 * 1024
#: Binary MiB divisor in bytes.
MIB_BYTES = 1024 * 1024
#: Maximum read size for structured observation JSON manifests.
MAX_OBSERVATION_JSON_BYTES = 50 * MIB_BYTES
#: Microseconds per millisecond and nanoseconds per microsecond.
US_PER_MS = 1000.0
NS_PER_US = 1000.0
#: Default number of top gap-radar entries surfaced in executive summaries.
DEFAULT_GAP_RADAR_TOP_K = 20
#: Absolute tolerance floor (us) used alongside 10% APE pass-rate reporting.
RECONCILE_ABS_FLOOR_US = 2.0
#: Relative APE threshold (%) for 1-to-N measured reconciliation summary.
RECONCILE_APE_THRESHOLD_PCT = 10.0
#: Quantile constants for measured APE summary statistics.
P50_QUANTILE = 0.5
P90_QUANTILE = 0.9
#: Minimum semantic token length for automated kernel_details.csv matching.
MIN_MATCH_TOKEN_LEN = 2
#: Default context fallback only when caller passes neither request nor step_eval.
DEFAULT_EXPORT_CONTEXT_TOKENS = 1024
#: Minimum default batched-token floor for decode scenario auto-composition.
DEFAULT_MIN_BATCHED_TOKENS = 768
#: Minimum default model length headroom added to prompt + output tokens.
DEFAULT_MODEL_LEN_HEADROOM = 1024
#: Minimum default model length floor.
DEFAULT_MIN_MODEL_LEN = 4096
#: Default prompt tokens and output tokens for one-shot scenario export.
DEFAULT_SCENARIO_PROMPT_TOKENS = 4096
DEFAULT_SCENARIO_OUTPUT_TOKENS = 128
#: Default decode/prefill SLO budgets (ms) for zero-boilerplate scenario export.
DEFAULT_TPOT_SLO_MS = 50.0
DEFAULT_TTFT_SLO_MS = 5000.0
#: Default sparse DSA Lightning Indexer top-k fallback when unspecified on layer plan.
DEFAULT_SPARSE_INDEX_TOPK = 2048
#: Fallback sort key for any unrecognized archetype ordering.
ARCHETYPE_ORDER_FALLBACK = 999
#: Microseconds per second, bytes per KiB/TB, and denominator floor constants.
US_PER_SECOND = 1_000_000.0
KIB_BYTES = 1024
TB_BYTES = 1e12
MIN_POSITIVE_SECONDS = 1e-12
MIN_DENOM_EPS = 1e-9
#: Maximum character length for compact concatenated source-ref strings in Sheet 4.
MAX_SOURCE_REF_SUMMARY_LEN = 160
#: Closed-form MXFP microscaling block width along the local K dimension.
MXFP_GROUP_SIZE = 32
#: Closed-form KV quantization scale group width (elements per FP32 scale).
KV_QUANT_GROUP_SIZE = 128
#: Element width constants in bytes.
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

#: Kernel backend / ecosystem taxonomy (Item 9).
BACKEND_CANN = "CANN aclnn/AscendC"
BACKEND_CANNBOT_DSL = "cannbot-dsl"
BACKEND_TRITON = "Triton-Ascend"
BACKEND_TORCH_NPU = "torch_npu Custom Op"
BACKEND_HCCL_MC2 = "HCCL / MC2"
#: Normalized mapping from core ``structure.py`` ``backend_stack`` tokens.
_STRUCTURE_BACKEND_MAP: dict[str, str] = {
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

#: Kernel readiness / development status labels (Item 7).
KERNEL_STATUS_EXISTING = "现成可用 (Existing CANN/HCCL)"
KERNEL_STATUS_FRAMEWORK = "框架适配使能 (Framework Enablement)"
KERNEL_STATUS_NEW_OP = "[需新开发算子]"
KERNEL_STATUS_EXTEND_TILING = "[需扩展算子 Tiling]"
KERNEL_STATUS_NEW_OR_EXTEND = "[需新开发/扩展算子]"
#: Normalized mapping from core ``structure.py`` ``readiness`` tokens.
_STRUCTURE_READINESS_MAP: dict[str, tuple[str, bool]] = {
    "existing": (KERNEL_STATUS_EXISTING, False),
    "framework_enable": (KERNEL_STATUS_FRAMEWORK, False),
    "new_kernel_required": (KERNEL_STATUS_NEW_OP, True),
    "extend_tiling": (KERNEL_STATUS_EXTEND_TILING, True),
    "new_or_extend": (KERNEL_STATUS_NEW_OR_EXTEND, True),
}

#: Canonical domain-standard Ascend submodule labels (Item 2 — zero "Other").
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

#: Hardware + topology preset aliases for zero-boilerplate CLI/Python UX (Item 8).
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

_LAYER_PATTERN = re.compile(r"(?:^|\.)l(\d+)(?:\.|$)")
_CONTEXT_KV_PATTERN = re.compile(r"context_kv(\d+)(?:\.|$)")
_RECURRENT_MICRO_PATTERN = re.compile(r"^(.*)\.(recurrent_scan|chunk_kda)(\d+)$")
_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "reduce_scatter", "alltoall",
    "alltoallv", "mc2", "mc2_transport", "mc2_protocol", "dispatch", "combine",
    "collective", "hccl", "p2p", "tp_allreduce", "engram_ids", "engram_broadcast",
})
_CSV_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "name": ("name", "kernel_name", "op name", "op_name"),
    "op_type": ("type", "op type", "op_type", "kernel_type"),
    "core": ("accelerator core", "core", "kernel_type", "task type"),
    "start_us": ("start time(us)", "task start time(us)", "task_start(us)", "start_us"),
    "duration_us": ("duration(us)", "task duration(us)", "task_time(us)", "duration_us"),
    "stream_id": ("stream id", "stream_id"),
    "task_id": ("task id", "task_id"),
    "input_shapes": ("input shapes", "input_shapes"),
    "input_dtypes": ("input data types", "input_dtypes"),
}
_ENGRAM_CATEGORIES = frozenset({"engram_table", "ple_table", "gemma4_ple"})
_FUSED_KERNEL_HINTS: dict[str, tuple[str, ...]] = {
    "qkv_projection": ("qkv", "q_proj", "k_proj", "v_proj", "matmul", "mm"),
    "add_rmsnorm_quant": ("add", "rmsnorm", "rms_norm", "quant", "dynamicquant"),
    "add_rmsnorm": ("add", "rmsnorm", "rms_norm"),
    "mla_prolog": ("mla", "prolog", "rope", "rotary", "quant", "matmul"),
    "gate_up_projection": ("gate", "up", "matmul", "mm"),
    "grouped_gemm": ("grouped", "gmm", "moe", "matmul"),
    "attention": ("flashattention", "mla", "attention", "fia", "ifa", "pfa", "sfa"),
    "allreduce": ("allreduce", "hccl", "hcom"),
    "allgather": ("allgather", "hccl", "hcom"),
    "reduce_scatter": ("reducescatter", "hccl", "hcom"),
    "reducescatter": ("reducescatter", "hccl", "hcom"),
    "alltoall": ("alltoall", "dispatch", "combine", "hccl", "hcom"),
    "rmsnorm": ("rmsnorm", "rms_norm", "norm"),
    "rotary": ("rotary", "rope"),
    "swiglu": ("swiglu", "silu", "mul"),
    "quantize": ("quant", "dynamicquant"),
}
_WEIGHT_QUANT_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "w8a8": "W8A8",
    "int8": "INT8",
    "w4a8": "W4A8",
    "int4": "INT4",
    "mxfp8": "MXFP8",
    "w8a8_mxfp8": "MXFP8",
    "w8a8-mxfp8": "MXFP8",
    "mxfp4": "MXFP4",
    "mxfp4_routed_only": "MXFP4",
    "w4a8_mxfp": "MXFP4",
    "w4a8-mxfp": "MXFP4",
    "w4a8_mxfp4": "MXFP4",
    "w4a8-mxfp4": "MXFP4",
}
_CACHE_DTYPE_ALIASES: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "c8": "C8",
    "int8": "C8",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
}
_QUANT_FORMAT_DISPLAY: dict[str, str] = {
    "bf16": "BF16",
    "bfloat16": "BF16",
    "fp16": "BF16",
    "float16": "BF16",
    "fp32": "FP32",
    "float32": "FP32",
    "fp8": "FP8",
    "fp8_e4m3": "FP8",
    "fp8_e5m2": "FP8",
    "int8": "W8A8",
    "w8a8": "W8A8",
    "int4": "W4A8",
    "w4a8": "W4A8",
    "mxfp8": "MXFP8",
    "mx_fp8": "MXFP8",
    "mxfp4": "MXFP4",
    "mx_fp4": "MXFP4",
    "fp4": "MXFP4",
    "fp4_e2m1": "MXFP4",
}
#: Known N->1 fusion membership rules when Before/After graphs fuse ops across tracks.
_KNOWN_FUSION_SOURCES: dict[str, tuple[str, ...]] = {
    "mla_prolog_v3": (
        "q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
        "q_norm", "q_b", "q_b.quantize", "kv_norm", "rotary", "q_rotary",
        "kv_rotary", "kv_store", "q_absorb",
    ),
    "mla_prolog": (
        "q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
        "q_norm", "q_b", "q_b.quantize", "kv_norm", "rotary", "q_rotary",
        "kv_rotary", "kv_store", "q_absorb",
    ),
    "megamoe_arch35": (
        "dispatch", "dispatch.received_quantize", "expert_up", "expert_act",
        "expert_down.quantize", "expert_down", "unpack", "join",
    ),
    "dispatch_ffn_combine": (
        "dispatch", "dispatch.received_quantize", "expert_up", "expert_act",
        "expert_down.quantize", "expert_down", "unpack", "join",
    ),
    "grouped_matmul_situ_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "grouped_matmul_swiglu_quant": ("expert_up", "expert_act", "expert_down.quantize"),
    "quant_lightning_indexer": ("index_query_quantize", "index_scores", "index_topk"),
    "matmul_reduce_scatter": ("output", "output_reduce", "output_reduce.reduce_scatter", "down", "tp", "tp.reduce_scatter", "linear_output", "linear_reduce.reduce_scatter"),
    "all_gather_matmul": ("attention.sp_gather", "ffn.sp_gather", "shared.sp_gather"),
    "add_rmsnorm": ("residual", "norm1", "norm2", "final_norm"),
    "add_rmsnorm_quant": ("residual", "norm1", "norm2", "quantize"),
    "recurrent_scan": ("qk_l2norm", "decode_q_l2norm", "decode_k_l2norm", "delta_gates", "recurrent_scan"),
}
_BOUND_DISPLAY_LABELS: dict[str, str] = {
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
class NewKernelBacklogItem:
    """Pre-filled hardware/contract spec + open Agent design slot for a required new kernel."""

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


def resolve_hardware_alias(
    hardware: str, topology_preset: str | None = None,
) -> tuple[str, str | None]:
    """Expand intuitive CLI hardware/pod aliases into ``(sku, topology_preset)``."""
    key = str(hardware or "").strip().lower()
    if key in HARDWARE_PRESET_ALIASES:
        sku, preset = HARDWARE_PRESET_ALIASES[key]
        return sku, (topology_preset if topology_preset is not None else preset)
    return str(hardware).strip(), topology_preset


def compose_export_case(
    model: str,
    hardware: str,
    *,
    topology_preset: str | None = None,
    dp: int = 1,
    tp: int = 1,
    dcp: int = 1,
    ep: int = 1,
    pp: int = 1,
    pcp: int = 1,
    prompt_tokens: int = DEFAULT_SCENARIO_PROMPT_TOKENS,
    output_tokens: int = DEFAULT_SCENARIO_OUTPUT_TOKENS,
    concurrency: int = 1,
    tpot_slo_ms: float = DEFAULT_TPOT_SLO_MS,
    ttft_slo_ms: float = DEFAULT_TTFT_SLO_MS,
    mtp_proposal_tokens: int = 0,
    mtp_acceptance_length: float = 1.0,
    feature_track: str = "merged",
    quant: str | None = None,
    kv_cache_dtype: str | None = None,
    engram_cpu_offload: bool | None = None,
) -> tuple[dict[str, Any], StepRequest]:
    """Zero-boilerplate scenario composer returning ``(case_dict, StepRequest)``."""
    from mindie_sim.compose import compose_case

    sku, resolved_preset = resolve_hardware_alias(hardware, topology_preset)
    spec_toks = max(0, int(mtp_proposal_tokens))
    conc = max(1, int(concurrency))
    p_tok = max(1, int(prompt_tokens))
    o_tok = max(1, int(output_tokens))
    min_batched = max(conc * (1 + spec_toks), DEFAULT_MIN_BATCHED_TOKENS)
    model_len = max(p_tok + o_tok + DEFAULT_MODEL_LEN_HEADROOM, DEFAULT_MIN_MODEL_LEN)
    compose_kw: dict[str, Any] = {
        "tp": max(1, int(tp)),
        "dp": max(1, int(dp)),
        "ep": max(1, int(ep)),
        "pp": max(1, int(pp)),
        "dcp": max(1, int(dcp)),
        "pcp": max(1, int(pcp)),
        "prompt_tokens": p_tok,
        "output_tokens": o_tok,
        "concurrency": conc,
        "max_num_seqs": max(conc, 1),
        "max_num_batched_tokens": min_batched,
        "max_model_len": model_len,
        "mtp_proposal_tokens": spec_toks,
        "mtp_acceptance_length": float(mtp_acceptance_length),
        "tpot_slo_ms": float(tpot_slo_ms),
        "ttft_slo_ms": float(ttft_slo_ms),
        "feature_track": str(feature_track or "merged"),
    }
    if resolved_preset:
        compose_kw["topology_preset"] = resolved_preset
    if quant is not None:
        compose_kw["quant"] = quant
    if kv_cache_dtype is not None:
        compose_kw["kv_cache_dtype"] = kv_cache_dtype
    if engram_cpu_offload is not None:
        compose_kw["engram_cpu_offload"] = bool(engram_cpu_offload)
    raw = compose_case(model, sku, **compose_kw)
    step_req = StepRequest(
        phase="decode",
        tokens=1 + spec_toks,
        context=p_tok,
        sequences=conc,
        scope="model",
    )
    return raw, step_req


def evaluate_weight_storage_bytes(
    local_rows: int, local_cols: int, quant_format: str,
) -> tuple[int, int, int, int]:
    """Closed-form ``(payload, scale, aux, total)`` bytes for one post-TP local matrix."""
    key = str(quant_format or "BF16").strip().upper()
    spec = WEIGHT_QUANT_FORMATS.get(key)
    if spec is None:
        raise ModelingError(f"unsupported_weight_quant_format:{quant_format}")
    rows = int(local_rows)
    cols = int(local_cols)
    if rows < 1 or cols < 1:
        raise ModelingError(f"invalid_local_weight_shape:({rows},{cols})")
    payload = rows * math.ceil(cols * float(spec["bytes"]))
    group = int(spec["group"])
    scale = (
        rows * ((cols + group - 1) // group) * int(spec["scale"])
        if group > 0 else 0
    )
    aux = rows * int(spec["aux"]) if int(spec["aux"]) > 0 else 0
    return payload, scale, aux, payload + scale + aux


def _normalize_quant_preset(quant: QuantSpec | None, fallback: str = "BF16") -> str:
    """Resolve base quant preset directly from ``QuantSpec`` structural fields."""
    if quant is None:
        return fallback
    name = str(getattr(quant, "name", "") or "").strip().lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "") or "").strip().lower()
    w_fmt = str(getattr(quant, "weight_format", "") or "").strip().lower()
    combined = f"{name} {exp_fmt} {w_fmt}"
    if "mxfp4" in combined or "mx_fp4" in combined:
        return "MXFP4"
    if "mxfp8" in combined or "mx_fp8" in combined:
        return "MXFP8"
    if "w4a8" in combined or "int4" in combined or "fp4" in combined:
        return "W4A8"
    if "w8a8" in combined or "int8" in combined:
        return "W8A8"
    if "fp8" in combined:
        return "FP8"
    return "BF16"


def _describe_quant_display(
    quant: QuantSpec | None, resolved_wq: str, *, explicit_override: bool,
) -> tuple[str, tuple[str, ...]]:
    """Build human-readable quant policy banner respecting ``QuantSpec.exclusions``."""
    exclusions = tuple(getattr(quant, "exclusions", ()) or ()) if quant is not None else ()
    if quant is None or not exclusions:
        return resolved_wq, ()
    excl_set = set(exclusions)
    routed_only = (
        resolved_wq == "MXFP4"
        and ({"attn", "mla", "kda"} & excl_set)
        and "shared_expert" in excl_set
    )
    if routed_only:
        return "MXFP4 (Routed-Only, Attn/Shared BF16)", exclusions
    w_fmt = str(getattr(quant, "weight_format", "bf16") or "bf16").lower()
    exp_fmt = str(getattr(quant, "expert_weight_format", "bf16") or "bf16").lower()
    if not explicit_override and w_fmt != exp_fmt and "fp8" in w_fmt and "mxfp4" in exp_fmt:
        return "FP8 Linear + MXFP4 Routed Experts (Router/Norm/Head BF16)", exclusions
    return f"{resolved_wq} (Excl: {','.join(exclusions[:MAX_EXPLICIT_LAYER_LABELS])})", exclusions


def _canonicalize_weight_quant_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _WEIGHT_QUANT_ALIASES.get(token, str(raw).strip().upper())


def _canonicalize_cache_dtype_override(raw: str) -> str:
    token = str(raw).strip().lower()
    return _CACHE_DTYPE_ALIASES.get(token, str(raw).strip().upper())


def _normalize_cache_dtype(model: Any) -> str:
    quant = getattr(model, "quant", None)
    raw = str(
        (getattr(quant, "kv_dtype", "") if quant is not None else "")
        or getattr(model, "kv_dtype", "")
        or "bf16"
    ).strip().lower()
    if "int8" in raw or raw == "c8":
        return "C8"
    if "fp8" in raw:
        return "FP8"
    return "BF16"


def extract_layer_index(op_id: str, op_layer: int | None = None) -> int | None:
    """Resolve integer decoder layer index from ``OpSpec.layer`` or ``op_id``."""
    if op_layer is not None and op_layer >= 0:
        return int(op_layer)
    match = _LAYER_PATTERN.search(op_id)
    if match is not None:
        return int(match.group(1))
    ctx_match = _CONTEXT_KV_PATTERN.search(op_id)
    return int(ctx_match.group(1)) if ctx_match is not None else None


def _strip_op_prefix(op_id: str) -> str:
    """Return layer-relative suffix (e.g. ``s0.l3.index_q`` -> ``index_q``)."""
    text = str(op_id or "")
    text = re.sub(r"^draft\d+\.", "", text)
    text = re.sub(r"^context_kv\d+\.", "", text)
    text = re.sub(r"^s\d+\.l\d+\.", "", text)
    text = re.sub(r"^l\d+\.", "", text)
    return text


def classify_submodule(
    op_id: str,
    family: str,
    recipe: str = "",
    *,
    layer_attention: str = "",
    layer_ffn: str = "",
) -> str:
    """Classify an operator into a domain-standard Ascend stage (ZERO ``Other``!)."""
    oid = str(op_id or "").lower()
    suffix = _strip_op_prefix(op_id).lower()
    fam = str(family or "").strip().lower()
    rec = str(recipe or "").strip().lower()
    attn = str(layer_attention or "").strip().lower()
    ffn = str(layer_ffn or "").strip().lower()

    if _is_embed_or_tail_op(oid, suffix, fam, rec):
        return SUB_EMBED_TAIL
    if any(tok in suffix or tok in rec for tok in ("vision", "aligner", "projector")):
        return SUB_VISION
    if any(tok in suffix or tok in fam or tok in rec for tok in ("engram", "ple_")):
        return SUB_ENGRAM_PLE
    if _is_shared_expert_op(suffix, rec):
        return SUB_MOE_SHARED
    if _is_attn_res_or_norm_op(suffix, fam, rec):
        return SUB_ATTN_RES
    if _is_dsa_indexer_op(suffix, fam, rec):
        return SUB_DSA_LI
    if _is_mla_prolog_op(suffix, fam, rec, attn):
        return SUB_MLA_PROLOG
    if _is_latent_attn_op(suffix, fam, rec, attn):
        return SUB_LATENT_ATTN
    if _is_gated_mla_output_op(suffix, fam, rec, attn):
        return SUB_GATED_MLA_O
    if _is_linear_attn_op(suffix, fam, rec, attn):
        return SUB_GDN if _is_gdn_op(suffix, fam, rec, attn) else SUB_KDA
    if _is_moe_router_dispatch_op(suffix, fam, rec):
        return SUB_MOE_ROUTER
    if _is_moe_routed_op(suffix, fam, rec):
        return SUB_MOE_ROUTED
    if _is_dense_ffn_op(suffix, fam, rec, ffn):
        return SUB_DENSE_FFN
    if _is_std_attn_op(suffix, fam, rec, attn):
        return SUB_STD_ATTN
    if "ffn" in suffix:
        return SUB_MOE_ROUTED if ffn == "moe" else SUB_DENSE_FFN
    if attn == "kda":
        return SUB_KDA
    if attn == "gdn":
        return SUB_GDN
    if attn in {"mla", "mla_dsa", "dsa"}:
        return SUB_LATENT_ATTN
    return SUB_ATTN_RES


def _is_embed_or_tail_op(oid: str, suffix: str, fam: str, rec: str) -> bool:
    if fam in {"embedding", "logits", "acceptance"}:
        return True
    if oid.startswith("final.") or suffix.startswith(("final_", "logits", "embedding")):
        return True
    if suffix in {"sp_shard", "sp_output_gather", "accept", "features", "features_gather", "context_norm",
                  "query_embedding", "markov_lookup", "markov", "markov_bias"}:
        return True
    return any(tok in rec for tok in ("draft_features", "draft_context_norm", "draft_query_embedding",
                                      "markov", "vocab_logits", "speculative_acceptance", "hc_lowrank_head"))


def _is_shared_expert_op(suffix: str, rec: str) -> bool:
    return suffix.startswith("shared.") or suffix.startswith("shared_gate") or "shared_gate" in rec


def _is_attn_res_or_norm_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"attn_res_prefix", "mhc"} or "attn_res" in suffix or ".hc_" in f".{suffix}":
        return True
    if suffix in {"norm1", "norm2", "input_layernorm", "post_attention_layernorm",
                  "attention.residual", "ffn.residual", "residual"}:
        return True
    return rec in {"rmsnorm", "add_rmsnorm", "residual_add", "grouped_gemma_rmsnorm", "hc_silu",
                   "hc_values_fp32", "hc_projection", "hc_lowrank_down_inject", "hc_lowrank_up"}


def _is_dsa_indexer_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"indexer", "index_projection", "key_pool", "indexer_quant_cache"}:
        return True
    if suffix.startswith(("index_", "candidate_", "cann_keypool", "cann_key_pool", "cann_pool_key")):
        return True
    return any(tok in rec for tok in ("lightning_indexer", "index_scores", "index_topk",
                                      "candidate_", "keypool", "key_pool", "csa_summary"))


def _is_mla_prolog_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if "mla_prolog" in suffix or "mla_prolog" in rec:
        return True
    if suffix in {"q_a", "q_a.quantize", "kv_a", "kv_a.quantize", "qkv_a", "qkv_a.quantize",
                  "q_norm", "q_b", "q_b.quantize", "kv_norm", "q_rotary", "kv_rotary",
                  "kv_store", "kv_scatter", "q_absorb"}:
        return True
    if suffix == "rotary" and attn in {"mla", "mla_dsa", "dsa", ""}:
        return True
    return suffix == "attention.sp_gather" and attn in {"mla", "mla_dsa", "dsa"}


def _is_latent_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if suffix in {"dcp_query_pack", "dcp_query_allgather", "latent_attention_batch",
                  "mqsmla", "dcp_output_pack", "dcp_output_alltoall", "dcp_lse_merge",
                  "v_expand", "output_inverse_rotary"}:
        return True
    return (fam in {"attention", "attention_merge"} and attn in {"mla", "mla_dsa", "dsa"})


def _is_gated_mla_output_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if suffix in {"gated_output", "grouped_output_a", "grouped_output_a.quantize",
                  "grouped_output_b", "grouped_output_b.quantize"}:
        return True
    if attn in {"mla", "mla_dsa", "dsa"} and (
        suffix in {
            "output_gate", "output", "output.quantize", "output_reduce",
            "output_reduce.reduce_scatter", "output_reduce.reduce_scatter.pad",
            "tp_attn", "tp_attn.reduce_scatter",
        }
        or suffix.startswith("output_reduce.")
    ):
        return True
    return False


def _is_gdn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if attn == "gdn":
        return True
    return any(tok in f"{suffix} {fam} {rec}" for tok in (
        "gdn", "linear_qkvz", "linear_beta_alpha", "decode_recurrent_scan",
        "alpha_contiguous", "beta_contiguous", "z_norm_materialize",
    ))


def _is_linear_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if attn in {"kda", "gdn"} and suffix == "attention.sp_gather":
        return True
    if fam in {"recurrent_scan", "recurrent_vector", "gdn_gating", "gdn_prepare",
               "gdn_state", "gdn_output", "short_convolution", "causal_conv1d_fn"}:
        return True
    return (
        suffix.startswith(("linear_", "recurrent_", "chunk_kda", "recover_ssm", "short_conv",
                           "decode_q_l2norm", "decode_k_l2norm", "decode_recurrent", "delta_gates",
                           "qk_l2norm", "core_output_", "beta_contiguous", "alpha_contiguous",
                           "convolution_input_", "q_flat", "k_flat", "v_flat", "qkv_concatenate",
                           "z_norm_materialize"))
        or any(tok in rec for tok in ("kda", "gdn", "recover_ssm", "short_conv", "causal_conv"))
    )


def _is_moe_router_dispatch_op(suffix: str, fam: str, rec: str) -> bool:
    if fam in {"router", "moe_init_routing"}:
        return True
    if suffix in {"router", "moe_gating_topk", "latent_down", "dispatch",
                  "dispatch.received_quantize", "route_pack"}:
        return True
    return rec in {"router", "moe_gating_topk", "route_pack"} or suffix.startswith("dispatch.")


def _is_moe_routed_op(suffix: str, fam: str, rec: str) -> bool:
    if fam == "grouped_gemm":
        return True
    if suffix in {"expert_up", "expert_act", "expert_down", "expert_down.quantize",
                  "unpack", "combine", "latent_norm", "latent_up", "join", "route_unpack"}:
        return True
    return any(tok in rec for tok in ("grouped_gate_up", "grouped_down", "grouped_situ",
                                      "grouped_swiglu", "megamoe", "expert_add", "route_unpack"))


def _is_dense_ffn_op(suffix: str, fam: str, rec: str, ffn: str) -> bool:
    if fam in {"gate_up_projection", "down_projection", "swiglu", "situ"}:
        return True
    if suffix in {"ffn.sp_gather", "gate_up", "gate_up.quantize", "swiglu", "situ",
                  "down", "down.quantize", "tp", "tp.reduce_scatter"}:
        return True
    return rec in {"dense_gate_up", "dense_down"} or (ffn == "dense" and "ffn" in suffix)


def _is_std_attn_op(suffix: str, fam: str, rec: str, attn: str) -> bool:
    if fam in {"qkv_projection", "output_projection", "attention", "rotary", "cache_store"}:
        return True
    return suffix in {
        "attention.sp_gather", "qkv", "qkv.quantize", "qk_norm", "qkv.value", "qkv.gate",
        "rope", "rotary", "cache", "attention", "output_gate", "output", "output.quantize",
        "tp_attn", "tp_attn.reduce_scatter", "output_reduce", "output_reduce.reduce_scatter",
    }


def _classify_stream_index(stream_name: str, family: str, op_id: str) -> int:
    lower = str(stream_name or "").strip().lower()
    fam = str(family or "").strip().lower()
    if lower in {"shared_stream", "aux", "mso", "shared", "stream1", "1"}:
        return 1
    if lower in {"comm", "hccl", "csa", "stream2", "2", "cp_decode_comm_stream"} or fam in _COLLECTIVE_FAMILIES:
        return 2
    if "shared_expert" in op_id.lower() and lower not in {"main", ""}:
        return 1
    return 0


def _classify_task_type(
    family: str, flops: float, vector_ops: float, link_bytes: float,
) -> str:
    if str(family or "").strip().lower() in _COLLECTIVE_FAMILIES or link_bytes > 0.0:
        return "HCCL"
    if flops > 0.0 and vector_ops > 0.0:
        return "MIX_AIC"
    if flops > 0.0:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


def _layer_archetype_key(
    lp: Any, stage: int, *, cand_producers: frozenset[int] = frozenset(),
) -> tuple[Any, ...]:
    """Structural equivalence key grouping identical decoder layers into archetypes."""
    idx = int(getattr(lp, "index", 0))
    kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
    idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
    topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
    cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
    idx_src = int(getattr(lp, "index_source_layer_id", -1))
    return (
        int(stage),
        str(getattr(lp, "attention", "")),
        str(getattr(lp, "ffn", "")),
        str(getattr(lp, "architecture", "")),
        bool(getattr(lp, "indexer", False)),
        str(getattr(lp, "indexer_type", "")),
        str(getattr(lp, "mla_variant", "")),
        bool(getattr(lp, "mla_output_gate", False)),
        bool(getattr(lp, "mla_use_nope", False)),
        int(getattr(lp, "num_heads", 0)),
        int(getattr(lp, "num_kv_heads", 0)),
        int(getattr(lp, "head_dim", 0)),
        int(getattr(lp, "intermediate_size", 0)),
        int(getattr(lp, "num_experts", 0)),
        int(getattr(lp, "top_k", 0)),
        int(getattr(lp, "moe_intermediate_size", 0)),
        int(getattr(lp, "num_shared_experts", 0)),
        int(getattr(lp, "shared_intermediate_size", 0)),
        int(getattr(lp, "linear_heads", 0)),
        int(getattr(lp, "linear_head_dim", 0)),
        int(getattr(lp, "expert_latent_size", 0)),
        bool(getattr(lp, "engram", False)),
        bool(getattr(lp, "hyper_connections", False)),
        int(getattr(lp, "window", 0)),
        str(getattr(lp, "cache_kind", "")),
        str(getattr(lp, "compressor_kind", "")),
        str(getattr(lp, "ffn_activation", "")),
        (kv_own >= 0, kv_own == idx),
        (idx_own >= 0, idx_own == idx),
        (topk_own >= 0, topk_own == idx),
        bool(getattr(lp, "candidate_filter", False)),
        int(getattr(lp, "candidate_topk_blocks", 0)),
        idx in cand_producers,
        cand_src >= 0,
        (idx_src >= 0, idx_src == idx),
    )


def _format_layer_range(indices: Sequence[int]) -> str:
    if not indices:
        return ""
    ordered = sorted(set(int(x) for x in indices))
    if len(ordered) == 1:
        return f"L{ordered[0]}"
    if len(ordered) <= MAX_EXPLICIT_LAYER_LABELS:
        return ", ".join(f"L{x}" for x in ordered)
    diffs = {b - a for a, b in zip(ordered, ordered[1:])}
    if len(diffs) == 1:
        step = next(iter(diffs))
        if step == 1:
            return f"L{ordered[0]}..L{ordered[-1]}"
        return f"L{ordered[0]}, L{ordered[1]}, ..., L{ordered[-1]}"
    return f"L{ordered[0]}..L{ordered[-1]} ({len(ordered)}L)"


def _describe_layer_plan(lp: Any) -> tuple[str, str]:
    """Return ``(full_description, short_label)`` derived 100% from ``LayerPlan``."""
    attn = str(getattr(lp, "attention", "gqa")).lower()
    ffn = str(getattr(lp, "ffn", "dense")).lower()
    indexer = bool(getattr(lp, "indexer", False))
    gated_mla = bool(getattr(lp, "mla_output_gate", False))
    latent_moe = int(getattr(lp, "expert_latent_size", 0) or 0) > 0
    engram = bool(getattr(lp, "engram", False))
    window = int(getattr(lp, "window", 0) or 0)

    if attn == "kda":
        attn_full, attn_short = "KDA", "KDA"
    elif attn == "gdn":
        attn_full, attn_short = "GDN", "GDN"
    elif attn in {"mla", "mla_dsa", "dsa"}:
        prefix = "Gated-MLA" if gated_mla else "MLA"
        attn_full = f"{prefix} + DSA (Lightning Indexer)" if indexer else prefix
        attn_short = f"{prefix}+DSA" if indexer else prefix
    elif window > 0 or attn == "swa":
        attn_full, attn_short = f"SWA (W={window})", "SWA"
    else:
        attn_full, attn_short = attn.upper(), attn.upper()
        if indexer:
            attn_full += " + Sparse Indexer"
            attn_short += "+LI"

    if ffn == "moe":
        ffn_full = "Latent MoE" if latent_moe else "MoE"
        ffn_short = "LatentMoE" if latent_moe else "MoE"
    else:
        ffn_full, ffn_short = "Dense MLP", "Dense"

    suffix = " + Engram/PLE" if engram else ""
    short_suf = "+Engram" if engram else ""
    return f"{attn_full} + {ffn_full}{suffix}", f"{attn_short}+{ffn_short}{short_suf}"


def _discover_model_archetypes(
    case: Case, *, include_draft: bool,
) -> tuple[
    tuple[LayerArchetypeSpec, ...],
    tuple[TraceScheduleEntry, ...],
    dict[int, Any],
    int,
    int,
]:
    """Discover unique Target & Draft layer archetypes + full-model replay schedule."""
    stage_layers = _stage_layer_counts(case)
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in case.model.layer_plans
        if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    groups: OrderedDict[tuple[Any, ...], list[Any]] = OrderedDict()
    for lp in case.model.layer_plans:
        st = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        groups.setdefault(_layer_archetype_key(lp, st, cand_producers=cand_producers), []).append(lp)

    archetypes: list[LayerArchetypeSpec] = [
        LayerArchetypeSpec(
            archetype_id="target_embed",
            scope="target",
            title="Target Embedding & Pre-Process (×1)",
            short_label="Embed",
            repeat_count=1,
            layer_indices=(),
            layer_range_text="Pre",
            rep_layer_index=None,
        ),
    ]
    rep_lp_by_idx: dict[int, Any] = {}
    arch_id_by_full_layer: dict[int, tuple[str, str]] = {}
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, lps in enumerate(groups.values()):
        rep = lps[0]
        rep_lp_by_idx[idx] = rep
        full_desc, short_lbl = _describe_layer_plan(rep)
        indices = tuple(int(x.index) for x in lps)
        range_txt = _format_layer_range(indices)
        tag = letters[idx] if idx < len(letters) else str(idx)
        layer_prefix = f"Layer {indices[0]}: " if len(indices) == 1 else ""
        arch_id = f"target_arch_{idx}"
        archetypes.append(LayerArchetypeSpec(
            archetype_id=arch_id,
            scope="target",
            title=f"Target Archetype {tag} — {layer_prefix}{full_desc} (×{len(indices)}, {range_txt})",
            short_label=short_lbl,
            repeat_count=len(indices),
            layer_indices=indices,
            layer_range_text=range_txt,
            rep_layer_index=idx,
        ))
        for full_idx in indices:
            arch_id_by_full_layer[full_idx] = (arch_id, short_lbl)

    archetypes.append(LayerArchetypeSpec(
        archetype_id="target_tail",
        scope="target",
        title="Target Final Norm & LM Head / Spec Verify (×1)",
        short_label="LMHead/Verify",
        repeat_count=1,
        layer_indices=(),
        layer_range_text="Post",
        rep_layer_index=None,
    ))

    aux = getattr(case.model, "auxiliary", None)
    dspark = getattr(aux, "dspark", None) if aux is not None else None
    mtp_spec = getattr(aux, "mtp", None) if aux is not None else None
    draft_layers = 0
    draft_label = "DSpark" if dspark is not None else "MTP"
    if include_draft and (dspark is not None or mtp_spec is not None):
        d_spec = dspark or mtp_spec
        draft_layers = max(1, int(getattr(d_spec, "layers", 1) or 1))
        d_desc = "MLA + Dense FFN" if str(getattr(d_spec, "attention_kind", "mla")).lower() == "mla" else "Draft Decoder"
        archetypes.extend([
            LayerArchetypeSpec(
                archetype_id="draft_pre",
                scope="draft",
                title=f"Draft [{draft_label}] — Context Feature Prep (×1)",
                short_label=f"Draft[{draft_label}]-Prep",
                repeat_count=1,
                layer_indices=(),
                layer_range_text="Pre",
                rep_layer_index=None,
            ),
            LayerArchetypeSpec(
                archetype_id="draft_context_kv",
                scope="draft",
                title=f"Draft [{draft_label}] — Context KV Prolog (×{draft_layers})",
                short_label=f"Draft[{draft_label}]-CtxKV",
                repeat_count=draft_layers,
                layer_indices=tuple(range(draft_layers)),
                layer_range_text=f"L0..L{draft_layers - 1}" if draft_layers > 1 else "L0",
                rep_layer_index=0,
            ),
            LayerArchetypeSpec(
                archetype_id="draft_arch_0",
                scope="draft",
                title=f"Draft [{draft_label}] — Parallel Block Draft Layer: {d_desc} (×{draft_layers})",
                short_label=f"Draft[{draft_label}]-Layer",
                repeat_count=draft_layers,
                layer_indices=tuple(range(draft_layers)),
                layer_range_text=f"L0..L{draft_layers - 1}" if draft_layers > 1 else "L0",
                rep_layer_index=0,
            ),
            LayerArchetypeSpec(
                archetype_id="draft_tail",
                scope="draft",
                title=f"Draft [{draft_label}] — Final Norm, Logits & Markov Head (×1)",
                short_label=f"Draft[{draft_label}]-Head",
                repeat_count=1,
                layer_indices=(),
                layer_range_text="Post",
                rep_layer_index=None,
            ),
        ])

    # Build full-model streaming schedule (Target Embed -> L0..L{N-1} -> Target Tail -> Draft).
    schedule: list[TraceScheduleEntry] = [
        TraceScheduleEntry("target_embed", "target", "Target Embedding & Pre-Process", None),
    ]
    for lp in case.model.layer_plans:
        a_id, short_lbl = arch_id_by_full_layer[int(lp.index)]
        schedule.append(TraceScheduleEntry(
            a_id, "target", f"Layer {lp.index} [{short_lbl}]", int(lp.index),
        ))
    schedule.append(TraceScheduleEntry("target_tail", "target", "Target Final Norm & LM Head / Verify", None))
    if draft_layers > 0:
        schedule.append(TraceScheduleEntry("draft_pre", "draft", f"Draft [{draft_label}] Feature Prep", None))
        for d_idx in range(draft_layers):
            schedule.append(TraceScheduleEntry(
                "draft_context_kv", "draft", f"Draft [{draft_label}] Context KV L{d_idx}", d_idx,
            ))
        for d_idx in range(draft_layers):
            schedule.append(TraceScheduleEntry(
                "draft_arch_0", "draft", f"Draft [{draft_label}] Block Layer L{d_idx}", d_idx,
            ))
        schedule.append(TraceScheduleEntry("draft_tail", "draft", f"Draft [{draft_label}] Logits & Markov", None))

    return tuple(archetypes), tuple(schedule), rep_lp_by_idx, len(case.model.layer_plans), draft_layers


def _build_archetype_representative_case(
    case: Case,
    rep_lp_by_idx: Mapping[int, Any],
    *,
    allow_draft_dedup: bool,
) -> Case:
    """Build a compact Case holding ONLY 1 layer per unique Target/Draft archetype."""
    old_to_new: dict[int, int] = {}
    cand_producers = frozenset(
        int(getattr(lp, "candidate_source_layer_id", -1))
        for lp in case.model.layer_plans
        if int(getattr(lp, "candidate_source_layer_id", -1)) >= 0
    )
    last_kv_prod = -1
    last_idx_prod = -1
    last_topk_prod = -1
    last_cand_prod = -1
    rep_list: list[Any] = []
    for new_idx, lp in sorted(rep_lp_by_idx.items()):
        old_idx = int(lp.index)
        old_to_new[old_idx] = new_idx
        kv_own = int(getattr(lp, "kv_owner_layer_id", -1))
        idx_own = int(getattr(lp, "index_k_owner_layer_id", -1))
        topk_own = int(getattr(lp, "topk_owner_layer_id", -1))
        cand_src = int(getattr(lp, "candidate_source_layer_id", -1))
        idx_src = int(getattr(lp, "index_source_layer_id", -1))

        prev_kv_prod, prev_idx_prod, prev_topk_prod, prev_cand_prod = (
            last_kv_prod, last_idx_prod, last_topk_prod, last_cand_prod,
        )
        if kv_own == old_idx:
            new_kv = new_idx
            last_kv_prod = new_idx
        elif kv_own >= 0:
            new_kv = old_to_new.get(kv_own, prev_kv_prod)
        else:
            new_kv = -1

        if idx_own == old_idx:
            new_idx_own = new_idx
            last_idx_prod = new_idx
        elif idx_own >= 0:
            new_idx_own = old_to_new.get(idx_own, prev_idx_prod)
        else:
            new_idx_own = -1

        if topk_own == old_idx:
            new_topk_own = new_idx
            last_topk_prod = new_idx
        elif topk_own >= 0:
            new_topk_own = old_to_new.get(topk_own, prev_topk_prod)
        else:
            new_topk_own = -1

        if old_idx in cand_producers:
            last_cand_prod = new_idx

        new_idx_src = (
            new_idx if idx_src == old_idx
            else (old_to_new.get(idx_src, prev_idx_prod) if idx_src >= 0 else -1)
        )
        new_cand_src = old_to_new.get(cand_src, prev_cand_prod) if cand_src >= 0 else -1
        rep_list.append(replace(
            lp,
            index=new_idx,
            kv_owner_layer_id=new_kv,
            index_k_owner_layer_id=new_idx_own,
            topk_owner_layer_id=new_topk_own,
            candidate_source_layer_id=new_cand_src,
            index_source_layer_id=new_idx_src,
        ))
    rep_plans = tuple(rep_list)
    if not rep_plans:
        return case
    aux = getattr(case.model, "auxiliary", None)
    rep_aux = aux
    if allow_draft_dedup and aux is not None and getattr(aux, "dspark", None) is not None:
        dsp = aux.dspark
        if dsp.layer_plans:
            d0 = replace(dsp.layer_plans[0], index=0, kv_owner_layer_id=0)
            clamped = tuple(min(int(lid), len(rep_plans) - 1) for lid in dsp.target_layer_ids)
            rep_dsp = replace(dsp, layers=1, layer_plans=(d0,), target_layer_ids=clamped)
            rep_aux = replace(aux, dspark=rep_dsp)
    rep_engram = getattr(case.model, "engram", None)
    if rep_engram is not None and getattr(rep_engram, "tables", ()):
        by_orig_layer = {int(t.layer_id): t for t in rep_engram.tables}
        default_tbl = rep_engram.tables[0]
        remapped_tables = tuple(
            replace(by_orig_layer.get(int(orig_lp.index), default_tbl), layer_id=new_idx)
            for new_idx, orig_lp in sorted(rep_lp_by_idx.items())
            if bool(getattr(orig_lp, "engram", False))
        )
        if remapped_tables:
            rep_engram = replace(rep_engram, tables=remapped_tables)
    rep_model = replace(
        case.model, layers=len(rep_plans), layer_plans=rep_plans,
        auxiliary=rep_aux, engram=rep_engram,
    )
    # Keep pp=1 partition clean when deduplicated to <= pp layers.
    p = case.runtime.parallel
    if p.pp > len(rep_plans):
        return case
    rep_runtime = replace(case.runtime, pp_layer_partition=())
    return replace(case, model=rep_model, runtime=rep_runtime)


def _format_dtype_upper(raw: str) -> str:
    tok = str(raw or "bf16").strip().lower()
    mapping = {
        "bf16": "BF16", "bfloat16": "BF16", "fp16": "FP16", "float16": "FP16",
        "fp32": "FP32", "float32": "FP32", "fp8": "FP8", "fp8_e4m3": "FP8",
        "fp8_e5m2": "FP8", "mxfp4": "MXFP4", "mx_fp4": "MXFP4", "fp4": "MXFP4",
        "fp4_e2m1": "MXFP4", "mxfp8": "MXFP8", "mx_fp8": "MXFP8",
        "int8": "INT8", "int4": "INT4", "int32": "INT32", "int64": "INT64",
    }
    return mapping.get(tok, tok.upper() or "BF16")


def _format_tensor_contract(
    op_id: str,
    family: str,
    recipe: str,
    shape: Sequence[int],
    dtype: str,
    meta: Mapping[str, Any],
    case: Case,
    layer_plan: Any | None,
) -> str:
    """Format a readable rank-local tensor contract (never bare ambiguous element counts)."""
    suffix = _strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    act_dt = _format_dtype_upper(str(meta.get("activation_format") or meta.get("input_format") or dtype or "bf16"))
    w_dt = _format_dtype_upper(str(meta.get("weight_format") or dtype or "bf16"))
    out_dt = _format_dtype_upper(str(meta.get("output_format") or dtype or "bf16"))
    dims = tuple(int(d) for d in shape)
    tp = max(1, int(case.runtime.parallel.tp))

    if fam == "grouped_gemm" and len(dims) == GEMM_SHAPE_RANK:
        m_dim, n_dim, k_dim = dims
        e_local = int(meta.get("matrix_weight_bank_groups") or meta.get("local_experts") or 0)
        if e_local <= 0 and layer_plan is not None:
            ep = max(1, int(case.runtime.parallel.ep))
            e_local = max(1, int(getattr(layer_plan, "num_experts", 0) or 0) // ep)
        w_shape = f"[{e_local},{n_dim},{k_dim}]" if e_local > 0 else f"[{n_dim},{k_dim}]"
        return f"x:[{m_dim},{k_dim}] {act_dt} × W:{w_shape} {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"

    if len(dims) == GEMM_SHAPE_RANK and (
        fam in {"gemm", "projection", "gate_up_projection", "down_projection",
                "qkv_projection", "output_projection", "router", "logits", "index_projection"}
        or meta.get("program_kind") == "matmul"
    ):
        m_dim, n_dim, k_dim = dims
        if suffix in {"q_absorb", "v_expand"}:
            return f"x:[{m_dim},{k_dim}] {act_dt} × W_bmm:[{k_dim},{n_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"
        return f"x:[{m_dim},{k_dim}] {act_dt} × W:[{n_dim},{k_dim}] {w_dt} -> y:[{m_dim},{n_dim}] {out_dt}"

    if fam == "attention":
        if len(dims) == ATTN_5D_RANK:
            q_tok, ctx, nh, dqk, dv = dims
            return f"Q:[{q_tok},{nh},{dqk}] {act_dt} × KV:[{ctx},1,{dqk}] -> O:[{q_tok},{nh},{dv}] {out_dt}"
        if len(dims) == ATTN_4D_RANK:
            q_tok, ctx, nh, hd = dims
            return f"Q:[{q_tok},{nh},{hd}] {act_dt} × KV:[{ctx},{hd}] -> O:[{q_tok},{nh},{hd}] {out_dt}"
        if len(dims) == GEMM_SHAPE_RANK:
            q_tok, nh, dv = dims
            topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else 0
            kv_txt = f"SparseKV(topk={topk})" if topk > 0 else "PagedKV"
            return f"Q:[{q_tok},{nh},{dv}] {act_dt} × {kv_txt} -> O:[{q_tok},{nh},{dv}] {out_dt}"

    if fam in {"allgather", "reduce_scatter", "reducescatter", "allreduce", "alltoall", "mc2_protocol"}:
        if len(dims) >= MATRIX_2D_RANK:
            sh_txt = ",".join(str(d) for d in dims)
            return f"comm:[{sh_txt}] {act_dt} (TP={tp}, EP={case.runtime.parallel.ep})"
        if len(dims) == 1:
            rows = int(meta.get("graph_active_query_rows") or 0)
            if rows > 0 and dims[0] % rows == 0:
                return f"comm:[{rows},{dims[0] // rows}] {act_dt} (TP={tp})"
            return f"comm:[{dims[0]}] {act_dt} (TP={tp})"

    if len(dims) == 1 and dims[0] > 0:
        rows = int(meta.get("graph_active_query_rows") or 0)
        hidden = int(case.model.hidden_size or 0)
        if rows > 0 and dims[0] % rows == 0:
            return f"x:[{rows},{dims[0] // rows}] {act_dt} -> y:[{rows},{dims[0] // rows}] {out_dt}"
        if hidden > 0 and dims[0] % hidden == 0:
            return f"x:[{dims[0] // hidden},{hidden}] {act_dt} -> y:[{dims[0] // hidden},{hidden}] {out_dt}"
        return f"x:[{dims[0]}] {act_dt}"

    if dims:
        sh_txt = ",".join(str(d) for d in dims)
        return f"x:[{sh_txt}] {act_dt} -> y:[{sh_txt}] {out_dt}"
    return f"scalar {act_dt}"


def _format_source_ref(
    resolved_sources: Sequence[Any],
    default_ref: str,
) -> str:
    """Format canonical repository/path source reference (Item 9)."""
    parts: list[str] = []
    for src in resolved_sources:
        if isinstance(src, Mapping):
            repo = str(src.get("repository") or "").strip()
            rel = str(src.get("relative_path") or "").strip()
        else:
            repo = str(getattr(src, "repository", "") or "").strip()
            rel = str(getattr(src, "relative_path", "") or "").strip()
        if rel:
            parts.append(f"{repo}/{rel}" if repo else rel)
    if parts:
        return "; ".join(dict.fromkeys(parts))
    return default_ref


def _native_kernel_from_structure(
    op_id: str, spec: OpSpec | None, shape: Sequence[int], dtype: str,
) -> str:
    if spec is None:
        return ""
    meta = dict(spec.metadata) if spec.metadata else {}
    op_dict = {
        "op_id": op_id,
        "family": spec.family,
        "recipe": spec.recipe,
        "metadata": meta,
        "local_shape": list(spec.local_shape or shape),
        "padded_shape": list(spec.padded_shape or shape),
        "shape": list(shape),
        "dtype": dtype or spec.dtype or "bf16",
        "phase": spec.phase or "decode",
        "source_boundary": spec.source_boundary or "",
    }
    try:
        contract = predicted_contract(op_dict)
        nat = native_implementation(op_dict, contract)
    except Exception:
        return ""
    if isinstance(nat, Mapping):
        kinds = nat.get("native_kinds") or ()
        if kinds:
            return str(kinds[0])
    return ""


def _extract_native_candidate(struct_node: Mapping[str, Any] | None) -> str:
    """Read primary native kernel symbol from ``export_structure`` node/candidates."""
    if not isinstance(struct_node, Mapping):
        return ""
    nat_exec = struct_node.get("native_execution")
    if isinstance(nat_exec, Mapping):
        sel = nat_exec.get("selected")
        if isinstance(sel, Mapping) and sel.get("native"):
            cand = str(sel["native"]).strip()
            if cand and cand not in BANNED_IMPLEMENTATION_NAMES:
                return cand
    for raw_list in (
        struct_node.get("native_candidates"),
        (struct_node.get("implementation") or {}).get("native_kinds")
        if isinstance(struct_node.get("implementation"), Mapping) else (),
    ):
        if isinstance(raw_list, str):
            raw_list = (raw_list,)
        for item in raw_list or ():
            cand = str(item.get("native") if isinstance(item, Mapping) else item or "").strip()
            if cand and cand not in BANNED_IMPLEMENTATION_NAMES:
                return cand
    return ""


def _resolve_ascend_kernel_identity(
    op_id: str,
    spec: OpSpec | None,
    family: str,
    recipe: str,
    shape: Sequence[int],
    dtype: str,
    case: Case,
    layer_plan: Any | None,
    resolved_sources: Sequence[Any],
    struct_node: Mapping[str, Any] | None = None,
    subgraph_info: Mapping[str, Any] | None = None,
) -> AscendKernelDescriptor:
    """Map an op using ``export_structure`` first, with clean semantic fallbacks."""
    suffix = _strip_op_prefix(op_id).lower()
    fam = str(family or "").lower()
    rec = str(recipe or "").lower()
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    track = str(dict(case.runtime.metadata).get("feature_track_report", {}).get("feature_track", "") or "")
    is_ideal = track == "expected_ideal"
    is_raw_eager = track == "raw_eager"
    is_quant_mm = str(meta.get("activation_format") or "").lower() in {"int8", "fp8", "fp8_e4m3", "fp8_e5m2"}
    is_mxfp4_w = str(meta.get("weight_format") or "").lower() in {"mxfp4", "mx_fp4", "fp4", "fp4_e2m1"}
    struct_nat = _extract_native_candidate(struct_node)
    nat_kind = struct_nat or _native_kernel_from_structure(op_id, spec, shape, dtype)
    contract_txt = _format_tensor_contract(op_id, fam, rec, shape, dtype, meta, case, layer_plan)

    role, kernel, backend, status, def_src, needs_new, new_spec, canon_key = _classify_kernel_tuple(
        suffix=suffix, fam=fam, rec=rec, meta=meta, nat_kind=nat_kind,
        is_ideal=is_ideal, is_raw_eager=is_raw_eager,
        is_quant_mm=is_quant_mm, is_mxfp4_w=is_mxfp4_w,
        layer_plan=layer_plan,
    )
    # Prefer core export_structure registry fields as the single source of truth when present.
    if isinstance(struct_node, Mapping):
        if struct_nat and not needs_new and not (is_raw_eager and (fam == "rmsnorm" or suffix == "linear_output_norm_gate")):
            kernel = "aclnnLayerNorm_LayerNormV3" if (fam == "layernorm" and struct_nat in {"aclnnLayerNorm", "LayerNormV3"}) else struct_nat
        raw_backend = str(struct_node.get("backend_stack") or "").strip().lower()
        if raw_backend in _STRUCTURE_BACKEND_MAP:
            backend = _STRUCTURE_BACKEND_MAP[raw_backend]
        raw_readiness = str(struct_node.get("readiness") or "").strip().lower()
        if raw_readiness in _STRUCTURE_READINESS_MAP:
            mapped_status, mapped_new = _STRUCTURE_READINESS_MAP[raw_readiness]
            if mapped_new or not needs_new:
                status = mapped_status if not new_spec else f"{mapped_status} {new_spec}"
                needs_new = mapped_new
        struct_src = str(struct_node.get("source_ref") or "").strip()
        if struct_src:
            def_src = struct_src
    if isinstance(subgraph_info, Mapping):
        f_status = str(subgraph_info.get("fusion_status") or "").strip().lower()
        if f_status == "fused_new_kernel_required" and not needs_new:
            reason = str(subgraph_info.get("reason") or "").strip()
            status = f"{KERNEL_STATUS_NEW_OP} {reason}".strip()
            needs_new = True
            new_spec = new_spec or reason
        elif f_status == "fused_framework_enable" and status == KERNEL_STATUS_EXISTING:
            status = KERNEL_STATUS_FRAMEWORK
    if kernel in BANNED_IMPLEMENTATION_NAMES or not kernel:
        kernel = "aclnnCustomOp"
    return AscendKernelDescriptor(
        logical_role=role,
        ascend_kernel=kernel,
        kernel_backend=backend,
        kernel_status=status,
        source_ref=_format_source_ref(resolved_sources, def_src),
        tensor_contract=contract_txt,
        needs_new_kernel=needs_new,
        new_kernel_spec=new_spec,
        canonical_op_key=canon_key,
    )


def _classify_kernel_tuple(
    *,
    suffix: str,
    fam: str,
    rec: str,
    meta: Mapping[str, Any],
    nat_kind: str,
    is_ideal: bool,
    is_raw_eager: bool = False,
    is_quant_mm: bool,
    is_mxfp4_w: bool,
    layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    """Return ``(role, kernel, backend, status, default_src, needs_new, new_spec, canon_key)``."""
    mm_kernel = "QuantBatchMatmulV3" if is_quant_mm else "MatMulV2"
    mm_src = "vllm-ascend/vllm_ascend/ops/linear.py"

    if suffix.endswith(".pad") or suffix == "pad":
        return (
            "TP-Aligned Sequence Padding", "PadV3 / TensorMove", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/distributed/parallel_state.py",
            False, "", suffix,
        )
    if suffix == "sp_shard":
        return (
            "Sequence-Parallel Token Shard Split", "Slice / Split", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/distributed/parallel_state.py",
            False, "", "sp_shard",
        )
    if suffix == "markov_bias":
        return (
            "Draft Markov Logits Bias Add", "aclnnAdd", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py",
            False, "", "markov_bias",
        )

    # 1. AttnRes / mHC / Norms / Quantize
    if fam == "attn_res_prefix" or "attn_res" in suffix:
        return (
            "Block Attention Residual Prefix (Norm+Score+Softmax+BMM)",
            "vllm_ascend_attn_res_prefix",
            BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} AttnRes Cross-Layer Fused Prefix",
            "vllm-ascend/vllm_ascend/models/kimi_k3/attn_res.py",
            True,
            "Fused block residual norm+projection+softmax+weighted sum across block history",
            suffix,
        )
    if fam == "mhc" or ".hc_" in f".{suffix}":
        return (
            f"mHC Hyper-Connection ({suffix})",
            "mhc_sinkhorn_mix_kernel" if "sinkhorn" in suffix or "mix" in suffix else ("MatMulV2" if fam == "gemm" else "Cast / RmsNorm"),
            BACKEND_TRITON if fam == "mhc" else BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK if fam == "mhc" else KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/mhc.py",
            False, "", suffix,
        )
    if fam in {"rmsnorm", "add_rmsnorm", "add_rmsnorm_quant", "rmsnorm_quant", "layernorm", "quantize"}:
        return _classify_norm_quant_tuple(suffix, fam, rec, meta, nat_kind, is_raw_eager=is_raw_eager)

    # 2. MLA Prolog / Attention / DSA Lightning Indexer
    if "mla_prolog" in suffix or "mla_prolog" in rec:
        return (
            "Fused MLA Prolog (Q/KV-A + RMSNorm + Q-B + W_UK Absorb + KV Cache)",
            "aclnnMlaPrologV3",
            BACKEND_CANN,
            f"{KERNEL_STATUS_FRAMEWORK} (96->128 head pad & NoPE P^-1)",
            "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "mla_prolog_v3",
        )
    if suffix in {"q_absorb", "v_expand"}:
        role = "W_UK Absorbed Q BMM" if suffix == "q_absorb" else "W_UV Value Expand BMM"
        return (
            role, "BatchMatMulV2 (npu_transpose_batchmatmul)", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", suffix,
        )
    if fam == "attention":
        return _classify_attention_tuple(suffix, rec, nat_kind, layer_plan)
    if fam == "attention_merge" or suffix == "dcp_lse_merge":
        return (
            "DCP Attention LSE Merge", "aclnnAttentionUpdate", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/context_parallel/mla_cp.py",
            False, "", "dcp_lse_merge",
        )
    if _is_dsa_indexer_op(suffix, fam, rec):
        return _classify_indexer_tuple(suffix, fam, rec, mm_kernel, layer_plan)

    # 3. KDA & GDN Linear Attention
    if _is_linear_attn_op(suffix, fam, rec, "kda" if "kda" in rec else ""):
        return _classify_linear_attn_tuple(suffix, fam, rec, meta, is_ideal, mm_kernel, is_raw_eager=is_raw_eager)

    # 4. MoE Router / Dispatch / Routed Experts / Shared Experts / Dense FFN
    if _is_moe_router_dispatch_op(suffix, fam, rec) or _is_moe_routed_op(suffix, fam, rec) or _is_shared_expert_op(suffix, rec):
        return _classify_moe_tuple(suffix, fam, rec, meta, nat_kind, is_ideal, is_quant_mm, is_mxfp4_w, mm_kernel)

    # 5. Collectives
    if fam in _COLLECTIVE_FAMILIES:
        return _classify_collective_tuple(suffix, fam, rec)

    # 6. Cache stores / Rotary / Projections / FFN / Head / Elementwise
    return _classify_proj_and_misc_tuple(suffix, fam, nat_kind, mm_kernel, mm_src)


def _classify_proj_and_misc_tuple(
    suffix: str, fam: str, nat_kind: str, mm_kernel: str, mm_src: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    if fam in {"cache_store", "kv_compress_epilog", "kv_cache_write"} or suffix in {"kv_store", "kv_scatter", "cache"}:
        return (
            "Paged KV Cache Scatter Write", "aclnnScatterPaKvCache", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "kv_store",
        )
    if fam == "rotary" or suffix in {"rotary", "rope", "q_rotary", "kv_rotary", "output_inverse_rotary"}:
        return (
            f"Rotary Position Embedding ({suffix})", "InplacePartialRotaryMul", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/rotary_embedding.py",
            False, "", suffix,
        )
    if fam in {"swiglu", "situ"}:
        if fam == "situ":
            return (
                "K3 SiTU Gated Activation (beta=4, linear_beta=25)",
                "K3 SituAndMul",
                BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} K3 SituAndMul(beta=4, linear_beta=25)",
                "vllm-ascend/vllm_ascend/models/kimi_k3/modeling_kimi_k3.py",
                True,
                "Fused K3 SituAndMul activation (non-standard SiLU variant not in shipping CANN SwiGlu)",
                suffix,
            )
        return (
            "SwiGLU Activation", "SwiGlu", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/activation.py", False, "", suffix,
        )
    if fam in {"embedding", "engram_gather"}:
        return (
            "Token / Engram Table Gather", "aclnnEmbedding (GatherV2)", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm/vllm/model_executor/layers/vocab_parallel_embedding.py",
            False, "", suffix,
        )
    if fam == "logits":
        role = "Draft Markov Logits Head" if "markov" in suffix else "Vocab Parallel LM Head GEMM"
        return (
            role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm/vllm/model_executor/layers/logits_processor.py", False, "", suffix,
        )
    if fam == "acceptance":
        return (
            "Speculative Verification & Token Acceptance",
            "speculative_verify_accept",
            BACKEND_TORCH_NPU,
            KERNEL_STATUS_FRAMEWORK,
            "vllm-ascend/vllm_ascend/spec_decode/eagle_proposer.py",
            False, "", "accept",
        )
    if fam in {"gemm", "projection", "gate_up_projection", "down_projection",
               "qkv_projection", "output_projection"}:
        role_map = {
            "q_a": "Q-A Low-Rank Projection Linear",
            "kv_a": "KV-A Compressed Projection Linear",
            "qkv_a": "Fused Q/KV-A Low-Rank Projection",
            "q_b": "Q-B Head Expansion Linear",
            "qkv": "Fused QKV Projection Linear",
            "output_gate": "Gated-MLA Output Gate Linear (W_g)",
            "output": "Attention Output Projection (O-Proj)",
            "grouped_output_a": "Grouped Low-Rank O-Proj Stage A",
            "grouped_output_b": "Grouped Low-Rank O-Proj Stage B",
            "gate_up": "Dense FFN Gate+Up Projection",
            "down": "Dense FFN Down Projection",
            "features": "DSpark Context Feature Projection",
        }
        return (
            role_map.get(suffix, f"Linear Projection ({suffix})"),
            mm_kernel,
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            mm_src,
            False, "", suffix,
        )
    if suffix == "gated_output":
        return (
            "Gated-MLA Sigmoid × Attention Output",
            "aclnnSigmoid + aclnnMul",
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/attention/mla_v1.py",
            False, "", "gated_output",
        )
    if suffix in {"dcp_query_pack", "dcp_output_pack", "candidate_pack"} or fam in {"tensor_copy", "copy", "tensor_fill"}:
        return (
            f"Tensor Pack / Layout Materialize ({suffix})",
            "TensorMove / Reshape",
            BACKEND_CANN,
            KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/attention/context_parallel/mla_cp.py",
            False, "", suffix,
        )
    eager_vec_map = {
        "cast": ("Dtype Cast", "Cast"),
        "pow": ("Elementwise Power", "Pows"),
        "pows": ("Elementwise Power", "Pows"),
        "reducemean": ("Reduction Mean", "ReduceMean"),
        "rsqrt": ("Reciprocal Sqrt", "Rsqrt"),
        "sigmoid": ("Sigmoid Activation", "Sigmoid"),
        "mul": ("Elementwise Multiply", "Mul"),
        "adds": ("Scalar Bias Add", "Adds"),
        "add": ("Elementwise Add", "Add"),
    }
    if suffix in eager_vec_map or fam in eager_vec_map:
        e_role, e_kern = eager_vec_map.get(suffix) or eager_vec_map[fam]
        return (
            e_role, e_kern, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/common.py", False, "", suffix or fam,
        )
    return (
        f"Elementwise / Aux Op ({suffix or fam})",
        nat_kind if (nat_kind and nat_kind not in BANNED_IMPLEMENTATION_NAMES) else "aclnnAdd / VectorOp",
        BACKEND_CANN,
        KERNEL_STATUS_EXISTING,
        "vllm-ascend/vllm_ascend/ops/common.py",
        False, "", suffix or fam,
    )


def _classify_norm_quant_tuple(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], nat_kind: str,
    *, is_raw_eager: bool = False,
) -> tuple[str, str, str, str, str, bool, str, str]:
    out_fmt = str(meta.get("output_format") or "").lower()
    is_mx = out_fmt in {"fp8_e4m3", "mxfp8"} and str(meta.get("activation_scale_format") or "") == "fp8_e8m0"
    if fam == "layernorm":
        return (
            "Indexer K LayerNorm (1-kernel FP32-UB)" if "index" in suffix else "LayerNorm (1-kernel FP32-UB)",
            "aclnnLayerNorm_LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    if fam == "add_rmsnorm_quant":
        kern = "AddRmsNormDynamicMxQuant" if is_mx else "AddRmsNormDynamicQuant"
        return (
            "Fused Residual Add + RMSNorm + Dynamic Quant", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py",
            False, "", suffix,
        )
    if fam == "rmsnorm_quant":
        kern = "RmsNormDynamicMxQuant" if is_mx else "RmsNormDynamicQuant"
        return (
            "Fused RMSNorm + Dynamic Quant", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py",
            False, "", suffix,
        )
    if fam == "add_rmsnorm":
        return (
            "Fused Residual Add + RMSNorm", "AddRmsNorm", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/layernorm.py",
            False, "", suffix,
        )
    if fam == "quantize":
        kern = "DynamicMxQuant" if is_mx else "DynamicQuant"
        return (
            f"Dynamic Activation Quantize ({suffix})", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/quantization/w8a8_dynamic.py",
            False, "", suffix,
        )
    role_map = {
        "q_norm": "Q Latent RMSNorm",
        "kv_norm": "KV Latent RMSNorm",
        "latent_norm": "MoE Routed Latent RMSNorm",
        "final_norm": "Final Model RMSNorm",
        "context_norm": "DSpark Context RMSNorm",
        "qk_norm": "Q/K Head RMSNorm",
    }
    if is_raw_eager or "unfused" in rec or "eager" in rec:
        return (
            f"Unfused Eager {role_map.get(suffix, f'RMSNorm ({suffix})')} (8 kernels)",
            "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast",
            BACKEND_TORCH_NPU, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
        )
    return (
        role_map.get(suffix, f"RMSNorm ({suffix})"),
        "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING,
        "vllm-ascend/vllm_ascend/ops/layernorm.py", False, "", suffix,
    )


def _classify_attention_tuple(
    suffix: str, rec: str, nat_kind: str, layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else 0
    if "mixed_quant_sparse" in rec or suffix == "mqsmla":
        return (
            f"Mixed-Quant Sparse Flash MLA (topk={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            "aclnnMixedQuantSparseFlashMla", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "cann/ops-transformer/src/mc2_or_sfa/mixed_quant_sparse_flash_mla.cpp",
            False, "", "mqsmla",
        )
    if topk > 0 or "sparse" in rec or "paired_mla" in rec:
        return (
            f"Sparse Flash Attention (TopK={topk or DEFAULT_SPARSE_INDEX_TOPK})",
            "aclnnSparseFlashAttention", BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/attention/sfa_v1.py",
            False, "", "latent_attention_batch",
        )
    return (
        "Fused Infer Attention Score v2 (FIA v2)",
        "aclnnFusedInferAttentionScoreV2", BACKEND_CANN, KERNEL_STATUS_EXISTING,
        "vllm-ascend/vllm_ascend/attention/mla_v1.py",
        False, "", suffix,
    )


def _classify_indexer_tuple(
    suffix: str, fam: str, rec: str, mm_kernel: str, layer_plan: Any | None,
) -> tuple[str, str, str, str, str, bool, str, str]:
    topk = int(getattr(layer_plan, "index_topk", 0) or 0) if layer_plan is not None else DEFAULT_SPARSE_INDEX_TOPK
    src = "vllm-ascend/vllm_ascend/attention/sfa_v1.py"
    if suffix in {"index_scores", "quant_lightning_indexer"} or fam == "indexer":
        kern = "aclnnPoolKeyIndexer" if "pool_key" in rec else "aclnnQuantLightningIndexer"
        return (
            f"DSA Lightning Indexer Score (topk={topk})", kern, BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_scores",
        )
    if suffix == "index_topk":
        return (
            f"DSA Local TopK-{topk} Selection", "aclnnTopk", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_topk",
        )
    if suffix == "candidate_merge":
        return (
            f"DSA Cross-DCP TopK-{topk} Candidate Merge", "aclnnTopk / CandidateMerge", BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK, src, False, "", "candidate_merge",
        )
    if suffix == "candidate_gather":
        return (
            "DSA Candidate Indices AllGather (DCP)", "hcom_allGather_", BACKEND_HCCL_MC2,
            KERNEL_STATUS_EXISTING, src, False, "", "candidate_gather",
        )
    if suffix == "index_store" or fam in {"indexer_quant_cache", "kv_cache_write"}:
        return (
            "DSA Indexer K Cache Scatter Write", "aclnnScatterPaKvCache", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, src, False, "", "index_store",
        )
    if suffix == "index_norm":
        return ("DSA Indexer K LayerNorm", "LayerNormV3", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "index_norm")
    if suffix == "index_query_quantize":
        return ("DSA Indexer Query Dynamic Quantize", "DynamicQuant", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "index_query_quantize")
    if suffix in {"index_k", "index_q", "index_weights", "index_wq", "index_wk"}:
        labels = {
            "index_k": "DSA Indexer Key Projection (W_k)",
            "index_q": "DSA Indexer Query Projection (W_q)",
            "index_weights": "DSA Indexer Head Weights Linear",
            "index_wq": "DSA Indexer Query Projection",
            "index_wk": "DSA Indexer Key Projection",
        }
        return (labels[suffix], mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", suffix)
    if suffix == "candidate_pack":
        return ("DSA Candidate Indices Pack", "TensorMove / GatherV2", BACKEND_CANN, KERNEL_STATUS_EXISTING, src, False, "", "candidate_pack")
    return (
        f"DSA Indexer Aux ({suffix})", "TensorMove / GatherV2", BACKEND_CANN,
        KERNEL_STATUS_EXISTING, src, False, "", suffix,
    )


def _classify_linear_attn_tuple(
    suffix: str, fam: str, rec: str, meta: Mapping[str, Any], is_ideal: bool, mm_kernel: str,
    *, is_raw_eager: bool = False,
) -> tuple[str, str, str, str, str, bool, str, str]:
    kda_src = "vllm/vllm/third_party/flash_linear_attention/ops/kda.py"
    if suffix.startswith("recurrent_scan") or "recurrent_kda" in rec:
        has_spec = bool(meta.get("kda_recurrent_params", {}).get("recover_ssm")) or is_ideal
        if is_ideal:
            return (
                "Fused Recurrent KDA (Spec Verify + Persistent State)",
                "flash_kda_persistent",
                BACKEND_CANNBOT_DSL,
                f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec 2D State-Slot Writeback & Persistent UB",
                "cann/cannbot-dsl/samples/attention/flash_kda/flash_kda_persistent.py",
                True,
                "Persistent cannbot-dsl / AscendC recurrent KDA with 2D accepted-token state slot writeback",
                "recurrent_scan",
            )
        return (
            "Fused Recurrent KDA Decode",
            "fused_recurrent_kda",
            BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative KDA State-Slot Writeback" if has_spec else KERNEL_STATUS_EXISTING,
            kda_src,
            has_spec,
            "Speculative 2D state-slot writeback extension for fused_recurrent_kda" if has_spec else "",
            "recurrent_scan",
        )
    if suffix == "decode_recurrent_scan" or fam == "recurrent_vector":
        return (
            "Fused Recurrent GDN Decode", "RecurrentGatedDeltaRule", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/gdn.py", False, "", "decode_recurrent_scan",
        )
    if "recover_ssm" in suffix or "recover_ssm" in rec:
        return (
            "Speculative Accepted-Token SSM State Commit (RecoverSSM)",
            "recover_ssm_state_commit",
            BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Speculative Accepted-State Commit Kernel",
            "vllm-ascend/vllm_ascend/ops/linear_attention/recover_ssm.py",
            True,
            "Compact accepted-token SSM/KDA state replay & single HBM writeback commit kernel",
            "recover_ssm_commit",
        )
    if suffix == "short_convolution" or fam in {"short_convolution", "causal_conv1d_fn"}:
        return (
            "Causal Conv1d + SiLU (Kernel=4)",
            "CausalConv1d",
            BACKEND_CANN,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} Spec Accepted-Token Rolling Conv Update" if is_ideal else KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/ops/causal_conv1d.py",
            is_ideal,
            "Speculative multi-token rolling conv_state update for CausalConv1d" if is_ideal else "",
            "short_convolution",
        )
    if suffix == "linear_output_norm_gate":
        if is_raw_eager or (not is_ideal and ("unfused" in rec or "eager" in rec)):
            return (
                "Unfused Eager RMSNorm + Sigmoid Gate (11 kernels)",
                "Cast->Pows->ReduceMean->Adds->Rsqrt->Mul->Cast->Mul->Sigmoid->Mul->Cast",
                BACKEND_TORCH_NPU,
                KERNEL_STATUS_EXISTING,
                kda_src,
                False, "", "linear_output_norm_gate",
            )
        return (
            "Fused RMSNorm + Sigmoid Output Gate",
            "FusedRMSNormGated",
            BACKEND_TRITON,
            f"{KERNEL_STATUS_NEW_OR_EXTEND} FusedRMSNormGated(eps=1e-5, sigmoid)",
            kda_src,
            True,
            "Fused RMSNorm + sigmoid gate kernel on [T, Hv, Dv] replacing 11-kernel eager chain",
            "linear_output_norm_gate",
        )
    if "l2norm" in suffix:
        return (
            f"Q/K L2 Normalization ({suffix})", "l2norm_fwd_kernel2", BACKEND_TRITON,
            KERNEL_STATUS_EXISTING, "vllm/vllm/third_party/flash_linear_attention/ops/l2norm.py",
            False, "", suffix,
        )
    if suffix == "delta_gates" or fam == "gdn_gating":
        return (
            "Delta Decay Gate Transform", "fused_gdn_gating_kernel", BACKEND_TRITON,
            KERNEL_STATUS_EXISTING, "vllm-ascend/vllm_ascend/ops/gdn.py", False, "", "delta_gates",
        )
    if fam in {"reduce_scatter", "allgather", "allreduce"}:
        return _classify_collective_tuple(suffix, fam, rec)
    if fam in {"gemm", "projection"} or suffix.startswith("linear_"):
        lin_roles = {
            "linear_qkvbfg_a": "KDA Fused Q/K/V/Beta/Decay-A/Gate Projection",
            "linear_decay_b": "KDA Low-Rank Decay-B Projection",
            "linear_gate_b": "KDA Low-Rank Gate-B Projection",
            "linear_output": "KDA Output Projection (O-Proj)",
            "linear_qkvz": "GDN Fused QKVZ Projection",
            "linear_beta_alpha": "GDN Beta/Alpha Gate Projection",
        }
        return (
            lin_roles.get(suffix, f"Linear Attention Projection ({suffix})"),
            mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING,
            "vllm-ascend/vllm_ascend/models/kimi_k3/kda.py", False, "", suffix,
        )
    return (
        f"Linear Attention Layout / Copy ({suffix})", "TensorMove / Slice", BACKEND_CANN,
        KERNEL_STATUS_EXISTING, kda_src, False, "", suffix,
    )


def _classify_moe_tuple(
    suffix: str,
    fam: str,
    rec: str,
    meta: Mapping[str, Any],
    nat_kind: str,
    is_ideal: bool,
    is_quant_mm: bool,
    is_mxfp4_w: bool,
    mm_kernel: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    moe_src = "vllm-ascend/vllm_ascend/ops/fused_moe/fused_moe.py"
    if suffix == "router" or fam == "router":
        role = "Shared Expert Gate Linear" if "shared" in suffix else "MoE Router Gate Linear (W_router)"
        return (role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "moe_gating_topk":
        return (
            "MoE Sigmoid + Correction Bias TopK Gating", "MoeGatingTopK", BACKEND_CANN,
            KERNEL_STATUS_EXISTING, moe_src, False, "", "moe_gating_topk",
        )
    if suffix in {"latent_down", "latent_up"}:
        role = "MoE Routed Latent Down Projection (H->Hm)" if suffix == "latent_down" else "MoE Routed Latent Up Projection (Hm->H)"
        return (role, mm_kernel, BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "latent_norm":
        return ("MoE Routed Latent RMSNorm", "RmsNorm", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", "latent_norm")
    if suffix == "dispatch" or "dispatch" in rec:
        kern = nat_kind if (nat_kind and "Dispatch" in nat_kind) else "MoeDistributeDispatchV2"
        return (
            "MoE EP Token Distribute Dispatch (MC2)", kern, BACKEND_HCCL_MC2,
            KERNEL_STATUS_EXISTING, moe_src, False, "", "dispatch",
        )
    if suffix in {"unpack", "combine"} or "combine" in rec:
        kern = nat_kind if (nat_kind and "Combine" in nat_kind) else "MoeDistributeCombineV2"
        return (
            "MoE EP Token Distribute Combine (MC2)", kern, BACKEND_HCCL_MC2,
            KERNEL_STATUS_EXISTING, moe_src, False, "", "unpack",
        )
    if "megamoe" in rec:
        return (
            "Fused MegaMoE (Dispatch + GMM1 + Act + GMM2 + Combine)",
            "dispatch_ffn_combine (MegaMoE)",
            BACKEND_CANN,
            f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 MegaMoE Compact Tiling & AIV/Cube Ping-Pong",
            "vllm-ascend/vllm_ascend/ops/fused_moe/moe_comm_method.py",
            True,
            "Fused MegaMoE dispatch_ffn_combine supporting packed MXFP4 uint8 + K3 SiTU + compact tiling",
            "megamoe_arch35",
        )
    if suffix in {"expert_up", "expert_down"} or fam == "grouped_gemm":
        is_up = suffix == "expert_up" or "gate_up" in rec
        q_tag = "MXFP4" if is_mxfp4_w else ("W8A8" if is_quant_mm else "BF16")
        role = f"Routed Expert {'Gate+Up GMM1' if is_up else 'Down GMM2'} ({q_tag})"
        if is_ideal and is_mxfp4_w:
            return (
                role, "GroupedMatmul (MXFP4 Compact Tiling)", BACKEND_CANN,
                f"{KERNEL_STATUS_EXTEND_TILING} MXFP4 GroupedMatmul Compact Tiling",
                "cann/ops-transformer/src/mc2/grouped_matmul.cpp",
                True,
                f"MXFP4 group=32 compact tiling + AIV/Cube double-buffering for {role}",
                suffix,
            )
        return (role, "GroupedMatmul", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "expert_act" or suffix == "shared.situ":
        if fam == "situ" or "situ" in rec:
            return (
                "Routed Expert K3 SiTU+Mul Activation" if suffix == "expert_act" else "Shared Expert K3 SiTU+Mul Activation",
                "K3 SituAndMul",
                BACKEND_TRITON,
                f"{KERNEL_STATUS_NEW_OP} K3 SituAndMul(beta=4, linear_beta=25) + MXQuant",
                "vllm-ascend/vllm_ascend/models/kimi_k3/modeling_kimi_k3.py",
                True,
                "Fused K3 SituAndMul(beta=4, linear_beta=25) + optional DynamicMxQuant epilogue",
                suffix,
            )
        return ("Routed Expert SwiGLU Activation", "SwiGlu", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)
    if suffix == "join":
        return (
            "MoE Routed + Shared Residual Add", "aclnnAdd", BACKEND_CANN,
            KERNEL_STATUS_FRAMEWORK if is_ideal else KERNEL_STATUS_EXISTING,
            moe_src, False, "", "join",
        )
    if suffix.startswith("shared."):
        if fam in _COLLECTIVE_FAMILIES:
            return _classify_collective_tuple(suffix, fam, rec)
        s_roles = {
            "shared.gate_up": "Shared Expert Gate+Up Linear (MSO Stream 1)",
            "shared.down": "Shared Expert Down Linear (MSO Stream 1)",
            "shared.swiglu": "Shared Expert SwiGLU (MSO Stream 1)",
        }
        kern = "SwiGlu" if "swiglu" in suffix else mm_kernel
        st = f"{KERNEL_STATUS_FRAMEWORK} (MSO Stream 1 Overlap)" if is_ideal else KERNEL_STATUS_EXISTING
        return (s_roles.get(suffix, f"Shared Expert ({suffix})"), kern, BACKEND_CANN, st, moe_src, False, "", suffix)
    return (f"MoE Aux ({suffix})", "aclnnAdd / VectorOp", BACKEND_CANN, KERNEL_STATUS_EXISTING, moe_src, False, "", suffix)


def _classify_collective_tuple(
    suffix: str, fam: str, rec: str,
) -> tuple[str, str, str, str, str, bool, str, str]:
    hccl_src = "vllm-ascend/vllm_ascend/distributed/parallel_state.py"
    if fam == "allgather" or "allgather" in rec:
        role = "DCP Query AllGather" if "dcp_query" in suffix else (
            "Sequence-Parallel AllGather" if "sp_gather" in suffix or "sp_output" in suffix else (
                "Vocab Logits AllGather" if "logits" in suffix else f"HCCL AllGather ({suffix})"
            )
        )
        return (role, "hcom_allGather_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    if fam in {"reduce_scatter", "reducescatter"} or "reduce_scatter" in rec:
        role = "Sequence-Parallel ReduceScatter" if ("sp_scatter" in suffix or "reduce_scatter" in suffix) else f"HCCL ReduceScatter ({suffix})"
        return (role, "hcom_reduceScatter_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    if fam in {"alltoall", "alltoallv"} or "alltoall" in rec:
        role = "DCP Output AllToAllV" if "dcp_output" in suffix else f"HCCL AllToAllV ({suffix})"
        return (role, "hcom_allToAllV_", BACKEND_HCCL_MC2, KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix)
    return (
        f"TP Sum AllReduce ({suffix})", "hcom_allReduce_", BACKEND_HCCL_MC2,
        KERNEL_STATUS_EXISTING, hccl_src, False, "", suffix,
    )


def _clean_text(raw: Any) -> str:
    text = str(raw or "").strip().strip('"').strip()
    return "" if text.upper() in {"N/A", "NA", "NONE", "NULL"} else text


def _parse_float(raw: str, label: str) -> float:
    cleaned = _clean_text(raw)
    if not cleaned:
        raise ModelingError(f"kernel_csv_missing_{label}")
    try:
        value = float(cleaned)
    except ValueError as exc:
        raise ModelingError(f"kernel_csv_invalid_{label}:{raw!r}") from exc
    if not math.isfinite(value):
        raise ModelingError(f"kernel_csv_nonfinite_{label}:{raw!r}")
    return value


def load_kernel_details_csv(path: str | Path) -> tuple[KernelObservationRow, ...]:
    """Stream an Ascend ``msprof`` ``kernel_details.csv`` into typed observation rows."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"kernel_details_csv_not_found:{target}")
    rows: list[KernelObservationRow] = []
    with target.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ModelingError(f"kernel_csv_empty_header:{target.name}")
        cols = {str(key).strip().lower(): key for key in reader.fieldnames}
        selected = {
            field: next((cols[alias] for alias in aliases if alias in cols), None)
            for field, aliases in _CSV_COLUMN_ALIASES.items()
        }
        if not selected["name"] or not selected["duration_us"]:
            raise ModelingError(f"kernel_csv_missing_required_columns:{target.name}")
        for idx, raw in enumerate(reader):
            name = _clean_text(raw.get(selected["name"]))
            if not name:
                continue
            dur_us = _parse_float(raw.get(selected["duration_us"], ""), "duration_us")
            if dur_us <= 0.0:
                continue
            start_raw = raw.get(selected["start_us"], "") if selected["start_us"] else ""
            start_us = _parse_float(start_raw, "start_us") if _clean_text(start_raw) else float(idx)
            rows.append(KernelObservationRow(
                index=idx,
                name=name,
                op_type=_clean_text(raw.get(selected["op_type"])) if selected["op_type"] else "",
                core=_clean_text(raw.get(selected["core"])) if selected["core"] else "",
                start_us=start_us,
                duration_us=dur_us,
                stream_id=_clean_text(raw.get(selected["stream_id"])) if selected["stream_id"] else "",
                task_id=_clean_text(raw.get(selected["task_id"])) if selected["task_id"] else "",
                input_shapes=_clean_text(raw.get(selected["input_shapes"])) if selected["input_shapes"] else "",
                input_dtypes=_clean_text(raw.get(selected["input_dtypes"])) if selected["input_dtypes"] else "",
            ))
    rows.sort(key=lambda item: (item.start_us, item.index))
    return tuple(rows)


def load_observation_bundle(
    path: str | Path,
) -> tuple[tuple[KernelObservationRow, ...], dict[str, tuple[str, ...]]]:
    """Load an observation JSON bundle or ``kernel_details.csv`` with optional 1-to-N map."""
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"observation_bundle_not_found:{target}")
    if target.suffix.lower() == ".csv":
        return load_kernel_details_csv(target), {}
    if target.stat().st_size > MAX_OBSERVATION_JSON_BYTES:
        raise ModelingError("observation_json_exceeds_50_mib_limit")
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ModelingError("observation_bundle_must_be_mapping")
    kernels: list[KernelObservationRow] = []
    explicit_map: dict[str, tuple[str, ...]] = {}
    if payload.get("kernel_csv"):
        csv_path = Path(str(payload["kernel_csv"]))
        if not csv_path.is_absolute():
            csv_path = (target.parent / csv_path).resolve()
        kernels.extend(load_kernel_details_csv(csv_path))
    for idx, entry in enumerate(payload.get("kernels", ()) or ()):
        if not isinstance(entry, Mapping):
            continue
        dur = float(entry.get("duration_us", 0.0) or 0.0)
        if dur <= 0.0 or not math.isfinite(dur):
            continue
        kernels.append(KernelObservationRow(
            index=len(kernels),
            name=str(entry.get("name", f"kernel_{idx}")),
            op_type=str(entry.get("op_type", entry.get("type", ""))),
            core=str(entry.get("core", "")),
            start_us=float(entry.get("start_us", idx) or idx),
            duration_us=dur,
            stream_id=str(entry.get("stream_id", "")),
            task_id=str(entry.get("task_id", "")),
            input_shapes=str(entry.get("input_shapes", "")),
            input_dtypes=str(entry.get("input_dtypes", "")),
        ))
    for group in (payload.get("mappings") or payload.get("groups") or ()):
        if not isinstance(group, Mapping):
            continue
        sim_id = str(group.get("sim_op_id") or group.get("op_id") or "").strip()
        if not sim_id:
            continue
        names: list[str] = []
        for k_entry in group.get("kernels", ()) or ():
            if isinstance(k_entry, str):
                names.append(k_entry)
            elif isinstance(k_entry, Mapping):
                kname = str(k_entry.get("name", "")).strip()
                kdur = float(k_entry.get("duration_us", 0.0) or 0.0)
                if kname and kdur > 0.0:
                    names.append(kname)
                    kernels.append(KernelObservationRow(
                        index=len(kernels), name=kname,
                        op_type=str(k_entry.get("op_type", "")),
                        core=str(k_entry.get("core", "")),
                        start_us=float(k_entry.get("start_us", len(kernels)) or len(kernels)),
                        duration_us=kdur,
                        stream_id=str(k_entry.get("stream_id", "")),
                        task_id=str(k_entry.get("task_id", "")),
                        input_shapes=str(k_entry.get("input_shapes", "")),
                        input_dtypes=str(k_entry.get("input_dtypes", "")),
                    ))
        if names:
            explicit_map[sim_id] = tuple(names)
    return tuple(kernels), explicit_map


def _operator_match_tokens(op: OperatorExportRow) -> tuple[str, ...]:
    tokens: list[str] = []
    suffix = op.op_id.rsplit(".", 1)[-1].lower()
    if suffix and len(suffix) >= MIN_MATCH_TOKEN_LEN:
        tokens.append(suffix)
    for source in (op.family, op.recipe, op.ascend_kernel or op.implementation, *op.absorbed_source_ops):
        lower = source.strip().lower()
        if lower and len(lower) >= MIN_MATCH_TOKEN_LEN:
            tokens.append(lower)
            for hint in _FUSED_KERNEL_HINTS.get(lower, ()):
                tokens.append(hint)
    return tuple(dict.fromkeys(tokens))


def _kernel_matches_tokens(kernel: KernelObservationRow, tokens: Sequence[str]) -> bool:
    haystack = f"{kernel.name} {kernel.op_type}".lower()
    return any(tok in haystack for tok in tokens)


def reconcile_operators_with_kernels(
    operators: Sequence[OperatorExportRow],
    kernels: Sequence[KernelObservationRow],
    *,
    explicit_map: Mapping[str, Sequence[str]] | None = None,
    source_label: str = "kernel_details.csv",
) -> tuple[dict[str, ReconciledKernelGroup], ReconciliationSummary]:
    """Perform sample-conserving 1-to-N reconciliation of fused ``sim`` ops to kernels."""
    consumed: set[int] = set()
    groups: dict[str, ReconciledKernelGroup] = {}
    explicit = dict(explicit_map or {})
    by_name: dict[str, list[KernelObservationRow]] = defaultdict(list)
    for row in kernels:
        by_name[row.name].append(row)

    for op in operators:
        wanted = explicit.get(op.op_id)
        if not wanted:
            continue
        matched: list[KernelObservationRow] = []
        for pattern in wanted:
            chosen = next((k for k in by_name.get(pattern, ()) if k.index not in consumed), None)
            if chosen is None:
                chosen = next((k for k in kernels if k.index not in consumed and pattern.lower() in k.name.lower()), None)
            if chosen is not None:
                consumed.add(chosen.index)
                matched.append(chosen)
        if matched:
            groups[op.op_id] = _build_reconciled_group(op, tuple(matched))

    cursor = 0
    for op in operators:
        if op.op_id in groups:
            continue
        tokens = _operator_match_tokens(op)
        matched = []
        scan_idx = cursor
        max_group = max(1, len(op.absorbed_source_ops) + 1)
        while scan_idx < len(kernels) and len(matched) < max_group:
            cand = kernels[scan_idx]
            scan_idx += 1
            if cand.index in consumed:
                continue
            if _kernel_matches_tokens(cand, tokens):
                consumed.add(cand.index)
                matched.append(cand)
                cursor = scan_idx
            elif matched:
                break
        if matched:
            groups[op.op_id] = _build_reconciled_group(op, tuple(matched))

    unmatched = tuple(k for k in kernels if k.index not in consumed)
    apes = sorted(g.ape_pct for g in groups.values() if g.ape_pct is not None and math.isfinite(g.ape_pct))
    med_ape = float(median(apes)) if apes else None
    p90_ape = _nearest_rank_percentile(apes, P90_QUANTILE) if apes else None
    within_count = sum(1 for g in groups.values() if g.within_gate is True)
    within_pct = (100.0 * within_count / len(groups)) if groups else None
    matched_sim_sum = sum(op.dur_us for op in operators if op.op_id in groups)
    matched_meas_sum = sum(g.measured_sum_us for g in groups.values() if g.measured_sum_us is not None)
    summary = ReconciliationSummary(
        source_label=source_label,
        total_sim_ops=len(operators),
        matched_sim_ops=len(groups),
        total_kernels=len(kernels),
        matched_kernels=len(consumed),
        unmatched_kernels=unmatched,
        matched_sim_sum_us=matched_sim_sum,
        matched_measured_sum_us=matched_meas_sum,
        unmatched_measured_us=sum(k.duration_us for k in unmatched),
        median_ape_pct=med_ape,
        p90_ape_pct=p90_ape,
        within_gate_count=within_count,
        within_gate_pct=within_pct,
    )
    return groups, summary


def _build_reconciled_group(
    op: OperatorExportRow, matched: tuple[KernelObservationRow, ...],
) -> ReconciledKernelGroup:
    meas_sum = sum(k.duration_us for k in matched)
    if meas_sum <= 0.0:
        return ReconciledKernelGroup(op.op_id, matched, None, None, None)
    abs_err = abs(op.dur_us - meas_sum)
    ape_pct = 100.0 * abs_err / meas_sum
    gate_limit = max((RECONCILE_APE_THRESHOLD_PCT / 100.0) * meas_sum, RECONCILE_ABS_FLOOR_US)
    return ReconciledKernelGroup(
        sim_op_id=op.op_id,
        kernels=matched,
        measured_sum_us=meas_sum,
        ape_pct=ape_pct,
        within_gate=abs_err <= gate_limit,
    )


def _nearest_rank_percentile(sorted_vals: Sequence[float], frac: float) -> float:
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, math.ceil(frac * len(sorted_vals)) - 1))
    return float(sorted_vals[idx])


def _die_peaks_for_stage(case: Case, stage: int, replica: int = 0) -> dict[str, Any]:
    peaks = extract_hardware_peaks(case, stage=stage, replica=replica)
    die = peaks["die"]
    dpath = datapath_for_die(die)
    hbm_peak = float(peaks["hbm_peak_bytes_per_s"])
    hbm_r = max(float(dpath.hbm_read_bytes_per_s), hbm_peak)
    hbm_w = max(float(dpath.hbm_write_bytes_per_s), hbm_peak)
    return {
        **peaks,
        "hbm_read_bytes_per_s": hbm_r,
        "hbm_write_bytes_per_s": hbm_w,
        "hbm_rw_alpha": float(dpath.hbm_rw_alpha),
    }


def _die_cube_peak_rate(die: Any, dtype: str, default_dtype: str = "bf16") -> float:
    for candidate in (dtype, default_dtype, "bf16", "fp16"):
        if not candidate:
            continue
        try:
            return float(die.peak(str(candidate).strip().lower()))
        except ModelingError:
            continue
    return 0.0


def _op_lean_resource_demands_ns(
    *,
    flops: float,
    vector_ops: float,
    hbm_read: float,
    hbm_write: float,
    link_bytes: float,
    dtype: str,
    peaks: Mapping[str, Any],
) -> tuple[tuple[int, int, int, int, int], float]:
    default_dt = str(peaks.get("default_dtype") or "bf16")
    cube_rate = _die_cube_peak_rate(peaks["die"], dtype, default_dt)
    vec_rate = float(peaks["vector_peak_ops_per_s"] or 0.0)
    hbm_r_rate = float(peaks["hbm_read_bytes_per_s"] or 0.0)
    hbm_w_rate = float(peaks["hbm_write_bytes_per_s"] or 0.0)
    link_rate = float(peaks["link_peak_bytes_per_s"] or 0.0)
    alpha = float(peaks.get("hbm_rw_alpha", 0.0) or 0.0)

    def _demand_ns(work: float, rate: float) -> int:
        if work <= 0.0 or rate <= 0.0:
            return 0
        return inward_floor_ns(Fraction(work) * NS_PER_SECOND / Fraction(rate))

    d_cube = _demand_ns(flops, cube_rate)
    d_vec = _demand_ns(vector_ops, vec_rate)
    d_hbm_r = _demand_ns(hbm_read + alpha * hbm_write, hbm_r_rate)
    d_hbm_w = _demand_ns(hbm_write + alpha * hbm_read, hbm_w_rate)
    d_link = _demand_ns(link_bytes, link_rate)
    return (d_cube, d_vec, d_hbm_r, d_hbm_w, d_link), cube_rate


def _classify_dominant_bound(
    demands_ns: tuple[int, int, int, int, int], dur_us: float, link_bytes: float,
) -> tuple[str, float | None]:
    """Derive dominant hardware bound & roofline efficiency directly from physical demands."""
    d_cube, d_vec, d_hbm_r, d_hbm_w, d_link = demands_ns
    t_cube = float(d_cube) / NS_PER_US
    t_vec = float(d_vec) / NS_PER_US
    t_mem = float(max(d_hbm_r, d_hbm_w)) / NS_PER_US
    t_link = float(d_link) / NS_PER_US
    best_work = max(t_cube, t_vec, t_mem, t_link)
    eff = min(1.0, best_work / dur_us) if dur_us > 0.0 and best_work > 0.0 else None
    if link_bytes > 0.0 and t_link >= max(t_cube, t_vec, t_mem):
        return "link_bound", eff
    if dur_us > 0.0 and dur_us <= LAUNCH_BOUND_SHORT_US and best_work < dur_us * LAUNCH_BOUND_WORK_SHARE:
        return "launch_bound", eff
    if best_work <= 0.0:
        return "launch_bound", eff
    pairs = (
        ("memory_bound", t_mem),
        ("cube_bound", t_cube),
        ("vector_bound", t_vec),
        ("link_bound", t_link),
    )
    return max(pairs, key=lambda item: item[1])[0], eff


def _hbm_read_write_l2(spec: OpSpec | None) -> tuple[float, float, float, float, float, float, float]:
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    mte2 = float((spec.mte2_bytes if spec is not None else 0.0) or 0.0)
    mte3 = float((spec.mte3_bytes if spec is not None else 0.0) or 0.0)
    fixpipe = float(meta.get("fixpipe_bytes", 0.0) or 0.0)
    hbm_total = float(spec.hbm_bytes if spec is not None else 0.0)
    raw_write = mte3 + fixpipe
    copy_total = mte2 + raw_write
    if "l2_hit_rate" in meta and meta["l2_hit_rate"] is not None:
        l2_hit = max(0.0, min(1.0, float(meta["l2_hit_rate"])))
    elif copy_total > 0.0 and hbm_total > 0.0 and hbm_total < copy_total:
        l2_hit = max(0.0, min(1.0, 1.0 - hbm_total / copy_total))
    else:
        l2_hit = 0.0
    if copy_total > 0.0 and hbm_total > 0.0:
        scale = hbm_total / copy_total
        hbm_read, hbm_write = mte2 * scale, raw_write * scale
    elif "read_bytes" in meta:
        hbm_read = float(meta["read_bytes"] or 0.0)
        hbm_write = max(0.0, hbm_total - hbm_read)
    else:
        hbm_read, hbm_write = hbm_total, 0.0
    return hbm_read, hbm_write, hbm_total, mte2, mte3, fixpipe, l2_hit * 100.0


def _resolve_shard_axis_and_rule(term: WeightTerm) -> tuple[str, int]:
    shard = str(term.shard or "replicated")
    role = str(term.placement_role or "")
    axis = 1 if (role == "attention_output" or shard in {"in_tp", "dense_down", "expert_down", "group_in"}) else 0
    if shard in {"expert_gate_up", "expert_down"}:
        return "EP", axis
    if shard == "replicated":
        return "Replicated", axis
    if shard == "node":
        return "Node", axis
    if role == "attention_output":
        return "Local-TP (o_proj)", axis
    if role == "attention_query":
        return "Local-TP (q_proj)", axis
    if term.category == "shared_expert":
        return "Shared-TP", axis
    return f"TP ({shard})", axis


def _stage_layer_counts(case: Case) -> tuple[int, ...]:
    p = case.runtime.parallel
    if case.runtime.pp_layer_partition:
        return tuple(int(x) for x in case.runtime.pp_layer_partition)
    return tuple(
        case.model.layers // p.pp + (1 if stage < case.model.layers % p.pp else 0)
        for stage in range(p.pp)
    )


def _compute_cache_ledger_rows(
    case: Case,
    *,
    stage_layers: Sequence[int],
    cache_dtype: str,
    seq_len: int,
    concurrency: int,
    block_size: int,
    spec_tokens: int,
) -> tuple[int, tuple[CacheLedgerRow, ...], list[float]]:
    model = case.model
    p = case.runtime.parallel
    dcp = max(1, int(p.cp))
    local_tokens = math.ceil(max(1, seq_len) / (max(1, block_size) * dcp)) * max(1, block_size)
    layer_stage = [s for s, count in enumerate(stage_layers) for _ in range(count)]
    comp_stage_bytes: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(stage_layers))
    comp_stage_layers: dict[str, list[int]] = defaultdict(lambda: [0] * len(stage_layers))
    comp_scalable: dict[str, bool] = {}

    scale_override = _cache_dtype_override_ratio(model, cache_dtype)
    for lp in model.layer_plans:
        stage = layer_stage[lp.index] if lp.index < len(layer_stage) else 0
        raw_comps = layer_cache_components(
            model, lp, p, max(1, seq_len),
            capacity_padded=True,
            interleave=INDEX_INTERLEAVE_TOKENS,
            runtime=case.runtime,
            query_tokens=max(1, 1 + spec_tokens),
        )
        for comp_name, val in raw_comps.items():
            if val <= 0.0:
                continue
            scalable = comp_name not in {"recurrent_state", "conv_state", "compressor_state"}
            comp_scalable[comp_name] = scalable
            adjusted = float(val) * (scale_override if scalable else 1.0)
            comp_stage_bytes[comp_name][stage] += adjusted
            comp_stage_layers[comp_name][stage] += 1

    stage_totals = [0.0] * len(stage_layers)
    rows: list[CacheLedgerRow] = []
    for comp_name in sorted(comp_stage_bytes):
        per_stage = comp_stage_bytes[comp_name]
        layers_per_stage = tuple(comp_stage_layers[comp_name])
        max_layers = max(layers_per_stage, default=0)
        max_req_bytes = max(per_stage, default=0.0)
        scalable = comp_scalable.get(comp_name, True)
        b_per_tok = (
            max_req_bytes / float(max(1, max_layers * local_tokens))
            if (scalable and max_layers > 0) else 0.0
        )
        for st_idx, st_val in enumerate(per_stage):
            stage_totals[st_idx] += st_val * float(concurrency)
        rows.append(CacheLedgerRow(
            component=comp_name,
            cache_kind=cache_dtype if scalable else "State (Fixed)",
            active_layers_per_stage=layers_per_stage,
            max_stage_layers=max_layers,
            bytes_per_token_per_layer=b_per_tok,
            per_request_bytes=max_req_bytes,
            batch_total_bytes=max_req_bytes * float(concurrency),
            scales_with_tokens=scalable,
        ))
    return local_tokens, tuple(rows), stage_totals


def _cache_dtype_override_ratio(model: Any, target_dtype: str) -> float:
    base = _normalize_cache_dtype(model)
    target = str(target_dtype or base).strip().upper()
    if target == base:
        return 1.0
    width_by_dtype = {
        "BF16": float(BF16_BYTES),
        "C8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
        "FP8": 1.0 + float(FP32_BYTES) / float(KV_QUANT_GROUP_SIZE),
    }
    base_w = width_by_dtype.get(base, float(BF16_BYTES))
    target_w = width_by_dtype.get(target, float(BF16_BYTES))
    return target_w / max(base_w, MIN_DENOM_EPS)


def _resolve_term_quant_format(
    term: WeightTerm,
    model_quant: QuantSpec | None,
    resolved_wq: str,
    *,
    explicit_override: bool,
) -> tuple[str, bool, str]:
    """Respect ``QuantSpec.exclusions`` per weight row (Item 6.1)."""
    cat = str(term.category or "")
    fam = cat.split(".", 1)[0]
    unquantized_cats = {"norm", "embedding", "lm_head", "param_fp32", "engram_table", "conv_weight_f32", "a_log", "dt_bias"}
    if fam in {"param_fp32", "conv_weight_f32", "a_log", "dt_bias"}:
        return "FP32", True, "FP32 param"
    if fam in unquantized_cats or cat in unquantized_cats:
        return "BF16", True, "quantization_config.ignore -> BF16"

    term_quant = term.runtime_quant or term.quant or model_quant
    if term_quant is not None:
        exclusions = set(getattr(term_quant, "exclusions", ()) or ())
        if cat in exclusions or fam in exclusions:
            native = _QUANT_FORMAT_DISPLAY.get(str(term_quant.native_format or "bf16").lower(), "BF16")
            return native, True, "quantization_config.ignore -> BF16"
        if not explicit_override:
            eff_raw = str(term_quant.format_for(cat) or "bf16").lower()
            mapped = _QUANT_FORMAT_DISPLAY.get(eff_raw, resolved_wq)
            note = "MXFP4 (group=32)" if mapped == "MXFP4" else mapped
            return mapped, (mapped == "BF16"), note
    return resolved_wq, (resolved_wq == "BF16"), ("MXFP4 (group=32)" if resolved_wq == "MXFP4" else resolved_wq)


def _collect_weight_ledger_rows(
    case: Case,
    mem: Memory,
    stage_layers: Sequence[int],
    resolved_wq: str,
    *,
    explicit_override: bool,
) -> tuple[tuple[WeightLedgerRow, ...], list[float], list[float], list[float]]:
    p = case.runtime.parallel
    stage_terms = stage_weight_terms(
        case.model, p,
        include_draft=bool(mem.draft_kind),
        include_engram=True,
        include_engram_tables=True,
        layer_partition=stage_layers,
        draft_kind=mem.draft_kind or None,
        slot_plans=mem.slot_plans,
    )
    engram_names = {t.name for t in engram_table_terms(case.model)}
    grouped: dict[tuple[str, str, int, int, str, bool], dict[str, Any]] = {}
    for stage_idx, terms in enumerate(stage_terms):
        for term in terms:
            if not term.runtime_owned:
                continue
            local_r, local_c, local_cnt = _sharded_shape(term, p, case.machine.ranks_per_node)
            shard_rule, axis = _resolve_shard_axis_and_rule(term)
            is_engram = (
                term.category in _ENGRAM_CATEGORIES
                or term.name in engram_names
                or "engram_table" in term.name.lower()
                or "ple_table" in term.name.lower()
            )
            key = (term.category, term.name, int(term.rows), int(term.cols), shard_rule, is_engram)
            entry = grouped.get(key)
            if entry is None:
                quant_fmt, is_excl, q_note = _resolve_term_quant_format(
                    term, case.model.quant, resolved_wq, explicit_override=explicit_override,
                )
                entry = {
                    "category": term.category,
                    "name": term.name,
                    "global_rows": int(term.rows),
                    "global_cols": int(term.cols),
                    "shard_rule": shard_rule,
                    "axis": axis,
                    "local_rows": int(local_r),
                    "local_cols": int(local_c),
                    "stage_repeats": [0] * p.pp,
                    "is_engram": is_engram,
                    "quant": quant_fmt,
                    "is_excluded": is_excl,
                    "quant_note": q_note,
                }
                grouped[key] = entry
            entry["stage_repeats"][stage_idx] += max(1, int(local_cnt))

    weight_rows: list[WeightLedgerRow] = []
    stage_w_payload = [0.0] * p.pp
    stage_w_scale = [0.0] * p.pp
    stage_engram_total = [0.0] * p.pp
    for entry in grouped.values():
        payload_1, scale_1, aux_1, total_1 = evaluate_weight_storage_bytes(
            entry["local_rows"], entry["local_cols"], entry["quant"],
        )
        repeats = tuple(entry["stage_repeats"])
        max_rep = max(repeats, default=0)
        for st_idx, rep in enumerate(repeats):
            if rep <= 0:
                continue
            if entry["is_engram"]:
                stage_engram_total[st_idx] += float(total_1 * rep)
            else:
                stage_w_payload[st_idx] += float(payload_1 * rep)
                stage_w_scale[st_idx] += float((scale_1 + aux_1) * rep)
        weight_rows.append(WeightLedgerRow(
            group=_weight_group_label(entry["category"]),
            name=entry["name"],
            category=entry["category"],
            global_rows=entry["global_rows"],
            global_cols=entry["global_cols"],
            shard_rule=entry["shard_rule"],
            shard_axis=entry["axis"],
            local_rows=entry["local_rows"],
            local_cols=entry["local_cols"],
            stage_repeats=repeats,
            max_stage_repeats=max_rep,
            quant_format=entry["quant"],
            payload_bytes=payload_1 * max_rep,
            scale_bytes=scale_1 * max_rep,
            aux_bytes=aux_1 * max_rep,
            per_die_bytes=total_1 * max_rep,
            is_engram_or_ple=entry["is_engram"],
            is_quant_excluded=entry["is_excluded"],
            quant_note=entry["quant_note"],
        ))
    weight_rows.sort(key=lambda r: (r.group, r.name))
    return tuple(weight_rows), stage_w_payload, stage_w_scale, stage_engram_total


def compute_memory_quant_ledger(
    case: Case,
    *,
    phase: str = "decode",
    seq_len: int | None = None,
    concurrency: int | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
) -> MemoryCalculatorLedger:
    """Derive closed-form per-die Weight/Quant/Cache/Engram-PLE/PP HBM budget ledger."""
    p = case.runtime.parallel
    stage_layers = _stage_layer_counts(case)
    resolved_seq = max(1, int(seq_len if seq_len is not None else case.runtime.max_model_len))
    resolved_conc = max(1, int(concurrency if concurrency is not None else max(1, len(case.workload.requests))))
    explicit_override = weight_quant is not None
    resolved_wq = (
        _canonicalize_weight_quant_override(weight_quant)
        if weight_quant is not None else _normalize_quant_preset(case.model.quant)
    )
    if resolved_wq not in WEIGHT_QUANT_FORMATS:
        raise ModelingError(f"unsupported_weight_quant:{resolved_wq}")
    wq_display, q_exclusions = _describe_quant_display(
        case.model.quant, resolved_wq, explicit_override=explicit_override,
    )
    resolved_cd = (
        _canonicalize_cache_dtype_override(cache_dtype)
        if cache_dtype is not None else _normalize_cache_dtype(case.model)
    )
    if resolved_cd not in SUPPORTED_CACHE_DTYPES:
        raise ModelingError(f"unsupported_cache_dtype:{resolved_cd}")
    resolved_offload = (
        bool(case.runtime.engram_cpu_offload)
        if engram_host_offload is None else bool(engram_host_offload)
    )
    block_size = int(getattr(case.runtime, "block_size", DEFAULT_CACHE_BLOCK_SIZE) or DEFAULT_CACHE_BLOCK_SIZE)
    spec_tokens = int(dict(case.runtime.metadata).get("draft_proposal_tokens", 0) or 0)

    mem = Memory(case)
    alloc = mem.allocation_report(resolved_seq)
    weight_rows, stage_w_payload, stage_w_scale, stage_engram_total = _collect_weight_ledger_rows(
        case, mem, stage_layers, resolved_wq, explicit_override=explicit_override,
    )
    local_tokens, cache_rows, stage_cache_bytes = _compute_cache_ledger_rows(
        case,
        stage_layers=stage_layers,
        cache_dtype=resolved_cd,
        seq_len=resolved_seq,
        concurrency=resolved_conc,
        block_size=block_size,
        spec_tokens=spec_tokens,
    )
    ws_by_stage = [float(x) for x in alloc["workspace"]["total_bytes_per_stage_rank"]]
    deq_by_stage = [float(x) for x in alloc["temporary_dequant"]["bytes_per_stage_rank"]]
    reserve_bytes = float(alloc["reserve"]["total_bytes_per_rank"])
    die_hbm = float(min(d.hbm_bytes for d in case.machine.dies))
    util = float(case.runtime.gpu_memory_utilization)
    usable_hbm = die_hbm * util

    stage_budgets: list[StageBudgetRow] = []
    for st in range(p.pp):
        engram_hbm = 0.0 if resolved_offload else stage_engram_total[st]
        engram_host = stage_engram_total[st] if resolved_offload else 0.0
        ws_peak = (ws_by_stage[st] if st < len(ws_by_stage) else 0.0) + (
            deq_by_stage[st] if st < len(deq_by_stage) else 0.0
        )
        total_hbm = (
            stage_w_payload[st] + stage_w_scale[st] + engram_hbm
            + stage_cache_bytes[st] + ws_peak + reserve_bytes
        )
        stage_budgets.append(StageBudgetRow(
            stage=st,
            owned_layers=stage_layers[st],
            weight_payload_bytes=stage_w_payload[st],
            weight_scale_aux_bytes=stage_w_scale[st],
            engram_ple_hbm_bytes=engram_hbm,
            engram_ple_host_bytes=engram_host,
            cache_bytes=stage_cache_bytes[st],
            workspace_peak_bytes=ws_peak,
            reserve_bytes=reserve_bytes,
            total_hbm_bytes=total_hbm,
            usable_hbm_budget_bytes=usable_hbm,
            headroom_bytes=usable_hbm - total_hbm,
            fits_hbm=total_hbm <= usable_hbm,
        ))

    local_tp = max(
        int(getattr(p, "effective_o_proj_tp", p.tp) or p.tp),
        int(getattr(p, "q_proj_tp", 1) or 1),
        int(getattr(p, "dcp_linear_tp", 1) or 1),
    )
    shared_dp = bool(dict(case.runtime.metadata).get("enable_shared_expert_dp", False))
    shared_tp = 1 if shared_dp else int(p.tp)
    sim_cache_by_stage = tuple(
        float(sum(comp["total"] for comp in entry["allocated"].values()))
        for entry in alloc["cache"]["components"]
    )
    return MemoryCalculatorLedger(
        model_name=str(case.model.name),
        hardware_sku=str(case.machine.sku or case.machine.name),
        phase=phase,
        weight_quant=resolved_wq,
        cache_dtype=resolved_cd,
        engram_host_offload=resolved_offload,
        dp=int(p.dp), tp=int(p.tp), ep=int(p.ep), pp=int(p.pp),
        dcp=int(p.cp), pcp=int(p.pcp), local_tp=int(local_tp), shared_tp=int(shared_tp),
        total_dies=len(case.machine.dies),
        seq_len=resolved_seq, concurrency=resolved_conc,
        block_size=block_size, spec_tokens=spec_tokens, local_cache_tokens=local_tokens,
        die_hbm_capacity_bytes=die_hbm, gpu_memory_utilization=util, usable_hbm_bytes=usable_hbm,
        weight_rows=weight_rows,
        cache_rows=cache_rows,
        stage_budgets=tuple(stage_budgets),
        max_stage_hbm_bytes=max((row.total_hbm_bytes for row in stage_budgets), default=0.0),
        max_stage_host_offload_bytes=max((row.engram_ple_host_bytes for row in stage_budgets), default=0.0),
        fits_hbm=all(row.fits_hbm for row in stage_budgets),
        sim_memory_weight_bytes_by_stage=tuple(float(x) for x in alloc["weights"]["total_bytes_per_stage_rank"]),
        sim_memory_cache_bytes_by_stage=sim_cache_by_stage,
        weight_quant_display=wq_display,
        quant_exclusions=q_exclusions,
    )


def _weight_group_label(category: str) -> str:
    lower = category.lower()
    if lower in _ENGRAM_CATEGORIES or "engram" in lower or "ple" in lower:
        return "Engram / PLE"
    if "expert" in lower or "router" in lower or "moe" in lower:
        return "MoE"
    if "mla" in lower or "indexer" in lower or "attn" in lower or "compressor" in lower:
        return "Attention / MLA"
    if "kda" in lower or "gdn" in lower or "conv" in lower or "linear" in lower:
        return "LinearAttn"
    if "mlp" in lower or "ffn" in lower:
        return "Dense FFN"
    if "vision" in lower or "projector" in lower or "image" in lower:
        return "Vision"
    if lower in {"embedding", "lm_head"}:
        return "Head / Embedding"
    return "Norm / Residual"


def _certify_step_lower_bound(
    op_rows: Sequence[OperatorExportRow],
    per_op_demands_ns: Sequence[tuple[int, int, int, int, int]],
) -> tuple[float, bool, tuple[str, ...]]:
    """Compute and verify full-model scaled Lean 4 ``L*`` certificate."""
    if not op_rows:
        return 0.0, True, ()
    res_pipes = ("cube", "vector", "hbm_read", "hbm_write", "link")
    die_coords = sorted({
        (row.occurrence.replica, row.occurrence.stage, int(row.occurrence.rank or 0))
        for row in op_rows
    })
    coord_to_offset = {coord: idx * CERT_RESOURCE_COUNT for idx, coord in enumerate(die_coords)}
    num_resources = len(die_coords) * CERT_RESOURCE_COUNT
    totals = [0] * num_resources
    resource_labels = [
        f"{pipe}@r{replica}.p{stage}.d{rank}"
        for replica, stage, rank in die_coords
        for pipe in res_pipes
    ]
    node_quanta: list[int] = []
    for row, demand in zip(op_rows, per_op_demands_ns):
        rep = max(1, int(row.repeat_count))
        coord = (row.occurrence.replica, row.occurrence.stage, int(row.occurrence.rank or 0))
        base = coord_to_offset[coord]
        for idx in range(CERT_RESOURCE_COUNT):
            totals[base + idx] += demand[idx] * rep
        node_quanta.append(max(demand) * rep)

    index_by_occ = {row.occurrence: idx for idx, row in enumerate(op_rows)}
    edges: list[DagEdge] = []
    for dst_idx, row in enumerate(op_rows):
        for pred in row.predecessors:
            src_idx = index_by_occ.get(pred)
            if src_idx is not None and src_idx < dst_idx:
                edges.append(DagEdge(src=src_idx, dst=dst_idx, wire_ns=0))
    prog = ResourceChoiceProgram(num_resources=num_resources, global_work=tuple(totals), site_modes=())
    dag = EventDagSpec(node_quantum_ns=tuple(node_quanta), edges=tuple(edges))
    cert, dual_report = certify_lower_bound(prog, dag)
    valid = check_optimality_certificate(prog, dag, cert)
    tight = tuple(
        resource_labels[int(tag[1:])]
        if tag.startswith("r") and tag[1:].isdigit() and int(tag[1:]) < len(resource_labels)
        else tag
        for tag in dual_report.tight_resources
    )
    return float(cert.l_star_lean_ns) / NS_PER_US, valid, tight


def _merge_intervals(intervals: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted((max(0.0, a), max(a, b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged: list[tuple[float, float]] = [ordered[0]]
    for lo, hi in ordered[1:]:
        prev_lo, prev_hi = merged[-1]
        if lo <= prev_hi:
            merged[-1] = (prev_lo, max(prev_hi, hi))
        else:
            merged.append((lo, hi))
    return merged


def _exposed_length_outside(
    interval: tuple[float, float], cover: Sequence[tuple[float, float]],
) -> float:
    lo, hi = interval
    if hi <= lo:
        return 0.0
    cur = lo
    exposed = 0.0
    for c_lo, c_hi in cover:
        if c_hi <= cur:
            continue
        if c_lo >= hi:
            break
        if c_lo > cur:
            exposed += min(hi, c_lo) - cur
        cur = max(cur, c_hi)
        if cur >= hi:
            return 0.0
    if cur < hi:
        exposed += hi - cur
    return exposed


def _assign_op_archetype(
    op_id: str,
    spec: OpSpec | None,
    archetypes_by_id: Mapping[str, LayerArchetypeSpec],
    target_arch_by_rep_layer: Mapping[int, LayerArchetypeSpec],
) -> LayerArchetypeSpec:
    """Assign an op to its exact owning Model Scope + Layer Archetype (never mixing Draft & Target)."""
    oid = str(op_id or "")
    layer_idx = extract_layer_index(oid, spec.layer if spec is not None else None)
    if oid.startswith("draft"):
        if ".context_kv" in oid:
            return archetypes_by_id.get("draft_context_kv") or next(iter(archetypes_by_id.values()))
        if ".s" in oid and layer_idx is not None:
            return archetypes_by_id.get("draft_arch_0") or next(iter(archetypes_by_id.values()))
        tail_tokens = ("final_norm", "logits", "markov")
        if any(tok in oid for tok in tail_tokens):
            return archetypes_by_id.get("draft_tail") or next(iter(archetypes_by_id.values()))
        return archetypes_by_id.get("draft_pre") or next(iter(archetypes_by_id.values()))

    if oid.startswith("final.") or oid in {"sp_output_gather", "final_norm", "logits", "logits_gather", "accept"}:
        return archetypes_by_id["target_tail"]
    if layer_idx is not None and layer_idx in target_arch_by_rep_layer:
        return target_arch_by_rep_layer[layer_idx]
    if oid in {"embedding", "sp_shard", "iteration_input"} or (layer_idx is None and "final" not in oid):
        return archetypes_by_id["target_embed"]
    return archetypes_by_id["target_tail"]


def _collapse_micro_split_ops(
    rows: list[OperatorExportRow],
    demands: list[tuple[int, int, int, int, int]],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]:
    """Collapse artificial simulation micro-splits (e.g. ``recurrent_scan0..7``) into one logical op."""
    out_rows: list[OperatorExportRow] = []
    out_demands: list[tuple[int, int, int, int, int]] = []
    group_pos: dict[tuple[str, str], int] = {}

    for row, dem in zip(rows, demands):
        match = _RECURRENT_MICRO_PATTERN.match(row.op_id)
        if match is None:
            out_rows.append(row)
            out_demands.append(dem)
            continue
        key = (row.archetype_id, f"{match.group(1)}.{match.group(2)}")
        prev_idx = group_pos.get(key)
        if prev_idx is None:
            group_pos[key] = len(out_rows)
            clean_id = f"{match.group(1)}.{match.group(2)}"
            out_rows.append(replace(row, op_id=clean_id))
            out_demands.append(dem)
            continue
        base = out_rows[prev_idx]
        b_dem = out_demands[prev_idx]
        merged_dem = tuple(b_dem[i] + dem[i] for i in range(CERT_RESOURCE_COUNT))
        new_dur = base.dur_us + row.dur_us
        new_l_star = float(max(merged_dem)) / NS_PER_US
        out_rows[prev_idx] = replace(
            base,
            start_us=min(base.start_us, row.start_us),
            end_us=max(base.end_us, row.end_us),
            dur_us=new_dur,
            exposed_dur_us=base.exposed_dur_us + row.exposed_dur_us,
            flops=base.flops + row.flops,
            vector_ops=base.vector_ops + row.vector_ops,
            hbm_read_bytes=base.hbm_read_bytes + row.hbm_read_bytes,
            hbm_write_bytes=base.hbm_write_bytes + row.hbm_write_bytes,
            hbm_total_bytes=base.hbm_total_bytes + row.hbm_total_bytes,
            mte2_bytes=base.mte2_bytes + row.mte2_bytes,
            mte3_bytes=base.mte3_bytes + row.mte3_bytes,
            fixpipe_bytes=base.fixpipe_bytes + row.fixpipe_bytes,
            link_bytes=base.link_bytes + row.link_bytes,
            l_star_lean_us=new_l_star,
            gap_us=new_dur - new_l_star,
            gap_ratio=(new_dur / new_l_star) if new_l_star > 0.0 else None,
        )
        out_demands[prev_idx] = merged_dem  # type: ignore[assignment]
    return out_rows, out_demands


def _fast_graph_id(ops: Sequence[OpSpec], model_name: str, phase: str) -> str:
    h = hashlib.sha256()
    h.update(f"{model_name}:{phase}:".encode("utf-8"))
    for op in ops:
        h.update(f"{op.op_id}|{op.family}|{op.recipe}|{op.local_shape}|".encode("utf-8"))
    return f"sha256:{h.hexdigest()[:GRAPH_DIGEST_HEX_LEN]}"


def _extract_structure_registry_view(
    eff_case: Case,
    exec_graphs: Mapping[int, Sequence[OpSpec]],
    step_req: StepRequest,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Run ``export_op_structure(..., native_evidence=True)`` on replica 0 as primary SoT."""
    if 0 not in exec_graphs:
        return {}, {}
    try:
        from mindie_sim.structure import export_op_structure, structure_inputs_from_graphs
        r0_graphs = {0: tuple(exec_graphs[0])}
        s_graphs, options = structure_inputs_from_graphs(
            eff_case,
            r0_graphs,
            phase=step_req.phase,
            tokens=step_req.tokens,
            context=step_req.context,
            sequences=step_req.sequences,
            scope=step_req.scope,
            replicas=[0],
            physical_graphs=r0_graphs,
        )
        report = export_op_structure(s_graphs, native_evidence=True, **options)
    except Exception:
        return {}, {}
    node_by_op: dict[str, dict[str, Any]] = {}
    for node in (*report.get("nodes", ()), *report.get("resource_operators", ())):
        if isinstance(node, Mapping) and node.get("op_id"):
            node_by_op[str(node["op_id"])] = dict(node)
    subgraph_by_op: dict[str, dict[str, Any]] = {}
    for sg in report.get("subgraphs", ()):
        if not isinstance(sg, Mapping):
            continue
        sg_dict = dict(sg)
        for nid in sg_dict.get("node_ids") or ():
            op_key = str(nid).split(":op:", 1)[-1] if ":op:" in str(nid) else str(nid).split(":", 1)[-1]
            subgraph_by_op.setdefault(op_key, sg_dict)
    return node_by_op, subgraph_by_op


def _evaluate_representative_step(
    spine: Any,
    rep_case: Case,
    step_req: StepRequest,
    replay: ReplayOptions,
) -> tuple[
    Case,
    list[dict[str, Any]],
    dict[tuple[int, str], OpSpec],
    dict[str, Sequence[Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    str,
]:
    """Single-pass lowering + ``graph_costs`` + replica-0 ``export_op_structure``."""
    base = spine.stacks.resolve_base(rep_case, LoweringRequest(step=step_req))
    rewritten = spine.rewrite.apply(base)
    eff_case = (
        consume_target_declaration(base.case, rewritten.target_graphs)
        if rewritten.target_graphs is not None else base.case
    )
    book = ReplayRecipeBook.from_mapping(eff_case.runtime.kernels, family_registry=spine.families)
    book.graph = rewritten
    evaluation = EvaluationContext.create(eff_case, book, None)
    graphs = {entry.replica: entry.ops for entry in rewritten.executable}
    ext_inputs = (
        {"analysis_input": 0.0}
        if any("analysis_input" in op.deps for ops in graphs.values() for op in ops)
        else None
    )
    report = graph_costs(
        eff_case,
        graphs,
        recipes=book,
        external_inputs=ext_inputs,
        include_submission=replay.include_submission,
        isolated=replay.isolated,
        trace_limit=replay.event_limit,
        compact=False,
        evaluation=evaluation,
        stream_plans=rewritten.streams or None,
        request_views=rewritten.request_views,
        padded_tokens=step_req.padded_tokens,
        sequence_layout=step_req.layout,
    )
    op_spec_by_key = {
        (entry.replica, op.op_id): op
        for entry in rewritten.executable
        for op in entry.ops
    }
    struct_by_op, subgraph_by_op = _extract_structure_registry_view(eff_case, graphs, step_req)
    sources_by_op: dict[str, Sequence[Any]] = {}
    r0_ops = graphs.get(0, ())
    graph_id = _fast_graph_id(r0_ops, str(eff_case.model.name), step_req.phase)
    timeline = list(report.get("timeline") or report.get("operators") or ())
    return eff_case, timeline, op_spec_by_key, sources_by_op, struct_by_op, subgraph_by_op, graph_id


def _timeline_to_raw_records(
    timeline: Sequence[Any],
    *,
    is_step_eval: bool,
    case: Case,
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
) -> list[dict[str, Any]]:
    """Normalize ``StepEvaluation.operators`` or ``graph_costs`` timeline rows (Replica 0)."""
    p = case.runtime.parallel
    width = p.tp * p.pcp
    records: list[dict[str, Any]] = []
    occ_by_node: dict[str, OpOccurrence] = {}

    if is_step_eval:
        for row in timeline:
            if row.occurrence.replica != 0:
                continue
            spec = op_spec_by_key.get((0, row.occurrence.op_id))
            records.append({
                "ordinal": row.ordinal,
                "occurrence": row.occurrence,
                "op_id": row.occurrence.op_id,
                "spec": spec,
                "family": row.family,
                "recipe": spec.recipe if spec is not None else "",
                "stream": row.stream,
                "shape": row.shape,
                "dtype": row.dtype or (spec.dtype if spec is not None else "") or case.model.compute_dtype or "bf16",
                "predecessors": row.predecessors,
                "start_us": max(0.0, float(row.start_ms or 0.0) * US_PER_MS),
                "end_us": max(0.0, float(row.end_ms or row.start_ms or 0.0) * US_PER_MS),
            })
        return records

    for row in timeline:
        rep = int(row.get("replica", 0))
        if rep != 0:
            continue
        oid = str(row.get("op_id", ""))
        spec = op_spec_by_key.get((0, oid))
        stage = int(spec.stage) if spec is not None else int(row.get("stage", 0))
        rank_idx = stage * width
        phys_rank = int(case.machine.dies[rank_idx].rank) if 0 <= rank_idx < len(case.machine.dies) else 0
        occ = OpOccurrence(replica=0, stage=stage, rank=phys_rank, op_id=oid)
        occ_by_node[f"r0:{oid}"] = occ
        if row.get("id"):
            occ_by_node[str(row["id"])] = occ
        start_us = max(0.0, float(row.get("start_ms", row.get("ready_ms", 0.0)) or 0.0) * US_PER_MS)
        dur_ms = float(row.get("duration_ms", 0.0) or 0.0)
        end_ms = row.get("complete_ms", row.get("end_ms"))
        end_us = max(start_us, float(end_ms) * US_PER_MS) if end_ms is not None else (start_us + max(0.0, dur_ms) * US_PER_MS)
        records.append({
            "ordinal": len(records),
            "occurrence": occ,
            "op_id": oid,
            "spec": spec,
            "family": str(spec.family if spec is not None else row.get("family", "")),
            "recipe": str(spec.recipe if spec is not None else row.get("recipe", "")),
            "stream": str((spec.stream if spec is not None else "") or row.get("stream", "") or "main"),
            "shape": tuple(spec.local_shape) if spec is not None else tuple(row.get("shape", ())),
            "dtype": str((spec.dtype if spec is not None else "") or row.get("dtype", "") or case.model.compute_dtype or "bf16"),
            "raw_preds": tuple(row.get("predecessors") or ()),
            "start_us": start_us,
            "end_us": end_us,
        })
    for rec in records:
        rec["predecessors"] = tuple(
            occ_by_node[p_id] for p_id in rec.pop("raw_preds", ()) if p_id in occ_by_node
        )
    return records


def _filter_barriers_and_rewire_preds(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Drop zero-work DAG bookkeeping barriers while preserving causal predecessor links."""
    barrier_preds: dict[OpOccurrence, tuple[OpOccurrence, ...]] = {}
    kept: list[dict[str, Any]] = []
    for rec in records:
        dur = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        is_barrier = str(rec["family"]).lower() == "barrier" and dur <= OVERLAP_EXPOSED_EPS_US
        if is_barrier:
            barrier_preds[rec["occurrence"]] = tuple(rec["predecessors"])
            continue
        rewired: list[OpOccurrence] = []
        for pred in rec["predecessors"]:
            if pred in barrier_preds:
                rewired.extend(barrier_preds[pred])
            else:
                rewired.append(pred)
        copy = dict(rec)
        copy["predecessors"] = tuple(dict.fromkeys(rewired))
        kept.append(copy)
    return kept


def _compute_stream_overlap_and_archetype_spans(
    rows: list[OperatorExportRow],
    archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], tuple[LayerArchetypeSpec, ...]]:
    """Compute MSO Stream-1 overlap exposure and exact single-layer archetype spans."""
    main_intervals_by_arch: dict[str, list[tuple[float, float]]] = defaultdict(list)
    all_intervals_by_arch: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        if row.dur_us <= 0.0:
            continue
        all_intervals_by_arch[row.archetype_id].append((row.start_us, row.end_us))
        if row.stream_index != 1:
            main_intervals_by_arch[row.archetype_id].append((row.start_us, row.end_us))

    merged_main = {aid: _merge_intervals(ivs) for aid, ivs in main_intervals_by_arch.items()}
    updated: list[OperatorExportRow] = []
    for row in rows:
        if row.stream_index == 1 and row.dur_us > 0.0:
            exp_us = _exposed_length_outside((row.start_us, row.end_us), merged_main.get(row.archetype_id, ()))
            overlapped = exp_us <= OVERLAP_EXPOSED_EPS_US
            eff_exp = 0.0 if overlapped else exp_us
            eff_gap = 0.0 if overlapped else max(0.0, eff_exp - row.l_star_lean_us)
            eff_ratio = None if overlapped else ((eff_exp / row.l_star_lean_us) if row.l_star_lean_us > 0.0 else None)
            b_disp = "Overlapped on MSO Stream 1 (0 exposed)" if overlapped else row.bound_display
            updated.append(replace(
                row,
                is_overlapped=overlapped,
                exposed_dur_us=eff_exp,
                gap_us=eff_gap,
                gap_ratio=eff_ratio,
                bound_display=b_disp,
            ))
        else:
            updated.append(replace(row, is_overlapped=False, exposed_dur_us=row.dur_us))

    arch_updated: list[LayerArchetypeSpec] = []
    for arch in archetypes:
        ivs = _merge_intervals(all_intervals_by_arch.get(arch.archetype_id, ()))
        span = sum(hi - lo for lo, hi in ivs)
        arch_updated.append(replace(arch, single_span_us=span))
    return updated, tuple(arch_updated)


def _reorder_rows_by_archetype(
    rows: list[OperatorExportRow],
    demands: list[tuple[int, int, int, int, int]],
    archetypes: Sequence[LayerArchetypeSpec],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]]]:
    """Order rows canonically by Archetype (re-attaching detached ``recover_ssm_commit``)."""
    arch_order = {arch.archetype_id: idx for idx, arch in enumerate(archetypes)}
    paired = list(zip(rows, demands))
    paired.sort(key=lambda item: (arch_order.get(item[0].archetype_id, ARCHETYPE_ORDER_FALLBACK), item[0].ordinal))
    reordered_rows: list[OperatorExportRow] = []
    reordered_demands: list[tuple[int, int, int, int, int]] = []
    for new_ord, (row, dem) in enumerate(paired):
        reordered_rows.append(replace(row, ordinal=new_ord))
        reordered_demands.append(dem)
    return reordered_rows, reordered_demands


def _compute_network_efficiency_summary(
    rows: Sequence[OperatorExportRow],
    step_span_us: float,
    peaks: Mapping[str, Any],
) -> tuple[float, float, float, float, float, dict[str, float], dict[str, dict[str, float]]]:
    """Compute 4D Roofline MFU/Vec/MBU/Link and bottleneck mix in one O(N) pass."""
    tot_flops = sum(r.flops * r.repeat_count for r in rows)
    tot_vec = sum(r.vector_ops * r.repeat_count for r in rows)
    tot_hbm = sum(r.hbm_total_bytes * r.repeat_count for r in rows)
    tot_link = sum(r.link_bytes * r.repeat_count for r in rows)
    step_s = max(step_span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    default_dt = str(peaks.get("default_dtype") or "bf16")
    cube_peak = _die_cube_peak_rate(peaks["die"], default_dt, default_dt)
    vec_peak = float(peaks.get("vector_peak_ops_per_s") or 0.0)
    hbm_peak = float(peaks.get("hbm_peak_bytes_per_s") or 0.0)
    link_peak = float(peaks.get("link_peak_bytes_per_s") or 0.0)

    cube_mfu = min(100.0, 100.0 * (tot_flops / step_s) / cube_peak) if cube_peak > 0.0 else 0.0
    vec_mfu = min(100.0, 100.0 * (tot_vec / step_s) / vec_peak) if vec_peak > 0.0 else 0.0
    hbm_mbu = min(100.0, 100.0 * (tot_hbm / step_s) / hbm_peak) if hbm_peak > 0.0 else 0.0
    link_util = min(100.0, 100.0 * (tot_link / step_s) / link_peak) if link_peak > 0.0 else 0.0

    by_bound_dur: dict[str, float] = defaultdict(float)
    by_bound_cnt: dict[str, int] = defaultdict(int)
    by_bound_eff: dict[str, list[float]] = defaultdict(list)
    exposed_us = {"memory": 0.0, "cube": 0.0, "vector": 0.0, "communication": 0.0}
    total_scaled_dur = 0.0
    for r in rows:
        scaled = r.exposed_dur_us * r.repeat_count
        total_scaled_dur += scaled
        by_bound_dur[r.dominant_bound] += scaled
        by_bound_cnt[r.dominant_bound] += r.repeat_count
        if r.roofline_efficiency is not None:
            by_bound_eff[r.dominant_bound].append(r.roofline_efficiency)
        if r.dominant_bound == "memory_bound":
            exposed_us["memory"] += scaled
        elif r.dominant_bound == "cube_bound":
            exposed_us["cube"] += scaled
        elif r.dominant_bound == "vector_bound":
            exposed_us["vector"] += scaled
        elif r.dominant_bound == "link_bound":
            exposed_us["communication"] += scaled

    bottleneck_mix: dict[str, dict[str, float]] = {}
    for b_key in ("memory_bound", "cube_bound", "vector_bound", "link_bound", "launch_bound"):
        dur_b = by_bound_dur.get(b_key, 0.0)
        effs = by_bound_eff.get(b_key, [])
        avg_eff = (sum(effs) / len(effs) * 100.0) if effs else 0.0
        bottleneck_mix[b_key] = {
            "count": float(by_bound_cnt.get(b_key, 0)),
            "duration_us": dur_b,
            "share_pct": (100.0 * dur_b / total_scaled_dur) if total_scaled_dur > 0.0 else 0.0,
            "roofline_eff_pct": avg_eff,
        }
    tot_l_star = sum(r.l_star_lean_us * r.repeat_count for r in rows)
    net_roof = min(100.0, 100.0 * tot_l_star / step_span_us) if step_span_us > 0.0 else 0.0
    return cube_mfu, vec_mfu, hbm_mbu, link_util, net_roof, exposed_us, bottleneck_mix


def _finalize_ranked_rows(
    prelim_rows: Sequence[OperatorExportRow],
    observations_path: str | Path | None,
) -> tuple[tuple[OperatorExportRow, ...], list[int], ReconciliationSummary | None]:
    rank_order = sorted(
        range(len(prelim_rows)),
        key=lambda i: (
            -(prelim_rows[i].gap_us * prelim_rows[i].repeat_count),
            -(prelim_rows[i].gap_ratio or 0.0),
            prelim_rows[i].ordinal,
        ),
    )
    rank_by_idx = {idx: pos + 1 for pos, idx in enumerate(rank_order)}
    reconciled: dict[str, ReconciledKernelGroup] = {}
    reconcile_summary: ReconciliationSummary | None = None
    if observations_path is not None:
        obs_kernels, explicit_map = load_observation_bundle(observations_path)
        reconciled, reconcile_summary = reconcile_operators_with_kernels(
            prelim_rows, obs_kernels,
            explicit_map=explicit_map,
            source_label=Path(observations_path).name,
        )
    final_rows = tuple(
        replace(
            item,
            gap_radar_rank=rank_by_idx[idx],
            measured_kernels=reconciled[item.op_id].kernels if item.op_id in reconciled else (),
            measured_sum_us=reconciled[item.op_id].measured_sum_us if item.op_id in reconciled else None,
            ape_pct=reconciled[item.op_id].ape_pct if item.op_id in reconciled else None,
        )
        for idx, item in enumerate(prelim_rows)
    )
    return final_rows, rank_order, reconcile_summary


def _resolve_op_submodule(
    rec: Mapping[str, Any],
    struct_node: Mapping[str, Any] | None,
    lp: Any | None,
    arch_scope: str,
) -> str:
    """Read ``export_structure`` ``submodule`` first, falling back to semantic metadata."""
    if isinstance(struct_node, Mapping):
        raw_sub = str(struct_node.get("submodule") or "").strip()
        if raw_sub and raw_sub.lower() not in {"other", "unknown", "none"}:
            return raw_sub
    return classify_submodule(
        rec["op_id"], rec["family"], rec["recipe"],
        layer_attention=getattr(lp, "attention", "mla" if arch_scope == "draft" else ""),
        layer_ffn=getattr(lp, "ffn", "dense" if arch_scope == "draft" else ""),
    )


def _build_prelim_operator_rows(
    filtered_records: Sequence[Mapping[str, Any]],
    *,
    eff_case: Case,
    archetypes: Sequence[LayerArchetypeSpec],
    rep_lp_by_idx: Mapping[int, Any],
    use_archetype_dedup: bool,
    sources_by_op: Mapping[str, Sequence[Any]],
    struct_by_op: Mapping[str, Mapping[str, Any]],
    subgraph_by_op: Mapping[str, Mapping[str, Any]],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[int, dict[str, Any]]]:
    archetypes_by_id = {a.archetype_id: a for a in archetypes}
    target_arch_by_rep_layer = (
        {a.rep_layer_index: a for a in archetypes if a.scope == "target" and a.rep_layer_index is not None}
        if use_archetype_dedup
        else {full_idx: a for a in archetypes if a.scope == "target" for full_idx in a.layer_indices}
    )
    peaks_by_stage: dict[int, dict[str, Any]] = {}
    prelim_rows: list[OperatorExportRow] = []
    demands_ns: list[tuple[int, int, int, int, int]] = []
    seen_full_dedup: set[tuple[str, str]] = set()
    for rec in filtered_records:
        occ: OpOccurrence = rec["occurrence"]
        spec: OpSpec | None = rec["spec"]
        if occ.stage not in peaks_by_stage:
            peaks_by_stage[occ.stage] = _die_peaks_for_stage(eff_case, occ.stage, 0)
        peaks = peaks_by_stage[occ.stage]
        arch = _assign_op_archetype(rec["op_id"], spec, archetypes_by_id, target_arch_by_rep_layer)
        lp = rep_lp_by_idx.get(arch.rep_layer_index) if (arch.scope == "target" and arch.rep_layer_index is not None) else None
        rep_count = arch.repeat_count if use_archetype_dedup else 1
        if use_archetype_dedup and arch.scope == "draft":
            dedup_key = (arch.archetype_id, _strip_op_prefix(rec["op_id"]))
            if dedup_key in seen_full_dedup:
                continue
            seen_full_dedup.add(dedup_key)

        s_node = struct_by_op.get(rec["op_id"])
        sg_info = subgraph_by_op.get(rec["op_id"])
        layer_idx = extract_layer_index(rec["op_id"], spec.layer if spec is not None else None)
        submodule = _resolve_op_submodule(rec, s_node, lp, arch.scope)
        stream_idx = _classify_stream_index(rec["stream"], rec["family"], rec["op_id"])
        dur_us = max(0.0, float(rec["end_us"]) - float(rec["start_us"]))
        flops = float((spec.flops if spec is not None else 0.0) or 0.0)
        vec_ops = float((spec.vector_ops if spec is not None else 0.0) or 0.0)
        link_bytes = float((spec.link_bytes if spec is not None else 0.0) or 0.0)
        hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit = _hbm_read_write_l2(spec)
        op_demand, cube_peak = _op_lean_resource_demands_ns(
            flops=flops, vector_ops=vec_ops, hbm_read=hbm_r, hbm_write=hbm_w,
            link_bytes=link_bytes, dtype=rec["dtype"], peaks=peaks,
        )
        l_star_us = float(max(op_demand)) / NS_PER_US
        dom_bound, roof_eff = _classify_dominant_bound(op_demand, dur_us, link_bytes)
        desc = _resolve_ascend_kernel_identity(
            rec["op_id"], spec, rec["family"], rec["recipe"], rec["shape"], rec["dtype"],
            eff_case, lp, sources_by_op.get(rec["op_id"], ()),
            struct_node=s_node, subgraph_info=sg_info,
        )
        meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
        absorbed = tuple(str(x) for x in (meta.get("absorbed_source_ops") or meta.get("fused_from") or ()))
        demands_ns.append(op_demand)
        prelim_rows.append(OperatorExportRow(
            ordinal=len(prelim_rows), occurrence=occ, op_id=rec["op_id"],
            layer=layer_idx, layer_label=arch.title, submodule=submodule,
            family=rec["family"], recipe=rec["recipe"], implementation=desc.ascend_kernel,
            stream=rec["stream"], stream_index=stream_idx,
            task_type=_classify_task_type(rec["family"], flops, vec_ops, link_bytes),
            shape=rec["shape"], dtype=rec["dtype"], predecessors=rec["predecessors"],
            start_us=float(rec["start_us"]), end_us=float(rec["end_us"]), dur_us=dur_us,
            flops=flops, vector_ops=vec_ops,
            hbm_read_bytes=hbm_r, hbm_write_bytes=hbm_w, hbm_total_bytes=hbm_tot,
            mte2_bytes=mte2, mte3_bytes=mte3, fixpipe_bytes=fixpipe,
            link_bytes=link_bytes, l2_hit_pct=l2_hit, dominant_bound=dom_bound,
            l_star_lean_us=l_star_us, gap_us=dur_us - l_star_us,
            gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
            gap_radar_rank=0, roofline_efficiency=roof_eff,
            absorbed_source_ops=absorbed,
            archetype_id=arch.archetype_id, archetype_title=arch.title,
            model_scope=arch.scope, repeat_count=rep_count,
            logical_role=desc.logical_role, ascend_kernel=desc.ascend_kernel,
            kernel_backend=desc.kernel_backend, kernel_status=desc.kernel_status,
            source_ref=desc.source_ref, tensor_contract=desc.tensor_contract,
            exposed_dur_us=dur_us, cube_peak_flops_per_s=cube_peak,
            bound_display=_BOUND_DISPLAY_LABELS.get(dom_bound, dom_bound),
            needs_new_kernel=desc.needs_new_kernel,
            new_kernel_spec=desc.new_kernel_spec,
            canonical_op_key=desc.canonical_op_key,
        ))
    return prelim_rows, demands_ns, peaks_by_stage


def build_export_bundle(
    case: CaseInput,
    *,
    request: StepRequest | None = None,
    step_eval: StepEvaluation | None = None,
    replay: ReplayOptions = ReplayOptions(),
    observations_path: str | Path | None = None,
    weight_quant: str | None = None,
    cache_dtype: str | None = None,
    engram_host_offload: bool | None = None,
    top_k_radar: int = DEFAULT_GAP_RADAR_TOP_K,
) -> ExportBundle:
    """Evaluate unique layer archetypes once on ``runtime.Engine`` into ``ExportBundle``."""
    parsed = case if isinstance(case, Case) else parse_case(dict(case))
    spec_tokens = int(dict(parsed.runtime.metadata).get("draft_proposal_tokens", 0) or 0)
    step_req = request or (
        step_eval.request if step_eval is not None else StepRequest(
            phase="decode",
            tokens=1 + spec_tokens,
            context=max(1, min(parsed.runtime.max_model_len, DEFAULT_EXPORT_CONTEXT_TOKENS)),
            sequences=max(1, len(parsed.workload.requests)),
            scope="model" if spec_tokens > 0 else "body",
        )
    )
    seq_count = (
        int(step_req.sequences)
        if isinstance(step_req.sequences, int)
        else max(1, len(step_req.sequences))
    )
    include_draft = (
        step_req.scope == "model"
        and spec_tokens > 0
        and step_req.tokens > 1
        and getattr(parsed.model, "auxiliary", None) is not None
    )
    archetypes, full_schedule, rep_lp_by_idx, total_target_layers, total_draft_layers = (
        _discover_model_archetypes(parsed, include_draft=include_draft)
    )
    use_archetype_dedup = (
        step_eval is None
        and observations_path is None
        and (len(rep_lp_by_idx) < len(parsed.model.layer_plans) or total_draft_layers > 1)
    )
    eval_case = (
        _build_archetype_representative_case(parsed, rep_lp_by_idx, allow_draft_dedup=True)
        if use_archetype_dedup else parsed
    )
    spine = default_spine()
    if step_eval is not None:
        prepared = spine.lower(eval_case, LoweringRequest(step=step_req))
        eff_case = prepared.case
        op_spec_by_key = {(e.replica, op.op_id): op for e in prepared.graph.executable for op in e.ops}
        exec_graphs = {e.replica: e.ops for e in prepared.graph.executable}
        struct_by_op, subgraph_by_op = _extract_structure_registry_view(eff_case, exec_graphs, step_req)
        raw_records = _timeline_to_raw_records(
            step_eval.operators, is_step_eval=True, case=eff_case, op_spec_by_key=op_spec_by_key,
        )
        sources_by_op: dict[str, Sequence[Any]] = {
            r.occurrence.op_id: r.sources for r in prepared.operators if r.occurrence.replica == 0
        }
        graph_id = step_eval.graph_id
        pipelines = step_eval.pipelines
    else:
        eff_case, timeline, op_spec_by_key, sources_by_op, struct_by_op, subgraph_by_op, graph_id = (
            _evaluate_representative_step(spine, eval_case, step_req, replay)
        )
        raw_records = _timeline_to_raw_records(
            timeline, is_step_eval=False, case=eff_case, op_spec_by_key=op_spec_by_key,
        )
        pipelines = ()

    filtered_records = _filter_barriers_and_rewire_preds(raw_records)
    prelim_rows, demands_ns, peaks_by_stage = _build_prelim_operator_rows(
        filtered_records,
        eff_case=eff_case,
        archetypes=archetypes,
        rep_lp_by_idx=rep_lp_by_idx,
        use_archetype_dedup=use_archetype_dedup,
        sources_by_op=sources_by_op,
        struct_by_op=struct_by_op,
        subgraph_by_op=subgraph_by_op,
    )
    prelim_rows, demands_ns = _collapse_micro_split_ops(prelim_rows, demands_ns)
    prelim_rows, archetypes = _compute_stream_overlap_and_archetype_spans(prelim_rows, archetypes)
    prelim_rows, demands_ns = _reorder_rows_by_archetype(prelim_rows, demands_ns, archetypes)
    final_rows, rank_order, reconcile_summary = _finalize_ranked_rows(prelim_rows, observations_path)

    return _assemble_export_bundle(
        parsed=parsed,
        eff_case=eff_case,
        step_req=step_req,
        step_eval=step_eval,
        seq_count=seq_count,
        graph_id=graph_id,
        pipelines=pipelines,
        final_rows=final_rows,
        demands_ns=demands_ns,
        rank_order=rank_order,
        reconcile_summary=reconcile_summary,
        archetypes=archetypes,
        full_schedule=full_schedule,
        total_target_layers=total_target_layers,
        total_draft_layers=total_draft_layers,
        use_archetype_dedup=use_archetype_dedup,
        peaks_by_stage=peaks_by_stage,
        weight_quant=weight_quant,
        cache_dtype=cache_dtype,
        engram_host_offload=engram_host_offload,
        top_k_radar=top_k_radar,
    )


def _assemble_export_bundle(
    *,
    parsed: Case,
    eff_case: Case,
    step_req: StepRequest,
    step_eval: StepEvaluation | None,
    seq_count: int,
    graph_id: str,
    pipelines: tuple[FusedPipelineDescriptor, ...],
    final_rows: tuple[OperatorExportRow, ...],
    demands_ns: Sequence[tuple[int, int, int, int, int]],
    rank_order: Sequence[int],
    reconcile_summary: ReconciliationSummary | None,
    archetypes: Sequence[LayerArchetypeSpec],
    full_schedule: Sequence[TraceScheduleEntry],
    total_target_layers: int,
    total_draft_layers: int,
    use_archetype_dedup: bool,
    peaks_by_stage: Mapping[int, dict[str, Any]],
    weight_quant: str | None,
    cache_dtype: str | None,
    engram_host_offload: bool | None,
    top_k_radar: int,
) -> ExportBundle:
    if use_archetype_dedup:
        active_arch_ids = {row.archetype_id for row in final_rows}
        step_span_us = sum(
            a.single_span_us * a.repeat_count for a in archetypes if a.archetype_id in active_arch_ids
        )
    elif step_eval is not None:
        step_span_us = float(step_eval.span_ms) * US_PER_MS
    else:
        step_span_us = max((r.end_us for r in final_rows), default=0.0)

    op_sum_us = sum(r.dur_us * r.repeat_count for r in final_rows)
    step_l_star_us, cert_valid, tight_res = _certify_step_lower_bound(final_rows, demands_ns)
    p0 = peaks_by_stage.get(0) or _die_peaks_for_stage(eff_case, 0, 0)
    cube_mfu, vec_mfu, hbm_mbu, link_util, net_roof, exposed_us, b_mix = _compute_network_efficiency_summary(
        final_rows, step_span_us, p0,
    )
    gap_radar = tuple(
        GapRadarRow(
            rank=final_rows[idx].gap_radar_rank,
            layer_label=final_rows[idx].archetype_title or final_rows[idx].layer_label,
            submodule=final_rows[idx].submodule,
            op_id=final_rows[idx].op_id,
            family=final_rows[idx].family,
            implementation=final_rows[idx].ascend_kernel or final_rows[idx].implementation,
            dominant_bound=final_rows[idx].bound_display or final_rows[idx].dominant_bound,
            l_star_lean_us=final_rows[idx].l_star_lean_us * final_rows[idx].repeat_count,
            pred_us=final_rows[idx].exposed_dur_us * final_rows[idx].repeat_count,
            gap_us=final_rows[idx].gap_us * final_rows[idx].repeat_count,
            gap_ratio=final_rows[idx].gap_ratio,
            share_of_step_pct=(
                100.0 * final_rows[idx].exposed_dur_us * final_rows[idx].repeat_count / step_span_us
                if step_span_us > 0.0 else 0.0
            ),
            repeat_count=final_rows[idx].repeat_count,
            logical_role=final_rows[idx].logical_role,
            kernel_status=final_rows[idx].kernel_status,
        )
        for idx in rank_order[: max(1, int(top_k_radar))]
    )
    step_s = max(step_span_us / US_PER_SECOND, MIN_POSITIVE_SECONDS)
    step_tokens = max(1, int(step_req.tokens) * seq_count * int(parsed.runtime.parallel.dp))
    tps = float(step_tokens) / step_s
    num_dies = max(1, len(parsed.machine.dies))
    default_dt = str(p0.get("default_dtype") or "bf16")
    hw_peaks_summary = {
        "cube_peak_flops_per_s": _die_cube_peak_rate(p0["die"], default_dt, default_dt),
        "vector_peak_ops_per_s": float(p0["vector_peak_ops_per_s"]),
        "hbm_peak_bytes_per_s": float(p0["hbm_peak_bytes_per_s"]),
        "mte2_peak_bytes_per_s": float(p0.get("mte2_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "mte3_peak_bytes_per_s": float(p0.get("mte3_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "fixpipe_peak_bytes_per_s": float(p0.get("fixpipe_peak_bytes_per_s") or p0["hbm_peak_bytes_per_s"]),
        "link_peak_bytes_per_s": float(p0["link_peak_bytes_per_s"]),
    }
    mem_ledger = compute_memory_quant_ledger(
        parsed, phase=step_req.phase,
        seq_len=max(step_req.context + step_req.tokens, step_req.tokens),
        concurrency=seq_count, weight_quant=weight_quant,
        cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
    )
    d0 = parsed.machine.dies[0]
    d_bufs = dict(d0.buffers)
    aiv_cnt = int(d0.owner_count("vector") or 0)
    die_summary = (
        f"AIC={d0.aic_count}, AIV={aiv_cnt}, "
        f"L2={int(d_bufs.get('l2', 0)) // MIB_BYTES} MiB, "
        f"UB={int(d_bufs.get('ub', 0)) // KIB_BYTES} KiB, "
        f"HBM={hw_peaks_summary['hbm_peak_bytes_per_s'] / TB_BYTES:.2f} TB/s"
    )
    track_name = str(dict(parsed.runtime.metadata).get("feature_track_report", {}).get("feature_track", "") or "")
    active_ids = {r.archetype_id for r in final_rows}
    filtered_schedule = tuple(e for e in full_schedule if e.archetype_id in active_ids) if use_archetype_dedup else ()
    return ExportBundle(
        schema=EXPORT_BUNDLE_SCHEMA,
        model_name=str(parsed.model.name),
        hardware_sku=str(parsed.machine.sku or parsed.machine.name),
        phase=step_req.phase, scope=step_req.scope, graph_id=graph_id,
        tokens_per_seq=int(step_req.tokens), context_tokens=int(step_req.context), sequences=seq_count,
        step_span_us=step_span_us, operator_sum_us=op_sum_us,
        step_l_star_lean_us=step_l_star_us, step_certificate_valid=cert_valid,
        step_tight_resources=tight_res,
        throughput_tokens_per_s=tps, throughput_per_die_tokens_per_s=tps / float(num_dies),
        network_cube_mfu=cube_mfu, network_vec_mfu=vec_mfu,
        network_mbu=hbm_mbu, network_link_util=link_util,
        network_roofline_efficiency=net_roof,
        exposed_breakdown_us=exposed_us, bottleneck_mix=b_mix,
        hardware_peaks=hw_peaks_summary,
        operators=final_rows, pipelines=pipelines,
        gap_radar=gap_radar, memory_ledger=mem_ledger, reconciliation=reconcile_summary,
        archetypes=tuple(a for a in archetypes if a.archetype_id in active_ids),
        full_trace_schedule=filtered_schedule,
        total_target_layers=total_target_layers,
        total_draft_layers=total_draft_layers,
        graph_capture_mode=str(parsed.runtime.graph_capture_mode or "full"),
        feature_track=track_name,
        hardware_die_summary=die_summary,
    )


def _fusion_absorbed_keys(after_op: OperatorExportRow) -> set[str]:
    keys = {_strip_op_prefix(x).lower() for x in after_op.absorbed_source_ops if x}
    canon = (after_op.canonical_op_key or _strip_op_prefix(after_op.op_id)).lower()
    keys.update(_KNOWN_FUSION_SOURCES.get(canon, ()))
    kern = (after_op.ascend_kernel or "").lower()
    if "mlaprolog" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["mla_prolog_v3"])
    elif "megamoe" in kern or "dispatch_ffn_combine" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["megamoe_arch35"])
    elif "situquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_situ_quant"])
    elif "swigluquant" in kern:
        keys.update(_KNOWN_FUSION_SOURCES["grouped_matmul_swiglu_quant"])
    return keys


def _build_action_note(
    relation: str,
    before_ops: Sequence[OperatorExportRow],
    after_ops: Sequence[OperatorExportRow],
) -> str:
    if relation == "N->1" and after_ops:
        a = after_ops[0]
        return f"N->1 Fused ({len(before_ops)} ops -> {a.ascend_kernel}); {a.kernel_status}"
    if relation == "eliminated":
        return "Eliminated by After graph/fusion pass (0 exposed ms)"
    if relation == "added" and after_ops:
        a = after_ops[0]
        return f"New op in After ({a.logical_role}); {a.kernel_status}"
    if after_ops:
        a = after_ops[0]
        if a.is_overlapped:
            return "Overlapped on MSO Stream 1 (0 exposed ms)"
        if a.needs_new_kernel and a.new_kernel_spec:
            return f"{a.kernel_status}: {a.new_kernel_spec}"
        if before_ops:
            b = before_ops[0]
            if b.ascend_kernel != a.ascend_kernel:
                return f"Kernel upgrade: {b.ascend_kernel} -> {a.ascend_kernel}"
            if a.gap_us > 1.0 and a.l_star_lean_us > 0.0:
                return f"Gap vs L* ({a.l_star_lean_us:.1f}us vs {a.dur_us:.1f}us): {a.bound_display}"
        return f"{a.bound_display} | {a.kernel_status}"
    return ""


def align_before_after_operators(
    before_bundle: ExportBundle,
    after_bundle: ExportBundle,
) -> tuple[AlignedComparisonRow, ...]:
    """Align Before vs After operators per ``(Archetype, Submodule)`` supporting N->1 / 1->N."""
    arch_order: list[tuple[str, str, int]] = []
    seen_arch: set[str] = set()
    for src in (before_bundle, after_bundle):
        for op in src.operators:
            aid = op.archetype_id or op.layer_label
            if aid not in seen_arch:
                seen_arch.add(aid)
                arch_order.append((aid, op.archetype_title or op.layer_label, op.repeat_count))

    before_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    after_by_arch: dict[str, list[OperatorExportRow]] = defaultdict(list)
    for op in before_bundle.operators:
        before_by_arch[op.archetype_id or op.layer_label].append(op)
    for op in after_bundle.operators:
        after_by_arch[op.archetype_id or op.layer_label].append(op)

    aligned: list[AlignedComparisonRow] = []
    for aid, title, rep in arch_order:
        b_list = before_by_arch.get(aid, [])
        a_list = after_by_arch.get(aid, [])
        sub_order: list[str] = []
        for op in (*b_list, *a_list):
            if op.submodule not in sub_order:
                sub_order.append(op.submodule)
        for sub in sub_order:
            b_sub = [op for op in b_list if op.submodule == sub]
            a_sub = [op for op in a_list if op.submodule == sub]
            aligned.extend(_align_submodule_ops(aid, title, rep, sub, b_sub, a_sub))
    return tuple(aligned)


def _align_submodule_ops(
    aid: str,
    title: str,
    rep: int,
    sub: str,
    b_sub: Sequence[OperatorExportRow],
    a_sub: Sequence[OperatorExportRow],
) -> list[AlignedComparisonRow]:
    used_b: set[int] = set()
    used_a: set[int] = set()
    entries: list[tuple[int, AlignedComparisonRow]] = []

    b_by_key: dict[str, list[int]] = defaultdict(list)
    for idx, op in enumerate(b_sub):
        b_by_key[(op.canonical_op_key or _strip_op_prefix(op.op_id)).lower()].append(idx)

    # Pass 1: Detect N->1 fusions where one After op absorbs multiple Before ops.
    for a_idx, a_op in enumerate(a_sub):
        a_key = (a_op.canonical_op_key or _strip_op_prefix(a_op.op_id)).lower()
        absorbed = _fusion_absorbed_keys(a_op)
        matched_b: list[int] = []
        for b_idx, b_op in enumerate(b_sub):
            if b_idx in used_b:
                continue
            b_key = (b_op.canonical_op_key or _strip_op_prefix(b_op.op_id)).lower()
            if b_key in absorbed and b_key != a_key:
                matched_b.append(b_idx)
        if len(matched_b) > 1 or (len(matched_b) == 1 and a_key not in b_by_key):
            if a_key in b_by_key:
                for b_idx in b_by_key[a_key]:
                    if b_idx not in used_b and b_idx not in matched_b:
                        matched_b.append(b_idx)
            matched_b.sort()
            used_b.update(matched_b)
            used_a.add(a_idx)
            b_tuple = tuple(b_sub[i] for i in matched_b)
            rel = "N->1" if len(b_tuple) > 1 else "1->1"
            entries.append((matched_b[0], AlignedComparisonRow(
                aid, title, rep, sub, b_tuple, (a_op,), rel, _build_action_note(rel, b_tuple, (a_op,)),
            )))

    # Pass 2: Exact canonical key 1->1 matches.
    for a_idx, a_op in enumerate(a_sub):
        if a_idx in used_a:
            continue
        a_key = (a_op.canonical_op_key or _strip_op_prefix(a_op.op_id)).lower()
        cand_idx = next((i for i in b_by_key.get(a_key, ()) if i not in used_b), None)
        if cand_idx is not None:
            used_b.add(cand_idx)
            used_a.add(a_idx)
            b_op = b_sub[cand_idx]
            entries.append((cand_idx, AlignedComparisonRow(
                aid, title, rep, sub, (b_op,), (a_op,), "1->1", _build_action_note("1->1", (b_op,), (a_op,)),
            )))

    # Pass 3: Remaining Before-only (eliminated) and After-only (added) rows.
    for b_idx, b_op in enumerate(b_sub):
        if b_idx not in used_b:
            entries.append((b_idx, AlignedComparisonRow(
                aid, title, rep, sub, (b_op,), (), "eliminated", _build_action_note("eliminated", (b_op,), ()),
            )))
    for a_idx, a_op in enumerate(a_sub):
        if a_idx not in used_a:
            entries.append((len(b_sub) + a_idx, AlignedComparisonRow(
                aid, title, rep, sub, (), (a_op,), "added", _build_action_note("added", (), (a_op,)),
            )))
    entries.sort(key=lambda item: item[0])
    return [row for _, row in entries]


def build_new_kernel_backlog(
    aligned_rows: Sequence[AlignedComparisonRow],
    after_bundle: ExportBundle,
) -> tuple[NewKernelBacklogItem, ...]:
    """Aggregate all ``[需新开发算子]`` / ``[需扩展算子 Tiling]`` items into Sheet 4 backlog."""
    grouped: dict[str, dict[str, Any]] = {}
    for row in aligned_rows:
        for a_op in row.after_ops:
            if not a_op.needs_new_kernel:
                continue
            key = a_op.ascend_kernel
            before_ms = sum(b.exposed_dur_us * row.repeat_count / US_PER_MS for b in row.before_ops)
            after_ms = 0.0 if a_op.is_overlapped else (a_op.exposed_dur_us * row.repeat_count / US_PER_MS)
            gain_ms = max(0.0, before_ms - after_ms)
            if gain_ms <= 0.0 and a_op.gap_us > 0.0:
                gain_ms = a_op.gap_us * row.repeat_count / US_PER_MS
            entry = grouped.get(key)
            if entry is None:
                entry = {
                    "kernel_key": key,
                    "target_kernel": a_op.ascend_kernel,
                    "backend": a_op.kernel_backend,
                    "status": a_op.kernel_status,
                    "replaces": [],
                    "sources": [],
                    "archetypes": [],
                    "calls": 0,
                    "gain_ms": 0.0,
                    "contract": a_op.tensor_contract,
                    "microarch": (
                        f"After Bound: {a_op.bound_display} | L*={a_op.l_star_lean_us:.2f}us "
                        f"vs Sim={a_op.dur_us:.2f}us | {after_bundle.hardware_die_summary}"
                    ),
                    "spec_note": a_op.new_kernel_spec,
                }
                grouped[key] = entry
            for b_op in row.before_ops:
                entry["replaces"].append(b_op.ascend_kernel)
                if b_op.source_ref:
                    entry["sources"].append(b_op.source_ref)
            if not row.before_ops and a_op.source_ref:
                entry["sources"].append(a_op.source_ref)
            short_arch = re.sub(r"\s*\(×\d+[^)]*\)\s*$", "", row.archetype_title.split("—")[0].strip())
            entry["archetypes"].append(f"{short_arch} (×{row.repeat_count})")
            entry["calls"] += row.repeat_count
            entry["gain_ms"] += gain_ms

    ordered = sorted(grouped.values(), key=lambda x: (-x["gain_ms"], -x["calls"], x["target_kernel"]))
    items: list[NewKernelBacklogItem] = []
    for prio, entry in enumerate(ordered, start=1):
        replaces_txt = " + ".join(dict.fromkeys(entry["replaces"])) or "New target kernel"
        sources_txt = "; ".join(dict.fromkeys(entry["sources"]))[:MAX_SOURCE_REF_SUMMARY_LEN]
        arch_txt = "; ".join(dict.fromkeys(entry["archetypes"]))
        items.append(NewKernelBacklogItem(
            priority=prio,
            kernel_key=entry["kernel_key"],
            target_kernel=f"{entry['target_kernel']} [{entry['backend']}]",
            suggested_backend=entry["backend"],
            status=entry["status"],
            replaces_chain=replaces_txt,
            source_refs=sources_txt,
            archetype_summary=arch_txt,
            total_calls_per_step=int(entry["calls"]),
            expected_gain_ms=float(entry["gain_ms"]),
            tensor_contract=entry["contract"],
            microarch_target=entry["microarch"],
            default_design_slot=f"[待 Agent 补充设计方案] {entry['spec_note'] or 'Tiling / L1-UB Ping-Pong / AIC-AIV 配比'}",
            default_acceptance_slot="[待 Agent 补充验收标准] 精度误差 <= 1e-3 (BF16/MXFP4) & L* 达成率目标",
        ))
    return tuple(items)


__all__ = [
    "AlignedComparisonRow",
    "AscendKernelDescriptor",
    "BACKEND_CANN",
    "BACKEND_CANNBOT_DSL",
    "BACKEND_HCCL_MC2",
    "BACKEND_TORCH_NPU",
    "BACKEND_TRITON",
    "BANNED_IMPLEMENTATION_NAMES",
    "CacheLedgerRow",
    "DEFAULT_GAP_RADAR_TOP_K",
    "EXPORT_BUNDLE_SCHEMA",
    "ExportBundle",
    "GIB_BYTES",
    "GapRadarRow",
    "HARDWARE_PRESET_ALIASES",
    "KERNEL_STATUS_EXISTING",
    "KERNEL_STATUS_EXTEND_TILING",
    "KERNEL_STATUS_FRAMEWORK",
    "KERNEL_STATUS_NEW_OP",
    "KERNEL_STATUS_NEW_OR_EXTEND",
    "KernelObservationRow",
    "LayerArchetypeSpec",
    "MIB_BYTES",
    "MXFP_GROUP_SIZE",
    "MemoryCalculatorLedger",
    "NewKernelBacklogItem",
    "OperatorExportRow",
    "RECONCILE_ABS_FLOOR_US",
    "RECONCILE_APE_THRESHOLD_PCT",
    "ReconciledKernelGroup",
    "ReconciliationSummary",
    "SUPPORTED_CACHE_DTYPES",
    "SUPPORTED_WEIGHT_QUANTS",
    "StageBudgetRow",
    "TraceScheduleEntry",
    "US_PER_MS",
    "WEIGHT_QUANT_FORMATS",
    "WeightLedgerRow",
    "align_before_after_operators",
    "build_export_bundle",
    "build_new_kernel_backlog",
    "classify_submodule",
    "compose_export_case",
    "compute_memory_quant_ledger",
    "evaluate_weight_storage_bytes",
    "extract_layer_index",
    "load_kernel_details_csv",
    "load_observation_bundle",
    "reconcile_operators_with_kernels",
    "resolve_hardware_alias",
]
