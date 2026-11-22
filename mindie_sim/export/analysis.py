"""Authoritative simulation data extraction, Lean L* gap radar, and memory ledger.

[Role] Build a single presentation-agnostic ``ExportBundle`` from ``mindie_sim``
  ``Case`` and ``PredictionSpine`` / ``analyze_network_efficiency`` / ``Memory``
  results, compute checked Lean-4 physical lower bounds (``L*``) and Gap Radar
  rankings, reconcile 1-to-N fused ``sim`` operators against ``msprof``
  ``kernel_details.csv`` observations, and derive the closed-form per-die
  Memory/Quant/PP calculator ledger.
[Invariants] Zero Excel/OOXML or Chrome Trace serialization lives here. Zero
  ``0.80x`` heuristic fudge factors or silent hardware defaults are introduced:
  every peak rate and HBM capacity comes strictly from the parsed ``Case``.
  Whole-step ``L*`` is verified via ``check_optimality_certificate``. Every
  measured kernel in a ``kernel_details.csv`` observation is assigned at most
  once, preserving exact sample conservation.
[Timeouts & Locks] Streaming CSV reads with bounded line parsing; finite in-memory
  case analysis honoring optional ``ReplayOptions.deadline_mono``; no locks or
  device contexts.
[Side-Effect Policy] Read-only over caller cases and observation files; never
  mutates inputs or writes files.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from fractions import Fraction
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
from mindie_sim.catalog.weights import (
    engram_table_terms,
    stage_weight_terms,
)
from mindie_sim.optimization.dual_bound import (
    NS_PER_SECOND,
    DagEdge,
    EventDagSpec,
    ResourceChoiceProgram,
    certify_lower_bound,
    check_optimality_certificate,
    inward_floor_ns,
)
from mindie_sim.prediction.capacity import Memory
from mindie_sim.prediction.efficiency import (
    analyze_network_efficiency,
    extract_hardware_peaks,
)
from mindie_sim.prediction.graph_analysis import graph_costs
from mindie_sim.prediction.operator_analysis import operator_costs
from mindie_sim.prediction.results import (
    CaseInput,
    FusedPipelineDescriptor,
    OpOccurrence,
    ReplayOptions,
    StepEvaluation,
    StepRequest,
)
from mindie_sim.prediction.spine import default_spine
from mindie_sim.prediction.structure import parse_case
from mindie_sim.schema.case import Case
from mindie_sim.schema.common import ModelingError
from mindie_sim.schema.datapath import datapath_for_die
from mindie_sim.schema.operators import OpSpec
from mindie_sim.stack_registry import LoweringRequest

#: Export bundle schema version.
EXPORT_BUNDLE_SCHEMA = "mindie-sim.export-bundle/1"
#: Binary GiB divisor in bytes.
GIB_BYTES = 1024 * 1024 * 1024
#: Binary MiB divisor in bytes.
MIB_BYTES = 1024 * 1024
#: Maximum read size for structured observation JSON manifests.
MAX_OBSERVATION_JSON_BYTES = 50 * MIB_BYTES
#: Microseconds per millisecond.
US_PER_MS = 1000.0
#: Nanoseconds per microsecond.
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
CONV_INPUT_PROJECTIONS = 3
#: Resource tuple slot count for single-op / whole-step Lean L* certificates.
CERT_RESOURCE_COUNT = 5

#: Closed-form storage specification matching se-tools@c356948 post-TP local shapes.
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
_COLLECTIVE_FAMILIES = frozenset({
    "allreduce", "allgather", "reducescatter", "alltoall", "alltoallv",
    "mc2", "mc2_transport", "mc2_protocol", "dispatch", "combine",
    "collective", "hccl", "p2p",
})
_MLA_PROLOG_TOKENS = (
    "mla_prolog", "mla_absorb", "kv_compress", "q_a", "q_b", "kv_a", "kv_b",
    "q_lora", "kv_lora", "mlapo",
)
_LINEAR_ATTN_TOKENS = (
    "gdn", "kda", "linear_attention", "linear_qkv", "linear_beta", "causal_conv",
    "conv_state", "convolution", "delta_gates", "recurrent", "ssm", "gate_beta",
)
_MOE_TOKENS = (
    "moe", "grouped_gemm", "expert", "router", "topk", "routing", "route_pack",
    "shared_expert", "combine", "dispatch",
)
_ATTENTION_TOKENS = (
    "attention", "flash_attention", "paged_mla", "latent_attention", "qkv",
    "rotary", "rope", "kv_store", "cache_store", "indexer", "sparse_indexer",
    "o_proj", "output_projection", "keypool",
)
_DENSE_FFN_TOKENS = (
    "gate_up", "down_projection", "dense_mlp", "swiglu", "mlp", "ffn",
)
_NORM_QUANT_TOKENS = (
    "rmsnorm", "layernorm", "norm", "quant", "mhc", "hc_", "residual", "cast",
    "tensor_copy", "elementwise", "contiguous",
)
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
    "reducescatter": ("reducescatter", "hccl", "hcom"),
    "alltoall": ("alltoall", "dispatch", "combine", "hccl", "hcom"),
    "rmsnorm": ("rmsnorm", "rms_norm", "norm"),
    "rotary": ("rotary", "rope"),
    "swiglu": ("swiglu", "silu", "mul"),
    "quantize": ("quant", "dynamicquant"),
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
class OperatorExportRow:
    """Complete physical, timing, Lean ``L*`` floor, and optional measured row per op."""

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


@dataclass(frozen=True, slots=True)
class GapRadarRow:
    """Ranked Physical-Limit Gap Radar entry ordered by ``T_pred - L*_Lean``."""

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
    """Single source of truth consumed by both ``export-trace`` and ``export-xlsx``."""

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
    if quant is None:
        return fallback
    scheme = str(getattr(quant, "scheme", "") or "").strip().lower()
    w_dtype = str(getattr(quant, "weight_dtype", "") or "").strip().lower()
    if "mxfp4" in scheme or "mxfp4" in w_dtype:
        return "MXFP4"
    if "mxfp8" in scheme or "mxfp8" in w_dtype:
        return "MXFP8"
    if "w4" in scheme or "int4" in w_dtype or "fp4" in w_dtype:
        return "W4A8"
    if "w8" in scheme or "int8" in w_dtype:
        return "W8A8"
    if "fp8" in w_dtype:
        return "MXFP8"
    return "BF16"


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
    return int(match.group(1)) if match is not None else None


def classify_submodule(op_id: str, family: str, recipe: str = "") -> str:
    """Map an operator occurrence onto its canonical hierarchical submodule bucket."""
    fam = family.strip().lower()
    text = f"{op_id} {family} {recipe}".lower()
    if fam in _COLLECTIVE_FAMILIES or any(
        tok in text for tok in ("allreduce", "allgather", "reducescatter", "alltoall", "mc2", "hccl")
    ):
        return "Collective"
    if any(tok in text for tok in _MLA_PROLOG_TOKENS):
        return "MLAProlog"
    if any(tok in text for tok in _LINEAR_ATTN_TOKENS):
        return "LinearAttn"
    if any(tok in text for tok in _MOE_TOKENS):
        return "MoE"
    if any(tok in text for tok in _ATTENTION_TOKENS):
        return "Attention"
    if any(tok in text for tok in _DENSE_FFN_TOKENS):
        return "DenseFFN"
    if any(tok in text for tok in _NORM_QUANT_TOKENS):
        return "NormQuant"
    return "Other"


def _classify_stream_index(stream_name: str, submodule: str, op_id: str) -> int:
    lower = stream_name.strip().lower()
    if lower in {"comm", "hccl", "csa", "stream2", "2"} or submodule == "Collective":
        return 2
    if lower in {"aux", "mso", "shared", "stream1", "1"} or "shared_expert" in op_id.lower():
        return 1
    return 0


def _classify_task_type(
    submodule: str, flops: float, vector_ops: float, link_bytes: float,
) -> str:
    if submodule == "Collective" or link_bytes > 0.0:
        return "HCCL"
    if flops > 0.0 and vector_ops > 0.0:
        return "MIX_AIC"
    if flops > 0.0:
        return "AI_CORE"
    return "AI_VECTOR_CORE"


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
            raise ModelingError(
                f"kernel_csv_missing_required_columns:{target.name}"
            )
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
    raw_groups = payload.get("mappings") or payload.get("groups") or ()
    for group in raw_groups:
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
                        index=len(kernels),
                        name=kname,
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
    for source in (op.family, op.recipe, op.implementation, *op.absorbed_source_ops):
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
            chosen = next(
                (k for k in by_name.get(pattern, ()) if k.index not in consumed),
                None,
            )
            if chosen is None:
                chosen = next(
                    (k for k in kernels if k.index not in consumed and pattern.lower() in k.name.lower()),
                    None,
                )
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
    apes = sorted(
        g.ape_pct for g in groups.values() if g.ape_pct is not None and math.isfinite(g.ape_pct)
    )
    med_ape = float(median(apes)) if apes else None
    p90_ape = _nearest_rank_percentile(apes, P90_QUANTILE) if apes else None
    within_count = sum(1 for g in groups.values() if g.within_gate is True)
    within_pct = (100.0 * within_count / len(groups)) if groups else None
    matched_sim_sum = sum(
        op.dur_us for op in operators if op.op_id in groups
    )
    matched_meas_sum = sum(
        g.measured_sum_us for g in groups.values() if g.measured_sum_us is not None
    )
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
    """Nearest-rank quantile ``ceil(frac * n) - 1`` matching repo verification gates."""
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
) -> tuple[int, int, int, int, int]:
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
    return d_cube, d_vec, d_hbm_r, d_hbm_w, d_link


def _hbm_read_write_l2(
    spec: OpSpec | None,
    eff_row: Mapping[str, Any] | None,
) -> tuple[float, float, float, float, float, float, float]:
    meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
    mte2 = float(
        (eff_row.get("mte2_bytes") if eff_row else None)
        or (spec.mte2_bytes if spec is not None else 0.0)
        or 0.0
    )
    mte3 = float(
        (eff_row.get("mte3_bytes") if eff_row else None)
        or (spec.mte3_bytes if spec is not None else 0.0)
        or 0.0
    )
    fixpipe = float(meta.get("fixpipe_bytes", 0.0) or 0.0)
    spec_hbm = float(spec.hbm_bytes if spec is not None else 0.0)
    eff_hbm = float(eff_row.get("hbm_bytes", 0.0) or 0.0) if eff_row else 0.0
    hbm_total = eff_hbm if eff_hbm > 0.0 else spec_hbm
    raw_write = mte3 + fixpipe
    copy_total = mte2 + raw_write

    if "l2_hit_rate" in meta and meta["l2_hit_rate"] is not None:
        l2_hit = max(0.0, min(1.0, float(meta["l2_hit_rate"])))
    elif copy_total > 0.0 and spec_hbm > 0.0 and spec_hbm < copy_total:
        l2_hit = max(0.0, min(1.0, 1.0 - spec_hbm / copy_total))
    else:
        l2_hit = 0.0

    if copy_total > 0.0 and hbm_total > 0.0:
        scale = hbm_total / copy_total
        hbm_read = mte2 * scale
        hbm_write = raw_write * scale
    elif "read_bytes" in meta:
        hbm_read = float(meta["read_bytes"] or 0.0)
        hbm_write = max(0.0, hbm_total - hbm_read)
    else:
        hbm_read = hbm_total
        hbm_write = 0.0
    return hbm_read, hbm_write, hbm_total, mte2, mte3, fixpipe, l2_hit * 100.0


def _resolve_shard_axis_and_rule(term: WeightTerm) -> tuple[str, int]:
    shard = str(term.shard or "replicated")
    role = str(term.placement_role or "")
    if role == "attention_output" or shard in {"in_tp", "dense_down", "expert_down", "group_in"}:
        axis = 1
    else:
        axis = 0
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
    return target_w / max(base_w, 1e-9)


def _collect_weight_ledger_rows(
    case: Case,
    mem: Memory,
    stage_layers: Sequence[int],
    resolved_wq: str,
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
    unquantized_cats = {"norm", "embedding", "lm_head", "param_fp32", "engram_table"}
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
                quant = (
                    resolved_wq if term.category not in unquantized_cats
                    else ("FP32" if term.category == "param_fp32" else "BF16")
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
                    "quant": quant,
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
    resolved_wq = (
        _canonicalize_weight_quant_override(weight_quant)
        if weight_quant is not None else _normalize_quant_preset(case.model.quant)
    )
    if resolved_wq not in WEIGHT_QUANT_FORMATS:
        raise ModelingError(f"unsupported_weight_quant:{resolved_wq}")
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
        case, mem, stage_layers, resolved_wq,
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
        sim_memory_weight_bytes_by_stage=tuple(
            float(x) for x in alloc["weights"]["total_bytes_per_stage_rank"]
        ),
        sim_memory_cache_bytes_by_stage=sim_cache_by_stage,
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
    return "Norm / Other"


def _certify_step_lower_bound(
    op_rows: Sequence[OperatorExportRow],
    per_op_demands_ns: Sequence[tuple[int, int, int, int, int]],
) -> tuple[float, bool, tuple[str, ...]]:
    """Compute and verify the per-die ``(replica, stage, die, pipe)`` Lean 4 ``L*`` bound."""
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
    resource_labels: list[str] = []
    for replica, stage, rank in die_coords:
        for pipe in res_pipes:
            resource_labels.append(f"{pipe}@r{replica}.p{stage}.d{rank}")

    for row, demand in zip(op_rows, per_op_demands_ns):
        coord = (row.occurrence.replica, row.occurrence.stage, int(row.occurrence.rank or 0))
        base = coord_to_offset[coord]
        for idx in range(CERT_RESOURCE_COUNT):
            totals[base + idx] += demand[idx]

    node_quanta = tuple(max(d) for d in per_op_demands_ns)
    index_by_occ = {row.occurrence: idx for idx, row in enumerate(op_rows)}
    edges: list[DagEdge] = []
    for dst_idx, row in enumerate(op_rows):
        for pred in row.predecessors:
            src_idx = index_by_occ.get(pred)
            if src_idx is not None and src_idx < dst_idx:
                edges.append(DagEdge(src=src_idx, dst=dst_idx, wire_ns=0))
    prog = ResourceChoiceProgram(
        num_resources=num_resources,
        global_work=tuple(totals),
        site_modes=(),
    )
    dag = EventDagSpec(node_quantum_ns=node_quanta, edges=tuple(edges))
    cert, dual_report = certify_lower_bound(prog, dag)
    valid = check_optimality_certificate(prog, dag, cert)
    tight = tuple(
        resource_labels[int(tag[1:])]
        if tag.startswith("r") and tag[1:].isdigit() and int(tag[1:]) < len(resource_labels)
        else tag
        for tag in dual_report.tight_resources
    )
    return float(cert.l_star_lean_ns) / NS_PER_US, valid, tight


def _collect_operator_export_rows(
    case: Case,
    resolved_step: StepEvaluation,
    op_spec_by_key: Mapping[tuple[int, str], OpSpec],
    eff_by_op_id: Mapping[str, Mapping[str, Any]],
) -> tuple[list[OperatorExportRow], list[tuple[int, int, int, int, int]], dict[int, dict[str, Any]]]:
    peaks_by_stage: dict[int, dict[str, Any]] = {}
    prelim_rows: list[OperatorExportRow] = []
    demands_ns: list[tuple[int, int, int, int, int]] = []
    for row in resolved_step.operators:
        occ = row.occurrence
        spec = op_spec_by_key.get((occ.replica, occ.op_id))
        eff_row = eff_by_op_id.get(occ.op_id)
        if occ.stage not in peaks_by_stage:
            peaks_by_stage[occ.stage] = _die_peaks_for_stage(case, occ.stage, occ.replica)
        peaks = peaks_by_stage[occ.stage]

        layer_idx = extract_layer_index(occ.op_id, spec.layer if spec is not None else None)
        layer_label = f"Layer {layer_idx}" if layer_idx is not None else "Head/Tail"
        recipe = str(spec.recipe if spec is not None else row.implementation)
        submodule = classify_submodule(occ.op_id, row.family, recipe)
        stream_idx = _classify_stream_index(row.stream, submodule, occ.op_id)

        start_us = max(0.0, float(row.start_ms or 0.0) * US_PER_MS)
        end_us = max(start_us, float(row.end_ms or row.start_ms or 0.0) * US_PER_MS)
        dur_us = max(0.0, end_us - start_us)
        flops = float((eff_row.get("flops") if eff_row else None) or (spec.flops if spec is not None else 0.0) or 0.0)
        vec_ops = float((eff_row.get("vector_ops") if eff_row else None) or (spec.vector_ops if spec is not None else 0.0) or 0.0)
        link_bytes = float((eff_row.get("link_bytes") if eff_row else None) or (spec.link_bytes if spec is not None else 0.0) or 0.0)
        hbm_r, hbm_w, hbm_tot, mte2, mte3, fixpipe, l2_hit = _hbm_read_write_l2(spec, eff_row)
        op_demand = _op_lean_resource_demands_ns(
            flops=flops, vector_ops=vec_ops, hbm_read=hbm_r, hbm_write=hbm_w,
            link_bytes=link_bytes, dtype=row.dtype, peaks=peaks,
        )
        demands_ns.append(op_demand)
        l_star_us = float(max(op_demand)) / NS_PER_US
        dom_bound = str(eff_row.get("dominant_bound", "launch_bound")) if eff_row else (
            "link_bound" if link_bytes > 0.0 else "memory_bound"
        )
        meta = dict(spec.metadata) if (spec is not None and spec.metadata) else {}
        absorbed = tuple(str(x) for x in (meta.get("absorbed_source_ops") or meta.get("fused_from") or ()))
        prelim_rows.append(OperatorExportRow(
            ordinal=row.ordinal, occurrence=occ, op_id=occ.op_id,
            layer=layer_idx, layer_label=layer_label, submodule=submodule,
            family=row.family, recipe=recipe, implementation=row.implementation,
            stream=row.stream, stream_index=stream_idx,
            task_type=_classify_task_type(submodule, flops, vec_ops, link_bytes),
            shape=row.shape, dtype=row.dtype or case.model.compute_dtype or "bf16",
            predecessors=row.predecessors,
            start_us=start_us, end_us=end_us, dur_us=dur_us,
            flops=flops, vector_ops=vec_ops,
            hbm_read_bytes=hbm_r, hbm_write_bytes=hbm_w, hbm_total_bytes=hbm_tot,
            mte2_bytes=mte2, mte3_bytes=mte3, fixpipe_bytes=fixpipe,
            link_bytes=link_bytes, l2_hit_pct=l2_hit, dominant_bound=dom_bound,
            l_star_lean_us=l_star_us, gap_us=dur_us - l_star_us,
            gap_ratio=(dur_us / l_star_us) if l_star_us > 0.0 else None,
            gap_radar_rank=0,
            roofline_efficiency=(
                float(eff_row["roofline_efficiency"])
                if eff_row and eff_row.get("roofline_efficiency") is not None else None
            ),
            absorbed_source_ops=absorbed,
        ))
    return prelim_rows, demands_ns, peaks_by_stage


def _finalize_ranked_rows(
    prelim_rows: Sequence[OperatorExportRow],
    observations_path: str | Path | None,
) -> tuple[tuple[OperatorExportRow, ...], list[int], ReconciliationSummary | None]:
    rank_order = sorted(
        range(len(prelim_rows)),
        key=lambda i: (-prelim_rows[i].gap_us, -(prelim_rows[i].gap_ratio or 0.0), prelim_rows[i].ordinal),
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
    final_rows: list[OperatorExportRow] = []
    for idx, item in enumerate(prelim_rows):
        rec = reconciled.get(item.op_id)
        final_rows.append(OperatorExportRow(
            ordinal=item.ordinal, occurrence=item.occurrence, op_id=item.op_id,
            layer=item.layer, layer_label=item.layer_label, submodule=item.submodule,
            family=item.family, recipe=item.recipe, implementation=item.implementation,
            stream=item.stream, stream_index=item.stream_index, task_type=item.task_type,
            shape=item.shape, dtype=item.dtype, predecessors=item.predecessors,
            start_us=item.start_us, end_us=item.end_us, dur_us=item.dur_us,
            flops=item.flops, vector_ops=item.vector_ops,
            hbm_read_bytes=item.hbm_read_bytes, hbm_write_bytes=item.hbm_write_bytes,
            hbm_total_bytes=item.hbm_total_bytes, mte2_bytes=item.mte2_bytes,
            mte3_bytes=item.mte3_bytes, fixpipe_bytes=item.fixpipe_bytes,
            link_bytes=item.link_bytes, l2_hit_pct=item.l2_hit_pct,
            dominant_bound=item.dominant_bound, l_star_lean_us=item.l_star_lean_us,
            gap_us=item.gap_us, gap_ratio=item.gap_ratio, gap_radar_rank=rank_by_idx[idx],
            roofline_efficiency=item.roofline_efficiency,
            absorbed_source_ops=item.absorbed_source_ops,
            measured_kernels=rec.kernels if rec is not None else (),
            measured_sum_us=rec.measured_sum_us if rec is not None else None,
            ape_pct=rec.ape_pct if rec is not None else None,
        ))
    return tuple(final_rows), rank_order, reconcile_summary


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
    """Evaluate one case on ``runtime.Engine`` and assemble the unified ``ExportBundle``."""
    parsed = case if isinstance(case, Case) else parse_case(dict(case))
    step_req = request or (
        step_eval.request if step_eval is not None else StepRequest(
            phase="decode",
            tokens=1,
            context=max(1, min(parsed.runtime.max_model_len, DEFAULT_EXPORT_CONTEXT_TOKENS)),
            sequences=max(1, len(parsed.workload.requests)),
            scope="body",
        )
    )
    seq_count = (
        int(step_req.sequences)
        if isinstance(step_req.sequences, int)
        else max(1, len(step_req.sequences))
    )
    spine = default_spine()
    prepared = spine.lower(parsed, LoweringRequest(step=step_req))
    resolved_step = step_eval or spine.run_step(prepared, replay)

    graphs = {entry.replica: entry.ops for entry in prepared.graph.executable}
    op_spec_by_key: dict[tuple[int, str], OpSpec] = {
        (entry.replica, op.op_id): op
        for entry in prepared.graph.executable
        for op in entry.ops
    }
    ext_inputs = (
        {"analysis_input": 0.0}
        if any("analysis_input" in op.deps for ops in graphs.values() for op in ops)
        else None
    )
    gr_report = graph_costs(
        prepared.case, graphs,
        external_inputs=ext_inputs,
        include_submission=replay.include_submission,
        isolated=False, compact=True,
    )
    op_report = operator_costs(prepared.case, graphs)
    eff = analyze_network_efficiency(
        prepared.case,
        phase=step_req.phase, tokens=step_req.tokens, context=step_req.context,
        sequences=seq_count, scope=step_req.scope,
        operator_report=op_report, graph_report=gr_report,
    )
    eff_by_op_id = {str(r.get("op_id", "")): r for r in eff.get("operators", ())}
    prelim_rows, demands_ns, peaks_by_stage = _collect_operator_export_rows(
        prepared.case, resolved_step, op_spec_by_key, eff_by_op_id,
    )
    final_rows, rank_order, reconcile_summary = _finalize_ranked_rows(
        prelim_rows, observations_path,
    )

    step_span_us = float(resolved_step.span_ms) * US_PER_MS
    step_l_star_us, cert_valid, tight_res = _certify_step_lower_bound(final_rows, demands_ns)
    gap_radar = tuple(
        GapRadarRow(
            rank=final_rows[idx].gap_radar_rank,
            layer_label=final_rows[idx].layer_label, submodule=final_rows[idx].submodule,
            op_id=final_rows[idx].op_id, family=final_rows[idx].family,
            implementation=final_rows[idx].implementation, dominant_bound=final_rows[idx].dominant_bound,
            l_star_lean_us=final_rows[idx].l_star_lean_us, pred_us=final_rows[idx].dur_us,
            gap_us=final_rows[idx].gap_us, gap_ratio=final_rows[idx].gap_ratio,
            share_of_step_pct=(100.0 * final_rows[idx].dur_us / step_span_us if step_span_us > 0.0 else 0.0),
        )
        for idx in rank_order[: max(1, int(top_k_radar))]
    )
    step_s = max(step_span_us / 1_000_000.0, 1e-12)
    step_tokens = max(1, int(step_req.tokens) * seq_count * int(prepared.case.runtime.parallel.dp))
    tps = float(step_tokens) / step_s
    num_dies = max(1, len(prepared.case.machine.dies))
    exposed_us = {str(k): float(v) * US_PER_MS for k, v in (gr_report.get("exposed_breakdown_ms") or {}).items()}
    bottleneck_mix = {
        str(bound_name): {
            "count": float(info.get("occurrences", 0)),
            "duration_us": float(info.get("duration_ms", 0.0)) * US_PER_MS,
            "share_pct": float(info.get("share_of_operator_sum", 0.0)) * 100.0,
            "roofline_eff_pct": float(info.get("roofline_efficiency", 0.0)) * 100.0,
        }
        for bound_name, info in (eff.get("breakdown_by_bound") or {}).items()
    }
    p0 = peaks_by_stage.get(0) or _die_peaks_for_stage(prepared.case, 0, 0)
    default_dt = str(p0.get("default_dtype") or "bf16")
    hw_peaks_summary = {
        "cube_peak_flops_per_s": _die_cube_peak_rate(p0["die"], default_dt, default_dt),
        "vector_peak_ops_per_s": float(p0["vector_peak_ops_per_s"]),
        "hbm_peak_bytes_per_s": float(p0["hbm_peak_bytes_per_s"]),
        "link_peak_bytes_per_s": float(p0["link_peak_bytes_per_s"]),
    }
    mem_ledger = compute_memory_quant_ledger(
        prepared.case, phase=step_req.phase,
        seq_len=max(step_req.context + step_req.tokens, step_req.tokens),
        concurrency=seq_count, weight_quant=weight_quant,
        cache_dtype=cache_dtype, engram_host_offload=engram_host_offload,
    )
    op_sum_by_replica: dict[int, float] = defaultdict(float)
    for row in final_rows:
        op_sum_by_replica[row.occurrence.replica] += row.dur_us
    per_replica_op_sum_us = max(op_sum_by_replica.values(), default=0.0)
    return ExportBundle(
        schema=EXPORT_BUNDLE_SCHEMA,
        model_name=str(prepared.case.model.name),
        hardware_sku=str(prepared.case.machine.sku or prepared.case.machine.name),
        phase=step_req.phase, scope=step_req.scope, graph_id=resolved_step.graph_id,
        tokens_per_seq=int(step_req.tokens), context_tokens=int(step_req.context), sequences=seq_count,
        step_span_us=step_span_us, operator_sum_us=per_replica_op_sum_us,
        step_l_star_lean_us=step_l_star_us, step_certificate_valid=cert_valid,
        step_tight_resources=tight_res,
        throughput_tokens_per_s=tps, throughput_per_die_tokens_per_s=tps / float(num_dies),
        network_cube_mfu=float(eff.get("network_cube_mfu", 0.0) or 0.0) * 100.0,
        network_vec_mfu=float(eff.get("network_vec_mfu", 0.0) or 0.0) * 100.0,
        network_mbu=float(eff.get("network_mbu", 0.0) or 0.0) * 100.0,
        network_link_util=float(eff.get("network_link_util", 0.0) or 0.0) * 100.0,
        network_roofline_efficiency=float(eff.get("network_roofline_efficiency", 0.0) or 0.0) * 100.0,
        exposed_breakdown_us=exposed_us, bottleneck_mix=bottleneck_mix,
        hardware_peaks=hw_peaks_summary,
        operators=final_rows, pipelines=resolved_step.pipelines,
        gap_radar=gap_radar, memory_ledger=mem_ledger, reconciliation=reconcile_summary,
    )


__all__ = [
    "CacheLedgerRow",
    "DEFAULT_GAP_RADAR_TOP_K",
    "EXPORT_BUNDLE_SCHEMA",
    "ExportBundle",
    "GIB_BYTES",
    "GapRadarRow",
    "KernelObservationRow",
    "MIB_BYTES",
    "MXFP_GROUP_SIZE",
    "MemoryCalculatorLedger",
    "OperatorExportRow",
    "RECONCILE_ABS_FLOOR_US",
    "RECONCILE_APE_THRESHOLD_PCT",
    "ReconciledKernelGroup",
    "ReconciliationSummary",
    "SUPPORTED_CACHE_DTYPES",
    "SUPPORTED_WEIGHT_QUANTS",
    "StageBudgetRow",
    "US_PER_MS",
    "WEIGHT_QUANT_FORMATS",
    "WeightLedgerRow",
    "build_export_bundle",
    "classify_submodule",
    "compute_memory_quant_ledger",
    "evaluate_weight_storage_bytes",
    "extract_layer_index",
    "load_kernel_details_csv",
    "load_observation_bundle",
    "reconcile_operators_with_kernels",
]
